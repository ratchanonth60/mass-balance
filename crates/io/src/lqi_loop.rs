//! Port of the in-class `MainConstantTs` (LQI run mode). Same one-`step()`
//! shape as `crate::mpc_loop`. `InitLQR(app)` is called at loop **entry**
//! (not on connect) to pick up live GUI tuning weights.

use crate::commands::{DEFAULT_ACC, DEFAULT_SPD, MassSolver};
use crate::mks_ops;
use crate::telemetry::Telemetry;
use crate::transport::Bus;
use control::{allocation, dare, kf, lqi_model};
use nalgebra::{DMatrix, DVector, Vector3};
use std::thread::sleep;
use std::time::Duration;

const MOTOR_PAUSE: Duration = Duration::from_millis(100);
const ENC_PAUSE: Duration = Duration::from_millis(50);
const D_MIN: f64 = 0.001;
const D_MAX: f64 = 0.495;
const D0_CENTER: f64 = 0.20; // "CoR reference plane" nominal center

/// GUI-tunable LQI weights (`InitLQR.m` lines 105-114 and 163-164).
///
/// `InitLQR.m` reads `app.qiEditField.Value` for a single shared integral
/// weight — but that component **does not exist** in AutoMass_MPC.mlapp (0
/// occurrences in document.xml; only `qi1EditField`/`qi2EditField` are ever
/// created). So MATLAB's LQI mode threw "Unrecognized property" at loop
/// entry and never ran with GUI weights at all. Ported as the two separate
/// weights the GUI actually has, which is what `ModelInit.m` already does
/// for the MPC path.
#[derive(Debug, Clone, Copy)]
pub struct LqiWeights {
    pub qx: [f64; 5],
    pub qi: [f64; 2],
    pub r: f64,
    pub du_max: f64,
    pub d_track_tol: f64,
}

impl Default for LqiWeights {
    fn default() -> Self {
        Self {
            qx: [100.0, 100.0, 50.0, 50.0, 25.0],
            qi: [1.0, 1.0],
            r: 50.0,
            du_max: 0.05,
            d_track_tol: 0.08,
        }
    }
}

impl From<crate::commands::TuneWeights> for LqiWeights {
    fn from(w: crate::commands::TuneWeights) -> Self {
        // The Tuning tab drives one shared set of fields for both run modes,
        // same as the MATLAB GUI. `qd` has no LQI counterpart (no d-state in
        // the 7-state augmented model) and is dropped.
        Self {
            qx: w.q,
            qi: w.qi,
            r: w.r,
            du_max: w.du_max,
            d_track_tol: w.d_track_tol,
        }
    }
}

/// `Qx`/`Qi`/`R` assembly + `dlqr`. `None` if the Riccati recursion doesn't
/// converge — reachable now that these come from UI DragValues (r = 0 makes
/// the problem singular), and this runs outside the IO thread's
/// `catch_unwind`, so it must not panic.
fn solve_gains(weights: &LqiWeights) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
    // The DARE needs R positive-definite and Q positive-semi-definite. The
    // recursion happily returns finite garbage for R = 0 or a negative
    // weight instead of failing, so reject those up front — the Tuning tab's
    // DragValues have no lower bound and drag straight through zero.
    if !(weights.r > 0.0 && weights.r.is_finite())
        || weights
            .qx
            .iter()
            .chain(weights.qi.iter())
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return None;
    }

    let (ad, bd) = lqi_model::build_augmented(1.0);
    let mut q_aug = DMatrix::<f64>::zeros(7, 7);
    q_aug.view_mut((0, 0), (5, 5)).copy_from(&DMatrix::from_diagonal(
        &DVector::from_row_slice(&weights.qx),
    ));
    q_aug[(5, 5)] = weights.qi[0];
    q_aug[(6, 6)] = weights.qi[1];
    // `InitLQR.m:114` builds `diag([r r r/2])`, but line 112 crashes first, so
    // that halved yaw penalty has never actually run. `r*I3` is what produced
    // the real `Ctrl_LQR.mat` gains the tests check against — keep it.
    let r = DMatrix::from_diagonal(&DVector::from_row_slice(&[weights.r; 3]));

    let k = dare::solve(&ad, &bd, &q_aug, &r, 1e-10, 2000).ok()?.k;
    let (kx, ki) = (k.columns(0, 5).into_owned(), k.columns(5, 2).into_owned());
    if kx.iter().chain(ki.iter()).all(|v| v.is_finite()) {
        Some((kx, ki))
    } else {
        None
    }
}

pub struct LqiLoop {
    kx: DMatrix<f64>,
    ki: DMatrix<f64>,
    x_est: DVector<f64>,
    p_kf: DMatrix<f64>,
    e_int: [f64; 2],
    d_running: [f64; 4], // the loop's own "d" accumulator (distinct from d_cmd/d_meas)
    d_cmd_prev: [f64; 4],
    d_meas_prev: [f64; 4],
    du_max: f64,
    d_track_tol: f64,
    cycle: u64,
    t: f64,

    pub controller_on: bool,
    pub setpoint_deg: [f64; 2],
    pub mass_solver: MassSolver,
    /// Same GUI spinners the MPC loop uses — see `mpc_loop::MpcLoop::spd`.
    pub spd: f64,
    pub acc: u8,
}

/// Same rationale as `mpc_loop::INIT_POLL_TIMEOUT` — not MATLAB parity,
/// bounds an otherwise-infinite encoder-priming poll that would block the
/// whole IO thread with no feedback if masses start too close to home.
const INIT_POLL_TIMEOUT: Duration = Duration::from_secs(60);

impl LqiLoop {
    /// `InitLQR(app)` + the encoder-priming sequence. `None` if the poll
    /// never reaches the 0.05m threshold within [`INIT_POLL_TIMEOUT`] —
    /// masses need to already be jogged above ~50mm before Start Auto.
    pub fn init(bus: &mut dyn Bus, weights: &LqiWeights) -> Option<Self> {
        let (kx, ki) = solve_gains(weights)?;

        let mut d0 = [0.0f64; 4];
        for addr in 1..=4u8 {
            mks_ops::run_rel(bus, addr, -0.0015, 50.0, 32);
            sleep(MOTOR_PAUSE);
        }
        let started = std::time::Instant::now();
        while d0.iter().any(|&d| d < 0.05) {
            if started.elapsed() > INIT_POLL_TIMEOUT {
                return None;
            }
            for (i, addr) in (1..=4u8).enumerate() {
                if let Some(v) = mks_ops::read_encoder(bus, addr) {
                    d0[i] = v;
                }
                sleep(Duration::from_millis(250));
            }
        }
        let d_init = saturate(d0);

        Some(Self {
            kx,
            ki,
            x_est: DVector::zeros(5),
            p_kf: DMatrix::identity(5, 5) * 1e4,
            e_int: [0.0, 0.0],
            d_running: [0.0; 4],
            d_cmd_prev: d_init,
            d_meas_prev: d_init,
            du_max: weights.du_max,
            d_track_tol: weights.d_track_tol,
            cycle: 0,
            t: 0.0,
            controller_on: true,
            setpoint_deg: [0.0, 0.0],
            mass_solver: MassSolver::Pinv,
            spd: DEFAULT_SPD,
            acc: DEFAULT_ACC,
        })
    }

    /// `Tune Now` for the LQI path: re-solve the gains and pick up the new
    /// rate limit / tracking gate without restarting the run (so the KF
    /// estimate, integral error and encoder priming all survive). `false`
    /// if the new weights don't produce usable gains — old ones are kept,
    /// the loop keeps running on them.
    pub fn retune(&mut self, weights: &LqiWeights) -> bool {
        match solve_gains(weights) {
            Some((kx, ki)) => {
                self.kx = kx;
                self.ki = ki;
                self.du_max = weights.du_max;
                self.d_track_tol = weights.d_track_tol;
                true
            }
            None => false,
        }
    }

    pub fn one_step(&mut self, bus: &mut dyn Bus, plant_t: f64) -> Telemetry {
        let mut d_meas = self.d_meas_prev;
        for (i, addr) in (1..=4u8).enumerate() {
            if let Some(v) = mks_ops::read_encoder(bus, addr)
                && v.is_finite() {
                    d_meas[i] = v;
                }
            sleep(ENC_PAUSE);
        }

        bus.flush_input();
        let imu = mks_ops::read_imu(bus);
        sleep(Duration::from_millis(50));

        let (r, p) = match imu {
            Some(f) if f.angle_deg[..2].iter().all(|v| v.is_finite()) => {
                (f.angle_deg[0].to_radians(), f.angle_deg[1].to_radians())
            }
            _ => (self.x_est[0], self.x_est[1]),
        };
        let (wx, wy, wz) = match imu {
            Some(f) if f.gyro_dps.iter().all(|v| v.is_finite()) => (
                f.gyro_dps[0].to_radians(),
                f.gyro_dps[1].to_radians(),
                f.gyro_dps[2].to_radians(),
            ),
            _ => (self.x_est[2], self.x_est[3], self.x_est[4]),
        };
        let y_meas = DVector::from_row_slice(&[r, p, wx, wy, wz]);

        // KF process/measurement noise hardcoded in the live `MainConstantTs`
        // loop itself (not GUI-tunable): q=[100,100,10000,10000,100], r=[1]*5.
        let ad_op = DMatrix::from_iterator(5, 5, lqi_model::ad_op().iter().copied());
        let qkf = DMatrix::from_diagonal(&DVector::from_row_slice(&[
            100.0, 100.0, 10000.0, 10000.0, 100.0,
        ]));
        let ckf = DMatrix::<f64>::identity(5, 5);
        let rkf = DMatrix::<f64>::identity(5, 5);
        let (x_pred, p_pred) = kf::predict(&self.x_est, &self.p_kf, &ad_op, &qkf);
        let (x_new, p_new) = kf::update(&x_pred, &p_pred, &y_meas, &ckf, &rkf);
        let prev = if self.cycle > 0 { Some(&self.x_est) } else { None };
        if !kf::is_diverged(&x_new, prev) {
            self.x_est = x_new;
            self.p_kf = p_new;
        }

        let ref_rad = [self.setpoint_deg[0].to_radians(), self.setpoint_deg[1].to_radians()];
        let e = [self.x_est[0] - ref_rad[0], self.x_est[1] - ref_rad[1]];
        self.e_int[0] += plant_t * e[0];
        self.e_int[1] += plant_t * e[1];

        let (d_target, exitflag) = if !self.controller_on {
            self.e_int = [self.e_int[0], self.e_int[1]]; // frozen (already not advanced further this branch in MATLAB; kept simple here)
            (self.d_cmd_prev, 0)
        } else {
            // u = -Kx*x_est - Ki*e_int (LQR torque law)
            let e_int_vec = DVector::from_row_slice(&self.e_int);
            let u = -&self.kx * &self.x_est - &self.ki * &e_int_vec;
            let tgt = Vector3::new(u[0] / 0.25, u[1] / 0.25, u[2] / 0.25);

            let gb = Vector3::new(0.0, 0.0, -9.81); // gravity in body frame at small-angle approx
            let a = allocation::build_a(gb);
            let mean_offset = d_meas.iter().sum::<f64>() / 4.0;
            let lb = nalgebra::Vector4::from_element(D_MIN - mean_offset);
            let ub = nalgebra::Vector4::from_element(D_MAX - mean_offset);

            let d_sol = match self.mass_solver {
                MassSolver::Pinv => allocation::allocate_pinv(&a, tgt),
                MassSolver::Qp => allocation::allocate_qp(&a, tgt, lb, ub).unwrap_or_else(|| allocation::allocate_pinv(&a, tgt)),
            };

            // du_max rate limit relative to the loop's own running `d` state.
            let mut du = [0.0; 4];
            for i in 0..4 {
                du[i] = (d_sol[i] - self.d_running[i]).clamp(-self.du_max, self.du_max);
                self.d_running[i] += du[i];
            }

            let d_err_inf = (0..4).map(|i| (d_meas[i] - self.d_cmd_prev[i]).abs()).fold(0.0, f64::max);
            if d_err_inf > self.d_track_tol {
                (self.d_cmd_prev, -99)
            } else {
                let mut target = [0.0; 4];
                for i in 0..4 {
                    target[i] = (self.d_running[i] + D0_CENTER).clamp(D_MIN, D_MAX);
                }
                (target, 1)
            }
        };

        self.d_cmd_prev = saturate(d_target);
        self.d_meas_prev = d_meas;

        if self.controller_on {
            let (spd, acc) = (self.spd, self.acc);
            mks_ops::run_abs(bus, 1, self.d_cmd_prev[0], spd, acc);
            sleep(MOTOR_PAUSE);
            mks_ops::run_rel(bus, 2, self.d_cmd_prev[1] - d_meas[1], spd, acc);
            sleep(MOTOR_PAUSE);
            mks_ops::run_abs(bus, 3, self.d_cmd_prev[2], spd, acc);
            sleep(MOTOR_PAUSE);
            mks_ops::run_abs(bus, 4, self.d_cmd_prev[3], spd, acc);
            sleep(MOTOR_PAUSE);
        }

        self.cycle += 1;
        self.t += plant_t;

        Telemetry {
            connected: true,
            is_sample: true,
            manual_read: None,
            t: self.t,
            roll_deg: self.x_est[0].to_degrees(),
            pitch_deg: self.x_est[1].to_degrees(),
            gyro_dps: [wx.to_degrees(), wy.to_degrees(), wz.to_degrees()],
            d_meas,
            d_cmd: self.d_cmd_prev,
            e_deg: [e[0].to_degrees(), e[1].to_degrees()],
            exitflag,
            status: "Running...".to_string(),
            cycle: self.cycle,
        }
    }
}

fn saturate(d: [f64; 4]) -> [f64; 4] {
    d.map(|v| v.clamp(D_MIN, D_MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockBus;
    use protocol::{checksum::sum8, mks};

    fn push_encoder_reply(bus: &mut MockBus, addr: u8, meters: f64) {
        let total_angle_deg = meters * 90000.0;
        let rotation_turns = (total_angle_deg / 360.0).floor().max(0.0);
        let degree = (total_angle_deg - rotation_turns * 360.0).clamp(1.0, 179.0);
        let ticks14 = (degree * 16383.0 / 180.0).round() as u16;
        let carry = u32::MAX - rotation_turns as u32;
        let mut reply = [0u8; 10];
        reply[0] = mks::RX_HEADER;
        reply[1] = addr;
        reply[2] = 0x30;
        reply[3..7].copy_from_slice(&carry.to_be_bytes());
        reply[7..9].copy_from_slice(&ticks14.to_be_bytes());
        reply[9] = sum8(&reply[..9]);
        bus.push_reply(reply.to_vec());
    }

    /// `LqiWeights::default()` was picked to match `fixtures/ctrl_lqr.json`'s
    /// real tuning (`Qx=[100,100,50,50,25]`, `Qi=[1,1]`, `R=50*I3`) — so this
    /// end-to-end check (through the same `LqiLoop::init` path the live loop
    /// uses, not just `control::dare` directly as in M2) should reproduce the
    /// real `InitLQR.m`-saved `Kx`/`Ki` to the same tolerance.
    #[test]
    fn init_reproduces_real_init_lqr_fixture_gains() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("../../fixtures/ctrl_lqr.json").unwrap(),
        )
        .unwrap();
        let to_mat = |v: &serde_json::Value| -> DMatrix<f64> {
            let rows = v.as_array().unwrap();
            let nrows = rows.len();
            let ncols = rows[0].as_array().unwrap().len();
            let data: Vec<f64> = rows
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
                .collect();
            DMatrix::from_row_slice(nrows, ncols, &data)
        };
        let kx_expected = to_mat(&fixture["Kx"]);
        let ki_expected = to_mat(&fixture["Ki"]);

        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let lqi = LqiLoop::init(&mut bus, &LqiWeights::default()).unwrap();

        for i in 0..3 {
            for j in 0..5 {
                assert!((lqi.kx[(i, j)] - kx_expected[(i, j)]).abs() < 1e-6);
            }
            for j in 0..2 {
                assert!((lqi.ki[(i, j)] - ki_expected[(i, j)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn init_solves_dare_and_primes_encoders() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let lqi = LqiLoop::init(&mut bus, &LqiWeights::default()).unwrap();
        // Kx should be a real, finite, non-degenerate gain (not NaN/all-zero).
        assert!(lqi.kx.iter().all(|v| v.is_finite()));
        assert!(lqi.kx.iter().any(|&v| v.abs() > 1e-6));
    }

    /// `Tune Now` on a running LQI loop has to actually move the gains, and
    /// degenerate weights coming straight off a UI DragValue must fail as a
    /// `false`/`None`, not a panic — the gain solve runs on the IO thread
    /// outside `thread::run`'s `catch_unwind`, so a panic there would kill
    /// the thread and freeze the UI for good.
    #[test]
    fn retune_updates_gains_and_rejects_degenerate_weights() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let mut lqi = LqiLoop::init(&mut bus, &LqiWeights::default()).unwrap();
        let kx_before = lqi.kx.clone();

        let mut w = LqiWeights::default();
        w.r = 500.0; // 10x the control penalty -> softer gains
        w.du_max = 0.02;
        assert!(lqi.retune(&w));
        assert!((&lqi.kx - &kx_before).amax() > 1e-6);
        assert_eq!(lqi.du_max, 0.02);

        // R = 0 makes the Riccati recursion singular.
        let mut bad = LqiWeights::default();
        bad.r = 0.0;
        let kx_now = lqi.kx.clone();
        assert!(!lqi.retune(&bad));
        assert_eq!(lqi.kx, kx_now, "failed retune must keep the old gains");
        assert!(LqiLoop::init(&mut bus, &bad).is_none());
    }

    #[test]
    fn controller_off_holds_command() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let mut lqi = LqiLoop::init(&mut bus, &LqiWeights::default()).unwrap();
        lqi.controller_on = false;
        let prev = lqi.d_cmd_prev;

        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let tel = lqi.one_step(&mut bus, 3.25);
        assert_eq!(tel.exitflag, 0);
        assert_eq!(tel.d_cmd, prev);
    }
}
