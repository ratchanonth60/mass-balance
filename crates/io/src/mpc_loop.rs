//! Port of `MainConstantTs11.m`. One `step()` call = one MATLAB `while true`
//! loop body (through the motor-command sends); the `plantT`-cadence sleep
//! and `Stop`-command check between iterations live in `crate::thread`, not
//! here, so `step()` stays testable with a `MockBus` and no wall-clock
//! dependency beyond the small inter-command `pause()`s that are genuine
//! bus-turnaround delays (matches the live code exactly).

use crate::commands::{DEFAULT_ACC, DEFAULT_SPD};
use crate::mks_ops;
use crate::telemetry::Telemetry;
use crate::transport::Bus;
use control::mpc_model::{self, CtrlMats, Preset, TuningWeights};
use control::{kf, mpc as mpc_solve};
use nalgebra::{DMatrix, DVector};
use std::thread::sleep;
use std::time::Duration;

const PLANT_T: f64 = 2.00; // hardcoded in MainConstantTs11.m, overrides ctrl.Ts
const MOTOR_PAUSE: Duration = Duration::from_millis(120);
const ENC_PAUSE: Duration = Duration::from_millis(50);
const IMU_PAUSE: Duration = Duration::from_millis(50);
const D_MIN: f64 = 0.001;
const D_MAX: f64 = 0.495;

/// `0` = controller off (no command sent), `-99` = actuator still tracking
/// previous command (solve skipped), `1` = solved, `-1` = infeasible/fallback.
pub type ExitFlag = i32;

pub struct MpcLoop {
    preset: Preset,
    d_op: [f64; 4],   // fixed at init (d0), used for every re-linearization
    d_init: [f64; 4], // saturated d0, keepZ's beq target for the whole run
    ctrl: CtrlMats,

    qkf: DMatrix<f64>,
    rkf: DMatrix<f64>,
    ckf: DMatrix<f64>,
    x_est: DVector<f64>,
    p_kf: DMatrix<f64>,

    d_meas_prev: [f64; 4],
    d_cmd_prev: [f64; 4],
    e_int: [f64; 2],
    t: f64,
    cycle: u64,

    pub controller_on: bool,
    pub setpoint_deg: [f64; 2],
    /// `app.SpeedSpinner.Value`/`app.AccSpinner.Value` — MATLAB's loop reads
    /// the *same* GUI spinners the manual-jog buttons use (MainConstantTs11.m
    /// lines 37-38) and passes them to every `mksRunAbsAxis`/`mksRunRelAxis`.
    /// Porting these as hardcoded 100/50 made every commanded move ~1.8x
    /// slower than MATLAB's default 180/64, so the axes couldn't reach
    /// `d_cmd` inside one 2s `plantT` — the `d_track_tol` gate then held the
    /// command (`exitflag = -99`) and skipped the solve, cycle after cycle.
    pub spd: f64,
    pub acc: u8,
}

/// Ceiling on `init`'s encoder-priming poll (MATLAB's original loop has no
/// bound at all — it just spins until every axis reads >= the threshold,
/// which never happens if masses start too close to home). Not MATLAB
/// parity, added because an unbounded poll here blocks the whole IO thread
/// with zero feedback: `Stop` can't even be processed, and no `Telemetry`
/// gets sent so the UI looks frozen with no explanation. `None` on timeout
/// lets the caller report a clear error and stay `Idle` instead of hanging.
const INIT_POLL_TIMEOUT: Duration = Duration::from_secs(60);

/// `init`'s encoder-priming poll won't return until every axis clears this —
/// exposed so the UI can warn *before* Start Auto blocks for up to
/// [`INIT_POLL_TIMEOUT`] and fails.
pub const INIT_MIN_D: f64 = 0.01;

impl MpcLoop {
    /// Replicates the init sequence: nudge all 4 axes, poll encoders until
    /// every reading is >= [`INIT_MIN_D`], then linearize at that `d0`.
    /// `None` if that never happens within [`INIT_POLL_TIMEOUT`] — masses
    /// need to already be jogged above ~10mm before Start Auto.
    pub fn init(bus: &mut dyn Bus, xy_pre_balance: bool, weights: &TuningWeights) -> Option<Self> {
        for addr in 1..=4u8 {
            mks_ops::run_rel(bus, addr, -0.001, 50.0, 128);
            sleep(MOTOR_PAUSE);
        }
        let mut d0 = [0.0f64; 4];
        let started = std::time::Instant::now();
        while d0.iter().any(|&d| d < INIT_MIN_D) {
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

        let preset = if xy_pre_balance {
            mpc_model::PRESET_POST_BALANCE
        } else {
            mpc_model::PRESET_MODEL_INIT
        };
        let ctrl = mpc_model::build_ctrl(preset, d0, Some(weights));

        let d_init = saturate(d0);

        Some(Self {
            preset,
            d_op: d0,
            d_init,
            qkf: DMatrix::from_diagonal(&DVector::from_row_slice(&[
                100.0, 100.0, 10000.0, 10000.0, 100.0,
            ])),
            rkf: DMatrix::from_diagonal(&DVector::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0])),
            ckf: DMatrix::identity(5, 5),
            x_est: DVector::zeros(5),
            p_kf: DMatrix::identity(5, 5) * 1e4,
            d_meas_prev: d_init,
            d_cmd_prev: d_init,
            e_int: [0.0, 0.0],
            t: 0.0,
            cycle: 0,
            ctrl,
            controller_on: true,
            setpoint_deg: [0.0, 0.0],
            spd: DEFAULT_SPD,
            acc: DEFAULT_ACC,
        })
    }

    /// `TuneNowPushed`: re-linearize/re-tune at the *original* `d0`, never
    /// the current measured position — matches `app.newTuning` handling.
    pub fn retune(&mut self, weights: &TuningWeights) {
        self.ctrl = mpc_model::build_ctrl(self.preset, self.d_op, Some(weights));
    }

    pub fn one_step(&mut self, bus: &mut dyn Bus) -> Telemetry {
        // --- read encoders ---
        let mut d_meas = self.d_meas_prev;
        for (i, addr) in (1..=4u8).enumerate() {
            if let Some(v) = mks_ops::read_encoder(bus, addr)
                && v.is_finite() {
                    d_meas[i] = v;
                }
            sleep(ENC_PAUSE);
        }
        d_meas = saturate(d_meas);

        // --- read IMU ---
        bus.flush_input();
        let imu = mks_ops::read_imu(bus);
        sleep(IMU_PAUSE);

        let (r, p) = match imu {
            Some(frame) if frame.angle_deg.iter().all(|v| v.is_finite()) => {
                (frame.angle_deg[0].to_radians(), frame.angle_deg[1].to_radians())
            }
            _ => (self.x_est[0], self.x_est[1]),
        };
        let (wx, wy, wz) = match imu {
            Some(frame) if frame.gyro_dps.iter().all(|v| v.is_finite()) => (
                frame.gyro_dps[0].to_radians(),
                frame.gyro_dps[1].to_radians(),
                frame.gyro_dps[2].to_radians(),
            ),
            _ => (self.x_est[2], self.x_est[3], self.x_est[4]),
        };
        let y_meas = DVector::from_row_slice(&[r, p, wx, wy, wz]);

        // --- KF with divergence guard ---
        let a5 = self.ctrl.ad.view((0, 0), (5, 5)).into_owned();
        let (x_pred, p_pred) = kf::predict(&self.x_est, &self.p_kf, &a5, &self.qkf);
        let (x_new, p_new) = kf::update(&x_pred, &p_pred, &y_meas, &self.ckf, &self.rkf);

        let prev_for_guard = if self.cycle > 0 {
            Some(&self.x_est)
        } else {
            None
        };
        if kf::is_diverged(&x_new, prev_for_guard) {
            // hold previous estimate; P_KF also held at its pre-update value
        } else {
            self.x_est = x_new;
            self.p_kf = p_new;
        }

        // --- reference + integral error ---
        let ref_rad = [
            self.setpoint_deg[0].to_radians(),
            self.setpoint_deg[1].to_radians(),
        ];
        let e = [self.x_est[0] - ref_rad[0], self.x_est[1] - ref_rad[1]];
        self.e_int[0] += PLANT_T * e[0];
        self.e_int[1] += PLANT_T * e[1];

        // --- MPC ---
        let (u_cmd, exitflag) = if !self.controller_on {
            (self.d_cmd_prev, 0)
        } else {
            let d_err_inf = (0..4)
                .map(|i| (d_meas[i] - self.d_cmd_prev[i]).abs())
                .fold(0.0, f64::max);
            if d_err_inf > self.ctrl.d_track_tol {
                (self.d_cmd_prev, -99)
            } else {
                self.solve_mpc(d_meas)
            }
        };

        self.d_cmd_prev = saturate(u_cmd);
        self.d_meas_prev = d_meas;

        // --- send commands (only when controller is on) ---
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
        self.t += PLANT_T;

        Telemetry {
            run_state: crate::telemetry::RunState::Mpc,
            connected: true,
            is_sample: true,
            manual_read: None,
            t: self.t,
            roll_deg: self.x_est[0].to_degrees(),
            pitch_deg: self.x_est[1].to_degrees(),
            gyro_dps: [
                self.x_est[2].to_degrees(),
                self.x_est[3].to_degrees(),
                self.x_est[4].to_degrees(),
            ],
            d_meas,
            d_cmd: self.d_cmd_prev,
            e_deg: [e[0].to_degrees(), e[1].to_degrees()],
            exitflag,
            status: "Running...".to_string(),
            cycle: self.cycle,
        }
    }

    fn solve_mpc(&self, d_meas: [f64; 4]) -> ([f64; 4], ExitFlag) {
        let x11 = DVector::from_row_slice(&[
            self.x_est[0],
            self.x_est[1],
            self.x_est[2],
            self.x_est[3],
            self.x_est[4],
            self.e_int[0],
            self.e_int[1],
            d_meas[0],
            d_meas[1],
            d_meas[2],
            d_meas[3],
        ]);
        let ref_ = DVector::from_row_slice(&[
            self.setpoint_deg[0].to_radians(),
            self.setpoint_deg[1].to_radians(),
        ]);
        let u_prev = DVector::from_row_slice(&self.d_cmd_prev);
        let n = self.ctrl.n_mpc;

        // Build the same condensed matrices the solver will (again)
        // internally rebuild, purely to construct the extra constraints —
        // matches MainConstantTs11.m's redundant double-build exactly.
        let mats = mpc_solve::build_batch_mats(
            &self.ctrl.ad,
            &self.ctrl.bd,
            &self.ctrl.gd,
            n,
            &x11,
            &ref_,
            &u_prev,
        );

        let du_max = self.ctrl.du_max;
        let mut a_ineq = DMatrix::<f64>::zeros(8 * n, 4 * n);
        a_ineq.view_mut((0, 0), (4 * n, 4 * n)).copy_from(&mats.d);
        a_ineq
            .view_mut((4 * n, 0), (4 * n, 4 * n))
            .copy_from(&(-&mats.d));
        let mut b_ineq = DVector::<f64>::zeros(8 * n);
        for i in 0..4 * n {
            b_ineq[i] = mats.d0[i] + du_max;
            b_ineq[4 * n + i] = -mats.d0[i] + du_max;
        }

        // keepZ: sum(d1..d4) constant (= sum(d_init)) at every horizon step.
        let mut cd = DMatrix::<f64>::zeros(4, 11);
        for i in 0..4 {
            cd[(i, 7 + i)] = 1.0;
        }
        let mut cdbar = DMatrix::<f64>::zeros(4 * n, 11 * n);
        for k in 0..n {
            cdbar.view_mut((k * 4, k * 11), (4, 11)).copy_from(&cd);
        }
        let mut zsel = DMatrix::<f64>::zeros(n, 4 * n);
        for k in 0..n {
            for j in 0..4 {
                zsel[(k, k * 4 + j)] = 1.0;
            }
        }
        let a_eq = &zsel * &cdbar * &mats.gamma;
        let sum_d_init: f64 = self.d_init.iter().sum();
        let b_eq = DVector::from_element(n, sum_d_init) - &zsel * &cdbar * &mats.x_free;

        let extra = mpc_solve::ExtraConstraints {
            a_ineq: Some(a_ineq),
            b_ineq: Some(b_ineq),
            a_eq: Some(a_eq),
            b_eq: Some(b_eq),
        };

        let dmin = DVector::from_element(4, D_MIN);
        let dmax = DVector::from_element(4, D_MAX);

        let result = mpc_solve::solve_qp(
            &self.ctrl.ad,
            &self.ctrl.bd,
            &self.ctrl.gd,
            &self.ctrl.q11,
            &self.ctrl.r,
            &self.ctrl.pf11,
            n,
            &x11,
            &ref_,
            &u_prev,
            &dmin,
            &dmax,
            &extra,
        );

        let u0: [f64; 4] = [result.u0[0], result.u0[1], result.u0[2], result.u0[3]];
        (u0, if result.feasible { 1 } else { -1 })
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

    /// Builds a reply decoding to *approximately* `meters` (within ~1mm —
    /// `ticks14` alone (carry=0) can only reach ~2mm, so most of the target
    /// has to come from the `rotation`/`carry` field; exact inversion isn't
    /// needed for these behavioral tests, just "close, and definitely above
    /// the 0.01m init-poll threshold").
    fn push_encoder_reply(bus: &mut MockBus, addr: u8, meters: f64) {
        let total_angle_deg = meters * 90000.0; // inverse of *(4/1000/360)
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

    fn push_imu_reply(bus: &mut MockBus, roll_deg: f64, pitch_deg: f64) {
        let mut reply = [0u8; 35];
        reply[0] = 0x50;
        let enc_angle = |deg: f64| {
            let raw = (deg * 1000.0) as i32 as u32;
            (
                (raw & 0xFF00) >> 8,
                raw & 0xFF,
                (raw >> 24) & 0xFF,
                (raw >> 16) & 0xFF,
            )
        };
        let (b0, b1, b2, b3) = enc_angle(roll_deg);
        reply[21] = b0 as u8;
        reply[22] = b1 as u8;
        reply[23] = b2 as u8;
        reply[24] = b3 as u8;
        let (b0, b1, b2, b3) = enc_angle(pitch_deg);
        reply[25] = b0 as u8;
        reply[26] = b1 as u8;
        reply[27] = b2 as u8;
        reply[28] = b3 as u8;
        bus.push_reply(reply.to_vec());
    }

    fn weights() -> TuningWeights {
        TuningWeights {
            q: [100.0, 100.0, 50.0, 50.0, 25.0],
            qi: [0.3, 0.3],
            qd: 15.0,
            r: 150.0,
            du_max: 0.05,
            d_track_tol: 0.5, // loose gate so tests actually solve
        }
    }

    #[test]
    fn init_polls_until_all_encoders_above_threshold() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let mpc = MpcLoop::init(&mut bus, true, &weights()).unwrap();
        assert!(mpc.d_op.iter().all(|&d| (d - 0.2).abs() < 2e-3));
    }

    #[test]
    fn controller_off_sends_no_motor_commands() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let mut mpc = MpcLoop::init(&mut bus, true, &weights()).unwrap();
        mpc.controller_on = false;

        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        push_imu_reply(&mut bus, 0.0, 0.0);
        let writes_before = bus.writes.len();
        let tel = mpc.one_step(&mut bus);
        assert_eq!(tel.exitflag, 0);
        // Only the 4 encoder-read requests + 1 IMU-read command should have
        // been written, no run_abs/run_rel motor commands.
        assert_eq!(bus.writes.len() - writes_before, 5);
    }

    /// The motor commands must carry the GUI's speed/accel (MATLAB reads
    /// `app.SpeedSpinner`/`app.AccSpinner` in the loop), not a hardcoded
    /// pair — running slower than MATLAB is what made the axes miss their
    /// target inside one `plantT` and stall the loop on the tracking gate.
    #[test]
    fn motor_commands_use_configured_speed_accel() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let mut mpc = MpcLoop::init(&mut bus, true, &weights()).unwrap();
        mpc.spd = 180.0;
        mpc.acc = 64;

        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        push_imu_reply(&mut bus, 0.0, 0.0);
        // Skip init's own 4 nudge frames, which use MATLAB's fixed 50/128.
        let writes_before = bus.writes.len();
        mpc.one_step(&mut bus);

        // spd = clamp(180*16, 0, 3000) = 2880 = 0x0B40, acc in byte 5.
        let motor_writes: Vec<_> = bus.writes[writes_before..]
            .iter()
            .filter(|w| w.len() == 11 && (w[2] == 0xF5 || w[2] == 0xF4))
            .collect();
        assert_eq!(motor_writes.len(), 4);
        for w in motor_writes {
            assert_eq!((w[3], w[4]), (0x0B, 0x40), "speed field");
            assert_eq!(w[5], 64, "accel field");
        }
    }

    #[test]
    fn d_track_tol_gate_holds_previous_command() {
        let mut bus = MockBus::default();
        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.2);
        }
        let mut w = weights();
        w.d_track_tol = 0.0001; // impossibly tight -> always gated
        let mut mpc = MpcLoop::init(&mut bus, true, &w).unwrap();
        let prev_cmd = mpc.d_cmd_prev;

        for addr in 1..=4u8 {
            push_encoder_reply(&mut bus, addr, 0.25); // moved -> triggers gate
        }
        push_imu_reply(&mut bus, 0.0, 0.0);
        let tel = mpc.one_step(&mut bus);
        assert_eq!(tel.exitflag, -99);
        assert_eq!(tel.d_cmd, prev_cmd);
    }
}
