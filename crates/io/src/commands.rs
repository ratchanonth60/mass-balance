//! Messages the UI thread sends into the IO thread. `Bus` (see
//! `crate::transport`) is owned exclusively by the IO thread, so every
//! manual/UI-triggered serial action — jog, home, encoder read — has to be a
//! command through this queue rather than a direct call, or it would
//! interleave frames with whatever automatic loop is running.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Mpc,
    Lqi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassSolver {
    Pinv,
    Qp,
}

/// GUI-tunable MPC weights (`app.q1EditField.Value` etc — the `Tuning` tab).
/// Only consulted by the `ModelInit.m` preset (`XYPreCheckbox` off);
/// `ModelInit_PostBalance.m` ignores these. Persisted as `tune_weights.json`
/// (M5), replacing `TuneWeights.mat`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TuneWeights {
    pub q: [f64; 5],
    pub qi: [f64; 2],
    pub qd: f64,
    pub r: f64,
    pub du_max: f64,
    pub d_track_tol: f64,
    /// Manual-jog speed (RPM) / accel — not an MPC/LQI model weight, but
    /// bundled here on request so it saves/loads alongside tuning instead of
    /// needing a second persisted file for two DragValues.
    #[serde(default = "default_jog_spd")]
    pub jog_spd: f64,
    #[serde(default = "default_jog_acc")]
    pub jog_acc: u8,
}

/// `app.SpeedSpinner`/`app.AccSpinner` `createComponents` defaults. Used for
/// manual jog *and* — as in MATLAB — for the MPC/LQI loops' own motor
/// commands. `TuneWeights.mat` never existed in the MATLAB tree, so the app's
/// startup load fell through and these are what the rig actually ran with.
pub const DEFAULT_SPD: f64 = 180.0;
pub const DEFAULT_ACC: u8 = 64;

fn default_jog_spd() -> f64 {
    DEFAULT_SPD
}

fn default_jog_acc() -> u8 {
    DEFAULT_ACC
}

impl Default for TuneWeights {
    fn default() -> Self {
        // The real `createComponents` values of `app.q1EditField` etc, read
        // out of AutoMass_MPC.mlapp's document.xml — these are what the rig
        // actually ran with, since `TuneWeights.mat` was never saved so the
        // app's startup load never overrode them. (Earlier these were guessed
        // from ModelInit_PostBalance.m's hardcoded tuning, which gave a
        // 3x-tighter `d_track_tol` than MATLAB — that gate holding is what
        // stalls convergence.) Only consulted for the `ModelInit` preset;
        // `ModelInit_PostBalance` hardcodes its own on both sides.
        Self {
            q: [50.0, 50.0, 25.0, 25.0, 50.0],
            qi: [0.3, 0.3],
            qd: 5.0,
            r: 400.0,
            du_max: 0.010,
            d_track_tol: 0.030,
            jog_spd: default_jog_spd(),
            jog_acc: default_jog_acc(),
        }
    }
}

impl From<TuneWeights> for control::mpc_model::TuningWeights {
    fn from(w: TuneWeights) -> Self {
        control::mpc_model::TuningWeights {
            q: w.q,
            qi: w.qi,
            qd: w.qd,
            r: w.r,
            du_max: w.du_max,
            d_track_tol: w.d_track_tol,
        }
    }
}

#[derive(Debug, Clone)]
pub enum UiCommand {
    Connect(String),
    Disconnect,
    RefreshPorts,
    /// `HomeButtonPushed`: sends `0x91` to all 4 axes in sequence.
    Home,
    /// `EncoderButtonPushed`: reads one axis's encoder into the UI.
    ReadEncoder(u8),
    /// `GoButtonPushed`/`SetButtonPushed`: absolute move on one axis.
    JogAbsolute {
        addr: u8,
        dist_m: f64,
        spd: f64,
        acc: u8,
    },
    /// `ShiftButton`: relative move on one axis.
    JogRelative {
        addr: u8,
        dist_m: f64,
        spd: f64,
        acc: u8,
    },
    /// `ControllerSwitch` value-changed.
    SetController(bool),
    /// `RollSpinner`/`PitchSpinner` value-changed: MPC/LQI setpoint.
    SetSetpoint {
        roll_deg: f64,
        pitch_deg: f64,
    },
    /// `XYPreCheckbox` value-changed: selects `ModelInit` vs
    /// `ModelInit_PostBalance` preset.
    SetXyPreBalance(bool),
    SetMassSolver(MassSolver),
    UpdateWeights(TuneWeights),
    /// `AutoButtonPushed`, after the MPC-vs-LQR `questdlg`.
    StartAuto(RunMode),
    /// `StopRequestForAutoBalanceSwitch` -> "Cancle".
    Stop,
    /// `TuneNowPushed`: hot-reload weights into the running MPC loop.
    TuneNow,
    /// Not in the original MATLAB app: continuously polls all 4 encoders
    /// (no motor commands, no control math) so the Motor Control cards
    /// update live while manually jogging, instead of needing a
    /// `ReadEncoder` click after every move. Only takes effect while
    /// `Idle` — does not interrupt a running MPC/LQI loop.
    SetManualPoll(bool),
}
