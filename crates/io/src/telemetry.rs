//! Snapshots sent from the IO thread to the UI thread after every control
//! cycle (replacing `DrawPlot`/`drawnow limitrate nocallbacks`) and after
//! manual commands complete.

#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    /// Whether the IO thread currently holds an open `Bus`. Also sent as a
    /// standalone snapshot (all other fields default) right after every
    /// `Connect`/`Disconnect` command, since those don't otherwise produce
    /// telemetry — the UI has no other way to learn a `Connect` failed.
    pub connected: bool,
    /// True only for a real control-loop cycle (`MpcLoop`/`LqiLoop::one_step`).
    /// False for the standalone status snapshots sent after `Connect`/
    /// `Disconnect`/`ReadEncoder` etc — those only carry `connected`/`status`,
    /// every other field is a meaningless default (`t=0`, angles/positions
    /// all 0) and must not be plotted or shown as a live reading.
    pub is_sample: bool,
    /// `(addr, value_m)` set only by a manual `ReadEncoder` command's result
    /// snapshot — lets the UI show the reading on that axis's card even
    /// though no loop is running to produce real `is_sample` telemetry.
    pub manual_read: Option<(u8, f64)>,
    pub t: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub gyro_dps: [f64; 3],
    pub d_meas: [f64; 4],
    pub d_cmd: [f64; 4],
    pub e_deg: [f64; 2],
    /// `info.exitflag` from the MPC solve (`0` = held/off, `-99` = actuator
    /// still tracking previous command, `<=0` = infeasible/fallback,
    /// `>0` = solved), or a synthetic code for the LQI path.
    pub exitflag: i32,
    pub status: String,
    pub cycle: u64,
}
