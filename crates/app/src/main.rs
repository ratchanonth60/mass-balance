use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use io::commands::{MassSolver, RunMode, TuneWeights, UiCommand};
use io::telemetry::{RunState, Telemetry};
use io::transport::SerialBus;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender};

const HISTORY_LEN: usize = 300;
/// Live plot only shows the trailing window of `history`, not the whole
/// 300-sample buffer — see `AutomassApp::windowed_history`.
const WINDOW_SECS: f64 = 120.0;
const WEIGHTS_PATH: &str = "tune_weights.json";
const PRESET_XY_PATH: &str = "preset_xy.json";
/// Rail travel range — matches `D_MAX` in `mpc_loop`/`lqi_loop`. Used only
/// for the axis progress bars' fill fraction, not sent anywhere.
const RAIL_MAX_M: f64 = 0.495;

/// `PresetXY.mat` -> `preset_xy.json` (M5). Matches the field name the M0
/// `.mat`->JSON converter emitted (`fixtures/preset_xy.json`'s `presetXY` key).
#[derive(serde::Deserialize)]
struct PresetXy {
    #[serde(rename = "presetXY")]
    preset_xy: [f64; 4],
}

/// `(label, color)` for a `RunState` — used in the header chip.
fn run_state_chip(state: RunState) -> (&'static str, egui::Color32) {
    match state {
        RunState::Idle => ("Idle", egui::Color32::GRAY),
        RunState::Manual => ("Manual poll", egui::Color32::from_rgb(90, 140, 220)),
        RunState::Mpc => ("▶ MPC", egui::Color32::from_rgb(80, 180, 80)),
        RunState::Lqi => ("▶ LQI", egui::Color32::from_rgb(80, 180, 80)),
    }
}

/// `(label, color)` for an `exitflag` — same codes in both loops
/// (`mpc_loop.rs`/`lqi_loop.rs`): `1` solved, `-99` the actuator-tracking
/// gate held and the solve was skipped, `-1` infeasible (MPC only, holds the
/// previous command), `0` controller off.
fn exitflag_badge(flag: i32) -> (&'static str, egui::Color32) {
    match flag {
        1 => ("✔ solved", egui::Color32::from_rgb(80, 180, 80)),
        -99 => ("⏳ gate held", egui::Color32::from_rgb(220, 160, 60)),
        -1 => ("⚠ infeasible", egui::Color32::from_rgb(200, 80, 60)),
        0 => ("off", egui::Color32::GRAY),
        _ => ("?", egui::Color32::GRAY),
    }
}

struct AutomassApp {
    tx: Sender<UiCommand>,
    rx: Receiver<Telemetry>,

    ports: Vec<String>,
    selected_port: String,
    connected: bool,
    connecting: bool,

    run_mode: RunMode,
    xy_pre_balance: bool,
    controller_on: bool,
    mass_solver: MassSolver,
    running: bool,

    setpoint_roll: f64,
    setpoint_pitch: f64,
    weights: TuneWeights,
    /// Result line under Tune Now/Save/Load — both used to swallow their
    /// `io::Result`/`serde_json::Result` and fail completely silently.
    weights_msg: String,

    jog_dist_cm: f64,
    manual_poll: bool,

    latest: Telemetry,
    history: VecDeque<Telemetry>,
    status_line: String,
    /// Last manual `Read encoder` result per axis (index = addr-1) — shown
    /// on the rig panel's axis cards even while idle, when `latest.d_meas`
    /// is meaningless (no loop running to have produced it).
    manual_d_meas: [Option<f64>; 4],
    /// Wall-clock baseline for the plot's zero-order-hold synthetic point —
    /// reset alongside `history.clear()` at every new session start. See
    /// `hold_extended`.
    session_start: std::time::Instant,
    /// Roll/pitch as of the *previous* real sample, and when the current
    /// one landed — `tilt_plate` eases between them instead of snapping.
    /// See `eased_angles`.
    prev_roll: f64,
    prev_pitch: f64,
    angle_update_at: std::time::Instant,
}

impl AutomassApp {
    fn new(tx: Sender<UiCommand>, rx: Receiver<Telemetry>) -> Self {
        let weights = std::fs::read_to_string(WEIGHTS_PATH)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self {
            tx,
            rx,
            ports: SerialBus::list_ports(),
            selected_port: String::new(),
            connected: false,
            connecting: false,
            run_mode: RunMode::Mpc,
            // MATLAB's `XYPreCheckbox` is an unchecked `uicheckbox` (only the
            // XY-preset button ever sets it), so `MainConstantTs11.m:76`
            // takes the `else` branch and runs `ModelInit(app, d0)` — the
            // GUI-tunable preset. Defaulting this to `true` here silently put
            // the rig on `ModelInit_PostBalance`'s hardcoded tuning instead,
            // with the whole Tuning panel inert.
            xy_pre_balance: false,
            controller_on: true,
            mass_solver: MassSolver::Pinv,
            running: false,
            setpoint_roll: 0.0,
            setpoint_pitch: 0.0,
            weights,
            weights_msg: String::new(),
            jog_dist_cm: 0.0,
            manual_poll: false,
            latest: Telemetry::default(),
            history: VecDeque::with_capacity(HISTORY_LEN),
            status_line: "Disconnected".to_string(),
            manual_d_meas: [None; 4],
            session_start: std::time::Instant::now(),
            prev_roll: 0.0,
            prev_pitch: 0.0,
            angle_update_at: std::time::Instant::now(),
        }
    }

    fn send(&self, cmd: UiCommand) {
        let _ = self.tx.send(cmd);
    }

    fn drain_telemetry(&mut self) {
        while let Ok(tel) = self.rx.try_recv() {
            self.status_line = tel.status.clone();
            self.connected = tel.connected;
            self.connecting = false;
            // Always current, unlike the rest of `latest` below — a failed
            // `StartAuto` (init timeout, gain solve failure) or a loop panic
            // reports `Idle` through a standalone status snapshot
            // (`is_sample` false), not a real sample, so it has to be read
            // here or the header chip and the `running` correction below
            // would never see it.
            self.latest.run_state = tel.run_state;
            if tel.run_state == RunState::Idle {
                // Recovers `running` from a failed Start Auto or a loop
                // panic without waiting on the Stop button — see the
                // `RunState` doc comment in `io::telemetry` for why this
                // exists.
                self.running = false;
            }
            if let Some((addr, v)) = tel.manual_read {
                self.manual_d_meas[addr as usize - 1] = Some(v);
            }
            // Command-status snapshots (Connect/Disconnect/ReadEncoder/...)
            // carry no real reading (`is_sample` false, every other field is
            // a meaningless default) — only real control-loop cycles should
            // update the live grid or get plotted, or the graph snaps back
            // to (t=0, y=0) every time a manual command fires.
            if tel.is_sample {
                self.prev_roll = self.latest.roll_deg;
                self.prev_pitch = self.latest.pitch_deg;
                self.angle_update_at = std::time::Instant::now();
                self.latest = tel.clone();
                if self.history.len() >= HISTORY_LEN {
                    self.history.pop_front();
                }
                self.history.push_back(tel);
            }
        }
    }

    /// Best-known reading per axis for the preflight check: live if a loop
    /// or the manual live-read poll is running, else the last one-shot
    /// `Read encoder` result, else unknown (`None`) — a stale one-shot read
    /// is the weakest signal, and no reading at all means "ask the operator
    /// to check" rather than silently assuming either way.
    fn axis_readiness(&self) -> [Option<f64>; 4] {
        if matches!(
            self.latest.run_state,
            RunState::Mpc | RunState::Lqi | RunState::Manual
        ) {
            self.latest.d_meas.map(Some)
        } else {
            self.manual_d_meas
        }
    }

    /// Share of the plot window's samples that landed on the `d_track_tol`
    /// gate (`exitflag == -99`, solve skipped). A single instantaneous flag
    /// doesn't show a *trend* — a climbing gate rate is the actual signal
    /// that motor speed/tolerance need attention, one held cycle isn't.
    fn gate_rate_pct(&self) -> f64 {
        let mut total = 0usize;
        let mut gated = 0usize;
        for t in self.windowed_history() {
            total += 1;
            if t.exitflag == -99 {
                gated += 1;
            }
        }
        if total == 0 {
            0.0
        } else {
            100.0 * gated as f64 / total as f64
        }
    }
}

impl eframe::App for AutomassApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_telemetry();
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));

        // Header: connection, live roll/pitch, run state + solver health,
        // and Stop are always visible — no tabs, so nothing to switch away
        // from to reach them.
        egui::Panel::top("header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(if self.connected { "🟢" } else { "🔴" });
                if !self.connected {
                    if ui.button("⟳").on_hover_text("Refresh ports").clicked() {
                        self.ports = SerialBus::list_ports();
                    }
                    egui::ComboBox::from_id_salt("port_header")
                        .selected_text(if self.selected_port.is_empty() {
                            "Select port"
                        } else {
                            self.selected_port.as_str()
                        })
                        .show_ui(ui, |ui| {
                            for p in &self.ports {
                                ui.selectable_value(&mut self.selected_port, p.clone(), p);
                            }
                        });
                    let label = if self.connecting {
                        "Connecting…"
                    } else {
                        "Connect"
                    };
                    if ui
                        .add_enabled(!self.connecting, egui::Button::new(label))
                        .clicked()
                        && !self.selected_port.is_empty()
                    {
                        self.send(UiCommand::Connect(self.selected_port.clone()));
                        self.connecting = true;
                        self.status_line = "Connecting…".to_string();
                    }
                } else {
                    ui.label(&self.selected_port);
                    if ui.button("Disconnect").clicked() {
                        self.send(UiCommand::Disconnect);
                        self.running = false;
                        self.latest.run_state = RunState::Idle;
                    }
                }

                ui.separator();
                ui.label(
                    egui::RichText::new(format!("Roll {:.2}°", self.latest.roll_deg)).strong(),
                );
                ui.label(
                    egui::RichText::new(format!("Pitch {:.2}°", self.latest.pitch_deg)).strong(),
                );

                ui.separator();
                let (chip_text, chip_color) = run_state_chip(self.latest.run_state);
                ui.label(egui::RichText::new(chip_text).color(chip_color).strong());
                if matches!(self.latest.run_state, RunState::Mpc | RunState::Lqi) {
                    ui.label(format!("cycle {}", self.latest.cycle));
                    let (flag_text, flag_color) = exitflag_badge(self.latest.exitflag);
                    ui.label(egui::RichText::new(flag_text).color(flag_color));
                    ui.label(format!("gate {:.0}%", self.gate_rate_pct()));
                }

                ui.separator();
                if self.running
                    && ui
                        .add(
                            egui::Button::new("■ Stop")
                                .fill(egui::Color32::from_rgb(140, 20, 20)),
                        )
                        .clicked()
                {
                    self.send(UiCommand::Stop);
                    self.running = false;
                    // `Stop` doesn't produce a `Telemetry` on the IO side
                    // (nothing left to send it from once the loop stops
                    // iterating) — patch the chip directly instead of
                    // leaving it stuck on the last mode shown.
                    self.latest.run_state = RunState::Idle;
                }
            });
        });

        // Full status/diagnostic line at the bottom — long diagnostic
        // strings (e.g. "Encoder[1]: 10 bytes, unparsable: [FB, 01, ...]")
        // get the full window width instead of competing with tab labels.
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.label(format!("Status: {}", self.status_line));
        });

        egui::Panel::left("rig")
            .default_size(300.0)
            .min_size(240.0)
            .show(ui, |ui| self.rig_panel(ui));

        egui::Panel::right("run")
            .default_size(320.0)
            .min_size(260.0)
            .show(ui, |ui| self.run_panel(ui));

        egui::CentralPanel::default().show(ui, |ui| self.monitor_panel(ui));
    }
}

impl AutomassApp {
    /// Left panel: everything about the physical rig — per-axis readouts
    /// and jog controls, home, encoder reads, XY preset. Grouped here
    /// instead of a separate tab because prepping the rig (jogging masses
    /// above the auto-balance init threshold) is the step right before
    /// hitting Start on the right, and used to require a tab switch between
    /// the two.
    fn rig_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Rig");
        // Jog/home while an MPC/LQI loop is running fights the controller —
        // the IO thread drains manual commands between control cycles, so
        // they *do* fire, just not safely. Reading encoders is harmless
        // (read-only), so only motion commands are gated on `!running`.
        let can_move = self.connected && !self.running;
        if !self.connected {
            ui.label(
                egui::RichText::new("Connect first — motor commands need a live port.").weak(),
            );
        } else if self.running {
            ui.label(
                egui::RichText::new(
                    "Auto-balance running — manual moves disabled, use header Stop first.",
                )
                .weak(),
            );
        }
        ui.add_space(4.0);

        egui::ScrollArea::vertical()
            .id_salt("rig_axes_scroll")
            .show(ui, |ui| {
                for i in 0..4 {
                    let addr = i as u8 + 1;
                    ui.group(|ui| {
                        ui.set_min_width(ui.available_width());
                        let row = ui.horizontal(|ui| {
                            ui.strong(format!("Ax{addr}"));
                            ui.label(format!("{:.2} cm", self.latest.d_meas[i] * 100.0));
                        });
                        row.response.on_hover_text(format!(
                            "d_cmd {:.3} cm — last one-shot read: {}",
                            self.latest.d_cmd[i] * 100.0,
                            match self.manual_d_meas[i] {
                                Some(v) => format!("{:.3} cm", v * 100.0),
                                None => "-".to_string(),
                            }
                        ));
                        let frac = (self.latest.d_meas[i] / RAIL_MAX_M).clamp(0.0, 1.0) as f32;
                        ui.add(egui::ProgressBar::new(frac));
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(can_move, egui::Button::new("→"))
                                .on_hover_text("Go (absolute)")
                                .clicked()
                            {
                                self.send(UiCommand::JogAbsolute {
                                    addr,
                                    dist_m: self.jog_dist_cm / 100.0,
                                    spd: self.weights.jog_spd,
                                    acc: self.weights.jog_acc,
                                });
                            }
                            if ui
                                .add_enabled(can_move, egui::Button::new("±"))
                                .on_hover_text("Shift (relative)")
                                .clicked()
                            {
                                self.send(UiCommand::JogRelative {
                                    addr,
                                    dist_m: self.jog_dist_cm / 100.0,
                                    spd: self.weights.jog_spd,
                                    acc: self.weights.jog_acc,
                                });
                            }
                            if ui
                                .add_enabled(self.connected, egui::Button::new("⟳"))
                                .on_hover_text("Read encoder")
                                .clicked()
                            {
                                self.send(UiCommand::ReadEncoder(addr));
                            }
                        });
                    });
                }
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("dist");
            ui.add(egui::DragValue::new(&mut self.jog_dist_cm).speed(0.1).suffix(" cm"));
        });
        // Speed/accel live in the Tuning panel — saved and loaded alongside
        // the MPC weights instead of a separate untracked pair of fields.
        // Shown read-only here so it's still visible while jogging.
        ui.label(format!(
            "Speed {:.0} RPM / Accel {} (edit in Tuning)",
            self.weights.jog_spd, self.weights.jog_acc
        ));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_move, egui::Button::new("Go all"))
                .clicked()
            {
                for addr in 1..=4u8 {
                    self.send(UiCommand::JogAbsolute {
                        addr,
                        dist_m: self.jog_dist_cm / 100.0,
                        spd: self.weights.jog_spd,
                        acc: self.weights.jog_acc,
                    });
                }
            }
            if ui
                .add_enabled(can_move, egui::Button::new("Shift all"))
                .clicked()
            {
                for addr in 1..=4u8 {
                    self.send(UiCommand::JogRelative {
                        addr,
                        dist_m: self.jog_dist_cm / 100.0,
                        spd: self.weights.jog_spd,
                        acc: self.weights.jog_acc,
                    });
                }
            }
        });
        if ui
            .add_enabled(self.connected, egui::Button::new("Read all encoders"))
            .clicked()
        {
            for addr in 1..=4u8 {
                self.send(UiCommand::ReadEncoder(addr));
            }
        }

        ui.separator();
        // `HomeButtonPushed`: sends 0x91 (MKS "go home") to all 4 axes —
        // a real seek back to the driver's stored home/zero point, not just
        // zeroing the encoder in place. Fire-and-forget (no completion
        // wait), so give immediate feedback instead of looking dead for the
        // several seconds homing actually takes.
        if ui
            .add_enabled(
                can_move,
                egui::Button::new("🏠 Home all (all motors → zero)"),
            )
            .clicked()
        {
            self.send(UiCommand::Home);
            self.manual_d_meas = [None; 4]; // stale the instant homing starts
            self.status_line = "Homing all 4 axes — wait for motion to stop".to_string();
        }
        // `XYButtonPushed`/`XYZButtonPushed`: load('PresetXY.mat') then drive
        // all 4 axes there — replaced by `preset_xy.json` (M5).
        if ui
            .add_enabled(can_move, egui::Button::new("Go to XY preset"))
            .clicked()
        {
            match std::fs::read_to_string(PRESET_XY_PATH) {
                Ok(s) => match serde_json::from_str::<PresetXy>(&s) {
                    Ok(preset) => {
                        for (i, &d) in preset.preset_xy.iter().enumerate() {
                            self.send(UiCommand::JogAbsolute {
                                addr: i as u8 + 1,
                                dist_m: d,
                                spd: self.weights.jog_spd,
                                acc: self.weights.jog_acc,
                            });
                        }
                        self.status_line = format!(
                            "Sent XY preset: {:.3} {:.3} {:.3} {:.3} m",
                            preset.preset_xy[0],
                            preset.preset_xy[1],
                            preset.preset_xy[2],
                            preset.preset_xy[3]
                        );
                    }
                    Err(e) => {
                        self.status_line = format!("{PRESET_XY_PATH}: bad JSON ({e})");
                    }
                },
                Err(e) => {
                    self.status_line = format!("{PRESET_XY_PATH}: {e}");
                }
            }
        }

        ui.separator();
        if ui
            .checkbox(
                &mut self.manual_poll,
                "🔴 Live read (poll encoders, no motor commands)",
            )
            .changed()
        {
            if self.manual_poll {
                self.history.clear(); // new session, `t` clock restarts at 0
                self.session_start = std::time::Instant::now();
            }
            self.send(UiCommand::SetManualPoll(self.manual_poll));
        }
        if self.manual_poll {
            ui.label("Polling all 4 axes every 0.5s.");
        }
    }

    /// Right panel: mode/preset/controller, setpoints, preflight, Start —
    /// then the Tuning weights grid below, since tuning and running are the
    /// two things an operator alternates between second-to-second and used
    /// to be a tab switch apart.
    fn run_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Run");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.run_mode, RunMode::Mpc, "MPC");
            ui.radio_value(&mut self.run_mode, RunMode::Lqi, "LQI");
        });
        ui.checkbox(
            &mut self.xy_pre_balance,
            "XY pre-balanced (ModelInit_PostBalance)",
        );
        if ui
            .checkbox(&mut self.controller_on, "Controller On")
            .changed()
        {
            self.send(UiCommand::SetController(self.controller_on));
        }
        if !self.controller_on {
            ui.label(
                egui::RichText::new(
                    "⚠ Controller off — loop keeps running (KF/solve) but sends no motor commands.",
                )
                .color(egui::Color32::from_rgb(220, 120, 60)),
            );
        }

        ui.add_space(4.0);
        ui.label("Allocation solver (LQI path)");
        ui.horizontal(|ui| {
            if ui
                .radio_value(&mut self.mass_solver, MassSolver::Pinv, "Pseudo-inverse")
                .changed()
                || ui
                    .radio_value(&mut self.mass_solver, MassSolver::Qp, "QP")
                    .changed()
            {
                self.send(UiCommand::SetMassSolver(self.mass_solver));
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Roll setpoint (deg)");
            if ui
                .add(egui::DragValue::new(&mut self.setpoint_roll).speed(0.1))
                .changed()
            {
                self.send(UiCommand::SetSetpoint {
                    roll_deg: self.setpoint_roll,
                    pitch_deg: self.setpoint_pitch,
                });
            }
        });
        ui.horizontal(|ui| {
            ui.label("Pitch setpoint (deg)");
            if ui
                .add(egui::DragValue::new(&mut self.setpoint_pitch).speed(0.1))
                .changed()
            {
                self.send(UiCommand::SetSetpoint {
                    roll_deg: self.setpoint_roll,
                    pitch_deg: self.setpoint_pitch,
                });
            }
        });

        ui.separator();
        // Preflight: both loops' `init` blocks the IO thread — with zero
        // feedback beyond a "Initializing..." status line — for up to a
        // full minute before reporting failure if the masses aren't already
        // jogged above this threshold. Warn ahead of time instead of
        // silently eating that minute. Never blocks Start: a stale one-shot
        // read is not authoritative, and the operator may know the rig is
        // ready even if the UI's last reading is old.
        let min_d = match self.run_mode {
            RunMode::Mpc => io::mpc_loop::INIT_MIN_D,
            RunMode::Lqi => io::lqi_loop::INIT_MIN_D,
        };
        let readiness = self.axis_readiness();
        if readiness.iter().any(|v| v.is_none()) {
            ui.label(
                egui::RichText::new("↻ Read all encoders (left panel) to check readiness")
                    .weak(),
            );
        } else {
            let below: Vec<String> = readiness
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    v.filter(|&d| d < min_d)
                        .map(|d| format!("Ax{} {:.1} cm", i + 1, d * 100.0))
                })
                .collect();
            if below.is_empty() {
                ui.label(
                    egui::RichText::new("✔ all axes ready")
                        .color(egui::Color32::from_rgb(80, 180, 80)),
                );
            } else {
                ui.label(
                    egui::RichText::new(format!(
                        "⚠ {} below {:.1} cm — jog up first",
                        below.join(", "),
                        min_d * 100.0
                    ))
                    .color(egui::Color32::from_rgb(220, 160, 60)),
                );
            }
        }

        if ui
            .add_enabled(
                !self.running && self.connected,
                egui::Button::new("▶ Start Auto"),
            )
            .clicked()
        {
            // Manual live-read poll occupies the same `Run` slot on
            // the IO thread — StartAuto silently supersedes it
            // there, so drop the client-side flag to match.
            self.manual_poll = false;
            // Every new session's `t` clock restarts at 0 on the IO
            // side (`MpcLoop`/`LqiLoop::init`) — old history entries
            // from a previous session have larger `t`, so leaving
            // them in makes the plot line jump backward in time and
            // connect across sessions. Start clean.
            self.history.clear();
            self.session_start = std::time::Instant::now();
            self.send(UiCommand::SetXyPreBalance(self.xy_pre_balance));
            self.send(UiCommand::UpdateWeights(self.weights));
            self.send(UiCommand::SetMassSolver(self.mass_solver));
            self.send(UiCommand::StartAuto(self.run_mode));
            self.running = true;
        }

        ui.add_space(8.0);
        ui.separator();
        // Nested scroll area, not the whole panel — Run controls above stay
        // pinned and reachable even on a short window; only the (long)
        // weights grid needs its own scroll to still reach Tune Now.
        egui::ScrollArea::vertical()
            .id_salt("tuning_scroll")
            .show(ui, |ui| self.tuning_panel(ui));
    }

    fn tuning_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Tuning");
        // `ModelInit_PostBalance.m` hardcodes its own q/qi/qd/R/du_max/
        // d_track_tol and never reads the GUI — so with the preset
        // checkbox on, every weight below except speed/accel is dead and
        // `Tune Now` re-linearizes with the same numbers. Silent in
        // MATLAB; say it out loud instead of letting an operator drag
        // knobs that do nothing.
        if self.xy_pre_balance && self.run_mode == RunMode::Mpc {
            ui.label(
                egui::RichText::new(
                    "⚠ 'XY pre-balanced' is on — MPC uses ModelInit_PostBalance's \
                     hardcoded weights and ignores everything below except speed/accel.",
                )
                .color(egui::Color32::from_rgb(220, 160, 60)),
            );
        }
        egui::Grid::new("weights_grid")
            .num_columns(2)
            .show(ui, |ui| {
                for (i, label) in ["q1", "q2", "q3", "q4", "q5"].iter().enumerate() {
                    ui.label(*label);
                    ui.add(egui::DragValue::new(&mut self.weights.q[i]).speed(1.0));
                    ui.end_row();
                }
                for (i, label) in ["qi1", "qi2"].iter().enumerate() {
                    ui.label(*label);
                    ui.add(egui::DragValue::new(&mut self.weights.qi[i]).speed(0.1));
                    ui.end_row();
                }
                ui.label("qd");
                ui.add(egui::DragValue::new(&mut self.weights.qd).speed(1.0));
                ui.end_row();
                ui.label("R");
                ui.add(egui::DragValue::new(&mut self.weights.r).speed(1.0));
                ui.end_row();
                ui.label("du_max (m)");
                ui.add(egui::DragValue::new(&mut self.weights.du_max).speed(0.001));
                ui.end_row();
                ui.label("d_track_tol (m)");
                ui.add(egui::DragValue::new(&mut self.weights.d_track_tol).speed(0.001));
                ui.end_row();
                // Not an MPC/LQI model weight, but MATLAB's loop feeds
                // these same two spinners to every motor command it
                // sends — so they set both manual-jog and auto-balance
                // move speed. `Tune Now` pushes them into a running loop.
                ui.label("motor speed (RPM)");
                ui.add(egui::DragValue::new(&mut self.weights.jog_spd).speed(1.0));
                ui.end_row();
                ui.label("motor accel");
                ui.add(egui::DragValue::new(&mut self.weights.jog_acc).range(0..=255));
                ui.end_row();
            });

        ui.horizontal(|ui| {
            if ui.button("Tune Now").clicked() {
                self.send(UiCommand::UpdateWeights(self.weights));
                self.send(UiCommand::TuneNow);
                self.weights_msg = "Sent updated weights + Tune Now".to_string();
            }
            if ui.button("Save weights").clicked() {
                self.weights_msg = match serde_json::to_string_pretty(&self.weights) {
                    Ok(json) => match std::fs::write(WEIGHTS_PATH, json) {
                        Ok(()) => format!("Saved to {WEIGHTS_PATH}"),
                        Err(e) => format!("Save failed: {e}"),
                    },
                    Err(e) => format!("Save failed: {e}"),
                };
            }
            if ui.button("Load weights").clicked() {
                self.weights_msg = match std::fs::read_to_string(WEIGHTS_PATH) {
                    Ok(s) => match serde_json::from_str(&s) {
                        Ok(w) => {
                            self.weights = w;
                            format!("Loaded {WEIGHTS_PATH}")
                        }
                        Err(e) => format!("Load failed: bad JSON ({e})"),
                    },
                    Err(e) => format!("Load failed: {e}"),
                };
            }
        });
        if !self.weights_msg.is_empty() {
            ui.label(egui::RichText::new(&self.weights_msg).weak());
        }
    }

    /// Points from the trailing `WINDOW_SECS` of `history` only — bounds the
    /// plot to a scrolling live window instead of the whole 300-sample
    /// buffer (up to ~16min at MPC's 2s cadence), so the view doesn't keep
    /// stretching wider and the line reads as a live trace, not a slowly
    /// growing archive. Real sample cadence (2s MPC / 3.25s LQI) is control-
    /// loop timing, not UI-fakeable — this just picks a better window, it
    /// doesn't invent points between real ones.
    fn windowed_history(&self) -> impl Iterator<Item = &Telemetry> {
        let max_t = self.history.back().map_or(0.0, |t| t.t);
        let cutoff = max_t - WINDOW_SECS;
        self.history.iter().filter(move |t| t.t >= cutoff)
    }

    /// `windowed_history` mapped through `extract`, plus one synthetic
    /// trailing point at wall-clock "now" repeating the last real value —
    /// a zero-order-hold visual so the line keeps extending forward every
    /// repaint (~100ms) instead of sitting flat/jumping only when a real
    /// sample lands every plantT (2-3.25s). Doesn't invent new values, only
    /// repeats the last real one at a synthetic timestamp; never written to
    /// `history`, recomputed fresh each frame.
    fn hold_extended(&self, extract: impl Fn(&Telemetry) -> f64) -> PlotPoints<'_> {
        let mut pts: Vec<[f64; 2]> = self.windowed_history().map(|t| [t.t, extract(t)]).collect();
        if let Some(&[_, last_y]) = pts.last() {
            pts.push([self.session_start.elapsed().as_secs_f64(), last_y]);
        }
        pts.into()
    }

    /// Isometric-projected square, tilted by live roll/pitch, drawn beside a
    /// flat gray "level" reference square — a visual sanity check for
    /// "does the plate actually tilt the direction the numbers say", not a
    /// scaled model of the rig (no real rail geometry, just roll about X /
    /// pitch about Y on a generic square). Plain 2D painter math, no 3D
    /// engine dependency for something this simple.
    /// Roll/pitch eased from the previous real sample toward the current
    /// one over `ANGLE_EASE_SECS`, instead of snapping the instant a new
    /// sample lands (2-3.25s cadence otherwise makes the tilt plate jump).
    /// Pure tween between two real values — never extrapolates past the
    /// current one.
    fn eased_angles(&self) -> (f64, f64) {
        const ANGLE_EASE_SECS: f64 = 0.4;
        let t = (self.angle_update_at.elapsed().as_secs_f64() / ANGLE_EASE_SECS).min(1.0);
        let roll = self.prev_roll + (self.latest.roll_deg - self.prev_roll) * t;
        let pitch = self.prev_pitch + (self.latest.pitch_deg - self.prev_pitch) * t;
        (roll, pitch)
    }

    fn tilt_plate(&self, ui: &mut egui::Ui) {
        ui.label("Platform tilt (visual sanity check, not to rig scale — gray = level)");
        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(240.0, 240.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);

        let center = rect.center();
        let px_per_unit = 60.0_f32;
        let iso = 30.0_f64.to_radians();

        // Rx(roll) then Ry(pitch) on a unit square in the XY plane (Z=0),
        // then a standard isometric projection to 2D. Order/axis convention
        // is a judgment call for a debug visual, not load-bearing math —
        // what matters is it's consistent with roll_deg/pitch_deg's sign.
        let project = |x: f64, y: f64, roll_deg: f64, pitch_deg: f64| -> egui::Pos2 {
            let roll = roll_deg.to_radians();
            let pitch = pitch_deg.to_radians();
            let (y1, z1) = (y * roll.cos(), y * roll.sin());
            let x2 = x * pitch.cos() + z1 * pitch.sin();
            let z2 = -x * pitch.sin() + z1 * pitch.cos();
            let y2 = y1;
            let sx = (x2 - y2) * iso.cos();
            let sy = (x2 + y2) * iso.sin() - z2;
            egui::pos2(
                center.x + (sx as f32) * px_per_unit,
                center.y - (sy as f32) * px_per_unit, // screen Y grows down
            )
        };

        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

        let level: Vec<egui::Pos2> = corners
            .iter()
            .map(|&(x, y)| project(x, y, 0.0, 0.0))
            .collect();
        for i in 0..4 {
            painter.line_segment(
                [level[i], level[(i + 1) % 4]],
                egui::Stroke::new(1.0, egui::Color32::GRAY),
            );
        }

        let (roll, pitch) = self.eased_angles();
        let tilted: Vec<egui::Pos2> = corners
            .iter()
            .map(|&(x, y)| project(x, y, roll, pitch))
            .collect();
        painter.add(egui::Shape::convex_polygon(
            tilted,
            egui::Color32::from_rgba_unmultiplied(255, 140, 0, 60),
            egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 140, 0)),
        ));

        ui.label(format!("roll {roll:.2}°  pitch {pitch:.2}°"));
    }

    /// Center panel: live states strip, tilt-plate sanity check, and the two
    /// history plots. Everything here is only ever fed by `is_sample`
    /// telemetry (see `drain_telemetry`) — command-status snapshots no
    /// longer leak in as bogus `(t=0, y=0)` points, which was the main
    /// cause of the plot jumping/snapping instead of drawing a continuous
    /// line.
    fn monitor_panel(&mut self, ui: &mut egui::Ui) {
        // Roll/pitch are what an operator actually watches second to
        // second — call them out big, everything else stays in the compact
        // grid beside it.
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label("Roll (deg)");
                ui.heading(format!("{:.3}", self.latest.roll_deg));
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.label("Pitch (deg)");
                ui.heading(format!("{:.3}", self.latest.pitch_deg));
            });
            ui.separator();
            egui::Grid::new("states_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Rates (dps)");
                    ui.label(format!(
                        "{:.3} {:.3} {:.3}",
                        self.latest.gyro_dps[0], self.latest.gyro_dps[1], self.latest.gyro_dps[2]
                    ));
                    ui.end_row();
                    ui.label("Error (deg)");
                    ui.label(format!(
                        "{:.3} {:.3}",
                        self.latest.e_deg[0], self.latest.e_deg[1]
                    ));
                    ui.end_row();
                    ui.label("d_meas (cm)");
                    ui.label(format!(
                        "{:.3} {:.3} {:.3} {:.3}",
                        self.latest.d_meas[0] * 100.0,
                        self.latest.d_meas[1] * 100.0,
                        self.latest.d_meas[2] * 100.0,
                        self.latest.d_meas[3] * 100.0
                    ));
                    ui.end_row();
                    ui.label("d_cmd (cm)");
                    ui.label(format!(
                        "{:.3} {:.3} {:.3} {:.3}",
                        self.latest.d_cmd[0] * 100.0,
                        self.latest.d_cmd[1] * 100.0,
                        self.latest.d_cmd[2] * 100.0,
                        self.latest.d_cmd[3] * 100.0
                    ));
                    ui.end_row();
                    ui.label("Solve / cycle / gate");
                    ui.horizontal(|ui| {
                        let (flag_text, flag_color) = exitflag_badge(self.latest.exitflag);
                        ui.label(egui::RichText::new(flag_text).color(flag_color));
                        ui.label(format!(
                            "· cycle {} · {:.0}%",
                            self.latest.cycle,
                            self.gate_rate_pct()
                        ));
                    });
                    ui.end_row();
                });
        });

        ui.separator();
        self.tilt_plate(ui);
        ui.separator();

        // Stacked (Angle above Actuator positions), not side-by-side — full
        // panel width per plot instead of splitting it, easier to read the
        // trace against the longer time axis. Wrapped in a scroll area
        // since two full-width plots plus the states strip and tilt plate
        // no longer reliably fit a short window.
        egui::ScrollArea::vertical()
            .id_salt("monitor_plots_scroll")
            .show(ui, |ui| {
                let roll = self.hold_extended(|t| t.roll_deg);
                let pitch = self.hold_extended(|t| t.pitch_deg);
                ui.label("Angle (deg)");
                Plot::new("angle_plot")
                    .height(220.0)
                    .x_axis_label("t (s)")
                    .y_axis_label("deg")
                    .legend(Legend::default())
                    // Fixed home-range Y anchor so the axis doesn't rescale
                    // on every new point while roll/pitch sit near 0 (still
                    // expands automatically if the angle actually exceeds
                    // this).
                    .include_y(-5.0)
                    .include_y(5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("roll", roll).width(2.0));
                        plot_ui.line(Line::new("pitch", pitch).width(2.0));
                    });

                ui.add_space(8.0);
                ui.label("Actuator positions (cm)");
                Plot::new("d_plot")
                    .height(220.0)
                    .x_axis_label("t (s)")
                    .y_axis_label("cm")
                    .legend(Legend::default())
                    .include_y(0.0)
                    .include_y(50.0) // rail range is ~0-49.5cm (D_MAX)
                    .show(ui, |plot_ui| {
                        for axis in 0..4 {
                            let meas = self.hold_extended(|t| t.d_meas[axis] * 100.0);
                            let cmd = self.hold_extended(|t| t.d_cmd[axis] * 100.0);
                            plot_ui
                                .line(Line::new(format!("d{} meas", axis + 1), meas).width(2.0));
                            plot_ui.line(Line::new(format!("d{} cmd", axis + 1), cmd).width(2.0));
                        }
                    });
            });
    }
}

fn main() -> eframe::Result {
    let (tx_cmd, rx_cmd) = std::sync::mpsc::channel::<UiCommand>();
    let (tx_tel, rx_tel) = std::sync::mpsc::channel::<Telemetry>();

    std::thread::spawn(move || {
        io::thread::run(rx_cmd, tx_tel);
    });

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "AutoMass",
        options,
        Box::new(|_cc| Ok(Box::new(AutomassApp::new(tx_cmd, rx_tel)))),
    )
}
