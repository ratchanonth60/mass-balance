//! The IO thread: owns the one `Bus`, drains `UiCommand`s between control
//! cycles, and runs whichever loop (`MpcLoop`/`LqiLoop`) is active at its
//! `plantT` cadence. This is the piece that makes the port not just
//! "MATLAB rewritten in Rust" but actually safe under an immediate-mode
//! UI: the control loop's blocking `pause()`-equivalents run here, never on
//! the UI thread, and every manual jog/home/encoder-read command is funneled
//! through the same queue the automatic loop's commands go through — the
//! `Bus` never has two writers.

use crate::commands::{MassSolver, RunMode, TuneWeights, UiCommand};
use crate::lqi_loop::LqiLoop;
use crate::mks_ops;
use crate::mpc_loop::MpcLoop;
use crate::telemetry::{RunState, Telemetry};
use crate::transport::{Bus, SerialBus};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

const MPC_PLANT_T: Duration = Duration::from_millis(2000);
const LQI_PLANT_T: Duration = Duration::from_millis(3250); // matches Ctrl_LQR.mat's plantT=3.25
/// Not MATLAB parity — cadence for `Run::Manual`'s live encoder poll.
const MANUAL_POLL_T: Duration = Duration::from_millis(500);

#[derive()]
enum Run {
    Idle,
    /// Live encoder polling for manual jogging — no control math, no motor
    /// commands, just repeated `ReadEncoder` on all 4 axes. `t` is this
    /// mode's own clock (reset each time it's entered), separate from
    /// `MpcLoop`/`LqiLoop`'s. `d_meas` holds the last successful reading per
    /// axis so a single failed poll doesn't flash that axis back to 0 —
    /// matches `MpcLoop`/`LqiLoop`'s hold-previous behavior on a bad read.
    Manual {
        t: f64,
        d_meas: [f64; 4],
    },
    Mpc {
        loop_: MpcLoop,
    },
    Lqi {
        loop_: LqiLoop,
    },
}

/// Blocks forever (intended to be the whole body of a dedicated OS thread).
pub fn run(rx: Receiver<UiCommand>, tx: Sender<Telemetry>) {
    let mut bus: Option<Box<dyn Bus>> = None;
    let mut run = Run::Idle;
    let mut weights = TuneWeights::default();
    let mut mass_solver = MassSolver::Pinv;
    let mut xy_pre_balance = false;

    loop {
        // Drain every pending command before doing loop work — this is also
        // where `Stop`/`SetController`/`TuneNow` interrupt a running loop.
        while let Ok(cmd) = rx.try_recv() {
            handle_command(
                cmd,
                &mut bus,
                &mut run,
                &mut weights,
                &mut mass_solver,
                &mut xy_pre_balance,
                &tx,
            );
        }

        // A panic inside a loop's `one_step` (bad matrix solve, index out of
        // range, anything) would otherwise silently kill this whole thread —
        // no more telemetry ever again, UI just looks frozen with zero
        // explanation. Caught below and reported; `crashed` is applied to
        // `run` after the match ends (can't reassign the enum discriminant
        // from inside an arm that's borrowing it via match ergonomics).
        let mut crashed: Option<String> = None;

        match (&mut bus, &mut run) {
            (Some(bus), Run::Mpc { loop_ }) => {
                let started = Instant::now();
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    loop_.one_step(bus.as_mut())
                })) {
                    Ok(tel) => {
                        let _ = tx.send(tel);
                        sleep_remaining(started, MPC_PLANT_T);
                    }
                    Err(e) => crashed = Some(format!("MPC loop crashed: {}", panic_msg(&e))),
                }
            }
            (Some(bus), Run::Lqi { loop_ }) => {
                let started = Instant::now();
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    loop_.one_step(bus.as_mut(), LQI_PLANT_T.as_secs_f64())
                })) {
                    Ok(tel) => {
                        let _ = tx.send(tel);
                        sleep_remaining(started, LQI_PLANT_T);
                    }
                    Err(e) => crashed = Some(format!("LQI loop crashed: {}", panic_msg(&e))),
                }
            }
            (Some(bus), Run::Manual { t, d_meas }) => {
                let started = Instant::now();
                for (i, addr) in (1..=4u8).enumerate() {
                    if let Some(v) = mks_ops::read_encoder(bus.as_mut(), addr)
                        && v.is_finite()
                    {
                        d_meas[i] = v;
                    }
                    // else: hold the previous value for this axis, don't
                    // zero it out over one bad reply.
                }
                *t += MANUAL_POLL_T.as_secs_f64();
                let _ = tx.send(Telemetry {
                    run_state: RunState::Manual,
                    connected: true,
                    is_sample: true,
                    manual_read: None,
                    t: *t,
                    d_meas: *d_meas,
                    status: "Manual live read".to_string(),
                    ..Default::default()
                });
                sleep_remaining(started, MANUAL_POLL_T);
            }
            _ => {
                // Idle or disconnected: block on the next command instead of
                // busy-polling.
                match rx.recv() {
                    Ok(cmd) => handle_command(
                        cmd,
                        &mut bus,
                        &mut run,
                        &mut weights,
                        &mut mass_solver,
                        &mut xy_pre_balance,
                        &tx,
                    ),
                    Err(_) => return, // UI side dropped -> shut down
                }
            }
        }

        if let Some(status) = crashed {
            run = Run::Idle;
            let _ = tx.send(Telemetry {
                run_state: RunState::Idle,
                connected: true,
                status,
                ..Default::default()
            });
        }
    }
}

fn run_state(run: &Run) -> RunState {
    match run {
        Run::Idle => RunState::Idle,
        Run::Manual { .. } => RunState::Manual,
        Run::Mpc { .. } => RunState::Mpc,
        Run::Lqi { .. } => RunState::Lqi,
    }
}

fn panic_msg(e: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn sleep_remaining(started: Instant, plant_t: Duration) {
    let elapsed = started.elapsed();
    if elapsed < plant_t {
        std::thread::sleep(plant_t - elapsed);
    }
    // else: overran the cadence — proceed immediately to the next cycle
    // rather than replicating MATLAB's same-index-reuse `continue` quirk;
    // the control action already went out either way, only the logged
    // cadence timestamp would differ.
}

fn handle_command(
    cmd: UiCommand,
    bus: &mut Option<Box<dyn Bus>>,
    run: &mut Run,
    weights: &mut TuneWeights,
    mass_solver: &mut MassSolver,
    xy_pre_balance: &mut bool,
    tx: &Sender<Telemetry>,
) {
    match cmd {
        UiCommand::Connect(port) => {
            // `Connect`/`Disconnect` don't otherwise produce telemetry (no
            // loop is running yet), so the UI would never learn whether this
            // actually succeeded — send a standalone snapshot either way.
            //
            // Timeout is 300ms, not MATLAB's original 150ms: the real rig
            // isn't a direct wired RS485 adapter, it's PC --USB-- RT4AE01
            // (local) tunneled to a second RT4AE01 (remote, on the rig's
            // RS485 bus) — that extra hop adds latency the original 150ms
            // budget didn't need to cover.
            match SerialBus::open(&port, 115_200, Duration::from_millis(300)) {
                Ok(b) => {
                    *bus = Some(Box::new(b));
                    let _ = tx.send(Telemetry {
                        run_state: run_state(run),
                        connected: true,
                        status: format!("Connected ({port})"),
                        ..Default::default()
                    });
                }
                Err(e) => {
                    *bus = None;
                    let _ = tx.send(Telemetry {
                        run_state: run_state(run),
                        connected: false,
                        status: format!("Connect failed: {e}"),
                        ..Default::default()
                    });
                }
            }
        }
        UiCommand::Disconnect => {
            *bus = None;
            *run = Run::Idle;
            let _ = tx.send(Telemetry {
                run_state: RunState::Idle,
                connected: false,
                status: "Disconnected".to_string(),
                ..Default::default()
            });
        }
        UiCommand::RefreshPorts => {
            // Port list is UI-thread-visible state; the caller is expected
            // to poll `SerialBus::list_ports()` directly for the dropdown
            // (no serial I/O involved, doesn't need to go through `Bus`).
        }
        UiCommand::Home => {
            if let Some(bus) = bus {
                for addr in 1..=4u8 {
                    mks_ops::home(bus.as_mut(), addr);
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
        UiCommand::ReadEncoder(addr) => {
            // Same problem as `Connect`: no loop is running yet, so nothing
            // else would ever report this result to the UI — send a
            // standalone snapshot with the reading in `status`.
            //
            // Bypasses `mks_ops::read_encoder` (not just calling it) so a
            // failed parse can still report the raw bytes that came back —
            // 0 bytes means the port timed out with nothing at all (wrong
            // addr/baud/wiring), N bytes means something answered but didn't
            // parse (bridge noise, wrong device on that addr, etc). Without
            // this, "no/bad reply" is not enough to tell those apart.
            if let Some(bus) = bus {
                bus.flush_input();
                bus.write_bytes(&protocol::mks::build_read_encoder(addr));
                let reply = bus.read_up_to(10);
                let parsed = protocol::mks::parse_encoder_reply(addr, &reply);
                let status = match parsed {
                    Some(v) => format!("Encoder[{addr}] = {v:.4} m"),
                    None if reply.is_empty() => {
                        format!("Encoder[{addr}]: timeout, 0 bytes back (check addr/baud/wiring)")
                    }
                    None => format!(
                        "Encoder[{addr}]: {} bytes, unparsable: {reply:02X?}",
                        reply.len()
                    ),
                };
                let _ = tx.send(Telemetry {
                    run_state: run_state(run),
                    connected: true,
                    manual_read: parsed.map(|v| (addr, v)),
                    status,
                    ..Default::default()
                });
            }
        }
        UiCommand::JogAbsolute {
            addr,
            dist_m,
            spd,
            acc,
        } => {
            if let Some(bus) = bus {
                mks_ops::run_abs(bus.as_mut(), addr, dist_m, spd, acc);
                // Bus settle time — same as `Home`'s inter-axis pause below.
                // A single manual click never needed this (naturally spaced
                // by human click timing), but the "apply to all 4 axes"
                // buttons fire 4 of these back-to-back through this same
                // handler with zero gap otherwise, and some axes drop the
                // command without it.
                std::thread::sleep(Duration::from_millis(120));
            }
        }
        UiCommand::JogRelative {
            addr,
            dist_m,
            spd,
            acc,
        } => {
            if let Some(bus) = bus {
                mks_ops::run_rel(bus.as_mut(), addr, dist_m, spd, acc);
                std::thread::sleep(Duration::from_millis(120));
            }
        }
        UiCommand::SetController(on) => match run {
            Run::Mpc { loop_ } => loop_.controller_on = on,
            Run::Lqi { loop_ } => loop_.controller_on = on,
            Run::Idle | Run::Manual { .. } => {}
        },
        UiCommand::SetSetpoint {
            roll_deg,
            pitch_deg,
        } => match run {
            Run::Mpc { loop_ } => loop_.setpoint_deg = [roll_deg, pitch_deg],
            Run::Lqi { loop_ } => loop_.setpoint_deg = [roll_deg, pitch_deg],
            Run::Idle | Run::Manual { .. } => {}
        },
        UiCommand::SetXyPreBalance(v) => {
            // Consulted at `StartAuto`, not while a loop is already running
            // (matches the live app: the checkbox only takes effect on the
            // next `AutoButtonPushed`).
            *xy_pre_balance = v;
        }
        UiCommand::SetMassSolver(solver) => {
            *mass_solver = solver;
            if let Run::Lqi { loop_ } = run {
                loop_.mass_solver = solver;
            }
        }
        UiCommand::UpdateWeights(w) => {
            *weights = w;
            // Motor speed/accel apply to the *running* loop — MATLAB only
            // reads its spinners once at loop entry, but `Tune Now` already
            // hot-reloads the MPC weights mid-run, so carrying these across
            // too costs nothing and avoids a Stop/Start just to speed up.
            match run {
                Run::Mpc { loop_ } => {
                    loop_.spd = w.jog_spd;
                    loop_.acc = w.jog_acc;
                }
                Run::Lqi { loop_ } => {
                    loop_.spd = w.jog_spd;
                    loop_.acc = w.jog_acc;
                }
                _ => {}
            }
        }
        UiCommand::StartAuto(mode) => {
            if let Some(bus) = bus {
                // `init()` blocks the IO thread for up to its poll timeout
                // (encoder-priming poll) — tell the UI it's happening, since
                // otherwise it just looks frozen for that whole stretch.
                // Reports the *target* state, not `run_state(run)` (which is
                // still whatever ran before Start Auto — typically `Idle`).
                // The UI treats an `Idle` snapshot as "Start Auto didn't
                // happen, clear the optimistic running flag" (see fix 1's
                // rationale in the app crate); reporting the stale pre-init
                // state here would trip that the instant this snapshot
                // arrives, flickering "running" off during the several
                // seconds `init`'s encoder-priming poll actually takes.
                let _ = tx.send(Telemetry {
                    run_state: match mode {
                        RunMode::Mpc => RunState::Mpc,
                        RunMode::Lqi => RunState::Lqi,
                    },
                    connected: true,
                    status: "Initializing...".to_string(),
                    ..Default::default()
                });
                match mode {
                    RunMode::Mpc => {
                        match MpcLoop::init(bus.as_mut(), *xy_pre_balance, &(*weights).into()) {
                            Some(mut loop_) => {
                                // `From<TuneWeights>` deliberately drops the
                                // jog fields (not control-math weights), so
                                // they have to be carried across by hand.
                                loop_.spd = weights.jog_spd;
                                loop_.acc = weights.jog_acc;
                                *run = Run::Mpc { loop_ };
                            }
                            None => {
                                *run = Run::Idle;
                                let _ = tx.send(Telemetry {
                                    run_state: RunState::Idle,
                                    connected: true,
                                    status: "MPC init timed out: jog all 4 axes above ~10mm \
                                             first, then retry Start Auto"
                                        .to_string(),
                                    ..Default::default()
                                });
                            }
                        }
                    }
                    RunMode::Lqi => match LqiLoop::init(bus.as_mut(), &(*weights).into()) {
                        Some(mut l) => {
                            l.mass_solver = *mass_solver;
                            l.spd = weights.jog_spd;
                            l.acc = weights.jog_acc;
                            *run = Run::Lqi { loop_: l };
                        }
                        None => {
                            *run = Run::Idle;
                            let _ = tx.send(Telemetry {
                                run_state: RunState::Idle,
                                connected: true,
                                status: "LQI init failed: jog all 4 axes above ~50mm first, \
                                         or check the tuning weights (R = 0 or an all-zero \
                                         Q makes the gain solve singular)"
                                    .to_string(),
                                ..Default::default()
                            });
                        }
                    },
                };
            }
        }
        UiCommand::Stop => *run = Run::Idle,
        UiCommand::TuneNow => match run {
            Run::Mpc { loop_ } => loop_.retune(&(*weights).into()),
            // Re-solving the LQI gains means a fresh Riccati recursion right
            // here, between control cycles — a few ms, one cadence boundary
            // at worst, and cheaper than the Stop/Start it replaces.
            Run::Lqi { loop_ } => {
                if !loop_.retune(&(*weights).into()) {
                    let _ = tx.send(Telemetry {
                        run_state: RunState::Lqi, // still running, just kept the old gains
                        connected: true,
                        status: "Tune Now: LQI gain solve failed, keeping previous gains \
                                 (check R and Q — R = 0 is singular)"
                            .to_string(),
                        ..Default::default()
                    });
                }
            }
            _ => {}
        },
        UiCommand::SetManualPoll(on) => {
            if on {
                if matches!(run, Run::Idle) {
                    *run = Run::Manual {
                        t: 0.0,
                        d_meas: [0.0; 4],
                    };
                }
            } else if matches!(run, Run::Manual { .. }) {
                *run = Run::Idle;
            }
        }
    }
}
