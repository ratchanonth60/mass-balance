//! Control math for the AutoMass rig port: plant linearization, Kalman
//! filter, discrete LQR (DARE), and constrained batch MPC.
//!
//! No shared `geometry` module across [`mpc_model`] and [`lqi_model`] —
//! `ModelInit.m` and `InitLQR.m` use materially different mass/geometry
//! constants for the same physical rig, and the live app never reconciles
//! them (see `lqi_model`'s module docs). Unifying them would silently change
//! controller behavior relative to what ships today.

pub mod allocation;
pub mod dare;
pub mod kf;
pub mod lqi_model;
pub mod mpc;
pub mod mpc_model;
