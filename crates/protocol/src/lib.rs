//! Byte-exact wire framing for the AutoMass rig's shared RS485 bus
//! (115200 baud, one physical port, one writer at a time — enforced by the
//! `io` crate, not here). Pure encode/decode functions, no I/O.
//!
//! See `DISCREPANCIES.md` for every place this intentionally preserves a
//! MATLAB quirk (buggy validation, asymmetric branches, etc.) instead of
//! "fixing" it — the physical rig runs on that exact behavior today.

pub mod checksum;
pub mod imu;
pub mod mks;
