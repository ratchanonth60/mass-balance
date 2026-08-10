//! MKS closed-loop stepper driver framing (4 axes, addresses 1-4 on a shared
//! RS485 bus, 115200 baud). Ported byte-exact from the *live* class-method
//! implementations inside `AutoMass_MPC.mlapp` (`app.ReadEncoder`,
//! `app.mksRunAbsAxis`, `app.mksRunRelAxis`, `app.ReadError`, `app.ReadRPM`,
//! `app.isMotorMoving`, `app.HomeButtonPushed` via `app.MotorCommand`) — see
//! `DISCREPANCIES.md` for where these diverge from the top-level `functions/*.m`
//! files of the same name, which are dead code not reachable from the live app.

use crate::checksum::sum8;

pub const TX_HEADER: u8 = 0xFA;
pub const RX_HEADER: u8 = 0xFB;

const FUNC_HOME: u8 = 0x91;
const FUNC_READ_ENCODER: u8 = 0x30;
const FUNC_READ_ERROR: u8 = 0x39;
const FUNC_READ_RPM: u8 = 0x32;
const FUNC_IS_MOVING: u8 = 0xF1;
const FUNC_RUN_ABS: u8 = 0xF5;
const FUNC_RUN_REL: u8 = 0xF4;

const PITCH_M_PER_REV: f64 = 0.004;
const TICKS_PER_REV: f64 = 16384.0; // 0x4000
const ENCODER_TICKS_TO_DEG: f64 = 180.0 / 16383.0; // interp1([16383,0],[180,0], .)
const ENCODER_TOTAL_ANGLE_TO_M: f64 = 4.0 / 1000.0 / 360.0;
const ERROR_TICKS_PER_DEG: f64 = 51200.0 / 360.0;

/// `[0xFA, addr, func, checksum]` — every live 4-byte MKS request uses this
/// shape (`app.MotorCommand`'s l==4 branch, and `ReadError`/`ReadRPM`/`isMotorMoving`
/// build the identical shape inline). Checksum is genuinely computed here in
/// the live code path (no literal-0x00 quirk survives into `app.*`).
fn build_simple(addr: u8, func: u8) -> [u8; 4] {
    let mut frame = [TX_HEADER, addr, func, 0];
    frame[3] = sum8(&frame[..3]);
    frame
}

pub fn build_home(addr: u8) -> [u8; 4] {
    build_simple(addr, FUNC_HOME)
}

pub fn build_read_encoder(addr: u8) -> [u8; 4] {
    build_simple(addr, FUNC_READ_ENCODER)
}

pub fn build_read_error(addr: u8) -> [u8; 4] {
    build_simple(addr, FUNC_READ_ERROR)
}

pub fn build_read_rpm(addr: u8) -> [u8; 4] {
    build_simple(addr, FUNC_READ_RPM)
}

pub fn build_is_moving(addr: u8) -> [u8; 4] {
    build_simple(addr, FUNC_IS_MOVING)
}

/// Shared by `build_run_abs`/`build_run_rel`. Replicates, byte-exact:
/// - unconditional `d = -d` (the `addr != 3`/`addr == 3` guards are commented
///   out in both live methods, so every axis gets negated),
/// - `spd = clamp(0, spd*16, 3000)` (multiply-then-clamp; this is the ordering
///   used by the live `app.mksRunAbsAxis`/`app.mksRunRelAxis`, giving a usable
///   caller range of ~0-187.5 RPM before saturation).
fn build_run(func: u8, addr: u8, dist_m: f64, spd_rpm: f64, acc: u8) -> [u8; 11] {
    let d = -dist_m;
    let spd = (spd_rpm * 16.0).clamp(0.0, 3000.0).round() as u16;
    let axis_ticks = (d / PITCH_M_PER_REV * TICKS_PER_REV).round() as i32;
    let ticks_be = (axis_ticks as u32).to_be_bytes();
    let spd_be = spd.to_be_bytes();

    let mut frame = [0u8; 11];
    frame[0] = TX_HEADER;
    frame[1] = addr;
    frame[2] = func;
    frame[3] = spd_be[0];
    frame[4] = spd_be[1];
    frame[5] = acc;
    frame[6..10].copy_from_slice(&ticks_be);
    frame[10] = sum8(&frame[..10]);
    frame
}

/// Absolute-position move (position mode 4, function `0xF5`).
pub fn build_run_abs(addr: u8, dist_m: f64, spd_rpm: f64, acc: u8) -> [u8; 11] {
    build_run(FUNC_RUN_ABS, addr, dist_m, spd_rpm, acc)
}

/// Relative-position move, function `0xF4`.
pub fn build_run_rel(addr: u8, dist_m: f64, spd_rpm: f64, acc: u8) -> [u8; 11] {
    build_run(FUNC_RUN_REL, addr, dist_m, spd_rpm, acc)
}

/// Decodes a 10-byte encoder reply into linear position (metres).
///
/// Replicates `app.MotorCRCCheck` + `app.ReadEncoder` byte-exact, including a
/// genuine bug in the live code: the header/addr/func check only rejects the
/// reply when **all three** fields (`byte0!=0xFB && byte1!=addr && byte2!=0x30`)
/// are simultaneously wrong, so a reply with any one field correct is accepted
/// even if the other two are garbage. See `DISCREPANCIES.md`.
pub fn parse_encoder_reply(addr: u8, bytes: &[u8]) -> Option<f64> {
    if bytes.len() != 10 {
        return None;
    }
    let header_ok = bytes[0] == RX_HEADER || bytes[1] == addr || bytes[2] == FUNC_READ_ENCODER;
    if !header_ok {
        return None;
    }
    if sum8(&bytes[..9]) != bytes[9] {
        return None;
    }

    let carry = u32::from_be_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    // MotorNum==5 is a special-cased alternate firmware; this rig only ever
    // addresses 1-4, so the general branch always applies.
    let rotation = if carry > 0 {
        4_294_967_295.0 - carry as f64
    } else {
        0.0
    };

    let ticks14 = u16::from_be_bytes([bytes[7], bytes[8]]) as f64;
    let degree = ticks14 * ENCODER_TICKS_TO_DEG;

    // NB: the `else` branch does *not* multiply `rotation` by 360 — replicated
    // as-is, this asymmetry is in the live MATLAB.
    let total_angle = if degree > 0.0 && degree < 360.0 {
        rotation * 360.0 + degree
    } else {
        rotation
    };

    Some(total_angle * ENCODER_TOTAL_ANGLE_TO_M)
}

/// Decodes an 8-byte position-error reply (function `0x39`) into degrees.
/// Strict header/addr/func/checksum validation (unlike the encoder path).
pub fn parse_error_reply(addr: u8, bytes: &[u8]) -> Option<f64> {
    if bytes.len() != 8 || bytes[0] != RX_HEADER || bytes[1] != addr || bytes[2] != FUNC_READ_ERROR
    {
        return None;
    }
    if sum8(&bytes[..7]) != bytes[7] {
        return None;
    }
    // MATLAB reverses to Rx(7:-1:4) then typecasts little-endian on a
    // little-endian host — net effect is a plain big-endian read of bytes[3..7).
    let raw = i32::from_be_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    Some(raw as f64 / ERROR_TICKS_PER_DEG)
}

/// Decodes a 6-byte speed reply (function `0x32`) into raw RPM (no scaling —
/// the live code disp `RPM = double(RPMhex)` directly).
pub fn parse_rpm_reply(addr: u8, bytes: &[u8]) -> Option<f64> {
    if bytes.len() != 6 || bytes[0] != RX_HEADER || bytes[1] != addr || bytes[2] != FUNC_READ_RPM {
        return None;
    }
    if sum8(&bytes[..5]) != bytes[5] {
        return None;
    }
    let raw = i16::from_be_bytes([bytes[3], bytes[4]]);
    Some(raw as f64)
}

/// Decodes a moving-status reply (function `0xF1`). Deliberately lenient,
/// matching `app.isMotorMoving` exactly: only requires >=4 bytes, header byte,
/// and function-code byte — no address check, no checksum check.
pub fn parse_is_moving_reply(bytes: &[u8]) -> Option<bool> {
    if bytes.len() < 4 || bytes[0] != RX_HEADER || bytes[2] != FUNC_IS_MOVING {
        return None;
    }
    Some(bytes[3] != 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_frame_bytes() {
        // addr=1: FA 01 91 checksum(0xFA+0x01+0x91=0x18C -> 0x8C)
        assert_eq!(build_home(1), [0xFA, 0x01, 0x91, 0x8C]);
    }

    #[test]
    fn read_encoder_frame_bytes() {
        assert_eq!(build_read_encoder(1), [0xFA, 0x01, 0x30, 0x2B]);
    }

    #[test]
    fn run_abs_negates_and_scales() {
        // 0.01 m -> -0.01 m -> ticks = -0.01/0.004*16384 = -40960 = 0xFFFF_600 0? check sign
        let frame = build_run_abs(1, 0.01, 100.0, 50);
        assert_eq!(frame[0], 0xFA);
        assert_eq!(frame[1], 1);
        assert_eq!(frame[2], 0xF5);
        // spd = clamp(100*16, 0, 3000) = 1600 = 0x0640
        assert_eq!(&frame[3..5], &[0x06, 0x40]);
        assert_eq!(frame[5], 50);
        let ticks = i32::from_be_bytes([frame[6], frame[7], frame[8], frame[9]]);
        assert_eq!(ticks, -40960); // d negated: -0.01 m / 0.004 * 16384
        assert_eq!(frame[10], sum8(&frame[..10]));
    }

    #[test]
    fn run_rel_uses_f4() {
        let frame = build_run_rel(2, -0.005, 500.0, 100);
        assert_eq!(frame[2], 0xF4);
        // spd = clamp(500*16=8000, 0, 3000) = 3000 = 0x0BB8
        assert_eq!(&frame[3..5], &[0x0B, 0xB8]);
    }

    #[test]
    fn encoder_reply_roundtrip() {
        // carry=0 (no turns yet), ticks14 = 8000 -> degree = 8000*180/16383 ~= 87.9deg (>0,<360)
        // rotation stays 0 (carry==0 branch), total_angle = 0*360 + degree = degree
        let mut bytes = [0u8; 10];
        bytes[0] = 0xFB;
        bytes[1] = 1;
        bytes[2] = 0x30;
        bytes[3..7].copy_from_slice(&0u32.to_be_bytes());
        bytes[7..9].copy_from_slice(&8000u16.to_be_bytes());
        bytes[9] = sum8(&bytes[..9]);
        let m = parse_encoder_reply(1, &bytes).unwrap();
        let expected_degree = 8000.0 * ENCODER_TICKS_TO_DEG;
        assert!((m - expected_degree * ENCODER_TOTAL_ANGLE_TO_M).abs() < 1e-9);
    }

    #[test]
    fn encoder_reply_rejects_bad_checksum() {
        let mut bytes = [0u8; 10];
        bytes[0] = 0xFB;
        bytes[1] = 1;
        bytes[2] = 0x30;
        bytes[9] = 0xFF; // wrong checksum
        assert!(parse_encoder_reply(1, &bytes).is_none());
    }

    #[test]
    fn encoder_reply_lenient_header_bug_preserved() {
        // header byte wrong (not 0xFB) but addr matches -> still accepted per
        // the live buggy AND-of-not-equal check.
        let mut bytes = [0u8; 10];
        bytes[0] = 0x00; // wrong
        bytes[1] = 1; // correct addr -> saves it
        bytes[2] = 0x99; // wrong func
        bytes[9] = sum8(&bytes[..9]);
        assert!(parse_encoder_reply(1, &bytes).is_some());
    }

    #[test]
    fn is_moving_ignores_addr_and_checksum() {
        // addr intentionally not checked by the live code.
        let bytes = [0xFB, 0x99, 0xF1, 0x00, 0x00];
        assert_eq!(parse_is_moving_reply(&bytes), Some(true));
        let bytes_stopped = [0xFB, 0x99, 0xF1, 0x01, 0x00];
        assert_eq!(parse_is_moving_reply(&bytes_stopped), Some(false));
    }

    #[test]
    fn error_and_rpm_strict_validation() {
        let mut err = [0xFB, 1, 0x39, 0, 0, 0, 0, 0];
        err[7] = sum8(&err[..7]);
        assert!(parse_error_reply(1, &err).is_some());
        assert!(parse_error_reply(2, &err).is_none()); // wrong addr -> strict reject

        let mut rpm = [0xFB, 1, 0x32, 0, 100, 0];
        rpm[5] = sum8(&rpm[..5]);
        assert_eq!(parse_rpm_reply(1, &rpm), Some(100.0));
    }
}
