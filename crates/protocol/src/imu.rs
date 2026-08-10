//! WitMotion IMU framing (WT901C485/JY901-family attitude+inertial sensor at
//! Modbus slave `0x50`, HWT101CL single-axis gyro at slave `0x60`), Modbus RTU
//! function `0x03` (read holding registers) / `0x06` (write single register).
//! All command frames are fixed byte arrays baked into the live app (never
//! computed at runtime) — reproduced verbatim here, and verified against
//! `crc16_modbus` in tests.

use crate::checksum::crc16_modbus;

/// Combined acc+gyro+mag+angle read, slave 0x50, registers 0x34..0x34+0x0F.
/// This is the command `readHWT9053.m`/`app`'s main control-loop IMU read uses
/// (35-byte reply).
pub const CMD_READ_ACC_GYRO_MAG_ANGLE: [u8; 8] = [0x50, 0x03, 0x00, 0x34, 0x00, 0x0F, 0x49, 0x81];
/// Angle-only read (6 registers @ 0x3D), slave 0x50 — `app.CommandReadAngle`.
pub const CMD_READ_ANGLE: [u8; 8] = [0x50, 0x03, 0x00, 0x3D, 0x00, 0x06, 0x59, 0x85];
/// Yaw-only read, HWT101CL slave 0x60 — `app.CommandReadYaw`.
///
/// NB: this constant's trailing CRC16 bytes do **not** match a CRC16/Modbus
/// computation over the preceding 6 bytes (unlike every other command in this
/// module) — a stale/copy-paste bug in the original MATLAB source, not a
/// transcription error here (see `DISCREPANCIES.md`). Not used by the live
/// control loop; kept byte-exact anyway.
pub const CMD_READ_YAW: [u8; 8] = [0x60, 0x03, 0x00, 0x3D, 0x00, 0x06, 0xC3, 0x06];
/// Yaw angular velocity, HWT101CL slave 0x60 — `app.CommandReadYawVelo`.
/// Same stale-CRC caveat as [`CMD_READ_YAW`].
pub const CMD_READ_YAW_VELO: [u8; 8] = [0x60, 0x03, 0x00, 0x37, 0x00, 0x03, 0xC5, 0x45];
/// Acceleration-only read, slave 0x50 — `app.CommandReadACC`.
pub const CMD_READ_ACC: [u8; 8] = [0x50, 0x03, 0x00, 0x34, 0x00, 0x03, 0x49, 0x84];
/// Angular-velocity-only read, slave 0x50 — `app.CommandReadVelo`.
pub const CMD_READ_VELO: [u8; 8] = [0x50, 0x03, 0x00, 0x37, 0x00, 0x03, 0xB9, 0x84];
/// Full-block dump, HWT101CL slave 0x60 — `app.CollectAll`.
pub const CMD_READ_ALL_HWT101: [u8; 8] = [0x60, 0x03, 0x00, 0x30, 0x00, 0x30, 0x4D, 0xA0];

/// Modbus function 0x06 "write single register" calibration/config commands.
pub const CMD_UNLOCK: [u8; 8] = [0x50, 0x06, 0x00, 0x69, 0xB5, 0x88, 0x22, 0xA1];
pub const CMD_CALIBRATE_XY: [u8; 8] = [0x50, 0x06, 0x00, 0x01, 0x00, 0x08, 0xD4, 0x4D];
pub const CMD_CALIBRATE_Z: [u8; 8] = [0x50, 0x06, 0x00, 0x01, 0x00, 0x04, 0xD4, 0x48];
pub const CMD_SAVE: [u8; 8] = [0x50, 0x06, 0x00, 0x00, 0x00, 0x00, 0x84, 0x4B];
pub const CMD_MOD_DELAY: [u8; 8] = [0x50, 0x06, 0x00, 0x20, 0x00, 0x08, 0x84, 0x47];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuFrame {
    pub acc_g: [f64; 3],
    pub gyro_dps: [f64; 3],
    pub angle_deg: [f64; 3],
    pub mag_raw: [f64; 3],
}

/// 3x int16, big-endian, starting at `start` (0-based byte offset).
fn read_int16_triplet(bytes: &[u8], start: usize) -> [i16; 3] {
    core::array::from_fn(|i| i16::from_be_bytes([bytes[start + 2 * i], bytes[start + 2 * i + 1]]))
}

/// 3x int32 with the device's non-standard word order: value =
/// `(hi(reg1)<<8 | lo(reg1)) | (hi(reg2)<<24 | lo(reg2)<<16)` — i.e. the
/// *second* 16-bit register holds the high word. Matches `charToInt`/`HexToAng`.
fn read_int32_triplet_swapped(bytes: &[u8], start: usize) -> [i32; 3] {
    core::array::from_fn(|i| {
        let base = start + 4 * i;
        let b0 = bytes[base] as u32; // hi byte of reg1, <<8
        let b1 = bytes[base + 1] as u32; // lo byte of reg1
        let b2 = bytes[base + 2] as u32; // hi byte of reg2, <<24
        let b3 = bytes[base + 3] as u32; // lo byte of reg2, <<16
        ((b0 << 8) | b1 | (b2 << 24) | (b3 << 16)) as i32
    })
}

/// Parses the 35-byte reply to [`CMD_READ_ACC_GYRO_MAG_ANGLE`].
///
/// Matches `readHWT9053.m` exactly: only checks `bytes[0] == 0x50` and the
/// expected length — **no CRC16 verification on receive** (the command's CRC
/// is a pre-baked constant, and the reply's CRC is never checked in the live
/// code). See `DISCREPANCIES.md`.
pub fn parse_combined_reply(bytes: &[u8]) -> Option<ImuFrame> {
    if bytes.len() != 35 || bytes[0] != 0x50 {
        return None;
    }
    let acc = read_int16_triplet(bytes, 3); // MATLAB start index 4 (1-based) -> 3 (0-based)
    let gyro = read_int16_triplet(bytes, 9);
    let mag = read_int16_triplet(bytes, 15);
    let angle = read_int32_triplet_swapped(bytes, 21);

    Some(ImuFrame {
        acc_g: acc.map(|v| v as f64 / 32768.0 * 16.0),
        gyro_dps: gyro.map(|v| v as f64 / 32768.0 * 2000.0),
        angle_deg: angle.map(|v| v as f64 / 1000.0),
        mag_raw: mag.map(|v| v as f64),
    })
}

/// Parses a 17-byte angle-only reply to [`CMD_READ_ANGLE`] (3x int32 @ offset
/// 3, same word-swap layout), verifying CRC16 unlike the combined-read path —
/// matches `ReadAngles.m`, the one live-protocol reader that actually checks it.
pub fn parse_angle_only_reply(addr: u8, bytes: &[u8]) -> Option<[f64; 3]> {
    if bytes.len() != 17 || bytes[0] != addr || bytes[1] != 0x03 {
        return None;
    }
    let crc = crc16_modbus(&bytes[..bytes.len() - 2]);
    if crc.to_le_bytes() != [bytes[bytes.len() - 2], bytes[bytes.len() - 1]] {
        return None;
    }
    let angle = read_int32_triplet_swapped(bytes, 3);
    Some(angle.map(|v| v as f64 / 1000.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_crc_matches(cmd: [u8; 8]) {
        let crc = crc16_modbus(&cmd[..6]);
        assert_eq!(crc.to_le_bytes(), [cmd[6], cmd[7]], "cmd={:02X?}", cmd);
    }

    #[test]
    fn most_fixed_commands_have_valid_crc16() {
        for cmd in [
            CMD_READ_ACC_GYRO_MAG_ANGLE,
            CMD_READ_ANGLE,
            CMD_READ_ACC,
            CMD_READ_VELO,
            CMD_READ_ALL_HWT101,
            CMD_UNLOCK,
            CMD_CALIBRATE_XY,
            CMD_CALIBRATE_Z,
            CMD_SAVE,
            CMD_MOD_DELAY,
        ] {
            assert_crc_matches(cmd);
        }
    }

    /// Documents (rather than hides) the stale-CRC bug in these two constants
    /// — see `DISCREPANCIES.md`. If this test starts failing because someone
    /// "fixed" the CRC bytes, that's a byte-exactness regression, not a bug fix.
    #[test]
    fn read_yaw_commands_have_known_stale_crc16() {
        assert_ne!(
            crc16_modbus(&CMD_READ_YAW[..6]).to_le_bytes(),
            [CMD_READ_YAW[6], CMD_READ_YAW[7]]
        );
        assert_ne!(
            crc16_modbus(&CMD_READ_YAW_VELO[..6]).to_le_bytes(),
            [CMD_READ_YAW_VELO[6], CMD_READ_YAW_VELO[7]]
        );
    }

    #[test]
    fn combined_reply_rejects_short_or_bad_header() {
        assert!(parse_combined_reply(&[0u8; 34]).is_none());
        let mut bytes = [0u8; 35];
        bytes[0] = 0x51;
        assert!(parse_combined_reply(&bytes).is_none());
    }

    #[test]
    fn combined_reply_decodes_angle_word_swap() {
        let mut bytes = [0u8; 35];
        bytes[0] = 0x50;
        // angle[0] raw = 1000 (=> 1.0 deg). Encode per the swapped layout:
        // val = (b0<<8|b1) | (b2<<24|b3<<16); want val=1000=0x3E8.
        // Put low word (0x03E8) in b0,b1 and high word (0) in b2,b3.
        bytes[21] = 0x03; // b0 hi
        bytes[22] = 0xE8; // b1 lo
        bytes[23] = 0x00; // b2
        bytes[24] = 0x00; // b3
        let frame = parse_combined_reply(&bytes).unwrap();
        assert!((frame.angle_deg[0] - 1.0).abs() < 1e-9);
    }
}
