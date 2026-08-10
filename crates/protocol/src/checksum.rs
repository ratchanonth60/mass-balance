//! Checksum/CRC helpers shared by both wire protocols on the bus.

/// MKS-driver checksum-8: sum of all preceding bytes, mod 256.
/// Matches every live class-method encoder (`MotorCommand`, `mksRunAbsAxis`,
/// `mksRunRelAxis`, `ReadError`, `ReadRPM`, `isMotorMoving`): `mod(sum(bytes), 256)`.
pub fn sum8(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |acc, &b| acc.wrapping_add(b))
}

/// CRC16/MODBUS: poly 0xA001 (reflected), init 0xFFFF, LSB-first byte processing.
/// Matches `IMUCRCCheck`/`ReadAngles.m`'s from-scratch CRC16 implementation.
pub fn crc16_modbus(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in bytes {
        crc ^= b as u16;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum8_wraps() {
        assert_eq!(sum8(&[0xFA, 0x01, 0x30]), 0x2B);
        assert_eq!(sum8(&[0xFF, 0xFF]), 0xFE);
    }

    #[test]
    fn crc16_matches_read_angle_command() {
        // CommandReadAngle = [0x50 0x03 0x00 0x3D 0x00 0x06 0x59 0x85]
        // trailing 2 bytes are CRC16 low,high.
        let crc = crc16_modbus(&[0x50, 0x03, 0x00, 0x3D, 0x00, 0x06]);
        assert_eq!(crc.to_le_bytes(), [0x59, 0x85]);
    }
}
