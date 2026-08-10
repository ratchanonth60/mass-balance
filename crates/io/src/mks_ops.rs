//! I/O-performing wrappers around `protocol::mks`/`protocol::imu`'s pure
//! encode/decode functions — the request/reply lengths here match the live
//! `app.*` class methods exactly (`app.ReadEncoder` reads 10 bytes,
//! `app.ReadError` 8, `app.ReadRPM` 6, `app.isMotorMoving` 5,
//! `readHWT9053.m`'s combined read 35).

use crate::transport::Bus;
use protocol::{imu, mks};

// Every function below flushes stale input before writing its request. Not
// in the original MATLAB (which only flushes before the IMU read) — added
// because `home`/`run_abs`/`run_rel` are fire-and-forget here but the MKS
// driver answers them anyway; an unread ack from an earlier manual command
// sits in the buffer and gets misread as the reply to the next unrelated
// request otherwise (seen in practice: a `ReadEncoder` right after `Home`
// picked up 2 queued home-acks instead of the real encoder reply).

pub fn read_encoder(bus: &mut dyn Bus, addr: u8) -> Option<f64> {
    bus.flush_input();
    bus.write_bytes(&mks::build_read_encoder(addr));
    let reply = bus.read_up_to(10);
    mks::parse_encoder_reply(addr, &reply)
}

pub fn read_error(bus: &mut dyn Bus, addr: u8) -> Option<f64> {
    bus.flush_input();
    bus.write_bytes(&mks::build_read_error(addr));
    let reply = bus.read_up_to(8);
    mks::parse_error_reply(addr, &reply)
}

pub fn read_rpm(bus: &mut dyn Bus, addr: u8) -> Option<f64> {
    bus.flush_input();
    bus.write_bytes(&mks::build_read_rpm(addr));
    let reply = bus.read_up_to(6);
    mks::parse_rpm_reply(addr, &reply)
}

pub fn is_moving(bus: &mut dyn Bus, addr: u8) -> Option<bool> {
    bus.flush_input();
    bus.write_bytes(&mks::build_is_moving(addr));
    let reply = bus.read_up_to(5);
    mks::parse_is_moving_reply(&reply)
}

/// `app.HomeButtonPushed` / `app.MotorCommand(motornum,"Home")`: fire-and-forget,
/// no reply read.
pub fn home(bus: &mut dyn Bus, addr: u8) {
    bus.write_bytes(&mks::build_home(addr));
}

/// `app.mksRunAbsAxis`: fire-and-forget.
pub fn run_abs(bus: &mut dyn Bus, addr: u8, dist_m: f64, spd_rpm: f64, acc: u8) {
    bus.write_bytes(&mks::build_run_abs(addr, dist_m, spd_rpm, acc));
}

/// `app.mksRunRelAxis`: fire-and-forget.
pub fn run_rel(bus: &mut dyn Bus, addr: u8, dist_m: f64, spd_rpm: f64, acc: u8) {
    bus.write_bytes(&mks::build_run_rel(addr, dist_m, spd_rpm, acc));
}

/// `readHWT9053.m`: flush, send the combined-read command, read 35 bytes.
/// Caller is responsible for the `flush(app.SerialPortCon)` that precedes
/// this in the live loop (it happens once per control-loop iteration, not
/// once per call, in `MainConstantTs11.m`).
pub fn read_imu(bus: &mut dyn Bus) -> Option<imu::ImuFrame> {
    bus.write_bytes(&imu::CMD_READ_ACC_GYRO_MAG_ANGLE);
    let reply = bus.read_up_to(35);
    imu::parse_combined_reply(&reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockBus;

    #[test]
    fn read_encoder_sends_correct_frame_and_parses_reply() {
        let mut bus = MockBus::default();
        let mut reply = [0u8; 10];
        reply[0] = 0xFB;
        reply[1] = 1;
        reply[2] = 0x30;
        reply[9] = protocol::checksum::sum8(&reply[..9]);
        bus.push_reply(reply.to_vec());

        let result = read_encoder(&mut bus, 1);
        assert_eq!(bus.writes[0], vec![0xFA, 1, 0x30, 0x2B]);
        assert!(result.is_some());
    }

    #[test]
    fn read_imu_short_reply_returns_none() {
        let mut bus = MockBus::default();
        bus.push_reply(vec![0x50; 10]); // too short
        assert!(read_imu(&mut bus).is_none());
    }
}
