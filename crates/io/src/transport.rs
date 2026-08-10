//! Serial bus abstraction. One physical RS485 port shared by the IMU and all
//! 4 MKS drivers (`app.SerialPortCon` in the live app) — `Bus` is owned
//! exclusively by the IO thread (see `crate::thread`), so there is never a
//! second writer to interleave frames with.

use std::io::{Read, Write};
use std::time::Duration;

pub trait Bus {
    fn write_bytes(&mut self, bytes: &[u8]);
    /// Reads up to `n` bytes, returning early (possibly with fewer than `n`
    /// bytes, even zero) once the read times out — matches MATLAB's
    /// `read(port, n, 'uint8')`, which returns whatever arrived within the
    /// port's configured timeout rather than erroring.
    fn read_up_to(&mut self, n: usize) -> Vec<u8>;
    /// Discards any buffered input (`flush(app.SerialPortCon)`).
    fn flush_input(&mut self);
}

/// Real serial transport: `serialport::new(path, 115200).timeout(...)`.
/// MATLAB used `serialport(port, 115200, "Timeout", 0.15)` for a direct
/// wired RS485 adapter; the actual rig tunnels through a pair of RT4AE01
/// bridges (see `crate::thread`'s `Connect` handler), so the caller opens
/// with a longer timeout to cover that extra hop.
pub struct SerialBus {
    port: Box<dyn serialport::SerialPort>,
}

impl SerialBus {
    pub fn open(path: &str, baud: u32, timeout: Duration) -> Result<Self, serialport::Error> {
        let port = serialport::new(path, baud).timeout(timeout).open()?;
        Ok(Self { port })
    }

    pub fn list_ports() -> Vec<String> {
        serialport::available_ports()
            .map(|ports| ports.into_iter().map(|p| p.port_name).collect())
            .unwrap_or_default()
    }
}

impl Bus for SerialBus {
    fn write_bytes(&mut self, bytes: &[u8]) {
        let _ = self.port.write_all(bytes);
    }

    fn read_up_to(&mut self, n: usize) -> Vec<u8> {
        let mut buf = vec![0u8; n];
        let mut got = 0;
        while got < n {
            match self.port.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(_) => break, // timeout or I/O error: return what we have
            }
        }
        buf.truncate(got);
        buf
    }

    fn flush_input(&mut self) {
        let _ = self.port.clear(serialport::ClearBuffer::Input);
    }
}

/// In-memory mock for testing loop logic without hardware: a queue of
/// canned replies (one `Vec<u8>` per expected `read_up_to` call) and a log
/// of every write, so tests can assert exact frame bytes were sent.
#[derive(Default)]
pub struct MockBus {
    pub writes: Vec<Vec<u8>>,
    pub replies: std::collections::VecDeque<Vec<u8>>,
}

impl MockBus {
    pub fn push_reply(&mut self, bytes: Vec<u8>) {
        self.replies.push_back(bytes);
    }
}

impl Bus for MockBus {
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.writes.push(bytes.to_vec());
    }

    fn read_up_to(&mut self, n: usize) -> Vec<u8> {
        let mut reply = self.replies.pop_front().unwrap_or_default();
        reply.truncate(n);
        reply
    }

    fn flush_input(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_bus_records_writes_and_serves_replies() {
        let mut bus = MockBus::default();
        bus.push_reply(vec![0xFB, 1, 0x30]);
        bus.write_bytes(&[0xFA, 1, 0x30, 0x2B]);
        assert_eq!(bus.writes[0], vec![0xFA, 1, 0x30, 0x2B]);
        assert_eq!(bus.read_up_to(10), vec![0xFB, 1, 0x30]);
    }
}
