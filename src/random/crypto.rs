use std::fs::File;
use std::io::Read;

use crate::random::Rng;

/// Cryptographically secure RNG sourcing randomness from `/dev/urandom`.
pub struct CryptoRng {
    file: File,
}

impl CryptoRng {
    /// Open `/dev/urandom` and create a new generator.
    pub fn new() -> Self {
        let file = File::open("/dev/urandom").expect("failed to open /dev/urandom");
        Self { file }
    }
}

impl Rng for CryptoRng {
    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.file
            .read_exact(&mut buf)
            .expect("failed reading from /dev/urandom");
        u64::from_ne_bytes(buf)
    }
}

/// Convenience constructor for [`CryptoRng`].
pub fn crypto_rng() -> CryptoRng {
    CryptoRng::new()
}
