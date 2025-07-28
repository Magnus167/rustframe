use std::time::{SystemTime, UNIX_EPOCH};

use crate::random::Rng;

/// Simple XorShift64-based pseudo random number generator.
#[derive(Clone)]
pub struct Prng {
    state: u64,
}

impl Prng {
    /// Create a new generator from the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Create a generator seeded from the current time.
    pub fn from_entropy() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self::new(nanos)
    }
}

impl Rng for Prng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Convenience constructor using system entropy.
pub fn rng() -> Prng {
    Prng::from_entropy()
}
