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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;
    use std::collections::HashSet;

    #[test]
    fn test_crypto_rng_nonzero() {
        let mut rng = CryptoRng::new();
        let mut all_same = true;
        let mut prev = rng.next_u64();
        for _ in 0..5 {
            let val = rng.next_u64();
            if val != prev {
                all_same = false;
            }
            prev = val;
        }
        assert!(!all_same, "CryptoRng produced identical values");
    }

    #[test]
    fn test_crypto_rng_variation_large() {
        let mut rng = CryptoRng::new();
        let mut values = HashSet::new();
        for _ in 0..100 {
            values.insert(rng.next_u64());
        }
        assert!(values.len() > 90, "CryptoRng output not varied enough");
    }

    #[test]
    fn test_crypto_rng_random_range_uniform() {
        let mut rng = CryptoRng::new();
        let mut counts = [0usize; 10];
        for _ in 0..1000 {
            let v = rng.random_range(0..10usize);
            counts[v] += 1;
        }
        for &c in &counts {
            assert!(
                (c as isize - 100).abs() < 50,
                "Crypto RNG counts far from uniform: {c}"
            );
        }
    }

    #[test]
    fn test_crypto_normal_distribution() {
        let mut rng = CryptoRng::new();
        let mean = 0.0;
        let sd = 1.0;
        let n = 2000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let val = rng.normal(mean, sd);
            sum += val;
            sum_sq += val * val;
        }
        let sample_mean = sum / n as f64;
        let sample_var = sum_sq / n as f64 - sample_mean * sample_mean;
        assert!(sample_mean.abs() < 0.1);
        assert!((sample_var - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_two_instances_different_values() {
        let mut a = CryptoRng::new();
        let mut b = CryptoRng::new();
        let va = a.next_u64();
        let vb = b.next_u64();
        assert_ne!(va, vb);
    }

    #[test]
    fn test_crypto_rng_helper_function() {
        let mut rng = crypto_rng();
        let _ = rng.next_u64();
    }
}
