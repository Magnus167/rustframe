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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;

    #[test]
    fn test_prng_determinism() {
        let mut a = Prng::new(42);
        let mut b = Prng::new(42);
        for _ in 0..5 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_random_range_f64() {
        let mut rng = Prng::new(1);
        for _ in 0..10 {
            let v = rng.random_range(-1.0..1.0);
            assert!(v >= -1.0 && v < 1.0);
        }
    }

    #[test]
    fn test_random_range_usize() {
        let mut rng = Prng::new(9);
        for _ in 0..100 {
            let v = rng.random_range(10..20);
            assert!(v >= 10 && v < 20);
        }
    }

    #[test]
    fn test_gen_bool_balance() {
        let mut rng = Prng::new(123);
        let mut trues = 0;
        for _ in 0..1000 {
            if rng.gen_bool() {
                trues += 1;
            }
        }
        let ratio = trues as f64 / 1000.0;
        assert!(ratio > 0.4 && ratio < 0.6);
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Prng::new(7);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mean = 5.0;
        let sd = 2.0;
        let n = 5000;
        for _ in 0..n {
            let val = rng.normal(mean, sd);
            sum += val;
            sum_sq += val * val;
        }
        let sample_mean = sum / n as f64;
        let sample_var = sum_sq / n as f64 - sample_mean * sample_mean;
        assert!((sample_mean - mean).abs() < 0.1);
        assert!((sample_var - sd * sd).abs() < 0.2 * sd * sd);
    }

    #[test]
    fn test_prng_from_entropy_unique() {
        use std::{collections::HashSet, thread, time::Duration};
        let mut seen = HashSet::new();
        for _ in 0..5 {
            let mut rng = Prng::from_entropy();
            seen.insert(rng.next_u64());
            thread::sleep(Duration::from_micros(1));
        }
        assert!(seen.len() > 1, "Entropy seeds produced identical outputs");
    }

    #[test]
    fn test_prng_uniform_distribution() {
        let mut rng = Prng::new(12345);
        let mut counts = [0usize; 10];
        for _ in 0..10000 {
            let v = rng.random_range(0..10usize);
            counts[v] += 1;
        }
        for &c in &counts {
            // "PRNG counts far from uniform: {c}"
            assert!((c as isize - 1000).abs() < 150);
        }
    }

    #[test]
    fn test_prng_different_seeds_different_output() {
        let mut a = Prng::new(1);
        let mut b = Prng::new(2);
        let va = a.next_u64();
        let vb = b.next_u64();
        assert_ne!(va, vb);
    }

    #[test]
    fn test_prng_gen_bool_varies() {
        let mut rng = Prng::new(99);
        let mut seen_true = false;
        let mut seen_false = false;
        for _ in 0..100 {
            if rng.gen_bool() {
                seen_true = true;
            } else {
                seen_false = true;
            }
        }
        assert!(seen_true && seen_false);
    }

    #[test]
    fn test_random_range_single_usize() {
        let mut rng = Prng::new(42);
        for _ in 0..10 {
            let v = rng.random_range(5..6);
            assert_eq!(v, 5);
        }
    }

    #[test]
    fn test_random_range_single_f64() {
        let mut rng = Prng::new(42);
        for _ in 0..10 {
            let v = rng.random_range(1.234..1.235);
            assert!(v >= 1.234 && v < 1.235);
        }
    }

    #[test]
    fn test_prng_normal_zero_sd() {
        let mut rng = Prng::new(7);
        for _ in 0..5 {
            let v = rng.normal(3.0, 0.0);
            assert_eq!(v, 3.0);
        }
    }

    #[test]
    fn test_random_range_extreme_usize() {
        let mut rng = Prng::new(5);
        for _ in 0..10 {
            let v = rng.random_range(0..usize::MAX);
            assert!(v < usize::MAX);
        }
    }

    #[test]
    fn test_prng_chi_square_uniform() {
        let mut rng = Prng::new(12345);
        let mut counts = [0usize; 10];
        let samples = 10000;
        for _ in 0..samples {
            let v = rng.random_range(0..10usize);
            counts[v] += 1;
        }
        let expected = samples as f64 / 10.0;
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();
        //  "chi-square statistic too high: {chi2}"
        assert!(chi2 < 20.0);
    }

    #[test]
    fn test_prng_monobit() {
        let mut rng = Prng::new(42);
        let mut ones = 0usize;
        let samples = 1000;
        for _ in 0..samples {
            ones += rng.next_u64().count_ones() as usize;
        }
        let total_bits = samples * 64;
        let ratio = ones as f64 / total_bits as f64;
        // "bit ratio far from 0.5: {ratio}"
        assert!((ratio - 0.5).abs() < 0.01);
    }
}
