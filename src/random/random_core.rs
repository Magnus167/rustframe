use std::f64::consts::PI;
use std::ops::Range;

/// Trait implemented by random number generators.
pub trait Rng {
    /// Generate the next random `u64` value.
    fn next_u64(&mut self) -> u64;

    /// Generate a value uniformly in the given range.
    fn random_range<T>(&mut self, range: Range<T>) -> T
    where
        T: RangeSample,
    {
        T::from_u64(self.next_u64(), &range)
    }

    /// Generate a boolean with probability 0.5 of being `true`.
    fn gen_bool(&mut self) -> bool {
        self.random_range(0..2usize) == 1
    }

    /// Sample from a normal distribution using the Box-Muller transform.
    fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        let u1 = self.random_range(0.0..1.0);
        let u2 = self.random_range(0.0..1.0);
        mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Conversion from a raw `u64` into a type within a range.
pub trait RangeSample: Sized {
    fn from_u64(value: u64, range: &Range<Self>) -> Self;
}

impl RangeSample for usize {
    fn from_u64(value: u64, range: &Range<Self>) -> Self {
        let span = range.end - range.start;
        (value as usize % span) + range.start
    }
}

impl RangeSample for f64 {
    fn from_u64(value: u64, range: &Range<Self>) -> Self {
        let span = range.end - range.start;
        range.start + (value as f64 / u64::MAX as f64) * span
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{CryptoRng, Prng, SliceRandom};

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
    fn test_shuffle_slice() {
        let mut rng = Prng::new(3);
        let mut arr = [1, 2, 3, 4, 5];
        let orig = arr.clone();
        arr.shuffle(&mut rng);
        assert_eq!(arr.len(), orig.len());
        let mut sorted = arr.to_vec();
        sorted.sort();
        assert_eq!(sorted, orig.to_vec());
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
}
