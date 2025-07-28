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

    #[test]
    fn test_range_sample_usize_boundary() {
        assert_eq!(<usize as RangeSample>::from_u64(0, &(0..1)), 0);
        assert_eq!(<usize as RangeSample>::from_u64(u64::MAX, &(0..1)), 0);
    }

    #[test]
    fn test_range_sample_f64_boundary() {
        let v0 = <f64 as RangeSample>::from_u64(0, &(0.0..1.0));
        let vmax = <f64 as RangeSample>::from_u64(u64::MAX, &(0.0..1.0));
        assert!(v0 >= 0.0 && v0 < 1.0);
        assert!(vmax > 0.999999999999 && vmax <= 1.0);
    }

    #[test]
    fn test_range_sample_usize_varied() {
        for i in 0..5 {
            let v = <usize as RangeSample>::from_u64(i, &(10..15));
            assert!(v >= 10 && v < 15);
        }
    }

    #[test]
    fn test_range_sample_f64_span() {
        for val in [0, u64::MAX / 2, u64::MAX] {
            let f = <f64 as RangeSample>::from_u64(val, &(2.0..4.0));
            assert!(f >= 2.0 && f <= 4.0);
        }
    }

    #[test]
    fn test_range_sample_usize_single_value() {
        for val in [0, 1, u64::MAX] {
            let n = <usize as RangeSample>::from_u64(val, &(5..6));
            assert_eq!(n, 5);
        }
    }

    #[test]
    fn test_range_sample_f64_negative_range() {
        for val in [0, u64::MAX / 3, u64::MAX] {
            let f = <f64 as RangeSample>::from_u64(val, &(-2.0..2.0));
            assert!(f >= -2.0 && f <= 2.0);
        }
    }
}
