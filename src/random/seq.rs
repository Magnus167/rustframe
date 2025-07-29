use crate::random::Rng;

/// Trait for randomizing slices.
pub trait SliceRandom {
    /// Shuffle the slice in place using the provided RNG.
    fn shuffle<R: Rng>(&mut self, rng: &mut R);
}

impl<T> SliceRandom for [T] {
    fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        for i in (1..self.len()).rev() {
            let j = rng.random_range(0..(i + 1));
            self.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{CryptoRng, Prng};

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
    fn test_slice_shuffle_deterministic_with_prng() {
        let mut rng1 = Prng::new(11);
        let mut rng2 = Prng::new(11);
        let mut a = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut b = a.clone();
        a.shuffle(&mut rng1);
        b.shuffle(&mut rng2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_slice_shuffle_crypto_random_changes() {
        let mut rng1 = CryptoRng::new();
        let mut rng2 = CryptoRng::new();
        let orig = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut a = orig.clone();
        let mut b = orig.clone();
        a.shuffle(&mut rng1);
        b.shuffle(&mut rng2);
        assert!(a != orig || b != orig, "Shuffles did not change order");
        assert_ne!(a, b, "Two Crypto RNG shuffles produced same order");
    }

    #[test]
    fn test_shuffle_single_element_no_change() {
        let mut rng = Prng::new(1);
        let mut arr = [42];
        arr.shuffle(&mut rng);
        assert_eq!(arr, [42]);
    }

    #[test]
    fn test_multiple_shuffles_different_results() {
        let mut rng = Prng::new(5);
        let mut arr1 = [1, 2, 3, 4];
        let mut arr2 = [1, 2, 3, 4];
        arr1.shuffle(&mut rng);
        arr2.shuffle(&mut rng);
        assert_ne!(arr1, arr2);
    }

    #[test]
    fn test_shuffle_empty_slice() {
        let mut rng = Prng::new(1);
        let mut arr: [i32; 0] = [];
        arr.shuffle(&mut rng);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_shuffle_three_uniform() {
        use std::collections::HashMap;
        let mut rng = Prng::new(123);
        let mut counts: HashMap<[u8; 3], usize> = HashMap::new();
        for _ in 0..6000 {
            let mut arr = [1u8, 2, 3];
            arr.shuffle(&mut rng);
            *counts.entry(arr).or_insert(0) += 1;
        }
        let expected = 1000.0;
        let chi2: f64 = counts
            .values()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();
        assert!(chi2 < 30.0, "shuffle chi-square too high: {chi2}");
    }
}
