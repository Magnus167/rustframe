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
