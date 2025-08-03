//! Random number generation utilities.
//!
//! Provides both a simple pseudo-random generator [`Prng`](crate::random::Prng) and a
//! cryptographically secure alternative [`CryptoRng`](crate::random::CryptoRng). The
//! [`SliceRandom`](crate::random::SliceRandom) trait offers shuffling of slices using any RNG
//! implementing [`Rng`](crate::random::Rng).
//!
//! ```
//! use rustframe::random::{rng, SliceRandom};
//!
//! let mut rng = rng();
//! let mut data = [1, 2, 3, 4];
//! data.shuffle(&mut rng);
//! assert_eq!(data.len(), 4);
//! ```
pub mod crypto;
pub mod prng;
pub mod random_core;
pub mod seq;

pub use crypto::{crypto_rng, CryptoRng};
pub use prng::{rng, Prng};
pub use random_core::{RangeSample, Rng};
pub use seq::SliceRandom;

pub mod prelude {
    pub use super::seq::SliceRandom;
    pub use super::{crypto_rng, rng, CryptoRng, Prng, RangeSample, Rng};
}
