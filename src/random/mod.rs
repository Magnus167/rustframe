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
