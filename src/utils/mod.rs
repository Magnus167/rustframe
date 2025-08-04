//! Assorted helper utilities.
//!
//! Currently this module exposes date generation utilities in [`dateutils`](crate::utils::dateutils),
//! including calendar and business date sequences.
//!
//! ```
//! use rustframe::utils::DatesList;
//! use rustframe::utils::DateFreq;
//! let dates = DatesList::new("2024-01-01".into(), "2024-01-03".into(), DateFreq::Daily);
//! assert_eq!(dates.count().unwrap(), 3);
//! ```
pub mod dateutils;
pub mod spigots;

pub use dateutils::{BDateFreq, BDatesGenerator, BDatesList};
pub use dateutils::{DateFreq, DatesGenerator, DatesList};
