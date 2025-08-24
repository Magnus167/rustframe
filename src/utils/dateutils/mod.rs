//! Generators for sequences of calendar and business dates.
//!
//! See [`dates`] for all-day calendars and [`bdates`] for business-day aware
//! variants.
//!
//! ```
//! use rustframe::utils::dateutils::{DatesList, DateFreq};
//! let list = DatesList::new("2024-01-01".into(), "2024-01-02".into(), DateFreq::Daily);
//! assert_eq!(list.count().unwrap(), 2);
//! ```
pub mod bdates;
pub mod dates;

pub use bdates::{BDateFreq, BDatesGenerator, BDatesList};
pub use dates::{DateFreq, DatesGenerator, DatesList};
