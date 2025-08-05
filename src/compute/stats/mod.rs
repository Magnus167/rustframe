//! Statistical routines for matrices.
//!
//! Functions are grouped into submodules for descriptive statistics,
//! correlations, probability distributions and basic inferential tests.
//!
//! ```
//! use rustframe::compute::stats;
//! use rustframe::matrix::Matrix;
//!
//! let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//! let cov = stats::covariance(&m, &m);
//! assert!((cov - 1.25).abs() < 1e-8);
//! ```
pub mod correlation;
pub mod descriptive;
pub mod distributions;
pub mod inferential;

pub use correlation::*;
pub use descriptive::*;
pub use distributions::*;
pub use inferential::*;
