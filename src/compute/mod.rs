//! Algorithms and statistical utilities built on top of the core matrices.
//!
//! This module groups together machine‑learning models and statistical helper
//! functions. For quick access to basic statistics see [`stats`](crate::compute::stats), while
//! [`models`](crate::compute::models) contains small learning algorithms.
//!
//! ```
//! use rustframe::compute::stats;
//! use rustframe::matrix::Matrix;
//!
//! let m = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1);
//! assert_eq!(stats::mean(&m), 2.0);
//! ```
pub mod models;

pub mod stats;
