//! Core matrix types and operations.
//!
//! The [`Matrix`](crate::matrix::Matrix) struct provides a simple column‑major 2D array with a
//! suite of numeric helpers. Additional traits like [`SeriesOps`](crate::matrix::SeriesOps) and
//! [`BoolOps`](crate::matrix::BoolOps) extend functionality for common statistics and logical reductions.
//!
//! # Examples
//!
//! ```
//! use rustframe::matrix::Matrix;
//!
//! let m = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
//! assert_eq!(m.shape(), (2, 2));
//! assert_eq!(m[(0,1)], 3);
//! ```
pub mod boolops;
pub mod mat;
pub mod seriesops;

pub use boolops::*;
pub use mat::*;
pub use seriesops::*;
