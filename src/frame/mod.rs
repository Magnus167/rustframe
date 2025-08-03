//! High-level interface for working with columnar data and row indices.
//!
//! The [`Frame`](crate::frame::Frame) type combines a matrix with column labels and a typed row
//! index, similar to data frames in other data-analysis libraries.
//!
//! # Examples
//!
//! ```
//! use rustframe::frame::{Frame, RowIndex};
//! use rustframe::matrix::Matrix;
//!
//! // Build a frame from two columns labelled "A" and "B".
//! let data = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
//! let frame = Frame::new(data, vec!["A", "B"], None);
//!
//! assert_eq!(frame["A"], vec![1.0, 2.0]);
//! assert_eq!(frame.index(), &RowIndex::Range(0..2));
//! ```
pub mod base;
pub mod ops;

pub use base::*;

#[allow(unused_imports)]
pub use ops::*;
