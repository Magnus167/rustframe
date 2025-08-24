//! Lightweight machine‑learning models built on matrices.
//!
//! Models are intentionally minimal and operate on the [`Matrix`](crate::matrix::Matrix) type for
//! inputs and parameters.
//!
//! ```
//! use rustframe::compute::models::linreg::LinReg;
//! use rustframe::matrix::Matrix;
//!
//! let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
//! let y = Matrix::from_vec(vec![2.0, 3.0, 4.0, 5.0], 4, 1);
//! let mut model = LinReg::new(1);
//! model.fit(&x, &y, 0.01, 1000);
//! let preds = model.predict(&x);
//! assert_eq!(preds.rows(), 4);
//! ```
pub mod activations;
pub mod dense_nn;
pub mod gaussian_nb;
pub mod k_means;
pub mod linreg;
pub mod logreg;
pub mod pca;
