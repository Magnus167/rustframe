# rustframe

<!-- # <img align="center" alt="Rustframe" src=".github/rustframe_logo.png" height="50px" /> rustframe -->

<!-- though the centre tag doesn't work as it would noramlly, it achieves the desired effect -->

📚 [Docs](https://magnus167.github.io/rustframe/) | 🐙 [GitHub](https://github.com/Magnus167/rustframe) | 🌐 [Gitea mirror](https://gitea.nulltech.uk/Magnus167/rustframe) | 🦀 [Crates.io](https://crates.io/crates/rustframe) | 🔖 [docs.rs](https://docs.rs/rustframe/latest/rustframe/)

<!-- [![Last commit](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/rustframe/last-commit-date.json)](https://github.com/Magnus167/rustframe) -->

[![codecov](https://codecov.io/gh/Magnus167/rustframe/graph/badge.svg?token=J7ULJEFTVI)](https://codecov.io/gh/Magnus167/rustframe)
[![Coverage](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/docs/tarpaulin-badge.json)](https://magnus167.github.io/rustframe/docs/tarpaulin-report.html)

---

## Rustframe: _A lightweight dataframe & math toolkit for Rust_

Rustframe provides intuitive dataframe, matrix, and series operations small-to-mid scale data analysis and manipulation.

Rustframe keeps things simple, safe, and readable. It is handy for quick numeric experiments and small analytical tasks, but it is **not** meant to compete with powerhouse crates like `polars` or `ndarray`.

### What it offers

- **Math that reads like math** - element‑wise `+`, `−`, `×`, `÷` on entire frames or scalars.
- **Broadcast & reduce** - sum, product, any/all across rows or columns without boilerplate.
- **Boolean masks made simple** - chain comparisons, combine with `&`/`|`, get a tidy `BoolMatrix` back.
- **Date‑centric row index** - business‑day ranges and calendar slicing built in.
- **Pure safe Rust** - 100 % safe, zero `unsafe`.

### Heads up

- **Not memory‑efficient (yet)** - footprint needs work.
- **Feature set still small** - expect missing pieces.

### On the horizon

- Optional GPU help (Vulkan or similar) for heavier workloads.
- Straightforward Python bindings using `pyo3`.

---

## Quick start

```rust
use chrono::NaiveDate;
use rustframe::{
    frame::{Frame, RowIndex},
    matrix::{BoolOps, Matrix, SeriesOps},
    utils::{DateFreq, BDatesList},
};

let n_periods = 4;

// Four business days starting 2024‑01‑02
let dates: Vec<NaiveDate> =
    BDatesList::from_n_periods("2024-01-02".to_string(), DateFreq::Daily, n_periods)
        .unwrap()
        .list().unwrap();

let col_names: Vec<String> = vec!["a".to_string(), "b".to_string()];

let ma: Matrix<f64> =
    Matrix::from_cols(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
let mb: Matrix<f64> =
    Matrix::from_cols(vec![vec![4.0, 3.0, 2.0, 1.0], vec![8.0, 7.0, 6.0, 5.0]]);

let fa: Frame<f64> = Frame::new(
    ma.clone(),
    col_names.clone(),
    Some(RowIndex::Date(dates.clone())),
);
let fb: Frame<f64> = Frame::new(mb, col_names, Some(RowIndex::Date(dates)));

// Math that reads like math
let result: Frame<f64> = &fa * &fb; // element‑wise multiply
let total: f64 = result.sum_vertical().iter().sum::<f64>();
assert_eq!(total, 184.0);

// broadcast & reduce
let result: Matrix<f64> = ma.clone() + 1.0; // add scalar
let result: Matrix<f64> = result + &ma - &ma; // add matrix
let result: Matrix<f64> = result - 1.0; // subtract scalar
let result: Matrix<f64> = result * 2.0; // multiply by scalar
let result: Matrix<f64> = result / 2.0; // divide by scalar

let check: bool = result.eq_elem(ma.clone()).all();
assert!(check);

// The above math can also be written as:
let check: bool = (&(&(&(&ma + 1.0) - 1.0) * 2.0) / 2.0)
    .eq_elem(ma.clone())
    .all();
assert!(check);

// The above math can also be written as:
let check: bool = ((((ma.clone() + 1.0) - 1.0) * 2.0) / 2.0)
    .eq_elem(ma.clone())
    .all();
assert!(check);

// Matrix multiplication
let mc: Matrix<f64> = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
let md: Matrix<f64> = Matrix::from_cols(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
let mul_result: Matrix<f64> = mc.matrix_mul(&md);
// Expected:
// 1*5 + 3*6 = 5 + 18 = 23
// 2*5 + 4*6 = 10 + 24 = 34
// 1*7 + 3*8 = 7 + 24 = 31
// 2*7 + 4*8 = 14 + 32 = 46
assert_eq!(mul_result.data(), &[23.0, 34.0, 31.0, 46.0]);

// Dot product (alias for matrix_mul for FloatMatrix)
let dot_result: Matrix<f64> = mc.dot(&md);
assert_eq!(dot_result, mul_result);

// Transpose
let original_matrix: Matrix<f64> = Matrix::from_cols(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
// Original:
// 1 4
// 2 5
// 3 6
let transposed_matrix: Matrix<f64> = original_matrix.transpose();
// Transposed:
// 1 2 3
// 4 5 6
assert_eq!(transposed_matrix.rows(), 2);
assert_eq!(transposed_matrix.cols(), 3);
assert_eq!(transposed_matrix.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

```

### More examples

See the [examples](./examples/) directory for some demonstrations of Rustframe's syntax and functionality.

To run the examples, use:

```bash
cargo run --example <example_name>
```

E.g. to run the `game_of_life` example:

```bash
cargo run --example game_of_life
```
