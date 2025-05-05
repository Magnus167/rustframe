
# <img align="center" alt="Rustframe" src=".github/rustframe_logo.png" height="50" /> rustframe

<!-- though the centre tag doesn't work as it would noramlly, it achieves the desired effect -->

📚 [Docs](https://magnus167.github.io/rustframe/) | 🐙 [GitHub](https://github.com/Magnus167/rustframe) | 🌐 [Gitea mirror](https://gitea.nulltech.uk/Magnus167/rustframe) | 🦀 [Crates.io](https://crates.io/crates/rustframe) | 🔖 [docs.rs](https://docs.rs/rustframe/latest/rustframe/)

<!-- [![Last commit](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/rustframe/last-commit-date.json)](https://github.com/Magnus167/rustframe) -->
[![codecov](https://codecov.io/gh/Magnus167/rustframe/graph/badge.svg?token=J7ULJEFTVI)](https://codecov.io/gh/Magnus167/rustframe)
[![Coverage](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/docs/tarpaulin-badge.json)](https://magnus167.github.io/rustframe/docs/tarpaulin-report.html)

---

## Rustframe: *A lightweight dataframe & math toolkit for Rust*

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
    utils::{BDateFreq, BDatesList},
};

let n_periods = 4;

// Four business days starting 2024‑01‑02
let dates: Vec<NaiveDate> =
    BDatesList::from_n_periods("2024-01-02".to_string(), BDateFreq::Daily, n_periods)
        .unwrap()
        .list()
        .unwrap();

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
    .eq_elem(ma)
    .all();
assert!(check);


```
