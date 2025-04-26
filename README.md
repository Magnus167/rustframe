# rustframe

📚 [Docs](https://magnus167.github.io/rustframe/) | 🐙 [GitHub](https://github.com/Magnus167/rustframe) | 🌐 [Gitea mirror](https://gitea.nulltech.uk/Magnus167/rustframe) | 🦀 [Crates.io](https://crates.io/crates/rustframe) | 🔖 [docs.rs](https://docs.rs/rustframe/latest/rustframe/)

<!-- [![Last commit](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/rustframe/last-commit-date.json)](https://github.com/Magnus167/rustframe) -->
[![codecov](https://codecov.io/gh/Magnus167/rustframe/graph/badge.svg?token=J7ULJEFTVI)](https://codecov.io/gh/Magnus167/rustframe)
[![Coverage](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/rustframe/tarpaulin-badge.json)](https://magnus167.github.io/rustframe/rustframe/tarpaulin-report.html)

---

## Rustframe: *A lightweight dataframe helper for Rust*

Rustframe is a simple dataframe helper for simple math and data manipulation in Rust.

Rustframe keeps things simple, safe, and readable. It is handy for quick numeric experiments and small analytical tasks, but it is **not** meant to compete with powerhouse crates like `polars` or `ndarray`.

### What it offers

- **Math that reads like math** – element‑wise `+`, `−`, `×`, `÷` on entire frames or scalars.
- **Broadcast & reduce** – sum, product, any/all across rows or columns without boilerplate.
- **Boolean masks made simple** – chain comparisons, combine with `&`/`|`, get a tidy `BoolMatrix` back.
- **Date‑centric row index** – business‑day ranges and calendar slicing built in.
- **Pure safe Rust** – 100 % safe, zero `unsafe`.

### Heads up

- **Not memory‑efficient (yet)** – footprint needs work.
- **Feature set still small** – expect missing pieces.

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

let col_names = vec!["a".to_string(), "b".to_string()];

let ma = Matrix::from_cols(vec![
    vec![1.0, 2.0, 3.0, 4.0],
    vec![5.0, 6.0, 7.0, 8.0],
]);
let mb = Matrix::from_cols(vec![
    vec![4.0, 3.0, 2.0, 1.0],
    vec![8.0, 7.0, 6.0, 5.0],
]);

let fa = Frame::new(ma, col_names.clone(), Some(RowIndex::Date(dates.clone())));
let fb = Frame::new(mb, col_names, Some(RowIndex::Date(dates)));

// Math that reads like math
let result = &fa * &fb;            // element‑wise multiply
let total = result.matrix().sum_vertical().iter().sum::<f64>();
assert_eq!(total, 184.0);
```
