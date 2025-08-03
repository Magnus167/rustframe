# Introduction

📚 [Docs](https://magnus167.github.io/rustframe/) | 🐙 [GitHub](https://github.com/Magnus167/rustframe) | 🦀 [Crates.io](https://crates.io/crates/rustframe) | 🔖 [docs.rs](https://docs.rs/rustframe/latest/rustframe/)

Welcome to the **Rustframe User Guide**. Rustframe is a lightweight dataframe
and math toolkit for Rust written in 100% safe Rust. It focuses on keeping the
API approachable while offering handy features for small analytical or
educational projects.

Rustframe bundles:

- column‑labelled frames built on a fast column‑major matrix
- familiar element‑wise math and aggregation routines
- a growing `compute` module for statistics and machine learning
- utilities for dates and random numbers

```rust
# extern crate rustframe;
use rustframe::{frame::Frame, matrix::{Matrix, SeriesOps}};

let data = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
let frame = Frame::new(data, vec!["A", "B"], None);

// Perform column wise aggregation
assert_eq!(frame.sum_vertical(), vec![3.0, 7.0]);
```

## Resources

- [GitHub repository](https://github.com/Magnus167/rustframe)
- [Crates.io](https://crates.io/crates/rustframe) & [API docs](https://docs.rs/rustframe)
- [Code coverage](https://codecov.io/gh/Magnus167/rustframe)

This guide walks through the main building blocks of the library. Each chapter
contains runnable snippets so you can follow along:

1. [Data manipulation](./data-manipulation.md) for loading and transforming data
2. [Compute features](./compute.md) for statistics and analytics
3. [Machine learning](./machine-learning.md) for predictive models
4. [Utilities](./utilities.md) for supporting helpers and upcoming modules
