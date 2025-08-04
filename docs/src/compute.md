# Compute Features

The `compute` module hosts numerical routines for exploratory data analysis.
It covers descriptive statistics, correlations, probability distributions and
some basic inferential tests.

## Basic Statistics

```rust
# extern crate rustframe;
use rustframe::compute::stats::{mean, mean_vertical, stddev, median};
use rustframe::matrix::Matrix;

let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
assert_eq!(mean(&m), 2.5);
assert_eq!(stddev(&m), 1.118033988749895);
assert_eq!(median(&m), 2.5);
// column averages returned as 1 x n matrix
let col_means = mean_vertical(&m);
assert_eq!(col_means.data(), &[1.5, 3.5]);
```

## Correlation

```rust
# extern crate rustframe;
use rustframe::compute::stats::{pearson, covariance};
use rustframe::matrix::Matrix;

let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let y = Matrix::from_vec(vec![2.0, 4.0, 6.0, 8.0], 2, 2);
let corr = pearson(&x, &y);
let cov = covariance(&x, &y);
assert!((corr - 1.0).abs() < 1e-8);
assert!((cov - 2.5).abs() < 1e-8);
```

## Distributions

Probability distribution helpers are available for common PDFs and CDFs.

```rust
# extern crate rustframe;
use rustframe::compute::stats::distributions::normal_pdf;
use rustframe::matrix::Matrix;

let x = Matrix::from_vec(vec![0.0, 1.0], 1, 2);
let pdf = normal_pdf(x, 0.0, 1.0);
assert_eq!(pdf.data().len(), 2);
```

### More Compute Examples

```rust
# extern crate rustframe;
use rustframe::matrix::Matrix;
use rustframe::compute::stats::inferential::t_test;

let sample1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
let sample2 = Matrix::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
let (t_statistic, p_value) = t_test(&sample1, &sample2);
assert!((t_statistic + 5.0).abs() < 1e-5);
assert!(p_value > 0.0 && p_value < 1.0);
```

With the basics covered, explore predictive models in the
[machine learning](./machine-learning.md) chapter.
