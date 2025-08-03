# Compute Features

The `compute` module provides statistical routines like descriptive
statistics and correlation measures.

## Basic Statistics

```rust
# extern crate rustframe;
use rustframe::compute::stats::{mean, stddev};
use rustframe::matrix::Matrix;

let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let mean_val = mean(&m);
let std_val = stddev(&m);
```

## Correlation

```rust
# extern crate rustframe;
use rustframe::compute::stats::pearson;
use rustframe::matrix::Matrix;

let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let y = Matrix::from_vec(vec![2.0, 4.0, 6.0, 8.0], 2, 2);
let corr = pearson(&x, &y);
```

With the basics covered, explore predictive models in the
[machine learning](./machine-learning.md) chapter.
