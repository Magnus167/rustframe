# Compute Features

The `compute` module hosts numerical routines for exploratory data analysis.
It covers descriptive statistics, correlations, probability distributions and
some basic inferential tests.

## Basic Statistics

```rust
# extern crate rustframe;
use rustframe::compute::stats::{mean, mean_horizontal, mean_vertical, stddev, median, population_variance, percentile};
use rustframe::matrix::Matrix;

let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
assert_eq!(mean(&m), 2.5);
assert_eq!(stddev(&m), 1.118033988749895);
assert_eq!(median(&m), 2.5);
assert_eq!(population_variance(&m), 1.25);
assert_eq!(percentile(&m, 50.0), 3.0);
// column averages returned as 1 x n matrix
let row_means = mean_horizontal(&m);
assert_eq!(row_means.data(), &[2.0, 3.0]);
let col_means = mean_vertical(&m);
assert_eq!(col_means.data(), & [1.5, 3.5]);
```

### Axis-specific Operations

Operations can be applied along specific axes (rows or columns):

```rust
# extern crate rustframe;
use rustframe::compute::stats::{mean_vertical, mean_horizontal, stddev_vertical, stddev_horizontal};
use rustframe::matrix::Matrix;

// 3x2 matrix
let m = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

// Mean along columns (vertical) - returns 1 x cols matrix
let col_means = mean_vertical(&m);
assert_eq!(col_means.shape(), (1, 2));
assert_eq!(col_means.data(), &[3.0, 4.0]); // [(1+3+5)/3, (2+4+6)/3]

// Mean along rows (horizontal) - returns rows x 1 matrix
let row_means = mean_horizontal(&m);
assert_eq!(row_means.shape(), (3, 1));
assert_eq!(row_means.data(), &[1.5, 3.5, 5.5]); // [(1+2)/2, (3+4)/2, (5+6)/2]

// Standard deviation along columns
let col_stddev = stddev_vertical(&m);
assert_eq!(col_stddev.shape(), (1, 2));

// Standard deviation along rows
let row_stddev = stddev_horizontal(&m);
assert_eq!(row_stddev.shape(), (3, 1));
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

### Additional Distributions

Rustframe provides several other probability distributions:

```rust
# extern crate rustframe;
use rustframe::compute::stats::distributions::{normal_cdf, binomial_pmf, binomial_cdf, poisson_pmf};
use rustframe::matrix::Matrix;

// Normal distribution CDF
let x = Matrix::from_vec(vec![0.0, 1.0], 1, 2);
let cdf = normal_cdf(x, 0.0, 1.0);
assert_eq!(cdf.data().len(), 2);

// Binomial distribution PMF
// Probability of k successes in n trials with probability p
let k = Matrix::from_vec(vec![0_u64, 1, 2, 3], 1, 4);
let pmf = binomial_pmf(3, k.clone(), 0.5);
assert_eq!(pmf.data().len(), 4);

// Binomial distribution CDF
let cdf = binomial_cdf(3, k, 0.5);
assert_eq!(cdf.data().len(), 4);

// Poisson distribution PMF
// Probability of k events with rate parameter lambda
let k = Matrix::from_vec(vec![0_u64, 1, 2], 1, 3);
let pmf = poisson_pmf(2.0, k);
assert_eq!(pmf.data().len(), 3);
```

### Inferential Statistics

Rustframe provides several inferential statistical tests:

```rust
# extern crate rustframe;
use rustframe::matrix::Matrix;
use rustframe::compute::stats::inferential::{t_test, chi2_test, anova};

// Two-sample t-test
let sample1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
let sample2 = Matrix::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
let (t_statistic, p_value) = t_test(&sample1, &sample2);
assert!((t_statistic + 5.0).abs() < 1e-5);
assert!(p_value > 0.0 && p_value < 1.0);

// Chi-square test of independence
let observed = Matrix::from_vec(vec![12.0, 5.0, 8.0, 10.0], 2, 2);
let (chi2_statistic, p_value) = chi2_test(&observed);
assert!(chi2_statistic > 0.0);
assert!(p_value > 0.0 && p_value < 1.0);

// One-way ANOVA
let group1 = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
let group2 = Matrix::from_vec(vec![2.0, 3.0, 4.0], 1, 3);
let group3 = Matrix::from_vec(vec![3.0, 4.0, 5.0], 1, 3);
let groups = vec![&group1, &group2, &group3];
let (f_statistic, p_value) = anova(groups);
assert!(f_statistic > 0.0);
assert!(p_value > 0.0 && p_value < 1.0);
```

With the basics covered, explore predictive models in the
[machine learning](./machine-learning.md) chapter.
