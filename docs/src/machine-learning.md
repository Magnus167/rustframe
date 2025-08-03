# Machine Learning

The `compute::models` module bundles several learning algorithms that operate on
`Matrix` structures. These examples highlight the basic training and prediction
APIs. For more end‑to‑end walkthroughs see the examples directory in the
repository.

Currently implemented models include:

- Linear and logistic regression
- K‑means clustering
- Principal component analysis (PCA)
- Gaussian Naive Bayes
- Dense neural networks

## Linear Regression

```rust
# extern crate rustframe;
use rustframe::compute::models::linreg::LinReg;
use rustframe::matrix::Matrix;

let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
let y = Matrix::from_vec(vec![2.0, 3.0, 4.0, 5.0], 4, 1);
let mut model = LinReg::new(1);
model.fit(&x, &y, 0.01, 100);
let preds = model.predict(&x);
assert_eq!(preds.rows(), 4);
```

## K-means Walkthrough

```rust
# extern crate rustframe;
use rustframe::compute::models::k_means::KMeans;
use rustframe::matrix::Matrix;

let data = Matrix::from_vec(vec![1.0, 1.0, 5.0, 5.0], 2, 2);
let (model, _labels) = KMeans::fit(&data, 2, 10, 1e-4);
let new_point = Matrix::from_vec(vec![0.0, 0.0], 1, 2);
let cluster = model.predict(&new_point)[0];
```

For helper functions and upcoming modules, visit the
[utilities](./utilities.md) section.

## Logistic Regression

```rust
# extern crate rustframe;
use rustframe::compute::models::logreg::LogReg;
use rustframe::matrix::Matrix;

let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
let y = Matrix::from_vec(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
let mut model = LogReg::new(1);
model.fit(&x, &y, 0.1, 200);
let preds = model.predict_proba(&x);
assert_eq!(preds.rows(), 4);
```

## Principal Component Analysis

```rust
# extern crate rustframe;
use rustframe::compute::models::pca::PCA;
use rustframe::matrix::Matrix;

let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let pca = PCA::fit(&data, 1, 0);
let transformed = pca.transform(&data);
assert_eq!(transformed.cols(), 1);
```

For helper functions and upcoming modules, visit the
[utilities](./utilities.md) section.
