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

### Gaussian Naive Bayes

Gaussian Naive Bayes classifier for continuous features:

```rust
# extern crate rustframe;
use rustframe::compute::models::gaussian_nb::GaussianNB;
use rustframe::matrix::Matrix;

// Training data with 2 features
let x = Matrix::from_rows_vec(vec![
    1.0, 2.0,
    2.0, 3.0,
    3.0, 4.0,
    4.0, 5.0
], 4, 2);

// Class labels (0 or 1)
let y = Matrix::from_vec(vec![0.0, 0.0, 1.0, 1.0], 4, 1);

// Train the model
let mut model = GaussianNB::new(1e-9, true);
model.fit(&x, &y);

// Make predictions
let predictions = model.predict(&x);
assert_eq!(predictions.rows(), 4);
```

### Dense Neural Networks

Simple fully connected neural network:

```rust
# extern crate rustframe;
use rustframe::compute::models::dense_nn::{DenseNN, DenseNNConfig, ActivationKind, InitializerKind, LossKind};
use rustframe::matrix::Matrix;

// Training data with 2 features
let x = Matrix::from_rows_vec(vec![
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
], 4, 2);

// XOR target outputs
let y = Matrix::from_vec(vec![0.0, 1.0, 1.0, 0.0], 4, 1);

// Create a neural network with 2 hidden layers
let config = DenseNNConfig {
    input_size: 2,
    hidden_layers: vec![4, 4],
    output_size: 1,
    activations: vec![ActivationKind::Sigmoid, ActivationKind::Sigmoid, ActivationKind::Sigmoid],
    initializer: InitializerKind::Uniform(0.5),
    loss: LossKind::MSE,
    learning_rate: 0.1,
    epochs: 1000,
};
let mut model = DenseNN::new(config);

// Train the model
model.train(&x, &y);

// Make predictions
let predictions = model.predict(&x);
assert_eq!(predictions.rows(), 4);
```

For helper functions and upcoming modules, visit the
[utilities](./utilities.md) section.
