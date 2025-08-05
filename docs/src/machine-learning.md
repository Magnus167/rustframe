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

## Gaussian Naive Bayes

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

## Dense Neural Networks

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

## Real-world Examples

### Housing Price Prediction

```rust
# extern crate rustframe;
use rustframe::compute::models::linreg::LinReg;
use rustframe::matrix::Matrix;

// Features: square feet and bedrooms
let features = Matrix::from_rows_vec(vec![
    2100.0, 3.0,
    1600.0, 2.0,
    2400.0, 4.0,
    1400.0, 2.0,
], 4, 2);

// Sale prices
let target = Matrix::from_vec(vec![400_000.0, 330_000.0, 369_000.0, 232_000.0], 4, 1);

let mut model = LinReg::new(2);
model.fit(&features, &target, 1e-8, 10_000);

// Predict price of a new home
let new_home = Matrix::from_vec(vec![2000.0, 3.0], 1, 2);
let predicted_price = model.predict(&new_home);
println!("Predicted price: ${}", predicted_price.data()[0]);
```

### Spam Detection

```rust
# extern crate rustframe;
use rustframe::compute::models::logreg::LogReg;
use rustframe::matrix::Matrix;

// 20 e-mails × 5 features = 100 numbers (row-major, spam first)
let x = Matrix::from_rows_vec(
    vec![
        // ─────────── spam examples ───────────
        2.0, 1.0, 1.0, 1.0, 1.0, // "You win a FREE offer - click for money-back bonus!"
        1.0, 0.0, 1.0, 1.0, 0.0, // "FREE offer! Click now!"
        0.0, 2.0, 0.0, 1.0, 1.0, // "Win win win - money inside, click…"
        1.0, 1.0, 0.0, 0.0, 1.0, // "Limited offer to win easy money…"
        1.0, 0.0, 1.0, 0.0, 1.0, // ...
        0.0, 1.0, 1.0, 1.0, 0.0, // ...
        2.0, 0.0, 0.0, 1.0, 1.0, // ...
        0.0, 1.0, 1.0, 0.0, 1.0, // ...
        1.0, 1.0, 1.0, 1.0, 0.0, // ...
        1.0, 0.0, 0.0, 1.0, 1.0, // ...
        // ─────────── ham examples ───────────
        0.0, 0.0, 0.0, 0.0, 0.0, // "See you at the meeting tomorrow."
        0.0, 0.0, 0.0, 1.0, 0.0, // "Here's the Zoom click-link."
        0.0, 0.0, 0.0, 0.0, 1.0, // "Expense report: money attached."
        0.0, 0.0, 0.0, 1.0, 1.0, // ...
        0.0, 1.0, 0.0, 0.0, 0.0, // "Did we win the bid?"
        0.0, 0.0, 0.0, 0.0, 0.0, // ...
        0.0, 0.0, 0.0, 1.0, 0.0, // ...
        1.0, 0.0, 0.0, 0.0, 0.0, // "Special offer for staff lunch."
        0.0, 0.0, 0.0, 0.0, 0.0, // ...
        0.0, 0.0, 0.0, 1.0, 0.0,
    ],
    20,
    5,
);

// Labels: 1 = spam, 0 = ham
let y = Matrix::from_vec(
    vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 10 spam
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 10 ham
    ],
    20,
    1,
);

// Train
let mut model = LogReg::new(5);
model.fit(&x, &y, 0.01, 5000);

// Predict
// e.g. "free money offer"
let email_data = vec![1.0, 0.0, 1.0, 0.0, 1.0];
let email = Matrix::from_vec(email_data, 1, 5);
let prob_spam = model.predict_proba(&email);
println!("Probability of spam: {:.4}", prob_spam.data()[0]);
```

### Iris Flower Classification

```rust
# extern crate rustframe;
use rustframe::compute::models::gaussian_nb::GaussianNB;
use rustframe::matrix::Matrix;

// Features: sepal length and petal length
let x = Matrix::from_rows_vec(vec![
    5.1, 1.4, // setosa
    4.9, 1.4, // setosa
    6.2, 4.5, // versicolor
    5.9, 5.1, // virginica
], 4, 2);

let y = Matrix::from_vec(vec![0.0, 0.0, 1.0, 2.0], 4, 1);
let names = vec!["setosa", "versicolor", "virginica"];

let mut model = GaussianNB::new(1e-9, true);
model.fit(&x, &y);

let sample = Matrix::from_vec(vec![5.0, 1.5], 1, 2);
let predicted_class = model.predict(&sample);
let class_name = names[predicted_class.data()[0] as usize];
println!("Predicted class: {} ({:?})", class_name, predicted_class.data()[0]);
```

### Customer Segmentation

```rust
# extern crate rustframe;
use rustframe::compute::models::k_means::KMeans;
use rustframe::matrix::Matrix;

// Each row: [age, annual_income]
let customers = Matrix::from_rows_vec(
    vec![
        25.0, 40_000.0, 34.0, 52_000.0, 58.0, 95_000.0, 45.0, 70_000.0,
    ],
    4,
    2,
);

let (model, labels) = KMeans::fit(&customers, 2, 20, 1e-4);

let new_customer = Matrix::from_vec(vec![30.0, 50_000.0], 1, 2);
let cluster = model.predict(&new_customer)[0];
println!("New customer belongs to cluster: {}", cluster);
println!("Cluster labels: {:?}", labels);
```

For helper functions and upcoming modules, visit the
[utilities](./utilities.md) section.
