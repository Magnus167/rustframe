use crate::compute::models::activations::{drelu, relu, sigmoid};
use crate::matrix::{Matrix, SeriesOps};
use rand::prelude::*;

/// Supported activation functions
#[derive(Clone)]
pub enum ActivationKind {
    Relu,
    Sigmoid,
    Tanh,
}

impl ActivationKind {
    /// Apply activation elementwise
    pub fn forward(&self, z: &Matrix<f64>) -> Matrix<f64> {
        match self {
            ActivationKind::Relu => relu(z),
            ActivationKind::Sigmoid => sigmoid(z),
            ActivationKind::Tanh => z.map(|v| v.tanh()),
        }
    }

    /// Compute elementwise derivative w.r.t. pre-activation z
    pub fn derivative(&self, z: &Matrix<f64>) -> Matrix<f64> {
        match self {
            ActivationKind::Relu => drelu(z),
            ActivationKind::Sigmoid => {
                let s = sigmoid(z);
                s.zip(&s, |si, sj| si * (1.0 - sj))
            }
            ActivationKind::Tanh => z.map(|v| 1.0 - v.tanh().powi(2)),
        }
    }
}

/// Weight initialization schemes
#[derive(Clone)]
pub enum InitializerKind {
    /// Uniform(-limit .. limit)
    Uniform(f64),
    /// Xavier/Glorot uniform
    Xavier,
    /// He (Kaiming) uniform
    He,
}

impl InitializerKind {
    pub fn initialize(&self, rows: usize, cols: usize) -> Matrix<f64> {
        let mut rng = rand::rng();
        let fan_in = rows;
        let fan_out = cols;
        let limit = match self {
            InitializerKind::Uniform(l) => *l,
            InitializerKind::Xavier => (6.0 / (fan_in + fan_out) as f64).sqrt(),
            InitializerKind::He => (2.0 / fan_in as f64).sqrt(),
        };
        let data = (0..rows * cols)
            .map(|_| rng.random_range(-limit..limit))
            .collect::<Vec<_>>();
        Matrix::from_vec(data, rows, cols)
    }
}

/// Supported losses
#[derive(Clone)]
pub enum LossKind {
    /// Mean Squared Error: L = 1/m * sum((y_hat - y)^2)
    MSE,
    /// Binary Cross-Entropy: L = -1/m * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
    BCE,
}

impl LossKind {
    /// Compute gradient dL/dy_hat (before applying activation derivative)
    pub fn gradient(&self, y_hat: &Matrix<f64>, y: &Matrix<f64>) -> Matrix<f64> {
        let m = y.rows() as f64;
        match self {
            LossKind::MSE => (y_hat - y) * (2.0 / m),
            LossKind::BCE => (y_hat - y) * (1.0 / m),
        }
    }
}

/// Configuration for a dense neural network
pub struct DenseNNConfig {
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    /// Must have length = hidden_layers.len() + 1
    pub activations: Vec<ActivationKind>,
    pub output_size: usize,
    pub initializer: InitializerKind,
    pub loss: LossKind,
    pub learning_rate: f64,
    pub epochs: usize,
}

/// A multi-layer perceptron with full configurability
pub struct DenseNN {
    weights: Vec<Matrix<f64>>,
    biases: Vec<Matrix<f64>>,
    activations: Vec<ActivationKind>,
    loss: LossKind,
    lr: f64,
    epochs: usize,
}

impl DenseNN {
    /// Build a new DenseNN from the given configuration
    pub fn new(config: DenseNNConfig) -> Self {
        let mut sizes = vec![config.input_size];
        sizes.extend(&config.hidden_layers);
        sizes.push(config.output_size);

        assert_eq!(
            config.activations.len(),
            sizes.len() - 1,
            "Number of activation functions must match number of layers"
        );

        let mut weights = Vec::with_capacity(sizes.len() - 1);
        let mut biases = Vec::with_capacity(sizes.len() - 1);

        for i in 0..sizes.len() - 1 {
            let w = config.initializer.initialize(sizes[i], sizes[i + 1]);
            let b = Matrix::zeros(1, sizes[i + 1]);
            weights.push(w);
            biases.push(b);
        }

        DenseNN {
            weights,
            biases,
            activations: config.activations,
            loss: config.loss,
            lr: config.learning_rate,
            epochs: config.epochs,
        }
    }

    /// Perform a full forward pass, returning pre-activations (z) and activations (a)
    fn forward_full(&self, x: &Matrix<f64>) -> (Vec<Matrix<f64>>, Vec<Matrix<f64>>) {
        let mut zs = Vec::with_capacity(self.weights.len());
        let mut activs = Vec::with_capacity(self.weights.len() + 1);
        activs.push(x.clone());

        let mut a = x.clone();
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let z = &a.dot(w) + &Matrix::repeat_rows(b, a.rows());
            let a_next = self.activations[i].forward(&z);
            zs.push(z);
            activs.push(a_next.clone());
            a = a_next;
        }

        (zs, activs)
    }

    /// Train the network on inputs X and targets Y
    pub fn train(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) {
        let m = x.rows() as f64;
        for _ in 0..self.epochs {
            let (zs, activs) = self.forward_full(x);
            let y_hat = activs.last().unwrap().clone();

            // Initial delta (dL/dz) on output
            let mut delta = match self.loss {
                LossKind::BCE => self.loss.gradient(&y_hat, y),
                LossKind::MSE => {
                    let grad = self.loss.gradient(&y_hat, y);
                    let dz = self
                        .activations
                        .last()
                        .unwrap()
                        .derivative(zs.last().unwrap());
                    grad.zip(&dz, |g, da| g * da)
                }
            };

            // Backpropagate through layers
            for l in (0..self.weights.len()).rev() {
                let a_prev = &activs[l];
                let dw = a_prev.transpose().dot(&delta) / m;
                let db = Matrix::from_vec(delta.sum_vertical(), 1, delta.cols()) / m;

                // Update weights & biases
                self.weights[l] = &self.weights[l] - &(dw * self.lr);
                self.biases[l] = &self.biases[l] - &(db * self.lr);

                // Propagate delta to previous layer
                if l > 0 {
                    let w_t = self.weights[l].transpose();
                    let da = self.activations[l - 1].derivative(&zs[l - 1]);
                    delta = delta.dot(&w_t).zip(&da, |d, a| d * a);
                }
            }
        }
    }

    /// Run a forward pass and return the network's output
    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let mut a = x.clone();
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let z = &a.dot(w) + &Matrix::repeat_rows(b, a.rows());
            a = self.activations[i].forward(&z);
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    /// Compute MSE = 1/m * Σ (ŷ - y)²
    fn mse_loss(y_hat: &Matrix<f64>, y: &Matrix<f64>) -> f64 {
        let m = y.rows() as f64;
        y_hat
            .zip(y, |yh, yv| (yh - yv).powi(2))
            .data()
            .iter()
            .sum::<f64>()
            / m
    }

    #[test]
    fn test_predict_shape() {
        let config = DenseNNConfig {
            input_size: 1,
            hidden_layers: vec![2],
            activations: vec![ActivationKind::Relu, ActivationKind::Sigmoid],
            output_size: 1,
            initializer: InitializerKind::Uniform(0.1),
            loss: LossKind::MSE,
            learning_rate: 0.01,
            epochs: 0,
        };
        let model = DenseNN::new(config);
        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1);
        let preds = model.predict(&x);
        assert_eq!(preds.rows(), 3);
        assert_eq!(preds.cols(), 1);
    }

    #[test]
    fn test_train_no_epochs_does_nothing() {
        let config = DenseNNConfig {
            input_size: 1,
            hidden_layers: vec![2],
            activations: vec![ActivationKind::Relu, ActivationKind::Sigmoid],
            output_size: 1,
            initializer: InitializerKind::Uniform(0.1),
            loss: LossKind::MSE,
            learning_rate: 0.01,
            epochs: 0,
        };
        let mut model = DenseNN::new(config);
        let x = Matrix::from_vec(vec![0.0, 1.0], 2, 1);
        let y = Matrix::from_vec(vec![0.0, 1.0], 2, 1);

        let before = model.predict(&x);
        model.train(&x, &y);
        let after = model.predict(&x);

        for i in 0..before.rows() {
            for j in 0..before.cols() {
                // "prediction changed despite 0 epochs"
                assert!((before[(i, j)] - after[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_train_one_epoch_changes_predictions() {
        // Single-layer sigmoid regression so gradients flow.
        let config = DenseNNConfig {
            input_size: 1,
            hidden_layers: vec![],
            activations: vec![ActivationKind::Sigmoid],
            output_size: 1,
            initializer: InitializerKind::Uniform(0.1),
            loss: LossKind::MSE,
            learning_rate: 1.0,
            epochs: 1,
        };
        let mut model = DenseNN::new(config);

        let x = Matrix::from_vec(vec![0.0, 1.0], 2, 1);
        let y = Matrix::from_vec(vec![0.0, 1.0], 2, 1);

        let before = model.predict(&x);
        model.train(&x, &y);
        let after = model.predict(&x);

        // At least one of the two outputs must move by >ϵ
        let mut moved = false;
        for i in 0..before.rows() {
            if (before[(i, 0)] - after[(i, 0)]).abs() > 1e-8 {
                moved = true;
            }
        }
        assert!(moved, "predictions did not change after 1 epoch");
    }

    #[test]
    fn test_training_reduces_mse_loss() {
        // Same single‐layer sigmoid setup; check loss goes down.
        let config = DenseNNConfig {
            input_size: 1,
            hidden_layers: vec![],
            activations: vec![ActivationKind::Sigmoid],
            output_size: 1,
            initializer: InitializerKind::Uniform(0.1),
            loss: LossKind::MSE,
            learning_rate: 1.0,
            epochs: 10,
        };
        let mut model = DenseNN::new(config);

        let x = Matrix::from_vec(vec![0.0, 1.0, 0.5], 3, 1);
        let y = Matrix::from_vec(vec![0.0, 1.0, 0.5], 3, 1);

        let before_preds = model.predict(&x);
        let before_loss = mse_loss(&before_preds, &y);

        model.train(&x, &y);

        let after_preds = model.predict(&x);
        let after_loss = mse_loss(&after_preds, &y);

        // MSE did not decrease (before: {}, after: {})
        assert!(after_loss < before_loss);
    }

    #[test]
    fn test_activation_kind_forward_tanh() {
        let input = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![-0.76159415595, 0.0, 0.76159415595], 3, 1);
        let output = ActivationKind::Tanh.forward(&input);

        for i in 0..input.rows() {
            for j in 0..input.cols() {
                // Tanh forward output mismatch at ({}, {})
                assert!((output[(i, j)] - expected[(i, j)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_activation_kind_derivative_relu() {
        let input = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.0, 0.0, 1.0], 3, 1);
        let output = ActivationKind::Relu.derivative(&input);

        for i in 0..input.rows() {
            for j in 0..input.cols() {
                // "ReLU derivative output mismatch at ({}, {})"
                assert!((output[(i, j)] - expected[(i, j)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_activation_kind_derivative_tanh() {
        let input = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.41997434161, 1.0, 0.41997434161], 3, 1); // 1 - tanh(x)^2
        let output = ActivationKind::Tanh.derivative(&input);

        for i in 0..input.rows() {
            for j in 0..input.cols() {
                // "Tanh derivative output mismatch at ({}, {})"
                assert!((output[(i, j)] - expected[(i, j)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_initializer_kind_xavier() {
        let rows = 10;
        let cols = 20;
        let initializer = InitializerKind::Xavier;
        let matrix = initializer.initialize(rows, cols);
        let limit = (6.0 / (rows + cols) as f64).sqrt();

        assert_eq!(matrix.rows(), rows);
        assert_eq!(matrix.cols(), cols);

        for val in matrix.data() {
            // Xavier initialized value out of range
            assert!(*val >= -limit && *val <= limit);
        }
    }

    #[test]
    fn test_initializer_kind_he() {
        let rows = 10;
        let cols = 20;
        let initializer = InitializerKind::He;
        let matrix = initializer.initialize(rows, cols);
        let limit = (2.0 / rows as f64).sqrt();

        assert_eq!(matrix.rows(), rows);
        assert_eq!(matrix.cols(), cols);

        for val in matrix.data() {
            // He initialized value out of range
            assert!(*val >= -limit && *val <= limit);
        }
    }

    #[test]
    fn test_loss_kind_bce_gradient() {
        let y_hat = Matrix::from_vec(vec![0.1, 0.9, 0.4], 3, 1);
        let y = Matrix::from_vec(vec![0.0, 1.0, 0.5], 3, 1);
        let expected_gradient = Matrix::from_vec(vec![0.1 / 3.0, -0.1 / 3.0, -0.1 / 3.0], 3, 1); // (y_hat - y) * (1.0 / m)
        let output_gradient = LossKind::BCE.gradient(&y_hat, &y);

        assert_eq!(output_gradient.rows(), expected_gradient.rows());
        assert_eq!(output_gradient.cols(), expected_gradient.cols());

        for i in 0..output_gradient.rows() {
            for j in 0..output_gradient.cols() {
                // BCE gradient output mismatch at ({}, {})
                assert!((output_gradient[(i, j)] - expected_gradient[(i, j)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_training_reduces_bce_loss() {
        // Single-layer sigmoid setup; check BCE loss goes down.
        let config = DenseNNConfig {
            input_size: 1,
            hidden_layers: vec![],
            activations: vec![ActivationKind::Sigmoid],
            output_size: 1,
            initializer: InitializerKind::Uniform(0.1),
            loss: LossKind::BCE,
            learning_rate: 1.0,
            epochs: 10,
        };
        let mut model = DenseNN::new(config);

        let x = Matrix::from_vec(vec![0.0, 1.0, 0.5], 3, 1);
        let y = Matrix::from_vec(vec![0.0, 1.0, 0.5], 3, 1);

        let before_preds = model.predict(&x);
        // BCE loss calculation for testing
        let before_loss = -1.0 / (y.rows() as f64)
            * before_preds
                .zip(&y, |yh, yv| yv * yh.ln() + (1.0 - yv) * (1.0 - yh).ln())
                .data()
                .iter()
                .sum::<f64>();

        model.train(&x, &y);

        let after_preds = model.predict(&x);
        let after_loss = -1.0 / (y.rows() as f64)
            * after_preds
                .zip(&y, |yh, yv| yv * yh.ln() + (1.0 - yv) * (1.0 - yh).ln())
                .data()
                .iter()
                .sum::<f64>();

        // BCE did not decrease (before: {}, after: {})
        assert!(after_loss < before_loss,);
    }

    #[test]
    fn test_train_backprop_delta_propagation() {
        // Network with two layers to test delta propagation to previous layer (l > 0)
        let config = DenseNNConfig {
            input_size: 2,
            hidden_layers: vec![3],
            activations: vec![ActivationKind::Sigmoid, ActivationKind::Sigmoid],
            output_size: 1,
            initializer: InitializerKind::Uniform(0.1),
            loss: LossKind::MSE,
            learning_rate: 0.1,
            epochs: 1,
        };
        let mut model = DenseNN::new(config);

        // Store initial weights and biases to compare after training
        let initial_weights_l0 = model.weights[0].clone();
        let initial_biases_l0 = model.biases[0].clone();
        let initial_weights_l1 = model.weights[1].clone();
        let initial_biases_l1 = model.biases[1].clone();

        let x = Matrix::from_vec(vec![0.1, 0.2, 0.3, 0.4], 2, 2);
        let y = Matrix::from_vec(vec![0.5, 0.6], 2, 1);

        model.train(&x, &y);

        // Verify that weights and biases of both layers have changed,
        // implying delta propagation occurred for l > 0

        
        // Weights of first layer did not change, delta propagation might not have occurred
        assert!(model.weights[0] != initial_weights_l0);
        // Biases of first layer did not change, delta propagation might not have occurred
        assert!(model.biases[0] != initial_biases_l0);
        // Weights of second layer did not change
        assert!(model.weights[1] != initial_weights_l1);
        // Biases of second layer did not change
        assert!(model.biases[1] != initial_biases_l1);
    }
}
