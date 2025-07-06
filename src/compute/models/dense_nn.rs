use crate::matrix::{Matrix, SeriesOps};
use crate::compute::activations::{relu, sigmoid, drelu};  
use rand::Rng;

pub struct DenseNN {
    w1: Matrix<f64>, // (n_in, n_hidden)
    b1: Matrix<f64>, // (1,    n_hidden)
    w2: Matrix<f64>, // (n_hidden, n_out)
    b2: Matrix<f64>, // (1,    n_out)
}

impl DenseNN {
    pub fn new(n_in: usize, n_hidden: usize, n_out: usize) -> Self {
        let mut rng = rand::rng();
        let mut init = |rows, cols| {
            let data = (0..rows * cols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect::<Vec<_>>();
            Matrix::from_vec(data, rows, cols)
        };
        Self {
            w1: init(n_in, n_hidden),
            b1: Matrix::zeros(1, n_hidden),
            w2: init(n_hidden, n_out),
            b2: Matrix::zeros(1, n_out),
        }
    }

    pub fn forward(&self, x: &Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
        // z1 = X·W1 + b1 ; a1 = ReLU(z1)
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = relu(&z1);
        // z2 = a1·W2 + b2 ; a2 = softmax(z2) (here binary => sigmoid)
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = sigmoid(&z2); // binary output
        (a1, z2, a2)           // keep intermediates for back-prop
    }

    pub fn train(&mut self, x: &Matrix<f64>, y: &Matrix<f64>, lr: f64, epochs: usize) {
        let m = x.rows() as f64;
        for _ in 0..epochs {
            let (a1, _z2, y_hat) = self.forward(x);

            // -------- backwards ----------
            // dL/da2 = y_hat - y  (BCE derivative)
            let dz2 = &y_hat - y;                           // (m, n_out)
            let dw2 = a1.transpose().dot(&dz2) / m;         // (n_h, n_out)
            // let db2 = dz2.sum_vertical() * (1.0 / m);       // broadcast ok
            let db2 = Matrix::from_vec(dz2.sum_vertical(), 1, dz2.cols()) * (1.0 / m); // (1, n_out)
            let da1 = dz2.dot(&self.w2.transpose());        // (m,n_h)
            let dz1 = da1.zip(&a1, |g, act| g * drelu(&Matrix::from_cols(vec![vec![act]])).data()[0]); // (m,n_h)
            
            // real code: drelu returns Matrix, broadcasting needed; you can optimise.

            let dw1 = x.transpose().dot(&dz1) / m;          // (n_in,n_h)
            let db1 = Matrix::from_vec(dz1.sum_vertical(), 1, dz1.cols()) * (1.0 / m); // (1, n_h)

            // -------- update ----------
            self.w2 = &self.w2 - &(dw2 * lr);
            self.b2 = &self.b2 - &(db2 * lr);
            self.w1 = &self.w1 - &(dw1 * lr);
            self.b1 = &self.b1 - &(db1 * lr);
        }
    }
}
