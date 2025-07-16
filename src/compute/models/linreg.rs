use crate::matrix::{Matrix, SeriesOps};

pub struct LinReg {
    w: Matrix<f64>, // shape (n_features, 1)
    b: f64,
}

impl LinReg {
    pub fn new(n_features: usize) -> Self {
        Self {
            w: Matrix::from_vec(vec![0.0; n_features], n_features, 1),
            b: 0.0,
        }
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        // X.dot(w) + b
        x.dot(&self.w) + self.b
    }

    pub fn fit(&mut self, x: &Matrix<f64>, y: &Matrix<f64>, lr: f64, epochs: usize) {
        let m = x.rows() as f64;
        for _ in 0..epochs {
            let y_hat = self.predict(x);
            let err = &y_hat - y; // shape (m,1)

            // grads
            let grad_w = x.transpose().dot(&err) * (2.0 / m); // (n,1)
            let grad_b = (2.0 / m) * err.sum_vertical().iter().sum::<f64>();
            // update
            self.w = &self.w - &(grad_w * lr);
            self.b -= lr * grad_b;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_linreg_fit_predict() {
        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let y = Matrix::from_vec(vec![2.0, 3.0, 4.0, 5.0], 4, 1);
        let mut model = LinReg::new(1);
        model.fit(&x, &y, 0.01, 10000);
        let preds = model.predict(&x);
        assert!((preds[(0, 0)] - 2.0).abs() < 1e-2);
        assert!((preds[(1, 0)] - 3.0).abs() < 1e-2);
        assert!((preds[(2, 0)] - 4.0).abs() < 1e-2);
        assert!((preds[(3, 0)] - 5.0).abs() < 1e-2);
    }
}
