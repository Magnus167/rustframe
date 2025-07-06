use crate::compute::activations::sigmoid;
use crate::matrix::{Matrix, SeriesOps};

pub struct LogReg {
    w: Matrix<f64>,
    b: f64,
}

impl LogReg {
    pub fn new(n_features: usize) -> Self {
        Self {
            w: Matrix::zeros(n_features, 1),
            b: 0.0,
        }
    }

    pub fn predict_proba(&self, x: &Matrix<f64>) -> Matrix<f64> {
        sigmoid(&(x.dot(&self.w) + self.b)) // σ(Xw + b)
    }

    pub fn fit(&mut self, x: &Matrix<f64>, y: &Matrix<f64>, lr: f64, epochs: usize) {
        let m = x.rows() as f64;
        for _ in 0..epochs {
            let p = self.predict_proba(x); // shape (m,1)
            let err = &p - y; // derivative of BCE wrt pre-sigmoid
            let grad_w = x.transpose().dot(&err) / m;
            let grad_b = err.sum_vertical().iter().sum::<f64>() / m;
            self.w = &self.w - &(grad_w * lr);
            self.b -= lr * grad_b;
        }
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        self.predict_proba(x)
            .map(|p| if p >= 0.5 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logreg_fit_predict() {
        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let y = Matrix::from_vec(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
        let mut model = LogReg::new(1);
        model.fit(&x, &y, 0.01, 10000);
        let preds = model.predict(&x);
        assert_eq!(preds[(0, 0)], 0.0);
        assert_eq!(preds[(1, 0)], 0.0);
        assert_eq!(preds[(2, 0)], 1.0);
        assert_eq!(preds[(3, 0)], 1.0);
    }
}
