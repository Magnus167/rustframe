use crate::matrix::{Matrix};
use std::collections::HashMap;

pub struct GaussianNB {
    classes: Vec<f64>, // distinct labels
    priors: Vec<f64>,  // P(class)
    means: Vec<Matrix<f64>>,
    variances: Vec<Matrix<f64>>,
    eps: f64, // var-smoothing
}

impl GaussianNB {
    pub fn new(var_smoothing: f64) -> Self {
        Self {
            classes: vec![],
            priors: vec![],
            means: vec![],
            variances: vec![],
            eps: var_smoothing,
        }
    }

    pub fn fit(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) {
        let m = x.rows();
        let n = x.cols();
        assert_eq!(y.rows(), m);
        assert_eq!(y.cols(), 1);
        if m == 0 || n == 0 {
            panic!("Input matrix x or y is empty");
        }

        // ----- group samples by label -----
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for i in 0..m {
            let label_bits = y[(i, 0)].to_bits();
            groups.entry(label_bits).or_default().push(i);
        }
        if groups.is_empty() {
            panic!("No class labels found in y");
        }

        self.classes = groups
            .keys()
            .cloned()
            .map(f64::from_bits)
            .collect::<Vec<_>>();
        // Note: If NaN is present in class labels, this may panic. Ensure labels are valid floats.
        self.classes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.priors.clear();
        self.means.clear();
        self.variances.clear();

        for &c in &self.classes {
            let label_bits = c.to_bits();
            let idx = &groups[&label_bits];
            let count = idx.len();
            if count == 0 {
                panic!("Class group for label {c} is empty");
            }
            self.priors.push(count as f64 / m as f64);

            let mut mean = Matrix::zeros(1, n);
            let mut var = Matrix::zeros(1, n);

            // mean
            for &i in idx {
                for j in 0..n {
                    mean[(0, j)] += x[(i, j)];
                }
            }
            for j in 0..n {
                mean[(0, j)] /= count as f64;
            }

            // variance
            for &i in idx {
                for j in 0..n {
                    let d = x[(i, j)] - mean[(0, j)];
                    var[(0, j)] += d * d;
                }
            }
            for j in 0..n {
                var[(0, j)] = var[(0, j)] / count as f64 + self.eps; // always add eps after division
                if var[(0, j)] <= 0.0 {
                    var[(0, j)] = self.eps; // ensure strictly positive variance
                }
            }

            self.means.push(mean);
            self.variances.push(var);
        }
    }

    /// Return class labels (shape m×1) for samples in X.
    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let m = x.rows();
        let k = self.classes.len();
        let n = x.cols();
        let mut preds = Matrix::zeros(m, 1);
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();

        for i in 0..m {
            let mut best_class = 0usize;
            let mut best_log_prob = f64::NEG_INFINITY;
            for c in 0..k {
                // log P(y=c) + Σ log N(x_j | μ, σ²)
                let mut log_prob = self.priors[c].ln();
                for j in 0..n {
                    let mean = self.means[c][(0, j)];
                    let var = self.variances[c][(0, j)];
                    let diff = x[(i, j)] - mean;
                    log_prob += -0.5 * (diff * diff / var + var.ln() + ln_2pi);
                }
                if log_prob > best_log_prob {
                    best_log_prob = log_prob;
                    best_class = c;
                }
            }
            preds[(i, 0)] = self.classes[best_class];
        }
        preds
    }
}
