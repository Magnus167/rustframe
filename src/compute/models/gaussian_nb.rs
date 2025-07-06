use crate::matrix::Matrix;
use std::collections::HashMap;

/// A Gaussian Naive Bayes classifier.
///
/// # Parameters
/// - `var_smoothing`: Portion of the largest variance of all features to add to variances for stability.
/// - `use_unbiased_variance`: If `true`, uses Bessel's correction (dividing by (n-1)); otherwise divides by n.
///
pub struct GaussianNB {
    // Distinct class labels
    classes: Vec<f64>,
    // Prior probabilities P(class)
    priors: Vec<f64>,
    // Feature means per class
    means: Vec<Matrix<f64>>,
    // Feature variances per class
    variances: Vec<Matrix<f64>>,
    // var_smoothing
    eps: f64,
    // flag for unbiased variance
    use_unbiased: bool,
}

impl GaussianNB {
    /// Create a new GaussianNB.
    ///
    /// # Arguments
    /// * `var_smoothing` - small float added to variances for numerical stability.
    /// * `use_unbiased_variance` - whether to apply Bessel's correction (divide by n-1).
    pub fn new(var_smoothing: f64, use_unbiased_variance: bool) -> Self {
        Self {
            classes: Vec::new(),
            priors: Vec::new(),
            means: Vec::new(),
            variances: Vec::new(),
            eps: var_smoothing,
            use_unbiased: use_unbiased_variance,
        }
    }

    /// Fit the model according to the training data `x` and labels `y`.
    ///
    /// # Panics
    /// Panics if `x` or `y` is empty, or if their dimensions disagree.
    pub fn fit(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) {
        let m = x.rows();
        let n = x.cols();
        assert_eq!(y.rows(), m, "Row count of X and Y must match");
        assert_eq!(y.cols(), 1, "Y must be a column vector");
        if m == 0 || n == 0 {
            panic!("Input matrix x or y is empty");
        }

        // Group sample indices by label
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for i in 0..m {
            let label = y[(i, 0)];
            let bits = label.to_bits();
            groups.entry(bits).or_default().push(i);
        }
        if groups.is_empty() {
            panic!("No class labels found in y");
        }

        // Extract and sort class labels
        self.classes = groups.keys().cloned().map(f64::from_bits).collect();
        self.classes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.priors.clear();
        self.means.clear();
        self.variances.clear();

        // Precompute max variance for smoothing scale
        let mut max_var_feature = 0.0;
        for j in 0..n {
            let mut col_vals = Vec::with_capacity(m);
            for i in 0..m {
                col_vals.push(x[(i, j)]);
            }
            let mean_all = col_vals.iter().sum::<f64>() / m as f64;
            let var_all = col_vals.iter().map(|v| (v - mean_all).powi(2)).sum::<f64>() / m as f64;
            if var_all > max_var_feature {
                max_var_feature = var_all;
            }
        }
        let smoothing = self.eps * max_var_feature;

        // Compute per-class statistics
        for &c in &self.classes {
            let idx = &groups[&c.to_bits()];
            let count = idx.len();
            if count == 0 {
                panic!("Class group for label {} is empty", c);
            }
            // Prior
            self.priors.push(count as f64 / m as f64);

            let mut mean = Matrix::zeros(1, n);
            let mut var = Matrix::zeros(1, n);

            // Mean
            for &i in idx {
                for j in 0..n {
                    mean[(0, j)] += x[(i, j)];
                }
            }
            for j in 0..n {
                mean[(0, j)] /= count as f64;
            }

            // Variance
            for &i in idx {
                for j in 0..n {
                    let d = x[(i, j)] - mean[(0, j)];
                    var[(0, j)] += d * d;
                }
            }
            let denom = if self.use_unbiased {
                (count as f64 - 1.0).max(1.0)
            } else {
                count as f64
            };
            for j in 0..n {
                var[(0, j)] = var[(0, j)] / denom + smoothing;
                if var[(0, j)] <= 0.0 {
                    var[(0, j)] = smoothing;
                }
            }

            self.means.push(mean);
            self.variances.push(var);
        }
    }

    /// Perform classification on an array of test vectors `x`.
    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let m = x.rows();
        let n = x.cols();
        let k = self.classes.len();
        let mut preds = Matrix::zeros(m, 1);
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();

        for i in 0..m {
            let mut best = (0, f64::NEG_INFINITY);
            for c_idx in 0..k {
                let mut log_prob = self.priors[c_idx].ln();
                for j in 0..n {
                    let diff = x[(i, j)] - self.means[c_idx][(0, j)];
                    let var = self.variances[c_idx][(0, j)];
                    log_prob += -0.5 * (diff * diff / var + var.ln() + ln_2pi);
                }
                if log_prob > best.1 {
                    best = (c_idx, log_prob);
                }
            }
            preds[(i, 0)] = self.classes[best.0];
        }
        preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn test_simple_two_class() {
        // Simple dataset: one feature, two classes 0 and 1
        // Class 0: values [1.0, 1.2, 0.8]
        // Class 1: values [3.0, 3.2, 2.8]
        let x = Matrix::from_vec(vec![1.0, 1.2, 0.8, 3.0, 3.2, 2.8], 6, 1);
        let y = Matrix::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 6, 1);
        let mut clf = GaussianNB::new(1e-9, false);
        clf.fit(&x, &y);
        let test = Matrix::from_vec(vec![1.1, 3.1], 2, 1);
        let preds = clf.predict(&test);
        assert_eq!(preds[(0, 0)], 0.0);
        assert_eq!(preds[(1, 0)], 1.0);
    }

    #[test]
    fn test_unbiased_variance() {
        // Same as above but with unbiased variance
        let x = Matrix::from_vec(vec![2.0, 2.2, 1.8, 4.0, 4.2, 3.8], 6, 1);
        let y = Matrix::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 6, 1);
        let mut clf = GaussianNB::new(1e-9, true);
        clf.fit(&x, &y);
        let test = Matrix::from_vec(vec![2.1, 4.1], 2, 1);
        let preds = clf.predict(&test);
        assert_eq!(preds[(0, 0)], 0.0);
        assert_eq!(preds[(1, 0)], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_empty_input() {
        let x = Matrix::zeros(0, 0);
        let y = Matrix::zeros(0, 1);
        let mut clf = GaussianNB::new(1e-9, false);
        clf.fit(&x, &y);
    }

    #[test]
    #[should_panic = "Row count of X and Y must match"]
    fn test_mismatched_rows() {
        let x = Matrix::from_vec(vec![1.0, 2.0], 2, 1);
        let y = Matrix::from_vec(vec![0.0], 1, 1);
        let mut clf = GaussianNB::new(1e-9, false);
        clf.fit(&x, &y);
        clf.predict(&x);
    }
}
