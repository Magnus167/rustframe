use crate::matrix::{Matrix, SeriesOps};
use rand;

/// Returns the `n_components` principal axes (rows) and the centred data’s mean.
pub struct PCA {
    pub components: Matrix<f64>,   // (n_components, n_features)
    pub mean:       Matrix<f64>,   // (1, n_features)
}

impl PCA {
    pub fn fit(x: &Matrix<f64>, n_components: usize, iters: usize) -> Self {
        let m = x.rows();
        let n = x.cols();
        assert!(n_components <= n);

        // ----- centre data -----
        let mean_vec = {
            let mut v = Matrix::zeros(1, n);
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += x[(i, j)];
                }
                v[(0, j)] = s / m as f64;
            }
            v
        };
        let x_centered = x - &mean_vec;

        // ----- covariance matrix C = Xᵀ·X / (m-1) -----
        let cov = x_centered.transpose().dot(&x_centered) * (1.0 / (m as f64 - 1.0));

        // ----- power iteration to find top eigenvectors -----
        let mut comp = Matrix::zeros(n_components, n);
        let mut b = Matrix::zeros(1, n);            // current vector
        for c in 0..n_components {
            // random initial vector
            for j in 0..n {
                b[(0, j)] = rand::random::<f64>() - 0.5;
            }
            // subtract projections on previously found components
            for prev in 0..c {
                // let proj = b.dot(Matrix::from_vec(data, rows, cols).transpose())[(0, 0)];
                // let proj = b.dot(&comp.row(prev).transpose())[(0, 0)];
                let proj = b.dot(&Matrix::from_vec(comp.row(prev).to_vec(), 1, n).transpose())[(0, 0)];
                // subtract projection to maintain orthogonality
                for j in 0..n {
                    b[(0, j)] -= proj * comp[(prev, j)];
                }
            }
            // iterate
            for _ in 0..iters {
                // b = C·bᵀ
                let mut nb = cov.dot(&b.transpose()).transpose();
                // subtract projections again to maintain orthogonality
                for prev in 0..c {
                    let proj = nb.dot(&Matrix::from_vec(comp.row(prev).to_vec(), 1, n).transpose())[(0, 0)];
                    for j in 0..n {
                        nb[(0, j)] -= proj * comp[(prev, j)];
                    }
                }
                // normalise
                let norm = nb.data().iter().map(|v| v * v).sum::<f64>().sqrt();
                for j in 0..n {
                    nb[(0, j)] /= norm;
                }
                b = nb;
            }
            // store component
            for j in 0..n {
                comp[(c, j)] = b[(0, j)];
            }
        }
        Self {
            components: comp,
            mean: mean_vec,
        }
    }

    /// Project new data on the learned axes.
    pub fn transform(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let x_centered = x - &self.mean;
        x_centered.dot(&self.components.transpose())
    }
}
