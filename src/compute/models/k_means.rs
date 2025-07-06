use crate::matrix::Matrix;
use rand::seq::SliceRandom;

pub struct KMeans {
    pub centroids: Matrix<f64>, // (k, n_features)
}

impl KMeans {
    /// Fit with k clusters.
    pub fn fit(x: &Matrix<f64>, k: usize, max_iter: usize, tol: f64) -> (Self, Vec<usize>) {
        let m = x.rows();
        let n = x.cols();
        assert!(k <= m, "k must be ≤ number of samples");

        // ----- initialise centroids: pick k distinct rows at random -----
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..m).collect();
        indices.shuffle(&mut rng);
        let mut centroids = Matrix::zeros(k, n);
        for (c, &i) in indices[..k].iter().enumerate() {
            for j in 0..n {
                centroids[(c, j)] = x[(i, j)];
            }
        }

        let mut labels = vec![0usize; m];
        for _ in 0..max_iter {
            // ----- assignment step -----
            let mut changed = false;
            for i in 0..m {
                let mut best = 0usize;
                let mut best_dist = f64::MAX;
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..n {
                        let d = x[(i, j)] - centroids[(c, j)];
                        dist += d * d;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best = c;
                    }
                }
                if labels[i] != best {
                    labels[i] = best;
                    changed = true;
                }
            }

            // ----- update step -----
            let mut counts = vec![0usize; k];
            let mut centroids = Matrix::zeros(k, n);
            for i in 0..m {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..n {
                    centroids[(c, j)] += x[(i, j)];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..n {
                        centroids[(c, j)] /= counts[c] as f64;
                    }
                }
            }

            // ----- convergence test -----
            if !changed {
                break; // assignments stable
            }
            if tol > 0.0 {
                // optional centroid-shift tolerance
                let mut shift: f64 = 0.0;
                for c in 0..k {
                    for j in 0..n {
                        let d = centroids[(c, j)] - centroids[(c, j)]; // previous stored?
                        shift += d * d;
                    }
                }
                if shift.sqrt() < tol {
                    break;
                }
            }
        }
        (Self { centroids }, labels)
    }

    /// Predict nearest centroid for each sample.
    pub fn predict(&self, x: &Matrix<f64>) -> Vec<usize> {
        let m = x.rows();
        let k = self.centroids.rows();
        let n = x.cols();
        let mut labels = vec![0usize; m];
        for i in 0..m {
            let mut best = 0usize;
            let mut best_dist = f64::MAX;
            for c in 0..k {
                let mut dist = 0.0;
                for j in 0..n {
                    let d = x[(i, j)] - self.centroids[(c, j)];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            labels[i] = best;
        }
        labels
    }
}
