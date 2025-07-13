use crate::compute::stats::mean_vertical;
use crate::matrix::Matrix;
use rand::rng;
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

        // ----- initialise centroids -----
        let mut centroids = Matrix::zeros(k, n);
        if k > 0 && m > 0 {
            // case for empty data
            if k == 1 {
                let mean = mean_vertical(x);
                centroids.row_copy_from_slice(0, &mean.data()); // ideally, data.row(0), but thats the same
            } else {
                // For k > 1, pick k distinct rows at random
                let mut rng = rng();
                let mut indices: Vec<usize> = (0..m).collect();
                indices.shuffle(&mut rng);
                for c in 0..k {
                    centroids.row_copy_from_slice(c, &x.row(indices[c]));
                }
            }
        }

        let mut labels = vec![0usize; m];
        let mut distances = vec![0.0f64; m];

        for _iter in 0..max_iter {
            let mut changed = false;
            // ----- assignment step -----
            for i in 0..m {
                let sample_row = x.row(i);
                let mut best = 0usize;
                let mut best_dist_sq = f64::MAX;

                for c in 0..k {
                    let centroid_row = centroids.row(c);

                    let dist_sq: f64 = sample_row
                        .iter()
                        .zip(centroid_row.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best = c;
                    }
                }

                distances[i] = best_dist_sq;

                if labels[i] != best {
                    labels[i] = best;
                    changed = true;
                }
            }

            // ----- update step -----
            let mut new_centroids = Matrix::zeros(k, n);
            let mut counts = vec![0usize; k];
            for i in 0..m {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..n {
                    new_centroids[(c, j)] += x[(i, j)];
                }
            }

            for c in 0..k {
                if counts[c] == 0 {
                    // This cluster is empty. Re-initialize its centroid to the point
                    // furthest from its assigned centroid to prevent the cluster from dying.
                    let mut furthest_point_idx = 0;
                    let mut max_dist_sq = 0.0;
                    for (i, &dist) in distances.iter().enumerate() {
                        if dist > max_dist_sq {
                            max_dist_sq = dist;
                            furthest_point_idx = i;
                        }
                    }

                    for j in 0..n {
                        new_centroids[(c, j)] = x[(furthest_point_idx, j)];
                    }
                    // Ensure this point isn't chosen again for another empty cluster in the same iteration.
                    if m > 0 {
                        distances[furthest_point_idx] = 0.0;
                    }
                } else {
                    // Normalize the centroid by the number of points in it.
                    for j in 0..n {
                        new_centroids[(c, j)] /= counts[c] as f64;
                    }
                }
            }

            // ----- convergence test -----
            if !changed {
                centroids = new_centroids; //  update before breaking
                break; // assignments stable
            }

            let diff = &new_centroids - &centroids;
            centroids = new_centroids; // Update for the next iteration

            if tol > 0.0 {
                let sq_diff = &diff * &diff;
                let shift = sq_diff.data().iter().sum::<f64>().sqrt();
                if shift < tol {
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

        if m == 0 {
            return Vec::new();
        }

        let mut labels = vec![0usize; m];
        for i in 0..m {
            let sample_row = x.row(i);
            let mut best = 0usize;
            let mut best_dist_sq = f64::MAX;

            for c in 0..k {
                let centroid_row = self.centroids.row(c);

                let dist_sq: f64 = sample_row
                    .iter()
                    .zip(centroid_row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best = c;
                }
            }
            labels[i] = best;
        }
        labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FloatMatrix;

    fn create_test_data() -> (FloatMatrix, usize) {
        // Simple 2D data for testing K-Means
        // Cluster 1: (1,1), (1.5,1.5)
        // Cluster 2: (5,8), (8,8), (6,7)
        let data = vec![
            1.0, 1.0, // Sample 0
            1.5, 1.5, // Sample 1
            5.0, 8.0, // Sample 2
            8.0, 8.0, // Sample 3
            6.0, 7.0, // Sample 4
        ];
        let x = FloatMatrix::from_rows_vec(data, 5, 2);
        let k = 2;
        (x, k)
    }

    // Helper for single cluster test with exact mean
    fn create_simple_integer_data() -> FloatMatrix {
        // Data points: (1,1), (2,2), (3,3)
        FloatMatrix::from_rows_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 3, 2)
    }

    #[test]
    fn test_k_means_fit_predict_basic() {
        let (x, k) = create_test_data();
        let max_iter = 100;
        let tol = 1e-6;

        let (kmeans_model, labels) = KMeans::fit(&x, k, max_iter, tol);

        // Assertions for fit
        assert_eq!(kmeans_model.centroids.rows(), k);
        assert_eq!(kmeans_model.centroids.cols(), x.cols());
        assert_eq!(labels.len(), x.rows());

        // Check if labels are within expected range (0 to k-1)
        for &label in &labels {
            assert!(label < k);
        }

        // Predict with the same data
        let predicted_labels = kmeans_model.predict(&x);

        // The exact labels might vary due to random initialization,
        // but the clustering should be consistent.
        // We expect two clusters. Let's check if samples 0,1 are in one cluster
        // and samples 2,3,4 are in another.
        let cluster_0_members = vec![labels[0], labels[1]];
        let cluster_1_members = vec![labels[2], labels[3], labels[4]];

        // All members of cluster 0 should have the same label
        assert_eq!(cluster_0_members[0], cluster_0_members[1]);
        // All members of cluster 1 should have the same label
        assert_eq!(cluster_1_members[0], cluster_1_members[1]);
        assert_eq!(cluster_1_members[0], cluster_1_members[2]);
        // The two clusters should have different labels
        assert_ne!(cluster_0_members[0], cluster_1_members[0]);

        // Check predicted labels are consistent with fitted labels
        assert_eq!(labels, predicted_labels);

        // Test with a new sample
        let new_sample_data = vec![1.2, 1.3]; // Should be close to cluster 0
        let new_sample = FloatMatrix::from_rows_vec(new_sample_data, 1, 2);
        let new_sample_label = kmeans_model.predict(&new_sample)[0];
        assert_eq!(new_sample_label, cluster_0_members[0]);

        let new_sample_data_2 = vec![7.0, 7.5]; // Should be close to cluster 1
        let new_sample_2 = FloatMatrix::from_rows_vec(new_sample_data_2, 1, 2);
        let new_sample_label_2 = kmeans_model.predict(&new_sample_2)[0];
        assert_eq!(new_sample_label_2, cluster_1_members[0]);
    }

    #[test]
    fn test_k_means_fit_k_equals_m() {
        // Test case where k (number of clusters) equals m (number of samples)
        let (x, _) = create_test_data(); // 5 samples
        let k = 5; // 5 clusters
        let max_iter = 10;
        let tol = 1e-6;

        let (kmeans_model, labels) = KMeans::fit(&x, k, max_iter, tol);

        assert_eq!(kmeans_model.centroids.rows(), k);
        assert_eq!(labels.len(), x.rows());

        // Each sample should be its own cluster. Due to random init, labels
        // might not be [0,1,2,3,4] but will be a permutation of it.
        let mut sorted_labels = labels.clone();
        sorted_labels.sort_unstable();
        sorted_labels.dedup();
        // Labels should all be unique when k==m
        assert_eq!(sorted_labels.len(), k);
    }

    #[test]
    #[should_panic(expected = "k must be ≤ number of samples")]
    fn test_k_means_fit_k_greater_than_m() {
        let (x, _) = create_test_data(); // 5 samples
        let k = 6; // k > m
        let max_iter = 10;
        let tol = 1e-6;

        let (_kmeans_model, _labels) = KMeans::fit(&x, k, max_iter, tol);
    }

    #[test]
    fn test_k_means_fit_single_cluster() {
        // Test with k=1
        let x = create_simple_integer_data(); // Use integer data
        let k = 1;
        let max_iter = 100;
        let tol = 1e-6;

        let (kmeans_model, labels) = KMeans::fit(&x, k, max_iter, tol);

        assert_eq!(kmeans_model.centroids.rows(), 1);
        assert_eq!(labels.len(), x.rows());

        // All labels should be 0
        assert!(labels.iter().all(|&l| l == 0));

        // Centroid should be the mean of all data points
        let expected_centroid_x = x.column(0).iter().sum::<f64>() / x.rows() as f64;
        let expected_centroid_y = x.column(1).iter().sum::<f64>() / x.rows() as f64;

        assert!((kmeans_model.centroids[(0, 0)] - expected_centroid_x).abs() < 1e-9);
        assert!((kmeans_model.centroids[(0, 1)] - expected_centroid_y).abs() < 1e-9);
    }

    #[test]
    fn test_k_means_predict_empty_matrix() {
        let (x, k) = create_test_data();
        let max_iter = 10;
        let tol = 1e-6;
        let (kmeans_model, _labels) = KMeans::fit(&x, k, max_iter, tol);

        // The `Matrix` type not support 0xN or Nx0 matrices.
        // test with a 0x0 matrix is a valid edge case.
        let empty_x = FloatMatrix::from_rows_vec(vec![], 0, 0);
        let predicted_labels = kmeans_model.predict(&empty_x);
        assert!(predicted_labels.is_empty());
    }

    #[test]
    fn test_k_means_predict_single_sample() {
        let (x, k) = create_test_data();
        let max_iter = 10;
        let tol = 1e-6;
        let (kmeans_model, _labels) = KMeans::fit(&x, k, max_iter, tol);

        let single_sample = FloatMatrix::from_rows_vec(vec![1.1, 1.2], 1, 2);
        let predicted_label = kmeans_model.predict(&single_sample);
        assert_eq!(predicted_label.len(), 1);
        assert!(predicted_label[0] < k);
    }

    #[test]
    fn test_k_means_fit_empty_cluster_reinitialization() {
        // Create data where one cluster is likely to become empty
        // Two distinct groups of points, but we ask for 3 clusters.
        // This should cause one cluster to be empty and re-initialized.
        let data = vec![
            1.0, 1.0,
            1.1, 1.1,
            1.2, 1.2,
            // Large gap to ensure distinct clusters
            100.0, 100.0,
            100.1, 100.1,
            100.2, 100.2,
        ];
        let x = FloatMatrix::from_rows_vec(data, 6, 2);
        let k = 3; // Request 3 clusters for 2 natural groups
        let max_iter = 100;
        let tol = 1e-6;

        // The test aims to verify the empty cluster re-initialization logic.
        // With random initialization, it's highly probable that one of the
        // three requested clusters will initially be empty or become empty
        // during the first few iterations, triggering the re-initialization.

        let (kmeans_model, labels) = KMeans::fit(&x, k, max_iter, tol);

        assert_eq!(kmeans_model.centroids.rows(), k);
        assert_eq!(labels.len(), x.rows());

        // Verify that all labels are assigned and within bounds
        for &label in &labels {
            assert!(label < k);
        }

        // Count points assigned to each cluster
        let mut counts = vec![0; k];
        for &label in &labels {
            counts[label] += 1;
        }

        // The crucial assertion: After re-initialization, no cluster should remain empty.
        // This verifies that the "furthest point" logic successfully re-assigned a point
        // to the previously empty cluster.
        assert!(counts.iter().all(|&c| c > 0));

        // The crucial assertion: After re-initialization, no cluster should remain empty.
        // This verifies that the "furthest point" logic successfully re-assigned a point
        // to the previously empty cluster.
        assert!(counts.iter().all(|&c| c > 0));
    }
}
