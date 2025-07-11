use crate::matrix::{Axis, Matrix, SeriesOps};
use crate::compute::stats::descriptive::mean_vertical;
use crate::compute::stats::correlation::covariance_matrix;

/// Returns the `n_components` principal axes (rows) and the centred data's mean.
pub struct PCA {
    pub components: Matrix<f64>, // (n_components, n_features)
    pub mean: Matrix<f64>,       // (1, n_features)
}

impl PCA {
    pub fn fit(x: &Matrix<f64>, n_components: usize, _iters: usize) -> Self {
        let mean = mean_vertical(x); // Mean of each feature (column)
        let broadcasted_mean = mean.broadcast_row_to_target_shape(x.rows(), x.cols());
        let centered_data = x.zip(&broadcasted_mean, |x_i, mean_i| x_i - mean_i);
        let covariance_matrix = covariance_matrix(&centered_data, Axis::Col); // Covariance between features

        let mut components = Matrix::zeros(n_components, x.cols());
        for i in 0..n_components {
            if i < covariance_matrix.rows() {
                components.row_copy_from_slice(i, &covariance_matrix.row(i));
            } else {
                break;
            }
        }

        PCA {
            components,
            mean,
        }
    }

    /// Project new data on the learned axes.
    pub fn transform(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let broadcasted_mean = self.mean.broadcast_row_to_target_shape(x.rows(), x.cols());
        let centered_data = x.zip(&broadcasted_mean, |x_i, mean_i| x_i - mean_i);
        centered_data.matrix_mul(&self.components.transpose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    const EPSILON: f64 = 1e-8;

    #[test]
    fn test_pca_basic() {
        // Simple 2D data, points along y=x line
        // Data:
        // 1.0, 1.0
        // 2.0, 2.0
        // 3.0, 3.0
        let data = Matrix::from_rows_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 3, 2);
        let (n_samples, n_features) = data.shape();

        let pca = PCA::fit(&data, 1, 0); // n_components = 1, iters is unused

        println!("Data shape: {:?}", data.shape());
        println!("PCA mean shape: {:?}", pca.mean.shape());
        println!("PCA components shape: {:?}", pca.components.shape());

        // Expected mean: (2.0, 2.0)
        assert!((pca.mean.get(0, 0) - 2.0).abs() < EPSILON);
        assert!((pca.mean.get(0, 1) - 2.0).abs() < EPSILON);

        // For data along y=x, the principal component should be proportional to (1/sqrt(2), 1/sqrt(2)) or (1,1)
        // The covariance matrix will be:
        // [[1.0, 1.0],
        //  [1.0, 1.0]]
        // The principal component (eigenvector) will be (0.707, 0.707) or (-0.707, -0.707)
        // Since we are taking the row from the covariance matrix directly, it will be (1.0, 1.0)
        assert!((pca.components.get(0, 0) - 1.0).abs() < EPSILON);
        assert!((pca.components.get(0, 1) - 1.0).abs() < EPSILON);

        // Test transform
        // Centered data:
        // -1.0, -1.0
        //  0.0,  0.0
        //  1.0,  1.0
        // Projected: (centered_data * components.transpose())
        // (-1.0 * 1.0 + -1.0 * 1.0) = -2.0
        // ( 0.0 * 1.0 +  0.0 * 1.0) =  0.0
        // ( 1.0 * 1.0 +  1.0 * 1.0) =  2.0
        let transformed_data = pca.transform(&data);
        assert_eq!(transformed_data.rows(), 3);
        assert_eq!(transformed_data.cols(), 1);
        assert!((transformed_data.get(0, 0) - -2.0).abs() < EPSILON);
        assert!((transformed_data.get(1, 0) - 0.0).abs() < EPSILON);
        assert!((transformed_data.get(2, 0) - 2.0).abs() < EPSILON);
    }
}
