use crate::compute::stats::{mean, mean_horizontal, mean_vertical};
use crate::matrix::{Axis, Matrix, SeriesOps};

/// Population covariance between two equally-sized matrices (flattened)
pub fn covariance(x: &Matrix<f64>, y: &Matrix<f64>) -> f64 {
    assert_eq!(x.rows(), y.rows());
    assert_eq!(x.cols(), y.cols());

    let n = (x.rows() * x.cols()) as f64;
    let mean_x = mean(x);
    let mean_y = mean(y);

    x.data()
        .iter()
        .zip(y.data().iter())
        .map(|(&a, &b)| (a - mean_x) * (b - mean_y))
        .sum::<f64>()
        / n
}

fn _covariance_axis(x: &Matrix<f64>, axis: Axis) -> Matrix<f64> {
    match axis {
        Axis::Row => {
            // Covariance between each pair of columns → cols x cols
            let num_rows = x.rows() as f64;
            let means = mean_vertical(x); // 1 x cols
            let p = x.cols();
            let mut data = vec![0.0; p * p];

            for i in 0..p {
                let mu_i = means.get(0, i);
                for j in 0..p {
                    let mu_j = means.get(0, j);
                    let mut sum = 0.0;
                    for r in 0..x.rows() {
                        let d_i = x.get(r, i) - mu_i;
                        let d_j = x.get(r, j) - mu_j;
                        sum += d_i * d_j;
                    }
                    data[i * p + j] = sum / num_rows;
                }
            }

            Matrix::from_vec(data, p, p)
        }
        Axis::Col => {
            // Covariance between each pair of rows → rows x rows
            let num_cols = x.cols() as f64;
            let means = mean_horizontal(x); // rows x 1
            let n = x.rows();
            let mut data = vec![0.0; n * n];

            for i in 0..n {
                let mu_i = means.get(i, 0);
                for j in 0..n {
                    let mu_j = means.get(j, 0);
                    let mut sum = 0.0;
                    for c in 0..x.cols() {
                        let d_i = x.get(i, c) - mu_i;
                        let d_j = x.get(j, c) - mu_j;
                        sum += d_i * d_j;
                    }
                    data[i * n + j] = sum / num_cols;
                }
            }

            Matrix::from_vec(data, n, n)
        }
    }
}

/// Covariance between columns (i.e. across rows)
pub fn covariance_vertical(x: &Matrix<f64>) -> Matrix<f64> {
    _covariance_axis(x, Axis::Row)
}

/// Covariance between rows (i.e. across columns)
pub fn covariance_horizontal(x: &Matrix<f64>) -> Matrix<f64> {
    _covariance_axis(x, Axis::Col)
}

/// Calculates the covariance matrix of the input data.
/// Assumes input `x` is (n_samples, n_features).
pub fn covariance_matrix(x: &Matrix<f64>, axis: Axis) -> Matrix<f64> {
    let (n_samples, n_features) = x.shape();

    let centered_data = match axis {
        Axis::Col => {
            let mean_matrix = mean_vertical(x); // 1 x n_features
            x.zip(
                &mean_matrix.broadcast_row_to_target_shape(n_samples, n_features),
                |val, m| val - m,
            )
        }
        Axis::Row => {
            let mean_matrix = mean_horizontal(x); // n_samples x 1
                                                  // Manually create a matrix by broadcasting the column vector across columns
            let mut broadcasted_mean = Matrix::zeros(n_samples, n_features);
            for r in 0..n_samples {
                let mean_val = mean_matrix.get(r, 0);
                for c in 0..n_features {
                    *broadcasted_mean.get_mut(r, c) = *mean_val;
                }
            }
            x.zip(&broadcasted_mean, |val, m| val - m)
        }
    };

    // Calculate covariance matrix: (X_centered^T * X_centered) / (n_samples - 1)
    // If x is (n_samples, n_features), then centered_data is (n_samples, n_features)
    // centered_data.transpose() is (n_features, n_samples)
    // Result is (n_features, n_features)
    centered_data.transpose().matrix_mul(&centered_data) / (n_samples as f64 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    const EPS: f64 = 1e-8;

    #[test]
    fn test_covariance_scalar_same_matrix() {
        // M =
        // 1,2
        // 3,4
        // mean = 2.5
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = Matrix::from_vec(data.clone(), 2, 2);

        // flatten M: [1,2,3,4], mean = 2.5
        // cov(M,M) = variance of flatten = 1.25
        let cov = covariance(&m, &m);
        assert!((cov - 1.25).abs() < EPS);
    }

    #[test]
    fn test_covariance_scalar_diff_matrix() {
        // x =
        // 1,2
        // 3,4
        // y = 2*x
        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::from_vec(vec![2.0, 4.0, 6.0, 8.0], 2, 2);

        // mean_x = 2.5, mean_y = 5.0
        // cov = sum((xi-2.5)*(yi-5.0))/4 = 2.5
        let cov_xy = covariance(&x, &y);
        assert!((cov_xy - 2.5).abs() < EPS);
    }

    #[test]
    fn test_covariance_vertical() {
        // M =
        // 1,2
        // 3,4
        // cols are [1,3] and [2,4], each var=1, cov=1
        let m = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let cov_mat = covariance_vertical(&m);

        // Expect 2x2 matrix of all 1.0
        for i in 0..2 {
            for j in 0..2 {
                assert!((cov_mat.get(i, j) - 1.0).abs() < EPS);
            }
        }
    }

    #[test]
    fn test_covariance_horizontal() {
        // M =
        // 1,2
        // 3,4
        // rows are [1,2] and [3,4], each var=0.25, cov=0.25
        let m = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let cov_mat = covariance_horizontal(&m);

        // Expect 2x2 matrix of all 0.25
        for i in 0..2 {
            for j in 0..2 {
                assert!((cov_mat.get(i, j) - 0.25).abs() < EPS);
            }
        }
    }

    #[test]
    fn test_covariance_matrix_vertical() {
        // Test with a simple 2x2 matrix
        // M =
        // 1, 2
        // 3, 4
        // Expected covariance matrix (vertical, i.e., between columns):
        // Col1: [1, 3], mean = 2
        // Col2: [2, 4], mean = 3
        // Cov(Col1, Col1) = ((1-2)^2 + (3-2)^2) / (2-1) = (1+1)/1 = 2
        // Cov(Col2, Col2) = ((2-3)^2 + (4-3)^2) / (2-1) = (1+1)/1 = 2
        // Cov(Col1, Col2) = ((1-2)*(2-3) + (3-2)*(4-3)) / (2-1) = ((-1)*(-1) + (1)*(1))/1 = (1+1)/1 = 2
        // Cov(Col2, Col1) = 2
        // Expected:
        // 2, 2
        // 2, 2
        let m = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let cov_mat = covariance_matrix(&m, Axis::Col);

        assert!((cov_mat.get(0, 0) - 2.0).abs() < EPS);
        assert!((cov_mat.get(0, 1) - 2.0).abs() < EPS);
        assert!((cov_mat.get(1, 0) - 2.0).abs() < EPS);
        assert!((cov_mat.get(1, 1) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_covariance_matrix_horizontal() {
        // Test with a simple 2x2 matrix
        // M =
        // 1, 2
        // 3, 4
        // Expected covariance matrix (horizontal, i.e., between rows):
        // Row1: [1, 2], mean = 1.5
        // Row2: [3, 4], mean = 3.5
        // Cov(Row1, Row1) = ((1-1.5)^2 + (2-1.5)^2) / (2-1) = (0.25+0.25)/1 = 0.5
        // Cov(Row2, Row2) = ((3-3.5)^2 + (4-3.5)^2) / (2-1) = (0.25+0.25)/1 = 0.5
        // Cov(Row1, Row2) = ((1-1.5)*(3-3.5) + (2-1.5)*(4-3.5)) / (2-1) = ((-0.5)*(-0.5) + (0.5)*(0.5))/1 = (0.25+0.25)/1 = 0.5
        // Cov(Row2, Row1) = 0.5
        // Expected:
        // 0.5, -0.5
        // -0.5, 0.5
        let m = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let cov_mat = covariance_matrix(&m, Axis::Row);

        assert!((cov_mat.get(0, 0) - 0.5).abs() < EPS);
        assert!((cov_mat.get(0, 1) - (-0.5)).abs() < EPS);
        assert!((cov_mat.get(1, 0) - (-0.5)).abs() < EPS);
        assert!((cov_mat.get(1, 1) - 0.5).abs() < EPS);
    }
}
