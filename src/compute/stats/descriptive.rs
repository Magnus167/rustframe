//! Descriptive statistics for matrices.
//!
//! Provides means, variances, medians and other aggregations computed either
//! across the whole matrix or along a specific axis.
//!
//! ```
//! use rustframe::compute::stats::descriptive;
//! use rustframe::matrix::Matrix;
//!
//! let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//! assert_eq!(descriptive::mean(&m), 2.5);
//! ```
use crate::matrix::{Axis, Matrix, SeriesOps};

pub fn mean(x: &Matrix<f64>) -> f64 {
    x.data().iter().sum::<f64>() / (x.rows() * x.cols()) as f64
}

pub fn mean_vertical(x: &Matrix<f64>) -> Matrix<f64> {
    let m = x.rows() as f64;
    Matrix::from_vec(x.sum_vertical(), 1, x.cols()) / m
}

pub fn mean_horizontal(x: &Matrix<f64>) -> Matrix<f64> {
    let n = x.cols() as f64;
    Matrix::from_vec(x.sum_horizontal(), x.rows(), 1) / n
}

fn population_or_sample_variance(x: &Matrix<f64>, population: bool) -> f64 {
    let m = (x.rows() * x.cols()) as f64;
    let mean_val = mean(x);
    x.data()
        .iter()
        .map(|&v| (v - mean_val).powi(2))
        .sum::<f64>()
        / if population { m } else { m - 1.0 }
}

pub fn population_variance(x: &Matrix<f64>) -> f64 {
    population_or_sample_variance(x, true)
}

pub fn sample_variance(x: &Matrix<f64>) -> f64 {
    population_or_sample_variance(x, false)
}

fn _population_or_sample_variance_axis(
    x: &Matrix<f64>,
    axis: Axis,
    population: bool,
) -> Matrix<f64> {
    match axis {
        Axis::Row => {
            // Calculate variance for each column (vertical variance)
            let num_rows = x.rows() as f64;
            let mean_of_cols = mean_vertical(x); // 1 x cols matrix
            let mut result_data = vec![0.0; x.cols()];

            for c in 0..x.cols() {
                let mean_val = mean_of_cols.get(0, c); // Mean for current column
                let mut sum_sq_diff = 0.0;
                for r in 0..x.rows() {
                    let diff = x.get(r, c) - mean_val;
                    sum_sq_diff += diff * diff;
                }
                result_data[c] = sum_sq_diff / (if population { num_rows } else { num_rows - 1.0 });
            }
            Matrix::from_vec(result_data, 1, x.cols())
        }
        Axis::Col => {
            // Calculate variance for each row (horizontal variance)
            let num_cols = x.cols() as f64;
            let mean_of_rows = mean_horizontal(x); // rows x 1 matrix
            let mut result_data = vec![0.0; x.rows()];

            for r in 0..x.rows() {
                let mean_val = mean_of_rows.get(r, 0); // Mean for current row
                let mut sum_sq_diff = 0.0;
                for c in 0..x.cols() {
                    let diff = x.get(r, c) - mean_val;
                    sum_sq_diff += diff * diff;
                }
                result_data[r] = sum_sq_diff / (if population { num_cols } else { num_cols - 1.0 });
            }
            Matrix::from_vec(result_data, x.rows(), 1)
        }
    }
}

pub fn population_variance_vertical(x: &Matrix<f64>) -> Matrix<f64> {
    _population_or_sample_variance_axis(x, Axis::Row, true)
}

pub fn population_variance_horizontal(x: &Matrix<f64>) -> Matrix<f64> {
    _population_or_sample_variance_axis(x, Axis::Col, true)
}

pub fn sample_variance_vertical(x: &Matrix<f64>) -> Matrix<f64> {
    _population_or_sample_variance_axis(x, Axis::Row, false)
}

pub fn sample_variance_horizontal(x: &Matrix<f64>) -> Matrix<f64> {
    _population_or_sample_variance_axis(x, Axis::Col, false)
}

pub fn stddev(x: &Matrix<f64>) -> f64 {
    population_variance(x).sqrt()
}

pub fn stddev_vertical(x: &Matrix<f64>) -> Matrix<f64> {
    population_variance_vertical(x).map(|v| v.sqrt())
}

pub fn stddev_horizontal(x: &Matrix<f64>) -> Matrix<f64> {
    population_variance_horizontal(x).map(|v| v.sqrt())
}

pub fn median(x: &Matrix<f64>) -> f64 {
    let mut data = x.data().to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

fn _median_axis(x: &Matrix<f64>, axis: Axis) -> Matrix<f64> {
    let mx = match axis {
        Axis::Col => x.clone(),
        Axis::Row => x.transpose(),
    };

    let mut result = Vec::with_capacity(mx.cols());
    for c in 0..mx.cols() {
        let mut col = mx.column(c).to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = col.len() / 2;
        if col.len() % 2 == 0 {
            result.push((col[mid - 1] + col[mid]) / 2.0);
        } else {
            result.push(col[mid]);
        }
    }
    let (r, c) = match axis {
        Axis::Col => (1, mx.cols()),
        Axis::Row => (mx.cols(), 1),
    };
    Matrix::from_vec(result, r, c)
}

pub fn median_vertical(x: &Matrix<f64>) -> Matrix<f64> {
    _median_axis(x, Axis::Col)
}

pub fn median_horizontal(x: &Matrix<f64>) -> Matrix<f64> {
    _median_axis(x, Axis::Row)
}

pub fn percentile(x: &Matrix<f64>, p: f64) -> f64 {
    if p < 0.0 || p > 100.0 {
        panic!("Percentile must be between 0 and 100");
    }
    let mut data = x.data().to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((p / 100.0) * (data.len() as f64 - 1.0)).round() as usize;
    data[index]
}

fn _percentile_axis(x: &Matrix<f64>, p: f64, axis: Axis) -> Matrix<f64> {
    if p < 0.0 || p > 100.0 {
        panic!("Percentile must be between 0 and 100");
    }
    let mx: Matrix<f64> = match axis {
        Axis::Col => x.clone(),
        Axis::Row => x.transpose(),
    };
    let mut result = Vec::with_capacity(mx.cols());
    for c in 0..mx.cols() {
        let mut col = mx.column(c).to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((p / 100.0) * (col.len() as f64 - 1.0)).round() as usize;
        result.push(col[index]);
    }
    let (r, c) = match axis {
        Axis::Col => (1, mx.cols()),
        Axis::Row => (mx.cols(), 1),
    };
    Matrix::from_vec(result, r, c)
}

pub fn percentile_vertical(x: &Matrix<f64>, p: f64) -> Matrix<f64> {
    _percentile_axis(x, p, Axis::Col)
}
pub fn percentile_horizontal(x: &Matrix<f64>, p: f64) -> Matrix<f64> {
    _percentile_axis(x, p, Axis::Row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    const EPSILON: f64 = 1e-8;

    #[test]
    fn test_descriptive_stats_regular_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Matrix::from_vec(data, 1, 5);

        // Mean
        assert!((mean(&x) - 3.0).abs() < EPSILON);

        // Variance
        assert!((population_variance(&x) - 2.0).abs() < EPSILON);

        // Standard Deviation
        assert!((stddev(&x) - 1.4142135623730951).abs() < EPSILON);

        // Median
        assert!((median(&x) - 3.0).abs() < EPSILON);

        // Percentile
        assert!((percentile(&x, 0.0) - 1.0).abs() < EPSILON);
        assert!((percentile(&x, 25.0) - 2.0).abs() < EPSILON);
        assert!((percentile(&x, 50.0) - 3.0).abs() < EPSILON);
        assert!((percentile(&x, 75.0) - 4.0).abs() < EPSILON);
        assert!((percentile(&x, 100.0) - 5.0).abs() < EPSILON);

        let data_even = vec![1.0, 2.0, 3.0, 4.0];
        let x_even = Matrix::from_vec(data_even, 1, 4);
        assert!((median(&x_even) - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_descriptive_stats_outlier() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let x = Matrix::from_vec(data, 1, 5);

        // Mean should be heavily affected by outlier
        assert!((mean(&x) - 22.0).abs() < EPSILON);

        // Variance should be heavily affected by outlier
        assert!((population_variance(&x) - 1522.0).abs() < EPSILON);

        // Standard Deviation should be heavily affected by outlier
        assert!((stddev(&x) - 39.0128183970461).abs() < EPSILON);

        // Median should be robust to outlier
        assert!((median(&x) - 3.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "Percentile must be between 0 and 100")]
    fn test_percentile_panic_low() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Matrix::from_vec(data, 1, 3);
        percentile(&x, -1.0);
    }

    #[test]
    #[should_panic(expected = "Percentile must be between 0 and 100")]
    fn test_percentile_panic_high() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Matrix::from_vec(data, 1, 3);
        percentile(&x, 101.0);
    }

    #[test]
    fn test_mean_vertical_horizontal() {
        // 2x3 matrix:
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = Matrix::from_vec(data, 2, 3);

        // Vertical means (per column): [(1+4)/2, (2+5)/2, (3+6)/2]
        let mv = mean_vertical(&x);
        assert!((mv.get(0, 0) - 2.5).abs() < EPSILON);
        assert!((mv.get(0, 1) - 3.5).abs() < EPSILON);
        assert!((mv.get(0, 2) - 4.5).abs() < EPSILON);

        // Horizontal means (per row): [(1+2+3)/3, (4+5+6)/3]
        let mh = mean_horizontal(&x);
        assert!((mh.get(0, 0) - 2.0).abs() < EPSILON);
        assert!((mh.get(1, 0) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_variance_vertical_horizontal() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = Matrix::from_vec(data, 2, 3);

        // cols: {1,4}, {2,5}, {3,6} all give 2.25
        let vv = population_variance_vertical(&x);
        for c in 0..3 {
            assert!((vv.get(0, c) - 2.25).abs() < EPSILON);
        }

        let vh = population_variance_horizontal(&x);
        assert!((vh.get(0, 0) - (2.0 / 3.0)).abs() < EPSILON);
        assert!((vh.get(1, 0) - (2.0 / 3.0)).abs() < EPSILON);

        // sample variance vertical: denominator is n-1 = 1, so variance is 4.5
        let svv = sample_variance_vertical(&x);
        for c in 0..3 {
            assert!((svv.get(0, c) - 4.5).abs() < EPSILON);
        }

        // sample variance horizontal: denominator is n-1 = 2, so variance is 1.0
        let svh = sample_variance_horizontal(&x);
        assert!((svh.get(0, 0) - 1.0).abs() < EPSILON);
        assert!((svh.get(1, 0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_stddev_vertical_horizontal() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = Matrix::from_vec(data, 2, 3);

        // Stddev is sqrt of variance
        let sv = stddev_vertical(&x);
        for c in 0..3 {
            assert!((sv.get(0, c) - 1.5).abs() < EPSILON);
        }

        let sh = stddev_horizontal(&x);
        // sqrt(2/3) ≈ 0.816497
        let expected = (2.0 / 3.0 as f64).sqrt();
        assert!((sh.get(0, 0) - expected).abs() < EPSILON);
        assert!((sh.get(1, 0) - expected).abs() < EPSILON);

        // sample stddev vertical: sqrt(4.5) ≈ 2.12132034
        let ssv = sample_variance_vertical(&x).map(|v| v.sqrt());
        for c in 0..3 {
            assert!((ssv.get(0, c) - 2.1213203435596424).abs() < EPSILON);
        }

        // sample stddev horizontal: sqrt(1.0) = 1.0
        let ssh = sample_variance_horizontal(&x).map(|v| v.sqrt());
        assert!((ssh.get(0, 0) - 1.0).abs() < EPSILON);
        assert!((ssh.get(1, 0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_median_vertical_horizontal() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = Matrix::from_vec(data, 2, 3);

        let mv = median_vertical(&x).row(0);

        let expected_v = vec![2.5, 3.5, 4.5];
        assert_eq!(mv, expected_v, "{:?} expected: {:?}", expected_v, mv);

        let mh = median_horizontal(&x).column(0).to_vec();
        let expected_h = vec![2.0, 5.0];
        assert_eq!(mh, expected_h, "{:?} expected: {:?}", expected_h, mh);
    }

    #[test]
    fn test_percentile_vertical_horizontal() {
        // vec of f64 values 1..24 as a 4x6 matrix
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let x = Matrix::from_vec(data, 4, 6);

        // columns contain sequences increasing by four starting at 1 through 4

        let er0 = vec![1., 5., 9., 13., 17., 21.];
        let er50 = vec![3., 7., 11., 15., 19., 23.];
        let er100 = vec![4., 8., 12., 16., 20., 24.];

        assert_eq!(percentile_vertical(&x, 0.0).data(), er0);
        assert_eq!(percentile_vertical(&x, 50.0).data(), er50);
        assert_eq!(percentile_vertical(&x, 100.0).data(), er100);

        let eh0 = vec![1., 2., 3., 4.];
        let eh50 = vec![13., 14., 15., 16.];
        let eh100 = vec![21., 22., 23., 24.];

        assert_eq!(percentile_horizontal(&x, 0.0).data(), eh0);
        assert_eq!(percentile_horizontal(&x, 50.0).data(), eh50);
        assert_eq!(percentile_horizontal(&x, 100.0).data(), eh100);
    }

    #[test]
    #[should_panic(expected = "Percentile must be between 0 and 100")]
    fn test_percentile_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Matrix::from_vec(data, 1, 3);
        percentile(&x, -10.0); // Should panic
    }

    #[test]
    #[should_panic(expected = "Percentile must be between 0 and 100")]
    fn test_percentile_vertical_out_of_bounds() {
        let m = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let _ = percentile_vertical(&m, -0.1);
    }
}
