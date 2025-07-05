use crate::matrix::{Axis, BoolMatrix, FloatMatrix};

/// "Series-like" helpers that work along a single axis.
///
/// *All* the old methods (`sum_*`, `prod_*`, `is_nan`, …) are exposed
/// through this trait, so nothing needs to stay on an `impl Matrix<f64>`;
/// just `use SeriesOps` to make the extension methods available.
pub trait SeriesOps {
    /// Generic helper: apply `f` to every column/row and collect its
    /// result in a `Vec`.
    fn apply_axis<U, F>(&self, axis: Axis, f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U;

    fn matrix_mul(&self, other: &Self) -> FloatMatrix;
    fn dot(&self, other: &Self) -> FloatMatrix;

    fn sum_vertical(&self) -> Vec<f64>;
    fn sum_horizontal(&self) -> Vec<f64>;

    fn prod_vertical(&self) -> Vec<f64>;
    fn prod_horizontal(&self) -> Vec<f64>;

    fn cumsum_vertical(&self) -> FloatMatrix;
    fn cumsum_horizontal(&self) -> FloatMatrix;

    fn count_nan_vertical(&self) -> Vec<usize>;
    fn count_nan_horizontal(&self) -> Vec<usize>;

    fn is_nan(&self) -> BoolMatrix;
}

impl SeriesOps for FloatMatrix {
    fn apply_axis<U, F>(&self, axis: Axis, mut f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U,
    {
        match axis {
            Axis::Col => {
                let mut out = Vec::with_capacity(self.cols());
                for c in 0..self.cols() {
                    out.push(f(self.column(c)));
                }
                out
            }
            Axis::Row => {
                let mut out = Vec::with_capacity(self.rows());
                let mut buf = vec![0.0; self.cols()]; // reusable buffer
                for r in 0..self.rows() {
                    for c in 0..self.cols() {
                        buf[c] = self[(r, c)];
                    }
                    out.push(f(&buf));
                }
                out
            }
        }
    }

    fn sum_vertical(&self) -> Vec<f64> {
        self.apply_axis(Axis::Col, |col| {
            col.iter().copied().filter(|v| !v.is_nan()).sum::<f64>()
        })
    }

    fn sum_horizontal(&self) -> Vec<f64> {
        self.apply_axis(Axis::Row, |row| {
            row.iter().copied().filter(|v| !v.is_nan()).sum::<f64>()
        })
    }

    fn prod_vertical(&self) -> Vec<f64> {
        self.apply_axis(Axis::Col, |col| {
            col.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(1.0, |acc, x| acc * x)
        })
    }

    fn prod_horizontal(&self) -> Vec<f64> {
        self.apply_axis(Axis::Row, |row| {
            row.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(1.0, |acc, x| acc * x)
        })
    }

    fn cumsum_vertical(&self) -> FloatMatrix {
        let mut data = Vec::with_capacity(self.rows() * self.cols());
        for c in 0..self.cols() {
            let mut acc = 0.0;
            for r in 0..self.rows() {
                let v = self[(r, c)];
                if !v.is_nan() {
                    acc += v;
                }
                data.push(acc);
            }
        }
        FloatMatrix::from_vec(data, self.rows(), self.cols())
    }

    fn cumsum_horizontal(&self) -> FloatMatrix {
        // Compute cumulative sums for each row and store in a temporary buffer
        let mut row_results: Vec<Vec<f64>> = Vec::with_capacity(self.rows());
        for r in 0..self.rows() {
            let mut row_data = Vec::with_capacity(self.cols());
            let mut acc = 0.0;
            for c in 0..self.cols() {
                let v = self[(r, c)];
                if !v.is_nan() {
                    acc += v;
                }
                row_data.push(acc);
            }
            row_results.push(row_data);
        }

        // Assemble the final data vector in column-major format
        let mut final_data = Vec::with_capacity(self.rows() * self.cols());
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                // Extract the element at (r, c) from the temporary row-wise results
                final_data.push(row_results[r][c]);
            }
        }

        FloatMatrix::from_vec(final_data, self.rows(), self.cols())
    }

    fn count_nan_vertical(&self) -> Vec<usize> {
        self.apply_axis(Axis::Col, |col| col.iter().filter(|x| x.is_nan()).count())
    }

    fn count_nan_horizontal(&self) -> Vec<usize> {
        self.apply_axis(Axis::Row, |row| row.iter().filter(|x| x.is_nan()).count())
    }

    fn is_nan(&self) -> BoolMatrix {
        let data = self.data().iter().map(|v| v.is_nan()).collect();
        BoolMatrix::from_vec(data, self.rows(), self.cols())
    }

    fn matrix_mul(&self, other: &Self) -> FloatMatrix {
        let (m, n) = (self.rows(), self.cols());
        let (n2, p) = (other.rows(), other.cols());
        assert_eq!(
            n, n2,
            "Cannot multiply: left is {}x{}, right is {}x{}",
            m, n, n2, p
        );

        // Column-major addressing: element (row i, col j) lives at j * m + i
        let mut data = vec![0.0; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self[(i, k)] * other[(k, j)];
                }
                data[j * m + i] = sum; // <-- fixed index
            }
        }
        FloatMatrix::from_vec(data, m, p)
    }
    fn dot(&self, other: &Self) -> FloatMatrix {
        self.matrix_mul(other)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // Helper function to create a FloatMatrix for SeriesOps testing
    fn create_float_test_matrix() -> FloatMatrix {
        // 3x3 matrix (column-major) with some NaNs
        // 1.0  4.0  7.0
        // 2.0  NaN  8.0
        // 3.0  6.0  NaN
        let data = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0, 7.0, 8.0, f64::NAN];
        FloatMatrix::from_vec(data, 3, 3)
    }

    fn create_float_test_matrix_4x4() -> FloatMatrix {
        // 4x4 matrix (column-major) with some NaNs
        // 1.0  5.0  9.0  13.0
        // 2.0  NaN  10.0 NaN
        // 3.0  6.0  NaN  14.0
        // NaN  7.0  11.0 NaN
        // first make array with 16 elements
        FloatMatrix::from_vec(
            (0..16)
                .map(|i| if i % 5 == 0 { f64::NAN } else { i as f64 })
                .collect(),
            4,
            4,
        )
    }

    // --- Tests for SeriesOps (FloatMatrix) ---

    #[test]
    fn test_series_ops_sum_vertical() {
        let matrix = create_float_test_matrix();
        // Col 0: 1.0 + 2.0 + 3.0 = 6.0
        // Col 1: 4.0 + NaN + 6.0 = 10.0 (NaN ignored)
        // Col 2: 7.0 + 8.0 + NaN = 15.0 (NaN ignored)
        let expected = vec![6.0, 10.0, 15.0];
        assert_eq!(matrix.sum_vertical(), expected);
    }

    #[test]
    fn test_series_ops_sum_horizontal() {
        let matrix = create_float_test_matrix();
        // Row 0: 1.0 + 4.0 + 7.0 = 12.0
        // Row 1: 2.0 + NaN + 8.0 = 10.0 (NaN ignored)
        // Row 2: 3.0 + 6.0 + NaN = 9.0 (NaN ignored)
        let expected = vec![12.0, 10.0, 9.0];
        assert_eq!(matrix.sum_horizontal(), expected);
    }

    #[test]
    fn test_series_ops_prod_vertical() {
        let matrix = create_float_test_matrix();
        // Col 0: 1.0 * 2.0 * 3.0 = 6.0
        // Col 1: 4.0 * NaN * 6.0 = 24.0 (NaN ignored, starts with 1.0)
        // Col 2: 7.0 * 8.0 * NaN = 56.0 (NaN ignored, starts with 1.0)
        let expected = vec![6.0, 24.0, 56.0];
        assert_eq!(matrix.prod_vertical(), expected);
    }

    #[test]
    fn test_series_ops_prod_horizontal() {
        let matrix = create_float_test_matrix();
        // Row 0: 1.0 * 4.0 * 7.0 = 28.0
        // Row 1: 2.0 * NaN * 8.0 = 16.0 (NaN ignored, starts with 1.0)
        // Row 2: 3.0 * 6.0 * NaN = 18.0 (NaN ignored, starts with 1.0)
        let expected = vec![28.0, 16.0, 18.0];
        assert_eq!(matrix.prod_horizontal(), expected);
    }

    #[test]
    fn test_series_ops_cumsum_vertical() {
        let matrix = create_float_test_matrix();
        // Col 0: [1.0, 1.0+2.0=3.0, 3.0+3.0=6.0]
        // Col 1: [4.0, 4.0+NaN=4.0, 4.0+6.0=10.0] (NaN ignored, cumulative sum doesn't reset)
        // Col 2: [7.0, 7.0+8.0=15.0, 15.0+NaN=15.0]
        // Expected data (column-major): [1.0, 3.0, 6.0, 4.0, 4.0, 10.0, 7.0, 15.0, 15.0]
        let expected_data = vec![1.0, 3.0, 6.0, 4.0, 4.0, 10.0, 7.0, 15.0, 15.0];
        let expected_matrix = FloatMatrix::from_vec(expected_data, 3, 3);
        assert_eq!(matrix.cumsum_vertical(), expected_matrix);
    }

    #[test]
    fn test_series_ops_cumsum_horizontal() {
        let matrix = create_float_test_matrix();
        // Row 0: [1.0, 1.0+4.0=5.0, 5.0+7.0=12.0]
        // Row 1: [2.0, 2.0+NaN=2.0, 2.0+8.0=10.0] (NaN ignored, cumulative sum doesn't reset)
        // Row 2: [3.0, 3.0+6.0=9.0, 9.0+NaN=9.0]
        // Expected data (column-major construction from row results):
        // Col 0: (R0,C0)=1.0, (R1,C0)=2.0, (R2,C0)=3.0  => [1.0, 2.0, 3.0]
        // Col 1: (R0,C1)=5.0, (R1,C1)=2.0, (R2,C1)=9.0  => [5.0, 2.0, 9.0]
        // Col 2: (R0,C2)=12.0, (R1,C2)=10.0, (R2,C2)=9.0 => [12.0, 10.0, 9.0]
        // Combined data: [1.0, 2.0, 3.0, 5.0, 2.0, 9.0, 12.0, 10.0, 9.0]
        let expected_data = vec![1.0, 2.0, 3.0, 5.0, 2.0, 9.0, 12.0, 10.0, 9.0];
        let expected_matrix = FloatMatrix::from_vec(expected_data, 3, 3);
        assert_eq!(matrix.cumsum_horizontal(), expected_matrix);
    }

    #[test]
    fn test_series_ops_count_nan_vertical() {
        let matrix = create_float_test_matrix();
        // Col 0: 0 NaNs
        // Col 1: 1 NaN
        // Col 2: 1 NaN
        let expected = vec![0, 1, 1];
        assert_eq!(matrix.count_nan_vertical(), expected);
    }

    #[test]
    fn test_series_ops_count_nan_horizontal() {
        let matrix = create_float_test_matrix();
        // Row 0: 0 NaNs
        // Row 1: 1 NaN
        // Row 2: 1 NaN
        let expected = vec![0, 1, 1];
        assert_eq!(matrix.count_nan_horizontal(), expected);
    }

    #[test]
    fn test_series_ops_is_nan() {
        let matrix = create_float_test_matrix();
        // Original data (col-major): [1.0, 2.0, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, NaN]
        // is_nan() applied:          [F,   F,   F,   F,   T,   F,   F,   F,   T]
        let expected_data = vec![false, false, false, false, true, false, false, false, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 3, 3);
        assert_eq!(matrix.is_nan(), expected_matrix);
    }

    #[test]
    fn test_series_ops_matrix_mul() {
        let a = FloatMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2); // 2x2 matrix
        let b = FloatMatrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2); // 2x2 matrix
                                                                       // result should be: 23, 34, 31, 46
        let expected = FloatMatrix::from_vec(vec![23.0, 34.0, 31.0, 46.0], 2, 2);
        assert_eq!(a.matrix_mul(&b), expected);

        assert_eq!(a.dot(&b), a.matrix_mul(&b)); // dot should be the same as matrix_mul for FloatMatrix
    }
    #[test]
    fn test_series_ops_matrix_mul_with_nans() {
        let a = create_float_test_matrix(); // 3x3 matrix with some NaNs
        let b = create_float_test_matrix(); // 3x3 matrix with some NaNs

        let mut result_vec = Vec::new();
        result_vec.push(30.0);
        for _ in 1..9 {
            result_vec.push(f64::NAN);
        }
        let expected = FloatMatrix::from_vec(result_vec, 3, 3);

        let result = a.matrix_mul(&b);

        assert_eq!(result.is_nan(), expected.is_nan());
        assert_eq!(
            result.count_nan_horizontal(),
            expected.count_nan_horizontal()
        );
        assert_eq!(result.count_nan_vertical(), expected.count_nan_vertical());
        assert_eq!(result[(0, 0)], expected[(0, 0)]);
    }

    #[test]
    #[should_panic(expected = "Cannot multiply: left is 3x3, right is 4x4")]
    fn test_series_ops_matrix_mul_errors() {
        let a = create_float_test_matrix();
        let b = create_float_test_matrix_4x4();

        a.dot(&b); // This should panic due to dimension mismatch
    }

    // --- Edge Cases for SeriesOps ---

    #[test]
    fn test_series_ops_1x1() {
        let matrix = FloatMatrix::from_vec(vec![42.0], 1, 1);
        assert_eq!(matrix.sum_vertical(), vec![42.0]);
        assert_eq!(matrix.sum_horizontal(), vec![42.0]);
        assert_eq!(matrix.prod_vertical(), vec![42.0]);
        assert_eq!(matrix.prod_horizontal(), vec![42.0]);
        assert_eq!(matrix.cumsum_vertical().data(), &[42.0]);
        assert_eq!(matrix.cumsum_horizontal().data(), &[42.0]);
        assert_eq!(matrix.count_nan_vertical(), vec![0]);
        assert_eq!(matrix.count_nan_horizontal(), vec![0]);
        assert_eq!(matrix.is_nan().data(), &[false]);

        let matrix_nan = FloatMatrix::from_vec(vec![f64::NAN], 1, 1);
        assert_eq!(matrix_nan.sum_vertical(), vec![0.0]); // sum of empty set is 0
        assert_eq!(matrix_nan.sum_horizontal(), vec![0.0]);
        assert_eq!(matrix_nan.prod_vertical(), vec![1.0]); // product of empty set is 1
        assert_eq!(matrix_nan.prod_horizontal(), vec![1.0]);
        assert_eq!(matrix_nan.cumsum_vertical().data(), &[0.0]); // cumsum starts at 0, nan ignored
        assert_eq!(matrix_nan.cumsum_horizontal().data(), &[0.0]);
        assert_eq!(matrix_nan.count_nan_vertical(), vec![1]);
        assert_eq!(matrix_nan.count_nan_horizontal(), vec![1]);
        assert_eq!(matrix_nan.is_nan().data(), &[true]);
    }

    #[test]
    fn test_series_ops_1xn_matrix() {
        let matrix = FloatMatrix::from_vec(vec![1.0, f64::NAN, 3.0, 4.0], 1, 4); // 1 row, 4 cols
                                                                                 // Data: [1.0, NaN, 3.0, 4.0]

        // Vertical (sums/prods/counts per column - each col is just one element)
        assert_eq!(matrix.sum_vertical(), vec![1.0, 0.0, 3.0, 4.0]); // NaN sum is 0
        assert_eq!(matrix.prod_vertical(), vec![1.0, 1.0, 3.0, 4.0]); // NaN prod is 1
        assert_eq!(matrix.count_nan_vertical(), vec![0, 1, 0, 0]);
        assert_eq!(matrix.cumsum_vertical().data(), &[1.0, 0.0, 3.0, 4.0]); // Cumsum on single element column

        // Horizontal (sums/prods/counts for the single row)
        // Row 0: 1.0 + NaN + 3.0 + 4.0 = 8.0
        // Row 0: 1.0 * NaN * 3.0 * 4.0 = 12.0
        // Row 0: 1 NaN
        assert_eq!(matrix.sum_horizontal(), vec![8.0]);
        assert_eq!(matrix.prod_horizontal(), vec![12.0]);
        assert_eq!(matrix.count_nan_horizontal(), vec![1]);

        // Cumsum Horizontal
        // Row 0: [1.0, 1.0+NaN=1.0, 1.0+3.0=4.0, 4.0+4.0=8.0]
        // Data (col-major): [1.0, 1.0, 4.0, 8.0] (since it's 1 row, data is the same as the row result)
        assert_eq!(matrix.cumsum_horizontal().data(), &[1.0, 1.0, 4.0, 8.0]);

        // is_nan
        // Data: [1.0, NaN, 3.0, 4.0]
        // Expected: [F, T, F, F]
        assert_eq!(matrix.is_nan().data(), &[false, true, false, false]);
    }

    #[test]
    fn test_series_ops_nx1_matrix() {
        let matrix = FloatMatrix::from_vec(vec![1.0, 2.0, f64::NAN, 4.0], 4, 1); // 4 rows, 1 col
                                                                                 // Data: [1.0, 2.0, NaN, 4.0]

        // Vertical (sums/prods/counts for the single column)
        // Col 0: 1.0 + 2.0 + NaN + 4.0 = 7.0
        // Col 0: 1.0 * 2.0 * NaN * 4.0 = 8.0
        // Col 0: 1 NaN
        assert_eq!(matrix.sum_vertical(), vec![7.0]);
        assert_eq!(matrix.prod_vertical(), vec![8.0]);
        assert_eq!(matrix.count_nan_vertical(), vec![1]);

        // Cumsum Vertical
        // Col 0: [1.0, 1.0+2.0=3.0, 3.0+NaN=3.0, 3.0+4.0=7.0]
        // Data (col-major): [1.0, 3.0, 3.0, 7.0] (since it's 1 col, data is the same as the col result)
        assert_eq!(matrix.cumsum_vertical().data(), &[1.0, 3.0, 3.0, 7.0]);

        // Horizontal (sums/prods/counts per row - each row is just one element)
        assert_eq!(matrix.sum_horizontal(), vec![1.0, 2.0, 0.0, 4.0]); // NaN sum is 0
        assert_eq!(matrix.prod_horizontal(), vec![1.0, 2.0, 1.0, 4.0]); // NaN prod is 1
        assert_eq!(matrix.count_nan_horizontal(), vec![0, 0, 1, 0]);
        assert_eq!(matrix.cumsum_horizontal().data(), &[1.0, 2.0, 0.0, 4.0]); // Cumsum on single element row

        // is_nan
        // Data: [1.0, 2.0, NaN, 4.0]
        // Expected: [F, F, T, F]
        assert_eq!(matrix.is_nan().data(), &[false, false, true, false]);
    }

    #[test]
    fn test_series_ops_all_nan_matrix() {
        let matrix = FloatMatrix::from_vec(vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN], 2, 2);
        // NaN NaN
        // NaN NaN
        // Data: [NaN, NaN, NaN, NaN]

        assert_eq!(matrix.sum_vertical(), vec![0.0, 0.0]);
        assert_eq!(matrix.sum_horizontal(), vec![0.0, 0.0]);
        assert_eq!(matrix.prod_vertical(), vec![1.0, 1.0]);
        assert_eq!(matrix.prod_horizontal(), vec![1.0, 1.0]);
        assert_eq!(matrix.cumsum_vertical().data(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(matrix.cumsum_horizontal().data(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(matrix.count_nan_vertical(), vec![2, 2]);
        assert_eq!(matrix.count_nan_horizontal(), vec![2, 2]);
        assert_eq!(matrix.is_nan().data(), &[true, true, true, true]);
    }
}
