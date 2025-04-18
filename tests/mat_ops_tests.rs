#[cfg(test)]
mod tests {
    use rustframe::matrix::*;

    // Helper function to create a FloatMatrix for SeriesOps testing
    fn create_float_test_matrix() -> FloatMatrix {
        // 3x3 matrix (column-major) with some NaNs
        // 1.0  4.0  7.0
        // 2.0  NaN  8.0
        // 3.0  6.0  NaN
        let data = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0, 7.0, 8.0, f64::NAN];
        FloatMatrix::from_vec(data, 3, 3)
    }

    // Helper function to create a BoolMatrix for BoolOps testing
    fn create_bool_test_matrix() -> BoolMatrix {
        // 3x3 matrix (column-major)
        // T F T
        // F T F
        // T F F
        let data = vec![true, false, true, false, true, false, true, false, false];
        BoolMatrix::from_vec(data, 3, 3)
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

    // --- Tests for BoolOps (BoolMatrix) ---

    #[test]
    fn test_bool_ops_any_vertical() {
        let matrix = create_bool_test_matrix();
        // Col 0: T | F | T = T
        // Col 1: F | T | F = T
        // Col 2: T | F | F = T
        let expected = vec![true, true, true];
        assert_eq!(matrix.any_vertical(), expected);
    }

    #[test]
    fn test_bool_ops_any_horizontal() {
        let matrix = create_bool_test_matrix();
        // Row 0: T | F | T = T
        // Row 1: F | T | F = T
        // Row 2: T | F | F = T
        let expected = vec![true, true, true];
        assert_eq!(matrix.any_horizontal(), expected);
    }

    #[test]
    fn test_bool_ops_all_vertical() {
        let matrix = create_bool_test_matrix();
        // Col 0: T & F & T = F
        // Col 1: F & T & F = F
        // Col 2: T & F & F = F
        let expected = vec![false, false, false];
        assert_eq!(matrix.all_vertical(), expected);
    }

    #[test]
    fn test_bool_ops_all_horizontal() {
        let matrix = create_bool_test_matrix();
        // Row 0: T & F & T = F
        // Row 1: F & T & F = F
        // Row 2: T & F & F = F
        let expected = vec![false, false, false];
        assert_eq!(matrix.all_horizontal(), expected);
    }

    #[test]
    fn test_bool_ops_count_vertical() {
        let matrix = create_bool_test_matrix();
        // Col 0: count true in [T, F, T] = 2
        // Col 1: count true in [F, T, F] = 1
        // Col 2: count true in [T, F, F] = 1
        let expected = vec![2, 1, 1];
        assert_eq!(matrix.count_vertical(), expected);
    }

    #[test]
    fn test_bool_ops_count_horizontal() {
        let matrix = create_bool_test_matrix();
        // Row 0: count true in [T, F, T] = 2
        // Row 1: count true in [F, T, F] = 1
        // Row 2: count true in [T, F, F] = 1
        let expected = vec![2, 1, 1];
        assert_eq!(matrix.count_horizontal(), expected);
    }

    #[test]
    fn test_bool_ops_any_overall() {
        let matrix = create_bool_test_matrix(); // Has true values
        assert!(matrix.any());

        let matrix_all_false = BoolMatrix::from_vec(vec![false; 9], 3, 3);
        assert!(!matrix_all_false.any());
    }

    #[test]
    fn test_bool_ops_all_overall() {
        let matrix = create_bool_test_matrix(); // Has false values
        assert!(!matrix.all());

        let matrix_all_true = BoolMatrix::from_vec(vec![true; 9], 3, 3);
        assert!(matrix_all_true.all());
    }

    #[test]
    fn test_bool_ops_count_overall() {
        let matrix = create_bool_test_matrix(); // Data: [T, F, T, F, T, F, T, F, F]
        // Count of true values: 4
        assert_eq!(matrix.count(), 4);

        let matrix_all_false = BoolMatrix::from_vec(vec![false; 5], 5, 1); // 5x1
        assert_eq!(matrix_all_false.count(), 0);

        let matrix_all_true = BoolMatrix::from_vec(vec![true; 4], 2, 2); // 2x2
        assert_eq!(matrix_all_true.count(), 4);
    }

    // --- Edge Cases for BoolOps ---

    #[test]
    fn test_bool_ops_1x1() {
        let matrix_t = BoolMatrix::from_vec(vec![true], 1, 1);
        assert_eq!(matrix_t.any_vertical(), vec![true]);
        assert_eq!(matrix_t.any_horizontal(), vec![true]);
        assert_eq!(matrix_t.all_vertical(), vec![true]);
        assert_eq!(matrix_t.all_horizontal(), vec![true]);
        assert_eq!(matrix_t.count_vertical(), vec![1]);
        assert_eq!(matrix_t.count_horizontal(), vec![1]);
        assert!(matrix_t.any());
        assert!(matrix_t.all());
        assert_eq!(matrix_t.count(), 1);

        let matrix_f = BoolMatrix::from_vec(vec![false], 1, 1);
        assert_eq!(matrix_f.any_vertical(), vec![false]);
        assert_eq!(matrix_f.any_horizontal(), vec![false]);
        assert_eq!(matrix_f.all_vertical(), vec![false]);
        assert_eq!(matrix_f.all_horizontal(), vec![false]);
        assert_eq!(matrix_f.count_vertical(), vec![0]);
        assert_eq!(matrix_f.count_horizontal(), vec![0]);
        assert!(!matrix_f.any());
        assert!(!matrix_f.all());
        assert_eq!(matrix_f.count(), 0);
    }

    #[test]
    fn test_bool_ops_1xn_matrix() {
        let matrix = BoolMatrix::from_vec(vec![true, false, false, true], 1, 4); // 1 row, 4 cols
        // Data: [T, F, F, T]

        assert_eq!(matrix.any_vertical(), vec![true, false, false, true]);
        assert_eq!(matrix.all_vertical(), vec![true, false, false, true]);
        assert_eq!(matrix.count_vertical(), vec![1, 0, 0, 1]);

        assert_eq!(matrix.any_horizontal(), vec![true]); // T | F | F | T = T
        assert_eq!(matrix.all_horizontal(), vec![false]); // T & F & F & T = F
        assert_eq!(matrix.count_horizontal(), vec![2]); // count true in [T, F, F, T] = 2

        assert!(matrix.any());
        assert!(!matrix.all());
        assert_eq!(matrix.count(), 2);
    }

    #[test]
    fn test_bool_ops_nx1_matrix() {
        let matrix = BoolMatrix::from_vec(vec![true, false, false, true], 4, 1); // 4 rows, 1 col
        // Data: [T, F, F, T]

        assert_eq!(matrix.any_vertical(), vec![true]); // T|F|F|T = T
        assert_eq!(matrix.all_vertical(), vec![false]); // T&F&F&T = F
        assert_eq!(matrix.count_vertical(), vec![2]); // count true in [T, F, F, T] = 2

        assert_eq!(matrix.any_horizontal(), vec![true, false, false, true]);
        assert_eq!(matrix.all_horizontal(), vec![true, false, false, true]);
        assert_eq!(matrix.count_horizontal(), vec![1, 0, 0, 1]);

        assert!(matrix.any());
        assert!(!matrix.all());
        assert_eq!(matrix.count(), 2);
    }
}
