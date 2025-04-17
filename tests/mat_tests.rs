// Unit tests for Matrix<T> and its f64‑specific helpers.

#[cfg(test)]
mod tests {
    use rustframe::frame::mat::{Axis, Matrix};
    use std::f64::NAN;

    const EPS: f64 = 1e-12;

    fn assert_vec_f64_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((*x - *y).abs() < EPS, "index {i} differs: {x} vs {y}");
        }
    }

    // Constructors

    #[test]
    fn from_cols_basic() {
        let m = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 0)], 2);
        assert_eq!(m[(0, 1)], 3);
        assert_eq!(m[(1, 1)], 4);
    }

    #[test]
    #[should_panic(expected = "need at least one column")]
    fn from_cols_zero_columns_panics() {
        let _ = Matrix::<i32>::from_cols(vec![]);
    }

    #[test]
    #[should_panic(expected = "col 1")]
    fn from_cols_mismatched_lengths_panics() {
        let _ = Matrix::from_cols(vec![vec![1, 2], vec![3]]);
    }

    #[test]
    fn from_vec_basic() {
        // column‑major order: (r,c) = value
        // (0,0)=1 (1,0)=2 (0,1)=3 (1,1)=4 (0,2)=5 (1,2)=6
        let data = vec![1, 2, 3, 4, 5, 6];
        let m = Matrix::from_vec(data, 2, 3);
        assert_eq!(m[(0, 2)], 5);
        assert_eq!(m[(1, 2)], 6);
    }

    // Indexing & mutation

    #[test]
    fn index_mut_works() {
        let mut m = Matrix::from_cols(vec![vec![1, 2]]);
        *m.get_mut(0, 0) = 10;
        assert_eq!(m[(0, 0)], 10);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn index_out_of_bounds_panics() {
        let m = Matrix::from_cols(vec![vec![1]]);
        let _ = m[(1, 0)];
    }

    // Column swapping

    #[test]
    fn swap_columns_basic() {
        let mut m = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
        m.swap_columns(0, 1);
        let expected = Matrix::from_cols(vec![vec![3, 4], vec![1, 2]]);
        assert_eq!(m, expected);
    }

    // ── Element‑wise ops ─

    #[test]
    fn elementwise_add_sub_mul_div() {
        let a = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
        let b = Matrix::from_cols(vec![vec![10, 20], vec![30, 40]]);

        assert_eq!(&a + &b, Matrix::from_cols(vec![vec![11, 22], vec![33, 44]]));
        assert_eq!(&b - &a, Matrix::from_cols(vec![vec![9, 18], vec![27, 36]]));
        assert_eq!(
            &a * &b,
            Matrix::from_cols(vec![vec![10, 40], vec![90, 160]])
        );
        assert_eq!(&b / &a, Matrix::from_cols(vec![vec![10, 10], vec![10, 10]]));
    }

    // f64‑specific helpers

    #[test]
    fn reductions_with_nan() {
        let m = Matrix::from_cols(vec![vec![1.0, NAN, 3.0], vec![4.0, 5.0, NAN]]);
        // Matrix: 3 rows × 2 columns
        assert_vec_f64_eq(&m.sum_vertical(), &[4.0, 9.0]);
        assert_vec_f64_eq(&m.sum_horizontal(), &[5.0, 5.0, 3.0]);
        assert_vec_f64_eq(&m.prod_vertical(), &[3.0, 20.0]);
        assert_vec_f64_eq(&m.prod_horizontal(), &[4.0, 5.0, 3.0]);

        assert_eq!(m.count_nan_vertical(), vec![1, 1]);
        assert_eq!(m.count_nan_horizontal(), vec![0, 1, 1]);
    }

    #[test]
    fn apply_axis_dispatch() {
        let m = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let col_sums = m.apply_axis(Axis::Col, |col| col.iter().sum::<f64>());
        let row_sums = m.apply_axis(Axis::Row, |row| row.iter().sum::<f64>());
        assert_vec_f64_eq(&col_sums, &[3.0, 7.0]);
        assert_vec_f64_eq(&row_sums, &[4.0, 6.0]);
    }

    #[test]
    fn is_nan_mask() {
        let m = Matrix::from_cols(vec![vec![1.0, NAN]]);
        let expected = Matrix::from_cols(vec![vec![false, true]]);
        assert_eq!(m.is_nan(), expected);
    }
}
