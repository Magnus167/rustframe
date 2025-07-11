use crate::matrix::{Axis, BoolMatrix};

/// Boolean operations on `Matrix<bool>`
pub trait BoolOps {
    /// Generic helper: apply `f` to every column/row and collect its
    /// result in a `Vec`.
    fn apply_axis<U, F>(&self, axis: Axis, f: F) -> Vec<U>
    where
        F: FnMut(&[bool]) -> U;

    fn any_vertical(&self) -> Vec<bool>;
    fn any_horizontal(&self) -> Vec<bool>;
    fn all_vertical(&self) -> Vec<bool>;
    fn all_horizontal(&self) -> Vec<bool>;
    fn count_vertical(&self) -> Vec<usize>;
    fn count_horizontal(&self) -> Vec<usize>;
    fn any(&self) -> bool;
    fn all(&self) -> bool;
    fn count(&self) -> usize;
}

impl BoolOps for BoolMatrix {
    fn apply_axis<U, F>(&self, axis: Axis, mut f: F) -> Vec<U>
    where
        F: FnMut(&[bool]) -> U,
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
                let mut buf = vec![false; self.cols()]; // reusable buffer
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
    fn any_vertical(&self) -> Vec<bool> {
        self.apply_axis(Axis::Col, |col| col.iter().any(|&v| v))
    }
    fn any_horizontal(&self) -> Vec<bool> {
        self.apply_axis(Axis::Row, |row| row.iter().any(|&v| v))
    }
    fn all_vertical(&self) -> Vec<bool> {
        self.apply_axis(Axis::Col, |col| col.iter().all(|&v| v))
    }
    fn all_horizontal(&self) -> Vec<bool> {
        self.apply_axis(Axis::Row, |row| row.iter().all(|&v| v))
    }
    fn count_vertical(&self) -> Vec<usize> {
        self.apply_axis(Axis::Col, |col| col.iter().filter(|&&v| v).count())
    }
    fn count_horizontal(&self) -> Vec<usize> {
        self.apply_axis(Axis::Row, |row| row.iter().filter(|&&v| v).count())
    }
    fn any(&self) -> bool {
        self.data().iter().any(|&v| v)
    }
    fn all(&self) -> bool {
        self.data().iter().all(|&v| v)
    }
    fn count(&self) -> usize {
        self.data().iter().filter(|&&v| v).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a BoolMatrix for BoolOps testing
    fn create_bool_test_matrix() -> BoolMatrix {
        // 3x3 matrix (column-major)
        // T F T
        // F T F
        // T F F
        let data = vec![true, false, true, false, true, false, true, false, false];
        BoolMatrix::from_vec(data, 3, 3)
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
