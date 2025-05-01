use std::ops::{Index, IndexMut, Not};

/// A column‑major 2D matrix of `T`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    /// Build from columns (each inner Vec is one column)
    pub fn from_cols(cols_data: Vec<Vec<T>>) -> Self {
        let cols = cols_data.len();
        assert!(cols > 0, "need at least one column");
        // Handle empty cols_data
        let rows = cols_data.get(0).map_or(0, |c| c.len()); 
        // Allow 0-row matrices if columns are empty, but not 0-col matrices if rows > 0
        assert!(
            rows > 0 || cols == 0,
            "need at least one row if columns exist"
        );

        for (i, col) in cols_data.iter().enumerate() {
            assert!(
                col.len() == rows,
                "col {} has len {}, expected {}",
                i,
                col.len(),
                rows
            );
        }
        // Flatten column data directly
        let data = cols_data.into_iter().flatten().collect();
        Matrix { rows, cols, data }
    }

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 || cols == 0,
            "need at least one row if columns exist"
        );
        assert!(
            cols > 0 || rows == 0,
            "need at least one column if rows exist"
        );
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Matrix { rows, cols, data }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn as_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, r: usize, c: usize) -> &T {
        &self[(r, c)]
    }
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        &mut self[(r, c)]
    }

    #[inline]
    pub fn column(&self, c: usize) -> &[T] {
        assert!(
            c < self.cols,
            "column index {} out of bounds for {} columns",
            c,
            self.cols
        );
        let start = c * self.rows;
        &self.data[start..start + self.rows]
    }
    #[inline]
    pub fn column_mut(&mut self, c: usize) -> &mut [T] {
        assert!(
            c < self.cols,
            "column index {} out of bounds for {} columns",
            c,
            self.cols
        );
        let start = c * self.rows;
        &mut self.data[start..start + self.rows]
    }

    pub fn iter_columns(&self) -> impl Iterator<Item = &[T]> {
        (0..self.cols).map(move |c| self.column(c))
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = MatrixRow<'_, T>> {
        (0..self.rows).map(move |r| MatrixRow {
            matrix: self,
            row: r,
        })
    }

    /// Swaps two columns in the matrix.
    pub fn swap_columns(&mut self, c1: usize, c2: usize) {
        assert!(
            c1 < self.cols,
            "column index c1={} out of bounds for {} columns",
            c1,
            self.cols
        );
        assert!(
            c2 < self.cols,
            "column index c2={} out of bounds for {} columns",
            c2,
            self.cols
        );
        if c1 == c2 {
            // Indices are equal; no operation required
            return;
        }

        // Iterate over each row to swap corresponding elements
        for r in 0..self.rows {
            // Compute the one-dimensional index for (row r, column c1)
            let idx1 = c1 * self.rows + r;
            // Compute the one-dimensional index for (row r, column c2)
            let idx2 = c2 * self.rows + r;

            // Exchange the two elements in the internal data buffer
            self.data.swap(idx1, idx2);
        }
    }

    /// Deletes a column from the matrix.
    pub fn delete_column(&mut self, col: usize) {
        assert!(col < self.cols, "column index out of bounds");
        for r in (0..self.rows).rev() {
            self.data.remove(col * self.rows + r);
        }
        self.cols -= 1;
    }

    /// Deletes a row from the matrix.
    pub fn delete_row(&mut self, row: usize) {
        assert!(row < self.rows, "row index out of bounds");
        for c in (0..self.cols).rev() {
            self.data.remove(c * self.rows + row);
        }
        self.rows -= 1;
    }
}

impl<T: Clone> Matrix<T> {
    /// Adds a column to the matrix at the specified index.
    pub fn add_column(&mut self, index: usize, column: Vec<T>) {
        assert!(index <= self.cols, "add_column index {} out of bounds for {} columns", index, self.cols);
        assert_eq!(column.len(), self.rows, "column length mismatch");

        for (r, value) in column.into_iter().enumerate() {
            self.data.insert(index * self.rows + r, value);
        }
        self.cols += 1;
    }

    /// Adds a row to the matrix at the specified index.
    pub fn add_row(&mut self, index: usize, row: Vec<T>) {
        assert!(index <= self.rows, "add_row index {} out of bounds for {} rows", index, self.rows);
        assert_eq!(row.len(), self.cols, "row length mismatch");

        for (c, value) in row.into_iter().enumerate() {
            self.data.insert(c * (self.rows + 1) + index, value);
        }
        self.rows += 1;
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &T {
        // Validate that the requested indices are within bounds
        assert!(
            r < self.rows && c < self.cols,
            "index out of bounds: ({}, {}) vs {}x{}",
            r,
            c,
            self.rows,
            self.cols
        );
        // Compute column-major offset and return reference
        &self.data[c * self.rows + r]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        // Validate that the requested indices are within bounds
        assert!(
            r < self.rows && c < self.cols,
            "index out of bounds: ({}, {}) vs {}x{}",
            r,
            c,
            self.rows,
            self.cols
        );
        // Compute column-major offset and return mutable reference
        &mut self.data[c * self.rows + r]
    }
}

/// Represents an immutable view of a single row in the matrix.
pub struct MatrixRow<'a, T> {
    matrix: &'a Matrix<T>,
    row: usize,
}

impl<'a, T> MatrixRow<'a, T> {
    /// Returns a reference to the element at the given column in this row.
    pub fn get(&self, c: usize) -> &T {
        &self.matrix[(self.row, c)]
    }

    /// Returns an iterator over all elements in this row.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.matrix.cols).map(move |c| &self.matrix[(self.row, c)])
    }
}

/// Specifies the axis along which to perform a reduction operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Apply reduction along columns (vertical axis).
    Col,
    /// Apply reduction along rows (horizontal axis).
    Row,
}

/// A trait to turn either a `Matrix<T>` or a scalar T into a `Vec<T>` of
/// length `rows*cols` (broadcasting the scalar).
pub trait Broadcastable<T> {
    fn to_vec(&self, rows: usize, cols: usize) -> Vec<T>;
}

impl<T: Clone> Broadcastable<T> for T {
    fn to_vec(&self, rows: usize, cols: usize) -> Vec<T> {
        vec![self.clone(); rows * cols]
    }
}

impl<T: Clone> Broadcastable<T> for Matrix<T> {
    fn to_vec(&self, rows: usize, cols: usize) -> Vec<T> {
        assert_eq!(self.rows, rows, "row count mismatch");
        assert_eq!(self.cols, cols, "col count mismatch");
        self.data.clone()
    }
}

/// Generates element-wise eq, lt, le, gt and ge methods
/// where the rhs can be a `Matrix<T>` or a scalar T.
macro_rules! impl_elementwise_cmp {
    (
        $( $method:ident => $op:tt ),* $(,)?
    ) => {
        impl<T: PartialOrd + Clone> Matrix<T> {
            $(
            #[doc = concat!("Element-wise comparison `self ", stringify!($op), " rhs`,\n\
                             where `rhs` may be a `Matrix<T>` or a scalar T.")]
            pub fn $method<Rhs>(&self, rhs: Rhs) -> BoolMatrix
            where
                Rhs: Broadcastable<T>,
            {
                // Prepare broadcasted rhs-data
                let rhs_data = rhs.to_vec(self.rows, self.cols);

                // Pairwise compare
                let data = self
                    .data
                    .iter()
                    .cloned()
                    .zip(rhs_data.into_iter())
                    .map(|(a, b)| a $op b)
                    .collect();

                BoolMatrix::from_vec(data, self.rows, self.cols)
            }
            )*
        }
    };
}

// Instantiate element-wise comparison implementations for matrices.
impl_elementwise_cmp! {
    eq_elementwise => ==,
    lt_elementwise => <,
    le_elementwise => <=,
    gt_elementwise => >,
    ge_elementwise => >=,
}

/// Generates element-wise arithmetic implementations for matrices.
macro_rules! impl_elementwise_op {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, 'b, T> std::ops::$OpTrait<&'b Matrix<T>> for &'a Matrix<T>
        where
            T: Clone + std::ops::$OpTrait<Output = T>,
        {
            type Output = Matrix<T>;

            fn $method(self, rhs: &'b Matrix<T>) -> Matrix<T> {
                // Ensure both matrices have identical dimensions
                assert_eq!(self.rows, rhs.rows, "row count mismatch");
                assert_eq!(self.cols, rhs.cols, "col count mismatch");
                // Apply the operation element-wise and collect into a new matrix
                let data = self
                    .data
                    .iter()
                    .cloned()
                    .zip(rhs.data.iter().cloned())
                    .map(|(a, b)| a $op b)
                    .collect();
                Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
    };
}

// Instantiate element-wise addition, subtraction, multiplication, and division
impl_elementwise_op!(Add, add, +);
impl_elementwise_op!(Sub, sub, -);
impl_elementwise_op!(Mul, mul, *);
impl_elementwise_op!(Div, div, /);

/// Generates element-wise arithmetic implementations for matrices with scalars.
macro_rules! impl_elementwise_op_scalar {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, T> std::ops::$OpTrait<T> for &'a Matrix<T>
        where
            T: Clone + std::ops::$OpTrait<Output = T>,
        {
            type Output = Matrix<T>;

            fn $method(self, rhs: T) -> Matrix<T> {
                // Apply the operation element-wise and collect into a new matrix
                let data = self.data.iter().cloned().map(|a| a $op rhs.clone()).collect();
                Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
    };
}

// Instantiate element-wise addition, subtraction, multiplication, and division
impl_elementwise_op_scalar!(Add, add, +);
impl_elementwise_op_scalar!(Sub, sub, -);
impl_elementwise_op_scalar!(Mul, mul, *);
impl_elementwise_op_scalar!(Div, div, /);

/// Generates element-wise bitwise operations for boolean matrices.
macro_rules! impl_bitwise_op {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, 'b> std::ops::$OpTrait<&'b Matrix<bool>> for &'a Matrix<bool> {
            type Output = Matrix<bool>;

            fn $method(self, rhs: &'b Matrix<bool>) -> Matrix<bool> {
                // Ensure both matrices have identical dimensions
                assert_eq!(self.rows, rhs.rows, "row count mismatch");
                assert_eq!(self.cols, rhs.cols, "col count mismatch");
                // Apply the bitwise operation element-wise
                let data = self
                    .data
                    .iter()
                    .cloned()
                    .zip(rhs.data.iter().cloned())
                    .map(|(a, b)| a $op b)
                    .collect();
                Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
    };
}

// Instantiate bitwise AND, OR, and XOR for boolean matrices
impl_bitwise_op!(BitAnd, bitand, &);
impl_bitwise_op!(BitOr, bitor, |);
impl_bitwise_op!(BitXor, bitxor, ^);

impl Not for Matrix<bool> {
    type Output = Matrix<bool>;

    fn not(self) -> Matrix<bool> {
        // Invert each boolean element in the matrix
        let data = self.data.iter().map(|&v| !v).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

pub type FloatMatrix = Matrix<f64>;
pub type BoolMatrix = Matrix<bool>;
pub type IntMatrix = Matrix<i32>;
pub type StringMatrix = Matrix<String>;

#[cfg(test)]
mod tests {
    use crate::matrix::BoolOps;

    use super::{BoolMatrix, FloatMatrix, Matrix, StringMatrix};

    // Helper function to create a basic Matrix for testing
    fn create_test_matrix() -> Matrix<i32> {
        // Column-major data:
        // 1 4 7
        // 2 5 8
        // 3 6 9
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        Matrix::from_vec(data, 3, 3)
    }

    // Another helper for a different size
    fn create_test_matrix_2x4() -> Matrix<i32> {
        // Column-major data:
        // 1 3 5 7
        // 2 4 6 8
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        Matrix::from_vec(data, 2, 4)
    }

    #[test]
    fn test_from_vec_basic() {
        let data = vec![1, 2, 3, 4, 5, 6]; // 2 rows, 3 cols (column-major)
        let matrix = Matrix::from_vec(data, 2, 3);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.data(), &[1, 2, 3, 4, 5, 6]);

        // Check some elements
        assert_eq!(matrix[(0, 0)], 1); // First row, first col
        assert_eq!(matrix[(1, 0)], 2); // Second row, first col
        assert_eq!(matrix[(0, 1)], 3); // First row, second col
        assert_eq!(matrix[(1, 2)], 6); // Second row, third col
    }

    #[test]
    #[should_panic(expected = "data length mismatch")]
    fn test_from_vec_wrong_length() {
        let data = vec![1, 2, 3, 4, 5]; // Should be 6 for 2x3
        Matrix::from_vec(data, 2, 3);
    }

    #[test]
    #[should_panic(expected = "need at least one row")]
    fn test_from_vec_zero_rows() {
        let data = vec![1, 2, 3];
        Matrix::from_vec(data, 0, 3);
    }

    #[test]
    #[should_panic(expected = "need at least one column")]
    fn test_from_vec_zero_cols() {
        let data = vec![1, 2, 3];
        Matrix::from_vec(data, 3, 0);
    }

    #[test]
    fn test_from_cols_basic() {
        // Representing:
        // 1 4 7
        // 2 5 8
        // 3 6 9
        let cols_data = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let matrix = Matrix::from_cols(cols_data);

        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        // Internal data should be column-major
        assert_eq!(matrix.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Check some elements
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(2, 0)], 3);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(0, 2)], 7);
    }

    #[test]
    fn test_from_cols_1x1() {
        let cols_data = vec![vec![42]];
        let matrix = Matrix::from_cols(cols_data);
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 1);
        assert_eq!(matrix.data(), &[42]);
        assert_eq!(matrix[(0, 0)], 42);
    }

    #[test]
    #[should_panic(expected = "need at least one column")]
    fn test_from_cols_empty_cols() {
        let empty_cols: Vec<Vec<i32>> = vec![];
        Matrix::from_cols(empty_cols);
    }

    #[test]
    #[should_panic(expected = "need at least one row")]
    fn test_from_cols_empty_rows() {
        let empty_row: Vec<Vec<String>> = vec![vec![], vec![]];
        Matrix::from_cols(empty_row);
    }

    #[test]
    #[should_panic(expected = "col 1 has len 2, expected 3")]
    fn test_from_cols_mismatched_lengths() {
        let cols_data = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]];
        Matrix::from_cols(cols_data);
    }

    #[test]
    fn test_getters() {
        let matrix = create_test_matrix();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_index_and_get() {
        let matrix = create_test_matrix();
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(2, 2)], 9);

        assert_eq!(*matrix.get(0, 0), 1);
        assert_eq!(*matrix.get(1, 1), 5);
        assert_eq!(*matrix.get(2, 2), 9);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_row() {
        let matrix = create_test_matrix(); // 3x3
        let _ = matrix[(3, 0)];
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_col() {
        let matrix = create_test_matrix(); // 3x3
        let _ = matrix[(0, 3)];
    }

    #[test]
    fn test_index_mut_and_get_mut() {
        let mut matrix = create_test_matrix(); // 3x3

        matrix[(0, 0)] = 10;
        matrix[(1, 1)] = 20;
        matrix[(2, 2)] = 30;

        assert_eq!(matrix[(0, 0)], 10);
        assert_eq!(matrix[(1, 1)], 20);
        assert_eq!(matrix[(2, 2)], 30);

        *matrix.get_mut(0, 1) = 15;
        *matrix.get_mut(2, 1) = 25;

        assert_eq!(matrix[(0, 1)], 15);
        assert_eq!(matrix[(2, 1)], 25);

        // Check underlying data consistency (column-major)
        // Should be:
        // 10 15  7
        //  2 20  8
        //  3 25 30
        assert_eq!(matrix.data(), &[10, 2, 3, 15, 20, 25, 7, 8, 30]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds_row() {
        let mut matrix = create_test_matrix(); // 3x3
        matrix[(3, 0)] = 99;
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds_col() {
        let mut matrix = create_test_matrix(); // 3x3
        matrix[(0, 3)] = 99;
    }

    #[test]
    fn test_column() {
        let matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        assert_eq!(matrix.column(0), &[1, 2]);
        assert_eq!(matrix.column(1), &[3, 4]);
        assert_eq!(matrix.column(2), &[5, 6]);
        assert_eq!(matrix.column(3), &[7, 8]);
    }

    #[test]
    #[should_panic(expected = "range end index")]
    fn test_column_out_of_bounds() {
        let matrix = create_test_matrix_2x4(); // 2x4
        matrix.column(4);
    }

    #[test]
    fn test_column_mut() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        let col1_mut = matrix.column_mut(1);
        col1_mut[0] = 30;
        col1_mut[1] = 40;

        let col3_mut = matrix.column_mut(3);
        col3_mut[0] = 70;

        // Check changes via indexing
        assert_eq!(matrix[(0, 1)], 30);
        assert_eq!(matrix[(1, 1)], 40);
        assert_eq!(matrix[(0, 3)], 70);
        assert_eq!(matrix[(1, 3)], 8); // Unchanged

        // Check underlying data (column-major)
        // Should be:
        // 1 30 5 70
        // 2 40 6  8
        assert_eq!(matrix.data(), &[1, 2, 30, 40, 5, 6, 70, 8]);
    }

    #[test]
    #[should_panic(expected = "range end index")]
    fn test_column_mut_out_of_bounds() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        matrix.column_mut(4);
    }

    #[test]
    fn test_iter_columns() {
        let matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        let cols: Vec<&[i32]> = matrix.iter_columns().collect();
        assert_eq!(cols.len(), 4);
        assert_eq!(cols[0], &[1, 2]);
        assert_eq!(cols[1], &[3, 4]);
        assert_eq!(cols[2], &[5, 6]);
        assert_eq!(cols[3], &[7, 8]);
    }

    #[test]
    fn test_iter_rows() {
        let matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        let rows: Vec<Vec<i32>> = matrix
            .iter_rows()
            .map(|row| row.iter().cloned().collect())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![1, 3, 5, 7]);
        assert_eq!(rows[1], vec![2, 4, 6, 8]);
    }

    // test data_mut
    #[test]
    fn test_data_mut() {
        let mut matrix = create_test_matrix(); // 3x3
        // 1 4 7
        // 2 5 8
        // 3 6 9

        let data_mut = matrix.data_mut();
        data_mut[0] = 10;
        data_mut[1] = 20;

        assert_eq!(matrix[(0, 0)], 10);
        assert_eq!(matrix[(1, 0)], 20);
    }

    #[test]
    fn test_matrix_row_get_and_iter() {
        let matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        let row0 = matrix.iter_rows().next().unwrap();
        assert_eq!(*row0.get(0), 1);
        assert_eq!(*row0.get(1), 3);
        assert_eq!(*row0.get(3), 7);
        let row0_vec: Vec<i32> = row0.iter().cloned().collect();
        assert_eq!(row0_vec, vec![1, 3, 5, 7]);

        let row1 = matrix.iter_rows().nth(1).unwrap();
        assert_eq!(*row1.get(0), 2);
        assert_eq!(*row1.get(2), 6);
        let row1_vec: Vec<i32> = row1.iter().cloned().collect();
        assert_eq!(row1_vec, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_swap_columns() {
        let mut matrix = create_test_matrix(); // 3x3
        // 1 4 7
        // 2 5 8
        // 3 6 9

        matrix.swap_columns(0, 2); // Swap first and last

        // Should be:
        // 7 4 1
        // 8 5 2
        // 9 6 3

        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], 7);
        assert_eq!(matrix[(1, 0)], 8);
        assert_eq!(matrix[(2, 0)], 9);
        assert_eq!(matrix[(0, 1)], 4); // Middle col unchanged
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(2, 1)], 6);
        assert_eq!(matrix[(0, 2)], 1);
        assert_eq!(matrix[(1, 2)], 2);
        assert_eq!(matrix[(2, 2)], 3);

        // Swap the same column (should do nothing)
        let original_data = matrix.data().to_vec();
        matrix.swap_columns(1, 1);
        assert_eq!(matrix.data(), &original_data); // Data should be identical

        // Check underlying data (column-major) after swap(0, 2)
        assert_eq!(matrix.data(), &[7, 8, 9, 4, 5, 6, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "column index out of bounds")]
    fn test_swap_columns_out_of_bounds() {
        let mut matrix = create_test_matrix(); // 3x3
        matrix.swap_columns(0, 3);
    }

    #[test]
    fn test_delete_column() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        matrix.delete_column(1); // Delete the second column (index 1)

        // Should be:
        // 1 5 7
        // 2 6 8

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 0)], 2);
        assert_eq!(matrix[(0, 1)], 5);
        assert_eq!(matrix[(1, 1)], 6);
        assert_eq!(matrix[(0, 2)], 7);
        assert_eq!(matrix[(1, 2)], 8);

        // Check underlying data (column-major)
        assert_eq!(matrix.data(), &[1, 2, 5, 6, 7, 8]);

        // Delete the first column
        matrix.delete_column(0);
        // Should be:
        // 5 7
        // 6 8
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.data(), &[5, 6, 7, 8]);

        // Delete the last column
        matrix.delete_column(1);
        // Should be:
        // 5
        // 6
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 1);
        assert_eq!(matrix.data(), &[5, 6]);

        // Delete the only column
        matrix.delete_column(0);
        // Should be empty
        assert_eq!(matrix.rows(), 2); // Rows stay the same
        assert_eq!(matrix.cols(), 0); // Cols becomes 0
        assert_eq!(matrix.data(), &[]);
    }

    #[test]
    #[should_panic(expected = "column index out of bounds")]
    fn test_delete_column_out_of_bounds() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        matrix.delete_column(4);
    }

    #[test]
    fn test_delete_row() {
        let mut matrix = create_test_matrix(); // 3x3
        // 1 4 7
        // 2 5 8
        // 3 6 9

        matrix.delete_row(1); // Delete the second row (index 1)

        // Should be:
        // 1 4 7
        // 3 6 9

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(0, 1)], 4);
        assert_eq!(matrix[(1, 1)], 6);
        assert_eq!(matrix[(0, 2)], 7);
        assert_eq!(matrix[(1, 2)], 9);

        // Check underlying data (column-major)
        // Original: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        // Delete row 1: [1, 3, 4, 6, 7, 9]
        assert_eq!(matrix.data(), &[1, 3, 4, 6, 7, 9]);

        // Delete the first row
        matrix.delete_row(0);
        // Should be:
        // 3 6 9
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.data(), &[3, 6, 9]);

        // Delete the last (and only) row
        matrix.delete_row(0);
        // Should be empty
        assert_eq!(matrix.rows(), 0); // Rows becomes 0
        assert_eq!(matrix.cols(), 3); // Cols stay the same
        assert_eq!(matrix.data(), &[]);
    }

    #[test]
    #[should_panic(expected = "row index out of bounds")]
    fn test_delete_row_out_of_bounds() {
        let mut matrix = create_test_matrix(); // 3x3
        matrix.delete_row(3);
    }

    #[test]
    fn test_add_column() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        let new_col = vec![9, 10];
        matrix.add_column(2, new_col); // Add at index 2

        // Should be:
        // 1 3 9 5 7
        // 2 4 10 6 8

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 5);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 0)], 2);
        assert_eq!(matrix[(0, 1)], 3);
        assert_eq!(matrix[(1, 1)], 4);
        assert_eq!(matrix[(0, 2)], 9);
        assert_eq!(matrix[(1, 2)], 10);
        assert_eq!(matrix[(0, 3)], 5); // Shifted
        assert_eq!(matrix[(1, 3)], 6);
        assert_eq!(matrix[(0, 4)], 7); // Shifted
        assert_eq!(matrix[(1, 4)], 8);

        // Check underlying data (column-major)
        // Original: [1, 2, 3, 4, 5, 6, 7, 8]
        // Add [9, 10] at index 2: [1, 2, 3, 4, 9, 10, 5, 6, 7, 8]
        assert_eq!(matrix.data(), &[1, 2, 3, 4, 9, 10, 5, 6, 7, 8]);

        // Add a column at the beginning
        let new_col_start = vec![11, 12];
        matrix.add_column(0, new_col_start);
        // Should be:
        // 11 1 3 9 5 7
        // 12 2 4 10 6 8
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 6);
        assert_eq!(matrix[(0, 0)], 11);
        assert_eq!(matrix[(1, 0)], 12);
        assert_eq!(matrix.data(), &[11, 12, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8]);

        // Add a column at the end
        let new_col_end = vec![13, 14];
        matrix.add_column(6, new_col_end);
        // Should be:
        // 11 1 3 9 5 7 13
        // 12 2 4 10 6 8 14
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 7);
        assert_eq!(matrix[(0, 6)], 13);
        assert_eq!(matrix[(1, 6)], 14);
        assert_eq!(
            matrix.data(),
            &[11, 12, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 13, 14]
        );
    }

    #[test]
    #[should_panic(expected = "column index out of bounds")]
    fn test_add_column_out_of_bounds() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        let new_col = vec![9, 10];
        matrix.add_column(5, new_col); // Index 5 is out of bounds for 4 columns
    }

    #[test]
    #[should_panic(expected = "column length mismatch")]
    fn test_add_column_length_mismatch() {
        let mut matrix = create_test_matrix_2x4(); // 2x4 (2 rows)
        let new_col = vec![9, 10, 11]; // Wrong length
        matrix.add_column(0, new_col);
    }

    #[test]
    fn test_add_row() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        // 1 3 5 7
        // 2 4 6 8

        let new_row = vec![9, 10, 11, 12];
        matrix.add_row(1, new_row); // Add at index 1

        // Should be:
        // 1  3  5  7
        // 9 10 11 12
        // 2  4  6  8

        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);

        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 3);
        assert_eq!(matrix[(0, 2)], 5);
        assert_eq!(matrix[(0, 3)], 7);
        assert_eq!(matrix[(1, 0)], 9);
        assert_eq!(matrix[(1, 1)], 10);
        assert_eq!(matrix[(1, 2)], 11);
        assert_eq!(matrix[(1, 3)], 12);
        assert_eq!(matrix[(2, 0)], 2);
        assert_eq!(matrix[(2, 1)], 4);
        assert_eq!(matrix[(2, 2)], 6);
        assert_eq!(matrix[(2, 3)], 8);

        // Check underlying data (column-major)
        // Original: [1, 2, 3, 4, 5, 6, 7, 8] (rows 0, 1)
        // Add [9, 10, 11, 12] at index 1 (new row will be index 1, original row 1 becomes index 2)
        // Col 0: [1, 9, 2]
        // Col 1: [3, 10, 4]
        // Col 2: [5, 11, 6]
        // Col 3: [7, 12, 8]
        // Data: [1, 9, 2, 3, 10, 4, 5, 11, 6, 7, 12, 8]
        assert_eq!(matrix.data(), &[1, 9, 2, 3, 10, 4, 5, 11, 6, 7, 12, 8]);

        // Add a row at the beginning
        let new_row_start = vec![13, 14, 15, 16];
        matrix.add_row(0, new_row_start);
        // Should be:
        // 13 14 15 16
        // 1  3  5  7
        // 9 10 11 12
        // 2  4  6  8
        assert_eq!(matrix.rows(), 4);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix[(0, 0)], 13);
        assert_eq!(matrix[(0, 3)], 16);
        // Check some existing elements to ensure they shifted correctly
        assert_eq!(matrix[(1, 0)], 1);
        assert_eq!(matrix[(2, 1)], 10);
        assert_eq!(matrix[(3, 3)], 8);

        // Add a row at the end
        let new_row_end = vec![17, 18, 19, 20];
        matrix.add_row(4, new_row_end);
        // Should be:
        // 13 14 15 16
        // 1  3  5  7
        // 9 10 11 12
        // 2  4  6  8
        // 17 18 19 20
        assert_eq!(matrix.rows(), 5);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix[(4, 0)], 17);
        assert_eq!(matrix[(4, 3)], 20);
    }

    #[test]
    #[should_panic(expected = "row index out of bounds")]
    fn test_add_row_out_of_bounds() {
        let mut matrix = create_test_matrix_2x4(); // 2x4
        let new_row = vec![9, 10, 11, 12];
        matrix.add_row(3, new_row); // Index 3 is out of bounds for 2 rows
    }

    #[test]
    #[should_panic(expected = "row length mismatch")]
    fn test_add_row_length_mismatch() {
        let mut matrix = create_test_matrix_2x4(); // 2x4 (4 cols)
        let new_row = vec![9, 10, 11]; // Wrong length
        matrix.add_row(0, new_row);
    }

    #[test]
    fn test_elementwise_add() {
        let matrix1 = create_test_matrix(); // 3x3
        let matrix2 = Matrix::from_vec(vec![9, 8, 7, 6, 5, 4, 3, 2, 1], 3, 3); // 3x3

        let result = &matrix1 + &matrix2;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Expected:
        // 1+9  4+6  7+3  => 10 10 10
        // 2+8  5+5  8+2  => 10 10 10
        // 3+7  6+4  9+1  => 10 10 10
        // Column-major data: [10, 10, 10, 10, 10, 10, 10, 10, 10]
        assert_eq!(result.data(), &[10, 10, 10, 10, 10, 10, 10, 10, 10]);
        assert_eq!(result[(0, 0)], 10);
        assert_eq!(result[(1, 1)], 10);
        assert_eq!(result[(2, 2)], 10);
    }

    #[test]
    fn test_elementwise_sub() {
        let matrix1 = create_test_matrix(); // 3x3
        let matrix2 = Matrix::from_vec(vec![1, 1, 1, 2, 2, 2, 3, 3, 3], 3, 3); // 3x3

        let result = &matrix1 - &matrix2;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Expected:
        // 1-1  4-2  7-3  => 0 2 4
        // 2-1  5-2  8-3  => 1 3 5
        // 3-1  6-2  9-3  => 2 4 6
        // Column-major data: [0, 1, 2, 2, 3, 4, 4, 5, 6]
        assert_eq!(result.data(), &[0, 1, 2, 2, 3, 4, 4, 5, 6]);
        assert_eq!(result[(0, 0)], 0);
        assert_eq!(result[(1, 1)], 3);
        assert_eq!(result[(2, 2)], 6);
    }

    #[test]
    fn test_elementwise_mul() {
        let matrix1 = create_test_matrix(); // 3x3
        let matrix2 = Matrix::from_vec(vec![1, 2, 3, 1, 2, 3, 1, 2, 3], 3, 3); // 3x3

        let result = &matrix1 * &matrix2;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Expected:
        // 1*1  4*1  7*1  => 1 4 7
        // 2*2  5*2  8*2  => 4 10 16
        // 3*3  6*3  9*3  => 9 18 27
        // Column-major data: [1, 4, 9, 4, 10, 18, 7, 16, 27]
        assert_eq!(result.data(), &[1, 4, 9, 4, 10, 18, 7, 16, 27]);
        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(1, 1)], 10);
        assert_eq!(result[(2, 2)], 27);
    }

    #[test]
    fn test_elementwise_div() {
        let matrix1 = create_test_matrix(); // 3x3
        let matrix2 = Matrix::from_vec(vec![1, 1, 1, 2, 2, 2, 7, 8, 9], 3, 3); // 3x3

        let result = &matrix1 / &matrix2; // Integer division

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Expected:
        // 1/1  4/2  7/7  => 1 2 1
        // 2/1  5/2  8/8  => 2 2 1  (integer division)
        // 3/1  6/2  9/9  => 3 3 1
        // Column-major data: [1, 2, 3, 2, 2, 3, 1, 1, 1]
        assert_eq!(result.data(), &[1, 2, 3, 2, 2, 3, 1, 1, 1]);
        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(1, 1)], 2);
        assert_eq!(result[(2, 2)], 1);
    }

    #[test]
    #[should_panic(expected = "row count mismatch")]
    fn test_elementwise_op_row_mismatch() {
        let matrix1 = create_test_matrix(); // 3x3
        let matrix2 = create_test_matrix_2x4(); // 2x4
        let _ = &matrix1 + &matrix2; // Should panic
    }

    #[test]
    #[should_panic(expected = "row count mismatch")]
    fn test_elementwise_op_col_mismatch() {
        let matrix1 = create_test_matrix(); // 3x3
        let matrix2 = create_test_matrix_2x4(); // 2x4
        let _ = &matrix1 * &matrix2; // Should panic
    }

    #[test]
    fn test_bitwise_and() {
        let data1 = vec![true, false, true, false, true, false]; // 2x3
        let data2 = vec![true, true, false, false, true, true]; // 2x3
        let matrix1 = BoolMatrix::from_vec(data1, 2, 3);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);

        // Expected column-major results:
        // T & T = T
        // F & T = F
        // T & F = F
        // F & F = F
        // T & T = T
        // F & T = F
        // Data: [T, F, F, F, T, F]
        let expected_data = vec![true, false, false, false, true, false];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = &matrix1 & &matrix2;
        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn test_bitwise_or() {
        let data1 = vec![true, false, true, false, true, false]; // 2x3
        let data2 = vec![true, true, false, false, true, true]; // 2x3
        let matrix1 = BoolMatrix::from_vec(data1, 2, 3);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);

        // Expected column-major results:
        // T | T = T
        // F | T = T
        // T | F = T
        // F | F = F
        // T | T = T
        // F | T = T
        // Data: [T, T, T, F, T, T]
        let expected_data = vec![true, true, true, false, true, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = &matrix1 | &matrix2;
        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn test_bitwise_xor() {
        let data1 = vec![true, false, true, false, true, false]; // 2x3
        let data2 = vec![true, true, false, false, true, true]; // 2x3
        let matrix1 = BoolMatrix::from_vec(data1, 2, 3);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);

        // Expected column-major results:
        // T ^ T = F
        // F ^ T = T
        // T ^ F = T
        // F ^ F = F
        // T ^ T = F
        // F ^ T = T
        // Data: [F, T, T, F, F, T]
        let expected_data = vec![false, true, true, false, false, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = &matrix1 ^ &matrix2;
        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn test_bitwise_not() {
        let data = vec![true, false, true, false, true, false]; // 2x3
        let matrix = BoolMatrix::from_vec(data, 2, 3);

        // Expected column-major results:
        // !T = F
        // !F = T
        // !T = F
        // !F = T
        // !T = F
        // !F = T
        // Data: [F, T, F, T, F, T]
        let expected_data = vec![false, true, false, true, false, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = !matrix; // Not consumes the matrix
        assert_eq!(result, expected_matrix);
    }

    #[test]
    #[should_panic(expected = "col count mismatch")]
    fn test_bitwise_op_row_mismatch() {
        let data1 = vec![true, false, true, false]; // 2x2
        let data2 = vec![true, true, false, false, true, true]; // 2x3
        let matrix1 = BoolMatrix::from_vec(data1, 2, 2);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);
        let _ = &matrix1 & &matrix2; // Should panic
    }

    #[test]
    #[should_panic(expected = "col count mismatch")]
    fn test_bitwise_op_col_mismatch() {
        let data1 = vec![true, false, true, false]; // 2x2
        let data2 = vec![true, true, false, false, true, true]; // 2x3
        let matrix1 = BoolMatrix::from_vec(data1, 2, 2);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);
        let _ = &matrix1 | &matrix2; // Should panic
    }

    // Test with String type (requires Clone, PartialEq)
    #[test]
    fn test_string_matrix() {
        let data = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let matrix = StringMatrix::from_vec(data.clone(), 2, 2); // 2x2

        assert_eq!(matrix[(0, 0)], "a".to_string());
        assert_eq!(matrix[(1, 0)], "b".to_string());
        assert_eq!(matrix[(0, 1)], "c".to_string());
        assert_eq!(matrix[(1, 1)], "d".to_string());

        // Test modification
        let mut matrix = matrix;
        matrix[(0, 0)] = "hello".to_string();
        assert_eq!(matrix[(0, 0)], "hello".to_string());

        // Test add_column (requires Clone)
        let new_col = vec!["e".to_string(), "f".to_string()];
        matrix.add_column(1, new_col); // Add at index 1
        // Should be:
        // hello c d
        // b     e f
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], "hello".to_string());
        assert_eq!(matrix[(1, 0)], "b".to_string());
        assert_eq!(matrix[(0, 1)], "e".to_string()); // New col
        assert_eq!(matrix[(1, 1)], "f".to_string()); // New col
        assert_eq!(matrix[(0, 2)], "c".to_string()); // Shifted
        assert_eq!(matrix[(1, 2)], "d".to_string()); // Shifted

        // Test add_row (requires Clone)
        let new_row = vec!["g".to_string(), "h".to_string(), "i".to_string()];
        matrix.add_row(0, new_row); // Add at index 0
        // Should be:
        // g     h     i
        // hello e c
        // b     f d
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], "g".to_string());
        assert_eq!(matrix[(0, 1)], "h".to_string());
        assert_eq!(matrix[(0, 2)], "i".to_string());
        assert_eq!(matrix[(1, 0)], "hello".to_string()); // Shifted
        assert_eq!(matrix[(2, 2)], "d".to_string()); // Shifted
    }

    #[test]
    fn test_float_matrix_ops() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let data2 = vec![0.5, 1.5, 2.5, 3.5]; // 2x2
        let matrix1 = FloatMatrix::from_vec(data1, 2, 2);
        let matrix2 = FloatMatrix::from_vec(data2, 2, 2);

        let sum = &matrix1 + &matrix2;
        let diff = &matrix1 - &matrix2;
        let prod = &matrix1 * &matrix2;
        let div = &matrix1 / &matrix2;

        // Check sums (col-major): [1.5, 3.5, 5.5, 7.5]
        assert_eq!(sum.data(), &[1.5, 3.5, 5.5, 7.5]);

        // Check diffs (col-major): [0.5, 0.5, 0.5, 0.5]
        assert_eq!(diff.data(), &[0.5, 0.5, 0.5, 0.5]);

        // Check prods (col-major): [0.5, 3.0, 7.5, 14.0]
        assert_eq!(prod.data(), &[0.5, 3.0, 7.5, 14.0]);

        // Check divs (col-major): [2.0, 1.333..., 1.2, 1.14...]
        // Using element access for more specific checks on floating point results
        assert_eq!(div.rows(), 2);
        assert_eq!(div.cols(), 2);
        assert!((div[(0, 0)] - 1.0 / 0.5).abs() < 1e-9); // 2.0
        assert!((div[(1, 0)] - 2.0 / 1.5).abs() < 1e-9); // 1.333...
        assert!((div[(0, 1)] - 3.0 / 2.5).abs() < 1e-9); // 1.2
        assert!((div[(1, 1)] - 4.0 / 3.5).abs() < 1e-9); // 1.14...
    }

    fn create_test_matrix_i32() -> Matrix<i32> {
        Matrix::from_cols(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
    }

    #[test]
    fn test_matrix_swap_columns_directly() {
        let mut matrix = create_test_matrix_i32();

        // Store the initial state of the columns we intend to swap AND one that shouldn't change
        let initial_col0_data = matrix.column(0).to_vec(); // Should be [1, 2, 3]
        let initial_col1_data = matrix.column(1).to_vec(); // Should be [4, 5, 6]
        let initial_col2_data = matrix.column(2).to_vec(); // Should be [7, 8, 9]

        // Perform the swap directly on the matrix
        matrix.swap_columns(0, 2); // Swap column 0 and column 2

        // --- Assertions ---

        // 1. Verify the dimensions are unchanged
        assert_eq!(matrix.rows(), 3, "Matrix rows should remain unchanged");
        assert_eq!(matrix.cols(), 3, "Matrix cols should remain unchanged");

        // 2. Verify the column that was NOT swapped is unchanged
        assert_eq!(
            matrix.column(1),
            initial_col1_data.as_slice(), // Comparing slice to slice
            "Column 1 data should be unchanged"
        );

        // 3. Verify the data swap occurred correctly using the COLUMN ACCESSOR
        // The data originally at index 0 should now be at index 2
        assert_eq!(
            matrix.column(2),
            initial_col0_data.as_slice(),
            "Column 2 should now contain the original data from column 0"
        );
        // The data originally at index 2 should now be at index 0
        assert_eq!(
            matrix.column(0),
            initial_col2_data.as_slice(),
            "Column 0 should now contain the original data from column 2"
        );

        // 4. (Optional but useful) Verify the underlying raw data vector
        // Original data: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        // Expected data after swapping col 0 and col 2: [7, 8, 9, 4, 5, 6, 1, 2, 3]
        assert_eq!(
            matrix.data(),
            &[7, 8, 9, 4, 5, 6, 1, 2, 3],
            "Underlying data vector is incorrect after swap"
        );

        // 5. Test swapping with self (should be a no-op)
        let state_before_self_swap = matrix.clone();
        matrix.swap_columns(1, 1);
        assert_eq!(
            matrix, state_before_self_swap,
            "Swapping a column with itself should not change the matrix"
        );

        // 6. Test swapping adjacent columns
        let mut matrix2 = create_test_matrix_i32();
        let initial_col0_data_m2 = matrix2.column(0).to_vec();
        let initial_col1_data_m2 = matrix2.column(1).to_vec();
        matrix2.swap_columns(0, 1);
        assert_eq!(matrix2.column(0), initial_col1_data_m2.as_slice());
        assert_eq!(matrix2.column(1), initial_col0_data_m2.as_slice());
        assert_eq!(matrix2.data(), &[4, 5, 6, 1, 2, 3, 7, 8, 9]);
    }

    // Test broadcastable operations
    #[test]
    fn test_comparision_broadcast() {
        let matrix = create_test_matrix();
        // test all > 0
        let result = matrix.gt_elementwise(0).as_vec();
        let expected = vec![true; result.len()];
        assert_eq!(result, expected);

        let ma = create_test_matrix();
        let mb = create_test_matrix();

        let result = ma.eq_elementwise(mb);
        assert!(result.all());

        let result = matrix.lt_elementwise(1e10 as i32).all();
        assert!(result);

        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                let vx = matrix[(i, j)];
                let c = &(matrix.le_elementwise(vx)) & &(matrix.ge_elementwise(vx));
                assert_eq!(c.count(), 1);
            }
        }
    }

    #[test]
    fn test_arithmetic_broadcast() {
        let matrix = create_test_matrix();
        let result = &matrix + 1;
        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                assert_eq!(result[(i, j)], matrix[(i, j)] + 1);
            }
        }

        // test mul and div
        let result = &matrix * 2;
        let result2 = &matrix / 2;
        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                assert_eq!(result[(i, j)], matrix[(i, j)] * 2);
                assert_eq!(result2[(i, j)], matrix[(i, j)] / 2);
            }
        }
    }
}
