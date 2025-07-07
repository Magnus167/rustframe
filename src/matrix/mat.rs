//! A simple column-major Matrix implementation with element-wise operations.

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Index, IndexMut, Mul, Not, Sub};

/// A column‑major 2D matrix of `T`. Index as `Array(row, column)`.
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

    /// Build from a flat Vec, assuming column-major order.
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 || cols == 0,
            "need at least one row if columns exist"
        );
        assert!(
            cols > 0 || rows == 0,
            "need at least one column if rows exist"
        );
        if rows * cols != 0 {
            // Only assert length if matrix is non-empty
            assert_eq!(
                data.len(),
                rows * cols,
                "data length mismatch: expected {}, got {}",
                rows * cols,
                data.len()
            );
        } else {
            assert!(data.is_empty(), "data must be empty for 0-sized matrix");
        }

        Matrix { rows, cols, data }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the Matrix and returns its underlying data Vec.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Creates a new `Vec<T>` containing the matrix data (cloned).
    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get element reference (immutable). Panics on out-of-bounds.
    pub fn get(&self, r: usize, c: usize) -> &T {
        &self[(r, c)]
    }

    /// Get element reference (mutable). Panics on out-of-bounds.
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

    /// Swaps two columns in the matrix. Panics on out-of-bounds.
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
        if c1 == c2 || self.rows == 0 || self.cols == 0 {
            return;
        }

        let (start1, end1) = (c1 * self.rows, (c1 + 1) * self.rows);
        let (start2, end2) = (c2 * self.rows, (c2 + 1) * self.rows);

        if (start1 < start2 && end1 > start2) || (start2 < start1 && end2 > start1) {
            panic!("Cannot swap overlapping columns");
        }

        // element-wise swap
        for r in 0..self.rows {
            self.data.swap(start1 + r, start2 + r);
        }
    }

    /// Deletes a column from the matrix. Panics on out-of-bounds.
    /// This is O(N) where N is the number of elements.
    pub fn delete_column(&mut self, col: usize) {
        assert!(
            col < self.cols,
            "column index {} out of bounds for {} columns",
            col,
            self.cols
        );
        let start = col * self.rows;
        self.data.drain(start..start + self.rows); // Efficient removal
        self.cols -= 1;
    }

    #[inline]
    pub fn row(&self, r: usize) -> Vec<T> {
        assert!(
            r < self.rows,
            "row index {} out of bounds for {} rows",
            r,
            self.rows
        );
        let mut row_data = Vec::with_capacity(self.cols);
        for c in 0..self.cols {
            row_data.push(self[(r, c)].clone()); // Clone each element
        }
        row_data
    }
    pub fn row_copy_from_slice(&mut self, r: usize, values: &[T]) {
        assert!(
            r < self.rows,
            "row index {} out of bounds for {} rows",
            r,
            self.rows
        );
        assert!(
            values.len() == self.cols,
            "input slice length {} does not match number of columns {}",
            values.len(),
            self.cols
        );

        for (c, value) in values.iter().enumerate() {
            let idx = r + c * self.rows; // column-major index
            self.data[idx] = value.clone();
        }
    }

    /// Deletes a row from the matrix. Panics on out-of-bounds.
    /// This is O(N) where N is the number of elements, as it rebuilds the data vec.
    pub fn delete_row(&mut self, row: usize) {
        assert!(
            row < self.rows,
            "row index {} out of bounds for {} rows",
            row,
            self.rows
        );
        if self.rows == 0 {
            return;
        } // Nothing to delete

        let old_rows = self.rows;
        let new_rows = self.rows - 1;
        let mut new_data = Vec::with_capacity(new_rows * self.cols);

        for c in 0..self.cols {
            let col_start_old = c * old_rows;
            for r in 0..old_rows {
                if r != row {
                    // Must clone as we are reading from the old data while building the new one
                    new_data.push(self.data[col_start_old + r].clone());
                }
            }
        }
        self.data = new_data;
        self.rows = new_rows;
    }

    pub fn transpose(&self) -> Matrix<T> {
        let (m, n) = (self.rows, self.cols);
        let mut transposed_data = Vec::with_capacity(m * n);

        // In the transposed matrix the old rows become the new columns.
        for j in 0..m {
            // new column index = old row index
            for i in 0..n {
                // new row index = old col index
                transposed_data.push(self[(j, i)].clone()); // A(T)[i,j] = A[j,i]
            }
        }

        Matrix::from_vec(transposed_data, n, m) // size is n × m
    }
}

impl<T: Clone> Matrix<T> {
    /// Adds a column to the matrix at the specified index. Panics if index > cols or length mismatch.
    /// This is O(N) where N is the number of elements.
    pub fn add_column(&mut self, index: usize, column: Vec<T>) {
        assert!(
            index <= self.cols,
            "add_column index {} out of bounds for {} columns",
            index,
            self.cols
        );
        assert_eq!(
            column.len(),
            self.rows,
            "column length mismatch: expected {}, got {}",
            self.rows,
            column.len()
        );
        if self.rows == 0 && self.cols == 0 {
            // Special case: adding first column to empty matrix
            assert!(index == 0, "index must be 0 for adding first column");
            self.data = column;
            self.cols = 1;
            // self.rows should be correctly set by column.len() assertion
        } else {
            let insert_pos = index * self.rows;
            self.data.splice(insert_pos..insert_pos, column); // Efficient insertion
            self.cols += 1;
        }
    }

    /// Adds a row to the matrix at the specified index. Panics if index > rows or length mismatch.
    /// This is O(N) where N is the number of elements, as it rebuilds the data vec.
    pub fn add_row(&mut self, index: usize, row: Vec<T>) {
        assert!(
            index <= self.rows,
            "add_row index {} out of bounds for {} rows",
            index,
            self.rows
        );
        assert_eq!(
            row.len(),
            self.cols,
            "row length mismatch: expected {} (cols), got {}",
            self.cols,
            row.len()
        );

        if self.cols == 0 && self.rows == 0 {
            // Special case: adding first row to empty matrix
            assert!(index == 0, "index must be 0 for adding first row");
            // Cannot add a row if there are no columns yet. Maybe panic or change API?
            assert!(
                self.cols > 0 || row.is_empty(),
                "cannot add non-empty row to matrix with 0 columns"
            );
            if row.is_empty() {
                return;
            } // Adding empty row to empty matrix is no-op
        }

        let old_rows = self.rows;
        let new_rows = self.rows + 1;
        let mut new_data = Vec::with_capacity(new_rows * self.cols);
        let mut row_iter = row.into_iter(); // Consume the input row vec

        for c in 0..self.cols {
            let old_col_start = c * old_rows;
            for r in 0..new_rows {
                if r == index {
                    // Take the next element from the provided row vector
                    new_data.push(row_iter.next().expect("Row iterator exhausted prematurely - should have been caught by length assert"));
                } else {
                    // Calculate the corresponding old row index
                    let old_r = if r < index { r } else { r - 1 };
                    // Must clone as we are reading from the old data while building the new one
                    new_data.push(self.data[old_col_start + old_r].clone());
                }
            }
        }
        self.data = new_data;
        self.rows = new_rows;
    }

    /// Return a new matrix where row 0 of `self` is repeated `n` times.
    pub fn repeat_rows(&self, n: usize) -> Matrix<T>
    where
        T: Clone,
    {
        let mut data = Vec::with_capacity(n * self.cols());
        let zeroth_row = self.row(0);
        for value in &zeroth_row {
            for _ in 0..n {
                data.push(value.clone()); // Clone each element
            }
        }
        Matrix::from_vec(data, n, self.cols)
    }

    /// Creates a new matrix filled with a specific value of the specified size.
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols], // Fill with the specified value
        }
    }
}

impl Matrix<f64> {
    /// Creates a new matrix filled with zeros of the specified size.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::filled(rows, cols, 0.0)
    }

    /// Creates a new matrix filled with ones of the specified size.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix::filled(rows, cols, 1.0)
    }

    /// Creates a new matrix filled with NaN values of the specified size.
    pub fn nan(rows: usize, cols: usize) -> Matrix<f64> {
        Matrix::filled(rows, cols, f64::NAN)
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

// --- Row Iterator Helper ---

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
        (0..self.matrix.cols).map(move |c| self.get(c))
    }
}

// --- Reduction Axis Enum ---

/// Specifies the axis along which to perform a reduction operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Apply reduction along columns (vertical axis). Result has 1 row.
    Col,
    /// Apply reduction along rows (horizontal axis). Result has 1 column.
    Row,
}

// --- Broadcasting ---

/// A trait to turn either a `Matrix<T>` or a scalar T into a `Vec<T>` of
/// length `rows*cols` (broadcasting the scalar). Used for comparisons.
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
        assert_eq!(self.rows, rows, "row count mismatch in broadcast");
        assert_eq!(self.cols, cols, "col count mismatch in broadcast");
        self.data.clone() // Clone the data for the broadcasted vec
    }
}

// --- Element-wise Comparisons ---

/// Generates element-wise eq, lt, le, gt and ge methods
/// where the rhs can be a `Matrix<T>` or a scalar T.
macro_rules! impl_elementwise_cmp {
    (
        $( $method:ident => $op:tt ),* $(,)?
    ) => {
        impl<T: PartialOrd + Clone> Matrix<T> {
            $(
            #[doc = concat!("Element-wise comparison `self ", stringify!($op), " rhs`,\n\
                             where `rhs` may be a `Matrix<T>` or a scalar T.\n\
                             Returns a `BoolMatrix`.")]
            pub fn $method<Rhs>(&self, rhs: Rhs) -> BoolMatrix
            where
                Rhs: Broadcastable<T>,
            {
                // Prepare broadcasted rhs-data using the trait
                let rhs_data = rhs.to_vec(self.rows, self.cols);

                // Pairwise compare using iterators
                let data = self
                    .data
                    .iter() // Borrow self's data
                    .zip(rhs_data.iter()) // Borrow rhs's broadcasted data
                    .map(|(a, b)| a $op b) // Perform comparison op
                    .collect();

                // Create BoolMatrix from result
                Matrix::<bool>::from_vec(data, self.rows, self.cols)
            }
            )*
        }
    };
}

// Instantiate element-wise comparison implementations for matrices.
impl_elementwise_cmp! {
    eq_elem => ==,
    ne_elem => !=,
    lt_elem => <,
    le_elem => <=,
    gt_elem => >,
    ge_elem => >=,
}

// --- Element-wise Arithmetic Operations (Macros generating all ownership variants) ---

fn check_matrix_dims_for_ops<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) {
    assert!(
        lhs.rows == rhs.rows,
        "Row count mismatch: left has {} rows, right has {} rows",
        lhs.rows,
        rhs.rows
    );
    assert!(
        lhs.cols == rhs.cols,
        "Column count mismatch: left has {} columns, right has {} columns",
        lhs.cols,
        rhs.cols
    );
}

/// Generates element-wise arithmetic implementations for Matrix + Matrix
macro_rules! impl_elementwise_op_matrix_all {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        // &Matrix + &Matrix
        impl<'a, 'b, T> $OpTrait<&'b Matrix<T>> for &'a Matrix<T>
        where T: Clone + $OpTrait<Output = T> {
            type Output = Matrix<T>;
            fn $method(self, rhs: &'b Matrix<T>) -> Matrix<T> {
                check_matrix_dims_for_ops(self, rhs);
                let data = self.data.iter().cloned().zip(rhs.data.iter().cloned()).map(|(a, b)| a $op b).collect();
                Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
        // Matrix + &Matrix (Consumes self)
        impl<'b, T> $OpTrait<&'b Matrix<T>> for Matrix<T>
        where T: Clone + $OpTrait<Output = T> {
            type Output = Matrix<T>;
            fn $method(mut self, rhs: &'b Matrix<T>) -> Matrix<T> { // Make self mutable for potential in-place modification
                check_matrix_dims_for_ops(&self, rhs);
                // Modify data in place
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter().cloned()) {
                    *a = a.clone() $op b; // Requires T: Clone for the *a = part
                }
                // Return modified self (its data vec was consumed conceptually)
                self
                // Alternative: Collect into new Vec if T is not Clone or in-place is complex
                // let data = self.data.into_iter().zip(rhs.data.iter().cloned()).map(|(a, b)| a $op b).collect();
                // Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
         // &Matrix + Matrix (Consumes rhs)
        impl<'a, T> $OpTrait<Matrix<T>> for &'a Matrix<T>
        where T: Clone + $OpTrait<Output = T> {
            type Output = Matrix<T>;
            fn $method(self, mut rhs: Matrix<T>) -> Matrix<T> { // Make rhs mutable
                check_matrix_dims_for_ops(self, &rhs);
                 // Modify rhs data in place
                 for (a, b) in self.data.iter().cloned().zip(rhs.data.iter_mut()) {
                    *b = a $op b.clone(); // Requires T: Clone for the *b = part
                }
                // Return modified rhs
                rhs
                // Alternative: Collect into new Vec
                // let data = self.data.iter().cloned().zip(rhs.data.into_iter()).map(|(a, b)| a $op b).collect();
                // Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
        // Matrix + Matrix (Consumes both)
        impl<T> $OpTrait<Matrix<T>> for Matrix<T>
        where T: Clone + $OpTrait<Output = T> {
            type Output = Matrix<T>;
            fn $method(mut self, rhs: Matrix<T>) -> Matrix<T> { // Make self mutable
                check_matrix_dims_for_ops(&self, &rhs);
                // Modify self data in place
                 for (a, b) in self.data.iter_mut().zip(rhs.data.into_iter()) {
                    *a = a.clone() $op b; // Requires T: Clone
                }
                // Return modified self
                self
                // Alternative: Collect into new Vec
                // let data = self.data.into_iter().zip(rhs.data.into_iter()).map(|(a, b)| a $op b).collect();
                // Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
    };
}

/// Generates element-wise arithmetic implementations for Matrix + Scalar
macro_rules! impl_elementwise_op_scalar_all {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        // &Matrix + Scalar
        impl<'a, T> $OpTrait<T> for &'a Matrix<T>
        where T: Clone + $OpTrait<Output = T> {
            type Output = Matrix<T>;
            fn $method(self, rhs: T) -> Matrix<T> {
                let data = self.data.iter().cloned().map(|a| a $op rhs.clone()).collect();
                Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
        // Matrix + Scalar (Consumes self)
        impl<T> $OpTrait<T> for Matrix<T>
        where T: Clone + $OpTrait<Output = T> {
            type Output = Matrix<T>;
            fn $method(mut self, rhs: T) -> Matrix<T> { // Make self mutable
                 // Modify self data in place
                for a in self.data.iter_mut() {
                     *a = a.clone() $op rhs.clone(); // Requires T: Clone
                }
                // Return modified self
                self
                // Alternative: Collect into new Vec
                // let data = self.data.into_iter().map(|a| a $op rhs.clone()).collect();
                // Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
        // NOTE: Scalar + Matrix (e.g., 1.0 + matrix) is NOT implemented here.
        // It would require `impl Add<Matrix<T>> for T`, which is discouraged
        // for primitive types unless inside the crate defining T.
    };
}

// Instantiate ALL combinations for arithmetic ops using the new macros
impl_elementwise_op_matrix_all!(Add, add, +);
impl_elementwise_op_matrix_all!(Sub, sub, -);
impl_elementwise_op_matrix_all!(Mul, mul, *); // Element-wise multiplication
impl_elementwise_op_matrix_all!(Div, div, /); // Element-wise division

impl_elementwise_op_scalar_all!(Add, add, +);
impl_elementwise_op_scalar_all!(Sub, sub, -);
impl_elementwise_op_scalar_all!(Mul, mul, *);
impl_elementwise_op_scalar_all!(Div, div, /);

// --- Element-wise Bitwise Operations (BoolMatrix) ---

macro_rules! impl_bitwise_op_all {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        // &Matrix<bool> OP &Matrix<bool>
        impl<'a, 'b> $OpTrait<&'b Matrix<bool>> for &'a Matrix<bool> {
            type Output = Matrix<bool>;
            fn $method(self, rhs: &'b Matrix<bool>) -> Matrix<bool> {
                check_matrix_dims_for_ops(self, rhs);
                let data = self.data.iter().cloned().zip(rhs.data.iter().cloned()).map(|(a, b)| a $op b).collect();
                Matrix { rows: self.rows, cols: self.cols, data }
            }
        }
        // Matrix<bool> OP &Matrix<bool>
        impl<'b> $OpTrait<&'b Matrix<bool>> for Matrix<bool> {
            type Output = Matrix<bool>;
            fn $method(mut self, rhs: &'b Matrix<bool>) -> Matrix<bool> {
                check_matrix_dims_for_ops(&self, rhs);
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) { *a = *a $op *b; } // bool is Copy
                self
            }
        }
        // &Matrix<bool> OP Matrix<bool>
        impl<'a> $OpTrait<Matrix<bool>> for &'a Matrix<bool> {
            type Output = Matrix<bool>;
            fn $method(self, mut rhs: Matrix<bool>) -> Matrix<bool> {
                check_matrix_dims_for_ops(self, &rhs);
                for (a, b) in self.data.iter().zip(rhs.data.iter_mut()) { *b = *a $op *b; } // bool is Copy
                rhs
            }
        }
        // Matrix<bool> OP Matrix<bool>
        impl $OpTrait<Matrix<bool>> for Matrix<bool> {
            type Output = Matrix<bool>;
            fn $method(mut self, rhs: Matrix<bool>) -> Matrix<bool> {
                check_matrix_dims_for_ops(&self, &rhs);
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) { *a = *a $op *b; } // bool is Copy
                self
            }
        }
    };
}

// Instantiate ALL combinations for bitwise ops
impl_bitwise_op_all!(BitAnd, bitand, &);
impl_bitwise_op_all!(BitOr, bitor, |);
impl_bitwise_op_all!(BitXor, bitxor, ^);

// --- Logical Not ---

// `!Matrix<bool>` (consumes matrix)
impl Not for Matrix<bool> {
    type Output = Matrix<bool>;
    fn not(mut self) -> Matrix<bool> {
        // Take by value, make mutable
        for val in self.data.iter_mut() {
            *val = !*val; // Invert in place
        }
        self // Return the modified matrix
    }
}

// `!&Matrix<bool>` (borrows matrix, returns new matrix)
impl Not for &Matrix<bool> {
    type Output = Matrix<bool>;
    fn not(self) -> Matrix<bool> {
        // Take by reference
        let data = self.data.iter().map(|&v| !v).collect(); // Create new data vec
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// --- Type Aliases ---
pub type FloatMatrix = Matrix<f64>;
pub type BoolMatrix = Matrix<bool>;
pub type IntMatrix = Matrix<i32>;
pub type StringMatrix = Matrix<String>;

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use crate::matrix::BoolOps;

    use super::*; // Import items from outer scope

    // Helper to create a 2x2 f64 matrix easily
    fn make_f64_matrix(a: f64, b: f64, c: f64, d: f64) -> FloatMatrix {
        Matrix::from_cols(vec![vec![a, c], vec![b, d]])
    }

    // Helper to create a 2x2 bool matrix easily
    fn make_bool_matrix(a: bool, b: bool, c: bool, d: bool) -> BoolMatrix {
        Matrix::from_cols(vec![vec![a, c], vec![b, d]])
    }

    // --- Arithmetic Tests ---

    #[test]
    fn test_add_f64() {
        let m1 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let m2 = make_f64_matrix(5.0, 6.0, 7.0, 8.0);
        let expected = make_f64_matrix(6.0, 8.0, 10.0, 12.0);

        assert_eq!(m1.clone() + m2.clone(), expected, "M + M");
        assert_eq!(m1.clone() + &m2, expected, "M + &M");
        assert_eq!(&m1 + m2.clone(), expected, "&M + M");
        assert_eq!(&m1 + &m2, expected, "&M + &M");
    }

    #[test]
    fn test_add_scalar_f64() {
        let m1 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let scalar = 10.0;
        let expected = make_f64_matrix(11.0, 12.0, 13.0, 14.0);

        assert_eq!(m1.clone() + scalar, expected, "M + S");
        assert_eq!(&m1 + scalar, expected, "&M + S");
    }

    #[test]
    fn test_sub_f64() {
        let m1 = make_f64_matrix(10.0, 20.0, 30.0, 40.0);
        let m2 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let expected = make_f64_matrix(9.0, 18.0, 27.0, 36.0);

        assert_eq!(m1.clone() - m2.clone(), expected, "M - M");
        assert_eq!(m1.clone() - &m2, expected, "M - &M");
        assert_eq!(&m1 - m2.clone(), expected, "&M - M");
        assert_eq!(&m1 - &m2, expected, "&M - &M");
    }

    #[test]
    fn test_sub_scalar_f64() {
        let m1 = make_f64_matrix(11.0, 12.0, 13.0, 14.0);
        let scalar = 10.0;
        let expected = make_f64_matrix(1.0, 2.0, 3.0, 4.0);

        assert_eq!(m1.clone() - scalar, expected, "M - S");
        assert_eq!(&m1 - scalar, expected, "&M - S");
    }

    #[test]
    fn test_mul_f64() {
        // Element-wise
        let m1 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let m2 = make_f64_matrix(5.0, 6.0, 7.0, 8.0);
        let expected = make_f64_matrix(5.0, 12.0, 21.0, 32.0);

        assert_eq!(m1.clone() * m2.clone(), expected, "M * M");
        assert_eq!(m1.clone() * &m2, expected, "M * &M");
        assert_eq!(&m1 * m2.clone(), expected, "&M * M");
        assert_eq!(&m1 * &m2, expected, "&M * &M");
    }

    #[test]
    fn test_mul_scalar_f64() {
        let m1 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let scalar = 3.0;
        let expected = make_f64_matrix(3.0, 6.0, 9.0, 12.0);

        assert_eq!(m1.clone() * scalar, expected, "M * S");
        assert_eq!(&m1 * scalar, expected, "&M * S");
    }

    #[test]
    fn test_div_f64() {
        // Element-wise
        let m1 = make_f64_matrix(10.0, 20.0, 30.0, 40.0);
        let m2 = make_f64_matrix(2.0, 5.0, 6.0, 8.0);
        let expected = make_f64_matrix(5.0, 4.0, 5.0, 5.0);

        assert_eq!(m1.clone() / m2.clone(), expected, "M / M");
        assert_eq!(m1.clone() / &m2, expected, "M / &M");
        assert_eq!(&m1 / m2.clone(), expected, "&M / M");
        assert_eq!(&m1 / &m2, expected, "&M / &M");
    }

    #[test]
    fn test_div_scalar_f64() {
        let m1 = make_f64_matrix(10.0, 20.0, 30.0, 40.0);
        let scalar = 10.0;
        let expected = make_f64_matrix(1.0, 2.0, 3.0, 4.0);

        assert_eq!(m1.clone() / scalar, expected, "M / S");
        assert_eq!(&m1 / scalar, expected, "&M / S");
    }

    #[test]
    fn test_chained_ops_f64() {
        let m = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let result = (((m.clone() + 1.0) * 2.0) - 4.0) / 2.0;
        // Expected:
        // m+1 = [2, 3], [4, 5]
        // *2  = [4, 6], [8, 10]
        // -4  = [0, 2], [4, 6]
        // /2  = [0, 1], [2, 3]
        let expected = make_f64_matrix(0.0, 1.0, 2.0, 3.0);
        assert_eq!(result, expected);
    }

    // --- Boolean Logic Tests ---

    #[test]
    fn test_bitand_bool() {
        let m1 = make_bool_matrix(true, false, true, false);
        let m2 = make_bool_matrix(true, true, false, false);
        let expected = make_bool_matrix(true, false, false, false);

        assert_eq!(m1.clone() & m2.clone(), expected, "M & M");
        assert_eq!(m1.clone() & &m2, expected, "M & &M");
        assert_eq!(&m1 & m2.clone(), expected, "&M & M");
        assert_eq!(&m1 & &m2, expected, "&M & &M");
    }

    #[test]
    fn test_bitor_bool() {
        let m1 = make_bool_matrix(true, false, true, false);
        let m2 = make_bool_matrix(true, true, false, false);
        let expected = make_bool_matrix(true, true, true, false);

        assert_eq!(m1.clone() | m2.clone(), expected, "M | M");
        assert_eq!(m1.clone() | &m2, expected, "M | &M");
        assert_eq!(&m1 | m2.clone(), expected, "&M | M");
        assert_eq!(&m1 | &m2, expected, "&M | &M");
    }

    #[test]
    fn test_bitxor_bool() {
        let m1 = make_bool_matrix(true, false, true, false);
        let m2 = make_bool_matrix(true, true, false, false);
        let expected = make_bool_matrix(false, true, true, false);

        assert_eq!(m1.clone() ^ m2.clone(), expected, "M ^ M");
        assert_eq!(m1.clone() ^ &m2, expected, "M ^ &M");
        assert_eq!(&m1 ^ m2.clone(), expected, "&M ^ M");
        assert_eq!(&m1 ^ &m2, expected, "&M ^ &M");
    }

    #[test]
    fn test_not_bool() {
        let m = make_bool_matrix(true, false, true, false);
        let expected = make_bool_matrix(false, true, false, true);

        assert_eq!(!m.clone(), expected, "!M (consuming)");
        assert_eq!(!&m, expected, "!&M (borrowing)");

        // Check original is unchanged when using !&M
        let original = make_bool_matrix(true, false, true, false);
        let _negated_ref = !&original;
        assert_eq!(original, make_bool_matrix(true, false, true, false));
    }

    // --- Comparison Tests ---
    #[test]
    fn test_comparison_eq_elem() {
        let m1 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let m2 = make_f64_matrix(1.0, 0.0, 3.0, 5.0);
        let s = 3.0;
        let expected_m = make_bool_matrix(true, false, true, false);
        let expected_s = make_bool_matrix(false, false, true, false);

        assert_eq!(m1.eq_elem(m2), expected_m, "eq_elem matrix");
        assert_eq!(m1.eq_elem(s), expected_s, "eq_elem scalar");
    }

    #[test]
    fn test_comparison_gt_elem() {
        let m1 = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let m2 = make_f64_matrix(0.0, 3.0, 3.0, 5.0);
        let s = 2.5;
        let expected_m = make_bool_matrix(true, false, false, false);
        let expected_s = make_bool_matrix(false, false, true, true);

        assert_eq!(m1.gt_elem(m2), expected_m, "gt_elem matrix");
        assert_eq!(m1.gt_elem(s), expected_s, "gt_elem scalar");
    }

    // Add more comparison tests (lt, le, ge, ne) if desired...

    // --- Basic Method Tests ---
    #[test]
    fn test_indexing() {
        let m = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
        assert_eq!(*m.get(1, 0), 3.0); // Test get() too
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_row() {
        let m = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let _ = m[(2, 0)]; // Row 2 is out of bounds
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_col() {
        let m = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        let _ = m[(0, 2)]; // Col 2 is out of bounds
    }

    #[test]
    fn test_dimensions() {
        let m = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 3.0, 2.0, 4.0]; // Column major: [col0_row0, col0_row1, col1_row0, col1_row1]
        let m = Matrix::from_vec(data, 2, 2);
        let expected = make_f64_matrix(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m, expected);
        assert_eq!(m.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    // Helper function to create a basic Matrix for testing
    fn static_test_matrix() -> Matrix<i32> {
        // Column-major data:
        // 1 4 7
        // 2 5 8
        // 3 6 9
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        Matrix::from_vec(data, 3, 3)
    }

    // Another helper for a different size
    fn static_test_matrix_2x4() -> Matrix<i32> {
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
    fn test_transpose() {
        let matrix = static_test_matrix();
        let transposed = matrix.transpose();
        let round_triped = transposed.transpose();
        assert_eq!(
            round_triped, matrix,
            "Transposing twice should return original matrix"
        );
        for r in 0..matrix.rows() {
            for c in 0..matrix.cols() {
                assert_eq!(matrix[(r, c)], transposed[(c, r)]);
            }
        }
    }

    #[test]
    fn test_transpose_big() {
        let data: Vec<i32> = (1..=20000).collect(); //
        let matrix = Matrix::from_vec(data, 100, 200);
        let transposed = matrix.transpose();
        assert_eq!(transposed.rows(), 200);
        assert_eq!(transposed.cols(), 100);
        assert_eq!(transposed.data().len(), 20000);
        assert_eq!(transposed[(0, 0)], 1);

        let round_trip = transposed.transpose();

        assert_eq!(
            round_trip, matrix,
            "Transposing back should return original matrix"
        );
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
        let matrix = static_test_matrix();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_index_and_get() {
        let matrix = static_test_matrix();
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(2, 2)], 9);

        assert_eq!(*matrix.get(0, 0), 1);
        assert_eq!(*matrix.get(1, 1), 5);
        assert_eq!(*matrix.get(2, 2), 9);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_row_alt() {
        let matrix = static_test_matrix();
        let _ = matrix[(3, 0)];
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_col_alt() {
        let matrix = static_test_matrix();
        let _ = matrix[(0, 3)];
    }

    #[test]
    fn test_index_mut_and_get_mut() {
        let mut matrix = static_test_matrix();

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

        assert_eq!(matrix.data(), &[10, 2, 3, 15, 20, 25, 7, 8, 30]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds_row() {
        let mut matrix = static_test_matrix();
        matrix[(3, 0)] = 99;
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds_col() {
        let mut matrix = static_test_matrix();
        matrix[(0, 3)] = 99;
    }

    #[test]
    fn test_row() {
        let ma = static_test_matrix();
        assert_eq!(ma.row(0), &[1, 4, 7]);
        assert_eq!(ma.row(1), &[2, 5, 8]);
        assert_eq!(ma.row(2), &[3, 6, 9]);
    }

    #[test]
    fn test_row_copy_from_slice() {
        let mut ma = static_test_matrix();
        let new_row = vec![10, 20, 30];
        ma.row_copy_from_slice(1, &new_row);
        assert_eq!(ma.row(1), &[10, 20, 30]);
    }

    #[test]
    fn test_shape() {
        let ma = static_test_matrix_2x4();
        assert_eq!(ma.shape(), (2, 4));
        assert_eq!(ma.rows(), 2);
        assert_eq!(ma.cols(), 4);
    }

    #[test]
    fn test_repeat_rows() {
        let ma = static_test_matrix();
        // Returns a new matrix where row 0 of `self` is repeated `n` times.
        let repeated = ma.repeat_rows(3);
        // assert all rows are equal to the first row
        for r in 0..repeated.rows() {
            assert_eq!(repeated.row(r), ma.row(0));
        }
    }

    #[test]
    #[should_panic(expected = "row index 3 out of bounds for 3 rows")]
    fn test_row_out_of_bounds() {
        let ma = static_test_matrix();
        ma.row(3);
    }

    #[test]
    fn test_column() {
        let matrix = static_test_matrix_2x4();

        assert_eq!(matrix.column(0), &[1, 2]);
        assert_eq!(matrix.column(1), &[3, 4]);
        assert_eq!(matrix.column(2), &[5, 6]);
        assert_eq!(matrix.column(3), &[7, 8]);
    }

    #[test]
    #[should_panic(expected = "column index 4 out of bounds for 4 columns")]
    fn test_column_out_of_bounds() {
        let matrix = static_test_matrix_2x4();
        matrix.column(4);
    }

    #[test]
    fn test_column_mut() {
        let mut matrix = static_test_matrix_2x4();

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

        assert_eq!(matrix.data(), &[1, 2, 30, 40, 5, 6, 70, 8]);
    }

    #[test]
    #[should_panic(expected = "column index 4 out of bounds for 4 columns")]
    fn test_column_mut_out_of_bounds() {
        let mut matrix = static_test_matrix_2x4();
        matrix.column_mut(4);
    }

    #[test]
    fn test_iter_columns() {
        let matrix = static_test_matrix_2x4();

        let cols: Vec<&[i32]> = matrix.iter_columns().collect();
        assert_eq!(cols.len(), 4);
        assert_eq!(cols[0], &[1, 2]);
        assert_eq!(cols[1], &[3, 4]);
        assert_eq!(cols[2], &[5, 6]);
        assert_eq!(cols[3], &[7, 8]);
    }

    #[test]
    fn test_iter_rows() {
        let matrix = static_test_matrix_2x4();

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
        let mut matrix = static_test_matrix();

        let data_mut = matrix.data_mut();
        data_mut[0] = 10;
        data_mut[1] = 20;

        assert_eq!(matrix[(0, 0)], 10);
        assert_eq!(matrix[(1, 0)], 20);
    }

    #[test]
    fn test_matrix_row_get_and_iter() {
        let matrix = static_test_matrix_2x4();
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
        let mut matrix = static_test_matrix();

        matrix.swap_columns(0, 2); // swap first and last

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

        // swap the same column (should do nothing)
        let original_data = matrix.data().to_vec();
        matrix.swap_columns(1, 1);
        assert_eq!(matrix.data(), &original_data); // Data should be identical

        // Check underlying data (column-major) after swap(0, 2)
        assert_eq!(matrix.data(), &[7, 8, 9, 4, 5, 6, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "column index c2=3 out of bounds for 3 columns")]
    fn test_swap_columns_out_of_bounds() {
        let mut matrix = static_test_matrix();
        matrix.swap_columns(0, 3);
    }

    #[test]
    fn test_delete_column() {
        let mut matrix = static_test_matrix_2x4();
        matrix.delete_column(1); // Delete the second column (index 1)

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 0)], 2);
        assert_eq!(matrix[(0, 1)], 5);
        assert_eq!(matrix[(1, 1)], 6);
        assert_eq!(matrix[(0, 2)], 7);
        assert_eq!(matrix[(1, 2)], 8);

        // check underlying data
        assert_eq!(matrix.data(), &[1, 2, 5, 6, 7, 8]);

        // Delete the first column
        matrix.delete_column(0);
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.data(), &[5, 6, 7, 8]);

        // Delete the last column
        matrix.delete_column(1);
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
    #[should_panic(expected = "column index 4 out of bounds for 4 columns")]
    fn test_delete_column_out_of_bounds() {
        let mut matrix = static_test_matrix_2x4();
        matrix.delete_column(4);
    }

    #[test]
    fn test_delete_row() {
        let mut matrix = static_test_matrix();

        matrix.delete_row(1); // Delete the second row

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(0, 1)], 4);
        assert_eq!(matrix[(1, 1)], 6);
        assert_eq!(matrix[(0, 2)], 7);
        assert_eq!(matrix[(1, 2)], 9);

        // check underlying data (column-major)
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
    #[should_panic(expected = "row index 3 out of bounds for 3 rows")]
    fn test_delete_row_out_of_bounds() {
        let mut matrix = static_test_matrix();
        matrix.delete_row(3);
    }

    #[test]
    fn test_add_column() {
        let mut matrix = static_test_matrix_2x4();
        let new_col = vec![9, 10];
        matrix.add_column(2, new_col);

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

        // Check underlying data
        assert_eq!(matrix.data(), &[1, 2, 3, 4, 9, 10, 5, 6, 7, 8]);

        // Add a column at the beginning
        let new_col_start = vec![11, 12];
        matrix.add_column(0, new_col_start);
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 6);
        assert_eq!(matrix[(0, 0)], 11);
        assert_eq!(matrix[(1, 0)], 12);
        assert_eq!(matrix.data(), &[11, 12, 1, 2, 3, 4, 9, 10, 5, 6, 7, 8]);

        // Add a column at the end
        let new_col_end = vec![13, 14];
        matrix.add_column(6, new_col_end);
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
    #[should_panic(expected = "add_column index 5 out of bounds for 4 columns")]
    fn test_add_column_out_of_bounds() {
        let mut matrix = static_test_matrix_2x4();
        let new_col = vec![9, 10];
        matrix.add_column(5, new_col); // Index 5 is out of bounds for 4 columns
    }

    #[test]
    #[should_panic(expected = "column length mismatch")]
    fn test_add_column_length_mismatch() {
        let mut matrix = static_test_matrix_2x4();
        let new_col = vec![9, 10, 11]; // Wrong length
        matrix.add_column(0, new_col);
    }

    #[test]
    fn test_add_row() {
        let mut matrix = static_test_matrix_2x4();
        let new_row = vec![9, 10, 11, 12];
        matrix.add_row(1, new_row);

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
        assert_eq!(matrix.data(), &[1, 9, 2, 3, 10, 4, 5, 11, 6, 7, 12, 8]);

        // Add a row at the beginning
        let new_row_start = vec![13, 14, 15, 16];
        matrix.add_row(0, new_row_start);
        assert_eq!(matrix.rows(), 4);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix[(0, 0)], 13);
        assert_eq!(matrix[(0, 1)], 14);
        assert_eq!(matrix[(0, 2)], 15);
        assert_eq!(matrix[(0, 3)], 16);
        assert_eq!(matrix[(1, 0)], 1);
        assert_eq!(matrix[(2, 1)], 10);
        assert_eq!(matrix[(3, 3)], 8);

        // Add a row at the end
        let new_row_end = vec![17, 18, 19, 20];
        matrix.add_row(4, new_row_end);
        assert_eq!(matrix.rows(), 5);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix[(4, 0)], 17);
        assert_eq!(matrix[(4, 3)], 20);
    }

    #[test]
    #[should_panic(expected = "add_row index 3 out of bounds for 2 rows")]
    fn test_add_row_out_of_bounds() {
        let mut matrix = static_test_matrix_2x4();
        let new_row = vec![9, 10, 11, 12];
        matrix.add_row(3, new_row); // Index 3 is out of bounds for 2 rows
    }

    #[test]
    #[should_panic(expected = "row length mismatch")]
    fn test_add_row_length_mismatch() {
        let mut matrix = static_test_matrix_2x4();
        let new_row = vec![9, 10, 11]; // Wrong length
        matrix.add_row(0, new_row);
    }

    #[test]
    fn test_elementwise_add() {
        let matrix1 = static_test_matrix();
        let matrix2 = Matrix::from_vec(vec![9, 8, 7, 6, 5, 4, 3, 2, 1], 3, 3);

        let result = &matrix1 + &matrix2;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Expect all 10s
        assert_eq!(result.data(), &[10, 10, 10, 10, 10, 10, 10, 10, 10]);
        assert_eq!(result[(0, 0)], 10);
        assert_eq!(result[(1, 1)], 10);
        assert_eq!(result[(2, 2)], 10);
    }

    #[test]
    fn test_elementwise_sub() {
        let matrix1 = static_test_matrix();
        let matrix2 = Matrix::from_vec(vec![1, 1, 1, 2, 2, 2, 3, 3, 3], 3, 3);

        let result = &matrix1 - &matrix2;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        assert_eq!(result.data(), &[0, 1, 2, 2, 3, 4, 4, 5, 6]);
        assert_eq!(result[(0, 0)], 0);
        assert_eq!(result[(1, 1)], 3);
        assert_eq!(result[(2, 2)], 6);
    }

    #[test]
    fn test_elementwise_mul() {
        let matrix1 = static_test_matrix();
        let matrix2 = Matrix::from_vec(vec![1, 2, 3, 1, 2, 3, 1, 2, 3], 3, 3);

        let result = &matrix1 * &matrix2;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Expected
        assert_eq!(result.data(), &[1, 4, 9, 4, 10, 18, 7, 16, 27]);
        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(1, 1)], 10);
        assert_eq!(result[(2, 2)], 27);
    }

    #[test]
    fn test_elementwise_div() {
        let matrix1 = static_test_matrix();
        let matrix2 = Matrix::from_vec(vec![1, 1, 1, 2, 2, 2, 7, 8, 9], 3, 3);

        let result = &matrix1 / &matrix2; // Integer division

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        assert_eq!(result.data(), &[1, 2, 3, 2, 2, 3, 1, 1, 1]);
        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(1, 1)], 2);
        assert_eq!(result[(2, 2)], 1);
    }

    #[test]
    #[should_panic(expected = "Row count mismatch: left has 3 rows, right has 2 rows")]
    fn test_elementwise_op_row_mismatch() {
        let matrix1 = static_test_matrix();
        let matrix2 = static_test_matrix_2x4();
        let _ = &matrix1 + &matrix2; // Should panic
    }

    #[test]
    #[should_panic(expected = "Row count mismatch: left has 3 rows, right has 2 ro")]
    fn test_elementwise_op_col_mismatch() {
        let matrix1 = static_test_matrix();
        let matrix2 = static_test_matrix_2x4();
        let _ = &matrix1 * &matrix2; // Should panic
    }

    #[test]
    fn test_bitwise_and() {
        let data1 = vec![true, false, true, false, true, false];
        let data2 = vec![true, true, false, false, true, true];
        let matrix1 = BoolMatrix::from_vec(data1, 2, 3);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);

        let expected_data = vec![true, false, false, false, true, false];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = &matrix1 & &matrix2;
        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn test_bitwise_or() {
        let data1 = vec![true, false, true, false, true, false];
        let data2 = vec![true, true, false, false, true, true];
        let matrix1 = BoolMatrix::from_vec(data1, 2, 3);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);

        let expected_data = vec![true, true, true, false, true, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = &matrix1 | &matrix2;
        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn test_bitwise_xor() {
        let data1 = vec![true, false, true, false, true, false];
        let data2 = vec![true, true, false, false, true, true];
        let matrix1 = BoolMatrix::from_vec(data1, 2, 3);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);

        let expected_data = vec![false, true, true, false, false, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = &matrix1 ^ &matrix2;
        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn test_bitwise_not() {
        let data = vec![true, false, true, false, true, false];
        let matrix = BoolMatrix::from_vec(data, 2, 3);

        let expected_data = vec![false, true, false, true, false, true];
        let expected_matrix = BoolMatrix::from_vec(expected_data, 2, 3);

        let result = !matrix; // Not consumes the matrix
        assert_eq!(result, expected_matrix);
    }

    #[test]
    #[should_panic(expected = "Column count mismatch: left has 2 columns, right has 3 columns")]
    fn test_bitwise_op_row_mismatch() {
        let data1 = vec![true, false, true, false];
        let data2 = vec![true, true, false, false, true, true];
        let matrix1 = BoolMatrix::from_vec(data1, 2, 2);
        let matrix2 = BoolMatrix::from_vec(data2, 2, 3);
        let _ = &matrix1 & &matrix2; // Should panic
    }

    #[test]
    #[should_panic(expected = "Column count mismatch: left has 2 columns, right has 3 columns")]
    fn test_bitwise_op_col_mismatch() {
        let data1 = vec![true, false, true, false];
        let data2 = vec![true, true, false, false, true, true];
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
        let matrix = StringMatrix::from_vec(data.clone(), 2, 2);

        assert_eq!(matrix[(0, 0)], "a".to_string());
        assert_eq!(matrix[(1, 0)], "b".to_string());
        assert_eq!(matrix[(0, 1)], "c".to_string());
        assert_eq!(matrix[(1, 1)], "d".to_string());

        // Test modification
        let mut matrix = matrix;
        matrix[(0, 0)] = "hello".to_string();
        assert_eq!(matrix[(0, 0)], "hello".to_string());

        // Test add_column
        let new_col = vec!["e".to_string(), "f".to_string()];
        matrix.add_column(1, new_col);

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
        matrix.add_row(0, new_row);

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
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![0.5, 1.5, 2.5, 3.5];
        let matrix1 = FloatMatrix::from_vec(data1, 2, 2);
        let matrix2 = FloatMatrix::from_vec(data2, 2, 2);

        let sum = &matrix1 + &matrix2;
        let diff = &matrix1 - &matrix2;
        let prod = &matrix1 * &matrix2;
        let div = &matrix1 / &matrix2;

        assert_eq!(sum.data(), &[1.5, 3.5, 5.5, 7.5]);

        assert_eq!(diff.data(), &[0.5, 0.5, 0.5, 0.5]);

        assert_eq!(prod.data(), &[0.5, 3.0, 7.5, 14.0]);

        // Check divs (col-major): [2.0, 1.333..., 1.2, 1.14...]
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

        // Verify the dimensions are unchanged
        assert_eq!(matrix.rows(), 3, "Matrix rows should remain unchanged");
        assert_eq!(matrix.cols(), 3, "Matrix cols should remain unchanged");

        // Verify the column that was NOT swapped is unchanged
        assert_eq!(
            matrix.column(1),
            initial_col1_data.as_slice(), // Comparing slice to slice
            "Column 1 data should be unchanged"
        );

        // Verify the data swap occurred correctly using the COLUMN ACCESSOR
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

        // Verify the underlying raw data vector
        assert_eq!(
            matrix.data(),
            &[7, 8, 9, 4, 5, 6, 1, 2, 3],
            "Underlying data vector is incorrect after swap"
        );

        // Test swapping with self (should be a no-op)
        let state_before_self_swap = matrix.clone();
        matrix.swap_columns(1, 1);
        assert_eq!(
            matrix, state_before_self_swap,
            "Swapping a column with itself should not change the matrix"
        );

        // Test swapping adjacent columns
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
        let matrix = static_test_matrix();
        // test all > 0
        let result = matrix.gt_elem(0).into_vec();
        let expected = vec![true; result.len()];
        assert_eq!(result, expected);

        let ma = static_test_matrix();
        let mb = static_test_matrix();

        let result = ma.eq_elem(mb);
        assert!(result.all());

        let result = matrix.lt_elem(1e10 as i32).all();
        assert!(result);

        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                let vx = matrix[(i, j)];
                let c = &(matrix.le_elem(vx)) & &(matrix.ge_elem(vx));
                assert_eq!(c.count(), 1);
            }
        }
    }

    #[test]
    fn test_arithmetic_broadcast() {
        let matrix = static_test_matrix();
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

    #[test]
    fn test_matrix_zeros_ones_filled() {
        // Test zeros
        let m = Matrix::<f64>::zeros(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Test ones
        let m = Matrix::<f64>::ones(3, 2);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Test filled
        let m = Matrix::<f64>::filled(2, 2, 42.5);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &[42.5, 42.5, 42.5, 42.5]);

        // test with an integer matrix
        let m = Matrix::<i32>::filled(2, 3, 7);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data(), &[7, 7, 7, 7, 7, 7]);

        // test with nans
        let m = Matrix::nan(3, 3);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);
        for &value in m.data() {
            assert!(value.is_nan(), "Expected NaN, got {}", value);
        }
    }
}
