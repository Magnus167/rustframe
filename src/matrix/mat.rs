use std::ops::{Index, IndexMut, Not};

/// A column‑major 2D matrix of `T`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Build from columns (each inner Vec is one column)
    pub fn from_cols(cols_data: Vec<Vec<T>>) -> Self {
        let cols = cols_data.len();
        assert!(cols > 0, "need at least one column");
        let rows = cols_data[0].len();
        assert!(rows > 0, "need at least one row");
        for (i, col) in cols_data.iter().enumerate().skip(1) {
            assert!(
                col.len() == rows,
                "col {} has len {}, expected {}",
                i,
                col.len(),
                rows
            );
        }
        let mut data = Vec::with_capacity(rows * cols);
        for col in cols_data {
            data.extend(col);
        }
        Matrix { rows, cols, data }
    }

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "need at least one row");
        assert!(cols > 0, "need at least one column");
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Matrix { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn data(&self) -> &[T] {
        &self.data
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
        let start = c * self.rows;
        &self.data[start..start + self.rows]
    }
    #[inline]
    pub fn column_mut(&mut self, c: usize) -> &mut [T] {
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
            c1 < self.cols && c2 < self.cols,
            "column index out of bounds"
        );
        if c1 == c2 {
            return; // No-op if indices are the same
        }

        // Loop through each row
        for r in 0..self.rows {
            // Calculate the flat index for the element in row r, column c1
            let idx1 = c1 * self.rows + r;
            // Calculate the flat index for the element in row r, column c2
            let idx2 = c2 * self.rows + r;

            // Swap the elements directly in the data vector
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
        assert!(index <= self.cols, "column index out of bounds");
        assert_eq!(column.len(), self.rows, "column length mismatch");

        for (r, value) in column.into_iter().enumerate() {
            self.data.insert(index * self.rows + r, value);
        }
        self.cols += 1;
    }

    /// Adds a row to the matrix at the specified index.
    pub fn add_row(&mut self, index: usize, row: Vec<T>) {
        assert!(index <= self.rows, "row index out of bounds");
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
        assert!(r < self.rows && c < self.cols, "index out of bounds");
        &self.data[c * self.rows + r]
    }
}
impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        assert!(r < self.rows && c < self.cols, "index out of bounds");
        &mut self.data[c * self.rows + r]
    }
}

/// A view of one row
pub struct MatrixRow<'a, T> {
    matrix: &'a Matrix<T>,
    row: usize,
}
impl<'a, T> MatrixRow<'a, T> {
    pub fn get(&self, c: usize) -> &T {
        &self.matrix[(self.row, c)]
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.matrix.cols).map(move |c| &self.matrix[(self.row, c)])
    }
}

/// Macro to generate element‐wise impls for +, -, *, /
macro_rules! impl_elementwise_op {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, 'b, T> std::ops::$OpTrait<&'b Matrix<T>> for &'a Matrix<T>
        where
            T: Clone + std::ops::$OpTrait<Output = T>,
        {
            type Output = Matrix<T>;

            fn $method(self, rhs: &'b Matrix<T>) -> Matrix<T> {
                assert_eq!(self.rows, rhs.rows, "row count mismatch");
                assert_eq!(self.cols, rhs.cols, "col count mismatch");
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

// invoke it 4 times:
impl_elementwise_op!(Add, add, +);
impl_elementwise_op!(Sub, sub, -);
impl_elementwise_op!(Mul, mul, *);
impl_elementwise_op!(Div, div, /);

pub type FloatMatrix = Matrix<f64>;
pub type BoolMatrix = Matrix<bool>;
pub type IntMatrix = Matrix<i32>;
pub type StringMatrix = Matrix<String>;

// implement bit ops - and, or, xor, not -- using Macros

macro_rules! impl_bitwise_op {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, 'b> std::ops::$OpTrait<&'b Matrix<bool>> for &'a Matrix<bool> {
            type Output = Matrix<bool>;

            fn $method(self, rhs: &'b Matrix<bool>) -> Matrix<bool> {
                assert_eq!(self.rows, rhs.rows, "row count mismatch");
                assert_eq!(self.cols, rhs.cols, "col count mismatch");
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
impl_bitwise_op!(BitAnd, bitand, &);
impl_bitwise_op!(BitOr, bitor, |);
impl_bitwise_op!(BitXor, bitxor, ^);

impl Not for Matrix<bool> {
    type Output = Matrix<bool>;

    fn not(self) -> Matrix<bool> {
        let data = self.data.iter().map(|&v| !v).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

/// Axis along which to apply a reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Operate column‑wise (vertical).
    Col,
    /// Operate row‑wise (horizontal).
    Row,
}
