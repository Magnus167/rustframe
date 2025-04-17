use std::ops::{Index, IndexMut};

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

    pub fn iter_columns(&self) -> impl Iterator<Item = &[T]> {
        (0..self.cols).map(move |c| self.column(c))
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = Row<'_, T>> {
        (0..self.rows).map(move |r| Row {
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
            return;
        }

        for r in 0..self.rows {
            self.data.swap(c1 * self.rows + r, c2 * self.rows + r);
        }
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
pub struct Row<'a, T> {
    matrix: &'a Matrix<T>,
    row: usize,
}
impl<'a, T> Row<'a, T> {
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

// === New code begins here =====================================================
/// Axis along which to apply a reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Operate column‑wise (vertical).
    Col,
    /// Operate row‑wise (horizontal).
    Row,
}

pub type FloatMatrix = Matrix<f64>;
pub type FloatVector = Vec<f64>;
pub type BoolMatrix = Matrix<bool>;
pub type IntMatrix = Matrix<i32>;

impl Matrix<f64> {
    /// Apply a function along *columns* and collect its result in a `Vec`.
    /// This is very fast because each column is contiguous in memory.
    #[inline]
    fn apply_colwise<U, F>(&self, mut f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U,
    {
        let mut out = Vec::with_capacity(self.cols);
        for c in 0..self.cols {
            out.push(f(self.column(c)));
        }
        out
    }

    /// Apply a function along *rows* and collect its result in a `Vec`.
    /// Slower than the column version because data are not contiguous, but a single
    /// reusable buffer is used to minimize allocations.
    #[inline]
    fn apply_rowwise<U, F>(&self, mut f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U,
    {
        let mut out = Vec::with_capacity(self.rows);
        // Re‑use one buffer for all rows to avoid repeated allocations.
        let mut buf = vec![0.0f64; self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                buf[c] = self[(r, c)];
            }
            out.push(f(&buf));
        }
        out
    }

    /// Generic helper that dispatches to [`Matrix::apply_colwise`] or
    /// [`Matrix::apply_rowwise`] depending on `axis`.
    #[inline]
    pub fn apply_axis<U, F>(&self, axis: Axis, f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U,
    {
        match axis {
            Axis::Col => self.apply_colwise(f),
            Axis::Row => self.apply_rowwise(f),
        }
    }

    // ---------------------------------------------------------------------
    // Convenience reductions built on top of `apply_axis`.
    // By convention "vertical" = column‑wise, "horizontal" = row‑wise.
    // ---------------------------------------------------------------------

    /// Column‑wise sum, ignoring `NaN`s.
    pub fn sum_vertical(&self) -> FloatVector {
        self.apply_colwise(|col| col.iter().copied().filter(|v| !v.is_nan()).sum())
    }

    /// Row‑wise sum, ignoring `NaN`s.
    pub fn sum_horizontal(&self) -> FloatVector {
        self.apply_rowwise(|row| row.iter().copied().filter(|v| !v.is_nan()).sum())
    }

    /// Column‑wise product, ignoring `NaN`s.
    pub fn prod_vertical(&self) -> FloatVector {
        self.apply_colwise(|col| {
            col.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(1.0, |acc, x| acc * x)
        })
    }

    /// Row‑wise product, ignoring `NaN`s.
    pub fn prod_horizontal(&self) -> FloatVector {
        self.apply_rowwise(|row| {
            row.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(1.0, |acc, x| acc * x)
        })
    }

    /// Column‑wise count of `NaN`s.
    pub fn count_nan_vertical(&self) -> Vec<usize> {
        self.apply_colwise(|col| col.iter().filter(|x| x.is_nan()).count())
    }

    /// Row‑wise count of `NaN`s.
    pub fn count_nan_horizontal(&self) -> Vec<usize> {
        self.apply_rowwise(|row| row.iter().filter(|x| x.is_nan()).count())
    }

    // ---------------------------------------------------------------------
    // Existing helpers
    // ---------------------------------------------------------------------

    pub fn is_nan(&self) -> BoolMatrix {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                data.push(self[(r, c)].is_nan());
            }
        }
        BoolMatrix::from_vec(data, self.rows, self.cols)
    }


}
