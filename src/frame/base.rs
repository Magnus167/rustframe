use crate::matrix::*;
use std::collections::HashMap;
use std::ops::{Index, IndexMut, Not};


/// A data frame – a Matrix with string‑identified columns (column‑major).
///
/// Restricts the element type T to anything that is at least Clone –
/// this guarantees we can duplicate data when adding columns or performing
/// ownership‑moving transformations later on. (Further trait bounds are added
/// per‑method when additional capabilities such as arithmetic are needed.)
///
/// # Examples
///
/// ```
/// use rustframe::frame::Frame; // Assuming Frame is in the root of rustframe
/// use rustframe::matrix::Matrix; // Assuming Matrix is in rustframe::matrix
///
/// // 1. Create a frame
/// let matrix = Matrix::from_cols(vec![
///     vec![1.0, 2.0, 3.0], // Column "temp"
///     vec![5.5, 6.5, 7.5], // Column "pressure"
/// ]);
/// let mut frame = Frame::new(matrix, vec!["temp", "pressure"]);
///
/// assert_eq!(frame.column_names, vec!["temp", "pressure"]);
///
/// // 2. Access data
/// assert_eq!(frame.column("temp"), &[1.0, 2.0, 3.0]);
/// assert_eq!(frame["pressure"].to_vec(), &[5.5, 6.5, 7.5]);
/// assert_eq!(frame.column_index("temp"), Some(0));
///
/// // 3. Mutate data
/// frame["temp"][0] = 1.5;
/// assert_eq!(frame["temp"].to_vec(), &[1.5, 2.0, 3.0]);
///
/// frame.column_mut("pressure")[1] = 6.8;
/// assert_eq!(frame["pressure"].to_vec(), &[5.5, 6.8, 7.5]);
///
/// // 4. Add a column
/// frame.add_column("humidity", vec![50.0, 55.0, 60.0]);
/// assert_eq!(frame.column_names, vec!["temp", "pressure", "humidity"]);
/// assert_eq!(frame["humidity"].to_vec(), &[50.0, 55.0, 60.0]); // i32 mixed with f64 needs generic adjustment or separate examples
///
/// // 5. Rename a column
/// frame.rename("temp", "temperature");
/// assert_eq!(frame.column_names, vec!["temperature", "pressure", "humidity"]);
/// assert!(frame.column_index("temp").is_none());
/// assert_eq!(frame.column_index("temperature"), Some(0));
/// assert_eq!(frame["temperature"].to_vec(), &[1.5, 2.0, 3.0]);
///
/// // 6. Swap columns
/// frame.swap_columns("temperature", "humidity");
/// assert_eq!(frame.column_names, vec!["humidity", "pressure", "temperature"]);
/// assert_eq!(frame["humidity"].to_vec(), &[50.0, 55.0, 60.0]); // Now holds original temp data
///
/// // 7. Sort columns
/// frame.sort_columns();
/// assert_eq!(frame.column_names, vec!["humidity", "pressure", "temperature"]); // Already sorted after swap
/// // Let's add one more to see sorting:
/// // frame.add_column("altitude", vec![100.0, 110.0, 120.0]);
/// // frame.sort_columns();
/// // assert_eq!(frame.column_names, vec!["altitude", "humidity", "pressure", "temperature"]);
///
/// // 8. Delete a column
/// let deleted_pressure = frame.delete_column("pressure");
/// assert_eq!(deleted_pressure, vec![5.5, 6.8, 7.5]);
/// assert_eq!(frame.column_names, vec!["humidity", "temperature"]);
/// assert!(frame.column_index("pressure").is_none());
///
/// // 9. Element-wise operations (requires compatible frames)
/// let matrix_offset = Matrix::from_cols(vec![
///     vec![0.1, 0.1, 0.1], // humidity offset
///     vec![1.0, 1.0, 1.0], // temperature offset
/// ]);
/// let frame_offset = Frame::new(matrix_offset, vec!["humidity", "temperature"]);
///
/// let adjusted_frame = &frame + &frame_offset; // Add requires frame: Frame<f64>
/// // Need to ensure frame is Frame<f64> for this op
/// // assert_eq!(adjusted_frame["humidity"], &[1.6, 2.1, 3.1]); // Original temp + 0.1
/// // assert_eq!(adjusted_frame["temperature"], &[51.0, 56.0, 61.0]); // Original humidity + 1.0

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame<T: Clone> {
    /// **Public** vector holding the column names in their current order.
    pub column_names: Vec<String>,

    matrix: Matrix<T>,
    /// Maps a label to the column index for **O(1)** lookup.
    pub lookup: HashMap<String, usize>,
}

impl<T: Clone> Frame<T> {
    /* ---------- Constructors ---------- */
    /// Creates a new [`Frame`] from a matrix and column names.
    ///
    /// # Panics
    /// * if the number of names differs from `matrix.cols()`
    /// * if names are not unique.
    pub fn new<L: Into<String>>(matrix: Matrix<T>, names: Vec<L>) -> Self {
        assert_eq!(matrix.cols(), names.len(), "column name count mismatch");
        let mut lookup = HashMap::with_capacity(names.len());
        let column_names: Vec<String> = names
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                let s = n.into();
                if lookup.insert(s.clone(), i).is_some() {
                    panic!("duplicate column label: {}", s);
                }
                s
            })
            .collect();
        Self {
            matrix,
            column_names,
            lookup,
        }
    }

    /* ---------- Immutable / mutable access ---------- */

    #[inline]
    pub fn matrix(&self) -> &Matrix<T> {
        &self.matrix
    }
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut Matrix<T> {
        &mut self.matrix
    }

    /// Returns an immutable view of the column `name`.
    pub fn column(&self, name: &str) -> &[T] {
        let idx = self
            .lookup
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("unknown column label: {}", name));
        self.matrix.column(idx)
    }

    /// Returns a mutable view of the column `name`.
    pub fn column_mut(&mut self, name: &str) -> &mut [T] {
        let idx = self
            .lookup
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("unknown column label: {}", name));
        // SAFETY: the column is stored contiguously (column‑major layout).
        self.matrix.column_mut(idx)
    }

    /// Index of a column label, if it exists.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.lookup.get(name).copied()
    }

    /* ---------- Column manipulation ---------- */

    /// Swaps two columns identified by their labels.
    /// Internally defers to the already‑implemented [`Matrix::swap_columns`].
    pub fn swap_columns<L: AsRef<str>>(&mut self, a: L, b: L) {
        let ia = self
            .column_index(a.as_ref())
            .unwrap_or_else(|| panic!("unknown column label: {}", a.as_ref()));
        let ib = self
            .column_index(b.as_ref())
            .unwrap_or_else(|| panic!("unknown column label: {}", b.as_ref()));
        if ia == ib {
            return; // nothing to do
        }
        self.matrix.swap_columns(ia, ib); // <‑‑ reuse existing impl
        self.column_names.swap(ia, ib);
        // update lookup values
        self.lookup.get_mut(a.as_ref()).map(|v| *v = ib);
        self.lookup.get_mut(b.as_ref()).map(|v| *v = ia);
    }

    /// Renames a column.
    ///
    /// # Panics
    /// * if `old` is missing
    /// * if `new` already exists.
    pub fn rename<L: Into<String>>(&mut self, old: &str, new: L) {
        let idx = self
            .column_index(old)
            .unwrap_or_else(|| panic!("unknown column label: {}", old));
        let new = new.into();
        if self.lookup.contains_key(&new) {
            panic!("duplicate column label: {}", new);
        }
        self.column_names[idx] = new.clone();
        self.lookup.remove(old);
        self.lookup.insert(new, idx);
    }

    /// Adds a column to the **end** of the frame.
    pub fn add_column<L: Into<String>>(&mut self, name: L, column: Vec<T>) {
        let name = name.into();
        if self.lookup.contains_key(&name) {
            panic!("duplicate column label: {}", name);
        }
        self.matrix.add_column(self.matrix.cols(), column);
        self.column_names.push(name.clone());
        self.lookup.insert(name, self.matrix.cols() - 1);
    }

    /// Deletes a column and returns its data.
    pub fn delete_column(&mut self, name: &str) -> Vec<T> {
        let idx = self
            .column_index(name)
            .unwrap_or_else(|| panic!("unknown column label: {}", name));
        let mut col = Vec::with_capacity(self.matrix.rows());
        col.extend_from_slice(self.matrix.column(idx));
        self.matrix.delete_column(idx);
        self.column_names.remove(idx);
        self.rebuild_lookup();
        col
    }

    /// Sorts columns **lexicographically** by their names, *in‑place*.
    ///
    /// The operation is performed exclusively through calls to
    /// [`swap_columns`](Frame::swap_columns), which themselves defer to
    /// `Matrix::swap_columns`; thus we never re‑implement swapping logic.
    pub fn sort_columns(&mut self) {
        // Simple selection sort; complexity O(n²) but stable w.r.t matrix data.
        let n = self.column_names.len();
        for i in 0..n {
            let mut min = i;
            for j in (i + 1)..n {
                if self.column_names[j] < self.column_names[min] {
                    min = j;
                }
            }
            if min != i {
                // Use public API; keeps single source of truth.
                let col_i = self.column_names[i].clone();
                let col_min = self.column_names[min].clone();
                self.swap_columns(col_i, col_min);
            }
        }
    }

    /* ---------- helpers ---------- */

    fn rebuild_lookup(&mut self) {
        self.lookup.clear();
        for (i, name) in self.column_names.iter().enumerate() {
            self.lookup.insert(name.clone(), i);
        }
    }
}

/* ---------- Indexing ---------- */

impl<T: Clone> Index<&str> for Frame<T> {
    type Output = [T];
    fn index(&self, name: &str) -> &Self::Output {
        self.column(name)
    }
}
impl<T: Clone> IndexMut<&str> for Frame<T> {
    fn index_mut(&mut self, name: &str) -> &mut Self::Output {
        self.column_mut(name)
    }
}

/* ---------- Element‑wise numerical ops ---------- */
macro_rules! impl_elementwise_frame_op {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, 'b, T> std::ops::$OpTrait<&'b Frame<T>> for &'a Frame<T>
        where
            T: Clone + std::ops::$OpTrait<Output = T>,
        {
            type Output = Frame<T>;
            fn $method(self, rhs: &'b Frame<T>) -> Frame<T> {
                assert_eq!(self.column_names, rhs.column_names, "column names mismatch");
                let matrix = (&self.matrix).$method(&rhs.matrix);
                Frame::new(matrix, self.column_names.clone())
            }
        }
    };
}
impl_elementwise_frame_op!(Add, add, +);
impl_elementwise_frame_op!(Sub, sub, -);
impl_elementwise_frame_op!(Mul, mul, *);
impl_elementwise_frame_op!(Div, div, /);

/* ---------- Boolean‑specific bitwise ops ---------- */
macro_rules! impl_bitwise_frame_op {
    ($OpTrait:ident, $method:ident, $op:tt) => {
        impl<'a, 'b> std::ops::$OpTrait<&'b Frame<bool>> for &'a Frame<bool> {
            type Output = Frame<bool>;
            fn $method(self, rhs: &'b Frame<bool>) -> Frame<bool> {
                assert_eq!(self.column_names, rhs.column_names, "column names mismatch");
                let matrix = (&self.matrix).$method(&rhs.matrix);
                Frame::new(matrix, self.column_names.clone())
            }
        }
    };
}
impl_bitwise_frame_op!(BitAnd, bitand, &);
impl_bitwise_frame_op!(BitOr, bitor, |);
impl_bitwise_frame_op!(BitXor, bitxor, ^);

impl Not for Frame<bool> {
    type Output = Frame<bool>;
    fn not(self) -> Frame<bool> {
        Frame::new(!self.matrix, self.column_names)
    }
}
