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

// Unit Tests
#[cfg(test)]
mod tests {
    use super::{Frame, Matrix};

    // Helper function to create a standard test frame
    fn create_test_frame_i32() -> Frame<i32> {
        let matrix = Matrix::from_cols(vec![
            vec![1, 2, 3], // Col "A"
            vec![4, 5, 6], // Col "B"
            vec![7, 8, 9], // Col "C"
        ]);
        Frame::new(matrix, vec!["A", "B", "C"])
    }

    fn create_test_frame_bool() -> Frame<bool> {
        let matrix = Matrix::from_cols(vec![
            vec![true, false], // Col "P"
            vec![false, true], // Col "Q"
        ]);
        Frame::new(matrix, vec!["P", "Q"])
    }

    #[test]
    fn test_new_frame_success() {
        let matrix = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
        let frame = Frame::new(matrix.clone(), vec!["col1", "col2"]);

        assert_eq!(frame.column_names, vec!["col1", "col2"]);
        assert_eq!(frame.matrix(), &matrix);
        assert_eq!(frame.lookup.get("col1"), Some(&0));
        assert_eq!(frame.lookup.get("col2"), Some(&1));
        assert_eq!(frame.lookup.len(), 2);
    }

    #[test]
    #[should_panic(expected = "column name count mismatch")]
    fn test_new_frame_panic_name_count_mismatch() {
        let matrix = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
        Frame::new(matrix, vec!["col1"]); // Only one name for two columns
    }

    #[test]
    #[should_panic(expected = "duplicate column label: col1")]
    fn test_new_frame_panic_duplicate_names() {
        let matrix = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
        Frame::new(matrix, vec!["col1", "col1"]); // Duplicate name
    }

    #[test]
    fn test_accessors() {
        let mut frame = create_test_frame_i32();

        // matrix()
        assert_eq!(frame.matrix().rows(), 3);
        assert_eq!(frame.matrix().cols(), 3);

        // column()
        assert_eq!(frame.column("A"), &[1, 2, 3]);
        assert_eq!(frame.column("C"), &[7, 8, 9]);

        // column_mut()
        frame.column_mut("B")[1] = 50;
        assert_eq!(frame.column("B"), &[4, 50, 6]);

        // column_index()
        assert_eq!(frame.column_index("A"), Some(0));
        assert_eq!(frame.column_index("C"), Some(2));
        assert_eq!(frame.column_index("Z"), None);

        // matrix_mut() - check by modifying through matrix_mut
        *frame.matrix_mut().get_mut(0, 0) = 100; // Modify element at (0, 0) which is A[0]
        assert_eq!(frame.column("A"), &[100, 2, 3]);
    }
    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_column_panic_unknown_label() {
        let frame = create_test_frame_i32();
        frame.column("Z");
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_column_mut_panic_unknown_label() {
        let mut frame = create_test_frame_i32();
        frame.column_mut("Z");
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_swap_columns_panic_unknown_a() {
        let mut frame = create_test_frame_i32();
        frame.swap_columns("Z", "B");
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_swap_columns_panic_unknown_b() {
        let mut frame = create_test_frame_i32();
        frame.swap_columns("A", "Z");
    }

    #[test]
    fn test_rename_column() {
        let mut frame = create_test_frame_i32();
        let original_b_data = frame.column("B").to_vec();

        frame.rename("B", "Beta");

        // Check names
        assert_eq!(frame.column_names, vec!["A", "Beta", "C"]);

        // Check lookup
        assert_eq!(frame.column_index("A"), Some(0));
        assert_eq!(frame.column_index("Beta"), Some(1));
        assert_eq!(frame.column_index("C"), Some(2));
        assert_eq!(frame.column_index("B"), None); // Old name gone

        // Check data accessible via new name
        assert_eq!(frame.column("Beta"), original_b_data);
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_rename_panic_unknown_old() {
        let mut frame = create_test_frame_i32();
        frame.rename("Z", "Omega");
    }

    #[test]
    #[should_panic(expected = "duplicate column label: C")]
    fn test_rename_panic_duplicate_new() {
        let mut frame = create_test_frame_i32();
        frame.rename("A", "C"); // "C" already exists
    }

    #[test]
    fn test_add_column() {
        let mut frame = create_test_frame_i32();
        let new_col_data = vec![10, 11, 12];

        frame.add_column("D", new_col_data.clone());

        // Check names
        assert_eq!(frame.column_names, vec!["A", "B", "C", "D"]);

        // Check lookup
        assert_eq!(frame.column_index("D"), Some(3));

        // Check matrix dimensions
        assert_eq!(frame.matrix().cols(), 4);
        assert_eq!(frame.matrix().rows(), 3);

        // Check data of new column
        assert_eq!(frame.column("D"), new_col_data);
        // Check old columns are still there
        assert_eq!(frame.column("A"), &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "duplicate column label: B")]
    fn test_add_column_panic_duplicate_name() {
        let mut frame = create_test_frame_i32();
        frame.add_column("B", vec![0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "column length mismatch")]
    fn test_add_column_panic_length_mismatch() {
        let mut frame = create_test_frame_i32();
        // Matrix::add_column panics if lengths mismatch
        frame.add_column("D", vec![10, 11]); // Only 2 elements, expected 3
    }

    #[test]
    fn test_delete_column() {
        let mut frame = create_test_frame_i32();
        let original_b_data = frame.column("B").to_vec();
        let original_c_data = frame.column("C").to_vec(); // Need to check data shift

        let deleted_data = frame.delete_column("B");

        // Check returned data
        assert_eq!(deleted_data, original_b_data);

        // Check names
        assert_eq!(frame.column_names, vec!["A", "C"]);

        // Check lookup (rebuilt)
        assert_eq!(frame.column_index("A"), Some(0));
        assert_eq!(frame.column_index("C"), Some(1));
        assert_eq!(frame.column_index("B"), None);

        // Check matrix dimensions
        assert_eq!(frame.matrix().cols(), 2);
        assert_eq!(frame.matrix().rows(), 3);

        // Check remaining data
        assert_eq!(frame.column("A"), &[1, 2, 3]);
        assert_eq!(frame.column("C"), original_c_data); // "C" should now be at index 1
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_delete_column_panic_unknown() {
        let mut frame = create_test_frame_i32();
        frame.delete_column("Z");
    }

    #[test]
    fn test_sort_columns() {
        let matrix = Matrix::from_cols(vec![
            vec![7, 8, 9], // Col "C"
            vec![1, 2, 3], // Col "A"
            vec![4, 5, 6], // Col "B"
        ]);
        let mut frame = Frame::new(matrix, vec!["C", "A", "B"]);

        let orig_a = frame.column("A").to_vec();
        let orig_b = frame.column("B").to_vec();
        let orig_c = frame.column("C").to_vec();

        frame.sort_columns();

        // Check names order
        assert_eq!(frame.column_names, vec!["A", "B", "C"]);

        // Check lookup map
        assert_eq!(frame.column_index("A"), Some(0));
        assert_eq!(frame.column_index("B"), Some(1));
        assert_eq!(frame.column_index("C"), Some(2));

        // Check data integrity (data moved with the names)
        assert_eq!(frame.column("A"), orig_a);
        assert_eq!(frame.column("B"), orig_b);
        assert_eq!(frame.column("C"), orig_c);
    }

    #[test]
    fn test_sort_columns_single_column() {
        let matrix = Matrix::from_cols(vec![vec![1, 2, 3]]);
        let mut frame = Frame::new(matrix.clone(), vec!["Solo"]);
        let expected = frame.clone();
        frame.sort_columns();
        assert_eq!(frame, expected); // Should be unchanged
    }

    #[test]
    fn test_index() {
        let frame = create_test_frame_i32();
        assert_eq!(frame["A"].to_vec(), vec![1, 2, 3]);
        assert_eq!(frame["C"].to_vec(), vec![7, 8, 9]);
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_index_panic_unknown() {
        let frame = create_test_frame_i32();
        let _ = frame["Z"];
    }

    #[test]
    fn test_index_mut() {
        let mut frame = create_test_frame_i32();
        frame["B"][0] = 42;
        frame["C"][2] = 99;

        assert_eq!(frame["B"].to_vec(), &[42, 5, 6]);
        assert_eq!(frame["C"].to_vec(), &[7, 8, 99]);
    }

    #[test]
    #[should_panic(expected = "unknown column label: Z")]
    fn test_index_mut_panic_unknown() {
        let mut frame = create_test_frame_i32();
        let _ = &mut frame["Z"];
    }

    // --- Test Ops ---
    #[test]
    fn test_elementwise_ops_numeric() {
        let frame1 = create_test_frame_i32(); // A=[1,2,3], B=[4,5,6], C=[7,8,9]
        let matrix2 = Matrix::from_cols(vec![
            vec![10, 10, 10], // Col "A"
            vec![2, 2, 2],    // Col "B"
            vec![1, 1, 1],    // Col "C"
        ]);
        let frame2 = Frame::new(matrix2, vec!["A", "B", "C"]); // Must have same names and dims

        // Add
        let frame_add = &frame1 + &frame2;
        assert_eq!(frame_add.column_names, frame1.column_names);
        assert_eq!(frame_add["A"].to_vec(), &[11, 12, 13]);
        assert_eq!(frame_add["B"].to_vec(), &[6, 7, 8]);
        assert_eq!(frame_add["C"].to_vec(), &[8, 9, 10]);

        // Sub
        let frame_sub = &frame1 - &frame2;
        assert_eq!(frame_sub.column_names, frame1.column_names);
        assert_eq!(frame_sub["A"].to_vec(), &[-9, -8, -7]);
        assert_eq!(frame_sub["B"].to_vec(), &[2, 3, 4]);
        assert_eq!(frame_sub["C"].to_vec(), &[6, 7, 8]);

        // Mul
        let frame_mul = &frame1 * &frame2;
        assert_eq!(frame_mul.column_names, frame1.column_names);
        assert_eq!(frame_mul["A"].to_vec(), &[10, 20, 30]);
        assert_eq!(frame_mul["B"].to_vec(), &[8, 10, 12]);
        assert_eq!(frame_mul["C"].to_vec(), &[7, 8, 9]);

        // Div
        let frame_div = &frame1 / &frame2; // Integer division
        assert_eq!(frame_div.column_names, frame1.column_names);
        assert_eq!(frame_div["A"].to_vec(), &[0, 0, 0]); // 1/10, 2/10, 3/10
        assert_eq!(frame_div["B"].to_vec(), &[2, 2, 3]); // 4/2, 5/2, 6/2
        assert_eq!(frame_div["C"].to_vec(), &[7, 8, 9]); // 7/1, 8/1, 9/1
    }

    #[test]
    #[should_panic] // Exact message depends on Matrix op panic message ("row count mismatch" or "col count mismatch")
    fn test_elementwise_op_panic_dimension_mismatch() {
        let frame1 = create_test_frame_i32(); // 3x3
        let matrix2 = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]); // 2x2
        let frame2 = Frame::new(matrix2, vec!["X", "Y"]);
        let _ = &frame1 + &frame2; // Should panic due to dimension mismatch
    }

    #[test]
    fn test_bitwise_ops_bool() {
        let frame1 = create_test_frame_bool(); // P=[T, F], Q=[F, T]
        let matrix2 = Matrix::from_cols(vec![
            vec![true, true],   // P
            vec![false, false], // Q
        ]);
        let frame2 = Frame::new(matrix2, vec!["P", "Q"]);

        // BitAnd
        let frame_and = &frame1 & &frame2;
        assert_eq!(frame_and.column_names, frame1.column_names);
        assert_eq!(frame_and["P"].to_vec(), &[true, false]); // T&T=T, F&T=F
        assert_eq!(frame_and["Q"].to_vec(), &[false, false]); // F&F=F, T&F=F

        // BitOr
        let frame_or = &frame1 | &frame2;
        assert_eq!(frame_or.column_names, frame1.column_names);
        assert_eq!(frame_or["P"].to_vec(), &[true, true]); // T|T=T, F|T=T
        assert_eq!(frame_or["Q"].to_vec(), &[false, true]); // F|F=F, T|F=T

        // BitXor
        let frame_xor = &frame1 ^ &frame2;
        assert_eq!(frame_xor.column_names, frame1.column_names);
        assert_eq!(frame_xor["P"].to_vec(), &[false, true]); // T^T=F, F^T=T
        assert_eq!(frame_xor["Q"].to_vec(), &[false, true]); // F^F=F, T^F=T
    }

    #[test]
    fn test_not_op_bool() {
        let frame = create_test_frame_bool(); // P=[T, F], Q=[F, T]
        let frame_not = !frame; // Note: consumes the original frame

        assert_eq!(frame_not.column_names, vec!["P", "Q"]);
        assert_eq!(frame_not["P"].to_vec(), &[false, true]);
        assert_eq!(frame_not["Q"].to_vec(), &[true, false]);
    }

    #[test]
    fn test_swap_columns() {
        let mut frame = create_test_frame_i32();
        let initial_a_data = frame.column("A").to_vec(); // [1, 2, 3]
        let initial_c_data = frame.column("C").to_vec(); // [7, 8, 9]

        frame.swap_columns("A", "C");

        // Check names order
        assert_eq!(frame.column_names, vec!["C", "B", "A"]);

        // Check lookup map
        assert_eq!(frame.column_index("A"), Some(2));
        assert_eq!(frame.column_index("B"), Some(1));
        assert_eq!(frame.column_index("C"), Some(0));

        // Check data using new names (should be swapped)

        // Accessing by name "C" (now at index 0) should retrieve the data
        // that was swapped INTO index 0, which was the *original C data*.
        assert_eq!(
            frame.column("C"),
            initial_c_data.as_slice(),
            "Data for name 'C' should be original C data"
        );

        // Accessing by name "A" (now at index 2) should retrieve the data
        // that was swapped INTO index 2, which was the *original A data*.
        assert_eq!(
            frame.column("A"),
            initial_a_data.as_slice(),
            "Data for name 'A' should be original A data"
        );

        // Column "B" should remain unchanged in data and position.
        assert_eq!(
            frame.column("B"),
            &[4, 5, 6],
            "Column 'B' should be unchanged"
        );

        // Test swapping with self
        let state_before_self_swap = frame.clone();
        frame.swap_columns("B", "B");
        assert_eq!(frame, state_before_self_swap);
    }
}
