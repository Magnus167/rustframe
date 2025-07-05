use crate::matrix::Matrix;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut, Not, Range};

// Helper enums and structs for indexing support

/// Represents the different types of row indices a Frame can have.
#[derive(Debug, Clone, PartialEq, Eq)] // Derive Eq to enable usage as HashMap keys
pub enum RowIndex {
    /// Integer-based index (e.g., 0, 1, 2, ...). Values must be unique.
    Int(Vec<usize>),
    /// Date-based index. Values must be unique. Order is preserved as given.
    Date(Vec<NaiveDate>),
    /// Default range index (0..num_rows) used when no specific index is provided.
    Range(Range<usize>),
}

impl RowIndex {
    /// Returns the number of elements in the index.
    pub fn len(&self) -> usize {
        match self {
            RowIndex::Int(v) => v.len(),
            RowIndex::Date(v) => v.len(),
            RowIndex::Range(r) => r.end.saturating_sub(r.start),
        }
    }

    /// Checks if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Internal helper for fast lookups from index value to physical row position.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RowIndexLookup {
    Int(HashMap<usize, usize>),
    Date(HashMap<NaiveDate, usize>),
    None, // Variant for default range-based indexing
}

// Frame struct definition and associated implementations

/// A data frame - a Matrix with string-identified columns and a typed row index.
///
/// `Frame` extends the concept of a `Matrix` by adding named columns
/// and an index for rows, which can be integers, dates, or a default range.
/// It allows accessing data by column name (using `[]`) and by row index value
/// (using `get_row`, `get_row_mut`, `get_row_date`, `get_row_date_mut` methods).
///
/// Direct row indexing with `frame[row_key]` is not supported due to Rust's
/// lifetime and ownership rules clashing with the `std::ops::Index` trait when
/// returning row views or temporary copies. Use the explicit `get_row*` methods instead.
///
/// The internal data is stored column-major in a `Matrix<T>`.
///
/// # Type Parameters
///
/// * `T`: The data type of the elements within the frame. Must be `Clone + PartialEq`.
///        Numerical/Boolean traits required for specific operations (e.g., `Add`, `BitAnd`).
///
/// # Examples
///
/// ```
/// use rustframe::frame::{Frame, RowIndex};
/// use rustframe::matrix::Matrix; // Assume Matrix is available
/// use chrono::NaiveDate;
///
/// // Helper fn for dates
/// fn d(y: i32, m: u32, d: u32) -> NaiveDate { NaiveDate::from_ymd_opt(y,m,d).unwrap() }
///
/// // --- Example 1: Basic Creation and Access ---
/// let matrix_f64 = Matrix::from_cols(vec![
///     vec![1.0, 2.0, 3.0], // Column "A"
///     vec![4.0, 5.0, 6.0], // Column "B"
/// ]);
/// let mut frame1 = Frame::new(matrix_f64, vec!["A", "B"], None);
///
/// assert_eq!(frame1.columns(), &["A", "B"]);
/// assert_eq!(frame1["A"], vec![1.0, 2.0, 3.0]); // Compare with slice literal
/// assert_eq!(frame1.index(), &RowIndex::Range(0..3));
/// let row0 = frame1.get_row(0);
/// assert_eq!(row0["A"], 1.0);
/// assert_eq!(row0[1], 4.0); // Column "B"
///
/// // --- Example 2: Date Index and Mutation ---
/// let dates = vec![d(2024, 1, 1), d(2024, 1, 2)];
/// let matrix_string = Matrix::from_cols(vec![ vec!["X".to_string(), "Y".to_string()], ]);
/// let mut frame2 = Frame::new(matrix_string, vec!["Label"], Some(RowIndex::Date(dates.clone())));
///
/// assert_eq!(frame2.index(), &RowIndex::Date(dates));
/// assert_eq!(frame2.get_row_date(d(2024, 1, 2))["Label"], "Y");
/// frame2.get_row_date_mut(d(2024, 1, 1)).set_by_index(0, "Z".to_string());
/// assert_eq!(frame2["Label"], vec!["Z", "Y"]);
///
/// // --- Example 3: Element-wise Addition ---
/// let m1 = Matrix::from_cols(vec![ vec![1.0, 2.0], vec![3.0, 4.0] ]);
/// let f1 = Frame::new(m1, vec!["C1", "C2"], None);
/// let m2 = Matrix::from_cols(vec![ vec![0.1, 0.2], vec![0.3, 0.4] ]);
/// let f2 = Frame::new(m2, vec!["C1", "C2"], None);
///
/// let f_sum = &f1 + &f2;
/// assert_eq!(f_sum["C1"], vec![1.1, 2.2]);
/// assert_eq!(f_sum["C2"], vec![3.3, 4.4]);
/// assert_eq!(f_sum.index(), &RowIndex::Range(0..2));
///
/// // --- Example 4: Element-wise Multiplication ---
/// let f_prod = &f1 * &f2;
/// assert!((f_prod["C1"][0] - 0.1_f64).abs() < 1e-9);
/// assert!((f_prod["C1"][1] - 0.4_f64).abs() < 1e-9);
/// assert!((f_prod["C2"][0] - 0.9_f64).abs() < 1e-9);
/// assert!((f_prod["C2"][1] - 1.6_f64).abs() < 1e-9);
///
/// // --- Example 5: Column Manipulation and Sorting ---
/// let mut frame_manip = Frame::new(
///     Matrix::from_cols(vec![ vec![1, 2], vec![3, 4] ]), // Example uses i32
///     vec!["DataC", "DataA"], // Column names (out of order)
///     None
/// );
/// assert_eq!(frame_manip["DataC"], vec![1, 2]);
/// assert_eq!(frame_manip["DataA"], vec![3, 4]);
/// frame_manip.add_column("DataB", vec![5, 6]);
/// assert_eq!(frame_manip.columns(), &["DataC", "DataA", "DataB"]);
/// frame_manip.rename("DataA", "DataX"); // Rename A -> X
/// assert_eq!(frame_manip.columns(), &["DataC", "DataX", "DataB"]);
/// assert_eq!(frame_manip["DataX"], vec![3, 4]); // X has A's original data
/// let deleted_c = frame_manip.delete_column("DataC");
/// assert_eq!(deleted_c, vec![1, 2]);
/// assert_eq!(frame_manip.columns(), &["DataX", "DataB"]);
/// frame_manip.sort_columns();
/// assert_eq!(frame_manip.columns(), &["DataB", "DataX"]);
/// assert_eq!(frame_manip["DataB"], vec![5, 6]);
/// assert_eq!(frame_manip["DataX"], vec![3, 4]);
/// ```

// Custom Debug implementation for Frame
impl<T: Clone + PartialEq + fmt::Debug> fmt::Debug for Frame<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("column_names", &self.column_names)
            .field("index", &self.index)
            .field("matrix_dims", &(self.matrix.rows(), self.matrix.cols()))
            .field("col_lookup", &self.col_lookup)
            .field("index_lookup", &self.index_lookup)
            // Matrix data is omitted from Debug output by default
            // .field("matrix", &self.matrix)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub struct Frame<T: Clone + PartialEq> {
    /// Vector holding the column names in their current order.
    column_names: Vec<String>,
    /// The underlying column-major matrix storing the data.
    matrix: Matrix<T>,
    /// Maps a column name to its physical column index for **O(1)** lookup.
    col_lookup: HashMap<String, usize>,
    /// The row index values (Int, Date, or Range).
    index: RowIndex,
    /// Internal lookup for mapping index values to physical row positions.
    index_lookup: RowIndexLookup,
}

impl<T: Clone + PartialEq> Frame<T> {
    // Constructors

    /// Creates a new [`Frame`] from a matrix, column names, and an optional row index.
    /// Panics if the underlying `Matrix` requirements are not met (e.g., rows>0, cols>0 for standard constructors),
    /// or if column name/index constraints are violated.
    pub fn new<L: Into<String>>(matrix: Matrix<T>, names: Vec<L>, index: Option<RowIndex>) -> Self {
        // Validate that the number of column names matches the matrix's column count.
        if matrix.cols() != names.len() {
            panic!(
                "Frame::new: column name count mismatch (names: {}, matrix: {})",
                names.len(),
                matrix.cols()
            );
        }
        // Matrix creation already enforces non-zero rows and columns.

        // Build column name list and lookup map, ensuring no duplicates.
        let mut col_lookup = HashMap::with_capacity(names.len());
        let column_names: Vec<String> = names
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                let s = n.into();
                if col_lookup.insert(s.clone(), i).is_some() {
                    panic!("Frame::new: duplicate column label: {}", s);
                }
                s
            })
            .collect();

        // Validate and construct the row index and its lookup structure.
        let num_rows = matrix.rows();
        let (index_values, index_lookup) = match index {
            Some(RowIndex::Int(vals)) => {
                if vals.len() != num_rows {
                    panic!(
                        "Frame::new: Int index length ({}) mismatch matrix rows ({})",
                        vals.len(),
                        num_rows
                    );
                }
                // Build integer-to-physical-row mapping.
                let mut lookup = HashMap::with_capacity(num_rows);
                for (physical_row, index_val) in vals.iter().enumerate() {
                    if lookup.insert(*index_val, physical_row).is_some() {
                        panic!("Frame::new: duplicate Int index value: {}", index_val);
                    }
                }
                (RowIndex::Int(vals), RowIndexLookup::Int(lookup))
            }
            Some(RowIndex::Date(vals)) => {
                if vals.len() != num_rows {
                    panic!(
                        "Frame::new: Date index length ({}) mismatch matrix rows ({})",
                        vals.len(),
                        num_rows
                    );
                }
                // Build date-to-physical-row mapping.
                let mut lookup = HashMap::with_capacity(num_rows);
                for (physical_row, index_val) in vals.iter().enumerate() {
                    if lookup.insert(*index_val, physical_row).is_some() {
                        panic!("Frame::new: duplicate Date index value: {}", index_val);
                    }
                }
                (RowIndex::Date(vals), RowIndexLookup::Date(lookup))
            }
            Some(RowIndex::Range(ref r)) => {
                // If the length of the range does not match the number of rows, panic.
                if r.end.saturating_sub(r.start) != num_rows {
                    panic!(
                        "Frame::new: Range index length ({}) mismatch matrix rows ({})",
                        r.end.saturating_sub(r.start),
                        num_rows
                    );
                }
                // return the range as is.
                (RowIndex::Range(r.clone()), RowIndexLookup::None)
            }
            None => {
                // Default to a sequential range index.
                (RowIndex::Range(0..num_rows), RowIndexLookup::None)
            }
        };

        Self {
            matrix,
            column_names,
            col_lookup,
            index: index_values,
            index_lookup,
        }
    }

    // Accessors

    /// Returns an immutable reference to the underlying `Matrix`.
    #[inline]
    pub fn matrix(&self) -> &Matrix<T> {
        &self.matrix
    }

    /// Returns a mutable reference to the underlying `Matrix`.
    /// Use with caution: direct matrix edits bypass frame-level name/index consistency checks.
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut Matrix<T> {
        &mut self.matrix
    }

    /// Returns the list of column names in their current order.
    #[inline]
    pub fn columns(&self) -> &[String] {
        &self.column_names
    }

    /// Returns a reference to the frame's row index.
    #[inline]
    pub fn index(&self) -> &RowIndex {
        &self.index
    }

    /// Returns the number of rows in the frame.
    #[inline]
    pub fn rows(&self) -> usize {
        self.matrix.rows()
    }

    /// Returns the number of columns in the frame.
    #[inline]
    pub fn cols(&self) -> usize {
        self.matrix.cols()
    }

    /// Returns the physical column index for the given column name, if present.
    #[inline]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.col_lookup.get(name).copied()
    }

    /// Internal helper to compute the physical row index from a logical key.
    fn get_physical_row_index<Idx>(&self, index_key: Idx) -> usize
    where
        Self: RowIndexLookupHelper<Idx>, // Requires `RowIndexLookupHelper` for `Idx`
    {
        <Self as RowIndexLookupHelper<Idx>>::lookup_row_index(
            index_key,
            &self.index,
            &self.index_lookup,
        )
    }

    /// Returns an immutable slice of the specified column's data.
    /// Panics if the column name is not found.
    pub fn column(&self, name: &str) -> &[T] {
        let idx = self
            .column_index(name)
            .unwrap_or_else(|| panic!("Frame::column: unknown column label: '{}'", name));
        self.matrix.column(idx)
    }

    /// Returns a mutable slice of the specified column's data.
    /// Panics if the column name is not found.
    pub fn column_mut(&mut self, name: &str) -> &mut [T] {
        let idx = self
            .column_index(name)
            .unwrap_or_else(|| panic!("Frame::column_mut: unknown column label: '{}'", name));
        self.matrix.column_mut(idx)
    }

    // Row access methods

    /// Returns an immutable view of the row for the given integer key.
    /// Panics if the key is invalid or if the index type is `Date`.
    pub fn get_row(&self, index_key: usize) -> FrameRowView<'_, T> {
        let idx = self.get_physical_row_index(index_key);
        FrameRowView {
            frame: self,
            physical_row_idx: idx,
        }
    }

    /// Returns a mutable view of the row for the given integer key.
    /// Panics if the key is invalid or if the index type is `Date`.
    pub fn get_row_mut(&mut self, index_key: usize) -> FrameRowViewMut<'_, T> {
        let idx = self.get_physical_row_index(index_key);
        FrameRowViewMut {
            frame: self,
            physical_row_idx: idx,
        }
    }

    /// Returns an immutable view of the row for the given date key.
    /// Panics if the key is invalid or if the index type is not `Date`.
    pub fn get_row_date(&self, index_key: NaiveDate) -> FrameRowView<'_, T> {
        let idx = self.get_physical_row_index(index_key);
        FrameRowView {
            frame: self,
            physical_row_idx: idx,
        }
    }

    /// Returns a mutable view of the row for the given date key.
    /// Panics if the key is invalid or if the index type is not `Date`.
    pub fn get_row_date_mut(&mut self, index_key: NaiveDate) -> FrameRowViewMut<'_, T> {
        let idx = self.get_physical_row_index(index_key);
        FrameRowViewMut {
            frame: self,
            physical_row_idx: idx,
        }
    }

    // Column manipulation

    /// Internal helper to swap two columns, updating the matrix, name list, and lookup map.
    /// Not intended for public use; prefer `sort_columns`.
    fn _swap_columns_internal(&mut self, a: &str, b: &str) {
        // Retrieve physical indices for the given column names.
        let ia = self.column_index(a).unwrap_or_else(|| {
            panic!("Frame::_swap_columns_internal: unknown column label: {}", a)
        });
        let ib = self.column_index(b).unwrap_or_else(|| {
            panic!("Frame::_swap_columns_internal: unknown column label: {}", b)
        });

        if ia == ib {
            return; // No action needed if columns are identical
        }

        // Swap the columns in the underlying matrix.
        self.matrix.swap_columns(ia, ib);
        // Swap the names in the ordered list.
        self.column_names.swap(ia, ib);
        // Update lookup entries to reflect the new positions.
        self.col_lookup.insert(a.to_string(), ib);
        self.col_lookup.insert(b.to_string(), ia);
    }

    /// Renames an existing column.
    /// Panics if the source name is missing or if the target name is already in use.
    pub fn rename<L: Into<String>>(&mut self, old: &str, new: L) {
        let new_name = new.into();
        if old == new_name {
            panic!(
                "Frame::rename: new name '{}' cannot be the same as the old name",
                new_name
            );
        }
        let idx = self
            .column_index(old)
            .unwrap_or_else(|| panic!("Frame::rename: unknown column label: '{}'", old));
        if self.col_lookup.contains_key(&new_name) {
            panic!(
                "Frame::rename: new column name '{}' already exists",
                new_name
            );
        }

        // Update lookup and ordered name list.
        self.col_lookup.remove(old);
        self.col_lookup.insert(new_name.clone(), idx);
        self.column_names[idx] = new_name;
    }

    /// Adds a new column at the end of the frame.
    /// Panics if the column name exists or if the data length mismatches the row count.
    pub fn add_column<L: Into<String>>(&mut self, name: L, column_data: Vec<T>) {
        let name_str = name.into();
        if self.col_lookup.contains_key(&name_str) {
            panic!("Frame::add_column: duplicate column label: {}", name_str);
        }
        // Matrix enforces that the new column length matches the frame's row count.
        let new_col_idx = self.matrix.cols();
        self.matrix.add_column(new_col_idx, column_data);

        // Update metadata to include the new column.
        self.column_names.push(name_str.clone());
        self.col_lookup.insert(name_str, new_col_idx);
    }

    /// Deletes a column by name and returns its data.
    /// Panics if the column name is not found.
    pub fn delete_column(&mut self, name: &str) -> Vec<T> {
        let idx = self
            .column_index(name)
            .unwrap_or_else(|| panic!("Frame::delete_column: unknown column label: '{}'", name));

        // Clone out the data before removal.
        let deleted_data = self.matrix.column(idx).to_vec();
        self.matrix.delete_column(idx);

        // Remove from metadata and rebuild lookup for shifted columns.
        self.column_names.remove(idx);
        self.col_lookup.remove(name);
        self.rebuild_col_lookup();

        // If all columns are removed, reset default range index.
        if self.cols() == 0 {
            if let RowIndex::Range(_) = self.index {
                self.index = RowIndex::Range(0..self.rows());
                self.index_lookup = RowIndexLookup::None;
            }
        }

        deleted_data
    }


    /// Returns a new `Matrix` that is the transpose of the current frame's matrix.
    pub fn transpose(&mut self) -> Matrix<T> {
        self.matrix.transpose()
    }

    /// Sorts columns alphabetically by name, preserving data associations.
    pub fn sort_columns(&mut self) {
        let n = self.column_names.len();
        if n <= 1 {
            return; // Nothing to sort
        }

        // Selection sort on column names.
        for i in 0..n {
            let mut min_idx = i;
            for j in (i + 1)..n {
                if self.column_names[j] < self.column_names[min_idx] {
                    min_idx = j;
                }
            }
            if min_idx != i {
                let col_i_name = self.column_names[i].clone();
                let col_min_name = self.column_names[min_idx].clone();
                self._swap_columns_internal(&col_i_name, &col_min_name);
            }
        }

        // Debug-only consistency check.
        #[cfg(debug_assertions)]
        {
            let mut temp_lookup = HashMap::with_capacity(self.cols());
            for (idx, name) in self.column_names.iter().enumerate() {
                temp_lookup.insert(name.clone(), idx);
            }
            assert_eq!(
                self.col_lookup, temp_lookup,
                "Inconsistent col_lookup after sort_columns"
            );
        }
    }

    // Internal helpers

    /// Rebuilds the column lookup map to match the current `column_names` ordering.
    fn rebuild_col_lookup(&mut self) {
        self.col_lookup.clear();
        for (i, name) in self.column_names.iter().enumerate() {
            self.col_lookup.insert(name.clone(), i);
        }
    }
}

// Trait for resolving logical to physical row indices
/// Internal trait to abstract the logic for looking up physical row indices.
trait RowIndexLookupHelper<Idx> {
    fn lookup_row_index(key: Idx, index_values: &RowIndex, index_lookup: &RowIndexLookup) -> usize;
}

impl<T: Clone + PartialEq> RowIndexLookupHelper<usize> for Frame<T> {
    fn lookup_row_index(
        key: usize,
        index_values: &RowIndex,
        index_lookup: &RowIndexLookup,
    ) -> usize {
        match (index_values, index_lookup) {
            (RowIndex::Int(_), RowIndexLookup::Int(lookup)) => {
                // Constant-time lookup using hash map
                *lookup.get(&key).unwrap_or_else(|| {
                    panic!("Frame index: integer key {} not found in Int index", key)
                })
            }
            (RowIndex::Range(range), RowIndexLookup::None) => {
                // Direct-range mapping with boundary check
                if range.contains(&key) {
                    // Since Range always starts at 0, the physical index equals the key
                    key
                } else {
                    panic!(
                        "Frame index: integer key {} out of bounds for Range index {:?}",
                        key, range
                    );
                }
            }
            (RowIndex::Date(_), _) => {
                panic!("Frame index: incompatible key type usize for Date index")
            }
            // Fallback panic on inconsistent internal index state
            #[allow(unreachable_patterns)]
            _ => {
                panic!(
                    "Frame index: inconsistent internal index state (lookup type mismatch for usize key)"
                )
            }
        }
    }
}

impl<T: Clone + PartialEq> RowIndexLookupHelper<NaiveDate> for Frame<T> {
    fn lookup_row_index(
        key: NaiveDate,
        index_values: &RowIndex,
        index_lookup: &RowIndexLookup,
    ) -> usize {
        match (index_values, index_lookup) {
            (RowIndex::Date(_), RowIndexLookup::Date(lookup)) => {
                // Constant-time lookup via hash map
                *lookup.get(&key).unwrap_or_else(|| {
                    panic!("Frame index: date key {} not found in Date index", key)
                })
            }
            (RowIndex::Int(_), _) | (RowIndex::Range(_), _) => {
                panic!("Frame index: incompatible key type NaiveDate for Int or Range index")
            }
            // Fallback panic on inconsistent internal index state
            #[allow(unreachable_patterns)]
            _ => {
                panic!(
                    "Frame index: inconsistent internal index state (lookup type mismatch for NaiveDate key)"
                )
            }
        }
    }
}

// Row view types for frame rows

/// An immutable view of a single row in a `Frame`. Allows access via `[]`.
pub struct FrameRowView<'a, T: Clone + PartialEq> {
    frame: &'a Frame<T>,
    physical_row_idx: usize,
}

impl<'a, T: Clone + PartialEq + fmt::Debug> fmt::Debug for FrameRowView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Gather row data for debug output
        let row_data: Vec<&T> = (0..self.frame.cols())
            .map(|c| &self.frame.matrix[(self.physical_row_idx, c)])
            .collect();
        f.debug_struct("FrameRowView")
            .field("physical_row_idx", &self.physical_row_idx)
            .field("columns", &self.frame.column_names)
            .field("data", &row_data)
            .finish()
    }
}

impl<'a, T: Clone + PartialEq> FrameRowView<'a, T> {
    /// Returns a reference to the element at the given physical column index.
    /// Panics with a descriptive message if `col_idx` is out of bounds.
    pub fn get_by_index(&self, col_idx: usize) -> &T {
        if col_idx >= self.frame.cols() {
            panic!(
                "FrameRowView::get_by_index: column index {} out of bounds (frame has {} columns)",
                col_idx,
                self.frame.cols()
            );
        }
        &self.frame.matrix[(self.physical_row_idx, col_idx)]
    }

    /// Returns a reference to the element in the column named `col_name`.
    /// Panics if the column name is not found.
    pub fn get(&self, col_name: &str) -> &T {
        let idx = self
            .frame
            .column_index(col_name)
            .unwrap_or_else(|| panic!("FrameRowView::get: unknown column '{}'", col_name));
        self.get_by_index(idx)
    }
}

// Immutable indexing by column name
impl<'a, T: Clone + PartialEq> Index<&str> for FrameRowView<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, col_name: &str) -> &Self::Output {
        self.get(col_name)
    }
}

// Immutable indexing by physical column index
impl<'a, T: Clone + PartialEq> Index<usize> for FrameRowView<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, col_idx: usize) -> &Self::Output {
        self.get_by_index(col_idx)
    }
}

/// A mutable view of a single row in a `Frame`.  
/// Supports indexed access and mutation via methods or `[]` operators.
pub struct FrameRowViewMut<'a, T: Clone + PartialEq> {
    frame: &'a mut Frame<T>,
    physical_row_idx: usize,
}

impl<'a, T: Clone + PartialEq + fmt::Debug> fmt::Debug for FrameRowViewMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only display the row index and column names to avoid conflicting borrows
        f.debug_struct("FrameRowViewMut")
            .field("physical_row_idx", &self.physical_row_idx)
            .field("columns", &self.frame.column_names)
            .finish()
    }
}

impl<'a, T: Clone + PartialEq> FrameRowViewMut<'a, T> {
    /// Returns a mutable reference to the element at the given physical column index.
    /// Panics if `col_idx` is out of bounds.
    pub fn get_by_index_mut(&mut self, col_idx: usize) -> &mut T {
        let num_cols = self.frame.cols();
        if col_idx >= num_cols {
            panic!(
                "FrameRowViewMut::get_by_index_mut: column index {} out of bounds (frame has {} columns)",
                col_idx, num_cols
            );
        }
        &mut self.frame.matrix[(self.physical_row_idx, col_idx)]
    }

    /// Returns a mutable reference to the element in the column named `col_name`.
    /// Panics if the column name is not found.
    pub fn get_mut(&mut self, col_name: &str) -> &mut T {
        let idx = self
            .frame
            .column_index(col_name)
            .unwrap_or_else(|| panic!("FrameRowViewMut::get_mut: unknown column '{}'", col_name));
        self.get_by_index_mut(idx)
    }

    /// Sets the element at the given physical column index to `value`.
    /// Panics if `col_idx` is out of bounds.
    pub fn set_by_index(&mut self, col_idx: usize, value: T) {
        *self.get_by_index_mut(col_idx) = value;
    }

    /// Sets the element in the column named `col_name` to `value`.
    /// Panics if the column name is not found.
    pub fn set(&mut self, col_name: &str, value: T) {
        *self.get_mut(col_name) = value;
    }

    // Read-only helpers for Index implementations

    /// Immutable reference to the element at the given physical column index.
    fn get_by_index_ref(&self, col_idx: usize) -> &T {
        if col_idx >= self.frame.cols() {
            panic!(
                "FrameRowViewMut::get_by_index_ref: column index {} out of bounds (frame has {} columns)",
                col_idx,
                self.frame.cols()
            );
        }
        &self.frame.matrix[(self.physical_row_idx, col_idx)]
    }

    /// Immutable reference to the element in the column named `col_name`.
    fn get_ref(&self, col_name: &str) -> &T {
        let idx = self
            .frame
            .column_index(col_name)
            .unwrap_or_else(|| panic!("FrameRowViewMut::get_ref: unknown column '{}'", col_name));
        self.get_by_index_ref(idx)
    }
}

// Immutable indexing by column name
impl<'a, T: Clone + PartialEq> Index<&str> for FrameRowViewMut<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, col_name: &str) -> &Self::Output {
        self.get_ref(col_name)
    }
}

// Immutable indexing by physical column index
impl<'a, T: Clone + PartialEq> Index<usize> for FrameRowViewMut<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, col_idx: usize) -> &Self::Output {
        self.get_by_index_ref(col_idx)
    }
}

// Mutable indexing by column name
impl<'a, T: Clone + PartialEq> IndexMut<&str> for FrameRowViewMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, col_name: &str) -> &mut Self::Output {
        self.get_mut(col_name)
    }
}

// Mutable indexing by physical column index
impl<'a, T: Clone + PartialEq> IndexMut<usize> for FrameRowViewMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, col_idx: usize) -> &mut Self::Output {
        self.get_by_index_mut(col_idx)
    }
}

/// Enables immutable access to a column's data via `frame["col_name"]`.
impl<T: Clone + PartialEq> Index<&str> for Frame<T> {
    type Output = [T];
    #[inline]
    fn index(&self, name: &str) -> &Self::Output {
        self.column(name)
    }
}

/// Enables mutable access to a column's data via `frame["col_name"]`.
impl<T: Clone + PartialEq> IndexMut<&str> for Frame<T> {
    #[inline]
    fn index_mut(&mut self, name: &str) -> &mut Self::Output {
        self.column_mut(name)
    }
}

/* ---------- Element-wise Arithmetic Operations ---------- */
/// Generates implementations for element-wise arithmetic (`+`, `-`, `*`, `/`) on `Frame<T>`.
/// Panics if column labels or row indices differ between operands.
macro_rules! impl_elementwise_frame_op {
    ($OpTrait:ident, $method:ident) => {
        // &Frame<T> $OpTrait &Frame<T>
        impl<'a, 'b, T> std::ops::$OpTrait<&'b Frame<T>> for &'a Frame<T>
        where
            T: Clone + PartialEq + std::ops::$OpTrait<Output = T>,
        {
            type Output = Frame<T>;
            fn $method(self, rhs: &'b Frame<T>) -> Frame<T> {
                if self.column_names != rhs.column_names {
                    panic!(
                        "Element-wise {}: column names do not match. Left: {:?}, Right: {:?}",
                        stringify!($method),
                        self.column_names,
                        rhs.column_names
                    );
                }
                if self.index != rhs.index {
                    panic!(
                        "Element-wise {}: row indices do not match. Left: {:?}, Right: {:?}",
                        stringify!($method),
                        self.index,
                        rhs.index
                    );
                }
                let result_matrix = (&self.matrix).$method(&rhs.matrix);
                let new_index = match self.index {
                    RowIndex::Range(_) => None,
                    _ => Some(self.index.clone()),
                };
                Frame::new(result_matrix, self.column_names.clone(), new_index)
            }
        }
        // Frame<T> $OpTrait &Frame<T>
        impl<'b, T> std::ops::$OpTrait<&'b Frame<T>> for Frame<T>
        where
            T: Clone + PartialEq + std::ops::$OpTrait<Output = T>,
        {
            type Output = Frame<T>;
            fn $method(self, rhs: &'b Frame<T>) -> Frame<T> {
                (&self).$method(rhs)
            }
        }
        // &Frame<T> $OpTrait Frame<T>
        impl<'a, T> std::ops::$OpTrait<Frame<T>> for &'a Frame<T>
        where
            T: Clone + PartialEq + std::ops::$OpTrait<Output = T>,
        {
            type Output = Frame<T>;
            fn $method(self, rhs: Frame<T>) -> Frame<T> {
                self.$method(&rhs)
            }
        }
        // Frame<T> $OpTrait Frame<T>
        impl<T> std::ops::$OpTrait<Frame<T>> for Frame<T>
        where
            T: Clone + PartialEq + std::ops::$OpTrait<Output = T>,
        {
            type Output = Frame<T>;
            fn $method(self, rhs: Frame<T>) -> Frame<T> {
                (&self).$method(&rhs)
            }
        }
    };
}

impl_elementwise_frame_op!(Add, add);
impl_elementwise_frame_op!(Sub, sub);
impl_elementwise_frame_op!(Mul, mul);
impl_elementwise_frame_op!(Div, div);

/* ---------- Boolean Bitwise Operations ---------- */
/// Generates implementations for element-wise bitwise operations (`&`, `|`, `^`) on `Frame<bool>`.
/// Panics if column labels or row indices differ between operands.
macro_rules! impl_bitwise_frame_op {
    ($OpTrait:ident, $method:ident) => {
        // &Frame<bool> $OpTrait &Frame<bool>
        impl<'a, 'b> std::ops::$OpTrait<&'b Frame<bool>> for &'a Frame<bool> {
            type Output = Frame<bool>;
            fn $method(self, rhs: &'b Frame<bool>) -> Frame<bool> {
                if self.column_names != rhs.column_names {
                    panic!(
                        "Bitwise {}: column names do not match. Left: {:?}, Right: {:?}",
                        stringify!($method),
                        self.column_names,
                        rhs.column_names
                    );
                }
                if self.index != rhs.index {
                    panic!(
                        "Bitwise {}: row indices do not match. Left: {:?}, Right: {:?}",
                        stringify!($method),
                        self.index,
                        rhs.index
                    );
                }
                let result_matrix = (&self.matrix).$method(&rhs.matrix);
                let new_index = match self.index {
                    RowIndex::Range(_) => None,
                    _ => Some(self.index.clone()),
                };
                Frame::new(result_matrix, self.column_names.clone(), new_index)
            }
        }
        // Frame<bool> $OpTrait &Frame<bool>
        impl<'b> std::ops::$OpTrait<&'b Frame<bool>> for Frame<bool> {
            type Output = Frame<bool>;
            fn $method(self, rhs: &'b Frame<bool>) -> Frame<bool> {
                (&self).$method(rhs)
            }
        }
        // &Frame<bool> $OpTrait Frame<bool>
        impl<'a> std::ops::$OpTrait<Frame<bool>> for &'a Frame<bool> {
            type Output = Frame<bool>;
            fn $method(self, rhs: Frame<bool>) -> Frame<bool> {
                self.$method(&rhs)
            }
        }
        // Frame<bool> $OpTrait Frame<bool>
        impl std::ops::$OpTrait<Frame<bool>> for Frame<bool> {
            type Output = Frame<bool>;
            fn $method(self, rhs: Frame<bool>) -> Frame<bool> {
                (&self).$method(&rhs)
            }
        }
    };
}

impl_bitwise_frame_op!(BitAnd, bitand);
impl_bitwise_frame_op!(BitOr, bitor);
impl_bitwise_frame_op!(BitXor, bitxor);

/* ---------- Logical NOT ---------- */
/// Implements logical NOT (`!`) for `Frame<bool>`, consuming the frame.
impl Not for Frame<bool> {
    type Output = Frame<bool>;

    fn not(self) -> Frame<bool> {
        // Apply NOT to the underlying matrix
        let result_matrix = !self.matrix;

        // Determine index for the result
        let new_index = match self.index {
            RowIndex::Range(_) => None,
            _ => Some(self.index),
        };

        Frame::new(result_matrix, self.column_names, new_index)
    }
}

/// Implements logical NOT (`!`) for `&Frame<bool>`, borrowing the frame.
impl Not for &Frame<bool> {
    type Output = Frame<bool>;

    fn not(self) -> Frame<bool> {
        // Apply NOT to the underlying matrix
        let result_matrix = !&self.matrix;

        // Determine index for the result
        let new_index = match self.index {
            RowIndex::Range(_) => None,
            _ => Some(self.index.clone()),
        };

        Frame::new(result_matrix, self.column_names.clone(), new_index)
    }
}

//  --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    // Assume Matrix is available from crate::matrix or similar
    use crate::matrix::{BoolOps, Matrix};
    use chrono::NaiveDate;
    // HashMap needed for direct inspection in tests if required
    use std::collections::HashMap;
    // Use a fixed tolerance for float comparisons
    const FLOAT_TOLERANCE: f64 = 1e-9;

    // --- Test Helpers ---
    fn create_test_matrix_f64() -> Matrix<f64> {
        Matrix::from_cols(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]) // 3 rows, 2 cols
    }
    fn create_test_matrix_f64_alt() -> Matrix<f64> {
        Matrix::from_cols(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]) // 3 rows, 2 cols
    }
    fn create_test_matrix_bool() -> Matrix<bool> {
        Matrix::from_cols(vec![vec![true, false], vec![false, true]]) // 2 rows, 2 cols
    }
    fn create_test_matrix_bool_alt() -> Matrix<bool> {
        Matrix::from_cols(vec![vec![true, true], vec![false, false]]) // 2 rows, 2 cols
    }
    fn create_test_matrix_string() -> Matrix<String> {
        Matrix::from_cols(vec![
            vec!["r0c0".to_string(), "r1c0".to_string()], // Col 0
            vec!["r0c1".to_string(), "r1c1".to_string()], // Col 1
        ]) // 2 rows, 2 cols
    }
    fn d(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }
    fn create_test_frame_f64() -> Frame<f64> {
        Frame::new(create_test_matrix_f64(), vec!["A", "B"], None)
    }
    fn create_test_frame_f64_alt() -> Frame<f64> {
        Frame::new(create_test_matrix_f64_alt(), vec!["A", "B"], None)
    }
    fn create_test_frame_bool() -> Frame<bool> {
        Frame::new(create_test_matrix_bool(), vec!["P", "Q"], None)
    }
    fn create_test_frame_bool_alt() -> Frame<bool> {
        Frame::new(create_test_matrix_bool_alt(), vec!["P", "Q"], None)
    }
    fn create_test_frame_int() -> Frame<i32> {
        Frame::new(
            Matrix::from_cols(vec![vec![1, -2], vec![3, -4]]), // 2 rows, 2 cols
            vec!["X", "Y"],
            None,
        )
    }
    fn create_test_frame_int_alt() -> Frame<i32> {
        Frame::new(
            Matrix::from_cols(vec![vec![10, 20], vec![30, 40]]), // 2 rows, 2 cols
            vec!["X", "Y"],
            None,
        )
    }

    // --- Frame::new Tests ---
    #[test]
    fn frame_new_default_index() {
        let frame = create_test_frame_f64(); // 3 rows, 2 cols
        assert_eq!(frame.rows(), 3);
        assert_eq!(frame.cols(), 2);
        assert_eq!(frame.columns(), &["A", "B"]);
        assert_eq!(frame.index(), &RowIndex::Range(0..3));
        assert_eq!(frame.col_lookup.len(), 2);
        assert_eq!(frame.col_lookup["A"], 0);
        assert_eq!(frame.col_lookup["B"], 1);
        assert_eq!(frame.index_lookup, RowIndexLookup::None);
        assert_eq!(frame["A"], vec![1.0, 2.0, 3.0]);
        assert_eq!(frame["B"], vec![4.0, 5.0, 6.0]);
    }
    #[test]
    fn frame_new_int_index() {
        let matrix = create_test_matrix_f64(); // 3 rows
        let index_vec = vec![10, 20, 5];
        let index = RowIndex::Int(index_vec.clone());
        let frame = Frame::new(matrix, vec!["A", "B"], Some(index.clone()));
        assert_eq!(frame.index(), &index);
        assert!(matches!(frame.index_lookup, RowIndexLookup::Int(_)));
        if let RowIndexLookup::Int(lookup) = &frame.index_lookup {
            assert_eq!(lookup.len(), 3);
            assert_eq!(lookup[&10], 0); // value 10 -> physical row 0
            assert_eq!(lookup[&20], 1); // value 20 -> physical row 1
            assert_eq!(lookup[&5], 2); // value 5  -> physical row 2
        }
        assert_eq!(frame.get_row(10)["A"], 1.0); // Access by index value
        assert_eq!(frame.get_row(20)["A"], 2.0);
        assert_eq!(frame.get_row(5)["A"], 3.0);
    }
    #[test]
    fn frame_new_date_index() {
        let matrix = create_test_matrix_string(); // 2 rows
        let dates = vec![d(2024, 1, 10), d(2024, 1, 5)]; // Order preserved
        let index = RowIndex::Date(dates.clone());
        let frame = Frame::new(matrix, vec!["X", "Y"], Some(index.clone()));
        assert_eq!(frame.rows(), 2);
        assert_eq!(frame.cols(), 2);
        assert_eq!(frame.index(), &index);
        assert!(matches!(frame.index_lookup, RowIndexLookup::Date(_)));
        if let RowIndexLookup::Date(lookup) = &frame.index_lookup {
            assert_eq!(lookup.len(), 2);
            assert_eq!(lookup[&d(2024, 1, 10)], 0); // date -> physical row 0
            assert_eq!(lookup[&d(2024, 1, 5)], 1); // date -> physical row 1
        }
        assert_eq!(frame["X"], vec!["r0c0", "r1c0"]);
        assert_eq!(frame.get_row_date(d(2024, 1, 10))["X"], "r0c0");
        assert_eq!(frame.get_row_date(d(2024, 1, 5))["X"], "r1c0");
    }
    #[test]
    fn frame_new_one_by_one() {
        let matrix = Matrix::from_cols(vec![vec![100]]); // 1 row, 1 col
        let frame = Frame::new(matrix, vec!["Single"], None);
        assert_eq!(frame.rows(), 1);
        assert_eq!(frame.cols(), 1);
        assert_eq!(frame.columns(), &["Single"]);
        assert_eq!(frame.index(), &RowIndex::Range(0..1));
        assert_eq!(frame["Single"], vec![100]);
        assert_eq!(frame.get_row(0)[0], 100);
        assert_eq!(frame.get_row(0)["Single"], 100);
    }
    // Removed test frame_new_zero_rows_zero_cols as Matrix constructors prevent it

    // --- Frame::new Panic Tests ---
    #[test]
    #[should_panic(expected = "Frame::new: column name count mismatch (names: 1, matrix: 2)")]
    fn frame_new_panic_col_count() {
        let matrix = create_test_matrix_f64();
        Frame::new(matrix, vec!["A"], None);
    }
    #[test]
    #[should_panic(expected = "duplicate column label: A")]
    fn frame_new_panic_duplicate_col() {
        let matrix = create_test_matrix_f64();
        Frame::new(matrix, vec!["A", "A"], None);
    }
    #[test]
    #[should_panic(expected = "Int index length (2) mismatch matrix rows (3)")]
    fn frame_new_panic_index_len() {
        let matrix = create_test_matrix_f64(); // 3 rows
        let index = RowIndex::Int(vec![10, 20]); // Only 2 index values
        Frame::new(matrix, vec!["A", "B"], Some(index));
    }
    #[test]
    #[should_panic(expected = "Date index length (1) mismatch matrix rows (2)")]
    fn frame_new_panic_date_index_len() {
        let matrix = create_test_matrix_string(); // 2 rows
        let index = RowIndex::Date(vec![d(2024, 1, 1)]); // Only 1 index value
        Frame::new(matrix, vec!["X", "Y"], Some(index));
    }
    #[test]
    #[should_panic(expected = "duplicate Int index value: 10")]
    fn frame_new_panic_duplicate_int_index() {
        let matrix = create_test_matrix_f64(); // 3 rows
        let index = RowIndex::Int(vec![10, 20, 10]); // Duplicate 10
        Frame::new(matrix, vec!["A", "B"], Some(index));
    }
    #[test]
    #[should_panic(expected = "duplicate Date index value: 2024-01-10")]
    fn frame_new_panic_duplicate_date_index() {
        let matrix = create_test_matrix_string(); // 2 rows
        let index = RowIndex::Date(vec![d(2024, 1, 10), d(2024, 1, 10)]); // Duplicate date
        Frame::new(matrix, vec!["X", "Y"], Some(index));
    }
    #[test]
    #[should_panic(expected = "Frame::new: Range index length (4) mismatch matrix rows (3)")]
    fn frame_new_panic_invalid_explicit_range_index() {
        let matrix = create_test_matrix_f64(); // 3 rows
        let index = RowIndex::Range(0..4); // Range 0..4 but only 3 rows
        Frame::new(matrix, vec!["A", "B"], Some(index));
    }

    // --- RowIndex Method Tests ---
    #[test]
    fn test_row_index_methods() {
        let idx_int = RowIndex::Int(vec![10, 20, 5]);
        assert_eq!(idx_int.len(), 3);
        assert!(!idx_int.is_empty());
        let idx_date = RowIndex::Date(vec![d(2024, 1, 1), d(2024, 1, 2)]);
        assert_eq!(idx_date.len(), 2);
        assert!(!idx_date.is_empty());
        let idx_range = RowIndex::Range(0..5);
        assert_eq!(idx_range.len(), 5);
        assert!(!idx_range.is_empty());
        let idx_empty_int = RowIndex::Int(vec![]);
        assert_eq!(idx_empty_int.len(), 0);
        assert!(idx_empty_int.is_empty());
        let idx_empty_date = RowIndex::Date(vec![]);
        assert_eq!(idx_empty_date.len(), 0);
        assert!(idx_empty_date.is_empty());
        let idx_empty_range = RowIndex::Range(3..3);
        assert_eq!(idx_empty_range.len(), 0);
        assert!(idx_empty_range.is_empty());
        let idx_range_zero = RowIndex::Range(0..0);
        assert_eq!(idx_range_zero.len(), 0);
        assert!(idx_range_zero.is_empty());
    }

    // --- Frame Accessor Tests ---
    #[test]
    fn frame_column_access() {
        let mut frame = create_test_frame_f64(); // A=[1,2,3], B=[4,5,6]
        assert_eq!(frame.column("A"), &[1.0, 2.0, 3.0]);
        assert_eq!(frame["B"], vec![4.0, 5.0, 6.0]); // Index trait
        assert_eq!(frame.column_index("A"), Some(0));
        assert_eq!(frame.column_index("B"), Some(1));
        assert_eq!(frame.column_index("C"), None);

        // Mutation
        frame.column_mut("A")[1] = 2.5;
        assert_eq!(frame["A"], vec![1.0, 2.5, 3.0]);
        frame["B"][0] = 4.1; // IndexMut trait
        assert_eq!(frame["B"], vec![4.1, 5.0, 6.0]);
    }
    #[test]
    #[should_panic(expected = "unknown column label: 'C'")]
    fn frame_column_access_panic() {
        let frame = create_test_frame_f64();
        let _ = frame.column("C");
    }
    #[test]
    #[should_panic(expected = "unknown column label: 'C'")]
    fn frame_column_access_mut_panic() {
        let mut frame = create_test_frame_f64();
        let _ = frame.column_mut("C");
    }
    #[test]
    #[should_panic(expected = "unknown column label: 'C'")]
    fn frame_column_index_panic() {
        let frame = create_test_frame_f64();
        let _ = frame["C"]; // Panics when Index calls column internally
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'C'")]
    fn frame_column_index_mut_panic() {
        let mut frame = create_test_frame_f64();
        frame["C"][0] = 0.0; // Panics when IndexMut calls column_mut internally
    }

    #[test]
    fn frame_row_access_default_index() {
        let frame = create_test_frame_f64(); // Index 0..3
        let row1 = frame.get_row(1); // Get row for index value 1 (physical row 1)
        assert_eq!(row1.get("A"), &2.0);
        assert_eq!(row1.get_by_index(1), &5.0); // Access by physical column index
        assert_eq!(row1["A"], 2.0); // Index by name
        assert_eq!(row1[1], 5.0); // Index by physical column index
        assert_eq!(frame.get_row(0)["B"], 4.0); // Index value 0 -> physical row 0
    }
    #[test]
    fn frame_row_access_int_index() {
        let matrix = create_test_matrix_f64(); // 3 rows
        let index = RowIndex::Int(vec![100, 50, 200]);
        let frame = Frame::new(matrix, vec!["A", "B"], Some(index));
        let row50 = frame.get_row(50); // Access by index value 50 (physical row 1)
        assert_eq!(row50["A"], 2.0);
        assert_eq!(row50[1], 5.0); // Column B (physical index 1)
        assert_eq!(frame.get_row(200)["A"], 3.0); // Index value 200 -> physical row 2
    }
    #[test]
    fn frame_row_access_date_index() {
        let matrix = create_test_matrix_string(); // 2 rows
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]);
        let frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        let row_may10 = frame.get_row_date(d(2023, 5, 10)); // Access by date (physical row 1)
        assert_eq!(row_may10["X"], "r1c0");
        assert_eq!(row_may10[1], "r1c1"); // Column Y (physical index 1)
        assert_eq!(frame.get_row_date(d(2023, 5, 1))["Y"], "r0c1"); // Date -> physical row 0
    }

    #[test]
    #[should_panic(expected = "integer key 99 not found in Int index")]
    fn frame_row_access_int_index_panic_not_found() {
        let matrix = create_test_matrix_f64();
        let index = RowIndex::Int(vec![100, 50, 200]);
        let frame = Frame::new(matrix, vec!["A", "B"], Some(index));
        frame.get_row(99); // 99 is not in the index values
    }
    #[test]
    #[should_panic(expected = "integer key 99 not found in Int index")]
    fn frame_row_access_int_index_mut_panic_not_found() {
        let matrix = create_test_matrix_f64();
        let index = RowIndex::Int(vec![100, 50, 200]);
        let mut frame = Frame::new(matrix, vec!["A", "B"], Some(index));
        frame.get_row_mut(99); // 99 is not in the index values
    }
    #[test]
    #[should_panic(expected = "integer key 3 out of bounds for Range index 0..3")]
    fn frame_row_access_default_index_panic_out_of_bounds() {
        let frame = create_test_frame_f64(); // Index 0..3
        frame.get_row(3); // 3 is not in the range [0, 3)
    }
    #[test]
    #[should_panic(expected = "integer key 3 out of bounds for Range index 0..3")]
    fn frame_row_access_default_index_mut_panic_out_of_bounds() {
        let mut frame = create_test_frame_f64(); // Index 0..3
        frame.get_row_mut(3); // 3 is not in the range [0, 3)
    }
    #[test]
    #[should_panic(expected = "date key 2023-05-02 not found in Date index")]
    fn frame_row_access_date_index_panic_not_found() {
        let matrix = create_test_matrix_string();
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]);
        let frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        frame.get_row_date(d(2023, 5, 2)); // Date not in index
    }
    #[test]
    #[should_panic(expected = "date key 2023-05-02 not found in Date index")]
    fn frame_row_access_date_index_mut_panic_not_found() {
        let matrix = create_test_matrix_string();
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]);
        let mut frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        frame.get_row_date_mut(d(2023, 5, 2)); // Date not in index
    }
    #[test]
    #[should_panic(expected = "incompatible key type usize for Date index")]
    fn frame_row_access_type_mismatch_panic_usize_on_date() {
        let matrix = create_test_matrix_string();
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]);
        let frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        frame.get_row(0); // Using usize key with Date index
    }
    #[test]
    #[should_panic(expected = "incompatible key type usize for Date index")]
    fn frame_row_access_type_mismatch_mut_panic_usize_on_date() {
        let matrix = create_test_matrix_string();
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]);
        let mut frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        frame.get_row_mut(0); // Using usize key with Date index
    }
    #[test]
    #[should_panic(expected = "incompatible key type NaiveDate for Int or Range index")]
    fn frame_row_access_type_mismatch_panic_date_on_int() {
        let matrix = create_test_matrix_f64();
        let index = RowIndex::Int(vec![100, 50, 200]);
        let frame = Frame::new(matrix, vec!["A", "B"], Some(index));
        frame.get_row_date(d(2023, 5, 1)); // Using Date key with Int index
    }
    #[test]
    #[should_panic(expected = "incompatible key type NaiveDate for Int or Range index")]
    fn frame_row_access_type_mismatch_mut_panic_date_on_int() {
        let matrix = create_test_matrix_f64();
        let index = RowIndex::Int(vec![100, 50, 200]);
        let mut frame = Frame::new(matrix, vec!["A", "B"], Some(index));
        frame.get_row_date_mut(d(2023, 5, 1)); // Using Date key with Int index
    }
    #[test]
    #[should_panic(expected = "incompatible key type NaiveDate for Int or Range index")]
    fn frame_row_access_type_mismatch_panic_date_on_range() {
        let frame = create_test_frame_f64(); // Range index
        frame.get_row_date(d(2023, 5, 1)); // Using Date key with Range index
    }
    #[test]
    #[should_panic(expected = "incompatible key type NaiveDate for Int or Range index")]
    fn frame_row_access_type_mismatch_mut_panic_date_on_range() {
        let mut frame = create_test_frame_f64(); // Range index
        frame.get_row_date_mut(d(2023, 5, 1)); // Using Date key with Range index
    }
    #[test]
    #[should_panic(expected = "inconsistent internal index state")]
    fn frame_row_access_inconsistent_state_int_none() {
        // Manually create inconsistent state (Int index, None lookup)
        let frame = Frame::<i32> {
            matrix: Matrix::from_cols(vec![vec![1]]),
            column_names: vec!["A".to_string()],
            col_lookup: HashMap::from([("A".to_string(), 0)]),
            index: RowIndex::Int(vec![10]),
            index_lookup: RowIndexLookup::None, // Inconsistent
        };
        frame.get_row(10); // Should panic due to inconsistency
    }
    #[test]
    #[should_panic(expected = "inconsistent internal index state")]
    fn frame_row_access_inconsistent_state_date_none() {
        // Manually create inconsistent state (Date index, None lookup)
        let frame = Frame::<i32> {
            matrix: Matrix::from_cols(vec![vec![1]]),
            column_names: vec!["A".to_string()],
            col_lookup: HashMap::from([("A".to_string(), 0)]),
            index: RowIndex::Date(vec![d(2024, 1, 1)]),
            index_lookup: RowIndexLookup::None, // Inconsistent
        };
        frame.get_row_date(d(2024, 1, 1)); // Should panic due to inconsistency
    }
    #[test]
    #[should_panic(expected = "inconsistent internal index state")]
    fn frame_row_access_inconsistent_state_range_int() {
        // Manually create inconsistent state (Range index, Int lookup)
        let frame = Frame::<i32> {
            matrix: Matrix::from_cols(vec![vec![1]]),
            column_names: vec!["A".to_string()],
            col_lookup: HashMap::from([("A".to_string(), 0)]),
            index: RowIndex::Range(0..1),
            index_lookup: RowIndexLookup::Int(HashMap::new()), // Inconsistent
        };
        frame.get_row(0); // Should panic due to inconsistency
    }

    // --- Frame Row Mutation Tests ---
    #[test]
    fn frame_row_mutate_default_index() {
        let mut frame = create_test_frame_f64(); // Index 0..3, A=[1,2,3], B=[4,5,6]
                                                 // Mutate using set("col_name", value)
        frame.get_row_mut(1).set("A", 2.9); // Mutate row index 1, col A
        assert_eq!(frame["A"], vec![1.0, 2.9, 3.0]);
        // Mutate using IndexMut by physical column index
        frame.get_row_mut(0)[1] = 4.9; // Mutate row index 0, col B (index 1)
        assert_eq!(frame["B"], vec![4.9, 5.0, 6.0]);
        // Mutate using IndexMut by column name
        frame.get_row_mut(2)["A"] = 3.9; // Mutate row index 2, col A
        assert_eq!(frame["A"], vec![1.0, 2.9, 3.9]);
    }
    #[test]
    fn frame_row_mutate_date_index() {
        let matrix = create_test_matrix_string(); // r0=["r0c0","r0c1"], r1=["r1c0","r1c1"]
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]); // r0=May1, r1=May10
        let mut frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        let key_may10 = d(2023, 5, 10);
        let key_may1 = d(2023, 5, 1);

        // Mutate using set_by_index(col_idx, value)
        frame
            .get_row_date_mut(key_may10) // Get row for May 10 (physical row 1)
            .set_by_index(0, "r1c0_mod".to_string()); // Set col X (index 0)
        assert_eq!(frame["X"], vec!["r0c0", "r1c0_mod"]);

        // Mutate using IndexMut by column name
        frame.get_row_date_mut(key_may1)["Y"] = "r0c1_mod".to_string(); // Row May 1, col Y
        assert_eq!(frame["Y"], vec!["r0c1_mod", "r1c1"]);

        // Mutate using IndexMut by physical column index
        frame.get_row_date_mut(key_may10)[1] = "r1c1_mod2".to_string(); // Row May 10, col Y (index 1)
        assert_eq!(frame["Y"], vec!["r0c1_mod", "r1c1_mod2"]);
    }

    // --- FrameRowView / FrameRowViewMut Indexing Tests ---
    #[test]
    fn test_row_view_mut_readonly_index() {
        // Test that read-only indexing still works on a mutable view
        let mut frame = create_test_frame_f64(); // A=[1,2,3], B=[4,5,6]
        let row_mut = frame.get_row_mut(1); // Get mutable view of row index 1
        assert_eq!(row_mut["A"], 2.0); // Read via Index<&str>
        assert_eq!(row_mut[1], 5.0); // Read via Index<usize> (col B)
    }
    #[test]
    #[should_panic(expected = "column index 2 out of bounds")] // Expect more specific message now
    fn test_row_view_index_panic() {
        let frame = create_test_frame_f64(); // 2 cols (0, 1)
        let row_view = frame.get_row(0);
        let _ = row_view[2]; // Access column index 2 (out of bounds)
    }
    #[test]
    #[should_panic(expected = "unknown column 'C'")]
    fn test_row_view_name_panic() {
        let frame = create_test_frame_f64();
        let row_view = frame.get_row(0);
        let _ = row_view["C"]; // Access non-existent column Z
    }
    #[test]
    #[should_panic(expected = "column index 3 out of bounds")] // Check specific message
    fn test_row_view_get_by_index_panic() {
        let frame = create_test_frame_f64(); // 2 cols (0, 1)
        let row_view = frame.get_row(0);
        let _ = row_view.get_by_index(3);
    }
    #[test]
    #[should_panic(expected = "column index 2 out of bounds")] // Expect more specific message now
    fn test_row_view_mut_index_panic() {
        let mut frame = create_test_frame_f64(); // 2 cols (0, 1)
        let mut row_view_mut = frame.get_row_mut(0);
        row_view_mut[2] = 0.0; // Access column index 2 (out of bounds)
    }
    #[test]
    #[should_panic(expected = "unknown column 'C'")]
    fn test_row_view_mut_name_panic() {
        let mut frame = create_test_frame_f64();
        let mut row_view_mut = frame.get_row_mut(0);
        row_view_mut["C"] = 0.0; // Access non-existent column name
    }
    #[test]
    #[should_panic(expected = "column index 3 out of bounds")] // Check specific message
    fn test_row_view_mut_get_by_index_mut_panic() {
        let mut frame = create_test_frame_f64(); // 2 cols (0, 1)
        let mut row_view_mut = frame.get_row_mut(0);
        let _ = row_view_mut.get_by_index_mut(3);
    }
    #[test]
    #[should_panic(expected = "column index 3 out of bounds")] // Check specific message
    fn test_row_view_mut_set_by_index_panic() {
        let mut frame = create_test_frame_f64(); // 2 cols (0, 1)
        let mut row_view_mut = frame.get_row_mut(0);
        row_view_mut.set_by_index(3, 0.0);
    }
    #[test]
    #[should_panic(expected = "unknown column 'C'")] // Panic from view set -> get_mut
    fn test_row_view_mut_set_panic() {
        let mut frame = create_test_frame_f64();
        let mut row_view_mut = frame.get_row_mut(0);
        row_view_mut.set("C", 0.0); // Access non-existent column name
    }

    // --- Frame Column Manipulation & Sorting Tests ---
    #[test]
    fn frame_column_manipulation_and_sort() {
        // Initial: C=[1,2], A=[3,4] (names out of alphabetical order)
        let mut frame = Frame::new(
            Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]), // 2 rows, 2 cols
            vec!["C", "A"],
            None,
        );
        assert_eq!(frame.columns(), &["C", "A"]);
        assert_eq!(frame["C"], vec![1, 2]);
        assert_eq!(frame["A"], vec![3, 4]);
        assert_eq!(frame.column_index("C"), Some(0));
        assert_eq!(frame.column_index("A"), Some(1));
        assert_eq!(frame.col_lookup.len(), 2);

        // Add B=[5,6]: C=[1,2], A=[3,4], B=[5,6]
        frame.add_column("B", vec![5, 6]);
        assert_eq!(frame.columns(), &["C", "A", "B"]);
        assert_eq!(frame["B"], vec![5, 6]);
        assert_eq!(frame.column_index("C"), Some(0));
        assert_eq!(frame.column_index("A"), Some(1));
        assert_eq!(frame.column_index("B"), Some(2)); // B added at the end
        assert_eq!(frame.col_lookup.len(), 3);

        // Rename C -> X: X=[1,2], A=[3,4], B=[5,6]
        frame.rename("C", "X");
        assert_eq!(frame.columns(), &["X", "A", "B"]); // Name 'C' at index 0 replaced by 'X'
        assert_eq!(frame["X"], vec![1, 2]); // Data remains
        assert_eq!(frame.column_index("X"), Some(0));
        assert_eq!(frame.column_index("A"), Some(1));
        assert_eq!(frame.column_index("B"), Some(2));
        assert!(frame.column_index("C").is_none()); // Old name gone
        assert_eq!(frame.col_lookup.len(), 3);

        // Delete A: X=[1,2], B=[5,6]
        let deleted_a = frame.delete_column("A");
        assert_eq!(deleted_a, vec![3, 4]);
        // Deleting col A (at physical index 1) shifts B left
        assert_eq!(frame.columns(), &["X", "B"]); // Remaining columns in physical order
        assert_eq!(frame.rows(), 2);
        assert_eq!(frame.cols(), 2);
        assert_eq!(frame["X"], vec![1, 2]); // X data unchanged
        assert_eq!(frame["B"], vec![5, 6]); // B data unchanged
                                            // Check internal state after delete + rebuild_col_lookup
        assert_eq!(frame.column_index("X"), Some(0)); // X is now physical col 0
        assert_eq!(frame.column_index("B"), Some(1)); // B is now physical col 1
        assert!(frame.column_index("A").is_none());
        assert_eq!(frame.col_lookup.len(), 2);

        // Sort Columns [X, B] -> [B, X]
        frame.sort_columns();
        assert_eq!(frame.columns(), &["B", "X"]); // Alphabetical order of names
                                                  // Verify data remained with the correct logical column after sort
        assert_eq!(frame["B"], vec![5, 6], "Data in B after sort"); // B should still have [5, 6]
        assert_eq!(frame["X"], vec![1, 2], "Data in X after sort"); // X should still have [1, 2]
                                                                    // Verify internal lookup map is correct after sort
        assert_eq!(frame.column_index("B"), Some(0), "Index of B after sort"); // B is now physical col 0
        assert_eq!(frame.column_index("X"), Some(1), "Index of X after sort"); // X is now physical col 1
        assert_eq!(frame.col_lookup.len(), 2);
        assert_eq!(*frame.col_lookup.get("B").unwrap(), 0);
        assert_eq!(*frame.col_lookup.get("X").unwrap(), 1);
    }

    // Tests specific to the old public swap_columns API are removed as it's now internal.
    // Test internal swap via sort_columns edge cases.
    #[test]
    fn test_sort_columns_already_sorted() {
        let mut frame = create_test_frame_f64(); // A, B (already sorted)
        let original_frame = frame.clone();
        frame.sort_columns();
        assert_eq!(frame.columns(), &["A", "B"]);
        assert_eq!(frame["A"], original_frame["A"]);
        assert_eq!(frame["B"], original_frame["B"]);
        assert_eq!(frame.col_lookup, original_frame.col_lookup);
    }
    #[test]
    fn test_sort_columns_reverse_sorted() {
        let mut frame = Frame::new(
            Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]),
            vec!["Z", "A"], // Z, A (reverse sorted)
            None,
        );
        frame.sort_columns();
        assert_eq!(frame.columns(), &["A", "Z"]);
        assert_eq!(frame["A"], vec![3, 4]); // A keeps its original data
        assert_eq!(frame["Z"], vec![1, 2]); // Z keeps its original data
        assert_eq!(frame.column_index("A"), Some(0));
        assert_eq!(frame.column_index("Z"), Some(1));
    }

    #[test]
    #[should_panic(expected = "new column name 'B' already exists")]
    fn test_rename_to_existing() {
        let mut frame = create_test_frame_f64(); // Has cols "A", "B"
        frame.rename("A", "B"); // Try renaming A to B (which exists)
    }
    #[test]
    #[should_panic(expected = "new name 'A' cannot be the same as the old name")]
    fn test_rename_to_self() {
        let mut frame = create_test_frame_f64();
        frame.rename("A", "A");
    }
    #[test]
    #[should_panic(expected = "unknown column label: 'Z'")]
    fn test_rename_panic_unknown() {
        let mut frame = create_test_frame_f64();
        frame.rename("Z", "Y"); // Try renaming non-existent column Z
    }

    #[test]
    #[should_panic(expected = "duplicate column label: A")]
    fn test_add_column_panic_duplicate() {
        let mut frame = create_test_frame_f64(); // Has col "A"
        frame.add_column("A", vec![0.0, 0.0, 0.0]); // Try adding "A" again
    }
    #[test]
    #[should_panic(expected = "column length mismatch")] // Panic comes from Matrix::add_column
    fn test_add_column_panic_len_mismatch() {
        let mut frame = create_test_frame_f64(); // Expects len 3
        frame.add_column("C", vec![0.0, 0.0]); // Provide len 2
    }
    // Removed tests for adding columns to 0-row frames as Matrix constructors prevent 0-row frames.

    #[test]
    fn test_delete_last_column_range_index() {
        // Start with a 1-column frame (since Matrix requires cols >= 1)
        let matrix = Matrix::from_cols(vec![vec![1, 2]]); // 2 rows, 1 col
        let mut frame = Frame::new(matrix, vec!["Single"], None); // Range index 0..2
        assert_eq!(frame.cols(), 1);
        assert_eq!(frame.rows(), 2);

        let deleted_data = frame.delete_column("Single");
        assert_eq!(deleted_data, vec![1, 2]);

        // Assuming Matrix allows 0 columns after deletion, but rows remain
        assert_eq!(frame.cols(), 0);
        assert!(frame.columns().is_empty());
        assert!(frame.col_lookup.is_empty());
        assert_eq!(frame.rows(), 2); // Rows remain
        assert_eq!(frame.index(), &RowIndex::Range(0..2)); // Range index should still reflect rows
    }
    #[test]
    fn test_delete_last_column_int_index() {
        let matrix = Matrix::from_cols(vec![vec![1, 2]]); // 2 rows, 1 col
        let index = RowIndex::Int(vec![10, 20]);
        let mut frame = Frame::new(matrix, vec!["Single"], Some(index.clone()));
        assert_eq!(frame.cols(), 1);
        assert_eq!(frame.rows(), 2);

        let deleted_data = frame.delete_column("Single");
        assert_eq!(deleted_data, vec![1, 2]);

        assert_eq!(frame.cols(), 0);
        assert!(frame.columns().is_empty());
        assert!(frame.col_lookup.is_empty());
        assert_eq!(frame.rows(), 2);
        assert_eq!(frame.index(), &index); // Int index remains unchanged
    }
    #[test]
    #[should_panic(expected = "unknown column label: 'Z'")]
    fn test_delete_column_panic_unknown() {
        let mut frame = create_test_frame_f64();
        frame.delete_column("Z"); // Try deleting non-existent column Z
    }

    #[test]
    fn test_sort_columns_empty_and_single() {
        // Test sorting an empty frame (0 cols) - Create via delete
        let mut frame0 = Frame::new(Matrix::from_cols(vec![vec![1]]), vec!["A"], None);
        frame0.delete_column("A");
        assert_eq!(frame0.cols(), 0);
        assert_eq!(frame0.rows(), 1); // Row remains
        let frame0_clone = frame0.clone();
        frame0.sort_columns(); // Should be a no-op
        assert_eq!(frame0, frame0_clone);
        assert_eq!(frame0.columns(), &[] as &[String]); // Ensure columns are empty

        // Test sorting a frame with a single column
        let mut frame1 = Frame::new(Matrix::from_cols(vec![vec![1.0]]), vec!["Z"], None); // 1x1
        assert_eq!(frame1.cols(), 1);
        let frame1_clone = frame1.clone();
        frame1.sort_columns(); // Should be a no-op
        assert_eq!(frame1, frame1_clone);
        assert_eq!(frame1.columns(), &["Z"]);
    }

    // --- Element-wise Arithmetic Ops Tests ---
    #[test]
    fn test_frame_arithmetic_ops_f64() {
        let f1 = create_test_frame_f64(); // A=[1,2,3], B=[4,5,6]
        let f2 = create_test_frame_f64_alt(); // A=[0.1,0.2,0.3], B=[0.4,0.5,0.6]

        // Addition
        let f_add = &f1 + &f2;
        assert_eq!(f_add.columns(), f1.columns());
        assert_eq!(f_add.index(), f1.index());
        assert!((f_add["A"][0] - 1.1).abs() < FLOAT_TOLERANCE);
        assert!((f_add["A"][1] - 2.2).abs() < FLOAT_TOLERANCE);
        assert!((f_add["A"][2] - 3.3).abs() < FLOAT_TOLERANCE);
        assert!((f_add["B"][0] - 4.4).abs() < FLOAT_TOLERANCE);
        assert!((f_add["B"][1] - 5.5).abs() < FLOAT_TOLERANCE);
        assert!((f_add["B"][2] - 6.6).abs() < FLOAT_TOLERANCE);

        // Subtraction
        let f_sub = &f1 - &f2;
        assert_eq!(f_sub.columns(), f1.columns());
        assert_eq!(f_sub.index(), f1.index());
        assert!((f_sub["A"][0] - 0.9).abs() < FLOAT_TOLERANCE);
        assert!((f_sub["A"][1] - 1.8).abs() < FLOAT_TOLERANCE);
        assert!((f_sub["A"][2] - 2.7).abs() < FLOAT_TOLERANCE);
        assert!((f_sub["B"][0] - 3.6).abs() < FLOAT_TOLERANCE);
        assert!((f_sub["B"][1] - 4.5).abs() < FLOAT_TOLERANCE);
        assert!((f_sub["B"][2] - 5.4).abs() < FLOAT_TOLERANCE);

        // Multiplication
        let f_mul = &f1 * &f2;
        assert_eq!(f_mul.columns(), f1.columns());
        assert_eq!(f_mul.index(), f1.index());
        assert!((f_mul["A"][0] - 0.1).abs() < FLOAT_TOLERANCE); // 1.0 * 0.1
        assert!((f_mul["A"][1] - 0.4).abs() < FLOAT_TOLERANCE); // 2.0 * 0.2
        assert!((f_mul["A"][2] - 0.9).abs() < FLOAT_TOLERANCE); // 3.0 * 0.3
        assert!((f_mul["B"][0] - 1.6).abs() < FLOAT_TOLERANCE); // 4.0 * 0.4
        assert!((f_mul["B"][1] - 2.5).abs() < FLOAT_TOLERANCE); // 5.0 * 0.5
        assert!(
            (f_mul["B"][2] - 3.6).abs() < FLOAT_TOLERANCE,
            "Check B[2] multiplication"
        ); // 6.0 * 0.6

        // Division
        let f_div = &f1 / &f2;
        assert_eq!(f_div.columns(), f1.columns());
        assert_eq!(f_div.index(), f1.index());
        assert!((f_div["A"][0] - 10.0).abs() < FLOAT_TOLERANCE); // 1.0 / 0.1
        assert!((f_div["A"][1] - 10.0).abs() < FLOAT_TOLERANCE); // 2.0 / 0.2
        assert!((f_div["A"][2] - 10.0).abs() < FLOAT_TOLERANCE); // 3.0 / 0.3
        assert!((f_div["B"][0] - 10.0).abs() < FLOAT_TOLERANCE); // 4.0 / 0.4
        assert!((f_div["B"][1] - 10.0).abs() < FLOAT_TOLERANCE); // 5.0 / 0.5
        assert!((f_div["B"][2] - 10.0).abs() < FLOAT_TOLERANCE); // 6.0 / 0.6
    }

    #[test]
    fn test_frame_arithmetic_ops_int() {
        let frame1 = create_test_frame_int(); // X=[1,-2], Y=[3,-4]
        let frame2 = create_test_frame_int_alt(); // X=[10,20], Y=[30,40]

        let frame_add = &frame1 + &frame2; // X=[11,18], Y=[33,36]
        assert_eq!(frame_add.columns(), frame1.columns());
        assert_eq!(frame_add.index(), frame1.index());
        assert_eq!(frame_add["X"], vec![11, 18]);
        assert_eq!(frame_add["Y"], vec![33, 36]);

        let frame_sub = &frame1 - &frame2; // X=[-9,-22], Y=[-27,-44]
        assert_eq!(frame_sub.columns(), frame1.columns());
        assert_eq!(frame_sub.index(), frame1.index());
        assert_eq!(frame_sub["X"], vec![-9, -22]);
        assert_eq!(frame_sub["Y"], vec![-27, -44]);

        let frame_mul = &frame1 * &frame2; // X=[10,-40], Y=[90,-160]
        assert_eq!(frame_mul.columns(), frame1.columns());
        assert_eq!(frame_mul.index(), frame1.index());
        assert_eq!(frame_mul["X"], vec![10, -40]);
        assert_eq!(frame_mul["Y"], vec![90, -160]);

        // Integer division (truncates)
        let frame_div = &frame2 / &frame1; // X=[10/1, 20/-2]=[10,-10], Y=[30/3, 40/-4]=[10,-10]
        assert_eq!(frame_div.columns(), frame1.columns());
        assert_eq!(frame_div.index(), frame1.index());
        assert_eq!(frame_div["X"], vec![10, -10]);
        assert_eq!(frame_div["Y"], vec![10, -10]);
    }

    #[test]
    fn tests_for_frame_arithmetic_ops() {
        let ops: Vec<(
            &str,
            fn(&Frame<f64>, &Frame<f64>) -> Frame<f64>,
            fn(&Frame<f64>, &Frame<f64>) -> Frame<f64>,
        )> = vec![
            ("addition", |a, b| a + b, |a, b| (&*a) + (&*b)),
            ("subtraction", |a, b| a - b, |a, b| (&*a) - (&*b)),
            ("multiplication", |a, b| a * b, |a, b| (&*a) * (&*b)),
            ("division", |a, b| a / b, |a, b| (&*a) / (&*b)),
        ];

        for (op_name, owned_op, ref_op) in ops {
            let f1 = create_test_frame_f64();
            let f2 = create_test_frame_f64_alt();
            let result_owned = owned_op(&f1, &f2);
            let expected = ref_op(&f1, &f2);

            assert_eq!(
                result_owned.columns(),
                f1.columns(),
                "Column mismatch for {}",
                op_name
            );
            assert_eq!(
                result_owned.index(),
                f1.index(),
                "Index mismatch for {}",
                op_name
            );

            let bool_mat = result_owned.matrix().eq_elem(expected.matrix().clone());
            assert!(bool_mat.all(), "Element-wise {} failed", op_name);
        }
    }

    // test not , and or on frame
    #[test]
    fn tests_for_frame_bool_ops() {
        let ops: Vec<(
            &str,
            fn(&Frame<bool>, &Frame<bool>) -> Frame<bool>,
            fn(&Frame<bool>, &Frame<bool>) -> Frame<bool>,
        )> = vec![
            ("and", |a, b| a & b, |a, b| (&*a) & (&*b)),
            ("or", |a, b| a | b, |a, b| (&*a) | (&*b)),
            ("xor", |a, b| a ^ b, |a, b| (&*a) ^ (&*b)),
        ];
        for (op_name, owned_op, ref_op) in ops {
            let f1 = create_test_frame_bool();
            let f2 = create_test_frame_bool_alt();
            let result_owned = owned_op(&f1, &f2);
            let expected = ref_op(&f1, &f2);

            assert_eq!(
                result_owned.columns(),
                f1.columns(),
                "Column mismatch for {}",
                op_name
            );
            assert_eq!(
                result_owned.index(),
                f1.index(),
                "Index mismatch for {}",
                op_name
            );

            let bool_mat = result_owned.matrix().eq_elem(expected.matrix().clone());
            assert!(bool_mat.all(), "Element-wise {} failed", op_name);
        }
    }

    #[test]
    fn test_frame_arithmetic_ops_date_index() {
        let dates = vec![d(2024, 1, 1), d(2024, 1, 2)];
        let index = Some(RowIndex::Date(dates));
        let m1 = Matrix::from_cols(vec![vec![1, 2], vec![10, 20]]);
        let m2 = Matrix::from_cols(vec![vec![3, 4], vec![30, 40]]);
        let f1 = Frame::new(m1, vec!["A", "B"], index.clone());
        let f2 = Frame::new(m2, vec!["A", "B"], index.clone());

        let f_add = &f1 + &f2;
        assert_eq!(f_add.columns(), f1.columns());
        assert_eq!(f_add.index(), f1.index());
        assert_eq!(f_add["A"], vec![4, 6]);
        assert_eq!(f_add["B"], vec![40, 60]);
        assert_eq!(f_add.get_row_date(d(2024, 1, 1))["A"], 4);
        assert_eq!(f_add.get_row_date(d(2024, 1, 2))["B"], 60);
    }

    #[test]
    fn test_bitwise_ops_and_not() {
        let frame_a = create_test_frame_bool(); // P=[T,F], Q=[F,T]
        let frame_b = create_test_frame_bool_alt(); // P=[T,T], Q=[F,F]

        // Bitwise AND
        let frame_and = &frame_a & &frame_b; // P=[T&T, F&T]->[T,F], Q=[F&F, T&F]->[F,F]
        assert_eq!(frame_and.columns(), frame_a.columns());
        assert_eq!(frame_and.index(), frame_a.index());
        assert_eq!(frame_and["P"], vec![true, false]);
        assert_eq!(frame_and["Q"], vec![false, false]);

        // Logical NOT (takes ownership)
        let frame_a_clone = frame_a.clone(); // Clone frame_a as Not consumes it
        let frame_not = !frame_a_clone;
        assert_eq!(frame_not.columns(), frame_a.columns());
        assert_eq!(frame_not.index(), frame_a.index());
        assert_eq!(frame_not["P"], vec![false, true]);
        assert_eq!(frame_not["Q"], vec![true, false]);

        // Check original frame_a is unchanged
        assert_eq!(frame_a["P"], vec![true, false]);
    }
    #[test]
    fn test_bitwise_ops_or_xor() {
        let frame_a = create_test_frame_bool(); // P=[T,F], Q=[F,T]
        let frame_b = create_test_frame_bool_alt(); // P=[T,T], Q=[F,F]

        // Bitwise OR
        let frame_or = &frame_a | &frame_b; // P=[T|T, F|T]->[T,T], Q=[F|F, T|F]->[F,T]
        assert_eq!(frame_or.columns(), frame_a.columns());
        assert_eq!(frame_or.index(), frame_a.index());
        assert_eq!(frame_or["P"], vec![true, true]);
        assert_eq!(frame_or["Q"], vec![false, true]);

        // Bitwise XOR
        let frame_xor = &frame_a ^ &frame_b; // P=[T^T, F^T]->[F,T], Q=[F^F, T^F]->[F,T]
        assert_eq!(frame_xor.columns(), frame_a.columns());
        assert_eq!(frame_xor.index(), frame_a.index());
        assert_eq!(frame_xor["P"], vec![false, true]);
        assert_eq!(frame_xor["Q"], vec![false, true]);
    }

    #[test]
    #[should_panic(expected = "row indices do not match")]
    fn frame_elementwise_ops_panic_index() {
        let frame1 = create_test_frame_f64(); // Range index 0..3
        let matrix2 = create_test_matrix_f64_alt(); // 3 rows
        let index2 = RowIndex::Int(vec![10, 20, 30]); // Different index type/values
        let frame2 = Frame::new(matrix2, vec!["A", "B"], Some(index2));
        let _ = &frame1 + &frame2; // Should panic due to index mismatch
    }
    #[test]
    #[should_panic(expected = "column names do not match")]
    fn frame_elementwise_ops_panic_cols() {
        let frame1 = create_test_frame_f64(); // Columns ["A", "B"]
        let matrix2 = create_test_matrix_f64_alt();
        let frame2 = Frame::new(matrix2, vec!["X", "Y"], None); // Different column names
        let _ = &frame1 + &frame2; // Should panic due to column name mismatch
    }
    #[test]
    #[should_panic(expected = "row indices do not match")]
    fn frame_bitwise_ops_panic_index() {
        let frame1 = create_test_frame_bool(); // Range index 0..2
        let matrix2 = create_test_matrix_bool_alt(); // 2 rows
        let index2 = RowIndex::Int(vec![10, 20]); // Different index
        let frame2 = Frame::new(matrix2, vec!["P", "Q"], Some(index2));
        let _ = &frame1 & &frame2;
    }
    #[test]
    #[should_panic(expected = "column names do not match")]
    fn frame_bitwise_ops_panic_cols() {
        let frame1 = create_test_frame_bool(); // Cols P, Q
        let matrix2 = create_test_matrix_bool_alt();
        let frame2 = Frame::new(matrix2, vec!["X", "Y"], None); // Cols X, Y
        let _ = &frame1 | &frame2;
    }

    // --- Debug Format Tests ---
    #[test]
    fn test_frame_debug_format() {
        let frame = create_test_frame_f64();
        let debug_str = format!("{:?}", frame);
        println!("Frame Debug: {}", debug_str); // Print for manual inspection
        assert!(debug_str.starts_with("Frame"));
        assert!(debug_str.contains("column_names: [\"A\", \"B\"]"));
        assert!(debug_str.contains("index: Range(0..3)"));
        assert!(debug_str.contains("matrix_dims: (3, 2)"));
        // Check for key-value pairs independently of order
        assert!(debug_str.contains("\"A\": 0"));
        assert!(debug_str.contains("\"B\": 1"));
        assert!(debug_str.contains("index_lookup: None"));
    }
    #[test]
    fn test_frame_debug_format_date_index() {
        let matrix = create_test_matrix_string();
        let index = RowIndex::Date(vec![d(2023, 5, 1), d(2023, 5, 10)]);
        let frame = Frame::new(matrix, vec!["X", "Y"], Some(index));
        let debug_str = format!("{:?}", frame);
        println!("Frame Debug Date Index: {}", debug_str);
        assert!(debug_str.starts_with("Frame"));
        assert!(debug_str.contains("column_names: [\"X\", \"Y\"]"));
        assert!(debug_str.contains("index: Date([2023-05-01, 2023-05-10])"));
        assert!(debug_str.contains("matrix_dims: (2, 2)"));
        assert!(debug_str.contains("\"X\": 0"));
        assert!(debug_str.contains("\"Y\": 1"));
        assert!(
            debug_str.contains("index_lookup: Date({2023-05-10: 1, 2023-05-01: 0})")
                || debug_str.contains("index_lookup: Date({2023-05-01: 0, 2023-05-10: 1})")
        );
    }
    #[test]
    fn test_row_view_debug_format() {
        let frame = create_test_frame_f64();
        let row_view = frame.get_row(1); // Physical row 1
        let debug_str = format!("{:?}", row_view);
        println!("RowView Debug: {}", debug_str); // Print for manual inspection
        assert!(debug_str.starts_with("FrameRowView"));
        assert!(debug_str.contains("physical_row_idx: 1"));
        assert!(debug_str.contains("columns: [\"A\", \"B\"]"));
        assert!(debug_str.contains("data: [2.0, 5.0]")); // Data from row 1
    }
    #[test]
    fn test_row_view_mut_debug_format() {
        let mut frame = create_test_frame_f64();
        let row_view_mut = frame.get_row_mut(0); // Physical row 0
        let debug_str = format!("{:?}", row_view_mut);
        println!("RowViewMut Debug: {}", debug_str); // Print for manual inspection
        assert!(debug_str.starts_with("FrameRowViewMut"));
        assert!(debug_str.contains("physical_row_idx: 0"));
        assert!(debug_str.contains("columns: [\"A\", \"B\"]"));
        // Debug format doesn't show data for Mut view to avoid borrow issues
    }

    // --- Miscellaneous Tests ---
    #[test]
    fn test_simple_accessors() {
        let matrix = create_test_matrix_f64(); // 3 rows, 2 cols
        let matrix_dims = (matrix.rows(), matrix.cols());
        let mut frame = Frame::new(matrix.clone(), vec!["A", "B"], None);
        assert_eq!(frame.rows(), 3);
        assert_eq!(frame.cols(), 2);
        assert_eq!(frame.columns(), &["A", "B"]);
        assert_eq!(frame.index(), &RowIndex::Range(0..3));
        // Check matrix accessors
        assert_eq!((frame.matrix().rows(), frame.matrix().cols()), matrix_dims);
        assert_eq!(frame.matrix().get(0, 0), matrix.get(0, 0)); // Get (0,0) via matrix ref
        assert_eq!(frame.matrix().get(0, 0), &1.0);
        // Check mutable matrix accessor (use with caution)
        assert_eq!(
            (frame.matrix_mut().rows(), frame.matrix_mut().cols()),
            matrix_dims
        );
        *frame.matrix_mut().get_mut(0, 0) = 99.0; // Mutate matrix directly
        assert_eq!(frame.matrix().get(0, 0), &99.0); // Verify change via matrix ref
        assert_eq!(frame["A"][0], 99.0); // Verify change via Frame column access
    }
}
