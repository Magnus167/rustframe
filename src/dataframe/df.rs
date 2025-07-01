use crate::frame::{Frame, RowIndex};
use crate::matrix::Matrix;
use std::collections::HashMap;
use std::fmt;

// Add these constants at the top of the file, perhaps after the use statements
const DEFAULT_DISPLAY_ROWS: usize = 5;
const DEFAULT_DISPLAY_COLS: usize = 10; // Display up to 10 columns by default

/// Represents a typed Frame that can hold multiple columns of a single type.
/// This will be the underlying storage for DataFrame.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedFrame {
    F64(Frame<f64>),
    I64(Frame<i64>),
    Bool(Frame<bool>),
    String(Frame<String>),
    // Add more types as needed
}

macro_rules! impl_typed_frame_common_methods {
    ($($method:ident $(($($arg:ident: $arg_ty:ty),*))? -> $ret_ty:ty),*) => {
        impl TypedFrame {
            $(
                pub fn $method(&self $(, $($arg: $arg_ty),*)?) -> $ret_ty {
                    match self {
                        TypedFrame::F64(f) => f.$method($($($arg),*)?),
                        TypedFrame::I64(f) => f.$method($($($arg),*)?),
                        TypedFrame::Bool(f) => f.$method($($($arg),*)?),
                        TypedFrame::String(f) => f.$method($($($arg),*)?),
                    }
                }
            )*
        }
    };
}

impl_typed_frame_common_methods! {
    rows -> usize,
    cols -> usize,
    columns -> &[String],
    index -> &RowIndex
}

macro_rules! impl_typed_frame_column_accessors {
    ($fn_name:ident, $ret_type:ident, $frame_method:ident) => {
        pub fn $fn_name(&self, name: &str) -> $ret_type<'_> {
            match self {
                TypedFrame::F64(f) => $ret_type::F64(f.$frame_method(name)),
                TypedFrame::I64(f) => $ret_type::I64(f.$frame_method(name)),
                TypedFrame::Bool(f) => $ret_type::Bool(f.$frame_method(name)),
                TypedFrame::String(f) => $ret_type::String(f.$frame_method(name)),
            }
        }
    };
    ($fn_name:ident, $ret_type:ident, $frame_method:ident, mut) => {
        pub fn $fn_name(&mut self, name: &str) -> $ret_type<'_> {
            match self {
                TypedFrame::F64(f) => $ret_type::F64(f.$frame_method(name)),
                TypedFrame::I64(f) => $ret_type::I64(f.$frame_method(name)),
                TypedFrame::Bool(f) => $ret_type::Bool(f.$frame_method(name)),
                TypedFrame::String(f) => $ret_type::String(f.$frame_method(name)),
            }
        }
    };
}

impl TypedFrame {
    impl_typed_frame_column_accessors!(column, DataFrameColumn, column);
    impl_typed_frame_column_accessors!(column_mut, DataFrameColumnMut, column_mut, mut);
}

/// Represents a view of a single column within a DataFrame.
/// It borrows data from an underlying TypedFrame.
#[derive(Debug, PartialEq)]
pub enum DataFrameColumn<'a> {
    F64(&'a [f64]),
    I64(&'a [i64]),
    Bool(&'a [bool]),
    String(&'a [String]),
}

impl<'a> DataFrameColumn<'a> {
    pub fn len(&self) -> usize {
        match self {
            DataFrameColumn::F64(s) => s.len(),
            DataFrameColumn::I64(s) => s.len(),
            DataFrameColumn::Bool(s) => s.len(),
            DataFrameColumn::String(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Add methods to get specific typed slices
    pub fn as_f64(&self) -> Option<&'a [f64]> {
        if let DataFrameColumn::F64(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<&'a [i64]> {
        if let DataFrameColumn::I64(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_bool(&self) -> Option<&'a [bool]> {
        if let DataFrameColumn::Bool(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_string(&self) -> Option<&'a [String]> {
        if let DataFrameColumn::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
}

/// Represents a mutable view of a single column within a DataFrame.
#[derive(Debug)]
pub enum DataFrameColumnMut<'a> {
    F64(&'a mut [f64]),
    I64(&'a mut [i64]),
    Bool(&'a mut [bool]),
    String(&'a mut [String]),
}

impl<'a> DataFrameColumnMut<'a> {
    pub fn len(&self) -> usize {
        match self {
            DataFrameColumnMut::F64(s) => s.len(),
            DataFrameColumnMut::I64(s) => s.len(),
            DataFrameColumnMut::Bool(s) => s.len(),
            DataFrameColumnMut::String(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Add methods to get specific typed mutable slices
    pub fn as_f64_mut(&mut self) -> Option<&mut [f64]> {
        if let DataFrameColumnMut::F64(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_i64_mut(&mut self) -> Option<&mut [i64]> {
        if let DataFrameColumnMut::I64(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_bool_mut(&mut self) -> Option<&mut [bool]> {
        if let DataFrameColumnMut::Bool(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_string_mut(&mut self) -> Option<&mut [String]> {
        if let DataFrameColumnMut::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
}

/// A DataFrame capable of holding multiple data types.
///
/// Internally, a DataFrame manages a collection of `TypedFrame` instances,
/// each holding data of a single, homogeneous type. The logical column
/// order is maintained separately from the physical storage.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    /// The logical order of column names.
    pub column_names: Vec<String>,
    /// A map from a unique ID to a TypedFrame (the underlying storage).
    pub subframes: HashMap<usize, TypedFrame>,
    /// A map from logical column name to its location: (subframe_id, column_name_in_subframe).
    pub column_locations: HashMap<String, (usize, String)>,
    /// The common row index for all columns in the DataFrame.
    pub index: RowIndex,
    /// The number of rows in the DataFrame.
    pub rows: usize,
    /// Counter for generating unique subframe IDs.
    pub next_subframe_id: usize,
}

impl DataFrame {
    /// Creates a new DataFrame from a vector of column data.
    ///
    /// Each inner `Vec<T>` represents a column, and the outer `Vec` contains
    /// these columns in the desired order. The `column_names` must match
    /// the number of columns provided.
    ///
    /// All columns must have the same number of rows.
    ///
    /// # Arguments
    /// * `columns` - A vector of `TypedFrame` instances, each representing a column.
    /// * `column_names` - A vector of strings, providing names for each column.
    /// * `index` - An optional `RowIndex` to be used for the DataFrame. If `None`, a default
    ///             `Range` index will be created.
    ///
    /// # Panics
    /// * If `column_names` length does not match the number of `columns`.
    /// * If columns have inconsistent row counts.
    /// * If column names are duplicated.
    /// * If the provided `index` length does not match the row count.
    pub fn new(
        columns: Vec<TypedFrame>, // Changed from DataFrameColumn to TypedFrame
        column_names: Vec<String>,
        index: Option<RowIndex>,
    ) -> Self {
        if columns.is_empty() {
            return Self {
                column_names: Vec::new(),
                subframes: HashMap::new(),
                column_locations: HashMap::new(),
                index: index.unwrap_or(RowIndex::Range(0..0)),
                rows: 0,
                next_subframe_id: 0,
            };
        }

        let num_rows = columns[0].rows();
        let common_index = index.unwrap_or(RowIndex::Range(0..num_rows));

        if common_index.len() != num_rows {
            panic!(
                "DataFrame::new: provided index length ({}) mismatch column rows ({})",
                common_index.len(),
                num_rows
            );
        }

        let mut subframes = HashMap::new();
        let mut column_locations = HashMap::new();
        let mut next_subframe_id = 0;

        // Process each provided TypedFrame
        for typed_frame in columns {
            if typed_frame.rows() != num_rows {
                panic!(
                    "DataFrame::new: TypedFrame has inconsistent row count ({} vs {})",
                    typed_frame.rows(),
                    num_rows
                );
            }
            if typed_frame.index() != &common_index {
                panic!("DataFrame::new: TypedFrame has inconsistent index with common index",);
            }

            let subframe_id = next_subframe_id;
            next_subframe_id += 1;

            for col_name_in_subframe in typed_frame.columns() {
                if column_locations.contains_key(col_name_in_subframe) {
                    panic!(
                        "DataFrame::new: duplicate column name: {}",
                        col_name_in_subframe
                    );
                }
                column_locations.insert(
                    col_name_in_subframe.clone(),
                    (subframe_id, col_name_in_subframe.clone()),
                );
            }
            subframes.insert(subframe_id, typed_frame);
        }

        // Ensure all column_names provided are actually present in the subframes
        for name in &column_names {
            if !column_locations.contains_key(name) {
                panic!(
                    "DataFrame::new: column name '{}' not found in provided TypedFrames",
                    name
                );
            }
        }

        Self {
            column_names, // This now represents the logical order of ALL columns from all subframes
            subframes,
            column_locations,
            index: common_index,
            rows: num_rows,
            next_subframe_id,
        }
    }

    /// Returns the number of rows in the DataFrame.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the DataFrame.
    #[inline]
    pub fn cols(&self) -> usize {
        self.column_names.len()
    }

    /// Returns a slice of the column names in their logical order.
    #[inline]
    pub fn columns(&self) -> &[String] {
        &self.column_names
    }

    /// Returns a reference to the DataFrame's row index.
    #[inline]
    pub fn index(&self) -> &RowIndex {
        &self.index
    }

    /// Returns an immutable view to a column by its name.
    /// Panics if the column is not found.
    pub fn column(&self, name: &str) -> DataFrameColumn<'_> {
        let (subframe_id, col_in_subframe_name) = self
            .column_locations
            .get(name)
            .unwrap_or_else(|| panic!("DataFrame::column: unknown column label: '{}'", name));

        let subframe = self.subframes.get(subframe_id).unwrap_or_else(|| {
            panic!(
                "DataFrame::column: internal error, subframe ID {} not found",
                subframe_id
            )
        });

        match subframe {
            TypedFrame::F64(f) => DataFrameColumn::F64(f.column(col_in_subframe_name)),
            TypedFrame::I64(f) => DataFrameColumn::I64(f.column(col_in_subframe_name)),
            TypedFrame::Bool(f) => DataFrameColumn::Bool(f.column(col_in_subframe_name)),
            TypedFrame::String(f) => DataFrameColumn::String(f.column(col_in_subframe_name)),
        }
    }

    /// Returns a mutable view to a column by its name.
    /// Panics if the column is not found.
    pub fn column_mut(&mut self, name: &str) -> DataFrameColumnMut<'_> {
        let (subframe_id, col_in_subframe_name) = self
            .column_locations
            .get(name)
            .unwrap_or_else(|| panic!("DataFrame::column_mut: unknown column label: '{}'", name));

        // We need to get a mutable reference to the subframe.
        // This requires a bit of care because we're borrowing `self.column_locations`
        // and then `self.subframes` mutably.
        // To avoid double-borrowing, we can get the subframe_id and col_in_subframe_name first,
        // then use them to access the subframe mutably.
        let subframe_id_copy = *subframe_id;
        let col_in_subframe_name_copy = col_in_subframe_name.clone();

        let subframe = self
            .subframes
            .get_mut(&subframe_id_copy)
            .unwrap_or_else(|| {
                panic!(
                    "DataFrame::column_mut: internal error, subframe ID {} not found",
                    subframe_id_copy
                )
            });

        match subframe {
            TypedFrame::F64(f) => DataFrameColumnMut::F64(f.column_mut(&col_in_subframe_name_copy)),
            TypedFrame::I64(f) => DataFrameColumnMut::I64(f.column_mut(&col_in_subframe_name_copy)),
            TypedFrame::Bool(f) => {
                DataFrameColumnMut::Bool(f.column_mut(&col_in_subframe_name_copy))
            }
            TypedFrame::String(f) => {
                DataFrameColumnMut::String(f.column_mut(&col_in_subframe_name_copy))
            }
        }
    }

    /// Adds a new column to the DataFrame.
    /// This involves either adding it to an existing TypedFrame if types match,
    /// or creating a new TypedFrame.
    /// Panics if a column with the same name already exists or if the new column's
    /// row count or index does not match the DataFrame's.
    /// Adds a new column to the DataFrame.
    /// The `column_data` must be a `TypedFrame` containing exactly one column,
    /// and its name must match the `name` parameter.
    pub fn add_column(&mut self, name: String, column_data: TypedFrame) {
        if self.column_locations.contains_key(&name) {
            panic!("DataFrame::add_column: duplicate column label: {}", name);
        }
        if column_data.rows() != self.rows {
            panic!(
                "DataFrame::add_column: new column '{}' has inconsistent row count ({} vs {})",
                name,
                column_data.rows(),
                self.rows
            );
        }
        if column_data.index() != &self.index {
            panic!(
                "DataFrame::add_column: new column '{}' has inconsistent index with DataFrame's index",
                name
            );
        }
        // Ensure the provided TypedFrame contains exactly one column, and its name matches `name`
        if column_data.cols() != 1 || column_data.columns()[0] != name {
            panic!(
                "DataFrame::add_column: provided TypedFrame must contain exactly one column named '{}'",
                name
            );
        }

        let subframe_id = self.next_subframe_id;
        self.next_subframe_id += 1;

        self.subframes.insert(subframe_id, column_data);
        self.column_locations
            .insert(name.clone(), (subframe_id, name.clone()));

        self.column_names.push(name);
    }

    /// Deletes a column by name and returns its data as a new single-column TypedFrame.
    /// Panics if the column name is not found.
    pub fn delete_column(&mut self, name: &str) -> TypedFrame {
        let (subframe_id, col_in_subframe_name) =
            self.column_locations.remove(name).unwrap_or_else(|| {
                panic!("DataFrame::delete_column: unknown column label: '{}'", name)
            });

        // Remove from logical column names
        if let Some(pos) = self.column_names.iter().position(|n| n == name) {
            self.column_names.remove(pos);
        }

        let subframe = self.subframes.get_mut(&subframe_id).unwrap_or_else(|| {
            panic!(
                "DataFrame::delete_column: internal error, subframe ID {} not found",
                subframe_id
            )
        });

        macro_rules! delete_column_from_typed_frame {
            ($frame_type:ident, $frame_variant:ident, $f:ident, $col_name:expr) => {{
                let data = $f.delete_column(&$col_name);
                TypedFrame::$frame_variant(Frame::new(
                    Matrix::from_cols(vec![data]),
                    vec![$col_name.clone()],
                    Some($f.index().clone()),
                ))
            }};
        }

        let deleted_data_frame = match subframe {
            TypedFrame::F64(f) => {
                delete_column_from_typed_frame!(f64, F64, f, col_in_subframe_name)
            }
            TypedFrame::I64(f) => {
                delete_column_from_typed_frame!(i64, I64, f, col_in_subframe_name)
            }
            TypedFrame::Bool(f) => {
                delete_column_from_typed_frame!(bool, Bool, f, col_in_subframe_name)
            }
            TypedFrame::String(f) => {
                delete_column_from_typed_frame!(String, String, f, col_in_subframe_name)
            }
        };

        // If the subframe becomes empty after deletion, remove it from the map
        if subframe.cols() == 0 {
            self.subframes.remove(&subframe_id);
        }

        deleted_data_frame
    }

    /// Renames an existing column.
    pub fn rename_column(&mut self, old_name: &str, new_name: String) {
        if old_name == new_name {
            return; // No change needed
        }
        if self.column_locations.contains_key(&new_name) {
            panic!(
                "DataFrame::rename_column: new column name '{}' already exists",
                new_name
            );
        }

        let (subframe_id, col_in_subframe_name) =
            self.column_locations.remove(old_name).unwrap_or_else(|| {
                panic!(
                    "DataFrame::rename_column: unknown column label: '{}'",
                    old_name
                )
            });

        // Update the column_locations map
        self.column_locations
            .insert(new_name.clone(), (subframe_id, new_name.clone()));

        // Update the logical column_names vector
        if let Some(pos) = self.column_names.iter().position(|n| n == old_name) {
            self.column_names[pos] = new_name.clone();
        }

        // Rename the column in the underlying TypedFrame
        let subframe = self.subframes.get_mut(&subframe_id).unwrap_or_else(|| {
            panic!(
                "DataFrame::rename_column: internal error, subframe ID {} not found",
                subframe_id
            )
        });

        match subframe {
            TypedFrame::F64(f) => f.rename(&col_in_subframe_name, new_name),
            TypedFrame::I64(f) => f.rename(&col_in_subframe_name, new_name),
            TypedFrame::Bool(f) => f.rename(&col_in_subframe_name, new_name),
            TypedFrame::String(f) => f.rename(&col_in_subframe_name, new_name),
        }
    }

    /// Sorts columns alphabetically by name.
    pub fn sort_columns(&mut self) {
        self.column_names.sort();
    }

    /// Returns a new DataFrame containing the first `n` rows.
    /// If `n` is greater than the number of rows in the DataFrame,
    /// the entire DataFrame is returned.
    pub fn head_n(&self, n: usize) -> Self {
        let num_rows_to_take = n.min(self.rows());

        if num_rows_to_take == 0 {
            return DataFrame::new(vec![], vec![], Some(RowIndex::Range(0..0)));
        }

        let new_index = match &self.index {
            RowIndex::Range(r) => RowIndex::Range(r.start..r.start + num_rows_to_take),
            RowIndex::Int(v) => RowIndex::Int(v[0..num_rows_to_take].to_vec()),
            RowIndex::Date(v) => RowIndex::Date(v[0..num_rows_to_take].to_vec()),
        };

        let mut new_typed_frames = Vec::new();
        for col_name in self.columns() {
            let col_data = self.column(col_name);
            match col_data {
                DataFrameColumn::F64(s) => {
                    let new_data = s[0..num_rows_to_take].to_vec();
                    new_typed_frames.push(TypedFrame::F64(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
                DataFrameColumn::I64(s) => {
                    let new_data = s[0..num_rows_to_take].to_vec();
                    new_typed_frames.push(TypedFrame::I64(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
                DataFrameColumn::Bool(s) => {
                    let new_data = s[0..num_rows_to_take].to_vec();
                    new_typed_frames.push(TypedFrame::Bool(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
                DataFrameColumn::String(s) => {
                    let new_data = s[0..num_rows_to_take].to_vec();
                    new_typed_frames.push(TypedFrame::String(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
            }
        }

        DataFrame::new(new_typed_frames, self.column_names.clone(), Some(new_index))
    }

    /// Returns a new DataFrame containing the last `n` rows.
    /// If `n` is greater than the number of rows in the DataFrame,
    /// the entire DataFrame is returned.
    pub fn tail_n(&self, n: usize) -> Self {
        let num_rows_to_take = n.min(self.rows());

        if num_rows_to_take == 0 {
            return DataFrame::new(vec![], vec![], Some(RowIndex::Range(0..0)));
        }

        let start_row_idx = self.rows() - num_rows_to_take;

        let new_index = match &self.index {
            RowIndex::Range(r) => RowIndex::Range(r.start + start_row_idx..r.start + self.rows()),
            RowIndex::Int(v) => RowIndex::Int(v[start_row_idx..].to_vec()),
            RowIndex::Date(v) => RowIndex::Date(v[start_row_idx..].to_vec()),
        };

        let mut new_typed_frames = Vec::new();
        for col_name in self.columns() {
            let col_data = self.column(col_name);
            match col_data {
                DataFrameColumn::F64(s) => {
                    let new_data = s[start_row_idx..].to_vec();
                    new_typed_frames.push(TypedFrame::F64(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
                DataFrameColumn::I64(s) => {
                    let new_data = s[start_row_idx..].to_vec();
                    new_typed_frames.push(TypedFrame::I64(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
                DataFrameColumn::Bool(s) => {
                    let new_data = s[start_row_idx..].to_vec();
                    new_typed_frames.push(TypedFrame::Bool(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
                DataFrameColumn::String(s) => {
                    let new_data = s[start_row_idx..].to_vec();
                    new_typed_frames.push(TypedFrame::String(Frame::new(
                        Matrix::from_cols(vec![new_data]),
                        vec![col_name.clone()],
                        Some(new_index.clone()),
                    )));
                }
            }
        }

        DataFrame::new(new_typed_frames, self.column_names.clone(), Some(new_index))
    }

    /// Returns a new DataFrame containing the first 5 rows.
    /// This is a convenience method for `head_n(5)`.
    pub fn head(&self) -> Self {
        self.head_n(DEFAULT_DISPLAY_ROWS)
    }

    /// Returns a new DataFrame containing the last 5 rows.
    /// This is a convenience method for `tail_n(5)`.
    pub fn tail(&self) -> Self {
        self.tail_n(DEFAULT_DISPLAY_ROWS)
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.rows() == 0 || self.cols() == 0 {
            return write!(
                f,
                "Empty DataFrame\nRows: {}, Columns: {}",
                self.rows(),
                self.cols()
            );
        }

        let mut output = String::new();
        let mut column_widths = HashMap::new();

        // Calculate max width for index column
        let mut max_index_width = 0;
        for i in 0..self.rows() {
            let index_str = match &self.index {
                RowIndex::Range(r) => format!("{}", r.start + i),
                RowIndex::Int(v) => format!("{}", v[i]),
                RowIndex::Date(v) => format!("{}", v[i]),
            };
            max_index_width = max_index_width.max(index_str.len());
        }
        // Ensure index header "Index" fits
        max_index_width = max_index_width.max("Index".len());

        // Calculate max width for each data column
        for col_name in self.columns() {
            let mut max_width = col_name.len();
            let col_data = self.column(col_name);
            for i in 0..self.rows() {
                let cell_str = match col_data {
                    DataFrameColumn::F64(s) => format!("{:.2}", s[i]), // Format floats
                    DataFrameColumn::I64(s) => format!("{}", s[i]),
                    DataFrameColumn::Bool(s) => format!("{}", s[i]),
                    DataFrameColumn::String(s) => format!("{}", s[i]),
                };
                max_width = max_width.max(cell_str.len());
            }
            column_widths.insert(col_name, max_width);
        }

        // --- Print Header ---
        output.push_str(&format!("{:>width$} ", "Index", width = max_index_width));
        let mut _displayed_cols = 0;
        let total_cols = self.cols();
        let mut cols_to_display = Vec::new();

        // Decide which columns to display
        if total_cols <= DEFAULT_DISPLAY_COLS {
            cols_to_display.extend_from_slice(self.columns());
        } else {
            // Display first few and last few columns
            let num_first_cols = DEFAULT_DISPLAY_COLS / 2;
            let num_last_cols = DEFAULT_DISPLAY_COLS - num_first_cols;
            cols_to_display.extend_from_slice(&self.columns()[0..num_first_cols]);
            cols_to_display.push("...".to_string()); // Placeholder for omitted columns
            cols_to_display.extend_from_slice(&self.columns()[total_cols - num_last_cols..]);
        }

        for col_name in &cols_to_display {
            if col_name == "..." {
                output.push_str(&format!("{:>width$} ", "...", width = 5)); // Fixed width for ellipsis
            } else {
                output.push_str(&format!(
                    "{:>width$} ",
                    col_name,
                    width = column_widths[col_name]
                ));
                _displayed_cols += 1;
            }
        }
        output.push('\n');

        // --- Print Separator ---
        output.push_str(&format!("{:-<width$}-", "", width = max_index_width));
        for col_name in &cols_to_display {
            if col_name == "..." {
                output.push_str(&format!("{:-<width$}-", "", width = 5));
            } else {
                output.push_str(&format!(
                    "{:-<width$}-",
                    "",
                    width = column_widths[col_name]
                ));
            }
        }
        output.push('\n');

        // --- Print Rows ---
        let total_rows = self.rows();
        let mut rows_to_display = Vec::new();

        if total_rows <= DEFAULT_DISPLAY_ROWS {
            rows_to_display = (0..total_rows).collect();
        } else {
            // Display first few and last few rows
            let num_first_rows = DEFAULT_DISPLAY_ROWS / 2;
            let num_last_rows = DEFAULT_DISPLAY_ROWS - num_first_rows;
            rows_to_display.extend((0..num_first_rows).collect::<Vec<usize>>());
            rows_to_display.push(usize::MAX); // Sentinel for ellipsis row
            rows_to_display
                .extend((total_rows - num_last_rows..total_rows).collect::<Vec<usize>>());
        }

        for &row_idx in &rows_to_display {
            if row_idx == usize::MAX {
                // Ellipsis row
                output.push_str(&format!("{:>width$} ", "...", width = max_index_width));
                for _ in &cols_to_display {
                    output.push_str(&format!("{:>width$} ", "...", width = 5)); // Use a fixed width for ellipsis cells
                }
                output.push('\n');
                continue;
            }

            // Print index
            let index_str = match &self.index {
                RowIndex::Range(r) => format!("{}", r.start + row_idx),
                RowIndex::Int(v) => format!("{}", v[row_idx]),
                RowIndex::Date(v) => format!("{}", v[row_idx]),
            };
            output.push_str(&format!("{:>width$} ", index_str, width = max_index_width));

            // Print data cells
            for col_name in &cols_to_display {
                if col_name == "..." {
                    output.push_str(&format!("{:>width$} ", "...", width = 5));
                } else {
                    let col_data = self.column(col_name);
                    let cell_str = match col_data {
                        DataFrameColumn::F64(s) => format!("{:.2}", s[row_idx]),
                        DataFrameColumn::I64(s) => format!("{}", s[row_idx]),
                        DataFrameColumn::Bool(s) => format!("{}", s[row_idx]),
                        DataFrameColumn::String(s) => format!("{}", s[row_idx]),
                    };
                    output.push_str(&format!(
                        "{:>width$} ",
                        cell_str,
                        width = column_widths[col_name]
                    ));
                }
            }
            output.push('\n');
        }

        // --- Print Footer ---
        output.push_str(&format!(
            "\n[{} rows x {} columns]\n",
            self.rows(),
            self.cols()
        ));

        write!(f, "{}", output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::matrix::Matrix;
    use chrono::NaiveDate;

    // Helper for dates
    fn d(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    // Helper to create a simple f64 TypedFrame
    fn create_f64_typed_frame(name: &str, data: Vec<f64>, index: Option<RowIndex>) -> TypedFrame {
        let rows = data.len();
        let matrix = Matrix::from_cols(vec![data]);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        TypedFrame::F64(Frame::new(
            matrix,
            vec![name.to_string()],
            Some(frame_index),
        ))
    }

    // Helper to create a simple i64 TypedFrame
    fn create_i64_typed_frame(name: &str, data: Vec<i64>, index: Option<RowIndex>) -> TypedFrame {
        let rows = data.len();
        let matrix = Matrix::from_cols(vec![data]);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        TypedFrame::I64(Frame::new(
            matrix,
            vec![name.to_string()],
            Some(frame_index),
        ))
    }

    // Helper to create a simple String TypedFrame
    fn create_string_typed_frame(
        name: &str,
        data: Vec<String>,
        index: Option<RowIndex>,
    ) -> TypedFrame {
        let rows = data.len();
        let matrix = Matrix::from_cols(vec![data]);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        TypedFrame::String(Frame::new(
            matrix,
            vec![name.to_string()],
            Some(frame_index),
        ))
    }

    // Helper to create a multi-column f64 TypedFrame
    fn create_multi_f64_typed_frame(
        names: Vec<&str>,
        data: Vec<Vec<f64>>,
        index: Option<RowIndex>,
    ) -> TypedFrame {
        let rows = data[0].len();
        let matrix = Matrix::from_cols(data);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        TypedFrame::F64(Frame::new(
            matrix,
            names.into_iter().map(|s| s.to_string()).collect(),
            Some(frame_index),
        ))
    }

    #[test]
    fn test_dataframe_new_basic() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0], None);
        let col_b = create_i64_typed_frame("B", vec![4, 5, 6], None);
        let col_c = create_string_typed_frame(
            "C",
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            None,
        );

        let df = DataFrame::new(
            vec![col_a, col_b, col_c],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            None,
        );

        assert_eq!(df.rows(), 3);
        assert_eq!(df.cols(), 3);
        assert_eq!(df.columns(), &["A", "B", "C"]);
        assert_eq!(df.index(), &RowIndex::Range(0..3));

        // Check column data using the new DataFrameColumn view
        if let DataFrameColumn::F64(slice_a) = df.column("A") {
            assert_eq!(slice_a, &[1.0, 2.0, 3.0]);
        } else {
            panic!("Column A is not f64");
        }
        if let DataFrameColumn::I64(slice_b) = df.column("B") {
            assert_eq!(slice_b, &[4, 5, 6]);
        } else {
            panic!("Column B is not i64");
        }
        if let DataFrameColumn::String(slice_c) = df.column("C") {
            assert_eq!(
                slice_c,
                &["x".to_string(), "y".to_string(), "z".to_string()]
            );
        } else {
            panic!("Column C is not String");
        }
    }

    #[test]
    fn test_dataframe_rows_cols_columns_index() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0], None);
        let col_b = create_i64_typed_frame("B", vec![4, 5, 6], None);
        let df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );

        assert_eq!(df.rows(), 3);
        assert_eq!(df.cols(), 2);
        assert_eq!(df.columns(), &["A", "B"]);
        assert_eq!(df.index(), &RowIndex::Range(0..3));

        let empty_df = DataFrame::new(vec![], vec![], None);
        assert_eq!(empty_df.rows(), 0);
        assert_eq!(empty_df.cols(), 0);
        assert_eq!(empty_df.columns(), &[] as &[String]);
        assert_eq!(empty_df.index(), &RowIndex::Range(0..0));
    }

    #[test]
    fn test_dataframe_column_access() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20], None);
        let col_c =
            create_string_typed_frame("C", vec!["foo".to_string(), "bar".to_string()], None);

        let df = DataFrame::new(
            vec![col_a, col_b, col_c],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            None,
        );

        // Test f64 column
        let col_a_view = df.column("A");
        assert!(col_a_view.as_f64().is_some());
        assert_eq!(col_a_view.as_f64().unwrap(), &[1.0, 2.0]);
        assert!(col_a_view.as_i64().is_none());
        assert_eq!(col_a_view.len(), 2);
        assert!(!col_a_view.is_empty());

        // Test i64 column
        let col_b_view = df.column("B");
        assert!(col_b_view.as_i64().is_some());
        assert_eq!(col_b_view.as_i64().unwrap(), &[10, 20]);

        // Test String column
        let col_c_view = df.column("C");
        assert!(col_c_view.as_string().is_some());
        assert_eq!(
            col_c_view.as_string().unwrap(),
            &["foo".to_string(), "bar".to_string()]
        );
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'D'")]
    fn test_dataframe_column_panic_unknown() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        let df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.column("D");
    }

    #[test]
    fn test_dataframe_column_mut_access() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20], None);
        let col_c =
            create_string_typed_frame("C", vec!["foo".to_string(), "bar".to_string()], None);

        let mut df = DataFrame::new(
            vec![col_a, col_b, col_c],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            None,
        );

        // Test f64 column mut
        if let DataFrameColumnMut::F64(slice_a_mut) = df.column_mut("A") {
            slice_a_mut[0] = 100.0;
        } else {
            panic!("Column A is not f64 mut");
        }
        assert_eq!(df.column("A").as_f64().unwrap(), &[100.0, 2.0]);

        // Test i64 column mut
        if let DataFrameColumnMut::I64(slice_b_mut) = df.column_mut("B") {
            slice_b_mut[1] = 200;
        } else {
            panic!("Column B is not i64 mut");
        }
        assert_eq!(df.column("B").as_i64().unwrap(), &[10, 200]);

        // Test String column mut
        if let DataFrameColumnMut::String(slice_c_mut) = df.column_mut("C") {
            slice_c_mut[0] = "baz".to_string();
        } else {
            panic!("Column C is not String mut");
        }
        assert_eq!(
            df.column("C").as_string().unwrap(),
            &["baz".to_string(), "bar".to_string()]
        );
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'D'")]
    fn test_dataframe_column_mut_panic_unknown() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.column_mut("D");
    }

    #[test]
    fn test_dataframe_add_column() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);

        let new_col_b = create_i64_typed_frame("B", vec![10, 20], None);
        df.add_column("B".to_string(), new_col_b);

        assert_eq!(df.rows(), 2);
        assert_eq!(df.cols(), 2);
        assert_eq!(df.columns(), &["A", "B"]);
        assert_eq!(df.column("A").as_f64().unwrap(), &[1.0, 2.0]);
        assert_eq!(df.column("B").as_i64().unwrap(), &[10, 20]);

        let new_col_c =
            create_string_typed_frame("C", vec!["x".to_string(), "y".to_string()], None);
        df.add_column("C".to_string(), new_col_c);
        assert_eq!(df.cols(), 3);
        assert_eq!(df.columns(), &["A", "B", "C"]);
        assert_eq!(
            df.column("C").as_string().unwrap(),
            &["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    #[should_panic(expected = "duplicate column label: B")]
    fn test_dataframe_add_column_panic_duplicate_name() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        let new_col_b = create_i64_typed_frame("B", vec![10], None);
        df.add_column("B".to_string(), new_col_b);
        let another_col_b = create_i64_typed_frame("B", vec![20], None);
        df.add_column("B".to_string(), another_col_b);
    }

    #[test]
    #[should_panic(expected = "new column 'B' has inconsistent row count (1 vs 2)")]
    fn test_dataframe_add_column_panic_inconsistent_rows() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        let new_col_b = create_i64_typed_frame("B", vec![10], None); // Mismatch
        df.add_column("B".to_string(), new_col_b);
    }

    #[test]
    #[should_panic(expected = "provided TypedFrame must contain exactly one column named 'B'")]
    fn test_dataframe_add_column_panic_multi_column_typedframe() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        let multi_col_b = create_multi_f64_typed_frame(
            vec!["B", "C"],
            vec![vec![10.0, 20.0], vec![30.0, 40.0]],
            None,
        );
        df.add_column("B".to_string(), multi_col_b);
    }

    #[test]
    fn test_dataframe_delete_column() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20], None);
        let col_c = create_string_typed_frame("C", vec!["x".to_string(), "y".to_string()], None);

        let mut df = DataFrame::new(
            vec![col_a, col_b, col_c],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            None,
        );

        let deleted_col_b = df.delete_column("B");
        assert_eq!(df.cols(), 2);
        assert_eq!(df.columns(), &["A", "C"]);
        assert_eq!(df.column("A").as_f64().unwrap(), &[1.0, 2.0]);
        assert_eq!(
            df.column("C").as_string().unwrap(),
            &["x".to_string(), "y".to_string()]
        );

        if let TypedFrame::I64(frame_b) = deleted_col_b {
            assert_eq!(frame_b.column("B"), &[10, 20]);
        } else {
            panic!("Deleted column B is not i64 TypedFrame");
        }

        let deleted_col_a = df.delete_column("A");
        assert_eq!(df.cols(), 1);
        assert_eq!(df.columns(), &["C"]);
        assert_eq!(
            df.column("C").as_string().unwrap(),
            &["x".to_string(), "y".to_string()]
        );
        if let TypedFrame::F64(frame_a) = deleted_col_a {
            assert_eq!(frame_a.column("A"), &[1.0, 2.0]);
        } else {
            panic!("Deleted column A is not f64 TypedFrame");
        }

        let deleted_col_c = df.delete_column("C");
        assert_eq!(df.cols(), 0);
        assert_eq!(df.columns(), &[] as &[String]);
        if let TypedFrame::String(frame_c) = deleted_col_c {
            assert_eq!(frame_c.column("C"), &["x".to_string(), "y".to_string()]);
        } else {
            panic!("Deleted column C is not String TypedFrame");
        }
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'D'")]
    fn test_dataframe_delete_column_panic_unknown() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.delete_column("D");
    }

    #[test]
    fn test_dataframe_rename_column() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20], None);
        let mut df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );

        df.rename_column("A", "Alpha".to_string());
        assert_eq!(df.columns(), &["Alpha", "B"]);
        assert_eq!(df.column("Alpha").as_f64().unwrap(), &[1.0, 2.0]);
        assert!(!df.column_locations.contains_key("A"));
        assert!(df.column_locations.contains_key("Alpha"));

        df.rename_column("B", "Beta".to_string());
        assert_eq!(df.columns(), &["Alpha", "Beta"]);
        assert_eq!(df.column("Beta").as_i64().unwrap(), &[10, 20]);
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'C'")]
    fn test_dataframe_rename_column_panic_unknown_old_name() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.rename_column("C", "D".to_string());
    }

    #[test]
    #[should_panic(expected = "new column name 'B' already exists")]
    fn test_dataframe_rename_column_panic_new_name_exists() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        let col_b = create_i64_typed_frame("B", vec![10], None);
        let mut df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );
        df.rename_column("A", "B".to_string());
    }

    #[test]
    fn test_dataframe_sort_columns() {
        let col_c = create_f64_typed_frame("C", vec![1.0], None);
        let col_a = create_i64_typed_frame("A", vec![2], None);
        let col_b = create_string_typed_frame("B", vec!["x".to_string()], None);

        let mut df = DataFrame::new(
            vec![col_c, col_a, col_b],
            vec!["C".to_string(), "A".to_string(), "B".to_string()],
            None,
        );

        assert_eq!(df.columns(), &["C", "A", "B"]);
        df.sort_columns();
        assert_eq!(df.columns(), &["A", "B", "C"]);

        // Ensure data integrity after sort
        assert_eq!(df.column("A").as_i64().unwrap(), &[2]);
        assert_eq!(df.column("B").as_string().unwrap(), &["x".to_string()]);
        assert_eq!(df.column("C").as_f64().unwrap(), &[1.0]);
    }

    #[test]
    fn test_dataframe_new_with_multi_column_subframe() {
        let multi_f64_frame = create_multi_f64_typed_frame(
            vec!["X", "Y"],
            vec![vec![1.0, 2.0], vec![10.0, 20.0]],
            None,
        );
        let single_string_frame =
            create_string_typed_frame("Z", vec!["a".to_string(), "b".to_string()], None);

        let df = DataFrame::new(
            vec![multi_f64_frame, single_string_frame],
            vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
            None,
        );

        assert_eq!(df.rows(), 2);
        assert_eq!(df.cols(), 3);
        assert_eq!(df.columns(), &["X", "Y", "Z"]);

        if let DataFrameColumn::F64(slice_x) = df.column("X") {
            assert_eq!(slice_x, &[1.0, 2.0]);
        } else {
            panic!("Column X is not f64");
        }
        if let DataFrameColumn::F64(slice_y) = df.column("Y") {
            assert_eq!(slice_y, &[10.0, 20.0]);
        } else {
            panic!("Column Y is not f64");
        }
        if let DataFrameColumn::String(slice_z) = df.column("Z") {
            assert_eq!(slice_z, &["a".to_string(), "b".to_string()]);
        } else {
            panic!("Column Z is not String");
        }
    }

    #[test]
    fn test_dataframe_new_with_int_index() {
        let index = RowIndex::Int(vec![10, 20, 30]);
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0], Some(index.clone()));
        let col_b = create_i64_typed_frame("B", vec![4, 5, 6], Some(index.clone()));

        let df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            Some(index.clone()),
        );

        assert_eq!(df.rows(), 3);
        assert_eq!(df.cols(), 2);
        assert_eq!(df.index(), &index);
    }

    #[test]
    fn test_dataframe_new_with_date_index() {
        let index_vec = vec![d(2024, 1, 1), d(2024, 1, 2)];
        let index = RowIndex::Date(index_vec.clone());
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0], Some(index.clone()));
        let col_b = create_string_typed_frame(
            "B",
            vec!["hello".to_string(), "world".to_string()],
            Some(index.clone()),
        );

        let df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            Some(index.clone()),
        );

        assert_eq!(df.rows(), 2);
        assert_eq!(df.cols(), 2);
        assert_eq!(df.index(), &index);
    }

    #[test]
    #[should_panic(expected = "column name 'B' not found in provided TypedFrames")]
    fn test_dataframe_new_panic_col_name_not_found_in_subframes() {
        let col_a = create_f64_typed_frame("A", vec![1.0], None);
        DataFrame::new(vec![col_a], vec!["A".to_string(), "B".to_string()], None);
    }

    #[test]
    #[should_panic(expected = "TypedFrame has inconsistent row count (2 vs 3)")]
    fn test_dataframe_new_panic_inconsistent_rows() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0], None);
        let col_b = create_i64_typed_frame("B", vec![4, 5], None); // Mismatch
        DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "duplicate column name: A")]
    fn test_dataframe_new_panic_duplicate_col_name() {
        let col_a1 = create_f64_typed_frame("A", vec![1.0], None);
        let col_a2 = create_i64_typed_frame("A", vec![2], None); // Duplicate name
        DataFrame::new(
            vec![col_a1, col_a2],
            vec!["A".to_string(), "A".to_string()],
            None,
        );
    }

    #[test]
    fn test_dataframe_head_n() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20, 30, 40, 50, 60, 70], None);
        let df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );

        // Test head_n with n < rows
        let head_df = df.head_n(3);
        assert_eq!(head_df.rows(), 3);
        assert_eq!(head_df.cols(), 2);
        assert_eq!(head_df.columns(), &["A", "B"]);
        assert_eq!(head_df.index(), &RowIndex::Range(0..3));
        assert_eq!(head_df.column("A").as_f64().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(head_df.column("B").as_i64().unwrap(), &[10, 20, 30]);

        // Test head_n with n > rows
        let head_all_df = df.head_n(10);
        assert_eq!(head_all_df.rows(), 7);
        assert_eq!(head_all_df.cols(), 2);
        assert_eq!(head_all_df.columns(), &["A", "B"]);
        assert_eq!(head_all_df.index(), &RowIndex::Range(0..7));
        assert_eq!(
            head_all_df.column("A").as_f64().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            head_all_df.column("B").as_i64().unwrap(),
            &[10, 20, 30, 40, 50, 60, 70]
        );

        // Test head_n with n = 0
        let empty_head_df = df.head_n(0);
        assert_eq!(empty_head_df.rows(), 0);
        assert_eq!(empty_head_df.cols(), 0);
        assert!(empty_head_df.columns().is_empty());
        assert_eq!(empty_head_df.index(), &RowIndex::Range(0..0));

        // Test with Int index
        let int_index = RowIndex::Int(vec![100, 101, 102, 103, 104, 105, 106]);
        let col_a_int = create_f64_typed_frame(
            "A",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            Some(int_index.clone()),
        );
        let col_b_int = create_i64_typed_frame(
            "B",
            vec![10, 20, 30, 40, 50, 60, 70],
            Some(int_index.clone()),
        );
        let df_int = DataFrame::new(
            vec![col_a_int, col_b_int],
            vec!["A".to_string(), "B".to_string()],
            Some(int_index),
        );
        let head_int_df = df_int.head_n(3);
        assert_eq!(head_int_df.rows(), 3);
        assert_eq!(head_int_df.index(), &RowIndex::Int(vec![100, 101, 102]));
    }

    #[test]
    fn test_dataframe_tail_n() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20, 30, 40, 50, 60, 70], None);
        let df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );

        // Test tail_n with n < rows
        let tail_df = df.tail_n(3);
        assert_eq!(tail_df.rows(), 3);
        assert_eq!(tail_df.cols(), 2);
        assert_eq!(tail_df.columns(), &["A", "B"]);
        assert_eq!(tail_df.index(), &RowIndex::Range(4..7));
        assert_eq!(tail_df.column("A").as_f64().unwrap(), &[5.0, 6.0, 7.0]);
        assert_eq!(tail_df.column("B").as_i64().unwrap(), &[50, 60, 70]);

        // Test tail_n with n > rows
        let tail_all_df = df.tail_n(10);
        assert_eq!(tail_all_df.rows(), 7);
        assert_eq!(tail_all_df.cols(), 2);
        assert_eq!(tail_all_df.columns(), &["A", "B"]);
        assert_eq!(tail_all_df.index(), &RowIndex::Range(0..7));
        assert_eq!(
            tail_all_df.column("A").as_f64().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            tail_all_df.column("B").as_i64().unwrap(),
            &[10, 20, 30, 40, 50, 60, 70]
        );

        // Test tail_n with n = 0
        let empty_tail_df = df.tail_n(0);
        assert_eq!(empty_tail_df.rows(), 0);
        assert_eq!(empty_tail_df.cols(), 0);
        assert!(empty_tail_df.columns().is_empty());
        assert_eq!(empty_tail_df.index(), &RowIndex::Range(0..0));

        // Test with Int index
        let int_index = RowIndex::Int(vec![100, 101, 102, 103, 104, 105, 106]);
        let col_a_int = create_f64_typed_frame(
            "A",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            Some(int_index.clone()),
        );
        let col_b_int = create_i64_typed_frame(
            "B",
            vec![10, 20, 30, 40, 50, 60, 70],
            Some(int_index.clone()),
        );
        let df_int = DataFrame::new(
            vec![col_a_int, col_b_int],
            vec!["A".to_string(), "B".to_string()],
            Some(int_index),
        );
        let tail_int_df = df_int.tail_n(3);
        assert_eq!(tail_int_df.rows(), 3);
        assert_eq!(tail_int_df.index(), &RowIndex::Int(vec![104, 105, 106]));
    }

    #[test]
    fn test_dataframe_head() {
        let col_a = create_f64_typed_frame(
            "A",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            None,
        );
        let df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);

        let head_df = df.head();
        assert_eq!(head_df.rows(), DEFAULT_DISPLAY_ROWS);
        assert_eq!(
            head_df.column("A").as_f64().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0]
        );

        // Test with fewer rows than DEFAULT_DISPLAY_ROWS
        let col_b = create_f64_typed_frame("B", vec![1.0, 2.0], None);
        let df_small = DataFrame::new(vec![col_b], vec!["B".to_string()], None);
        let head_small_df = df_small.head();
        assert_eq!(head_small_df.rows(), 2);
        assert_eq!(head_small_df.column("B").as_f64().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_dataframe_tail() {
        let col_a = create_f64_typed_frame(
            "A",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            None,
        );
        let df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);

        let tail_df = df.tail();
        assert_eq!(tail_df.rows(), DEFAULT_DISPLAY_ROWS);
        assert_eq!(
            tail_df.column("A").as_f64().unwrap(),
            &[6.0, 7.0, 8.0, 9.0, 10.0]
        );

        // Test with fewer rows than DEFAULT_DISPLAY_ROWS
        let col_b = create_f64_typed_frame("B", vec![1.0, 2.0], None);
        let df_small = DataFrame::new(vec![col_b], vec!["B".to_string()], None);
        let tail_small_df = df_small.tail();
        assert_eq!(tail_small_df.rows(), 2);
        assert_eq!(tail_small_df.column("B").as_f64().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_dataframe_display_empty() {
        let empty_df = DataFrame::new(vec![], vec![], None);
        let expected_output = "Empty DataFrame\nRows: 0, Columns: 0";
        assert_eq!(format!("{}", empty_df), expected_output);
    }

    #[test]
    fn test_dataframe_display_basic() {
        let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0], None);
        let col_b = create_i64_typed_frame("B", vec![10, 20, 30], None);
        let col_c = create_string_typed_frame(
            "C",
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            None,
        );
        let df = DataFrame::new(
            vec![col_a, col_b, col_c],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            None,
        );

        let expected_output = "\
 Index      A   B  C 
------ ---- --- ---
     0 1.00  10   x 
     1 2.00  20   y 
     2 3.00  30   z 

[3 rows x 3 columns]
";
        assert_eq!(format!("{}", df), expected_output);
    }

    #[test]
    fn test_dataframe_display_truncation_rows() {
        let col_a = create_f64_typed_frame("A", (1..=10).map(|i| i as f64).collect(), None);
        let col_b = create_i64_typed_frame("B", (11..=20).collect(), None);
        let df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );

        let expected_output = "\
 Index      A   B 
------ ---- ---
     0 1.00  11 
     1 2.00  12 
     ...  ... ...
     8 9.00  19 
     9 10.00  20 

[10 rows x 2 columns]
";
        assert_eq!(format!("{}", df), expected_output);
    }

    #[test]
    fn test_dataframe_display_truncation_cols() {
        let mut cols_data = Vec::new();
        let mut col_names = Vec::new();
        for i in 0..15 {
            // 15 columns, more than DEFAULT_DISPLAY_COLS
            cols_data.push((1..=3).map(|r| (i * 10 + r) as f64).collect());
            col_names.push(format!("Col{}", i));
        }
        let typed_frame = create_multi_f64_typed_frame(
            col_names.iter().map(|s| s.as_str()).collect(),
            cols_data,
            None,
        );
        let df = DataFrame::new(vec![typed_frame], col_names, None);

        let expected_output = "\
 Index Col0 Col1 Col2 Col3 Col4 ... Col10 Col11 Col12 Col13 Col14 
------ ---- ---- ---- ---- ---- --- ----- ----- ----- ----- -----
     0 1.00 11.00 21.00 31.00 41.00 ... 101.00 111.00 121.00 131.00 141.00 
     1 2.00 12.00 22.00 32.00 42.00 ... 102.00 112.00 122.00 132.00 142.00 
     2 3.00 13.00 23.00 33.00 43.00 ... 103.00 113.00 123.00 133.00 143.00 

[3 rows x 15 columns]
";
        assert_eq!(format!("{}", df), expected_output);
    }

    #[test]
    fn test_dataframe_display_truncation_both() {
        let mut cols_data = Vec::new();
        let mut col_names = Vec::new();
        for i in 0..15 {
            // 15 columns
            cols_data.push((1..=10).map(|r| (i * 10 + r) as f64).collect()); // 10 rows
            col_names.push(format!("Col{}", i));
        }
        let typed_frame = create_multi_f64_typed_frame(
            col_names.iter().map(|s| s.as_str()).collect(),
            cols_data,
            None,
        );
        let df = DataFrame::new(vec![typed_frame], col_names, None);

        let expected_output = "\
 Index Col0 Col1 Col2 Col3 Col4 ... Col10 Col11 Col12 Col13 Col14 
------ ---- ---- ---- ---- ---- --- ----- ----- ----- ----- -----
     0 1.00 11.00 21.00 31.00 41.00 ... 101.00 111.00 121.00 131.00 141.00 
     1 2.00 12.00 22.00 32.00 42.00 ... 102.00 112.00 122.00 132.00 142.00 
     ...  ...  ...  ...  ...  ... ...   ...   ...   ...   ...   ...
     8 9.00 19.00 29.00 39.00 49.00 ... 109.00 119.00 129.00 139.00 149.00 
     9 10.00 20.00 30.00 40.00 50.00 ... 110.00 120.00 130.00 140.00 150.00 

[10 rows x 15 columns]
";
        assert_eq!(format!("{}", df), expected_output);
    }
}
