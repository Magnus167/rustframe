use crate::frame::{Frame, RowIndex};
use crate::matrix::Matrix;
use std::collections::HashMap;
use std::fmt;

/// Represents a column in a DataFrame, holding data of a specific type.
/// Each variant wraps a `Frame<T>` where T is the data type.
#[derive(Debug, Clone, PartialEq)]
pub enum DataFrameColumn {
    F64(Frame<f64>),
    I64(Frame<i64>),
    Bool(Frame<bool>),
    String(Frame<String>),
    // Add more types as needed
}

impl DataFrameColumn {
    /// Returns the number of rows in the column.
    pub fn rows(&self) -> usize {
        match self {
            DataFrameColumn::F64(f) => f.rows(),
            DataFrameColumn::I64(f) => f.rows(),
            DataFrameColumn::Bool(f) => f.rows(),
            DataFrameColumn::String(f) => f.rows(),
        }
    }

    /// Returns the column name.
    /// Panics if the internal frame has more than one column (which it shouldn't for a single DataFrameColumn).
    pub fn name(&self) -> &str {
        match self {
            DataFrameColumn::F64(f) => &f.columns()[0],
            DataFrameColumn::I64(f) => &f.columns()[0],
            DataFrameColumn::Bool(f) => &f.columns()[0],
            DataFrameColumn::String(f) => &f.columns()[0],
        }
    }

    /// Returns a reference to the underlying RowIndex.
    pub fn index(&self) -> &RowIndex {
        match self {
            DataFrameColumn::F64(f) => f.index(),
            DataFrameColumn::I64(f) => f.index(),
            DataFrameColumn::Bool(f) => f.index(),
            DataFrameColumn::String(f) => f.index(),
        }
    }
}

/// A DataFrame capable of holding multiple data types.
///
/// Internally, a DataFrame manages a collection of `Frame` instances,
/// each holding data of a single, homogeneous type. The logical column
/// order is maintained separately from the physical storage.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    /// The logical order of column names.
    column_names: Vec<String>,
    /// A map from column name to its corresponding DataFrameColumn (which wraps a Frame).
    data: HashMap<String, DataFrameColumn>,
    /// The common row index for all columns in the DataFrame.
    index: RowIndex,
    /// The number of rows in the DataFrame.
    rows: usize,
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
    /// * `columns` - A vector of `DataFrameColumn` instances, each representing a column.
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
        columns: Vec<DataFrameColumn>,
        column_names: Vec<String>,
        index: Option<RowIndex>,
    ) -> Self {
        if columns.len() != column_names.len() {
            panic!(
                "DataFrame::new: column data count ({}) mismatch column names count ({})",
                columns.len(),
                column_names.len()
            );
        }

        if columns.is_empty() {
            return Self {
                column_names: Vec::new(),
                data: HashMap::new(),
                index: index.unwrap_or(RowIndex::Range(0..0)),
                rows: 0,
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

        let mut data_map = HashMap::with_capacity(columns.len());
        let mut final_column_names = Vec::with_capacity(column_names.len());

        for (i, col_name) in column_names.into_iter().enumerate() {
            let col = columns[i].clone();

            if col.rows() != num_rows {
                panic!(
                    "DataFrame::new: column '{}' has inconsistent row count ({} vs {})",
                    col_name,
                    col.rows(),
                    num_rows
                );
            }

            if col.index() != &common_index {
                panic!(
                    "DataFrame::new: column '{}' has inconsistent index with common index",
                    col_name
                );
            }

            if data_map.insert(col_name.clone(), col).is_some() {
                panic!("DataFrame::new: duplicate column name: {}", col_name);
            }
            final_column_names.push(col_name);
        }

        Self {
            column_names: final_column_names,
            data: data_map,
            index: common_index,
            rows: num_rows,
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

    /// Returns an immutable reference to a column by its name.
    /// Panics if the column is not found.
    pub fn column(&self, name: &str) -> &DataFrameColumn {
        self.data
            .get(name)
            .unwrap_or_else(|| panic!("DataFrame::column: unknown column label: '{}'", name))
    }

    /// Returns a mutable reference to a column by its name.
    /// Panics if the column is not found.
    pub fn column_mut(&mut self, name: &str) -> &mut DataFrameColumn {
        self.data
            .get_mut(name)
            .unwrap_or_else(|| panic!("DataFrame::column_mut: unknown column label: '{}'", name))
    }

    /// Adds a new column to the DataFrame.
    /// Panics if a column with the same name already exists or if the new column's
    /// row count or index does not match the DataFrame's.
    pub fn add_column(&mut self, name: String, column_data: DataFrameColumn) {
        if self.data.contains_key(&name) {
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

        self.column_names.push(name.clone());
        self.data.insert(name, column_data);
    }

    /// Deletes a column by name and returns it.
    /// Panics if the column name is not found.
    pub fn delete_column(&mut self, name: &str) -> DataFrameColumn {
        let removed_col = self.data.remove(name).unwrap_or_else(|| {
            panic!("DataFrame::delete_column: unknown column label: '{}'", name)
        });

        // Remove from column_names vector and rebuild if necessary (to maintain order)
        if let Some(pos) = self.column_names.iter().position(|n| n == name) {
            self.column_names.remove(pos);
        }

        removed_col
    }

    /// Renames an existing column.
    pub fn rename_column(&mut self, old_name: &str, new_name: String) {
        if old_name == new_name {
            return; // No change needed
        }
        if self.data.contains_key(&new_name) {
            panic!(
                "DataFrame::rename_column: new column name '{}' already exists",
                new_name
            );
        }

        let column = self.data.remove(old_name).unwrap_or_else(|| {
            panic!(
                "DataFrame::rename_column: unknown column label: '{}'",
                old_name
            )
        });
        let new_name_clone = new_name.clone();
        self.data.insert(new_name, column);
        if let Some(pos) = self.column_names.iter().position(|n| n == old_name) {
            self.column_names[pos] = new_name_clone.clone();
        }

        // rename the column in the underlying Frame as well
        if let Some(col) = self.data.get_mut(&new_name_clone) {
            match col {
                DataFrameColumn::F64(frame) => {
                    frame.rename(old_name, new_name_clone.clone());
                }
                DataFrameColumn::I64(frame) => {
                    frame.rename(old_name, new_name_clone.clone());
                }
                DataFrameColumn::String(frame) => {
                    frame.rename(old_name, new_name_clone.clone());
                }
                DataFrameColumn::Bool(frame) => {
                    frame.rename(old_name, new_name_clone.clone());
                }
            }
        }
    }

    /// Sorts columns alphabetically by name.
    pub fn sort_columns(&mut self) {
        self.column_names.sort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    // Helper for dates
    fn d(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    // Helper to create a simple f64 Frame
    fn create_f64_frame(name: &str, data: Vec<f64>, index: Option<RowIndex>) -> DataFrameColumn {
        let rows = data.len();
        let matrix = Matrix::from_cols(vec![data]);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        DataFrameColumn::F64(Frame::new(
            matrix,
            vec![name.to_string()],
            Some(frame_index),
        ))
    }

    // Helper to create a simple i64 Frame
    fn create_i64_frame(name: &str, data: Vec<i64>, index: Option<RowIndex>) -> DataFrameColumn {
        let rows = data.len();
        let matrix = Matrix::from_cols(vec![data]);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        DataFrameColumn::I64(Frame::new(
            matrix,
            vec![name.to_string()],
            Some(frame_index),
        ))
    }

    // Helper to create a simple String Frame
    fn create_string_frame(
        name: &str,
        data: Vec<String>,
        index: Option<RowIndex>,
    ) -> DataFrameColumn {
        let rows = data.len();
        let matrix = Matrix::from_cols(vec![data]);
        let frame_index = index.unwrap_or(RowIndex::Range(0..rows));
        DataFrameColumn::String(Frame::new(
            matrix,
            vec![name.to_string()],
            Some(frame_index),
        ))
    }

    #[test]
    fn test_dataframe_new_basic() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0, 3.0], None);
        let col_b = create_i64_frame("B", vec![4, 5, 6], None);
        let col_c = create_string_frame(
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

        // Check column data
        if let DataFrameColumn::F64(frame_a) = df.column("A") {
            assert_eq!(frame_a["A"], vec![1.0, 2.0, 3.0]);
        } else {
            panic!("Column A is not f64");
        }
        if let DataFrameColumn::I64(frame_b) = df.column("B") {
            assert_eq!(frame_b["B"], vec![4, 5, 6]);
        } else {
            panic!("Column B is not i64");
        }
        if let DataFrameColumn::String(frame_c) = df.column("C") {
            assert_eq!(frame_c["C"], vec!["x", "y", "z"]);
        } else {
            panic!("Column C is not String");
        }
    }

    #[test]
    fn test_dataframe_new_with_int_index() {
        let index = RowIndex::Int(vec![10, 20, 30]);
        let col_a = create_f64_frame("A", vec![1.0, 2.0, 3.0], Some(index.clone()));
        let col_b = create_i64_frame("B", vec![4, 5, 6], Some(index.clone()));

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
        let col_a = create_f64_frame("A", vec![1.0, 2.0], Some(index.clone()));
        let col_b = create_string_frame(
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
    #[should_panic(expected = "column data count (1) mismatch column names count (2)")]
    fn test_dataframe_new_panic_col_count_mismatch() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        DataFrame::new(vec![col_a], vec!["A".to_string(), "B".to_string()], None);
    }

    #[test]
    #[should_panic(expected = "column 'B' has inconsistent row count (2 vs 3)")]
    fn test_dataframe_new_panic_inconsistent_rows() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0, 3.0], None);
        let col_b = create_i64_frame("B", vec![4, 5], None); // Mismatch
        DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "duplicate column name: A")]
    fn test_dataframe_new_panic_duplicate_col_name() {
        let col_a1 = create_f64_frame("A", vec![1.0], None);
        let col_a2 = create_i64_frame("A", vec![2], None); // Duplicate name
        DataFrame::new(
            vec![col_a1, col_a2],
            vec!["A".to_string(), "A".to_string()],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "provided index length (2) mismatch column rows (3)")]
    fn test_dataframe_new_panic_index_len_mismatch() {
        let index = RowIndex::Int(vec![10, 20]); // Length 2
        let col_a = create_f64_frame("A", vec![1.0, 2.0, 3.0], None); // Length 3
        DataFrame::new(vec![col_a], vec!["A".to_string()], Some(index));
    }

    #[test]
    #[should_panic(expected = "column 'A' has inconsistent index with common index")]
    fn test_dataframe_new_panic_inconsistent_column_index() {
        let common_index = RowIndex::Int(vec![10, 20]);
        let col_a = create_f64_frame("A", vec![1.0, 2.0], None); // Uses Range index by default
        DataFrame::new(vec![col_a], vec!["A".to_string()], Some(common_index));
    }

    #[test]
    fn test_dataframe_add_column() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);

        let new_col_b = create_i64_frame("B", vec![10, 20], None);
        df.add_column("B".to_string(), new_col_b);

        assert_eq!(df.cols(), 2);
        assert_eq!(df.columns(), &["A", "B"]);
        if let DataFrameColumn::I64(frame_b) = df.column("B") {
            assert_eq!(frame_b["B"], vec![10, 20]);
        } else {
            panic!("Column B is not i64");
        }

        let new_col_c = create_string_frame("C", vec!["foo".to_string(), "bar".to_string()], None);
        df.add_column("C".to_string(), new_col_c);
        assert_eq!(df.cols(), 3);
        assert_eq!(df.columns(), &["A", "B", "C"]);
    }

    #[test]
    #[should_panic(expected = "duplicate column label: A")]
    fn test_dataframe_add_column_panic_duplicate() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        let new_col_a = create_i64_frame("A", vec![2], None);
        df.add_column("A".to_string(), new_col_a);
    }

    #[test]
    #[should_panic(expected = "new column 'B' has inconsistent row count (1 vs 2)")]
    fn test_dataframe_add_column_panic_inconsistent_rows() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        let new_col_b = create_i64_frame("B", vec![10], None); // Mismatch
        df.add_column("B".to_string(), new_col_b);
    }

    #[test]
    #[should_panic(expected = "new column 'B' has inconsistent index with DataFrame's index")]
    fn test_dataframe_add_column_panic_inconsistent_index() {
        let df_index = RowIndex::Int(vec![1, 2]);
        let col_a = create_f64_frame("A", vec![1.0, 2.0], Some(df_index.clone()));
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], Some(df_index));

        let new_col_b = create_i64_frame("B", vec![10, 20], None); // Uses Range index
        df.add_column("B".to_string(), new_col_b);
    }

    #[test]
    fn test_dataframe_delete_column() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0], None);
        let col_b = create_i64_frame("B", vec![4, 5], None);
        let mut df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );

        assert_eq!(df.cols(), 2);
        assert_eq!(df.columns(), &["A", "B"]);

        let deleted_col = df.delete_column("A");
        assert_eq!(df.cols(), 1);
        assert_eq!(df.columns(), &["B"]);
        if let DataFrameColumn::F64(frame_a) = deleted_col {
            assert_eq!(frame_a["A"], vec![1.0, 2.0]);
        } else {
            panic!("Deleted column was not f64");
        }

        // Delete the last column
        df.delete_column("B");
        assert_eq!(df.cols(), 0);
        assert!(df.columns().is_empty());
        assert!(df.data.is_empty());
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'C'")]
    fn test_dataframe_delete_column_panic_unknown() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.delete_column("C");
    }

    #[test]
    fn test_dataframe_rename_column() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);

        df.rename_column("A", "X".to_string());
        assert_eq!(df.columns(), &["X"]);
        assert!(df.data.contains_key("X"));
        assert!(!df.data.contains_key("A"));
        if let DataFrameColumn::F64(frame_x) = df.column("X") {
            assert_eq!(frame_x["X"], vec![1.0, 2.0]);
        } else {
            panic!("Column X is not f64");
        }
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'Z'")]
    fn test_dataframe_rename_column_panic_old_not_found() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.rename_column("Z", "Y".to_string());
    }

    #[test]
    #[should_panic(expected = "new column name 'B' already exists")]
    fn test_dataframe_rename_column_panic_new_exists() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        let col_b = create_i64_frame("B", vec![2], None);
        let mut df = DataFrame::new(
            vec![col_a, col_b],
            vec!["A".to_string(), "B".to_string()],
            None,
        );
        df.rename_column("A", "B".to_string());
    }

    #[test]
    fn test_dataframe_sort_columns() {
        let col_c = create_f64_frame("C", vec![1.0, 2.0], None);
        let col_a = create_i64_frame("A", vec![3, 4], None);
        let col_b = create_string_frame("B", vec!["x".to_string(), "y".to_string()], None);

        let mut df = DataFrame::new(
            vec![col_c, col_a, col_b],
            vec!["C".to_string(), "A".to_string(), "B".to_string()],
            None,
        );

        assert_eq!(df.columns(), &["C", "A", "B"]);
        df.sort_columns();
        assert_eq!(df.columns(), &["A", "B", "C"]);

        // Verify data integrity after sort
        if let DataFrameColumn::I64(frame_a) = df.column("A") {
            assert_eq!(frame_a["A"], vec![3, 4]);
        }
        if let DataFrameColumn::String(frame_b) = df.column("B") {
            assert_eq!(frame_b["B"], vec!["x", "y"]);
        }
        if let DataFrameColumn::F64(frame_c) = df.column("C") {
            assert_eq!(frame_c["C"], vec![1.0, 2.0]);
        }
    }

    #[test]
    fn test_dataframe_empty() {
        let df = DataFrame::new(vec![], vec![], None);
        assert_eq!(df.rows(), 0);
        assert_eq!(df.cols(), 0);
        assert!(df.columns().is_empty());
        assert!(df.data.is_empty());
        assert_eq!(df.index(), &RowIndex::Range(0..0));
    }

    #[test]
    fn test_dataframe_column_accessors() {
        let col_a = create_f64_frame("A", vec![1.0, 2.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);

        // Immutable access
        let col_ref = df.column("A");
        if let DataFrameColumn::F64(frame_a) = col_ref {
            assert_eq!(frame_a["A"], vec![1.0, 2.0]);
        } else {
            panic!("Column A is not f64");
        }

        // Mutable access
        let col_mut_ref = df.column_mut("A");
        if let DataFrameColumn::F64(frame_a_mut) = col_mut_ref {
            frame_a_mut["A"][0] = 99.0;
        } else {
            panic!("Column A is not f64");
        }

        // Verify mutation
        if let DataFrameColumn::F64(frame_a) = df.column("A") {
            assert_eq!(frame_a["A"], vec![99.0, 2.0]);
        }
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'Z'")]
    fn test_dataframe_column_panic_unknown() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        let df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.column("Z");
    }

    #[test]
    #[should_panic(expected = "unknown column label: 'Z'")]
    fn test_dataframe_column_mut_panic_unknown() {
        let col_a = create_f64_frame("A", vec![1.0], None);
        let mut df = DataFrame::new(vec![col_a], vec!["A".to_string()], None);
        df.column_mut("Z");
    }
}
