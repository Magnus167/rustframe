use crate::frame::{Frame, RowIndex};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt; // Import TypeId

const DEFAULT_DISPLAY_ROWS: usize = 5;
const DEFAULT_DISPLAY_COLS: usize = 10;

// Trait to enable type-agnostic operations on Frame objects within DataFrame
pub trait SubFrame: Send + Sync + fmt::Debug + Any {
    fn rows(&self) -> usize;
    fn get_value_as_string(&self, physical_row_idx: usize, col_name: &str) -> String;
    fn clone_box(&self) -> Box<dyn SubFrame>;
    fn delete_column_from_frame(&mut self, col_name: &str);
    fn get_frame_cols(&self) -> usize; // Add a method to get the number of columns in the underlying frame

    // Methods for downcasting to concrete types
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// Implement SubFrame for any Frame<T> that meets the requirements
impl<T> SubFrame for Frame<T>
where
    T: Clone + PartialEq + fmt::Display + fmt::Debug + 'static + Send + Sync + Any,
{
    fn rows(&self) -> usize {
        self.rows()
    }

    fn get_value_as_string(&self, physical_row_idx: usize, col_name: &str) -> String {
        self.get_row(physical_row_idx).get(col_name).to_string()
    }

    fn clone_box(&self) -> Box<dyn SubFrame> {
        Box::new(self.clone())
    }

    fn delete_column_from_frame(&mut self, col_name: &str) {
        self.delete_column(col_name);
    }

    fn get_frame_cols(&self) -> usize {
        self.cols()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct DataFrame {
    frames_by_type: HashMap<TypeId, Box<dyn SubFrame>>, // Maps TypeId to the Frame holding columns of that type
    column_to_type: HashMap<String, TypeId>,            // Maps column name to its TypeId
    column_names: Vec<String>,
    index: RowIndex,
}

impl DataFrame {
    pub fn new() -> Self {
        DataFrame {
            frames_by_type: HashMap::new(),
            column_to_type: HashMap::new(),
            column_names: Vec::new(),
            index: RowIndex::Range(0..0), // Initialize with an empty range index
        }
    }

    /// Returns the number of rows in the DataFrame.
    pub fn rows(&self) -> usize {
        self.index.len()
    }

    /// Returns the number of columns in the DataFrame.
    pub fn cols(&self) -> usize {
        self.column_names.len()
    }

    /// Returns a reference to the vector of column names.
    pub fn get_column_names(&self) -> &Vec<String> {
        &self.column_names
    }

    /// Returns the number of internal Frame objects (one per unique data type).
    pub fn num_internal_frames(&self) -> usize {
        self.frames_by_type.len()
    }

    /// Returns a reference to a column of a specific type, if it exists.
    pub fn get_column<T>(&self, col_name: &str) -> Option<&[T]>
    where
        T: Clone + PartialEq + fmt::Display + fmt::Debug + 'static + Send + Sync + Any,
    {
        let expected_type_id = TypeId::of::<T>();
        if let Some(actual_type_id) = self.column_to_type.get(col_name) {
            if *actual_type_id == expected_type_id {
                if let Some(sub_frame_box) = self.frames_by_type.get(actual_type_id) {
                    if let Some(frame) = sub_frame_box.as_any().downcast_ref::<Frame<T>>() {
                        return Some(frame.column(col_name));
                    }
                }
            }
        }
        None
    }

    /// Returns a HashMap representing a row, mapping column names to their string values.
    pub fn get_row(&self, row_idx: usize) -> Option<HashMap<String, String>> {
        if row_idx >= self.rows() {
            return None;
        }

        let mut row_data = HashMap::new();
        for col_name in &self.column_names {
            if let Some(type_id) = self.column_to_type.get(col_name) {
                if let Some(sub_frame_box) = self.frames_by_type.get(type_id) {
                    let value = sub_frame_box.get_value_as_string(row_idx, col_name);
                    row_data.insert(col_name.clone(), value);
                }
            }
        }
        Some(row_data)
    }

    pub fn add_column<T>(&mut self, col_name: &str, data: Vec<T>)
    where
        T: Clone + PartialEq + fmt::Display + fmt::Debug + 'static + Send + Sync + Any,
    {
        let type_id = TypeId::of::<T>();
        let col_name_string = col_name.to_string();

        // Check for duplicate column name across the entire DataFrame
        if self.column_to_type.contains_key(&col_name_string) {
            panic!(
                "DataFrame::add_column: duplicate column name: '{}'",
                col_name_string
            );
        }

        // If this is the first column being added, set the DataFrame's index
        if self.column_names.is_empty() {
            self.index = RowIndex::Range(0..data.len());
        } else {
            // Ensure new column has the same number of rows as existing columns
            if data.len() != self.index.len() {
                panic!(
                    "DataFrame::add_column: new column '{}' has {} rows, but existing columns have {} rows",
                    col_name_string,
                    data.len(),
                    self.index.len()
                );
            }
        }

        // Check if a Frame of this type already exists
        if let Some(sub_frame_box) = self.frames_by_type.get_mut(&type_id) {
            // Downcast to the concrete Frame<T> and add the column
            if let Some(frame) = sub_frame_box.as_any_mut().downcast_mut::<Frame<T>>() {
                frame.add_column(col_name_string.clone(), data);
            } else {
                // This should ideally not happen if TypeId matches, but good for safety
                panic!(
                    "Type mismatch when downcasting existing SubFrame for TypeId {:?}",
                    type_id
                );
            }
        } else {
            // No Frame of this type exists, create a new one
            // The Frame::new constructor expects a Matrix and column names.
            // We create a Matrix from a single column vector.
            let new_frame = Frame::new(
                crate::matrix::Matrix::from_cols(vec![data]),
                vec![col_name_string.clone()],
                Some(self.index.clone()), // Pass the DataFrame's index to the new Frame
            );
            self.frames_by_type.insert(type_id, Box::new(new_frame));
        }

        // Update column mappings and names
        self.column_to_type.insert(col_name_string.clone(), type_id);
        self.column_names.push(col_name_string);
    }

    /// Drops a column from the DataFrame.
    /// Panics if the column does not exist.
    pub fn drop_column(&mut self, col_name: &str) {
        let col_name_string = col_name.to_string();

        // 1. Get the TypeId associated with the column
        let type_id = self
            .column_to_type
            .remove(&col_name_string)
            .unwrap_or_else(|| {
                panic!(
                    "DataFrame::drop_column: column '{}' not found",
                    col_name_string
                );
            });

        // 2. Remove the column name from the ordered list
        self.column_names.retain(|name| name != &col_name_string);

        // 3. Find the Frame object and delete the column from it
        if let Some(sub_frame_box) = self.frames_by_type.get_mut(&type_id) {
            sub_frame_box.delete_column_from_frame(&col_name_string);

            // 4. If the Frame object for this type becomes empty, remove it from frames_by_type
            if sub_frame_box.get_frame_cols() == 0 {
                self.frames_by_type.remove(&type_id);
            }
        } else {
            // This should not happen if column_to_type was consistent
            panic!(
                "DataFrame::drop_column: internal error, no frame found for type_id {:?}",
                type_id
            );
        }
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display column headers
        for col_name in self.column_names.iter().take(DEFAULT_DISPLAY_COLS) {
            write!(f, "{:<15}", col_name)?;
        }
        if self.column_names.len() > DEFAULT_DISPLAY_COLS {
            write!(f, "...")?;
        }
        writeln!(f)?;

        // Display data rows
        let mut displayed_rows = 0;
        for i in 0..self.index.len() {
            if displayed_rows >= DEFAULT_DISPLAY_ROWS {
                writeln!(f, "...")?;
                break;
            }
            for col_name in self.column_names.iter().take(DEFAULT_DISPLAY_COLS) {
                if let Some(type_id) = self.column_to_type.get(col_name) {
                    if let Some(sub_frame_box) = self.frames_by_type.get(type_id) {
                        write!(f, "{:<15}", sub_frame_box.get_value_as_string(i, col_name))?;
                    } else {
                        // This case indicates an inconsistency: column_to_type has an entry,
                        // but frames_by_type doesn't have the corresponding Frame.
                        write!(f, "{:<15}", "[ERROR]")?;
                    }
                } else {
                    // This case indicates an inconsistency: column_names has an entry,
                    // but column_to_type doesn't have the corresponding column.
                    write!(f, "{:<15}", "[ERROR]")?;
                }
            }
            if self.column_names.len() > DEFAULT_DISPLAY_COLS {
                write!(f, "...")?;
            }
            writeln!(f)?;
            displayed_rows += 1;
        }
        Ok(())
    }
}

impl fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataFrame")
            .field("column_names", &self.column_names)
            .field("index", &self.index)
            .field("column_to_type", &self.column_to_type)
            .field("frames_by_type", &self.frames_by_type)
            .finish()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::matrix::Matrix;

    #[test]
    fn test_dataframe_new() {
        let df = DataFrame::new();
        assert_eq!(df.rows(), 0);
        assert_eq!(df.cols(), 0);
        assert!(df.get_column_names().is_empty());
        assert!(df.frames_by_type.is_empty());
        assert!(df.column_to_type.is_empty());
    }

    #[test]
    fn test_dataframe_add_column_initial() {
        let mut df = DataFrame::new();
        let data = vec![1, 2, 3];
        df.add_column("col_int", data.clone());

        assert_eq!(df.rows(), 3);
        assert_eq!(df.cols(), 1);
        assert_eq!(df.get_column_names(), &vec!["col_int".to_string()]);
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert_eq!(df.column_to_type.get("col_int"), Some(&TypeId::of::<i32>()));

        // Verify the underlying frame
        let sub_frame_box = df.frames_by_type.get(&TypeId::of::<i32>()).unwrap();
        let frame = sub_frame_box.as_any().downcast_ref::<Frame<i32>>().unwrap();
        assert_eq!(frame.rows(), 3);
        assert_eq!(frame.cols(), 1);
        assert_eq!(frame.columns(), &vec!["col_int".to_string()]);
    }

    #[test]
    fn test_dataframe_add_column_same_type() {
        let mut df = DataFrame::new();
        df.add_column("col_int1", vec![1, 2, 3]);
        df.add_column("col_int2", vec![4, 5, 6]);

        assert_eq!(df.rows(), 3);
        assert_eq!(df.cols(), 2);
        assert_eq!(
            df.get_column_names(),
            &vec!["col_int1".to_string(), "col_int2".to_string()]
        );
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert_eq!(
            df.column_to_type.get("col_int1"),
            Some(&TypeId::of::<i32>())
        );
        assert_eq!(
            df.column_to_type.get("col_int2"),
            Some(&TypeId::of::<i32>())
        );

        // Verify the underlying frame
        let sub_frame_box = df.frames_by_type.get(&TypeId::of::<i32>()).unwrap();
        let frame = sub_frame_box.as_any().downcast_ref::<Frame<i32>>().unwrap();
        assert_eq!(frame.rows(), 3);
        assert_eq!(frame.cols(), 2);
        assert_eq!(
            frame.columns(),
            &vec!["col_int1".to_string(), "col_int2".to_string()]
        );
    }

    #[test]
    fn test_dataframe_add_column_different_type() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3]);
        df.add_column(
            "col_string",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        assert_eq!(df.rows(), 3);
        assert_eq!(df.cols(), 3);
        assert_eq!(
            df.get_column_names(),
            &vec![
                "col_int".to_string(),
                "col_float".to_string(),
                "col_string".to_string()
            ]
        );

        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<String>()));

        assert_eq!(df.column_to_type.get("col_int"), Some(&TypeId::of::<i32>()));
        assert_eq!(
            df.column_to_type.get("col_float"),
            Some(&TypeId::of::<f64>())
        );
        assert_eq!(
            df.column_to_type.get("col_string"),
            Some(&TypeId::of::<String>())
        );

        // Verify underlying frames
        let int_frame = df
            .frames_by_type
            .get(&TypeId::of::<i32>())
            .unwrap()
            .as_any()
            .downcast_ref::<Frame<i32>>()
            .unwrap();
        assert_eq!(int_frame.columns(), &vec!["col_int".to_string()]);

        let float_frame = df
            .frames_by_type
            .get(&TypeId::of::<f64>())
            .unwrap()
            .as_any()
            .downcast_ref::<Frame<f64>>()
            .unwrap();
        assert_eq!(float_frame.columns(), &vec!["col_float".to_string()]);

        let string_frame = df
            .frames_by_type
            .get(&TypeId::of::<String>())
            .unwrap()
            .as_any()
            .downcast_ref::<Frame<String>>()
            .unwrap();
        assert_eq!(string_frame.columns(), &vec!["col_string".to_string()]);
    }

    #[test]
    fn test_dataframe_get_column() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3]);
        df.add_column(
            "col_string",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        // Test getting existing columns with correct type
        assert_eq!(
            df.get_column::<i32>("col_int").unwrap(),
            vec![1, 2, 3].as_slice()
        );
        assert_eq!(
            df.get_column::<f64>("col_float").unwrap(),
            vec![1.1, 2.2, 3.3].as_slice()
        );
        assert_eq!(
            df.get_column::<String>("col_string").unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()].as_slice()
        );

        // Test getting non-existent column
        assert_eq!(df.get_column::<i32>("non_existent"), None);

        // Test getting existing column with incorrect type
        assert_eq!(df.get_column::<f64>("col_int"), None);
        assert_eq!(df.get_column::<i32>("col_float"), None);
    }

    #[test]
    fn test_dataframe_get_row() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3]);
        df.add_column(
            "col_string",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        // Test getting an existing row
        let row0 = df.get_row(0).unwrap();
        assert_eq!(row0.get("col_int"), Some(&"1".to_string()));
        assert_eq!(row0.get("col_float"), Some(&"1.1".to_string()));
        assert_eq!(row0.get("col_string"), Some(&"a".to_string()));

        let row1 = df.get_row(1).unwrap();
        assert_eq!(row1.get("col_int"), Some(&"2".to_string()));
        assert_eq!(row1.get("col_float"), Some(&"2.2".to_string()));
        assert_eq!(row1.get("col_string"), Some(&"b".to_string()));

        // Test getting an out-of-bounds row
        assert_eq!(df.get_row(3), None);
    }

    #[test]
    #[should_panic(expected = "DataFrame::add_column: duplicate column name: 'col_int'")]
    fn test_dataframe_add_column_duplicate_name() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.add_column("col_int", vec![4, 5, 6]);
    }

    #[test]
    #[should_panic(
        expected = "DataFrame::add_column: new column 'col_int2' has 2 rows, but existing columns have 3 rows"
    )]
    fn test_dataframe_add_column_mismatched_rows() {
        let mut df = DataFrame::new();
        df.add_column("col_int1", vec![1, 2, 3]);
        df.add_column("col_int2", vec![4, 5]);
    }

    #[test]
    fn test_dataframe_display() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3, 4, 5, 6]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
        df.add_column(
            "col_string",
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
                "f".to_string(),
            ],
        );

        let expected_output = "\
col_int        col_float      col_string     
1              1.1            a              
2              2.2            b              
3              3.3            c              
4              4.4            d              
5              5.5            e              
...
";
        assert_eq!(format!("{}", df), expected_output);
    }

    #[test]
    fn test_dataframe_debug() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3]);

        let debug_output = format!("{:?}", df);
        assert!(debug_output.contains("DataFrame {"));
        assert!(debug_output.contains("column_names: [\"col_int\", \"col_float\"]"));
        assert!(debug_output.contains("index: Range(0..3)"));
        assert!(debug_output.contains("column_to_type: {"));
        assert!(debug_output.contains("frames_by_type: {"));
    }

    #[test]
    fn test_dataframe_drop_column_single_type() {
        let mut df = DataFrame::new();
        df.add_column("col_int1", vec![1, 2, 3]);
        df.add_column("col_int2", vec![4, 5, 6]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3]);

        assert_eq!(df.cols(), 3);
        assert_eq!(
            df.get_column_names(),
            &vec![
                "col_int1".to_string(),
                "col_int2".to_string(),
                "col_float".to_string()
            ]
        );
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));

        df.drop_column("col_int1");

        assert_eq!(df.cols(), 2);
        assert_eq!(
            df.get_column_names(),
            &vec!["col_int2".to_string(), "col_float".to_string()]
        );
        assert!(df.column_to_type.get("col_int1").is_none());
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>())); // Frame<i32> should still exist
        let int_frame = df
            .frames_by_type
            .get(&TypeId::of::<i32>())
            .unwrap()
            .as_any()
            .downcast_ref::<Frame<i32>>()
            .unwrap();
        assert_eq!(int_frame.columns(), &vec!["col_int2".to_string()]);

        df.drop_column("col_int2");

        assert_eq!(df.cols(), 1);
        assert_eq!(df.get_column_names(), &vec!["col_float".to_string()]);
        assert!(df.column_to_type.get("col_int2").is_none());
        assert!(!df.frames_by_type.contains_key(&TypeId::of::<i32>())); // Frame<i32> should be removed
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));
    }

    #[test]
    fn test_dataframe_drop_column_mixed_types() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.add_column("col_float", vec![1.1, 2.2, 3.3]);
        df.add_column(
            "col_string",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        assert_eq!(df.cols(), 3);
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<String>()));

        df.drop_column("col_float");

        assert_eq!(df.cols(), 2);
        assert_eq!(
            df.get_column_names(),
            &vec!["col_int".to_string(), "col_string".to_string()]
        );
        assert!(df.column_to_type.get("col_float").is_none());
        assert!(!df.frames_by_type.contains_key(&TypeId::of::<f64>())); // Frame<f64> should be removed
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<String>()));

        df.drop_column("col_int");
        df.drop_column("col_string");

        assert_eq!(df.cols(), 0);
        assert!(df.get_column_names().is_empty());
        assert!(df.frames_by_type.is_empty());
        assert!(df.column_to_type.is_empty());
    }

    #[test]
    #[should_panic(expected = "DataFrame::drop_column: column 'non_existent' not found")]
    fn test_dataframe_drop_column_non_existent() {
        let mut df = DataFrame::new();
        df.add_column("col_int", vec![1, 2, 3]);
        df.drop_column("non_existent");
    }

    #[test]
    fn test_dataframe_add_column_reuses_existing_frame() {
        let mut df = DataFrame::new();
        df.add_column("col_int1", vec![1, 2, 3]);
        df.add_column("col_float1", vec![1.1, 2.2, 3.3]);

        // Initially, there should be two frames (one for i32, one for f64)
        assert_eq!(df.frames_by_type.len(), 2);
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));

        // Add another integer column
        df.add_column("col_int2", vec![4, 5, 6]);

        // The number of frames should still be 2, as the existing i32 frame should be reused
        assert_eq!(df.frames_by_type.len(), 2);
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));

        // Verify the i32 frame now contains both integer columns
        let int_frame = df.frames_by_type.get(&TypeId::of::<i32>()).unwrap().as_any().downcast_ref::<Frame<i32>>().unwrap();
        assert_eq!(int_frame.columns(), &vec!["col_int1".to_string(), "col_int2".to_string()]);
        assert_eq!(int_frame.cols(), 2);

        // Add another float column
        df.add_column("col_float2", vec![4.4, 5.5, 6.6]);

        // The number of frames should still be 2, as the existing f64 frame should be reused
        assert_eq!(df.frames_by_type.len(), 2);
        assert!(df.frames_by_type.contains_key(&TypeId::of::<i32>()));
        assert!(df.frames_by_type.contains_key(&TypeId::of::<f64>()));

        // Verify the f64 frame now contains both float columns
        let float_frame = df.frames_by_type.get(&TypeId::of::<f64>()).unwrap().as_any().downcast_ref::<Frame<f64>>().unwrap();
        assert_eq!(float_frame.columns(), &vec!["col_float1".to_string(), "col_float2".to_string()]);
        assert_eq!(float_frame.cols(), 2);
    }
}
