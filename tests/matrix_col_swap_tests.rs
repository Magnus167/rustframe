#[cfg(test)]
mod tests {
    use rustframe::frame::*;
    use rustframe::matrix::*;
    // Or explicitly: use crate::matrix::Matrix;

    // --- Include your other tests here ---

    /// Creates a standard 3x3 matrix used in several tests.
    /// Column 0: [1, 2, 3]
    /// Column 1: [4, 5, 6]
    /// Column 2: [7, 8, 9]
    fn create_test_matrix_i32() -> Matrix<i32> {
        Matrix::from_cols(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
    }

    // --- The new test ---
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

    // --- Include your failing Frame test_swap_columns here as well ---
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

    fn create_test_frame_i32() -> Frame<i32> {
        // Ensure this uses the same logic/data as create_test_matrix_i32
        let matrix = create_test_matrix_i32();
        Frame::new(matrix, vec!["A", "B", "C"])
    }
} // end mod tests
