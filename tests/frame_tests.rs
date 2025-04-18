// --- PASTE THE FRAME STRUCT AND IMPL HERE ---
// #[derive(Debug, Clone, PartialEq, Eq)]
// pub struct Frame<T: Clone> { ... }
// impl<T: Clone> Frame<T> { ... }
// impl<T: Clone> Index<&str> for Frame<T> { ... }
// impl<T: Clone> IndexMut<&str> for Frame<T> { ... }
// macro_rules! impl_elementwise_frame_op { ... }
// impl_elementwise_frame_op!(Add, add, +);
// ... etc ...
// impl Not for Frame<bool> { ... }
// --- END OF FRAME CODE ---

// Unit Tests
#[cfg(test)]
mod tests {
    use rustframe::frame::*;
    use rustframe::matrix::*;

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

    // #[test]
    // fn test_swap_columns() {
    //     let mut frame = create_test_frame_i32();
    //     let initial_a_data = frame.column("A").to_vec();
    //     let initial_c_data = frame.column("C").to_vec();

    //     frame.swap_columns("A", "C");

    //     // Check names order
    //     assert_eq!(frame.column_names, vec!["C", "B", "A"]);

    //     // Check lookup map
    //     assert_eq!(frame.column_index("A"), Some(2));
    //     assert_eq!(frame.column_index("B"), Some(1));
    //     assert_eq!(frame.column_index("C"), Some(0));

    //     // Check data using new names (should be swapped)
    //     assert_eq!(frame.column("C"), initial_a_data); // "C" now has A's old data
    //     assert_eq!(frame.column("A"), initial_c_data); // "A" now has C's old data
    //     assert_eq!(frame.column("B"), &[4, 5, 6]); // "B" should be unchanged

    //     // Test swapping with self
    //     let state_before_self_swap = frame.clone();
    //     frame.swap_columns("B", "B");
    //     assert_eq!(frame, state_before_self_swap);
    // }

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
}
