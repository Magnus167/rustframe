# <img align="center" alt="Rustframe" src=".github/rustframe_logo.png" height="50" /> rustframe

<!-- though the centre tag doesn't work as it would noramlly, it achieves the desired effect -->

📚 [Docs](https://magnus167.github.io/rustframe/) | 🐙 [GitHub](https://github.com/Magnus167/rustframe) | 🌐 [Gitea mirror](https://gitea.nulltech.uk/Magnus167/rustframe) | 🦀 [Crates.io](https://crates.io/crates/rustframe) | 🔖 [docs.rs](https://docs.rs/rustframe/latest/rustframe/)

<!-- [![Last commit](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/rustframe/last-commit-date.json)](https://github.com/Magnus167/rustframe) -->

[![codecov](https://codecov.io/gh/Magnus167/rustframe/graph/badge.svg?token=J7ULJEFTVI)](https://codecov.io/gh/Magnus167/rustframe)
[![Coverage](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/docs/tarpaulin-badge.json)](https://magnus167.github.io/rustframe/docs/tarpaulin-report.html)

---

## Rustframe: _A lightweight dataframe & math toolkit for Rust_

Rustframe provides intuitive dataframe, matrix, and series operations small-to-mid scale data analysis and manipulation.

Rustframe keeps things simple, safe, and readable. It is handy for quick numeric experiments and small analytical tasks, but it is **not** meant to compete with powerhouse crates like `polars` or `ndarray`.

### What it offers

- **Math that reads like math** - element‑wise `+`, `−`, `×`, `÷` on entire frames or scalars.
- **Broadcast & reduce** - sum, product, any/all across rows or columns without boilerplate.
- **Boolean masks made simple** - chain comparisons, combine with `&`/`|`, get a tidy `BoolMatrix` back.
- **Date‑centric row index** - business‑day ranges and calendar slicing built in.
- **Pure safe Rust** - 100 % safe, zero `unsafe`.

### Heads up

- **Not memory‑efficient (yet)** - footprint needs work.
- **Feature set still small** - expect missing pieces.

### On the horizon

- Optional GPU help (Vulkan or similar) for heavier workloads.
- Straightforward Python bindings using `pyo3`.

---

## Quick start

```rust
use chrono::NaiveDate;
use rustframe::{
    frame::{Frame, RowIndex},
    matrix::{BoolOps, Matrix, SeriesOps},
    utils::{DateFreq, BDatesList},
};

let n_periods = 4;

// Four business days starting 2024‑01‑02
let dates: Vec<NaiveDate> =
    BDatesList::from_n_periods("2024-01-02".to_string(), DateFreq::Daily, n_periods)
        .unwrap()
        .list().unwrap();

let col_names: Vec<String> = vec!["a".to_string(), "b".to_string()];

let ma: Matrix<f64> =
    Matrix::from_cols(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
let mb: Matrix<f64> =
    Matrix::from_cols(vec![vec![4.0, 3.0, 2.0, 1.0], vec![8.0, 7.0, 6.0, 5.0]]);

let fa: Frame<f64> = Frame::new(
    ma.clone(),
    col_names.clone(),
    Some(RowIndex::Date(dates.clone())),
);
let fb: Frame<f64> = Frame::new(mb, col_names, Some(RowIndex::Date(dates)));

// Math that reads like math
let result: Frame<f64> = &fa * &fb; // element‑wise multiply
let total: f64 = result.sum_vertical().iter().sum::<f64>();
assert_eq!(total, 184.0);

// broadcast & reduce
let result: Matrix<f64> = ma.clone() + 1.0; // add scalar
let result: Matrix<f64> = result + &ma - &ma; // add matrix
let result: Matrix<f64> = result - 1.0; // subtract scalar
let result: Matrix<f64> = result * 2.0; // multiply by scalar
let result: Matrix<f64> = result / 2.0; // divide by scalar

let check: bool = result.eq_elem(ma.clone()).all();
assert!(check);

// The above math can also be written as:
let check: bool = (&(&(&(&ma + 1.0) - 1.0) * 2.0) / 2.0)
    .eq_elem(ma.clone())
    .all();
assert!(check);

// The above math can also be written as:
let check: bool = ((((ma.clone() + 1.0) - 1.0) * 2.0) / 2.0)
    .eq_elem(ma)
    .all();
assert!(check);


```
---

## DataFrame Usage Example

```rust
use rustframe::dataframe::DataFrame;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::any::TypeId; // Required for checking TypeId

// Helper for NaiveDate
fn d(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// Create a new DataFrame
let mut df = DataFrame::new();

// Add columns of different types
df.add_column("col_int1", vec![1, 2, 3, 4, 5]);
df.add_column("col_float1", vec![1.1, 2.2, 3.3, 4.4, 5.5]);
df.add_column("col_string", vec!["apple".to_string(), "banana".to_string(), "cherry".to_string(), "date".to_string(), "elderberry".to_string()]);
df.add_column("col_bool", vec![true, false, true, false, true]);
df.add_column("col_date", vec![d(2023,1,1), d(2023,1,2), d(2023,1,3), d(2023,1,4), d(2023,1,5)]);

println!("DataFrame after initial column additions:\n{}", df);

// Demonstrate frame re-use when adding columns of existing types
let initial_frames_count = df.num_internal_frames();
println!("\nInitial number of internal frames: {}", initial_frames_count);

df.add_column("col_int2", vec![6, 7, 8, 9, 10]);
df.add_column("col_float2", vec![6.6, 7.7, 8.8, 9.9, 10.0]);

let frames_after_reuse = df.num_internal_frames();
println!("Number of internal frames after adding more columns of existing types: {}", frames_after_reuse);
assert_eq!(initial_frames_count, frames_after_reuse); // Should be equal, demonstrating re-use

println!("\nDataFrame after adding more columns of existing types:\n{}", df);


// Get number of rows and columns
println!("Rows: {}", df.rows()); // Output: Rows: 5
println!("Columns: {}", df.cols()); // Output: Columns: 5

// Get column names
println!("Column names: {:?}", df.get_column_names());
// Output: Column names: ["col_int", "col_float", "col_string", "col_bool", "col_date"]

// Get a specific column by name and type
let int_col = df.get_column::<i32>("col_int1").unwrap();
println!("Integer column (col_int1): {:?}", int_col); // Output: Integer column: [1, 2, 3, 4, 5]

let int_col2 = df.get_column::<i32>("col_int2").unwrap();
println!("Integer column (col_int2): {:?}", int_col2); // Output: Integer column: [6, 7, 8, 9, 10]

let float_col = df.get_column::<f64>("col_float1").unwrap();
println!("Float column (col_float1): {:?}", float_col); // Output: Float column: [1.1, 2.2, 3.3, 4.4, 5.5]

// Attempt to get a column with incorrect type (returns None)
let wrong_type_col = df.get_column::<bool>("col_int1");
println!("Wrong type column: {:?}", wrong_type_col); // Output: Wrong type column: None

// Get a row by index
let row_0 = df.get_row(0).unwrap();
println!("Row 0: {:?}", row_0);
// Output: Row 0: {"col_int1": "1", "col_float1": "1.1", "col_string": "apple", "col_bool": "true", "col_date": "2023-01-01", "col_int2": "6", "col_float2": "6.6"}

let row_2 = df.get_row(2).unwrap();
println!("Row 2: {:?}", row_2);
// Output: Row 2: {"col_int1": "3", "col_float1": "3.3", "col_string": "cherry", "col_bool": "true", "col_date": "2023-01-03", "col_int2": "8", "col_float2": "8.8"}

// Attempt to get an out-of-bounds row (returns None)
let row_out_of_bounds = df.get_row(10);
println!("Row out of bounds: {:?}", row_out_of_bounds); // Output: Row out of bounds: None

// Drop a column
df.drop_column("col_bool");
println!("\nDataFrame after dropping 'col_bool':\n{}", df);

println!("Columns after drop: {}", df.cols());
println!("Column names after drop: {:?}", df.get_column_names());

// Drop another column, ensuring the underlying Frame is removed if empty
df.drop_column("col_float1");
println!("\nDataFrame after dropping 'col_float1':\n{}", df);

println!("Columns after second drop: {}", df.cols());
println!("Column names after second drop: {:?}", df.get_column_names());

// Attempt to drop a non-existent column (will panic)
// df.drop_column("non_existent_col"); // Uncomment to see panic
```

### More examples

See the [examples](./examples/) directory for some demonstrations of Rustframe's syntax and functionality.
