
# <img align="center" alt="Rustframe" src=".github/rustframe_logo.png" height="50" /> rustframe

<!-- though the centre tag doesn't work as it would noramlly, it achieves the desired effect -->

📚 [Docs](https://magnus167.github.io/rustframe/) | 🐙 [GitHub](https://github.com/Magnus167/rustframe) | 🌐 [Gitea mirror](https://gitea.nulltech.uk/Magnus167/rustframe) | 🦀 [Crates.io](https://crates.io/crates/rustframe) | 🔖 [docs.rs](https://docs.rs/rustframe/latest/rustframe/)

<!-- [![Last commit](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/rustframe/last-commit-date.json)](https://github.com/Magnus167/rustframe) -->
[![codecov](https://codecov.io/gh/Magnus167/rustframe/graph/badge.svg?token=J7ULJEFTVI)](https://codecov.io/gh/Magnus167/rustframe)
[![Coverage](https://img.shields.io/endpoint?url=https://magnus167.github.io/rustframe/docs/tarpaulin-badge.json)](https://magnus167.github.io/rustframe/docs/tarpaulin-report.html)

---

## Rustframe: *A lightweight dataframe & math toolkit for Rust*

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
use rustframe::{
    dataframe::{DataFrame, TypedFrame, DataFrameColumn},
    frame::{Frame, RowIndex},
    matrix::Matrix,
};

// Helper to create a simple f64 TypedFrame (similar to test helpers)
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

fn main() {
    // 1. Create a DataFrame with different data types
    let col_a = create_f64_typed_frame("A", vec![1.0, 2.0, 3.0], None);
    let col_b = create_i64_typed_frame("B", vec![10, 20, 30], None);
    let col_c = create_string_typed_frame(
        "C",
        vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()],
        None,
    );

    let mut df = DataFrame::new(
        vec![col_a, col_b, col_c],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        None,
    );

    println!("Initial DataFrame:\n{:?}", df);
    println!("Columns: {:?}", df.columns());
    println!("Rows: {}", df.rows());

    // 2. Accessing columns
    if let DataFrameColumn::F64(col_a_data) = df.column("A") {
        println!("Column 'A' (f64): {:?}", col_a_data);
    }

    if let DataFrameColumn::String(col_c_data) = df.column("C") {
        println!("Column 'C' (String): {:?}", col_c_data);
    }

    // 3. Add a new column
    let new_col_d = create_f64_typed_frame("D", vec![100.0, 200.0, 300.0], None);
    df.add_column("D".to_string(), new_col_d);
    println!("\nDataFrame after adding column 'D':\n{:?}", df);
    println!("Columns after add: {:?}", df.columns());

    // 4. Rename a column
    df.rename_column("A", "Alpha".to_string());
    println!("\nDataFrame after renaming 'A' to 'Alpha':\n{:?}", df);
    println!("Columns after rename: {:?}", df.columns());

    // 5. Delete a column
    let _deleted_col_b = df.delete_column("B");
    println!("\nDataFrame after deleting column 'B':\n{:?}", df);
    println!("Columns after delete: {:?}", df.columns());
}
```
