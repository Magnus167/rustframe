# Data Manipulation

Rustframe's `Frame` type couples tabular data with
column labels and a typed row index. Frames expose a familiar API for loading
data, selecting rows or columns and performing aggregations.

## Creating a Frame

```rust
# extern crate rustframe;
use rustframe::frame::{Frame, RowIndex};
use rustframe::matrix::Matrix;

let data = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
let frame = Frame::new(data, vec!["A", "B"], None);
assert_eq!(frame["A"], vec![1.0, 2.0]);
```

## Indexing Rows

Row labels can be integers, dates or a default range. Retrieving a row returns a
view that lets you inspect values by column name or position.

```rust
# extern crate rustframe;
# extern crate chrono;
use chrono::NaiveDate;
use rustframe::frame::{Frame, RowIndex};
use rustframe::matrix::Matrix;

let d = |y, m, d| NaiveDate::from_ymd_opt(y, m, d).unwrap();
let data = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
let index = RowIndex::Date(vec![d(2024, 1, 1), d(2024, 1, 2)]);
let mut frame = Frame::new(data, vec!["A", "B"], Some(index));
assert_eq!(frame.get_row_date(d(2024, 1, 2))["B"], 4.0);

// mutate by row key
frame.get_row_date_mut(d(2024, 1, 1)).set_by_index(0, 9.0);
assert_eq!(frame.get_row_date(d(2024, 1, 1))["A"], 9.0);
```

## Column operations

Columns can be inserted, renamed, removed or reordered in place.

```rust
# extern crate rustframe;
use rustframe::frame::{Frame, RowIndex};
use rustframe::matrix::Matrix;

let data = Matrix::from_cols(vec![vec![1, 2], vec![3, 4]]);
let mut frame = Frame::new(data, vec!["X", "Y"], Some(RowIndex::Range(0..2)));

frame.add_column("Z", vec![5, 6]);
frame.rename("Y", "W");
let removed = frame.delete_column("X");
assert_eq!(removed, vec![1, 2]);
frame.sort_columns();
assert_eq!(frame.columns(), &["W", "Z"]);
```

## Aggregations

Any numeric aggregation available on `Matrix` is forwarded to `Frame`.

```rust
# extern crate rustframe;
use rustframe::frame::Frame;
use rustframe::matrix::{Matrix, SeriesOps};

let frame = Frame::new(Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]), vec!["A", "B"], None);
assert_eq!(frame.sum_vertical(), vec![3.0, 7.0]);
assert_eq!(frame.sum_horizontal(), vec![4.0, 6.0]);
```

## Matrix Operations

```rust
# extern crate rustframe;
use rustframe::matrix::Matrix;

let data1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let data2 = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);

let sum = data1.clone() + data2.clone();
assert_eq!(sum.data(), vec![6.0, 8.0, 10.0, 12.0]);

let product = data1.clone() * data2.clone();
assert_eq!(product.data(), vec![5.0, 12.0, 21.0, 32.0]);

let scalar_product = data1.clone() * 2.0;
assert_eq!(scalar_product.data(), vec![2.0, 4.0, 6.0, 8.0]);

let equals = data1 == data1.clone();
assert_eq!(equals, true);
```

### Advanced Matrix Operations

Matrices support a variety of advanced operations:

```rust
# extern crate rustframe;
use rustframe::matrix::{Matrix, SeriesOps};

// Matrix multiplication (dot product)
let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
let product = a.matrix_mul(&b);
assert_eq!(product.data(), vec![23.0, 34.0, 31.0, 46.0]);

// Transpose
let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let transposed = m.transpose();
assert_eq!(transposed.data(), vec![1.0, 3.0, 2.0, 4.0]);

// Map function over all elements
let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let squared = m.map(|x| x * x);
assert_eq!(squared.data(), vec![1.0, 4.0, 9.0, 16.0]);

// Zip two matrices with a function
let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
let zipped = a.zip(&b, |x, y| x + y);
assert_eq!(zipped.data(), vec![6.0, 8.0, 10.0, 12.0]);
```

### Matrix Reductions

Matrices support various reduction operations:

```rust
# extern crate rustframe;
use rustframe::matrix::{Matrix, SeriesOps};

let m = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

// Sum along columns (vertical)
let col_sums = m.sum_vertical();
assert_eq!(col_sums, vec![9.0, 12.0]); // [1+3+5, 2+4+6]

// Sum along rows (horizontal)
let row_sums = m.sum_horizontal();
assert_eq!(row_sums, vec![3.0, 7.0, 11.0]); // [1+2, 3+4, 5+6]

// Cumulative sum along columns
let col_cumsum = m.cumsum_vertical();
assert_eq!(col_cumsum.data(), vec![1.0, 4.0, 9.0, 2.0, 6.0, 12.0]);

// Cumulative sum along rows
let row_cumsum = m.cumsum_horizontal();
assert_eq!(row_cumsum.data(), vec![1.0, 3.0, 5.0, 3.0, 7.0, 11.0]);
```

With the basics covered, continue to the [compute features](./compute.md)
chapter for statistics and analytics.
