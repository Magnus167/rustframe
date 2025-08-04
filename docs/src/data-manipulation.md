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

With the basics covered, continue to the [compute features](./compute.md)
chapter for statistics and analytics.
