# Data Manipulation

RustFrame's `Frame` type couples tabular data with
column labels and a typed row index.

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

```rust
# extern crate rustframe;
use rustframe::frame::{Frame, RowIndex};
use rustframe::matrix::Matrix;

let data = Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
let index = RowIndex::Int(vec![10, 20]);
let frame = Frame::new(data, vec!["A", "B"], Some(index));
assert_eq!(frame.get_row(20)["B"], 4.0);
```

## Aggregations

```rust
# extern crate rustframe;
use rustframe::frame::Frame;
use rustframe::matrix::{Matrix, SeriesOps};

let frame = Frame::new(Matrix::from_cols(vec![vec![1.0, 2.0]]), vec!["A"], None);
assert_eq!(frame.sum_vertical(), vec![3.0]);
```

When you're ready to run analytics, continue to the
[compute features](./compute.md) chapter.
