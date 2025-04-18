use crate::matrix::{Axis, BoolMatrix};

/// Boolean operations on `Matrix<bool>`
pub trait BoolOps {
    /// Generic helper: apply `f` to every column/row and collect its
    /// result in a `Vec`.
    fn apply_axis<U, F>(&self, axis: Axis, f: F) -> Vec<U>
    where
        F: FnMut(&[bool]) -> U;

    fn any_vertical(&self) -> Vec<bool>;
    fn any_horizontal(&self) -> Vec<bool>;
    fn all_vertical(&self) -> Vec<bool>;
    fn all_horizontal(&self) -> Vec<bool>;
    fn count_vertical(&self) -> Vec<usize>;
    fn count_horizontal(&self) -> Vec<usize>;
    fn any(&self) -> bool;
    fn all(&self) -> bool;
    fn count(&self) -> usize;
}

impl BoolOps for BoolMatrix {
    fn apply_axis<U, F>(&self, axis: Axis, mut f: F) -> Vec<U>
    where
        F: FnMut(&[bool]) -> U,
    {
        match axis {
            Axis::Col => {
                let mut out = Vec::with_capacity(self.cols());
                for c in 0..self.cols() {
                    out.push(f(self.column(c)));
                }
                out
            }
            Axis::Row => {
                let mut out = Vec::with_capacity(self.rows());
                let mut buf = vec![false; self.cols()]; // reusable buffer
                for r in 0..self.rows() {
                    for c in 0..self.cols() {
                        buf[c] = self[(r, c)];
                    }
                    out.push(f(&buf));
                }
                out
            }
        }
    }
    fn any_vertical(&self) -> Vec<bool> {
        self.apply_axis(Axis::Col, |col| col.iter().any(|&v| v))
    }
    fn any_horizontal(&self) -> Vec<bool> {
        self.apply_axis(Axis::Row, |row| row.iter().any(|&v| v))
    }
    fn all_vertical(&self) -> Vec<bool> {
        self.apply_axis(Axis::Col, |col| col.iter().all(|&v| v))
    }
    fn all_horizontal(&self) -> Vec<bool> {
        self.apply_axis(Axis::Row, |row| row.iter().all(|&v| v))
    }
    fn count_vertical(&self) -> Vec<usize> {
        self.apply_axis(Axis::Col, |col| col.iter().filter(|&&v| v).count())
    }
    fn count_horizontal(&self) -> Vec<usize> {
        self.apply_axis(Axis::Row, |row| row.iter().filter(|&&v| v).count())
    }
    fn any(&self) -> bool {
        self.data().iter().any(|&v| v)
    }
    fn all(&self) -> bool {
        self.data().iter().all(|&v| v)
    }
    fn count(&self) -> usize {
        self.data().iter().filter(|&&v| v).count()
    }
}
// use macros to generate the implementations for BitAnd, BitOr, BitXor, and Not
