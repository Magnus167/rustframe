use crate::matrix::{Axis, BoolMatrix, FloatMatrix};

/// “Series–like” helpers that work along a single axis.
///
/// *All* the old methods (`sum_*`, `prod_*`, `is_nan`, …) are exposed
/// through this trait, so nothing needs to stay on an `impl Matrix<f64>`;
/// just `use SeriesOps` to make the extension methods available.
pub trait SeriesOps {
    /// Generic helper: apply `f` to every column/row and collect its
    /// result in a `Vec`.
    fn apply_axis<U, F>(&self, axis: Axis, f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U;

    fn sum_vertical(&self) -> Vec<f64>;
    fn sum_horizontal(&self) -> Vec<f64>;

    fn prod_vertical(&self) -> Vec<f64>;
    fn prod_horizontal(&self) -> Vec<f64>;

    fn cumsum_vertical(&self) -> FloatMatrix;
    fn cumsum_horizontal(&self) -> FloatMatrix;

    fn count_nan_vertical(&self) -> Vec<usize>;
    fn count_nan_horizontal(&self) -> Vec<usize>;

    fn is_nan(&self) -> BoolMatrix;
}

impl SeriesOps for FloatMatrix {
    fn apply_axis<U, F>(&self, axis: Axis, mut f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U,
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
                let mut buf = vec![0.0; self.cols()]; // reusable buffer
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

    fn sum_vertical(&self) -> Vec<f64> {
        self.apply_axis(Axis::Col, |col| {
            col.iter().copied().filter(|v| !v.is_nan()).sum::<f64>()
        })
    }

    fn sum_horizontal(&self) -> Vec<f64> {
        self.apply_axis(Axis::Row, |row| {
            row.iter().copied().filter(|v| !v.is_nan()).sum::<f64>()
        })
    }

    fn prod_vertical(&self) -> Vec<f64> {
        self.apply_axis(Axis::Col, |col| {
            col.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(1.0, |acc, x| acc * x)
        })
    }

    fn prod_horizontal(&self) -> Vec<f64> {
        self.apply_axis(Axis::Row, |row| {
            row.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(1.0, |acc, x| acc * x)
        })
    }

    fn cumsum_vertical(&self) -> FloatMatrix {
        let mut data = Vec::with_capacity(self.rows() * self.cols());
        for c in 0..self.cols() {
            let mut acc = 0.0;
            for r in 0..self.rows() {
                let v = self[(r, c)];
                if !v.is_nan() {
                    acc += v;
                }
                data.push(acc);
            }
        }
        FloatMatrix::from_vec(data, self.rows(), self.cols())
    }

    fn cumsum_horizontal(&self) -> FloatMatrix {
        // 1. Store row-wise cumulative sums temporarily
        let mut row_results: Vec<Vec<f64>> = Vec::with_capacity(self.rows());
        for r in 0..self.rows() {
            let mut row_data = Vec::with_capacity(self.cols());
            let mut acc = 0.0;
            for c in 0..self.cols() {
                let v = self[(r, c)];
                if !v.is_nan() {
                    acc += v;
                }
                row_data.push(acc);
            }
            row_results.push(row_data);
        }

        // 2. Build the final data vector in column-major order
        let mut final_data = Vec::with_capacity(self.rows() * self.cols());
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                // Get the element from row 'r', column 'c' of the row_results
                final_data.push(row_results[r][c]);
            }
        }

        // 3. Construct the matrix using the correctly ordered data
        FloatMatrix::from_vec(final_data, self.rows(), self.cols())
    }

    fn count_nan_vertical(&self) -> Vec<usize> {
        self.apply_axis(Axis::Col, |col| col.iter().filter(|x| x.is_nan()).count())
    }

    fn count_nan_horizontal(&self) -> Vec<usize> {
        self.apply_axis(Axis::Row, |row| row.iter().filter(|x| x.is_nan()).count())
    }

    fn is_nan(&self) -> BoolMatrix {
        let data = self.data().iter().map(|v| v.is_nan()).collect();
        BoolMatrix::from_vec(data, self.rows(), self.cols())
    }
}
