use crate::frame::Frame;
use crate::matrix::{Axis, BoolMatrix, BoolOps, FloatMatrix, SeriesOps};

// Macro to delegate method calls to self.matrix()
macro_rules! delegate_to_matrix {
    ($($method_name:ident -> $return_type:ty),* $(,)?) => {
        $(
            fn $method_name(&self) -> $return_type {
                self.matrix().$method_name()
            }
        )*
    };
}

impl SeriesOps for Frame<f64> {
    #[allow(unused_mut)]
    fn apply_axis<U, F>(&self, axis: Axis, mut f: F) -> Vec<U>
    where
        F: FnMut(&[f64]) -> U,
    {
        self.matrix().apply_axis(axis, f)
    }

    fn map<F>(&self, f: F) -> FloatMatrix
    where
        F: Fn(f64) -> f64,
    {
        self.matrix().map(f)
    }

    fn zip<F>(&self, other: &Self, f: F) -> FloatMatrix
    where
        F: Fn(f64, f64) -> f64,
    {
        self.matrix().zip(other.matrix(), f)
    }

    fn matrix_mul(&self, other: &Self) -> FloatMatrix {
        self.matrix().matrix_mul(other.matrix())
    }

    fn dot(&self, other: &Self) -> FloatMatrix {
        self.matrix().dot(other.matrix())
    }

    delegate_to_matrix!(
        sum_vertical -> Vec<f64>,
        sum_horizontal -> Vec<f64>,
        prod_horizontal -> Vec<f64>,
        prod_vertical -> Vec<f64>,
        cumsum_horizontal -> FloatMatrix,
        cumsum_vertical -> FloatMatrix,
        count_nan_vertical -> Vec<usize>,
        count_nan_horizontal -> Vec<usize>,
        is_nan -> BoolMatrix
    );
}

impl BoolOps for Frame<bool> {
    fn apply_axis<U, F>(&self, axis: Axis, f: F) -> Vec<U>
    where
        F: FnMut(&[bool]) -> U,
    {
        self.matrix().apply_axis(axis, f)
    }

    delegate_to_matrix!(
        any_vertical -> Vec<bool>,
        any_horizontal -> Vec<bool>,
        all_vertical -> Vec<bool>,
        all_horizontal -> Vec<bool>,
        count_vertical -> Vec<usize>,
        count_horizontal -> Vec<usize>,
        any -> bool,
        all -> bool,
        count -> usize
    );
}

// use crate::frame::Frame;
// use crate::matrix::{Axis, SeriesOps, FloatMatrix, BoolMatrix};

// impl SeriesOps for Frame<f64> {
//     fn apply_axis<U, F>(&self, axis: Axis, mut f: F) -> Vec<U>
//     where
//         F: FnMut(&[f64]) -> U,
//     {
//         self.matrix().apply_axis(axis, f)
//     }

//     fn sum_vertical(&self) -> Vec<f64> {
//         self.matrix().sum_vertical()
//     }
//     fn sum_horizontal(&self) -> Vec<f64> {
//         self.matrix().sum_horizontal()
//     }
//     fn prod_horizontal(&self) -> Vec<f64> {
//         self.matrix().prod_horizontal()
//     }
//     fn prod_vertical(&self) -> Vec<f64> {
//         self.matrix().prod_vertical()
//     }
//     fn cumsum_horizontal(&self) -> FloatMatrix {
//         self.matrix().cumsum_horizontal()
//     }
//     fn cumsum_vertical(&self) -> FloatMatrix {
//         self.matrix().cumsum_vertical()
//     }

//     fn count_nan_vertical(&self) -> Vec<usize> {
//         self.matrix().count_nan_vertical()
//     }
//     fn count_nan_horizontal(&self) -> Vec<usize> {
//         self.matrix().count_nan_horizontal()
//     }
//     fn is_nan(&self) -> BoolMatrix {
//         self.matrix().is_nan()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn test_series_ops() {
        let col_names = vec!["A".to_string(), "B".to_string()];
        let frame = Frame::new(
            Matrix::from_cols(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            col_names.clone(),
            None,
        );
        assert_eq!(frame.sum_vertical(), frame.matrix().sum_vertical());
        assert_eq!(frame.sum_horizontal(), frame.matrix().sum_horizontal());
        assert_eq!(frame.prod_horizontal(), frame.matrix().prod_horizontal());
        assert_eq!(frame.prod_vertical(), frame.matrix().prod_vertical());
        assert_eq!(
            frame.cumsum_horizontal(),
            frame.matrix().cumsum_horizontal()
        );
        assert_eq!(frame.cumsum_vertical(), frame.matrix().cumsum_vertical());
        assert_eq!(
            frame.count_nan_vertical(),
            frame.matrix().count_nan_vertical()
        );
        assert_eq!(
            frame.count_nan_horizontal(),
            frame.matrix().count_nan_horizontal()
        );
        assert_eq!(frame.is_nan(), frame.matrix().is_nan());
        assert_eq!(frame.apply_axis(Axis::Row, |x| x[0] + x[1]), vec![4.0, 6.0]);

        assert_eq!(
            frame.matrix_mul(&frame),
            frame.matrix().matrix_mul(&frame.matrix())
        );
        assert_eq!(frame.dot(&frame), frame.matrix().dot(&frame.matrix()));

        // test transpose - returns a matrix.
        let frame_transposed_mat = frame.transpose();
        let frame_mat_transposed = frame.matrix().transpose();
        assert_eq!(frame_transposed_mat, frame_mat_transposed);
        assert_eq!(frame.matrix(), &frame.matrix().transpose().transpose());

        // test map
        let mapped_frame = frame.map(|x| x * 2.0);
        let expected_matrix = frame.matrix().map(|x| x * 2.0);
        assert_eq!(mapped_frame, expected_matrix);

        // test zip
        let other_frame = Frame::new(
            Matrix::from_cols(vec![vec![5.0, 6.0], vec![7.0, 8.0]]),
            col_names.clone(),
            None,
        );
        let zipped_frame = frame.zip(&other_frame, |x, y| x + y);
        let expected_zipped_matrix = frame.matrix().zip(other_frame.matrix(), |x, y| x + y);
        assert_eq!(zipped_frame, expected_zipped_matrix);
    }
    #[test]

    fn test_bool_ops() {
        let col_names = vec!["A".to_string(), "B".to_string()];
        let frame = Frame::new(
            Matrix::from_cols(vec![vec![true, false], vec![false, true]]),
            col_names,
            None,
        );
        assert_eq!(frame.any_vertical(), vec![true, true]);
        assert_eq!(frame.any_horizontal(), vec![true, true]);
        assert_eq!(frame.all_horizontal(), vec![false, false]);
        assert_eq!(frame.all_vertical(), vec![false, false]);
        assert_eq!(frame.count_vertical(), vec![1, 1]);
        assert_eq!(frame.count_horizontal(), vec![1, 1]);
        assert_eq!(frame.any(), true);
        assert_eq!(frame.all(), false);
        assert_eq!(frame.count(), 2);
        assert_eq!(
            frame.apply_axis(Axis::Row, |x| x[0] && x[1]),
            vec![false, false]
        );
    }
}
