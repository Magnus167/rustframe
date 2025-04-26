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
