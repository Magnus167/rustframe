use rustframe::compute::stats::{mean, mean_horizontal, mean_vertical, median, percentile, stddev};
use rustframe::matrix::Matrix;

/// Demonstrates descriptive statistics utilities.
///
/// Part 1: simple mean/stddev/median/percentile on a vector.
/// Part 2: mean across rows and columns.
fn main() {
    simple_stats();
    println!("\n-----\n");
    axis_stats();
}

fn simple_stats() {
    println!("Basic stats\n-----------");
    let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
    println!("mean      : {:.2}", mean(&data));
    println!("stddev    : {:.2}", stddev(&data));
    println!("median    : {:.2}", median(&data));
    println!("90th pct. : {:.2}", percentile(&data, 90.0));
}

fn axis_stats() {
    println!("Row/column means\n----------------");
    // 2x3 matrix
    let data = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let v = mean_vertical(&data); // 1x3
    let h = mean_horizontal(&data); // 2x1
    println!("vertical means  : {:?}", v.data());
    println!("horizontal means: {:?}", h.data());
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-8;

    #[test]
    fn test_simple_stats() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
        assert!((mean(&data) - 3.0).abs() < EPS);
        assert!((stddev(&data) - 1.4142135623730951).abs() < EPS);
        assert!((median(&data) - 3.0).abs() < EPS);
        assert!((percentile(&data, 90.0) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_axis_stats() {
        let data = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let v = mean_vertical(&data);
        assert_eq!(v.data(), &[2.5, 3.5, 4.5]);
        let h = mean_horizontal(&data);
        assert_eq!(h.data(), &[2.0, 5.0]);
    }
}

