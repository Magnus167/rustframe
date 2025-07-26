use rustframe::compute::stats::{
    chi2_test, covariance, covariance_matrix, mean, median, pearson, percentile, stddev, t_test,
};
use rustframe::matrix::{Axis, Matrix};

/// Demonstrates some of the statistics utilities in Rustframe.
///
/// The example is split into three parts:
///   1. Basic descriptive statistics on a small data set.
///   2. Covariance and correlation calculations.
///   3. Simple inferential tests (t-test and chi-square).
fn main() {
    descriptive_demo();
    println!("\n-----\n");
    correlation_demo();
    println!("\n-----\n");
    inferential_demo();
}

fn descriptive_demo() {
    println!("Descriptive statistics\n----------------------");
    let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
    println!("mean           : {:.2}", mean(&data));
    println!("std dev        : {:.2}", stddev(&data));
    println!("median         : {:.2}", median(&data));
    println!("25th percentile: {:.2}", percentile(&data, 25.0));
}

fn correlation_demo() {
    println!("Covariance and Correlation\n--------------------------");
    let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let y = Matrix::from_vec(vec![1.0, 2.0, 3.0, 5.0], 2, 2);
    let cov = covariance(&x, &y);
    let cov_mat = covariance_matrix(&x, Axis::Col);
    let corr = pearson(&x, &y);
    println!("covariance : {:.2}", cov);
    println!("cov matrix : {:?}", cov_mat.data());
    println!("pearson r  : {:.2}", corr);
}

fn inferential_demo() {
    println!("Inferential statistics\n----------------------");
    let s1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
    let s2 = Matrix::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
    let (t_stat, t_p) = t_test(&s1, &s2);
    println!("t statistic : {:.2}, p-value: {:.4}", t_stat, t_p);

    let observed = Matrix::from_vec(vec![12.0, 5.0, 8.0, 10.0], 2, 2);
    let (chi2, chi_p) = chi2_test(&observed);
    println!("chi^2       : {:.2}, p-value: {:.4}", chi2, chi_p);
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-8;

    #[test]
    fn test_descriptive_demo() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
        assert!((mean(&data) - 3.0).abs() < EPS);
        assert!((stddev(&data) - 1.4142135623730951).abs() < EPS);
        assert!((median(&data) - 3.0).abs() < EPS);
        assert!((percentile(&data, 25.0) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_correlation_demo() {
        let x = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 5.0], 2, 2);
        let cov = covariance(&x, &y);
        assert!((cov - 1.625).abs() < EPS);
        let cov_mat = covariance_matrix(&x, Axis::Col);
        assert!((cov_mat.get(0, 0) - 2.0).abs() < EPS);
        assert!((cov_mat.get(1, 1) - 2.0).abs() < EPS);
        let corr = pearson(&x, &y);
        assert!((corr - 0.9827076298239908).abs() < 1e-6);
    }

    #[test]
    fn test_inferential_demo() {
        let s1 = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
        let s2 = Matrix::from_rows_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
        let (t_stat, p_value) = t_test(&s1, &s2);
        assert!((t_stat + 5.0).abs() < 1e-5);
        assert!(p_value > 0.0 && p_value < 1.0);

        let observed = Matrix::from_rows_vec(vec![12.0, 5.0, 8.0, 10.0], 2, 2);
        let (chi2, p) = chi2_test(&observed);
        assert!(chi2 > 0.0);
        assert!(p > 0.0 && p < 1.0);
    }
}
