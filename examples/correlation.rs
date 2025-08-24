use rustframe::compute::stats::{covariance, covariance_matrix, pearson};
use rustframe::matrix::{Axis, Matrix};

/// Demonstrates covariance and correlation utilities.
fn main() {
    pairwise_cov();
    println!("\n-----\n");
    matrix_cov();
}

fn pairwise_cov() {
    println!("Covariance & Pearson r\n----------------------");
    let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let y = Matrix::from_vec(vec![1.0, 2.0, 3.0, 5.0], 2, 2);
    println!("covariance : {:.2}", covariance(&x, &y));
    println!("pearson r  : {:.3}", pearson(&x, &y));
}

fn matrix_cov() {
    println!("Covariance matrix\n-----------------");
    let data = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let cov = covariance_matrix(&data, Axis::Col);
    println!("cov matrix : {:?}", cov.data());
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-8;

    #[test]
    fn test_pairwise_cov() {
        let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::from_vec(vec![1.0, 2.0, 3.0, 5.0], 2, 2);
        assert!((covariance(&x, &y) - 1.625).abs() < EPS);
        assert!((pearson(&x, &y) - 0.9827076298239908).abs() < 1e-5,);
    }

    #[test]
    fn test_matrix_cov() {
        let data = Matrix::from_rows_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let cov = covariance_matrix(&data, Axis::Col);
        assert_eq!(cov.data(), &[2.0, 2.0, 2.0, 2.0]);
    }
}
