use rustframe::compute::stats::{binomial_cdf, binomial_pmf, normal_cdf, normal_pdf, poisson_pmf};
use rustframe::matrix::Matrix;

/// Demonstrates some probability distribution helpers.
fn main() {
    normal_example();
    println!("\n-----\n");
    binomial_example();
    println!("\n-----\n");
    poisson_example();
}

fn normal_example() {
    println!("Normal distribution\n-------------------");
    let x = Matrix::from_vec(vec![0.0, 1.0], 1, 2);
    let pdf = normal_pdf(x.clone(), 0.0, 1.0);
    let cdf = normal_cdf(x, 0.0, 1.0);
    println!("pdf : {:?}", pdf.data());
    println!("cdf : {:?}", cdf.data());
}

fn binomial_example() {
    println!("Binomial distribution\n---------------------");
    let k = Matrix::from_vec(vec![0_u64, 1, 2], 1, 3);
    let pmf = binomial_pmf(4, k.clone(), 0.5);
    let cdf = binomial_cdf(4, k, 0.5);
    println!("pmf : {:?}", pmf.data());
    println!("cdf : {:?}", cdf.data());
}

fn poisson_example() {
    println!("Poisson distribution\n--------------------");
    let k = Matrix::from_vec(vec![0_u64, 1, 2], 1, 3);
    let pmf = poisson_pmf(3.0, k);
    println!("pmf : {:?}", pmf.data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_example() {
        let x = Matrix::from_vec(vec![0.0, 1.0], 1, 2);
        let pdf = normal_pdf(x.clone(), 0.0, 1.0);
        let cdf = normal_cdf(x, 0.0, 1.0);
        assert!((pdf.get(0, 0) - 0.39894228).abs() < 1e-6);
        assert!((cdf.get(0, 1) - 0.8413447).abs() < 1e-6);
    }

    #[test]
    fn test_binomial_example() {
        let k = Matrix::from_vec(vec![0_u64, 1, 2], 1, 3);
        let pmf = binomial_pmf(4, k.clone(), 0.5);
        let cdf = binomial_cdf(4, k, 0.5);
        assert!((pmf.get(0, 2) - 0.375).abs() < 1e-6);
        assert!((cdf.get(0, 2) - 0.6875).abs() < 1e-6);
    }

    #[test]
    fn test_poisson_example() {
        let k = Matrix::from_vec(vec![0_u64, 1, 2], 1, 3);
        let pmf = poisson_pmf(3.0, k);
        assert!((pmf.get(0, 1) - 3.0_f64 * (-3.0_f64).exp()).abs() < 1e-6);
    }
}
