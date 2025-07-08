use crate::matrix::{Matrix, SeriesOps};

use std::f64::consts::PI;

/// Approximation of the error function (Abramowitz & Stegun 7.1.26)
fn erf_func(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // coefficients
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Approximation of the error function for matrices
pub fn erf(x: Matrix<f64>) -> Matrix<f64> {
    x.map(|v| erf_func(v))
}

/// PDF of the Normal distribution
fn normal_pdf_func(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    (1.0 / (sd * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
}

/// PDF of the Normal distribution for matrices
pub fn normal_pdf(x: Matrix<f64>, mean: f64, sd: f64) -> Matrix<f64> {
    x.map(|v| normal_pdf_func(v, mean, sd))
}

/// CDF of the Normal distribution via erf
fn normal_cdf_func(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / (sd * 2.0_f64.sqrt());
    0.5 * (1.0 + erf_func(z))
}

/// CDF of the Normal distribution for matrices
pub fn normal_cdf(x: Matrix<f64>, mean: f64, sd: f64) -> Matrix<f64> {
    x.map(|v| normal_cdf_func(v, mean, sd))
}

/// PDF of the Uniform distribution on [a, b]
fn uniform_pdf_func(x: f64, a: f64, b: f64) -> f64 {
    if x < a || x > b {
        0.0
    } else {
        1.0 / (b - a)
    }
}

/// PDF of the Uniform distribution on [a, b] for matrices
pub fn uniform_pdf(x: Matrix<f64>, a: f64, b: f64) -> Matrix<f64> {
    x.map(|v| uniform_pdf_func(v, a, b))
}

/// CDF of the Uniform distribution on [a, b]
fn uniform_cdf_func(x: f64, a: f64, b: f64) -> f64 {
    if x < a {
        0.0
    } else if x <= b {
        (x - a) / (b - a)
    } else {
        1.0
    }
}

/// CDF of the Uniform distribution on [a, b] for matrices
pub fn uniform_cdf(x: Matrix<f64>, a: f64, b: f64) -> Matrix<f64> {
    x.map(|v| uniform_cdf_func(v, a, b))
}

/// Gamma Function (Lanczos approximation)
fn gamma_func(z: f64) -> f64 {
    // Lanczos coefficients
    let p: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        PI / ((PI * z).sin() * gamma_func(1.0 - z))
    } else {
        let z = z - 1.0;
        let mut x = 0.99999999999980993;
        for (i, &pi) in p.iter().enumerate() {
            x += pi / (z + (i as f64) + 1.0);
        }
        let t = z + p.len() as f64 - 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

pub fn gamma(z: Matrix<f64>) -> Matrix<f64> {
    z.map(|v| gamma_func(v))
}

/// Lower incomplete gamma via series expansion (for x < s+1)
fn lower_incomplete_gamma_func(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = sum;
    for n in 1..100 {
        term *= x / (s + n as f64);
        sum += term;
    }
    sum * x.powf(s) * (-x).exp()
}

/// Lower incomplete gamma for matrices
pub fn lower_incomplete_gamma(s: Matrix<f64>, x: Matrix<f64>) -> Matrix<f64> {
    s.zip(&x, |s_val, x_val| lower_incomplete_gamma_func(s_val, x_val))
}

/// PDF of the Gamma distribution (shape k, scale θ)
fn gamma_pdf_func(x: f64, k: f64, theta: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    let coef = 1.0 / (gamma_func(k) * theta.powf(k));
    coef * x.powf(k - 1.0) * (-(x / theta)).exp()
}

/// PDF of the Gamma distribution for matrices
pub fn gamma_pdf(x: Matrix<f64>, k: f64, theta: f64) -> Matrix<f64> {
    x.map(|v| gamma_pdf_func(v, k, theta))
}

/// CDF of the Gamma distribution via lower incomplete gamma
fn gamma_cdf_func(x: f64, k: f64, theta: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma_func(k, x / theta) / gamma_func(k)
}

/// CDF of the Gamma distribution for matrices
pub fn gamma_cdf(x: Matrix<f64>, k: f64, theta: f64) -> Matrix<f64> {
    x.map(|v| gamma_cdf_func(v, k, theta))
}

/// Factorials and Combinations ///

/// Compute n! as f64 (works up to ~170 reliably)
fn factorial(n: u64) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

/// Compute "n choose k" without overflow
fn binomial_coeff(n: u64, k: u64) -> f64 {
    let k = k.min(n - k);
    let mut numer = 1.0;
    let mut denom = 1.0;
    for i in 0..k {
        numer *= (n - i) as f64;
        denom *= (i + 1) as f64;
    }
    numer / denom
}

/// PMF of the Binomial(n, p) distribution
fn binomial_pmf_func(n: u64, k: u64, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    binomial_coeff(n, k) * p.powf(k as f64) * (1.0 - p).powf((n - k) as f64)
}

/// PMF of the Binomial(n, p) distribution for matrices
pub fn binomial_pmf(n: u64, k: Matrix<u64>, p: f64) -> Matrix<f64> {
    Matrix::from_vec(
        k.data()
            .iter()
            .map(|&v| binomial_pmf_func(n, v, p))
            .collect::<Vec<f64>>(),
        k.rows(),
        k.cols(),
    )
}

/// CDF of the Binomial(n, p) via summation
fn binomial_cdf_func(n: u64, k: u64, p: f64) -> f64 {
    (0..=k).map(|i| binomial_pmf_func(n, i, p)).sum()
}

/// CDF of the Binomial(n, p) for matrices
pub fn binomial_cdf(n: u64, k: Matrix<u64>, p: f64) -> Matrix<f64> {
    Matrix::from_vec(
        k.data()
            .iter()
            .map(|&v| binomial_cdf_func(n, v, p))
            .collect::<Vec<f64>>(),
        k.rows(),
        k.cols(),
    )
}

/// PMF of the Poisson(λ) distribution
fn poisson_pmf_func(lambda: f64, k: u64) -> f64 {
    lambda.powf(k as f64) * (-lambda).exp() / factorial(k)
}

/// PMF of the Poisson(λ) distribution for matrices
pub fn poisson_pmf(lambda: f64, k: Matrix<u64>) -> Matrix<f64> {
    Matrix::from_vec(
        k.data()
            .iter()
            .map(|&v| poisson_pmf_func(lambda, v))
            .collect::<Vec<f64>>(),
        k.rows(),
        k.cols(),
    )
}

/// CDF of the Poisson distribution via summation
fn poisson_cdf_func(lambda: f64, k: u64) -> f64 {
    (0..=k).map(|i| poisson_pmf_func(lambda, i)).sum()
}

/// CDF of the Poisson(λ) distribution for matrices
pub fn poisson_cdf(lambda: f64, k: Matrix<u64>) -> Matrix<f64> {
    Matrix::from_vec(
        k.data()
            .iter()
            .map(|&v| poisson_cdf_func(lambda, v))
            .collect::<Vec<f64>>(),
        k.rows(),
        k.cols(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_funcs() {
        // Test erf function
        assert!((erf_func(0.0) - 0.0).abs() < 1e-7);
        assert!((erf_func(1.0) - 0.8427007).abs() < 1e-7);
        assert!((erf_func(-1.0) + 0.8427007).abs() < 1e-7);

        // Test gamma function
        assert!((gamma_func(1.0) - 1.0).abs() < 1e-7);
        assert!((gamma_func(2.0) - 1.0).abs() < 1e-7);
        assert!((gamma_func(3.0) - 2.0).abs() < 1e-7);
        assert!((gamma_func(4.0) - 6.0).abs() < 1e-7);
        assert!((gamma_func(5.0) - 24.0).abs() < 1e-7);

        let z = 0.3;
        let expected = PI / ((PI * z).sin() * gamma_func(1.0 - z));
        assert!((gamma_func(z) - expected).abs() < 1e-7);
    }

    #[test]
    fn test_math_matrix() {
        let x = Matrix::filled(5, 5, 1.0);
        let erf_result = erf(x.clone());
        assert!((erf_result.data()[0] - 0.8427007).abs() < 1e-7);

        let gamma_result = gamma(x);
        assert!((gamma_result.data()[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_normal_funcs() {
        assert!((normal_pdf_func(0.0, 0.0, 1.0) - 0.39894228).abs() < 1e-7);
        assert!((normal_cdf_func(1.0, 0.0, 1.0) - 0.8413447).abs() < 1e-7);
    }

    #[test]
    fn test_normal_matrix() {
        let x = Matrix::filled(5, 5, 0.0);
        let pdf = normal_pdf(x.clone(), 0.0, 1.0);
        let cdf = normal_cdf(x, 0.0, 1.0);
        assert!((pdf.data()[0] - 0.39894228).abs() < 1e-7);
        assert!((cdf.data()[0] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_uniform_funcs() {
        assert_eq!(uniform_pdf_func(0.5, 0.0, 1.0), 1.0);
        assert_eq!(uniform_cdf_func(-1.0, 0.0, 1.0), 0.0);
        assert_eq!(uniform_cdf_func(0.5, 0.0, 1.0), 0.5);

        // x<a (or x>b) should return 0
        assert_eq!(uniform_pdf_func(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(uniform_pdf_func(1.5, 0.0, 1.0), 0.0);

        // for cdf x>a AND x>b should return 1
        assert_eq!(uniform_cdf_func(1.5, 0.0, 1.0), 1.0);
        assert_eq!(uniform_cdf_func(2.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_uniform_matrix() {
        let x = Matrix::filled(5, 5, 0.5);
        let pdf = uniform_pdf(x.clone(), 0.0, 1.0);
        let cdf = uniform_cdf(x, 0.0, 1.0);
        assert_eq!(pdf.data()[0], 1.0);
        assert_eq!(cdf.data()[0], 0.5);
    }

    #[test]
    fn test_binomial_funcs() {
        let pmf = binomial_pmf_func(5, 2, 0.5);
        assert!((pmf - 0.3125).abs() < 1e-7);
        let cdf = binomial_cdf_func(5, 2, 0.5);
        assert!((cdf - (0.03125 + 0.15625 + 0.3125)).abs() < 1e-7);

        let pmf_zero = binomial_pmf_func(5, 6, 0.5);
        assert!(pmf_zero == 0.0, "PMF should be 0 for k > n");
    }

    #[test]
    fn test_binomial_matrix() {
        let k = Matrix::filled(5, 5, 2 as u64);
        let pmf = binomial_pmf(5, k.clone(), 0.5);
        let cdf = binomial_cdf(5, k, 0.5);
        assert!((pmf.data()[0] - 0.3125).abs() < 1e-7);
        assert!((cdf.data()[0] - (0.03125 + 0.15625 + 0.3125)).abs() < 1e-7);
    }

    #[test]
    fn test_poisson_funcs() {
        let pmf: f64 = poisson_pmf_func(3.0, 2);
        assert!((pmf - (3.0_f64.powf(2.0) * (-3.0 as f64).exp() / 2.0)).abs() < 1e-7);
        let cdf: f64 = poisson_cdf_func(3.0, 2);
        assert!((cdf - (pmf + poisson_pmf_func(3.0, 0) + poisson_pmf_func(3.0, 1))).abs() < 1e-7);
    }

    #[test]
    fn test_poisson_matrix() {
        let k = Matrix::filled(5, 5, 2);
        let pmf = poisson_pmf(3.0, k.clone());
        let cdf = poisson_cdf(3.0, k);
        assert!((pmf.data()[0] - (3.0_f64.powf(2.0) * (-3.0 as f64).exp() / 2.0)).abs() < 1e-7);
        assert!(
            (cdf.data()[0] - (pmf.data()[0] + poisson_pmf_func(3.0, 0) + poisson_pmf_func(3.0, 1)))
                .abs()
                < 1e-7
        );
    }

    #[test]
    fn test_gamma_funcs() {
        // For k=1, θ=1 the Gamma(1,1) is Exp(1), so pdf(x)=e^-x
        assert!((gamma_pdf_func(2.0, 1.0, 1.0) - (-2.0 as f64).exp()).abs() < 1e-7);
        assert!((gamma_cdf_func(2.0, 1.0, 1.0) - (1.0 - (-2.0 as f64).exp())).abs() < 1e-7);

        // <0 case
        assert_eq!(gamma_pdf_func(-1.0, 1.0, 1.0), 0.0);
    }
    #[test]
    fn test_gamma_matrix() {
        let x = Matrix::filled(5, 5, 2.0);
        let pdf = gamma_pdf(x.clone(), 1.0, 1.0);
        let cdf = gamma_cdf(x, 1.0, 1.0);
        assert!((pdf.data()[0] - (-2.0 as f64).exp()).abs() < 1e-7);
        assert!((cdf.data()[0] - (1.0 - (-2.0 as f64).exp())).abs() < 1e-7);
    }

    #[test]
    fn test_lower_incomplete_gamma() {
        let s = Matrix::filled(5, 5, 2.0);
        let x = Matrix::filled(5, 5, 1.0);
        let expected = lower_incomplete_gamma_func(2.0, 1.0);
        let result = lower_incomplete_gamma(s, x);
        assert!((result.data()[0] - expected).abs() < 1e-7);
    }
}
