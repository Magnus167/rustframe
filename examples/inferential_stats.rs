use rustframe::compute::stats::{anova, chi2_test, t_test};
use rustframe::matrix::Matrix;

/// Demonstrates simple inferential statistics tests.
fn main() {
    t_test_demo();
    println!("\n-----\n");
    chi2_demo();
    println!("\n-----\n");
    anova_demo();
}

fn t_test_demo() {
    println!("Two-sample t-test\n-----------------");
    let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
    let b = Matrix::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
    let (t, p) = t_test(&a, &b);
    println!("t statistic: {:.2}, p-value: {:.4}", t, p);
}

fn chi2_demo() {
    println!("Chi-square test\n---------------");
    let observed = Matrix::from_vec(vec![12.0, 5.0, 8.0, 10.0], 2, 2);
    let (chi2, p) = chi2_test(&observed);
    println!("chi^2: {:.2}, p-value: {:.4}", chi2, p);
}

fn anova_demo() {
    println!("One-way ANOVA\n-------------");
    let g1 = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
    let g2 = Matrix::from_vec(vec![2.0, 3.0, 4.0], 1, 3);
    let g3 = Matrix::from_vec(vec![3.0, 4.0, 5.0], 1, 3);
    let (f, p) = anova(vec![&g1, &g2, &g3]);
    println!("F statistic: {:.2}, p-value: {:.4}", f, p);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_test_demo() {
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
        let b = Matrix::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
        let (t, _p) = t_test(&a, &b);
        assert!((t + 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_chi2_demo() {
        let observed = Matrix::from_vec(vec![12.0, 5.0, 8.0, 10.0], 2, 2);
        let (chi2, p) = chi2_test(&observed);
        assert!(chi2 > 0.0);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_anova_demo() {
        let g1 = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let g2 = Matrix::from_vec(vec![2.0, 3.0, 4.0], 1, 3);
        let g3 = Matrix::from_vec(vec![3.0, 4.0, 5.0], 1, 3);
        let (f, p) = anova(vec![&g1, &g2, &g3]);
        assert!(f > 0.0);
        assert!(p > 0.0 && p < 1.0);
    }
}
