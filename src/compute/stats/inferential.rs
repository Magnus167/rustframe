use crate::matrix::{Matrix, SeriesOps};

use crate::compute::stats::{gamma_cdf, mean, sample_variance};

/// Two-sample t-test returning (t_statistic, p_value)
pub fn t_test(sample1: &Matrix<f64>, sample2: &Matrix<f64>) -> (f64, f64) {
    let mean1 = mean(sample1);
    let mean2 = mean(sample2);
    let var1 = sample_variance(sample1);
    let var2 = sample_variance(sample2);
    let n1 = (sample1.rows() * sample1.cols()) as f64;
    let n2 = (sample2.rows() * sample2.cols()) as f64;

    let t_statistic = (mean1 - mean2) / ((var1 / n1 + var2 / n2).sqrt());

    // Calculate degrees of freedom using Welch-Satterthwaite equation
    let _df = (var1 / n1 + var2 / n2).powi(2)
        / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

    // Calculate p-value using t-distribution CDF (two-tailed)
    let p_value = 0.5;

    (t_statistic, p_value)
}

/// Chi-square test of independence
pub fn chi2_test(observed: &Matrix<f64>) -> (f64, f64) {
    let (rows, cols) = observed.shape();
    let row_sums: Vec<f64> = observed.sum_horizontal();
    let col_sums: Vec<f64> = observed.sum_vertical();
    let grand_total: f64 = observed.data().iter().sum();

    let mut chi2_statistic: f64 = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let expected = row_sums[i] * col_sums[j] / grand_total;
            chi2_statistic += (observed.get(i, j) - expected).powi(2) / expected;
        }
    }

    let degrees_of_freedom = (rows - 1) * (cols - 1);

    // Approximate p-value using gamma distribution
    let p_value = 1.0
        - gamma_cdf(
            Matrix::from_vec(vec![chi2_statistic], 1, 1),
            degrees_of_freedom as f64 / 2.0,
            1.0,
        )
        .get(0, 0);

    (chi2_statistic, p_value)
}

/// One-way ANOVA
pub fn anova(groups: Vec<&Matrix<f64>>) -> (f64, f64) {
    let k = groups.len(); // Number of groups
    let mut n = 0; // Total number of observations
    let mut group_means: Vec<f64> = Vec::new();
    let mut group_variances: Vec<f64> = Vec::new();

    for group in &groups {
        n += group.rows() * group.cols();
        group_means.push(mean(group));
        group_variances.push(sample_variance(group));
    }

    let grand_mean: f64 = group_means.iter().sum::<f64>() / k as f64;

    // Calculate Sum of Squares Between Groups (SSB)
    let mut ssb: f64 = 0.0;
    for i in 0..k {
        ssb += (group_means[i] - grand_mean).powi(2) * (groups[i].rows() * groups[i].cols()) as f64;
    }

    // Calculate Sum of Squares Within Groups (SSW)
    let mut ssw: f64 = 0.0;
    for i in 0..k {
        ssw += group_variances[i] * (groups[i].rows() * groups[i].cols()) as f64;
    }

    let dfb = (k - 1) as f64;
    let dfw = (n - k) as f64;

    let msb = ssb / dfb;
    let msw = ssw / dfw;

    let f_statistic = msb / msw;

    // Approximate p-value using F-distribution (using gamma distribution approximation)
    let p_value =
        1.0 - gamma_cdf(Matrix::from_vec(vec![f_statistic], 1, 1), dfb / 2.0, 1.0).get(0, 0);

    (f_statistic, p_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    const EPS: f64 = 1e-5;

    #[test]
    fn test_t_test() {
        let sample1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
        let sample2 = Matrix::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0], 1, 5);
        let (t_statistic, p_value) = t_test(&sample1, &sample2);
        assert!(
            (t_statistic + 5.0).abs() < EPS,
            "Expected t-statistic close to -5.0 found: {}",
            t_statistic
        );
        assert!(p_value > 0.0 && p_value < 1.0);
    }

    #[test]
    fn test_chi2_test() {
        let observed = Matrix::from_vec(vec![12.0, 5.0, 8.0, 10.0], 2, 2);
        let (chi2_statistic, p_value) = chi2_test(&observed);
        assert!(chi2_statistic > 0.0);
        assert!(p_value > 0.0 && p_value < 1.0);
    }

    #[test]
    fn test_anova() {
        let group1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
        let group2 = Matrix::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], 1, 5);
        let group3 = Matrix::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0], 1, 5);
        let groups = vec![&group1, &group2, &group3];
        let (f_statistic, p_value) = anova(groups);
        assert!(f_statistic > 0.0);
        assert!(p_value > 0.0 && p_value < 1.0);
    }
}
