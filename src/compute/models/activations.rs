use crate::matrix::{Matrix, SeriesOps};

pub fn sigmoid(x: &Matrix<f64>) -> Matrix<f64> {
    x.map(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn dsigmoid(y: &Matrix<f64>) -> Matrix<f64> {
    // derivative w.r.t. pre-activation; takes y = sigmoid(x)
    y.map(|v| v * (1.0 - v))
}

pub fn relu(x: &Matrix<f64>) -> Matrix<f64> {
    x.map(|v| if v > 0.0 { v } else { 0.0 })
}

pub fn drelu(x: &Matrix<f64>) -> Matrix<f64> {
    x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

pub fn leaky_relu(x: &Matrix<f64>) -> Matrix<f64> {
    x.map(|v| if v > 0.0 { v } else { 0.01 * v })
}

pub fn dleaky_relu(x: &Matrix<f64>) -> Matrix<f64> {
    x.map(|v| if v > 0.0 { 1.0 } else { 0.01 })
}

mod tests {
    use super::*;

    // Helper function to round all elements in a matrix to n decimal places
    fn _round_matrix(mat: &Matrix<f64>, decimals: u32) -> Matrix<f64> {
        let factor = 10f64.powi(decimals as i32);
        let rounded: Vec<f64> = mat
            .to_vec()
            .iter()
            .map(|v| (v * factor).round() / factor)
            .collect();
        Matrix::from_vec(rounded, mat.rows(), mat.cols())
    }

    #[test]
    fn test_sigmoid() {
        let x = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.26894142, 0.5, 0.73105858], 3, 1);
        let result = sigmoid(&x);
        assert_eq!(_round_matrix(&result, 6), _round_matrix(&expected, 6));
    }

    #[test]
    fn test_sigmoid_edge_case() {
        let x = Matrix::from_vec(vec![-1000.0, 0.0, 1000.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.0, 0.5, 1.0], 3, 1);
        let result = sigmoid(&x);

        for (r, e) in result.data().iter().zip(expected.data().iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_relu() {
        let x = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.0, 0.0, 1.0], 3, 1);
        assert_eq!(relu(&x), expected);
    }

    #[test]
    fn test_relu_edge_case() {
        let x = Matrix::from_vec(vec![-1e-10, 0.0, 1e10], 3, 1);
        let expected = Matrix::from_vec(vec![0.0, 0.0, 1e10], 3, 1);
        assert_eq!(relu(&x), expected);
    }

    #[test]
    fn test_dsigmoid() {
        let y = Matrix::from_vec(vec![0.26894142, 0.5, 0.73105858], 3, 1);
        let expected = Matrix::from_vec(vec![0.19661193, 0.25, 0.19661193], 3, 1);
        let result = dsigmoid(&y);
        assert_eq!(_round_matrix(&result, 6), _round_matrix(&expected, 6));
    }

    #[test]
    fn test_dsigmoid_edge_case() {
        let y = Matrix::from_vec(vec![0.0, 0.5, 1.0], 3, 1); // Assume these are outputs from sigmoid(x)
        let expected = Matrix::from_vec(vec![0.0, 0.25, 0.0], 3, 1);
        let result = dsigmoid(&y);

        for (r, e) in result.data().iter().zip(expected.data().iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_drelu() {
        let x = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.0, 0.0, 1.0], 3, 1);
        assert_eq!(drelu(&x), expected);
    }

    #[test]
    fn test_drelu_edge_case() {
        let x = Matrix::from_vec(vec![-1e-10, 0.0, 1e10], 3, 1);
        let expected = Matrix::from_vec(vec![0.0, 0.0, 1.0], 3, 1);
        assert_eq!(drelu(&x), expected);
    }

    #[test]
    fn test_leaky_relu() {
        let x = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![-0.01, 0.0, 1.0], 3, 1);
        assert_eq!(leaky_relu(&x), expected);
    }

    #[test]
    fn test_leaky_relu_edge_case() {
        let x = Matrix::from_vec(vec![-1e-10, 0.0, 1e10], 3, 1);
        let expected = Matrix::from_vec(vec![-1e-12, 0.0, 1e10], 3, 1);
        assert_eq!(leaky_relu(&x), expected);
    }

    #[test]
    fn test_dleaky_relu() {
        let x = Matrix::from_vec(vec![-1.0, 0.0, 1.0], 3, 1);
        let expected = Matrix::from_vec(vec![0.01, 0.01, 1.0], 3, 1);
        assert_eq!(dleaky_relu(&x), expected);
    }

    #[test]
    fn test_dleaky_relu_edge_case() {
        let x = Matrix::from_vec(vec![-1e-10, 0.0, 1e10], 3, 1);
        let expected = Matrix::from_vec(vec![0.01, 0.01, 1.0], 3, 1);
        assert_eq!(dleaky_relu(&x), expectewd);
    }
}
