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
