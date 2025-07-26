use rustframe::compute::models::pca::PCA;
use rustframe::matrix::Matrix;

/// Two dimensionality reduction examples using PCA.
///
/// Example 1 reduces 3D sensor readings to two components.
/// Example 2 compresses a small four-feature dataset.
fn main() {
    sensor_demo();
    println!("\n-----\n");
    finance_demo();
}

fn sensor_demo() {
    println!("Example 1: 3D sensor data");

    // Ten 3D observations from an accelerometer
    let raw = vec![
        2.5, 2.4, 0.5, 0.5, 0.7, 1.5, 2.2, 2.9, 0.7, 1.9, 2.2, 1.0, 3.1, 3.0, 0.6, 2.3, 2.7, 0.9,
        2.0, 1.6, 1.1, 1.0, 1.1, 1.9, 1.5, 1.6, 2.2, 1.1, 0.9, 2.1,
    ];
    let x = Matrix::from_rows_vec(raw, 10, 3);

    let pca = PCA::fit(&x, 2, 0);
    let reduced = pca.transform(&x);

    println!("Components: {:?}", pca.components.data());
    println!("First row -> {:.2?}", [reduced[(0, 0)], reduced[(0, 1)]]);
}

fn finance_demo() {
    println!("Example 2: 4D finance data");

    // Four daily percentage returns of different stocks
    let raw = vec![
        0.2, 0.1, -0.1, 0.0, 0.3, 0.2, -0.2, 0.1, 0.1, 0.0, -0.1, -0.1, 0.4, 0.3, -0.3, 0.2, 0.0,
        -0.1, 0.1, -0.1,
    ];
    let x = Matrix::from_rows_vec(raw, 5, 4);

    // Keep two principal components
    let pca = PCA::fit(&x, 2, 0);
    let reduced = pca.transform(&x);

    println!("Reduced shape: {:?}", reduced.shape());
    println!("First row -> {:.2?}", [reduced[(0, 0)], reduced[(0, 1)]]);
}