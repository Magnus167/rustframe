use rustframe::compute::models::k_means::KMeans;
use rustframe::matrix::Matrix;

/// Two quick K-Means clustering demos.
///
/// Example 1 groups store locations on a city map.
/// Example 2 segments customers by annual spending habits.
fn main() {
    city_store_example();
    println!("\n-----\n");
    customer_spend_example();
}

fn city_store_example() {
    println!("Example 1: store locations");

    // (x, y) coordinates of stores around a city
    let raw = vec![
        1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0, 8.0, 2.0, 10.0, 2.0, 9.0, 3.0,
    ];
    let x = Matrix::from_rows_vec(raw, 9, 2);

    // Group stores into two areas
    let (model, labels) = KMeans::fit(&x, 2, 100, 1e-4);

    println!("Centres: {:?}", model.centroids.data());
    println!("Labels: {:?}", labels);

    let new_points = Matrix::from_rows_vec(vec![0.0, 0.0, 8.0, 3.0], 2, 2);
    let pred = model.predict(&new_points);
    println!("New store assignments: {:?}", pred);
}

fn customer_spend_example() {
    println!("Example 2: customer spending");

    // (grocery spend, electronics spend) in dollars
    let raw = vec![
        200.0, 150.0, 220.0, 170.0, 250.0, 160.0, 800.0, 750.0, 820.0, 760.0, 790.0, 770.0,
    ];
    let x = Matrix::from_rows_vec(raw, 6, 2);

    let (model, labels) = KMeans::fit(&x, 2, 100, 1e-4);

    println!("Centres: {:?}", model.centroids.data());
    println!("Labels: {:?}", labels);

    let new_customers = Matrix::from_rows_vec(vec![230.0, 155.0, 810.0, 760.0], 2, 2);
    let pred = model.predict(&new_customers);
    println!("Cluster of new customers: {:?}", pred);
}
