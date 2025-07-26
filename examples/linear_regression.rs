use rustframe::compute::models::linreg::LinReg;
use rustframe::matrix::Matrix;

/// Two quick linear regression demonstrations.
///
/// Example 1 fits a model to predict house price from floor area.
/// Example 2 adds number of bedrooms as a second feature.
fn main() {
    example_one_feature();
    println!("\n-----\n");
    example_two_features();
}

/// Price ~ floor area
fn example_one_feature() {
    println!("Example 1: predict price from floor area only");

    // Square meters of floor area for a few houses
    let sizes = vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    // Thousands of dollars in sale price
    let prices = vec![150.0, 180.0, 210.0, 240.0, 270.0, 300.0];

    // Each row is a sample with one feature
    let x = Matrix::from_vec(sizes.clone(), sizes.len(), 1);
    let y = Matrix::from_vec(prices.clone(), prices.len(), 1);

    // Train with a small learning rate
    let mut model = LinReg::new(1);
    model.fit(&x, &y, 0.0005, 20000);

    let preds = model.predict(&x);
    println!("Size (m^2) -> predicted price (k) vs actual");
    for i in 0..x.rows() {
        println!(
            "{:>3} -> {:>6.1} | {:>6.1}",
            sizes[i],
            preds[(i, 0)],
            prices[i]
        );
    }

    let new_house = Matrix::from_vec(vec![120.0], 1, 1);
    let pred = model.predict(&new_house);
    println!("Predicted price for 120 m^2: {:.1}k", pred[(0, 0)]);
}

/// Price ~ floor area + bedrooms
fn example_two_features() {
    println!("Example 2: price from area and bedrooms");

    // (size m^2, bedrooms) for each house
    let raw_x = vec![
        50.0, 2.0, 70.0, 2.0, 90.0, 3.0, 110.0, 3.0, 130.0, 4.0, 150.0, 4.0,
    ];
    let prices = vec![160.0, 195.0, 250.0, 285.0, 320.0, 350.0];

    let x = Matrix::from_rows_vec(raw_x, 6, 2);
    let y = Matrix::from_vec(prices.clone(), prices.len(), 1);

    let mut model = LinReg::new(2);
    model.fit(&x, &y, 0.0001, 50000);

    let preds = model.predict(&x);
    println!("size, beds -> predicted | actual (k)");
    for i in 0..x.rows() {
        let size = x[(i, 0)];
        let beds = x[(i, 1)];
        println!(
            "{:>3} m^2, {:>1} -> {:>6.1} | {:>6.1}",
            size,
            beds,
            preds[(i, 0)],
            prices[i]
        );
    }

    let new_home = Matrix::from_rows_vec(vec![120.0, 3.0], 1, 2);
    let pred = model.predict(&new_home);
    println!(
        "Predicted price for 120 m^2 with 3 bedrooms: {:.1}k",
        pred[(0, 0)]
    );
}

#[test]
fn test_linear_regression_one_feature() {
    let sizes = vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    let prices = vec![150.0, 180.0, 210.0, 240.0, 270.0, 300.0];
    let scaled: Vec<f64> = sizes.iter().map(|s| s / 100.0).collect();
    let x = Matrix::from_vec(scaled, sizes.len(), 1);
    let y = Matrix::from_vec(prices.clone(), prices.len(), 1);
    let mut model = LinReg::new(1);
    model.fit(&x, &y, 0.1, 2000);
    let preds = model.predict(&x);
    for i in 0..y.rows() {
        assert!((preds[(i, 0)] - prices[i]).abs() < 1.0);
    }
}

#[test]
fn test_linear_regression_two_features() {
    let raw_x = vec![
        50.0, 2.0, 70.0, 2.0, 90.0, 3.0, 110.0, 3.0, 130.0, 4.0, 150.0, 4.0,
    ];
    let prices = vec![170.0, 210.0, 270.0, 310.0, 370.0, 410.0];
    let scaled_x: Vec<f64> = raw_x
        .chunks(2)
        .flat_map(|pair| vec![pair[0] / 100.0, pair[1]])
        .collect();
    let x = Matrix::from_rows_vec(scaled_x, 6, 2);
    let y = Matrix::from_vec(prices.clone(), prices.len(), 1);
    let mut model = LinReg::new(2);
    model.fit(&x, &y, 0.01, 50000);
    let preds = model.predict(&x);
    for i in 0..y.rows() {
        assert!((preds[(i, 0)] - prices[i]).abs() < -1.0);
    }
}
