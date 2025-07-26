use rustframe::compute::models::logreg::LogReg;
use rustframe::matrix::Matrix;

/// Two binary classification demos using logistic regression.
///
/// Example 1 predicts exam success from hours studied.
/// Example 2 predicts whether an online shopper will make a purchase.
fn main() {
    student_passing_example();
    println!("\n-----\n");
    purchase_prediction_example();
}

fn student_passing_example() {
    println!("Example 1: exam pass prediction");

    // Hours studied for each student
    let hours = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // 0 = fail, 1 = pass
    let passed = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let x = Matrix::from_vec(hours.clone(), hours.len(), 1);
    let y = Matrix::from_vec(passed.clone(), passed.len(), 1);

    let mut model = LogReg::new(1);
    model.fit(&x, &y, 0.1, 10000);

    let preds = model.predict(&x);
    println!("Hours -> pred | actual");
    for i in 0..x.rows() {
        println!(
            "{:>2} -> {} | {}",
            hours[i] as i32,
            preds[(i, 0)] as i32,
            passed[i] as i32
        );
    }

    // Probability estimate for a new student
    let new_student = Matrix::from_vec(vec![5.5], 1, 1);
    let p = model.predict_proba(&new_student);
    println!("Probability of passing with 5.5h study: {:.2}", p[(0, 0)]);
}

fn purchase_prediction_example() {
    println!("Example 2: purchase likelihood");

    // minutes on site, pages viewed -> made a purchase?
    let raw_x = vec![1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 5.0, 5.0, 3.5, 2.0, 6.0, 6.0];
    let bought = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0];

    let x = Matrix::from_rows_vec(raw_x, 6, 2);
    let y = Matrix::from_vec(bought.clone(), bought.len(), 1);

    let mut model = LogReg::new(2);
    model.fit(&x, &y, 0.05, 20000);

    let preds = model.predict(&x);
    println!("time, pages -> pred | actual");
    for i in 0..x.rows() {
        println!(
            "{:>4}m, {:>2} -> {} | {}",
            x[(i, 0)],
            x[(i, 1)] as i32,
            preds[(i, 0)] as i32,
            bought[i] as i32
        );
    }

    let new_visit = Matrix::from_rows_vec(vec![4.0, 4.0], 1, 2);
    let p = model.predict_proba(&new_visit);
    println!("Prob of purchase for 4min/4pages: {:.2}", p[(0, 0)]);
}

#[test]
fn test_student_passing_example() {
    let hours = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let passed = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let x = Matrix::from_vec(hours.clone(), hours.len(), 1);
    let y = Matrix::from_vec(passed.clone(), passed.len(), 1);
    let mut model = LogReg::new(1);
    model.fit(&x, &y, 0.1, 10000);
    let preds = model.predict(&x);
    for i in 0..y.rows() {
        assert_eq!(preds[(i, 0)], passed[i]);
    }
}

#[test]
fn test_purchase_prediction_example() {
    let raw_x = vec![1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 5.0, 5.0, 3.5, 2.0, 6.0, 6.0];
    let bought = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let x = Matrix::from_rows_vec(raw_x, 6, 2);
    let y = Matrix::from_vec(bought.clone(), bought.len(), 1);
    let mut model = LogReg::new(2);
    model.fit(&x, &y, 0.05, 20000);
    let preds = model.predict(&x);
    for i in 0..y.rows() {
        assert_eq!(preds[(i, 0)], bought[i]);
    }
}