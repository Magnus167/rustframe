use rustframe::random::{crypto_rng, rng, Rng};

/// Demonstrates simple statistical checks on random number generators.
fn main() {
    chi_square_demo();
    println!("\n-----\n");
    monobit_demo();
}

fn chi_square_demo() {
    println!("Chi-square test on PRNG");
    let mut rng = rng();
    let mut counts = [0usize; 10];
    let samples = 10000;
    for _ in 0..samples {
        let v = rng.random_range(0..10usize);
        counts[v] += 1;
    }
    let expected = samples as f64 / 10.0;
    let chi2: f64 = counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected;
            diff * diff / expected
        })
        .sum();
    println!("counts: {:?}", counts);
    println!("chi-square: {:.3}", chi2);
}

fn monobit_demo() {
    println!("Monobit test on crypto RNG");
    let mut rng = crypto_rng();
    let mut ones = 0usize;
    let samples = 1000;
    for _ in 0..samples {
        ones += rng.next_u64().count_ones() as usize;
    }
    let ratio = ones as f64 / (samples as f64 * 64.0);
    println!("ones ratio: {:.4}", ratio);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_square_demo_runs() {
        chi_square_demo();
    }

    #[test]
    fn test_monobit_demo_runs() {
        monobit_demo();
    }
}

