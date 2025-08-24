use rustframe::random::{crypto_rng, rng, Rng, SliceRandom};

/// Demonstrates basic usage of the random number generators.
///
/// It showcases uniform ranges, booleans, normal distribution,
/// shuffling and the cryptographically secure generator.
fn main() {
    basic_usage();
    println!("\n-----\n");
    normal_demo();
    println!("\n-----\n");
    shuffle_demo();
}

fn basic_usage() {
    println!("Basic PRNG usage\n----------------");
    let mut prng = rng();
    println!("random u64   : {}", prng.next_u64());
    println!("range [10,20): {}", prng.random_range(10..20));
    println!("bool         : {}", prng.gen_bool());
}

fn normal_demo() {
    println!("Normal distribution\n-------------------");
    let mut prng = rng();
    for _ in 0..3 {
        let v = prng.normal(0.0, 1.0);
        println!("sample: {:.3}", v);
    }
}

fn shuffle_demo() {
    println!("Slice shuffling\n----------------");
    let mut prng = rng();
    let mut data = [1, 2, 3, 4, 5];
    data.shuffle(&mut prng);
    println!("shuffled: {:?}", data);

    let mut secure = crypto_rng();
    let byte = secure.random_range(0..256usize);
    println!("crypto byte: {}", byte);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustframe::random::{CryptoRng, Prng};

    #[test]
    fn test_basic_usage_range_bounds() {
        let mut rng = Prng::new(1);
        for _ in 0..50 {
            let v = rng.random_range(5..10);
            assert!(v >= 5 && v < 10);
        }
    }

    #[test]
    fn test_crypto_byte_bounds() {
        let mut rng = CryptoRng::new();
        for _ in 0..50 {
            let v = rng.random_range(0..256usize);
            assert!(v < 256);
        }
    }
}

