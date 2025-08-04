/// Iterator producing successive approximations of π using the Nilakantha series.
pub struct PiSpigot {
    k: u64,
    current: f64,
}

impl Iterator for PiSpigot {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.k == 0 {
            self.k = 1;
            self.current = 3.0;
            return Some(self.current);
        }
        let k = self.k as f64;
        let term = 4.0 / ((2.0 * k) * (2.0 * k + 1.0) * (2.0 * k + 2.0));
        if self.k % 2 == 1 {
            self.current += term;
        } else {
            self.current -= term;
        }
        self.k += 1;
        Some(self.current)
    }
}

/// Generator yielding approximations of π indefinitely.
pub fn pi_spigot() -> PiSpigot {
    PiSpigot { k: 0, current: 0.0 }
}

/// Return the first `n` approximations of π as a vector.
pub fn pi_values(n: usize) -> Vec<f64> {
    pi_spigot().take(n).collect()
}

/// Generator yielding approximations of τ = 2π indefinitely.
pub fn tau_spigot() -> impl Iterator<Item = f64> {
    pi_spigot().map(|v| v * 2.0)
}

/// Return the first `n` approximations of τ as a vector.
pub fn tau_values(n: usize) -> Vec<f64> {
    tau_spigot().take(n).collect()
}

/// Iterator producing successive approximations of the Euler-Mascheroni constant γ.
pub struct GammaSpigot {
    n: u64,
    harmonic: f64,
}

impl Iterator for GammaSpigot {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.n += 1;
        self.harmonic += 1.0 / self.n as f64;
        let value = self.harmonic - (self.n as f64).ln();
        Some(value)
    }
}

/// Generator yielding approximations of γ indefinitely.
pub fn gamma_spigot() -> GammaSpigot {
    GammaSpigot {
        n: 0,
        harmonic: 0.0,
    }
}

/// Return the first `n` approximations of γ as a vector.
pub fn gamma_values(n: usize) -> Vec<f64> {
    gamma_spigot().take(n).collect()
}

/// Iterator producing successive approximations of e using the series Σ 1/n!.
pub struct ESpigot {
    n: u64,
    sum: f64,
    factorial: f64,
}

impl Iterator for ESpigot {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            self.n = 1;
            self.sum = 1.0;
            self.factorial = 1.0;
            return Some(self.sum);
        }
        self.factorial *= self.n as f64;
        self.sum += 1.0 / self.factorial;
        self.n += 1;
        Some(self.sum)
    }
}

/// Generator yielding approximations of e indefinitely.
pub fn e_spigot() -> ESpigot {
    ESpigot {
        n: 0,
        sum: 0.0,
        factorial: 1.0,
    }
}

/// Return the first `n` approximations of e as a vector.
pub fn e_values(n: usize) -> Vec<f64> {
    e_spigot().take(n).collect()
}

/// Iterator producing successive approximations of √2 using Newton's method.
pub struct Sqrt2Spigot {
    x: f64,
    first: bool,
}

impl Iterator for Sqrt2Spigot {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            Some(self.x)
        } else {
            self.x = 0.5 * (self.x + 2.0 / self.x);
            Some(self.x)
        }
    }
}

/// Generator yielding approximations of √2 indefinitely.
pub fn sqrt2_spigot() -> Sqrt2Spigot {
    Sqrt2Spigot {
        x: 1.0,
        first: true,
    }
}

/// Return the first `n` approximations of √2 as a vector.
pub fn sqrt2_values(n: usize) -> Vec<f64> {
    sqrt2_spigot().take(n).collect()
}

fn look_and_say(s: &str) -> String {
    let mut chars = s.chars().peekable();
    let mut result = String::new();
    while let Some(c) = chars.next() {
        let mut count = 1;
        while let Some(&next) = chars.peek() {
            if next == c {
                chars.next();
                count += 1;
            } else {
                break;
            }
        }
        result.push_str(&format!("{}{}", count, c));
    }
    result
}

/// Iterator producing successive ratios of lengths of the look-and-say sequence.
pub struct ConwaySpigot {
    current: String,
}

impl Iterator for ConwaySpigot {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let next = look_and_say(&self.current);
        let ratio = next.len() as f64 / self.current.len() as f64;
        self.current = next;
        Some(ratio)
    }
}

/// Generator yielding approximations of Conway's constant λ indefinitely.
pub fn conway_spigot() -> ConwaySpigot {
    ConwaySpigot {
        current: "1".to_string(),
    }
}

/// Return the first `n` approximations of Conway's constant as a vector.
pub fn conway_values(n: usize) -> Vec<f64> {
    conway_spigot().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI, TAU};

    #[test]
    fn test_pi_spigot() {
        let vals = pi_values(1000);
        let approx = vals.last().cloned().unwrap();
        assert!((approx - PI).abs() < 1e-8);
    }

    #[test]
    fn test_tau_spigot() {
        let vals = tau_values(1000);
        let approx = vals.last().cloned().unwrap();
        assert!((approx - TAU).abs() < 1e-8);
    }

    #[test]
    fn test_gamma_spigot() {
        let vals = gamma_values(100000);
        let approx = vals.last().cloned().unwrap();
        let gamma_true = 0.5772156649015329_f64;
        assert!((approx - gamma_true).abs() < 1e-5);
    }

    #[test]
    fn test_e_spigot() {
        let vals = e_values(10);
        let approx = vals.last().cloned().unwrap();
        assert!((approx - E).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt2_spigot() {
        let vals = sqrt2_values(6);
        let approx = vals.last().cloned().unwrap();
        assert!((approx - 2_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_conway_spigot() {
        let vals = conway_values(25);
        let approx = vals.last().cloned().unwrap();
        let conway = 1.3035772690342964_f64;
        assert!((approx - conway).abs() < 1e-2);
    }
}
