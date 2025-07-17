//! randomx.rs
//!
//! Shared random API + fast pseudo-random engine (xorshift*).
//! Sister secure engine lives in `randomx_secure.rs`.
//!
//! Not crypto-secure (unless you use the SecureRandomX type from the other file).

#![allow(dead_code)]

use core::f64::consts::PI;
use core::ops::Range;
use std::sync::{LazyLock, OnceLock};

// Engine abstraction

/// Minimal trait every random *engine* must satisfy: produce the next u64 of
/// raw randomness. Higher-level sampling is built generically on top of this.
pub trait Engine {
    /// Produce fresh 64 random bits. Must be well mixed; may block (OS engines).
    fn next_u64(&mut self) -> u64;
}

/// A full-featured RNG façade built over any `Engine`.
///
/// All user-facing methods (uniforms, distributions, shuffles…) live here,
/// so they are **shared** by pseudo and secure RNG types.
#[derive(Debug, Clone)]
pub struct RandomApi<E: Engine> {
    engine: E,
}

impl<E: Engine> RandomApi<E> {
    /* ----- ctor ----- */

    pub fn from_engine(engine: E) -> Self {
        Self { engine }
    }

    /* ----- core draws ----- */

    #[inline]
    pub fn u64(&mut self) -> u64 {
        self.engine.next_u64()
    }

    #[inline]
    pub fn u32(&mut self) -> u32 {
        self.u64() as u32
    }

    /// Uniform `[0,1)` double with 53 random mantissa bits.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        const DEN: f64 = (1u64 << 53) as f64;
        ((self.u64() >> 11) as f64) / DEN
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        debug_assert!((0.0..=1.0).contains(&p));
        self.f64() < p
    }

    /* ----- uniform ranges ----- */

    #[inline]
    pub fn range_u32(&mut self, low: u32, high: u32) -> u32 {
        assert!(low < high);
        let span = (high - low) as u64;
        (low as u64 + self.u64() % span) as u32
    }

    #[inline]
    pub fn range_u64(&mut self, low: u64, high: u64) -> u64 {
        assert!(low < high);
        let span = high - low;
        low + (self.u64() % span)
    }

    #[inline]
    pub fn range_u32_r(&mut self, r: Range<u32>) -> u32 {
        self.range_u32(r.start, r.end)
    }

    #[inline]
    pub fn range_u64_r(&mut self, r: Range<u64>) -> u64 {
        self.range_u64(r.start, r.end)
    }

    #[inline]
    pub fn range_f64(&mut self, low: f64, high: f64) -> f64 {
        assert!(low < high);
        low + (high - low) * self.f64()
    }

    /* ----- basic distributions ----- */

    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        debug_assert!(std_dev >= 0.0);
        let u1 = self.f64();
        let u2 = self.f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z0
    }

    pub fn exponential(&mut self, lambda: f64) -> f64 {
        debug_assert!(lambda > 0.0);
        -self.f64().ln() / lambda
    }

    pub fn gamma(&mut self, k: f64, theta: f64) -> f64 {
        debug_assert!(k > 0.0 && theta > 0.0);

        if k < 1.0 {
            let u = self.f64();
            return self.gamma(k + 1.0, theta) * u.powf(1.0 / k);
        }

        let d = k - 1.0 / 3.0;
        let c = 1.0 / (3.0 * d).sqrt();
        loop {
            let x = self.normal(0.0, 1.0);
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.f64();
            if u < 1.0 - 0.0331 * x * x * x * x {
                return theta * d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return theta * d * v;
            }
        }
    }

    pub fn poisson(&mut self, lambda: f64) -> u64 {
        debug_assert!(lambda >= 0.0);
        if lambda == 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            // Knuth
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.f64();
                if p <= l {
                    return k - 1;
                }
            }
        }
        // Rejection-ish fallback
        let sq = lambda.sqrt();
        loop {
            let y = (PI * self.f64()).tan();
            let x = sq * y + lambda;
            if x < 0.0 {
                continue;
            }
            let k = x.floor() as u64;
            let log_p_k = (k as f64) * lambda.ln() - lambda - ln_factorial(k);
            let u = self.f64();
            if u.ln() <= log_p_k - 0.5 * y * y {
                return k;
            }
        }
    }

    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        debug_assert!((0.0..=1.0).contains(&p));
        if n == 0 || p == 0.0 {
            return 0;
        }
        if p == 1.0 {
            return n;
        }
        if n < 25 {
            let mut c = 0;
            for _ in 0..n {
                if self.bernoulli(p) {
                    c += 1;
                }
            }
            return c;
        }
        let np = n as f64 * p;
        if np < 1.0 {
            return self.poisson(np).min(n);
        }
        let mean = np;
        let std = (n as f64 * p * (1.0 - p)).sqrt();
        loop {
            let s = self.normal(mean, std).round();
            if (0.0..=(n as f64)).contains(&s) {
                return s as u64;
            }
        }
    }

    /* ----- slice helpers ----- */

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range_u64(0, (i + 1) as u64) as usize;
            slice.swap(i, j);
        }
    }

    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        if slice.is_empty() {
            None
        } else {
            Some(&slice[self.range_u64(0, slice.len() as u64) as usize])
        }
    }

    pub fn choose_mut<'a, T>(&mut self, slice: &'a mut [T]) -> Option<&'a mut T> {
        if slice.is_empty() {
            None
        } else {
            let idx = self.range_u64(0, slice.len() as u64) as usize;
            Some(&mut slice[idx])
        }
    }

    pub fn sample<'a, T>(&mut self, slice: &'a [T], k: usize) -> Vec<&'a T> {
        if k == 0 || slice.is_empty() {
            return Vec::new();
        }
        if k >= slice.len() {
            return slice.iter().collect();
        }
        // Reservoir
        let mut out: Vec<&T> = slice.iter().take(k).collect();
        for (i, item) in slice.iter().enumerate().skip(k) {
            let j = self.range_u64(0, (i + 1) as u64) as usize;
            if j < k {
                out[j] = item;
            }
        }
        out
    }

    pub fn choose_weighted<'a, T>(&mut self, items: &'a [T], weights: &[f64]) -> Option<&'a T> {
        assert_eq!(items.len(), weights.len());
        if items.is_empty() {
            return None;
        }
        let mut total = 0.0;
        for &w in weights {
            if w > 0.0 {
                total += w;
            }
        }
        if total == 0.0 {
            return None;
        }
        let mut r = self.range_f64(0.0, total);
        for (item, &w) in items.iter().zip(weights.iter()) {
            if w > 0.0 {
                if r < w {
                    return Some(item);
                }
                r -= w;
            }
        }
        Some(&items[items.len() - 1])
    }

    /* ----- bulk convenience ----- */

    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut i = 0;
        while i + 8 <= buf.len() {
            buf[i..i + 8].copy_from_slice(&self.u64().to_le_bytes());
            i += 8;
        }
        if i < buf.len() {
            let rem = buf.len() - i;
            buf[i..].copy_from_slice(&self.u64().to_le_bytes()[..rem]);
        }
    }

    pub fn vec_u32(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.u32()).collect()
    }

    pub fn vec_f64(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.f64()).collect()
    }

    pub fn vec_normal(&mut self, n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
        (0..n).map(|_| self.normal(mean, std_dev)).collect()
    }
}

/* =============================================================================
 * Pseudo engine (xorshift*) + convenience constructors
 * ========================================================================== */

#[derive(Clone, Copy, Debug)]
pub struct PseudoEngine {
    state: u64,
}

impl PseudoEngine {
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "seed must be non-zero");
        Self { state: seed }
    }

    /// Cheap non-crypto entropy seed.
    pub fn from_entropy() -> Self {
        use core::mem;
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let addr = &mem::size_of::<usize>() as *const usize as usize as u64;
        let mut seed = t ^ addr.rotate_left(17);
        if seed == 0 {
            seed = 0xD_E_A_D_B_E_E_F;
        }
        Self::new(seed)
    }
}

impl Engine for PseudoEngine {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/* ----- Type alias preserving your old name ----- */

pub type RandomX = RandomApi<PseudoEngine>;

impl RandomX {
    /// Create deterministic pseudo RNG from seed.
    pub fn from_seed(seed: u64) -> Self {
        Self::from_engine(PseudoEngine::new(seed))
    }

    /// Create pseudo RNG from non-crypto entropy.
    pub fn from_entropy() -> Self {
        Self::from_engine(PseudoEngine::from_entropy())
    }
}

/* =============================================================================
 * ln_factorial helper + backend integration hook
 * ========================================================================== */

fn ln_factorial(n: u64) -> f64 {
    // Small-table + Stirling; replace with your backend factorial() if desired.
    const N: usize = 32;
    static TABLE: OnceLock<[f64; N]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0.0f64; N];
        let mut acc: u128 = 1;
        let mut i = 0;
        while i < N {
            if i > 0 {
                acc *= i as u128;
            }
            t[i] = (acc as f64).ln();
            i += 1;
        }
        t
    });
    if (n as usize) < N {
        return table[n as usize];
    }
    static LN_FACTORIAL_LAZY: LazyLock<[f64; N]> = LazyLock::new(|| {
        let mut t = [0.0f64; N];
        let mut acc: u128 = 1;
        let mut i = 0;
        while i < N {
            if i > 0 {
                acc *= i as u128;
            }
            t[i] = (acc as f64).ln();
            i += 1;
        }
        t
    });
    let nf = n as f64;
    nf * nf.ln() - nf + 0.5 * (2.0 * PI * nf).ln() + 1.0 / (12.0 * nf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_repeatable() {
        let mut a = RandomX::from_seed(123);
        let mut b = RandomX::from_seed(123);
        assert_eq!(a.u64(), b.u64());
        assert_eq!(a.normal(0.0, 1.0), b.normal(0.0, 1.0));
    }

    #[test]
    fn shuffle_works() {
        let mut rng = RandomX::from_seed(1);
        let mut xs = [1, 2, 3, 4, 5];
        rng.shuffle(&mut xs);
        assert_eq!(xs.len(), 5);
    }
}
