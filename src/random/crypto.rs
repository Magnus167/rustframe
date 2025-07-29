#[cfg(unix)]
use std::{fs::File, io::Read};

use crate::random::Rng;

#[cfg(unix)]
pub struct CryptoRng {
    file: File,
}

#[cfg(unix)]
impl CryptoRng {
    /// Open `/dev/urandom`.
    pub fn new() -> Self {
        let file = File::open("/dev/urandom").expect("failed to open /dev/urandom");
        Self { file }
    }
}

#[cfg(unix)]
impl Rng for CryptoRng {
    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.file
            .read_exact(&mut buf)
            .expect("failed reading from /dev/urandom");
        u64::from_ne_bytes(buf)
    }
}

#[cfg(windows)]
pub struct CryptoRng;

#[cfg(windows)]
impl CryptoRng {
    /// No handle is needed on Windows.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(windows)]
impl Rng for CryptoRng {
    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        win_fill(&mut buf).expect("BCryptGenRandom failed");
        u64::from_ne_bytes(buf)
    }
}

/// Fill `buf` with cryptographically secure random bytes using CNG.
///
/// * `BCryptGenRandom(NULL, buf, len, BCRYPT_USE_SYSTEM_PREFERRED_RNG)`
///   asks the OS for its system‑preferred DRBG (CTR_DRBG on modern
///   Windows).
#[cfg(windows)]
fn win_fill(buf: &mut [u8]) -> Result<(), ()> {
    use core::ffi::c_void;

    type BcryptAlgHandle = *mut c_void;
    type NTSTATUS = i32;

    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;

    #[link(name = "bcrypt")]
    extern "system" {
        fn BCryptGenRandom(
            hAlgorithm: BcryptAlgHandle,
            pbBuffer: *mut u8,
            cbBuffer: u32,
            dwFlags: u32,
        ) -> NTSTATUS;
    }

    // NT_SUCCESS(status) == status >= 0
    let status = unsafe {
        BCryptGenRandom(
            core::ptr::null_mut(),
            buf.as_mut_ptr(),
            buf.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };

    if status >= 0 {
        Ok(())
    } else {
        Err(())
    }
}

/// Convenience constructor for [`CryptoRng`].
pub fn crypto_rng() -> CryptoRng {
    CryptoRng::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;
    use std::collections::HashSet;

    #[test]
    fn test_crypto_rng_nonzero() {
        let mut rng = CryptoRng::new();
        let mut all_same = true;
        let mut prev = rng.next_u64();
        for _ in 0..5 {
            let val = rng.next_u64();
            if val != prev {
                all_same = false;
            }
            prev = val;
        }
        assert!(!all_same, "CryptoRng produced identical values");
    }

    #[test]
    fn test_crypto_rng_variation_large() {
        let mut rng = CryptoRng::new();
        let mut values = HashSet::new();
        for _ in 0..100 {
            values.insert(rng.next_u64());
        }
        assert!(values.len() > 90, "CryptoRng output not varied enough");
    }

    #[test]
    fn test_crypto_rng_random_range_uniform() {
        let mut rng = CryptoRng::new();
        let mut counts = [0usize; 10];
        for _ in 0..1000 {
            let v = rng.random_range(0..10usize);
            counts[v] += 1;
        }
        for &c in &counts {
            // "Crypto RNG counts far from uniform: {c}"
            assert!((c as isize - 100).abs() < 50);
        }
    }

    #[test]
    fn test_crypto_normal_distribution() {
        let mut rng = CryptoRng::new();
        let mean = 0.0;
        let sd = 1.0;
        let n = 2000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let val = rng.normal(mean, sd);
            sum += val;
            sum_sq += val * val;
        }
        let sample_mean = sum / n as f64;
        let sample_var = sum_sq / n as f64 - sample_mean * sample_mean;
        assert!(sample_mean.abs() < 0.1);
        assert!((sample_var - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_two_instances_different_values() {
        let mut a = CryptoRng::new();
        let mut b = CryptoRng::new();
        let va = a.next_u64();
        let vb = b.next_u64();
        assert_ne!(va, vb);
    }

    #[test]
    fn test_crypto_rng_helper_function() {
        let mut rng = crypto_rng();
        let _ = rng.next_u64();
    }

    #[test]
    fn test_crypto_normal_zero_sd() {
        let mut rng = CryptoRng::new();
        for _ in 0..5 {
            let v = rng.normal(10.0, 0.0);
            assert_eq!(v, 10.0);
        }
    }

    #[test]
    fn test_crypto_shuffle_empty_slice() {
        use crate::random::SliceRandom;
        let mut rng = CryptoRng::new();
        let mut arr: [u8; 0] = [];
        arr.shuffle(&mut rng);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_crypto_chi_square_uniform() {
        let mut rng = CryptoRng::new();
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
        assert!(chi2 < 40.0, "chi-square statistic too high: {chi2}");
    }

    #[test]
    fn test_crypto_monobit() {
        let mut rng = CryptoRng::new();
        let mut ones = 0usize;
        let samples = 1000;
        for _ in 0..samples {
            ones += rng.next_u64().count_ones() as usize;
        }
        let total_bits = samples * 64;
        let ratio = ones as f64 / total_bits as f64;
        // "bit ratio far from 0.5: {ratio}"
        assert!((ratio - 0.5).abs() < 0.02);
    }
}
