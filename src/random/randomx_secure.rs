//! randomx_secure.rs
//!
//! Cryptographically secure RNG built on operating system entropy sources.
//! Reuses the generic API defined in `randomx.rs` via `RandomApi<OsEngine>`.
//!
//! Usage:
//! ```
//! use rustframe::random::randomx_secure::SecureRandomX;
//! let mut rng = SecureRandomX::new().expect("secure rng");
//! let x = rng.normal(0.0, 1.0);
//! ```

#![allow(dead_code)]

use super::randomx::{Engine, RandomApi}; // reuse everything

/* =============================================================================
 * Platform-specific secure entropy
 * ========================================================================== */

const BUF_LEN: usize = 4096;

pub struct OsEngine {
    buf: [u8; BUF_LEN],
    idx: usize,
}

impl core::fmt::Debug for OsEngine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OsEngine")
            .field("remaining", &((BUF_LEN - self.idx) as u64))
            .finish()
    }
}

impl OsEngine {
    /// Create a new engine, prefilled with OS randomness.
    pub fn new() -> std::io::Result<Self> {
        let mut eng = Self {
            buf: [0u8; BUF_LEN],
            idx: BUF_LEN, // force immediate fill
        };
        eng.refill()?;
        Ok(eng)
    }

    #[inline]
    fn need_refill(&self) -> bool {
        self.idx >= BUF_LEN
    }

    fn refill(&mut self) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            use std::fs::File;
            use std::io::Read;
            let mut f = File::open("/dev/urandom")?;
            f.read_exact(&mut self.buf)?;
        }
        #[cfg(windows)]
        {
            // Call RtlGenRandom (SystemFunction036). Safe wrapper.
            unsafe {
                if !rtl_gen_random(self.buf.as_mut_ptr(), self.buf.len()) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "RtlGenRandom failed",
                    ));
                }
            }
        }
        self.idx = 0;
        Ok(())
    }
}

impl Engine for OsEngine {
    fn next_u64(&mut self) -> u64 {
        if self.need_refill() {
            // Best effort: panic if refill fails; alternatively propagate error by redesigning Engine.
            self.refill().expect("OS RNG refill failed");
        }
        // read 8 bytes little-endian
        let bytes = &self.buf[self.idx..self.idx + 8];
        self.idx += 8;
        u64::from_le_bytes(bytes.try_into().unwrap())
    }
}

/* ----- type alias & constructors for user convenience ----- */

pub type SecureRandomX = RandomApi<OsEngine>;

impl SecureRandomX {
    /// Get a crypto-secure RNG seeded from the OS.
    pub fn new() -> std::io::Result<Self> {
        OsEngine::new().map(RandomApi::from_engine)
    }
}

/* =============================================================================
 * Windows FFI (no external crate)
 * ========================================================================== */
#[cfg(windows)]
unsafe fn rtl_gen_random(buf: *mut u8, len: usize) -> bool {
    // SystemFunction036 in advapi32 (undocumented alias RtlGenRandom).
    // Signature: BOOLEAN SystemFunction036(PVOID RandomBuffer, ULONG RandomBufferLength);
    #[link(name = "advapi32")]
    extern "system" {
        fn SystemFunction036(RandomBuffer: *mut core::ffi::c_void, RandomBufferLength: u32)
            -> u8;
    }
    let ok = SystemFunction036(buf as *mut _, len as u32);
    ok != 0
}

/* =============================================================================
 * Tests (these will actually read OS randomness; mark ignore if needed)
 * ========================================================================== */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_draw() {
        let mut rng = SecureRandomX::new().unwrap();
        // Just make sure it runs and varies
        let a = rng.u64();
        let b = rng.u64();
        assert_ne!(a, b);
    }
}
