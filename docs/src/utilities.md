# Utilities

Utilities provide handy helpers around the core library. Existing tools
include:

- Date utilities for generating calendar sequences and business‑day sets
- Random number generators for simulations and testing

## Date Helpers

```rust
# extern crate rustframe;
use rustframe::utils::dateutils::{BDatesList, BDateFreq, DatesList, DateFreq};

// Calendar sequence
let list = DatesList::new("2024-01-01".into(), "2024-01-03".into(), DateFreq::Daily);
assert_eq!(list.count().unwrap(), 3);

// Business days starting from 2024‑01‑02
let bdates = BDatesList::from_n_periods("2024-01-02".into(), BDateFreq::Daily, 3).unwrap();
assert_eq!(bdates.list().unwrap().len(), 3);
```

## Random Numbers

The `random` module offers deterministic and cryptographically secure RNGs.

```rust
# extern crate rustframe;
use rustframe::random::{Prng, Rng};

let mut rng = Prng::new(42);
let v1 = rng.next_u64();
let v2 = rng.next_u64();
assert_ne!(v1, v2);
```

Upcoming utilities will cover:

- Data import/export helpers
- Visualization adapters
- Streaming data interfaces

Contributions to these sections are welcome!
