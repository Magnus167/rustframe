# Utilities

Utilities provide handy helpers around the core library. Existing tools
include:

- Date utilities for generating calendar sequences.

## Date Helpers

```rust
# extern crate rustframe;
use rustframe::utils::dateutils::{DatesList, DateFreq};

let list = DatesList::new("2024-01-01".into(), "2024-01-03".into(), DateFreq::Daily);
assert_eq!(list.count().unwrap(), 3);
```

Upcoming utilities will cover:

- Data import/export helpers
- Visualization adapters
- Streaming data interfaces

Contributions to these sections are welcome!
