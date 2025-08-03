#!/usr/bin/env sh
# Build and test the RustFrame user guide using mdBook.
set -e
# Ensure the library is compiled so examples can link against it.

cargo clean

cargo build --manifest-path ../Cargo.toml
# Run embedded code examples as tests.
mdbook test -L ../target/debug/deps "$@"
# Finally, render the book.
mdbook build "$@"

cargo build
cargo build --release
