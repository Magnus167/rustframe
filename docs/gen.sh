#!/usr/bin/env sh

set -e

cargo clean

cargo build --manifest-path ../Cargo.toml

mdbook test -L ../target/debug/deps "$@"

mdbook build "$@"

cargo build
# cargo build --release
