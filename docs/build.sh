#!/usr/bin/env sh
# Build and test the RustFrame user guide using mdBook.
set -e

cd docs
bash gen.sh "$@"
cd ..