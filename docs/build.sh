#!/usr/bin/env sh
# Build and test the Rustframe user guide using mdBook.
set -e

cd docs
bash gen.sh "$@"
cd ..