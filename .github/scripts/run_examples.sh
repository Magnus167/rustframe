cargo build --release --examples 

for ex in examples/*.rs; do
  name=$(basename "$ex" .rs)
  echo
  echo "🟡 Running example: $name"

  if ! cargo run --release --example "$name" -- --debug; then
    echo
    echo "❌ Example '$name' failed. Aborting."
    exit 1
  fi
done

echo
echo "✅ All examples ran successfully."
