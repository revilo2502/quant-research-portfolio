#!/usr/bin/env bash
# Runs all three projects' pipelines in sequence. Each one is also fully
# self-contained if you'd rather run them individually (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

for proj in 01-hft-curve-predictor 02-option-pricing-stochastic-calculus 03-avellaneda-stoikov-market-making; do
  echo
  echo "################################################################"
  echo "# Running: $proj"
  echo "################################################################"
  ( cd "$proj" && chmod +x run_all.sh && ./run_all.sh )
done

echo
echo "All three projects complete."
