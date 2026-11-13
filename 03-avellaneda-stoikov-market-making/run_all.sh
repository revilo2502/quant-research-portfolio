#!/usr/bin/env bash
# Runs everything: closed-form formula checks, Python Monte Carlo comparison,
# plots, cross-validation tests, and the C++ engine (requires cmake + g++).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Python deps =="
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== 2. Closed-form reservation price + optimal spread sanity checks =="
python3 python/avellaneda_stoikov.py

echo
echo "== 3. Monte Carlo comparison: Avellaneda-Stoikov vs naive symmetric spread =="
python3 python/simulate.py

echo
echo "== 4. Plotting example path + P&L distribution =="
python3 python/plot_results.py

echo
echo "== 5. Cross-validation test suite =="
python3 -m pytest tests/ -v

echo
echo "== 6. C++ simulation engine (requires cmake + g++) =="
if command -v cmake >/dev/null 2>&1; then
  cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
  cmake --build cpp/build -j"$(nproc 2>/dev/null || echo 4)"
  echo
  echo "C++ engine (500 sims x 200 steps):"
  ./cpp/build/as_simulate 500 200 42
else
  echo "cmake not found -- skipping C++ build. Install cmake + a C++ compiler to enable it."
fi

echo
echo "All done. Key outputs:"
echo "  math/derivation.md          - full HJB derivation (read this first)"
echo "  data/example_path.png       - quotes + inventory over time, AS vs naive"
echo "  data/pnl_distribution.png   - terminal P&L distribution comparison"
echo "  data/summary.csv            - Sharpe/inventory summary stats"
echo "  cpp/build/as_simulate       - fast C++ Monte Carlo engine"
