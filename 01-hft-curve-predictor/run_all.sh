#!/usr/bin/env bash
# End-to-end pipeline: generate data -> features -> train -> backtest -> C++ build -> benchmark.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Python deps =="
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== 2. Generate synthetic LOB data (200k ticks) =="
python3 python/synth_data.py --n-steps 200000 --out data/synthetic_lob.csv

echo "== 3. Build features (multi-horizon targets) =="
python3 python/features.py --in data/synthetic_lob.csv --out data/features.csv

echo "== 4. Train multi-horizon models =="
python3 python/train.py --features data/features.csv --out-dir models

echo "== 5. Backtest h=1 direction model =="
python3 python/backtest.py --features data/features.csv --model-dir models --horizon 1

echo "== 6. Build C++ order book engine =="
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j"$(nproc 2>/dev/null || echo 4)"

echo "== 7. Latency benchmark: Python vs C++ =="
python3 benchmarks/latency_compare.py --csv data/synthetic_lob.csv --cpp-binary cpp/build/replay_engine --n-rows 50000

echo
echo "All done. Key outputs:"
echo "  data/synthetic_lob.csv       - raw synthetic order book"
echo "  data/features.csv        - engineered features + multi-horizon targets"
echo "  models/metrics.json          - per-horizon accuracy/RMSE vs naive baseline"
echo "  models/backtest_results.json - PnL/Sharpe/hit-rate from the backtest"
echo "  benchmarks/latency_results.json - Python vs C++ feature-pipeline latency"
echo "  cpp/build/replay_engine      - compiled C++ order book engine"
