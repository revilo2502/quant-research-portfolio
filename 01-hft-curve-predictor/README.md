# HFT Curve Predictor

Limit order book (LOB) microstructure prediction: forecasts short-horizon
mid-price direction and return ("the curve") from order book state, with a
Python training/backtesting stack and a C++ order book reconstruction +
feature engine for the low-latency path.

> Honest scope note: this is a microstructure-prediction research project,
> not a production HFT system (no colo, no FPGA, no exchange connectivity).
> That's the right boundary for a portfolio project — it uses the real
> techniques (LOB imbalance, micro-price, multi-horizon return targets,
> C++/Python split for latency) without overclaiming production-grade infra.

## Quickstart

```bash
chmod +x run_all.sh
./run_all.sh
```

This generates synthetic data, builds features, trains multi-horizon models,
runs a backtest, builds the C++ order book engine with CMake, and runs the
Python-vs-C++ latency benchmark. Takes ~2-5 minutes depending on your machine.

## Step by step (what `run_all.sh` does)

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Synthetic order book data (swap for real data later -- see data/README.md)
python3 python/synth_data.py --n-steps 200000 --out data/synthetic_lob.csv

# 3. Feature engineering: OBI, micro-price, realized vol, multi-horizon targets
python3 python/features.py --in data/synthetic_lob.csv --out data/features.csv

# 4. Train LightGBM (or sklearn fallback) models for horizons [1,5,10,20,50] ticks
python3 python/train.py --features data/features.csv --out-dir models

# 5. Backtest the 1-tick-ahead direction model
python3 python/backtest.py --features data/features.csv --model-dir models --horizon 1

# 6. Build the C++ order book engine
cd cpp && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cd ..

# 7. Benchmark: Python/pandas feature pipeline vs C++ engine
python3 benchmarks/latency_compare.py --cpp-binary cpp/build/replay_engine
```

## Project structure

```
hft-curve-predictor/
  data/
    synth_data.py       # synthetic LOB generator (regime-switching + trade flow signal)
    README.md           # how to swap in real Binance/LOBSTER data
  python/
    features.py         # OBI, micro-price, realized vol, multi-horizon fwd-return targets
    train.py             # trains per-horizon classifier+regressor, optional ONNX export
    backtest.py          # event-driven backtest: PnL, Sharpe, hit rate vs baselines
  cpp/
    orderbook/
      orderbook.hpp       # BookSnapshot, O(1) rolling stats, OrderBookEngine
      replay_engine.cpp   # replays CSV through the engine, times it
    inference/
      infer_onnx.cpp       # loads the ONNX model, benchmarks inference latency
    CMakeLists.txt
  benchmarks/
    latency_compare.py   # Python pipeline vs C++ engine, side by side
  models/                 # trained models + metrics land here
  run_all.sh
```

## Optional: C++ ONNX inference (train in Python, infer in C++)

By default `train.py` saves models with `joblib`. To also export to ONNX
and build the C++ inference benchmark:

```bash
pip install skl2onnx onnxmltools onnxruntime
python3 python/train.py --features data/features.csv --out-dir models --export-onnx

# Install ONNX Runtime C++ (prebuilt release, adjust version/arch as needed):
# https://github.com/microsoft/onnxruntime/releases -> onnxruntime-linux-x64-<ver>.tgz
# Extract it and set ONNXRUNTIME_ROOT to the extracted path, e.g.:
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.17.0
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

./cpp/build/infer_onnx models/clf_h1.onnx 100.0 0.02 100.01 0.01 0.1 0.05 0.03 0.02 0.0001 0.001 0.001 0.002 0.0005 0 0
```

If ONNX Runtime isn't found, CMake just skips `infer_onnx` and builds
`replay_engine` normally -- nothing else in the repo depends on it.

## What to extend next
- Real data (data/README.md) instead of synthetic
- Sequence models (temporal CNN / small LSTM) for the full "curve" (multi-step
  path), not just per-horizon point predictions
- Queue-position-aware backtest (currently ignores fill probability at a
  price level, which matters a lot at HFT horizons)
- Feed the C++ engine from a live WebSocket instead of CSV replay
