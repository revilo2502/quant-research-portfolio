"""
Compares feature-computation latency: Python/pandas pipeline vs the C++
order book engine (cpp/orderbook/replay_engine). This is the headline
"why C++" result for the project writeup.

Run after: cmake --build (produces cpp/build/replay_engine)
"""
import subprocess
import time
import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from features import build_features  # noqa: E402


def bench_python(csv_path, n_rows=None):
    df = pd.read_csv(csv_path, nrows=n_rows)
    t0 = time.perf_counter()
    feats = build_features(df)
    elapsed = time.perf_counter() - t0
    return {
        "rows_processed": len(feats),
        "elapsed_sec": elapsed,
        "rows_per_sec": len(feats) / elapsed if elapsed > 0 else float("inf"),
        "avg_latency_ns_per_row": elapsed * 1e9 / len(feats) if len(feats) else None,
    }


def bench_cpp(binary_path, csv_path, n_rows=None):
    args = [str(binary_path), str(csv_path)]
    if n_rows:
        args.append(str(n_rows))
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    out = {}
    for line in result.stdout.strip().splitlines():
        k, v = line.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/synthetic_lob.csv")
    ap.add_argument("--cpp-binary", default="cpp/build/replay_engine")
    ap.add_argument("--n-rows", type=int, default=50_000)
    args = ap.parse_args()

    print(f"Benchmarking on {args.n_rows:,} rows...\n")

    py_result = bench_python(args.csv, n_rows=args.n_rows)
    print("Python/pandas pipeline:")
    for k, v in py_result.items():
        print(f"  {k}: {v}")

    cpp_binary = Path(args.cpp_binary)
    if not cpp_binary.exists():
        print(f"\nC++ binary not found at {cpp_binary}. Build it first:")
        print("  cd cpp && cmake -B build -S . && cmake --build build -j")
        return

    print()
    cpp_result = bench_cpp(cpp_binary, args.csv, n_rows=args.n_rows)
    print("C++ order book engine:")
    for k, v in cpp_result.items():
        print(f"  {k}: {v}")

    speedup = py_result["rows_per_sec"] / cpp_result["rows_per_sec"]
    print(f"\n==> C++ engine is {1/speedup:.1f}x faster than the Python/pandas pipeline"
          if speedup < 1 else
          f"\n==> Python pipeline was faster in this run ({speedup:.2f}x) -- "
          f"check build flags / -O3 / --march=native support")

    Path("benchmarks").mkdir(exist_ok=True)
    with open("benchmarks/latency_results.json", "w") as f:
        json.dump({"python": py_result, "cpp": cpp_result}, f, indent=2)


if __name__ == "__main__":
    main()
