"""
Simple event-driven backtester for the h=1 direction model.

Strategy: go long/short/flat according to predicted direction each step,
hold for 1 step, pay a fixed round-trip cost (spread + fee) per trade.
Reports PnL, Sharpe, hit rate, and turnover -- and compares against a
naive "always flat" and "always follow last tick" baseline.

This is intentionally simple (no latency/queue modeling yet) -- it's the
scaffold. benchmarks/latency_compare.py is what's actually about latency.
"""
import argparse
import json

import numpy as np
import pandas as pd
import joblib


def run_backtest(df, feat_cols, clf, fee_bps=0.5, spread_cost_mult=1.0):
    X = df[feat_cols].values
    pred_dir = clf.predict(X)

    mid = df["mid_price"].values
    spread = df["spread"].values
    fwd_ret = df["fwd_ret_1"].values

    position = pred_dir  # +1 / 0 / -1
    gross_pnl = position * fwd_ret

    # cost: half-spread (as fraction of price) + fee, charged whenever position changes
    pos_change = np.abs(np.diff(position, prepend=0))
    half_spread_frac = (spread / mid) / 2 * spread_cost_mult
    fee_frac = fee_bps / 10_000
    cost = pos_change * (half_spread_frac + fee_frac)

    net_pnl = gross_pnl - cost
    cum_pnl = np.cumsum(net_pnl)

    sharpe = net_pnl.mean() / (net_pnl.std() + 1e-12) * np.sqrt(252 * 6.5 * 3600)  # rough annualization for tick data
    hit_rate = np.mean(np.sign(net_pnl[position != 0]) == np.sign(position[position != 0])) if np.any(position != 0) else 0
    turnover = pos_change.sum()

    # baselines
    baseline_flat_pnl = np.zeros_like(net_pnl)
    baseline_momentum = np.sign(np.diff(mid, prepend=mid[0]))
    baseline_mom_pnl = baseline_momentum * fwd_ret - np.abs(np.diff(baseline_momentum, prepend=0)) * (half_spread_frac + fee_frac)

    return {
        "n_steps": len(df),
        "total_net_pnl": float(net_pnl.sum()),
        "cum_pnl_final": float(cum_pnl[-1]),
        "annualized_sharpe_approx": float(sharpe),
        "hit_rate": float(hit_rate),
        "turnover": float(turnover),
        "avg_cost_per_trade": float(cost[cost > 0].mean()) if np.any(cost > 0) else 0.0,
        "baseline_momentum_total_pnl": float(baseline_mom_pnl.sum()),
    }, cum_pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    with open(f"{args.model_dir}/metrics.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    clf = joblib.load(f"{args.model_dir}/clf_h{args.horizon}.joblib")

    # use the same trailing test slice as training (last ~1/6 of data, time-ordered)
    split = int(len(df) * 5 / 6)
    test_df = df.iloc[split:].reset_index(drop=True)

    results, cum_pnl = run_backtest(test_df, feat_cols, clf)

    print(f"Backtest on {results['n_steps']:,} out-of-sample ticks (horizon={args.horizon}):")
    for k, v in results.items():
        print(f"  {k:30s}: {v}")

    np.save(f"{args.model_dir}/backtest_cum_pnl.npy", cum_pnl)
    with open(f"{args.model_dir}/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved cum_pnl array + results.json to {args.model_dir}/")


if __name__ == "__main__":
    main()
