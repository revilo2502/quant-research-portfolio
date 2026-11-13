"""
Synthetic limit order book (LOB) data generator.

Why synthetic: real L2/L3 tick data (LOBSTER, exchange feeds) needs paid
licenses or API keys. This generator produces a realistic mid-price process
(mean-reverting microstructure noise + short-horizon momentum regimes) plus
a matching synthetic order book (10 levels each side) so the whole pipeline
runs end-to-end with zero external dependencies.

Swap this out for real data later: data/README.md explains how to plug in
Binance/Bybit L2 websocket captures or a LOBSTER CSV with the same schema.
"""
import numpy as np
import pandas as pd
import argparse


def generate_lob(n_steps=200_000, seed=42, tick_size=0.01, start_price=100.0):
    rng = np.random.default_rng(seed)

    # --- mid-price process: regime-switching drift + mean-reverting noise ---
    mid = np.empty(n_steps)
    mid[0] = start_price
    regime_len = 500
    n_regimes = n_steps // regime_len + 2
    drifts = rng.choice([-1, 0, 1], size=n_regimes, p=[0.3, 0.4, 0.3]) * 0.0008
    vol = 0.02

    regime_idx = 0
    steps_in_regime = 0
    noise_state = 0.0
    for t in range(1, n_steps):
        if steps_in_regime >= regime_len:
            regime_idx += 1
            steps_in_regime = 0
        steps_in_regime += 1

        drift = drifts[regime_idx]
        noise_state = 0.7 * noise_state + rng.normal(0, vol)
        mid[t] = mid[t - 1] + drift + noise_state * 0.3
        mid[t] = max(mid[t], 1.0)

    mid = np.round(mid / tick_size) * tick_size

    # --- synthetic order book around mid price ---
    n_levels = 10
    spread = np.round(rng.uniform(1, 3, n_steps)) * tick_size
    best_bid = mid - spread / 2
    best_ask = mid + spread / 2

    book = {"timestamp": np.arange(n_steps), "mid_price": mid, "spread": spread}
    base_size = 50

    for lvl in range(n_levels):
        bid_px = best_bid - lvl * tick_size
        ask_px = best_ask + lvl * tick_size
        decay = np.exp(-0.15 * lvl)
        bid_sz = np.maximum(
            rng.poisson(base_size * decay, n_steps) + rng.normal(0, 5, n_steps), 1
        )
        ask_sz = np.maximum(
            rng.poisson(base_size * decay, n_steps) + rng.normal(0, 5, n_steps), 1
        )
        book[f"bid_px_{lvl}"] = np.round(bid_px, 2)
        book[f"bid_sz_{lvl}"] = bid_sz.astype(int)
        book[f"ask_px_{lvl}"] = np.round(ask_px, 2)
        book[f"ask_sz_{lvl}"] = ask_sz.astype(int)

    # trade flow: signed volume correlated with next-step drift (gives model signal)
    future_ret = np.diff(mid, append=mid[-1])
    trade_flow = np.sign(future_ret) * rng.poisson(20, n_steps) + rng.normal(0, 8, n_steps)
    book["trade_flow"] = trade_flow

    df = pd.DataFrame(book)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/synthetic_lob.csv")
    args = ap.parse_args()

    df = generate_lob(n_steps=args.n_steps, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(df.head())
