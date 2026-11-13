"""
Feature engineering for LOB mid-price direction/curve prediction.

Features implemented (standard microstructure literature):
  - order book imbalance (OBI) at multiple depths
  - micro-price (size-weighted mid)
  - bid-ask spread & its rolling stats
  - realized volatility over short rolling windows
  - trade flow imbalance (rolling)
  - multi-horizon forward returns (the "curve" targets)
"""
import numpy as np
import pandas as pd


def micro_price(df):
    return (df["bid_px_0"] * df["ask_sz_0"] + df["ask_px_0"] * df["bid_sz_0"]) / (
        df["bid_sz_0"] + df["ask_sz_0"]
    )


def order_book_imbalance(df, depth=1):
    bid_vol = sum(df[f"bid_sz_{i}"] for i in range(depth))
    ask_vol = sum(df[f"ask_sz_{i}"] for i in range(depth))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)


def build_features(df: pd.DataFrame, horizons=(1, 5, 10, 20, 50)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["mid_price"] = df["mid_price"]
    out["spread"] = df["spread"]
    out["micro_price"] = micro_price(df)
    out["micro_minus_mid"] = out["micro_price"] - out["mid_price"]

    for depth in (1, 3, 5, 10):
        out[f"obi_depth{depth}"] = order_book_imbalance(df, depth=depth)

    # rolling realized volatility of mid-price returns
    ret1 = df["mid_price"].pct_change().fillna(0)
    out["ret_1"] = ret1
    for w in (10, 50, 200):
        out[f"realized_vol_{w}"] = ret1.rolling(w).std().fillna(0)
        out[f"momentum_{w}"] = df["mid_price"].pct_change(w).fillna(0)

    # trade flow imbalance, rolling
    out["trade_flow"] = df["trade_flow"]
    for w in (10, 50):
        out[f"trade_flow_roll_{w}"] = df["trade_flow"].rolling(w).mean().fillna(0)

    # spread dynamics
    out["spread_roll_50"] = df["spread"].rolling(50).mean().bfill()

    # ---- targets: forward returns at multiple horizons = "the curve" ----
    for h in horizons:
        out[f"fwd_ret_{h}"] = df["mid_price"].shift(-h) / df["mid_price"] - 1
        out[f"fwd_dir_{h}"] = np.sign(out[f"fwd_ret_{h}"]).astype("Int64")

    out = out.dropna().reset_index(drop=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/synthetic_lob.csv")
    ap.add_argument("--out", default="data/features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    feats = build_features(df)
    feats.to_csv(args.out, index=False)
    print(f"Wrote {len(feats):,} rows x {feats.shape[1]} cols to {args.out}")
    print(feats.describe().T[["mean", "std", "min", "max"]])
