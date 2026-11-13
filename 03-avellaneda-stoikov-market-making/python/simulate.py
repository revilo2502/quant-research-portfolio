"""
Simulates a market maker quoting continuously against Poisson order flow,
comparing the Avellaneda-Stoikov strategy to a naive fixed-symmetric-spread
baseline.

Matches the model setup in math/derivation.md Section 1 exactly:
  - mid-price: dS_t = sigma * dW_t  (driftless arithmetic Brownian motion)
  - order arrival intensity: lambda(delta) = A * exp(-k*delta)
  - inventory q_t, cash X_t updated on each fill
  - terminal mark-to-market: X_T + q_T * S_T

Discretized with small time steps dt; within each step, a fill occurs with
probability ~ lambda(delta)*dt (thinned Poisson approximation, valid for
small dt -- this is the standard simulation approach for these models).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from avellaneda_stoikov import optimal_quotes, naive_symmetric_quotes

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def simulate_mid_price(S0, sigma, T, n_steps, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0, np.sqrt(dt), n_steps)
    S = S0 + np.concatenate([[0], np.cumsum(sigma * dW)])
    return S, dt


def run_market_maker(S, dt, A, k, strategy="avellaneda_stoikov",
                      gamma=0.1, sigma=2.0, fixed_half_spread=None, seed=None):
    """Runs one simulation path. strategy in {'avellaneda_stoikov', 'naive'}.
    Returns a DataFrame with per-step state (inventory, cash, quotes, fills)."""
    rng = np.random.default_rng(seed)
    n_steps = len(S) - 1
    T = n_steps * dt

    q = 0  # inventory
    X = 0.0  # cash
    records = []

    for i in range(n_steps):
        t = i * dt
        s = S[i]

        if strategy == "avellaneda_stoikov":
            bid, ask = optimal_quotes(s, q, gamma, sigma, T, t, k)
        elif strategy == "naive":
            half_spread = fixed_half_spread if fixed_half_spread is not None else (1 / k)
            bid, ask = naive_symmetric_quotes(s, half_spread)
        else:
            raise ValueError("strategy must be 'avellaneda_stoikov' or 'naive'")

        delta_b = s - bid  # distance of bid from mid
        delta_a = ask - s  # distance of ask from mid

        # thinned-Poisson fill probabilities over this dt
        p_fill_bid = A * np.exp(-k * delta_b) * dt
        p_fill_ask = A * np.exp(-k * delta_a) * dt

        filled_bid = rng.uniform() < p_fill_bid
        filled_ask = rng.uniform() < p_fill_ask

        if filled_bid:
            q += 1
            X -= bid
        if filled_ask:
            q -= 1
            X += ask

        records.append({
            "t": t, "mid": s, "bid": bid, "ask": ask,
            "inventory": q, "cash": X,
            "filled_bid": filled_bid, "filled_ask": filled_ask,
        })

    df = pd.DataFrame(records)
    final_mark = X + q * S[-1]
    df.attrs["final_mark_to_market"] = final_mark
    df.attrs["final_inventory"] = q
    return df


def run_monte_carlo_comparison(n_sims=500, S0=100.0, sigma=2.0, T=1.0, n_steps=200,
                                A=140.0, k=1.5, gamma=0.1, seed=42):
    """Runs n_sims independent paths for both strategies (same mid-price
    path per pair, for a fair comparison) and collects terminal P&L +
    inventory statistics."""
    rng = np.random.default_rng(seed)
    results = {"avellaneda_stoikov": [], "naive": []}

    for sim in range(n_sims):
        path_seed = rng.integers(0, 2**31)
        S, dt = simulate_mid_price(S0, sigma, T, n_steps, seed=path_seed)

        for strategy in ["avellaneda_stoikov", "naive"]:
            df = run_market_maker(S, dt, A, k, strategy=strategy, gamma=gamma,
                                   sigma=sigma, seed=path_seed + 1)
            results[strategy].append({
                "final_pnl": df.attrs["final_mark_to_market"],
                "final_inventory": df.attrs["final_inventory"],
                "max_abs_inventory": df["inventory"].abs().max(),
                "n_fills": int(df["filled_bid"].sum() + df["filled_ask"].sum()),
            })

    return {k: pd.DataFrame(v) for k, v in results.items()}


def summarize(results):
    summary = {}
    for strategy, df in results.items():
        summary[strategy] = {
            "mean_pnl": df["final_pnl"].mean(),
            "std_pnl": df["final_pnl"].std(),
            "sharpe": df["final_pnl"].mean() / df["final_pnl"].std() if df["final_pnl"].std() > 0 else np.nan,
            "mean_abs_final_inventory": df["final_inventory"].abs().mean(),
            "mean_max_abs_inventory": df["max_abs_inventory"].mean(),
            "mean_n_fills": df["n_fills"].mean(),
        }
    return pd.DataFrame(summary).T


if __name__ == "__main__":
    print("Running Monte Carlo comparison: Avellaneda-Stoikov vs naive symmetric spread...")
    results = run_monte_carlo_comparison(n_sims=500)
    summary = summarize(results)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + summary.to_string())

    print("\nKey comparison:")
    as_sharpe = summary.loc["avellaneda_stoikov", "sharpe"]
    naive_sharpe = summary.loc["naive", "sharpe"]
    as_inv = summary.loc["avellaneda_stoikov", "mean_max_abs_inventory"]
    naive_inv = summary.loc["naive", "mean_max_abs_inventory"]
    print(f"  Sharpe:            AS={as_sharpe:.4f}  naive={naive_sharpe:.4f}")
    print(f"  Mean max |inventory|: AS={as_inv:.2f}  naive={naive_inv:.2f}")
    print(f"  -> AS should show lower inventory risk (tighter inventory control) "
          f"{'(confirmed)' if as_inv < naive_inv else '(NOT confirmed -- check params)'}")

    DATA_DIR.mkdir(exist_ok=True)
    results["avellaneda_stoikov"].to_csv(DATA_DIR / "as_results.csv", index=False)
    results["naive"].to_csv(DATA_DIR / "naive_results.csv", index=False)
    summary.to_csv(DATA_DIR / "summary.csv")
    print(f"\nSaved results to {DATA_DIR}")
