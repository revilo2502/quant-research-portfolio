"""Plots a single example path (quotes + inventory over time) and the
Monte Carlo PnL distribution comparison. Run simulate.py first."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from avellaneda_stoikov import optimal_quotes, naive_symmetric_quotes
from simulate import simulate_mid_price, run_market_maker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def plot_example_path():
    S0, sigma, T, n_steps = 100.0, 2.0, 1.0, 300
    A, k, gamma = 140.0, 1.5, 0.1

    S, dt = simulate_mid_price(S0, sigma, T, n_steps, seed=7)
    df_as = run_market_maker(S, dt, A, k, "avellaneda_stoikov", gamma=gamma, sigma=sigma, seed=8)
    df_naive = run_market_maker(S, dt, A, k, "naive", seed=8)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(df_as["t"], df_as["mid"], color="black", lw=1, label="mid price")
    axes[0].plot(df_as["t"], df_as["bid"], color="tab:blue", lw=0.8, alpha=0.7, label="AS bid/ask")
    axes[0].plot(df_as["t"], df_as["ask"], color="tab:blue", lw=0.8, alpha=0.7)
    axes[0].plot(df_naive["t"], df_naive["bid"], color="tab:orange", lw=0.8, alpha=0.5, label="naive bid/ask")
    axes[0].plot(df_naive["t"], df_naive["ask"], color="tab:orange", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("Price")
    axes[0].set_title("Quotes: Avellaneda-Stoikov (skews with inventory) vs Naive (fixed)")
    axes[0].legend(fontsize=8)

    axes[1].plot(df_as["t"], df_as["inventory"], color="tab:blue", label="AS inventory")
    axes[1].plot(df_naive["t"], df_naive["inventory"], color="tab:orange", label="naive inventory")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("Inventory")
    axes[1].set_xlabel("time")
    axes[1].set_title("Inventory over time -- AS should mean-revert faster toward 0")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = DATA_DIR / "example_path.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_pnl_distribution():
    as_df = pd.read_csv(DATA_DIR / "as_results.csv")
    naive_df = pd.read_csv(DATA_DIR / "naive_results.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(as_df["final_pnl"], bins=30, alpha=0.6, label="Avellaneda-Stoikov", color="tab:blue")
    ax.hist(naive_df["final_pnl"], bins=30, alpha=0.6, label="Naive symmetric", color="tab:orange")
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Frequency")
    ax.set_title("Terminal P&L distribution across simulations\n(AS: higher Sharpe via tighter inventory control)")
    ax.legend()

    plt.tight_layout()
    out = DATA_DIR / "pnl_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_example_path()
    plot_pnl_distribution()
