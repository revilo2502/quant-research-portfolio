"""Plots the fitted implied vol surface (strike x expiry x IV) as a 3D
surface + a 2D smile slice. Run implied_vol.py first to generate the CSV."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    df = pd.read_csv(DATA_DIR / "implied_vol_surface.csv")

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    pivot = df.pivot(index="expiry", columns="strike", values="fitted_iv")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    ax1.plot_surface(X, Y, pivot.values, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Expiry (years)")
    ax1.set_zlabel("Implied Vol")
    ax1.set_title("Fitted Implied Volatility Surface")

    ax2 = fig.add_subplot(1, 2, 2)
    for T in sorted(df["expiry"].unique()):
        sub = df[df["expiry"] == T].sort_values("strike")
        ax2.plot(sub["strike"], sub["fitted_iv"], marker="o", markersize=3,
                 label=f"T={T*365:.0f}d")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Implied Vol")
    ax2.set_title("Volatility Smile by Expiry")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = DATA_DIR / "vol_surface.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
