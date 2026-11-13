"""
Validates the closed-form formulas against the theoretical properties
derived in math/derivation.md, plus sanity checks on the simulation.

Run: python3 -m pytest tests/ -v   (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np
import pytest

from avellaneda_stoikov import (
    reservation_price, optimal_spread, optimal_quotes,
    gamma_to_zero_limit_spread,
)
from simulate import simulate_mid_price, run_market_maker, run_monte_carlo_comparison, summarize

S0, SIGMA, T, K, GAMMA = 100.0, 2.0, 1.0, 1.5, 0.1


def test_reservation_price_equals_mid_when_flat():
    r = reservation_price(S0, q=0, gamma=GAMMA, sigma=SIGMA, T=T, t=0.3)
    assert abs(r - S0) < 1e-10


def test_reservation_price_skews_down_when_long():
    r = reservation_price(S0, q=10, gamma=GAMMA, sigma=SIGMA, T=T, t=0.0)
    assert r < S0


def test_reservation_price_skews_up_when_short():
    r = reservation_price(S0, q=-10, gamma=GAMMA, sigma=SIGMA, T=T, t=0.0)
    assert r > S0


def test_reservation_price_antisymmetric_in_inventory():
    r_long = reservation_price(S0, q=10, gamma=GAMMA, sigma=SIGMA, T=T, t=0.0)
    r_short = reservation_price(S0, q=-10, gamma=GAMMA, sigma=SIGMA, T=T, t=0.0)
    assert abs((S0 - r_long) - (r_short - S0)) < 1e-10


def test_reservation_price_converges_to_mid_near_horizon():
    """As t -> T, (T-t) -> 0, so skew should vanish regardless of inventory."""
    r = reservation_price(S0, q=50, gamma=GAMMA, sigma=SIGMA, T=T, t=T - 1e-9)
    assert abs(r - S0) < 1e-5


def test_spread_positive():
    spread = optimal_spread(GAMMA, SIGMA, T, t=0.0, k=K)
    assert spread > 0


def test_spread_shrinks_as_time_runs_out():
    """Inventory-risk component of the spread -> 0 as t -> T, so the total
    spread should shrink toward the pure-liquidity term 2/gamma*ln(1+gamma/k)."""
    spread_early = optimal_spread(GAMMA, SIGMA, T, t=0.0, k=K)
    spread_late = optimal_spread(GAMMA, SIGMA, T, t=T - 1e-6, k=K)
    assert spread_late < spread_early


def test_gamma_to_zero_limit():
    """See derivation.md Section 6: spread -> 2/k as gamma -> 0."""
    small_gamma_spread = optimal_spread(gamma=1e-8, sigma=SIGMA, T=T, t=0.0, k=K)
    limit = gamma_to_zero_limit_spread(K)
    assert abs(small_gamma_spread - limit) < 1e-4


def test_higher_risk_aversion_widens_spread():
    low_gamma_spread = optimal_spread(gamma=0.01, sigma=SIGMA, T=T, t=0.0, k=K)
    high_gamma_spread = optimal_spread(gamma=0.5, sigma=SIGMA, T=T, t=0.0, k=K)
    assert high_gamma_spread > low_gamma_spread


def test_higher_volatility_widens_spread():
    low_vol_spread = optimal_spread(GAMMA, sigma=1.0, T=T, t=0.0, k=K)
    high_vol_spread = optimal_spread(GAMMA, sigma=4.0, T=T, t=0.0, k=K)
    assert high_vol_spread > low_vol_spread


def test_quotes_straddle_reservation_price():
    bid, ask = optimal_quotes(S0, q=5, gamma=GAMMA, sigma=SIGMA, T=T, t=0.0, k=K)
    r = reservation_price(S0, q=5, gamma=GAMMA, sigma=SIGMA, T=T, t=0.0)
    assert bid < r < ask


def test_simulation_runs_and_tracks_inventory():
    S, dt = simulate_mid_price(S0, SIGMA, T, n_steps=200, seed=1)
    df = run_market_maker(S, dt, A=140.0, k=K, strategy="avellaneda_stoikov",
                           gamma=GAMMA, sigma=SIGMA, seed=2)
    assert len(df) == 200
    assert "final_mark_to_market" in df.attrs
    # inventory should never be wildly large relative to typical fill counts
    assert df["inventory"].abs().max() < 200


def test_as_strategy_controls_inventory_better_than_naive():
    """The headline comparative result: Avellaneda-Stoikov's active inventory
    skew should produce meaningfully tighter inventory control than a fixed
    symmetric spread, across many simulations (not just one lucky path)."""
    results = run_monte_carlo_comparison(n_sims=150, S0=S0, sigma=SIGMA, T=T,
                                          n_steps=150, A=140.0, k=K, gamma=GAMMA, seed=99)
    summary = summarize(results)
    as_inv = summary.loc["avellaneda_stoikov", "mean_max_abs_inventory"]
    naive_inv = summary.loc["naive", "mean_max_abs_inventory"]
    assert as_inv < naive_inv


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
