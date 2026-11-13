"""
Cross-validates all three independent pricing methods (closed-form,
Monte Carlo, finite-difference PDE) against each other across a grid of
moneyness/maturity, plus sanity checks (put-call parity, Greeks behavior).

This is the actual proof-of-correctness artifact for the project: if all
three independently-derived methods agree to within their expected error
tolerances, that's strong evidence the math (and its implementation) is
right -- much stronger than any single method "looking reasonable."

Run: python3 -m pytest tests/ -v   (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np
import pytest

from black_scholes import bs_price, bs_greeks, put_call_parity_check
from monte_carlo import mc_price, mc_delta_pathwise
from finite_difference import crank_nicolson_price

R = 0.05
SIGMA = 0.2
T = 1.0

GRID = [
    (80.0, 100.0),   # OTM call / ITM put
    (100.0, 100.0),  # ATM
    (120.0, 100.0),  # ITM call / OTM put
]


@pytest.mark.parametrize("S,K", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_matches_closed_form(S, K, option_type):
    bs = bs_price(S, K, R, SIGMA, T, option_type)
    mc, se = mc_price(S, K, R, SIGMA, T, option_type, n_paths=1_000_000, seed=123)
    # 5-sigma tolerance on MC standard error -- should pass essentially always
    # if the implementation is correct; a tighter bound would be flaky.
    assert abs(bs - mc) < 5 * se, f"BS={bs:.4f} MC={mc:.4f} se={se:.5f}"


@pytest.mark.parametrize("S,K", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_finite_difference_matches_closed_form(S, K, option_type):
    bs = bs_price(S, K, R, SIGMA, T, option_type)
    fd, _, _ = crank_nicolson_price(S, K, R, SIGMA, T, option_type, n_S=400, n_T=400)
    # grid-based method: tolerance reflects discretization error, not noise
    assert abs(bs - fd) < 0.02, f"BS={bs:.4f} FD={fd:.4f}"


@pytest.mark.parametrize("S,K", GRID)
def test_put_call_parity(S, K):
    lhs, rhs, diff = put_call_parity_check(S, K, R, SIGMA, T)
    assert diff < 1e-10, f"Put-call parity violated: {lhs:.6f} != {rhs:.6f}"


def test_delta_pathwise_matches_analytical():
    S, K = 100.0, 100.0
    analytical = bs_greeks(S, K, R, SIGMA, T, "call")["delta"]
    mc_delta, se = mc_delta_pathwise(S, K, R, SIGMA, T, "call", n_paths=1_000_000, seed=123)
    assert abs(analytical - mc_delta) < 5 * se


def test_greeks_signs_and_bounds():
    """Sanity checks any quant should be able to state from memory:
    0 <= call delta <= 1, gamma >= 0, vega >= 0 (always, for vanilla options)."""
    S, K = 100.0, 100.0
    g = bs_greeks(S, K, R, SIGMA, T, "call")
    assert 0 <= g["delta"] <= 1
    assert g["gamma"] >= 0
    assert g["vega"] >= 0

    g_put = bs_greeks(S, K, R, SIGMA, T, "put")
    assert -1 <= g_put["delta"] <= 0


def test_deep_itm_call_delta_near_one():
    """Deep ITM call should have delta -> 1 (behaves like the stock itself)."""
    g = bs_greeks(S=200.0, K=100.0, r=R, sigma=SIGMA, T=T, option_type="call")
    assert g["delta"] > 0.97


def test_deep_otm_call_delta_near_zero():
    g = bs_greeks(S=50.0, K=100.0, r=R, sigma=SIGMA, T=T, option_type="call")
    assert g["delta"] < 0.03


def test_american_put_geq_european_put():
    """American puts have an early-exercise premium -> price >= European put,
    always. Checked via the finite-difference solver (the only one of the
    three methods that handles American exercise)."""
    S, K = 100.0, 110.0  # ITM put, where early exercise premium is largest
    euro, _, _ = crank_nicolson_price(S, K, R, SIGMA, T, "put", american=False)
    amer, _, _ = crank_nicolson_price(S, K, R, SIGMA, T, "put", american=True)
    assert amer >= euro - 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
