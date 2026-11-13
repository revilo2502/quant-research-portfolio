"""
Closed-form Black-Scholes pricing + analytical Greeks.

Every formula here corresponds 1:1 to math/derivation.md Section 5-6 --
read that first. This file is the "ground truth" the Monte Carlo and
finite-difference solvers get checked against.
"""
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, r, sigma, T, option_type="call"):
    """Closed-form Black-Scholes price. S=spot, K=strike, r=risk-free rate,
    sigma=volatility, T=time to expiry (years)."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, r, sigma, T, option_type="call"):
    """Analytical Greeks -- see math/derivation.md Section 6 for the derivation."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1  # per unit sigma (not per 1%)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,  # per year; divide by 365 for per-day
        "rho": rho,
    }


def put_call_parity_check(S, K, r, sigma, T):
    """C - P should equal S - K*exp(-rT). Returns (lhs, rhs, abs_diff)."""
    c = bs_price(S, K, r, sigma, T, "call")
    p = bs_price(S, K, r, sigma, T, "put")
    lhs = c - p
    rhs = S - K * np.exp(-r * T)
    return lhs, rhs, abs(lhs - rhs)


if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    call = bs_price(S, K, r, sigma, T, "call")
    put = bs_price(S, K, r, sigma, T, "put")
    greeks = bs_greeks(S, K, r, sigma, T, "call")

    print(f"S={S} K={K} r={r} sigma={sigma} T={T}")
    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")
    print("Call Greeks:", {k: round(v, 5) for k, v in greeks.items()})

    lhs, rhs, diff = put_call_parity_check(S, K, r, sigma, T)
    print(f"\nPut-call parity check: C-P={lhs:.6f}  S-Ke^(-rT)={rhs:.6f}  diff={diff:.2e}")
