"""
Monte Carlo option pricing under the risk-neutral measure Q.

Implements exactly the risk-neutral expectation from math/derivation.md
Section 5:  V = exp(-rT) * E^Q[payoff(S_T)]

S_T is simulated via the exact (not Euler-discretized) solution of GBM:
  S_T = S_0 * exp((r - sigma^2/2) T + sigma * sqrt(T) * Z),  Z ~ N(0,1)
This is exact because GBM has a known closed-form solution (no discretization
error from the simulation itself -- only Monte Carlo sampling error remains,
which is why this should converge to the Black-Scholes price as N -> inf).

Greeks via two independent methods so they cross-check each other:
  1. Pathwise derivative estimator (for Delta) -- differentiate the payoff
     function w.r.t. S_0 analytically, using common random numbers.
  2. Finite-difference bump-and-reprice (for all Greeks) -- the general
     fallback that works for any payoff, at the cost of more noise.
"""
import numpy as np


def simulate_terminal_price(S0, r, sigma, T, n_paths, seed=None, antithetic=True):
    """Exact GBM terminal price simulation with optional antithetic variates
    (variance reduction: halves the effective noise for the same n_paths)."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion), z


def mc_price(S0, K, r, sigma, T, option_type="call", n_paths=1_000_000, seed=42):
    ST, _ = simulate_terminal_price(S0, r, sigma, T, n_paths, seed=seed)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr


def mc_delta_pathwise(S0, K, r, sigma, T, option_type="call", n_paths=1_000_000, seed=42):
    """Pathwise derivative estimator for Delta.

    d/dS0 [exp(-rT) * max(S_T - K, 0)] = exp(-rT) * 1{S_T > K} * dS_T/dS0
    and since S_T = S0 * exp(...), dS_T/dS0 = S_T / S0. This is exact in
    expectation (not a finite-difference approximation) -- a standard
    technique worth knowing cold for interviews.
    """
    ST, _ = simulate_terminal_price(S0, r, sigma, T, n_paths, seed=seed)
    dST_dS0 = ST / S0
    if option_type == "call":
        indicator = (ST > K).astype(float)
    else:
        indicator = -(ST < K).astype(float)
    pathwise = np.exp(-r * T) * indicator * dST_dS0
    return pathwise.mean(), pathwise.std(ddof=1) / np.sqrt(n_paths)


def mc_greeks_bump(S0, K, r, sigma, T, option_type="call", n_paths=200_000, seed=42,
                    bump_S=0.01, bump_sigma=0.001, bump_T=1 / 365, bump_r=0.0001):
    """Finite-difference 'bump and reprice' Greeks using common random numbers
    (same seed for bumped and base runs) to cancel Monte Carlo noise in the
    difference -- without common random numbers these would be far too noisy
    to be useful."""
    base, _ = mc_price(S0, K, r, sigma, T, option_type, n_paths, seed)

    up, _ = mc_price(S0 + bump_S, K, r, sigma, T, option_type, n_paths, seed)
    down, _ = mc_price(S0 - bump_S, K, r, sigma, T, option_type, n_paths, seed)
    delta = (up - down) / (2 * bump_S)
    gamma = (up - 2 * base + down) / (bump_S**2)

    vega_up, _ = mc_price(S0, K, r, sigma + bump_sigma, T, option_type, n_paths, seed)
    vega = (vega_up - base) / bump_sigma

    theta_dn, _ = mc_price(S0, K, r, sigma, T - bump_T, option_type, n_paths, seed)
    theta = -(base - theta_dn) / bump_T  # decay as T decreases

    rho_up, _ = mc_price(S0, K, r + bump_r, sigma, T, option_type, n_paths, seed)
    rho = (rho_up - base) / bump_r

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


if __name__ == "__main__":
    from black_scholes import bs_price, bs_greeks

    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    N = 2_000_000

    bs_call = bs_price(S, K, r, sigma, T, "call")
    mc_call, mc_se = mc_price(S, K, r, sigma, T, "call", n_paths=N)
    print(f"Black-Scholes call price: {bs_call:.4f}")
    print(f"Monte Carlo call price:   {mc_call:.4f}  (stderr={mc_se:.5f}, n_paths={N:,})")
    print(f"Diff: {abs(bs_call - mc_call):.5f}  ({abs(bs_call - mc_call) / mc_se:.2f} std errors)")

    bs_g = bs_greeks(S, K, r, sigma, T, "call")
    mc_delta, mc_delta_se = mc_delta_pathwise(S, K, r, sigma, T, "call", n_paths=N)
    print(f"\nDelta -- Black-Scholes: {bs_g['delta']:.5f}  "
          f"MC pathwise: {mc_delta:.5f} (stderr={mc_delta_se:.5f})")

    mc_g = mc_greeks_bump(S, K, r, sigma, T, "call", n_paths=200_000)
    print("\nAll Greeks via MC bump-and-reprice vs Black-Scholes:")
    for k in ["delta", "gamma", "vega", "theta", "rho"]:
        print(f"  {k:6s}: BS={bs_g[k]:10.5f}   MC={mc_g[k]:10.5f}")
