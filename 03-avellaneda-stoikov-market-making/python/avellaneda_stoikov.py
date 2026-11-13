"""
Closed-form Avellaneda-Stoikov reservation price and optimal spread.

Every formula here corresponds 1:1 to math/derivation.md Sections 4-5 --
read that first. This is the "ground truth" formula layer; simulate.py is
what actually runs a market maker using these quotes against simulated
order flow.
"""
import numpy as np


def reservation_price(s, q, gamma, sigma, T, t):
    """r(s,q,t) = s - q*gamma*sigma^2*(T-t)  -- see derivation.md Section 4."""
    return s - q * gamma * sigma**2 * (T - t)


def optimal_spread(gamma, sigma, T, t, k):
    """delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
    -- see derivation.md Section 5."""
    inventory_risk_term = gamma * sigma**2 * (T - t)
    liquidity_term = (2 / gamma) * np.log(1 + gamma / k)
    return inventory_risk_term + liquidity_term


def optimal_quotes(s, q, gamma, sigma, T, t, k):
    """Returns (bid, ask) -- symmetric half-spreads around the reservation price."""
    r = reservation_price(s, q, gamma, sigma, T, t)
    spread = optimal_spread(gamma, sigma, T, t, k)
    bid = r - spread / 2
    ask = r + spread / 2
    return bid, ask


def naive_symmetric_quotes(s, half_spread):
    """Baseline: fixed symmetric spread around mid-price, no inventory skew
    and no time dependence. This is what most beginner market-making
    implementations do -- the comparison point for why AS is better."""
    return s - half_spread, s + half_spread


def gamma_to_zero_limit_spread(k):
    """As gamma -> 0: spread -> 2/k (pure liquidity provision, no inventory
    risk management). See derivation.md Section 6 -- a sanity check."""
    return 2 / k


if __name__ == "__main__":
    s, q, gamma, sigma, T, t, k = 100.0, 0, 0.1, 2.0, 1.0, 0.0, 1.5

    print("=== Flat inventory (q=0), t=0 ===")
    bid, ask = optimal_quotes(s, q, gamma, sigma, T, t, k)
    print(f"  mid={s}  bid={bid:.4f}  ask={ask:.4f}  spread={ask-bid:.4f}")
    r = reservation_price(s, q, gamma, sigma, T, t)
    print(f"  reservation price={r:.4f} (should equal mid since q=0)")

    print("\n=== Long inventory (q=+10), t=0 ===")
    bid, ask = optimal_quotes(s, 10, gamma, sigma, T, t, k)
    r = reservation_price(s, 10, gamma, sigma, T, t)
    print(f"  mid={s}  reservation={r:.4f} (< mid: skewed down to encourage selling)")
    print(f"  bid={bid:.4f}  ask={ask:.4f}")

    print("\n=== Short inventory (q=-10), t=0 ===")
    bid, ask = optimal_quotes(s, -10, gamma, sigma, T, t, k)
    r = reservation_price(s, -10, gamma, sigma, T, t)
    print(f"  mid={s}  reservation={r:.4f} (> mid: skewed up to encourage buying)")
    print(f"  bid={bid:.4f}  ask={ask:.4f}")

    print("\n=== Near horizon (t -> T), any inventory ===")
    bid, ask = optimal_quotes(s, 10, gamma, sigma, T, t=0.999, k=k)
    r = reservation_price(s, 10, gamma, sigma, T, t=0.999)
    print(f"  reservation={r:.4f} (should ~= mid: no time left for inventory risk to matter)")
    print(f"  bid={bid:.4f}  ask={ask:.4f}  spread={ask-bid:.4f} (shrinks toward liquidity-only term)")

    print(f"\n=== gamma -> 0 limit check ===")
    small_gamma_spread = optimal_spread(gamma=1e-6, sigma=sigma, T=T, t=t, k=k)
    limit_spread = gamma_to_zero_limit_spread(k)
    print(f"  spread at gamma=1e-6: {small_gamma_spread:.6f}")
    print(f"  theoretical limit 2/k: {limit_spread:.6f}")
    print(f"  diff: {abs(small_gamma_spread - limit_spread):.2e}")
