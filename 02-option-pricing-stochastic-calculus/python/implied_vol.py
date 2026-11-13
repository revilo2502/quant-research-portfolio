"""
Implied volatility via Newton-Raphson, and a synthetic vol surface showing
the smile/skew that real markets exhibit (which Black-Scholes itself cannot
produce -- BS assumes constant sigma, so the fact that implied vol varies by
strike/expiry in real markets IS the statement that BS's assumptions are
wrong, and is a direct talking point in interviews).

No live market data feed is wired in here (would need a brokerage/exchange
API) -- synth_smile() generates a realistic-shaped surface (skew + smile +
term structure flattening) so the fitting code can be demonstrated and
tested end-to-end. Swap in a real option chain (e.g. via a broker API or a
downloaded CBOE dataset) by replacing synth_smile()'s output with real
(strike, expiry, market_price) tuples -- fit_implied_vol() doesn't change.
"""
from pathlib import Path

import numpy as np
from black_scholes import bs_price, bs_greeks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def implied_vol_newton(market_price, S, K, r, T, option_type="call",
                        sigma0=0.2, tol=1e-8, max_iter=100):
    """Newton-Raphson: sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n).
    Vega is always positive for a valid option, so this converges quickly
    (usually < 10 iterations) and is the standard method -- bisection is the
    fallback when vega is near zero (deep ITM/OTM, very short-dated)."""
    sigma = sigma0
    for _ in range(max_iter):
        price = bs_price(S, K, r, sigma, T, option_type)
        vega = bs_greeks(S, K, r, sigma, T, option_type)["vega"]
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-10:
            break  # fall through to bisection
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)  # keep sigma positive
    return _implied_vol_bisect(market_price, S, K, r, T, option_type)


def _implied_vol_bisect(market_price, S, K, r, T, option_type="call",
                         lo=1e-4, hi=5.0, tol=1e-8, max_iter=200):
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = bs_price(S, K, r, mid, T, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def synth_smile(S0=100.0, r=0.03, seed=7):
    """Generates a synthetic but realistic implied-vol surface: skew (higher
    vol for OTM puts / low strikes -- the well-documented equity 'leverage
    effect' / crash-insurance premium) + smile + term structure that
    flattens with maturity. Returns a DataFrame of (strike, expiry, true_iv,
    market_price) you can feed straight into fit_implied_vol()."""
    import pandas as pd
    rng = np.random.default_rng(seed)

    strikes = np.linspace(70, 130, 25)
    expiries = np.array([30, 60, 90, 180, 365]) / 365.0

    rows = []
    for T in expiries:
        moneyness = np.log(strikes / S0)
        atm_vol = 0.20 + 0.03 * (1 - np.exp(-3 * T))  # term structure: slight upward slope
        skew = -0.35 * moneyness  # negative skew: low strikes (puts) have higher vol
        smile = 0.5 * moneyness**2  # convexity away from ATM
        term_decay = np.exp(-1.5 * T)  # smile flattens for longer expiries
        iv = atm_vol + term_decay * (skew + smile)
        iv = np.clip(iv + rng.normal(0, 0.003, len(strikes)), 0.03, None)

        for K, true_iv in zip(strikes, iv):
            option_type = "put" if K <= S0 else "call"
            price = bs_price(S0, K, r, true_iv, T, option_type)
            rows.append({"strike": K, "expiry": T, "option_type": option_type,
                         "true_iv": true_iv, "market_price": price})

    return pd.DataFrame(rows), S0, r


def fit_implied_vol(df, S0, r):
    """Fits implied vol for every row via Newton-Raphson, returns df with a
    'fitted_iv' column. This is what you'd run on real market option-chain
    data (strike, expiry, bid/ask mid) to reconstruct the vol surface."""
    df = df.copy()
    df["fitted_iv"] = df.apply(
        lambda row: implied_vol_newton(
            row["market_price"], S0, row["strike"], r, row["expiry"], row["option_type"]
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    df, S0, r = synth_smile()
    fitted = fit_implied_vol(df, S0, r)

    fitted["abs_error"] = (fitted["fitted_iv"] - fitted["true_iv"]).abs()
    print(f"Fitted implied vol for {len(fitted)} (strike, expiry) pairs")
    print(f"Max |fitted_iv - true_iv| = {fitted['abs_error'].max():.6f}  "
          f"(should be ~0 -- confirms Newton-Raphson recovers the exact vol "
          f"used to generate the synthetic prices)")

    print("\nSample smile at T=30d:")
    sub = fitted[np.isclose(fitted["expiry"], 30 / 365)]
    print(sub[["strike", "true_iv", "fitted_iv"]].to_string(index=False))

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "implied_vol_surface.csv"
    fitted.to_csv(out_path, index=False)
    print(f"\nSaved full surface to {out_path}")
