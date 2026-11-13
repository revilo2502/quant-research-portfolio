# Option Pricing & Greeks from Stochastic Calculus

Black-Scholes derived from first principles (Itô's lemma → PDE → closed
form) and then verified three independent ways: closed-form algebra, Monte
Carlo simulation, and a finite-difference PDE solver — plus implied
volatility surface fitting.

**Start with `math/derivation.md`.** That's the actual deliverable for
interview prep — being able to reproduce that derivation on a whiteboard is
what gets tested, not the code. The code exists to verify the math is right
and to build intuition, not to replace the derivation.

## Quickstart

```bash
chmod +x run_all.sh
./run_all.sh
```

Runs all three pricers, fits an implied vol surface, plots it, and runs the
20-test cross-validation suite (closed-form vs Monte Carlo vs finite
difference, put-call parity, Greeks sanity checks, American vs European).

## Step by step

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 python/black_scholes.py      # closed-form price + Greeks + put-call parity check
python3 python/monte_carlo.py        # risk-neutral MC simulation, pathwise Delta, bump-and-reprice Greeks
python3 python/finite_difference.py  # Crank-Nicolson PDE solver + American put early-exercise premium
python3 python/implied_vol.py        # Newton-Raphson IV solver + synthetic smile/skew surface
python3 python/plot_surface.py       # 3D vol surface + smile plot -> data/vol_surface.png

python3 -m pytest tests/ -v          # cross-validates all three methods against each other
```

## What's actually being tested (`tests/test_cross_validation.py`)

| Check | Why it matters |
|---|---|
| Monte Carlo price ≈ closed-form (within 5 std errors) | confirms the risk-neutral simulation is unbiased |
| Finite-difference price ≈ closed-form | confirms the PDE discretization is correct |
| Put-call parity exact to 1e-10 | model-free identity; any violation means a bug, not approximation error |
| MC pathwise Delta ≈ analytical Delta | validates the pathwise derivative estimator independently of bump-and-reprice |
| Greeks signs/bounds (0≤Δ_call≤1, Γ≥0, Vega≥0) | the sanity checks you should be able to state from memory |
| Deep ITM/OTM Delta → 1 / 0 | limiting behavior any quant should recognize instantly |
| American put ≥ European put | early-exercise premium must be non-negative, always |

All 20 pass. That's the point — three independently-derived methods landing
on the same answer is real evidence of correctness, not just "the code ran."

## Project structure

```
option-pricing-stochastic-calculus/
  math/
    derivation.md          # Itô's lemma -> replicating portfolio -> PDE -> closed form -> Greeks
  python/
    black_scholes.py        # closed-form price + analytical Greeks (ground truth)
    monte_carlo.py           # risk-neutral GBM simulation, pathwise + bump-and-reprice Greeks
    finite_difference.py     # Crank-Nicolson solver on the PDE directly, handles American exercise
    implied_vol.py            # Newton-Raphson IV solver + synthetic smile/skew/term-structure surface
    plot_surface.py           # 3D vol surface plot
  tests/
    test_cross_validation.py # the 20 tests described above
  data/                       # generated: implied_vol_surface.csv, vol_surface.png
  run_all.sh
```

## Using real market data instead of the synthetic smile

`implied_vol.py`'s `synth_smile()` generates a realistic-shaped vol surface
(skew + smile + term structure) with zero external dependencies, so the
whole pipeline runs standalone. To fit a real surface: pull an option
chain (strike, expiry, bid/ask mid) from a broker API or a free source
(e.g. Yahoo Finance's option chain endpoint via `yfinance`), build the same
`(strike, expiry, option_type, market_price)` rows, and pass that DataFrame
into `fit_implied_vol()` — nothing else changes.

## What to extend next
- **Heston / local volatility model** — Black-Scholes assumes constant
  sigma; the smile you'll fit from real data proves that assumption is
  false. Heston (stochastic vol) or Dupire local vol are the natural next
  derivations.
- **American option pricing via Longstaff-Schwartz** (regression-based
  Monte Carlo) as a second method to cross-check the finite-difference
  American result.
- **Greeks for the whole surface** — compute Delta/Gamma/Vega across all
  strikes/expiries at once, visualize how they change shape (this is what a
  market maker actually watches intraday).
- Tie this into the **Avellaneda-Stoikov market-making** project — the Greeks
  here are exactly the inventory-risk quantities that model manages.
