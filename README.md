# Quant Research Portfolio — Oliver Williams Robin Manickam

Three connected projects spanning market microstructure, derivatives
pricing, and market making — each built the same way: a hand derivation of
the underlying math first, then an implementation that's cross-validated
multiple independent ways, with results (including the inconvenient ones)
reported honestly rather than tuned to look clean.

| # | Project | Core math | What it proves |
|---|---|---|---|
| 1 | [HFT Curve Predictor](01-hft-curve-predictor/) | order book microstructure, multi-horizon forecasting | can build a full data→model→backtest→systems pipeline, Python + C++ |
| 2 | [Option Pricing & Greeks](02-option-pricing-stochastic-calculus/) | Itô's Lemma → Black-Scholes PDE → closed form | can derive and verify a classic quant result three independent ways |
| 3 | [Avellaneda-Stoikov Market Making](03-avellaneda-stoikov-market-making/) | HJB equation, CARA utility, optimal control | understands inventory risk management, the core of real market making |

**Read each project's `math/` (or equivalent) derivation before the code** —
in every case, the derivation is the actual interview-relevant artifact;
the code exists to verify the math and build intuition, not replace
understanding it.

## How the three connect

- The **reservation-price skew** in project 3 ($\gamma\sigma^2$) is
  structurally the same risk quantity as **Vega/Gamma** in project 2 — both
  measure how a position's value responds to the combination of volatility
  and time-to-horizon. Project 3's README spells this out directly.
- The **order book imbalance / fill-probability features** in project 1 are
  the natural real-data calibration target for project 3's $A$ and $k$
  parameters (currently assumed, not fit) — a concrete next step linking
  all three.
- All three follow the same validation discipline: closed-form math checked
  against simulation, baseline comparisons kept even when the sophisticated
  method doesn't win outright (project 1's negative backtest, project 3's
  naive strategy actually having higher raw P&L), and every honest finding
  (project 1's CSV-parsing latency result) reported rather than hidden.

## Running everything

Each project is self-contained with its own `run_all.sh`:

```bash
cd 01-hft-curve-predictor && ./run_all.sh && cd ..
cd 02-option-pricing-stochastic-calculus && ./run_all.sh && cd ..
cd 03-avellaneda-stoikov-market-making && ./run_all.sh && cd ..
```

Or run all three in sequence:

```bash
chmod +x run_all.sh
./run_all.sh
```

Each needs Python 3 + a C/C++ toolchain (g++, cmake) for the systems
components; every script installs its own Python dependencies into a local
virtualenv and degrades gracefully if cmake/lightgbm/onnxruntime aren't
present (see each project's own README for specifics).

## Honest scope note

None of these are production trading systems — no exchange connectivity, no
colocation, no real capital at risk, and project 1 uses synthetic data by
default (real-data instructions are in its README). They're research-grade
implementations of real, textbook/paper-level quant models, built to
demonstrate mathematical fluency and end-to-end engineering ability, not to
claim production HFT infrastructure.
