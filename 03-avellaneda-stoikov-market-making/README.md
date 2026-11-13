# Avellaneda-Stoikov Optimal Market Making

The classic optimal market-making model (Avellaneda & Stoikov, 2008):
derive the HJB equation for a CARA-utility market maker facing Poisson
order flow, solve for the closed-form reservation price and optimal
spread, then prove in simulation that actively skewing quotes by inventory
beats a naive fixed-spread baseline — and quantify by how much.

**Start with `math/derivation.md`.** Same philosophy as the option-pricing
project: the derivation is the actual interview-prep deliverable (this
model, or the Black-Scholes HJB it structurally parallels, is one of the
most commonly asked "derive this" questions at prop shops). The code
verifies the math and shows the strategy actually working, it doesn't
replace understanding the derivation.

This project also directly reuses ideas from two earlier projects: the
reservation-price skew is driven by the same $\gamma\sigma^2$ risk
quantity that shows up in options Greeks (`option-pricing-stochastic-
calculus`), and the order-flow/fill-probability model here is the natural
counterpart to the order book features from the HFT curve-prediction
project.

## Quickstart

```bash
chmod +x run_all.sh
./run_all.sh
```

Runs the closed-form sanity checks, a 500-path Monte Carlo comparison
(Avellaneda-Stoikov vs. naive fixed-spread quoting), generates plots, runs
the 13-test cross-validation suite, and builds + runs the C++ engine (if
cmake is available).

## Step by step

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 python/avellaneda_stoikov.py   # closed-form reservation price + spread, several scenarios
python3 python/simulate.py              # Monte Carlo: AS vs naive, 500 simulated paths
python3 python/plot_results.py          # example path + P&L distribution plots
python3 -m pytest tests/ -v             # 13 tests validating the formulas + strategy comparison

# C++ engine (optional, needs cmake + g++):
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/as_simulate 500 200 42      # args: n_sims n_steps seed
```

## Headline result (from my test run, 500 simulated paths)

| | Avellaneda-Stoikov | Naive symmetric |
|---|---|---|
| Sharpe ratio | **10.42** | 5.13 |
| Mean max \|inventory\| | **3.95** | 10.55 |
| Mean terminal P&L | 65.19 | 68.94 |

The naive strategy actually earns slightly *higher* mean P&L (wider,
non-skewed quotes capture a bit more raw spread) but with roughly **2x the
volatility and 2.7x the peak inventory risk** — exactly the trade-off the
theory predicts: AS gives up a little expected spread capture in exchange
for actively managing inventory risk, netting a much better risk-adjusted
outcome. This is the single most important number to be able to explain in
an interview: *why would a strategy with lower raw P&L be the better
strategy* — because Sharpe (risk-adjusted return), not raw P&L, is what a
market maker actually optimizes for, and that's precisely what the CARA
utility objective in the derivation encodes.

## What's tested (`tests/test_avellaneda_stoikov.py`)

| Check | What it validates |
|---|---|
| Reservation price = mid when inventory is flat | q=0 case should recover vanilla symmetric quoting |
| Reservation price skews down when long, up when short | the core inventory-control mechanism |
| Skew is antisymmetric in inventory | $r(q) - s = -(r(-q) - s)$, a direct algebraic property of the formula |
| Reservation price → mid as $t \to T$ | no time left ⇒ no inventory risk to manage, formula should reflect that |
| Spread shrinks as time runs out | same mechanism as option Theta decay — risk exposure shrinks with time remaining |
| $\gamma \to 0$ limit: spread → $2/k$ | the "pure liquidity provision" limiting case from the derivation |
| Higher risk aversion / volatility ⇒ wider spread | monotonicity any quant should expect without computing |
| Quotes straddle the reservation price | bid < r < ask, structurally required |
| AS inventory control beats naive, across 150 simulations | the actual comparative claim, not just one lucky path |

## Project structure

```
avellaneda-stoikov-market-making/
  math/
    derivation.md           # HJB equation -> reservation price -> optimal spread, full derivation
  python/
    avellaneda_stoikov.py    # closed-form formulas (ground truth)
    simulate.py               # Poisson order-flow Monte Carlo, AS vs naive comparison
    plot_results.py           # example path + P&L distribution plots
  cpp/
    as_simulate.cpp           # fast C++ port of the same simulation (~95k sims/sec)
    CMakeLists.txt
  tests/
    test_avellaneda_stoikov.py
  data/                        # generated plots + CSVs
  run_all.sh
```

## What to extend next
- **Asymmetric spread splitting** — this project uses the common
  simplification of splitting the spread symmetrically around the
  reservation price; the original paper's full solution splits it
  asymmetrically based on the derivative of the value function. Worth
  implementing for the full rigor.
- **Stochastic volatility** — replace constant $\sigma$ with a GARCH or
  Heston-driven volatility process (ties to the option-pricing project's
  "what to extend next" note) and see how the optimal spread responds to
  vol clustering.
- **Real fill-rate calibration** — fit $A$ and $k$ (the order-flow
  intensity parameters) to real order book data instead of assuming them,
  using the LOB engine from the HFT curve-prediction project.
- **Finite inventory limits / risk limits** — add a hard inventory cap and
  see how the optimal strategy needs to adapt near the boundary (the
  closed-form solution assumes unconstrained inventory).
