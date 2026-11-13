#!/usr/bin/env bash
# Runs every piece: closed-form, Monte Carlo, finite-difference, implied vol
# surface + plot, and the full cross-validation test suite.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Python deps =="
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== 2. Closed-form Black-Scholes pricing + Greeks =="
python3 python/black_scholes.py

echo
echo "== 3. Monte Carlo pricing + Greeks (risk-neutral simulation) =="
python3 python/monte_carlo.py

echo
echo "== 4. Crank-Nicolson finite-difference PDE solver =="
python3 python/finite_difference.py

echo
echo "== 5. Implied volatility surface (Newton-Raphson fit) =="
python3 python/implied_vol.py

echo
echo "== 6. Plotting the vol surface =="
python3 python/plot_surface.py

echo
echo "== 7. Cross-validation test suite (closed-form vs MC vs FD) =="
python3 -m pytest tests/ -v

echo
echo "All done. Key outputs:"
echo "  math/derivation.md              - full hand derivation (read this first)"
echo "  data/implied_vol_surface.csv    - fitted vol surface"
echo "  data/vol_surface.png            - 3D surface + smile plot"
echo "  tests/                          - 20 cross-validation tests, all three pricing methods agree"
