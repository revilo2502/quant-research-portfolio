"""
Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

Solves the boxed PDE from math/derivation.md Section 4 directly on a grid,
instead of using the closed-form solution -- this is the third, fully
independent verification path (closed-form algebra, Monte Carlo simulation,
numerical PDE), and it's the technique that generalizes to options with no
closed form (American options, barriers, path-dependent payoffs).

We solve in log-price space x = ln(S) (so the PDE has constant coefficients,
avoiding the S^2 term which would otherwise need a non-uniform or carefully
scaled grid), backward in time from the terminal payoff at t=T to the price
at t=0.

Crank-Nicolson = average of explicit and implicit Euler in time -- it's
unconditionally stable (unlike pure explicit) and second-order accurate in
both time and space (unlike pure implicit, which is only first-order in
time). The tridiagonal system at each time step is solved directly (no
iteration needed) via the Thomas algorithm, which is what scipy's banded
solver does under the hood.
"""
import numpy as np
from scipy.linalg import solve_banded


def crank_nicolson_price(S0, K, r, sigma, T, option_type="call",
                          S_max_mult=4.0, n_S=400, n_T=400, american=False):
    """Returns (price_at_S0, S_grid, V_grid_at_t0) via Crank-Nicolson on the
    log-price Black-Scholes PDE."""
    x_min, x_max = np.log(K / S_max_mult), np.log(K * S_max_mult)
    dx = (x_max - x_min) / n_S
    dt = T / n_T

    x_grid = np.linspace(x_min, x_max, n_S + 1)
    S_grid = np.exp(x_grid)

    # terminal payoff (t = T)
    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)

    # PDE in log-space: dV/dt + (r - sigma^2/2) dV/dx + sigma^2/2 d2V/dx2 - rV = 0
    # (this is the standard transform of the boxed PDE; constant coefficients
    # in x make the finite-difference stencil simple and uniform)
    nu = r - 0.5 * sigma**2
    a = 0.25 * dt * (sigma**2 / dx**2 - nu / dx)       # sub-diagonal coeff
    b = -0.5 * dt * (sigma**2 / dx**2 + r)              # diagonal coeff
    c = 0.25 * dt * (sigma**2 / dx**2 + nu / dx)        # super-diagonal coeff

    n = n_S + 1
    # Implicit (LHS) tridiagonal matrix M1: (I - A) V^{n} = (I + A) V^{n+1}
    lower = -a * np.ones(n)
    diag_lhs = (1 - b) * np.ones(n)
    upper = -c * np.ones(n)

    diag_rhs = (1 + b) * np.ones(n)

    # boundary rows: far from the strike, use linear/Dirichlet boundary conditions
    lower[0] = 0; diag_lhs[0] = 1; upper[0] = 0
    lower[-1] = 0; diag_lhs[-1] = 1; upper[-1] = 0

    ab = np.zeros((3, n))  # banded storage for solve_banded

    for step in range(n_T):
        t_remaining = T - step * dt  # time remaining AFTER this backward step
        rhs = np.empty(n)
        rhs[1:-1] = a * V[:-2] + diag_rhs[1:-1] * V[1:-1] + c * V[2:]

        # Dirichlet boundaries: deep ITM/OTM behavior is known analytically
        if option_type == "call":
            rhs[0] = 0.0
            rhs[-1] = S_grid[-1] - K * np.exp(-r * t_remaining)
        else:
            rhs[0] = K * np.exp(-r * t_remaining) - S_grid[0]
            rhs[-1] = 0.0

        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag_lhs
        ab[2, :-1] = lower[1:]

        V = solve_banded((1, 1), ab, rhs)

        if american:
            if option_type == "call":
                V = np.maximum(V, S_grid - K)
            else:
                V = np.maximum(V, K - S_grid)

    price = np.interp(np.log(S0), x_grid, V)
    return price, S_grid, V


if __name__ == "__main__":
    from black_scholes import bs_price

    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    bs_call = bs_price(S, K, r, sigma, T, "call")
    fd_call, S_grid, V_grid = crank_nicolson_price(S, K, r, sigma, T, "call", n_S=400, n_T=400)

    print(f"Black-Scholes call price:      {bs_call:.4f}")
    print(f"Crank-Nicolson FD call price:  {fd_call:.4f}")
    print(f"Diff: {abs(bs_call - fd_call):.5f}")

    bs_put = bs_price(S, K, r, sigma, T, "put")
    fd_put, _, _ = crank_nicolson_price(S, K, r, sigma, T, "put", n_S=400, n_T=400)
    print(f"\nBlack-Scholes put price:       {bs_put:.4f}")
    print(f"Crank-Nicolson FD put price:   {fd_put:.4f}")
    print(f"Diff: {abs(bs_put - fd_put):.5f}")

    # American put -- no closed form, this is where finite differences earn their keep
    fd_put_american, _, _ = crank_nicolson_price(S, K, r, sigma, T, "put", n_S=400, n_T=400, american=True)
    print(f"\nAmerican put (FD, no closed form to compare): {fd_put_american:.4f}")
    print(f"  (should be >= European put price of {fd_put:.4f}, early-exercise premium: "
          f"{fd_put_american - fd_put:.4f})")
