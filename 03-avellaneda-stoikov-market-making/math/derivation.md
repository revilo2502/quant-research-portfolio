# Avellaneda-Stoikov Optimal Market Making: Full Derivation

Based on Avellaneda & Stoikov (2008), "High-frequency trading in a limit
order book." This is the model that formalizes the core market-maker
problem: quote a bid and ask around the mid-price to earn the spread, while
managing the risk that your inventory accumulates in one direction if the
price moves against you before you can unwind it.

---

## 1. Setup

**Mid-price** follows arithmetic Brownian motion (driftless — the market
maker is assumed to have no directional view):

$$dS_t = \sigma\, dW_t$$

**The market maker** posts a bid $S_t - \delta^b_t$ and ask $S_t + \delta^a_t$
continuously. Market orders (buyers/sellers) arrive as independent Poisson
processes whose arrival rate decays with how far the quote sits from the
mid-price — a wider quote is less likely to get hit:

$$\lambda^b(\delta^b) = A e^{-k\delta^b}, \qquad \lambda^a(\delta^a) = A e^{-k\delta^a}$$

This exponential intensity is itself a modeling choice (not derived — it's
assumed), motivated by order book depth typically decaying roughly
exponentially away from the touch. $A$ is the baseline fill intensity at
zero distance, $k$ controls how fast intensity decays with distance.

**Inventory** $q_t$ = (filled buy orders) − (filled sell orders), i.e. the
market maker's net position. **Cash** $X_t$ evolves by $\pm$(quote price)
whenever a quote is filled.

**Objective**: maximize expected utility of terminal wealth at a fixed
horizon $T$, under exponential (CARA) utility with risk-aversion $\gamma$:

$$\max_{\delta^b, \delta^a} \; \mathbb{E}\left[-e^{-\gamma(X_T + q_T S_T)}\right]$$

The $q_T S_T$ term marks the terminal inventory to market — this is what
makes holding inventory at time $T$ costly if $\gamma > 0$: uncertainty in
$S_T$ translates into utility risk proportional to $q_T^2$.

---

## 2. The value function and HJB equation

Define the value function $u(x, s, q, t) = \max \mathbb{E}_t[-e^{-\gamma(X_T + q_T S_T)}]$
given state $(X_t, S_t, q_t) = (x, s, q)$ at time $t$. By the dynamic
programming principle, $u$ satisfies the **Hamilton-Jacobi-Bellman (HJB)
equation**:

$$u_t + \tfrac{1}{2}\sigma^2 u_{ss} + \max_{\delta^b}\lambda^b(\delta^b)\left[u(x-s+\delta^b, s, q+1, t) - u\right] + \max_{\delta^a}\lambda^a(\delta^a)\left[u(x+s+\delta^a, s, q-1, t) - u\right] = 0$$

**Reading this equation:** the first two terms are the usual diffusion
generator (Itô's lemma applied to the $S_t$ process, same mechanism as the
Black-Scholes derivation — see the option-pricing project). The two `max`
terms are the *impulse control* parts: each one says "choose the quote
distance $\delta$ to maximize [probability of fill] × [value gained from
the jump in cash/inventory that a fill causes]." This is the defining
feature of market-making models vs. option pricing: the control variable
(quote placement) affects a **jump process** (random fills), not a
diffusion.

---

## 3. The key simplifying ansatz

Guess the value function has the separable (exponential-CARA-consistent) form:

$$u(x, s, q, t) = -\exp\left(-\gamma\left(x + qs + \theta(s, q, t)\right)\right)$$

Substituting into the HJB equation and simplifying (this is the long,
mechanical part — each $\max_\delta$ term becomes a one-dimensional
optimization once the ansatz is plugged in, since the jump sizes only
affect the exponent linearly) leads, after a further small-risk
approximation, to the closed-form results below. The full algebra is in
the original paper; the result is what matters for application.

---

## 4. The reservation price

The central result: a market maker who is currently **long** inventory
($q > 0$) should skew both quotes *down* (be more eager to sell, less eager
to buy) to encourage mean-reversion of inventory toward zero — and vice
versa when short. This skew is captured by the **reservation price**
(sometimes called the "indifference price"):

$$\boxed{r(s, q, t) = s - q\gamma\sigma^2(T-t)}$$

This is *not* the mid-price — it's the price at which the market maker is
indifferent to a marginal trade, given their current inventory and risk
aversion. Notice:
- If $q = 0$ (flat), $r = s$ — no skew, quote symmetrically around mid.
- If $q > 0$ (long), $r < s$ — reservation price below mid, meaning quotes
  get skewed down, making the ask more attractive to hit (sell down
  inventory) and the bid less attractive (slow down buying more).
- The skew is proportional to $\gamma$ (more risk-averse ⇒ more aggressive
  inventory control) and $\sigma^2$ (more volatile ⇒ more urgency to
  offload risk) and $(T-t)$ (more time left ⇒ more risk still to bear, so
  more urgency early, tapering to zero at $T$).

---

## 5. The optimal spread

The total quoted spread (distance from reservation price to each quote,
summed) comes out to:

$$\boxed{\delta^a + \delta^b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)}$$

Two components:
- $\gamma\sigma^2(T-t)$ — the **same inventory-risk term** as the
  reservation-price skew; grows the spread as risk aversion, volatility,
  or remaining time increase.
- $\frac{2}{\gamma}\ln(1+\gamma/k)$ — a term that depends only on risk
  aversion and the order-flow decay parameter $k$, **not on inventory or
  time**. This is the "liquidity-provision" component of the spread: even a
  flat, infinite-horizon market maker charges this much just for the
  privilege of being picked off by informed order flow (adverse selection,
  implicitly, via the $k$ parameter controlling how "thin" quotes near the
  touch are).

**Individual quotes**, symmetric around the reservation price:

$$\delta^a_t = r(s,q,t) \text{ 's half of the spread above}, \qquad \text{ask} = r + \tfrac{1}{2}(\delta^a+\delta^b)$$
$$\text{bid} = r - \tfrac{1}{2}(\delta^a+\delta^b)$$

(The paper's exact formula splits the spread asymmetrically in a more
refined version; the symmetric split around $r$ is the standard simplified
implementation used in practice and in this project's code.)

---

## 6. The $\gamma \to 0$ limit (sanity check)

As risk aversion $\gamma \to 0$: the reservation-price skew vanishes (no
inventory urgency — the MM is risk-neutral about terminal wealth), and the
spread collapses to $\frac{2}{k}$ — purely a function of order-flow decay,
independent of volatility. This is the "pure liquidity provision, no risk
management" limit, worth checking numerically (see `tests/` — it's one of
the cross-checks in this project).

---

## 7. What interviewers actually probe

1. **"Why does the market maker skew quotes with inventory?"** — to induce
   mean reversion in inventory before the horizon, trading off expected
   spread capture against the increasing marginal cost of inventory risk
   under CARA utility.
2. **"What's the difference between the reservation price and the
   mid-price?"** — reservation price is the inventory-adjusted indifference
   price; mid-price is the market's price. The spread is centered on the
   former, not the latter — a common point of confusion.
3. **"Derive the limiting behavior as $T \to t$"** (near the horizon) — the
   $(T-t)$ factor goes to zero, so skew and the inventory-risk component
   of spread both vanish right at expiry: the MM stops caring about
   inventory because there's no time left to be exposed to it. This is
   exactly analogous to Theta decay for options — the mechanism is the
   same (diminishing time-at-risk).
4. **"Why exponential order arrival intensity? What if it weren't
   exponential?"** — exponential is assumed, not derived, for
   mathematical convenience (it makes the $\max_\delta$ step in the HJB
   solvable in closed form). A good answer acknowledges this and discusses
   what changes with a different intensity function — generally, you lose
   the closed form and need numerical/dynamic programming solutions
   instead.
5. **Connects directly to the option-pricing project**: $\gamma\sigma^2$
   appearing in both the reservation-price skew and the spread is
   structurally the same kind of quantity as Vega/Gamma risk in options —
   a market maker's inventory in the underlying is directly analogous to a
   delta-hedger's unhedged Greek exposure.
