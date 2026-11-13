# Black-Scholes: Full Derivation

This is the derivation to actually reproduce by hand on a whiteboard, not just
read. Each step states *why*, not just *what*, since that's what gets probed
in interviews.

---

## 1. Setup: the stock price process

Assume the stock price $S_t$ follows geometric Brownian motion (GBM) under
the real-world measure $\mathbb{P}$:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

where $W_t$ is a standard Brownian motion, $\mu$ is the drift, $\sigma$ is
the (constant) volatility.

We want to price a derivative $V(S_t, t)$ written on this stock — e.g. a
European call with payoff $V(S_T, T) = \max(S_T - K, 0)$.

---

## 2. Itô's Lemma

$V(S_t, t)$ is a (twice-differentiable) function of a stochastic process, so
plain calculus doesn't apply — we need Itô's Lemma.

**Itô's Lemma (statement):** if $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ and
$V = V(S_t, t)$ is $C^{2,1}$, then

$$dV = \left(\frac{\partial V}{\partial t} + \mu S_t \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2}\right)dt + \sigma S_t \frac{\partial V}{\partial S}\, dW_t$$

**Why the extra term vs. the chain rule:** a Taylor expansion of $V$ gives

$$dV = \frac{\partial V}{\partial t}dt + \frac{\partial V}{\partial S}dS + \frac{1}{2}\frac{\partial^2 V}{\partial S^2}(dS)^2 + \dots$$

Ordinary calculus drops the $(dS)^2$ term since it's second-order and would
vanish as $dt \to 0$. But for Brownian motion, $(dW_t)^2$ doesn't vanish —
it has quadratic variation $(dW_t)^2 = dt$ (this is the key, non-obvious
fact of stochastic calculus; it's why "Brownian motion has infinite
variation but finite quadratic variation"). So:

$$(dS_t)^2 = \sigma^2 S_t^2 (dW_t)^2 + O(dt^{3/2}) = \sigma^2 S_t^2\, dt$$

Substituting into the Taylor expansion and keeping all $O(dt)$ terms gives
Itô's Lemma above. The $\frac{1}{2}\sigma^2 S^2 V_{SS}$ term is the entire
reason option pricing isn't ordinary calculus.

---

## 3. The replicating portfolio / no-arbitrage argument

Construct a portfolio $\Pi$ that is **long one option** and **short $\Delta$
shares of stock**:

$$\Pi = V - \Delta S$$

Its change over $dt$:

$$d\Pi = dV - \Delta\, dS$$

Substitute $dV$ from Itô's Lemma and $dS = \mu S\,dt + \sigma S\,dW$:

$$d\Pi = \left(V_t + \mu S V_S + \tfrac{1}{2}\sigma^2S^2V_{SS}\right)dt + \sigma S V_S\,dW - \Delta(\mu S\,dt + \sigma S\,dW)$$

**Choose $\Delta = V_S$** (the option's Delta) specifically to cancel the
$dW$ (stochastic) terms:

$$d\Pi = \left(V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS}\right)dt$$

This portfolio is now **instantaneously riskless** — no $dW$ term, no
dependence on $\mu$ (the real-world drift drops out entirely; this is *why*
option pricing doesn't need an estimate of expected stock return, only
volatility — the single most commonly tested conceptual point in quant
interviews).

---

## 4. No-arbitrage ⇒ riskless portfolio earns the risk-free rate

Since $\Pi$ is riskless, absence of arbitrage requires:

$$d\Pi = r\Pi\, dt = r(V - S V_S)\, dt$$

Setting the two expressions for $d\Pi$ equal:

$$V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS} = r(V - SV_S)$$

Rearranged, this is the **Black-Scholes PDE**:

$$\boxed{\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} - rV = 0}$$

with terminal condition $V(S,T) = \max(S-K, 0)$ for a call.

---

## 5. Solving the PDE: closed form

A change of variables ($\tau = T-t$, $x = \ln S$, then a further shift to
remove the first-derivative term) reduces this PDE to the standard **heat
equation** $\frac{\partial u}{\partial \tau} = \frac{1}{2}\sigma^2
\frac{\partial^2 u}{\partial x^2}$, which has a known Gaussian solution. Un-
winding the substitutions gives the **Black-Scholes formula**:

$$C(S,t) = S\,N(d_1) - Ke^{-r(T-t)}N(d_2)$$
$$P(S,t) = Ke^{-r(T-t)}N(-d_2) - S\,N(-d_1)$$

where

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \qquad d_2 = d_1 - \sigma\sqrt{T-t}$$

and $N(\cdot)$ is the standard normal CDF.

**Risk-neutral interpretation** (the faster way to re-derive this in an
interview without solving a PDE): the no-arbitrage argument above is
equivalent to saying that under a measure $\mathbb{Q}$ where
$dS_t = rS_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}}$ (drift replaced by $r$),
the option price is the discounted expected payoff:

$$V(S,t) = e^{-r(T-t)}\,\mathbb{E}^{\mathbb{Q}}\left[\max(S_T - K, 0)\mid S_t\right]$$

Since $S_T$ is log-normal under $\mathbb{Q}$, this expectation can be
computed in closed form — giving the same formula. This is also *exactly*
why Monte Carlo pricing (code side) works: simulate $S_T$ under
$\mathbb{Q}$, average the discounted payoff.

---

## 6. The Greeks (derived, not just stated)

All Greeks are partial derivatives of $C$ w.r.t. its inputs. Differentiating
the closed-form formula (using $N'(d_1)S = N'(d_2)Ke^{-r(T-t)}$, a useful
identity that falls out of the algebra):

| Greek | Formula (call) | Meaning |
|---|---|---|
| Delta $\Delta = \partial C/\partial S$ | $N(d_1)$ | sensitivity to spot |
| Gamma $\Gamma = \partial^2 C/\partial S^2$ | $\dfrac{N'(d_1)}{S\sigma\sqrt{T-t}}$ | sensitivity of Delta to spot |
| Vega $= \partial C/\partial \sigma$ | $S\sqrt{T-t}\,N'(d_1)$ | sensitivity to volatility |
| Theta $= \partial C/\partial t$ | $-\dfrac{S N'(d_1)\sigma}{2\sqrt{T-t}} - rKe^{-r(T-t)}N(d_2)$ | time decay |
| Rho $= \partial C/\partial r$ | $K(T-t)e^{-r(T-t)}N(d_2)$ | sensitivity to rates |

Note Gamma and Vega are identical in form up to a factor — both come from
the same $N'(d_1)$ (the "convexity" term) and both are **highest for
at-the-money, short-dated options**, which is why ATM options are where
dealers run the most inventory risk and market makers (see the
Avellaneda-Stoikov project) need to actively hedge.

---

## 7. What interviewers actually probe

1. **"Derive Black-Scholes from scratch"** — the full argument above, usually
   stopping at the PDE (not the heat-equation solving) unless it's a very
   math-heavy shop.
2. **"Why does $\mu$ (real-world drift) not appear in the formula?"** — because
   the replicating portfolio is constructed to be riskless, so by
   no-arbitrage it must earn $r$ regardless of the stock's real-world drift;
   equivalently, we're pricing under $\mathbb{Q}$, not $\mathbb{P}$.
3. **"What does Itô's Lemma's second-order term represent economically?"** —
   convexity / Gamma. It's why options have value beyond intrinsic value even
   with zero drift.
4. **"Put-call parity"** — $C - P = S - Ke^{-r(T-t)}$, derivable directly
   from the formula or, faster, from a model-free replicating argument
   (long call + short put = forward contract). Good to have both ready.
5. **Mental math**: being able to approximate $N(d_1)$ and reason about
   Greeks' signs/magnitudes without a calculator, under time pressure.
