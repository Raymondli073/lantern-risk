# Methodology

Notation: K factors, daily factor moves `f_t` (K-vector), factor covariance `S` (daily),
portfolio NAV `V`.

## 1. Factors (`lantern/factors/universe.py`, `build.py`)

| Kind | Unit of a move | Examples |
|---|---|---|
| `return` | daily log return | EQ.US = ln(SPY_t / SPY_{t-1}); STYLE.MOM = r(MTUM) − r(SPY); FX.JPY = −r(USDJPY) |
| `rate` | change in yield, decimal | UST.10Y = Δ(^TNX)/100 |
| `spread` | change in spread, decimal | CREDIT.IG = −(r_LQD − (D_LQD/D_IEF) r_IEF) / D_LQD |
| `vol` | change in implied vol, decimal | VOL.VIX = Δ(VIX)/100 |

Series are aligned to the NYSE calendar. 24/7 and FX series take their last value on or
before each date, so weekend moves are booked on Monday. Moves larger than a kind-specific
cap are treated as bad prints. When a factor's primary series starts late, it is back-filled
before that start with `β · proxy`, where β is the no-intercept regression slope on the
overlap. For example, STYLE.MOM is back-filled with the Fama-French `Mom` factor before
MTUM's 2013 launch.

## 2. Covariance (`covariance.py`)

Zero-mean, exponentially weighted moments over a lookback of up to 10 years. Volatility and
correlation use different half-lives:

```
σ_i²   = Σ_t w^v_t f_it² / Σ_t w^v_t                     w^v_t = 2^(−age/HL_vol),  HL_vol = 63d
ρ_ij   = Σ_t w^c_t f_it f_jt / √(Σ w^c f_it² m_jt · Σ w^c f_jt² m_it),               HL_corr = 252d
S      = D ρ D,  then eigenvalue clipping to the nearest PSD matrix (diagonal restored)
```

The sums are pairwise-complete (`m` = observed mask), so short-history factors use only
the dates they exist. The **stressed** covariance is the equal-weighted covariance over
Sep 2008 to Mar 2009 and Feb to Apr 2020. A factor absent in the stress window keeps its
long-run correlations and gets a volatility scaled by the median stress/normal volatility
ratio.

## 3. Exposures (`exposures.py`)

For equity-like securities, daily log returns `y_t` are regressed on factor moves with
EWMA weights (half-life 252 days, 3-year window):

1. Standardize the weighted design. Ridge estimate `b̂` (α = 1e-3).
2. Adaptive LASSO: rescale column j by `|b̂_j|^γ` (γ = 1) and run a coordinate-descent LASSO
   path. Choose the penalty by BIC with the noise variance from the full fit and the
   effective sample size `n_eff = 1/Σ w²`.
3. Refit the selected factors by weighted least squares, which removes the shrinkage bias.
4. Specific volatility is the EWMA standard deviation of the residual.

Bonds and options get structural exposures (below), not regressions.

## 4. Instruments (`instruments/`)

**Curve.** Zero rate `z(t) = Σ_k W_k(t) y_k`, with linear interpolation weights between the
key-rate nodes (3M, 5Y, 10Y, 30Y) and flat extrapolation outside them.

**Bonds.** `PV = F Σ_i c_i exp(−(z(t_i) + s) t_i)`. The key-rate exposures are exact
derivatives: `∂PV/∂y_k = −F Σ_i c_i t_i W_k(t_i) e^{…}`. They sum to the curve duration.
The spread exposure is `−F Σ c_i t_i e^{…}`. In scenarios the bond is fully repriced, so
convexity is captured. An optional idiosyncratic spread volatility adds specific risk.

**Options.** Black-Scholes-Merton with the rate taken from the curve at expiry and a dividend
yield. Implied volatility is `σ = β_vol · VIX + spread` and moves by `β_vol · ΔVIX`. The
underlying's return is its factor-implied return plus its own specific shock, which is
shared with the underlying position. Linear exposures are delta × underlying betas plus
vega on VOL.VIX. Scenarios reprice exactly.

**Equities and ETFs.** `P&L = MV (exp(b'f + u·z) − 1)`.

## 5. Ex-ante risk (`risk/parametric.py`)

With `e = Σ_positions e_p` ($ per unit factor move) and specific terms `a_ik` ($ per unit of
specific shock k, unit vol `u_k`):

```
σ² = e'Se + Σ_k (Σ_i a_ik)² u_k²
factor k contribution    = e_k (Se)_k / σ
position i contribution  = e_i'Se / σ + Σ_k a_ik u_k² A_k / σ
```

Both decompositions are Euler allocations, so they add up to σ. Results are annualized with
√252 and expressed in % of NAV. Tracking error applies the same formula to portfolio minus
benchmark, with the benchmark scaled to the portfolio's NAV.

## 6. Simulation and VaR (`simulation/engine.py`, `risk/var.py`)

| Simulator | Factor draw | Specific draw |
|---|---|---|
| Gaussian | `L z √h`, `LL' = S` | N(0, h) |
| Student-t | `L z √h · √((ν−2)/χ²_ν)`, variance-matched (ν = 5) | same χ² mixing, so tails are joint |
| Filtered HS | bootstrap dates of `f_t / σ_{t−1}` (EWMA), × today's σ, summed over h days | bootstrapped standardized residuals |
| Regime mixture | Gaussian with prob 1−p (current S), p (stressed S) | ×1.5 in stress |

Every draw is fully revalued. VaR is the α-quantile of the losses and ES is the mean loss
beyond it. The ES contribution of position i is `E[−PnL_i | L ≥ VaR]`, and the contributions
add up to ES. **Historical simulation** applies each historical h-day window of factor moves,
together with the actual security residuals, to today's portfolio.

**Backtests.** Rolling one-day VaR (500-day historical or EWMA-normal) is compared with the
realized hypothetical P&L. The tests are Kupiec's unconditional-coverage LR and
Christoffersen's first-order-Markov independence LR. The Basel zone uses the exceptions in
the last 250 days: green ≤ 4, yellow 5 to 9, red ≥ 10.

**Risk cone.** Linear multi-period paths from the chosen simulator, monthly steps.

## 7. Scenarios (`scenarios/`)

Given explicit shocks `s` on the variable set k, scaled to an h-day horizon (`S_h = h S`):

```
μ_u   = S_uk S_kk⁻¹ s                     implied moves of all other factors
Σ_u|k = S_uu − S_uk S_kk⁻¹ S_ku           their uncertainty
PnL   = full revaluation at (s, μ_u)
±95%  = 1.96 √(e'Σ_u|k e + specific variance of unshocked keys)
d²    = s' S_kk⁻¹ s ~ χ²(|k|)            plausibility; return period = 1 / (P(χ² ≥ d²) · 252/h)
```

A **security shock** augments the system with that security's return:
`Cov(f, r_i) = S b_i` and `Var(r_i) = b_i'S b_i + u_i²`. The security then moves exactly by
the shock, and its implied specific return is `s_i − b_i'μ_f`.

**Historical replays** sum the daily factor moves between the episode's dates. Factors that
did not exist at the time are filled by conditional expectation, the same way vendors proxy
missing history.

## 8. Explosiveness (`systematic/explosive.py`)

For a price level `Y_t`, the DF regression `ΔY_t = μ + β Y_{t−1} + e_t` is fitted on every
window `[r1, t]` with `t − r1 ≥ r0` (r0 = 36 months):

```
BSADF_t = sup_{r1} t-stat(β̂; r1, t)
```

Prefix sums of `Y`, `Y²`, `ΔY`, `ΔY²` and `YΔY` make each window O(1), so BSADF_t costs
O(t). Critical values come from Monte Carlo under `Y_t = Σ_{s≤t} (ε_s + 1/T)`: the
quantiles of the sup-ADF sequence `SADF_t = sup_{s≤t} ADF_0^s`, from 20,000 paths with
T = 1000. They are packaged in `systematic/resources/`. The indicator is
`I_t = 1{BSADF_t > cv_t, r_t > 0}`. PSY date-stamping sets the origination at the first
crossing above the critical value.

**Common explosiveness** (`common.py`). The first principal component of the standardized
log-price panel is tracked in real time by CCIPCA (Weng et al. 2003). Each observation is
projected on the loading available when it arrives, and the resulting factor is tested
with BSADF.

**Bubble-burst scenarios** (`scenarios/bubble.py`). For each explosive target, the shock is
−α × ln(P_now / P_before-origination) with α ∈ {0.25, 0.5, 1}, propagated under the
stressed covariance.

## 9. Factor timing (`systematic/signals.py`)

| Strategy | Rule |
|---|---|
| TSFM (Ehsani & Linnainmaa 2022) | long factors with a positive trailing 12-month return |
| TSFE (Li & Urga) | long factors with `I_t = 1` |

**L** holds the signalled factors at equal weight, or cash when none is signalled.
**LS** is long the signalled factors and short the rest, at equal weight in each leg.
Positions are formed at month-end t and earn month t + 1. Costs are cost/2 per unit of
one-way turnover, where `cost_bps` is the round-trip cost. t-statistics and FF3 alphas use
Newey-West standard errors with 6 lags.

## 10. Portfolio construction (`systematic/optimizer.py`)

The asset covariance is `Σ = B S B' + diag(u²)`, where B holds the fitted betas.

- **Factor targeting** minimizes `(w − w_ref)'Σ(w − w_ref) + λ Σ_k ((B'w)_k − τ_k)² σ_k²`
  subject to `1'w = 1` and `lo ≤ w ≤ hi`.
- **Risk parity** uses the log-barrier ERC, with a bounded fallback.
- **Minimum variance** and **maximum diversification** are solved by SLSQP.
