# Track A — closure of the ADMM stabilization track

**Documentation only. No solves. The track is closed.**

## What the track established

**S1 — criterion validity.** `stationarity_pf` binds on 93 of 93 cycles but is *satisfied
at termination* in all nine preserved runs (terminal slacks 1.1035 / 1.0644 / 1.0005) with
a positive log-slack trend throughout: a binding criterion behaving as one should, not a
broken test. The rho/tolerance coupling is real but roughly half-compensated,
`p ~ 0.42–0.68`. Comparisons at termination are circular, because the stopping rule halts
at slack just above 1. *Rediscovery recorded:* the mechanism was already documented in
`p59_b_adaptive.py`'s docstring, with an A/B attached; S1's results were not.

**S2 — cost model.** 51 solves per cycle. Cold uncapped `>= 21` cycles (lower bound only —
the trajectory was stopped by configuration); warm from template 6 cycles / 306 solves;
warm continuation 1 cycle / 51 solves. Initialization dominates rho as a cost lever, and
the P5.10 sweep varied the weaker of the two. Residual balancing was **already evaluated**
in P5.9-B — 16 cycles fixed against 6 adaptive at `rho_pf` 1000, `rho_pf` driven to 131.687.

**AB1 — adaptive rho on the cold path.** The treatment "converged" at 32 cycles with
`rho_pf` collapsing to 3.4683 while the control failed at the 50-cycle cap. **The
comparison is invalid**: dividing rho out, the control's iterate step was *smaller* than
the treatment's at its declared convergence (2.907e-04 against 6.079e-04), the two arms
being held to standards differing by 86.50x. Both motion-based standards are gameable by
rho in opposite directions — `rho*|dz|` inflates with rho, bare `|dz|` deflates — so
neither measures anything rho-independently. **The objective criterion is the only
rho-independent member of the composite test, and it is the member that refused the
control.**

> **Refined 2026-09-12 by Track C1.** Rho-independent, yes — but **not an optimality
> test**. It tests the objective's *rate of change*, so a path whose increments decay
> quickly stops early regardless of where it stands in objective terms. **All three members
> of the composite test are motion-based**, and no reweighting of them yields an optimality
> criterion. C1 measured the consequence: warm and cold stop 8.5e6 apart at 1e-4.

**The 2x2.** One candidate, one formulation, four cells. `Q` agrees across structurally
different initializations to **~0.1%** (cold vs warm_fixed −0.1275%, vs warm_adaptive
−0.1094%), with a **systematic** sign: the templated oracle sits ~0.11% high. Independence
costs **6.60x** (1683 solves against 255). Adaptation's large objective effect is
**cold-specific** (−39.8% cold at matched cycle 32, −0.018% warm); warm it is a 29% cost
saving. C3 proved numerically inert on this instance, reproducing a pre-C3 run to sixteen
digits.

## The resolution finding — the disqualifier

`objective_tolerance = 827,945` against a stabilized best-to-second gap of **32.87**: a
factor of **25,188**, and still **25x** the pre-stabilized `PLANNING_SIGNAL = 33,031`.
`Q` is defined only to within ~±8.3e5 by its own stopping rule, while the quantity to be
resolved is 32.87.

The observed cross-depth uncertainty of **22.09** is four orders of magnitude below that
bound. **The path-identity mechanism** explains the discrepancy: identical code paths
visiting identical iterates stop at identical points, so the stopping slack cancels
exactly in a difference. P5.10's ranking stability is therefore a consequence of
**determinism, not of resolution**.

What that generalises to, and what it does not:

- **It does generalise** to reruns of the same candidates on the same code and data —
  which is what P5.10 measured, and why 22.09 is small.
- **It does not generalise** to any comparison where the paths differ: a different
  candidate, a solver retry, a code edit, a different initialization. Each can move a
  stopping point by up to a tolerance width, i.e. by ~25,000 times the signal.

**Consequence for cut-based methods:** `Q`'s *resolution*, not its convexity, is the
immediate disqualifier. A cut or a finite-difference slope built on differences of `Q`
inherits an uncertainty four orders of magnitude larger than the effects being ranked.
Convexity remains unestablished and is now a second-order question.

### The codebase already encodes the gate it fails

`benders_parameters.py:17` sets `minimum_signal_to_noise_ratio = 10.0`, and the case file
sets `benders.finite_difference.enabled = false`. The current signal-to-resolution ratio is
`32.87 / 827,945 = 3.97e-05`, which fails that gate by a factor of **2.5e5** — more than
five orders of magnitude.

Someone built the right check and switched it off. It is independent corroboration of the
resolution finding, arrived at from the opposite direction.

## Ninth rule promoted

> Report a difference with its resolution. Any difference of two iteratively-computed
> quantities is reported with the error implied by where each computation stopped, and a
> difference smaller than that error is indeterminate, not a result.

## Closure reason

- ~~`Q` is well defined across structurally different initializations (~0.1%, systematic).~~
  **RETRACTED by C1**: the two initializations do not share a fixed point; the offset grows
  to 1.03% at one decade tighter, determinate at 65.6x its error bar. The closure stands on
  the remaining reasons, and the `Q` path-dependence is now an **open finding** carried into
  the outer-layer discussion rather than a settled property.
- ADMM converges: 32 cycles cold adaptive, 4–6 warm.
- Local solves fail once in 1,095 (and once in 4,284 across the AB1/X22 stages).
- A warm evaluation costs 255 solves.
- **No remaining lever has an identified decision-relevant payoff.**

The cycle-21 mechanism stays open and unpursued; the breadth line stays closed at n = 3.
