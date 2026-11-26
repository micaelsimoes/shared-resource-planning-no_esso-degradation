# Track C1 — one decade of objective tolerance, 1e-3 → 1e-4

**Both cells converged inside the cap. Guards clean, identities exact. The outcome falls
OUTSIDE the three predeclared classes, and it retracts the 2x2's headline.**

Frozen spec `data/SRP1/Results/P514C/frozen_c0_tolerance_sweep_v1_43da5cb6.json`
(`43da5cb6`), frozen before either cell ran. The case file was never edited; the tolerance
was overridden in the harness.

## 1. The two cells

| | cold 1e-3 (2x2 baseline) | cold 1e-4 | warm 1e-3 (2x2 baseline) | warm 1e-4 |
|---|---|---|---|---|
| cycles | 32 | **68** | 4 | **5** |
| recourse | 826,829,641.1 | **819,145,341.2** | 827,735,214.6 | **827,653,036.6** |
| terminal per-cycle change | 706,604 | 47,545 | 212,489 | 82,178 |
| solves | 1683 | **3519** | 255 | 306 |
| final `rho_pf` | 3.4683 | 3.4683 | 88.889 | 88.889 |
| cap (90) hit | — | no | — | no |

Solve identities exact in both new cells: `3519 = 51x68 + 51`, `306 = 51x5 + 51`.
Zero blocked solves. `rho_pf` endpoints unchanged by the tolerance, as they should be —
the two mechanisms are independent, and that is a passing internal check.

## 2. The offset, with its error bar (ninth rule)

| tolerance | offset (warm − cold) | error bar | offset / error | verdict |
|---|---|---|---|---|
| 1e-3 | 905,573.5 | 919,092.4 | **0.99** | **INDETERMINATE** |
| 1e-4 | **8,507,695.4** | 129,722.8 | **65.6** | **DETERMINATE** |

**The offset grew by 9.39x** — where proportional shrink would have given ~0.1x — and it
crossed from indeterminate to firmly resolvable. As a fraction of the objective it went
from 0.1094% to **1.0279%**.

**This outcome is outside the three predeclared classes.** The spec predeclared
proportional shrink, plateau, or indeterminate. Growth was not among them, and it is
recorded as such rather than silently mapped onto "plateau" — though its implication is
the plateau implication, amplified roughly tenfold.

## 3. What it means: the two initializations do NOT share a fixed point

At a tighter tolerance the cold path descended a **further 7,684,300 over 36 extra
cycles**, while the warm path moved only **82,178 in one extra cycle**. Tightening did not
bring the two together; it pulled them apart.

**The 2x2's headline is retracted.** "`Q` agrees across structurally different
initializations to ~0.1%" is **false**. It was already flagged as indeterminate under the
ninth rule (offset 0.99x its own error bar); C1 now shows it wrong in substance. The
apparent agreement was a **coincidence of two early stopping points**, not agreement about
`Q`.

What survives from the 2x2: the *direction* is unchanged and now firmly resolved — the
templated oracle is **systematically high**, by 1.03% at 1e-4 rather than 0.11%.

## 4. The mechanism, and a correction to Track A

The stopping rule tests the **per-cycle change**, i.e. a rate, not proximity to an
optimum. The warm path's increments decay quickly because the template starts it inside a
basin; the cold path's decay slowly (realised ratio **0.8451**). For *any* tolerance the
warm path therefore stops early, and tightening by a decade bought warm one cycle and cold
thirty-six.

Note where the two cells stopped relative to their own thresholds: warm at **99.3%** of its
tolerance — a marginal stop — and cold at 58%.

**This corrects Track A.** Track A recorded that "the objective criterion is the only
rho-independent member of the composite test". That remains true, but it is **not** an
optimality test: it is a motion-based test on the objective's rate of change. **All three
members of the composite test are motion-based.** The remedy direction is therefore
reinforced and broadened: an optimality-based criterion — a KKT residual on the original
coupled problem — is required, and no reweighting or rescaling of the existing three
delivers one.

## 5. Cost, measured rather than estimated

The independence ratio at 1e-4 is **3519 / 306 = 11.5x**, up from 6.60x at 1e-3. The cost
of independence grows as the tolerance tightens, because the cold path is the one that
keeps descending.

**The ratio-based decade estimate was measured and found badly wrong.** Before C1 the
arithmetic gave `ln(10)/(−ln 0.8216) ≈ 11.7` extra cycles for the cold cell. The measured
cost was **36 extra cycles** — a **3.1x underestimate**. This is a direct, empirical
falsification of the decrement-ratio extrapolation that has been flagged three times in
this project, and it is now retired as a method rather than merely cautioned against.

For the C2 decision: the realised ratio has drifted **0.8216 → 0.8451**, i.e. the descent
is slowing, so the next decade costs **at least** the 36 cycles the last one did and
probably more. At 51 solves and ~35 s per cycle that is **>= 1,836 solves and >= 35
minutes** for the cold cell alone, plausibly double. No single figure is offered, and the
ratio method is not used to produce one.

## 6. Stop

C2 stops here for the author's decision on a second decade. Nothing further is run.
