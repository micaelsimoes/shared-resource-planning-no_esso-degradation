# P5.9 — rescaled ADMM stabilization

Branch `feature/derivative-free-planning`. Canonical runtime
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`, checksum
`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`, IPOPT
`3.14.18` / ASL `20241111` at `/usr/local/bin/ipopt`, HSL `ma97` — all four
asserted by the R0 provenance gate in every harness.

> **Runtime path correction.** The stage brief named
> `/opt/anaconda3/envs/opf_env_py311/bin/python` as canonical. That is the
> MacBook Air path; it does not exist on this machine and was corrected in the
> governing documents in commit `c06f2cd4`. The checksum in the brief is
> unchanged and is what the gate enforces. Every result below was produced under
> the corrected path, which the migration verified by reproducing all twelve
> steps of the P5.6-D base chain within `5e-07`.

**No production change was made.** `data/SRP1/SRP1_params.json` was never
written. The rescaled objective is the process-local wrapper P5.8 introduced
(`p58_rescale`), removed again before every polish; rho overrides are applied to
a per-evaluation deep copy and to a private copy of the T0 template. IPOPT
options, ADMM tolerances, stopping criteria, the anchor policy, ESS equations,
Benders and the convex models are untouched, and nothing was merged. No
investment optimization, derivative-free search, GPS/MADS or convex planning
model was run.

New files: `p59_rho.py`, `p59_eval.py`, `p59_a_sweep.py`, `p59_a_refine.py`,
`p59_b_adaptive.py`, `p59_c_criteria.py`, `p59_d_replay.py`, `p59_e_anchor.py`.
Evidence: `data/SRP1/Results/P59/`.

---

## 0 — reproduction gate, and a correction to the stage's premise

### The gate

RESCALED at the template's own penalties, one generation from frozen T0,
reproduces P5.8-C generation 1 **exactly**:

| quantity | P5.8-C gen 1 | P5.9-A0 | delta |
|---|---|---|---|
| pre-polish net operational recourse | 825 814 074.49 | 825 814 074.4930633 | **+0.00** |
| polish outcome | FAILURE, 2 blocks | FAILURE, 2 blocks | — |
| failed blocks | `DSO5\|2025\|Spring`, `DSO5\|2030\|Summer` | identical | — |

Independently, the stage D CURRENT arm reproduces the accepted P5.6-D base chain
to the digit at every generation (`828021090.3608505`, `827415318.5639443`,
`826824028.8454782`, …). Everything below rests on those two reproductions.

### The premise correction: rho 1.0 is not what the oracle runs

The brief states that the existing `rho_v = rho_pf = rho_ess = 1` were
calibrated for the previous scaling. That is true of
`data/SRP1/SRP1_params.json`. **It is not what the oracle uses.** The frozen T0
template carries

```
rho_v = 1.5      rho_pf = 2.25      rho_ess = 1.0
```

— the values production's own adaptive rule left at the end of T0's cold build
(`1.5^1` and `1.5^2`). Production's warm-start path clones
`initial_state['models']` (`shared_resources_planning.py:2173-2179`) and never
rebuilds the augmented objectives, so **every warm-started evaluation from
P5.6-B onward has inherited T0's adapted penalties, and the parameter file's rho
is dead on that path.**

This is a second instance of the inherited-state channel P5.8-A0 documented for
`consecutive_converged_cycles` (`shared_resources_planning.py:2093-2094`). It was
found here because the first A0 attempt wrote the parameter-file values into the
template and missed P5.8-C's generation 1 by `-174 723.78`; the gate caught it
and the harness aborted rather than proceeding
(`p59_a_sweep_ABORTED_baseline_error.log`). The sweep was then re-anchored on the
inherited values, and the parameter-file configuration retained as case A0b.

The two baselines differ materially, which is why this matters:

| baseline | rho v / pf | pre-polish net recourse | `primal_pf` ratio | failed polish blocks |
|---|---|---|---|---|
| **A0** template, inherited | 1.5 / 2.25 | 825 814 074.49 | 0.3258 | 2 |
| **A0b** parameter file | 1.0 / 1.00 | 825 639 350.71 | 0.5495 | 4 |

Lower penalties buy a better local objective and worse agreement — the trade-off
this stage exists to resolve.

---

## A — rescaled ADMM and penalty calibration

Every case: one generation from the same frozen T0 primal state, RESCALED
objective, production's IPOPT options / ADMM tolerances / stopping criteria /
midpoint anchor untouched, `adaptive_penalty` left at its production value so
that what is swept is the initial penalty of the real oracle. Sweep values are
absolute as specified; the ratio to the inherited baseline is what each is
actually a multiple of.

The selection rule was declared in `p59_a_sweep.py` before any result existed:
prefer zero failed polish blocks; among those the lowest polished total
objective; failing that, fewest failed blocks tie-broken by lowest pre-polish
recourse.

| case | rho_v | rho_pf | cycles | pre-polish net recourse | polished total | ADMM→polish | failed blocks |
|---|---|---|---|---|---|---|---|
| A0 inherited | 1.5 | 2.25 | 4 | **825 814 074.49** | — | — | 2 |
| A0b param file | 1.0 | 1.00 | 4 | 825 639 350.71 | — | — | 4 |
| A1 | 10 | 2.25 | 4 | 825 816 101.85 | — | — | 2 |
| A1 | 100 | 2.25 | 4 | 825 817 216.36 | — | — | 2 |
| A1 | 1000 | 2.25 | 4 | 825 817 402.09 | — | — | 2 |
| A2 | 1.5 | 10 | 3 | 827 324 640.72 | — | — | 4 |
| A2 | 1.5 | 100 | 3 | 827 511 520.87 | — | — | 3 |
| **A2** | **1.5** | **1000** | 6 | 827 845 392.20 | **828 298 310.22** | +402 918.02 | **0** |
| A3 | 10 | 10 | 3 | 827 325 867.28 | — | — | 4 |
| A3 | 100 | 100 | 3 | 827 512 971.79 | — | — | 4 |
| A3 | 1000 | 1000 | 6 | 827 848 116.07 | 828 301 051.33 | +402 935.26 | 0 |
| A3b | 1.5 | 300 | 4 | 827 735 214.59 | 829 856 279.85 | +2 071 065.26 | 0 |
| A3b | 1.5 | 500 | 5 | 827 755 610.40 | 829 477 961.00 | +1 672 350.61 | 0 |

CURRENT reference at the same generation: pre-polish `837 188 510.90`, polished
`828 021 090.36`, ADMM→polish `−9 217 420.54`.

### A1 — rho_v is inert

Raising `rho_v` from the inherited 1.5 to 1000, a factor of **667**, moves the
pre-polish objective by **+3 327.60** on `8.26e8`, leaves `primal_pf_ratio`
unchanged at `0.3257`, and does not change the polish outcome — the same two
`DSO5` blocks fail in all four cases. The reason is measured, not inferred:
voltage consensus is never anywhere near binding. Across all 59 recorded cycles
the `consensus_v` slack ranges from **53×** to **4.26e5×** inside tolerance
(§C). There is nothing for `rho_v` to buy.

### A2 — rho_pf is the only lever, and agreement is monotone in it

`primal_pf_ratio` falls monotonically with `rho_pf`:

```
rho_pf   2.25      10       100      1000
ratio   0.3258   0.1858   0.0936   0.0384
```

**But polish success is not monotone.** Two blocks fail at 2.25, four at 10,
three at 100, and zero at 1000 — and the identity of the failures changes, from
`DSO5` blocks at low `rho_pf` to 2035 blocks including the TSO at intermediate
values. Better interface agreement is therefore necessary but not sufficient for
the exact-consensus polish to succeed; the failure is not a simple function of
the pf residual.

### A2/A3b — the cost of a working polish is the entire rescaling gain

This is the stage's central measurement. P5.8-C's headline was that RESCALED at
generation 1 returns `825 814 074.49`, beating the current pipeline's **fully
polished** generation-1 net recourse of `827 971 090.36` by `2 157 015.87`.
Reaching the only penalty at which the polish succeeds costs

```
827 845 392.20 − 825 814 074.49  =  2 031 317.71
```

i.e. **94.2 % of that gain is handed straight back**. What survives the polish is
worse than the current pipeline outright:

| | polished total objective | vs CURRENT gen 1 |
|---|---|---|
| CURRENT | 828 021 090.36 | — |
| RESCALED, rho_pf = 1000 | 828 298 310.22 | **+277 219.86 worse** |
| RESCALED, rho_pf = 500 | 829 477 961.00 | +1 456 870.64 worse |
| RESCALED, rho_pf = 300 | 829 856 279.85 | +1 835 189.49 worse |

The informed refinement at `rho_pf ∈ {300, 500}` was run precisely to test
whether an intermediate penalty keeps more of the gain while still polishing.
It does not: both polish cleanly and both land **worse** than `rho_pf = 1000`,
because the polish itself degrades the objective further the lower the penalty
(`+2 071 065` at 300, `+1 672 351` at 500, `+402 918` at 1000). Within the
tested range there is no penalty that delivers a high-quality economic solution
*and* a successful polish.

### The ADMM→polish gap changes sign

Under CURRENT the polish is a large **improvement**: `−9 217 420.54`. Under
RESCALED at the selected penalty it is a **degradation** of `+402 918.02`. The
magnitude does collapse — 23× smaller — but the polish has stopped being a
correction that recovers base-objective slack and become a constraint that costs
objective, because the rescaled ADMM solution is already better than the point
the exact-consensus polish can reach. This is the same phenomenon P5.8-C saw at
its generation 8 (`+35 647 836`), now visible at every penalty tested.

### A4 — rho_ess was not run, and the condition for running it is not met

A4 was conditional on ESS consensus binding. It does not bind under any
configuration tested: `primal_ess_ratio` lies between **0.0012 and 0.0057**
across all thirteen cases, i.e. 175× to 833× inside tolerance, and the
`consensus_ess` slack never falls below **175** in any of the 59 recorded cycles.
This independently reconfirms P5.8-A0 under conditions P5.8 never varied — there,
tightening the ESS tolerances left the trajectory bit-identical; here, sweeping
the penalties leaves the ESS residual three orders of magnitude inside its
threshold throughout.

```
A CONCLUSION — rho_v is inert, rho_ess does not bind, and rho_pf trades the
rescaling gain against polish success at close to 1:1.  The only configuration
that polishes cleanly, rho_pf = 1000, returns 94.2 % of the gain and still lands
277 220 WORSE than the current pipeline.
```

Selected for stages B, D and E by the declared rule: **`A2_rho_pf_1000`**
(`rho_v = 1.5`, `rho_pf = 1000`, `rho_ess = 1.0`).

---

## B — adaptive penalty audit

Nothing was tuned. Production's `_update_admm_penalties`
(`shared_resources_planning.py:5440-5520`) is unmodified; the only variable is
whether it is enabled. Both arms run four generations from the same frozen T0 at
`rho_v = 1.5`, `rho_pf = 1000`.

### The rule decays every raised penalty, and the mechanism is structural

Requested `rho_pf` against the value actually in force when the run ends:

| requested | terminal | equals |
|---|---|---|
| 10 | 10.00 | unchanged |
| 100 | 44.44 | `100 / 1.5²` |
| 300 | 88.89 | `300 / 1.5³` |
| 500 | 98.77 | `500 / 1.5⁴` |
| 1000 | 131.69 | `1000 / 1.5⁵` |

The larger the requested penalty, the more decrease steps the rule applies, so
terminal values are compressed into a narrow band around 90–130. The cause is in
the residual definition, not in the thresholds. Production measures

```
dual = rho * |z_current - z_prev| / base        (lines 4897, 4938)
```

which is **linear in rho**, and then compares `primal_ratio` against
`dual_ratio`, decreasing rho when `dual_ratio > 3 * primal_ratio` for the pf
family. Raising rho therefore raises the measured dual residual proportionally
and trips its own decrease branch. **The rule cannot hold a high interface
penalty after rescaling — it is a negative feedback against exactly the
correction stage A found necessary.**

### What the adaptation costs

| arm | generation 1 → 4 | step deltas | cycles at gen 1 | polish |
|---|---|---|---|---|
| adaptive **on** (production) | 828 298 310.22 → 827 844 231.91 | **+706 558, −1 052 043, −108 594** | 6 | clean |
| adaptive **off**, rho pinned at 1000 | 828 011 656.16 → 827 953 362.69 | **−21 309, −20 186, −16 798** | 16 | clean |

The pinned arm is monotone, decaying, and its generation-1 objective is
**9 434.20 better than the current production pipeline** (`828 021 090.36`). The
adaptive arm is erratic: it moves the objective *up* by 706 558 at generation 2
before recovering. Adaptation ends 109 130.78 lower in absolute terms, but by a
path that is not a refinement — it is the rho decay working through the chain.
Pinning costs 16 ADMM cycles at generation 1 instead of 6, and nothing
thereafter.

**Answer to B: no, adaptive rho balancing does not behave correctly after
rescaling.** It is not merely mistuned; its dual-residual definition makes rho
self-cancelling.

### Proposed rules — proposed only, NOT implemented

1. **Normalize the dual residual by rho before comparing it to the primal
   ratio**, or equivalently compare `|z_current - z_prev| / base` against a
   rho-independent threshold. The current form conflates "the agents are still
   moving" with "the penalty is large", and only the first is a convergence
   statement.
2. **Exempt a deliberately calibrated penalty from decrease**, e.g. a per-family
   floor, so that a value established by a calibration stage is not undone by
   the rule inside the first run.
3. **Re-derive the balance thresholds** (`5.0`, and `3.0` for pf decrease) after
   (1); they were chosen against a residual definition this stage shows to be
   rho-coupled.

No values are selected here, and none of the three was implemented or tested.

---

## C — convergence criterion audit

Nothing was changed. 96 recorded cycles across A, B, D, D2, D3 and E2; 24 of
them are the cycle that actually ended an ADMM run. "Binding" is the criterion
with the least relative slack, slack being `threshold / observed`, so 1.0 is
exactly at the boundary. The question is only meaningful on a terminating cycle,
so the headline counts are restricted to those 24.

### 1. Which criterion terminates the ADMM

| binding criterion | terminating cycles |
|---|---|
| **`stationarity_pf`** | **17** |
| `objective` | 7 |

Slack on the 24 terminating cycles:

| criterion | min | median | max |
|---|---|---|---|
| `stationarity_pf` | **1.00** | **1.34** | 5.29 |
| `objective` | 1.01 | 4.50 | 43.5 |
| `consensus_pf` | 1.82 | 22.4 | 643 |
| `consensus_pf_mean` | 4.38 | 100 | 1.21e+04 |
| `stationarity_v` | 31.8 | 123 | 129 |
| `consensus_v` | 169 | 554 | 8.79e+04 |
| `consensus_v_mean` | 167 | 547 | 1.73e+05 |
| `consensus_ess` | 175 | 356 | 1.74e+04 |
| `stationarity_ess` | 840 | 895 | 912 |
| `consensus_ess_mean` | 2.60e+03 | 6.20e+03 | 3.53e+04 |

**This is a change from the current formulation.** P5.8-D found the objective
test binding under CURRENT. After rescaling, `stationarity_pf` binds on 71 % of
terminating cycles and sits at a median slack of 1.34 — effectively on its
threshold — while the objective test has median slack 4.5. The ADMM now stops
because the interface consensus variables have stopped moving, not because the
recourse has stopped changing.

### 2. Which criteria are meaningful after rescaling

- **`stationarity_pf` and the objective test are the only live criteria.** They
  are the only two that ever bind.
- **`consensus_v`, `consensus_v_mean`, `stationarity_v` are inert** — never
  closer than 32× to their thresholds. This is the same fact that makes `rho_v`
  inert in §A.
- **Every ESS criterion is inert** — never closer than 175×, and
  `consensus_ess_mean` runs 2 600× to 35 300× inside tolerance. Third
  independent confirmation of P5.8-A0.
- `consensus_pf` is occasionally close (min slack 1.82) but never the binding
  one on a terminating cycle.

### 3. What the objective tolerance should be based on

The tolerance is `max(1e3, 1e-3 × recourse)`, measured at **826 461 … 838 497**
across this stage. Against P5.6-D's best-to-second candidate gap of `33 031`
that is **25×** too coarse, and **59 of 96 cycles were declared
objective-converged while the recourse was still moving by more than that gap** —
unchanged from P5.8-D, as expected, since nothing was modified.

There is now a further reason to distrust the form, which §D2/§D3 supply: the
`33 031` signal scale was itself derived from the CURRENT oracle, and the
rescaled oracle puts the genuine operational sensitivity to a ±10 % capacity
perturbation at roughly `10²`, not `10⁴`. **A tolerance scaled to the recourse
level (`8.3e8`) is four to six orders of magnitude away from the quantity the
criterion feeds, whichever signal scale is correct.**

Recommendations, offered as options with no values selected and nothing
implemented:

1. **An absolute floor tied to the decision resolution**, `max(floor, rel × recourse)`,
   so the test is sensitive to the only quantity it feeds. The floor must be
   re-derived from the rescaled oracle's signal scale, not from `tau_planning`.
2. **Per-subproblem local KKT residual in base-objective units.** Already in the
   IPOPT logs and already parsed by `p58_rescale.read_log_since`; P5.8-B
   measured it 390–1 488× tighter under RESCALED, so after rescaling it is a
   meaningful optimality measure rather than a scaling artefact.
3. **Base-objective improvement between cycles**, tracked separately from the
   augmented objective, which mixes economics with consensus penalties.
4. **Reset or make explicit the inherited `consecutive_converged_cycles`** — and,
   per §0, the inherited **rho**. Both cross warm starts through
   `initial_state`, and both make a configured value mean something different on
   the second run of a chain than on the first.

---

## D — full rescaled oracle replay

Three candidates, both formulations, eight warm-start generations each from the
same frozen T0. One process per candidate, inside P5.6-B7's four-worker ceiling;
every solve is deterministic given its inputs and each process used its own copy
of T0.

**Control:** the CURRENT arm at the base candidate reproduces the accepted
P5.6-D chain to the digit at all eight generations.

### Coverage is fixed

| candidate, generation 1 direct from T0 | CURRENT | RESCALED |
|---|---|---|
| `base` | VALID | VALID |
| `se\|node5\|2025\|-10%` | **POLISH_FAILURE**, 2 blocks | **VALID** |
| `se\|node9\|2025\|-10%` | **POLISH_FAILURE**, 2 blocks | **VALID** |

`se|node9|2025|-10%` is the P5.7-D target whose direct solve failed and which
only capacity continuation could reach. Under RESCALED it evaluates directly.
**Continuation is no longer needed for reachability on these candidates** —
success criterion 2, met.

### The chain

Base candidate, polished total objective:

| | CURRENT | RESCALED |
|---|---|---|
| generation 1 | 828 021 090.36 | 828 298 310.22 |
| generation 8 | 824 795 363.72 | 827 614 426.31 |
| drift over 8 generations | **−3 225 726.64** | **−683 883.91** |
| final step delta | −313 345.98 | −38 021.09 |

Absolute drift is reduced 4.7× and the final step 8.2×, but it has not stopped.

### The result that matters: the landscape is far more stable

The planning problem does not consume absolute objective values; it consumes
differences between candidates at a common depth. Those are transformed:

| | CURRENT | RESCALED |
|---|---|---|
| cross-depth spread of `Δ(se\|node5)` | 420 636.12 | **540.92** |
| cross-depth spread of `Δ(se\|node9)` | 459 695.59 | **50.74** |
| landscape uncertainty (max spread) | **459 695.59** | **540.92** |
| mean best-to-second gap | 18 085.03 | 89.54 |
| **uncertainty / signal** | **25.4×** | **6.0×** |

The uncertainty-to-signal ratio improves **4.2×**, and the raw cross-depth
uncertainty improves **850×**. For context, P5.6-D reported
`tau_planning_refined = 811 438.05` against a benchmark of `4.25e5`, and
best-to-second gaps `25` to `97` times the signal.

The `540.92` spread for `se|node5` is dominated by a single excursion: its
per-generation deltas are `−586, −601, −602, −137, −678, −620, −619, −626`, so
generation 4 alone accounts for it. Excluding that one generation the spread is
`92.34` against a mean signal of `89.54`, i.e. a ratio near **1.0**. That
exclusion is reported for what it shows about the residual defect, **not** used
as the headline: a one-generation excursion of 480 in an otherwise ±50 band is
itself an instability, and the honest figure is 6.0×.

Ranking is not yet resolved between these two candidates: `se|node9` is best in
6 of 8 generations, with two flips at generations 5 and 6. But the two differ by
only 3.81 to 500.99 (mean 89.54) — they are genuinely near-tied, which is a
different situation from CURRENT, where they differ by up to 35 651 and the
ordering still flips twice.

---

## D2 / D3 — the investment signal itself

These were not in the stage specification. They were added because D produced a
number that could not be interpreted without them, and the interpretation
changes the stage's conclusion.

### D2 — the collapse is caused by rescaling, not by rho

At generation 1, separation from the base candidate:

| rho_pf | polish clean | `se\|node5` | `se\|node9` | × τ_planning |
|---|---|---|---|---|
| 2.25 (inherited, nothing raised) | no | +415.73 | +315.81 | 0.013 |
| 100 | no | +561.62 | +631.06 | 0.019 |
| 300 | yes | +404.73 | +303.90 | 0.012 |
| 1000 | yes | +128.20 | +51.92 | 0.004 |
| **CURRENT, same generation** | — | **−293 998.92** | **−162 172.04** | **8.9** |

The separation is a few hundred **at every penalty, including the one nothing was
done to**. Raising `rho_pf` makes it modestly smaller; it does not cause it.

### D3 — the rescaled oracle is not blind

Two readings survived D2: either the small separation is physical and CURRENT's
is numerical residue, or the rescaled coordination suppresses it. A ±10 %
perturbation cannot distinguish them; a large one can.

| candidate | investment | CURRENT separation | RESCALED separation | ratio |
|---|---|---|---|---|
| `se\|node5\|2025\|-10%` | 45 000 (node) | −293 999 | +416 | 0.0014 |
| `se\|node9\|2025\|-10%` | — | −162 172 | +316 | 0.0019 |
| `e\|node9\|2030\|+50%` | 51 604 | −32 228 | −384 | 0.012 |
| **`se\|ALL\|x19` (budget boundary)** | **950 000** | **−367 790** | **−130 201** | **0.354** |

**The rescaled oracle responds to a 19× capacity change with a separation of
130 201 — 3.94× τ_planning.** It resolves capacity perfectly well when there is
capacity to resolve.

The CURRENT column is the diagnostic one. Across perturbations spanning a factor
of roughly 190 in capacity, its separations vary only from 32 228 to 367 790 — a
factor of 11. They are nearly independent of the size of the physical change.
The RESCALED column varies by a factor of ~340 over the same candidates and
tracks the perturbation. **A response that barely depends on its input is a
noise floor, not a signal**, and this is the clearest available evidence that the
~3e5 separations the CURRENT oracle reports for ±10 % candidates are dominated by
the 8.9e6 of base-objective slack P5.7-A4 measured, not by economics.

If that reading holds, it re-frames P5.6-C and P5.6-D rather than contradicting
them: a Spearman of `−0.033` and 6-of-9 improvement-sign reversals is what
ranking on numerical residue looks like, and `tau_planning = 33 031` — derived
from those values — is not the signal scale of a correctly scaled oracle. The
genuine operational sensitivity to a ±10 % shared-ESS perturbation appears to be
of order `10²`, consistent with P5.7 §2's finding that the shared ESS is nearly
inert at the base bootstrap capacity (schedules moved `1.72e-05 p.u.` across
twelve refinements).

**This is offered as the reading the evidence supports, not as a settled
result.** It rests on one large-perturbation pair at one generation. Confirming
it needs a perturbation sweep across several capacity magnitudes, run to a fixed
depth under both formulations — which is a stage, not an addendum.

---

## E — interface anchor revisit

Run after A–D, as instructed. **No anchor policy was changed.**

**E1 — the rescaling does not touch the anchor.** Against the T0 template's own
fixed ADN interface load:

```
max | pc(T0) - pc(rescaled + rho template) |  =  0.000000e+00
max | qc(T0) - qc(rescaled + rho template) |  =  0.000000e+00
```

Neither the objective multiplier nor the rho override perturbs the fixed anchor,
which is required: they act on the objective only.

**E2 — flexibility behaviour is qualitatively unchanged and quantitatively
smaller.** Weighted cost families over four generations at the base candidate,
using production's own per-block accessor:

| | flexibility_cost | generation_cost | flex per unit of generation saved |
|---|---|---|---|
| CURRENT | 196 210 984.28 → 197 076 874.81 (**+865 890.52**) | 631 726 248.26 → 629 252 007.94 (**−2 474 240.32**) | 0.35 |
| RESCALED | 197 509 627.88 → 197 808 280.84 (**+298 652.96**) | 630 589 620.76 → 629 898 243.54 (**−691 377.22**) | 0.43 |

The CURRENT generation-1 values reproduce P5.7 §1's polish `K=0` row exactly
(`196 210 984.28` / `631 726 248.26`).

The mechanism P5.7 §5 identified persists: refinement finds a better dispatch
and pays more flexibility to hold the interface away from a reference frozen in
T0. Under RESCALED it operates at about one third the magnitude over the same
number of generations, and the flexibility paid per unit of generation cost
saved is slightly higher. **P5.8-E's conclusion stands and the anchor remains a
modelling question about what the flexibility reference should be, not a source
of instability.** Nothing here promotes it to a priority.

---

## F — success criteria

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | rescaled ADMM produces stable solutions | **partly** | every generation VALID for every candidate; relative deltas stable to ±50 for `se\|node9`; but a 480-unit single-generation excursion at `se\|node5` gen 4, and absolute drift of −683 884 over 8 generations |
| 2 | exact-consensus polish succeeds without continuation | **met** | 0 failed blocks at `rho_pf ≥ 300`; both previously `POLISH_FAILURE` candidates evaluate directly, including P5.7-D's continuation-only target |
| 3 | ADMM-to-polish gap small relative to the previous 9M | **met in magnitude** | `+402 918` against `−9 217 421`, 23× smaller — but the sign inverts: the polish now degrades the objective instead of recovering it |
| 4 | H2/H4/H8 refinement drift disappears or becomes small | **substantially improved, not eliminated** | cross-depth landscape uncertainty `459 696 → 541` (850×); uncertainty/signal `25.4× → 6.0×`; absolute drift still −683 884 |
| 5 | no new instability introduced | **not met** | the adaptive rule decays any raised penalty (`1000 → 131.7`) and is self-cancelling by construction; the polish degrades the objective, by `+2 071 065` at `rho_pf = 300` |

Three of five are met or substantially met. The two that are not are both
properties of machinery calibrated to the pre-rescaling operating point — the
adaptive penalty rule and the exact-consensus polish — rather than of the
rescaling itself.

---

## What was not done

- No production equation, solver setting, ADMM tolerance, stopping criterion,
  penalty rule, anchor policy or master/Benders code was modified, and nothing
  was merged. `data/SRP1/SRP1_params.json` was never written.
- No investment optimization, derivative-free search, GPS/MADS, Benders run or
  convex planning model was executed.
- **A4 (`rho_ess`) was not run.** Its precondition is not met: `primal_ess_ratio`
  stays between `0.0012` and `0.0057` under every configuration tested, and no
  ESS criterion came closer than 175× to its threshold on any of 96 cycles.
- The proposed adaptive-penalty rules and convergence measures in §B and §C were
  **not implemented and not tested**. No values were selected.
- The rescaled chains were run from a T0 built under the CURRENT formulation, by
  design, so the comparison isolates the change. A self-consistent rescaled T0
  build was not performed — carried over from P5.8.
- `rho_v` and `rho_pf` were swept; the proximal regularization gammas, the
  `min`/`max` penalty clamps and the consensus/stationarity tolerances were not.
- D3 rests on one large-perturbation pair at one generation. The signal-scale
  reading it supports is stated as a reading, not a result.
- The adaptive-penalty-off configuration was run for 4 generations at one
  candidate (§B). It was **not** run across the candidate set, so its effect on
  the investment landscape is unmeasured — this is the most obvious gap and the
  most promising next experiment.

---

## Verdict

Rescaling is confirmed as necessary and it delivers more than P5.8 could
measure. It fixes oracle coverage outright: candidates that were
`POLISH_FAILURE` direct from T0, including the one P5.7-D could only reach by
continuation, now evaluate directly, so continuation is no longer needed for
reachability. It collapses the ADMM-to-polish gap 23×. And it improves the
quantity P5.6-C and P5.6-D actually failed on — cross-depth landscape
uncertainty — from `459 696` to `541`, an 850× reduction that takes the
uncertainty-to-signal ratio from `25.4×` to `6.0×`.

It also produced the stage's most consequential measurement, which was not
asked for. The rescaled oracle separates ±10 % capacity candidates by a few
hundred currency units at every penalty tested, while the current formulation
separates them by two to three hundred thousand — and the current formulation's
separations barely change when the capacity perturbation is made 190× larger,
whereas the rescaled oracle's grow by a factor of ~340 and reach `130 201` on a
19× capacity change. The evidence points to the investment signal the previous
stages were ranking on having been largely numerical residue from subproblems
stopping `8.9e6` short of base-optimality. If that holds, `tau_planning = 33 031`
is not the resolution the planning problem needs, and the target has to be
re-derived.

But the oracle is not yet stable, and the reasons are specific and located. The
adaptive penalty rule is self-cancelling after rescaling: because the dual
residual is defined as `rho * |Δz| / base`, raising rho raises the measured dual
residual and trips the rule's own decrease branch, decaying every requested
penalty toward 90–130 — `1000 → 131.69` is exactly `1000 / 1.5⁵`. With the rule
disabled and rho pinned, the chain becomes monotone with steps of `−21 309 →
−16 798` and beats the current pipeline by `9 434` at generation 1; with it
enabled, the same configuration moves the objective *up* by `706 558` at
generation 2. The exact-consensus polish is calibrated to the same old operating
point and now degrades the objective rather than recovering it, by `+402 918` at
the selected penalty and `+2 071 065` at `rho_pf = 300`. And no penalty in the
tested range delivers both a high-quality economic solution and a working polish:
the only clean configuration hands back 94.2 % of the rescaling gain and lands
`277 220` worse than production.

Objective scaling was the arithmetic defect and it is understood. What remains is
not another calibration — the calibration was performed here and it does not
close — but two pieces of coordination machinery, the adaptive penalty rule and
the exact-consensus polish, that were fitted to an under-solved ADMM and do not
transfer to a correctly scaled one.

```
P5.9-B — rescaling helps but ADMM coordination requires further redesign
```

```
P5.9 COMPLETE — ready for planner review
```
