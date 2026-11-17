# P5.8 — ADMM numerical scaling validation

Branch `feature/derivative-free-planning`. Canonical runtime
`/opt/anaconda3/envs/opf_env_py311/bin/python`, checksum
`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`, enforced by
the R0 provenance gate in every harness.

**No production code was modified.** `data/SRP1/SRP1_params.json` was never
written; the A0 tolerance overrides are applied to a per-evaluation deep copy, and
the rescaled objective is a process-local wrapper that
`p58_rescale.patched_admm_objectives` / `rescale_state_models` install on a clone
and that `restore_for_polish` removes again before the polish. Nothing was
merged. No derivative-free search, GPS/MADS, investment optimization, Benders run
or convex planning model was executed.

New files: `p58_rescale.py`, `p58_eval.py`, `p58_a0_tolerances.py`,
`p58_b_rescale.py`, `p58_c_replay.py`, `p58_e_anchor.py`. Evidence:
`data/SRP1/Results/P58/`.

**Reproduction gate.** A0 case A — production parameters, unchanged — reproduced
the accepted P5.6-D base chain **exactly** at all four generations
(`828021090.3608505`, `827415318.5639443`, `826824028.8454782`,
`826405022.1934375`). B2 reproduced P5.7-A2's aggregate to the digit
(`−383401.68` / `−8892463.76`). Everything below rests on that.

---

## A0 — ESS and consensus tolerance screen

Three cases at the canonical base candidate, from the **same** frozen T0, so only
the tolerances differ. Four generations each, because a single evaluation cannot
show refinement drift.

| case | `ess_mean` | `ess` | min consecutive cycles | chain (planning objective) |
|---|---|---|---|---|
| **A** current | 1e-2 | 1e-1 | 1 | 828021090.3608505 → 827415318.5639443 → 826824028.8454782 → 826405022.1934375 |
| **B** harmonized ESS | 1e-3 | 1e-2 | 1 | **bit-identical to A at every generation** |
| **C** harmonized + 2 cycles | 1e-3 | 1e-2 | 2 | 827415318.5639443 → 826824028.8454782 → 826405022.1934375 → 825961521.8823205 |

| case | drift gen 1 → gen 4 | mean step | ADMM cycles, gen 2-4 |
|---|---|---|---|
| A | −1 616 068.17 | −538 689.39 | 1, 1, 1 |
| B | **−1 616 068.17** | **−538 689.39** | 1, 1, 1 |
| C | −1 453 796.68 | −484 598.89 | 1, 1, 1 |

**Case B is bit-identical to case A.** The reason is measurable and complete: the
ESS consensus test was never binding in either case.

| generation | `primal_ess` | `primal_ess_mean` | case A tolerance | case B tolerance |
|---|---|---|---|---|
| 1 | 2.235e-04 | 6.459e-06 | 1e-1 / 1e-2 | 1e-2 / 1e-3 |
| 2 | 2.227e-04 | 6.251e-06 | — | — |
| 3 | 2.048e-04 | 6.682e-06 | — | — |
| 4 | 1.931e-04 | 5.466e-06 | — | — |

The realized ESS residual is **450×** below case A's max tolerance and **1 500×**
below its mean tolerance; even case B's tightened values sit 45× and 150× above
it. Tightening a test that never fires changes nothing, and it did not.

**Case C** does not change the trajectory either — it advances along it. Its
generation 1 returns exactly case A's generation 2, in 3 ADMM cycles instead of
2, and its generations 2-4 are case A's 3-5. Requiring two consecutive converged
cycles is worth exactly one refinement step, once.

> **Incidental finding, relevant to D.** From generation 2 onward case C also
> converges in a single cycle, despite requiring two. `_run_operational_planning`
> restores `consecutive_converged_cycles` from `initial_state`
> (`shared_resources_planning.py:2093-2094`), so on a warm start the counter is
> inherited and `minimum_consecutive_converged_cycles` binds only on the **first**
> run of a chain.

```
DECISION — ESS consensus tolerance is NOT a contributor.  It is not binding, and
harmonizing it leaves the refinement drift bit-identical.  Proceed with objective
scaling as the primary mechanism.
```

---

## B — objective rescaling validation

```
CURRENT    base_objective / effective_scale  +  ADMM terms
RESCALED   base_objective  +  effective_scale * ADMM terms
```

Held identical: the initial point (the same converged ADMM state), the consensus
variables, rho, the proximal terms, the IPOPT settings, and the model objects up
to a Pyomo clone.

**The two formulations are verified to be the same problem.** On all 48 template
subproblems, `rescaled_expr` against `effective_scale × admm_expr`:

```
48 blocks checked, worst relative error 0.000e+00
effective_scale in [9.414572e+04, 1.160242e+05]
```

A positive constant multiple leaves the feasible set, the argmin and the KKT
point identical, so any difference in what is returned is a solver-path effect by
construction.

### B1 — one-block experiments

Each block re-solved from its own converged ADMM point.

| block | variant | IPOPT iterations | base-objective delta | unscaled dual infeasibility |
|---|---|---|---|---|
| `DSO5\|2035\|Autumn` | CURRENT | 26 | **0.00** | 1.221507e-06 |
| `DSO5\|2035\|Autumn` | RESCALED | 36 | **−257 894.90** | 9.525718e-05 |
| `TSO\|2025\|Spring` | CURRENT | 23 | −53 107.54 | 8.007963e-09 |
| `TSO\|2025\|Spring` | RESCALED | 24 | −100 756.90 | 5.833706e-05 |
| `DSO7\|2035\|Autumn` | CURRENT | 27 | **−0.00** | 2.616332e-06 |
| `DSO7\|2035\|Autumn` | RESCALED | 44 | **−320 612.44** | 7.781175e-04 |

The raw dual infeasibility looks *worse* under RESCALED, and that is the point:
it is measured on the objective as written, and the two objectives differ by a
factor of ~1e5. Put both in **base-objective units** — multiply the CURRENT
residual by that block's `effective_scale`, since CURRENT minimizes
`base / effective_scale`:

| block | CURRENT, base units | RESCALED, base units | RESCALED tighter by |
|---|---|---|---|
| `DSO5\|2035\|Autumn` | 1.417245e-01 | 9.525718e-05 | **1 488×** |
| `TSO\|2025\|Spring` | 7.539154e-04 | 5.833706e-05 | 13× |
| `DSO7\|2035\|Autumn` | 3.035579e-01 | 7.781175e-04 | **390×** |

IPOPT is not failing. It is converging correctly to the problem it was handed,
and in the units the planning problem is measured in that problem's stationarity
tolerance is two to three orders of magnitude looser than it appears.

### B2 — full 48-block base replay

| variant | blocks solved | total base-objective delta | IPOPT iterations | max unscaled dual infeasibility | max unscaled constraint violation | runtime |
|---|---|---|---|---|---|---|
| CURRENT | 48 / 48 | **−383 401.68** | 1 309 | 9.3133e-06 | **1.2241e-05** | 16.7 s |
| RESCALED | 48 / 48 | **−8 892 463.76** | 1 740 | 9.0192e-03 | **1.7603e-07** | 27.1 s |

Additional base objective recovered by rescaling alone: **−8 509 062.08**, for
33 % more IPOPT iterations. This reproduces P5.7-A2 exactly.

**A second, unlooked-for gain: primal feasibility improves by 70×.** The largest
unscaled constraint violation falls from 1.2241e-05 to 1.7603e-07. The
"1.1e-05 p.u. residual, i.e. IPOPT's own feasibility tolerance" that P5.6-A
through P5.7 recorded on every certificate is itself a scaling artefact: with the
objective at unit coefficient the overall NLP error is dominated by
stationarity, IPOPT iterates further, and it tightens the constraints on the way.

```
B ACCEPTANCE MET — the rescaled formulation reduces the ADMM-to-polished gap by
8 509 062 of 9 217 421 (92.3 %) on a single pass over the same 48 subproblems.
```

---

## C — full ADMM replay

Both chains start from the **same** frozen T0 primal state; for the rescaled
chain the template's 48 network subproblem objectives are multiplied by their own
`effective_scale` once, before the chain starts. Production's warm-start path
clones `initial_state['models']` and never rebuilds the augmented objectives, so
every generation then runs the RESCALED formulation from an identical initial
point. P5.7-B measured initialization sensitivity at 303 on 8.3e8, so what
remains is the formulation. The polish is production's own in both chains.

The CURRENT chain is not re-run here: it is the accepted P5.6-D base chain,
reproduced twice (P5.7 to 4.8e-07, P5.8-A0 case A exactly).

### The like-for-like comparison is pre-polish net operational recourse

| generation | CURRENT ADMM | RESCALED ADMM | RESCALED better by | CURRENT step | RESCALED step |
|---|---|---|---|---|---|
| 1 | 837 188 510.90 | **825 814 074.49** | **11 374 436.41** | — | — |
| 2 | 836 578 781.24 | 825 257 390.49 | 11 321 390.75 | −609 730 | −556 684 |
| 3 | 835 829 460.61 | 824 796 861.45 | 11 032 599.17 | −749 321 | −460 529 |
| 4 | 835 374 067.36 | 824 408 917.11 | 10 965 150.25 | −455 393 | −387 944 |
| 5 | — | 824 042 878.24 | — | −443 500 | −366 039 |
| 6 | — | 823 675 323.49 | — | −430 215 | −367 555 |
| 7 | — | 823 352 299.32 | — | −422 597 | −323 024 |
| 8 | — | 823 101 056.86 | — | −313 346 | −251 242 |

**The desired outcome is achieved, and then some.** One rescaled ADMM run, with
no polish at all, returns 825 814 074 — which is

- **11 374 436 better** than the current ADMM at the same generation;
- **2 157 016 better than the current pipeline's fully polished generation-1
  result** (827 971 090);
- a value the current polished chain does not reach until generation 5-6.

After eight generations the rescaled ADMM stands at 823 101 057 — **580 277
better than the current pipeline achieves after twelve polished refinements**
(823 681 334). The ADMM solution is now close to, and here better than, the
economically polished one, which is precisely what C set out to test.

### But the drift is not removed

Rescaled step deltas fall −556 684 → −251 242 over seven transitions, against the
current chain's −609 730 → −313 346 over the same span. **Rescaling shifts the
whole chain down by ~11e6; it reduces the per-step drift by only 10-20 %.** From
generation 2 onward the rescaled ADMM again converges in **one cycle**. Section D
says why, and it is not the scaling.

| | CURRENT | RESCALED |
|---|---|---|
| refinements to bring the step below `tau_planning` = 33 031 | not reached in 12 | not reached in 8 |
| refinements to bring the step below `tau_numerical` = 10 | not reached | not reached |

### The rescaled ADMM breaks the downstream exact-consensus polish

| generation | ADMM cycles | polish | failed blocks |
|---|---|---|---|
| 1 | 4 | FAILURE | `DSO5\|2025\|Spring`, `DSO5\|2030\|Summer` |
| 2 | 1 | FAILURE | `DSO5\|2025\|Spring`, `DSO5\|2030\|Spring`, `DSO5\|2030\|Summer` |
| 3 | 1 | FAILURE | 4 blocks |
| 4 | 1 | FAILURE | 4 blocks |
| 5 | 1 | FAILURE | 3 blocks |
| 6 | 1 | FAILURE | 2 blocks |
| 7 | 1 | FAILURE | `DSO9\|2035\|Summer` |
| 8 | 1 | VALID | — |

These are hard numerical failures, read from IPOPT's own logs for generation 1:

```
DSO5|2025|Spring   EXIT: Restoration Failed!                     1875 iterations
DSO5|2030|Summer   EXIT: Maximum Number of Iterations Exceeded.  3000 iterations
```

And where the polish does succeed, at generation 8, it **degrades** the objective
by **+35 647 836** — from an ADMM recourse of 823 101 057 to a polished
858 748 893 — while still passing every audit (coordinated residual 1.11e-16,
network violation 1.39e-05, ESSO feasible).

The mechanism is visible in the consensus residuals. The rescaled local solves
pursue their own base objective harder and therefore **agree less**: `primal_pf`
reaches 1.017e-02 at the tolerance boundary of 1e-2 in the first cycles, against
6.19e-03 for the current formulation. The exact-consensus polish then has to drag
each agent onto a midpoint that is further from where either of them wanted to
be — and for several DSO blocks that fixed interface point is at or beyond what
the block can deliver, which is what a restoration failure looks like.

**The polish, and by implication the rho and adaptive-penalty settings, are
implicitly calibrated to the current formulation's operating point.** Both are
locked for this stage, and re-tuning them is not a P5.8 authorization.

---

## D — convergence criteria audit

No stopping criterion was modified. What follows is what the current ones are and
what they do.

### The current stopping conditions

`_run_operational_planning` accepts a cycle when

```
cycle_convergence = residual_convergence AND objective_convergence
convergence       = consecutive_converged_cycles >= minimum_consecutive_converged_cycles
```

`residual_convergence` = `check_admm_convergence` = **consensus** (max and mean
primal residual, per family, against `tol['consensus']`) **AND stationarity**
(mean *dual* residual — the ADMM's `z`-change — against `tol['stationarity']`).
`objective_convergence` compares the change in **net operational recourse**
between cycles against

```
objective_tolerance = max(tol['objective']['abs'], tol['objective']['rel'] * recourse_scale)
                    = max(1e3, 1e-3 * 8.28e8)
                    = 827 971
```

So consensus residuals are **not** the only criterion; there is already an
objective test, and a local-stationarity proxy in the dual residual.

### The problem is not that a criterion is missing. It is that this one is 25× too coarse.

Measured, case A:

| generation | recourse change | objective tolerance | accepted as converged |
|---|---|---|---|
| 1 | 17 442.70 | 837 205.95 | yes |
| 2 | **609 729.66** | 837 188.51 | yes |
| 3 | **749 320.63** | 836 578.78 | yes |
| 4 | **455 393.25** | 835 829.46 | yes |

and, rescaled:

| generation | recourse change | objective tolerance | accepted |
|---|---|---|---|
| 1 | 787 755.97 | 826 601.83 | yes |
| 2 | 556 684.00 | 825 814.07 | yes |
| … | … | … | … |
| 8 | 251 242.46 | 823 352.30 | yes |

Every refinement step P5.6-D and P5.7 observed is **inside** this tolerance.
Against the quantity the planning problem must resolve — P5.6-D's best-to-second
candidate gap of **33 031** — the criterion is **25× too coarse**. The ADMM is
behaving exactly as specified; the specification permits it to stop while the
recourse is still moving by three quarters of a million.

Meanwhile the consensus residuals are comfortably inside their own tolerances
(`primal_pf` 5.1e-03 against 1e-02, `primal_v` 3.9e-05 against 1e-02,
`primal_ess` 1.9e-04 against 1e-01), so the residual test is not what stops the
iteration — the objective test is, and it stops it early.

### Is an additional optimality measure needed?

**Yes, and A0 already shows why the obvious substitutes will not do.** Requiring
two consecutive converged cycles (case C) advanced the chain by exactly one step
and then reverted to one cycle per generation, because
`consecutive_converged_cycles` is restored from `initial_state`
(`shared_resources_planning.py:2093-2094`) and the counter is inherited across
warm starts.

Candidate diagnostics, **not implemented** and offered for a later stage:

1. **A recourse-change criterion scaled to the planning signal, not to the
   recourse level.** `1e-3 × recourse` is 827 971; the decision it feeds needs
   33 031. An absolute floor tied to `tau_planning` would be the direct fix, and
   it is a parameter change, not a formulation change.
2. **Local KKT residual per subproblem, in base-objective units.** B1 shows
   IPOPT's own unscaled dual infeasibility is already available in the solver log
   and differs by 390-1 488× between the two formulations. It is the natural
   optimality measure and it costs nothing to read.
3. **Base-objective improvement between ADMM iterations**, separately from the
   augmented objective — the quantity the planning problem actually cares about.
4. **Reset or make explicit the inherited `consecutive_converged_cycles`** on a
   warm start, so a persistence requirement means what it says.

---

## E — interface anchor audit

Run after B/C, as instructed. **No anchor policy was changed.**

`_prepare_transmission_objectives_for_admm` fixes the TSO's ADN load `pc`/`qc` at
the DSO's then-current consensus interface power
(`shared_resources_planning.py:2905-2919`, via `fix_or_set`); the transmission
system then pays `flexibility_cost` for downward deviation from that anchor. On a
warm start `_run_operational_planning` clones `initial_state['models']`
(`2173-2179`) and never re-runs that preparation.

**E1 — anchor evolution.** Exactly constant, in both formulations, over every
generation tested (864 `pc` entries across the 12 TSO blocks):

```
CURRENT chain,  max |pc(gen j) - pc(gen 1)| :  gen 2, 4, 8, 12  ->  0.0
RESCALED ADMM,  max |pc(gen j) - pc(gen 1)| :  gen 2, 4, 8      ->  0.0
max |pc(CURRENT gen 1) - pc(RESCALED gen 8)|                    ->  0.0
max |qc(CURRENT gen 1) - qc(RESCALED gen 8)|                    ->  0.0
```

The rescaling does not touch the anchor, and refinement does not move it.

**E2 — candidate dependence.** A different candidate, `se|node5|2025|-10%`,
warm-started from T0:

```
max | pc(T0) - pc(other candidate, warm) |  =  0.000000e+00
```

The anchor is identical across candidates under the locked T0 policy, so it
cannot distort the investment landscape between them.

**E3 — cold versus warm. This refutes a P5.7 conjecture.** The same candidate,
started **cold**, so the anchor is set from that run's own initial DSO solves:

```
max | pc(T0) - pc(other candidate, cold) |  =  1.003037e-07 p.u.   (worst 2025|Summer)
```

1e-07 p.u. is 0.01 W — solver-tolerance noise.

P5.7 §5 flagged, explicitly as a code-path reading and explicitly as **not
measured**, that a cold start might set a candidate-dependent anchor and
therefore give cold and T0-warm evaluations different objective functions, and
offered that as a candidate explanation for the P5.6-B cold-versus-T0 gaps of
1.32e6 … 1.09e7. **That conjecture is now measured and it is wrong.** The DSOs'
initial local solves return essentially the same interface power whether the
model is built cold or inherited, so the anchor is common across candidates *and*
across start policies. The P5.6-B cold-versus-T0 gaps are genuine
branch/solution differences, as P5.6-B recorded them.

What survives from P5.7 §5 is the narrower, still-true observation: the anchor is
inherited from T0's initialization and never refreshed, so the growth in
`flexibility_cost` along a refinement chain (+3 040 929 over eleven generations,
P5.7 §1) is measured against a reference that does not move. That is a modelling
question about what the flexibility reference *should* be, not a source of
inconsistency between candidates or between start policies.

---

## What was not done

- No production equation, solver setting, ADMM tolerance, stopping criterion,
  anchor policy or master-cut code was modified, and nothing was merged.
- `data/SRP1/SRP1_params.json` was never written.
- No derivative-free search, GPS/MADS, investment optimization, Benders run or
  convex planning model was executed.
- The rho and adaptive-penalty settings were **not** re-tuned for the rescaled
  formulation, which §C identifies as the likely reason the exact-consensus
  polish fails there. That is the obvious next experiment and it is outside this
  stage's authorization.
- The rescaled chain was run from a T0 built under the CURRENT formulation, by
  design, so that the comparison isolates the objective scaling. A self-consistent
  rescaled T0 build was not performed.
- The ESSO subproblem was not rescaled: production does not divide it by
  `objective_scale`, so there is nothing there to undo.

---

## Verdict

Objective scaling is confirmed, and it is large. The two formulations are the
same problem to machine zero, and switching between them is worth
**8 509 062 of the 9 217 421** ADMM-to-polish gap on a single pass, an ADMM
solution **11 374 436** better at the same generation, a **70×** improvement in
primal feasibility, and stationarity **390-1 488×** tighter in base-objective
units. A single rescaled ADMM run, unpolished, beats what the current pipeline
delivers after five to six polished refinements, and eight rescaled generations
beat twelve current ones. P5.7-A's mechanism is validated.

It does not, however, resolve the oracle instability.

- The refinement drift persists at 80-90 % of its former magnitude: rescaled step
  deltas run −556 684 → −251 242 over seven transitions, against −609 730 →
  −313 346 for the current chain. Neither reaches `tau_planning` = 33 031.
- From generation 2 the rescaled ADMM again converges in **one cycle**, for a
  reason that has nothing to do with scaling: the recourse-change criterion is
  `max(1e3, 1e-3 × recourse)` = **827 971**, which is 25× the planning signal and
  larger than every refinement step either formulation takes. Convergence is
  declared while the recourse is still moving by a quarter to three quarters of a
  million.
- Rescaling introduces a new failure downstream. Better local optimality comes
  with worse agreement — `primal_pf` rises to the 1e-2 tolerance boundary — and
  the exact-consensus polish then fails on 1-4 DSO blocks in seven of eight
  generations (`Restoration Failed`, `Maximum Number of Iterations Exceeded`) and
  degrades the objective by 35 647 836 in the one generation where it succeeds.
  The polish, rho and adaptive-penalty settings are calibrated to the current
  operating point and were not re-tuned, as they are locked.

The scaling change is necessary and validated; it is not sufficient. The binding
constraint on a unique oracle is now the stopping criterion, and the consensus
settings that support it.

```
P5.8-B — objective scaling improves stability but additional ADMM issues remain
```

```
P5.8 COMPLETE — ready for planner review
```
