# AB1 — adaptive rho on the cold path — INTERIM REPORT

**Status: treatment arm COMPLETE. Control arm IN FLIGHT (cycle ~11 of 50 at the time of
writing). No conclusion is drawn, because the decision-relevant quantity is the
comparison, and half of it does not exist yet.**

Frozen design: `data/SRP1/Results/P514S2/frozen_ab1_adaptive_cold_v2_6e23cac0.json`
(SHA-256 `6e23cac003f67426d24ece5a7a2f6664381d4ba266e832e5f5f775d999afdf6f`), frozen and
committed (`eaaa196a`) before either arm ran.
Results so far: `data/SRP1/Results/P514AB1/ab1_treatment.json`.

## 1. Treatment arm (adaptive_penalty = True, cold, rho_pf start 300, cap 50)

| Quantity | Value |
|---|---|
| cycles to convergence | **32** (cap 50 — **not** hit) |
| local solves | 1683 |
| wall clock | 1293 s (21.5 min), ~39 s/cycle |
| final `rho_pf` | **3.4683** |
| decrease / increase actions | 11 / 0 |
| final recourse | 826,829,641.1 |
| local solve failures | **1**, at cycle 11 |

### The rho walk

| cycle | `rho_pf` after | action |
|---|---|---|
| 1 | 200.0000 | decreased |
| 2 | **133.3333** | decreased |
| 3 | 88.8889 | decreased |
| 4 | 59.2593 | decreased |
| 5 | 39.5062 | decreased |
| 6 | 26.3374 | decreased |
| 7 | 17.5583 | decreased |
| 8 | 11.7055 | decreased |
| 9 | 7.8037 | decreased |
| 10 | 5.2025 | decreased |
| 12 | **3.4683** | decreased — then held for 20 cycles |

## 2. The secondary prediction is FALSIFIED

Predeclared: `rho_pf` settles in **[80, 200]**, most likely **133.33**, on the reasoning
that the update rule's fixed point is a property of the problem and the dead band rather
than of the starting value — P5.9-B having settled at 131.687 from a start of 1000.

Observed: **3.4683**, roughly **38x below** the predicted band. The walk passed *through*
133.33 at cycle 2 and kept descending for nine more steps.

The attractor hypothesis, as stated, is wrong. The frozen spec recorded in advance that
a different cold-path fixed point would be *informative rather than a failure* — the
cold iterates differ from the warm ones — and that only oscillation or non-settling would
falsify the attractor *idea*. Neither occurred: the rule settled cleanly, held for 20
cycles, and never reversed. So what is falsified is the specific prediction that the
fixed point is initialization-independent. **It is not: warm gives 131.687, cold gives
3.4683, from the same rule and the same dead band.**

That is a sharper finding than the prediction would have been. It says the adaptive
rule's operating point is a property of *the trajectory it is placed on*, which is the
same initialization-dependence the S2 cost model identified as the dominant effect.

## 3. Was the convergence real, or manufactured by shrinking rho?

This must be asked, because the dual residual is proportional to rho
(`shared_resources_planning.py:4938`), so collapsing rho satisfies `stationarity_pf` by
construction. A 38x rho collapse followed by a convergence declaration is exactly what
the failure mode would look like.

Two independent pieces of evidence say the convergence is real:

1. **The primal residual does not depend on rho, and it is genuinely satisfied.** At
   cycle 32, `primal_pf_ratio = 0.4526` and `primal_pf = 0.004526` against a tolerance of
   0.01. Consensus was achieved, not priced away. Final slacks: consensus_pf 2.21,
   stationarity_pf 4.74, consensus_v 297.6, objective 1.17 — every criterion satisfied
   with margin, and the binding one is now the objective at 1.17, not `stationarity_pf`.
2. **The final dual reduction occurred at constant rho.** `rho_pf` was fixed at 3.4683
   from cycle 12 onward. Over that stretch `dual_pf_mean_ratio` fell 1.855 (cycle 15) ->
   0.797 (20) -> 0.396 (25) -> 0.211 (32). With rho held, that reduction can only come
   from shrinking iterate increments — which is convergence, not rescaling.

The recourse trajectory corroborates: 2,355M (cycle 1) -> 1,466M (5) -> 855M (20) ->
826.8M (32), with the per-cycle change falling to 706,603 against a tolerance of 827,536.

## 4. Caveats recorded now, not later

**A local solve failed at cycle 11** (`recourse = None`), during the rho collapse. The
run continued and converged 21 cycles later. This is a fresh instance of the S3 class:
1 failure in 1683 solves in this arm. No causal claim is made linking it to the collapse;
the coincidence in timing is recorded as an observation. Production holds the penalty
update after a solver failure (`action = 'held after solver failure'`), so the failure
perturbed the adaptation path by one cycle.

**Both arms run under C3.** The objective is therefore **not comparable** with any
pre-C3 figure — not with P5.9-B's 827.84M, not with P5.12-R's trajectory. C3 changed the
degradation term, so comparing across that boundary would repeat the salvage
incomparability error. The A/B comparison between the two AB1 arms is unaffected, because
both run under the same formulation.

**Solve-profile identity.** Observed 1683 permitted solves against the predeclared
identity `51 x 32 = 1632` — a difference of exactly 51, one initialization block,
matching the P5.12-R ledger structure where cycle 0 is initialization. Reported, not
absorbed, as the frozen spec required. Guards: 1683 permitted, **0 blocked**.

## 5. What cannot be said yet

**Nothing about whether adaptation helps.** 32 cycles is a number, not a comparison. If
the control arm converges in fewer than 32, adaptation *hurt* on the cold path despite
converging; if it hits the cap, the control is inconclusive by predeclaration and the
cold-path cost remains unknown — in which case AB1 will have established that adaptive
rho converges the cold path in 32 cycles and nothing about the counterfactual.

The control arm is the deliverable that answers what the cold path actually costs. It is
in flight at roughly 49 s/cycle; at the cap it completes in about 41 minutes.

## 6. What the treatment arm has already established

- The cold path **can** converge: 32 cycles, 1683 solves, ~22 minutes, with every
  criterion satisfied and consensus genuinely achieved. No previous artifact showed a
  converged cold trajectory — P5.12-R was capped at 21 and still descending.
- The adaptive rule's fixed point is **initialization-dependent** (3.47 cold vs 131.687
  warm), which falsifies the attractor prediction and reinforces the S2 finding that
  initialization dominates.
- `stationarity_pf` is not the binding criterion at convergence here; the objective is.
