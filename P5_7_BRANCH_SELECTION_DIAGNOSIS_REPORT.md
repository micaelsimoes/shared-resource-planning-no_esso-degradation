# P5.7 — nonlinear operational branch-selection diagnosis

Branch `feature/derivative-free-planning`. Canonical runtime
`/opt/anaconda3/envs/opf_env_py311/bin/python`, checksum
`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`, verified by
the R0 provenance gate in every harness; `provenance.json` is written under
`data/SRP1/Results/P57/`.

**Nothing production was modified.** The nonlinear SMOPF equations, the ADMM
mathematics and tolerances, H1, the IPOPT settings, the P5.5 convex/MISOCP
diagnostic models and the Benders/master-cut code are untouched. No search of
any kind was implemented or run: no GPS, no MADS, no investment search, no
surrogate. Every experiment below runs on a per-evaluation deep copy, and every
objective rescale is a **positive constant multiple**, which leaves the argmin
and the feasible set identical by construction.

New files: `p57_fingerprint.py`, `p57_eval.py`, `p57_d1_chain.py`,
`p57_analyse.py`, `p57_hypotheses.py`, `p57_d4_penalty.py`. Evidence:
`data/SRP1/Results/P57/`.

---

## 0 — the diagnostic set, and that it is the right one

**The fixed investment candidate is `x0`, the canonical positive-bootstrap
base.** That choice is deliberate and it matters: under the P5.6-D oracle every
continuation point of `H_K(x0)` *is* `x0`, so the twelve solves below differ in
nothing except how many times the state has been re-solved. Investment is held
exactly constant, and any movement between the branches is a pure local-solution
effect with every other variable removed.

The six solutions P5.7 asks for are steps of one chain:

| requested | what it is |
|---|---|
| ADMM | step 1, before the polish |
| polish `K=0` | step 1, polished — no continuation refinement at all |
| `H2` | step 2, polished |
| `H4` | step 4, polished |
| `H8` | step 8, polished |
| `H12` | step 12, polished |

**The chain reproduces the accepted P5.6-D base chain exactly.** Every step that
P5.6-D published was re-derived here, in a fresh process, through a
re-implemented evaluation path:

| step | P5.6-D | P5.7 | delta |
|---|---|---|---|
| 1 | 828 021 090.360850 | 828 021 090.3608505 | +4.8e-07 |
| 2 | 827 415 318.563944 | 827 415 318.5639443 | +3.6e-07 |
| 4 | 826 405 022.193437 | 826 405 022.1934375 | +4.8e-07 |
| 8 | 824 795 363.718628 | 824 795 363.7186280 | 0.0 |
| 12 | 823 731 333.558647 | 823 731 333.5586472 | +1.2e-07 |

The residual is float printing. A diagnosis of a re-implementation that drifted
from the thing being diagnosed would be worthless, so this check is a hard gate
in `p57_d1_chain.py`.

Every step is fully certified: max coordinated residual `1.11e-16`, ESSO
production violation `4.35e-12` (feasible), H1 complementarity violation exactly
`0.0`, converter-capability violation `1.00e-08`, largest network nonlinear
residual `1.15e-05 p.u.` — i.e. IPOPT's own feasibility tolerance, unchanged
across the chain. **The branches are not distinguished by feasibility.**

---

## 1 — objective decomposition across the branches

Weighted base objective, summed over all 48 blocks:

| solution | base objective | vs polish `K=0` | planning objective |
|---|---|---|---|
| ADMM | 837 191 939.32 | **+9 217 420.61** | — |
| polish `K=0` | 827 974 518.72 | 0.00 | 828 021 090.360850 |
| `H2` | 827 368 741.36 | −605 777.35 | 827 415 318.563944 |
| `H4` | 826 358 434.70 | −1 616 084.02 | 826 405 022.193437 |
| `H8` | 824 748 757.55 | −3 225 761.16 | 824 795 363.718628 |
| `H12` | 823 684 710.57 | −4 289 808.14 | 823 731 333.558647 |

By cost family:

| family | ADMM | polish `K=0` | `H12` | `K=0` → `H12` |
|---|---|---|---|---|
| generation_cost | 632 112 173.22 | 631 726 248.26 | 624 416 152.83 | **−7 310 095.44** |
| flexibility_cost | 200 501 812.96 | 196 210 984.28 | 199 251 913.49 | **+3 040 929.21** |
| slack_penalties | 4 511 497.43 | −11 796.94 | −11 796.94 | **−0.00** |
| gen_curtailment_penalty | 53 535.92 | 46 142.92 | 27 077.31 | −19 065.61 |
| ess_complementarity_penalties | 12 919.79 | 2 940.19 | 1 363.88 | −1 576.31 |
| load_curtailment_cost | 0.00 | 0.00 | 0.00 | 0.00 |
| ess_utilization_cost_penalty | 0.00 | 0.00 | 0.00 | 0.00 |

Two mechanisms, and they are **not the same mechanism**:

- **ADMM → polish `K=0`** is slack and flexibility cleanup: `slack_penalties`
  collapses from 4 511 497 to −11 797, `flexibility_cost` drops 4 290 829.
  This reproduces P5.6-A2's corrected attribution.
- **polish `K=0` → `H12`** is a **dispatch re-optimization**: 7 310 095 of
  generation cost bought with 3 040 929 of flexibility cost, and
  `slack_penalties` does not move at all — it is identical to the second decimal
  across all eleven refinements. The slack structure is frozen after the first
  polish; what keeps moving is the economic dispatch.

By agent, over `K=0` → `H12`: **TSO −5 760 064.29**, DSO5 −141 682.57,
DSO7 **+1 279 105.76**, DSO9 **+332 832.95**. Refinement is not a uniform
improvement — it is a redistribution in which the transmission system gains and
two of the three distribution systems lose, netting −4 289 808.

---

## 2 — which variables move between branches, and which do not

Per variable family, in the component's own index order, differenced element by
element between archived solutions (p.u. on a 100 MVA base).

**ADMM → polish `K=0`** (the cleanup step):

| group | max abs delta | L1 | entries moved > 1e-6 |
|---|---|---|---|
| voltage state (`e`,`f`,`vmag_sqr`) | 2.49e-01 | 3.71e+03 | 92 486 / 95 040 |
| flexibility | 7.21e-02 | 4.83e+01 | 113 975 / 114 048 |
| voltage slacks | 4.28e-06 | 2.19e-01 | 58 176 / 62 208 |
| generation dispatch | 4.34e-02 | 6.98e+00 | 10 161 / 13 824 |
| branch flows / OLTC | 1.20e-01 | 3.60e+01 | 4 158 / 33 696 |
| ESS schedules | 1.33e-06 | 5.75e-04 | 8 / 8 640 |
| **TSO/DSO interface (`pc`,`qc`)** | **0.00** | **0.00** | **0** |

**polish `K=0` → `H12`** (the eleven refinements):

| group | max abs delta | L1 | entries moved > 1e-6 |
|---|---|---|---|
| voltage state | 2.27e-01 | 5.62e+02 | 90 214 / 95 040 |
| generation dispatch | 1.30e-01 | 3.36e+01 | 7 700 / 13 824 |
| branch flows / OLTC | 1.30e-01 | 3.38e+01 | 4 306 / 33 696 |
| flexibility | 1.30e-01 | 2.78e+01 | 16 554 / 114 048 |
| ESS schedules | 1.72e-05 | 1.47e-02 | 3 922 / 8 640 |
| **voltage slacks** | **2.19e-13** | **8.85e-12** | **0 / 62 208** |
| **TSO/DSO interface (`pc`,`qc`)** | **0.00** | **0.00** | **0** |

Ranked by L1 over `K=0` → `H12`: `vmag_sqr` 3.66e+02, `voltage_product_real`
3.59e+02, `e` 1.80e+02, `pg` 1.77e+01 (max 1.30e-01 at `DSO9|2030|Spring`),
`qg` 1.59e+01, `f` 1.52e+01, `flex_p_down` 1.27e+01,
`shared_es_pch_hat` 1.26e+01, `r_sqr` 1.09e+01, `flex_p_up` 9.74e+00.

**What moves:** the voltage state, generation dispatch, branch flows, the OLTC
ratio, and load flexibility. Peak movement is 0.13 p.u. = 13 MW on a single
generator — a real dispatch difference, not numerical drift.

**What does not move:**

- **Voltage slacks are frozen** after the first polish — 2.19e-13 across eleven
  refinements, zero entries above 1e-6. The `slack_penalties` row above says the
  same thing in money. Voltage-slack usage is settled by the first polish and is
  not part of the branch difference.
- **ESS schedules barely move** — 1.72e-05 p.u., i.e. 1.7 kW. At the base
  bootstrap capacity the shared ESS is essentially inert, so the branch
  phenomenon is **not** an ESS or H1 phenomenon.
- **The TSO/DSO interface load `pc`/`qc` is exactly constant** — see §5, which is
  a structural finding rather than an absence of one.

---

## 3 — hypothesis C: active-constraint differences

Active means an inequality body within `1e-7` of a bound; equalities are
excluded, since they are always active and carry no information.

| pair | active-set symmetric difference | variable bound-activity changes |
|---|---|---|
| ADMM → polish `K=0` | **3 695** | 138 869 |
| polish `K=0` → `H2` | 70 | 1 214 |
| `H2` → `H4` | 149 | 735 |
| `H4` → `H8` | 191 | 1 742 |
| `H8` → `H12` | 67 | 1 015 |
| polish `K=0` → `H12` | **213** | 3 868 |

The active set **does** change between branches, and it changes in only two
families:

| family | `K=0` → `H12` | active count |
|---|---|---|
| `flex_energy_balance_p` | 69 entered, 34 left | 286 → 321 of 1 152 |
| `sg_capability` | 22 entered, 77 left | 3 554 → 3 499 of 3 732 |
| `sess_comp` | 0 entered, 8 left | 8 → 0 of 1 728 |
| `gen_pf_lower` | 0 entered, 3 left | 10 → 7 of 3 732 |

So the branch difference is a genuine combinatorial change, but a *small* one:
213 rows out of roughly 40 000 inequalities, all of them either the RES
converter-capability circle or the flexibility energy balance.

The ADMM → polish step is a different animal: `sg_capability` goes from **164
active to 3 554 of 3 732** — from 4 % to 95 %. The RES units are pinned to their
capability circle at the polished solution and are not at the ADMM solution.
That is the P5.3-B2-R geometry (`Q_available = 0`, so
`S_available = P_available` and an uncurtailed RES unit sits exactly on the
circle) meeting the scaling result of §4: in the ADMM subproblem the curtailment
penalty that pins them there is divided by ~1.05e5 and is effectively invisible.

---

## 4 — hypothesis A: objective scaling. This is the cause.

Production builds each ADMM subproblem objective as

```
admm_objective = base_objective / effective_scale  +  consensus terms
```

(`shared_resources_planning.py:3592-3596`). Measured over the 48 blocks here,
`effective_scale` ranges `9.41e+04 … 1.16e+05`, median **1.05e+05** — confirming
the ~1.1e5 that P5.6-A2.1 estimated on two blocks.

### A2 — the ADMM subproblems, re-solved from their own converged point

Each of the 48 blocks was re-solved twice from the state the ADMM left it in,
with the consensus parameters untouched: once with production's augmented
objective **as-is**, and once with the **same objective multiplied by that
block's own `effective_scale`**. The multiplier is a positive constant, so the
two problems have identical argmin, identical feasible set, identical starting
point and identical solver settings.

| variant | blocks solved | total base-objective change |
|---|---|---|
| as-is | 48 / 48 | **−383 401.68** |
| × `effective_scale` | 48 / 48 | **−8 892 463.76** |

A factor of **23**. Per block the contrast is sharper still — re-solving as-is
returns `0.00` on most blocks, meaning IPOPT correctly reports that it is already
at a solution of the problem it was given:

| block | `effective_scale` | as-is | rescaled |
|---|---|---|---|
| `DSO7\|2035\|Autumn` | 1.160e+05 | −0.00 | **−320 612.44** |
| `DSO5\|2030\|Autumn` | 1.051e+05 | 0.00 | **−309 589.95** |
| `DSO9\|2030\|Autumn` | 1.051e+05 | 0.00 | **−293 247.64** |
| `DSO5\|2035\|Winter` | 1.160e+05 | 0.00 | **−266 457.40** |
| `DSO5\|2025\|Winter` | 9.518e+04 | −0.00 | **−264 550.78** |

The recovered **8 892 464 is 96.5 % of the entire 9 217 421 gap** between the
ADMM solution and production's polish. The ADMM's local solutions are not
base-objective-optimal, and the reason is arithmetic: at a scale of 1.05e5 an
IPOPT stationarity tolerance on the augmented objective corresponds to a slack
five orders of magnitude larger in base-objective units.

### A1 — the polish NLPs, under equivalent scalings

The same experiment from the other side. The 48 polish NLPs — same fixed
consensus, same capacities, same start — solved under four positive constant
multiples of their own objective:

| polish objective multiplied by | planning objective | vs production scale |
|---|---|---|
| 1e3 | 828 020 345.44 | **−744.92** |
| 1 (production) | 828 021 090.36 | 0.00 |
| 1e-3 | 828 121 726.44 | **+100 636.08** |
| 1 / `effective_scale` (≈ 9.1e-06) | **837 067 437.16** | **+9 046 346.80** |

Scaling the polish objective down to the ADMM's magnitude **puts the polish back
where the ADMM was**: 837 067 437 sits 171 074 below the ADMM point and
9 046 347 above production's own polish. The solution quality is monotone in the
scale over four orders of magnitude, with the same argmin throughout.

**Hypothesis A is confirmed, in both directions, and quantitatively.** The
difference between the ADMM branch and the polished branch is not economics and
not physics. It is the objective scale.

---

## 5 — a structural finding: the interface anchor is frozen at `T0`

The `pc`/`qc` result in §2 — exactly `0.00` movement, at every generation — is
not a null result. `_prepare_transmission_objectives_for_admm` fixes the TSO's
ADN load `pc`/`qc` at the DSO's *then-current* consensus interface power
(`shared_resources_planning.py:2905-2919`, via `fix_or_set`), and the
transmission system then pays `flexibility_cost` for downward deviation from that
anchor. On a **warm start**, `_run_operational_planning` takes the
`_clone_operational_models(initial_state['models'])` path
(`shared_resources_planning.py:2173-2179`) and never re-runs that preparation.

Measured directly against the `T0` template's own models:

```
max | pc(T0) - pc(chain step 1)  |  =  0.0
max | pc(T0) - pc(chain step 12) |  =  0.0        72 of 72 entries fixed
```

The anchor is inherited from `T0`'s cold initialization and is never refreshed —
for every generation and, since `_update_operational_models_with_candidate`
changes capacities only, for every candidate warm-started from `T0`. Meanwhile
the flexibility used against it grows: on `TSO|2025|Spring`,
`sum flex_p_down` moves 6.68332 → 6.98477 p.u. from step 1 to step 12.

Two consequences, stated at the level the evidence supports:

1. It explains the sign of §1's decomposition. Refinement finds a better
   dispatch and *pays more flexibility cost* to hold the interface away from a
   reference frozen in `T0`. The +3 040 929 of flexibility cost is partly the
   price of a stale anchor, not only the price of physical flexibility.
2. Under the locked `T0`-only policy the anchor is at least **common to every
   candidate**, so it does not distort the landscape between candidates. Under a
   **cold** start it is set from that run's own initialization and is therefore
   candidate-dependent — which would mean cold and `T0`-warm evaluations do not
   share an objective function. That reading follows from the code path cited
   above; **it was not measured in this stage** and should be measured before it
   is relied on. It is a candidate explanation for the P5.6-B cold-versus-`T0`
   discrepancy of 1.32e6 … 1.09e7, which was recorded there as a branch effect.

---

## 6 — hypothesis B: initialization is not the cause

The **same** polish NLP — generation 2 at `x0`, same ADMM result, same fixed
consensus, same capacities — solved from four genuinely different starting
points. The transfers are verified: 383 904 values written, 10 368 fixed
coordinated variables correctly skipped, 0 misaligned.

| start | planning objective | vs the ADMM start |
|---|---|---|
| ADMM state (production default) | 827 415 318.5639443 | 0.0000 |
| previous continuation state (step-1 polished) | 827 415 319.3431265 | **+0.78** |
| best known state (step-12 polished) | 827 415 321.9000114 | **+3.34** |
| cold state (fresh model defaults) | 827 415 621.8072588 | **+303.24** |

Total spread **303**, on an objective of 8.3e8 — a relative spread of 3.7e-07.

Set against the numbers this stage has to explain — a 9 217 421 ADMM-to-polish
gap, a 4 289 808 refinement drift, and P5.6-D's 811 438 of landscape uncertainty
— initialization contributes **nothing**. The local NLP, at fixed consensus, has
an essentially unique solution: starting it from the deepest branch found
anywhere in twelve generations moves the answer by 3.34.

That is the single most important negative result in this stage. **The branch
multiplicity is not in the local NLP.**

---

## 7 — hypothesis D: continuation path

At the P5.6-D target `se|node9|2025|-10%`:

| path | solves | result |
|---|---|---|
| direct solve (`K=1` from `T0`) | 1 | **POLISH_FAILURE** |
| capacity continuation (`K=4`) | 4 | VALID, **825 127 965.3618163** |

The `K=4` value reproduces P5.6-D's `H_4` for this candidate exactly
(`H_4(x0) = 826 405 022.19`, `Δ₄ = −1 277 056.83`). Its step objectives are
828 008 580.71 → 826 824 735.61 → 825 952 055.07 → 825 127 965.36, against the
base chain's 828 021 090.36 → 827 415 318.56 → 826 824 028.85 → 826 405 022.19 at
the same step counts. The candidate path and the base path move together; the
capacity change contributes a small part of the movement.

**Self-refinement at fixed capacity could not be run at this target**, because
the direct solve it would start from fails. The control exists at `x0` instead,
and it is decisive: at `x0` the capacity continuation is *by construction* pure
self-refinement — no capacity moves at any step — and eleven of them are worth
−4 289 808. So the improvement along a continuation comes from **re-solving**,
not from traversing capacity space. What capacity continuation contributes
that repeated solving cannot is **reachability**: it turns a POLISH_FAILURE into
a fully certified VALID point.

### D, completed at `x0` — penalty continuation, and the control that settles it

The main harness placed D4 at `se|node9|2025|-10%`, where the direct polish
already fails; its first penalty step failed too and the homotopy never started.
That is a defect of where the experiment was placed, not a result, so it was
re-run at `x0` (`p57_d4_penalty.py`). Both sequences below reuse **one** ADMM
result, so the consensus is identical throughout and cannot contribute anything.

**Penalty continuation** — four polishes with the objective multiplied by
`(1 / effective_scale) ** t`, each warm-started from the previous:

| `t` | multiplier | planning objective |
|---|---|---|
| 1 | 8.62e-06 … 1.06e-05 | 837 067 437.1586325 |
| 2/3 | 4.20e-04 … 4.83e-04 | 828 252 439.8990103 |
| 1/3 | 2.05e-02 … 2.20e-02 | 828 114 036.7070286 |
| 0 | 1.0 | **828 021 088.2782328** |

The homotopy lands **2.08 below** the direct single polish at production scale
(828 021 090.36) — the same point, to eight significant figures. Its `t = 1`
value is bit-identical to A1's, an incidental determinism check. So the scaling
homotopy recovers exactly what a correctly scaled single solve recovers, and
nothing beyond it: at a fixed consensus, there is one answer.

**Fixed-consensus control** — four polishes at production scale, chained, same
solve count as a `K=4` continuation:

| solve | planning objective | change |
|---|---|---|
| 1 | 828 021 090.3608505 | — |
| 2 | 828 021 086.9741508 | −3.39 |
| 3 | 828 021 092.0681605 | +5.09 |
| 4 | 828 021 094.2799644 | +2.21 |

Spread **7.3**, oscillating, no trend — solver-tolerance noise.

Four polish solves at a **fixed** consensus are worth **7**. Four solves of the
refinement chain, which re-run the ADMM and therefore move the consensus, are
worth **−1 616 084**. A ratio of about 230 000. **The refinement creep is carried
entirely by the consensus, not by the local polish NLPs.**

---

## 8 — the ADMM converges in one cycle, on a consensus that is still sliding

| step | planning objective | step delta | ADMM cycles | converged |
|---|---|---|---|---|
| 1 | 828 021 090.360850 | — | 2 | True |
| 2 | 827 415 318.563944 | −605 771.80 | **1** | True |
| 3 | 826 824 028.845478 | −591 289.72 | **1** | True |
| 4 | 826 405 022.193437 | −419 006.65 | **1** | True |
| 5 | 825 961 521.882321 | −443 500.31 | **1** | True |
| 6 | 825 531 306.746985 | −430 215.14 | **1** | True |
| 7 | 825 108 709.695181 | −422 597.05 | **1** | True |
| 8 | 824 795 363.718628 | −313 345.98 | **1** | True |
| 9 | 824 488 243.726694 | −307 119.99 | **1** | True |
| 10 | 824 210 384.187959 | −277 859.54 | **1** | True |
| 11 | 823 978 992.771602 | −231 391.42 | **1** | True |
| 12 | 823 731 333.558647 | −247 659.21 | **1** | True |

Every warm-started generation declares ADMM convergence after a **single cycle**.
The consensus residual test is satisfied immediately — but the consensus point
itself keeps moving, monotonically, generation after generation. Measured on the
fixed interface rows (the DSO reference generator, which the polish pins at the
consensus), against step 1:

| block | step 2 | step 3 | step 4 | step 8 | step 12 |
|---|---|---|---|---|---|
| `DSO5\|2025\|Spring` | 6.13e-03 | 1.28e-02 | 1.77e-02 | 3.69e-02 | **5.30e-02** |
| `DSO7\|2030\|Summer` | 5.63e-03 | 1.11e-02 | 1.66e-02 | 3.88e-02 | **6.09e-02** |
| `DSO9\|2035\|Winter` | 1.37e-03 | 2.46e-03 | 3.36e-03 | 5.98e-03 | 7.87e-03 |

That is 5.3 MW of drift at one interface, accumulating steadily, across twelve
consecutive runs each of which reported itself converged. The shared-ESS
consensus, by contrast, is inert: 6.13e-06 p.u. at the base bootstrap capacity.

**ADMM residual convergence is not optimality here.** Small residuals certify
that the agents *agree*; they say nothing about whether the point they agree on
is the one a correctly scaled local solve would have produced — and §4 measures
that discrepancy at 8 892 464.

The step deltas decay geometrically, ratio **0.926** over steps 5–12 (mean over
all consecutive pairs 0.9225). Extrapolating that ratio gives a remaining tail of
about **−3 097 403** beyond step 12, i.e. a limit near **820 633 931**, and
reaching P5.6-D's required resolution of 33 031 would need roughly **60 further
refinements — `K ≈ 72`**, about 2.1 h per candidate at the measured ~105 s per
step. That extrapolation assumes the geometric rate holds; it rests on eleven
deltas with a stable ratio, which is more support than P5.6-D's two transitions
had, but it is still an extrapolation and is offered as an order of magnitude,
not a target.

---

## 9 — the diagnosis

P5.7 asks which of five candidates is primarily responsible. The evidence
separates them cleanly.

| candidate cause | verdict | the measurement that decides it |
|---|---|---|
| **augmented-objective scaling** | **PRIMARY** | rescaling by a constant recovers 8 892 464 of the 9 217 421 gap (96.5 %) in one pass, and scaling the polish *down* to the ADMM's magnitude puts it back within 171 074 of the ADMM point |
| ADMM initialization | **not the cause** | the same NLP from four different starts, including the deepest branch found anywhere, spreads by **303** on 8.3e8 |
| IPOPT local convergence | **not the cause** | IPOPT converges correctly to what it is given: re-solving an ADMM subproblem as-is returns **0.00**; the identical rescaled problem moves 250 000–320 000 |
| missing globalisation / continuation | **not the cause, but it has one real use** | at a fixed consensus, four chained polishes are worth 7 and the scaling homotopy lands 2 from the direct solve — continuation adds nothing to solution quality. It adds **reachability**: it turns a POLISH_FAILURE into a certified VALID point |
| insufficiently constrained formulation | **not the cause** | every branch satisfies the same audits to the same tolerances; the active-set difference is 213 rows in ~40 000, and the branches differ in dispatch, not in feasibility |

The mechanism, end to end:

1. The subproblem objective is `base / effective_scale + consensus terms`, with
   `effective_scale` ≈ 1.05e5. A stationarity tolerance on that objective is
   five orders of magnitude looser in base-objective units, so every local solve
   stops far from base-optimal — worth 8 892 464 across the 48 blocks (§4).
2. The ADMM's convergence test is on consensus residuals. On a warm start it
   passes after **one cycle**, certifying agreement among under-solved local
   solutions (§8).
3. The polish recovers the base-objective slack at that consensus — uniquely,
   start-independently (§6) and idempotently (§7).
4. Re-running from the polished state hands the ADMM a better warm start; its
   still-under-solved local solves land slightly better; the consensus slides
   ~0.5 MW; the polish recovers a little more. Iterating this is exactly the
   `H_K` chain, and it is why depth changes the answer.

**This is not branch multiplicity of a well-posed problem.** It is one problem
being solved progressively better by a sequence whose termination point is a free
parameter. The signature is directional and monotone — better scaling gives a
better solution across four orders of magnitude, the chain decays geometrically
toward a limit, and the local NLP has one answer regardless of where it starts.
Genuine multi-modality looks nothing like that.

---

## 10 — what would have to change

Stated as findings and recommendations. **None of this was implemented**, per the
stage's instruction not to modify production equations.

1. **Rescale the ADMM subproblem objective, not the base cost.** Production
   forms `base / effective_scale + penalties`. Multiplying through by
   `effective_scale` gives `base + effective_scale · penalties` — identical
   argmin, identical feasible set, and the base objective back at unit
   coefficient. §4 measures the value at 8.89e6 recovered in a single pass. This
   is a numerically-equivalent reformulation, not a change of model, but it does
   change what the ADMM's stationarity and convergence tests see, so it needs its
   own validation stage and an end-to-end A/B before anything is claimed.
2. **The ADMM convergence test needs an optimality component.** Residual-only
   convergence after one cycle certified twelve consecutive points that were
   still sliding by 0.5 MW per generation. Whatever the fix in (1) achieves,
   a test that cannot distinguish "agreed" from "optimal" will keep producing
   depth-dependent answers.
3. **Decide what the frozen `T0` interface anchor should be** (§5). At present
   the flexibility-cost reference is inherited from `T0`'s cold initialization
   and never refreshed on a warm start. It is at least consistent across
   candidates under the locked `T0` policy; under a cold start it is not, and
   that should be measured before the cold-versus-`T0` gaps recorded in P5.6-B
   are interpreted as branch effects.
4. **Retain continuation for reachability only.** It buys nothing in solution
   quality at a fixed consensus (7 across four solves) and it is the difference
   between POLISH_FAILURE and a certified VALID point at the P5.6-D target.
5. **Do not pursue depth as the remedy.** `K ≈ 72` at ~2.1 h per candidate is
   what the current oracle would need to resolve the signal P5.6-D requires, and
   that is treating the symptom.

---

## 11 — what was not done

- No production equation, solver setting, ADMM tolerance or master-cut code was
  modified.
- No search of any kind was implemented or run.
- Nothing was installed.
- The end-to-end rescaled ADMM of recommendation (1) was **not** run; §4 measures
  it one subproblem at a time from the converged point, which is the strongest
  statement available without a production change.
- The cold-start anchor claim in §5 is a code-path reading, explicitly **not**
  measured here.
- Self-refinement at fixed capacity at the `se|node9|2025|-10%` target could not
  be run because the direct solve it starts from fails; the equivalent control
  exists at `x0`, where continuation is by construction self-refinement.

---

## Verdict

The local nonlinear NLP is not multi-modal in any way that matters: at a fixed
consensus it returns the same solution to within 303 from four starts spanning
twelve generations of refinement, and to within 7 under repeated re-solution. The
whole depth dependence traces to one arithmetic defect — the base objective
divided by ~1.05e5 inside the ADMM subproblem — which is worth 8 892 464, is 96.5
% of the gap being chased, and is one numerically-equivalent reformulation away.
The chain decays geometrically toward a limit rather than wandering between
basins.

That does not make the fix free: rescaling changes what the ADMM's convergence
test sees, its residual-only criterion needs an optimality component, and the
frozen `T0` interface anchor is a separate unresolved coupling of the objective
to history. Each needs its own validation before the oracle can be called unique.
But the cause is identified, localized and numerical, not structural.

```
P5.7-A — unique operational oracle can likely be recovered
```
