# P5.15 — Gates G1–G5 HELD. The degradation input is a function of the solver, not the model.

**Planner report. Supersedes the "run G1–G5" step of `PLANNER_BRIEF_2026-09-13.md`
Addendum 2's dispatch order.**

## Status

| step | state |
|---|---|
| (a) repair `p514_n_instrumented_cstar.py`, `p514_l_capacity_ladder.py` | done |
| (b) ε sensitivity check | done — **hit its predeclared STOP condition** |
| (c) gates G1–G5 | **NOT DISPATCHED** |
| (d) `REVISION_CONTEXT.md` rewrite | held until the remedy is chosen |

The predeclared rule was: *gates run at ε = 1e-3 regardless, unless the 1e-3 arm itself shows a
complementarity detector above 1e-6.* It showed **4.6319e-4**. The rule fired, and it overrides
the dispatch order. No gate was run.

---

## 1. The finding

P5.15 Step 1 item 3 deleted the ESSO's enforced complementarity rows on a stated assumption,
written into the source at `shared_energy_storage_data.py:400-412`:

> "...since charging and discharging both cost the same epsilon per unit of throughput and no
> other term in the objective rewards using both directions at once, **an LP optimum never has
> pch > 0 and pdch > 0 simultaneously** unless forced to by another binding constraint."

**The LP statement is true. The inference that IPOPT returns it is false.** IPOPT is an
interior-point method; it does not return vertices. The measured leak obeys

```
min(pch, pdch)  =  mu_final / (2 · s_obj · eps)
```

where `mu_final` is the terminal barrier parameter and `s_obj` IPOPT's objective scaling.
Verified by the Planner directly from the committed log
(`data/SRP1/Results/P5151/eps_check_logs/eps1e-3/optim_log_node_7.txt`):

| term | value | source |
|---|---|---|
| `s_obj` | **0.1** exactly | scaled/unscaled `Objective` pair, 2.3830335995985320e-03 / …e-02 |
| `mu_final` | 9.2306180592606601e-08 | scaled `Complementarity` line |
| predicted `mu/(2·s_obj·eps)` | **4.6153e-4** | the identity |
| measured `max min(pch,pdch)` | **4.6319e-4** | per-period artifact |
| agreement | **0.36 %** | — |
| identity `mu·(1/pch + 1/pdch)` vs `2·s_obj·eps` | 2.0020e-4 vs 2.0000e-4 | **0.10 %** |

**Therefore: `es_avg_ch_dch_per_unit` — the input to the degradation law — is a function of the
solver's stopping tolerance, objective scaling and barrier parameter.** It moves with `tol`,
with `PENALTY_ESSO_SLACK` (through `s_obj`), with `nlp_scaling_method`, and with warm versus
cold start. **A physical quantity that depends on solver tolerance is not a model output.**

That is the finding. The 4.63e-4 is a symptom of it.

### What this retires

- **"1e-3 is 154× better than 1e-7, so the regularization works" is wrong.** Both arms have the
  same LP optimum (min = 0). The 154× measures `mu/eps`, nothing else. The ε=1e-7 arm is the
  degenerate case: its scaled objective gradient (1e-8) is below the dual-feasibility tolerance,
  so the iterate stops wherever Newton left it — which is why the identity fails there by 92×
  rather than holding.
- **Raising ε cannot fix it.** Reaching the detector's `1e-6·s_max` requires
  `eps ≥ mu/(2·s_obj·1e-6) = 0.46`, which is ~5e4× the measured AL consensus gradient. The ESSO
  would become "minimize throughput" with the ADMM coupling as noise.
- **An exact penalty or big-M on `min(pch,pdch)` is the same lever, not a different one.** The
  offset is `mu/(2·s_obj·weight)` for any weight. Exactness of the penalty is irrelevant,
  because the offset is not an inexactness of the penalty — it is an inexactness of the solver.

### Rule ten, applied

The ε=1e-3 arm terminated at constraint violation **9.8960e-07** against a 1e-6 tolerance —
**98.96 % of its threshold**. That run was stopped, not settled. The ε=1e-7 arm stopped at
9.9e-13, i.e. 0.0001 %. The contrast is itself diagnostic and is exactly what the barrier
mechanism predicts.

---

## 2. Hypotheses

- **H1 — interior-point barrier residual. CONFIRMED.** Verified to 0.1 % by the Planner from
  committed artifacts, independently of the Advisor who proposed it.
- **H2 — structural shadow price: simultaneous ch/dch as the model's only dissipation channel,
  bought whenever a bound requires energy to disappear. REFUTED.** This was the Planner's
  hypothesis and it is wrong at the level of the constraint inventory. `_build_subproblem`
  creates ten `ConstraintList` families and **not one of them is a state-of-charge recursion or
  an energy balance**; there is no SoC variable in the ESSO at all. `eff_ch`/`eff_dch` appear in
  exactly one place, line 516, inside the degradation accounting — which enters the objective
  **nowhere**. With no energy balance there is nothing to dissipate, no shadow price, and the
  LP optimum is `min = 0` for every ε > 0.
- **H3 — multi-cohort split degeneracy. OPEN, distinct, survives every remedy.** With two or
  more active cohorts the ε term is indifferent to how `|pnet|` is split among them, so
  **per-cohort `D` and per-cohort SoH are non-unique even under perfect complementarity.** Not
  caused by Step 1; masked in the current single-cohort fixtures. If G1's SoH gate is
  per-cohort, this is an independent reason it may not be well-posed.
- **H4 — the G1 reference is itself leaky. CONFIRMED, and more leaky than the new model**
  (0.007729 vs 4.6319e-4 by ratio). See §6.

---

## 3. Materiality

At ε=1e-3 the spurious throughput is **0.9175 %** of the total on the ε fixture (0.2667 of
29.067 p.u., present in **288/288** cohort-periods). Reconciles exactly: 288 cohort-periods at
|p_req| = 0.10 gives a true throughput of 28.80, plus `2·4.63e-4·288 = 0.267`, total 29.067.

The leak δ is **absolute** — it does not scale with `s_max` — so the *relative* inflation falls
at material capacity. At C\*-scale duty (EFC/day ≈ 0.99, E_rated ≈ 3.24) it is **≈ 0.35 %**.
Propagated through `D ∝ throughput` and `soh = Π exp(−D)`:

| quantity | shift at C\*-scale (0.35 %) | shift at fixture rate (0.92 %) | gate | over by |
|---|---|---|---|---|
| SoH year 1 | 5.2e-4 | 1.35e-3 | 1e-6 abs | 520× – 1,350× |
| SoH terminal | 1.0e-3 | 2.69e-3 | 1e-6 abs | 1,030× – 2,690× |
| EFC/day | 3.9e-3 | 1.0e-2 | 1e-4 | 39× – 100× |

**G1's 1e-6 SoH gate is unattainable by two to three orders of magnitude, and the error
compounds year on year.** The C\*-scale column is an extrapolation, not a measurement; the
conclusion is robust to a factor of five either way.

### A correction to the Planner's own reading

I first inferred from the leak's uniformity (coefficient of variation 2.55e-4 across all 288
periods) that it was a structural floor rather than demand-driven. **That inference was
unsound**: the fixture holds `|pnet|` constant at 0.1000 in every period, so a demand-driven
leak would look identical. The fixture cannot discriminate those two mechanisms. The barrier
identity is what discriminates them, and it does so decisively.

### A defect in the ε check's headline ask, which was the Planner's

`create_shared_energy_storage_model` **hard-fixes** `es_pnet` to `p_req`, so the measured
displacement of exactly 0.0 is true by construction and tests nothing. The Planner's §3 algebra
predicting ~4e-7 p.u. assumed the ADMM form, which frees `es_pnet` — i.e. the brief specified an
instance that cannot exercise the mechanism its own prediction was about. The Worker measured
what was asked, flagged the mismatch, and did not silently switch fixtures. That was correct.

The mismatch does **not** narrow the finding. With `pnet` free, its stationarity drives the AL
gradient toward ∓`eps_s`, restoring the same leak within a factor of two (4.6e-4 to 9.3e-4).
And the fixed-`pnet` path is production, not a toy: it is the ADMM *initialization* solve, and
`get_updated_capacities` reads SoH from exactly that solve to feed `es_e_available` into the
TSO/DSO models for the entire run.

### One consequence that is worse in the ADMM form

`s_obj = min(1, 100/‖∇f‖_∞)`. Today `‖∇f‖_∞ = PENALTY_ESSO_SLACK = 1e3`, giving `s_obj = 0.1`.
As ADMM duals grow, `dual/(2S)` can exceed 1e3, `s_obj` falls, and **the leak — hence the
degradation input, hence the capacity fed to the networks — grows across cycles.** This is a
derivation, not a measurement. Given P5.14-N's 74-of-90 cycles with local-solve failures under
a 15.4 % `cl_eff` perturbation, it is worth knowing before anything is attributed to the ageing
model itself.

---

## 4. What the L ladder actually shows, and what it costs a remedy

The repaired ladder now reports **all three nodes optimal at 1.00 MVA**, where node 7 was
`infeasible` pre-reformulation. That is not evidence that deleting complementarity fixed it.
The preserved pre-repair log `data/SRP1/Results/P514L/rung_1.00.log` records the violated
components at node 7:

```
rated_s_capacity_unit: 6.845e-05,  rated_s_capacity: 2.282e-05,
energy_storage_operation_agg: 1e-08,  energy_storage_normalization: 0.0
```

**The infeasibility was on the capacity rows. The complementarity family was at 0.0.** The
P5.14-L report had already falsified complementarity as the suspect. `shared_energy_storage_data.py:446`
now reads `es_s_rated_per_unit == es_s_investment_fixed` — a `Var == Param` row where it was
`Var == Var`. **Candidate 2 (investments as parameters) is what fixed the ladder**, not Step 1
item 3.

This matters because it removes the main argument for keeping the deletion.

---

## 5. Remedies, ranked on correctness first

- **(a) Restore the enforced row — REJECT.** It would make the *measured* detector worse, not
  better. The deleted row was `pch_hat·pdch_hat ≤ slack_es_ch_comp_per_unit + 1e-4` in
  normalized variables with the slack **penalized, not forbidden** — a relaxed complementarity
  admitting `min/s_max` up to ~1e-3 at the tested duty and ~1e-2 at idle, against the 4.63e-4
  now observed -- and Diagnostic A **measured** exactly that: the old row delivered 0.007729,
  16.7x the leak it would be restored to cure. It also reinstates the LICQ-degenerate idle
  vertex and a slack that prices a modelling inconsistency. If there is a case for restoring it, it is the aggregate-compatibility
  argument with the network side, which is a different argument on different merits.
- **(b) Raise ε and (c) exact penalty / big-M — REJECT, and they are one lever.** ε ≈ 0.46
  required; the offset is `mu/(2·s_obj·weight)` for any weight.
- **(e) post-solve correction only — REJECT.** It leaves the in-model `soh_min` floor,
  `es_e_available` and salvage running on the leaky value while reporting a corrected one. Two
  SoH numbers is worse than one wrong one.
- **(d)/(e2) single signed per-cohort power — RANKED FIRST.** Replace the directional pair with
  one signed variable and define the degradation input as a deterministic smooth function of it,
  e.g. `t = eff_ch·(√(p²+δ²)+p)/2 + (1/eff_dch)·(√(p²+δ²)−p)/2`, exact to O(δ²/|p|). No
  complementarity, **no `mu` dependence**, half the variables, and the LICQ-degenerate idle
  vertex is gone. This is sound *because* the pair is a pure accounting device: `pch`/`pdch`
  appear in production only in the degradation accounting, the box limits, the aggregate `pnet`
  row, the ε term, cohort fixing, and results/detector — there is no SoC balance for a net-based
  definition to leave wrong. The artifact confirms the net is clean while both legs are inflated
  by the same δ: `0.10046 − 0.00046 = 0.09999`.

**This must be classified honestly.** (d)/(e2) *does* change the mathematical formulation. The
justification is not "it converges better" — it is that **the present definition does not define
a quantity**. That is the strong form of the argument and it meets the bar for a formulation
change. "The detector fails" does not.

---

## 6. G1 is not well-posed. Three independent reasons.

Diagnostic A (zero solves, guard-verified 0/0, on the preserved pre-reformulation control
pickle) settles this. Integrity confirmed independently by the Planner: mtime 2026-09-13
**11:35:16Z**, four hours before the reformulation commit `b03c9b14` (15:30:51Z);
`es_degradation_per_unit` and `energy_storage_complementarity` present, `es_D_per_unit` absent.

### Reason 1 — the reference is itself a solver artifact, and a leakier one

| quantity | OLD control (the G1 reference) | NEW reformulated |
|---|---|---|
| `max min(pch,pdch)/s_max` | **0.007729** | 4.6319e-4 |
| spurious throughput fraction | **1.386 – 1.391 %** | 0.9175 % |

**The old reference leaks 16.7× more by ratio and ~1.5× more by throughput fraction than the
model it is being used to gate.** So the reformulation did **not** introduce a leak — it
*reduced* it. What it did not do is remove the defect: it swapped one arbitrary constant for
another. The old leak was set by the relaxed row's **1e-4 product tolerance**
(`pch_hat·pdch_hat ≤ slack + 1e-4`, which admits `min/s_max` up to 1e-2 when the legs are
equal — consistent with the 0.0077 measured, and `slack_es_ch_comp_per_unit` was **0.0 across
all 288 evaluated periods**, so the row sat exactly on its own tolerance). The new leak is set
by `mu/(2·s_obj·eps)`. Neither is physics.

**This also corrects the published number.** The trajectory `1.0 → 0.8387 → 0.7284 → 0.6248`
and its "37.5 % capacity loss" embed ≈1.39 % spurious throughput, so the reported degradation
is overstated by roughly that fraction of `D`.

### Reason 2 — the reference is one node's trajectory, quoted without naming the node

The brief states the reference as `1.0 → 0.8387 → 0.7284 → 0.6248`. Measured per node in the
control artifact:

| node | yr 1 | yr 2 | yr 3 | max deviation from the stated reference |
|---|---|---|---|---|
| 5 | 0.838692 | 0.728405 | 0.624795 | 8.4e-06 (**8×** the gate) |
| 7 | 0.838724 | 0.726780 | 0.623318 | 1.62e-03 (**1,620×**) |
| 9 | 0.838693 | 0.726012 | 0.622817 | 2.39e-03 (**2,388×**) |

The reference is **node 5's**. Nodes 7 and 9 were never within 1e-6 of it. **The control run
that produced the reference fails G1's own gate at two of its three nodes** — before any
reformulation, any leak and any remedy. The gate is unsatisfiable as written.

This is the project's own identifier rule: a reported statistic must uniquely denote its
content, and this one silently denotes one node of three.

### Reason 3 — H3, latent here

The multi-cohort split degeneracy leaves per-cohort `D` and SoH non-unique. It is **masked in
this artifact** — only cohort `y_inv = 0` is active; `(1,*)` and `(2,*)` sit at SoH 1.0 — so it
did not contribute to the numbers above. It would bite any future multi-cohort instance and
survives every remedy on the list.

### Consequence

**G1 must be re-specified as part of the remedy, not after it**, and it cannot be re-specified
against this reference. Note also that the ε rule that produced this STOP set a 1e-6 threshold
**no interior-point solve of this formulation could have met at any usable ε**. The Worker was
right to stop; the threshold was mis-specified, by the Planner.

## 7. Actions taken and not taken

Taken: the two G-dependent harnesses are repaired and verified; the ε check ran guard-exact
(6/6 solves); Diagnostic A dispatched (zero solves).

**Not taken, and requiring author decision:**

- No remedy implemented. No production formulation change.
- **The false comment at `shared_energy_storage_data.py:400-412` still stands.** It asserts the
  proposition this measurement falsified, and it is the stated justification for three deletions.
  It must be amended whatever is decided.
- `REVISION_CONTEXT.md` rewrite held: its content depends on the remedy.
