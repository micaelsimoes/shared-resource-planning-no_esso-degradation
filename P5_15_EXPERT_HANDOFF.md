# P5.15 — Handoff to the external expert

**Subject: the ESSO degradation input is a function of the solver, not of the model.
Gates G1–G5 are held. Gate G1 is not well-posed and cannot be re-specified against its
current reference.**

Planner report, 2026-09-13. Supersedes the "run G1–G5" step of the Addendum-2 dispatch order.
Detail and derivations: `P5_15_GATE_HOLD_REPORT.md`.

---

## 1. What was asked, and where it stopped

Addendum 2 dispatched: (a) repair two gate-dependent harnesses → (b) ε sensitivity check →
(c) gates G1–G5 → (d) `REVISION_CONTEXT.md` rewrite.

| step | state |
|---|---|
| (a) harness repair | done, verified |
| (b) ε check, 1e-3 vs 1e-7 | done — **hit its predeclared STOP condition** |
| (c) gates G1–G5 | **NOT RUN** |
| (d) `REVISION_CONTEXT.md` rewrite | held — its content now depends on the remedy |

The predeclared rule was: *gates run at ε = 1e-3 regardless, unless the 1e-3 arm itself shows a
complementarity detector above 1e-6.* It showed **4.6319e-4**. The rule fired and overrode the
dispatch order.

**Note the rule's own defect**: 1e-6 was a threshold **no interior-point solve of this
formulation could have met at any usable ε** (§2). The stop was correct; the threshold was
mis-specified, by the Planner.

---

## 2. The finding

P5.15 Step 1 item 3 deleted the ESSO's enforced complementarity rows on an assumption written
into `shared_energy_storage_data.py:400-412`:

> "…since charging and discharging both cost the same epsilon per unit of throughput and no
> other term in the objective rewards using both directions at once, **an LP optimum never has
> pch > 0 and pdch > 0 simultaneously** unless forced to by another binding constraint."

**The LP statement is true. The inference that IPOPT returns it is false.** IPOPT is an
interior-point method and does not return vertices. The leak obeys

```
min(pch, pdch)  =  mu_final / (2 · s_obj · eps)
```

Verified by the Planner directly from the committed log, independently of the Advisor who
proposed it:

| term | value | source |
|---|---|---|
| `s_obj` | **0.1** exactly | scaled/unscaled `Objective` pair |
| `mu_final` | 9.2306180592606601e-08 | scaled `Complementarity` line |
| predicted `mu/(2·s_obj·eps)` | 4.6153e-4 | the identity |
| measured `max min(pch,pdch)` | 4.6319e-4 | per-period artifact |
| **agreement** | **0.36 %** | — |
| identity `mu·(1/pch+1/pdch)` vs `2·s_obj·eps` | 2.0020e-4 vs 2.0000e-4 | **0.10 %** |

**Consequence.** `es_avg_ch_dch_per_unit` — the input to the degradation law — moves with `tol`,
with `PENALTY_ESSO_SLACK` (through `s_obj`), with `nlp_scaling_method`, and with warm versus
cold start. A physical quantity that depends on solver tolerance is not a model output. That is
the defect; the 4.63e-4 is a symptom.

**Three things this retires.**

- "1e-3 is 154× better than 1e-7, so the regularization works" is wrong. Both arms share the LP
  optimum `min = 0`; the 154× measures `mu/eps`. (The ε=1e-7 arm is the degenerate case: its
  scaled objective gradient falls below dual-feasibility tolerance, so the iterate stops wherever
  Newton left it — which is why the identity fails there by 92× rather than holding.)
- **Raising ε cannot work**: reaching `1e-6·s_max` needs `eps ≥ 0.46`, ~5e4× the measured AL
  consensus gradient. The ESSO would become "minimize throughput" with ADMM coupling as noise.
- **An exact penalty or big-M is the same lever**, not a different one: the offset is
  `mu/(2·s_obj·weight)` for any weight. The offset is an inexactness of the *solver*, not of the
  penalty, so exactness is irrelevant.

**Rule ten.** The ε=1e-3 arm terminated at constraint violation 9.8960e-07 against a 1e-6
tolerance — **98.96 % of threshold**. Stopped, not settled. The 1e-7 arm stopped at 0.0001 % of
threshold. Exactly what the barrier mechanism predicts.

### A hypothesis of the Planner's that was refuted

I proposed that simultaneous charge/discharge was being **bought as a dissipation channel at a
shadow price** (round-trip efficiency < 1, purchased whenever a bound required energy to
disappear). This is wrong at the level of the constraint inventory: `_build_subproblem` creates
ten `ConstraintList` families and **not one is a state-of-charge recursion or an energy
balance**; there is no SoC variable in the ESSO at all, and `eff_ch`/`eff_dch` appear in exactly
one place — the degradation accounting, which enters the objective nowhere. With no energy
balance there is nothing to dissipate and no shadow price.

I also read the leak's uniformity (CoV 2.55e-4 over 288/288 periods) as evidence of a structural
floor. **Unsound**: the fixture holds `|pnet|` constant, so a demand-driven leak would look
identical. The barrier identity is what discriminates, and it does so decisively.

---

## 3. Gate G1 is not well-posed — three independent reasons

Settled by **Diagnostic A**: zero solves, guard-verified 0/0, on the preserved pre-reformulation
control model set. Integrity confirmed by the Planner independently of the Worker: mtime
**11:35:16Z**, four hours before the reformulation commit `b03c9b14` (15:30:51Z);
`es_degradation_per_unit` and `energy_storage_complementarity` present, `es_D_per_unit` absent.

### Reason 1 — the reference is a *leakier* artifact than the model it gates

| quantity | OLD control (the G1 reference) | NEW reformulated |
|---|---|---|
| `max min(pch,pdch)/s_max` | **0.007729** | 4.6319e-4 |
| spurious throughput fraction | **1.386 – 1.391 %** | 0.9175 % |

**16.7× leakier by ratio.** So the reformulation did **not** introduce a leak — it *reduced*
it. What it failed to do is remove the defect: it swapped one arbitrary constant for another.
The old leak was set by the relaxed row's **1e-4 product tolerance**
(`pch_hat·pdch_hat ≤ slack + 1e-4`, admitting `min/s_max` up to 1e-2 with equal legs — consistent
with 0.0077 measured, and `slack_es_ch_comp_per_unit` was **0.0 across all 288 evaluated
periods**, i.e. the row sat exactly on its own tolerance). The new leak is set by
`mu/(2·s_obj·eps)`. **Neither is physics.**

**This corrects a published number.** `1.0 → 0.8387 → 0.7284 → 0.6248` and its "37.5 % capacity
loss" embed ≈1.39 % spurious throughput; the reported degradation is overstated by about that
fraction of `D`.

### Reason 2 — the reference is one node's trajectory, quoted without naming the node

| node | yr 1 | yr 2 | yr 3 | max deviation from the stated reference |
|---|---|---|---|---|
| 5 | 0.838692 | 0.728405 | 0.624795 | 8.4e-06 (**8×** the gate) |
| 7 | 0.838724 | 0.726780 | 0.623318 | 1.62e-03 (**1,620×**) |
| 9 | 0.838693 | 0.726012 | 0.622817 | 2.39e-03 (**2,388×**) |

The reference is **node 5's**. **The control run that produced the reference fails G1's own
1e-6 gate at two of its three nodes** — before any reformulation, leak or remedy. The gate is
unsatisfiable as written. This is the repository's own identifier rule: a reported statistic must
uniquely denote its content.

### Reason 3 — the multi-cohort split degeneracy (latent here)

The ε term is indifferent to how `|pnet|` is split among active cohorts, so per-cohort `D` and
SoH are non-unique **even under perfect complementarity**. Masked in this artifact (only cohort
`y_inv = 0` active) but it would bite any multi-cohort instance and **survives every remedy
below**.

---

## 4. Materiality of the leak

At ε=1e-3 the spurious throughput is **0.9175 %** on the ε fixture (288/288 cohort-periods;
reconciles exactly: 288 × 0.10 = 28.80 true, plus 2·4.63e-4·288 = 0.267, total 29.067 measured).
δ is **absolute** — it does not scale with `s_max` — so relative inflation falls at material
capacity: **≈0.35 %** at C\*-scale. Propagated through `D ∝ throughput`, `soh = Π exp(−D)`:

| quantity | at C\*-scale | at fixture rate | gate | over by |
|---|---|---|---|---|
| SoH year 1 | 5.2e-4 | 1.35e-3 | 1e-6 abs | 520× – 1,350× |
| SoH terminal | 1.0e-3 | 2.69e-3 | 1e-6 abs | 1,030× – 2,690× |
| EFC/day | 3.9e-3 | 1.0e-2 | 1e-4 | 39× – 100× |

The C\*-scale column is an extrapolation, not a measurement; the conclusion is robust to a factor
of five either way.

**One consequence that is worse in the ADMM form** (derivation, not measurement):
`s_obj = min(1, 100/‖∇f‖_∞)`. Today `‖∇f‖_∞ = PENALTY_ESSO_SLACK = 1e3`, so `s_obj = 0.1`. As
ADMM duals grow, `dual/(2S)` can exceed 1e3, `s_obj` falls, and **the leak — hence the
degradation input, hence the capacity fed to the networks — grows across cycles.** Worth knowing
before anything is attributed to the ageing model itself, given P5.14-N's 74-of-90 cycles with
local-solve failures under a 15.4 % `cl_eff` perturbation.

---

## 5. A collapsed argument: what actually fixed the capacity ladder

The repaired ladder now reports all three nodes optimal at 1.00 MVA, where node 7 was
`infeasible` pre-reformulation. **This is not evidence that deleting complementarity fixed it.**
The preserved pre-repair log records node 7's violated components as
`rated_s_capacity_unit: 6.845e-05`, `rated_s_capacity: 2.282e-05`, with
**`energy_storage_normalization: 0.0`**. The infeasibility was on the **capacity** rows; the
complementarity family was at zero, and the P5.14-L report had already falsified it as the
suspect. `shared_energy_storage_data.py:446` now reads `es_s_rated_per_unit ==
es_s_investment_fixed` — a `Var == Param` row where it was `Var == Var`.

**Candidate 2 (investments as parameters) fixed the ladder, not Step 1 item 3.** This removes
the main standing argument for keeping the deletion.

---

## 6. Remedies, ranked on correctness first

- **(a) restore the enforced row — REJECT on measurement.** Diagnostic A measured what it
  delivers: **0.007729, i.e. 16.7× the leak it would be restored to cure.** It also reinstates
  the LICQ-degenerate idle vertex and a slack that prices a modelling inconsistency. If a case
  for restoring it exists it is the aggregate-compatibility argument with the network side —
  a different argument, on its own merits.
- **(b) raise ε / (c) exact penalty or big-M — REJECT; they are one lever.** ε ≈ 0.46 required;
  offset `mu/(2·s_obj·weight)` for any weight.
- **(e) post-solve correction only — REJECT.** Leaves the in-model `soh_min` floor,
  `es_e_available` and salvage running on the leaky value while reporting a corrected one. Two
  SoH numbers is worse than one wrong one.
- **(d)/(e2) single signed per-cohort power — RANKED FIRST.** Replace the directional pair with
  one signed variable, degradation input a deterministic smooth function of it, e.g.
  `t = eff_ch·(√(p²+δ²)+p)/2 + (1/eff_dch)·(√(p²+δ²)−p)/2`, exact to O(δ²/|p|). No
  complementarity, **no `mu` dependence**, half the variables, idle vertex gone. Sound *because*
  the pair is a pure accounting device — `pch`/`pdch` appear in production only in degradation
  accounting, box limits, the aggregate `pnet` row, the ε term, cohort fixing and
  results/detector; **there is no SoC balance for a net-based definition to leave wrong.** The
  artifact confirms the net is clean while both legs are inflated by the same δ:
  `0.10046 − 0.00046 = 0.09999`.

**Classification, stated honestly:** (d)/(e2) **does change the mathematical formulation**. The
justification is not "it converges better" — it is that **the present definition does not define
a quantity**. That is the strong form and it meets the bar for a formulation change. "The
detector fails" does not.

---

## 7. Decisions required

1. **Authorize or reject (d)/(e2).** It is a formulation change and therefore the author's, not
   the Planner's. Nothing is implemented pending this.
2. **Re-specify G1 as part of the remedy, not after it**, and not against the current reference.
   It needs: a named node or an explicitly aggregate quantity; a tolerance justified by the
   leak's own magnitude rather than 1e-6; and a position on H3 (per-cohort vs aggregate).
3. **The false comment at `shared_energy_storage_data.py:400-412` must be amended whatever is
   decided.** It asserts as established fact the proposition this measurement falsified, and it
   is the stated justification for three deletions. Left in place deliberately, so the decision
   is visible rather than quietly absorbed.
4. **Addendum 2 says "the other seven" broken-historical harnesses; there are nine**, verified
   by name. Five break on reformulation-retired symbols, four on Candidate 2 (capacity `Var`s →
   `Param`s, retired sensitivity channel) — the latter group is probably what the count missed.
   **None of the nine was repaired**, the conservative reading of "repair only the two".
5. Carried forward, unchanged: Candidate 3′ dropped; `convex_oracle.py` marked historical,
   unrepaired; the dead `limited-memory` entries are the author's to remove.

---

## 8. What is NOT established

- **No gate has run.** Nothing in Steps 1, 1a or 1b is gated. G1–G5 would have established that
  the SoH trajectory, EFC and recourse survive the reformulation; none of that is known.
- The C\*-scale materiality figures are extrapolated from the ε fixture, not measured at C\*.
- The ADMM dual-magnitude drift (§4) is a derivation from `s_obj`'s definition, **not a
  measurement**; no ADMM-form ESSO solve was made.
- The leak-free recomputation of `avg_ch_dch` and the leak-free SoH trajectory could **not** be
  computed from the control pickle: `eff_ch`, `eff_dch`, `dt`, `cl_eff` are baked into constraint
  coefficients at construction and are not stored on the model. Recovering them from current code
  was forbidden, since current values may differ from those in force when the artifact was made.
- H3 (multi-cohort split degeneracy) is **reasoned, not measured** — the available fixtures are
  single-cohort.
- No claim that the network-side cycle-21 mechanism is resolved beyond Step 0's single fixture.

---

## 9. Evidence inventory (sha256, first 16)

| artifact | hash |
|---|---|
| `data/SRP1/Results/P5151/eps_sensitivity_check_summary.json` | `888e432b1df2235a` |
| `data/SRP1/Results/P5151/eps_sensitivity_check_full.json` | `806e0dd01b73572e` |
| `…/eps_check_logs/eps1e-3/optim_log_node_7.txt` | `7cf9e25aa9da38bc` |
| `…/eps_check_logs/eps1e-7/optim_log_node_7.txt` | `5c7e7e753c6ff57d` |
| `data/SRP1/Results/P5151/p5151_diagA_old_control_leak.json` | `e488eba586b4b0ed` |
| `data/SRP1/Results/P514N/esso_models_control.pkl` | `071052d19de4d480` |
| `data/SRP1/Results/P514L/rung_1.00.log` | `592204f66ea6e62f` |

Reports: `P5_15_GATE_HOLD_REPORT.md` (detail), `WORKER_REPORT_EPS.md` (ε check),
`WORKER_REPORT_DIAGA.md` (control leak), `BROKEN_HISTORICAL_HARNESSES.md`,
`P5_15_PLANNER_STEP1B_REPORT.md`.

**Gap, recorded rather than papered over:** the harness-repair Worker's report was **not
persisted to disk** — it was returned inline only, and `WORKER_REPORT.md` does not exist. Its
primary evidence survives as the committed diff of `p514_n_instrumented_cstar.py` and
`p514_l_capacity_ladder.py`, plus `data/SRP1/Results/P514L/ladder_s1.json`. The narrative
verification output (build-only `hasattr` checks, the zero-solve guard profile) is **not
recoverable** and would have to be re-run to be re-verified.

## 10. Process incidents recorded this stage

- A verification step **overwrote `data/SRP1/Results/P514L/ladder_s1.json`**, the pre-repair
  result cited as the central finding of `P5_14_L_CAPACITY_LADDER_REPORT.md`. Values survive in
  `rung_1.00.log` and the report; the JSON does not. Cause: a Planner task instruction that said
  "run the harness" without naming an output path. New `CLAUDE.md` rule added; provenance note
  left in the directory.
- **Deleting four retired rule callables broke unpickling of every preserved fixture** (Step 1b).
  Repaired by retaining them unwired. New `CLAUDE.md` rule: deactivate and unwire, never delete.
  Candidate 1's "delete both rows" wording corrected to "unwire".
