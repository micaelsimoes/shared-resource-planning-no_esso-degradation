# P5.15 — Handoff to the external expert, round 2

**Subject: remedy (h) works and the barrier identity is now predictive. H3 is implemented and
inert on every current fixture. Gate G5 FAILS by its predeclared rule — but not because of a
nonconvexity. G1–G4 and the `REVISION_CONTEXT.md` rewrite are held pending two gate
re-specifications.**

Planner report, 2026-09-13. Continues `P5_15_EXPERT_HANDOFF.md`. Authority:
`PLANNER_BRIEF_2026-09-13.md`, Addendum 3.

---

## 1. Status against the Addendum-3 dispatch

| step | state |
|---|---|
| commit held reports and artifacts | done — `8af3e242` |
| remedy (h): ESSO `tol=1e-8`, detector + estimate logged per solve | **done, verified** |
| item 4: amend the false comment at `shared_energy_storage_data.py:400-412` | **done, then corrected again by the Planner — see §3** |
| H3 pro-rata cohort-split rows | **done, verified, inert on all current fixtures** |
| **G5** | **FAIL by its predeclared rule; cause identified; not a nonconvexity** |
| G1–G4 | **NOT RUN** |
| `REVISION_CONTEXT.md` rewrite | **not started** |

---

## 2. Remedy (h): the identity is now predictive, not merely fitted

| | control `tol=1e-6` | remedy (h) `tol=1e-8` |
|---|---|---|
| `mu_final` | 9.2306e-08 | 9.0961e-10 |
| detector `max min(pch,pdch)/s_max` | 4.6319e-04 | **4.5351e-06** |
| spurious throughput | **0.9175 %** | **0.0091 %** |
| iterations, node 7 | 16 | **17** |
| termination | optimal | optimal, zero recovery |

`mu_final` fell **101.5×** and the leak fell **102.1×** — agreement to **0.65 %**. The original
0.10 % figure was a *fit*; this is the identity **predicting a result in advance across two
decades of μ**, which is far stronger evidence. The cost is **one iteration**.

Your correction was right and the omission was mine: §6 of the first handoff listed ε and
`s_obj` as the levers and treated `mu_final` as fixed, when `tol` controls it directly. The
Advisor had flagged the `tol` variant as the primary discriminator and I failed to carry it into
the remedy list.

---

## 3. One correction the Planner made to the authorized wording

Addendum 3 directs that the quantity `2·N·mu_final/(2·s_obj·ε)` be reported as a **closed-form
bound** and stated in the manuscript. **The measurement does not support "bound".**

| arm | measured / estimate |
|---|---|
| `tol=1e-6` | **1.00323** (exceeds) |
| `tol=1e-8` | **0.99716** (undershoots) |

It is an **unbiased estimator with roughly ±0.35 % scatter**, exceeded in one direction and
undershot in the other — not an upper bound. Interpretation, flagged in code as Planner
inference and *not* verified: IPOPT's reported `Complementarity` is an aggregate over all bound
pairs, not the exact barrier parameter governing this one.

The Worker's amended comment had faithfully written "CLOSED-FORM UPPER BOUND", reproducing the
exact error class the amendment existed to fix. **I corrected it in place** to "ESTIMATE, NOT AN
UPPER BOUND", with both ratios recorded and an instruction not to describe it as a bound in the
manuscript nor use it to certify a leak ceiling — the ratio detector is the direct measurement
and should carry that role.

**Decision needed: the manuscript should say *estimate* (±0.35 % demonstrated), not *bound*.**

---

## 4. H3 — implemented, correct, and inert

Implemented as `N_active − 1` rows tied to the existing `agg_pnet` expression, with the share a
Python float from the `es_e_investment_fixed` **Param** (not a `Var` ratio, which would have made
the row nonlinear).

A hazard I flagged before implementation and which the Worker respected: read literally the rule
is one row per cohort, but the shares sum to 1 and `agg_pnet` is *defined* as the sum of the same
per-cohort powers, so **N rows would be linearly dependent and would reintroduce an LICQ
violation** — the defect class this programme exists to remove. `N−1` avoids it; a numerical rank
check confirms independence against the aggregate row (rank 2).

Verification: single-cohort regression **bit-identical, max abs diff 0.0** on objective, `pnet`,
`pch`/`pdch`, SoH and `D` at all three nodes, with zero active H3 rows. Multi-cohort
construction: exactly `N−1` active rows, all-float coefficients, shares summing to 1. One
authorized multi-cohort solve: the pro-rata identity holds to **1.39e-17**.

**Every current fixture, including C\*, has a single active cohort, so H3 is inert and G1–G5
will not test it.** It is a correctness fix for future multi-cohort instances and a manuscript
claim (homogeneous-fleet approximation, answering §2.4 / R3.3), not something the gates confirm.

On the Worker's open question — "zero rows" in the **active** sense (0/864), not the structural
sense. ESSO models are cloned and re-updated across ADMM cycles rather than rebuilt, so a
dynamically-rebuilt design would risk cross-cycle row accumulation over 90 cycles; build-once /
activate-per-candidate is already the file's convention. The Worker's Arm B, which physically
removed the components, was bit-identical to Arm A — empirical proof the structural entries are
inert.

---

## 5. Gate G5 — FAIL, and what it actually means

A1 (warm), A3 (warm + `mu_strategy=adaptive`), A4 (cold), on the reformulated ESSO, nodes 7
and 9. All six solves optimal, zero `maxIterations`, zero recovery, rule-ten ratios ~4.5 % of
threshold — well settled.

`|A1−A4|/|A1| = 5.37e-12`. `|A1−A3|/|A1| = 2.86e-04`, i.e. **286× the 1e-6 gate.**

### The decomposition that settles it (node 7; node 9 identical)

| variable | A1 vs **A3** (`mu_strategy`) | A1 vs **A4** (warm vs cold) |
|---|---|---|
| `es_pnet` (control; hard-fixed) | 0.000e+00 | 0.000e+00 |
| **net power `pch − pdch`** | **4.163e-16** | 2.776e-16 |
| `es_avg_ch_dch_per_unit` | **7.633e-05** | 5.655e-12 |
| `es_D_per_unit` | **7.633e-05** | 5.655e-12 |
| `es_soh_per_unit_cumul` | **2.182e-05** | 1.993e-12 |

**All three arms reach the same dispatch decision to machine precision.** The disagreement is
confined to the derived degradation chain, for a structural reason: net power is **subtractive**,
so the leak cancels exactly; throughput is **additive**, so it accumulates at `2·min(pch,pdch)`.
The degradation law is fed the additive quantity.

**A1 versus A4 — warm against cold start, the most different initial points available — agree to
1e-12 on every quantity.** That is where multiple local optima would appear first. They do not.

### Conclusions

1. **No nonconvexity is demonstrated.** Same decision, from maximally different starts.
2. **The degradation chain remains solver-dependent at 7.6e-05** across `mu_strategy` — ~100×
   better than before remedy (h), not zero. (h) shrinks the artifact; it does not remove it,
   exactly as its own rationale said.
3. Under the **production** barrier strategy (A1, A4) the reported SoH is reproducible to 1e-12.
   A3 is not a production setting.

### Why the gate cannot pass as written

`model.objective` on this fixture is **exactly** `PENALTY_ESSO_SLACK·Σ(slack_up+slack_down) +
eps·Σ(pch+pdch)` — confirmed to **8.4e-17** by substituting measured slacks — and nothing else,
because `es_pnet` is hard-fixed. Both terms are barrier artifacts, and A3 differs precisely in
the parameter that sets the barrier path. **G5 as specified requires two artifacts to agree to
1e-6.** Same defect as G1, on a different quantity.

---

## 6. A Planner prediction that was falsified, and is withdrawn

I predeclared that the slack pair would show its own barrier residual,
`min(slack_up, slack_down) = mu_final/(2·s_obj·PENALTY_ESSO_SLACK)`, predicting 4.548e-12 (A1)
and 9.352e-12 (A3).

**Falsified.** Measured slacks are **negative, ≈ −1e-8, essentially constant across all 288
periods** — an IPOPT bound-relaxation floor (`bound_relax_factor = 1e-8`), not a μ-scaled
residual. Measured/predicted ≈ −2197 (A1), −1067 (A3).

The supporting arithmetic I circulated ("predicted 2.767e-6 against an observed residual of
4.4035e-6, factor 1.59, right order and right mechanism") rested on a formula that does not apply
to this pair. It was labelled supported-not-established at the time; it is now **withdrawn**.

---

## 7. Decisions required

1. **Re-specify G5.** Gate on quantities the model determines:
   (i) dispatch invariance across A1/A3/A4 — *currently passes at 4.2e-16*;
   (ii) reproducibility under the production barrier strategy, A1 vs A4 — *currently passes at
   5.7e-12*;
   (iii) the cross-`mu_strategy` spread on the degradation chain (**7.6e-05**) declared as a
   stated SoH uncertainty rather than required to vanish.
   Retain the 1e-6-on-objective form only if the objective is first purged of artifact terms,
   which no remedy on the table does.
2. **G1's re-specification (Addendum 3 item 3) is accepted but untested** — it is a per-node
   reconciliation gate, and the "new" leak fraction it reconciles against is now `tol=1e-8`'s
   0.0091 %, not the 0.9175 % measured when the gate was written. Confirm the reconciliation
   should use the run's own detector, as Addendum 3 says, and note the predicted Δ is now ~100×
   smaller, which may put it below the resolution of the 10 % agreement criterion.
3. **Manuscript wording: "estimate", not "bound"** (§3).
4. The Addendum-3 fallback trigger for (d)/(e2) — detector exceeding 1e-4 relative throughput at
   C\* — measures **9.1e-05 on the ε fixture**, below threshold, **but has not been evaluated at
   C\*** and cannot be until G1 runs.

---

## 8. What is NOT established

- **G1–G4 have not run.** Nothing about the SoH trajectory, EFC, recourse or ADMM convergence at
  C\* under the reformulated model is known.
- The fallback trigger has not been evaluated at C\*.
- H3 is verified structurally and on one synthetic multi-cohort solve; **no production instance
  exercises it.**
- The mechanism by which the residual leak propagates into SoH is inferred from the
  additive/subtractive structure and is consistent with the measurements, but was **not verified
  against raw per-period `pch`/`pdch`**, which this run did not capture.
- The ±0.35 % estimator interpretation (§3) is Planner inference, not verified.

---

## 9. Evidence inventory (sha256, first 16; paths under `data/SRP1/Results/`)

| artifact | hash |
|---|---|
| `P5151/tol_remedy_check_summary.json` | `bde91d9762f52415` |
| `P5151/h3_single_cohort_regression_summary.json` | `b3ee8020989f654a` |
| `P5151/h3_multicohort_construction_test_summary.json` | `b36cf692f4ab36a1` |
| `P5151/h3_multicohort_solve_check_summary.json` | `de068237f7b7019d` |
| `P5151/g5_gate_summary.json` | `4e522921c2cd129d` |
| `P5151/g5b_physical_variable_capture_summary.json` | `f5156a4130c9c0f8` |

Reports: `P5_15_G5_REPORT.md`, `WORKER_REPORT_H.md`, `WORKER_REPORT_H3.md`,
`WORKER_REPORT_G5.md`, `WORKER_REPORT_G5B.md`. Prior round: `P5_15_EXPERT_HANDOFF.md`,
`P5_15_GATE_HOLD_REPORT.md` (committed at `8af3e242`).

## 10. Process note

The original G5 task required the objective, detector, `mu_final` and throughput — but **not**
the solution vector or the slacks, the quantities that discriminate "artifact" from
"nonconvexity". The gate therefore returned an uninterpretable FAIL and cost a second six-solve
run. That is the rule-eleven failure mode again, and the spec was mine. **Proposed extension:
rule eleven should require capturing not only what the spec needs, but what would distinguish the
competing readings of a failure.**
