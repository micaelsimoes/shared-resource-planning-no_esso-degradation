# Stage P5.1-B — Shared-ESS scaling-window diagnostic (executed)

Diagnostic only. **No production code was changed.**

## Provenance

- Scripts: `p51b_scaling_window_diagnostic.py` (ladder + gate) and
  `p51b_gate_rerun.py` (gate re-run with a corrected result enumerator — the
  first pass computed the authoritative production verdict correctly but
  mis-descended into Pyomo's dict-like `SolverResults`, losing the per-solve
  identities).
- Raw output: `data/SRP1/Results/P51B/p51b_report.json`,
  `data/SRP1/Results/P51B/p51b_gate_cap1000_report.json`.
- Git `f77d829359ff…` (P4.6-B2); tracked working tree clean throughout.
- Scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` —
  identical to P5 and P5.1.
- Candidate from the real production generator
  `_build_positive_bootstrap_candidate`; pre-solve initialization states built
  by replaying the production `create_distribution_networks_models_sequential`
  sequence and stopping immediately before IPOPT. Machinery reused from
  `p51_small_capacity_scaling_diagnostic` rather than duplicated.

## Controls (§4) — verified for all 20 (case, Kmax) branches

Every branch was a fresh clone of the *same* pre-solve state. Before each solve
the branch was verified identical to the reference in: primal starting values,
bounds fingerprint, objective at the starting point, constraint component names,
`sess_snet_def` index tuples, total constraint-data count, row activation state,
`sess_comp` row count, and shared-ESS capacities. **The only differing quantity
was `sess_snet_def_kappa`.** All solves were cold (`from_warm_start=False`) with
identical IPOPT options, MA97 and the exact-Hessian primary path. After every
solve, the `sess_snet_def` `ConstraintData` object ids were confirmed unchanged
— no component was recreated, replaced, reindexed or deactivated. All 20
branches reported `identity: all identical` and
`rows unchanged after solve = True`.

---

## 1. Case 4 baseline and regression, in full (§2)

`case33_3` · node 9 · **2025 Autumn** — `S_rated = 0.010635 MVA = 1.0635e-04 p.u.`,
production `kappa = 9403.01`.

| | **Untouched production** (`1/S_rated`) | **Cap 100** |
|---|---|---|
| Status / termination | `ok` / **`optimal`** | `warning` / **`maxIterations`** |
| IPOPT exit | Optimal Solution Found | Maximum Number of Iterations Exceeded |
| Iterations | **438** | **3000** |
| Objective | −8.2604212445e-04 | **1.8159674197e+02** |
| Primal infeasibility | 5.6497e-14 | 5.0792e-03 |
| Dual infeasibility | 2.7389e-06 | **1.2314e+03** |
| Complementarity | 9.0909e-09 | **1.0000e-01** |
| Overall NLP error | 2.7389e-06 | **9.8184e+01** |
| Runtime | 6.9 s | 17.6 s |
| `sch` range | [3.341e-06, 4.121e-06] | **[0, 0]** |
| `sdch` range | [3.109e-06, 3.845e-06] | **[0, 0]** |
| `pnet` range | [−5.042e-07, 1.012e-06] | **[0, 0]** |
| `qnet` range | [−7.411e-13, 1.056e-13] | **[0, 0]** |
| max ｜g｜ | 4.951e-23 | 0.000e+00 (variables never moved) |
| max ｜g｜ / `S_rated²` | **4.377e-15** | 0.000e+00 |

The cap-100 endpoint is the same pathology seen in the P5.1 production failures,
only mirrored: 3000 iterations, dual infeasibility ~1.2e+03, complementarity
pinned at 1.0e-01, and every shared-ESS variable still exactly zero. The
`max|g| = 0` entry is an artefact of the solver never leaving its starting
point, not evidence of feasibility.

---

## 2. Scaling ladder — full metrics (§5)

`kappa = min(1/S_rated, Kmax)`; the `production` column is untouched
`1/S_rated`. Recovery was not attempted anywhere except one cell, flagged below.

### case33_1 · node 5 · 2030 Winter — `S = 0.021270 MVA = 2.1270e-04 p.u.`, production κ = 4701.50

| Kmax | κ | 2κ | Result | Iters | Objective | Primal inf | Dual inf | Compl | NLP err | Runtime | max｜g｜/S² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 200 | **optimal** | 240 | −8.0130e-04 | 4.12e-14 | 8.15e-06 | 9.09e-09 | 8.15e-06 | 2.5 s | 7.235e-11 |
| 300 | 300 | 600 | **maxIterations** | 3000 | 2.5195e+01 | 2.39e-07 | 3.24e+01 | 2.83e-03 | 4.98e+00 | 28.5 s | — (vars at 0) |
| 1000 | 1000 | 2000 | **optimal** | 334 | −8.0130e-04 | 6.25e-14 | 6.00e-08 | 9.09e-09 | 6.00e-08 | 2.7 s | 4.182e-14 |
| 3000 | 3000 | 6000 | **internalSolverError** | 408 | 1.6939e+02 | 1.35e-03 | 1.00e+03 | 3.81e-01 | 7.12e+02 | 3.9 s | — (vars at 0) |
| production | 4701.50 | 9403.01 | **maxIterations** | 3000 | 1.6680e+02 | 8.29e-03 | 1.16e+06 | 1.00e-01 | 3.55e+04 | 19.2 s | — (vars at 0) |

### case33_1 · node 5 · 2035 Winter — `S = 0.031905 MVA = 3.1905e-04 p.u.`, production κ = 3134.34

| Kmax | κ | 2κ | Result | Iters | Objective | Primal inf | Dual inf | Compl | NLP err | Runtime | max｜g｜/S² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 200 | **optimal** | 434 | −8.0863e-04 | 5.02e-14 | 3.03e-06 | 9.09e-09 | 3.03e-06 | 3.5 s | 2.421e-11 |
| 300 | 300 | 600 | **optimal** | 178 | −8.0863e-04 | 5.06e-14 | 2.07e-06 | 9.09e-09 | 2.07e-06 | 2.1 s | 2.443e-10 |
| 1000 | 1000 | 2000 | **optimal** | 545 | −8.0863e-04 | 1.71e-12 | 7.21e-06 | 9.09e-09 | 7.21e-06 | 7.0 s | 1.681e-08 |
| 3000 | 3000 | 6000 | **optimal** | 469 | −8.0863e-04 | 1.15e-11 | 8.83e-06 | 9.09e-09 | 8.83e-06 | 3.9 s | 3.774e-08 |
| production | 3134.34 | 6268.67 | **maxIterations** | 3000 | 1.9685e+02 | 1.25e-01 | 1.31e+06 | 1.25e-01 | 6.76e+04 | 18.6 s | — (vars at 0) |

### case33_3 · node 9 · 2025 Summer — `S = 0.010635 MVA = 1.0635e-04 p.u.`, production κ = 9403.01

| Kmax | κ | 2κ | Result | Iters | Objective | Primal inf | Dual inf | Compl | NLP err | Runtime | max｜g｜/S² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 200 | **optimal** (Solved To Acceptable Level) | 154 | −8.1520e-04 | 4.99e-14 | 3.35e-05 | 9.09e-09 | 3.35e-05 | 1.7 s | 1.708e-12 |
| 300 | 300 | 600 | **optimal** — ⚠ **recovery used** | 165 | −8.4532e-04 | 5.01e-14 | 9.70e-06 | 6.09e-11 | 9.70e-06 | 3.5 s | 9.114e-12 |
| 1000 | 1000 | 2000 | **optimal** | 385 | −8.1521e-04 | 6.16e-14 | 3.98e-06 | 9.09e-09 | 3.98e-06 | 4.3 s | 1.008e-11 |
| 3000 | 3000 | 6000 | **optimal** | 646 | −8.1521e-04 | 3.44e-13 | 3.91e-06 | 9.09e-09 | 3.91e-06 | 6.1 s | 1.414e-11 |
| production | 9403.01 | 18806.02 | **maxIterations** | 3000 | 1.7579e+02 | 1.40e-02 | 2.68e+03 | 1.00e-01 | 2.15e+02 | 23.9 s | — (vars at 0) |

⚠ The `Kmax = 300` cell is the **only** branch in the entire ladder where
recovery was attempted (and succeeded). It is therefore *not* a clean primary
success, and it converged to a materially different point: `sch ≈ 7.9e-09`
versus `≈ 3.3e-06` in every other successful branch, with objective
−8.4532e-04 versus −8.152e-04. Reported rather than smoothed over.

### case33_3 · node 9 · 2025 Autumn — `S = 0.010635 MVA = 1.0635e-04 p.u.`, production κ = 9403.01

| Kmax | κ | 2κ | Result | Iters | Objective | Primal inf | Dual inf | Compl | NLP err | Runtime | max｜g｜/S² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 200 | **maxIterations** | 3000 | 1.8160e+02 | 5.08e-03 | 1.23e+03 | 1.00e-01 | 9.82e+01 | 17.6 s | — (vars at 0) |
| 300 | 300 | 600 | **optimal** | 154 | −8.2604e-04 | 5.33e-14 | 2.75e-06 | 9.09e-09 | 2.75e-06 | 1.6 s | 4.368e-15 |
| 1000 | 1000 | 2000 | **optimal** | 378 | −8.2604e-04 | 6.63e-14 | 2.74e-06 | 9.09e-09 | 2.74e-06 | 4.2 s | 4.377e-15 |
| 3000 | 3000 | 6000 | **optimal** | 154 | −8.2604e-04 | 4.34e-14 | 2.75e-06 | 9.09e-09 | 2.75e-06 | 2.2 s | 4.368e-15 |
| production | 9403.01 | 18806.02 | **optimal** | 438 | −8.2604e-04 | 5.65e-14 | 2.74e-06 | 9.09e-09 | 2.74e-06 | 6.9 s | 4.377e-15 |

In every successful branch across all four cases the **original unscaled
physical equality** holds to `|g|/S_rated² ≤ 3.8e-08`, far inside
`EQUALITY_TOLERANCE = 1e-5`. Scaling never relaxes the physics.

---

## 3. Success/failure matrix (§5)

Cell = result / IPOPT iterations.

| Case | 100 | 300 | 1000 | 3000 | production |
|---|---|---|---|---|---|
| n5 2030 Winter | **OK** / 240 | FAIL / 3000 | **OK** / 334 | FAIL / 408 | FAIL / 3000 |
| n5 2035 Winter | **OK** / 434 | **OK** / 178 | **OK** / 545 | **OK** / 469 | FAIL / 3000 |
| n9 2025 Summer | **OK** / 154 | **OK** / 165 ⚠ | **OK** / 385 | **OK** / 646 | FAIL / 3000 |
| n9 2025 Autumn | FAIL / 3000 | **OK** / 154 | **OK** / 378 | **OK** / 154 | **OK** / 438 |

**Success is not monotone in `Kmax`.** `n5 2030 Winter` succeeds at 100, fails
at 300, succeeds at 1000, then fails at 3000 and at production — and its 3000
failure is a *different mode* (`internalSolverError` after 408 iterations)
than its 300 and production failures (`maxIterations` at 3000). This is the
single most important structural observation in the stage: the four cases do not
share a well-behaved monotone scaling window; `Kmax = 1000` happens to be the
one tested value where all four land on the success side.

## 4. Common-window decision (§6)

| | |
|---|---|
| Tested values succeeding for **all four** cases | **`1000` only** |
| Selected diagnostic candidate | **`Kmax = 1000`** |

Only one tested value is common-successful, so the "largest common-successful"
rule selects it trivially. No success was inferred at any untested intermediate
value, and no additional value was tried.

## 5. Full initialization gate (§7) — the decisive negative result

The complete P5 iteration-2 `positive_bootstrap` operational initialization was
re-run with `kappa = min(1/S_rated, 1000)` applied **in memory only** (the
production `shared_ess_snet_def_scale` was temporarily wrapped inside the
diagnostic process and restored afterwards). The real production initialization
path was executed. ADMM and the outer planning loop were **not** entered.

| | |
|---|---|
| Total local solves | **51** |
| DSO solves | 36 — **4 failures** |
| TSO solves | 12 — 0 failures |
| ESSO solves | 3 — 0 failures |
| `_admm_local_solves_succeeded` | **False** |
| **Would the system enter ADMM?** | **No** |

Exact failure identities under `Kmax = 1000`:

| Solve | Status / termination |
|---|---|
| `dso / 5 / 2030 / Spring` | `error` / **`internalSolverError`** |
| `dso / 5 / 2035 / Autumn` | `error` / **`internalSolverError`** |
| `dso / 7 / 2030 / Summer` | `warning` / **`maxIterations`** |
| `dso / 7 / 2035 / Spring` | `warning` / **`maxIterations`** |

**None of these four is one of the three original P5 failures, nor the cap-100
regression case.** The failure set has moved again — and now includes node 7,
which was untouched in both P5 and the P5.1 cap-100 run.

Per §7, a new failure appeared, so the investigation **stops here**. No other
cap value was tried.

---

## 6. Interpretation (§10) — kept narrow

1. **Established.** The four decisive cases each have a numerically usable
   scaling range, and `Kmax = 1000` is a tested value inside all four
   simultaneously. Row scaling demonstrably governs whether these particular
   cold initialization solves converge, and the physical equality is preserved
   to ≤ 3.8e-08 normalized in every success.
2. **Established, and it cuts the other way.** A single scalar upper bound on
   `1/S_rated` does **not** make the bootstrap initialization solvable. At
   `Kmax = 1000` the initialization still fails — with **four** failures, at
   different networks/years/days than before, including a network (node 7) that
   had not failed under either previously tested scale. Scalar capping
   *relocates* the failures rather than removing them.
3. **Reinforcing evidence.** Success is non-monotone in `Kmax` (n5 2030 Winter:
   OK, FAIL, OK, FAIL, FAIL), and the failure mode itself varies
   (`maxIterations` vs `internalSolverError`). Both are inconsistent with a
   clean, well-defined scaling window and consistent with path-dependent
   behaviour of a nonconvex cold-start problem in which row scale is one
   influential factor among several.
4. **Not established.** That row scaling is the only relevant factor; that any
   untested `Kmax` would succeed; or that a cap is productionizable. This stage
   authorizes none of that.

Per §10, the next planner decision is between a more fundamental
capacity-normalized shared-ESS formulation and separately investigating
initialization/recovery behaviour. **Neither was started.**

## 7. Scope compliance (§9)

No production change of any kind. `shared_ess_snet_def_scale` and every accepted
P4 formulation are untouched; no cap was committed to `definitions.py`,
`network.py` or `model_construction_helpers.py`; the `maxIterations` recovery
policy was not modified; no ADMM cycle or outer planning iteration was run; MA57
was not used and no IPOPT tolerance was altered. The only changed quantity in
any branch was `model.sess_snet_def_kappa` on cloned in-memory models.

---

```
P5.1-B PARTIAL — targeted window found but full initialization still fails — waiting for planner review
```
