# Stage P5.1 — Small-capacity shared-ESS scaling diagnostic (executed)

Diagnostic only. **No production code was changed.**

## Provenance

- Script: `p51_small_capacity_scaling_diagnostic.py`. Raw output:
  `data/SRP1/Results/P51/p51_report.json`.
- Git `f77d829359ff…` (P4.6-B2), tracked working tree clean at run time.
- Scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` —
  identical to P5 and to the `REVISION_CONTEXT.md` reproducibility identity.
- The bootstrap candidate came from the **real production generator**
  `_build_positive_bootstrap_candidate(planning, params.benders.positive_bootstrap)`,
  never reconstructed by hand. It reproduces the P5 iteration-2 candidate
  exactly (total capacity, identical at nodes 5/7/9):

  | Year | S [MVA] | E [MVAh] | S [p.u.] | production `kappa_A = 1/S_rated` |
  |---|---|---|---|---|
  | 2025 | 0.010635 | 0.021270 | 1.0635e-04 | **9403.0** |
  | 2030 | 0.021270 | 0.042540 | 2.1270e-04 | **4701.5** |
  | 2035 | 0.031905 | 0.063810 | 3.1905e-04 | **3134.3** |

  (P5's console rounded these to 0.011 / 0.021 / 0.032 MVA.)

## Method

Each DSO initialization model was built by replaying the production pre-solve
sequence of `create_distribution_networks_models_sequential` — same calls, same
order (`update_data_with_candidate_solution` → `build_model` →
`update_model_with_candidate_solution` → expected-interface vars/constraints →
`configure_shared_ess_operational_state` → `_add_dso_scenario_deviation_penalty`)
— stopping **immediately before IPOPT**. That pre-solve state was preserved and
`.clone()`d twice per case:

- **A** — production scaling, `kappa_A = 1/S_rated`, untouched.
- **B100** — the **only** change is the existing mutable shared-ESS scale
  parameter `model.sess_snet_def_kappa[idx]` set to `min(1/S_rated, 100)`.

Both branches solved through the production path
`Network.run_smopf(..., from_warm_start=False)` (cold, as in the real
initialization). No component was created, deactivated, reindexed or
re-expressed; no variable, bound, objective, SOC equation, `sess_comp`, ADMM
quantity or solver option was touched.

### Pre-solve equivalence (§4) — verified for all six cases

For every case, A and B were confirmed identical in: primal starting values
(variable-data count and value fingerprint), variable bounds, objective value at
the starting point, constraint component names, `sess_snet_def` index tuples,
total constraint-data count, row activation state, `sess_comp` row count,
shared-ESS capacities (`s_rated_fixed`, `e_rated_fixed`), and presence of the
ordinary-ESS scale. **The only difference was `sess_snet_def_kappa`.** Both
branches were cold (`from_warm_start=False`); no multiplier suffix was
introduced, cleared or transformed in either branch.

### Structural verification (§7) — verified for all six cases

After solving: the `sess_snet_def` component object is unchanged in both
branches, every `ConstraintData` object id for the shared-ESS index is unchanged
under B, no replacement component exists, constraint counts are unchanged, row
activation (zero-capacity gating) is untouched, and the ordinary-ESS
normalization (`ess_snet_def_scale`) is present and unmodified in both.

---

## 1. The three P5 failures (§5)

All three are the same story: under production scaling IPOPT burns the full
3000 iterations without ever moving the shared-ESS variables off zero.

### case33_1 · node 5 · 2030 · Winter — `S_rated = 0.021270 MVA = 2.1270e-04 p.u.`

| | **A** `kappa_A = 4701.50` (`2κ = 9403.01`) | **B100** `kappa_B = 100.00` (`2κ = 200.00`) |
|---|---|---|
| Status / termination | `warning` / **`maxIterations`** | `ok` / **`optimal`** |
| IPOPT exit | Maximum Number of Iterations Exceeded | **Optimal Solution Found** |
| Iterations | **3000** | **240** |
| Objective | 1.66799e+02 | −8.01298e-04 |
| Primal infeasibility | 8.2866e-03 | **4.1197e-14** |
| Dual infeasibility | **1.1613e+06** | **8.1476e-06** |
| Complementarity | 1.0000e-01 | 9.0909e-09 |
| Overall NLP error | **3.5483e+04** | **8.1476e-06** |
| Runtime | 19.8 s | 1.9 s |
| `sch` / `sdch` range | [0, 0] / [0, 0] | [3.48e-06, 3.98e-06] / [3.25e-06, 3.72e-06] |
| `pnet` / `qnet` range | [0, 0] / [0, 0] | [−2.4e-07, 7.3e-07] / [−0.0, 0.0] |
| max ｜g｜ (original, unscaled) | 0.000e+00 (variables never left zero) | 3.273e-18 |
| max ｜g｜ / `S_rated²` | 0.000e+00 | **7.235e-11** |

### case33_1 · node 5 · 2035 · Winter — `S_rated = 0.031905 MVA = 3.1905e-04 p.u.`

| | **A** `kappa_A = 3134.34` (`2κ = 6268.67`) | **B100** `kappa_B = 100.00` |
|---|---|---|
| Status / termination | `warning` / **`maxIterations`** | `ok` / **`optimal`** |
| IPOPT exit | Maximum Number of Iterations Exceeded | **Optimal Solution Found** |
| Iterations | **3000** | **434** |
| Objective | 1.96853e+02 | −8.08633e-04 |
| Primal infeasibility | 1.2454e-01 | **5.0204e-14** |
| Dual infeasibility | **1.3080e+06** | **3.0256e-06** |
| Complementarity | 1.2454e-01 | 9.0909e-09 |
| Overall NLP error | **6.7564e+04** | **3.0256e-06** |
| Runtime | 20.1 s | 3.6 s |
| `sch` / `sdch` range | [0, 0] / [0, 0] | [3.56e-06, 3.89e-06] / [3.32e-06, 3.63e-06] |
| `pnet` / `qnet` range | [0, 0] / [0, 0] | [−7e-08, 5.8e-07] / [−1e-08, 0.0] |
| max ｜g｜ / `S_rated²` | 0.000e+00 | **2.421e-11** |

### case33_3 · node 9 · 2025 · Summer — `S_rated = 0.010635 MVA = 1.0635e-04 p.u.`

| | **A** `kappa_A = 9403.01` (`2κ = 18806.02`) | **B100** `kappa_B = 100.00` |
|---|---|---|
| Status / termination | `warning` / **`maxIterations`** | `ok` / **`optimal`** |
| IPOPT exit | Maximum Number of Iterations Exceeded | Solved To Acceptable Level |
| Iterations | **3000** | **154** |
| Objective | 1.75793e+02 | −8.15204e-04 |
| Primal infeasibility | 1.3987e-02 | **4.9887e-14** |
| Dual infeasibility | **2.6793e+03** | **3.3515e-05** |
| Complementarity | 1.0000e-01 | 9.0909e-09 |
| Overall NLP error | **2.1539e+02** | **3.3515e-05** |
| Runtime | 24.8 s | 1.7 s |
| `sch` / `sdch` range | [0, 0] / [0, 0] | [3.08e-06, 4.48e-06] / [2.86e-06, 4.19e-06] |
| `pnet` / `qnet` range | [0, 0] / [0, 0] | [−1.11e-06, 1.61e-06] / [−0.0, −0.0] |
| max ｜g｜ / `S_rated²` | 0.000e+00 | **1.708e-12** |

**Characterization of the `maxIterations` baseline (§5).** Parsing the IPOPT
logs rather than reading only the termination code shows these are not
near-misses. In all three, dual infeasibility ends at `2.7e+03`–`1.3e+06`,
complementarity is pinned at `1.0e-01`, the overall NLP error is
`2.2e+02`–`6.8e+04`, and every shared-ESS variable is **still exactly zero**
after 3000 iterations. The solver never left the neighbourhood of its starting
point. The `max|g| = 0` entries for branch A are therefore an artefact of that
— not evidence of physical feasibility.

Under B100 all three converge in 154–434 iterations with primal infeasibility
`~5e-14`, and the **original unscaled** physical equality holds to
`|g|/S_rated² ≤ 7.2e-11` — far inside `EQUALITY_TOLERANCE = 1e-5`. The physical
relation is preserved, not relaxed.

## 2. Matched successful controls (§6)

All three controls were first confirmed **successful under A**, then re-run
under B100. None regressed; all three improved in iteration count.

| Control | `S_rated` [p.u.] | `kappa_A` | A | B100 |
|---|---|---|---|---|
| case33_1 · n5 · 2025 · Winter | 1.0635e-04 | 9403.0 | optimal, **122** iters, dual inf 7.33e-08 | optimal, **76** iters, dual inf 7.44e-06 |
| case33_3 · n9 · 2030 · Summer | 2.1270e-04 | 4701.5 | optimal, **488** iters, dual inf 1.60e-06 | optimal, **370** iters, dual inf 2.84e-06 |
| case33_2 · n7 · 2025 · Winter | 1.0635e-04 | 9403.0 | optimal, **1058** iters, dual inf 4.94e-07 | optimal, **196** iters, dual inf 3.00e-07 |

Objectives agree to ~9 significant figures between A and B in every control
(e.g. −7.9644313097e-04 vs −7.9644313310e-04), and the shared-ESS dispatch
ranges are identical — as expected from a constant positive rescaling of one
equality row. The original unscaled residual stays at `|g|/S_rated² ≤ 3.7e-11`
in every control under both branches.

## 3. Decision gate (§8)

| Gate condition | Result |
|---|---|
| All three failed cases fail under A | **True** |
| All three failed cases succeed under B100 | **True** |
| All controls succeed under A | **True** |
| All controls still succeed under B100 | **True** |
| → proceed to integration check | **Yes** |

The §8 STOP condition ("if any of the three failed cases still fails under
B100") did **not** trigger, so the integration check was run — and nothing
else was tried: no other cap, no `kappa = 1`, no `sess_comp`, MA57, solver
tolerance, recovery-policy or bootstrap-size variation.

### Integration check — the decisive negative result

The complete iteration-2 `positive_bootstrap` initialization was re-run with the
B100 cap applied **in memory only** (the production
`shared_ess_snet_def_scale` function was temporarily wrapped inside the
diagnostic process and restored afterwards; the production file was not
modified). ADMM and the outer loop were **not** entered.

> **Required question — do all local initialization solves complete so the model
> could enter ADMM?**
>
> **No.**

| | |
|---|---|
| DSO initialization solves | 36 |
| Optimal under the cap | **35** |
| Non-optimal under the cap | **1** — `dso_9_2025_Autumn` → `maxIterations` |
| `_admm_local_solves_succeeded` | **False** |
| Would enter ADMM | **No** |

All three originally-failing cases are cleared, **but a different case that
succeeded under production scaling — `case33_3` · node 9 · 2025 · **Autumn** —
fails under the cap.** It was not among P5's three failures, so under
`kappa_A` it solved successfully. It was also not in the control set, which is
a limitation of the control selection: three matched controls were not enough to
detect this regression.

## 4. Interpretation (§11)

Held deliberately cautious:

1. **Established.** For the three P5 initialization failures, the very large
   production row scaling (`kappa_A` between 3.1e+03 and 9.4e+03, i.e. row
   second-derivative coefficients `2κ` from 6.3e+03 to 1.9e+04) is a
   **proximate numerical cause**. Capping it to 100 turns three hard
   3000-iteration stalls — with dual infeasibility up to 1.3e+06 — into clean
   convergence in 154–434 iterations, with the original physical equality
   preserved to ~1e-11, and with every structural invariant and every non-kappa
   input held identical.
2. **Not established.** That this is the *only* relevant source of
   small-capacity conditioning trouble. The integration check directly
   contradicts any such reading: applying the same cap across the whole
   initialization moved the failure elsewhere rather than removing it. A single
   global cap of 100 is therefore **not** sufficient to make this bootstrap
   candidate initializable.
3. **Also observed, not investigated.** `case33_1`'s params define **no**
   `recovery_options` at all (unlike `case33_2`/`case33_3`). This is a
   configuration asymmetry, not a cause of the above — recovery would not have
   triggered regardless, since the production recovery path applies only to
   `internalSolverError`. Recorded per §10; not acted on.

**Overall reading:** the small-capacity regime does expose the accepted
`1/S_rated` scaling far outside its P4.3–P4.5 validated range (κ = 100–50
validated vs κ ≈ 3.1e+03–9.4e+03 here), and capping demonstrably fixes the
specific diagnosed states — but the capped scale is not itself a complete
remedy at initialization scale. That mixture is why this stage is not a PASS.

## 5. Scope compliance (§9, §10)

No production scaling change was made. Production was **not** changed to
`min(1/S_rated, 100)` — that remains a separately approved P5.2 decision. The
`maxIterations` recovery policy was not touched. No ADMM cycle and no outer
planning iteration was run. Tracked working tree was clean before and after the
production-code sense (only the new diagnostic script, report and evidence JSON
are added).

---

```
P5.1 PARTIAL — capped scaling evidence is mixed — waiting for planner review
```
