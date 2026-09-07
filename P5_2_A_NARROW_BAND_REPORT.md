# Stage P5.2-A — Narrow-band relaxation of shared-ESS `sess_snet_def` (executed)

Diagnostic only. **No production code was changed.**

## Provenance

- Script: `p52a_narrow_band_diagnostic.py`. Raw output:
  `data/SRP1/Results/P52A/p52a_report.json`.
- Git `f77d829359ff…` (P4.6-B2); tracked working tree clean throughout.
- Scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` —
  identical to P5, P5.1 and P5.1-B.
- Candidate from the real production `_build_positive_bootstrap_candidate`;
  pre-solve initialization states built by replaying the production
  `create_distribution_networks_models_sequential` sequence and stopping
  immediately before IPOPT. Machinery reused from
  `p51_small_capacity_scaling_diagnostic`.

## Design

Both branches use the **untouched production `kappa = 1/S_rated`** (no cap).
The only mathematical difference is the constraint form:

| | Form |
|---|---|
| **A** | `kappa * g == 0` (hard equality, production) |
| **B** | `-kappa·ε·S_rated² ≤ kappa * g ≤ +kappa·ε·S_rated²`, `ε = 1e-5` |

Branch B is applied **in place** on the existing `sess_snet_def`
`ConstraintData` objects via `set_value(pe.inequality(lb, body, ub))`, reusing
the same body expression — so the component, row objects and indices survive.
24 rows converted per case (one shared ESS × 24 periods).

### Implementation discipline (§5) — verified for all 11 cases

Every check passed in every case: `sess_snet_def` component object unchanged,
row object ids unchanged, index tuples unchanged, total constraint-data count
unchanged, no replacement component, identical constraint component-name sets,
identical primal starting values and bounds fingerprint, identical objective at
the starting point, identical `sess_comp` and ordinary-ESS row counts, identical
`kappa`, identical shared-ESS capacity, identical row activation state. Branch A
rows are all equalities and branch B rows are all ranged
(`A_rows_all_equality = True`, `B_rows_all_ranged = True`). All solves cold
(`from_warm_start=False`) with identical IPOPT/MA97/exact-Hessian settings.

---

## 1. Eight sensitive cases (§4, §6)

`it` = IPOPT iterations. All `S_rated`/`kappa` values are the production ones.

| Case | Group | S [MVA] | κ | **A** | **B** |
|---|---|---|---|---|---|
| n5 2030 Winter | P5 original failure | 0.02127 | 4701.5 | **maxIterations**, it=3000 | **optimal**, it=352 |
| n5 2035 Winter | P5 original failure | 0.03190 | 3134.3 | **maxIterations**, it=3000 | **optimal**, it=430 |
| n9 2025 Summer | P5 original failure | 0.01063 | 9403.0 | **maxIterations**, it=3000 | **optimal**, it=430 |
| n9 2025 Autumn | P5.1 cap-100 regression | 0.01063 | 9403.0 | optimal, it=438 | **optimal**, it=463 |
| n5 2030 Spring | P5.1-B cap-1000 regression | 0.02127 | 4701.5 | optimal, it=152 | **optimal**, it=621 |
| n5 2035 Autumn | P5.1-B cap-1000 regression | 0.03190 | 3134.3 | optimal, it=267 | **optimal**, it=338 |
| n7 2030 Summer | P5.1-B cap-1000 regression | 0.02127 | 4701.5 | optimal, it=160 | **optimal**, it=116 ⚠ **recovery used** |
| n7 2035 Spring | P5.1-B cap-1000 regression | 0.03190 | 3134.3 | optimal, it=454 | **optimal**, it=208 |

**All eight sensitive B cases succeed.** The three original P5 failures convert
from hard 3000-iteration stalls into clean convergence in 352–430 iterations.

Note on the five "regression" cases: under **untouched production `kappa`** and
in isolation they already succeed under A. They were regressions of the earlier
*capped-kappa* experiments, so here the requirement is that B must not break
them — and it does not.

⚠ **One exception, reported rather than smoothed over.** `n7 2030 Summer`
branch B did **not** succeed on the primary exact-Hessian path: its primary
solve failed with `internalSolverError` and the production recovery path
(limited-memory Hessian) then converged it. Under A the same case succeeded on
the primary path in 160 iterations. This is the single case where the band made
the primary path worse.

## 2. Objective comparison (A vs B)

| Case | A objective | B objective | Δ |
|---|---|---|---|
| n5 2030 Winter | 1.66799e+02 † | −8.012997e-04 | — † |
| n5 2035 Winter | 1.96853e+02 † | −8.086337e-04 | — † |
| n9 2025 Summer | 1.75793e+02 † | −8.152143e-04 | — † |
| n9 2025 Autumn | −8.2604212e-04 | −8.2604235e-04 | −2.22e-10 |
| n5 2030 Spring | −8.0806137e-04 | −8.0806152e-04 | −1.51e-10 |
| n5 2035 Autumn | −8.4502696e-04 | −8.4502699e-04 | −3.65e-11 |
| **n7 2030 Summer** | −8.2259349e-04 | −8.5219537e-04 | **−2.96e-05** ⚠ |
| n7 2035 Spring | −8.1432892e-04 | −8.1432895e-04 | −2.40e-11 |
| n5 2025 Winter (ctrl) | −7.9644313e-04 | −7.9644336e-04 | −2.27e-10 |
| n7 2025 Winter (ctrl) | −7.9499596e-04 | −7.9499619e-04 | −2.32e-10 |
| n9 2030 Summer (ctrl) | −8.2277936e-04 | −8.2277951e-04 | −1.47e-10 |

† A never converged, so its objective is not a meaningful comparison point.

For every genuinely-converged pair the objective difference is `1e-11`–`1e-10`,
i.e. numerically indistinguishable. The one outlier is again **n7 2030 Summer**
(`−2.96e-05`), the recovery case, which landed on a different local solution.

## 3. Physical-relaxation metrics (§7)

`ε_abs = ε·S_rated²` is the physical band on `g`; the scaled band is `±κ·ε_abs`.

| Case | ε_abs [p.u.²] | scaled bound | max｜g｜ | max｜g｜/S² | **max band ratio** | max ΔS [p.u.] | max ΔS [MVA] | **ΔS/S_rated** |
|---|---|---|---|---|---|---|---|---|
| n5 2030 Winter | 4.524e-13 | ±2.127e-09 | 8.348e-14 | 1.845e-06 | 0.1845 | 6.506e-08 | 6.51e-06 | 3.059e-04 |
| n5 2035 Winter | 1.018e-12 | ±3.190e-09 | 9.280e-14 | 9.117e-07 | 0.0912 | 8.964e-08 | 8.96e-06 | 2.810e-04 |
| n9 2025 Summer | 1.131e-13 | ±1.063e-09 | 2.859e-14 | 2.528e-06 | 0.2528 | 2.976e-08 | 2.98e-06 | 2.798e-04 |
| n9 2025 Autumn | 1.131e-13 | ±1.063e-09 | 2.858e-14 | 2.527e-06 | 0.2527 | 2.976e-08 | 2.98e-06 | 2.798e-04 |
| n5 2030 Spring | 4.524e-13 | ±2.127e-09 | 8.372e-14 | 1.851e-06 | 0.1851 | 6.515e-08 | 6.52e-06 | 3.063e-04 |
| n5 2035 Autumn | 1.018e-12 | ±3.190e-09 | 9.214e-14 | 9.051e-07 | 0.0905 | 8.916e-08 | 8.92e-06 | 2.795e-04 |
| n7 2030 Summer | 4.524e-13 | ±2.127e-09 | 1.263e-17 | 2.792e-10 | 2.79e-05 | 1.190e-09 | 1.19e-07 | 5.595e-06 |
| n7 2035 Spring | 1.018e-12 | ±3.190e-09 | 9.280e-14 | 9.116e-07 | 0.0912 | 8.963e-08 | 8.96e-06 | 2.809e-04 |
| n5 2025 Winter (ctrl) | 1.131e-13 | ±1.063e-09 | 2.859e-14 | 2.528e-06 | 0.2528 | 2.976e-08 | 2.98e-06 | 2.798e-04 |
| n7 2025 Winter (ctrl) | 1.131e-13 | ±1.063e-09 | 2.859e-14 | 2.528e-06 | 0.2528 | 2.976e-08 | 2.98e-06 | 2.798e-04 |
| n9 2030 Summer (ctrl) | 4.524e-13 | ±2.127e-09 | 8.351e-14 | 1.846e-06 | 0.1846 | 6.508e-08 | 6.51e-06 | 3.060e-04 |

**Band activity — the economic-exploitation question.** In **every** case, all
**24 of 24** periods are classified *well inside the band*:

- periods within 10 % of either boundary: **0 of 24**, in all 11 cases;
- periods exactly active at a boundary: **0 of 24**, in all 11 cases;
- largest band ratio anywhere: **0.253** (i.e. `|g|` reaches at most a quarter
  of the allowed band).

**The relaxation is not being economically exploited.** It is helping
numerically — by making the zero-dispatch start strictly interior rather than
equality-active — not by buying feasibility slack. The residual apparent-power
mismatch is at most `ΔS = 9.0e-06 MVA` (≈ 9 VA), i.e. `3.1e-04` of rated power,
and `max|g|/S_rated² ≤ 2.5e-06`, well inside `EQUALITY_TOLERANCE = 1e-5`.

## 4. Matched-success controls (§8)

One previously-successful bootstrap initialization per DSO, all successful under
both branches:

| Control | A | B | Δ objective | ΔS/S_rated |
|---|---|---|---|---|
| node 5 — n5 2025 Winter | optimal, it=122 | optimal, it=298 | −2.27e-10 | 2.798e-04 |
| node 7 — n7 2025 Winter | optimal, it=1058 | optimal, it=**319** | −2.32e-10 | 2.798e-04 |
| node 9 — n9 2030 Summer | optimal, it=488 | optimal, it=398 | −1.47e-10 | 3.060e-04 |

No control regressed; the node-7 control improved substantially (1058 → 319
iterations).

## 5. Full initialization (§9) — the headline result

The complete P5 iteration-2 `positive_bootstrap` operational initialization was
re-run with the narrow band applied through the production path (by wrapping
`configure_shared_ess_operational_state` inside the diagnostic process and
restoring it afterwards) and the **untouched production `kappa = 1/S_rated`**.
ADMM and the outer planning loop were **not** entered.

| | |
|---|---|
| Total local solves | **51** |
| DSO | 36 — **0 failures** |
| TSO | 12 — **0 failures** |
| ESSO | 3 — **0 failures** |
| Total failures | **0** |
| `_admm_local_solves_succeeded` | **True** |
| **Would initialization enter ADMM?** | **Yes** |

This is the first intervention in P5.1 / P5.1-B / P5.2-A that makes the complete
bootstrap initialization succeed. For comparison: production scaling gave 3
failures, cap-100 gave 1 (different) failure, cap-1000 gave 4 (different)
failures, and the narrow band gives **0**.

**Caveat on this run:** recovery usage was not separately instrumented inside
the 51-solve gate, so this report cannot claim all 51 succeeded on the primary
exact-Hessian path — only that all 51 succeeded.

---

## 6. Verdict reasoning — why PARTIAL and not PASS

The headline evidence is strong and, unlike every earlier intervention, does not
relocate failures: all eight sensitive cases succeed, all three controls hold,
the physical mismatch is negligible, the band is never active, and the full
51-solve initialization completes and would enter ADMM.

However, §9's gate condition is *"all eight sensitive B cases succeed **on the
primary solve**"*, and that condition is **not strictly met**: `n7 2030 Summer`
branch B failed its primary solve with `internalSolverError` and converged only
via the recovery path, where branch A had succeeded on the primary path. That
same case is also the only one with a non-negligible objective shift
(`−2.96e-05`, versus `~1e-10` everywhere else), indicating a different local
solution.

Reporting PASS would require overlooking a stated gate condition that a targeted
case did not meet. The result is therefore recorded as PARTIAL — with the
explicit note that the shortfall is a single primary-path recovery event, not a
failure, and that the full-initialization outcome is clean.

## 7. Interpretation (§11) — kept narrow

1. **Supported.** The hypothesis holds up: the hard zero-gradient equality —
   not merely its scalar normalization — is a major source of cold-start KKT
   fragility. At `sch = sdch = pnet = qnet = 0` we have `g = 0` and
   `grad(g) = 0`, so the row is active with a vanishing gradient and no finite
   `kappa` can repair it; making the point strictly interior does. The three
   states that sat exactly at that point for 3000 iterations under every tested
   `kappa` converge in 352–430 iterations once the row is banded.
2. **Supported and materially stronger than the scaling experiments.** Unlike
   cap-100 and cap-1000, the band did not move the failure set — it emptied it
   (0 of 51).
3. **Not established.** That `ε = 1e-5` is the right or a safe production value;
   that behaviour holds through ADMM or the outer planning loop (neither was
   entered); that the single recovery event at n7 2030 Summer is benign; or that
   no larger-capacity regime is adversely affected. Only one ε was tested, as
   instructed.
4. **No production relaxation is authorized by this report.** Per §11,
   productionization would require planner review of the maximum normalized
   physical mismatch (here ≤ 2.5e-06 relative, ≤ 9 VA absolute), band activity
   (never active, max ratio 0.253), objective impact (≤ 1e-10 except the one
   recovery case) and full initialization behaviour (51/51 clean).

## 8. Scope compliance (§10)

Production `sess_snet_def`, `kappa`, `shared_ess_snet_def_scale`, the recovery
policy, `sess_comp`, SOC, degradation, ordinary ESS, IPOPT settings, MA97, ADMM
and planning logic are all untouched. No cap was applied. The band exists only
on cloned in-memory models and inside a temporarily wrapped function that was
restored. ADMM and the outer planning loop were not entered. Only
`ε_rel = 1e-5` was tested; no other value was tried.

---

```
P5.2-A PARTIAL — narrow-band evidence mixed — waiting for planner review
```
