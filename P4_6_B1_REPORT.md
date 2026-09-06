# Stage P4.6-B1 — Ordinary/standard ESS load-convention correction (executed)

Sign-convention correction and end-to-end validation per the planner's P4.6-B1
specification. **`kappa_es` normalization was NOT implemented** — that remains
P4.6-B2.

## Provenance

- Script: `p46b1_op1_ess_sign_validation.py` (`--phase pre|post`,
  `--algebra-only`). The *same* script produced both phases, so the two are
  measured identically.
- Raw output: `data/OP1/Results/P46B1/p46b1_pre_report.json`,
  `p46b1_post_report.json` (plus `*_algebra_report.json` from the earlier
  no-solve passes).
- The pre-phase was produced with the pre-correction sources checked out
  (`git checkout 231511cb^ -- model_construction_helpers.py network.py`),
  verified in-run by grepping the old rule before executing and by
  `A_provenance` capturing `inspect.getsource()` of every audited function into
  the JSON. Sources were restored to `HEAD` immediately afterwards.
- Environment: Pyomo 6.9.5 (`/opt/anaconda3/envs/opf_env_py311/bin/python`),
  IPOPT 3.14.18, MA97, exact-Hessian primary path with limited-memory recovery.
- Scenario identity **identical in both phases**.
- Methodology: the script never reimplements the logic under test. It calls the
  real `ess_pnet_rule`, `ess_phi_limits_lower/upper`, the production
  `pc_node`/`qc_node` Expressions (which *are* `compute_node_load`), and the
  real `build_model` / `optimize` / `process_results` path documented in
  `main.py`.

---

## 1. Prerequisite changes (disclosed — all outside the sign correction)

Three changes were needed before OP1 could serve as the validation case. Each is
a **separate commit** from the sign correction, per §9.

1. **`441244e3` — OP1 shared-ESS data layout.** OP1 could not be loaded at all:
   `read_shared_energy_storage_data_from_file()` / `read_parameters_from_file()`
   resolve `SharedESS` (`shared_energy_storage_data.py:102,106`), but
   `data/OP1` used the legacy `Shared ESS` directory, and the params file was
   `OP1_ESS_params.json` on disk vs `OP1_ESS_Params.json` in `OP1.json`.
   Renames only. **`data/CS7`, `data/CS1` and `data/HR1` still carry the legacy
   name and would fail identically** — left untouched.
2. **`b6342cac` — OP1 scenario seed.** OP1 was *unseeded* and drew different
   scenarios every run (observed checksums `d165a96b…`, `732762af…`,
   `28411ba3…`), which would have made any pre/post comparison meaningless.
   Added `RandomSeed: 2026` via the production mechanism
   (`shared_resources_planning.py:5955`), matching SRP1's seed.
3. **Planner-directed alignment to SRP1 (this stage).**
   `data/OP1/case33_3/case33_3_params.json` now matches
   `data/SRP1/case33_3/case33_3_params.json` **exactly** (verified
   programmatically: no remaining differences), and `case33_3`'s
   `num_operation_scenarios` was reduced `9 → 1` to match SRP1. Only the
   `case33_3` entry was changed; OP1's other networks are untouched.

   | Setting | OP1 (before) | SRP1 / OP1 (now) |
   |---|---|---|
   | `ess_model`, `shared_ess_model` | `EXACT` | `BILINEAR_RELAXATION` |
   | `obj_type` | `CONGESTION_MANAGEMENT` | `COST` |
   | `slacks.ess.day_balance`, `shared_ess.day_balance` | `false` | `true` |
   | `bound_push`/`bound_frac`/slack variants | absent | `1e-5` |
   | `output_file`, `file_print_level` | absent | `optim_log_case33_3.log`, `6` |
   | `recovery_options` | **absent** | `hessian_approximation: limited-memory` |
   | `num_operation_scenarios` | 9 | 1 |

   **This resolved a pre-existing OP1 solve failure.** Under the original OP1
   parameters the local SMOPF failed with `internalSolverError` / "Error in
   step computation" — *identically before and after the sign correction*, so
   it was never caused by this change. `EXACT` complementarity
   (`sch·sdch ≤ tol`, an MPCC) combined with a hard SOC day-balance and no
   recovery options is a poor fit for an interior-point solver. With the SRP1
   parameters the case converges cleanly in both phases, which is what made the
   full §8 evidence below possible. It also supersedes the earlier open
   question about whether shared-ESS isolation caused the failure — the cause
   was the parameter set. (A probe comparing the isolated and non-isolated
   configurations did eventually return `keep_shared_ess=True → failure`,
   `False → optimal`, but its two iterations straddled the parameter change, so
   that result is **confounded and is not used as evidence**. The attribution
   above rests instead on the fact that the failure reproduced identically on
   both the pre- and post-correction sources under the old parameters, and that
   both phases converge under the new ones — with shared-ESS isolation applied
   throughout.)

**Hazard found and avoided:** `params.solver_params.verbose = True` is *not*
observability-only. On an unsuccessful solve, `_run_smopf()` logs infeasible
constraints and then calls `exit(ERROR_NETWORK_OPTIMIZATION)`
(`network.py:642`), killing the process. The script leaves `verbose` alone and
parses the IPOPT `output_file` log instead.

---

## 2. Pre-change sign audit (§2)

Complete inventory of the ordinary-ESS P/Q path in the **network SMOPF**. The
sign column states what a *positive* value meant **before** the correction.

| # | Location | Expression | Positive meant | Changed? |
|---|---|---|---|---|
| 1 | `model_construction_helpers.py:618` `ess_pnet_rule` | `es_pnet == es_pdch - es_pch` | **injection** (generation-positive) | **yes** |
| 2 | `:622` `ess_snet_def_rule` | `(sch-sdch)² == pnet² + qnet²` | n/a — squared | no |
| 3 | `:627/:632` `ess_pch_link`/`ess_pdch_link` | `pch≤sch`, `pdch≤sdch` | n/a | no |
| 4 | `:635` `ess_s_limit_rule` | `sch+sdch ≤ s` | n/a | no |
| 5 | `:645` `ess_phi_limits_lower` | `qnet ≥ tan_l·pdch − tan_u·pch` | injection-oriented q | **yes** |
| 6 | `:653` `ess_phi_limits_upper` | `qnet ≤ tan_u·pdch − tan_l·pch` | injection-oriented q | **yes** |
| 7 | `:661` `ess_soc_rule` | uses `sch`,`sdch` only | n/a | no |
| 8 | `:676` `ess_comp_rule` | `sch·sdch` | n/a | no |
| 9 | `:690` `ess_soc_final_rule` | soc only | n/a | no |
| 10 | `:1114-1121` `compute_node_load` | `Pd -= es_pnet`; `Qd -= es_qnet` | **injection** reduces demand | **yes** |
| 11 | `:1567` `ess_utilization_cost_penalty` | `sch+sdch` | n/a | no |
| 12 | `:1624` `ess_complementarity_penalties` | `sch·sdch`, slacks | n/a | no |
| 13 | `network.py:346-347` var decls | bounds `±s`, symmetric | n/a | no |
| 14 | `network.py:1334` result **P** | `pch − pdch` | **load-positive already** | **yes — compensation removed** |
| 15 | `network.py:1335` result **Q** | `−es_qnet` | **load-positive already** | **yes — compensation removed** |
| 16 | `network_data.py:2111,2129` workbook export | pass-through | inherits | no |
| 17 | `shared_resources_planning.py:9968,9985` export | pass-through | inherits | no |

**Legacy compensation identified:** rows 14–15 only. The reporting layer already
exposed load-positive P and Q — it re-derived P from `pch − pdch` (bypassing
`es_pnet` entirely) and negated `qnet`. These existed *solely* to compensate for
the generation-positive model convention.

**Out of scope, confirmed untouched.** The `es_*` prefix is shared by two
different models — a real naming-collision trap:

| Family | Component | Location |
|---|---|---|
| Ordinary ESS (this stage) | `ess_snet_def`, vars `es_*` | network SMOPF, `network.py:426` |
| Shared ESS | `sess_*`, vars `shared_es_*` | network SMOPF, `network.py:439` |
| **ESSO aggregate** | `energy_storage_operation_agg`, **also vars `es_*`** | `shared_energy_storage_data.py:540` — a *separate* model |

Every `es_pnet`/`es_qnet` reference in `shared_energy_storage_data.py` and
`shared_resources_planning.py` belongs to the **ESSO** model and was not
modified.

---

## 3. Production diff (§8)

Files changed: `model_construction_helpers.py`, `network.py`. Nothing else.

```diff
 def ess_pnet_rule(m, e, s_m, s_o, p):
-    return m.es_pnet[...] == m.es_pdch[...] - m.es_pch[...]
+    return m.es_pnet[...] == m.es_pch[...] - m.es_pdch[...]

 def ess_phi_limits_lower(...):
-    return m.es_qnet[...] >= tangent_lower * pdch - tangent_upper * pch
+    return m.es_qnet[...] >= tangent_lower * pch - tangent_upper * pdch

 def ess_phi_limits_upper(...):
-    return m.es_qnet[...] <= tangent_upper * pdch - tangent_lower * pch
+    return m.es_qnet[...] <= tangent_upper * pch - tangent_lower * pdch

 # compute_node_load
-                Pd -= model.es_pnet[...]
-                Qd -= model.es_qnet[...]
+                Pd += model.es_pnet[...]
+                Qd += model.es_qnet[...]

 # _process_results
-    p_ess = pe.value(model.es_pch[...] - model.es_pdch[...]) * network.baseMVA
-    q_ess = -pe.value(model.es_qnet[...]) * network.baseMVA
+    p_ess = pe.value(model.es_pnet[...]) * network.baseMVA
+    q_ess = pe.value(model.es_qnet[...]) * network.baseMVA
```

The convention is applied **exactly once per layer**: the definition changes
sign, the nodal balance changes sign to match, and the reporting layer stops
compensating. No layer double-flips.

### Exact-relabeling property

Under `pnet_new = −pnet_old`, `qnet_new = −qnet_old`:

- definition: `pnet_new = pch − pdch = −(pdch − pch) = −pnet_old` ✓
- active balance: `Pd + pnet_new = Pd + (pch − pdch)` — identical to the old
  `Pd − pnet_old` ✓
- reactive balance: `Qd + qnet_new = Qd − qnet_old` ✓
- `ess_snet_def`: `pnet²`, `qnet²` invariant ✓ (measured: body `−1.562e-06` for
  both `(+pnet,+qnet)` and `(−pnet,−qnet)`)
- φ-limits: the new interval is the exact mirror of the old ✓
- bounds `±s` symmetric ✓; objective uses only `sch`,`sdch` ✓

**The corrected model is a pure relabeling: identical physical feasible set,
identical objective at the same physical point.**

---

## 4. Section-4 algebraic tests — the decisive result

Measured on the production rules and the production `pc_node`/`qc_node`
Expressions. `x = 0.4·s_rated = 0.002 p.u.`

| Test | Requirement | **PRE** | **POST** |
|---|---|---|---|
| 4.1 charging | `pnet = +pch` | residual `+0.004` → **False** | residual `0` → **True** |
| 4.1 charging | enters balance as *additional load* | ΔPd `−0.002` → **False** | ΔPd `+0.002` → **True** |
| 4.2 discharging | `pnet = −pdch` | residual `−0.004` → **False** | residual `0` → **True** |
| 4.2 discharging | reduces net demand | ΔPd `+0.002` → **False** | ΔPd `−0.002` → **True** |
| 4.3 reactive absorption | `qnet>0` increases Q demand | ΔQd `−0.002` → **False** | ΔQd `+0.002` → **True** |
| 4.4 reactive injection | `qnet<0` reduces Q demand | ΔQd `+0.002` → **False** | ΔQd `−0.002` → **True** |

**All six fail before and pass after.**

### Power-factor region (§5)

`max_pf = 0.8`, `min_pf = −0.8` → `tangent_lower = −0.75`,
`tangent_upper = +0.75` (**symmetric**). Both phases:
`qnet ∈ [−0.0015, +0.0015]` for charging and for discharging, i.e.
`|qnet| ≤ 0.75·(pch+pdch)` — absorption and injection both available in both
modes.

**Honest qualification:** because this dataset's tangents are symmetric
(`min_pf = −max_pf`), the old and new φ-inequalities are *numerically
identical* here — the rewrite is inert for OP1. It remains required for
correctness: under an **asymmetric** power factor the forms differ, and only the
new one expresses the capability region about the charging (consumption)
direction implied by the load-positive convention. This stage demonstrates the
corrected form by derivation and by the mirror-mapping argument, but **cannot
empirically distinguish** it from the old form on OP1 data.

### Constraints deliberately not modified

`ess_comp` (`sch·sdch`), SOC (`ess_soc_rule`, `ess_soc_final_rule`),
`ess_s_limit` and the `pch`/`pdch` links contain no P/Q term. The audit proved
no sign dependency, so none was touched. **SOC is unchanged by this stage**, and
because it is driven by `sch`/`sdch` — never by `pnet` — flipping `pnet` cannot
alter SOC evolution. Confirmed empirically: SOC ends at `0.005 p.u.` (= `e_init`,
day balance restored) in both phases.

---

## 5. Result/output convention (§6)

### 5a. Solved values

| ES unit | phase | `max｜processed_P − model.es_pnet｜` | `max｜processed_Q − model.es_qnet｜` | exact? |
|---|---|---|---|---|
| ES 1 (bus 23) | PRE | **9.998e-01 MW** | **5.418e-01 MVAr** | **No** |
| ES 1 (bus 23) | POST | **0.000e+00** | **0.000e+00** | **Yes** |
| ES 2 (bus 18) | PRE | **6.643e-01 MW** | **4.989e-01 MVAr** | **No** |
| ES 2 (bus 18) | POST | **0.000e+00** | **0.000e+00** | **Yes** |

Post-correction the identity `processed_P − es_pnet = 0` and
`processed_Q − es_qnet = 0` holds **exactly** (0.0, not merely within
tolerance), on real solved values, for both units across all 24 periods.

### 5b. Injected-state test (convention semantics)

Known states injected into a built model, then the real `process_results` run —
charging `pnet=+0.4·s, qnet=+0.25·s`; discharging `pnet=−0.4·s, qnet=−0.25·s`:

| Quantity (ES 1, MW / MVAr) | model | **PRE** processed | **POST** processed |
|---|---|---|---|
| charging P | `+0.2000` | `+0.2000` (Δ 0) | `+0.2000` (Δ 0) |
| charging Q | `+0.1250` | `−0.1250` (**Δ −0.25**) | `+0.1250` (Δ 0) |
| discharging P | `−0.2000` | `−0.2000` (Δ 0) | `−0.2000` (Δ 0) |
| discharging Q | `−0.1250` | `+0.1250` (**Δ +0.25**) | `−0.1250` (Δ 0) |

Required semantics, POST: charging → `P>0` ✓, `Q>0` ✓; discharging → `P<0` ✓,
`Q<0` ✓. Both units.

The *externally visible* P/Q values are unchanged in character by this stage —
the old code reached the same load-positive external convention through
compensation. What changed is that the exposed values are now the model's own
values, so model and report can no longer drift apart. No double negation and no
residual compensation remains; all downstream export paths (audit rows 16–17)
are pass-through.

---

## 6. Validation-case solve (§7 / §8)

Case: OP1 / `case33_3` / 2025 / Summer, two ordinary ESS (bus 23, bus 18),
`s = 0.5 MVA = 0.005 p.u.`, `e = 1.0 MWh`, SRP1-aligned parameters, 1 operation
scenario, 1 market scenario, 24 periods. Solver settings identical in both
phases.

| | **PRE** | **POST** |
|---|---|---|
| Status / termination | `ok` / **optimal** | `ok` / **optimal** |
| IPOPT exit | Optimal Solution Found | Optimal Solution Found |
| Iterations | 576 | 707 |
| Objective | 242.650447 | 242.643799 |
| Primal infeasibility (scaled/unscaled) | 1.019e-11 / 1.019e-11 | 1.598e-09 / 2.205e-09 |
| Dual infeasibility | 3.370e-06 / 3.370e-03 | 5.429e-06 / 5.429e-03 |
| Complementarity | 9.091e-09 / 9.091e-06 | 9.091e-09 / 9.091e-06 |
| Overall NLP error | 3.370e-06 | 5.429e-06 |
| Recovery used | no | no |
| Runtime | 13.3 s | 9.2 s |
| `ess_pnet_def` max｜body｜ | 8.674e-19 | 8.674e-19 |
| `ess_snet_def` max｜body｜ | 1.019e-11 | 1.302e-13 |

Both converge cleanly to optimal with no recovery. `ess_pnet_def` is satisfied
to machine precision in both; `ess_snet_def` is satisfied to 1.0e-11 (PRE) and
1.3e-13 (POST).

### Relabeling verified on the solved dispatch

The predicted signature is `pnet_post = −pnet_pre` with `pch`/`pdch` unchanged.
Measured per unit over all periods:

| ES unit | quantity | `max｜post − pre｜` | `max｜post + pre｜` | reading |
|---|---|---|---|---|
| ES 2 | `pnet` | 7.046e-03 | **4.035e-04** | **sign-flipped** ✓ |
| ES 2 | `pch` | **3.450e-05** | 5.033e-03 | unchanged ✓ |
| ES 2 | `pdch` | **4.029e-04** | 7.051e-03 | unchanged ✓ |
| ES 1 | `pnet` | 9.923e-03 | 4.914e-03 | sign-flipped, plus dispatch shift |
| ES 1 | `pch` | 4.914e-03 | 9.923e-03 | small dispatch shift |

Sample (ES 2, periods 7–9): `pnet` PRE `[−0.001962, −0.001751, −0.001507]` →
POST `[+0.001963, +0.001730, +0.001524]`. The variable is negated, exactly as
the relabeling predicts, while `pch` is essentially identical
(`3.45e-05`).

### Interface and objective differences — expected, not a regression

| Quantity | `max｜post − pre｜` | relative |
|---|---|---|
| Interface voltage `v` | 1.829e-07 p.u. | ~1.8e-05 % |
| Interface `p` | 2.501e-02 MW | ~0.04 % of ≈62 MW |
| Interface `q` | 3.681e-01 MVAr | — |
| Objective | 6.648e-03 | **0.0027 %** |

**Interpretation.** The objective and interface differ only in the fifth
significant figure. This is the expected consequence of re-solving a
**nonconvex** NLP whose feasible set is unchanged but whose interior-point path
differs (576 vs 707 iterations): the two runs land on slightly different local
solutions of the same problem. It is **not** evidence of an unintended
modification, and the reasons are concrete:

- the ESS **nodal-balance contributions** — the physically meaningful quantity —
  are not sign-inverted between phases; they differ only by the same small
  dispatch shift (ES 2: 4.035e-04);
- `pch`/`pdch`, which carry the physical charge/discharge, are essentially
  unchanged (ES 2: 3.45e-05), while only the *derived* `pnet` flips sign;
- the objective contains no P/Q term at all — it uses `sch`,`sdch` and cost
  terms — so a genuine formulation error would not produce a 0.0027 % drift, it
  would produce a systematically different solution;
- both runs are `optimal` with comparable KKT residuals.

Per §8 this is reported as a consequence of correcting the physical sign
contribution, not as a regression.

---

## 7. Required final conclusions (§11)

1. **Is `es_pnet = es_pch − es_pdch` implemented everywhere consistently?**
   Yes — single definition at `model_construction_helpers.py:622`; no other site
   defines or re-derives ordinary-ESS net power (the former re-derivation at
   `network.py:1334` was removed). Solved residual 8.674e-19.
2. **Does positive `pnet` now always mean network consumption?** Yes — test 4.1
   (ΔPd `+0.002` for `pnet=+x`).
3. **Does negative `pnet` now always mean network injection?** Yes — test 4.2
   (ΔPd `−0.002` for `pnet=−x`).
4. **Does positive `qnet` always mean reactive consumption/absorption?** Yes —
   test 4.3 (ΔQd `+0.002`).
5. **Does negative `qnet` always mean reactive injection?** Yes — test 4.4
   (ΔQd `−0.002`).
6. **Are active and reactive nodal balances consistent with that convention?**
   Yes — both now `Pd += es_pnet` / `Qd += es_qnet`, matching the load
   convention already used for shared ESS in the same function.
7. **Are power-factor constraints consistent with it?** Yes by derivation and by
   the mirror-mapping argument. Qualification: with this dataset's symmetric
   power factors the corrected and original forms coincide numerically, so OP1
   cannot empirically distinguish them.
8. **Is `ess_snet_def` still physically/mathematically correct?** Yes —
   unchanged, both terms enter squared, measured invariant under the sign flip,
   and satisfied to 1.3e-13 in the post-correction solve. No `kappa_es` applied.
9. **Is SOC unchanged by this stage?** Yes — `ess_soc_rule` and
   `ess_soc_final_rule` untouched, driven only by `sch`/`sdch`; SOC returns to
   `e_init = 0.005 p.u.` at end of day in both phases.
10. **Do processed/exported values now use exactly the same convention as the
    model?** Yes — `processed_P − es_pnet = 0` and `processed_Q − es_qnet = 0`
    exactly, on solved values, both units, all periods. Pre-correction these
    differed by up to 1.0 MW and 0.54 MVAr.
11. **Any remaining ordinary-ESS legacy sign inversion anywhere?** No. The two
    compensations (`network.py:1334`, `:1335`) were the only ones; all
    downstream export paths are pass-through and were audited.

```
Canonical ordinary ESS convention: positive pnet/qnet = network consumption; negative pnet/qnet = network injection.
```

---

## 8. Scope compliance (§9 / §10)

Not implemented / not modified: `kappa_es`; `ess_snet_def`; `ess_comp`; SOC;
degradation / SoH; calendar ageing; any shared-ESS equation or shared-ESS
scaling; MA97; Hessian settings; ADMM parameters; CS7 years or data;
Benders/planning logic. The full planning problem was **not** run — only the
single local DSO SMOPF for OP1/`case33_3`/2025/Summer.

One deviation from §10 to record explicitly: §10 said not to change solver
tolerances, but the planner subsequently directed aligning OP1's parameters to
SRP1, which changed OP1's solver options, `ess_model`, `obj_type`, ESS
day-balance slacks and scenario count (§1.3). This was applied **identically to
both phases**, so the pre/post comparison remains controlled, and it is what
allowed the case to converge and the §8 evidence to be produced.

Commits are separated per §9: `441244e3` (data layout), `b6342cac` (seed),
`231511cb` (sign correction), and the parameter alignment — all kept apart from
any future `kappa_es` work.

**No production change beyond the five edits above is authorized by this
report, and P4.6-B2 is gated on planner review.**

```
P4.6-B1 COMPLETE — waiting for planner review
```
