# Stage P4.6-B1 — Ordinary/standard ESS load-convention correction (executed)

Sign-convention correction and end-to-end validation per the planner's P4.6-B1
specification. **`kappa_es` normalization was NOT implemented** — that remains
P4.6-B2.

## Provenance

- Script: `p46b1_op1_ess_sign_validation.py` (`--phase pre|post`,
  `--algebra-only`). The *same* script produced both phases, so the two are
  measured identically.
- Raw output:
  `data/OP1/Results/P46B1/p46b1_pre_report.json`,
  `p46b1_post_report.json`, `p46b1_pre_algebra_report.json`,
  `p46b1_post_algebra_report.json`.
- Pre-phase runs were produced with the production diff `git stash`ed, so the
  baseline is the genuine pre-correction code (verified in-run by grepping the
  old rule before executing, and by `A_provenance` capturing
  `inspect.getsource()` of every audited function into the JSON).
- Environment: Pyomo 6.9.5 at `/opt/anaconda3/envs/opf_env_py311/bin/python`,
  IPOPT 3.14.18, MA97, exact Hessian — all unchanged.
- Scenario identity: **identical in both phases**, checksum
  `28411ba309a33c439906675968fdb60fe38aaa820a81df248c9bfcbe87bca8ea`.
- Methodology: the script never reimplements the logic under test. It calls the
  real `ess_pnet_rule`, `ess_phi_limits_lower/upper`, the production
  `pc_node`/`qc_node` Expressions (which *are* `compute_node_load`), and the
  real `build_model` / `optimize` / `process_results` path documented in
  `main.py`.

---

## 1. Prerequisite fixes (disclosed — both outside the sign correction)

Two blockers had to be cleared before OP1 could be used at all. Each is a
**separate commit** from the sign correction, per §9.

1. **`441244e3` — OP1 shared-ESS data layout.** OP1 could not be loaded:
   `read_shared_energy_storage_data_from_file()` /
   `read_parameters_from_file()` resolve `SharedESS`
   (`shared_energy_storage_data.py:102,106`), but `data/OP1` still used the
   legacy `Shared ESS` directory, and the params file was
   `OP1_ESS_params.json` on disk vs `OP1_ESS_Params.json` in `OP1.json`.
   Renames only; no file contents changed. Approved by the planner in-session.
   **`data/CS7`, `data/CS1` and `data/HR1` still carry the legacy name and
   would fail identically** — left untouched.
2. **`RandomSeed: 2026` added to `data/OP1/OP1.json`.** OP1 was *unseeded*
   (`[INFO] Scenario random seed: unseeded`) and drew different scenarios every
   run — three consecutive runs gave checksums `d165a96b…`, `732762af…`,
   `28411ba3…`. Pre/post comparison would have been meaningless. This uses the
   production mechanism (`RandomSeed`, `shared_resources_planning.py:5955`) and
   the same seed value SRP1 uses. One-line insertion; no other spec field
   touched.

**Hazard found and avoided:** `params.solver_params.verbose = True` is *not*
observability-only in this codebase. On an unsuccessful solve, `_run_smopf()`
logs infeasible constraints and then calls `exit(ERROR_NETWORK_OPTIMIZATION)`
(`network.py:642`), killing the process. The script therefore leaves `verbose`
alone and takes metrics from the returned result object.

---

## 2. Pre-change sign audit (§2)

Complete inventory of the ordinary-ESS P/Q path in the **network SMOPF**. Sign
column states what a *positive* value meant **before** the correction.

| # | Location | Expression | Positive meant | Sign-dependent? |
|---|---|---|---|---|
| 1 | `model_construction_helpers.py:618` `ess_pnet_rule` | `es_pnet == es_pdch - es_pch` | **injection** (generation-positive) | **yes — changed** |
| 2 | `:622` `ess_snet_def_rule` | `(sch-sdch)² == pnet² + qnet²` | n/a — squared | no |
| 3 | `:627/:632` `ess_pch_link` / `ess_pdch_link` | `pch≤sch`, `pdch≤sdch` | n/a | no |
| 4 | `:635` `ess_s_limit_rule` | `sch+sdch ≤ s` | n/a | no |
| 5 | `:645` `ess_phi_limits_lower` | `qnet ≥ tan_l·pdch − tan_u·pch` | injection-oriented q | **yes — changed** |
| 6 | `:653` `ess_phi_limits_upper` | `qnet ≤ tan_u·pdch − tan_l·pch` | injection-oriented q | **yes — changed** |
| 7 | `:661` `ess_soc_rule` | uses `sch`,`sdch` only | n/a | no |
| 8 | `:676` `ess_comp_rule` | `sch·sdch` | n/a | no |
| 9 | `:690` `ess_soc_final_rule` | soc only | n/a | no |
| 10 | `:1114-1121` `compute_node_load` | `Pd -= es_pnet`; `Qd -= es_qnet` | **injection** reduces demand | **yes — changed** |
| 11 | `:1567` `ess_utilization_cost_penalty` | `sch+sdch` | n/a | no |
| 12 | `:1624` `ess_complementarity_penalties` | `sch·sdch`, slacks | n/a | no |
| 13 | `network.py:346-347` var decls | bounds `±s`, symmetric | n/a | no |
| 14 | `network.py:1334` result **P** | `pch − pdch` | **load-positive already** | **yes — compensation removed** |
| 15 | `network.py:1335` result **Q** | `−es_qnet` | **load-positive already** | **yes — compensation removed** |
| 16 | `network_data.py:2111,2129` workbook export | pass-through | inherits | no |
| 17 | `shared_resources_planning.py:9968,9985` export | pass-through | inherits | no |

**Legacy compensation identified:** rows 14–15. The reporting layer already
exposed load-positive P and Q — it re-derived P from `pch − pdch` (bypassing
`es_pnet` entirely) and negated `qnet`. These existed *only* to compensate for
the generation-positive model convention.

**Explicitly out of scope, and confirmed untouched.** The `es_*` prefix is
shared by two different models — a real naming-collision trap:

| Family | Component | Location |
|---|---|---|
| Ordinary ESS (this stage) | `ess_snet_def`, vars `es_*` | network SMOPF, `network.py:426` |
| Shared ESS | `sess_*`, vars `shared_es_*` | network SMOPF, `network.py:439` |
| **ESSO aggregate** | `energy_storage_operation_agg`, **also vars `es_*`** | `shared_energy_storage_data.py:540`, a *separate* model |

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

The convention is applied **exactly once** per layer: the definition changes
sign, the nodal balance changes sign to match, and the reporting layer stops
compensating. No layer double-flips.

### Exact-relabeling property

Under `pnet_new = −pnet_old`, `qnet_new = −qnet_old`:

- definition: `pnet_new = pch − pdch = −(pdch − pch) = −pnet_old` ✓
- active balance: `Pd + pnet_new = Pd + (pch − pdch)` — identical to the old
  `Pd − pnet_old = Pd − (pdch − pch)` ✓
- reactive balance: `Qd + qnet_new = Qd − qnet_old` ✓
- `ess_snet_def`: `pnet²`, `qnet²` invariant ✓ (measured: body
  `−1.562e-06` for both `(+pnet,+qnet)` and `(−pnet,−qnet)`)
- φ-limits: the new interval is the exact mirror of the old, so
  `qnet_old ∈ [tan_l·pdch − tan_u·pch, tan_u·pdch − tan_l·pch]` maps onto the
  new interval ✓
- bounds `±s` symmetric ✓; objective uses only `sch`,`sdch` ✓

**The corrected model is therefore a pure relabeling of the old one: identical
physical feasible set, identical objective.** Any material change in objective
or interface quantities would indicate an unintended modification — none was
introduced.

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

**All six requirements fail before and pass after.**

### Power-factor region (§5)

`max_pf = 0.8`, `min_pf = −0.8` → `tangent_lower = −0.75`,
`tangent_upper = +0.75` (**symmetric**).

| Mode | PRE `qnet ∈` | POST `qnet ∈` |
|---|---|---|
| charging (`pch=0.002`) | `[−0.0015, +0.0015]` | `[−0.0015, +0.0015]` |
| discharging (`pdch=0.002`) | `[−0.0015, +0.0015]` | `[−0.0015, +0.0015]` |

Both absorption and injection remain available in both modes, bounded by
`|qnet| ≤ 0.75·(pch+pdch)`.

**Honest qualification:** because this dataset's tangents are symmetric
(`min_pf = −max_pf`), the old and new φ-inequalities are *numerically
identical* here — the rewrite is inert for OP1. It is still required for
correctness: under an **asymmetric** power factor the two forms differ, and
only the new form expresses the capability region about the charging
(consumption) direction that the load-positive convention implies. This stage
therefore **demonstrates** the corrected form is right by derivation and by the
mirror-mapping argument, but **cannot empirically distinguish** it from the old
form on OP1 data.

### Constraints deliberately not modified

`ess_comp` (`sch·sdch`), SOC (`ess_soc_rule`, `ess_soc_final_rule`),
`ess_s_limit` (`sch+sdch ≤ s`) and `ess_pch_link`/`ess_pdch_link` contain no P/Q
term. The audit proved no mathematical sign dependency, so per the
specification none was touched. **SOC is unchanged by this stage**, and since it
is driven by `sch`/`sdch` — never by `pnet` — flipping `pnet` cannot alter SOC
evolution.

---

## 5. Result/output convention (§6)

Because the OP1 solve does not converge (§6 below), the exposed convention was
validated by **injecting known states** into a freshly built model and running
the *real* `network.process_results` — a test that does not depend on
convergence. Per unit: charging `pnet=+0.4·s`, `qnet=+0.25·s`; discharging
`pnet=−0.4·s`, `qnet=−0.25·s`.

| Quantity (ES 1, MW / MVAr) | model | **PRE** processed | **POST** processed |
|---|---|---|---|
| charging P | `+0.2000` | `+0.2000` (Δ 0) | `+0.2000` (Δ 0) |
| charging Q | `+0.1250` | `−0.1250` (**Δ −0.25**) | `+0.1250` (Δ 0) |
| discharging P | `−0.2000` | `−0.2000` (Δ 0) | `−0.2000` (Δ 0) |
| discharging Q | `−0.1250` | `+0.1250` (**Δ +0.25**) | `−0.1250` (Δ 0) |

Identical for ES 2.

- **PRE:** `all_match_model_exactly = False` — the reporting layer did not
  expose the model's own values; Q was inverted.
- **POST:** `all_match_model_exactly = True` — `processed_P − es_pnet = 0` and
  `processed_Q − es_qnet = 0` **exactly** (0.0, not merely within tolerance).

Required sign semantics, POST: charging → `P>0` ✓ and `Q>0` ✓; discharging →
`P<0` ✓ and `Q<0` ✓. Both units.

Note the *externally visible* P/Q values are **unchanged** by this stage — the
old code reached the same load-positive external convention through
compensation. What changed is that the exposed values are now the model's own
values, so model and report cannot drift apart. No double negation and no
residual compensation remains anywhere in the chain (rows 16–17 of the audit
table are pass-through).

---

## 6. Validation-case solve (§7 / §8) — and its limitation

Case: OP1 / `case33_3` / 2025 / Summer, two ordinary ESS (bus 23, bus 18),
`s = 0.5 MVA = 0.005 p.u.`, `e = 1.0 MWh`, `ess_model = EXACT`, solver options
unchanged (`tol 1e-5`, `acceptable_tol 1e-4`, `acceptable_iter 5`,
`linear_solver ma97`).

| | **PRE** | **POST** |
|---|---|---|
| Status / termination | `error` / `internalSolverError` | `error` / `internalSolverError` |
| Solver message | `Ipopt 3.14.18: Error in step computation.` | identical |
| Solve time | 88.2 s | 97.6 s |
| Recovery used | no — `case33_3_params.json` defines no `recovery_options` | no |
| Objective | not available | not available |
| Scenario checksum | `28411ba3…` | `28411ba3…` (identical) |

**The OP1 local SMOPF fails identically before and after the correction.** This
is a **pre-existing failure, not a regression** — it reproduces on the untouched
baseline code, with the same message and the same absence of recovery.

**Consequence, stated plainly:** because solutions are never loaded on a failed
solve, the §8 items that require a converged solution — iteration count,
objective, primal/dual infeasibility, complementarity, solved `pch`/`pdch`/
`pnet`/`qnet`, SOC trajectory, ESS nodal-balance contribution, interface P/Q/V,
and the solved raw-vs-processed comparison — **could not be produced in either
phase**. Every ESS quantity in both JSONs sits at its initialization value
(`pch = pdch = pnet = qnet = 0`, `soc = e_init = 0.005 p.u.`), and the
`raw_vs_processed` block in the solve section is therefore *vacuously* equal in
both phases. It carries no evidential weight; the §5 injection test above is
what actually establishes the reporting convention.

The IPOPT iteration summary is likewise unavailable: `case33_3_params.json`
configures no `output_file`, so no IPOPT log is written, and the only way to
surface the iteration block would be `verbose=True` — which, as noted, aborts
the process on failure. Neither was changed, per §10.

**Not established:** whether the failure is intrinsic to OP1 or an artifact of
clearing shared ESS (the isolation step from `main.py`'s documented
uncoordinated recipe). A probe comparing both configurations was launched twice
and did not return a result before this report was written. This is recorded as
an open question, not guessed at.

**Suggestive but unproven:** these units sit at `0.005 p.u.`, giving
`kappa_es = 200` — twice as weakly scaled as the shared-ESS row whose poor
conditioning P3.5-D proved causal. Whether P4.6-B2's normalization resolves this
failure is exactly the question that stage should answer; it is **not** claimed
here.

---

## 7. Required final conclusions (§11)

1. **Is `es_pnet = es_pch − es_pdch` implemented everywhere consistently?**
   Yes — single definition at `model_construction_helpers.py:622`; no other
   site defines or re-derives ordinary-ESS net power (the former re-derivation
   at `network.py:1334` was removed).
2. **Does positive `pnet` now always mean network consumption?** Yes — verified
   by test 4.1 (ΔPd `+0.002` for `pnet=+x`).
3. **Does negative `pnet` now always mean network injection?** Yes — test 4.2
   (ΔPd `−0.002` for `pnet=−x`).
4. **Does positive `qnet` always mean reactive consumption/absorption?** Yes —
   test 4.3 (ΔQd `+0.002`).
5. **Does negative `qnet` always mean reactive injection?** Yes — test 4.4
   (ΔQd `−0.002`).
6. **Are active and reactive nodal balances consistent with that convention?**
   Yes — both now `Pd += es_pnet` / `Qd += es_qnet`, matching the load
   convention already used for shared ESS in the same function.
7. **Are power-factor constraints consistent with it?** Yes by derivation and
   by the mirror-mapping argument. Qualification: with this dataset's symmetric
   power factors the corrected and original forms coincide numerically, so OP1
   cannot empirically distinguish them.
8. **Is `ess_snet_def` still physically/mathematically correct?** Yes —
   unchanged, and both terms enter squared. Measured invariant under the sign
   flip (identical body `−1.562e-06`). No `kappa_es` applied.
9. **Is SOC unchanged by this stage?** Yes — `ess_soc_rule` and
   `ess_soc_final_rule` untouched and driven only by `sch`/`sdch`.
10. **Do processed/exported values now use exactly the same convention as the
    model?** Yes — `processed_P − es_pnet = 0` and `processed_Q − es_qnet = 0`
    exactly, for both units and both directions.
11. **Any remaining ordinary-ESS legacy sign inversion anywhere?** No. The two
    compensations (`network.py:1334`, `:1335`) were the only ones; all
    downstream export paths are pass-through and were audited (rows 16–17).

```
Canonical ordinary ESS convention: positive pnet/qnet = network consumption; negative pnet/qnet = network injection.
```

---

## 8. Scope compliance (§9 / §10)

Not implemented / not modified: `kappa_es`; `ess_snet_def`; `ess_comp`; SOC;
degradation / SoH; calendar ageing; any shared-ESS equation or shared-ESS
scaling; solver tolerances, MA97, Hessian settings; ADMM parameters; CS7 years
or data; Benders/planning logic. The full planning problem was **not** run —
only the single local DSO SMOPF for OP1/`case33_3`/2025/Summer.

Commits are separated per §9: `441244e3` (OP1 data layout) and the OP1 seed +
sign correction, kept apart from any future `kappa_es` work.

**No production change beyond the five edits above is authorized by this
report, and P4.6-B2 is gated on planner review.**

```
P4.6-B1 COMPLETE — waiting for planner review
```
