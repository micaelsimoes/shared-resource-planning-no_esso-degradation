# Stage P4.6-B2 — Ordinary/standard ESS `ess_snet_def` normalization (executed)

Production normalization `kappa_es[e] * g_es == 0` with
`kappa_es[e] = 1 / S_rated[e]`, plus controlled OP1 A/B validation.

## Provenance

- Script: `p46b2_op1_ess_kappa_validation.py` (`--phase a|b`,
  `--construction-only`). The *same* script produced both phases.
- Raw output: `data/OP1/Results/P46B2/p46b2_phasea_report.json`,
  `p46b2_phaseb_report.json`, `p46b2_phaseb_construction_report.json`.
- **B2-A was captured before any production edit** (verified: tracked working
  tree clean at capture time, and `A_provenance.ess_snet_def_rule_source`
  records the unscaled rule verbatim).
- Case: OP1 / `case33_3` / 2025 / Summer, two ordinary ESS units, `RandomSeed`
  2026, one operation scenario, `BILINEAR_RELAXATION`, `COST` objective,
  IPOPT 3.14.18 + MA97, exact-Hessian primary with limited-memory recovery —
  all identical between phases. Scenario checksum identical:
  `824af655f515d93e755a94de04ecca1594a3f4130459d34e7bf9e1a7b8848b84`.
- B2-A reproduced the accepted B1 post-correction solve exactly
  (objective `242.643799`, 707 iterations), confirming the baseline is the
  accepted B1 state.
- Methodology: the script calls the real production `build_model` / `optimize`
  / `process_results` path and the real rule functions, and only observes.

---

## 1. Production implementation (§3)

Files changed: `definitions.py`, `model_construction_helpers.py`,
`network.py`. **+37 / −1 lines.** No shared-ESS line was touched (verified: the
full diff contains no `sess_*` or `shared_es*` change).

```diff
+ORDINARY_ESS_MIN_RATED_POWER = 1e-10   # definitions.py

+def ordinary_ess_snet_def_scale(s_rated, es_id=None):
+    if s_rated is None or s_rated <= ORDINARY_ESS_MIN_RATED_POWER:
+        raise ValueError(f'Energy Storage {es_id}: invalid ordinary ESS rated '
+                         f'apparent power s_rated={s_rated} p.u. ...')
+    return 1.0 / s_rated
+
+def ess_snet_def_scale_init(m, e, network):
+    ess = network.energy_storages[e]
+    return ordinary_ess_snet_def_scale(ess.s, ess.es_id)

 def ess_snet_def_rule(m, e, s_m, s_o, p):
     snet = m.es_sch[...] - m.es_sdch[...]
-    return snet ** 2 == m.es_pnet[...] ** 2 + m.es_qnet[...] ** 2
+    kappa = m.ess_snet_def_scale[e]
+    g = snet ** 2 - m.es_pnet[...] ** 2 - m.es_qnet[...] ** 2
+    return kappa * g == 0

 # network.py, inside `if params.es_reg:`
+model.ess_snet_def_scale = pe.Param(model.energy_storages,
+    initialize=partial(ess_snet_def_scale_init, network=network),
+    within=pe.PositiveReals)
```

Design notes:

- `ess_snet_def_scale` is an **immutable** `Param` over `model.energy_storages`
  — build-time data, not a model object. Confirmed empirically:
  `model.ess_snet_def_scale[e]` resolves to a plain Python `float`
  (`scale_is_plain_numeric_data = True`, `scale_is_mutable = False`).
- The existing `ess_snet_def` component is modified in place. **No replacement
  component** was created (`has_replacement_component = False`; the set of
  constraint component names is identical between phases —
  `components added in B: none`).
- **No `_sync_sess_snet_def_scale` port and no ordinary-ESS dual-remapping
  machinery** — per the P4.6 audit, ordinary-ESS rated power is a
  construction-time Python value that cannot change on a live model, so the
  `lambda_new = lambda_old * (kappa_old / kappa_new)` rule is not required.
  Verified: `no_ordinary_dual_remapping_helper = True`, and the shared-ESS sync
  helper remains shared-only.

### Zero / near-zero rating policy (§4)

Ordinary ESS has no zero-capacity gating, so a placeholder `kappa_es = 1.0`
would leave a *live, degenerate* nonlinear row — unlike shared ESS, where the
row is deactivated in that regime. The production rule is therefore: an
instantiated ordinary ESS **must** have `s_rated > 1e-10 p.u.`, and anything
else fails at construction with a descriptive error. Networks with **no**
ordinary ESS continue to build normally. No activation/deactivation machinery
was introduced.

---

## 2. Construction validation (§5)

| Check | Result |
|---|---|
| `S_rated` both units | `0.005 p.u.` (0.5 MVA, baseMVA 100) ✓ |
| `kappa_es` both units | **`200.0`**, matches `1/S_rated` ✓ |
| Scale is fixed numerical data | immutable `Param` → plain `float` ✓ |
| Scale is not a variable | ✓ |
| No symbolic division by a decision variable | `body_contains_division_by_variable = False` ✓ |
| Existing `ess_snet_def` component used | `component_local_name = ess_snet_def` ✓ |
| No `ess_snet_def_scaled` / replacement | ✓ (`components added in B: none`) |
| Row count | 48 → 48 ✓ |
| Index tuples | identical ✓ |
| Whole-model constraint-data count | **7921 → 7921** ✓ |
| `ess_comp` rows | 48 → 48, rule untouched ✓ |
| `ess_s_limit` rows | 48 → 48, rule untouched ✓ |
| `ess_soc_def` rows | 48 → 48, rule untouched ✓ |
| `ess_pnet_def` rows (B1 sign rule) | 48 → 48, rule untouched ✓ |
| Ordinary-ESS result processing | untouched ✓ |
| Shared-ESS equations / scaling | untouched ✓ (no `sess_*` line in the diff) |

### Feasible-set equivalence and sign preservation

`kappa_es = 200 > 0`, evaluated on the real constraint body:

| Point | `g_es` | constraint body | `body = κ·g` | zero ⟺ zero | sign preserved |
|---|---|---|---|---|---|
| on-surface | `+0.000000e+00` | `+0.000000e+00` | ✓ | ✓ | ✓ |
| positive residual | `+3.000000e-06` | `+6.000000e-04` | ✓ (×200) | ✓ | ✓ |
| negative residual | `−3.000000e-06` | `−6.000000e-04` | ✓ (×200) | ✓ | ✓ |

`g_es = 0 ⟺ kappa_es · g_es = 0`, and because `kappa_es > 0` the sign of any
nonzero residual is preserved exactly.

---

## 3. Zero-capacity construction tests (§6)

Performed on **deep copies of the in-memory network object**; no production
data file was modified.

| Test | Required | Result |
|---|---|---|
| **A** — network with empty ordinary-ESS set | constructs normally, 0 rows, no scaling exception | **constructed = True, `ess_snet_def` rows = 0, exception = None** ✓ |
| **B1** — explicit `s_rated = 0.0` | rejected cleanly, no divide-by-zero, no degenerate row | **rejected = True**, `ValueError` ✓ |
| **B2** — explicit `s_rated = 1e-14` (below tolerance) | same | **rejected = True**, `ValueError` ✓ |

Error text (both cases identifies the unit and the invalid quantity):

```
Energy Storage 1: invalid ordinary ESS rated apparent power s_rated=0.0 p.u.
An instantiated ordinary energy storage must have strictly positive rated
apparent power greater than 1e-10 p.u. A network with no ordinary energy
storage is fine; a zero-rated one is not.
```

Rejection happens during `Param` construction, i.e. **before** any nonlinear
row is built — no divide-by-zero is ever evaluated.

---

## 4. A/B comparison (§7 / §8)

**A** = B1-corrected sign convention + **unscaled** `ess_snet_def`
**B** = B1-corrected sign convention + **kappa-scaled** `ess_snet_def`

| Metric | **A unscaled** | **B scaled** |
|---|---|---|
| Status / termination | `ok` / **optimal** | `ok` / **optimal** |
| IPOPT exit | Optimal Solution Found | Optimal Solution Found |
| **Primary iterations** | **707** | **331** (−53 %) |
| Recovery | no | no |
| Objective | 242.643799 | 242.540313 |
| Primal infeasibility (scaled) | 1.598e-09 | 6.954e-09 |
| Dual infeasibility (scaled) | 5.429e-06 | **1.986e-06** |
| Complementarity (scaled) | 9.091e-09 | 9.362e-09 |
| Overall NLP error | 5.429e-06 | **1.986e-06** |
| **Runtime** | 9.00 s | **5.16 s** (−43 %) |
| **Max ｜g_es｜/S_rated² — ES 1** | 1.045e-09 | **8.523e-12** |
| **Max ｜g_es｜/S_rated² — ES 2** | 5.207e-09 | **4.068e-10** |
| Max ｜scaled row body｜ — ES 1 | 2.613e-14 | 4.262e-14 |
| Max ｜scaled row body｜ — ES 2 | 1.302e-13 | 2.034e-12 |

The **original, unscaled physical equality** is satisfied far inside normal
feasibility tolerance in both phases, and is **1–2 orders of magnitude tighter**
under scaling (worst case `4.07e-10` vs `5.21e-09`, against
`EQUALITY_TOLERANCE = 1e-5`).

### Physical / economic differences

| Quantity | max ｜B − A｜ |
|---|---|
| ES 1 `pch` / `pdch` / `pnet` / `qnet` / `soc` | 4.96e-03 / 3.77e-03 / 8.74e-03 / 2.70e-03 / 8.02e-03 |
| ES 2 `pch` / `pdch` / `pnet` / `qnet` / `soc` | 5.00e-03 / 3.92e-03 / 8.73e-03 / 2.94e-03 / 7.22e-03 |
| Interface `v` | 1.364e-09 p.u. |
| Interface `p` | 2.502e-02 MW (≈0.04 % of ≈62 MW) |
| Interface `q` | 3.686e-01 MVAr |
| Objective | 1.035e-01 (**0.043 %**) |

**Interpretation.** The interface and objective move only in the fourth/fifth
significant figure, and the objective moves **down** (242.6438 → 242.5403) on a
`COST`-minimisation problem — i.e. B found a *slightly better* local solution,
not a worse one. There is no economic regression.

The ESS dispatch differences (up to ≈5e-03 p.u. on units rated 5e-03 p.u.) are
larger in relative terms, and are stated plainly rather than minimised: ESS
arbitrage in this case is near-degenerate — many dispatch profiles are almost
cost-equivalent — so a nonconvex NLP that halves its iteration count will land
on a visibly different, near-equivalent schedule. This is **not** evidence of a
formulation change, because:

- the reformulation is a **constant positive multiple** of the same row, so the
  feasible set is provably identical (verified numerically in §2);
- every structural invariant is unchanged, including the whole-model constraint
  count (7921 = 7921) and the constraint component name set;
- the original physical equality is satisfied *more* tightly in B;
- both phases terminate `optimal` on the primary exact-Hessian path with no
  recovery.

Per §8 this is a no-regression + conditioning-consistency result, not a
failure-repair test — OP1 has no known ordinary-ESS failure snapshot.

---

## 5. Sign-convention re-confirmation (§9)

Measured on the solved B model, all 24 periods, both units:

| Check | ES 1 | ES 2 |
|---|---|---|
| `max｜processed_P − es_pnet·baseMVA｜` | **0.00e+00** | **0.00e+00** |
| `max｜processed_Q − es_qnet·baseMVA｜` | **0.00e+00** | **0.00e+00** |
| Charging periods (`pnet > 0`) | 5 | 7 |
| …all report **positive** P | ✓ | ✓ |
| Discharging periods (`pnet < 0`) | 19 | 17 |
| …all report **negative** P | ✓ | ✓ |

`ess_pnet_rule` is unchanged (`es_pnet == es_pch − es_pdch`), the active and
reactive nodal balances still use the B1 load-positive contribution
(`Pd += es_pnet`, `Qd += es_qnet`), and result processing still returns the
model's own values. **The normalization did not alter sign semantics.**

---

## 6. Required final conclusions (§13)

1. **Is `kappa_es[e] = 1/S_rated[e]` a fixed per-unit quantity?** Yes —
   immutable `Param` over `model.energy_storages`, resolving to a plain float.
2. **Do both OP1 units produce `kappa_es = 200`?** Yes — `200.0` for both,
   matching `1/0.005`.
3. **Is the original `ess_snet_def` component retained without replacement?**
   Yes — same component name, same 48 rows, identical index tuples, no added
   constraint component.
4. **Is the feasible set mathematically unchanged?** Yes — constant positive
   multiple; `g_es = 0 ⟺ κ·g_es = 0` verified on on-surface, positive- and
   negative-residual points, with sign preserved.
5. **Are zero/near-zero explicit ratings rejected safely?** Yes — both `0.0`
   and `1e-14` rejected at construction with a descriptive `ValueError`, before
   any row is built; no divide-by-zero.
6. **Do networks with no ordinary ESS still construct normally?** Yes — builds
   fine, 0 `ess_snet_def` rows, no exception.
7. **Is any dual-remapping machinery present?** **No** — as the P4.6 audit
   predicted; no ordinary-ESS sync helper exists and the shared-ESS one remains
   shared-only.
8. **Is the B1 load-positive P/Q convention unchanged?** Yes — §5 above.
9. **Is SOC unchanged?** Yes — `ess_soc_rule` / `ess_soc_final_rule` untouched,
   48 rows both phases.
10. **Is `ess_comp` unchanged?** Yes — untouched, 48 rows both phases.
11. **Does the scaled OP1 case solve cleanly?** Yes — `optimal` on the primary
    exact-Hessian path, no recovery, in 331 iterations / 5.16 s (vs 707 / 9.00 s).
12. **Is the original unscaled physical equality satisfied within tolerance?**
    Yes — max `|g_es|/S_rated² = 4.068e-10`, far inside `EQUALITY_TOLERANCE = 1e-5`,
    and tighter than the unscaled baseline.
13. **Are processed/exported P/Q still identical in convention and value to the
    model?** Yes — difference exactly `0.00e+00` for both P and Q, both units.
14. **Did any unrelated production behaviour change?** No — 3 files, +37/−1
    lines, all ordinary-ESS scoped; no shared-ESS, solver, ADMM, SOC,
    degradation, `ess_comp`, or planning change.

---

## 7. Scope compliance (§10 / §11) and deferred item (§12)

Not modified: OP1 validation parameter alignment (kept as accepted); the B1
sign convention; ordinary ESS SOC; `ess_comp`; degradation/SoH; calendar
ageing; shared-ESS equations and shared-ESS scaling; IPOPT/MA97/Hessian
settings; ADMM settings; CS7; Benders/planning logic. The full planning problem
was **not** run — only the single local DSO SMOPF.

**Deferred configuration review (recorded, not investigated):** determine later
whether the SRP1-aligned OP1 operational parameters should become the permanent
OP1 baseline, or whether a separate OP1-specific stable parameterization should
be restored or designed. Not a blocker for this stage.

```
P4.6-B2 PASS — ordinary ESS normalization validated — waiting for planner review
```
