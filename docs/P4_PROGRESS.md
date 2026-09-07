# P4 Progress Log

Detailed status of Stage P4 (`LOCAL_NLP_STABILITY_PLAN.md`), structured to
mirror the plan's §10 "Required worker report" sections A–F, so the final
consolidated report can be assembled directly from this log once P4.6 is
done. This log was written retroactively from the session that executed
P4.1–P4.5; treat the stage report files it references as the primary
source of truth, and this file as the index/summary tying them together.

---

## A. P4.1 — Capacity/model/dual lifecycle audit

Findings (validated empirically by P4.3's adversarial dual-injection
tests, not just inspected):

1. **Scale source.** Installed shared-ESS power rating (`S_scale_e`)
   enters the network-side SMOPF via `model.shared_es_s_rated_fixed[e]`,
   set by `configure_shared_ess_operational_state()` in
   `model_construction_helpers.py`.
2. **Model rebuild/reuse.** TSO and DSO SMOPF models are **reused**, not
   rebuilt, across ADMM cycles and across planning candidates.
   `configure_shared_ess_operational_state()` is the single production
   choke point where a capacity change is applied to a live model.
3. **Constraint dual lifecycle.** `sess_snet_def` `ConstraintData` objects
   are never deactivated-and-recreated — only `.activate()`/`.deactivate()`
   — and `model.dual` (an IMPORT/EXPORT `Suffix`) entries persist on those
   same objects across capacity updates on a live model. This means an
   imported IPOPT multiplier can survive a live kappa change and must be
   explicitly rescaled (or explicitly cleared) rather than left alone.
4. **Zero-capacity transitions.** Zero/near-zero capacity deactivates all
   `sess_snet_def` rows for that shared-ESS index and fixes all 9
   operational variable groups (`pch/pdch/sch/sdch/pnet/qnet/soc` + both
   SOC-final slacks) to zero. This existing gating is unchanged by P4 and
   is what makes the zero-capacity kappa placeholder value inconsequential
   (the row is never solved in that regime).
5. **Dual-remapping decision.** Three cases, implemented in the new
   `_sync_sess_snet_def_scale()` helper:
   - **active → active** (live kappa change on a still-active row): KKT-
     consistent transfer on the *same* live `ConstraintData` object,
     `lambda_new = lambda_old * (kappa_old / kappa_new)`.
   - **inactive → active** (reactivation): any lingering `model.dual` entry
     was not produced while the row was live under any kappa — it is
     cleared rather than transferred.
   - **active → inactive, or inactive → inactive**: left alone either way
     (excluded from the solve regardless).

## B. P4.2 — Production implementation

Files changed: `network.py`, `model_construction_helpers.py`,
`definitions.py`. Full diff is committed at `0171f451` — inspect with:

```
git show 0171f451 -- definitions.py model_construction_helpers.py network.py
```

Summary of the change:

- `definitions.py`: adds `SHARED_ESS_SNET_DEF_SAFE_KAPPA = 1.0`, the
  finite placeholder used when capacity is (near-)zero. Never affects the
  solved feasible set because that row is deactivated in that regime.
- `network.py`: adds `model.sess_snet_def_kappa`, a mutable `Param` over
  `model.shared_energy_storages`, initialized to the safe placeholder.
- `model_construction_helpers.py`:
  - `sess_snet_def_rule()` now returns `kappa_e * g == 0` instead of
    `g == 0` (where `g = (sch-sdch)^2 - pnet^2 - qnet^2`). `kappa_e` is a
    **mutable Param**, never a symbolic division by a decision variable.
  - New `shared_ess_snet_def_scale(s_capacity)`: returns
    `1/s_capacity` for positive capacity above the existing zero-capacity
    tolerance, else the safe placeholder. No division by zero is possible.
  - `configure_shared_ess_operational_state()` now captures the
    before-state (`was_active`, `kappa_old`) before applying a capacity
    change, and calls the new `_sync_sess_snet_def_scale()` helper after,
    implementing the three dual-lifecycle cases from section A above.
  - New `_sync_sess_snet_def_scale()`: the dual-lifecycle logic itself.

No symbolic division by a variable anywhere; no parallel replacement
constraint component was created — the production component name, index
set, and row identities are exactly as before. `sess_comp`, SOC, and
solver/ADMM settings are untouched.

## C. P4.3 — Construction/equivalence validation

Script: `p43_production_kappa_validation.py`. Report: `P4_3_P4_4_REPORT.md`
(P4.3 section). Raw JSON: `data/SRP1/Results/FrozenSMOPF/P43/p43_report.json`.

Builds fresh DSO (node 7, `case33_2`) and TSO (`case9`) models via the
real `SharedResourcesPlanning`/`build_model` path (not frozen pickles),
and calls the real production functions directly. Result:
`all_invariants_hold: true` for both DSO and TSO — positive-capacity
scaling (`s=0.01→κ=100`, `s=0.02→κ=50`), zero/near-zero-capacity safety,
every structural invariant (component identity, name, index tuples, row
counts, `sess_comp` untouched), the full adversarial dual-lifecycle
sequence (stale-dual injection, live capacity change with exact KKT
transfer, deactivate-leaves-dual-alone, reactivate-clears-stale-dual), and
feasible-set equivalence (`g=0 ⟺ κ·g=0`) at three sample points — all
passed with zero exceptions.

## D. P4.4 — Frozen regression

Script: `p44_production_frozen_regression.py`. Report: `P4_3_P4_4_REPORT.md`
(P4.4 section). Raw JSON: `data/SRP1/Results/FrozenSMOPF/P44/p44_report.json`.

Replays 9 frozen pre-P4 pickles (2 plan-mandated decisive cases + 5 other
preserved P3 failures + 2 bonus matched-success controls) through the
**real, unmodified current production functions**, called directly (not
reimplemented) — `sess_snet_def_rule`, `shared_ess_snet_def_scale`,
`_sync_sess_snet_def_scale`. The script enforces the plan's "required
first gate" as real control flow: it runs the 2 decisive cases first and
only proceeds to the other 7 if both pass. Result: gate passed; all 9
cases succeed (`optimal`, no recovery) with 24/24 KKT-consistent dual
transforms verified per case; the 2 bonus controls show no regression.

## E. P4.5 — Seed-2026 operational smoke

Script: `p45_seed2026_smoke_test.py` (a thin wrapper around the
unmodified production entry point — never modifies solver/ADMM
settings; only calls `run_operational_planning(type='distributed', ...)`
with the same construction `main.py` uses, and serializes diagnostics the
production code already computes). Report: `P4_5_REPORT.md`. Raw JSON/log:
`data/SRP1/Results/FrozenSMOPF/P45/p45_report.json` /
`p45_console.log`.

Candidate: `s_inv=1.00 MVA, e_inv=3.00 MVAh, node_id=7, investment_year=2025`,
seed 2026 (confirmed from `SRP1.json`). Confirmed this is **not** the full
Benders/planning loop — `run_planning_problem()` is never called, exactly
as in `main.py` (where that call is commented out).

| | P2.10 (pre-P4 baseline) | P4.5 (post-P4.2) |
|---|---|---|
| Primary local failures | 14 | 2 |
| Persistent-for-cycle failures | 7 | 2 |
| ADMM cycles to convergence | 15 | 12 |
| Runtime | 853.84 s | 672.78 s (608.65 s pure ADMM loop) |
| Convergence | Converged | Converged |

The 2 failures that did occur (DSO node 7 `case33_2`, cycles 7 and 8,
Summer and Spring days) are the same already-characterized failure mode
(`internalSolverError` on primary, `maxIterations` on recovery) and both
cleared on the very next cycle's warm start. Rho evolution stable, no
`[WARNING][RECOURSE JUMP]`/`[WARNING][SLACK COMPONENTS]` warnings, 0 of 12
voltage-slack rows active at convergence, no ESSO-level recovery needed.
Full residual/rho/recourse detail is in `P4_5_REPORT.md`.

## F. P4.6 — Standard-ESS audit/extension gate

**Not started.** Per `LOCAL_NLP_STABILITY_PLAN.md` §8: audit the standard/
ordinary ESS's analogous nonlinear apparent-power/net-power relation —
**do not assume equivalence** to the shared-ESS case just validated above.
Audit exact relation, rated-power variable/parameter semantics,
zero-capacity behavior, model rebuild/reuse lifecycle, imported
constraint-dual lifecycle, warm-start behavior, indexing, and existing
active standard-ESS cases/tests. If (and only if) an active standard-ESS
test case exists and equivalence is verified, recommend the analogous
`kappa_es * g_es = 0` normalization and validate it the same way (P4.3/P4.4
style: positive/zero-capacity checks, physical equivalence, a
representative local-solve replay). If no active test case exists, return
a design recommendation and stop — do not silently generalize the P4.2
edit to standard ESS.

## Final consolidated P4 report

**Not started** — depends on P4.6. Per §10, must cover sections A–F (this
document already covers A–E) and end with exactly one of:

```
P4 PASS — recommend planner approval for reduced planning baseline
P4 PARTIAL — planner review required before further execution
P4 FAIL — do not proceed
```

---

## File manifest (all committed at `0171f451` unless noted)

| File | Purpose |
|---|---|
| `REVISION_CONTEXT.md` | Background/history — read first |
| `LOCAL_NLP_STABILITY_PLAN.md` | Authoritative staged plan (P1–P4.6) |
| `P3_AUDIT_REPORT.md` | P3.1–P3.4 structural-conditioning audit |
| `P3_5A_REPORT.md`–`P3_5D_REPORT.md` | P3.5 A/B scaling diagnostics (A: removal; B: replacement component, reset dual; C: replacement component, transferred dual; D: in-place scaling, transferred dual — decisive) |
| `p35a_sess_snet_def_diagnostic.py`–`p35d_...py` | P3.5 diagnostic scripts (source of the `sha256_of`, `load_frozen_model`, `solve_case` helpers reused by P4.4) |
| `definitions.py`, `model_construction_helpers.py`, `network.py` | Production files carrying the P4.2 kappa-scaling fix |
| `p43_production_kappa_validation.py` | P4.3 script |
| `p44_production_frozen_regression.py` | P4.4 script |
| `p45_seed2026_smoke_test.py` | P4.5 wrapper script |
| `P4_3_P4_4_REPORT.md` | P4.3 + P4.4 stage report |
| `P4_5_REPORT.md` | P4.5 stage report |
| `data/SRP1/Results/FrozenSMOPF/P43/p43_report.json` | P4.3 raw output |
| `data/SRP1/Results/FrozenSMOPF/P44/p44_report.json` | P4.4 raw output |
| `data/SRP1/Results/FrozenSMOPF/P45/p45_report.json`, `p45_console.log` | P4.5 raw output |
| `data/SRP1/Results/FrozenSMOPF/P3Preserved/*.pkl`, top-level `FrozenSMOPF/*.pkl` | Frozen pre-P4 model snapshots used by P4.4 (**not committed** — large binaries; SHA-256 hashes recorded in `P4_3_P4_4_REPORT.md` and `P3_AUDIT_REPORT.md` are the integrity check) |
