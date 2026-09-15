# P5.15 Step 3.0 — Baseline determinism: PASS (bitwise)

**Planner report, 2026-09-15.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 9 item 3.0. Run: `s30` at `bb17845e`
(production unchanged since `92e3fafe`: ε = 1e-5, ESSO tol 1e-10 / 1e-9, explicit recovery with tier 2), one background
run through the tool, stderr captured, exclusive lock, per-cycle heartbeat, **Python exit code 0**.

## Result

The Step 3.0 repeat is **bitwise identical** to the new-baseline G1 re-run (`P515G1B`, committed `d8cf2dcc`).

| compared | entries | differences |
|---|---|---|
| cycles run / converged at / recourse / terminal step / rule ten / local failures / solves | 7 | **0** |
| per-cycle trajectory: recourse, gross cost, objective step and tolerance, V/PF/ESS primal and dual residuals (max and mean), local-solve and convergence flags — 71 cycles | 1,278 | **0** |
| ESSO `es_soh_per_unit_cumul`, `es_D_per_unit`, `es_avg_ch_dch_per_unit`, all nodes and cohort-years | 81 | **0** |
| per ESSO solve: measured detector, terminal `lg(mu)`, `s_obj`, spurious throughput — 216 solves | 864 | **0** |
| network-failure classes by network, year, day, cycle | 22 events | **0** |

Run values (identical in both): 71 cycles, recourse **817,520,272.93** (`gross_operational_cost`), rule ten 0.847,
3,694 solves, zero local-solve failures, shared `FrozenSMOPF` untouched.

**Verdict: the new baseline is deterministic.** It is the reference every Step 3 change is measured against. This
closes open item 2 of the Step 1 closing report (G4 had been bitwise only on the ε = 1e-3 baseline).

Pickle hashes are not compared; pickled Pyomo models are not byte-stable across numerically identical runs.

## Evidence

- Comparison: `p515_s30_determinism_compare.py` → `data/SRP1/Results/P515S30/s30_determinism_comparison.json`
- Run evidence and full sha256 manifest: `data/SRP1/Results/P515S30/` (`evidence_manifest_sha256.json`; per-period
  capture hash-recorded, not committed). Key hashes (first 16): `g_baseline_rep.json` `6873b7a376dd9466`,
  `stdout_baseline_rep.log` `9ebb0ab9d2307876`, `leak_classification_baseline_rep.jsonl` `d3fac40cc8cfe1b3`.

## Step 3 status

| item | state |
|---|---|
| 3.0 baseline determinism | **PASS** |
| 3.1 penalty classification | **draft committed** (`P5_15_S31_PENALTY_TABLE_DRAFT.md`, `dc923f4f`) — awaits expert review and author signature |
| 3.2–3.5 | **not started** — wait on the signed table |
