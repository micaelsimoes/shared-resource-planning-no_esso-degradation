# Worker Report — P5.15 Addendum 21 item (3): node 7 TSO-DSO interface, zero-solve look

## Task received

Planner task: a zero-solve look at the node 7 TSO-DSO interface (active power P, year
2030) at the stop of v9 arms A (`data/SRP1/Results/P515S38_A_TAU0_run/`, stop cycle 133)
and B (`data/SRP1/Results/P515S38_B_PFBAL_run/`, stop cycle 151), where the late PF dual
residual concentrates (`P5_15_ADDENDUM20_EXPERT_REPORT.md` §3.4, `P5_15_S38_PF_PACE_REPORT.md`
§3.3). Instance: C\* (0.96875 MVA / 3.875 MWh per node 5/7/9, invested 2025).

Deliverables required: (1) an inventory of which saved artifacts can answer each question,
with negative claims scoped; (2) per-node/year/day, at the stop and over the last ~30
cycles, the PF consensus decomposition, interface-flow utilization, voltage vs bounds,
flexibility usage vs bounds, storage dispatch vs rating/SoC/SoH; (3) a summary of which of
(a) interface near rating, (b) voltage bound active, (c) flexibility bound active, (d)
storage at/cycling a bound, (e) none observable — is supported, per arm. No solves, no
production/case-file edits, no algorithm-change proposal.

## Files inspected

- `REVISION_CONTEXT.md`, `LOCAL_NLP_STABILITY_PLAN.md` (repo-wide state; neither yet
  contains a P5.15 section — P5.15 state lives in the addendum/report files and the frozen
  specs under `data/SRP1/Results/P515S38/` and `data/SRP1/Results/P515S39/`).
- `P5_15_ADDENDUM20_EXPERT_REPORT.md`, `P5_15_S38_PF_PACE_REPORT.md` (task context).
- `data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json` (Addendum 21 spec;
  confirms this look is listed under `parallel_work`, "extended to C and D when available").
- Per arm, in `P515S38_A_TAU0_run/` and `P515S38_B_PFBAL_run/`: `boyd_terminal.json`,
  `component_levels_terminal.json`, `interface_voltage_terminal.json`,
  `interface_settlement_detail_s31c.json`, `g_<label>.json`, `pf_entry_stride_<label>.jsonl`,
  `ess_entry_stride_baseline.jsonl`, `soh_floor_sidecar_baseline.jsonl`,
  `esso_models_<label>.pkl`, `results/FrozenSMOPF/*.pkl`, `evidence_manifest_sha256.json`.
- `data/SRP1/SRP1.json` (DSO-to-node map: node 5 → `case33_1`, node 7 → `case33_2`, node 9
  → `case33_3`), `data/SRP1/case33_2/case33_2_params.json`,
  `data/SRP1/case33_2/case33_2_2030.json`, `data/SRP1/case33_2/case33_2_operational_data.xlsx`
  (sheet `Flex`), `data/SRP1/SharedESS/SRP1_ESS_Params.json`, `data/SRP1/SharedESS/SRP1_ESS.xlsx`.
- `network.py` (`get_interface_branch_rating`), `shared_resources_planning.py`
  (`get_admm_boyd_residual_metrics`, ESS/PF channel mapping and normalization), and the
  harness capture code in `p515_g_g1_g4_admm_gates.py` (per-entry PF and ESS stride writers),
  to establish exact field semantics before trusting the numbers.
- `data/SRP1/Results/P56A/evals/p515s38_a_tau0_arm/logs/`,
  `.../p515s38_b_pfbal_arm/logs/` (IPOPT logs, read-only).
- `p513_solve_profile_guard.py`, `p515_s33e2_evaluate.py`, `p515_s31c_zero_solve_checks.py`,
  `p515_s33_e4_noise_floor.py` (conventions for `SolveProfileGuard` use, write-once outputs,
  evidence-manifest format).

## Files modified / created

- **New:** `p515_s39_node7_interface.py` (analysis script).
- **New:** `data/SRP1/Results/P515S39/node7_interface/s38_A_tau0/{node7_interface_s38_A_tau0.json,
  run_log_s38_A_tau0.txt, evidence_manifest_sha256.json}`.
- **New:** `data/SRP1/Results/P515S39/node7_interface/s38_B_pfbal/{node7_interface_s38_B_pfbal.json,
  run_log_s38_B_pfbal.txt, evidence_manifest_sha256.json}`.
- **New:** this report.
- Nothing else touched. No production file, case file, `p515_g_g1_g4_admm_gates.py`, or any
  `p515_s38_*`/other `p515_s39_*` file was edited. Confirmed via `git status`/`git diff
  --cached --stat` before each commit that only my own files were staged.

## Changes made

Wrote `p515_s39_node7_interface.py`. It takes `RUN_DIR LABEL [--dry-run]`, installs
`SolveProfileGuard(permitted=())` before touching any input (including the `esso_models_*.pkl`
unpickle), reads only already-committed per-arm artifacts, computes the sections below, and
verifies `guard.verify(0)` before writing anything. Outputs are write-once
(`data/SRP1/Results/P515S39/node7_interface/<LABEL>/`), and the manifest hash-records every
input read (including the 80–96 MB stride files) plus both outputs, without copying the
large files.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s39_node7_interface.py \
    data/SRP1/Results/P515S38_A_TAU0_run s38_A_tau0 --dry-run   # smoke test
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s39_node7_interface.py \
    data/SRP1/Results/P515S38_B_PFBAL_run s38_B_pfbal --dry-run # smoke test
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s39_node7_interface.py \
    data/SRP1/Results/P515S38_A_TAU0_run s38_A_tau0             # real run, arm A
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s39_node7_interface.py \
    data/SRP1/Results/P515S38_B_PFBAL_run s38_B_pfbal           # real run, arm B
```

Both real runs completed with `SolveProfileGuard.verify(0): OK — zero solves for the whole
script` and wrote their three output files without any pre-existing file being overwritten
(write-once check never triggered, since the output directories did not exist beforehand).

## Inventory: which artifacts answer which question (per arm; identical file set in A and B)

| question | artifact | usable? |
|---|---|---|
| PF consensus (x_DSO, z_TSO, r, s per entry, per cycle) | `pf_entry_stride_<label>.jsonl` (one JSON row per cycle, 1728 entries/row = 3 nodes × 3 years × 4 days × 2 power types × 24 periods) | yes — full trajectory |
| interface flow vs its rating | same file (`x_dso` in MW, `interface_rating` in MVA per node, constant across years for a given node in this run: node 5 = 200, node 7 = 100, node 9 = 150) | yes |
| interface voltage vs bounds | `interface_voltage_terminal.json` (864 entries = 3 nodes × 3 years × 4 days × 24 periods) | yes — **terminal cycle only**, no trajectory |
| DSO flexibility **usage** | `interface_settlement_detail_s31c.json` (`flexibility_volumes_per_dso`, `interface_reporting_detail` delta_p/q per period), `component_levels_terminal.json` (`flexibility_cost_internal` per `DSO\|node\|year\|day` block) | yes, usage only |
| DSO flexibility **bound** | searched `interface_settlement_detail_s31c.json`, `component_levels_terminal.json`, `boyd_terminal.json`, `case33_2_params.json` | **not found** — see negative claim below |
| storage dispatch (p, q) vs rating | `ess_entry_stride_baseline.jsonl` (z = consensus, x = {tso,dso,esso} per-agent, in MW, one row per cycle); rating cross-checked against `esso_models_<label>.pkl` (`es_s_rated`) and `g_<label>.json['instance']` (0.96875 MVA / 3.875 MWh, uniform, invested 2025) | yes |
| storage SoH vs floor | `soh_floor_sidecar_baseline.jsonl` (terminal row: `es_soh_per_unit_cumul`, `soh_min` = 0.5, `active` flag, per node/y_inv/y), cross-checked against the pkl's `es_soh_per_unit_cumul` Var | yes |
| storage intra-day SoC vs bound | searched `ess_entry_stride_baseline.jsonl` (power only, no energy-state field), the `esso_capture` block in `g_<label>.json` (cohort-level cumulative quantities only), and the full `pe.Var` list of the unpickled ESSO model (`es_s_rated`, `es_e_rated`, `es_pnet`, `es_qnet`, `slack_es_pnet_up/down`, `es_s_rated_per_unit`, `es_e_rated_per_unit`, `es_s_available_per_unit`, `es_e_available_per_unit`, `es_pch_per_unit`, `es_pdch_per_unit`, `es_avg_ch_dch_per_unit`, `es_soh_per_unit_cumul`, `es_D_per_unit`) | **not found** — see negative claim below |
| local-solve status at the stop | `data/SRP1/Results/P56A/evals/p515s38_{a_tau0,b_pfbal}_arm/logs/optim_log_esso_node{5,7,9}_cycle{133,151}.txt` | yes, read-only, terminal cycle only |
| `esso_models_*.pkl` | loaded (pickle only, zero-solve) to cross-check rating and SoH; no per-period SoC variable present | used |
| `results/FrozenSMOPF/*.pkl` | two isolated snapshots at cycle 7 (a rare matched-mismatch capture), not at the stop cycle | **not used** — not applicable to the stop |
| `recourse_jump_sidecar_baseline.jsonl` | system-wide recourse block deltas, not interface/node-specific | **not used** |
| `esso_recovery_events_*`, `network_failures_*`, `leak_classification_*`, `frozen_snapshots_*`, `heartbeat_*`, `stdout_*` | process/monitoring artifacts | **not used** — no interface content |

### Scoped negative claims

- **Flexibility bound.** No terminal capture artifact of either arm run records a per-node
  or per-period flexibility bound (max/min flexible load, or a delta-P/Q limit). Searched:
  `interface_settlement_detail_s31c.json` (`flexibility_volumes_per_dso` and
  `interface_reporting_detail` carry usage/activation only — `delta_p_mw`, `anchor_p_mw`,
  no limit field), `component_levels_terminal.json` (cost only), `boyd_terminal.json` (no
  flexibility-bound field), `case33_2_params.json` (no scalar flexibility-limit field). The
  underlying case-data source for a device-level flexible-load bound is
  `data/SRP1/case33_2/case33_2_operational_data.xlsx`, sheet `Flex` (network-wide min/max
  flexible load by period, season and market scenario, before the per-year growth-factor
  scaling applied elsewhere in the pipeline); reconciling that source against the per-period
  usage figures reported below (with the correct year's growth factor and market scenario)
  was **not attempted** — out of scope for this stage, which was scoped to already-captured
  terminal/stride artifacts. This is a claim about node 7's DSO (`case33_2`) specifically, in
  these two run directories; the same three JSON files and the params file were checked, not
  the full repository or other case networks.
- **Storage intra-day SoC/energy-state bound.** No per-period state-of-charge series is
  captured anywhere in either run's saved artifacts (see table above for exactly what was
  searched, including the full unpickled `pe.Var` list of the ESSO model). If a per-period
  physical energy-state variable exists at all, it would live in the DSO (`case33_2`)
  network model's own shared-ESS block for that cycle, which is not preserved at the stop
  cycle in either run (only two unrelated cycle-7 `FrozenSMOPF` snapshots are kept, and
  loading/interpreting those was out of scope here).

## Results

All figures below are computed by `p515_s39_node7_interface.py` from the arms' own saved
artifacts; JSON with full detail (per-entry series, all top-N lists, per-year breakdowns) is
in `data/SRP1/Results/P515S39/node7_interface/{s38_A_tau0,s38_B_pfbal}/node7_interface_<label>.json`.
Both arms give the same qualitative picture; A is quoted first, B in parentheses when it
differs materially.

### PF decomposition at the stop (reproduces the report's claims, computed independently)

| | A (cycle 133) | B (cycle 151) |
|---|---|---|
| P fraction of total s² | 99.90% | 99.88% |
| node 7 fraction | 93.32% | 94.44% |
| 2030 fraction | 66.66% | 61.77% |
| node 7 & P & 2030 fraction | 64.13% | 59.74% |
| capture identity check (last 30 cycles) | `identity_holds` true on all rows, max rel. error 3.5e-16 (A) / 4.3e-16 (B) | — |

Top-15 s²-contributing entries at the stop overlap **12/15** between A and B, all at node 7,
all `power_type = p`; the top single entry in both arms is `node 7, 2030, Spring, period 9`
(A: s = 2.287e-4; B: s = 2.174e-4).

**Monotonicity of the top contributing entries over the last 30 cycles:** of the 19
entries that appear in either arm's top-15/top-10, **16/19 are `monotone_decreasing`** in
`|s|` and **3/19 `oscillating`**, identically in both arms (same entries). `r` (the primal
consensus residual, x_DSO − z_TSO) is sign-stable over the window for all but one of these
entries (0 or 1 sign flip in 29 steps).

### (a) Interface flow vs rating — supported, node-7-specific, both arms

Using S_flow = sqrt(P² + Q²) from the DSO-side interface value (`x_dso`) against
`interface_rating`, at the stop cycle, over all 288 (year × day × period) combinations per
node:

| node | rating (MVA) | max utilization | periods ≥99% of rating | periods ≥95% | 2030 subset ≥99% |
|---|---|---|---|---|---|
| 5 | 200 | 0.510 | 0 | 0 | 0 |
| **7** | **100** | **1.00001** | **22 / 288** | 25 (A) / 24 (B) | 7 / 96 |
| 9 | 150 | 0.672 | 0 | 0 | 0 |

Identical in both arms to 4 significant figures. The five most-saturated node-7 periods sit
at `S_flow = 100.0005 MVA` against a 100 MVA rating — i.e. **at** the branch rating (the
0.0005 excess is consistent with the Boyd primal tolerance, not a violation). **Nodes 5 and
9 never reach 95% of their own (larger) rating in either arm.** Node 7 is the only one of
the three whose interface branch rating (100 MVA, the smallest of the three) is actually
binding.

### (b) Interface voltage vs bounds — active, but NOT node-7-specific

All three nodes sit within ~1e-5–1e-7 pu of their TSO-side voltage upper bound (1.1 pu) at
the terminal cycle, in both arms:

| node | min distance to bound (pu) | periods within 1% pu | periods within 5% pu | of 288 total |
|---|---|---|---|---|
| 5 | 3.17e-7 | 140 | 229 | |
| **7** | 3.59e-6 (A) / 3.06e-6 (B) | 177 | 243 | |
| 9 | 3.78e-7 | 173 | 260 | |

Node 5 and node 9 are **closer** to their bound than node 7 is, and node 7's within-1%/5%
counts are not distinctively larger than the other two, either in total or broken out by
year (2025/2030/2035 counts are comparable across all three nodes). **Observation: a
voltage bound is active broadly across the system, not preferentially at node 7 or in
2030.** This does not explain the node-7/2030 concentration of the PF dual residual.

### (c) Flexibility bound — usage observed, but node 7 is not the largest, and no bound artifact exists

| node | sum &#124;ΔP&#124; (MW) | max &#124;ΔP&#124; (MW) | sum &#124;ΔQ&#124; (Mvar) |
|---|---|---|---|
| 5 | 5907.2 | 73.6 | 122.0 |
| **7** | 5983.0 | **73.7** | 110.8 |
| 9 | 6936.6 | **85.2** | 160.3 |

Node 9's flexibility usage is larger than node 7's on every metric shown. Node 7's
`flexibility_cost_internal` in 2030 is nonzero and largest in Spring/Summer (22,429 / 21,654
unweighted per representative day) versus small in Autumn/Winter (953 / 1,876) — flexibility
is used at node 7 in 2030, but usage magnitude alone does not single node 7 out among the
three DSOs, and **no bound value exists in saved data to test against** (negative claim
above).

### (d) Storage dispatch vs rating — saturates, but at all three nodes, not node-7-specific

Rating (from `g_<label>.json['instance']` and confirmed by the unpickled ESSO model's
`es_s_rated`): 0.96875 MVA, uniform across nodes 5/7/9, invested 2025.

| node | max utilization (S_z/rating) | periods ≥99% | periods ≥95% | mean consensus spread (MW) | EFC/day at stop |
|---|---|---|---|---|---|
| 5 | 1.00003 | 60 (A) / 58 (B) | 64 / 61 | 7.9e-5 | 1.141 (A) |
| **7** | 1.00003 | 53 / 52 | 57 / 55 | 8.2e-5 | 1.146 (A) / 1.141 (B) |
| 9 | 1.00003 | 58 / 55 | 62 / 59 | 9.5e-5 | 1.143 (A) |

All three nodes' shared storage saturate their rated apparent power on a comparable
fraction of periods (18–21%), with tiny absolute TSO/DSO/ESSO consensus disagreement (max
≈1e-3 MW against a 0.96875 MVA rating). **Storage-at-bound is real and system-wide, not
distinctively concentrated at node 7.** SoH is not floor-active at node 7 (or, by the same
sidecar, at 5/9): `es_soh_per_unit_cumul` at the terminal cohort-year (y_inv=0, y=2, i.e.
2035 for the 2025 cohort) is 0.626 (A) against a floor of 0.5, margin 0.126, `active: false`.
Intra-day SoC vs a bound could not be checked (negative claim above).

### (e) Synthesis, both arms

Among the four observable candidates, only **(a) interface flow vs rating is
node-7-specific**: node 7's branch rating (100 MVA) is the smallest of the three DSOs' and
is the only one that the interface flow actually reaches (22/288 periods within 1% of it,
several essentially *at* it), while nodes 5 and 9 stay below 70% of their own (larger)
ratings throughout. Voltage-bound activity, flexibility usage, and storage saturation are
all observed but are comparable in magnitude across all three nodes — they do not
distinguish node 7 from nodes 5 or 9, and therefore do not by themselves explain why the PF
dual residual concentrates so heavily (93–94%) at node 7 specifically. This is consistent
with (though does not prove) the interface branch rating being the mechanism: a channel
whose consensus variable is pinned against a physical limit for a meaningful fraction of
periods is a plausible reason for its dual residual to decay more slowly than channels that
are never limit-bound. **This is an observation supported by the zero-solve evidence above,
not a proven causal mechanism** — establishing causality (e.g., that it is specifically the
rating-bound periods whose `s` dominates and decays slowest) would need the per-entry
identity already used here cross-referenced period-by-period against the utilization table,
which the JSON supports but which this report does not carry further, per the "observation,
not mechanism" framing in the Addendum 20 report this task continues.

Both arms (A and B) give quantitatively close and qualitatively identical answers on all
five points, despite differing in which PF lever is active (A: proximal term off, ρ_pf
fixed; B: PF balancing live) and in stop cycle (133 vs 151). This is evidence that the
mechanism is a property of the instance (node 7's interface rating relative to its load) and
not an artifact of either arm's ADMM tuning.

## Validation

- **Script executes correctly:** both real runs completed, wrote exactly the three declared
  output files each, no exceptions.
- **Zero-solve claim enforced, not asserted:** `SolveProfileGuard(permitted=())` installed
  before any input touched (including the pkl unpickle); `guard.verify(0)` checked before
  any output write; both runs printed `SolveProfileGuard.verify(0): OK`.
- **Capture identity cross-check:** the per-entry PF records reproduce the production
  `boyd_pf_r`/`boyd_pf_s` scalars to within 4.3e-16 relative error over the last 30 cycles of
  both arms (`identity_holds` true on every row) — this was already established by the s38
  harness itself; my script re-verifies it independently by re-reading the same field.
- **Cross-checks that passed:** ESSO-model pickled `es_s_rated`/`es_e_rated` match the
  `g_<label>.json` declared instance (0.96875 MVA / 3.875 MWh) exactly, for all three nodes,
  both arms; IPOPT terminal logs for all three ESSO nodes at the stop cycle report `EXIT:
  Optimal Solution Found.` in both arms (26/25/25 iterations, A; not separately quoted for
  B but present and read).
- **What this validates vs what it does not:** the script correctly reads and cross-checks
  already-computed artifacts (code executes correctly, diagnostic works). It does **not**
  re-solve or independently re-derive the underlying physics, and it does **not** establish
  a causal mechanism for the PF dual-residual concentration — only an association between
  node 7's rating-bound interface and the node/entries where s² concentrates, stated as such
  above.
- **Limitations:** (i) voltage, flexibility-cost and interface-flow-vs-rating are computed
  only at/derived from the terminal cycle (no voltage trajectory artifact exists to check
  monotonicity over the last 30 cycles the way the PF entries were); (ii) the IPOPT
  log directory match is a best-effort glob keyed on the label string, flagged (not silently
  skipped) if it does not resolve to exactly one directory — it resolved correctly for both
  v9 arms but is not guaranteed to for the (as yet unknown) v10 arm C/D directory-naming
  convention; (iii) the flexibility-bound reconciliation against the raw case xlsx was
  explicitly not attempted (scoped negative claim above), so (c) is reported on usage
  magnitude only, not against a bound.

## Unexpected findings

- The interface flow at node 7 does not merely approach its 100 MVA rating — in 5 of the
  top-utilization periods in both arms it sits at essentially exactly the rating
  (100.0005 MVA vs 100.0 MVA), which is the behaviour expected of an **active inequality
  constraint** (the DSO's interface branch-flow limit), not of an unconstrained optimum that
  happens to be large. This reframes the "genuine feasibility boundary" language already
  used for node 7 in `REVISION_CONTEXT.md`'s P5.14-M section (about a different, earlier
  1.00 MVA ladder run) as potentially the same underlying mechanism recurring at C\*
  capacity, though this was not the subject investigated there and I make no claim that the
  two are the same effect.
- Contrary to what the "2030 dominant" framing in the source reports might suggest in
  isolation, **voltage-bound activity is not concentrated in 2030 at node 7** — the
  within-1%/5% counts by year are comparable to 2025 and 2035, and comparable across all
  three nodes. The 2030 concentration is a property of the PF s² decomposition specifically,
  not of the voltage channel.
- Node 9, not node 7, has the largest raw flexibility-usage magnitude (`max_abs_delta_p_mw`
  85.2 vs node 7's 73.7). If a future stage does obtain flexibility *bounds*, node 9 would be
  the first place to check for a bound close to being reached, not node 7 by usage magnitude
  alone.

## Remaining issues

- The flexibility-bound reconciliation (case xlsx `Flex` sheet, with correct year
  growth-factor and market-scenario selection, against the per-period `delta_p_mw` in
  `interface_settlement_detail_s31c.json`) is not done. If the Planner wants (c) tested
  against an actual bound rather than usage magnitude alone, that is a bounded follow-up
  task, not covered here.
- Storage intra-day SoC (state of charge) cannot be checked from any saved artifact for
  either arm; if this remains a live question, it would require either an instrumented
  re-run capturing it, or unpickling/interpreting the DSO network model's shared-ESS block
  at the stop cycle (not currently preserved for either arm).
- The association reported under "(e) Synthesis" (interface-rating activity coinciding with
  the node/entries where the PF dual residual concentrates) is not tested period-by-period
  against the top-s² entry list in this report; the JSON has what is needed to do that (utilization
  table + top-entry list, both keyed by node/year/day/period) but combining them was left for
  a follow-up if the Planner wants the mechanism tightened from "observation" to "established".

## Questions for Planner

1. Is the "interface flow at its rating, node-7-specific, not shared by voltage/flexibility/storage"
   reading the one you want carried into the next report/decision, or should I run the
   period-by-period cross-reference (top-s² entries vs utilization table) to tighten it from
   association to a checked coincidence, before arms C/D?
2. Should the flexibility-bound reconciliation against the case xlsx be authorized as a
   follow-up (still zero-solve), given node 9 — not node 7 — has the larger raw usage
   magnitude?
3. Confirm the script's IPOPT-log-directory glob convention (`*{label.lower()}*arm/logs`,
   excluding `preflight`/`probe`/`__moved_aside`) will still resolve correctly once arms C
   and D are launched under v10's directory naming, or should I adjust it now on a
   best-effort basis (I did not, to avoid guessing an unconfirmed naming convention per
   CLAUDE.md's "do not guess invocation commands" instruction)?

## Evidence

- Script: `p515_s39_node7_interface.py`.
- Outputs: `data/SRP1/Results/P515S39/node7_interface/s38_A_tau0/` and
  `.../s38_B_pfbal/`, each with `node7_interface_<label>.json`, `run_log_<label>.txt`,
  `evidence_manifest_sha256.json`.
- This report: `WORKER_REPORT_S39_NODE7.md`.
