# Worker Report — P5.15 Addendum 17 step 1: zero-solve diagnostics + data-availability audit

## Task received

Bounded ZERO-SOLVE diagnostic (Planner instruction, 2026-09-16), Addendum 17 step 1:

- **PART 0**, first: audit the committed artifacts of run 1 (`P515S35_REF_run`, reference,
  Boyd stop cycle 477) and gate 3 (`P515S35_PT_run`, price-taker init, cap 150) for five
  items — storage consensus duals per agent, storage consensus z/agent copies x, TSO/DSO
  nodal prices at the storage buses, cycle-0 LMPs, and the ESSO's own local duals — reporting
  per-entry / aggregate-norm-only / absent for each, with the exact files searched.
- **PART 1**, only what PART 0 shows is computable: (a) dual-direction comparison, gate 3 at
  cycle 150 vs run 1 at cycle 477; (b) price-taker LP with run 1's terminal nodal prices;
  (c) the same LP with cycle-0 LMPs. EFC comparison table against run 1's 1.059, gate 3's
  1.184, and the market-price LP's 1.1918.
- No Pyomo/IPOPT solves; scipy LPs permitted with a declared, checked count; output to a new
  directory; script named `p515_s36_a17_diagnostics.py`; commit script + outputs + manifest +
  this report, explicit pathspec only.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 17 (lines 735-767) and Addendum 18 (768-802, read for
  context, not acted on).
- `P5_15_ADDENDUM16_EXPERT_REPORT.md` §4-§8 (price-taker implementation, gate 3 verdict,
  claims to avoid, decisions requested, evidence index).
- `P5_15_S35PT_GATE3_REPORT.md` (full).
- `data/SRP1/Results/P515S35_REF_run/`: `g_baseline.json`, `boyd_terminal.json`,
  `component_levels_terminal.json`, `interface_settlement_detail_s31c.json`,
  `interface_voltage_terminal.json`, `ess_entry_stride_baseline.jsonl`,
  `esso_models_baseline.pkl`, `esso_capture/baseline/*.jsonl`, `frozen_snapshots_baseline.jsonl`,
  `results/FrozenSMOPF/*.pkl`, `stdout_baseline.log`.
- `data/SRP1/Results/P515S35_PT_run/`: the same file set.
- `data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json` (`precycle1_capture_method`).
- `data/SRP1/Results/P515S35/preflight_ref/`, `preflight_pt/` (listed, not opened further —
  2-cycle sanity runs, no standalone/cold OPF solve found in their manifest).
- `data/SRP1/Results/P515S35/Z2/z2_floor_slackness_results.json` (market-price LP EFC values).
- `shared_resources_planning.py` (lines 2480-2579 main ADMM loop showing `dual_vars['ess']['tso'|'dso'|'esso']`
  are distinct per-agent dicts; lines 4564-4587, 5124-5525 where `dual_p_req`/`dual_q_req`
  (ESSO) and `dual_ess_p_req`/`dual_ess_q_req` (TSO, DSO) are set from those dicts immediately
  before each agent's own solve).
- `shared_ess_price_taker.py` (full — production price-taker LP interface, `cost_energy_p`
  as its only price input, no per-node nodal-price argument).
- `p513_solve_profile_guard.py` (guard mechanism used).
- `p515_s35_z2_floor_slackness.py`, `p515_s35pt_evaluate.py` (existing conventions for
  zero-solve scripts, guard usage, output/manifest structure).

## Files modified / created

- Created `p515_s36_a17_diagnostics.py` (new diagnostic script).
- Created `data/SRP1/Results/P515S36/A17_diagnostics/a17_diagnostics_results.json`.
- Created `data/SRP1/Results/P515S36/A17_diagnostics/sha256_manifest.json`.
- Created this report.
- No production, case-file, or harness files modified.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_a17_diagnostics.py
```
Exit 0. Guard result: `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`
(0 permitted, 0 blocked expected — matched). Declared scipy LP calls = 0, observed = 0
(`shared_ess_price_taker.get_lp_call_count()` before/after the whole script), because PART 0
showed (b) and (c) are not computable, so the production LP was never called with a
substituted or fabricated price. A number of ad hoc read-only `python3`/canonical-interpreter
one-liners were used first to inspect file schemas (listed above under Files inspected); none
of them solved anything (JSON/`pickle.load` reads and `pyomo.environ.value()` calls only).

## Results

### PART 0 — data-availability table

| item | run 1 (cycle 477) | gate 3 (cycle 150) |
|---|---|---|
| 1. Storage consensus duals λ, per agent, p/q | **Aggregate norm only**, every cycle 1-477, per agent (`g_baseline.json` → `cycle_trajectory[*].boyd_ess_norm_y_{tso,dso,esso}`). **Per-entry: ESSO only**, at the run's own terminal cycle (`esso_models_baseline.pkl`, `dual_p_req`/`dual_q_req` Params, 288 entries/node/power-type, nodes 5/7/9 — this is the ESSO's own copy, set from `dual_vars['ess']['esso']...` at `shared_resources_planning.py:5519-5525`). **TSO/DSO per-entry: absent** — no TSO or DSO model is ever pickled in either run's committed evidence. | Same structure, at cycle 150 (the run's actual terminal cycle; `esso_models_baseline.pkl` freezes the in-memory state at whatever cycle the run stopped, Boyd or cap). |
| 2. Storage consensus z, agent copies x | **Per-entry, every cycle** (`ess_entry_stride_baseline.jsonl`, stride=1, cycles 1-477, keys `z`/`x{tso,dso,esso}` per (node,year,day,power_type), 72 entries/cycle). Terminal cycle 477 present exactly. | **Per-entry, stride=5**, cycles {1,6,...,146}. **Terminal cycle 150 is NOT recorded** — closest is cycle 146, 4 cycles short. |
| 3. Nodal prices/LMPs (TSO node-balance duals at buses 5/7/9; DSO reference-node balance duals) | **Absent** at the terminal cycle, at any granularity, in `g_baseline.json`, `boyd_terminal.json`, `component_levels_terminal.json`, `interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json` (recursive key scan for `lmp`/`nodal_price`/`node_balance_dual`/`marginal_price`: zero hits), and in `stdout_baseline.log` (grep, zero hits). The two `results/FrozenSMOPF/matched_success_*.pkl` fixtures DO carry a populated `dual` Suffix on `node_balance_p`/`node_balance_q` (confirmed by direct inspection — e.g. TSO node-balance-p dual at index (0,0,0,0) = 12174.29), proving the underlying capture mechanism is live in production, but both are frozen at a fixed early cycle (**7**), for an unrelated regression-fixture purpose, not the terminal cycle. | Same absence; same two-fixture caveat (its own cycle-7 fixtures exist too). |
| 4. Cycle-0 LMPs (standalone/initialization OPF solves before cycle 1) | **Absent — and the solves themselves do not exist.** `pt_phase2_checks/phase2_checks_results.json`'s own `precycle1_capture_method` states the pre-cycle-1 state is captured by a monkeypatched stop that raises **before** the first DSO solve is called through — by construction, zero IPOPT solves have executed at that point in either run. Cycle 1's own first DSO solve (duals at their initial value) is the first solve; no earlier "cycle-0" network solve exists in this pipeline. | Same. |
| 5. ESSO's own duals (e.g. `energy_storage_operation_agg`) | **Per-entry, every cycle** (`esso_capture/baseline/node{5,7,9}_cycle{001..477}.jsonl` + `*_init.jsonl`, 478 files/node; each cycle file has 288 lines, one per (year,day,period); each line's `duals` list carries `{component, index, dual}` for `energy_storage_limits`, `energy_storage_operation_agg`, `energy_storage_cohort_pnet_share_h3`, `energy_storage_capacity_degradation`). These are the ESSO's own **local NLP** KKT multipliers, distinct from the consensus λ in item 1. | Same structure, 151 files/node (cycles 1-150 + init). |

**Search record (scoped, per CLAUDE.md rule five):** for items 3/4, searched — in both run
directories — `g_*.json`, `boyd_terminal.json`, `component_levels_terminal.json`,
`interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json` (recursive key
scan), `stdout_*.log` (grep), `results/FrozenSMOPF/*.pkl` (direct Pyomo inspection),
`frozen_snapshots_baseline.jsonl`; and, once (mechanism is run-independent),
`P515S35/pt_phase2_checks/phase2_checks_results.json` and `P515S35/preflight_ref/`,
`preflight_pt/` directory listings. Not searched: `esso_capture/` contents beyond the `duals`
component-name set (these are ESSO-local, not node-balance, by construction — see item 5);
raw per-solve IPOPT `.log` files referenced from `network_failures_baseline.jsonl` (these are
failure-path logs, not a systematic per-cycle capture, and were not opened). This negative
finding is scoped to exactly the paths above, not to "the repository."

### PART 1(a) — dual-direction comparison, gate 3 (c150) vs run 1 (c477)

Per-agent aggregate norms (both from `cycle_trajectory`, same ρ_ess = 0.16875 in both runs,
frozen, not at clamp — directly comparable scale):

| agent | ‖y‖ run 1 (c477) | ‖y‖ gate 3 (c150) | ratio gate3/run1 | direction determinable? |
|---|---|---|---|---|
| tso | 6.267942e-03 | 1.928906e-03 | 0.3077 | no (aggregate norm only) |
| dso | 5.340651e-03 | 1.616537e-03 | 0.3027 | no (aggregate norm only) |
| esso | 2.152421e-03 | 6.180071e-04 | 0.2872 | **yes** (per-entry, below) |

ESSO per-entry (1,728 entries = 3 nodes × 288 × {p,q}), cosine/norm-ratio/sign-agreement
between run 1's and gate 3's terminal `dual_p_req`/`dual_q_req` vectors:

| | p only (864) | q only (864) | p+q combined (1,728) |
|---|---|---|---|
| cosine | 0.5914 | 0.9165 | 0.5912 |
| norm ratio (gate3/run1) | 0.2841 | 0.8166 | 0.2863 |
| sign agreement, all entries | 0.5694 | 1.0000 | 0.7847 |
| sign agreement, excluding both \|·\|<1e-8 | 0.5583 (643/864 clean) | 1.0000 (864/864 clean) | 0.8115 (1,507/1,728 clean) |

Reading: ESSO's terminal dual direction is **partially aligned, not orthogonal and not
identical** — reactive-power duals (q) are strongly aligned (cosine 0.92, full sign
agreement); active-power duals (p) are only moderately aligned (cosine 0.59, sign agreement
~56-57%). TSO and DSO direction is **not determinable** — only the two runs' scalar norms
exist for these agents, and both are lower at gate 3 than at run 1 by almost the same factor
(~0.30-0.31), consistent with (but not proof of) all three agents' duals heading toward the
same point from a common zero start.

"Moving toward run 1" (norm trend only, since no per-entry history exists for any agent
across cycles — only ONE terminal ESSO snapshot per run): gate 3's per-agent aggregate norms
increase monotonically over cycles {100,110,113,120,130,140,145,148,149,150} for tso, dso and
esso alike (e.g. esso: 5.420e-4 → 6.180e-4), consistent with the gate-3 report's already-cited
linear build-up (5.63e-6/cycle) toward run 1's terminal norm — reported as a norm trend, not a
direction trend.

### PART 1(b) / (c) — NOT COMPUTABLE

Both are gated directly on PART 0 items 3 and 4: no per-entry (or any-granularity) TSO/DSO
node-balance dual exists for run 1's terminal cycle, and no cycle-0 LMP exists anywhere
(the solve producing one does not exist in this pipeline). The production price-taker LP
(`shared_ess_price_taker.solve_price_taker_schedule`) was **not called** with a substituted or
fabricated price series — doing so would have approximated a missing quantity. The script
reports both as `computable: false` with the exact reason and the minimal capture that would
supply each (read the already-populated `dual` Suffix on `node_balance_p`/`node_balance_q` at
the relevant cycle before the model is discarded — zero additional solves either way, a
serialization change only).

### EFC comparison table

| quantity | value |
|---|---|
| run 1, certified (c477) | 1.058952279550704 |
| gate 3, terminal (c150) | 1.1842435189565115 |
| market-price LP (Z2, `part_A_summary.efc_per_day_max_by_node`, uniform across nodes) | 1.191777063142861 |
| (b) run-1-terminal-price LP | NOT COMPUTABLE |
| (c) cycle-0-LMP LP | NOT COMPUTABLE |

Pairwise: gate3 − run1 = 0.125291; market_LP − run1 = 0.132825; market_LP − gate3 = 0.007534.
Restated caveat (verbatim from the expert report, not re-derived): run 1 is a certified
**lower** bound only (EFC/day still rising, slope halving 3.79e-4→1.77e-4/cycle over its last
77 cycles); gate 3 is a cap-stopped point still **falling** (rule ten 0.088, PF ratio 1.4); the
bracket [1.059, 1.184] is one-sided.

### Addendum 17 step 2 readiness

Step 2's mapping test ("applied to run 1's terminal prices, must reproduce run 1's terminal
storage duals per agent") **cannot be validated as specified today**: it needs run 1's
terminal per-entry nodal prices (absent, item 3) and run 1's terminal per-agent duals for all
three agents (only ESSO's is per-entry available; TSO's and DSO's are aggregate-norm only,
item 1). The minimal bounded capture that would close the gap — stated in the output JSON and
here — is (i) pickle the TSO model and each DSO model at the terminal cycle, mirroring the
existing ESSO capture, and (ii) read the already-populated `dual` Suffix on
`node_balance_p`/`node_balance_q` for those same models at the terminal cycle. Both are
read-outs of state the existing IPOPT solve already produces; neither requires an additional
solve.

## Validation

- Script executed successfully (exit 0); guard verified `permitted_solve=0, permitted_exec=0,
  blocked_solve=0, blocked_exec=0` against a declared 0; scipy LP call count verified 0 vs
  declared 0.
- `git diff --cached --name-only` (checked before commit) lists only the four files below.
- Spot-checked the script's own audit values against independent one-off inspection performed
  earlier in this session (e.g. `dual_p_req` entry counts, stride cycle lists, FrozenSMOPF
  metadata) — all match what the script recomputed and wrote.
- What was validated: the code executes correctly, and the requested diagnostics (a) and the
  data-availability audit are computed directly from committed artifacts. (b) and (c) are
  **not** computed (correctly reported as not computable) — this is not a limitation of the
  script but the documented state of the evidence base.
- Not validated / out of scope: whether ESSO's `dual_p_req`/`dual_q_req` sign convention is
  exactly the same at both runs' terminal cycles beyond the ρ_ess match already checked (both
  0.16875, frozen, not at clamp) — a full audit of `admm_esso_al_scale`/`sigma` scaling
  parity between the two runs was not performed (out of the authorized scope: "no algorithm
  changes, only what PART 0 shows is computable").

## Unexpected findings

- The `results/FrozenSMOPF/matched_success_*.pkl` regression fixtures (unrelated to this
  task's authority — `p44_production_frozen_regression.py`) demonstrate that Pyomo's `dual`
  Suffix IS populated with real node-balance duals during production solves (confirmed
  numerically, e.g. TSO node-balance-p dual ≈ 12174.29 at cycle 7, index (0,0,0,0)). The
  capture mechanism exists; it is simply not wired to persist at the terminal cycle for either
  run. This directly informs the "capture needed" statements in PART 0 items 3/4 and the step
  2 readiness section.
- The z/x consensus-entry stride file uses **stride=1 for run 1 but stride=5 for gate 3**
  (case-file/spec difference between the two arms, not a bug) — gate 3's terminal cycle 150 is
  consequently not directly present in that file; the closest recorded cycle is 146. This
  matters for any future step that wants gate 3's terminal z/x exactly, not just its terminal
  duals (which ARE available exactly, from the separately-frozen `esso_models_baseline.pkl`).
- A possible (not attempted, out of authorized scope) further zero-solve avenue for closing
  item 1's TSO/DSO gap **without new capture**: since both runs' storage duals start at zero
  and the committed z/x stride trajectory plus the per-cycle ρ_ess trajectory are both fully
  recorded for run 1 (stride=1), the ADMM dual-update rule (`shared_resources_planning.py`
  around line 6068, `y_agent = dual_vars['ess'][agent]['current'][...]`) could in principle
  let TSO's and DSO's per-entry duals be **reconstructed** by cumulative summation over
  cycles, then cross-validated against the one ESSO agent for which both the reconstruction
  and the directly-captured value exist. This was **not attempted** here: it is a nontrivial
  derived computation outside what PART 0/PART 1 were scoped to do, and an error in the exact
  update-rule/scaling would silently produce a wrong "reconstructed" dual. Flagging for the
  Planner's consideration rather than implementing it.

## Remaining issues

- PART 1(b) and (c) remain open pending a Planner-authorized capture (TSO/DSO terminal model
  pickling + node-balance dual read-out), which is itself zero-solve (a serialization change
  to code that already runs) but was not authorized as part of this task and was not
  performed.
- Addendum 17 step 2 cannot be validated per-agent until that same capture exists for TSO and
  DSO (ESSO alone is already sufficient for a per-agent-partial validation).

## Questions for Planner

1. Should the TSO/DSO terminal-model pickling + node-balance dual capture (stated as the
   minimal capture needed for items 1/3 and step 2) be authorized as the next bounded,
   zero-additional-solve task?
2. Is the possible dual-reconstruction-from-primal-trajectory avenue (Unexpected findings,
   third bullet) worth a dedicated bounded diagnostic, given it could supply TSO/DSO per-entry
   duals for run 1 (stride=1) with no new capture at all, if the exact update rule can be
   confirmed and cross-validated against ESSO's directly-captured value?
3. Given ESSO's own per-entry direction result (p: cosine 0.59, moderately aligned; q: cosine
   0.92, strongly aligned) — does this partial alignment change how Addendum 17's decision
   list (§8) should be read, ahead of any nodal-price capture?

## Evidence / paths

- Script: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_a17_diagnostics.py`
- Output: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/A17_diagnostics/a17_diagnostics_results.json`
- Manifest: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/A17_diagnostics/sha256_manifest.json`
- Reference run: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35_REF_run/`
- Gate 3 run: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35_PT_run/`
