# P5.15 S56 — EV pilot at 2×: result (Worker, 2026-11-24)

Authority: TASKS.md critical path step 4; frozen pilot spec `data/SRP1/Results/P515S56/pilot_ev_v1/pilot_spec_92547279.json`
(sha256 `925472791ffd…`, predictions PP1–PP8 embedded, `pilot_ev_v1_predictions.json` `9cb89ef3…`); the Planner's
pilot-evaluation task of 2026-11-24 (settlement identity tolerance ruling).

**Objective convention on every table:** F = I(x) + gross_operational_cost − TV credit, € of total discounted planning
cost (cost-breakdown schema v2; gross settlement-excluded). Differences are F(b) − F(a). Every number below is
**measured** from the committed records unless marked **projected**.

## Blocked on Planner

Nothing blocks. All eight recorded predictions held. Four items need the Planner's reading:

1. **F keeps descending after K_EV, by more than S shows** (measured; Found §5). Between K_EV = 175 and K = 300, F falls
   by 92–2,550 € per run. S, the 10-cycle range, is at most 436 € at K_EV. Over the same span the pair differences move
   by up to 2,458 € (staged − x0_i1). That is about 7× the largest V_pair at K = 300 (362 €), and just under R/4.
   e_EV = 17,494 € covers it. Every pair below stays determinate at both K_EV and 300. This is the
   CLAUDE.md "bar bounds stopping slack, not path divergence" pattern: S at K_EV does not bound the remaining descent.
   Whether to act on it is the Planner's call. I changed nothing.
2. **K_EV = 175 by the frozen rule, but `K_EV_holds_at_every_later_K` is false** (Found §4). x0_i2 rises to S = 546–681 €
   on K = 177–182. Every run holds S ≤ 500 from K = 183 through 300. The rule as frozen gives 175. I did not change it.
3. **The projection needs counts that are not declared yet** (Found §7). These are the screen ladder length L, the number
   of screen candidates, and the number of moves and confirmations per search. I give a range under stated assumptions.
4. **Collision-scan scope** (Not confirmed). This commit puts `launches/launch_00{1,2}.json` and
   `campaign_spec_s56pilot_ev_v1_l00{1,2}_*.json` into the scope of `freeze_collision_scan`, which scans tracked
   `campaign_spec_*` files and `launches/launch_*.json`. The pilot's keys are settled, stop-disabled and cap-300 keys.
   A search settled at K_EV = 175 has different keys, because the cap is in the key, so no collision is expected.
   I did not run a freeze scan to confirm that.

## Changed

| commit | content |
|---|---|
| `28ce9f08` | `p515_s56_search_driver.py`: `pricing_components`' `blocks_reconcile_with_record` **and** `contracted_equals_total` now use the Expert-adopted identity tolerance \|a − b\| ≤ max(64·u·Σ\|per-block settlement totals\|, 1e-6 €), u = 2⁻⁵³ (handback 3 ruling 2; Planner ruling). Constants `IDENTITY_TOL_*` and `identity_tolerance()`. The settlement check now records the tolerance, both absolute differences and the rule. Test: `p515_s56_search_oracle_tests.py` t2 `settlement_identity_checks` (9 new checks). |
| (this commit) | pilot evidence (`pilot_ev_v1/`, launch records, logs, eval dirs by the S56 selection rule, manifest); `pilot_ev_v1_result/` (attempt 2, pre-fix) and `pilot_ev_v1_result_r2/` (post-fix); `pilot_ev_v1_evaluate_logs/`; `p515_s56_pilot_report_eval.py` + `pilot_ev_v1_report_eval/`; the two oracle-suite runs; this report |

- **`contracted_equals_total` used the same 1e-9·max(1, |total|) form, so it now uses the same rule** (logged as asked).
  On all 7 pilot records contracted equals total exactly (difference 0.0), because deviation is 0 at one scenario.
- Configuration, R, s_share, the cap, the class rules, K_EV and the e_EV rules: **unchanged.** No production file was
  touched, and nothing was solved.

### Diagnosis of the 3–5e-7 € discrepancy (zero solves)

- **Where `interface_settlement_total` is computed:** `shared_resources_planning.py:1254–1260`, in
  `_get_operational_recourse_components`.
  - The production code first calls `_get_operational_interface_settlement_blocks` (`:892–915`; weighted per-block
    `_get_local_interface_settlement`).
  - It then computes `interface_settlement_tso = sum(TSO blocks)` and accumulates the DSO blocks per node.
  - The total is `interface_settlement_tso + sum(per-node DSO)`, returned at `:1339`.
  - The harness writes it into `evaluation_record.recourse_components`.
- **Where the mapping's sum comes from:** the harness pricing capture calls the same production function
  (`p515_s44_campaign_harness.py:2985–2990`). It stores each block's `float(value)` in `s56_pricing_terminal.json`.
  - The driver sums the list in order (`p515_s56_search_driver.py`, `pricing_components`, the `settlement_blocks` loop).
- **Cause: summation order under cancellation. It is not reduced precision.**
  - The TSO blocks sum to about −1.2952e9 € and the DSO blocks to about +1.2952e9 €. The net is 6–72 €.
  - ulp(1.3e9) = 2.4e-7 €. The list-order and production-order sums differ by 5.2e-8 to 4.7e-7 €, that is, 1–2 ulp of
    the partial sums.
  - **Re-summing the captured blocks in production's order reproduces the record's `interface_settlement_total` bitwise
    on all 7 runs.** It reproduces `interface_settlement_contracted_total` bitwise too.
  - Both sides are float64: JSON `repr` with 17 significant digits. No float32 or text-precision field is in the path.
  - So the stop condition of the ruling (reduced precision) does not apply.
- **Under the adopted rule:** Σ|blocks| ≈ 2.59e9 €, so the tolerance is 1.84e-5 €. The observed differences are at
  ratio 0.003–0.026 of it. The old 1e-9 form failed on all 7.

| run | \|list-order − record\| (€) | tolerance (€) | ratio |
|---|---|---|---|
| x0_i1 | 4.69e-7 | 1.841e-5 | 0.026 |
| x0_i2 | 3.58e-7 | 1.841e-5 | 0.019 |
| x0_i3 | 1.56e-7 | 1.841e-5 | 0.008 |
| unit_i1 | 3.28e-7 | 1.841e-5 | 0.018 |
| unit_i2 | 5.22e-8 | 1.841e-5 | 0.003 |
| unit_i3 | 1.49e-7 | 1.841e-5 | 0.008 |
| staged_i1 | 3.20e-7 | 1.841e-5 | 0.017 |

### Tests (zero solves; guard `permitted=()` verified at 0; pickle 0)

- `--zero-solve --items 2`, before the commit (HEAD `b17e1711` + the working-tree change; code hashes recorded):
  **32/32**. Output in `search_oracle_v2/zero_solve_settlement_identity/`, `zero_solve.json` `b4bdd609ab42…`.
- `--zero-solve` (all items, from `28ce9f08`): **119/119** (110 before + 9 new); stderr empty. Output in
  `search_oracle_v2/zero_solve_settlement_identity_full/`, `zero_solve.json` `a4b76502c7bc…`.
- The new checks:
  - the tolerance values;
  - a SYNTHETIC cancelling fixture where the production-order and list-order sums differ: the old form fails it and the
    new form passes it;
  - a record offset and a contracted offset of 2 × tolerance each give harness_error;
  - the 1e-6 € floor: 0.9e-6 passes and 1.1e-6 fails;
  - on the 7 real pilot records, the production-order sum equals the record bitwise (total and contracted), and the old
    form fails on all 7 while the new one holds.
- I moved the two runs' stdout/stderr logs into their run dirs (`run_stdout.log`, `run_stderr.log`) so that each
  directory's manifest covers them. The oracle suite's earlier runs keep their logs beside the directory.

### Re-evaluation r2

- Command, from `28ce9f08`, attached, logs in `pilot_ev_v1_evaluate_logs/attempt3_*`:
  `p515_s56_search_driver.py --pilot-evaluate --root data/SRP1/Results/P515S56/pilot_ev_v1 --out
  data/SRP1/Results/P515S56/pilot_ev_v1_result_r2`.
- The tool created the directory. Exit 0; stderr empty.
- `pilot_result.json` sha256 `c49ac8a0ab59…` (attempt 2, kept: `e7164ff84c71…`).
- Against attempt 2, **only** the terminal fields changed: terminal status `harness_error` → `settled`, terminal_V,
  gates_pass, leak, and `V_pair_terminal` on the 15 pairs. K_EV, e_EV, c*, F, the offsets and the timings are identical.

### Evidence

- **Prediction evaluation:** `p515_s56_pilot_report_eval.py` (zero solves, guard armed) wrote
  `pilot_ev_v1_report_eval/report_eval.json` (`04a0a67f543e…`). Its logs are
  `pilot_ev_v1_evaluate_logs/report_eval_{stdout,stderr}.log`. Every derived number in this report is defined in that
  script's docstring and code.
- **Manifests** were written by `p515_s56_memory_probe.py --manifest`, which applies the S56 commit-selection rule of
  `p515_s56_ev_harness_tests.committed_by_rule`:
  - excluded directories: `/planning_read/`, `/results/`, `/esso_capture/`, `/scratch`;
  - excluded suffixes: `.pkl`, `.xlsx`, `.pdf`, `.png`, `.nl`;
  - size limit: ≤ 1,300,000 bytes. Everything else is sha256 only.

| manifest | files | committed | sha256 |
|---|---|---|---|
| `pilot_ev_v1/manifest_sha256.json` | 6,994 | 225 (57.7 MB) | `6bc1b9f12118…` |
| `pilot_ev_v1_evaluate_logs/manifest_sha256.json` | 8 | 8 | `52c5f7dd800a…` |
| `search_oracle_v2/zero_solve_settlement_identity/manifest_sha256.json` | 3 | 3 | `6d9e9fb9dd23…` |
| `search_oracle_v2/zero_solve_settlement_identity_full/manifest_sha256.json` | 383 | 3 committed + `test_trees/` hash-only | `852463577821…` |

- **Hash-only pilot files:** `stdout_s39_D.log` and `child_stdout.log` (up to 2.2 MB each), `g_s39_D.json`,
  `interface_duals_per_cycle.jsonl`, `network_ipopt_solve_records*.jsonl`, `ess_entry_stride_baseline.jsonl`,
  `pf_entry_stride_s39_D.jsonl` (up to 192 MB), the ESSO pickles, `results/`, `esso_capture/` and `planning_read/`.
  - The mapping's inputs are all committed: `evaluation_record.json`, `per_cycle_record.jsonl`,
    `convergence_depth_tail_state.json`, `s56_cost_breakdown.json`, `s56_pricing_terminal.json` and `exit_code.txt`.
  - PP8(a)'s per-cycle walls come from `stdout_s39_D.log`, which is hash-recorded only.
- **Oracle `test_trees/` are not committed.** They are hash-recorded, as in that script's own `scratch_note`: they hold
  frozen specs carrying EV eval keys, which a later collision scan would hit. Their manifest marks them `committed: true`
  by the size rule; they were excluded by hand.

## Found

### 1. The run (measured)

- The author ran it from his terminal.
  - Block 1 (5-way): 09:32–12:10 UTC, launch wall 2.63 h.
  - Block 2 (2-way): 12:10–14:28 UTC, launch wall 2.29 h.
  - Projected: 2.50 h and ≤ 2.50 h.
- 7/7 exit 0, 300 cycles each, child stderr empty, record status `certified` with the stop disabled.
- Preflight memory available: 21.8 / 22.0 GiB against 20.0 GiB required.

### 2. Recorded predictions against outcomes

| | prediction (as frozen) | evaluated as (spec `evaluated_as`) | outcome | |
|---|---|---|---|---|
| PP1 | 300 cycles, 0 local-solve failures, c* ≤ 200 on all 7 | exit 0, cycles_run = 300 = cap, local_solve_failures 0, local_solves_ok on every row, c* exists ≤ 200 | all 7; **c* 109–143** (120, 143, 121, 126, 109, 132, 127); recomputed c* = pilot_outputs c* | **HELD** |
| PP2 | every run settled at K = 300 | `map_record(settled)` status `settled` (tail depth and would-have-certified on 291–300, guards exact) | 7/7 `settled`; not_settled_reasons empty | **HELD** |
| PP3 | S ≤ 500 € at K = 300 on all 7 | max − min of F over 291–300 | **13–172 €** (max staged_i1 171.65) | **HELD** |
| PP4 | max pair offset \|dF_exit − dF_set\| ≤ 15,000 € | pilot_outputs.max_pair_offset (dF_set at K_EV) | **8,747 €** (unit_i2 − x0_i3); at K = 300: 9,681 € (same pair) | **HELD** |
| PP5 | dF_set spread (unit − x0, 3 × 3) ≤ 2,500 € | pilot_outputs.dF_set_spread_3x3 at K_EV | **1,695 €**; at K = 300: 382 € | **HELD** |
| PP6 | loss-priced leak ≤ 25 € on unit and staged | `leak.priced_loss_eur` at K = 300 on unit_i1–i3, staged_i1 | **0.0 €** on all four. x0 is 0.0 too. Full-L price 0.0 and L 0.0 MWh; `max_simultaneous_mw` 0.0 | **HELD** |
| PP7 | peak child RSS ≤ 4.0 GiB | `peak_rss.child_python_process_ru_maxrss` / 2³⁰ | **2.77–2.86 GiB**. By ruling 7 the search spec takes max × 1.15 = **3.29 GiB** | **HELD** |
| PP8 | mean s/cycle 20–35 at 1-way, and ψ ≤ 1.3 | (a) mean of production's "Iteration k" walls over 1–300; block 1 ÷ 1.21996 (gate F κ), block 2 undeflated; (b) ψ = max over blocks of max t_exit / mean t_exit | (a) **22.85–27.06** s; (b) **ψ 1.087** (block 1), 1.083 (block 2) | **HELD** |

### 3. Per run (measured; F in €, convention above)

| run | block | ρ₀ scale | c* | F_exit | bar_exit | F(K_EV = 175) | F(300) | F(175) − F(300) | S(175) | S(300) |
|---|---|---|---|---|---|---|---|---|---|---|
| x0_i1 | 1 | 1.0 | 120 | 812,009,209 | 4,271 | 812,007,702 | 812,007,610 | 92 | 45 | 13 |
| x0_i2 | 1 | 0.5 | 143 | 812,007,661 | 2,942 | 812,008,951 | 812,007,900 | 1,051 | 436 | 30 |
| x0_i3 | 2 | 2.0 | 121 | 812,005,900 | 5,134 | 812,008,364 | 812,007,744 | 620 | 349 | 17 |
| unit_i1 | 1 | 1.0 | 126 | 811,988,276 | 1,406 | 811,986,775 | 811,985,759 | 1,016 | 45 | 17 |
| unit_i2 | 1 | 0.5 | 109 | 811,993,505 | 5,283 | 811,987,221 | 811,985,667 | 1,554 | 131 | 30 |
| unit_i3 | 2 | 2.0 | 132 | 811,982,598 | 8,677 | 811,986,871 | 811,985,701 | 1,170 | 399 | 35 |
| staged_i1 | 1 | 1.0 | 127 | 811,976,795 | 2,490 | 811,976,142 | 811,973,593 | 2,550 | 243 | 172 |

- **I(x):** x0 0; reference unit `n7_y2025_p0.25_e1` 317,957.01 €; staged plan
  `n5_y2030_p0.5_e1__n7_y2025_p0.25_e1__n7_y2030_p0.25_e0.5__n9_y2035_p0.25_e0.5` 862,548.70 €.
- **F_set at K_EV** is F(175). **F at K = 300** equals the settled summary's F and the cost-breakdown total
  (`F_equals_breakdown_total` True on all 7).

| run | PR (€) | A (€) | G (€) | leak loss (€) | gates_pass | max \|ΔP\| (kW) | peak RSS (GiB) | t_exit (h) | t_set (h) | child wall to 300 (h) | s/cycle 1–300 | s/cycle 1–c* | s/cycle c*+1–300 | PP8 1-way equiv. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x0_i1 | 5.46 | 33.34 | 0.00 | 0.0 | True | 0.072 | 2.794 | 0.935 | 1.367 | 2.364 | 27.88 | 26.74 | 28.63 | 22.85 |
| x0_i2 | −70.19 | 82.87 | 0.00 | 0.0 | True | 0.081 | 2.779 | 1.126 | 1.382 | 2.387 | 28.10 | 27.14 | 28.97 | 23.03 |
| x0_i3 | −59.76 | 67.94 | 0.00 | 0.0 | True | 0.067 | 2.768 | 0.860 | 1.239 | 2.135 | 25.17 | 24.41 | 25.69 | 25.17 |
| unit_i1 | 19.66 | 37.24 | 3.59 | 0.0 | True | 0.059 | 2.805 | 1.067 | 1.485 | 2.527 | 29.83 | 29.25 | 30.24 | 24.45 |
| unit_i2 | −36.13 | 66.32 | 3.38 | 0.0 | True | 0.055 | 2.813 | 0.938 | 1.505 | 2.559 | 30.22 | 29.57 | 30.59 | 24.77 |
| unit_i3 | −56.57 | 78.92 | 3.66 | 0.0 | True | 0.071 | 2.858 | 1.015 | 1.341 | 2.292 | 27.06 | 26.63 | 27.40 | 27.06 |
| staged_i1 | 82.05 | 106.52 | 7.84 | 0.0 | True | 0.264 | 2.848 | 1.115 | 1.539 | 2.624 | 31.02 | 30.44 | 31.46 | 25.43 |

- PR, A and G are at K = 300, from the settled mapping; they are € of total discounted planning cost.
- **Gates:** all five pass on all 7 runs; no failing gate. They are status/guards, ΔP ≤ 12.5 kW, ESSO slack ≤ 1e-5,
  closure ≤ τ_cl, and D1. ESSO slack ≤ 1.1e-15, closure ≤ 3.0e-11 MWh.
- The cycle time at tail depth is 3–7 % above the pre-c* mean on every run. The s/cycle columns are production's own
  "Iteration k: X s" lines.
- **Terminal-step-to-threshold** (fields `objective_change_ratio` and `boyd_*_ratio` of the committed
  `per_cycle_record.jsonl`):

| run | objective ratio at c* | objective ratio at 300 | max Boyd ratio at c* | max Boyd ratio at 300 |
|---|---|---|---|---|
| x0_i1 | 0.0031 | 1.5e-4 | 0.19 | 0.018 |
| x0_i2 | 0.0087 | 3.8e-5 | 0.64 | 0.21 |
| x0_i3 | 0.0166 | 2.2e-5 | 0.60 | 0.12 |
| unit_i1 | 0.0101 | 9.2e-5 | 0.35 | 0.086 |
| unit_i2 | 0.0161 | 6.3e-5 | 0.60 | 0.093 |
| unit_i3 | 0.0099 | 3.6e-5 | 0.56 | 0.13 |
| staged_i1 | 0.0051 | 2.3e-4 | 0.32 | 0.22 |

- Every run stops well inside its thresholds. The objective threshold is about 81,200 €, so (rule ten) it is far too
  loose to see the drift in §5.

### 4. K_EV and e_EV (measured)

- **K_EV = 175.** It is the smallest K ≥ max c* (143) with S(K) ≤ 500 on every run; recomputed independently, 175.
- **`K_EV_holds_at_every_later_K = false`:**
  - Only x0_i2 exceeds 500 after 175: S = 562, 630, 672, 681, 631, 546 € on K = 177–182.
  - From K = 183 through 300 every run holds S ≤ 500.
  - **Mechanism (observation, x0_i2 `per_cycle_record.jsonl`, cycles 160–194):**
    - F oscillates with a period of about 18–20 cycles and a decaying amplitude of about ±340 € around a slow descent.
      For example, F − F(175) is −519 at 162, +362 at 170 and −326 at 180.
    - The W = 10 window spans about half a period, so S(K) depends on the phase.
    - K = 175 fell in a low-range phase between two windows that cover a full swing.
  - Largest S after K_EV: x0_i2 681, unit_i3 422, unit_i2 276, staged 243, unit_i1 211, x0_i3 349, x0_i1 55 €.
- **e_EV = 17,494.05 € = 2 × max pair offset.** The terms:
  - 2 × 8,747.02 = 17,494.05 (binding);
  - dF_exit spread over 3 × 3 = 14,215.22;
  - max bar sum = 13,811.32 (unit_i3 − x0_i3).
  - Recomputed in `report_eval.json`: identical.

### 5. The 15 pairs, and the determinacy rule (measured; € of total discounted planning cost, F(b) − F(a))

- The determinacy rule is `S56_PLAN_v4_1.md` §3: determinate iff |ΔF| ≥ R = 10,000, V_pair ≤ R/4, A + A′ ≤ R/2, and all
  gates pass on both runs.
- V is captured only at K = 300 (terminal pricing), so V_pair is S + S′ + |PR − PR′| + G + G′ at 300.

| pair | dF_exit | dF_set (K_EV) | dF (300) | offset (K_EV) | offset (300) | dF_set − dF(300) | bar sum | V_pair | A + A′ | \|dF(300)\| ≥ R | determinate (300) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_i1 − x0_i1 | −20,933 | −20,928 | −21,852 | 5 | 919 | 924 | 5,677 | 48.6 | 70.6 | yes | yes |
| unit_i2 − x0_i1 | −15,704 | −20,481 | −21,944 | 4,777 | 6,240 | 1,463 | 9,555 | 88.5 | 99.7 | yes | yes |
| unit_i3 − x0_i1 | −26,611 | −20,831 | −21,910 | 5,779 | 4,701 | 1,078 | 12,948 | 114.4 | 112.3 | yes | yes |
| unit_i1 − x0_i2 | −19,385 | −22,176 | −22,141 | 2,791 | 2,756 | −35 | 4,348 | 140.7 | 120.1 | yes | yes |
| unit_i2 − x0_i2 | −14,156 | −21,730 | −22,233 | 7,573 | 8,077 | 504 | 8,225 | 97.5 | 149.2 | yes | yes |
| unit_i3 − x0_i2 | −25,063 | −22,080 | −22,199 | 2,983 | 2,864 | 120 | 11,619 | 82.6 | 161.8 | yes | yes |
| unit_i1 − x0_i3 | −17,624 | −21,589 | −21,985 | 3,965 | 4,361 | 396 | 6,540 | 117.5 | 105.2 | yes | yes |
| unit_i2 − x0_i3 | −12,395 | −21,142 | −22,077 | 8,747 | 9,681 | 934 | 10,418 | 74.3 | 134.3 | yes | yes |
| unit_i3 − x0_i3 | −23,302 | −21,493 | −22,043 | 1,809 | 1,259 | 550 | 13,811 | 59.4 | 146.9 | yes | yes |
| staged_i1 − x0_i1 | −32,414 | −31,560 | −34,018 | 854 | 1,604 | 2,458 | 6,761 | 269.5 | 139.9 | yes | yes |
| staged_i1 − x0_i2 | −30,866 | −32,809 | −34,308 | 1,942 | 3,441 | 1,499 | 5,432 | 361.7 | 189.4 | yes | yes |
| staged_i1 − x0_i3 | −29,105 | −32,221 | −34,151 | 3,116 | 5,046 | 1,930 | 7,624 | 338.5 | 174.5 | yes | yes |
| staged_i1 − unit_i1 | −11,481 | −10,632 | −12,166 | 849 | 685 | 1,534 | 3,896 | 262.8 | 143.8 | yes | yes |
| staged_i1 − unit_i2 | −16,710 | −11,079 | −12,074 | 5,631 | 4,636 | 995 | 7,773 | 331.1 | 172.8 | yes | yes |
| staged_i1 − unit_i3 | −5,803 | −10,729 | −12,108 | 4,925 | 6,305 | 1,379 | 11,167 | 357.1 | 185.4 | yes | yes |

- **(unit − x0), settled, 9 pairs:**
  - dF(300) is −21,852 to −22,233 €; spread 382 €.
  - At K_EV: −20,481 to −22,176 €; spread 1,695 €.
  - V_pair 48.6–140.7 €, A + A′ ≤ 161.8 €, gates pass. **|ΔF| ≥ R on all 9, so determinate on all 9**, at K = 300 and
    with dF at K_EV.
- **(staged − x0), settled, 3 pairs:**
  - dF(300) is −34,018 to −34,308 €; at K_EV, −31,560 to −32,809 €.
  - V_pair 269.5–361.7 €. **Determinate on all 3.**
- **(staged − unit), beside:**
  - dF(300) −12,074 to −12,166 €, determinate at 1.21× R.
  - At K_EV, −10,632 to −11,079 €. The margin over R there is 632 €, which is less than that pair's own K_EV → 300 drift
    (995–1,534 €).
- **The drift (Blocked 1):**
  - dF_set(K_EV) − dF(300) runs from −35 to +2,458 €, all but one positive. The plans with investment mostly descend
    further after K_EV: staged 2,550 €, unit 1,016–1,554 €, x0 92–1,051 €. The one negative pair is unit_i1 − x0_i2
    (1,016 against 1,051).
  - The largest drift, 2,458 €, is 6.8× the largest V_pair(300) (362 €).
  - Exit classification is unaffected: e_EV = 17,494 € covers every offset at K = 300 too (max 9,681 €).

### 6. Timing (measured)

- **t_exit = 1.008 h** (mean of 7 = overhead + walls 1..c*); range 0.860–1.126 h.
- **t_set = 1.409 h** (to K_EV = 175); range 1.239–1.539 h.
- **ψ = 1.087** (block 1), 1.083 (block 2).
- Mean s/cycle is 27.9–31.0 at 5-way (block 1) and 25.2–27.1 at 2-way (block 2). The full 300-cycle child wall is
  2.14–2.62 h.

### 7. Search wall-time projection (PROJECTED, not measured)

- **Formula** (`S56_SEARCH_DRIVER_DESIGN.md` "Count and time"; driver `project_wall_time`):
  T = κ·[ψ·t_exit·Σblocks + t_set·(confirmation batches)].
  - Inputs: κ = 1.18 (gate F) with the measured t_exit = 1.008 h, t_set = 1.409 h, ψ = 1.087.
  - Counts come from driver `count_projection`, with S_0 = 1, B = 5 and no cache hits.
- **t_exit and t_set were measured inside concurrent blocks, so they already carry contention.** κ multiplies it again.
  The κ = 1 column is the same count without it.

| case | assumption | evaluations | blocks | batches | T (h), κ 1.18 | T (h), κ 1 |
|---|---|---|---|---|---|---|
| EV screen proper, L = 0 | 3 nodes × 3 years × 2 durations = 18, exit depth | 18 | 4 | 0 | **5.2** | 4.4 |
| EV screen proper, L = 7 | 18 + a 7-point ladder | 25 | 5 | 0 | **6.5** | 5.5 |
| per screen candidate | 1 settled confirmation + a complete poll at \|N\| = 39 | 40 | 8 | 1 | **12.0** | 10.2 |
| search at \|N\| = 39, low | no accepted move; start + complete final poll | 40 | 9 | 0 | **11.6** | 9.9 |
| search at \|N\| = 39, mid | 3 moves, each improver in the first block (E_j = 5), 1 confirmation each | 58 | 12 | 3 | **20.5** | 17.4 |
| search at \|N\| = 39, high | 6 moves, each improver last (E_j = 39), 2 confirmations each, final 39 + 2 | 288 | 57 | 14 | **96.9** | 82.2 |

- **One search at the median |N| projects at about 12–97 h** (κ 1.18), most likely toward the low-to-mid end if
  improvers come early.
- Plan v4.1 §8 projected 15–50 h per EV search.
- **The screen projects at 5–6.5 h** plus about 12 h per screen candidate.
- L, the number of candidates, M, p_j and c_j are assumptions, not declared values.

## Not confirmed

- **The summation-order cause** is shown on the captured float64 block values (bitwise reproduction). It was not
  re-derived from the Pyomo models, which would mean unpickling the ESSO and loading the models, which this task did not
  allow. Searched: `shared_resources_planning.py:892–915, 1245–1345`; `p515_s44_campaign_harness.py:2755–3010`; driver
  `pricing_components`.
- **Collision scan after this commit:** I did not run `freeze_collision_scan` against the newly tracked
  `launch_00N.json` and `campaign_spec_s56pilot_*`. The no-collision expectation rests on the cap being in the eval key
  (K_EV 175 against 300), not on a scan.
- **The projection's counts** are assumptions (§7). The screen ladder length, the candidate count and the moves per
  search are not declared anywhere I searched: `S56_PLAN_v4_1.md` §4 and §8, `S56_SEARCH_DRIVER_DESIGN.md`, TASKS.md
  critical path.
- **PP8(a)'s block-1 deflator** (1.21996) is gate F's 5-way contention on cycles 1–3. Tail-depth contention was not
  measured separately, so the "1-way equivalent" for block 1 is approximate.
- **The leak is exactly 0.0 on all 7.** I read `s56_cost_breakdown.json` `leak_detector`: L, the priced values and
  `max_simultaneous_mw` are 0.0, with per-node zeros. On x0 that is trivially so, with no cohorts. I did not inspect the
  per-cohort pch/pdch series behind it.
- **PP8(a)'s per-cycle walls are not committed.** They come from `stdout_s39_D.log`, which is hash-recorded only.
- **The F oscillation** of x0_i2 (§4) was read directly from its committed `per_cycle_record.jsonl`, cycles 160–194. No
  fit was made, and the period is approximate.
