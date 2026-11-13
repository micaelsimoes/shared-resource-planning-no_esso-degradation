# P5.15 W187 — the 1× unit-neighbourhood screen at the production exit, no ageing: 15/15 cells landed; P3 read against the outcome (2026-11-13)

**Authority:** Addendum 91 (W187 go) and its amendments 2 (ruling 2) and 3 (rulings (a), (b)); Addendum 85 ruling 4
(vii); STEP7_SIMULATION_PLAN §5.3; P3 of Addendum 89.

**Spec:** spec v1.1 `25071d2f`; screen stage spec `5cbf4a55`; production `f6c23977`; launcher `069a7b88…`; cap 500;
production exit (ten consecutive passing cycles); no settling_resettle declaration.

**Objective conventions** (on every table below):
- **Q** = `gross_operational_cost`, with the contracted interface settlement excluded (Addendum 83 ruling 2). Salvage is
  reported separately; net = gross − terminal salvage.
- **F** = I(x) + Q(x). I(x) comes from the pinned W2 table through `p515_s47_phase_b_record.Lattice`, so F(0) = Q(0).
  Salvage is excluded from F (`p515_s47_phase_b_record.OBJECTIVE_CONVENTION`, Addendum 27 item 3).

**Solves in this task: 0.** The report script ran under `SolveProfileGuard(permitted=())` and the PB parent guard. Both
were verified at 0. Pickle calls were 0 and Pyomo model builds were 0.

## Blocked on Planner

1. **What Δ is: Q alone, as ordered, or F, as the frozen comparator uses (expert; it decides the P3 verdict).**
   - **The two definitions.**
     - The task orders Δ = Q(neighbour) − Q(x0), the operational cost only.
     - The comparator frozen in every screen cell spec is "compared with Q(x = 0 at 1x on the final code) by
       max{bar sum, sigma_Q = 18449.66…} (plan 5.3; Addendum 85 ruling 4 (vii))". Plan §5.3 calls this "the search's
       acceptance criterion". That criterion is F(inc) − F(x) > max(bar_x + bar_inc, σ_Q) with F = I + Q
       (`p515_s47_phase_b_record.RESOLUTION_RULE`).
   - **Under ΔQ.** Every neighbour has lower operational cost than x = 0, by 100,545 € to 570,370 €. That is 1.04× to
     15.3× the threshold (table 2), so **P3 fails both clauses** under either threshold.
   - **Under ΔF.** No neighbour beats x = 0, and 9 (or 6) are within resolution, so **P3 holds both clauses** under
     either threshold.
   - The report records both readings and does not choose between them. Choosing is a ruling on the objective
     convention of the comparison.
2. **What b is in max{3b, 2τ} (expert).**
   - The ordered threshold uses "3 × larger band width". A production-exit cell has no rule-v6 band, k\*, k₀ or branch,
     so that threshold cannot be computed.
   - I report two forms:
     - **T_v6bar** = max(3 × larger screen bar, 2τ). This is `settling_criterion_v6.determinacy_threshold` with b read
       as the screen's recorded bar.
     - **T_frozen** = max(bar sum, σ_Q), the frozen comparator.
   - Under ΔF the two forms agree that no neighbour beats x = 0. They split the remaining 14 differently: x0 better 5
     and within 9 under T_v6bar; 8 and 6 under T_frozen.
   - The bar refinement (CLAUDE.md) applies. At production exit the bar is a local measure of stopping slack, not a
     band.
3. **For the record, not a decision.**
   - Crediting salvage (ΔF_net, informational, outside F by Addendum 27 item 3) puts all 14 within resolution under
     T_v6bar.
   - Salvage is non-zero only on the 2030 cells: 25,782.45 € per unit. On the 2025 cells it is ≤ 1e-34.

## Blocked on the author

- Push from PyCharm.

## Changed

| commit | content |
|---|---|
| `fa9f93fc` | The run outputs of all 15 cell roots, 480 files: 15 × 31, plus the 15 launch logs, including `run_x0_A_1x_launch_r2.log` (`9beb6a8b…`). The killed run's `run_x0_A_1x_launch.log` (`a7d9230b…`) was already committed and is unchanged. |
| `0ba6b90c` | `p515_s55_w187_screen_report.py` (new; sha256 `5cf2602f…`), committed before its repository run. |
| this commit | This report; `e1_screen_1x/w187_screen_report.json` (`1afc8a78…`); its sha256 inventory `w187_screen_report_manifest_sha256.json` (`7755a5a0…`), which covers 89 files: the script, the output, the log and the 86 inputs; and `w187_screen_report_launch.log` (`58115919…`). |

**Manifests before staging (`fa9f93fc`).**
- Every campaign manifest verified against disk:

  | cell root | entries |
  |---|---|
  | x0 | 435 |
  | y2025 n5+n7+n9 | 408 |
  | y2025 n5+n7 | 414 |
  | y2025 n5+n9 | 435 |
  | y2025 n5 | 438 |
  | y2025 n7+n9 | 417 |
  | y2025 n7 | 417 |
  | y2025 n9 | 447 |
  | y2030 n5+n7+n9 | 429 |
  | y2030 n5+n7 | 426 |
  | y2030 n5+n9 | 468 |
  | y2030 n5 | 438 |
  | y2030 n7+n9 | 453 |
  | y2030 n7 | 453 |
  | y2030 n9 | 426 |

  The only file on disk absent from a manifest is that manifest itself.
- Every child manifest verified in full (400–460 entries).

**Treatment of large files.** As W184 `d7d4875e` and W185, the W152 exclusion set is hash-recorded in each campaign
manifest and **not committed**:
- `esso_models_s39_D.pkl` (15);
- `pf_entry_stride_s39_D.jsonl` (15);
- `ess_entry_stride_baseline.jsonl` (15);
- `esso_capture/` (5,964 files);
- `results/` (30 FrozenSMOPF pickles).

No `certified_models.pkl` exists: the stage spec sets `persist_certified_models` false.

**Not changed:** production, the harness, the hooks, the launcher, the specs and every committed artifact. Every write
was write-once (`'x'` / `noclobber`). `TASKS.md`, `REVISION_CONTEXT.md` and the brief are untouched. Nothing was pushed.

### Execution history

1. **First launch, 2026-11-12 13:29:19 UTC** (lock `started_utc`), in the session tool's background mode.
   Amendment 2 ruling 2 accepted the mode and ruled that the cell was not to be restarted.
2. **Killed at cycle 122, last heartbeat 14:18:22 UTC**, by the Claude session restart. There was no traceback and no
   `campaign_results.json`. The record is `P5_15_W187_INTERRUPTED_REPORT.md` and commit `996cb81a` (14:21 UTC), which
   holds the sha256 inventory of the partial root.
3. **Amendment 3 moves, `0cb6512e` (14:27 UTC).** Nothing was deleted.
   - The partial `evals/` and the heartbeat moved to `w187_cell1_interruption_20261112/partial_root/` (394/394
     hash-verified).
   - The stale lock (PID 31351) moved to the same directory.
   - The P56A working dirs moved there too (420/420).
   - The preconditions-only run `_r3` held.
   - `w187_terminal_chain.sh` holds the 15 frozen `launch_command` strings joined by `&&` under `set -e`. Cell 1's log
     is named `_r2`.
4. **Terminal.app relaunch under `caffeinate -i`.**
   - Cell 1 started at 14:46:58 UTC.
   - The last cell finished on 2026-11-13 at 05:11:04 UTC: 14 h 24 min elapsed, 14.36 h of summed child time, against
     ≈ 21 h planned.
   - Every launch log ends `exit 0; guards True` and `final guards: all_zero=True pickle={'load': 0, 'loads': 0}
     build={'ConcreteModel.__init__': 0}`.
   - `git_head_at_run`: `81962edb` for cell 1 and `0a81c619` for cells 2–15. The commits in between touched only
     `manuscript_review/` scripts. The launcher's code-unchanged precondition held for every cell.

## Found

### P3 against the outcome

**P3 (Addendum 89, recorded blind):** "no no-ageing unit neighbour beats x = 0 by more than max{3b, 2τ}; at least one is
within resolution."

**"Unit neighbour" is defined**, so the 14-cell reading is used:
- the stage spec `phase_text` and Addendum 85 ruling 4 (vii): "the 14 admissible unit neighbours of x = 0 under N";
- plan §5.3: "the smallest unit at any non-empty subset of the three nodes, 2025 or 2030".

The script confirmed that the 14 screen candidates equal `p515_s47_phase_b_record.Lattice.neighbourhood(x0)` (14
points). The 6 single-node cells are reported beside the 14 for information only.

τ = `settling_criterion_v6.TAU` = 4,539.07 €; 2τ = 9,078.14 €. σ_Q = 18,449.66 €.

| reading (Δ, threshold) | neighbour better | x0 better | within resolution | clause 1 (none beats x0) | clause 2 (≥ 1 within) | single-node 6 (better / x0 / within) |
|---|---|---|---|---|---|---|
| ΔQ (as ordered), T_v6bar | **14** | 0 | 0 | **failed** | **failed** | 6 / 0 / 0 |
| ΔQ (as ordered), T_frozen | **14** | 0 | 0 | **failed** | **failed** | 6 / 0 / 0 |
| ΔF (frozen comparator's quantity), T_v6bar | 0 | 5 | 9 | **held** | **held** | 0 / 2 / 4 |
| ΔF (frozen comparator's quantity), T_frozen | 0 | 8 | 6 | **held** | **held** | 0 / 3 / 3 |

Decisions 1 and 2 decide which row is the verdict.

P3 is a margin/sign prediction. Under the ΔQ reading it fails on the margin, and the sign is opposite to the
prediction. Addendum 91 says such a failure is reported, and the paper proceeds without W187. Nothing was changed to
make it hold.

### Table 1 — the 15 cells

Q is `gross_operational_cost` with the settlement excluded. Every cell is certified at the production exit, and in
every cell the Boyd stop came 9 cycles before the last cycle (10 consecutive passing cycles).

Columns:
- **Boyd conv.** is the cycle at which the Boyd stop first held.
- **Stopping ratio** is the terminal step divided by the 1e-4 Q tolerance.
- **Band, k\*, k₀ and branch** do not apply at production exit. The screen records none of them.

| # | cell | eval key | I (€) | Q (€) | salvage (€) | bar (€) | cycles | Boyd conv. | stopping ratio | solves (= expected) | wall (h) | network failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `x0_A_1x` | `5cfe69a6` | 0.00 | 653,858,731.57 | 0.00 | 9,629.98 | 132 | 123 | 2.361e-03 | 6,783 | 0.86 | 0 |
| 2 | `y2025__n9_p0.25_e0.5_N_1x` | `c01c5f8b` | 191,018.17 | 653,679,029.47 | 0.00 | 3,560.65 | 136 | 127 | 2.191e-02 | 6,988 | 0.97 | 1 recovered tier 1 |
| 3 | `y2030__n9_p0.25_e0.5_N_1x` | `7c5afbc0` | 141,882.02 | 653,749,647.62 | 25,782.45 | 5,455.85 | 129 | 120 | 1.638e-02 | 6,630 | 0.88 | 0 |
| 4 | `y2025__n7_p0.25_e0.5_N_1x` | `a4bce87e` | 191,018.17 | 653,684,016.28 | 0.00 | 22,210.32 | 126 | 117 | 1.840e-02 | 6,478 | 0.89 | 1 recovered tier 1 |
| 5 | `y2030__n7_p0.25_e0.5_N_1x` | `8cf1c0a6` | 141,882.02 | 653,758,186.49 | 25,782.45 | 7,872.26 | 138 | 129 | 1.111e-01 | 7,090 | 0.93 | 1 recovered tier 1 |
| 6 | `y2025__n7…__n9…_N_1x` | `296fe199` | 382,036.34 | 653,493,472.24 | 0.00 | 15,539.35 | 126 | 117 | 4.374e-02 | 6,478 | 0.96 | 1 recovered tier 1 |
| 7 | `y2030__n7…__n9…_N_1x` | `383729be` | 283,764.04 | 653,622,588.89 | 51,564.91 | 9,453.80 | 138 | 129 | 4.389e-02 | 7,089 | 0.98 | 0 |
| 8 | `y2025__n5_p0.25_e0.5_N_1x` | `62daa014` | 191,018.17 | 653,673,723.29 | 0.00 | 21,031.44 | 133 | 124 | 7.527e-03 | 6,835 | 0.95 | 1 recovered tier 1 |
| 9 | `y2030__n5_p0.25_e0.5_N_1x` | `274f66d7` | 141,882.02 | 653,737,846.07 | 25,782.45 | 5,561.88 | 133 | 124 | 5.873e-03 | 6,835 | 0.91 | 1 recovered tier 1 |
| 10 | `y2025__n5…__n9…_N_1x` | `6216b355` | 382,036.34 | 653,497,759.08 | 0.00 | 9,654.69 | 132 | 123 | 1.181e-01 | 6,783 | 1.04 | 0 |
| 11 | `y2030__n5…__n9…_N_1x` | `50c908f6` | 283,764.04 | 653,671,920.45 | 51,564.91 | **59,925.54** | 143 | 134 | 5.493e-03 | 7,344 | 1.05 | 0 |
| 12 | `y2025__n5…__n7…_N_1x` | `ecf01f76` | 382,036.34 | 653,485,440.23 | 0.00 | 12,246.88 | 125 | 116 | 1.606e-02 | 6,426 | 0.96 | 0 |
| 13 | `y2030__n5…__n7…_N_1x` | `5bd7af4d` | 283,764.04 | 653,611,984.35 | 51,564.91 | 11,057.10 | 129 | 120 | 2.581e-02 | 6,630 | 0.94 | 0 |
| 14 | `y2025__n5…__n7…__n9…_N_1x` | `e943b8d6` | 573,054.51 | 653,288,361.08 | 0.00 | 12,426.06 | 123 | 114 | 2.041e-02 | 6,324 | 1.04 | 0 |
| 15 | `y2030__n5…__n7…__n9…_N_1x` | `de3aa8e2` | 425,646.07 | 653,533,669.71 | 77,347.36 | 8,362.00 | 130 | 121 | 1.188e-01 | 6,682 | 1.01 | 1 recovered tier 1 |

`…` stands for `_p0.25_e0.5`. The full eval and candidate keys are in `w187_screen_report.json` `cells`.

**Run conditions across all 15 cells:**
- 15/15 cells: launcher exit 0, `faults` and `stops` empty, guards all zero.
- Solve-profile identity: holds on 15/15. 101,395 solves in all.
- 0 local solve failures. 0 ESSO recovery events.
- Peak RSS ≤ 2.64 GB.
- The script's 15 per-cell checks were all true. Among them, the recorded bar equals the bar re-derived from
  `per_cycle_record.jsonl`, and Q equals the trajectory's last gross value.
- **ρ read-back:** not present. The screen declares no ρ₀ scale; the read-back is an E2 check. The terminal
  `rho_v_after` / `rho_pf_after` are 0.017325 / 0.132 in every cell, all frozen. `rho_ess_after` is 0.86498 at x = 0,
  0.050625 at y2025 n5, and 0.0759375 in the other 13 storage cells.
- **Signed slack content (`detector_penalty_total`):** −13,072.26 € to −13,012.57 €, the relaxation residue
  W188A describes. It is not examined further here.

### Table 2 — the 14 neighbours against x0_A_1x

Q is gross with the settlement excluded; F = I + Q; m = value(n) − value(x0), so m < 0 means the neighbour is cheaper.

Column notes:
- **ratio = |m| / threshold.**
- **T_v6bar** = max(3 × larger bar, 2τ).
- **T_frozen** = max(bar_n + bar_x0, σ_Q).
- **T_band** is not computable (no band).
- **ΔQ ratio** is shown against T_v6bar. Under ΔQ every verdict is "neighbour better" in both forms.

| neighbour | ΔQ (€) | ΔQ ratio (T_v6bar) | I(n) (€) | ΔF (€) | T_v6bar (€) | ratio | verdict | T_frozen (€) | ratio | verdict | ΔF_net (€), informational |
|---|---|---|---|---|---|---|---|---|---|---|---|
| y2025 n9 | −179,702.09 | 6.22 | 191,018.17 | +11,316.08 | 28,889.94 | 0.39 | within | 18,449.66 (σ_Q) | 0.61 | within | +11,316.08 |
| y2030 n9 | −109,083.95 | 3.78 | 141,882.02 | +32,798.07 | 28,889.94 | 1.14 | x0 better | 18,449.66 (σ_Q) | 1.78 | x0 better | +7,015.62 |
| y2025 n7 | −174,715.29 | 2.62 | 191,018.17 | +16,302.88 | 66,630.95 | 0.24 | within | 31,840.29 | 0.51 | within | +16,302.88 |
| y2030 n7 | −100,545.08 | 3.48 | 141,882.02 | +41,336.94 | 28,889.94 | 1.43 | x0 better | 18,449.66 (σ_Q) | 2.24 | x0 better | +15,554.49 |
| y2025 n7+n9 | −365,259.33 | 7.84 | 382,036.34 | +16,777.01 | 46,618.05 | 0.36 | within | 25,169.33 | 0.67 | within | +16,777.01 |
| y2030 n7+n9 | −236,142.68 | 8.17 | 283,764.04 | +47,621.37 | 28,889.94 | 1.65 | x0 better | 19,083.78 | 2.50 | x0 better | −3,943.54 |
| y2025 n5 | −185,008.28 | 2.93 | 191,018.17 | +6,009.89 | 63,094.32 | 0.10 | within | 30,661.42 | 0.20 | within | +6,009.89 |
| y2030 n5 | −120,885.50 | 4.18 | 141,882.02 | +20,996.52 | 28,889.94 | 0.73 | within | 18,449.66 (σ_Q) | 1.14 | x0 better | −4,785.93 |
| y2025 n5+n9 | −360,972.49 | 12.46 | 382,036.34 | +21,063.85 | 28,964.08 | 0.73 | within | 19,284.67 | 1.09 | x0 better | +21,063.85 |
| y2030 n5+n9 | −186,811.12 | 1.04 | 283,764.04 | +96,952.92 | 179,776.61 | 0.54 | within | 69,555.51 | 1.39 | x0 better | +45,388.02 |
| y2025 n5+n7 | −373,291.34 | 10.16 | 382,036.34 | +8,745.00 | 36,740.64 | 0.24 | within | 21,876.86 | 0.40 | within | +8,745.00 |
| y2030 n5+n7 | −246,747.22 | 7.44 | 283,764.04 | +37,016.83 | 33,171.30 | 1.12 | x0 better | 20,687.08 | 1.79 | x0 better | −14,548.08 |
| y2025 n5+n7+n9 | −570,370.49 | 15.30 | 573,054.51 | +2,684.02 | 37,278.17 | 0.07 | within | 22,056.04 | 0.12 | within | +2,684.02 |
| y2030 n5+n7+n9 | −325,061.86 | 11.25 | 425,646.07 | +100,584.21 | 28,889.94 | 3.48 | x0 better | 18,449.66 (σ_Q) | 5.45 | x0 better | +23,236.85 |

**ΔF_net = ΔF − (salvage_n − salvage_x0).** It is informational and lies outside F. Under T_v6bar every row is within
resolution.

**The pinned W2 unit costs**, discounted (€/MVA, €/MWh):
- 2025: 256,317.32 and 253,877.68;
- 2030: 190,384.09 and 188,571.99.

The I(x) closed form matches the W2 table on 105/105 candidates, with a worst difference of 2.3e-10 €.

**Notes on the resolution:**
- The smallest ΔF is +2,684.02 € (y2025 n5+n7+n9) at ratio 0.07. ΔF on y2025 n5 is +6,009.89 € (0.10). Both are
  indeterminate.
- The bar refinement (CLAUDE.md) applies to every "x0 better" verdict. It rules out stopping slack only, measured by a
  production-exit bar. It is not a statement about the limit.
- Cell 11's bar, 59,925.54 €, is 6× the median bar. That one cell sets both thresholds on its row.

### The screen's `bar_eur`, cross-checked

`outcome.screen.bar_eur` is not the rule-v6 threshold and not a band width. It is copied from
`evaluation_record.bar.value` (`p515_s55_step7_launcher.py` `cell_outcome`, l. 1508–1510). The harness defines it as
"max over the last 10 cycles of |gross_operational_cost[k] − gross_operational_cost[k−1]|". The script re-derived it
from each `per_cycle_record.jsonl` and found it equal on 15/15 cells.

It enters the frozen comparator as one term of the bar sum (`p515_s47_phase_b_record.resolution`). Under the rule-v6
form (`settling_criterion_v6.determinacy_threshold`, whose docstring reads "a bar is a certified cell's band width"), b
should be a band width. Using `bar_eur` there is a substitution, recorded as decision 2.

At x = 0 the bar is 9,629.98 €: 3 × bar is 28,889.94 € and bar + bar is 19,259.96 €.

### Table 3 — Q(x = 0, 1×) against the committed x = 0 holders (informational)

Q is gross with the settlement excluded; salvage is 0 on all three.

| holder | eval key / candidate key | head at run | protocol | Q (€) | Q(screen x0) − Q(holder) (€) | resolution |
|---|---|---|---|---|---|---|
| screen `x0_A_1x` | `5cfe69a6…` / `8435c718…` | `81962edb` (production `f6c23977`) | production exit, 132 cycles, bar 9,629.98, terminal step 154.39 | 653,858,731.57 | — | — |
| W86 tail recert (spec `ddd6cd44`) | `5cfe69a6…` (same) / `8435c718…` | `31a37efb` | production exit, 132 cycles, bar 9,629.98, terminal step 154.39 | 653,858,731.57 | **0.00 (bitwise)** | — |
| C_OPEN = E.21 row 1 (W101 `d110bd1a…`, step-6 `ref:7aa017f0`) | `d110bd1a…` / `8435c718…` | `aad37370` | rule v6 certified, k\* 181, band 4,209.17, terminal step 100.42 | 653,873,702.19 | −14,970.62 | 0.52 × max(3 × max(9,629.98, 4,209.17), 2τ) = 28,889.94; 0.81 × max(9,629.98 + 102.40, σ_Q) = 18,449.66; terminal steps 254.80 |

- **The screen's x = 0 reproduces W86 bitwise.** All 29 fields of `per_cycle_record.jsonl` are equal on all 132 rows,
  on a different code head. The screen reported W86 as a holder and did not reuse it.
- **Against C_OPEN the difference is within resolution in both forms.** It is a mixed comparison: a production-exit
  value against a v6-settled one, on different heads. Its stopping ratios are 2.36e-3 and 1.54e-3. Plan §5.3 makes row
  1 "beside, never the comparator".

### Launcher quirk (finding; not fixed)

Every cell's final `[RUN …]` line prints `k* None band None Q_end None bar None stopping ratio None`.

The `_log` line (`p515_s55_step7_launcher.py` l. 1675–1678) reads these values with `rep.get(…)` from
`outcome['cell_report']`. That key exists only for the certified phases (`CERTIFIED_PHASES = ('e3', 'e2')`). A screen
outcome carries `outcome['screen']` instead (`status`, `certified_cost`, `cycles_run`, `bar_eur`, `rule_ten`), so every
value prints None.

The status is printed correctly, through its fallback to `outcome['screen']`. The values are complete in each
`campaign_results.json`.

## Not confirmed

- **The objective convention of P3** (decision 1) and the meaning of b at production exit (decision 2). I looked in:
  - the brief: Addenda 85 (vii), 88 R3, 89 P3 and ruling 2, 90 and 91;
  - plan §5.3;
  - the stage spec and the cell specs' `comparator`;
  - `p515_s47_phase_b_record` (`OBJECTIVE_CONVENTION`, `RESOLUTION_RULE`) and `settling_criterion_v6`.

  None of them states P3's Δ explicitly as Q or as F.
- **Addendum 88 R3's "coordinate unit neighbourhood"** (±1 in one coordinate, plus activation moves) is a different
  construct, stated for the general lattice. I did not map it onto x = 0. The single-node 6 are shown for information
  only.
- **Whether any "x0 better" verdict survives settlement under rule v6** is not known. No screen cell was re-settled,
  and the production-exit bar is local (the bar refinement).
- **The ESSO-side state, the P56A working dirs and the excluded files** were not read. The working dirs are
  `P56A/evals/p515s44_s55_step7_*`, untracked and outside the cell roots; they are not hash-recorded here. The excluded
  files are hash-recorded in the campaign manifests only.
- **The kill signal of the first launch** and its sender remain as `P5_15_W187_INTERRUPTED_REPORT.md` records them.

## What the result means for §4.3 (facts only; the sentence is the expert's)

The screen evaluated the 14 admissible unit neighbours of x = 0 (0.25 MVA / 0.5 MWh at every non-empty subset of nodes
5, 7 and 9, in 2025 or 2030) under the no-ageing model at 1×, at the production exit, on production `f6c23977`. All 15
evaluations certified at the production exit. x = 0 reproduced the committed W86 value bitwise (653,858,731.57 €).

Every neighbour lowers the operational cost Q, gross with the settlement excluded, by 100.5 k€ to 570.4 k€. Every
neighbour's investment cost I is larger than that reduction, by +2.7 k€ to +100.6 k€ in F = I + Q. So no neighbour
improves F on x = 0 beyond either resolution form:
- 9 of 14 are within max(3 × larger bar, 2τ), and 5 are determinately worse;
- 6 are within max(bar sum, σ_Q), and 8 are determinately worse.

The closest are the three-node 2025 unit (+2.7 k€) and the node-5 2025 unit (+6.0 k€). Both are indeterminate.

Whether §4.3 states this on F (the frozen comparator) or on Q (as ordered for P3) is decision 1. On Q, every neighbour
is determinately cheaper to operate.

---

## Planner ruling on Blocked items 1–2 (2026-11-13; the Worker's sections above are unchanged)

**Δ is on F = I + Q, the frozen comparator. P3 is read on ΔF.** The ΔQ reading came from the Planner's task brief,
which wrote "Q(neighbour) − Q(x0)". That was the Planner's error, not the record's.

- **Plan §5.3** (`STEP7_SIMULATION_PLAN_2026-10-25.md`), frozen into every screen cell spec through Addendum 85
  ruling 4 (vii), says the fourteen neighbours are "compared with Q(0) by **the search's acceptance criterion**
  max{bar sum, σ_Q}". The question it asks is whether a neighbour "improves on x = 0" (H1d). The search's acceptance
  criterion is F(inc) − F(x) > max(bar_x + bar_inc, σ_Q), with F = I(x) + Q and salvage excluded
  (`p515_s47_phase_b_record`; Addendum 27 item 3).
- **The same section** records the reference unit's N cell as "−4.1 k€ certified". That is an F difference; its ΔQ is
  of order −180 k€.
- **ΔQ alone is negative for every non-empty investment.** Storage only adds flexibility, so ΔQ < 0 is no planning
  statement.

**Threshold.** P3 is worded max{3b, 2τ}, but production-exit cells have no rule-v6 band. The frozen screen comparator
is max{bar sum, σ_Q}, and the Worker's T_v6bar form reads b as the screen bar. **P3 holds under both forms, so no ruling
between them is needed:**

| clause | T_frozen = max{bar sum, σ_Q} (the frozen comparator) | T_v6bar = max{3 × larger bar, 2τ} |
|---|---|---|
| no unit neighbour beats x = 0 beyond the threshold | **held**: 0 of 14 | **held**: 0 of 14 |
| at least one within resolution | **held**: 6 within (8 with x = 0 determinately better) | **held**: 9 within (5 x = 0 better) |

- **ΔF.** It is positive on all 14 cells: the smallest is +2,684 € (2025, all three nodes) and the largest +100,584 €
  (2030, all three nodes). The minimum is not a near-miss: under the frozen comparator the closest row is at 0.12× its
  threshold.
- **H1d (plan §5.3) holds.** No neighbour improves on x = 0 beyond the criterion.
- **The paper's scope sentence is plan §5.3's:** "a bounded unit-neighbourhood screen at the baseline price under the
  no-ageing model … not a completed search and not a globally preferred zero investment". It is the N-side match of
  the s47 aged-arm result.

**Salvage (record).** Crediting salvage (ΔF_net) puts all 14 within resolution under T_v6bar. Salvage stays outside F
(Addendum 27 item 3; plan §6 E5), so this is a sensitivity, not the comparator.

**Planner spot-check.** Re-read from the committed `campaign_results.json` files:
- Q(x0) = 653,858,731.5686 €;
- ΔQ = −570,370.49 € (2025, n5+n7+n9), −325,061.86 € (2030, n5+n7+n9) and −185,008.28 € (2025, n5);
- all equal to the report's table.

**Status.** The screen ran in time for the numbers freeze (11-15 12:00). Whether, and in what words, it enters §4.3 and
R3.6 is the expert's round-10 call.
