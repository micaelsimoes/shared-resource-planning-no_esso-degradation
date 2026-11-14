# P5.15 W180a: Appendix E regenerated and re-audited at Overleaf e5322f8 (Addendum 79, W180 item (a))

**Worker, 2026-10-16.** Zero solves, zero model builds, zero pickle loads. The manuscript clone, `manuscript_review/`,
the frozen JSON, the specs, the records and every W179 script and output were only read. Nothing in the manuscript was
changed.

## Blocked on Planner

None. All four expert predictions of item (a) held, and so did the verifier count. Three observations are left for the
Planner (see Found, items 6–8). None of them is a MISMATCH, and no run depends on them:

1. **E.4, phase B is described accurately but incompletely.** The text calls it "a re-solve … with its no-reverse-flow
   rows switched off". The code does two more things in that phase:
   - it fixes every DSO decision at the arm solution, so the phase is the DN power flow at the TN voltage;
   - it deactivates the voltage and thermal rows together with the no-reverse-flow rows.

   The sentence does not contradict the code, so it is rated MATCH with this qualification recorded. Whether to say
   more is a wording choice for the expert.
2. **E.6 prints "0.004τ" to three decimals.** The intro's rounding sentence names only the residual ratio of Table E.24
   as a three-decimal exception. Read strictly, this τ-multiple in the prose is a second, unnamed one. At two decimals
   it would print as 0.00τ.
3. **The scratchpad is shared between the parallel Workers.** A W180c trial wrote into my scratch trial directory and
   into its log. Only a throwaway trial was affected; the real run is clean. The W180b/c/d outputs may want the same
   check.

## Changed

- **`7b6ba5fe`**: new harness `p515_s53_w180a_appendixE_check.py`, sha256 `cc73c85d…`.
  - Committed before the run. The run records `git_head` `7b6ba5fe40079cfd8cfe859fd229f1538a0c1d36`.
  - Sections 0–6 (helpers, plan decoding, conditions, the E.20–E.24 checks, needles, W163 lookup) are copied from
    `p515_s53_w179a_appendixE_check.py`. That script was not imported, edited or run.
  - **One adaptation**, marked `W180a ADAPTATION` in `check_e21`, implements the row-16 convention of Addendum 79
    ruling 1. A frozen cell with `superseded = true` is expected to print:
    - Status "superseded", with its ^s marker kept;
    - Rule "---". This is derived, not assumed: v6 applied to the cell's record (W142 item 1) must not decide it, and
      for `pb_y2025_n5` it reads "not decided within the recorded cycles 1..167", 1 veto.
- **This commit**: this report and `data/SRP1/Results/P515S53/w180a_appendixE/`. The directory did not exist before the
  run; it was created by `mkdir` in the launch command and written once.

| file | sha256 |
|---|---|
| `w180a_appendixE_check.json` | `b584b8136f789683cc390b2db954b875f711e68fb6633b9b2ec8fac7ef9db48d` |
| `w180a_appendixE_check.md` | `8477953cf6ca23d3574cdfdde9dc91d0b2bc895bc97334a9e81461b5d5e83e2f` |
| `appE_generated_w180a.tex` (generator output) | `79ff8094d1559f564850ca10957f565833097c5a3460a70a8675fd1f97329444` |
| `verify_appE_output.txt` (full verifier output) | `7a0c47e9eee3b365075f0b46bdc6d20e7ec0ff40eab7f68c8213695861d213a7` |
| `manifest_sha256.json` (45 inputs, main.tex, `git:…@48099ab:main.tex`, script, outputs) | `d805da85d1b7093a1a8805b50fcdf265bae052a810a164c5a07bbdf9d2566112` |
| `launch.log` (stdout + stderr, attached, exit 0) | `097ad2f3dca0b3ac7cedb61dd73bf4c75cf50e93e176729a0778f55e53361eb7` |
| `w180a_bool_typing_test.json` / `.log` (W100 test, **PASS**, F1–F5 = 0) | `398bfb2e…` / `dc435ebf…` |
| `manifest_post_run_sha256.json` | `a46b2aa796080ec5f9f5f8f6c7579944335f835fc630dee11d3663549e7b5fbb` |

**Commands**, run from the repo root with the canonical interpreter, attached, with no nohup, screen or `&`:

```
mkdir data/SRP1/Results/P515S53/w180a_appendixE && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180a_appendixE_check.py \
  > data/SRP1/Results/P515S53/w180a_appendixE/launch.log 2>&1                                   # exit 0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \
  --out data/SRP1/Results/P515S53/w180a_appendixE/w180a_bool_typing_test.json \
  > data/SRP1/Results/P515S53/w180a_appendixE/w180a_bool_typing_test.log 2>&1                   # exit 0, PASS
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180a_appendixE_check.py --post-run   # exit 0
```

**Identity, checked by the harness before anything ran:**
- **Clone.** `manuscript/6a67305f25e8348fb71380c3` is at HEAD `e5322f85dced8034312c9afb97c19001bce6bfa0`, and
  `main.tex` is clean in the clone.
- **main.tex at `e5322f8`.** sha256 `b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492`, 2,710 lines,
  byte-identical to `manuscript_review/main_48099ab_round6.tex`.
- **main.tex at `48099ab`.** Read with `git show 48099ab:main.tex` (full hash `48099abb…`). It is `28008ab0…`,
  2,717 lines, the file W179a audited, and is byte-identical to `main_e926fb7_appE.tex`.
- **Inputs.** All 45 are committed and clean, including `gen_appE.py` (`a47f5dc8…`), `verify_appE.py` (`b43476f6…`),
  the 16 files of `export_W160/`, `appendixE_evaluation_record.tex` (`79ff8094…`) and the frozen JSON (`590088fe…`).

**Guards:**
- `SolveProfileGuard(permitted=())` was installed before any project import. At the end it read 0 permitted and
  0 blocked solves and execs, and `verify(0)` returned no failures.
- `pickle.load` and `pickle.loads` were blocked for the whole run and called 0 times.
- JSON outputs go only through `gate_result_io`; the bool-typing test passed.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit;
value = Q(0) − Q(x); F = Q + I. Values are in € in the JSON and in k€ in the manuscript.

**Rounding rule:** a printed number is a MATCH when |printed − value| ≤ half a unit of its last printed digit.

## Found

Each expert prediction of Addendum 79 item (a), against its outcome:

| Prediction | Outcome |
|---|---|
| (a1) the regenerated block equals `main.tex` bitwise | **HELD** |
| (task) `verify_appE.py`: 33 PASS, 0 FAIL | **HELD** |
| (a2) 0 MISMATCH in 1,623 entries | **HELD** |
| (a3) exactly two entries changed (row 16 Status, Rule) | **HELD** |
| (a4) all five prose sentences MATCH | **HELD** |

- **(a1)** The generator's output occurs exactly once in `main.tex` at `e5322f8`, at **lines 2375–2694** (320 lines,
  bytes 188,555–222,479). It starts and ends on line boundaries. It is also byte-identical to:
  - the committed `appendixE_evaluation_record.tex`;
  - the verifier's own regeneration in its temp directory.
- **(task)** The verifier printed 33 PASS and 0 FAIL, with no exception.
- **(a2)** 1,623 entries:
  - 1,443 MATCH on frozen fields and 0 MISMATCH;
  - 180 entries that are not frozen fields all agree with their named source, and 0 disagree;
  - 0 mapping problems.
- **(a3)** Of 1,637 table cells compared, exactly 2 changed, and they are the predicted ones. Captions and legends
  changed separately (2), as listed in item 3 below.
- **(a4)** Intro, E.3, E.4 (with the E.23 caption), E.5 and E.6 have 0 MISMATCH: 10 sub-items MATCH and 1 DERIVED.
  Details are in item 4.

### 1. Generator and verifier

- **Generator.** `gen_appE.py` ran in-process under the guards, with `sys.argv[1]` set to
  `data/SRP1/Results/P515S53/w180a_appendixE/appE_generated_w180a.tex`. The harness refused any target outside the
  output directory or onto a `manuscript_review/` file.
  - stdout `ok 58 61`.
  - No `appE.tex` appeared in the cwd or in `manuscript_review/`.
  - `git status` of `manuscript_review/` is clean after the run.
- **Verifier.** `verify_appE.py` ran in-process. It regenerates into its own `tempfile.mkdtemp()` directory.
  - **33 PASS, 0 FAIL**, no exception.
  - It also prints three informational lines, which are not checks: median k*/end over the 42 = 186.5, certified only
    174.0; k* − k0 21 / 65 / 89; median range/τ 0.86.
  - Its full output is in `verify_appE_output.txt`.

### 2. Bitwise identity

- **In main.tex.** One occurrence at lines **2375–2694**, line-aligned. This equals STEP6 §A.10's "lines 2375–2694 of
  the new main.tex", which was located by content, not taken from STEP6.
  - Taken from the `% ====` rule to the line before `\section*{Declaration…}`, the Appendix E block is 2375–2695 at
    `e5322f8`. The one extra line, 2695, is blank.
  - The same block is 2374–2702 at `48099ab`. STEP6 says 2374–2701; line 2702 is the blank line before Declaration.
- **Against the committed file.** It equals the committed `manuscript_review/appendixE_evaluation_record.tex`
  (`79ff8094…`).
- **Against the verifier.** It equals the verifier's tempdir regeneration (`79ff8094…`).

### 3. Entry check at `e5322f8` against `frozen_step6_tables_v1_590088fe.json`, and the diff from `48099ab`

| Table | Rows | Entries | MATCH | MISMATCH | Not a frozen field, agrees | Disagrees |
|---|---:|---:|---:|---:|---:|---:|
| E.20 `tab:appE_claims` | 58 | 639 | 581 | 0 | 58 | 0 |
| E.21 `tab:appE_evaluations` | 61 | 854 | 732 | 0 | 122 | 0 |
| E.22 `tab:appE_neighbours` (+ note) | 8 | 18 | 18 | 0 | 0 | 0 |
| E.23 `tab:appE_benchmark` (+ note) | 7 | 49 | 49 | 0 | 0 | 0 |
| E.24 `tab:appE_gap` | 9 | 63 | 63 | 0 | 0 | 0 |
| **Total** | | **1,623** | **1,443** | **0** | **180** | **0** |

These tallies are identical to W179a's at `48099ab`. The other checks:
- **The E.21 map.** It is one-to-one onto the 61 frozen cells, identical at `48099ab` and `e5322f8`, and 0 mapping
  problems.
- **Row 16 now MATCHes.** It is `pb_y2025_n5`, mapped through its ^s marker:
  - Status "superseded" = frozen `superseded: true`;
  - Rule "---" agrees with W142 item 1 (v6 not decided within 1..167, 1 veto);
  - k* 167, range 0.99 and band 4.5 MATCH the frozen W118 v2 certificate.
- **Cross-check on `48099ab`.** The same checker under the round-6 convention flags exactly row 16 Status (printed
  "cert., osc.") and row 16 Rule (printed "record"), and nothing else.

**Entry-by-entry diff, `48099ab` → `e5322f8`.** 1,637 raw table cells were compared; this includes the row-number
columns, which are not rated entries.

| Kind | Changed | Detail |
|---|---:|---|
| **Table cells** | **2** | E.21 row 16 Status: `cert., osc.$^{s}$` → `superseded$^{s}$` (l. 2525 → 2525); E.21 row 16 Rule: `record` → `---` |
| Captions | 2 | **E.21** (l. 2494): "branch, or the cause … growth test)." → "branch, the cause … growth test), or superseded."; "$^{s}$: superseded by evaluation~15." → "…~15; certified under an earlier version of the rule, whose cycle, range and band are shown, and not decided under the rule of \ref{app:admm_updated_certification} within its recorded cycles." **E.23** (l. 2630 → 2626): "settlement excluded." → "settlement excluded, after the consistency pass."; "over the horizon." → "over the horizon, from the first phase (before the pass)." |
| Notes / legends | 0 | The E.20, E.21, E.22 (51.3 k€, 3.79×) and E.23 notes are unchanged |
| Group headers, other table lines, row sets | 0 | |

**Line diff of the whole Appendix E block: 13 hunks, each matched to a §A.10 item.**
1. The header comment: the `[CONFIRM — W179]` mapping note becomes the audit note.
2. The intro rounding sentence (l. 2388).
3. The E.21 caption.
4. Row 16.
5. E.3 (l. 2588).
6. The removed E.3 CONFIRM comment.
7. E.4 (l. 2623 → 2621).
8. The removed E.4 CONFIRM comment.
9. The E.23 caption.
10. E.5 (l. 2661 → 2657).
11. The E.6 continued-evaluations sentence (l. 2692 → 2688).
12. The removed E.6 CONFIRM comment.
13. The E.6 replay sentence (l. 2699 → 2692).

No other line of the appendix changed. The prose not listed above therefore keeps W179a's ratings (`P5_15_W179A_APPENDIX_E.md`).

**Mutation test.** Each case is a single-entry edit to an in-memory copy of the `e5322f8` lines; nothing was written to
disk. Every edit was caught.
- **For a cell edit**, two conditions must both hold:
  - **Frozen check:** the symmetric difference of its non-agreeing set against the baseline is exactly the edited
    (table, row, key).
  - **Diff:** the diff against `e5322f8` is exactly that cell, and against `48099ab` it moves by exactly that cell.
- **For a caption edit**, the diff must show exactly one caption change.

| id | edit | frozen check | diff |
|---|---|---|---|
| M1 | E.20 row 3 Δ gross 208.3 → 208.4 | caught | caught |
| M2 | E.21 row 1 band 4.2 → 4.3 | caught | caught |
| M3 | E.21 row 16 Status → `cert., osc.$^{s}$` (the 48099ab form) | caught | caught (the row-16 change vanishes) |
| M4 | E.21 row 16 Rule → `record` | caught | caught |
| M5 | E.22 eval 19 × 3.40 → 3.41 | caught | caught |
| M6 | E.23 passive / cold NRF n 29 → 30 | caught | caught |
| M7 | E.24 eval 42 Resid. 0.692 → 0.693 | caught | caught |
| M8 | E.21 row 40 Rule `record` → `run` | caught | caught |
| M9 | E.20 row 34 net verdict `det.` → `w.r.` | caught | caught |
| M10 | E.23 caption "after the consistency pass" → "before …" | n/a (caption) | caught |
| M11 | E.21 caption "within its recorded cycles" → "within its cycles" | n/a (caption) | caught |
| M12 | E.22 note 51.3 → 51.4 k€ | caught | caught |

### 4. Round-6 prose ratings

Every sentence was found verbatim in `main.tex` at `e5322f8`. All values and sources are in the JSON under `prose`.

| id (line) | Sentence (abridged) | Rating | Evidence |
|---|---|---|---|
| P1 (2388) | intro: "ratios to two decimals, except the residual ratio of Table E.24, given to three" | **MATCH** | All 284 k€ amounts in E.20–E.24 print 1 dp. All 171 ratio entries of E.20–E.22 (×, range/τ) print 2 dp. All 7 E.24 Resid. entries print 3 dp. Beside these: the 21 E.24 shares (%) print 1 dp. See item 7 for the prose. |
| P2a (2588) | E.3: "at its own resolution ($\sigma_Q$, or the sum of the two evaluations' bars where larger" | **DERIVED** | For all 14 W173 `item2.x0.rows`, `search_resolution_eur` = max(bar_x + bar_inc, σ_Q), recomputed from S47 `campaign_results.json` (points.<label>.bar, final_incumbent x0 bar 9,629.98 €, σ_Q 18,449.66 €), 14 of 14 exact. 8 are at σ_Q and 6 at the bar sum (22.1–47.2 k€), as in W179a E3.5. S47 `resolution_rule` and Algorithm l. 512 state the max. |
| P2b (2588) | E.3: "The other six … two remaining pairs of nodes and at all three nodes, in 2025 or 2030, … search only, on an earlier state of the code and stopped by the production exit; … 3.47--7.17 times the search resolution." | **MATCH** | **The six:** the W173 rows with empty `frozen_table_rows` are exactly 2025 {5,7}, {7,9}, {5,7,9} and 2030 {5,9}, {7,9}, {5,7,9}. These are the pairs that E.22 lacks in each year (E.22 has 2025 {5,9} and 2030 {5,7}) plus the triple; none is among the 61 frozen cells. **Ratios:** (F − F(0)) / resolution = 5.52, 6.16, 3.47, 6.07, 5.57, 7.17, so 3.47–7.17; this equals W179c's range, all are > 1 and all are "worse". **Code state:** S47 `git_head_at_run` is `353e094b`, where W179c records a production diff of 5 files, +1,507/−84 against HEAD. **Production exit:** for each of the six, S47 `cycles_run` = `certification_cycle` under the production rule; also W179c report l. 283 and main.tex l. 1706. |
| P3a (2621) | E.4 phase 1: "the DNs' dispatch and the TN's dispatch against it, from which the curtailment columns are taken" | **MATCH** | `uncoordinated_benchmark.run_operational_planning_uncoordinated` solves the DSOs first, then the TSO with the interface P/Q hard-fixed to the DSO schedule. W116 phase A, l. 565; `curtailment_rows_are_phase_A` at l. 669. Frozen field: `curtailment_phase_A_by_start` (E.23 columns MATCH). |
| P3b (2621) | E.4 phase 2: "a re-solve of each DN at the TN's interface voltage with its NRF rows switched off, in which the exceedance columns count the network-hours (of 864: three networks, twelve blocks, 24 h) …" | **MATCH**, qualified | The count is taken in phase B: W116 l. 617 and 620, UB l. 208 (NRF row deactivated in the re-evaluation). One row per period: UB l. 193. Frozen `where` = "3 DNs x 3 years x 4 days … 24 rows per block at SRP1", so 3 × 12 × 24 = 864. n = 29 (passive) and 70 (price-taker), at every start. **Qualification:** phase B also fixes every DSO decision at the arm solution and deactivates the voltage and thermal rows (UB l. 1413, 1415; frozen `consistency_reevaluation`). The sentence omits these but does not contradict them. |
| P3c (2621) | E.4 phase 3: "the consistency pass, a re-solve of each DN at that voltage with the rows on, followed by the TN's dispatch, whose cost is the Q of Table E.23" | **MATCH** | Phase C runs only if triggered (W116 l. 637). It pins the DN voltage (l. 638), re-solves the DSOs (l. 641) and then the TSO (l. 644); the arm cost is phase C (l. 666). `trigger` is true for all 6 arm/starts, and `arm_cost_source_by_start` is `phase_C_sequential_pass` for all 6. |
| P3d (2621) | "The consistency column is the change of Q from the first phase to the pass." | **MATCH** | `effect_on_q_eur` = Q(phase C) − Q(phase A), W116; frozen `pass_effect_eur` (E.23 column MATCH). |
| P3e (2626) | E.23 caption: "Q: gross operational cost, settlement excluded, after the consistency pass. Curtailment: day-weighted, over the horizon, from the first phase (before the pass)." | **MATCH** | `benchmark.objective_convention` reads "settlement EXCLUDED". Q is from phase C for all 6. Curtailment = `curtailment_phase_A_by_start.*.mwh_day_weighted`, printed / 1000 in GWh (MATCH). |
| P4 (2657) | E.5: "(for evaluations 40 and 41 the split and the residual were not recorded)" | **MATCH** | Of the 7 fingerprint rows (39, 40, 41, 42, 45, 50, 51), exactly 40 (`ref:5ca4f86c`) and 41 (`ref:e28de4ac`) have `share_by_node` and `pf_primal_last` set to None in `dead_zone`, and E.24 prints "---" in all four columns for both. The other five record 55.2–55.3 / 13.7 / 31.0–31.1 % and 0.682–0.700. |
| P5a (2688) | E.6: "The three evaluations continued past their certificates, 12, 14 and 3, moved by a further 0.004τ, 0.74τ and 0.78--0.88τ." | **MATCH** | W163 N25 (status `match`) has exactly three cells: `b_2a0ba8b2` = row 12, `b_4649234b` = row 14, `d_c52e1670` = row 3. The movements are |Δ|/τ = 0.00389, 0.738, and 0.781 or 0.882 (the latter on the last-pair reading 148), printing 0.004, 0.74 and 0.78–0.88. Each is measured from the E.21 k* (173, 148, 150). Row 12: +17.668 € from 173 to 213, and 17.668 / 4,539.07 = 0.00389. |
| P5b (2692) | E.6: "the three single-scenario reference continuations (the plan without storage, the reference unit and a larger plan not tabulated here) replayed bitwise through the stopping cycles of the residual-based runs they continue, cycles 132, 112 and 87" | **MATCH** | W101 `campaign_results.json`, for x0 / unit / C*: G19 is true with `bitwise_through_N`, N = 132 / 112 / 87, and G13 `replay_bitwise_through_cycle` = 132 / 112 / 87. This equals W163 N49 and the W101 summary N. The A53 report calls these the "old certification cycle" (l. 87, 94–96). The candidates: x0 = row 1's plan, and unit = 7: 0.25/1 = row 2's. C* is 0.96875/3.875 at each of nodes 5, 7 and 9 in 2025: at least the unit at every node, and absent from the 61 frozen cells. The new wording no longer conflicts with E.21's k* (181, 172). |
| P6 (2688) | E.6: "Ten of the 46 certificates the tables use" | **DERIVED** | Registry 49 = the certified frozen cells; 49 − 1 superseded (row 16) − 2 twins (rows 56 and 61) = 46; `n_counted` 10. The count is unchanged by the relabel: row 16's frozen status is still `certified` under its own rule. |
| P7 (2494) | E.21 legend: "$^{s}$: superseded by evaluation 15; certified under an earlier version of the rule, whose cycle, range and band are shown, and not decided under the rule of A.5 within its recorded cycles"; Status legend "…, or superseded" | **MATCH** | Row 15 = `pb_y2025_n5_v6`. Row 16's `certifying_spec` is W118 r2, criterion version 2. Its printed k*, range and band MATCH that certificate. W142 item 1 gives v6 "not decided within the recorded cycles 1..167". |

### 5. Observations for the Planner (not MISMATCHes)

6. **E.4 phase B (P3b)** is accurate but incomplete. The re-solve has every DSO decision fixed (it is the DN power flow
   at the TN voltage), and the voltage and thermal rows are switched off together with the no-reverse-flow rows. W179a
   (v) recorded the same mechanism. Addendum 79 ruling 3 took the three-phase wording, so this is the expert's call
   only if the omission matters.
7. **Prose decimals.** The only prose decimal with three or more places in the appendix is "0.004$\tau$" (E.6, l. 2688).
   - The intro's exception names only the E.24 residual ratio. The E.5 "residual ratio of 0.027" is that ratio, so it
     is covered.
   - P1 is rated on the tables, which is W179a's I2 scope, and holds there.
   - If the intro is read as covering prose, "0.004τ" is an unnamed exception. At two decimals it would print as
     0.00τ.
8. **Provenance of the generator.** `gen_appE.py`, `verify_appE.py` and `appendixE_evaluation_record.tex` were last
   changed in `c831ea46` (2026-10-15 17:55, message "Appendix E rebuilt from export_W160; letter v6; Addendum 78 (W179
   order)"). Their content is the round-6 version: the generator holds the `SUPERSEDED` / "superseded" status and the
   row-16 Rule "---". The round-6 delivery commit `465da61a` did not touch them. This matches the Planner's statement
   that they are in `c831ea46`; only the message reads as round 5.
9. **Shared scratchpad.** During a scratch trial (`--trial-out-dir` in the session scratchpad) a W180c process wrote
   `w180c_text_letter.json` / `.md` into my trial directory `trial1`, and its log lines into my trial `launch.log`. The
   parallel Workers share one scratchpad. My real run used the fresh `data/` directory and a second trial in a
   uniquely named directory, so no committed artefact is affected. Trial directories named generically (`trial1`,
   `trial`) may hold mixed output from several Workers.

## Not confirmed

- **The completeness of the "three continued evaluations".** The set is W163 N25's. No further search for continuations
  past a certificate was made: W179a's scope was rows 12, 13, 14 and 3, and the v6-stage cells stop at k* by
  construction. The N25 movements were not recomputed from the per-cycle records.
- **"Stopped by the production exit" for the six search-only neighbours.** This rests on two things only:
  - S47 `cycles_run` = `certification_cycle` with status "certified" (the production rule at `353e094b`);
  - W179c's statement and main.tex l. 1706.

  The production stopping test at `353e094b` was not re-read from code.
- **The C* plan is "larger".** It was checked component-wise against the reference unit (≥ at every node, and not
  equal). It was not checked against other plans or by investment cost.
- **The verifier's tempdir file.** Its path was recorded and its sha256 equals the generator output, but the file was
  not preserved: it is under the system temp directory, outside the repository.
- **Compilation.** The manuscript was not compiled. The expert's compile check (0 errors, labels unchanged) was not
  repeated; it is outside item (a).
- **Main text and letter.** Only Appendix E was checked. They are W180b, W180c and W180d's scope.
