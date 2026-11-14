# P5.15 W179a: Appendix E against the frozen tables (Addendum 78, W179 item (a))

**Worker, 2026-10-15.** Zero solves, zero model builds and zero pickle loads. The manuscript clone, `manuscript_review/`,
the frozen JSON, the specs and the records were only read.

**Manuscript.** `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `48099ab` (full
`48099abb09db45a1668c1ef8441a3b7c658913a8`).
- sha256 `28008ab0d79b477c4e8d86a0c3a4d1acf9c95ed9c4921f4757f1600cda1c60af`, 2,717 lines.
- It is byte-identical to `manuscript_review/main_e926fb7_appE.tex` (checked). It is clean in the clone.
- Line numbers below are at `48099ab`.

**Frozen tables.** `data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json`, sha256
`590088fe…` (verified).

**Script.** `p515_s53_w179a_appendixE_check.py`, committed at `dd74d033` before the run (sha256 `9acb0b30…`).
- It is the previous Worker's draft, reviewed and fixed. The changes:
  - Entries that are not frozen fields (the E.21 Condition and Rule columns, the E.20 comparison text) are rated
    *not a frozen field, agrees / disagrees*, with their source named, rather than MATCH.
  - The `_find_first` fallback in (iv) is replaced by explicit field paths. The ext v3 and A64 specs carry no
    `canonical`, so they are resolved through `candidate_key`.
  - The rounding check is extended to E.22–E.24.
  - The §4.6 salvage items, the 46 identity, the R2 wording and evidence-conditioned outcomes for (v), (vii) and (viii)
    are added.
- **Independence from the expert's generator.** The script does not import or read `gen_appE.py` / `verify_appE.py`. It
  maps E.21 rows to cells by decoding plan text, year and condition against `candidate_canonical` and each cell's stage
  spec.
  - After the run, as a separate cross-check, its E.21 map was compared with the cell order written in `gen_appE.py`:
    they are identical, 61 of 61.
- **Mutation test (scratchpad only, before the real run).** Nine single-entry edits to a copy of the LaTeX were each
  detected as MISMATCH or disagreement at the right row and column. The edits were: a band, a ×, an NRF count, a share,
  an E.22 Δ, a Rule label, a marker, an Eval. pair and a note figure.

**Outputs.** `data/SRP1/Results/P515S53/w179a_appendixE/`, written once (the directory did not exist before the run).

| file | sha256 |
|---|---|
| `w179a_appendixE_check.json` | `c69ad4df6718424631c05e52e6ccc8d2c20811fe67252c33b4d9f42e168797de` |
| `w179a_appendixE_check.md` | `977eeae2934e445451e760ff7556cdfbb3ccd855477ed76cb96d272457edcb39` |
| `manifest_sha256.json` (24 inputs + main.tex + script) | `66a6c5b7f43b484adb45ab2363c75a5852f1509b7f73a23c835e9b88c2ecaba6` |
| `launch.log` (stdout + stderr, attached, exit 0) | `edb7c620e5f1f5a71df3ca7cfa39370e01bbbf0201bfc163f43e28cf4ed11f13` |
| `w179a_bool_typing_test.json` / `.log` (W100: **PASS**) | `849eb648…` / `ab578a7a…` |
| `manifest_post_run_sha256.json` | `6a5ae2ea5457302502f84897158957e70a40958b53783702901a6402fd568626` |

**Guards.**
- `SolveProfileGuard(permitted=())` was verified at exactly 0 solves and 0 execs.
- `pickle.load` / `pickle.loads` were blocked for the whole run and called 0 times.
- All 24 inputs are committed and clean (asserted before the run).

**Objective convention.** Q = `gross_operational_cost`, settlement excluded. Net = Q − terminal salvage credit. Value =
Q(0) − Q(x). F = Q + I. Values are in € in the JSON and k€ in the manuscript.

**Rounding rule.** An entry is a MATCH when |printed − value| ≤ half a unit of the last printed digit.

---

## Blocked on Planner

1. **Prediction "at most two of (i)–(ix) corrected" FAILED: three are corrected, (v), (vii) and (viii).** (vi) is
   confirmed.
   - (viii) is a wording correction: the record holds, and only the appendix's own definition of "certification cycle"
     makes the sentence wrong.
   - If the Planner reads (viii) as a clarification, the count is two, but they would be (v) and (vii), not (v) and (vi).
   - This is an interpretation-count prediction, not a margin or sign prediction. No run depends on it.
2. **Expert wording needed. No manuscript text was touched.** Each item is evidenced under (i)–(ix) or the prose checks.
   - (a) **E.4, l. 2623.** The consistency-pass sentence:
     - The exceedances are counted in the re-evaluation that triggers the pass.
     - The "Consistency" column is the change in Q from the pass.
     - The table's Q is the cost after the pass, while the curtailment columns are from before it.
     - See (v).
   - (b) **E.6, l. 2692.** "The two evaluations continued past their certificates, 14 and 3" should read three:
     evaluation 12 moved 0.004τ. The §4.7 bound "at most 0.9τ" (l. 1690) still holds. See (vii).
   - (c) **E.6, l. 2699.** "replayed bitwise to their certification cycles". The records give 132, 112 and 87, the cycles
     at which the residual-based rule had certified them. Appendix E's certification cycles are k* = 181 and 172, and
     C* is not in the appendix. See (viii).
   - (d) **Intro, l. 2388.** "ratios to two decimals": the E.24 Resid. column, a ratio, is printed to three decimals
     (0.027, 0.682, …) in all 7 rows that have it.
   - (e) **Row 16 against §4.7, l. 1689.**
     - §4.7 says: "Every certificate reported in this paper was decided, or re-decided from its committed record, under
       the rule … and certifies at the cycle reported".
     - E.21 row 16 (`pb_y2025_n5`, "cert., osc.^s", k* 167, Rule "record") is now a reported certificate.
     - But v6 applied to its record is "not decided within the recorded cycles 1..167", with 1 veto (W142 item 1).
     - The printed k* 167, 0.99 and 4.5 are the W118 v2 certificate.
     - This sits with W179c's Blocked 1 (the same row, closure slack); one expert ruling on row 16 would cover both.

## Prediction vs outcome

| Prediction (expert, Addendum 78) | Outcome |
|---|---|
| (a) 0 MISMATCH in table entries | **HELD.** 1,623 entries across E.20–E.24: 1,443 MATCH on frozen fields, **0 MISMATCH**. A further 180 entries are not frozen fields and **all agree with their named source** (0 disagree). 0 mapping problems. |
| At most two of (i)–(ix) corrected, most likely (v) and (vi) | **FAILED: 3 corrected, (v), (vii), (viii).** (vi) is confirmed. (iii) is confirmed with a qualification for row 16. |

---

## Table checks (E.20–E.24)

| Table | Rows | Entries | MATCH | MISMATCH | Not a frozen field, agrees | Disagrees |
|---|---:|---:|---:|---:|---:|---:|
| E.20 `tab:appE_claims` | 58 | 639 | 581 | 0 | 58 (comparison text) | 0 |
| E.21 `tab:appE_evaluations` | 61 | 854 | 732 | 0 | 122 (Condition, Rule) | 0 |
| E.22 `tab:appE_neighbours` (+ note) | 8 | 18 | 18 | 0 | 0 | 0 |
| E.23 `tab:appE_benchmark` (+ note) | 7 | 49 | 49 | 0 | 0 | 0 |
| E.24 `tab:appE_gap` | 9 | 63 | 63 | 0 | 0 | 0 |

**Every MISMATCH: none.**

**What each check compared.**
- **E.21 (61 entries per column).** Each row was checked against `tables.cells` and `tables.cells_appended_w160`:
  - status and branch;
  - the markers ^f, ^t, ^s, from `flag_non_clean_turning_point`, the twin field and `superseded`;
  - k0, k*/end, range/τ, band;
  - |gap| and |slack| for uncertified cells;
  - N.c. as `len(non_clean_after_N)`;
  - Replay as `replay_bitwise_through`.
- **Sources of the two not-frozen columns.**
  - Condition comes from each cell's stage spec: v4, v5, v6, ext v3, W118 v2, A64 v1 and the W101 campaign specs.
  - Rule is derived from the frozen `certifying_spec` together with W142 item 1.
- **E.20.** Each row was checked against `tables.claims` (T1) or `tables.a64.rows` (T10):
  - Δ, threshold or bar, ×, and the gross verdict;
  - Δ, × and the net verdict. For rows 17–20 these were read from the `G…:net` claim, which was also checked equal to
    the `:gross` claim's net fields.
  - The net threshold is the same as the gross one (as the caption says).
  - The ^u / ^f markers.
  - The comparison text, decoded against both cells.
- **E.22.** Δ, threshold and × were checked against `phase_b` (M_gross, v6_threshold, v6_multiple). Rows 15, 22 and 23
  were also checked against the C claims. The note's 51.3 k€ (3.79×) is `phase_b` of `pb_y2025_n5` = E.21 row 16.
- **E.23.** `benchmark.w160_additions.arms_in_full`:
  - Q by start, NRF n, largest excess (p.u.), `pass_effect_eur`, and the TN / DN curtailment (GWh);
  - the ^b marker against `best_start`;
  - the note: spreads 25.8 / 0.0 k€ (`multimodality_band`), band 71.9 k€, and the unconstrained sweep blocks (1 of 12,
    6 h; 8 of 12, 25 h, the blocks named).
- **E.24.** `dead_zone.cells`: `t_sum_at_cap`, `share_by_node`, `pf_primal_last`, `lapse_events`. Plan, year,
  condition and cause were checked to be identical to the E.21 row of the same number.

**Row mapping (task item 1).**
- **E.21.** 61 rows, numbered 1–61, map one-to-one onto the 61 frozen cells: the 49 in `tables.cells` and the 12 in
  `cells_appended_w160`.
  - Three plan/condition pairs match more than one cell. Each is decided by the printed marker:
    - row 15 (no ^s) is the v6 re-run `pb_y2025_n5_v6`;
    - row 16 (^s) is the superseded `pb_y2025_n5`;
    - rows 2, 56 and 61 differ by condition family (production base / ext-v3 "baseline" arm / A64 0.70 floor).
  - W179c already found that E.21 agrees with T2 on status and k*/end on all 61 rows, and that row 16 is the superseded
    certificate. This is cited, not repeated. This check adds every other column.
- **E.20.**
  - **The map.** 58 rows, numbered 1–58. Rows 1–21 and 23–51 and 53–57 map to 55 distinct T1 claims; rows 22, 52 and 58
    map to the 3 distinct T10 claims `H:m1.75:value_minus_I`, `E:soh050:value_minus_I` and
    `E:soh050:delta_value_vs_070`.
  - **The five unused T1 claims.** They are exactly the four `G:…:net` duplicates and `CHECK:headline_V_minus_I_settled`.
  - **The CHECK claim.** It is row 1 with the sign of value − I: d_gross is exactly opposite, with the same cells and
    the same threshold.
  - **The unused T10 claim.** It is exactly `E:soh050:delta_value_vs_g070_REPORT_BESIDE`. It equals row 58: d_gross is
    bitwise equal, and threshold, × and verdict are equal at print.
- **Cross-references by number.** All of these resolve to the intended cell:
  - every E.20 "Eval." pair (resolved through the E.21 map, and the claim found by that pair);
  - every E.22 "Eval." (plan and year checked);
  - every E.24 "Eval." (plan, year, condition and cause equal to E.21);
  - the prose references: 4 and 38; 56/2; 61; 16/15; 42; 14, 3 and 12; 36; 32, 33, 60, 61; the ten IDs; 1, 2, 40, 41;
    and "evaluation 40" in the (c) group header.

## Prose checks (intro, E.1, E.3, E.4, E.5, E.6; plus the §4.6 items the task lists)

66 items: **53 MATCH, 10 DERIVED, 3 MISMATCH.** The full list, with values and sources, is in
`w179a_appendixE_check.json` → `prose`.

| Item (line) | Written | Rating | Evidence |
|---|---|---|---|
| I1 (2388) | τ = 4,539.07 € | MATCH | `constants.TAU` 4,539.068 |
| **I2 (2388)** | amounts to 0.1 k€, ratios to two decimals | **MISMATCH** | 284 k€ amounts are all 1 dp, and the 171 ×/range ratios in E.20–E.22 are all 2 dp. But the E.24 Resid. ratios are 3 dp (7 of 7). |
| E1.1–E1.3 (2395) | 58; 55 gross and net; the remaining three gross only | MATCH | rows 22, 52 and 58 = T10, which has no net field |
| E1.5 | max{3b, 2τ} | DERIVED | Recomputed for all 41 certified-certified claims; threshold and verdict reproduced 41/41 |
| E1.6 | uncertified bar | DERIVED | 3 × max(|gap|, |slack|) over the uncertified cells, reproduced 23/23. Verdict = det. iff gross × > 1 and Q_cc × > 1, consistent on all |
| E1.7–E1.8 | flagged 4 and 38; none changes verdict | MATCH | flags on `d_4a82a64a` and `j_a11d7966`. The uncertified forms of rows 3 and 26 are determinate, and rows 30 and 31 are already uncertified. |
| N1–N7 (2481, note) | the E.20 note figures | MATCH / DERIVED | Row 3: 70.7 k€, 2.94×. Row 26: 15.9 k€, 11.12×. Rows 30 and 31: uncertified form. 56 is the twin of 2 through 172. Row 58 = the REPORT_BESIDE row. ^n = 2 energy steps (|3.5 − 2.5| / 0.5). Rows 17–20 are G gross/net pairs. |
| E2.1–E2.4 | 61; k0 not tabulated for 1, 2, 40, 41; k0 = 123; row 16 note | MATCH | `k0_run` is None for the four. W168 `decision_primary` N = k0 = 123. The `pb_y2025_n5` T5 note reads "the W118 certificate sat on a non-Optimal cycle". |
| E3.1–E3.4 (2588) | 14 evaluated; 8 re-evaluated; their plans; all determinate | MATCH | W173 item2.x0: `b_box` 14/14, 8 re-settled. E.22 plans checked against `candidate_canonical`. All verdicts determinate. |
| E3.5 | "at its resolution σ_Q" | DERIVED | The search accepts at max(bar_x + bar_inc, σ_Q) (Algorithm, l. 512). This equals σ_Q = 18,449.66 € for 8 of the 14 neighbours and is 22.1–47.2 k€ for the other 6. Fine as shorthand. |
| E4.1–E4.2 (2623) | three starts; best reported; spread = band | MATCH | `n_starts` 3/3; `multimodality_band` = max − min of Q by start, exactly |
| E5.1 (2661) | gap −8.8 to −9.4 k€ (39, 40, 41, 42, 45, 50, 51) | MATCH | −8.8 / −9.2 / −9.3 / −9.3 / −9.4 / −9.2 / −9.2 |
| E5.2–E5.3 | 55/14/31 %; 0.68–0.70 | MATCH | For the five rows that record them (39, 42, 45, 50, 51): shares 55.2–55.3 / 13.7 / 31.0–31.1; residual 0.682–0.700. **Rows 40 and 41 print "—" for both**, so for them "share one fingerprint" rests on the gap alone. |
| E5.4–E5.7 | evaluation 42: two TSO recoveries; 31: −2.5 k€, same split, 0.027; 6: growth test, opposite sign; "two others" | MATCH | Lapse resets at 218 and 222. Evaluation 31: 54.6/14.5/30.9, 0.027. Evaluation 6: +1.1 k€, growth test after five TSO recoveries. E.24 has 7 gap-refused rows + 42 + 6. |
| E6.1 (2692) | 42 = 3–11, 15, 22, 23, 26–31, 34–39, 42–59 | MATCH | 49 T2 cells − 3 B cells (rows 12–14) − 4 references (1, 2, 40, 41) = 42. This equals W153's 42 and the frozen `certification_stats_included` set. |
| E6.2–E6.3 | 32 = 24 + 8; 10 = 5 + 2 + 3 | MATCH | Frozen status, branch and cause over the 42 |
| E6.4–E6.6 | median k* 174; k* − k0 median 65 (21–89); range/τ median 0.86 | DERIVED | Over the 32 certified: 174.0; 65 (21–89); 0.8603 |
| E6.7 | 32, 33, 60, 61 added later, all certified | MATCH | the A64 cells |
| E6.8 | 46 certificates | DERIVED | Registry 49 = exactly the certified rows of E.21. 49 − 1 superseded (row 16) − 2 twins (rows 56 and 61, both twins of row 2) = 46 |
| E6.9–E6.10 | ten ≥ 0.95τ = 2, 3, 5, 10, 11, 12, 14, 17, 20, 23; 36 at 0.94, monotone | MATCH | `at_or_above_0_95_tau.counted_cells` numbered by the E.21 map; `i_5a6a88b4` 0.9389, monotone |
| **E6.11** | "The two evaluations continued past their certificates, 14 and 3, moved by a further 0.74τ and 0.78–0.88τ" | **MISMATCH (count)** | The movements match: 14 is 0.738τ; 3 is 0.781τ, or 0.882τ on the last-pair reading. **But W163 N25 records a third: 12** (`b_2a0ba8b2`, a v3 run to 213 past k* 173, +17.67 € = 0.004τ). See (vii). |
| E6.12 | 21 non-clean cycles in 10 evaluations, all TSO | MATCH | `non_clean_after_N` over the 42: 21 cycles in evaluations 3, 4, 6, 8, 9, 15, 37, 38, 39, 42. W153 by family: TSO 21, DSO 0. |
| E6.13 | 119 acceptable exits, all DSO, counted clean | MATCH | W153 totals: 119 (DSO 119): 114 in evaluation 9 (`d_f759dd48`) and 5 in evaluation 22 (`c_156ce2d1`). See (vi). |
| R1, R3–R14 (2699) | 72 cycles; M2 Max, 32 GiB, macOS 27.0; thread caps; HSL serial; two DN blocks; 10×; 1.0–1.2 × 10⁻⁶; tolerances; 10⁻⁵ p.u.; 7.8 k€; 1.2 × 10⁻⁵ | MATCH | Via W163's committed checks N47, N56, N57, N55, N45, N46, N58, N59, N63, N60, N33, N64, N65. N63: the three values −1.036, −1.116 and −1.200 × 10⁻⁶ read 1.0–1.2 at one decimal. |
| **R2** | "the three single-scenario reference continuations replayed bitwise to their certification cycles" | **MISMATCH (wording)** | Replay was bitwise through 132, 112 and 87, the *old* (residual-rule) certification cycles (W101 G13/G19; Addendum 53 report l. 87). Appendix E calls k* the certification cycle and prints 181 and 172. See (viii). |
| R15 | "three" | MATCH | x0, unit, C* |
| S1–S3 (§4.6, l. 1681) | 55 comparisons; no sign change; one verdict change, row 34, 0.95× → 1.22×, in the returned plan's favour | MATCH / DERIVED | Over the 55: no sign change. Exactly one verdict change: row 34 (`L:y2030__n5_p0.25_e0.5__n7_p1_e3`), gross 0.95× within the bar, net 1.22× determinate, d_net > 0. |

## Interpretations (i)–(ix)

**(i) The 60 → 55 + 3 mapping: CONFIRMED.**
- T1 has 60 claims. Its 55 distinct comparisons are the 60 less the four `G:…:net` duplicates and
  `CHECK:headline_V_minus_I_settled`, which is row 1 with the sign of value − I.
- T10 has 4 claims. The 3 distinct ones are used, and the fourth, the g070 REPORT_BESIDE claim, equals row 58 at print
  (d_gross is bitwise equal).

**(ii) The plan behind every E.21 cell: CONFIRMED.**
- All 61 printed plans and years equal `candidate_canonical`, one-to-one.
- The expert's named decodings each equal the frozen candidate:
  - `yl_y2030` / `yl_y2035` = 7: 0.25/1 in 2030 / 2035;
  - the six `pb_*` = 0.25/0.5 at a single node in 2025 / 2030;
  - `ref:5ca4f86c` (row 40) = 2030, 5: 0.25/0.5; 7: 1/3.5;
  - `ref:e28de4ac` (row 41) = 5: 0.25/1; 7: 1/3.
- Conditions are read from each cell's own stage spec. The ageing labels agree with Table 4.3's values.

**(iii) The Rule column: CONFIRMED, with one qualification.**
- All 61 labels agree with the classification derived from the frozen `certifying_spec`:
  - "run" for the v6 stage, ext v3, A64 v1 and the uncertified v6-stage cells;
  - "record" for the v1 references, W118 v2 (including F2 40 and 41), v4, v5 and `d_c52e1670`.
- For the "record" rows, v6 applied to the record (W142 item 1) reproduces the printed k* (or the cap, for 40 and 41)
  on 15 of 16 rows.
- **The exception is row 16.** v6 on its record is *not decided within cycles 1..167* (1 veto). The "cert." k* 167 /
  0.99 / 4.5 it prints are the W118 v2 certificate. Its ^s and the note say it was superseded, but not that the rule
  does not certify it. This contradicts §4.7 l. 1689 (Blocked 2(e)).

**(iv) The six other box neighbours of x = 0 were evaluated by the search only: CONFIRMED (scoped).**
- The six W173 rows with no frozen-table row are exactly the six named: two-node 2025 {5,7} and {7,9}; two-node 2030
  {5,9} and {7,9}; three-node 2025 and 2030.
- **What was searched.** None of the six candidates is in the 61 frozen cells. None is in the cells of the six
  certification specs: v4 `e11fbc89`, v5 `ab32ffc9`, v6 `96c23404`, ext v3 `84775dc4`, W118 v2 `fc791891` and A64 v1
  `44a2dce8`. For ext v3 and A64 v1 the candidate was resolved through `candidate_key`.
- This is the same finding as W179c (e) (`3994ac64`).

**(v) T6: the unit of "NRF violations (n)", the consistency pass, and "warm": CORRECTED (the reading of the consistency
pass); the unit and "warm" are confirmed.**
- **The unit is confirmed.**
  - n counts violated rows `uncoord_no_reverse_flow[s_m, s_o, p]` of the DSO blocks
    (`uncoordinated_benchmark.py` l. 193 and l. 1566).
  - At SRP1 (single scenario) that is one per DN interface-hour of a (year, representative day) block.
  - So n = 29 and 70 are counts out of 36 × 24 = 864. Any positive violation counts.
- **The consistency pass is not as E.4 reads it.** In `p515_s53_w116_benchmark_nrf.py`:
  - the exceedances are counted in phase B, the consistency *re-evaluation*. There each DN is re-solved at the TN's
    actual interface voltage with its voltage, thermal and NRF rows deactivated, and the NRF row is then evaluated as a
    hard DN limit (l. 620, `uncoordinated_benchmark.py` l. 208);
  - a violation triggers phase C, the *sequential pass*: each DN is re-solved with the NRF rows at that voltage (l. 641)
    and the TN is re-dispatched;
  - the "Consistency (k€)" column is `pass_effect_eur` = Q(phase C) − Q(phase A), the effect of the pass.
- **So E.4's two clauses describe different phases.** "The consistency pass is the second distribution solve at the TN's
  interface voltage" combined with "the exceedance columns count the entries of that pass" puts the count and the cost
  effect on the same solve, and they are on different ones.
- **Further, the E.23 caption states neither of these:**
  - E.23's Q is the phase-C cost for every start of both arms (`arm_cost_source_by_start` = all
    `phase_C_sequential_pass`);
  - the curtailment columns are phase A (`curtailment_rows_are_phase_A`).
- A suggested reading is in the JSON (`interpretations.(v).evidence.suggested_reading`). The wording is the expert's.
- **"Warm" is confirmed.** `warm = UB.extract_model_values(planning, certified)`, with certified =
  `BENCH.COORDINATED['certified_models']` and `COORDINATED = SETTLED`, eval key `d110bd1a…`. That is evaluation 1 (the
  settled cycle-181 x = 0 cell). `report_v3.json` names `…/d110bd1a5977df1e_x0/certified_models.pkl`.

**(vi) "119 exits at IPOPT's acceptable level … all in distribution blocks": CONFIRMED.**
- An event is one block's final accepted exit in one cycle. The v5 classification is `classify_block_exit`, "ONE
  block's final accepted exit", with reason `acceptable_primary_within_factor`: Acceptable on the primary attempt, every
  metric within 10× (`settling_criterion_v5.py` l. 121, 233).
- The hooks list these per cycle. W153 sums the blocks over cycles after N, over the 42.
- The total is 119: TSO 0, DSO 119, ESSO 0. 109 are in DSO|5|2035|Winter and 10 in DSO|7|2025|Winter, which matches the
  two blocks named at l. 2699.

**(vii) The continued cells and their movements: CORRECTED (the set).**
- The movements are confirmed:
  - evaluation 14 (`b_4649234b`): v4 run to 193, past the v5/v6 k* 148, −3,351.3 € = 0.738τ;
  - evaluation 3 (`d_c52e1670`): v5 run to 198, past the v6 k* 150, −3,543.1 € = 0.781τ, or 0.882τ past 148 on the
    last-pair reading.
- **W163 N25 records a third: evaluation 12** (`b_2a0ba8b2`).
  - Its v3 run went 213 cycles, uncertified under v3 (`w132_resettle_v3` campaign_results: `cycles_run` 213, report-only
    γ k* 173).
  - It continued past the certificate at 173 that the bitwise v4 re-run and v6-on-record give: +17.67 € = 0.004τ.
  - That is structurally the same case as 14.
- W142 item 3 lists only 14 and 3 because its scope was "the 18 re-settle records of the W142 floor replay".
- So "the two evaluations … 14 and 3" should read three: 12, 14 and 3. The bound "at most 0.9τ" (§4.7) holds over all
  three.

**(viii) The "three single-scenario reference continuations": CORRECTED (wording).**
- **Which cells.** They are x0 (evaluation 1), the unit `n7_4h_e1` (evaluation 2) and C*. C* is not in Appendix E (no
  new certificate, cap 187, per the Addendum 53 report).
- **The Replay column.** T2 has no replay field for 1 and 2 (None), so E.21 prints "—", as the expert stated.
- **The replay cycles.** They replayed bitwise to 132, 112 and 87 (W101 G13/G19 via W163 N49), which the Addendum 53
  report calls the "old certification cycle". These are not the certification cycles k* that Appendix E prints (181,
  172).
- The record supports the claim; the E.6 wording conflicts with the appendix's own definition (E.21 caption: "k*/end:
  certification cycle").

**(ix) k₀ = 123 for x = 0 is not in T2; E.21 prints "—" with a note: CONFIRMED.**
- T2 `k0_run` is None for `ref:7aa017f0`.
- W168 `decision_primary`: N = k0 = 123, k* 181.
- The Fig. 9 caption (l. 1700) reads "from the first residual pass $k_0 = 123$ to the certification cycle $k^{*} = 181$".
- 181 − 123 = 58, matching §4.7 l. 1694 ("58 cycles after").

## Not confirmed

- **The E.6 reproducibility facts R3–R14 were not re-measured.** They are rated through W163's committed checks of the
  same fragments, read from `w163_paragraphs_v4_figure_check.json`. They were not re-read from the IPOPT logs or the
  machine. The E.6 wording differs slightly from the paragraphs_v4 text W163 checked; the numbers are the same.
- **The internals of the earlier-stage checks were not re-derived.**
  - The run lengths for (vii) were confirmed from the campaign_results of the v3, v4 and v5 stages only. W163 N25's Q
    movements were not recomputed from the per-cycle records.
  - The W101 G19/G13 replay gates were taken as recorded.
- **No other Appendix E evaluation was searched for continuation past a certificate.** Only rows 12, 13 and 14 (and 3,
  through W142/W163) were looked at. Row 13 (`b_0dd237f0`) has a v4 run only, stopping at its k* 173, so it is not a
  continuation. Under `w132_resettle_v3` it has a campaign spec only: no `campaign_results.json` and no `evals/`. The v6-stage cells stop at their k* by construction
  (W142 item 3 scope).
- **The main-text tables were not compared with Appendix E.** E.23's best-start Q was not compared with Table
  `tab:res_benchmark` in §4.5. That is W179b's scope (committed `8662ca72`).
