# P5.15 W182 — re-audit of Overleaf `edd7a63` and evidence inventory for Step 7 (Addendum 82 v2; zero-solve)

**Authority:** Addendum 82 v2 (W182 order). Addendum 81's W182 is withdrawn. This addendum authorizes no campaign run.

**HEAD audited: Overleaf `edd7a63`** (`edd7a63a6c2e…`).
- It is **67 commits** after `8f051a2`; Addendum 82 says eight, and TASKS.md said 66.
- `main.tex`: 2,956 lines, sha256 `b08a3365607e56f2…`, equal to the critical review's fingerprint.
- `git diff 8f051a2 edd7a63 -- main.tex` has 49 hunks (176 at -U0).
- The letter is the `d2ebb0d` version, headed v9; it contains Addendum 81's v9 sentences.

**Code head: repo HEAD `a17f6ba7`.**
- Tracked files were clean at the start.
- The last production-code commit is `575bfec4` (root-level non-`p*` `.py` files).
- Each Worker verified that no production, rule or case file differs between `a17f6ba7` and its run HEAD. The parallel
  Workers committed during the task.

**Objective convention:** Q = `gross_operational_cost` = TSO + DSO `objective_function_rule`, less the **contracted**
interface settlement and the voltage pin; net = Q − terminal salvage. See decision 2: "settlement excluded" is exact
only for the single-scenario instance.

**Runs:** zero solves and no model build in every part.
- `SolveProfileGuard(permitted=())` was armed before any project import and verified at exactly 0.
- Pickle was blocked: 0 loads, 0 unpickling.
- JSON was written through `gate_result_io`, with sha256 manifests.
- Each part ran attached, with both output streams captured.
- Each harness was committed before its real run.
- The seven parts ran in parallel, as static readings with separate scripts and directories. Scratch trials used
  uniquely named directories.

| part | items | report | commits |
|---|---|---|---|
| W182a | A1 | `P5_15_W182A_EQUATIONS_CODE.md` | `89171f00` |
| W182b | A2 | `P5_15_W182B_NUMBER_CHECK.md` | `b9643b84`, `7394eeee` |
| W182c | A3–A6 | `P5_15_W182C_LABELS_POINTERS_APPE.md` | `179c9355`, `6a9b1f85` |
| W182d | A7, A8 | `P5_15_W182D_CLAIMS.md` | `ab1d8d7b`, `7614bcbf` |
| W182e | Z1, Z6, C4 | `P5_15_W182E_CODE_PATHS.md` | `2f4df40c`, `7fa0177d` |
| W182f | Z2–Z5, Z7 | `P5_15_W182F_RECORDS.md` | `7e5dd80e` |
| W182g | C1–C3 | `P5_15_W182G_COST_HORIZON_BUDGET.md` | `a21b03e6`, `c39663e4` |

**Stop.**
- Prediction A6 (Appendix E bitwise) failed, and Addendum 82 names no fallback.
- Prediction Z7 failed in the favourable direction.
- Z1/Z6/C4 show that E1–E3 cannot run on the committed code without changes outside the frozen configuration.

No run is queued. The campaign spec is not frozen.

## Decisions needed

1. **Appendix E: the generator and the manuscript have diverged (expert). This is A6.**
   - **The finding.** `gen_appE.py`'s output occurs 0 times in main.tex at `edd7a63` (once at `8f051a2`; the Planner
     re-checked both). The author hand-edited the block in six hunks:
     - a new E.1 "Terminal Salvage Credit" (30 lines);
     - the E.21 group-(d) header;
     - the lattice figure and paragraph moved into E.4;
     - E.5 (1 line → 8);
     - E.7 (1 line → 20);
     - one blank line.
   - 0 of the 1,637 table entries changed. `verify_appE.py` still gives 33 PASS, but it never reads main.tex.
   - Regenerating now would bring back `\ref{tab:res_flex_ladder}`, a label removed at `edd7a63`.
   - **Options:**
     - (a) Port the six hunks into `gen_appE.py` and keep the bitwise gate. Cost: one expert patch, and a W-check of the
       next round.
     - (b) Declare the block hand-maintained, and replace the bitwise gate with the entry check alone (W180a/W181a
       method).
   - **Recommendation: (a).** The generator is the only mechanism that ties E.20–E.24 to `590088fe` automatically, and
     every later round will edit around it.

2. **The objective-convention label "settlement excluded" (expert; affects the manuscript and the project
   convention).**
   - `shared_resources_planning.py:1302–1304` subtracts only the **contracted** interface settlement and the voltage
     pin (the Planner read the lines).
   - At 3 × 3, the deviation settlement (x = 0: −748,746 €) and the DSO-only row-18 premium (+25,082,401 €, about
     3.0 % of Q) stay **inside** Q.
   - At one scenario the deviation part is 0, so the label is exact there.
   - Appendix E l. 2567 ("settlement excluded") and the convention line in TASKS.md and the brief carry the same
     wording. This agrees with audit §3.8. No number changes.
   - **Recommendation:** "contracted interface settlement excluded" everywhere the convention is stated, with one
     sentence in §3 / Table 13 saying that the deviation settlement and the row-18 premium are inside Q at 3 × 3.

3. **E1–E3 are not runnable on the committed code (expert rules the formulation points; then a Worker order per
   change).** Source: W182e (93 of 93 citations hold).
   - **(i) E1 no-ageing search.**
     - No-ageing is the attribute `SharedEnergyStorageData.ageing_enabled`, set only through the harness's
       `model_variant`.
     - Both committed Phase-B search launchers assert that there is no model variant (`p515_s47_phase_b_record.py:874–875`,
       which the Planner read, and `p515_s51_f2_phase_b.py:362–363`).
     - So a new launcher is needed. That is harness code; production is unchanged.
     - `eol_retention_r` enters the evaluation key, although it is inert without ageing: T8 used 0.50, and the
       baseline is 0.80. The expert should fix the value.
   - **(ii) E2 ρ₀ × 0.5 / × 2.**
     - In production ρ₀ is the case-file `admm.rho` (`admm_parameters.py:511–513`, read by the Planner).
     - The harness accepts only the `anderson_acceleration` override. Every child refuses a ρ different from the S39
       constants, and ρ is not in the evaluation key.
     - A declared, keyed harness option is needed. It could reuse `p59_rho.apply_rho_to_params`.
   - **(iii) E2 stage 2, perturbed coordinated start.** The benchmark perturbation touches only the static arms.
     ADMM builds and initialises inside `_run_operational_planning`, with no hook. Options:
     - (a) a production hook between build and the first solve;
     - (b) a perturbed `initial_state` continuation, which is a different start semantics;
     - (c) the existing `shared_ess_initialization='price_taker'`, which the harness refuses and which changes only
       the ESS channel.

     Addendum 82 made stage 2 conditional on Z6. **Recommendation:** defer stage 2 unless the expert accepts (c),
     labelled as such.
   - **(iv) E3 NRF option.**
     - No option exists in the coordinated path. One is needed on both copies: `pc_adn` (TSO builder) and `pg_adn`
       (both DSO builders). That is a production change.
     - **Formulation point.** `pg_adn` is the reference-generator P **minus the shared-ESS net power** at the reference
       bus (`model_construction_helpers.py:1287–1297`). For x > 0, "no reverse flow" on `pg_adn` and on the reference
       generator are different constraints. W116's row uses `pg_adn`.
   - **(v) Concurrency.**
     - The only concurrent-vs-serial gate (`p515_s44_gate.py`) is pinned to pre-baseline literals: it is stale.
     - Cross-concurrency bitwise replays on the final model code exist: W86 → W101, and s47 PB → W142.
     - All 49 final-code campaigns ran at concurrency 1.
     - Four slots need a launcher that declares them, a memory preflight, and a new gate against a serial reference.
       Peak RSS is 15.2 GiB per 3 × 3 cell (W182f Z4).
   - **(vi) What "final code head" means.** The campaign harness changed after `575bfec4` (`7914860f`, `b4c3dc7a`; 41
     rows ran a version 3 router lines older). E1's spec should pin the harness sha256 as well as the production
     head.
   - **(vii) Z1, empirically.**
     - Q(0) is exactly independent of the ageing model by code reading.
     - No x = 0 evaluation with ageing off exists among the 263 committed specs.
     - A build-only NL comparison of the two x = 0 models (no solve) would settle it.
     - **Recommendation:** make that the first E1 gate.

4. **E6 inputs for rows 1–2 (author: an artefact decision; expert: the pickle read).** Source: W182f Z5.
   - The ESS hourly P (all copies) and the reactive power exist only in the uncommitted
     `ess_entry_stride_baseline.jsonl`: 13.6 MB for row 1 and 17.9 MB for row 2. Both are hash-recorded.
   - The hourly SoC exists **only** in `certified_models.pkl`. Row 2's pickle is pinned at `d0a3d512…`.
   - **Recommendations:**
     - commit the two JSONL files unmodified (`git add -f`, about 31.5 MB; precedent: the committed 13.7 MB
       `interface_duals_per_cycle.jsonl`);
     - authorize a zero-solve read of row 2's pickle under an armed guard, with the pickle-block lifted for that file
       only.

5. **Manuscript, letter, highlights and cover letter (author / expert round 8).** Nothing below changes a number or a
   run.
   - **Letter v9 (A3): 65 of 173 pointers MISMATCH.**
     - 1 is AMBIGUOUS: l. 97 "Section 3.3". The old §3.3 was split into Appendix C (C.12) and the merged §3.3.
     - The main shifts: §2.2.7 → 2.2.6 (11 places), §3.5 → 3.4 (7), §2.3.4 → 2.3.3 (5), Tables 6/11/12/14/15 →
       3/7/8/10/11, Figures 3/4/7 → 2/E.19/5, Eqs. (38)–(39) → (E.1)–(E.2), E.6 → E.7.
     - Removed targets: Figure 2 (l. 125, 126, 288) and §2.2.6 (l. 136).
     - **Citations shifted.** `boyd_admm_2011` moved [36] → [44]. Naumann is now [42] (was [43]) and EVE is now [43]
       (was [44]), so the letter's [43]/[44] are wrong. This is static citation order, not compiled.
   - **Letter v9 misdescribes the abstract (A8/A7).**
     - l. 77 "names the chemistry": it does not.
     - l. 80 says 13.9 %: the abstract reports 12.2 %.
     - l. 76 "says both": it does not say the cost uncertainty is future technology cost.
   - **13.9 % called a saving.** 13.9 % = benefit / coordinated Q, the price-taker's cost increase. The saving is
     12.2 % = benefit / price-taker Q. It is still called a saving in:
     - `highlights.tex` l. 14 ("worth 13.9 % of operating cost");
     - letter l. 53 and l. 80;
     - `manuscript_review/highlights_keywords_v2.tex`.

     The frozen field is named `tables.benchmark.benefit_relative` = 0.139012; the name likely seeded the wording. Not
     changed.
   - **Claims (A7): 16 sentences listed.**

     | Where | Sentences |
     |---|---|
     | main.tex | 4: abstract l. 145 (×2), highlight 1 l. 163, contribution 1 l. 387 |
     | highlights.tex | 2: l. 12–13 |
     | Letter | 9 |
     | Cover letter | 1 |

     By category: multi-scenario search 3, stochastic value 3, timing 3, "certified" for uncertified 6, certification
     bounds error or resolves a ratio 7. No sentence states a preferred year.
   - **Out-of-budget points (A8).**
     - Seven evaluations of 2025 node-7 plans are over B = 1 M€:
       - 1/4, 1,271.8 k€ (evaluations 5 and 38);
       - 1.25/5, 1,589.8 k€ (6 and 39);
       - **the 2 h rungs**: 1.5/3, 1,146.1 k€; 2/4, 1,528.1 k€; 2.5/5, 1,910.2 k€.

       The audit named only the 4 h rungs.
     - They appear in **34 places; 1 is labelled** (Appendix E l. 2921) and 1 partly (l. 2797).
     - 5 of the 10 points of the §4.2 break-even fit are out of budget.
   - **The cover letter is still the submitted text** (bi-level, Benders, "significantly reduce operating costs").
   - **Wording (L).**
     - P^I direction and its ESS exclusion are unstated (W182a).
     - y^End = 2040 (1 Jan 2040) is not given, in either configuration (W182a, W182g).
     - "Load curtailment" is listed in Q (l. 724), but `l_curt` is false in every network file, so it is never built
       (W182g).
     - l. 1290: the ADN growth rates sit in the merged TN+DN section; the TN rates are 1.00/2.50 % (W182b).
     - l. 2919: the moved sentence points τ at `subsubsec:certification`; τ now lives in `eq:acceptance_tolerance`
       (W182b).
     - At 3 × 3 the SoH floor binds in 2037. l. 1573 "first binds in 2035" is single-scenario only (W182g).
   - **Lost `\textcolor{blue}` (A5).** The five captions of `fig:investment_lattice`, `tab:ageing_calibrations`,
     `tab:res_value_ladder`, `tab:res_energy_ladder_m2` and `tab:res_ageing`, plus the notes of Tables 4, 6 and 7. In
     all, 23 lines (list in W182c).
   - **Addendum 82's "two removed figures".** Only `fig:search_generic` was removed (commented out).
     `fig:investment_lattice` moved to E.19.

6. **Plan files not in the repository (author).**
   - Missing: `STEP7_SIMULATION_PLAN_2026-10-25.md` (plan v2), `simulation_study_brief_for_expert_2026-10-25.md` and
     `response_to_reviewers_v9.tex` / `letter_v8_v9.diff`.
   - The campaign spec is to be frozen after W182, so at least the plan must be committed first.

## Blocked on the author

- Commit the three files of decision 6.
- Push, if the terminal push of this commit fails.

## Changed

| commit | content |
|---|---|
| the parts' commits (table above) | seven harnesses, seven part reports, seven evidence directories under `data/SRP1/Results/P515S53/w182{a..g}_*/` |
| `7e5dd80e` (also) | Z5: row 2's `…/3f084f2ffaeef2b7_n7_4h_e1/esso_capture/s39_D/node7_cycle172.jsonl` committed unmodified; 2,238 uncommitted files of rows 1–2 hash-recorded in `z5_rows12_file_manifest.json` (pickles hashed as bytes, never opened) |
| this commit | `P5_15_W182_REPORT.md`, TASKS.md, REVISION_CONTEXT.md |

- No production, spec, case, frozen, clone or `manuscript_review/` file changed.
- No earlier harness was edited or re-run onto its output directory.

## Found — predictions against outcomes (expert, Addendum 82 v2)

| prediction | outcome |
|---|---|
| A1 at most one M row (interface-deviation sign) | **held** — 3 MATCH, 2 L, 0 M, 0 H. The named M was not found: the code's sign equals the printed one (`model_construction_helpers.py:1866–1867`, read by the Planner). All five items describe code that predates the edits. The printed salvage formula reproduces all 49 records to 9.5 × 10⁻¹¹ €. τ = 0.07 × 259,375.33 / 4 = 4,539.07 € (`settling_criterion.py:55–57`), used only in the stopping and comparison rules; the three searches ran before τ existed (their threshold was max(bar sum, σ_Q)) |
| A2 (no prediction) | 825 tokens on 665 `+` lines: **701 MATCH, 31 DERIVED, 0 MISMATCH, 0 UNSOURCED**, 93 not quantities. The headline 12.2 % = 90,896,608.40 / 744,770,310.59 is DERIVED (Planner re-computed 0.12205); all 30 cells of `tab:res_capacity_summary` MATCH; negative control 732/732 refused |
| A3 ≥ 10 letter pointers MISMATCH | **held** — 65 (decision 5). Label map: 132 → 136 labels; 71 unchanged, 59 renumbered, 2 removed, 6 added. main.tex has 0 undefined refs |
| A6 bitwise, 33 PASS | **bitwise failed** (decision 1; Planner re-checked); **33 PASS held** |
| A7 ≥ 6 sentences listed | **held** — 16 listed, 11 borderline |
| Z1 Q(0) exactly independent of the ageing model | **held by code reading** — at x = 0 every ageing-dependent row is deactivated and D/SoH/pch fixed; salvage is outside every objective; no division by S^Av/E^Av is reachable. Not measured (decision 3 vii). Caveat: 'mid' mode carries φ^(n/2) at x = 0; E1 does not use it |
| Z2 row 40 not reachable within the cap | **held** — cap 281 (N_old + 100), all used, 0 remaining. v6 on record is uncertified over 1–281, with `settling_criterion_v6.replay` reproducing it; the gap clause alone refuses (45 refusals at 237–281, \|t_sum\| ≥ 4.06 × τ/2) |
| Z6 ρ₀ a configuration parameter | **held in production**; the harness refuses ρ ≠ S39 (decision 3 ii) |
| Z6 NRF needs a new declared option (code change, both copies) | **held** (decision 3 iv) |
| Z7 the four reverse-flow hours recoverable from `certified_models.pkl` only | **failed (favourably)** — all four are in the committed `interface_settlement_detail_s31c.json`, node 7, 2025: Autumn h19 −3.157 MW, Spring h22 −2.402, Spring h2 −1.602, Summer h22 −0.067; weights 455/460/460/455; prices 161.80/134.53/117.73/131.72 €/MWh; bitwise equal to W116's pickle-derived list |

**Other findings with no prediction.**
- **Z3: code heads of the 61 E.21 rows.**
  - 0 ran at `575bfec4` itself.
  - 45 (all the v6 runs) have production modules identical to it.
  - 14 differ only by settling-rule modules added later.
  - Rows 1–2 also differ in `uncoordinated_benchmark.py`, which is not on their import path (static check).
  - Configuration-hash groups: 1–29 baseline, 30–31 (m 1.5), 32–33 (m 1.75), 34–53 (m 2), one each for 54–61.
  - Row 40's `evaluation_record` says "certified". That is the harness's residual certificate, not the rule
    decision.
- **Z4: wall time and memory.**
  - 3 × 3 cells: median 373–414 s per cycle, with cycle 1 about 300 s slower; peak RSS 15.16–15.22 GiB.
  - A five-block single-scenario build is not recorded in the 54 files searched.
- **C1: cost boundary.**
  - The term list of `objective_function_rule` is complete (`model_construction_helpers.py:1813–1842`).
  - RES-curtailment and ESS-usage penalty weights are 0 on the ADMM path.
  - The investment cost, salvage, contracted settlement and voltage pin are outside Q.
  - The augmented-Lagrangian terms and the whole storage-agent objective sit in the ADMM local objectives only.
  - Q reconciles with its parts to 6 × 10⁻⁷ €.
- **C2: horizon end.**
  - y^End = 2025 + 15 = 2040 in both configurations. The last block is 2035 (covering 2035–2039) or 2037 (covering
    2037–2039).
  - Terminal discount is 1.02⁻¹⁵ = 0.743015. Terminal SoH is the end-of-block SoH of the last block.
  - Salvage reproduced exactly: 11,759.26 € (unit in 2030), 56,504.17 € (unit in 2035), 0 at 3 × 3 (SoH at the 0.70
    floor).
- **C3: budget admissibility.**
  - B = 1,000,000 € (`SRP1_ESS_Params.json` `budget`; enforced at `shared_resources_planning.py:2266`).
  - Table 1 matches the cost file in all 18 cells and 3 probabilities.
  - Of the **47 distinct plans named in main.tex**: 42 admissible, of which 4 sit at 992,268.11 € (0.77 % slack,
    including the returned 2× plan); 5 inadmissible (the 2025 ladder probes of decision 5).
  - The audit's 992.3 k€ is confirmed, as are the counts 12 / 14 / 61.
  - No returned plan and no certified comparator is out of budget. Algorithm 1 admits only budget-feasible
    incumbents.
- **C4:** see decision 3 v.

## Not confirmed

- **Compilation.** Nothing was compiled; there is no TeX here and no `.aux`/`.bbl` in the clone. Label, figure and
  citation numbers are static (`elsarticle.dtx`, citation order).
- **Letter v9's content** was not compared with the expert's own copy, which is not in the repository. Each pointer's
  intended target is taken as the object holding that number at `8f051a2`.
- **Z1** rests on code reading. No x = 0 no-ageing NL build exists.
- **C4.** The process-level shared writes of the W101-and-later settling hooks at 4 slots were not traced.
- **Z3.** The HEAD at each run's launch is not recorded; equality was judged at the run-end head. The import check is
  a static regex.
- **A7/A8 coverage.** The listed / borderline / excluded assignments are the Worker's judgement, with reasons recorded
  in the JSON. §4.6 and the rest of the Introduction were scanned only for the keywords.
- **C3** covers main.tex only, not plans named in the letter, highlights or cover letter. W182g did not read
  `uncoordinated_benchmark.py`, so the benchmark's own cost boundary is unchecked.
- **A2.** 25 MATCH tokens rest on the committed W163 registry and were not recomputed. Figures were checked by
  caption and, for Fig. 4, by PDF identity.
- **Not examined by any part:** `main_latexdiff.tex`, which is of `8f051a2` and now stale.
