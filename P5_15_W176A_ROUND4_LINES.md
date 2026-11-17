# P5.15 W176a — the round-4 lines at Overleaf `8b76423`

**Worker, 2026-10-08. ZERO SOLVES, zero model builds, zero pickle loads; read-only on production code, specs, case
files, the frozen JSON and the manuscript clone.**

- Manuscript: `manuscript/6a67305f25e8348fb71380c3/` at clone HEAD `8b76423` (verified).
  - `main.tex`: sha256 `7aa9e105deccdf61cced665f03d95b02ea736fe5c34f3402f6281db89d19624b`, 2,156 lines.
  - `response_to_reviewers_draft.tex`: `b7a269d567139478…`.
  - No file of the clone is modified. `c71ca2e` → `8b76423` changed main.tex l. 858 only.
- Line numbers are at `8b76423`. Code lines are at repo HEAD `bf4b626c`. No production file has changed since
  `a8eb9b69`, where W174b proved code identity with every table campaign.
- Authority: Addenda 71–72 and `STEP6_ROUND4_CORRECTIONS.md` (sha256 `8f2a91a6…`). Rating scale as in
  `P5_15_W174B_REAUDIT.md`:
  - **H**: a referee re-deriving from the text gets a different model or number;
  - **M**: imprecise, but not a different model;
  - **L**: notation or wording.
- Helper `p515_s53_w176a_round4_lines_checks.py`:
  - guards: `SolveProfileGuard(permitted=())` verified at exactly 0; pickle blocked, 0 calls; no production module
    imported; exit 0;
  - outputs in `data/SRP1/Results/P515S53/w176a_round4_lines/` (see "Changed").
- W100 typing test: PASS (129,326 files).

## Blocked on Planner

Nothing blocks the check. Three points for the Planner:

1. **Prediction scoring (expert's, Addendum 72).** I count two new L rows, and neither is on the round-4 "Final pass"
   list. Both are wording in the expert's own round-4 text:
   - **R4-1, item 1:** "each is described by its recorded certificate". s51 has no recorded certificate.
   - **R4-2, item 2:** "the coarser resolution rule". This is false for every F2 comparison.

   If the Planner rates both as wording to take at the whole-paper read, the prediction holds in substance. As
   scored here, it **fails on its second clause**.

2. **The letter still carries an author placeholder.** `response_to_reviewers_draft.tex` l. 174 (R2.8) reads
   "[AUTHOR: one clause per reference saying what it contributes and where it is cited. Status: …
   10.1109/ISGTEUROPE62998.2024.10863557 and 10.1016/j.apenergy.2022.120569 not yet added.]".
   - It is in-text, not a `%` comment, so the exact token `[AUTHOR]` counts 0 (section (c)).
   - It has been present since `6191c6c`. It is not a round-4 line.
   - Addendum 72's "no `[AUTHOR]` comment remains" holds for comments only.
   - It is the author's: the R2.8 references.

3. **The `% [AUTHOR]` note under the ageing table was deleted.** It was removed between `260bd83` and `c71ca2e`.
   - The round-4 file said it "stays until the citations are in", and the citations are not in.
   - Addendum 72 records "no `[AUTHOR]` comment remains" and leaves the datasheet citations to the author's final
     pass.
   - I take Addendum 72 as superseding. Recorded, not counted.

## Prediction vs outcome

| prediction (expert, Addendum 72) | outcome |
|---|---|
| W176(a): no M or H row among the round-4 lines | **HELD** (H 0, M 0) |
| W176(a): any L row is already on the round-4 "Final pass" list | **FAILED as scored here.** 2 new L rows, R4-1 and R4-2, are not on the list (Blocked 1). The other round-4-adjacent L points are already on the list: the literal "Section~4" in the immediate sentences of items 1, 3 and 9 ("Literal 'Section~4.x' references"), and the k₀ hold origin in item 13 ("Hold origin") |

**Presence: every round-4 edit is in, 14 of 14, plus the l. 858 reference.**
- 21 checks are verbatim, counting sub-parts: items 1a/1b, 4a/4b, 6a/6b/6c, 7a/7b, 13a/13b, 14a/14b.
- Every old wording is gone (17 old texts checked).
- The two `% [CONFIRM — W175]` comments are gone.
- **No row is H or M.**
- The round-4 lines resolve the following earlier rows:
  - Addendum 71's H at l. 647;
  - W174b's M rows NEW-1 and NEW-3 and the T1-4 residual;
  - W174b's L rows NEW-2, NEW-4, NEW-5, NEW-6, NEW-7, NEW-8, NEW-9 and NEW-10;
  - the two "elsewhere" rows: "representative year" and "optimal status".

## Per-edit table

Presence was searched with all whitespace removed. "Old" is the wording the corrections file replaces. Items 4a/4b,
6c and item 9's old sentence are not quoted verbatim in the corrections file; the script names their sources.

| item | line (8b76423) | (i) presence | (ii) checked against | status / row | (iii) L row on the final-pass list? |
|---|---|---|---|---|---|
| 1a "inadmissible at every **evaluated** poll" | 445 | verbatim; old absent | `poll_history` of s47 (`906f9da3…`) and s51 (`2cdd5d60…`). Admissible directions per poll: s47 0/0/0; s51 0/0/0/0/**1**. s51 poll 4 was refused at the cap with nothing evaluated | consistent: every executed variant-A poll had all 8 directions inadmissible (NEW-4 resolved) | — |
| 1b "two stopped when a unit poll failed and one was stopped for review…" | 445 | verbatim; old ("stopped at their evaluation budgets") absent | `termination.reason`: s47 `mesh_local_optimum_unit_poll_failed` (14/20 new); s51 `STOP_FOR_REVIEW_completion_cap` (completion 61 > cap 30; 20/20); s53 `poll_failure_at_unit_mesh` (7/60; `61b9c501…`) | stop reasons consistent (NEW-3 M resolved). **R4-1 (L):** "each is described by its recorded certificate" — see the new-rows table | **new — not on the list** |
| 2 l. 647 rewrite | 639 | verbatim; old ("accepts a poll point … only by a determinate margin") absent | `p515_s47_phase_b_record.py:519` `max(bar_x + bar_inc, sigma_q)`, stated in Algorithm 1 l. 434. The reported values are re-evaluations: x = 0 → `ref:7aa017f0` (w101, certified k\* 181); 5ca4f86c → `ref:5ca4f86c` (w118 r2 f2_incumbent). The frozen claims are `590088fe` `tables.claims` | rule as run, and the re-evaluation statement, consistent: Addendum 71's H and the T1-4 residual are resolved. **R4-2 (L):** "coarser" | **new — not on the list** |
| 3 "at the end of the block" | 671 | verbatim; old absent | `shared_energy_storage_data.py:628`: E^Av = E^Rated · SoH_{y_inv,y}, the end-of-block SoH, block width Y = 5. The mid variant is at :625 | consistent | — (the same sentence's literal "Section~4" is on the list) |
| 4a l. 748 → `sec:case_ess_params` | 748 | verbatim (full sentence); old absent | `\label{sec:case_ess_params}` at l. 1059 (§3.5). l. 1062 prints η^Ch/η^Dch, SoC^Min/Max/0, ε^Cl, c^Cl and ε^C | consistent (NEW-8) | — |
| **4b l. 858** → `sec:case_ess_params` | 858 | verbatim; old absent (the only change in `8b76423`) | l. 1062 prints ε^E = 10⁻⁵ and c^σ = 10³. The α reference (l. 899) and the recovery-settings reference (l. 1605) stay on `sec:case_settings` (l. 1092), which prints both | consistent (NEW-8) | — |
| 5 "\eqref{eq:soh_chain}, the available-energy product" | 833 | verbatim; old absent | — (grammar) | consistent (NEW-10) | — |
| 6a chemistry, §3.5 | 1062 | Addendum 72's text, verbatim; "lithium iron phosphate" absent | `SRP1_ESS.xlsx` (`e17bd588…`), all cells scanned: one chemistry string, "Li-Ion Battery cabinet" (sheet "Cost breakdown NREL", A2). `SRP1_ESS_Params.json`, `SRP1.json` and `SRP1_params.json` have none. Bib `nrel_ess_costs` = NREL, "Utility-Scale Battery Storage, 2024" | consistent: lithium-ion is the chemistry the input records. No LFP / iron phosphate / LiFePO in main.tex, the letter, the cover letter or the highlights | — |
| 6b letter R1.2(iv) | letter 76 | verbatim; "lithium iron phosphate (LFP)" absent | as 6a. "Section~3.5" = §3.5 Shared Energy Storage Parameters (§3's 5th subsection) | consistent. The untouched continuation "and the abstract names the chemistry" stays true (abstract: lithium-ion). "Datasheet readings": no datasheet is recorded for the baseline's 10,000 cycles (W175). The manuscript's l. 1066 says the same; Addendum 72 leaves the citation to the author — not a row | — |
| 6c abstract | 143 | "utility-scale lithium-ion battery storage" present | as 6a | consistent | — |
| 7a C4 row footnote mark | 1083 | verbatim | — | consistent | — |
| 7b caption footnote | 1073 | verbatim | ext spec v3 `84775dc4` C4 arm: `eol_retention_r` 0.7, φ 1.0, point end, on the base calibration (`cycles_n` 10000, `reference_dod_d` 0.8). Entered triple (10,000, 0.80, 0.70). k = 8,000/(−ln 0.7) = 22,429.386 = `k_closed_form_by_arm.C4`. The printed triple (8,000, 1.00, 0.70) gives the same k; printed 22,429. `shared_energy_storage_parameters.py:91`; eq. (cycle_life_calibration) l. 824–825 | consistent. "which is all … uses" refers to the (N, δ) pair. R^DS also enters, but it is 0.70 in both triples and printed in the row — not a row | — |
| 8 "Every certified evaluation reported … earlier states of the code, without the tight tail…" | 1095 | verbatim; old ("Every recourse evaluation ran…") absent | **Certified cells:** the frozen tables' 49 certified cells come from 7 campaign roots (w101, w118, w137, w139, w142 v6, w142 ext v6, w155). Every `git_head_at_run` shows no change against HEAD on the 20 production modules + SRP1 case data (control `353e094b` live: 5 files). Every spec declares the tail (`compl_inf_tol` 1e-6, enabled) and AA (memory 5). **Searches:** heads `353e094b` / `c39c4836` / `e1f98be8` differ from HEAD (5 / 4 / 3 files); no spec declares the tail; 0 tail occurrences at those heads. **Incumbents:** both re-evaluated as in item 2 | consistent (the T1-10 / Addendum 70 decision 4 claim is now scoped). Appendix A counterpart: 13b | — |
| 9 benchmark replacement | 1120 (paragraph 1119–1121) | verbatim (the latex block); opening clause kept; "Each arrangement is reported…" absent; CONFIRM gone | Frozen benchmark: instance x0 (`8435c718…`, all nodes 0), SRP1 single scenario; `uncoordinated_benchmark.py:264` `require_single_scenario`. The six NRF run records (`6dd2b7a3`, `17332cc7`, `d702e0a0`, `b6b34331`, `b0362c51`, `d3e2194a`): **36/36 DN blocks of DNs 5/7/9 triggered** the consistency pass in every run; `arm_cost.source` = `phase_C_sequential_pass` in all six; evaluation tie-breaker 0.0; passive decision tie-breaker DSO 1.0 (€/MWh), price-taker 0.0. Best of three: passive 895,049,918.47 (perturbed), price-taker 744,770,310.59 (warm) | consistent. Pass magnitudes (−44.0 / −69.8 / +0.234 k€) not printed; the paragraph's numbers are "4.4" and "1" only. Note: the code's trigger is global (any DN-block violation re-solves every DN: `uncoordinated_benchmark.py:1605`; harness `p515_s53_w116_benchmark_nrf.py:637–641`), while the text says "where a DN limit is violated". The two coincide in every reported run (all 36 blocks violated) — not a row | — (the paragraph's "Section~4.4" is on the list) |
| 10 App. A preamble | 1471 | verbatim; old continuation absent | ρ balancing, AA, tail: W174b anchors (`shared_resources_planning.py:3406, 3275, 8687`) | consistent (NEW-5) | — |
| 11 κ^E sentence | 1488 | verbatim; "unscaled in every agent" absent | `shared_resources_planning.py:5515–5518` (`admm_esso_al_scale` multiplies the agent's AL terms only). argmin f + κ·AL = argmin f/κ + AL for κ > 0 | consistent (NEW-6) | — |
| 12 "solved (optimal or acceptable)" | 1595 | verbatim; "an optimal status" absent | `helper_functions.py:74–85` (status ok and termination ∈ {optimal, locallyOptimal, globallyOptimal}); `shared_resources_planning.py:7734` (every TSO/DSO/agent block). Pyomo 6.9.5 `opt/plugins/sol.py:117–121` maps AMPL codes 0–99 → optimal/ok | consistent. The IPOPT code of an acceptable exit is not re-verified (as in W172) | — |
| 13a AA-off clause restored | 1601 | verbatim; truncated form absent | `admm_anderson_acceleration.py:477` ("off (all channels within Boyd tolerance)"). In the certifying regime: `p515_s53_w118_resettle_hooks.py:526` (`c > self.first_pass`) and `:722` (forced all-pass → AA off); imported by the w142 v6 / ext v6 hooks (`:33`) | consistent (NEW-1 M resolved; Algorithm 2's "(unless off)" now has its antecedent). "after k₀" = after the first residual pass (§2.2.7 l. 616: the regime opens at k₀ and "the holds stay") | — (the k₀ / first-pass origin is the "Hold origin" entry) |
| 13b "enabled in every campaign behind the reported tables" | 1605 | verbatim; old absent | 49/49 certified cells' specs declare the tail (as item 8). W174b: every table campaign incl. the 3 × 3 pair (46 heads). The searches have no tail | consistent (NEW-7) | — |
| 14a initialisation clause | 1632 | verbatim; old absent | `shared_resources_planning.py:2925–2926` (prepare objectives), then `:2931` (convert). `:5340` / `:5050` settlement weight 1 in every instance; `:5343` row 18 activated with it (not constructed at one scenario, W174b `model_construction_helpers.py:1917`) | consistent (NEW-9) | — |
| 14b `$k \gets k + 1$\;` restored | 1655 | present once | Structural: the line before is `}` closing `\If{…}{\textbf{exit}\;`; the line after is `}` closing `\While{$k \le k^{\max}$}` (l. 1636). Code: the loop counter at `shared_resources_planning.py:3066` (W174b) | consistent (NEW-2) | — |

### New rows

| row | line | text | evidence | how it differs | impact | final-pass list |
|---|---|---|---|---|---|---|
| **R4-1** | 445 | "…and one was stopped for review when its completion set exceeded the cap; each is described by its recorded certificate." | s51 `campaign_results.json` (`2cdd5d60…`): `termination_certificate` **null**, `STOP_FOR_REVIEW` true. `claim_scope`: "on any other termination no mesh-local claim is made". The F2 plan's statement in the same paragraph comes from s53's certificate (`61b9c501…`), the variant-B run that continued from s51's incumbent | the run stopped for review has no certificate; it is described by its stop record and continued by s53. No number changes | **L** | **not on the list** |
| **R4-2** | 639 | "The master search … accepted its poll points by the coarser resolution rule stated there" | Search threshold max(bar_x + bar_inc, σ_Q) ≥ σ_Q = 18,449.66 at every evaluated point (s47 18,449.66–47,205.25; s51 and s53 18,449.66 at every point). §2.2.7 thresholds in the frozen claims: 37 certified pairs 9,734.72–13,571.15 (≤ 3τ = 13,617.20); 23 uncertified-form 18,751.08–41,380.26; the 14 F2 neighbour claims (L:) 27,703.27–28,205.57 | "coarser" holds against the certified rule max{3b, 2τ}: the whole x = 0 search. It **fails** against the uncertified form, which governs every reported F2 comparison, because the F2 reference is uncertified: there the search accepted at 18,449.66 against a reported bar of 27.7–28.2 k. A rule is stated, and correctly; only the comparative is wrong for one of the two searches | **L** (my rating; M is arguable, since it is the F2 case the sentence covers) | **not on the list** |

Note, not a row: "every cell it proposed was re-evaluated under this rule before being reported" (l. 639) is consistent
if "proposed" means the incumbents, as l. 445 says ("it proposes the incumbents"). On the wider reading (every
evaluated poll point), 10 evaluated points were never re-evaluated:
- the 6 multi-node neighbours of x = 0, at 3.47–7.17× the search resolution;
- the 4 F2 2035 points, at 24.2–27.7×.

These enter only l. 445's scope counts ("fourteen", "seventeen"), not a reported value (W173 E2.1, E2.3).

## (c) Literal section-4 references in §2–§3; CONFIRM / AUTHOR

**Search pattern:** `(?:Sub)?[Ss]ections?(?:~|\s)4(?:\.\d+)?|Sec\.(?:~|\s)?4(?:\.\d+)?`. It covers "Section~4.x",
"Section 4.x", "Sections~4", "Subsection~4.x" and "Sec.~4" over every line of main.tex. The section ranges come from the
`\section{` lines: §2 is l. 315–904, §3 is l. 905–1125.

**8 occurrences in §2–§3, all in text, none in comments:**

| line | § | text |
|---|---|---|
| 377 | 2 | "…the remaining calendar life of each installation (Section~4)." |
| 415 | 2 | "…the budget as the sensitivity study of Section~4, and their combinations)" (Algorithm 1) |
| 445 | 2 | "…the resolution the search used is reported in Section~4.7." |
| 631 | 2 | "…and the ratio of Section~4.5 is reported as a band that allows for it." |
| 635 | 2 | "…evaluations continued past certification moved by at most $0.9\,\tau$ (Section~4.7)." |
| 671 | 2 | "…the mid-block evaluation point is a sensitivity in Section~4 (…)" (item 3's sentence) |
| 1066 | 3 | "The other rows are the sensitivities of Section~4.3: …" |
| 1119 | 3 | "The value of coordination (Section~4.4) is measured against…" (item 9's paragraph) |

The pattern finds no other occurrence anywhere in main.tex: none in §1, §4, §5 or the appendices.

**Comments.** Exact-token counts (`[CONFIRM`, `[AUTHOR]`, `[AUTHOR:`), plus case-insensitive line regexes
`\[\s*confirm` and `\[\s*author`:

| file | `[CONFIRM` | `[AUTHOR]` | `[AUTHOR:` | regex hits |
|---|---|---|---|---|
| main.tex | **0** | **0** | 0 | `[authoryear` in the commented `\documentclass` line 25 (not a note) |
| response_to_reviewers_draft.tex | **0** | **0** | **1** | l. 174, the in-text R2.8 placeholder (Blocked 2) |
| cover_letter.tex | 0 | 0 | 0 | — |
| highlights.tex | 0 | 0 | 0 | — |
| section2_expert_draft.tex (not compiled; `7e347dc8`, unchanged) | 11 | 1 | 0 | the draft's own notes |

## Not confirmed

- **The IPOPT return code of an acceptable exit.** That it is 1, and so falls in Pyomo's 0–99 → optimal band, is not
  re-verified. Searched: Pyomo's `sol.py` in the canonical environment. The committed solve records carry the IPOPT
  message ("Solved To Acceptable Level.") but not the Pyomo termination. Same limit as W172.
- **The NREL ATB page.** Not opened, so whether the ATB 2024 category is LFP-specific or generic lithium-ion is not
  established. The bib entry has no URL and does not say "ATB". The cost file names only "Li-Ion Battery cabinet",
  under a sheet titled "Cost breakdown NREL".
- **Baseline datasheet.** Searched the repository and the inputs; no source is recorded for the 10,000-cycle count
  (W175). The letter's "datasheet readings" cannot be checked against a record.
- **Reported uncertified evaluations.** Item 8 is scoped to certified evaluations, so the identity check covers the 49
  certified cells. The uncertified reported cells were not re-checked here: they are in the w142 v6 root (covered by
  W174b's 46 heads) and the two w118 F2 cells. Neither changes the sentence.
- **Compiled output.** No PDF; the footnote mark, the caption text and the algorithm layout were checked in source
  only.
- **Scope of the negative claims.**
  - "No [CONFIRM / [AUTHOR] left": the five files listed, every line.
  - "No other literal Section~4": the regex above, every line of main.tex. A reference written another way (e.g.
    "the results section") is not caught.
  - "No chemistry other than lithium-ion": main.tex, letter, cover letter and highlights for
    lfp | iron phosphate | lifepo | lithium | li-ion; inputs as in item 6a.

## Unexpected findings

1. The letter's `\rchanges{}` lines still carry bracket placeholders, e.g. l. 168 (R2.5/R2.6)
   "Section~[limitations]; Table~[T2]". Seen while reading l. 174; not searched systematically; not in scope.
2. Item 9's text says "where a DN limit is violated", but the code's trigger is global (per-table row note). If the
   benchmark is ever re-run on an instance where only some DNs violate, the text and the code would differ.

## Changed

- New script `p515_s53_w176a_round4_lines_checks.py` (sha256 `5842fbf745e32584be7874d694588d351bfc0008d37fdc8874daae528ed25ff0`).
  Run at repo HEAD `bf4b626c`, canonical interpreter, attached, both streams to `launch.log`, exit 0.
- New outputs in `data/SRP1/Results/P515S53/w176a_round4_lines/`:

  | file | sha256 |
  |---|---|
  | `w176a_checks.json` | `7f01c3b69e8de707026e635ee80ff6e810e8b9f508df7071c4887edf5168887f` |
  | `manifest_sha256.json` (output, script, every input) | `e64935203e181b05848067b8ffd15034a5bcfb8482843af9495b122677c1c028` |
  | `launch.log` | `1edaa6ceb8cc0bec1a54fdd9a63080dfee14e71c5def6a2b2a83831582cb41c1` |
  | `w176a_bool_typing_test.json` | `4c30a7d51a9d9994b576314beebd2d419999679c190eeab40ce4c65a60f62c64` |
  | `w176a_bool_typing_test.log` | `31a62d90e607d454ad54596fc4188d6eaf2345f4a90cb5eb0099d3fb4135892e` |
  | `manifest_post_run_sha256.json` | hashes of all of the above and the script |

- **A first run was discarded.** It completed with exit 0, but it misclassified the frozen claims' status strings and
  had a dead Pyomo pattern. Its three outputs were uncommitted and cited nowhere; I deleted them before the final run.
  Only the final run's outputs are listed above.
- This report.
- Nothing else: no production file, spec, case file, frozen artifact, committed artifact or clone file was touched.
