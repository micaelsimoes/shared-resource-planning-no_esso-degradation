# P5.15 Addendum 73 — W177a: check of the expert's §4 frame `section4_expert_draft.tex` (zero-solve)

Frame: `section4_expert_draft.tex` (repository root, committed in `4f941b29`, sha256 `e73b6d91…`, not edited).
Script: `p515_s53_w177a_section4_frame_check.py` (commit `d8cb49c2`; new file; imports W176b → W174a → … → W160, none
edited). Outputs: `data/SRP1/Results/P515S53/w177a_section4_frame/` (write-once; JSON `76266f7f`, MD `e966d594`,
manifest `96701576`, launch.log `6cd213f0`, typing test `c6fd15a3` / `e8248ff8`, post-run manifest). ZERO SOLVES:
`SolveProfileGuard(permitted=())` plus the 37 guards the W176 chain arms (38 in all), all verified at 0; pickle blocked, every
counter 0; exit 0; 13/13 integrity checks True (frame sha = declared = its blob at HEAD; frozen JSON `590088fe`
unchanged; clone HEAD `8b764234` and main.tex `7aa9e105` as declared, clone `*.tex` clean, never pulled or modified;
every declaration found and evaluated; every one of the 1,057 tokens assigned; 43 inputs committed clean). W100 typing
test PASS (129,334 files).

**Objective convention (every figure below).** Q = `gross_operational_cost`, settlement excluded (the frozen
primary); net = Q − terminal salvage credit; value = Q(0) − Q(x); F = Q + I; amounts in k€ unless stated.

## Blocked on Planner

1. **The salvage equation needs two corrections (expert's formulation — the expert rules).** `eq:salvage_credit`
   (l. 28–35, where-clause l. 42) does not match the production credit behind every reported net. Items (a)–(d) of
   the l. 43 comment all hold as worded and `eq:salvage_remaining_life` holds; but (i) the cohort cost is the expected
   **energy** cost only (`cost_basis EXPECTED_INSTALLATION_ENERGY_COST`; no c^{Inv,S} S term), and (ii) the health
   factor is **(SoH − SoH^Min)/(1 − SoH^Min)** (`health_basis NORMALIZED_ABOVE_MINIMUM_SOH`), not SoH. Proof on records:
   the production form reproduces the recorded salvage of `yl_y2030` / `yl_y2035` exactly (11,759.26 / 56,504.17 €;
   relative difference 0.0); the frame's equation gives 49,624.20 / 104,914.88 € (4.22× / 1.86×). Proposed text:

   `C^Salvage(x) = Σ_e Σ_{y^Inv} [c̄^{Inv,E}_{y^Inv} E^Inv_{e,y^Inv} / (1+r)^{y^End−y_0}] · (T^Rem/T^Cal) ·
   (SoH_{e,y^Inv,y^End} − SoH^Min)/(1 − SoH^Min)`, with c̄^{Inv,E}_{y^Inv} = Σ_c ω_c c^{Inv,E}_{y^Inv,c}, and the
   where-clause "the expected undiscounted **energy** cost of the cohort". This uses the configured
   `energy_recovery_fraction` 1 and `recycling_floor_fraction` 0; the code's general form is
   f_rec·[φ·E + (1−φ)·E·(SoH − SoH^Min)/(1 − SoH^Min)].
   The same equation is in `manuscript_review/section2_part_d.tex` and `manuscript_review/section2_expert_draft.tex`
   (Part D.2). It is not in main.tex at `8b76423`.
2. **The F2 plan is misdescribed (l. 131).** Behind `ref:5ca4f86c` the plan is **2030, node 5 0.25 MVA / 0.5 MWh and
   node 7 1 MVA / 3.5 MWh**, not "node 7 1 MVA / 3 MWh". Corrected text: "node 5 0.25 MVA / 0.5 MWh and node 7
   1 MVA / 3.5 MWh". (If the plan were 1 MVA / 3 MWh, it would be the same plan as its own L neighbour
   `y2030__n5_p0.25_e0.5__n7_p1_e3`.)
3. **Addendum 64 ruling 4 vs `tab:res_energy_ladder_m2` (l. 202).** The ruling says "the manuscript uses the
   conservative figures (61.3 k€/MWh; **J at its uncertified bar**)", and the frozen claim `J:e4_value_minus_I` carries
   the note "the MANUSCRIPT uses J 4 MWh at its uncertified bar". The frame's 4 MWh row prints the flagged certified
   threshold (9.7 k€, 18.14×) in its main columns and puts the uncertified bar (15.9 k€, 11.12×) in the caption. The
   numbers match T1; which form goes in the main column is for the Planner or expert to rule. No verdict changes.
4. **Recorded predictions: 2 of 3 failed** (next section). Addendum 73 names no fallback. These are text-check
   predictions and no run depends on them; I have recorded them and stopped. W177b stays held, as Addendum 73
   orders.
5. **Addendum 73's author step list for literal references needs updating** (§ Labels/figures/references below). At
   `8b76423`, "(Section~4.4)" is at l. **1119**, not l. 1116. Three more literals are not on the list: l. 377
   "(Section~4)", l. 635 "(Section~4.7)" and l. 671 "Section~4". The targets I would suggest are
   `Section~\ref{sec:results}`, `Subsection~\ref{subsec:res_certification}` and
   `Subsection~\ref{subsec:res_degradation}`. These are my suggestions, not decided.
6. **Author-slot facts that do not hold as written.** These go to the author for drafting §4.1–4.6, and they are
   not number MISMATCHes:
   - l. 58: "every evaluated plan … 4.39× to 45.94×". That range covers the B rows only; the C rows of the same
     table are at 3.94×, 8.09× and 7.00×, so the table's range is 3.94–45.94×.
   - l. 63: "Two-node plans lose more than their parts would" is **not resolved** by the records. The 2025 pair
     loses 102.1 against parts of 49.7 + 52.5 = 102.2 (pair 0.1 lower). The 2030 pair loses 93.9 against parts of
     43.3 + 49.0 = 92.3 (+1.6). Each term carries a threshold of 12.6–13.5 k€, so neither difference is determinate.
   - l. 65: "identical d net, since all cohorts reach y^End with the same remaining life". It is true that d net =
     d gross on every 2025 plan. The reason is that the 2025 cohort's remaining life at y^End is **zero** (C8).
   - l. 381: "Deferring the reference unit changes little gross and nothing net" contradicts the figures that follow
     it. 2035 costs 111.0 gross against 64.4 in 2025 (+46.6), and 2035 − 2030 is 42.5 gross, determinate (3.46×). Net
     is 56.7 / 54.5 against 64.4, and no frozen claim resolves 2025 against 2030 or 2035.
   - l. 484 (4.9 (i)): "with the datasheet calibration and the 0.70 floor". The baseline "reads the datasheet count"
     (main.tex l. 1066), but the table has a separate row "Datasheet 8,000 cycles" (C4), and accepted sentence 3 says
     "baseline calibration". The wording is ambiguous; the author decides.
7. **Optional fill of the 0.50-floor row (l. 262, CONFIRM l. 269).** The values are value 258.5 k€, AE 0.778,
   EFC/day 1.08, floor binds never. They come from a committed named record (W155 summary), not from the frozen
   T1–T11. The author decides whether to fill the row.

## Prediction vs outcome (expert, Addendum 73)

| prediction | outcome |
|---|---|
| zero MISMATCH in the composed tables and captions | **HELD**: 0 MISMATCH among 466 table / caption tokens (431 match, 2 approximate, 19 unchecked notation/compound/language, 14 excluded structure) |
| the salvage equation needs one correction among its four points | **FAILED as worded**: none of items (a)–(d) is false; the equation needs **two** corrections, in the definition of I (energy cost only, adjacent to (c)) and in the SoH factor (normalized above SoH^Min, adjacent to (b)) |
| the four other structural CONFIRMs (S/E, I/J keys, F2 plan, 16 cycles) hold | **FAILED (3 of 4)**: S/E confirmed, I/J keys confirmed, 16 cycles confirmed (wording note), F2 plan **differs** (node 7 is 1 MVA / 3.5 MWh) |

## CONFIRM points

| id | frame lines | point | outcome | evidence |
|---|---|---|---|---|
| C1 | 28–45 (comment 43–45) | eq:salvage_credit, eq:salvage_remaining_life, items (a)–(d) against `_get_terminal_salvage_value_*` | **differs** (2 corrections; (a)–(d) hold) | No function `_get_terminal_salvage_value_*` exists: `git grep` over tracked `*.py` at HEAD finds only this script's own lines. The reported net uses `_build_terminal_salvage_value_expression` (`shared_energy_storage_data.py` l. 866; `model.salvage_value` l. 849), summed by `get_salvage_value` (l. 145–148); `shared_resources_planning.py` l. 1305–1306: `net_operational_recourse = gross_operational_cost - terminal_salvage_value`. Other variants: `_get_salvage_value_results` (l. 974; report/Excel, same product, gated by an active cohort with remaining life > tolerance) and `_get_salvage_value_sensitivities` (l. 932; dC/dE for cuts). None of the other variants feeds the net. (a) `terminal_date = years[0] + Σ blocks` = 2040 = 2035 + 5 (l. 979); `age_at_terminal` = blocks from y^Inv on (l. 919). (b) E_avail = E_rated · `es_soh_per_unit_cumul[y_inv, terminal]` (l. 628), the end of the last block, after its 5 years of fade (l. 724); the mid-block variant re-reads the end-of-block SoH (l. 889). It enters as (E_avail − SoH^Min E)/(1 − SoH^Min) (l. 890–893). (c) `_get_expected_energy_investment_cost` (l. 925–928): Σ_s prob_s · cost_investment['energy'][s][y], undiscounted at y^Inv; **energy only**. (d) `_get_terminal_discount_factor` (l. 910–912): 1/(1+r)^{Σ blocks} = 1/1.02^15 = 0.743015. Remaining life (l. 920–921): max(T_cal − age, 0)/T_cal, i.e. eq:salvage_remaining_life. Parameters: `SRP1_ESS_Params.json` (`39106f93`) energy_recovery_fraction 1.0, recycling_floor_fraction 0.0, calendar_life_years 15, minimum_soh 0.70; `SRP1.json` Years {2025: 5, 2030: 5, 2035: 5}, r 0.02. Reproduction: yl_y2030 has recorded terminal SoH 0.768414 (`…r2_yl_y2030/…/boyd_terminal.json` `3e9ab33b`), c̄^E_2030 = 208,198.72 €/MWh (workbook `e17bd588`, probabilities 0.35/0.55/0.10), RLF 1/3, giving 11,759.26 € = recorded `year_ladder.per_year.2030.salvage_last`. yl_y2035 has SoH 0.877581 (`87bd2612`), c̄^E 192,708.12, RLF 2/3, giving 56,504.17 € = recorded. |
| C2 | 83–98 (103–105) | S/E/duration of each B and C cell | **confirmed** | 16 of 16 rows: node(s), S, E, E/S (= the duration) and year equal the frozen candidate of the claim's cell. Each cell's own `evaluation_record.json` (tracked, found by eval key: W137 v4, W139 v5, W142 v6, W142 ext v6, W101) has the same candidate_canonical, candidate_key and eval_key. The S column is E/duration on every row. |
| C3 | 208–209 | I/J cells = node-7 4 h plans 2–5 MWh (keys cc4eb5df, fb21d822, 11725e45, cab98853) | **confirmed** | i_5a6a88b4 cc4eb5df = d_c52e1670 [0.5, 2]; j_f3aa335e fb21d822 = d_4a82a64a [0.75, 3]; j_a11d7966 11725e45 = d_3632b0ae [1, 4]; j_5f3cccb4 cab98853 = d_36686489 [1.25, 5]. All at node 7 in 2025; evaluation records (W142 v6) agree. |
| C4 | 130–131 | the F2 plan behind ref:5ca4f86c | **differs** | Frozen `cells[ref:5ca4f86c].candidate_canonical`: 2030, n5 [0.25, 0.5], **n7 [1.0, 3.5]**. Its evaluation record (`…w118_resettle_r2_f2_incumbent/…/evaluation_record.json` `bad08f6b`, label f2_incumbent) agrees, and so does W173 (`0aa337fc`): label `y2030__n5_p0.25_e0.5__n7_p1_e3.5`, z [1, 1, 4, 7, 0, 0, 1]. Corrected text: "node 5 0.25 MVA / 0.5 MWh and node 7 1 MVA / 3.5 MWh". |
| C5 | 262, 269 | value, AE/EFC of e_soh050 | **confirmed (values available)** | value 258,456.29 € (258.5 k€) = value − I (−59,500.72) + I (317,957.01) from the frozen A64 row. This equals Q(0) − Q(e_soh050) from the W155 summary (`3c71d71d`). AE 0.7781 → 0.778 and EFC/day 1.0763 → 1.08, PV-weighted, from `reports.e_soh050.AE / EFC`; the code uses the same `M46A.pv` definition as T8. minimum_soh 0.5; floor never binds (\|dual\| ≤ 5.1e-10). Not in T1–T11. |
| C6 | 342 | the 3 × 3 "16 cycles" | **confirmed** (wording note) | W98 r2 replays bitwise through N = 72, the production exit (`b0d58764`). The continuation record covers cycles 1–88 (`d879551e`), so the evaluation ran 16 cycles past the exit (73–88), stopped by the frozen \|dQ\| < 500 € × 3 rule. The W99 damped-cosine fit window is [72, 88] (17 points), L = −6,234.94 € = D (T11 6.2). Wording: the fit was not "continued"; the evaluation was. Suggested: "from a damped-cosine fit to the evaluation continued 16 cycles past the exit". |
| C7 | 390–392 | W153 "0 sign changes in 60 claims"; one verdict changes category | **confirmed** | W153 (`215fd21d`): n_claims 60, n_sign_differs 0, n_verdict_differs 1 = L:y2030__n5_p0.25_e0.5__n7_p1_e3, within the uncertified bar gross → determinate net. Frozen T1: 60 claims, 0 sign changes; that claim is at 0.9472× gross and 1.2229× net (0.95 / 1.22). |
| C8 | 403, 410–411 | the 2025 cohort's zero salvage | **confirmed** | Code: age_at_terminal(2025) = 15, so T^Rem = max(15 − 15, 0) = 0 and the 2025 term is exactly 0. Records: 30 cells with a 2025 plan have max \|salvage\| 5.8e-33 €, numerical residue of the zero-investment cohorts' terms; ref:bd504ecf salvage 0.0. T1: d_net = d_gross on every claim whose cells are all 2025 plans. The row may print 0.0. |
| C9 | 452 | 41 new recourse evaluations (14, 20, 7; budgets 20, 20, 60) | **confirmed** | W173 item1: s47 14/20 (mesh local optimum, unit poll failed), s51 20/20 (completion cap stop), s53 7/60 (poll failure at unit mesh); sum 41. |

## Number check (every token of the frame)

1,057 tokens: **match 888, MISMATCH 2, approximate 14**, unchecked 64 (notation 20, language 12, compound adjective
15, instruction 8, cross-reference 9), excluded 89 (identifiers 75, layout and figure options 14).

| part | match | MISMATCH | approximate | unchecked | excluded |
|---|---|---|---|---|---|
| tables (tabular) | 305 | 0 | 0 | 7 | 0 |
| table captions | 59 | 0 | 0 | 8 | 9 |
| figure captions | 67 | 0 | 2 | 4 | 5 |
| expert prose (preamble, 4.7, 4.8, 4.9) | 107 | 0 | 8 | 11 | 2 |
| equations | 0 | 0 | 0 | 2 | 0 |
| comments ([AUTHOR] slots, source notes, header) | 350 | 2 | 4 | 32 | 73 |

Sources: frozen JSON `590088fe` at the export rounding of `export/README.md` (k€ 1 dp; multiples 2 dp; €/MWh nearest
10; % 1 dp; EFC 2 dp; AE/SoH/R 3 dp; R predicted 4 dp; GWh 1 dp; p.u. 3 s.f.). Other figures were checked against
named records with path and sha256: W153, W155, W173, W98/W99, W165 figure data, the cost workbook, the case files,
`manuscript_submitted/main.tex` (the 18.25 / 92.16 % submitted-version figures, l. 140 / 1122), or a committed W163
check, reused for the §4.7–4.8 environment, tolerance and offset figures (55 tokens). Plan descriptors (node, S, E,
duration, year; 192 tokens) were checked against each cell's frozen candidate.

**MISMATCH (2, both in comments):**
- l. 3 (header) "1448": main.tex at `8b76423` has `\section{Results}` at 1126 and the CONCLUSIONS banner at 1446–1448,
  so the line before the banner is **1445**, as Addendum 73 says. The header's own rule gives 1126–1445.
- l. 131 (4.2 slot) "3" (node-7 MWh of the F2 plan): the record says **3.5** (C4).

**approximate (14): printed coarser than the export rule, consistent at its own precision:**
- l. 281 (slot) and l. 305 (fig:res_benchmark caption) "1,264": `benchmark.multiple` 1,263.75 (multiples 2 dp; the
  W165 captions.md also writes 1,264).
- l. 284 (slot) and l. 305 (caption) "0.81" TWh: price-taker best-start TN curtailment 807.6 GWh (export GWh 1 dp).
- l. 349 (slot) "7", "9" %: 1 − R = 6.6 % / 9.1 % (T11 0.934 / 0.909).
- l. 465 (4.8) "55", "31", "14" %: T9 node shares of the five dead-zone cells, 55.2–55.3 / 31.0–31.1 / 13.7 (export
  1 dp). "0.69": T9 pf_primal last 0.682–0.700 (a range summary).
- l. 485 (4.9) "171", "193", "254" k€/MWh: T3 171.31–192.56 and energy cost 253.88 (nearest 10 €/MWh = 2 dp).
- l. 487 (4.9) "654" M€: coordinated Q 653,873.7 k€.

**Text checks (42; 41 OK):** the verdict words and the u/f superscripts of every row of the value-ladder, flexibility,
energy-ladder, ageing, year-ladder, discount and multi-scenario tables match the frozen claims (34 rows). The four
accepted sentences are verbatim against `paragraphs_v5.md` (`b8665706`). The T4 caption sentence (l. 383–384) is
verbatim. Every claim or row named in the source comments exists (24 claim ids; T8 arms; T6 and T3 row names). The
one item not OK is Addendum 64 ruling 4 for the J 4 MWh row (Blocked item 3).

## Labels, figures, references

- **Labels:** 26 introduced, no duplicates. The set equals the header list. None collides with a main.tex label
  outside the replaced range (l. 1126–1445). Every `\ref`/`\eqref` of the frame (body and comments) resolves to a
  frame label or to a main.tex label outside the range. Outside the range, main.tex references only one old §4
  label, `sec:results` (l. 308, 481, 2031), which the frame keeps. `subsubsec:certification` is 2.2.7 at `8b76423`,
  as the l. 8 comment says.
- **Figures:** the five `\includegraphics` files (`fig1_breakeven.pdf`, `fig2_flexibility_ladder.pdf`,
  `fig3_ageing_arms.pdf`, `fig4_q_versus_cycle.pdf`, `fig5_benchmark.pdf`) are exactly `export/figures/*.pdf`. Each
  sha256 matches the captions.md prefix and the W165 manifest (99ec6a6b, 7d15f697, 8f243807, 45755c82, 3a0184db).
  None exists in the clone yet.
- **Literal references in main.tex at the clone HEAD `8b764234` (outside l. 1126–1445):**

  | line | text | on Addendum 73's list |
  |---|---|---|
  | 377 | (Section~4) | **no** (salvage sentence of §2.1) |
  | 415 | Section~4 | yes |
  | 445 | Section~4.7 | yes |
  | 631 | Section~4.5 | yes |
  | 635 | (Section~4.7) | **no** (the 0.9τ sentence) |
  | 671 | Section~4 | **no** (the mid-block sensitivity) |
  | 1066 | Section~4.3 | yes |
  | 1119 | (Section~4.4) | listed as l. **1116**: not at that line |

  The letter (`response_to_reviewers_draft.tex`) also carries literal "Section~4.x" references (e.g. l. 49–53, 82–83,
  95). They are outside this order's scope.

## Found (beyond the CONFIRMs)

- The frame's choices (1)–(8) from Addendum 73 were checked: (1) "reference unit" is consistent with the lattice
  (0.25 MVA / 0.5 MWh is admissible, and the C row at 49.7 k€ is evaluated). (2) S = E/duration holds (C2). (3) The
  I/J keys hold (C3). (4) The F2 plan **differs** (C4). (5) The best-start curtailments match (807.6 / 0.6 and 781.7 /
  201.2 GWh). (6) 16 cycles holds (C6). (7) 41 holds (C9). (8) 0 sign changes in 60 holds (C7).
- Every [AUTHOR] slot figure is a frozen cell or a named record at its printed rounding, except the two MISMATCHes
  and the interpretive statements in Blocked item 6.

## Not confirmed

- **Compilation:** not attempted; there is no TeX installation on this machine (export README). The frame's
  "compiles" statement is unverified here.
- **Figure content:** I checked filenames and sha256 only. The PDFs' visual content against the adapted captions was
  not inspected. Caption numbers were checked, but the caption text was not diffed against captions.md beyond the
  "smallest unit" → "reference unit" change.
- **W163-reused figures** (macOS 27.0, M2 Max, 32 GiB, thread caps, tolerances 10⁻⁶ / 10⁻⁴ / 5×10⁻⁴, 10⁻⁵ slack,
  7.8 k€, 1.2×10⁻⁵, 62 MWh, 3τ, L = 44, 15–21 k€, 0.9τ, 46 certificates): taken from W163's committed check results
  (`34f53a2c`, status match). This script did not re-derive them.
- **Salvage reproduction** was done on two cells (yl_y2030, yl_y2035). The 2030/2035 G, L and C cells were not
  reproduced; their terminal SoH was not extracted.
- **e_soh050 AE / EFC definition:** confirmed equal to T8's by code inspection (same `M46A.pv` calls in the W155 and
  W142-ext harnesses). Not recomputed from the ageing series.
- **"no `_get_terminal_salvage_value_*`":** scoped to tracked `*.py` at HEAD (`git grep`). Untracked files,
  `.claude/worktrees/` and other branches were not searched.
- **Qualitative statements** in 4.7–4.9 (for example "each stopped by the production exit", "a known property of the
  method", "left curtailed") were not audited. That is the statement audit of W177b.
