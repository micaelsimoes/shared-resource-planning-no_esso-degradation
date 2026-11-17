# P5.15 W182b: A2 number check of the 49 hunks, Overleaf 8f051a2 to edd7a63

Addendum 82 v2, W182 order, item A2. Zero solves: guard verified at 0 across all 47 guards in the imported chain; pickle 0; no model loaded. The clone was read only.
Objective convention: Q = gross_operational_cost, settlement excluded (the frozen primary). Net = Q − terminal salvage credit; value = Q(0) − Q(x); F = Q + I. Every rated monetary token in the JSON carries its own convention field.

## Blocked on Planner

None. There is no expert prediction for A2, so nothing was scored against one.

Two observations about context are listed under Unexpected findings (1 and 2) for the Planner to route. Neither is a number fault.

## Changed

- `b9643b84`: new harness `p515_s53_w182b_number_check.py`. It is a new file and edits nothing. It imports `p515_s53_w178b_number_check.py` unedited, which pulls in the W177b-1′ → W177b-1 → W177a → … → W160 chain. It uses a new declarations version, W182b, written for edd7a63. No earlier checker was re-run, and no earlier output directory was written to.
- Results commit (this report plus the evidence base): `data/SRP1/Results/P515S53/w182b_number_check/`. This is a fresh directory created with `mkdir`, so the harness's write-once check refused nothing. It holds:
  - `w182b_number_check.json` (sha256 943fe8c5)
  - `w182b_number_check.md` (90641669)
  - `manifest_sha256.json` (1e971c33): 129 inputs, the script, and every clone `*.tex` plus `bibliography.bib`
  - `launch.log` (99900400; stdout and stderr, run attached)
  - `w182b_bool_typing_test.json` / `.log` (VERDICT PASS)
  - `manifest_post_run_sha256.json`
- Run state:
  - Clone HEAD `edd7a63a6c2e…`; main.tex sha256 `b08a3365…` (2,956 lines).
  - Repository HEAD at run `b9643b84`. The code head at task start was `a17f6ba7`; the only change since is the harness commit. The harness was committed clean at run time.
  - Exit 0. All 19 integrity checks hold.
- Command: the MODE block of the harness docstring, `--overleaf-commit edd7a63 --expect <sha256 of all six clone *.tex>`, then the bool-typing test, then `--post-run`.

## Found

### State re-verified

`git diff 8f051a2 edd7a63 -- main.tex` has **49 hunks** at default context and **176** at `-U0`. The harness checks this as `hunk_counts_as_declared`.

There are 665 '+' lines and 424 '−' lines. The '+' line set is identical under both contexts.

The frozen JSON `590088fe` is unchanged.

### Counts

825 numeric tokens sit on the 665 '+' lines.

| A2 rating | n | checker status |
|---|---:|---|
| MATCH | 701 | match 701 (0 approximate) |
| DERIVED | 31 | derived 31 (formula in the harness) |
| MISMATCH | **0** | — |
| UNSOURCED | **0** | — |
| NOT RATED (not a quantity) | 93 | unchecked 59 + excluded 34 |

**Why tokens are NOT RATED (93):**

| category | n |
|---|---:|
| notation (indices, exponents, ≥ 0, x = 0) | 26 |
| compound adjective ("single-scenario", "two-stage") | 20 |
| LaTeX table structure | 16 |
| language ("one purpose", "two outcomes", "three phases") | 11 |
| identifier (C1–C5, M1–M3, "M2 Max") | 8 |
| length / layout | 7 |
| figure option / filename | 2 |
| `\ref` key | 1 |
| commented-out caption | 1 |
| model structure ("one shared-ESS agent per interface node") | 1 |

Every NOT RATED token is listed with its line and category in the JSON (`tokens`) and in the md.

**By section** (MATCH / DERIVED / NOT RATED):

| section | MATCH | DERIVED | NOT RATED |
|---|---:|---:|---:|
| abstract and highlights | 10 | 3 | 1 |
| §2 | 6 | 0 | 12 |
| §3 | 123 | 5 | 9 |
| §4 | 289 | 14 | 23 |
| §5 | 8 | 3 | 1 |
| Appendix A | 18 | 0 | 21 |
| Appendices B–D | 151 | 0 | 12 |
| Appendix E | 96 | 6 | 14 |

**Moved versus new.** 209 tokens sit on moved lines, meaning the identical text exists on a '−' line at 8f051a2: 180 MATCH, 4 DERIVED, 25 NOT RATED. The other 616 sit on new or changed lines: 521 MATCH, 27 DERIVED, 68 NOT RATED.

Every moved line that carries a rated token has a context note. All say the move left the number's meaning unchanged, except for the two caveats in Unexpected findings 1 and 2.

**Negative control.** All 732 rated tokens were perturbed in memory: last digit, exponent or number word, and the two BOUND tokens downward. All 732 were refused.

### Non-MATCH ratings: none MISMATCH, none UNSOURCED; 31 DERIVED

Each formula lives in the harness and is applied to the named record. The full arithmetic is in the JSON `counterpart` field and the md table.

| line | written | formula / source | convention |
|---|---|---|---|
| 146, 167, 1799 | 12.2 (%) | `benchmark.benefit` 90,896,608.40 / `benchmark.arms.price_taker.Q_best` 744,770,310.59 = **12.2046 %**. The denominator is the price-taker cost. The consistency check `benefit` = `decomposition.price_taker_NRF_minus_coordinated_eur` holds. | gross, x = 0 |
| 1584, 1799 | 90.9 (M€) | `benchmark.benefit` / 1e6 = 90.8966 | gross |
| 146, 1785, 1799 | 108 | case9 nodes 9 + 3 × 33 | n/a |
| 1310 | 80 (%) | 0.985^15 = 79.72 % (SRP1_ESS_Params) | n/a |
| 1334 | 8,000; 10,000; 80; 8,000 | The C4 entry as recorded in the e_c4 cell of the frozen ext spec v3 84775dc4 is cycles_n 10,000, reference_dod_d 0.80, eol override 0.70. N × d = 8,000. Cross-check: T8 k_C4 22,429.386 × (−ln 0.70) = 8,000.000. | n/a |
| 1425 | 24, 33 (%) | 1 − e*/e_cost from T3 conservative_A64 (24.15 %, 32.52 %) | unit cost |
| 1431 | 2 (in "2τ") | T3 intervals: (half_width − bar)/τ = 2.0 for all four intervals | n/a |
| 1636 | nine | 3 × 3 (SRP1__s53_3x3.json; instance_record derive_overrides) | n/a |
| 1640 | 230.5 (k€) | T11 rows[0] V 236,695.25 − rows[5] D 6,234.94 = 230,460.31 EUR. Consistent with rows[8] R = (V − D)/V_single. | gross, 3 × 3 |
| 1640 | 7, 9 (%) | 1 − R: 6.64 % and 9.10 % (T11 rows 7 and 8) | n/a |
| 1677 | 54.5, 64.4 (k€) | min / max of M net over 2025, 2030, 2035 (T1 d_net; T4 M_net) | net of salvage |
| 1739, 2934 | three | the count of W163 N25 `last_cycles` (continued evaluations) | n/a |
| 1754 | 60 (cycles) | fig4 data `b.band_window` [135, 194] → 60. Shaded by `p515_s53_w165_step6_figures.py` (axvspan of band_window). The clone's fig4 PDF equals the W165 export. | n/a |
| 1760 | 61.4 (h) | W153 `wall_s_total` 220,876.14 s / 3600 = 61.354 h | n/a |
| 2797 | twelve; 0.25; 0.5; one | The single-node 2025 lattice recomputed from the A1 lattice steps, SRP1_ESS_Params (2–4 h, 5 MWh, 1 M€) and the W167 2025 expected costs: 12 admissible plans; smallest 0.25/0.5; one plan one unit step from x = 0. | n/a |
| 2847 | 864 | 3 DNs × 3 years × 4 days × 24 h (SRP1.json) | n/a |

### Items the order names

- **Headline "12.2 % against the price-taker benchmark"** (abstract l. 146, highlights l. 167, conclusions l. 1799): **DERIVED**, 12.2046 %, gross Q with settlement excluded, at x = 0.
  - §4.4 (l. 1583) keeps 13.9 % and 36.9 %, now worded as "higher … than coordinated operation". This is the benchmark-relative definition `benefit_relative` = (Q_pt − Q_coord)/Q_coord, which gives **MATCH** 13.9 % and **MATCH** 36.9 %.
  - Read together, the two denominators are used as the critical review §3.5 requires.
- **`tab:res_capacity_summary`** (l. 1268–1280): all 30 cells **MATCH**. The table has 24 MW sums, the node labels 5, 7, 9 and the three header years.
  - Each MW cell is the sum of Pmax over the WIND or PV generators of the network's case file for 2025, 2030 and 2035 (MW): TN 85/130/150 and 47/75/95; ADN 5: 40/50/50 and 43/60/90; ADN 7: 80/90/90 and 20/30/50; ADN 9: 20/50/50 and 90/90/120.
  - Each node label is mapped through SRP1.json `connection_node_id`.
- **Merged §3.3/3.4** (l. 1255–1291): all rated tokens **MATCH**.
  - The year mapping (l. 1255) was checked over the four networks: every WIND and PV Pmax is equal in 2025/2028, 2030/2031, 2030/2034 and 2035/2037. The loader `network.py` reads `<name>_<year>.json`.
  - "32 Flexible Loads": every case33 file has 32 loads, each with `fl_reg` 1.
  - Growth 1.25 % / 3.00 %: the "Growth Factors" sheet of the three ADN operational workbooks. See Unexpected finding 1.
  - Generation priced at the wholesale price has no number to rate.
- **Moved text, re-rated at the new location:**
  - The three RES generator tables moved to Appendices B and C (l. 2158–2496). Every gen_id, bus and Pmax matches the case files for 2025, 2028, 2031, 2034 and 2037.
  - The lattice paragraph moved to E.3 (l. 2797/2803).
  - The §3.6 settings moved to E.6 (l. 2919–2921). These are matched to the W163 registry (N16, N19, N20), the frozen τ, the frozen re-settling specs and the W177b-3′ run settings.
  - The closure-slack sentence moved to E.6 (l. 2929). It is matched to W180d's committed record: 1.76e-6 MWh, 5.04e-6 of the energy, row 16 = pb_y2025_n5.
  - The certification-statistics paragraph moved within E.6 (l. 2934), and 56 of its 57 tokens were rated (the 57th is the compound adjective "single-scenario"). The E.21 row numbers come from W181a's committed `e21_map`, which is valid only because the E.21 data rows at edd7a63 are byte-identical to those at cd57efc (integrity check). The other values come from the W153 totals, the frozen 0.95 τ registry and W163 N25.
  - Storage settings moved to Appendix A (l. 2113–2114): CODE ratings against definitions.py, model_construction_helpers.py and shared_energy_storage_data.py, each with the lines where the constant is used.
  - The passive 1 €/MWh coefficient moved to E.4 (l. 2841): benchmark spec v5 `tie_breaker.decision.passive_dso`.

### Statement checks

These are claims that sit beside rated numbers.

- MATCH l. 1404 (`tab:res_value_ladder` note): all six rows are certified on both cells, and every cost increase is determinate.
- MATCH l. 1500 (`tab:res_energy_ladder_m2` note): all margins and increments are determinate. The flagged 4 MWh row and its increments are also determinate in the uncertified form.
- MATCH l. 1507: none of the 13 re-evaluated neighbours is determinately better.
- MATCH l. 1565: only the no-ageing margin is within resolution.
- MATCH l. 2925: "All new evaluations in the variant-A searches came from unit-neighbour completion." s51 poll 4 listed one admissible direction point, but it was not evaluated because the poll was refused at the cap.

## Unexpected findings

1. **Context, l. 1290.** The demand-growth sentence ("grow annually by 1.25 % and 3.00 %") used to sit under an ADN-only heading. It is now in the merged "Transmission and Distribution Networks" section, inside the paragraph that opens "Each ADN contains 32 Flexible Loads". The numbers match the three ADN workbooks. The TN workbook (`case9_operational_data.xlsx`) has 1.00 % and 2.50 %. Read in its paragraph, the sentence still means ADN demand; read against the section heading, it could be taken as system-wide. This is not a number fault.
2. **Pointer, l. 2919.** The moved sentence still opens "Certification (Subsection~\ref{subsubsec:certification}) uses δ_R = 0.07…". That subsection no longer defines τ = δ_R𝒱/4; the formula is now `eq:acceptance_tolerance` in Appendix A. The numbers keep their meaning. The pointer is A4's scope.
3. **Tokenizer adjustment (declared).** Two kinds of maths coefficient (4 tokens) were split before rating; the W164 tokenizer would otherwise exclude them as identifiers:
   - "3b" in max{3b, 2τ} (l. 743, 2029), rated MATCH against the W163 registry N28;
   - "2P_" in 2P_max (l. 2023), rated MATCH against N15 and N17.
4. **Removed values.** 128 distinct written values occur only on '−' lines. Examples: the × multiple columns; `tab:res_flex_ladder` (1.63, 10.8, 1.32); 13.9 in the abstract and highlights; 1,264; and the ten value-ladder rows that were dropped, including the 4 and 5 MWh out-of-budget probes now absent from the main-text value ladder. They are listed in the JSON `summary.removed_only_written_values` and are not rated, since they are no longer in main.tex. Whether the letter, `highlights.tex` or the cover letter still cite them belongs to A3, A7 and A8.
5. `TASKS.md` was already modified in the working tree before this task. It was not touched here.

## Not confirmed

- **25 MATCH tokens rest on the committed W163 check registry** (`w163_paragraphs_v4_figure_check.json`, ids N4, N5, N8, N12, N15–N20, N23–N26, N28, N29, N33, N35, N45, N59, N62, N73). That registry's status was taken as recorded; those values were not recomputed here. The registry file is hash-recorded in the manifest.
- **E.21 row numbers** come from W181a's committed `e21_map` together with the byte-identity check of the E.21 rows. `gen_appE.py` was not re-run; that is A6's scope.
- **Closure-slack magnitudes (l. 2929)** come from W180d's committed record. The IPOPT logs were not re-read.
- **Figures.** Only captions were checked. The fig4 PDF was verified identical to the W165 export. Numbers drawn inside the other figure PDFs were not checked.
- **Out of scope:** `main_latexdiff.tex`, the letter (`response_to_reviewers_draft.tex` also changed in this range), `highlights.tex`, the cover letter, and text unchanged between 8f051a2 and edd7a63.
- **Moved / new classification** compares whole lines. A number carried over on a lightly edited line counts as "new or changed", and it was rated in full either way.
- **NOT RATED tokens** (notation, structure, language) are listed but not rated. `SoH = 1` in `eq:soh_initial` and the salvage-equation constants are left to A1.

Evidence: `data/SRP1/Results/P515S53/w182b_number_check/` (JSON, md, manifests, launch log). Harness: `p515_s53_w182b_number_check.py`.
