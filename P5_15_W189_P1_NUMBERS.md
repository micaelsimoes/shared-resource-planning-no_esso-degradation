# P5.15 W189 part 1: round-9b audit at Overleaf 46c9d77, numbers and claims (Addenda 89 and 90)

**Worker, 2026-11-10.** This was a static reading. There were zero solves, zero model builds and zero pickle loads.
- The clone, `manuscript_review/`, code, records, `TASKS.md`, `REVISION_CONTEXT.md` and the brief were read and not edited.
- Nothing was compiled, because there is no TeX on this machine.
- W189 part 2 (pointers, labels, structure) and W189a (supplement v2) are other tasks. Letter pointer numbers are therefore listed as NOT RATED here.

**Objective convention.** No objective value is computed. Every rated money figure carries its own convention field in the JSON. Unless a row says otherwise, that convention is Q = `gross_operational_cost` with the contracted interface settlement excluded (the frozen primary).

## Blocked on Planner

None. No stop condition was met, and no result depends on a ruling. Four points go to the Planner/expert for wording (details under Found):

1. **The G10 parenthetical calls the agent's net-power pair a penalised slack "in Q". W188 (a) says it is not in Q.**
   - Appendix B.4, l. 2310, reads: "every penalised slack in $Q$ (daily closure, network feasibility and the agent's net-power pair)".
   - W188 (a)'s code reading: the ESSO σ± pair is penalised only in the ESSO `feasibility_penalty`. That penalty is never summed into Q ("In Q: no"; "σ± has 0 € in Q by construction").
   - The pair does sit at its lower bound on all 25 rows (W188 (a)), so the factual claim "at its zero bound" holds. Only the "in Q" classification is inaccurate.
   - Ruling 3 quotes this sentence with an ellipsis exactly where the parenthetical sits.
2. **Addendum 89 ruling 7 says verify_appE.py is extended to the reduced table. It was not.**
   - `verify_appE.py` is unchanged since `c831ea46`, and `gen_appE.py` since `22e19b56`. Neither was extended.
   - The reduced Appendix (now B) is not the generator's output. It is the generator's output with two subsections deleted and five prose lines edited by hand (Found 1b).
   - The comparisons table was kept in full (58 rows). It was not limited to the rows the main text cites.
   - All printed entries still equal the frozen JSON (0 MISMATCH), so this is a process / ruling-text point, not a number fault.
3. **Figure 6 paragraph: "the unit cycles once or twice a day between the 10 % and 90 % limits" holds for 6 of the 8 days.**
   - On the two Winter days the SoC does not reach the 10 % limit. The minimum is 0.1222 MWh in 2025 (13.9 % of E^Av) and 0.1078 MWh in 2035 (15.4 %).
   - Cycles per day, measured as charged energy over the usable band, range from 1.03 to 1.95 on every day. So "once or twice a day" holds.
4. **The B.4 figures 171 € and −13.0 k€ are exact only at the precision noted.**
   - **171 €** is the row total for superseded evaluation 16: voltage 169.27 € + closure 1.42 € = 170.69 €. The sentence calls it "voltage-slack content". The voltage part alone is ≤ 169.27 €. The bound as printed is true, and it is the figure ruling 3 itself uses.
   - **−13.0 k€**: one of the 20 certified rows, `c_156ce2d1` at −13,052.36 €, reads −13.1 k€ at 1 dp. The sentence's "to within 40 €" covers it (spread 39.80 €).
   - Also: "carries a positive flexibility-balance content of at most 2.2 €" (3 × 3 unit) is a bound. W188 (a) did not show the content to be positive.

## Changed

- **`1eac8803`**: the harness `p515_s55_w189_p1_numbers.py` (sha256 `7ab5009c…`), a new file. It was committed before the run, and the run records it as `committed_clean: true`.
  - It imports W183h `p515_s53_w183h_round8.py` unedited, and through it W182c, W182b, W181a, W177a, W177b-1, W177b-1′ and W178b.
  - The new declarations version is W189p1, keyed by HEAD line.
- **Evidence commit** (hash in the final message): this report plus `data/SRP1/Results/P515S55/w189_p1_numbers/`.
  - The directory was fresh: it did not exist before `mkdir`, and every output was opened `'x'`.
  - The run was attached, with stdout and stderr in `launch.log`. Exit 0, 0 check failures.

| file | sha256 |
|---|---|
| `w189_p1_numbers.json` | `a48a0765b720177bbe4c01bd16575ce0bdb071f56cdf2c0e9e2b965415b18828` |
| `w189_p1_numbers.md` (generated tables) | `c28a9c626835063491559315f4f86896cdeb10df94e545d69f053618322351fe` |
| `manifest_sha256.json` (every input with sha256 and committed status; clone blobs at a87e195 / 46c9d77) | `b48841ecbeaef8da611ffbde66c23c61ad88a65d91a61a98639b7c7cce9a194d` |
| `manifest_post_run_sha256.json` | `66daaed91692338457ee7479c6f2567536884bb74cf138ba980e30d79cb1df68` |
| `launch.log` | `82646fa3bf17334ef211cc7df5eda17862f4ce4b33714aed5b43231b414a1d56` |
| `appE_generated_w189p1.tex` (gen_appE.py output) | `66bd3dd0a4d7b623232d95c420d1ba50cdb4bb0c075cfc0093af38e34b3a3d14` |
| `verify_appE_output.txt` | `7a0c47e9eee3b365075f0b46bdc6d20e7ec0ff40eab7f68c8213695861d213a7` |
| `verify_appE_tmp/tmpcxswe48e/appE_check.tex` (the verifier's own regeneration, redirected here) | = `66bd3dd0…` |
| `fig_rerun/fig_soc_soh_reference_unit.pdf` / `.png` (figure script re-run) | `2a981ce6…` / `49becd92…` |
| `fig_rerun/*.csv` (copies of the three W188 (d) CSVs, hash-checked against the W188 (d) manifest) | as W188 (d) |

- **Not committed:** `fig_rerun/mplconfig/`, the matplotlib font cache written by the re-run.
- **Repo state at the run:** HEAD `1eac8803`. The only modified tracked file was `TASKS.md` (the Planner's; not staged).
- **Trial runs:** three, into session-scratchpad directories `w189p1_trial1..3`. Trial 1 crashed in the negative control: W182b's word perturbation has no neighbour for "twenty-five". I added a numeral fallback before the commit. Trials 2 and 3 exited 0.
- **Guards:**
  - `SolveProfileGuard(permitted=())` was installed before any project import. All 58 distinct guards of the chain verified at exactly 0.
  - `pickle.load`, `pickle.loads` and `pickle.Unpickler` were blocked for the whole run; all counters 0.
  - JSON was written only through `gate_result_io`.
- **Commands** (repo root, canonical interpreter):
  ```
  mkdir data/SRP1/Results/P515S55/w189_p1_numbers && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p1_numbers.py \
      > data/SRP1/Results/P515S55/w189_p1_numbers/launch.log 2>&1          # exit 0
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p1_numbers.py --post-run   # exit 0
  ```

## Found

### (0) State re-verified: holds

- **Clone HEAD.** `46c9d779743b40a42da73c0050d8923e6af6fb55`, parent `a87e1950…`. No tracked changes; the only untracked entry is `.DS_Store`.
- **Diff.** `git diff a87e195 46c9d77` touches `main.tex` (34 hunks at default context), `response_to_reviewers_draft.tex`, `cover_letter.tex` and the new `fig_soc_soh_reference_unit.pdf`.
- **Byte identity with the review copies:**

| clone file | md5 | = review copy |
|---|---|---|
| `main.tex` | `49139cc2…` | `main_a87e195_round9b.tex` |
| letter | `ed8f5b13…` | `response_to_reviewers_v12.tex` |
| cover letter | `0b9e4c21…` | `cover_letter_v3.tex` |
| `fig_soc_soh_reference_unit.pdf` | — | `manuscript_review/fig_soc_soh_reference_unit.pdf` |

### (1) Numbers on every '+' line: 0 MISMATCH, 0 UNSOURCED

The method is W182b's: its tokenizer, spec kinds, evaluator and perturbation. The scope is every '+' line of `git diff -U0 a87e195 46c9d77` for main.tex, the letter and the cover letter.
- **Carried sentences** take their specs from the W182b (edd7a63), W183h (a87e195) or W177b-1 declaration of the same sentence. Those specs are re-evaluated here, not carried as ratings.
- **New numbers** are rated against:
  - the frozen JSON `590088fe`;
  - the A89 claims JSON (`6aa72591…`, pinned);
  - the W188 (a) JSON (`fdcb1ba1…`, pinned);
  - the W188 (d) CSVs (hash-checked against their manifest);
  - `SRP1_params.json`, `SRP1_market_data.xlsx`, the case files and `manuscript_submitted/main.tex`.

| file | '+' lines | tokens | MATCH | DERIVED | MISMATCH | UNSOURCED | NOT RATED |
|---|---:|---:|---:|---:|---:|---:|---:|
| main.tex | 74 | 264 | 208 | 16 | **0** | **0** | 40 |
| letter | 31 | 203 | 67 | 5 | **0** | **0** | 131 |
| cover letter | 2 | 5 | 3 | 1 | **0** | **0** | 1 |
| **total** | 107 | 472 | 278 | 22 | **0** | **0** | 172 |

- **NOT RATED, main.tex:** 20 identifiers (S1–S6); 7 compound adjectives; 3 notation; 3 language; 3 model structure; 3 table structure; 1 figure option.
- **NOT RATED, letter:**
  - 67 pointer or citation numbers ("Section~4.7", "Table~7", "[42]"), left to W189 part 2;
  - 18 tokens on the line-1 LaTeX comment (version history, not typeset);
  - 34 identifiers;
  - 5 model structure, 4 compound adjectives, 3 language.
- **NOT RATED, cover letter:** "round 10".
- **Negative control:** all 300 rated tokens were perturbed and **300/300 were refused**.
- **The new-number sources, by place:**

| place | numbers | source |
|---|---|---|
| §4.4 NRF sentence | 20.5 k€, 1.62, 0.02 %, 12.2 % (enabled), 12.2 % (prohibited) | A89 claims JSON |
| §4.4 NRF sentence | 90.9 M€ | frozen `benchmark.benefit` (DERIVED) |
| §4.7 penalty paragraph | eight, three, two, 1.6 / 1.5 k€, seven of eight, 0.7 k€ | A89 claims JSON part 2 |
| §4.7 penalty paragraph | 0.12 | DERIVED: the baseline ratio is 0.1205 at 1× and 0.1184 at 2×; all six arms lie in 0.110–0.132 |
| B.4 G10 sentence | −13.0 k€, −2 × 10⁻⁵ | W188 (a), DERIVED (see Blocked 4) |
| B.4 G10 sentence | 40 €, 2.2 €, 171 € | rated as upper bounds rounded up: 39.80, 2.159, 170.69 |
| B.4 G10 sentence | 16 | generator E.21 row of `pb_y2025_n5` |
| Figure 6 paragraph and caption | four days, 2025 / 2035, 10 / 90 % | `definitions.py` |
| Figure 6 paragraph and caption | 0.70 floor, 0.70 → 0.56 MWh, seven, 50 %, 2035, 0.50, 0.68 | W188 (d) CSVs |
| §4.2 | 182,700, 173,490, 190,380 | frozen `break_even.committed_W145`; these are the "flagged certificate kept" row of the removed Table 5, moved into prose |
| §3 | 2.50 / 2.00 % price growth | `SRP1_market_data.xlsx`, sheet Growth Factors; production reads this sheet |
| Appendix A (l. 1953) | residual-balancing constants 1.5 / 5 / 3 / 10^±4 / one / five / ten / 200 | `SRP1_params.json admm.penalty_update` |
| letter l. 283 | ±250 / ±100 | the submitted and a87e195 table rows; the case33_2 REF generator in all 17 year files |
| cover letter | 18.25 / 92.16 | `manuscript_submitted/main.tex` l. 905 / 911 |

- The full token table, with counterpart, value at the written precision and convention, is in `w189_p1_numbers.md`.

**Evaluation numbering.** Evaluation numbers now point at supplementary Table S4.
- W182b's e21-based specs use W181a's committed `e21_map`. That map is valid only while the E.21 rows equal those at cd57efc.
- I checked it on the generator's E.21 rows, which are what S4 is generated from, after removing the round-8 `$^{b}$` marks: equal. This adaptation is declared in the JSON.

### (1b) Reduced Appendix (now B)

- **Is it bitwise the generator output? No.**
  - `gen_appE.py` is at its committed version `22e19b56`; the expert did not change it.
  - Run here, it prints `ok 58 61` and writes 391 lines, sha `66bd3dd0…`. That output:
    - equals `manuscript_review/appendixE_evaluation_record.tex`;
    - equals the verifier's own regeneration;
    - occurs exactly once in `a87e195:main.tex`;
    - occurs **0 times** in HEAD's `main.tex`.
  - HEAD's block (l. 2104–2321, 218 lines) has 213 lines equal, in order, to the generator output. It differs in 7 hunks:

| hunk | generator lines | HEAD lines | what |
|---|---|---|---|
| replace | 17 | 2120 | opening paragraph rewritten (supplement pointers S1–S6) |
| replace | 59 | 2162 | E.20 caption "Table~\ref{tab:appE_evaluations}" → "Table~S4" |
| delete | 143–242 (100 lines) | — | subsection "Evaluations" (E.21) |
| delete | 287–360 (74 lines) | — | "Static Arrangements: Solver Starts" (E.23) and the gap-refused table (E.24) |
| replace | 379 | 2308 | "Table~\ref{tab:appE_evaluations}" → "Table~S4" |
| insert | — | 2310 | the G10 sentence |
| replace | 384 | 2314 | "Table~\ref{tab:appE_evaluations}." → "Table~S4." |

- **verify_appE.py: 33 PASS, 0 FAIL.** It is not extended to the reduced table:
  - it is unchanged since `c831ea46`;
  - it never reads main.tex;
  - it checks the full generator output.
- **Entry check of the reduced table** (W180a/W181a `table_check`, against the frozen JSON):
  - Evaluation numbers are mapped through the generator's E.21, which is the S4 numbering.
  - **657 printed entries, 0 MISMATCH, 0 non-frozen disagreements, 0 mapping problems.** E.20 has 639 entries (581 frozen-field MATCH, 58 non-frozen agree). E.22 has 18 (18 MATCH).
  - The generator's unprinted E.21 also checks: 854 entries, 0 MISMATCH.
  - Method control: the same check on the a87e195 tables gives 0 MISMATCH.
- **Rows compared.**
  - E.20 at HEAD has **58 rows, cell-identical** to a87e195 and to the generator. E.22 has 8 rows, identical to both.
  - Only `tab:appE_claims` and `tab:appE_neighbours` are printed.
- **Rows the main text cites.** `tab:appE_claims` is cited at l. 1411, 1490 and 1495 without row numbers, and at l. 1676 as "rows~17--20". Rows 17–20 are the four G claims; W182b's `appE_claims_rows_17_20` re-evaluates MATCH.
- **The table is not limited to the cited rows** (see Blocked 2).

### (2) The W184 / W185 / G10 sentences: numbers MATCH; wording holds against rulings 1–3, with one statement differing from W188 (a)

**Located once each:** §4.4 l. 1627; §4.7 l. 1728; B.4 l. 2310; letter benchmark bullet l. 279; letter R2.6 l. 170.

**Ruling 1 (§4.4 NRF sentence and the letter bullet).**

| check | held |
|---|---|
| The §4.4 sentence contains the ruling-1 text verbatim after LaTeX normalisation | yes |
| One decimal: no 12.20x, 0.0225, 20,463, 1,572 or 682.5 in any of the five sentences | yes |
| The NRF cost is in k€ (20.5 k€) | yes |
| "Determinate" matches the record: verdict determinate, sign + | yes |
| The 90.9 M€ "saving" is the enabled gap = frozen `benchmark.benefit` | yes |
| Order "12.2 % enabled, 12.2 % prohibited" = 12.205 % and 12.202 % | yes |
| The letter bullet carries 20.5 k€, 0.02 %, 90.9 M€ and 12.2 %, plus "1 of 12" and "8 of 12" | yes |

**Ruling 2 (§4.7 penalty paragraph and R2.6).**

| check | held |
|---|---|
| "All eight re-evaluations certified": 8 cells, all certified | yes |
| "Halved and doubled": ρ0 scale {0.5, 2.0} | yes |
| "Base and twice the base": prices {1×, 2×} | yes |
| "Every sign and verdict was unchanged" | yes |
| Seven of eight inside their frozen window; the eighth (`e2_x0_2x_rho0p5`) 682.59 € **above** it (0.7 k€) | yes |
| The "operating-cost difference" spread = the V spread | yes |
| The initialisation caveat is present ("covers the penalty schedule, not the start of the coordination … initialisation … the limitation below stands") | yes |
| R2.6 carries the same, with "the check covers the penalty schedule and not the initialisation" | yes |
| No mention of the monotone-branch cell | yes |

**Ruling 3 (B.4 slack sentence).**
- All four wording parts are present.
- Exception 1 (the 3 × 3 unit, ≤ 2.2 €) and exception 2 (superseded evaluation 16, ≤ 171 € of voltage-slack content) are both present.
- **Differs from W188 (a):** the parenthetical lists "the agent's net-power pair" as a penalised slack in Q (Blocked 1).

**Checks, not predictions.** P1, P2 and P5 are called "check" (§4.7 "The check covers…"). No sentence calls them predictions (item 5).

### (3) Figure 6: prose MATCH; PDF identity and data hold; re-run not bitwise (expected), plotted data identical

**Prose numbers.** All MATCH, against:
- the limits CSV: usable band 0.7924 − 0.0880 = 0.7044 → 0.70 MWh, and 0.6300 − 0.0700 = 0.5600 → 0.56 MWh;
- the SoH CSV:
  - floor rows active only in the 2035 block, for C2_calfade, C3_unit and C3_midblock;
  - none for C2 (cycling only), C4 (datasheet), no ageing and the 0.50 floor;
  - the 0.50-floor arm ends 2035 at 0.677290 → 0.68;
- `definitions.py`: 10 % / 90 %;
- the frozen plan: node 7, 0.25 MVA / 1 MWh, 2025.

**Other statement checks.** All hold except the one in Blocked 3 (Winter, 10 % limit):
- "Seven calibrations" = the 7 rows of `tab:ageing_calibrations`, the last of which is "Baseline, 0.50 floor".
- Panel (a) is the baseline reference unit: the SoC export's E^Av (row 2) equals the e_c2_calfade E^Av in 2025 and in 2035, bitwise.
- Charged energy over the usable band: 1.07, 1.03, 1.90, 1.95 (2025) and 1.03, 1.03, 1.03, 1.60 (2035).

**PDF.**
- The clone's `fig_soc_soh_reference_unit.pdf` equals the `manuscript_review` copy byte for byte (md5 `d129adda…`). It is Matplotlib 3.11.2, created 2026-11-10 14:33:42 +01:00.
- **Clone PDF against the CSVs directly:**
  - Panel (a): all 8 SoC polylines (24 points each) match `soc_tso_copy_mwh` of their (year, day), to ≤ 5.0e-7 pt.
  - Panel (b): all 7 SoH polylines match [1] plus `soh_end_of_block` of their calibration, to ≤ 4.6e-7 pt.
  - Negative control: one SoC value moved by 0.01 MWh is refused.

**Re-run of `fig_soc_soh_reference_unit.py`.**
- It is zero-solve by inspection: it imports only csv, collections and matplotlib, reads the three named CSVs, and writes two files relative to its working directory.
- It ran as a subprocess, `-I`, with its working directory inside the output directory and `MPLCONFIGDIR` inside that directory. Return code 0.
- Watched paths (repo root, `manuscript_review/`, the clone, W188 (d), each `.pdf`/`.png`) were unchanged before and after.
- **Not bitwise:** the local Matplotlib is 3.10.8 and the creation date differs. This was expected.
- **Axes-relative polylines:** 150 / 150 polylines; 147 are identical to 1e-4 pt. The other 3 are the legend handle segments, displaced by 0.0059 pt vertically. The re-run's data check also holds.

### (4) Ruling-6 disclosure: holds

Letter v12 l. 283:
- "The table of the distribution networks' transmission connection in the submitted Appendix~D printed $\pm$250~MW/MVAr for every network; the network at node~7 is connected at $\pm$100~MW/MVAr in the case file that every reported run used. The supplementary table (Section~S3) is generated from the case files and gives one row per network."

Checks:
- **Case files:** the case33_2 (node 7) REF generator is ±100 in all 17 year files (2025–2045). case33_1 and case33_3 are ±250 in all of theirs.
- **Old tables:** a87e195 and the submitted Appendix D each print one row "1 & 1 & 250.00 & -250.00 & 250.00 & -250.00 & 1.00".
- **HEAD:** the table has left the paper.
- **Supplement v1** `D_ieee33_tn_connection.csv` (zip `a94d8193…`, read in memory): node 5 ±250, node 7 ±100, node 9 ±250.
- Negative control: a copy of the letter with the line removed is refused.

### (5) "predict": 0 applied to P1 / P2 / P5

Scanned: main.tex, the letter, the cover letter and highlights.tex. There is **one** occurrence:
- letter l. 1 (a LaTeX comment, not typeset): "…first-order characterization of a certified point in R2.5 and R3.2; **prediction**-record bullet dropped)".
- It refers to the dropped prediction-record bullet, not to P1, P2 or P5.
- main.tex, the cover letter and highlights have 0 occurrences.
- Negative control: a planted "as P2 predicted" beside the penalty paragraph is found and attributed.

### Non-MATCH list

- **Numbers:** none (0 MISMATCH, 0 UNSOURCED, 0 UNRATED over 300 rated tokens).
- **Statements and process, each listed above:**
  - B.4 parenthetical "the agent's net-power pair" in Q (Blocked 1);
  - the reduced Appendix not bitwise generator output; `verify_appE.py` not extended; the comparisons table not limited to cited rows (Blocked 2);
  - Figure 6 "between the 10 % and 90 % limits" on the two Winter days (Blocked 3);
  - attribution and precision notes on 171 €, −13.0 k€ and 2.2 € (Blocked 4).

## Unexpected findings (reported, not acted on)

1. **Stale appendix header comment.** The comment above the appendix block (l. 2105–2113) still says "Replaces the submitted Appendix E … All table entries are generated … by gen_appE.py; none is typed by hand". The table entries are still the generator's, cell for cell. The prose around them now includes hand edits.
2. **Cover letter placeholder.** Cover letter v3 l. 21 keeps the bracketed placeholder "[Paragraph on the additional experiments … to be completed from their results in round 10.]". `STEP6_ROUND9_CORRECTIONS.md` says this is expected and is filled in round 10.
3. **Relative size of the offset.** "−2 × 10⁻⁵ of Q" is −1.99e-5 on the 1× rows and −1.60e-5 on the 2× rows, where Q is about 811 M€. All mantissas round to −2 at one significant figure.

## Not confirmed

- **Compilation.** There is no TeX here, so no label, page or float numbers were produced. Pointers and labels belong to W189 part 2.
- **The supplement's Table S4 numbering.** I took it to equal the generator's E.21 numbering, because the supplement is generated from the same export. I did not read the supplement's `E21_evaluations.csv`; that is W189a's scope.
- **Carried registries.** As in W182b and W183h, some carried MATCH tokens rest on committed records that were not recomputed here: the W163 registry (N4, N5, N25, N33, N45…), W182d's I(x) table and W182g's records.
- **Hourly rendering of panel (a).** It was checked only through the PDF polylines. The text labels, legend and annotations of the figure were not compared, because font subsetting differs between the Matplotlib versions.
- **"Every reported run used the case files"** (letter l. 283). Verified only as the content of the case files. I did not trace each run's inputs.
- **The moved/new classification of W182b.** It was not produced. Each carried line instead names its source declaration in the JSON `context` field.
