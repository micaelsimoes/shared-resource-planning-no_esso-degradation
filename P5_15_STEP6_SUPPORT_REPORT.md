# P5.15 Step 6 support — W164–W167 (the author's zero-solve request of 2026-10-07)

**Authority:** the author's request of 2026-10-07 and `STEP6_REVISION_MAP.md` §§D–E. Addendum 67 (paragraphs_v5 accepted)
is committed at `d60172a7`. Manuscript: Overleaf clone `manuscript/6a67305f25e8348fb71380c3/`, pulled at Overleaf HEAD
`6191c6c` (already up to date; nothing new on Overleaf). `main.tex` there is still the **submitted** text: sha256
`3cedbb6e37bf…`, 2,101 lines. The order gave `3cedbb6e39c8…`; that was the Planner's typo, and every script pins the real
hash. The response letter is `response_to_reviewers_draft.tex` (`8df94963…`).
**Objective convention:** Q = `gross_operational_cost`, settlement excluded; value = Q(0) − Q(x).
**Runs:** zero solves in every task (guards armed and verified at 0, pickle blocked). Nothing else ran.

## Decisions needed

1. **Letter R3.3 — the storage schedule (question 5; expert, wording).** The code has one storage schedule per block,
   shared by **all nine market × operation scenarios** in the ESSO, TSO and DSO models.
   - The non-anticipativity is **hard**: one aliased variable, no penalty (`model_construction_helpers.py:811–852`,
     Addenda 37(iv)/38(B)).
   - Deviations are taken on the network side: TN dispatch and each DSO's substation import, per scenario.
   - The priced term (row 18, α = 0.5) is a linear premium on each **DSO's** interface P/Q deviation from its expected
     import (`:1861–1998`). It does not touch the storage. The TSO interface is scenario-free.
   - The daily closure is **soft**: within 5 % of available energy plus 10⁻⁵, with the slack penalised at 10³ €/MWh
     inside Q.

   The letter (l. 283–290) is right that the schedule is common, so its CONFIRM comment can be removed. Three clauses
   need fixing: "common to the operation scenarios" → common to all market × operation scenarios; "deviations priced
   by the non-anticipativity term" → the storage is non-anticipative by construction, and the priced term is the
   DSO-side interface premium; "the day ends at its initial state" → the soft closure.
   **Options:** (a) adopt W166's plain paragraph (`P5_15_W166_EQUATION_CODE_AUDIT.md`, "Answer to question (5)");
   (b) keep the current wording. **Recommendation: (a).** It costs one paragraph of wording.

2. **§3 states two horizons, not one (author/expert).** The 3 × 3 instance ran on **five years — 2025, 2028, 2031,
   2034, 2037 — each a 3-year block** (`SRP1__s53_3x3.json` "Years"; spec `231558f0`; `p515_s53_w89_3x3_campaign.py:10–13`).
   SRP1 ran on three years, 2025/2030/2035, each a 5-year block (`SRP1.json`). W166 and W167 found this independently,
   and the Planner confirmed it in both case files.
   - Map §B §3 and letter l. 97 ("three representative years standing for five-year blocks") hold for SRP1 only.
   - **Options:** (a) state both horizons in §3 and in the T11 caption, and scope "five-year blocks" to SRP1 in the
     letter. No cost. (b) Re-run the 3 × 3 on SRP1's horizon: multi-day, not in any order, and the author's machine
     time. **Recommendation: (a).**

3. **§3.1 investment costs (question 4; author).** The map's premise is wrong. `SRP1_ESS.xlsx` at `7ce1d1ab` (identical
   to HEAD) carries **three cost trajectories weighted 0.35 / 0.55 / 0.10** (NREL ATB low/mid/high; sheet `Scenarios`).
   Production reads all three, and I(x) is the probability-weighted sum (`shared_energy_storage_data.py:1743–1747`;
   `p56a_oracle.investment_cost` :253–270). Since I(x) is linear in the costs, this equals using the expected trajectory.
   - The map's ≈254 k€/MWh and ≈256 k€/MVA are the **2025 expectations**: 253,877.68 and 256,317.32. The Planner
     recomputed 253,877.68 from the sheet.
   - What changed since submission is the **energy row only**, ×1.25: the old file `072b1310` divided the 4-h $/kW cost
     by 5 instead of 4. For 2025 the printed values are 169.71 / 211.99 / 271.13 and the current file gives
     212.13 / 264.98 / 338.91 k€/MWh. The power-cost rows and the probabilities agree.
   - **Recommendation:** keep §3.1's three-trajectory description and replace its table with W167's 2025/2030/2035
     fragment. State in §3.1 that I(x) is the expectation. The letter says what the correction was (÷4 h, not ÷5).

4. **Figure 4 panel (a) — the x = 0 reference's certificate (expert).** `ref:7aa017f0` (k* 181) anchors every T1 B row.
   It was certified by **settling-rule version 1** in the W101 continuation; paragraphs_v5 §(ii) describes version 6.
   - Its certifying regime is held from cycle 133. Cycles 123–132 were a bitwise replay of the original run up to its
     residual stop at N = 132. The paragraph says the regime holds "from k₀ onwards".
   - Panel (b), `h_f9eae48f` under v6, holds from k₀ + 1.
   - Nobody has checked whether the reference also certifies at 181 under v6. A search for `7aa017f0` and `W101` in the
     brief and in REVISION_CONTEXT.md found no ruling on it; the frozen tables carry it as certified at k* 181.
   - **Options:** (a) keep the panel; its caption already states both facts. (b) Run a zero-solve v6 replay of the
     settling decision on its committed per-cycle record (minutes), then keep the panel or replace it with a
     v6-certified cell chosen by a stated rule. (c) Replace the panel without the replay.
   - **Recommendation: (b).** The replay is cheap, and it answers a question a referee could ask of the headline anchor.
     It is not run here because the order said nothing else runs.

5. **Letter wording points from the W164 check (expert).** Of 86 checked numbers, 86 match. There is one MISMATCH and
   seven findings:
   - **l. 256** says "the one result that depends on the convention". Two verdicts change from gross to net: T4
     2035 − 2030, and T1 `L:y2030__n5_p0.25_e0.5__n7_p1_e3`. The latter is the "one verdict" the preceding sentence
     already counts, so the year ladder is the second. Suggested wording: "the other comparison that depends on the
     convention".
   - **l. 254–256:** "a Phase B neighbour" is ambiguous. T1 uses "Phase B certificate" for its C rows, which do not
     change verdict.
   - **l. 327–328:** "terminal available-energy fraction" — T8's AE is PV-weighted, and T8 has no terminal-AE column.
   - **l. 212–216:** "three references were added" conflicts with "not yet added" for two of the three.
   - **l. 320–322:** "harsher calibrations −73.7 and −45.2" — C4 (−45.2) loses less than the baseline (−64.4), and
     C3_midblock (−65.9, determinate) is not mentioned.
   - **l. 158–160:** 48 blocks counts the network subproblems only, not the ESSO.
   - **R1.5:** "Figure 2" — in this main.tex the framework is figure 1.
   - **The title says "revision 1"** while l. 100 refers to "our first reply".

   "Single machine" (l. 360) has no record field. It stands on the author's attestation (Addendum 67, settled).
   **Recommendation:** fix each in the letter's next round.

6. **main.tex l. 1058 (a red note) says minimum SoH values of 60 % and 80 % "are evaluated".** No such run exists; the
   frozen tables carry only 0.70 and 0.50, and the map has no line for this sentence. **Recommendation:** remove it in
   the §3.4 rewrite (author).

7. **Figure 1, extra panel (author's layout).** W165 added (b), the 2 h points of the same fits, beside the requested
   4 h panel, because both fits use all ten points. **Recommendation:** keep it.

## Blocked on the author

- **The reviewers' original comment documents.** The 62 quoted tokens in the letter can only be checked against those
  documents (map §C asks for them in the repository).
- **The submitted manuscript source** for `latexdiff` (map §E: from Overleaf History or the submission zip, stored as
  `manuscript_submitted/`).
- **The manuscript clone is not tracked by this repository.** Every output pins the Overleaf commit and the sha256 of
  each `.tex` instead.

## Changed

| task | script | outputs / report | content |
|---|---|---|---|
| state | — | `d60172a7` | Addendum 67 committed; `STEP6_REVISION_MAP.md`; TASKS.md Step 6 order |
| W164 | `77fcf136` `p515_s53_w164_manuscript_number_check.py` | `b005ac7f` `data/SRP1/Results/P515S53/w164_manuscript_check/overleaf_6191c6c/` | number check of `manuscript/*.tex` |
| W165 | `bcd0e5d9` `p515_s53_w165_step6_figures.py` | `3d0dadbc` `…/w160_step6_frozen/export/figures/` | figures 1–5, PDF + data JSON + `captions.md` |
| W166 | — (static reading) | `e7b9a938` `P5_15_W166_EQUATION_CODE_AUDIT.md` | equation-vs-code audit; question (5) |
| W167 | `4523cd3d` `p515_s53_w167_nomenclature_years.py` | `5cf26940` `data/SRP1/Results/P515S53/w167_nomenclature_years/`, `P5_15_W167_NOMENCLATURE_YEARS_COSTS_REPORT.md` | nomenclature audit; year tables; question (4) |

No production file, spec, case file or frozen artifact changed: the diff `d60172a7..b005ac7f` adds files only. The
frozen JSON `590088fe` is unchanged. Each run was attached, captured stdout and stderr, exited 0, and passed the W100
typing test.

## Found

**Task (1) — number checker (W164).** The checker takes the clone, the declared Overleaf commit and every `.tex`
sha256, and refuses to run on any difference or on an existing output directory.

| file | in scope | match | MISMATCH | no counterpart | submitted-version | unchecked |
|---|---:|---:|---:|---:|---:|---:|
| response letter | 289 | 86 | 1 | 1 | — | 201 |
| highlights | 0 | — | — | — | — | — |
| cover letter | 2 | 2 | 0 | 0 | — | 0 |
| main.tex | 1,838 | 105 | 0 | 2 | 610 | 1,121 |

- In main.tex, the submitted results figures are classed as "submitted-version", each citing its map line. They are
  not traced: 18.25 %, 92.16 %, 1.62 / 3.24, 0.50 % and 30,560 s are all in that class.
- The automatic value index is computed for every token but never decides a status in main.tex. Its only candidates
  are noise: node numbers 5, 7 and 9 match 14–51 JSON paths each. The Planner accepts this precedence.
- **Maintenance cost per round:** the declared checks are located by text fragment, and the main.tex line rules are
  pinned to `3cedbb6e`. Each revised round therefore needs a new declarations version, and a stale declaration exits 3
  rather than passing silently.

**Task (2) — figures (W165).**

| figure | width | PDF sha256 | sources |
|---|---|---|---|
| fig1_breakeven | 190 mm | `99ec6a6b` | T3 conservative_A64; intercept from `W145.banded_breakeven_fit` (reproduces every T3 coefficient bitwise) |
| fig2_flexibility_ladder | 90 mm | `7d15f697` | T1 CHECK / H rows, T10 |
| fig3_ageing_arms | 90 mm | `8f243807` | T8, thresholds from T1 E |
| fig4_q_versus_cycle | 190 mm | `45755c82` | (a) `ref:7aa017f0` W101 record; (b, c) `h_f9eae48f` W142 v6 record |
| fig5_benchmark | 190 mm | `3a0184db` | T6; split from W130 record |

- **Determinism:** each figure was rendered twice and the PDFs are byte-identical. Fonts are embedded TrueType, and the
  PDFs carry no dates.
- **Fig 4:** 41 checks of the trajectory against T2, all exact.
- **Fig 4 panel 2:** the cell was chosen by a rule fixed before any trajectory was read — the first T2 row whose cause
  is the gap clause.
- **Fig 5:** the 77.2 % / 22.8 % split (TSO 70,166.6 k€; DSOs 6,945.5 / 7,605.2 / 6,179.4 k€) is not in T6. It comes
  from the committed record W130 (`2c1b731a`), cross-checked against T6 to 10⁻⁶ €.
- **Arial dependency:** the figures use Arial, a macOS system font, so the script needs Arial installed. The font
  file's hash is recorded.
- The Planner viewed all five previews; each shows what map §D asks.

**Task (3) — audits.**

- **Equation-vs-code (W166).** 68 rows: 22 match, 38 differ, 8 not implemented; 36 are high impact (a referee
  re-deriving from the print would get a different model).
  - **Retired:**
    - eqs (13)–(14), (25), (26), (28) and the slacks (29)–(30);
    - (28) is replaced by the converter circle P² + Q² ≤ S²_rated.
  - **Changed in form:**
    - eq (21): k = N·DoD/(−ln R) = 35,851;
    - eqs (22), (27): active cell-side throughput η_ch·p_ch·Δt + p_dch·Δt/η_dch, with η = 0.97 / 0.96;
    - eq (23): SoH_y = SoH_{y−1}·exp(−D)·φ_cal^Y, with φ_cal = 0.985;
    - ESSO objective: 10³ × the P-net slack pair + ε Σ(p_ch + p_dch), with ε = 10⁻⁵; salvage is built but not
      optimised;
    - l. 733 "instantiated per scenario": all scenarios share one model per block.
  - **Implemented but not stated in the text or the map:**
    - the ESSO has no SoC limits; SoC, the 10–90 % limits and the closure live in the network models;
    - those models use the **degraded available capacities**, which the ESSO sends them every cycle. This is how
      degradation reaches the SoC limits.
  - **Appendix A:**
    - the algorithm checks convergence once per cycle, after the ESSO;
    - every agent's target is the consensus z;
    - normalisation is a fixed 2·S_ref = 5 MVA, not S^Rated;
    - there is one dual per agent;
    - ρ follows residual balancing (×/÷1.5, clamp [10⁻⁴, 10⁴]) with the storage two-phase schedule and the freeze;
    - there are no separate P/Q update rates.
  - **Map §B claims borne out:** AA type-II with memory 5 and safeguards, the tight tail, S_ref, σ = 93,635,360, and
    the initial ρ values.
  - **Map §B claims only partly borne out:** "ESSO objective ε only" (it also carries the slack pair) and "red note →
    row-18 description" (row 18 does not touch the storage).
  - The production and case files are unchanged between HEAD and all 46 campaign commits behind the frozen tables.
- **Nomenclature (W167).**
  - (i) symbols defined but unused: 0.
  - (ii) used but undefined: 89 symbols, plus 4 hatted variants and 12 index letters.
  - (iii) 3 duplicate keys, 1 shared description, 5 conflicts (S^Rated / E^Rated: per-unit in the nomenclature, total
    in the body; E^Inv_{e,y,c} carries a scenario index; φ has no year index; SoH^min has two indexings), and 12
    overloads.
  - (iv) 18 Benders symbols (L^B l. 202, α^Down l. 216, the (l) superscripts), and 25 symbols that occur only in
    Algorithm 1.
  - (v) of the 19 revision symbols, 16 are used in paragraphs_v5. The lattice sizes (0.25 MVA / 0.5 MWh) and the 2–4 h
    bounds appear in neither text: main.tex still says 2–10 h (l. 629). Six paragraphs_v5 symbols clash with main.tex:
    W, V, δR, t_sum, π_t and L.
- **Year tables (W167).**
  - Five input-data tables were regenerated for 2025/2030/2035 through the production readers, as LaTeX fragments
    (`year_tables/`).
  - The cross-check of the 2025 column compared 126 cells. 15 disagree, all of them energy-cost cells, and every
    printed value equals the old file `072b1310` (Decision 3). RES, power-cost and probability cells all agree.
  - The 8 year-indexed results tables are old-method outputs; the map deletes or replaces them, so they were listed,
    not regenerated.
  - 15 year-indexed figures are 2025-only. Production draws a separate profile per year (the seed includes the year),
    so the l. 1642 sentence misdescribes them. Adding 2030/2035 versions is the author's call.

**Task (4).** See Decision 3: three trajectories at 0.35 / 0.55 / 0.10; I(x) is the expectation. I(unit) = 317,957.0085 €
reproduces the committed records. Among the 347 committed campaign and frozen specs, none has a scenario-selection
field; the 31 that pin the cost file all pin `e17bd588`.

**Task (5).** See Decision 1: one common schedule; hard non-anticipativity; deviations on the network side. The run
records show row 18 wired on all 60 DSO blocks of the 3 × 3 instance.

**Predictions.** None were recorded for this order: it is zero-solve reading and export.

## Not confirmed

- **Letter checks:**
  - The reviewers' quotations were not checked; their documents are not in the repository.
  - Section, table and figure numbers were not checked; the final numbering is pending.
- **main.tex values never searched:** demand growth 1.25 / 3.00 %, market growth 2.50 / 2.00 %, and the Appendix D data
  values (W164).
- **Question (5):**
  - The identity of the per-scenario storage schedules was established from the code (aliasing), not checked
    numerically in the 3 × 3 output workbooks.
  - No `.nl` capture of a 3 × 3 block exists to check against.
  - The 3 × 3 scenario probabilities and the interface transformer ratings were not read (W166).
- **The x = 0 reference under v6:** not evaluated (Decision 4).
- **"No selection field":** the search covered the committed `campaign_spec*` and `frozen_*` JSON under `data/SRP1/Results`;
  launcher scripts were not searched (W167).
- **Figure PDFs:** not opened in a TeX build — there is no TeX installation on this machine. They were checked by
  parsing the PDF bytes and viewing the PNG previews.
- **Printed equation numbers:** computed from the `.tex` source, not read from a compiled PDF (W166).
