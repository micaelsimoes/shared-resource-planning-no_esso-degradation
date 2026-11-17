# P5.15 W179 — Appendix E and the Addendum 78 changes at Overleaf `48099ab` (Addendum 78; zero-solve)

**Authority:** Addendum 78 (W179 order).

**HEAD audited: Overleaf `48099ab`** (2026-10-15 12:44 UTC).
- It is one commit after `e926fb7`.
- main.tex sha256 `28008ab0d79b477c…`, 2,717 lines. It is byte-identical to `manuscript_review/main_e926fb7_appE.tex`.
- The letter (`6ecd0ee7…`) is byte-identical to `response_to_reviewers_v6.tex`.
- At `e926fb7`, main.tex equalled `main_9fd19e7_fixed.tex` and the letter equalled v5.
- Overleaf had not moved when the tasks finished.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit.

**Runs:** zero solves in every part (guards armed and verified at 0, pickle blocked). Nothing else ran.
- W179a and W179b were cut off by a session limit and restarted from their uncommitted draft scripts. The new Workers
  reviewed and fixed the drafts, then committed them before the real runs.
- W179a's checks were mutation-tested: nine single-entry edits were each caught. W179b's had a negative control:
  altered numbers were refused.

**Part reports:**

| item | report | commits |
|---|---|---|
| (a) Appendix E vs frozen JSON; (i)–(ix) | `P5_15_W179A_APPENDIX_E.md` | `dd74d033`, `522b5d26` |
| (b) main text and letter; (c) G1–G12 and v5 | `P5_15_W179B_TEXT_LETTER.md` | `5b7ffcbc`, `8662ca72` |
| (d) closure slack; (e) two statements | `P5_15_W179C_CLOSURE_SCOPE.md` | `3994ac64` |

**Stop.** Three recorded predictions failed — the (i)–(ix) count, (c) and (d) — and Addendum 78 names no fallback.
All are wording or count predictions; no run depends on them.

## Decisions needed

1. **E.21 row 16 (`pb_y2025_n5`, the superseded certificate) — one ruling closes two findings (expert).**
   - **The finding.** Appendix E now reports the row as "cert., osc.^s", k\* 167; it is also cited in the E.22 note
     (51.3 k€, 3.79×). Two problems follow:
     - **§4.7 l. 1689.** The text says every reported certificate "was decided, or re-decided from its committed record,
       under the rule … and certifies at the cycle reported". Under v6, row 16's record is "not decided within the
       recorded cycles 1..167", with one veto (W142 `v6_from_records`; the Planner read it). Its k\* 167 is the W118 v2
       certificate. Addenda 58/59 excluded it, and Addendum 69 ruling 6 said it lay "outside the tables".
     - **§2.3.3 l. 968.** The text says the closure slacks were zero within tolerance "at every certified evaluation
       reported in this study". Row 16's closure sum is positive, +1.764 × 10⁻⁶ MWh (5.04 × 10⁻⁶ E^Av). The block is
       TSO 2035 Spring, a recovered maxIterations solve.
   - **Options:**
     - (a) Mark row 16 in E.21 as "superseded (W118 v2 certificate; not decided under the rule)". Scope §4.7 and
       §2.3.3 to the current certificates. In §2.3.3, quote row 16's figure.
     - (b) Drop row 16 from Appendix E, and with it the E.22 note's 51.3 k€.
   - **Recommendation: (a).** The row is a recorded fact, and the tables should keep it, labelled. Draft sentences
     are in `P5_15_W179C_CLOSURE_SCOPE.md`.

2. **§2.3.3: the closure sentence at the other points (expert; L).**
   - Every current certificate with storage sits at IPOPT's relaxed lower bound: 44 of 44 within 5.1 × 10⁻¹⁰ MWh.
   - `d_c52e1670` at cycle 150 is now covered from a committed per-cycle record (`creep_diagnostic_per_cycle.jsonl`).
     The record's field equals the terminal levels bitwise in 47 of 47 cells.
   - At x = 0 the slack is **exactly** 0 by construction: the slacks are fixed at 0 and the closure row is
     deactivated (`model_construction_helpers.py:1143, 1166, 1176`). "Within solver tolerance" is the wrong description
     there.
   - **Recommendation:** "… at every current certificate with storage (at the solver's bound tolerance); without
     storage the closure does not arise", plus the row-16 clause of decision 1.

3. **Appendix E wording (expert).** Source: W179a.
   - **E.4 l. 2623, the consistency pass.** State the phases:
     - the NRF exceedances are counted in phase B (each DN re-solved at the TN voltage with the NRF rows off);
     - the pass is phase C (each DN re-solved with the NRF rows, then the TSO);
     - "Consistency" = Q(C) − Q(A);
     - E.23's Q is phase C, and its curtailment columns are phase A.
   - **E.6 l. 2692, "the two evaluations continued past their certificates, 14 and 3".** There are three. Evaluation
     12 (`b_2a0ba8b2`) ran to cycle 213 past its certificate at 173 and moved +17.67 € (0.004 τ; W163 N25). The 0.9 τ
     bound still holds.
   - **E.6 l. 2699, "replayed bitwise to their certification cycles".** The records show replay through cycles 132,
     112 and 87, the old residual-rule stops, not the k\* that E.21 prints. The third reference (C\*) is not in the
     appendix.
   - **Intro l. 2388, "ratios to two decimals".** E.24's residual column prints three decimals.
   - **E.3, "at its resolution σ_Q".** The search's threshold equalled σ_Q for 8 of the 14 neighbours and was
     22.1–47.2 k€ for the other 6.
   - **E.24 rows 40 and 41** print "—" for share and residual. "Share one fingerprint" rests on the gap alone there.

4. **§4.8 Scope and §2.2.7 Determinacy (expert).** Prediction (e) held. Source: W179c.
   - **§4.8 l. 1724:** "…worse, determinately for the eight re-evaluated under the certification rule (Table E.22) and
     beyond the search resolution for the other six."
   - **§2.2.7 l. 743, "the bands, gaps and slacks of every evaluation used in a reported comparison are tabulated in
     Appendix E".** Not covered:
     - the 3 × 3 pair (only a resolution, in Table 13);
     - the 0/5/8 % discount rows (E.21 has the 2 % bands only);
     - the benchmark's 71.9 k€ "reproducibility band" (not E.21 row 1's 4.2 k€);
     - the search-only neighbours (counts only).
   - **Recommendation:** "of every single-scenario certified evaluation …", with the multi-scenario pair pointed to
     Table 13. The adjacent "every cell it proposed was re-evaluated … before being reported" should say "every
     incumbent".

5. **Two W178 rows narrowed but not closed, so prediction (c) failed (expert/author; L).**
   - **C-1, Conclusions.** "with its remaining movement measured" overstates. Only x = 0 was continued, and its
     6.2 k€ is a post-hoc fit. The §4.1 search evaluations are not covered either.
   - **R14-2, letter.** The reply says "the introduction now states" fuel. "Fuel" occurs 0 times in main.tex (the
     Planner checked). The added "(Section 3.3)" lands on the mechanism.
   - **Recommendation:** reword both to what the text holds.

6. **Small items (author).**
   - **Prefix at l. 1899:** `Subsection~\ref{app:admm_updated_stopping}` prints "Subsection Appendix A.4". The Planner
     confirmed it is the last such prefix; drop "Subsection~".
   - **G12 follow-on:** the nomenclature now reserves "fixed parameters of the operational subproblem" for Ŝ/Ê, but
     §2.2.6 l. 697 still says it of S^Inv/E^Inv.
   - **§4.7 and the letter, "every evaluation … listed in Appendix E":** the 41 search evaluations and the 3 × 3 pair
     are not listed. The same applies to letter R2.2 and the "every recourse evaluation" bullet.
   - **§4.8, "same proportions":** shown for 5 of the 7 m = 2 evaluations; rows 40 and 41 print "—".
   - **Remaining comments:** four `% [CONFIRM — W179]` comments remain in Appendix E (l. 2379, 2590, 2625, 2694). Two
     cite working table numbers (E.1, E.2). W179a and W179c confirm their content, so all four can go.
   - **R2.5, "bitwise replays of every gated evaluation":** not rated (W179b, Not confirmed).

## Blocked on the author

- **Push:** everything is committed locally. The terminal cannot authenticate to GitHub, so push from PyCharm.

## Changed

| commit | content |
|---|---|
| `67c081be` | TASKS.md (order) |
| `3994ac64` | W179c: `p515_s53_w179c_closure_scope.py`; `data/SRP1/Results/P515S53/w179c_closure_scope/` (manifest of 367 inputs, 51 uncommitted, each hashed); `P5_15_W179C_CLOSURE_SCOPE.md` |
| `5b7ffcbc`, `8662ca72` | W179b: `p515_s53_w179b_checks.py`; `…/w179b_text_letter/`; `P5_15_W179B_TEXT_LETTER.md` |
| `dd74d033`, `522b5d26` | W179a: `p515_s53_w179a_appendixE_check.py`; `…/w179a_appendixE/`; `P5_15_W179A_APPENDIX_E.md` |

- No production, spec, case, frozen or manuscript file changed.
- **Erratum to W169** (recorded, no artefact touched): W169 said no record carries `d_c52e1670`'s closure at cycle
  150, and that per-cycle records carry the closure inside `slack_penalties`. Both are wrong. The closure is in
  `ess_complementarity_penalties`, which the committed `creep_diagnostic_per_cycle.jsonl` carries per block and per
  cycle. Nothing W169 scored changes.

## Found — predictions against outcomes (expert, Addendum 78)

| prediction | outcome |
|---|---|
| (a) 0 MISMATCH in table entries | **held** — 1,623 entries: 1,443 MATCH on frozen fields, 0 MISMATCH; 180 non-frozen entries all agree with their named sources |
| (a) at most two of (i)–(ix) corrected, most likely (v) and (vi) | **failed** — three corrected: (v), (vii), (viii). (vi) confirmed. (viii) is a definitional correction; without it the count is two, but they are (v) and (vii) |
| (b) 0 pointer MISMATCH | **held** — 165 pointers: 160 MATCH, 5 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS. Labels outside Appendix E are unchanged; Appendix E tables are E.20–E.24 |
| (c) all W178 rows addressed close | **failed** — of 20 addressed, 18 closed and 2 narrowed, still L (C-1, R14-2) |
| (d) the six missing points have zero closure slack, so 51 of 51 | **failed on the count** — the six are zero (five x = 0 exactly by construction; `d_c52e1670` at the bound floor), but the claim holds at **50 of 51**. Row 16 (decision 1), now reported, is the exception |
| (e) §4.8 Scope needs "under the certification rule" / "at the search resolution" | **held** — 8 under the rule (3.40–8.09×), 6 search-only (3.47–7.17× the search resolution) |

**(a) Appendix E against the frozen tables** (W179a).
- **Row mapping:**
  - E.21's 61 rows map one-to-one onto T2.
  - E.20's 58 rows are 55 distinct T1 claims plus 3 T10 claims. The unused claims are exactly the four G :net
    duplicates, the CHECK row and g070 REPORT_BESIDE.
  - Every by-number cross-reference resolves.
  - The worker's independent mapping equals `gen_appE.py`'s, 61 of 61.
- **Prose:** 66 items — 53 MATCH, 10 DERIVED, 3 MISMATCH (the "two evaluations continued", "certification cycles"
  and "two decimals" points in decision 3).
- **Counts that hold:** 55 / 0 / 1 (row 34); 42 = 49 − 3 − 4; 32 = 24 + 8; 10 = 5 + 2 + 3; median k\* 174; k\* − k₀
  median 65 (21–89); range/τ median 0.86; the ten ≥ 0.95 τ IDs; 46 = 49 − 16 − twins 56 and 61; 21 non-clean cycles
  in 10 evaluations; the T9 fingerprint; 0.74 τ and 0.78–0.88 τ.
- **Interpretations:** (i), (ii), (iv), (vi) and (ix) confirmed; (iii) confirmed with the row-16 qualification.

**(b) Main text and letter** (W179b).
- §4.6–§4.8: 70 tokens, 0 MISMATCH. The new 85.1–223.6 k€ is the minimum and maximum of the four G claims, gross and
  net, rows 17–20.
- 10 of 10 moved items and 9 of 9 kept items verified; the reproducibility paragraph moved whole to E.6.
- Letter: 41 tokens, 0 MISMATCH. Claims: 7 consistent, 2 L (decision 6).

**(c) G1–G12 and letter v5** (W179b).
- All 14 hunks are verbatim at `e926fb7` and at the HEAD.
- G3 inserts AE 0.778 and EFC/day 1.08, from W155's 0.77811 and 1.07630.
- G4 inserts 92 / 91 / 91 / 91, which holds for both instances.
- All six letter-v5 edits are verbatim.
- LM-9, 4.6-2 and R3.7(a) are closed by Appendix E.

**(d) and (e):** see decisions 1, 2 and 4.

## Not confirmed

- **Compilation:** nothing was compiled; there is no TeX here. The "Subsection Appendix A.4" rendering and the E.20–E.24
  numbering are static readings of the clone's `elsarticle.dtx`, not of Overleaf's installed class.
- **E.6 reproducibility facts:** rated through W163's committed checks, not re-measured.
- **Closure sum components:** the individual s⁺ and s⁻ at `d_c52e1670` cycle 150 are not recorded. Their sum S is
  read from the record, and with IPOPT's default `bound_relax_factor` it pins them at the bound.
- **R2.5 "every gated evaluation":** not rated.
- **The search configuration** beyond its code head `353e094b` (e.g. its ageing calibration) was not checked.
