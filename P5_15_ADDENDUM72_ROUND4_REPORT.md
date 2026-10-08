# P5.15 Addendum 72 order — W176 (round 4 checks, zero-solve)

**Authority:** Addendum 72 and `STEP6_ROUND4_CORRECTIONS.md`.
**Manuscript:** Overleaf clone at HEAD `8b76423`. main.tex sha256 `7aa9e105…`, 2,156 lines. The l. 858 reference now
points to `sec:case_ess_params`; the Planner confirmed this before launch, and W176a confirms it.
**Runs:** zero solves (guards armed and verified at 0, pickle blocked). Nothing else ran.

## Decisions needed

1. **R4-1, l. 445 (expert, L; not on the final-pass list).** The sentence "each is described by its recorded
   certificate" does not hold for s51.
   - s51's `termination_certificate` is null (`data/SRP1/Results/P515S51/campaign_s51_f2_ladder/campaign_results.json`;
     the Planner checked this).
   - Its claim scope reads "on any other termination no mesh-local claim is made".
   - The F2 plan's statement rests on s53's certificate.
   - **Recommendation:** "s47 and s53 are described by their recorded certificates; s51, stopped for review, carries
     none, and the F2 plan's statement rests on s53's".

2. **R4-2, l. 639 (expert, L — arguably M; not on the final-pass list).** The text reads "accepted its poll points by
   the coarser resolution rule".
   - The search threshold max(bar_x + bar_inc, σ_Q) is at least σ_Q = 18,449.66 €.
   - That exceeds every certified-pair threshold in the tables: 37 claims, 9,734.72–13,571.15 €. So for the x = 0
     search the rule is coarser.
   - It is finer than the uncertified-form bars: 23 claims, 18,751.08–41,380.26 €. These include all 14 F2-neighbour
     claims (27,703.27–28,205.57 €), and every reported F2 comparison uses that form.
   - **Recommendation:** drop "coarser" and write "by the resolution rule stated there".

3. **How to score the Addendum 72 predictions (expert).** See "Found" below.
   - **W176(a):** the "no M or H row" clause held. The "any L row already listed" clause failed, on R4-1 and R4-2.
   - **W176(b), item 4** expected a reduction in the submitted-version count. The count is unchanged at 559 → 559. The
     premise did not apply: at `260bd83` W174a had already treated §3.5–3.6 (l. 1066–1151) as a revised passage, with
     no submitted-version figure in it. The "only where" clause holds, but there was nothing to reduce. The Planner
     scores the item as **not applicable, premise wrong**, not as held.

4. **"ATB" has no source in the inputs (author).**
   - §3.5 and the abstract say "the NREL ATB utility-scale battery category".
   - "Lithium-ion" is sourced: the cost workbook's `'Cost breakdown NREL'!A2` reads "Li-Ion Battery cabinet".
   - "NREL utility-scale battery" is sourced by the workbook's NREL sheets and the bib entry `nrel_ess_costs` ("Utility-Scale
     Battery Storage, 2024").
   - "ATB" appears in neither the workbook, the bib entry nor `SRP1_ESS_Params.json`.
   - **Recommendation:** add the URL of the ATB page (atb.nrel.gov/electricity/2024/utility-scale_battery_storage) to the
     bib entry, or drop "ATB".

5. **Two open author items in the letter, neither a round-4 line.**
   - l. 174 (R2.8) still carries the in-text placeholder `[AUTHOR: one clause per reference … not yet added.]`. It has
     been there since `6191c6c`.
   - Several `\rchanges` lines still have bracket placeholders, e.g. l. 168, "Section~[limitations]; Table~[T2]".
     These are expected until §4 is written.

## Blocked on the author

- **The PDF the reviewers read,** for the figure numbering in R1.5 and the "Figure 15" reference (letter findings G3
  and G4, still open).
- **The datasheet citations** behind the 10,000-cycle and 8,000-cycle counts. No source for the baseline's 10,000
  cycles is recorded anywhere in the repository or the inputs.
- **The `% [AUTHOR]` note under the ageing table** was deleted between `260bd83` and `c71ca2e`. Round 4 had said it
  should stay until the citations were in. It is recorded here; it is superseded by Addendum 72, which leaves the
  citations to the final pass.

## Changed

| task | script | outputs / report |
|---|---|---|
| order | — | `bf4b626c` TASKS.md |
| W176a | `p515_s53_w176a_round4_lines_checks.py` | `a46d1dd8` `P5_15_W176A_ROUND4_LINES.md`, `data/SRP1/Results/P515S53/w176a_round4_lines/` |
| W176b | `de7b55db` `p515_s53_w176_manuscript_number_check.py` (declarations v4; imports W174a) | `9b8ad153` `data/SRP1/Results/P515S53/w176_manuscript_check/overleaf_8b76423/` |

- No production, spec, case, frozen or manuscript file changed.
- W176a discarded its own first run, which had misread the claim status strings. Its outputs were uncommitted and
  cited nowhere.

## Found — predictions against outcomes

| prediction (expert, Addendum 72) | outcome |
|---|---|
| W176(a): no M or H row among the round-4 lines | **held** — H 0, M 0 |
| W176(a): any L row is already on the final-pass list | **failed** — R4-1 and R4-2 are new (decisions 1–2); both are wording in round-4 text, and no number changes |
| W176(b): zero MISMATCH in main.tex | **held** — 0 |
| W176(b): zero MISMATCH in the letter | **held** — 0 |
| W176(b): the same twelve approximates as W174a | **held** — the same 12 tokens (35,851 ×5, 80 %, 11,542 ×2, 22,429, 227,211, 386,259, 4,539.07, 18,449.66), none added or removed |
| W176(b): submitted-version count reduced only where §3.5–3.6 replaced submitted text | **premise wrong** — 559 → 559, unchanged (decision 3) |

**W176a — round-4 lines.**
- All fourteen edits and the l. 858 reference are present: 21 sub-checks verbatim, 17 old wordings absent.
- The following are consistent with records and code:
  - the stop reasons of s47, s51 and s53, and "every evaluated poll";
  - SoH at the end of the block;
  - both §3.5 references;
  - the chemistry, now lithium-ion;
  - the C4 footnote (10,000 × 0.80 = 8,000 × 1.00; k = 22,429.386);
  - "one frozen configuration", scoped (49 certified cells from 7 campaign roots, code identical to HEAD; the three
    search campaigns had no tail);
  - the benchmark paragraph (x = 0 at the single-scenario instance; consistency pass on 36/36 DN blocks in all six
    runs; evaluation tie-breaker 0; passive tie-breaker 1 €/MWh);
  - κ^E, the restored AA-off clause, and `k ← k + 1` placed before the While closes.
- **"Section 4.x" literals remaining in §2–§3:** 8 occurrences, at l. 377, 415, 445, 631, 635, 671, 1066 and 1119.
  There are none elsewhere in main.tex. They are on the final-pass list until §4 is written.
- **Remaining comments:** no `[CONFIRM` in main.tex, the letter, the cover letter or the highlights. No `[AUTHOR]` or
  `[AUTHOR:` in main.tex. The letter has one `[AUTHOR:` (decision 5).
- **Noted, not counted as rows:**
  - l. 639, "every cell it proposed": consistent if it means the incumbents. On the wider reading, ten search
    evaluations appear only as counts and were never re-evaluated.
  - The code triggers the consistency pass globally, where the text says "where a DN limit is violated". The two
    coincide in every reported run.

**W176b — number check.**
- **main.tex:** 2,173 tokens in scope. 388 match (305 declared, 83 by rule), 12 approximate, 0 MISMATCH, 559
  submitted-version, 1,214 unchecked, 0 unassigned.
- **Letter:** 97 match, 0 MISMATCH, 62 quotation numbers verified. "Single machine" stands on the author's
  attestation.
- **Cover letter:** 2/2 match.
- **Declarations:** of the 220 v3 declarations, 217 were carried and 3 replaced, with the same evaluation. 8 new v4
  numbers were declared and all match:
  - stop reasons;
  - the C4 footnote;
  - three search campaigns;
  - 1 €/MWh;
  - the benchmark instance;
  - the settlement weight set to one (`shared_resources_planning.py:5050, 5340`).
- **§3.5–3.6 value table:** 145 values (133 match, 12 approximate) and 27 named settings (26 match, 1 without a
  source: "ATB").
- **Scope of the round-4 changes:** 34 main.tex lines changed, all inside §2, §3.5–3.6 and Appendix A, and 1 letter
  line.

## Not confirmed

- **IPOPT acceptable-exit code:** whether it falls in Pyomo's optimal band. The solve records carry the message, not
  the Pyomo status. W172 had the same limit.
- **The NREL ATB page:** not opened.
- **Reflowed number words** ("one frozen configuration", "three", "one"): left rule-unchecked, as in W174a.
- **"Earlier states of the code":** rests on W174b's record that the tail code is absent at each search campaign's
  run head.
- **Compiled PDF:** none exists; everything was checked in source only.

## Standing by

Next: the number check (the W176 script lineage, with new declarations at the round's HEAD) at the close of each
writing round:
- the author's §3 data figures and §3.3–3.4;
- §4 from the frozen tables and the W165 figures;
- §1, §5, front matter and Appendices B–E.

No run is queued.
