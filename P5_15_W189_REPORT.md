# P5.15 W189 — round-9b audit at Overleaf `46c9d77`, supplement v2 (W189a), and W187 spec v1.1 with its gates (Addendum 90)

**Authority:** Addendum 90 (Planner order: W189a, W187 spec v1.1 + gates, W189).

**Overleaf HEAD `46c9d77`** (`46c9d779743b…`), one commit after `a87e195`, with round 9b pasted. The following are
byte-identical to their deliverables:
- `main.tex` (md5 `49139cc2…`) = `main_a87e195_round9b.tex`;
- the letter (`ed8f5b13…`) = `response_to_reviewers_v12.tex`;
- `highlights.tex` (`4cedd45c…`) = v3;
- `cover_letter.tex` (`0b9e4c21…`) = v3.

The clone diffs equal the expert's diffs (main 34/34 hunks, letter 10/10).

**Code.**
- Production head `f6c23977`; the three P5.16 modules are re-pinned in spec v1.1.
- Step-7 launcher `069a7b88…` (G0 re-pin only).

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded.

**Runs.**
- **Solves: 153, all in W187 gate (ii)** (51 × 3, 0 retries, verified by the gate's armed guard).
- **Everything else** was zero-solve, with guards 0 and pickle loads 0.
- **Order.** The supplement and the two audit parts ran in parallel. Spec v1.1 and its gate then ran alone.

| part | report | commits |
|---|---|---|
| W189a supplement v2 | `P5_15_W188B_SUPPLEMENT_v2.md` | `80778c12`, `3f396b1d` |
| W189 part 1, numbers and claims | `P5_15_W189_P1_NUMBERS.md` | `1eac8803`, `4d03909d` |
| W189 part 2, identity, labels, pointers, structure | `P5_15_W189_P2_STRUCTURE.md` | `8c931b0b`, `f116cbba` |
| W187 spec v1.1, gates (i)–(ii), screen freeze | `P5_15_W187_SPEC_V1_1.md` | `68a612ed`, `cc03bbaf`, `31b38203`, `9e470279`, `c2743027` |

## Decisions needed

1. **Launch W187 (author/Planner; authorized by ruling 4, "launch by 2026-11-20").**
   - **Readiness.** Spec v1.1 is frozen and gate (ii) passed under the ruled comparator. The screen is frozen at
     `P515S55/e1_screen_1x`: 15 cells, cap 500, stage spec `frozen_s55_step7_screen_spec_v1_5cbf4a55.json`. All
     preconditions of the first cell hold.
   - **The run.** About 21 h serial, attached, one cell per process, each committed before the next.
   - **P3** stands as a blind prediction (recorded 11-10, before the run).
   - I held the launch because this turn's instruction ended at the gates. Recommend launching now.
2. **The reduced Appendix E was hand-edited (expert).** This breaks Addendum 83 ruling 1, under which Appendix E is
   edited only through the generator.
   - The block at `46c9d77` is the `gen_appE.py` output (generator unchanged since `22e19b56`) with two subsections
     deleted (E.21, 100 lines; E.23/E.24, 74 lines) and five prose lines edited by hand.
   - `verify_appE.py` was not extended to the reduced table, though Addendum 89 ruling 7 says it was. The comparisons
     table keeps all 58 rows, not only the rows the main text cites.
   - Every printed entry still equals the frozen JSON: 657 entries, 0 MISMATCH.
   - **Recommendation:** port the reduction into `gen_appE.py` (a `--reduced` mode) and extend `verify_appE.py`, so the
     next W-check is bitwise again.
3. **Supplement v2 is not submittable yet (expert).**
   - **The README prose describes v1.** v2 is `SRP1_supplement_v2_864a5779.zip`; the verifier passes with 0 mismatches
     over 1,623 entries and 70,851 case-file cells, and `D_ieee33_tn_connection.csv` is unchanged (node 7 ±100). But the
     prose, prepended verbatim as ordered, says "Supplement version 1", says "70,743 case-file cells" (v2 checks
     70,851), and misdescribes `E_table_notes.csv`.
   - **Recommendation:** prose v2 from the expert, then a v3 regeneration (predecessor `864a5779`; about 6 min, zero
     solves).
   - **Naming.** The manuscript cites "Section S5" (l. 1353, 1627) for what l. 2120 and the README call "Table S5". One
     name should be used throughout.
4. **Round-9b wording (expert).** No number is affected (W189 part 1: 0 MISMATCH over 472 tokens; the W184/W185/G10
   wording holds against rulings 1–3; "prediction" appears 0 times for P1/P2/P5).
   - **B.4 l. 2310 calls the agent's net-power pair a slack "in Q".** That pair is penalised only in the ESSO objective
     and is not in Q (W188 (a)); "at its zero bound" is true. The Planner read the line.
   - **Figure 6: "cycles … between the 10 % and 90 % limits" holds on 6 of 8 days.** The two Winter days stay above 10 %
     (minima 13.9 % and 15.4 % of E^Av). The other Figure 6 numbers match the W188 (d) CSVs, and the plotted lines match
     the CSVs (≤ 5e-7 pt).
   - **171 €** is evaluation 16's total positive content: voltage 169.27 € plus closure 1.42 €. **−13.0 k€** reads
     −13.1 k€ on one row; "within 40 €" covers it.
   - **Letter v12 l. 285 prints "Fig.~B.8".** The static count gives **B.9**: elsarticle does not reset the figure
     counter at `\appendix`, and there are 8 main-text figures. Please read the number from the Overleaf compile.
   - **Five changed lines lack `\textcolor{blue}`:** l. 1250, 1685, 1695, 1709, 1710 (the Table 9 panel headers and
     notes).
5. **Trimming target not met (author; Addendum 89 decision 4).**
   - **Main text:** 17 floats (9 tables + 8 figures) against ≤ 10; 41 numbered equations; 6,692 prose words (6,130 at
     `a87e195`).
   - **Appendix A:** 2,605 words.
   - **Appendix B (the reduced E):** 1,550 words, 2 tables, 1 figure.
   - **Appendices:** 2, down from 5.
   - A.1 and A.6 still exist (Addendum 89 ruling 7 said they were folded into A.2; what was folded is the old A.7 into
     A.1).
6. **Spec v1.1 pointer (record).**
   - **The pointer.** One status string points to `p516_inputs.static_search`; the content sits under
     `spec_v1_1.p516_inputs`. The spec is write-once, so this is left as recorded, unless the expert wants a v1.2.
   - **E1 sections.** They are carried from v1 unchanged (production `6f43479e`). The E1 launcher still pins the old G0,
     so any E1 launch would need its own re-pin; W186 is withdrawn.

## Blocked on the author

- **Push from PyCharm.** The terminal cannot authenticate.
- **The compiled figure number** for letter l. 285 (decision 4).
- **The deadline date.**

## Changed

- See the part table.
- **Launcher:** `G0_ARTIFACT_REL`/`G0_ARTIFACT_SHA256` re-pointed to the pin-check artifact
  `pincheck_nl_head_f6c23977/w183_nl_digest.json.gz` (`315ca94a…`, built at `f6c23977`). The Planner read the diff:
  exactly those two lines. This was ruled by the Planner within Addendum 90 ruling 4's re-pin authority, because
  `g0_check(require_current_production=True)` refused the `6f43479e`-era artifact.
- **The two P5.16 inputs** stay pinned by the launcher, which is stricter than excluded; spec v1.1 records them as
  outside every SRP1 run's read set (static search).
- **Spec v1.1:** `frozen_step7_campaign_spec_v1_1_25071d2f.json` (`25071d2f…`, predecessor `96345241…`).
- **Supplement:** `manuscript_review/supplement_v2/` (v1 untouched); generator and verifier extended.
- **No change to:** production, harness, hooks, gate chain, spec v1, the clone, `gen_appE.py`/`verify_appE.py`, or
  committed artifacts.

## Found — predictions against outcomes

| prediction | outcome |
|---|---|
| W187 gate (ii): 0 genuine, `GATE_PASS_RULED` True, provenance ≤ 4 (Addendum 90 ruling 4) | **held** — 153/153 solves, 0 retries. GATE_PASS (frozen) False on the W64 field-list item only; RULED True. 0 genuine vs W48 (provenance 3) and vs W83 (provenance 4). Q = 740,922,817.2425401, bitwise equal to W48/W83. `per_cycle_record.jsonl` byte-identical to the W183i arm's |
| W187 gate (i): the pin check cited | **held** — `09b3e3cf`, 696/696 blocks, Q aggregation AST-equal; cited in spec v1.1 |
| W189a: supplement v2 verifier 0 mismatches; TN-connection CSV unchanged | **held** — 1,623 entries 0 MISMATCH; 70,851 case-file cells 0 differences; paper Table 2 rebuilt 48/48; v1→v2 diff = README + the two RES CSVs (+ SHA256SUMS); determinism 3 runs byte-identical |
| W189: numbers 0 MISMATCH vs the frozen tables, reduced Appendix E included | **held** — 278 MATCH, 22 DERIVED, 0 MISMATCH; reduced Appendix E 657 entries 0 MISMATCH; `verify_appE.py` 33 PASS (not extended; decision 2) |
| W189: letter v12 pointers; Table 7 at the benchmark bullet and R2.6 | **1 MISMATCH** of 180 (l. 285, decision 4): 171 MATCH, 8 SUBMITTED. Table 7 appears once, at R2.6 (l. 170), MATCH; the benchmark bullet carries section pointers only, all MATCH |
| W189: 0 `[CONFIRM`, 0 `\b(Appendix\|Section)~\ref{app:`, blue wrappers | 0 and 0 **held**; blue **missing on 5 lines** (decision 4) |
| W189: Figure 6 prose vs the W188 (d) CSVs | **held on every number**; one sentence holds on 6/8 days (decision 4) |
| W189: no dangling references | **held** — 0 undefined, 0 duplicate labels, 0 uses of the 35 removed labels; 136 → 102 labels |

## Not confirmed

- **Compilation.** Nothing was compiled; there is no TeX here. Label, figure, page and float numbers are static, and
  the expert's 63 pages is not reproduced.
- **The screen run past its preconditions** (lock, `H.evaluate`) has not executed.
- **The P5.16 read-set claim** is static: name search, AST listing-call search and `SRP1.json` references. No file
  opens were traced.
- **The parallel DSO builder** is not covered by gate (ii); it is sequential-only, as configured.
