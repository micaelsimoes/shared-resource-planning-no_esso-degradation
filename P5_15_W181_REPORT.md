# P5.15 W181 — round 7 at Overleaf `cd57efc` (Addendum 80; zero-solve; short check)

**Authority:** Addendum 80 (W181 order). This is the short check of ruling 5, not a re-audit.

**HEAD audited: Overleaf `cd57efc`** (`cd57efcb5439…`, 2026-10-16 13:23 UTC), one commit after `e5322f8`.
- The clone was fast-forwarded from `e5322f8`; it has no local changes.
- **main.tex** (sha256 `ff7b045e…`, 2,713 lines) is **byte-identical to `manuscript_review/main_e5322f8_round7.tex`**.
- **The letter** (`06cc2460…`, 292 lines) is byte-identical to `response_to_reviewers_v8.tex`.
- The Planner checked both with `cmp`, and W181a re-verified them.
- The round-7 delivery is in the repository: `fc2fad77` / `2e3ec946` (Addendum 80, `STEP6_ROUND7_CORRECTIONS.md`, the
  `manuscript_review/` files).

**Objective convention:** Q = `gross_operational_cost`, settlement excluded.

**Runs:** zero solves.
- `SolveProfileGuard(permitted=())` was armed before any project import and verified at exactly 0, with pickle
  blocked (0 calls), in both parts.
- W181b imported no production module and read source files as text.
- Both parts ran in parallel, as static readings with separate scripts and output directories.
- Scratch trials went into uniquely named directories; the W180 scratchpad collision did not recur.

| item | report | commits |
|---|---|---|
| (a) identity, diffs; (b) Appendix E | `P5_15_W181A_IDENTITY_APPENDIX_E.md` | `ed6490c6` (harness, before the run), `4bcc2463` |
| (c) five sentences, R22-b; (d) citations | `P5_15_W181B_SENTENCES_CITATIONS.md` | `0692c852` |

**Every prediction held. Addendum 80's order is complete.** What follows is the expert's latexdiff, the cover letter,
and submission once the citations are in.

## Decisions needed

None. Every recorded prediction held, and no wording residue beyond those Addendum 80 accepted was found.

## Blocked on the author

1. **The three ageing citations (Addendum 80 ruling 3) are still absent at `cd57efc`.**
   - §3.5 (l. 1264–1309) has one `\cite`, `nrel_ess_costs`, and it resolves in `bibliography.bib`.
   - The bibliography has no entry for EVE, MB31, calendar ageing or LFP.
   - Still to add:
     - (i) the 10,000-cycle row at 0.80 DoD;
     - (ii) the 8,000 full-cycle row, from the EVE MB31 datasheet;
     - (iii) the 0.985/yr calendar retention, from an LFP calendar-ageing study.
   - Then delete the `% [AUTHOR]` comment at l. 1307–1308 and tell the expert which references were used, so the
     letter's R3.4 and R3.5 replies can name them.
2. **Push:** if the terminal push of this commit fails, push from PyCharm.

## Changed

| commit | content |
|---|---|
| `ed6490c6`, `4bcc2463` | W181a: `p515_s53_w181a_identity_appendixE.py`; `data/SRP1/Results/P515S53/w181a_identity_appendixE/` (generator output `appE_generated_w181a.tex`, verifier output, check JSON/MD, manifests, launch log, bool-typing test); `P5_15_W181A_IDENTITY_APPENDIX_E.md` |
| `0692c852` | W181b: `p515_s53_w181b_sentences_citations.py`; `…/w181b_sentences_citations/` (JSON, manifest of 56 inputs, all committed, launch log); `P5_15_W181B_SENTENCES_CITATIONS.md` |
| this commit | `P5_15_W181_REPORT.md`, TASKS.md, REVISION_CONTEXT.md |

- No production, spec, case, frozen, `manuscript_review/` or manuscript file changed.
- No earlier W-script or output was re-run or overwritten.
- `gen_appE.py` wrote only into `w181a_identity_appendixE/`.

## Found — predictions against outcomes (expert, Addendum 80)

| prediction | outcome |
|---|---|
| (a) main.tex and letter identical; the clone diff equals the expert's diffs | **held** — `git diff e5322f8 cd57efc` touches only main.tex (6 hunks) and the letter (2 hunks). Their +/− lines, ranges and context equal `main_e5322f8_round7.diff` and `letter_v7_v8.diff` in order. Negative control caught |
| (b) Appendix E regenerated bitwise | **held** — the patched generator's output (321 lines, `623ddbbd…`) occurs once in main.tex, at l. 2377–2697, line-aligned. It also equals the committed `appendixE_evaluation_record.tex`. The Planner re-checked both with `cmp` |
| (b) `verify_appE.py` 33 PASS | **held** — 33 PASS, 0 FAIL |
| (b) 0 MISMATCH | **held** — 1,623 entries: 1,443 frozen MATCH, 0 MISMATCH; 180 non-frozen agree with their sources. Mutation test 13/13 caught |
| (b) 0 cells changed | **held** — 0 of 1,637 cells, and no caption or group header changed. The block differs in three hunks: E.4 prose (l. 2624), the E.21 note (l. 2583) and the generator's header comment (l. 2383–2384). The comment is the expert's hunk 4 and does not print. The Planner's task text had named only the first two |
| (c) all five touched sentences MATCH | **held** — see below |
| (c) R22-b closes | **held** — closed |
| (d) citations absent, or present with three keys | **held: absent** (Blocked 1) |
| 0 `[CONFIRM`; one `[AUTHOR` | **held** — 0 and 0 in main.tex and the letter; 1 `[AUTHOR` (main.tex l. 1307) |

**(c) the five sentences** (W181b; the Planner read each line at the HEAD).
1. **§2.3.3, l. 969 — MATCH.** "Without storage the closure row is deactivated and its slacks are fixed at zero."
   - `network.py:511` builds `sess_soc_final` unconditionally in `_build_model`.
   - In `model_construction_helpers.py`, under `inactive`:
     - `:1143` `entry.fix(0.0)`;
     - `:1166–1169`, bounds [0, 0];
     - `:1176` `entry.deactivate()`.
   - The x = 0 candidate reaches this path, as a capacity within 1e-10 of zero. The lines are unchanged at all five
     x = 0 run heads.
2. **Table 13, l. 1621 — MATCH.** "(cycle 72 for the plan without storage, 69 for the unit)". The row is at l. 1621,
   not ≈ 1619; the restored comment shifts it.
   - In `w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json`, x0 stopped at 72 and `n7_4h_e1` at 69.
   - `value_eur` 236,695.25 prints as 236.7 k€, unchanged.
   - The frozen JSON label still reads "(cycle 72)". It is frozen and was not edited.
3. **E.4, l. 2624 — MATCH.** The second phase is fed the phase-A DSO solution at the TN interface voltage, with no TSO
   solve (W116 harness l. 615–617).
   - In `uncoordinated_benchmark.build_consistency_reevaluation_block` (l. 1411):
     - the DSO decisions and the non-reference generators are fixed at phase A (l. 1442, 1452);
     - the voltage and thermal rows are deactivated (l. 1487–1492), and so is the no-reverse-flow row (l. 1495–1498).
   - All six NRF arm records show these five families on all 36 blocks.
   - The code also fixes some slacks, relaxes some bounds and leaves the reference generator free. The sentence
     omits these, and none contradicts it.
4. **E.21 note, l. 2583 — MATCH.**
   - Rows 1 and 2 are reference continuations (W163 N49).
   - Rows 40 and 41 were issued under v2 and re-decided by v6 from their records (W142 item 1; W180b); they are not
     continuations.
5. **Letter R2.2, l. 125 — MATCH; R22-b closed.**
   - "single-scenario" removes the 3 × 3 residue W180c found.
   - The six search-only neighbours sit in R2.2's search class, already L (Addendum 80 ruling 4).
   - Between v7 and v8 only l. 1 and l. 125 changed, and the pointer list is unchanged.

## Not confirmed

- **Compilation:** nothing was compiled; there is no TeX here. "Table 13" and "Table 6" rest on the labels and on
  W180c's static numbering; the expert's "0 errors, labels unchanged" was not reproduced.
- **The solved problem:** whether the NLP handed to IPOPT omits the deactivated row was not tested. Ruling 2 rests the
  sentence on the code only.
- **Phase B** was not replayed. S3 rests on the source, the module hashes recorded at the run and the per-block
  records.
- **Pointers:** only v7 = v8 equality of the pointer list was checked. Their targets are W180c's check, not re-run.
- **Citation search scope:** §3.5, main.tex and letter comment lines, and `bibliography.bib`. Not searched: other
  Overleaf files, or cycle-life citations elsewhere in main.tex.
- **The Overleaf remote** was fetched by the Planner before the fast-forward. The Workers read the clone as found.
