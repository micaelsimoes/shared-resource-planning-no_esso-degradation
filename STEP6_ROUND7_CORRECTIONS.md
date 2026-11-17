# Step 6, round 7 — after `P5_15_W180_REPORT.md` (`a1ab9e99`; audited Overleaf `e5322f8`) (expert, 2026-10-16)

Rulings: `PLANNER_BRIEF_2026-09-13.md` Addendum 80. Base: Overleaf `e5322f8` (`main.tex` =
`manuscript_review/main_48099ab_round6.tex`, letter = v7; both byte-identical per W180 and the expert's md5).
Four one-sentence edits in `main.tex`, one word in the letter, and one comment restored. Replacement text is blue
where the surrounding text is.

**Deliverables (`manuscript_review/`):** `main_e5322f8_round7.tex` (full `main.tex`; + `main_e5322f8_round7.diff`,
6 hunks), `response_to_reviewers_v8.tex` (+ `letter_v7_v8.diff`, 2 hunks), `gen_appE.py` (patched: E.4 second
phase, E.21 note), `appendixE_evaluation_record.tex` (regenerated), `verify_appE.py` unchanged (33 PASS).

**Compile check:** 0 errors, 0 undefined, 0 multiply-defined; every label number identical to `e5322f8`; the
appendix block equals the generator's output. The letter compiles alone.

## A. `main.tex`

1. **§2.3.3, closure sentence (W180 decision 2).** "Without storage the closure row is not built." → "Without
   storage the closure row is deactivated and its slacks are fixed at zero." (`network.py:511` builds the row;
   `model_construction_helpers.py:1143, 1166, 1176` fix the slacks at 0, bound them to [0, 0] and deactivate the row.)

2. **Table 13, l. 1619 (W180 decision 4).** "at operational stopping (cycle 72)" → "at operational stopping
   (cycle 72 for the plan without storage, 69 for the unit)". The value 236.7 k€ is untouched; the frozen JSON's label
   still says "cycle 72" and is not edited (frozen artefact). Source: W180d, `campaign_results.json` (`n7_4h_e1` 69).

3. **Appendix E.4, second phase (W180 decision 4; generated).** "a re-solve of each distribution network at the
   transmission network's interface voltage with its no-reverse-flow rows switched off, in which …" → "a
   re-evaluation of each distribution network at the transmission network's interface voltage, with its decisions
   held at the first phase and its voltage, thermal and no-reverse-flow rows switched off, in which …". Source:
   W180a's reading of `p515_s53_w116_benchmark_nrf.py` phase B.

4. **Table E.21 note (W180 decision 4; generated).** "Evaluations~1, 2, 40 and~41 continue earlier runs; their first
   residual passes are not tabulated" → "Evaluations~1 and~2 continue earlier runs, and evaluations~40 and~41 are
   re-decided from the records of earlier runs; their first residual passes are not tabulated". Source: W180b (rows
   40, 41 uncertified under v2 and v6 on record); W163 N49 (the continuations are x = 0, unit, C\*).

5. **§3.5, under Table 6 (W180 decision 3).** The `% [AUTHOR]` reminder is restored as a comment: the datasheet
   citations for the two cycle-life statements (10,000 cycles at 0.80 DoD; 8,000 full cycles, EVE MB31) and for the
   calendar retention 0.985/yr are **not in the source** (the only `\cite` in §3.5 is `nrel_ess_costs` for the
   technology category). They must be added before submission — author's action, see Addendum 80.

**Accepted as L, not changed (W180 decision 4):** "0.004τ" printed to three decimals in E.6 (the intro's rounding
sentence is about the tables); the six search-only neighbours' bands not tabulated (covered by Addendum 79 ruling 4's
residues); the cause "gap clause" for rows 39 and 41 resting on W153's first-match rule, which E.21's caption and
§4.7 state as the rule used.

## B. Letter v8

- **R2.2 (W180 decision 1, option b).** "and the cycles of every evaluation behind a reported comparison in
  Appendix~E" → "and the cycles of every single-scenario evaluation behind a reported comparison in Appendix~E".
  No pointer changes.

## C. Author's steps

1. Paste `main_e5322f8_round7.tex` over `main.tex` and `response_to_reviewers_v8.tex` over
   `response_to_reviewers_draft.tex`; compile (0 errors, numbering unchanged).
2. Pull the clone on the Air; stage by filename `STEP6_ROUND7_CORRECTIONS.md`, `PLANNER_BRIEF_2026-09-13.md`,
   `manuscript_review/{main_e5322f8_round7.tex, main_e5322f8_round7.diff, response_to_reviewers_v8.tex,
   letter_v7_v8.diff, gen_appE.py, appendixE_evaluation_record.tex, W177B_CHANGE_LOG.md}`; commit
   ("Round 7: closure wording, Table 13 cycles, E.4 phase B, E.21 note, letter v8 (Addendum 80)"); push; pull on the
   Studio (`git pull --no-rebase`).
3. Forward the W181 prompt (Addendum 80).
4. **Before submission, independently of W181:** add the three ageing citations under §3.5 and delete the
   `% [AUTHOR]` comment; tell me which references you add so the letter's R3.4/R3.5 replies can cite them.
