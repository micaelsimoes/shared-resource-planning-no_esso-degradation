# Step 6, round 6 — after `P5_15_W179_REPORT.md` (`43c72e06`; audited Overleaf `48099ab`) (expert, 2026-10-15)

Rulings: `PLANNER_BRIEF_2026-09-13.md` Addendum 79. Lines refer to the Overleaf clone at `48099ab`
(`main.tex` = `manuscript_review/main_e926fb7_appE.tex`, letter = `response_to_reviewers_v6.tex`; both verified
byte-identical by md5 before this round). Replacement text is blue, as before.

**Deliverables (all in `manuscript_review/`):**

| file | content |
|---|---|
| `main_48099ab_round6.tex` | the full corrected `main.tex` — paste over Overleaf's `main.tex` |
| `main_48099ab_round6.diff` | its diff against `48099ab` (16 hunks) |
| `response_to_reviewers_v7.tex` (+ `letter_v6_v7.diff`, 5 hunks) | letter v7 — paste over `response_to_reviewers_draft.tex` |
| `gen_appE.py`, `verify_appE.py` | generator and verifier, patched (row 16, E.3–E.6 prose, comments closed; three new checks) |
| `appendixE_evaluation_record.tex` | regenerated Appendix E (= lines 2375–2694 of the new `main.tex`) |

**Compile check (demo graphics, `elsarticle.cls` rebuilt from the clone's `.ins`):** 0 errors, 0 undefined
references, 0 multiply-defined labels; every `\newlabel` number identical to `48099ab` (equations, tables, figures,
sections, algorithms); one label added (`subsubsec:capacity`, §2.3.2, for edit 6b); page count 81 → 82. The letter
compiles on its own. `verify_appE.py`: 33 PASS, 0 FAIL. The generator reproduces the committed Appendix E bitwise
before the patch, and the new appendix block is byte-identical to the patched generator's output.

---

## A. `main.tex` — the edits, in file order (what they answer in brackets)

1. **§2.2.6, l. 697 [W179b Blocked 3, G12 follow-on].** "are fixed parameters of the operational subproblem. The
   resulting evaluated operating cost" → "are held fixed in the operational subproblem, where they enter as the
   parameters $\hat{S}^{\text{Inv}}_{e,y^{\text{Inv}}}$ and $\hat{E}^{\text{Inv}}_{e,y^{\text{Inv}}}$ of
   Subsection~\ref{subsubsec:capacity}. The resulting evaluated operating cost". A `\label{subsubsec:capacity}` is
   added under the §2.3.2 heading (l. 768).

2. **§2.2.7 Determinacy, l. 743 [decision 4].** "every cell it proposed was re-evaluated under this rule before
   being reported" → "every incumbent it proposed was re-evaluated …"; and "the bands, gaps and slacks of every
   evaluation used in a reported comparison are tabulated in~\ref{app:results}." → "the bands, gaps and slacks of
   every single-scenario evaluation used in a reported comparison are tabulated in~\ref{app:results}; the
   multi-scenario evaluations, stopped by the production exit, carry the resolution stated with them in
   Table~\ref{tab:res_multiscenario}."

3. **§2.3.3 closure sentence, l. 968 [decisions 1 and 2].** Replace the sentence by:

   ```latex
   At every certificate used in a reported comparison, and at both multi-scenario evaluations, both daily closure
   slacks sat at their lower bound to within the solver's bound relaxation ($10^{-6}$~MWh per slack); in the
   superseded evaluation~16 of Table~\ref{tab:appE_evaluations} their sum reached $1.8 \times 10^{-6}$~MWh,
   $5 \times 10^{-6}$ of the available energy. Without storage the closure row is not built.
   ```

   Source: W179c (44 of 44 current certificates with storage within $5.1 \times 10^{-10}$ MWh of the relaxed bound;
   five x = 0 cells exactly 0 by construction, `model_construction_helpers.py:1143, 1166, 1176`; `d_c52e1670` at
   k\* = 150 from the committed per-cycle record; row 16: $+1.764 \times 10^{-6}$ MWh, $5.04 \times 10^{-6}$ E^Av,
   TSO 2035 Spring after a recovered maxIterations solve; T11 pair at the bound).

4. **§4.7, l. 1689 [decision 1].** "Every certificate reported in this paper was decided, …; the single-scenario
   tables carry these certificates." → "Every certificate used in a reported comparison was decided, or re-decided
   from its committed record, under the rule …, and certifies at the cycle reported; the single-scenario tables
   carry these certificates. One earlier certificate, issued under a previous version of the rule and since
   superseded, is recorded in \ref{app:results} and used in no comparison."

5. **§4.7, l. 1690, last sentence [W179b C47-1].** "Every comparison and every evaluation behind this section is
   listed in~\ref{app:results}, together with the full certification statistics and the reproducibility record."
   → "Every comparison of this section, and every single-scenario evaluation behind it, is listed
   in~\ref{app:results}, together with the full certification statistics and the reproducibility record; the
   search's own evaluations, stopped at the production exit, are reported by their counts, and the multi-scenario
   pair in Table~\ref{tab:res_multiscenario}."

6. **§4.8, l. 1716 [W179b C48-1].** "split in the same proportions among the three interface nodes
   (Table~\ref{tab:appE_gap})" → "split in the same proportions among the three interface nodes wherever the split
   was recorded (Table~\ref{tab:appE_gap})".

7. **§4.8 Scope, l. 1724 [decision 4; prediction (e) held].** "with every evaluated unit neighbour of the plan
   without storage determinately worse." → "with every evaluated unit neighbour of the plan without storage worse,
   determinately for the eight re-evaluated under the certification rule (Table~\ref{tab:appE_neighbours}) and
   beyond the search resolution for the other six."

8. **Appendix A.3, l. 1899 [W179b Blocked 2].** "(Subsection~\ref{app:admm_updated_stopping})" →
   "(\ref{app:admm_updated_stopping})". This was the last `<word>~\ref{app:…}` prefix.

9. **Conclusions, l. 1736 [decision 5, C-1].** "or, at the multi-scenario instance, stopped by the production exit
   with its remaining movement measured --- and every planning comparison is reported with the resolution at which
   it is decided." → "or, at the multi-scenario instance, stopped by the production exit, with the remaining
   movement of the plan without storage estimated from a continued run --- and every planning comparison is
   reported with the resolution at which it is decided, the search's own evaluations at the search resolution."

10. **Appendix E (regenerated; the block l. 2374–2701 of `48099ab` is replaced whole) [decisions 1 and 3].**
    - Header comment: the `[CONFIRM — W179]` mapping note → an audit note (W179a, 0 MISMATCH in 1,623 entries).
    - Intro, l. 2388: "ratios to two decimals." → "ratios to two decimals, except the residual ratio of
      Table~\ref{tab:appE_gap}, given to three."
    - **Table E.21, row 16:** Status "cert., osc.$^{s}$" → "superseded$^{s}$"; Rule "record" → "---"; k\* 167,
      range 0.99 and band 4.5 (the earlier certificate's figures) kept. Caption: Status legend "…, or the cause that
      left the evaluation uncertified (…), or superseded"; "$^{s}$: superseded by evaluation~15" → "$^{s}$:
      superseded by evaluation~15; certified under an earlier version of the rule, whose cycle, range and band are
      shown, and not decided under the rule of \ref{app:admm_updated_certification} within its recorded cycles".
      The E.21 note and the E.22 note (51.3 k€, 3.79×) are unchanged: the figure stays as a recorded fact.
    - E.3, l. 2588: "at its resolution $\sigma_Q$" → "at its own resolution ($\sigma_Q$, or the sum of the two
      evaluations' bars where larger; …)"; new last sentence: "The other six, the smallest unit at the two remaining
      pairs of nodes and at all three nodes, in 2025 or 2030, were evaluated by the search only, on an earlier state
      of the code and stopped by the production exit; each cost more than the plan without storage by 3.47--7.17
      times the search resolution." (Source: W173 `w173_search_counts.json` item2.x0.rows, via W179c: 5.52, 6.16,
      5.57, 6.07, 3.47, 7.17×; the acceptance threshold equalled $\sigma_Q$ for 8 of the 14 and 22.1–47.2 k€ for the
      other 6, W179a E3.5.) The `[CONFIRM — W179]` under it is removed.
    - E.4, l. 2623 (the consistency pass): the second sentence → "Each arrangement is solved in three phases
      (Section~\ref{sec:case_settings}): the distribution networks' dispatch and the transmission network's dispatch
      against it, from which the curtailment columns are taken; a re-solve of each distribution network at the
      transmission network's interface voltage with its no-reverse-flow rows switched off, in which the exceedance
      columns count the network-hours (of 864: three networks, twelve blocks, 24~h) whose interface flow exceeds the
      no-reverse-flow limit and give the largest exceedance; and the consistency pass, a re-solve of each
      distribution network at that voltage with the rows on, followed by the transmission network's dispatch, whose
      cost is the $Q$ of Table~\ref{tab:appE_benchmark}. The consistency column is the change of $Q$ from the first
      phase to the pass." Table E.23 caption: "$Q$: gross operational cost, settlement excluded, after the
      consistency pass. Curtailment: day-weighted, over the horizon, from the first phase (before the pass)." The
      `[CONFIRM — W179]` is removed. (Source: W179a (v): `p515_s53_w116_benchmark_nrf.py` phases A/B/C,
      `pass_effect_eur` = Q(C) − Q(A), `arm_cost_source_by_start` = phase C, `curtailment_rows_are_phase_A`;
      n = 29 and 70 out of 36 × 24.)
    - E.5, l. 2661: "… at 0.68--0.70 of its tolerance in the last cycle." → "… in the last cycle (for
      evaluations~40 and~41 the split and the residual were not recorded)."
    - E.6, l. 2692: "The two evaluations continued past their certificates, 14 and 3, moved by a further 0.74$\tau$
      and 0.78--0.88$\tau$." → "The three evaluations continued past their certificates, 12, 14 and 3, moved by a
      further 0.004$\tau$, 0.74$\tau$ and 0.78--0.88$\tau$." (W163 N25: evaluation 12 `b_2a0ba8b2`, +17.67 € past
      k\* 173.) The `[CONFIRM — W179]` is removed.
    - E.6, l. 2699: "the three single-scenario reference continuations replayed bitwise to their certification
      cycles;" → "the three single-scenario reference continuations (the plan without storage, the reference unit
      and a larger plan not tabulated here) replayed bitwise through the stopping cycles of the residual-based runs
      they continue, cycles 132, 112 and 87;" (W101 G13/G19 via W163 N49; Addendum 53: x = 0 132, unit 112, C\* 87).

**Not changed, on purpose.** The 46-certificate count in E.6 (49 certified T2 cells − row 16 − the two twins; the
verifier still derives it from T2, where row 16 remains `certified` under its own rule). The discount-rate thresholds
(Table 15 carries them; E.21 bands are at 2 %) and the benchmark's 71.9 k€ reproducibility band (E.23 note) — both
accepted as L residues of the Determinacy sentence, which now says "single-scenario evaluation" and points the
multi-scenario pair to Table 13. The `% [AUTHOR]` note under the ageing table stays until the citations are in.

## B. Letter v7 (`response_to_reviewers_v7.tex`)

- **R1.4 [decision 5, R14-2].** Fuel taken out of the "introduction now states" clause; new closing sentence: "Fuel
  costs are not represented separately: conventional generation in the transmission network is priced at the
  wholesale energy price of the scenario (Section~3.3), so fuel-cost uncertainty enters only through the market-price
  scenarios." (The introduction, l. 381–382, covers the other sources exactly as listed.)
- **R2.2 [W179b R22-b].** "and the cycles of every evaluation in Appendix~E" → "and the cycles of every evaluation
  behind a reported comparison in Appendix~E".
- **R2.5 [not rated by W179b].** "bitwise replays of every gated evaluation (Section~4.7 and Appendix~E)" →
  "bitwise replays of every evaluation repeated from an earlier record (Section~4.7 and Appendix~E.6)".
- **Further changes, Appendix E bullet [W179b FC-2c].** "every recourse evaluation, with its stopping status, …"
  → "every single-scenario recourse evaluation those comparisons use, with its stopping status, …".
- Header comment dated v7. No pointer changes beyond "Appendix~E.6".

## C. Author's steps

1. Paste `main_48099ab_round6.tex` over `main.tex` and `response_to_reviewers_v7.tex` over
   `response_to_reviewers_draft.tex` in Overleaf; compile (expect 0 errors, numbering unchanged).
2. Pull the Overleaf clone on the Air; commit and push the repo from PyCharm (this round's files, plus the local
   handover commit `c627c7ce`, which is still unpushed; `43c72e06` is on origin and must be merged in).
3. Pull on the Studio and forward the W180 prompt (Addendum 79).
