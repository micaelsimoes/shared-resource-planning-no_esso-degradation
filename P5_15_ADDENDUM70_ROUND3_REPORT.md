# P5.15 Addendum 70 order — W174a, W174b, W175 (round 3 checks, zero-solve)

**Authority:** Addendum 70 and `STEP6_ROUND3_CORRECTIONS.md` §D.

**Manuscript:** Overleaf clone at HEAD `260bd83`; main.tex is 2,209 lines, sha256 `cb237b2d…`.

- Rounds 2 and 3 were taken as pasted from what the Overleaf state shows; the author's explicit confirmation was not
  awaited.
- The markers checked were the F2 "seventeen" sentence, the A.6 "not constructed" sentence, Δ₀ = 4, and §3.5 / §3.6
  present.
- W174b then found every round-2 and round-3 block present.

**Runs:** zero solves (guards armed and verified at 0, pickle blocked). Nothing else ran.

## Decisions needed

1. **Two passages were deleted while pasting round 3, and no correction named either (author: restore).**
   - **NEW-1, Appendix A, l. 1640 (M).** The clause "and the acceleration is switched off on any cycle in which every
     channel passes and, in the certifying regime," is gone.
     - The appendix no longer says when Anderson acceleration is off. The code turns it off whenever every channel
       passes (`admm_anderson_acceleration.py:477`) and forces it off from k₀ + 1 (`p515_s53_w118_resettle_hooks.py:526`).
     - "from the cycle after k₀ on" now attaches to clearing the memory, which the code does at any cycle.
   - **NEW-2, Algorithm 2 (L).** The line `$k \gets k + 1$\;` is gone, so the loop never increments.
   - **Recommendation:** restore both verbatim. After that, the CONFIRM comments at l. 1647 and l. 1706 can be deleted.

2. **§2.2.7, l. 647 — the master search's acceptance rule (expert; the W174 prediction turns on this row).**
   - The sentence reads: "the master search of Algorithm 1 accepts a poll point as an improvement only by a
     determinate margin". It follows "this threshold" (max{3b, 2τ}).
   - Algorithm 1 (l. 434) and l. 449 correctly give the rule as run: F(inc) − F(x) > max(bar_x + bar_inc, σ_Q)
     (`p515_s47_phase_b_record.py:519`).
   - W174b rates the row M, on the ground that Algorithm 1 states the rule two paragraphs earlier. The Planner rates it
     H, because placed after "this threshold" the sentence reads as the §2.2.7 rule, which is exactly the claim W171b
     T1-4 found false.
   - **Recommendation:** "… accepts a poll point as an improvement only beyond the search resolution of Algorithm 1;
     the threshold above applies to the comparisons reported". The expert scores the prediction.

3. **l. 449: "the runs reported here stopped at their evaluation budgets" is false for all three searches (expert; M).**
   - s47 stopped on a failed unit poll, after 14 of 20 evaluations.
   - s51 stopped for review at the completion cap (61 > 30), after 20 of 20.
   - s53 stopped on a failed unit poll, after 7 of 60.
   - The same paragraph's "every rounded poll direction was inadmissible at every poll" (NEW-4, L) has one exception:
     s51's poll 4 had one admissible direction. It was never evaluated, because the poll was refused at the cap.
   - **Recommendation:** "s47 and s53 stopped when a unit poll failed; s51 stopped for review when its completion set
     exceeded the cap", and "at every evaluated poll".

4. **§3.6, l. 1108: "Every recourse evaluation ran … under one frozen configuration" (expert; M).**
   - This repeats in §3.6 the claim that W171b T1-10 removed from §2.
   - The three search campaigns ran at older code heads (`353e094b`, `c39c4836`, `e1f98be8`) without the tight tail.
   - **Recommendation:** scope the sentence to the reported certified evaluations, and say the searches ran under the
     earlier configuration, with their incumbents re-settled under this one.
   - The Appendix A sentence "the tail is … enabled in every campaign" (l. 1644, NEW-7) needs the same scoping.

5. **The benchmark paragraph needs two wording changes, so the W175 prediction fails on count (expert).** The
   predicted change is one of the two.
   - **Predicted:** state that the passive DNs set their curtailment by a 1 €/MWh minimum-curtailment term, and that
     every arrangement is costed at the common evaluation Q (settlement excluded, that term set to 0).
   - **Not predicted:** every reported arm value comes after the one sequential consistency pass of Addendum 49 (each
     DSO re-solved at the TN's actual interface voltage, then the TSO). The pass moved the passive starts by −44.0 /
     −69.8 / −69.8 k€ and the price-taker by +234 €. The 90.9 M€ (13.9 %) claim is unaffected.
   - **Recommendation:** W175's two-sentence replacement (`P5_15_W175_CONFIRM_3_5_3_6.md`).

6. **Wrong cross-references, l. 756 and l. 866 (author; L).**
   - Both send the reader to §3.6 (`sec:case_settings`) for η, SoC^Min/Max/0, ε^Cl, c^Cl, ε^C, c^σ and ε^E. Those
     values are printed in §3.5 (`sec:case_ess_params`).
   - Round 3 §C mapped every "Section 3.5" to `sec:case_settings`; only the α reference (l. 907) is right.
   - **Recommendation:** point both to `sec:case_ess_params`.

7. **The chemistry sentence has no source (author).**
   - §3.5 says "utility-scale lithium iron phosphate". No input file names a chemistry: the cost file says only
     "Li-Ion Battery cabinet" (NREL).
   - The only record is the spec v18 citation block (EVE MB31 / Hithium LFP datasheets), marked "author to confirm".
   - The letter (l. 88–89) also says the cycling calibrations come "from LFP datasheets (Section 3.4)". Only C4 has a
     named datasheet; the baseline's 10,000 cycles has no recorded source; and the calibrations are now in §3.5.
   - **Recommendation:** the author names the source, or the sentence says "lithium-ion" and cites the NREL ATB
     category.

8. **The remaining L rows (author, at the final pass).** The full table is in `P5_15_W174B_REAUDIT.md`.
   - **Holds:** they start one cycle after the stopping cycle or the first pass, not at it (T2-1, A3-3, A4-3).
   - **Window bounds:**
     - the oscillatory window may start at k₀ (T2-4);
     - the monotone window needs lo ≥ k₀ + 3 (T2-5);
     - the gap is tested at the certifying cycle only (T2-5).
   - **Notation:**
     - T^Cal and Y are indexed by the cohort's investment year (T2-11, T3-7, T3-8);
     - for the storage agent, "ten times" applies to its own tolerances (T2-9);
     - the rounding of the poll direction and the s53 snap tie-break are not stated (T1-3).
   - **Wording damaged in the paste:**
     - a lost "and" at l. 841 (NEW-10);
     - "updated with residual balancing" misattached at l. 1497 (NEW-5);
     - "unscaled in every agent" at l. 1515 contradicts κ^E (NEW-6);
     - "the settlement are activated here" at l. 1678: the weight goes 0 → 1 in every instance (NEW-9).
   - **Elsewhere:**
     - "end of the representative year" at l. 679: the code applies SoH at the end of the block;
     - "ended at an optimal status" at l. 1636, beside l. 1644, which counts acceptable exits as success.
   - **Comments:** the CONFIRM comments at l. 444, 452, 1517, 1579, 1628 and 1717 can be deleted.

## Blocked on the author

- **The PDF the reviewers read,** for the figure numbering in R1.5 and the "Figure 15" reference. Still open from
  round 1.

## Changed

| task | script | outputs / report |
|---|---|---|
| order | — | `a8eb9b69` TASKS.md |
| W174a | `bb3830fb` `p515_s53_w174_manuscript_number_check.py` | `b4f93d7b` `data/SRP1/Results/P515S53/w174_manuscript_check/overleaf_260bd83/` |
| W174b | `p515_s53_w174b_reaudit_checks.py` | `02aa735f` `P5_15_W174B_REAUDIT.md`, `…/w174b_reaudit/` |
| W175 | `p515_s53_w175_confirm_3_5_3_6.py` | `f7d165b7` `P5_15_W175_CONFIRM_3_5_3_6.md`, `…/w175_confirm/` |

- No production file, spec, case file, frozen artifact or manuscript file changed.
- The production code equals all 46 table campaigns' `git_head_at_run` (W174b, with a positive control).

## Found — predictions against outcomes

| prediction (expert, Addendum 70) | outcome |
|---|---|
| W174: every §3.5–3.6 value matches its source | **held** — 141 values: 129 match, 12 approximate (printed rounding of the source, e.g. k 35,851 for 35,851.36), 0 MISMATCH. 20 named settings: 19 match, 1 without a source (the chemistry, decision 7) |
| W174: no H row remains in §2 or Appendix A | **held on W174b's rating** (H 0, M 3, L 19). **Fails on the Planner's rating** (1 H, l. 647, decision 2) |
| W175: the benchmark paragraph needs one wording change at most | **failed on count** — two (decision 5); the predicted one is among them |

**W174a, number check.**
- MISMATCH: none in main.tex, the letter, the cover letter or the highlights.
- main.tex: 380 match, 12 approximate, 559 submitted-version figures (§4–§5 and Appendix E, still to be rewritten).
- Letter: 97 match, 62 numbers in quotations verified, plus "single machine" (the author's attestation).
- Declarations: 128 of the 145 v2 declarations were carried, 17 replaced, and 92 added.
- Resolved from W171a: letter findings G1 (the floor year), G2, G5, G6 and G7; main.tex findings M1–M4.
- Still open: G3/G4 (figure numbering, waiting on the PDF) and G8 (the references status note, by design).
- Literal "Section 4.x" references still point at the submitted §4.

**W174a, §3.5–3.6 values matched to source.**
- Solver versions and thread caps (W159).
- Per-agent IPOPT options and retry policy.
- σ, S^ref, R^I 200/100/150 MVA, κ^E, ρ₀, the balancing rule, the Boyd tolerances, AA, the production exit and caps,
  α, the voltage weight, δ_R, 𝒱 and P_max.
- Δ₀ = 4, the budgets 20/20/60 and the completion cap 30.
- The ageing table: all seven arms as run. C4 ran as (10,000, 0.80); the datasheet gives (8,000, 1.00). Both have
  N·DoD = 8,000, the only product k uses.

**W174b, re-audit.**
- All 10 W171b H rows and W172's A6-1 are resolved.
- Every round-2 and round-3 block is present. The two deletions in decision 1 are the paste's.
- The C.3 caption gives 200/100/150 MVA.

**W175, CONFIRM points.**
- Mid-block SoH point: confirmed (`shared_energy_storage_data.py:625-626`; arm C3_midblock only; it sets E^Av).
- No-ageing arm: confirmed in effect. The code replaces the cycling loss by D = 0 and sets φ = 1, with the floor
  inactive; the table's ∞ is the correct limit.
- Benchmark definitions: confirmed (passive, price-taker and TSO arm; no-reverse-flow rule `pg_adn ≥ 0`; best of
  three starts; common-Q gate PASS_BITWISE).

## Not confirmed

- **Versions and single-thread setting:** they cover the v6 campaigns (W159). The earlier search campaigns are not
  covered.
- **The C4 datasheet values:** taken from the spec v18 record; the datasheet itself was not opened.
- **The old v2 certificates under the clean veto:** W174b relied on W142's committed replay for them.
- **Equation and algorithm numbering:** no compiled PDF exists, so it was not checked.
