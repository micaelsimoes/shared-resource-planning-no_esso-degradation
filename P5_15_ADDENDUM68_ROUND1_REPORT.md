# P5.15 Addendum 68 order — W168–W171 (round 1 checks, zero-solve)

**Authority:** Addendum 68 and `STEP6_ROUND1_CORRECTIONS.md` §C.
**Manuscript:** Overleaf clone at HEAD `42794d4`. That commit holds the §2 corrections: correction A.1 is at
main.tex l. 654, and W171b found every A.1–A.8 block present. The author's explicit confirmation was not awaited;
the Overleaf state was taken as the evidence.
**Objective convention:** Q = `gross_operational_cost`, settlement excluded.
**Runs:** zero solves in every task (guards armed and verified at 0, pickle blocked). Nothing else ran.

## Decisions needed

1. **Letter R3.6 (l. 260) contradicts T8, and so does the ruling it came from (expert).**
   - The letter says the floor binds "(within the horizon it never does)". That wording is from Addendum 68 Decision 5
     and correction B.5.
   - T8's `floor_year_070` is **2035** for C2_calfade (the baseline), C3_unit and C3_midblock, and "never" only for
     C2 and C4. Figure 3 shows the same.
   - **Recommendation:** "the floor binds in 2035 under the baseline, C3_unit and C3_midblock calibrations, and not
     within the horizon under C2 and C4". Correct the addendum's sentence by erratum.
   - Cost: one sentence.

2. **§2.2.7, l. 639: "The rule above, with its holds, produced every single-scenario evaluation reported in this paper"
   (expert).**
   - W171b found the cells were *decided* under several rule versions: v1 for both references, v2 for nine cells,
     v4 for two, v5 for one.
   - W142 (`w142_v6_from_records.json`, Addendum 61) had already re-decided every pre-v6 record under v6, and every
     certificate certifies at its original cycle: x = 0 at 181, the unit at 172, five Phase B cells (`pb_y2025_n5`
     stays excluded under Addenda 58/59), both year-ladder cells, `b_0dd237f0`, `b_2a0ba8b2` and `b_4649234b`.
   - W168 reproduces the x = 0 case: k\*' = 181 and ΔQ = 0. So `ref:bd504ecf` needs no new replay.
   - What does not hold is "with its holds". In the evaluations continued from an earlier run, the holds began at
     that run's stopping cycle, not at k₀ + 1. For x = 0 that is cycle 133 against 124. The held state was in force
     from 124 anyway (AA off, tail passing, ρ frozen since cycle 42), but holding from 124 was not shown to be
     bitwise neutral.
   - **Recommendation:** "Every certificate reported in this paper was decided, or re-decided from its committed
     record, under this rule, and certifies at the cycle reported; in evaluations continued from an earlier run the
     holds began at that run's stopping cycle."
   - Cost: one sentence.

3. **§2.2.7 regressed from the accepted text in two places (expert; impact H).**
   - **P̂, l. 624–625:** the text says P̂ is "measured between the first and third turning points". The code uses the
     three most recent, T[−1] − T[−3] (`settling_criterion_v2.py:166`, inherited by v6). Under the text's P̂, 19 of
     46 decisions would change.
   - **Settling slack, l. 638 and 643:** the text has it as the movement since the residual pass. The code uses
     |Q(end) − Q(N_old)|, with N_old the earlier certificate cycle. Under the text's definition, 7 of 23
     uncertified-form bars would change, with no verdict changing.
   - paragraphs_v5 (Addendum 67) had both right.
   - **Recommendation:** restore the paragraphs_v5 definitions verbatim.

4. **§2.1 / Algorithm 1 describe a search other than the one run — five H rows (expert).** Source: W171b, rows T1-1
   to T1-5.
   - **T1-1, recourse:** the text writes it as separable per scenario. Each block solves all scenarios jointly, with
     a shared storage schedule and a shared TSO interface. §2.3 already says so, so §2.1 contradicts §2.3.
   - **T1-2, plan:** the text has x = (S, E) per node and year, n = 18. The plan as run has 7 variables: one (P, E)
     per node plus one investment year common to all nodes. Staging was never evaluated.
   - **T1-3, poll:**
     - The text has 2n directions with snapping, and completion "if |P| < n + 1".
     - Searches s47/s51 used n + 1 directions, rejected infeasible points rather than snapping, and added the
       completion set at every unit poll, capped at 30 (s51 stopped at 61 for review).
     - s53 used 2n directions with snapping, but kept the poll size at 1 after a success.
   - **T1-4, acceptance:**
     - The text says candidates are accepted by the §2.2.7 rule with margin max{3b, 2τ}.
     - Search evaluations stopped at the production exit (ten consecutive passing cycles, cap 500) and were accepted
       on F(inc) − F(x) > max(bar_x + bar_inc, σ_Q), with σ_Q = 18,449.66 €.
     - The §2.2.7 rule was applied afterwards, to the re-settled cells.
   - **T1-5, termination:** the text claims a positive spanning set. The F2 certificate records itself as "NOT a
     positive-spanning-set certificate, NOT the full-box certificate" (rank 5): 44 of its 61 feasible box neighbours
     were never evaluated. The x = 0 certificate covers all 14 feasible neighbours, which is the full box.
   - **Recommendation:**
     - Algorithm 1 describes the search as run, naming the two variants.
     - The text says the reported comparisons rest on re-settled, certified cells.
     - The F2 plan is described as better than all 12 of its evaluated neighbours (7 determinate, 5 within the
       bar), not as a mesh-local optimum.
     - This is a claim-level change in the methods text, so it is the expert's call.

5. **§2.3 — two H rows, so the expert's W171 prediction failed (expert).**
   - **T3-1, l. 656:** the text puts "charging, discharging and reactive power" in consensus. The storage channel
     couples only the net active power P^Net and the reactive power Q^Net. The agent's own charge/discharge split,
     which drives ageing, is not in consensus with the networks' split.
   - **T3-2, l. 866–888 and 897:** the text defines P̄ and the deviations over operation scenarios weighted ω_o. The
     code runs them over all market × operation pairs, weighted ω_m ω_o, and wires them whenever there is more than
     one pair. The sentence "vanish at a single operation scenario" (l. 897) is wrong.
   - **Recommendation:** fix both.
   - This is a wording prediction, not a margin or sign prediction, so no fallback is needed (CLAUDE.md stopping
     conditions).

6. **W169 — how to score the slack prediction, and the l. 747 / l. 857 sentences (expert).**
   - **Where the slacks sit:** every inactive slack sits on IPOPT's relaxed lower bound, −10⁻⁸ p.u. per variable,
     with `bound_relax_factor` 10⁻⁸ and original bounds not honoured (as in S48). That gives −2.0 × 10⁻⁶ MWh per
     closure pair and −2.0 × 10⁻⁸ MW per σ element.
   - **σ:** at its lower bound at all 46 scored certified points. |σ|/S ≤ 8 × 10⁻⁸. The prediction **held**.
   - **Closure, signed reading (s⁺ + s⁻ ≤ 10⁻⁶ E^Av):**
     - Holds at every current certificate; the largest signed value is −4.5 × 10⁻⁷ E^Av.
     - It fails at one point: the superseded `pb_y2025_n5` certificate (T5, excluded under Addenda 58/59). There,
       at a block recovered after maxIterations (TSO 2035 Spring, k\* 167), the slack is +1.76 × 10⁻⁶ MWh,
       or 5 × 10⁻⁶ E^Av.
   - **Closure, absolute reading:** fails at 35 of 45. The cause is the bound residue alone: a 10⁻⁶ relative
     threshold is tighter than the solver's bound relaxation for any unit with E^Av below 2 MWh.
   - **Coverage:** 45 of 51 certified points are scorable.
     - Five are x = 0 cells, where the slack is bounded [0, 0] and every block value is exactly 0.
     - `d_c52e1670` has no record at its v6 certificate (150), only at cycle 198.
   - **Recommendation:**
     - Score the prediction "at the lower bound to solver tolerance". On that reading it **held** at every current
       certificate.
     - l. 747: "The closure slack was at its lower bound, to solver tolerance, at every certificate used in this
       paper."
     - l. 857: "The slack pair σ was at its lower bound at every certified point (|σ|/S ≤ 8 × 10⁻⁸)." Then state
       that the post-solve min(P^Ch, P^Dch)/S check runs after every ESSO solve and is **recorded, not enforced**:
       it is above zero in 28 of 64 evaluations, at most 3.6 × 10⁻⁵ (e_c2).
     - Accept the scoped coverage for `d_c52e1670`.

7. **§4.5 — what the 0.933 prediction was (expert).** Source: W170.
   - It is W52's `R_r2` for the realized prefix draw [1, 2, 3] (`selection_3x3.json`, `7550f441`).
   - It is price-only: the daily 4 h market spread, discount-weighted **averages** of the 3 × 3 horizon over the SRP1
     horizon (90.53 / 97.02).
   - It therefore folds in both the scenario-set effect and the horizon effect.
   - The measured R is a ratio of discounted **totals**. On totals, the same model gives **0.915**.
   - Both values lie inside the measured [0.909, 0.934].
   - W52's headline set [2, 3, 4] × [4, 5, 1] (0.939) was not run.
   - **Recommendation:** cite 0.933 as recorded, as the price-spread prediction for the draw that was run. State that
     it compares average spreads across two horizons, give 0.915 on the totals basis beside it, and do not replace
     the recorded figure after the run.

8. **Letter, smaller points (author/expert).** Source: W171a.
   - **R3.5, l. 251:** "the sentence of the submitted version announcing 0.5 % and 2 % cases" is not in
     `manuscript_submitted/main.tex`. It is first-reply text, still at Overleaf main.tex l. 1053. Reword it to "of
     the earlier revision", or drop it.
   - **Correction B.6 is missing:** the qualifier on the multi-scenario horizon is absent.
   - **Decision 3:** the cost correction (÷4 h, not ÷5) is not named in "Further changes".
   - **R1.5:** the Changes line still reads "Figure~2". In the submitted source, the reviewer's "Figure 2" is the
     network diagram and "Figure 15 in Section 4.4" matches no figure. The author should confirm against the PDF the
     reviewers read.
   - **Stray character:** a `"` at l. 77.
   - **Quotations:** 29 of 31 are verbatim, and all 62 numbers in them match. Two differ in form only: added quote
     marks in R1.2(iii), and h(.) for h(⋅) in R3.3.

9. **Housekeeping (author).**
   - **Draft l. 496 writes C4 k = 22,430;** the record is 22,429.39 and T8 reads 22,429. The map has the same.
   - **Missing sections:** §2 cites §3.5 (three times) and §4.7, which do not exist yet.
   - **Recovery options:** Addendum 68's §3.5 list gives recovery `acceptable_tol` 10⁻⁴ / `acceptable_iter` 1 for the
     networks. These are set for the **TSO only** (`case9_params.json`).
   - **TN generation cost** in the code is controllable output priced at the scenario market price
     (`model_construction_helpers.py:2001-2008`). §3 should say so.
   - **σ evidence:** the per-element σ capture files are uncommitted but hash-recorded in W169's input manifest,
     which satisfies the "commit or hash-record" rule. **Recommendation:** leave them hash-recorded.

## Blocked on the author

- **The PDF the reviewers read,** for the figure numbering in R1.5 and in the "Figure 15" reference (decision 8).

## Changed

| task | script | outputs / report |
|---|---|---|
| order | — | `71f8d2aa` TASKS.md |
| W168 | `ff42f1e7` `p515_s53_w168_x0_v6_replay.py` | `96037a16` `data/SRP1/Results/P515S53/w168_x0_v6_replay/` |
| W169 | `2412edd4` `p515_s53_w169_slack_inventory.py` | `73dbfa63` `data/SRP1/Results/P515S53/w169_slack_inventory/` |
| W170 | `p515_s53_w170_r_prediction_check.py` | `f45b0b17` `P5_15_W170_R_PREDICTION_PROVENANCE.md`, `…/w170_r_prediction/` |
| W171a | `a379b481` `p515_s53_w171_manuscript_number_check.py` | `5cb84658` `…/w171_manuscript_check/overleaf_42794d4/` |
| W171b | `p515_s53_w171b_section2_record_checks.py` | `6fbd0dcf` `P5_15_W171_SECTION2_AUDIT.md`, `…/w171b_section2_audit/` |

- No production file, spec, frozen artifact or manuscript file changed; the diff adds new files only.
- The frozen JSON `590088fe` is unchanged.
- The W171a commit message says 34 guards; the run verified 35, and its JSON is correct.
- W171b deleted three outputs of its own first trial before re-running into the same directory. Those files were
  new, uncommitted and cited nowhere.

## Found — predictions against outcomes

| prediction (expert, Addendum 68) | outcome |
|---|---|
| W168: certifies under v6 at k ∈ [174, 195] | **held** — k\*' = 181 |
| W168: window range ≤ τ | **held** — range/τ 0.927 (window [150, 181], n_w 32) |
| W168: value within 0.93 τ of the tabulated one | **held** — ΔQ = 0 (same cycle; terminal-step bar 200.83 €) |
| W169: σ ≤ 10⁻⁶ relative at every certified point | **held** — 46/46, at the relaxed lower bound |
| W169: closure ≤ 10⁻⁶ relative at every certified point | **held at every current certificate** on the signed reading, 44/45 scorable. The failure is the superseded `pb_y2025_n5` (+5 × 10⁻⁶ E^Av). The absolute reading fails 35/45 on bound residue alone (decision 6) |
| W171: no H-impact row remains in §2.3 | **failed** — 2 H rows (T3-1, T3-2). §2.1 has 5 H rows and §2.2 has 3, not predicted |

Further results:
- **W168:**
  - Two v6 inputs were not captured in the W101 per-cycle record, t_sum_k and all_clean_k. Both were derived from
    the run's own artifacts with committed readers: t_sum matches production's terminal value to 1.06 × 10⁻⁷, and
    8,688 network and 543 ESSO exits are all Optimal on their primary attempt.
  - The rule read t_sum only at cycle 181, where |t_sum| = 142.5 € against τ/2 = 2,269.5 €.
  - The same result was already in W142's records; W168 reproduces it.
- **W171a:**
  - **main.tex:** 146 match; 3 MISMATCH (the §2.1 poll and P̂ rows above); 2 with no counterpart (l. 1049,
    60/80 %, removal ruled by Decision 6, not yet made); 610 submitted-version figures in §§3–5.
  - **Letter:** 89 match, 2 MISMATCH (R3.5 0.5 % / 2 %), and 62 numbers in quotations verified.
  - Of the 31 parameters in Addendum 68's §3.5 list, 30 match code, spec or case file. The exception is the recovery
    options (decision 9).
- **W171b:**
  - A.1–A.8 are all present.
  - The consistent statements are 25 in §2.1, 33 in §2.2 and 46 in §2.3.
  - The v6 replay reproduces 46/46 committed decisions, and the recomputed bars match the frozen tables 23/23.
  - Production code is unchanged against all 46 table campaigns. The three master-search campaigns ran on older
    production code.
- **W169:** the variables are `slack_shared_es_soc_final_up/down` (network.py:407-408; bound [0, 0.05 E + 10⁻⁵] at
  model_construction_helpers.py:1166; penalty 10³ × baseMVA inside Q) and `slack_es_pnet_up/down`
  (shared_energy_storage_data.py:460-461; penalty 10³). The min(pch, pdch) detector is in the production ESSO path
  (:1378) and recorded per solve.

## Not confirmed

- **Holding from k₀ + 1:** whether holding from 124 would have reproduced the x = 0 trajectory bitwise over cycles
  124–132. That would need a run.
- **The v2 cells:** W171b's records have no per-block exit classes for them. W142's floor replay recomputed the
  classes from the logs, and that is the basis for decision 2.
- **Per-element σ** at nodes with no capacity, and in years before a 2030 investment. Only the committed sum is
  available, read under the relaxed-bound floor assumption; the exception is the three T11 workbooks.
- **`d_c52e1670`:** the closure at its v6 certificate (150) is not recorded.
- **Figure numbering in the reviewers' PDF:** the PDF is not in the repository.
- **Mapping the spread model to V:** efficiency, ageing, network limits, and bus-7 price against market price are
  untested anywhere in the record (W170).
