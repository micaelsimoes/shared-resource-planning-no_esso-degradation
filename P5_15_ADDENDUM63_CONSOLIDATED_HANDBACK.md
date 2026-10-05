# P5.15 consolidated handback: the SRP1 re-settling campaign is complete

**Planner handoff report, 2026-10-05.** For the External Expert and the Author; self-contained. It consolidates
everything since Addendum 63 and collects every decision still open from earlier reports.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 58–63.
- **Specs:** stage spec v6 `96c23404` (34 cells, complete); extension spec v3 `84775dc4` (6 ageing arms +
  `pb_y2025_n5`, complete); banded-fit scorer `88cf7163`; Step 5 zero-solve rows `48e76c9f`.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; value = Q(0) − Q(x);
  F = Q + I; net = gross − salvage. τ = 4,539.07 €. Determinacy threshold max(3 × larger bar, 2τ) (Addendum 61). For
  an uncertified cell the bar is 3·max(|gap|, |slack|).
- **Detail reports:**
  - `P5_15_ADDENDUM62_CLAIM_POINT12_REPORT.md` (`9d2f458f`)
  - `P5_15_ADDENDUM62_H_CLAIM_STOP_NOTE.md` (`53cd9e57`, ruled in Addendum 63)
  - `P5_15_ADDENDUM63_JCG_CLAIM_POINTS_REPORT.md` (`bf84fd41`)
  - `P5_15_ADDENDUM63_L_AND_AGEING_REPORT.md` (`acedf45c`, with its Step 5 supplement `67645e89`)
- **Status:** the Mac is idle. Everything in the current order that needs no decision is done. The next stop in the
  order is the review before the Step 6 tables are frozen; Decisions 1–2 are what remains before it.

## Decisions needed

### For the author (machine time)

**1. One cell at flexibility multiplier m = 1.75 (≈ 1.5 h).** The ladder currently reads "pays at m = 2 (+46.3 k€,
3.77×); break-even at or below ≈ 1.75 (nominal 1.62), lower end open" (Addendum 63). One cell at 1.75 brackets the
crossing from above and turns the one-sided bound into an interval.
- **Recommendation: run it** if the flexibility ladder is a headline row.
- No plumbing is needed: it is a v6 cell like the H cells.

**2. The Step 5 row at soh_min 0.50 (≈ 1.5 h run, plus plumbing).** Of the brief's Step 5 list, only this row needs a
solve:

| row | status |
|---|---|
| discount, salvage, certification statistics | done, zero-solve |
| uncoordinated benchmark | done (Addendum 57) |
| α | done on the 2 × 2 (Addendum 44) |

The remaining cell is the C2_calfade unit at soh_min 0.50, which isolates the floor that the ageing row now shows
binding in 2035. `apply_model_variant` cannot change soh_min, so the run needs:
- a variant key built in new files;
- an Advisor design check;
- a bitwise gate (C2_calfade at 0.70 = `3f084f2f` through 172, the G28 precedent);
- a frozen spec.

**Recommendation: authorise it.** The expert's "≈ 10 Step 5 cells" reduces to this one.

### For the expert (reporting and rules; the machine does not wait on these)

**3. Which "slope" the 5 % break-even prediction refers to** (claim point 12 report, Decision 1). The verdict was
pre-registered on the energy coefficient b; the 4 h marginal slope b + c/4 is reported beside it. Both hold at the
midpoints:

| reading | relative difference at the midpoints | worst corner of the intervals |
|---|---|---|
| b | 0.59 % | 7.1 % |
| b + c/4 | 0.31 % | 3.7 % |

**Recommendation: b + c/4**, the quantity the break-even price is built from.

**4. Two certificates rest on a non-clean turning point** (JCG report Decision 1, extended by the certification
statistics):

| cell | non-clean turning point | what happened |
|---|---|---|
| `j_a11d7966` (J, 4 MWh) | 138, a TSO recovery | gave P̂ = 2; the cell then descended with growing steps (−208 €/cycle at k\*) |
| `d_4a82a64a` (D, 3 MWh) | 108 | — |

In both cases the non-clean cycle sits outside the certifying window, so v6's veto did not apply. **No verdict
depends on either:**
- **The J claims:** they are bound by other cells' bars; the J value − I at 4 MWh goes from 18.1× to 11.1×.
- **The break-even fit:** adding `d_4a82a64a` as a fourth interval (bar 70.7 k€ + 2τ) raises the worst-case
  break-even by 0.0273 × 79.8 k = 2.2 k€/MWh. The worst-case margin to cost becomes **61.3 k€/MWh** (from 63.5). This
  is Planner arithmetic from the scorer's committed weights.

**Recommendation:** keep both certificates, flag them, and show the uncertified form beside them. Record "a non-clean
cycle cannot be a turning point" as a post-revision rule question. It is the mirror of `d_c52e1670`, where a blip
blocked certification.

**5. The determinacy floor changed its first verdict.** `E:C2_calfade vs C3` (+9,329 €) is 0.71× its threshold:
within resolution. Under the superseded rule it was determinate. The Addendum 61 prediction "the floor changes no
verdict" therefore fails on one claim. **Recommendation:** report it as within resolution, with S1 restated to "within
resolution of C3 at a 0.70 floor".

**6. The ageing statements are restated at the 0.70 floor.**
- **ε_AE** over the resolvable arms is **1.04–1.85**, against ≈ 0.6 in the 0.50 era.
- **The expert's prediction** (Addendum 58 supplement: floor-binding arms show a smaller value change per unit of
  cycle life) is scorable only on C2_calfade. Its point estimate holds (ε_k 0.0331 < 0.0394), but the difference is
  **within resolution** (0.20×). On C2 and C4, where the floor never binds, ε_k rises determinately.

**Recommendation:** "consistent in direction, not resolved at this precision", with S5 restated.

**7. Dead-zone table entries.**
- **(a) `l_0ee93aca`:** it shows the dead-zone fingerprint, but two TSO recoveries reset the rule before the gap
  clause was reached. The scorer records it as "not gap-refused".
- **(b) `h_f9eae48f`, the m = 1.5 cell:** the F2 mechanism is only partially reproduced. TN conventional generation
  sits at Pmin there and also in the x = 0 control. The column that separates it from the control (the TSO's cheapest
  lever is the shared ESS, 0–9 €/pu against 6.8–16.3 k€/pu for a generator) was added after looking at the data.

**Recommendation:** (a) enter `l_0ee93aca` by its signature, with the cause stated. (b) Carry the TSO-lever column as
report-only and say that it was added post hoc.

## Blocked on the author

Decisions 1 and 2 (machine time). Nothing else; the Mac is idle until then.

## Changed (since Addendum 63)

| what | commits |
|---|---|
| I cell, J cells (`j_5f3cccb4` launched by the author from Terminal), summarize points #17, #20 | `232af066` `4b30ea00` `b8a6e689` `04db6a12` `8ab183cc` `289297dd` |
| `h_f9eae48f` interface prices (dead-zone table) | `241e299c` |
| C and G cells, summarize points #22, #26 | `dad59dca` `3105b60d` `3d9b8d94` `bf810183` `4ab435aa` `58f7209e` `28851414` `abfd4c88` |
| L cells, summarize point #38 | `f42e7299` … `48a447fa`, `b95ee706` |
| extension: six ageing arms + `pb_y2025_n5`, two summaries | `22c7c116` `23cee034` `afc6dbd7` `4c735dc3` `d7602e1a` `dbbf6ccf` `72d52c23` `2495b3c8` `184229bb` |
| Step 5 zero-solve rows | `48e76c9f` |
| CLAUDE.md: reporting-ruling stops do not idle the machine (Addendum 63) | `efa38fd7` |

**Configuration of the runs:**
- Every launch command was taken verbatim from its frozen spec.
- Every gate passed on every cell, apart from G27 (the D-cell certification prediction, recorded and not a stop) and
  one non-stopping G6 (`d_f759dd48`).
- Every gated cell replayed bitwise through its first residual pass.

**Machine-local setting** (not committed): `BASH_MAX_TIMEOUT_MS` = 14,400,000 lets attached runs exceed the tool's
2 h limit. A 2 h 2 min check survived it, and so did `l_2ab0ce2d` at 10,759 s.

## Found

### The campaign at a glance

- **Cells:** 42 (34 v6 + 7 extension + `d_c52e1670` from records). 32 certified (24 oscillatory, 8 monotone).
- **10 uncertified:**
  - 5 by the gap clause (the dual dead zone);
  - 2 by lapse resets;
  - 3 by the growth test.
- **Claims:** 60 complete; **50 determinate**, 10 within their bar or resolution. **No recorded sign prediction failed**;
  the one unresolved sign is H at m = 1.5 (ruled in Addendum 63).

### Results the manuscript carries (gross; margin over threshold in brackets)

| item | result |
|---|---|
| **Break-even fit (node 7)** | certified-only e\* 182,702 €/MWh; with the three uncertified points as intervals, **173,486–190,380**; margin to energy cost (253,878) **≥ 63.5 k€/MWh** (≥ 61.3 k if `d_4a82a64a` is also treated as an interval). Before settling: 182,231 |
| **Flexibility ladder, m = 2** | 1 MWh value − I **+46.3 k€** (3.77×); second MWh **+46.5 k€** (3.64×); 2 MWh value − I +92.8 k€ (7.26×) |
| **Flexibility ladder, m = 1.5** | −14.5 k€, within its bar; crossing at **m ≤ 1.75**, nominal 1.62 (Addendum 63) |
| **Energy ladder, m = 2** | 2→3 MWh +41.0 k€ (1.48×); 3→4 MWh +42.8 k€ (1.55×); 4→5 MWh +51.8 k€ (1.95×); value − I at 3/4/5 MWh +134 / +177 / +228 k€ |
| **Phase B certificates** | `pb_y2025_n5` re-run **+49.7 k€ (3.94×)**; the flag leaves the table, and all Phase B certificates are now of one strength |
| **Phase A unit claims (B)** | every D-cell value − I determinate (4.4–45.9×; uncertified cells 8.7–14.5×) |
| **C and G (x = 0 wins)** | every row, gross and net: C +102.1 / +93.9 k€ (8.1× / 7.0×); G +99.0 to +223.6 k€ gross, +85.1 to +113.1 k€ net (6.7–17.7×) |
| **L: the F2 certificate** | **holds**: all 12 neighbour differences positive (7 determinately worse, 5 within the bar), also net of salvage |
| **Ageing at soh_min 0.70** | no aged arm pays (value − I −31.6 to −73.7 k€, all determinate); without ageing, break-even within resolution (−4.1 k€). The floor binds in 2035 for C3_unit, C2_calfade (the baseline) and C3_midblock |
| **Discount rate** | value − I negative and determinate at 0 / 2 / 5 / 8 % (−40.8 / −64.4 / −92.7 / −114.6 k€); captured spread 59.0 € per MWh-cycle |
| **Salvage convention** | 0 sign changes in 60 claims; 1 verdict change (F2 incumbent vs `l_0ee93aca` becomes determinate net, in the certificate's favour) |
| **Uncoordinated benchmark** (Addendum 57, unchanged) | coordination saves **+90.9 M€ (13.9 %)**, determinate |

### Predictions against outcomes (since Addendum 62)

| prediction | outcome |
|---|---|
| Expert (A62): the two break-even fits agree on the slope within 5 % | **held** (0.59 % on b; 0.31 % on b + c/4) |
| Planner: break-even within ±6 k€/MWh of 182.2 k; margin > 60 k€/MWh | **held** |
| Planner (A61): the D cells certify under v6 | **failed** on 3 of 8 (recorded; Addendum 62) |
| H m = 1.5 value − I negative, 15–45 k€ | sign held, size missed low, determinacy missed (Addendum 63) |
| H m = 2 +35–60 k€; I second MWh +30–55 k€ | **held** / **held** |
| J 2→3 25–45 k€; 3→4 35–60 k€; 4→5 indeterminate | **held**; **held**; **missed** (determinate, 1.95×) |
| `j_5f3cccb4` dead zone (gap-refused, t_sum −8 to −10 k€) | **held** |
| C and G certify; G at k0 + 50–75 | **held** (k0 + 62–71) |
| C / G magnitudes (Advisor design review) | **missed low on all six rows**; signs and determinacy held. The miss lies in the prediction anchoring: the C cells' old margins (≈ 97.1 / 130.6 k€) were already below the ranges |
| G net ≥ 90 k€ | missed on 2 of 4 (88.3, 85.1 k€) |
| L small cells certify; s ∈ [−5, +10] k€; margins move ≤ 10 k€ | **held**, **held**, **held** |
| L dead-zone cells gap-refused | **3 of 4** (Decision 7a) |
| Planner G28: C2_calfade reproduces the settled unit bitwise through 172 | **held** |
| Expert (A58 supplement): floor-binding arms, smaller value change per unit cycle life | point estimate held; **within resolution** (Decision 6) |
| Expert (A61): the determinacy floor changes no verdict | **failed** on one claim (Decision 5) |
| Expert (A28): captured spread 55–60 €/MWh | **held** (59.0) |

### For the limitations and reproducibility paragraphs

- **Solver events.** Every non-clean cycle after the first residual pass, across all 42 cells, is a **TSO recovery
  ending "Acceptable"** (21 cycles in 10 cells; 2,100–6,100 × complementarity). TSO 2035 Spring is the recurring
  block. These events veto, reset or block certification, and they also supplied the turning points in Decision 4.
- **Acceptable on primary attempts.** DSO7 2025 Winter and DSO5 2035 Winter end Acceptable on their primary attempts
  (1.2–4.2 ×) and count as clean under the 10 × rule.
- **The dual dead zone** appears with large storage and a high flexibility price. It carries one fingerprint across
  all five 2030 cells and the F2 incumbent: a priced gap of −9.2 to −9.4 k€, split node 5 55 %, node 9 31 %, node 7
  14 %, with pf_primal flat at 0.69.
- **Six certificates stop at ≥ 0.95 τ.** The rule bounds each cell's stopping error at about τ, not well inside it.
  This is consistent with the ≤ 0.9 τ post-certification movement measured earlier.

## Not confirmed

- **What causes the TSO recoveries** (2035 Spring, 2030 Autumn and Winter, 2035 Summer). The logs are preserved, but
  the cause is not investigated; it belongs to the post-revision cleanup.
- **Whether `j_a11d7966` and `d_4a82a64a` would have stayed near their certified values.** Neither was continued past
  k\*. The 61.3 k€/MWh sensitivity is linear arithmetic, not a fit that was run.
- **Why the 0.70 floor raises ε_AE.** The late-life-tail reading is the Planner's, not a measured decomposition.
- **Net-of-salvage verdicts for 52 of the 60 claims** are computed with the committed scorer. Their method was
  validated only against the 8 recorded G rows.
- **The soh_min plumbing** (Decision 2) has not been designed.
