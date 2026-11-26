# P5.15 stop at the H claim point: the m = 1.5 unit sits within its bar of break-even

**Planner interim note, 2026-10-02.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 62, and CLAUDE.md "Stopping conditions": a manuscript claim
  left indeterminate by the uncertified form is a stop.
- **Spec:** stage spec v6 `96c23404`.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; value − I = Q(x = 0) −
  Q(unit) − I; τ = 4,539.07 €.
- **Status:** nothing is running. Group 2 has stopped before `i_5a6a88b4`.

## Decisions needed

**1. How to report `H:m1.5:value_minus_I` (expert).** The 1 MWh node-7 unit (0.25 MVA, I = 317,957 €) at flexibility
multiplier m = 1.5 has **value − I = −14,450 € gross (−12,520 € Q_cc)**. The other cell, `h_f9eae48f`, is uncertified,
so the claim is scored on the uncertified form: bar = 3·max(|gap| 2,482, |slack| 6,250) = **18,751 €**. That puts the
claim at **0.77× its bar: indeterminate**.

The cell's *objective* has settled. The rule would have certified it on every cycle from 136 to the cap (59 in a row),
with range/τ falling from 0.92 to 0.06. Over the last W = 27 cycles its range is 268 € and its drift −10 €/cycle.
**Only the gap clause refused it.** The priced interface gap is stationary at **−2,450 to −2,550 €** from cycle 138 to
the cap (bound τ/2 = 2,270 €). All three DSO interfaces share the same sign (node 5 ≈ 54 %, node 9 ≈ 31 %, node 7 ≈ 15 %),
while the MW gap keeps shrinking (0.059 → 0.042 MW) and pf_primal sits at 0.027. This is the F2 signature, a
set-valued interface dual, at about a quarter of F2's magnitude. The bar is dominated by the *slack*: the distance the
cell moved from its old certificate (−6,250 €). That distance does not measure the uncertainty of an objective that has
settled.

| option | what | H:m1.5 verdict | cost |
|---|---|---|---|
| **(a) uncertified form as frozen** | report "value − I = −14.5 k€ (−12.5 k€ Q_cc), within its 18.8 k€ bar; sign not resolved" | indeterminate | none |
| (b) a gap-refused form for cells whose objective settled | bar = max(3·max(band_ref, \|gap\|), 2τ) = max(3 × 4,131, 9,078) = 12,393 € | determinate at **1.17×** gross, **1.01×** Q_cc | a scoring-rule change (yours); a rule written to rescue a claim that passes by 1 % on the second convention |
| (c) restate the H claim as a crossing | m = 2 is determinate (+46,285 €, 3.77×). Interpolating linearly in m through both bars, the unit breaks even at **m\* ≈ 1.62**, and at **m\* ≤ 1.75** at worst | the ladder claim becomes "the 1 MWh unit pays at m = 2 and breaks even at a multiplier of at most ≈ 1.75" | no run; linearity in m is an assumption (two points) |

**Recommendation: (a) together with (c).** The claim is honest as it stands. The −14.5 k€ is 4.5 % of I, which is
near break-even at any resolution. (b) would pass by a hair in Q_cc and invite the charge of rule-shopping. A
continuation of `h_f9eae48f` would not help: the gap is stationary, as on F2.

**2. Resume the queue while Decision 1 is open (author).** The next claim points (I `5a6a88b4`; J `f3aa335e`,
`a11d7966`, `5f3cccb4`) do not depend on H:m1.5. **Recommendation:** resume now (≈ 5–6 h to the J point). The cost
of not resuming is an idle machine until the ruling.

## Blocked on the author

Decision 2: machine time continues only on your word.

## Changed

| what | commits |
|---|---|
| H cells 9–12 | `40568e6a`, `3740b476`, `72d74291`, `2c75fe40` |
| claim point summarize after `h_74eda68d` | `0fc489c6` |

Every cell: G19 bitwise through k0, every gate PASS (G6 included), zero non-clean cycles after k0, ≈ 2.5 min between
cells.

## Found

| cell | m | unit | v6 outcome | k\* / cap | range/τ | s (€) | t_sum (€) |
|---|---|---|---|---|---|---|---|
| `h_aa8a76d7` | 1.5 | x = 0 | certified | 153 | 0.91 | +10,375 | −552 |
| `h_f9eae48f` | 1.5 | n7 0.25 / 1 | **uncertified, gap clause** | cap 194 | 0.06 (last W) | −6,250 | −2,482 |
| `h_50dea31c` | 2.0 | x = 0 | certified | 146 | 0.48 | −1,646 | +722 |
| `h_74eda68d` | 2.0 | n7 0.25 / 1 | certified | 150 | 0.90 | +465 | −1,175 |

### Predictions against outcomes

| prediction (Planner, Addendum 58 design review) | outcome |
|---|---|
| H m1.5: value − I negative, \|margin\| 15–45 k€ | **failed**: −14.45 k€, negative but below the range and indeterminate (0.77× bar) |
| H m2: value − I positive, 35–60 k€ | **held**: +46.3 k€, determinate at 3.77× |
| Group 2: cells with P ≤ 1.0 MVA certify, gap ≤ τ/2 | **failed** on `h_f9eae48f` (gap 2,482 > 2,270) |
| Group 2: settled slack s ∈ [+8, +30] k€ (m 1.5), [−5, +20] k€ (m 2) | m 1.5: +10.4 k held, −6.3 k failed; m 2: −1.6 k and +0.5 k held |
| gate-ability (Advisor) | held on all four |

## Not confirmed

- That the stationary gap on `h_f9eae48f` is a set-valued dual, as on F2. The per-period interface duals are
  captured (`interface_duals_per_cycle.jsonl`, committed) but have not been examined. The F2 mechanism
  (storage drives TN conventional to Pmin) is unlikely to apply at 0.25 MVA; the flexibility price is the candidate
  here.
- Option (c)'s linearity in m: two points cannot test it.
