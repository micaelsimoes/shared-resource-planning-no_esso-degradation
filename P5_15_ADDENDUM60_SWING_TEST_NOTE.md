# P5.15 Addendum 60 follow-up: cell 3 certifies under v5; the first break-even-fit cell is stopped by a blip in the swing test

**Planner interim note, 2026-09-30.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 58–60 (with the Addendum 58 and 59 supplements).
- **Specs:** stage spec v5 `ab32ffc9` (criterion v5, predecessor v4 `e11fbc89`); extension spec `6e5f60f7`.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; τ = 4,539.07 €.
- **Status:** nothing is running. The campaign stopped because a recorded prediction failed (Addendum 58: "no stop
  between claims unless a prediction fails").

## Decisions needed

**1. The "swings not growing" test is defeated by a sub-τ blip (expert: certification rule).**
The first break-even-fit cell, `d_c52e1670` (n7 4 h, 2 MWh), ended **uncertified at its cap (198) with zero
vetoes** — the Addendum 60 veto played no part, and DSO7 2025 Winter never appeared on this cell. A three-cycle blip
at cycles 111–113 (steps +72, −226, +125 €, each above the 45 € step floor) registered two tiny turning points
(swings 226 € and 125 €). The next real swing was 11,609 €, so the **all-pairs** "swings non-increasing" test fails
permanently from then on, although the range test at the cap passes easily (0.49 τ). The blip coincides with **no
solver event** (no non-Optimal exit, recovery or failure in cycles 105–116); it is a flat top where TSO and DSO block
movements nearly cancel.

Zero-solve replay of the candidate amendments on every committed re-settle record (W141, `d02efe69`; the control
reproduces every committed decision):

| option | d_c52e1670 | effect on committed certificates |
|---|---|---|
| **(a) swing noise floor F = τ/10 (≈ 454 €)**: a turning point registers only if the swing it closes is ≥ F | certifies at **150** (band 0.98 τ) | **none** — the floor rejects nothing on any other record; τ/20 gives identical results |
| (b) last-pair test: compare only the last two swings | certifies at **148** | none |
| (c) keep v5 | uncertified; reported on the uncertified form | — |

**Recommendation: (a).** It keeps the all-pairs guard against a growing oscillation and removes only swings too
small to be evidence of one; it is derived from τ, not tuned to this cell. (b) is weaker against beat-like
trajectories. Cost: a v6 re-freeze (≈ 1–2 h); the eight remaining break-even-fit cells then run (≈ 11 h), and
`d_c52e1670` is re-run or re-read — your choice (a re-run buys a certificate whose spec names its rule, ≈ 1.4 h).

**2. What τ bounds — an observation from two cells (expert).** The records show how far the objective still moves
after the rule would certify:

| cell | certifying cycle | Q(cap) − Q(k\*) | as a fraction of τ | trend at the cap |
|---|---|---|---|---|
| `d_c52e1670` (under (a) / (b)) | 150 / 148 | −3,543 / −4,004 € | 0.78 / 0.88 τ | ≈ −80 €/cycle, not decaying |
| cell 3 `b_4649234b` (certified under v5) | 148 | −3,351 € (from its v4 records) | 0.74 τ | — |

So the one-period range test bounds the stopping error at roughly τ per cell, not well inside it. That is within the
δR = 0.07 budget (τ = δR·V_SRP1/4) but at its edge; the manuscript statement should say so, or the budget should carry
a margin. A creep flag of the form |mean step over the window| > τ/L cannot serve as a check: it fires on 6 of the 12
existing certificates.

**3. Recorded for the Addendum 59 supplement (no action proposed).** The supplement asked v4 to assert that no sub-test
reads outside the last W cycles. As implemented it cannot: the oscillatory branch's turning-point detection, the
all-pairs swing test, P̂ and W read turning points before the window, and the monotone branch reads k0, the carry-in
sign and Q(lo − 1). v4 and v5 therefore **enumerate** these reads per certificate instead of asserting their absence.
On cell 1 and pb_y2025_n9 the certifying turning points were located through sign-step spans that contain their
non-Optimal cycles (120/128/141; 134), though none of those cycles is itself a turning point.

## Blocked on the author

Nothing new; machine time for the remaining queue is already allocated (Addendum 58).

## Changed

| what | commits |
|---|---|
| v5 build: criterion, live tier + metric capture, router branch, v5-from-records table, specs | `87ee2b14`, `b37ea166`, `f02f9839`, `ed767408`, `48cf000b`, `2b9e3243`, `5e0f8faf` |
| cell 3 re-run under v5 | `4b3be392` |
| break-even-fit cell `d_c52e1670` under v5 | `51280961` |
| swing-test variants replayed from records | `d02efe69` |

Readings accepted by the Planner in the v5 build: an Optimal exit on any attempt tier is clean (otherwise v5 would be
stricter than v4, against Addendum 60; the literal reading changes no outcome on the records); the ESSO is judged
against its own solver options; the stage spec holds 36 cells.

## Found

### Predictions against outcomes

| prediction | outcome |
|---|---|
| Expert (A60): cell 3 certifies under v5 | **held** — certified at 148 |
| Planner (recorded in v5): cell 3 is bitwise identical to its v4 run through 148 and certifies at 148 | **held** (G26) |
| Expert (A60): the nine break-even-fit cells meet the same block and certify | **failed** on the first — not by the block (it never appeared) but by the swing test |
| Group 1: settled slack s ∈ [+8, +30] k€ | cell 3 **+35.6 k€ — outside**; d_c52e1670 +5.5 k€ at the cap — outside |

### Certificates to date

| cell | certifying spec | k\* | s (k€) | gap at k\* (€) | margin M (gross) |
|---|---|---|---|---|---|
| B n5_4h_e1 | v4 `e11fbc89` | 173 | +18.7 | 1,016 | 58.0 k€ (6.7×) |
| B n9_4h_e1 | v4 | 173 | +19.5 | 592 | 61.1 k€ (7.3×) |
| B n9_4h_e3 (cell 3) | v5 `ab32ffc9` | 148 | +35.6 | 2,029 | ≈ 199 k€ (provisional) |
| D n7_4h_e2 | — | uncertified at 198 | +5.5 at cap | 836 | — |

Cell 3's three DSO7 2025 Winter Acceptable exits (≤ 2.69×) were counted clean under the 10× rule, as Addendum 60
intended; its only non-clean cycles are TSO recoveries before the first residual pass. Live metric capture was
confirmed on both runs (0 disagreements against the logs).

### Reproducibility note (for the manuscript, per Addenda 59–60)

- **DSO7 (case33_2) 2025 Winter** converges to ≈ 2.5× the complementarity tolerance on primary attempts in some cycles
  (13 exits across 4 records, 2.51–2.69×); accepted as clean under the 10× rule.
- **TSO (case9) 2035 Spring** produces recoveries ending "Acceptable" at 2,234–3,871× complementarity; always non-clean.
- Both are candidates for the post-revision cleanup (a scaling or bound-multiplier issue at those blocks).

## Not confirmed

- Whether the other eight break-even-fit cells hit the same blip pattern; the replay covers committed records only.
- Whether the post-certificate drift in Decision 2 is typical: only two cells have records beyond a certifying cycle.
- The creep flag's definition (internal vs carry-in mean) was the Worker's choice; neither version discriminates.
