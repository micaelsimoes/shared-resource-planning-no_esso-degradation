# P5.15 claim point #12: the break-even fit survives three uncertified points; the 5 % slope prediction holds

**Planner handoff report, 2026-10-02.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 58–62.
- **Specs:** stage spec v6 `96c23404` (criterion v6); banded-fit scorer `88cf7163`, pre-registered before any of the
  five cells below had a result.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; value = Q(0) − Q(cell),
  with Q(0) the settled x = 0 (`d110bd1a`, Q181); τ = 4,539.07 €; 2τ = 9,078.14 €.
- **Status:** not stopped. Per Addendum 62 the campaign continued straight into group 2: the four H cells are
  running (W147), with a report at the H claim point.

## Decisions needed

**1. Which "slope" the 5 % prediction refers to (expert; does not block anything).** The fit is
value = a + b·E + c·P (E in MWh, P in MVA). Before any result existed, the Planner fixed the verdict on the energy
coefficient **b** (`e487cbde`), with the 4 h marginal slope **b + c/4** reported beside it. The break-even price is
built from b + c/4 (e\* = b + c/4 − p_cost/4). Both readings hold at the interval midpoints. They differ only on a
stronger, report-only test: whether 5 % holds at every corner of the intervals.

| reading | certified-only | banded, midpoints | relative difference | worst corner of the intervals |
|---|---|---|---|---|
| b (€/MWh), the verdict | 236,612 | 235,217 | **0.59 %** | 7.1 % (fails 5 %) |
| b + c/4 (€/MWh), report-only | 246,781 | 246,013 | **0.31 %** | 3.7 % (holds 5 %) |

**Recommendation:** confirm b + c/4 as the manuscript's "slope", because it is the quantity the break-even is made
of. The prediction holds under either reading, so the choice changes no verdict.

## Blocked on the author

Nothing. The machine is running group 2 under the allocation already given.

## Changed

| what | commits |
|---|---|
| banded break-even scorer and its self-test (reproduces the Phase A fit exactly) | `88cf7163`, `c3d90807` |
| slope reading fixed before results | `e487cbde` |
| v6 D cells 4–8 | `02affadd`, `73d41460`, `dfdc2ccf`, `394fc4b4`, `31b85b85` |
| claim point #12 summarize | `9b8fb124` |
| banded fit at claim point #12 | `7c1a4ecb` |

No code changed in the campaign. Every cell replayed bitwise against its original record through its first residual
pass (G19). Commit approvals had held the machine idle between cells (≈ 1 h and ≈ 4 h). Those commits are now
pre-approved.

## Found

### The eight D cells and d_c52e1670 (node 7, gross)

| cell | P / E (MVA / MWh) | v6 outcome | k\* or cap | range/τ | s (k€) | uncertified bar (k€) | what blocked certification |
|---|---|---|---|---|---|---|---|
| `d_c52e1670` | 0.5 / 2 (4 h) | certified (from records) | 150 | 0.98 | +9.1 | — | — |
| `d_4a82a64a` | 0.75 / 3 (4 h) | certified | 142 | 0.72 | +23.6 | — | — |
| `d_3632b0ae` | 1.0 / 4 (4 h) | certified | 153 | **0.997** | +18.1 | — | — |
| `d_36686489` | 1.25 / 5 (4 h) | uncertified | 191 | — | +8.0 | 23.9 | 5 TSO recoveries (2030 Autumn, 2035 Spring) |
| `d_d3709599` | 0.5 / 1 (2 h) | certified | 175 | 0.86 | +17.7 | — | — |
| `d_a12d95a2` | 1.0 / 2 (2 h) | uncertified | 203 | 1.03–1.07 before the reset | +8.1 | 24.2 | a Boyd lapse at 164 (ESS primal 1.136) reset the rule |
| `d_f759dd48` | 1.5 / 3 (2 h) | uncertified | 198 | — | +13.8 | 41.4 | TSO recoveries at 141/144 vetoed 17 certifications, and one at 170 failed the growth test |
| `d_c7fee8be` | 2.0 / 4 (2 h) | certified | 145 | **0.984** | +37.1 | — | — |
| `d_9246ed01` | 2.5 / 5 (2 h) | certified | 138 | **0.985** | +25.7 | — | — (terminal step 10.4 × EPS0) |

**Every non-clean cycle after the first residual pass, in every uncertified D cell, is a TSO recovery ending
"Acceptable"** (2,100–6,100 × complementarity tolerance), in the 2025 Spring, 2030 Autumn, 2030 Winter and 2035
Spring blocks. No DSO recovery was non-clean. On `d_f759dd48` the DSO5 (case33_1) 2035 Winter block exited Acceptable
on its primary attempt in every cycle from 90 to 198 (1.2–4.2 ×). Under the 10 × rule these exits count as clean. At
the terminal round it failed G6, which does not stop the campaign. This is new to the reproducibility note.

**Certificates sit at their threshold.** Four of the six live D certificates have range ≥ 0.95 τ. Under CLAUDE.md's
terminal-to-threshold rule, these cells were *stopped* by the range test rather than settled well inside it. This
matches Addendum 60's Decision 2: the rule bounds the stopping error at about τ per cell.

### The break-even fit (W145, `7c1a4ecb`)

The three uncertified points enter as intervals Q_cap ± (bar + 2τ), with half-widths 33.3 k€ (`n7_2h_e2`), 50.5 k€
(`n7_2h_e3`) and 33.0 k€ (`n7_4h_e5`). The extremes are exact: the scorer evaluates every corner of the intervals.

| fit | n | break-even e\* (€/MWh) | margin to energy cost 253,878 €/MWh |
|---|---|---|---|
| Phase A (before settling) | 10 | 182,231 ± 2,591 | 71,647 |
| certified-only | 7 | 182,702 ± 1,712 | 71,176 |
| banded, midpoints | 10 | 181,933 ± 1,784 | 71,944 |
| banded, over the intervals | 10 | **173,486 – 190,380** | **63,497 – 80,392** |

**The conclusion holds under both fits over the whole interval:** storage at node 7 breaks even at about 182 k€/MWh,
at least 63 k€/MWh below its energy cost.

**Held-out check.** The certified-only fit did not use the three uncertified points, yet it predicts each of them
inside its interval. The offsets are 0.32, 0.11 and 0.02 of the interval half-width.

### Claims completed at this point (summarize `9b8fb124`)

- Fifteen claims are complete, and 14 of them are determinate.
- **B claims** (value − I of each D cell):
  - The three uncertified cells are determinate on the uncertified form, at 9.6×, 8.7× and 14.5× their bars.
  - Certified pairs range from 4.4× to 45.9×.
  - The tightest are B `n5_4h_e1` (4.4×), B `n9_4h_e1` (4.8×) and B `n7_4h_e1` (4.9×).
- **The one indeterminate claim is the F2 certificate pair:** margin 6,108 €, against an uncertified bar of 27,900 €
  (0.22×). It is already reported as uncertified (dead zone, Addendum 58), and it needs no determinate improvement.

### Predictions against outcomes

| prediction | outcome |
|---|---|
| Expert (A62): the two fits agree on the slope within 5 % | **held**: 0.59 % on b, 0.31 % on b + c/4 |
| Planner (A58): the settled break-even lies within ±6 k€/MWh of 182.2 k€/MWh | **held**: 181.9 (banded midpoints) / 182.7 (certified-only) |
| Planner (A58): margin to cost stays above 60 k€/MWh | **held**: 63.5 k€/MWh at the worst corner |
| Planner (A58) watch: ≥ 2 of the 4 single-node D cells with E ≥ 4 MWh uncertified → stop | 1 of 4 → no stop |
| Planner (A61): the D cells certify under v6 | **failed** on 3 of 8; recorded, not a stop (Addendum 62) |
| Group 1: settled slack s ∈ [+8, +30] k€ | outside on `d_c7fee8be` (+37.1 k€); the uncertified cells trip the failure clause |

## Not confirmed

- Why the TSO 2030 Winter block recovers on `d_a12d95a2` and `d_f759dd48`, and why DSO5 2035 Winter sits at
  Acceptable throughout `d_f759dd48`. The solver logs are preserved (hash-recorded), but neither question has been
  investigated. Both belong to the post-revision cleanup.
- Whether the Boyd-lapse reset on `d_a12d95a2` comes from a genuine excursion or from a single noisy ESS primal
  reading. The rule behaved as frozen; the cause was not examined.
- `d_9246ed01` certified with a terminal step of 10.4 × EPS0. Its range (0.985 τ) passed, but nothing beyond the
  certifying cycle has been measured.
