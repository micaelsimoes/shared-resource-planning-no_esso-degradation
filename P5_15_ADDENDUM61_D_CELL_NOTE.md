# P5.15 Addendum 61 follow-up: a 5 MWh break-even cell is genuinely unsettled at its cap — keep the rule, change the stop

**Planner interim note, 2026-10-01.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 58–61.
- **Spec:** stage spec v6 `96c23404` (criterion v6: v5 plus the τ/10 swing floor on the growth test and the turning
  points); extension spec `84775dc4`.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; τ = 4,539.07 €.
- **Status:** nothing is running. The campaign stopped because the Planner's prediction "the eight remaining
  break-even-fit cells certify under v6" failed on the third (gate G27).

## Decisions needed

**1. Keep criterion v6 as frozen (recommended), and record `d_36686489` as uncertified (expert).**
`d_36686489` (node 7, 1.25 MVA / 5 MWh) ended uncertified at its cap (191) with zero vetoes. Five non-clean TSO
recovery exits after its first residual pass (2030 Autumn at 94, 113 and 176; 2035 Spring at 119 and 164; 2,400–2,900×
complementarity, all recovery tier) inject jumps into Q. The swing ending on cycle 113 (631 €) clears the τ/10 floor and
the next (2,148 €) is larger, so the all-pairs growth test fails from 115 to the cap. The window veto cannot reach these
cycles: the growth test reads the whole swing history (the out-of-window reads enumerated under Addendum 61 ruling 3).

The candidate amendment that follows Addendum 59's principle — exclude swings that contain a non-clean cycle — was
replayed with zero solves on all 21 committed records (W144, `663abe62`):

| variant | d_36686489 | then moves by the cap | committed certificates |
|---|---|---|---|
| v6 as frozen | uncertified at 191 | — (drift −85 €/cycle at the cap) | — |
| C1: exclude swings spanning a non-clean cycle | certified 139 — **with no swing pairs left to compare** | **−1.01 τ** | unchanged |
| C2: C1 + no turning point at a non-clean cycle | certified 144 | **−0.95 τ** | **loses `d_4a82a64a`** (v6 certificate at 142) |
| C3: growth test over the last three swings | uncertified | — | unchanged |

C1 and C2 would certify a cell that then moves beyond the "at most 0.9 τ" statement Addendum 61 placed in the manuscript,
and C2 fails Addendum 61's invariance test. **The cell had not settled within its cap; v6's refusal is the correct
verdict.** The pattern is specific to it: across all records, only this 5 MWh cell shows repeated TSO recoveries (5
after its first residual pass, in two TSO blocks; every other record has 0 or 1). Every TSO recovery after the first
residual pass, in every record, is non-clean; every DSO recovery is clean.

**2. Change the procedure, not the rule: certification status is reported, not a stop condition (expert).**
Three stops in a row (claim point 3, the first D cell, now this one) were triggered by "certifies" predictions failing.
The uncertified form already decides large margins (3 × max(gap, slack)), and a cell that does not settle within its cap
is a result. **Recommendation:** for the remaining queue, a failure of a *certification-status* prediction is recorded
and the campaign continues; the campaign stops only on harness faults, gate failures other than G6/G27, or a failed
*margin or sign* prediction. Cost: none; it saves a stop per uncertified cell (≈ 12–24 h of idle machine each time a
ruling is awaited).

**3. The break-even fit with uncertified points (expert).** The D fit is a 10-point regression. If `d_36686489` (and any
later uncertified D cell) enters the fit at its cap value, its point carries a band, not a certificate. Options: (a) fit
on all points, carrying each uncertified point's band into the slope's stopping-slack bound (Addendum 61's report-only
bound, extended); (b) fit on certified points only and report the uncertified ones beside the fit. **Recommendation:
(a)**, with (b) reported beside it; the break-even margin to cost (≈ 72 k€/MWh before settling) is large against any
plausible shift.

## Blocked on the author

Nothing; machine time is allocated.

## Changed

| what | commits |
|---|---|
| v6 build (both floors carried; determinacy floor; `d_c52e1670` certified from records at 150) | `575bfec4`, `7914860f`, `367f06f2`, `ab289125`, `689340d2`, `39ecd06f` |
| v6 D cells 1–3 | `354b362a`, `48625326`, `f546dec3` |
| clean-swing variants replayed from records | `663abe62` |

## Found

| cell (node 7, MVA / MWh) | criterion | k\* | range/τ | s (k€) | gap at k\* (€) |
|---|---|---|---|---|---|
| `d_c52e1670` (0.5 / 2) | v6 from records | 150 | 0.98 | +9.1 | 1,316 |
| `d_4a82a64a` (0.75 / 3) | v6 | 142 | 0.72 | +23.6 | 1,000 |
| `d_3632b0ae` (1.0 / 4) | v6 | 153 | **0.997** | +18.1 | 843 |
| `d_36686489` (1.25 / 5) | v6 | uncertified at 191 | 0.63 (last W at the cap) | +8.0 at the cap | 1,082 |

`d_3632b0ae` certified at 0.997 τ — stopped at its threshold rather than settled well inside it (CLAUDE.md's
terminal-to-threshold rule); recorded.

### Predictions against outcomes

| prediction | outcome |
|---|---|
| Planner (v6): the eight D cells certify | **failed** on the third |
| Expert (A61): the determinacy floor changes no verdict | **held** |
| Planner (A61): "moved at most 0.9 τ" on continued cells | **held** (0.74, 0.78 τ); C1/C2 would have broken it (1.01, 0.95 τ) |

## Not confirmed

- Whether the remaining D cells (`d3709599`, `a12d95a2`, `f759dd48`, `c7fee8be`, `9246ed01`; two of them ≥ 4 MWh)
  show the same TSO-recovery pattern; one 5 MWh cell cannot establish a dependence on storage size.
- Why the TSO 2030 Autumn and 2035 Spring blocks recover at all on this cell (post-revision cleanup, with the DSO7 2025
  Winter block).
- `d_36686489`'s overlap at its old certificate (−3.28 × 10⁻⁶) is 2.5× the other cells'; not investigated.
