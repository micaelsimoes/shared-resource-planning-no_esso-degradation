# P5.15 claim points I, J, C and G: every claim keeps its sign; one certificate rests on a solver blip

**Planner handoff report, 2026-10-03.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 62–63.
- **Spec:** stage spec v6 `96c23404`.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; F = Q + I; τ =
  4,539.07 €.
- **Status:** not stopped. The 12 L cells are running (W151, ≈ 22–26 h) while this report is read (Addendum 63).

## Decisions needed

**1. `j_a11d7966`'s certificate rests on a solver blip (expert; no verdict depends on it).** The 1.0 MVA / 4 MWh J
cell certified on the oscillatory branch at cycle 158 (range 0.715 τ):
- **The oscillation was a blip.** Its turning points 138, 139 and 140 surround a non-clean TSO recovery at 138 (2030
  Autumn, ≈ 2,440 × complementarity), which gave P̂ = 2 and W = 20.
- **The window missed it.** The window [139, 158] is all clean, so v6's veto did not apply. Cycle 138 is read only
  through P̂ and the swing pair, as an out-of-window read (accepted under Addendum 61).
- **The cell was still descending.** After 140 the cost descends with no turning point, and the steps *grow* to
  −208 €/cycle at k\* (4.59 × EPS0).

So a blip supplied the oscillation that certified a drifting cell. This mirrors `d_c52e1670`, where a blip blocked
certification.

| option | what | effect on verdicts |
|---|---|---|
| **(a) keep the v6 certificate, flag it, and show the uncertified form beside it** | bar 3·max(\|gap\| 642, \|slack\| 5,293) = 15,880 € | **none**: J e3→e4 and e4→e5 are bound by their other cells' bars (27,694 / 26,544); J e4 value − I goes from 18.1× to 11.1×, still determinate |
| (b) for the post-revision rule: a turning point at a non-clean cycle cannot set P̂ | a criterion change | none in this campaign; it would have left the cell uncertified |

**Recommendation: (a) now, and (b) recorded as a post-revision rule question.** It belongs with the Addendum 60–62
history of how solver events interact with the turning-point tests.

**2. Long cells: done.** The author chose option (b): `BASH_MAX_TIMEOUT_MS` 14,400,000 in local settings. A
2 h 2 min background run survived it in this session (`1fcbce4e`). Each L launch is estimated against its own
worst-case wall time and refused above 13,800 s. Nothing is needed.

## Blocked on the author

Nothing. The optional m = 1.75 H cell (≈ 1.5 h) stays queued after the remainder, as the order puts it. Your call
when we reach it.

## Changed

| what | commits |
|---|---|
| I cell, summarize #17 | `232af066`, `4b30ea00` |
| J cells (`j_5f3cccb4` launched by the author in Terminal: the 2 h tool limit) | `b8a6e689`, `04db6a12`, `8ab183cc` |
| summarize #20 | `289297dd` |
| C cells, summarize #22 | `dad59dca`, `3105b60d`, `3d9b8d94` |
| G cells (ungated first C2 evaluations), summarize #26 | `bf810183`, `4ab435aa`, `58f7209e`, `28851414`, `abfd4c88` |
| `h_f9eae48f` interface prices for the dead-zone table | `241e299c` |

Every gate passed on every cell. G19 replayed bitwise wherever the cell is gated.

## Found

### Claims (36 complete; 34 determinate)

| claim | result | margin over bar | prediction (A58 design review) |
|---|---|---|---|
| I: second MWh at m = 2 | +46,515 | 3.64× | +30–55 k → **held** |
| I: 2 MWh value − I at m = 2 | +92,800 | 7.26× | — |
| J: 2 → 3 MWh | +40,973 | 1.48× | 25–45 k → **held** |
| J: 3 → 4 MWh | +42,807 | 1.55× | 35–60 k → **held** |
| J: 4 → 5 MWh | +51,820 (Q_cc +61,310) | 1.95× | indeterminate → **missed** (determinate) |
| J: value − I at 3 / 4 / 5 MWh | +133,773 / +176,580 / +228,401 | 4.8× / 18.1× / 8.6× | — |
| C y2025 (n5 + n9), F − F(0) | +102,097 | 8.09× | 120–150 k → **missed low** |
| C y2030 (n5 + n7), F − F(0) | +93,878 | 7.00× | 150–180 k → **missed low** |
| G n7_2h_e1 y2030, gross / net | +99,027 / +88,338 | 7.7× / 6.9× | 115–145 / ≥ 90 → **missed / missed** |
| G n7_4h_e2 y2030 | +137,432 / +113,066 | 10.9× / 9.0× | 160–190 / ≥ 90 → **missed / held** |
| G n7_2h_e1 y2035 | +139,660 / +85,089 | 11.1× / 6.7× | 190–220 / ≥ 90 → **missed / missed** |
| G n7_4h_e2 y2035 | +223,583 / +109,134 | 17.7× / 8.6× | 235–265 / ≥ 90 → **missed / held** |
| L y2025 n7_p0.75_e3 (vs the F2 certificate) | +94,347 | 3.41× | — |

**Every sign holds and every claim above is determinate.** In the C and G rows, x = 0 wins on every row, gross and
net. The two indeterminate claims are unchanged: H m1.5 (ruled in Addendum 63) and the F2 pair.

**The magnitude misses lie in the predictions, not in the settling.** For the gated C cells, the old certificate's
margin follows from M_new = M_old + s − 14,240.96 (Q181 minus the old-tail x0). That gives M_old ≈ 97.1 k (y2025) and
≈ 130.6 k (y2030), both already below the predicted ranges. The ranges were mis-anchored before any cell ran, the same
pattern as the Advisor's B ranges (0 of 3, Addendum 60 report). The G cells are first C2 evaluations: no old value
exists under the current configuration to anchor a range. G timing held: each certified at k0 + 62 to 71, inside
[50, 75]. Other predictions:

| prediction | outcome |
|---|---|
| `j_5f3cccb4` dead zone: gap-refused, t_sum ≈ −8…−10 k | **held**: 22 refusals, \|t_sum\| 8.7–9.1 k, −8,848 at the cap |
| C cells certify | **held** (k\* 176, 193) |
| gate-ability | **held** on every gated cell run |

### Cells

| cell | outcome | k\* / cap | range/τ | s (€) | note |
|---|---|---|---|---|---|
| `i_5a6a88b4` | certified (monotone) | 198 | 0.939 | −3,284 | — |
| `j_f3aa335e` | uncertified | cap 260 | monotone 1.009 | −9,231 | growth fail at a non-clean TSO turning point (178) |
| `j_a11d7966` | certified | 158 | 0.715 | −5,293 | Decision 1 |
| `j_5f3cccb4` | uncertified, gap clause | cap 270 | 0.609 (band) | −6,190 | dead zone, as predicted |
| `c_156ce2d1` | certified | 176 | 0.858 | +19,266 | — |
| `c_6597a79d` | certified | 193 | **0.984** | −22,520 | at threshold |
| `g_37b5c499` / `g_47dce43c` / `g_48749148` / `g_9abf31d4` | certified | 185 / 174 / 170 / 172 | 0.94 / 0.84 / 0.93 / 0.85 | — (ungated) | — |

### Dead-zone table (W149, `241e299c`)

`h_f9eae48f` (m 1.5, 0.25 MVA / 1 MWh):
- **The gap sits in one block.** All of the priced gap (−2.48 k€) is in TSO 2035 Spring: node 5 55 %, node 9 31 %,
  node 7 14 %. MW gaps are frozen while prices drift linearly, which is the F2 signature at a quarter of the size.
- **F2 is only partially reproduced.** TN conventional sits at Pmin, but also in the x = 0 control, so the storage does
  not cause it here. DN flexibility is interior, not at its zero kink.
- **What separates it from the control:** the TSO's cheapest lever is the shared ESS (0–9 €/pu, against a generator at
  6.8–16.3 k€/pu). This column was added after looking, so it is report-only.
- **A shared block.** 2035 Spring is also the recurring TSO-recovery block.

## Not confirmed

- **Whether `j_a11d7966` would have settled near its certified value:** nothing beyond k\* was run. Its −208 €/cycle
  terminal step suggests a further ≈ 1 τ of descent; no verdict is sensitive to it.
- **The Advisor's anchoring for the group-4 ranges:** the design review's arithmetic was not reconstructed. The
  M_old figures above are Planner-derived from the settled records and the 14,240.96 offset.
- **How the earlier runs over 2 h were launched:** W148 searched the W125/W110 commits and the state files and found
  no record.
