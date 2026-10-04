# P5.15 the re-settling queue is complete: the F2 certificate holds, the ageing row is restated at a 0.70 floor

**Planner handoff report, 2026-10-04.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 58 (with its supplement), 62 and 63.
- **Specs:** stage spec v6 `96c23404` (34 cells, complete); extension spec v3 `84775dc4` (7 cells, complete).
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded; F = Q + I; value =
  Q(0) − Q(x); net = gross − salvage. τ = 4,539.07 €. Determinacy threshold max(3 × larger bar, 2τ) (Addendum 61).
- **Status:** the machine is free. A Worker is computing the Step 5 rows that need no solves (W153: discount, salvage
  add-back, certification statistics). Nothing else is queued until Decisions 1–2 are given.

## Decisions needed

**1. The optional m = 1.75 H cell (author; Addendum 63).** About 1.5 h. It brackets the flexibility-ladder crossing
from above, so "break-even at m ≤ 1.75" becomes an interval. **Recommendation: run it** if the ladder is a headline
row. It is the only cell that would change a stated claim.

**2. The Step 5 rows at SRP1 (author: the parameter values are yours).** The brief's Step 5 list, set against what
now exists:

| row | status | proposal |
|---|---|---|
| discount 0/2/5/8 % | zero-solve; restating on the settled unit now (W153) | none needed |
| salvage add-back | zero-solve; tabulated now (W153) | none needed |
| certification statistics | zero-solve; tabulated now (W153) | none needed |
| uncoordinated benchmark | done (Addendum 57: +90.9 M€, determinate) | none needed |
| α row | done on the 2 × 2 (Addendum 44: five cells) | stands, paper scale only |
| **soh_min 0.50 vs 0.70 at C2 + fade** | **one evaluation**: the C2_calfade unit at soh_min 0.50 | **needs plumbing:** `apply_model_variant` cannot change soh_min (W134). New files only, an Advisor design check, a frozen spec, then ≈ 1.5 h |

The expert's "≈ 10 cells" for Step 5 reduces to **one compute cell** (soh_min 0.50) plus the optional m = 1.75 cell.
**Recommendation:** authorise both. Order: m = 1.75 (no plumbing), then the soh_min row once its plumbing passes a
bitwise check (C2_calfade at 0.70 = `3f084f2f` through 172, the G28 precedent). Then the stop for review before the
Step 6 tables are frozen.

**3. One verdict changed by the determinacy floor (expert).** `E:C2_calfade vs C3` (value difference +9,329 €) is
0.71× its threshold: within resolution. Under the superseded settled-vs-settled rule it was determinate. This is the
first verdict the Addendum 61 floor has changed, so the expert's prediction "the floor changes no verdict" now fails
on one claim. **Recommendation:** report it as within resolution. The restated statement S1 becomes
"C2_calfade is within resolution of C3 at a 0.70 floor (x_C3 1.038)".

**4. The ageing elasticity statement is restated, and the expert's prediction is unresolved (expert).**
- At the 0.70 floor, ε_AE over the resolvable arms (C2, C4, no-ageing) is **1.04–1.85**, against ≈ 0.6 in the
  0.50 era. The value is now more elastic to available energy. A plausible reading, not measured, is that the floor
  removes the late-life tail that diluted it.
- The expert's prediction (Addendum 58 supplement) is scorable only on C2_calfade, the one non-reference arm where
  the floor binds. The point estimate holds (ε_k 0.0331 < 0.0394), but the difference is **within resolution**
  (0.20×). On C2 and C4, where the floor does not bind, ε_k rises determinately.

**Recommendation:** state the prediction as "consistent in direction, not resolved at this precision", and restate
S5 with the new band.

**5. `l_0ee93aca` and the dead-zone operationalisation (expert; no verdict depends on it).**
- The cell shows the dead-zone fingerprint (t_sum −9.3 k€, pf_primal flat at 0.69).
- But two non-clean TSO 2035 Summer recoveries (218, 222) reset the rule, so the gap clause was never reached. The
  scorer therefore records "not gap-refused".

**Recommendation:** record it in the dead-zone table by signature, with the cause stated.

## Blocked on the author

Decisions 1 and 2: machine time. The Mac is idle until then, apart from W153.

## Changed

| what | commits |
|---|---|
| L cells 23–34, summarize #38 | `f42e7299` … `48a447fa`, `b95ee706` |
| extension: six ageing arms, two summaries | `22c7c116` `23cee034` `afc6dbd7` `4c735dc3` `d7602e1a` `dbbf6ccf`, `72d52c23` |
| pb_y2025_n5 re-run, summarize | `2495b3c8`, `184229bb` |
| long-cell arrangement (`BASH_MAX_TIMEOUT_MS` 14,400,000 in local settings; a 2 h 2 min check survived) | `1fcbce4e` |

Every gate passed on every cell. No run was killed: `l_2ab0ce2d` ran 10,759 s under the raised limit.

## Found

### L: the F2 certificate (12 neighbours against the incumbent)

**All 12 differences are positive: no neighbour is determinately better, so the certificate holds.** Seven are
determinately worse (1.03–2.22× their thresholds). Five are within the uncertified bar (0.20–0.95×), because the
incumbent and four of them sit in the dual dead zone.

| prediction (A58 design review) | outcome |
|---|---|
| the 7 small cells certify | **held**, 7/7 |
| their s ∈ [−5, +10] k€ | **held** (−1.4 to −0.35 k€) |
| margins move ≤ 10 k€ | **held** (the move equals s) |
| 4 dead-zone cells gap-refused | **3 of 4** (`l_0ee93aca`: Decision 5) |
| `l_2ab0ce2d` borderline | recorded: certified monotone at 400, t_sum +0.89 € |

**The dead-zone fingerprint is common to all five 2030 dead-zone cells** (and to the F2 incumbent): t_sum −9.2 to
−9.4 k€, split node 5 55 %, node 9 31 %, node 7 14 %, with pf_primal flat at 0.69. The m = 1.5 H cell shows the
same split at a quarter of the size (W149).

### Ageing at soh_min 0.70 (six arms, unit y2025 n7 0.25 MVA / 1 MWh)

| arm | floor binds | value | value − I | verdict |
|---|---|---|---|---|
| C3_unit | 2035 | 244,210 | −73,747 | determinate (5.7×) |
| C2 | never | 286,350 | −31,607 | determinate (2.5×) |
| C4 | never | 272,760 | −45,197 | determinate (3.6×) |
| **C2_calfade (baseline)** | 2035 | **253,540** | **−64,417** | determinate (4.9×); **G28: bitwise = the settled unit through 172** |
| C3_midblock | 2035 | 252,065 | −65,892 | determinate (5.1×) |
| no ageing | — | 313,816 | −4,141 | within resolution (0.32×) |

No aged arm pays. Without ageing the unit sits at break-even within resolution, so ageing is what makes storage
uneconomic at SRP1. This matches the 0.50-era reading, now on settled numbers.

### pb_y2025_n5 re-run

- **The run:** G19 bitwise through 110. Certified at 199 after 29 vetoes, all from one non-clean TSO 2035 Spring
  recovery at 167 (also read outside the window by the lapse and sign spans; recorded). s = +20,913 €, inside the
  group-1 range.
- **The claim:** the Phase B margin is **+49,693 €, determinate at 3.94×**.
- **The flag leaves the table.** Every Phase B certificate now has the same strength: this one at 49.7 k€, beside
  the six W129 neighbours at 43.3–54.9 k€, all determinate.

### Predictions against outcomes (this report)

| prediction | outcome |
|---|---|
| Planner G28: C2_calfade reproduces 3f084f2f bitwise through 172 | **held** |
| Expert (A58 supplement): floor-binding arms show a smaller value change per unit cycle life | point estimate held; **within resolution** |
| Expert (A61): the floor changes no verdict | **failed on 1 claim** (Decision 3) |
| L predictions | table above |

## Not confirmed

- The new ε_AE band rests on three resolvable arms. Why the floor raises elasticity (the late-life tail argument
  above) is the Planner's reading, not a measured decomposition.
- **TSO 2035 Spring** recurs as the block whose non-clean recoveries veto, reset or block certification, now on
  `pb_y2025_n5_v6` as well. Its cause is not investigated; it belongs to the post-revision cleanup.
- The soh_min plumbing has not been designed. The "one evaluation" scope is read from the brief's Step 5 list.

## Supplement: Step 5 zero-solve rows (W153, `48e76c9f`; zero solves, guards at 0)

**Discount row** (settled unit `3f084f2f` against x = 0 `d110bd1a`; gross; the rate stays at 2 %, not tuned):

| rate | value | value / I | value − I | determinacy |
|---|---|---|---|---|
| 0 % | 277,120 | 0.872 | −40,837 | determinate (2.6×) |
| **2 %** | **253,540** | **0.797** | **−64,417** | determinate (4.9×) |
| 5 % | 225,232 | 0.708 | −92,725 | determinate (7.1×) |
| 8 % | 203,367 | 0.640 | −114,590 | determinate (8.8×) |

- **Discount rate:** no rate makes the unit pay. The rate cannot decide the sign, as Addendum 28 stated.
- **Captured spread:** 59.0 € per MWh-cycle, inside the expert's 55–60 expectation (Addendum 28: **held**). That is
  0.60 of the market's top-4 h spread.
- **Comparing with the pre-settling row:** that row also differs in ageing configuration (C3, soh_min 0.50), so the
  difference is not a settling effect alone. 2035 carries the change: per-year value 85.4 k against 101.2 k, because
  the 0.70 floor binds there (EFC/day 0.64).

**Salvage add-back** (60 claims; 52 net verdicts computed with the committed scorer and validated on the 8 recorded
G rows):
- **No sign changes** between gross and net, and all 12 L neighbour differences stay positive.
- **One verdict changes:** F2 incumbent vs `l_0ee93aca` goes from within the bar (gross) to determinate (net).
  That is in the certificate's favour.

**Certification statistics** (42 cells: 34 v6 + 7 extension + `d_c52e1670`):
- **Outcomes:** 32 certified (24 oscillatory, 8 monotone). 10 uncertified: 5 by the gap clause, 2 by lapse resets,
  3 by the growth test.
- **Timing:** k\* − k0 has median 63.5 (range 21–89).
- **Stopping point:** range/τ at k\* has median 0.86. Six certificates are at ≥ 0.95 τ.
- **Non-clean cycles:** all 21 after the first residual pass are TSO events; none is DSO or ESSO. Wall time is
  25.6 s per cycle (median).
- **New for Decision 1 of the I/J/C/G report:** a second certificate, `d_4a82a64a` (turning point 108), also has a
  non-clean cycle among its turning points, like `j_a11d7966`. The post-revision rule question covers both.
