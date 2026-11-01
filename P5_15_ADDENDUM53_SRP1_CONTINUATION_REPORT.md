# P5.15 Addendum 53: the SRP1 references settle 15–21 k€ above their certified values; the value and its sign survive; C\* does not settle within 100 cycles

**Planner report, 2026-09-28.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 53 (Rulings 1–4).
- **Frozen spec:** v39 `8a612429` (predecessor v38 `8bc0ffa6`); cell specs x0 `d1d5c3bc`, unit `434f46fc`, C\* `1a7483ef`.
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded. s = Q at the new
  certification − Q at the old certification (Q_N). V = Q(x = 0) − Q(unit). I = 317,957.01.
- **Status:** nothing is running. **Stopped for review**, as Addendum 53 orders; the triage and re-settling campaign are
  sized, not launched.

## Decisions needed

**1. C\* did not settle within 100 cycles (expert).** After the post-AA rebound, C\* has drifted down at an
almost constant −245 to −290 €/cycle for the last ≈ 45 cycles (mean −265 €/cycle over 143–187). The steps are not
decaying, so the stop rule's monotone branch refuses; its range over 44 cycles is 11,392 € (2.5 τ). The pf_primal
residual ratio has been **rising** steadily since cycle 152 (0.013 → 0.127), still far below 1 and without a lapse. The
spec names no fallback for an uncertified cell.

| option | what | cost |
|---|---|---|
| **(a) extend C\*** (recommended, as a diagnostic) | new spec v40 (cap only): bitwise replay to 187, then up to 200 more cycles under the same holds and rule | ≈ 1.3 h replay + ≤ 1.4 h, Mac |
| (b) accept as uncertified | report C\* as "uncertified at cap; band [−4.4, +11.3] k€ over the last 56 cycles and drifting −265 €/cycle" | none |

C\*'s own claims do not depend on it (value − I = −749 k€ against a drift of ≈ 25 k€ per 100 cycles). The reason to
extend is methodological: the re-settling campaign (Decision 2) will meet cells like this, and the rule's behaviour on a
non-decaying creep — ends, turns, or the residuals leave tolerance — should be known before 33 cells are sized around
it.

**2. The re-settling campaign: 33 SRP1 cells flagged, VM-sized (expert, then author for machine time).** Threshold =
3 × the largest settled slack = 3 × 20,806 = **62,419 €** (provisional: C\* could raise it only if its settled |s|
exceeded 20.8 k€, which at the present drift would take ≥ 62 more cycles). 33 cells fall below it (42 at 75 k€); both
exceed ten, so per Ruling 2 the campaign is **the VM's first job after the equivalence gate** (≈ 40 h serial on the
Mac; not launched). Two questions need a ruling before it can be specified:
- **(a) Continuation or re-run?** Every flagged cell was certified with the **old tail** (compl_inf_tol 1e-4). The
  continuations here replay bitwise only because the three references had been re-certified under the tight tail.
  Either the campaign spec pins the old tail for these cells (a true continuation, ≈ 1.2 h/cell) or it re-runs them
  under the tight tail plus the new criterion (a new configuration for those tables, ≈ 1.5–2 h/cell).
  **Recommendation:** the re-run — it gives every manuscript SRP1 number one configuration.
- **(b) Order by risk.** The references were certified at the bottom of the post-AA fall or on its rebound — exactly the
  states giving s > 0 — and their slacks were alike and largely cancelled in V. Cells certified **while still
  descending** could settle less or even below their certified value, which would **shrink** margins rather than cancel:
  all six Phase B neighbours (margins 33–57 k€, every one mid-descent against x = 0 at its trough), the year ladder's
  2030 cell (−2,433 €/cycle at certification), and the F2 certificate's challenger. **Recommendation:** run these
  phase-mismatched pairs first.

**3. The benchmark (author).** Per the order, the uncoordinated arms follow this review. Ruling 4 makes the coordinated
arm the settled x = 0 value, **Q181 = 653,873,702.19**. Its models were persisted at cycle 181 (`certified_models.pkl`,
sha `99ab1070…`, hash-recorded in `e4b5c992`). **Recommendation:** run the common-Q gate and the λ_t look on the
**settled cycle-181 models** rather than the cycle-132 W86 models Addendum 49 named, so that every coordinated
quantity refers to the settled state; the rest of Addendum 49 unchanged.

**4. A truncated commit message (housekeeping).** Commit `279c732b` (C\* literal-reading evidence) has a message cut at
"(W103s" by a shell-quoting error; its content is correct. Amending would rewrite history (unpushed). **Recommendation:**
leave it; the intended text is recorded here under Changed.

## Blocked on the author

Nothing runs until Decisions 1–3 are ruled.

## Changed

- Stop rule and harness: W101 `ab0bcbc9` (code), `07589e40` (zero-solve checks), `4ca1573d` (v39), `31917489` (cell
  specs). The per-period interface consensus duals (λ_t per node) are now in the **default** per-cycle records
  (`interface_duals_per_cycle.jsonl`; 3,456 floats/cycle at SRP1), proven write-only.
- Gate-result hygiene: W100 `ed5ba42a`, `88a8cd1d` — one shared writer refusing string flags; repository-wide typing
  test (passes after every run here).
- Runs (each attached, alone, concurrency 1):

| cell | evidence | wall | cycles run |
|---|---|---|---|
| x0 | `e4b5c992` | 1.19 h | 181 |
| unit n7_4h_e1 | `3ee62863` | 1.22 h | 172 |
| C\* | `82dcebb7`, `279c732b` | 1.48 h | 187 (cap) |

- Literal-reading scripts `953b9bcd` (x0, unit), `279c732b` (C\*). **`279c732b`'s intended message**, recorded here:
  "… W103's labels kept verbatim in the output). c_star: rule UNCERTIFIED at cap 187; literal reading certifies at
  k = 132 ONLY (P_hat 22, W 25, window [108,132], range 4,514.71 ≤ τ, Q_132 − Q_N = +11,344.08); K1 exact; K2 13/13
  (w104_k2_uncertified_check.json); K3 bitwise; W100 bool test PASS 89,214 files 97/97; manifests 10/10 and 12/12;
  inputs at 82dcebb7".
- Frozen scorer (`--summarize`, zero solves): `62bdeafe` (`w101_three_reference_summary.json`, sha `87234aed…`).

## Found

### Reproducibility

**All three references replayed bitwise to their old certification cycle** (132, 112, 87), at concurrency 1 against
originals run at concurrency 3 — including a recovered solver failure at cycle 6 of the unit cell, reproduced exactly.

### Settling

| cell | old cert. N | new cert. k\* | branch | half-swings (€) | period | band width (range/τ) | **s = Q_k\* − Q_N** |
|---|---|---|---|---|---|---|---|
| x0 | 132 | **181** | oscillatory | 20,095 / 7,027 / 2,605 | 29 | 4,209 (0.93) | **+14,971** |
| unit | 112 | **172** | oscillatory | 40,514 / 17,195 / 6,311 / 2,382 | 30 | 4,351 (0.96) | **+20,806** |
| C\* | 87 | **none (cap 187)** | — | 29,886 | — | 4,736 over last 20 | −4,379 at cap, drifting |

- The old rule stopped **every** reference at the bottom of the post-AA fall or on its rebound; each then rebounded
  20–30 k€. x0 and unit then settled through damped oscillations (ratio per half-swing ≈ 0.37–0.38, twice the 3 × 3
  period); C\* rebounded once and began a slow, non-decaying descent.
- No residual lapse on any cell. pf_primal stayed ≤ 0.29 after N on x0 and unit, ending at 0.005; on C\* it fell to
  0.013 and has risen since (0.127 at the cap). The plateau the AA note warned of (2.8–4.5) did not recur.
- **The literal two-sign-change reading agrees on x0 and unit (181, 172) but would have certified C\* at cycle 132
  (+11,344), after which Q fell a further 15.7 k€.** This supports the stricter three-turning-point reading the Planner
  adopted.

### The value

| quantity | old (certified) | settled | resolution |
|---|---|---|---|
| V_SRP1 = R_ref | 259,375.33 | **253,539.62** | 8,561 (band_x0 + band_unit) |
| ΔV = s_x0 − s_unit | — | **−5,836** | 8,561 → indeterminate vs 0 |
| **value − I** | −58,582 | **−64,417** | 8,561 → **determinate (7.5×)** |

**x = 0 is optimal at SRP1 under the baseline, settled.** The slack was largely common-mode, as the expert argued, and
the sign margin grew slightly.

**3 × 3 R, restated against the settled reference:** 0.9336 at 3 × 3 certification; **0.9090** with the 3 × 3 x0
cell's post-hoc settled descent (6.2 k€; node-7 was never continued). **R ∈ [0.909, 0.934] against 0.9331
predicted.** The SRP1 side now contributes ± 0.032 to R, inside the δR = 0.07 budget. The "0.03 gap" Addendum 53
recorded was an artefact of the unsettled reference.

### Predictions against outcomes (frozen scorer, `62bdeafe`)

| prediction | outcome |
|---|---|
| Expert P1: each settled slack in [5, 25] k€ | x0 **held** (14,971); unit **indeterminate** (20,806 — within its 4,351 band of 25 k); C\* **not scoreable** (uncertified) |
| Expert P2: \|ΔV\| ≤ 20 k€ | **held** (5,836; resolution 8,561); reopen rule (> 40 k€) not triggered |
| Expert P3: slacks alike within ≈ 5 k€ | **not scoreable** (C\* uncertified); x0 vs unit differ by 5,836 — indeterminate, as pre-declared |
| Advisor: k\* x0 [162, 173], unit [141, 151], C\* [121, 131] | **all missed late** (181, 172, none) — the SRP1 period is ≈ 29–30 cycles, not the 20 borrowed from 3 × 3 |
| Advisor H-b: only the monotone branch fires, s_x0 < 0 | **refuted** on x0 and unit |
| AA-note warning: pf_primal returns to its plateau, Boyd lapses, Q returns +38–53 k€ | **not observed** (no lapse; Q returned at most +33.6 k€ at the unit's first peak, then settled) |

### Triage (Ruling 2): SRP1 differences with margin < 3 × 20,806 = 62,419 €

Inventory from committed reports and records (read-only Advisor, spot-checked by the Planner against five cells'
per-cycle records). Phase at old certification: T trough, R rising flank, D decelerating descent, M mid-descent.

| item | claim | margin (€) | cells to re-settle | phase |
|---|---|---|---|---|
| A | headline sign, value − I | 58,582 → **−64,417 settled** | **done** | — |
| B | Phase A ladder, "x = 0 minimises F" | 53,607 / 55,800 / 58,529 | old-tail x0, n7, n5, n9 (4) | x0 T; n7, n9 R; n5 D |
| C | Phase B, 14 neighbours | 33,459 … 57,411 (6 below) | 6 | all D vs x0 T — **mismatched** |
| E | ageing ×1.126 and variants | 485 … 56,634 | 6 | C2, C3 T; no-ageing D |
| F | discount-rate row (reweighting) | 30,116 (0 %) | 0 (shares E) | — |
| G | year ladder 2030/2035 | 43,209 / 48,023 (net) | 2 | 2030 **M** |
| H | flexibility ladder ×1.5 / ×2 | 31,075 / 48,396 | 4 | ×1.5 T vs R (mismatched); ×2 R/R |
| I | ×2 second MWh | 42,766 | 1 | D |
| J | F2 ladder marginal MWh | 35,025 / 46,746 / 50,923 | 3 | flat |
| L | F2 certificate "no determinate improvement" | 6,338 … 53,170 | 7 | incumbent R vs challenger **M** |
| D | break-even (slope of a 10-point fit) | 71,646 per MWh | 0 at 62.4 k (9 at 75 k) | slack cancels in a slope |

**33 cells at 62.4 k€, 42 at 75 k€.** Not at risk: F2 plan vs corner (102 k), ×3 rows, C\* value − I (749 k). The
α row is 2 × 2, not SRP1 (value − I −50 k€; its instance's slack is unmeasured) — a separate decision. The break-even
"± 2,591 €/MWh" is a fit standard error, not a settling error, and should carry that caveat regardless.

## Not confirmed

- Where C\*'s drift ends, and why pf_primal rises on C\* alone. Three readings fit the records: a mode with a period
  longer than 2 × P_MAX (the 109–114 stall a sub-floor turning point), a slow drift to a lower limit, or the onset of a
  slow instability of the plain map at frozen ρ. Only an extension distinguishes them (Decision 1).
- Node-7 (3 × 3) settled value — never continued; 3 × 3 R's lower end rests on x0's post-hoc descent.
- The triage phases are read from the last three steps of each window, not the full transient; the inventory covered
  Addenda 27–53, `REVISION_CONTEXT.md` and the Addendum 27–38 reports, not the manuscript source.
- `git_head_at_run` in each campaign's results records HEAD after the run (a Planner `TASKS.md` commit), not at launch;
  no code changed between.
