# P5.15 Addendum 57: coordination beats the static no-reverse-flow rule by 90.9 M€; eight of ten re-run cells settle and every restated margin holds; the F2 pair stalls in a dual dead zone

**Planner report, 2026-09-29.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 55–57.
- **Specs:** benchmark v5 `bca69f97` (v1 `46b6ba96` → v2 `bb659122` → v3 `47b37d6e` → v4 `a50ed4c3`); re-settling
  campaign `fc791891` (superseding `d902a85c`, never run); settling criterion v2 (`settling_criterion_v2.py`; v1
  byte-identical, so the references' certificates are untouched).
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded. t_sum = the priced
  interface-consensus gap Σ w·π·baseMVA·(p_DSO − p_TSO); Q_cc = Q + t_sum, a first-order diagnostic; verdicts are read on
  gross. Salvage is excluded from gross (net = gross − salvage).
- **Status:** nothing is running. **Stopped for review**, as Addendum 57 orders.

## Decisions needed

**1. The gap clause cannot be met where storage creates a dual dead zone (expert).** Both F2 cells (×2 flexibility
price; storage at nodes 5 and 7) settled in the objective (drift −2 and −6 €/cycle at the cap) but ended **uncertified
by the gap clause**. Their priced gap sat at −9.3 / −9.2 k€ (≈ 4 τ) against the 2,270 € bound, with the pf residual
frozen at 0.6917 for 70–90 cycles. W126/W127 (zero solves) located and explained it:

- 99.97 % of the residual is **one TSO hour — 2030 Autumn, hour 4 (the day's cheapest), interface P at all three
  nodes** — where the DSO and TSO copies have been frozen 0.03–0.13 MW apart **since cycle 52** (so the old certificates
  carried it too), while their duals grow linearly under a standard update.
- **Not an infeasibility.** It is a dead zone between two kinks: every DSO sits at zero flexibility, and moving means
  buying up-flexibility at ≈ 50 €/MWh (×2 price); the TSO's conventional units sit at their 0 MW floor because the
  shared storage — economically indifferent in the ESSO (H_ess-flat) — fully discharges 1.25 MW in the cheapest hour,
  leaving ≈ 0.24 MW of free surplus the TSO can only push across the interfaces. The dual crosses the gap at
  ≈ 0.1 €/MWh per cycle: **≈ 500 more cycles** (first-order extrapolation). The settled x = 0 has no such kink; its gap
  closes.
- The incumbent's per-node gap equals the challenger's to a few euros: the stall is not plan-specific.

| option | effect |
|---|---|
| **(a) report as is** (recommended for this revision) | F2 cells stated as "objective settled (drift ≤ 6 €/cycle); certification refused by the consensus-gap clause; priced gap 9.2–9.3 k€ in one hour, a dual dead zone"; the F2 certificate row (challenger − incumbent, ≈ 6.6 k€ on the old certificates) indeterminate, which Addendum 57 already expected |
| (b) amend the clause | e.g. certify on the objective and *report* the gap when it is localised and stationary — a certification-rule change |
| (c) a method change for storage cells | unfreeze ρ_pf, or re-enable AA, after the first residual pass — changes the certifying regime |
| (d) the ESSO economic tie-breaker | removes the arbitrary discharge that creates the TSO kink — scheduled post-revision (Addendum 56) |

**2. One Phase B certificate rests on a recovered solve (expert).** pb_y2025_n5 certified at cycle 167 on the
oscillatory branch — **the cycle at which the TSO 2035 Spring solve hit maxIterations and was recovered only to
"Solved To Acceptable Level"** (G6 failed); that recovery's +439 € jump registered the fifth turning point. Its margin
(M = 51.3 k€, resolution 8.7 k€) is large enough that no plausible certification cycle changes the verdict.
**Recommendation:** accept it, flagged; and for future specs, a rule that a cycle whose accepted solve is not
`Optimal` cannot be the certifying cycle.

**3. The year ladder depends on the salvage convention (expert and author).** Both cells are first C2 evaluations
(no gate), certified cleanly:

| | M gross (I + Q − Q181) | terminal salvage | M net of salvage |
|---|---|---|---|
| 2030 | 68,498.9 | 11,759.3 | 56,739.6 |
| 2035 | 110,957.6 | 56,504.2 | 54,453.4 |
| **2035 − 2030** | **+42,458.7 — determinate** (res. 8,136.5) | | **−2,286.2 — indeterminate** |

"Investing later is worse" holds in gross and not net of salvage. The frozen formula is gross; the manuscript's earlier
year-ladder claim was stated in both. **Recommendation:** state both; the claim as a ranking is supportable only in
gross.

**4. The remaining re-settling cells (author: machine time).** W117's re-run list is 47 cells; this campaign covered 10.
The other 37 (≈ 1.3–1.7 h each, ≈ 50–60 h on the Mac) await the VM or further Mac time. The phase-mismatch risk that
motivated the order did not materialise (below), which lowers their priority.

## Blocked on the author

Machine time for the remaining 37 cells (Decision 4).

## Changed

| what | commits |
|---|---|
| NRF benchmark: v3 build, v4 retry accounting, v5 re-evaluation fix; stages; extract | `79b945f9`, `e4bffcad`, `5d923082`, `4858b3a5`, `8d42dfb8` |
| benchmark reviews: Advisor decomposition; 12-block energy check | `0c3451cf` |
| triage recompute (Addendum 57 rule) | `3b2e76de` |
| re-settling campaign build and spec | `e2c1ac09`, `d011937b`, `b4aab001`, `5227384a` |
| ten cells | `aad4197b`, `075823cf`, `6eb5f7ca`, `ad4e4747`, `9dff25d0`, `d6333b6f`, `bcc7669b`, `af2210d2`, `d5f150f6`, `1250d846` |
| F2 stall diagnostics | `3d67820d`, `2f90194b` |
| campaign summary (frozen scorer, zero solves) | `eb61fb11` |

## Found

### The benchmark (Addendum 57, Decision 1)

- **Sweep (no interface rule, report-only):** the TN cannot accept the DNs' exchange in **1 of 12 blocks (6 h) under
  the passive arm and 8 of 12 (25 h) under the price-taker** — the TN is a pure transit network and absorbs only its
  losses (≈ 4–5 MW). Planner predictions (passive 1–4, price-taker 2–8) held.
- **Static no-reverse-flow (NRF) arms:** feasible at every TSO block in all six runs (prediction held); every run needed
  the consistency pass (TN interface at 1.1 p.u. against the DN setpoint 1.0).

| arrangement | Q (gross) | band over 3 starts |
|---|---|---|
| coordinated (settled x = 0) | 653,873,702.19 | — |
| **price-taker NRF (best)** | **744,770,310.59** | 0.004 |
| passive NRF | 895,049,918.47 | 25,805 |

- **Claim: coordination beats the best static NRF arrangement by +90,896,608 € (13.9 %), determinate** (≈ 1,264× the
  71.9 k€ resolution); prediction "positive" held. Like-for-like verified independently (same evaluation function and
  pricing parameters, settlement excluded both sides, arm Q after the pass); every asymmetry found favours the arms.
- **Where it comes from:** TSO conventional energy **+70.2 M€**, DN flexibility **+20.7 M€**; by year 7.4 / 29.6 / 33.2 M€.
  The price-taker DSOs import the same daily energy as coordinated (within 0.5 % in all 12 blocks) but **mis-time demand
  relative to the hours when transmission renewables are free** — pricing flexibility at π_t rather than λ_t — leaving
  **0.81 TWh of TN renewable output curtailed** (0 coordinated) and re-procuring it from conventional units. This fits
  10 of 12 blocks (2025 Winter just outside the frozen rule; 2025 Autumn undetermined). Night-time vacating is **not**
  uniform (the Winter blocks and 2030 Autumn shift the other way), so the manuscript should say "mis-timed", not "night".
- **Caveats for the manuscript:** transmission "generation cost" is conventional energy at the wholesale price; the
  coordinated solution uses reverse flow at 4 interface-hours (3,308 MWh), worth a first-order ≈ 0.3–0.4 M€ (< 0.5 %)
  — part of the benefit is the value of allowing reverse flow; the passive arm is voltage-inconsistent after its one
  pass (0.049 p.u.) and its interface Q is tie-breaker-dependent at P = 0 hours — neither touches the claim.

### The re-settling campaign (Addendum 57, Decision 3)

| cell | gate (bitwise through) | certified at | s = Q_k\* − Q_old | t_sum at k\* | M (gross) | resolution | verdict |
|---|---|---|---|---|---|---|---|
| F2 challenger | 152 | — (gap clause) | — | −9,300 at cap | — | — | indeterminate |
| F2 incumbent | 172 | — (gap clause) | — | −9,234 at cap | — | — | indeterminate |
| pb_y2030_n9 | 120 | 192 (k0+72) | +27,488 | +453 | 46,706 | 8,485 | determinate |
| pb_y2030_n7 | 125 | 195 (k0+70) | +20,994 | +723 | 49,022 | 8,721 | determinate |
| pb_y2025_n5 | 110 | 167 (k0+57), **G6-flagged** | +22,483 | −587 | 51,263 | 8,718 | determinate |
| pb_y2030_n5 | 122 | 182 (k0+60) | +13,891 | +245 | 43,326 | 8,459 | determinate |
| pb_y2025_n9 | 113 | 174 (k0+61) | +12,212 | +431 | 52,460 | 7,906 | determinate |
| pb_y2025_n7 | 112 | 172 (k0+60) | +11,770 | +223 | 54,941 | 8,631 | determinate |
| yl 2030 | ungated | 169 (k0+62) | — | +493 | 68,499 | 8,294 | determinate |
| yl 2035 | ungated | 167 (k0+61) | — | +577 | 110,958 | 8,261 | determinate |

- **The phase-mismatch risk did not materialise.** The six Phase B neighbours settled **+12 to +27 k€** above their old
  certificates — the same direction and order as x = 0's +15.0 k€ — so their margins against the settled x = 0 **grew**
  from 33–57 k€ to **43.3–54.9 k€**, all determinate in gross and in Q_cc (43.4–55.0 k€). "x = 0 minimises F against its
  Phase B neighbours" holds.
- **The gap closed in all eight certified cells** (|t_sum| 223–723 € at k\*, against 21–46 k€ on the old certificates).
- Every certified cell passed through the new criterion's oscillatory branch (4–5 turning points, period 19–30).

### Predictions against outcomes

| prediction | outcome |
|---|---|
| gates bitwise through k0 (8 gated cells) | **held** 8/8 — including the never-demonstrated ×2 price path |
| tail overlap at N_old in [−1.5, −0.8] × 10⁻⁶ | **held** 8/8 (−1.00 to −1.17 × 10⁻⁶) |
| Phase B: certified at k\* ∈ [k0+50, k0+70] | held 5/6 (n9 at k0+72) |
| Phase B: s ∈ [+8, +30] k€ | **held** 6/6 |
| Phase B: \|s − 14,971\| ≤ 10 k€ | held 5/6 (n9: 12,517) |
| Phase B: gap closes below τ/2 | **held** 6/6 |
| F2 challenger: C\*-like creep, uncertified with p ≥ 0.5 | uncertified — but **not by creep** (drift −2 €/cycle); by the gap clause |
| F2 incumbent: first turning point ≤ k0 + 15 | **held** (k0 + 4) |
| year ladder: k\* ∈ [k0+50, k0+70] | **held** 2/2 |
| NRF feasible at every TSO block; NRF claim positive; sweep ranges | **held** |

## Not confirmed

- The dead-zone release time (≈ 500 cycles) is a first-order extrapolation; the constraint look loaded the challenger's
  models only, not the incumbent's.
- Why the ×2 flexibility price raises the DSO flexibility threshold ≈ 12× relative to x = 0; why the ESS discharges in
  the cheapest hour (H_ess-flat is consistent, not traced).
- The benchmark mechanism is verified hour by hour in one block, at block level in the rest.
- The remaining 37 re-run cells, and the triage items they carry, are still pending under the Addendum 57 rule.
- Record defects, not fixed: the child harness's finish line reads `status=certified` for uncertified runs; one launch
  manifest was committed with mode 755 (`d6333b6f`); several Worker commit messages carry small errors recorded in
  `TASKS.md` rather than amended.
