# P5.15 Addendum 51: the continuation — reproducible bitwise, settles through a damped oscillation; stage 2 not required

**Planner report, 2026-09-27.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 51 (with Addenda 48–50).
- **Frozen spec:** v38 `8bc0ffa6` (v37 `774d083f` superseded, never run); campaign spec `c2b02e21`.
- **Objective convention on every value:** Q = certified `gross_operational_cost`, settlement excluded. Value V = Q(x = 0) − Q(unit) = 236,695.25 €. R_ref = 259,375.33.
- **Status:** nothing is running. Stopped for review, as Addendum 51 ordered.

## Decisions needed

**1. The settling criterion — the early-stop rule is unsound for an oscillating objective.** Addendum 50 ordered an Advisor review of the settling criterion *after* the continuation data; that data now exists and bears directly on it. The frozen early stop (|ΔQ| < 500 €/cycle for 3 cycles) fired at cycle 88 on the steps +327, −1, −301 — which straddle a **turning point** of the oscillation (the fitted maximum at cycle 86.5). **A step bound does not bound the distance to the limit:** a 500 € bound admits an oscillation amplitude of ≈ 1,464 €, and a stop anywhere from cycle 86 on would have returned between −4,190 and −6,948 € depending only on phase. A sound criterion must bound the oscillation's envelope, or span more than a half-period (≈ 9 cycles here). **Recommendation:** proceed to the ordered Advisor review with this finding as its input; it costs no compute. It did not change any decision here (below).

**2. A defect class that recurs — fix it structurally before the next run.** Gate G14 failed because new code (W98's hooks) wrote a boolean flag as the **string `"True"`**: a numpy bool passed through `json.dumps(default=str)`, while the gate checks `is True`. This is the defect found and patched in W74–W76, **reintroduced** because the fix was applied to particular writers rather than made impossible to repeat. The SRP1 benchmark (next in the order) adds new writers and gates. **Recommendation:** one shared JSON-writing helper used by every harness, and a check that rejects `default=str` on any gate artifact — before the benchmark runs.

## Blocked on the author

Nothing blocks. The next items in the order (SRP1 benchmark, Step 5 rows) await the review.

## Changed

- **Stage 1** ran 2026-09-27 07:11 → 16:47 UTC (34,593 s, 7,387 solves, no retries). Evidence `4689475e` (manifests 363/363 and 354/354 re-verified; 347 IPOPT logs hash-recorded). Post-hoc analyses `3fd1c15b`.
- **Stage 2 did not run**, by the frozen rule.
- **G6 as restated** (Optimal + four metrics) passes 80/80 on the continuation.

## Found

### Reproducibility

**The replay reproduced the certified x = 0 run bitwise, 72 of 72 cycles**, including the initial state, every Anderson-acceleration decision (memory clears, safeguard rejections, acceptances, and the final extrapolation at cycle 62), and the tight tail engaging at cycle 64 through the continuation's hooks. **This is the instance's reproducibility result for the manuscript.** It also shows the hooks are transparent under a live transition, not only while idle.

### Settling

After certification the objective **did not decay monotonically; it oscillated and settled.**

| cycle | 72 (cert.) | 73 | 75 | **77** | 80 | 82 | **86** | 88 |
|---|---|---|---|---|---|---|---|---|
| Q − Q72 (€) | 0 | −3,814 | −9,684 | **−12,014** | −10,000 | −7,288 | **−4,190** | −4,492 |

| quantity | frozen (decided stage 2) | post-hoc damped-cosine fit, cycles 72–88 |
|---|---|---|
| descent | **4,492.39 €** (to the early stop) | **L = −6,235 €** (jackknife SE 36.5; estimator range 5.1–6.3 k€) |
| shape | geometric fit **invalid** | half-period 9.15 cycles; per-cycle contraction 0.892; rms 20 € |
| validation | — | fitted on 72–84, predicts 85–88 within 48–107 € |

The early stop recorded 4,492 €, but the run sat 1,742 € above the fitted limit when it stopped. **The post-hoc limit is the better estimate of the settled descent; the frozen figure is what the rule decided on.** Both are far below the stage-2 threshold, so the decision is robust.

**Per-block, all 80 blocks now recorded (Addendum 50 item i, fully recovered):** two motions are superimposed.
- A **one-directional redistribution** — 56 of 80 blocks keep one sign over cycles 73–88 and carry 92.5 % of the block movement, but nearly cancel (DSO7 Summer falls, DSO7 Winter rises, DSO9/DSO5 Summer rise, TSO Summer/Winter fall). W95's DSO7 "lead blocks" belong here and carry only 0.098 of the oscillation.
- The **oscillation itself is carried by Spring (0.818 share) and the TSO (0.847)**, spread over ≈ 30 blocks, coherent by season and agent. It is **not** the block set earlier analyses identified from the truncated top-10.

### The value and R, three ways

| basis | R |
|---|---|
| **at certification** | **0.9126** |
| **after continuation** (x = 0 settled; storage cell's descent bounded above by x = 0's, per Addendum 51) | **[0.8952, 0.9126]** frozen · [0.8885, 0.9126] post-hoc |
| **with the drift as explicit uncertainty** | **[0.888, 0.913]** |

Every figure lies within the ratio resolution (0.1405) of the recorded **R = 0.9331** — indistinguishable. **value − I = −81,262 €** at certification (determinate at 4.5×); continued descent of x = 0 can only make it more negative. **x = 0 is optimal under the baseline at both instances.**

**Manuscript certification statement (multi-scenario instance), proposed:** *"Certified on residuals at cycle 72; the objective then settled through a damped oscillation (half-period ≈ 9 cycles, per-cycle contraction 0.89), descending a further ≈ 6.2 k€ (≈ 7 × 10⁻⁶ relative). The run is bitwise reproducible through certification."*

### Predictions against outcomes

| prediction (Addendum 51) | outcome |
|---|---|
| replay bitwise through cycle 72 | **held** (72/72) |
| step peak within a few cycles of certification (≤ cycle 77) | **held** — the largest descent step was at 72 itself |
| geometric decay, ratio 0.80–0.95 | **refuted** — oscillatory; though the envelope contracts 0.892/cycle, inside the band |
| D_x0 = 20–60 k€ | **refuted** — 4.5 k€ frozen, 6.2 k€ post-hoc |
| stage 2 triggered | **refuted** |
| R_settled ≥ 0.80 | **held** |
| H1 / H2 / H3 | **none fits** — an underdamped oscillation that settles |

## Not confirmed

- **The damped-cosine fit is post-hoc** and rests on 17 points spanning ≈ 1.8 half-periods; model uncertainty exceeds the statistical SE, which is why the estimator range (5.1–6.3 k€) is the honest figure. A slow drift of up to ≈ ±40 €/cycle cannot be excluded.
- **The storage cell was not continued**, by the frozen rule. Its settled descent is bounded above by x = 0's, per Addendum 51 — an assumption supported by it descending 2–2.8× more slowly at certification, not a measurement.
- **The numpy type behind G14's string flag** is inferred from the source and the output, not observed in the running process.
- A 1.2 × 10⁻⁸ discrepancy between the frozen `R_at_certification` and V / R_ref, attributable to R_ref's rounding; immaterial.

## Corrections to the Planner's interim reports

Both were reported to the author before this report and are corrected here:
- **G14's mechanism** was first stated as exact equality of a floating-point sum. It is the stringified boolean above.
- **The oscillation limit** was first stated as ≈ −7,281 € from a three-extremum fit said to "predict the third extremum to within 15 €". That fit was circular — a three-point fit passes through its three points by construction — and it treated cycle 72, a mid-descent point, as a turning point. The proper fit gives −6,235 €.
