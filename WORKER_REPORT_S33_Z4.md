# Worker Report — P5.15 Addendum 15 item 4 (Z4): attribute the ~30-cycle oscillation in s33e2

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 15 item 4 ("The ~30-cycle mode: one zero-solve
attribution (period per channel), <= 1 hour; not blocking."). Bounded zero-solve diagnostic, no
production-code changes, no solves.

## Task received

Attribute the non-monotonic behaviour of `gross_operational_cost` in gate `s33e2` over cycles
101-150 (span 185,571 = 2.8x the 65,383 objective tolerance, 20/50 positive steps, a 16-cycle rise
of +87,052 from cycle 128 to 143, apparent period ~30 cycles): which quantity oscillates, at what
period and amplitude, and whether it is coupled to a channel, the rho/gamma freeze (cycle 30), the
18 network failures, or something else. Full spec in the task message (5 numbered sub-questions).

## Files inspected

- `data/SRP1/Results/P515S33_E2_run/g_baseline.json` (150-row `cycle_trajectory`, all `boyd_*`,
  `rho_*`, `gamma_*`, legacy `primal_*`/`dual_*`, `slack_*` fields)
- `data/SRP1/Results/P515S33_E2_run/network_failures_baseline.jsonl` (18 rows)
- `data/SRP1/Results/P515S33_E2_run/esso_capture/baseline/node{5,7,9}_cycle{001..150}.jsonl`
  (450 files, 288 rows each; `pch`, `pdch`, `pnet`, `class`, `class_bar`)
- `data/SRP1/Results/P515S33_E2_run/stdout_baseline.log` (14,562 lines; `[RECOURSE JUMP]`,
  `[ADMM RHO]`, `[ADMM RHO BOYD]`, `[TSO PROX]` blocks)
- `data/SRP1/Results/P515S32_run/g_baseline.json` (fixed gamma, no freeze, 150 cycles, contrast)
- `data/SRP1/Results/P515S31C_run/g_baseline.json` (cap 90, contrast, used as a bonus check)
- `p515_s33e2_z1_esso_drift.py`, `p515_s33e2_z2_price_alignment.py`, `p515_s33e2_channel_analysis.py`,
  `p515_s33e2_evaluate.py` (existing Z1/Z2/analysis scripts, for output/guard conventions — not
  re-run, not modified)
- `p513_solve_profile_guard.py` (guard API)
- `PLANNER_BRIEF_2026-09-13.md` Addendum 15 (task authority), `REVISION_CONTEXT.md` (s33e2 context)

## Files modified

New files only; nothing existing was edited or re-run.

- `p515_s33e2_z4_oscillation.py` (new script, 734 lines)
- `data/SRP1/Results/P515S33/Z4/z4_oscillation.json` (new output, write-once; directory did not
  exist before this task)
- `data/SRP1/Results/P515S33/Z4/z4_manifest_sha256.json` (new; sha256 of the two files above)
- `WORKER_REPORT_S33_Z4.md` (this file)

## Changes made

`p515_s33e2_z4_oscillation.py` performs, with `SolveProfileGuard(permitted=(), ...)` armed for the
whole run and `guard.verify(expected_solves=0)` checked before any output is written:

1. **Cost characterisation** (windows 31-150, 61-150, 101-150): centered-moving-average detrend
   (window 9, primary) and linear detrend (robustness check); autocorrelation top-3 peaks; DFT
   top-3 periods/amplitudes; amplitude and peak-to-peak reported as multiples of the run's own
   `objective_tolerance`.
2. **Extrema/envelope decay analysis** (added beyond the spec's minimum, because the amplitude
   visibly falls by more than an order of magnitude across the window, which invalidates the
   stationarity assumption behind ACF/DFT): peak/trough sequence (located on a further 3-cycle-
   smoothed copy to reject single-cycle noise), period from peak/trough spacing, zero-crossing
   period, and an OLS fit of ln|amplitude| vs cycle to extract a decay rate and half-life.
3. **Same-period survey** (window 61-150): ~40 series (every `boyd_{v,pf,ess}_{r,s,norm_x,norm_z,
   norm_y,primal_ratio,dual_ratio}`, legacy `primal_*`/`dual_*`/`*_mean`, `recourse`,
   `recourse_change`, and ESSO throughput/`Sigma|pnet|` aggregate and per-node computed from the
   captures) — DFT-dominant period, "shares cost's dominant period" flag (+/-15%), and
   cross-correlation lag/sign vs the cost residual.
4. **Coupling tests**: (a) failure-cycle circular phase-lock (observed resultant length R vs a
   5,000-draw permutation null) and a Mann-Whitney U test on the signed cost step at failure vs
   non-failure cycles; (b) freeze coupling via 5 non-overlapping 28-30-cycle windows spanning
   cycle 30; (c) lagged Pearson correlation (lags 0-3) of the signed cost step against 15
   per-channel Boyd quantities; (d) block localisation by parsing the stdout `[RECOURSE JUMP]
   ... Aggregate signed changes:` blocks (regex-based, cycle -> {TSO, DSO node=5/7/9, SALVAGE}
   delta); (e) bound/active-set coupling via `slack_consensus_*`, `slack_stationarity_*`,
   `slack_objective`, and a per-cycle count of `class`/`class_bar == 'indeterminate'` entries in
   the ESSO captures.
5. **Contrast**: identical cost characterisation applied to s32 (windows 31-150, 61-150) and, as a
   bonus, s31c (window 31-90, its only available window at cap 90).

All formulas are reproduced in the script's module docstring (CLAUDE.md evidence rule).

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s33e2_z4_oscillation.py
```
Run twice during development (output directory removed and rebuilt each time while iterating on
the extrema-detection smoothing and the window-slicing bug below); the committed artifact is from
the final run. No other commands solve or touch production code. A stand-alone sanity check
(`total rows 129,600`, 1.21s) confirmed the ESSO capture read is complete and fast before wiring
it into the guarded script.

One implementation bug caught before commit: the ESSO throughput/`Sigma|pnet|` series were first
computed over the full 150-cycle range but fed into the 61-150 survey window, causing a
`numpy` shape-mismatch crash (arrays of length 150 vs 90). Fixed by computing them over the
matching 61-150 range only; re-run confirmed no other shape errors and the guard's
`blocked_solve`/`blocked_exec` counts stayed at 0 throughout.

## Results

**Guard**: `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`,
`verify_failures: []` — zero solves, as declared, for every run.

**What oscillates and at what period.** `gross_operational_cost` (== `recourse`, salvage is 0 in
this run) is the primary oscillating quantity. Three independent, MA-detrend-based period
estimators converge on **~20 cycles**, not the ~30 estimated by eye from the two local minima at
cycles 106 and 128 — the "16-cycle rise, cycle 128 to 143" the Planner flagged is the second half
of one ~20-cycle cycle riding on a still-decaying trend, not a full new period:
- DFT (31-150 window, n=120, primary): top-3 periods 20.0 / 30.0 / 24.0 cycles, amplitudes
  28,157 / 20,707 / 19,921 — a **broad** spectral band (top-3 amplitudes within 30% of each
  other), not a sharp single tone.
- Peak/trough spacing (31-150 window, after rejecting sub-9-cycle noise wiggles): peak spacings
  15, 8\*, 16, 20, 24, 21, 5\* cycles (\*= short entries are secondary wobbles/edge effects, not
  full periods); trough spacings 6\*, 16, 12, 21, 22, 16. Excluding the starred short entries,
  the mean is ~20-21 cycles.
- Zero-crossing period (amplitude-independent): **20.3 cycles** (31-150 window), **21.2 cycles**
  (61-150 window).
- Autocorrelation is **weak** everywhere it is computed: the largest ACF value at any window is
  0.28 (31-150, lag 23), close to the ~2/sqrt(n) white-noise significance threshold (~0.18-0.22
  at these window lengths). This is a genuine, real, but *weakly* periodic signal, not a strong
  resonance.

**It is a decaying transient, not a sustained oscillation.** The peak/trough amplitude sequence
fits `ln|amplitude| = ln(A0) - k*cycle` with **R^2 = 0.87** (31-150 window) and **R^2 = 0.84**
(61-150 window), giving a **multiplicative decay of ~0.966-0.967 per cycle (~3.4-3.9%/cycle)** and
a **half-life of ~20 cycles**. Raw peak amplitudes fall from 242,257 (cycle 34, 3.7x tolerance) to
2,835-6,201 (cycles 138-143, 0.04-0.09x tolerance) — more than a 20x reduction over the window.

**Amplitude vs tolerance, by window** (MA-detrended residual, peak-to-peak, as multiple of the
run's own `objective_tolerance` at that window's terminal cycle):
| window | raw span / tol | MA-residual ptp / tol | DFT dominant-sinusoid amplitude / tol |
|---|---|---|---|
| 31-150 | 40.3x | 5.15x | 0.43x |
| 61-150 | 10.3x | 2.51x | 0.17x |
| 101-150 | **2.84x** (matches Planner's 185,571/65,383) | **0.38x** | **0.086x** |

Most of the Planner-quoted 2.8x-tolerance raw span in 101-150 is the still-not-fully-decayed tail
of the transient (the trend a 9-cycle moving average removes), not fresh periodic content: once
detrended, the residual is well under 1x tolerance, and the single dominant sinusoid's amplitude
is under 0.1x tolerance.

**Same period elsewhere.** In the 61-150 window (where the transient's SNR is already reduced,
the DFT spectrum is broad/flat across periods 10-18 cycles rather than sharply peaked at 20 — see
Limitations below). Series sharing the window's ~11.25-cycle nominal dominant period (+/-15%):
`boyd_v_r`, `boyd_v_norm_x`, `boyd_v_norm_z`, `boyd_v_primal_ratio`, `primal_v`, `dual_v`,
`primal_v_mean`, `boyd_pf_r`, `boyd_pf_primal_ratio`, `primal_pf_mean`, `dual_pf_mean`,
`recourse` (trivial, ccf=1.0, lag 0, since recourse == cost here). The tightest phase alignment is
**`boyd_v_norm_x`/`boyd_v_norm_z`** (V channel's primal/dual iterate norms) at **lag 1, ccf =
-0.78** — nearly in phase (one-cycle lag) and strongly anti-correlated with the cost residual.
No ESS-channel or ESSO-throughput series appears in the sharing list in this window (ESS *does*
show up as the strongest lag-0 correlate of the raw cost *step*, see channel coupling below — a
different, non-spectral test).

## Coupling verdicts

- **Failure coupling — NOT supported, marginal at best.** Circular phase-lock test (14 failure
  cycles in 31-150, period=20, 5,000-draw permutation null): R_observed = 0.459, empirical
  p = 0.035. In 61-150 (9 failure cycles, period=11.25): R = 0.226, p = 0.622. Only one of two
  windows crosses p<0.05, and it does not survive even a mild Bonferroni correction for testing
  two windows (threshold 0.025). Mann-Whitney U test on the signed cost step at failure vs other
  cycles: p = 0.349 (31-150), p = 0.904 (61-150) — no detectable difference in step magnitude or
  sign. **Verdict: no robust evidence of failure-cycle phase-locking or magnitude coupling.**
- **Freeze coupling — present but not causal.** The freeze (cycle 30) is immediately followed
  (cycles 31-35) by the transient's single largest swing (residual up to +242,257, 3.7x
  tolerance) — but the *same period* (DFT-dominant 20.0 cycles, 31-150 window) is present in s32,
  which has **no freeze at all** (`gamma_policy: null`, `freeze_after_cycle: null`, adaptive rho
  for all 150 cycles), and s32's relative amplitude is **larger**, not smaller, than s33e2's
  (`residual_ptp_over_tolerance` 14.40 vs s33e2's 5.15 in 31-150; 5.42 vs 2.51 in 61-150; see
  Contrast below). **Verdict: the ~20-cycle mode pre-exists the freeze mechanism; freezing
  rho/gamma does not create it and, if anything, correlates with a smaller relative amplitude than
  the non-frozen control. What the freeze does correlate with is *visibility*: before cycle 30 the
  monotonic ~10^8-scale contraction (residual std ~91M in the 3-30 window, a detrending artifact
  of that non-stationary transient, not a comparable oscillation) masks the much smaller
  (~10^5-scale) mode; once the freeze lets the trend flatten quickly, the pre-existing mode
  becomes the dominant visible signal.**
- **Channel coupling.** Strongest lag-0 Pearson correlations of the signed cost step against raw
  channel levels (61-150 window): `boyd_ess_s` (r=-0.618), `boyd_ess_dual_ratio` (r=-0.618),
  `boyd_v_s` (r=-0.598), `boyd_v_dual_ratio` (r=-0.598), all at lag 0 or 1. **Verdict: moderate,
  concurrent coupling shared between the V and ESS channels' dual-residual quantities; PF is
  notably absent from the top correlates. No single channel is a clean unique driver** — V and ESS
  move together with the cost step within the same cycle, consistent with both being downstream of
  the same per-cycle primal update rather than one driving the other.
- **Block localisation — NOT DETERMINABLE from committed artifacts.** The `[RECOURSE JUMP]`
  per-block breakdown in `stdout_baseline.log` is gated by a jump-detection threshold in
  production and is emitted only through **cycle 62** (cycles 2-46, 54-56, 60-62; none after).
  Coverage in the requested windows: **0 of 50 cycles in 101-150**, **2 of 90 in 61-150**
  (cycles 61, 62 only — at those two points TSO dominates the aggregate signed change, -146,406
  and -138,433, against DSO node=5/7/9 contributions an order of magnitude smaller). Two points
  cannot establish periodicity. **What would be needed**: the per-cycle block-level recourse
  decomposition captured unconditionally every cycle (or persisted to a structured artifact),
  not gated behind the jump-detection print, for the full 150-cycle run.
- **Bound/active-set coupling.** `slack_stationarity_v`, `slack_consensus_ess`, and
  `esso_indeterminate_class_count` nominally match the (weak, broadband — see Limitations)
  61-150-window dominant period within +/-15%; `slack_consensus_v`, `slack_consensus_pf`,
  `slack_stationarity_pf`, `slack_stationarity_ess`, `slack_objective` do not. Given how broad the
  underlying spectral peak already is in this window (section 3 above), this is weak
  corroborating evidence, not a clean resonance match. **What is not serialized**: per-entry
  bound-crossing events (voltage/branch limit or ESS `pch`/`pdch` bound activity) keyed by
  (node, year, day, period) with a timestamp/cycle — only the aggregate `slack_*` scalars and the
  ESSO complementarity `class`/`class_bar` flags are available at per-cycle granularity.

## Contrast with s32 (fixed gamma, no freeze)

| window | s33e2 raw span/tol | s33e2 residual ptp/tol | s33e2 DFT top period | s32 raw span/tol | s32 residual ptp/tol | s32 DFT top period |
|---|---|---|---|---|---|---|
| 31-150 | 40.3x | 5.15x | **20.0** | 151.1x | **14.40x** | **20.0** |
| 61-150 | 10.3x | 2.51x | 11.25 | 32.9x | **5.42x** | 18.0 |

s32's DFT-dominant period in 31-150 is **exactly** 20.0 cycles, matching s33e2's. s32's
autocorrelation is also somewhat stronger (0.29 at lag 19 in 61-150 vs s33e2's 0.21 at lag 12) —
still weak in absolute terms, but the cleanest ACF signal found in either run. **This is the
central finding for the freeze-vs-pre-existing question: the ~20-cycle mode is a property of this
ADMM configuration's dynamics common to both runs, and s32 (no freeze) shows it at 2-3x the
relative amplitude of s33e2 (with freeze).** As a bonus (not required by the task), s31c (cap 90,
also no freeze) was checked over its only available window (31-90): ACF is essentially flat noise
(top value 0.013) and `residual_ptp_over_tolerance` is 58.7x — inconclusive, most plausibly because
s31c's available window is still dominated by non-stationary contraction, not because the mode is
absent.

## Validation

- The script's own printed cross-checks reproduce the Planner's numbers exactly: 101-150 window,
  n=50, 20/50 positive steps, raw span 185,571.03 (matches "185,571" and "2.8x" the 65,383.2
  tolerance to the reported precision).
- `recourse_change` was cross-checked against `gross_operational_cost[k] - gross_operational_cost[k-1]`
  at a sample cycle (k=101): -12,810.60 both ways.
- Guard counts verified zero solves on every run (see Results).
- Code executes correctly and the requested diagnostic runs to completion; this establishes the
  attribution evidence above, not a fix to the underlying non-monotonicity (out of scope for this
  task).

## Unexpected findings

- The Planner's ~30-cycle estimate (from eyeballing cycles 106->128->143) is close to but somewhat
  above every independent period estimate this diagnostic produced (18-25 cycles across windows
  and methods, converging near 20 in the higher-SNR 31-150 window and via zero-crossings). The
  eyeballed 30-cycle figure appears to conflate the tail of one ~20-cycle oscillation with residual
  trend curvature.
- The mode is present, with comparable or larger relative amplitude, in s32 which has **no**
  rho/gamma freeze — this was not anticipated going in (the task framed the freeze as one of the
  primary coupling hypotheses) and is the strongest single piece of evidence in this diagnostic:
  it rules out the freeze as the *origin* of the oscillation, though the freeze still coincides
  with when the mode becomes the dominant visible signal in s33e2 (by unmasking it from the prior
  monotonic contraction).
- The `[RECOURSE JUMP]` block-level diagnostic's hard cutoff at cycle 62 was not previously
  documented as a coverage gap for this kind of late-cycle question; it is a genuine capture gap
  (CLAUDE.md evidence rule twelve: per-cycle state should be a default, not an afterthought).

## Limitations / what is not determinable

- **Spectral resolution and stationarity.** DFT/ACF period estimates in the 61-150 (n=90) and
  especially 101-150 (n=50) windows are on a coarse frequency grid (period bins at n/k) and the
  underlying signal's amplitude is itself decaying within the window, which biases both estimators
  toward whatever frequency dominates the higher-amplitude early part of the window. The 31-150
  window (n=120, includes the highest-SNR part of the transient) is the most reliable single
  estimate; even there the spectral peak spans periods 20-30 within 30% of each other in amplitude,
  so "the" period is better stated as "very likely low-to-mid twenties cycles, on a broad,
  weakly-resonant band" than as a single sharp number.
- **Block localisation is not recoverable** for the requested windows from committed artifacts (see
  above); this is stated as a negative result, not inferred.
- **Fine-grained bound/active-set events are not recoverable** below the aggregate `slack_*`
  scalars and the ESSO capture's per-entry `class`/`class_bar` flags (see above).
- The channel-coupling and same-period-elsewhere batteries used lag-0..3 (channel coupling) and
  a single +/-15% period-match criterion (survey), which are simple, stated, reproducible choices,
  not an exhaustive search; a stronger claim would need a formal multi-channel state-space /
  eigenvalue analysis of the linearized ADMM map, which is out of scope for a zero-solve,
  <=1-hour diagnostic.

## Questions for Planner

1. Given the mode pre-exists the freeze (present in s32 at larger relative amplitude) and is
   decaying with ~20-cycle half-life, is there any remaining reason to treat 101-150's
   non-monotonicity as a stopping-criterion risk, or does "decaying transient, already down to
   ~0.09-0.38x tolerance by 101-150" close this item as anticipated by Addendum 15 item 4's framing
   ("not blocking")?
2. Do you want the `[RECOURSE JUMP]` unconditional-per-cycle capture gap (block localisation) added
   to a future-run capture checklist now, or deferred until/unless a stage needs it?

## Solve-profile guard counts (every run of the committed script)

`{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`,
`verify_failures: []`.
