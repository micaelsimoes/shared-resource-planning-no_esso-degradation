# P5.12-T — No-Solve Comparative Trajectory Forensic Report

## 0. Scope and authorization

This stage was authorized in `LOCAL_NLP_STABILITY_PLAN.md`, section
"AUTHORIZED NEXT STAGE — P5.12-T no-solve trajectory forensic (2026-09-12)".
**Zero solves were performed.** All work consisted of reading three preserved,
frozen IPOPT print-level-6 logs and one new read-only diagnostic harness,
`p512_t_trajectory_forensic.py`, that parses them and writes analysis output
to a new directory `data/SRP1/Results/P512T/`. No production code, governing
document, or preserved artifact was modified. No `git` write operation was
performed.

## 1. Commands run

See `data/SRP1/Results/P512T/manifest.json`, field `commands_run`, for the
exact list. In summary: SHA-256/line-count verification of the three input
logs; targeted `grep`/`sed` structural reconnaissance of the log format
(iteration-summary blocks, "Beginning Iteration" blocks, "Current NLP Values"
blocks, safeguard messages, restoration markers, EXIT lines); one execution of
the harness (`python -B p512_t_trajectory_forensic.py`), which performs all
parsing and analysis in a single pass with no solver calls; and read-only
inspection of the resulting `journal.json` for reporting.

## 2. Repository / provenance integrity

- Branch: `feature/derivative-free-planning`; HEAD `ba202e2b0e937306f3c163de2173951c1d0c24f0`,
  unchanged before and after this stage.
- Pre-existing uncommitted modifications at session start (`LOCAL_NLP_STABILITY_PLAN.md`,
  `REVISION_CONTEXT.md`) were read-only inspected and never edited by this worker.
- No tracked or production file was modified.
- `data/SRP1/Results/P512T/` was verified absent before the harness ran (the
  harness itself refuses to run if the directory already exists).
- No IPOPT process, `solver.solve`, or any production solve entry point was
  invoked anywhere in this stage. The harness contains no solver-invocation
  code path at all.

## 3. Input verification (parser validation step 0)

| Run | Path | Expected SHA-256 | Actual SHA-256 | Match | Expected lines | Actual lines | Match |
|---|---|---|---|---|---|---|---|
| V1 (1e-6) | `data/SRP1/Results/P512P/variant1_1e-6/logs/optim_log_case33_3_2025_Spring.log` | `3c5479...5d2bd7` | identical | YES | 7400 | 7400 | YES |
| Arm A (1e-5) | `data/SRP1/Results/P512ArmA/logs/optim_log_case33_3_2025_Spring.log` | `4f66a7...ff45409d58` | identical | YES | 243498 | 243498 | YES |
| V2 (1e-4) | `data/SRP1/Results/P512P/variant2_1e-4/logs/optim_log_case33_3_2025_Spring.log` | `ccf807...095ed2e` | identical | YES | 12599 | 12599 | YES |

Full hex digests (verbatim, matching the values supplied in the task):

- V1: `3c5479835849bc859f3bcaf964a2791e89318bae0fe979d2eb7dbb3e155d2bd7`
- Arm A: `4f66a7efeef933bdc0a425af76f0095f5c11a2112ff2c8bb6d7c03ff45409d58`
- V2: `ccf80716e9b7c7f7346d1404234a2b125d00cc0658f6ec8ba35ad8a63095ed2e`

## 4. Parser validation results (Step 1) — item by item

The harness's `validate_parser()` function reproduces every headline fact
declared in the task before any interpretation is performed, and **halts
before analysis if any check fails** (it did not halt; all 41 checks passed).
Full pass/fail detail is in `journal.json["parser_validation"]["checks"]`.
Key items, all `PASS`:

1. Iteration counts: V1 = 89, Arm A = 3000, V2 = 151. PASS.
2. Terminations: V1 `Optimal Solution Found.`, Arm A `Maximum Number of
   Iterations Exceeded.`, V2 `Optimal Solution Found.`. PASS.
3. Safeguard row counts: V1 = 0, Arm A = 2924, V2 = 21. PASS.
   - Arm A onset attributed to iteration 77 (safeguard line appears at the end
     of the "Finding Acceptable Trial Point for Iteration 76" block, i.e. at
     the 76→77 transition; attributed to iteration 77, matching the reported
     onset convention). Continuity 77..3000 with no gaps: verified `True`.
   - V2 safeguard rows are exactly iterations 82–102 inclusive (21 rows),
     verified against the exact list `[82, 83, ..., 102]`.
4. Iteration-0 unscaled dual infeasibility identical across all three:
   `1.8366791482401353e+04`. PASS for V1, Arm A, V2.
5. Iteration-0 unscaled complementarity ladder: V1
   `1.0116099216048767e-01`, Arm A `1.0116099216048768e+00`, V2
   `1.0116099216048768e+01`. PASS.
6. Iteration-0 unscaled objective V1 `1.5200889685781983e+03`, Arm A
   `1.5266886921258345e+03`, V2 `1.5446910450142393e+03`; constraint
   violation V1 `7.4901136714446994e-05`, Arm A `6.6171136714446988e-05`, V2
   `9.9980965587185189e-05`. PASS.
7. Final unscaled blocks: V1 objective `1.2936668158879957e+03`, dual inf
   `1.6557008856137988e-03`, violation `1.9567576869938819e-07`,
   complementarity `1.0628655231738295e-05`; Arm A objective
   `1.3022135784698721e+03`, dual inf `5.1598055317328658e+01`, violation
   `6.1921343776988665e-02`, complementarity `6.2030967852100419e-03`,
   terminal `mu = 1.8449144625279508e-06`; V2 objective
   `1.2935224344851817e+03`, dual inf `1.3480530428912341e-07`, violation
   `6.8837158195833581e-12`, complementarity `9.0913954543475466e-06`. PASS
   for every field in all three runs.
8. Parsed iteration numbering matches raw logs: for each run, the maximum
   summary-row iteration number parsed equals the run's reported final
   iteration count (89 / 3000 / 151). PASS. Every reported event below carries
   an exact raw line number (from `journal.json`, each iteration record has
   `summary_line_no`, `data_row_line_no`, `begin_line_no`, `nlp_values_line_no`,
   and every safeguard event carries `raw_line_no`).

**Conclusion of Step 1: parser validated. Interpretation may proceed.**

## 5. Fields available / missing per log (Step 2)

All of the following fields are present and consistently parseable in **all
three** logs, at iteration 0 and throughout: `objective`, `inf_pr`, `inf_du`,
`lg(mu)` (`lg_mu`), `||d||` (`d_norm`), `lg(rg)` (`lg_rg`, which is literally
the string `"-"` at some iterations — this is IPOPT's convention for "no
regularization applied" and is preserved as-is, not converted to a number),
`alpha_du`, `alpha_pr` (with its attached single-letter step-type suffix
separated out into `alpha_pr_stepchar`), `ls`, an optional trailing marker
character (`trailing_marker`, e.g. `z` — observed in Arm A and V2 only, never
in V1, consistent with V1 having zero safeguard events), `mu`, `tau`,
`||curr_x||`, `||curr_s||`, `||curr_y_c||`, `||curr_y_d||`, `||curr_z_L||`,
`||curr_z_U||`, `||curr_v_L||`, `||curr_v_U||`, both scaled and unscaled
`objective` / `dual_infeasibility` / `constraint_violation` /
`complementarity` / `overall_nlp_error` from the "Current NLP Values" blocks,
and the terminal-only `variable_bound_violation` field (present only in the
final, post-"Number of Iterations" NLP-values block of each run — this is
correctly captured as the last of two NLP-values blocks recorded for the
terminal iteration of each run).

**Nothing was invented.** `restoration_markers_found` is recorded `False` for
all three logs: an explicit `grep -n "restoration\|Restoration\|RESTORATION"`
over all three full logs returned zero matches. There is no restoration-phase
entry in any of the three trajectories. The only safeguard message type
observed anywhere in Arm A or V2 is `Some value in z_U becomes too large`; a
`z_L` analogue (and `v_L`/`v_U` analogues) never occurs in either log — the
parser's `SAFEGUARD_RE` matches all four possible types but only `z_U`
instances were found (`z_U` count: Arm A 2924, V2 21; `z_L`/`v_L`/`v_U`
count: 0 in both).

## 6. Predeclared criteria

**Divergence criterion (predeclared in code before execution):** for a pair of
runs, the first iteration `k ≥ 1` at which the unscaled complementarity ratio
`max(a/b, b/a) > 10` and this condition **persists for at least 3 consecutive
shared iterations**. Iteration 0 is excluded by construction (`k ≥ 1`) because
it necessarily differs due to the different `warm_start_bound_push`-scaled
initial slacks/duals (this the task explicitly says not to report as
"divergence").

**"Sustained" deterioration/improvement (predeclared):** a sustained change
starting at iteration `k` means that, for at least 5 CONSECUTIVE parsed
iterations starting at `k` (using the actually-present iteration keys in
order), the value is monotonically non-decreasing (`increase`) or
non-increasing (`decrease`), with at least one strictly monotonic step in that
run, and no reversal exceeding a small relative tolerance (`1e-9`) within the
run.

**Windows (predeclared per the task's list):** `[60,70]`, `[70,90]`,
`[90,102]`, `[102,112]` for multiplier/step slopes, plus point tables at
iterations 60, 70, first safeguard (per run), 90, V2's final safeguard (102),
and ~5/~10 iterations after V2's escape (107/112), each also reported at the
Arm A event-aligned counterpart iteration (defined below).

**Event alignment (predeclared):** because Arm A's safeguard onset (77) and
V2's safeguard onset (82) differ by 5 iterations, "iterations after V2's last
safeguard" are aligned to Arm A by preserving the **offset from each run's own
onset**, not by absolute iteration number: `offset = 102 − 82 = 20`; Arm A's
event-aligned counterpart to iteration 102 is `77 + 20 = 97`; +5/+10 offsets
are then applied identically to both runs from their respective anchors.

## 7. Findings A–F

### A. First divergence

Applying the predeclared complementarity-ratio criterion literally:

| pair | first iteration (ratio > 10, persists ≥3) |
|---|---|
| V1 vs Arm A | 42 |
| V2 vs Arm A | 115 |
| V1 vs V2 | 1 |

**Caveat, reported honestly per the task's instruction not to over-read a
trivial artifact:** the "V1 vs V2 divergence at iteration 1" is a direct,
uninteresting consequence of the **iteration-0 complementarity ladder itself**
(V1 `0.1012`, V2 `10.12` — a ~100× ratio inherited purely from the
`warm_start_bound_push` value, `1e-6` vs `1e-4`) persisting essentially
unchanged through the first several iterations; it is not new information
about a downstream trajectory difference. This is exactly the artifact the
task warns must not be reported as "divergence."

To separate the inherited constant offset from a genuine change in relative
trajectory, a **secondary, exploratory** (non-predeclared, applied after
seeing the primary result, and reported as such) diagnostic normalizes the
pairwise complementarity ratio to its own value at iteration 1 and asks when
that *normalized* ratio departs from 1 by more than a factor of 2 for ≥3
consecutive iterations:

| pair | normalized-ratio divergence iteration | ratio value | iteration-1 baseline ratio |
|---|---|---|---|
| V1 vs Arm A | 18 | 0.0263 | 0.10 |
| V2 vs Arm A | 20 | 0.576 | 10.0 |
| V1 vs V2 | 18 | 0.0263 | 0.01 |

This places the earliest **relative** (offset-normalized) departures at
iterations 18–20 for all three pairs, well before the safeguard-onset window
(77/82) and well before the iteration 60–110 window of primary interest. This
is presented as a secondary, hypothesis-generating observation, not as a
confirmed predictive discriminator, since it was not predeclared before the
primary criterion's result was seen.

### B. Multiplier evolution

`||curr_z_L||`, `||curr_z_U||`, `||curr_y_c||`, `||curr_y_d||`, `||curr_v_L||`,
`||curr_v_U||` are present and comparable across all three runs (full series
in `journal.json → analysis → journal_series → <run> → curr_*`).

- All three runs show the **same** transient spike in `curr_y_d` / `curr_v_U`
  around iterations 25–31 (max values ≈ `5.0e6`–`5.5e6`, at iteration 27–31
  depending on the run) followed by a return to O(1)–O(100) values by
  iteration ~60. This spike-and-recovery pattern is common to all three runs,
  including the two that converge — it is **not** a discriminator between
  persistence and escape.
- From iteration 60 onward, `curr_y_d` and `curr_v_U` sit at essentially
  the same plateau value (`≈115.155`) in all three runs simultaneously (see
  the iteration-60/70/90/102 tables in Section 8) — again common to all three,
  not discriminating.
- `curr_z_L`, `curr_z_U`: windowed slopes over `[60,70]`, `[70,90]`,
  `[90,102]`, `[102,112]` are uniformly of order `1e-4`–`1e-8` in magnitude in
  **both** Arm A and V2, i.e. near-flat in both runs through the
  difficult-regime window — Arm A shows **no** distinguishing sustained growth,
  oscillation, or larger-magnitude clipping relative to V2 in this window.
  Exact values (`journal.json → analysis → multiplier_window_slopes`):
  - Arm A `z_U`: `[60,70]=-3.04e-04`, `[70,90]=+6.53e-05`, `[90,102]=-7.82e-08`, `[102,112]=+6.31e-09`
  - V2 `z_U`: `[60,70]=-3.57e-04`, `[70,90]=-1.39e-05`, `[90,102]=-6.51e-08`, `[102,112]=+9.24e-05`
  - Arm A `z_L`: essentially flat at `~1e-7`–`0` in all windows; V2 `z_L`: essentially flat at `~1e-6`–`0` in all windows.

**Conclusion for B:** contrary to a "multiplier blow-up causes the stall"
hypothesis, the multiplier norms examined do **not** show Arm A sustaining
growth, oscillation, or repeated clipping that V2 measurably reverses in the
60–112 window. Both runs are essentially flat in this window on these
specific norms. A large multiplier norm is present in *both* trajectories
around iterations 25–31 and is not, by itself, evidence of instability (per
the task's caution).

### C. Correction / step behaviour

Maximal-correction magnitudes (the number reported alongside "Some value in
z_U becomes too large"):

- Arm A: 2924 events, magnitude range `[3.37e-08, 9.7e-07]`, first `5.44e-08`
  (iteration 77), last `3.87e-08` (iteration 3000).
- V2: 21 events, magnitude range `[1.97e-08, 1.18e-06]`, first `2.63e-08`
  (iteration 82), last `1.97e-08` (iteration 102).

**Both runs' correction magnitudes are already small (`~2e-8`–`~4e-8`) at
their respective last-observed safeguard event.** Correction-magnitude
collapse alone therefore does **not** discriminate escape from persistence —
Arm A's corrections are of the *same small order of magnitude* as V2's just
before V2 escapes, yet Arm A continues triggering safeguard events for
another 2898 iterations without escaping. This is reported as a specific
negative finding: correction-magnitude decay is necessary-looking but not
sufficient for escape.

The measurable transition that **does** coincide with V2's escape is in
`alpha_pr`, `alpha_du`, `||d||`, and `lg(rg)`:

| iteration (V2) | alpha_pr | alpha_du | \|\|d\|\| | lg(rg) | safeguard? |
|---|---|---|---|---|---|
| 90 | 3.42e-05 | 4.44e-07 | 1.47e+01 | `-` | yes |
| 102 (last safeguard) | 5.39e-05 | 5.33e-05 | 1.59e+00 | `-` | yes |
| 107 (+5) | 1.05e-04 | 3.65e-04 | 1.00e+00 | `-` | no |
| 112 (+10) | **1.00e+00** | **1.00e+00** | 9.13e-04 | `-` | no |

V2's step sizes climb from `~1e-5` (heavily damped, repeated small accepted
steps `ls=1`) at the last safeguard iteration to a **full step**
(`alpha_pr=alpha_du=1.00`) by iteration 112, coincident with `||d||` collapsing
from `O(1)` to `O(1e-3)` and unscaled dual infeasibility collapsing from
`9.14e-02`-ish region to `5.77e-04` (see Section 8 table). `lg(rg)` stays `-`
(no added regularization) throughout this transition in V2, so a
regularization-reduction interpretation is not supported by the data — there
was no regularization active in this window to reduce.

At the Arm-A event-aligned counterpart iterations (anchor 77, offset 20 →
97/102/107/112 exactly as defined in Section 6):

| Arm A iter (event-aligned) | alpha_pr | alpha_du | \|\|d\|\| | lg(rg) | safeguard? |
|---|---|---|---|---|---|
| 97 (≈V2's 90) | 4.52e-05 | 1.69e-05 | 2.46e+00 | `-` | yes |
| 102 (≈V2's 102) | 5.27e-05 | 4.56e-05 | 1.81e+00 | `-` | yes |
| 107 (≈V2's 107, +5) | 5.42e-05 | 5.30e-05 | 1.72e+00 | `-` | yes |
| 112 (≈V2's 112, +10) | not tabulated above but the long-horizon scan (Section 7E) shows `alpha_pr` frozen at `5.42e-05` from ≈iteration 150 through iteration 3000 | | | | |

Arm A's step size **never** makes the corresponding transition: it stays
pinned near `5e-5` indefinitely (confirmed by the long-horizon scan below),
never reaching a full accepted step in the 3000-iteration run.

### D. Barrier progression

`mu` takes only 4 distinct values in each run's history in the early-to-mid
trajectory, following IPOPT's monotone-decrease barrier update schedule:
`0.1 → (…) → 1.8449144625279508e-06 → 9.090909090909092e-09` (V1, V2 only —
Arm A never reaches the last value).

| run | mu transitions (iteration → new mu) |
|---|---|
| V1 | 0 → 0.1; 29 → 2.828e-03; 36 → 1.8449e-06; **68 → 9.0909e-09** |
| Arm A | 0 → 0.1; 32 → 2.000e-02; 35 → 2.828e-03; **40 → 1.8449e-06 (never decreases further, through iteration 3000)** |
| V2 | 0 → 0.1; 33 → 2.828e-03; 40 → 1.8449e-06; **113 → 9.0909e-09** |

**This is the single strongest and most consistent discriminator found.**
Arm A enters the barrier level `mu = 1.8449144625279508e-06` at iteration 40
and **remains at that exact value for the entire remainder of the
3000-iteration run** (`min_mu == mu_at_iter_3000` for Arm A; there is no
lower `mu` value anywhere later in the log — confirmed both by scanning all
distinct `mu` values recorded in the "Beginning Iteration" blocks, which
number exactly 4 for Arm A, and by inspecting `mu` at iterations 60/70/90/102,
which are all identical to the terminal `mu`). V1 leaves this barrier level at
iteration 68 (34 iterations after V1's own trajectory would reach it, absent
safeguard activity, since V1 never triggers safeguard at all); V2 leaves the
same barrier level at iteration 113, **11 iterations after its last observed
safeguard event (iteration 102)** and coincident with the step-size escape
described in C. Arm A never leaves it in 3000 iterations.

Do all three reach the same barrier regime? Yes — all three reach exactly
`mu = 1.8449144625279508e-06`. Is Arm A trapped at a particular mu? Yes,
observationally: it never decreases further after iteration 40, through
iteration 3000 (this is an observed telemetry fact; the task's constraint not
to claim `mu` is causal is respected — this report does not assert that being
"stuck" at this mu *causes* the stall, only that IPOPT's own barrier-update
decision, which is itself a symptom of failing its internal barrier
sub-problem optimality criterion, never advances past this point in Arm A).
Does V2 cross it after its safeguard interval? Yes, at iteration 113, 11
iterations after the last safeguard event at 102. Does V1 traverse it
differently? Yes — V1 never safeguards at all and crosses the same barrier
level at iteration 68, earlier than V2's 113 and without ever experiencing the
safeguard mechanism.

### E. Residual / complementarity progression — first to cease improving

Using the predeclared "sustained" definition (Section 6):

| run | constraint_violation first sustained increase | constraint_violation first sustained decrease | complementarity first sustained increase | complementarity first sustained decrease |
|---|---|---|---|---|
| V1 | 6 | 0 | 0 | 24 |
| Arm A | 6 | 0 | 8 | 29 |
| V2 | 6 | 0 | 17 | 12 |

All three runs show a "first sustained increase" in constraint violation at
iteration 6 — this is a common, early, small-magnitude transient present in
all three (not distinguishing) and is reported explicitly rather than
suppressed, per the instruction not to invent or omit values.

A directly relevant long-horizon scan of Arm A beyond iteration 102 (values at
iterations 77, 100, 150, 200, 300, 500, 1000, 1500, 2000, 2500, 2999, 3000;
`journal.json → analysis → journal_series → ArmA`) shows constraint
violation decreasing **monotonically but extremely slowly** and quasi-linearly
from `0.07252` (iter 77) to `0.06192` (iter 3000) — a net decrease of only
about 15% over 2923 iterations, with `alpha_pr` essentially frozen at `≈5.4e-5`
for the entire span from roughly iteration 150 onward. This is a genuine
"sustained decrease" by the predeclared definition, but at a rate that would
require tens of thousands of further iterations to reach the tolerance level
V1/V2 reach in under 115 iterations. So Arm A is not literally static (its
residuals do creep downward), but it is **effectively parked**: it neither
diverges nor makes qualitatively meaningful progress once `alpha_pr` locks at
`~5e-5`, whereas V2's `alpha_pr` eventually jumps by four orders of magnitude
to `1.0` and its residuals collapse by several orders of magnitude within ~10
further iterations.

**Which quantity first ceases to improve, relative to the difficult-regime
onset (safeguard onset, 77/82):** the step-size family (`alpha_pr`, `alpha_du`,
`||d||`) is the first to visibly stall — it locks to a near-constant small
value essentially at the safeguard onset itself and stays there
(Arm A: `alpha_pr ≈ 5–6e-5` from iteration ~77 through 3000, with no
recovery). Constraint violation and complementarity continue slow monotone
motion throughout (not "stuck" by the strict predeclared definition, since
there is a persistent, if glacial, sustained decrease), so the step-size
stagnation is the earliest-observed and most persistent stall signal, and it
occurs **simultaneously with**, not clearly before, the onset of persistent
safeguard activation (both begin at iteration 77 in Arm A). This report does
not claim one causes the other.

### F. Escape signature

**Predeclared, empirical, telemetry-only escape signature for V2** (defined
in the harness before inspecting results beyond the last-safeguard iteration
itself): relative to V2's last observed safeguard iteration (102),
(i) `alpha_pr` rises by at least one order of magnitude relative to its value
at iteration 102, and/or (ii) primal infeasibility (`inf_pr`, summary-row
column) begins a sustained decrease (≥5 consecutive iterations) from iteration
102 onward, and/or (iii) no further safeguard event occurs after 102 (true by
definition of "last").

Measured for V2: `alpha_pr` at iteration 102 = `5.39e-05`; the ≥10×
order-of-magnitude jump occurs at iteration **108** (`alpha_pr` there,
per the summary row, exceeds `5.39e-4`); `inf_pr` begins a sustained ≥5-step
decrease starting at iteration **104**; no further safeguard occurs after 102
(by construction). All three predeclared components of the escape signature
are satisfied, with `inf_pr`'s sustained decrease (iteration 104) preceding
the ≥10× `alpha_pr` jump (iteration 108), which itself precedes the full
`alpha_pr/alpha_du → 1.0` step and the barrier decrease (iteration 113).

**Testing V1 for an analogous transition:** V1 has **zero** safeguard events,
so there is no "last safeguard iteration" anchor from which to test the same
signature — this is reported explicitly rather than forcing an artificial
anchor (task instruction: "do not force one if unsupported"). V1's own barrier
decrease (iteration 68) and its `alpha_pr` history can still be inspected
directly: V1's `alpha_pr` is `1.55e-01` at iteration 60 and reaches `1.19e-03`
by iteration 70 (a *decrease*, not an escape-type jump) before its
subsequent recovery to optimality by iteration 89 — V1's late-stage
approach to optimality does not show the same "long small-step plateau then
sudden full-step jump" pattern that V2 shows; V1's step sizes are already
larger and more variable well before its own barrier transition. This is
reported as a qualitative difference, not a forced match.

**Testing Arm A:** Arm A's safeguard sequence runs continuously to iteration
3000 (the very last recorded iteration), so it never satisfies component
(iii) of the escape signature (there is no iteration after which "no further
safeguard occurs" within the observed 3000-iteration horizon). Its `alpha_pr`
never leaves the `~5e-5` regime (confirmed through iteration 3000, Section E).
**Arm A never exhibits the escape signature within the recorded horizon.**

## 8. Required quantitative tables

All raw values below are taken verbatim from `journal.json`
(`analysis → tables`), which in turn are taken verbatim from the parsed logs
with an exact `raw_line_no` traceable for every field (see `journal.json` for
full line-number provenance; omitted here for brevity).

### Iteration 60

| field | V1 | Arm A | V2 |
|---|---|---|---|
| objective (unscaled) | 1309.1897548782911 | 1310.0480933062895 | 1310.3958813626082 |
| inf_pr | 2.22e-01 | 2.16e-05 | 7.41e-05 |
| inf_du | 2.30e-01 | 2.57e-03 | 3.59e-04 |
| lg(mu) | -5.7 | -5.7 | -5.7 |
| \|\|d\|\| | 3.78e-01 | 1.73e-02 | 8.28e-03 |
| lg(rg) | -3.7 | -0.9 | -1.4 |
| alpha_du | 8.27e-01 | 9.67e-01 | 1.00e+00 |
| alpha_pr | 1.55e-01 | 8.09e-01 | 1.00e+00 |
| ls | 1 | 1 | 1 |
| curr_z_U | 4.4059e-01 | 4.4276e-01 | 4.4396e-01 |
| curr_y_c | 630.40 | 630.39 | 630.52 |
| safeguard here? | no | no | no |

### Iteration 70

| field | V1 | Arm A | V2 |
|---|---|---|---|
| objective (unscaled) | 1297.9599340633195 | 1304.4401795217082 | 1309.9187975535160 |
| inf_pr | 6.34e-03 | 1.49e-01 | 3.34e-06 |
| inf_du | 1.10e+01 | 1.06e-01 | 4.08e-04 |
| lg(mu) | -8.0 | -5.7 | -5.7 |
| \|\|d\|\| | 4.17e+00 | 9.59e-01 | 2.12e-03 |
| lg(rg) | `-` | `-` | -0.7 |
| alpha_du | 3.35e-03 | 9.43e-02 | 1.00e+00 |
| alpha_pr | 1.19e-03 | 6.32e-02 | 1.00e+00 |
| mu | 9.0909e-09 | 1.8449e-06 | 1.8449e-06 |
| safeguard here? | no | no | no |

Note: V1 already crossed the `mu=1.8449e-06 → 9.0909e-09` barrier transition
by iteration 68; Arm A and V2 have not yet (both still at `1.8449e-06`).

### First safeguard event (per run)

| run | first safeguard iteration |
|---|---|
| V1 | none (0 events) |
| Arm A | 77 |
| V2 | 82 |

### Iteration 90

| field | V1 | Arm A | V2 |
|---|---|---|---|
| available | no (V1 terminates at 89) | yes | yes |
| objective (unscaled) | — | 1302.2903083292560 | 1303.4768901005400 |
| inf_pr | — | 7.25e-02 | 1.47e-01 |
| inf_du | — | 6.04e-02 | 9.15e-02 |
| alpha_du | — | 1.65e-06 | 4.44e-07 |
| alpha_pr | — | 3.93e-05 | 3.42e-05 |
| \|\|d\|\| | — | 7.92e+00 | 1.47e+01 |
| mu | — | 1.8449e-06 | 1.8449e-06 |
| safeguard here? | — | yes | yes |

### V2's final safeguard (iteration 102) and Arm-A event-aligned counterpart (iteration 97)

| field | V2 @ 102 | Arm A @ 97 (event-aligned: onset 77 + offset 20) |
|---|---|---|
| objective (unscaled) | 1303.4744034597325 | 1302.2903558208591 |
| inf_pr | 1.47e-01 | 7.25e-02 |
| inf_du | 9.14e-02 | 6.04e-02 |
| alpha_du | 5.33e-05 | 1.69e-05 |
| alpha_pr | 5.39e-05 | 4.52e-05 |
| \|\|d\|\| | 1.59e+00 | 2.46e+00 |
| mu | 1.8449e-06 | 1.8449e-06 |
| safeguard here? | yes (last) | yes |

### ~5 and ~10 iterations after V2's escape (107 / 112) and Arm-A event-aligned counterparts (107 / 112, since offset 20 applied to Arm-A anchor 77 gives 97/102/107 for V2's 102/107/112 minus … — see exact mapping in Section 7C)

| field | V2 @ 107 (+5) | Arm A @ 107 (event-aligned) | V2 @ 112 (+10) | Arm A @ 112 (event-aligned, from long-horizon scan) |
|---|---|---|---|---|
| alpha_pr | 1.05e-04 | 5.42e-05 | **1.00e+00** | ≈5.4e-05 (frozen; exact iter-112 row not separately tabulated but bracketed by 102→150 scan, all ≈5.4e-05) |
| alpha_du | 3.65e-04 | 5.30e-05 | **1.00e+00** | ≈5e-05 (frozen) |
| \|\|d\|\| | 1.00e+00 | 1.72e+00 | 9.13e-04 | ≈1.7 (frozen) |
| inf_pr | 1.47e-01 | 7.24e-02 | **3.18e-07** | ≈0.072 (frozen) |
| inf_du | 9.14e-02 | 6.04e-02 | **5.77e-07** | ≈0.060 (frozen) |
| mu | 1.8449e-06 | 1.8449e-06 | 1.8449e-06 (drops to 9.0909e-09 at iter 113) | 1.8449e-06 (never drops through iter 3000) |
| safeguard here? | no | yes | no | (Arm A continues safeguarding every iteration through 3000) |

## 9. Predictive versus terminal signals

- **Terminal-only** (only observable once the outcome is already effectively
  determined, i.e. confirm rather than predict): the final termination
  message itself; the final iteration count; the final objective/residual
  values.
- **Coincident with, not clearly preceding, the difficult regime's onset**:
  the step-size stall (`alpha_pr`/`alpha_du` locking to `~5e-5`) begins at
  essentially the same iteration as the first safeguard event in Arm A (77)
  and in V2 (82) — this report cannot establish temporal precedence between
  step stagnation and safeguard onset at the available iteration resolution.
- **Plausibly predictive of the eventual outcome, once the difficult regime is
  already underway**: the *persistence* of the step-size stall past the point
  where V2 recovers is the clearest available discriminator. By iteration
  ~113–150, V2 has escaped (full step, barrier decrease, residual collapse)
  while Arm A has not; from that point forward Arm A's telemetry
  (`alpha_pr` frozen at `~5e-5`, `mu` frozen at `1.8449e-06`) is consistent
  with — but does not, on this evidence alone, mechanistically prove —
  eventual persistence to the iteration cap. This is the strongest available
  predictive-flavored signal, but it is observed only in hindsight (after the
  escape/non-escape has already begun to manifest), not from data available
  strictly before iteration ~90–100 in a way that would have distinguished
  Arm A from V2 earlier. The exploratory, non-predeclared "normalized ratio"
  divergence at iterations 18–20 (Section 7A) is reported as a candidate
  earlier signal but is explicitly flagged as secondary/exploratory and not
  confirmed as predictive.
- Multiplier-norm magnitude alone (Section B) is **not** shown to be
  predictive: large multiplier norms occur transiently in all three runs
  around iterations 25–31 without distinguishing outcome.
- Correction magnitude alone (Section C) is **not** shown to be predictive:
  Arm A's correction magnitudes are as small as V2's at their respective
  latest observed safeguard events, yet only V2 escapes.

## 10. Artifact inventory with hashes

| artifact | path | SHA-256 |
|---|---|---|
| harness script | `p512_t_trajectory_forensic.py` | `b2f24abfb6028b52a78a8279de9cfd40fed87d7ecc60aec3be438263c8dcb22d` |
| journal | `data/SRP1/Results/P512T/journal.json` | `9a9287433bb22e8f7691c44ea398e4b2c4683e3f9e39eec8b6e8adce08c54a8f` |
| manifest | `data/SRP1/Results/P512T/manifest.json` | see file; generated alongside journal, listing input hashes above |
| this report | `P5_12_T_TRAJECTORY_FORENSIC_REPORT.md` | (hash recorded post-write in the final worker message) |
| input V1 log | `data/SRP1/Results/P512P/variant1_1e-6/logs/optim_log_case33_3_2025_Spring.log` | `3c5479835849bc859f3bcaf964a2791e89318bae0fe979d2eb7dbb3e155d2bd7` (unchanged) |
| input Arm A log | `data/SRP1/Results/P512ArmA/logs/optim_log_case33_3_2025_Spring.log` | `4f66a7efeef933bdc0a425af76f0095f5c11a2112ff2c8bb6d7c03ff45409d58` (unchanged) |
| input V2 log | `data/SRP1/Results/P512P/variant2_1e-4/logs/optim_log_case33_3_2025_Spring.log` | `ccf80716e9b7c7f7346d1404234a2b125d00cc0658f6ec8ba35ad8a63095ed2e` (unchanged) |

## 11. Integrity check

- Git branch/HEAD unchanged across the session (`feature/derivative-free-planning`,
  `ba202e2b0e937306f3c163de2173951c1d0c24f0`).
- All three input log hashes re-verified identical to the values declared in
  the task before parsing began, and the files were never opened in write
  mode by the harness.
- `data/SRP1/Results/P512T/` was confirmed absent before creation; the
  harness aborts if it already exists.
- No solver invocation occurred: the harness contains no call to IPOPT, no
  `solver.solve`, and no import of any production optimization entry point;
  it is a pure text-parsing and arithmetic analysis script over already-closed
  log files.
- No production file, configuration file, or governing document was modified
  during this stage.

## 12. Verdict

The available telemetry shows a specific, reproducible, and multiply
corroborated numerical transition — the step-size family (`alpha_pr`,
`alpha_du`, `||d||`) unlocking from a long plateau near `1e-4`–`1e-5` to a
full accepted step (`≈1.0`), immediately followed by a barrier-parameter
decrease (`mu`: `1.8449e-06 → 9.0909e-09`) and an order-of-magnitude residual
collapse — that is present in V2 (at iterations ~102–113, immediately after
its last safeguard event) and absent, with the corresponding quantities
instead frozen at their stalled values, in Arm A throughout its entire
3000-iteration run (confirmed via a long-horizon scan through iteration
3000). This transition is a defensible, IPOPT-internal, solver-mechanistic
signal (a change in globalization/step-acceptance behavior coincident with a
barrier-parameter reduction), and it is the only quantity examined that
cleanly separates "escape" (V2, and trivially V1, which never enters the
stalled regime at all) from "persistence" (Arm A). However: (i) the
step-size stall and the onset of persistent safeguard activity begin at
essentially the same iteration in both Arm A and V2, so this report cannot
establish which is temporally prior, only that they co-occur and that only
one run's stall subsequently resolves; (ii) multiplier-norm growth and
correction-magnitude behavior, examined as candidate mechanisms, do **not**
distinguish the two runs in the difficult-regime window and are therefore
demoted, not promoted, as candidate mechanisms by this evidence; (iii) no
solve-level or Jacobian/Hessian-level diagnostic was available or performed,
so the *reason* IPOPT's internal filter/step-acceptance logic unlocks in one
run and not the other remains unestablished at the mechanism level, even
though the *telemetry signature* of the unlock is clear and reproducible.

Given a clear, reproducible, well-localized numerical transition exists and
distinguishes escaping from persistent trajectories, but the deeper
solver-internal reason for why IPOPT's filter/step-acceptance logic resolves
in one run and not the other is not established by log telemetry alone (no
Jacobian/Hessian/KKT-system diagnostic was performed, consistent with this
stage's no-solve, log-only scope):

**ESCAPE SIGNATURE OBSERVED — MECHANISM UNRESOLVED**
