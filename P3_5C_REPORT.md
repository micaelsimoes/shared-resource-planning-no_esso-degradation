# Stage P3.5-C — `sess_snet_def` dual-multiplier warm-start confound check (executed)

Executed by the user via `p35c_sess_snet_def_dual_warmstart_diagnostic.py`.
Raw output: `data/SRP1/Results/FrozenSMOPF/P35C/p35c_report.json`. Frozen-file
hashes verified identical to P3.5-A/B for both cases (not re-tabulated here).

---

## Part 1 — did P3.5-B already transfer the frozen duals? (answered before any new run)

**No.** Static review of `p35b_sess_snet_def_scaling_diagnostic.py` shows it
never references `dual` or `Suffix` anywhere — it only deactivates the
original rows and creates the new `sess_snet_def_scaled` component. Because
a Pyomo `Suffix` maps by object identity, the 24 new rows in every P3.5-A
"B" and P3.5-B "B" solve started with **no** `model.dual` entry at all.

The frozen pickles **do** carry substantial, non-negligible multipliers for
all 24 original rows in both cases:

| | DSO | TSO |
|---|---|---|
| Rows with a `model.dual` entry | 24 / 24 | 24 / 24 |
| \|dual\| min | `1.804` | `39.70` |
| \|dual\| median | `83.40` | `261.04` |
| \|dual\| max | `595.05` | `3650.76` |

These are large relative to the scaled constraint's own natural gradient
scale established in P3.5-B (`O(4e-3)`–`O(1.5)`), so the confound the
planner flagged was real and worth checking — this was not a negligible
effect to rule out a priori. Since the mapping was never done, P3.5-C was
required and was executed.

---

## Part 2 — A / R1 / R100 results

### DSO (`case33_2`, node 7, Autumn, cycle 8)

| | A (baseline) | R1 (kappa=1, dual copied) | R100 (kappa=100, dual copied /100) |
|---|---|---|---|
| Status / termination | error / internalSolverError | **ok / optimal** | warning / **maxIterations** |
| Recovery | Yes (also failed) | **No** | Yes (exhausted, and worse than A's) |
| Iterations | 474 → 449 | **629** | 396 → 3000 |
| Objective | `9.1247e-03` (unconverged) | `9.1330e-03` | `4.7536` → `3.030e-03` (unconverged) |
| Dual infeasibility | `5.58e+08` → `7.05e-03` | `3.04e-08` | `2.00e+10` → `1.076e+01` |
| Constraint violation | `4.30e-08` → `8.85e-14` | `4.22e-09` | `1.80e-09` → `8.54e-05` |
| Complementarity | `8.45e-06` → `1.57e-07` | `1.84e-06` | `2.98e-01` → `1.19e-06` |
| Overall NLP error | `2.47e+06` → `7.05e-03` | `1.84e-06` | `9.79e+01` → `1.076e+01` |
| CPU time | `4.17s+3.82s≈7.99s` | `5.44s` | `4.29s+26.54s≈30.83s` |
| Final residual (normalized) | — | `1.19e-16` | `1.40e-16` |

**R1 succeeded outright; R100 failed — and failed more severely than the
untouched baseline A** (primary dual infeasibility peaks at `2.0e10`,
two orders of magnitude worse than A's `5.58e8`, and the recovery attempt's
final overall NLP error, `10.76`, never approaches tolerance even after
exhausting `3000` iterations).

### TSO (`case9`, Summer, cycle 6)

| | A (baseline) | R1 (kappa=1, dual copied) | R100 (kappa=100, dual copied /100) |
|---|---|---|---|
| Status / termination | warning / maxIterations | warning / **maxIterations** | **ok / optimal** |
| Recovery | Yes (exhausted) | Yes (exhausted) | **No** |
| Iterations | 66 → 3000 | 66 → 3000 | **103** |
| Objective | `2.19432` → `2.20877` | `2.19432` → `2.20877` | `2.19432` |
| Dual infeasibility | `3.168e+07` → `2.591e+03` | `3.168e+07` → `2.591e+03` | `6.06e-07` |
| Constraint violation | `4.00e-08` → `7.23e-06` | `4.00e-08` → `7.23e-06` | `3.84e-14` |
| Complementarity | `3.39e-06` → `8.95e-05` | `3.39e-06` → `8.95e-05` | `1.85e-06` |
| Overall NLP error | `7.30e+05` → `2.59e+03` | `7.30e+05` → `2.59e+03` | `1.85e-06` |
| CPU time | `0.13s+7.79s≈7.92s` | `0.13s+7.72s≈7.85s` | `0.20s` |
| Final residual (normalized) | — | `1.08e-09` | `3.84e-12` |

**R1 reproduces A to essentially every reported digit** (same iteration
counts, same objective/dual-inf/complementarity in both primary and
recovery). **R100 succeeds outright**, primary only, no recovery.

**Pattern check** (`A FAIL / R1 FAIL / R100 PRIMARY SUCCESS`):

| | DSO | TSO |
|---|---|---|
| Matches expected pattern | **No** | **Yes** |

---

## Interpretation

1. **TSO cleanly confirms the hypothesis, in the strongest possible form.**
   R1 — a new component, mathematically identical to the original
   (`kappa=1`), with the exact original multiplier faithfully carried over —
   fails identically to A, digit for digit. This rules out "merely being a
   freshly-created replacement component" and "merely losing/resetting the
   frozen multiplier" as the explanation for TSO's earlier P3.5-A/B success:
   here the multiplier was *not* reset, it was properly continued (and, for
   R100, correctly transformed by the KKT-consistent rule
   `dual_new = dual_old/kappa`), and the scaled version still converges to
   `optimal` on the primary attempt with the tightest infeasibility numbers
   seen anywhere in this whole investigation (`dual_inf=6.06e-7`). For TSO,
   **Candidate 1 (poor scaling) is now confirmed independent of the
   dual-warm-start confound.**

2. **DSO gives the opposite pattern, and does not confirm the scaling
   hypothesis in isolation.** R1 (no scaling benefit expected) succeeds; R100
   (the scaling the planner expected to fix things) fails, and fails *worse*
   than the untouched baseline (primary dual infeasibility of `2.0e10`,
   ~40x worse than A's `5.58e8`). Since R1 and R100 are built the same way —
   deactivate the same 24 rows, add a new one-for-one replacement component,
   transfer the correctly-scaled dual — the only things that differ between
   them are the value of `kappa` (1 vs 100) and the resulting multiplier
   magnitude (`O(100)` vs `O(1-10)`). That is enough, for this specific
   frozen DSO state, to flip the outcome in the *opposite* direction from
   TSO.

3. **A previously un-planned confound is now surfaced: component identity /
   reordering.** R1 is algebraically a no-op relative to A (`kappa=1`,
   dual copied exactly), yet it does not behave like A for DSO — it
   converges cleanly in 629 iterations where A fails outright. The only
   remaining structural difference is that `sess_snet_def_scaled` is a
   freshly declared `Constraint` component appended after the original
   model construction, rather than the original in-place rows — which
   changes the row/column ordering IPOPT/MA97 sees, and can change the
   floating-point pivot sequence for an already near-singular KKT system
   even though the underlying equations are identical in exact arithmetic.
   This is consistent with the DSO instance being more on-the-edge/fragile
   than TSO (recall P3.5-A's starting-point diagnostics: 8 of 24 DSO rows
   already had gradient-∞-norm below the loosest `1e-4` threshold, vs. 0 of
   24 for TSO) — small, solver-path-level perturbations are more likely to
   flip its outcome either way. This was not something P3.5-A/B could have
   revealed, since neither had an identity-only (`kappa=1`) control; it took
   the R1 control specified for this stage to expose it.

4. **Original unscaled relation at the final solution:** in every case where
   a solve succeeded (DSO R1, TSO R100) or nearly did (DSO R100, despite not
   converging overall), the physical `sess_snet_def` relation itself was
   satisfied to nowhere-near-material tolerance (`1.2–1.4e-16` normalized,
   DSO; `1.1e-9` and `3.8e-12` normalized, TSO) — consistent with P3.5-A/B,
   this is not in question. The open question this stage raises is purely
   about *which combination of formulation and multiplier continuity
   determines whether IPOPT reaches that solution at all*, not about
   whether the solution is physically valid once reached.

5. **Net effect on confidence in the P3.5-B result.** P3.5-B's outcome is
   **not invalidated** — the physical-equivalence finding stands unchanged —
   but its causal story is now known to be **incomplete and case-dependent**:
   for TSO, scaling is confirmed sufficient on its own, independent of the
   multiplier-continuity confound. For DSO, the picture is genuinely mixed:
   P3.5-B's success (kappa=100, no dual transferred at all, i.e. an
   effectively cold/default multiplier) cannot be attributed to scaling
   alone, because supplying a *correctly* continued nonzero multiplier at
   the same kappa=100 makes the DSO case fail, worse than baseline. Whatever
   let P3.5-B's DSO variant converge appears entangled with the multiplier
   being reset rather than with the scaling itself, and a newly-surfaced
   component-reordering effect may also be a contributing factor
   independent of both. **This should be treated as a genuinely unresolved,
   case-dependent finding, not forced into either a confirming or refuting
   verdict for Candidate 1 as a whole.**

No production formulation, solver option, ADMM parameter, or `sess_comp`
change is authorized by this result, consistent with every prior stage.
Given the DSO/TSO divergence and the newly-surfaced reordering confound,
a follow-up stage isolating reordering from multiplier value (e.g. an
identity replacement with the dual explicitly zeroed, on DSO only) would be
needed before drawing a firm conclusion for the DSO case specifically — that
is offered as an observation for planner consideration, not proposed as an
authorized next step.

---

`P3.5-C COMPLETE — waiting for planner review`
