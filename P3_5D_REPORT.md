# Stage P3.5-D — `sess_snet_def` in-place equivalent scaling (executed)

Executed by the user via `p35d_sess_snet_def_inplace_scaling_diagnostic.py`.
Raw output: `data/SRP1/Results/FrozenSMOPF/P35D/p35d_report.json`. Frozen-file
hashes verified identical to all prior stages for both cases.

This stage removes the confound P3.5-C surfaced: no `Constraint` component
was created or deactivated anywhere. The original `sess_snet_def`
`ConstraintData` objects were mutated in place via `.set_value(...)`, and
each row's `model.dual` entry was updated **on that same object** from
`dual_before` to `dual_before/100.0`.

---

## Required proof, before solving B (both cases)

| Check | DSO | TSO |
|---|---|---|
| `sess_snet_def` Python component object unchanged | ✅ | ✅ |
| Component local name still `sess_snet_def` | ✅ | ✅ |
| No new constraint component created (full before/after name list compared) | ✅ (51/51 identical) | ✅ (51/51 identical) |
| No `sess_snet_def_scaled` attribute exists | ✅ | ✅ |
| Total constraint-data count unchanged | ✅ (`7829 → 7829`) | ✅ (`3528 → 3528`) |
| All `sess_snet_def` index tuples unchanged (whole component, not just the installed index) | ✅ (`24 → 24`) | ✅ (`72 → 72`) |
| All 24 target rows still active (never deactivated) | ✅ | ✅ |
| Per-row `ConstraintData` object identity (`id()`) unchanged, all 24 rows | ✅ (24/24) | ✅ (24/24) |
| Every dual transformed on the *same* object, `dual_after == dual_before/100` exactly | ✅ (24/24) | ✅ (24/24) |

Every required invariant holds exactly, for both cases, with no exceptions.
This is the strongest possible form of the proof the plan asked for: not
"a new but equivalent constraint," but the literal original object, mutated
in place, carrying its own continuously-updated multiplier.

---

## A vs B100-inplace

| | DSO A | DSO B100-inplace | TSO A | TSO B100-inplace |
|---|---|---|---|---|
| Status / termination | error / internalSolverError | **ok / optimal** | warning / maxIterations | **ok / optimal** |
| Recovery | Yes (also failed) | **No** | Yes (exhausted) | **No** |
| Iterations | 474 → 449 | **383** | 66 → 3000 | **104** |
| Objective | `9.1247e-03` (unconverged) | `9.1439e-03` | `2.19432` → `2.20877` (unconverged) | `2.19370` |
| Dual infeasibility | `5.58e+08` → `7.05e-03` | `2.06e-08` | `3.168e+07` → `2.591e+03` | `1.98e-08` |
| Constraint violation | `4.30e-08` → `8.85e-14` | `8.14e-14` | `4.00e-08` → `7.23e-06` | `2.70e-11` |
| Complementarity | `8.45e-06` → `1.57e-07` | `1.84e-06` | `3.39e-06` → `8.95e-05` | `9.09e-07` |
| Overall NLP error | `2.47e+06` → `7.05e-03` | `1.84e-06` | `7.30e+05` → `2.59e+03` | `9.09e-07` |
| CPU time | `4.09s+3.84s≈7.93s` | `3.87s` | `0.13s+7.86s≈7.98s` | `0.21s` |
| Final residual (normalized) | — | `3.90e-12` | — | `1.14e-12` |

**Both cases now succeed on the primary exact-Hessian attempt, with no
recovery needed** — including DSO, which in P3.5-C *failed* (worse than
baseline) under the equivalent scaled-replacement-component variant with a
correctly transferred dual. The only difference between that failing P3.5-C
DSO run and this succeeding one is exactly the thing this stage was
designed to remove: whether the scaled equation lives on a freshly-created
component or on the original object in place.

The final, unscaled `(sch-sdch)^2 - pnet^2 - qnet^2` residual is at machine
precision in both cases (`3.90e-12` DSO, `1.14e-12` TSO, normalized by
`s_rated^2`) — the physical relation is preserved, not relaxed, consistent
with every prior scaling (as opposed to removal) experiment.

---

## Interpretation

1. **This resolves the P3.5-C DSO anomaly.** P3.5-C showed DSO's R100
   (new component, kappa=100, correctly transferred dual) failing worse
   than the untouched baseline, which could not be distinguished from "the
   scaling doesn't actually help DSO" without also controlling for
   component identity. This stage controls for exactly that: same
   `ConstraintData` objects, same container, same position in the model, no
   new component anywhere — and DSO now succeeds cleanly. The P3.5-C DSO
   failure is therefore attributable to the act of replacing the constraint
   with a *new* component object (and whatever row/column reordering that
   caused for MA97), not to any incompatibility between scaling and a
   properly continued nonzero multiplier.

2. **Candidate 1 (poor intrinsic scaling of `sess_snet_def`) is now
   confirmed in both cases, free of every confound identified across this
   whole investigation.** Not "removed" (P3.5-A, which materially relaxed
   the physical relation at one hour per case), not "replaced by a new,
   algebraically-equivalent component with a reset multiplier" (P3.5-B),
   not "replaced by a new component with a correctly-transferred multiplier"
   (P3.5-C, which failed for DSO) — but the literal original equation,
   scaled by a constant, evaluated on the same object, carrying its own
   correctly continued multiplier. This is the cleanest and most
   direct evidence in the investigation that the practical local-NLP
   failure is a **conditioning/scaling problem intrinsic to
   `sess_snet_def`**, not an artifact of warm-start bookkeeping or of how
   any of the diagnostic scripts happened to implement the test.

3. **Recovery was never needed.** Both cases converged to `optimal` on the
   primary exact-Hessian attempt, matching every previous successful
   scaling/removal variant.

4. **No production change is authorized by this result.** As with every
   prior stage, this is a frozen, read-only diagnostic on in-memory clones.
   It provides the strongest evidence yet that a permanent, principled
   rescaling of `sess_snet_def` (in place, preserving object identity —
   exactly as demonstrated here, not as a parallel replacement component)
   would be a promising direction for a future, separately-approved
   production-formulation stage, together with attention to how any such
   change interacts with ADMM's cross-iteration dual warm-starting. That is
   offered as an observation for planner consideration, not as an
   authorized next step.

---

`P3.5-D COMPLETE — waiting for planner review`
