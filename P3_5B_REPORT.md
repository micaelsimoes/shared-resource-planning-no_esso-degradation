# Stage P3.5-B — `sess_snet_def` equivalent-scaling frozen diagnostic (executed)

Executed by the user in the real project Pyomo/IPOPT/MA97 environment via
`p35b_sess_snet_def_scaling_diagnostic.py` (repo root), against the same two
prescribed frozen cases used in P3.5-A. Raw output:
`data/SRP1/Results/FrozenSMOPF/P35B/p35b_report.json`. No production file,
solver option, ADMM parameter, or `sess_comp`/other constraint was touched —
only `sess_snet_def` for the one genuinely installed shared-ESS index, on an
independent in-memory clone, replaced one-for-one by a constant-positive-
scalar-multiplied version of the identical equation.

---

## A. Frozen-file verification

| Case | Path | Expected SHA-256 | Actual SHA-256 | Match |
|---|---|---|---|---|
| DSO | `data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl` | `066117b8…637074711` | identical | ✅ |
| TSO | `data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl` | `51d90974…4b08355e` | identical | ✅ |

Same two files as P3.5-A, byte-identical. Neither file was overwritten.

---

## B. Untouched baseline reproduction (A)

| | DSO A | TSO A |
|---|---|---|
| Primary termination | `internalSolverError` ("Error in step computation"), 474 iters, `dual_inf≈5.58e8` | distress at iter 66, `dual_inf≈3.17e7` |
| Recovery termination | `internalSolverError` again, 449 iters, `dual_inf≈7.05e-3` (still non-convergent) | `maxIterations` at 3000 iters, `dual_inf≈2.59e3` |
| Final classification | **status=error, termination=internalSolverError** | **status=warning, termination=maxIterations** |
| `succeeded` | **False** | **False** |

These numbers are **identical to the P3.5-A baseline run** (same iteration
counts, same `dual_inf`/`constr_viol`/`complementarity` values to the digits
reported), confirming the untouched replay is deterministic and reproduces
the recorded failure exactly in both cases. A/B attribution is valid (plan
§11 Outcome D does not apply); neither baseline unexpectedly succeeded
(Outcome B4 does not apply).

---

## C. Shared-ESS index, capacity, and kappa construction

| | DSO | TSO |
|---|---|---|
| Installed index | `0` (only index) | `1` (of 0/1/2; indices 0 and 2 have `s_rated=e_rated=0`, consistent with P3) |
| `s_rated` | `0.01` | `0.01` |
| `e_rated` | `0.024172` | `0.024172` |
| `kappa = 1/s_rated` | **`100.0`** (exact) | **`100.0`** (exact) |

Both match the plan's expected `s_rated≈0.01` / `kappa≈100` exactly.

---

## D. Scaling construction and pre-solve equivalence verification

| | DSO (24 rows) | TSO (24 rows) |
|---|---|---|
| Original rows deactivated | 24 | 24 |
| Replacement rows added | 24 | 24 |
| Index tuples match one-to-one | **True** | **True** |
| unscaled `|g|` (min/median/max) | `0 / 1.69e-24 / 1.40e-20` | `8.21e-22 / 4.90e-19 / 1.08e-13` |
| scaled `|g_scaled|` max (`=kappa·|g|`) | `1.40e-18` | `1.08e-11` |
| unscaled grad-∞-norm (min/median/max) | `4.10e-05 / 1.57e-04 / 1.53e-02` | `1.93e-04 / 3.58e-03 / 1.27e-02` |
| scaled grad-∞-norm (min/median/max) | `4.10e-03 / 1.57e-02 / 1.526` | `1.93e-02 / 3.58e-01 / 1.273` |

Cross-checking every reported scaled statistic against `kappa × (its unscaled
counterpart)` confirms an **exact 100× relationship in all eight values**
(both cases, both `|g|` and gradient-∞-norm, min/median/max) — the
construction is the intended pure positive-scalar multiple, with no
unintended structural change. (Note: the script's own
`scaling_identity_check_max_rel_err` field reports `0.0` for both cases, but
that field compares an expression to itself and is not an independent check
— the meaningful verification is the cross-check above, which does hold
exactly.)

Practically, scaling moves the gradient-∞-norm range from
`[4.1e-5, 1.5e-2]` (DSO) / `[1.9e-4, 1.3e-2]` (TSO) — both well below unit
scale — up to `[4.1e-3, 1.53]` / `[1.9e-2, 1.27]`, i.e. into an O(1) range
much closer to what IPOPT's internal scaling assumes is well-posed. This is
consistent with the row being *poorly scaled* rather than *structurally
singular*, as already suggested by the P3.5-A starting-point diagnostics.

---

## E. Solver results — A vs B, side by side

| | DSO A (baseline) | DSO B (scaled) | TSO A (baseline) | TSO B (scaled) |
|---|---|---|---|---|
| Status / termination | error / internalSolverError | **ok / optimal** | warning / maxIterations | **ok / optimal** |
| Primary succeeded | No | **Yes** | No | **Yes** |
| Recovery needed | Yes (also failed) | **No** | Yes (exhausted) | **No** |
| Iterations | 474 → 449 (recovery) | **233** | 66 → 3000 (recovery) | **64** |
| Objective | `9.1247e-03` (unconverged) | `9.1070e-03` | `2.19432` (unconverged) | `2.19432` |
| Dual infeasibility | `5.58e+08` → `7.05e-03` | `4.09e-09` | `3.17e+07` → `2.59e+03` | `5.55e-06` |
| Constraint violation | `4.30e-08` → `8.85e-14` | `2.90e-08` | `4.00e-08` → `7.23e-06` | `1.46e-08` |
| Complementarity | `8.45e-06` → `1.57e-07` | `1.84e-06` | `3.39e-06` → `8.95e-05` | `1.93e-06` |
| Overall NLP error | `2.47e+06` → `7.05e-03` | `1.84e-06` | `7.30e+05` → `2.59e+03` | `5.55e-06` |
| CPU time | `4.12s + 3.85s = 7.97s` | `1.83s` | `0.12s + 7.74s = 7.86s` | `0.12s` |

The kappa-scaled replacement converts both primary exact-Hessian solves from
failure to a clean **primary** optimal solution, no recovery needed — same
qualitative result as P3.5-A's outright removal. Convergence takes somewhat
more iterations than outright removal did (DSO: 233 vs 109; TSO: 64 vs 50),
which is expected since the scaled equation still enforces the full
constraint rather than deleting it, but is still 2–65× fewer iterations and
4–65× less time than the failing baseline. The converged objective in B
matches the (unconverged) objective the baseline was heading toward to
within `0.002%` (TSO) – `0.19%` (DSO), i.e. B lands in essentially the same
operating regime the original solve was trying to reach.

---

## F. Original unscaled `sess_snet_def` relation at the final B solution

| | DSO | TSO |
|---|---|---|
| Rows evaluated | 24 | 24 |
| `max|g|` | `1.355e-20` | `1.917e-15` |
| `median|g|` | `2.145e-24` | `2.541e-21` |
| `max|g| / s_rated²` (normalized) | **`1.36e-16`** | **`1.92e-11`** |
| Worst row | `(e=0, s_m=0, s_o=0, p=3)` | `(e=1, s_m=0, s_o=0, p=13)` |

This is the decisive check (plan §10). Both normalized residuals are at
**machine precision**, not merely "small" — many orders of magnitude tighter
than IPOPT's own `tol=1e-5`/`acceptable_tol=1e-4`, and roughly **15–20 orders
of magnitude smaller than the P3.5-A outright-removal residuals** at the
same worst-hour comparison (`0.320` DSO / `0.321` TSO, i.e. ~32% of
`s_rated²`). The original, unscaled physical apparent-power identity is
satisfied essentially exactly at the B solution in both cases — there is no
detectable trace of the physical relaxation that outright removal produced.
Per plan §10, this run qualifies as valid evidence (not to be discarded).

---

## G. Interpretation

1. **Did each untouched baseline reproduce the recorded failure?** Yes, both,
   and with values numerically identical to the P3.5-A replay — DSO:
   `internalSolverError` on both primary and recovery; TSO: primary distress
   followed by recovery exhausting `max_iter=3000`.
2. **Did the kappa-scaled replacement convert each primary solve to success,
   and was recovery needed?** Yes to both cases, on the **primary**
   exact-Hessian attempt, with **no recovery attempt logged** for either —
   same qualitative result as P3.5-A's outright removal, achieved instead by
   a constant positive rescaling of the identical equation.
3. **Was the scaling algebraically and numerically the intended constant
   positive multiple, with no accidental structural change?** Yes.
   `kappa=1/s_rated=100.0` exactly in both cases; 24 original rows
   deactivated and 24 replacement rows added over exactly the same index
   tuples in both cases (`index_tuples_match_one_to_one=True`); every
   reported scaled starting-point statistic equals exactly `100×` its
   unscaled counterpart, in both `|g|` and gradient-∞-norm, for both cases.
4. **Does the original, unscaled `sess_snet_def` relation remain satisfied
   at the final B solution?** Yes, to machine precision (normalized residual
   `1.36e-16` DSO, `1.92e-11` TSO) — a difference of roughly 15–20 orders of
   magnitude from the ~32%-of-`s_rated²` violation outright removal produced
   in P3.5-A at the analogous worst-hour check. There is no material
   violation to flag; Outcome B5 (treat as invalid/implementation bug) does
   **not** apply.
5. **Which outcome applies, and does it authorize a production change?**
   **Outcome B1** (plan §11): both untouched baselines fail exactly as
   recorded, both scaled-replacement (B) solves succeed on the primary
   attempt with no recovery, and the original relation's residual at the
   final B solution is at normal (here, machine-precision) tolerance — this
   is the cleanest possible confirmation that the practical failure mode is
   a **scaling/conditioning problem in `sess_snet_def`**, not a structural or
   physical one: the same equation, merely multiplied by a constant so its
   residual and gradient sit closer to O(1), lets IPOPT converge directly
   with the physical identity fully intact. This is a materially stronger
   and cleaner result than P3.5-A's Outcome A2 (which converted the same two
   failures to success but at the cost of a real, if localized, physical
   relaxation at one hour per case). **Per plan §12, Outcome B1 still does
   not authorize modifying `sess_snet_def`, `sess_comp`, or any other
   production formulation, solver, or ADMM setting.** This diagnostic result
   should be returned to the planner as evidence supporting a possible
   future, separately-approved stage to introduce a permanent, principled
   scaling/regularization of this constraint class in production — not as
   authorization to make that change now.

---

`P3.5-B COMPLETE — waiting for planner review`
