# Stage P3.5-A — `sess_snet_def`-only frozen diagnostic (executed)

Executed by the user in the real project Pyomo/IPOPT/MA97 environment via
`p35a_sess_snet_def_diagnostic.py` (repo root), against the two prescribed
frozen cases. Raw output: `data/SRP1/Results/FrozenSMOPF/P35A/p35a_report.json`.
No production file, solver option, ADMM parameter, or `sess_comp`/other
constraint was touched — only `sess_snet_def` for the one genuinely
installed shared-ESS index, on an independent in-memory clone.

---

## A. Frozen-file verification

| Case | Path | Expected SHA-256 | Actual SHA-256 | Match |
|---|---|---|---|---|
| DSO | `data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl` | `066117b8…637074711` | identical | ✅ |
| TSO | `data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl` | `51d90974…4b08355e` | identical | ✅ |

Metadata confirmed: DSO = agent DSO, node 7, `case33_2`, 2025/Autumn, cycle 8,
warm-started. TSO = agent TSO, `case9`, 2025/Summer, cycle 6, warm-started,
originally labeled `failure` with `status=warning, termination=maxIterations`.

Shared-ESS capacity check: DSO has exactly one shared-ESS index (`0`),
`s_rated=0.01`, `e_rated≈0.02417` p.u. — genuinely installed, as expected.
TSO has three indices; only index `1` is genuinely installed
(`s_rated=0.01`, `e_rated≈0.02417` p.u.); indices `0` and `2` are exactly
zero, confirming the P3 finding that the other two DSO interfaces carry no
capacity in this candidate solution.

Neither file was overwritten.

---

## B. Untouched baseline reproduction (A)

| | DSO A | TSO A |
|---|---|---|
| Primary termination | `internalSolverError` ("Error in step computation") | `internalSolverError`-type stall (huge dual infeasibility spike at iter 66, `dual_inf≈3.17e7`) |
| Recovery attempted | Yes (limited-memory) | Yes (limited-memory) |
| Recovery termination | `internalSolverError` again (449 iters, still `dual_inf≈7.05e-3`, `overall_nlp_error≈7.05e-3` — did not reach tolerance) | ran to `max_iter=3000` → `maxIterations` |
| **Final classification** | **status=error, termination=internalSolverError** | **status=warning, termination=maxIterations** |
| `succeeded` | **False** | **False** |

Both baselines reproduce the recorded P2.10/P3 failure signatures exactly
(DSO: primary+recovery both "Error in step computation", persistent, matching
the original Autumn/cycle8 record; TSO: primary distress → recovery exhausts
3000 iterations, matching the original Summer/cycle6 `maxIterations` record).
**The A/B attribution is valid for both cases** (plan §11, Outcome D does not
apply).

---

## C. Starting-point `sess_snet_def` diagnostics (before solving B)

| | DSO (24 rows) | TSO (24 rows) |
|---|---|---|
| `sch` range | `[2.13e-06, 7.63e-03]` | `[2.96e-06, 6.37e-03]` |
| `sdch` range | `[1.96e-06, 7.50e-03]` | `[2.37e-06, 5.41e-03]` |
| `pnet` range | `[-7.50e-03, 6.88e-03]` | `[-5.40e-03, 6.02e-03]` |
| `qnet` range | `[-2.26e-05, 3.29e-03]` | `[-1.85e-04, 2.75e-03]` |
| `|g|` (min / median / max) | `0 / 1.69e-24 / 1.40e-20` | `8.21e-22 / 4.90e-19 / 1.08e-13` |
| gradient ∞-norm (min / median / max) | `4.10e-05 / 1.57e-04 / 1.53e-02` | `1.93e-04 / 3.58e-03 / 1.27e-02` |
| rows with grad-∞-norm `< 1e-8 / < 1e-6 / < 1e-4` | `0 / 0 / 8 of 24` | `0 / 0 / 0 of 24` |
| `max|g| / max(s_rated², 1e-12)` | `1.40e-16` | `1.08e-09` |

Honest reading: at the frozen starting point the row is satisfied almost
exactly (`|g|` at the `1e-13`–`1e-24` scale, i.e. essentially machine-zero) in
both cases — expected, since these are warm-started from a nearly-converged
prior ADMM iterate. The gradient is genuinely small but **not literally
vanishing at the strict `1e-6`/`1e-8` thresholds** the plan specified: for
the DSO, 8 of 24 hourly rows fall below the loosest `1e-4` threshold
(the near-idle hours where `sch≈sdch` and `pnet,qnet≈0` simultaneously); for
the TSO, none do, though every row's gradient (`1.9e-4`–`1.3e-2`) is still
2–5 orders of magnitude below the `dual_inf` blow-up (`~1e7`) seen once the
solver starts struggling. So the row is weakly/poorly scaled relative to the
rest of the active set rather than exactly singular in isolation — consistent
with a conditioning contribution rather than a literal zero row.

---

## D. Variant results (B — `sess_snet_def` deactivated for the installed index only)

| | DSO A (baseline) | DSO B (variant) | TSO A (baseline) | TSO B (variant) |
|---|---|---|---|---|
| Status / termination | error / internalSolverError | **ok / optimal** | warning / maxIterations | **ok / optimal** |
| Primary succeeded | No | **Yes** | No | **Yes** |
| Recovery needed | Yes (also failed) | **No** | Yes (exhausted) | **No** |
| Iterations (primary, or final attempt) | 474 → 449 (recovery) | **109** | 66 → 3000 (recovery) | **50** |
| Objective at reported point | `9.1247e-03` (unconverged) | `9.1034e-03` | `2.19432` (unconverged) | `2.19425` |
| Dual infeasibility | `5.58e+08` → `7.05e-03` | `4.23e-09` | `3.17e+07` → `2.59e+03` | `6.18e-07` |
| Constraint violation | `4.30e-08` → `8.85e-14` | `3.92e-08` | `4.00e-08` → `7.23e-06` | `8.48e-09` |
| Complementarity | `8.45e-06` → `1.57e-07` | `1.84e-06` | `3.39e-06` → `8.95e-05` | `1.85e-06` |
| Overall NLP error | `2.47e+06` → `7.05e-03` | `1.84e-06` | `7.30e+05` → `2.59e+03` | `1.85e-06` |
| Solver-reported CPU time | `4.09s + 3.85s = 7.93s` | `0.79s` | `0.13s + 7.75s = 7.88s` | `0.09s` |

Deactivating **only** `sess_snet_def` for the installed index converts both
primary exact-Hessian solves from failure to a clean **primary** optimal
solution — no recovery attempt was needed in either case — in roughly
4–10× fewer iterations and 10–90× less solver time than the failed attempts.
The converged objective in B is within `0.003%`–`0.23%` of the (unconverged)
objective the failing baseline was heading toward, i.e. B lands in
essentially the same operating regime the original solve was trying to
reach, not a qualitatively different one.

---

## E. Removed-equation residual at the B solution

| | DSO | TSO |
|---|---|---|
| Rows evaluated | 24 | 24 |
| `max|g|` | `3.199e-05` | `3.208e-05` |
| `median|g|` | `5.53e-09` | `2.47e-06` |
| `max|g| / s_rated²` (normalized) | **`0.320`** | **`0.321`** |
| Worst row | `(e=0, s_m=0, s_o=0, p=23)` — last hour | `(e=1, s_m=0, s_o=0, p=0)` — first hour |
| Worst-row state | `sch=8.754e-03, sdch=1.71e-06, pnet=6.025e-03, qnet=2.883e-03` | `sch=6.849e-03, sdch=2.19e-06, pnet=3.474e-03, qnet=1.654e-03` |

At the worst hour in **both** cases the relation is broken by a **material**
~32% (normalized by `s_rated²`): the model's charging "apparent power" `sch`
(`≈8.75e-03` DSO / `≈6.85e-03` TSO) ends up noticeably larger than
`√(pnet²+qnet²)` (`≈6.68e-03` DSO / `≈3.85e-03` TSO) — i.e. once the
tying equation is gone, the optimizer uses some of the freed `sch/sdch`
headroom (still bounded by the linear `sess_s_limit`/link inequalities, but
no longer forced to match the dispatched P/Q) rather than reproducing the
physical apparent-power identity. This is localized: all other 23 of 24
hourly rows in both cases stay at or near machine-precision residual
(`median|g|` at the `1e-9`–`1e-6` scale) — only the single hour with the
most aggressive charging shows the effect.

---

## F. Interpretation

1. **Did each untouched baseline reproduce failure?** Yes, both. DSO:
   primary and recovery both terminate `internalSolverError`, matching the
   recorded Autumn/cycle8 outcome exactly. TSO: primary distress followed by
   a recovery that exhausts `max_iter=3000`, matching the recorded
   Summer/cycle6 `maxIterations` outcome exactly.
2. **Did `sess_snet_def`-only deactivation convert each primary solve to
   success?** Yes, both — DSO and TSO variants both report
   `status=ok, termination=optimal` on the **primary** exact-Hessian solve.
3. **Did either success require recovery?** No. Both B solves converged on
   the primary attempt; no recovery log was produced for either.
4. **How close did the final B solution remain to the removed
   `sess_snet_def` manifold?** Mixed. 23 of 24 hourly rows in each case
   remain essentially on the manifold (residual at or near machine
   precision). But the single most-active hour in each case shows a
   material violation (~32% of `s_rated²`) — this is **Outcome A2** from the
   plan (§11), not the stronger A1: the relaxed model likely converges partly
   by exploiting physically unavailable apparent-power freedom at that one
   timestep, not purely by removing a numerically-damaging-but-physically-inert
   row.
5. **Is Candidate 1 strongly confirmed, partially supported, or not
   confirmed?** **Partially supported / confirmed as a proximate numerical
   trigger, under Outcome A2.** The evidence is unambiguous that
   `sess_snet_def` (for the genuinely installed shared-ESS index) is
   sufficient, on its own, to flip both frozen failures to clean primary
   convergence, with no other change. But because the B solution does not
   stay uniformly close to the removed physical manifold (one hour per case
   is materially off it), this does **not** establish that the equality is
   numerically harmful while physically inert — it establishes that removing
   it helps IPOPT, at the cost of a real (if localized, single-timestep)
   physical relaxation. Per plan §12, this does **not** authorize deleting,
   reformulating, or replacing `sess_snet_def`, changing `sess_comp`, or any
   other production change. A follow-up structural design (e.g. a better-
   scaled or regularized version of the same physical relation, rather than
   its outright removal) would need a new planner-approved stage.

---

`P3.5-A COMPLETE — waiting for planner review`
