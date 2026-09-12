# P5.12-G — NO-SOLVE Tier 0 + Tier 1 telemetry and local-geometry forensic

**Authorship note.** The diagnostic `p512_g_bound_row_geometry.py` was written by the
Worker, whose session was terminated by an external API spend limit before it could
run the script or write this report. The Planner verified the script's safety
properties (solver guards installed and asserted; writes confined to
`data/SRP1/Results/P512G/`; refuses a pre-existing output directory), executed it
once, and authored this report from the resulting journal. No frozen artifact was
modified; repository integrity was verified before and after.

Artifacts: `data/SRP1/Results/P512G/p512g_journal.json`, `.../manifest.json`.
Runtime 3.79 s. **Zero solver calls** (`OptSolver.solve` = 0,
`SystemCallSolver._execute_command` = 0, guards armed to raise on entry).

This forensic establishes observational facts about one frozen instance. It does not
establish a causal mechanism.

## Gates

All Tier 0 validation gates pass: summary rows 90 / 3001 / 152; factorization events
89 / 3000 / 151; multi-trial events 36 / 27 / 70; Arm A safeguard rows 2924 first at
77; V2 rows 82-102; Arm A terminal violation `6.1921343776988665e-02` and terminal
`mu = 1.8449144625279508e-06`. Two `termination` entries report `match: false`; this
is a string-comparison artifact (full IPOPT exit text versus shorthand), not a
substantive mismatch.

Tier 1 gates: snapshot semantic digest `3fd294d7…a85f` matches; optimization space
**9166** columns (124 lower-only, 7266 two-sided, 102 equal-bound excluded), all
exact; **0 AD/derivative failures**; 4 Jacobian builds of 7826 rows at ~0.24 s each
(budget 6, time limit 1200 s).

## ESTABLISHED

**Regularization.** `delta_c` is nonzero in **zero** factorizations in **all three**
runs — not merely Arm A. `delta_x` is nonzero in 36 (V1, iterations 7-84), 27
(Arm A, iterations 7-68), and 70 (V2, iterations 7-76 and 113-144+) events. **No
`delta_x` event occurs during the Arm A stall** (none at iteration >= 77). The
converged controls regularize on 40-46% of iterations; the stalled run on 0.9%, all
before the stall.

**Step truncation.** Every line-search event in all three runs is single-trial
(89 / 3000 / 151; zero multi-trial). At Arm A iterations 2990-2999 the first trial
alpha equals the accepted alpha (`5.42e-05`), sufficient reduction and filter
acceptability both pass, and `ALPHA_MIN ~ 1.06e-13` is eight orders lower. The step
is therefore **never backtracked or rejected**; it is the maximum the line search
attempts. That this maximum is the fraction-to-boundary limit follows from the
standard semantics of IPOPT's "Starting checks for alpha (primal)" line and is
INFERRED, not proven from preserved data.

**Safeguard corrections.** Arm A: 2924 events, iterations 76-2999, all on `z_U`
(`n_zL = 0`), corrections min `3.37e-08`, max `9.7e-07`, mean `4.02e-08`. V2: 21
events, iterations 81-101, min `1.97e-08`, max `1.18e-06`, mean `3.74e-07`. Against
`||z||_inf ~ 1e2`, `mu = 1.8e-06` and an accepted step of `5.4e-05`, these are orders
below the multiplier scale. (The earlier "~3.4e-08" characterization was the
*minimum*; the maximum is `9.7e-07`.)

**Constraint-row geometry.** Raw and reconstructed-scaled row-norm ladders are
essentially identical across S0/SA/S1/S2: below 1.0 → 1996 rows at every state;
below 0.1 → 55; below 0.01 → 36 (SA 32); below 0.001 → 24; below 1e-4 → **0**
everywhere. The Arm-A-unique small-gradient row set at scaled 1e-3 is **empty**.
Scaled norms are a documented reconstruction (`min(1, 100/||row||_inf)` evaluated at
S0), not IPOPT's internal factors, which are not printed.

**Bound geometry.** Of 7390 bounded variables, Arm-A-unique near-bound variables at
1e-4 number **1** (`flex_p_up[2,0,0,6]`), against 4313 shared with both controls and
400 near-bound only in the controls. At the tightest threshold Arm A has *fewer*
variables very close to bounds (1e-8: 5, versus 3652 for S0/S1/S2).

**Blocking variable.** `NEAR-BOUND CANDIDATE SET ONLY — BLOCKING VARIABLE NOT
IDENTIFIABLE FROM PRESERVED ARTIFACTS.` The logs and `.sol` files preserve the
aggregate step and the post-step iterate, but not the internal search direction
`d_x`, without which no variable can be shown to impose the fraction-to-boundary
limit.

**P5.3 relation.** The historical `sess_snet_def` row family is **absent** from the
current model, which uses `sess_pnet_def`. No recurrence is inferred.

## SUPPORTED (present in Arm A, materially weaker in both controls)

**Primal/dual activity disagreement.** Rows dual-active but primal-inactive: Arm A
**1912**, V1 249, V2 251 (~7.6x). Dual-active rows: 6182 versus 4483 and 4483.
Primal-active-but-dual-inactive is 0 in all three states.

**Confound, stated explicitly.** SA is infeasible at termination
(`6.19e-02`) while S1 (`1.96e-07`) and S2 (`6.88e-12`) are feasible. A violated row
carries a large multiplier while its slack is large, which produces exactly this
disagreement pattern. The signal is therefore substantially, and possibly wholly, a
restatement of infeasibility rather than independent evidence of degenerate geometry.

## REFUTED or WEAKENED

- **Strong-form constraint-Jacobian rank failure requiring constraint
  regularization**: unsupported — `delta_c` is identically zero in all three runs.
  This does not prove LICQ, full numerical rank, or good conditioning.
- **`delta_x` frequency as a failure signature**: refuted — the converged controls
  regularize far more often, and Arm A does not regularize at all during its stall.
- **Safeguard corrections as a numerical driver**: weakened to cosmetic by magnitude.
- **An Arm-A-specific near-bound cluster**: not found (one variable).
- **An Arm-A-specific weak constraint row**: not found (empty set, identical ladders).
- **Line-search rejection as the cause of the tiny step**: refuted — zero backtracking.

## Answers to the required questions

1. Constraint-side anomaly unique to Arm A? **No.**
2. Arm-A-specific near-bound set? **No** — one variable at 1e-4.
3. Blocking variable identifiable? **No** — `d_x` not preserved.
4. Is `alpha_pr ~ 5e-5` fraction-to-boundary limited rather than rejected?
   Rejection is **excluded**; the fraction-to-boundary reading is strongly indicated
   but INFERRED from IPOPT's log semantics.
5. `delta_c` zero through the Arm A stall? **Yes** — and in both controls too.
6. `delta_x`? 36 / 27 / 70 events; none during the Arm A stall; more frequent in the
   converged runs. It cannot explain the failure.
7. Safeguard corrections? **Cosmetic by magnitude.**
8. Tier-2 spectral question left? Trigger 2 (activity disagreement) is met on its
   face, but is confounded by infeasibility; triggers 1 and 3 are not met. On this
   evidence Tier 2 is **not** justified.

## Verdict

NO ARM-A-UNIQUE LOCAL-GEOMETRY SIGNATURE
