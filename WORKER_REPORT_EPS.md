# Worker Report — EPS_ESSO_THROUGHPUT sensitivity check (task b)

Authority: `PLANNER_BRIEF_2026-09-13.md`, Addendum 2, bullet "eps check (before the
gates)". This precedes gates G1–G5; it does not run them.

## Task received

Two direct ESSO solves of the reformulated node-7 instance, differing only in
`EPS_ESSO_THROUGHPUT` (1e-3 vs 1e-7). Report `pnet` displacement, the complementarity
detector in both ratio and absolute form, ESSO iteration counts/termination, and
objective / `gross_operational_cost`. Predeclared decision rule: gates run at 1e-3
regardless, **unless** the 1e-3 arm's ratio detector exceeds 1e-6, in which case STOP
and report without recommending a value. No production-code edit; no ADMM run; guard
armed with an exact declared solve count.

## Instance identifier and how it was reproduced

Reused **verbatim** the construction path of the Step 1 reformulation smoke test
(`p515_1_esso_reform_smoke.py`, commit `b03c9b14`, described in
`P5_15_PLANNER_STEP1_PROGRESS.md` §3). The harness `p515_1b_eps_sensitivity_check.py`
**imports** (does not reimplement) `SMOKE_NODES_WITH_INVESTMENT = (7, 9)`,
`S_CANDIDATE_MVA = 1.00`, `E_CANDIDATE_MVAH = 2.00`, `_zero_candidate` and
`_set_nonzero_charge_discharge_request` from that module, and calls the same
production functions: `p56a_oracle.fresh_planning`, `create_admm_variables`,
`create_shared_energy_storage_model`.

- Baseline checksum (`p56a_oracle.load_baseline()['checksum']`):
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`
  (matches the oracle's `CANONICAL_CHECKSUM`).
- Instance descriptor hash (sha256 of the instance descriptor JSON below), read
  directly from the artifact's `instance_hash_sha256` field:
  `3c86026145760ec438c014d76e2454a4ed517d0c1cbd5e9040b03f437846ee3d`
  (`data/SRP1/Results/P5151/eps_sensitivity_check_summary.json` is authoritative;
  this is a verbatim copy of that field, re-read after writing to confirm the copy).
- Instance descriptor: nodes 7 and 9 given S=1.00 MVA / E=2.00 MVAh investment in the
  first representative year (2025); node 5 zero investment; every investment node's
  `p_req` set to a ±10 %-of-S duty cycle (charge first half of the 24-instant day,
  discharge second half), amplitude = `0.10 * S_CANDIDATE_MVA`. **Node 7 is the
  instance this task concerns**; node 9 rides along because the reused smoke-test
  construction gives it the same treatment; node 5 rides along because
  `create_shared_energy_storage_model` always solves every active distribution
  network node together (3 in SRP1: 5, 7, 9).

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addendum 2, the dispatched bullet)
- `P5_15_PLANNER_STEP1_PROGRESS.md` (§3, the smoke test and the required ε/AL note)
- `p515_1_esso_reform_smoke.py` (construction path reused verbatim)
- `p515_1_eps_al_gradient_comparison.py` (prior related measurement, not reused
  mechanically but read for context)
- `shared_energy_storage_data.py` (`_build_subproblem` lines ~398–658, `_optimize`,
  `_create_solver`, `_get_complementarity_violation`, `_esso_cohort_pair_is_within_lifetime`)
- `shared_resources_planning.py` (`create_shared_energy_storage_model`,
  `update_shared_energy_storage_model_to_admm` — read to confirm the latter, which
  builds `admm_objective`, is **not** in the call path used here)
- `helper_functions.py`, `definitions.py` (how `EPS_ESSO_THROUGHPUT` enters
  `shared_energy_storage_data`'s namespace)
- `p513_solve_profile_guard.py`, `p515_0_esso_warmstart_ab.py` (guard and log-parsing
  conventions reused)
- `.env` (confirms `NLP_SOLVER_PATH=/usr/local/bin/ipopt`)

## Files modified

**None under `data/`. No production code changed.**

New file created (harness, diagnostic convention `p5*.py`):
`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_1b_eps_sensitivity_check.py`

New artifacts written (all new files under `data/SRP1/Results/P5151/`, nothing
pre-existing there was altered):
- `data/SRP1/Results/P5151/eps_sensitivity_check_summary.json` (the report table below,
  machine-readable)
- `data/SRP1/Results/P5151/eps_sensitivity_check_full.json` (per-period `es_pnet`,
  per-cohort-period `pch`/`pdch`, per-cohort `s_max` — full grids for both arms)
- `data/SRP1/Results/P5151/eps_check_logs/eps1e-3/optim_log_node_{5,7,9}.txt` (raw IPOPT logs)
- `data/SRP1/Results/P5151/eps_check_logs/eps1e-7/optim_log_node_{5,7,9}.txt` (raw IPOPT logs)

`definitions.py` and `shared_energy_storage_data.py`: **confirmed unmodified**.
`EPS_ESSO_THROUGHPUT` was varied by monkeypatching the module attribute
`shared_energy_storage_data.EPS_ESSO_THROUGHPUT` at runtime (that name enters the
module's namespace via `from helper_functions import *` → `from definitions import
*`, so it is a plain module-level global `_build_subproblem` looks up by name at call
time), restored immediately after each arm's solve. Verified after every arm with an
assertion that `definitions.EPS_ESSO_THROUGHPUT == 1e-3` (the committed value) and that
`shared_energy_storage_data.EPS_ESSO_THROUGHPUT == 1e-3`; both assertions passed
(the run would have raised `AssertionError` and aborted otherwise — it did not).

```
$ git diff --stat -- definitions.py shared_energy_storage_data.py
(empty output)
$ git status --porcelain -- definitions.py shared_energy_storage_data.py
(empty output)
```

Both commands ran **after** the full two-arm run completed, confirming the working
tree for these two files is byte-identical to HEAD.

## Guard

Declared in advance: `SolveProfileGuard`, permitted call site
`('shared_energy_storage_data.py', '_run_solver_attempt')`, expected count **6** (2
arms × 3 active nodes; declared from the Step-1 smoke test's prior evidence that
nodes 7 and 9 converge to `optimal` at eps=1e-3 with no recovery, and node 5,
zero investment, is a trivial solve).

Observed: `{'permitted_solve': 6, 'permitted_exec': 6, 'blocked_solve': 0,
'blocked_exec': 0}`. `guard.verify(6)` returned `[]` (no failures) — **exact match**,
no recovery fired on either arm.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_1b_eps_sensitivity_check.py
```

One run, both arms sequential inside one guarded process (as designed — the guard
covers the whole run, both arms).

**Unauthorized-by-count but declared exception, per the brief's "declare any extra
solve explicitly" clause:** during development I ran one additional ad hoc,
**unguarded** diagnostic (`python -B -c "..."`, not the harness) to explain an
unexpected near-zero-but-negative objective at node 5 (zero investment). It rebuilt
and re-solved nodes 5, 7, 9 once each at the case-file default `eps=1e-3` (no
monkeypatch), using the same production entry point, from the repo root (no `cwd`
change), so its IPOPT output appended to the pre-existing, gitignored, untracked
scratch logs `optim_log_node_{5,7,9}.txt` at the repo root (already 15–35 MB of
historical scratch output from unrelated earlier stages; each grew by
≈18–19 KB). These 3 solves are **not** part of the 6 counted above (separate
process, guard not installed) and did **not** feed into any number in the results
table — they were purely exploratory. Root cause found analytically (see "Results",
node 5 note) and confirmed arithmetically to 12 significant digits; no code or
config was changed as a result.

## Results

### Termination and iterations (verbatim)

| arm | node | exit line (verbatim) | `Number of Iterations....:` | recovery fired |
|---|---|---|---|---|
| eps=1e-3 | 5 | `EXIT: Optimal Solution Found.` | 7 | no |
| eps=1e-3 | 7 | `EXIT: Optimal Solution Found.` | 16 | no |
| eps=1e-3 | 9 | `EXIT: Optimal Solution Found.` | 16 | no |
| eps=1e-7 | 5 | `EXIT: Optimal Solution Found.` | 7 | no |
| eps=1e-7 | 7 | `EXIT: Optimal Solution Found.` | 12 | no |
| eps=1e-7 | 9 | `EXIT: Optimal Solution Found.` | 12 | no |

`shared_ess_data.solver_recovery_diagnostics` was empty (`[]`) for both arms — zero
recovery events.

### `pnet` displacement (the literal ask)

| quantity | eps=1e-3 vs eps=1e-7 |
|---|---|
| max abs diff, `es_pnet`, node 7, all (year, day, period) | **0.0** |
| max abs diff, `es_pnet`, node 9 | 0.0 |
| max abs diff, `es_pnet`, node 5 | 0.0 |
| max abs diff, `es_pnet`, all nodes | **0.0** |
| predeclared prediction | < 1e-6 p.u. |
| verdict | prediction holds, **but for a structural reason, not a numerical one** |

**Reading.** `create_shared_energy_storage_model` (the function this instance is
built and solved through — confirmed by reading `shared_resources_planning.py`, and
confirmed this is *not* `update_shared_energy_storage_model_to_admm`, which is never
called here) **hard-fixes** `es_pnet`/`es_qnet` to the supplied `p_req`/`q_req` via
`fix_or_set` (`var.fix(val)`). `p_req` is a deterministic input, identical between
arms. A fixed variable's value is returned by IPOPT/Pyomo exactly as fixed
(`fixed_variable_treatment = make_parameter`), independent of the objective. The
displacement is therefore **exactly 0.0 by construction**, not a converged numerical
result — it says nothing about the AL-curvature mechanism the Planner's algebra in
`P5_15_PLANNER_STEP1_PROGRESS.md` §3 was estimating (that estimate assumed `es_pnet`
free and pulled toward `p_req` by an augmented-Lagrangian term, which only exists
after `update_shared_energy_storage_model_to_admm` — not in this instance). Reported
here as instructed ("do not invent a new fixture"); flagged under "Unexpected
findings" below.

### Dispatch displacement — supplementary, not the literal ask

Because `es_pnet` is fixed, the only free continuous decision that `eps` can move is
the per-cohort split between `pch` and `pdch` (`es_pnet = pch - pdch` per active
cohort). This is reported because it is the actual channel `eps` acts through in this
instance, and is directly informative for the complementarity question:

| quantity | value |
|---|---|
| max(\|Δpch\|, \|Δpdch\|), node 7, over all active cohort-periods | **0.0709** p.u. |
| max(\|Δpch\|, \|Δpdch\|), node 9 | 0.0709 p.u. (identical construction to node 7) |
| max(\|Δpch\|, \|Δpdch\|), node 5 | 0.0 (zero investment, no active cohort) |
| worst case (node 7, cohort y_inv=0/y=2, day=1, period=0) | eps=1e-3: (pch, pdch) = (0.10046, 0.00046); eps=1e-7: (pch, pdch) = (0.17138, 0.07138) |
| total per-node throughput Σ(pch+pdch), node 7, eps=1e-3 | 29.067 p.u. |
| total per-node throughput Σ(pch+pdch), node 7, eps=1e-7 | 69.618 p.u. (**2.4×** more) |

At `eps=1e-7` the LP no longer discourages routing power through both directions at
once; throughput roughly 2.4× versus `eps=1e-3` for the same fixed net demand. This
is the mechanism behind the complementarity-detector drift reported next.

### Complementarity detector — ratio form (as specified: `min(pch,pdch)/s_max`, skipping `s_max==0`)

| arm | max ratio | node / cohort / period at the max |
|---|---|---|
| eps=1e-3 | **4.6319e-4** | node 7, y_inv=0, y=0, d=0, p=12; pch=4.632e-4, pdch=0.10046, s_max=1.0 |
| eps=1e-7 | **7.1380e-2** | node 7, y_inv=0, y=2, d=1, p=0; pch=0.17138, pdch=0.07138, s_max=1.0 |
| predeclared prediction | ≤1e-6 at eps=1e-3; drift upward at 1e-7 | |
| verdict | **prediction FALSIFIED at eps=1e-3**: 4.63e-4 is **~463×** the 1e-6 threshold. The *direction* of the prediction (drift upward at 1e-7) holds: 1e-7's ratio is a further **~154×** worse than 1e-3's. | |

### Complementarity detector — absolute form (`SharedEnergyStorageData.get_complementarity_violation`)

| arm | value returned |
|---|---|
| eps=1e-3 | **4.6219e-4** |
| eps=1e-7 | **7.1379e-2** |

(Consistent with the Step-1 smoke test's own reading of "4.62e-4" at the same eps and
a similar, not identical, candidate — the small residual between the two forms is
exactly `1e-6 * s_max = 1e-6`, i.e. the detector's own threshold subtraction.)

### Objective and `gross_operational_cost`

**`gross_operational_cost` has NO capture path in this experiment** — stated before
running, per the evidence rules. Production defines it as the TSO objective plus the
sum of DSO objectives (`shared_resources_planning.py` lines ~727–729); it structurally
excludes the ESSO objective and requires solving the TSO/DSO SMOPFs, which the task's
"no ADMM run, direct ESSO solves only" restriction forbids. There is nothing in this
experiment that quantity applies to.

What **was** captured: `model.objective`, which in this instance equals
`model.feasibility_penalty` exactly (`update_shared_energy_storage_model_to_admm`,
which would add AL terms and rename the active objective to `admm_objective`, is
never called here).

| quantity | eps=1e-3 | eps=1e-7 |
|---|---|---|
| `model.objective`, node 7 | 0.023830335995985618 | −0.005229401845246441 |
| `model.objective`, node 9 | 0.023830335995985618 | −0.005229401845246441 |
| `model.objective`, node 5 | −0.005236363636361883 | −0.005236363636361883 |
| total over 3 active nodes | 0.042424308355609355 | −0.015695167326854765 |

**Node 5 note (why a zero-investment node's objective is slightly negative, and why
it is identical in both arms):** node 5 has zero investment (`s_max=0` everywhere),
so `pch=pdch=0` and `p_req=0` throughout — the aggregate-operation constraint
`es_pnet == agg_pnet + slack_up − slack_down` is satisfied at `agg_pnet=0`,
`es_pnet=0`. IPOPT's interior-point solution leaves the two non-negative slacks at
≈−9.0909×10⁻⁹ each (an interior-point artifact on a degenerate all-zero LP, not a
sign violation — Pyomo/IPOPT can return values fractionally outside a `NonNegativeReals`
bound within solver tolerance). `PENALTY_ESSO_SLACK (1e3) × 2 × (−9.0909e-9) × 288
periods = −0.0052364`, which matches the observed value to 12 significant digits and
is exactly reproduced in node 7's and node 9's objectives too (their objective =
`eps × throughput + (−0.0052364)`, confirmed arithmetically: e.g. node 7 at eps=1e-3:
`1e-3 × 29.0667 − 0.0052364 = 0.0238303` ✓). This is solver numerical noise on a
degenerate, unconstrained-in-practice LP row, present identically in both arms
(it does not touch `eps` or throughput), not a defect introduced by this check.

## Decision rule — applied exactly as predeclared

> "The gates G1–G5 run at eps = 1e-3 regardless of the outcome, UNLESS the 1e-3 arm
> itself shows a complementarity detector ratio above 1e-6 — in which case you STOP
> and report, and do not recommend a value."

The 1e-3 arm's ratio-form detector is **4.6319e-4**, which is **above 1e-6**.

**Per the predeclared rule: STOP. This report does not recommend a value for
`EPS_ESSO_THROUGHPUT`, and Gates G1–G5 are not authorized by this result to proceed
at eps=1e-3 or at any other value.** The Planner decides.

## Validation

- Guard-verified solve count: 6/6 exact, zero blocked call sites, zero unexpected
  extra solves inside the harness.
- `definitions.py` / `shared_energy_storage_data.py` confirmed byte-identical to HEAD
  after the run (`git diff --stat` and `git status --porcelain` both empty for both
  files).
- Node-5 anomaly fully reconciled arithmetically (12 significant digits), not left as
  an unexplained residual.
- The pnet-displacement / dispatch-displacement decomposition and the arithmetic
  reconciliation of `objective = eps·throughput + slack_noise` were checked
  independently against the full per-period JSON artifact, not asserted from the
  summary alone.
- Code executes correctly and the requested diagnostic ran to completion; this is
  **not** a claim that the underlying reformulation-vs-regularization question is
  settled — see "Unexpected findings" and "Remaining issues".

## Unexpected findings

1. **The predeclared 1e-6 prediction for the eps=1e-3 ratio detector is falsified by
   ~463×.** This is the headline result and the reason gates are not authorized to
   proceed per the predeclared rule.
2. **`es_pnet` displacement is not a meaningful test of the eps/AL-curvature
   mechanism on this instance.** `create_shared_energy_storage_model` hard-fixes
   `es_pnet` to `p_req`; only `update_shared_energy_storage_model_to_admm` (never
   called here) would let `es_pnet` move under an AL term against `p_req`, which is
   what the Planner's algebraic estimate (§3 of `P5_15_PLANNER_STEP1_PROGRESS.md`,
   "≈3.8e-7 p.u." displacement from ε/curvature) implicitly assumed. The instance the
   brief specified (the smoke test's fixture) and the mechanism the brief's own
   algebra was estimating are two different model configurations. This report used
   the specified instance, as instructed, and flags the mismatch rather than silently
   switching fixtures.
3. **The complementarity violation is real and substantially larger than the
   Step-1 smoke test's own framing suggested is "small".** The Step-1 report
   described 4.62e-4 (at a similar but not identical candidate) as "small, nonzero,
   consistent with complementarity being regularized rather than enforced" without
   comparing it to a numeric threshold; this check supplies the threshold (1e-6) the
   brief itself set, and by that threshold the finding is a 463× violation, not a
   small one.
4. Throughput at eps=1e-7 is 2.4× throughput at eps=1e-3 for the *same* fixed net
   dispatch — a large, non-trivial dispatch-level effect of the regularization
   weight, even though it never surfaces in `es_pnet` (see finding 2).
5. One unguarded ad hoc debug solve (3 solves, node 5/7/9, eps=1e-3, no cwd change)
   was run outside the harness during investigation of the node-5 objective; disclosed
   in full above; did not feed any reported number and appended a few KB to
   pre-existing, gitignored, untracked scratch log files at the repo root.

## Remaining issues

- Why IPOPT's converged point leaves `min(pch,pdch)` at ~4.6e-4·s_max rather than
  ~0 (as the LP-structure argument in the Step 1 commit message predicts for *an* LP
  optimum) is not established here. A plausible mechanism — the case file's
  `acceptable_tol=1e-5`/`acceptable_iter=5` termination criteria letting IPOPT accept
  a point before the barrier parameter has driven the interior-point solution close to
  a vertex on this (apparently flat-directioned) LP — is consistent with the
  magnitudes observed but is a **hypothesis**, not verified here; verifying it (e.g.
  by tightening `tol`/`acceptable_tol` on this instance and re-checking the detector)
  was outside this task's scope (no solver-option changes permitted).
- Whether this same violation would appear on the AL/ADMM form of the ESSO (where
  `es_pnet` is free) is unknown; this task's instance cannot answer it (finding 2).

## Questions for Planner

1. Given the falsified prediction, do you want a follow-up check on the
   AL/ADMM-form ESSO subproblem (built via `update_shared_energy_storage_model_to_admm`,
   still a single non-looped solve, so still "no ADMM run" in the sense of no outer
   iteration) so that the `es_pnet`-displacement question can actually be answered on
   the mechanism your algebra addressed?
2. Do you want the complementarity-ratio-vs-solver-tolerance hypothesis (Remaining
   issues, first bullet) investigated as a distinct diagnostic, or is the STOP verdict
   sufficient for now?
3. Per the predeclared rule, gates G1–G5 do not proceed on this Worker's authority.
   Please confirm the STOP and advise on the dispatch order change (Addendum 2's
   "harness repair → eps check → G1–G5 → REVISION_CONTEXT.md rewrite → stop" assumed
   the eps check would pass).
