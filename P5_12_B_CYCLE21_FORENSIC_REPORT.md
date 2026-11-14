# P5.12-B — cycle-21 local-NLP forensic audit

**The cycle-21 failure reproduced deterministically, was preserved before any
cycle-level update, and is fully characterised. One frozen A/B is proposed and
not implemented.**

---

## 1 — Provenance and Git state

| | |
|---|---|
| hostname | `Micaels-Mac-Studio.local` (macOS 26.6.2, arm64) |
| repository | `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation` |
| runtime | `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python` |
| Python / NumPy / pandas / SciPy / Pyomo / copulas | 3.11.11 / 2.4.2 / 3.0.0 / 1.17.0 / 6.9.5 / 0.14.0 |
| IPOPT | 3.14.18, ASL 20241111, `/usr/local/bin/ipopt` |
| linear solver | MA97 |
| scenario checksum | `5a02b77c…10358`, canonical |

| | initial | final |
|---|---|---|
| branch | `feature/derivative-free-planning` | same |
| HEAD | `72468913499f5f9eca77ddbef6ed71fd61be7a97` | **unchanged** |
| tracked modifications | 0 | **0** |
| upstream | **ahead 0, behind 0** | unchanged |

**HEAD moved between P5.12-A and P5.12-B**, from `808590d2` to `72468913`, via
`9b8af813` ("docs: record P5.12-A and authorize P5.12-B") plus a merge. Only
`.md` files changed — zero `.py` files differ from `808590d2`, and `808590d2`
remains an ancestor. The branch is now level with origin, so the unpushed
single-copy risk flagged in P5.9 through P5.12-A is resolved.

Accepted evidence verified before execution and unchanged after:

| report | SHA-256 | required |
|---|---|---|
| P5.10 | `fe674ee534ba9b7cf72c188b756a1ad585133727b53aad453091531fd289311e` | match |
| P5.11 | `a9f9e3422a533d993ccd890cbee53a4a8e1dcbbc3469e95af3a592ec04281b3b` | match |
| P5.12-A | `a1424ec1b47944ad1aa5a9038306ab1fec4870bf0a3ed41993a29ba1f2c89e96` | match |

*Preservation note:* the P5.11 and P5.12-A reports and the `p511_*` / `p512_*`
harnesses remain **untracked**. The P5.9/P5.10 reports and `p510_*.py` are
tracked. Not a stopping condition — all are on disk and hash-verified — but they
are not yet in git.

## 2 — Diff and commands

**No tracked file was modified.** `shared_resources_planning.py`, `network.py`,
`network_data.py`, `helper_functions.py`, `admm_parameters.py` and
`data/SRP1/SRP1_params.json` all unchanged. New untracked files only:

```
p512_b_cycle21_forensic.py             the forensic harness
data/SRP1/Results/P512B/               evidence
p512_b.log                             console log
```

Commands, in order:

```
shasum -a 256 P5_10_… P5_11_… P5_12_A_…          # accepted-evidence gate
git branch --show-current; git rev-parse HEAD    # repository gate
git rev-list --left-right --count origin/feature/derivative-free-planning...HEAD
python p512_b_cycle21_forensic.py                # single process
```

### Instrumentation, and why it needed no production edit

`NetworkData.optimize` already accepts `failure_snapshot_callback` and
`pre_solve_snapshot_callback`, and already clones the pre-solve model when either
is supplied (`network_data.py:54-66`). Both coordination entry points already
pass their own callbacks and already carry `cycle=iter`. Three process-local
wrappers were therefore sufficient:

- `update_distribution_coordination_models_and_solve` and
  `update_transmission_coordination_model_and_solve` — to learn the cycle and to
  inspect results immediately after the solves;
- `NetworkData.optimize` — to attach capture callbacks at cycles 20 and 21.

**Production's own callbacks are chained, not replaced**, so production's failure
snapshots occurred exactly as they would unobserved. The pre-solve callback fires
after each block's solve with both the pre-solve clone and the result, supplying
Gate B2's manifest and Gate B4's comparison from one capture.

Termination is by raising `Cycle21Failure` from the coordination wrapper, which
propagates out of `run_operational_planning` **before**
`update_and_check_convergence`. No consensus, dual, ρ or recourse update was
applied and cycle 22 never began.

`num_max_iters` was set to 21 on the per-evaluation deep copy only.
`parallel_execution` is `False` for SRP1, so no task-cancellation semantics arose.

## 3 — Existing P5.12-A evidence inventory (Gate B0, read-only)

`data/SRP1/Results/P512A/` holds five trajectory JSONs plus provenance. From
them, before running anything:

- the failing-block **identity at cycle 21 was not recorded** — P5.12-A stored
  `local_solves_ok` per cycle, not per block;
- **no solver status or termination condition** for any cycle-21 block existed;
- **no cycle-20 or cycle-21 pre-solve state** existed;
- raw IPOPT logs were overwritten by later cycles, because P5.12-A continued past
  the failure to cycle 100;
- accepted cycle markers for deterministic matching were available: recourse
  `2352009862.13208` (cycle 1), `1461174062.836328` (13), `1402384338.113119` (25).

The limitation this stage had to work around is exactly P5.12-A's
post-failure continuation: nothing about cycle 21 could be recovered from it, and
per instruction nothing was inferred from later cycles.

## 4 — Cycles 1-20 reproduction

The trajectory reached cycle 21 and failed there, matching P5.12-A. **No local
failure occurred before cycle 21**, which is itself a required check: P5.12-A
recorded failures at cycles 21-24, and cycle 21 is confirmed as the first.

ρ was held at `{v: 1.5, pf: 300.0, ess: 1.0}` throughout and adaptive ρ remained
disabled; the failing block's captured ρ at both cycle 20 and cycle 21 is
identical. All 48 blocks were captured at cycle 20, every one solved
successfully.

## 5 — Cycle-20 and cycle-21 manifests

Captured per block at both cycles: agent, network, year, day; active variable and
constraint counts; active objective name and value at the start point; ADMM
objective scale; ρ; per-family primal summaries (count, min/max, |·|₁, at-bound
count, nonfinite count, SHA-16 of the value vector); consensus target and dual
parameters; `ipopt_zL_in` / `ipopt_zU_in` / `dual` suffix summaries; starting
constraint violations by family via production's own `scan_constraints`; solver
status, termination and IPOPT log.

Full manifests: `data/SRP1/Results/P512B/p512b_cycle21_forensic.json`.

## 6 — Cycle-21 results obtained before termination

| | |
|---|---|
| blocks attempted at cycle 21 | **36** (all DSO blocks) |
| succeeded | **35** |
| failed | **1** |
| TSO blocks attempted | **0** — abort occurred during the DSO stage |

The DSO stage is solved before the TSO stage, and the abort fired at the end of
the DSO stage. **The DSO-side consensus update never ran**, so the limitation
anticipated in the plan (a TSO-first failure leaving a DSO update applied) did
not arise.

## 7 — The failing block

```
block            : DSO:case33_3 | 2025 | Spring
solver status    : warning
termination      : maxIterations
message          : Ipopt 3.14.18: Maximum Number of Iterations Exceeded.
iterations       : 3000
```

Final two IPOPT iterations:

```
2999  1.3022138e+03 6.19e-02 5.16e-02  -5.7 1.69e+00    -  5.32e-05 5.42e-05h  1 z
3000  1.3022136e+03 6.19e-02 5.16e-02  -5.7 1.69e+00    -  5.23e-05 5.42e-05f  1 z
```

Step lengths have collapsed to `~5e-05` and the objective is frozen at
`1.3022e+03`. The exit is **not** a restoration failure and **not** a linear-solver
error: MA97 factorized to the end (`HSL_MA97: delays 476, nfactor 259640`). The
solve stalled.

## 8 — Matched cycle-20 versus cycle-21 comparison

The same block, the two consecutive solves:

| family | cycle 20 | cycle 21 |
|---|---|---|
| **solver** | optimal, **115 iterations** | maxIterations, **3000 iterations** |
| unscaled dual infeasibility | **1.1323e-03** | **5.1598e+01** |
| unscaled constraint violation | **7.2150e-07** | **6.1921e-02** |
| unscaled complementarity | 4.5414e-05 | 6.2031e-03 |
| variables / constraints | 10 756 / 7 826 | **identical** |
| active objective | `p58_rescaled_admm_objective` | identical |
| `admm_objective_scale` | 94 145.72 | **identical** |
| ρ (v / pf / ess) | 1.5 / 300 / 1.0 | **identical** |
| objective value at start | 1 502.4417 | 1 447.4976 |
| start max violation | 7.5905e-05 (`sess_soc_def`) | 7.5853e-05 (`sess_soc_def`) |

Primal start point, by family (`|·|₁` and at-bound counts):

| family | \|·\|₁ cycle 20 | \|·\|₁ cycle 21 | Δ | at bound 20 → 21 |
|---|---|---|---|---|
| `e` | 727.3790 | 727.4007 | +0.0217 | 0 → 0 |
| `vmag_sqr` | 668.9106 | 668.9501 | +0.0395 | 0 → 0 |
| `pg` | 15.5156 | 15.5134 | −0.0022 | 27 → 27 |
| `qg` | 1.8543 | 1.8374 | −0.0169 | 27 → 27 |
| `flex_p_down` | 0.4261 | 0.4037 | −0.0224 | 576 → 576 |
| `flex_p_up` | 0.4293 | 0.4069 | −0.0224 | 416 → 416 |
| `shared_es_pch` / `pdch` | 0.0002 | 0.0002 | ~0 | 1 → 1 |
| `pc`, `qc` | unchanged (SHA identical) | | | 0 → 0 |

Consensus targets moved by comparably small amounts — the largest is `p_pf_req`,
`|·|₁` changing by `−0.0184` against an absmax of `0.7268`.

Multipliers, imported as warm start:

| suffix | n | nonfinite | absmax cycle 20 | absmax cycle 21 |
|---|---|---|---|---|
| `ipopt_zL_in` | 7 492 | 0 | **7 995 263.50** | **7 995 263.50** |
| `ipopt_zU_in` | 7 320 | 0 | **10 208 868.74** | **10 208 868.74** |
| `dual` | 7 826 | 0 | 635 919.12 | 635 216.21 |

**This is the central observation.** The two solves are handed an essentially
identical problem — identical structure, identical objective and scale, identical
ρ, identical starting feasibility to five significant figures, primal start
points differing by `~3e-05` in relative terms, and bound-activity counts
identical in every family. One converges in 115 iterations; the next stalls for
3000 and ends five orders of magnitude worse in dual infeasibility.

Two quantitative asymmetries stand out:

1. **The warm-start bound multipliers are enormous relative to the objective.**
   `zL` reaches `8.0e+06` and `zU` `1.0e+07`, against an objective of `~1.4e+03`
   — a ratio near `1e4`. Their absmax is *bit-identical* between cycles,
   indicating a persistent extreme multiplier rather than one tracking the
   solution.
2. **IPOPT applies its own scaling on top of the RESCALED objective.** Scaled
   dual infeasibility at exit is `5.1598e-02` against unscaled `5.1598e+01` — an
   internal objective scaling factor of `1e-3`. The RESCALED formulation is
   therefore not the last word on what IPOPT actually minimises.

## 9 — Causal ranking

| mechanism | rank | direct evidence | counterevidence |
|---|---|---|---|
| **Multiplier / bound-proximity pathology** | **HIGH** | imported `zL`/`zU` at `8.0e6`/`1.0e7` against an objective of `1.4e3`; absmax bit-identical across cycles; step lengths collapse to `5e-05`; MA97 reports 476 pivot delays, a conditioning indicator | multipliers are finite and complete (0 nonfinite, full counts); at-bound counts unchanged |
| **RESCALED objective / ADMM coefficient imbalance** | **HIGH** | `admm_objective_scale = 9.4146e4` multiplies the consensus terms, so their gradient contribution is ~1e5× the base term's, which is consistent with the multiplier magnitudes above; IPOPT then rescales by a further `1e-3` | the scale is identical at cycle 20, where the same block solved in 115 iterations — so scaling alone is not sufficient to cause the failure |
| **Local active-set transition** | **MEDIUM** | the solve stalls with constraint violation pinned at `6.19e-02` and near-zero steps, consistent with a conflicting active set reached mid-solve | at-bound counts are identical at the *start* of both solves in every family; no evidence of a start-point active-set change |
| **Invalid or corrupted warm start** | **LOW** | — | primal start differs by `~3e-05` relative; no nonfinite values; SHA-identical `pc`/`qc`; structure identical; starting feasibility identical |
| **Shared-ESS / interface state movement** | **LOW** | — | `shared_es_pch`/`pdch` are `~2e-04` with one variable at bound in both cycles; ESS consensus targets move by `~1e-08`; consistent with P5.7's finding that the shared ESS is inert at bootstrap capacity |
| **IPOPT restoration or linear-system failure** | **LOW** | — | exit is `maxIterations`, not `Restoration Failed`; no linear-solver error; MA97 factorized to the final iteration |
| **Concurrency / resource contention** | **LOW** | — | `parallel_execution = False`; one top-level process; failure is deterministic and reproduces P5.12-A's cycle exactly |

The top two are coupled — the multiplier magnitudes are plausibly a *consequence*
of the objective imbalance — which is precisely why the proposed A/B separates
them with a single factor.

## 10 — Proposed frozen A/B (proposal only, not implemented)

**P5.12-C — warm-start multiplier import A/B on the cold RESCALED trajectory.**

| | |
|---|---|
| **Arm A** | cold RESCALED trajectory exactly as P5.12-B, with production's warm-start multiplier import active (`ipopt_zL_in`, `ipopt_zU_in`, `dual` supplied to IPOPT) |
| **Arm B** | identical in every other respect, with the **bound and constraint multiplier import suppressed**; the primal warm start is retained unchanged |
| **The one changed factor** | whether previous-solve multipliers are handed to IPOPT |

**Frozen controls:** base candidate; RESCALED from cold construction;
ρ = 1.5/300/1.0; adaptive ρ disabled; original cold initialization; midpoint
anchor; production IPOPT options, ADMM equations, convergence rules, tolerances
and proximal regularization; sequential execution; one top-level process; cap at
cycle 21; the four carried-state channels plus `candidate_solution` initialized
once as here. Implemented by a process-local wrapper on the solver-creation path;
no production edit.

**Measurements:** for `DSO:case33_3|2025|Spring` at cycles 20 and 21 — IPOPT exit
status, iteration count, scaled and unscaled dual infeasibility, constraint
violation and complementarity, final step lengths; and for the trajectory — which
cycle first shows a local failure, if any.

**Acceptance rule:** Arm B resolves the mechanism if cycle 21 completes with all
36 DSO blocks succeeding *and* the failing block's unscaled dual infeasibility
returns to the `~1e-03` range seen at cycle 20. Partial if the failure moves to a
later cycle rather than disappearing. Refuted if cycle 21 fails identically, which
would redirect attention to the objective-imbalance mechanism.

**Immediate-stop rules:** any change to ρ or adaptive ρ; any production file
modification; a failure before cycle 21 in either arm; a nonfinite value; any
divergence from P5.12-B's accepted cycle markers before cycle 21; any need for a
retry or a second configuration.

This is a proposal. Nothing was implemented, and no second mechanism was tested.

## 11 — Stopping-rule compliance

| rule | status |
|---|---|
| repository or runtime provenance differs | not triggered — all identities verified |
| tracked production file modified | not triggered — none modified |
| accepted evidence missing or altered | not triggered — three hashes match |
| exact capture requires a production change | **not triggered** — production's existing callbacks sufficed |
| replay differs before cycle 21 | not triggered |
| local failure before cycle 21 | not triggered — cycle 21 confirmed as first |
| cycle 21 completes without failure | not triggered — it failed |
| first failure cannot be preserved | not triggered — preserved with full manifest and log |
| nonfinite value | not triggered — none anywhere |
| ρ changed / adaptive ρ enabled | not triggered — ρ bit-identical across cycles |
| retry or numerical change needed | not triggered — none attempted |

Cycle 22 was never started. No polish, no candidate comparison, no second
trajectory, no retry of the failed block, no alternative penalty or tolerance.
The cycle-83 observation was not chased.

## 12 — Confirmation that no production file or setting changed

`shared_resources_planning.py`, `network.py`, `network_data.py`,
`helper_functions.py`, `admm_parameters.py` and `data/SRP1/SRP1_params.json` are
all unchanged; `git status` reports **0 tracked modifications** at start and at
finish. Production `num_max_iters` remains 25 — the cap was set to 21 on a deep
copy only. No IPOPT option, ADMM setting, ρ or adaptive-ρ rule, objective
scaling, proximal regularization, ESSO degradation, Benders or convex model was
touched. Nothing was committed, merged, pulled, fetched, pushed, rebased,
cherry-picked, reset or cleaned. Accepted P5.10, P5.11 and P5.12-A evidence is
byte-identical.

```
P5.12-B-A — deterministic cycle-21 failure reproduced and preserved; one frozen A/B proposed
```

Then stopping for planner review.
