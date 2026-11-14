# P5.15 — Planner review of Steps 0 and 2

**Status: Steps 0 and 2 complete. Step 1 NOT started, and will not start until the author
has reviewed this.** Prepared for the author and the external expert.

Authority: `PLANNER_BRIEF_2026-09-13.md`, which supersedes the "NUMERICAL PROGRAMME CLOSED"
verdict in `REVISION_CONTEXT.md` and `COWORK_HANDOFF.md`.

| step | executed by | deliverable | commit |
|---|---|---|---|
| 0 — ESSO/DSO warm-start A/B | **Worker** (available; it was not from P5.12-G onward) | `P5_15_0_REPORT.md`, `data/SRP1/Results/P5150/` | `3bcb899e` |
| 2 — SMOPF conditioning audit | **Advisor** (read-only, in parallel) | `P5_15_2_SMOPF_CONDITIONING_AUDIT.md` | `5324f832` |

---

## 1. What the Planner verified independently

The Worker's and Advisor's reports are subordinate evidence, not conclusions. Before
accepting either I re-derived the load-bearing claims from primary sources:

| claim | how it was checked | result |
|---|---|---|
| no production file was modified in Step 0 | `git status --porcelain` | confirmed — only the new harness, report and results dir appeared |
| the arm results | recomputed terminations, iteration counts, objectives and relative gaps from `p5150_report.json` and the per-arm IPOPT logs, not from the report's prose | confirmed, all figures below |
| `option_overrides` is clobbered | read `shared_energy_storage_data.py:881` against the `from_warm_start` block | **confirmed** |
| node 9 is a failing instance | `grep "^EXIT" optim_log_node_9.txt | tail -1` | **confirmed** — `Maximum Number of Iterations Exceeded` |
| the fixture the DSO arm actually used | `p5150_report.json → dso.preflight.target` | **`DSO:case33_3|2025|Spring`** — see §4 |
| no other artifact asserts "node 9 converged" | inspected `n1_k10000.json` | confirmed — it records a cycle-level failure count, not per-node status |

---

## 2. Step 0 — H-WS is supported on both solver families

**Both reproduction gates passed**: under current production policy, ESSO node 7 and the
cycle-21 DSO fixture each hit `maxIterations` at 3000 iterations.

**ESSO, node 7** (objective compared against cold-start A4):

| arm | policy | termination | iters | wall | rel. to A4 |
|---|---|---|---|---|---|
| A0 | production (all five pushes 1e-9) | maxIterations | 3000 | 3.70 s | — |
| **A1** | pushes 1e-3, fracs at IPOPT defaults | optimal | **30** | 0.14 s | **8.4e-11** |
| **A2** | A1, primal-only (no multiplier import) | optimal | **37** | 0.15 s | **2.9e-13** |
| A3 | A1 + `mu_strategy=adaptive` | optimal | 25 | 0.14 s | **6.7e-2 — different local optimum** |
| A4 | cold | optimal | 59 | 0.19 s | reference |
| A5 | A0 + `max_iter=500` | maxIterations | 500 | 0.69 s | — |

**DSO, `case33_3` 2025 Spring cycle 21**: A0 maxIterations 3000 / 16.54 s; **A1 85 iters at
3.6e-9**; A2 89 iters; A4 cold 77 iters; **A3 37 iters but 1.03e-5 — misses the 1e-6 bar**.

**Controls**: node 5 A0 optimal in **0** iterations (a true control). Node 9 A0
`maxIterations` — see §3.

**Verdict.** H-WS **SUPPORTED for the ESSO**, on two independent failing instances, and
**SUPPORTED for the DSO**, on the one preserved failing fixture. The success criterion
(< 200 iterations, objective within 1e-6 relative of cold, violation < 1e-6) is met by **A1
and A2 on both families**; A3 fails it on both, for different reasons.

**A3 is not a refutation of H-WS**, but it is a finding in its own right: on the ESSO it
converges to a materially different local optimum, which is direct evidence that this NLP has
multiple local optima at this operating point.

---

## 3. Two findings beyond the brief

**(a) `_create_solver` silently clobbers `option_overrides`.** Overrides are applied at
`shared_energy_storage_data.py:881`, and the `from_warm_start` branch then hard-assigns all
five `warm_start_*` keys to 1e-9 unconditionally. **No caller can influence the ESSO
warm-start policy** — including `recovery_options`, which matters directly for the brief's
recommendation (b) to extend the recovery path. The Worker had to post-adjust `solver.options`
after calling production `_create_solver`, and documented that it did so.

**(b) Node 9 is a second genuine failing instance, not a control.** The brief states "nodes 5
and 9 as controls — they converged". The last `EXIT` in `optim_log_node_9.txt` is
`Maximum Number of Iterations Exceeded`, and `node9_A0` reproduces it. Only node 5 is a
control. **This strengthens the ESSO result**: H-WS now rests on two independent failing
instances rather than one. No repository artifact asserted the wrong status; the brief did.

---

## 4. A cross-step finding neither subordinate report could make alone

The Step 2 audit opens by correcting the brief: it names "`case33_2`, node 7, the fixture that
failed at cycle 21", but the preserved record shows **`case33_2` node 7 is the *converged*
comparator** (P5.12-W/X, `Optimal Solution Found`, 91 iterations) while **the cycle-21 failure
is `case33_3` node 9** (P5.12-A/B/ArmA/P/R/T).

**The Planner checked whether that mislabel contaminated Step 0.** It did not:
`p5150_report.json → dso.preflight` records the target as **`DSO:case33_3|2025|Spring`** —
the correct failing fixture, reached through the P5.12-P harness's own loading path. The
Worker used the right instance despite the brief naming the wrong one.

**Action required:** correct the fixture identification in `PLANNER_BRIEF_2026-09-13.md`
before it is used again. P5.12-P/ArmA/R/T tooling is all wired to `case33_3`.

---

## 5. Step 2 — audit summary

Read-only, zero solves. Full inventory of every network constraint family with its
nonlinearity, scaling, penalty coefficient and active-set evidence, plus the complete IPOPT
option inventory. Five ranked reformulation candidates, no implementation.

Two carry dependencies the author must resolve before they can even be evaluated:

- **Candidate 2** (capacity `Var`s → `Param`s; turns an indefinite quadratic into a convex
  SOC) **removes the Benders capacity-sensitivity channel**. It is viable only if the outer
  method drops local cuts — which is Step 4's open decision.
- **Candidate 3** (separate RES converter capability from stochastic availability) is blocked
  **solely by an author data-semantics decision**: whether the `Pmax`/`Qmax` already in the
  network JSON denote an inverter nameplate. The audit corrects the earlier record, which said
  the rating data was missing.

Three audit findings bear on the current work rather than on any candidate:

1. **`_is_recoverable_network_failure` fires only on `internalSolverError`** — so the cycle-21
   `maxIterations` exit received no retry at all. Independent corroboration of the brief's
   recommendation (b).
2. **`warm_start_mult_bound_push` is derived from `bound_push`** on the network path
   (`network.py:534`), so any A/B varying `bound_push` moves two mechanisms at once.
3. **The `dual` suffix is `IMPORT_EXPORT` and is never cleared on a nominally cold solve.** If
   Pyomo's NL writer exports it, **Step 0's arm A4 was not cold on the constraint-multiplier
   side**. This is a hypothesis about writer behaviour, answerable from a preserved `.nl` with
   no solve.

---

## 6. What is NOT established

- **No production change is justified yet by Step 0 alone.** The DSO half rests on **one**
  preserved failing fixture. P5.12-P's own caution stands: two data points on one captured
  failure would be tuning to that instance. Step 0 adds three more configurations on that same
  fixture — it strengthens the path-dependence case, it does not broaden the sample.
- **`max_iter=500` was tested only on the ESSO**, not on the network family.
- **A3 was not evaluated as a recovery-only option**, which is where it might belong.
- **Arm A4's coldness is qualified** pending the `dual`-suffix question above.
- Step 0 says nothing about whether the reformulated ESSO of Step 1 will behave better; that
  is what Step 1's gate exists to measure.

---

## 7. Decisions requested

1. **Schedule the `option_overrides` clobbering fix separately from Step 1?** It is independent
   of the warm-start *value* decision and it currently disables the recovery path's ability to
   change policy.
2. **Run a one-arm `max_iter=500` check on the network family before Step 1?**
3. **Resolve the `dual`-suffix question before Step 1**, since it qualifies A4's status? No
   solve required.
4. **Correct the fixture identification in the brief** (§4).

---

## 8. Evidence inventory

| artifact | state |
|---|---|
| `P5_15_0_REPORT.md`, `p5150_report.json`, `p515_0_esso_warmstart_ab.py` | committed, `3bcb899e` |
| `P5_15_2_SMOPF_CONDITIONING_AUDIT.md` | committed, `5324f832` |
| 15 per-arm IPOPT logs, 11.2 MB | **left untracked; hash-recorded** in `data/SRP1/Results/P5150/log_hash_inventory.json` per the CLAUDE.md artifact rule |
| `PLANNER_BRIEF_2026-09-13.md`, `EXPERT_REVIEW_2_ACTION_PLAN.md` | untracked (author-supplied) |

`REVISION_CONTEXT.md` has **not** been rewritten. Per the brief that happens at the end of
Step 1, and it must then also withdraw the `C*` feasibility-boundary claim, which the action
plan §1(a) argues is a restoration-phase artifact rather than a physical boundary.
