# P5.14-N — the ageing model at material capacity

**Control arm: determinism gate PASSES exactly. Perturbation arm: hit the 90-cycle cap
with 74 failure-cycles. The objective comparison is INCONCLUSIVE by predeclaration — and
the ageing model is emphatically NOT inert.**

Frozen spec `data/SRP1/Results/P514N/frozen_n1_control_arm_v1_f1bbddf4.json`, frozen before
either arm ran. Rule eleven asserted before execution in both arms.

## 1. Control arm — the gate passes, and the instrumentation is proved non-perturbing

| check | reference | observed |
|---|---|---|
| recourse | 816,121,464.1554238 | **816,121,464.1554238, delta = 0.0** |
| cycles | 67 | 67 |
| solves | 3,468 | 3,468 |
| rule ten | 0.9329 | 0.9329 |
| local-solve failures | 0 | 0 |

Exact reproduction confirms the campaign's only material-capacity number **and** proves the
added capture is non-perturbing, so no separate neutrality run was needed — the gate did
both jobs as designed. ESSO models serialized (5.3 MB) so the next stage cannot hit this
wall again.

## 2. The EFC answer — cycling is substantial

| node | (0,0) | (0,1) | (0,2) | max | margin to 1.4612 |
|---|---|---|---|---|---|
| 5 | 1.1124 | 0.8916 | 0.9703 | **1.1124** | 0.3488 (76.1% of threshold) |
| 7 | 1.1122 | 0.9059 | 0.9711 | 1.1122 | 0.3490 |
| 9 | 1.1124 | 0.9124 | 0.9695 | 1.1124 | 0.3488 |

**1.11 EFC/day peak, ~0.99 average — above the 0.76 interpretability guide and below the
1.4612 floor-binding threshold.** Cumulative SoH falls `1.0 -> 0.8387 -> 0.7284 -> 0.6248`:
a **37.5% capacity loss** over the horizon. The ageing model was already visibly doing work
before the perturbation.

## 3. Perturbation arm — the mechanism responds exactly, and the solve destabilizes

`cl_eff` set to 10,000 (from 11,541.56): the same 15.4% change characterised at bootstrap
capacity, where it moved nothing to sixteen digits.

**The mechanism responded precisely as designed:**

| quantity (node 5) | control `k=11541.56` | perturbed `k=10000` | ratio |
|---|---|---|---|
| degradation/day, cohort (0,0) | 9.64e-05 | 1.112e-04 | **1.1535** (expected 1.1542) |
| terminal SoH | 0.6248 | **0.5867** | −6.10% |
| EFC/day max | 1.1124 | 1.1118 | ~unchanged |

**But the run did not converge:**

| | control | perturbed |
|---|---|---|
| cycles | 67 (converged) | **90 — hit the cap** |
| local-solve failures | **0** | **74 of 90 cycles** |
| solves | 3,468 | 4,641 (`51x90+51`, identity exact) |
| recourse | 816,121,464.2 | **none — no value may be read from a capped cell** |

## 4. Verdict against the predeclared branches

The predeclared test was: *moves by more than the 164,149 bar -> the ageing model is live;
moves by less or not at all -> inert.* **Neither branch applies, because the perturbed arm
produced no value to compare.** By the standing rule, a capped cell yields no number and
must not be extrapolated from. **The objective comparison is INCONCLUSIVE.**

But the question behind the branches is answered, and answered the other way from the one
we were braced for:

> **The ageing model is not inert at material capacity — it is live and consequential
> enough that a 15.4% change in its constant takes the problem from converging in 67 cycles
> with zero local-solve failures to not converging in 90 cycles with 74 failure-cycles.**

The direction is physically coherent: smaller `k` means faster degradation
(`delta = throughput / (2 k E)`), so available capacity `es_e_available = es_e_rated x
soh_cumul` shrinks faster, tightening the feasible set through the horizon. The SoH floor
itself is *not* the cause — terminal SoH is 0.5867, still above the 0.50 floor, with margin
0.0867 against the control's 0.1248.

## 5. What this does and does not license

- **Does**: retire the "degradation modelling contributes nothing" branch *at material
  capacity*. At bootstrap capacity it contributed nothing measurable; at C\* it dominates
  the solve's behaviour.
- **Does not**: attribute the 0.37% storage effect to degradation value. That needed the
  objective comparison, which is inconclusive.
- **Raises a robustness concern that did not exist before**: the C\* baseline sits close to
  a regime where a 15.4% parameter change causes local-solve failure in 82% of cycles. The
  converged baseline is real, but its neighbourhood is not benign.

## 6. Two harness defects, recorded

**(a) The determinism gate was applied to an arm it does not apply to.** The perturbed
artifact carries `determinism_gate.verdict = "FAIL — NON-DETERMINISM AT MATERIAL
CAPACITY..."`, which is **spurious**: the gate compares against the control's reference,
and the perturbation is supposed to differ — it did not even converge, so its recourse is
`None` and the comparison is undefined. The harness computes the gate unconditionally
instead of only when `k_override is None`. Corrected in
`data/SRP1/Results/P514N/n1_k10000_gate_correction.json` rather than by editing the result
artifact. **The real determinism result is the control arm's PASS.**

**(b) A second capture gap.** Neither report stores the per-cycle rows, so the cycle at
which the perturbed arm's failures began cannot be recovered. Rule eleven covered the ESSO
quantities the frozen spec named; it did not cover the per-cycle trajectory, which this
stage turned out to need. The rule works as written — the gap is in what the spec required,
not in the assertion.

---

# Addendum — the failure mode, and what it decides

## The termination conditions

| source | `Optimal` | `Acceptable` | **`Max Iterations`** | `Locally infeasible` |
|---|---|---|---|---|
| network solves (4,368 in the perturbed arm) | 4,348 | 19 | **1** | 0 |
| ESSO node 5 (last 120) | 120 | — | **0** | 0 |
| **ESSO node 7 (last 120)** | 47 | — | **73** | 0 |
| ESSO node 9 (last 120) | 112 | — | **8** | 0 |

**The failures are `Maximum Number of Iterations Exceeded`, in the ESSO subproblems, and
overwhelmingly at node 7.** No solve reported local infeasibility. `max_iter` is unset, so
each failure burned IPOPT's default 3,000 iterations — which is also why the arm took
3,348 s against the control's 2,342 s.

## Which reading that selects

By the predeclared dichotomy this is **numerical fragility, not genuine tightness**:

- **Genuine tightness** would have produced `locally infeasible`. And we know this solver
  reports that when it concludes it: **the same node 7, at 1.00 MVA in the ladder, returned
  exactly `Converged to a point of local infeasibility`.** Same solver, same subproblem
  family, same machine — it says "infeasible" when it finds infeasibility, and it did not
  say so here.
- **Numerical fragility** predicts `maxIterations`, which is what occurred.

**The honest qualification:** `maxIterations` proves non-convergence within 3,000
iterations, not that the feasible set is untightened. A tighter set can manifest as slow
convergence. What it does establish is that the perturbed problem **did not look infeasible
to IPOPT**, which is the distinction the two readings turn on.

So the sentence for the paper is the fragility one: at material capacity, a 6.1% reduction
in terminal available energy (2.421 -> 2.274 MWh) leaves the solver unable to navigate the
problem in 82% of cycles — not because the plan becomes physically infeasible, but because
the subproblem becomes numerically intractable within its iteration budget.

## The mechanism validation is a positive result and should be read as one

`degradation/day` moved `9.64e-05 -> 1.112e-04`, a ratio of **1.1535** against the law's
`11541.56 / 10000 = 1.15416` — a **0.06% match**.

This is the **first direct validation that the degradation implementation does what the law
says**, on the component we had four independent reasons to suspect was inert. The ageing
model is live and consequential at material capacity. **That branch is closed, and closed
favourably.**

## P5.12 in retrospect

P5.12 spent weeks on **one local NLP failure in 1,095 solves** at bootstrap capacity,
treating it as a rare anomaly whose mechanism had to be found.

It was an early symptom of a fragility that becomes dominant at realistic scale. **One
parameter-step from the material-capacity baseline, that failure mode is the normal
behaviour** — 74 of 90 cycles. This makes the P5.12 effort look better rather than worse:
it was chasing something real, at the only magnitude where it was then visible.

## The campaign's question, answered negatively

| component | status |
|---|---|
| resolution | **fine** — 28x headroom at `rel` 1e-4 |
| reliability at a point | **fine** — zero failures in 3,468 solves at C\* |
| **reliability across the design space** | **NOT ESTABLISHED, and the one data point is catastrophic** |

A planning campaign varies the **candidate**, which is a far larger perturbation than 15.4%
of a fixed model constant. If a constant-step of that size produces 82% cycle failure, **a
campaign cannot be expected to hold together.** That is the finding the whole of Track D was
for.

**And it puts C\*'s own numbers in question.** The 816,121,464 baseline sits one small
parameter-step from 82% failure and stopped at 93% of its threshold. **Wherever the 0.37%
storage effect is reported, this must be reported with it:** the number is real, and the
ground it stands on is not stable.
