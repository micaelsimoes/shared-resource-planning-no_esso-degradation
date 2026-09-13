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
