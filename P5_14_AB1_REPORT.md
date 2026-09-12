# AB1 — adaptive rho on the cold path — FINAL REPORT

**Both arms complete. The headline result is not the one the design expected, and it
inverts on inspection: the treatment's convergence declaration is largely an artifact of
the criterion, while its objective descent is real and large.**

Frozen design `data/SRP1/Results/P514S2/frozen_ab1_adaptive_cold_v2_6e23cac0.json`,
committed (`eaaa196a`) before either arm ran. Results:
`data/SRP1/Results/P514AB1/ab1_{treatment,control}.json`. Guards armed throughout:
**0 blocked solves in either arm.**

## 1. Primary outcome

| | treatment (adaptive on) | control (adaptive off) |
|---|---|---|
| cycles to convergence | **32** | **none — hit the cap at 50** |
| status | converged | **INCONCLUSIVE by predeclaration** |
| local solves | 1683 | 2601 |
| wall clock | 1293 s | 2119 s |
| final `rho_pf` | 3.4683 | 300 (fixed) |
| final recourse | 826,829,641 | 1,310,469,508 (still descending) |
| local solve failures | 1 (cycle 11) | 0 |

**The cold-path cost without adaptation remains unknown.** The control was capped, so no
cost number may be extrapolated from it — that was predeclared, and it is the second time
this trajectory has been stopped by a cap rather than by convergence. What is now known
is that 50 cycles is *not enough*, which is itself new: the previous bound was 21.

## 2. The secondary prediction is falsified

Predicted `rho_pf` settles in **[80, 200]**, most likely 133.33. Observed **3.4683** —
eleven consecutive decreases, zero increases, held for the last 20 cycles. The walk
passed *through* 133.33 at cycle 2 and kept descending.

No oscillation and no failure to settle, so the attractor *idea* survives; what fails is
the claim that the fixed point is initialization-independent. The same rule with the same
dead band settles at **131.687 warm** and **3.4683 cold** — a factor of 38. The adaptive
rule's operating point is a property of the trajectory it is placed on, which
independently reinforces the S2 finding that initialization dominates.

## 3. The result inverts: the two arms were not held to the same standard

The dual residual is `rho * |Δz| / base`, so a criterion stated on it moves with rho.
Dividing rho out recovers the underlying iterate step, which is the physical quantity:

| arm | cycle | `dual_pf_mean` | `rho_pf` | **`|Δz|/base`** | dual ratio | verdict |
|---|---|---|---|---|---|---|
| treatment | 32 | 2.108e-03 | 3.4683 | **6.079e-04** | 0.211 | "converged" |
| control | 32 | 9.977e-02 | 300 | **3.326e-04** | 9.977 | not converged |
| control | 50 | 8.722e-02 | 300 | **2.907e-04** | 8.722 | not converged |

**At matched cycle 32 the control's iterates were moving 1.83x LESS than the
treatment's, and at cycle 50 2.09x less than the treatment's at its declared
convergence — yet the control fails the test and the treatment passes.** The standards
differ by exactly `300 / 3.4683 = 86.50`.

On a common standard, neither arm converged. To pass at `rho_pf = 300` the step must be
below `3.33e-05`; the treatment sits at **18.2x** that threshold and the control at
**8.7x**. Measured identically, **the control is closer to convergence than the
treatment ever got.**

So "adaptation converged the cold path in 32 cycles" cannot be read as evidence that
adaptation converges anything. It is substantially the S1 coupling operating at its
extreme: the rule relaxed rho by 86x, and the criterion relaxed with it.

## 4. What IS real: the descent

The objective difference is not a criterion artifact.

| cycle | control recourse | treatment recourse |
|---|---|---|
| 1 | 2,355,181,727 | 2,355,181,727 |
| 20 | 1,424,799,221 | 855,180,304 |
| 32 | 1,372,687,800 | **826,829,641** |
| 50 | 1,310,469,508 | — (converged at 32) |

At matched cycle 32 the treatment is **40% lower**. The control's own descent is slowing:
decrements over the last five cycles are 3.273, 3.206, 3.183, 3.152, 3.127 M, a ratio of
0.890 over ten cycles, against a remaining gap to the treatment's value of 484 M.

**No extrapolation is offered from those numbers**, per the frozen design — that is the
error this stage exists to avoid, and the decelerating ratio makes extrapolation unsafe
in both directions. The observation is restricted to what was measured: at every matched
cycle, the low-rho trajectory is far lower in objective.

The mechanism is the ordinary rho trade-off. At `rho_pf = 300` the consensus penalty
dominates each agent's local objective, holding the iterates near consensus
(`primal_pf = 2.15e-04`, 21x tighter than the treatment's `4.53e-03`) but descending
slowly. At `rho_pf = 3.47` the agents optimize economically, consensus is looser but
still inside the declared tolerance, and the objective falls much faster.

## 5. What this says about the convergence machinery

Two findings follow that are larger than the A/B question.

**The stationarity criterion cannot compare runs at different rho.** It is not merely
scale-sensitive, as S1 established; it is *incommensurable* across rho settings. Any
comparison of convergence between an adaptive run and a fixed-rho run — including the
headline of this stage — is comparing against two different tests. Reporting "cycles to
convergence" across arms with different rho is therefore not a valid cost comparison
unless the standard is restated on a rho-free quantity.

**The declared consensus tolerance admits a wide objective band.** Both arms sit inside
`primal_pf ≤ 0.01`, and differ by 40% in recourse at matched cycles. The treatment's
declared convergence is a point with consensus 21x looser than the control's. Whether
`Q(x)` is well defined at this tolerance is now a live question, and it bears directly on
the outer layer: a cut or a ranking built on `Q` inherits that band.

## 6. Caveats

**Both arms run under C3**, so no objective here is comparable with any pre-C3 figure
(P5.9-B's 827.84 M, P5.12-R's trajectory). The A/B is internally valid because both arms
share the formulation.

**One local solve failed** in the treatment at cycle 11, during the rho collapse;
production held the penalty update for that cycle and the run recovered. Control had
none. 1 failure in 4284 solves across the stage.

**Solve-profile identity.** Treatment 1683 observed against `51 x 32 = 1632`; control
2601 against `51 x 50 = 2550`. Both differ by exactly 51 — one initialization block —
matching the P5.12-R ledger structure. Reported, not absorbed. Guards: 0 blocked in both.

## 7. Verdict against the predeclared outcomes

- `adaptation_helps` — **not established**. The cycle comparison it rests on is invalid,
  because the two arms were held to standards differing by 86.5x.
- `adaptation_harms` — not established either.
- `INCONCLUSIVE` — **this is the outcome for the cost question.** The control hit the cap;
  the cold-path cost without adaptation is still unknown.
- The attractor prediction is **falsified**, and informatively so.
- Unpredeclared but established: adaptation descends the objective far faster, and the
  convergence test is incommensurable across rho.

**AB1 does not license a cost figure for an independent oracle.** What it licenses is a
sharper question for S2: on a rho-free standard, what does the cold path cost — and is
the present consensus tolerance tight enough for `Q(x)` to mean anything?
