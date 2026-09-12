# Track D — resize the candidate set (PROPOSAL ONLY; nothing run)

**The conditioning problem is in the experiment, not the solver.**

P5.10's candidates differ by a 10% perturbation of roughly 0.01 MVA, giving a
best-to-second signal of **32.87** against a base of ~8.28e8 — **four parts in a hundred
million**. The oracle's resolution is `objective_tolerance = 827,945`. No solver would
resolve that, and no criterion change makes it resolvable at that candidate spacing.

## At the effect size the paper actually claims, today's oracle is adequate

| comparison | claimed effect | signal (absolute) | signal / resolution | passes the built-in SNR gate of 10? |
|---|---|---|---|---|
| P5.10 candidate perturbation | 0.0000040% | 32.87 | **0.00004x** | no, by 2.5e5 |
| incremental storage benefit | 2.01% | 16,642,800 | **20.1x** | **yes** |
| coordination + storage | 18.25% | 151,110,000 | **182.5x** | **yes** |

Both effects the paper claims clear the `minimum_signal_to_noise_ratio = 10.0` gate that
`benders.finite_difference` encodes and has switched off. The resolution finding
disqualifies **fine-grained candidate ranking**, not the paper's headline comparisons.

## Proposed candidate set

Four evaluations, structured as the two comparisons the paper makes:

| # | cell | construction | pairs with | expected signal |
|---|---|---|---|---|
| D1 | no storage, uncoordinated | zero shared-ESS investment; no TSO–DSO coordination | D2, D3 | — |
| D2 | no storage, coordinated | zero shared-ESS investment; ADMM coordination | D1 | 18.25% claim, minus the storage part |
| D3 | proposed storage, uncoordinated | the paper's proposed capacities; no coordination | D1 | — |
| D4 | proposed storage, coordinated | the paper's proposed capacities; ADMM coordination | D3 (storage benefit), D1 (combined) | 2.01% against D3; 18.25% against D1 |

Each pair's signal is estimated from the claimed effect and set against the ~828,000
resolution in the table above. The **proposed capacities must be taken from the manuscript**
rather than invented here; that is an input to this proposal, not an output of it.

## The caveat that must be measured, not assumed

**Larger capacities change the coupling.** Neither of the two properties this project has
established transfers automatically:

- **The 255-solve warm evaluation cost.** It was measured at `e = 0.0213` p.u., where the
  storage is nearly inert — recall that C3, a 15.4% change in the degradation constant,
  moved nothing to sixteen digits precisely because the storage is that small. At the
  paper's capacities the ESSO subproblem becomes active, the consensus has real work to do,
  and the cycle count may change materially in either direction.
- **The 1-in-1,095 local-solve reliability.** The complementarity rows (`pch_hat·pdch_hat ≤
  1e-4`), the capability circle and the SOC limits all become binding at larger capacity;
  the failure rate was measured where they are slack.

**The first evaluation must check both explicitly**, and must be reported as a
feasibility/cost probe before any comparison number is read from it.

## Live dependency

**Penalty classification becomes live again when this track executes.** It blocks any
ranking-baseline re-derivation: `gross_operational_cost` may contain artificial penalty
terms by its own docstring, so a comparison of storage against no-storage is an economic
comparison only once each penalty is classified as an economic cost or as a detector that
must vanish. At the capacities proposed here the complementarity slack (site 1 of P5.13-A)
is far more likely to be active than at 0.0213 p.u., which makes the classification
question **more** pressing in this track, not less.

## What is being proposed for decision

1. Whether to run the four cells at all.
2. If so, the capacities to use, taken from the manuscript.
3. Whether penalty classification is settled first, or the probe runs with the
   classification recorded as an open dependency on every number it produces.

**Nothing is run, and no capacity is chosen, pending that decision.**
