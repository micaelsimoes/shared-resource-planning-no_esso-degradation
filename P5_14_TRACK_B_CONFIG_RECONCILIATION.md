# Track B — configuration reconciliation

**Zero solves, enforced (0/0/0/0). Artifacts and git history only. Nothing changed.**
Evidence: `data/SRP1/Results/P514B/b_config_reconciliation.json`.

## 1. Which configuration actually governed each preserved stage

Tolerances are **derived from each stage's own numbers** (`tolerance = residual / ratio`)
rather than read from the case file, because the file's tolerances changed repeatedly over
the period the evidence spans.

| stage | date | `rho_v` | `rho_pf` | `adaptive` | `num_max_iters` | stat. pf | cons. pf | objective tol |
|---|---|---|---|---|---|---|---|---|
| P5.9-B | 2026-09-09 | 1.5 | 1000 | on / off arms | not serialized | 0.01 | 0.01 | 828,029 |
| P5.10-B | 2026-09-09 | 1.5 | 300 | False | not serialized | 0.01 | 0.01 | 827,945 |
| P5.12-R | 2026-09-11 | 1.5 | 300 | False | **21** | — | — | — |
| AB1 control | 2026-09-12 | 1.5 | 300 | False | 50 | 0.01 | 0.01 | 1,313,596 |
| AB1 treatment | 2026-09-12 | 1.5 | 300 | **True** | 50 | 0.01 | 0.01 | 827,536 |
| X22 warm_fixed | 2026-09-12 | 1.5 | 300 | False | 50 | 0.01 | 0.01 | 827,945 |
| X22 warm_adaptive | 2026-09-12 | 1.5 | 300 | **True** | 50 | 0.01 | 0.01 | 827,948 |

Every stage ran at `rho_v = 1.5` and `rho_pf` of 300 or 1000, with stationarity and
consensus tolerances of 0.01. `num_max_iters` is not serialized by the older stages — a
gap worth noting against the eighth rule.

## 2. The mechanism of the divergence — override, not drift

`git log --full-history` over `data/SRP1/SRP1_params.json` returns 29 commits. The file has
carried **`rho.{v,pf,ess} = 1.0` and `adaptive_penalty = true` continuously since
`784346d7`, 2025-12-15** — which *pre-dates every preserved stage*.

**So the file was not edited after those runs.** The divergence is **programmatic
override**: the harnesses call `p59_rho.apply_rho_to_params` and
`p59_rho.set_adaptive_penalty` on the in-memory planning problem, so the case file's rho and
adaptive flag never reached any preserved run.

What *did* change in the file: the **tolerances**, repeatedly — stationarity
`0.05 -> 0.001 -> 0.01`, objective `rel 0.0005 -> 0.005 -> 0.001 -> 0.01 -> 0.001` — and
`num_max_iters 50 -> 25` on 2026-09-04 (`4b4002a7`, "Debug."). Tolerances are **not**
overridden by the harnesses, so each stage ran with whatever the file held at its commit.
The per-stage derived values in §1 are therefore the authority, and they happen to be
uniform at 0.01 because the evidence base post-dates `3839cad1` (2026-08-27).

## 3. What a plain production run would do today

`rho = 1.0` on every family; `adaptive_penalty = True`; `num_max_iters = 25`; stationarity
tolerances 0.01; objective `(abs 1000, rel 0.001)`; TSO proximal **on**, DSO proximal
**off**; C3 active.

**That configuration matches no preserved stage.** Every stage used an overridden rho, and
all but P5.9-B's treatment arm used `adaptive_penalty = False`.

## The decision, with both readings

**Reading 1 — deliberate forward-looking baseline.** P5.9-B showed adaptation works, and
the 2x2 confirmed it: warm-adaptive reached the best warm objective in the fewest cycles,
and the balancing rule finds its own level from any start (300 -> 88.9, 1000 -> 131.7,
adjacent grid points). `adaptive_penalty = True` with `rho.pf = 1.0` is a rational
configuration that lets the rule find its level rather than pinning it at a hand-tuned
value. If this is the intent, it should be **recorded as the new intended baseline with
that rationale**, and the corollary stated plainly: **every preserved result was produced
under a now-superseded configuration**, so the evidence base documents a configuration the
tool no longer runs.

One caveat for this reading, measurable but unmeasured: the rule has only ever been started
from 300 or 1000, both well above its settling range. Starting at 1.0 puts it *below* every
observed operating point, where the *increase* branch would have to fire — and no preserved
run has ever recorded an increase action. All observed adaptation is one-sided.

**Reading 2 — drift.** The tolerance edits are labelled "Debug." and the values moved five
times in six weeks, which is the signature of experimentation rather than decision. On this
reading the file should be restored to the evidence-base values (`rho_v = 1.5`,
`rho_pf = 300`, `adaptive_penalty = False`) so that a plain run reproduces the evidence.

**Not changed pending your decision.** Both readings are consistent with the artifacts; the
history distinguishes intent for the tolerances (repeated, labelled "Debug") less clearly
than for rho and the adaptive flag (untouched since 2025-12-15, i.e. never part of the
experimental sequence at all).
