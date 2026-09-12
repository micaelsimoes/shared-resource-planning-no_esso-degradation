# S2 — cost model and lever designs (design stage; nothing run)

**Zero solves, enforced: `SolveProfileGuard` in blocking mode observed 0 solves and 0
process launches. No parameter, tolerance, criterion or rho was changed. No lever was
run.**

Artifacts: `data/SRP1/Results/P514S2/s2_cost_model.json`,
`frozen_ab1_adaptive_cold_v1_5d996737.json`,
`frozen_ab2_dso_proximal_v1_76e0051d.json`.

## 0. Correction that must come first — one lever HAS been evaluated

The S2 brief states that neither disabled lever has an evaluation artifact. **That is
false for residual balancing**, and my own S1 report invited the error: I wrote that no
evaluation artifact exists *for any branch*, which is true of the branches and not of the
stages. I should have searched the stage artifacts too.

`p59_b_adaptive.py` and `data/SRP1/Results/P59/p59_b_adaptive.json` are a two-arm A/B of
exactly this lever, with a control. Worse — or better — its docstring already states the
mechanism S1 "found":

> "Production measures the dual residual as `dual = rho * |z_current - z_prev| / base`
> (lines 4897, 4938), i.e. LINEAR IN RHO, and the update rule compares `primal_ratio`
> against `dual_ratio`, increasing rho when `primal_ratio > 5 * dual_ratio` and
> decreasing it when `dual_ratio > 5 * primal_ratio` (3 for pf). Raising rho therefore
> raises the measured dual residual proportionally and pushes the rule toward DECREASING
> it again. Whether that negative feedback actually cancels a deliberately raised penalty
> is measured below, not asserted."

So S1's mechanism analysis is a **rediscovery** of something already written down, and
the question the brief poses — self-correcting or oscillating — was already answered.

### What P5.9-B measured

Same template, same starting `rho_pf = 1000`, first generation:

| arm | cycles | runtime | final `rho_pf` | objective |
|---|---|---|---|---|
| adaptive **off** | **16** | 783.6 s | 1000 (fixed) | 827,953,362.7 |
| adaptive **on** | **6** | 295.4 s | **131.687** | 827,844,231.9 |

A **2.67x reduction in cycles**, a better objective by 109,131 (of 8.28e8, i.e. 0.013%),
and the coupling proves **self-correcting, not oscillatory**: the rule drives `rho_pf`
down by a factor of 7.6 and holds it there across three further generations.

The 16 cycles of the adaptive-off arm independently reproduces P5.10-B's 16 cycles at
`rho_pf = 1000` — two stages agreeing on the same number.

**Proximal regularization is the lever with genuinely no evaluation.** Searched:
the only matches in `data/SRP1/Results` are provenance dumps and a docstring recording
that it was left unchanged. For that lever the brief's claim stands.

## 1. Cost model

Solves per cycle is **51** — 36 DSO + 12 TSO + 3 ESSO — verified from the P5.12-R
ledger: 1095 attempted solves = 780 DSO + 252 TSO (12 blocks x 21) + 63 ESSO (3 x 21).

| Path | cycles | ADMM local solves | initialization | `Q(x)` depends on |
|---|---|---|---|---|
| **cold, uncapped** | **>= 21** | **>= 1071** | none | `x` alone |
| warm from template, gen 1, `rho_pf` 300 | 6 | 306 | frozen T0, history-neutralised | `x` **and the template** |
| warm from template, gen 1, `rho_pf` 1000 | 16 | 816 | frozen T0, history-neutralised | `x` and the template |
| warm from template, gen 1, adaptive from 1000 | 6 | 306 | frozen T0, history-neutralised | `x` and the template |
| warm **continuation**, gens 2+ | 1 | 51 | inherits the previous generation | `x` and **the whole campaign history** |

The polish phase is **not separately counted** in the preserved ledgers. Inferred as one
pass over the 48 network blocks (12 TSO + 36 DSO) per generation from `blocks_rescaled =
48` and the `failed_blocks` accounting; excluded from the counts above, and it would add
~48 solves per warm generation. Flagged as an inference, not a measurement.

### Withdrawn figures, including a correction to the correction

- **~69 further cycles / ~4,600 solves per evaluation — withdrawn.** The cold trajectory
  carried `cap: 21` in its own configuration (`data/SRP1/Results/P512R/run.json`). It was
  stopped by configuration, not by stalling. The descent arithmetic stands; the
  extrapolation does not.
- **The 15x cold-to-warm ratio — also withdrawn**, because it inherited the 4,600. The
  ratio that the evidence actually supports is **>= 3.5x** (>= 1071 against 306), and the
  true cold cost is *unknown*, bounded below only. Replacing one unsupported number with
  another would repeat the original error.

### The headline the model supports

Cold at `rho_pf` 300, capped at 21 cycles, still descending. Warm-from-template at
`rho_pf` 300, converged in 6. Same rho, same tolerances. **Initialization dominates rho
as a cost lever, and the P5.10 sweep varied the weaker of the two.**

## 2. The tension the outer layer will have to resolve

The cheap path is cheap *because* it inherits a template, and inheritance is exactly what
makes `Q(x)` depend on history rather than on `x` alone. `history_neutralised` exists to
manage that, and it is the same concern the Expert raised about P5.10's inherited-template
comparison.

So the choice is: **a cheap oracle whose value depends partly on where it started, or an
independent oracle at several times the cost.** Any outer method — Benders or otherwise —
needs the second for its values to mean anything; any campaign needs the first to be
affordable. The warm-continuation row makes the tension concrete: at 1 cycle and 51
solves, generations 2+ are 20x cheaper than a cold evaluation's *lower bound*, and their
`Q` depends on the entire preceding campaign.

This is a formulation problem, not a tuning one, and it is recorded here as the question
the outer-layer discussion must answer rather than as anything S2 can settle.

## 3. Frozen lever designs — costed, not run

**AB1 — adaptive rho on the COLD path** (`5d996737`). The warm path is already answered
by P5.9-B, so re-running it would buy nothing; the untested case is the cold path, where
the cost problem lives. One varied setting: `adaptive_penalty` False vs True, cold, same
`rho_pf = 300` start, identical caps declared in advance.

*Analytically derived prediction, in the `cl_nom/k` manner:* the update rule has negative
feedback in rho with a fixed point inside its dead band, and P5.9-B located that point at
`rho_pf = 131.687` starting from 1000. **Attractor hypothesis: the fixed point is a
property of the problem and the dead band, not of the starting value, so starting from
300 the adaptive arm should settle in `rho_pf` ∈ [80, 200].** Falsified by settling
outside that band, or by oscillation between the increase and decrease branches — the
specific failure the coupling makes plausible.

Cost: **>= 1071 local solves per arm, >= 2142 for both**, and the cold cost is a lower
bound, not an estimate. The most expensive item on the agenda.

**AB2 — DSO proximal regularization** (`76e0051d`). `proximal_regularization.tso.enabled
= true` with gamma 1.0, `dso.enabled = false`: a standard nonconvex-ADMM convergence aid
applied to one side of the interface and not the other, never measured. One varied
setting on the warm path at `rho_pf` 300, where the control's cost is known exactly.

*Prediction:* a proximal term damps an agent's own iterate movement, so DSO block
movement should fall. **No directional prediction is made for the cycle count** — damping
helps if the DSO iterates oscillate and hurts if they do not, and the mechanism does not
imply which. Either outcome is informative: movement down with no cycle penalty argues
the asymmetry is an oversight; movement down with a cycle penalty argues it was a
trade-off nobody wrote down.

Cost: **~306 solves per arm, ~612 for both**, roughly a fifth of AB1's lower bound.
**AB2 should run first if both are authorized.** Its control must reproduce P5.10-B's
6-cycle result before any treatment number is read.

## 4. Prior art in the two unmerged branches

`residual_balancing_mod` (42 commits, June 2024) — **misnamed**. Its diff *removes* the
blocks labelled `# Augmented Lagrangian -- Interface power flow (residual balancing)` and
replaces the interface normalization with an average-interface-power form, the ancestor
of today's `/interface_rating`. 41 files, mostly IEEE9/ieee18_3 case data. Every commit
message is `Update` or `Update.`

`warm_start_tests` (5 commits, May 2024) — small: 40 changed lines in
`shared_resources_planning.py` plus a case-file edit, setting `from_warm_start = True`.
Commit messages: `Blegh`, `Debug`, `Correction`, `Small correction`, `Test.` Warm start
is in production today, so whatever this branch established arrived by another route.

**Neither branch has an evaluation artifact, so what they showed is unrecoverable.** That
statement is now correctly scoped to the branches — the stage artifacts are a different
matter, as §0 shows.

## What S2 establishes

The cost model is pinned and its weakest number is labelled as a lower bound rather than
an estimate. One of the two levers turns out to be measured already, with a 2.67x cycle
reduction and a self-correcting rho. The other is genuinely untested and is the cheaper
experiment. The initialization/independence tension is stated as the formulation question
it is.

**No lever was run. S2 stops here.**
