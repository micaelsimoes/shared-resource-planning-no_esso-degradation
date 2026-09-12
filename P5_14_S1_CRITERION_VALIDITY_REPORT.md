# S1 — criterion validity for `stationarity_pf`

**Diagnose only. Zero solves, enforced: `SolveProfileGuard` in blocking mode observed
0 solves and 0 process launches. No tolerance, criterion, rho or parameter was changed.**

Frozen spec: `data/SRP1/Results/P514S1/frozen_s1_criterion_spec_v1_6c1d0a81.json`
(SHA-256 `6c1d0a81a46e82124811efece74e918be2ae1c033550cfdcd6b72a41f93d3194`), frozen
before the analysis. Results: `data/SRP1/Results/P514S1/s1_criterion_scaling.json`,
survey: `s1_prior_art_survey.json`.

## Headline

**The primary hypothesis is only partially confirmed, and its strong form is
contradicted.** The rho/tolerance coupling is real by construction, but the criterion is
**not unsatisfiable**: it is satisfied at termination in all nine preserved runs. What
raising `rho_pf` costs is *cycles*, not satisfiability.

## 0. A harness defect, caught by the predeclared cross-check

The first run re-derived **9** stationarity slacks against the 93 preserved — because I
read each artifact's top-level `rows`, which are the three terminal summaries per file,
not the per-cycle data. The per-cycle records live in each row's `cycle_detail`:
3 repeats x 6, 9 and 16 cycles = **18 + 27 + 48 = 93**, matching
`p510_e_criteria.json` exactly. Corrected, the cross-check matches **93 of 93 to 1e-12**.

The cross-check was in the frozen spec precisely to catch this, and it did. The
pre-correction numbers are void and are not reported.

## 1. The mechanism, as predeclared

`shared_resources_planning.py:4938` computes `rho * |z^k − z^{k−1}| / interface_rating`;
`:2454` divides it by an **absolute** tolerance; `p510_e_criteria.py:83` inverts that
into slack. Direction established from the code before predicting: slack is
*tolerance over residual*, so a higher `rho_pf` must **lower** it.

**P1 — increment invariance.** `g = dual_pf_mean / rho_pf` (the increment with rho
divided out) is **not** invariant:

| comparison | full trajectory | matched cycles 1–6 |
|---|---|---|
| `g(500)/g(300)` | 0.6467 | 0.7596 |
| `g(1000)/g(300)` | 0.3379 | 0.4981 |

The iterate increments *shrink* as rho rises — nearly in proportion over full
trajectories (0.338 against the 0.300 of exact 1/rho), about half in proportion at
matched early cycles.

**P2 — slack scaling.** Predicted 0.300 if increments were unchanged:

| measure | observed | predeclared classification |
|---|---|---|
| full trajectory | **0.877** | `compensating — mechanism ABSENT` |
| matched cycles 1–6 | **0.600** | `partial — quantify the exponent` |
| exponent `p` in `slack ~ rho^(−p)` | 0.106 full / **0.422** matched | — |
| cycle 1 only (cleanest) | 0.1164 / 0.2621 = **0.444** | `p = 0.675` |

The frozen spec predeclared that a material disagreement between the two would itself be
a finding. It is one, and it has a cause: see §3.

## 2. The criterion is satisfied — it is not a broken test

This contradicts the framing that one criterion is "never satisfiable".

| `rho_pf` | cycles per repeat | median slack (matched 1–6) | terminal slack | log-slack trend/cycle |
|---|---|---|---|---|
| 300 | 6 | 0.7014 | **1.1035** | **+0.0293** |
| 500 | 9 | 0.5495 | **1.0644** | **+0.0196** |
| 1000 | 16 | 0.4206 | **1.0005** | **+0.0106** |

Median slack by cycle, `rho_pf = 300`: `0.262 → 0.486 → 0.623 → 0.780 → 0.965 → 1.104`.

The slack rises monotonically and crosses 1 exactly at termination, in **all nine runs**.
Answering Q1 directly: **approaching satisfaction**, not static and not diverging.

So "binding on 93 of 93 cycles" does not mean a criterion that can never be met. It
means the criterion with the least slack throughout — the **active constraint that sets
the stopping time**. That is what a binding criterion is supposed to look like.

## 3. A selection effect that must not be read past

The run *stops* when residual convergence is met, so the terminal slack is pinned just
above 1 **by the stopping rule**, in every setting. Terminal values therefore carry no
cross-rho information, and any comparison made at termination is circular.

This is what inflates the full-trajectory ratio to 0.877: it mixes trajectories of
different lengths, each ending at a value the stopping rule fixed. The matched-cycle
comparison is the apples-to-apples one, and cycle 1 — before paths diverge — is the
cleanest. The honest reading is `p ≈ 0.42–0.68`: **partial rho domination, roughly half
to two-thirds of the algebraic effect surviving the increments' compensation.**

## 4. The real cost, which is the S2 question

| `rho_pf` | cycles to satisfy the criterion |
|---|---|
| 300 | 6 |
| 500 | 9 |
| 1000 | 16 |

**Raising `rho_pf` by 3.33x nearly triples the cycles required.** The criterion does not
become unmeetable; it becomes slower to meet. Since one candidate evaluation costs ~51
local solves per cycle, this is a direct cost multiplier and it belongs to S2.

(These runs are `RESCALED_v1.5 … ad0 nh1`, history-neutralised, and converge in 6–16
cycles — a different configuration from the production trajectory of P5.12-R, which was
still descending at cycle 21. The counts are not interchangeable.)

## 5. Q2 — commensurability, a mismatch in kind

The stationarity tolerance in force is **0.01, absolute**, identical across all three
settings (re-derived as `dual_pf_mean / dual_pf_mean_ratio`). It is applied to a
dimensionless residual that has already been scaled by rho and normalized by
`interface_rating`. The objective criterion uses `max(1e3, 1e-3 × recourse)` — a
**relative** tolerance.

So the two criteria differ not in magnitude but in kind: one is absolute on a
rho-scaled quantity, the other relative to the objective. The standard treatments
(Boyd et al., §3.3.1) use tolerances with both an absolute and a relative part, the
relative part scaled by the problem data entering each residual. The formula
`rho·(z^k − z^{k−1})` is textbook; pairing it with a purely absolute tolerance is not.

## 6. Q3 — aggregation asymmetry

Consensus criteria use **both** a max and a mean per family; stationarity uses only
`dual_<group>_mean_ratio`. The max is preserved in `cycle_detail["dual_pf"]`, and

| `rho_pf` | median `max / mean` |
|---|---|
| 300 | 8.37 |
| 500 | 8.02 |
| 1000 | 8.12 |

A max-based stationarity criterion would therefore bind about **8x harder** than the
mean-based one in force, stably across settings. Recorded as an asymmetry, not a
recommendation.

## 7. Scope limit, predeclared

All 93 cycles share one binding criterion, so the dataset contains **no counterexample**.
It supports a characterisation of `stationarity_pf`'s own behaviour and nothing about
what distinguishes a binding criterion from a non-binding one — the same
selection-on-success shape as the P5.12-W/X/Y breadth probe. P1 and P2 are
within-criterion comparisons across settings and are unaffected.

## 8. Prior-art survey — read-only

Two results change the picture.

**Most of these branches are merged, not abandoned.** Six of the nine are ancestors of
HEAD with zero commits ahead of their merge base: `origin/admm_residual_balancing_tests`,
`origin/check_convergence_per_adn`, `admm_initialization`, `admm_prev_iter_vars`,
`primal_value_update`, `admm_loop_corrections`. `consensus_vars_sess_prev_iter` does not
exist in this repository. Only `residual_balancing_mod` (42 commits, June 2024) and
`warm_start_tests` (5 commits, May 2024) carry unmerged work.

**`residual_balancing_mod` is not what its name suggests.** Its diff *removes* the blocks
labelled `# Augmented Lagrangian -- Interface power flow (residual balancing)` and
replaces the interface normalization with one based on average interface power — an
ancestor of today's `/interface_rating`. Every commit message is `Update` or `Update.`;
no evaluation artifact accompanies any of them.

**Boyd-style residual balancing already exists in production, and is switched off.**
`shared_resources_planning.py:5472-5502`: rho increases when
`primal_ratio > increase_balance_ratio × dual_ratio`, decreases when
`dual_ratio > decrease_balance_ratio × primal_ratio`, with the resulting factor applied
to `rho_v`, `rho_pf` and `rho_ess` across the TSO and all DSO models.
`admm_parameters.py:18` defaults `adaptive_penalty = False`, and every P5.10
configuration is labelled `ad0`. Crucially it balances **ratios** — residual over
tolerance — which is exactly the quantity the coupling in §1 distorts.

Commit messages are claims, not evidence: **no evaluation artifact was found for any
branch**, so what these experiments showed is not recoverable from the repository.

## What S1 establishes, and what it does not

Established: the coupling is real but partial (`p ≈ 0.42–0.68` at matched cycles); the
criterion is satisfied at termination in every preserved run and improves monotonically;
the cost of higher rho is cycles (6 → 9 → 16); the stationarity tolerance is absolute
where the objective's is relative; a max-based variant would bind ~8x harder; and the
standard remedy is implemented and disabled.

Not established, and not attempted: what tolerance is appropriate for a rho-scaled
quantity; whether adaptive rho helps here; anything about non-binding criteria. **No
lever is proposed. S1 stops here.**
