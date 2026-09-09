# Local NLP Stability Investigation Plan

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role and scope

Act as an implementation and diagnostic agent.

Read `REVISION_CONTEXT.md` first, then read this file. For the current P5.6 work, this file takes precedence regarding what may and may not be changed. The filename is retained for continuity even though the active scope has moved from local-NLP repair to deterministic nonlinear-oracle landscape robustness, coverage and derivative-free pre-search validation.

Work in small isolated experiments. After each stage report:

- files changed;
- exact diagnostic or code changes;
- commands executed;
- solver outcome;
- relevant numerical diagnostics;
- interpretation;
- whether the acceptance criterion was met.

Do not automatically proceed from a diagnostic result to a production formulation change.

---

# COMPLETED STAGE — P5.7 nonlinear operational branch-selection investigation (verdict P5.7-A, accepted)

P5.4-R completed canonical-environment revalidation. P5.5-D closed the rigorous lower-bound architecture decision and remains accepted as:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

The continuous-convex/Benders route and MISOCP lower-bound route are retired as the planning architecture. Their code/evidence remains diagnostic/benchmark-only.

Current accepted nonlinear production baseline:

`06e921e5`

Accepted nonlinear decisions remain locked:

- active-energy ESS physics;
- H1 normalized complementarity with `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- D2-P sensitivity-clean shared-S local-branch derivative bookkeeping;
- nonlinear AC SMOPF + original nonlinear ESSO retained as the physical feasibility/upper-bound model.

Active development branch:

`feature/derivative-free-planning`

## Accepted P5.6 lineage

`P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved`.

`P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved`.

`P5.6-C-C — derivative-free search is not ready`.

`P5.6-D-C — uniform refinement does not stabilize the investment landscape`.

## The conclusion P5.7 starts from

The nonlinear operational oracle is deterministic and feasible, but the evaluated investment landscape depends on the local NLP branch-selection path. Different deterministic refinement depths produce different investment rankings and even different preferred investment directions.

The planning problem is therefore **not ready for optimization**.

Derivative-free optimization is blocked. Do not implement GPS, MADS, any investment search, or surrogate optimization.

## P5.7 objectives

1. Characterize why repeated continuation/polishing finds progressively lower branches.
2. Determine whether the issue is primarily ADMM initialization, augmented-objective scaling, IPOPT local convergence, missing globalisation/continuation strategy, or an insufficiently constrained operational formulation.
3. Do not modify production equations yet.

## P5.7 required diagnostics

For **one fixed investment candidate**, collect:

- the ADMM solution;
- polish `K=0`;
- `H2`;
- `H4`;
- `H8`;
- `H12`.

For every solution compare:

- objective decomposition;
- generation dispatch;
- flexibility;
- voltage slacks;
- ESS schedules;
- TSO/DSO interface variables;
- active constraints.

Identify which variables move between branches.

## P5.7 controlled hypotheses

**A) Objective scaling.** Solve selected local NLPs with equivalent base-objective scaling and verify whether the solution branch changes.

**B) Initialization sensitivity.** Run the same NLP from the ADMM state, the previous continuation state, the best known state and a cold state.

**C) Constraint activity.** Identify active-set differences between branches.

**D) Continuation path.** Compare capacity continuation, penalty continuation and direct solve.

No planning search.

## P5.7 required output

`P5_7 — branch-selection diagnosis report`, ending with exactly one of:

`P5.7-A — unique operational oracle can likely be recovered`

or

`P5.7-B — branch multiplicity is intrinsic and planning must be reformulated`

Then stop.

# ACCEPTED P5.7 RESULTS

Verdict:

`P5.7-A — unique operational oracle can likely be recovered`.

- **hypothesis A confirmed as primary.** Production forms the ADMM subproblem objective as `base / effective_scale + consensus terms`, `effective_scale` measured at `9.41e4 .. 1.16e5` (median `1.05e5`). Re-solving all 48 subproblems from their own converged point moves the base objective by `-383401.68` as-is and `-8892463.76` rescaled -- `96.5%` of the `9217420.61` ADMM-to-polish gap;
- **hypothesis B rejected:** the same polish NLP from four starts spreads by `303.24` on `8.3e8`;
- **hypothesis C:** only `213` active rows of about `40000` change between `polish K=0` and `H12`, in `flex_energy_balance_p` and `sg_capability`;
- **hypothesis D:** at fixed consensus, four chained polishes are worth `7.3`; four refinement solves are worth `-1616084`. Continuation buys reachability, not quality;
- the ADMM converges in ONE cycle at every warm-started generation while the consensus keeps sliding;
- the TSO ADN interface anchor is inherited from `T0` and never refreshed on a warm start. The P5.7 section 5 conjecture that cold starts therefore use a different objective was **refuted by P5.8-E3** and is corrected in place in the P5.7 report.

# DELIVERED, PENDING PLANNER REVIEW — P5.8 ADMM numerical scaling validation

Report: `P5_8_ADMM_SCALING_VALIDATION_REPORT.md`. Evidence: `data/SRP1/Results/P58/`.

Delivered verdict:

`P5.8-B — objective scaling improves stability but additional ADMM issues remain`.

Reproduction gate: A0 case A reproduced the accepted P5.6-D base chain exactly; B2 reproduced P5.7-A2 to the digit. No production code, parameter file, stopping criterion or anchor policy was modified, and nothing was merged.

**A0 -- ESS/consensus tolerance screen.** Case B (`ess_mean` `1e-2 -> 1e-3`, `ess` `1e-1 -> 1e-2`) is **bit-identical** to case A at every generation, because the ESS consensus test never fires: realized `primal_ess` about `2.2e-4` and `primal_ess_mean` about `6.5e-6`, i.e. `450x` and `1500x` inside case A's tolerances. Case C (`minimum_consecutive_converged_cycles = 2`) advances the chain by exactly one refinement step and then reverts to one cycle per generation, because `consecutive_converged_cycles` is restored from `initial_state` (`shared_resources_planning.py:2093-2094`) and is inherited across warm starts. **ESS consensus tolerance is not a contributor; objective scaling remains the primary mechanism.**

**B -- rescaling validation.** `RESCALED = effective_scale * CURRENT` verified as an exact constant multiple on all 48 blocks (worst relative error `0.000e+00`). One-block: base-objective recovery `0.00 -> -257894.90` (`DSO5|2035|Autumn`) and `-0.00 -> -320612.44` (`DSO7|2035|Autumn`); expressed in base-objective units the terminal stationarity is `390x` to `1488x` tighter under RESCALED. Full 48-block replay: total base delta `-383401.68` (CURRENT) versus `-8892463.76` (RESCALED) for `33%` more IPOPT iterations, and max unscaled constraint violation improves `70x`, from `1.2241e-05` to `1.7603e-07`.

**C -- full ADMM replay** (same T0 primal state; only the objective scaling differs). Pre-polish net recourse, CURRENT versus RESCALED: `837188510.90 / 825814074.49` (gen 1, better by `11374436.41`), `836578781.24 / 825257390.49`, `835829460.61 / 824796861.45`, `835374067.36 / 824408917.11`. One rescaled ADMM run, unpolished, beats the current pipeline's fully polished generation-1 result by `2157016` and is not matched until generation 5-6; eight rescaled generations reach `823101056.86`, `580277` better than twelve current polished refinements. **But** the drift persists at `80-90%` of its former magnitude (rescaled steps `-556684 -> -251242` against `-609730 -> -313346`), neither chain reaches `tau_planning = 33031`, and the rescaled ADMM **breaks the exact-consensus polish**: failures on 1-4 DSO blocks in 7 of 8 generations (`Restoration Failed`, `Maximum Number of Iterations Exceeded`) and a `+35647836` degradation in the one generation that succeeds. Cause: better local optimality comes with worse agreement, `primal_pf` rising to the `1e-2` tolerance boundary. Rho and adaptive-penalty settings were not re-tuned; they are locked.

**D -- convergence criteria audit** (nothing modified). The stopping rule is `residual_convergence AND objective_convergence`, so consensus residuals are already not the only criterion. The binding one is the objective test: `objective_tolerance = max(1e3, 1e-3 * recourse) = 827971`, which is `25x` the `33031` planning signal and larger than every refinement step either formulation takes. Measured accepted recourse changes: `609729.66`, `749320.63`, `455393.25` (CURRENT); `787755.97 ... 251242.46` (RESCALED). Consensus residuals sit comfortably inside their tolerances throughout. Candidate future measures, **not implemented**: a recourse-change criterion tied to `tau_planning` rather than to the recourse level; per-subproblem local KKT residual in base-objective units (already in the IPOPT logs); base-objective improvement between cycles; and making the inherited `consecutive_converged_cycles` explicit.

**E -- interface anchor audit** (no policy changed). Anchor exactly constant across generations and across both formulations (`0.0` for `pc` and `qc`); identical for a different candidate warm-started from T0 (`0.000000e+00`); and within `1.003037e-07 p.u.` for that candidate started **cold**. The P5.7 conjecture that cold starts use a candidate-dependent anchor is therefore refuted, and the P5.6-B cold-versus-T0 gaps stand as genuine branch differences.

# HARD REPRODUCIBILITY GATE

All paper-instance work must use:

`/opt/anaconda3/envs/opf_env_py311/bin/python`

Canonical SRP1 checksum:

`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`

Every active harness that loads SRP1 must record:

- `sys.executable`;
- resolved conda environment;
- Python version;
- NumPy/pandas/SciPy versions;
- Pyomo version;
- IPOPT path/version and HSL solver when relevant;
- Gurobi/gurobipy version and licence status when relevant;
- realized scenario checksum.

Abort if the checksum differs. `srp_env` is noncanonical and its D3/D4 numerical evidence is historical only.

---

# ACCEPTED P5.4-R CANONICAL RESULTS

## R1 — nonlinear production regression

- DSO `36/36`;
- TSO `12/12`;
- ESSO `3/3`;
- primary failures / recoveries / persistent failures `0/0/0`;
- H1 complementarity violations `0/1728`;
- converter-capability violations `0/1728`;
- total network IPOPT iterations `3424`;
- mean / median / max `71.3 / 65.0 / 134`;
- runtime about `42 s`;
- representative equality Jacobians full row rank;
- zero zero-gradient ESS equality rows.

Earlier `3442` bootstrap iterations from `srp_env` are noncanonical.

## R2 — canonical D3

Canonical base distributed recourse:

`Q0 = 838496830.813414`

with `17` ADMM cycles and bit-identical repeated evaluations under the same canonical history.

Canonical cut threshold:

`tol_cut = 7.164e5`.

Canonical cut test:

- `2/8` decisive violations;
- worst `cut_gap ≈ -2.680808e6`;
- second decisive violation about `-1.315031e6`;
- all 18 aggregated S/E coefficients negative;
- predicted linear capacity effect remains below recourse-resolution scale.

Accepted verdict:

`P5.4-R-D3 FAIL — canonical nonlinear-recourse cuts are demonstrably unsafe`.

## R3 — canonical D4

Canonical cold base:

`838496830.81`.

Best observed recovered base branch:

`836586463.43`.

Improvement:

about `1.910367e6` (`0.228%`).

The branch fingerprint again localizes mainly to **TSO generation dispatch on Spring representative days**, partly offset by Summer changes.

Canonical hardened-cut test:

- `8/8` tested candidates decisively violate the hardened cut;
- gaps approximately `-0.96e6` to `-1.13e6`;
- all exceed canonical `tol_cut`;
- linear capacity contributions remain only O(`1e4`).

Accepted verdict:

`P5.4-R-D4 C — canonical hardened cuts remain demonstrably unsafe`.

Therefore the nonlinear-recourse derivative-cut/Benders machinery must not be used as a global lower-bound method.

## Environment sensitivity and call-history impurity are separate

Do not conflate:

- different environments generating different stochastic scenarios / different branches;
- persistent in-place data mutation causing call-history dependence.

The first is a reproducibility/environment issue. The second remains an independent software-hygiene issue to fix before nonlinear planning-UB certification.

---

# P5.5-A / P5.5-B STATUS

## P5.5-A

Historical verdict:

`P5.5-A PARTIAL — architecture is promising but unresolved convexity/bound-direction issues remain`.

Accepted inventory:

- master is an LP;
- TSO `case9`: one independent cycle;
- DSO `case33_1/2/3`: radial after preprocessing;
- one controllable continuous OLTC per DSO, range `[0.83,1.17]`;
- no phase shift/discrete tap logic;
- no controllable capacitor bank/switched shunt;
- retain conventional generation P/Q, RES P/Q/curtailment/PF control, active/reactive flexibility, shared ESS P/Q/SOC and interface P/Q;
- ordinary ESS absent in SRP1;
- lifted W-space AC SOC/QC is the intended convex family;
- no DC-OPF or LinDistFlow substitution.

## P5.5-B

Accepted verdict:

`P5.5-B PARTIAL — one or more lower-bound/cut/interface issues remain unresolved`.

This status reflects unmeasured real-model relaxation tightness and incomplete formal cut certification, not a rejected formulation.

### B1 — angle/QC safety accepted

Dormant ±30-degree angle constraints are not part of the production feasible set and are **not** lower-bound safe merely because dormant rules exist. Only class-A production constraints and class-B implications may tighten the certified LB model.

Retain `WijR >= 0`. Do not add ±30-degree limits.

### B2 — transformed OLTC accepted

Use:

`U_i = r^2 * W_ii`

`C_ij = r * W_ij^R`

`D_ij = r * W_ij^I`.

All transformer P/Q equations are affine in `U_i,C_ij,D_ij,W_jj`, verified against production to about `5.9e-15`.

Physical rank relation:

`C_ij^2 + D_ij^2 = U_i * W_jj`.

Convex relaxation:

`C_ij^2 + D_ij^2 <= U_i * W_jj`.

Continuous tap box:

`r_min^2 * W_ii <= U_i <= r_max^2 * W_ii`.

Because `W_ii > 0`, `r = sqrt(U_i/W_ii)` exists in `[r_min,r_max]`. SRP1 has no tap cost, tap-movement penalty, intertemporal coupling, discrete tap positions or phase shift. Therefore `r` and `r_sqr` are eliminated from the convex oracle. The old McCormick proposal is superseded.

### B3 — interface signs accepted; available-capacity coupling added by planner

DSO `REF` is the mathematical TSO import/export representation, not a separately costed physical generator.

The first centralized model must retain separate TSO/DSO/ESSO copies and replace ADMM consensus with exact affine equalities.

P5.5-C confirmed and implemented the planner correction that production passes ESSO **available** capacities into TSO/DSO network models. Centralization therefore includes exact capacity equalities:

`S_TSO_network = S_DSO_network = S_ESSO_available`

`E_TSO_network = E_DSO_network = E_ESSO_available`

with explicit unit/base conversion. Network SOC limits/anchors must use `E_available`, not rated E directly.

### B4 — ESSO lower-bound relaxation accepted

In the LB oracle drop:

- H1 hats/links and charge/discharge complementarity;
- non-negative complementarity objective penalty;
- ESSO degradation and SoH equalities;
- minimum-SoH restriction;
- associated non-negative slack penalties.

Retain:

`0 <= E_available <= E_rated`.

This is jointly affine and lower-bound safe. Weakness is acceptable; unknown-direction tightening is not.

### B5 — salvage planner correction

Current net recourse is gross operating cost minus terminal salvage, so salvage is a **credit**.

For a rigorous lower bound, subtract an **upper bound** on attainable salvage. Under non-negative configured salvage coefficients this maximum occurs at:

`E_available = E_rated`, equivalently `SoH = 1`.

P5.5-C verified the configured coefficient signs and derived:

`V_salvage_max(x) = sum gamma[node,cohort] * E_investment[node,cohort]`

then use:

`-V_salvage_max(x)`

in the planning lower-bound objective. The maximum-credit case is `SoH=1`; `soh_min` is not the safe lower-bound choice.

Salvage is excluded from operational `R(x)`. The safe affine salvage bound has no direct S coefficient.

### B6/B7 — Gurobi prototype evidence accepted, formal cut remains open

Use Gurobi in the canonical environment. Preferred interface:

`gurobi_persistent`.

Toy convex tests verified:

- fixing-row dual matches analytic derivative;
- `QCPDual=1` gives conic duals;
- `ObjVal`-anchored cut can be slightly unsafe at its own anchor;
- `ObjBound`-anchored cut passed a 12-point sweep.

For a scalar LB use `ObjBound`.

For a planning cut, do **not** assume:

`[ObjBound(x_k)-sigma] + g_k^T(x-x_k)`

is formally valid merely because `sigma` exceeds the scalar primal-dual gap. The desired formal contract is one dual-feasible affine function:

`L_k(x) = beta_k + g_k^T x`

with weak-duality proof:

`L_k(x) <= R(x)` for all admissible x.

P5.5-C investigated this on the real convex model and found no usable full-model dual certificate or reliable real-model derivative contract. The planning master remains blocked.

---

# ACCEPTED P5.5-C / P5.5-D RESULTS

## P5.5-C

Accepted verdict:

`P5.5-C PARTIAL`.

The centralized continuous conic oracle is implemented and useful diagnostically, but it has no usable full-model dual certificate and is far too loose for planning integration. The transformed OLTC and radial DSO SOC relaxations are not material gap sources; the meshed TSO is the dominant remaining AC-relaxation defect.

## P5.5-D

Accepted verdict:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

Authoritative D-stage conclusions:

- the H1 convex hull in `(pch,pdch,S)` is already the active-sum triangle; ~`0.5 S` circulation is a convex-hull midpoint, not a missing continuous convex inequality;
- the lower-bound-safe fixed-mode screen reduces circulation from ~`0.5 S` to ~`0.01 S` but changes the full objective by only about `322532` (`0.049%`) and closes ~`0.42%` of the gap;
- the fixed-mode value remains ~`10.51%` below the polished nonlinear feasible-network reference; because `Q_MISOCP <= Q_fixed_mode`, the unrestricted MISOCP cannot repair the gap;
- no binary stage was run; the potential ~`1728` binary full formulation was rejected before branch-and-bound on a rigorous zero-binary screen;
- stronger TSO QC/SDP work is optional future benchmark work, not the active planning path.

---

# ACCEPTED P5.6-A RESULTS

P5.6-A is accepted as:

`P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved`.

## A1 — complete nonlinear feasibility certificate

Canonical START-1 base candidate:

- max coordinated mismatch `5.551e-17`;
- original nonlinear ESSO max constraint violation `2.035e-13`;
- largest network constraint residual `1.093e-05 p.u.` at `DSO5|2030|Winter`, treated as the local IPOPT feasibility tolerance;
- gross operational cost `829291677.522120`;
- physical salvage `3439.659877`;
- net operational recourse `829288237.862242`;
- investment cost `50000.000000`;
- total planning objective `829338237.862242`.

The old P5.5-D `729.361e6` figure is superseded: P5.6-A proved that ~`95.5%` of the apparent 13% gain came from an invalid polish convention that erased TSO flexibility cost. Corrected polishing gives a genuine ~`9.21e6` (`1.0982%`) improvement over the ADMM-evaluated branch, mainly through slack/flexibility cleanup.

## A3/A4/A7 — purity, contract and caching

Per-evaluation deep copies, isolated solver-log paths and reset diagnostics close the A12 call-history mutation issue for the new oracle. The same base candidate is reproduced bit-identically across different call histories.

`evaluate_planning_candidate(...)` now checks first-stage feasibility before operational solves, executes nonlinear ADMM + original nonlinear ESSO + exact-consensus polish + full feasibility audit, returns explicit failure statuses, and caches only VALID completed results.

The canonical base lies at the minimum E/S ratio `E=2S`, so negative E-only search moves are first-stage infeasible.

## A5/A6 — start and runtime evidence

START-2 uses one fixed archived base-candidate template and is materially better/cheaper than cold on the tested candidates:

- START-1 cold base: `829338237.862242`, ~`562 s`, `17` ADMM cycles;
- START-2 template base: `828021090.3608505`, `2` ADMM cycles; one-off build included in the first ~`603 s` evaluation, then ~`100 s` on subsequent candidates;
- START-2 beats cold by roughly `0.64e6` to `1.32e6` on all three dual-start candidates;
- cold objective span across the three is ~`674337`; template span is ~`2024`.

Historical P5.6-A note, superseded by P5.6-B: the A report defined `Q_oracle` as the minimum over both starts, which would cost both solves. P5.6-B subsequently locked the recurring direct search start to `T0 ONLY`; cold is now final-certification / periodic-audit only. The ~`115 s` recurring successful-evaluation cost therefore refers to T0-only evaluation.

## A6 — interface-anchor issue

Midpoint polishing failed on one of four benchmark candidates. DSO-side anchoring repaired it. On the base candidate, DSO anchoring changes the total objective by `13053.97`, equal to `0.018 * tol_cut` but larger than the observed ~`2024` template-started candidate span.

Do not use the old Benders `tol_cut` as the derivative-free search-resolution threshold. P5.6-B subsequently established exact numerical repeatability and proposed `10.0`; P5.6-C renames this `tau_numerical = 10.0` and must derive a separate `tau_planning` from cross-template relative landscape variation.

---

# ACCEPTED P5.6-B RESULTS

P5.6-B is accepted as:

`P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved`.

The verdict wording is historical; planner interpretation is more specific: midpoint-only anchor and T0 direct start are locked, while template **landscape representativeness**, false hidden-infeasibility coverage and planning-level branch uncertainty remain unresolved.

## B0/B1 — best incumbent and template chain

Current best rigorously feasible nonlinear planning incumbent at the canonical base investment:

- total planning objective `828021090.360850`;
- net operational recourse `827971090.360850`;
- gross operational cost `827974518.717105`;
- physical salvage `3428.356255`;
- investment cost `50000.000000`.

Template refinement on the same base candidate:

- T0 `828021090.360850`;
- T1 `827415318.563944`;
- T2 `826824028.845478`;
- T3 `826405022.193437`;
- T4 `825961521.882321`.

No stabilization is established. T0 is frozen by an explicit reproducibility rule, not because it is a fixed point. T0 id:

`a81f7f5191dd42dbf50d1726149b8909`.

Purity under T0 remains bit-identical across different call histories.

## B2 — midpoint anchor locked

On the 12-candidate benchmark population, midpoint and DSO anchors have identical success rate on operationally evaluated candidates (`8/11 = 72.7%`). Under T0 the DSO anchor rescues no midpoint failure. Anchor-induced objective changes range approximately `-13365.80 .. +11175.98` and can reverse candidate orderings.

Recurring search anchor:

`MIDPOINT ONLY`.

Both anchors remain available only for final incumbent certification.

## B3 — T0 direct start locked, coverage incomplete

Cold is never better than T0 on objective where both succeed, by about `1.32e6 .. 1.09e7`, so the recurring direct search start is:

`T0 ONLY`.

Cold is final-certification / periodic-audit only.

However, T0 and cold have different failure sets. At least one T0 `POLISH_FAILURE` candidate is VALID under cold, and the budget-boundary candidate crashes under both. Therefore operational failure is a hidden solver/branch-access issue unless physical infeasibility is separately proved.

## B4 — numerical threshold

Same-candidate repeatability under the locked T0+midpoint policy is exactly zero. P5.6-B proposed:

`tau_search = 10.0`.

For P5.6-C terminology this becomes:

`tau_numerical = 10.0`.

`tau_planning` is still pending and must be derived from cross-template **relative-to-base** landscape variation. Do not use historical Benders `tol_cut` as a derivative-free resolution criterion.

## B5/B6 — coordinates and search family

Use `(S,h)` with:

`h = E - 2S >= 0`, `E = 2S + h`.

This is a bijective change of variables on the active minimum-duration relation. Remaining max-duration, cumulative-capacity, lifetime and budget constraints are exact first-stage checks; do not project requested candidates silently.

Deterministic MADS/pattern search remains the selected family in principle. OrthoMADS is preferred conceptually, but no search is authorized until P5.6-C audits actual implementation availability and deterministic batch semantics.

## B7 — cost

Under T0 + midpoint:

- successful uncached evaluation ~`115 s`;
- blended observed cost ~`134 s`;
- final multi-start/both-anchor certification ~`740 s`;
- one-off T0 construction ~`518 s`;
- cache hit ~`0.040 s`.

Four workers are recommended. Eight workers risk resource contention and have already coincided with IPOPT process crashes.

---

# ACCEPTED P5.6-C RESULTS

Verdict:

`P5.6-C-C — derivative-free search is not ready`.

- the investment landscape is **not** stable across deterministic template generations: Spearman T0 vs T4 = `-0.033`, and 6 of 9 candidates reversed the sign of their improvement over base;
- deterministic continuation from the fixed origin `x0` rescues every failure, but shifts even direct-VALID controls by about `2.9e6`, so continuation cannot be a failure-only fallback — it has to be the oracle itself, applied identically to every candidate;
- `tau_numerical = 10.0` and `P56C_TEMPLATE_LANDSCAPE_UNCERTAINTY = 4.25e5` were separated as required;
- deterministic batch polling semantics were fixed;
- no trustworthy MADS implementation is installed, and nothing was installed;
- the P5.6-B `se|ALL|x19` `SOLVER_CRASH` record was **withdrawn**: it is not reproducible and is VALID under T0/T2/T4 and under continuation;
- the P5.6-B7 cost figure of ~`115 s` was ADMM+polish only; end-to-end medians are `135 s` VALID and `186 s` failed.

# ACCEPTED P5.6-D RESULTS

Verdict:

`P5.6-D-C — uniform refinement does not stabilize the investment landscape`.

The uniformly refined oracle:

```
x0     = the canonical positive-bootstrap base investment, FIXED FOREVER
T0     = a81f7f5191dd42dbf50d1726149b8909
anchor = MIDPOINT

lambda_j = j / K,  x_j = (1 - lambda_j) * x0 + lambda_j * x,  j = 1..K
state_0 = T0;  each VALID step's state initializes the next
H_K(x) = total planning objective at x_K = x
```

Generation labelling: the INPUT state of step `j` is `state_{j-1}`, the OUTPUT state is `state_j`. Every delta is `Delta_K(x) = H_K(x) - H_K(x0)` at the **same** `K`.

Accepted base refinement chain at `x0` (bit-identical reproduction of the P5.6-C first four values in a different process):

| step | `H(base)` |
|---|---|
| 1 | `828021090.360850` |
| 2 | `827415318.563944` = `H_2(x0)` |
| 3 | `826824028.845478` |
| 4 | `826405022.193437` = `H_4(x0)` |
| 5 | `825961521.882321` |
| 6 | `825531306.746985` |
| 7 | `825108709.695181` |
| 8 | `824795363.718628` = `H_8(x0)` |
| 9 | `824488243.726694` = `H_8plus(x0)` |
| 12 | `823731333.558647` = `H_12(x0)` |

What uniform refinement fixed:

- coverage is `100%` from `K=4` onward, including the budget-boundary candidate and the three that direct-T0 could not polish;
- improvement **directions** are largely consistent — 1 sign reversal in 7, against 6 in 9 under the P5.6-C policy.

What it did not fix:

- `tau_planning_refined = 811438.05`, against the `4.25e5` benchmark it had to beat — a factor of `1.9` in the wrong direction;
- the best candidate changes identity at every tested depth transition: `se|node5|2025|-10%` (K=2, K=4) -> `se|node9|2025|-10%` (K=8) -> `se|node9|2025|+10%` (K=12);
- best-to-second gaps are `8335 .. 33031`, so the uncertainty is `25` to `97` times the signal;
- depth movement decays by a factor of only `2.10` between the two transitions measured, leaving `386978` at `K=12`;
- **decisively**, at node 9 / 2025 the oracle reverses the preferred SIGN of the investment between `K=8` and `K=12`;
- `D8` found no `K_STAR`, so the `D10` path-schedule check was not run.

Best known rigorous feasible nonlinear planning incumbent, promoted in D0.1:

`se|node5|2025|-10%` at `825109566.571083`, confirmed by a bit-identical K=4 re-run.

Cost, end-to-end median per candidate: `250.3 s` (K=2), `503.6 s` (K=4), `999.1 s` (K=8), `1489.2 s` (K=12); `100.8 s` per terminal self-refinement; one-off T0 build about `500 s`.

# CURRENT SOLVER / ORACLE POLICY FOR P5.6-C/D AND P5.7

Use only the accepted nonlinear production solvers and settings. Do not retune IPOPT/MA97, ADMM, H1, proximal regularization, recovery logic or adaptive rho during C.

For direct recurring evaluations, the declared baseline policy is:

- template/start: `T0 ONLY`;
- T0 id: `a81f7f5191dd42dbf50d1726149b8909`;
- anchor: `MIDPOINT ONLY`;
- coordinates: `(S,h)`, `h = E - 2S`;
- numerical comparison threshold: `tau_numerical = 10.0`;
- cache key includes the exact template id and anchor convention.

Do not interpret a T0 failure as physical infeasibility. P5.6-C must test deterministic continuation coverage before any search uses an extreme barrier for these points.

Cold and alternate anchors remain available for final certification/diagnostics only unless P5.6-C explicitly changes the recurring policy.

---

# COMPLETED STAGE — P5.6-C landscape robustness and oracle coverage gate (historical stage specification)

This is the final gate before any derivative-free optimization run.

## C1 — correct the P5.6-B interpretation

Record that P5.6-B settled midpoint-only anchoring and T0 direct start. The unresolved issues are:

- cross-template investment-landscape representativeness;
- domain coverage / false hidden infeasibility;
- planning-level branch uncertainty;
- deterministic parallel poll semantics;
- practical MADS/pattern-search implementation path.

## C2 — cross-template landscape test

Use the fixed P5.6-B master-feasible benchmark population. Evaluate jointly reachable candidates under:

`T0`, `T2`, `T4`

with midpoint anchor only and the complete original nonlinear oracle contract.

For every candidate/template compute:

`Delta_Tk(x) = Q_Tk(x) - Q_Tk(base)`.

Across jointly VALID candidates report:

- Spearman rank correlation;
- Kendall rank correlation;
- complete ranking;
- maximum rank displacement;
- pairwise ordering reversals and worst reversal magnitude;
- `Delta_T0`, `Delta_T2`, `Delta_T4` per candidate;
- per-candidate delta range/std across templates;
- diagnostic affine fits between template objective levels.

Do not treat high correlation as sufficient if improvement over the base changes sign materially.

Classify LANDSCAPE-STABLE only if relative improvement signs/rankings are sufficiently preserved and the template effect is predominantly a common offset. If stable, keep T0. If unstable, do not launch MADS.

## C3 — deterministic continuation fallback

For each master-feasible target `x`, define the straight feasible first-stage path from canonical base `x0`:

`x(lambda) = (1-lambda)*x0 + lambda*x`.

Use fixed schedule only:

`lambda = 0.25, 0.50, 0.75, 1.00`.

Starting from the frozen T0/base state, solve each point in order. A complete VALID state may initialize the next point. Every point must run:

- original nonlinear ESSO;
- production ADMM;
- midpoint-only exact-consensus polish;
- full physical feasibility audit.

No adaptive lambda schedule and no anchor switching in C.

Test on:

- every T0/midpoint `POLISH_FAILURE` from B;
- the budget-boundary `SOLVER_CRASH` candidate;
- at least two direct-T0 VALID controls.

For controls compare direct and continuation objectives to determine whether continuation is a neutral coverage rescue or a materially different branch-selection surface.

## C4 — lock coverage policy

Compare:

A. direct T0 only;

B. direct T0, fixed continuation only on direct failure;

C. fixed continuation for every candidate.

Prefer B only if continuation improves coverage and gives compatible objectives on direct-valid controls. If continuation materially changes valid-control objectives, either use C for every candidate or return PARTIAL. Never mix incompatible surfaces silently.

Recompute success rate and ordinary/failed/blended cost under the chosen policy.

## C5 — define `tau_planning`

Keep:

`tau_numerical = 10.0`

unless new repeatability evidence requires otherwise.

Define separately:

`tau_planning`

from cross-template relative-to-base delta variation. This threshold describes robustness of investment improvements to known branch/template ambiguity. Do not include absolute common template offsets if rankings/deltas are stable.

Future search may use `tau_numerical` internally on the locked surface, but incumbent promotion and scientific claims must respect `tau_planning`.

## C6 — deterministic parallel polling

Future parallel search must be deterministic batch polling, never asynchronous first-finish opportunism:

1. generate ordered deterministic poll set;
2. first-stage feasibility screen;
3. remove duplicate/cache-hit points;
4. select batch deterministically;
5. evaluate up to four points in parallel;
6. wait for the complete selected batch;
7. choose the best VALID objective;
8. break ties within `tau_numerical` by deterministic poll index;
9. only then update incumbent/mesh.

This requirement preserves reproducibility under variable solver completion times.

## C7 — implementation inventory

Without installing anything, audit:

- NOMAD / PyNOMAD availability;
- repository pattern/MADS utilities;
- licence compatibility;
- hidden-constraint evaluator support;
- exact first-stage feasibility screening;
- deterministic direction/seed control;
- cache integration;
- deterministic batch/parallel evaluation.

Do not hand-code “OrthoMADS” casually. If a trustworthy MADS implementation is unavailable or disproportionate to integrate, a deterministic positive-spanning generalized pattern search is acceptable for the first campaign with limitations stated explicitly.

## C8 — no optimization run

Do not run the 200-evaluation campaign during P5.6-C.

## C9 — required verdict

End one:

`P5.6-C-A — derivative-free search is ready to launch`

or

`P5.6-C-B — derivative-free search is ready only on a restricted validated domain`

or

`P5.6-C-C — derivative-free search is not ready`.

For C-B, explicitly define the validated local domain from tested successful directions. For C-A, state the final recurring oracle/coverage policy, `tau_numerical`, `tau_planning`, deterministic poll semantics and selected implementation path.

Then:

`P5.6-C COMPLETE — ready for planner review before any optimization campaign`.

---

# LOCKED PRODUCTION DECISIONS — in force during P5.6-C/D and carried into P5.7

Do not change:

- nonlinear AC SMOPF equations;
- active-energy ESS formulation;
- D2-P sensitivity-clean shared-S formulation;
- H1 dimensionless complementarity and `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- IPOPT tolerances/options;
- MA97/exact-Hessian policy;
- recovery/adaptive-rho/proximal policies;
- existing master/Benders equations;
- P5.5 convex/MISOCP diagnostic code.

Do not yet implement:

- derivative-free investment optimization;
- replacement outer planning loop;
- production MISOCP planning;
- distributed convex ADMM;
- QCP/Benders cut recovery;
- TSO SDP/QC strengthening.

---

# OLD P5.4-G STATUS

P5.4-G is permanently blocked for the old nonlinear-recourse derivative cuts. Do not run it.

Any later planner is a new architecture.

---

# DEFERRED ITEMS

Keep deferred during P5.6-C:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R until defensible converter `Smax` data exists;
- nonlinear solver/ADMM retuning;
- distributed convex ADMM;
- full/production MISOCP planning;
- QCP/Benders cut recovery;
- TSO SDP/QC strengthening except as optional future diagnostic benchmark;
- actual derivative-free optimization run;
- surrogate-assisted planning unless later selected explicitly;
- any change to accepted nonlinear model mathematics.

# COMPLETED STAGE — P5.3 (historical execution record)

This historical section superseded the older frozen-cycle-only instructions during P5.3. It no longer defines the active scope; P5.4 above is authoritative.

The historical frozen-model work remains valid evidence and is summarized later in this file, but the primary P5.3 population is now the real current SRP1 positive-bootstrap cold initialization produced through the production candidate/model-construction path.

Current accepted production checkpoint:

`f77d829359ffd873367f556882546bc2dcc8ec99`

Current reduced-scenario identity:

- seed `2026`;
- combined scenario checksum:
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

P5.3 was a diagnostic/reformulation review. Its B3 direction has now received planner approval for controlled P5.4 productionization, not direct acceptance of the diagnostic wrapper.

## Current P5.3 invariants

Do not:

- tune IPOPT `tol`, `acceptable_tol`, `acceptable_iter`, `max_iter`, or pushes;
- switch production MA97/MA57 policy;
- change recovery classification;
- change ADMM rho rules or tolerances;
- change recourse-stationarity criteria;
- change ADMM objective scaling;
- change TSO proximal regularization;
- change Benders/local-cut logic;
- add generic feasibility slacks;
- productionize the P5.2 shared-ESS narrow band;
- cap shared-ESS `kappa`;
- change the stochastic scenario values during exact RES algebra A/B tests;
- add calendar degradation;
- modify terminal salvage;
- silently change ESS complementarity tolerance.

The active-power ESS network prototype described below is explicitly authorized as a diagnostic physical reformulation. Full ESSO cycling-degradation conversion is not part of P5.3-B3.

---

# P5.3 checkpoint after A/A2 — AUTHORITATIVE

P5.3-A and P5.3-A2 are complete. The detailed A/A-RES instructions retained below are historical execution records and **must not be rerun wholesale** unless a later B experiment explicitly requests a targeted derivative control.

Authoritative findings:

- full equality Jacobians are exactly row-rank deficient at the positive-bootstrap cold start because of `sess_snet_def` only;
- zero equality rows: 24 per DSO model, 72 per TSO model;
- full `sigma_min = 0` and full condition number is formally infinite;
- after removing only the exactly-zero rows, the tested reduced equality Jacobians have full row rank and no additional nullity;
- corrected reduced condition numbers are approximately `8.98e4` for DSOs and `1.42e3` for the TSO;
- the previous claim that TSO equality conditioning is worse is withdrawn;
- the previous near-zero `pij/qij` DSO column result is withdrawn as a derivative-audit artifact caused by uninitialized `r_sqr` rows being skipped;
- `sess_snet_def` remains HIGH risk: exact zero gradient, always-active equality, sole source of exact equality-rank deficiency, curvature up to about `18806`;
- `sess_comp` remains HIGH risk: tiny Jacobian and tiny `S_rated^2`-scaled margin;
- `sg_capability` remains HIGH risk: 3732 active rows, zero cold-start margin, gradients down to about `5.44e-5`;
- all 144 current SRP1 curtailable RES instances have `power_factor_control=True`; therefore `gen_pf_profile` is never instantiated and the old exact RES B2 cleanup is cancelled for SRP1;
- current RES synthetic `q_available` is zero, so `sg_available = P_available` and the capability circle collapses with stochastic active availability;
- RES `abs()` reflection is negligible; upper-support overshoot is material, reaching about `33.5%` above historical maxima in some season/type calls;
- the realized reduced SRP1 population has no values in `(0,1e-5]`, but 17 live values in `(1e-5,1e-4]`;
- cross-generator spatial correlation is not preserved by the current same-type pooled sampling assignment;
- DSO interface magnitude is intentionally freed in ADMM; the earlier claim of persistent DSO magnitude pinning is withdrawn;
- IPOPT internally scales the objective gradient but does not supply comparable constraint-row scaling.

## Historical B-series execution order (completed)

The user deliberately selected the following order because the RES capability experiment is easier and quicker to debug and validate than the active-power ESS refactor:

1. **P5.3-B1 — exact reference-angle gauge**;
2. **P5.3-B2-R — RES capability semantics and conditioning**;
3. **P5.3-B3 — active-power ESS structural prototype**.

B3 remains the highest expected structural-payoff experiment, but it is executed third for practical validation reasons.

Do not test more shared-ESS epsilon values or scalar `kappa` caps. Do not retune solver options.

---

# P5.3-A — Quantitative structural conditioning audit (COMPLETE — historical execution instructions)

## A1. Reproduce the real positive-bootstrap population

Generate the exact P5 iteration-2 positive-bootstrap candidate using production:

`_build_positive_bootstrap_candidate(...)`

and replay the real pre-solve initialization path.

Audit every network SMOPF immediately before IPOPT:

- 36 DSO models;
- 12 TSO models.

Retain ESSO models for structural context, but the primary derivative audit is the 48 network SMOPFs.

Do not reconstruct the bootstrap candidate manually.

The old frozen cycle-10 pickle may be used only as historical/control evidence.

## A2. Inspect the installed derivative environment first

Determine the safest available mechanism for evaluating the NLP Jacobian and related derivatives.

Prefer analytic Pyomo/PyNumero/NLP interfaces already available in the environment.

Do not introduce a large external dependency merely for this audit.

If an exact sparse Jacobian interface is unavailable, report that limitation before implementing a materially different numerical approach.

## A3. Constraint-family inventory

Enumerate every active network constraint component and classify it as:

- linear equality;
- linear inequality;
- nonlinear equality;
- nonlinear inequality;
- ranged nonlinear inequality.

For every family report:

- row count;
- polynomial degree where available;
- variables participating;
- typical variable magnitude;
- whether variables/row residuals can naturally be near zero;
- whether a row can be active at a zero-gradient point;
- whether stochastic data can switch the row on/off structurally.

Explicitly include at least:

- `voltage_mag_def`;
- `voltage_mag_sqr_def`;
- `voltage_setpoint_cons`;
- `voltage_product_real_def`;
- `voltage_product_imag_def`;
- `r_sqr_def`;
- `sg_capability`;
- `gen_pf_upper`;
- `gen_pf_lower`;
- `gen_pf_profile`;
- `flex_energy_balance_p`;
- `ess_pnet_def`;
- `ess_snet_def`;
- `ess_comp`;
- `ess_soc_def`;
- `sess_pnet_def`;
- `sess_snet_def`;
- `sess_comp`;
- `sess_soc_def`;
- `node_balance_p`;
- `node_balance_q`;
- `branch_flow_limit`;
- `branch_flow_limit_ji`;
- all distributed coordination/interface constraints added after base model construction.

## A4. Jacobian diagnostics at the cold start

For every active row, where technically available, compute:

- absolute constraint residual / violation;
- `||grad g||_2`;
- `||grad g||_inf`;
- smallest nonzero absolute Jacobian coefficient;
- largest absolute Jacobian coefficient;
- intra-row coefficient ratio;
- distance to the nearest inequality bound;
- `distance_to_bound / IPOPT_tol` using that model's actual configured `tol`.

Do not hide zero-gradient rows.

Report counts of rows with gradient norm below:

- `1e-12`;
- `1e-10`;
- `1e-8`;
- `1e-6`;
- `1e-4`.

These are diagnostic bins only.

Group by component, agent/network, year, and representative day.

Produce ranked top-N summaries for:

- smallest row norms;
- largest row norms;
- smallest inequality margins relative to IPOPT tolerance;
- largest intra-row derivative-scale ratios.

## A5. Jacobian column and near-dependence diagnostics

Report:

- zero/near-zero derivative columns;
- smallest/largest column norms;
- suspicious variable families.

Where practical, estimate:

- equality-Jacobian numerical rank;
- smallest singular values;
- largest singular value;
- a condition estimate.

At minimum do this for:

- one representative TSO model;
- one representative model from each DSO;
- every previously sensitive/failing positive-bootstrap state.

If full SVD is impractical, use a sparse extremal-singular-value or rank-revealing alternative and state the limitation.

Also identify pairs/groups of equality rows with nearly collinear normalized gradients.

## A6. Constraint-curvature audit

For nonlinear quadratic/bilinear rows report raw second-derivative/Hessian coefficient scales where practical.

Pay particular attention to:

- `sess_snet_def`;
- `ess_snet_def`;
- `sess_comp`;
- `ess_comp`;
- `sg_capability`;
- branch-current/apparent-power limits;
- squared-voltage/product definitions.

For shared ESS explicitly evaluate the bootstrap power ratings:

- `1.0635e-4 p.u.`;
- `2.1270e-4 p.u.`;
- `3.1905e-4 p.u.`;

with current:

`kappa = 1/S_rated`.

Distinguish clearly among:

- small Jacobian;
- large curvature;
- tiny inequality margin;
- exact rank degeneracy;
- near-linear dependence.

Do not collapse them into one generic "bad conditioning" label.

## A7. Objective-gradient scale

At the cold start report objective-gradient norms/coefficient ranges for major components that are present:

- physical/economic SMOPF objective;
- scenario-deviation regularization;
- proximal terms;
- ADMM augmentation.

Compare objective-gradient scale with suspicious constraint-Jacobian scales.

Do not change objective scaling.

---

# P5.3-A-RES — Stochastic RES and low-output audit (COMPLETE — historical execution instructions)

The current load and RES realizations come from historical-data-based copula/KDE models. Do not replace the copula model in this stage.

## RES1. Raw support and `abs()` post-processing

Instrument RES generation so the inverse-transformed samples are inspected **before**:

`np.abs(...)`.

For each network / season / RES type (PV, Wind) report:

- number of sampled hourly values;
- number/percentage negative before `abs`;
- minimum negative value;
- total/magnitude of positive generation created solely by reflection of negative samples;
- values above historical maximum;
- values below historical minimum;
- relevant quantiles before and after post-processing.

Do not alter `np.abs` during this audit.

## RES2. Tiny-generation population in the realized SMOPFs

After conversion to p.u. and after the actual one-scenario realization is selected, count available RES values in:

- exactly zero;
- `(0, 1e-6]`;
- `(1e-6, 1e-5]`;
- `(1e-5, 1e-4]`;
- `(1e-4, 1e-3]`;
- `>1e-3`.

For tiny values report exact:

- network;
- generator id/type;
- year;
- day;
- hour.

Cross-reference them with:

- `renewable_generation_is_unavailable`;
- `sg_capability`;
- `gen_pf_profile`;
- PF-control rows;
- local solve failures;
- unusually high IPOPT iteration counts where available.

Quantify how often the current structural switch at `EQUALITY_TOLERANCE` is exercised or nearly exercised.

Do not select a replacement threshold yet.

## RES3. Reactive-power/profile assumptions

For every curtailable RES generator report:

- `power_factor_control`;
- `min_pf`, `max_pf`;
- physical `pmax`, `qmin`, `qmax`;
- stochastic `pg_available`;
- stochastic `qg_available`;
- whether `qg_available` is identically zero;
- whether `gen_pf_profile` is instantiated;
- whether `sg_capability` is instantiated.

Explicitly identify rows where:

`q_available * pg == p_available * qg`

reduces to:

`p_available * qg == 0`

with very small positive `p_available`.

## RES4. Apparent-power capability interpretation

Determine whether current stochastic `sg_available = sqrt(pg_available^2 + qg_available^2)` is intentionally representing converter MVA rating or only stochastic renewable availability.

If `qg_available = 0`, the current capability radius collapses to `pg_available`.

Report whether the data/model contains a separate physical inverter MVA rating that could support a cleaner separation:

- stochastic active availability: `0 <= pg <= P_available`;
- converter capability: `pg^2 + qg^2 <= S_converter^2`.

Do not implement this physical change in the exact RES B2 experiment unless separately approved; it may change the feasible set.

## RES5. Spatial/scenario correlation observation

Audit how generated PV/wind profiles are assigned to individual generators.

Determine whether generator-to-generator spatial correlation is preserved or whether same-type generators effectively draw independently from a common synthetic pool.

Report this as a scenario-realism finding, separate from local NLP conditioning.

Do not change the sampling architecture in P5.3.

---

# P5.3-A-extra — Other structural checks

## Reference-angle/gauge audit

Confirm current reference-bus treatment of `f` and quantify any residual rotational/gauge degree of freedom.

Also audit DSO reference-bus `e` bounds and determine whether they effectively pin the coordinated interface voltage despite `enforce_vg = false`.

Do not change them in Phase A.

## Transformer auxiliary audit

Determine whether `r` and `r_sqr` variables constructed for non-transformer branches enter the generated NL problem.

If the writer eliminates them, classify as code cleanliness only.

If they reach IPOPT as unused/weakly-connected variables, classify as a conditioning issue.

## Branch-current audit

Quantify cancellation and derivative scales in current-limited branch rows based on:

`V_i^2 + V_j^2 - 2*W_ij_real`

multiplied by branch series-admittance magnitude squared.

Report whether low-impedance DSO branches create extreme coefficients or cancellation-sensitive constraints.

Do not reformulate branch currents unless this family ranks materially high.

---

# P5.3-A required output

Produce:

`P5_3_A_SMOPF_CONDITIONING_AUDIT.md`

Include a ranked `HIGH / MEDIUM / LOW` table for suspicious formulation families.

For every HIGH-risk item state:

- mathematical form;
- observed numerical evidence;
- physical role;
- exact failure mode: zero gradient, poor scale, near dependence, tiny margin, large curvature, etc.;
- possible exact reformulation;
- possible deliberate physical reformulation;
- expected numerical benefit;
- risk of changing the feasible set.

P5.3-A/A2 are complete. Do not rerun the full audit before B1/B2-R/B3; use only targeted derivative controls required by the specific experiment.

---

# P5.3-B1 — Exact reference-angle A/B

Proceed only if Phase A confirms that fixing the reference imaginary voltage does not violate an intentional interface convention.

Start from fresh accepted production models.

Do not carry any P5.2 narrow-band ESS change into this branch.

A — current production reference treatment.

B — exact gauge:

`f_ref = 0`.

Change only this fixing/bound condition.

Keep every other production equation and solver option unchanged.

Run the complete positive-bootstrap initialization.

Report:

- 51 final local outcomes;
- 48 network primary/recovery outcomes;
- IPOPT iterations;
- KKT metrics;
- objective values;
- failures by identity;
- equality-Jacobian rank/smallest-singular-value change on representative models;
- interface V/P/Q differences.

Acceptance:

- no material physical/economic change;
- no new failure family;
- gauge ambiguity removed;
- conditioning not worse.

Diagnostic only. Do not productionize automatically.

---

# P5.3-B2-R — RES capability semantics and conditioning

Run this **second**, after B1 and before B3.

Use a fresh accepted production baseline. Do not stack B1.

The old B2 exact fixed-PF/profile cleanup is cancelled for SRP1 because all 144 curtailable SRP1 RES instances have `power_factor_control=True`, so `gen_pf_profile` is never instantiated.

B2-R is a semantics-first diagnostic. It must not invent a converter rating.

## B2-R.1 Rating semantics first

For every curtailable SRP1 RES generator inspect:

- `pmax`;
- `pmin`;
- `qmax`;
- `qmin`;
- `min_pf`, `max_pf`;
- generator type;
- any explicit `S_rated`, inverter rating, converter rating, nameplate MVA field, or equivalent metadata;
- network JSON comments/metadata;
- historical operational-data units/meaning.

Determine whether the repository contains an **explicit defensible converter apparent-power rating**.

Do not infer `S_converter` merely because `pmax` and `qmax/qmin` exist unless their documented semantics make that inference unambiguous.

If no defensible converter MVA rating exists, STOP B2-R before formulation implementation and recommend the minimum data-model extension required.

Do not synthesize a rating from an arbitrary heuristic.

## B2-R.2 Explain the current RES feasible set

Current production effectively has:

`0 <= pg <= P_available`

plus:

`pg^2 + qg^2 <= S_available^2`

where:

`S_available = sqrt(P_available^2 + Q_available^2)`.

For the current synthetic SRP1 RES data:

`Q_available = 0`,

so:

`S_available = P_available`.

Explain mathematically, together with the active PF cone, what reactive-power capability remains at:

- `pg = P_available`;
- partial active dispatch;
- very low `P_available`;
- `P_available = 0`.

State whether the current behavior appears physically intentional or is likely a conflation of stochastic resource availability with converter/inverter MVA capability.

## B2-R.3 Conditional physical prototype

Proceed only if B2-R.1 finds an explicit defensible `S_converter`.

Create an isolated diagnostic formulation:

stochastic resource availability:

`0 <= pg <= P_available`

converter capability:

`pg^2 + qg^2 <= S_converter^2`

while retaining the existing PF-control inequalities and all stochastic scenario values unchanged.

This is a deliberate feasible-set change, not an exact algebraic rewrite.

Do not change:

- PF limits;
- RES-off threshold;
- stochastic samples;
- objective penalties;
- IPOPT settings;
- ADMM/planning settings.

## B2-R.4 Numerical and physical comparison

If the conditional prototype is implemented, compare production vs B2-R for:

- all 17 realized low-output rows in `(1e-5,1e-4] p.u.`;
- representative normal-output rows;
- all 48 network positive-bootstrap initialization solves;
- final 51 local outcomes where the unchanged ESSO initialization is relevant.

Report:

- `sg_capability` cold-start margin;
- row-gradient norm;
- whether the capability row is active at the cold start;
- primary/recovery/persistent failure counts;
- IPOPT iteration distribution and runtime;
- P/Q capability and dispatch;
- RES curtailment;
- local objective;
- interface V/P/Q;
- any new physical degrees of freedom introduced by separating availability from converter rating.

Determine whether the tiny active capability circle at low stochastic generation disappears when the converter radius is based on a fixed physical rating.

## B2-R.5 Stochastic-support recommendation — diagnostic only

Do not change the copula/KDE generator in B2-R.

Based on P5.3-A2, provide a separate recommendation on:

- handling of synthetic samples above physical/historical support;
- whether post-generation clipping to an explicit physical/nameplate maximum is justified;
- whether the marginal model itself should be bounded;
- how generator-site/spatial dependence should be represented in a later scenario-model revision;
- whether the current negligible `abs()` reflection still merits cleanup for physical clarity even though it is not a material numerical driver.

Keep stochastic-model recommendations separate from the NLP capability result.

## B2-R acceptance

Classify B2-R as one of:

- `PRODUCTIONIZE CANDIDATE` — explicit converter rating exists and the separated formulation improves physical semantics/conditioning without unacceptable regressions;
- `CONTINUE TESTING` — data/formulation is promising but evidence is incomplete;
- `DEFER — DATA MODEL REQUIRED` — no defensible converter rating exists;
- `REJECT` — reformulation is unsupported or creates unacceptable behavior.

Do not productionize automatically.

---

# P5.3-B3 — Active-power ESS structural prototype

Run this **third**, after B1 and B2-R. It remains the highest expected structural-payoff experiment, but is intentionally sequenced last because it is the largest refactor to debug and validate.

Use another **fresh accepted production baseline**.

Do not stack B1 or B2-R.

This is an authorized diagnostic physical reformulation, not an exact algebraic rewrite.

## B3.1 Trace all affected consumers first

Before changing a diagnostic branch, trace every network/coordination/ESSO consumer of:

- `sch`;
- `sdch`;
- `pch`;
- `pdch`;
- `pnet`;
- `qnet`;
- SOC;
- complementarity;
- converter limits;
- result processing;
- ADMM shared-ESS P/Q coupling;
- degradation/throughput.

Confirm that network-agent coordination is based on aggregate P/Q schedules and identify every place that would break if `sch/sdch` were removed.

## B3.2 Network SMOPF prototype

For ordinary and shared network ESS prototype:

`pnet = pch - pdch`

SOC from active power:

`SOC_t = SOC_{t-1} + eta_ch*pch*Delta_t - pdch*Delta_t/eta_dch`

Use the actual model time basis for `Delta_t`; do not assume it silently.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`.

Reactive power remains converter loading but does not directly change stored battery energy.

Remove from the network diagnostic prototype:

- `sch`;
- `sdch`;
- `ess_snet_def`;
- `sess_snet_def`;
- `ess_pch_link`;
- `ess_pdch_link`;
- `sess_pch_link`;
- `sess_pdch_link`.

Replace the old apparent-power sum limit with explicit active-power bounds/sum constraints justified from the original device rating and intended physical behavior.

Complementarity acts on:

`pch * pdch`.

Do not silently change the configured complementarity tolerance.

Audit the new complementarity row's scale explicitly. If it remains a high-risk tiny bilinear inequality, report that rather than hiding it with arbitrary scaling.

Shared-ESS zero-capacity gating must remain safe.

## B3.3 Required physics tests

Demonstrate:

- pure Q: `pch = pdch = pnet = 0`, `qnet != 0` leaves SOC unchanged;
- pure charging changes SOC according to efficiency and time step;
- pure discharging changes SOC according to efficiency and time step;
- reactive power remains feasible up to converter capability;
- simultaneous active charging/discharging respects the intended complementarity semantics;
- zero-capacity shared ESS remains completely inactive;
- ordinary/shared P/Q sign conventions remain consistent with nodal balance and exported results.

## B3.4 Numerical tests

Run the complete positive-bootstrap **network initialization** and compare against current production:

- primary failures;
- recovery attempts;
- persistent failures;
- iterations;
- zero/near-zero Jacobian-row counts;
- smallest singular values / rank estimates where available;
- suspicious curvature coefficients;
- objective;
- P/Q schedules;
- SOC trajectories.

Do not enter ADMM with a partially converted ESS formulation unless all required ESSO/state-mapping dependencies are implemented consistently.

Do not rewrite ESSO cycling degradation in B3.

If the network prototype is favorable, produce a precise follow-on end-to-end plan covering:

- ESSO per-cohort `pch/pdch`;
- aggregate P/Q coordination;
- active/cell-side throughput;
- cycling degradation;
- SoH;
- sensitivities;
- state mapping;
- result exports.

Calendar degradation remains out of scope.

---

# Isolation and commit discipline

B1, B2-R, and B3 are independent A/B experiments.

Each begins from the same accepted current production baseline.

Do not accumulate B1 into B2-R, B2-R into B3, or otherwise stack favorable diagnostic changes during P5.3-B.

Diagnostic scripts/tests may be added, but production source files must be restored to the accepted baseline at the end of each diagnostic branch unless a later planner instruction explicitly authorizes a production commit.

Do not combine production edits from this stage into existing accepted commits.

---

# P5.3-B required report

Produce:

`P5_3_B_REFORMULATION_REPORT.md`

with separate sections for:

1. B1 — exact reference-angle gauge;
2. B2-R — RES capability semantics/conditioning, or the documented data-model stop if no defensible converter rating exists;
3. B3 — active-power ESS structural prototype.

For each candidate give one of:

`PRODUCTIONIZE / CONTINUE TESTING / REJECT / DEFER`.

The report must include, where applicable:

- before/after NLP dimensions;
- primary/recovery/persistent failure counts;
- IPOPT iteration/runtime comparisons;
- Jacobian rank/singular-value/curvature evidence;
- physical-equivalence statement for exact changes;
- deliberate-feasible-set-change statement for physical reformulations;
- interface V/P/Q effects;
- objective/dispatch effects;
- stochastic-data implications kept separate from NLP formulation implications.

Answer explicitly:

- Should `f_ref = 0` be adopted?
- Does B2-R confirm that current RES capability conflates stochastic availability with converter MVA rating?
- Is there sufficient data to reformulate RES converter capability safely?
- If B2-R is blocked by missing rating data, what exact data-model extension is required?
- What should be done about copula upper-support overshoot and missing spatial correlation?
- Does the active-power ESS formulation remove the exact equality-rank deficiency?
- Does it materially improve bootstrap NLP robustness?
- Does `pch * pdch` complementarity remain numerically problematic?
- Can `sch/sdch` be removed safely from the network SMOPF?
- Is the P5.2 narrow-band workaround still necessary after B3?
- Which nonlinear family is the highest remaining numerical risk after the successful reformulations?

Finish exactly:

`P5.3-B COMPLETE — reformulation experiments ready for planner decision`

Then stop.

---

# Historical evidence retained for reference

The following summarizes earlier local-NLP work. It remains valid evidence but does not constrain P5.3 where the current-stage instructions above explicitly supersede it.

## Original frozen reference failure

Historical frozen model:

`data/SRP1/Results/FrozenSMOPF/frozen_DSO_node7_case33_2_2025_Winter_cycle10.pkl`

Metadata:

- DSO node 7;
- `case33_2`;
- 2025 Winter;
- ADMM cycle 10;
- warm start;
- `rho_v = 1.5`;
- `rho_pf = 2.25`;
- `rho_ess = 3.375`.

Repeated baseline:

- primary exact-Hessian MA97: `internalSolverError / Error in step computation`;
- limited-memory recovery: `maxIterations`.

Tightening only explicit `vmag` or `vmag_sqr` bounds did not help.

Removing non-reference explicit `vmag` variables/equalities from the active DSO NLP converted the decisive frozen failure into a clean primary exact-Hessian success. This led to the accepted production `vmag_nodes` refactor.

## P2/P3 progression

The production `vmag_nodes` refactor passed multiple preserved local failures and live seed-2026 operational smoke tests.

Residual failures then localized around network-side shared-ESS operation. Audits identified the shared-ESS squared-magnitude equality as a major candidate because all derivatives vanish near zero dispatch.

Removing the row entirely fixed failures but materially violated the relation and was rejected.

Equivalent in-place scaling experiments proved that row normalization itself could clear decisive DSO and TSO exact-Hessian failures while preserving the physical equality.

## P4 accepted scaling

Shared ESS:

`kappa * ((sch-sdch)^2 - pnet^2 - qnet^2) = 0`, `kappa = 1/S_rated`.

Ordinary ESS uses the analogous immutable build-time scale.

The shared scale is mutable on reused models and preserves KKT multiplier consistency when capacity changes.

## P5 planning integration

Zero-investment operational evaluation converged cleanly, but the tiny positive-bootstrap candidate exposed new cold-start failures before ADMM.

Scalar caps on shared-ESS `kappa` produced strongly non-monotone convergence and relocated failures.

## P5.2 narrow-band evidence

Replacing the hard shared-ESS zero-gradient equality by a tiny two-sided ranged row gave the first full positive-bootstrap initialization with zero persistent failures. At `epsilon_rel = 1e-4`, all 48 network solves succeeded on the primary exact-Hessian path with zero recovery.

However, the nominal physical band was below the solver's effective feasibility resolution at tiny capacity, especially in TSO models. This is why P5.3 now prioritizes a broader structural formulation audit rather than productionizing the narrow-band workaround.
