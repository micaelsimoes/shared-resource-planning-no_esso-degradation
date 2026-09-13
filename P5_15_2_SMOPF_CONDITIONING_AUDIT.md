# P5.15-2 — SMOPF Constraint-Family Conditioning Audit

**Stage:** Step 2 of `PLANNER_BRIEF_2026-09-13.md` (Advisor, read-only).
**Date:** 2026-09-13. **Repository:** `feature/derivative-free-planning`.
**Scope:** formulation-level inventory of the network SMOPF constraint families, their
nonlinearity, scaling, penalty coefficients and solver policy, for one DSO (`case33_2`,
node 7) and the TSO (`case9`).
**Method:** static inspection of source, parameter and network-data files, plus the
preserved P5.3-A / P5.7 / P5.12-G / P5.12-R / P5.12-T / P5.12-X evidence.
**Zero solves. No file was modified.**

> **Planner note (2026-09-13).** This document is the Advisor's deliverable, persisted
> verbatim. Two of its findings are independently confirmed by the Planner and should be
> read first: §0.1 (the brief misidentifies the cycle-21 fixture) and §4.3 items 2, 5 and 6
> (the multiplier push is derived from the primal push; `maxIterations` triggers no
> recovery; the recovery path changes three things at once). §0.1 was cross-checked against
> the Step 0 evidence: the Worker solved `DSO:case33_3|2025|Spring`, the correct fixture.

---

## 0. Corrections and limits that condition everything below

### 0.1 The brief's fixture identification is inconsistent with the preserved record

The Step 2 objective names "one DSO model (`case33_2`, node 7, the fixture that failed at
cycle 21)". The preserved record does not support that conjunction. Two distinct fixtures exist:

| fixture | network | node | year/day | cycle | outcome |
|---|---|---|---|---|---|
| cycle-21 failure (P5.12-A/B/ArmA/P/R/T) | **`case33_3`** | **9** | 2025 / Spring | 21 | `Maximum Number of Iterations Exceeded` at 3000 iterations (Arm A) |
| frozen comparator (P5.12-W/X) | `case33_2` | 7 | 2025 / Autumn | 7 | **`Optimal Solution Found`**, 91 iterations, objective 354.15536755956111 |

**Consequence.** `case33_2` node 7 is the *converged* comparator; `case33_3` node 9 is the
*failing* fixture. Because `case33_1/2/3` are built from the same 33-bus template with
different operational data, the formulation inventory below applies identically to both.

### 0.2 The committed ADMM parameters do not reproduce the preserved fixture

`SRP1_params.json:43-75` sets `num_max_iters = 25`, `adaptive_penalty = true`, and
`rho.v = rho.pf = rho.ess = 1.00`. The preserved P5.12 fixtures were produced with
`rho = 1.5 / 300 / 1.0` and adaptive rho **off**, set in a deep copy by the stabilized-oracle
harness, not in the parameters file. Any follow-up that rebuilds from `SRP1_params.json` will
not reproduce the fixture's objective Hessian. A provenance hazard, not a defect.

### 0.3 What could not be established

- **Per-variable values at the converged solution.** The fixtures are pickles; reading them
  requires execution, which this task did not have. Every "typical magnitude" below is either
  quoted from a preserved report or **[derived]** from bounds, parameters and network data.
- **Whether any given inequality is active at the converged solution.** Only two families have
  measured active-set evidence (`sg_capability`, `flex_energy_balance_p`, from P5.7 §3).

---

## 1. Model census

### 1.1 Configuration in force

3 years (2025/2030/2035, weight 5), 4 days, `NumInstants = 24`, `DiscountFactor = 0.02`,
**`NumMarketScenarios = 1`**, **`num_operation_scenarios = 1`** for all four networks. DSO
connection nodes 5 / 7 / 9. `obj_type = COST`, `transf_reg`/`es_reg`/`fl_reg`/`rg_curt` true,
**`l_curt = false`**, `enforce_vg = false`, `branch_limit_type = MIXED`,
`ess_model = shared_ess_model = BILINEAR_RELAXATION`. Slacks: voltage **on**; branch flow
**off**; flexibility day-balance **off**; ordinary- and shared-ESS day-balance **on**;
node-balance **off**.

### 1.2 Structural facts that switch whole families off

1. **There are no ordinary energy-storage devices.** `network.py:779` reads energy storages
   only `if 'energy_storages' in network_data`; neither case file contains that key. **Every
   `ess_*` family instantiates zero rows** despite `es_reg = true`. `PENALTY_ESS_BALANCE` and
   the ordinary-ESS branch of `PENALTY_ESS_COMPLEMENTARITY` are inert.
2. **`case9` has no transformer.** Under `BRANCH_LIMIT_MIXED`,
   `model.apparent_power_limited_branches` is **empty for the TSO**: no `pij/qij/pji/qji`, no
   `pij_def…qji_def`, no `branch_flow_limit_ji`. `model.r`/`r_sqr` appear in no TSO row.
3. **Parallel-circuit pre-processing collapses the DSO topology.** 82 branch records become
   **32 branches, index 31 the sole transformer** (`network.py:1958-1999`, `z_eq`, `rate_eq`).
   Cross-checked three ways: P5.12-G's 9166 columns + 102 equal-bound = 9268 declared;
   the variable census at 32 branches gives exactly 9268; P5.3-A independently reported 9268.
   **[derived]** Merged admittances reach `g ≈ 220`, `b ≈ −72.9`, `g²+b² ≈ 5.4e4`, reproducing
   P5.3-A's measured maximum `branch_flow_limit` row gradient of `1.32e5 ≈ 2.45·(g²+b²)`.

### 1.3 Per-model dimensions

| | DSO `case33_2` | TSO `case9` |
|---|---|---|
| nodes | 33 (1 REF, 32 PQ) | 9 (1 REF, 2 PV, 6 PQ) |
| branches after pre-processing | 32 (1 transformer) | 9 (0 transformers) |
| generators | 5 → 4 curtailable | 9 → 6 curtailable |
| loads | 32 (all `fl_reg`) | 3 physical + 3 synthetic `ADN_*` |
| ordinary ESS | 0 | 0 |
| shared ESS | 1 (reference node) | 3 (buses 5/7/9) |
| apparent-power-limited branches | 1 (transformer) | **0** |
| declared columns | 9268 (9166 after equal-bound removal) | ~2748 |
| `baseMVA` | 100.0 | 100.0 |

**Per-unit scale observation.** `baseMVA = 100` for a 12.66 kV ~4 MW feeder: power variables
live at `1e-3`–`1e-2` p.u. while `g, b` reach `220`/`−73`. `EQUALITY_TOLERANCE = 1e-5` is
applied **absolutely** to bounds on those power variables, so the same "tolerance" is 0.1 % of
a 0.01 p.u. quantity and 10 % of a 1e-4 p.u. quantity. Context, not a shortlist candidate —
changing `baseMVA` would move every published number.

---

## 2. Constraint-family inventory — DSO `case33_2`

### 2.1 Voltage

| family | rule | rows | type | magnitude / bounds | zero grad? | penalty |
|---|---|---|---|---|---|---|
| `voltage_mag_def` | `mch:420-421` | 24 | quadratic eq `vmag_sqr == vmag²` | `vmag ∈ [0.9,1.1]` | no (−2·vmag) | — |
| `voltage_mag_sqr_def` | `:416-417` | 792 | quadratic eq `vmag_sqr == e²+f²` | `e,f ∈ ±1.1501`; at REF after ADMM `e ∈ [0,1.1501]`, `f ∈ ±1e-5` | no, but the `f` partial ≈ 0 | — |
| `voltage_setpoint_cons` | `:424-429` | **0** | — | skipped (`enforce_vg=false`) | — | — |
| `voltage_magnitude_lower/upper_cons` | `:432-445` | 1584 | linear | `slack_v_sqr_*` forced to `[0,0]` at REF | no | `PENALTY_VOLTAGE_SQUARED = 5e4` |
| `voltage_product_real/imag_def` | `:448-459` | 1536 | **bilinear eq** | `vp_real ≈ 1`, `vp_imag ≈ Δθ ≈ 1e-3` **[derived]** | no (‖∇‖≈1.4) | — |
| `voltage_product_real_nonnegative` | `:462-463` | 768 | linear `vp_r ≥ 0` | margin ≈ 1 | no | — |
| `branch_angle_difference_*` | `:466-475` | **0** | — | **commented out at `network.py:404-405`** | — | — |

**Notes.** The branch angle-difference limits are **disabled in production**; the only angle
restriction in force is `vp_real ≥ 0`, i.e. `|Δθ| < 90°`. `BRANCH_DEFAULT_ANGLE_MIN/MAX = ∓30°`
is dead data. The manuscript must not claim angle limits are enforced.

**The reference-angle gauge is still only approximately fixed.** `f_bounds` returns
`(−1e-5, +1e-5)` for `BUS_REF` in both families. P5.3-B1 (exact `f_ref = 0`) was **never
productionized**. With `bound_push = bound_frac = 1e-5`, the initial push into this 2e-5-wide
box is `2e-10` **[derived]** — a variable whose barrier terms are governed by a box five orders
narrower than any other.

**The DSO reference real voltage `e` is no longer pinned** in ADMM
(`srp:3697-3699` frees it): P5.3-A's MEDIUM item is withdrawn for the ADMM path.

### 2.2 Generation and RES capability — the P5.3 HIGH-risk family

| family | rule | rows | type | evidence |
|---|---|---|---|---|
| `sg_capability` | `:500-504` | ≤96 | **convex quadratic (SOC)** `pg²+qg² ≤ sg_avail²`; `sg_avail` is a **Param** | P5.3-A: 3732 rows, ‖∇g‖ from **5.44e-05** to 0.949, **min margin exactly 0.00** at cold start |
| `gen_pf_upper/lower` | `:507-532` | ≤192 | linear `|qg| ≤ 0.75·pg` | at `pg=0` both PF rows and `pg ≥ 0` are active ⇒ **LICQ fails at that vertex** **[derived]** |
| `gen_pf_profile` | `:535-545` | **0** | — | never instantiated; all 144 curtailable generators have `power_factor_control = True` |

**Assessment.** Three facts must travel together: (1) the row conflates converter capability
with stochastic availability — `sg_avail = pg_available` because `q_available ≡ 0`, so reactive
capability collapses to zero at zero availability; (2) B2-R was **deferred, not resolved**; (3)
at the ADMM solution the row is active in **164 of 3732** rows but **3554 of 3732** at the
polish, because the curtailment penalty is divided by `effective_scale ≈ 1.05e5`. So it is a
**cold-start, polish and branch-selection** risk — the evidence does **not** support it as the
driver of a warm-started in-loop ADMM failure.

**Correction to the recorded blocker.** P5.3-B says the repository contains no defensible
converter rating; the network JSON carries per-generator `Pmax`/`Qmax` (WIND 40/40, PV 10/10)
and `qg_bounds` already uses `qmin/qmax`. Whether those denote an inverter nameplate is an
**author data-semantics decision**, not missing data. That decision is the entire blocker.

### 2.3 Flexible loads

| family | rows | type | evidence |
|---|---|---|---|
| `flex_energy_balance_p` | 32 | **two-sided linear range** `−1e-4 ≤ Σp_up − Σp_down ≤ +1e-4` (day-balance slack off) | ‖∇g‖=√48≈6.93; band 2e-4 p.u.·h = 0.02 MWh; P5.7: active in **286→321 of 1152**, with **69 entering / 34 leaving** between branches |
| `flex_energy_balance_q` | **0** | — | commented out at `network.py:427` |
| `flex_energy_balance_s` | **0** | quadratic | dead code; adds `flex_p`+`flex_q` **before** squaring — dimensionally wrong |

It is one of only **two** active-set-changing families. The row is well-scaled; the defect is
the two-sidedness, not the magnitude.

### 2.4 Shared energy storage

`eff_ch = 0.97`, `eff_dch = 0.96`, `max_pf = 0.90` ⇒ `tan(acos(0.9)) = 0.4843`.

| family | rule | rows | type | note |
|---|---|---|---|---|
| `sess_pnet_def` | `:876-877` | 24 | linear | `pnet, qnet` unbounded |
| `sess_pch_hat_link` / `pdch_hat_link` | `:805-820` | 48 | **bilinear eq** (`S_rated` is a `Var`) | P5.4-H1 rescaled; ‖∇g‖≈1.4 |
| `sess_converter_capability` | `:772-780` | 24 | **indefinite quadratic** `pnet²+qnet² ≤ S_rated²` | `∇²g = diag(2,2,−2)` because `S_rated` is a variable |
| `sess_active_sum_limit` | `:783-792` | 24 | linear | |
| `sess_phi_limit_lower/upper` | `:742-755` | 48 | linear `|qnet| ≤ 0.4843·(pch+pdch)` | ‖∇g‖≈1.21 |
| `sess_soc_def` | `:845-864` | 24 | linear + one bilinear term at `p=0` | |
| `sess_soc_limit_upper/lower` | `:795-802` | 48 | linear | |
| `sess_soc_final` | `:867-873` | 1 | linear eq with slacks | **slacks unbounded above**; `PENALTY_SHARED_ESS_BALANCE = 1e3` × `baseMVA` |
| `sess_comp` | `:823-842` | 24 | **bilinear inequality** `pch_hat·pdch_hat ≤ 1e-4` | P5.7: **8 → 0 of 1728 active**; **gradient exactly zero iff `pch_hat = pdch_hat = 0`**, i.e. every idle period; penalty at `:1726` is on the **physical** product — a scale mismatch with the row |
| `shared_energy_storage_s/e_sensitivities` | `:1059-1064` | 2 | linear eq pinning `Var` to `Param` | duals consumed at `network_data.py:2849-2874` |

**(a) `sess_snet_def` is gone** — P5.3-A's top HIGH item was removed by P5.4-A.

**(b) `sess_phi_limit_*` create an LICQ-degenerate vertex whenever the device idles.**
**[derived]** At `pch = pdch = 0` four inequalities are active in a 3-dimensional subspace, so
at most three gradients are independent and **LICQ fails**. MFCQ still holds, so multipliers are
bounded but **non-unique** — exactly the structure that makes interior-point dual iterates drift
and warm-started bound multipliers unrepresentative of the current solve.

**(c) The ESSO and the networks do not agree on the shared ESS's reactive feasible set.** The
ESSO imposes only the converter circle (`shared_energy_storage_data.py:606`) and **no**
power-factor rows; the networks additionally impose `|qnet| ≤ 0.4843·(pch+pdch)`. The ESSO's
reactive set is **strictly larger**. Not an infeasibility — `q = 0` is feasible for all three —
but an asymmetry the augmented Lagrangian must overcome, and a credible source of persistent
primal residual on the `ess`-`q` channel. `SRP1_params.json:38` sets the `ess` consensus
tolerances **ten times looser** than `v` and `pf`, consistent with a harder channel.

These rows also contradict the project's stated physics: `mch:773-777` says reactive power "no
longer participates in the stored battery energy", and the action plan's R2.10/R3.4 position is
that reactive provision is "a converter-capability matter only".

**(d) The capacity variables make three families nonlinear that need not be.** Because
`shared_es_s_rated`/`e_rated` are `Var`s pinned by equalities whose duals are the Benders
sensitivities: the circle is **indefinite** rather than convex SOC; the hat-links are bilinear;
and four more families carry a decision variable where a constant would do. P5.12-G measured
`δ_x ≠ 0` in **36 / 27 / 70** factorizations of 89 / 3000 / 151 — 40–46 % in the two converged
runs — and established that `δ_x` frequency **does not discriminate** failure from success. A
conditioning argument, not a failure-cause argument.

### 2.5 Node balance and branch flow

| family | rows | type | magnitude |
|---|---|---|---|
| `node_balance_p` / `_q` | 1584 | linear for non-transformer branches; **bilinear** for the transformer | `g` up to **220**, `b` to **−73**; transformer `g = 138.0`, `b = −70.3` **[derived]**; row value `O(1e-3)` p.u.; slacks **off** |
| `r_sqr_def` | 24 | quadratic eq | `r ∈ [0.83,1.17]`; **`r_sqr` has no `initialize`** — P5.3-A's remaining cold-start gap |
| `pij_def…qji_def` | 96 | **multilinear eq** | **largest Hessian entries in the DSO model:** `|∂²pij/∂vmag_sqr ∂r_sqr| = 138.0`, `|∂²/∂vp_imag ∂r| = 70.3` **[derived]** |
| `branch_flow_limit` | 768 | linear (non-transformer) / quadratic (transformer) | P5.3-A: **largest row gradient anywhere, 1.32e5**; strictly inactive, min margin `1.6e4·tol` |
| `branch_flow_limit_ji` | 24 | quadratic | 864/864 zero-gradient at cold start, strictly interior |

**A correction to a plausible-looking but wrong conclusion.** The `1.32e5` gradient against an
`O(1e-6)` body is **not** catastrophic cancellation: the body is a linear combination of three
`O(1)` variables, so absolute rounding error is `~3e-16`. IPOPT's `gradient-based` scaling with
`nlp_scaling_max_gradient = 100` already divides the row by `≈7.7e-4`, and P5.12-G found **no
scaled row below `1e-4`** and **no Arm-A-unique weak row**. Not shortlisted; reasoning recorded
so it is not re-derived.

### 2.6 Interface / consensus definitional rows

`expected_interface_vmag_def`, `expected_interface_pf_p_def`/`_q_def`,
`expected_shared_ess_p_def`/`_q_def` — all **linear equalities**. With 1×1 scenarios `π = 1`, so
each reduces to an identity between the expected variable and the single-scenario variable.

### 2.7 Objective composition and penalty coefficients

**`total_gen_cost` is identically zero for the DSO**: `generation_cost` includes only
`is_controllable()` generators and excludes `GEN_REFERENCE` in a distribution network, and
`case33_2`'s only controllable generator *is* the reference. The DSO prices no energy in its own
objective; import is priced only through the ADMM consensus.

`_prepare_distribution_objectives_for_admm` zeroes `penalty_ess_usage`;
`_add_dso_scenario_deviation_penalty` adds its term; `update_distribution_models_to_admm`
divides the whole objective by `effective_scale` and adds the AL terms **undivided**. P5.7
measured `effective_scale ∈ [9.41e4, 1.16e5]`, **median 1.05e5**.

**Effective coefficients inside the ADMM subproblem** (÷1.05e5) **[derived]**:

| term | raw | effective |
|---|---|---|
| generation cost | — | **0** (excluded for the DSO reference generator) |
| RES curtailment penalty | `1e0 × 100` | 9.5e-4 |
| ESS usage penalty | `1e-1` | **0** (zeroed for ADMM) |
| squared-voltage slack | `5e4` (**not** ×`baseMVA`) | **0.476** |
| node-balance / branch-flow / flexibility slacks | 1e6 / 1e3 / 1e3 × 100 | **inert** (slacks off) |
| shared-ESS complementarity penalty | `1e2 × 100` | 0.095 per p.u.² of `pch·pdch` (indefinite, ±0.095) |
| shared-ESS day-balance slack | `1e3 × 100` | 0.952; slack **unbounded above** |
| scenario deviation — voltage | `9e4` | 0.857 |
| scenario deviation — interface P/Q | `9e4 × 100` | **85.7** |
| scenario deviation — shared ESS | `1e4 × 100` | 9.52 |
| AL — voltage consensus | `ρ_v` undivided | **1.5** |
| AL — interface P/Q | `ρ_pf / rating²` | **300** |
| AL — shared-ESS | `ρ_ess / (2·max(S,0.10 MVA)/100)²` | **2657** at 0.97 MVA; **952** at 1.62 MVA; **2.5e5** at the floor |

**The dominant objective-scale fact.** At `effective_scale ≈ 1.05e5` the entire physical cost
layer sits at `1e-3 … 1` while the consensus layer sits at `1e0 … 1e4`. P5.7 measured that
re-solving each converged block with its objective multiplied back recovers **8,892,464**, i.e.
**96.5 %** of the whole ADMM-to-polish gap. The ADMM subproblems are not base-objective-optimal,
and the cause is arithmetic.

**A second, capacity-dependent fact.** The shared-ESS AL normalization `2·max(S, 0.10)` makes
the ESS consensus Hessian scale as `1/S²`, spanning **952 → 2.5e5** over the capacity range the
campaign explores — the same `O(1/S²)` pathology P5.3-A identified in the retired
`sess_snet_def`, relocated from a constraint row into the objective. **Referred to Step 3.**

**A third fact, exact and checkable.** With one market and one operation scenario the five
deviation residuals are **identically zero on the feasible set**, enforced by explicit
equalities. Value and gradient are zero at any feasible point; the only effects are 120 rank-one
PSD Hessian blocks per DSO model and a non-zero gradient at *infeasible* iterates.

---

## 3. Constraint-family inventory — TSO `case9`

1. **No transformer ⟹ no apparent-power branch**; `pij_def…qji_def`, `branch_flow_limit_ji`
   and `r_sqr_def` instantiate zero rows — **the largest-curvature family in the DSO model is
   absent from the TSO**.
2. **Branch coefficients two orders smaller** **[derived]**: `g²+b² ≈ 110 … 301` against
   `5.4e4`. Branches 1, 4, 7 have **`r = 0` exactly**, so `g = 0` and their active-power
   contribution depends only on `vp_imag`.
3. **Three shared ESS** ⇒ 72 rows per `sess_*` family.
4. **Loads are variables, not parameters**: `pc`/`qc` are `Var`s in `[pd ∓ 1e-5]` — 288
   variables in a `2e-5`-wide box. The three synthetic `ADN_*` loads carry the DSO consensus.
5. **Reference bus is not pinned in `e`** for the TSO; `f_ref ∈ ±1e-5` as in the DSO.
6. **The TSO condition-number claim was withdrawn.** Corrected reduced condition numbers are
   **≈8.98e4 (DSO)** and **≈1.42e3 (TSO)** — the *opposite* ordering to P5.3-A's table.
7. **The TSO objective differs**: `_prepare_transmission_objectives_for_admm` zeroes **both**
   `penalty_ess_usage` **and** `penalty_gen_curtailment`; the DSO version zeroes only the
   former. RES curtailment is penalized in the DSO and **free in the TSO**. The TSO also carries
   the **proximal regularization** block (enabled TSO, disabled DSO, `γ = 1.0`).
8. **The TSO's controllable generators are priced**; the DSO's are not.

---

## 4. Warm-start policy and complete IPOPT option inventory

### 4.1 Where the policy lives

`network.py:485-536`. Order: `fixed_variable_treatment` pinned → params-file `options` →
`option_overrides` → **the `from_warm_start` block at `:521-534`, which wins**. `from_warm_start`
is `True` for **every** ADMM cycle after a successful initialization.

### 4.2 Options in force

| option | DSO | TSO | IPOPT default |
|---|---|---|---|
| `tol` | 1e-5 | 1e-5 | 1e-8 |
| `acceptable_tol` | 1e-4 | 1e-4, **overridden to 1e-5 on warm starts** | 1e-6 |
| `acceptable_iter` | 5 | 5, **overridden to 0 on warm starts** | 15 |
| `compl_inf_tol` | not set | 5e-4 | 1e-4 |
| `linear_solver` | **ma97** | **ma97** | ma27 |
| `bound_push` / `bound_frac` / `slack_bound_*` | 1e-5 | 1e-6 | 1e-2 |
| `warm_start_init_point` | yes | yes | no |
| **all five `warm_start_*_push/frac`** | **1e-5** | **1e-6** | **1e-3** |
| `max_iter` | **not set** | **not set** | **3000** |
| `mu_strategy` | not set | not set | monotone |
| `nlp_scaling_method` | not set | not set | gradient-based, max_gradient 100 |
| `hessian_approximation` | exact (recovery: limited-memory) | exact (recovery: limited-memory) | exact |

Verified against P5.12-R §7 (effective options at both prepared boundaries), P5.12-T §4 (exactly
3000 iterations, matching unset `max_iter`) and P5.12-T §7D (four distinct `mu` values, matching
`monotone`).

### 4.3 Six policy observations

1. **All five warm-start pushes are 100× (DSO) or 1000× (TSO) tighter than IPOPT's defaults** —
   the network-level analogue of the ESSO's 1e-9, at a milder magnitude.
2. **`warm_start_mult_bound_push` is derived from `bound_push`.** Varying `bound_push` to affect
   the *primal* push silently changes the *multiplier* push by the same factor — a design hazard
   for any A/B that varies `bound_push`.
3. **The `dual` suffix is `IMPORT_EXPORT` and is never cleared on a nominally cold solve.**
   `replace_warm_start_suffix` touches only `ipopt_zL_in`/`zU_in`, and only when
   `from_warm_start`. If Pyomo's NL writer exports `dual`, **arm A4 would not be cold on the
   constraint-multiplier side**. Stated as a hypothesis; the discriminating check is in §7.2.
4. **The TSO's acceptable-point fallback is disabled on warm starts** (`acceptable_iter = 0`,
   `acceptable_tol = tol`), while a DSO may exit on an acceptable point. Undocumented asymmetry;
   state it wherever TSO and DSO failure rates are compared.
5. **`_is_recoverable_network_failure` fires only on `internalSolverError`.** A `maxIterations`
   exit — exactly what cycle 21 produced — triggers **no** recovery attempt.
6. **The recovery path changes three things at once** (cold start, cleared multiplier suffixes,
   `limited-memory` Hessian), so a recovery success does not identify which was responsible.

---

## 5. P5.3 HIGH-risk families: current status

| rank | family | status |
|---|---|---|
| HIGH | `sess_snet_def` | **removed** by P5.4-A |
| HIGH | TSO equality-Jacobian conditioning | **withdrawn**; corrected ordering is the opposite |
| HIGH | `sess_comp` | **rescaled** by P5.4-H1 to an `O(1)` row; P5.7 measured 8 → 0 of 1728 active. Residual: gradient exactly zero at every idle period; penalty on the physical product vs row on the dimensionless pair. **Downgraded.** |
| HIGH | RES `sg_capability` | **unchanged**; B2-R `DEFER`. Remains HIGH for cold starts, polish and branch selection; **not** demonstrated as an in-loop ADMM risk |
| MEDIUM | reference gauge `f_ref ∈ ±1e-5` | **unchanged**; B1 never productionized |
| MEDIUM | shared-ESS link / PF rows | links rescaled; **PF rows unchanged** and now carry the LICQ-degenerate idle vertex |
| MEDIUM | DSO reference `e` pinning | **withdrawn** for the ADMM path |
| LOW | `branch_flow_limit(_ji)` | unchanged; neutralized by IPOPT row scaling |
| LOW | `r` / `r_sqr` | unchanged; `r_sqr` still has no `initialize` |

---

## 6. Ranked shortlist of reformulation candidates

**No implementation is recommended.** Items 1 and 3 change the mathematical problem and require
substantially stronger justification than 2, 4 and 5.

### Candidate 1 — Detach shared-ESS reactive capability from instantaneous active throughput

**Rows.** `mch:742-747` and `:750-755` (`sess_phi_limits_lower/upper`), wired at
`network.py:450-451`. 24 rows per DSO, 72 per TSO. With `max_pf = 0.90` the pair is exactly
`|qnet| ≤ 0.4843·(pch + pdch)`.

**Proposed replacement.** Either (a) delete both rows and let `sess_converter_capability` carry
the reactive limit alone, or (b) replace by `|qnet| ≤ tan(acos(pf_min))·S_rated` — two linear
rows not involving `pch`/`pdch`.

**What changes mathematically.** **The feasible set changes**: the device may provide reactive
power while idling. It also removes the asymmetry with the ESSO subproblem.

**Expected conditioning effect.** Removes the LICQ-degenerate vertex at every idle period and
its non-unique multiplier set — the structure most likely to make imported bound multipliers
unrepresentative between cycles; removes a feasible-set asymmetry on the `ess`-`q` channel whose
tolerance is already ten times looser; aligns the code with its own stated physics.

**Cheapest test.** No solve for the first half: on the preserved fixtures, count periods where
`pch + pdch ≤ 1e-8·S_rated` and report whether both rows are at their bound and their duals.
*Large count with non-zero, unequal duals* ⇒ the degeneracy is live. *Non-trivial dispatch in
nearly every period* ⇒ the vertex is rarely visited and the candidate drops to a physics item.

### Candidate 2 — Make the shared-ESS capacity parameters `Param`s rather than `Var`s

**Rows.** `network.py:360-361`, `:457-458`, `mch:1059-1064`, affecting the circle, both
hat-links, `sess_active_sum_limit`, both SOC limits, `sess_soc_rule` at `p=0` and
`sess_soc_final_rule`.

**Proposed replacement.** Use the existing mutable `Param`s directly and delete the two `Var`s
and the two pinning rows. `configure_shared_ess_operational_state` already sets both to the same
value, so the numerical content is unchanged.

**What changes mathematically.** **The solution set is unchanged** — capacity is already pinned
by an equality. What is lost is the **Benders capacity sensitivity channel**
(`network_data.py:2839-2874`). **Conditional on the outer-method decision**: if the author
retains local cuts, this candidate must be rejected.

**Expected conditioning effect.** The circle becomes **convex SOC** instead of indefinite; the
hat-links become **linear**; four more families become linear. Plausibly fewer regularized
factorizations — but P5.12-G established `δ_x` frequency does **not** discriminate failure from
success, so the expected benefit is cleaner KKT systems, **not** a fix for cycle 21.

**Cheapest test.** No-solve: confirm the circle is the only indefinite family and cross-reference
P5.12-G's `δ_x` event iterations. Solve: re-solve the cycle-21 fixture with the two capacity
variables Pyomo-`fix()`ed — under `fixed_variable_treatment = make_parameter` IPOPT removes them,
reproducing the proposed formulation **without editing production** — and compare iteration
count, `δ_x` events and termination.

### Candidate 3 — Separate RES converter capability from stochastic availability (unblock B2-R)

**Rows.** `mch:493-497`, `:500-504`, `:167-174`, `:238-242`, wired at `network.py:419`.

**Proposed replacement.** Two rows: `pg ≤ P_available` (already a variable bound) and
`pg² + qg² ≤ S_converter²` with `S_converter` a time-invariant `Param`.

**What changes mathematically.** **The feasible set changes**: reactive capability no longer
collapses with stochastic active availability.

**Correction to the recorded blocker.** The network JSON carries `Pmax`/`Qmax` per generator.
Whether they denote an inverter nameplate is an **author data-semantics decision**, not a data
gap — and that single decision unblocks the candidate.

**Expected conditioning effect.** Removes the family with zero cold-start margin on 3732 rows
and gradients to `5.44e-5`. Benefit concentrated in **initialization and polish** (3554/3732
active) rather than in the loop (164/3732). Ranked third for that reason.

**Cheapest test.** Classify `sg_capability` rows by `pg_available`; confirm the 17 realized
values in `(1e-5, 1e-4]` p.u. carry the `5.44e-5` gradients; then re-solve *one* initialization
block with the two-row form at `S_converter = Pmax/baseMVA`.

### Candidate 4 — Replace the flexibility day-balance band by an exact equality

**Rows.** `mch:549-567`, executing branch `:565`:
`pe.inequality(-SMALL_TOLERANCE, p_up - p_down, SMALL_TOLERANCE)`. 32 rows per DSO, 6 per TSO;
band `2e-4` p.u.·h = 0.02 MWh per load per day.

**Proposed replacement.** `p_up == p_down`, or the already-implemented slacked-equality branch
with `slacks.flexibility.day_balance = true`.

**What changes mathematically.** The feasible set **shrinks** by the band (equality form). The
band permits a small net creation or destruction of daily flexible energy with no stated
justification.

**Expected conditioning effect.** Removes a two-sided range **active on 286–321 of 1152 rows**
that **enters and leaves the active set between branches (69 in / 34 out)** — one of only two
families doing so. An equality is always active, eliminating the switching. Gain is in
**active-set stability and branch reproducibility**, not row scaling.

**Cheapest test.** No-solve: count rows within `1e-7` of either side of the band and report
duals. Then a one-solve A/B with the rule replaced by an equality in a harness.

### Candidate 5 — Guard the scenario-deviation penalties off when there is one scenario

**Rows.** `srp:2811-2834` and `:2785-2808`, called at `:3018`, `:3125`, `:5861`, `:2945`,
`:3271`. `PENALTY_SCENARIO_DEVIATION = 9e4`, `PENALTY_SHARED_ESS_SCENARIO_DEVIATION = 1e4`.

**Proposed replacement.** Skip the block when `len(scenarios_market) * len(scenarios_operation) == 1`.

**What changes mathematically.** **Nothing.** The five deviation residuals are identically zero
by explicit equalities already in the model; value **and** gradient are exactly zero at every
feasible point. The only provable no-op on the shortlist.

**Expected conditioning effect — stated with its uncertainty.** 120 rank-one PSD blocks per DSO
model at effective `2×0.857`, `2×85.7`, `2×9.52`. Because they have the form `c·aaᵀ` with `a` a
Jacobian row, they are **AL-like and may improve KKT inertia rather than degrade it**, and they
leave the reduced Hessian unchanged. **No net benefit is claimed.** What is claimed: they inflate
`‖W‖` by up to 171, contribute a non-zero gradient at infeasible iterates, and are unambiguously
redundant. Ranked fifth because the test is nearly free but the expected sign is uncertain.

**Cheapest test.** One solve of the cycle-21 fixture with the deviation term removed from
`admm_objective`. Objective and primal solution **must** be identical to within tolerance; only
the path may differ.

### Considered and deliberately not shortlisted

| item | why not |
|---|---|
| `branch_flow_limit` cancellation-free form | Cancellation is benign (`3e-16` against `1e-6`); IPOPT scaling already divides by `7.7e-4`; P5.12-G found no scaled row below `1e-4`; rows strictly inactive. Rewriting trades a large-coefficient *linear* row for a small-gradient row with `O(1e5)` curvature — sign not established. |
| Exact reference gauge `f_ref = 0` (B1) | Real and unresolved, but P5.3-A's supporting evidence was **withdrawn as a derivative-audit artifact**. No measured support. Cheap no-solve check: report `|f_ref|` at the fixtures. |
| Shared-ESS AL normalization `2·max(S, 0.10)` | Real and quantified (`O(1/S²)`, 952 → 2.5e5). **Referred to Step 3**, which owns ADMM penalties; changing it here would confound that baseline. |
| `baseMVA = 100` for a ~4 MW feeder | Largest systemic scaling mismatch, but changing it moves every published number. Belongs to a formulation-rewrite decision. |
| `PENALTY_ESS_COMPLEMENTARITY` on the physical product vs the dimensionless row | Genuine inconsistency; effective coefficient `0.095` on an `O(1e-4)` product — negligible. Code hygiene. |
| Reinstating the disabled angle limits | A formulation **gap**, not a conditioning item. Author decision. |
| `r_sqr` has no `initialize` | One-line, zero-risk, affects only the transformer's 24 entries at cold start. |
| `ess_complementarity_penalties_rule` passes `params` twice | Harmless — the formal argument is immediately shadowed. Record only. |
| `slack_penalties` calls `sum(...)` on a scalar | Would raise at build time; unreachable because the slack is disabled. Latent bug; record only. |

---

## 7. Open questions this audit could not close

1. **Which rows are active, and with what multipliers, at the preserved converged solutions.**
   Everything marked "unmeasured" is blocked on this; the P5.12-R captures contain the state and
   extracting it requires one no-solve harness run.
2. **Whether the `dual` suffix is exported on solves that are nominally cold.** This determines
   whether Step 0's arm A4 tests what it claims. P5.12-R preserved `.nl` exports at both
   boundaries, so it is answerable by reading a preserved file: **if the `.nl` suffix section
   contains constraint-multiplier entries, arm A4 must additionally clear `model.dual`; if not,
   A4 stands as written.** No solve, no rebuild.
3. **The per-term magnitude of the DSO ADMM objective at the fixture** — the split between the
   dual-linear terms `λᵀr` and the quadratic AL terms.
4. **Whether P5.7's `effective_scale ≈ 1.05e5` still holds under the committed
   `SRP1_params.json`**, which differs from the stabilized-oracle configuration (§0.2).

---

## 8. Summary

After P5.4 the network SMOPF is substantially cleaner than P5.3-A described: its top HIGH item
(`sess_snet_def`) is gone, its second (TSO Jacobian conditioning) is withdrawn, and its third
(`sess_comp`) has been rescaled from `1e-12` to `1e-4`. What remains, in the units IPOPT sees:

- one **indefinite quadratic** family and two **bilinear** families, all three caused by two
  capacity variables that are pinned to constants and exist only to carry Benders duals;
- one **LICQ-degenerate vertex** at every idle shared-ESS period, from power-factor rows that
  tie reactive capability to instantaneous active throughput — rows that also disagree with the
  ESSO's own reactive feasible set and with the project's stated physics;
- one **unresolved RES family** whose reformulation is blocked by a one-sentence author decision
  about what `Pmax`/`Qmax` mean, and whose measured impact is concentrated outside the ADMM loop;
- one **two-sided linear band** that is one of only two active-set-changing families;
- one **provably redundant penalty family** under the single-scenario configuration;
- and, dominating all of them, an **`effective_scale ≈ 1.05e5`** that divides the physical cost
  layer while leaving the consensus layer undivided — already quantified as 96.5 % of the
  ADMM-to-polish gap — together with a shared-ESS consensus normalization whose curvature scales
  as `1/S²` across the capacity range the campaign is about to explore.

The solver policy is 100× (DSO) to 1000× (TSO) tighter than IPOPT's defaults on all five
warm-start pushes, sets no `max_iter`, derives the multiplier push from the primal push, silently
disables the TSO's acceptable-point fallback on warm starts, and attempts recovery only on
`internalSolverError` — so the cycle-21 `maxIterations` exit received no retry at all.
