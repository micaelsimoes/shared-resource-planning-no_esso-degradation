# P5.5-A — Convex planning architecture: exact convexity and controllable-resource audit

**Audit and design only. No convex model, planning loop, ADMM change, solver
install or production edit was made.** Every finding below is read from the
SRP1 data files and the production model-construction code, not assumed.

Evidence: `p55a_model_inventory.py`, `data/SRP1/Results/P55A/p55a_inventory.json`
(a live inventory of the built TSO, DSO and ESSO models).

## Development branch

| Item | Value |
|---|---|
| Source branch | `admm_residual_balancing_tests` |
| Source commit | `e11b0b05` ("Update governing documents with the planner's P5.4-D4 scope") |
| New branch | `feature/convex-planning` |
| Worktree at start | 2 modified tracked files (`LOCAL_NLP_STABILITY_PLAN.md`, `REVISION_CONTEXT.md` — the planner's P5.5 scope update); no untracked production files |

**The branch already existed at `e11b0b05` when this stage began — I did not run
`git switch -c`.** `e11b0b05` is the tip of `admm_residual_balancing_tests` and a
descendant of the last D4 commit, so the required starting point is satisfied.
No rebase, reset or history rewrite was performed, and
`admm_residual_balancing_tests` was not modified.

---

# Architecture summary

### 1. Exact SRP1 controllable-resource inventory

| Resource | Framework | Instantiated in SRP1 | Controllable |
|---|---|---|---|
| Conventional generator P | yes | **yes** (TSO only: 3 CONV; DSO: 1 REF each) | **yes** |
| Conventional generator Q | yes | **yes** | **yes** |
| RES active generation | yes | **yes** (TSO 6, DSO 4 per network) | **yes**, bounded by per-scenario availability |
| RES curtailment | yes | **yes** (`rg_curt: true`) | **yes**, implicit via `pg ≤ pg_avail` |
| RES reactive power | yes | **yes** | **yes** |
| RES power-factor control | yes | **yes** (`pf_control=1` on every RES) | **yes**, cone between `pf_min`/`pf_max` |
| Active load flexibility | yes | **yes** (`fl_reg: true`; TSO 3/3 loads, DSO 32/32) | **yes** |
| Reactive load flexibility | yes | **yes** | **yes** |
| Load curtailment | yes | **`l_curt: false`** | **no — disabled** |
| Ordinary ESS | yes | **NO** — no `energy_storages` key in any SRP1 network file; all `ess_*` families build **0 rows** | absent |
| Shared network ESS | yes | **yes** (TSO 3, DSO 1 each) | **yes** |
| Shared ESS `pch`/`pdch` | yes | **yes** | **yes** |
| Shared ESS `qnet` | yes | **yes** | **yes** |
| Shared ESS SOC | yes | **yes** | state variable |
| Voltage magnitude control | — | `vmag_sqr` bounded per node; `enforce_vg: false` | **PV setpoints NOT enforced** |
| **Transformer tap ratio (OLTC)** | yes | **yes — 1 per DSO** | **yes, continuous** |
| Phase-shifting transformers | **no** — no phase-shift field exists in `Branch` or the reader | absent | absent |
| Capacitor banks / switched shunts | node `Gs`/`Bs` only, fixed | **all zero in SRP1** | **absent** |
| Branch switching | `status` is build-time data | fixed | **absent** |
| Operational slacks | yes | voltage slacks **on**; branch-flow, node-balance, flexibility-balance, ESS-complementarity slacks **off**; ESS/shared-ESS day-balance slacks **on** | — |
| TSO–DSO P/Q exchange | yes | **yes** | ADMM consensus |

### 2. OLTC / capacitor-bank conclusion

```
SRP1 HAS controllable OLTCs
```

One per distribution network (`case33_1`, `case33_2`, `case33_3`): branch_id 1,
bus 1 → bus 2, `ratio = 1.0`, `vmag_reg = 1`, and `transf_reg: true` in every
params file. `model.r` is a **Var** bounded by
`[TRANSFORMER_MINIMUM_RATIO, TRANSFORMER_MAXIMUM_RATIO] = [0.83, 1.17]`,
**continuous, indexed per scenario and period** (no discrete tap logic exists
anywhere in the code). `case9` (TSO) has **no** transformers, so `model.r` is
pinned to `1.0 ± 1e-5` on all its branches.

Capacitor banks / switched shunts:

```
SRP1 HAS NO controllable capacitor banks or switched shunts;
all node Gs and Bs are exactly zero and no switching variable exists.
```

Only fixed shunts are representable (`node.gs`, `node.bs`), and in SRP1 they are
zero on every node of every case and year. (`case18_*` does carry nonzero `Bs`,
but SRP1 does not use `case18_*`.)

### 3. Exact list of nonconvex constraint families

| Family | Why nonconvex |
|---|---|
| `voltage_mag_sqr_def` | `Wii = e² + f²` — nonconvex **equality** |
| `voltage_product_real_def` | `WijR = e_i e_j + f_i f_j` — bilinear equality |
| `voltage_product_imag_def` | `WijI = f_i e_j − e_i f_j` — bilinear equality |
| `voltage_mag_def` | `Wii = vmag²` — nonconvex equality (ADMM-only variable) |
| `r_sqr_def` | `r_sqr = r²` — nonconvex equality (**DSO only**) |
| `node_balance_p/q`, `pij/qij/pji/qji_def`, `branch_flow_limit` | contain `r·WijR`, `r·WijI`, `r_sqr·Wii` — **bilinear in tap × voltage** (**DSO only**) |
| `sess_comp` / `ess_comp` | `pch_hat · pdch_hat ≤ eps` — bilinear (nonconvex) |
| `sess_pch_hat_link` / `sess_pdch_hat_link` | `pch − S·pch_hat = 0` — bilinear when `S` is a planning variable |
| ESSO `energy_storage_capacity_degradation` | `degradation · (2·cl_nom·E_rated) = throughput` — bilinear |
| ESSO cumulative SoH | `soh_cumul = prev_soh · soh^(365·num_years)` — variable raised to a power, times a variable |
| ESSO `available_e_capacity_unit` | `E_available = E_rated · soh_cumul` — bilinear |
| ESSO `energy_storage_normalization` | `pch − S_cohort·pch_hat = 0` — bilinear |
| ESSO `energy_storage_operation_agg` (capability) | `pnet² + qnet² ≤ es_s_rated²` — **nonconvex** because `S` is a variable on the RHS |

Everything else audited is affine, convex quadratic or SOC-representable.

### 4. Recommended formulation

| Network | Topology (built model) | Recommendation |
|---|---|---|
| **DSO** `case33_*` | 33 buses, 32 branches, **1 component, cyclomatic 0 — RADIAL** | **SOC** |
| **TSO** `case9` | 9 buses, 9 branches, 1 component, **cyclomatic 1 — MESHED** | **SOC + QC strengthening** |

Justification in A4.

### 5. Resources retained exactly in the convex LB model

Conventional P/Q, RES P/Q with availability and PF cones, active and reactive
load flexibility with day balance, shared-ESS `pch`/`pdch`/`qnet`/SOC with the
active-energy recursion, converter capability, active-sum envelope, voltage
limits, branch thermal limits, voltage slacks, and TSO–DSO interface P/Q.
**No controllable resource is dropped.** No DC-OPF or LinDistFlow approximation
is proposed.

### 6. Resources / relationships relaxed

1. Rank-one voltage coupling → SOC inequality (outer relaxation).
2. Charge/discharge complementarity → **dropped** (enlarges the feasible set).
3. H1 `pch_hat`/`pdch_hat` and their bilinear links → **removed** (they exist
   only to express complementarity).
4. Tap–voltage products → McCormick/SOC envelopes over `r ∈ [0.83, 1.17]`.
5. ESSO degradation chain → relaxed to `E_available ≤ E_rated`.
6. `vmag` and `voltage_mag_def` → removed; coordination uses `Wii` directly.

### 7. Master-side nonconvexities

**None.** The Benders master (`_build_master_problem`) is a pure **LP**:
continuous nonnegative `es_s_investment` / `es_e_investment`, linear capacity
accumulation over calendar life, linear max-capacity, linear E/S ratio bounds,
a linear budget constraint, a linear investment-cost objective plus `alpha`, and
linear cuts. **No integer or binary variables and no bilinear terms.** The master
is globally solvable today.

### 8. Proposed LB/UB algorithm

`R(x) ≤ Q*_AC(x) ≤ Q^feas_AC(x)`; master LB from the convex cuts, UB from the
best validated nonlinear incumbent. Full specification in A9.

### 9. Solver recommendation

**Blocker: no SOCP-capable solver is currently available.** Available: IPOPT
(local NLP), CBC (MILP/LP), CLP (LP). No Mosek/Gurobi/CPLEX/Xpress/SCIP/HiGHS,
and no `cvxpy`/`ecos`/`scs`/`clarabel`/`osqp`. Pyomo 6.9.2 provides conic
*modelling* (`pyomo.core.kernel.conic`) but nothing to consume it. See A11.

### 10. Blockers before implementation

1. **No conic solver** for a certified global LB with reliable duals.
2. **OLTC tap–voltage bilinearity** needs an explicit envelope design and
   numerical validation of relaxation tightness (DSO only).
3. **ESSO degradation** relaxation changes what the model represents and needs
   planner sign-off on bound direction.
4. **Non-pure operational oracle** (A12) — shared network data is mutated in
   place across calls.

---

# A1 — Exact current SRP1 control set

Cases actually used by `SRP1.json`: TSO `case9`; DSO `case33_1` (node 5),
`case33_2` (node 7), `case33_3` (node 9). 3 years (2025/2030/2035, weight 5
each), 4 representative days, 24 instants, 1 market and 1 operation scenario,
seed 2026.

## Live counts (built models, `p55a_inventory.json`)

| | TSO `case9` | DSO `case33_1` |
|---|---|---|
| Nodes | 9 | 33 |
| Branches after pre-processing | 9 | 32 |
| Generators | 9 (3 CONV, 3 WIND, 3 PV) | 5 (1 REF, 1 WIND, 3 PV) |
| Loads / flexible loads | 3 / 3 | 32 / 32 |
| Shared ESS | 3 | 1 |
| **Ordinary ESS** | **0** | **0** |
| Transformers | **0** | **1** |
| Branches with line charging | 6 | 0 |
| Nodes with fixed shunt | 0 | 0 |

Params identical across all four cases: `obj_type=COST`, `transf_reg=true`,
`es_reg=true`, `fl_reg=true`, `rg_curt=true`, `l_curt=false`,
`enforce_vg=false`, `branch_limit_type=MIXED`,
`ess_model=shared_ess_model=BILINEAR_RELAXATION`; voltage slacks on, branch-flow
and node-balance slacks off, ESS/shared-ESS day-balance slacks on,
complementarity slacks off.

## Item-by-item

| Resource | supported | instantiated | controllable | fixed | absent |
|---|---|---|---|---|---|
| Conventional generator P | ✓ | ✓ | ✓ (`pg ∈ [pmin, pmax]`) | | |
| Conventional generator Q | ✓ | ✓ | ✓ (`qg ∈ [qmin, qmax]`) | | |
| RES active generation | ✓ | ✓ | ✓ (`pg ∈ [0, pg_avail(s_o,p)]`) | | |
| RES curtailment | ✓ | ✓ | ✓ (implicit — the upper bound is availability) | | |
| RES reactive power | ✓ | ✓ | ✓ | | |
| RES PF control | ✓ | ✓ (all RES `pf_control=1`) | ✓ (`gen_pf_lower/upper`, 69 rows DSO) | | |
| RES fixed-PF profile | ✓ | — | | | **`gen_pf_profile` = 0 rows** |
| Active load flexibility | ✓ | ✓ | ✓ (`flex_p_up/down`) | | |
| Reactive load flexibility | ✓ | ✓ | ✓ (`flex_q_up/down`) | | |
| Load curtailment | ✓ | | | | **`l_curt=false`** |
| Ordinary ESS (all families) | ✓ | | | | **absent — 0 rows** |
| Shared ESS P/Q/SOC | ✓ | ✓ | ✓ | | |
| PV voltage setpoint | ✓ | | | **`enforce_vg=false` → `voltage_setpoint_cons` = 0 rows** | |
| **OLTC tap** | ✓ | ✓ (DSO only) | **✓ continuous, per period** | pinned to 1.0 on TSO | |
| Phase shifter | | | | | **not in the data model at all** |
| Capacitor bank / switched shunt | fixed `Gs`/`Bs` only | | | | **all zero** |
| Branch switching | build-time `status` | | | fixed | |
| Voltage slacks | ✓ | ✓ | ✓ | | |
| TSO–DSO P/Q exchange | ✓ | ✓ | ✓ (ADMM) | | |

## OLTC resolution

| From | To | Nominal tap | Phase shift | Tap variable or parameter | Tap control enabled | Discrete tap logic |
|---|---|---|---|---|---|---|
| 1 | 2 (`case33_1`) | 1.0 | **none — no phase-shift field exists** | **Var** `model.r[b,s_m,s_o,p]` | **yes** (`transf_reg` ∧ `vmag_reg`) | **no** |
| 1 | 2 (`case33_2`) | 1.0 | none | Var | yes | no |
| 1 | 2 (`case33_3`) | 1.0 | none | Var | yes | no |
| — | — | `case9`: no transformer branches | — | `r` pinned to `1.0 ± 1e-5` | n/a | n/a |

```
SRP1 HAS controllable OLTCs
```

Transformers are read **only** from a `'transformers'` key
(`network.py:745-768`); `case33_*` carry exactly one such entry per year file,
`case9` carries none. `transformer_ratio_bounds` returns `[0.83, 1.17]` when
`params.transf_reg and branch.vmag_reg`, else `ratio ± 1e-5`.

*Correction of record:* an intermediate check in this stage looked at
`case9_2025.json` only and concluded no transformers existed anywhere. A full
scan of all 71 SRP1 network files corrected that; the conclusion above is from
the full scan and from the live built model.

Bus 1 of each `case33_*` carries the reference generator at 345 kV and is
connected to the feeder **only through this transformer** — which is why the
`lines` array starts at `branch_id` 2 and why bus 1 has no line. The shared ESS
of each DSO sits at this reference bus.

Also note the raw `lines` arrays contain 81 entries that are parallel duplicates
of 31 distinct node pairs; `_pre_process_network` merges parallel branches into
one equivalent branch, so the built DSO model has 32 branches (31 lines + 1
transformer).

---

# A2 — Row-by-row operational convexity map

Classes: **L** linear/affine · **CQ** convex quadratic · **SOC**
second-order-cone representable · **NCQ** nonconvex quadratic · **BL** bilinear ·
**NC** other nonconvex. "Enlarges?" refers to the proposed convex form.

## AC network

| Component | Expression | Variables | Applies | Class | Unchanged? | Proposed convex form | Enlarges? |
|---|---|---|---|---|---|---|---|
| `voltage_mag_sqr_def` | `Wii = e² + f²` | `e, f, vmag_sqr` | both | **NCQ** (equality) | no | drop `e,f`; `Wii` becomes primitive | yes (with SOC below) |
| `voltage_product_real_def` | `WijR = e_i e_j + f_i f_j` | `e, f` | both | **BL** | no | `WijR` primitive | yes |
| `voltage_product_imag_def` | `WijI = f_i e_j − e_i f_j` | `e, f` | both | **BL** | no | `WijI` primitive | yes |
| *(implied rank-one)* | `WijR² + WijI² = Wii·Wjj` | W | both | **NCQ** | no | `WijR² + WijI² ≤ Wii·Wjj` (rotated SOC) | **yes — the relaxation** |
| `voltage_mag_def` | `Wii = vmag²` | `vmag` | both | **NCQ** | no | **remove**; `vmag` exists only for ADMM consensus | yes |
| `voltage_magnitude_lower/upper_cons` | `Wii + slack ≥ v_min²`, `Wii − slack ≤ v_max²` | `Wii`, slacks | both | **L** | **yes** | unchanged | no |
| `voltage_setpoint_cons` | `Wii = vg²` | | both | L | **0 rows** (`enforce_vg=false`) | — | — |
| `voltage_product_real_nonnegative` | `WijR ≥ 0` | | both | **L** | **yes** | unchanged | no |
| Reference gauge | `f_ref ∈ ±1e-5`; DSO: `e_ref = vg ± 1e-4` | `e, f` | both | **L** (bounds) | — | in W-space: fix `W_ref,ref = vg²` (DSO); angle reference is implicit | no |
| Branch angle-difference limits | `WijI ⋛ tan(θ)·WijR` | | both | **L** | **not instantiated** (commented out at `network.py:403-404`) | available as free QC strengthening | no |
| `node_balance_p` | `Pg = Pd + gs·Wii + Σ_b [ g·Wii·r² − r(g·WijR + b·WijI) ]` | W, `r`, `r_sqr` | both | **L in W if `r` fixed**; **BL** with variable `r` | TSO: yes. DSO: no | McCormick envelopes on `r·WijR`, `r·WijI`, `r_sqr·Wii` | DSO: yes |
| `node_balance_q` | `Qg = Qd − bs·Wii + Σ_b [ −(b + b_sh/2)·Wii·r² + r(b·WijR − g·WijI) ]` | as above | both | same | same | same | same |
| `pij/qij/pji/qji_def` | `p = g·W_term·r² − r(g·WijR + b·WijI)`; `q = −(b + b_sh/2)·W_term·r² + r(b·WijR − g·WijI)` | as above | both (only on apparent-power-limited branches: 24 rows = the 1 DSO transformer × 24 periods) | same | same | same | same |
| `branch_flow_limit` (current, non-transformer) | `(g²+b²)(Wii + r²·Wjj − 2r·WijR) ≤ rate²` | W | both | **L in W** (`r ≡ 1` on lines) | **yes** | unchanged | no |
| `branch_flow_limit`/`_ji` (apparent, transformer) | `pij² + qij² ≤ rate²` | `pij, qij` | DSO | **CQ / SOC** | **yes** | unchanged | no |
| `r_sqr_def` | `r_sqr = r²` | `r` | DSO | **NCQ** (equality) | no | `r_sqr ≥ r²` (convex) + secant upper envelope | yes |
| Fixed shunts | `gs·Wii`, `−bs·Wii` | `Wii` | both | **L** | **yes** (zero in SRP1) | unchanged | no |
| Controllable shunts | — | — | — | — | **absent** | — | — |
| Interface voltage | `vmag` at interface nodes | `vmag` | both | see A10 | no | couple on `Wii` | no |

## Generation

| Component | Expression | Class | Unchanged? | Notes |
|---|---|---|---|---|
| Conventional P bounds | `pg ∈ [pmin, pmax]` | **L** | **yes** | |
| Conventional Q bounds | `qg ∈ [qmin, qmax]` | **L** | **yes** | |
| RES P bounds | `pg ∈ [0, pg_avail(s_o,p)]` | **L** | **yes** | availability is data |
| `sg_capability` | `pg² + qg² ≤ sg_avail²` | **CQ / SOC** | **yes** | RHS is a `Param` |
| PV setpoints | — | — | **0 rows** | `enforce_vg=false` |
| Generation cost | linear in `pg` (COST objective) | **L** | **yes** | |
| Market terms | linear, price is data | **L** | **yes** | |

## RES

| Component | Expression | Class | Unchanged? |
|---|---|---|---|
| Active availability | `pg ≤ pg_avail` (bound) | **L** | **yes** |
| Curtailment | implicit in the bound | **L** | **yes** |
| Reactive capability | `sg_capability` | **SOC** | **yes** |
| `gen_pf_lower/upper` | `qg ≥ tan_lo·pg`, `qg ≤ tan_hi·pg` | **L** | **yes** |
| `gen_pf_profile` | `q_avail·pg = p_avail·qg` | **L** (coefficients are data) | **0 rows in SRP1** |
| Scenario-dependent limits | bounds per `(s_o, p)` | **L** | **yes** |

## Demand / flexibility

| Component | Expression | Class | Unchanged? |
|---|---|---|---|
| Nodal demand aggregation `pc_node`/`qc_node` | affine sum of `pc`, `flex_p_up/down` | **L** | **yes** |
| Flexibility bounds | box bounds from data | **L** | **yes** |
| `flex_energy_balance_p` | `Σ_p flex_p_up = Σ_p flex_p_down` | **L** | **yes** |
| Reactive flexibility | box bounds | **L** | **yes** |
| Load curtailment | — | — | **disabled** |

## Ordinary ESS

**0 rows in SRP1.** The families exist and would be classified exactly as the
shared-ESS ones below, with the simplification that `S_rated` is fixed network
data rather than a decision variable — which makes `pch + pdch ≤ S`,
`pnet² + qnet² ≤ S²` and the H1 links all **convex/affine as written**. Only
`ess_comp` (`pch_hat·pdch_hat ≤ eps`) is nonconvex. Nothing needs to be designed
for SRP1 today, but a case that instantiates ordinary ESS would be immediately
convex-ready apart from complementarity.

## Shared network ESS (post-D2-P production formulation)

| Component | Expression | Class | Unchanged? | Proposed | Enlarges? |
|---|---|---|---|---|---|
| `sess_pnet_def` | `pnet = pch − pdch` | **L** | **yes** | unchanged | no |
| `sess_active_sum_limit` | `pch + pdch ≤ S_rated` | **L** (jointly, `S` variable) | **yes** | unchanged | no |
| `sess_converter_capability` | `pnet² + qnet² ≤ S_rated²` | **NCQ** as written (`S` variable on RHS) | no | **rotated SOC**: `pnet² + qnet² ≤ S_rated·S̄` with `S̄ ≥ S_rated`, or directly `‖(pnet,qnet)‖₂ ≤ S_rated` | **no — this is exact** |
| `sess_phi_limit_lower/upper` | `qnet ⋛ tan·pch − tan·pdch` | **L** | **yes** | unchanged | no |
| `sess_soc_def` | `SOC_t = SOC_{t−1} + η_ch·pch·dt − pdch·dt/η_dch` | **L** | **yes** | unchanged | no |
| `sess_soc_limit_lower/upper` | `SOC ≥ 0.1·E_rated`, `SOC ≤ 0.9·E_rated` | **L** (jointly, `E` variable) | **yes** | unchanged | no |
| `sess_soc_final` | `SOC_T = 0.5·E_rated + slacks` | **L** | **yes** | unchanged | no |
| `sess_pch_hat_link` / `sess_pdch_hat_link` | `pch − S_rated·pch_hat = 0` | **BL** (`S` variable) | no | **remove** with the hats | yes |
| `sess_comp` | `pch_hat·pdch_hat ≤ eps` | **BL** | no | **drop** | **yes — see A5 proof** |
| `shared_energy_storage_s/e_sensitivities` | `S_fixed = S_rated` | **L** | **yes** | unchanged — this is the cut coefficient carrier | no |

> **Note on `sess_converter_capability`.** `‖(pnet, qnet)‖₂ ≤ S_rated` is a
> second-order cone and is **jointly convex in `(pnet, qnet, S_rated)`**. The
> production row is written squared, `pnet² + qnet² ≤ S_rated²`, which is *not*
> convex because of the `−S²` term. Writing it in norm form is an **exact**
> restatement for `S_rated ≥ 0`, not a relaxation. This is a free win.

## ESSO

| Component | Expression | Class | Unchanged? |
|---|---|---|---|
| `rated_s/e_capacity_unit` | `S_rated_per_unit = S_investment` | **L** | **yes** |
| `rated_s/e_capacity` | `S_rated = Σ_cohorts S_rated_per_unit` | **L** | **yes** |
| `available_s_capacity_unit` | `S_avail = S_rated_per_unit` | **L** | **yes** |
| `available_e_capacity_unit` | `E_avail = E_rated_per_unit · soh_cumul` | **BL** | **no** |
| `energy_storage_limits` | `pch ≤ s_max`, `pdch ≤ s_max` | **L** | **yes** |
| `energy_storage_normalization` | `pch − s_max·pch_hat = 0` | **BL** | **no** |
| `energy_storage_complementarity` | `pch_hat·pdch_hat ≤ eps` | **BL** | **no** |
| `energy_storage_operation_agg` (P) | `es_pnet = Σ_cohorts (pch − pdch) + slacks` | **L** | **yes** |
| `energy_storage_operation_agg` (capability) | `es_pnet² + es_qnet² ≤ es_s_rated²` | **NCQ** | **no** → norm form (exact) |
| `energy_storage_operation_agg` (aggregate links/comp) | `Σpch − S_total·hat = 0`, `hat·hat ≤ eps` | **BL** | **no** |
| `energy_storage_charging_discharging` | `avg = Σ (num_days/365)(η_ch·pch·dt + pdch·dt/η_dch)` | **L** | **yes** |
| `energy_storage_capacity_degradation` (daily) | `degradation·(2·cl_nom·E_rated) = avg` | **BL** | **no** |
| — (annual SoH) | `soh = 1 − degradation` | **L** | **yes** |
| — (cumulative SoH) | `soh_cumul = prev_soh · soh^(365·num_years)` | **NC** | **no** |
| — (minimum SoH) | `E_avail ≥ soh_min·E_rated_per_unit` | **L** | **yes** |
| — (cumulative degradation) | `deg_cumul = 1 − soh_cumul` | **L** | **yes** |
| Lifetime/calendar gating | build-time index filtering + `fix()` | structural, not algebraic | **yes** |

**The ESSO degradation model is NOT convex.** Live counts confirm it is active
in SRP1: `energy_storage_capacity_degradation` has 30 rows, and 6 of 9
`es_soh_per_unit` entries are free.

---

# A3 — W-space AC representation

## Lifted variables

```
Wii  = |V_i|²          = e_i² + f_i²
WijR = Re(V_i V_j*)    = e_i e_j + f_i f_j
WijI = Im(V_i V_j*)    = f_i e_j − e_i f_j
```

**The production model already carries all three as explicit variables**
(`vmag_sqr`, `voltage_product_real`, `voltage_product_imag`), coupled to `e`/`f`
only through `voltage_mag_sqr_def`, `voltage_product_real_def` and
`voltage_product_imag_def`. Dropping `e`/`f` and those three equalities, and
adding the SOC inequality, is therefore a **local, surgical change** — the rest
of the AC model is already written in W.

## Orientation convention

`_branch_voltage_products` returns `(WijR, WijI)` when the terminal is the
*from* bus and `(WijR, −WijI)` when it is the *to* bus. So a single
`(WijR, WijI)` pair per branch serves both terminals, with `WijI` negated —
consistent with `W_ji = conj(W_ij)`.

## Affine maps, for the actual production branch model

With `g = r/(r²+x²)`, `b = −x/(r²+x²)` (`compute_series_admittance`), line
charging `b_sh`, tap `r_ij` at the *from* bus, and **no phase shift** (absent
from the data model):

Nodal injections (`node_balance_p/q`, node `i`, over incident branches):

```
P_i = gs_i·Wii + Σ_{b: i=from} [ g_b·Wii·r_b²  − r_b(g_b·WijR + b_b·WijI) ]
              + Σ_{b: i=to}   [ g_b·Wii        − r_b(g_b·WijR − b_b·WijI) ]

Q_i = −bs_i·Wii + Σ_{b: i=from} [ −(b_b + b_sh/2)·Wii·r_b² + r_b(b_b·WijR − g_b·WijI) ]
                + Σ_{b: i=to}   [ −(b_b + b_sh/2)·Wii       + r_b(b_b·WijR + g_b·WijI) ]
```

Branch terminal flows (`compute_branch_terminal_power`):

```
P_ij = g_b·Wii·r_b²                 − r_b(g_b·WijR + b_b·WijI)
Q_ij = −(b_b + b_sh/2)·Wii·r_b²     + r_b(b_b·WijR − g_b·WijI)
P_ji = g_b·Wjj                      − r_b(g_b·WijR − b_b·WijI)
Q_ji = −(b_b + b_sh/2)·Wjj          + r_b(b_b·WijR + g_b·WijI)
```

Branch current-magnitude limit (non-transformer branches, `MIXED`):

```
(g_b² + b_b²)·( Wii + r_b²·Wjj − 2·r_b·WijR ) ≤ rate²
```

> **These are affine in `(Wii, Wjj, WijR, WijI)` exactly when `r_b` is fixed.**
> That holds for **every TSO branch** and for **every DSO branch except the one
> transformer**, where `r_b` and `r_b²` are variables and the expressions become
> bilinear. Line charging enters only as the constant `b_sh/2` shunt term
> (6 TSO branches, 0 DSO branches). Shunt conductance `g_sh` is zero throughout.
> No phase-shift term exists to carry.

## Validity of the SOC outer relaxation

For any complex `V_i, V_j`, the lifted triple satisfies exactly

```
(WijR)² + (WijI)² = (e_i e_j + f_i f_j)² + (f_i e_j − e_i f_j)²
                  = (e_i² + f_i²)(e_j² + f_j²) = Wii · Wjj
```

(Lagrange's identity — verified directly against
`voltage_product_real_rule` / `voltage_product_imag_rule` / `vmag_sqr_def`.)

Replacing it by

```
(WijR)² + (WijI)² ≤ Wii · Wjj ,     Wii ≥ 0, Wjj ≥ 0
```

admits every point the equality admits, so **every feasible point of the current
rectangular AC model maps to a feasible point of the relaxed model with the same
objective**. It is therefore a valid outer relaxation and its optimum is a lower
bound. In rotated-SOC form: `‖(2·WijR, 2·WijI, Wii − Wjj)‖₂ ≤ Wii + Wjj`.

The existing `voltage_product_real_nonnegative` (`WijR ≥ 0`) is affine and is
retained — it is a valid restriction of the *physical* set (it encodes
|angle difference| < 90°) and tightens the relaxation for free.

---

# A4 — TSO versus DSO relaxation choice

Topology from the built models (after parallel-branch merging):

| Network | Buses | Branches | Distinct pairs | Components | Cyclomatic | Status |
|---|---|---|---|---|---|---|
| `case9` (TSO) | 9 | 9 | 9 | 1 | **1** | **MESHED** |
| `case33_1` (DSO) | 33 | 32 | 32 | 1 | **0** | **RADIAL** |
| `case33_2` (DSO) | 33 | 32 | 32 | 1 | **0** | **RADIAL** |
| `case33_3` (DSO) | 33 | 32 | 32 | 1 | **0** | **RADIAL** |

*(The raw JSON contains 81 line entries per `case33`, but 24 node pairs are
duplicated up to 3×; `_pre_process_network` merges them. On the raw
branch-multigraph the cyclomatic number is 40, which would be misleading — the
rank-one coupling is per **node pair**, so the W-space graph is what matters,
and it is a tree.)*

## DSO — **SOC is the right first candidate**

The W-space graph is a tree, so on a radial network with the usual sufficient
conditions the SOC/branch-flow relaxation is known to be tight or near-tight,
and any residual gap is typically small. **However**, exactness results for
radial networks assume a fixed network; the controllable OLTC introduces the
tap–voltage bilinearity, whose envelope tightness must be checked empirically.
Recommendation: **SOC**, with tap envelopes, and measure the relaxation gap
against the nonlinear solution before trusting it.

## TSO — **SOC + QC strengthening**

`case9` has one independent cycle, so plain pairwise SOC loses the cycle
(angle-consistency) condition and can be loose. Recommended strengthening, in
increasing order of effort and all valid outer relaxations:

1. **Angle-difference envelopes** — `WijI ⋛ tan(θ_min/max)·WijR`. These rules
   **already exist** in the code (`branch_angle_difference_lower/upper_rule`)
   and are merely commented out at `network.py:403-404`, with `±30°` defaults.
   Essentially free.
2. **Bound tightening** on `Wii` from `v_min²/v_max²` (already present as
   bounds) and on `WijR`, `WijI` from those plus the angle envelopes.
3. **QC envelopes** on the voltage-magnitude products.
4. **Cycle constraints** for the single cycle.

**No SDP at this stage**, per instruction.

```
Recommended common formulation: SOC, with QC strengthening applied to the TSO.
```

---

# A5 — Planning-variable joint convexity

The lower-bound model must be jointly convex in operational variables **and**
the capacity variables `S`, `E`, or the cut coefficient is meaningless.

| Relation | Joint convexity in `(operational, S, E)` |
|---|---|
| `pch + pdch ≤ S` | **affine → jointly convex** ✓ |
| `‖(pnet, qnet)‖₂ ≤ S` | **SOC → jointly convex** ✓ (exact restatement of the production row) |
| `SOC_t ≤ 0.9·E` | **affine** ✓ |
| `SOC_t ≥ 0.1·E` | **affine** ✓ |
| `SOC_T = 0.5·E + slacks` | **affine** ✓ |
| `SOC_t = SOC_{t−1} + η_ch·pch·dt − pdch·dt/η_dch` | **affine**, no `S`/`E` ✓ |
| `S_fixed = S_rated`, `E_fixed = E_rated` | **affine** ✓ — the cut carriers survive |
| `pch = S·pch_hat` | **bilinear** ✗ — must be removed |
| `pch_hat·pdch_hat ≤ eps` | **bilinear** ✗ — must be dropped |

So the whole active-energy ESS block is jointly convex **once the H1 layer is
removed**, and the D2-P sensitivity-clean formulation is exactly what makes this
work: after D2-P there are **no capacity-dependent variable bounds**, so `S`
enters only through the affine/SOC rows above. **D2-P is a prerequisite for this
architecture, not merely a bug fix.**

## Proof that dropping complementarity preserves the lower-bound direction

Let `F` be the feasible set of the nonlinear model and `F'` the same set with
`sess_comp` (and the hat variables and their links) removed.

The hat variables appear **only** in `sess_pch_hat_link`,
`sess_pdch_hat_link` and `sess_comp` — verified by inspection of the built
model: no objective term and no other constraint references them (the D2-P
audit confirmed the objective uses physical `pch`/`pdch` only). Removing a
constraint and the variables that appear only in it therefore yields
`proj(F) ⊆ F'`: every point of `F` remains feasible in `F'`, since the dropped
rows are the only ones it must satisfy in the hat block.

Hence `min_{F'} f ≤ min_{F} f` for the same objective `f`, i.e. dropping
complementarity **can only lower** the optimal value. **The bound direction is
preserved.** The physical price is that the relaxed model may charge and
discharge simultaneously; this is acceptable in a lower-bound oracle and is
exactly the kind of relaxation artefact that the nonlinear UB evaluation exists
to catch.

The same argument applies to the ESSO per-cohort and aggregate complementarity
rows and their hat links.

---

# A6 — Planning / master convexity map

| Component | Expression | Class | Convex? |
|---|---|---|---|
| `es_s_investment`, `es_e_investment` | continuous, `NonNegativeReals` | — | ✓ |
| `es_s_rated`, `es_e_rated` | continuous, `NonNegativeReals` | — | ✓ |
| `alpha` | continuous, lower-bounded by `−budget·1e3` | — | ✓ |
| `rated_s/e_capacity` | `S_rated[e,y] = Σ_{y'≤y, within t_cal} S_investment[e,y']` | **L** | ✓ |
| `energy_storage_maximum_capacity` | `E_rated ≤ max_capacity` | **L** | ✓ |
| `energy_storage_power_to_energy_factor` | `E_inv ≥ ratio_min·S_inv`, `E_inv ≤ ratio_max·S_inv` | **L** | ✓ |
| `energy_storage_investment` (budget) | `Σ annualization·ω·(c_s·S_inv + c_e·E_inv) ≤ budget` | **L** | ✓ |
| `investment_cost` | same affine expression | **L** | ✓ |
| `objective` | `investment_cost + alpha` | **L** | ✓ |
| `benders_cuts` | `alpha ≥ Q + gᵀ(x − x₀)` | **L** | ✓ |
| Cohort lifetime | `t_cal`-based index filtering at build time | structural | ✓ |
| Salvage | enters via the **cut coefficient**, not as a master constraint | **L** | ✓ |
| Degradation-dependent investment/lifetime logic | **none in the master** — degradation lives entirely in the ESSO subproblem | — | ✓ |
| Integer / binary variables | **none** | — | ✓ |
| Bilinear / nonlinear planning equations | **none** | — | ✓ |

> **The current master is a linear program and can be solved globally today.**
> Nothing on the master side prevents a convex planning architecture. The
> obstruction is entirely in the operational recourse and in the ESSO
> subproblem.

---

# A7 — ESSO lower-bound treatment

Traced from `_build_subproblem` in `shared_energy_storage_data.py`.

| Relation | Expression | Affine / convex? |
|---|---|---|
| Throughput | `avg_ch_dch[y_inv,y] = Σ_d Σ_p (num_days/365)·(η_ch·pch·dt + pdch·dt/η_dch)` | **affine ✓** |
| Daily degradation | `degradation·(2·cl_nom·E_rated_per_unit) = avg_ch_dch` | **bilinear ✗** |
| Annual SoH | `soh = 1 − degradation` | **affine ✓** |
| Cumulative SoH | `soh_cumul = prev_soh · soh^(365·num_years)` | **nonconvex ✗** (variable to a large power, times a variable) |
| Available energy | `E_avail = E_rated_per_unit · soh_cumul` | **bilinear ✗** |
| Minimum SoH | `E_avail ≥ soh_min·E_rated_per_unit` | **affine ✓** |
| Cumulative degradation | `deg_cumul = 1 − soh_cumul` | **affine ✓** |
| Lifetime / cohort gating | build-time filtering plus `fix()` | **structural ✓** |

## Proposed treatment of each nonconvex term — **not implemented**

| Term | Proposal | Bound direction |
|---|---|---|
| `degradation·(2·cl_nom·E_rated) = avg` | **Omission that provably enlarges**: drop the equality in the LB model. `degradation` then only feeds SoH, which only feeds `E_avail`. | Relaxing an equality enlarges the set → **safe** |
| `soh_cumul = prev_soh·soh^(365·N)` | **Omission**: drop; replace by `0 ≤ soh_cumul ≤ 1`. | Enlarges → **safe** |
| `E_avail = E_rated·soh_cumul` | **Valid convex relaxation**: `E_avail ≤ E_rated` (since `soh_cumul ≤ 1`). Optionally McCormick over `soh_cumul ∈ [soh_min, 1]` for tightness. | More available energy → larger operational feasible set → lower recourse → **safe** |
| `E_avail ≥ soh_min·E_rated` | **Omission** (it is a restriction). | Enlarges → **safe** |
| `pch − S_cohort·pch_hat = 0` and cohort/aggregate `hat·hat ≤ eps` | **Omission** with the hat variables, as in A5. | Enlarges → **safe** |
| `es_pnet² + es_qnet² ≤ es_s_rated²` | **Exact convex reformulation**: `‖(es_pnet, es_qnet)‖₂ ≤ es_s_rated`. | Exact → **safe** |

> Every proposal above either enlarges the feasible set or is exact, so the
> relaxed ESSO cannot raise the recourse. **No proposal has an unknown bound
> direction.** The cost is that the LB model no longer represents degradation at
> all, which weakens the bound and — more importantly — means the LB ignores an
> economic effect the UB model still charges for. **This needs planner
> sign-off**: it is a modelling decision, not just a numerical one.

---

# A8 — Proposed convex SMOPF resource-retention matrix

| Resource / feature | Current nonlinear SMOPF | Proposed convex LB model | Exact / relaxed | LB-safe? |
|---|---|---|---|---|
| Conventional generator P | `pg ∈ [pmin, pmax]` | identical | **exact** | ✓ |
| Conventional generator Q | `qg ∈ [qmin, qmax]` | identical | **exact** | ✓ |
| RES active generation / curtailment | `pg ∈ [0, pg_avail]` | identical | **exact** | ✓ |
| RES reactive power | `pg² + qg² ≤ sg_avail²` | identical (already SOC) | **exact** | ✓ |
| RES power-factor control | `qg ⋛ tan·pg` | identical | **exact** | ✓ |
| Active load flexibility | box + day balance | identical | **exact** | ✓ |
| Reactive load flexibility | box | identical | **exact** | ✓ |
| Load curtailment | disabled | disabled | — | ✓ |
| **OLTC tap** | `r ∈ [0.83, 1.17]`, bilinear in W | **retained** via McCormick envelopes on `r·WijR`, `r·WijI`, `r_sqr·Wii`, plus `r_sqr ≥ r²` and its secant | **relaxed** | ✓ (outer) |
| Voltage magnitudes | `Wii = e²+f²` + limits | `Wii` primitive + identical limits | **relaxed** (via rank-one SOC) | ✓ |
| Voltage-product coupling | `WijR²+WijI² = Wii·Wjj` | `≤` (rotated SOC) | **relaxed** | ✓ |
| `WijR ≥ 0` | present | retained | **exact** | ✓ |
| Angle-difference limits | present but commented out | **enable** as free QC strengthening | tightening of the relaxation | ✓ |
| Branch current limits (lines) | affine in W | identical | **exact** | ✓ |
| Branch apparent limits (transformer) | `pij²+qij² ≤ rate²` | identical | **exact** | ✓ |
| Fixed shunts | `gs·Wii`, `−bs·Wii` | identical (zero in SRP1) | **exact** | ✓ |
| Controllable shunts / phase shifters / switching | absent | absent | — | ✓ |
| Voltage slacks | present | identical | **exact** | ✓ |
| Shared-ESS `pch`/`pdch` | present | identical | **exact** | ✓ |
| Shared-ESS `pnet`, `qnet` | `pnet = pch − pdch`; `pnet²+qnet² ≤ S²` | `pnet = pch − pdch`; `‖(pnet,qnet)‖₂ ≤ S` | **exact** (norm form) | ✓ |
| Shared-ESS active-sum | `pch + pdch ≤ S` | identical | **exact** | ✓ |
| Shared-ESS PF cone | `qnet ⋛ tan·pch − tan·pdch` | identical | **exact** | ✓ |
| Shared-ESS SOC + limits + day balance | affine | identical | **exact** | ✓ |
| **Shared-ESS complementarity** | `pch_hat·pdch_hat ≤ eps` | **dropped** with the hat variables | **relaxed** | ✓ (A5 proof) |
| Capacity-fixing rows | `S_fixed = S_rated` | identical — the cut carrier | **exact** | ✓ |
| Ordinary ESS | 0 rows in SRP1 | same families, convex except complementarity | **exact / relaxed** | ✓ |
| ESSO cohort power/limits/aggregation | affine | identical | **exact** | ✓ |
| ESSO capability | `pnet²+qnet² ≤ S²` | norm form | **exact** | ✓ |
| ESSO complementarity (cohort + aggregate) | bilinear | **dropped** | **relaxed** | ✓ |
| ESSO throughput | affine | retained | **exact** | ✓ |
| **ESSO degradation / SoH** | bilinear + power | **dropped**; `E_avail ≤ E_rated` | **relaxed** | ✓ (bound-safe, but see A7 caveat) |
| TSO–DSO interface P/Q | ADMM consensus | **single centralized coupling** (A10) | **exact** | ✓ |
| Interface voltage | `vmag` consensus | couple on `Wii` | **exact** | ✓ |

**Every controllable resource identified in A1 is retained.** No DC-OPF or
LinDistFlow substitution is proposed anywhere.

---

# A9 — Lower-bound / upper-bound architecture

## Definitions

- `x` — investment vector (`es_s_investment`, `es_e_investment`), the master's
  decision.
- `Q*_AC(x)` — the **true** global optimum of the full nonlinear AC recourse at
  `x`. Not computable.
- `R(x)` — optimal value of the **convex relaxed** operational problem at `x`,
  solved globally.
- `Q^feas_AC(x)` — objective of any **feasible** point of the full nonlinear AC
  recourse at `x` (what the current ADMM produces).

## Required sandwich

```
R(x)  ≤  Q*_AC(x)  ≤  Q^feas_AC(x)
```

The left inequality holds because every convex-model relaxation in A8 is either
exact or enlarges the feasible set (A5 and A7 give the direction proofs for the
dropped families). The right inequality holds because a feasible point's
objective cannot be below the optimum.

**This is exactly what P5.4-D3/D4 lacked.** There the cut was anchored at
`Q^feas_AC(x)`, which is an **upper** bound, and the cut was then used as if it
were a lower support — which is why candidates were repeatedly found below it.

## Convex supporting cut

At iteration `k` with candidate `x^k`:

```
α  ≥  R(x^k) + g_kᵀ (x − x^k),        g_k ∈ ∂R(x^k)
```

`g_k` is the vector of duals of the capacity-fixing rows
(`shared_energy_storage_s/e_sensitivities`), carried through the same
`objective_scale / baseMVA`, `annualization · num_years · num_days`,
TSO+DSO aggregation and available-capacity → investment mapping that
`_get_operational_sensitivities` already implements. Because `R` is the value
function of a **convex** program that is **jointly convex in `(operational, x)`**
(A5), `R` is convex in `x` and `g_k` is a genuine subgradient, so the cut is a
valid global under-estimator — the property D3/D4 proved the current cuts do not
have.

## Bounds and gap

```
planning LB  = master objective at its optimum = investment_cost(x*) + α*
planning UB  = min over validated incumbents of [ investment_cost(x) + Q^feas_AC(x) ]
gap          = (UB − LB) / |UB|
```

The master remains an LP (A6), so its optimum is a true global LB provided every
cut is valid.

## AC-feasibility polishing

An incumbent should be labelled a **rigorous** UB only after an AC-feasibility
polishing step: take the relaxed solution, recover a physically feasible AC
operating point (fix or warm-start the nonlinear SMOPF from it, restore
complementarity and the rank-one coupling), and confirm the nonlinear model
converges with the physical residual checks P5.4-E/H1 already use
(complementarity, converter capability, node balance). Until then an incumbent
is a *candidate* UB.

**Not implemented in this stage.**

---

# A10 — Centralized first

The first prototype must be a **single centralized convex program** covering the
TSO and all three DSOs. Assembly:

1. **One model per `(year, day, scenario)`** containing the TSO buses and all
   three DSO bus sets, kept in disjoint index spaces.
2. **Interface active/reactive power.** The TSO already represents each ADN as a
   load at its interface node (`get_adn_load_idx`), and each DSO represents the
   TSO as its reference bus. Centrally, replace the ADMM consensus with **direct
   equality**: `Pc_adn[node] = P_interface[dso]` and `Qc_adn[node] =
   Q_interface[dso]` — affine, exact, and it removes rho, dual variables and the
   consensus loop entirely.
3. **Common interface voltage.** Couple on `Wii`:
   `W_TSO[interface_node] = W_DSO[reference_node]` — **affine in W**.

> This is why `vmag` should be dropped. `vmag` exists only so ADMM can trade a
> voltage *magnitude*; it is tied to the rest of the model by the nonconvex
> `voltage_mag_def` (`Wii = vmag²`). Coupling on `Wii` instead is **affine**,
> needs no `sqrt(Wii)`, and removes a nonconvex equality. Introducing
> `sqrt(Wii)` would reintroduce nonlinearity for no modelling gain.

4. **Shared ESS.** In the centralized model the shared ESS at a given node
   appears once, so the TSO/DSO/ESSO triplication and its consensus disappear;
   the ESSO cohort/investment structure attaches directly.
5. **Objective.** Sum of the local objectives with their existing coefficients;
   `objective_scale` is an ADMM artefact and is not needed centrally.

Distributed convex ADMM is a separate, later question and must not be attempted
before the centralized relaxation is validated against the nonlinear model.

---

# A11 — Solver inventory

Measured on this machine; nothing installed or changed.

| Solver | Available | License | Pyomo interface | Duals | Conic (SOCP) |
|---|---|---|---|---|---|
| **IPOPT** 3.14.18 (`/usr/local/bin/ipopt`) | **yes** | EPL, open | yes (ASL) | **yes** (`dual`, `ipopt_zL/zU_out`) | **no** — general NLP, local method |
| **CBC** (`/opt/homebrew/opt/cbc/bin/cbc`) | **yes** | EPL, open | yes | LP duals only | **no** |
| **CLP** (`~/dist/bin/clp`) | **yes** | EPL, open | yes | yes (LP) | **no** |
| Gurobi | executable **not** found (Pyomo plugin present) | commercial | plugin only | — | would support SOCP |
| Mosek | **no** | commercial | — | — | would support SOCP/SDP |
| CPLEX / Xpress / SCIP / HiGHS / GLPK | **no** | — | — | — | — |
| Bonmin / Couenne / Knitro / BARON | **no** | — | — | — | — |
| CSDP / SDPA | **no** | — | — | — | reference only |

Python conic stack: `cvxpy`, `mosek`, `ecos`, `scs`, `clarabel`, `osqp`,
`gurobipy`, `cplex`, `xpress`, `highspy`, `pyscipopt` — **all absent**.
Pyomo 6.9.2 exposes `pyomo.core.kernel.conic` (`quadratic`,
`rotated_quadratic`), i.e. conic **modelling** with no solver to consume it.

## Recommendation

**No currently available solver provides a certified global conic solution.**
Two options, for planner decision:

1. **Interim: IPOPT on the convex model.** For a genuinely convex program with a
   constraint qualification, any KKT point is a global optimum, so IPOPT would
   in practice return the global value and does provide the duals the cut needs.
   This is usable for prototyping and is **not** "selecting IPOPT by default" —
   it is selecting it knowing the convexity does the work, not the solver.
   Caveats: no optimality certificate, and interior-point methods can struggle
   at the SOC apex (`Wii = Wjj = 0`), which must be guarded by the existing
   strictly positive voltage bounds.
2. **Preferred: install a conic solver.** Mosek (academic licence) or
   Gurobi ≥ 9 would give certified SOCP with duals. **Requires planner approval
   — no installation was performed.**

---

# A12 — Deterministic-state issue

P5.4-D4 observed that the same nominal source and multiplier policy produced
base recourse values 1.5e5 apart depending on the order of preceding production
calls. Traced, not fixed.

## Mutable shared objects most likely responsible

| Object | Mutated by | Effect |
|---|---|---|
| `network_planning.network[year][day].shared_energy_storages[i]` — `.s`, `.e`, `.e_init`, `.e_min`, `.e_max` | `_update_data_with_candidate_solution` (`network_data.py:2885`) | **In-place mutation of persistent network data.** Every later `build_model` and every rule closure reading `network.shared_energy_storages[e]` sees the last candidate's capacities. |
| `shared_ess_data.shared_energy_storages[year][i]` | `shared_ess_data.update_data_with_candidate_solution` | same, on the ESSO side |
| Pyomo model objects held in `state['models']` | `_update_operational_models_with_candidate`, `configure_shared_ess_operational_state` | mutated in place; `_clone_operational_models` clones, but the *network data* behind them is shared |
| `network.py` module-level solver log paths | `run_smopf` | concurrent/ordered writes to identical filenames |
| `shared_ess_data.solver_recovery_diagnostics` | reset at the start of `_run_operational_planning` | accumulates across calls otherwise |

## What must be reset or cloned for a pure oracle

For `evaluate(x, initial_state)` to be reproducible independently of call history:

1. **Deep-copy or rebuild the network data objects** (`Network.nodes`,
   `.branches`, `.generators`, `.loads`, `.shared_energy_storages`) per
   evaluation, or make `update_data_with_candidate_solution` write into a
   per-evaluation copy rather than the persistent planning object.
2. **Derive all capacity-dependent data from `x` alone** at the start of an
   evaluation, never from residual state.
3. **Isolate solver log paths** per evaluation.
4. **Reset accumulators** (`solver_recovery_diagnostics`, any cached scale
   factors such as `objective_scale`).
5. **Assert purity** with a regression test: evaluate the same `x` twice from
   different call histories and require bit-identical recourse — the identical
   check that returned exactly 0.0 spread in D3 when the history *was* the same.

This must be fixed **before** the nonlinear model is used for planning UB
validation, since a non-pure UB oracle would make the optimality gap
irreproducible.

---

# Verdict

The architecture is mathematically coherent and every relaxation has a proven
bound direction. Three substantive items are unresolved, and one of them is a
hard external dependency:

1. **No SOCP solver is available.** The A9 oracle requires a globally solved
   convex program with reliable duals. IPOPT can stand in on a convex model, but
   without a certificate — and the choice needs planner approval either way.
2. **The OLTC tap–voltage bilinearity is real and unavoidable.** The DSO tap is
   genuinely controllable, so it cannot be fixed to nominal (that would *shrink*
   the feasible set and destroy the lower-bound property). The proposed
   McCormick/secant envelopes are valid but their tightness on a radial feeder
   with a single regulating transformer is unverified.
3. **Discarding the ESSO degradation chain is bound-safe but changes what the
   LB model represents** — it ignores an economic effect the UB still charges
   for, which will inflate the optimality gap and needs an explicit modelling
   decision.

None of these is a dead end, and the master is already an LP, but "fully
specified" would overstate what has been established.

```
P5.5-A PARTIAL — architecture is promising but unresolved convexity/bound-direction issues remain
```

```
P5.5-A COMPLETE — ready for planner review before convex implementation
```
