# Local NLP Stability Investigation Plan

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role and scope

Act as an implementation and diagnostic agent.

Read `REVISION_CONTEXT.md` first, then read this file. For the current P5.5 work, this file takes precedence regarding what may and may not be changed. The filename is retained for continuity even though the active scope has moved from local-NLP repair to convex planning-architecture design.

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

# CURRENT AUTHORIZED STAGE — P5.5-B mathematical closure of the convex lower-bound architecture

P5.4-R has completed canonical-environment revalidation. The old nonlinear-recourse derivative-cut/Benders architecture remains retired and P5.5 may resume.

Current accepted nonlinear production baseline:

`06e921e5`

Accepted nonlinear decisions remain:

- active-energy ESS physics;
- H1 normalized complementarity with `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- D2-P sensitivity-clean shared-S local-branch derivative bookkeeping;
- nonlinear AC SMOPF retained as physical truth / feasible-upper-bound model.

The active development branch is:

`feature/convex-planning`

P5.5-B is **design/audit only**. Do not implement the convex SMOPF yet.

---

# HARD REPRODUCIBILITY GATE

All active paper-instance work must use:

`/opt/anaconda3/envs/opf_env_py311/bin/python`

Canonical SRP1 checksum:

`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`

Every P5.5 validation/design harness that loads the paper case must record at minimum:

- `sys.executable`;
- resolved conda environment;
- Python version;
- NumPy/pandas/SciPy versions;
- Pyomo version;
- IPOPT path/version and configured HSL solver when relevant;
- Gurobi/gurobipy version and licence status when relevant;
- realized scenario checksum.

Abort rather than continue if the checksum differs.

`srp_env` is noncanonical. Its D3/D4 numerical evidence is historical only.

---

# ACCEPTED P5.4-R CANONICAL RESULTS

## R1 — operational production regression

Authoritative canonical gate:

- DSO `36/36`;
- TSO `12/12`;
- ESSO `3/3`;
- primary failures / recoveries / persistent failures `0/0/0`;
- H1 complementarity violations `0/1728`;
- converter-capability violations `0/1728`;
- total network IPOPT iterations `3424`;
- mean / median / max `71.3 / 65.0 / 134`;
- runtime about `42 s`;
- full-row-rank representative equality Jacobians;
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
- the predicted linear capacity effect remains below the recourse-resolution scale.

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
- gaps are approximately `-0.96e6` to `-1.13e6`;
- all exceed canonical `tol_cut`;
- linear capacity contributions remain only O(`1e4`).

Accepted verdict:

`P5.4-R-D4 C — canonical hardened cuts remain demonstrably unsafe`.

Therefore the current nonlinear-recourse derivative-cut/Benders machinery must not be used as a global lower-bound method.

## Environment sensitivity and call-history impurity are separate

Do not conflate:

- different environments generating different stochastic scenarios / different branches;
- persistent in-place data mutation causing call-history dependence.

The first is a reproducibility/environment issue. The second remains an independent software-hygiene issue that must be fixed before a nonlinear oracle is used to certify planning upper bounds.

---

# P5.5-A ACCEPTED FINDINGS

P5.5-A remains:

`P5.5-A PARTIAL — architecture is promising but unresolved convexity/bound-direction issues remain`.

Accepted facts:

- current master is an LP;
- TSO `case9`: 9 buses, 9 branches, one independent cycle;
- DSO `case33_1/2/3`: 33 buses, 32 branches after preprocessing, radial;
- SRP1 has one controllable continuous OLTC per DSO on branch 1, bus 1 -> bus 2;
- OLTC range `[0.83, 1.17]`;
- no phase shift/discrete tap logic;
- no controllable capacitor bank or switched shunt in SRP1;
- conventional generation P/Q, RES P/Q/curtailment/PF control, active/reactive flexibility, shared ESS P/Q/SOC and interface P/Q must all be retained;
- ordinary ESS is absent from SRP1;
- lifted W-space AC SOC/QC remains the intended convex family;
- no DC-OPF or LinDistFlow substitution is authorized.

The following A-stage design statements are **not final** and are reopened in B:

- dormant ±30-degree angle limits are not automatically lower-bound safe;
- McCormick tap-voltage envelopes are not yet the preferred OLTC formulation;
- centralized REF/ADN/shared-ESS sign semantics need exact tracing;
- ESSO degradation/SoH relaxation needs an objective-level proof including salvage;
- Gurobi dual-to-cut mapping and intercept construction are not yet established.

---

# SOLVER DECISION — GUROBI SELECTED FOR THE PROTOTYPE

P5.4-R4 corrected the old solver inventory.

Canonical environment provides:

- `gurobipy 13.0.1`;
- academic licence valid through `2027-04-10`;
- Pyomo `gurobi`, `gurobi_direct`, and `gurobi_persistent` available;
- `QCPDual=1` supported;
- linear and quadratic/conic duals verified;
- `ObjVal`, `ObjBound`, and reported optimality gap available.

Use Gurobi as the preferred convex-LB prototype solver. IPOPT may be used only as a diagnostic cross-check of a genuinely convex formulation, not as the official certified LB oracle.

For a scalar rigorous lower bound, prefer the solver dual bound `ObjBound` rather than `ObjVal`.

For a **planning cut**, do not mix `ObjBound` with an independently read sensitivity vector. P5.5-B must derive one affine function from one dual-feasible solution:

`L_k(x) = beta_k + g_k^T x`

and prove:

`L_k(x) <= R(x)`

for every admissible planning candidate.

---

# NEW ARCHITECTURE — convex lower bound plus nonlinear AC feasible upper bound

For planning candidate `x = (S,E)`:

`R(x)` = globally solved convex-relaxed AC SMOPF recourse;

`Q_AC_feas(x)` = feasible recourse of the full accepted nonlinear AC SMOPF, eventually AC-polished before being labelled a rigorous planning upper bound.

Required sandwich:

`R(x) <= Q_AC*(x) <= Q_AC_feas(x)`.

The nonlinear model supplies feasible incumbents/upper bounds. It must not supply global lower cuts.

The intended first convex family is a lifted **W-space AC SOC/QC relaxation**.

---

# LOCKED PRODUCTION DECISIONS DURING P5.5-B

Do not change:

- nonlinear AC SMOPF equations;
- active-energy ESS formulation;
- D2-P sensitivity-clean shared-S formulation;
- H1 dimensionless complementarity in the nonlinear model;
- `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- IPOPT tolerances/options;
- MA97/exact-Hessian policy;
- recovery policy;
- adaptive-rho logic;
- proximal regularization;
- nonlinear objective scaling/objective coefficients;
- current old master/Benders equations;
- production data.

Do not implement yet:

- the centralized convex SMOPF;
- the replacement planning loop;
- distributed convex ADMM;
- SDP;
- trust-region/local-cut planning;
- derivative-free pattern/coordinate search;
- surrogate optimization.

---

# B0 — historical-record cleanup

Before further design work, make the P5.4 report internally consistent without deleting historical evidence.

## B0.1 — mark noncanonical D2-P numerical regression evidence

The structural D2-P proof and PASS remain accepted.

Mark numerical results produced under `srp_env` as:

`NONCANONICAL — superseded by P5.4-R1/R2 for numerical validation`.

Use R1's canonical `3424` bootstrap iterations for the paper-instance operational regression.

## B0.2 — correct P5.4 open-items/current-summary numbers

Do not retain the following as current paper-instance facts:

- ~`1.3e7` branch gap;
- `11/11` hardened violations;
- noncanonical `1.5e5` call-order magnitude as though it were revalidated canonically;
- `3442` as canonical D2-P bootstrap iterations.

Current canonical planning evidence is:

- `Q_base_cold = 838496830.813414`;
- D3 worst cut gap about `-2.680808e6`;
- D3 decisive violations `2/8`;
- recovered-base improvement about `1.910367e6`;
- hardened test `8/8` decisive;
- R1 bootstrap iterations `3424`.

## B0.3 — separate environment sensitivity from mutable-state impurity

Correct any statement implying that changing environment proves the A12 call-history bug.

## B0.4 — provenance audit of older H1/F evidence

If an evidence file records canonical checksum/provenance, label it canonical.

If provenance cannot be established, do not silently call it canonical. Do not rerun historical stages unless a currently active conclusion requires it; R1/R2 already supply the operational paper-instance gate needed for P5.5.

---

# P5.5-B — MATHEMATICAL CLOSURE

## B1 — admissible relaxation strengthening

Correct the A4/A8 statement that the dormant ±30-degree angle constraints are a free tightening.

Classify every prospective strengthening as:

- `A` — present in the original nonlinear feasible set;
- `B` — mathematically implied by original nonlinear constraints;
- `C` — additional restriction, therefore **not admissible** in a certified lower-bound relaxation.

Only A/B may enter the certified LB model.

Retain `WijR >= 0` because production already imposes it.

Derive QC/voltage-product bounds only from production-safe information such as:

- active voltage bounds;
- existing `WijR >= 0`;
- active branch limits;
- other constraints already enforced by the nonlinear model.

Do not introduce ±30-degree bounds unless they are proved redundant for the original nonlinear feasible set.

## B2 — transformed continuous-OLTC formulation

Treat the transformed formulation as the preferred candidate unless the proof fails.

For each DSO OLTC define:

`U_i = r^2 * W_ii`

`C_ij = r * W_ij^R`

`D_ij = r * W_ij^I`.

Starting from the exact production equations, derive all transformer:

- nodal active-power contributions;
- nodal reactive-power contributions;
- `Pij`, `Qij`, `Pji`, `Qji`;
- thermal-limit expressions.

Verify signs directly from current implementation.

Test whether the equations become affine in:

`U_i, C_ij, D_ij, W_jj`.

Derive the physical rank relation:

`C_ij^2 + D_ij^2 = U_i * W_jj`

and the SOC relaxation:

`C_ij^2 + D_ij^2 <= U_i * W_jj`.

Audit every occurrence of `r` and `r_sqr`. Confirm whether SRP1 has:

- no tap cost;
- no tap-movement penalty;
- no intertemporal tap coupling;
- no discrete tap positions;
- no phase shift.

Because production voltage lower bounds imply `W_ii > 0`, test whether the continuous tap condition is represented exactly by:

`r_min^2 * W_ii <= U_i <= r_max^2 * W_ii`

with recoverable:

`r = sqrt(U_i/W_ii)`.

If yes and `r` appears nowhere else, determine whether `r` and `r_sqr` can be eliminated from the convex LB model.

Compare transformed-SOC versus A-stage McCormick in:

- validity;
- tightness;
- auxiliary count;
- joint convexity;
- dual interpretation;
- whether any additional relaxation beyond the standard voltage-product SOC is introduced.

Do not claim exactness until proved against every production occurrence.

## B3 — exact centralized TSO/DSO interface semantics

Trace the DSO `REF` generator from data through node balance, objective and ADMM consensus.

Resolve whether it is:

- a physical generator;
- the representation of TSO import/export;
- or a special combination.

Produce an explicit sign table for:

- TSO ADN active power;
- DSO REF `pg`;
- TSO ADN reactive power;
- DSO REF `qg`;
- shared-ESS P/Q copies;
- interface-voltage consensus.

For the **first convex prototype**, preserve separate TSO/DSO/ESSO copies and replace ADMM consensus with exact affine coupling equalities.

Prove that, before AC relaxation, the centralized equalities describe exactly the zero-residual consensus feasible set of the current decomposed formulation.

Do not collapse duplicate variables yet.

## B4 — ESSO objective-level lower-bound proof

Extend the feasible-set argument to the complete objective.

Trace every term involving:

- `pch`, `pdch`;
- throughput;
- degradation;
- SoH;
- available E;
- ESS usage;
- complementarity;
- slacks;
- salvage.

For every term affected by relaxation report:

`original expression | coefficient/sign | minimum possible contribution | proposed LB expression | proof the change cannot increase the relaxed optimum`.

Investigate the minimal relaxation:

`0 <= E_available <= E_rated`

with degradation/SoH relations omitted.

Check:

- joint convexity in investment E;
- finiteness/boundedness;
- whether E becomes too weak operationally to yield useful planning sensitivity.

A weak but rigorous LB is acceptable. An unproven tightening is not.

## B5 — salvage placement

Trace terminal salvage end to end.

Determine whether it depends on:

- investment S;
- investment E;
- available E;
- degradation/SoH;
- operational variables.

Decide whether salvage belongs:

1. inside relaxed recourse;
2. exactly in the master;
3. as a separate affine planning term.

Write the final mathematical definition of `R(x)` including all objective terms and prove:

`R(x) <= Q_AC*(x)`.

## B6 — rigorous Gurobi dual/cut contract

Build tiny parameterized convex test problems that mimic the intended capacity-fixing structure:

`S = S_fixed`

plus an SOC/QCP capability such as:

`||(p,q)||_2 <= S`.

Use `QCPDual = 1`.

Through the intended modelling interface verify:

- linear fixing-row dual;
- conic/QCP dual;
- sign convention;
- `ObjVal`;
- `ObjBound`;
- reported gap.

### Critical cut requirement

Do **not** assume:

`ObjBound(x_k) + g_k^T(x-x_k)`

is valid merely because `ObjBound` is a scalar lower bound.

Derive the cut from one dual-feasible solution:

`L_k(x) = beta_k + g_k^T x`.

Establish analytically how `beta_k` is constructed from:

- dual multipliers;
- fixed RHS/constants;
- objective constants;
- capacity-fixing rows.

At the generating point compare:

- `L_k(x_k)`;
- `ObjBound`;
- `ObjVal`.

Sweep the capacity parameter and require:

`L_k(x) <= R_solved(x) + tol`

for every test point.

Prefer an additional toy problem where the dual function is analytically evaluable.

The output of B6 must be the exact cut-intercept contract that the future planner will implement.

## B7 — Pyomo/Gurobi modelling-interface decision

Compare:

- `gurobi_direct`;
- `gurobi_persistent`;
- native `gurobipy` if required.

Evaluate:

- SOC/QCP representation;
- QCP dual access;
- linear dual access;
- `ObjBound` access;
- model-update cost across planning candidates;
- stable identification of capacity-fixing rows;
- numerical scaling;
- implementation burden.

Choose one interface for the first centralized prototype.

Do not install another solver.

## B8 — exact centralized prototype specification

If B1-B7 close successfully, specify the first implementation so no architecture decision remains.

The first prototype must:

- enforce canonical environment/checksum gate;
- be centralized;
- preserve every SRP1 controllable resource;
- preserve continuous OLTC control;
- use transformed-OLTC SOC if B2 proves it;
- preserve REF/ADN signs exactly;
- retain separate TSO/DSO/ESSO copies initially;
- replace ADMM with exact affine consensus constraints;
- use W-space AC SOC relaxation;
- use only lower-bound-safe QC strengthening;
- drop ESS complementarity only in the LB model;
- leave H1 unchanged in the nonlinear UB model;
- use the proven ESSO relaxation;
- expose a rigorous dual-derived S/E cut;
- return `ObjVal`, `ObjBound`, primal feasibility, dual information and primal-dual gap.

Do not implement it during B.

---

# P5.5-B acceptance and required report

Extend:

`P5_5_CONVEX_PLANNING_ARCHITECTURE_REPORT.md`

with:

`P5.5-B — mathematical closure`.

Correct A4/A8 as required:

- remove unsafe dormant ±30-degree language;
- replace “OLTC bilinearity is unavoidable” if B2 eliminates it;
- preserve Gurobi as selected prototype solver.

Update the P5.4 report through B0.

End with exactly one:

`P5.5-B PASS — convex lower-bound architecture is mathematically closed and ready for implementation`

or:

`P5.5-B PARTIAL — one or more lower-bound/cut/interface issues remain unresolved`

or:

`P5.5-B FAIL — no practical rigorous convex lower-bound formulation could be established`

then:

`P5.5-B COMPLETE — ready for planner review before implementation`.

Stop. Do not implement the convex model.

---

# OLD P5.4-G STATUS

P5.4-G is permanently blocked for the old nonlinear-recourse derivative cuts.

Do not run it.

Any future convex-relaxation planning loop is a **new planning architecture**, not validation of old P5.4-G.

---

# DEFERRED ITEMS

Keep deferred during P5.5-B:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R until defensible converter `Smax` data exists;
- nonlinear solver/ADMM retuning;
- implementation of the convex relaxation itself;
- distributed convex ADMM;
- SDP;
- trust-region/local-cut planning;
- derivative-free pattern/coordinate search;
- surrogate-assisted planning;
- production fix for nonlinear-oracle call-history impurity, except for tracing/specification if B-stage design requires it.

---

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
