# Local NLP Stability Investigation Plan

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role and scope

Act as an implementation and diagnostic agent.

Read `REVISION_CONTEXT.md` first, then read this file. For the current P5.5 work, this file takes precedence regarding what may and may not be changed. The filename is retained for continuity even though the active scope has moved from local-NLP repair to convex planning-architecture audit.

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

# CURRENT AUTHORIZED STAGE - P5.5-A exact convexity and controllable-resource audit

P5.4 is closed as an architectural investigation. Accepted statuses include:

- `P5.4-H1 PASS - complementarity is numerically resolved consistently across agents`
- `P5.4-F ADMM PASS - net-P/Q coordination converged with locally consistent charge/discharge`
- `P5.4-D2-P PASS - structurally complete S sensitivity productionized`
- `P5.4-D3 FAIL - current cuts are demonstrably unsafe under observed recourse branches`
- `P5.4-D4 C - hardened recourse still produces demonstrably unsafe cuts`

Current accepted nonlinear operational production baseline:

`06e921e5`

Accepted nonlinear behavior remains:

- active-energy ESS physics;
- H1 normalized complementarity with `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only ADMM coordination;
- positive-bootstrap local robustness;
- fixed-candidate distributed ADMM convergence in 9 cycles;
- sensitivity-clean shared-S local branch derivative bookkeeping.

Latest accepted diagnostic evidence:

- `65b261ba` - D2 root-cause audit;
- `e1afa8e9` - D3 cut-consistency FAIL;
- `1a68f409` - D4 hardened-cut class C result.

The old P5.4-G nonlinear-recourse Benders gate is permanently blocked. The current authorized sequence is:

1. create a separate branch `feature/convex-planning` from current HEAD;
2. execute **P5.5-A audit/design only**;
3. stop for planner review before any convex implementation.

---

# Accepted D4 conclusion - nonlinear derivative cuts are retired

D4 established that the problem is not repaired by better local sensitivity bookkeeping or by hardening the nonlinear recourse anchor.

Accepted evidence:

- the exact base candidate can reach a best observed recourse around `835315903.22`, about `12.94e6` below the production cold branch;
- lower branches transfer back to the base candidate feasibly and converge in 2-5 cycles;
- the dominant objective gap is TSO generation dispatch on Spring representative days, not ESS operation;
- deterministic branch templates are operationally useful but do not stabilize a valid global support function;
- hardened cuts failed on `11/11` neighbouring candidates, all beyond the derived cut tolerance;
- worst hardened cut gap is about `-1.16e6`, whereas the linear capacity contribution is only about `1.3e4`;
- successive hardening therefore follows different local branches rather than converging to a safe recourse oracle;
- call-order dependence of roughly `1.5e5` for the same nominal transferred start reveals mutable shared state that must later be cleaned for a reproducible nonlinear upper-bound oracle.

Do not call any observed branch the global optimum.

The accepted planning conclusion is:

**The current nonlinear-recourse derivative-cut/Benders planning machinery must not be used as a global lower-bound method.**

---

# New architecture - convex lower bound plus nonlinear AC feasible upper bound

For planning candidate `x = (S,E)`, the target architecture is:

`R(x)` = globally solved convex-relaxed AC SMOPF recourse;

`Q_AC_feas(x)` = feasible recourse of the full accepted nonlinear AC SMOPF, eventually AC-polished before being labelled a rigorous planning upper bound.

Required bound relation:

`R(x) <= Q_AC*(x) <= Q_AC_feas(x)`.

If `R` is jointly convex in operational variables and S/E, the lower-bound master may use supporting cuts:

`alpha >= R(x_k) + g_k^T (x - x_k)`.

The nonlinear model does **not** supply the lower cut. It supplies physically feasible incumbents/upper bounds and remains the validation truth model.

The intended first convex family is a lifted **W-space AC SOC/QC relaxation**. Do not substitute DC-OPF or LinDistFlow.

---

# New development branch

Before any P5.5-A audit work, create:

```bash
git switch -c feature/convex-planning
```

from the repository's **current HEAD at execution time**.

Record:

- source branch;
- source commit;
- new branch;
- clean/dirty worktree state.

Do not rebase, reset or rewrite previous history. Do not modify the previous branch.

---

# Locked production decisions during P5.5-A

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
- objective scaling;
- objective coefficients;
- current master/Benders equations;
- production data.

Do not add or implement during P5.5-A:

- a convex OPF model;
- a new planning loop;
- new ADMM consensus variables;
- binary charge/discharge variables;
- trust-region/local-cut modifications;
- pattern search;
- surrogate optimization;
- SDP;
- solver installations/license changes.

P5.5-A is **audit/design only**.

---

# P5.5-A - exact convexity and controllable-resource audit

## A1 - establish the exact current SRP1 control set

Inspect actual data and model-construction logic for:

- TSO `case9`;
- DSO `case33_1` at node 5;
- DSO `case33_2` at node 7;
- DSO `case33_3` at node 9.

For every operational feature distinguish explicitly:

`framework-supported | instantiated in SRP1 | controllable in SRP1 | fixed in SRP1 | absent from SRP1`.

Audit at least:

- conventional generator P and Q;
- RES active generation;
- RES curtailment;
- RES Q control and PF control;
- active flexibility;
- reactive flexibility;
- ordinary ESS;
- shared ESS;
- ESS charge/discharge P;
- ESS Q;
- ESS SOC;
- voltage control;
- transformer tap ratios;
- phase-shifting transformers;
- capacitor banks / shunts;
- branch switching, if any;
- load shedding / operational slacks;
- TSO-DSO P/Q exchanges and interface voltage.

### OLTC requirement

Resolve OLTC status from both data and model construction. For every transformer branch report:

- from bus;
- to bus;
- nominal tap;
- phase shift;
- whether tap is a variable or fixed quantity;
- whether tap control is enabled;
- whether discrete tap logic is enabled.

End with exactly one unambiguous statement:

`SRP1 HAS controllable OLTCs`

or:

`SRP1 DOES NOT HAVE controllable OLTCs; transformer ratios are fixed`.

Do the equivalent explicit check for controllable capacitor banks/shunts.

Do not infer these answers from previous experiments.

## A2 - row-by-row operational convexity map

For every active constraint/objective family report:

- component name;
- mathematical expression;
- variables;
- TSO/DSO applicability;
- classification;
- exact/relaxed status in a future convex model;
- proposed convex representation;
- proof/direction that the proposed change enlarges the feasible set or otherwise preserves lower-bound validity.

Use classes:

- `L` - affine/linear;
- `CQ` - convex quadratic;
- `SOC` - second-order-cone representable;
- `NCQ` - nonconvex quadratic;
- `BL` - bilinear;
- `NC` - other nonconvex.

Audit at least the following.

### AC network

- active nodal balance;
- reactive nodal balance;
- rectangular voltage variables;
- `vmag_sqr`;
- voltage-magnitude definitions;
- voltage bounds;
- reference-angle treatment;
- branch P/Q flow definitions at both ends;
- branch thermal limits;
- angle-difference constraints if present;
- transformer equations;
- fixed shunts;
- controllable shunts;
- interface-voltage constraints.

### Generation

- conventional P bounds;
- conventional Q bounds;
- generator capability;
- PV voltage setpoints;
- generation cost;
- market/economic terms.

### RES

- scenario availability;
- curtailment;
- reactive capability;
- PF constraints.

### Demand/flexibility

- demand equations;
- upward/downward flexibility;
- flexibility bounds;
- day-energy balance;
- reactive flexibility.

### Ordinary and shared network ESS

- `pch/pdch`;
- `pnet`;
- `qnet`;
- active-energy SOC;
- SOC limits;
- converter capability;
- active-sum limit;
- H1 normalized complementarity;
- day balance.

Use the **post-D2-P** shared-ESS production formulation.

### ESSO

- cohort variables;
- available S/E capacity;
- active-power aggregation;
- converter capability;
- per-cohort complementarity;
- aggregate complementarity;
- throughput;
- degradation;
- SoH;
- lifetime/calendar gating.

Do not assume ESSO degradation/SoH is convex.

## A3 - derive the actual W-space AC representation

Derive lifted variables:

`Wii  = |Vi|^2`

`WijR = Re(Vi Vj*)`

`WijI = Im(Vi Vj*)`.

Starting from the actual production rectangular equations, show explicitly how:

- nodal P;
- nodal Q;
- branch `Pij/Qij`;
- branch `Pji/Qji`

become affine functions of W for the actual line/transformer model.

Include any production terms for:

- series conductance/susceptance;
- line charging;
- fixed tap ratio;
- fixed phase shift.

Do not replace the production branch model with a simplified textbook line model.

The exact voltage-product relation is:

`WijR^2 + WijI^2 = Wii * Wjj`.

The candidate SOC relaxation is:

`WijR^2 + WijI^2 <= Wii * Wjj`.

Prove that the inequality is a valid outer relaxation of the rank-one AC voltage relation.

## A4 - topology and TSO/DSO relaxation choice

From actual graph data report for each case:

- buses;
- branches;
- connected components;
- cyclomatic number;
- radial/meshed status.

Assess:

### DSO

whether standard SOC/W-space relaxation is a defensible first lower-bound candidate.

### TSO

whether pairwise SOC should be strengthened by some combination of:

- QC envelopes;
- angle-difference envelopes;
- cycle constraints;
- bound tightening.

Do not implement SDP in P5.5-A.

State the recommended first formulation separately for TSO and DSO, e.g. `SOC` or `SOC + QC strengthening`, and justify it.

## A5 - joint convexity in planning capacities S/E

A fixed-capacity convex OPF is not sufficient for valid planning cuts.

Audit joint convexity with available S and E capacities.

For shared ESS, prefer forms such as:

`pch + pdch <= S`

and:

`||(pnet,qnet)||_2 <= S`.

Check SOC limits/anchors such as:

`SOC_t <= alpha * E`

`SOC_t >= beta * E`.

In the convex lower-bound model:

- H1 hat variables should normally disappear;
- `S * p_hat` bilinear links must not remain;
- charge/discharge complementarity should be dropped/relaxed if needed for convexity.

Prove that dropping complementarity enlarges the feasible set and therefore preserves the lower-bound direction for a minimization recourse problem.

Do **not** change H1 in the nonlinear production model.

## A6 - planning/master convexity map

Audit the current planning model itself, including:

- S investment variables;
- E investment variables;
- investment cost;
- yearly accumulation;
- available-capacity mapping;
- investment cohorts;
- lifetime logic;
- salvage;
- degradation-dependent investment/lifetime logic;
- ESSO SoH coupling;
- integer/binary variables if any;
- all bilinear/nonlinear master equations.

Classify every planning-side term with the same convexity categories.

State whether the current master is itself globally solvable as a convex problem. If not, identify exactly which rows/objective terms prevent that.

## A7 - ESSO lower-bound treatment

Trace the actual ESSO degradation equations and determine whether:

- throughput is affine;
- degradation mapping is convex;
- SoH mapping is convex;
- lifetime/cohort constraints are convex;
- products/powers create nonconvexity.

For each nonconvex term propose, but do not implement, exactly one justified treatment class:

- exact convex reformulation;
- convex epigraph;
- valid convex relaxation;
- omission that provably lowers objective or enlarges feasible region;
- master-side treatment.

Do not use an approximation whose bound direction is unknown.

## A8 - proposed convex SMOPF resource-retention matrix

Produce a final table:

`Resource / feature | Current nonlinear SMOPF | Proposed convex LB model | Exact / relaxed | Lower-bound-safe?`

It must include **every controllable resource discovered in A1**.

Design rule:

**retain every controllable resource whenever possible.**

Do not use DC-OPF or LinDistFlow as replacements that silently remove AC control capability.

## A9 - lower-bound / upper-bound planning architecture specification

Specify mathematically:

`R(x)` = globally solved convex-relaxed operational recourse;

`Q_AC_feas(x)` = feasible full nonlinear AC recourse.

Required relation:

`R(x) <= Q_AC*(x) <= Q_AC_feas(x)`.

If joint convexity holds, define supporting cut:

`alpha >= R(x_k) + g_k^T (x - x_k)`.

Define:

- planning lower bound = master objective;
- planning upper bound = best validated nonlinear feasible incumbent;
- planning optimality gap.

Important nonlinear incumbents should eventually be AC-polished/feasibility-validated before they are called rigorous UBs.

Do not implement the planning loop in P5.5-A.

## A10 - centralized convex prototype comes first

The first future convex prototype must be centralized.

Do not immediately reproduce TSO/DSO ADMM.

Specify how a centralized convex relaxation would preserve:

- TSO/DSO active interface power;
- TSO/DSO reactive interface power;
- common interface voltage.

For voltage coordination in W-space, assess direct coordination of `Wii = |Vi|^2` rather than introducing nonlinear `sqrt(Wii)` variables.

Distributed convex ADMM is a later question after centralized mathematical validation.

## A11 - solver inventory

Without installing anything, inspect available solvers for:

- LP/QP;
- SOCP;
- convex QCQP;
- SDP only as a reference capability.

For each solver report:

- installed/available yes/no;
- license status if known;
- Pyomo/native interface availability;
- conic support;
- reliable dual support;
- whether it can provide a globally solved convex lower-bound oracle.

Do not default to IPOPT merely because it is already configured.

Do not install commercial software or change licenses without planner approval.

## A12 - deterministic-state issue

Trace, but do not yet fix, the D4 call-order dependency.

Identify mutable shared objects modified by:

- candidate updates;
- operational model updates;
- ADMM;
- warm-start transfers;
- result processing.

State what must eventually be cloned/reset so that:

`evaluate(x, initial_state)`

is a reproducible/pure oracle independent of previous candidate evaluations.

This must be resolved before the nonlinear model is trusted as a planning upper-bound validation oracle.

---

# P5.5-A acceptance and required report

Create:

`P5_5_CONVEX_PLANNING_ARCHITECTURE_REPORT.md`.

The report must contain A1-A12 and a top-level architecture summary covering:

1. exact SRP1 controllable-resource inventory;
2. definitive OLTC/cap-bank conclusion;
3. exact list of nonconvex operational families;
4. recommended SOC/QC formulation;
5. exact resources retained;
6. exact relationships relaxed;
7. master-side nonconvexities, if any;
8. proposed LB/UB algorithm;
9. available solver recommendation;
10. blockers before implementation.

End with exactly one:

`P5.5-A PASS - a mathematically valid convex lower-bound SMOPF architecture is fully specified`

`P5.5-A PARTIAL - architecture is promising but unresolved convexity/bound-direction issues remain`

`P5.5-A FAIL - the current model cannot be converted into a practical valid convex lower-bound oracle without a larger redesign`

Then stop with:

`P5.5-A COMPLETE - ready for planner review before convex implementation`

Do not implement the convex model yet.

---

# Old P5.4-G status

The old nonlinear-recourse `run_planning_problem()` Benders path is **permanently blocked** by D3/D4 evidence.

Do not run it during P5.5-A.

If a convex-relaxation planning loop is later authorized, treat it as a **new planning architecture**, not as validation of old P5.4-G.

---

# Deferred items

Keep deferred during P5.5-A:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R until defensible converter `Smax` data exists;
- calendar-degradation experimentation beyond convexity/bound-direction classification;
- nonlinear solver/ADMM retuning;
- implementation of convex relaxation itself;
- distributed convex ADMM;
- SDP;
- trust-region/local-cut planner;
- derivative-free pattern/coordinate search;
- surrogate-assisted planning.

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
