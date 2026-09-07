# Local NLP Stability Investigation Plan

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role and scope

Act as an implementation and diagnostic agent.

Read `REVISION_CONTEXT.md` first, then read this file. For the current P5.4 work, this file takes precedence regarding what may and may not be changed.

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

# CURRENT AUTHORIZED STAGE — P5.4-D2 capacity-sensitivity root-cause audit

P5.4-A/B/C/D/E/E2/H1/F are complete. H1 and F are accepted. The current authorized stage is **P5.4-D2 — shared-S/E sensitivity root-cause audit before Benders**.

Do not run P5.4-G until planner review of D2.

## Current production/evidence checkpoints

Production active-energy stack:

- `a4a0bae8` — shared network active-energy ESS;
- `1e86d40e` — ordinary network ESS parity;
- `58f4911b` — ESSO active-energy conversion;
- `c3526ec8` — lifecycle/sensitivity audit state and post-B/C/D validation;
- `93974d83` — normalized dimensionless complementarity across shared network ESS, ordinary ESS and ESSO, including aggregate ESSO complementarity.

Accepted live coordination evidence:

- `2917b9c9` — fixed positive-bootstrap distributed ADMM diagnostic; net-P/Q coordination only.

Retained diagnostic evidence:

- `b0e53bc4` — E2 complementarity-significance diagnostics.

Reduced-scenario identity remains:

- seed `2026`;
- combined checksum `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

## Accepted P5.4 operational baseline — authoritative

The active-energy ESS physical correction is productionized and accepted:

- shared and ordinary network SOC use active `pch/pdch`;
- pure reactive power does not change stored energy;
- `sess_snet_def`, `ess_snet_def`, apparent `sch/sdch` variables and associated `kappa` machinery are retired;
- ESSO has no SOC variable and none was invented;
- ESSO degradation throughput uses cell-side active-energy throughput while preserving the existing degradation law/weights;
- every load-bearing coordination path remains based on net `pnet/qnet`.

H1 complementarity is also accepted:

`pch = S_rated * pch_hat`

`pdch = S_rated * pdch_hat`

`pch_hat * pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE`

with:

`ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`.

The same relative semantics apply to shared network ESS, ordinary ESS, ESSO per cohort and ESSO aggregate. The old absolute ESSO complementarity is superseded.

H1 positive-bootstrap results:

- `36/36` DSO;
- `12/12` TSO;
- `3/3` ESSO;
- `0` primary failures;
- `0` recoveries;
- `0` persistent failures;
- `3499` total network IPOPT iterations;
- network mean/median/max about `72.9 / 64 / 185`;
- representative network equality Jacobians full row rank;
- `0` zero-gradient equality rows;
- `0/1728` converter-capability violations;
- `0/1728` network complementarity violations;
- `0/1728` ESSO per-cohort complementarity violations;
- `0/864` ESSO aggregate complementarity violations.

Observed circulation after H1 stays below the current theoretical `sqrt(eps)=1%` allowance. Do not tighten epsilon in D2.

## Accepted coordination decision

The fixed positive-bootstrap candidate converged through the real distributed operational planner in **9 ADMM cycles** with every local solve successful and no recovery diagnostics. ADMM settings were unchanged.

Coordination remains **net electrical P/Q only**.

Do not add consensus variables for:

- `pch`;
- `pdch`;
- circulation;
- SOC;
- cell-energy rate;
- throughput.

At final consensus, DSO-vs-TSO cell-energy-rate disagreement tracked the remaining net-P disagreement at solver-tolerance scale. `pch/pdch` remain local sanity diagnostics only.

## P5.4 global invariants

Unless explicitly authorized by a later planner decision, do not:

- tune IPOPT `tol`, `acceptable_tol`, `acceptable_iter`, `max_iter`, or bound/slack pushes;
- change MA97/MA57 policy;
- broaden recovery classification;
- change ADMM rho rules/tolerances;
- change recourse-stationarity criteria;
- change ADMM objective scaling;
- change TSO proximal regularization;
- change `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- remove or alter H1 dimensionless complementarity;
- normalize converter capability;
- change Benders/local-cut logic during D2;
- add generic feasibility slacks;
- reintroduce `sess_snet_def`/`ess_snet_def`;
- reintroduce P5.2 narrow bands or `kappa` scaling;
- modify RES capability or stochastic samples;
- add calendar degradation;
- include B1 `f_ref=0` in the current baseline.

---

# P5.4-D2 — shared-S/E sensitivity root-cause audit before Benders

## Objective

Determine whether the current dual-based local sensitivities with respect to installed shared-ESS power capacity `S` and energy capacity `E` are mathematically complete and reliable enough for Benders/local cuts.

The previous D audit showed that the shared-S fixing-row dual does not reproduce central finite differences and may disagree in sign. This issue predates P5.4 but remains a live cut-quality risk.

D2 is diagnostic/root-cause work. Do not run the outer planning loop and do not modify Benders equations yet.

## D2.1 — trace the exact sensitivity contract

Trace, from the local network NLP to the outer planner:

- `shared_es_s_rated_fixed`;
- `shared_es_s_rated`;
- `shared_es_e_rated_fixed`;
- `shared_es_e_rated`;
- `shared_energy_storage_s_sensitivities`;
- `shared_energy_storage_e_sensitivities`;
- IPOPT dual extraction;
- local objective weighting/scaling;
- sign conversion;
- aggregation across TSO/DSOs;
- Benders/local-cut coefficient construction.

State exactly what the outer algorithm assumes the returned S and E quantities mean. Derive the intended signs of:

`dQ/dS`

and:

`dQ/dE`.

Do not infer sign conventions only from observed values.

## D2.2 — inventory every direct dependence on S and E

For representative active shared ESS models, enumerate every way a change in installed `S` or `E` changes the NLP. Classify each dependence as:

- **A — symbolic rated-variable dependence:** expression uses `shared_es_s_rated` / `shared_es_e_rated`;
- **B — direct mutable-parameter dependence:** expression uses the fixed mutable capacity parameter directly;
- **C — numerically rewritten variable bound:** lifecycle code updates `lb/ub` as a function of S/E;
- **D — structural/gating dependence:** constraints/variables activate, deactivate, fix or unfix when capacity crosses a threshold.

Produce complete S and E dependency tables. Include H1 hat variables/link rows and zero-capacity lifecycle logic.

## D2.3 — capacity-dependent variable-bound audit

Inspect the production bounds of at least:

- `shared_es_pch`;
- `shared_es_pdch`;
- `shared_es_pnet`;
- `shared_es_qnet`;
- `shared_es_soc`;
- `shared_es_pch_hat`;
- `shared_es_pdch_hat`;
- any additional shared-ESS operational/state variables.

For every bound report:

- whether it depends on S;
- whether it depends on E;
- whether it is mathematically redundant with symbolic constraints;
- whether it is active at the tested optimum.

Extract IPOPT bound multipliers:

- `zL`;
- `zU`.

This is required.

## D2.4 — derive the complete envelope derivative

Starting from the actual Pyomo/IPOPT formulation, derive the parametric value-function derivative for S and E. Separate:

`dQ/dtheta = fixing-row contribution + direct-expression contribution + variable-bound contribution`,

where `theta` is S or E.

Use the actual IPOPT/Pyomo multiplier sign convention. If necessary validate that convention on a trivial one-variable parametric NLP.

For a parameter-dependent upper bound `x <= u(theta)` and lower bound `x >= l(theta)`, derive explicitly how `zU`, `zL`, `du/dtheta` and `dl/dtheta` contribute.

At representative solved models compute separately:

- fixing-row-only derivative;
- total bound contribution;
- direct-expression contribution;
- corrected total derivative.

## D2.5 — controlled finite differences on current H1 HEAD

Re-run S and E finite differences on the current production formulation. Use central differences wherever both perturbations remain in the same positive-capacity structural regime.

Suggested relative perturbations:

- `0.5`;
- `0.25`;
- `0.1`;
- `0.05`;
- `0.02`;
- `0.01`;
- `0.005`;
- `0.001`.

For every perturbation record:

- `Q(theta+h)`;
- `Q(theta-h)`;
- solve termination;
- iterations;
- KKT/feasibility diagnostics available from the normal solve path;
- whether the same apparent local branch/basin was reached;
- central finite difference;
- fixing-row dual prediction;
- corrected derivative prediction from D2.4.

Do not assume the smallest step is best; objective noise and local switching may dominate at tiny h.

## D2.6 — exact sensitivity-clean S-bound reformulation test

If D2.2/D2.3 confirm that physical variable bounds such as:

- `pch <= S`;
- `pdch <= S`;
- `-S <= pnet <= S`;
- `-S <= qnet <= S`

are rewritten numerically from the fixed capacity parameter, first prove whether they are redundant with the symbolic formulation.

In the current active-energy model:

- nonnegative `pch/pdch` plus `pch + pdch <= S` should imply the individual active-power upper bounds;
- `pnet^2 + qnet^2 <= S^2` should imply the net-P/Q box bounds for positive S.

If equivalence is confirmed, create an isolated diagnostic A/B:

- **A — current production:** retain capacity-dependent numerical bounds;
- **B — sensitivity-clean formulation:** remove only mathematically redundant capacity-dependent numerical bounds while retaining nonnegativity, fixed `[0,1]` H1 hat bounds, symbolic active-sum constraint, symbolic converter capability, SOC/energy constraints and zero-capacity gating.

Do not otherwise change the feasible set. Use capacity-independent finite safety bounds only if the NLP writer genuinely requires them; document their derivation.

The purpose is to route S dependence through symbolic constraints and the rated-capacity variable rather than hidden mutable bounds.

## D2.7 — E/SOC analogue

Perform the same analysis for energy capacity E. In particular inspect whether SOC bounds are rewritten numerically from `E_fixed` rather than represented symbolically against `shared_es_e_rated`.

If an exact symbolic form can replace a parameter-dependent variable bound without changing SOC semantics, test that reformulation independently.

Do not change SOC fractions, day balance, efficiencies or energy semantics.

## D2.8 — sensitivity-clean validation

For any exact A/B reformulation from D2.6/D2.7 compare:

- objective;
- `pnet/qnet`;
- `pch/pdch`;
- SOC;
- generator dispatch;
- interface quantities;
- constraint residuals;
- IPOPT iterations.

Require physical/economic equivalence to numerical tolerance. Then repeat the S/E finite-difference validation.

The key question is whether the rated-capacity fixing-row dual becomes a complete derivative once hidden capacity-dependent numerical bounds are removed, or whether explicit bound/direct terms must still be included.

## D2.9 — active-set/local-branch stability

For every finite-difference pair report changes in at least:

- active converter-capability rows;
- active `pch+pdch<=S` rows;
- active SOC limits;
- complementarity rows near their bound;
- active generator/voltage/branch constraints.

Flag perturbations that cross active-set or local-optimum transitions as unsuitable for smooth derivative validation. Seek at least one perturbation window with a stable active set and reproducible local branch.

## D2.10 — broader validation population

Do not conclude from a single DSO/day. Validate the final candidate sensitivity treatment on at least:

- one DSO at node 5;
- one DSO at node 7;
- one DSO at node 9;
- one TSO case;
- multiple years/days;
- at least one case where S is operationally binding;
- at least one case where E is operationally binding.

Report S and E separately.

## D2 production decision

Do not modify Benders/local-cut equations in this stage.

If the current dual is incomplete because of parameter-dependent variable bounds and an exact sensitivity-clean local formulation fixes the issue, recommend productionizing that **local formulation correction** first.

If the complete envelope derivative including bound/direct terms matches finite differences but the existing Benders extraction uses only the fixing-row dual, report the exact additional terms required by Benders for planner review.

If neither explanation resolves the mismatch, identify whether the remaining cause is:

- local-optimum switching;
- insufficient local solve accuracy relative to the objective perturbation;
- complementarity/active-set nonsmoothness;
- another hidden parameter dependence.

Do not tune IPOPT to force agreement.

D2 must end with exactly one:

`P5.4-D2 PASS — S/E sensitivities validated for Benders`

`P5.4-D2 PARTIAL — sensitivity root cause identified but production correction still required`

`P5.4-D2 FAIL — sensitivity remains unexplained`

Then stop with:

`P5.4-D2 COMPLETE — ready for planner review before reduced planning`

---

# P5.4-G — reduced planning gate

Run reduced `run_planning_problem()` only after planner review explicitly authorizes it based on D2.

Preserve current Benders/positive-bootstrap settings. Do not tune Benders in response to a failure during this gate.

Record:

- outer iterations;
- candidate sequence;
- initialization failures;
- ADMM failures;
- local NLP primary/recovery outcomes;
- S/E sensitivities and cuts;
- incumbent evolution;
- termination reason;
- complementarity/circulation diagnostics.

Stop and report any new behavior rather than stacking another fix.

---

# Deferred physical complementarity-tolerance decision

The normalized `eps=1e-4` condition is numerically enforced and accepted for the current baseline. Whether its theoretical `sqrt(eps)=1%` circulation allowance is physically too loose is a separate modelling question.

Do not test `1e-5` / `1e-6` during D2. A later isolated one-change A/B may test those values with the H1 formulation held fixed.

---

# Current required report

Continue:

`P5_4_ACTIVE_ENERGY_ESS_PRODUCTIONIZATION_REPORT.md`.

For the next checkpoint add:

`D2 — shared-S/E sensitivity root-cause audit`.

Do not issue the final P5.4 verdict and do not run G.

Finish D2 with the required D2 verdict and:

`P5.4-D2 COMPLETE — ready for planner review before reduced planning`

then stop.

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
