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

# CURRENT AUTHORIZED STAGE — P5.4 H1/F

P5.4-A/B/C/D/E/E2 are complete. The current authorized implementation stage is now **P5.4-H1 — dimensionless complementarity**, followed only if it passes by **P5.4-F — live net-P/Q ADMM**.

The historical P5.3 material later in this file remains evidence only.

## Current production/evidence checkpoints

Production active-energy stack:

- `a4a0bae8` — shared network active-energy ESS;
- `1e86d40e` — ordinary network ESS parity;
- `58f4911b` — ESSO active-energy conversion;
- `c3526ec8` — lifecycle/sensitivity audit state and final pre-ADMM production validation.

Diagnostic evidence:

- `b0e53bc4` — E2 complementarity-significance diagnostics; no formulation change.

Reduced-scenario identity remains:

- seed `2026`;
- combined checksum `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

## Completed P5.4 results — authoritative

The final post-B/C/D positive-bootstrap production validation gives:

- `36/36` DSO success;
- `12/12` TSO success;
- `3/3` ESSO success;
- `0` primary failures;
- `0` recoveries;
- `0` persistent failures;
- `1556` total network IPOPT iterations;
- network mean/median/max about `32.4 / 28.5 / 119`;
- representative network equality Jacobians full row rank;
- `0` zero-gradient equality rows from the ESS formulation;
- `0/1728` shared-network converter-capability violations.

The active-energy physical correction is productionized:

- shared and ordinary network SOC use active `pch/pdch`;
- pure reactive power does not change network ESS SOC;
- `sess_snet_def`, `ess_snet_def`, their apparent `sch/sdch` variables, and their `kappa` machinery are retired;
- ESSO has no SOC variable and none was invented;
- ESSO degradation throughput now uses cell-side active-energy throughput while preserving the existing degradation law/weights;
- every load-bearing coordination path remains based on net `pnet/qnet`.

## E2 complementarity evidence — authoritative

The physical circulation metric is:

`p_circ = min(pch, pdch)`.

Do not infer directional circulation from `sqrt(pch*pdch)`.

Network shared-ESS E2 results:

- maximum `p_circ/S = 0.11562`;
- DSO maximum about `0.01669`;
- TSO maximum about `0.11562`;
- `694/1728` rows exceed `1e-2*S` circulating power;
- worst representative-day artificial circulation loss about `3.0%` of `E_rated`;
- worst representative-day circulation loss about `3.4%` of legitimate throughput.

Network complementarity semantics are relative:

`pch*pdch <= ESS_COMPLEMENTARITY_TOLERANCE*S^2`.

The current value remains:

`ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`.

At bootstrap S this physical RHS is numerically tiny compared with IPOPT's absolute feasibility scale.

ESSO has a pre-existing inconsistency: its per-cohort complementarity is currently absolute (`pch*pdch <= 1e-4`) rather than relative to installed power. It is effectively vacuous at bootstrap capacities; aggregate ESSO circulation reaches about `0.1912*S`.

## Coordination decision

ADMM continues to coordinate **net electrical P/Q only**.

Do not add consensus variables for:

- `pch`;
- `pdch`;
- circulation;
- SOC;
- cell-energy rate;
- throughput.

With tight local complementarity, `pnet = pch - pdch` determines charge/discharge direction to numerical accuracy. `pch/pdch` remain local physical variables and may be logged as diagnostics.

## P5.4 global invariants

Unless explicitly authorized by a later planner decision, do not:

- tune IPOPT `tol`, `acceptable_tol`, `acceptable_iter`, `max_iter`, or bound/slack pushes;
- change MA97/MA57 policy;
- broaden recovery classification;
- change ADMM rho rules/tolerances;
- change recourse-stationarity criteria;
- change ADMM objective scaling;
- change TSO proximal regularization;
- change Benders/local-cut logic;
- add generic feasibility slacks;
- reintroduce `sess_snet_def`/`ess_snet_def`;
- reintroduce P5.2 narrow bands or `kappa` scaling;
- modify RES capability or stochastic samples;
- add calendar degradation;
- include B1 `f_ref=0` in the current baseline.

For H1 specifically:

- **do not change the numerical value** `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- dimensionless variables are authorized **for charge/discharge complementarity only**;
- do not normalize converter capability because its audited physical violation is zero;
- do not add new complementarity objective penalties or retune existing coefficients.

---

# P5.4-H1 — dimensionless charge/discharge complementarity

## Objective

Make the existing relative complementarity tolerance numerically resolvable without physically tightening it.

Separate two questions:

1. Can IPOPT enforce the current `1e-4` relative complementarity when it is written at O(1) scale?
2. If it can, is that physical tolerance itself sufficiently tight?

H1 answers question 1 only.

## H1.1 — shared network ESS

For every positive-capacity active shared ESS introduce dimensionless nonnegative internal variables:

`shared_es_pch_hat`

`shared_es_pdch_hat`

with nominal bounds `[0,1]`.

Link them to the existing physical active powers with equalities of the form:

`shared_es_pch - shared_es_s_rated * shared_es_pch_hat = 0`

`shared_es_pdch - shared_es_s_rated * shared_es_pdch_hat = 0`.

Keep the physical variables `shared_es_pch/shared_es_pdch`; they remain the quantities used by SOC, `pnet`, objectives, results, diagnostics and sensitivity logic.

Replace only the complementarity inequality:

old:

`shared_es_pch * shared_es_pdch <= eps * shared_es_s_rated^2`

new:

`shared_es_pch_hat * shared_es_pdch_hat <= eps`

with:

`eps = ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`.

For positive capacity this is an exact reformulation of the same relative feasible set.

Keep unchanged:

- `pnet = pch - pdch`;
- SOC;
- `pch + pdch <= S`;
- physical variable bounds;
- converter capability;
- PF/reactive rows;
- physical objective terms.

Do not replace usage/complementarity objective terms with normalized quantities in H1.

## H1.2 — zero-capacity and reused-model lifecycle

Do not divide by `S` anywhere.

At zero/inactive capacity:

- fix `pch_hat = 0`;
- fix `pdch_hat = 0`;
- retain existing physical-variable zero-capacity gating;
- deactivate or safely configure the normalized complementarity row consistently with the existing lifecycle.

Test in place on one reused model:

- zero -> positive;
- positive -> different positive;
- positive -> zero;
- zero -> positive again.

Track component identity and bounds/fixed status.

## H1.3 — derivative/rank audit

Before solving representative DSO and TSO models, report:

- link-row Jacobian norms;
- complementarity-row gradient norm/margin at the cold start;
- relevant second derivatives;
- zero-gradient equality-row count;
- `sigma_min(full)` and row rank on representative equality Jacobians.

Required structural properties:

- link equalities retain coefficient `+1` on physical `pch/pdch`, so they are not zero-gradient at zero dispatch;
- no exact equality-rank defect is reintroduced;
- normalized complementarity RHS is `1e-4`, rather than O(`S^2`);
- normalized complementarity curvature is O(1).

A zero gradient on the normalized complementarity inequality at `pch_hat=pdch_hat=0` is acceptable because the row is strictly interior by `1e-4`; distinguish it from the old zero-gradient equality defect.

## H1.4 — ordinary network ESS parity

Apply the same complementarity semantics to ordinary ESS:

`es_pch - S_rated * es_pch_hat = 0`

`es_pdch - S_rated * es_pdch_hat = 0`

`es_pch_hat * es_pdch_hat <= eps`.

Here `S_rated` is fixed network data, but use the same relative semantics for consistency.

Re-run the OP1 validation used in P5.4-B.

Require:

- successful primary solve;
- no converter-capability violation;
- normalized complementarity enforced;
- no material iteration regression relative to the accepted active-energy OP1 result (`144` iterations), allowing ordinary solver variation.

## H1.5 — ESSO per-cohort complementarity

First trace the exact per-cohort installed-power quantity and its units. Do not guess the cohort rating.

Replace the current absolute per-cohort complementarity with the same **relative** semantics used by network agents.

For every active cohort, introduce dimensionless internal variables linked to the cohort physical active powers using the actual cohort installed power:

`pch_cohort - S_cohort * pch_hat_cohort = 0`

`pdch_cohort - S_cohort * pdch_hat_cohort = 0`

and enforce:

`pch_hat_cohort * pdch_hat_cohort <= ESS_COMPLEMENTARITY_TOLERANCE`.

Preserve the numerical value `1e-4`.

This is an authorized semantic consistency correction: the previous absolute `1e-4` condition is superseded.

Do not modify degradation coefficients or throughput weighting.

## H1.6 — ESSO aggregate complementarity

Per-cohort complementarity alone permits one cohort to charge while another cohort discharges at the same node/time.

The network agent represents one aggregate shared ESS and enforces complementarity on aggregate charge/discharge. The ESSO aggregate feasible set must therefore be compatible.

For each shared-ESS node/year/time define/identify:

`pch_agg = sum_active_cohorts(pch_cohort)`

`pdch_agg = sum_active_cohorts(pdch_cohort)`.

Use the actual aggregate installed power:

`S_total = sum/production-defined aggregate installed power for the active cohorts`.

Introduce aggregate dimensionless variables through link equalities with a unit coefficient on the physical aggregate expression/variable, and enforce:

`pch_hat_agg * pdch_hat_agg <= ESS_COMPLEMENTARITY_TOLERANCE`.

Use the correctly spelled production constant `ESS_COMPLEMENTARITY_TOLERANCE`; the line above states the mathematics only.

Do not invent a new aggregate rating or oversizing factor.

Preserve per-cohort complementarity as well unless a separate proof demonstrates it is redundant and planner review authorizes removal.

## H1.7 — objective policy

Do not tune objectives in H1.

Keep physical usage/complementarity penalty terms expressed with physical active powers and existing coefficients.

The normalized **hard constraint** is responsible for enforcing complementarity.

Report the objective contribution of the existing complementarity penalty only as a diagnostic.

## H1.8 — production positive-bootstrap gate

After implementing H1 consistently across shared network ESS, ordinary ESS and ESSO, rerun the exact production positive-bootstrap population.

Require:

- `36/36` DSO;
- `12/12` TSO;
- `3/3` ESSO;
- zero persistent failures.

Report:

- primary/recovery split;
- total/mean/median/max iterations;
- runtime;
- objectives;
- representative equality rank/singular values;
- converter-capability residuals.

Compare to the accepted pre-H1 active-energy baseline:

- `1556` total network iterations;
- about `32 s` runtime;
- zero failures.

A modest numerical-cost increase is acceptable if complementarity becomes physically enforced. A large regression or new failure family fails the gate.

## H1.9 — complementarity acceptance metrics

For every network, ordinary-ESS and ESSO complementarity population report:

- `pch_hat`;
- `pdch_hat`;
- `pch_hat*pdch_hat`;
- `max(pch_hat*pdch_hat - eps, 0)`;
- physical `min(pch,pdch)/S`.

For shared network agents and ESSO aggregate report:

- max;
- mean;
- median;
- p95;
- p99;
- count `>1e-3`;
- count `>1e-2`.

With `eps=1e-4`, exact enforcement implies that equal-direction simultaneous charging/discharging cannot exceed:

`sqrt(eps) = 1e-2` of rating.

Verify the actual physical circulation empirically rather than assuming equality of directional powers.

Recompute artificial circulation loss:

`E_circ_loss = min(pch,pdch) * dt * (1/eta_dch - eta_ch)`

and report:

- MWh;
- fraction of `E_rated`;
- fraction of legitimate throughput.

Compare with E2:

- network max `p_circ/S = 0.11562`;
- ESSO aggregate max about `0.1912`.

## H1.10 — physical-tolerance decision is deferred

Do not reduce epsilon in H1.

If the normalized row is enforced with negligible violation but circulation approaches the permitted ~1% scale and that is still physically unacceptable, recommend a later isolated physical-tolerance A/B (`1e-5`, `1e-6`, etc.).

Do not perform that A/B automatically.

## H1 pass gate

H1 passes only if:

1. all production positive-bootstrap local solves succeed;
2. no equality-rank defect is introduced;
3. normalized network complementarity is solver-resolved;
4. ESSO per-cohort complementarity uses the same relative semantics and is solver-resolved;
5. ESSO aggregate complementarity is solver-resolved;
6. physical `min(pch,pdch)/S` is consistent with the `sqrt(eps)=1%` allowance plus small numerical error;
7. converter capability remains clean;
8. no end-to-end active-energy semantic inconsistency is introduced.

If H1 fails, stop before F and report.

---

# P5.4-F — live distributed ADMM with net P/Q coordination only

Run F **only if H1 passes**.

Run the exact positive-bootstrap candidate through live distributed ADMM.

Do not run the outer planning loop.

Keep unchanged:

- seed/scenarios;
- IPOPT options;
- MA97/exact-Hessian policy;
- recovery policy;
- ADMM rho values/adaptation;
- ADMM tolerances;
- proximal regularization;
- objective scaling.

## Standard instrumentation

Record every cycle:

- cycle number;
- rho values;
- V primal max/mean and dual mean;
- P/Q-flow primal max/mean and dual mean;
- shared-ESS net-P/Q primal max/mean and dual mean;
- recourse-stationarity metrics;
- local solver status, recovery use and iterations;
- recourse/objective evolution.

## Complementarity sanity diagnostics — not consensus

At each cycle and at convergence record for DSO, TSO and ESSO aggregate:

- `pch`;
- `pdch`;
- `pnet`;
- `min(pch,pdch)/S`;
- normalized complementarity product/violation.

For ESSO also retain per-cohort diagnostics.

These are not ADMM residuals and must not be fed into consensus updates.

## Energy-consistency sanity diagnostic

At final electrical consensus compute locally:

`p_cell = eta_ch*pch - pdch/eta_dch`.

Compare DSO/TSO/ESSO for the same shared ESS only as a diagnostic.

With tight complementarity and net-P agreement, charge/discharge direction should agree automatically. A large `p_cell` disagreement despite tight local complementarity indicates an implementation inconsistency.

## F stopping rules

- If a new reproducible persistent local-NLP failure appears, stop and report it.
- If ADMM converges, complete the physical sanity diagnostics and stop for planner review.
- If the configured maximum cycles are reached, report the last-cycle diagnostics.

Do not add `pch/pdch` consensus in response to an F issue without planner review.

F verdict:

`P5.4-F ADMM PASS — net-P/Q coordination converged with locally consistent charge/discharge`

or:

`P5.4-F ADMM PARTIAL — electrical coordination converged but a local physical inconsistency remains`

or:

`P5.4-F ADMM FAIL — distributed coordination did not converge robustly`.

---

# P5.4-D2 — shared-S sensitivity root-cause audit before Benders

Outer planning remains blocked even if F passes.

P5.4-D found that the analytic shared-S capacity sensitivity does not converge to the finite-difference estimate and can change sign. The issue predates P5.4 but remains a risk to Benders cut quality.

Before P5.4-G, perform a targeted audit of all dependence on shared power capacity `S`.

Explicitly inventory whether S enters through:

- the rated-capacity fixing equality;
- converter-capability RHS;
- active-sum limit RHS;
- dimensionless H1 linking equalities;
- physical variable upper/lower bounds updated by lifecycle code;
- zero-capacity activation/deactivation logic;
- objective terms;
- any mutable Params or expressions;
- ESSO/network coupling and sensitivities.

Determine whether the current analytic sensitivity based on the rated-capacity fixing-row dual captures **all** local value-function dependence on S. In particular, mutable variable bounds whose numerical limits are directly changed with S may contribute sensitivity that is not represented by that single dual.

Use targeted finite differences only for this sensitivity validation. Control for local-optimum switching by:

- using symmetric perturbations where feasible;
- reporting solve status/objective/KKT diagnostics;
- comparing warm and cold perturbation probes only if necessary and without changing production policy;
- checking multiple perturbation sizes.

Do not change Benders/local-cut logic in D2. First establish the cause.

D2 must end with one of:

- `S sensitivity validated`;
- `S sensitivity formula incomplete — concrete missing dependence identified`;
- `S sensitivity indeterminate because local value function is not smooth/reproducible at this point`.

---

# P5.4-G — reduced planning gate

Run reduced `run_planning_problem()` only if:

1. H1 passes;
2. F is sufficiently clean;
3. D2 establishes a trustworthy capacity-sensitivity path or a planner-approved alternative.

Preserve current Benders/positive-bootstrap settings.

Record:

- outer iterations;
- candidate sequence;
- initialization failures;
- ADMM failures;
- local NLP primary/recovery outcomes;
- capacity sensitivities and cuts;
- incumbent evolution;
- termination reason;
- complementarity/circulation diagnostics.

Do not tune Benders in response to a failure during this gate. Stop and report new behavior.

---

# Later complementarity-tolerance decision

H1 keeps `eps=1e-4` fixed. If H1/F demonstrate that this tolerance is numerically enforced but its physical allowance remains too loose, prepare a separate one-change A/B on epsilon.

Do not mix physical epsilon tightening with dimensionless scaling or ADMM tuning.

---

# Current required report

Continue:

`P5_4_ACTIVE_ENERGY_ESS_PRODUCTIONIZATION_REPORT.md`.

For the next checkpoint add:

- H1 implementation and validation;
- F only if H1 passes.

Do not issue the final P5.4 verdict before the sensitivity/Benders gate is reviewed.

For H1/F checkpoint finish with:

`P5.4 H1/F CHECKPOINT COMPLETE — ready for planner review before Benders`

and stop.

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
