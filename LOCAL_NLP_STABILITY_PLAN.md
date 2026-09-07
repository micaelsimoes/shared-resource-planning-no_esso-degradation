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

# CURRENT AUTHORIZED STAGE — P5.4-D2-P productionization, then P5.4-D3 cut-consistency audit

P5.4-A/B/C/D/D2/E/E2/H1/F are complete. H1 and F are accepted. D2 ended:

`P5.4-D2 PARTIAL — sensitivity root cause identified but production correction still required`

The current authorized sequence is:

1. **P5.4-D2-P — productionize the exact sensitivity-clean shared-S formulation**;
2. **P5.4-D3 — audit the actual distributed Benders/local-cut coefficient against nearby observed recourse branches**;
3. stop for planner review;
4. **do not run P5.4-G** unless explicitly authorized after D3.

## Current production/evidence checkpoints

Production active-energy stack:

- `a4a0bae8` — shared network active-energy ESS;
- `1e86d40e` — ordinary network ESS parity;
- `58f4911b` — ESSO active-energy conversion;
- `c3526ec8` — lifecycle/sensitivity audit state;
- `93974d83` — normalized dimensionless complementarity across network ESS, ordinary ESS and ESSO.

Accepted live coordination evidence:

- `2917b9c9` — fixed positive-bootstrap distributed ADMM; net-P/Q coordination only; converged in 9 cycles with no local failures/recoveries.

Latest D2 diagnostic evidence:

- `65b261ba` — S/E sensitivity root-cause audit; diagnostic only, no production or Benders change.

Reduced-scenario identity remains:

- seed `2026`;
- checksum `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

## Accepted operational invariants

Keep unchanged unless a later planner instruction explicitly authorizes otherwise:

- active-energy ESS SOC/throughput physics;
- H1 normalized complementarity;
- `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only ADMM coordination;
- current ADMM rho/tolerances, proximal regularization and recourse-stationarity logic;
- IPOPT/MA97/exact-Hessian and current recovery policy;
- converter-capability formulation;
- objective scaling and all objective coefficients;
- current RES formulation/data;
- B1 `f_ref=0` remains deferred.

Do not add `pch`, `pdch`, circulation, SOC, cell-energy rate or throughput as ADMM consensus variables.

---

# Accepted P5.4-D2 findings

D2 established two independent issues.

## D2 root cause 1 — incomplete S derivative bookkeeping

For positive shared-ESS capacity, production numerically rewrites bounds on:

- `shared_es_pch`: `[0,S]`;
- `shared_es_pdch`: `[0,S]`;
- `shared_es_pnet`: `[-S,S]`;
- `shared_es_qnet`: `[-S,S]`.

The local NLP also contains symbolic S-dependence through the rated-S Var and its fixing row. Production Benders currently extracts only the fixing-row multiplier.

The calibrated IPOPT/Pyomo sign convention is:

`dual[param == var] = dQ/dparam`

and for numerical bounds:

`dQ/du = zU`

`dQ/dl = zL`.

Therefore the complete local derivative is:

`dQ/dS = fixing-row contribution + bound contribution + direct-expression contribution`.

D2 found no extra active fixed-parameter expressions outside the fixing row, but the bound contribution is substantial and systematic. The fixing-row-only quantity is not a complete S derivative and can even have the wrong sign relative to S-feasible-set monotonicity.

For E, no E-scaled operational variable bound exists. E dependence is already symbolic through `shared_es_e_rated`; the E bound contribution was zero throughout the audit. Do not invent an E reformulation without new evidence.

## D2 root cause 2 — reproducible local-solution multiplicity

Nearby S/E candidates can converge to several deterministic local branches. The branch gap is larger than the first-order capacity effect being measured. Re-solving an identical problem is bit-identical, so this is not random solver noise.

Consequences:

- finite differences cannot presently certify the local dual against one unique nearby value function;
- continuation/warm starts do not reliably keep the perturbed candidate on the same branch as the base cold solve;
- after the S bookkeeping correction, the fixing-row dual should be treated as the complete derivative **of the returned local branch**, not automatically as a globally valid Benders subgradient.

---

# P5.4-D2-P — sensitivity-clean shared-S productionization

## D2-P objective

Productionize D2.6 variant B, which is an exact feasible-set-preserving reformulation for positive S.

Remove only the positive-capacity numerical S-dependent bounds from the lifecycle for:

- `shared_es_pch` upper bound;
- `shared_es_pdch` upper bound;
- `shared_es_pnet` lower/upper bounds;
- `shared_es_qnet` lower/upper bounds.

Do not add bound-multiplier corrections to Benders.

## D2-P retained formulation

Keep:

- `shared_es_pch >= 0`;
- `shared_es_pdch >= 0`;
- `shared_es_pch_hat, shared_es_pdch_hat in [0,1]`;
- `pch = S_rated*pch_hat`;
- `pdch = S_rated*pdch_hat`;
- `pch + pdch <= S_rated`;
- `pnet = pch - pdch`;
- `pnet^2 + qnet^2 <= S_rated^2`;
- active-energy SOC rows;
- H1 normalized complementarity;
- PF/reactive rows;
- all objective terms and coefficients;
- explicit zero-capacity fixing/gating.

At zero capacity the physical variables must still be explicitly fixed to zero. Do not divide by S anywhere.

## D2-P.1 — exact redundancy proof/tests

Document and test:

- `pch <= S` follows from `pch,pdch >= 0` plus `pch+pdch<=S`; it also follows from `pch=S*pch_hat`, `pch_hat<=1`;
- `pdch <= S` analogously;
- `|pnet|<=S` and `|qnet|<=S` follow from `pnet^2+qnet^2<=S^2` for positive S.

The production correction must not change the mathematical feasible set.

## D2-P.2 — lifecycle/reused-model validation

Test in place:

- zero -> positive;
- positive -> different positive;
- positive -> zero;
- zero -> positive again.

Verify:

- explicit zero-capacity fixing remains safe;
- positive-capacity variables are free with no hidden S-dependent numerical bounds;
- symbolic constraints/H1 links keep the positive-capacity model bounded;
- component identity is preserved across reuse;
- warm-start suffix handling remains intact.

## D2-P.3 — sensitivity-structure audit

Across at least the original eight D2 cases confirm:

- no positive-capacity numerical bound on `pch/pdch/pnet/qnet` depends on S;
- no active expression uses `shared_es_s_rated_fixed` outside the fixing row;
- S bound contribution is exactly zero;
- the fixing-row dual is therefore the structurally complete **local branch** derivative with respect to S;
- E remains structurally clean and needs no bound cleanup.

Do not claim global Benders validity from this structural result alone.

## D2-P.4 — operational regression gate

Re-run the production positive-bootstrap initialization.

Require:

- DSO `36/36`;
- TSO `12/12`;
- ESSO `3/3`;
- zero persistent failures;
- no new recovery/failure family;
- H1 complementarity still physically enforced;
- converter-capability residuals remain clean;
- equality-rank diagnostics remain acceptable.

Then rerun the accepted fixed-candidate distributed ADMM through the real production path.

Require robust convergence with:

- net-P/Q coordination unchanged;
- no new local failure family;
- normalized complementarity satisfied;
- converter capability satisfied;
- no `pch/pdch` consensus added.

Do **not** run the outer planning loop.

D2-P verdict must be:

`P5.4-D2-P PASS — structurally complete S sensitivity productionized`

or an explicit failure/partial status with the blocking evidence.

---

# P5.4-D3 — distributed cut-consistency / local-branch audit

Run D3 only if D2-P passes.

## D3 objective

Test the quantity the planning algorithm actually uses: the fully aggregated distributed recourse value and production cut coefficient.

The central question is:

**Does the dual generated by the actual distributed recourse solve produce a safe and useful local cut at nearby candidates, despite the observed nonconvex branch multiplicity?**

Do not alter the master/Benders equations during D3.

## D3.1 — base distributed recourse and exact production coefficient

At the exact positive-bootstrap candidate:

1. run `run_operational_planning(type='distributed', ...)` to convergence;
2. record total recourse `Q0`, gross operational cost and terminal salvage;
3. extract the exact production S/E coefficients after:
   - local objective rescaling;
   - day/year weighting;
   - TSO + DSO aggregation;
   - available-capacity -> investment-capacity mapping;
   - salvage-value sensitivity;
4. persist every contribution by node, year, S/E and agent.

Call the resulting investment-space sensitivity vector `g0`.

## D3.2 — construct, but do not add, the candidate cut

Construct:

`L(x) = Q0 + g0^T (x - x0)`.

Verify units/signs independently. Do not add the cut to the master problem.

## D3.3 — nearby candidate population

Generate controlled feasible perturbations around the positive-bootstrap investment candidate.

One variable at a time first:

- S at nodes 5, 7, 9;
- E at nodes 5, 7, 9.

Use relative perturbations such as:

- `+/-0.5%`;
- `+/-1%`;
- `+/-2%`;
- `+/-5%`;
- `+/-10%`.

Respect cumulative investment/available-capacity mapping and do not cross the zero-capacity structural boundary. Then test a small controlled set of multi-variable perturbations.

## D3.4 — solve every perturbed candidate through the real distributed path

For every candidate run production distributed operational planning with unchanged settings.

Record:

- ADMM convergence and cycles;
- every local primary/recovery outcome;
- recourse;
- gross operational objective;
- terminal salvage;
- extracted S/E sensitivities;
- local objective/branch signatures needed to diagnose branch switching.

Do not invoke the master/planning loop.

## D3.5 — controlled multiple-start lower-recourse probe

Because D2 proved local-solution multiplicity, evaluate each candidate from at least:

- **A — production initialization**;
- **B — continuation/warm start from a converged neighboring candidate**, where technically safe.

If inexpensive and deterministic, also use:

- **C — one additional feasible deterministic initialization** derived from a known feasible solution, not random noise.

For each candidate retain:

`Q_best_observed = min(Q_A, Q_B, Q_C)`

among converged feasible runs.

This is a falsification benchmark only. Never call it the global optimum.

## D3.6 — cut-safety falsification test

Evaluate:

`cut_gap = Q_best_observed - L(x)`.

For the current minimization lower-cut convention:

- `cut_gap < -tol_cut` proves that the generated affine cut lies above an observed feasible recourse value and is therefore demonstrably unsafe;
- `cut_gap >= -tol_cut` does **not** prove global validity because `Q_best_observed` is still local.

Derive `tol_cut` from observed identical-candidate objective repeatability, operational solve accuracy and recourse scaling. Do not choose it arbitrarily.

## D3.7 — local linearity / branch classification

For every perturbation report:

- `Q_observed - Q0`;
- `g0^T(x-x0)`;
- absolute error;
- relative error where meaningful;
- apparent branch identity.

Separate:

- same-branch evaluations;
- branch-switching evaluations.

This distinguishes derivative quality from branch-selection effects.

## D3.8 — S monotonicity sanity check

For each S sweep, tabulate all observed recourse branches and their lower observed envelope.

Increasing S enlarges the physical feasible set, so the **best observed lower envelope** should not systematically increase when the lower branch is successfully recovered.

If it does, state that the global/lower branch has not been reliably recovered. Do not reinterpret an individual local branch's positive slope as physical non-monotonicity.

## D3.9 — required classification

End D3 with exactly one:

`P5.4-D3 PASS — current distributed sensitivities are suitable for the reduced planning gate`

`P5.4-D3 HEURISTIC — sensitivities are locally useful but global cut validity is not established`

`P5.4-D3 FAIL — current cuts are demonstrably unsafe under observed recourse branches`

If D3 is HEURISTIC or FAIL, do not modify Benders automatically. Propose the smallest next planning-level remedy for planner review. Possible directions include:

- trust-region/local cuts;
- cut activation only near the generating candidate;
- deterministic branch-continuation policy;
- multi-start recourse selection;
- planning-capacity movement regularization;
- another decomposition framework appropriate for nonconvex recourse.

Do not assume classical convex Benders guarantees apply to the current nonlinear recourse.

---

# P5.4-G — blocked pending D3 planner review

Do not invoke `run_planning_problem()` during D2-P or D3.

G may run only after explicit planner authorization based on D3. A structurally complete S dual by itself is **not** sufficient; the distributed cut behavior in the observed nonconvex branch landscape must also be reviewed.

---

# Deferred items

- Physical complementarity tolerance `1e-5` / `1e-6`: defer; keep H1 `1e-4` baseline fixed during D2-P/D3.
- B1 exact `f_ref=0`: deferred.
- RES B2-R: deferred until a defensible converter `Smax` exists.
- Calendar degradation: deferred.

---

# Current required report

Continue:

`P5_4_ACTIVE_ENERGY_ESS_PRODUCTIONIZATION_REPORT.md`.

Add:

- `D2-P — sensitivity-clean productionization`;
- `D3 — distributed cut-consistency audit`.

Do not issue the final P5.4 verdict and do not run G.

Finish with:

`P5.4-D2-P/D3 COMPLETE — ready for planner decision on P5.4-G`

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
