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

# CURRENT AUTHORIZED STAGE — P5.4

P5.3-B is complete. P5.4 is now the only authorized active local-NLP/ESS implementation stage unless a later planner instruction says otherwise.

The accepted production checkpoint at the start of P5.4 remains:

`f77d829359ffd873367f556882546bc2dcc8ec99`

The P5.3-B3 active-power shared-ESS formulation is the **approved production direction**, but the diagnostic wrapper itself is not production code. P5.4 must implement and validate the semantics end to end before a new production checkpoint is accepted.

Current reduced-scenario identity:

- seed `2026`;
- combined scenario checksum:
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

## Authoritative P5.3-B decisions

### B1 — exact reference angle

`f_ref = 0` removed the gauge cleanly but relocated the three bootstrap failures rather than reducing their count.

Decision:

`CONTINUE TESTING — do not productionize yet`.

Do not include B1 in the initial P5.4 baseline. Retest it only after the active-energy ESS production baseline is stable.

### B2-R — RES capability semantics

The current RES `sg_capability` conflates stochastic active availability with apparent-power capability, but the repository has no defensible converter MVA rating.

Decision:

`DEFER — insufficient physical rating data for safe reformulation`.

Do not alter RES capability or copula behavior during P5.4.

Future RES work requires an explicit converter `Smax`/`s_rated` field. Reactive-only/STATCOM behavior is a separate modelling decision because the existing PF cone forces `qg = 0` when `pg = 0`.

### B3 — active-power ESS

Decision:

`PRODUCTIONIZE CANDIDATE — approved production direction`.

B3 established:

- DSO zero-gradient equality rows `24 -> 0`;
- TSO zero-gradient equality rows `72 -> 0`;
- full equality Jacobian becomes full row rank;
- positive-bootstrap network primary failures `3 -> 0`;
- persistent failures `3 -> 0`;
- total network IPOPT iterations `33073 -> 1545`;
- worst iteration count `3000 -> 109`;
- runtime about `274 s -> 37 s`;
- pure Q gives exactly zero SOC change;
- active charging/discharging SOC behavior is physically correct.

The P5.2 narrow-band workaround and further `sess_snet_def`/`kappa` tuning are abandoned.

## P5.4 global invariants

Do not during P5.4 unless explicitly authorized:

- tune IPOPT `tol`, `acceptable_tol`, `acceptable_iter`, `max_iter`, or bound/slack pushes;
- change MA97/MA57 policy;
- broaden recovery classification;
- change ADMM rho rules/tolerances;
- change recourse-stationarity criteria;
- change ADMM objective scaling;
- change TSO proximal regularization;
- change Benders/local-cut logic;
- add generic feasibility slacks;
- productionize the P5.2 shared-ESS narrow band;
- add a new arbitrary `kappa` or row multiplier to the new ESS inequalities;
- change `ESS_COMPLEMENTARITY_TOLERANCE`;
- modify RES capability or stochastic samples;
- add calendar degradation;
- modify terminal salvage unless required solely to preserve existing active-energy semantics and explicitly approved.

Work in small checkpoints. Do not bundle unrelated changes.

---

# P5.4-A — productionize shared network ESS

Implement the accepted B3 shared-network formulation in production.

## Remove/retire from the shared network NLP

Retire the shared-network apparent charge/discharge representation:

- `shared_es_sch`;
- `shared_es_sdch`;
- `sess_snet_def`;
- `sess_pch_link`;
- `sess_pdch_link`;
- old `sess_s_limit`;
- old apparent-power-based `sess_soc_def`;
- old `sess_comp`.

Also retire production machinery whose only purpose is `sess_snet_def` scaling:

- `sess_snet_def_kappa`;
- `shared_ess_snet_def_scale`;
- `_sync_sess_snet_def_scale`;
- any dual/multiplier transfer logic specific only to that row.

Do not remove unrelated warm-start handling.

Preserve zero-capacity gating and reused-model lifecycle behavior.

## Keep and implement

Keep the canonical load-positive convention:

`pnet = pch - pdch`.

SOC:

`SOC_t = SOC_(t-1) + eta_ch * pch * dt - pdch * dt / eta_dch`.

Use an explicit/reusable production time-step convention. For current 24-hour representative days verify `dt = 1 h` exactly.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`.

Active envelope derived from the production feasible set:

`pch + pdch <= S_rated`.

Use appropriate redundant explicit bounds where helpful and exact:

- `0 <= pch <= S_rated`;
- `0 <= pdch <= S_rated`;
- `-S_rated <= pnet <= S_rated`;
- `-S_rated <= qnet <= S_rated`.

Complementarity:

`pch * pdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2`.

Preserve `ESS_COMPLEMENTARITY_TOLERANCE` exactly.

Preserve existing PF/reactive-power constraints unless their derivation explicitly depends on `sch/sdch`; if so, re-derive them from the active-power convention rather than copying them mechanically.

## Objective

Replace shared-ESS usage and complementarity objective terms consistently:

- usage: old `sch + sdch` -> active `pch + pdch`;
- complementarity penalty: old `sch * sdch` -> active `pch * pdch`.

Preserve all coefficients. Do not retune penalties.

## Output and diagnostic compatibility

Audit every field previously derived from `sch/sdch`.

Do not silently change an output contract.

For the shared-ESS result currently represented as `s_ess = sch - sdch`, determine whether consumers expect:

- a signed charge/discharge quantity;
- apparent-power magnitude;
- display-only data.

If apparent magnitude is intended, use:

`sqrt(pnet^2 + qnet^2)`

and rename/document as needed. If a signed legacy field is required, preserve it explicitly and add a separate magnitude.

Update ADMM charge/discharge diagnostics to use active `pch/pdch`.

Units must be correct:

- `pch/pdch`: MW;
- apparent magnitude: MVA.

---

# P5.4-B — ordinary network ESS parity

Audit ordinary ESS consumers and apply the same active-energy semantics where applicable.

Current ordinary ESS has the same hard apparent-power geometry through `ess_snet_def` and its P4.6-B2 scale.

Production direction:

- preserve load-positive `es_pnet = es_pch - es_pdch`;
- SOC from `es_pch/es_pdch` active power;
- converter capability from `es_pnet/es_qnet`;
- active envelope on `es_pch/es_pdch`;
- complementarity on `es_pch * es_pdch`;
- remove `es_sch/es_sdch` where no longer required;
- remove `ess_snet_def` and `ess_snet_def_scale`/`kappa_es` machinery.

If SRP1 has no ordinary ESS, still run focused construction/physics tests using the previously validated ordinary-ESS case.

Do not alter unrelated network physics.

---

# P5.4-C — ESSO active-energy conversion

This phase is required before live ADMM.

Trace the complete ESSO formulation before editing it.

Convert per-cohort apparent charge/discharge semantics to physically active `pch/pdch` while preserving aggregate network-facing P/Q and installed S/E semantics.

SOC must use active energy:

`SOC_t = SOC_(t-1) + eta_ch * pch * dt - pdch * dt / eta_dch`.

Converter capability remains an apparent-power equipment constraint on aggregate P/Q.

Ensure ESSO and network agents describe the same feasible shared battery schedules.

## Throughput and cycling degradation

Do not invent a new degradation law.

Trace the existing degradation equation, weighting, and normalization first.

Replace apparent-power throughput with **cell-side active-energy throughput** while preserving the existing degradation-law structure.

Candidate physical cell-energy contributions:

charging:

`eta_ch * P_ch * dt`

discharging:

`P_dch * dt / eta_dch`.

Apply these only after verifying:

- representative-day weights;
- year weights;
- scenario weights;
- capacity normalization;
- equivalent-cycle definition;
- SoH linkage.

Calendar degradation remains deferred.

## ESSO consistency tests

Require:

1. pure Q -> zero SOC change and zero cycling throughput;
2. pure charging -> correct SOC and cell-energy throughput;
3. pure discharging -> correct SOC and cell-energy throughput;
4. aggregate P/Q convention matches the network agents;
5. power/energy investment sensitivities remain available;
6. zero capacity remains safe;
7. cohort aggregation remains exact.

---

# P5.4-D — sensitivity, lifecycle, and warm-start audit

Verify that removing `sess_snet_def` scaling machinery does not break:

- shared-S sensitivity;
- shared-E sensitivity;
- Benders sensitivity extraction/cuts;
- live candidate updates;
- zero -> positive capacity transitions;
- positive -> different-positive transitions;
- positive -> zero transitions;
- warm-start suffix behavior;
- reused-model component identity where required.

The old `kappa` multiplier-transfer logic should disappear if no remaining row needs it.

Do not introduce replacement multiplier transformations without a demonstrated need.

---

# P5.4-E — production validation before ADMM

Run focused unit/construction tests first.

Then rerun the complete positive-bootstrap initialization using the **actual production implementation**, not the B3 diagnostic wrapper.

Required population:

- 36 DSO models;
- 12 TSO models;
- 3 ESSO models.

Target gate:

- `36/36` DSO success;
- `12/12` TSO success;
- `3/3` ESSO success;
- zero persistent failures;
- record primary/recovery split;
- compare iteration/runtime statistics with B3.

Repeat targeted derivative/rank diagnostics and confirm:

- `sess_snet_def` no longer exists;
- no zero-gradient equality rows are introduced by the new ESS formulation;
- representative full equality Jacobians remain full row rank.

## Physical residual audit

Across every active shared-ESS row audit normalized physical violations.

Converter capability:

`max((pnet^2 + qnet^2 - S^2) / S^2, 0)`.

Complementarity:

`max((pch * pdch - eps * S^2) / S^2, 0)`.

Report:

- maximum;
- mean;
- p95;
- number/fraction above meaningful physical thresholds;
- worst row identities.

Do not judge these only from IPOPT's unscaled reported feasibility metric.

---

# P5.4-F — live distributed ADMM gate

Only after P5.4-E passes, run reduced SRP1 operational planning for the exact positive-bootstrap candidate.

Keep unchanged:

- seed/scenarios;
- ADMM parameters;
- IPOPT options;
- proximal settings;
- objective scaling;
- Benders settings.

Record:

- ADMM convergence;
- cycles;
- every local primary/recovery failure;
- residual evolution;
- recourse evolution;
- shared-ESS active P/Q schedules;
- SOC;
- converter-capability normalized residuals;
- active complementarity normalized residuals.

Compare against P4.5/P5 history.

Do not run the full outer planning loop yet.

---

# P5.4-G — reduced planning gate

Only if P5.4-F is sufficiently clean, rerun reduced:

`run_planning_problem()`.

Preserve current Benders/positive-bootstrap settings.

Record:

- outer iterations;
- candidate sequence;
- initialization failures;
- ADMM failures;
- local NLP failures/recoveries;
- sensitivities;
- incumbent evolution;
- termination reason;
- normalized converter/complementarity residuals.

Do not alter Benders logic in response to a failure during this stage. Stop and report a new failure family.

---

# P5.4-H — remaining inequality-conditioning decision

After live ADMM/planning evidence, decide whether the remaining tiny-scale inequalities need reformulation.

Do not default to another arbitrary row multiplier.

If normalized physical violations are materially non-negligible, prepare a proposal for **true dimensionless ESS internal variables**, e.g. fixed local numerical scaling such that:

`p_hat`, `q_hat`, `pch_hat`, `pdch_hat`

are `O(1)` and:

`p_hat^2 + q_hat^2 <= 1`

`pch_hat * pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE`.

Any proposal must preserve physical P/Q at the network/ADMM boundary and safely handle changing planning capacities on reused models.

Do not implement dimensionless-variable normalization unless the physical residual audit demonstrates a need.

---

# P5.4 required report

Produce:

`P5_4_ACTIVE_ENERGY_ESS_PRODUCTIONIZATION_REPORT.md`

with separate sections A-H.

Stop immediately if an end-to-end semantic inconsistency is found that cannot be resolved without changing the approved physical model.

Final verdict must be one of:

`P5.4 PASS — active-energy ESS productionization validated end-to-end`

`P5.4 PARTIAL — production formulation implemented but live coordination/planning questions remain`

`P5.4 FAIL — active-energy ESS productionization introduces unresolved issues`

Then stop for planner review.

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
