# Revision Context — Shared Resources Planning

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role

Act as a technical planner and mathematical-programming reviewer for the shared energy-storage planning repository.

Read this file first. When checking the mathematical formulation, also consult `simoes_2026_revisions.pdf` where relevant and inspect the current implementation before proposing changes. Prefer reviewer-driven implementation and validation plans before production edits.

This file is the repository-wide source of context. `LOCAL_NLP_STABILITY_PLAN.md` contains the currently authorized local-NLP diagnostic scope and takes precedence for P5.3 experiments.

---

# CURRENT SOURCE OF TRUTH — 2026-09-06

This section supersedes older solver-policy and local-NLP instructions recorded later in this file where they conflict with the live code.

## Current accepted production checkpoint

Accepted production baseline:

`f77d829359ffd873367f556882546bc2dcc8ec99`

Separate accepted checkpoints leading to it include:

- `feca8618b21ef8d7ae72202201e9f7af79397dbc` — reduced explicit voltage-magnitude indexing;
- `0171f451` — shared-ESS `sess_snet_def` normalization;
- `231511cb` — ordinary-ESS load-positive P/Q convention;
- `5639f397` — OP1 parameter alignment used as the current ordinary-ESS validation baseline;
- `f77d829359ffd873367f556882546bc2dcc8ec99` — ordinary-ESS `ess_snet_def` normalization.

No P5.2 narrow-band shared-ESS relaxation has been accepted into production. The live production shared-ESS row remains the scaled hard equality.

## Current reduced SRP1 reproducibility identity

- Random seed: `2026`
- Years: `2025`, `2030`, `2035`
- Representative days: `Spring`, `Summer`, `Autumn`, `Winter`
- Instants: `24`
- Market scenarios: `1`
- Operation scenarios per network: `1`
- TSO: `case9`
- DSOs: `case33_1` at node 5, `case33_2` at node 7, `case33_3` at node 9
- Combined realized scenario checksum:
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`

Use this checksum when comparing P5/P5.1/P5.2/P5.3 reduced runs. Older checksum values appearing later in this historical file belong to earlier repository states and are not the current P5 baseline.

## Current live network solver policy

The live source and current JSON files supersede older notes that referred to a warm-only `acceptable_iter = 1` policy and `1e-9` warm-start pushes.

Current cold network configuration:

- IPOPT exact-Hessian primary path;
- MA97 for TSO and DSOs;
- `tol = 1e-5`;
- `acceptable_tol = 1e-4`;
- `acceptable_iter = 5` in the network parameter files;
- `case9`: `bound_push`, `bound_frac`, `slack_bound_frac`, `slack_bound_push` = `1e-6`;
- `case33_1`, `case33_2`, `case33_3`: the same four push/fraction values = `1e-5`.

Current warm-start handling in `network._create_smopf_solver`:

- bound and constraint multipliers are supplied when `from_warm_start=True`;
- TSO warm starts override `acceptable_iter = 0` and `acceptable_tol = tol`, preventing acceptable-level early termination from reintroducing the previously diagnosed voltage-barrier artifact;
- DSO warm starts retain their configured acceptable settings unless another explicit option override is supplied;
- warm-start push/fraction settings inherit the configured network values unless explicitly provided.

Network recovery:

- production recovery is attempted only for IPOPT `internalSolverError`;
- recovery clears multiplier suffixes and retries from the current primal point with configured recovery options;
- `case33_2` and `case33_3` define limited-memory recovery profiles;
- `case33_1` currently defines no `recovery_options`, an acknowledged configuration asymmetry;
- `maxIterations` does not currently trigger immediate recovery.

Do not retune these settings during P5.3.

---

# Current local-NLP checkpoint — P1 through P5.2-A3

## P1/P2 — voltage-magnitude structural conditioning

The original local failure family was traced to redundant explicit voltage-magnitude variables/equalities and later to TSO acceptable-level barrier behavior.

Accepted structural change:

- keep `e`, `f`, `vmag_sqr`, and `vmag_sqr = e^2 + f^2` on all physical nodes;
- create explicit `vmag` and `vmag_sqr = vmag^2` only where `vmag` is actually consumed:
  - DSO: reference/interface node;
  - TSO: active DSO-interface nodes.

This eliminated the decisive frozen `case33_2 / node 7 / 2025 Winter / cycle 10` exact-Hessian failure and subsequently passed live operational smoke tests.

The TSO warm-start policy was then tightened so acceptable-level restoration cannot stop at a materially larger barrier parameter. The resulting operational run converged with zero active voltage slack at convergence.

## P3/P4 — shared-ESS nonlinear equality normalization

The network-side shared-ESS magnitude row was identified as a major local KKT conditioning trigger:

`g_sess = (sch - sdch)^2 - pnet^2 - qnet^2 = 0`

At small dispatch all four derivatives can approach zero. Exact-Hessian MA97 failures in both DSO and TSO frozen states were removed by scaling the existing row in place.

Accepted production formulation:

`kappa_e * g_sess = 0`

with fixed numerical scale:

`kappa_e = 1 / S_rated[e]`

for active positive-capacity shared ESS. Zero/near-zero shared ESS uses the existing operational gating: variables are fixed to zero, operational rows are deactivated, and the finite placeholder scale never affects the active NLP.

Because shared-ESS capacity can change on a reused live model, the implementation keeps the scale synchronized with installed power capacity and transforms an imported row multiplier consistently when the scale changes.

This change reduced the P2.10 live primary local failures from 14 to 2 in the accepted P4.5 smoke and retained ADMM convergence.

## P4.6 — ordinary ESS sign convention and normalization

Ordinary network ESS now uses the canonical load-positive convention:

`es_pnet = es_pch - es_pdch`

so:

- `pnet > 0`: charging / active consumption;
- `pnet < 0`: discharging / active injection;
- `qnet > 0`: reactive absorption;
- `qnet < 0`: reactive injection.

Nodal balance and result processing use the same convention end to end.

Ordinary `ess_snet_def` is normalized analogously:

`kappa_es[e] * ((sch - sdch)^2 - pnet^2 - qnet^2) = 0`

with immutable build-time:

`kappa_es[e] = 1 / S_rated[e]`.

Unlike shared ESS, ordinary ESS has no zero-capacity gating. An explicitly instantiated ordinary ESS must therefore have rated apparent power greater than `1e-10 p.u.`; zero/near-zero explicit ratings are rejected at construction.

The OP1 validation case with two `0.005 p.u.` devices produced `kappa_es = 200`, remained a clean primary exact-Hessian solve, approximately halved IPOPT iterations, and preserved the physical equality and output sign convention.

## P5 — reduced planning baseline

The exact current P5 reduced planning baseline was run from production checkpoint `f77d8293...`.

Iteration 1, zero investment:

- operational ADMM converged in 9 cycles;
- no local primary network failures;
- no ESSO failures;
- zero active voltage slacks at the accepted solution;
- recourse stationarity passed.

Iteration 2, production positive-bootstrap candidate:

- initialization failed before ADMM;
- `case33_1 / node 5 / 2030 Winter` -> `maxIterations`;
- `case33_1 / node 5 / 2035 Winter` -> `maxIterations`;
- `case33_3 / node 9 / 2025 Summer` -> `maxIterations`;
- no recovery was attempted because `maxIterations` is outside the current recoverable class.

The previously problematic `case33_2 / node 7` and TSO shared-ESS-interface failure families did not reappear in P5.

## P5.1 / P5.1-B — small-capacity shared-ESS scaling diagnostics

The positive-bootstrap power ratings are very small:

- 2025: `0.010635 MVA = 1.0635e-4 p.u.` -> production `kappa ~= 9403`;
- 2030: `0.021270 MVA = 2.1270e-4 p.u.` -> production `kappa ~= 4701.5`;
- 2035: `0.031905 MVA = 3.1905e-4 p.u.` -> production `kappa ~= 3134.3`.

Capping `kappa` proved that row scaling directly controls convergence in several sensitive cold starts, but no scalar cap was robust across the full initialization population:

- cap 100 cleared the three original P5 failures but introduced a different `case33_3 / 2025 Autumn` failure;
- a tested scaling ladder showed strongly non-monotone behavior;
- `Kmax = 1000` was the only tested cap that solved the four targeted states simultaneously, but full initialization then failed at four different DSO states, including node 7.

Conclusion:

**do not productionize a scalar cap on `1/S_rated`.** Row scale is influential, but capping relocates path-dependent failures rather than robustly eliminating them.

## P5.2-A / A2 / A3 — narrow-band diagnostic

Hypothesis tested:

The hard equality

`g_sess = 0`

has exactly zero gradient at:

`sch = sdch = pnet = qnet = 0`.

A finite scalar multiplier cannot remove this exact zero-gradient equality degeneracy.

Diagnostic replacement:

`-epsilon_rel * S_rated^2 <= g_sess <= +epsilon_rel * S_rated^2`

while keeping the accepted production `kappa = 1/S_rated` unchanged in the scaled row.

### `epsilon_rel = 1e-5`

All eight known sensitive states ultimately solved, and the full 51-solve positive-bootstrap initialization had zero persistent failures. One targeted network state (`case33_2 / node 7 / 2030 Summer`) required the existing limited-memory recovery after a primary `internalSolverError`.

### epsilon sensitivity

Targeted sensitivity considered `1e-5`, `3e-5`, and `1e-4`:

- `1e-5`: outstanding node-7 case remained recovery-dependent;
- `3e-5`: target became a clean primary success but a previously successful node-5 control failed outright;
- `1e-4`: target and all three matched controls succeeded on the primary exact-Hessian path.

### Full initialization at `epsilon_rel = 1e-4`

Strong solver-side result:

- 51/51 local initialization solves successful;
- 36/36 DSO;
- 12/12 TSO;
- 3/3 ESSO;
- 48/48 network solves clean on the primary exact-Hessian path;
- zero network recovery attempts;
- zero persistent failures;
- initialization would enter ADMM.

Blocking physical/numerical finding:

The nominal band is below IPOPT's effective constraint-feasibility resolution for the tiny bootstrap devices.

Across 1728 active shared-ESS network rows:

- max `|g| / S_rated^2 = 2.6331e-4`;
- mean = `1.2871e-5`;
- 95th percentile = `5.9805e-5`;
- max nominal band utilization = `2.6331`;
- 126 rows (7.29%) exceeded 0.5 nominal utilization;
- 22 rows (1.27%) exceeded 0.9;
- 20 rows were at or beyond the nominal boundary within the audit criterion;
- worst cases were concentrated in TSO rows;
- maximum apparent-power mismatch remained small in absolute terms (`~48 VA`, max `DeltaS/S_rated ~= 2.25e-3`) but the declared band itself was not a reliable physical error budget.

Interpretation:

- converting the hard zero-gradient equality into an inequality is strongly beneficial structurally;
- the current tolerance-band construction is not yet a principled production physical constraint because the declared band is finer than the network solver's feasibility resolution;
- **do not productionize the P5.2 narrow band yet**;
- stop epsilon and scalar-kappa tuning and perform a broader structural conditioning audit.

---

# P5.3 — newly authorized structural SMOPF review

P5.3 is the immediate priority.

Its purpose is to identify hard, degenerate, redundant, poorly scaled, or near-dependent SMOPF rows before selecting further production changes.

`LOCAL_NLP_STABILITY_PLAN.md` contains the detailed execution protocol.

## Priority formulation families

### 1. Shared and ordinary ESS geometry

Current shared and ordinary network ESS use auxiliary apparent charging/discharging variables `sch/sdch`, a squared-magnitude equality, link inequalities, an apparent-power sum limit, SOC driven by `sch/sdch`, and bilinear charge/discharge complementarity.

The shared row is known to have zero gradient at the natural zero-dispatch cold start, while its accepted normalization makes its raw curvature scale as `O(1/S_rated)` for very small installed power.

The bilinear complementarity relaxation is also numerically suspicious at tiny capacity because its right-hand side scales with `S_rated^2` and can be many orders below IPOPT's feasibility tolerance.

### 2. Active-power ESS structural prototype

A diagnostic physical reformulation is explicitly authorized for P5.3 after the quantitative audit.

Target network formulation:

`pnet = pch - pdch`

`SOC_t = SOC_{t-1} + eta_ch * pch * Delta_t - pdch * Delta_t / eta_dch`

`pnet^2 + qnet^2 <= S_rated^2`

with charging/discharging complementarity acting on active `pch/pdch`.

The prototype should investigate whether `sch/sdch`, `ess_snet_def`, `sess_snet_def`, and their link equations can be removed from the network SMOPF.

This is not yet a production authorization. The network prototype must first trace every consumer of the removed variables and show pure-Q operation leaves SOC unchanged.

End-to-end ESSO throughput/degradation conversion remains a follow-on stage if the network prototype is favorable.

### 3. RES low-output constraints

Curtailable RES currently has:

- stochastic available active/reactive generation;
- nonlinear apparent-power capability `pg^2 + qg^2 <= sg_avail^2`;
- PF cone inequalities for controllable PF;
- a cross-multiplied profile equality for fixed PF:
  `q_available * pg == p_available * qg`;
- a structural unavailable/available switch based on `EQUALITY_TOLERANCE`.

Very small positive stochastic availability can therefore activate nonlinear rows with very small derivatives.

Where stochastic `q_available` is exactly zero, the profile equality reduces to:

`p_available * qg = 0`,

which is algebraically equivalent to `qg = 0` for positive `p_available` but can have an arbitrarily small Jacobian coefficient.

P5.3 may test mathematically equivalent unit-scaled/linear RES forms while holding the stochastic scenario values fixed.

### 4. RES stochastic-generation preprocessing

The RES synthetic scenario generator fits the existing Gaussian-multivariate/copula/KDE model to historical data and currently applies:

`np.abs(inverse_transformed_samples)`

after sampling.

P5.3 must measure, without changing the stochastic model:

- negative raw inverse-transformed samples before `abs`;
- positive generation created solely by reflecting a negative sample;
- samples outside historical support;
- tiny positive values around the RES availability threshold;
- the relationship between these values and local NLP difficulty.

Do not replace the copula model during P5.3.

The audit must also determine whether same-type generator assignments preserve spatial correlation adequately or effectively draw independently from a common pool.

### 5. Reference-angle/gauge treatment

The DSO/T SO rectangular formulation currently allows a reference-bus imaginary voltage within a narrow band rather than fixing the gauge exactly.

P5.3 is authorized to test the exact reformulation:

`f_ref = 0`

provided the audit confirms this does not conflict with an intentional coordinated-interface convention.

Also inspect DSO reference-bus `e` bounds: determine whether they nearly pin the coordinated DSO interface voltage even when `enforce_vg = false`.

### 6. Branch-current conditioning

For current-limited branches, quantify cancellation and derivative scales in expressions based on:

`V_i^2 + V_j^2 - 2 W_ij_real`

multiplied by series-admittance magnitude squared.

Low-impedance distribution branches may generate large coefficients and cancellation-sensitive rows. Do not reformulate unless the quantitative audit ranks them materially high.

### 7. Transformer auxiliaries

The model currently constructs transformer-ratio variables across the branch index set even though non-transformer equations use constant ratio 1 and `r_sqr_def` skips non-transformers.

P5.3 must determine whether unused non-transformer variables are eliminated by the generated NL problem or reach IPOPT. Classify this as code cleanup only if they are eliminated.

---

# P5.3 invariants and prohibitions

During P5.3:

- do not tune IPOPT tolerances;
- do not increase `max_iter` as a solution;
- do not switch production MA97/MA57 policy;
- do not change ADMM rho rules or tolerances;
- do not change recourse-stationarity criteria;
- do not change common ADMM objective scaling;
- do not change TSO proximal regularization;
- do not change Benders/local-cut logic;
- do not add generic feasibility slacks;
- do not productionize the P5.2 narrow band;
- do not implement a scalar cap on shared-ESS `kappa`;
- do not change the stochastic samples during the exact RES algebra A/B;
- do not add calendar degradation;
- do not change terminal salvage;
- do not silently change complementarity tolerances.

P5.3 diagnostic A/B branches must be isolated. Reference-angle, RES cleanup, and active-power ESS prototypes each start from the same accepted production baseline rather than stacking changes.

---

# Completed repository-wide work retained from earlier stages

## First-stage investment formulation

- ESS power and energy investments are scenario-independent variables indexed by ESS and investment year.
- Scenario-dependent investment-cost coefficients/probabilities remain active.
- The budget uses expected scenario-weighted expenditure.
- Results report one implementable physical investment plan.

## Solver separation

- LP master: Clp.
- Nonlinear operational subproblems: IPOPT.
- NLP and LP solver paths/configuration remain separated.

## Benders-type objective accounting

- Master estimate = investment cost + alpha.
- Gross recourse aggregates discounted/annualized TSO and DSO base SMOPF objectives.
- Net recourse subtracts terminal shared-ESS salvage.
- ESSO feasibility penalties and ADMM augmentation are excluded from economic recourse.
- The procedure is described as Benders-type with local sensitivity cuts; global lower-bound guarantees are not claimed.
- Operational non-convergence or material ESSO infeasibility stops the outer loop rather than generating a formal feasibility cut.

## Minimum SoH

- Minimum SoH remains enforced through the available-energy inequality and the rated-energy/cumulative-SoH identity.
- `soh_min = 0.50` remains intentionally hard-coded for the current baseline.
- Configurable minimum SoH remains deferred.

## ADMM convergence and adaptive penalties

- Convergence is evaluated after complete DSO-TSO-ESSO cycles.
- Interface voltage, P/Q flow, and shared-ESS consensus residuals are monitored separately.
- Economic recourse stationarity is required.
- Adaptive rho updates use tolerance-normalized residual balancing and hold groups already satisfying both primal and dual criteria.
- Failed local NLP cycles do not count as converged and do not update penalties from unreliable residuals.
- The method remains a nonconvex ADMM heuristic.

## Failure gating

- Failed initialization stops before ADMM.
- Failed ADMM-cycle blocks do not replace retained successful schedules or update their coupled duals.
- Success predicates use termination condition, not solver status alone.

## Terminal salvage

- Terminal salvage is an outer net-recourse credit, not part of the ESSO feasibility objective.
- SRP1 values battery energy capacity only, with remaining-calendar-life and normalized-health factors.
- Power-converter capacity is excluded from battery-health salvage.
- Calibration of the provisional salvage fractions remains required before final paper runs.

## Incumbent preservation and local validation

- The best feasible incumbent is preserved separately from later rejected candidates.
- Outer termination is explicitly classified.
- Local sensitivity/finite-difference validation infrastructure exists, but final reviewer-facing derivative validation remains deferred until the formulation is stable.

## Direct voltage-magnitude slack formulation

- Rectangular component slacks and `e_actual/f_actual` auxiliaries were removed.
- Voltage-limit relaxation uses lower/upper nonnegative squared-voltage slacks around the physical `e,f` voltage.
- Reference/interface/enforced-PV nodes retain hard voltage behavior as configured.

## Directional branch apparent-power limits

- Apparent-power limits are enforced at both terminals where required.
- Sending/reverse terminal reactive-flow definitions use consistent half-shunt accounting.
- Apparent-power auxiliary variables are indexed only where actually needed.
- Directional branch-loading/slack results are exported consistently.

---

# Active-energy SOC and degradation correction

A physical inconsistency remains explicitly recognized:

- network ESS SOC currently uses apparent charging/discharging variables;
- shared-ESS degradation throughput also uses apparent charge/discharge quantities;
- pure reactive-power operation can therefore alter stored battery energy and cycling degradation.

The intended physical baseline is active/cell-side energy throughput:

`SOC_t = SOC_{t-1} + eta_ch * P_ch * Delta_t - P_dch * Delta_t / eta_dch`

and representative-year cell-side throughput:

`E_throughput = sum_d sum_t (D_d / 365) * Delta_t * (eta_ch * P_ch[d,t] + P_dch[d,t] / eta_dch)`.

Reactive power remains constrained by converter apparent-power capability but should not directly change battery SOC or battery-cell cycling SoH.

P5.3-B3 is allowed to prototype this correction in the network SMOPF because it may simultaneously remove numerically difficult `sch/sdch` geometry. Do not implement the full ESSO degradation rewrite in the same diagnostic branch.

If the prototype is accepted, the subsequent end-to-end correction must cover ordinary ESS, shared network ESS, ESSO per-cohort active-power allocation, aggregate P/Q coordination, throughput, cycling degradation, SoH, sensitivities, state mapping, and exports consistently.

---

# Calendar degradation

Calendar degradation remains authorized only **after** the active-energy SOC/cycling baseline is validated end to end.

Target conceptual extension:

`SoH_cumul[k,y] = SoH_cumul[k,y-1] * SoH_cycle[k,y]^(365 * Delta_y) * phi_cal[k]^Delta_y`

with `0 < phi_cal <= 1` and disabled compatibility case `phi_cal = 1`.

Keep `phi_cal` conceptually separate from the existing calendar-life/retirement parameter used for cohort retirement and salvage.

Do not begin calendar-degradation implementation during P5.3.

---

# Final-paper work still deferred

After the numerical and physical formulation stabilizes:

1. validate nonzero salvage on a controlled later investment cohort;
2. validate local sensitivities/finite differences at polished operational states;
3. revisit incumbent-centered trust-region/local-cut stabilization only if the evidence requires it;
4. run no-degradation / cycling-only / cycling+calendar experiment matrix;
5. run calendar-retention and discount-rate sensitivities;
6. calibrate provisional salvage parameters;
7. reconcile manuscript equations, algorithms, terminology, convergence claims, numerical tables, and response letter with the verified implementation.

Never describe the local-cut master estimate as a rigorous global lower bound or the procedure as globally convergent Benders decomposition.

---

# Immediate instruction

The immediate task is P5.3 as defined in `LOCAL_NLP_STABILITY_PLAN.md`:

1. quantitative conditioning map over the exact current positive-bootstrap network population;
2. RES stochastic-data/low-output audit;
3. isolated exact reference-angle A/B;
4. isolated exact RES algebra cleanup A/B;
5. isolated active-power ESS network prototype;
6. evidence-based ranked production proposal.

Do not start production edits automatically after a favorable diagnostic result. Report and wait for planner approval.
