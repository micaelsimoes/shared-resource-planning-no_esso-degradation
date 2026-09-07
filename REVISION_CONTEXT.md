# Revision Context — Shared Resources Planning

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role

Act as a technical planner and mathematical-programming reviewer for the shared energy-storage planning repository.

Read this file first. When checking the mathematical formulation, also consult `simoes_2026_revisions.pdf` where relevant and inspect the current implementation before proposing changes. Prefer reviewer-driven implementation and validation plans before production edits.

This file is the repository-wide source of context. `LOCAL_NLP_STABILITY_PLAN.md` contains the currently authorized local-NLP/ESS implementation scope and takes precedence for the active P5.4 work.

---

# CURRENT SOURCE OF TRUTH — 2026-09-07

This section supersedes older solver-policy and local-NLP instructions recorded later in this file where they conflict with the live code.

## Current accepted production checkpoint

The active-energy ESS production stack is implemented in production through:

- `a4a0bae8` — shared-network active-energy ESS productionization;
- `1e86d40e` — ordinary network ESS active-energy parity;
- `58f4911b` — ESSO active-energy conversion and throughput correction;
- `c3526ec8` — lifecycle/sensitivity audit state and post-B/C/D production validation;
- `93974d83` — dimensionless charge/discharge complementarity across network ESS, ordinary ESS and ESSO, including ESSO aggregate complementarity;
- `2917b9c9` — fixed-candidate live distributed ADMM diagnostic on the H1 production baseline; no Benders/planning change.

Diagnostic evidence retained in this lineage includes:

- `b0e53bc4` — P5.4-E2 complementarity-significance instrumentation/reporting; no formulation change.

The earlier checkpoint `f77d829359ffd873367f556882546bc2dcc8ec99` remains the historical pre-P5.4 baseline used for controlled comparisons, not the current active-energy production state.

The P5.2 narrow-band shared-ESS relaxation remains rejected. `sess_snet_def` / `ess_snet_def` and their `kappa` machinery have been removed from the production ESS formulations rather than relaxed.

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

Use this checksum when comparing P5/P5.1/P5.2/P5.3/P5.4 reduced runs. Older checksum values appearing later in this historical file belong to earlier repository states and are not the current P5 baseline.

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

Do not retune these settings during P5.4 unless a later planner instruction explicitly authorizes it.

---

# CURRENT P5.4 CHECKPOINT — H1/F accepted; D2 is the active gate before planning

P5.4 sections A, B, C, D, E, E2, H1 and F are complete. The active-energy ESS formulation, normalized local complementarity, and net-P/Q ADMM coordination are accepted as the current production direction. The reduced outer planning loop P5.4-G has **not** been run on this final formulation because the shared-capacity sensitivity path used by Benders remains unresolved.

## Accepted active-energy production state

Current production lineage:

- `a4a0bae8` — shared-network active-energy ESS;
- `1e86d40e` — ordinary network ESS parity;
- `58f4911b` — ESSO active-energy conversion;
- `c3526ec8` — lifecycle/sensitivity audit and pre-H1 validation;
- `93974d83` — dimensionless complementarity with consistent network/ESSO relative semantics;
- `2917b9c9` — live fixed-candidate ADMM diagnostic on that baseline.

The final positive-bootstrap validation after H1 gives:

- DSO: `36/36` success;
- TSO: `12/12` success;
- ESSO: `3/3` success;
- primary failures: `0`;
- recoveries: `0`;
- persistent failures: `0`;
- total network IPOPT iterations: `3499`;
- mean/median/max network iterations: about `72.9 / 64 / 185`;
- representative network equality Jacobians remain full row rank;
- zero zero-gradient equality rows from the ESS formulation;
- shared-ESS converter-capability violations: `0/1728`;
- normalized network complementarity violations: `0/1728`;
- normalized ESSO per-cohort complementarity violations: `0/1728`;
- normalized ESSO aggregate complementarity violations: `0/864`.

The H1 numerical cost is accepted: iterations rose from `1556` to `3499` and bootstrap runtime from roughly `32 s` to `44 s`, but no new failure family or rank defect appeared. This cost corresponds to IPOPT actually enforcing a previously under-resolved physical condition.

## Complementarity decision — accepted numerical formulation

The current accepted local complementarity semantics are relative and dimensionless:

`pch = S_rated * pch_hat`

`pdch = S_rated * pdch_hat`

`pch_hat * pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE`

with:

`ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`.

This was shown to be an exact reformulation of the previous positive-capacity feasible set, while amplifying the numerical scale of the hard complementarity row from O(`S^2`) to O(1). The link equalities retain a unit coefficient on the physical variables and do not reintroduce the old zero-gradient equality defect.

Apply these semantics consistently to:

- shared network ESS;
- ordinary network ESS;
- ESSO per-cohort active charge/discharge;
- ESSO aggregate active charge/discharge.

The pre-H1 ESSO absolute complementarity convention is superseded.

Do **not** reduce `ESS_COMPLEMENTARITY_TOLERANCE` as part of D2. Whether `1e-4` is physically tighter than needed or still too loose is a later isolated physical-tolerance A/B. Current observed post-H1 circulation remains below the theoretical `sqrt(eps)=1%` allowance:

- network max `min(pch,pdch)/S ≈ 0.00834`;
- ESSO aggregate max `≈ 0.00793` in the bootstrap gate;
- live ADMM network max `≈ 0.00841`;
- live ADMM ESSO aggregate max `≈ 0.00763`.

## Coordination decision — net P/Q is sufficient

The fixed positive-bootstrap candidate converged through the real distributed operational planner in **9 ADMM cycles** with all local solves successful and no recovery diagnostics. All current rho adaptation, tolerances, proximal regularization, objective scaling, IPOPT settings, MA97/exact-Hessian policy, seed and scenarios were unchanged.

At final consensus, DSO-vs-TSO cell-energy-rate disagreement tracked the remaining net-P disagreement at the same solver-tolerance scale. This confirms the coordination decision:

- coordinate shared-ESS `pnet`;
- coordinate shared-ESS `qnet`;
- retain existing interface-voltage coordination;
- do **not** add `pch`, `pdch`, circulation, SOC, cell-energy rate or throughput as ADMM consensus variables.

`pch/pdch` remain local physical variables and may be recorded as sanity diagnostics. With local complementarity enforced, net active-power agreement determines charge/discharge direction to numerical accuracy.

## Active blocker — P5.4-D2 shared-S/E sensitivity root-cause audit

P5.4-G is blocked by the local capacity-sensitivity path used for Benders cuts. The previous D audit found that the dual of the shared-S rated-capacity fixing row did not reproduce central finite differences and could even disagree in sign. This issue predates P5.4, is numerically improved by the active-energy formulation, but remains unresolved.

D2 must now audit **both power capacity S and energy capacity E**, not only S.

The leading concrete hypothesis to test is derivative bookkeeping around capacity-dependent mutable variable bounds and other direct parameter dependencies. The local NLP contains symbolic rated-capacity variables and fixing rows, but lifecycle code also rewrites some physical operational bounds numerically when S/E changes. If those bounds are active, the value-function derivative may contain bound-multiplier contributions that are not represented by the fixing-row dual alone.

D2 must:

1. trace the exact S/E sensitivity contract from local fixing rows and IPOPT duals through weighting/sign conversion and Benders cut construction;
2. inventory every dependence of the local NLP on S and E, distinguishing symbolic rated-variable dependence, direct mutable-parameter dependence, numerically rewritten variable bounds, and structural/gating dependence;
3. extract IPOPT `zL/zU` for every capacity-dependent operational bound and determine whether those bounds are active;
4. derive the complete envelope-theorem derivative, including fixing-row, direct-expression and variable-bound contributions with the actual IPOPT sign convention;
5. rerun controlled central finite differences for both S and E on the current H1 production HEAD;
6. test whether mathematically redundant capacity-dependent numerical bounds can be removed in an exact sensitivity-clean local reformulation so that capacity dependence flows symbolically through the rated-capacity variables;
7. perform the analogous SOC/E-bound audit without changing SOC semantics;
8. validate any candidate sensitivity treatment across multiple DSOs, a TSO case, multiple years/days, and cases where S or E is operationally binding.

Do not modify Benders/local-cut equations during D2. First identify and validate the correct local derivative path.

## P5.4-G remains blocked

Do not run `run_planning_problem()` until D2 establishes one of the following with evidence:

- the existing S/E sensitivity extraction is correct; or
- a sensitivity-clean local formulation makes the fixing-row dual complete; or
- the exact additional derivative terms required by Benders are identified and planner-approved.

If the remaining mismatch is instead caused by local-optimum switching or nonsmooth active-set transitions, report that explicitly rather than forcing agreement by solver tuning.

## Deferred items

- B1 exact `f_ref = 0`: still deferred; do not add it to the current baseline.
- RES B2-R: deferred until a defensible converter `Smax` exists.
- Physical complementarity-tolerance A/B at `1e-5` / `1e-6`: deferred until after the sensitivity/planning path is stabilized.
- Calendar degradation: deferred.

# HISTORICAL DECISION — P5.3-B complete; P5.4 authorized

P5.3-B is complete. The B-series results are now authoritative and supersede the older “current P5.3 execution order” later in this file. Those older sections remain historical evidence only.

The accepted production checkpoint is still:

`f77d829359ffd873367f556882546bc2dcc8ec99`

The successful B3 formulation is **not yet a production commit**. It is the approved production direction to be implemented and validated end to end in P5.4.

## P5.3-B1 — exact reference-angle gauge

Diagnostic change:

`f_ref = 0`

instead of the current narrow reference-angle band.

Result:

- gauge freedom was removed cleanly;
- the `f_ref` variable disappeared from the NLP;
- equality-rank deficiency was unchanged because `sess_snet_def` remained present;
- the three original positive-bootstrap failures were repaired, but three different failures appeared;
- the failure count therefore remained 3 and two harsher modes (`Error in step computation`, `Restoration Failed`) appeared.

Decision:

**B1 = CONTINUE TESTING — do not productionize yet.**

Retest `f_ref = 0` only after the active-power ESS formulation is stable in production. It is not part of the initial P5.4 production baseline.

## P5.3-B2-R — RES capability semantics

The current curtailable-RES formulation uses:

`0 <= pg <= P_available`

and:

`pg^2 + qg^2 <= S_available^2`,

with current synthetic:

`Q_available = 0`

so:

`S_available = P_available`.

Therefore stochastic irradiance/wind availability directly sets the P/Q capability-circle radius. In reduced SRP1:

- every live RES cold start lies exactly on `sg_capability`;
- `sg_capability` is the binding reactive restriction in 100% of live cold-start points;
- 17 realized low-output points in `(1e-5, 1e-4] p.u.` create very small active capability circles;
- static `qmin/qmax` are effectively unreachable over much of the operating range.

However, the repository contains **no explicit, defensible inverter/converter apparent-power rating**. `Pmax`, `Qmax`, PF limits, historical maxima, and arbitrary oversizing factors must not be reinterpreted as `S_converter` without documented physical semantics.

Decision:

**B2-R = DEFER — insufficient physical rating data for safe reformulation.**

Future data-model direction:

- add an explicit optional generator field such as `Smax` [MVA];
- store it as a dedicated apparent-power rating, e.g. `Generator.s_rated` in p.u.;
- require at minimum `Smax > 0` and `Smax >= Pmax` when provided;
- keep legacy cases on the current formulation when no rating is supplied;
- do not adopt arbitrary plausibility thresholds such as `Smax > 2*Pmax` without equipment evidence.

Important separate semantic point:

An explicit `Smax` alone would **not** enable reactive-only/STATCOM operation at `pg = 0`, because the current PF cone also forces `qg = 0` at `pg = 0`. Converter rating and reactive-only operating policy are separate future modelling decisions.

Stochastic-model findings retained for later work:

- `abs()` reflection of negative RES samples is quantitatively negligible but should eventually be replaced by physical lower clipping (`max(sample, 0)`) for correctness;
- the material support issue is upper overshoot;
- historical-max exceedance is not automatically a physical violation — future auditing should quantify exceedance of installed `Pmax` / capacity factor > 1;
- cross-generator spatial correlation is currently not preserved.

Do not modify the RES formulation or copula during P5.4.

## P5.3-B3 — active-power shared-ESS prototype — ACCEPTED PRODUCTION DIRECTION

The diagnostic network formulation replaced the apparent-power charge/discharge geometry with active-power battery dynamics.

Core accepted physical direction:

`pnet = pch - pdch`

`SOC_t = SOC_(t-1) + eta_ch * pch * dt - pdch * dt / eta_dch`

with the current representative-day time basis verified as `dt = 1 h`.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`.

Active charging/discharging envelope derived from the old feasible set:

`pch + pdch <= S_rated`.

Complementarity moves from:

`sch * sdch`

to:

`pch * pdch`

with the existing `ESS_COMPLEMENTARITY_TOLERANCE` unchanged.

The diagnostic prototype removed/deactivated the shared-network internal rows/variables that depended on `sch/sdch`, including:

- `shared_es_sch`;
- `shared_es_sdch`;
- `sess_snet_def`;
- `sess_pch_link`;
- `sess_pdch_link`;
- old `sess_s_limit`;
- old apparent-power `sess_soc_def`;
- old `sess_comp`.

### Structural result

B3 removed the dominant structural defect completely:

- DSO zero-gradient equality rows: `24 -> 0`;
- TSO zero-gradient equality rows: `72 -> 0`;
- full equality Jacobian changed from exact rank deficiency to full row rank;
- DSO `sigma_min(full)` became approximately `5.925e-3`;
- TSO `sigma_min(full)` became approximately `3.287e-2`;
- the previous `sess_snet_def` curvature peak of about `18806` disappeared;
- largest remaining reported structural curvature was about `138`, a reduction of roughly 136x.

### Bootstrap robustness result

Exact positive-bootstrap A/B:

Production A:

- DSO success: `33/36`;
- persistent failures: `3`;
- total network IPOPT iterations: `33073`;
- mean iterations: `689`;
- median iterations: `468`;
- max: `3000`;
- runtime: about `274 s`.

Active-power B:

- DSO success: `36/36`;
- TSO success: `12/12`;
- ESSO unchanged success: `3/3`;
- primary failures: `0`;
- recoveries: `0`;
- persistent failures: `0`;
- total network IPOPT iterations: `1545`;
- mean iterations: `32.2`;
- median iterations: `27.5`;
- max iterations: `109`;
- runtime: about `37 s`.

The three original P5 failures were eliminated **without relocation**.

### Physics result

Required unit tests passed:

- pure Q with `pch = pdch = pnet = 0` produces exactly `Delta_SOC = 0`;
- pure charging changes SOC by `eta_ch * pch * dt`;
- pure discharging changes SOC by `-pdch * dt / eta_dch`;
- converter P/Q capability behaves correctly;
- zero-capacity gating remains safe.

This corrects the previous physical inconsistency where reactive apparent power affected stored battery energy.

### Remaining numerical issues after B3

These do **not** block the production direction, but must remain explicit.

Active-power complementarity:

`pch * pdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2`

remains under-resolved at tiny bootstrap capacity. Its RHS is roughly `1.1e-12` to `1.0e-11`, far below the network IPOPT absolute tolerance, and a tiny accepted physical violation was measured.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`

is also below the absolute solver feasibility scale for the smallest bootstrap devices.

These are now **inequality-resolution problems**, not equality-rank deficiencies.

Do not immediately respond with another arbitrary row multiplier. If later live ADMM/planning residual audits show material physical violations, prefer a true dimensionless ESS internal-variable formulation.

### Network/coordination compatibility

The B3 consumer trace established that load-bearing coordination components use `pnet/qnet`:

- nodal balance;
- ADMM consensus;
- expected shared-ESS P/Q schedules;
- scenario-deviation terms;
- Benders sensitivity extraction.

Two non-load-bearing consumers must be corrected during productionization:

1. the exported shared-ESS apparent-power/result field that currently derives from `sch - sdch`;
2. the ADMM residual diagnostic that reports charge/discharge using `sch/sdch`.

Do not silently change output semantics. If an existing `s_ess` field means apparent-power magnitude, use `sqrt(pnet^2 + qnet^2)` and document/rename as needed. If any consumer expects a signed quantity, preserve compatibility explicitly. Active charging/discharging quantities must be labelled in MW, not MVA.

## Current production decision

**B3 = PRODUCTIONIZE CANDIDATE / approved production direction.**

This does **not** mean the diagnostic wrapper is accepted production code. P5.4 must implement the active-energy semantics consistently across the production network model, ordinary ESS where relevant, ESSO, degradation/throughput, diagnostics, result exports, sensitivities, lifecycle handling, and live distributed execution.

The P5.2 narrow-band workaround is now abandoned. Do not productionize it. Do not continue tuning `sess_snet_def` or its `kappa` scale.

## Next authorized stage

**P5.4 — End-to-end active-energy ESS productionization.**

`LOCAL_NLP_STABILITY_PLAN.md` contains the detailed P5.4 execution protocol and takes precedence for implementation and validation.

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

# P5.3 — completed structural SMOPF review (historical execution record)

P5.3 is complete. The quantitative audit, corrected RES/Jacobian follow-up, and isolated B-series reformulation tests are retained below as historical execution evidence. The authoritative P5.3-B decisions and P5.4 next stage are stated near the top of this file.

`LOCAL_NLP_STABILITY_PLAN.md` contains the detailed execution protocol and takes precedence for the B experiments.

## P5.3-A / A2 — authoritative completed findings

The original P5.3-A row-wise audit correctly identified the shared-ESS nonlinear geometry as the dominant structural risk, but two global Jacobian conclusions were later corrected in P5.3-A2. The following statements are now authoritative.

### 1. `sess_snet_def` is the sole source of exact equality-row rank deficiency at the bootstrap cold start

Current production shared-ESS row:

`kappa * ((sch - sdch)^2 - pnet^2 - qnet^2) = 0`

with:

`kappa = 1 / S_rated`.

At the natural zero-dispatch cold start:

- every active `sess_snet_def` row has exactly zero first derivative;
- DSO models contain 24 exactly-zero equality rows;
- TSO models contain 72 exactly-zero equality rows;
- therefore the full equality Jacobian has `sigma_min = 0` and is exactly row-rank deficient;
- after removing only those exactly-zero rows, the tested reduced equality Jacobians have full row rank with no additional nullity.

Corrected reduced-spectrum conditioning on representative models:

- DSO reduced equality Jacobian condition number: approximately `8.98e4`;
- TSO reduced equality Jacobian condition number: approximately `1.42e3`.

The earlier claim that the TSO equality Jacobian was materially worse conditioned than the DSO Jacobian is **withdrawn**. The corrected result is the opposite on the nonzero subspace.

The accepted P4 normalization also gives the shared row curvature:

`2 * kappa = 2 / S_rated`,

reaching approximately `18806` at the smallest positive-bootstrap rating. Thus `sess_snet_def` combines:

- exact zero first derivative;
- an always-active equality;
- exact rank deficiency;
- curvature growing as `O(1/S_rated)`.

This remains the highest-priority structural defect.

### 2. `sess_comp` remains HIGH risk

The bilinear shared-ESS complementarity relaxation:

`sch * sdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2`

has, at the positive-bootstrap scale:

- cold-start Jacobian norms around `2e-8` to `6e-8`;
- an RHS/margin scaling with `S_rated^2`, reaching roughly `1e-12` at the smallest bootstrap capacity;
- a physical inequality margin many orders below the network IPOPT feasibility tolerance.

Do not hide this with another arbitrary scalar normalization. The active-power ESS prototype must re-audit complementarity after moving it to `pch * pdch`.

### 3. Corrected column diagnostics

The previous report of approximately 48 near-zero DSO Jacobian columns (`pij/qij`) was a diagnostic artifact.

Root cause:

- `r_sqr` has no Pyomo initial value;
- reverse-mode numeric differentiation failed on rows referencing it;
- the original audit swallowed the exceptions and skipped 120 DSO equality rows;
- this made `pij/qij` appear disconnected even though their defining equations contain unit coefficients.

After supplying a nominal diagnostic value, the derivative failures disappear and the `pij/qij/pji/qji` own-variable coefficients are exactly 1 as expected.

The near-zero DSO-column conclusion and the earlier `f_ref`-column red flag are therefore **withdrawn**. The remaining production observation is only that `r_sqr` lacks an explicit cold-start initialization.

### 4. RES `sg_capability` remains HIGH risk

For curtailable RES:

`pg^2 + qg^2 <= sg_available^2`.

The current reduced SRP1 population has:

- 3732 active `sg_capability` rows;
- zero cold-start margin for these rows because the initial `pg` is placed at availability;
- gradient norms down to approximately `5.44e-5` for the lowest live availability values;
- curvature 2.

There are 17 realized RES availability values in `(1e-5, 1e-4] p.u.`. These are the main low-output nonlinear RES rows to inspect in B2-R.

### 5. Old exact RES B2 is cancelled for SRP1

All 144 curtailable SRP1 generator instances have:

`power_factor_control = True`.

Their stochastic reactive availability is identically zero, but `qg` remains a controlled variable inside the PF cone. Therefore:

- `gen_pf_profile` is never instantiated in SRP1;
- there is no cross-multiplied fixed-profile equality to clean up;
- replacing `pg^2 + qg^2 <= S_available^2` by `pg <= S_available` would change the feasible set because reactive power is not fixed to zero.

The old P5.3-B2 exact PF-profile cleanup is therefore a no-op for SRP1 and is superseded by **B2-R — RES capability semantics and conditioning**.

### 6. Current RES availability/converter semantics need review

Synthetic RES currently has `q_available = 0`, so:

`sg_available = sqrt(pg_available^2 + qg_available^2) = pg_available`.

The same stochastic active-power availability is therefore used as the radius of the P/Q capability circle. Reactive capability collapses as stochastic active availability falls.

This may conflate:

- stochastic primary-resource availability; and
- inverter/converter nameplate MVA capability.

B2-R may test a separated formulation only if the repository contains an explicit, defensible converter apparent-power rating. Do not invent one from a heuristic.

### 7. RES stochastic-support findings

The historical-data copula/KDE scenario process itself remains the baseline, but P5.3-A2 established:

- negative inverse-transformed RES samples before `abs()` are very rare (`0` to `0.17%` per 2400 values in the recorded calls);
- positive mass created solely by reflecting negatives through `abs()` is negligible (`<= 0.01%` of post-`abs` mass);
- the important support issue is **upper overshoot**: some season/type calls have up to approximately `33.5%` of synthetic values above the historical maximum;
- in the realized reduced SRP1 population, 30.6% of RES values are exact zero;
- there are no realized values in `(0, 1e-5]`;
- therefore the current `EQUALITY_TOLERANCE = 1e-5` availability switch is not being exercised marginally in this reduced run;
- 17 live values lie in `(1e-5, 1e-4]` and instantiate small active capability circles.

The `abs()` hypothesis is downgraded. Future stochastic-model work should prioritize physical upper support and spatial dependence.

### 8. Spatial RES correlation is not preserved

The copula is fitted per `(season, RES type)` with 24 hourly dimensions, so it preserves temporal dependence within a daily profile.

Physical generator identity is pooled out at fit time. Each same-type physical generator then samples independently from the common synthetic pool using a generator-specific seed.

Therefore the current workflow preserves temporal dependence but **does not preserve cross-generator spatial correlation**.

Do not redesign the copula during P5.3-B; keep this as a later scenario-model revision.

### 9. DSO interface-voltage semantics are intentional

At initial model construction, a DSO reference voltage is tightly initialized/bounded around the local generator setpoint. However, the production ADMM setup explicitly frees the interface magnitude while retaining the reference angle.

Therefore the earlier concern that the DSO interface magnitude remained effectively pinned throughout ADMM is **withdrawn**. The cold-start pinning is an initialization boundary condition; the ADMM magnitude is deliberately released.

This strengthens the exact reference-angle B1 test: the code already intends to retain the angle reference, so `f_ref = 0` is the cleaner gauge formulation to validate.

### 10. IPOPT scales the objective but not the constraint rows

Current network solves rely on IPOPT's default gradient-based NLP scaling. Production logs show the large raw objective gradient is scaled internally to approximately the configured maximum-gradient scale.

However, constraint scaling is not supplied. Thus the raw disparity among:

- zero-gradient rows;
- `~1e-8` complementarity rows;
- `~1e5` branch-related rows;

remains exposed to the KKT system. MA97 has also reported scaling activation due to excess delays.

Do not respond by retuning IPOPT during P5.3. Prefer formulation improvements.

## Historical P5.3 execution order (completed)

For practical debugging and validation, proceed in this order:

### B1 — exact reference-angle gauge

Test:

`f_ref = 0`

against the current narrow `+/- EQUALITY_TOLERANCE` reference-angle band.

This is the lowest-risk, mathematically exact cleanup and should be validated first.

### B2-R — RES capability semantics and conditioning

Run this **second**, before the larger ESS refactor, because it is easier and quicker to debug and validate.

First inspect every curtailable RES generator for an explicit, defensible converter/inverter apparent-power rating.

If such a rating exists, the diagnostic candidate is conceptually:

`0 <= pg <= P_available`

for stochastic resource availability, together with:

`pg^2 + qg^2 <= S_converter^2`

for converter capability, retaining the existing PF-control constraints.

This is a deliberate feasible-set change, not an exact algebraic rewrite.

If no defensible `S_converter` exists, stop B2-R before implementation and recommend the minimum data-model extension instead. Do not infer a rating from an arbitrary heuristic.

Keep the stochastic-support recommendations separate from the NLP formulation experiment.

### B3 — active-power ESS structural prototype

Run this **third**. It remains the highest-payoff structural reformulation, but it is deliberately postponed until after B2-R because B2-R is faster to isolate and validate.

Diagnostic target:

`pnet = pch - pdch`

`SOC_t = SOC_{t-1} + eta_ch * pch * Delta_t - pdch * Delta_t / eta_dch`

`pnet^2 + qnet^2 <= S_rated^2`

with complementarity on `pch * pdch`.

The prototype should determine whether `sch/sdch`, `ess_snet_def`, `sess_snet_def`, and the associated link equations can be removed safely from the network SMOPF.

This is a deliberate physical reformulation and is not authorized for production until the consumer trace, physics checks, rank/conditioning tests, and bootstrap solver comparison pass.

End-to-end ESSO throughput/degradation conversion remains a follow-on stage if B3 is favorable.

## Current decision discipline

- Do not test further shared-ESS epsilon values.
- Do not test further scalar `kappa` caps.
- Do not use solver-option tuning as a substitute for formulation work.
- B1, B2-R, and B3 are isolated experiments and each starts from the same accepted production baseline.
- A favorable B result is reported for planner review before any productionization or stacking.

# P5.3 invariants and prohibitions (historical)

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

P5.3 diagnostic A/B branches must be isolated. Reference-angle (B1), RES capability semantics (B2-R), and active-power ESS (B3) prototypes each start from the same accepted production baseline rather than stacking changes.

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

# Active-energy SOC and degradation correction — IMPLEMENTED BASELINE

The earlier physical inconsistency in which apparent charge/discharge could alter stored battery energy has now been corrected in production.

Current network ESS baseline:

`pnet = pch - pdch`

`SOC_t = SOC_(t-1) + eta_ch * pch * Delta_t - pdch * Delta_t / eta_dch`.

Reactive power remains constrained by converter apparent-power capability but does not directly change battery stored energy.

Ordinary network ESS uses the same active-energy convention.

The ESSO model has no SOC state variable. Its existing degradation/throughput path has been corrected to use cell-side active-energy throughput:

`E_throughput = sum_d sum_t weight_d * Delta_t * (eta_ch * P_ch[d,t] + P_dch[d,t] / eta_dch)`

while preserving the pre-existing representative-day, cohort, year, equivalent-cycle and SoH semantics.

The remaining ESS issue is local charge/discharge complementarity resolution, not active-energy SOC physics. P5.4-H1 is now authorized to introduce dimensionless internal charge/discharge variables for that complementarity only, with the existing `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4` initially unchanged.

---

# Calendar degradation

Calendar degradation remains authorized only **after** the active-energy SOC/cycling baseline is validated end to end.

Target conceptual extension:

`SoH_cumul[k,y] = SoH_cumul[k,y-1] * SoH_cycle[k,y]^(365 * Delta_y) * phi_cal[k]^Delta_y`

with `0 < phi_cal <= 1` and disabled compatibility case `phi_cal = 1`.

Keep `phi_cal` conceptually separate from the existing calendar-life/retirement parameter used for cohort retirement and salvage.

Do not begin calendar-degradation implementation during P5.4.

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

The immediate task is **P5.4-D2 — shared-S/E capacity-sensitivity root-cause audit**. H1 and F are complete and accepted; do not rerun them except for narrowly targeted controls required by D2.

Current controlled sequence:

1. **P5.4-D2 — S/E sensitivity root-cause audit:** trace all parameter dependence, bound multipliers and the exact envelope derivative; validate against controlled finite differences on the current H1 production HEAD.
2. **Sensitivity-clean local formulation A/B, only if exact:** if capacity-dependent numerical bounds are redundant and can be removed without changing the feasible set, test that reformulation and repeat S/E derivative validation.
3. **Planner review of D2:** decide whether the existing fixing-row dual is sufficient, whether additional bound/direct terms are required, or whether local nonsmoothness prevents trustworthy cuts at the tested point.
4. **P5.4-G — reduced planning gate:** only after a trustworthy S/E sensitivity path is established or a planner-approved alternative is identified.
5. **Later physical complementarity-tolerance A/B:** keep H1's normalized formulation fixed and test `eps=1e-5` / `1e-6` only as a separate physical modelling question, not as a numerical fix.

Do not:

- add `pch/pdch` ADMM consensus;
- change `ESS_COMPLEMENTARITY_TOLERANCE` during D2;
- change H1 dimensionless complementarity semantics;
- introduce arbitrary row multipliers or `kappa` scaling;
- retune IPOPT/ADMM/proximal settings;
- modify Benders/local-cut logic before the derivative root cause is established;
- modify RES capability/copula behavior;
- include B1 `f_ref = 0` in the current baseline;
- run the outer planning loop before D2 planner review.

`LOCAL_NLP_STABILITY_PLAN.md` contains the authoritative D2 implementation and validation protocol.
