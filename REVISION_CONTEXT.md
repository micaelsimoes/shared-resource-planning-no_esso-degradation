# Revision Context — Shared Resources Planning

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

## Role

Act as a technical planner and mathematical-programming reviewer for the shared energy-storage planning repository.

Read this file first. When checking the mathematical formulation, also consult `simoes_2026_revisions.pdf` where relevant and inspect the current implementation before proposing changes. Prefer reviewer-driven implementation and validation plans before production edits.

This file is the repository-wide source of context. `LOCAL_NLP_STABILITY_PLAN.md` contains the currently authorized implementation/audit scope and takes precedence for the active P5.5 planning-architecture work. Despite its legacy filename, that plan now governs the convex-planning architecture rather than another local-NLP repair stage.

---

# CURRENT SOURCE OF TRUTH — 2026-09-07

This section supersedes older solver-policy, P5.4 planning and P5.5-A instructions recorded later in this file where they conflict with the current state.

## Current accepted nonlinear production checkpoint

The accepted active-energy ESS production lineage is:

- `a4a0bae8` — shared-network active-energy ESS productionization;
- `1e86d40e` — ordinary network ESS active-energy parity;
- `58f4911b` — ESSO active-energy conversion and throughput correction;
- `c3526ec8` — lifecycle/sensitivity audit state and post-B/C/D validation;
- `93974d83` — dimensionless charge/discharge complementarity across network ESS, ordinary ESS and ESSO, including ESSO aggregate complementarity;
- `06e921e5` — P5.4-D2-P sensitivity-clean shared-S productionization; redundant positive-capacity S-dependent numerical bounds removed with no feasible-set change.

Accepted nonlinear coordination evidence includes:

- `2917b9c9` — historical fixed-candidate live distributed ADMM diagnostic on the H1 production baseline; net-P/Q coordination only.

Diagnostic/planning evidence includes:

- `b0e53bc4` — P5.4-E2 complementarity-significance instrumentation;
- `65b261ba` — P5.4-D2 S/E sensitivity root-cause audit;
- `e1afa8e9` — original D3 cut-consistency audit, now retained as **NONCANONICAL historical evidence** because it was executed in `srp_env`;
- `1a68f409` — original D4 branch-recovery/hardened-cut audit, also **NONCANONICAL historical evidence**;
- `51a3f4e4`, `5bd1f0ce`, `cbc043d4` — P5.4-R canonical-environment revalidation, which supplies the authoritative D3/D4 paper-instance verdicts.

The old pre-P5.4 checkpoint `f77d829359ffd873367f556882546bc2dcc8ec99` remains historical only.

## Canonical paper environment and reproducibility identity — HARD REQUIREMENT

All future paper-instance validation, benchmark and planning-oracle work must use:

`/opt/anaconda3/envs/opf_env_py311/bin/python`

Canonical environment/provenance from P5.4-R:

- Python `3.11.11`;
- NumPy `2.4.2`;
- pandas `3.0.0`;
- SciPy `1.17.0`;
- Pyomo `6.9.5`;
- IPOPT `3.14.18` at `/usr/local/bin/ipopt`;
- HSL linear solver `ma97`;
- Gurobi / `gurobipy` `13.0.1`, academic licence, expiry `2027-04-10`.

Canonical SRP1 configuration:

- random seed `2026`;
- years `2025`, `2030`, `2035`;
- representative days `Spring`, `Summer`, `Autumn`, `Winter`;
- `24` instants;
- `1` market scenario;
- `1` operation scenario per network;
- TSO `case9`;
- DSOs `case33_1` at node 5, `case33_2` at node 7, `case33_3` at node 9;
- combined realized scenario checksum:

`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

The fail-fast provenance gate introduced in P5.4-R is now a required infrastructure convention: paper-case harnesses must record interpreter/environment/package/solver provenance and abort if the checksum differs. A rejected environment must never silently produce evidence that is mixed with canonical results.

`srp_env` is noncanonical for paper validation because it generates a different stochastic realization (`4d948b9b...`). Seed `2026` alone is therefore **not** a sufficient reproducibility specification.

## Current live nonlinear network solver policy

The live source and current JSON files supersede older notes referring to earlier warm-start policies.

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
- DSO warm starts retain configured acceptable settings unless explicitly overridden;
- warm-start push/fraction settings inherit configured network values unless explicitly provided.

Recovery remains limited to IPOPT `internalSolverError`; configured limited-memory recovery exists for `case33_2` and `case33_3`, while `case33_1` still has no explicit `recovery_options` block. Do not retune the nonlinear solver during P5.5-B.

---

# CURRENT PLANNING CHECKPOINT — P5.4-R closed; P5.5-B authorized

P5.4-R has now revalidated the planning diagnosis under the canonical paper environment. The environment inconsistency changed the numerical magnitudes substantially, but **the planning verdict survived**.

P5.4-G was never run and remains permanently blocked for the old nonlinear-recourse derivative cuts. The active workstream is the new branch `feature/convex-planning` and the next authorized stage is **P5.5-B — mathematical closure of the convex lower-bound architecture**.

## Canonical operational regression after D2-P

P5.4-R1 is the authoritative paper-instance numerical gate for the current production formulation:

- DSO `36/36`;
- TSO `12/12`;
- ESSO `3/3`;
- primary failures / recoveries / persistent failures `0/0/0`;
- H1 complementarity violations `0/1728`;
- converter-capability violations `0/1728`;
- total network IPOPT iterations `3424`;
- mean / median / max iterations `71.3 / 65.0 / 134`;
- runtime about `42 s`;
- representative equality Jacobians full row rank with zero zero-gradient ESS equality rows;
- maximum `min(pch,pdch)/S = 8.6060e-03`, below the `sqrt(1e-4)=1e-2` allowance.

Earlier D2-P bootstrap figures such as `3442` iterations from `srp_env` are historical/noncanonical and must not be quoted as the paper-instance regression.

## Complementarity and nonlinear coordination remain locked

The accepted nonlinear complementarity formulation remains:

`pch = S_rated * pch_hat`

`pdch = S_rated * pdch_hat`

`pch_hat * pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE`

with:

`ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`.

The future `1e-5` / `1e-6` A/B remains a separate physical-tolerance question and is not part of P5.5-B.

Nonlinear distributed coordination remains net electrical P/Q plus the existing interface-voltage quantity. Do not add `pch`, `pdch`, circulation, SOC, cell-energy rate or throughput as nonlinear ADMM consensus variables.

## D2-P accepted conclusion — local-branch sensitivity bookkeeping is fixed

D2-P removed only the four mathematically redundant positive-capacity numerical S-dependent bounds on:

- `shared_es_pch`;
- `shared_es_pdch`;
- `shared_es_pnet`;
- `shared_es_qnet`.

After D2-P:

- no positive-capacity operational variable bound depends numerically on S;
- the omitted bound-multiplier contribution to `dQ/dS` is zero;
- the fixing-row dual is structurally complete for the **local nonlinear branch returned by the solver**;
- E remains structurally sensitivity-clean.

This stays in production. It does not make the nonlinear recourse value function convex and does not create a globally valid Benders subgradient.

## Canonical D3 verdict — nonlinear-recourse cuts are unsafe

P5.4-R2 recomputed D3 from scratch under `opf_env_py311`.

Canonical base result:

- `Q0 = 838496830.813414`;
- `17` ADMM cycles;
- deterministic repeat spread `0.0`;
- all 18 aggregated S/E coefficients negative;
- `tol_cut = 7.164e5` from the canonical stopping drift;
- the tested linear sensitivity effect remained much smaller than this numerical resolution.

Canonical cut test:

- `2/8` candidates produced decisive violations;
- worst violation approximately `-2.680808e6` at `S(node9,2025) = 0.90*S0`;
- the `S(node9,2025) = 0.95*S0` point also violated decisively, around `-1.315031e6`;
- no tested candidate remained on the base branch.

Authoritative verdict:

`P5.4-R-D3 FAIL — canonical nonlinear-recourse cuts are demonstrably unsafe`.

The earlier `srp_env` D3 magnitudes (for example `Q0 = 848258809.814117`, 6/8 decisive violations) remain historical only.

## Canonical D4 verdict — anchor hardening does not repair the method

P5.4-R3 regenerated branch states canonically and reran the branch-recovery/hardened-cut audit.

Canonical branch-recovery result:

- cold base `Q = 838496830.81`;
- best observed recovered base branch `Q = 836586463.43`;
- improvement about `1.910367e6` (`0.228%`), not the noncanonical ~`1.28e7` figure;
- branch recovery remains carried by the primal state rather than multiplier transfer;
- the objective-gap fingerprint again localizes primarily to **TSO generation dispatch on Spring representative days**, partly offset by Summer changes.

Canonical hardened-cut result:

- `8/8` tested neighbouring candidates decisively violated the hardened cut;
- hardened cut gaps were roughly `-0.96e6` to `-1.13e6`, all beyond canonical `tol_cut = 7.164e5`;
- the capacity-linear contribution remained only O(`1e4`), far below the branch-driven change.

Authoritative verdict:

`P5.4-R-D4 C — canonical hardened cuts remain demonstrably unsafe`.

Therefore the old nonlinear-recourse derivative-cut/Benders architecture is retired as a global lower-bound method.

## Important distinction — environment sensitivity vs mutable-state impurity

Two separate issues must not be conflated:

1. **Environment/scenario sensitivity.** `srp_env` and `opf_env_py311` generate different stochastic profiles and select different nonlinear branches. This is why only canonical evidence is admissible for the paper instance.
2. **Mutable-state/call-history impurity.** Independent code inspection and earlier D4 call-order experiments showed persistent network/shared-ESS data are mutated in place across candidate evaluations. This remains a software-hygiene issue for any future nonlinear upper-bound oracle.

The environment experiment does **not** prove the call-history bug; it separately demonstrates the branch sensitivity of the nonlinear problem.

---

# P5.5 ARCHITECTURE STATUS

## P5.5-A accepted as PARTIAL

P5.5-A completed the exact controllable-resource and convexity audit. Its structural findings are accepted, with several design statements explicitly reopened for B-stage correction.

Accepted SRP1 resource/topology facts:

- conventional generator P/Q are controllable;
- RES P/Q, curtailment and PF control are retained;
- active and reactive load flexibility are retained;
- load curtailment is disabled;
- ordinary ESS is absent from SRP1;
- shared ESS P/Q/SOC are present;
- each `case33_*` has **one controllable continuous OLTC** on branch 1, bus 1 -> bus 2, with `r in [0.83,1.17]`;
- there is no discrete tap logic, no phase shift, and no controllable capacitor bank/switched shunt in SRP1;
- `case33_1/2/3` are radial after preprocessing (33 buses, 32 branches, cyclomatic number 0);
- `case9` is meshed with one independent cycle (9 buses, 9 branches, cyclomatic number 1);
- the planning master is an LP with no integer/binary or bilinear master-side terms.

The intended convex family remains a lifted **W-space AC SOC/QC relaxation**. Do not replace it with DC-OPF or LinDistFlow.

## Two-oracle planning architecture remains the target

For planning candidate `x = (S,E)` define:

`R(x)` = globally solved convex-relaxed AC SMOPF recourse;

`Q_AC_feas(x)` = feasible full nonlinear AC SMOPF recourse, eventually AC-polished before being labelled a rigorous upper bound.

Required relation:

`R(x) <= Q_AC*(x) <= Q_AC_feas(x)`.

The planning lower bound is generated only from the convex oracle. The nonlinear model remains the physical truth model and upper-bound/incumbent validator.

A future convex cut must be a globally valid affine under-estimator of the relaxed value function. It must not be obtained by mixing an unrelated scalar solver bound with a local sensitivity vector.

## Gurobi is the selected prototype solver

P5.4-R4 corrected P5.5-A's solver inventory.

In `opf_env_py311`:

- `gurobipy 13.0.1` is available;
- academic licence is valid through `2027-04-10`;
- `gurobi`, `gurobi_direct`, and `gurobi_persistent` Pyomo interfaces report available;
- `QCPDual=1` is accepted;
- both linear duals and quadratic/conic duals were returned on a verified SOCP/QCP test;
- Gurobi reports both `ObjVal` and `ObjBound`.

For a rigorous scalar lower bound, use the solver's **dual bound (`ObjBound`)**, not merely the primal incumbent `ObjVal`.

However, for the **planning cut**, P5.5-B must establish a stronger contract: the intercept and S/E coefficient must be derived from the **same dual-feasible conic solution** so that the entire affine function is globally valid. Do not assume `ObjBound(x_k) + g_k^T(x-x_k)` is safe without that proof.

## P5.5-A statements explicitly reopened in P5.5-B

The following A-stage statements are not accepted as final and must be corrected/closed before implementation:

1. **Dormant ±30-degree angle constraints are not automatically LB-safe.** They are commented out in production and cannot be enabled merely as tightening. Any QC/angle strengthening must either already be present in the nonlinear feasible set or be mathematically implied by it.
2. **OLTC McCormick relaxation is not yet the preferred formulation.** A transformed continuous-tap formulation may eliminate explicit tap bilinearity without adding relaxation beyond the usual voltage-product SOC relaxation.
3. **Centralized interface semantics need exact sign tracing.** The DSO REF generator and TSO ADN load/interface variables must be mapped exactly before replacing ADMM by affine equalities. Do not collapse duplicate TSO/DSO/ESSO variables in the first convex prototype.
4. **ESSO degradation/SoH relaxation needs an objective-level bound proof**, including salvage placement; feasible-set enlargement alone is insufficient if objective terms are also changed.
5. **The Gurobi dual-cut contract and modelling interface remain unvalidated on the intended capacity-fixing structure.**
6. **The nonlinear upper-bound oracle is not yet pure with respect to call history.** This is not a blocker for mathematical B-stage design, but it must be fixed before future planning-UB certification.

---

# CURRENT ACTIVE STAGE — P5.5-B mathematical closure

P5.5-B is **audit/design only**. Do not implement the convex SMOPF or a new planning loop yet.

The B-stage must close the following items before any implementation is authorized:

- historical-report cleanup so canonical P5.4-R figures supersede noncanonical numerical claims;
- admissibility classification of every proposed QC/angle strengthening;
- exact transformed continuous-OLTC SOC derivation and proof, including whether `r` and `r_sqr` can be eliminated;
- exact TSO ADN / DSO REF / shared-ESS / voltage consensus sign semantics for a centralized convex model;
- complete ESSO objective-level lower-bound proof and salvage placement;
- rigorous Gurobi dual-derived affine cut contract, including the cut intercept;
- choice among `gurobi_direct`, `gurobi_persistent`, or native `gurobipy` for the first prototype;
- exact centralized prototype specification, with no remaining architecture decisions.

### Preferred OLTC hypothesis to test, not yet an accepted production formula

For each DSO transformer, investigate transformed variables:

`U_i = r^2 * W_ii`

`C_ij = r * W_ij^R`

`D_ij = r * W_ij^I`.

The hypothesis is that the actual transformer P/Q equations become affine in `U_i`, `C_ij`, `D_ij`, `W_jj`, with physical rank relation:

`C_ij^2 + D_ij^2 = U_i * W_jj`,

relaxed by the standard rotated SOC inequality.

If `r` appears nowhere else and voltage lower bounds guarantee `W_ii > 0`, test whether continuous tap existence is represented exactly by:

`r_min^2 * W_ii <= U_i <= r_max^2 * W_ii`,

since `r = sqrt(U_i/W_ii)` would then exist in `[r_min,r_max]`.

Do not claim this is exact until every production occurrence of `r`/`r_sqr` has been audited.

### Rigorous cut requirement

The future convex planning cut must be constructed from a single dual-feasible convex solution:

`L_k(x) = beta_k + g_k^T x`

with:

`L_k(x) <= R(x)` for all admissible planning capacities.

P5.5-B must determine analytically and numerically how `beta_k` and `g_k` are derived from the Gurobi dual solution, including capacity-fixing rows and all constant/RHS terms. At the generating point compare `L_k(x_k)`, `ObjBound`, and `ObjVal`; do not assume equality except as justified by the solver tolerances/dual formulation.

## New development branch

Active branch:

`feature/convex-planning`

It was created from the current P5.4/P5.5 lineage and must remain isolated from the previous nonlinear-planning branch. Do not rebase or rewrite the previous branch history.

## Locked decisions during P5.5-B

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
- nonlinear objective scaling or objective coefficients;
- old master/Benders equations;
- production data.

Do not yet implement:

- the centralized convex SMOPF;
- the new LB/UB planning loop;
- distributed convex ADMM;
- SDP;
- trust-region/local-cut planning;
- derivative-free pattern/coordinate search;
- surrogate-assisted planning.

## Old P5.4-G status

P5.4-G is permanently blocked for the old nonlinear-recourse derivative cuts. It must not be revived as validation of the previous Benders architecture.

A future relaxation-based planning loop is a **new planning architecture** and requires its own validation stages.

## Deferred items

Remain deferred during P5.5-B:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R reformulation until defensible converter `Smax` data exists;
- nonlinear solver/ADMM retuning;
- implementation of the convex model itself;
- distributed convex ADMM;
- SDP;
- trust-region/local-cut planner;
- derivative-free coordinate/pattern search;
- surrogate-assisted planning;
- final nonlinear upper-bound oracle purity fix, except for tracing/specification needed by B-stage design.

---

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

Local charge/discharge complementarity has since been resolved by accepted P5.4-H1 using dimensionless internal charge/discharge variables with `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`. That nonlinear formulation remains the accepted production baseline; complementarity is deliberately relaxed only in the proposed future convex lower-bound model if required for convexity.

---

# Calendar degradation

Calendar degradation remains **deferred**. The active-energy SOC/cycling baseline has been validated, but calendar-degradation implementation is not part of P5.5-A and requires separate planner authorization.

Target conceptual extension:

`SoH_cumul[k,y] = SoH_cumul[k,y-1] * SoH_cycle[k,y]^(365 * Delta_y) * phi_cal[k]^Delta_y`

with `0 < phi_cal <= 1` and disabled compatibility case `phi_cal = 1`.

Keep `phi_cal` conceptually separate from the existing calendar-life/retirement parameter used for cohort retirement and salvage.

Do not begin calendar-degradation implementation during P5.5-A.

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

The immediate task is **P5.5-A - exact convexity and controllable-resource audit** on a new development branch `feature/convex-planning`.

Execute in this order:

1. create `feature/convex-planning` from the repository's current HEAD, recording source branch/commit and worktree status;
2. inventory the exact controllable resources actually instantiated in SRP1 and resolve OLTC/cap-bank status from data and model construction;
3. build a row-by-row convexity map of the current operational SMOPF and the planning/master/ESSO equations;
4. derive the actual lifted W-space AC equations for the production branch model;
5. determine radial/meshed topology and recommend SOC versus SOC+QC strengthening for TSO and DSO;
6. prove joint convexity or identify remaining nonconvexity in planning capacities S/E;
7. specify only lower-bound-safe treatment of ESS complementarity and ESSO degradation/SoH/lifetime relationships;
8. produce the complete resource-retention matrix and the two-oracle LB/UB planning specification;
9. inspect solver availability without installing anything;
10. trace, but do not yet fix, the D4 mutable-state/call-order dependency;
11. write `P5_5_CONVEX_PLANNING_ARCHITECTURE_REPORT.md` and stop for planner review.

Do **not**:

- implement the convex OPF yet;
- run the old nonlinear-recourse `run_planning_problem()` path;
- modify the accepted nonlinear operational formulation;
- retune IPOPT/MA97/ADMM/rho/proximal terms;
- change `ESS_COMPLEMENTARITY_TOLERANCE`;
- assume SRP1 has or does not have controllable OLTCs/capacitor banks before the A1 audit;
- use DC-OPF or LinDistFlow as the proposed lower-bound model;
- drop controllable resources merely for convenience;
- claim any relaxation is lower-bound-valid without proving the direction.

`LOCAL_NLP_STABILITY_PLAN.md` contains the authoritative P5.5-A implementation/audit protocol despite its legacy filename.

