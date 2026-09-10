# Revision Context — Shared Resources Planning

Repository:
`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

Preparation-copy note: this document may be edited in another checkout before
being transferred to the Mac Studio. All authorized execution must nevertheless
use the Mac Studio repository path above.

## Role

Act as a technical planner and mathematical-programming reviewer for the shared energy-storage planning repository.

Read this file first. When checking the mathematical formulation, also consult `simoes_2026_revisions.pdf` where relevant and inspect the current implementation before proposing changes. Prefer reviewer-driven implementation and validation plans before production edits.

This file is the repository-wide source of context. `LOCAL_NLP_STABILITY_PLAN.md`
contains the currently authorized implementation/audit scope and takes
precedence for active execution. Despite its legacy filename, that plan now
governs the P5.12-B forensic audit of the first cold-RESCALED local NLP failure
rather than another local-NLP production repair stage.

---

# CURRENT SOURCE OF TRUTH — 2026-09-10

This section supersedes every older "current", "active" or "immediate"
instruction later in this file. Those sections remain historical evidence only.

## Repository and accepted branch state

The authorized worker checkout is the Mac Studio repository:

`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

The active branch remains:

`feature/derivative-free-planning`

At the end of P5.12-A the Mac Studio state was:

- HEAD `808590d23ad250627be9ff9d4ae51d059cbdbcdb`;
- tracked working tree clean;
- upstream `origin/feature/derivative-free-planning`;
- ahead 8, behind 0;
- no merge, pull, fetch, push, rebase, cherry-pick, reset or commit performed by
  the P5.12-A worker.

HEAD moved after P5.11 through a user merge that touched only
`REVISION_CONTEXT.md` and `LOCAL_NLP_STABILITY_PLAN.md`. That merge selected the
origin documents and again pruned the explicit P5.6-D, P5.7 and P5.8 record.
The accepted lineage remains recoverable from pre-merge commit `fbdb2f6a` and
local ancestor `bff465d7`. The recovered summary below restores those decisions.

The Mac Studio checkout remains several commits ahead of its remote. This is a
repository-preservation risk, but it is **not** authorization to push or rewrite
history. Any repository synchronization requires separate user approval.

## Current canonical Mac Studio runtime

The current paper-instance runtime on the Mac Studio is:

`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`

P5.11 independently enumerated ten interpreters and found exactly one matching
the accepted provenance. `/opt/anaconda3` does not exist on the Mac Studio.

Accepted runtime identity:

- Python `3.11.11`, arm64;
- NumPy `2.4.2`;
- pandas `3.0.0`;
- SciPy `1.17.0`;
- Pyomo `6.9.5`;
- copulas `0.14.0`;
- IPOPT `3.14.18` / ASL `20241111` at `/usr/local/bin/ipopt`;
- HSL `ma97`;
- canonical realized-scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

The old `/opt/anaconda3/envs/opf_env_py311/bin/python` path later in this file is
historical machine-specific provenance and is not the Mac Studio command.

## Recovered accepted P5.6-C through P5.9 lineage

The following accepted history was pruned by the documentation-only merge and
is restored here from the accepted pre-merge lineage.

### P5.6-C — landscape and coverage gate

Accepted verdict:

`P5.6-C-C — derivative-free search is not ready`.

T0 versus T4 Spearman correlation was `-0.033`; 6 of 9 improvement signs
reversed. Fixed continuation rescued every tested direct failure but changed
already-VALID controls by about `2.9e6`, so it could not be used only on failure
without mixing incompatible oracle surfaces. The reported P5.6-B
`se|ALL|x19` crash was withdrawn as non-reproducible. End-to-end costs were
corrected to about 135 seconds for VALID and 186 seconds for failed evaluations.

### P5.6-D — uniformly refined nonlinear oracle

Accepted verdict:

`P5.6-D-C — uniform refinement does not stabilize the investment landscape`.

Uniform refinement reached 100% coverage from `K=4` and substantially improved
direction consistency, but did not stabilize ordering. The accepted
`tau_planning_refined` was `811438.05`, the best candidate changed at every
tested depth transition, and node 9 / 2025 reversed its preferred investment
sign between `K=8` and `K=12`. No accepted `K_STAR` exists.

The best known feasible nonlinear planning incumbent from that stage was
`se|node5|2025|-10%` with total `825109566.571083`.

### P5.7 — branch-selection diagnosis

Accepted verdict:

`P5.7-A — unique operational oracle can likely be recovered`.

Objective scaling was the primary diagnosed mechanism. Production formed each
augmented subproblem objective as `base / effective_scale + consensus terms`,
with `effective_scale` about `9.41e4 .. 1.16e5`. Re-solving all 48 subproblems
from their own converged point changed the recovered base objective by
`-383401.68` under CURRENT scaling and `-8892463.76` under RESCALED scaling,
explaining about 96.5% of the ADMM-to-polish gap.

Initialization sensitivity was small by comparison: four starts of the same
polish NLP spread by only `303.24` on an objective around `8.3e8`. Active-set
differences were sparse, and continuation improved reachability rather than
fixed-consensus local quality. The earlier conjecture that cold starts used a
candidate-dependent TSO anchor was later refuted by P5.8.

### P5.8 — ADMM objective-scaling validation

Accepted verdict:

`P5.8-B — objective scaling improves stability but additional ADMM issues remain`.

`RESCALED = effective_scale * CURRENT` was verified as an exact positive
multiple on all 48 blocks. RESCALED recovered much more of the base objective
and improved the maximum unscaled constraint violation by about 70 times, but
the initial RESCALED replay still drifted, often broke exact-consensus polish,
and approached the active-power interface tolerance. The inherited
`consecutive_converged_cycles` channel was also exposed.

The then-current objective stopping threshold was about 25 times the planning
signal, while the anchor was shown to remain effectively common across starts.
No production scaling or stopping-rule change was authorized by P5.8.

### P5.9 — partial stabilization and parameter diagnosis

P5.9 carried RESCALED evaluation forward and established the facts later used
by P5.10:

- `rho_v` was numerically inert across the tested 667-fold range;
- ESS consensus remained far inside tolerance;
- the adaptive `rho_pf` rule compressed requested values 300, 500 and 1000 to
  approximately 88.89, 98.77 and 131.69 because its dual residual scaled with
  rho;
- the CURRENT eight-depth comparison still had two ranking flips and
  cross-depth relative uncertainty `459695.59` against a mean signal
  `18085.03`, an uncertainty/signal ratio of `25.42`;
- RESCALED improved coverage and the landscape but did not yet establish a
  stable, explicitly configured polished oracle.

P5.9 did not authorize production adaptive-rho changes or investment search.

## Accepted P5.10 stabilized-oracle evidence

Accepted report:

`P5_10_STABILIZED_RESCALLED_ADMM_ORACLE_REPORT.md`

SHA-256:

`fe674ee534ba9b7cf72c188b756a1ad585133727b53aad453091531fd289311e`

P5.10 made no production changes. It established an explicit harness-level
oracle configuration and identified four carried-state channels:

1. rho stored in the template;
2. objective scaling stored in cloned models;
3. `consecutive_converged_cycles`;
4. `last_recourse`.

`last_recourse` was a latent candidate-history asymmetry. Neutralizing all
history channels left the base result bit-identical and must remain part of the
declared oracle contract.

The accepted stabilized diagnostic oracle uses:

- `RESCALED` objective construction;
- fixed `rho_v = 1.5`, `rho_ess = 1.0`;
- fixed `rho_pf`, with 300 the faster working value and 1000 retained as a
  separate certification cross-check;
- adaptive rho disabled;
- histories neutralized before every evaluation/generation;
- midpoint anchor held identical across candidates;
- exact-consensus polish as the authoritative endpoint.

At fixed `rho_pf` 300, 500 and 1000, base, node-5 and node-9 candidates all
solved and polished directly with zero failed blocks. `rho_pf = 300` was about
2.4 times cheaper than 1000 while producing the same qualitative three-point
landscape. The original P5.10 selection rule between 300 and 1000 was internally
contradictory, so neither value is a new production setting.

Exact-consensus polish is mandatory. The unpolished ADMM states were locally
feasible but did not represent one common physical point. At `rho_pf = 1000`,
about 1.6 kW of interface disagreement produced an approximately 850-unit polish
correction against an investment signal around 600. Polished and unpolished
oracles ranked candidates in opposite orders.

The current adaptive-rho rule self-cancels under the RESCALED residual
definition because the measured dual residual scales linearly with rho and
therefore triggers rho reduction. Requested `rho_pf` values 300, 500 and 1000
decayed to approximately 88.89, 98.77 and 131.69. No adaptive-rho redesign was
implemented or authorized.

Across nine stabilized terminating cycles, `stationarity_pf` was the sole
binding convergence criterion. Objective stationarity, consensus, voltage and
ESS criteria were materially inside tolerance. This is an open diagnostic
question, not authorization to retune convergence tolerances.

The eight-generation replay at RESCALED, fixed `rho_pf = 1000`, adaptive rho
off and neutralized history produced:

- all generations VALID;
- zero failed polish blocks;
- node 9 preferred at every depth;
- zero ranking flips;
- cross-depth relative uncertainty reduced from about `459695.59` to `22.09`;
- uncertainty/signal reduced from `25.42` to `0.67`;
- best-to-second mean gap only `32.87`, still thin relative to the numerical
  floor `tau_numerical = 10`;
- absolute base-chain drift about `-201099`, so only same-depth relative
  comparisons are scientifically valid.

P5.10 validated only three candidates. It did not establish a broad investment
landscape, a self-consistent cold RESCALED T0, a production adaptive-rho policy,
or readiness for an investment-search campaign.

## Accepted P5.11 consolidation evidence

Accepted report:

`P5_11_STABILIZED_ORACLE_CONSOLIDATION_REPORT.md`

SHA-256:

`a9f9e3422a533d993ccd890cbee53a4a8e1dcbbc3469e95af3a592ec04281b3b`

P5.11.0 passed:

- the Mac Studio checkout and history were identified correctly;
- the governing documents were not rolled back to P5.6-C, although they had not
  yet been extended with P5.10;
- all P5.10 harnesses and 21 evidence files were present, tracked and unmodified;
- the R0 provenance gate passed;
- CURRENT polished total reproduced bit-identically as
  `828021090.3608505`;
- RESCALED pre-polish recourse reproduced bit-identically as
  `825814074.4930633`.

P5.11.1 attempted to construct a diagnostic T0 cold under RESCALED, with fixed
rho `1.5 / 300 / 1.0`, adaptive rho disabled and carried histories neutralized.
The 48 augmented objectives were RESCALED at construction. The diagnostic state
identity was:

- template id `P511-SELFCONSISTENT-T0`;
- consensus/dual state SHA-256
  `14a7ea85ce283a595a35b1768c93c2a60bda872d4f5138c5168b3691a625cbf0`;
- artifact file SHA-256
  `3629784b277724e1f2c406f4a6dbd4599ea92b9d7ee126ac67bb2148909a67bd`;
- configuration hash `a080612dedf2727e`.

The cold RESCALED construction did **not** converge within the frozen production
cap of 25 ADMM cycles. `cycle_convergence` was false at every cycle. Recourse
moved from approximately `2.3520e9` at cycle 1 to `1.4612e9` at cycle 13 and
`1.4024e9` at cycle 25; `primal_pf` was `3.5893e-4` at cycle 25 and had worsened
relative to cycle 13.

Candidate evaluations warm-started from this non-converged state returned local
and polish successes, but they do not validate a self-consistent template. The
inherited-versus-diagnostic relative-delta discrepancies were:

- `rho_pf = 300`: `122.37` for node 5 and `27.59` for node 9;
- `rho_pf = 1000`: `68.61` for node 5 and `81.55` for node 9.

All exceeded the required `22.09` bound, so the P5.11.1 stopping rule fired.
P5.11.2, P5.11.3 and P5.11.4 were correctly not run.

This result does not invalidate P5.10. The inherited-template oracle reproduced
the accepted gates exactly and its P5.11 values matched P5.10. What failed was
the attempted hardening through a cold-built RESCALED T0.

## Accepted P5.12-A cold-RESCALED diagnostic, with qualification

Accepted report text SHA-256:

`a1424ec1b47944ad1aa5a9038306ab1fec4870bf0a3ed41993a29ba1f2c89e96`

Accepted stage verdict:

`P5.12-A FAIL — cold RESCALED diagnostic encountered a numerical failure`.

Repository/runtime provenance, both P5.10 reproduction gates and the P5.11
cycle-25 state hashes passed. No tracked production file or parameter changed.

The decisive authorized evidence is the first local NLP failure at cycle 21.
Retrospective instrumentation also established that the P5.11 cold build had
local failures at cycles 21-24 that its earlier harness did not record. This
strengthens the rejection of the P5.11 diagnostic T0.

P5.12-A did not enforce its stop rule correctly. Instead of one trajectory that
terminated at the first failure, it launched four concurrent cold runs capped at
25, 50, 75 and 100. The runs continued after cycle 21, and later states were
formed while production retained schedules across failed local solves.
Therefore:

- cycle 21 is accepted as the binding failure evidence;
- cycle 25-100 results are exploratory only;
- the reported repeated failures from cycle 83 onward do not establish a valid
  "permanent breakdown" conclusion;
- detached polish results after the first failure cannot certify a T0;
- concurrency is an avoidable experimental confound for future replay.

The inherited CURRENT-built / RESCALED-evaluated template remains the only
validated construction path. P5.10 continues to stand for its restricted
three-candidate, exact-consensus-polished surface. The cold RESCALED construction
is invalid and must not be used for candidate ranking.

## CURRENT AUTHORIZED STAGE — P5.12-B only

P5.12-B is a read-only forensic preservation and matched-state audit of the
first cold-RESCALED local failure at cycle 21.

The stage must:

- use one process and one base-candidate trajectory only;
- reproduce the accepted cold RESCALED configuration through cycle 20;
- capture cycle-20 successful and cycle-21 pre-solve states before attempting
  the first failing local solves;
- identify every failing block, IPOPT termination condition and solver-log
  signature at cycle 21;
- compare each failing cycle-21 block with its matched successful cycle-20
  instance;
- terminate the entire experiment immediately when the first cycle-21 local
  failure is observed;
- propose exactly one later frozen A/B, without implementing it.

P5.12-B must not continue to cycle 22, retry a failed block, reproduce cycle 83,
or change any formulation, solver option, ADMM parameter, penalty, tolerance,
anchor, initialization or production source.

The authoritative protocol and stopping rules are in
`LOCAL_NLP_STABILITY_PLAN.md`.

## Locked prohibitions during P5.12-B

Do not change:

- production nonlinear formulation;
- IPOPT settings;
- production ADMM settings or iteration cap;
- adaptive-rho logic;
- convergence tolerances or stationarity definitions;
- objective scaling in production;
- TSO proximal regularization;
- ESSO degradation or active-energy degradation work;
- Benders logic or convex models;
- anchor or initialization policy;
- accepted P5.10/P5.11/P5.12-A evidence.

Do not run the full planning problem, derivative-free search, expanded candidate
population, stationarity sensitivity, another fixed-rho value, or any
post-failure ADMM cycle.

---

# HISTORICAL SOURCE OF TRUTH — 2026-09-08

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

Recovery remains limited to IPOPT `internalSolverError`; configured limited-memory recovery exists for `case33_2` and `case33_3`, while `case33_1` still has no explicit `recovery_options` block. Do not retune the nonlinear solver during P5.6-C.

---

# HISTORICAL PLANNING CHECKPOINT — P5.5-D accepted; P5.6-A/B reviewed; P5.6-C authorized

P5.4-R remains the canonical basis for the nonlinear production formulation and for retirement of the old nonlinear-recourse derivative cuts. P5.4-G was never run and remains permanently blocked.

The rigorous continuous-convex / MISOCP lower-bound route has been tested and retired as the planning architecture. P5.5-D remains accepted as:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

The decisive P5.5-D evidence remains:

- the continuous convex hull of the H1 charge/discharge set is already the active-sum triangle `pch >= 0`, `pdch >= 0`, `pch + pdch <= S`;
- a lower-bound-safe fixed-mode screen reduces simultaneous circulation from ~`0.5 S` to ~`0.01 S` but changes the full objective by only ~`3.23e5` (`0.049%`) and closes only ~`0.42%` of the gap;
- the fixed-mode value still sits about `10.51%` below the nonlinear feasible reference, so the unrestricted MISOCP cannot provide the required planning-scale lower bound;
- the binary stage was correctly not run;
- the remaining relaxation defect is dominated by the meshed TSO AC relaxation, while DSO SOC, transformed OLTC and the ESSO available-energy relaxation are comparatively negligible;
- full-size conic dual certification is unavailable and is no longer an active planning requirement.

The active development branch remains:

`feature/derivative-free-planning`

created from the accepted P5.5-D HEAD with P5.5 history preserved.

P5.6-A built and validated the deterministic nonlinear candidate-evaluation oracle and remains accepted as:

`P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved`.

P5.6-B then stabilized the **declared oracle policy** enough to specify the search mechanics, but exposed two deeper questions: whether different deterministic template generations induce the same investment ranking, and whether direct-start failures are false hidden infeasibility rather than physical infeasibility. P5.6-B is accepted as:

`P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved`.

The next authorized stage is:

**P5.6-C — landscape robustness and oracle coverage gate**.

P5.6-C is the final gate before any derivative-free optimization campaign. It does **not** authorize the 200-evaluation search yet. It must determine whether the frozen T0 surface is representative enough for investment ranking, whether deterministic continuation can rescue false hidden-infeasible points, define the planning-level robustness threshold separately from numerical repeatability, lock deterministic parallel poll semantics, and identify a trustworthy practical MADS/pattern-search implementation path.

## Accepted P5.6-A nonlinear-oracle evidence

The complete original nonlinear coupled system is audited, including the original nonlinear ESSO, all 48 network SMOPFs and all coordinated TSO/DSO/ESSO quantities.

Canonical START-1 base evaluation:

- coordinated residual max `5.551e-17`;
- original nonlinear ESSO max constraint violation `2.035e-13`;
- largest network nonlinear residual `1.093e-05 p.u.` at `DSO5|2030|Winter`, treated as local IPOPT feasibility tolerance rather than a coordination mismatch;
- gross operational cost `829291677.522120`;
- physical salvage `3439.659877`;
- net operational recourse `829288237.862242`;
- investment cost `50000.000000`;
- total planning objective `829338237.862242`.

The old P5.5-D polished value near `729.36e6` is superseded. P5.6-A proved that `95.5%` of the apparent ~13% gain came from an invalid polish convention that moved achieved interface power into fixed TSO `pc` and thereby erased production flexibility cost. With the corrected polish, the genuine START-1 improvement over the ADMM-evaluated branch is ~`9.21e6` (`1.0982%`), mainly slack/flexibility cleanup.

The A12 call-history impurity is closed for the oracle by per-evaluation deep copies, isolated solver-log directories and reset diagnostics. Repeated evaluations of the same candidate under the same declared policy are bit-identical after different call histories.

The candidate API:

- checks first-stage feasibility before operational solves;
- runs nonlinear ADMM, original nonlinear ESSO, exact-consensus polish and full physical audit;
- returns explicit statuses instead of fabricated penalty objectives;
- caches only VALID completed evaluations under a key containing candidate, canonical checksum, configuration/oracle version, start/template id and anchor convention.

The canonical base candidate lies exactly on the minimum energy-to-power ratio `E = 2S`; negative E-only perturbations are therefore first-stage infeasible and must not be used as ordinary search directions.

## Accepted P5.6-B policy evidence

### Best certified incumbent

P5.6-B re-audited the START-2 base result against the same complete original-nonlinear ESSO/network/coupling certificate and promoted it to the current best rigorous feasible nonlinear planning incumbent for the canonical base investment:

- total planning objective `828021090.360850`;
- net operational recourse `827971090.360850`;
- gross operational cost `827974518.717105`;
- physical salvage `3428.356255`;
- investment cost `50000.000000`.

The START-1 value `829338237.862242` remains a valid but weaker feasible point.

### Template chain and frozen recurring start

The deterministic base-template chain does **not** converge to a fixed point over the authorized refinements:

- `T0`: `828021090.360850`;
- `T1`: `827415318.563944`;
- `T2`: `826824028.845478`;
- `T3`: `826405022.193437`;
- `T4`: `825961521.882321`.

Total drift from T0 to T4 is about `-2.059568e6`, with no reliable deceleration. Therefore no claim is made that a stabilized fixed-point template exists.

For reproducibility, the recurring direct search start is frozen **by rule** as:

`T0` = the archived cold solution of the canonical base candidate,

id:

`a81f7f5191dd42dbf50d1726149b8909`.

T0 is deterministic and pure, but knowingly not the best available template. The key unresolved question is whether T0, T2 and T4 induce essentially the same **relative investment landscape** despite different absolute objective levels.

### Anchor policy is locked

Under the 12-candidate P5.6-B population, midpoint and DSO anchoring have the same success rate on operationally evaluated candidates (`8/11 = 72.7%`). Under T0, DSO anchoring rescues none of the midpoint failures. Anchor deltas range from about `-13365.80` to `+11175.98`, candidate rankings change, and Spearman rank correlation is only `0.9048`.

The recurring search-anchor policy is therefore locked to:

`MIDPOINT ONLY`.

Do **not** mix anchors within the recurring search objective. Both anchors may still be evaluated during final incumbent certification and the best VALID fully certified solution retained.

### Recurring start policy is locked, but coverage is incomplete

For the strategic dual-start subset, cold is never better than T0 on objective where both succeed; it is worse by about `1.32e6` to `1.09e7`. Therefore the recurring direct search policy is:

`T0 ONLY`.

Cold is demoted to final-certification / periodic-audit use.

However, the two starts have different failure sets: at least one master-feasible candidate that is `POLISH_FAILURE` under T0 is VALID under cold, and the budget-boundary candidate crashes under both. Therefore a direct T0 failure is **not** evidence of physical infeasibility. This is the principal oracle-coverage issue for P5.6-C.

### Numerical repeatability and search resolution

Under the fully locked T0 + midpoint policy, repeated evaluations are bit-identical. P5.6-B proposed `tau_search = 10.0` against a measured numerical repeatability floor of exactly zero and a smallest adjacent observed objective gap of `33.27`.

P5.6-C must refine the terminology and separate:

- `tau_numerical = 10.0` — numerical comparison threshold on the declared deterministic oracle surface;
- `tau_planning` — investment-improvement robustness threshold derived from cross-template **relative-to-base** landscape variation.

Do not inflate `tau_numerical` to cover template/cold branch differences. Those are heuristic landscape uncertainty, not numerical noise.

### Search coordinates and current method recommendation

The recommended first-stage coordinates are:

`h = E - phi_min*S`, with `phi_min = 2`,

so the search uses `(S,h)` with `S >= 0`, `h >= 0`, `E = 2S + h`. This is a bijective change of variables on the minimum-duration constraint and avoids losing half the native S/E poll directions at the base point. Remaining max-duration, cumulative-capacity, cohort/lifetime and budget constraints are checked exactly; do not silently project infeasible requested points.

P5.6-B selected deterministic MADS/pattern-search as the preferred family, specifically OrthoMADS in principle, because the oracle is expensive, deterministic, nonsmooth, has hidden operational failures, supports caching and allows independent poll evaluations. This is a **design recommendation**, not yet an implementation authorization. P5.6-C must audit whether a trustworthy implementation already exists and must not casually hand-code OrthoMADS incorrectly.

### Search cost

Under T0 + midpoint:

- successful uncached evaluation ~`115 s`;
- blended observed evaluation cost ~`134 s`;
- final multi-start/both-anchor certification ~`740 s`;
- one-off T0 construction ~`518 s`;
- cache hit ~`0.040 s`.

Four upper-level workers are the current recommended maximum. Eight workers risk resource contention; IPOPT process crashes have already been observed under heavy concurrency.

## P5.6-B planner interpretation

P5.6-B settled the **declared** anchor convention, recurring direct start, coordinates, numerical repeatability and provisional search family. What remains unresolved is more precise than the old B verdict wording:

1. **template/branch landscape representativeness** — does T0 preserve investment rankings/improvement signs relative to later deterministic templates T2/T4?;
2. **oracle domain coverage / false hidden infeasibility** — can a deterministic history-independent continuation fallback rescue T0 failures, including remote/budget-near candidates?;
3. **planning-level robustness threshold** — derive `tau_planning` from cross-template relative-delta variation, distinct from `tau_numerical`;
4. **deterministic parallel semantics** — no asynchronous “first finished improvement wins”; future polls must be batch-deterministic;
5. **search implementation path** — audit NOMAD/PyNOMAD/repository support before implementation.

These are the active P5.6-C questions.

# P5.5 ARCHITECTURE STATUS

## P5.5-A — historical PARTIAL, structural inventory accepted

P5.5-A established the actual SRP1 controllable-resource set and the convexity map. Accepted facts remain:

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

The intended convex family remains a lifted **W-space AC SOC/QC relaxation**. DC-OPF or LinDistFlow substitution is not authorized.

## Two-oracle bound principle remains useful; continuous-convex Benders is not yet viable

For planning candidate `x = (S,E)` define:

`R(x)` = globally solved convex-relaxed AC SMOPF recourse;

`Q_AC_feas(x)` = feasible full nonlinear AC SMOPF recourse, eventually AC-polished before being labelled a rigorous upper bound.

Required relation:

`R(x) <= Q_AC*(x) <= Q_AC_feas(x)`.

The lower-bound / feasible-upper-bound sandwich remains the right conceptual benchmark. The nonlinear model remains the physical truth model and upper-bound/incumbent validator. However, P5.5-C shows that the current **continuous** SOC relaxation cannot yet supply a useful certified planning lower bound or a usable cut oracle. A different lower-bound architecture is now under decision, not assumed.

## P5.5-B — reviewed as PARTIAL; major mathematical closures accepted

P5.5-B is accepted as:

`P5.5-B PARTIAL — one or more lower-bound/cut/interface issues remain unresolved`.

The `PARTIAL` status does **not** reflect a failed formulation. It reflects two items that can only be closed on the implemented real convex model: relaxation tightness/usefulness and formal cut certification.

### B1 accepted — only production-safe tightening

The dormant ±30-degree angle constraints are **not** part of the nonlinear production feasible set and must not be enabled in a certified lower-bound relaxation merely because rules exist in the code.

Only constraints that are:

- active in production; or
- mathematically implied by the production feasible set

may tighten the convex LB model.

Retain production `WijR >= 0`. Derive any further QC bounds only from active voltage bounds, active branch limits, the rank/SOC relation and other production-safe implications.

### B2 accepted — transformed continuous OLTC eliminates explicit tap bilinearity

For each DSO transformer use:

`U_i = r^2 * W_ii`

`C_ij = r * W_ij^R`

`D_ij = r * W_ij^I`.

All production transformer P/Q expressions become affine in `U_i`, `C_ij`, `D_ij`, `W_jj`, verified against the production equations to floating-point noise (~`5.9e-15`).

The physical transformer rank relation becomes:

`C_ij^2 + D_ij^2 = U_i * W_jj`

and is relaxed by the same rotated SOC used on ordinary branches:

`C_ij^2 + D_ij^2 <= U_i * W_jj`.

Continuous tap existence is represented exactly by:

`r_min^2 * W_ii <= U_i <= r_max^2 * W_ii`,

with `W_ii > 0` and recoverable `r = sqrt(U_i/W_ii)`.

For SRP1 there is no tap cost, tap-movement penalty, intertemporal tap coupling, discrete tap position or phase shift. Therefore `r` and `r_sqr` may be eliminated from the convex LB model. The earlier A-stage McCormick proposal is superseded.

### B3 accepted — P/Q/voltage semantics and available-capacity coupling now implemented in P5.5-C

The DSO `REF` generator is the mathematical representation of TSO import/export rather than a separately costed physical generator. The TSO ADN quantities are load-positive; DSO REF `pg/qg` are generation-positive; shared-ESS `pnet/qnet` are load-positive.

The first centralized convex prototype must retain separate TSO/DSO/ESSO copies and replace zero-residual ADMM consensus by exact affine equalities rather than immediately collapsing variables.

P5.5-C confirmed and implemented the planner correction that production passes ESSO **available** S/E capacities to the TSO/DSO operational models. The centralized model therefore imposes exact capacity coupling, after consistent unit conversion:

`S_network_TSO = S_network_DSO = S_available_ESSO`

`E_network_TSO = E_network_DSO = E_available_ESSO`.

Network SOC limits and anchors must use `E_available`, not rated investment E directly.

### B4 accepted — objective-level ESSO relaxation is lower-bound safe

The convex LB oracle may drop:

- ESS charge/discharge complementarity and H1 hats/links;
- the non-negative complementarity objective penalty;
- ESSO degradation and SoH equalities;
- minimum-SoH restriction;
- associated non-negative slack penalties.

The minimal convex ESSO energy relation is:

`0 <= E_available <= E_rated`.

This enlarges the feasible set and is jointly affine in available/rated E. The retained physical cost terms remain unchanged. The resulting LB may be weak, but its direction is safe.

### B5 planner correction — salvage must be a maximum-credit affine lower-bound term

Current net recourse is gross operating cost minus terminal salvage. Salvage is therefore a **credit**: a larger salvage value lowers the objective.

For a rigorous planning lower bound, the master must subtract an **upper bound on physically attainable salvage**. Under the intended non-negative salvage coefficients, the maximum credit occurs at terminal `E_available = E_rated` / `SoH = 1`, not at `soh_min`.

P5.5-C completed the sign trace and derived the exact affine term:

`V_salvage_max(x) = sum gamma[node,cohort] * E_investment[node,cohort]`

and use:

`-V_salvage_max(x)`

in the planning lower-bound objective. The explicit salvage formula has no direct S coefficient once this safe affine bound is used.

Salvage is excluded from operational `R(x)`.

### B6/B7 accepted as prototype evidence, not formal cut certification

Gurobi is the selected prototype solver and `gurobi_persistent` is the preferred interface.

A toy convex capacity problem verified:

- fixing-row dual matches the analytic value-function derivative;
- `QCPDual=1` returns conic duals;
- an `ObjVal`-anchored cut can be slightly invalid at the generating point;
- an `ObjBound`-anchored cut passed a 12-point sweep.

However, this does **not** yet prove that:

`[ObjBound(x_k) - sigma] + g_k^T (x-x_k)`

is globally valid on the real model. A scalar primal-dual gap protects the anchor, but does not by itself certify slope error away from the anchor.

The desired rigorous contract remains one affine dual function:

`L_k(x) = beta_k + g_k^T x`

with `beta_k` and `g_k` derived from one demonstrably dual-feasible conic solution, so weak duality proves:

`L_k(x) <= R(x)`

for every admissible x.

P5.5-C investigated this on the real convex model and found no usable full-model dual certificate; the replacement planning master therefore remains blocked.

## Gurobi is the selected convex-oracle solver

Canonical environment provides:

- `gurobipy 13.0.1`;
- academic licence valid through `2027-04-10`;
- `gurobi`, `gurobi_direct`, and `gurobi_persistent` available;
- `QCPDual=1` supported;
- linear and quadratic/conic duals available;
- `ObjVal`, `ObjBound`, and reported gap available.

Use `gurobi_persistent` first because planning-candidate updates will change capacity-fixing RHS values repeatedly and stable component-to-row handles are useful for dual extraction. `gurobi_direct` is the fallback.

For a scalar convex lower bound, use `ObjBound`. Do not treat `ObjVal` as a certified LB.

---

## P5.5-C — completed PARTIAL; continuous conic oracle implemented and diagnosed

Accepted implementation/evidence from the canonical `opf_env_py311` run:

- additive convex-oracle path only; nonlinear production SMOPF/ADMM/H1/D2-P/master untouched;
- parent model size `322596` variables, `244152` constraints and `35549` Gurobi second-order cones across 48 AC blocks;
- transformed OLTC implementation remains numerically exact and contributes negligible relaxation gap (`rho_tr` worst about `4.66e-06`);
- local nonlinear-to-convex mapped-point tests pass: objective accounting closes to machine precision and block-local violations match the nonlinear models' own residuals;
- the full-parent mapped state is **not yet an exact global-feasibility proof** because its source is an ADMM-converged state with nonzero interface consensus residuals (about `1.341 MW` active-power mismatch at worst and nonzero voltage mismatch);
- continuous full-model primal solve is reproducible with `ObjVal = 653192095.073059`, but Gurobi returns no dual-backed full-model certificate (`ObjBound = -inf` / QCP dual failure); this value is therefore **not a certified lower bound**;
- using that uncertified primal only as a tightness diagnostic gives a recourse gap of about `21.9%` to the best known nonlinear feasible base branch `836586463.43`, roughly `256-259 x` canonical `tol_cut`;
- dominant looseness is relaxed shared-ESS mode switching: `min(pch,pdch)/S` reaches about `0.49893`;
- second-order looseness is on the meshed TSO: AC rank gap worst about `2.4499e-02` and single-cycle angle inconsistency about `1.2344e-02 rad`;
- DSO SOC relaxation is essentially exact in comparison; ESSO `E_available/E_rated >= 0.99754`, so the relaxed degradation-energy interval is not a material gap source;
- the real-model sampled cut remains unusable: on the largest certifiable restriction the S/E dual signal is tiny/inconsistent with finite differences and apparent cut safety is dominated by solver `ObjBound-ObjVal` slack.

Planner interpretation:

1. Do **not** spend another stage trying to certify the current 22%-loose continuous oracle.
2. Do **not** add an arbitrary continuous circulation cap; because `(S,0)` and `(0,S)` are both H1-feasible operating modes, any convex set containing both contains `(S/2,S/2)`. The observed ~0.5S circulation may be intrinsic to the continuous convex hull of the mode disjunction.
3. A lower-bound-safe disjunctive outer approximation of H1 is worth one controlled diagnostic before abandoning rigorous lower bounds.
4. The planning master remains blocked.

# HISTORICAL ACTIVE STAGE — P5.6-C landscape robustness and oracle coverage gate

P5.6-C is the **final pre-search gate**. It must not launch the full derivative-free investment optimization campaign.

Accepted/frozen unless contradicted by C evidence:

- nonlinear production mathematics and solver/ADMM policy;
- `feature/derivative-free-planning`;
- current best certified base-investment incumbent `828021090.360850`;
- recurring direct template `T0`, id `a81f7f5191dd42dbf50d1726149b8909`;
- recurring anchor `MIDPOINT ONLY`;
- recurring direct start `T0 ONLY`;
- coordinates `(S,h)` with `h = E - 2S`;
- `tau_numerical = 10.0` as the provisional deterministic numerical threshold;
- deterministic MADS/pattern-search as the preferred search family in principle;
- four upper-level workers as the practical parallelism ceiling unless new measurements justify otherwise.

## C1 — correct the P5.6-B interpretation

Record that P5.6-B **did** lock the declared midpoint anchor and T0 direct start. The unresolved questions are landscape representativeness, domain coverage/false hidden infeasibility and deterministic search implementation semantics, not whether the anchor policy itself is still open.

## C2 — cross-template investment-landscape test

Use the deterministic P5.6-B benchmark population and evaluate the jointly reachable candidates under:

- `T0`;
- `T2`;
- `T4`;

with midpoint anchor only, a fresh candidate-specific baseline, original nonlinear ESSO, exact-consensus polish and full feasibility audit. Cache keys must include template id.

For every template and candidate compute:

`Delta_Tk(x) = Q_Tk(x) - Q_Tk(base)`.

The deltas, not absolute levels, are the central quantity.

Across jointly VALID candidates report:

- Spearman rank correlation;
- Kendall rank correlation;
- complete rankings;
- max rank displacement;
- number of pairwise ordering reversals;
- worst reversal magnitude;
- per-candidate `Delta_T0`, `Delta_T2`, `Delta_T4`, range and standard deviation;
- diagnostic affine fits `Q_T2 = a2 + b2*Q_T0` and `Q_T4 = a4 + b4*Q_T0`.

Do not infer robustness from correlation alone if a candidate that improves over base under one template becomes worse under another by a material amount.

Classify the landscape as stable only if improvement signs/rankings are sufficiently preserved and template drift is predominantly a common offset. If stable, keep T0 for the recurring search. If unstable, do not launch MADS; return for a fixed better template, fixed ensemble or continuation-oracle decision.

## C3 — deterministic continuation fallback

Direct T0 failure is not physical infeasibility. Test a fixed, history-independent continuation from the canonical base investment `x0` to a master-feasible target `x`:

`x(lambda) = (1-lambda)*x0 + lambda*x`

with fixed schedule:

`lambda = 0.25, 0.50, 0.75, 1.00`.

Because the first-stage feasible set is a polyhedron, every segment point remains first-stage feasible whenever both endpoints are feasible.

Starting from the frozen T0/base operational state, solve each continuation point in order and use each complete VALID state only to initialize the next. Every point must run the original nonlinear ESSO, production ADMM, midpoint-only exact-consensus polish and full feasibility audit. Do not switch anchors and do not use an adaptive lambda schedule in C.

Test continuation on:

- all T0/midpoint `POLISH_FAILURE` candidates from B;
- the budget-boundary `SOLVER_CRASH` candidate;
- at least two T0-direct VALID controls.

For controls compare direct-T0 and continuation objectives. This determines whether continuation is a neutral coverage rescue or a materially different branch-selection policy.

## C4 — lock oracle coverage policy

Compare:

- direct T0 only;
- direct T0, fixed continuation only on failure;
- fixed continuation for every candidate.

Prefer direct-T0 + continuation-on-failure only if continuation materially improves coverage **and** direct-vs-continuation objectives on already-valid controls are compatible enough that fallback does not mix incompatible objective surfaces.

If continuation materially changes valid-control objectives, either use continuation for every candidate or return PARTIAL. Never silently mix two incompatible surfaces.

Recompute success rate and ordinary/failed/blended evaluation costs under the chosen coverage policy.

## C5 — separate numerical and planning robustness thresholds

Use:

`tau_numerical = 10.0`

unless C evidence shows a larger deterministic numerical floor.

Define separately:

`tau_planning`

from cross-template **relative-to-base** delta variation. `tau_planning` governs whether an investment improvement is robust to known branch/template ambiguity. It is not a solver-repeatability tolerance.

Future search acceptance may use `tau_numerical` internally on the locked black-box surface, but promotion/scientific claims must respect `tau_planning`.

## C6 — deterministic parallel polling semantics

Correct the P5.6-B “parallel opportunistic first improvement” idea. Future search must use deterministic **batch polling**:

1. generate the ordered deterministic poll set;
2. apply exact first-stage feasibility checks;
3. remove duplicate/cache-hit points;
4. select the batch by deterministic poll order;
5. evaluate up to four candidates in parallel;
6. wait for the whole selected batch;
7. among all VALID results choose the best objective;
8. break ties within `tau_numerical` by deterministic poll index;
9. only then update incumbent and mesh.

No asynchronous “first process to finish wins” rule is allowed, because it would make the search path depend on OS/solver completion order.

## C7 — search implementation inventory

Without installing anything, audit:

- NOMAD / PyNOMAD availability in the canonical environment;
- existing repository pattern/MADS utilities;
- licences;
- hidden-constraint evaluator support;
- exact linear feasibility screening;
- deterministic direction/seed control;
- cache integration;
- deterministic batch/parallel evaluation.

Do not casually hand-code “OrthoMADS” without its actual mesh/direction requirements. If a trustworthy MADS implementation is unavailable or disproportionate to integrate, a deterministic generalized pattern search with a positive-spanning poll set is acceptable for the first campaign, with its limitations explicit.

## C8 — pre-search classification

End one:

`P5.6-C-A — derivative-free search is ready to launch`

only if template rankings are sufficiently stable, coverage policy is deterministic/acceptable, planning robustness is quantified, deterministic batch semantics are fixed and a practical search implementation path exists;

or

`P5.6-C-B — derivative-free search is ready only on a restricted validated domain`

if the local landscape is robust and nearby coverage is good but remote/budget-near candidates remain unreachable. Any restricted domain must be specified explicitly from tested successful directions;

or

`P5.6-C-C — derivative-free search is not ready`

if template generations materially reorder investment improvements, continuation cannot provide acceptable deterministic coverage, or no practical deterministic search implementation path exists.

Then:

`P5.6-C COMPLETE — ready for planner review before any optimization campaign`.

## Locked production decisions during P5.6-C

Do not change:

- nonlinear AC SMOPF equations;
- active-energy ESS formulation;
- D2-P sensitivity-clean shared-S formulation;
- H1 dimensionless complementarity and `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- IPOPT tolerances/options;
- MA97/exact-Hessian policy;
- recovery/adaptive-rho/proximal policies;
- existing master/Benders equations;
- P5.5 convex/MISOCP diagnostic code.

Do not yet run or productionize:

- the 200-evaluation derivative-free optimization campaign;
- replacement outer planning loop;
- full/production MISOCP planning;
- distributed convex ADMM;
- QCP/Benders cut recovery;
- TSO SDP/QC strengthening.

# COMPLETED STAGE — P5.5-D lower-bound architecture decision gate

Accepted verdict:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

The zero-binary fixed-mode screen was decisive: mode control reduced simultaneous circulation from ~`0.5 S` to ~`0.01 S` but closed only ~`0.42%` of the objective gap. Because `Q_MISOCP <= Q_fixed_mode` for minimization, the unrestricted MISOCP could not close the remaining ~`10.51%` gap, so no binary stage was run. The remaining relaxation defect is dominated by the meshed TSO AC model, but stronger TSO convexification is deferred as optional benchmark work rather than the active planning path.

## Old P5.4-G status

P5.4-G remains permanently blocked for the old nonlinear-recourse derivative cuts. Any later planner is a new architecture.

## Deferred items

Keep deferred during P5.6-C:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R until defensible converter `Smax` data exists;
- nonlinear solver/ADMM retuning;
- distributed convex ADMM;
- production MISOCP planning;
- TSO SDP/QC strengthening except as future diagnostic benchmark;
- Benders/QCP dual-cut recovery;
- actual derivative-free optimization run;
- surrogate-assisted planning unless selected later by explicit planner review.

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

Calendar degradation remains **deferred**. The active-energy SOC/cycling baseline has been validated, but calendar-degradation implementation is not part of P5.5-C and requires separate planner authorization.

Target conceptual extension:

`SoH_cumul[k,y] = SoH_cumul[k,y-1] * SoH_cycle[k,y]^(365 * Delta_y) * phi_cal[k]^Delta_y`

with `0 < phi_cal <= 1` and disabled compatibility case `phi_cal = 1`.

Keep `phi_cal` conceptually separate from the existing calendar-life/retirement parameter used for cohort retirement and salvage.

Do not begin calendar-degradation implementation during P5.5-C.

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

The immediate task is **P5.12-B — cycle-21 cold-RESCALED local-failure forensic
audit** on `feature/derivative-free-planning`.

Use one isolated process and the verified Mac Studio runtime. Reproduce only the
accepted base cold-RESCALED trajectory through cycle 20, preserve the matched
cycle-20 and cycle-21 pre-solve states, attempt cycle 21 once, and terminate the
whole experiment at the first local failure.

Do not continue to cycle 22, rerun the long-horizon experiment, retry with new
solver settings, change any production formulation/parameter, or launch any
candidate comparison or investment search. Stop after the P5.12-B audit and
propose exactly one frozen A/B for planner review.

`LOCAL_NLP_STABILITY_PLAN.md` is authoritative for the complete P5.12-B
execution protocol and stopping rules.
