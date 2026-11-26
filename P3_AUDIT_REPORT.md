# Stage P3 — Remaining Local SMOPF Structural-Conditioning Audit

Repository: `/Users/micaelsimoes/PycharmProjects/shared-resources-planning`
Branch: `admm_residual_balancing_tests` (HEAD `d06d4595`, on top of accepted checkpoint `feca8618`)
Scope executed: **P3.1 – P3.4 only**. No production formulation, solver, ADMM, ESSO/degradation,
or Benders logic was modified. No A/B experiment (P3.5) was executed.

Tooling note: the frozen `.pkl` snapshots are Pyomo `ConcreteModel` clones, and neither the
device shell nor the analysis container had `pyomo`/network access available to load them
normally. A generic placeholder-class unpickler (`~/p3_audit_scratch/generic_unpickle.py` on the
device, not part of the repo) was written to deserialize the raw variable/parameter state
(`_GeneralVarData`/`_ParamData` slot layout `[value, lb, ub, domain, fixed, stale]`, confirmed
empirically against known bounds such as `vmag ∈ [0.85, 1.15]`) without needing Pyomo installed.
This was read-only forensics; it changed nothing in the repository.

---

## A. Failure snapshot inventory

### A.1 DSO residual failures (`case33_2`, node 7) — P2.10 pattern

| Block | File | SHA-256 | Metadata | Original outcome (per LOCAL_NLP_STABILITY_PLAN.md table) |
|---|---|---|---|---|
| 2025/Summer/cycle1 | `data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Summer_cycle1.pkl` | `0da3dbe21029fc49621fde46ad0717be6ff4fd064712e3a4d8893478d1c34794` | agent=DSO, node_id=7, network=case33_2, year=2025, day=Summer, cycle=1, warm_start=True | primary "Error in step computation"; recovery `maxIterations` (persistent) |
| 2030/Winter/cycle1 | `.../P3Preserved/frozen_DSO_node7_case33_2_2030_Winter_cycle1.pkl` | `ec058125d7eb088b092312863a759d9c3ced6421675d3b8fb1ea2e6ceb4708fb` | node_id=7, network=case33_2, year=2030, day=Winter, cycle=1 | primary "Error in step computation"; recovery "Error in step computation" (persistent) |
| 2025/Autumn/cycle8 | `.../FrozenSMOPF/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl` (= `P3Preserved/…cycle8.pkl`, identical) | `066117b88085e5d8b20ec4da684555902d57565044d4cf293516796637074711` | node_id=7, network=case33_2, year=2025, day=Autumn, cycle=8 | primary "Error in step computation"; recovery "Error in step computation" (persistent) |
| 2025/Autumn/cycle12 | `.../frozen_DSO_node7_case33_2_2025_Autumn_cycle12.pkl` (= P3Preserved, identical) | `113909c05b89fbf71ca16e5470c3a48a4f0bdc258057e227f33b50444c8c3e23` | cycle=12 | primary "Error in step computation"; recovery `maxIterations` (persistent) |
| 2025/Autumn/cycle13 | `.../frozen_DSO_node7_case33_2_2025_Autumn_cycle13.pkl` (= P3Preserved, identical) | `92e0a1d5ada0f8565e9dce6431775f2330f961fa151b938fbf15264f372471cd` | cycle=13 | primary "Error in step computation"; recovery `maxIterations` (persistent) |

**Reproducibility note.** For Autumn cycle8/12/13 the top-level `FrozenSMOPF/` copy and the
`P3Preserved/` copy are byte-identical. For Summer/cycle1 and 2030-Winter/cycle1 the two copies
have **different SHA-256** because the top-level file was overwritten by the later replay run that
was executed specifically to capture the TSO snapshots below. Direct comparison of the
deserialized primal content (`shared_es_e_rated`, `shared_es_pnet`, `vmag`, `pg`) between the two
copies shows **zero numerical difference** — the hash divergence is pickling-order/`id()`-based
non-determinism (e.g. `ComponentMap`/weakref ordering inside `Suffix`/`dual`), not a different
physical state. The `P3Preserved/` copies are treated as the hash-of-record for this audit; do not
overwrite them further.

### A.2 TSO snapshots (newly captured for P3.1)

No TSO pre-solve capture existed before this stage. The minimal diagnostic hook already present in
the working tree (commit `b2d794e2`, `_save_frozen_network_block` in `shared_resources_planning.py`,
wired into `network_data.py`'s `optimize()` via a new `pre_solve_snapshot_callback` argument) was
exercised by re-running the exact P2.10 seed-2026 SRP1 configuration far enough to reach ADMM cycles
6–7 (commit `d06d4595` added `audit_p3_snapshots.py`, the comparison utility referenced in
`LOCAL_NLP_STABILITY_PLAN.md` §5). This did **not** touch solver/ADMM settings and did not run the
full planning problem — only the operational replay needed to reach the target cycles.

| Block | File | SHA-256 | Metadata | Outcome |
|---|---|---|---|---|
| case9/2025/Winter/cycle5 | `data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Winter_cycle5.pkl` | `228fedcda91c8db5e4f3d72a886c1f27614a35362da96e39ab36c291295df728` | agent=TSO, network=case9, year=2025, day=Winter, cycle=5, warm_start=True, label=failure | `status=warning, termination=maxIterations` |
| case9/2025/Summer/cycle6 | `data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl` | `51d9097418561612d61367d12600ea3929b622c367a5feeef1ab6efd4b08355e` | network=case9, year=2025, day=Summer, cycle=6, label=failure | `status=warning, termination=maxIterations` |

Both are genuine `case9`/2025 persistent-failure blocks named in the P2.10 table
(`LOCAL_NLP_STABILITY_PLAN.md` §2).

### A.3 Matched successful comparators (added for P3.3, per the plan's "smallest diagnostic capture" allowance)

| Block | File | SHA-256 | Outcome |
|---|---|---|---|
| DSO node7/case33_2/2025/Autumn/cycle7 | `matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` | `8eabd9ee566182a887e2e790e8ab7993ab922e22d79e3712fdf9288fb726acbe` | `status=ok, termination=optimal` |
| TSO/case9/2025/Summer/cycle7 | `matched_success_TSO_case9_2025_Summer_cycle7.pkl` | `15ce6ebef2511655b7202652cbce9327ef4295773458ddc2c9b7f94fc5daa8e9` | `status=ok, termination=optimal` |

These satisfy plan §5.1/§5.2 ("nearest available successful pre-solve state for this exact block" /
"same network/year/day"): Autumn cycle7 immediately precedes the failing cycle8, and Summer cycle7
immediately follows the failing cycle6 for the same `case9`/2025/Summer block.

No ESSO, `case33_1`, or `case33_3` failure snapshot exists, consistent with §2 of the plan (none
occurred in the P2.10 run).

---

## B. Nonlinear auxiliary-structure audit

### B.1 Priority A — network-side shared-ESS representation

All shared-ESS network-side variables/constraints live in `model_construction_helpers.py`
(rules, lines 699–811) and are registered in `network.py` (lines ~430–450); the zero-capacity
gating logic is `model_construction_helpers.py:813–897`.

| Component | File:line | Definition | Consumed by | Exists at zero capacity? | Gradient at (P,Q,S)=0 |
|---|---|---|---|---|---|
| `shared_es_pnet`/`sess_pnet_def` | 781/L436-ish | `pnet = pch − pdch` | node balance, ADMM `p_ess_req` coupling | Yes, but **deactivated** (see below) | linear, non-degenerate |
| `shared_es_qnet` | var only | free var, driven by `sess_phi_limit_*` | node balance, ADMM `q_ess_req` | deactivated when inactive | n/a (no defining equality) |
| `sess_pch_link` / `sess_pdch_link` | 716-721 | `pch ≤ sch`, `pdch ≤ sdch` | couples active/apparent | deactivated when inactive | linear inequality, always well-conditioned |
| `sess_s_limit` | 724-725 | `sch+sdch ≤ s_rated` | rated-capacity coupling | deactivated when inactive | linear, well-conditioned |
| **`sess_snet_def`** | **728-730** | `(sch−sdch)² = pnet² + qnet²` | node balance (via pnet/qnet), ADMM consensus | deactivated when inactive | **∂/∂sch,∂/∂sdch,∂/∂pnet,∂/∂qnet all → 0 as (sch,sdch,pnet,qnet)→0** |
| `sess_phi_limit_lower/upper` | 700-713 | linear power-factor cone in `pch,pdch,qnet` | reactive capability | deactivated when inactive | linear, well-conditioned |
| **`sess_comp`** (BILINEAR_RELAXATION, active model in SRP1) | **743-755** | `sch·sdch ≤ tol·s_rated²` | prevents simultaneous charge/discharge | deactivated when inactive | **∂/∂sch = sdch → 0, ∂/∂sdch = sch → 0 as (sch,sdch)→0** (only `∂/∂s_rated` stays non-zero) |
| `sess_soc_def` / `sess_soc_limit_*` / `sess_soc_final` | 733-779 | linear SOC recursion + bounds | day coupling | deactivated when inactive | linear, well-conditioned |
| `shared_es_s_rated`/`e_rated` + `_fixed` sensitivity constraints | 903-908 | `rated_fixed == rated` | Benders local sensitivities | always active (capacity variables themselves) | linear |

**Zero-capacity gating already exists and works correctly.** `configure_shared_ess_operational_state`
(`model_construction_helpers.py:863-897`) fixes all nine shared-ESS operational variables to `0.0`
and **deactivates** every constraint in `_SHARED_ESS_OPERATIONAL_CONSTRAINTS` (including
`sess_snet_def` and `sess_comp`) for any shared-ESS index whose `s_capacity`/`e_capacity` is below
`SHARED_ESS_ZERO_CAPACITY_TOLERANCE = 1e-10` (`definitions.py:82`). This was confirmed directly in
the frozen TSO snapshot: `shared_energy_storages = range(0,3)`, `shared_es_e_rated = [0.0, 0.0241, 0.0]`,
and indices 0 and 2 have every operational variable `fixed=True, value=0.0` — i.e. their nonlinear
rows are **not present in the active NLP at all**. This is exactly why `case33_1`/`case33_3`
(whose own shared-ESS candidate has ~zero installed capacity in this candidate solution) never
fail: they don't carry these rows.

**The genuinely-installed interface is the risk.** Index 1 in the TSO (mirrored as the single
`shared_energy_storages` entry, `range(0,1)`, in the `case33_2`/node-7 DSO model) has real installed
capacity (`s_rated=0.01`, `e_rated≈0.024` p.u.) and its `sess_snet_def`/`sess_comp` rows are **active**
— but every captured failing pre-solve state shows this interface dispatching at chronically tiny
magnitude (see §C). At that operating point the two nonlinear rows above are not "eliminated by
capacity gating" (correctly, since the device is real), but their gradients with respect to the four
operating variables that matter (`sch, sdch, pnet, qnet`) collapse toward zero simultaneously — the
exact "S² = P² + Q²"/"complementarity product" pattern the plan asked to check for.

Ordinary (non-shared) `es_*` ESS rules (`model_construction_helpers.py:618-698`) have the identical
mathematical structure (`ess_snet_def_rule`, `ess_comp_rule`), but are **not a factor here**: both
`case33_2` and `case9` construct with `energy_storages = range(0,0)` (no ordinary ESS device), confirmed
directly in the frozen snapshots.

### B.2 Priority B — other nonlinear auxiliary families (code-level scan, no tunnel vision)

| Family | File:line | Relation | Zero/near-zero condition | Present in non-failing networks too? | Risk |
|---|---|---|---|---|---|
| Voltage (`vmag_nodes`) | `model_construction_helpers.py:421-426` | `vmag_sqr=e²+f²`, `vmag_sqr=vmag²` | never near zero (`vmag≈1 p.u.`) | yes, everywhere | Accepted/closed (P2); not reopened |
| Branch voltage products / `r_sqr` | 453-467, 1188-1190 | bilinear `e_i·e_j+f_i·f_j` etc., `r_sqr=r²` | `r≈1` (transformers only), never near 0 | yes | Low — operating point far from singularity |
| Branch apparent-flow squares | `compute_branch_flow_squared` (991-1021), `flow_ij_sqr`/`flow_ji_sqr` | `S_ij²=P_ij²+Q_ij²`-type | true at every near-zero-loaded branch, every hour | **yes, in every network including case33_1/3** | Same vanishing-gradient family, but **not discriminating** — present everywhere without correlating to the observed failure pattern; lower priority |
| Generator apparent power | `sg_sqr_rule` (498-502), `sg_capability` (constraint at `network.py:411`) | `sg_sqr=pg²+qg²`, `sg_sqr≤sg_avail²` | true whenever a generator is dispatched at ~0 (e.g. curtailed/idle hours) | yes, every network, every generator | Same family; structurally plausible but **not yet distinguished** by matched-state evidence in this stage (not inspected per-generator in the failing snapshots) |
| RES availability | `renewable_available_apparent_power` (167-170) | `sqrt(pg²+qg²)` | n/a — this is a **plain Python float** used only to initialize a `Param`, not a live model equality/derivative | — | None (not a Jacobian row) |
| Flexibility apparent balance | `flex_energy_balance_s_rule` (596-610) | `Σ(flex_p+flex_q)²` | — | — | **Not constructed** — `network.py:420` shows the `Constraint(...)` call commented out. Dead code, no numerical exposure today |
| Power-factor profile / capability-circle (gen) | 512-553 | linear-in-tangent inequalities | — | yes | Linear, well-conditioned |

### B.3 Ranked structural inventory (plan §4.3 format, abbreviated to the material rows)

| Rank | Family | File/rule | Vars | Relation | Zero condition | Auxiliary/redundant? | Active in failing DSO? | Active in failing TSO? | Active in `case33_1`/`3`? | Only when shared-ESS candidate present? | Risk | Next diagnostic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Shared-ESS `sess_snet_def` | `model_construction_helpers.py:728-730` | `shared_es_sch,sdch,pnet,qnet` | `(sch−sdch)²=pnet²+qnet²` | dispatch → 0 | Yes — apparent-power definitional equality on a nearly-idle device | **Yes** (idx of node-7 interface) | **Yes** (TSO idx 1) | No (rows fully deactivated by capacity gating) | **Yes** | High | P3.5 below |
| 2 | Shared-ESS `sess_comp` (bilinear) | `model_construction_helpers.py:743-755` | `shared_es_sch,sdch` | `sch·sdch ≤ tol·s_rated²` | both → 0 | Yes — complementarity product | Yes | Yes | No | Yes | High (compounds #1 on the same variables) | P3.5 below |
| 3 | Generator `sg_sqr`/`sg_capability` | `model_construction_helpers.py:498-509` | `pg,qg,sg_avail` | `pg²+qg²≤sg_avail²` | near-zero dispatch hours | Yes | Present, not yet isolated | Present, not yet isolated | **Also present** (doesn't fail) | No | Medium-low (present everywhere, not discriminating) | Only if #1/#2 elimination fails to resolve — check whether any generator is at literal 0 output in the same period/scenario as the failing step |
| 4 | Branch `flow_ij_sqr`/`flow_ji_sqr` | `model_construction_helpers.py:991-1021,1265-1288` | `pij,qij,pji,qji` | `S²=P²+Q²` | lightly-loaded branch/hour | Yes | Present | Present | **Also present** (doesn't fail) | No | Low (ubiquitous, non-discriminating) | Not recommended next |
| 5 | Flexibility apparent balance | `flex_energy_balance_s_rule` | n/a | — | — | — | **Not constructed** (dead code) | — | — | — | None | None |

---

## C. Matched-state comparison

### C.1 Primary DSO comparator — `case33_2`/2025/Autumn, node 7 (cycle8 fail vs cycle7 success)

The single active shared-ESS index (`shared_es_e_rated[0] ≈ 0.0242` p.u., `s_rated=0.01` p.u.,
fixed) has essentially the same tiny operating point in both the failing and the successful replay
of the same block:

| Quantity | cycle7 (success) | cycle8 (fail) |
|---|---|---|
| `shared_es_pch` range | `[7.069e-07, 6.869e-03]` | `[7.063e-07, 6.883e-03]` |
| `shared_es_pdch` range | `[9.953e-07, 7.488e-03]` | `[9.931e-07, 7.503e-03]` |
| `shared_es_pnet` range | `[-7.487e-03, 6.868e-03]` | `[-7.502e-03, 6.882e-03]` |
| `shared_es_qnet` range | `[-2.325e-05, 3.290e-03]` | `[-2.259e-05, 3.292e-03]` |
| `shared_es_e_rated[0]` | `0.0241709` | `0.0241716` |
| `rho_ess` (ADMM penalty) | `3.375` | `3.375` (identical) |
| `p_ess_req[0..2]` | `-4.502e-05, -7.353e-03, -1.019e-03` | `-4.873e-05, -7.357e-03, -1.021e-03` |
| `dual_ess_p_req[0..2]` | `2.022e-03, 1.498e-03, 1.407e-03` | `1.174e-03, 1.332e-03, 1.000e-03` |

Interpretation: the **primal ESS block is essentially unchanged** between the block that failed
(cycle8) and the one that converged one cycle earlier (cycle7) at the same node — the near-zero
dispatch identified in §B.1 is present identically in *both*. The ADMM dual/target parameters
(`p_ess_req`, `dual_ess_p_req`) drift by 10–70% cycle-to-cycle at this stage (still-converging
outer ADMM iterate), which is the plausible trigger that pushes an already ill-conditioned local
KKT sub-block (rank 1/2 in §B.3) past IPOPT's exact-Hessian step-computation tolerance on some
cycles and not others.

### C.2 Primary TSO comparator — `case9`/2025/Summer (cycle6 fail vs cycle7 success)

This pair is far more striking: **every primal quantity checked is bit-identical** between the
failing cycle6 pre-solve state and the successful cycle7 pre-solve state — `e`, `f`, `vmag`, `pg`,
the full `shared_es_pch` array, `p_ess_req`, `dual_ess_p_req`, `dual_vmag_req` all compare
element-wise equal. The **only** difference found in the (partially-checked) parameter set is
`vmag_req` — the ADMM voltage-consensus target received from the coordinating side — which differs
in 68 of 72 entries by at most `5.8e-10` (machine-precision-scale).

| Quantity | cycle6 (`maxIterations`) | cycle7 (`optimal`) | max abs diff |
|---|---|---|---|
| `e`, `f`, `vmag`, `pg` | — | — | **0** (bit-identical) |
| `shared_es_pch` (all 72 entries) | — | — | **0** |
| `p_ess_req`, `dual_ess_p_req`, `dual_vmag_req` | — | — | **0** |
| `vmag_req` | — | — | `5.8e-10` (68/72 entries changed) |

Interpretation: this is the strongest single piece of evidence in this audit. An NLP instance that
is otherwise **identical down to machine precision** flips from "Maximum Number of Iterations
Exceeded" to "Optimal Solution Found" under a `~1e-10`-scale perturbation of one ADMM parameter
(`vmag_req`, the voltage-consensus target at the same node-7 interface that hosts the near-zero
shared-ESS dispatch). A well-conditioned NLP does not have its solvability decided at the
`1e-10` scale; this is the signature of an extremely flat/near-singular region of the KKT system in
the neighborhood of the current iterate — consistent with, and localized at, the same shared-ESS
interface identified structurally in §B. It also suggests the fragile direction couples the
near-zero ESS block with the **voltage** variable at that same node (not the ESS request itself,
which is untouched), i.e. the conditioning issue is not purely internal to the ESS block in
isolation — it interacts with the interface voltage tracking term at the same node.

(Warm-start dual suffixes `ipopt_zL_in`/`zU_in` were also inspected; their internal representation
is an id()-keyed `ComponentMap` that the placeholder unpickler cannot safely resolve to readable
diffs without the real Pyomo classes, so no quantitative suffix diff is reported here. The primal
evidence above is sufficient to support the ranking in §D without it.)

---

## D. Candidate ranking

**Candidate #1 (leading): `sess_snet_def` (`(sch−sdch)² = pnet² + qnet²`) at the single genuinely
installed shared-ESS interface (network-side, node 7 / `case33_2`, mirrored in `case9`'s TSO model).**
Evidence: exact code match to the "S²=P²+Q²" pattern the plan flagged; the constraint's gradient
w.r.t. all four participating variables provably → 0 as the operating point → 0; every captured
failing snapshot (DSO Summer c1, Winter c1 2030, Autumn c8/12/13; TSO Winter c5, Summer c6) shows
this exact interface dispatching at `1e-9`–`1e-2` p.u.; the family is **absent** (deactivated) at
`case33_1`/`case33_3`, exactly matching "no failure there"; and the TSO matched pair (§C.2) shows a
`1e-10`-scale perturbation flips solver outcome, which is the classic signature of a near-singular
sub-block rather than a normal Newton step failure.

**Candidate #2: `sess_comp` (`sch·sdch ≤ tol·s_rated²`, BILINEAR_RELAXATION) at the same interface.**
Same variables (`sch, sdch`), same vanishing-gradient behavior at the origin, active only where #1
is active. Ranked #2 rather than tied with #1 because its gradient w.r.t. `s_rated` remains
non-zero (the row is not *fully* degenerate, only degenerate in the `sch,sdch` sub-space), so on its
own it is somewhat less singular than #1 — but it sits on exactly the same tiny variables and very
plausibly compounds #1's conditioning problem in the same local Jacobian block.

**Candidate #3: generator `sg_sqr`/`sg_capability` at near-zero dispatch.** Structurally identical
"S²=P²+Q²" family, but not yet connected to the failing states by direct evidence (no per-generator
zero-dispatch check was performed on the failing snapshots in this stage) and, unlike #1/#2, this
family is *also* present in `case33_1`/`case33_3`, which never fail — weakening it as the primary
explanation for the observed failure pattern, though it could still be a secondary contributor.
Branch `flow_ij_sqr`/`flow_ji_sqr` (also "S²=P²+Q²", also ubiquitous and non-discriminating) is
judged even less likely than #3 for the same reason and is not carried forward as a separate
candidate.

#1 is preferred over #2 and #3 because it is the only family whose *presence* maps exactly onto the
observed failure/non-failure partition across networks (active only at the node-7 interface and its
TSO mirror; deactivated everywhere else by the existing, already-correct zero-capacity gating) and
because the matched-state evidence in §C.2 is best explained by a near-singular block located
exactly there.

---

## E. Recommended next frozen experiment (NOT implemented)

1. **Suspect family:** `sess_snet_def` and `sess_comp` (definitions in D/§B.1), restricted to the
   single shared-ESS index with real installed capacity (TSO `case9` index `1`; DSO `case33_2`
   node-7's only index `0`).
2. **Files/functions:** `model_construction_helpers.py:728-730` (`sess_snet_def_rule`),
   `model_construction_helpers.py:743-755` (`sess_comp_rule`); registered at `network.py:438` and
   `network.py:448`.
3. **Mathematical structure:** `(sch−sdch)² = pnet² + qnet²` and `sch·sdch ≤ tol·s_rated²`; both
   have Jacobian rows (w.r.t. `sch,sdch,pnet,qnet`) that vanish as the operating point → 0.
4. **Evidence connecting to failing states:** §C.1 and §C.2 — every failing snapshot dispatches
   this exact interface at `1e-9`–`1e-2` p.u.; the TSO comparator shows a `~1e-10` parameter
   perturbation flips the solver outcome.
5. **Evidence connecting to shared-ESS node 7 / TSO:** §B.1 — the family only exists (is only
   *active*, i.e. not deactivated by `configure_shared_ess_operational_state`) at the node-7
   interface and its TSO-side mirror; `case33_1`/`case33_3` never carry active rows of this family.
6. **Why more plausible than the next-ranked alternative:** unlike generator/branch "S²=P²+Q²"
   rows (candidate #3 and the branch family), this is the only candidate whose presence/absence
   exactly tracks which blocks fail; it is also the only candidate directly implicated by the
   bit-identical-except-for-one-parameter TSO comparator.
7. **Proposed test type:** **pure diagnostic elimination** (plan §6, option a) — not a fix, not a
   zero-capacity conditional removal (this device is not zero-capacity), not a production change.
8. **Exact invariant list:**
   - Operate only on an in-memory `clone()` of the frozen pickle, inside a standalone diagnostic
     script — never on `network.py`'s live constraint-construction path, never on the running ADMM.
   - Touch only `sess_snet_def` and `sess_comp` for the one real shared-ESS index in each clone
     (`Constraint.deactivate()`, not deletion of variables); every other constraint, variable,
     bound, and objective term stays exactly as captured.
   - IPOPT options, `acceptable_iter`, tolerances, linear solver (MA97), Hessian mode, and all
     warm-start push settings stay byte-identical to the frozen case file / `_create_smopf_solver`
     defaults used to produce the original failure.
   - No ADMM parameter (`rho_*`, `*_req`, `dual_*_req`) is touched.
   - Compare against an **untouched** second clone of the same frozen file solved with the current
     (accepted) formulation, so the only difference between the A and B runs is the two deactivated
     rows.
9. **Exact frozen case(s) to test:**
   - DSO: `data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl`
     (persistent "Error in step computation" / recovery "Error in step computation").
   - TSO: `data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl`
     (primary "Error in step computation" pattern class; recovered via limited-memory to
     `maxIterations` originally, and to `optimal` one cycle later per §C.2).
10. **Acceptance criterion:** deactivating `sess_snet_def`+`sess_comp` for that one index, with the
    identical exact-Hessian MA97 solver settings, converts the primary solve from
    `internalSolverError`/`maxIterations` to a primary success (`ok`/`optimal` or `acceptable`) on
    both frozen cases → confirms candidate #1/#2 as the proximate numerical cause. If either case
    still fails identically, the hypothesis is not confirmed and the audit should fall back to
    candidate #3 (generator dispatch) or to the deferred broad Jacobian/rank diagnostics
    (plan §10).
11. **Rollback procedure:** none needed at the repository level — the experiment reads a frozen
    `.pkl`, clones it in memory, and discards the modified clone after reporting; `network.py`,
    `model_construction_helpers.py`, solver configuration, and ADMM code are not edited to run it.

---

## F. Decision

`P3 AUDIT COMPLETE — waiting for planner approval of next A/B experiment`
