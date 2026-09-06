# LOCAL_NLP_STABILITY_PLAN.md

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

Read `REVISION_CONTEXT.md` first.

For the current task, this file is authoritative.

# Stage P4 — Production-safe shared-ESS normalization

## 1. Status

P3 diagnostics are COMPLETE.

P3.5-D decisively showed that in-place constant scaling of the original
`sess_snet_def` rows, with KKT-consistent same-object dual transformation,
turns both decisive frozen DSO/TSO failures into primary exact-Hessian
successes without changing the physical feasible set.

The current task is production design and validation, not further diagnosis.

---

## 2. P4 objective

Normalize the production network-side shared-ESS equality:

`(sch - sdch)^2 = pnet^2 + qnet^2`

as:

`kappa_e * ((sch - sdch)^2 - pnet^2 - qnet^2) = 0`

where `kappa_e` is a fixed numerical scale for shared ESS `e`.

Target:

`kappa_e = 1 / S_scale_e`

For the diagnosed active shared ESS:

`S_scale = 0.01`
`kappa = 100`.

Do not use symbolic division by an optimization variable.

Do not create a parallel replacement constraint component.

The production component must remain `sess_snet_def`.

---

## 3. P4.1 — Capacity/model/dual lifecycle audit

Before editing production code, determine:

1. where installed shared-ESS `s_capacity` enters each TSO/DSO SMOPF;
2. the component type and lifecycle of `shared_es_s_rated`;
3. whether local models are rebuilt or reused when planning capacity changes;
4. whether `sess_snet_def` objects survive such updates;
5. whether `model.dual` entries survive;
6. whether warm-start multiplier suffixes survive;
7. whether zero->positive or positive->zero capacity transitions occur on a
   reused model;
8. whether `kappa` can change while the same constraint object persists;
9. whether a fixed/mutable numerical scale parameter already exists.

Report the recommended source for `S_scale_e`.

If scale can change on a live row while its imported dual survives, specify
whether this rule is required:

`lambda_new = lambda_old * (kappa_old / kappa_new)`.

Do not add dual remapping unless the audit proves it is needed.

If lifecycle is ambiguous, STOP.

---

## 4. P4.2 — Shared-ESS production implementation

Proceed only after P4.1 is clear.

Introduce/reuse a repository-consistent fixed or mutable parameter containing
the numerical shared-ESS power scale.

For positive capacity:

`kappa_e = 1 / S_scale_e`.

For zero capacity:
- use a finite safe value such as `1.0`;
- never divide by zero;
- preserve existing zero-capacity operational gating.

Modify the existing `sess_snet_def` rule/component directly so it is constructed
in scaled form from the beginning.

Do NOT:
- add `sess_snet_def_scaled`;
- deactivate `sess_snet_def`;
- change its index set;
- change its component name;
- change `sess_comp`;
- change any ESS variable/bound;
- change SOC;
- change objective or ADMM terms.

If P4.1 requires dual remapping when scale changes on a live reused model,
implement the smallest targeted remapping mechanism and document it.

---

## 5. P4.3 — Construction/equivalence validation

Validate:

### Positive capacity
- `s_rated=0.01 -> kappa=100`;
- `s_rated=0.02 -> kappa=50`;
- scale is numerical/fixed within a solve;
- no symbolic division by a decision variable.

### Zero capacity
- no divide-by-zero path;
- finite scale;
- same operational rows deactivated as before;
- zero operational variables fixed as before.

### Structure
- same `sess_snet_def` component name;
- same index tuples;
- same row count;
- no extra constraint component;
- no `sess_comp` change;
- no SOC/objective change.

### Equivalence
Verify numerically/source-wise that for finite positive `kappa`:

`g = 0 <=> kappa*g = 0`.

If any structural invariant fails, STOP.

---

## 6. P4.4 — Frozen regression

Replay through the normal production solver path.

### Decisive DSO
`data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl`

### Decisive TSO
`data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl`

Report per case:
- status/termination;
- primary iterations;
- objective;
- scaled/unscaled primal infeasibility;
- dual infeasibility;
- complementarity;
- runtime;
- recovery yes/no;
- original unscaled `sess_snet_def` residual.

Required first gate:
- both decisive cases primary success;
- no recovery;
- original physical relation within normal feasibility tolerance.

If either fails, STOP.

If both pass, replay every other preserved residual P3 DSO/TSO failure snapshot
available and report each separately.

---

## 7. P4.5 — Seed-2026 distributed operational smoke

Only after P4.4 is satisfactory.

Run the same reduced distributed operational configuration used in the
post-`vmag_nodes` P2.10 smoke.

Keep unchanged:
- seed 2026;
- scenario checksum identity;
- candidate shared ESS;
- IPOPT options;
- MA97;
- exact-Hessian primary path;
- recovery;
- ADMM tolerances/rho policy;
- TSO proximal regularization;
- common objective scaling;
- ESSO formulation.

Report:
- initialization;
- ADMM cycles;
- every local primary failure;
- every recovery;
- every persistent-for-cycle failure;
- final convergence status;
- final residuals;
- recourse stationarity;
- rho evolution;
- voltage-slack diagnostics;
- runtime.

Compare against the previous P2.10 smoke:

- 14 primary local failures;
- 7 persistent-for-cycle failures;
- convergence in 15 ADMM cycles.

Stop after P4.5 for planner review before any full planning run.

---

## 8. P4.6 — Standard ESS audit/extension gate

Run only if P4.1-P4.5 are satisfactory.

The standard/ordinary ESS has an analogous nonlinear apparent-power/net-power
structure and may benefit from the same normalization.

Do NOT assume equivalence.

Audit:
- exact standard-ESS relation;
- rated-power variable/parameter semantics;
- zero-capacity behavior;
- model rebuild/reuse lifecycle;
- imported constraint-dual lifecycle;
- warm-start behavior;
- indexing;
- existing active standard-ESS cases/tests.

If the relation and lifecycle are equivalent, recommend:

`kappa_es * g_es = 0`

with:

`kappa_es = 1 / S_scale_es`.

Validation requirement:
- identify at least one active standard-ESS case if available;
- verify positive-capacity scaling;
- verify zero-capacity safety;
- verify physical equivalence;
- replay a representative local solve.

If no active standard-ESS test exists, do not silently generalize the production
edit. Return a design recommendation and wait for planner approval.

---

## 9. Strict prohibitions

During P4.1-P4.5 do NOT modify:
- standard ESS equations;
- `sess_comp`;
- shared-ESS SOC;
- standard ESS SOC;
- ESSO degradation/SoH;
- active/apparent throughput definitions;
- solver settings;
- ADMM settings;
- voltage/generator/branch formulations;
- Benders/local-cut logic.

Do not run the full planning problem.

---

## 10. Required worker report

Return:

### A. P4.1 lifecycle audit
- scale source;
- model rebuild/reuse;
- constraint dual lifecycle;
- zero-capacity transitions;
- dual-remapping decision.

### B. P4.2 implementation
- exact diff;
- files/functions;
- scale parameter design;
- zero-capacity behavior;
- dual handling if required.

### C. P4.3 validation
- positive/zero-capacity checks;
- row/index/component invariants;
- feasible-set equivalence.

### D. P4.4 frozen regression
- decisive DSO/TSO;
- all other preserved failures.

### E. P4.5 operational smoke
- local failures/recoveries;
- ADMM convergence/residuals/stationarity;
- runtime;
- comparison with P2.10.

### F. P4.6 standard ESS recommendation
- mathematical equivalence;
- lifecycle equivalence;
- active test availability;
- implement now vs defer.

End with exactly one of:

`P4 PASS — recommend planner approval for reduced planning baseline`

`P4 PARTIAL — planner review required before further execution`

`P4 FAIL — do not proceed`

Then stop.

---

## 11. Worker prompt

> Read `REVISION_CONTEXT.md` first and then `LOCAL_NLP_STABILITY_PLAN.md`.
>
> P3 diagnostics are complete. P3.5-D decisively confirmed poor numerical
> scaling of the original network-side `sess_snet_def` equality without
> changing its physical feasible set or constraint identity/order.
>
> Execute P4.1-P4.5 exactly as specified. First audit the production
> capacity/model/dual lifecycle. If clear, implement the shared-ESS
> normalization directly in the existing `sess_snet_def` component, validate
> construction/equivalence, replay all relevant frozen failures, and run the
> seed-2026 distributed operational smoke with solver/ADMM settings unchanged.
>
> Do not change ordinary/standard ESS equations during P4.1-P4.5.
>
> If P4.1-P4.5 are satisfactory, perform P4.6 as a separate audit/extension
> gate for standard ESS. Generalize the normalization only if its
> equation/capacity/zero-capacity/dual lifecycle is confirmed equivalent and
> an appropriate validation case exists.
>
> Do not change `sess_comp`, SOC, degradation, calendar ageing, solver
> settings, ADMM settings, Benders logic, or unrelated OPF equations.
>
> Do not run the full planning problem.
