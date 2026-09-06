# REVISION_CONTEXT.md

Repository:
`/Users/micaelsimoes/PycharmProjects/shared-resources-planning`

Read this file first, then `LOCAL_NLP_STABILITY_PLAN.md`. For the current
local-NLP task, the latter is authoritative.

## Current reproducibility identity

- Seed: `2026`
- Scenario checksum:
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`
- NLP solver: IPOPT
- Primary linear solver: MA97
- Primary Hessian: exact
- Limited-memory Hessian: recovery only

Do not change IPOPT tolerances, warm-start settings, MA97, ADMM parameters,
TSO proximal regularization, common ADMM objective scaling, or recourse
stationarity settings during the current workstream.

---

# 1. Accepted local-NLP work

## 1.1 `vmag_nodes` refactor

Accepted checkpoint:

`feca8618b21ef8d7ae72202201e9f7af79397dbc`

The production refactor retains explicit `vmag` only where it is genuinely
consumed:
- DSO reference/interface node;
- TSO active DSO-interface buses.

All physical buses retain `e`, `f`, `vmag_sqr`, and
`vmag_sqr = e^2 + f^2`.

The original cycle-10 DSO failure and seven additional historical frozen
failures became primary exact-Hessian successes.

Treat the old all-node explicit-`vmag` issue as resolved.

---

# 2. Residual shared-ESS failure family

After `vmag_nodes`, the seed-2026 distributed operational smoke still exposed
local NLP failures:

- all residual DSO failures were in `case33_2` / node 7;
- no `case33_1` or `case33_3` failures;
- several TSO failures occurred;
- node 7 is the genuinely installed shared-ESS interface;
- the TSO contains the mirror of that same shared-ESS interface;
- ESSO itself did not fail.

This led to a structural audit of the network-side shared-ESS formulation.

---

# 3. P3 audit conclusion

The leading nonlinear equality was:

`sess_snet_def`

`(sch - sdch)^2 = pnet^2 + qnet^2`

with row gradient:

`[2(sch-sdch), -2(sch-sdch), -2pnet, -2qnet]`.

At low shared-ESS dispatch this row is weakly scaled.

Existing zero-capacity gating is correct: zero-capacity shared-ESS operational
variables are fixed to zero and their operational constraints are deactivated.

Secondary candidates (`sess_comp`, generator magnitude-square constraints,
branch apparent-flow magnitude-square constraints) remain deferred.

---

# 4. P3.5 diagnostic sequence — COMPLETE

## P3.5-A — remove `sess_snet_def`

Both prescribed frozen failures changed from persistent failure to clean
primary success when only `sess_snet_def` was deactivated.

However, the relaxed solutions materially violated the removed physical
equality at the worst period (~0.32 normalized by `s_rated^2`).

Conclusion:
- the row is numerically important;
- it is physically active and not redundant;
- deletion is unacceptable.

## P3.5-B — equivalent `kappa=100` replacement

Replacing the row by:

`100 * ((sch-sdch)^2 - pnet^2 - qnet^2) = 0`

made both frozen failures solve cleanly while preserving the original physical
relation to machine precision.

A replacement-component / warm-start-dual confound remained.

## P3.5-C — replacement dual-control test

TSO cleanly confirmed the scaling hypothesis.

DSO exposed strong sensitivity to replacement-component identity/order:
an algebraically identical replacement could solve while a scaled replacement
could fail.

This showed that creating a new Pyomo constraint component can perturb the
MA97/IPOPT path of an already fragile KKT system.

## P3.5-D — decisive in-place scaling

P3.5-D removed all identified confounds.

For both DSO `case33_2 / 2025 / Autumn / cycle 8` and TSO
`case9 / 2025 / Summer / cycle 6`:

- the literal original `sess_snet_def` `ConstraintData` objects were retained;
- no constraint component was created or deactivated;
- component/index ordering and counts were unchanged;
- each row expression was changed in place from `g == 0` to `100*g == 0`;
- each existing constraint multiplier was transformed on the same object:
  `lambda -> lambda/100`.

Results:

### DSO
- baseline: primary `internalSolverError`, recovery failed;
- in-place scaled: primary `optimal`, no recovery;
- dual infeasibility ~`2.06e-08`;
- original unscaled physical residual normalized by `s_rated^2`:
  ~`3.90e-12`.

### TSO
- baseline: primary failure, recovery exhausted 3000 iterations;
- in-place scaled: primary `optimal`, no recovery;
- dual infeasibility ~`1.98e-08`;
- original unscaled physical residual normalized by `s_rated^2`:
  ~`1.14e-12`.

## Accepted conclusion

The residual DSO/TSO failure family is convincingly diagnosed as a numerical
scaling/conditioning problem in network-side `sess_snet_def`.

This conclusion is free of the identified confounds:
- no physical relaxation;
- no replacement component;
- no component reordering;
- no lost warm-start constraint multipliers;
- no solver or ADMM retuning.

P3 diagnostics are complete.

---

# 5. Current workstream — P4

Current task:

**P4 — Production-safe shared-ESS apparent-power equality normalization**

Target relation:

`kappa_e * ((sch - sdch)^2 - pnet^2 - qnet^2) = 0`

with a fixed numerical scale derived from the installed local-SMOPF shared-ESS
power rating.

For the diagnosed case:

- `s_rated = 0.01 p.u.`
- `kappa = 1/s_rated = 100`.

Do not divide symbolically by an optimization variable inside the nonlinear
equation.

For zero-capacity shared ESSs, use a finite safe scale (for example 1.0) and
rely on the existing zero-capacity gating.

Before implementation, audit how installed capacity, model reuse, and imported
constraint duals evolve across planning candidates.

If the same live row changes from `kappa_old` to `kappa_new` while retaining a
warm-start multiplier, preserve the equivalent Lagrangian contribution using:

`lambda_new = lambda_old * (kappa_old / kappa_new)`.

Only implement such remapping if the lifecycle audit proves it is needed.

---

# 6. Standard ESS policy

The ordinary/standard ESS formulation has an analogous apparent-power/net-power
nonlinear structure and is therefore a plausible candidate for the same
normalization.

However, direct failure evidence currently exists only for shared ESS.

Policy:

1. implement and validate the shared-ESS normalization first;
2. do not mix the standard ESS change into the first production edit;
3. after shared-ESS validation, audit the standard ESS relation, rating source,
   zero-capacity behavior, model lifecycle, imported dual lifecycle, and test
   coverage;
4. if they are equivalent, apply the same normalization in a separate,
   controlled substage;
5. validate at least one active standard-ESS case if such a case exists.

This preserves one-family-at-a-time attribution while still aiming for
consistent scaling across mathematically equivalent ESS models.

---

# 7. Deferred battery-physics work

The current ESS/shared-ESS SOC formulation uses apparent charging/discharging
quantities where physical battery energy should ultimately be driven by active
or DC energy throughput.

That correction remains deferred until P4 is complete.

Do not combine:
- SOC refactoring;
- degradation refactoring;
- calendar ageing;
- `sess_comp` changes

with the current scaling work.

---

# 8. Current prohibitions

During P4 do not change:

- `sess_comp`;
- SOC equations;
- ESSO degradation/SoH;
- active/apparent throughput definitions;
- generator/branch equations;
- voltage formulation;
- solver settings;
- ADMM settings;
- common objective scaling;
- TSO proximal regularization;
- Benders/local-cut logic.

Do not run the full paper experiment matrix before P4 validation is complete.
