# Track D1 — three-cell campaign: SUSPENDED

**Neither comparison is available. Cell 1 is blocked by a production defect; cell 3 failed
at initialization. Cell 2 succeeded. Reporting and stopping, as the protocol requires —
suspended, not caveated.**

Frozen spec `data/SRP1/Results/P514D/frozen_d1_campaign_v1_2c8c7bef.json` (`2c8c7bef`),
frozen before any cell ran. Overrides applied as predeclared varied settings:
`budget = 5.0e6` (derived, and verified inert on the cold path), `rel = 1e-4`.

## Scoping, as required

This is **not the paper's reported plan**. Three MVA and twelve MWh system-wide across
three case33 networks is a substantial penetration. Nothing here validates the 2.01%
figure, and nothing here is presented as doing so.

## 1. Outcomes

| cell | outcome | solves |
|---|---|---|
| 1 — uncoordinated, no storage | **BLOCKED — production defect** | 0 (aborted in model build) |
| 2 — coordinated, no storage | **succeeded** | 3,621 |
| 3 — coordinated, 1.00 MVA / 4.00 MWh | **FAILED at initialization** | 51 |

### Cell 1 — the uncoordinated mode is broken in the current tree

`_add_dso_scenario_deviation_penalty` unconditionally references
`model.expected_shared_ess_p` (`shared_resources_planning.py:2826`). The ADMM path builds
that variable and its definition first (`:3013-3018`); the uncoordinated path builds only
`expected_interface_vmag`, `_pf_p` and `_pf_q` (`:5850-5859`) and then calls the same
penalty, producing

```
AttributeError: 'ConcreteModel' object has no attribute 'expected_shared_ess_p'
```

Introduced by `99a59fec` (2026-08-18), the commit that added the penalty to the
uncoordinated site without the variables it needs. The `no_coordination` results workbook
on disk predates it, so the mode worked once and has been broken for roughly three weeks
without being run.

**Not fixed.** The one-line shape is clear, but whether an uncoordinated DSO should carry a
shared-ESS deviation penalty at all is a formulation question, not a typo, and production
changes are not authorized here.

### Cell 2 — succeeded

| quantity | value |
|---|---|
| cycles | 70 (cap 90 not hit) |
| solves | 3,621 = `51 x 70 + 51`, identity exact, 0 blocked |
| recourse (= gross; no storage, so no salvage) | **820,746,762.46** |
| terminal step / threshold (**rule ten**) | 69,063 / 82,082 = **0.8414** |
| local solve failures | **0** |
| final `rho_pf` | 5.2025 |
| non-vanishing slacks (threshold 1e-6) | **none of 17 families** |

*Correction to a premise in the frozen spec.* It assumed cell 2 "has no shared ESS and
therefore no such terms". It does build all 17 slack families — including
`slack_es_ch_comp_per_unit`, `slack_es_soh_per_unit_*` and `slack_shared_es_soc_final_*` —
at zero capacity, where they are trivially zero rather than absent. The asymmetry logic
survives; the wording was wrong.

### Cell 3 — infeasible ESSO subproblem at initialization

The candidate passed first-stage feasibility (`feasible: true`, empty reason), so budget,
capacity and ratio are all satisfied. ADMM never started:

```
[WARNING] Shared ESS solver did not converge for ESS node=7: termination=infeasible,
          Ipopt 3.14.18: Converged to a locally infeasible point.
[WARNING] Operational initialization failed because at least one local problem did not
          solve successfully. ADMM will not be started.
```

IPOPT's exit state at node 7: 165 iterations ending in restoration, **constraint violation
6.84e-05**, dual infeasibility 1.0e3, complementarity 1.9e-08.

**Nodes 5 and 9 solved optimally at identical capacity.** So this is node-specific — one
of three — rather than a universal infeasibility of the plan.

**What is NOT established: whether this is genuine infeasibility or numerical failure.**
The distinction decides what the result means — genuine would be a finding about the plan,
numerical a finding about the tool — and the evidence is ambiguous. A constraint violation
of 6.84e-05 is *small*, of the same order as `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`, which
points toward a numerically hard rather than truly empty feasible set; but two of three
nodes solving does not settle it either way. Resolving it is further work and is not
authorized.

## 2. The transfer caveat — measured, and it did not transfer

This is the campaign's principal result.

| property | at negligible capacity (0.0106 MVA) | at 1.00 MVA / 4.00 MWh |
|---|---|---|
| local-solve reliability | 1 failure in 1,095 (and 1 in 4,284 across AB1/X22) | **1 failure in 3 ESSO solves**, run aborted |
| evaluation cost | 3,519 solves (68 cycles) at `rel` 1e-4 | **not measurable — the run did not start** |

**Neither of the two properties this project established transfers to material capacity.**
The reliability figure was measured where the storage is nearly inert; at real capacity the
very first ESSO solve failed.

### The recovery path is inapplicable by design

`_is_recoverable_shared_ess_failure` (`shared_energy_storage_data.py:934-940`) returns true
**only** for `TerminationCondition.internalSolverError`. Node 7 terminated as `infeasible`,
so the retry never fired — confirmed independently by the solve count, exactly
`51 = 36 + 12 + 3`, one attempt per block.

So the ESSO recovery mechanism covers solver *crashes*, and the failure mode that actually
occurs at real capacity is the one it does not cover.

## 3. A structural confound that would affect cell 3 even if it ran

`_shared_ess_admm_normalization_mva = max(|rating|, floor)` with
`shared_ess_normalization_floor_mva = 0.10` (`admm_parameters.py:17`). So the shared-ESS
consensus terms are normalized by:

| cell | rating | normalization used |
|---|---|---|
| cell 2 (zero storage) | 0.00 | **0.10 (floor)** |
| C1 cold (bootstrap) | 0.0106 | **0.10 (floor)** |
| cell 3 (the plan) | 1.00 | **1.00** |

**Cell 2 and cell 3 would not be solved under the same ADMM weighting** — the ESS consensus
channel differs by a factor of 10 between them. That is a structural difference in addition
to the capacity, and any cell3-minus-cell2 difference would confound the two. It should be
resolved before the comparison is attempted again.

## 4. An unexplained anomaly in the one pair that is available

Cell 2 and the C1 cold cell share initialization, tolerance, configuration and
normalization, differing only in candidate:

| | recourse | terminal step | cycles |
|---|---|---|---|
| cell 2, no storage | 820,746,762.5 | 69,063 | 70 |
| C1 cold, bootstrap 0.0106 MVA | 819,145,341.2 | 47,545 | 68 |
| **difference** | **1,601,421.3** | error bar 116,608 | **13.7x — determinate** |

The difference is resolvable under the ninth rule. **But it cannot plausibly be storage
value**: the bootstrap plan is about 10.6 kW and 21 kWh per node, costing roughly 7,046 per
node, and 1.6 million of operating saving from that is not credible.

So either the zero-capacity case differs structurally from any-capacity case in some way
beyond the normalization floor, or the two trajectories settled in different basins — the
path-dependence C1 established. **Recorded as an anomaly requiring explanation, not as a
measurement of storage value.** It also warns that "no storage" may not be a clean control.

## 5. Side effect recorded

Three IPOPT logs totalling ~43 MB were written to the repository root
(`optim_log_node_{5,7,9}.txt`) because the ESSO solver's default `output_file` is relative.
Untracked; they should be added to `.gitignore` rather than committed.

## 6. What would be needed to complete the campaign

1. A decision on the uncoordinated path's shared-ESS deviation penalty (cell 1).
2. A diagnosis of node 7's infeasibility, genuine or numerical (cell 3).
3. A resolution of the normalization-floor confound between zero and non-zero capacity.
4. An explanation of the 1.6M anomaly in §4.

**None is authorized. Reporting and stopping.**
