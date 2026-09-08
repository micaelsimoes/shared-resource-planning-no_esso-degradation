# P5.6 — Nonlinear derivative-free planning

Branch `feature/derivative-free-planning`, created from the accepted P5.5-D HEAD
(`24350e8d`) with all P5.5 evidence and history preserved.

Canonical runtime throughout: `/opt/anaconda3/envs/opf_env_py311/bin/python`;
canonical SRP1 checksum
`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`, verified by
the oracle itself on load and by the R0 provenance gate in every harness.
Evidence under `data/SRP1/Results/P56A/`; P5.4 and P5.5 evidence untouched.

---

# P5.6-A — deterministic nonlinear recourse oracle

This stage builds and validates the candidate-evaluation oracle. It implements
no search, no cut, no master problem and no MISOCP. The accepted nonlinear
operational formulation (P5.4-D2-P / H1) is used exactly as production defines
it: every model is built by the production constructors and solved by the
production solvers, and no optimization mathematics was modified.

## A0 — corrections to the P5.5-D record

Applied to `P5_5_CONVEX_PLANNING_ARCHITECTURE_REPORT.md` before any new work.

**A0.1** `P5.5-D-C — practical rigorous lower-bound architecture is unavailable`
stands unchanged.

**A0.2** The fixed-mode objective is no longer called a lower bound. Fixing the
operating mode *restricts* the feasible set, so

```
Q_MISOCP*  <=  Q_fixed_mode        hence        UB - Q_MISOCP*  >=  UB - Q_fixed_mode
```

The screen delivers an **upper** bound on the MISOCP optimum and therefore a
**lower bound on the unavoidable MISOCP-to-UB gap** — which is what makes the
rejection decisive — but neither `Q_fixed_mode` nor
`Q_fixed_mode − V_salvage_max` is "the MISOCP lower bound". Every gap quoted in
D7 is a minimum; the true gap can only be larger.

**A0.3** The D1 value `729361086.642965` is renamed **"polished feasible-network
candidate, pending complete nonlinear ESSO/coupled feasibility certification"**.
D1 established feasibility of the 48 *network* SMOPFs only; it never audited the
original nonlinear ESSO. A1 below performs that certification.

**A0.4** The attribution of the ~13 % difference to the augmented ADMM objective
is **withdrawn as unproved** and recorded as a mechanism to be attributed in A2.
A2 then shows the original attribution was wrong twice over — see below.

**A0.5** D8 now reads "the TSO is the dominant remaining relaxation defect"
rather than asserting the whole gap is caused exclusively by the TSO. The
numerical per-agent decomposition is unchanged.

## A1 — complete feasibility certificate

The oracle follows A1's prescribed consistency sequence exactly, and reuses
production's own constructors rather than reimplementing the ESSO:

1. the canonical converged ADMM state;
2. the ADMM's explicit consensus **z** wherever one exists — the shared-ESS
   family has `consensus_vars['ess']['z']` and it is taken directly; the
   interface power-flow and voltage families have only TSO and DSO copies and no
   z, so a common value must be constructed and the midpoint is used, with the
   convention recorded per entry;
3. `create_shared_energy_storage_model` — production's own constructor — solves
   the **ORIGINAL nonlinear ESSO** with the investment and the common
   coordinated P/Q fixed;
4. `S_available` / `E_available` taken from that physical solution;
5. those exact capacities pushed into both the TSO and the DSO copies;
6. common interface P/Q, voltage and shared-ESS P/Q fixed;
7. all 48 nonlinear network SMOPFs re-solved;
8. mutual consistency verified.

**Result on the canonical base candidate — status VALID.**

| coordinated residual | value | target |
|---|---|---|
| interface P | 5.551e-17 | 1e-7 |
| interface Q | 2.776e-17 | 1e-7 |
| interface voltage | 0.000e+00 | 1e-7 |
| shared-ESS P, TSO vs DSO | 0.000e+00 | 1e-7 |
| shared-ESS Q, TSO vs DSO | 0.000e+00 | 1e-7 |
| shared-ESS P, networks vs **ESSO** | 0.000e+00 | 1e-7 |
| shared-ESS Q, networks vs **ESSO** | 0.000e+00 | 1e-7 |
| `S_available` | 0.000e+00 | 1e-7 |
| `E_available` | 3.469e-18 | 1e-7 |
| **max coordinated** | **5.551e-17** | **1e-7** |

| physical-feasibility audit | value |
|---|---|
| ORIGINAL nonlinear ESSO, max constraint violation | **2.035e-13** (worst family `energy_storage_capacity_degradation`) |
| production `get_feasibility_violation` on the ESSO | −3.240e-05 (tolerance 1e-03) → feasible |
| network H1 complementarity | **0.000e+00** |
| network converter capability | 1.628e-08 |
| network constraint residual | **1.093e-05** at `DSO5\|2030\|Winter`, family `node_balance_p` |

The ESSO audit scans every constraint family of the original formulation — the
cohort rated S/E rows, the availability rows including
`E_available = E_rated · soh_cumul`, throughput, degradation, annual and
cumulative SoH, minimum SoH, lifetime gating, cohort complementarity, the
aggregate P/Q capability and consensus rows and every ESSO slack — and reports
the worst per node as well as production's own aggregate measure.

> **Model-specific feasibility tolerance, reported separately as required.** The
> 1.093e-05 p.u. network residual is not a coordination residual: it is IPOPT's
> own constraint tolerance on the nonlinear SMOPF (1.09e-03 MW on a 100 MVA
> base, about 1 kW). It sits far above the 1e-7 coordinated target and far below
> anything physically meaningful, and it is a property of the nonlinear solver,
> not of the coupling.

**Exact objective accounting**

```
gross operational cost        829291677.522120
actual physical salvage            3439.659877
net operational recourse      829288237.862242
investment cost                   50000.000000
TOTAL PLANNING OBJECTIVE      829338237.862242
```

The complete original nonlinear system passes, so the point may be renamed

```
RIGOROUS FEASIBLE NONLINEAR UB INCUMBENT   =   829338237.862242  (total planning objective)
                                                829288237.862242  (net operational recourse)
```

Note this is **not** D1's `729361086.642965`. Two things changed: the capacities
now come from the physical ESSO rather than from the ADMM's own ESSO, and — far
more importantly — the polish itself was corrected. A2 explains why.

## A2 — the ~13 % explained, and largely withdrawn

### The original figure was 95.5 % an artefact of my own polish

Production's coordinated transmission model computes the interface power as
`pc_adn = pc + flex_p_up − flex_p_down`, **fixes** `pc` at the DSO's consensus
value (`shared_resources_planning.py:2905-2918`, via `fix_or_set`, which fixes),
and charges `flexibility_cost = c_flex · baseMVA · (flex_p_down + flex_q_down)`
for downward deviation. The transmission system pays to move away from the
consensus.

The first polish fixed `pc` at the *achieved* interface power and pinned the
interface flexibility to zero. That reproduces the same physical interface power
at **zero flexibility cost** — not a cheaper plan, only a different accounting
of the same one. Attributed by family, that single choice was:

| family | delta | share |
|---|---|---|
| **flexibility_cost** | **−104 228 155.90** | **95.50 %** |
| slack_penalties | −4 523 294.27 | 4.15 % |
| generation_cost | −367 282.79 | 0.34 % |
| others | −17 075.08 | 0.02 % |
| **total** | **−109 135 808.04** | −13.02 % |

Evidence retained at
`data/SRP1/Results/P56A/p56a_a2_attribution_ORACLE_p56a1_artifact_diagnosis.json`.

### The corrected polish, and the honest number

The polish now leaves `pc` exactly where production put it, leaves the interface
flexibility free within production's bounds, and pins only the **total**
interface power:

```
pc + flex_p_up - flex_p_down  ==  common_p          (and the reactive analogue)
```

so the flexibility cost is incurred wherever production incurs it. The honest
improvement is **−9 208 592.85, or −1.0982 %**:

| family | delta | share |
|---|---|---|
| slack_penalties | −4 523 294.27 | 49.12 % |
| flexibility_cost | −4 300 940.70 | 46.71 % |
| generation_cost | −367 282.79 | 3.99 % |
| ess_complementarity_penalties | −9 941.61 | 0.11 % |
| gen_curtailment_penalty | −7 133.47 | 0.08 % |
| load_curtailment_cost | 0.00 | — |
| ess_utilization_cost_penalty | 0.00 | — |

By agent: DSO7 −3 001 747.26, DSO5 −2 910 980.86, DSO9 −2 672 061.04,
TSO −623 803.68. The ten largest negative block deltas are all distribution
Summer/Autumn blocks, dominated by flexibility cost.

**Do the previously problematic TSO Spring blocks dominate? No.** Under the
corrected polish they account for **0.43 %** of the change. Under the artefact
they accounted for 52.45 % — which is precisely why the question had to be
re-asked after the correction rather than inherited from the first run.

### A2.1 — base versus augmented objective, at fixed consensus

One representative affected block, same fixed consensus values, same initial
point, solved twice.

| | `DSO5\|2035\|Autumn` | `TSO\|2025\|Spring` |
|---|---|---|
| base objective, A (augmented objective active) | 2554.812449 | 172745.031508 |
| base objective, B (base objective active) | 1850.740074 | 172639.378636 |
| A − B | **704.07** | **105.65** |
| augmented objective at A | 0.020741007 | 1.835012976 |
| augmented objective at B | 0.014672686 | 1.833890749 |
| d(augmented)/d(base), A→B | 8.62e-06 | 1.06e-05 |

Once every coordinated variable is fixed, the consensus penalties, the proximal
term and the single-scenario deviation penalty are all **constants**, so the
augmented objective reduces to `base / scale + constant` with a scale of order
1.1e5. The measured A→B ratio, 8.62e-06, is `1/116 000` — consistent with that.
The two problems therefore have the **same optimizer set**.

They nevertheless return solutions 704 apart on the distribution block. Per
A2.1's own instruction that is classified as **local NLP branch / solver-path
selection, not an economic effect of the ADMM objective.**

There is a quantitative reason it is easy for the solver to stop short: at a
scale of 1.1e5 the base objective contributes ~0.02 to an augmented objective of
~0.02, so an ordinary solver tolerance on the augmented objective corresponds to
a base-objective slack five orders of magnitude larger. The ADMM's local
solutions are not base-objective-optimal, and the polish — which minimises the
base objective directly — finds the remaining savings.

### A2.2 — branch fingerprint

| quantity | A (augmented) | B (base) | delta |
|---|---|---|---|
| base objective | 2554.812449 | 1850.740074 | −704.07 |
| flexibility cost | 2223.597855 | 1849.483999 | −374.11 |
| slack penalties | 328.013255 | −0.754036 | −328.77 |
| Σ\|pg\| | 16.912450 | 16.921685 | **+0.009** |
| Σ\|qg\| | 2.732858 | 2.799582 | +0.067 |
| Σ\|slack_v_sqr_up\| | 0.003280 | 0.000008 | −0.0033 |
| Σ\|shared_es_pch\| | 0.000805 | 0.000808 | +0.000003 |

Dispatch is essentially unchanged — `Σ|pg|` moves by 0.009 out of 16.9 — while
the entire difference sits in slack penalties and flexibility. **The polish is
not recovering the P5.4-D4 lower branch**, which was a dispatch-level branch
worth ~1.9e6 in recourse. It is a different mechanism: the same dispatch branch
with the slack and flexibility usage cleaned up.

## A3 — purity

`fresh_planning()` gives every evaluation its own deep copy of the baseline
(0.03 s, against 10.8 s to re-read the case), its own solver-log directory —
production configures IPOPT with `file_append='yes'`, so a shared path
interleaves runs — and its own `solver_recovery_diagnostics`. The baseline is
never mutated, so the A12 in-place mutation of
`network[year][day].shared_energy_storages[i]` and of the ESSO storage objects
can no longer leak between candidates.

**Tolerance declared before the run:** `|dQ| <= 1.0` absolute, which is
1.4e-09 of the total objective and **1.4e-06 × `tol_cut`** — materially below the
investment signal, as required. Bit-identical agreement was reported separately
and was not the pass criterion. Caching was disabled, since a cache hit would
return the first result and hide the defect under test.

| sequence | candidate | status | total objective |
|---|---|---|---|
| 1st | x_A | VALID | **829338237.8622425** |
| 2nd | x_B (`s\|node5\|2025\|-10%`) | VALID | 829325610.1726934 |
| 3rd | x_A | VALID | **829338237.8622425** |
| 4th | x_C (`e\|node9\|2025\|-10%`) | INVALID_INVESTMENT | — |
| 5th | x_A | VALID | **829338237.8622425** |

```
objective difference vs the first evaluation :  A2  0.000000e+00      A3  0.000000e+00
bit-identical : True          PASS : True
```

A separate purity check covers the START-2 template, which is shared across
evaluations: re-evaluating the first candidate after every other candidate had
consumed the template returned `828021090.3608505` against `828021090.3608505` —
**delta exactly 0.0**.

> **Incidental finding.** The canonical base candidate sits *exactly* at the
> minimum energy-to-power ratio of 2.0, so any E-only reduction is first-stage
> infeasible. P5.4-R/D3's `e|node9|2025|-x%` perturbation candidates violate
> `min_energy_to_power_ratio` and the oracle rejects them as
> `INVALID_INVESTMENT` before running anything. They were used for sensitivity
> work without a first-stage check. Retained here as a deliberate negative test.

## A4 — the candidate-evaluation contract

`p56a_oracle.evaluate_planning_candidate(x, start_policy, ...)`.

**Input** — the 18 investment variables (S and E for nodes 5/7/9 and years
2025/2030/2035), as a `(node, year) -> {'s','e'}` mapping or a flat sequence; the
canonical scenario and configuration; a declared initialization policy.

**Master feasibility is checked before any operational model runs**, reusing
production's own `_check_candidate_first_stage_feasibility` (nonnegativity,
minimum and maximum E/S ratio, maximum energy capacity, investment budget) on
totals rebuilt by production's own `_rebuild_candidate_total_capacities`, which
is the cohort/calendar-life mapping.

**Pipeline** — clean candidate data; apply investment; nonlinear operational
ADMM; common coordinated values; original nonlinear ESSO; exact-consensus
polish; audit ESSO and all networks; physical salvage; `total_objective =
investment_cost + polished_net_operational_recourse`.

**Status is always explicit**, and a failed evaluation never returns a
fabricated objective:

`VALID`, `INVALID_INVESTMENT`, `LOCAL_SOLVE_FAILURE`, `POLISH_FAILURE`,
`ESSO_SOLVE_FAILURE`, `ESSO_FEASIBILITY_FAILURE`, `COUPLING_FAILURE`,
`SOLVER_CRASH`.

`SOLVER_CRASH` was added during A6 after an IPOPT process failure propagated a
`pyomo.common.errors.ApplicationError` out of the contract. A search must be able
to continue past a crashed solver, so the ADMM, ESSO and polish stages are now
each wrapped and report the failing stage.

## A5 — initialization policy

Both policies are fixed by configuration and neither consults the history of
previously evaluated candidates. A "nearest previously evaluated candidate" warm
start was explicitly not implemented, because it would make the returned
objective depend on search order — exactly what P5.4-D3/D4 showed can move the
recourse by more than the investment signal.

- **START-1** — canonical cold initialization.
- **START-2** — one fixed archived branch template: the cold solution of the
  **canonical base candidate**, built once per process and remapped to each
  candidate's capacities by production's own `initial_state` warm-start path.

`Q_oracle(x)` is the minimum total objective among all VALID polished solutions
returned by the fixed start set.

**START-2 is both better and cheaper**, on every candidate where both succeeded:

| candidate | START-1 cold | START-2 template | benefit | cold cycles | template cycles |
|---|---|---|---|---|---|
| base | 829 338 237.86 (562.0 s) | **828 021 090.36** (603.2 s ¹) | 1 317 147.50 | 17 | **2** |
| `s\|node5\|2025\|-10%` | 829 325 610.17 (555.2 s) | **828 023 009.86** (101.7 s) | 1 302 600.31 | 17 | **2** |
| `s\|node9\|2025\|-5%` | 828 663 900.91 (580.3 s) | **828 020 986.17** (99.6 s) | 642 914.75 | 18 | **2** |

¹ includes the one-off template build; subsequent START-2 evaluations cost ~100 s.

The second start improved **3 of 3** candidates, by up to **1 317 147.50 = 1.84 ×
`tol_cut`** — a material benefit, not a marginal one — while converging in 2 ADMM
cycles instead of 17–18, i.e. 144 nonlinear local solves instead of 864. It does
not double runtime; after the template is amortised it *divides* runtime by about
5.5.

The start also changes the shape of the landscape. Across the three candidates
the cold objectives span **674 336.95** (0.94 × `tol_cut`) while the
template-started objectives span **2 023.69** (0.0028 × `tol_cut`). For a
derivative-free search that difference matters more than the levels do: the cold
oracle's start-dependence is comparable to the signal being searched for, and the
template-started oracle's is not.

## A6 — reproducibility and cost benchmark

No search was launched. The population is four candidates: the canonical base
plus three P5.4-R/D3 perturbations, all pre-existing rather than invented (the
third had to be changed from an E-only to a joint S+E perturbation because of the
E/S-ratio finding above).

**Cost of one trustworthy nonlinear planning evaluation**

| | value |
|---|---|
| valid evaluations measured | 6 |
| median | **558.6 s** |
| p95 | **603.2 s** |
| range | 99.6 – 603.2 s |
| mean | 417.0 s |
| cold evaluation | ~555 – 580 s |
| template-started evaluation | **~100 s** |
| nonlinear local solves, cold | 864 (17 ADMM cycles × 48, plus 48 polish) |
| nonlinear local solves, template | 144 (2 cycles × 48, plus 48 polish) |
| ESSO solves per evaluation | 3 |
| polish solves per evaluation | 48 |
| ADMM recovery solves observed | 0 |

**Benefit of polishing** — the ADMM recourse minus the polished recourse: median
**9 203 528.96**, range 9 091 171.73 … 9 217 422.35. That is 12.8 × `tol_cut`, so
polishing is not optional bookkeeping; without it the oracle would report a
materially worse point than the model actually admits.

**Failure rate: 1 of 4 candidates.** `s+e|node9|2025|-10%` returned
`POLISH_FAILURE` under **both** start policies. Diagnosed: the ADMM itself
converged (9 cycles, no recovery solves), but exactly **1 of the 48** network
solves failed — `DSO9|2030|Spring`. That candidate's ADMM interface residual is
larger than the base candidate's (1.967e-02 against 1.341e-02 p.u.), so the
midpoint asks that distribution block to move further than it can.

**Remedy tested.** Anchoring the common interface on the DSO's own achieved value
instead of the midpoint asks only the transmission side to move, and the
transmission side has interface flexibility for exactly that purpose. On the
failing candidate this **restores validity**: `POLISH_FAILURE` → `VALID` with
total objective 838 978 240.79.

**And it is nearly free.** A controlled comparison on the base candidate, same
start and same everything else:

| interface anchor | status | total objective | wall clock |
|---|---|---|---|
| midpoint (default) | VALID | 829 338 237.8622425 | 515.7 s |
| DSO-anchored | VALID | 829 351 291.8352123 | 515.4 s |

The DSO anchor costs **13 053.97**, which is **0.018 × `tol_cut`** — two orders
below the investment signal — and costs nothing in runtime. On this evidence it
is the better default, but it has been tested on two candidates only and is left
as a configuration option rather than switched blind; validating it across a
wider population is the obvious next step and is not something P5.6-A was
authorised to conclude on its own.

The midpoint control also reproduced `829338237.8622425` exactly, in a fresh
process — a fourth bit-identical reproduction of the base candidate on top of the
three in A3.

## A7 — caching

Deterministic evaluation caching keyed on the exact investment vector, the
canonical scenario checksum, a configuration hash (objective type, flexibility /
curtailment / storage flags, shared-ESS model, ADMM tolerances and iteration
cap, ESSO budget, capacity and ratio limits), the oracle version and the
initialization-policy id — plus the interface-anchor convention added in A6.

Only VALID completed evaluations are cached; failed or incomplete runs are never
stored as objective values. The cache changes no mathematics; it only avoids
recomputing an identical candidate.

**Demonstrated cache hit** on the base benchmark candidate:

```
cache_hit = True     wall clock 0.040 s     Q = 829338237.8622425
cache holds 6 VALID entries
```

against ~560 s for the computed evaluation — a factor of ~14 000, returning the
identical objective.

## A8 — convex / MISOCP status

The P5.5 convex and disjunctive code is retained in full and **classified as
diagnostic / research benchmark only**. It is not imported or invoked anywhere in
`p56a_oracle.py` or any P5.6-A harness. Nothing from it has been merged into the
nonlinear production path.

Not pursued during P5.6-A, as instructed: full MISOCP, QCP dual recovery, Benders
cuts, and TSO SDP/QC strengthening. The TSO-only stronger-relaxation idea is
recorded as **optional future benchmark work**, not the active planning path.

## A9 — no derivative-free search

None implemented: no coordinate search, pattern search, MADS, Bayesian
optimization, surrogate optimization, evolutionary algorithm or random search.
The search method should be selected against the measured evaluation cost,
repeatability and failure rate reported above.

## What the numbers imply for a search budget

At the template-started cost of ~100 s per evaluation plus a ~560 s one-off
template, a 200-evaluation search is about 5.7 hours and a 1000-evaluation search
about 28 hours, single-threaded. At the cold cost of ~560 s the same searches are
31 hours and 6.5 days. The template start is therefore not a convenience; it is
what makes a search of useful size practical at all. Candidates are independent,
so the wall clock divides by whatever parallelism the planner is willing to
spend.

Against that, two facts have to be weighed: the oracle failed to return a value
on 1 of 4 benchmark candidates under the default convention, and the choice of
interface anchor moves the returned objective by far more than `tol_cut`.

## Verdict

**What is resolved.**

- *Feasibility.* The complete original nonlinear coupled system — the ESSO in its
  original nonlinear formulation, all 48 network SMOPFs, and every coordinated
  quantity across the TSO, DSO and ESSO copies — is certified on the canonical
  base candidate, with a maximum coordinated residual of 5.551e-17 against a
  1e-7 target and an ESSO constraint violation of 2.035e-13. The only residual
  above target is IPOPT's own 1.093e-05 constraint tolerance on the nonlinear
  SMOPF, reported separately as A1 requires.
- *Purity.* Bit-identical across three different call histories in A3, and again
  in two further independent processes. The A12 mutable-state defect is closed by
  per-evaluation deep copies, isolated solver-log directories and reset
  accumulators, with no change to optimization mathematics.
- *Contract.* One API, master feasibility checked before any operational model
  runs, explicit statuses, no fabricated objectives, deterministic caching with a
  demonstrated 0.040 s hit.
- *Cost.* Measured, not estimated: ~560 s cold and ~100 s template-started per
  evaluation, 864 versus 144 nonlinear local solves.

**What is not.**

- *A failure rate that a search would meet.* One of four benchmark candidates
  returned no VALID objective under the default interface convention, from a
  single distribution block that could not reach the midpoint. The DSO-anchored
  remedy fixes that candidate and costs 0.018 × `tol_cut` on the control, but it
  has been tested on two candidates and is not yet the default. Until the failure
  rate is characterised on a wider population, a search cannot be told how often
  it will be handed nothing.
- *Start dependence larger than the signal.* `Q_oracle` over a fixed start set is
  deterministic, but its *level* depends on that set by up to 1.84 × `tol_cut`,
  and the spread across candidates differs by a factor of 300 between the two
  starts (674 337 cold against 2 024 template-started). Which start set to adopt
  is a planner decision with consequences larger than the investment signal.
- *A2.1's residue.* Two mathematically equivalent problems return local solutions
  704 apart on one distribution block. The oracle is deterministic about which one
  it takes, but the underlying local-solution sensitivity has not been removed,
  only pinned down.

Feasibility and purity are settled; failure rate and start/branch stability are
not. That is the definition of the middle verdict.

```
P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved
```

```
P5.6-A COMPLETE — ready for planner review before selecting the derivative-free search method
```
