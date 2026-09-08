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

> **Superseded by P5.6-B0.1.** This is the certificate for the **START-1 cold**
> evaluation. A5's START-2 evaluation of the same candidate returned a better
> value, and P5.6-B0.1 re-audited the persisted evidence to confirm it passed the
> *identical* complete pipeline — master feasibility, original nonlinear ESSO
> (solved at all three nodes, max violation 1.050e-13, production feasibility
> −3.240e-05 against the 1e-03 tolerance), exact TSO/DSO/ESSO P/Q consistency
> (0.000e+00), exact S/E availability consistency (0.000e+00 and 3.469e-18), the
> 48-block network audit (max 1.500e-05, H1 0.000e+00, converter capability
> 9.999e-09) and physical salvage taken from the physical ESSO. It is therefore
> promoted:
>
> ```
> BEST RIGOROUS FEASIBLE NONLINEAR PLANNING INCUMBENT
>     total planning objective   828021090.360850
>     net operational recourse   827971090.360850
>     gross operational cost     827974518.717105
>     physical salvage                3428.356255
>     investment cost                50000.000000
> ```
>
> for the canonical base investment candidate. The cold certificate above stands
> as a valid but weaker feasible point.

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

> **Correction of record (P5.6-B0.2).** `Q_oracle(x) = min(START-1, START-2)`
> requires **both** starts, so its recurring cost is approximately their **sum**,
> about **660 s**, not ~100 s. The ~100 s figure is the cost of a
> **START-2-only** recurring oracle after the one-off template construction has
> been amortised, and it is only available once a start policy has been locked —
> which is P5.6-B3's job, not A5's. Two quantities must be kept apart from here
> on:
>
> - **SEARCH EVALUATION COST** — the recurring per-candidate cost of whatever
>   single policy the search actually uses;
> - **FINAL INCUMBENT CERTIFICATION COST** — the one-off cost of certifying a
>   chosen incumbent, where evaluating several starts and both anchors and
>   keeping the best VALID polished solution is affordable and desirable.

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
interface anchor moves the returned objective.

> **Correction of record (P5.6-B0.3).** "Far more than `tol_cut`" is wrong. The
> measured base-candidate change from midpoint to DSO anchoring is
> **13 053.97 = 0.018 × `tol_cut`** — two orders *below* it. What makes it worth
> attention is a different comparison: it is about **six times** the ~2 024
> objective span observed across the three START-2 benchmark candidates, so an
> anchor switch could reorder candidates that the search is trying to
> distinguish. That is a resolution question, addressed in P5.6-B2 and B4, not a
> `tol_cut` question.

> **Correction of record (P5.6-B0.4) — `tol_cut` is retired as a resolution
> criterion.** Every ratio quoted against `tol_cut` in this P5.6-A section is
> retained only as *historical context*: `tol_cut = 7.164e5` was the acceptance
> threshold of the Benders cut machinery, and that architecture was retired at
> P5.5-D-C. It is not the tolerance a derivative-free search should use, and
> nothing in P5.6 should be accepted or rejected against it. A search-specific
> tolerance `tau_search` is derived from measured oracle behaviour in P5.6-B4.

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
  deterministic, but its *level* depends on that set by up to 1 317 147.50,
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

---

# P5.6-B — oracle policy stabilization and search-readiness

Branch `feature/derivative-free-planning`. Canonical runtime and checksum gate as
above, enforced by the oracle on load and by the R0 provenance gate in every
harness. Evidence under `data/SRP1/Results/P56B/`; P5.6-A evidence untouched.

Nothing in the accepted nonlinear SMOPF, ADMM, H1, IPOPT settings, the
convex/MISOCP diagnostics or the Benders/master-cut code was modified. `p56b_policy.py`
composes the P5.6-A building blocks unchanged and adds only two things they
lacked: an explicit template argument, so templates can be **chained**; and a
split between the operational stage and the polish, so one ADMM result can be
polished under two anchors from `_clone_operational_models` copies. No search was
launched.

The B0 corrections are applied in place in the P5.6-A section above.

## B1 — the template does not stabilize

START-2 archives the canonical base candidate's **cold** solution, yet START-2
then finds a materially better solution for that same candidate. So the template
is not a fixed point of its own procedure, and B1 chains the refinement to find
out whether it settles. Declared before evaluating: **stabilization tolerance
100 monetary units**, at most 5 refinements, every refinement from a fresh
candidate-specific deep copy, same candidate, same anchor policy, same physical
ESSO/polish pipeline, no dependence on any non-base candidate.

| refinement | template | total objective | delta | within 100 |
|---|---|---|---|---|
| 0 | T0 (archived cold base) | 828 021 090.360850 | — | — |
| 1 | T1 | 827 415 318.563944 | −605 771.80 | no |
| 2 | T2 | 826 824 028.845478 | −591 289.72 | no |
| 3 | T3 | 826 405 022.193437 | −419 006.65 | no |
| 4 | T4 | 825 961 521.882321 | −443 500.31 | no |

**Not stabilized.** The chain improves monotonically, drifts **−2 059 568.48**
over four refinements, and the last step is *larger* than the one before it, so
it is not even reliably decelerating. Per B1's own instruction the result is
reported as unstabilized rather than adopting whichever refinement happened to be
last.

Two consequences, and the second matters more than the first:

- **T_STAR cannot be frozen by stabilization.** It is frozen by **rule** instead:
  `T_STAR := T0`, the cold solution of the canonical base candidate — exactly what
  P5.6-A used, reproducible from the rule alone, id
  `a81f7f5191dd42dbf50d1726149b8909`, config hash and canonical checksum recorded
  with it. It is a **declared** template, not a converged fixed point, and the
  objective level it produces is demonstrably not the best available.
- **The template must be frozen for the whole search.** Candidates evaluated
  against different template generations are not comparable: one refinement moves
  the base objective by ~4–6e5, which is four orders above the smallest
  meaningful difference between candidates measured in B2 (33.27). Refining the
  template mid-search would silently reorder everything evaluated before it.

Purity was re-run against T_STAR under the locked pipeline: evaluating the base,
then another candidate, then the base again returned `828021090.3608505` both
times — **delta exactly 0.0**, bit-identical, matching the P5.6-A6 value from a
different process.

## B2 — interface anchor

Twelve master-feasible candidates, none random: the canonical base, S-only and
fixed-ratio S/E moves both negative and positive across nodes 5/7/9 and years
2025/2030/2035, two duration increases at fixed power, a global move, and one
candidate at the first-stage budget boundary (the base uses 5% of the budget, so
scaling every investment by 19 uses 95% — built from the existing planning rules,
not a new one). Master-infeasible candidates are never evaluated operationally.
For each candidate the ADMM was solved **once** with T_STAR and the polish run
**twice** from clones, so the comparison isolates the polish convention.

| candidate | midpoint | DSO | Q(DSO) − Q(mid) |
|---|---|---|---|
| base | VALID | VALID | +1 482.45 |
| `s\|node5\|2025\|-10%` | VALID | VALID | −702.16 |
| `se\|node5\|2025\|-10%` | **POLISH_FAILURE** | **POLISH_FAILURE** | — |
| `se\|node7\|2030\|-10%` | VALID | VALID | −771.08 |
| `se\|node9\|2025\|-10%` | **POLISH_FAILURE** | **POLISH_FAILURE** | — |
| `se\|node5\|2030\|+10%` | VALID | VALID | −705.00 |
| `se\|node7\|2035\|+10%` | VALID | VALID | +1 487.34 |
| `se\|node9\|2025\|+10%` | VALID | VALID | +11 175.98 |
| `e\|node5\|2025\|+25%` | VALID | VALID | −13 365.80 |
| `e\|node9\|2030\|+50%` | VALID | VALID | +5 607.92 |
| `se\|ALL\|-10%` | **POLISH_FAILURE** | **POLISH_FAILURE** | — |
| `se\|ALL\|x19` (boundary) | **SOLVER_CRASH** | — | — |

```
success rate, midpoint : 8 / 11 operationally evaluated  = 72.7 %
success rate, DSO      : 8 / 11 operationally evaluated  = 72.7 %
overall VALID          : 8 / 12 candidates               = 66.7 %
anchor delta   mean +526.21   std 6 507.45   range -13 365.80 .. +11 175.98
Spearman rank correlation between the two anchors : 0.9048
```

**The DSO anchor rescues nothing here.** All three polish failures fail under
*both* anchors, and the failing blocks are the same physics either way
(`se|node5|2025|-10%`: TSO 2025 Winter and DSO5 2025 Autumn; `se|node9|2025|-10%`:
TSO 2025 Autumn and DSO9 2025 Autumn; `se|ALL|-10%`: **33 of the 48 blocks**).

> **This contradicts P5.6-A and supersedes it.** In A6 the DSO anchor turned
> `se|node9|2025|-10%` from POLISH_FAILURE into VALID. That evaluation used a
> **cold** start. Under T_STAR the same candidate fails under both anchors. The
> A-stage rescue was therefore a property of the *start*, not of the anchor, and
> the "midpoint-first with DSO fallback" policy the A report recommended
> investigating buys nothing at T_STAR while costing a second polish (~35 s)
> every time it fires.

**Ranking is not preserved.** Five of the eight jointly valid candidates change
rank between anchors, and the worst pairwise reversal is
`s|node5|2025|-10%` against `se|node7|2035|+10%`: the midpoint ranks them
1 790.84 apart one way, the DSO anchor 398.67 apart the *other* way. Set against
the spread of the valid population (46 146.04 under midpoint) and, more sharply,
against the **smallest adjacent gap between candidates, 33.27**, the maximum
anchor-induced variation of 13 365.80 is **402 times** the resolution the search
would need. Mixing conventions within one search would therefore destroy the
ordering the search exists to discover.

**Locked recurring search-anchor policy: MIDPOINT-ONLY.** Both anchors have
identical success rates, so nothing is bought by the fallback; midpoint is the
convention under which the certified incumbent was computed, and a single fixed
convention removes the mixed-convention inconsistency entirely — which is what
makes `tau_search` meaningful in B4. DSO-always is equally defensible on success
rate alone and remains available as a configuration; what is *not* defensible is
mixing them. For **final incumbent certification** both anchors may still be run
and the best VALID polished solution retained.

## B3 — recurring start policy

T_STAR was evaluated on the whole population. Cold was evaluated only on the
strategic subset B3 prescribes: the base, the lowest and highest T_STAR
objectives, the boundary candidate, and one candidate that failed under T_STAR.

| candidate | cold | Q(cold) | Q(T_STAR) | cold − T_STAR |
|---|---|---|---|---|
| base | VALID | 829 338 237.86 | 828 021 090.36 | **+1 317 147.50** |
| `e\|node5\|2025\|+25%` | VALID | 829 326 565.42 | 827 989 614.23 | **+1 336 951.19** |
| `se\|node9\|2025\|+10%` | VALID | 838 922 190.37 | 828 035 760.28 | **+10 886 430.09** |
| `se\|ALL\|x19` (boundary) | SOLVER_CRASH | — | — (crash) | — |
| `se\|node5\|2025\|-10%` | **VALID** 838 974 140.19 | | **POLISH_FAILURE** | — |

**Cold is never better on objective**, on any point where both succeeded — it is
worse by 1.32e6 to 1.09e7. On the decision rule of B3 that authorizes

```
SEARCH START POLICY = T_STAR ONLY
Q_search(x) = the deterministic T_STAR result under the midpoint anchor
```

with cold demoted to a final-certification / periodic-audit start rather than a
recurring search cost.

**But the two starts have different failure sets, and that is not a detail.**
`se|node5|2025|-10%` is VALID under cold and POLISH_FAILURE under T_STAR; the
boundary candidate crashes under both. So a T_STAR-only search will be handed
nothing for candidates a cold start could have evaluated. That is tolerable for a
pattern search, which treats a failed poll point as unsuccessful, but it means the
recurring policy's ~27% failure rate is a property to design around rather than a
defect to be fixed by choosing the other start.

## B4 — search objective resolution, `tau_search`

`tol_cut` is retired (B0.4). `tau_search` is derived from the measured behaviour
of the **final proposed recurring policy** — T_STAR only, midpoint anchor only —
and the two kinds of variation are kept apart, as B4 requires.

**NUMERICAL REPEATABILITY** — what the same candidate under the same policy
returns twice:

| source | measured |
|---|---|
| repeat evaluation, identical candidate and policy (P5.6-A3) | **0.000000e+00**, three call histories |
| repeat evaluation against T_STAR (P5.6-B2) | **0.000000e+00** |
| repeat across independent processes (A6, A6 control, B2) | **0.000000e+00** |
| polish reproducibility | **0.000000e+00** |
| deterministic fallback path, with the anchor locked | **0.000000e+00** (no fallback exists) |

**HEURISTIC BRANCH UNCERTAINTY** — explicitly *excluded* from `tau_search`, per
B4's instruction not to inflate it:

| source | measured | why excluded |
|---|---|---|
| cold versus T_STAR | 1.32e6 … 1.09e7 | a different fixed start finding a better branch is an optimality limitation, not noise |
| template refinement (B1) | 4.19e5 … 6.06e5 per refinement | neutralised by freezing T_STAR for the whole search |
| anchor convention | up to 1.34e4 | neutralised by locking one anchor; it would be *included* if the fallback were kept |

**Smallest meaningful planning-objective difference observed** across the eight
valid B2 candidates: adjacent gaps of 33.27, 128.68, 436.42, 1 401.27, 1 790.84,
10 879.51, 31 476.13 — so **33.27** is the finest distinction the population
actually contains.

The numerical repeatability floor is exactly zero, so any positive threshold is
defensible against noise; the binding constraint is that `tau_search` must stay
well below the smallest difference worth resolving.

```
RECOMMENDED   tau_search = 10.0 monetary units

    > 0        the measured repeatability floor, with margin for any future
               non-determinism (a different BLAS, a threaded linear solver)
    < 33.27    the smallest meaningful difference observed, by a factor of 3.3

Acceptance rule:   accept a candidate only if   Q_new < Q_incumbent - tau_search
```

`tau_search` is only meaningful because the anchor is locked. Under
midpoint-first-with-DSO-fallback the effective within-policy uncertainty would be
13 365.80 — **402 ×** the smallest meaningful gap — and no acceptance threshold
could both reject that artefact and resolve real differences.

## B5 — first-stage search coordinates

The first-stage feasible set is a polyhedron in the 18 investment variables:
`s, e >= 0`; `e >= phi_min*s` with `phi_min = 2`; `e <= phi_max*s` with
`phi_max = 10`; cumulative energy per node/year `<= max_capacity = 5.0` under the
cohort/calendar-life mapping; and the expected discounted investment cost
`<= budget = 1e6`. The base candidate sits **exactly** on `e = phi_min*s` at all
nine (node, investment year) pairs, so the minimum-duration constraint is active
everywhere.

Feasible directions were counted, not argued (`p56b_b5_coordinates.py`):

| point | Option A, native S/E | Option B, `(S, h)` with `h = E − phi_min*S` |
|---|---|---|
| **base** (`E = phi_min*S` active) | **18 / 36** (50 %) — every `+s` and every `−e` blocked | **27 / 36** (75 %) — only `−h` blocked |
| interior (`E = 1.25*phi_min*S`) | 36 / 36 | 36 / 36 |

In native coordinates at the base, a step in `+s` alone drops the ratio below
`phi_min` and a step in `−e` alone does the same, so **half** the poll is rejected
before any model is solved — and the search could only ever decrease power and
increase energy. That is not a tuning inefficiency; it is a directionally
incomplete poll at exactly the point the problem starts from.

**Recommended: Option B — `(S, h)` with `h = E − phi_min*S >= 0`.** The map
`E = phi_min*S + h` is a bijection onto `{e >= phi_min*s, s >= 0}`, so it is a
change of variables and not a projection: the requested candidate is exactly
representable and nothing is silently altered. Its value is precisely that it
converts the binding general constraint `e >= 2s` into a simple **bound**
`h >= 0`, which every pattern-search implementation handles natively, and which
is where the base — and any minimum-duration-optimal solution — will sit. The
remaining constraints stay general and are handled as Option C prescribes:
`h <= (phi_max − phi_min)*s`, cumulative capacity and budget by exact
master-feasibility rejection, with tangent-cone poll directions added only if the
search actually reaches the capacity or budget face.

## B6 — search-method selection (design only, not implemented)

Measured properties to design against: 18 continuous variables; a linear
first-stage polyhedron; ~115 s per evaluation; a **deterministic** oracle with
**exactly zero** numerical noise; a ~27 % rate of operationally invalid points;
nonsmooth branch structure (P5.6-A2.1: two mathematically equivalent problems
return solutions 704 apart); a working cache; independent evaluations.

| method | verdict |
|---|---|
| **Deterministic pattern search / MADS** | **Selected.** Failed evaluations are handled natively as unsuccessful poll points (the extreme-barrier treatment of hidden constraints) — which is exactly the ~27 % case. Bound constraints are native in `(S, h)`. Polling is embarrassingly parallel. Fully reproducible. Smallest implementation burden of anything that handles hidden constraints properly. |
| Trust-region model-based DFO | Rejected. A linear model needs 19 well-poised points and a quadratic 190; failed evaluations poison the interpolation set, and the branch nonsmoothness breaks the model assumption the method's convergence rests on. |
| Bayesian / surrogate | Rejected. Needs a separate feasibility classifier for the 27 % failures; the GP smoothness prior is contradicted by the measured branch structure; 18 dimensions is marginal; and reproducibility requires pinning every seed, giving up the method's main advantage. |
| Evolutionary / random | Rejected on budget alone: thousands of evaluations at ~115 s. |

Between plain GPS and MADS, the evidence favours MADS-style: GPS with a fixed
`2n` direction set can stall on a nonsmooth function, and nonsmoothness is
measured here rather than assumed. **OrthoMADS** is the specific recommendation
because its direction sequence is deterministic, which preserves the
reproducibility the whole stage has been built on.

**Specification**

| item | value |
|---|---|
| coordinates | `(S, h)` per (node, investment year); 18 variables; `S >= 0`, `h >= 0` as bounds |
| general constraints | `h <= 8S`, cumulative capacity, budget — exact master rejection, no projection |
| initial poll size | `Delta_S = 0.10 x s_base = 1.063e-3`; `Delta_h = 0.10 x phi_min x s_base = 2.127e-3` (a 10 % power move and a 10 % duration move) |
| poll | OrthoMADS directions, opportunistic (accept the first improvement and move on) |
| expansion / contraction | `x2` on a successful poll, `x0.5` on an unsuccessful one |
| acceptance | `Q_new < Q_incumbent − tau_search`, `tau_search = 10.0` |
| stopping | poll size below `0.01 x Delta_0`, or the evaluation budget |
| max evaluations | 200 for a first campaign (see B7) |
| parallel poll | 4 workers recommended, 8 only with the contention caveat in B7 |
| cache | checked before every evaluation; key includes the template id and the anchor policy |
| `INVALID_INVESTMENT` | rejected by the master check at zero solver cost; an infeasible poll point, not charged to the evaluation budget |
| `POLISH_FAILURE` / `SOLVER_CRASH` | unsuccessful poll point under the extreme barrier; recorded, never assigned a value, never extrapolated |
| final certification | re-evaluate the incumbent under both starts and both anchors, keep the best VALID, and report the full A1-style certificate |

## B7 — search-cost estimate

Measured under the final proposed recurring policy (T_STAR, midpoint anchor):

| quantity | measured |
|---|---|
| ADMM with T_STAR | median **79.4 s** (23.4 – 90.6 s) |
| polish | median **35.2 s** (24.4 – 451.6 s) |
| **one uncached successful search evaluation** | **~115 s** |
| one failed evaluation (typical, 2 blocks fail) | 132 s |
| one failed evaluation (worst observed, 33 of 48 blocks fail) | 531 s |
| one crashed evaluation | 23 s |
| blended over the observed 8 valid / 3 failed / 1 crashed mix | **~134 s** |
| one fallback-anchor evaluation (if the fallback were kept) | ~150 s |
| one cold evaluation | 446 – 590 s |
| **one final multi-start certification** (cold + T_STAR, both anchors) | **~740 s** |
| one-off T_STAR construction | 518 s |
| cache hit | 0.040 s |

Wall clock at the blended 134 s, with the one-off 518 s template build included in every cell. **These
are ideal upper-level concurrency figures**: they assume candidates are dispatched
to workers with no scheduling loss and no resource contention, which is not what
was observed.

| uncached evaluations | 1 worker | 4 workers | 8 workers |
|---|---|---|---|
| 50 | 2.0 h | 0.6 h | 0.4 h |
| 100 | 3.9 h | 1.1 h | 0.6 h |
| 200 | 7.6 h | 2.0 h | 1.1 h |
| 500 | 18.8 h | 4.8 h | 2.5 h |

**Resource contention is an observed risk, not a hypothetical one.** Each worker
runs one evaluation, and each evaluation runs IPOPT sequentially over 48 network
blocks per ADMM cycle, so `W` workers means `W` simultaneous IPOPT processes on
top of whatever threading MA97 uses. On this 8-core machine, a `SOLVER_CRASH`
(`ApplicationError: Solver (ipopt) did not exit normally`) was actually observed
during P5.6-A while several heavy processes ran concurrently, and that failure
mode is now trapped and reported rather than propagated. **4 workers is the
recommended parallelism**; 8 saturates the physical cores and reproduces the
conditions under which that crash was seen.

## Verdict

**What B settled.**

- *Anchor policy* — locked to **midpoint-only**. Both anchors have identical
  72.7 % success, the DSO fallback rescues nothing at T_STAR, and mixing
  conventions would inject up to 13 365.80 of variation and reverse candidate
  orderings.
- *Start policy* — **T_STAR only** for the search, cold demoted to certification.
  Cold was never better on objective, by 1.32e6 to 1.09e7.
- *`tau_search` = 10.0*, derived from a measured repeatability floor of exactly
  zero and the smallest meaningful candidate gap of 33.27, with heuristic branch
  uncertainty deliberately excluded.
- *Coordinates* — `(S, h)`, which recovers the half of the poll that native S/E
  loses at the base.
- *Method and specification* — deterministic OrthoMADS pattern search, fully
  specified above.
- *Cost* — ~115 s per successful evaluation, ~134 s blended, ~740 s per final
  certification, with a measured contention caveat on parallelism.

**What B did not settle.**

- *The template is not a fixed point.* Four refinements moved the base objective
  by −2 059 568.48 with no sign of convergence, so T_STAR is frozen by rule and
  the objective level it reports is knowingly not the best available. The search
  will optimise a surface defined by that declared template, and a later
  refinement would shift the whole surface by ~4–6e5 — four orders above
  `tau_search`.
- *The failure rate is ~27 %, and the two starts fail on different candidates.*
  Three of eleven operationally evaluated candidates failed the polish under both
  anchors, one crashed the solver, and one candidate that T_STAR fails on is
  solvable from cold. A pattern search can absorb this as hidden constraints, but
  a quarter of the poll returning nothing is a real efficiency and coverage cost
  that has not been reduced, only characterised.
- *The budget-boundary candidate is unsolvable.* `se|ALL|x19` crashed IPOPT under
  both starts, so the region of the feasible polyhedron near the budget face is
  currently unreachable by the oracle — which is the region an investment search
  is most likely to want to explore.

The policies are locked and the search is fully specified; the surface those
policies define is stable to evaluate but not stable to refine, and a quarter of
it cannot be evaluated at all. That is the middle verdict.

```
P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved
```

```
P5.6-B COMPLETE — ready for planner review before launching derivative-free optimization
```
