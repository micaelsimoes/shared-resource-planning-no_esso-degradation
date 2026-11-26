# S56 design note v0: operational uncertainty and investment, and a unified core rerun (Planner draft for Advisor review, 2026-11-21)

**Status: draft.** Nothing has been built or run beyond four zero-solve audits. This note goes to the Advisor before any
harness is written. Its decisions then go to the author and the Expert.

## The author's two requests

1. **Uncertainty comparison.** Run matched investment searches:
   - *expected-profile (EV):* probability-weighted mean profiles, constructed and verified;
   - *stochastic (SP):* the full ensemble, with non-anticipativity kept.

   Both arms share the same grid, domain, budget, increments, network, ageing, discount rate, objective conventions
   and commitment/deviation formulation. Then:
   - evaluate both selected plans and x = 0 under one stochastic model, applying the acceptance criteria;
   - run 1× first, then 2×, kept separate;
   - report the result as an "achieved expected-cost improvement", not as a VSS or an optimality claim;
   - use safe parallelism: within-poll first, within an ADMM stage second, with a serial-versus-parallel comparison
     before scaling up;
   - start with the audit and a pilot, and pause on a modelling inconsistency.
2. **Unified core rerun** (assessment requested). The package covers:
   - EV vs SP;
   - the coordination benchmarks under the primary configuration;
   - the reference unit with no ageing, cycling only, and cycling + calendar;
   - the principal flexibility-price comparisons.

   The constraints:
   - **Grid:** fix one common grid first, and assess keeping 2025/2030/2035 rather than adopting the five-block grid.
   - **Conventions:** preserve the ageing and salvage conventions and the scenario-common storage schedules.
   - **Search:** use comparable initialization and effort in every search.
   - **Estimate:** measure runtime and memory, and check the persistent-worker / tight-tail incompatibility.
   - **Existing results:** preserve them, and reuse them only where they still apply.

## Facts established (audits A, B, C; `data/SRP1/Results/P515S56/input_audit_{A,B,C}/`)

**Instance and scenarios (A)**

- **The 3 × 3 instance.** It is `w89_3x3/instance/SRP1__s53_3x3.json` (`2a64e3c5`), derived from `SRP1.json` by changing
  only `Years` (2025/2028/2031/2034/2037) and the scenario counts.
- **How scenarios are drawn.** Profiles are generated at read time:
  - a Gaussian copula with KDE marginals is fitted on 3 historical profiles per season;
  - it produces a pool of 100;
  - `DataFrame.sample(n)` then draws from the pool, with seeds keyed by (year, day, type).
- **Probabilities:** 1/3 per market scenario × 1/3 per operation scenario, so 1/9 per pair.
- **A "scenario" is not a joint draw.** Each load quantity and each price uses its own seed. This does not affect the
  means.
- **SRP1's single scenario is draw row 0, not the mean.**
  - SRP1's 2025 arrays equal 3 × 3 scenario 0 bitwise (2,576 of 2,576).
  - Scenario 0 against the mean: energy price differs by 7.5 % in aggregate, wind by 9–10 %, loads by 1.6–1.8 %.
  - The 4 h price spread is 91.30 for the mean profile, against an ensemble expectation of 94.00 and 95.31 for
    scenario 0.
- **The EV variant is built and verified** (`instance/SRP1__s56_3x3_ev*.json`):
  - every array equals the probability-weighted mean bitwise;
  - the full object walk (68,365 leaves) finds 0 other differences from the 3 × 3;
  - checksum `d0d3a600`.

  **Loading it needs an injection hook.** It is a pass-through wrapper at the production scenario-checksum point,
  because production cannot read explicit profiles. It is not yet wired into the harness.

**Formulation (A)**

- **Scenario-common decisions:**
  - the investment;
  - the shared-ESS schedule, aliased to the first scenario pair (hard non-anticipativity);
  - the TSO interface P/Q (`interface_delta` is scenario-free).
- **The DSO side:**
  - the DSO interface flow is per scenario, and ADMM reaches consensus on its expectation;
  - the row-18 deviation premium (α = 0.5 in the 3 × 3 pair) and the price–flow covariance part of the settlement are
    inside Q;
  - the contracted part is a transfer and is excluded.
- **With 1 scenario, row 18, the deviation variables and the voltage pin are not built**
  (`model_construction_helpers.py:1917`). The covariance is identically 0. So the EV arm carries no deviation cost by
  construction.
- **Degradation.** The cumulative SoH per cohort couples the blocks inside the ESSO model, one per node, spanning all
  years. The cohort life is 15 years.
- **Investment domain on the five-block grid:** 3,281 admissible plans, against 1,704 on 2025/2030/2035.

**Search, runtime and parallelism (B, C)**

- **The search as run** (s47, s51) is `PB.run_mads` variant A:
  - 7 variables, Δ₀ = 4 (double or halve, floor 1), an n + 1 poll plus unit completion;
  - full poll with no opportunistic stop;
  - the incumbent is updated after the whole poll, by argmin F among determinate improvers, ties broken by lower I and
    then by label;
  - acceptance requires F_inc − F_x > max(bar_x + bar_inc, σ_Q), with σ_Q = 18,449.66 measured on SRP1;
  - a poll is batched at concurrency ≤ 5. The barrier stop is checked per batch, so the batch size can change the
    number of points evaluated, though not the verdicts.
- **Launchers.** The search launchers (PB, E1) refuse every year set except 2025/2030/2035, and refuse derived
  instances. Only the S44 harness accepts a derived instance (the 3 × 3 pair ran through it).
- **Measured runtime and memory per evaluation:**

  | instance | time | cycles | per cycle | memory |
  |---|---|---|---|---|
  | SRP1 single scenario, serial | 0.9–1.5 h | 121–200 | 23–29 s | RSS 3.2–3.5 GiB |
  | 3 × 3 five-block | 7.9 / 8.4 h | 72 / 69 | 391–436 s | footprint 19–20.5 GiB |

  - **SRP1 at concurrency 5:** contention costs 1.12–1.22× per cycle.
  - **3 × 3:** IPOPT is about 50 % of each cycle. Within it, the three DSOs take 58–103 s each and the TSO 6–13 s. The
    rest is Pyomo NL write and load.
  - **3 × 3 drift:** the objective was still descending at the production exit (−4.0 k€ and −1.7 k€ per cycle). The
    settling replay moves the x = 0 certificate from cycle 72 to about 92.
  - **Usable memory** after a reboot is 24.4 GiB.
- **The archived "≈ 8 h per 3 × 3 evaluation" is confirmed** (7.93 h and 8.37 h, production exit).
- **Parallelism:**
  - **Within an ADMM cycle, both paths are unusable under the tight tail.** The persistent-worker path is refused in code
    with the tail on (`shared_resources_planning.py:3049–3053`): worker processes keep their own solver-option holders,
    so the tail's `compl_inf_tol` change never reaches them, and their IPOPT records (the G6 input) never reach the
    parent. The `ParallelExecution` DSO pool loses those records too. The harness child refuses both
    (`p515_s44_campaign_harness.py:3280–3284`). The last persistent-worker gate predates AA, the tail and ageing. It
    measured 1.04–1.16×, against 5.09× projected.
  - **Within a cycle, the independent work is the 60 DSO blocks within the DSO stage and the 20 TSO blocks within the TSO
    stage.** The degradation coupling sits entirely inside the ESSO, which takes milliseconds. The upper bound on this
    lever is about 1.8–1.9× at 3 × 3.
  - **Candidate-level concurrency** is the only supported and gated path. It is memory-bound to 1 for 3 × 3
    evaluations.

**Core-package inventory (C)**

- **On 2025/2030/2035 with the single scenario,** every core result exists and stays applicable under its original
  assumptions:
  - benchmark: the coordinated cell `ref:7aa017f0` and the static arms;
  - ageing cells: `e_no_ageing`, `e_c2`, `e_c2_calfade` / `ref:bd504ecf`;
  - flexibility-price cells: m = 1.5, 1.75, 2, the break-even D cells, and E2.
- **On any multi-scenario grid,** none applies except the 3 × 3 pair.
- **The static benchmark refuses multi-scenario instances** (`uncoordinated_benchmark.require_single_scenario`). Its
  multi-scenario definition is a formulation question.
- **No multi-scenario case exists on 2025/2030/2035.** It can be derived exactly as the 3 × 3 was.

## Proposed design: decisions for review

**D1. Common grid: recommend 2025/2030/2035 with a 3 × 3 ensemble ("SRP1-3×3"),** derived from `SRP1.json` by
changing only the scenario counts.
- **Why this grid:**
  - **The same years, blocks, cost table and admissible domain (1,704)** as every published result, so the
    single-scenario results stay directly comparable on the same grid.
  - **A nested structure, if the prefix property holds at 2030 and 2035** (to verify): SRP1 single scenario = draw 0;
    EV = the mean; SP = the full ensemble. The published results then become a third, "one-draw" planning arm at no
    cost.
  - **Cheaper evaluations:** 12 blocks against 20, so a projected ~0.6× per cycle and ~0.6× memory (about 11–12 GiB,
    which may allow 2 concurrent).
- **Cost:**
  - the five-block 3 × 3 pair is not reusable as a result on this grid;
  - the 5-year block is coarser for degradation, as in the published results;
  - the 2030/2035 draws are new.
- **The alternative,** the five-block grid, reuses one pair, doubles the domain (3,281) and costs 8–10 h per stochastic
  evaluation.

**D2. EV construction.** Use the mean of the realised draws (A1, as the author specified). Load it through the
injection hook (option a), with no production change and a declared checksum.
- **A2 and A3:** plain means, as built.
- **A5:** keep production's constant for the reactive-flexibility arrays (a tolerance, not data). Rebuild if the
  Advisor agrees.

**D3. EV arm, commitment terms.** The EV arm has no row 18 and no deviation settlement, because they are vacuous with
one scenario.
- **The claim:** this is the defining property of expected-profile planning, the same formulation with its uncertainty
  terms empty. It is not a change of formulation.
- **What restores them:** evaluating the EV plan under the SP model.
- **Advisor:** confirm, or name the confound.
- **The SP arm** uses row 18 at α = 0.5, as in the 3 × 3 pair, and the frozen ESS parameters (`SRP1_ESS_Params.json`:
  SoH point "end", salvage outside the ESSO objective).

**D4. Search.** Both arms use the same `PB.run_mads` variant A: seed x = 0, Δ₀ = 4, the same caps (N^max 20,
completion cap 30, MAX_POLLS 60), the same tie rule and the same batch size, set to the smaller arm's concurrency, so
that the barrier-stop behaviour is identical. Needed:
- instance plumbing in the launcher;
- σ_Q for each arm's instance: on SRP1 it was measured from replicate starts, so how should it be measured cheaply for
  the SP instance? (open; Advisor)
- a settling treatment. Multi-scenario objectives were still drifting at the production exit, so search acceptance at
  the exit could accept on drift. Options:
  - (i) accept at the exit with σ_Q inflated by a measured drift bound;
  - (ii) settle every evaluation, at +20–30 % time.

**D5. Common evaluation.** Each selected plan and x = 0 is evaluated under the SP model to the settled certificate
(rule v6, or the 3 × 3 settling criterion), with bars. Every difference is reported with its resolution and
terminal-step ratio. If both arms select the same plan, the improvement is 0 by construction, reported as an
unchanged decision.

**D6. Parallelism.**
- **Candidate-level within each poll.** Deterministic order is preserved, because the poll completes before the
  incumbent update. Directories are isolated: the S44 harness already isolates the eval and working directories, and
  base-case diagram writes are redirected for derived instances.
- **Thread caps** stay at 1.
- **Worker count** is set from the measured footprint at 0.85 × 24.4 GiB.
- **Within-stage parallelism is not pursued in this study.** It is incompatible with the tail today, a fix touches
  production, the key and a gate, and the bound is ≤ 1.9×. Offer it as a separate work package.

**D7. Core rerun.** It is an author decision which configuration is "primary" (single scenario, or SP). The note
proposes:
- **Kept as published:** the single-scenario core results, unchanged, on the same grid, as the deterministic study.
- **New:**
  - the EV vs SP comparison at 1× and 2×;
  - the reference unit's three ageing variants;
  - the 1× vs 2× comparison under SP, as fixed-plan evaluations (x = 0 and the unit, at 1× and 2×; 3 variants at 1×).
- **The benchmark under uncertainty** needs a formulation ruling: what the price-taker DSO sees, its commitment, the
  passive schedule. Out of scope until ruled.

**D8. Pilot (before any search):**
1. **Build gates:** SRP1-3×3 against `SRP1.json`: the prefix property (row 0 = SRP1 in every year), the read checksum,
   and the 12-block structure.
2. **Short runs, 3 cycles each, under armed guards:**
   - SP x = 0 and EV x = 0, serial, for per-cycle time and footprint;
   - EV × 5 concurrent, for contention;
   - SP × 2 concurrent, only if the footprint ≤ 0.85 × A / 2.
3. **One full SP x = 0 evaluation** to the production exit and to settling, for cycles to exit and drift. Projected
   about 5–7 h, so it must be named.
4. **The serial-versus-parallel equivalence check** on the same inputs: the same Q within the bar and the same
   certification status.

**Output:** a measured projection for the core package.

## Questions for the Advisor

1. D1: is 2025/2030/2035 with an ensemble the right common grid? What does the coarser block cost for the uncertainty
   finding?
2. D3: is the EV arm with structurally absent commitment terms the correct expected-profile counterpart, or a
   confound? Is the "mean of realised draws" the right EV?
3. D4: how should σ_Q and the settling treatment be set for the multi-scenario search so that "determinate improver" is
   not a drift artefact? Is the matched batch size right?
4. D5 and D6: anything unsafe in candidate-level parallel polls with this harness, or in the comparison design?
5. What would make the EV vs SP difference uninterpretable? Which single check in the pilot best guards against it?
