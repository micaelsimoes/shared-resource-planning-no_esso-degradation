# Proposed additional simulations for the manuscript revision

**Handover to the Expert for preparation of an implementation and simulation plan for the Planner**

Date: 25 October 2026

## 1. Purpose and scope

The editor has invited a major revision of the shared-ESS planning paper. The manuscript has undergone substantial methodological and numerical corrections. The next studies should address specific gaps in the evidence while keeping the scope and eventual manuscript concise.

Please assess the proposed studies below and prepare a bounded implementation and execution plan. This brief proposes experiments; it does not report new simulation results or prescribe a desired outcome.

The main questions are:

1. Does representing degradation change the selected investment, and how does a degradation-neutral investment perform when ageing is represented?
2. Are the principal economic comparisons stable under reasonable changes in numerical initialisation and coordination settings?
3. How much of the reported coordination benefit remains when the coordinated and static arrangements have the same restriction on reverse active-power flow?

Two further questions depend on the contribution retained in the paper: the value of scenario-based investment selection and the effect of internalising terminal value.

| ID | Study | Priority | Intended contribution |
|---|---|---|---|
| E1 | Degradation-aware versus degradation-neutral planning, with cross-evaluation | Core | Establish the effect on investment decisions and subsequent economic performance |
| E2 | Numerical robustness of decisive paired comparisons | Core; screen before costly searches | Assess whether the economic conclusions depend on the numerical configuration |
| E3 | Coordination benchmark with matched reverse-flow restrictions | Core | Clarify the interpretation of the coordination benefit |
| E4 | Controlled scenario-based investment comparison | Conditional | Support a demonstrated stochastic-planning contribution |
| E5 | Terminal-value-aware optimisation | Conditional | Support claims about economically preferred installation timing |
| E6 | Physical explanation using existing operating results | Supporting; prioritize reuse | Connect storage operation and degradation with economic value |

E1–E3 form the proposed minimum simulation package. E6 should use existing final records wherever possible. E4 and E5 should receive separate effort estimates and recommendations.

## 2. Starting point and controls

Repository: /Users/micaelsimoes/PycharmProjects/shared-resources-planning/

Current manuscript: /Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript/6a67305f25e8348fb71380c3/main.tex

Audit snapshot: manuscript checkout commit edd7a63. Check for subsequent changes before implementing the plan.

The current manuscript reports:

- A 108-bus integrated system and a 15-year horizon.
- Single-scenario planning searches; a separate 3 × 3 market/operation scenario study evaluates fixed plans.
- One installation per selected node, with a common installation year in the implemented search.
- A 1 M€ expected discounted investment budget and a 2% baseline discount rate.
- A reference investment of 0.25 MVA / 1 MWh at node 7, installed in 2025.
- A reference-unit baseline-ageing margin of −64.4 k€; its no-ageing margin is −4.1 k€, within the reported numerical resolution.
- A reference-unit margin of +46.3 k€ at twice the baseline flexibility price.
- Static benchmarks prohibiting DN-to-TN reverse active-power flow, whereas the coordinated benchmark permits it.

These values are context from the manuscript, not numerical targets for new runs. Recover full-precision values and provenance from the underlying records.

For every controlled comparison, hold all unrelated assumptions fixed: corrected investment costs, admissible investment set, network and operating data, temporal grid, scenario probabilities, cost accounting, discounting, storage efficiencies, SOC conventions and numerical acceptance definitions.

Specify the comparison protocol and numerical thresholds before the production campaign. Document any changes prompted by pilot checks and apply them consistently to the affected comparison arms.

The Expert should resolve the following before campaign design:

- Explicit handling of zero-capacity storage in normalised complementarity, cohort allocation and degradation expressions.
- A consistent cost boundary, including treatment of contracted transfers, deviation settlement, commitment charges, regularisation and feasibility penalties.
- The numerical horizon-end date and its mapping to terminal SoH and representative blocks.
- Which archived evaluations remain comparable with the final implementation. Several planning searches used earlier solver settings; re-evaluating their incumbents does not establish what a search under updated settings would select.
- Budget admissibility: the 2025 four-hour units with 4 and 5 MWh cost approximately 1.272 and 1.590 M€, respectively. They may be exploratory valuation probes but cannot be feasible candidates under the 1 M€ budget.

## 3. E1 — Degradation and investment decisions

### Research question

Does degradation-aware planning select a different investment, and does a plan selected while ignoring ageing incur a different cost when assessed with the baseline ageing model?

### Proposed design

Compare two planning configurations:

- **A:** baseline cycling and calendar ageing.
- **N:** no cycling or calendar ageing, with SoH fixed at one.

Use the same investment domain, financial inputs, search algorithm, declared search limits and comparable starting conditions. Preserve all other storage physics. A cost evaluation from one ageing configuration cannot be reused as the operating cost of another.

Let x_A and x_N denote the returned plans. Re-evaluate both under the baseline ageing model, re-optimising operation while holding each investment fixed. The essential results are therefore:

| Investment plan | Evaluation with baseline ageing | Evaluation without ageing |
|---|---|---|
| x_A | Common physical/economic assessment | Useful additional diagnostic |
| x_N | Essential cross-evaluation | Assessment used for degradation-neutral selection |

Include the no-storage reference on a consistent basis. Reuse genuinely identical evaluations.

The primary economic setting is the baseline flexibility price. Predefine the existing doubled-price setting as a secondary comparison if resources allow, since it already represents an economically active storage case. Report the selected settings regardless of outcome.

### Required outputs

A compact table showing node(s), converter rating, energy capacity, installation year, investment expenditure, operating savings, investment margin and terminal available capacity/SoH.

Report the evaluated difference F_A(x_N) − F_A(x_A), where both costs use the baseline ageing model and the same accounting convention. Do not assume it is non-negative: the searches are heuristic and may return different-quality incumbents.

Record search limits, evaluated candidate counts, stopping reasons and neighbourhood coverage.

### Interpretation

Changed plans with a resolved cross-evaluation difference support a statement about the effect of degradation on planning. Identical plans support a narrower finding about the tested setting. Differences below the numerical criterion remain unresolved. None of these outcomes should trigger an unplanned search for a more favourable example.

## 4. E2 — Numerical robustness of economic comparisons

### Research question

Do changes in initialisation or ADMM penalty settings materially change the inferred value of storage or its investment margin?

### Proposed design

Use paired no-storage and reference-unit evaluations at the baseline and doubled flexibility prices. Both members of each pair must use the same economic assumptions and the same declared numerical protocol.

The Expert should propose a limited matrix that varies:

- Initialisation, using the baseline and a small number of documented alternative starts.
- Initial penalty levels separately, for example 0.5 and 2 times the baseline channel penalties, with the remaining adaptation/freeze rules unchanged.

A full factorial study is not assumed. As an illustrative upper screening matrix, five configurations for each of four plan/price combinations would require 20 evaluations before reuse: the baseline configuration, two alternative starts and two alternative penalty levels, varied separately. Recommend a smaller staged pilot if appropriate and estimate its cost.

Use a predefined rule for selecting or summarising multiple starts, applied consistently to every comparison. Report unsuccessful and unaccepted runs.

### Required outputs

For each configuration, report operating cost, storage value, investment margin, acceptance status, residuals/interface gap, major physical-feasibility measures, cycles and runtime.

Check the final returned schedules against final available capacities and the relevant network, SOC, closure and converter constraints. Report allowed slacks and violations in interpretable units.

The principal robustness measure is the variation in paired investment margins and rankings, not merely a small relative change in the much larger total system objective.

If practical, assess a small equivalent centralized reference problem. Verify that network physics, storage constraints, degradation and cost accounting match; do not assume an existing centralized utility already provides an equivalent benchmark. A local centralized solution is a reference, not a global certificate.

### Interpretation

Repeatable trajectories do not establish independence from initialisation. Empirical spreads do not constitute rigorous optimisation-error bounds. If plausible settings change a central economic conclusion, identify that uncertainty before expanding the planning campaign.

## 5. E3 — Coordination with matched reverse-flow restrictions

### Research question

How much benefit is obtained through coordination when both arrangements prohibit DN-to-TN reverse active-power flow?

### Proposed design

Under the baseline single-scenario configuration without storage, compare:

- **P_NRF:** the existing price-taker arrangement with no reverse flow.
- **C_NRF:** coordinated operation with the same no-reverse-flow restriction.
- **C_OPEN:** coordinated operation with reverse flow permitted, as in the current coordinated case.

Use the same temporal grid, operating data, cost function, control availability and feasibility criteria. Apply the reverse-flow constraint with consistent signs and physical meaning on both sides of each interface. Use comparable initialisation/solution-selection procedures.

Reuse the existing arrangements only after checking compatibility with the final implementation.

### Required outputs

Report discounted operating costs, principal cost components, curtailment, reverse-flow measures and numerical status.

Calculate:

- Coordination benefit conditional on NRF: Q(P_NRF) − Q(C_NRF).
- Benefit of permitting reverse flow conditional on coordination: Q(C_NRF) − Q(C_OPEN).

Their sum recovers Q(P_NRF) − Q(C_OPEN) on a consistent basis. This is a conditional decomposition along the chosen comparison path, not a universal separation of independent causal effects.

### Interpretation

State the denominator for every percentage. For the currently reported costs, 12.2% is the saving relative to the price-taker benchmark, while 13.9% is the benchmark's cost increase relative to coordination.

Report the computed outcomes even if the expected ordering fails; investigate numerical or modelling differences rather than enforcing a preferred ranking.

## 6. E4 — Scenario-based investment selection

**Conditional priority:** elevate this study if demonstrated stochastic investment selection remains a principal contribution.

Compare an expected-profile planning configuration with the chosen joint-scenario configuration. Construct expected profiles consistently from the scenario probabilities. Keep the temporal grid, investment domain, ageing model, financial assumptions and cost definitions common.

The current single-scenario and multi-scenario results differ in temporal discretisation and commitment treatment. They cannot directly serve as a controlled comparison of uncertainty.

Start with a common, budget-feasible candidate shortlist evaluated under both configurations. If rankings differ materially and computation is feasible, extend to comparable planning searches.

Re-evaluate both returned plans under the same joint-scenario configuration, retaining the model's actual restrictions on scenario-dependent dispatch, including its common ESS schedule.

Report the selected plans and their costs under this common assessment, together with feasibility, numerical status and computational burden.

A shortlist comparison establishes candidate-ranking sensitivity. Heuristic searches support an evaluated benefit of the selected scenario-based plan; do not label the difference an exact value of the stochastic solution or claim global stochastic optimality.

## 7. E5 — Terminal value and installation timing

**Conditional priority:** elevate this study if claims about economically preferred installation timing remain prominent.

Compare planning and operation under:

- Gross objective I + Q.
- Objective I + Q − terminal salvage credit.

Internalise the terminal credit wherever the relevant dispatch, SoH and investment decisions are optimised. Adding it only to the final score is insufficient when it depends on terminal SoH.

Use a justified terminal-value convention, with explicit discounting, remaining lifetime, terminal SoH and treatment of converter value. Clarify the horizon-end date.

Assess returned investment plans under a common declared terminal-value convention, with operation re-optimised consistently. Report timing, capacity, gross operating costs, terminal credit, net cost and terminal capacity separately.

If the effort is disproportionate, recommend narrowing the timing claim and retaining the current post-evaluation sensitivity. State explicitly that this does not constitute terminal-value-aware optimisation.

## 8. E6 — Physical explanation from existing results

Prepare one compact illustration connecting dispatch, degradation and economic value. Prefer final accepted records for the reference unit or an already selected economically active case.

Useful elements are:

- Charging/discharging and SOC, with the applicable energy limits, on a declared representative day.
- The associated reduction in flexibility activation or another relevant cost component against no storage.
- Available energy capacity across representative blocks.

Use a predefined day-selection rule and distinguish illustrative profiles from representative-day-weighted horizon estimates. Do not imply continuous annual chronology.

Identify whether existing records contain the required outputs. If a new export or solve is necessary, list it separately. Avoid using superseded trajectories alongside final economic values without establishing compatibility.

## 9. Requested implementation and execution plan

Please return:

1. **Evidence inventory:** which required results already exist, their provenance, and which evaluations can be reused.
2. **Technical specification:** equations/code paths to change, cost and terminal-date conventions, and physical checks.
3. **Bounded experiment matrix:** identifiers, purpose, plan or search domain, economic setting, ageing configuration, scenarios, time grid, initialisation, penalty settings and stopping criteria.
4. **Resource estimate:** new searches/evaluations, expected time and peak memory, pilot size, campaign limits and handling of unresolved runs.
5. **Execution order and decision points:** a small E2 screen first; E1 and E3 next; E6 extraction alongside these; separate recommendations for E4/E5.
6. **Validation protocol:** consistent feasibility and acceptance checks, full-precision paired comparisons, and treatment of failed or numerically unresolved evaluations.
7. **Reporting specification:** compact main-paper tables/figures, public supporting records, and reviewer-response explanations.

Preserve existing archived results. New outputs should have distinct identifiers and sufficient metadata to establish the model/data version and configuration. Reuse of cached costs must account for the full configuration; matching investment coordinates alone is insufficient.

Use declared finite budgets and document what each stopping point establishes. If a model correction changes earlier results, identify the affected comparisons and the necessary reruns.

## 10. Manuscript implications and limits

The studies should sharpen the paper rather than expand it into another long campaign report.

| Study | Likely main-paper output | Detailed supporting material |
|---|---|---|
| E1 | One planning/cross-evaluation table and interpretation | Search histories and additional operating indicators |
| E2 | Short robustness summary | Complete setting/status/feasibility matrix |
| E3 | Compact benchmark table or replacement of the existing table | Per-start and per-block checks |
| E4, if pursued | One common-assessment comparison | Scenario and search records |
| E5, if pursued | Timing/capacity and terminal-value comparison | Detailed dispatch and financial accounting |
| E6 | One physical illustration, preferably replacing less informative material | Additional representative-day profiles |

Additional ageing calibrations, exhaustive penalty sweeps, larger networks, full chronological annual simulation, operator-specific degradation allocation and new storage technologies are outside the proposed minimum package.

Report positive, negative, unchanged and unresolved results. Final contribution statements should follow the evidence. Numerical acceptance is an empirical assessment under the declared procedure; it does not establish global optimality.

## References and working material

- Current manuscript: /Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript/6a67305f25e8348fb71380c3/main.tex
- Submitted manuscript: /Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript_submitted/main.tex
- Review comments and response versions: /Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript_review/
- [Critical manuscript audit](/Users/micaelsimoes/.codex/.chatgpt-projects/g-p-6a2ac761c6508191bd87a3991e458e00/manuscript_critical_review_2026-10-25.md)
- [Boyd et al., Distributed Optimization and Statistical Learning via ADMM](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf), Section 9, for the general dependence of nonconvex ADMM outcomes on starts and penalties.
