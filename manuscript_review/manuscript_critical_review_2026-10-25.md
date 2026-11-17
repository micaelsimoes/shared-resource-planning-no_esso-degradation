# Critical review of the manuscript revision

25 October 2026

## Scope and recommendation

This review compares the submitted and current LaTeX manuscripts, the reviewer comments, response_to_reviewers_v9.tex, and the current standalone highlights and cover letter. The current manuscript snapshot is commit edd7a63; the main.tex content fingerprint is b08a3365607e56f2. The user reports that the editor invited a major revision.

**Recommendation: continue as a major revision, while treating the changes as a substantial scientific reconstruction. A new submission is not warranted merely by the amount rewritten. The current revision still needs some scientific decisions and consistency corrections before it is ready to submit.**

The research question, shared asset, transmission–distribution setting, network-data family, planning horizon and basic planning–operation architecture remain recognisable. Replacing an unjustified solution method and correcting battery physics are reasons for substantial revision. They do not by themselves turn the work into an unrelated paper.

However, the new results are not a numerical refresh of the original findings. The method, admissible decisions, cost accounting and experimental scope have changed, and important original conclusions have been withdrawn. The response and revision cover letter should state this directly. The editor will decide whether the revised contribution fits the invited revision.

Elsevier's revision instructions require an invitation to revise and preparation according to the decision letter and journal guidance. They do not provide a numerical threshold for how much a manuscript may change. This recommendation is therefore an editorial judgment, not a publisher rule about scientific continuity. [Elsevier revision procedure](https://www.elsevier.support/publishing/answer/how-do-i-revise-my-submission-in-editorial-manager).

No manuscript files were edited and no numerical experiments were rerun for this review. Numerical observations below concern the reported formulation and results, not independent reproduction of the simulations. Reference labels were checked statically; both versions have no duplicate labels or unresolved internal references under that check. The manuscript was not compiled or assessed for final page layout in this pass.

## 1. What remains and what has changed

| Aspect | Submitted manuscript | Current revision | Assessment |
|---|---|---|---|
| Research question | Shared ESS location, capacity and timing with coordinated TSO–DSO operation and degradation | Same central question | Strong continuity |
| Planning formulation | Described as bilevel; scenario-specific investments averaged into expected capacities | Two-stage formulation with one implementable, scenario-independent plan | Necessary conceptual correction |
| Outer solution method | Benders-type cuts, bounds and a reported optimality gap | Derivative-free discrete search with limited neighbourhood evidence | Major method replacement; no equivalent optimality certificate |
| Battery representation | Apparent-energy utilisation driving cycling degradation; essential SOC details not exposed | Cell-side active-energy throughput, explicit SOC dynamics, calendar ageing, retention calibration and minimum SoH | Necessary physical and reproducibility improvements |
| Cost inputs and accounting | Earlier investment inputs and a transfer-payment contribution | Corrected energy-capacity costs; contracted transfers excluded from reported cost | Scientific changes requiring disclosure, not editorial changes |
| Uncertainty experiments | Five market and five operation scenarios in the original planning study | Planning searches in the single-scenario configuration; 3 × 3 scenarios evaluated for fixed plans | Material reduction in demonstrated planning scope |
| Investment timing | General staged formulation and claims about an initially selected investment | Search restricted to one installation per selected node and one common year | Formulation capability exceeds demonstrated search capability |
| Main findings | Positive “optimal” investment and storage-associated operational benefits | No evaluated baseline storage plan recovers its cost; selected fixed plans become profitable at higher flexibility prices | A changed conclusion, which must be owned explicitly |
| Narrative emphasis | Storage planning and physical operating outcomes | More attention to evaluation acceptance, diagnostic records and numerical comparisons | The main source of editorial drift |

The response draft also records an energy-cost conversion error: the submitted energy cost was divided by five rather than four when converting a four-hour system cost. The stated correction increases the affected energy-capacity costs by 25%. This should be disclosed plainly in the response, together with the fact that the revised results use the corrected inputs. It should not be disguised as simply adopting an alternative cost scenario.

The original positive-investment conclusion and the new negative baseline result are not controlled experiments differing only in degradation. Several things changed simultaneously. Do not attribute the reversal to one cause without an appropriate controlled comparison.

## 2. Length: the appendices are the first target

An approximate source-based count, excluding comments, displayed equations, tables, figures and algorithms, gives:

| Narrative prose | Submitted | Revised |
|---|---:|---:|
| Main body, Introduction through Conclusions | About 5,100 words | About 5,800 words |
| Appendices | About 900 words | About 4,600 words |

These are approximate prose counts, not journal word counts or PDF page counts. They exclude the nomenclature and much of the visible table and figure content.

The main prose has grown by about 15%, whereas appendix prose has grown more than fivefold. The revised main body still has seven figures and has eleven tables versus twelve originally. The burden is therefore not simply “too many new main-text figures”: it is the density of qualifications and the much larger supporting record.

The additional individually numbered equations should not be a principal cutting target. Many supply physical definitions that the submitted version needed, and their separation follows the author's preferred style.

## 3. Scientific decisions before another language pass

### 3.1 Match each claim to the experiment actually performed

| Claim | Current evidence supports | Current evidence does not establish |
|---|---|---|
| Stochastic shared-storage planning | A formulation allowing uncertainty and fixed-plan evaluation with nine joint scenarios | An investment selected by a multi-scenario planning search |
| Staged deployment | A general model allowing cohorts and additions | Performance of a search over independent node-specific years or staged additions |
| Degradation matters | Lower operating value and available capacity for the tested fixed reference plan | A demonstrated change in selected location, size or timing under degradation-aware versus degradation-neutral planning |
| No storage at the baseline | The returned incumbent and unfavourable evaluated alternatives under the adopted criteria | A globally optimal zero-investment decision |
| Numerical acceptance | Residual, objective-window, interface-gap and local-solve checks | A rigorous bound on error relative to the true nonconvex optimum |
| Lower capital costs help storage | A fitted marginal energy-capacity threshold | A universal capital-cost reduction at which an entire investment pays back |

The manuscript often makes these distinctions correctly in the body and limitations. They must also govern the contribution list, abstract, conclusions, response and standalone submission files.

The current final contribution bullet still refers to the “role of degradation in the investment decision”. Without a controlled planning comparison, “effect of degradation on storage value and available capacity” is the supported claim.

Investment-cost trajectories enter a linear expected expenditure and an expected-budget constraint with common physical decisions. Within this risk-neutral formulation, they act through probability-weighted cost coefficients. This is a mathematical observation about the printed model, not a defect; it means that the paper should not imply that the experiments demonstrate adaptive investment or a separate benefit from modelling the dispersion of capital costs.

### 3.2 Degradation-neutral planning is a legitimate unresolved request

The fixed reference plan has a reported margin of −4.1 k€ without ageing, within the stated numerical resolution, versus −64.4 k€ under baseline ageing. This is useful evidence about valuation.

It is not evidence that the selected investment would remain the same under every ageing calibration. An affine value relationship measured for some baseline configurations does not establish that relationship under other calibrations, at other nodes or years, or for joint investments.

A proportionate additional experiment would compare the baseline and no-ageing planning searches over the same domain and under the same declared limits, then evaluate the returned plans under consistent acceptance settings. This need not become a complete search for all seven calibrations. If that experiment is not performed, keep the narrower valuation conclusion and say that the planning part of the reviewer's request was not met.

Do not retain the response draft's argument that fixed-plan losses make re-optimisation unnecessary.

### 3.3 Post-evaluation salvage is not salvage-aware optimisation

The manuscript clearly states that salvage is calculated after the operational evaluation and is absent from the investment objective. This transparency is good.

Nevertheless, a terminal credit depending on terminal SoH can affect dispatch as well as investment timing. Adding that credit after dispatch has been selected does not reproduce an optimisation that internalises terminal value.

The reference-plan timing comparisons therefore show sensitivity of those plans to a credit convention. They do not establish preferred investment timing under a salvage-aware planning objective.

Two defensible routes exist:

- Include terminal value consistently in a suitably scoped planning and operational comparison.
- Keep the present objective, justify its scope, report the credit as a post-evaluation sensitivity, and narrow timing claims accordingly. Explicitly acknowledge the remaining limitation in the reviewer response.

The second route is a scope decision, not full implementation of the reviewer's requested reformulation.

Also specify the numerical terminal date and its mapping to the final representative block. The salvage formula uses the horizon-end symbol directly as a SoH index, while SoH elsewhere is indexed by representative blocks. The convention should be unambiguous.

### 3.4 Use numerical diagnostics as empirical evidence

The new diagnostics are useful, especially because reported objective movement after the residual-based stopping point is large relative to some storage margins. Their role is to show whether a particular reported comparison is stable under a declared procedure.

They do not bound the distance to the optimum or prove that another initialisation or penalty schedule would return the same local solution. In particular:

- Bitwise replay demonstrates repeatability of a trajectory.
- Three-start static benchmark results concern those static arrangements.
- Neither is a sensitivity study of the coordinated ADMM recourse to initialisation or penalty choices.
- Finite-window objective ranges and consensus-gap multiples are not statistical confidence intervals or rigorous bounds on optimisation error.

If one additional numerical validation task is possible, favour targeted changes of starts and penalty settings for a few decisive paired comparisons: no storage versus the reference unit at the baseline, and at a clearly positive flexibility-price setting. Apply the same comparison basis to both arms. This is more informative than adding further pages of trajectory history.

Use “certified” only with the explicit restricted definition already given. Prefer “numerically accepted” or similarly plain wording in high-level prose. Avoid any statement that the acceptance rule establishes a guaranteed precision for a ratio of values.

### 3.5 Preserve the actual meaning of the coordination benchmark

The price-taker and passive arrangements prohibit DN-to-TN reverse active-power flow. The coordinated arrangement does not impose the same restriction. The comparison therefore measures the combined effect of the chosen coordination and interface arrangements. It does not isolate coordination alone under identical export permissions.

This does not invalidate the comparison. It changes its interpretation and limits generalisation. The restriction belongs with the benchmark definition and the headline result, not only in the response letter.

For the reported operating costs:

- Price-taker benchmark: 744,770.3 k€.
- Coordinated operation: 653,873.7 k€.
- Difference: 90,896.6 k€.
- Saving relative to the benchmark: about 12.2%.
- Benchmark cost increase relative to coordination: about 13.9%.

Both percentages are mathematically valid with their respective denominators. Calling 13.9% the saving relative to the price-taker benchmark is not. The current abstract uses 12.2% correctly; parts of the response and standalone highlights still use 13.9% as the benefit.

Do not attempt to preserve the original 18.25% figure by treating the new statistic as a replacement with the same meaning. The original and revised comparisons have different configurations, scopes and denominators.

### 3.6 Label the out-of-budget valuation probes

The 2025 probability-weighted costs derived from Table 1 are approximately 256.318 k€/MVA and 253.8755 k€/MWh. For a four-hour configuration:

| Energy / converter rating | Expected 2025 investment |
|---|---:|
| 1 MWh / 0.25 MVA | 318.0 k€ |
| 2 MWh / 0.50 MVA | 635.9 k€ |
| 3 MWh / 0.75 MVA | 953.9 k€ |
| 4 MWh / 1.00 MVA | 1,271.8 k€ |
| 5 MWh / 1.25 MVA | 1,589.8 k€ |

The last two exceed the stated 1 M€ expected discounted investment budget. They are used in the capacity ladders and fitting exercise; Appendix E mentions seeded points over budget.

Keep them if they help explain value, but mark them as exploratory evaluations outside the admissible planning set. Their positive margins do not make them feasible solutions to the stated budget-constrained problem.

The returned two-node 2030 plan is a different configuration and has an expected discounted investment of approximately 992.3 k€ from the same table; this particular budget concern does not apply to it.

### 3.7 Define the zero-investment case in the mathematics

The printed normalised complementarity condition divides by available converter rating. Cohort allocation divides by total rated energy, and the degradation exponent divides by cohort rated energy. These expressions need an explicit convention when no storage is installed or a node/year has no active positive-capacity cohort.

The baseline and all no-storage comparisons make this a central case. This is a formulation clarification; it is not evidence that the implementation divides by zero.

For the network complementarity expression, a product inequality with the squared available rating on the right avoids division and remains meaningful at zero rating when the associated power bounds are enforced. For allocation and cycling, define the equations only for positive-capacity active cohorts and state how inactive storage variables and rows are handled.

### 3.8 Keep the economic boundary consistent

The main text correctly distinguishes contracted interface transfers from the commitment and deviation charges retained in the multi-scenario cost. Appendix E's blanket “settlement excluded” wording and the response draft are broader.

Define the boundary once and use the same terminology throughout. Explain whether retained deviation charges represent an external balancing cost or an internal transfer, and why they are included in the system-level metric. This is a question requiring clarification, not a finding that the accounting is necessarily wrong.

The multi-scenario comparison also changes the temporal grid and introduces commitment/deviation costs. Differences from the single-scenario value cannot be assigned solely to uncertainty. The current qualified wording should be retained.

## 4. What should remain, move or disappear

| Material | Recommended destination | Reason |
|---|---|---|
| Investment decisions, objective, operating-cost aggregation, weights and discounting | Main manuscript | Defines the actual planning problem |
| Complete SOC balance, limits, closure and converter constraints | Main manuscript | Essential physical model; the original omission was substantive |
| Active cell-energy throughput, calendar ageing and SoH propagation | Main manuscript | Core storage contribution |
| One readable outer planning algorithm | Main manuscript | Makes the method reproducible and distinguishable |
| Actual search and scenario restrictions | Main manuscript | Necessary to interpret the contribution |
| Baseline investment result and selected profitability sensitivities | Main manuscript | Principal storage findings |
| Main fixed-plan degradation comparison | Main manuscript | Direct evidence for degradation-aware valuation |
| Benchmark interface restriction, salvage convention and absence of optimality guarantees | Main manuscript, stated concisely | Material limits on the central claims |
| Exact ADMM updates, residual definitions and solver settings | Compact appendix or public supplement | Reproducibility detail; not reviewer-response-only |
| Full numerical acceptance procedure | Compact appendix or public supplement | Needed to interpret the reported evaluation labels |
| Large profile collections and full network parameter tables | Public data/supplement, with sources and a compact case description in the paper | High volume, limited narrative value |
| Every evaluation, threshold, gap, solver flag and pairwise comparison | Structured public supporting record, preferably machine-readable | Traceability without a long printed ledger |
| Sensitivity of the affine fit to flagged certificates and interval-corner construction | Supporting material | Secondary to the physical/economic conclusion |
| Detailed discount-rate reweighting and all installation-year combinations | Supporting material, with key results summarised in main text | Useful sensitivity, not another central contribution |
| Superseded certificate histories, individual replay cycle counts and recovery chronology | Response or internal work log; archive underlying data as appropriate | Development history rather than scientific exposition |
| The 49-prediction revision ledger | Internal work log, or reviewer-only if genuinely useful | Not necessary to establish the paper's scientific contribution |
| Explanation of why the original cuts/results were withdrawn and inputs corrected | Response and revision cover letter | Transparent account to editor and reviewers |
| Repeated statements that every number comes from the archived record | One data/reproducibility statement | Repetition adds little scientific information |

Supporting evidence needed to assess the final method should remain accessible to future readers. Moving it to a response letter alone is insufficient. Journal-specific supplementary requirements still need checking; Elsevier notes that these vary by journal. [Elsevier supplementary-material guidance](https://www.elsevier.support/publishing/answer/what-are-the-requirements-for-my-supplementary-material).

## 5. Where reviewer requests deserve acceptance or pushback

**Accept as scientific necessities:** correcting the invalid interpretation of Benders bounds; defining implementable investment decisions; exposing SOC physics; separating reactive converter use from cell energy throughput; accounting for calendar ageing over a long horizon; defining cost aggregation; explaining representative-day limitations; and reporting numerical failure and acceptance honestly.

**Accept in proportion to the claims:** a controlled no-ageing planning benchmark if claiming changed investment choices; terminal-value treatment if claiming economically preferred timing; and targeted ADMM robustness checks if claiming stable planning comparisons.

**Push back with a clear scope rationale:**

- There is no unique operator attribution of degradation without an explicit allocation convention or counterfactual. A short scope statement is sufficient for this system-level planning paper; an allocation study could be separate work.
- A detailed PCS thermal-ageing model is not automatically required once reactive converter use is separated from battery cell-energy degradation and the scope is stated.
- A convergence theorem for this specific nonconvex coordinated system is not required merely to report an empirical heuristic, provided guarantees are not claimed and the validation is credible.
- Suggested references should be cited for the problem they actually address. They need not be presented as foundations of the current method when they are operational applications.
- Responding carefully does not require reproducing the entire revision investigation in the paper.

Conversely, the absence of a mathematical guarantee cannot be repaired by using increasingly elaborate certificate terminology. Nor can a missing planning experiment be replaced by a confident extrapolation from a fixed unit.

## 6. Section-by-section editorial plan

### Introduction

Retain the research motivation and integrated gap. Reduce repeated statements of the two-stage architecture across the Introduction, opening of Section 2, Section 2.1 and opening of the master problem. Keep one short explanation for using a derivative-free search; the history of rejecting the original cuts belongs in the response.

Revise the contribution list first. Separate formulation capability from demonstrated case-study scope, and replace the claim about degradation changing the investment decision with the supported valuation claim unless new experiments are performed. Describe the capital-cost result as marginal.

### Method

Keep the objective, admissible investment set, physical storage model, degradation equations and one outer-loop algorithm. Keep an accessible explanation of what the network operators and shared-ESS agent exchange. Place detailed polling variants, solver recovery tiers, numerical acceptance branches and implementation settings in supporting material.

Do not repeat the absence of global optimality after every equation or subsection. State it clearly with the algorithm and refer back where essential.

### Case study

Keep the system diagram, time representation, scenario selection and weights, capital costs, storage parameters, budget and key benchmark definitions. Preserve the explicit lithium-ion technology description and the distinction between technology-cost uncertainty and the discount-rate sensitivity.

The complete input profiles and network tables are a strong candidate for supplementary data. Keep the changes of configuration between planning and multi-scenario fixed-plan evaluation visible.

### Results

Organise around scientific questions:

1. Which investment was returned, and what do the evaluated alternatives show?
2. Under what capital-cost and flexibility-price assumptions does storage gain value?
3. How does degradation change operating value and available capacity?
4. What does coordination achieve under the specified benchmark arrangements?
5. How sensitive are the fixed-plan findings to scenarios and financial assumptions?
6. How reliable and expensive are the numerical evaluations, and what is their scope?

This can combine some short current subsections without removing their qualifications.

The affine-fit table and its figure overlap; keep one main presentation and move the alternative-fit details. The discount and installation-year tables can be shortened or moved. The full acceptance trajectory figure is supporting material unless it is essential to explain a principal result.

If archived results permit it, favour one physical illustration linking storage dispatch, available capacity and value. Do not add it merely to create another figure, and do not reintroduce withdrawn operating plots.

Keep the negative baseline result: a supported negative result is scientifically useful. Avoid implying that no shared-storage investment could be economic beyond the tested domain.

### Conclusions, abstract and highlights

Revise these after decisions about claims and any additional checks. The present concise conclusions are a good basis.

The latest abstract shortening removed two requested clarifications: battery technology and the meaning of investment-cost uncertainty. Restore “lithium-ion” and a short reference to future converter/battery technology costs. This is a local correction, not a reason to expand the abstract substantially.

Keep 12.2% tied to the restricted price-taker benchmark, identify fixed-plan ageing comparisons as such, and do not convert “within numerical resolution of break-even” into a demonstrated zero-profit result.

### Appendices and supporting material

The first large editorial reduction should concern Appendix E, followed by the implementation history in Appendix A and the bulk input profiles in B–D. Retain a compact method/acceptance reference and the information needed to reproduce the final experiments.

Do not move essential caveats to supplementary files simply to make the central claims sound stronger.

## 7. Submission-file consistency

The main text is more carefully qualified than several surrounding files:

- **response_to_reviewers_v9.tex:** still describes 13.9% as a coordination benefit; says every planning statement is based on certified evaluations; states that certification bounds stopping error or resolves ratios; extrapolates the baseline search to unsearched ageing calibrations; and says the datasheet case loses more than the baseline, although its reported shortfall is 45.2 k€ versus 64.4 k€ under baseline ageing. It also claims that the abstract names battery chemistry, which the current abstract does not.
- **Standalone highlights.tex:** still says every recourse evaluation is certified, presents no-ageing break-even as established and reports the 13.9% statistic. It differs from the updated highlights embedded in main.tex.
- **cover_letter.tex:** still describes a bilevel Benders method and positive storage-associated benefits from the earlier version. It needs a revision-specific rewrite.
- **Response copies and hard-coded references:** there is an older response draft inside the Overleaf checkout and later versions in manuscript_review. Choose one final canonical response. Check section, figure, equation and table references against the final compiled revision; the response still points to removed or moved figures and earlier subsection numbers.
- **main_latexdiff.tex:** regenerate after the clean manuscript and submission files are final. It should remain a derived file.

## 8. Recommended order of work

1. **Agree the supported claims and actual experimental scope.** Start with the contribution list and a short claim/evidence table, not another sentence-by-sentence pass.
2. **Decide the small number of substantive outstanding issues:** degradation-neutral planning, salvage-aware timing claims, and targeted numerical robustness. Either perform proportionate checks or narrow the claims transparently.
3. **Correct immediate coherence issues:** out-of-budget probe labels, zero-capacity equations/conventions, cost-boundary terminology and benchmark percentages.
4. **Reduce the supporting record:** Appendix E first; then implementation history and bulky input data. Preserve a public, usable reproducibility record.
5. **Remove main-text repetition and overlapping presentations.** The goal is a storage-planning paper whose method can be understood and whose evidence can be judged.
6. **Synchronise response, cover letter, standalone highlights and abstract.** Explain the original corrections candidly and do not claim requests were fully met when they were narrowed.
7. **Compile, inspect the final PDF and regenerate the change-marked version.**

The next editing unit should be the contribution list at the end of the Introduction. It controls what the later sections must demonstrate and will prevent further effort polishing claims that need to be narrowed.

## Source locations

- [Current manuscript](/Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript/6a67305f25e8348fb71380c3/main.tex)
- [Submitted manuscript](/Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript_submitted/main.tex)
- [Original reviewer comments](</Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript_review/Reviewers Comments.docx>)
- [Response draft v9](/Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript_review/response_to_reviewers_v9.tex)
- [Current standalone highlights](/Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript/6a67305f25e8348fb71380c3/highlights.tex)
- [Current cover letter](/Users/micaelsimoes/PycharmProjects/shared-resources-planning/manuscript/6a67305f25e8348fb71380c3/cover_letter.tex)

Useful current-main.tex anchors: abstract 139; contribution list 385; architecture 415; search domain 492; search algorithm 497; recourse cost and acceptance 703; network complementarity 946; cohort allocation 984; degradation exponent 1017; case-study search restrictions 1203; investment costs 1215; benchmark definition 1349; baseline result 1368; marginal-cost fit 1419; capacity ladder 1471; degradation comparison 1511; coordination comparison 1577; multi-scenario evaluation 1632; financial sensitivities 1673; numerical performance 1735; limitations 1765; Appendix A acceptance rule 1984; Appendix E 2563; salvage 2570; implementation history and statistics 2911.
