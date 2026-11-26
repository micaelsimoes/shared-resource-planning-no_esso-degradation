# P5.15 W182d — claims listing A7 and A8 at Overleaf `edd7a63` (Addendum 82 v2)

Zero solves. This is a listing, not a rewrite: one-line notes only. The clone was read through `git show edd7a63:<file>` only
(clone HEAD `edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838`, `main.tex` sha256 `b08a3365607e56f2…`, which matches the audit's
fingerprint). The repository code head at the run was `ab1d8d7b` (the harness commit). The letter is the clone's
`response_to_reviewers_draft.tex`, whose header reads "v9".

## Blocked on Planner

Nothing blocks. Four points need a ruling or a pass to the author or expert. None of them changes what runs next.

1. **Over-budget points beyond the audit's 4 h set.** Audit 3.6 names only the 4 h rungs of 4 and 5 MWh. Three 2 h
   rungs in 2025 are also over the budget: 7: 1.5/3 (1,146,109 €), 7: 2/4 (1,528,145 €) and 7: 2.5/5 (1,910,182 €).
   That puts 5 of the 10 points of the §4.2 affine fit outside the admissible set, and 3 of its 6 certified-only points.
   Whether the probe label should also cover the 2 h rungs and the fit is a manuscript decision for the expert or author.
2. **Scope of "§4.5".** I read it as the numbered subsection 4.5, *Multi-Scenario Evaluation*. Its wording is careful:
   no sentence in it is listed, and two are borderline. The installation-timing comparisons are in §4.6, which the order
   does not name. Only three sentences outside the named scope are listed, as a supplement, and they were picked by hand,
   not by a scan.
3. **The letter's statements about the abstract are false or partly false** (A8(3)):
   - l. 77 says "the abstract names the chemistry". It does not.
   - l. 80 says "the abstract reports … 13.9 %". The abstract reports 12.2 %.
   - l. 76 says "the abstract now says both". The abstract says "investment-cost uncertainty" but not that it is
     future technology cost.

   These are for the author or expert.
4. **The frozen record names 13.9 % as a benefit.** In `frozen_step6_tables_v1_590088fe.json`, `tables.benchmark.benefit_relative`
   = 0.139012 = benefit / coordinated Q. Its name reads as a saving, which is what the letter and standalone highlights
   print. I record this only. A frozen artefact is not to be changed.

## Changed

- `ab1d8d7b`: harness `p515_s53_w182d_claims.py`, committed before the real run.
- This commit adds:
  - the report `P5_15_W182D_CLAIMS.md`;
  - `data/SRP1/Results/P515S53/w182d_claims/`:
    - `w182d_claims.json`, sha256 `4388c277404b80a5980c51ab3fcfeaca69e05c915c83a485f9030c9d56c3f7b1`;
    - `manifest_sha256.json`, sha256 `f8c270fe…`, with 21 inputs, none uncommitted;
    - `launch.log`, sha256 `b654fea7…`.
- Run conditions:
  - Run attached, with stdout and stderr captured, into a fresh directory.
  - `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0 solves.
  - Pickle loads were blocked (0 attempted).
  - JSON was written through `gate_result_io`.
- Production was imported only for three readers and the closed-form first-stage check:
  - `SharedEnergyStorageParameters.read_parameters_from_file`;
  - `shared_energy_storage_data._get_operational_scenarios_info_from_excel_file`;
  - `shared_energy_storage_data._get_investment_costs_from_excel_file`;
  - `shared_resources_planning._check_candidate_first_stage_feasibility`.

  No model was built.
- Six negative controls were all refused, or flipped as declared: an altered quote; a planted abstract sentence; budget
  1.6e6 admits 1.25/5; 1/4 is admissible in 2030; a planted "lithium-ion"; 13.9 removed from the highlights.
- Nothing in the clone, `manuscript_review/`, production code, frozen files, TASKS.md, REVISION_CONTEXT.md or the
  brief was edited.

## Found

### A7 prediction against outcome

**Expert prediction: ≥ 6 sentences listed. Outcome: 16 listed, so the prediction held.** Of these, 4 are in `main.tex`,
2 in the standalone highlights, 9 in the letter and 1 in the cover letter. A further 11 are borderline and 5 are
"other" (audit points outside the four categories).

Counts of listed sentences by category (a sentence can carry two categories):

| Category | Count |
|---|---:|
| (i) multi-scenario search | 3 |
| (ii) value of stochastic selection | 3 |
| (iii) installation timing | 3 |
| (iv-s) certification claimed for evaluations that were not certified | 6 |
| (iv-b) certification said to bound error or resolve a ratio | 7 |

**No in-scope sentence states a preferred installation year.** The (iii) entries imply that timing or staging was a
searched or selected dimension.

**How coverage was checked.** Every sentence fragment in scope that matches the harness keyword pattern (stochastic,
uncertain, scenario, trajector, …, certif, resolution, determinat, bound, break-even; full pattern in the JSON) is
assigned. 34 are covered by an entry and 75 are excluded with a stated reason; 0 are unassigned.

Scope searched:

- `main.tex`:
  - abstract (l. 140–150);
  - highlights (l. 162–168);
  - contributions, the "To address this gap" paragraph plus the list (l. 381–391);
  - §4.5 (l. 1632–1672);
  - §5 (l. 1791–1813).
- The whole of `highlights.tex`.
- `cover_letter.tex`, inside `\begin{letter}`.
- The letter after `\begin{document}`, without the reviewer quotes (`\rcomment` blocks) and comment lines.

### A7 — listed (16)

Abbreviations in the table:

- "l." without a file means `main.tex` at `edd7a63`.
- "w.r." is the manuscript's "within numerical resolution" verdict, defined at l. 2037 as a comparison that does not
  meet the determinacy criterion.
- Every quote and every evidence passage is asserted at its line by the harness.

| Id | File : line | Quoted sentence | Cat. | Why — what was actually run |
|---|---|---|---|---|
| M-ABS-1 | main.tex:145 | "This paper proposes a two-stage stochastic framework for shared battery ESS siting, sizing and installation timing under investment-cost uncertainty." | ii, iii | Cost uncertainty enters only as probability-weighted coefficients of I(x) and of the expected-budget row: l. 541, l. 690; `shared_energy_storage_data.py:376`; the search lattice `p515_s47_phase_b_record.py:385` uses one expected discounted coefficient per year. With one common plan, the selection is that of the expected-cost problem, and no value of stochastic selection was measured. Timing: one cohort per node and a common year (l. 1203); node-specific timing and staging were not explored (l. 1781). |
| M-ABS-2 | main.tex:145 | "A derivative-free search explores discrete investment plans, and an … (ADMM)-based coordination procedure evaluates their operation under load, renewable-generation and market-price uncertainty." | i | The searches ran in the single-scenario configuration (l. 1203). Only two fixed plans were evaluated 3 × 3, and the search was not repeated there (l. 1636, 1781). |
| M-HL-1 | main.tex:163 | "A stochastic framework plans shared storage at transmission--distribution interfaces" | i, ii | As M-ABS-1/2 (l. 1203, 690). |
| M-CON-1 | main.tex:387 | "…that determines a multi-year investment plan and evaluates it against scenario-dependent coordinated TSO--DSO operation." | i, iii | The incumbent was evaluated single-scenario (l. 1203, 1636). The "multi-year plan" is one cohort in a common year (l. 1781). |
| H-3 | highlights.tex:12 | "Every recourse evaluation is certified; every planning difference has a resolution" | iv-s, iv-b | 10 of 42 campaign evaluations are uncertified (l. 1739). The 41 search evaluations used the residual exit (l. 1760). The 3 × 3 pair and the returned 2× plan are uncertified (l. 1636, 1507). There is no rigorous bound (l. 2003, 1777). |
| H-4 | highlights.tex:13 | "Without ageing the reference storage unit breaks even; with ageing it loses" | iv-b | Row 46 is w.r. (0.32×, l. 2676). w.r. means the rule was not met (l. 2037); it is not a bound placing the margin at 0. |
| L-01 | letter:42–43 | "in which each candidate plan is evaluated by the coordinated TSO--DSO recourse under an explicit stopping and certification rule" | iv-s | Search candidates used the production (residual) exit (l. 1760). |
| L-04 | letter:78 | "…the reference unit again loses determinately to the plan without storage, and its value is 0.909--0.934 of its single-scenario value, against 0.933 predicted from the mean-profile price spread (Section~4.5)." | iv-s, iv-b | §4.5 says these evaluations were "not certified" (l. 1636) and the range "is not a certified interval" (l. 1640). "0.933 predicted" does not occur in main.tex. |
| L-05 | letter:102 | "a derivative-free search driving certified recourse evaluations, with no cuts." | iv-s | l. 1760. |
| L-07 | letter:137 | "The method claims no optimality certificate; what it certifies is each evaluation (Section~2.2.7), and every planning statement in the paper is a pairwise comparison of certified evaluations with a stated determinacy threshold." | iv-s | Uncertified evaluations underlie stated results (l. 1739, 1760, 1636, 1507). |
| L-08 | letter:168–169 | "On sensitivity to the penalty schedule: the certification bounds the stopping error of each evaluation, not the basin the nonconvex recourse converges to" | iv-b | main.tex says the opposite (l. 2003; l. 1777: "does not provide a guaranteed bound"). |
| L-09 | letter:229–230 | "what is certified is each recourse evaluation, and each reported difference carries the threshold above which it is called determinate (Section~2.2.7)." | iv-s | l. 1739, 1760, 1636. |
| L-10 | letter:230–232 | "The resolution is set by $\tau = \delta_R V/4$ with $\delta_R = 0.07$ of the single-scenario storage value, so that a ratio of two values is resolved to $\delta_R$." | iv-b | `eq:acceptance_tolerance` (l. 1993) only prescribes τ. Nothing establishes that a ratio is resolved to δ_R (l. 2003). |
| L-12 | letter:263 | "without ageing $-4.1$~k\euro{} (within resolution, i.e.\ break-even)" | iv-b | Turns w.r. into break-even (row 46; l. 2037; audit 6). |
| L-13 | letter:264 | "Under the no-ageing calibration the unit is at break-even and the plan is indeterminate at this precision." | iv-b | As L-12, plus "precision" (l. 2003). |
| C-1 | cover_letter:14 | "The proposed approach is formulated as a bi-level stochastic optimization model in which the upper level determines siting, sizing, and staged investment decisions under investment-cost uncertainty, while the lower level evaluates these decisions through coordinated operation." | ii, iii | Staging was not searched (l. 1781). Cost uncertainty acts through expected coefficients (l. 690). This is unrevised submitted text. |

### A7 — borderline (11; not counted)

| Id | File : line | Quote (abridged) | Cat. | Note |
|---|---|---|---|---|
| M-CON-3 | main.tex:389 | "…explicit numerical acceptance criteria for assessing the resolution of planning comparisons." | iv-b | Implies a bound only when read with l. 368. |
| M-45-1 | main.tex:1636 | "The resulting shortfall of 81.3~k\euro{} exceeds the reported numerical resolution of 18.0~k\euro{}." | iv-b | A resolution is quoted for evaluations the same paragraph calls not certified. The rule behind 18.0 was not checked. |
| M-45-2 | main.tex:1640 | "…is not a certified interval." | iv-b | Implies that certification yields intervals. |
| M-5-1 | main.tex:1795 | "The two-stage stochastic formulation links investment decisions with coordinated TSO--DSO operation under uncertainty, …" | i | A statement about the formulation. |
| L-00 | letter:48–49 | "loses 64.4~k\euro{} over the horizon, determinately, as does every other plan evaluated" | iv-s | Six neighbours were "evaluated by the search only", under the production exit (l. 2808). |
| L-02 | letter:54–55 | "…to state the resolution at which each difference is claimed." | iv-b | |
| L-02b | letter:81 | "…and the resolution at which the difference is claimed (Section~2.2.7, …)" | iv-b | |
| L-03 | letter:58 | "…indeterminate at the precision of this study." | iv-b | |
| L-06 | letter:114–115 | "stage selects one scenario-independent multi-year plan; the second stage evaluates it under scenario-dependent coordinated operation." | i | A statement about the formulation. |
| L-11 | letter:221 | "2035 is worse than 2030 in gross terms (+42.5~k\euro{}, determinate) and indistinguishable net of salvage …" | iii | A fixed-plan comparison with a credit applied after evaluation (l. 1363). The paper says neither year is preferable (l. 1677). |
| L-14 | letter:264 | "…changes the value by $+4.9$~k\euro{}, within resolution, so the floor is not what makes storage uneconomic here" | iv-b | Reads a w.r. verdict (row 58, 0.38×) as the absence of an effect. |

### A7 — other, outside the four categories (5)

- **M-CON-4**, main.tex:390, "the role of degradation in the investment decision". This is the audit 3.1 point: the
  evidence consists of fixed-plan re-evaluations only.
- **L-O1**, letter:58, and **L-O3**, letter:264. Both extrapolate the baseline search to calibrations that were not
  searched (audit 7, 3.2), and both rest on the "affine … ladders", which include over-budget points (A8).
- **L-O2**, letter:249, "it loses more under the datasheet calibration". The datasheet calibration loses 45.2 k€ and the
  baseline loses 64.4 k€ (rows 48–49), so the datasheet case loses less.
- **C-O1**, cover_letter:24. A withdrawn discretization result, in an unrevised cover letter.

**Supplement, outside the named scope (picked by hand, not by a scan):**

- main.tex:368, "its stopping error is seldom quantified, so that planning differences are reported without a stated
  resolution" (iv-b).
- main.tex:361, "…siting, sizing and installation timing…" (iii).
- main.tex:536, "…including siting, sizing, and staged capacity expansion" (iii).

**The highlights copies:**

- The standalone `highlights.tex` equals the five bullets of `manuscript_review/highlights_keywords_v2.tex` (comments
  stripped). It differs from the highlights in `main.tex`.
- `manuscript_submitted/highlights.tex` holds the old bi-level bullets.
- Only the clone's standalone file is listed, so the v2 copy is not counted twice.

### A8 (1) — out-of-budget points: budget arithmetic

Inputs from the case and cost files, read through the production readers:

- Budget: 1,000,000 € (`SRP1_ESS_Params.json`).
- Rated energy: at most 5 MWh per node.
- Duration: 2–4 h.
- Discount rate: 2 % (`SRP1.json`).
- Cost file `SRP1_ESS.xlsx`, sha256 `e17bd588…`: trajectory probabilities 0.35 / 0.55 / 0.10.

2025 costs per trajectory:

| Trajectory | Converter (€/MVA) | Energy (€/MWh) |
|---|---:|---:|
| Low | 214,171.33 | 212,132.83 |
| Medium | 267,528.72 | 264,982.36 |
| High | 342,165.62 | 338,908.87 |

The expected 2025 costs are 256,317.32 €/MVA and 253,877.68 €/MWh, with discount multiplier 1. These equal:

- the W2 table (`P515S45/investment_cost/investment_cost_results.json`) to within 1e-6 €;
- main.tex Table 1 for every row (k€, two decimals, and probabilities);
- the printed values 256.32 / 253.88.

The expected discounted cost is I(x) = Σ_nodes (1.02)^-(y-2025) Σ_c w_c (c^S S + c^E E). This is the formula of
`_check_candidate_first_stage_feasibility` and of the master budget row. The production check agrees with the closed form
on all 61 evaluations of tab:appE_evaluations.

| Plan (node 7, 2025) | Duration | I(x) (€) | Budget slack (€) | Admissible |
|---|---|---:|---:|---|
| 0.25/1 (reference) | 4 h | 317,957.01 | 682,042.99 | yes |
| 0.5/2 | 4 h | 635,914.02 | 364,085.98 | yes |
| 0.75/3 (budget corner, the s51 record) | 4 h | 953,871.03 | 46,128.97 | yes |
| **1/4** (evaluations 5; 38 at m = 2) | 4 h | **1,271,828.03** | **−271,828.03** | **no** |
| **1.25/5** (evaluations 6; 39 at m = 2) | 4 h | **1,589,785.04** | **−589,785.04** | **no** |
| **1.5/3** (evaluation 9) | 2 h | **1,146,109.02** | −146,109.02 | **no** |
| **2/4** (evaluation 10) | 2 h | **1,528,145.36** | −528,145.36 | **no** |
| **2.5/5** (evaluation 11) | 2 h | **1,910,181.70** | −910,181.70 | **no** |

This reproduces the audit 3.6 table to 0.1 k€. The s51 F2 ladder record already calls E = 4 and E = 5 "both
budget-infeasible, evaluated to trace the ladder".

Every 2030 or 2035 plan evaluated is admissible. The largest are the returned 2× plan (evaluation 40, 5: 0.25/0.5;
7: 1/3.5) and evaluations 41, 45 and 51, each at 992,268.11 €, which is the audit's ≈ 992.3 k€. The single-node lattice
in 2025 has 12 admissible plans and 23 over the budget, which matches l. 2797 and `fig_lattice_case`.

### A8 (1) — every place the out-of-budget points appear, and whether each is labelled

Summary: 34 asserted places. One is labelled, one is partially labelled, and 32 are unlabelled. A scan of `main.tex` for
the five plan tokens in 2025 rows found no unlisted line.

**Unlabelled places in the text and figures:**

- §4.2, l. 1424, "An affine fit … is obtained from the evaluated 2~h and 4~h configurations". The fit uses evaluations
  2–11, of which 5 are over the budget.
- §4.2, l. 1424, "The six certified evaluations give $e^{*}=183{,}080$". Points 5, 10 and 11 are over the budget.
- §4.2, l. 1425, "Including all ten evaluations …". The next sentence says "within the fitted range", not "within the
  admissible set".
- Figure 1 caption, l. 1431. The figure was inspected visually: E runs from 1 to 5 MWh for both 4 h and 2 h, with no
  budget line or legend entry.
- tab:res_breakeven caption, l. 1437, and its rows at l. 1446–1447.
- §4.2, l. 1467, "all five evaluated 4~h configurations … have positive investment margins", and "Each additional 1~MWh
  yields a determinate increase …".
- tab:res_energy_ladder_m2: caption at l. 1472, rows "4$^{f}$" at l. 1487 and "5$^{u}$" at l. 1488. The marks u and f
  are certification marks only.
- §5, l. 1803, the marginal break-even sentence. It is derived from the fit.

**Unlabelled places in Appendix E:**

- tab:appE_claims, (a) rows 4, 5, 8, 9 and 10 (l. 2628–2634).
- tab:appE_claims, (b) rows 26, 27, 30 and 31 (l. 2652–2657).
- tab:appE_evaluations, group header l. 2716 and rows 5, 6, 9, 10 and 11 (l. 2721–2727).
- tab:appE_evaluations, rows 38 and 39 (l. 2760–2761).
- tab:appE_gap, rows 39 and 6 (l. 2899, 2906).

**Unlabelled places in the letter:**

- l. 58, "affine in storage size with a slope below cost".
- l. 264, "affine in size on the evaluated ladders". Both rest on the ladders.

**Labelled:** l. 2921 (E.7), "the cache of the baseline search was seeded with the single-node ladders of the initial
design, including ladder points over the budget". This is the only sentence that says so. It names no points and is not
linked from §4.2, Figure 1 or the tables.

**Partial:** l. 2797 / `fig_lattice_case` (E.3). The legend "within limits, over budget" marks all 23 over-budget
single-node plans in 2025, including the five rungs. It labels the lattice, not the evaluated ladder, and nothing in §4.2
cross-references it.

**Scan for labelling phrases** (over budget, outside the admissible, inadmissible, exploratory, probe, budget-infeasible,
…), over the four clone files: it finds only l. 2921 and the two algorithm lines l. 506 and l. 509 ("inadmissible ones
dropped"). Figure 2 (the flexibility ladder) shows only the reference unit and is not an appearance. The abstract, both
highlights copies and the cover letter contain no over-budget point.

### A8 (2) — 13.9 %

Source values, from the frozen benchmark and equal to main.tex Table `tab:res_benchmark`:

| Quantity | Value (€) |
|---|---:|
| Benefit | 90,896,608.40 |
| Coordinated Q | 653,873,702.19 |
| Price-taker Q | 744,770,310.59 |

The benefit is 13.90 % of the coordinated cost (the price-taker's cost increase) and 12.20 % of the price-taker cost (the
saving).

| File : line | Text | Framing |
|---|---|---|
| highlights.tex:14 | "Coordination is worth 13.9\,\% of operating cost against static interface practice" | **Saving.** Presents a benefit of 13.9 % of operating cost; the denominator is not stated. |
| manuscript_review/highlights_keywords_v2.tex:12 | same bullet | **Saving.** Staging copy. |
| letter:53 | "the value of coordination against the best static no-reverse-flow interface arrangement, 90.9~M\euro{} (13.9\,\%)" | **Saving.** |
| letter:80 | "the abstract reports instead the value of coordination, 13.9\,\% of the discounted operating cost …" | **Saving**, and the pointer is false: the abstract reports 12.2 % (l. 146). |
| main.tex:1583 | "The price-taker and passive arrangements incur 13.9\% and 36.9\% higher discounted operating costs than coordinated operation" | Correct: a cost increase. |
| main.tex:1597 | tab:res_benchmark "90\,896.6 (13.9)", with the note "percentages are relative to the coordinated cost" | Correct. |

Searched with the pattern `13[.,]9`:

- the four clone files at `edd7a63`;
- `manuscript_review/highlights_keywords_v2.tex`;
- `manuscript_submitted/{highlights,main,cover_letter}.tex`.

The cover letter has no 13.9. It still claims, at l. 24, that coordinated operation with shared ESSs "can significantly
reduce operating costs", which is old text. The submitted files contain no 13.9.

### A8 (3) — the abstract's technology and cost-uncertainty sentences

**Technology** (main.tex:145): "This paper proposes a two-stage stochastic framework for shared battery ESS siting, sizing
and installation timing under investment-cost uncertainty."

- The abstract says "battery". It does not contain "lithium", "Li-ion", "LFP" or "chemistr", so no chemistry is named.
- The case does name the technology: l. 1298, "Shared ESSs are modelled as utility-scale lithium-ion batteries". The
  cost file's NREL breakdown starts with "Li-Ion Battery cabinet".

**Cost uncertainty** (same sentence) and **operational uncertainty** (l. 145): "…evaluates their operation under load,
renewable-generation and market-price uncertainty."

- The abstract says "investment-cost uncertainty". It does not contain "technology", "trajector", "future" or
  "converter".
- The case models three weighted trajectories, low / medium / high, with probabilities 35 / 55 / 10 %. They cover
  "future converter and battery energy-capacity costs" (l. 1219, Table 1) and enter only as expected coefficients
  (l. 690).
- The cost of capital is handled separately, by the discount-rate reweighting (§4.6).

**What the letter says about the abstract:**

- l. 75–76 says the investment-cost uncertainty is future technology cost and that "the abstract now says both". This is
  partly false: the abstract does not say "technology" or "future".
- l. 77 says "the abstract names the chemistry". This is false.

## Not confirmed

- **The rule behind the 18.0 k€ "reported numerical resolution"** of the 3 × 3 evaluations (M-45-1). It is not stated in
  §4.5, and it was not checked against any record.
- **"0.933 predicted from the mean-profile price spread"** (L-04). I searched only `main.tex` (strings `0.933` and `price
  spread`); other records were not searched.
- **Figure contents.** Figures 1 and `fig_lattice_case` were inspected visually from the PDFs at `edd7a63` and are
  hash-recorded. No PDF text extraction tool was available (no `pypdf`, no `pdftotext`), so the statement "no budget
  marking in Figure 1" rests on viewing it.
- **The coverage claim is scoped.** It holds only for the harness keyword pattern over the regions listed above. A
  claim with none of the keywords would not be caught. §4.6 and the rest of the Introduction were not scanned; the
  three supplement entries were picked by hand.
- **`main_latexdiff.tex` was not examined.**
- **Classification is judgement.** The listed / borderline / other assignments, and the exclusion reasons for the 75
  excluded fragments, are mine; they are all in the JSON (`A7.coverage.excluded`). Line numbers are those at `edd7a63`.
  The audit's anchors differ by up to two lines (for example, its 1203 is the search-restriction line, which matches).

## Unexpected findings

- **Five of the 10 points of the marginal break-even fit are over the budget,** including three 2 h rungs that audit 3.6
  does not list. See Blocked, item 1.
- **The frozen field name `benefit_relative` = 0.139012 is benefit / coordinated Q.** It is probably the source of the
  "13.9 % benefit" wording in the letter and the standalone highlights.
- **Variable naming in production.** `shared_ess_data.prob_market_scenarios` holds the investment-cost trajectory
  probabilities read from the `Scenarios` sheet of `SRP1_ESS.xlsx` (0.35 / 0.55 / 0.10). This is a naming issue only;
  the values were verified.
- **Algorithm 1, l. 503, restricts the incumbent** to "certified, budget-feasible points" of the cache. So the
  over-budget seeds of l. 2921 could not become incumbents.
