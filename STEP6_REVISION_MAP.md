# Step 6 — Manuscript revision map

**Expert's plan for the author, 2026-10-06.** Target: `manuscript/.../main.tex` (Overleaf clone `6cb4492`,
2,101 lines; `elsarticle`, *Journal of Energy Storage*), `highlights.tex`, `cover_letter.tex`, and a new
response letter. Line numbers refer to that clone. Sources: `P5_15_STEP6_PACKAGE.md`, the frozen tables
`frozen_step6_tables_v1_590088fe.json` (T1–T11), `paragraphs_v5.md`, `STEP4_DFO_METHOD.md`,
`PLANNER_BRIEF_2026-09-13.md` Addenda 49–67. Convention: blue text marks revised passages for the
reviewers, as the manuscript already does.

---

## A. What changes at the level of claims

The submitted manuscript claims: an optimal plan of 1.62 MVA / 3.24 MWh at node 7 in 2025 spending the
whole budget; Benders convergence in four iterations to a 0.50 % gap; 18.25 % operating-cost and 92.16 %
curtailment reductions vs uncoordinated operation in 2037 with voltage violations eliminated; and that
planning-horizon discretization alters siting and sizing. None of these survives. What replaces them:

| submitted claim | revised result (gross Q, settlement excluded; sources) |
|---|---|
| optimal plan 1.62 MVA / 3.24 MWh at node 7, budget exhausted in 2025 | **x = 0 is optimal under the baseline at both instances**; the smallest unit loses 64.4 k€ at SRP1 (4.93×) and 81.3 k€ at 3 × 3 (4.5×). T1 headline; T11 |
| Benders, 4 iterations, 0.50 % gap | **derivative-free master (MADS over the granular lattice)** with certified recourse evaluations; no optimality gap is claimed. `STEP4_DFO_METHOD.md`; paragraphs (ii) |
| 18.25 % / 92.16 % vs uncoordinated; voltage violations eliminated | **coordination beats the best static no-reverse-flow arrangement by 90.9 M€ (13.9 %)**, 77 % TN conventional energy (0.81 TWh of TN RES left curtailed by the price-taker arm), 23 % DN flexibility; without any interface rule the TN cannot accept the DN exchange in 1/12 (passive) and 8/12 (price-taker) blocks. The mechanism sentence is mandatory. T6; Addendum 58 |
| degradation affects valuation (qualitative) | **without ageing the unit is at break-even (−4.1 k€, within resolution); with the baseline calibration and the 0.70 floor it loses 31.6–73.7 k€ across every aged arm, determinately.** T8 |
| — (new) | **break-even energy cost 171–193 k€/MWh against 254 k€/MWh, margin ≥ 61 k€/MWh** (T3); **the unit pays at 1.75× and 2× the flexibility price and breaks even between 1.5 and 1.75** (T1 H rows, T10); **value affine in size** (T1 I/J rows); **year ladder convention-dependent** (T4); **discount 0–8 %: never pays** (T7); **multi-scenario ratio 0.909–0.934 vs 0.933 predicted** (T11) |
| discretization sensitivity (5-year vs 1-year) | **not run — removed**; the 5-year block discretization is stated as a modelling choice and a limitation |
| calendar ageing 1 %/yr with 0.5 % / 2 % sensitivities (text, never run) | **calendar retention 0.985/yr in the baseline (C2_calfade); C2 (no calendar) is the sensitivity**; the 0.5 % / 2 % sentence is removed. T8 |

The story the revised paper tells: a degradation-aware planning tool with certified evaluations; applied
to this instance, storage does not pay under the baseline **because of degradation**, the tool identifies
the conditions under which it does (flexibility price, no ageing), and coordination itself is worth 13.9 %
against today's static practice. This is stronger than the submitted story and it is true.

---

## B. Section-by-section map of `main.tex`

### Front matter (lines 60–184)

- **Title** (l. 92): keep.
- **Abstract** (l. 139–148): rewrite. Keep sentences 1–3 (context, proposal, two-stage formulation in
  blue). Replace the Benders sentence by: a derivative-free search over a granular siting–sizing–timing
  lattice, each candidate evaluated by a certified consensus-ADMM recourse. Keep the instance sentence
  (108-bus system) but fix the horizon to what was run: three representative years (2025, 2030, 2035),
  each standing for a five-year block, four representative days; the multi-scenario instance with
  3 × 3 scenarios. Replace the 18.25 %/92.16 % sentence with: coordination 13.9 % against the best static
  no-reverse-flow arrangement; under the baseline no investment is justified (the smallest unit loses
  64 k€, determinately) and the result is attributable to degradation — without ageing the unit breaks
  even; it pays once flexibility is priced at 1.75× the base; break-even energy cost 171–193 k€/MWh vs
  254. Drop the discretization sentence.
- **Graphical abstract** (l. 154, `framework_v2.pdf`): the figure shows Benders cuts — redraw (Worker can
  produce a vector figure from a spec; the author decides the layout).
- **Highlights** (l. 157–163) and `highlights.tex`: rewrite five bullets: two-stage stochastic framework
  (keep); degradation explicitly linked to coordinated multi-year operation (keep); *derivative-free
  planning search with certified ADMM recourse evaluations and stated resolution* (new, replaces
  Benders); *coordination is worth 13.9 % against static interface limits; the mechanism is the TN
  marginal value vs the wholesale price* (new); *degradation decides the investment: break-even without
  ageing, determinately unprofitable with it* (new, replaces the discretization bullet).
- **Keywords** (l. 166–173): replace "Benders' decomposition" by "Derivative-free optimization" (or
  "Mesh adaptive direct search"); keep ADMM.
- **Nomenclature** (l. 185–266): remove `L^B` (Benders iterations, l. 202), `α^Down` (l. 216) and every
  cut-related symbol; add the lattice symbols (unit sizes 0.25 MVA / 0.5 MWh, duration bounds 2–4 h),
  τ, k₀, k\*, and the settling-rule symbols used in §2 and Appendix A. The Worker can produce the
  symbol audit (every symbol in text appears in the table and vice versa) — zero-solve.

### 1. Introduction (l. 268–314)

- Keep l. 268–296 (motivation, literature, four requirements; the blue uncertainty paragraph).
- Contributions (l. 300–305): bullet 3 is Benders → rewrite: "A nested solution strategy in which a
  derivative-free search over the investment lattice is driven by certified recourse evaluations,
  each a consensus-ADMM coordination of TSO and DSO operation with an explicit stopping and
  certification rule; the method claims no optimality certificate and states the resolution of every
  reported difference." Bullet 4 → "An extensive case study on an integrated 108-bus system that
  quantifies the value of coordination against static interface practice, the break-even conditions for
  shared storage, and the role of degradation in the investment decision, with every reported number
  carrying a stated resolution." Remove the discretization clause.
- Add one sentence (blue) after the contributions acknowledging what the revision changed: the cut-based
  master of the submitted version was replaced after the reviewers' comments; the results in this
  version are produced by the revised method and differ from the submitted ones in kind (the response
  letter says why in full).

### 2. Shared ESS planning framework (l. 315–910)

- **2 intro + 2.1 architecture** (l. 315–543, mostly blue): keep the two-stage framing. Every mention of
  "Benders-type procedure" / "cuts" → the derivative-free search. Algorithm 1 (l. 398–430,
  "Benders-type shared-ESS planning with ADMM-based operational recourse") is **replaced** by the MADS
  algorithm as implemented: initial point, poll on the granular lattice (0.25 MVA / 0.5 MWh units,
  duration 2–4 h, investment year), frame size rule (double/halve), completion rule (a legal search
  step), evaluation cache keyed on the candidate, budget of evaluations. Source:
  `STEP4_DFO_METHOD.md` §§1–5, 9. The Planner confirms each algorithm line against `p56a_oracle` and
  the campaign launcher before the text is final (equation/algorithm audit, zero-solve).
- **2.2 Master problem** (l. 544–723): keep objective (l. 550–578) minus `α^(l)` and the underestimator
  (l. 577); keep 2.2.2–2.2.5 (rated capacities, maximum installable capacity, E/P ratio, expenditure
  limit) with the lattice stated: capacities are integer multiples of the unit sizes; E/P ∈ [2, 4] h;
  ≤ 5 MWh per node; budget 1 M€. 2.2.6 coupling (l. 656–683): keep, reworded for an evaluation
  oracle. **2.2.7 Benders' cuts (l. 684–723): delete entirely**; replace by "2.2.7 Recourse evaluation
  and certification" — the certification paragraph of `paragraphs_v5.md` (ii), which is already
  manuscript-ready, plus one paragraph on what a certified value means for the master (two candidates
  are compared only when their difference exceeds max(3 × larger band, 2τ); the search treats smaller
  differences as ties).
- **2.3 Subproblem** (l. 724–904): keep the ESSO agent (Option A was retained). Equations (21)–(28) as
  printed describe apparent-energy throughput (R2.10/R3.4) — rewrite to the active cell-side energy
  `η_ch·p_ch + p_dch/η_dch`, the converter capability circle, the SoC recursion with daily closure
  (R2.11/R2.12/R3.3), the SoH chain as implemented, calendar retention φ_cal = 0.985 per year
  (R3.5), the end-of-life floor 0.70 as the baseline (the 0.50 row is a sensitivity), and the fact that
  the ESSO objective carries no economic term (ε-throughput regularizer only; salvage reporting-only).
  **Every equation is checked against code by the Worker before the text is final** (this is where a
  referee re-derives). The red note (l. 905–909) is removed; its content (expected schedule, scenario
  deviations with a priced premium, α = 0.50) becomes the description of the row-18 non-anticipativity
  term in the multi-scenario instance.

### 3. Case study (l. 911–1067)

- Instance as run: IEEE 9-bus TN with three IEEE 33-bus ADNs at nodes 5, 7, 9 (keep the diagrams
  l. 927–933); **three representative years 2025, 2030, 2035, each a five-year block (15-year horizon),
  four representative days**; SRP1 uses one scenario; the multi-scenario instance uses 3 market × 3
  operation scenarios (prefix draw [1,2,3] × [1,2,3]). Replace the 2025/2028/…/2037 columns everywhere
  (Tables l. 978–1050 are RES capacity by year — regenerate for 2025/2030/2035 from the case files;
  Worker, zero-solve).
- **3.1 Investment costs** (l. 945–969): corrected cost file (`SRP1_ESS.xlsx` at `7ce1d1ab`): ≈ 254 k€/MWh
  and ≈ 256 k€/MVA; the three cost-trajectory scenarios of the submitted version are not what was run —
  state the single expected trajectory used, or the probability weights if the file carries them
  (Planner confirms from the file).
- **3.2 Market data** (l. 970–973): keep; add the flexibility price ladder (multipliers 1.5, 1.75, 2 of the
  base flexibility price) as a sensitivity axis.
- **3.3–3.4 Networks** (l. 974–1067): keep structure; **l. 1062 calendar ageing sentence** → 0.985/yr
  retention in the baseline (1.5 %/yr), with C2 (no calendar fade) as the sensitivity; remove the
  0.5 %/2 % cases. Add the ageing calibration table: C2 (10,000 cycles at 0.80 DoD to 0.80 retention,
  k = 35,851), C3_unit (EoL retention 0.50), C4 (EVE MB31: 8,000 cycles to 0.70, k = 22,430),
  C3_midblock (SoH evaluation point), floor 0.70 (0.50 as a row). Source: T8 and the extension spec
  v3 arm definitions.
- Add **3.5 Evaluation and certification settings**: ρ initial values, residual balancing, Boyd
  tolerances, AA with memory 5, tail tolerances, τ, the uncoordinated arm definitions (passive and
  price-taker under no-reverse-flow, TSO arm with fixed interface P/Q), and the reproducibility note
  (iv) of `paragraphs_v5.md`.

### 4. Results (l. 1068–1390) — replace wholesale

Proposed structure (author decides main/supplementary; the expert's split is in `export/README.md`):

- **4.1 Planning result at the baseline.** x = 0 at both instances; the value ladder in E and S (affine;
  slope b + c/4); T1 headline, T1 B/I/J rows. Figure: break-even fit with the uncertified points as
  intervals (certified-only and banded fits).
- **4.2 Break-even conditions.** Energy cost 171–193 k€/MWh vs 254 (T3); flexibility price: pays at
  1.75× and 2×, break-even between 1.5 and 1.75 (T1 H rows, T10); the F2 two-node plan at 2× (T1 L rows:
  all 12 neighbour differences positive, 7 determinate, 5 within the bar) as the demonstration that the
  planning search finds a plan when one pays.
- **4.3 Degradation decides.** T8 arms with the verbatim sentence; floor-binding year per arm; the
  soh_min 0.50 row (T10): the floor is not what makes storage uneconomic; ε_AE restated (1.04–1.85) with
  the late-life-tail reading flagged as a hypothesis. Figure: ageing arms.
- **4.4 Value of coordination.** T6 with **both** arms, the sweep, the 13.9 % and its decomposition, the
  mechanism sentence verbatim, the 4 reverse-flow hours caveat; the two non-H1 blocks are positive. The
  old "voltage violations eliminated" claim is replaced by the infeasibility sweep.
- **4.5 Multi-scenario instance.** 3 × 3: x = 0 optimal; R ∈ [0.909, 0.934] against 0.933 predicted from
  the mean-profile spread; the row-18 commitment premium (C(α) curve from the 2 × 2, Addendum 44) if
  the author wants it. T11.
- **4.6 Sensitivities.** Year ladder gross and net with the convention sentence (T4); discount 0/2/5/8 %
  (T7); salvage: 0 sign changes in 60 claims (W153 totals).
- **4.7 Certification statistics and computational performance.** Replace "four Benders iterations,
  0.50 % gap, 30,560 s" (l. 1131–1142) by: 42 evaluations under the rule, 32 certified (24 oscillatory,
  8 monotone), the 10 uncertified by cause, median k\* 174, median 65 cycles after the first residual
  pass, median range/τ 0.86, ten certificates within 5 % of τ; wall time median 25.6 s per cycle; number
  of recourse evaluations in the planning search (from the campaign records); reproducibility: bitwise
  72/72, 3/3 and every gated cell. Figure: Q versus cycle for one certified cell with the window marked.
- **Delete**: 4.5 "Impact of planning horizon discretization" (l. 1282–1370) and its two tables; the
  operational-planning subsections built on the old recourse (l. 1144–1281: Table 6 summary, voltage
  profile figure, operational-cost and RES figures) — regenerate only what the new campaign supports
  (the benchmark's curtailment and cost decomposition from `report_v3.json`), and drop the rest.
- **Key insights** (l. 1371–1390): rewrite to four: (i) under the baseline the tool returns no
  investment, and the reason is degradation; (ii) the break-even conditions (energy cost, flexibility
  price) are quantified with stated resolution; (iii) coordination against static interface practice is
  worth 13.9 %, through the TN marginal value vs the wholesale price; (iv) certified recourse evaluation
  is what makes small planning differences meaningful — the stopping rule, not the residual test, decides
  what can be claimed.

### 5. Conclusions (l. 1391–1407)

Rewrite paragraphs 2–3 to the four insights; keep paragraph 1 (method summary) with the derivative-free
search named; future work: the post-revision cleanup items that are scientific (an economic tie-breaker in
the ESSO to remove the dual dead zone; the 5 × 5 instance; re-optimised plans per ageing arm; a stressed TN
variant where voltage support matters), not the engineering ones.

### Appendices

- **A. TSO–DSO coordinated operational planning** (l. 1409–1611): keep the three-agent ADMM; Algorithm
  (l. 1500) checked against code; ADMM implementation (l. 1504–1611) updated for: consensus channels and
  their scaling (σ, interface rating; S_ref and D5 on the ESS channel), ρ initial values and residual
  balancing, the ESS two-phase schedule and freeze, AA (type-II, memory 5, cleared at ρ change,
  safeguarded, off inside tolerance), the tight tail, the row-18 term for the multi-scenario instance.
  The settling criterion lives in §2.2.7, not here.
- **B. Market data, C. TN, D. DNs** (l. 1612–1972): keep; regenerate year-dependent tables for
  2025/2030/2035 where they show 2028/…/2037.
- **E. Results** (l. 1973–2086): replace the old breakdown table by the supplementary set: T2
  (certificates per cell), T5 (Phase B), T1 C/G/L rows, T9 (dead-zone table), the certification
  statistics, and the prediction scorecard (paragraphs (v)) as a supplementary table with its one-line
  introduction.
- **Declaration of AI use** (l. 2087): keep; it should state the use accurately (the author decides
  the wording; the record of this revision is the brief).

---

## C. The response letter (new file, `response_to_reviewers.tex`)

Skeleton: one block per reviewer item in the order of the reviewers' letters, each with *Comment*
(quoted verbatim from the reviewers' documents — add them to the repo), *Response*, *Changes* (section
and line of the revised manuscript; blue text). The reviewer map in `paragraphs_v5.md` (vi) gives the
answers and sources for every item; the editorial ones (R2.8 references, R2.11–R2.12 equations, R3.7–R3.8
wording) are the author's.

Two paragraphs the letter needs beyond the item-by-item answers, both honest and both in the authors'
favour: (1) **why the results changed in kind.** The submitted results were produced by a cut-based
master whose cuts the reviewers correctly identified as invalid for a nonconvex recourse, on a recourse
whose transfer-payment term was later retired; the revision replaced the master by a derivative-free
search, corrected the cost file, adopted a datasheet ageing calibration with calendar fade and an
end-of-life floor, and introduced a certification rule for every evaluation. Under the revised method the
baseline instance supports no investment; the paper now reports the conditions under which it does. The
18.25 % figure is replaced by a differently defined 13.9 % — different recourse, different definition —
and the letter says so in those words. (2) **what was not done and why**: R1.2(v) attribution (joint
schedule, no unique attribution); re-optimised plans per ageing arm (a sentence suffices: the smallest
lattice point loses determinately under every aged arm and the value is affine in size, so x = 0 is the
plan under each arm given that affinity); the 5 × 5 instance (memory; stated as future work).

---

## D. Figures (Planner/Worker generate from the frozen JSON, deterministic)

1. Break-even fit: value − I vs E at node 7, 4 h, certified points as dots, uncertified as intervals;
   both fits; the energy-cost line. (T3)
2. Flexibility ladder: value − I of the unit vs multiplier (1, 1.5, 1.75, 2) with bars. (T1 H, T10)
3. Ageing arms: value − I per arm with thresholds and the floor-binding year. (T8)
4. Q versus cycle for one certified cell (e.g. the settled x = 0 reference), with k₀, the window, the
   turning points and k\* marked; a second panel for one gap-refused cell. (records)
5. Benchmark: the three arrangements' Q and the decomposition of the benefit. (T6)
6. Redrawn framework figure (graphical abstract) without cuts.

---

## E. Process

- **Base for `latexdiff`:** the Overleaf Git bridge squashed the history (one commit); recover the
  submitted version from Overleaf's History panel (download the labelled version) or from the submission
  system's source zip, and store it in the repo as `manuscript_submitted/`. The expert can run
  `latexdiff` given both sources.
- **Blue text** marks revised passages; red notes are removed before submission.
- **Figure checks:** every number in the manuscript traced to a frozen-table cell or a named record (the
  Planner's W163 checker extended to `main.tex`); run before each review round closes.
- **Rounds:** (1) response skeleton + §2 methods; (2) §3–4 case study and results with the figures;
  (3) front matter, §1, §5, appendices; (4) whole-paper read for coherence; (5) `latexdiff` PDF and the
  letter's final pass. Each round: the author pushes to Overleaf, the Planner pulls and checks, the
  expert reviews the `.tex` and returns numbered edits.
- **Audits before the text is final (zero-solve Worker tasks):** equation-vs-code audit for §2.3 and
  Appendix A; nomenclature audit; year-dependent tables regenerated for 2025/2030/2035.
