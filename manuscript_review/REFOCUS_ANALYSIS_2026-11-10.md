# Refocus analysis — critical assessment and four-week action plan

Expert (Claude, Cowork), 2026-11-10. For the author's decision; nothing here is authorized until recorded in
Addendum 89 of `PLANNER_BRIEF_2026-09-13.md`.

**Basis read for this analysis.** `HANDOVER.md` (2026-10-15); Addenda 87–88 and the tail of the change log; `TASKS.md`
(realignment order, state at commit `5696706f`, 2026-11-01); `manuscript_review/manuscript_critical_review_2026-10-25.md`;
`REVISION_REALIGNMENT_ANALYSIS_2026-10-30.md` (v2); the three reviewers' letters (`Reviewers Comments.docx`);
`response_to_reviewers_v10.tex`; `EXPERT_REQUEST_2026-10-31b.md` (R8); the P5.16 D0 reports (T0.1a–c, Clarabel, IPOPT
interior); `STEP4_DFO_METHOD.md`; the submitted `main.tex`; and the current `main.tex`.

**Overleaf state verified.** Overleaf HEAD `a87e195`; `main.tex` md5 `39f6fd86f90b0fbe7c3cc335b1bb793c`, byte-identical to
`manuscript_review/main_edd7a63_round8.tex` (round 8). Repo branch `feature/derivative-free-planning` at `5696706f`.

**Author's answers that frame this plan (2026-11-10):** hard deadline in about four weeks; the Addendum 87 direction is
open to reconsideration; T0.2 is still blocked (no Gurobi key on the 24 GB machine; not run on the Studio).

---

## 1. Decisions needed

1. **Which paper is submitted in four weeks.**
   - (a) **The current manuscript (`a87e195`) plus round 9 plus a bounded set of additions (§6).** Cost: zero-solve
     editing plus ≤ 70 h of Studio compute, all on the frozen convention, all with ready launchers. Risk: a paper with a
     negative main result and no optimality bound; mitigated by the honest framing Reviewer 3 explicitly offered.
   - (b) **The Addendum 87 realignment (LB/UB, salvage in F, mid-block, 3-year blocks, general lattice).** Cost: a new
     instance whose every storage cell, every x = 0 cell, and both benchmark arrangements must be re-evaluated; an LB
     oracle, dual extraction and a lattice MILP master that do not exist yet; Table 1 extended to 2028/2031/2034/2037
     cost columns (author data); then a results section rewritten from scratch. Risk: it cannot be finished *and
     verified* in four weeks under this project's cadence (§3.2), so (b) chosen now means asking for an extension that
     may not be granted, and submitting with a half-landed method.
   - **Recommendation: (a), with (b) continued as the next paper** (and, if D0's T0.2 happens to run on the Studio
     in week 1, one tightness paragraph from it — §6, item A6 — without changing any convention).
2. **Whether the two solve-tasks already built and held (W184 NRF arm, W185 penalty pilot) are run under the frozen
   convention.** They were authorized conditionally in Addenda 85–86 and held by R6 only because the convention was
   expected to change. Under decision 1(a) the hold no longer has a reason. Recommendation: run both (≈ 25 h), they
   are the two measurements that most strengthen the weakest claims (§4, G3 and G4).
3. **Whether to spend ≈ 20–25 h on the 15-point no-ageing unit-neighbourhood screen** (R3.6, planning part).
   Recommendation: yes if W184/W185 have landed by the end of week 2; otherwise keep the narrowed claim.
4. **R8 (G-CC-3 slack bar).** Belongs to track (b). Ruling for the record: option (a) of the request (`PENALTY_ESSO_SLACK`
   = 1e4 on the new instance via the ESS-params file, default 1e3 unchanged, one gate re-run), not (b) or (c): the bar's
   unit is the slack's unit, and a bar re-read to pass is no bar. No action in the next four weeks.
5. **Numbers-freeze date.** Recommendation: T − 10 days. Any measurement not landed and verified by then is left out and
   the corresponding claim stays narrowed, as it is now.

**Blocked on the author:** the choice above; the Studio for ≈ 70 h of attached runs over weeks 1–3; the Gurobi key or a
Studio run for T0.2 (optional, A6).

---

## 2. Where the revision stands (facts)

- Overleaf `a87e195` is the round-8 manuscript: 76 pages; main body ≈ 6.3 k words of prose, Appendix A–D ≈ 3.1 k,
  Appendix E ≈ 2.0 k (my source count, comments/equations/tables excluded). Letter v10, highlights v3, cover letter v2
  are byte-identical to the deliverables. Round 9 (seven `Appendix~\ref` prefixes, E.23/E.24 marks, "break-even" wording
  at two places and in the cover letter, the dropped l. 286 bullet, the E6 figure, the bracketed cover-letter paragraph,
  the marked copy) is queued and zero-solve.
- The method is mathematically defensible as written: a two-stage program with one implementable plan; a granular
  MADS on a lattice with a mesh-local termination statement and no optimality claim; a consensus ADMM recourse with a
  stated residual test, a stated settling rule, and a stated resolution for every reported difference; a degradation
  chain in active cell-side energy with calendar ageing, a floor and datasheet calibrations.
- The evidence base is 61 evaluations (42 under the rule, 32 certified) on SRP1 (3 years × 4 days × 1 scenario), a
  3 × 3 fixed-plan evaluation, and two static benchmark arrangements with three starts each.
- The realignment track has: D0 built and gated to T0.1c (the relaxation is of production's problem; identity to
  1e-13); one IPOPT interior solve on one of twelve groups at x = 0 giving a gap of 5.84 k€ = 9.1e-5, first-order
  resolution about ±1 k€ (provisional H1 *on that group only*); Clarabel stalls unexplained; T0.2 (132 solves) blocked
  on a licence; the convention change C1/C3/C4/C5 built and gated except the R8 stop; T-perf Part A done, Part B
  waiting on a Studio run. Nothing of track (b) touches the manuscript yet.

---

## 3. Critical assessment of the two tracks

### 3.1 The current manuscript — what is right, what is weak

Right: every reviewer's mathematical objection to the submitted version (invalid cuts, undefined sensitivities,
non-implementable averaged plans, apparent-power degradation, missing SoC physics, missing calendar ageing,
unspecified ADMM termination) is answered by removing the invalid object rather than defending it, and the paper says
what it did. The cost-file error is disclosed (letter v10 l. 287). The determinacy rule makes the storage comparisons
honest at the precision the recourse actually has (σ_Q ≈ 10–18 k€ against margins of 46–135 k€).

Weak, in order of what a second-round reviewer will notice:

1. **The paper's main planning result is a corner.** Q is affine in x on SRP1 (227.7 k€/MWh, 51.7 k€/MVA, additive
   across nodes — Phase A) so the optimum of a budget-constrained affine problem is at x = 0 or at the budget, and the
   search finds exactly that. The MADS machinery is correct but does nothing a ladder would not. The text spends its
   weight on the search and the acceptance rule rather than on the physics and economics that make Q affine. One
   sentence saying so (and why that makes the ladders the natural presentation) would do more for the paper than
   another acceptance paragraph.
2. **R2.5/R2.6 are answered descriptively, not with a measurement.** The letter states the penalty schedule and that
   "a different schedule could in principle reach a different local solution"; no evaluation under a different schedule
   or start exists for the decisive pairs (x = 0 vs the unit at 1× and at 2×). The 25 Oct review and the realignment
   analysis both name this as the single most informative missing check. W185 (8 cells) is exactly it, and it is built.
3. **The 12.2 % headline is confounded by the interface restriction** (static arrangements prohibit DN→TN flow;
   coordination uses it in four interface-hours). The paper states the confound (§4.4, abstract) but cannot size it.
   W184 (coordinated with NRF imposed on the consensus exchange, 1 cell) sizes it, and it is built.
4. **No first-order characterization of a certified point.** Appendix A gives the residual test and the settling rule
   but never says what a fixed point *is*. The correct, cheap statement (realignment §6.4(iii)): at a fixed point of the
   sweep with every block at a KKT point of its augmented subproblem, the block points and scaled duals satisfy the KKT
   conditions of the coupled problem (proximal terms vanish; first-order conditions are additive), so a certified point
   is an approximate first-order stationary, AC-feasible-within-tolerance point of the coupled recourse. That is the
   "local guarantee" Reviewer 3 asked to be characterized. It is currently absent (no "stationary"/"KKT" in the text).
5. **Operational outcomes are gone.** The submitted paper had voltage, curtailment and RES figures; Reviewers 1 and 3
   engaged with them (R3.2 asks for a better explanation of Fig. 15, not its removal). The current results section has
   no physical quantity except two curtailment columns. The persisted certified cells (x = 0, unit, benchmark arms)
   contain the dispatch; the E6 figure (declared day 2025 Spring, 40,181 € = 15.8 %) is already exported and queued.
6. **Length and ledger.** Appendix E prints every comparison and every evaluation (five tables, 58 + 61 rows). The 25 Oct
   review's recommendation (compact E + machine-readable supplement) was not applied in round 8.
7. **Two equation families are undefined at zero capacity** (G1 below): the normalised complementarity divides by
   S^Av, the cohort allocation by ΣE^Rated, the cycling exponent by E^Rated,Unit — and x = 0 is the paper's central
   case. Flagged on 25 Oct, still present at `a87e195`.
8. **Scope narrowing is honest but concentrated:** single-scenario search, one common year, SRP1 not paper scale,
   3 × 3 uncertified, no degradation-neutral *search*, salvage post-evaluated. Each is stated. Together they are the
   likely ground of a second-round objection; the plan below widens two of them (R2.6 and R3.6) at bounded cost.

### 3.2 The realignment — merit, cost and why it cannot land in four weeks

Merit. The LB/UB structure of Addendum 87 is the right next piece of mathematics: a convex relaxation of the
*production* recourse (rank relaxation to rotated SOCs, SoH at the calendar-only value, exact coupling, identical
objective convention) gives Q_LB(x) ≤ Q_AC(x) ≤ Q_cert(x), is convex in x, separates by block when the degradation chain
is dropped, and so restores all 25 scenarios and staged years on the cheap side. Three things about it deserve to be
said plainly, because they decide how it should be used.

- **At x ≠ 0 the LB's SoH relaxation removes exactly the mechanism the paper is about.** With SoH at its calendar-only
  value, Q_LB(unit) is essentially the no-ageing (or calendar-only) recourse. The unit's no-ageing margin is −4.1 k€
  (within resolution) against −64.4 k€ aged, so the gap at the unit is ≈ 60 k€ ≈ 9e-5 of Q, the *same order as the
  storage stakes*. The relaxation gap at a storage plan is therefore not a numerical looseness but the degradation
  value itself. That is not a defect; it is the useful reading: **the LB master is a degradation-neutral planner and
  the UB is the degradation-aware evaluation, so the LB/UB pair is the controlled comparison Reviewer 3 asked for
  (R3.6), measured rather than argued.** The next paper should present it that way, with the gap attributed
  (rank/circulation vs degradation) by evaluating Q_cert under the no-ageing calibration at the same x.
- **Its certified-negative claim at baseline is a theorem about the lattice, not about storage.** F(0) − LB bounds the
  value of any lattice plan; with Q affine and the unit at −4.1 k€ unaged, the bound will be of order the resolution.
  Valuable, but it does not change the paper's conclusion; it certifies it.
- **D0 has measured one group of twelve at x = 0.** H1 is provisional on that group. The predictions recorded for T0.2
  (DSO blocks exact to ≤ 1e-5, circulation ≤ 0.05, overall gap ≤ 1e-2) are untested on the unit cell and on the other
  eleven groups; Clarabel's stall is unexplained; Gurobi's QCP duals failed once before (P5.5-C). The LB side is a
  research task with an honest chance of H2 or H3.

Cost against the clock. Decision 3 (mid-block + 3-year blocks) and decision 2 (salvage in F, realised on the UB side)
define a *new instance* (`SRP1_5x3y_1x1`, 80 + 3 blocks). Every number in §4 of the manuscript is on the old instance.
A minimal campaign to repopulate §4 is: x = 0; the reference unit; the 4 h ladder (5); the flexibility ladder (4);
the ageing arms (6); the year ladder (3); two benchmark arrangements × 3 starts; ≈ 25–30 certified evaluations. The
current campaign cost 61.4 h for 42 evaluations on 12 blocks; the new instance has 20 blocks, so ≈ 2.4 h per
evaluation and ≈ 65–75 h serial before any LB work, any paper-scale UB, or any failure — and this programme's
record is that every campaign surfaces something (the B+ refusal, the two-cycle gate drift, the G14 booleans, the
Clarabel stall). Add the LB oracle, dual extraction with margin, the lattice MILP, the cache validation gate, the
cost columns for 2028–2037, and a results section rewritten under a convention in which salvage is inside F (so every
"gross/net" table changes meaning). At this project's cadence (one Planner turn per author prompt, one expert ruling
per report) that is eight to twelve weeks, not four. Starting it now and stopping at the deadline leaves the manuscript
in the worst state: a new convention half-populated, the old numbers withdrawn.

What to keep from it now (all zero-solve or already paid for): the D0 construction gates and the IPOPT reading as a
*tightness remark* if T0.2 runs (A6); the T-perf A-zero finding that economic stationarity is tight (a one-line
reassurance in §4.7 that the 1e-3 scaled tolerance is 1e-4 €/MWh against economic gradients of 326–15,809); the
Advisor's and expert's rulings (R1–R4, F4) as the recorded design of the next paper. Everything else stays on the
branch for paper 2.

### 3.3 Recommendation

Submit (a). Spend the four weeks on the two measurements that answer the two reviewer requests currently answered only
in words (R2.6 penalty sensitivity; the NRF confound of the headline), on the first-order characterization of a certified
point, on restoring one operational figure, on the zero-capacity conventions, and on cutting Appendix E — and, if the
Studio allows it, on the 15-point no-ageing screen that converts R3.6's planning part from "not met" to "met at the
unit-neighbourhood level". Freeze numbers at T − 10 days. Record the realignment as the next paper in the cover letter's
future-work sentence, not in the manuscript.

---

## 4. Gaps and mistakes in `a87e195` (numbered; line anchors are in the current `main.tex`)

Severity: **H** a reviewer will object; **M** weakens a claim; **L** polish. Cost: Z zero-solve, S solves.

| # | Where | Finding | Sev. | Cost |
|---|---|---|---|---|
| G1 | l. 950 (eq. network complementarity), l. 989 (cohort allocation), l. 1023 (cycling exponent) | Division by S^Av, ΣE^Rated, E^Rated,Unit — undefined at x = 0 and for any inactive cohort, the paper's central case. Fix: write the complementarity as P^Ch·P^Dch ≤ ε^C (S^Av)² (equivalent for S^Av > 0, vacuous at 0 with the power bounds); state that allocation and the exponent are defined for active cohorts with E^Rated,Unit > 0 and that inactive storage rows are deactivated (which is what the code does). | H | Z |
| G2 | Appendix A.4/A.5; §2.2.7 | No statement of what a certified point *is*. Add one paragraph: KKT of the coupled recourse at a fixed point of the sweep (block KKT + vanishing proximal terms + consensus) ⇒ approximate first-order stationary, AC-feasible within the consensus tolerances; no global claim. Cite Boyd for the residual test only; do not cite Hong–Luo–Razaviyayn or Wang–Yin–Zeng as a guarantee (they do not cover sequential nonconvex *constrained* blocks). Letter R2.5/R3.2 point to it. | H | Z |
| G3 | Abstract l. 146; §4.4 l. 1584–1632; Table 11 | 12.2 % is measured against arrangements that prohibit reverse flow while coordination uses it. W184 (coordinated, NRF on the consensus exchange, 1 cell) gives the de-confounded number. Report both: "coordination saves X % at equal interface permissions and 12.2 % when reverse flow is also enabled". If W184 is not run, keep the present wording. | H | S (≈ 3 h) |
| G4 | §4.7, §4.8, letter R2.6 | Penalty-schedule sensitivity answered without a measurement. W185 (8 cells: the two decisive pairs under perturbed ρ₀ and a perturbed start) gives it: report the spread of Q and of the *difference* across arms against the determinacy threshold. This is the check the 25 Oct review ranked first. | H | S (≈ 20 h) |
| G5 | §4.3, §4.8, letter R3.6 | Degradation-neutral *planning* not done; the fixed-plan no-ageing margin is −4.1 k€ (within resolution), so the sign of the unit is unresolved. The 15-point screen (x = 0 + 14 unit neighbours, no ageing, 1×, production exit) settles whether the no-ageing search leaves x = 0 at the unit-neighbourhood level — either outcome is a result: "without ageing the baseline cannot discriminate storage from no storage at this resolution; with ageing, x = 0 is determinate". | M | S (≈ 20–25 h) |
| G6 | §4 (no operational figure) | Restore one physical illustration from persisted certified cells: the queued E6 figure (declared day, coordinated vs price-taker: TN generation / DN flexibility / curtailment, panel (b) in €) and, if cheap, the unit's daily SoC and the SoH trajectory per calibration (the Table 7 "binding year" column as a curve). Answers R3.2's second item on its own terms. | M | Z |
| G7 | Appendix E (l. 2568–2946) | Reduce to E.1 (salvage), one comparisons table restricted to the comparisons the main text cites (Table 7, 9, 10, 12–14 rows), E.6 (statistics, reproducibility). Move E.21–E.24 to a machine-readable supplement generated from `export_W160/*.csv` (the CSVs already exist; `gen_appE.py` produces the tables). Letter pointers to E.20 rows must be re-targeted. | M | Z |
| G8 | §2.2.7, A.5, throughout | "Certified/certificate" is defined narrowly and used consistently, but the noun "certificate" invites the reading Reviewer 3 objected to. Keep the adjective with its definition at first use; replace the noun "certificate" by "acceptance record" where it occurs in prose (not in table legends). If the next paper carries a true bound, the word must be free for it. | L | Z |
| G9 | l. 732 ("discounted to y₀ using the representative year y") | Block-level discounting at the representative year is an approximation to annual discounting within the block (a 5-year block at 2 % differs from its annuity by ≈ 4 % of the block's discount factor). State it as the approximation it is; it is the same on every arm so no comparison changes. | L | Z |
| G10 | l. 724 (Q includes slack penalties) | Q is reported as an economic cost but includes penalties on closure and feasibility slacks. The W169 inventory showed zero closure slacks on 45 of 51 certified points; the T-perf A-zero reading has every penalised slack at its bound. Have the Planner verify, zero-solve, that penalised slacks are zero at *every* evaluation behind a main-text table, and say so in one sentence; otherwise report the penalty content per row. | M | Z |
| G11 | Algorithm 1 l. 513 vs §2.2.7 l. 743 | Two acceptance thresholds coexist (search: max{bar_x + bar_inc, σ_Q}; reporting: max{3b, 2τ}). Both are stated, but the reader is not told why they differ (the search uses the cheap production exit; reported comparisons are re-evaluated under the rule). One sentence after Algorithm 1. | L | Z |
| G12 | §3.6 l. 1347, Table 10, Fig. 5 | Out-of-budget probes are now labelled $^b$ and explained — done in round 8. Check the letter and highlights say "marginal" break-even, not "break-even" (round-9 item). | L | Z |
| G13 | Table 1 (3 cost columns) | Fine for the frozen instance. Note for paper 2: 3-year blocks need 2028/2031/2034/2037 columns (F9) — author data, not interpolation. | — | — |
| G14 | §2.1 l. 493, §3 l. 1203, §4.8 | The "one common installation year" restriction is stated three times; the general formulation's staging is never exercised. Say once; add to future work that staging is decided by the salvage convention (Addendum 58) and is the subject of the next paper. | L | Z |
| G15 | §2.3.1 l. 790 "end-of-block state of health … conservative" | Correct for the frozen convention. Do not touch (the 'mid' point is a sensitivity row in Table 6 and the baseline only in paper 2). | — | — |
| G16 | Letter v10 R2.5 | Add the G2 statement; add the W185 measurement if it lands; keep "no convergence theorem". | H | Z |
| G17 | Cover letter v2 l. 19 and the bracketed E1–E3 paragraph | Resolve the bracket per decision 1: describe what was run (W184/W185/screen) and state the LB/UB extension as future work in one sentence. | M | Z |
| G18 | Round-9 queue (change log) | Seven `Appendix~\ref` prefixes; E.24 marks + legend; E.23 mark → $^{s}$; "break-even" wording l. 1520/1806 and cover letter; l. 286 bullet dropped; marked copy regenerated last. | L | Z |

Not found (checked): duplicate or unresolved labels; the cost-file disclosure (present, letter l. 287); SoC window
consistency (10–90 % stated once, §3.4; nomenclature l. 262; no conflicting table — R3.1's second item is closed);
"annual results" wording (removed; the text says "discounted operating costs over the 15-year horizon" — R3.8 closed);
the salvage horizon-end convention (E.1 states y^End = 2040 and the block mapping — the 25 Oct §3.3 second item is
closed).

---

## 5. Reviewer coverage after round 8, and what the plan adds

| Comment | Status at `a87e195` + letter v10 | Added by this plan |
|---|---|---|
| R1.2 (i)–(iv), (vi) abstract | Met (uncertainty source, 108-bus, year/basis, lithium-ion) | — |
| R1.2 (v) degradation attribution | Pushed back with scope statement; future work | — (correct position) |
| R1.4 (i) broader uncertainties | Met (Introduction scope paragraph) | — |
| R1.4 (ii) cost of capital | Met (discount 0–8 %, reweighted; approximation stated) | — |
| R1.5 Figure 2 | Met (two-panel system figure) | — |
| R2.1, R2.9 bilevel / averaged plans | Met (two-stage, one plan) | — |
| R2.2 outer-loop algorithm | Met (Algorithm 1 + Algorithm 2) | G11 sentence |
| R2.3, R2.4, R2.7 cuts, sensitivities, feasibility cuts | Met by removal, stated | — |
| R2.5 ADMM convergence | Met in part (empirical record; no characterization) | **G2** (first-order statement) |
| R2.6 adaptive penalty | Met in part (rule and values stated; no sensitivity) | **G4** (W185 measurement) |
| R2.8 references | Met | — |
| R2.10, R3.4 apparent vs cell throughput | Met (active cell-side energy; converter not aged; scope stated) | — |
| R2.11, R2.12, R3.3 sign, SoC physics | Met (eqs. 20–28) | G1 conventions |
| R3.1 salvage; SoC limits | Met in part (post-evaluated credit, timing claims narrowed); SoC closed | — (route 2 of the 25 Oct review; paper 2 internalizes it) |
| R3.2 0.50 % gap; Fig. 15 | Met (no bound claimed; local characterization asked for — **G2**); Fig. 15 removed | **G6** (one operational figure) |
| R3.5 calendar ageing | Met (0.985/yr, cited; arms) | — |
| R3.6 degradation-neutral benchmark | Met in part (valuation at a fixed plan; planning part not met) | **G5** (15-point screen) if run |
| R3.7 NPV aggregation | Met (eq. 13, weights, discounting) | G9 sentence |
| R3.8 "annual" wording | Met | — |

---

## 6. Four-week action plan (decision 1(a))

Conventions: everything on the frozen configuration (`frozen_step6_tables_v1_590088fe.json`, spec v1 declarations of
Addendum 85); one gate per task with the exact command; runs attached, alone, both streams captured; no cited artefact
overwritten; numbers re-derived by script; the manuscript edited by the expert as paste-ready `main.tex` + `.diff`,
compile-checked.

**Week 1 (zero-solve + launches).**
- A1 (expert, Z): round 9 corrections + G1, G2, G9, G11, G14 wording + G8 noun replacement → `main_<hash>_round9.tex`
  + diff; letter v11 (G16, l. 286 bullet dropped, pointers); cover letter v3 skeleton with the E1–E3 bracket resolved
  conditionally. Compile 0/0, numbering unchanged.
- A2 (Planner, Z): W183i as ordered (hooks module + router patch, gates (a), (b), (d)) — the prerequisite for the
  fixed-plan launcher; G10 slack inventory over every evaluation behind Tables 7, 9, 10, 12–14; the E6 figure.
- A3 (author, Studio): on W183i's gates, launch **W184** (NRF arm, 1 cell) then **W185** (8 cells), each attached,
  one per task, trajectories recorded, capture checklist asserted before each run.

**Week 2.**
- A4 (Planner): W184/W185 reports; G3 and G4 numbers re-derived by script; predictions (below) checked.
- A5 (author decision 3): if A3 is complete by day 12, launch the 15-point no-ageing screen (G5); else drop it.
- A6 (optional, author): if a Gurobi key or a Studio slot exists, run T0.2 as rehearsed (132 solves, minutes each).
  Use only as one tightness sentence in §4.7 ("a convex relaxation of each block reproduces the certified value to
  within g on the plan without storage and g′ on the reference unit") — no convention change, no LB master.

**Week 3.**
- A7 (expert, Z): round 10 — §4.4 with W184 (both percentages), §4.7/§4.8 with W185 (spread of differences vs
  threshold), §4.3 with the screen if run, G6 figure(s), Appendix E compacted (G7) with the supplement zipped from
  `export_W160`; letter v12; highlights v4; cover letter v3 final. Compile 0/0; `verify_appE.py` PASS; numbering.
- **Numbers freeze at T − 10 days.** Anything not landed stays as the narrowed claim already in the text.

**Week 4.**
- A8 (Planner, Z): W-audit of the final Overleaf HEAD (A1–A8 of Addendum 82 style: numbers 0 MISMATCH, pointers, blue
  wrappers, CONFIRM tags 0, prefixes 0).
- A9 (expert): latexdiff against `manuscript_submitted/main.tex` regenerated last; final letter pointer recheck; the
  submission file set (main, highlights, cover letter, response, marked copy, supplement) verified byte-identical in
  the Overleaf clone; Overleaf HEAD stated in the change log.

**Compute budget:** W184 ≈ 3 h; W185 ≈ 20 h; screen ≈ 20–25 h; T0.2 ≈ 1 h; total ≤ 50 h of Studio time, serial, all
with existing launchers. **Fallback at every step:** the current text.

**Predictions to record in Addendum 89 before any run** (expert):
- P1 W184: imposing NRF on the coordinated exchange raises Q(0) by less than 10 % of the price-taker gap (i.e. the
  de-confounded saving stays above 11 %); the four reverse-flow hours carry < 1 % of the gap.
- P2 W185: across the penalty/start arms, each certified Q moves by less than 3b of its own window, and every
  decisive *difference* (x = 0 vs unit at 1× and at 2×) keeps its sign and its determinacy class.
- P3 screen: no no-ageing unit neighbour beats x = 0 by more than max{3b, 2τ}; at least one is within resolution.
- P4 T0.2 (if run): DSO blocks exact to ≤ 1e-5; x = 0 overall gap ≤ 1e-4 (H1); unit gap between 5e-5 and 2e-4 with
  the calendar-only variant C, dominated by the degradation relaxation.
A failed margin or sign prediction stops and is reported; a failed P3 is itself a result (the no-ageing search leaves
x = 0) and is written up as such.

---

## 7. The realignment as the next paper (recorded, not acted on now)

Carry Addendum 87–88 forward unchanged as the design record: convex relaxation LB (rank relaxation + calendar-only SoH
per F4 + exact coupling + identical objective convention), dual-anchored cuts with margin, lattice MILP master with
staged years and all scenarios, certified ADMM UB at the master's plan, mesh-local certificate on the coordinate unit
neighbourhood (R3), salvage inside F on both sides (R1(a), R2 as amended), mid-block SoH, 3-year blocks, R8 = option
(a). Present the LB/UB pair as the degradation-neutral vs degradation-aware controlled comparison (§3.2), with the gap
attributed by a no-ageing Q_cert at the same plan. First gate: T0.2 on all 12 groups × both cells with the resolution
column of spec §15; decision H1/H2/H3 as recorded. Open items that belong to it: the Clarabel stall, Gurobi QCP duals
(P5.5-C), Table 1 cost columns for 3-year blocks, `mu_init` (R7) only after Part B.

---

## Not confirmed

- Compute times above are scaled from the 61.4 h / 42-evaluation campaign and the W183 estimates (E2 ≈ 8 cells,
  E3 ≈ 1); W183i's gates have not run, so the fixed-plan launcher's readiness is as reported in `3a299aa0`, not
  re-verified here.
- The G10 slack claim (penalised slacks zero at every reported evaluation) is stated as a check to run, not as a fact:
  the W169 inventory covered 45 of 51 certified points.
- The affinity of Q in x (Phase A slopes) is taken from the record; it is a measured property of SRP1 at the frozen
  configuration, not a theorem.
- No file other than this one was created; the manuscript, letter and brief are unchanged.
