# The manuscript as a paper — assessment and plan for the presentation pass (round 10)

Expert, 2026-11-12. Basis: Overleaf `3c379fa` (round 9c, 63 pages), `manuscript_submitted/main.tex`, letter v13,
the three reviewers' letters, the Journal of Energy Storage's readership. Numbers, tables, equations and the appendix
structure are **not** touched by this plan; it is a prose pass, so every audit the Planner runs (numbers 0 MISMATCH,
pointers, blue wrappers, `verify_appE.py`) stays valid, and the upload date of 2026-11-19 holds.

## 1. Assessment

**The diagnosis in one sentence.** The revised text is correct, complete and honest, and it reads as the report of a
verification campaign rather than as a paper about shared storage planning: the apparatus built to make the numbers
trustworthy has become the subject.

**The measurement.** In the main text (Introduction to Conclusions, 6.7 k words) the words that belong to the
verification apparatus occur more often than the words that belong to the problem: "certified/certification" 58
times, "evaluation" 60, "threshold" 20 and "bar" 21, "acceptance" 22, "stopping" 14, "cycle" 32, "determinate" 13,
"within numerical resolution" 6 — against "degradation" and "storage", which the paper is about. Every results
subsection opens by naming a table and closes with the status of its differences; the finding sits in the middle,
stated once, surrounded by qualifications. The reader is asked to understand the acceptance rule (§2.2.7, twelve
sentences) before seeing a single result, and is reminded of it in every table note.

**What the submitted version did better, and what it did worse.** The submitted paper (author's prose) is
declarative and compact: one idea per paragraph, sections named after objects ("Shared ESS Investment Plan",
"Degradation and Available Energy Capacity"), tables introduced by what they report, results discussed as physics and
economics. Its defects were in the mathematics and the claims (invalid cuts, averaged plans, apparent-power ageing,
missing SoC physics), not in the writing. The revision fixed the mathematics by replacing the method, and fixed the
claims by qualifying every one of them — and in doing so it replaced the author's voice with the auditor's. The
qualifications are all true; most of them belong in one place, not in every paragraph.

**Where it departs from the original, and whether that is a problem.** The departures a reviewer will register:
(i) the method (search instead of cuts) — necessary, explained, no problem; (ii) the finding (storage does not pay at
baseline) — a legitimate negative result, no problem if stated as a finding rather than apologised for; (iii) the
scope (single-scenario search, one common year, SRP1 instance) — narrowed, stated, acceptable; (iv) the *register* —
the problem. Reviewers 1 and 3 engaged with a power-systems paper about storage, coordination and degradation; the
revision answers them in the language of numerical analysis. Nothing in the reviewers' requests asks for that
language in the main text; Reviewer 2's requests (residuals, tolerances, penalty rule, failure handling) are answered
in Appendix A, where they belong.

**Three specific symptoms** (line numbers of `3c379fa`):
1. §2.2.7 "Recourse Evaluation and Numerical Acceptance" (l. 703–750): the definition of Q is three sentences; the
   acceptance machinery is nine. The section a reader needs is "what Q is and how it is aggregated"; the rest is A.5.
2. §4.1–4.6: each subsection carries 3–6 status clauses ("Both cost increases are determinate", "All listed margins
   and increments are determinate", "Some comparisons … remain within numerical resolution", "These evaluations …
   were not certified under the rule of …"). The information is already in the tables' threshold columns and marks.
3. §2.1 and Algorithm 1 (l. 490–531): the search is described at the level of its implementation (variants A and B,
   completion cap, N^max, settling bar, "production exit", OrthoMADS points snapped to the frame). For the paper, the
   search is a derivative-free local search on a lattice with a stated termination condition; the variants are a
   parameter table in Appendix B.4.

## 2. Principles of the pass (the author's style, restored)

- **Findings first.** Each results subsection opens with the result in one declarative sentence, in the author's
  register ("Under the baseline assumptions, no storage plan recovers its investment."), then the table, then the
  physical or economic explanation. Status words come last, once, or not at all when the table carries them.
- **One place for each piece of apparatus.** The acceptance rule is defined once (§2.2.7, four sentences) and
  specified once (A.5). "Certified" appears in the main text only in that definition and in table notes; in the prose
  a difference is "resolved" or "not resolved at the stated tolerance". "Determinate" leaves the prose and stays in
  the tables' legends.
- **Tables carry the status; prose carries the meaning.** Threshold columns and the $^{u}$/$^{f}$/$^{b}$ marks stay;
  the sentences that restate them go.
- **Scope statements are stated once, where the scope is set** (§3 settings, §4.7 limitations), not repeated at
  every use.
- **Nothing numerical moves.** No value, table, figure, equation or label changes; the Appendix B generator output is
  pasted unchanged; the letter's pointers survive because the section structure survives.
- **The author edits in his own voice after the expert's draft.** The expert produces the paste-ready round-10 file
  with the pass applied; the author rewrites any paragraph he would have written differently; the Planner audits the
  result as usual. The shared git clone makes this a two-day loop.

## 3. Section-by-section plan

| Section | Now | Pass | Words (±) |
|---|---|---|---|
| Abstract | Method sentences correct; the results sentences are a list of statuses ("within numerical resolution of break-even", "estimated margin remains negative") | Keep sentences 1–4. Results in three sentences: coordination value (12.2 %, with the benchmark named); storage does not recover its cost at baseline and the gap is degradation (−64 k€ aged against ≈ 0 unaged for the reference unit); storage pays at 1.75–2× the flexibility price. One sentence on the method's scope ("single operating scenario in the search; fixed plans re-evaluated under nine scenarios") | −20 |
| Introduction | Good through the gap statement. Contribution 3 advertises "explicit numerical acceptance criteria"; contribution 4 lists the case-study outputs | Contributions as four claims about the problem: (1) the planning formulation with one implementable plan; (2) the degradation model (active cell-side throughput, calendar ageing, floor, datasheet calibration); (3) the nested derivative-free search over ADMM-coordinated operation, "with the resolution of every reported comparison stated"; (4) the findings: coordination value, break-even in cost and flexibility price, degradation's share of the storage margin. The paragraph on Benders stays (it answers R2/R3) but shortened by a third | −60 |
| §2.1 Architecture | Two paragraphs of two-stage framing, the generic problem, Fig. 1, then the search at implementation level, Algorithm 1 | Keep the framing (once), eqs. (1)–(3), Fig. 1. The search in one paragraph: lattice, polling of unit neighbours, acceptance of an improvement only beyond the resolution of the two evaluations, termination when no unit neighbour improves; "variants A and B, poll sizes and limits are given in B.4". Algorithm 1 stays but loses the variant branches (one generic poll step; the variants described in B.4) | −120 |
| §2.2.1–2.2.6 Master problem | Fine; a few duplicated statements ("investment decisions are common to all cost trajectories" ×2) | Deduplicate; otherwise untouched | −40 |
| §2.2.7 Recourse evaluation | Q defined, then nine sentences of acceptance | Q and its aggregation (eq. 13, the cost components, the exclusions, the weights) in the author's register; then four sentences: every evaluation is stopped by the residual test; comparisons in §4 use evaluations re-run under the acceptance criteria of A.5; a difference is reported with the threshold it must exceed to be resolved; evaluations that do not meet the criteria are marked and compared with a wider bar. The first-order-stationarity sentence stays | −90 |
| §2.3 Operational subproblem | Good: this is the physics the reviewers asked for | Untouched except the two "the case study considers…" scope repeats, which move to §3 | −20 |
| §3 Case study | Good; §3.6 "Planning and Evaluation Settings" is where scope belongs and is already there | Fold the valuation-probe sentence into one; the benchmark definition stays. Untouched otherwise | −10 |
| §4 opening | Convention paragraph fine | Add one sentence that announces the order of the results (plan → break-even → degradation → coordination → scenarios → sensitivities → numerical scope) in the author's style | +30 |
| §4.1 Baseline plan | Finding stated, then four status clauses | Finding first; "Both cost increases are determinate" and "All listed evaluations are certified, and all cost increases are determinate" go to the table note (already there) | −40 |
| §4.2 Break-even | Correct but dense: fit definition, certified-only vs banded, probes, flagged certificate | Open with the result (24–33 % cost reduction needed); the fit in two sentences; the probe sentence once (it is also in §3.6 — keep the §3.6 one and a short pointer here). Flexibility ladder: open with "the unit pays at 1.75× and above" | −70 |
| §4.3 Degradation | Good structure; the floor paragraph is long; Figure 6 paragraph new | Open with the mechanism (ageing turns a break-even unit into a −64 k€ one); the floor paragraph halved; Figure 6 paragraph kept | −60 |
| §4.4 Coordination | Good; the NRF paragraph (W184) new | Open with the 12.2 % and the benchmark's rule in the same sentence; the two interface-restriction paragraphs merged into one | −40 |
| §4.5 Multi-scenario | Correct; two status caveats and the continuation-fit paragraph | Open with the finding (−81 k€, not resolved to break-even); the continuation fit in one sentence; "were not certified under the rule" → the table note | −50 |
| §4.6 Year and discount | Fine | Untouched | 0 |
| §4.7 Numerical performance, scope, limitations | Campaign statistics, Fig. 8, penalty check (W185), cost of the campaign, four limitation paragraphs | Reorder: (i) cost and scale of the evaluations (one paragraph); (ii) what the acceptance criteria found (32/42, the post-stopping movement, Fig. 8) in one paragraph; (iii) the penalty check; (iv) limitations as now. Remove "In three evaluations continued beyond certification…" and "61.4 h over 8,570 ADMM cycles" to B.4 | −80 |
| Conclusions | Four paragraphs, correct | Rewrite in the author's register: what was built (2 sentences), what was found (4 sentences with the numbers), what it means for planning shared storage (2 sentences: degradation and flexibility prices decide the case, coordination pays regardless), future work (as now) | ±0 |
| Appendix A | Dense, needed (R2.5/R2.6) | Remove process words only: "the production exit", "the campaign harness", "a declared option, enabled in every campaign behind the reported tables", "continued from an earlier run" (keep the A.5 statement). No equation or parameter changes | −60 |
| Appendix B | Generated | Untouched (B.4 may lose two sentences through the generator if the Planner has time; not required) | 0 |

Net: about −700 words of main-text prose, 17 → 17 floats (no float moves — they are what the reviewers asked for),
63 → ≈ 60 pages. The deeper structural cut (merging §4.5–4.6, dropping Fig. 8) is **not** in this pass: it would
move numbers and floats, and the window does not allow re-auditing that.

## 4. What stays as it is, and why

- The acceptance rule itself, Appendix A.5 and Appendix B: Reviewer 2 asked for exactly this, and the determinacy
  statements are what make a negative result publishable.
- The limitations paragraphs: they are the honest part and short.
- Every number, table, figure, equation and label: the audits depend on it, and so does the letter.
- The blue colouring: Elsevier wants the changes visible; the marked copy carries the rest.

## 5. Schedule inside the window

| Date | Who | What |
|---|---|---|
| 11-13 | expert | round-10 draft with the pass applied: `main_<HEAD>_round10.tex` + diff, end-matter statements inserted (CRediT and funding from the author), letter v14 if any quoted wording changed |
| 11-13/14 | author | edits in his own voice directly in the shared clone (prose only; any number or label change is flagged to the expert first) |
| 11-14 | expert | W187 sentence (if the report is in) or cover letter v5 without the bracket; marked copy regenerated; the final file set |
| 11-15 12:00 | — | numbers freeze (unchanged by this plan: no number moves) |
| 11-16 12:00 | — | **prose freeze**: the text after this point changes only on an audit finding |
| 11-16/17 | Planner | W190: numbers 0 MISMATCH, letter pointers, blue wrappers, `verify_appE.py`, the marked copy compiles, the file set byte-identical |
| 11-18 | expert | final pointer recheck; go |
| 11-19 | author | upload |

**Division of labour, stated once.** The expert drafts; the author owns the voice and may rewrite any sentence; the
Planner audits what neither of us can see (numbers, pointers, wrappers). To keep the two-day loop safe in a shared
clone: the author edits only `main.tex` prose, commits with a message that says "prose", and never touches a number,
a label, a table row or Appendix B; the expert regenerates the diff and the marked copy from the author's HEAD, not
from its own copy.

## 6. The author's inputs needed before 11-13

1. The CRediT roles and the funding line (end-matter).
2. Any paragraph of the current text he already knows he wants rewritten in his own words — a list of section
   numbers is enough; those are left to him and the expert does not draft them.
3. Confirmation that the abstract's three results sentences may carry the three numbers proposed above (12.2 %;
   −64.4 k€ aged against a margin within the resolution of zero unaged; positive at 1.75 and 2 times the flexibility
   price). They are the numbers already in the abstract, re-ordered as findings.
