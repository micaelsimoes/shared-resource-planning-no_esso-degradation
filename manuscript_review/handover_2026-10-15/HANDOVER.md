# Handover — shared-ESS planning paper revision (state at 2026-10-15)

Read this first. It replaces the conversation history of the previous account; everything else lives in the
repository and in the files listed in §9.

## 1. Who does what

- **Author:** Micael Simões (INESC TEC). Writes the manuscript himself in LaTeX on Overleaf and decides every
  wording and scope question. He wants step-by-step instructions and text that says exactly what the code did.
- **Expert (you, Claude in Cowork):** external power-systems optimization expert (NLP, OPF, ADMM, direct search).
  You:
  - rule on the Planner's reports in numbered Addenda of `PLANNER_BRIEF_2026-09-13.md`;
  - verify every Overleaf pull against the frozen results and the Planner's code reports;
  - produce paste-ready fixes, usually a full corrected `main.tex` plus a `.diff`;
  - keep `manuscript_review/W177B_CHANGE_LOG.md` up to date.
- **Planner:** a Claude Code agent run from the Claude desktop app on the Mac Studio. It works zero-solve: it audits
  text against code and the frozen records, and commits reports `P5_15_W1xx_*.md` to the repo. It is coordinated
  only through the brief; the author forwards a short prompt that points to the latest Addendum. The Advisor and
  Worker agents exist but are not used in this phase.

## 2. The paper

"Degradation-Aware Planning of Shared Battery Energy Storage Systems for Coordinated Transmission and Distribution
System Operation" (Simões, Peças Lopes, Soares), Journal of Energy Storage, elsarticle preprint 12pt, first revision,
three reviewers.

Main contribution (author's decision, 2026-10-13): planning of shared ESSs under a nonconvex operational recourse.
The ADMM/TSO–DSO coordination detail lives in Appendix A, and the main text keeps the planning focus.

**Method:**
- two-stage stochastic program;
- a derivative-free master (MADS/OrthoMADS on a granular lattice, variants A and B, completion cap 30);
- each candidate evaluated by consensus ADMM (TSO, three DSOs and a shared-ESS agent carrying the degradation chain);
- every evaluation behind a reported comparison certified by an explicit rule (v6, Appendix A.5), or reported in an
  uncertified form with its bar;
- no optimality certificate claimed.

## 3. Machines, paths, workflow

| | path |
|---|---|
| MacBook Air (author's desk, linked to Cowork) | `/Users/micaelsimoes/PycharmProjects/shared-resources-planning` |
| Mac Studio (runs the Planner) | `~/Projects/share-resource-planning-no_esso-degradation` |
| Branch | `feature/derivative-free-planning` |
| Overleaf clone (in both checkouts, git-ignored by the main repo) | `manuscript/6a67305f25e8348fb71380c3/` (`main.tex`, `response_to_reviewers_draft.tex`) |
| Submitted version | `manuscript_submitted/` |
| Expert deliverables, reviewer files, Planner exports | `manuscript_review/` |
| Brief, round corrections, method docs | repo root |

**Cycle:**
1. The expert delivers files into `manuscript_review/` on the Air.
2. The author pastes them into Overleaf and compiles.
3. He pulls the Overleaf clone on the Air, then commits and pushes the repo from PyCharm on the Air.
4. He pulls on the Studio (`git pull --no-rebase` if the branches diverged).
5. He forwards the Planner prompt.
6. The Planner commits its report and pushes; the author pulls on the Air; the expert reads the report there.

**Sync of every expert deliverable:**
1. Copy it to `/mnt/user-data/outputs`.
2. Send it with SendUserFile.
3. Write it to the Air's `manuscript_review/` with `device_commit_files`.
4. Save durable docs (brief, change log) to the claude.ai Project.

Before any audit, verify that the Overleaf clone on the Air is byte-identical to the file you delivered
(md5sum), and state the Overleaf HEAD.

## 4. Conventions

- **Rulings:** Addenda in `PLANNER_BRIEF_2026-09-13.md`. Each holds the rulings on the last report, an order
  (W-number) with numbered items, and the expert's predictions. The latest is **Addendum 78 (W179 order)**.
- **Change log:** `manuscript_review/W177B_CHANGE_LOG.md`, one entry per Overleaf commit or expert delivery.
- **Open checks in the manuscript:** `% [CONFIRM — W1xx]` LaTeX comments mark statements the Planner must check.
  They are removed once answered.
- **Colour:** revised text is wrapped in `\textcolor{blue}{…}`.
- **Number sources:**
  - every number comes from the frozen tables `frozen_step6_tables_v1_590088fe.json`, via the export
    `manuscript_review/export_W160/` (T1, T2, T5, T6, T9, T10 and README);
  - never type a table number by hand when it can be generated (see `gen_appE.py`);
  - derived numbers are re-derived in a script before they are used.
- **Wording:**
  - "determinate" means |Δ| ≥ max(3b, 2τ) between certified evaluations; with an uncertified evaluation, |Δ| must
    exceed 3·max(|gap|, |slack|);
  - smaller differences are "within resolution" / "within the uncertified bar";
  - τ = 4,539.07 €;
  - Q = gross operational cost, settlement excluded; net = Q − terminal salvage credit; value = Q(0) − Q(x);
    F = I + Q;
  - "reference unit" = node 7, 0.25 MVA / 1 MWh, 2025, never "the smallest unit".
- **Compile check in the cloud container (demo graphics):**
  - copy `elsarticle.cls` from the Overleaf clone, or rebuild it with `latex elsarticle.ins`;
  - prepend `\PassOptionsToPackage{demo}{graphicx}` to a copy of `main.tex`;
  - run `pdflatex` → `bibtex` → `pdflatex` ×2;
  - require 0 errors and 0 undefined references;
  - compare the `\newlabel` numbers in the `.aux` before and after any edit;
  - the letter compiles on its own (refs [7]–[9] are typed).

## 5. State at 2026-10-15 (end of the previous account's session)

- **Overleaf HEAD `48099ab`:**
  - `main.tex` is byte-identical to `manuscript_review/main_e926fb7_appE.tex` (Appendix E rebuilt);
  - `response_to_reviewers_draft.tex` is byte-identical to `manuscript_review/response_to_reviewers_v6.tex`.
- **Repo:** pushed at `e522c44a` ("Appendix E rebuilt from export_W160; letter v6; Addendum 78 (W179 order)").
- **Appendix E "Evaluation Record"** (label `app:results`):
  - Table E.20: 58 comparisons. These are T1's 55 distinct ones (the record's 60 claims less the four G `:net`
    duplicates and the CHECK headline) plus the 3 distinct T10 claims.
  - E.21: the 61 evaluations, numbered.
  - E.22: the eight neighbours of x = 0 re-evaluated under the rule.
  - E.23: the static arrangements by solver start.
  - E.24: the gap-refused evaluations.
  - E.6: certification statistics and reproducibility.
  - Generated by `manuscript_review/gen_appE.py` from `export_W160/*.csv`; `verify_appE.py` re-derives every prose
    count (31 PASS).
- **Next action:** if the author has not sent it yet, he forwards to the Planner on the Studio:
  > Read Addendum 78 in PLANNER_BRIEF_2026-09-13.md and run W179 (zero solves) at the current Overleaf HEAD; state
  > the HEAD. Report as P5_15_W179_REPORT.md, then commit and push.
- **When the W179 report arrives**, check it against Addendum 78's predictions. Items:
  - (a) Appendix E entries against `590088fe`, plus nine expert interpretations (i)–(ix). Most likely to need wording:
    the unit of "NRF exceedances" in E.23, and the "119 acceptable-level exits" sentence in E.6;
  - (b) the §4.6/§4.7/§4.8 edits and letter v6;
  - (c) G1–G12 and letter v5;
  - (d) the closure-slack claim in §2.3.3 ("both daily closure slacks were zero … at every certified evaluation"):
    the W169 inventory covered 45 of 51 certified points;
  - (e) §4.8 Scope "every evaluated unit neighbour of the plan without storage determinately worse": only 8 of the
    14 neighbours were re-evaluated under the rule, so it probably needs a qualifier. Also the §2.2.7 Determinacy
    sentence ("bands, gaps and slacks of every evaluation … tabulated in Appendix E").
- **Then:**
  - write Addendum 79 with the rulings and the fixes;
  - do a final letter pointer recheck after any renumbering;
  - run a `latexdiff` against `manuscript_submitted/`;
  - finally the cover letter and submission.

## 6. Results the paper reports (single-scenario unless stated; k€, 2 % discount)

- **Baseline:** the search returns x = 0, and all 14 admissible unit neighbours of x = 0 were evaluated by the
  search.
  - Reference unit: value 253.5 against I = 318.0, so value − I = −64.4 (4.93× its threshold of 13.1).
  - Value ladder: every plan loses determinately (Table 7).
- **Break-even energy cost (4 h at node 7):** 171–193 k€/MWh against 254, so the energy cost must fall by 24–33 %
  (certified-only fit 183.08).
- **Flexibility price:**
  - the reference unit pays at 1.75× (+14.2, 1.32×) and 2× (+46.3, 3.77×); at 1.5× it is −14.5, within the
    uncertified bar;
  - at 2× the search returns a two-node plan in 2030 (node 5 0.25/0.5 + node 7 1/3.5), uncertified (gap clause);
  - 17 of its 61 admissible neighbours were evaluated and 13 re-evaluated; it is determinately better than 7 and
    within resolution of 6.
- **Ageing (reference unit, plan fixed):**
  - no ageing −4.1 (within resolution);
  - cycling only −31.6; baseline −64.4; datasheet 8,000 cycles −45.2; retention 50 % −73.7; retention 50 %
    mid-block −65.9;
  - 0.50 floor −59.5 (value 258.5, AE 0.778, EFC/day 1.08);
  - the 0.70 floor binds in 2035 for the baseline and both retention-50 % arms.
- **Coordination vs best static no-reverse-flow arrangement (price-taker):** 90,896.6 (13.9 %), against
  241,176.2 (36.9 %) for the passive arrangement. Split: 77.2 % TN, 22.8 % DN flexibility.
- **3×3 instance:**
  - −81.3 (4.52× its resolution of 18.0);
  - R = 0.909–0.934 (0.9331 predicted);
  - production-exit evaluations; no search run there.
- **Investment year (unit):**
  - 2030: 68.5 gross / 56.7 net;
  - 2035: 111.0 / 54.5;
  - 2035 − 2030: +42.5 (determinate) gross, −2.3 (within resolution) net.
- **Discount 0–8 %:** never pays. Rates other than 2 % re-weight the 2 % evaluations (exact at x = 0, first-order
  at the unit).
- **Salvage:** over the 55 comparisons reported gross and net, 0 sign changes and 1 verdict change (Table E.20 row
  34, 0.95× → 1.22×).
- **Certification:**
  - 42 evaluations under the rule: 32 certified (24 oscillatory, 8 monotone) and 10 uncertified (5 gap clause,
    2 lapse resets, 3 growth test);
  - median k\* 174; k\* − k₀ median 65 (21–89); range/τ median 0.86;
  - 10 of the 46 certificates are within 5 % of τ;
  - the reference evaluations moved 15–21 k€ after the residual test passed.

## 7. Numbering (Overleaf `48099ab`)

- **Tables:** 1–15 in the main text, D.16–D.19, E.20–E.24 (`\addtocounter{table}{-1}` after each nomenclature
  longtable).
- **Equations:**
  - Q(x) (13); cohort ratings (14)–(15); available capacities to (19);
  - network storage block (20)–(28): SoC limits (22), power limit (26), circle (27), complementarity (28);
  - agent and degradation (29)–(35); salvage (38)–(39).
- **Figures:** 1 framework, 2 search, 3 test system, 4 lattice, 5 break-even, 6 flexibility ladder, 7 ageing,
  8 benchmark, 9 Q against cycle; B.10, C.11, D.12–D.20.
- **Sections:** §2.2.7 recourse value, concise certification, resolution, determinacy; §2.3.2–2.3.5;
  §3.6 "Planning and Certification Settings"; §4.1–4.8; Appendix A.1–A.8 (A.5 certification rule, A.8 solver and
  coordination settings); B market; C TN; D DNs; E Evaluation Record (E.1–E.6).

## 8. Author's standing preferences and decisions

- He debates reviewer comments rather than accepting them all; replies must state exactly what was and was not done.
- **Salvage:** reported, not optimized (a post-evaluated credit; the recourse does not see it).
- **Degradation:** no explicit cycling cost (a shared asset that is paid for has no operating cost of its own);
  degradation acts through available energy and the end-of-life floor.
- **TSO/DSO attribution:** no attribution of degradation to TSO vs DSO (joint schedule); a marginal allocation is
  future work.
- **RES capability:** tied to available active power (not modelled as STATCOMs).
- **Machines:** Planner runs on the Studio through the Claude desktop app. INESC TEC is also preparing a Linux VM
  (Ubuntu 24.04, 64 GB, 24 cores) as a compute target.

## 9. File index (all in the repo; the Project holds copies of the starred ones)

- \* `PLANNER_BRIEF_2026-09-13.md`: every ruling, Addenda 1–78.
- \* `manuscript_review/W177B_CHANGE_LOG.md`: every manuscript change since W177(b), and what is open.
- \* `STEP6_REVISION_MAP.md`: the section map of the revision (2026-10-06).
- `STEP6_ROUND1…5_CORRECTIONS.md`: the round-by-round fixes to §2, §3 and Appendix A.
- \* `STEP4_DFO_METHOD.md`: the derivative-free method specification.
- `section4_expert_draft.tex`, `section4_tables_refit.tex`, `manuscript_review/section2_expert_draft.tex`,
  `manuscript_review/appendixA_expert_draft.tex`: earlier drafts; superseded by the Overleaf text.
- \* `manuscript_review/appendixE_evaluation_record.tex`, `gen_appE.py`, `verify_appE.py`, `export_W160/`.
- \* `manuscript_review/main_e926fb7_appE.tex`: current manuscript (= Overleaf `48099ab`).
- \* `manuscript_review/response_to_reviewers_v6.tex`: current letter.
- Planner reports `P5_15_W17x_*.md` (repo root): W177B, W178 (+ A/B/C); W179 next.
- `REVISION_CONTEXT.md`, `TASKS.md` (repo root): the Planner's own state.
