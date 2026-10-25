# Step 7 — Additional simulations: bounded plan, v2 (expert, 2026-10-25)

Supersedes v1 of the same date. Inputs: the author's brief (`simulation_study_brief_for_expert_2026-10-25.md`),
decisions D1–D5 and the six protocol clarifications (author, 2026-10-25), `STEP7_PLAN_REVIEW_2026-10-25.md` and
`manuscript_critical_review_2026-10-25.md` (both read in full). Rulings and the Planner's W182 order are in Addendum 82
(revised). **Nothing here launches a run.** The campaign spec is frozen only after W182 reports; runs are then
authorized one gate per task, with the exact command, by a later addendum.

Cost basis used throughout (Appendix E.6, §4.7, Addendum 52): 25.6 s per ADMM cycle (median, SRP1, one evaluation at
a time, single thread); a certified evaluation stops at median cycle 174 (≈ 1.25 h; cap min{k₀ + 109, 300} ≈ 2.1 h
worst case); a search evaluation at the production exit ≈ 100–110 cycles (≈ 0.75 h); 2.2 GiB per SRP1 evaluation.
Concurrency figures are provisional: the searches ran batches on up to 10 slots with a bitwise gate (Addendum 25), but
per-cycle time under concurrency was never re-measured on the final code; the certified evaluations stay serial.

## 0. Decisions adopted

| | Decision | How the plan applies it |
|---|---|---|
| D1 | E3, staged E2, matched E1 at 2×, baseline-price unit-neighbourhood screen, E6; E4 and E5 deferred, claims narrowed | §§3–7; the narrowed claims in §8 |
| D2 | E1 primary setting 2×, a preselected setting where storage already has value; the baseline screen reported whatever it shows | §5.3, §5.4 |
| D3 | Aged arm re-run on the final implementation with the same protocol as the no-ageing arm | §5.1 |
| D4 | "Two-stage stochastic" kept where it describes the formulation; single-scenario searches and multi-scenario fixed-plan evaluations distinguished | §8, W182 A7 |
| D5 | W182 → E3 → E2 → (numerical issues resolved) → E1; E6 alongside; marked copy regenerated at the end | §9 |

## 1. What each study can and cannot establish (hypotheses, not expectations)

Every prediction below is a recorded hypothesis under the brief's §3 rule: no outcome triggers an unplanned search for
a preferred result, and unresolved differences are reported as unresolved. Specifically: a larger no-ageing plan and a
positive cross-evaluation difference are not required (a negative difference says the no-ageing search found a plan
that performs better under the aged model, not that ignoring ageing is preferable; a positive one is evidence about the
two returned plans, not a regret bound); the four reverse-flow interface-hours do not bound the cost of imposing NRF
(magnitudes, weights, prices and binding constraints decide it); passing the penalty pilot does not establish
independence from initialisation.

## 2. Evidence inventory

Status "reusable" means the record is on the final code head under the frozen configuration, or is exactly
model-independent; W182 confirms each in Z1–Z6 before it is reused.

| Need | Exists | Where | Status |
|---|---|---|---|
| Q(0) at 1× and 2×, certified | yes | E.21 rows 1 (`ref:7aa017f0`), 34 (`h_50dea31c`) | reusable for both ageing arms **if** Z1 shows Q(0) is exactly ageing-independent (storage rows deactivated at x = 0) |
| Reference unit at 1× under A, N | yes | rows 2, 54 | reusable as fixed-plan cells; the N cell is also a seed cost for the baseline screen |
| Reference unit at 2× under A | yes | row 35 | reusable (fixed-plan) |
| Historical 2× aged incumbent {5: 0.25/0.5; 7: 1/3.5, 2030} and its 13 re-evaluated neighbours | yes | rows 40–53 | **not** the controlled comparator (D3): searched on an earlier code state, A-then-B protocol, seeded caches; row 40 uncertified (gap clause). Kept as a reference row and as a common seed candidate |
| Historical 1× aged search: x_A = 0, 14 neighbours evaluated, 8 re-certified | yes | s47; T5; E.22 | reference only; the 1× screen is re-run under N on the final code |
| No-ageing search (any price) | no | — | E1 arm N |
| Aged search on the final code with the matched protocol | no | — | E1 arm A (D3) |
| Coordinated evaluation with NRF on both interface copies (C_NRF) | no | — | E3 |
| Reverse-flow hours, magnitudes and weights in C_OPEN | hours yes (4 of 864); magnitudes to be read | benchmark spec v3 sweep; `certified_models.pkl` of row 1 | W182 Z7 inventories magnitude × weight × price without solves |
| Three-start spreads of the static arms (25.8 / 0.0 k€) | yes | T6; E.23 | E3 reporting rule (static side) |
| Between-configuration spread of Q at C\* (≈ 72 k€) | yes | Step 3 (Addenda 22–25); E.23 note | E3 reporting rule (ADMM side, conservative); E2 context — a spread of Q, not of margins |
| ADMM cells with ρ₀ × 0.5 / × 2, or alternative starts, at the paired plans | no | — | E2 |
| Bitwise replays; DSO block multimodality ≈ 0.3 % | yes | E.6; Step 2 audit | E2 interpretation only (repeatability; local optima exist) |
| Per-block cost decomposition, E^Av per year, terminal SoH for rows 1 and 2 | yes, committed | `component_levels_terminal.json`, `evaluation_record.json` | E6 panels (b), (c) |
| Storage schedule and SOC for row 2 | schedule per cycle: file exists, **uncommitted** (hashed in the W179c manifest); SOC: in `certified_models.pkl` only | cell directory of row 2 | E6 panel (a): Z5 commits or hash-records the exact inputs before use |
| Expected-profile instance on the five-block grid | no | — | E4 (deferred) |

## 3. E3 — coordination with a matched reverse-flow restriction (first run)

**Arm.** C_NRF: the coordinated procedure at x = 0, baseline price, single-scenario configuration, frozen A.8
settings, plus the no-reverse-flow row of the benchmark arms (interface active power ≥ 0 in the DN-to-TN direction,
Addendum 57's sign convention) on **both** copies of each interface — the TSO's and the DSO's — so that consensus
cannot satisfy the row on one side only. Final feasibility is checked on both copies at the certifying cycle (the
harness hooks record interface P on each side). A labelled variant: never mixed with baseline cells in a table; the
frozen configuration file is untouched (the row is a declared option, off by default). One certified evaluation.

**Outputs.** Q(C_NRF) with status (certified: k\*, band; uncertified: gap, slack); the number of NRF-binding
interface-hours; the two conditional differences and their percentages with denominators: coordination under NRF,
Q(P_NRF) − Q(C_NRF), as a saving relative to Q(P_NRF); permitting reverse flow under coordination, Q(C_NRF) −
Q(C_OPEN), relative to Q(C_NRF). Both recomputed from the records (never from the printed 12.2 %). The decomposition
into TN conventional energy and DN flexibility for each difference, and the curtailment columns, as in Table 12/E.23.
The mechanism statement of §4.4 is re-derived from the new decomposition, not carried over.

**Reporting rule for the mixed comparison (clarification 5).** Q(C_NRF) − Q(C_OPEN) is between two ADMM evaluations:
rule v6 applies (determinate at ≥ max{3b, 2τ} if both certified; the uncertified form with its bar otherwise).
Q(P_NRF) − Q(C_NRF) is between a static NLP solved from three starts and an ADMM evaluation; the two uncertainties are
different objects and are reported side by side, never combined into one "band": the static arm's three-start spread
s (0.0 k€ for the price-taker arm, which is repeatability across starts, not exactness), and for the ADMM cell the
larger of its certification band (or bar) and the between-configuration spread of Q measured at C\* (≈ 72 k€), the
only empirical measure the record has of how far a coordinated evaluation can move under a configuration change. The
difference is called "resolved under the declared procedure" only if it exceeds three times each of the two measures;
the wording "determinate" is reserved for ADMM pairs under rule v6. This is an empirical rule, stated as such.

**Cost.** One certified evaluation, 1.5–2.5 h.

**Hypotheses (recorded).** H3a: Q(C_NRF) − Q(C_OPEN) is positive and within resolution (< 2τ = 9.1 k€). H3b: the
NRF-conditional saving Q(P_NRF) − Q(C_NRF) is 90.9 M€ ± 0.1 M€, 12.2 % at one decimal. If H3a fails (determinate), the
paper reports both conditional terms and the headline becomes the NRF-conditional saving; if H3b fails by more than
the rule's measures, §4.4 is rewritten from the new decomposition. Either way the benchmark's interface restriction
is stated with the headline (audit 3.5), which `edd7a63` already does.

## 4. E2 — numerical robustness, staged (second)

**Stage 1 — penalty pilot (8 evaluations, serial ≈ 10 h).** Plans {x = 0, reference unit} × prices {1×, 2×} ×
{ρ₀ × 0.5, ρ₀ × 2} on all three channels, with balancing, freeze, AA and tail rules unchanged. Each cell certified
under rule v6 or reported uncertified. Z6 confirms ρ₀ is a configuration parameter (no code change).

**Margin and verdict rule (clarification 3).** The margin of a configuration is value − I from *its own pair* (x = 0
and the unit under the same ρ₀). Each pair's verdict is decided by its own ranges, gaps and statuses under rule v6
(threshold max{3b, 2τ} from its own bands; the uncertified form if either member is uncertified). The baseline pair's
threshold (13.1 k€ at 1×, 12.3 k€ at 2×) is quoted as a descriptive reference only. Two quantities are reported
separately: (i) the range of paired margins across configurations, and (ii) the per-pair acceptance and verdict.
Whether local-solution differences cancel within a pair is a hypothesis the pilot tests, not an assumption.

**Feasibility checks per cell** (already captured by the harness hooks; the pre-run checklist assertion lists them):
closure slack at the bound, ESSO σ slack, converter-circle residual, max |P^E| / S^Av, SOC within limits, interface
gap and residual ratios, non-clean cycles. Unsuccessful or unaccepted runs are reported, not dropped.

**Stage 2 — coordinated-start perturbation (conditional on Z6; 4 evaluations, ≈ 5 h).** The procedure's start is the
networks' initial solves (A.5); the only documented alternative start is the benchmark harness's perturbed start,
written for the static arms. If Z6 finds it applicable to the coordinated procedure without new code, stage 2 runs the
four plan/price cells from the perturbed start at ρ₀. If not, E2 is described in the paper as sensitivity to initial
penalties, and the paper says it does not establish independence from starts.

**Centralized reference.** Not built (an all-block centralized NLP is outside the minimum package); a small
equivalent-instance check remains optional and is not scheduled.

**Hypothesis (recorded).** H2: over stage 1 the paired margin at each price stays within the baseline pair's threshold
of its baseline value and no pair's verdict changes. If H2 fails at either price, the campaign stops before E1 (D5),
the uncertainty is identified, and the paper reports the margin as a range over configurations.

## 5. E1 — matched degradation-aware and degradation-neutral searches (after E2)

### 5.1 Common protocol (clarification 1)

Both arms run on the **final code head** under the frozen configuration, with the same:

- **Search variant:** B throughout — the 2n = 14 OrthoMADS directions snapped to the incumbent's unit frame, Δ = 1,
  completion to n + 1 with the admissible unit box, the poll refused for review beyond 30 points. Variant A is not
  used: its completion at a two-node incumbent exceeds the cap (that is why the historical 2× run was stopped for
  review and continued with B), and a shared bounded single-variant protocol is what the review asks for.
- **Seed set S, fixed now, evaluated under each ageing model** (production exit; costs never carried across models):
  x = 0 (Q(0) reused if Z1 confirms exact independence; otherwise evaluated under N); the reference unit (7: 0.25/1,
  2025); 7: 0.5/2 in 2025; the historical 2× incumbent {5: 0.25/0.5; 7: 1/3.5, 2030}. Four seeds, so 7 or 8 seed
  evaluations in total (the 2× aged cells of the unit and the incumbent exist on the final code as certified
  re-evaluations, rows 35 and 40, but the search uses production-exit values of the same protocol, so they are
  re-evaluated too; 8 if Z1 fails).
- **Incumbent rule:** best production-exit F among budget-feasible seeds (ties to the lower investment); acceptance of
  a poll point only when it beats the incumbent by more than max{bar_x + bar_inc, σ_Q = 18,449.66 €}.
- **Budget:** N^max = 30 new search evaluations per arm (seeds and cross-evaluations counted separately); at most 10
  polls; stopping reasons recorded (unit poll failed / budget exhausted / completion cap). No extension without a
  ruling; a run that stops on the budget is reported as such.
- **Cache:** per arm, per ageing model, seeded only by that arm's seed evaluations; no cross-model reuse.

### 5.2 Primary setting: twice the base flexibility price (D2)

Arm A (baseline cycling + calendar ageing, floor 0.70) and arm N (no cycling or calendar ageing, SoH ≡ 1, floor
inactive; the T8 `no_ageing` implementation, Z1 confirms it is the same code path in the search). Returned plans x_A
and x_N. Cross-evaluation, all **certified under rule v6 on the final code** (or reported uncertified with the bar):
x_A under A, x_A under N, x_N under A, x_N under N; x = 0 under A exists (row 34). The contrast F_A(x_N) − F_A(x_A) is
decided by rule v6 if both certified; if either is a gap-clause cell (the historical incumbent was), by the uncertified
form, and an unresolved difference is reported as unresolved.

**Count (2×):** seeds 7–8 + search ≤ 60 + certified cross-evaluations 4 = ≤ 72 evaluations. **Time:** seeds + search
≈ 68 × 0.75 h ≈ 51 h serial; cross-evaluations ≈ 4 × 1.5 h ≈ 6 h serial; total ≈ 57 h serial (≈ 2.4 machine-days),
≈ 20 h if search batches run on four slots after the concurrency check (§9). Worst case; a search that ends on a failed
unit poll before the budget is cheaper.

**Outputs.** The brief's table for x_A, x_N and x = 0 under A and under N: node(s), rating, energy, year, I, value,
margin, terminal E^Av and SoH per node; F_A(x_N) − F_A(x_A) with its threshold or bar; per arm: evaluations used,
polls, stopping reason, neighbourhood coverage at the incumbent, barrier points; the historical incumbent as a
reference row, labelled with its code state and uncertified status.

**Hypotheses (recorded; none is required).** H1a: x_N ≠ x_A, x_N with more energy at node 7 (at the 1 M€ budget or
the 5 MWh cap). H1b: F_A(x_N) − F_A(x_A) > 0 and resolved. H1c: the final-code aged search returns a plan within
resolution of the historical incumbent. Outcome → claim mapping in §8.

### 5.3 Baseline-price unit-neighbourhood screen (clarification 2)

Not a search and not a poll of Algorithm 1: the fourteen admissible unit neighbours of x = 0 (the smallest unit at any
non-empty subset of the three nodes, 2025 or 2030) evaluated under N at the production exit, on the final code, and
compared with Q(0) by the search's acceptance criterion max{bar sum, σ_Q}. Fourteen evaluations ≈ 10.5 h serial. The
reference unit's N cell (row 54, −4.1 k€ certified) is a fixed-plan datum beside it, not one of the fourteen (it is
not a unit neighbour). Reported as "a bounded unit-neighbourhood screen at the baseline price under the no-ageing
model": if none improves beyond the criterion, that is the scope of the statement; it is not a completed search and
not a globally preferred zero investment. **Hypothesis H1d:** no neighbour improves on x = 0 beyond the criterion.

### 5.4 Aged arm at the baseline price

Not re-run: the historical s47 search at 1× found x_A = 0 with all fourteen neighbours evaluated and eight
re-certified; the screen above gives the matching N-side evidence at the same fourteen points, and the paper's
baseline claim remains the one already supported ("no evaluated unit neighbour improves", with the scope stated).

## 6. E4 and E5 — deferred, claims narrowed (D1, D4, clarification 6)

**E5.** Salvage stays a post-evaluation sensitivity. The manuscript (§4.6, `app:salvage_credit`) and the letter
(R3.1) say: dispatch and investment selection do not internalise terminal value; the credit is applied after
evaluation; the comparisons show the sensitivity of the reference plans to the credit convention and do **not**
establish the economically preferred installation year under a terminal-value-aware objective; this is a partial
response to R3.1, with the scope rationale (a credit that depends on terminal SoH would enter the storage agent's
objective and every certified comparison behind the timing result). The horizon-end date and its SoH index are made
unambiguous (W182 C2).

**E4.** Deferred. Wording (D4): the two-stage stochastic formulation is retained where it describes the model; the
contribution list, abstract and §4.5 say that the planning searches were run on the single-scenario instance and the
3 × 3 instance evaluates fixed plans; no sentence implies a multi-scenario search or a demonstrated benefit of
stochastic selection; the investment-cost trajectories enter through probability-weighted coefficients (audit 3.1).
If E4-lite is ever pursued: three candidates × two configurations = six cells, two reusable (Table 13), so **four new
evaluations** (three expected-profile cells and one joint cell); v1's count was wrong.

## 7. E6 — physical illustration (zero-solve export, alongside)

Source: the reference unit at the baseline price, row 2 (certified, k\* 172), and x = 0, row 1. Three panels: (a)
charging/discharging power and SOC on one representative day with the 10–90 % limits of E^Av; (b) per-block change in
DN flexibility activation and TN conventional cost against x = 0; (c) E^Av per representative year and node with the
0.70 floor. Day-selection rule, declared: the (year, day) block with the largest value contribution Q(0) − Q(x) in the
certified decomposition — labelled an illustrative day, not a typical-day estimate; horizon totals remain
representative-day-weighted. Inputs: (b) and (c) from committed `component_levels_terminal.json` and
`evaluation_record.json`; (a) needs the schedule (uncommitted `ess_schedule_per_cycle.jsonl`, hashed in the W179c
manifest) or the SOC in `certified_models.pkl`. Z5 names the exact inputs and commits or hash-records them before the
export; the export runs with the solve guard armed at zero.

## 8. Claims each outcome supports (for the contribution list, abstract, §4, conclusions, letter)

| Study | Outcome | Supported claim (and no more) |
|---|---|---|
| E3 | H3a holds | "Coordination reduces cost by Q(P_NRF) − Q(C_NRF) (x %) under an identical no-reverse-flow restriction; permitting reverse flow adds Q(C_NRF) − Q(C_OPEN), within resolution." |
| E3 | H3a fails | Both conditional terms reported; the headline is the NRF-conditional saving; the reverse-flow term is a separate, stated contribution of the interface arrangement |
| E2 | H2 holds | "The paired investment margins and their verdicts at 1× and 2× are unchanged across initial penalty levels × 0.5 and × 2 (and, if stage 2 ran, a perturbed start); this is an empirical sensitivity under the declared procedure, not independence from initialisation or a bound on optimisation error." |
| E2 | H2 fails | Margins reported as a range over configurations; any planning statement whose verdict changed is downgraded to "unresolved under configuration changes"; E1 waits |
| E1 (2×) | x_N ≠ x_A, contrast resolved > 0 | "Under matched searches at twice the base price, planning without ageing returned a different (larger) plan whose cost under the aged model exceeds that of the degradation-aware plan by Δ (resolved)." Evidence about the two returned plans under the declared protocol; not a regret bound |
| E1 (2×) | x_N ≠ x_A, contrast resolved < 0 | Reported as found: the no-ageing search returned a plan that performs better under the aged model, which says the aged search's incumbent is not the best evaluated point, not that ignoring ageing is preferable |
| E1 (2×) | contrast unresolved, or either cell uncertified | "The two plans differ by less than the resolution / the uncertified bar"; no directional claim |
| E1 (2×) | x_N = x_A | "At this setting the searches returned the same plan"; the degradation claim stays the fixed-plan valuation claim |
| E1 screen (1×) | H1d holds | "Under the no-ageing model none of the fourteen unit neighbours of the plan without storage improves on it beyond the search criterion" — scope stated; not a completed search |
| E1 screen (1×) | H1d fails | The improving neighbour(s) reported; the baseline-price statement becomes "the fixed reference plan is at break-even within resolution without ageing and at least one unit neighbour improves on x = 0 under the no-ageing model"; a no-ageing search at 1× is a decision for the author, not an automatic extension |
| E5 (deferred) | — | Salvage as post-evaluation sensitivity; no preferred timing under a terminal-value-aware objective; partial response to R3.1 |
| E4 (deferred) | — | Formulation is two-stage stochastic; searches single-scenario; multi-scenario evaluates fixed plans; no value of stochastic selection claimed |
| E6 | — | Descriptive: dispatch, available capacity and value components for one illustrative day and the horizon |

Independently of outcomes, the contribution bullet "role of degradation in the investment decision" becomes "effect
of degradation on storage value and available capacity" unless E1 at 2× returns different plans with a resolved
contrast (audit 3.1).

## 9. Execution order, finite limits, stopping points

| Step | Content | Limit | Stop / decision |
|---|---|---|---|
| 1 | **W182** (zero-solve): re-audit of `edd7a63` (A1–A8), inventory Z1–Z7, pre-campaign readings C1–C4 | — | expert rulings; campaign spec v1 frozen with hashes |
| 2 | Concurrency check: one search-type evaluation on 1 slot and a batch of 4, bitwise gate, peak RSS and per-cycle time | 5 evaluations | fixes the slot count used for E1; serial if the gate fails |
| 3 | **E3** | 1 certified evaluation | report; §4.4 rewritten from the records |
| 4 | **E2 stage 1** | 8 | report; **stop if H2 fails**; stage 2 if Z6 allows (4) |
| 5 | **E6** export (zero-solve), alongside 3–4 | — | figure draft |
| 6 | **E1 at 2×**: seeds (7–8) → arm N search (≤ 30) → arm A search (≤ 30) → cross-evaluations (4) | ≤ 72 | report; stop for review on a barrier point (brief §3) or a completion-cap refusal |
| 7 | **E1 screen at 1×** | 14 | report |
| 8 | Manuscript, letter v10, cover letter, highlights, marked copy; W183 audit | — | submission |

**Totals (serial, worst case):** E3 2.5 h; E2 10 h (+5 h stage 2); concurrency check 4 h; E1 57 h; screen 10.5 h;
≈ 84–89 h ≈ 3.5–3.7 machine-days. With four slots for the search batches after step 2: ≈ 50 h. These replace v1's
"four machine-days" estimate, which omitted the matched aged arm; a search ending before its budget is cheaper. Every
new output goes under a new campaign root with its own spec hash; no archived artefact is re-run onto or overwritten;
each cell records candidate hash, code head, configuration hash and the stopping ratio (Evidence rules 1, 9, 12, 14).

## 10. Remaining technical uncertainties (resolved by W182 or stated in the paper)

1. Exact ageing-independence of Q(0) (Z1): if not exact, Q(0) is evaluated under N (one more seed).
2. Zero-capacity conventions at S^Av = 0 for the normalised complementarity, cohort allocation and degradation rows
   (audit 3.7): code reading in Z1; the manuscript states the convention (product form for the complementarity row;
   allocation and cycling defined for positive-capacity cohorts; inactive rows deactivated).
3. Whether a certified value of the historical 2× incumbent is reachable within its cap (Z2); otherwise the reference
   row stays uncertified and is not used in the E1 contrast.
4. Code head and configuration of every E.21 row (Z3) — decides which archived cells are "final-code" cells.
5. Applicability of the perturbed start to the coordinated procedure (Z6) — decides E2 stage 2.
6. The reverse-flow magnitudes, weights and prices in C_OPEN (Z7) — context for E3, not a bound.
7. Concurrency: per-cycle time and determinism under four slots on the final code (step 2).
8. Cost boundary and horizon-end date (C1, C2): stated once in the paper and used consistently (audit 3.3, 3.8).
9. The historical searches' code state remains a reportable caveat for the rows that keep them (Z3).

## 11. Amendments after W182 (v2.1, 2026-10-26; Addendum 83)

- **E2 stage 2 is deferred for this revision.** No perturbed coordinated start exists; the `price_taker`
  initialisation perturbs only the storage channel and would be mislabelled. The paper states E2 as a sensitivity to
  initial penalty levels only.
- **E1 launcher and keys.** A new launcher accepts `model_variant` + `flex_price_multiplier` together (untested
  combination: gate G0 checks it); `eol_retention_r` = 0.50 (T8's declaration) so row 54's key is reusable.
- **Gate G0** precedes E1: build-only NL identity at x = 0 with ageing on/off (Z1 was held by code reading only).
- **Concurrency decided inside E1:** the seven serial seed evaluations are the reference; the four node-7 seeds are
  re-run as one 4-slot batch and compared on the gated per-cycle fields; bitwise identity authorizes 4 slots for the
  search batches, otherwise serial.
- **NRF row** on the consensus exchange (`pc_adn`, `pg_adn`) on both copies, both DSO builders; declared per-network
  option, default off; NL identity with the option off is the gate.
- **Counts (replace §9):** E3 1; E2 8; E1 ≤ 75 at 2× (seeds 7, batch 4, searches ≤ 60, cross-evaluations 4); screen
  14. Serial worst case ≈ 80 h; ≈ 48 h with the batch gate passed.
- **E6 inputs:** two JSONL files (31.5 MB) committed by the author; the SoC from row 2's pickle under a guarded
  zero-solve read.
- **Convention:** "contracted interface settlement excluded" everywhere; at 3 × 3 the deviation settlement and the
  commitment premium are inside Q.
- **Reverse-flow context (Z7):** four hours at node 7 in 2025 (−3.157, −2.402, −1.602, −0.067 MW; weights 455–460
  days; prices 118–162 €/MWh): the directly priced energy of those hours is of the order of 1 M€ over the horizon,
  which is context for H3a, not a bound on the cost of imposing NRF.

## 12. Amendments after W183 (v2.2, 2026-10-26; Addendum 85)

- **E1 poll rule B+ (replaces the "completion cap" sentence of §5.1).** Variant B as frozen, with one declared
  amendment, identical in both arms and at every poll: when fewer than n + 1 = 8 distinct feasible poll points remain
  after rounding and snapping, the unit-box completion is used if it has ≤ 30 feasible points; if it has more, the
  Halton index advances (at most 10 consecutive indices) and the 2n directions are re-drawn until at least n + 1
  distinct feasible points remain; if no index in the window does, the poll is refused for review as frozen. Reason,
  from W183b's zero-solve geometry table: the completion of the reference unit (31 points), of 7: 0.5/2 (34) and of
  the historical incumbent (61) exceeds the cap and the arm budget, and at k = 0 every storage seed has 6–7 distinct
  points, so the frozen rule would stop every storage incumbent at poll 0. The rule reads the snap table only (no
  objective value) and was declared before any E1 solve; the paper states it in one sentence where E1 is reported;
  the certificate scope stays as frozen (full-box only when the completion was used). Start index k₀ = 0 (t = 17) in
  both arms; the A7 barrier rule (2 per poll / 3 overall) confirmed.
- **Screen (§5.3): 15 evaluations** — the 14 unit neighbours of x = 0 under N at 1× plus x = 0 at 1× on the final code
  (evaluated once; the N alias by G0); row 1 and row 54 reported beside as fixed-plan data, not as the comparator.
- **Certified protocol for E3, E2 and the cross-evaluations:** the ext-v6 `settling_resettle` v6 declaration (spec
  `10f4aa7a`) on the final code; a returned E1 plan that is a seed is still re-evaluated under it.
- **E2 (§4) adds a per-cell runtime check:** the cycle-0 `rho_{v,pf,ess}_after` equal S39 × s (times the recorded
  adaptation when the action is not a hold). **E3 (§3) adds:** the terminal NRF capture (864 active rows per copy;
  binding hours by the expected-value rule and by the per-block row-body measure, which coincide at one scenario;
  per-node minima; tolerance 1e-3 MW) and hypothesis H3c: the four C_OPEN reverse-flow hours are among the binding
  hours of C_NRF.
- **Order (§9, D5):** W185 (E2) follows W184 (E3) without waiting for the §4.4 text, which is rewritten in round 9 from
  E3's records; the E2 stop rule stands; E1 and the screen wait for Addendum 86. W184 and W185 are authorized
  conditionally on W183g's zero-solve gates (Addendum 85).
- **Two-cycle bitwise gate:** read under the standing rule that keys absent in the reference and null in the new arm
  are field-list drift (the 22 W64 keys), with the W48 whole-arm reference kept and a zero-solve re-comparison
  against the W83 arm; the solve-bearing gate covers the sequential DSO builder, the only one the configuration runs.
- **E6 (§7):** panel (b) in € (TN conventional cost and DN flexibility cost per block against x = 0); declared day
  2025 Spring (40,181.22 €, 15.8 % of the value, determinate by 118×); the figure is drawn in round 9 from
  `P515S55/e6_export/e6_export.json`.
- **Counts (replace §11):** E3 1; E2 8; E1 ≤ 75; screen 15; serial worst case ≈ 81 h; ≈ 49 h with the batch gate
  passed.
