# P5.15 W174b — re-audit of the W171b rows (§2.1–2.3) and the W172 rows (Appendix A) at Overleaf `260bd83`

**Worker, 2026-10-08. ZERO SOLVES, zero model builds, zero pickle loads; read-only on production code, specs, the frozen
JSON and the manuscript clone.** Manuscript: `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `260bd83`,
sha256 `cb237b2d129e689a94dd50abec9a9cca739567ef97a486ec012a901f0e04b6df` (verified), 2,209 lines; `main.tex` and
`bibliography.bib` unmodified in the clone; nothing pulled or committed there. Manuscript lines below are at `260bd83`;
code lines are at repo HEAD `a8eb9b69`. The production files are identical at every table campaign's head (section
"Code identity") and at `f7d165b7` (W175, committed during this task; it touched no production file).

Helper `p515_s53_w174b_reaudit_checks.py`:
- `SolveProfileGuard(permitted=())` verified at exactly 0.
- `pickle.load`/`loads` blocked; 0 calls.
- No production module imported (asserted).
- Exit 0.

W100 typing test: PASS (129,318 files, 0 failures).

Impact scale, as in W171b and W172:
- **H**: a referee re-deriving from the text gets a different model or number.
- **M**: imprecise, but not a different model.
- **L**: notation or wording.

---

## Blocked on Planner

Nothing blocks the audit. Four items need a ruling or a forward to the expert, author or Overleaf editor:

1. **Two edits made while pasting round 3 into Overleaf go beyond the correction blocks.** Both are text lost, not wording.
   - **NEW-1 (M), App. A.4 l. 1640.** Pasting correction (m) deleted the clause "and the acceleration is switched off on
     any cycle in which every channel passes and, in the certifying regime,".
     - As a result, Appendix A no longer states when Anderson acceleration is off. In the code it is off on every
       cycle in which all channels pass (`admm_anderson_acceleration.py:477`), and it is forced off from k0 + 1
       (`p515_s53_w118_resettle_hooks.py:526, 722`).
     - The surviving "from the cycle after k0 on" now attaches to the clearing of the AA memory. The code clears that
       memory at any cycle (`shared_resources_planning.py:3422`; `admm_anderson_acceleration.py:398, 416`).
     - Algorithm 2, l. 1697, still says "(unless off)".
   - **NEW-2 (L), Algorithm 2 l. 1683–1702.** The "$k \gets k + 1$" line was deleted together with the lines that
     correction A.2(c) replaced, so the while loop has no increment.

   Both are restorations, not new wording. Recommendation: restore the clause in the form "…on any failed local solve;
   the acceleration is switched off on any cycle in which every channel passes and, in the certifying regime, from the
   cycle after k0 on (or after the earlier run's stopping cycle in a continued evaluation)", and restore the increment.
2. **NEW-3 (M), l. 449: "the runs reported here stopped at their evaluation budgets" is false for all three searches.**
   This sentence is round-2 text (A.3).
   - s47 terminated when its unit poll failed, having used 14 of 20 evaluations.
   - s51 stopped for review at the completion cap (61 > 30), with 20 of 20 used.
   - s53 terminated when its unit poll failed, having used 7 of 60.
   - No run ended with `evaluation_budget_exhausted`.

   The sentence also contradicts the certificate statements in the same paragraph. This is an expert wording call.
3. **T1-4 residual, rated M: l. 647 (§2.2.7, "Determinacy") still says "the master search … accepts a poll point as
   an improvement only by a determinate margin".**
   - Algorithm 1 (l. 434) and l. 449 now state the rule as it was run: F(inc) − F(x) > max(bar_x + bar_inc, σ_Q),
     with production-exit evaluations (`p515_s47_phase_b_record.py:519`).
   - I rate it M because the text contradicts itself while the algorithm carries the rule as run. Read alone, the
     sentence is an H statement.
   - **The prediction outcome below depends on this rating.**
4. **NEW-8 (L), l. 756 and l. 866: wrong cross-references.** They send the reader to §3.6 (`sec:case_settings`) for η,
   SoC^Min/Max/0, ε^Cl, c^Cl, ε^C, c^σ and ε^E, but those values are printed in §3.5 (`sec:case_ess_params`, l. 1070).
   This follows mechanically from round-3 §C's rule "every 'Section 3.5' → \ref{sec:case_settings}". The α reference
   (l. 907) and the retry reference (l. 1644) are correct.

## Prediction vs outcome

| prediction (expert, Addendum 70) | outcome |
|---|---|
| "no H row remains in §2 or Appendix A" | **HELD on these ratings: 0 H in §2.1, §2.2, §2.3 and Appendix A.** All 10 W171b H rows and the W172 H row (A6-1) are resolved. The nearest to H is T1-4's residual at l. 647, rated M (Blocked 3). If the Planner rates it H, the prediction fails for that one row. |

Counts of the W171b and W172 rows plus the three W173 extras. Each row with a status other than resolved is listed
with its remaining impact.

| section | rows | resolved | partly resolved | changed into a new difference | unresolved | remaining H / M / L |
|---|---|---|---|---|---|---|
| §2.1 incl. Algorithm 1 (W171b) | 12 | 10 | 2 (T1-3 L, T1-4 M) | 0 | 0 | 0 / 1 / 1 |
| §2.2 incl. §2.2.7 (W171b) | 11 | 6 | 3 (T2-1, T2-4, T2-5; all L) | 0 | 2 (T2-9, T2-11; L) | 0 / 0 / 5 |
| §2.3 (W171b) | 11 | 7 | 0 | 1 (T3-9 L) | 3 (T3-7, T3-8, T3-10; L) | 0 / 0 / 4 |
| Appendix A (W172) | 21 | 13 | 2 (A3-3, A4-3; L) | 6 (P-1, A1-3, A4-5, A5-2, A5-4 L; A4-6 M) | 0 | 0 / 1 / 7 |
| W173 extras (Algorithm 1) | 3 | 3 | 0 | 0 | 0 | 0 / 0 / 0 |
| **total** | **58** | **39** | **7** | **7** | **5** | **0 / 2 / 17** |

Three further new differences are attached to no row (NEW-3 M, NEW-4 L, NEW-8 L). Across all rows and new differences
the totals are **H 0, M 3, L 19**.

---

## Row status — W171b rows (§2.1–2.3)

### §2.1 — architecture and Algorithm 1

| ID (W171b impact) | current text at `260bd83` (line) | status | code at HEAD | impact if not resolved |
|---|---|---|---|---|
| T1-1 (H) | eq. (operational_recourse_function), l. 356–362: min over u⁰, {u_o} of Σ_o ω_o C_o(x, u⁰, u_o), with (u⁰, u_o) ∈ U_o(x). Items at l. 372–373 define u⁰: the storage schedule, the TSO interface exchange, the DSO committed import. L. 379: "the day-ahead decisions u⁰ do not [adapt] … itself a two-stage problem" | **resolved** | `model_construction_helpers.py:841, 846` (one scenario-free storage copy per block); `shared_resources_planning.py:4241` (TSO pc fixed at the initialisation exchange) and `:4270` (scenario-free Δ bounded by ±R^I); `model_construction_helpers.py:1877, 2416` (row 18; P̄ a block variable) | — |
| T1-2 (H) | l. 413: z = (S_e/Δ^S, E_e/Δ^E)_e × y^Inv ∈ ℤ^{2\|E^S\|+1}, one investment year common to all nodes. L. 449: "per-node timing and staged capacity … were not searched" | **resolved** | `p515_s47_phase_b_record.py:191` (`N_VARS = 2 * len(ACTIVE_NODES) + 1`) | — |
| T1-3 (H) | l. 421–430. Variant A: n + 1 OrthoMADS points z + Δv, inadmissible ones dropped; at Δ = 1 add the full ℓ∞ box, stop for review if \|P\| > 30. Variant B: 2n points z ± v, snapped to the unit frame; completion if fewer than n + 1. L. 435: success sets Δ ← 2Δ (A) or 1 (B) | **partly resolved.** The poll of each variant now matches the code. The direction scaling and rounding are not stated: the algorithm writes z + Δv with v a Householder column. Only l. 449 says "rounded". The s53 six-level snap tie-break is unstated; the CONFIRM comment calls it an implementation detail. At Δ = 1, \|P\| equals the completion count, because an admissible unit direction is a box point; so "\|P\| > 30" matches the code's test | A: `:194` (Δ₀ = 4), `:499` (`round_half_away(delta * a / m)`), `:502`, `:650` (cap), `:720` (×2), `:725` (terminate), `:730` (halve). B: `p515_s53_f2_certificate.py:321` (`directions_2n`), `:345` (`snap_key`), `:470`, `:474` (completion and cap) | **L** |
| T1-4 (H) | l. 432 (production exit, cap 500; bar = largest step over the last 10 cycles); l. 434: F(inc) − F(x) > max{bar_x + bar_inc, σ_Q}; l. 449 (search acceptance, then re-evaluation under §2.2.7). **L. 647: "the master search of Algorithm 1 accepts a poll point as an improvement only by a determinate margin"** | **partly resolved.** Algorithm 1 and l. 449 match the code. The l. 647 sentence, which W171b listed in this row, was not touched. "Determinate" is defined in that same paragraph as ≥ max{3b, 2τ}, so the paper states two acceptance rules | `p515_s47_phase_b_record.py:519` (`max(bar_x + bar_inc, sigma_q)`); `p515_s44_campaign_harness.py:1910` (`_max_step_last_n`) | **M** (internal contradiction; H if l. 647 is read alone) |
| T1-5 (H) | l. 449: x = 0 has "every admissible unit neighbour … (fourteen, the full box…)". F2 has "seventeen of its 61 … (the ten points of the final poll and seven earlier evaluations) and none is determinately better; of the thirteen re-evaluated … better than twelve … within resolution of the thirteenth — not … a mesh-local optimum" | **resolved** (W173 counts) | `p515_s53_f2_certificate.py:246` (CERTIFICATE_STATEMENT); `campaign_s47_phase_b` / `campaign_s53_f2_certificate_r1` `termination_certificate` | — (see NEW-3 and NEW-4 for new sentences in this paragraph) |
| T1-6 (M) | l. 414–417: cache = the designed initial search "run with and without the budget"; x^inc ← argmin F over the certified, budget-feasible points, x = 0 included, ties to the lower investment | **resolved** | `p515_s47_phase_b_record.py:538` (`initial_incumbent`) | — |
| T1-7 (M) | l. 405 "budget of new evaluations N^max"; l. 419 "While polls remain (at most 60) and a poll can be launched"; l. 431 the budget refusal; l. 433 "N ← N + new evaluations" | **resolved** | `:200` (20), `:201` (MAX_POLLS 60), `:603` (loop), `:663` (refusal); s53 `:156` (60), `:157` (MAX_POLLS = PB's) | — |
| T1-8 (M) | l. 416 "a cache hit is read, never re-evaluated"; l. 429 "fewer than n+1 distinct admissible points" | **resolved** (cache hits stay in the poll) | `p515_s47_phase_b_record.py:636` (`cache_hit`) | — |
| T1-9 (M) | l. 432 "…is a barrier point, F = +∞ (two in one poll, or three in all, stop the search for review)" | **resolved** | `:202–203`, `:685`; `p515_s44_campaign_harness.py:2017` (status ≠ certified → barrier) | — |
| T1-10 (M) | the l. 451 sentence (at `42794d4`) is gone. L. 449: "afterwards re-evaluated on the final code under the certification rule" | **resolved in §2**. §3.6 l. 1107 now says "Every recourse evaluation ran … under one frozen configuration"; that is W174a's scope (Unexpected findings 1) | three search heads differ from HEAD (Code identity, informational) | — |
| T1-11 (L) | l. 391 "the value Q(x) … and the record of its evaluation (its bar and exit), nothing else" | **resolved** | `:519` | — |
| T1-12 (L) | the "one certified evaluation per canonical plan" wording is gone (l. 414–416) | **resolved** (no claim on the key now) | `:636`; key = evaluation key | — |

### §2.2 — master problem and §2.2.7

| ID (W171b impact) | current text (line) | status | code at HEAD | impact if not resolved |
|---|---|---|---|---|
| T2-1 (H) | l. 639: "Every certificate reported in this paper was decided, or re-decided from its committed record, under this rule, and certifies at the cycle reported". L. 624: holds "from the next cycle in a fresh run, **or from the stopping cycle of the earlier run** in an evaluation continued from one" | **partly resolved.** The certificate claim matches the records (E3 below). **Residual:** the code holds from the cycle *after* the earlier run's stopping cycle. L. 1640 says "after", consistent with the code; l. 624 and l. 1625 say "from" | frozen tables `590088fe`: 36 v6, 2 v4, 1 v5, 7 v2, 2 v1 certified cells, plus the superseded pb_y2025_n5. `w142_v6_from_records.json`: v6 reproduces every committed k* (v2 172/174/182/195/192/169/167; v4 173/173; v5 148; v1 unit 172; x0 181), and W168 gives x0 181. Holds: `p515_s53_w101_settling_continuation_hooks.py:589` (`hold = iter > st.n`); `p515_s53_w118_resettle_hooks.py:526` (`c > self.first_pass`) | **L** (off by one) |
| T2-2 (H) | l. 630: P̂ is "the number of cycles spanned by the three most recent turning points" | **resolved** | `settling_criterion_v2.py:166` (`p_hat = T[-1][0] - T[-3][0]`) | — |
| T2-3 (H) | l. 639: settling slack = "the objective's movement between the certification cycle of its earlier, residual-based run and the cap" | **resolved** | `p515_s53_w132_resettle_v3_campaign.py:1000` (`s_signed = q_end - q_n_old`) | — |
| T2-4 (M) | l. 624 (lapse or failed solve resets k₀ and the turning-point record, holds stay); l. 630 (τ/100 sign floor; the first three cycles excluded); l. 632 "last n_w cycles, **all after k₀**" | **partly resolved.** All four clauses are now stated. **Residual:** the oscillatory window must satisfy lo ≥ k₀ (it may start at k₀); "all after k₀" excludes k₀ | `settling_criterion.py:58–59` (EPS0, K_EXCL); `settling_criterion_v2.py:244` (reset), `:258` (k < k₀ + K_EXCL), `:169` (`window_inside_run: lo >= self.k0`) | **L** |
| T2-5 (M) | l. 639: window of 2P_max cycles "**starting after k₀**"; steps decreasing as defined; range ≤ τ; \|last step\|·2P_max ≤ τ; "**conditions 4 and 5 hold on that window**" | **partly resolved.** The no-sign-change window, the half-window test and the clean veto are now stated. **Residuals:** (i) the code requires the window to start at ≥ k₀ + 3; (ii) condition 4 (the gap) is tested at the certifying cycle only, not over the window | `settling_criterion_v2.py:177` (`lo >= self.k0 + K_EXCL`), `:192` (steps decreasing), `:277` (`gap_ok` at k); `settling_criterion_v6.py:294` (monotone clean veto on the window) | **L** |
| T2-6 (M) | l. 647: "…exceeds three times the largest of their consensus gaps and settling slacks, in both gross and gap-corrected terms" | **resolved** | `p515_s53_w132_resettle_v3_campaign.py:1077` (`det = abs(d_q) > bar and m_cc > bar`) | — |
| T2-7 (M) | l. 639: "its reference evaluation was continued past that exit … the ratio of Section 4.5 is reported as a band" | **resolved** (one cell continued; the band is on the ratio) | brief Addenda 51–52; frozen `three_by_three` | — |
| T2-8 (M) | l. 618: TN generation at the scenario's market price, DN flexibility, load curtailment at its price, closure and feasibility slack penalties "at their solver bound"; l. 619 "Renewable curtailment carries no cost" | **resolved** | `model_construction_helpers.py:1811` (objective), `:2001` (`generation_cost`); `shared_resources_planning.py:5332` (RES curtailment penalty 0) | — |
| T2-9 (L) | l. 634: "within ten times the tight-tail tolerances" | **unresolved.** For the ESSO blocks the factor 10 applies to the ESSO's own tolerances; neither §2.2.7 nor §3.6 says so | `settling_criterion_v5.py:90` (`'esso': {'tol': 1e-10, …, 'compl_inf_tol': 1e-4}`) | **L** |
| T2-10 (L) | l. 647 "when it is at least max{3b, 2τ}" | **resolved** | `settling_criterion_v6.py:371` (`abs(margin) >= thr`) | — |
| T2-11 (L) | l. 497, 504: lower limit max(y − T^Cal_{e,y} + 1, y₀) | **unresolved** (the cohort's lifetime T^Cal_{e,y^Inv}; equal values) | `shared_energy_storage_data.py:564` (`tcal_norm` per cohort y_inv) | **L** |

### §2.3 — subproblem

| ID (W171b impact) | current text (line) | status | code at HEAD | impact if not resolved |
|---|---|---|---|---|
| T3-1 (H) | l. 664: "The net storage schedule — active and reactive power … — is a consensus variable … Each model keeps its own split …; the split that ages the cells is the agent's"; l. 756 "the net schedule the agent ages" | **resolved** | `model_construction_helpers.py:999` (network P_net = pch − pdch); `shared_energy_storage_data.py:782` (agent P_net = Σ(pch − pdch) + σ⁺ − σ⁻), `:787` (Q in the circle) | — |
| T3-2 (H) | l. 877–906: s = (m, o), ω_s = ω_m ω_o; P̄ = Σ_{s∈Ω_M×Ω_O} ω_s P^I_{i,s,t}; deviations per (i, s, t); charge Σ_s Σ_t ω_s α π̄_t (d⁺ + d⁻); "With a single scenario … vanish identically" | **resolved** (no Σ_o left in the subsection) | `model_construction_helpers.py:1877` (`add_scenario_commitment_terms`), `:1917` (`if n_scenarios == 1:` not constructed), `:2416` (`dn_interface_expected_pf_p_def`) | — |
| T3-3 (L) | l. 711: P^E = P^Ch − P^Dch "(positive when charging)", "consumption convention shared with the agent" | **resolved** | `model_construction_helpers.py:999`; `shared_energy_storage_data.py:782` | — |
| T3-4 (L) | l. 733: 0 ≤ s^± ≤ ε^Cl E^Av. §3.5 l. 1070: "(plus a numerical allowance of 10⁻⁵ p.u.)" | **resolved** (via §3.5, as round-1 A.3(e) assigned). L. 756 points to §3.6 for ε^Cl: NEW-8 | `model_construction_helpers.py:1166` (`… * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE`) | — |
| T3-5 (L) | l. 711 "from the constant pre-day state E^SoC_{e,y,d,0}"; l. 731 E^SoC_{e,y,d,0} = SoC⁰ E^Av | **resolved** | `model_construction_helpers.py:973` (`soc_prev = … * ENERGY_STORAGE_RELATIVE_INIT_SOC`) | — |
| T3-6 (L) | l. 755 "…in the objective of each network model that carries the storage" | **resolved** | `model_construction_helpers.py:1818` | — |
| T3-7 (L) | l. 693, 695: lower limit max(y − T^Cal_{e,y} + 1, y₀). L. 669 now uses T^Cal_{e,y^Inv} | **unresolved** (index; the text is now inconsistent between l. 669 and l. 693) | `shared_energy_storage_data.py:564` | **L** |
| T3-8 (L) | l. 804 (365 Y_y Ē / 2kE), l. 816 (φ^{Y_y}) | **unresolved** (the code uses Y_{y^Inv}; equal values) | `shared_energy_storage_data.py:678` (`num_years = …years[repr_years[y_inv]]`) | **L** |
| T3-9 (L) | l. 841: "\eqref{eq:soh_chain} the available-energy product in \eqref{eq:cohort_available} and the converter circle are the agent's nonlinear rows" | **changed into a new difference (NEW-10).** The circle is now listed. The paste moved the "and", leaving "soh_chain the available-energy product …" without a separator (`42794d4` l. 831 read "soh_chain and the available-energy product") | `shared_energy_storage_data.py:787` | **L** (wording) |
| T3-10 (L) | l. 840: "its economic effect appears through the available energy of later years and through the floor" | **unresolved** (with the end-of-year SoH, a year's own throughput also lowers that year's available energy) | `shared_energy_storage_data.py:17` (`'end'` default) | **L** |
| T3-11 (L) | l. 653: "a dedicated shared-ESS subproblem per interface node, spanning all representative years and days" | **resolved.** The untouched next sentence ("further decomposed across representative years and days") now reads as applying to the TSO/DSO models | `shared_energy_storage_data.py:74` (`build_subproblem`, one NLP per node) | — |

## Row status — W172 rows (Appendix A)

| ID (W172 impact) | current text (line) | status | code at HEAD (production identical to W172's; W172's line numbers still hold) | impact if not resolved |
|---|---|---|---|---|
| P-1 (L) | l. 1497: "run as a sweep (DSOs, then TSO, then the storage agent) that is Gauss–Seidel on the interface channels and, on the storage channel, a global-variable consensus in which all three agents solve against the same z before it is updated with residual balancing of the penalty parameters, Anderson acceleration and a tightened interior-point tail" | **changed into a new difference (NEW-5).** The substance matches the code. The original continuation "with residual balancing …" now attaches to "updated", so it reads as if z were updated with residual balancing | `shared_resources_planning.py:3115–3210` (sweep), `:8687` (z after the agent) | **L** |
| A1-1 (L) | l. 1504: "…the sum over blocks, after the exclusions of Subsection certification, is the discounted operating cost" | **resolved** | `shared_resources_planning.py:1302–1304` | — |
| A1-2 (M) | l. 1508: "Σ_{s∈Ω_M×Ω_O} ω_s P^I_{i,s,t}" | **resolved** | `model_construction_helpers.py:2416` | — |
| A1-3 (L) | l. 1514–1515: "The storage agent's augmented-Lagrangian terms are multiplied by κ^E …, which puts the agent's local objective on the same footing as a median block's scaled objective; **the consensus terms themselves are unscaled in every agent**" | **changed into a new difference (NEW-6).** The added clause contradicts the preceding clause and eq. (admm_esso_local) (l. 1562), where κ^E multiplies the agent's consensus terms. It holds only for the argmin-equivalent form f_E/κ^E + AL | `shared_resources_planning.py:5515` (`obj += …admm_esso_al_scale * (…)`) | **L** |
| A2-1 (L) | l. 856: κ^E(ℒ^{E,P} + ℒ^{E,Q}); l. 865 defines ℒ and κ^E | **resolved** | `shared_resources_planning.py:5515` | — |
| A3-1 (M) | l. 1606: "minimiser over z of the three agents' consensus terms taken with unit weight; the factor κ^E … does not enter the update" | **resolved** | `shared_resources_planning.py:8687` (`z_new = numerator / denominator`, no κ) | — |
| A3-2 (L) | l. 1621: own copy kept on failure; interface duals need TSO and DSO success; z and the storage duals need all three | **resolved** | `shared_resources_planning.py:8489, 8499` (interface gate); storage gate 8602–8609 (W172) | — |
| A3-3 (L) | l. 1625: "…frozen from the cycle after the first residual pass, **or from the stopping cycle of the earlier run** in an evaluation continued from one" | **partly resolved.** The continuation case is now stated, but one cycle early (as in T2-1) | `p515_s53_w101_settling_continuation_hooks.py:589` (`hold = iter > st.n`) | **L** |
| A4-1 (L) | l. 1636: "s_E … taken over the three agents (so √3 ρ^E α‖Δz‖)" | **resolved** | `shared_resources_planning.py:7227` | — |
| A4-2 (L) | l. 1636: "with λ the DSO-side dual on the interface channels and all three duals on the storage channel" | **resolved** | `shared_resources_planning.py:7203` (`y_pf = lambda_dso_pf / …`) | — |
| A4-3 (L) | l. 1636: "…stopped by the certification rule … or by its cycle cap, whichever came first; **the holds start at the first passing cycle**, while the rule's own k₀ resets on a lapse" | **partly resolved.** The cap and the k₀ reset are now stated. **Residual:** the holds start at the cycle *after* the first pass (after N_old in a continued evaluation). L. 624, 1625 and 1644 say "next" or "after" | `p515_s53_w118_resettle_hooks.py:526` (`c > self.first_pass`) | **L** |
| A4-4 (L) | l. 1640: AA on z and the λ_a/ρ on the storage channel; on the TSO copy and the DSO-side λ/ρ on the interface channels | **resolved** | `admm_anderson_acceleration.py:302` | — |
| A4-5 (L) | l. 1644: "restores each operator's own tolerance otherwise (the tail is a declared option, **enabled in every campaign**)" | **changed into a new difference (NEW-7).** The tolerance clause matches the code. "Enabled in every campaign" is false for the three master-search campaigns: their specs declare no tail, and the tail did not exist in the code at their heads (E2) | `admm_parameters.py:331` (default off); `p515_s44_campaign_harness.py:200` (off unless declared) | **L** |
| A4-6 (L) | l. 1640: "The memory is cleared on any change of a penalty parameter and on any failed local solve, from the cycle after k₀ on (or after the earlier run's stopping cycle in a continued evaluation)." L. 1644: "the certifying regime holds the tail on from the cycle after k₀" | **changed into a new difference (NEW-1).** The tail's one-cycle offset is fixed. The AA-off rule was deleted, and the hold clause now attaches to the memory clearing. Also, the tail hold in a continued evaluation (after N_old) is still unstated | `admm_anderson_acceleration.py:477` (off when all pass), `:398, :416`; `shared_resources_planning.py:3422`; `p515_s53_w118_resettle_hooks.py:526, 722` | **M** |
| A4-7 (L) | l. 1644: retried on "the iteration limit, infeasible or … a solver error … in two tiers, a cold restart with the agent's recovery settings and the same with the adaptive barrier strategy (§3.6)" | **resolved** | `network.py:771` (`_is_recoverable_network_failure`), `:999` (tier 2 `mu_strategy`); `shared_energy_storage_data.py:1218`, `:1082` | — |
| A5-1 (L) | l. 1658: "after the initialisation solves, the plan x enters a network model only through these capacities" | **resolved** | `shared_resources_planning.py:4204, 4360` (plan at initialisation); `:2980, :3697` (publication) | — |
| A5-2 (L) | l. 1678–1680: "Convert every model … with ρ_g ← ρ_{g,0} (**in the multi-scenario instance the commitment charge and the settlement are activated here**); z ← the average…; storage duals at zero; interface duals set by one dual-ascent step from zero…" | **changed into a new difference (NEW-9).** The order now matches the code. The interface settlement is switched from weight 0 to 1 at this point in **every** instance, single-scenario included; only row 18 is multi-scenario | `shared_resources_planning.py:2925–2926` (prepare objectives, no scenario guard) → `:5050, :5340` (`interface_settlement_weight.set_value(1.00)`); `model_construction_helpers.py:1701` (initialised at 0); `:2931` (convert), `:2934` (z), `:2951` (dual step) | **L** |
| A5-3 (L) | l. 1696: "record the objective, residuals and solve statuses … (the priced interface gap is recorded by the campaign harness)" | **resolved** | production `admm_diagnostics` has no gap field (W172) | — |
| A5-4 (M) | l. 1697–1701: "Apply the Anderson step (unless off); balance ρ_g (unless frozen) and clear the acceleration memory if any ρ_g changed; set the tight tail…; If the stopping rule … exit" | **changed into a new difference (NEW-2).** The order matches the code. The loop increment "k ← k + 1" was deleted | `shared_resources_planning.py:3275` (AA), `:3406` (ρ), `:3422` (clear), `:3395` (tail next state), `:3390` (exit predicate), `:3711` (exit), `:3066` (loop counter) | **L** |
| A6-1 (H) | l. 1714: "At a single scenario the commitment charge and the voltage regularisation are not constructed; the interface settlement remains in every local objective at full weight and is excluded from the reported cost as a transfer" | **resolved** | `model_construction_helpers.py:1917` (not constructed at one scenario), `:1823` (settlement in the objective); `shared_resources_planning.py:5050, 5340` (weight 1); `:1302–1304` (excluded from Q) | — |
| A6-2 (L) | l. 1714: "the DN's expected exchange at initialisation, held fixed, plus a scenario-free adjustment bounded by the interface rating … (the same construction at one scenario)" | **resolved** | `shared_resources_planning.py:4241, 4270` | — |

## Row status — W173 extras (Algorithm 1)

| ID | current text (line) | status | code at HEAD |
|---|---|---|---|
| W173 E1.6-3 (l. 401: Δ₀ not an input) | l. 403 KwIn "initial poll size Δ₀"; l. 417 "Δ ← Δ₀ (Δ₀ = 4 lattice units in variant A; Δ = 1 throughout in variant B)"; §3.6 l. 1134 "Δ₀ = 4" | **resolved** | `p515_s47_phase_b_record.py:194` (`DELTA_0 = 4`) |
| W173 E1.6-2 (l. 412: "While N < N^max") | l. 419 "While polls remain (at most 60) and a poll can be launched"; l. 431 refusal when the new points exceed N^max − N | **resolved** | `:201` (MAX_POLLS 60), `:603`, `:663`; s53 `:157` |
| W173 E1.6-1 (l. 423: completion "until n+1") | l. 429 "add every admissible unit neighbour (the poll is refused for review beyond 30 points)" | **resolved** | `p515_s53_f2_certificate.py:470` (trigger), `:474` (`over_cap`), `:155` (cap 30) |

---

## Correction-block presence

Method: each block was searched in the whole of main.tex with all whitespace removed (`w174b_checks.json`
`B_correction_blocks`; the ```latex blocks are parsed from the two files in order, and the inline find → replace
strings are transcribed verbatim in the script). Where a replacement names the old text, the old text was checked
absent. Result: 67 checks present verbatim; 2 present with changes (the two round-2 blocks that round 3 rewrote,
verbatim once the round-3 edits are applied); 1 absent as written (C.5(a), present merged); 2 checks of deleted text
pass.

### Round 2 (`STEP6_ROUND2_CORRECTIONS.md`)

| block | where | status |
|---|---|---|
| A.1 subequations (recourse) | l. 356–362 | present verbatim |
| A.1 "Here" items (x, u⁰/u_o) | l. 372–373 | present verbatim |
| A.1 sentence "By contrast, the network dispatch…" | l. 379 | present verbatim |
| A.2 Algorithm 1 | l. 394–442 | present with changes. These are exactly round-3 B.a–c (the KwIn item "initial poll size Δ₀", the Δ₀ clause, the While line, the variant-B completion), plus braces and line breaks in \KwIn/\KwOut. With B.a–c applied: verbatim |
| A.3 paragraph after Algorithm 1 | l. 448–450 | present with changes. These are exactly round-3 B.d. With B.d applied: verbatim |
| A.4 coupling sentence | l. 391 | present verbatim; old text absent |
| B.1 stopping and certification | l. 624 | present verbatim |
| B.2 items 1–2; item 3 "cycles, all after k₀," | l. 630–632 | present verbatim |
| B.3 monotone-branch paragraph | l. 638–640 | present verbatim |
| B.4 (a) uncertified determinacy; (b) "at least" | l. 647 | present verbatim; old text absent |
| B.5 (a) Q component clause; (b) "Renewable curtailment carries no cost…" | l. 618, 619 | present verbatim; old clause absent |
| C.1 2.3 intro paragraph | l. 663–665 | present verbatim |
| C.2 (a) sign convention; (b) "net schedule" | l. 711, 756 | present verbatim; "(positive when discharging)" absent |
| C.3 (a) E^SoC_{e,y,d,0} in eq. soc_closure | l. 731 | present; the block's `$…$` are correctly dropped inside the equation; t₀ absent |
| C.3 (b) "from the constant pre-day state"; (c) "each network model that carries the storage"; (d) closure-slack sentence | l. 711, 755, 757 | present verbatim; the l. 747–748 CONFIRM-W169 comment is gone |
| C.4 (a) σ "at its lower bound at every certified point"; (b) "and the converter circle are the agent's nonlinear rows" | l. 762, 841 | (a) present verbatim. (b) present verbatim, but the original "and" before "the available-energy product" was dropped (row T3-9) |
| C.5 (a) "It is recorded, not enforced; … below 4 × 10⁻⁵ … The slack pair σ …"; (b) "checked after every solve and recorded" | l. 866 | present with changes: (a) and (b) merged into one sentence, "…is checked after every solve and recorded, not enforced; at the certified points it was below 4 × 10⁻⁵ of the rating. The slack pair σ was at its lower bound at every certified point." The meaning is unchanged. The CONFIRM-W169 comment is gone |
| C.6 (a) scenario s = (m, o) sentence; (b) (∀ i, s, t); (c) single-scenario sentence; sums over s | l. 877, 878, 906; eqs. l. 883, 896 | present verbatim; no Σ_{o∈Ω_O} left in §2.3.6; old sentences absent |
| C.7 "a dedicated shared-ESS subproblem per interface node, spanning all representative years and days" | l. 653 | present verbatim |
| C.8 k(C4) = 22,429 | §3.5 table l. 1091 (`22\,429`) | present; 22,430 absent |

### Round 3 (`STEP6_ROUND3_CORRECTIONS.md`)

| block | where | status |
|---|---|---|
| A.1 (A.6, the H row) | l. 1714 | present verbatim; old sentence absent |
| A.2(a) Σ_{s∈Ω_M×Ω_O} | l. 1508 | present verbatim; Σ_o absent |
| A.2(b) z-update sentence | l. 1606 | present verbatim; old sentence absent |
| A.2(c) end of cycle (latex block) | l. 1697–1701 | present verbatim. **Change:** the `$k \gets k + 1$\;` line after the replaced lines was deleted (NEW-2) |
| A.3(d) preamble | l. 1497 | present verbatim. The untouched continuation "with residual balancing…" now follows "before it is updated" (NEW-5) |
| A.3(e) sum over blocks | l. 1504 | present verbatim |
| A.3(f) κ^E sentence | l. 1515 | present verbatim (NEW-6 is in the block's own text) |
| A.3(g) κ^E in eq. esso_objective; ℒ sentence | l. 856, 865 | present verbatim |
| A.3(h) failure rule | l. 1621 | present verbatim; old sentence absent |
| A.3(i) ρ freeze continuation | l. 1625 | present verbatim (row A3-3 residual is in the block's own text) |
| A.3(j) √3 in s_E; which λ | l. 1636 | present verbatim (both) |
| A.3(k) cap and k₀ | l. 1636 | present verbatim (row A4-3 residual is in the block's own text) |
| A.3(l) AA iterate | l. 1640 | present verbatim; old text absent |
| A.3(m) continued-evaluation clause; tail "from the cycle after k₀" | l. 1640, 1644 | present verbatim. **Change:** the preceding clause "and the acceleration is switched off on any cycle in which every channel passes and, in the certifying regime," was deleted (NEW-1) |
| A.3(n) tolerance restore / declared option | l. 1644 | present verbatim (NEW-7 is in the block's own text) |
| A.3(o) retry tiers | l. 1644 | present verbatim; "documented sequence" absent |
| A.3(p) plan enters only through capacities | l. 1658 | present verbatim |
| A.3(q) initialisation (latex block) | l. 1678–1680 | present verbatim (NEW-9 is in the block's own text) |
| A.3(r) gap recorded by the harness | l. 1696 | present verbatim |
| A.3(s) TSO representation of the DN | l. 1714 | present verbatim |
| B.a Δ₀ clause; KwIn item | l. 417, 403 | present verbatim |
| B.b While line (the budget line stays) | l. 419 (l. 431) | present verbatim; old While absent |
| B.c variant-B completion | l. 429 | present verbatim |
| B.d F2 sentence; variant-A clause | l. 449 | present verbatim; "evaluated twelve neighbours" absent (NEW-4 is in the clause's own text) |
| C.1 table = W167 fragment; sentence | l. 951–968; l. 972–974 | present verbatim (the fragment from `\begin{table}` on) |
| C.2 TN generation sentence | l. 983–985 | present verbatim |
| C.3 red paragraphs of §3.4 deleted; branch table branch 1 per ADN | —; caption l. 1816 | done (no `\textcolor{red}` left); the caption states 200/100/150 MVA (the "state it in the caption" option) |
| C references: "Section 3.4" → `sec:case_ess_params`; "Section 3.5" → `sec:case_settings` (×3); "Section 4.7" literal | l. 840; l. 756, 866, 907; l. 449, 643 | done as instructed; no literal "Section~3.4/3.5" left. Two of the three `sec:case_settings` targets are wrong (NEW-8) |
| (C.4 §3.5, C.5 §3.6: presence only; values are W174a's) | l. 1066–1103, 1105–1147 | present verbatim |

---

## New differences introduced by the round-2/3 text (touched passages only)

| ID | lines | text | code (file:line) | how it differs | arises from | impact |
|---|---|---|---|---|---|---|
| NEW-1 | 1640 | "The memory is cleared on any change of a penalty parameter and on any failed local solve, from the cycle after k₀ on (or after the earlier run's stopping cycle in a continued evaluation)." | `admm_anderson_acceleration.py:477` (AA off when all channels pass), `:398` (clear on ρ change), `:416` (skip on failure); `shared_resources_planning.py:3422`; `p515_s53_w118_resettle_hooks.py:526, 722` (AA forced off for c > first pass) | Pasting (m) deleted "and the acceleration is switched off on any cycle in which every channel passes and, in the certifying regime,". (i) The AA-off rule is gone from Appendix A, although Algorithm 2 l. 1697 still says "(unless off)". §2.2.7 l. 624 still says "acceleration off" for the certifying regime. The production rule (off on every passing cycle) governs the 3 × 3 runs and the search evaluations. (ii) Read literally, the memory is cleared only from k₀ + 1 on; the code clears it at any cycle | row A4-6; round-3 (m) paste | **M** |
| NEW-2 | 1683–1702 | Algorithm 2: "While k ≤ k^max { … exit }" with no "k ← k + 1" | `shared_resources_planning.py:3066` (`for iter in range(1, …num_max_iters + 1)`) | the loop counter is never incremented; the line existed at `407f8df` | row A5-4; round-3 A.2(c) paste | **L** |
| NEW-3 | 449 | "the runs reported here stopped at their evaluation budgets and are described by their recorded certificates" | campaign_results `termination`: s47 `mesh_local_optimum_unit_poll_failed` (14/20); s51 `STOP_FOR_REVIEW_completion_cap` (61 > 30; 20/20); s53 `poll_failure_at_unit_mesh` (7/60) | no run ended on its evaluation budget (`evaluation_budget_exhausted`, `p515_s47_phase_b_record.py:663`). Two ended on a failed unit poll and one on the completion cap. This contradicts the same paragraph's certificate statements | round-2 A.3 | **M** |
| NEW-4 | 449 | variant A, "in which every rounded poll direction was inadmissible at every poll, so that every evaluated point came from the unit-neighbour completion" | s51 `poll_history` poll 4 (Δ = 1): directions 7 `rejected_infeasible`, 1 `new_evaluation` (not evaluated: the poll was refused at the cap) | false at one poll (s51 poll 4: one admissible direction). The conclusion "every evaluated point came from the completion" holds | round-3 B.d | **L** |
| NEW-5 | 1497 | "…in which all three agents solve against the same z before it is updated with residual balancing of the penalty parameters, Anderson acceleration and a tightened interior-point tail." | `shared_resources_planning.py:8687` (z update), `:3406` (ρ balancing), `:3275` (AA) | the original continuation now modifies "updated", so it reads as if z were updated with residual balancing, AA and the tail. Grammar | row P-1; round-3 (d) | **L** |
| NEW-6 | 1514–1515 | "The storage agent's augmented-Lagrangian terms are multiplied by κ^E … ; the consensus terms themselves are unscaled in every agent." | `shared_resources_planning.py:5515` (agent AL × κ^E); eq. (admm_esso_local) l. 1562 | the clause contradicts the preceding clause and the equation. It holds only for the argmin-equivalent rescaling f_E/κ^E + AL | row A1-3; round-3 (f) | **L** |
| NEW-7 | 1644 | "(the tail is a declared option, enabled in every campaign)" | search specs `8cfa264e`, `5ce295e1`, `803571c0`: no `convergence_depth_tail` in `configuration`. At heads 353e094b / c39c4836 / e1f98be8, `_apply_convergence_depth_tail` and `convergence_depth_tail` occur 0 times in `shared_resources_planning.py`, `admm_parameters.py` and `p515_s44_campaign_harness.py` | true for every campaign behind the tables (v6, extension, A64, 3 × 3). False for the three master-search campaigns, whose evaluations (and σ_Q) ran without a tail | row A4-5; round-3 (n) | **L** (search-time values only; none is reported as a result) |
| NEW-8 | 756, 866 | "The values of η, SoC^Min, SoC^Max, SoC⁰, ε^Cl, c^Cl and ε^C are given in Section \ref{sec:case_settings}"; "The values of c^σ and ε^E are given in Section \ref{sec:case_settings}" | main.tex §3.5 l. 1070 (`sec:case_ess_params`) prints all nine; §3.6 (l. 1105–1147) prints none | wrong cross-reference, from the round-3 rule "every Section 3.5 → sec:case_settings" | round-3 §C references | **L** |
| NEW-9 | 1678 | "(in the multi-scenario instance the commitment charge and the settlement are activated here)" | `shared_resources_planning.py:2925–2926` → `:5050, :5340` (settlement weight 0 → 1 for every instance); `model_construction_helpers.py:1701` (initialised 0), `:1917` (row 18 not constructed at one scenario) | the settlement is activated here in every instance. Only the commitment charge (row 18) is multi-scenario. A reader would place the single-scenario settlement in the initialisation solves too | row A5-2; round-3 (q) | **L** |
| NEW-10 | 841 | "\eqref{eq:soh_chain} the available-energy product in \eqref{eq:cohort_available} and the converter circle are…" | — | missing separator after soh_chain ("and" moved by the paste) | row T3-9; round-2 C.4(b) | **L** (wording) |

The remaining defects in the round-2/3 text are listed as residuals of their own rows (status "partly resolved"):
- T2-1 / A3-3: "from the stopping cycle", one cycle early;
- T2-4: "all after k₀" against lo ≥ k₀;
- T2-5: "starting after k₀" against ≥ k₀ + 3, and the gap "on that window" against the certifying cycle;
- A4-3: "holds start at the first passing cycle", one cycle early.

---

## `% [CONFIRM — W172]` and `% [CONFIRM — W173]` comments still in main.tex

| line | comment | does the text now state what the audit confirmed? | delete? |
|---|---|---|---|
| 444 | W173: Δ₀ and double/halve (s47, s51); N^max 20/60; Householder n + 1; s53 snap tie-break "not stated, implementation detail" | yes. Δ₀ = 4 (l. 403, 417; §3.6 l. 1134); ×2 / ÷2 (l. 435, 438); N^max in KwIn with values in §3.6; n + 1 Householder directions (l. 422); the tie-break is unstated by choice | **yes** |
| 452 | W173: "fourteen" and "twelve" | yes. L. 449 carries fourteen (x = 0) and seventeen / thirteen / twelve (F2), W173's counts. The comment's own "twelve (F2 poll set)" is obsolete | **yes** |
| 1517 | W172 (a) w_b at r = 0.02; (b) σ fixed and asserted within ×3; (c) κ^E = σ/median; (d) R^I per DN | yes: (a) l. 1504; (c) l. 1514; (b) σ's value and (d) R^I per DN are printed in §3.6 (values: W174a). The ×3 assertion is a code check, not printed | **yes** |
| 1579 | W172 (a)–(d): AL terms, normalisation, κ only in the agent, w/σ · f, γ = 0 | yes: eqs. (admm_network_local) and (admm_esso_local); l. 1576–1577 | **yes** |
| 1628 | W172: balancing constants and initial ρ "(Section 3.5)" | yes: constants at l. 1625; initial ρ in §3.6 (the comment's "Section 3.5" pointer is stale) | **yes** |
| 1647 | W172 (a) residual forms; (b) statuses; (c) AA incl. "off when all channels pass"; (d) tail; (e) recovery tiers; walker_ni_2011 | (a), (b), (d) yes. (e) yes, correctly per agent in §3.6; the comment's own "1e-4 / 1" is TSO-only. walker_ni_2011 is present (1 entry). **(c) no:** "off when all channels pass" is no longer stated (NEW-1) | **no, not until NEW-1 is restored** |
| 1706 | W172: algorithm order and initialisation against `_run_operational_planning` | the order now matches (A5-2, A5-4), but the loop increment is missing (NEW-2) | **after NEW-2 is restored** |
| 1717 | W172 (a) row 18 activated with the settlement weight; (b) weight 1 and the contracted/deviation split; (c) pin 9e4 excluded | yes: l. 1714 (and the single-scenario sentence); 9 × 10⁴ in §3.6 | **yes** |
| 1101, 1144 | W175 | outside this task (W175, `f7d165b7`) | — |

---

## Code identity

`git diff --stat <git_head_at_run> HEAD -- <pathspec>` was run as one subprocess per head, from an argument list (no
shell). The pathspec is the 20 production modules plus `SRP1.json`, `SRP1_params.json`, `SharedESS/`, `case9/`,
`case33_{1,2,3}/` and `MarketData/`. Repo HEAD was `a8eb9b69`.

| set | result |
|---|---|
| every `campaign_results.json` under `w142_resettle_v6`, `w142_resettle_ext_v6`, `w155_a64_cells`, `w90_3x3/campaign_s53_w91_3x3_pair` | 46 files, 46 distinct heads, **NO CHANGE for all 46** |
| positive control `353e094b` | 5 files changed, +1,507 / −84: the pathspec is live |
| informational: the master-search heads | s47 `353e094b` 5 files +1,507/−84; s51 `c39c4836` 4 files +1,376/−81; s53 `e1f98be8` 3 files +504 |

W175 (`f7d165b7`) committed while this ran; its diff against `a8eb9b69` touches no file in the pathspec. The s53 figure
differs from W171b's two-file count because W171b's pathspec had 19 modules; I did not recompute W171b's.

## Record checks (E, `w174b_checks.json`)

- **E1 search polls.**
  - s47: every direction inadmissible at all 3 polls (admissible per poll 0/0/0).
  - s51: admissible per poll 0/0/0/0/**1**.
  - s53 (variant B): 1 admissible as rounded.
  - Terminations and budgets as in NEW-3.
- **E2 tail in the searches.** No spec declares it. At each search head the tail code is absent, with 0 occurrences in
  each of the three files.
- **E3 certificates.**
  - The frozen tables (61 cells) carry, by certifying criterion: v6 36, v5 1, v4 2, v2 8 (incl. the superseded
    `pb_y2025_n5`), uncertified 12, and the two v1 references (`ref:7aa017f0` k* 181, `ref:bd504ecf` k* 172).
  - `w142_v6_from_records.json` reproduces every committed k* under v6 on the 18 pre-v6 records.
  - `pb_y2025_n5` is not decided within its records; it is superseded by `pb_y2025_n5_v6`, and Addenda 58/59 keep it
    out of the tables.
  - W168 gives x0 v6 k* 181.

## Unexpected findings

1. **§3.6 l. 1107, "Every recourse evaluation ran … under one frozen configuration".** This reasserts in §3.6 the
   claim that W171b's T1-10 removed from §2. The search evaluations ran at three other code heads, without the tight
   tail (E2). This is W174a's scope (§3.6); recorded here only.
2. **The untouched l. 1636 still says "every local solve of the cycle ended at an optimal status".** L. 1644 counts
   acceptable exits as success, as the code does (`helper_functions.py:74–85` per W172). This is not a W172 row and was
   not touched by round 2 or 3, so it is outside this re-audit and not counted.
3. W175 committed during the task (see Code identity).

## Changed

- New: `p515_s53_w174b_reaudit_checks.py` (sha256 `40c49e2631b132c8aac8bbccb4c7847f50f4a6dc1e5e465d110f6bb7517fc7ff`).
- New outputs in `data/SRP1/Results/P515S53/w174b_reaudit/`:

  | file | sha256 |
  |---|---|
  | `w174b_checks.json` | `b58f5d83…85be383` |
  | `manifest_sha256.json` (output, script, 76 inputs) | `d9157d90…981fbc57d` |
  | `launch.log` | `e28b57fe…387bf40c` |
  | `w174b_bool_typing_test.json` | `3a7eac6b…54e8d24` |
  | `w174b_bool_typing_test.log` | `8b7a99a8…4abc9d796` |
  | `manifest_post_run_sha256.json` | `807504fd…3f4b97b` (hashes of all of the above and the script) |
- This report.
- Nothing else: no production file, spec, case file, frozen artifact, committed artifact or manuscript-clone file was
  touched.

## Not confirmed

- **Compiled output.** No PDF exists in the clone. Equation and algorithm references are to `\label`s and source lines.
- **§3.5–3.6 values.** These are W174a's task. Only the presence of C.4/C.5 was checked here, plus the targets of the
  §2 cross-references (NEW-8).
- **Whether the W118 (v2) certificates satisfy the clean veto on their own exit classes.** W142's v6-from-records
  replay used non-clean cycles recomputed from the logs (the floor replay); I relied on its committed result and did
  not re-derive the exit classes.
- **Rating of T1-4's residual (l. 647).** M is my rating, not a ruling (Blocked 3).
- **Scope of negative claims.**
  - "No run ended on its budget": the `termination` field of the three committed `campaign_results.json`.
  - "Tail absent at the search heads": `git show <head>:<file>` for the three files named, at the three heads.
  - "No CONFIRM-W172/W173 comment beyond those listed": regex over all 2,209 lines.
  - "No Σ_o left in §2.3.6": whitespace-stripped search, plus reading l. 873–908 (l. 610's Σ_{m}Σ_{o} in eq.
    recourse_value is §2.2.7 and correct).
  - "Values not in §3.6": reading l. 1105–1147.
