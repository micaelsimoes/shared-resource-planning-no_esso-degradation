# P5.15 W171b — equation-vs-code audit of the revised §2 (main.tex l. 315–901, Overleaf `42794d4`)

**Worker, 2026-10-08. ZERO SOLVES, zero model builds, zero pickle loads; read-only on production code, specs, the frozen
JSON and the manuscript clone.** Replaces the W166 §2.3 table (`P5_15_W166_EQUATION_CODE_AUDIT.md`, `e7b9a938`).
Appendix A is not audited (it awaits the expert's draft).

**What was audited.** `manuscript/6a67305f25e8348fb71380c3/main.tex` at HEAD `42794d4` (sha256 `7effd898…515230b7`
verified, 2,092 lines; clone not modified, pulled or committed): §2 l. 315–901 — §2.1 incl. Algorithm 1 (l. 328–454),
§2.2 incl. §2.2.7 (l. 455–644), §2.3 (l. 645–901) — and its two `% [CONFIRM — W169]` comments (l. 747–748, 857–858; no
other CONFIRM comment exists in the range).

**Configuration in force, identified as W166 did.**
- **Table cells.** Stage spec `w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json` (34 cells; P_MAX 30), extension
  `w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json` (7), A64 `w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json` (4),
  3 × 3 `w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json`. Case files `SRP1_params.json`
  (`dbfdb2a0…`), `SRP1_ESS_Params.json` (`39106f93…`), `SRP1.json` (`61a794a7…`).
- **Master search (Algorithm 1).** `p515_s47_phase_b_record.py` (PB, spec `campaign_spec_s47_phase_b_8cfa264e.json`),
  `p515_s51_f2_phase_b.py` (spec `…5ce295e1.json`), `p515_s53_f2_certificate.py` (spec `…_r1_803571c0.json`);
  `STEP4_DFO_METHOD.md`; `p56a_oracle.py` as the campaign harness uses it (`fresh_planning` l. 165: a deep-copied
  baseline per evaluation; l. 926-942: no warm start from previous candidates — the cold-start statement of l. 451).
  Its polish pipeline (docstring steps 4–8) did not produce the reported values: the tables carry
  `gross_operational_cost` at k* (`Q_k_star` of each decision record).
- **Settling rule.** `settling_criterion.py` (v1) … `settling_criterion_v6.py`; v6 hooks `p515_s53_w142_resettle_v6_hooks.py`
  (holds inherited from `p515_s53_w118_resettle_hooks.py`); scorers `p515_s53_w142_determinacy.py`,
  `p515_s53_w132_resettle_v3_campaign.py` (`resolve`, `view_from_report`).
- **Code identity.** `git diff` of the 19 production modules plus `SRP1.json`, `SRP1_params.json`, `SharedESS/`, `case9/`,
  `case33_*/`, `MarketData/` between HEAD `71f8d2aa` and each campaign's `git_head_at_run`: **NO CHANGE for all 46 table
  campaigns** (34 + 7 + 4 + the 3 × 3) — control: the same command against `353e094b` reports changes, so the pathspec is
  live. (A first attempt under zsh passed the pathspec list as one word and matched nothing; it was discarded and re-run
  under bash.) **The three master-search campaigns ran on older production code**: `353e094b` (s47 Phase B) 5 files
  +1,507/−84, `c39c4836` (s51 F2 Phase B) +1,376/−81, `e1f98be8` (s53 F2 certificate) `network.py` +157 and
  `shared_resources_planning.py` +329 lines differ from HEAD (row 18, the init fix, the tail code). Not re-verified here:
  the SRP1 bitwise gates the brief records for those changes.

Impact: **H** = a referee re-deriving from the text gets a different model or number; **M** = imprecise, not a different
model; **L** = notation or wording. Line numbers are main.tex lines at `42794d4`.

---

## Blocked on Planner

1. **The recorded prediction fails** (see next section): two H rows remain in §2.3 (T3-1 the consensus variables of the
   storage channel; T3-2 the expectation of the commitment row over operation scenarios only). This is a
   text-wording prediction, not a margin or sign prediction; I report it and stop on nothing else. Whether either row is
   to be reworded is the expert's call.
2. **The sentence l. 639 "The rule above, with its holds, produced every single-scenario evaluation reported in this
   paper" is false against the run records** (row T2-1). The two SRP1 references were certified under rule **v1**
   (W101): `ref:7aa017f0` (x = 0, k* 181, anchor of every T1 B row — W168 replays it under v6) **and `ref:bd504ecf`
   (k* 172), which no current task replays under v6** as far as I could find (TASKS.md Addendum 68 order). Nine further
   table cells were decided under v2 (W118: five Phase B neighbours and the two year-ladder cells certified, the F2 pair
   uncertified), two under v4 and one under v5. Needs a ruling on whether `ref:bd504ecf` gets a W168-style replay and how l. 639 is
   worded.
3. **§2.1 carries five H rows**: T1-1 (the recourse equation) and T1-2 … T1-5 (Algorithm 1): the printed algorithm is
   the 2n/snap/doubling poll with §2.2.7 certification and determinacy, over the full (S, E)_{e,y} lattice; the
   searches that produced the paper's incumbents ran a 7-variable common-year lattice, n + 1 directions with a full-box
   completion (s47, s51), a 2n snap poll without doubling (s53), the production-exit certification and the threshold
   max(bar_x + bar_inc, σ_Q); and the F2 termination certificate is recorded by the code as **not** a positive-spanning-set
   certificate. Expert wording decision; no formulation is proposed here.
4. Nothing else blocks. Every requested output is below.

## Prediction vs outcome

| prediction (expert, Addendum 68 / `STEP6_ROUND1_CORRECTIONS.md` §C) | outcome |
|---|---|
| "W171 — no H-impact row remains in §2.3 after the corrections" | **FAILED: 2 H rows in §2.3** (T3-1, T3-2). Neither is a defect the corrections introduced: T3-1 is l. 656, the expert draft's sentence (`manuscript_review/section2_expert_draft.tex` l. 284) that no correction touched, T3-2 is the commitment equation the corrections did not touch beyond A.6. |

Counts of differing rows (consistent statements are counted separately, last section):

| subsection | H | M | L | rows |
|---|---|---|---|---|
| §2.1 incl. Algorithm 1 (l. 315–454) | 5 | 5 | 2 | 12 |
| §2.2 incl. §2.2.7 (l. 455–644) | 3 | 5 | 3 | 11 |
| §2.3 (l. 645–901) | **2** | 0 | 9 | 11 |
| **total** | **10** | **10** | **14** | **34** |

Plus one printed claim in §2.3 whose truth is numeric and is not counted (P3-1, pending W169).

---

## Differing rows

### §2.1 — architecture and Algorithm 1 (l. 315–454)

| ID | lines | text as printed | code (file:line, name; spec) | code's form in the manuscript's notation | how it differs | impact |
|---|---|---|---|---|---|---|
| T1-1 | 356–362, 375, 381 | Q(x) = min_{u_o} Σ_o ω_o C^Op_o(x, u_o) s.t. u_o ∈ U_o(x) ∀o; "u_o contains the operational variables for scenario o"; "the operational variables u_o adapt to each realization of load, RES generation, and market prices" | `model_construction_helpers.py:841-852` (`sess_na_scenario`, `sess_row_is_duplicate`: one scenario-free storage copy per block); `shared_resources_planning.py:4234-4273` (TSO interface fixed + scenario-free Δ); `model_construction_helpers.py:1861-1998` (row 18, P̄ a block variable); 3 × 3 spec `231558f0` | Q(x) = min_{u⁰,{u_o}} Σ_o ω_o C_o(x, u⁰, u_o), (u⁰, u_o) ∈ U_o(x) ∀o, with u⁰ = (storage schedule, TSO interface exchange, DSO expectation P̄) common to all scenarios of a block | the printed recourse is separable per scenario (wait-and-see); the code's is a within-block two-stage problem with hard non-anticipativity of storage and TSO interface. §2.3 (l. 654, 745, 866) states the code's form, so §2.1 contradicts §2.3. Identical at one scenario | **H** |
| T1-2 | 411–415, 383, 319, 458 | 𝒳 = {x = (S^Inv_{e,y}, E^Inv_{e,y})_{e∈E^S, y∈Y}}; z = diag(ΔS, ΔE, …)^{-1}x ⊂ ℤ^n; master determines location, size and timing; "staged capacity expansion" | `p515_s47_phase_b_record.py:191` (`N_VARS = 7`), `327-468` (`Lattice`); ruling A1 (docstring l. 92); `p515_s44_campaign_harness.py:113-120` ("Multi-cohort (staging) candidates are NOT supported"); brief Addendum 28 ("Staging: not wanted") | z = (z_P5, z_E5, z_P7, z_E7, z_P9, z_E9, z_Y) ∈ ℤ^7: one cohort per node and **one investment year common to all nodes** (z_Y an ordinal year index); n = 7, not 2·|E^S|·|Y| = 18 | different decision space and dimension (hence different directions and neighbourhoods); per-node timing and staging never evaluated | **H** |
| T1-3 | 426–433, 441 | "Generate the 2n OrthoMADS directions ±v_1…±v_n"; round; infeasible → snapped to the nearest admissible point; completion "if \|𝒫\| < n+1 … up to the stated cap"; success → Δ ← 2Δ | s47 / s51: `p515_s47_phase_b_record.py:502-513` (`poll_directions`, `orthomads_n_plus_1_neg`), `606-611`, `663-668` (`run_mads`); s53: `p515_s53_f2_certificate.py:215-245` (POLL_RULE, SNAP_RULE, ITERATION_RULE) | s47 and s51: n + 1 = 8 directions (Householder columns h_j and −Σ h_j), infeasible ones rejected unevaluated (no snap); at Δ = 1 the poll is directions ∪ the **full ℓ∞-box** of feasible neighbours; box > 30 points → poll refused, STOP FOR REVIEW (s51 stopped there: 61 > 30). s53 (F2 certificate): Δ = 1 only, 2n = 14 directions, snap to the nearest feasible point of the incumbent's unit frame with a 6-level tie-break, completion only if < n + 1 distinct feasible points (cache hits counted), **Δ stays 1 after a success** | the printed poll is neither run's poll; the s53 rule has no doubling | **H** |
| T1-4 | 420, 435, 439–441 (and 643) | evaluations by Algorithm 2 "and the certification rule"; accept "by a determinate margin (§2.2.7)", i.e. max{3b, 2τ} | `p515_s47_phase_b_record.py:516-531` (`resolution`, `classify`), ruling A4; specs `8cfa264e`, `5ce295e1`, `803571c0` (cap 500, 10 consecutive cycles); σ_Q l. 62, 785-791 | every search evaluation was certified by the **production exit** (10 consecutive residual passes, cap 500, no holds); accept iff F(inc) − F(x) > max(bar_x + bar_inc, σ_Q), bar = max objective step over the last 10 cycles, σ_Q = 18,449.66 € | different certification and acceptance threshold during the search; the §2.2.7 rule and max{3b, 2τ} were applied afterwards to the re-settled Phase B cells (frozen `phase_b` table: `recorded_W118_rule` and `v6_verdict`, all determinate) | **H** |
| T1-5 | 452–453 | terminates "when a positive spanning set of lattice neighbours of the incumbent has been evaluated and none improves it"; "claims mesh-local optimality" | `p515_s53_f2_certificate.py:246-262` (CERTIFICATE_STATEMENT, `certificate_scope` 572-616); `campaign_s53_f2_certificate_r1/campaign_results.json` `termination_certificate.scope`; `campaign_s47_phase_b/campaign_results.json` `termination_certificate` | F2 (the method's demonstration): "poll failure over the recorded poll set at unit mesh, **NOT a positive-spanning-set certificate, NOT the full-box certificate** … rank 5 … does NOT positively span R^7 … 51 of 61 feasible box neighbours outside the poll set, 44 not evaluated"; x = 0: all 14 feasible ℓ∞-1 neighbours polled (incumbent at the bound, so no positive spanning set is feasible) | the printed certificate is stronger than the one the code recorded for F2 | **H** |
| T1-6 | 410, 419–422, 451 | 𝒳 "enforced in closed form before any evaluation"; designed set evaluated; x^inc ← argmin_𝒞 F | `p515_s47_phase_b_record.py:538-583` (`initial_incumbent`); STEP4 §4.1 ("The budget is not applied in A1") | the designed set (ladders) includes budget-infeasible points evaluated deliberately; x^inc = argmin F over certified, on-lattice, **budget-feasible** cache entries incl. x = 0, ties lower I then label | the designed set is not in 𝒳; the argmin is restricted | M |
| T1-7 | 422, 424, 438 | N ← \|𝒞\|; while N ≤ N^max; N ← N + \|𝒫\| | `p515_s47_phase_b_record.py:200-201, 663-668`; `p515_s53_f2_certificate.py:156` | N counts **new** evaluations of the run only (cache hits not counted); N^max = 20 (s47, s51), 60 (s53); a poll whose new evaluations would exceed the remainder is not launched; MAX_POLLS 60 | different counting and stopping test | M |
| T1-8 | 430 | "drop duplicates and cached points" | `p515_s47_phase_b_record.py:631-639`; POLL_RULE (s53, `p515_s53_f2_certificate.py:215-225`) | cached points stay in the poll as cache hits (read, not re-evaluated), are compared with the incumbent and (s53) count toward the n + 1 trigger | literal reading removes them from the comparison and the count | M |
| T1-9 | 436–437 | barrier points recorded with cause, F = +∞ | `p515_s47_phase_b_record.py:202-203, 684-688` (ruling A7) | as printed, plus STOP FOR REVIEW at ≥ 2 new barrier evaluations in one poll or ≥ 3 overall | stop rule not stated | M |
| T1-10 | 451 | "under one frozen configuration, so that Q(x) is a deterministic function of x and the cache is exact" | `git diff` above; specs `8cfa264e`, `5ce295e1`, `803571c0` vs `96c23404` | one configuration per search campaign, but three campaigns at three code heads (gated inert at SRP1 per the brief, not re-verified); the search used production-exit values, the tables report the same points re-settled under v6 | the Q the search compared is not the Q the tables report | M |
| T1-11 | 393 | the master "uses the value Q(x) returned by the recourse and nothing else" | `p515_s47_phase_b_record.py:516-519` | the acceptance test also reads each record's bar and σ_Q (l. 574 says so: "and its certification record") | wording conflict with l. 574 | L |
| T1-12 | 416 | cache: "one certified evaluation per canonical plan" | `p515_s47_phase_b_record.py:678` | keyed by the evaluation key (canonical candidate + configuration, + m for F2); barrier records are cached too | wording | L |

### §2.2 — master problem and §2.2.7 (l. 455–644)

| ID | lines | text as printed | code (file:line, name; spec) | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| T2-1 | 620, 639 | holds "from the next cycle on" after k₀; "The rule above, with its holds, produced every single-scenario evaluation reported in this paper." | frozen tables `590088fe` `cells[*].certifying_spec`; `p515_s53_w101_settling_continuation_hooks.py:1-30`; `p515_s53_w118_resettle_hooks.py:1-60`; `settling_criterion.py` (v1), `_v2.py` | rule versions behind the tables' single-scenario cells: v6 (34 + 7 + 4 + `d_c52e1670` from records), v5 (`b_4649234b`), v4 (`b_0dd237f0`, `b_2a0ba8b2`), v2 (W118: `pb_y2030_n9/n7/n5`, `pb_y2025_n9/n7` certified; year ladder 2030/2035 certified; `ref:5ca4f86c`, `ref:e28de4ac` uncertified), **v1** (W101: `ref:7aa017f0` k* 181, `ref:bd504ecf` k* 172; holds from N_old + 1, decisions only after N_old). v1 has no gap clause, no clean veto, no swing floor, and a monotone branch without \|step\|·L ≤ τ; v2 has no clean veto and no floor | the v4/v5/v2 certificates were checked against later readings (Addenda 59, 61 replays); the v1 references were not. Numeric part for `ref:7aa017f0`: **W168**; `ref:bd504ecf`: none queued | **H** |
| T2-2 | 624–625 | "its period P̂ (measured between the first and third turning points)" | `settling_criterion_v2.py:166` (and `settling_criterion.py:230`): `p_hat = T[-1][0] - T[-3][0]` | P̂ = t(T₋₁) − t(T₋₃): the **three most recent** turning points since k₀ (as the accepted `paragraphs_v5.md` l. 29–30 says) | different window length whenever > 3 turning points. Record check (below): of 46 committed decisions replayed (46/46 reproduced), the text's P̂ changes 19 — 14 certify at another cycle (\|ΔQ*\| ≤ 0.087 τ), 5 not certified by the code's k* (records end there) | **H** |
| T2-3 | 638 (643) | settling slack = "the objective's movement since the residual pass" | `p515_s53_w132_resettle_v3_campaign.py:1000, 1009-1018` (`s_signed`, `view_from_report`) | slack = \|Q(end) − Q(N_old)\|, Q(N_old) from the **original** run's record at its old certification cycle (N_old = k₀ + 9 on every gated cell); undefined for an ungated cell (`paragraphs_v5.md` l. 57 says so) | different number: in all 12 uncertified cells; the uncertified bar changes on 7 of 23 uncertified-form claims (×0.98 to ×2.44); **no verdict changes** (record check) | **H** |
| T2-4 | 624–635 | turning points, swings and window "since k₀" | `settling_criterion.py:58-59, 96-102, 294-310`; `settling_criterion_v2.py:239-258` | a step \|ΔQ\| < EPS0 = τ/100 = 45.39 € carries no sign; the first K_EXCL = 3 cycles after k₀ never enter the sign state; a Boyd lapse or failed cycle resets k₀ and clears T, A (holds stay: AA off latched); the window must lie inside the run (k − n_w + 1 ≥ k₀) | four clauses unstated | M |
| T2-5 | 638 | monotone branch: window 2P_max; "its steps are decreasing", range ≤ τ, \|last step\|·2P_max ≤ τ | `settling_criterion_v2.py:176-197, 277`; `settling_criterion_v6.py:260-295` | window [k − 59, k] starting ≥ k₀ + 3 with **no sign change**; "steps decreasing" = mean \|ΔQ\| over the second half < first half, or every \|ΔQ\| < τ/100; the **gap clause** (\|t_sum\| ≤ τ/2) and the **clean veto** on that window also apply | three conditions unstated, one paraphrased | M |
| T2-6 | 643 | uncertified: determinate "only if it exceeds three times the larger of that evaluation's consensus gap and settling slack" | `p515_s53_w132_resettle_v3_campaign.py:1060-1081` (`resolve`) | bar = 3·max over **all** uncertified cells of the pair of {\|t_sum(end)\|, slack}; determinate iff \|d_gross\| > bar **and** the same-sign gap-corrected margin (Q_cc = Q + t_sum) > bar | Q_cc condition and two-cell case unstated (`paragraphs_v5.md` has "in both gross and gap-corrected terms") | M |
| T2-7 | 639 | 3 × 3: production exit "and then continued past it; its values are reported as a band" | `P5_15_ADDENDUM51_CONTINUATION_REPORT.md` l. 12; brief Addenda 51–52; frozen `three_by_three` (`formula_R`) | x = 0 cell continued 16 cycles (72 → 88, regime held, early stop \|ΔQ\| < 500 €/cycle ×3); the storage cell was **not** continued (stage 2 not triggered); T11's band is on R: [(V − D)/V_SRP1, V/V_SRP1], D = 6,234.94 € from a post-hoc damped-cosine fit of the x = 0 continuation | only one of the two cells continued; the band is a ratio band from a fit | M |
| T2-8 | 617–618 | C^Op = TN generation cost, DN activated flexibility "and curtailment", closure-slack penalty, (multi-scenario) row 18 and the deviation settlement | `model_construction_helpers.py:1811-1846` (`objective_function_rule`), `2238-2316` (slack penalties), `2001-2008` (`generation_cost`); `shared_resources_planning.py:5042-5044, 5332-5334` | gross Q also contains the networks' feasibility slack penalties (voltage-squared 5e4, node balance 1e6·baseMVA, branch flow 1e3·baseMVA, flexibility day-balance 1e3·baseMVA) and load curtailment at 300 €/MWh in TN and DN; RES curtailment has **no** cost (penalty zeroed for ADMM); TN generation is priced at the scenario market price | at the x = 0 reference's terminal cycle the slack penalties total −13,038 € (bound-relaxation residue; `l_195156fa/component_levels_terminal.json`), load curtailment 0 | M |
| T2-9 | 632–634 | clean = acceptable "with all four termination metrics within ten times the tight-tail tolerances" | `settling_criterion_v5.py` `TOLERANCES` | for ESSO blocks the 10× is on the ESSO's own tolerances (tol 1e-10, compl 1e-4 default); the tail never touches the ESSO | wording | L |
| T2-10 | 643 | determinate when the difference "exceeds" max{3b, 2τ} | `settling_criterion_v6.py:366-371` | \|margin\| ≥ threshold | strict vs non-strict | L |
| T2-11 | 496, 503 | lower limit max(y − T^Cal_{e,y} + 1, y₀) | `shared_energy_storage_data.py:561-565` | the lifetime is the cohort's, T^Cal_{e,y^Inv} (block count round(T^Cal / Y_{y^Inv})) | index (equal values: T^Cal = 15 everywhere) | L |

### §2.3 — subproblem (l. 645–901)

| ID | lines | text as printed | code (file:line, name; spec) | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| T3-1 | 656 (753, 745) | "The storage schedule itself — **charging, discharging and reactive power** in every hour — is a consensus variable between the operators' network models and the shared-ESS agent" | networks: `model_construction_helpers.py:995-999` (`sess_pnet_rule`), `2450-2509` (`expected_shared_ess_p/q`), AL `shared_resources_planning.py:5180-5188` (TSO), `5436-5441` (DSO); ESSO: `shared_energy_storage_data.py:777-787` (`es_pnet`, `es_qnet`), AL `shared_resources_planning.py:5505-5523` | the ESS channel couples **net active power and reactive power only**: networks P^E_net = P^Ch − P^Dch, Q^E; agent P^Net = Σ_{y^Inv}(P^Ch − P^Dch) + σ⁺ − σ⁻, Q^Net. The agent's own charge/discharge split — the throughput that ages the cells — is not in consensus with the networks' split | 2 consensus quantities per hour, not 3. The splits are tied only through each model's own complementarity (network ε^C = 1e-4 on the normalised product; agent ε-regulariser). Also l. 745 "the schedule the agent ages is the one every scenario runs" holds for the net schedule, not the split | **H** |
| T3-2 | 866–888, 897 | \bar P^I_{i,t} = Σ_{o∈Ω_O} ω_o P^I_{i,o,t}; deviations per (i, o, t); charge Σ_o Σ_t ω_o α π̄_t (d⁺ + d⁻); o = "operation scenario … with its own load and RES realization"; "At a single operation scenario (…) vanish identically" | `model_construction_helpers.py:1861-1876` (row-18 rules), `1877-1998` (`add_scenario_commitment_terms`; wiring guard `n_scenarios == 1` l. 1917; charge l. 1979); `dn_interface_expected_pf_p_def` (2416-2443) | P̄_t = Σ_{m,o} ω_m ω_o P^I_{m,o,t}; deviation per **(m, o, t)**; charge Σ_{m,o} Σ_t ω_m ω_o α π̄_t baseMVA (d⁺_P + d⁻_P + d⁺_Q + d⁻_Q); wired whenever \|Ω_M\|·\|Ω_O\| > 1 | the code's commitment is not contingent on the market scenario and prices market-driven deviations (91 % of deviation at α = 0, Addendum 40); the text's sums over operation scenarios only and vanishes at one operation scenario. (If o is read as the (m, o) pair of l. 370–372, the equations match except the l. 897 sentence; the paragraph itself defines o as an operation scenario.) baseMVA is a units factor (L) | **H** |
| T3-3 | 701, 757 | network P^E = P^Dch − P^Ch (injection, "positive when discharging"); agent P^Net = Σ(P^Ch − P^Dch) | `model_construction_helpers.py:995-999`; `shared_energy_storage_data.py:777-782` | both consensus copies use the consumption sign (pch − pdch) | opposite sign conventions for the two copies of one consensus quantity | L |
| T3-4 | 722 | 0 ≤ s^± ≤ ε^Cl E^Av | `model_construction_helpers.py:1162-1169`; `ESS_DAY_BALANCE_SLACK_FRACTION` l. 410; `definitions.py:90` (`EQUALITY_TOLERANCE`) | 0 ≤ s^± ≤ 0.05 E^Av + 10⁻⁵ p.u. (re-applied at every capacity publication) | numerical 1e-5 omitted (A.3(e) assigns it to §3.5) | L |
| T3-5 | 720 | E^SoC_{e,y,d,t₀} = SoC⁰ E^Av | `model_construction_helpers.py:958-980` (`sess_soc_rule`, l. 972-973) | the pre-period state 0.5 E^Av is a constant in the first hour's recursion, not a variable; t₀ is defined nowhere in main.tex | undefined index | L |
| T3-6 | 744 | the closure slack "is penalised at c^Cl per MWh in the block objective" | `model_construction_helpers.py:2333-2342, 2369-2382, 1818` | the penalty is in **each** network model that carries the storage (the TSO block and the DSO block of node e), so Q holds it once per model copy | count unstated | L |
| T3-7 | 684, 686 | lower limit max(y − T^Cal_{e,y} + 1, y₀) | `shared_energy_storage_data.py:561-565` | cohort lifetime T^Cal_{e,y^Inv}, window round(T^Cal/Y_{y^Inv}) blocks | index (equal values) | L |
| T3-8 | 795, 807 | Y_y in D and in φ^{Y_y} | `shared_energy_storage_data.py:678, 699-703, 721` | Y_{y^Inv}, read once per cohort | index; identical in both instances (5,5,5 and 3,…,3) | L |
| T3-9 | 831 | "(soh_chain) and the available-energy product … are the agent's nonlinear rows" | `shared_energy_storage_data.py:786-787` (circle), comment 593-596 | the converter circle (P^Net)² + (Q^Net)² ≤ (S^Av)² is also a nonlinear (convex quadratic) row | incomplete list | L |
| T3-10 | 830 | economic effect of degradation "through the available energy of later years and through the floor" | `shared_energy_storage_data.py:615-628` ('end' point) | with the end-of-year SoH (eq. cohort_available) a year's own throughput also reduces that year's available energy | wording | L |
| T3-11 | 648 | "a dedicated shared ESS subproblem"; "further decomposed across representative years and representative days" | `shared_energy_storage_data.py:74-78` (`build_subproblem`, one NLP per node) | one agent NLP **per shared-ESS node** (3), each spanning all years and days; only the network models are per (y, d) | wording | L |

**Printed claim pending numbers (not counted).**

| ID | lines | text | code fact | status |
|---|---|---|---|---|
| P3-1 | 753 | the σ pair "is never active at a consensus point within the agent's ratings" | σ^± (`slack_es_pnet_up/down`, `shared_energy_storage_data.py:458-460, 782`) is a free slack penalised at c^σ = 10³ per MW-period, unweighted (l. 822-829), in an objective whose AL terms carry κ_ESSO; nothing structural keeps it at 0 when the SoH floor (0.70) limits throughput — the floor was recorded binding in 2035 (Addendum 32) | **numeric part: W169** (cells where the floor binds are where to look) |

---

## A.1–A.8 of `STEP6_ROUND1_CORRECTIONS.md` §A in main.tex

Method: each of the 12 ```latex replacement blocks of §A searched in l. 315–901 with all whitespace removed; each inline
find→replace checked by grep for the new text and for the absence of the old text.

| item | where | status |
|---|---|---|
| A.1 notational-compactness paragraph | l. 654 | present verbatim |
| A.2(a) end-of-year SoH sentence | l. 670 | present verbatim |
| A.2(b) `SoH_{e,y^Inv,y}` in eq. cohort_available; CONFIRM deleted | l. 675 | present (no `y-1` left in that equation; no `[CONFIRM]` left) |
| A.2(c) publication sentence | l. 693–696 | present (same words, re-wrapped inside `\textcolor{blue}{…}` over three lines) |
| A.3(a) eq. soc_closure | l. 718–725 | present verbatim |
| A.3(b) eq. converter_capability | l. 727–733 | present verbatim |
| A.3(c) eq. network_complementarity | l. 735–740 | present verbatim |
| A.3(d) "where η…" paragraph + `% [CONFIRM — W169]` | l. 742–748 | present verbatim |
| A.3(e) old CONFIRM deleted | — | done |
| A.4(a) agent paragraph, eq. esso_net_power, allocation sentence, eq. cohort_allocation; H3 CONFIRM deleted | l. 753–772 | present verbatim |
| A.4(b) linearity sentence; D-row CONFIRM deleted | l. 831 | present verbatim |
| A.5 objective body + `% [CONFIRM — W169]`; old EPS CONFIRM deleted | l. 835–858 | present verbatim |
| A.6(a) "scenario-free … re-dispatches its own network per scenario against it, and may be deviated from on the DSO side" | l. 866–867 | present verbatim (line break inside) |
| A.6(b) deviation-energy / voltage sentences | l. 892–896 | present verbatim; the replaced sentence ("The TSO operates on the committed schedule…") is gone |
| A.6(c) α CONFIRM deleted | — | done |
| A.7(a) "where C^Op…" paragraph; discount CONFIRM deleted | l. 615–618 | **present with one change**: `\noindent` appears twice (l. 615 and 616 — the block's own `\noindent` pasted after the original one); typesetting only |
| A.7(b) "held fixed from the next cycle on:" and the 3 × 3 sentences | l. 620, 639 | present verbatim |
| A.7(c) n_w, 𝒱 (three places) | l. 628, 641 | present verbatim; no `W = \max`, `$V$`, `V = 259` or `\delta_R V` left |
| A.8 half-open cohort window; `+1` in both sums of 2.3.2 and of 2.2.2 | l. 660, 684, 686, 496, 503 | present verbatim |

All of A.1–A.8 are in. (A.9, nomenclature, is a final-pass item and was not checked.)

## `% [CONFIRM — W16x]` comments in l. 315–901

| line | comment | what the code says |
|---|---|---|
| 747–748 | [CONFIRM — W169] whether the closure slack was active in any reported evaluation | s^± = `slack_shared_es_soc_final_up/down[e, s_m0, s_o0]`, one pair per network block model carrying the storage (TSO block and DSO block of node e), scenario-free (`model_construction_helpers.py:983-992`); bounds [0, 0.05 E^Av + 1e-5 p.u.] re-set at every capacity publication (l. 1162-1169); penalty 10³ × baseMVA × (s⁺ + s⁻) (`definitions.py:58`, `model_construction_helpers.py:2333-2342`) inside `objective_function_rule` via `total_ess_complementarity_penalties` (l. 1818) and therefore inside gross Q; reported per block as `shared_ess_day_balance_slack` (D-row) in each eval dir's `component_levels_terminal.json` (terminal cycle only). Example: `l_195156fa` terminal, TSO −12.599 € and DSO −12.599 € (weighted; negative = IPOPT bound relaxation, Addendum 35). **Numeric part: W169.** |
| 857–858 | [CONFIRM — W169] σ inactive at every certified point; the post-solve min(pch, pdch) detector in the production path | σ: see P3-1 (`shared_energy_storage_data.py:458-460, 777-782, 822-829`; `SRP1_ESS_Params.json` `"slacks": true`); recorded per node in `component_levels_terminal.json` `esso_feasibility_violation_D3` (W169's script reads it). **Detector: in the production path.** `SharedEnergyStorageData.optimize` (l. 87-111, the ESSO entry point the ADMM loop calls) → `_optimize` → after every ESSO subproblem solve whose solution loaded, `_get_esso_complementarity_diagnostics` (l. 1373-1384, 2095-2129): max over active cohort-periods of min(pch, pdch)/S^Rated,Unit, 2·Σ min(pch, pdch), μ_final and s_obj parsed from that solve's log, and the closed-form estimate; logged and appended to `esso_complementarity_diagnostics`; persisted per solve in `leak_classification_s39_D.jsonl` of each eval dir. It is a diagnostic, not a gate (no action on its value). Remedy (h) = `ESSO_TOL_OVERRIDES` tol 1e-10 / acceptable 1e-9 (l. 1082). **Numeric part (σ): W169.** |

## Statements checked and found consistent (no rows)

Counted per printed statement (an equation counts as one; a sentence with several independent claims counts per claim).

- **§2.1: 25.** Two-stage structure with scenario-independent x (l. 319); nested decomposition, ADMM of [simoes_2023] with
  an ESS agent (l. 323); not leader–follower (l. 323); eq. (1); eq. (2) (`shared_energy_storage_data.py:384-399`,
  weights 0.35/0.55/0.10); Ω_C with ω_c (l. 369); (m, o) pair with ω_m ω_o (l. 370–372); salvage outside the objective,
  reported gross and net, from terminal SoH and remaining life (l. 379; `shared_energy_storage_data.py:866-907`,
  `shared_resources_planning.py:1302-1306`); non-anticipativity of x (l. 381); siting at the TN–DN nodes {5, 7, 9}
  (l. 383); coupling through x only, no gradient/dual/cut upward (l. 393); agent tracks SoH and available energy (l. 393);
  TSO/DSO decomposed across years (l. 393); Algorithm inputs (l. 402–406); Δ^p ← Δ^p₀ = 4 (l. 422); Halton sequence
  t = 17 + k (l. 427); full poll in batches of the slots (5) (l. 435); barrier F = +∞ with cause (l. 436–437);
  incumbent ← best determinate improver, ties lower I then label (l. 441); halving and termination at Δ = 1 failure in
  s47/s51 (l. 443–444); lattice units and closed-form constraints on poll points (l. 451); cold start per evaluation
  (l. 451); MADS granular citations (l. 451); budget-exhausted termination with the poll size reached (l. 452); no
  global optimality claim (l. 453).
- **§2.2: 33.** Single plan with low/medium/high trajectories (l. 458); eq. master_objective incl. discounting at 2 %
  (`SRP1.json` DiscountFactor 0.02); definitions l. 484–487; salvage outside F (l. 488); eqs. master_rated_s/E in year
  units (+1 limit) for single-cohort plans; E^Rated ≤ E^Max (z_E ≤ 10); eq. master_duration (z_P ≤ z_E ≤ 2 z_P);
  lattice units and S = 0 ⇒ E = 0, values 0.25/0.5/2/4 (l. 544); eq. master_budget (`shared_energy_storage_data.py:366-378`,
  `Lattice.reasons`); expected-NPV budget (l. 569); fixed parameters, no averaging, Q + record upward (l. 574);
  eq. recourse_value (`network_data.py:96-103`, probability weights inside `objective_function_rule`); in C^Op: value at
  the consensus point; closure penalty inside; row 18 inside; deviation settlement inside; contracted settlement
  excluded; agent objective excluded; solver regularisation (voltage pin) excluded; exclusions reported separately;
  Y_y weighting and discount at the representative year (l. 617–618, `shared_resources_planning.py:1234-1306`); Boyd
  test per channel with 1e-5/1e-4; every solve successful in the cycle; holds AA off / tail on / ρ frozen from k₀ + 1
  (v6 and v2 cells); k* > k₀ (l. 620); clauses 1, 2 (incl. the τ/10 floor), 3 (n_w formula, range ≤ τ), 4 (gap ≤ τ/2),
  5 (clean window; clean definition) (l. 624–635); monotone window 2P_max with P_max instance-measured (30), range ≤ τ,
  \|last step\|·2P_max ≤ τ, no fit (l. 638); uncertified at cap with slack and gap (l. 638); production exit = 10
  consecutive passes with all solves successful (l. 639); τ = δ_R 𝒱/4, δ_R 0.07, 𝒱 259,375.33, τ 4,539.07, frozen
  (`settling_criterion.py:55-57`); max{3b, 2τ} with b the band (`settling_criterion_v6.py:361-371`); ties "within
  resolution" (l. 643).
- **§2.3: 46.** Network/agent decomposition; network models per (y, d); inter-year coupling only via the agent
  (l. 648); agent tracks SoH and available energy (l. 652); A.1's three claims (joint scenarios with probability-weighted
  objective; storage and TSO interface scenario-free; agent without scenario index) (l. 654); SoC and converter limits in
  the networks, ageing chain in the agent (l. 656); plan a fixed parameter; half-open cohort life (l. 660); eq.
  cohort_rated; S^Av,Unit = S^Rated,Unit; E^Av,Unit with end-of-year SoH; mid-block point exists as an arm (l. 670–677);
  eq. available_capacity sums; publication every cycle (`shared_resources_planning.py:2980, 3697, 6075-6082, 6409-6411`)
  (l. 680–696); network variables P^Ch, P^Dch ≥ 0, Q^E, E^SoC (l. 701); eq. soc_recursion (η 0.97/0.96, Δt = 1 h);
  eq. soc_limits (0.10/0.90); eq. soc_closure initial and final; sum limit and circle (eq. converter_capability);
  eq. network_complementarity (1e-4); closure penalty per MWh inside Q; no energy carried between days; discharge
  beyond the stored energy excluded; Q^E without effect on energy or ageing (l. 744); scenario-free schedule referenced
  by every scenario (l. 745); agent's own copy (as net P, Q) and eq. esso_net_power (l. 753–760); pro-rata on rated energy
  of the cohort-sum net, vacuous for single cohorts, every reported plan single-cohort (l. 763–771); cell-side energy not
  apparent power and eq. cell_throughput (l. 774–787); eq. cycling_loss; k_e meaning; eq. soh_chain with φ^Y; SoH = 1
  before the cohort; eq. soh_floor (0.70); calibration derivation and eq. cycle_life_calibration (k = 35,851.36);
  floor the only intertemporal trade-off, no cycling cost (l. 830); cycling_loss and soh_floor linear (l. 831); no
  economic objective, outside Q (l. 835); minimises AL + penalty + regulariser; eq. esso_objective (c^σ 1e3 unweighted,
  ε^E 1e-5 over all cohort-periods); subject-to set incl. per-cohort bounds and the aggregate circle with S^Av;
  complementarity by structure; detector after every solve; investment slacks and apparent-power aggregation retired,
  Q only via the circle (l. 854); joint schedule without unique attribution (l. 860); scenarios instantiated in every
  block; TSO interface scenario-free with per-scenario TSO re-dispatch; deviation on the DSO side (l. 866–867); DSO's own
  expectation as a block variable (l. 868); π̄_t over market scenarios; same term on Q; deviation energy at the scenario
  price inside Q, contracted part excluded; TSO without deviation term; voltage pin solver-only and excluded; storage
  not deviated from; single-scenario instance unaffected; α in §3.5 (l. 891–897).

---

## Record checks behind T2-2 and T2-3 (helper `p515_s53_w171b_section2_record_checks.py`)

Zero solves; pickle blocked before any import; asserted at exit that no production module (pyomo,
`shared_resources_planning`, `network*`, `shared_energy_storage*`, `model_construction_helpers`, `admm_*`,
`helper_functions`, `definitions`) was imported; the only imported rule code is the stdlib-only `settling_criterion*.py`.
Inputs: 99 committed run-record files (all git-tracked and clean), the v6 stage spec (P_MAX = 30) and the frozen tables.
Output `data/SRP1/Results/P515S53/w171b_section2_audit/w171b_record_checks.json` (sha256 `140eb624…48b473`), manifest
`manifest_sha256.json` (`12a67922…bc07579` — output, script `1abcc77e…cd10b` and every input), `launch.log`
(`4c323455…7be742d8`), exit 0. The first run of the script (before check C was added) wrote to the same new directory; its
three uncommitted, uncited outputs were removed by me before the second run.

- **A (P̂).** Gate: `settling_criterion_v6.SettlingRuleV6` fed each cell's committed per-cycle inputs (Q, boyd_k, t_sum,
  all_clean_k) **reproduces all 46 committed decisions** (34 v6 + 7 ext + 4 A64 + `d_c52e1670` from its v5 records):
  status, k*, branch and window. With the single change P̂ = t(T₂) − t(T₀): 27 unchanged; **14 certify at another cycle**
  (`c_156ce2d1` 176→173, `c_6597a79d` 193→191, `d_d3709599` 175→173, `g_37b5c499` 185→184, `g_48749148` 170→169,
  `e_c2` 172→170, `e_c2_calfade` 172→168, `e_c3_midblock` 174→171, `e_c4` 173→171, `e_no_ageing` 177→174,
  `pb_y2025_n5_v6` 199→198, `e_soh050` 184→182, `g070_neutrality` 172→168, `d_c52e1670` 150→151), \|ΔQ*\| ≤ 0.087 τ
  (max `e_c3_midblock`, −395 €); **5 not certified by the code's k*** (`d_3632b0ae`, `d_4a82a64a`, `d_9246ed01`,
  `d_c7fee8be`, `g_9abf31d4`; the records end at k*, so the outcome under the text's P̂ is unknown).
- **B (slack).** 12 uncertified cells (10 v6 + the uncertified F2 references from their W118 records). The code's slack
  equals the frozen table's in all 12. Text slack \|Q(end) − Q(k₀)\| vs code \|Q(end) − Q(N_old)\|: e.g. `d_36686489`
  19,465 vs 7,973 €, `d_f759dd48` 31,418 vs 13,793 €, `h_f9eae48f` 13,921 vs 6,250 €.
- **C (claims).** 23 frozen-table claims scored under the uncertified form: the recomputed code bar equals the frozen
  bar and verdict in **23/23**; with the text slack the bar changes on 7 (B:n7_2h_e2 24,236 → 49,873; B:n7_2h_e3
  41,380 → 94,253; B:n7_4h_e5 23,920 → 58,394; H:m1.5 18,751 → 41,762; J:e2_to_e3, J:e3_to_e4, J:e3_value_minus_I
  27,694 → 27,090) and **no verdict changes**.

## Unexpected findings

1. **`ref:bd504ecf` (v1, k* 172) has no v6 replay queued** (see Blocked 2).
2. **TN "generation cost" is controllable TN output priced at the scenario market price** (`generation_cost`,
   `model_construction_helpers.py:2001-2008`), not generator cost curves. §2 does not define it; §3 should.
3. **l. 615–616 duplicate `\noindent`** (A.7(a) paste), harmless.
4. **The settling slack and P̂ in the accepted methods text `paragraphs_v5.md` are the code's**; the revised §2 reverted
   both to the expert's earlier wording (Addendum 66 (d), (g)).

## Not confirmed

- **Appendix A notation in eq. esso_objective.** l. 845–846 sums ℒ^{E,P} + ℒ^{E,Q}; the code multiplies the agent's AL
  terms by κ_ESSO and normalises by 2 S_ref (`shared_resources_planning.py:5505-5523`). Consistent only if the new
  Appendix A defines ℒ that way; Appendix A was not audited (order).
- **Bitwise inertness of the code differences between the master-search heads and HEAD.** Recorded in the brief (SRP1
  gates); I did not re-read the gate records.
- **The v2 (W118) certificates under the clean veto and the swing floor.** W118 records carry no per-block exit classes
  (`resettle_cycle_record.jsonl` has no `all_clean_k`), so I could not replay v6 on them; I rely on Addenda 59 (Supplement)
  and 61 for their status.
- **Numeric parts.** Closure-slack and σ activity (W169); `ref:7aa017f0` under v6 (W168); the 15–21 k€, "ten of the
  certificates … within 5 % of τ" and "at most 0.9 τ" figures (W171a, number check) — not checked here.
- **§3.5 parameter values** (outside l. 315–901) — not audited; the values the corrections assign there are listed in
  Addendum 68 and W166 "Parameters in force".
- **Scope of negative claims.** "No other CONFIRM comment": grep of l. 315–901 for `CONFIRM`. "No v6 replay of
  `ref:bd504ecf` queued": TASKS.md Addendum 68 order and `STEP6_ROUND1_CORRECTIONS.md` §C only. "Staging never
  evaluated": harness docstring and brief Addendum 28; campaign results were not scanned for multi-cohort candidates.
