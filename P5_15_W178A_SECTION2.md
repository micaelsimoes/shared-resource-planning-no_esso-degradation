# P5.15 W178a — §2.1–§2.3 as rewritten after Overleaf `e10c104`, against the code (Addendum 77, W178 item a)

**Worker, 2026-10-15. ZERO SOLVES, zero model builds, zero pickle loads. Read-only on production code, specs, case
files, the frozen JSON, `manuscript_review/` and the manuscript clone.**

- **Manuscript:** `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `9fd19e7`, sha256
  `d18384d669b7a4fbbcfb95e7ead5181e6ffdc5001f51f60d773053e1473df15d` (verified), 2,511 lines, unmodified in the clone,
  byte-identical to `manuscript_review/main_820ee1b_fixed.tex`. Nothing was pulled or committed in the clone.
- **Scope:** `git diff -U0 e10c104 9fd19e7 -- main.tex` in the clone has 44 hunks (228 new lines) in §2 (l. 403–1098)
  and 0 hunks in Appendix A.2 (l. 1792–1866). Every statement on a changed line is audited below. The A.2 list was not
  edited, but the equations its ranges span were, so it is re-checked.
- **Line numbers:** manuscript lines are at `9fd19e7`. Code lines are at repo HEAD `3312c468`.
- **Code identity:** production is identical to the campaigns' heads. For all 46 `campaign_results.json` under v6,
  extension, A64 and 3 × 3 (46 distinct `git_head_at_run`), `git diff --stat <head> HEAD -- <20 production modules +
  SRP1 case data>` reports NO CHANGE. Each diff ran as its own subprocess from an argument list. The positive control
  `353e094b` reports 5 files changed, +1,507/−84, so the pathspec is live.
- **Configuration:** frozen specs v6 `96c23404`, ext v3 `84775dc4`, A64 `44a2dce8` and 3 × 3 `231558f0` (hash
  prefixes verified). The case files in force match the spec pins:
  - `SRP1_ESS_Params.json` `39106f93` (v6 `inputs_in_force_now.ess_params_file`);
  - `SRP1_params.json` `dbfdb2a0` (`inputs_in_force_now.case_file`);
  - `SRP1.json` `61a794a7` (the 3 × 3 spec's `derived_instance.source_case_sha256`).

  The only per-cell overrides in the four specs are the ageing `model_variant` fields (`ageing_enabled`,
  `available_energy_soh_point`, `calendar_retention_per_year`, `eol_retention_r`). No spec carries a field for the
  storage block of §2.3.3. This was checked by scanning the four specs for soc / closure / complementarity / day_balance
  / ess_model / model_variant keys.

**Helper:** `p515_s53_w178a_section2_checks.py` (sha256 `ff4d835b7bd3267fa3c5650713d2515656ba9ecf042763eeb62ef174020e0773`).
- Static checks: manuscript identity; the changed-line map; static equation numbering; labels and references; the
  A.2 ranges; code identity; 90 exact-text code anchors (every one found); configuration values read as text; and the
  active-cohort sets of eq. (7) against the code window.
- Guards: `SolveProfileGuard(permitted=())` verified at exactly 0 (all four counts 0). `pickle.load`/`loads` were
  blocked for the whole run, with 0 calls. No production module was imported (asserted). Exit 0.
- Two trial runs went to the session scratchpad and are not committed.

**Outputs** are in `data/SRP1/Results/P515S53/w178a_section2/`:

| file | sha256 |
|---|---|
| `w178a_checks.json` | `23891983a6abac8d7975349b7ddade4451e9f2bdb0f2b058d60fc21fec1a38a0` |
| `manifest_sha256.json` (output, script, 68 inputs) | `7a2bb815f046713a0f3be2a1ac29160cc10ad9cfc27a13c34cadea35596fd1b9` |
| `launch.log` | `91bd79b4ee8a8fca9cb4bd206ab65f67618bca4835c10d8a818b1a56aeace9aa` |
| `w178a_bool_typing_test.json` (W100: PASS, 129,360 files, 0 failures in F1–F5) | `c4648501828e94966fc64dfffe1ca6735d92f3cf2eb0cf4bdd8b75f40e131147` |
| `w178a_bool_typing_test.log` | `d3f7ab7dc0c486fed78fc3b1aeb2490280fba80efc004aa2d8f48b2930b1436a` |
| `manifest_post_run_sha256.json` (the five files above + the script) | `9a4d174fdd884da7a5f77fc75788282134eb3159d8284de8ad87e02b50e0c49c` |

**Impact scale** (as in W171b, W172, W174b):
- **H:** a referee re-deriving from the text gets a different model or number.
- **M:** imprecise, but not a different model.
- **L:** notation or wording.

---

## Blocked on Planner

Nothing blocks the audit. Two new L rows can be forwarded to the author or expert if wanted; neither is required by
any prediction.

1. **W178a-R1 (L), eq. (7), l. 584–594: `active_cohorts` window.**
   - **Text:** y − y^Inv < T^Cal_{e,y^Inv}, in calendar years.
   - **Code:** keeps a cohort for round(T^Cal / Y_{y^Inv}) blocks (`shared_energy_storage_data.py:564–566`).
   - **Effect:** the sets are identical in both instances (SRP1 5-year blocks; 3 × 3 3-year blocks; T^Cal = 15). They
     would differ only if T^Cal were not a multiple of the block width.
   - **Recommendation:** none needed. This is the same class as the resolved T2-11/T3-7 and the carried T3-8.
2. **W178a-R2 (L), l. 755: "detailed in Appendix~\ref{app:admm_updated_implementation} … (Appendix~\ref{app:admm_updated_algorithm})".**
   - In elsarticle, `\thesection` in the appendix is "Appendix A" (`elsarticle.dtx` l. 1465). The subsection reference
     therefore prints "Appendix A.2", and the sentence reads "Appendix Appendix A.2" (twice).
   - Every other appendix reference in the paper is a bare `\ref`; the old text at `e10c104` was bare too.
   - This is static only; no compile was available. Fix: drop the two "Appendix~" prefixes.

Carried, unchanged: the two W177b A.2 rows, C-1 and C-2 (both L; section "A.2 constraint list").

## Prediction vs outcome

| prediction (expert, Addendum 77, item a) | outcome |
|---|---|
| "the network storage block and the cohort equations hold against code; at most one M row, in the wording of the closure slack or of the power limit" | **HELD.** Every equation in (20)–(28) and in (14)–(19) matches the code; details in the §2.3.2 and §2.3.3 tables. **0 M rows.** The closure-slack wording (eq. (25) with §3.5's 10⁻⁵ p.u. allowance) and the power-limit wording (eq. (26)) both match. The two new L rows sit elsewhere: eq. (7) and the l. 755 cross-reference typography. If the Planner counts eq. (7), the active-cohort set, as a "cohort equation", R1 is an L row on it; the prediction bounds only M rows, so the outcome does not change. |

**Counts** cover the changed statements of §2.1–§2.3. Each is one row of the tables below.

| subsection | statements audited | match | rows H / M / L |
|---|---|---|---|
| §2.1 (architecture, search) | 1 | 1 | 0 / 0 / 0 |
| §2.2 (master problem, §2.2.7) | 16 | 15 | 0 / 0 / 1 (R1) |
| §2.3 (subproblem, §2.3.1–2.3.4) | 25 | 24 | 0 / 0 / 1 (R2) |
| **total** | **42** | **40** | **0 / 0 / 2** |
| A.2 list (unchanged text; ranges re-resolved) | 5 items | 3 | 0 / 0 / 2 (C-1, C-2, both carried from W177b) |

**Earlier rows the rewrite resolves:**
- W174b T2-11 and T3-7, and W166 S7/S9: the lower limit max(y − T^Cal_{e,y} + 1, y₀) is replaced by eq. (7) with
  T^Cal_{e,y^Inv}.
- W174b NEW-8 for the §2.3.3 parameter pointer: it is now `sec:case_ess_params`.
- W174b NEW-10 / T3-9: the separator in the nonlinear-rows sentence.
- W177b A-3: §2.2.7 now prices the gap "at the expected wholesale price".
- W177b B1–B3, as fixed at `9fd19e7`.

## Rows by subsection

"Matches" means the code computes what the text states, at the cited lines. Notation:
- `mch` = `model_construction_helpers.py`;
- `sed` = `shared_energy_storage_data.py`;
- `srp` = `shared_resources_planning.py`;
- `s47` = `p515_s47_phase_b_record.py`;
- `s53` = `p515_s53_f2_certificate.py`;
- `w118` = `p515_s53_w118_resettle_hooks.py`.

### §2.1 — architecture and planning search

| # | line | statement (as printed) | code (file:line, Pyomo / Python name; spec field) | verdict |
|---|---|---|---|---|
| 2.1-a | 524 | "terminates when no evaluated candidate in the final unit poll improves the incumbent …, or when a poll exceeds the completion cap or the evaluation budget is reached (Section 3.6)" (B3) | `s47:205` `COMPLETION_CAP = 30`, `:656` `STOP_FOR_REVIEW_completion_cap`; `:663`/`:666` a poll whose new points exceed N^max − N is not launched (`evaluation_budget_exhausted`); `s53:155–156` (cap from s47; budget 60). §3.6 l. 1316 lists the cap (30), the budgets (20/20/60), the barrier stop rule (`s47:685`) and the 60-poll limit | matches. Algorithm 1 l. 509 states the budget test exactly; §3.6, which the sentence cites, carries the barrier rule |

### §2.2 — master problem and §2.2.7

| # | line | statement | code | verdict |
|---|---|---|---|---|
| 2.2-a | 537–550 | eqs. (4)–(5): min F(x), F = I(x) + Q(x) | `s47:702` `f_x = c['I_x_eur'] + rec['Q']`; Q = the certified `gross_operational_cost` (settlement-excluded, no salvage): `p515_s44_campaign_harness.py:2042`, `srp:1302` | matches |
| 2.2-b | 554–571 | eq. (6) and where-clause: unit costs in installation year y^Inv under trajectory c; weighted by ω_c; discounted to y₀ at r | `srp:2252–2259` (`_check_candidate_first_stage_feasibility`: Σ_c ω_c (c^S S + c^E E) (1+r)^{−(y−y₀)}); r from the case, `srp:9002` (`SRP1.json` DiscountFactor 0.02); the search uses the pinned discounted expected unit costs, cross-checked against production (`s47:769–780`); cost file pinned in v6 `inputs_in_force_now.cost_file` `e17bd588` | matches |
| 2.2-c | 575–576 | decisions common to all cost trajectories; Q(x) as defined in §2.2.7 | one plan per evaluation (`s47` `Lattice`); Q as in 2.2-a | matches |
| 2.2-d | 584–596 | eq. (7): Y^Act_{e,y} = {y^Inv ∈ Y : y^Inv ≤ y, y − y^Inv < T^Cal_{e,y^Inv}}; "contributes only after it has been made and before its calendar lifetime expires" | `sed:563–570`: rows `rated_s/e_capacity_unit` for block y ∈ [y_inv, min(y_inv + round(T^Cal/Y_{y_inv}), \|Y\|)); `es_s/e_rated_per_unit` fixed at 0 elsewhere (`sed:467`). Checks H: identical pair sets for SRP1 (years 2025/2030/2035, Y = 5) and 3 × 3 (2025…2037, Y = 3), T^Cal = 15 (`SRP1_ESS_Params.json` `ageing.calendar_life_years`) | **R1 (L).** The code counts T^Cal in rounded blocks of the investment block's width; the text counts calendar years. They are identical in both instances and differ only for T^Cal not a multiple of the block width |
| 2.2-e | 600–627 | eqs. (8)–(9): rated sums over Y^Act; where-clause (T^Cal_{e,y^Inv} "of that installation"; rated = nameplate, degradation in the subproblem) | agent: `sed:573–582` (`rated_s_capacity`, `rated_e_capacity`: Σ over all y_inv, out-of-window terms fixed at 0, i.e. Σ over Y^Act); T^Cal read per cohort, `sed:563–564`; first stage: `srp:2263` (on `total_capacity`); degradation enters only E^Av (eq. (17)) | matches. This also resolves T2-11/T3-7 and the W166 S9 "T^Cal + 1 years" difference |
| 2.2-f | 631 | additions are non-negative integer multiples of Δ^S, Δ^E | `s47:186–187` (0.25 MVA, 0.5 MWh), `:354` (negative rejected) | matches |
| 2.2-g | 636–644 | E^Rated_{e,y} ≤ E^Max_{e,y}, "maximum total rated energy capacity … in representative year y" | `srp:2263` (total capacity per node and year vs `max_capacity`); `s47:363` (`ZE_MAX` = 10 granules = 5 MWh, single cohort); `SRP1_ESS_Params.json` `max_capacity` 5.0 | matches |
| 2.2-h | 650–664 | eq. (11) on each addition, φ in MWh/MVA | `srp:2248–2251`; `s47:361` (zp ≤ ze ≤ 2 zp ⇔ E/S ∈ [2, 4] h); `min/max_energy_to_power_factor` 2.0 / 4.0 (v6 `ess_params_file`) | matches |
| 2.2-i | 668 | at unity power factor the rating equals the maximum active-power output; nominal duration only | network: `mch:887` (P^Ch + P^Dch ≤ S^Av) and `mch:871` (circle); agent: `sed:748–749` (P ≤ S^Rated,Unit), `sed:787` (circle). At Q = 0 the maximum \|P\| is S | matches |
| 2.2-j | 674–686 | eq. (12) I(x) ≤ c^Inv; budget on the probability-weighted discounted expenditure, not per trajectory | `srp:2266`; `s47:367`; `budget` 1.0e6 | matches |
| 2.2-k | 696 | the additions are fixed parameters of the subproblem; Q(x) with its stopping status and numerical diagnostics returned | `sed:1427–1438` (`es_s/e_investment_fixed` set as Params); `s47:704` (record `Q_eur`, `F_eur`, `bar_eur`, status) | matches |
| 2.2-l | 703, 718–720 | C^Op_{y,d,m,o}: TN generation, DN flexibility, load curtailment, closure- and feasibility-slack penalties; multi-scenario: commitment charge and deviation settlement | Q = Σ of the network primal values (`srp:1237–1239`), `objective_function_rule` `mch:1811–1818` (generation, flexibility, curtailment, `total_slack_penalties`, closure penalty `mch:2341`); settlement split `srp:1251–1304` | matches (substance as W174b T2-8) |
| 2.2-m | 723–724 | contracted settlement excluded; agent and solver regularisation excluded; RES curtailment at no cost | `srp:1302–1304` (contracted settlement and voltage pin subtracted); agent objective not summed into Q; `penalty_shared_ess_usage` 0 in the ADMM path (`srp:5041, 5326`); `penalty_gen_curtailment` 0 (`srp:5042, 5332`) | matches |
| 2.2-n | 727–728 | Y_y, D_d = calendar years and days represented; discounted to y₀ at the representative year | `srp:3870–3873` `_get_admm_block_weight` = Y_y · D_d · (1+r)^{−(y−y₀)}; `SRP1.json` Years 5/5/5, Days 92/91/91/91; 3 × 3 Years 3 × 5 (`derived_instance`, case `2a64e3c5`) | matches |
| 2.2-o | 733 | "No convergence guarantee is available for consensus ADMM on a nonconvex recourse … stopped by an explicit rule" (A4) | not a code property; every evaluation ends by the settling rule or its cap (W174b T2-1, E3) | consistent (no code row) |
| 2.2-p | 734 | the gap clause: disagreement "priced at the expected wholesale price" | `w118:366` `term = w * pi * (xd[p] - zt[p])`, π = `srp:1059` `_expected_market_price` (Σ_m ω_m π_{m,t}), w = block weight; active power only (A.5 C4 states "block-weighted" and "active-power") | matches; resolves W177b A-3 in §2.2.7 |

### §2.3 — subproblem

| # | line | statement | code | verdict |
|---|---|---|---|---|
| 2.3-a | 748–750 | TSO and DSO subproblems per year–day pair; one shared-ESS subproblem per interface node spanning all years and days; coordinated by ADMM | network models per `[year][day]`; `sed:74–78` (`build_subproblem`, one model per node) | matches |
| 2.3-b | 755 | agents track SoH and available energy; the local problems are in "Appendix~\ref{app:admm_updated_implementation}", Algorithm 2 in "(Appendix~\ref{app:admm_updated_algorithm})" | content: as 2.3-a and A.2 | **R2 (L), typography.** Static reading gives "Appendix Appendix A.2" / "Appendix Appendix A.6" (elsarticle `\thesection` = "Appendix A") |
| 2.3-c | 763–764 | separate charge/discharge variables; relaxed complementarity in the network models, throughput regularisation in the agent; degradation from the agent's variables | `mch:933–951` (`sess_comp`); `sed:844` (`EPS_ESSO_THROUGHPUT` × throughput, 1e-5, `definitions.py:74`); `sed:665` (throughput from `es_pch/pdch_per_unit`) | matches |

#### §2.3.2 — installed and available capacity

| # | line | statement | code | verdict |
|---|---|---|---|---|
| 2.3-d | 770–784 | eqs. (14)–(15): S^Rated,Unit = Ŝ^Inv, E^Rated,Unit = Ê^Inv, ∀ y^Inv ∈ Y^Act_{e,y} | `sed:569` / `:570` (`rated_s_capacity_unit` / `rated_e_capacity_unit`, built only inside the window); Ŝ, Ê = Params `es_s/e_investment_fixed` (`sed:1427–1438`) | matches |
| 2.3-e | 788 | rating constant over the lifetime; available energy falls with SoH at the end of the block (conservative); midpoint in §4.3 | `sed:17` `AVAILABLE_ENERGY_SOH_POINT_DEFAULT = 'end'`; `sed:628`; midpoint row `sed:626` only for `available_energy_soh_point = "mid"` (ext v3 cell `e_c3_midblock`) | matches |
| 2.3-f | 792–802 | eqs. (16)–(17): S^Av,Unit = S^Rated,Unit; E^Av,Unit = E^Rated,Unit · SoH | `sed:619` (`available_s_capacity_unit`), `sed:628` (`available_e_capacity_unit`, × `es_soh_per_unit_cumul`) | matches |
| 2.3-g | 806–820 | eqs. (18)–(19): S^Av, E^Av = sums over the active cohorts | `sed:274–280` `get_available_capacities` (Σ over all y_inv of `es_s/e_available_per_unit`, out-of-window terms 0). Computed after the solve, not a model row (see C-1) | matches |
| 2.3-h | 824–825 (unchanged) | the agent publishes S^Av, E^Av to every block each cycle | `srp:3697` (each cycle); `srp:6075–6076, 6409–6410, 6619–6620` (`shared_es_s/e_rated_fixed` ← value / s_base); `mch:1134–1135` | matches |

#### §2.3.3 — storage operation in the network models

| # | line | statement | code | verdict |
|---|---|---|---|---|
| 2.3-i | 832–833 | P^Ch, P^Dch non-negative; consumption convention; Q^E separate | `network.py:395–396` (`NonNegativeReals`), `:404` (`shared_es_qnet`) | matches |
| 2.3-j | 837–844 | eq. (20) P^E = P^Ch − P^Dch | `mch:995–999` `sess_pnet_def` | matches |
| 2.3-k | 848 | E^SoC = stored energy at the end of interval t; starts from E^SoC_{e,y,d,0} | `mch:972–980` (`shared_es_soc[p]` after period p; at p = 0 the previous state is the expression SoC⁰ · E^Av, with no variable) | matches |
| 2.3-l | 852–861 | eq. (21); Δt; powers on the network side, η on the stored energy | `mch:977–980` (`sess_soc_def`); Δt = 24/\|T\| = 1 h (`mch:808`); η from the ESS data (0.97 / 0.96, §3.5); node balance uses P^E (`mch:1419` `Pd += shared_es_pnet`) | matches |
| 2.3-m | 865–879 | eq. (22) SoC^Min E^Av ≤ E^SoC ≤ SoC^Max E^Av; SoC⁰ defined | `mch:890–903` (`sess_soc_limit_lower/upper`, × `shared_es_e_rated_fixed` = E^Av); 0.10 / 0.90 / 0.50 (`definitions.py:38–40`) | matches |
| 2.3-n | 883–892 | eq. (23) E^SoC_{e,y,d,0} = SoC⁰ E^Av | `mch:973` | matches |
| 2.3-o | 896–906 | eq. (24) E^SoC_{\|T\|} = E^SoC_0 + s⁺ − s⁻ | `mch:983–990` (`sess_soc_final`: final state = SoC⁰ · E^Av + up − down); `slacks.shared_ess.day_balance` true in all four network params files | matches |
| 2.3-p | 910 | slacks bound the deviation; penalised at c^Cl per MWh in the network objectives | `mch:2333–2341` baseMVA · 1e3 · (s⁺ + s⁻) (p.u.·h × baseMVA = MWh; `PENALTY_SHARED_ESS_BALANCE` 1e3, `definitions.py:58`); in every TSO and DSO objective via `mch:2369–2382`, `:1818`; probability weights sum to one copy (`mch:2397–2402`); §3.5 c^Cl = 10³ €/MWh | matches |
| 2.3-q | 914–918 | eq. (25) 0 ≤ s^± ≤ ε^Cl E^Av | `mch:1162–1169` (lb 0, ub `e_capacity * 0.05 + EQUALITY_TOLERANCE`, i.e. 0.05 E^Av + 10⁻⁵ p.u.; `mch:410`, `definitions.py:90`); §3.5 l. 1266: ε^Cl = 0.05 "(plus a numerical allowance of 10⁻⁵ p.u.)" | matches (the prediction's closure-slack wording) |
| 2.3-r | 922–923 | no energy carried between days; in the multi-scenario instance one schedule per block for all scenarios (day-ahead commitment) | each block starts at SoC⁰ E^Av (2.3-n); `mch:841–852` (`sess_na_scenario`, `sess_row_is_duplicate`: the first scenario pair's copy is the scenario-free schedule) | matches |
| 2.3-s | 927–936 | eq. (26) P^Ch + P^Dch ≤ S^Av, "limited by the available converter rating" | `mch:874–887` `sess_active_sum_limit` (≤ `shared_es_s_rated_fixed` = S^Av; no tolerance) | matches (the prediction's power-limit wording) |
| 2.3-t | 940–949 | eq. (27) (P^E)² + (Q^E)² ≤ (S^Av)² | `mch:855–871` `sess_converter_capability` | matches |
| 2.3-u | 953 | reactive power uses the converter, not the stored energy or the cells | Q absent from `mch:977–978` and from `sed:665` | matches |
| 2.3-v | 956–962 | eq. (28) (P^Ch/S^Av)(P^Dch/S^Av) ≤ ε^C | `mch:949–951` `sess_comp` (BILINEAR_RELAXATION: `pch_hat · pdch_hat ≤ ESS_COMPLEMENTARITY_TOLERANCE`) with `mch:906–930` (`pch = S^Av · pch_hat`); `shared_ess_model` BILINEAR_RELAXATION in all four params files; ε^C = `SMALL_TOLERANCE` 1e-4 (`definitions.py:91–92`); rows off at zero capacity (`mch:1171–1178`) | matches |
| 2.3-w | 966 | the operating parameters are in Section 3.5 | `sec:case_ess_params` l. 1266 prints η, SoC window, SoC⁰, ε^Cl, c^Cl, ε^C | matches (resolves W174b NEW-8 for this sentence) |
| 2.3-x | 967 | "Both daily closure slacks were zero, within solver tolerance, at every certified evaluation reported in this study" | W169 record `data/SRP1/Results/P515S53/w169_slack_inventory/w169_slack_inventory.json`: 51 certified points; signed closure ≤ 1e-6 relative at every certified point except the superseded `pb_y2025_n5`; every negative level within IPOPT's relaxed-bound floor; Addendum 71 ruling 6 | matches the W169 record (not re-measured here) |

#### §2.3.4 — degradation chain (changed sentence only)

| # | line | statement | code | verdict |
|---|---|---|---|---|
| 2.3-y | 1051 | (32) and (34) linear; (33), the available-energy product in (17) and the converter circle are the agent's nonlinear rows (B2) | `sed:702–703` (D · const = const · Ē, linear), `:732` (floor, linear), `:724` (exp), `:628` (Var × Var), `:787` (circle); the code's own comment `sed:593–595` names the same three | matches (resolves W174b NEW-10) |

## A.2 constraint list (l. 1845; text unchanged since `e10c104`)

The sentence reads: "subject to \eqref{eq:cohort_rated}–\eqref{eq:available_capacity},
\eqref{eq:esso_net_power}–\eqref{eq:soh_floor}, the bounds 0 ≤ P^Ch, P^Dch ≤ S^Rated,Unit … and the converter capability
inequality … ≤ (S^Av_{e,y})²".

Both ranges were resolved statically:
- **(14)–(19)** = `cohort_rated`, `cohort_rated_energy`, `cohort_available`, `cohort_available_energy`,
  `available_power`, `available_capacity`. The label now sits on the S equation (14), so the range covers both rating
  equations; this is the expert's fix at `820ee1b`.
- **(29)–(34)** = `esso_net_power`, `cohort_allocation`, `cell_throughput`, `cycling_loss`, `soh_chain`, `soh_floor`.
  (35), `cycle_life_calibration`, is correctly outside the range: it is a parameter formula, not a row.

| item | code | verdict |
|---|---|---|
| (14)–(17) | `sed:569`, `:570`, `:619`, `:628` | matches |
| (18)–(19) | not model rows: computed after the solve (`sed:274–280`). The model instead builds the aggregate **rated** rows `rated_s/e_capacity` (`sed:581–582`), which the list omits | **C-1 (L), carried from W177b, unchanged** |
| (29)–(34) | `sed:782` (with σ±, `slacks: true`), `:805` (at most N_active − 1 rows, none for one cohort), `:665`, `:702–703`, `:724`, `:732` | matches |
| bounds | `sed:748–749` (`s_max = es_s_rated_per_unit`, which equals Ŝ in the window and 0 outside); `NonNegativeReals` `sed:470–471` | matches |
| circle | `sed:786–787` bounds by `es_s_rated[y]`; this equals S^Av because S^Av,Unit = S^Rated,Unit | **C-2 (L), carried from W177b, unchanged** (same model) |

f_E and the AL terms (eqs. A.2–A.3) were not edited; W177b (c) found no difference there.

## Labels and references (static)

- **Labels:** 121 labels, 121 distinct, **0 duplicates**. `eq:cohort_rated` (l. 776), `eq:cohort_rated_energy`
  (l. 783), `eq:available_power` (l. 812) and `eq:available_capacity` (l. 819) are each defined once, so B1's duplicate
  labels are gone.
- **References:** **0 undefined** `\ref`/`\eqref` in the whole document. §2 has 49 references and A.2 has 9; all are
  defined. `eq:cohort_available_energy` is cited at l. 1051 (B2).
- **Static numbering** (comment-stripped; subequations counted once; elsarticle appendix reset) agrees with the expert's
  compile numbers (Addendum 77 ruling 5) at every point checked:
  - Q(x) (13);
  - cohort ratings (14)–(15);
  - SOC block (20)–(28), with the SOC limits at (22);
  - agent and degradation (29)–(35);
  - salvage (38)–(39) (`eq:salvage_credit`, `eq:salvage_remaining_life`).

  The full map of §2 is (1)–(37), listed in `w178a_checks.json` `C_equations_section2` with a `changed` flag per
  equation.
- **Equations on changed lines:** (4), (5), (7), (8), (9), (12), (14)–(20), (22)–(27).

  Equations (21) and (28) are unchanged, but the text around them changed. (29)–(37) are unchanged.
- **Range `\eqref{eq:master_rated_s}--\eqref{eq:master_budget}`** (l. 498, 690) now spans (8)–(12). It leaves out the
  set definition (7), which (8)–(9) use. This is consistent with "admissibility conditions".

## Not confirmed

- **Compiled output.** No TeX is installed. Equation numbers are static; they agree with the expert's compile numbers.
  R2 ("Appendix Appendix A.2") is inferred from `elsarticle.dtx` l. 1459–1469 and has not been seen in a PDF.
- **Statement 2.3-x (closure slacks zero).** It rests on the committed W169 inventory and the Addendum 71 ruling, not on
  a new measurement. W169 scored 45 of 51 certified points from committed terminal levels. The six unscored cells were
  accepted as scoped in that ruling.
- **Scope of the negative claims:**
  - **"No spec overrides the storage block":** key scan of the four frozen specs named above only. Campaign-level
    specs under each root were not scanned.
  - **"Every other appendix reference is a bare \ref":** grep for `Appendix~\ref` / `Appendix \ref` over the whole of
    main.tex at `9fd19e7`, which finds only l. 755. The `e10c104` sentence was read in the clone diff.
  - **"0 duplicate labels / 0 undefined references":** the helper's regex. It covers `\label`, `\ref`, `\eqref`,
    `\autoref`, `\cref`, `\Cref`, over the comment-stripped document, preamble included for labels.
- **Eq. (7) in general.** "Differ only when T^Cal is not a multiple of the block width" follows from the code
  (`sed:564–566`), assuming uniform block widths. Non-uniform blocks are not representable consistently in the code's
  degradation rows either (W174b T3-8, carried); I did not explore that case.
- **Ŝ^Inv / Ê^Inv** (l. 770) are defined in the text but have no nomenclature entry. Nomenclature is outside this task
  (W177b Part 3 scope), and these symbols predate `e10c104`; recorded, not rated.
