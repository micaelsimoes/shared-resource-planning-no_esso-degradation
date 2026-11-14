# P5.15 W166 — equation-vs-code audit of manuscript §2.3, the red note and Appendix A; question (5)

**Worker, 2026-10-07. Zero solves, zero model builds, zero pickle loads; static reading only.** Python was used only to
read committed JSON records (`json.load`) and to count equation rows in `main.tex`; no production module was imported,
so no solve path existed to guard. No production code, spec, frozen JSON, manuscript-clone file or committed artifact was
edited. No helper script was written.

**What was audited.** `manuscript/6a67305f25e8348fb71380c3/main.tex` (Overleaf `6191c6c`, sha256 `3cedbb6e…11c965e19`,
the submitted text): §2.3 l. 724–903, the red note l. 905–906, Appendix A l. 1409–1611. Equation numbers are **computed
from the source** (numbered display rows counted from the top of the file; elsarticle `\appendix` at l. 1403). They are
not read from a compiled PDF. They agree with the map's "(21)–(28)" for l. 794–880.

**Production configuration audited (identified from the frozen specs, not from defaults).**
- **Code.** HEAD `d60172a7`. Production files are byte-identical between every one of the 46 campaign commits behind the
  frozen tables and HEAD. The commits are the `git_head_at_run` of every `campaign_results.json` under `w142_resettle_v6/`,
  `w142_resettle_ext_v6/`, `w155_a64_cells/` and `w90_3x3/campaign_s53_w91_3x3_pair/`. The check was `git diff` over
  `shared_energy_storage*.py`, `shared_resources_planning.py`, `network*.py`, `model_construction_helpers.py`,
  `admm_*.py`, `helper_functions.py`, `definitions.py`, `solver_parameters.py`, `planning_parameters.py` and the
  `node/branch/load/generator` modules: "NO CHANGE" for all 46. The case files `data/SRP1/{SRP1.json, SRP1_params.json,
  SharedESS/, case9/, case33_*/, MarketData/}` are also unchanged since every campaign commit. Only the harness router
  `p515_s44_campaign_harness.py` changed afterwards: the W155 router lines, 3 added lines, no ADMM mechanics.
- **SRP1 cells (T1–T10).** Stage spec `w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json`. Ageing arms:
  `w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json` (`model_variant.arms`). Case file
  `SRP1_params.json` sha256 `dbfdb2a0…`. ESS parameters `SRP1_ESS_Params.json` sha256 `39106f93…`. The configuration in
  force is the case file (AA `keep_memory`), the ESS ageing baseline "C2 + φ_cal 0.985 + soh_min 0.70" and the tight
  tail `{enabled: True, compl_inf_tol: 1e-6}`. After the first residual pass the run is held at AA off, tail on and ρ
  frozen (`holds_after_first_residual_pass`; the harness hooks `p515_s53_w118_resettle_hooks.py`, reused by W142 v6), and
  it stops on settling rule v6.
- **3 × 3 instance (T11).** Campaign spec
  `w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json`, under stage spec
  `frozen_s53_spec_v36_14bbddc7.json`. Run at `8a8ef150`. Launcher `p515_s53_w90_3x3_campaign.py` (sha256 `3b9ad1fb…`).
  Instance `w89_3x3/instance/SRP1__s53_3x3.json` (sha256 `2a64e3c5…`). Configuration: the same case file and ESS file,
  row 18 α = 0.5 with no floor, the tail {True, 1e-6}, cap 500, and `required_consecutive_cycles` 10. **No settling
  hooks:** both cells `stopped_by: "boyd"` (x0 at cycle 72, `evaluation_record.json`).

---

## Blocked on Planner

1. **The map's description of the 3 × 3 instance does not match the run (outside §2.3/App. A, but it changes question 5's
   "block").** The 3 × 3 instance has **five representative years, 2025/2028/2031/2034/2037, each a 3-year block**, not
   2025/2030/2035. Sources: `SRP1__s53_3x3.json` `"Years"`; the campaign spec `derived_instance.changes_vs_source`
   ("Years" {2025:5, 2030:5, 2035:5} → {2025:3, 2028:3, 2031:3, 2034:3, 2037:3}); `p515_s53_w89_3x3_campaign.py:10-13,
   153`; the solve identity "83 = (1 + 3 DSO) × 5 years × 4 days + 3 ESSO". Map §B §3 ("three representative years 2025,
   2030, 2035 … the multi-scenario instance uses 3 market × 3 operation scenarios") would misdescribe T11's instance. The
   Planner should decide how §3 states the two instances.
2. **No other block.** Everything else the task asked for is done. Nothing here needs an author- or expert-level ruling
   before the text is drafted. Where the code and the map disagree, both are reported below; no formulation is proposed.

## Answer to question (5)

Within one block (one representative year × one representative day) of the 3 × 3 instance, the storage schedule is
**one schedule, common to all nine market × operation scenarios, in all three agents**. It is not one schedule per
scenario.
- **ESSO.** Its variables carry no scenario index.
- **TSO and DSO block models.** Each (e, t) has one scenario-free copy, `shared_es_*[e, s_m0, s_o0, t]`. Every scenario's
  balance rows reference that copy. The storage rows are built once. This is **hard non-anticipativity by aliasing**:
  there are no per-scenario copies tied by equalities (Addenda 37 (iv) and 38 (B)).
- **SoC and SoH.** There is one SoC trajectory per block and one SoH chain per cohort; neither evolves per scenario.
- **TSO interface.** The TSO's interface exchange with each DN is also scenario-free (Addendum 38 (A)).
- **Where scenario deviations are taken.** On the network side:
  - inside the TN, by per-scenario TN dispatch;
  - in each DN, through the DSO's per-scenario substation import.

**Row 18 is not a storage term.** It is a *linear* imbalance premium in each DSO block, on that DSO's per-scenario
interface P and Q deviation from its own expected interface: Σ_s Σ_t ω_s α π̄_t baseMVA (d⁺ + d⁻), with α = 0.50. It sits
inside Q. The storage contributes identically in every scenario, so it contributes zero to that deviation.

**The letter's R3.3 clause agrees with the code on the main point** (common schedule; deviations on the network side).
It is incomplete or imprecise in three places:
1. The schedule is common across market scenarios too, not only operation scenarios.
2. The storage non-anticipativity is hard, not priced. The priced term is a premium on the DSO interface flow, on the
   DSO side only.
3. "The day ends at its initial state" holds only to within a penalized slack of at most 5 % of the available energy.

Evidence is in §Q5 below.

---

## Q5 — evidence (3 × 3 instance, code at HEAD = code at `8a8ef150`)

**Index sets of the storage variables, per agent.**

| agent | variables | index set | scenario-dependent? |
|---|---|---|---|
| ESSO (one NLP per shared-ESS node, all years × days) | `es_pch_per_unit`, `es_pdch_per_unit` | (y_inv, y, d, t) | no index |
| | `es_pnet`, `es_qnet`, `slack_es_pnet_up/down` | (y, d, t) | no index |
| | `es_avg_ch_dch_per_unit`, `es_D_per_unit`, `es_soh_per_unit_cumul`, `es_*_rated/available_per_unit` | (y_inv, y) | no index |
| | source | `shared_energy_storage_data.py:457-471, 536, 548-549` (`_build_subproblem`, l. 415) | |
| TSO block (y, d) | `shared_es_pch/pdch/pch_hat/pdch_hat/pnet/qnet/soc` | (e, s_m, s_o, t) declared | **only the (s_m0, s_o0) copy is referenced** |
| | `slack_shared_es_soc_final_up/down` | (e, s_m, s_o) declared | only the (s_m0, s_o0) copy |
| | `expected_shared_ess_p/q` | (e, t) | no index |
| | source | `network.py:393-408`; `shared_resources_planning.py:4279-4285` | |
| DSO block (y, d) | same families, e = the ESS at the DN reference node | as TSO | as TSO |
| | source | `network.py:393-408`; `shared_resources_planning.py:4374-4380` | |

**How the scenario-free copy is enforced** (`model_construction_helpers.py`):
- **`sess_na_scenario` (l. 841-843)** returns the first scenario pair (s_m0, s_o0).
- **`sess_row_is_duplicate` (l. 846-852)** makes every storage rule return `Constraint.Skip` for every other pair, so
  each row exists once per (e, t). The rules are `sess_converter_capability` 855, `sess_active_sum_limit` 874,
  `sess_soc_lower/upper` 890/898, `sess_pch/pdch_hat_link` 906/925, `sess_comp` 933, `sess_soc_rule` 958,
  `sess_soc_final_rule` 983 and `sess_pnet_rule` 995.
- **Every scenario's terms reference the (s_m0, s_o0) copy:**
  - the node balance (`compute_node_load` 1411-1420);
  - the DSO interface (`interface_pf_p/q_distribution_def` 1287-1310);
  - the expected-schedule definitions (`dn_/tn_interface_expected_sess_*` 2450-2509);
  - the usage and day-balance objective terms (2215-2218, 2339-2341).
- **The other copies are declared but unreferenced** ("retained and unwired", comment at l. 811-838).

**Consensus and coupling across scenarios.** There is no per-scenario consensus channel.
- **ESS channel.** One z per (node, y, d, t, P/Q) shared by the three agents
  (`_update_shared_energy_storage_variables`, `shared_resources_planning.py:8587-8698`).
- **Storage term in the TSO and DSO.** Their AL term is written on `expected_shared_ess_p` = Σ_s ω_s pnet[s0] = pnet[s0]
  (`update_transmission_model_to_admm` 5177-5188, `update_distribution_models_to_admm` 5436-5441).
- **Interface channels.** They couple the TSO's and DSO's **expected** interface (`expected_interface_pf_p/q`, 5152-5161
  and 5423-5433).

**The TSO interface is scenario-free:**
- the ADN load `pc` is **fixed** at the DSO's initialisation expected interface, the same value for every scenario
  (`create_transmission_network_model` 4234-4242);
- the flexibility legs are fixed at 0 (4250-4253);
- the single free channel `interface_delta_p/q` is freed only for (s_m0, s_o0) (4267-4273);
- it is referenced scenario-free in `interface_pf_p/q_transmission_def` (1256-1259, 1273-1275) and in
  `compute_node_load` (1388-1393).

**Row 18, exact form** (DSO side only; `add_scenario_commitment_terms`, `model_construction_helpers.py:1877-1998`;
defining rows 1861-1874). For every (s_m, s_o, t), with s = (s_m, s_o) and ω_s = ω_m ω_o:

- p_int[s,t] − P̄_t = d⁺_P[s,t] − d⁻_P[s,t], and q_int[s,t] − Q̄_t = d⁺_Q[s,t] − d⁻_Q[s,t], with d± ≥ 0.
- p_int[s,t] = `pg_adn` = pg_ref[s,t] − Σ_e pnet_e[s0,t]: the substation import minus the scenario-free storage
  (1287-1297).
- P̄_t = `expected_interface_pf_p[t]` = Σ_s ω_s p_int[s,t]: the block's own variable (2416-2424, 2442-2443).
- Charge = Σ_s Σ_t ω_s · α · π̄_t · baseMVA · (d⁺_P + d⁻_P + d⁺_Q + d⁻_Q), with π̄_t = Σ_m ω_m π_{m,t} (1759-1770,
  1974-1982). It enters `objective_function_rule` (1835-1836), so it is inside the gross Q (`_get_operational_recourse_
  components`, `shared_resources_planning.py:1234-1306`).

Rule details:
- **Settings.** α = 0.5, floor None, set per entry by the campaign spec (`candidates[*].interface_deviation_premium`).
- **TSO.** The TSO carries no row-18 term (`premium_alpha=0.0`, 4294-4299).
- **Activation.** Row 18 is inactive at the initialisation solve and activated with the settlement weight (5221-5343).
- **In the run records.** 60 of 80 blocks are DSO blocks with row 18 wired
  (`evals/f6e9cd53fdbb8ee8_x0/multiscenario_terminal.json` `summary.n_dso_blocks_row18_wired: 60`;
  `activation_readback.json all_ok: true`). The x0 cell's weighted row-18 charge is 25,082,401 € (same file).

**What row 18 ties.** It ties the DSO's per-scenario interface flow to the DSO's expected (committed) interface flow.
**It does not tie the storage**: the storage appears identically in every p_int[s,t] and cancels in d.

**Other multi-scenario terms (not row 18):**
- **Price–deviation covariance.** The settlement deviation, Σ_t baseMVA Cov_s(π_t, p_int[s,t]), is inside Q (1704-1724;
  Addendum 38 (C)). Only the contracted part is excluded from Q (`shared_resources_planning.py:1302-1304`).
- **Interface-voltage pin.** 9e4 · Σ_s ω_s (V_s − V̄)², in the TSO and the DSO. It is **solver-only** and subtracted out
  of Q (1925-1943; `definitions.py:75`).
- **Retired.** The former quadratic scenario-deviation penalties (9e4 on V and interface P/Q, 1e4 on storage) are
  retired and unwired (`_add_tso_/_add_dso_scenario_deviation_penalty`, `shared_resources_planning.py:4078ff`).

**SoC and SoH per scenario?** No.
- **SoC.** One recursion per block: SoC_t = SoC_{t−1} + η_ch p^ch_t Δt − p^dch_t Δt/η_dch, with SoC_{−1} = 0.5 E^av
  (`sess_soc_rule` 958-980). Closure: SoC_T = 0.5 E^av + s⁺ − s⁻ (983-992).
- **SoH.** The SoH chain is in the ESSO and has no scenario index.

**Plain statement for the letter** (Worker's wording of what the code does; the author owns the letter):

> Within each block (one representative year and one representative day), the shared storage has a single
> charge/discharge and state-of-charge schedule, common to all market and operation scenarios. In the network models it
> is one scenario-free variable per hour, referenced by every scenario's power balance, and the storage agent has no
> scenario index at all. The schedule is therefore non-anticipative by construction rather than by a penalty, and the
> state-of-health chain is driven by this one schedule. Scenario deviations are absorbed on the network side:
> - the transmission operator dispatches its own network per scenario against a scenario-free interface schedule;
> - each distribution operator's substation import varies by scenario, and its deviation from the operator's own
>   expected (committed) import is charged at α = 0.5 times the hour's expected market price, in P and Q, plus the
>   price-weighted deviation energy.
>
> The state of charge evolves over the hours of the day with charging and discharging efficiencies, within
> 10–90 % of the available energy, and closes at its initial level of 50 % up to a penalized slack of at most 5 % of the
> available energy.

**What `response_to_reviewers_draft.tex` currently says** (clone `6191c6c`, sha256 `8df94963…`, R3.3, l. 283–290,
verbatim):

> "On continuity: the SOC recursion runs over the hours of each representative day with a daily closure condition (the
> day ends at its initial state), so no energy is carried between representative days; the storage schedule within a
> block is common to the operation scenarios of that block (it is a day-ahead commitment, with scenario deviations on
> the network side priced by the non-anticipativity term of Section~2.3); and continuity across representative years is
> carried by the state-of-health chain, which maps the realized cell-side throughput of each year into the available
> energy of the next. Cumulative discharge beyond the stored energy is therefore excluded by the energy limits and the
> closure condition in every day and scenario."

It is followed by the comment `% [CONFIRM against code before submission: that the storage schedule is common to the
operation scenarios within a block …]` (l. 291–293).

**Agreement with the code.**

| clause | code | verdict |
|---|---|---|
| "common to the operation scenarios of that block" | common to all market × operation scenarios | **agrees** (the "CONFIRM" comment is resolved: common, not per scenario); incomplete, since it is also common across market scenarios |
| "scenario deviations on the network side" | DSO interface per scenario; TSO internal dispatch per scenario | **agrees** |
| "priced by the non-anticipativity term of Section 2.3" | the storage non-anticipativity is hard (aliasing, unpriced); the priced term (row 18) is a linear premium α π̄_t on the DSO interface P/Q deviation, DSO side only; the TSO interface is pinned scenario-free | **imprecise**: the priced term is an imbalance premium on the interface flow, not a non-anticipativity term on the storage |
| "the day ends at its initial state" | SoC_T = 0.5 E^av ± slack, slack ≤ 0.05 E^av + 1e-5, penalised 1e3 €/MWh | **approximately**: soft closure within 5 % |
| "no energy carried between representative days" | each block's SoC starts at 0.5 E^av | **agrees** |
| "SoH chain maps realized cell-side throughput of each year into the available energy of the next" | E^av_y = E^rated · SoH_y, with SoH_y the **end-of-block-y** SoH; it is used for block y itself and propagates to y+1 | **imprecise**: it also sets block y's own available energy (baseline 'end' point) |
| "excluded by the energy limits … in every day and scenario" | SoC ≥ 0.10 E^av is a hard row, once per block, scenario-free | **agrees** |

Side note (not asked): letter R2.11–12 (l. 238–240) says the SOC dynamics and capability inequality "sit in the TSO and
DSO network models for every representative day and scenario". In the code each row is built once per block (for the
scenario-free copy), and the capability inequality is *also* in the ESSO (`shared_energy_storage_data.py:786-787`).

---

## Audit table — §2.3 (l. 724–903) and the red note (l. 905–906)

Notation in the "code form" column: E^inv, S^inv are the candidate's investment (Params); k = cl_eff; φ = φ_cal;
Y_{y_inv} is the investment block's width; E^av = available energy. **Status** is one of **matches**, **differs**,
**not implemented**. **Impact** says whether a referee re-deriving from the text would get a different object:
H = changes the model or a number; L = notation, index or wording only; – = none.

| ID | lines | printed (LaTeX) | code (file:line, component, builder) | code form in the manuscript's notation | status | impact |
|---|---|---|---|---|---|---|
| S1 | 727 | TSO subproblem, set of DSO subproblems, a dedicated shared-ESS subproblem; decomposed across representative years and days; inter-year coupling only through ESS degradation, in the ESS subproblem | `network_data.py:46-52` (`build_model`, one NLP per (y,d)); `shared_energy_storage_data.py:74-78` (`build_subproblem`, one NLP per node over all y,d) | network NLP per (y,d); ESSO NLP per shared-ESS node spanning Y×D×T; only the ESSO links years (SoH chain) | **matches** (the ESSO is one NLP *per node*, 3 in total) | L |
| S2 | 731 | the ADMM of [simoes_2023] extended by a shared-ESS agent tracking SoH and available energy across years | `shared_resources_planning.py:2777-3785` (`_run_operational_planning`) | three-agent consensus ADMM; ESSO tracks SoH and E^av, published to the networks each cycle (l. 2980, 3697) | **matches** | – |
| S3 | 733 (blue) | scenario index o omitted; "operational models are instantiated for each scenario; objective values aggregated with scenario probabilities" | `network.py:259-260` (sets `scenarios_market`, `scenarios_operation` inside one block model); `model_construction_helpers.py:2230-2235, 2389-2402` (probability-weighted totals) | one NLP per block containing **all** (s_m,s_o), objective Σ_s ω_m ω_o c_s; the shared-ESS schedule and the TSO interface are scenario-free inside it; the ESSO has no scenario index | **differs**: the models are not instantiated per scenario; scenarios are joint within each block, with hard non-anticipativity of the storage and the TSO interface | H |
| S4 | 739-741 (13) | S^{Inv}_{e,y} = \hat S^{Inv}_{e,y} + S^{Up}_{e,y} − S^{Down}_{e,y} | retired P5.15-1: `shared_energy_storage_data.py:441-450, 554-557` (investment Vars, slacks and `energy_storage_capacity_fixing` retired; `es_s_investment_fixed` is a mutable Param) | S^inv is data | **not implemented** (retired) | H |
| S5 | 743-745 (14) | E^{Inv}_{e,y} = \hat E^{Inv}_{e,y} + E^{Up} − E^{Down} | as S4 (`es_e_investment_fixed`) | E^inv is data | **not implemented** (retired) | H |
| S6 | 749-750 | \hat S, \hat E = fixed first-stage plan (blue); S^{Up}… slack variables penalised to keep feasibility | as S4; the Param is set by `_update_model_with_candidate_solution` (1427-1446) | the plan is fixed with no slack | **not implemented** (the slacks are retired; the "fixed plan" clause matches) | H |
| S7 | 754-756 (15) | S^{Rated,Unit}_{e,y^{Inv},y} := S^{Inv}_{e,y^{Inv}}, y ∈ [y^{Inv}; y^{Inv}+T^{Cal}] | `shared_energy_storage_data.py:560-570` (`rated_s_capacity_unit`); outside the window the Var is fixed at 0 (467, 567) | S^{rated,unit}_{y_inv,y} = S^inv_{y_inv} for y ∈ {y_inv, …, min(y_inv+round(T^Cal/Y_{y_inv}),\|Y\|)−1} (block indices, half-open); 0 otherwise | **differs** (index set: block-index window of round(15/Y) blocks = T^Cal years exactly; the text's closed interval mixes year and block units) | L |
| S8 | 758-760 (16) | E^{Rated,Unit} := E^{Inv}, same window | `shared_energy_storage_data.py:560-570` (`rated_e_capacity_unit`) | as S7 for E | **differs** (index set, as S7) | L |
| S9 | 768-770 (17) | S^{Rated}_{e,y} = Σ_{y^{Inv}=max(y−T^{Cal},y_0)}^{y} S^{Rated,Unit} | `shared_energy_storage_data.py:573-582` (`rated_s_capacity`) | S^rated_y = Σ_{all y_inv} S^{rated,unit}_{y_inv,y}; out-of-window terms fixed at 0 | **differs** (index set: the printed lower limit counts a unit installed exactly T^Cal earlier, so T^Cal+1 years of life; the code counts T^Cal; immaterial in both instances, whose horizons end first) | L |
| S10 | 772-774 (18) | E^{Rated}_{e,y} = Σ E^{Rated,Unit} | `shared_energy_storage_data.py:573-582` (`rated_e_capacity`) | as S9 for E | **differs** (as S9) | L |
| S11 | 778-780 (19) | S^{Av,Unit} = S^{Rated,Unit} | `shared_energy_storage_data.py:615-619` (`available_s_capacity_unit`) | identical | **matches** | – |
| S12 | 782-784 (20) | E^{Av,Unit}_{e,y^{Inv},y} = E^{Rated,Unit} · SoH_{e,y^{Inv},y} | `shared_energy_storage_data.py:604-628` (`available_e_capacity_unit`, bilinear row 628) | E^av_{y_inv,y} = E^{rated,unit} · SoH_{y_inv,y} (SoH at the END of block y); the C3_midblock arm alone uses SoH_{y−1} e^{−D/2} φ^{Y/2} (625-626) | **matches** (baseline `available_energy_soh_point='end'`) | – |
| S13 | 786-790 | available power constant over life; available energy ∝ SoH | as S11/S12 | as stated | **matches** | – |
| S14 | 792 | "representative daily fractional capacity-loss term compounded over the calendar duration represented by each planning year" | `shared_energy_storage_data.py:668-724` (`energy_storage_capacity_degradation`) | log-domain block loss D_{y_inv,y} = 365·Y_{y_inv}·EFC/day / k, applied as e^{−D}, plus calendar retention φ^{Y} | **differs** (continuous compounding; calendar term) | H |
| S15 | 794-798 (21) | δ_{e,y^{Inv},y} = E^{Ch,Dch} / (2 CL^{Nom}_e E^{Rated,Unit}) | `shared_energy_storage_data.py:699-703` (`D·(2·cl_eff·E^inv) = 365·Y_{y_inv}·Ē`) | D_{y_inv,y} = 365 Y_{y_inv} Ē_{y_inv,y} / (2 k E^inv_{y_inv}), i.e. D = 365·Y·δ with **k = N·DoD/(−ln R) = 35,851.36** in place of CL^Nom = 10,000 | **differs**: coefficient k instead of CL^Nom (calibration C2 ACTIVE); no daily δ variable; E^inv Param instead of the E^{rated,unit} Var (equal inside the window) | H |
| S16 | 801 | throughput "measured on an apparent-energy basis" | `shared_energy_storage_data.py:632-665` (comment P5.4-C) | active cell-side energy | **differs** (apparent → active) | H |
| S17 | 805-814 (22) | E^{Ch,Dch} = Σ_d Σ_t (D_d/365) Δt (S^{Ch} + S^{Dch}) | `shared_energy_storage_data.py:653-666` (`energy_storage_charging_discharging`, l. 665) | Ē_{y_inv,y} = Σ_d Σ_t (D_d/365)(η_ch p^ch_{y_inv,y,d,t} Δt + p^dch_{y_inv,y,d,t} Δt/η_dch), with η_ch = 0.97, η_dch = 0.96, Δt = 1 h, p^ch/p^dch the ESSO's own active per-cohort powers (MW) | **differs**: the summand is the active cell-side energy η_ch p^ch + p^dch/η_dch, not apparent power | H |
| S18 | 816-821 | D_d = days represented; Δt; S^{Ch}, S^{Dch} apparent powers; D_d/365 weighting; apparent power as stress proxy | `SRP1.json` "Days" {92,91,91,91}; `model_construction_helpers.py:797-808` (`period_duration_hours`, Δt = 24/\|T\| = 1 h) | D_d and Δt match; the variables are active powers; no apparent-power proxy | **differs** (variables and proxy; D_d and Δt match) | H |
| S19 | 826-829 (23) | SoH_{e,y^{Inv},y} = SoH_{e,y^{Inv},y−1}(1−δ)^{365×Y_y} | `shared_energy_storage_data.py:713-724` (`energy_storage_capacity_degradation`, SoH row) | SoH_{y_inv,y} = SoH_{y_inv,y−1} · e^{−D_{y_inv,y}} · φ_cal^{Y_{y_inv}}, φ_cal = 0.985 | **differs**: e^{−365Yδ} instead of (1−δ)^{365Y} (equal to first order in δ); **calendar factor φ_cal^{Y} absent from the text** | H |
| S20 | 831-833 | Y_y = years represented by y; SoH_{y^{Inv}−1} = 1 | `shared_energy_storage_data.py:678, 714-716` | Y read once per cohort from the **investment** block (Y_{y_inv}) and used for every y; prev SoH = 1 at y = y_inv | **differs** (index Y_{y_inv} vs Y_y; numerically identical in both instances: SRP1 5,5,5 and 3 × 3 3,…,3); the initial condition matches | L |
| S21 | 837-839 (24) | SoH_{e,y^{Inv},y} ≥ SoH^{min}_{e,y^{Inv}} | `shared_energy_storage_data.py:726-732` (third row of `energy_storage_capacity_degradation`) | SoH_{y_inv,y} ≥ 0.70 (`SRP1_ESS_Params.json` `ageing.minimum_soh`) | **matches** | – |
| S22 | 841-842 | SoH^{min} per unit and investment year | `shared_energy_storage_parameters.py:153, 183` (`soh_min` from the file, one value for all years) | 0.70 everywhere; 0.50 only in the W155 `e_soh050` row | **matches** | – |
| S23 | 846-849 (25) | S^{Ch} S^{Dch} = 0 (ESSO) | ESSO: retired P5.15-1 (`shared_energy_storage_data.py:472-535`, 735-738); networks: `sess_comp` (`model_construction_helpers.py:933-955`) | ESSO: no complementarity row; ε-throughput regularizer only. Networks: \hat p^{ch}_t \hat p^{dch}_t ≤ 10^{-4}, \hat p = p/S (`shared_ess_model` BILINEAR_RELAXATION in `case9_params.json`, `case33_*_params.json`) | **not implemented** in the ESSO (retired); networks enforce the relaxed form only | H |
| S24 | 851 | nonconvex complementarity avoids binaries; "the relaxed formulation is used when needed" | as S23 | the networks always use the relaxed normalized form; the ESSO never has one | **differs** | L |
| S25 | 853-856 (26) | S^{Ch} S^{Dch} ≤ S^{S,Comp} (slack variable) | ESSO slack `slack_es_ch_comp_per_unit` retired (comment 472-478); networks: no slack (`slacks.shared_ess.complementarity: false`; fixed RHS 1e-4, `definitions.py:91-92`) | no slack variable anywhere | **not implemented** (as printed) | H |
| S26 | 858-859 | S^{S,Comp} penalised in the objective | as S25 | — | **not implemented** | H |
| S27 | 863-871 (27) | S^{Net}_{e,y,d,t} = Σ_{y^{Inv}} (S^{Ch} − S^{Dch}) | `shared_energy_storage_data.py:764-784` (`energy_storage_operation_agg`, l. 782) | P^{Net}_{y,d,t} = Σ_{y_inv}(p^ch − p^dch) + σ⁺_{y,d,t} − σ⁻_{y,d,t} (σ± = `slack_es_pnet_up/down`, `slacks: true` in the ESS file) | **differs**: active power, not apparent; added slack pair (penalised 1e3, see S30) | H |
| S28 | 876-880 (28) | (S^{Net})² = (P^{Net})² + (Q^{Net})² | retired P5.4-C (`shared_energy_storage_data.py:453-456, 769-776`); replacement l. 786-787 | (P^{Net}_{y,d,t})² + (Q^{Net}_{y,d,t})² ≤ (S^{rated}_y)² | **not implemented** (equality retired; replaced by the converter-circle inequality) | H |
| S29 | 882-883 | P^{Net}, Q^{Net} = "net active and reactive power requests issued by the SOs"; "power-factor limits are already enforced within the SO models" | ESSO `es_pnet`, `es_qnet` are the ESSO's own copies (`update_shared_energy_storage_model_to_admm` 5488-5523); SO models: `sess_converter_capability` 855-871, `sess_active_sum_limit` 874-887; PF rows `sess_phi_limits_*` retired P5.15-1b (779-795, unwired) | P, Q are the ESSO's local copies in consensus with z; the SO models enforce p²+q² ≤ S², p^ch+p^dch ≤ S and the 1e-4 complementarity, **no power-factor limit** | **differs** | H |
| S30 | 885-898 (29)-(30) | min Σ_e Σ_y (S^{Up}+S^{Down}+E^{Up}+E^{Down}) + Σ … S^{S,Comp} | `shared_energy_storage_data.py:815-854` (`feasibility_penalty`, `objective`); `definitions.py:63, 74` | min 10³ Σ_{y,d,t}(σ⁺ + σ⁻) + ε Σ_{y_inv,y,d,t}(p^ch + p^dch), **ε = 10⁻⁵** (plus the AL terms in the ADMM, Appendix A); `salvage_value` is built as an Expression (846-850) but is **not** in the objective | **differs**: investment and complementarity slack terms gone; P-net slack and ε-throughput terms present | H |
| S31 | 885, 900 | objective "penalizes deviations from the investment targets … and violations of the relaxed complementarity"; "a coordination-penalty function" | as S30; the ESSO objective is excluded from Q (`shared_resources_planning.py:1237-1239`: gross Q = TSO + DSO primal values only); salvage enters only `net_operational_recourse` (1305-1306) | auxiliary, no economic term; feasibility slack + ε regularizer | **differs** (what it penalises); "coordination-penalty, not economic" matches | H |
| S32 | 902 (blue) | degradation reflects total utilisation from coordinated operation; SO P/Q requests reconciled with the physical and degradation constraints; joint schedule; no unique attribution | ESSO throughput is its own p^ch/p^dch in consensus with z (8587-8698) | as stated | **matches** | – |
| R1 | 905-906 (red) | "coordination determines a probability-weighted expected day-ahead schedule; scenario-dependent realizations may deviate from this schedule, with weighted quadratic penalties used to limit schedule dispersion" | `model_construction_helpers.py:811-852` (scenario-free storage), 1877-1998 (row 18 + voltage pin); retired quadratics `shared_resources_planning.py:4078ff` | storage: a single scenario-free schedule (no deviation exists); TSO interface: scenario-free; DSO interface: per-scenario deviation d from its own expected import, **linear** premium α π̄_t baseMVA (\|d_P\|+\|d_Q\|), α = 0.5, inside Q; the only quadratic left is the interface-voltage pin, 9e4 Σω(V−V̄)², solver-only | **differs** (storage dispersion is zero by construction; the dispersion penalty is linear and on the DSO interface flow; the quadratic penalties are retired except the voltage pin) | H |

## Audit table — Appendix A (l. 1409–1611), including Algorithm 1 (l. 1421–1502)

Notation: a = 1/(2 S_ref), S_ref = 2.5 MVA; κ_ESSO = σ / median_b w_b; z = the three-agent ESS consensus variable;
x_i = agent i's copy.

| ID | lines | printed (LaTeX) | code (file:line, component, builder) | code form in the manuscript's notation | status | impact |
|---|---|---|---|---|---|---|
| A1 | 1412-1413 | consensus ADMM of [simoes_2023] plus a shared-ESS agent modelling degradation | `shared_resources_planning.py:2777ff` | as stated | **matches** | – |
| A2 | 1418-1419 | sequential update structure kept; third entity enforces consistency between the SOs' requested storage schedules and the physical and degradation constraints | loop 3066-3210 (DSO → TSO → ESSO); ESS consensus 8587-8698 | Gauss–Seidel DSO → TSO → ESSO; the storage is a three-agent global-variable consensus on z | **matches** (z is a consensus variable, not "the SOs' requested schedules", see A23) | L |
| A3 | 1424-1428 | DSO_i: create model; warm start assuming V = 1.0 p.u. at the interface; update to ADMM version | `create_distribution_networks_models_sequential` 4348-4424 (build, expected vars, row 18 inactive 4395, solve 4398); ref-node voltage `e_bounds` (`model_construction_helpers.py:72-80`, Vg ± 1e-4; Vg = 1.0 in `case33_*_<year>.json`); ADMM conversion `update_distribution_models_to_admm` 5346ff, called at 2931 after all three initial solves | as stated; the ADMM conversion runs after the TSO and ESSO initial solves, not per DSO | **matches** (the order of the conversion differs) | L |
| A4 | 1431-1433 | TSO: warm start "initializing V^{I,0}, P^{I,0}, Q^{I,0} with forecasted values"; update to ADMM | `create_transmission_network_model` 4197-4326 | the ADN load P/Q is **fixed** at the DSO's initialisation expected interface (4234-4242), not at forecasts; the interface V is free; a scenario-free Δ is the only interface flexibility | **differs** (initial interface = the DSO's initial solution) | L |
| A5 | 1436-1437 | k = 1; while k ≤ k^{max} | 3066 (`range(1, num_max_iters+1)`) | k^max = `num_max_iters` (case file 300; overridden to the spec cap: 3 × 3 cap 500; SRP1 cells per spec v6) | **matches** | – |
| A6 | 1442-1448 | DSO_i sets \hat V^I, \hat P^I, \hat Q^I, \hat P^E, \hat Q^E "to Shared ESSs' targets" | `update_distribution_coordination_models_and_solve_sequential` 6413-6427 | interface targets = the **TSO's** current copies (V, P, Q); storage targets = **z** | **differs** | H |
| A7 | 1449 | DSO_i solves | 6564-6569 (`distribution_network.optimize`) | as stated | **matches** | – |
| A8 | 1450-1455 | DSO_i updates π^{I,V}, π^{I,P}, π^{I,Q}, π^{E,P}, π^{E,Q} | 3129-3136 (`update_flags` dns only); 8473-8507 (interface duals only when `update_tn`); 8574-8586 (ESS duals only after the ESSO) | no dual update after the DSO solve: only the DSO's copies are recorded | **differs** | H |
| A9 | 1459-1460, 1480-1481, 1495-1496 | "ADMM: evaluate convergence criteria; if converged exit" after each agent | `check_convergence=False` at 3135, 3177, 3209; evaluation once at 3216-3390 | once per cycle, after the ESSO: Boyd r ≤ ε_pri and s ≤ ε_dual on V, PF and ESS, **and** every local NLP solved successfully (`_admm_local_solves_succeeded` 7734-7745); production exit after 10 consecutive such cycles (3390, case file `minimum_consecutive_converged_cycles` 10; the 3 × 3 cells stopped this way); SRP1 tables: settling rule v6 (harness) | **differs** | H |
| A10 | 1464-1470 | TSO sets \hat V, \hat P, \hat Q, \hat P^E, \hat Q^E "to DSOs' targets" | `update_transmission_coordination_model_and_solve` 6084-6099 | interface targets = the DSOs' fresh copies (matches); storage targets = **z** (not the DSOs') | **differs** (storage targets) | H |
| A11 | 1471 | TSO solves | 6201 | as stated | **matches** | – |
| A12 | 1472-1477 | TSO updates π^{I,·} and π^{E,P}, π^{E,Q} | 3171-3178 → 8488-8507 | interface duals of **both** TSO and DSO updated here: λ^{I}_{TSO} += ρ (x_TSO − x_DSO)/R_int, λ^{I}_{DSO} += ρ (x_DSO − x_TSO)/R_int (V unnormalised, kV); no ESS dual update | **differs** | H |
| A13 | 1485-1488 | ESSO sets \hat P^E, \hat Q^E "to TSO's targets" | `update_shared_energy_storages_coordination_model_and_solve` 6658-6681 (called with `consensus_vars['ess']['z']`, 3192-3196) | ESSO targets = **z** | **differs** | H |
| A14 | 1489 | ESSO solves | 6686 (`shared_ess_data.optimize`; IPOPT tol 1e-10 / acceptable 1e-9, `shared_energy_storage_data.py:1082, 96-112`) | as stated | **matches** | – |
| A15 | 1490-1492 | ESSO updates π^{E,P}, π^{E,Q} | 3199-3210 → 8587-8698 | z-update z = Σ_i(ρ_i a² x_i + λ_i a)/Σ_i ρ_i a², then λ_i += ρ_i a (x_i − z) for **all three** agents i ∈ {TSO, DSO, ESSO}; skipped for any (node,y,d) where any of the three solves failed | **differs** | H |
| A16 | 1498 | k = k+1 | loop | — | **matches** | – |
| A17 | 1507-1508 | the ESSO reconciles SO P/Q schedules with the storage's "internal physical limits" and degradation dynamics | ESSO model (`_build_subproblem` 415-863) | ESSO limits: per-cohort p ≤ S^{rated,unit} (739-749), the converter circle (786-787), the SoH chain; **no energy/SoC limit in the ESSO** (those live in the TSO/DSO models, Q5) | **differs** (the ESSO holds no energy limits) | H |
| A18 | 1515-1517 (A.1a) | min f(X) + Σ_{e,y,d,t} L^{E,P}(P^E, \hat P^E, π^{E,P}) + Σ L^{E,Q}(…) | `update_shared_energy_storage_model_to_admm` 5455-5529 | min f(X) + κ_ESSO Σ_{y,d,t}[λ^P (P−z_P)a + (ρ/2)((P−z_P)a)² + same for Q], κ_ESSO = σ/median w_b (`esso_al_scale: sigma_over_median_block_weight`) | **differs** (normalisation a = 1/(2S_ref); scale κ_ESSO; \hat P = z) | H |
| A19 | 1518-1519 (A.1b) | h(X, P^{E^k}_{e,t}, Q^{E^k}_{e,t}) ≤ 0 | ESSO constraints (S11-S29) | h = the ESSO constraint set, indexed (y,d,t) | **matches** (the subscript e,t omits y,d: notation) | L |
| A20 | 1526-1534 | sets E^S ⊆ E at TN–DN interface nodes; Y; D; T; K; X | `create_shared_energy_storages` 177-189 (one shared ESS per active DN node: 5, 7, 9) | E^S = {5,7,9}; \|T\| = 24; \|D\| = 4; Y = 3 blocks (SRP1) or 5 (3 × 3) | **matches** | – |
| A21 | 1537 | f(X): local objective of the shared-ESS subproblem | S30 | 10³Σσ± + εΣ(p^ch+p^dch) | **matches** (as a symbol; its content differs from (29)-(30), see S30) | – |
| A22 | 1541 | L^{E,P}, L^{E,Q}: augmented Lagrangians for storage P and Q | 5512-5523 | as A18 | **matches** (as a definition) | – |
| A23 | 1545-1546 | P^{E^k}_{e,y,d,t}, Q^{E^k}; "\hat P^{E^k}, \hat Q^{E^k} … local copies representing the counterpart SO requests" | 6673-6681 (`p_req` = z) | \hat P = z^k, the three-agent consensus variable (a ρ- and λ-weighted average of the TSO, DSO and ESSO copies), not an SO request | **differs** | H |
| A24 | 1547 | π^{E,P}_{e,y,d,t}, π^{E,Q}: dual variables | `create_admm_variables` 4767-4771, 4869-4871 | three multipliers per (node,y,d,t,P/Q): λ_TSO, λ_DSO, λ_ESSO | **differs** (one per agent, not one) | H |
| A25 | 1551 | h(·): internal constraints | as A19 | — | **matches** | – |
| A26 | 1557-1561 (A.2)-(A.3) | L^{E,P} = π^{E,P}((P−\hat P)/S^{Rated}_{e,y}) + (ρ^{E,P}/2)‖(P−\hat P)/S^{Rated}_{e,y}‖² | ESSO 5505-5523; TSO 5177-5188; DSO 5436-5441; normalisation `_shared_ess_admm_normalization_mva` 5018-5021 with `reference_mva` = 2.5 (`SRP1_params.json` `shared_ess_reference_rating_mva`) | L = κ(λ (P−z)/(2 S_ref) + (ρ/2)((P−z)/(2 S_ref))²), **2 S_ref = 5 MVA, fixed for every node, year and candidate** (κ = κ_ESSO in the ESSO, 1 in the TSO/DSO, whose base objective is instead divided by σ/w_b) | **differs** (per-unit base: 2·S_ref = 5 MVA, not S^{Rated}_{e,y}; factor 2; κ_ESSO) | H |
| A27 | 1563-1567 (A.4)-(A.5) | L^{E,Q}, same with Q | same lines | as A26 for Q | **differs** (as A26) | H |
| A28 | 1569-1570 | ρ^{E,P}, ρ^{E,Q}: penalties for the P and Q consensus | Params `rho_ess` (TSO 5122, DSO 5388), ESSO `rho` (5483) | **one** ρ_ess per agent for both P and Q, uniform across agents (asserted for AA, 7319-7362); initial 0.01 | **differs** (no separate P and Q penalties) | L |
| A29 | 1571 | the augmented terms steer both parties toward a common solution | as A26 | — | **matches** | – |
| A30 | 1575-1582 (A.6) | π^{E,P,k+1} := π^{E,P,k} + ρ^{E,P,k}(P^{E,k+1} − \hat P^{E,k+1}) | 8696-8698 | λ_i^{k+1} = λ_i^k + ρ_i a (x_i^{k+1} − z^{k+1}), i ∈ {TSO,DSO,ESSO}, after the z-update (8667-8691) | **differs** (normalisation a; three per-agent duals; z-update absent from the text) | H |
| A31 | 1584-1591 (A.7) | same for Q | same | as A30 | **differs** (as A30) | H |
| A32 | 1593-1594 | π = coordination multipliers penalising mismatches between ESS outputs and "the targets set by the SOs" | as A30 | mismatch against z | **differs** | L |
| A33 | 1598-1600 (A.8) | ρ^{E,P,k+1} := ρ^{E,P,k}(1 + r^{E,P}) | `_update_admm_penalties` 7978-8416; `_scale_admm_penalty` 8419-8421 | residual balancing per channel g ∈ {v, pf, ess}, one factor for all agents: ρ ← 1.5ρ if r/ε_pri > 5·s_ρ/ε_dual; ρ ← ρ/1.5 if s_ρ/ε_dual > 5·r/ε_pri (3 for the pf decrease); clamp [1e-4, 1e4]; **ESS exempt (fixed at 0.01) until its full dual ratio < 1 on 5 consecutive cycles**; a channel freezes after 10 unchanged cycles once it has acted; all channels freeze at cycle 200; SRP1 campaigns also freeze ρ from k0 (harness hold); failure cycles hold ρ | **differs** (balancing, not monotone growth) | H |
| A34 | 1602-1604 (A.9) | ρ^{E,Q,k+1} := ρ^{E,Q,k}(1 + r^{E,Q}) | as A33 | as A33 (P and Q share ρ_ess) | **differs** (as A33) | H |
| A35 | 1606-1607 | r^{E,P}, r^{E,Q} user-defined update rates | `admm_parameters.py` `penalty_update` (no such keys; factors 1.5/1.5 in `SRP1_params.json`) | — | **not implemented** (no per-P/Q rate parameter exists) | L |

(The text at l. 1609 is a LaTeX comment, not printed; no row.)

### Counts by status

Counted mechanically from the status and impact columns of the two tables above.

| | matches | differs | not implemented | total rows |
|---|---|---|---|---|
| §2.3 (S1–S32) + red note (R1) | 8 | 18 | 7 | 33 |
| Appendix A + Algorithm (A1–A35) | 14 | 20 | 1 | 35 |
| **total** | **22** | **38** | **8** | **68** |

- **matches:** S1, S2, S11, S12, S13, S21, S22, S32; A1, A2, A3, A5, A7, A11, A14, A16, A19, A20, A21, A22, A25, A29.
- **not implemented:** S4, S5, S6, S23, S25, S26, S28 (retired); A35 (never existed).
- **differs:** all other rows.
- **Impact:** H 36, L 14, none 18.

---

## Implemented but not in the text (production configuration, as run)

Each item is a constraint family, objective term or ADMM mechanism that the production configuration contains and that
§2.3 / Appendix A do not state.

**Shared-ESS physics in the TSO and DSO network models** (`network.py:393-408, 499-512`; rules
`model_construction_helpers.py:855-999`), once per block and scenario-free:
1. p^ch, p^dch ≥ 0; P = p^ch − p^dch.
2. p^ch + p^dch ≤ S.
3. P² + Q² ≤ S².
4. p^ch = S \hat p^ch, p^dch = S \hat p^dch, \hat p ∈ [0,1].
5. \hat p^ch \hat p^dch ≤ 10⁻⁴.
6. **SoC_t = SoC_{t−1} + η_ch p^ch_t Δt − p^dch_t Δt/η_dch**, with SoC_{−1} = 0.5 E.
7. 0.10 E ≤ SoC_t ≤ 0.90 E.
8. **SoC_T = 0.5 E + s⁺ − s⁻**, with 0 ≤ s± ≤ 0.05 E + 10⁻⁵ (1162-1169).
9. Penalty 10³·baseMVA·(s⁺ + s⁻), inside Q (2333-2342).

Here S and E are the ESSO's **available** capacities, S^rated_y and Σ E^rated·SoH_y. They are published to every network
block before each cycle (`get_updated_capacities`, `shared_energy_storage_data.py:264-280`; set at
`shared_resources_planning.py:6075-6082, 6409-6411`; refreshed at 3697). **This is the path by which degradation reaches
operation.** Zero-capacity gating fixes everything at 0 and deactivates the rows (1117-1202).

**ESSO model** (`shared_energy_storage_data.py`):
10. per-cohort p^ch, p^dch ≤ S^{rated,unit} (739-749);
11. H3 pro-rata cohort rows (751-805, 1639-1737), inactive for single-cohort candidates — every candidate in the tables;
12. cohort activation and deactivation (1586-1636);
13. ESSO IPOPT tol 1e-10 / acceptable 1e-9 (1082);
14. the terminal salvage Expression (866-907), reporting-only.

**Network objectives in the ADMM:**
15. interface energy settlement, weight 1 (`_prepare_*_objectives_for_admm` 5033-5050, 5316-5343): the contracted part
    is excluded from Q, the covariance part is inside Q (`model_construction_helpers.py:1693-1724`;
    `shared_resources_planning.py:1251-1304`);
16. shared-ESS usage penalty and TN/DN RES-curtailment penalty zeroed (5041-5042, 5326-5332).

**Multi-scenario terms** (3 × 3 only; vacuous at one scenario, `add_scenario_commitment_terms` 1917-1920):
17. hard non-anticipativity of the storage by aliasing (811-852);
18. scenario-free TSO interface Δ, with the ADN load fixed (4234-4273);
19. row 18 (α = 0.5, DSO side, inside Q), inactive at the initialisation solve and activated with the settlement
    (5221-5343);
20. the interface-voltage pin 9e4 Σω(V−V̄)², solver-only (1925-1943, 1837-1842).

**Common objective scale and block weights:**
21. TSO and DSO block objectives are divided by σ/w_b, with **σ = 93,635,360** (fixed, `SRP1_params.json`
    `objective_scale`; computed values 93,635,428 at SRP1 and 48,004,110 at the 3 × 3, asserted within ×3,
    3976-4011) and **w_b = Y_y · D_d · 1.02^{−(y−y0)}** (3870-3873; 5136-5142, 5401-5407).

**Interface-channel AL terms** (TSO 5144-5161, DSO 5419-5433; not printed — App. A covers storage only):
22. λ_V (V − \hat V) + (ρ_v/2)(V − \hat V)², in p.u.;
23. λ_PF (P − \hat P)/R_int + (ρ_pf/2)((P − \hat P)/R_int)², and the same for Q, with R_int the DN interface-transformer
    rating in p.u.;
24. the consensus is on the **expected** interface (Σ_s ω_s);
25. the dual updates are at 8488-8507.

**TSO proximal term:**
26. γ/2 ‖x − x_prev‖² on V, PF and ESS, wired with `gamma_policy: tied_to_rho`, **τ = 0, so γ = 0** (recorded
    `gamma_*: 0.0` on all 252 cycles of `l_195156fa` `g_s39_D.json`). It is effectively off.

**ESS consensus mechanics:**
27. three-agent weighted global-variable consensus with the z-update and per-agent duals (8587-8698);
28. z initialised as (x_TSO + x_DSO + x_ESSO)/3 after the three initial solves (4880-4894; `shared_ess_initialization:
    standalone`);
29. ESSO initialisation with P, Q **fixed** at the TSO's initial storage values (4518-4559);
30. one round of interface-dual initialisation from the initial solutions (2951-2960).

**Stopping test** (per channel; 7044-7304):
31. r = ‖a(x − z)‖ and s = ‖ρ a Δz‖ (ESS); r = ‖(x_DSO − x_TSO)/base‖ and s = ρ‖Δz_TSO/base‖ (V, PF);
32. ε_pri = √n ε_abs + ε_rel max(‖x‖,‖z‖), ε_dual = √n ε_abs + ε_rel ‖y‖, with ε_abs = 1e-5 and ε_rel = 1e-4;
33. pass AND every local solve successful (`solver_result_succeeded`: optimal / locallyOptimal / globallyOptimal,
    `helper_functions.py:74-85`);
34. 10 consecutive passing cycles (production exit; the 3 × 3 cells);
35. the objective-change and legacy consensus tests are computed but diagnostic only (3338-3375, 3384-3390).

**Residual balancing** (A33's row):
36. the ESS two-phase schedule, i.e. exempt until dual ratio < 1 for 5 cycles, then balanced (8203-8234;
    `balancing_exempt_until.ess`);
37. per-channel freeze after 10 unchanged cycles (8296-8322);
38. backstop at cycle 200 (8125);
39. failure hold (8277-8278).

**Anderson acceleration** (`admm_anderson_acceleration.py`; orchestration 3030-3042, 3092-3096, 3272-3280, 3419-3428):
40. type-II on w = (z, u = y/ρ) over the V, PF and ESS channels;
41. memory 5, Tikhonov 1e-10;
42. ratchet safeguard on √Σ(r² + s²): accept iff below the value at the last accepted step;
43. reject policy `keep_memory` (memory kept on rejection);
44. memory cleared on any ρ change and on a local-solve failure;
45. off on any cycle where all channels pass Boyd.

**Tight tail** (7427-7519, applied 3077-3085, 3394-3398):
46. `compl_inf_tol` = 1e-6 on every TSO and DSO solve in the cycle after a cycle with all channels inside Boyd and all
    solves successful; restored otherwise and at exit. The ESSO is untouched. Production values: TSO 5e-4
    (`case9_params.json`), DSO the IPOPT default 1e-4.

**Certification regime of the SRP1 tables** (harness-installed wrappers, not production code;
`p515_s53_w118_resettle_hooks.py:1-60`, reused by `p515_s53_w142_resettle_v6_hooks.py`):
47. from the run's first residual pass k0: AA off (even across a lapse), tail on, ρ frozen;
48. settling rule v6 decides the stop (paragraphs_v5 (ii)).

The 3 × 3 cells did **not** use these holds: production Boyd rule, 10 consecutive cycles, `stopped_by: "boyd"`. The map
says "the settling criterion lives in §2.2.7"; whichever section states it must say which instance used which rule.

**Failure handling:**
49. no consensus or dual update from a (node, y, d) whose TSO, DSO or ESSO solve failed (8602-8609; interface 8485-8489);
50. ρ held (8277);
51. AA memory cleared (3279-3280);
52. retry tiers: `recovery_options` acceptable_tol 1e-4, acceptable_iter 1 (case files) — accepted acceptable-level
    exits enter the clean-solve clause of the settling rule;
53. network `max_iter` 500 (`network.py:559`).

## Parameters in force

| parameter | value in force | source |
|---|---|---|
| η_ch, η_dch (shared ESS) | 0.97, 0.96 | `shared_energy_storage.py:14-15` class defaults; no file override found (the only `eff_ch` file read is for ordinary ESS, `network.py:1186`) |
| Δt | 1 h (24/\|T\|, \|T\| = 24) | `definitions.py:94`; `model_construction_helpers.py:797-808`; `SRP1.json` NumInstants 24 |
| D_d | Spring 92, Summer 91, Autumn 91, Winter 91 | `SRP1.json`, `SRP1__s53_3x3.json` "Days" |
| Y_y | SRP1: 5 (2025, 2030, 2035); 3 × 3: 3 (2025, 2028, 2031, 2034, 2037) | `SRP1.json`; `SRP1__s53_3x3.json` |
| discount r (block weight, annualisation) | 0.02 | `SRP1.json` DiscountFactor |
| k (cl_eff), baseline C2 | 35,851.36 = 10000·0.8/(−ln 0.8) | `SRP1_ESS_Params.json` `ageing.calibration` (ACTIVE); `shared_energy_storage_parameters.py:83-91, 188-201`; arm k in ext spec v3 `model_variant.k_closed_form_by_arm` (C3: 11,541.56; C4: 22,429.39) |
| CL^Nom, DoD_nom | 10,000; 0.80 (CL^Nom not consumed while the calibration is ACTIVE) | `SRP1_ESS_Params.json` |
| φ_cal | **0.985 /yr** (baseline, file); 1.0 in ext arms C2, C3_unit, C3_midblock, C4, no_ageing; 0.985 in C2_calfade | `SRP1_ESS_Params.json`; ext spec v3 `model_variant.arms`; 3 × 3 spec `ess_ageing_baseline` |
| T^Cal | 15 years → round(15/5) = 3 blocks (SRP1), round(15/3) = 5 blocks (3 × 3) | `SRP1_ESS_Params.json`; `shared_energy_storage_data.py:564-565` |
| SoH^min (floor) | **0.70** (baseline); 0.50 in the W155 `e_soh050` row | `SRP1_ESS_Params.json` `minimum_soh` |
| SoH point for E^av | 'end' (baseline); 'mid' only in C3_midblock | `shared_energy_storage_data.py:16-17`; ext spec v3 |
| SoC window, initial/final | 0.10–0.90 E^av; 0.5 E^av | `definitions.py:38-40` |
| SoC closure slack bound, penalty | 0.05 E^av + 1e-5; 1e3 × baseMVA | `model_construction_helpers.py:410, 1162-1169`; `definitions.py:58` |
| network complementarity | \hat p^ch \hat p^dch ≤ 1e-4 | `definitions.py:91-92`; `shared_ess_model: BILINEAR_RELAXATION` (`case9_params.json`, `case33_*_params.json`) |
| ε (ESSO throughput regularizer) | 1e-5 | `definitions.py:74` |
| ESSO P-net slack penalty | 1e3 (slacks enabled) | `definitions.py:63`; `SRP1_ESS_Params.json` "slacks": true |
| ESSO IPOPT | tol 1e-10, acceptable_tol 1e-9 (override), acceptable_iter 5, ma57; recovery acceptable_tol 1e-4 / iter 1 | `shared_energy_storage_data.py:1082`; `SRP1_ESS_Params.json` |
| network IPOPT | TSO tol 1e-5, acceptable 1e-4, compl_inf_tol 5e-4, ma97; DSO tol 1e-5, acceptable 1e-4, compl_inf_tol IPOPT default 1e-4, ma97; max_iter 500 | `case9_params.json`, `case33_*_params.json`; `network.py:559` |
| tight tail | compl_inf_tol 1e-6 (TSO and DSO) | v6 spec `inputs_in_force_now.configuration_now.convergence_depth_tail`; 3 × 3 spec `configuration.convergence_depth_tail` |
| baseMVA | 100 (TN and DN) | `case9_2025.json`, `case33_1_2025.json` |
| ρ initial | v 0.0077, pf 0.198, ess 0.01 (TSO and DSOs), ESSO 0.01 | `SRP1_params.json` `admm.rho`; recorded `rho_*_before` at cycle 1 |
| residual balancing | ratio 5 (pf decrease 3); ×1.5 / ÷1.5; clamp [1e-4, 1e4]; freeze 10 unchanged; backstop 200; ESS exempt until dual ratio < 1.0 on 5 cycles; adaptive on | `SRP1_params.json` `admm.penalty_update`, `adaptive_penalty` |
| Boyd | ε_abs 1e-5, ε_rel 1e-4 (`boyd_eps_source: case_file`) | `SRP1_params.json` `admm.tol.boyd` |
| consecutive cycles | 10 | `SRP1_params.json`; 3 × 3 spec `required_consecutive_cycles` |
| cap | 3 × 3: 500; SRP1: gated N_old+100 / ungated min(k0+109, 300) | 3 × 3 spec `cap`; `p515_s53_w142_resettle_v6_hooks.py` constants / v6 spec |
| σ (common objective scale) | 93,635,360 (fixed; assert factor 3) | `SRP1_params.json` `objective_scale`; recorded `sigma_fixed` |
| κ_ESSO (D5) | σ / median w_b = 227,210.997 (SRP1), 386,258.694 (3 × 3) | `esso_al_scale: sigma_over_median_block_weight`; recorded `al_scale_esso` (`l_195156fa/g_s39_D.json`; 3 × 3 x0 `g_s39_D.json`) |
| S_ref | 2.5 MVA (normalisation 2 S_ref = 5 MVA); the 0.10 MVA floor is inactive while S_ref is set | `SRP1_params.json` `shared_ess_reference_rating_mva`, `shared_ess_normalization_floor_mva` |
| interface normalisation | DN interface-branch rating (`get_interface_branch_rating()`) | `shared_resources_planning.py:5148, 5417` (values not read; see Not confirmed) |
| proximal | enabled, TSO only, tied_to_rho, τ = 0 → γ = 0 | `SRP1_params.json` `proximal_regularization`; recorded `gamma_*: 0.0` |
| AA | enabled, memory 5, regularization 1e-10, reject_policy keep_memory | `SRP1_params.json` `anderson_acceleration` |
| row 18 | α = 0.5, floor None (3 × 3 only) | 3 × 3 spec `candidates[*].interface_deviation_premium`; G10 `premium_applied.after.alpha 0.5` |
| voltage pin | 9e4 (3 × 3 only, solver-only) | `definitions.py:75` |
| shared-ESS initialisation | standalone | `SRP1_params.json` |

## Claims in map §B about §2.3 / Appendix A that the code does not bear out (or only in part)

1. **"The SoC recursion with daily closure."** It exists, but **only in the TSO and DSO network models**, not in the
   shared-ESS agent, which has no SoC or energy limit. The closure is **soft**: ±(0.05 E^av + 1e-5), penalised 1e3 €/MWh
   inside Q. Initial and final SoC are 0.5 × the *degraded* available energy.
2. **"The ESSO objective carries no economic term (ε-throughput regularizer only)."** "No economic term" holds: the ESSO
   objective is excluded from Q. "ε-throughput only" does not: the objective also carries **10³ × the P-net slack pair**
   (`shared_energy_storage_data.py:821-829`). The ADMM objective adds the AL terms.
3. **"Salvage reporting-only."** Borne out. Salvage is built but not in the ESSO objective; it enters only
   `net_operational_recourse`, reported beside the gross Q.
4. **"The red note's content (expected schedule, scenario deviations with a priced premium, α = 0.50) becomes the
   description of the row-18 non-anticipativity term."** Only partly.
   - The storage has **no** scenario deviations: its non-anticipativity is hard (structural aliasing), not a term.
   - Row 18 is a **linear imbalance premium on the DSO interface P/Q deviation** from the DSO's own expected import, DSO
     side only. The TSO interface is pinned scenario-free.
   - The red note's "weighted quadratic penalties" are **retired**, except the interface-voltage pin, which is quadratic
     and solver-only.
   - The letter (R3.3) carries the same conflation ("priced by the non-anticipativity term").
5. **"Equations (21)–(28) as printed describe apparent-energy throughput."** Only (22) is the apparent-energy throughput
   itself; (27) and (28) use apparent net power. (21) differs by its coefficient (k vs CL^Nom), (23) by its compounding
   form and the missing calendar factor, (25) and (26) by complementarity (retired in the ESSO). The rewrite has to cover
   all of them, plus (13)–(14) and (29)–(30), which are retired.
6. **"ESS two-phase schedule and freeze"; "AA (type-II, memory 5, cleared at ρ change, safeguarded, off inside
   tolerance)"; "the tight tail"; "S_ref and D5 on the ESS channel"; "ρ initial values and residual balancing".** All
   borne out, with three precisions:
   - AA memory is also cleared on a local-solve failure, and is **kept** on a safeguard rejection (`keep_memory`).
   - The tail in production is non-latching. It is held on after k0 only by the SRP1 campaign hooks, which also freeze
     ρ. The 3 × 3 ran without them.
   - The TSO proximal term is wired but at **γ = 0**.
7. **Not mentioned by the map but changes the Appendix:**
   - the algorithm's ordering and targets: convergence checked once per cycle; DSO and TSO storage targets = z, not each
     other's; ESS duals per agent after the ESSO only;
   - the per-agent normalisation factor 1/(2 S_ref) in the dual update;
   - the publication of the ESSO's available capacities to the network models each cycle.

## Unexpected findings

1. **The 3 × 3 instance uses five 3-year representative years** (see Blocked on Planner 1). This also sets its ESSO
   calendar window (5 blocks), its block weights and κ_ESSO = 386,258.69.
2. **Degradation reaches the network models through the SoC limits.** The networks' S and E are the ESSO's *available*
   capacities, refreshed every cycle (`shared_resources_planning.py:3697, 6075-6076`). Neither §2.3 nor the map states
   this coupling path. It is the mechanism behind "degradation decides".
3. **The ESSO slack-penalty loop indexes the P-net slacks with a loop variable named `y_inv` that runs over the year set**
   (`shared_energy_storage_data.py:822-829`). It sums each (y, d, t) exactly once, so it is correct — only the name is
   misleading. No action needed. Recorded so a reader of the code is not misled.
4. **The DSO interface expected-value rows have different names in the parallel builder.** The sequential DSO builder
   names them `expected_interface_*_def` (`shared_resources_planning.py:4376-4378`); the parallel builder names them
   `interface_expected_values_*` (4492-4494). The campaigns ran the sequential path (`"ParallelExecution": false` in both
   case files). This is a naming difference only, but any reader-side check by component name differs between the two
   paths.

## Not confirmed

- **Printed equation numbers.** Computed from the `.tex` source by counting numbered display rows; not checked against a
  compiled PDF (the clone has none). The map's "(21)–(28)" agrees with the computed numbers for l. 794–880.
- **Interface-transformer ratings R_int.** Values not read. Looked in `shared_resources_planning.py:5148, 5417` (call
  sites of `get_interface_branch_rating()`); the DN case JSONs were not parsed for them.
- **Scenario probabilities ω_m, ω_o of the 3 × 3 instance.** The code uses `network.prob_market_scenarios` and
  `prob_operation_scenarios`; their values were not read (`_read_market_data_from_file` / network data not traced).
- **The reference generator is the first generator in the DN case files** (Vg = 1.0 read from the first entry of each
  `case33_*_<year>.json` `generators` list). Assumed; `get_reference_gen_idx` was not traced.
- **The x0 cell's per-scenario storage schedules being numerically identical in the run outputs.** Not checked against
  the per-scenario result workbooks. Hard non-anticipativity is established from the code, which is unchanged since the
  run. No `.nl` or `.col` capture of a 3 × 3 block exists (searched `data/SRP1/Results/P515S53/row18_structural/` and
  `nl_varcount_w66/`: 0 `.col` files).
- **No model was built.** Every structural claim rests on static reading of the code at HEAD, plus the committed run
  records named above.
