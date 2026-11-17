# P5.15 W182g — cost boundary (C1), horizon end (C2), budget admissibility (C3)

Addendum 82 v2 of `PLANNER_BRIEF_2026-09-13.md`, W182 order part C, items C1–C3 (brief §2). Zero solves: code, case
files, the cost workbook, committed records and `main.tex` are read as text. Overleaf clone
`manuscript/6a67305f25e8348fb71380c3` at **`edd7a63`** (main.tex sha256 `b08a3365…`, 2,956 lines; the working copy equals
the HEAD blob). Code head: repo HEAD `a21b03e6` at the run; `a17f6ba7` is an ancestor. The newer commits are
parallel W182 harnesses and reports. No commit after `575bfec4` touches the eight production files read here. The
last commit that touches any of them is `a51ad9ba`.

**Objective convention:** Q = `gross_operational_cost` = Σ TSO/DSO `get_primal_value` − contracted interface
settlement − interface-voltage pin. net = Q − terminal salvage credit, and F = I + Q. "Settlement excluded" is exact
only at one market scenario (see C1).

## Blocked on Planner

None. The order is complete. Addendum 82 records no prediction for C1–C3. Points for the Planner's W182 report that
need an expert or author wording ruling (no run depends on them):

1. **C1, l. 724: "load curtailment" is listed as a component of Q, but it is inactive in every evaluation.**
   `l_curt` is false in all four network parameter files, so the curtailment variables are never built
   (network.py:352). All three records read here show `load_curtailment_cost` 0.0. Options:
   - (a) drop it from the case-study description;
   - (b) keep it and say it is disabled in the case study.
   Recommendation: (b), one clause.
2. **C1, l. 2567, Appendix E: "Throughout, Q is the gross operational cost, settlement excluded."** The code excludes
   only the *contracted* part. At 3 × 3 the deviation settlement (−0.75 M€) and the row-18 premium (+25.08 M€, about
   3.0 % of Q) are inside Q. The sentence is exact for every single-scenario Appendix E entry. The preceding sentence
   points to the multi-scenario table. The same reading applies to the TASKS/brief convention line. This agrees with
   audit §3.8. Recommendation: "contracted settlement excluded", or restrict the sentence to the single-scenario
   record.
3. **C2: the manuscript gives no numeric horizon end.** There is no "2040" or "2039" anywhere in main.tex. Eq. (salvage)
   indexes SoH by y^End, while SoH is block-indexed elsewhere. The code's mapping is unambiguous (C2 below).
   Recommendation: state y^End = y_0 + Σ_y Y_y = 2040 (1 January 2040, the end of 2039) in both configurations, and
   SoH_{e,y^Inv,y^End} ≡ the end-of-block SoH of the last representative block (2035, or 2037 at 3 × 3).
4. **C3: four admissible plans are within 0.77 % of the budget.** One of them is the plan returned at 2× (row 40).
   Five named plans are over budget: the 2025 ladder probes already listed by W182d A8. No plan selected or certified
   as a comparator is over budget.

## Changed

- `a21b03e6` — harness `p515_s53_w182g_cost_horizon_budget.py`, committed before the real run.
- This commit adds:
  - this report;
  - `data/SRP1/Results/P515S53/w182g_cost_horizon_budget/w182g_cost_horizon_budget.json` (sha256 `f73d8b19…3970`);
  - `manifest_sha256.json` (sha256 `2311efd9…f6c1`), holding 62 inputs with sha256 and status, of which 1 is
    uncommitted;
  - `launch.log` (sha256 `93b5feaa…63f5`).
- The script sha256 recorded in the manifest (`537538d5…`) equals the committed `a21b03e6` blob.
- Command, run from the repo root, attached, stdout and stderr both captured:
  `mkdir data/SRP1/Results/P515S53/w182g_cost_horizon_budget && …/opf_env_py311/bin/python -u p515_s53_w182g_cost_horizon_budget.py > …/launch.log 2>&1`.
  Exit 0.
- Guards:
  - SolveProfileGuard(permitted=()) verified at 0, with all counts 0;
  - pickle load/loads 0;
  - no repository module imported other than `gate_result_io` and `p513_solve_profile_guard`;
  - JSON written through `gate_result_io`.
- Checks: 111 code anchors found, each located by exact line text inside its named function (AST span). All 49
  manuscript quotes were found. Every integrity check holds (`failed_checks` = {}). All 6 negative controls refused.
- Nothing was edited in production code, case or cost files, specs, the clone, `manuscript_review/`, TASKS.md,
  REVISION_CONTEXT.md or the brief. `TASKS.md` was already modified in the working tree before this task; I did not
  touch it.

## Found

### C1 — cost boundary, term by term

Q is aggregated in `shared_resources_planning.py` `_get_operational_recourse_components`:
- :1237 and :1239 sum TSO + DSO `get_primal_value`;
- :1302–1304 compute gross = that sum − `interface_settlement_contracted_total` − `voltage_pin_total`;
- :1305 reads the salvage;
- :1306 sets net = gross − salvage;
- :3301 calls it every cycle.

`get_primal_value` evaluates `objective_function_rule`, not `admm_objective` (network.py:70). It is weighted by
Y_y·D_d·(1+r)^−(y−y0) (network_data.py:100, :102).

The AST pass over `objective_function_rule` (model_construction_helpers.py:1811) found exactly these statements, so
the term list below is complete:
- l. 1813 and 1815 (congestion-management branch, not used: obj_type "COST" in all four networks);
- l. 1816, 1817, 1818, 1823, 1836 and 1842.

All file:line references below are at code head `a21b03e6`. mch = model_construction_helpers.py,
srp = shared_resources_planning.py, essd = shared_energy_storage_data.py.

| Term | Built at | Weight in the ADMM path | Status |
|---|---|---|---|
| Generation cost: controllable generators at the scenario price; DSO reference generator excluded | mch:1813, :2005, :2007 | unchanged | **inside Q**. In the case data only the TN CONV units are controllable; DN = REF + RES |
| Flexibility cost: flex_p_down + flex_q_down at cost_flex | mch:1813, :2047. TSO ADN-interface loads skipped at :2044 | unchanged | **inside Q**. TN flexible loads sit only at buses 5/7/9, the interfaces, so this is DN flexibility only |
| Load-curtailment cost | mch:1813, :2088–2092 | COST_CONSUMPTION_CURTAILMENT = 300 (srp:5044, :5334) | **inside Q, identically 0**: `l_curt` false in case9/case33_1–3 params; variables not built (network.py:352) |
| RES-curtailment penalty | mch:2118 | **0** (srp:5042, :5332) | inside Q at weight 0, so it contributes 0 |
| Storage usage penalty | mch:1816, :2218 | shared-ESS weight **0** (srp:5041, :5326) | inside Q at weight 0. The local-ESS part is 0 in the records (no local ESS) |
| Network-feasibility slacks: V² slack, node balance, branch flow, flexibility P day-balance | mch:1817, :2247, :2258, :2271, :2297 | unchanged (definitions.py constants) | **inside Q**. Active families: voltage (case flag) and flexibility day-balance (pinned True in code, network_parameters.py:76, regardless of the case file) |
| Storage daily-closure slack: shared ESS 1e3 × baseMVA, i.e. c^Cl = 10³ €/MWh; local ESS | mch:1818, :2341, :2329 | unchanged | **inside Q** |
| Interface settlement, **contracted** part (expected exchange × mean price) | mch:1823, :1702, :1723. Weight Param :1701 = 0 | **1** (srp:5050, :5340) | **outside Q**: in every local objective, subtracted at srp:1303 |
| Interface settlement, **deviation** part (covariance, i.e. the imbalance settlement) | mch:1724 | 1 | **inside Q**: not subtracted. Identically 0 at one market scenario |
| Row-18 commitment-deviation premium α·π̄_t·baseMVA·(d⁺+d⁻), P and Q | mch:1836, :1979, :1982. DSO only (:1948). Not constructed at one scenario (:1917) | α = 0.5 (3 × 3 campaign). Activated with the settlement (srp:5343) | **inside Q** |
| Interface-voltage dispersion pin (9e4 × E_s(v−v̄)²) | mch:1842, :1940, :1941 | solver-only | **outside Q**: subtracted at srp:1304 |
| ADMM augmented-Lagrangian terms: duals and ρ/2 on V, P, Q and ESS P, Q; TSO proximal terms; previous-iterate ESS terms; base objective ÷ (σ/w_b) | TSO srp:5142, :5153, :5154, :5168, :5185, :5203–5204. DSO srp:5407, :5432, :5452 | — | **ADMM local objective only** |
| ESSO objective: PENALTY_ESSO_SLACK × pnet slacks + 1e-5 × throughput, plus its AL terms | essd:829, :844, :848, :853. srp:5502, :5520, :5526 | — | **ADMM local objective only**: the ESSO model is never summed into Q |
| Terminal salvage credit | essd:849 (an Expression, not in the ESSO objective :853), :148 | — | **outside Q**: net = Q − salvage (srp:1305–1306) |
| Investment cost I(x) | essd:376, :400 (master). srp:2257 (closed-form check) | — | **outside Q**: F = I + Q at planning level |

Retired terms are retained but have no caller in their own file:
- `adn_interface_flexibility_cost` (mch:2053) and `ess_complementarity_bilinear_value` (mch:2345), both reporting
  helpers only;
- `_add_tso/dso_scenario_deviation_penalty` (srp:4078, :4127), unwired.

**Record evidence** (committed, read-only):
- In all three records, gross = including-settlement − contracted − pin holds to ≤ 1e-12 relative, and
  net = gross − salvage.
- The weighted decomposition generation + flexibility + slacks + deviation + row 18 closes to ≤ 6e-7 €.

| Record (eval key) | Q (gross) | deviation settlement | row-18 premium | pin (excluded) | salvage |
|---|---|---|---|---|---|
| SRP1 yl_y2035 (`3a2387f4…`, cycle 167) | 653,786,670.24 | 0.00 | — (not built) | 0.00 | 56,504.17 |
| 3 × 3 x = 0 (`f6e9cd53…`, cycle 72) | 842,832,534.76 | −748,745.77 | 25,082,401.21 | 188.05 | 0 |
| 3 × 3 n7 0.25/1 (`c82522f4…`, cycle 69) | 842,595,839.51 | −746,992.08 | 25,081,114.02 | 188.14 | 0 |

Manuscript at `edd7a63` against the code:

| Line | Quoted text | Rating |
|---|---|---|
| 724 | "includes TN generation costs, activated DN flexibility, load curtailment, and penalties on storage energy-closure and network-feasibility slacks. In the multi-scenario instance, it also includes the commitment charge and the deviation component of interface settlement" | MATCH except "load curtailment": **L** (inactive, decision 1) |
| 728 | "contracted component … excluded as a transfer … Storage-agent and solver regularisation terms are also excluded. Renewable curtailment is assigned no cost" | MATCH |
| 732 | "Operating costs are discounted to y_0 using the representative year y" | MATCH (network_data.py:100–102) |
| 1153 | "No deviation premium is imposed on the TSO" | MATCH (mch:1948) |
| 1157–1158 | premium additional to scenario-price settlement; "deviation-charge term is omitted" with a single scenario | MATCH (mch:1917) |
| 2096 | contracted part removed, deviation part kept; voltage regularisation solver-side and removed; at one scenario not constructed | MATCH |
| 2567 | "Throughout, Q is the gross operational cost, settlement excluded" | **L** (decision 2) |
| 1350, 1619 | benchmark Q "excluding interface settlement" | MATCH for single scenario; benchmark code not read (Not confirmed) |
| 1828 | block weight w_{y,d} = Y_y D_d (1+r)^−(y−y0), objective ÷ σ | MATCH (srp:5142, effective_scale = σ/w) |
| 552, 1090, 1363, 1785, 2597 | F = I + Q; no monetary cycling cost; salvage after evaluation, not optimised | MATCH |
| 1300 | closure slack "priced at c^Cl = 10³ €/MWh" | MATCH (mch:2341, PENALTY_SHARED_ESS_BALANCE 1e3) |

### C2 — horizon end, terminal SoH and the final block

**Answer.** In both configurations the terminal date is y^End = y0 + Σ_y Y_y = 2025 + 15 = **2040**, that is
1 January 2040, the end of calendar year 2039.
- Code: shared_energy_storage_data.py:978–979. Case-file key `"Years"`:
  - data/SRP1/SRP1.json: {2025: 5, 2030: 5, 2035: 5};
  - data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json: {2025: 3, 2028: 3, 2031: 3, 2034: 3, 2037: 3}.
- Blocks are contiguous and of equal width in both files.
- The final block is **2035, covering 2035–2039** (SRP1), and **2037, covering 2037–2039** (3 × 3). Its end boundary
  is 2040.
- Terminal discount: (1 + `DiscountFactor`)^−15 = 1.02^−15 = 0.743014729989 (:911–912).
- Terminal SoH: `es_soh_per_unit_cumul[y_inv, N−1]` (:871, :883–890; reported at :1007). This is the **end-of-block**
  SoH of the last representative block. The recursion is SoH_y = SoH_{y−1}·exp(−D_y)·φ^n (:724), with
  D = 365·n·EFC/day / k (:703).
  - n is read from the *investment* block's width (:678). This equals the block's own width in both configurations,
    so it matches eq. (cycling_loss).
  - Under the mid-block variant the salvage still uses the end-of-block SoH (:889).
- Remaining life: T_rem = max(T_cal − (y^End − y^Inv), 0) (:919–921). This is identical to manuscript l. 2589 with
  y^End = 2040. The T_rem/T_cal fractions are:

  | Configuration | Fractions by installation year |
  |---|---|
  | SRP1 | 2025: 0, 2030: 1/3, 2035: 2/3 |
  | 3 × 3 | 2025: 0, 2028: 0.2, 2031: 0.4, 2034: 0.6, 2037: 0.8 |

- The credit is C = 1.02^−15 · c̄E(y^Inv) · E · (T_rem/T_cal) · (SoH_term − 0.70)/0.30. The cost basis is the
  expected undiscounted energy cost of the installation year (:895). It is an Expression, never in an objective.

**Reproduced from committed records** (zero solve; the sidecar and the per-cycle record end at the same cycle, which
is the certification cycle):

| Cell | Terminal SoH (end of last block) | SoH chain from recorded EFC/day | Salvage recomputed | Salvage recorded |
|---|---|---|---|---|
| SRP1 7: 0.25/1 @2030 (cycle 169) | 0.7684143079 (2035 block) | max rel. diff 4.3e-14 | 11,759.257933 | 11,759.257933 (diff 0) |
| SRP1 7: 0.25/1 @2035 (cycle 167) | 0.8775806078 | 7.4e-13 | 56,504.166416 | 56,504.166416 (diff 0) |
| 3 × 3 7: 0.25/1 @2025 (cycle 69) | 0.6999999900 (floor binding in the 2037 block, dual 2,921.8) | 1.6e-16 | 0 (T_rem = 0) | 0.0 |

The uncommitted 3 × 3 results workbook (`…/c82522f470b35b58_n7_4h_e1/results/SRP1_distributed_terminal.xlsx`, sha256
`e45d8da0…`, sheet "Terminal Salvage") records:
- terminal valuation date 2040 for every cohort;
- terminal representative year 2037;
- discount 0.7430147299885189;
- expected installation energy costs equal to the cost-file expectation for 2025/2028/2031/2034/2037.

Manuscript comparison:
- No numeric terminal date appears: 0 lines contain "2040" or "2039".
- l. 251 defines y^End as "end of the planning horizon". l. 2580 uses (1+r)^(y^End−y0). l. 2582 indexes SoH by y^End.
  l. 2596 defines y^End as the end of the planning horizon.
- l. 790 says "Available energy is evaluated using end-of-block state of health". l. 1036 sets the block-boundary
  convention.
- l. 1196–1198 give the two configurations (2025/2030/2035 × 5; 2025–2037 × 3), and l. 146 the "15-year horizon".

The code is consistent with these formulas once y^End := 2040 and SoH_{·,y^End} := the end-of-block SoH of the last
block. The manuscript does not state either identification (decision 3; agrees with audit §3.3).

### C3 — budget admissibility

**Budget.** B = 1,000,000 €, key `"budget"` in data/SRP1/SharedESS/SRP1_ESS_Params.json.
- It is read at shared_energy_storage_parameters.py:253 and enforced at shared_resources_planning.py:2266 (tolerance
  1e-8, :2238) and in the master row at shared_energy_storage_data.py:378.
- Both case files name the same params and cost files.
- Manuscript l. 1207 gives "1 M€". **Table 1 (l. 1224–1238) holds unit costs and probabilities only, not the budget.**

**Formula** (preserved in the harness as `investment_cost`; the same expression as srp:2254–2258 and essd:376):
I(x) = Σ_n (1.02)^−(y−2025) · Σ_c ω_c (c^S_{c,y} S_n + c^E_{c,y} E_n), with ω = (0.35, 0.55, 0.10) from the
'Scenarios' sheet.
- In `_read_planning_problem`, the ESS probability vector is first set from the market data (srp:9003). The cost
  file's sheet then overwrites it (read call at srp:9009).
- Verdict: **ADMISSIBLE** iff I ≤ B + 1e-8. **BOUNDARY** (declared here) iff admissible and B − I < 1 % of B.

**Cross-checks:**
- Table 1: all 18 printed cells equal the workbook to 0.01 k€, and the 3 probabilities match.
- The 2025 expectations are 256,317.32 €/MVA and 253,877.68 €/MWh. l. 1244 (256.32 / 253.88) MATCH, and
  l. 1424–1425 (256,320 / 253,880) MATCH.
- The 2025/2030/2035 unit costs equal the committed W2 table (production master) to 1e-6.
- I(x) equals W2 `I_new_eur` and the frozen Step-6 `I_other` wherever recorded, to 1e-6 €.
- Table-1-rounded costs give the same verdict for every plan.
- The printed 318.0 k€ (l. 1374, 1636, 1653, 1799) MATCH.

**Audit §3.6 "two-node 2030 plan ≈ 992.3 k€": CONFIRMED.**
- I = 992,268.11 €, slack 7,731.89 € (0.77 % of B). No further ΔE = 0.5 MWh step fits in 2030.
- The audit's 4 h ladder (318.0 / 635.9 / 953.9 / 1,271.8 / 1,589.8 k€) is confirmed.
- The audit's unit costs "256.318 k€/MVA, 253.8755 k€/MWh" are slightly off: the file gives 256.317 and 253.878, a
  difference of +0.68 €/MVA and −2.18 €/MWh. This is immaterial at 0.1 k€.

**Counts named in the text, enumerated in the search's l∞ box (W173 definition):**

| Text claim | Line | Enumerated | Detail |
|---|---|---|---|
| Twelve admissible single-node plans in 2025 | 2797 | 12 | of 35 rule-feasible; 23 over budget |
| Fourteen admissible neighbours of x = 0 | 2797 | 14 | 0 rejected by the budget |
| 61 admissible unit neighbours of the returned plan | 1507 | 61 | 2025: 1, 2030: 18, 2035: 42. 106 box points rejected by the budget, the cheapest at 1,017,950 € |

**All 47 distinct plans named in main.tex.** Sources:
- Table E.21 (61 rows);
- the claims table (58 rows, mapped through the Eval. column);
- the gap table;
- the neighbours table;
- Tables value_ladder, energy_ladder_m2 and year_ladder;
- 12 prose anchors;
- the six search-only neighbours of l. 2808.

| Plan (node: MVA/MWh @year) | I (€) | B − I (€) | Verdict | Where (Table E.21 rows; other) |
|---|---|---|---|---|
| 5: 0.25/0.5 @2025 | 191,018.17 | 808,981.83 | ADMISSIBLE | 15, 16; value ladder; neighbours; l. 1374 |
| 7: 0.25/0.5 @2025 | 191,018.17 | 808,981.83 | ADMISSIBLE | 17; neighbours; l. 2797 |
| 9: 0.25/0.5 @2025 | 191,018.17 | 808,981.83 | ADMISSIBLE | 18; neighbours; l. 2797 |
| 5: 0.25/0.5; 9: 0.25/0.5 @2025 | 382,036.34 | 617,963.66 | ADMISSIBLE | 22; value ladder; neighbours |
| 5: 0.25/0.5; 7: 0.25/0.5 @2025 | 382,036.34 | 617,963.66 | ADMISSIBLE | l. 2808 (search only) |
| 7: 0.25/0.5; 9: 0.25/0.5 @2025 | 382,036.34 | 617,963.66 | ADMISSIBLE | l. 2808 |
| 5, 7, 9: 0.25/0.5 each @2025 | 573,054.51 | 426,945.49 | ADMISSIBLE | l. 2808 |
| 5: 0.25/1 @2025 | 317,957.01 | 682,042.99 | ADMISSIBLE | 12 |
| 9: 0.25/1 @2025 | 317,957.01 | 682,042.99 | ADMISSIBLE | 13 |
| **7: 0.25/1 @2025 (reference)** | 317,957.01 | 682,042.99 | ADMISSIBLE | 2, 31, 33, 35, 54–61; tables value/energy/year ladder; 3 × 3; l. 146, 1374, 1515, 1645, 1711, 1799 |
| 7: 0.5/1 @2025 | 382,036.34 | 617,963.66 | ADMISSIBLE | 7; value ladder |
| 7: 0.5/2 @2025 | 635,914.02 | 364,085.98 | ADMISSIBLE | 3, 36; value ladder; energy ladder |
| 7: 1/2 @2025 | 764,072.68 | 235,927.32 | ADMISSIBLE | 8 |
| 7: 0.75/3 @2025 | 953,871.03 | 46,128.97 | ADMISSIBLE | 4, 37; energy ladder |
| 9: 0.75/3 @2025 | 953,871.03 | 46,128.97 | ADMISSIBLE | 14 |
| **7: 1/4 @2025** | 1,271,828.03 | −271,828.03 | **INADMISSIBLE** | 5, 38; energy ladder |
| **7: 1.25/5 @2025** | 1,589,785.04 | −589,785.04 | **INADMISSIBLE** | 6, 39; energy ladder; gap table |
| **7: 1.5/3 @2025** | 1,146,109.02 | −146,109.02 | **INADMISSIBLE** | 9 |
| **7: 2/4 @2025** | 1,528,145.36 | −528,145.36 | **INADMISSIBLE** | 10 |
| **7: 2.5/5 @2025** | 1,910,181.70 | −910,181.70 | **INADMISSIBLE** | 11 |
| 5: 0.25/0.5 @2030 | 141,882.02 | 858,117.98 | ADMISSIBLE | 19; neighbours |
| 7: 0.25/0.5 @2030 | 141,882.02 | 858,117.98 | ADMISSIBLE | 20; neighbours |
| 9: 0.25/0.5 @2030 | 141,882.02 | 858,117.98 | ADMISSIBLE | 21; neighbours |
| 5: 0.25/0.5; 7: 0.25/0.5 @2030 | 283,764.04 | 716,235.96 | ADMISSIBLE | 23; value ladder; neighbours; l. 1414 |
| 5: 0.25/0.5; 9: 0.25/0.5 @2030 | 283,764.04 | 716,235.96 | ADMISSIBLE | l. 2808 |
| 7: 0.25/0.5; 9: 0.25/0.5 @2030 | 283,764.04 | 716,235.96 | ADMISSIBLE | l. 2808 |
| 5, 7, 9: 0.25/0.5 each @2030 | 425,646.07 | 574,353.93 | ADMISSIBLE | l. 2808 |
| 7: 0.25/1 @2030 | 236,168.02 | 763,831.98 | ADMISSIBLE | 24; year ladder |
| 7: 0.5/1 @2030 | 283,764.04 | 716,235.96 | ADMISSIBLE | 26; l. 1677 |
| 7: 0.5/2 @2030 | 472,336.04 | 527,663.96 | ADMISSIBLE | 28; l. 1677 |
| 7: 0.75/3 @2030 | 708,504.06 | 291,495.94 | ADMISSIBLE | 47 |
| 7: 1/3 @2030 | 756,100.09 | 243,899.91 | ADMISSIBLE | 46 |
| 7: 1/3.5 @2030 | 850,386.09 | 149,613.91 | ADMISSIBLE | 48 |
| 7: 1/4 @2030 | 944,672.09 | 55,327.91 | ADMISSIBLE | 49 |
| 5: 0.25/0.5; 7: 0.75/2.5 @2030 | 756,100.09 | 243,899.91 | ADMISSIBLE | 53 |
| 5: 0.25/0.5; 7: 0.75/3 @2030 | 850,386.09 | 149,613.91 | ADMISSIBLE | 43 |
| 7: 0.75/3; 9: 0.25/0.5 @2030 | 850,386.09 | 149,613.91 | ADMISSIBLE | 52 |
| 5: 0.25/0.5; 7: 1/3 @2030 | 897,982.11 | 102,017.89 | ADMISSIBLE | 42; gap table |
| 7: 1/3; 9: 0.25/0.5 @2030 | 897,982.11 | 102,017.89 | ADMISSIBLE | 50; gap table |
| 5: 0.25/0.5; 7: 1.25/3 @2030 | 945,578.13 | 54,421.87 | ADMISSIBLE | 44 |
| **5: 0.25/0.5; 7: 1/3.5 @2030 (returned at 2×)** | 992,268.11 | 7,731.89 | **ADMISSIBLE (BOUNDARY)** | 40; gap table; l. 1507 |
| 5: 0.25/1; 7: 1/3 @2030 | 992,268.11 | 7,731.89 | **ADMISSIBLE (BOUNDARY)** | 41; gap table |
| 5: 0.25/0.5; 7: 0.75/3; 9: 0.25/0.5 @2030 | 992,268.11 | 7,731.89 | **ADMISSIBLE (BOUNDARY)** | 45; gap table |
| 7: 1/3.5; 9: 0.25/0.5 @2030 | 992,268.11 | 7,731.89 | **ADMISSIBLE (BOUNDARY)** | 51; gap table |
| 7: 0.25/1 @2035 | 197,989.52 | 802,010.48 | ADMISSIBLE | 25; year ladder |
| 7: 0.5/1 @2035 | 237,891.25 | 762,108.75 | ADMISSIBLE | 27; l. 1677 |
| 7: 0.5/2 @2035 | 395,979.03 | 604,020.97 | ADMISSIBLE | 29; l. 1677 |

Totals: 42 admissible (4 at the boundary) and 5 inadmissible. Every plan satisfies the ratio rule
(2 ≤ E/S ≤ 4) and E ≤ 5 MWh. The five inadmissible plans are the 2025 node-7 ladder probes used in the break-even fit
(l. 1424 "all ten evaluations") and the 2× energy ladder (Table energy_ladder_m2, 4 and 5 MWh). This agrees with
W182d A8 (7 evaluations, 5 plans). The four boundary plans have identical totals of converter rating and energy in
2030, hence identical I.

## Not confirmed

- **Benchmark cost boundary.** l. 1350, 1582, 1619 and 2841 (passive arrangement's 1 €/MWh RES-curtailment term
  excluded from Q): `uncoordinated_benchmark.py` was not read. Only the single-scenario identity
  (deviation settlement ≡ 0) supports these lines here.
- **3 × 3 cost-file path resolution.** The derived instance names `"SRP1_ESS_Params.json"` / `"SRP1_ESS.xlsx"`. I did
  not trace how the 3 × 3 launcher resolves these names to a directory. The uncommitted 3 × 3 workbook's expected
  energy costs for 2025–2037 equal the committed workbook's expectation, which indicates the same file. The 3 × 3
  plans named in the manuscript are 2025 plans only.
- **Local-ESS terms.** `es_reg` is true in all four parameter files. I did not verify from the network JSONs that no
  local ESS exists; the records show 0 for `ess_usage_cost` and `local_ess_day_balance_slack`.
- **Plan search scope.** Plan strings were collected from main.tex only. The letter, highlights and cover letter were
  not scanned (W182d covers A8 there). The claims table's plan text was not parsed: its plans are taken through its
  Eval. column, and every pair resolves in Table E.21.
- **BOUNDARY band.** The 1 %-of-B band is my declared definition, not a project rule. The slack and the "one ΔE step
  fits" flag are in the JSON for any other cut.

## Unexpected findings (not acted on)

- **Negative slack penalties.** The detector slack-penalty totals are negative in all three records (−12.7 to
  −13.0 k€). REVISION_CONTEXT l. 410 already records this as IPOPT bound-relaxation residue.
- **3 × 3 SoH floor.** At 3 × 3 the reference unit's SoH floor (0.70) binds in the terminal 2037 block. Manuscript
  l. 1573 ("first binds in 2035") concerns the single-scenario configuration. No 3 × 3 floor statement was checked.
- **Zero-capacity cohorts in the workbook.** The 3 × 3 workbook marks zero-capacity cohorts "Active in Terminal
  Representative Block: True". This is cosmetic: their salvage is 0.
- **Investment-block width in the D row.** The D row reads the investment block's width for every later block
  (essd:678, documented in the code). This is harmless while blocks have equal width, as in both case files.
