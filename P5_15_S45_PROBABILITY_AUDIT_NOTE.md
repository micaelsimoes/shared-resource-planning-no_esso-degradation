# P5.15 Addendum 27 item 5b (W3) — probability audit (zero solves)

Script: `p515_s45_probability_audit.py` (sha256 `8e8c29a0…`). Evidence: `data/SRP1/Results/P515S45/probability_audit/`
(`probability_audit.json` sha256 `007769db…`, `probability_audit_launch.log`, `production_stdout_capture.log`,
`manifest_sha256.json` sha256 `de9cae1e…`, 107 files hash-recorded). HEAD at run: `25470496`.
`SolveProfileGuard(permitted=())` was armed for the whole run and `verify(0)` passed exactly: 0 solves, 0 launches,
0 blocked. There were exactly 3 DSO, 1 TSO and 1 ESSO-initialisation `.optimize` intercepts, matching the declared counts.
The run was concurrent with the AA C* re-verification campaign, as the task authorised. The campaign lock was only read.
`git status --porcelain` shows one change during the run: the new output directory appeared. Every other file written
outside that directory during the run belongs to the running campaign (`P515S45/reverify_aa_c_star*`,
`P56A/evals/p515s44_s45_reverify_…`).

## Verdict

**No defect. No operational term and no part of `gross_operational_cost` is weighted by the ESS workbook's
investment-cost scenario probabilities. This holds at SRP1 and at paper scale.**

- **The overwrite.** `shared_energy_storage_data.py:1644` rebinds `shared_ess_data.prob_market_scenarios` to the
  workbook sheet `Scenarios`, which holds `[0.35, 0.55, 0.10]` in both the current and the previous cost file.
  - This rebinds only the ESS data object's own attribute.
  - The networks' vectors are separate list objects. They are identical to `planning_problem.prob_market_scenarios[year]`
    on all 48 SRP1 blocks and all 80 paper-scale blocks.
  - The previous value on the ESS object came from `shared_resources_planning.py:8012`. That assignment is dead, because
    the reader rebinds the attribute four lines later.
- **Every consumer of the workbook vector is on the investment side.** Each pairs it with `cost_investment[..][s]`, the
  workbook's own scenarios:
  - Benders master scenario set, budget row and objective (`shared_energy_storage_data.py:301/354/372`) — class c.
  - The investment cost I(x) in `p56a_oracle.py:227` — class c.
  - The first-stage budget check (`shared_resources_planning.py:1873`) — class c.
  - The bootstrap candidate (`shared_resources_planning.py:1914`) — class c.
  - Master-result reporting (`shared_energy_storage_data.py:2150`) — class c. The Excel writer (`:2287`) — class d.
  - The terminal-salvage unit cost E[c_E] (`shared_energy_storage_data.py:865`) — class b, **net only**. It enters
    `net_operational_recourse = gross − salvage`, never `gross_operational_cost`. It is not an objective term: the
    ESSO objective is `feasibility_penalty` plus the AL terms. Pairing investment-cost probabilities with
    investment-cost scenarios is what the salvage `cost_basis: EXPECTED_INSTALLATION_ENERGY_COST` specifies.
- **Every operational weight reads `network.prob_market_scenarios × network.prob_operation_scenarios`.** This covers:
  - the TSO/DSO objective aggregates, which are classes a and b;
  - the interface settlement;
  - the expected-value constraints, including the expected shared-ESS P/Q that the ESSO is coordinated on;
  - the scenario-deviation penalties.
  The ESSO model has no scenario index at all.

## SRP1 (built at C*, zero solves)

The overwrite **is visible** at SRP1, because the two vectors differ: networks use `[1.0] × [1.0]`, the workbook uses
`[0.35, 0.55, 0.10]`. It is harmless because no operational site reads the workbook vector.

| Built quantity | Values found |
|---|---|
| Weights on the 7 `total_*` aggregates, pg cost coefficient ÷ (c_p·baseMVA), expected-value-constraint weights (48 blocks) | all \|w\| = 1.0 |
| ESSO base-objective coefficients | {1e-5, 1e3} = `EPS_ESSO_THROUGHPUT`, `PENALTY_ESSO_SLACK` |
| ESSO ADMM-objective variables | es_pch/pdch_per_unit, es_pnet/qnet, slacks |
| ESSO ADMM-objective parameters | p_req/q_req, duals, rho |
| Q(x) block weight | N_y × N_d × annualisation (e.g. 460 = 5·92·1), no probability |
| Salvage E[c_E] (workbook-weighted) | 253,877.68 / 208,198.72 / 192,708.12 €/MWh for 2025 / 2030 / 2035 |
| C* cohort-2025 remaining-life fraction | 0.0, so the salvage coefficient is 0 and the workbook vector adds 0 to Q at C* |
| I(x) at C* (class c, not Q) | 3,696,250.22 € |

## Paper scale (data read only; model construction blocked, 0 blocked calls)

The run read the committed derived case `SRP1__paper.json`. Its scenario checksum `1e8bdd3e…` matches the value the scale
measurement recorded.

- **Networks.** All four networks have obj_type COST. On all 80 blocks the market and operation probabilities are both
  `[0.2]×5`, so each of the 25 combinations gets weight 0.04.
- **Workbook.** The workbook vector is still `[0.35, 0.55, 0.10]`. No operational site reads it.
- **Deviation penalty.** The scenario-deviation penalty is skipped at 1×1 but becomes active at 25 combinations. It uses
  the networks' probabilities through `_scenario_probability`.
- **What misuse would have meant (counterfactual only).** Market-scenario weights would have been off by up to
  |0.35 − 0.2| = 0.35 on s_m = 0..2, and s_m = 3, 4 would have raised IndexError. The code does neither.
- **Salvage.** Remaining-life fractions for the 2025 / 2028 / 2031 / 2034 / 2037 cohorts are 0 / 0.2 / 0.4 / 0.6 / 0.8.
  So the workbook-weighted salvage (net only, correct by design) is 0 for 2025 investments. It is non-zero only for
  later cohorts.

## Search scope (negative claims are scoped to this)

- **Files searched:** all 305 `*.py` files in the repository at all depths, excluding `.git`.
- **Primary patterns:** `prob_market_scenarios`, `prob_operation_scenarios`, `cost_investment`.
- **Broad patterns:** `prob_`, `market_scenarios`, `probability`, `omega`. Per-file counts are in the JSON, with full
  lines for the oracle-path modules.
- **Classification.** Every primary hit is classified by its enclosing function (read from the AST). This covers 125 read
  sites in 55 modules: 25 production modules together with the oracle-path import closure of `p56a_oracle.py`,
  `p515_g_g1_g4_admm_gates.py` and `p515_s44_campaign_harness.py`.
- **Completeness is enforced.** The script fails if any hit has no rule, or if any rule matches no hit.
- **Not searched:** non-Python files (the JSON/xlsx data were read, not grepped), other branches, stage reports.

## Latent hazards (not defects; no change made)

1. **Misleading names.**
   - `shared_ess_data.prob_market_scenarios` actually holds investment-cost scenario probabilities.
   - The Benders master's set of those scenarios is called `model.scenarios_market`.
   - The reader that loads them is called `_get_operational_scenarios_info_from_excel_file`.

   A later generalisation could easily confuse this vector with the networks' market-scenario vector. A minimal
   safeguard, described here but not implemented, would be two steps:
   - rename the attribute, for example to `prob_investment_cost_scenarios`, at its 11 production and oracle sites
     (8 in `shared_energy_storage_data.py`, `shared_resources_planning.py:1873/1914`, `p56a_oracle.py:227`) plus the
     diagnostic readers in `p514_d_preflight.py` and `p515_s44_addendum26_confirmations.py`;
   - delete the dead assignment at `shared_resources_planning.py:8012`.
2. **Dead reader.** `network.py:1069 _read_network_operational_data_from_file` would set `prob_operation_scenarios` from
   the `Main` sheet, but nothing calls it. Operation probabilities are always uniform (`network_data.py:191`).
