# P5.15 W167 — nomenclature audit, year-dependent tables (2025/2030/2035), question (4)

Worker report, 2026-10-07. Step 6 support order (TASKS.md), `STEP6_REVISION_MAP.md` §B and §E. ZERO SOLVES:
`SolveProfileGuard(permitted=())` verified at 0; `pickle.load`/`loads` blocked, both counts 0. Integrity failures 0,
exit 0. Nothing was edited in production code, case files, specs, the frozen JSON or the manuscript clone. Line numbers
`l. N` refer to the submitted `main.tex` (clone `manuscript/6a67305f25e8348fb71380c3`, Overleaf HEAD `6191c6c`).

## Blocked on Planner

Nothing stops the task. Four points need a ruling or a confirmation. None is a decision this task could make.

1. **Map §3.1's premise does not hold.** The map says the three cost-trajectory scenarios of the submitted version
   "are not what was run". They are what was run. The cost file at `7ce1d1ab` carries three scenarios with weights
   0.35/0.55/0.10, the same as the submitted Table `tab:investment_cost`, and production weights all three (answer
   below). What changed is the energy-cost row (finding B2). The Planner and the author decide what §3.1 says.
2. **The 3 × 3 instance does not have the SRP1 horizon.** It ran on five representative years: 2025, 2028, 2031,
   2034 and 2037, each a 3-year block. Its case file is `SRP1__s53_3x3.json` (blob `dd8dd781`), and its
   `instance_record.json` carries `derive_description` and `years_as_paper: true`. SRP1 ran on 2025/2030/2035, each a
   5-year block.
   - The map §3 sentence "three representative years 2025, 2030, 2035 … the multi-scenario instance uses 3 × 3"
     therefore describes SRP1 only.
   - For the 3 × 3 instance, the printed 2025/2028/…/2037 columns are the years as run. W166 (`e7b9a938`) found the
     same thing independently.
   - How §3 presents two horizons is for the author and the expert to decide. The fragments I produced are for SRP1
     (2025/2030/2035), as ordered. The printed columns for 2028–2037 were cross-checked against the case files of those
     years, so the existing tables can serve the 3 × 3 horizon as they stand, except for the energy-cost row.
3. **Scope reading for Task B: please confirm.** I regenerated the five year-indexed **input-data** tables only. The
   eight year-indexed **results** tables are solver outputs of the submitted Benders method, so they cannot be
   produced from the case files without a solve. I enumerated and classified them but did not regenerate them. The
   map replaces or deletes all eight.
4. **The main.tex hash in the order does not match the file.** The order gives the prefix `3cedbb6e39c8…`. The file
   reads `3cedbb6e37bfe31b…`. The first 8 hex characters agree and the 9th and later differ. The file is the clone's
   HEAD blob `c3de38f5`, unmodified (`git status` is clean for main.tex), at clone HEAD `6191c6cc`. I take this to be a
   typo in the order; please confirm. The script pins the actual sha256.

## Answer to question (4)

- **The cost input is a set of probability-weighted scenarios, not a single trajectory.**
  `SRP1_ESS.xlsx` at `7ce1d1ab` is byte-identical to HEAD: blob `8a7e3744` and sha256 `e17bd588` in both. It carries
  **three cost-trajectory scenarios with probability weights**. Sheet `Scenarios` row 1 reads `numScenarios 3 | 0.35 |
  0.55 | 0.10`. The sheets `Investment Cost, Power` and `Investment Cost, Energy` hold scenarios 1–3 as rows 2–4 and
  calendar years 2020–2054 as columns.
- **Where the values come from.** The scenarios are NREL ATB low/mid/high for a 4-h system, in $/kW, converted at
  1.0819 $/€. They are split into a power part and an energy part by the NREL cost breakdown. The power part takes the
  inverter plus half of the balance of system; the energy part takes the battery cabinet plus half of the balance of
  system. The energy part is divided by 4 h.
- **2025 values.** Power: 214.17 / 267.53 / 342.17 k€/MVA. Energy: 212.13 / 264.98 / 338.91 k€/MWh.
- **The map's figures are the 2025 probability-weighted expectations.** The ≈ 256 k€/MVA is 256,317.32 €/MVA and the
  ≈ 254 k€/MWh is 253,877.68 €/MWh. They are not a column of the file.
- **How production reads the file.** `shared_energy_storage_data.py:1743–1747` reads all three scenarios and the
  weights. The weights go into `shared_ess_data.prob_market_scenarios`, a name kept for historical reasons.
- **What the planning objective uses: the probability-weighted sum Σ_m ω_m c_{m,y}.**
  - The master expression is at `shared_energy_storage_data.py:372/390–399`.
  - The oracle's I(x) is `p56a_oracle.py:253–270`, which sums over `enumerate(esso.prob_market_scenarios)` at :266.
    It is recorded at :1053 and added to the net recourse at :1204.
  - `STEP4_DFO_METHOD.md:55–57` defines the same expression.
  - Because I(x) is linear in the costs, this equals using the **expected trajectory**. The plan x carries no scenario
    index.
- **No scenario is selected anywhere.** I searched 347 committed campaign/frozen specs for a selection field and found
  none (scope below). 31 specs pin the cost file, all with sha256 `e17bd588`; none pins the predecessor `14581474`.
- **Numerical check.**
  - The unit candidate (node 7, 2025, 0.25 MVA / 1 MWh) recomputes to I = 317,957.0085 € through
    `p56a_oracle.investment_cost`. That equals the committed `w101_three_reference_summary.json` value and the 3 × 3
    `instance_record.json` value.
  - Single-scenario values would be 265,676 / 331,865 / 424,450 €.
  - T3's energy-cost line (break-even + margin) is 253,877.68 in all four entries, the expected 2025 energy cost.
- **What the submitted §3.1 says.**
  - l. 939: "Three ESS investment-cost scenarios are considered at the planning level". Text and structure match what
    ran.
  - l. 947 and the table at l. 949–966 print low/medium/high trajectories at 35/55/10 % for 2025/2028/2031/2034/2037.
  - **The difference is the energy row.** The printed energy row comes from the predecessor file (`072b1310`). That
    file divided the 4-h $/kW figure by **5**; `7ce1d1ab` divides by **4**, so every energy cost is ×1.25. The power
    row is unchanged.
  - The submitted §4.1 also reports **scenario-wise plans** (l. 1078–1107). The current method does not produce these;
    it evaluates one plan against the expected cost.

## Changed

- `4523cd3d` — `p515_s53_w167_nomenclature_years.py` (sha256 `cae920e2…`), a new file. Script commit first.
- **Outputs**, in `data/SRP1/Results/P515S53/w167_nomenclature_years/` (written once; `mkdir` refuses an existing
  directory), with sha256 prefixes:
  - `nomenclature_audit.json` (`2e55fceb`) and `.md` (`41e5312a`)
  - `year_tables/tab_*.tex`, five fragments
  - `year_tables_crosscheck.json` (`11a88e9a`)
  - `year_indexed_items.json` (`c6025eda`)
  - `question4_cost_file.json` (`dff6a2b1`)
  - `sources_blob_hashes.json` (`4d2a0671`)
  - `manifest_inputs_sha256.json` (`92c39eb9`)
  - `run_record.json` (`aa754eba`)
  - `manifest_sha256.json` (`dc9624a1`, 13 entries)
  - `manifest_post_run_sha256.json` (`61a7f005`; covers launch.log, the manifest and the typing test)
  - `launch.log`, holding stdout and stderr
- **Run.** Attached, at HEAD `4523cd3d`, wall time 13.5 s. The W100 boolean-typing test passed: 129,197 files,
  F1–F5 = 0.
- **Commit.** These outputs and this report are committed together, as `P5.15 W167: …`.
- **Case files.** Each case file the campaign reads is listed in `sources_blob_hashes.json` with its HEAD git blob,
  working-tree sha256 and last commit. Every one is clean against HEAD. The set was derived from `SRP1.json` and the
  production reading code:
  - `network.py:215`: `<net>_<year>.json`
  - `network_data.py:156`: the operational data
  - `shared_resources_planning.py:8857` (Years) and `:9104` (MarketData)
  - `shared_energy_storage_data.py:170`: SharedESS
  - `p56a_oracle.py:105`: `SharedResourcesPlanning('data/SRP1', 'SRP1.json')`

  Main entries:

  | file | blob | last commit |
  |---|---|---|
  | `SRP1.json` | `9a6d7373` | |
  | `SRP1_params.json` | `a80056d0` | |
  | `MarketData/SRP1_market_data.xlsx` | `407201e0` | |
  | `SharedESS/SRP1_ESS_Params.json` | `45034b48` | |
  | `SharedESS/SRP1_ESS.xlsx` | `8a7e3744` | `2cada62b` |
  | `case9/case9_params.json` | `975332dd` | |
  | `case9/case9_operational_data.xlsx` | `a32f1189` | |
  | `case9/case9_{2025,2030,2035}.json` | `31e83a9b` / `49c9ae84` / `a5b26789` | |
  | `case33_1/case33_1_{2025,2030,2035}.json` | `5922cb45` / `ae77ca16` / `58211b8d` | |
  | `case33_2/case33_2_{2025,2030,2035}.json` | `c24bfddc` / `d1924a42` / `bb4420b1` | |
  | `case33_3/case33_3_{2025,2030,2035}.json` | `ee92dfbc` / `eaac8137` / `fb3a2814` | |

  The params and operational files of the three ADNs are in the JSON. The 2028/2031/2034/2037 files the 3 × 3
  instance reads are also listed. The 2028 file is blob-identical to 2025, 2031 and 2034 are identical to 2030, and
  2037 is identical to 2035.

## Found

### A — Nomenclature audit (report only)

**Normalisation** (stated in full in the JSON and .md):
- Font wrappers are dropped, so `\mathrm{Inv}` = `\text{Inv}`; bold is dropped; `\textcolor` is transparent.
- Decorations stay in the identity (`hat(S)`).
- Superscripts are labels after **iteration markers** are removed: `(l)`, `(k)`, `^k`, `^{k+1}`, `\ell`, `0`, `1`.
  So `S^{Inv(l)}` = `S^{Inv}`, `π^{{E,P}^k}` = `π^{E,P}` and `x̂^1` = `x̂^ℓ`.
- Subscripts are dropped as indices unless they are a label (`Ω_C`) or digits (`y_0`).
- Letter runs are split into single letters except SoH, LB, UB, CL and similar names.
- Index letters (c d e g i j k l m n o s t y ℓ, with no script and not bold) are reported separately.

**Coverage:** 34 nomenclature entries giving 31 keys; 1,370 math atoms in the body, giving 136 keys.

- **(i) Defined but never used: 0.** Every nomenclature symbol occurs in the text.
- **(ii) Used but not defined: 89 symbol keys.** Separately, 4 hatted variants of defined symbols and 12 index
  letters (105 in total). Each comes with its first line, count and every line.
  - §2 body (17 keys):
    - Objective and recourse: `x`, `u`, `𝒳`, `𝒰`, `Q(x)`, `C^{Op}`, `C^{Inv}(x)`, `C^{Salvage}`
    - Index-years: `y^{Inv}` (89 uses), `y^{End}`, `y_0`, `T^{Rem}`
    - ESSO formulation: the slacks `S^{Up/Down}` and `E^{Up/Down}`, `S^{Rated,Unit}`, `E^{Rated,Unit}`,
      `E^{Ch,Dch}`, `Δ(t)`, `S^{S,Comp}`, `S^{Net}`
  - Algorithm 1: 25 symbols, all listed separately.
  - Appendix A (the ADMM agent): `P^E`, `Q^E`, `P̂^E`, `Q̂^E`, `P̂^I`, `Q̂^I`, `V̂^I`, `π^{E,P}`, `π^{E,Q}`,
    `π^{I,V/P/Q}`, `ρ^{E,P}`, `ρ^{E,Q}`, `r^{E,P}`, `r^{E,Q}`, `ℒ^{E,P}`, `ℒ^{E,Q}`, `K`, `X`, `f`, `h`, `N^I`, `V`,
    `V^I`, `P^I`, `Q^I`, `k^{max}`
  - Appendix D table headers: `g_i`, `b_i`, `V^{Base}`, `V^{max}`, `V^{min}`, `b^{Sh}`, `P^{G,max/min}`,
    `Q^{G,max/min}`, `V^S`
  - Results table: `E^{Av}`
- **(iii) Defined twice, overloaded or conflicting.**
  - **Duplicate nomenclature keys (3):**
    - `Y` (set, l. 197) and `Y_y` (count, l. 219)
    - `D` (set, l. 198) and `D_d` (count, l. 218)
    - `ω_c` and `ω_o` (l. 205/206): distinguished only by the index letter, which is acceptable
  - **Same description under two keys (1):** `E^{Rated}` (l. 240) and `E^{Av,Unit}` (l. 244) are both "Available
    energy capacity of unit e installed in year y^Inv and operated in year y".
  - **Declared conflicts (5),** each pinned to verified lines:
    - `S^{Rated}`: the nomenclature (l. 238) means the per-unit quantity. The body (l. 769) uses it for the **total**;
      the per-unit quantity is `S^{Rated,Unit}`, which is not in the nomenclature.
    - `E^{Rated}`: the same pattern (l. 240 against l. 773). In addition, the description is that of the available
      energy.
    - `E^{Inv(l)}_{e,y,c}` (l. 621) carries a scenario index on a decision stated to be scenario-independent.
    - `φ^{Min/Max}_e` are described "in year y" but carry no year index (l. 213–214).
    - `SoH^{min}_{e,y}` in the nomenclature (l. 217) against `SoH^{min}_{e,y^Inv}` in the body (l. 842).
  - **Declared overloads (12), each use verified:**
    - `r` has four meanings: discount rate (l. 364/560/640), ADMM loop counter in Algorithm 1 (l. 448), penalty update
      rate `r^{E,P}` (l. 1599–1607), branch resistance `r_ij` (l. 1712).
    - `x`: investment vector, and reactance `x_ij`.
    - `g`: generator index, conductance `g_i`, and cut sensitivities `g^{S,ℓ}`.
    - `π`: scenario probabilities in Algorithm 1 (l. 478), and ADMM duals in Appendix A. paragraphs_v5 also uses
      `π_t` for the wholesale price.
    - `E`: a set (l. 363, 1528), `E^S`, and the energy letter.
    - `c^{Inv}` (the budget) against `C^{Inv}` (the cost function): they differ only in case.
    - `ε^{rel}`/`ε^{abs}`: the Benders gap tolerances (l. 529/531). paragraphs_v5 uses `ε_abs`/`ε_rel` for the
      ADMM residual test.
    - `P̂`: the local copy in Appendix A. paragraphs_v5 uses `P̂` for the period.
    - `y^{(k)}`, the cut constant (l. 689/700/706), reuses the year letter.
    - `S^{Rated}_{ij}`: the branch rating in the Appendix D header.
    - `D`/`D_d` and `Y`/`Y_y`.
- **(iv) Symbols of the removed Benders method.**
  - In the nomenclature: `L^B` (l. 202; used at l. 581, 605, 616) and `α^{Down}` (l. 216; used at l. 718, 722). The
    `(l)` "planning iteration" superscripts on `S^{Inv(l)}` and `E^{Inv(l)}` (nomenclature l. 234/236) are also
    Benders iteration markers.
  - Body only:
    - `α`/`α^{(l)}`: l. 414, 416, 488, 569, 577, 689, 718
    - `μ^{S(k)}`: l. 692, 701, 709; `μ^{E(k)}`: l. 695, 701, 712
    - `y^{(k)}`: l. 689, 700, 706
    - `g^{S,ℓ}`: l. 483, 492; `g^{E,ℓ}`: l. 483, 500
    - `LB`: l. 436, 519, 525, 531; `UB`: l. 436, 510, 511, 525, 526, 531
    - `gap^ℓ`: l. 524, 529
    - `Q̂^ℓ`: l. 472, 489, 512; `x̂^ℓ`: l. 437, 440, 479, 512, 520
    - `Ŝ^{Rated,ℓ}`: l. 496; `Ê^{Rated,ℓ}`: l. 504
    - `ℓ`: l. 436, 439, 535; `ℓ^{max}`: l. 439
    - `ε^{rel}`, `ε^{abs}`: l. 529, 531
    - the `(l)` superscript on `S^{Inv}`, `E^{Inv}`, `S^{Rated}`, `E^{Rated}`, `α`: l. 563–712; every line is in the
      JSON
  - The words Benders, cut(s), underestimator, optimality gap and lower/upper bound occur on 21 lines: 142, 161, 172,
    202, 216, 299, 323, 394, 398, 420, 486, 515, 577, 605, 684, 686, 715, 722, 1131, 1136, 1140.
  - 25 symbols occur **only** inside Algorithm 1 (l. 397–540), which the map replaces wholesale. These include the
    second cost notation `c^S_{y,c}`/`c^E_{y,c}` and the sets `𝒞`, `𝒟`, `ℰ`, `𝒴`, `Ω_M`, as well as `γ_y`, `N_y`,
    `W_d` and `Φ`.
- **(v) Symbols the revision needs.** I checked 19 items; paragraphs_v5.md uses 16 of them.
  - Absent from **both** paragraphs_v5 and main.tex: the lattice unit sizes 0.25 MVA and 0.5 MWh, and the duration
    bounds 2–4 h.
    - The only symbols main.tex has for these are `φ^{Min}`/`φ^{Max}` (l. 213/214, 620–628), with the text "2 h to
      10 h" (l. 629).
    - STEP4 writes them as `P_{n,y} ∈ 0.25ℤ≥0` MVA, `E_{n,y} ∈ 0.5ℤ≥0` MWh and `2 h ≤ E/P ≤ 4 h`.
  - Used in paragraphs_v5 and absent from main.tex: `τ`, `k₀`, `k\*`, `W`, `P̂`, `P_max`, `t_sum`, `δR`, `V`,
    `ε_abs`/`ε_rel`, `ρ`, `λ`/`λ_t`, `π_t`, `c_flex`, `L` (= 44), `ε_AE`. The ones that collide with main.tex
    notation:
    - `k` is the ADMM iteration in Appendix A, which is consistent with `k₀`/`k\*`.
    - `W_d` (Algorithm 1)
    - `P̂^{E^k}`
    - `V` and `V^I` (voltage)
    - `δ` (degradation rate) against `δR`
    - `t` (time index) against `t_sum`
    - `π` (duals) against `π_t`
    - `ε^{rel}`/`ε^{abs}` (Benders) against the Boyd tolerances
    - `L^B` and `ℒ^{E,P}` against `L`

### B — Year-dependent tables and figures

- **Enumerated** (`year_indexed_items.json`, line ranges):
  - **Input-data tables (5), regenerated:**
    - `tab:investment_cost` (l. 949–966)
    - `tab:cs3_tn_res_generators` (l. 978–995)
    - `tab:cs3_adn_node_5_res_generators` (l. 1002–1017)
    - `tab:cs3_adn_node_7_res_generators` (l. 1019–1034)
    - `tab:cs3_adn_node_9_res_generators` (l. 1036–1051)
  - **Results tables (8), not regenerated (Blocked 3):**
    - `tab:cs3_shared_ess_investment_plan` (l. 1078–1098)
    - `tab:cs3_shared_ess_specs` (l. 1110–1122)
    - `tab:cs3_results_operational_planning_summary` (l. 1153–1190)
    - the four discretization tables (l. 1296–1367)
    - `tab:cs3_results_summary_years_days` (l. 1979–2084)
  - **Appendices B–D (l. 1612–1972)** contain **no** year-indexed table. Their four tables (33-bus buses, branches, TN
    connection, loads) do not depend on the year. Their year-indexed content is figures only.
  - **Text lines** with years: 20. Of these, l. 275, 278 and 280 are false hits on citation keys (`..._2025`).
    Notably:
    - l. 915: "five years (2025, 2028, 2031, 2034, and 2037)"
    - l. 144: the abstract's "2037"
    - l. 1100, 1192, 1198, 1212, 1227, 1294, 1335: results text
    - l. 976, 1000, 1615, 1616, 1642, 1814, 1868, 1922: "2025 base year"
- **Fragments** (`year_tables/<label with : → _>.tex`): each keeps the same rows, units, caption and label, with
  columns 2025/2030/2035 only. They were read through production:
  - RES capacity: `network._read_network_from_json_file`, then `pmax × baseMVA`, which equals the raw JSON `Pmax` for
    every generator and year.
  - Costs: `shared_energy_storage_data._read_shared_energy_storage_data_from_file`.

  The cost fragment carries the expected 2025/2030/2035 values as LaTeX comments only:
  - power: 256.32 / 210.20 / 194.56 k€/MVA
  - energy: 253.88 / 208.20 / 192.71 k€/MWh

  New 2025/2030/2035 values:

  | table | values |
  |---|---|
  | TN | wind 20/30/50, 40/60/60, 25/40/40; PV 12/20/20, 20/30/50, 15/25/25 MW |
  | ADN 5 | 40/50/50, 15/25/25, 20/20/30, 8/15/35 |
  | ADN 7 | 40/50/50, 40/40/40, 10/20/20, 10/10/30 |
  | ADN 9 | 20/50/50, 30/30/30, 30/30/50, 30/30/40 |
  | power cost (scn 1/2/3) | 214.17/168.72/153.93; 267.53/224.25/206.96; 342.17/278.10/268.58 k€/MVA |
  | energy cost (scn 1/2/3) | 212.13/167.11/152.46; 264.98/222.12/204.99; 338.91/275.45/266.02 k€/MWh |

- **Cross-check of printed against produced** (`year_tables_crosscheck.json`, 126 cells): **15 disagree, all in the
  energy-cost row.**
  - Every RES cell agrees, including node ID and type, for 2025 and for every printed year 2028–2037. The power-cost
    cells (15) and the probabilities (6) agree.
  - **Printed 2025 energy cost against the file, scenarios 1 / 2 / 3: 169.71 / 211.99 / 271.13 against 212.13 / 264.98
    / 338.91 k€/MWh.**
  - Printed against file for the other years:

    | year | scn 1 | scn 2 | scn 3 |
    |---|---|---|---|
    | 2028 | 148.10 vs 185.12 | 191.41 vs 239.26 | 240.67 vs 300.83 |
    | 2031 | 131.35 vs 164.18 | 174.95 vs 218.69 | 218.85 vs 273.56 |
    | 2034 | 124.31 vs 155.39 | 166.73 vs 208.41 | 214.33 vs 267.91 |
    | 2037 | 117.28 vs 146.60 | 158.51 vs 198.14 | 209.80 vs 262.25 |

  - The cause is the cost correction: the energy formula `…/5…` at `072b1310` became `…/4…` at `7ce1d1ab`, a factor of
    1.25. I read this from both blobs. The printed values equal the predecessor file to 2 decimal places.
  - The data changed after submission; the table does not mislabel anything.
- **Figures.**
  - **Input-data figures (11), 2025 only:**
    - `fig:cs3_market_prices` (l. 1618–1637)
    - TN RES (l. 1644–1663)
    - for each ADN: RES, load and flexibility, at l. 1816–1971. The flexibility files are
      `case33_x_flexibility_scenarios_2025_Spring.pdf`.
  - **Results figures (4):**
    - `voltage_profile_node_3_2025_Winter.pdf` (l. 1200–1205)
    - the three `results_*_multi_year.pdf` (l. 1215–1253)

    These are outputs of the submitted method; the map deletes or replaces them, so no 2030/2035 version applies.
  - Whether 2030/2035 input figures are needed is for the author to decide. The facts, located in the code, are:
    - Production plots the **first representative year only**: `shared_resources_planning.py:287`,
      `network_data.py:147/170`.
    - Each (year, day) block **draws its own realisation**. The draw comes from a year-independent synthetic pool
      (`network_data.py:165`), with a seed that includes the year (`network_data.py:190`;
      `shared_resources_planning.py:9137`). It is then scaled by that year's capacity, load growth or price growth
      (`network.py:1459/1424`, `shared_resources_planning.py:9132`).
    - A 2030 profile is therefore a different draw, not a rescaled 2025 profile. The appendix sentence at l. 1642,
      "used as baseline trajectories in all representative years, scaled according to …", describes the scaling but
      not the per-year draw.

## Not confirmed

- **What the submitted figure PDFs were made from.** I did not open the PDFs, so I have not checked whether they were
  produced from the submitted 5 × 5 instance. A band plotted from SRP1's single operation scenario would collapse to a
  curve; that is inferred from `network_data.py:2780`, not checked against the files.
- **The code path for the 3 × 3 years.** The 3 × 3 horizon comes from its committed case file and instance record,
  plus W166's independent finding. I did not open a 3 × 3 evaluation record to read the years from the run itself.
- **Every reader of the costs.** Question (4)'s code evidence covers the master expression, the oracle's I(x), the
  salvage basis and STEP4. A repository-wide audit of every consumer of `prob_market_scenarios` was done earlier
  (`P5_15_S45_PROBABILITY_AUDIT_NOTE.md`, W3) and was not repeated.
- **Scope of the "no selection field" claim.** It covers the 347 git-tracked `campaign_spec*`/`frozen_*` JSON files
  under `data/SRP1/Results` at HEAD. The patterns are listed in `question4_cost_file.json`. The only hits are three
  explanatory sentences about `prob_market_scenarios`. Launcher scripts were not searched for overrides; the I(x)
  check against the committed records is the evidence that no override was applied for the unit candidate.
- **The nomenclature audit is mechanical.** Math inside `\text{}` words with punctuation, and symbols in plain text
  outside math (such as "DSO$_i$"), are not tokenised. The meanings in (iii) are my reading, pinned to verified lines;
  they are not a semantic proof.
