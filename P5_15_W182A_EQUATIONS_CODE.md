# P5.15 W182a — the four new equations and `app:salvage_credit` at Overleaf `edd7a63`, against the code (Addendum 82 v2, item A1)

**Worker, 2026-10-25. Zero solves, no model build, no pickle load, no production import.**
`SolveProfileGuard(permitted=())` was armed before any project import and verified at exactly 0 (0 permitted, 0 blocked).
`pickle.load` and `pickle.loads` were blocked: 0 calls. No production module was in `sys.modules`.
- **Manuscript.** `main.tex` was read with `git -C manuscript/6a67305f25e8348fb71380c3 show edd7a63:main.tex`. The clone
  was read-only: `rev-parse`, `show`, `status` and `merge-base` only.
- **Code.** Production source was read as text at the code head and parsed with `ast`; it was never executed.
- **Records.** Evaluation records were read with `json`, and `SRP1_ESS.xlsx` with openpyxl in read-only mode.
- **What was not edited.** The manuscript, `manuscript_review/`, production code, the frozen files, the specs, earlier
  W-outputs, `TASKS.md`, `REVISION_CONTEXT.md` and the brief.

Harness `p515_s53_w182a_equations_code.py`. It was run attached, with both streams captured in
`data/SRP1/Results/P515S53/w182a_equations_code/launch.log`, and exited with code 0.

---

## Blocked on Planner

None. Three items to note; none blocks A1.

1. **Commit count.** There are **67** commits from `8f051a2` to `edd7a63` (`git rev-list --count`, 0 merges), not 66
   (the task) and not "eight" (Addendum 82 v2, "Manuscript state"). `8f051a2` is an ancestor of `edd7a63`. The clone
   HEAD is `edd7a63a6c2e…`. The `main.tex` sha256 is `b08a3365607e56f2…`, which is the critical review's fingerprint.
2. **The predicted M row did not appear.** The code's deviation row has the printed sign: P − P̄ = d⁺ − d⁻, with d⁺ the
   excess over the commitment. What the manuscript lacks is a statement of **P^I's direction**: in the code it is
   TN → DN positive, and it is net of the co-located shared ESS. This is rated **L**. Neither a reversed direction nor
   the ESS term can change a number. Whether to add the definition is a wording decision for the author.
3. **The salvage `SoH_{e,y^Inv,y^End}` convention** (L, the critical review's §3.3) has the same status. The code uses
   the end-of-block SoH of the final representative block, and `main.tex` gives no number for y^End (2040 in the code).

## Changed

- **New harness:** `p515_s53_w182a_equations_code.py` (sha256 `e23e559c…6dd19`).
- **New outputs:** `data/SRP1/Results/P515S53/w182a_equations_code/`:
  - `w182a_equations_code.json` `493d4b03…f9367`;
  - `manifest_sha256.json` `6a1717bb…e2c4f` (130 inputs, 0 uncommitted);
  - `launch.log` `39a31b3e…78467`;
  - `manifest_post_run_sha256.json`.
- **This report.** The harness, outputs and report are committed together by explicit filename (outputs with
  `git add -f`). The commit hash is in the Worker's hand-off. Nothing was pushed.
- **Code head audited:** `a17f6ba7`. The last production or rule commit is `575bfec4`, which the harness verified.
- **Repo HEAD moved during the task.** Parallel W182 Workers committed, so HEAD was `7614bcbf` at the run. The harness
  required no production, rule, source or case-file difference between `a17f6ba7` and HEAD: none. It also required
  each source file in the working tree to equal `a17f6ba7`: all equal.
- **No production code, configuration or spec was changed.**

## Found

**Prediction (expert): "A1 at most one M row (the interface-deviation sign convention)."**
- **Outcome:** 0 M rows and 0 H rows.
- **Ratings:** 3 MATCH and 2 L.
- **"At most one M": held.**
- **The named M (the deviation sign convention): not found.** The printed and code rows have the same signed canonical
  form {P: +1, P̄: −1, d⁺: −1, d⁻: +1}. The only finding about P^I is that its direction is not stated in the manuscript
  (L).

Ratings were **computed** by the harness:
- **MISMATCH** (a refusal, leaving the M or H judgement to this report) if any check of a line anchor, a symbol
  correspondence or a record failed;
- otherwise **L** if a scoped wording finding was confirmed;
- otherwise **MATCH**.

All 9 negative controls were refused:
- NC1: altered code line number;
- NC2: sign swap in the code row;
- NC3: domain `pe.Reals`;
- NC4: initial SoH 0.95;
- NC5: τ denominator 2;
- NC6: δ_R 0.08;
- NC7: d⁺ and d⁻ swapped in the printed equation;
- NC8: salvage discounted over y^End − y^Inv;
- NC9: salvage product without its remaining-life factor.

### Row table (W166 form; code lines at `a17f6ba7`)

| ID | item (`edd7a63` lines) | printed | code (file:line) | symbol-by-symbol | rating | changed or always? |
|---|---|---|---|---|---|---|
| A1-1 | `eq:acceptance_tolerance` l. 1992–1995 (label 1994); values l. 2919 | τ = δ_R 𝒱 / 4; "same monetary units as Q(x)" (l. 1999); δ_R = 0.07, 𝒱 = 259,375.33 €, τ = 4,539.07 € (l. 2919) | `settling_criterion.py:57` `TAU = DELTA_R * R_REF / 4.0`; `:55` `DELTA_R = 0.07`; `:56` `R_REF = 259375.33` | δ_R ↔ `DELTA_R`; 𝒱 ↔ `R_REF`; 4 ↔ 4.0. Computed τ = 4,539.068275 €, which rounds to 4,539.07 (see below the table) | **MATCH** | Always, within the certification period (see below the table) |
| A1-2 | `eq:interface_deviation` l. 1115–1124 (label 1123) | P^I_{i,s,t} − P̄^I_{i,t} = d⁺_{i,s,t} − d⁻_{i,s,t}; Q "analogous" (l. 1153); "No deviation premium is imposed on the TSO" (l. 1153); omitted at one scenario (l. 1158) | `model_construction_helpers.py:1866–1867` `m.pg_adn[s_m, s_o, p] - m.expected_interface_pf_p[p] == m.row18_dev_p_up[…] - m.row18_dev_p_down[…]`; Q `:1873–1874`; rows `:1971–1972` | P^I ↔ `pg_adn`; P̄^I ↔ `expected_interface_pf_p`; d± ↔ `row18_dev_p_up/down`; s ↔ (s_m, s_o); t ↔ p; i ↔ the DSO block. Further mapping below the table | **L** | Always, since `1d310565` (see below the table) |
| A1-3 | `eq:interface_deviation_nonnegative` l. 1131–1134 (label 1133) | d^±_{i,s,t} ≥ 0 | `model_construction_helpers.py:1966–1969`: four Vars with `domain=pe.NonNegativeReals`, no `bounds=` | d⁺ and d⁻, P and Q, all `NonNegativeReals`, with no other bound. At the DSO initialisation solve each pair is fixed at 0 with its row deactivated (consistent with ≥ 0) | **MATCH** | Always: in force since `1d310565` (2026-09-22), unchanged. At 8f051a2 it was printed inside `eq:commitment_deviation` (l. 1075) |
| A1-4 | `eq:soh_initial` l. 1060–1063 (label 1062); prose l. 1036, l. 1056 | SoH_{e,y^Inv,(y^Inv)⁻} = 1 | `shared_energy_storage_data.py:687` `for y in range(y_inv, max_tcal_norm):`; `:714` `prev_soh = 1.00`; `:715–716`; `:723–724` | The code has no variable for the pre-installation boundary: the literal 1 is substituted. The chain row is `soh[y_inv, y] == prev_soh·exp(−D)·φ^{num_years}`, with `prev_soh = soh[y_inv, y−1]` only when y > y_inv (structure parsed by ast). Records below the table | **MATCH** | Always: the literal 1.00 dates from `fc9b1f6e` (2024-05-07), before the revision (see below the table) |
| A1-5 | `app:salvage_credit` l. 2570–2597 (label 2571): `eq:salvage_credit` l. 2577–2584, `eq:salvage_remaining_life` l. 2588–2591; prose l. 2596–2597, l. 1363, l. 2567 | C^Salvage = Σ_e Σ_{y^Inv} c̄ E^Inv/(1+r)^{y^End−y_0} · T^Rem/T^Cal · (SoH_{y^End} − SoH^Min)/(1 − SoH^Min); T^Rem = max{0, y^Inv + T^Cal − y^End} | `shared_energy_storage_data.py:899–905` product: `terminal_discount · energy_recovery_fraction · expected_unit_cost · remaining_life_fraction · residual_energy` | Factor-by-factor mapping and the numerical reproduction are below the table | **L** | Always: introduced `9a433546` (2026-08-05) (see below the table) |

**A1-1 — `eq:acceptance_tolerance`.**
- **Values.** The computed τ = 4,539.068275 € rounds to the printed 4,539.07. `R_REF` is `V_SRP1` = 259,375.32654… rounded to
  the cent (`p515_s53_w101_srp1_continuation_campaign.py:119`).
- **Rule chain.** v2:69 → v4:62 → v5:52 → v6:57. The v6 hooks import `settling_criterion_v6`
  (`p515_s53_w142_resettle_v6_hooks.py:30`).
- **Every printed use has its code line:**
  - C3, range ≤ τ: v2:172;
  - M2: v2:189;
  - M3, |last step|·2P_max ≤ τ: v2:182;
  - C4, |t_sum| ≤ τ/2: v2:74 and :277;
  - sign floor τ/100: `settling_criterion.py:58`;
  - swing floor τ/10: v6:80;
  - determinacy max{3b, 2τ}: v6:363.
- **Not in the model.** τ is **not in Q or the model**: no τ token occurs in any of the 10 production modules.
- **History.** The line was introduced in `ab0bcbc9` (2026-09-27, W101) and never changed. At every one of the 45
  certification run heads the rule files are byte-identical to the code head. At 8f051a2 the formula was printed
  inline (l. 739); the equation is a display of it.
- **The search code never contained τ** (see the next section).

**A1-2 — `eq:interface_deviation`.**
- **Further mapping.**
  - P^I = `pg[ref_gen] − Σ shared_es_pnet[e, s_m0, s_o0]` (`:1297`, `:1293`; `network.py:444`): the DN's import, net
    of the co-located ESS, whose net power is `pch − pdch` (`:999`).
  - P̄^I = Σ prob_m prob_o `pg_adn` (`:2416–2424`, `:2443`).
- **Sign.** The canonical forms are identical: printed = code.
- **Direction: TN → DN positive on both copies.**
  - DSO copy: the injection of the DN reference generator.
  - TSO copy: the ADN load `pc` + `interface_delta` at the TN node (`:1247`, `:1259`).
- **Wiring.** The rows are built only in DSO models (`:1948`), only when there is more than one scenario (`:1917`).
  The TSO has `premium_alpha=0.0` (`shared_resources_planning.py:4298`).
- **L (wording):**
  1. The direction of P^I is stated nowhere in `main.tex`.
  2. `main.tex` does not say that P^I excludes the co-located shared ESS.
  3. The code is in per unit on `baseMVA` (`:1979`); the print carries no unit.
- **Not H, for two reasons.**
  - The charge sums d⁺ + d⁻ (`:1979–1981`), so reversing the direction only swaps d⁺ and d⁻.
  - The ESS term is the scenario-free copy, so it cancels in P − P̄.
- **History.** The row was introduced in `1d310565` (2026-09-22, Addendum 38 W39) and never changed. The net-of-ESS form
  of the DSO interface dates from `f46cb382` (2026-08-18).
- **Run head.** The 3 × 3 run head `8a8ef150` carries production files identical to the code head.
- **At 8f051a2** the same equation was printed inside `eq:commitment_deviation` (l. 1075). The edit split a display
  and did not change the content.

**A1-4 — `eq:soh_initial`.**
- **Records.** Over the 49 committed evaluation records, `soh_prev_end` is exactly 1.0 at all **54** installation
  cells, and equals the previous block's `soh_end` at all **81** later cells.
- **History.** The literal 1.00 was introduced in `fc9b1f6e` (2024-05-07). The chain's form changed in P5.15-1; this
  literal did not. Later commits changing its occurrence count:
  - `4a54c7b3` and `9603fd72` (2024-08-22: the decomposed functions were added, then removed);
  - `e5721b6d` (2026-09-21: a second literal at l. 622, for the 'mid' variant).
- **At 8f051a2** the printed form was SoH_{e,y^Inv,y^Inv−1} = 1 (l. 1029). The new boundary notation (y^Inv)⁻ is
  closer to the code, because the code has a boundary state and no block y^Inv − 1.

**A1-5 — `app:salvage_credit`.**
- **Factor-by-factor mapping.**
  - **Discount.** 1/(1+r)^{y^End−y_0} ↔ `terminal_discount` = 1/(1 + 0.02)^{Σ block widths} (`:911–912`), with
    Σ widths = y^End − y_0 = 15 for contiguous blocks. r = `DiscountFactor` 0.02 = the printed 2 % (l. 1207).
  - **Expected cost.** c̄ ↔ `expected_unit_cost` = Σ_c ω_c c^{Inv,E}_{y^Inv,c} (`:928`). The ω_c are the "Scenarios"
    sheet of `SRP1_ESS.xlsx` (0.35, 0.55, 0.10), read at `:1743`; the costs are undiscounted.
  - **Remaining life.** T^Rem/T^Cal ↔ `remaining_life_fraction` = max(t_cal − Σ widths from y^Inv, 0)/t_cal
    (`:919–921`). The code's min(·, 1) is inactive in every evaluated case.
  - **Energy and SoH.** E^Inv (SoH − SoH^Min)/(1 − SoH^Min) ↔ `residual_energy` (`:890–893`) at
    `recycling_floor_fraction` = 0, with `e_available` = E_rated·SoH_end ('end' point, `:17`).
  - **Recovery fraction.** `energy_recovery_fraction` = 1.0 is absent from the print. `SRP1_ESS_Params.json` sets
    `salvage_value` {enabled: true, 1.0, 0.0}, so both extra code factors are identities.
  - **Converter.** The converter is not credited: the code has an energy term only.
- **Numerical reproduction.** The printed formula, recomputed from each record's terminal ageing cells and the case
  inputs, reproduces the recorded `terminal_salvage_value` of all **49** committed evaluation records.
  - **17** of them have a non-zero credit: installations in 2030 and 2035, from 9,854.56 € to 114,448.81 €.
  - The worst absolute difference is **9.5 × 10⁻¹¹ €**, on the record and on every node.
- **"Reported, not optimized" holds:**
  - The ESSO objective is `feasibility_penalty` only (`:851–854`); the salvage is a separate Expression (`:849`).
  - No network module references salvage.
  - The net form is gross − salvage (`shared_resources_planning.py:1306`).
  - All three searches exclude salvage from F, by the `OBJECTIVE_CONVENTION` at each search's run head.
- **L (wording).** SoH_{e,y^Inv,y^End} indexes SoH by a date, whereas SoH is indexed by representative-year blocks
  elsewhere (`eq:soh_chain`, l. 1046).
  - The code uses `terminal_year_idx = len(years) − 1` (`:871`), the end-of-block SoH of the final block. Under
    `eq:soh_chain`'s end-of-block convention that is the state at y^End.
  - `main.tex` states neither the mapping nor y^End = 2040: no line contains "2040". The y^End lines are 251, 315,
    2580, 2582, 2589 and 2596.
- **History.** The salvage code was introduced in `9a433546` (2026-08-05). It was then edited by `40846517`
  (2026-08-28, a reformatted call) and `e5721b6d` (2026-09-21, the 'mid' branch only; 'end' unchanged). The net line
  dates from `fc200780` (2026-09-15).
- **What the edit changed.** Both equations were in §4's introduction at 8f051a2 (l. 1343–1365). They were moved
  verbatim into `app:salvage_credit`; only the trailing commas were dropped (diff checked). The label is new.

**Every one of the five describes something the code already did.** No implementing line changed after `575bfec4`,
the last production or rule commit. At all **46** run heads the production and case files are identical to the code
head: the 45 certification heads of `w142_resettle_v6`, `w142_resettle_ext_v6` and `w155_a64_cells`, plus the 3 × 3
head `8a8ef150`. The rule files are identical at all 45 certification heads. Four of the five items were already in
`main.tex` at `8f051a2` in another form:
- inline τ, l. 739;
- the combined `eq:commitment_deviation`, l. 1075;
- the inline SoH_{y^Inv−1} = 1, l. 1029;
- the salvage equations in §4, l. 1345–1358.

The edits since then are presentational.

### The search code (for `eq:acceptance_tolerance`)

| search (App. E.21) | campaign results | git head at run (date) | `settling_criterion*.py` at the head | τ tokens in the search code at the head | improvement threshold |
|---|---|---|---|---|---|
| baseline, variant A (s47) | `P515S47/campaign_s47_phase_b` | `353e094b` (2026-09-21) | absent | 0 (`p515_s47_phase_b_record.py`, `p515_s44_campaign_harness.py`) | max(bar_x + bar_inc, σ_Q) |
| 2×, variant A (s51) | `P515S51/campaign_s51_f2_phase_b` | `c39c4836` (2026-09-23) | absent | 0 (s47, s51, s44) | same |
| 2×, variant B (s53) | `P515S53/campaign_s53_f2_certificate_r1` | `e1f98be8` (2026-09-24) | absent | 0 (s47, s51, s53, s44) | same |

τ was introduced in `ab0bcbc9` (2026-09-27), which is an ancestor of none of the three heads.
- **The searches never used τ.** They ran under the production exit and used the σ_Q resolution (18,449.66 €).
- **`main.tex` says so.** l. 2915: "The three planning searches used earlier versions of the implementation"; l. 2920
  gives σ_Q.
- **τ governs only the later certification campaigns**, through the stop decision of the v6 hooks.

## Not confirmed

- **Equation numbers.** They were not computed. Items are located by source line and `\label`, and no PDF was compiled.
- **Hours with π̄_t ≤ 0.** No check was made for such an hour in the 3 × 3 instance. If one existed, d⁺ and d⁻ could
  both be positive at the optimum. The 3 × 3 spec records `floor: null`, and `expected_market_price` was not
  recomputed from the market workbook. The printed d^± ≥ 0 matches the code either way.
- **The P^I direction scan has a scope.**
  - **Searched:** the commitment subsection (l. 1092–1163); the nomenclature line 336; every line carrying
    `P^{\text{I}}` (336, 1105, 1116, 1832). The words were: positive, import, export, withdraw, injection, inject,
    direction, sign convention, from the TN, from TN, TN-to-DN, TN--DN direction, into the DN, flowing.
  - **Found:** none.
  - **Whole document, not deciding:** only l. 1628 and l. 2847 ("reverse" flow of the static arrangements) carry such
    a word next to "interface" or "DN". They imply a conventional direction but do not define P^I's sign.
  - **By hand:** a separate grep for "as a load", "equivalent load", "as a generator", "reference bus" and "substation"
    found only l. 640, which is unrelated.
- **The ESS-exclusion scan** used the same windows with: exclud, net of, excluding, without the storage, storage
  excluded. Found: none.
- **The salvage reproduction** covers the 49 committed `evaluation_record.json` under `w142_resettle_v6`,
  `w142_resettle_ext_v6`, `w155_a64_cells` and `w90_3x3`. It does not cover:
  - the reference rows that appear only in the frozen tables (e.g. `ref:5ca4f86c`, salvage 39,043.21);
  - the `year_ladder` salvage values.

  Those carry no committed ageing cells in these roots.
- **The search runs' own salvage handling** was checked by text: the `OBJECTIVE_CONVENTION` string at each run head
  says salvage is excluded from F. How F was computed was not re-executed.
- **The audit is static.** Nothing was solved or built, so whether the built models carry exactly these rows rests on
  the source reading plus the record reproductions above.
