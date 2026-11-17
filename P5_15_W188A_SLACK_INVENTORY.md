# P5.15 W188 (a): slack inventory (G10), Addendum 89, prediction P5

**Worker, 2026-11-10.** Zero solves, zero model builds, zero pickle loads. I only read records, production source, case
files, the frozen tables and the Overleaf clone. I did not edit code, specs, records, the clone, `manuscript_review/`,
TASKS.md, REVISION_CONTEXT.md or the brief.

- **Manuscript:** `git show a87e195:main.tex` in `manuscript/6a67305f25e8348fb71380c3`. Blob `8b8577c6…`, md5
  `39f6fd86f90b0fbe7c3cc335b1bb793c`, which is the md5 Addendum 89 declares.
- **Frozen tables:** `frozen_step6_tables_v1_590088fe.json`. I checked its sha256 against `590088fe6b36…`.
- **Repository head at the run:** `687c1a53`, the commit that added the harness.

**Objective convention.** Q = `gross_operational_cost`, contracted settlement excluded (the frozen primary).

- The "€ content in Q" of a slack family is Σ_b w_b·V_b. V_b is the family's probability-weighted penalty value in
  block b. w_b = Y_y·D_d·(1+r)^−(y−y0) is the `get_primal_value` weight.
- The content is **signed** and is what Q actually contains. It is negative where IPOPT leaves a slack inside its
  relaxed lower bound (−1e-8).
- The **positive content** is Σ a_j·max(x_j, 0). That is the part a slack above zero would add to Q (the penalty it
  pays).

**Instance identity.** Every row below carries its eval key and candidate. The full keys are in
`summary_rows[*]` of the JSON.

---

## Blocked on Planner

1. **Table numbering.** I mapped the tables by content, not by number. The mapping is asserted in the harness: 34 of 34
   printed rows match the frozen JSON. I used the expert's numbering, because it is the only one under which "Table
   11's coordinated row" exists:

   | Table | Label |
   |---|---|
   | 7 | `tab:res_value_ladder` |
   | 9 | `tab:res_energy_ladder_m2` |
   | 10 | `tab:res_ageing` |
   | 11 | `tab:res_benchmark` |
   | 12 | `tab:res_multiscenario` |
   | 13 | `tab:res_year_ladder` |
   | 14 | `tab:res_discount` |

   - **A static count gives different numbers.** The three nomenclature longtables cancel themselves with
     `\addtocounter{table}{-1}`. On that count `main.tex` has 11 main-text tables: `res_value_ladder` would be 4 and
     `res_discount` 11.
   - **Earlier reports disagree with each other:**
     - W182c (`edd7a63`): Table 11 = `tab:res_discount`.
     - W184 (`a87e195`): "Table 12" = `tab:res_benchmark`.
   - I could not settle this: there is no TeX installation here and no compiled PDF in the clone.
   - **No P5 result depends on it.** Every frozen T2 / appended cell was inventoried, either in the named set or in the
     "beside" set. That includes the evaluations behind `tab:res_breakeven` (Table 8 in the expert's count), which the
     order does not name.
2. **Wording of the G10 sentence (expert).** P5 holds in its "at the zero bound" reading: the positive content is
   exactly 0 € on all 20 certified rows. Q nevertheless contains a signed residue from slacks that sit inside IPOPT's
   relaxed bound:
   - **Size.** −13,012.57 € to −13,052.36 € on the certified rows (−2.0 × 10⁻⁵ of Q). On the 3 × 3 rows it is
     −12,700.04 € and −12,728.07 €.
   - **It is common to every evaluation.** Across the 20 certified rows it varies by 39.80 €. All of that variation
     comes from the closure slacks: 6 to 20 € per active unit, depending on the installation year.
   - **So "Q is the economic cost alone" holds only up to a common offset of about −13.0 k€.** The offset cancels in
     every reported difference to within 40 €. Whether the sentence needs that qualifier is the expert's call.
3. **One Table 12 row is not established (uncertified, so outside P5's scope).** At `t11_3x3_unit` (production exit,
   cycle 69):
   - The flexibility P day-balance slacks are **not shown to be ≤ 0** from the records. Their pair bound is
     +4.98 × 10⁻⁸ p.u., at DSO 7, 2034 Autumn, scenario 0_1, load 32.
   - Their positive content is bounded by **2.16 €**. That is above 1 €, and it is not shown to be positive.
   - The records and pickles in the eval directory cannot resolve this. The workbook stores only the net up − down,
     and there is no `certified_models.pkl` (only `esso_models_s39_D.pkl`).
   - The Planner decides whether this needs a statement. For scale: the table's resolution is 18.0 k€.
4. **A superseded certificate in Appendix E has a positive voltage slack.** This is new; W179c and the l. 2934 text
   cover only its closure. The cell is `pb_y2025_n5` (E.21 row 16, "$^{s}$", not behind any named table):
   - In block `TSO|2035|Spring` the V² slack sum is **S = +6.456 × 10⁻⁶ p.u.²**, so at least one element is > 0. The
     largest element is ≤ 8.85 × 10⁻⁶ p.u.².
   - That block's signed voltage content is **+121.80 €**, against −45.27 € for a block at the floor.
   - The row's total positive content is bounded by **170.69 €**: 169.27 € voltage, 1.42 € closure (the W179c
     +1.76 × 10⁻⁶ MWh), 0 € flexibility.
   - P5 does not cover it, because it is not behind a named table. It is listed because the Appendix E slack sentence
     (l. 2934) names only this certificate's closure.

---

## Changed

- **`687c1a53`** — `p515_s55_w188a_slack_inventory.py` (sha256 `79fe6c34…`). Committed before the run.
- **The report commit** (its hash is in the Worker message to the Planner) — this report and the outputs:
  - `data/SRP1/Results/P515S55/w188a_slack_inventory/w188a_slack_inventory.json`: sha256
    `fdcb1ba18b88fceaf4c8ee85f44820e3888910a46450b8e46adfd1c37b033b21`.
  - `capture_plan.json` (`1920d046…`): written **before** extraction.
  - `manifest_sha256.json` (`c95316f5…`): 438 inputs with sha256 and tracked status, 194 of them uncommitted.
  - `launch.log`: the run was attached, both streams captured, exit 0.
- **Guards:**
  - `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0. All four
    counters are 0.
  - `pickle.load` and `pickle.loads` were blocked and called 0 times.
  - No production module was imported (asserted).
  - All JSON was written through `gate_result_io`.
- **Trial runs:** five trial runs went to scratchpad directories under the session scratchpad. They are not preserved.
  They served only to debug the harness. Trial 1 refused on the first version of the bound-option check. That check
  was then corrected to compare the stated values instead of the option names.

---

## Which penalised slacks enter Q (code at HEAD, anchors located by the harness)

| Family | Penalty in the block objective | Status in SRP1 | In Q |
|---|---|---|---|
| Voltage V² slacks `slack_v_sqr_up/down` | 5e4 · (s⁺ + s⁻) (mch:2247) | active (case flag). 2·24·5 Vars per TSO block: TSO interface buses fixed at srp:4263, REF bus [0, 0] at mch:94. 2·24·32 per DSO block | yes |
| Flexibility P day-balance `slack_flex_p_balance_up/down` | baseMVA · 1e3 · (s⁺ + s⁻) (mch:2297) | active (pinned True, network_parameters.py:80). 2·32 Vars per DSO block. TSO interface loads skipped (mch:2290) and fixed (network.py:350) | yes |
| Shared-ESS closure s± `slack_shared_es_soc_final_up/down` | baseMVA · 1e3 · (s⁺ + s⁻) (mch:2341) | active. 2 Vars per active unit, scenario-free. Fixed at 0 for inactive units | yes |
| Node balance, branch flow | mch:2258, :2271 | flags false in all four case files: no Vars; identically 0 in every record | yes (0) |
| Local-ESS closure | mch:2329 | no `energy_storages` in any case-year file read; identically 0 | yes (0) |
| RES curtailment, ESS usage, load curtailment | weight 0 or not built (W182g C1) | 0.0 in every record | yes (0) |
| **ESSO σ± = `slack_es_pnet_up/down`** | 1e3 · (σ⁺ + σ⁻) in the ESSO `feasibility_penalty` (essd:833, :852) | active (`slacks: true`) | **no** |

How Q is assembled:

- Q sums only the TSO and DSO `get_primal_value` (srp:1237, :1239, :1302).
- That value is `objective_function_rule` (network.py:70), which adds `total_slack_penalties` and
  `total_ess_complementarity_penalties` (mch:1817, :1818).
- The ESSO model is never summed into Q. So σ± has **0 €** in Q by construction.
- In the records, "σ" is never this pair. The pair is recorded as `esso_feasibility_violation_D3`, which equals
  `get_feasibility_violation`, the sum of σ⁺ + σ⁻ in MW (essd:141).

---

## Found

### P5 against its outcome

| Prediction (expert, Addendum 89 P5) | Outcome |
|---|---|
| Every penalised slack is at its zero bound on every certified row (€ content < 1 €). Any non-zero content is listed with its row. | **HELD: 20 of 20** certified rows behind the named tables. Every element of every penalised family in Q is shown to lie in [−1e-8, −7.1e-9]. The positive (penalty-paid) content is therefore **exactly 0 €** on each row. Of the 5 uncertified rows, 4 are proven the same way. `t11_3x3_unit` is not established (Blocked 3). The signed content is not < 1 €: it is about −13.0 k€, the relaxation residue (Blocked 2). The ESSO σ± pair sits at its relaxed lower bound in all 25 rows: every element ≤ −9.7e-9 MW. |

**How "at the zero bound" was established.** The proof uses records only.

- **The premise.** Every slack element satisfies x_j ≥ −f, with f = 1e-8. This is IPOPT's `bound_relax_factor`
  default on lb = 0, with `honor_original_bounds no`.
  - No case or ESS options file sets either option (scanned).
  - Three records state the options in force (the 3 × 3 cells), and each states exactly these values (asserted).
- **Single-scenario rows.** Each block records the slack sum S of a family. With N non-fixed Vars in the block, every
  element satisfies x_j ≤ S + (N − 1)·f.
  - N is taken from the code rules and the case files. It is confirmed independently for the 3 × 3 cells by the
    workbook's exact counts of non-zero entries: 43,200 TSO and 829,440 DSO voltage elements.
  - The recorded sums are within 0.28 f per block of −N·f. For example, the DSO voltage sum is −1,535.7207 f against
    N = 1,536.
  - That puts every element at ≤ −7.2 × 10⁻⁹ for voltage, ≤ −9.94 × 10⁻⁹ for flexibility and ≤ −9.996 × 10⁻⁹ for
    closure.
- **`d_c52e1670` at k* = 150.** This point is not in `component_levels_terminal.json` (that file is at cycle 198). It
  comes from the committed per-cycle `q_decomposition`, where voltage and flexibility are aggregated together.
  - I used a mixed bound: x_j ≤ (V + f·Σa)/a_min − f. It gives ≤ −7.09 × 10⁻⁹.
  - Positive control: at cycle 198 the per-cycle aggregates equal the component levels to 1.7 × 10⁻¹³ €.
- **3 × 3 rows** (multi-scenario, so the simple bound does not apply):
  - Voltage is taken element-wise from the run's results workbook (uncommitted, hashed): its maximum is ≤ −9.5e-9.
  - Flexibility uses a pair bound: the workbook net up − down plus the committed block sum and the scenario
    probabilities (1/9 each, from the committed `multiscenario_terminal.json`).
  - Closure uses S + (N − 1)·f.

### Per-evaluation table

Every value is at the evaluation's point P: k* for certified rows, the final or production-exit cycle otherwise.

How to read the columns:

- **€ columns:** signed € content in Q, weighted (gross convention).
- **"max element":** the largest proven upper bound on any slack element of the row, in p.u. or p.u.². It is ≤ 0
  where proven.
- **"positive €":** the positive content in Q. It is exactly 0 where every element is proven ≤ 0. Otherwise it is the
  bound.
- **"ESSO max":** the per-element upper bound on σ±, in MW. It is not in Q.
- **Node balance, branch flow, local-ESS closure:** exactly 0.0 in every block of every row, so they have no column.

| Cell | Table(s) | Status | P | Eval key | Plan | Voltage € | Flex P € | Closure € | Σ signed € | Max element | Positive € | ESSO max (MW) | P5 |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `pb_y2025_n5_v6` | 7 | cert. | 199 | `be95e576` | 5: 0.25/0.5 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `ref:7aa017f0` | 7, 10, 11, 12, 13, 14 | cert. | 181 | `d110bd1a` | x = 0 (2025) | −12,057.44 | −955.13 | 0.00 | −13,012.57 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `ref:bd504ecf` | 7, 12, 13, 14 | cert. | 172 | `3f084f2f` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `d_d3709599` | 7 | cert. | 175 | `64408bae` | 7: 0.5/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `d_c52e1670` | 7 | cert. | 150 | `af42a163` | 7: 0.5/2 (2025) | −13,012.57 (V + flex, aggregate) | (in V) | −19.90 | −13,032.47 | −7.089e-09 | 0.00 | −9.948e-09 | HOLDS |
| `c_156ce2d1` | 7 | cert. | 176 | `3a4c10c8` | 5: 0.25/0.5; 9: 0.25/0.5 (2025) | −12,057.44 | −955.13 | −39.80 | −13,052.36 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `c_6597a79d` | 7 | cert. | 193 | `3a0d03e8` | 5: 0.25/0.5; 7: 0.25/0.5 (2030) | −12,057.44 | −955.13 | −25.20 | −13,037.77 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `h_50dea31c` | 9 | cert. | 146 | `354b1bda` | x = 0 (2025), m = 2 | −12,057.44 | −955.13 | 0.00 | −13,012.57 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `h_74eda68d` | 9 | cert. | 150 | `dca08628` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `i_5a6a88b4` | 9 | cert. | 198 | `e046a8cf` | 7: 0.5/2 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `j_f3aa335e` | 9 | uncert. (final iter.) | 260 | `af066d6b` | 7: 0.75/3 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | holds |
| `j_a11d7966` | 9 | cert. (flagged) | 158 | `87efbc3d` | 7: 1/4 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `j_5f3cccb4` | 9 | uncert. (final iter.) | 270 | `cc7f0059` | 7: 1.25/5 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | holds |
| `e_c2_calfade` | 10 | cert. | 172 | `bbd82994` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `e_c2` | 10 | cert. | 172 | `ba969ad2` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `e_c3_unit` | 10 | cert. | 160 | `409740fb` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `e_c3_midblock` | 10 | cert. | 174 | `8b1eace9` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `e_c4` | 10 | cert. | 173 | `c59d7006` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `e_no_ageing` | 10 | cert. | 177 | `1b10e9e1` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `e_soh050` | 10 | cert. | 184 | `3f01b9ea` | 7: 0.25/1 (2025) | −12,057.44 | −955.13 | −19.90 | −13,032.47 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `t11_3x3_x0` | 12 | prod. exit | 72 | `f6e9cd53` | x = 0 (2025) | −11,765.67 | −934.36 | 0.00 | −12,700.04 | V −9.904e-09 (wb); flex −3.149e-09 | 0.00 | −9.738e-09 | holds |
| `t11_3x3_unit` | 12 | prod. exit | 69 | `c82522f4` | 7: 0.25/1 (2025) | −11,773.98 | −934.57 | −19.51 | −12,728.07 | V −9.547e-09 (wb); **flex +4.978e-08** | **≤ 2.16** | −9.738e-09 | **not established** |
| `t11_3x3_x0_continuation` | 12 (D fit) | cont. terminal | 88 | `25b92ae0` | x = 0 (2025) | −11,765.83 | −934.36 | 0.00 | −12,700.19 | V −9.904e-09 (wb); flex −3.149e-09 | 0.00 | −9.738e-09 | holds |
| `yl_y2030` | 13 | cert. | 169 | `6e4f11a9` | 7: 0.25/1 (2030) | −12,057.44 | −955.13 | −12.60 | −13,025.17 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |
| `yl_y2035` | 13 | cert. | 167 | `3a2387f4` | 7: 0.25/1 (2035) | −12,057.44 | −955.13 | −5.99 | −13,018.55 | −7.207e-09 | 0.00 | −9.843e-09 | HOLDS |

Notes on the table:

- **Table 14** reweights the two 2 % evaluations, `ref:7aa017f0` and `ref:bd504ecf`.
- **Table 12** also uses those two for its single-scenario value. Its D adjustment is fitted on the continuation's
  record.
- **The ESSO sum D3** is −1.72798 × 10⁻⁵ MW over 1,728 Vars (single-scenario) and −2.87997 × 10⁻⁵ MW over 2,880 Vars
  (3 × 3). Multiplied by c^σ it gives −0.0173 and −0.0288 in the ESSO objective's own units. Neither enters Q.

**Beside (not behind a named table): 39 other frozen T2 / appended cells.** 38 hold the same way: every element is
proven ≤ 0 and the positive content is 0 €. The exception is `pb_y2025_n5` (superseded; Blocked 4).

### Every non-zero content, with its row

- **Signed content.** It is non-zero on every row, and always negative: −12,700.04 € to −13,052.36 €, as in the table.
  It is the relaxation residue, not a penalty paid.
- **Positive content.** It is 0 € on every certified named row. It is not established on `t11_3x3_unit` (≤ 2.16 €,
  flexibility P). It is positive on the superseded `pb_y2025_n5` beside row (≤ 170.69 €).

### Reconciliation with W182g and T0.1c

- **W182g's "−12.7 to −13.0 k€".** That range is the signed content above, which equals `detector_penalty_total`:
  - yl_y2035: −13,018.55 €.
  - 3 × 3 x = 0: −12,700.04 €.
  - 3 × 3 unit: −12,728.07 €.
- **T0.1c's Q_cert⁺ − gross = 13,012.57 / 13,032.47 €.** These equal −`detector_penalty_total` at T0.1c's own cells,
  which are the W86 tight-tail pair (x = 0 at cycle 132, the unit at cycle 112). They match to 1.8 × 10⁻¹² €.
  - The reported cells this pair continues (`ref:7aa017f0` at 181, `ref:bd504ecf` at 172) carry −13,012.57 € and
    −13,032.47 € to the cent.
  - **Sign:** Q_cert⁺ = gross − (signed content), i.e. gross with every slack projected to max(x, 0). Q is lower than
    that by about 13.0 k€.

### Pickle-only items (listed, never opened)

The per-Var slack values exist only in each eval directory's `certified_models.pkl` (network models) and
`esso_models_s39_D.pkl`. The sha256 values below come from each directory's `child_manifest_sha256.json`.

- I did not establish from records whether `certified_models.pkl` holds cycle P or the terminal cycle. It is written
  by `post_certification`. The two differ only for `d_c52e1670` (P = 150, terminal 198).
- The 3 × 3 directories hold **no** `certified_models.pkl`, only the ESSO pickle.

| Cell | `certified_models.pkl` sha256 | `esso_models_s39_D.pkl` sha256 |
|---|---|---|
| `pb_y2025_n5_v6` | `f89bee0438b95e663132faf266a991ff193f00482c00f19ae6d6687435268c4d` | `951cce7871e410467e84bc759f9cf9b52525a0f4a0c760cba695f3c7e4b19eaa` |
| `ref:7aa017f0` | `99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a` | `118487e1853bbad176f324b1173df8c4198cb6c7709c25d7dce9f94ce4c199fa` |
| `ref:bd504ecf` | `d0a3d512fa58f41053befdb784bb842ad57ec0d71c989aef776a0aa668e6030d` | `73a6a67cac3a35c79a8746ec1693d80e9ba6f85f10ad0dfa07c1fa7c818d51e7` |
| `d_d3709599` | `daeb10b59bb5fae105dee8749fe5643f082ecf2b7c6dfcaea8c37032f2dbca8a` | `bea9e337afc00503db03399c83261f8100b36a6cce9f59013bdd958179c188c1` |
| `d_c52e1670` | `4b14b6d03a3f70561e3593be0ee2f6347a7a7a8e47c2cb1e44fa46cf2aecb041` | `befb19ec85cf7b0cdc9f11a310686f3fd46bbf2532bf93fb4992e3e8145349ca` |
| `c_156ce2d1` | `01df5703970e1a4350873ce305d9cf8fa6f1dc4f4cabc6bf82e1ed50930a2b02` | `ca7120b5c27fd08121eec759affbd90c68392bd6388daf58b68955e878e1037b` |
| `c_6597a79d` | `bdc6b32ca0a321cd95f3cf3f17a6ebbc0f49cc3e0a1113d3a0ae15ac12886820` | `5e9e9324d41d07df3438a1d8af8e5acc863a867c343edec5ba774ae784681285` |
| `h_50dea31c` | `dcb442049c709374dd72b59ad7710f4780e84acc769248d4d23464cd4a69e885` | `da5a9dba2796ddec59a218f51428355b524080ac533f87fe3c8ef0bb172f8d37` |
| `h_74eda68d` | `3006b71aa255e29c6c78c83590d7e8a2eee32af90b4f7a49ac05766352ab7be4` | `d52bba4f9f785ef1a874fc3725919f68a406f63e294025e8f98a92bd5d818bb9` |
| `i_5a6a88b4` | `bf12c1d1d22d521eb69f0e40931e583df051c0242d2a5359463c316af9c9c013` | `fa13df5110a406def4b8ec02a5ba541fae30ea96136a93cb92913d16aeb7e6d0` |
| `j_f3aa335e` | `45ca2e93c41a0563884016554971189e793f12af978ddd19476f94865140dacc` | `30868a2a7144a325e37fc687e4794b2b6cbd44567befc559795850033a52e30b` |
| `j_a11d7966` | `5c7afafedef3d1e45284e1dce1c19c4f3ebbaffc9d20db3026a42a9728a2244d` | `7e9f983288f5e62ae07789acd89b377aa5bcc55d2468ddd1dff94c7006da1b20` |
| `j_5f3cccb4` | `031692873006af8fcddf439699ae139ef0298890f7774e13ef829433377ab481` | `4ce100746ab844b51bce294e87839e30c3f8f3b247a5247c29f453d81b10eebe` |
| `e_c2_calfade` | `1373a20b1236b6f4a508b766d554a9b7252a72fbef03a8e711f5a52ce712f976` | `a3e1b9868fd58da2acf6f7d8b5c27edb0cda841430b79808cfac8832f3b98c99` |
| `e_c2` | `fe9b5a4482ebd30931ec9b0a2f644e8afd0258ba950df95fbd91fcd505e34778` | `8423e41545b88a89a0ee69c6ea601f8728f1e491d67692492d2e586d517e9d8a` |
| `e_c3_unit` | `3cb13febbe76ba61255b2a11280fa170260d7bdb5712c04cda71fb34bebd0bef` | `9e5300328af6e856d2310a99f0633df250357238a0b41d12077a7cce8bf97bd0` |
| `e_c3_midblock` | `0519f557322fd0c148603fad38e253c3f4574de6109cbdc6afce2db7dc725a25` | `044abe377a4978acf527c343daf8be34e3d394b09312c4f01470b3221c91c369` |
| `e_c4` | `fd29450386f79be29e1f1d71eb4e807d4776e512de6f9ff01dfc0efceba0d267` | `6bba2828c278a8279d496aac66597121bb84d475e9a12fcb99277f509e7e7c92` |
| `e_no_ageing` | `30b88f32e89b60d91b5bcb2c5236d74701f584d5e45d682e82b0c54da5c9c558` | `fa27f4e26dae683a0e88fbb0635c4e177454782e59ca110df94657828286dbde` |
| `e_soh050` | `fb337b90da3a66e7e94b57ff252e5c951593cab7f01e58b4cdf2f0b9fc4047fa` | `1a5b199e2cde10728c9297871664d5bbd7892355156a91bfbf2507534421f73c` |
| `t11_3x3_x0` | none | `76b66d50f09c045fce7c30cc3f71d34f49965a53bbebcbdd92b4f74ff842bf10` |
| `t11_3x3_unit` | none | `0dc2bed31dae790a388677e30d5ecc4b3271df281ba6828ca68218f1e716b376` |
| `t11_3x3_x0_continuation` | none | `6c193270614f4d48957d567daadde12fd4d248c0f508f1bac57394bf06da8b5e` |
| `yl_y2030` | `b5455a926c14f02b92e98adc299150eb83138f1a27037a43d4f7c5dbc7cffa59` | `1b367957b47638f169d0f8158b66c18731209413616d1c10ee66da952f2e41bf` |
| `yl_y2035` | `3ebfcd1a4b4e3c70ea18fa3501ceeed6386d9a72e2ee1dd8f4572159d0ab7eca` | `173dca155f0930f7ee59dcb78dcde0f198e83351e82936cb11c7c69539fc5c1e` |

Paths are under each row's `eval_dir`, given in `cells.<cell>.eval_dir` of the JSON. The beside cells' pickles are
listed there as well.

**No load is needed for P5.** The certified rows are settled from records.

---

## Unexpected findings (reported, not acted on)

1. **The superseded `pb_y2025_n5` has a positive voltage slack sum** in TSO|2035|Spring (Blocked 4). That is the same
   block whose closure sum W179c found positive; there the primary exited at max-iterations and the recovery ended at
   the acceptable level. The Appendix E sentence at l. 2934 mentions only the closure.
2. **Exploratory, not preserved:** the 3 × 3 unit workbook's "Voltage realized violation, up" reaches 1.67 × 10⁻⁹ p.u.
   (TSO) and 4.8 × 10⁻¹¹ p.u. (DSO). This is consistent with the bound relaxation on vmag, and it is not a penalised
   slack. I saw it while scoping the workbook in the scratchpad. The harness does not record it.
3. **The workbook stores flexibility and closure only as the net up − down.** It does not store them per leg
   (network.py:1830, :1852). So the per-leg values for multi-scenario rows exist only in the network models, and the
   3 × 3 runs did not persist those.

---

## Not confirmed

- **Table numbers as typeset.** No compile was possible: no TeX on this machine, no PDF or `.aux` in the clone tree at
  `a87e195`. See Blocked 1.
- **The floor premise at runtime.** "Every element ≥ −1e-8" rests on:
  - IPOPT's interior-point property;
  - the options being the defaults. I scanned the case and ESS parameter files and all 64 cell records. Only the three
    3 × 3 records state the options in force, and they state 1e-8 / no.
  I did not read the IPOPT logs' "Variable bound violation" lines here. W179c checked them at `d_c52e1670` only
  (≤ 9.9992e-9).
- **The code at each run's head.** The slack families and counts were read at repository HEAD. The records' detector
  fields are what each run computed. I did not re-check, for every run head, that production text is identical (W179c
  did so for five x = 0 runs).
  - The counts are confirmed independently by the 3 × 3 workbooks: 43,200 / 829,440 non-zero voltage elements, exactly
    the declared numbers.
  - The single-scenario counts are supported by the recorded sums: no implied count exceeds the declared one (asserted).
    Every block lies within 0.28 f per block of −N·f.
- **Per-leg flexibility values at `t11_3x3_unit`.** These are in no record. No pickle holds them in that eval directory
  (scoped to the directory's file listing).
- **Row 18 (the 3 × 3 commitment charge).** It is an economic charge, not a feasibility slack, so I did not inventory
  it. W182g records 25.08 M€.
- **The W169 capture plan** (`w169_slack_inventory/capture_plan.json`, committed) was used as the path index. Each
  cell's record eval key and candidate key were asserted equal to the frozen JSON's, which covers 61 of the 64 cells.
  The three 3 × 3 cells are not in T2, so for them the point was asserted equal to `cycles_run`.
