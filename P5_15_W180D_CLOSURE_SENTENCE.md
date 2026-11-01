# P5.15 W180d: §2.3.3 closure sentence as reworded in round 6 (Addendum 79, item (d))

**Worker, 2026-10-16.** Zero solves, zero model builds, zero pickle loads. W179c's outputs, the Overleaf clone,
production code, `manuscript_review/` and the brief were only read.

**Manuscript.** `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `e5322f85dced8034312c9afb97c19001bce6bfa0`
(`e5322f8`).
- sha256 `b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492`.
- It is byte-identical to the git object `e5322f8:main.tex` and to `manuscript_review/main_48099ab_round6.tex`.
- It is unmodified in the clone.
- The sentence is at **l. 969** (968 in the task's numbering). It equals the §A.3 block of
  `STEP6_ROUND6_CORRECTIONS.md` after whitespace normalisation. The old wording ("within solver tolerance, at every
  certified evaluation reported in this study") occurs 0 times.

**Repository.** HEAD `c76f6312`, branch `feature/derivative-free-planning`.

**Evidence base.**
- Harness: `p515_s53_w180d_closure_sentence.py` (sha256 `da4bc364…`).
- Output directory: `data/SRP1/Results/P515S53/w180d_closure_sentence/`. It was fresh before the run, and the harness
  refuses to start unless the directory holds only `launch.log`.
  - `w180d_closure_sentence.json`: sha256 `49a8f72d5d8cab093940b4a40c724bf7aab1c47891963363e75291957c26bc70`.
  - `manifest_sha256.json`: sha256 `13b24669…`. It lists 151 inputs; the 49 uncommitted ones are the IPOPT logs that
    W179c hash-recorded, each re-verified against W179c's manifest.
  - `launch.log`: sha256 `cfc314ba…`. The run was attached, with both streams captured; exit 0.
- Guards:
  - `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0, with all
    four counters at 0.
  - `pickle.load` and `pickle.loads` were blocked and called 0 times.
  - No production module was imported (asserted).
  - All JSON was written through `gate_result_io` (the shared writer).
- W179c source: verified **before use**.
  - `w179c_closure_scope.json` sha256 `417c270f…` equals both its manifest entry and the value in W179c's report.
  - The manifest is `d4197604…`.
  - The script `2b27da2d…` equals its manifest entry.
  - All three files are committed.
- A trial run of an identical copy of the harness, written into the scratchpad, gave the same console output. It is
  not preserved in the repository.

**Objective convention.** The closure level S = s⁺ + s⁻ is in MWh, summed over the units of a block. W179c read it as
production's penalty value divided by `PENALTY_SHARED_ESS_BALANCE` (1e3 €/MWh). No Q is quoted in this report.

**Instance identity.** Every point is recorded with its eval key, candidate key and canonical candidate under
`points[*]` in the JSON. Table E.21 rows are mapped to cells by the committed generator's order: `git show
HEAD:manuscript_review/gen_appE.py`, parsed with `ast` and not executed. That order agrees with W179c's row→cell
mapping for all 61 rows.

---

## Blocked on Planner

1. **Prediction (d) fails on the count only.** The sentence holds at **50 of 50** points, not 52 of 52. No point fails.
   - The population at `e5322f8` is 48 current E.21 certificates plus the two multi-scenario evaluations.
   - "50 current certificates + 2 T11" counts the T11 pair twice. Hypothesis, not verified with the expert: the
     expert's 50 is W179c's "50 of 51", and that 51 already included the two T11 points (49 E.21 certified rows at
     `48099ab` + 2).
   - There is nothing to change in the manuscript. The Planner should record the corrected count.
2. **Clause (iv) is L (wording).**
   - The manuscript says "Without storage the closure row is not built". In fact the row *is* built, for every unit
     (`network.py:511`, unconditionally inside `_build_model`). When the unit has no capacity,
     `configure_shared_ess_operational_state` then **deactivates** it (`model_construction_helpers.py:1176`), **fixes**
     both slacks at 0.0 (`:1143`) and **bounds** them to [0, 0] (`:1166`).
   - The substance holds: the value is exactly 0.0 at all five x = 0 points.
   - Expert's call. Candidate wording:
     - "Without storage the closure row is switched off and its slacks are fixed at zero."
     - Or, shorter: "Without storage the closure row is not part of the solved problem."
   - The second form rests on the Pyomo NL writer omitting deactivated rows and fixed Vars. That was **not tested
     here** (see Not confirmed). The first form rests only on the code read.

No run depends on either item.

---

## Changed

- The commit that adds this report (its hash is in the Worker report to the Planner; a commit cannot carry its own hash): `p515_s53_w180d_closure_sentence.py`, this report, and the three output files
  (`w180d_closure_sentence.json`, `manifest_sha256.json`, `launch.log`), staged by explicit filename.
- Nothing else was modified: no production code, specs, frozen files, `manuscript_review/`, clone, TASKS.md,
  REVISION_CONTEXT.md, brief, or W179 scripts or outputs. Not pushed.

---

## Found

### Prediction (d) against outcome

| Prediction (expert, Addendum 79) | Outcome |
|---|---|
| "the sentence holds at 52 of 52 points" | **Holds at 50 of 50; the count 52 is not reproduced.** 48 E.21 current certificates (44 with storage at the relaxed bound, 4 x = 0 exactly 0) + 2 multi-scenario evaluations (unit at the relaxed bound, x = 0 exactly 0). Row 16 is outside the population and is reported by the sentence itself, correctly. |

### Clause by clause

| # | Clause | Rating | Basis |
|---|---|---|---|
| (i) | "At every certificate used in a reported comparison" | **MATCH** (count 50, not 52) | See (i) below. |
| (ii) | "sat at their lower bound to within the solver's bound relaxation ($10^{-6}$~MWh per slack)" | **MATCH** | 10⁻⁸ p.u. × 100 MVA × 1 h = 10⁻⁶ MWh. bound_relax_factor is not user-set anywhere. Every slack lies in [−1.000, −0.995] × 10⁻⁶ MWh. |
| (iii) | "in the superseded evaluation~16 … their sum reached $1.8 \times 10^{-6}$~MWh, $5 \times 10^{-6}$ of the available energy" | **MATCH** | E.21 row 16 = `pb_y2025_n5` at k* 167. TSO 2035 Spring: S = +1.7637 × 10⁻⁶ MWh, S/E^Av = 5.039 × 10⁻⁶. It is the only positive block. |
| (iv) | "Without storage the closure row is not built." | **L** | The row is built, then deactivated; the slacks are fixed at 0.0 with bounds [0, 0]. Values are exactly 0.0. |
| (v) | "and at both multi-scenario evaluations" | **MATCH** | `t11_3x3_x0` at 72: 80 blocks, exactly 0. `t11_3x3_unit` at 69: 40 active blocks at the relaxed bound. |

### (i) The population, from Appendix E as printed at `e5322f8`

**E.21 at `e5322f8` against `48099ab`.**
- Over all 61 rows × 13 columns, exactly two entries differ:
  - row 16 Status: `cert., osc.$^{s}$` → `superseded$^{s}$`;
  - row 16 Rule: `record` → `---`.
- Every other row, including every k*/end, is unchanged. W179c's per-row points therefore apply unchanged.

**Certified rows at `e5322f8`: 48.** These are the rows whose Status begins with "cert.": 1–5, 7, 10–15, 17–30, 32–36,
38, 43, 44, 46–49 and 52–61.
- Non-certified rows (13): 6, 8, 9, 31, 37, 39–42, 45, 50 and 51 are uncertified, and **16 is superseded**.

**Use by the comparison tables.**

| Source | Evaluations named |
|---|---|
| E.20 Eval column | 1–15, 22, 23, 26–60 |
| E.22 | 15, 17–23 |
| E.24 | 6, 31, 39, 40, 41, 42, 45, 50, 51 (all uncertified) |
| E.20 note | 2, 56, 61 |
| E.21 note | 1, 2, 15, 16, 40, 41 |
| E.22 note | 16 |

**Certified rows used by E.20, E.22 or E.24: 45 of 48.** The other three are also used in reported comparisons:
- **24, 25**: E.21 group (c) serves `tab:res_year_ladder`. The 2035 − 2030 comparison is stated at l. 1638 ("The gross
  difference between the two deferral years, 42.5 k€, is determinate").
- **61**: the E.20 note ("Row 58 against evaluation 61 … gives the same figures"). It is also in group (f), which serves
  `tab:res_ageing`.

So all 48 current certificates are in the population.

**Twins.**
- Rows 56 and 61 are the ^t twins of evaluation 2. Their per-block closure sums are **bitwise equal** to evaluation 2's
  (checked).
- Counting distinct certificates instead gives 46 + 2 = **48 of 48** points. This is the "46 certificates the tables
  use" of l. 1691.

**Points, and how the x = 0 cells count.**

| Set | Points | At the relaxed lower bound | Exactly 0.0 (no storage) | Fails |
|---|---:|---:|---:|---:|
| E.21 current certificates | 48 | 44 | 4 (rows 1, 30, 32, 34) | 0 |
| Multi-scenario pair | 2 | 1 (`t11_3x3_unit`, 69) | 1 (`t11_3x3_x0`, 72) | 0 |
| **Total** | **50** | **45** | **5** | **0** |

- **x = 0 cells.** The five x = 0 points satisfy the first clause literally: each slack is exactly 0.0, which is its
  lower bound, at distance 0. The last sentence ("Without storage …") gives the reason, so it is a description of these
  points, not an exception to them.
- **`d_c52e1670` (row 3, k* 150)** is read from the committed per-cycle record, as in W179c.
- **The superseded row 16** is outside "every certificate used in a reported comparison", and the sentence reports it
  separately. Its first-certificate comparison still appears in the E.22 note ("gave 51.3 k€ (3.79×)"). The §4.7
  wording "used in no comparison" belongs to W180b/c; it is noted here only because the closure sentence's coverage
  does not depend on it.

### (ii) The bound relaxation and its conversion to MWh

**IPOPT version.** 3.14.18 with ma97: the only version line in all 49 logs that W179c hash-recorded.

**Options in force.** The relevant options are `bound_relax_factor`, `honor_original_bounds` and `constr_viol_tol`.
None of them is set:
- in any case file's `options` or `recovery_options` (case9, case33_1/2/3);
- in any of the 20 production source files;
- in any of the 8,809 "List of options" blocks of those 49 logs. The union of the printed options is: acceptable_iter,
  acceptable_tol, bound_frac, bound_push, compl_inf_tol, file_append, file_print_level, fixed_variable_treatment,
  linear_solver, max_iter, output_file, slack_bound_frac, slack_bound_push, tol, warm_start_init_point.

**IPOPT's rule.** Source: `IpOrigIpoptNLP.cpp` at the 3.14.18 release tag (URL and sha256 `c640f9df…` of the fetched
file recorded in the harness; quoted, not fetched at run time).
- The registered defaults are `bound_relax_factor` = 1e-8, `honor_original_bounds` = false and `constr_viol_tol` = 1e-4.
- `relax_bounds` moves each original, unscaled bound by min(constr_viol_tol, factor × max(|bound|, 1)).
- For lb = 0 that is min(10⁻⁴, 10⁻⁸) = **10⁻⁸** in the variable's own units. The logs show "No x scaling provided".
- Because `honor_original_bounds` is off, the returned point is not projected back. This is why the recorded S < 0.

**Units in production.**
- The slack Vars are `NonNegativeReals`, and `setlb(0.0)` is applied (`network.py:407–408`;
  `model_construction_helpers.py:1162–1169`, `setlb(0.0)` at `:1168`).
- Capacities enter as MWh / baseMVA (`network_data.py:2935, 2949`).
- The penalty and the result writer multiply the slack by `network.baseMVA` (`model_construction_helpers.py:2341`;
  `network.py:1830`).
- baseMVA = 100 in all 68 case files, and dt = 24 h / 24 periods = 1 h.

**Conversion.** 10⁻⁸ p.u. × 100 MVA × 1 h = **10⁻⁶ MWh per slack**, so the figure in the sentence is the correct
conversion.

**Observed values (W179c's per-block sums, recomputed here).**
- Each Var is ≥ −10⁻⁶ MWh: IPOPT's iterates stay inside the relaxed bounds. W179c's logged "Variable bound violation" is
  ≤ 9.9992 × 10⁻⁹ p.u. at `d_c52e1670`.
- Each Var is ≤ S + (2n − 1) × 10⁻⁶ MWh.
- Distance of every slack from the relaxed floor:
  - **≤ 3.76 × 10⁻¹⁰ MWh** over the 44 E.21 certificates with storage. The maximum is at `l_df1a5525` (row 53); the
    single-unit blocks are at ≈ 1.86 × 10⁻¹⁰.
  - **≤ 5.01 × 10⁻⁹ MWh** at `t11_3x3_unit`.
- Distance from the nominal bound 0: every slack lies between **0.995 × 10⁻⁶ and 1.000 × 10⁻⁶ MWh below 0**. That is,
  every slack sits at the relaxed bound, within the 10⁻⁶ MWh the sentence states.

### (iii) Row 16

**Identity of evaluation 16.** Evaluation 16 of `tab:appE_evaluations` (l. 2525) is "5: 0.25/0.5, 2025, base,
superseded^s, k_0 110, k* 167". It maps to `pb_y2025_n5`:
- eval key `ca29c5e818366a6e…`;
- candidate key `33031c08…`, candidate {2025; node 5: [0.25, 0.5]};
- W179c point 167, equal to the printed k*;
- the plan and year agree.

**Values at k* 167.**
- 24 active blocks. 23 are at the relaxed lower bound and **one is positive**: `TSO|2035|Spring`, S = +1.76367 × 10⁻⁶
  MWh.
- E^Av (node 5, 2035, terminal) = 0.35 MWh, so S/E^Av = 5.039 × 10⁻⁶.

**Printed figures.**
- "1.8 × 10⁻⁶ MWh" is S to two significant figures, and "5 × 10⁻⁶" is S/E^Av to one. Both match.
- "their sum reached" is accurate: this is the largest sum, and the only positive one.

### (iv) "Without storage the closure row is not built": the mechanism (production source at HEAD)

The relevant production files are unchanged since W179c's head `67c081be`; `git diff` over the 20 production files is
empty.

**What the code does.**
- **Construction.** `model.sess_soc_final = pe.Constraint(model.shared_energy_storages, …)` is at `network.py:511`. It
  sits at indentation 4 directly in `_build_model` (governing statement `def _build_model(network, params):`, l. 251),
  so it is built for every shared-ESS unit, whatever the capacity.
  - Its rule `sess_soc_final_rule` (`model_construction_helpers.py:983`; row expression at `:990`) has exactly one
    `Constraint.Skip`, the duplicate-scenario guard.
  - The slack Vars (`network.py:407–408`) are declared under `if params.slacks.shared_ess.day_balance:`. That flag is
    true in all four case parameter files.
- **Inactive unit.** `configure_shared_ess_operational_state` tests `abs(S) <= 1e-10 or abs(E) <= 1e-10`
  (`shared_ess_capacity_is_inactive`, `:1069`) and, for an inactive unit:
  - **`:1143`** `entry.fix(0.0)` inside `for variable_name in _SHARED_ESS_OPERATIONAL_VARIABLES`. Both slacks are in that
    list (`:1012–1013`).
  - **`:1166`** `slack_ub = 0.0 if inactive …`, followed by `setlb(0.0)` (`:1168`) and `setub(slack_ub)`, giving bounds [0, 0].
  - **`:1176`** `entry.deactivate()` inside `for constraint_name in _SHARED_ESS_OPERATIONAL_CONSTRAINTS`.
    `'sess_soc_final'` is in that list (`:1062`).
- W179c's cited lines 1143, 1166 and 1176 are exact at HEAD, each inside `configure_shared_ess_operational_state`.

**What the records show.** All five x = 0 points are exactly 0.0 in every block: rows 1, 30, 32 and 34 (48 blocks each)
and `t11_3x3_x0` (80 blocks).

**Rating: L.** The row is built and then deactivated, not "not built". The substance (no closure without storage) holds.

### (v) "Both multi-scenario evaluations"

**The pair.** `tab:res_multiscenario` (l. 1613) reports the pair.
- `t11_3x3_x0` (`f6e9cd53…`, candidate `8435c718…`, all zero), stopped at 72: 80 blocks exactly 0.0.
- `t11_3x3_unit` (`c82522f4…`, node 7 [0.25, 1]), stopped at 69: 40 active blocks at the relaxed lower bound, each slack
  within 5.0 × 10⁻⁹ MWh of it.

**The continuation.** Table 13's D comes from the x = 0 continuation `25b92ae0…` (candidate `8435c718…`, x = 0). Its
committed `component_levels_terminal.json` at cycle 88 is exactly 0.0 in all 80 blocks. So the clause would hold even if
the continuation were counted.

---

## Unexpected findings (reported, not acted on)

1. **W179c's "within 5.1 × 10⁻¹⁰ MWh" figure is not what its own data give.**
   - Recomputed from W179c's per-block sums, the largest distance to the relaxed floor over its 44 is **3.76 × 10⁻¹⁰**
     (`l_df1a5525`). W179c's own text names −0.99962 × 10⁻⁶ at `l_2ab0ce2d`, which is 3.75 × 10⁻¹⁰.
   - 5.1 × 10⁻¹⁰ is therefore a valid but loose bound, and its origin was not traced.
   - The figure is not in the manuscript. It appears only in `STEP6_ROUND6_CORRECTIONS.md` §A.3's source note and in
     W179's reports.
2. **Table `tab:res_multiscenario`, l. 1619, reads "Value Q(0) − Q(unit) at operational stopping (cycle 72)".**
   - The unit evaluation stopped at **69** (`campaign_results.json` `per_cell.n7_4h_e1.cycles_run` = 69, certification
     cycle 69). Only the x = 0 evaluation stopped at 72.
   - The frozen JSON's quantity label reads "(cycle 72)" as well.
   - This is outside (d), and possibly an intended shorthand. It is flagged for the W180 consolidation.

---

## Not confirmed

- **That the solved NLP omits the deactivated row and the fixed slack Vars.** This is Pyomo NL-writer behaviour. It was
  not tested: no model was built or written, and no IPOPT log variable count was compared between active and inactive
  units. Only the source text was read.
- **Each slack individually.** Only the block sums S are recorded. The per-Var interval rests on IPOPT's interior-point
  property (iterates stay inside the relaxed bounds). W179c's logged bound violations support that property at
  `d_c52e1670` (≤ 9.9992 × 10⁻⁹ p.u. over 24 solves); it was not re-checked here at other cells.
- **The options in force at every point.** Checked in the case files, in production source, and in the 49 logs that
  W179c hash-recorded. Those logs come only from the `pb_y2025_n5` and `d_c52e1670` runs. The campaign harnesses'
  run-time `option_overrides` at other cells were not re-read. Prior art (`p515_s53_w139_resettle_v5_hooks.py`,
  `p515_s53_w121_benchmark_spec_v5.py`) states that bound_relax_factor, honor_original_bounds and constr_viol_tol were
  not passed, and that was not re-verified here.
- **The IPOPT source.** It was fetched once from GitHub at the `releases/3.14.18` tag, with its sha256 recorded. It is
  not committed. The installed binary at `/usr/local/bin/ipopt` was not executed; its version is from the logs.
- **The expert's 52.** The double-count reading is a hypothesis. I searched the round-6 text (`STEP6_ROUND6_CORRECTIONS.md`
  §A.3) and Addendum 79 for another population that gives 52 and found none.
- **"Used in a reported comparison" for rows 24/25 and 61.** This rests on the E.21 group heading, the l. 1638 sentence
  and the E.20 note. Table `tab:res_year_ladder` and the Section 4 tables were not parsed cell by cell.
