# P5.15 W179c: closure slack at the six missing points, and two scope statements (Addendum 78, items (d) and (e))

**Worker, 2026-10-15.** Zero solves, zero model builds and zero pickle loads. Production code, records, specs, the
frozen JSON, `manuscript_review/` and the manuscript clone were only read.

**Manuscript.** `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `48099ab`.
- sha256 `28008ab0d79b477c4e8d86a0c3a4d1acf9c95ed9c4921f4757f1600cda1c60af` (verified); `main.tex` is unmodified in the
  clone.
- The line numbers below are at `48099ab`.

**Repository.** HEAD `67c081be`, branch `feature/derivative-free-planning`.

**Helper script.** `p515_s53_w179c_closure_scope.py` (sha256 `2b27da2d…`).
- Outputs: `data/SRP1/Results/P515S53/w179c_closure_scope/`, written once.
  - `w179c_closure_scope.json`: sha256 `417c270f8fae66c884271aa403ea15e92931af531ff70397300ef653c3f4e19c`.
  - `manifest_sha256.json`: sha256 `d4197604…`. It lists 367 inputs with sha256 and committed/uncommitted status;
    51 of them are uncommitted.
  - `launch.log`: sha256 `0e3f17dc…`. Both streams were captured and the run was attached; exit 0.
- Guards: `SolveProfileGuard(permitted=())` was verified at exactly 0 solves and 0 execs. `pickle.load`/`loads` were
  blocked and called 0 times. No production module was imported (asserted).

**Objective convention.** The closure level S is s⁺ + s⁻, in MWh, summed over the units of a block. It is read from
production's penalty value divided by `PENALTY_SHARED_ESS_BALANCE` (1e3 €/MWh). Any Q quoted is
`gross_operational_cost`, settlement excluded.

**Instance identity.** Every cell is identified by its eval key and candidate key. These are in
`w179c_closure_scope.json` under `D_per_cell.<cell>`. Table E.21 rows are mapped to cells by the generator's own order
(`manuscript_review/gen_appE.py`, parsed with `ast` and not executed). All 61 rows agree with `export_W160/T2.csv` on
status and k*/end.

---

## Blocked on Planner

1. **The superseded certificate is now a reported certified evaluation.**
   - Ruling 6 of Addendum 69 scored W169 as "held at every current certificate". It excluded the superseded
     `pb_y2025_n5` because that certificate was outside the tables under Addenda 58/59.
   - At `48099ab` it is inside them:
     - Table E.21 row 16, "cert., osc.^s", k* 167 (l. 2494 caption, l. 2580 note).
     - The Table E.22 note, "gave 51.3 k€ (3.79×)" (l. 2615).
   - In that evaluation, the TSO block 2035 Spring has a positive closure sum: S = +1.764 × 10⁻⁶ MWh, which is
     5.04 × 10⁻⁶ of E^Av.
   - The §2.3.3 sentence "at every certified evaluation reported in this study" (l. 968) therefore covers it.
   - **Decision needed (expert wording):**
     - (a) Scope the sentence to the current certificates and give the row-16 figure.
     - (b) Restate it as a magnitude bound that covers all 49.
     - (c) Drop row 16 from the reported set. This changes Appendix E and is not recommended.
   - Draft sentences are given under (d) below. Recommendation: (a).
2. **"Within solver tolerance" is the wrong description at x = 0.**
   - There, the slack Vars are fixed at 0.0 and the closure row is deactivated, so the slack is not in the solved NLP.
   - A short clause is proposed under (d). This is wording (L).
3. **§4.8 Scope (l. 1724) needs a qualifier.** It matches the prediction. The minimal wording is under (e).
4. **§2.2.7 (l. 743) needs a qualifier for the 3 × 3 pair.**
   - The other gaps listed under (e) are smaller and are left to the expert.
   - In the same sentence group, "every cell it proposed was re-evaluated under this rule before being reported" is
     not literally true for the 6 + 4 search-only neighbours, which are reported as counts. This is listed under (e)
     and needs an expert call.
5. **W169's coverage statement was wrong in one respect (record only; no action on W169's committed artefacts).**
   - W169 said no committed record carries the closure at cycle 150 for `d_c52e1670`. One does: see "Unexpected
     finding" under (d).
   - Nothing W169 scored changes.

---

## Prediction vs outcome

| # | Prediction (expert, Addendum 78) | Outcome |
|---|---|---|
| (d) | The six missing points also have zero closure slack within tolerance, so the claim holds at 51 of 51. | **PARTIAL.** The six points HOLD: five x = 0 cells are exactly 0.0 in every block, and `d_c52e1670` at k* = 150 is at the relaxed lower bound in all 24 node-7 blocks. **"51 of 51" does not hold on W169's signed reading: the count is 50 of 51.** The exception is `pb_y2025_n5` (E.21 row 16, superseded), at +1.76 × 10⁻⁶ MWh. That point is now reported in Appendix E, so it is inside the claim's scope. On an absolute p.u. reading (1.8 × 10⁻⁸ p.u., far below IPOPT tol 10⁻⁵) it would pass; the reading is the expert's call. |
| (e) | The §4.8 Scope wording needs "under the certification rule" or "at the search resolution" for the six search-only neighbours. | **HELD.** 8 of the 14 neighbours are in Table E.22 under the rule. The other 6 were evaluated by the search only, at 3.47–7.17× the search's own resolution (1.07–7.17× over all 14), on the search's code head `353e094b`. "Determinately worse" holds for the 8 only. |
| (e) §2.2.7 | No prediction recorded. | The wording needs a qualifier: the 3 × 3 evaluations behind Table 13 are absent from Appendix E, and other, smaller gaps exist (see (e)). |

---

## (d) Closure slack, §2.3.3 (l. 968)

### Coverage

**What this inventory covers.**
- **The 49 certified rows of Table E.21.** These are rows 1–5, 7, 10–30, 32–36, 38, 43, 44, 46–49 and 52–61. Every row
  whose Status column begins with "cert." is included.
- **The two production-exit points W169 also counted**, from Table 13 (T11): `t11_3x3_x0` at 72 and `t11_3x3_unit`
  at 69.
- **Total: 51.** This is exactly W169's certified set; the equality was checked.

**How each point was scored.** Every point was re-read here from records and scored, so 51 of 51 have a source.
- **50 points** come from the committed `component_levels_terminal.json`. For each of these, `cycles_run` equals the
  point, and this was asserted.
- **One point, `d_c52e1670` at 150**, comes from the committed per-cycle record. See the next subsection.

**Classification.** Each block is classified by its number of active units (published E^Av > 0) and its read sum S:
- **exact_zero**: no active unit and S == 0.0.
- **at_relaxed_lower_bound**: −2 × 10⁻⁶ × n_active ≤ S ≤ 0 MWh. The floor is IPOPT's default bound relaxation:
  10⁻⁸ p.u. on lb = 0, times baseMVA 100, per Var.
- Anything else counts as a failure.

| Set | Points | Signed reading holds | Exception |
|---|---:|---:|---|
| E.21 certified rows | 49 | 48 | row 16 `pb_y2025_n5` (superseded): `TSO|2035|Spring` S = +1.7637 × 10⁻⁶ MWh, S/E^Av = 5.04 × 10⁻⁶ |
| T11 production-exit points | 2 | 2 | — |
| **W169's 51** | 51 | **50** | as above |

**The 44 current certificates with storage.** These are the 49 rows, less the four x = 0 rows (1, 30, 32, 34) and less
row 16.
- Every block with an active unit is at the relaxed lower bound.
- The read sums lie between S = −1.99982 × 10⁻⁶ and −1.99962 × 10⁻⁶ MWh per active unit.

**Each slack Var is pinned by S alone.**
- Every Var is ≥ −10⁻⁶ MWh, the relaxed floor.
- The 2n Vars of a block sum to S.
- So each Var lies in [−10⁻⁶, S + (2n − 1) × 10⁻⁶] MWh.
- The largest upper end over the 44 is −0.99962 × 10⁻⁶ MWh, at `l_2ab0ce2d`. It is −0.99499 × 10⁻⁶ for
  `t11_3x3_unit`.
- **Every closure slack Var in every current certificate is therefore within 5.1 × 10⁻¹⁰ MWh of IPOPT's relaxed lower
  bound.** In W169's sense, "zero within solver tolerance" means "signed, at the relaxed bound", and it holds at every
  current certificate.

**Bound violations in the logs.** At `d_c52e1670`'s cycle 150, the per-solve "Variable bound violation" in the IPOPT
logs is ≤ 9.9992 × 10⁻⁹ p.u. in all 24 node-7 solves. That is consistent with the 10⁻⁸ floor.

**Row 16 (superseded).**
- With the same floor, each Var is ≤ 2.76 × 10⁻⁶ MWh. The recovered solve's log reports a bound violation of 0.0,
  which would give ≤ 1.76 × 10⁻⁶ MWh.
- This is the only positive closure sum among the 51.
- The block's solve at that cycle was a primary that exited on "Maximum Number of Iterations Exceeded", followed by a
  recovery that exited "Solved To Acceptable Level".

### The five x = 0 cells: identically zero, from records and from the code

| Cell | E.21 / table | P | Blocks | Every block S | Candidate | Published S^Av, E^Av | Run head (production diff to HEAD) |
|---|---|---:|---:|---|---|---|---|
| `ref:7aa017f0` | row 1 | 181 | 48 | exactly 0.0 | `8435c718` (all nodes [0, 0]) | 0.0 every node and year; `has_storage` false | `aad37370` (NO CHANGE) |
| `h_aa8a76d7` | row 30 | 153 | 48 | exactly 0.0 | `8435c718` | same | `9d2f458f` (NO CHANGE) |
| `h_x0_m175` | row 32 | 139 | 48 | exactly 0.0 | `8435c718` | same | `0e1d7c0e` (NO CHANGE) |
| `h_50dea31c` | row 34 | 146 | 48 | exactly 0.0 | `8435c718` | same | `3740b476` (NO CHANGE) |
| `t11_3x3_x0` | Table 13 | 72 | 80 | exactly 0.0 | `8435c718` | same (2025–2037) | `8a8ef150` (NO CHANGE) |

**Source of these values.** They come from the committed `component_levels_terminal.json` (`cycles_run` = P) and
`evaluation_record.json` (`candidate_canonical`, `storage_per_node.*.published_available_capacity_terminal`).
- In the three x = 0 cells that also have a per-cycle record, the per-cycle value at P is also exactly 0.0, and it
  matches `component_levels` bitwise.

**Is the closure row built when E^Av = 0? Built, then switched off.**
- `sess_soc_final_rule` builds the row for every unit (`model_construction_helpers.py:983`).
- `configure_shared_ess_operational_state` (`:1117`) is applied for each candidate. When the unit is inactive
  (`abs(S) <= 1e-10 or abs(E) <= 1e-10`, `:1069`), it:
  - fixes both slack Vars at 0.0 (`entry.fix(0.0)`, `:1143`; the slacks are in `_SHARED_ESS_OPERATIONAL_VARIABLES`,
    `:1012–1013`);
  - sets their bounds to [0, 0] (`slack_ub = 0.0 if inactive`, `:1166`);
  - deactivates `sess_soc_final` (it is in `_SHARED_ESS_OPERATIONAL_CONSTRAINTS`, `:1062`; `entry.deactivate()`,
    `:1176`).
- So the solved NLP has neither the row nor the slack Vars, and the value is 0.0 by construction.
- The same text is present at each run's git head (needles checked with `git show`). `git diff --stat <head> HEAD`
  over the 20 production files is empty for all five.

**Wording consequence.** "Zero within solver tolerance" is not the right description at x = 0. It is exactly zero
because the closure does not arise without storage. Proposed clause: "…(without storage the closure does not arise)".

### `d_c52e1670` at k* = 150

**The value is read from a committed record. It is not derived.**
- Source: `creep_diagnostic_per_cycle.jsonl` (committed, sha256 `52621dfe…8154f43`), cycle 150,
  `q_decomposition.blocks[*].ess_complementarity_penalties`. Each value is divided by the block's `admm_block_weight`
  and by 1e3.
- **Why this field is the closure.** Production's `total_ess_complementarity_penalties` is
  `local_ess_day_balance_slack_penalty + shared_ess_day_balance_slack_penalty` (`model_construction_helpers.py:2379–2382`).
  The local term is identically 0 here: no SRP1 case file has an `energy_storages` key (checked for 68 files), and
  `local_ess_day_balance_slack` is exactly 0.0 in every block of every cell read.
- **Positive control of the identity.** At the terminal cycle, per-cycle `ess_complementarity_penalties` equals
  `component_levels_terminal` weighted shared + local **bitwise (max |diff| 0.0) in all 47 cells** where both records
  exist. The four cells without a per-cycle file are `ref:7aa017f0`, `ref:bd504ecf` and the two T11 cells. The 47
  include `d_c52e1670` itself at 198.

**The instance at cycle 150.**
- The cycle-150 record's `gross` = 653,373,327.8473792. This is bitwise the v6 certificate's `Q_k_star`
  (`d_c52e1670_v6_from_records_certificate.json`, sha256 `c96e3588…`; k* 150).
- `local_solves_ok` is true.
- Candidate: `cc4eb5df` (2025, node 7: 0.5 MVA / 2 MWh). Eval key: `af42a163b14f0895`.

**Values.**
- **24 node-7 blocks (TSO and DSO7):** S ∈ [−1.999818, −1.999814] × 10⁻⁶ MWh, all at the relaxed lower bound.
  - S/E^Av (terminal E^Av) ranges from −1.14 × 10⁻⁶ to −1.43 × 10⁻⁶.
  - Each Var lies within 1.9 × 10⁻¹⁰ MWh of −10⁻⁶ MWh.
- **24 DSO5/DSO9 blocks:** exactly 0.0.

**Cross-check, derived from the schedule (labelled "derived").**
- Source: `ess_schedule_per_cycle.jsonl`, UNCOMMITTED, sha256
  `dad25fe99f361514520f3e1c41142c85d714b185d01b7e77b193bc9a672e9469`.
- Formula: s⁺ − s⁻ = D = Σ_t (0.97 P^ch_t − P^dch_t / 0.96) × 1 h, over the network copies' charge and discharge in MW.
  The writer has already applied prob × baseMVA (single scenario, prob 1).
- The initial and final targets are the same Param `shared_es_e_rated_fixed × 0.5`, so they cancel exactly.

**Precision of the cross-check.**
- The precision is 25 × CV × baseMVA plus a floating-point bound. CV is the final unscaled IPOPT "Constraint
  violation" of the block solve whose values were loaded at round 150.
- Those solves were located through `network_ipopt_solve_records.jsonl` `log_bytes` (round k = cycle k,
  `shared_resources_planning.py:2814`). The 24 logs are uncommitted; their sha256 values are in the manifest.
- Each of the 24 blocks has one primary attempt, "Optimal Solution Found", with CV between 7.3 × 10⁻¹⁵ and
  2.5 × 10⁻⁹ p.u.

**Result of the cross-check.**
- |D| ≤ 1.0 × 10⁻¹⁰ MWh in all 24 blocks.
- The precision bound ranges from 1.8 × 10⁻¹¹ to 6.2 × 10⁻⁶ MWh. The largest is `DSO|7|2035|Winter`, where
  CV = 2.5 × 10⁻⁹.
- D is consistent with s⁺ ≈ s⁻ ≈ −10⁻⁶ MWh. The cross-check confirms, but the read S already settles the value.

**Validation and control.**
- **Validation at the cell's terminal cycle 198**, against `component_levels_terminal.json`: |D| ≤ 2.7 × 10⁻¹⁰ MWh
  with the same pattern. Consistent.
- **Positive control `pb_y2025_n5` at 167: INCONCLUSIVE.** The block with the positive sum was recovered at
  CV = 7.2 × 10⁻⁸, so the precision bound there is 1.8 × 10⁻⁴ MWh, far wider than the value. The control does not
  discriminate. It is reported, not used.

### Re-rating of the claim (W174b form)

| Row | Text (l. 968) | Rating | Basis |
|---|---|---|---|
| d-1 | "Both daily closure slacks were zero, within solver tolerance, at every certified evaluation reported in this study." | **M: does not hold as worded at one reported certified evaluation** | 48 of 49 E.21 certified rows hold on the signed reading, and 2 of 2 T11 points. Row 16 (superseded, reported in E.21 and the E.22 note) has S = +1.76 × 10⁻⁶ MWh (5.0 × 10⁻⁶ E^Av). The magnitude is negligible, so this is not a different model. |
| d-2 | the same, at x = 0 | **L** | The slack is exactly zero because the row and Vars are not in the solved NLP; "within solver tolerance" misdescribes it. |

**Sentences the manuscript can carry, each scoped to what the evidence covers:**
- **(a), recommended:** "Both daily closure slacks were at their lower bound, to within the solver's bound relaxation,
  at every current certificate reported in this study (Table E.21 other than the superseded evaluation~16, where their
  sum was $1.8 \times 10^{-6}$~MWh); without storage the closure does not arise."
- **(b), magnitude form, covering all 49:** "At every certified evaluation reported in this study the daily closure
  slacks summed to at most $1.8 \times 10^{-6}$~MWh in any block ($5 \times 10^{-6}$ of the available energy), and to
  their lower bound, within the solver's bound relaxation, at every one except the superseded evaluation~16; without
  storage the closure does not arise."

**Coverage of these sentences.**
- Both cover the 49 E.21 certified rows. Both also hold at the two T11 points.
- For the 12 uncertified E.21 rows and the six T6 arms, I did not re-read anything; I cite W169. W169 found no other
  positive block among 63 of 64 ADMM evaluations at their points, and found the T6 arms' weighted totals to be exactly
  0.0.

**How `pb_y2025_n5` (+5 × 10⁻⁶ E^Av) relates to "every certified evaluation reported".**
- It is reported at `48099ab`:
  - E.21 row 16, with ^s "superseded by evaluation 15".
  - The E.21 note, l. 2580.
  - The E.22 note, l. 2615: 51.3 k€, 3.79×.
- It is therefore inside the claim's scope. It is the only point that fails the signed reading.

### Unexpected finding (W169 record correction)

**What W169 said.** For `d_c52e1670`, W169's "Cells not covered at P" (`w169_slack_inventory.md`; `records_carrying`
in its JSON) states that per-cycle records carry the closure only inside `slack_penalties` ("voltage + flexibility +
closure summed"), and that no record holds cycle 150.

**Both statements are wrong.**
- `slack_penalties` is voltage + node + branch + flexibility-P only (`model_construction_helpers.py:2301–2312`).
- The closure is in `ess_complementarity_penalties` (`:2369–2382`). The committed `creep_diagnostic_per_cycle.jsonl`
  q_decomposition carries that field per block, per cycle, for every W105/W118-hooked run.

**Impact.** Nothing W169 scored changes. Its committed artefacts were not touched.

---

## (e) Two scope statements

### §4.8 Scope (l. 1724): "with every evaluated unit neighbour of the plan without storage determinately worse"

**Records used.** W173 (`w173_search_counts.json`, committed, sha256 `0aa337fc…`, item2.x0.rows) and the parsed Table
E.22.
- Box: 14 neighbours, all evaluated by the S47 search's final unit poll.
- 8 are in Table E.22: evaluations 15, 17–23.
  - The 2025 node-5 neighbour maps to `pb_y2025_n5_v6` (row 15). The superseded row 16 is its first certificate.
  - All 8 are determinate under the certification rule: 3.40–8.09×.
- **6 were evaluated by the search only.** None appears in any table cell.

| Neighbour | Search ΔF (€) | Search resolution (€) | × |
|---|---:|---:|---:|
| 2025 nodes 7 + 9 | 113,663 | 18,450 (σ_Q) | 6.16 |
| 2030 nodes 7 + 9 | 102,830 | 18,450 | 5.57 |
| 2030 nodes 5 + 9 | 112,017 | 18,450 | 6.07 |
| 2025 nodes 5 + 7 | 101,829 | 18,450 | 5.52 |
| 2025 nodes 5 + 7 + 9 | 151,807 | 43,788 (max(bar sum, σ_Q)) | 3.47 |
| 2030 nodes 5 + 7 + 9 | 132,204 | 18,450 | 7.17 |

**Ratio ranges.** The six are at 3.47–7.17×. Over all 14 the range is 1.07–7.17×; the 1.07× is the 2025 node-5
neighbour, which is among the 8.

**Code and calibration of the search.**
- The search evaluations stopped at the production exit, on code head `353e094b`.
- `git diff --stat 353e094b HEAD` over the production files shows 5 files, +1,507/−84:
  `shared_resources_planning.py`, `model_construction_helpers.py`, `network.py`, `admm_parameters.py`,
  `solver_parameters.py`. **The six were therefore not evaluated on the final code.**
- σ_Q was measured under the C3 calibration (S47 record note: "not re-measured under the baseline").

**Verdict.** "Determinately worse" holds for the 8 only. The six are worse beyond the search resolution, not under the
rule. Expert interpretation (iv) of Addendum 78 is confirmed by the records. The E.3 text (l. 2588) already says this
correctly, and its CONFIRM (l. 2590) can be closed.

**Minimal qualifier (recommended):** "…with every evaluated unit neighbour of the plan without storage worse,
determinately for the eight re-evaluated under the certification rule (Table~\ref{tab:appE_neighbours}) and beyond the
search resolution for the other six."
- Shorter alternative: "…with every evaluated unit neighbour of the plan without storage worse at the search
  resolution, and determinately worse under the certification rule for the eight re-evaluated (Table E.22)."

### §2.2.7 Determinacy (l. 743): "the bands, gaps and slacks of every evaluation used in a reported comparison are tabulated in Appendix E"

**Evaluations used in reported comparisons that are absent from Table E.21, or whose band is not the one tabulated:**

| Comparison (where) | Evaluations | In E.21? | Band / gap / slack tabulated? |
|---|---|---|---|
| 3 × 3 unit vs x = 0 (Table 13, l. 1611–1630; "determinate") | `t11_3x3_x0` `f6e9cd53…` (cycle 72), `t11_3x3_unit` `c82522f4…`, continuation `25b92ae0…` | **No** (none of the three eval keys is in T2) | **No.** Table 13 gives a "resolution of the value" of 18.0 k€ (17,964.95 €), not a band/gap/slack. The Appendix E intro (l. 2388) itself says they are reported in Table 13. |
| Coordination vs static arrangements (Table 12, Fig. 8; "1,264 times the larger band (71.9 k€)") | the coordinated row = E.21 row 1; the six T6 arm solves | arms: no (not ADMM evaluations; Table E.23) | **Partly.** The arms' spreads are in E.23. The coordinated "reproducibility band" of 71.9 k€ (frozen `benchmark.coordinated_reproducibility_band`) is in the E.23 note, and it is not E.21 row 1's band of 4.2 k€. No gaps or slacks apply to the arms. |
| Discount rate 0/5/8 % (Table 10; thresholds 15.9 / 13.1 / 13.1 k€) | E.21 rows 1 and 2, re-weighted | yes (at 2 %) | **Not at the other rates.** E.21 bands are at 2 % (4.2 and 4.4 k€). The 0 % threshold of 15.9 k€ implies a re-weighted band that is not tabulated. |
| x = 0 search-only neighbours (§4.1 l. 1367, E.3, §4.8) | 6 search evaluations | **No** | No. Search resolution only (see above). |
| F2 search-only neighbours (l. 1475: "seventeen … evaluated", "thirteen re-evaluated") | 4 search evaluations (2035: n5 0.25/0.5 + n7 1/3, 1/4; n5 0.25/1 + n7 1/3, 1/4) | **No** | No. These are aggregate statements only. |

**All other reported comparisons are covered.** This includes the year-ladder 2035 − 2030 comparison (rows 24–25),
which is not in E.20 but whose evaluations are in E.21. The 58 rows of E.20 reference only E.21 evaluations.

**Minimal qualifier (recommended):** "…and the bands, gaps and slacks of every single-scenario evaluation used in a
reported comparison are tabulated in~\ref{app:results} (the multi-scenario evaluations, stopped by the production exit,
in Table~\ref{tab:res_multiscenario})."
- This covers the 3 × 3 gap.
- For the other rows, the expert decides whether to add "at the 2 % rate" (discount) and "the static arrangements'
  spreads in Table~\ref{tab:appE_benchmark}".
- The search-only neighbours are evaluations "used" only in aggregate "no improvement beyond the search resolution"
  statements. If the expert reads them as reported comparisons, the clause needs "certified" ("every certified
  evaluation…").

**Adjacent sentence, same paragraph (l. 743).** "every cell it proposed was re-evaluated under this rule before being
reported" does not hold literally for the 6 + 4 search-only neighbours. They are reported only as counts ("all
fourteen … evaluated", "seventeen … evaluated"). This is flagged for the expert and was not in the task's scope.

---

## Not confirmed

- **`d_c52e1670` at 150, the individual s⁺ and s⁻.**
  - Not recorded.
  - They are pinned within 1.9 × 10⁻¹⁰ MWh by the read S and the relaxed floor. That floor is IPOPT's documented
    default `bound_relax_factor` 10⁻⁸. It is not printed in the options list because it was not user-set.
  - It is consistent with the logged "Variable bound violation" ≤ 9.9992 × 10⁻⁹ at all 24 solves.
- **The derived cross-check rests on uncommitted files.** These are `ess_schedule_per_cycle.jsonl` and 24 IPOPT logs,
  hash-recorded in the manifest. The read value rests on a committed record.
- **The positive control is inconclusive** because of precision (see above). No other block with a nonzero closure
  difference was available to test the derivation.
- **E^Av at cycle 150.** The relative values use the published E^Av at the terminal cycle 198. No per-cycle E^Av record
  was searched. This affects only the normaliser, not the sign or the floor test.
- **The 12 uncertified E.21 rows and the T6 arms were not re-read here.** Their status is cited from W169.
- **The S47 search evaluations' configuration beyond the code head** (for example, the ageing calibration in force) was
  not checked. Only the head and the σ_Q note were read.
- **"Table 13" and "Table 10"/"Table 12"** are the task's numbering. I did not compile the manuscript, and I refer to
  the tables by label and line.
- **Scope of the search for a cycle-150 record.** I looked in the cell's eval directory (all files listed in W169's
  `records_not_carrying`, plus `creep_diagnostic_per_cycle.jsonl`, `ess_schedule_per_cycle.jsonl`,
  `network_ipopt_solve_records.jsonl` and the IPOPT logs) and in the W142 v6-from-records certificate. Pickles were not
  opened. Per W169, `certified_models.pkl` holds cycle 198.
