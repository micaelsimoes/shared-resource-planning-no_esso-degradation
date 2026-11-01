# P5.15 W181b: the five round-7 sentences re-rated, and the §3.5 ageing citations, at Overleaf `cd57efc` (Addendum 80, W181 items (c) and (d))

**Worker, 2026-10-16.** Zero solves, zero model builds, zero pickle loads. This is the short check of Addendum 80
ruling 5, not a re-audit. The clone, production code, `manuscript_review/`, the frozen JSON and all earlier W outputs
were only read.

**Manuscript.** `manuscript/6a67305f25e8348fb71380c3` at clone HEAD `cd57efcb54399cb652a0ac2dcd64e5d3a78d89fc`
(`cd57efc`). Its parent is `e5322f8`, so it is the only commit after it.
- `main.tex`: sha256 `ff7b045e…`. It equals the HEAD blob and `manuscript_review/main_e5322f8_round7.tex`.
- Letter: sha256 `06cc2460…`. It equals the HEAD blob and `response_to_reviewers_v8.tex`. The letter at `e5322f8`
  equals v7.
- `bibliography.bib`: sha256 `8db46aea…`. It is unchanged from `e5322f8` to HEAD.
- All three are unmodified in the clone. Only `.DS_Store` is untracked there.
- The full identity check (clone diff = the expert's diffs; Appendix E bitwise) is W181a's scope (`4bcc2463`) and is
  not repeated here.

**Repository.** HEAD `4bcc2463` at the run, branch `feature/derivative-free-planning`.

**Objective convention.** Q = `gross_operational_cost`, settlement excluded. Value = Q(0) − Q(unit). No Q is computed
here; the only number used is the recorded Table 13 difference.

---

## Blocked on Planner

None.

The ageing citations remain the **author's action**, as Addendum 80 ruling 3 already records. Nothing in this check
changes that.

---

## Changed

- **Commit:** the one that adds this report. Its hash is in the Worker report to the Planner, because a commit cannot
  carry its own hash. It contains:
  - the new harness `p515_s53_w181b_sentences_citations.py` (sha256 `b8a1fab0…`);
  - this report;
  - the outputs in `data/SRP1/Results/P515S53/w181b_sentences_citations/`. The directory was fresh, and the harness
    refuses to start unless it holds only `launch.log`.

  | file | sha256 |
  |---|---|
  | `w181b_sentences_citations.json` | `5ea3f8d4e91cc7e2491802fdfbdb36764a030561c5e695c5984df359fc9bbc9a` |
  | `manifest_sha256.json` (56 inputs, 0 uncommitted; also hashes the output JSON and the script) | `49f8c255c36317e607843543bb3583e251c4e6e8ec3ce66992a8b9d366c42508` |
  | `launch.log` (stdout + stderr, attached, exit 0) | `9915a28d1f3e692f49a431c88f738e172cb98e96bfafc0da43d7be65637513fc` |

- **Guards:**
  - `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0, with all four
    counters at 0.
  - `pickle.load` and `pickle.loads` were blocked and called 0 times.
  - No production module was imported (asserted). Production source and the W116 harness were read as text and parsed
    with `ast`, never executed.
  - All JSON was written through `gate_result_io` (the shared writer).
  - The harness refuses when another copy of itself is running.
  - It also asserts, before reading anything, that every source the order names exists (capture-path check).
  - The frozen JSON `590088fe…` was unchanged at the end.
- **Prior evidence verified before use** (sha256 = report value = manifest entry; all committed):
  - W179c JSON `417c270f…`
  - W180b JSON `bd6343ed…`
  - W180c JSON `88e1d1d9…`
  - the frozen JSON `590088fe…`
- **Trial runs:** three, into the scratchpad only (`w181b_trial_1`–`_3`).
  - Trial 1 refused at the single-copy check, because the check matched the shell wrapper's command line. It was fixed
    to match interpreter processes only.
  - Trial 2 failed on the nesting of W180c's `rows` key, and the read was fixed.
  - Between those two runs I also corrected one expected line: `inactive_flag` is l. 1121, not l. 1120. That was the
    harness's own expectation, not a manuscript citation.
  - Trial 3 gave the same verdicts as the committed run.
- **Nothing else changed.** `manuscript_review/`, the clone, production code, frozen files, specs, earlier W-scripts and
  outputs, TASKS.md, REVISION_CONTEXT.md and the brief are untouched.

---

## Found

### Predictions against outcomes

| Prediction (expert, Addendum 80) | Outcome |
|---|---|
| (c) all five MATCH and R22-b closes | **HELD.** S1–S5 are all MATCH, and R22-b is **closed** |
| (d) citations absent, or present with three keys | **HELD (absent).** None of the three statements carries a `\cite`. The only key in §3.5 is `nrel_ess_costs`, and it resolves |
| (task) `[CONFIRM` count 0 | **HELD:** 0 in main.tex and 0 in the letter, comment lines included |
| (task) `[AUTHOR` count 1 | **HELD:** 1 in main.tex (l. 1307) and 0 in the letter |

### (c) The five sentences

**S1. §2.3.3, l. 969: "Without storage the closure row is deactivated and its slacks are fixed at zero."**
**MATCH.**
- The sentence occurs once, and "not built" occurs 0 times.
- At repo HEAD, every cited line reads as expected and sits in the expected function (checked with `ast`):
  - `network.py:511` is `model.sess_soc_final = pe.Constraint(...)`. It is a direct statement of `_build_model`, with
    no `if` or loop around it, so the row is **built** for every unit.
  - `model_construction_helpers.py:1143` is `entry.fix(0.0)`.
    - It is in the body of `if inactive:` inside the loop over `_SHARED_ESS_OPERATIONAL_VARIABLES`.
    - Both closure slacks are in that tuple.
  - `:1166` is `slack_ub = 0.0 if inactive else …`.
    - It is inside the loop over the two closure slacks.
    - `setlb(0.0)` (`:1168`) and `setub(slack_ub)` (`:1169`) follow in the same loop. For an inactive unit that gives
      bounds [0, 0].
  - `:1176` is `entry.deactivate()`.
    - It is in the body of `if inactive:` inside the loop over `_SHARED_ESS_OPERATIONAL_CONSTRAINTS`.
    - That tuple contains `'sess_soc_final'`, the closure row.
  - All three lines are inside `configure_shared_ess_operational_state`. Its inactive flag (`:1121`) is
    `shared_ess_capacity_is_inactive` (`:1069`, `abs(S) <= tol or abs(E) <= tol`).
  - The tolerance is `SHARED_ESS_ZERO_CAPACITY_TOLERANCE = 1e-10` in `definitions.py`.
- **The path applies to the x = 0 cells.**
  - `network_data._update_model_with_candidate_solution` calls `configure_shared_ess_operational_state` with
    |candidate s, e| / s_base, on both the TN branch and the DN branch.
  - The five x = 0 cells have candidate `8435c718…`, with [0, 0] at nodes 5, 7 and 9 (W179c `D_x0`). So every unit is
    inactive.
- **The same code at each run head.** For each x = 0 run head I read `git show <head>:` of the four files. The same
  lines hold at the same numbers, and `git diff --stat <head> HEAD` over `network.py`, `model_construction_helpers.py`,
  `definitions.py` and `network_data.py` is empty. The runs and their heads:

  | run | head |
  |---|---|
  | `ref:7aa017f0` | `aad37370` |
  | `h_aa8a76d7` | `9d2f458f` |
  | `h_x0_m175` | `0e1d7c0e` |
  | `h_50dea31c` | `3740b476` |
  | `t11_3x3_x0` | `8a8ef150` |

**S2. Table 13 (`tab:res_multiscenario`), now l. 1621: "(cycle 72 for the plan without storage, 69 for the unit) &
236.7 k€ & recorded". MATCH.**
- The line is l. 1621, not 1619: the two-line `[AUTHOR]` comment at l. 1307–1308 shifts it by 2. The old
  "(cycle 72) &" occurs 0 times.
- **Source:** `data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json` (sha256
  `587d3f1a…`, committed, `git_head_at_run` `8a8ef150`).

  | cell | status | `cycles_run` | `certification_cycle` | Q |
  |---|---|---|---|---|
  | `x0` | certified | 72 | 72 | 842,832,534.76 |
  | `n7_4h_e1` | certified | 69 | 69 | 842,595,839.51 |

- `value_and_R.value_eur` = 236,695.249 = Q(x0) − Q(unit), which rounds to **236.7 k€**, unchanged.
- **Instance.** The W169 eval keys name the campaign's eval dirs:
  - `f6e9cd53…` is `_x0`, candidate `8435c718…`, all zero;
  - `c82522f4…` is `_n7_4h_e1`, candidate `db77e154…`, node 7 [0.25, 1].
- **Frozen JSON.** `three_by_three.rows[0]` holds the same value, and its label still reads "… at certification
  (cycle 72)". As STEP6 round 7 A.2 says, it is frozen and was not edited.

**S3. Appendix E.4, l. 2624: "a re-evaluation of each distribution network at the transmission network's interface
voltage, with its decisions held at the first phase and its voltage, thermal and no-reverse-flow rows switched off".
MATCH.**
- The sentence occurs once. The old "a re-solve … with its no-reverse-flow rows switched off" occurs 0 times.
- **Harness, `p515_s53_w116_benchmark_nrf.stage_nrf_arm`:**
  - phase A opens at l. 565 and solves at l. 566 (`UB.run_operational_planning_uncoordinated`);
  - `if consistency:` is at l. 614;
  - the TN interface voltage is taken at l. 615: `UB.interface_voltage_mismatch(…tso…, …dso…)` gives `v_actual_dn_pu`,
    which is the TSO `expected_interface_vmag` converted to the DN base;
  - phase B opens at l. 616;
  - l. 617 calls `UB.reevaluate_dso_at_actual_voltage(planning, arm_out['models']['dso'], …)`. Its input is the
    phase-A DSO solution, and no TSO solve takes place in phase B.
- **Production benchmark, `uncoordinated_benchmark.py`:**
  - `reevaluate_dso_at_actual_voltage` (l. 1580) takes `arm_block = dso_models[…]` (l. 1591) and builds the block with
    `build_consistency_reevaluation_block` (l. 1592).
  - That function (l. 1411) works on a clone (l. 1427).
    - It loops over `DSO_DECISION_VAR_FAMILIES` (l. 1434) and fixes each at the arm value (l. 1442).
    - It fixes the non-reference generators' P/Q (l. 1452). This is **"decisions held at the first phase"**.
    - It deactivates `VOLTAGE_LIMIT_ROWS + THERMAL_LIMIT_ROWS` (loop at l. 1487, deactivate at l. 1492). Those are
      `voltage_magnitude_lower/upper_cons` (l. 235) and `branch_flow_limit` / `_ji` (l. 236).
    - It deactivates `uncoord_no_reverse_flow` (l. 212; l. 1495, 1498).
  - The families the sentence names equal the five families the builder deactivates (checked set against set).
- **Run records.** I read the six committed NRF arm records, `nrf_arm_{passive,price_taker}_{cold,perturbed,
  warm_from_certified}_r2`. For each:
  - all 36 DSO blocks of phase B have `relaxed_row_families` equal to exactly those five, and `n_decisions_fixed` > 0
    (1,024 for passive, 4,096 for price-taker);
  - the recorded `module_sha256` of `uncoordinated_benchmark.py` and of the W116 script at the run (git head `5d923082`)
    equal the HEAD bytes read here.
- **What the builder also does, which the sentence omits.** None of this contradicts the sentence:
  - it fixes the `DSO_REEVALUATION_ZERO_SLACKS` at 0;
  - it relaxes the node-voltage and reference-generator bounds, recording the originals;
  - it leaves the reference generator (the TN supply) free, since it carries the re-evaluated interface flow;
  - it deactivates rows left without a free variable.

  W180a noted the same omissions.

**S4. Table E.21 note, l. 2583: "Evaluations~1 and~2 continue earlier runs, and evaluations~40 and~41 are re-decided
from the records of earlier runs; their first residual passes are not tabulated". MATCH.**
- The note occurs once, and the round-6 note occurs 0 times.
- The E.21 block has 61 rows, identical row for row to `e5322f8`. The note is its only changed line.
- **W163 N49.** The reference continuations are {x0, n7_4h_e1, c_star}, each replayed bitwise (G19).
- **Rows 1 and 2.** In W180b these are `is_reference_continuation_W163_N49` = true, with `w101_cell` x0 and n7_4h_e1.
  Their frozen eval keys `d110bd1a…` and `3f084f2f…` equal the W101 summary's x0 and n7_4h_e1 keys. So "continue
  earlier runs" is correct.
- **Rows 40 and 41** (`ref:5ca4f86c`, `ref:e28de4ac`):
  - they are not N49 continuations, and their eval keys are not among N49's;
  - each was issued under rule version 2 (W118 r2, `campaign_s53_w118_resettle_r2_f2_incumbent` /
    `_f2_challenger`);
  - each carries a v6 decision "on record (W142 item 1)" (`f2_incumbent` / `f2_challenger`), equal to the replay arm
    AB;
  - each prints Rule "record".

  So "re-decided from the records of earlier runs" is correct. v6 leaves both uncertified, at 281 and 261.
- $k_0$ is printed "---" for all four rows, which is consistent with "first residual passes are not tabulated".

**S5. Letter R2.2, l. 125: "… and the cycles of every single-scenario evaluation behind a reported comparison in
Appendix~E." MATCH; R22-b closed.**
- The v7 wording occurs 0 times. R2.2's own two classes are still stated: "the production exit during the search, or,
  for the evaluations behind the reported comparisons, when the certification rule …".
- **R22-b under W180c's definition.**
  - W180c's residue (JSON row R22-b: "still (narrowed)", L) was the 3 × 3 pair: behind a reported comparison but in
    neither class.
  - The pair is multi-scenario, so "single-scenario" excludes it. The Appendix E introduction sends the multi-scenario
    evaluations to Table 13.
  - Every E.21 row prints a numeric $k^*$/end (61 of 61).
  - W180c's secondary reading, the six search-only neighbours, stopped at the production exit (E.3 sentence present).
    So they fall in R2.2's search class, and Addendum 80 ruling 4 accepted their bands as L.
  - Hence **closed**.
- **Pointers v7 → v8.**
  - The changed lines are exactly {1, 125}: the header comment and R2.2.
  - The pointer multiset is equal over the whole letter: 177 tokens, with 0 only in v8 and 0 only in v7. It is also
    equal on l. 125 alone.
  - So no pointer changed.
  - Method: a regex over label word + number and `\ref`/`\eqref`/`\cite`. This is **not** W180c's 165-pointer mapping;
    the two counts are not comparable.

### (d) Ageing citations

- **§3.5** "Shared Energy Storage Parameters" runs from l. 1264 to 1309. It has exactly one `\cite` key:
  `nrel_ess_costs` (l. 1268, on "the NREL utility-scale battery category"). The key **resolves** in the clone's
  `bibliography.bib` (`@misc{nrel_ess_costs,`, l. 367).
  - The bibliography is `\bibliographystyle{elsarticle-num}` + `\bibliography{bibliography}`, with 48 entries.
  - There is no `thebibliography` in main.tex.
- **The three statements**, each located in the typeset prose of §3.5. Each has **no citation**:

  | statement | sentences found | with a `\cite` |
  |---|---|---|
  | (i) "10{,}000 cycles at 80\,\% depth of discharge" | 1 | 0 |
  | (ii) "8{,}000 full cycles" (EVE MB31 datasheet) | 1 | 0 |
  | (iii) $\phi^{\text{Cal}} = 0.985$ per year | 2 (prose, and the table rows) | 0 |

  - Table `tab:ageing_calibrations` (label l. 1305) carries no `\cite` in its caption or rows.
  - **No bib key** matches eve / mb31 / calendar / lfp / ageing / aging / degrad.
  - "MB31" and "EVE" occur in main.tex only in the comment at l. 1307.
- **The restored `% [AUTHOR]` comment** is at l. 1307–1308. It sits directly after `\end{table}` (l. 1306) of
  `tab:ageing_calibrations`, and before `\subsection{…Planning and Certification Settings}` (l. 1310):

  ```
  % [AUTHOR] datasheet citations for the two cycle-life statements (10,000 cycles at 0.80 DoD; 8,000 full cycles, EVE MB31) and for the
  % calendar retention 0.985/yr are still to be added before submission (round 4 item 6; Addendum 80).
  ```

- **Counts**, comment lines included:

  | file | `[CONFIRM` | `[AUTHOR` |
  |---|---|---|
  | main.tex | 0 | 1 |
  | letter | 0 | 0 |

### Negative controls (12 of 12 refused, in memory)

| control | result |
|---|---|
| NC1: S1 sentence altered ("fixed at one") | MISMATCH |
| NC2: S1 code line 1143 → 1144 | MISMATCH |
| NC3: Table 13 cycles swapped (69 for x = 0, 72 for the unit) | MISMATCH |
| NC4: `campaign_results` n7_4h_e1 `cycles_run` → 72 | MISMATCH |
| NC5: "thermal" dropped from E.4 | MISMATCH (families ≠ code) |
| NC6: W116 phase-B line 617 → 618 | MISMATCH |
| NC7: E.21 note with evaluations 2 and 40 swapped | MISMATCH |
| NC8: round-6 E.21 note | MISMATCH |
| NC9: the v7 letter | MISMATCH, R22-b still |
| NC10: R2.2 "Section~4.7" → "4.8" | MISMATCH (pointer multiset differs) |
| NC11: a `\cite` key not in the bib | reported unresolved |
| NC12: a `\cite` planted in the 10,000-cycle sentence | detected as present |

---

## Not confirmed

- **Typeset output.** Nothing was compiled. "Table 13" and "Table 6" are the task's numbers, and they rest on the
  labels `tab:res_multiscenario` and `tab:ageing_calibrations`. This harness did not re-derive the table numbering;
  W180c's static numbering gives `tab:res_multiscenario` = 13.
- **That the solved NLP omits the deactivated row and the fixed slacks.** That is Pyomo NL-writer behaviour, and it
  was not tested: no model was built. The S1 wording rests on the code alone, which is what Addendum 80 ruling 2 asks.
- **Run-time execution of phase B.** S3 rests on source read at HEAD, the byte-equality of the module hashes recorded
  at the run, and the per-block build records. The run itself was not replayed.
- **The letter pointers' semantics.** Only multiset equality v7 → v8 was checked. That a pointer refers to the right
  place is W180c's 165-pointer mapping at `e5322f8`, which was not re-run. The two changed lines contain no changed
  pointer.
- **Citation search scope.** I searched §3.5 (l. 1264–1309) of main.tex, every comment line of main.tex for
  `[AUTHOR` / `[CONFIRM`, the letter for the same, and the clone's `bibliography.bib`. I did not search other files of
  the Overleaf project (e.g. `section2_expert_draft.tex`, `cover_letter.tex`) or citations of the cycle-life
  calibration elsewhere in main.tex (e.g. §2 at `eq:cycle_life_calibration`).
