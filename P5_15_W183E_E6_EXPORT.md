# P5.15 W183e: E6 data export (Addendum 83 ruling 4; STEP7 plan sections 7 and 11)

**Worker, 2026-10-25.** Zero solves, zero model builds, and exactly one guarded load of one pickle: row 2's
`certified_models.pkl`, sha256 `d0a3d512…`. I edited no production code, harness, spec, frozen file, record,
TASKS.md, REVISION_CONTEXT.md or the brief, and touched neither the clone nor `manuscript_review/`. I loaded no other
pickle.

- Repo HEAD at the run: `5c1e6f23` (the harness commit), on `feature/derivative-free-planning`.
- The only tracked change in the working tree was `M TASKS.md`. It was there before this task and is not mine.

**Objective convention, used for every value below.** Q is the certified `gross_operational_cost`: gross operational
cost with the contracted interface settlement excluded. There is one market scenario, so the deviation settlement and
the commitment premium are both 0. The terminal salvage is 0 on both rows. Block values are block-weighted: year count
× day count × discount annualisation.

---

## Blocked on Planner

None. The export is complete and committed. Three points need a decision only if the export's scope should change:

1. **DN flexibility activation (panel b) is reported as a cost in €, not a volume in MWh.** The only per-block measure
   for both rows in committed records is the DSO `flexibility_cost_internal`.
   - Row 1's activation volume (`flex_p_up/down`) exists only in row 1's `certified_models.pkl` (`99ab1070…`). This
     task did not authorise reading it.
   - **Options:** (a) accept the cost measure (the current export); (b) authorise one more guarded read of row 1's
     pickle to export the MWh volume for both rows.
   - **Recommendation:** (a). The plan names `component_levels_terminal.json` as the panel-b input.
   - As a labelled secondary, the export also carries the TN-side signed interface delta (`interface_delta_p`). It
     comes from the committed settlement detail of both rows. It is an interface quantity, not the DN-internal
     activation volume.
2. **The export includes more than the declared day.** It also carries node 7's hourly TSO/DSO copy schedule and SoC
   for all 12 blocks of row 2 (`supporting_node7_all_blocks_row2`, labelled supporting). It comes from the same single
   read and avoids a second read if a different day is ever wanted. Please confirm it may stay, or I will drop it in
   a v2 of the export.
3. **The pickle was unpickled three times in total, once in each of three separate runs.** In every run the guard was
   armed, with exactly one allowed load.
   - Trial 1 (scratchpad `w183e_trial1`) loaded the file, then failed on a dict-key type (`KeyError: '2025'`: the
     years are not keyed by the string `'2025'`). I fixed the lookup.
   - Trial 2 (scratchpad `w183e_trial2`) completed.
   - The real run is the third load.
   - Two `--dry-run` passes (`w183e_dry1`, `w183e_dry2`) loaded nothing.
   - I read "a zero-solve read" as permitting these guarded trial reads into the scratchpad. If it permitted only one
     read in total, the two trial loads exceeded it.

---

## Changed

- **`5c1e6f23`**: the harness `p515_s53_w183e_e6_export.py`, committed before the real run. The run records the
  script as `committed`, with sha256 `7f121400…` (equal to the blob at `5c1e6f23`); see `e6_export.json` `script`.
- **Export commit** (hash in the final message): this report, plus `data/SRP1/Results/P515S55/e6_export/`, added with
  `git add -f` by filename. The directory was new, and each file was written once:

  | File | sha256 |
  |---|---|
  | `e6_export.json` (268,707 B) | `4875075728db185e2093e32e90332851d77f7d2195c576886fdafb63f83b25e0` |
  | `manifest_sha256.json` (23 inputs, with git status; output; script) | `c890dc70aac510546df9645b67eaf1b7b815a24e3ca82131a147f0580dff5fa2` |
  | `launch.log` (stdout + stderr, attached run, exit 0) | `e137f38edae978fa36953feb982d6822a48392f5e789742bf46b0b69e3c2338d` |
  | `manifest_post_run_sha256.json` (over the three files above; `--post-run` exit 0) | `412243fa59286534390bd12ff6778f3961a819264f80cf214a3419b57874a866` |

- **Inputs.** 22 of the 23 inputs are committed. The 23rd is the pickle, which is untracked and pinned by hash. The
  two stride JSONL inputs were verified in three ways:
  - working-file sha256 = pin = blob at `741e8ddf` = blob at HEAD: row 1 `0012cef5…`, row 2 `47de0cb1…`;
  - `741e8ddf` is an ancestor of HEAD;
  - the ESSO capture equals W182f's pin, `ac8bc61e…`.
- **Pickle verification before loading.** The file's sha256 equals the task pin and equals `post_certification.json`
  `persisted_models.sha256`: `d0a3d512fa58f41053befdb784bb842ad57ec0d71c989aef776a0aa668e6030d`. The path and size
  (164,685,741 B) also match. The gate then re-hashed the very bytes it unpickled and got the same digest.
- **Command.**

  ```
  mkdir -p data/SRP1/Results/P515S55/e6_export && \
    python -u p515_s53_w183e_e6_export.py > data/SRP1/Results/P515S55/e6_export/launch.log 2>&1
  ```

  followed by `--post-run`. The interpreter was the canonical one. The wall time was 8 s.

---

## Found

### Declared day: **2025 Spring** (illustrative day, not a typical-day estimate)

**Rule (plan §7).** Rank the (year, day) blocks by the value contribution c = Q_row1(x = 0) − Q_row2(x). The value of
a block is the sum, over its four network blocks (TSO and DSO 5/7/9), of these `component_levels_terminal.json`
weighted components:

- generation;
- flexibility_cost_internal;
- load curtailment;
- RES curtailment penalty;
- ESS usage;
- detector penalties.

**The decomposition is checked against Q, in two ways.**
- The sum over the 48 blocks equals `gross_operational_cost`. The residual is 2.4e-7 € for row 1 and 3.6e-7 € for
  row 2, a relative residual of ≤ 5.5e-16.
- The 12 contributions sum to 253,539.62 € = Q(0) − Q(x).

**Resolution.**
- **bar:** for each block, the sum of the two rows' block ranges over their certification windows (row 1 [150, 181],
  row 2 [140, 172]), taken from `recourse_blocks_all.jsonl`. It is a local stopping-slack measure only.
- **Next rank determinate:** the margin to the next rank is larger than the sum of the two blocks' bars.

| Rank | Block | Weight | c = Q(0) − Q(x) (€) | One-day c, unweighted (€) [rank] | bar (€) | Next rank determinate |
|---:|---|---:|---:|---:|---:|---|
| 1 | **2025 Spring** | 460 | **40,181.22** | 87.35 [1] | 56.10 | yes |
| 2 | 2025 Summer | 455 | 28,010.61 | 61.56 [3] | 47.27 | no |
| 3 | 2035 Spring | 377.36 | 26,072.72 | 69.09 [2] | 8,588.75 | no |
| 4 | 2030 Spring | 416.64 | 23,246.73 | 55.80 [5] | 103.15 | yes |
| 5 | 2030 Summer | 412.11 | 21,806.36 | 52.91 [6] | 222.03 | yes |
| 6 | 2030 Autumn | 412.11 | 21,405.52 | 51.94 [7] | 70.28 | yes |
| 7 | 2035 Summer | 373.26 | 20,964.15 | 56.17 [4] | 116.77 | yes |
| 8 | 2025 Autumn | 455 | 20,085.27 | 44.14 [8] | 13.32 | yes |
| 9 | 2025 Winter | 455 | 16,091.66 | 35.37 [9] | 10.37 | yes |
| 10 | 2035 Autumn | 373.26 | 13,082.00 | 35.05 [10] | 29.51 | yes |
| 11 | 2030 Winter | 412.11 | 12,622.69 | 30.63 [11] | 61.25 | yes |
| 12 | 2035 Winter | 373.26 | 9,970.69 | 26.71 [12] | 20.13 | — |

- **The selection is determinate.** The margin of rank 1 over rank 2 is 12,170.61 €, against a combined bar of
  103.37 € (118×). 2025 Spring is also rank 1 by the unweighted one-day contribution.
- 2025 Spring carries 15.8 % of the total value.
- The ranks of 2025 Summer, 2035 Spring and 2030 Spring among themselves are not determinate by this bar (see
  Unexpected findings).

### Export contents (`e6_export.json`)

**(a) Declared day, node 7, 2025 Spring.**
- Every hour carries:
  - **ESSO:** pch, pdch and pnet (capture), plus the stride p and q;
  - **TSO and DSO copies:** pch, pdch, pnet, qnet, SoC and expected p/q (pickle), plus the stride p and q;
  - **consensus z:** p and q;
  - **DERIVED SoC:** from the ESSO pch/pdch and from z;
  - **market price:** as context.
- **SoC limits.** The record gives E^Av = 0.880458 MWh, so the limits are 0.088046 and 0.792412 MWh. The pickle's
  `shared_es_e_rated_fixed` × 100 gives 0.880457 MWh on both copies.
- **The SoC touches both limits.** The minimum is 0.0880447 MWh, which is 1.0e-6 MWh below the lower limit. The
  maximum is 0.7924119 MWh, which is 1.0e-6 MWh above the upper limit. Both are solver-tolerance overshoots.
- **Day energy (TSO copy):** 0.7550 MWh charged and 0.7031 MWh discharged.
- **End of day:** the SoC returns to 0.5 E within 2.6e-11 MWh. The day-end slacks are about −1e-6 MWh.

| h | Price (€/MWh) | ESSO pnet | TSO pnet | DSO pnet | z p | TSO SoC | DSO SoC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 116.67 | −0.03442 | −0.03444 | −0.03443 | −0.03443 | 0.40436 | 0.40436 |
| 2 | 117.73 | −0.05396 | −0.05398 | −0.05398 | −0.05397 | 0.34813 | 0.34814 |
| 3–6 | 119–128 | 0 | 0 | 0 | 0 | 0.34813 | 0.34813 |
| 7 | 134.88 | −0.24966 | −0.24968 | −0.24967 | −0.24967 | 0.08804 | 0.08805 |
| 8–13 | 116 → 16 | 0 | 0 | 0 | 0 | 0.08804 | 0.08804 |
| 14 | 12.70 | 0.14764 | 0.14766 | 0.14766 | 0.14765 | 0.23127 | 0.23127 |
| 15 | 3.26 | 0.25000 | 0.25000 | 0.25000 | 0.25000 | 0.47377 | 0.47377 |
| 16 | 6.99 | 0.25000 | 0.25000 | 0.25000 | 0.25000 | 0.71627 | 0.71627 |
| 17 | 13.60 | 0.07848 | 0.07850 | 0.07850 | 0.07849 | 0.79241 | 0.79241 |
| 18–20 | 42 → 114 | 0 | 0 | 0 | 0 | 0.79241 | 0.79241 |
| 21 | 142.54 | −0.11503 | −0.11504 | −0.11504 | −0.11504 | 0.67258 | 0.67258 |
| 22 | 134.53 | 0.02883 | 0.02884 | 0.02884 | 0.02884 | 0.70055 | 0.70055 |
| 23 | 125.50 | 0 | 0 | 0 | 0 | 0.70055 | 0.70055 |
| 24 | 158.09 | −0.24991 | −0.24991 | −0.24991 | −0.24991 | 0.44023 | 0.44023 |

Units are MW and MWh. pnet is positive when charging. The rows marked as ranges have |pnet| < 1e-5 MW; the full
24-hour values are in the JSON.

**(b) Change per block against x = 0** (row 2 − row 1; weighted €; negative = saving; the full weighted and
unweighted tables are in the JSON).

| Block | TN conventional cost (TSO generation) | DN flexibility cost, 5 / 7 / 9 / total | ΔQ |
|---|---:|---|---:|
| 2025 Spring | −39,756.25 | −37.97 / −350.67 / −34.50 / −423.13 | −40,181.22 |
| 2025 Summer | −27,849.99 | −65.09 / −4.21 / −89.51 / −158.80 | −28,010.61 |
| 2025 Autumn | −20,083.45 | 0 / 0 / 0 / 0 | −20,085.27 |
| 2025 Winter | −5,995.94 | −2,237.49 / −5,108.35 / −2,748.07 / −10,093.90 | −16,091.66 |
| 2030 Spring | −16,290.58 | −2,441.71 / −794.27 / −3,718.51 / −6,954.48 | −23,246.73 |
| 2030 Summer | −15,584.45 | −1,381.28 / −3,584.28 / −1,254.70 / −6,220.26 | −21,806.36 |
| 2030 Autumn | +10,520.40 | −8,486.90 / −10,282.69 / −13,154.68 / −31,924.27 | −21,405.52 |
| 2030 Winter | −5,809.68 | −1,499.86 / −3,760.28 / −1,551.21 / −6,811.36 | −12,622.69 |
| 2035 Spring | −19,168.93 | −1,528.61 / −233.11 / −5,140.56 / −6,902.28 | −26,072.72 |
| 2035 Summer | −14,977.82 | −1,245.96 / −1,398.84 / −3,340.03 / −5,984.83 | −20,964.15 |
| 2035 Autumn | −4,258.74 | −3,708.26 / −2,663.72 / −2,449.79 / −8,821.77 | −13,082.00 |
| 2035 Winter | +316.84 | −2,603.43 / −4,249.34 / −3,433.28 / −10,286.05 | −9,970.69 |
| **Horizon** | **−158,938.59** | total **−94,581.13** | **−253,539.62** |

- The only other non-zero component change is the detector penalty, −1.49 to −1.84 € per block.
- The components sum to ΔQ within 1.6e-8 € in every block.

**(c) E^Av per representative year and node.** Row 2, node 7: E_rated = 1 MWh, floor 0.70 MWh, minimum_soh 0.70 from
the record (= declared).

| Year | E^Av (MWh) | SoH | Margin above floor (MWh) | EFC/day |
|---|---:|---:|---:|---:|
| 2025 | 0.880458 | 0.8805 | +0.180458 | 1.0165 |
| 2030 | 0.779817 | 0.7798 | +0.079817 | 0.9000 |
| 2035 | 0.699999990 | 0.7000 | −1.0e-8 (floor active: row (0, 2)) | 0.6367 |

- S^Av is 0.25 MVA in every year.
- Nodes 5 and 9, and every node of row 1, have zero capacity.

### Cross-checks (maximum absolute difference; none gated)

| Quantity | Sources | n | Max diff |
|---|---|---:|---:|
| TSO-copy P, Q | stride x.tso vs pickle `expected_shared_ess_p/q` × 100, all 12 blocks | 288 + 288 | **0.0** (bitwise) |
| DSO-copy P, Q | stride x.dso vs pickle, all 12 blocks | 288 + 288 | **0.0** (bitwise) |
| expected vs scenario copy | pickle expected_p vs pnet (TSO / DSO) | 288 each | 6.2e-18 / 0.0 MW |
| SoC, internal | pickle SoC vs recursion from the pickle's own pch/pdch (0.97/0.96, start 0.5 E) | TSO / DSO, 288 each | 1.6e-11 / 4.4e-16 MWh |
| **SoC vs recorded net power** | pickle SoC vs recursion from the stride x net power (split pch = max(pnet, 0)) | TSO / DSO | **5.6e-8 / 2.7e-5 MWh** |
| SoC, TSO vs DSO copy | pickle | 288 | 2.5e-5 MWh |
| Derived SoC, declared day | ESSO pch/pdch recursion / z recursion vs TSO-copy SoC | 24 | 6.5e-5 / 2.4e-5 MWh |
| ESSO pnet | capture vs stride x.esso | 288 | 0.0 |
| ESSO pnet definition | capture pch − pdch − pnet | 288 | 1.4e-15 MW |
| EFC per day | recomputed from the ESSO capture vs record, per year | 3 | 1.6e-15 |
| EFC max | stride line k* vs record | 1 | 0.0 |
| E^Av | pickle `shared_es_e_rated_fixed` × 100 vs record (S^Av: 0.0) | 24 | **1.2e-6 MWh** |
| E^Av | record published vs `ageing_trajectory_terminal` | 3 | 0.0 |
| Per-block Q / TSO generation / DSO flexibility | component_levels vs recourse_blocks_all line k* | 96 / 24 / 72 | 1.5e-8 / 0.0 / 0.0 € |
| Block weights | record vs SRP1.json formula | 96 | 5.7e-14 |
| Row 1 (x = 0) node-7 storage entries | stride at k* 181, all copies | 2,304 | 8.0e-25 MW |
| TSO shared ESS of nodes 5 and 9 | pickle capacity, pnet, qnet, SoC | — | 0.0 |

- **Largest differences.**
  - The 2.7e-5 MWh between the DSO-copy SoC and its net-power recursion is explained by the DSO copy's small
    simultaneous charge and discharge: the largest min(pch, pdch) is 1.8e-4 MW. The net-power split cannot represent
    this.
  - The 1.2e-6 MWh E^Av difference is consistent with the network copies holding the E^Av published after cycle
    k* − 1. This is a hypothesis; it is stated in the JSON and not verified.

### Guards (`e6_export.json` `guards`)

- **Solves:** `SolveProfileGuard(permitted=())` was installed before any project import. `verify(0)` returned no
  failures; the counts were 0 permitted and 0 blocked, for both solves and execs.
- **Builds:** `pyomo…Model.__init__` was blocked; it was called 0 times.
- **Pickle:**
  - **Delegated loads: exactly 1**, of the allowed path. The gate hashed the bytes it unpickled: `d0a3d512…`,
    164,685,741 B.
  - 3 load attempts and 1 `loads` attempt; 3 refusals, which is exactly the declared number (NC2, NC3, NC4); 0
    `Unpickler` attempts.
- **Negative controls:** all raised, and none delegated.
  - NC1: a gate with a wrong sha256 on the allow-list refused the real file, with a sha mismatch.
  - NC2: `pickle.loads(bytes)` was refused.
  - NC3: `pickle.load(BytesIO)` was refused (no path).
  - NC4: a second `pickle.load` of the allowed file was refused ("single declared load already made").
- **Production modules imported by unpickling** (name resolution only):
  - `branch`, `definitions`, `energy_storage`, `generator`, `helper_functions`, `hierarchical_coordination`, `load`;
  - `model_construction_helpers`, `network`, `network_parameters`, `node`, `shared_energy_storage`,
    `solver_parameters`.
- **Refusal paths tested (no load in any of them):** a non-fresh directory, a missing directory, and `--dry-run` into
  the default directory.

### Unexpected findings

- **2035 Spring holds almost all of the late-cycle oscillation of Q on both rows.**
  - Its block range over the certification window is 4,251 € on row 1 and 4,338 € on row 2. The terminal steps are
    105 € and 91 €.
  - Every other block's range is ≤ 222 €. Row 2's settling band is 4,351 €.
  - As a result, rank 3 (26,072.72 €) is not resolved against ranks 2 and 4 by the local bar.
  - This is an observation from records only. The declared day does not depend on it.
- **2025 Spring's value is almost entirely TN conventional cost:** −39,756 € of −40,181 €. The DN flexibility cost
  change there is only −423 €.
- **2030 Autumn and 2035 Winter show a TN generation cost increase** (+10,520 € and +317 €), outweighed by DN
  flexibility savings.
- **2025 Autumn has zero DN flexibility cost on both rows.**

---

## Not confirmed

- **The DN-internal flexibility activation volume (MWh).** I did not export it.
  - Committed records of both rows searched: every key matching `flex` in the JSON files and the first two JSONL
    lines of row 2's eval dir.
  - Result: per-block flexibility cost only. The volumes are DN-node totals of the TN-side interface delta
    (`flexibility_volumes_per_dso`).
  - Row 1's volume would need its pickle.
- **The cause of the 1.2e-6 MWh E^Av difference** (the one-cycle-lag hypothesis). I did not trace it in the cycle
  records.
- **Mapping of the TSO shared-ESS index to nodes 5 and 9.** The index is taken from the SRP1.json order. It is
  verified for node 7 only, through the bitwise stride match; nodes 5 and 9 are both all-zero, so they cannot be told
  apart.
- **The efficiencies (0.97/0.96) and SoC fractions in the recursions.** They come from the production defaults, which
  are asserted in the capture checklist by source text. I did not read them from the pickled constraint coefficients;
  the near-zero internal recursion residual (1.6e-11 MWh) is indirect confirmation.
- **The ESSO model has no intra-day SoC.** The ESSO SoC is therefore reported only as DERIVED.
- **I did not check that unpickling never calls a Pyomo `construct`.** The build guard covers `Model.__init__` only,
  and it read 0.
