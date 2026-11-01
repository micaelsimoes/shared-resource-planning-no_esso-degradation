# P5.15 W182f: E.21 row 40 cap, code heads and configuration hashes, 3 × 3 cost, E6 inputs, reverse-flow hours (Addendum 82 v2, part B: Z2, Z3, Z4, Z5, Z7)

**Worker, 2026-10-25.** There were no solves, no model builds and no pickle loads. Records, specs, production code,
frozen files, the Overleaf clone, `manuscript_review/`, earlier W-scripts and their outputs, TASKS.md,
REVISION_CONTEXT.md and the brief were only read.

**State at the run.**
- Repo HEAD was `ab1d8d7b`, on branch `feature/derivative-free-planning`. It had moved on from `a17f6ba7` because the
  parallel W182 Workers committed in the meantime.
- The only tracked change in the working tree was `M TASKS.md`. It was there before this task started and is not
  mine.
- The production modules at HEAD are blob-identical to those at `575bfec4`. A production module here means a
  git-tracked root-level `*.py` whose name does not start with `p` followed by a digit.
- Recomputing the last commit that touched a production module at HEAD gives `575bfec4`, which matches the commit
  declared in the task.

---

## Blocked on Planner

None. The task is complete. Two decisions about E6 inputs are listed below. Neither blocks W182f, but both affect how
E6 can be built.

**Decisions needed (for E6, not for W182f)**

1. **ESS hourly schedules at k\* for rows 1–2.** These are the ESSO copy, the TSO/DSO copies, the consensus z and the
   reactive power. Their only text carrier is `ess_entry_stride_baseline.jsonl`, which is not committed: 13,594,249 B
   for row 1 and 17,907,215 B for row 2. The ESSO-side P is also in `esso_capture/` (not committed). Row 2's file at
   k\* is now committed; row 1's capture files are all empty, because x = 0 has no storage. Both stride files are
   hash-recorded.
   - **Option (a):** commit both stride files as they are with `git add -f`, 31.5 MB in total. There is precedent:
     the 13.7 MB `interface_duals_per_cycle.jsonl` files are committed.
   - **Option (b):** leave them hash-recorded only. E6 then reads uncommitted inputs, pinned by sha256.
   - **Recommendation:** (a), if E6 uses either of the two copies or Q.
2. **Hourly state of charge (SoC) appears in no text record of rows 1–2.** The key scan (scope under Z5) found it
   only in `certified_models.pkl`.
   - **Option (a):** authorise a zero-solve, pickle-reading export of row 2's `certified_models.pkl`. Its sha256
     `d0a3d512…` is pinned twice: in `post_certification.json` and in this task's manifest.
   - **Option (b):** reconstruct SoC from the ESSO-side `pnet` through the SoC recursion. The result is derived, not
     the model variable, and must be labelled as such.
   - **Option (c):** drop SoC from E6.
   - **Recommendation:** (a), because the instance is pinned and the guard pattern already exists.

---

## Changed

- **Commit:** the hash is in the final message. This report is committed in that same commit, by explicit filename.
- **New harness:** `p515_s53_w182f_records.py`.
  - `SolveProfileGuard(permitted=())` is installed before any project import, and was verified at exactly 0: 0
    permitted and 0 blocked, for both solves and execs.
  - `pickle.load` and `pickle.loads` were blocked and called 0 times.
  - No production model module was imported. The project modules imported were the guard, `gate_result_io` and
    the stdlib-only `settling_criterion*` modules.
  - The harness refuses to run concurrently with another copy of itself, or into an output directory that is not
    fresh.
  - A capture-path checklist ran before any reading: 104 required inputs, all present.
  - JSON is written through `gate_result_io`.
  - The run was attached, with stdout and stderr both captured to `launch.log`, and exited 0. `--post-run` also
    exited 0.
  - Two trial runs went to scratchpad directories through `--out-dir` (`w182f_trial2`, `w182f_trial3`). They are not
    in the repository.
- **Outputs**, in `data/SRP1/Results/P515S53/w182f_records/`. The directory was new, and each file was written once.

  | File | sha256 |
  |---|---|
  | `w182f_records.json` | `405c79c9e146600421a76443692fb23eb69d0d35d90358de14c899d66ddda085` |
  | `z5_rows12_file_manifest.json` | `7f24d7c5d4b958d4107c25cf3cb21ade0f862a8e24d0a92457cf919bcf47b751` |
  | `manifest_sha256.json` | `7a7a2b32278298884b3d838ccbf0aeaf532a197bbc18e5e96432e91bdcb2fc35` |
  | `launch.log` | `aec33b481820b204b5b4058ca52b6c25e7f06acb4372facc2b67def309920847` |
  | `manifest_post_run_sha256.json` | over the four files above |

  - `w182f_records.json` holds Z2, Z3, Z4, Z5 and Z7, plus the definitions and the guards.
  - `z5_rows12_file_manifest.json` lists all 2,303 files of the row 1–2 eval dirs and their P56A working dirs, each
    with path, size, sha256, mtime (UTC), git status and disposition. Of these, 2,239 files (3,172,648,673 B) are
    not committed.
  - `manifest_sha256.json` hashes every input with its git status, the two outputs and the script.
- **Committed as primary evidence (Z5),** with `git add -f`, unmodified and not moved:
  `data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_n7_4h_e1/evals/3f084f2ffaeef2b7_n7_4h_e1/esso_capture/s39_D/node7_cycle172.jsonl`
  (384,593 B, sha256 `ac8bc61e6798cba31c1541c91df39d15311a0b4991be5ae9f547175c83db8a2a`). It is row 2's ESSO-side
  hourly `pch`, `pdch` and `pnet` at k\* = 172.

---

## Found

### Predictions against outcomes

| Prediction (expert, Addendum 82 v2) | Outcome |
|---|---|
| Z2: row 40 is **not reachable within the cap**. | **HELD.** The cap is 281 and 281 cycles were used, so 0 remain. v6 on the record over cycles 1–281 is uncertified. |
| Z7: the four reverse-flow hours are **recoverable from `certified_models.pkl` only**. | **DID NOT HOLD.** Row 1's committed `interface_settlement_detail_s31c.json` carries all four hours: magnitude on both sides, block weight and market price. They are bitwise equal to W116's pickle-derived list. W116's committed `reverse_flow_count_q181.json` also lists them. |

The Z7 prediction is an evidence-inventory prediction, not a margin or sign prediction. Its failure removes the need
to unpickle anything; nothing downstream changes.

### Z2: row 40 (`ref:5ca4f86c`, the historical 2× incumbent, "gap clause")

**The issuing run.**
- Run: `w118_resettle/campaign_s53_w118_resettle_r2_f2_incumbent`, eval key `24c5ccb6…`, candidate `59757776…`.
- Stage spec: v2 `fc791891`. Stop rule: `settling_criterion_v2.SettlingRuleV2`. Flexibility price ×2.
- It replays the historical `5ca4f86c` (s51 F2 phase B, 181 cycles) bitwise through k0 = 172.

**Cap.**
- The cap rule is `{"kind": "fixed", "formula": "N_old + 100", "cap": 281}`, with N_old = 181. The spec cap is 281.
- G15 records `k_cap` 281, `cycles_run` 281 and `stopped_by` "cap". `resettle_decision.json` records
  `decided_at_cycle` 281 and `gap_clause_refused_at_cap` true.
- **Remaining cap: 0 cycles.**

**Reachability of a v6 certificate within the cap: not reachable.** This rests on four things:
1. **The committed v6 decision on the run's record.** W142 item 1 (`w142_v6_from_records.json`, record
   `f2_incumbent`) gives uncertified, k_cap 281, band width 398.19, T = [[176, min, 809,793,410.61]], 0 vetoes. It is
   equal to the W142 floor-replay AB arm, whose reason is `["gap_clause"]`, with 45 gap refusals and non-clean cycles
   [47, 58]. `settling_criterion_v6.replay` stops at the first decision, so no cycle k ≤ 281 certifies.
2. **An independent call of the committed zero-solve replay.** `settling_criterion_v6.replay` was run on the run's
   committed `per_cycle_record.jsonl` and `resettle_cycle_record.jsonl`, with W142's inputs: cap 281, ceiling 300,
   p_max 30, non-clean cycles {47, 58}. It reproduces status, k_star, k_cap, band_width and T of W142 item 1, and
   status, reasons and the full gap-refusal list of the AB arm (`all_equal` true). The rule was called, not
   re-implemented.
3. **What blocks certification is the gap clause alone.**
   - There are 45 refusals, at cycles 237–281, all on the monotone branch. On each of them the branch test passed
     and only the gap clause refused.
   - |t_sum| ranges from 9,216.7 to 9,234.4 €, against a gap bound of τ/2 = 2,269.5 €, so it is at least 4.06 times
     the bound.
   - The least-squares slope of |t_sum| over the refusal cycles is +0.40 €/cycle. This is an observation, not a
     prediction.
4. **Nothing is recorded beyond cycle 281.** Whether the gap would close after the cap cannot be known from the
   records.

**Other committed runs of the same candidate** (from the campaign specs that list key `59757776…`):

| Spec | Cap | Eval key | Eval dir | Recorded |
|---|---:|---|---|---|
| s47 phase B `8cfa264e` | 500 | `bd3cbb82` (m = 1) | absent | — |
| s51 F2 phase B `5ce295e1` | 500 | `5ca4f86c` (m = 2) | present | harness residual certificate at 181, 181 cycles |
| s53 F2 certificate `3b9d6df9` | 500 | `5ca4f86c` | absent | `SUPERSEDED.md` present |
| s53 F2 certificate r1 `803571c0` | 500 | `5ca4f86c` | absent | — |
| W118 r1 `54d27495` | 281 | `24c5ccb6` | absent | — |
| W118 r2 `b1dd9b82` (issuing run) | 281 | `24c5ccb6` | present | settling v2: uncertified at the cap |

`evaluation_record.status` "certified" means the harness residual certificate, which counts consecutive Boyd-pass
cycles. It is not a settling-rule decision.

### Z3: code head, final-head check and configuration hash for each of the 61 E.21 rows

**Definitions** (preserved in `w182f_records.json`, under `Z3_e21_code_heads` and the script docstring).
- **Spec-freeze head:** the campaign spec's `git_head`, that is, HEAD when the spec was frozen, before launch.
- **Run-end head:** `campaign_results.git_head_at_run`, that is, HEAD when the results were written, after the run.
  No record holds the HEAD at the moment of launch; these two heads bracket the run.
- **Decision head:** for "record" rows, the `git_head` of the committed record that issued the v6 decision:
  `367f06f2`, from W142 `w142_v6_from_records.json` and the d_c52e1670 certificate. For v6 runs it is the run itself.
- **Final head F:** `575bfec4`, the last commit touching a production module (defined as above). No run record names
  any other "final" head.
- **Production modules = F:** every production-module blob at the run-end head is equal to F's.
- **Production diff vs F:** `A` means a module added after the run, which the run could not have imported. `M` means
  a module that was modified.
- **cfg_hash:** the sha256 of `json.dumps(PAYLOAD, sort_keys=True, separators=(',', ':'))`. PAYLOAD has these keys:
  - from the configuration: `case_file_sha256`, `case_file_anderson_acceleration`, `convergence_depth_tail`,
    `ess_ageing_baseline`, the `ess_params_file` sha256, `apply_rho`, `arm_label`;
  - from the entry: `effective_anderson_acceleration`, `flex_price_multiplier`, `model_variant`, `derived_instance`,
    `interface_deviation_premium`;
  - `overrides`, taken from both the configuration and the entry;
  - the model-affecting fields of `settling_resettle` (`kind`, `minimum_soh`, `flex_price_multiplier_expected`,
    `model_variant`), with null values omitted.

  The hash leaves out the candidate, the run control (cap, settling declaration, replay reference, stop rule),
  post-certification options and free text. The stop rule is reported in its own column. The runtime fields
  (interpreter, `NLP_SOLVER_PATH`, thread caps) are the same on all 61 rows (one distinct value), so they are
  compared rather than hashed.

| No | Cell | Rule | Spec-freeze head | Run-end head | Decision head | = F literally | Production modules = F | Production diff vs F | Harness sha256 | cfg_hash16 |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | `ref:7aa017f0` | v1 | `4ca1573d` | `aad37370` | `367f06f2` | no | no | A: sc_v2, sc_v3, sc_v4, sc_v5, sc_v6; M: uncoordinated_benchmark.py | `3ad624cd` | `83be7d9d34e05138` |
| 2 | `ref:bd504ecf` | v1 | `4ca1573d` | `6fbd4606` | `367f06f2` | no | no | A: sc_v2, sc_v3, sc_v4, sc_v5, sc_v6; M: uncoordinated_benchmark.py | `3ad624cd` | `83be7d9d34e05138` |
| 3 | `d_c52e1670` | v5 | `ed767408` | `4b3be392` | `367f06f2` | no | no | A: sc_v6 | `961dbb3e` | `83be7d9d34e05138` |
| 4 | `d_4a82a64a` | v6 | `0a6642ec` | `b64a4233` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 5 | `d_3632b0ae` | v6 | `0a6642ec` | `354b362a` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 6 | `d_36686489` | v6 | `0a6642ec` | `48625326` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 7 | `d_d3709599` | v6 | `0a6642ec` | `e487cbde` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 8 | `d_a12d95a2` | v6 | `0a6642ec` | `02affadd` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 9 | `d_f759dd48` | v6 | `0a6642ec` | `73d41460` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 10 | `d_c7fee8be` | v6 | `0a6642ec` | `dfdc2ccf` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 11 | `d_9246ed01` | v6 | `0a6642ec` | `d0a34c40` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 12 | `b_2a0ba8b2` | v4 | `3d610aa0` | `852fe22a` | `367f06f2` | no | no | A: sc_v5, sc_v6 | `25cc4559` | `83be7d9d34e05138` |
| 13 | `b_0dd237f0` | v4 | `3d610aa0` | `903657de` | `367f06f2` | no | no | A: sc_v5, sc_v6 | `25cc4559` | `83be7d9d34e05138` |
| 14 | `b_4649234b` | v5 | `ed767408` | `07eb8890` | `367f06f2` | no | no | A: sc_v6 | `961dbb3e` | `83be7d9d34e05138` |
| 15 | `pb_y2025_n5_v6` | v6 | `55bb1e7d` | `72d52c23` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 16 | `pb_y2025_n5` | v2 | `b4aab001` | `ad4e4747` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 17 | `pb_y2025_n7` | v2 | `b4aab001` | `bcc7669b` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 18 | `pb_y2025_n9` | v2 | `b4aab001` | `d6333b6f` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 19 | `pb_y2030_n5` | v2 | `b4aab001` | `5cda0cd0` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 20 | `pb_y2030_n7` | v2 | `b4aab001` | `6eb5f7ca` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 21 | `pb_y2030_n9` | v2 | `b4aab001` | `2cda084d` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 22 | `c_156ce2d1` | v6 | `0a6642ec` | `fdd70605` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 23 | `c_6597a79d` | v6 | `0a6642ec` | `1fcbce4e` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 24 | `yl_y2030` | v2 | `b4aab001` | `af2210d2` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 25 | `yl_y2035` | v2 | `b4aab001` | `d5f150f6` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `83be7d9d34e05138` |
| 26 | `g_37b5c499` | v6 | `0a6642ec` | `3d9b8d94` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 27 | `g_47dce43c` | v6 | `0a6642ec` | `bf810183` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 28 | `g_48749148` | v6 | `0a6642ec` | `4ab435aa` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 29 | `g_9abf31d4` | v6 | `0a6642ec` | `58f7209e` | own run | no | yes | none | `ca46892b` | `83be7d9d34e05138` |
| 30 | `h_aa8a76d7` | v6 | `0a6642ec` | `9d2f458f` | own run | no | yes | none | `ca46892b` | `eda23b85e19cf022` |
| 31 | `h_f9eae48f` | v6 | `0a6642ec` | `40568e6a` | own run | no | yes | none | `ca46892b` | `eda23b85e19cf022` |
| 32 | `h_x0_m175` | v6 | `c0dac227` | `0e1d7c0e` | own run | no | yes | none | `9c6bda19` | `74e21b9a7f6de88f` |
| 33 | `h_unit_m175` | v6 | `c0dac227` | `ae2b6574` | own run | no | yes | none | `9c6bda19` | `74e21b9a7f6de88f` |
| 34 | `h_50dea31c` | v6 | `0a6642ec` | `3740b476` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 35 | `h_74eda68d` | v6 | `0a6642ec` | `72d74291` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 36 | `i_5a6a88b4` | v6 | `0a6642ec` | `4611e197` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 37 | `j_f3aa335e` | v6 | `0a6642ec` | `4b30ea00` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 38 | `j_a11d7966` | v6 | `0a6642ec` | `b8a6e689` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 39 | `j_5f3cccb4` | v6 | `0a6642ec` | `49c658da` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 40 | `ref:5ca4f86c` | v2 | `b4aab001` | `90b64dc0` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `1cb225b67fa0ecfd` |
| 41 | `ref:e28de4ac` | v2 | `b4aab001` | `eae4a359` | `367f06f2` | no | no | A: sc_v3, sc_v4, sc_v5, sc_v6 | `4115fc83` | `1cb225b67fa0ecfd` |
| 42 | `l_0ee93aca` | v6 | `0a6642ec` | `eb9e3dd7` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 43 | `l_195156fa` | v6 | `0a6642ec` | `9af8c950` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 44 | `l_2ab0ce2d` | v6 | `0a6642ec` | `122f5bdf` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 45 | `l_b2251bc5` | v6 | `0a6642ec` | `56891699` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 46 | `l_76c78064` | v6 | `0a6642ec` | `a477ab96` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 47 | `l_8e4c220e` | v6 | `0a6642ec` | `dcf3e87e` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 48 | `l_e1da0984` | v6 | `0a6642ec` | `f42e7299` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 49 | `l_7b199ef9` | v6 | `0a6642ec` | `bf84fd41` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 50 | `l_45aa25a6` | v6 | `0a6642ec` | `714c9445` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 51 | `l_7c455554` | v6 | `0a6642ec` | `6f116e6c` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 52 | `l_7db09f6c` | v6 | `0a6642ec` | `ad818c3a` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 53 | `l_df1a5525` | v6 | `0a6642ec` | `7f2475bd` | own run | no | yes | none | `ca46892b` | `1cb225b67fa0ecfd` |
| 54 | `e_no_ageing` | v6 | `55bb1e7d` | `d7602e1a` | own run | no | yes | none | `ca46892b` | `73937249d93e6de5` |
| 55 | `e_c2` | v6 | `55bb1e7d` | `22c7c116` | own run | no | yes | none | `ca46892b` | `f89348e75af91413` |
| 56 | `e_c2_calfade` | v6 | `55bb1e7d` | `afc6dbd7` | own run | no | yes | none | `ca46892b` | `836dff235b95e925` |
| 57 | `e_c4` | v6 | `55bb1e7d` | `23cee034` | own run | no | yes | none | `ca46892b` | `09b127ff014d2930` |
| 58 | `e_c3_unit` | v6 | `55bb1e7d` | `13c6f5b0` | own run | no | yes | none | `ca46892b` | `1fae56a61ab56441` |
| 59 | `e_c3_midblock` | v6 | `55bb1e7d` | `4c735dc3` | own run | no | yes | none | `ca46892b` | `5f975df0796c8427` |
| 60 | `e_soh050` | v6 | `c0dac227` | `6cc4c8c1` | own run | no | yes | none | `9c6bda19` | `e68c772fe02fa96e` |
| 61 | `g070_neutrality` | v6 | `c0dac227` | `241d3030` | own run | no | yes | none | `9c6bda19` | `1d59c18b4a138aca` |

Rule is the stop-rule class of the issuing stage spec (v1 = `settling_criterion.SettlingRule`).

**Summary.**
- **Equal to F literally: 0 of 61.** No run ran at `575bfec4` itself.
- **Production modules equal to F: 45 of 61.** These are all the v6-stage runs (spec v6, ext v3 and A64 v1).
- **Differing only by modules added later: 14 rows**: 3, 12, 13, 14, 16–21, 24, 25, 40 and 41. The added modules are
  `settling_criterion_v3` to `_v6`.
- **Rows 1 and 2 (W101)** also differ by a modified `uncoordinated_benchmark.py` (W116 `72344762`, W121 `9098d4cd`).
- **The W118 rows (16–21, 24, 25, 40, 41):** HEAD moved during the run across a change to `uncoordinated_benchmark.py`
  (W121 `9098d4cd`).
- **`uncoordinated_benchmark.py` is not imported** by the campaign harness, by any production module, or by any row's
  campaign script. Its only importers at HEAD are 10 benchmark scripts, W93 to W130; no hooks module is among them.
  The check was a regex for
  `import M` / `from M import` over every tracked `*.py`; dynamic imports are not detected.
- **Harness.** `p515_s44_campaign_harness.py` is outside F's definition, and it changed after F: at `7914860f` (the
  v6 router) and at `b4c3dc7a` (the W155 router, 3 added lines).
  - It equals the HEAD harness `9c6bda19` on 4 rows: 32, 33, 60 and 61.
  - The 41 rows on `ca46892b` differ from HEAD only by those 3 router lines for W155 declarations.
- **Configuration groups:**

  | Group | Rows | Distinguished by |
  |---|---|---|
  | `83be7d9d…` | 1–29 | baseline, m = 1 |
  | `eda23b85…` | 30, 31 | m = 1.5 |
  | `74e21b9a…` | 32, 33 | m = 1.75 |
  | `1cb225b6…` | 34–53 | m = 2 |
  | 54–59 | one hash each | ageing model variants |
  | `e68c772f…` | 60 | minimum SoH 0.5 |
  | `1d59c18b…` | 61 | see below |

  Row 61 (`g070_neutrality`) differs from row 56 (`e_c2_calfade`) only because its run declaration states
  `kind: soh, minimum_soh: 0.7` explicitly. That is the identity value, by design of the neutrality gate.

### Z4: wall time per cycle and peak RSS, from the records only (for the E4 file)

Sources:
- `per_cycle_record.jsonl`, fields `cycle_wall_s`, `rss_bytes` and `ru_maxrss_bytes`;
- `evaluation_record.json`, fields `wall_time_s` and `peak_rss`;
- `wait4_rusage.json`.

`ru_maxrss` is in bytes on macOS. GiB = 2³⁰ B.

| Cell | Cycles | Wall per cycle, s: mean / median / min / max | Cycle 1, s | Child process wall, s | Peak RSS, child (GiB) | Max RSS, IPOPT subprocess (GiB) |
|---|---:|---|---:|---:|---:|---:|
| 3 × 3 pair x0 (`f6e9cd53`, certified 72) | 72 | 386.0 / 373.2 / 349.4 / 695.3 | 676.9 | 28,499.8 | 15.16 (16,276,684,800 B) | 0.38 |
| 3 × 3 pair n7_4h_e1 (`c82522f4`, certified 69) | 69 | 425.2 / 413.5 / 377.6 / 714.5 | 714.5 | 30,076.6 | 15.18 (16,302,620,672 B) | 0.38 |
| 3 × 3 continuation x0 (`25b92ae0`, certified 88) | 88 (cycles 73–88: 380.4 / 376.0 / 374.3 / 396.5) | 384.5 / 376.7 / 353.3 / 681.0 | 669.1 | 34,554.5 | 15.22 (16,343,547,904 B) | 0.38 |
| Smoke b on (beside, labelled) | 3 | 581.7 / 644.3 / 415.8 / 685.1 | 644.3 | 2,423.2 | 17.04 (production state 15.01) | 0.40 |
| Smoke b off (beside, labelled) | 3 | 593.4 / 668.3 / 421.3 / 690.4 | 668.3 | 3,104.6 | 18.52 | 0.39 |

- **Wall outside the cycles:** child wall minus Σ `cycle_wall_s` is 707 s for pair x0, 736 s for pair n7 and 722 s
  for the continuation. This is derived, and it includes the read, the build and post-certification.
- **Peak RSS** is the same in `wait4_rusage` and `evaluation_record.peak_rss` on every cell.
- **Five-block single-scenario build: not recorded.**
  - Five-block means the 3 × 3 horizon 2025/2028/2031/2034/2037: 80 network blocks with 1 market × 1 operation
    scenario.
  - Search scope: the 54 git-tracked `*.json` files that contain `"n_network_blocks_built"` or
    `"expected_block_counts"`, read in full, at every depth.
  - Block × scenario pairs found: (80, 4) 26 times, (80, 9) 13, (16, 4) 10, (48, 1) 7, (80, 25) 4, (80, unknown) 1.
    The pair (80, 1) does not occur.
- **Nearest recorded builds.** These are labelled and are not the requested build. They are model builds only, with
  no solver state.

  | Build | Blocks × scenarios | Source | Build wall | RSS |
  |---|---|---|---:|---|
  | SRP1 three-block, single scenario | 48 × 1 | W89 memory probe | 6.4 s | 0.94 GB after build |
  | SRP1 three-block, single scenario | 48 × 1 | S44 build records | 20.3–27.2 s | 0.90–1.17 GB |
  | five-block 2 × 2 | 80 × 4 | W89 probe | 37.2 s | 3.62 GB |
  | five-block 3 × 3 | 80 × 9 | W89 probe | 76.7 s | 7.44 GB |
  | paper five-block 5 × 5 | 80 × 25 | S44 build records | 232–234 s | RSS 15.1–15.3 GB; `ru_maxrss` 16.3–16.7 GB |

### Z5: E6 inputs for rows 1 and 2 (W101 continuations; k\* 181 for x0, k\* 172 for n7_4h_e1)

**`ess_schedule_per_cycle.jsonl` does not exist for rows 1–2.** It is a W105/W118 capture, and the W101 runs predate
it.
- Search scope: an `os.walk` over all of `data/`. 62 directories hold such a file. None of them has a path containing
  eval key `d110bd1a5977df1e` or `3f084f2ffaeef2b7`.
- The equivalent data for rows 1–2 is `ess_entry_stride_baseline.jsonl` (not committed) and `esso_capture/` (not
  committed).

| E6 quantity (at k\*, or terminal = k\*) | Class (rows 1 and 2 alike) | Carrier(s) |
|---|---|---|
| ESS hourly charge, discharge and net P, ESSO side | **uncommitted text only** | row 2: `esso_capture/s39_D/node7_cycle172.jsonl` (**now committed**) and `ess_entry_stride_baseline.jsonl`; row 1: stride file only (capture files empty, x = 0); `certified_models.pkl` |
| ESS hourly P, TSO/DSO copies and consensus z | **uncommitted text only** | `ess_entry_stride_baseline.jsonl`; `certified_models.pkl` |
| ESS hourly Q (all copies) | **uncommitted text only** | `ess_entry_stride_baseline.jsonl`; `certified_models.pkl` |
| ESS hourly SoC / stored energy | **`certified_models.pkl` only** | key scan below |
| SoH per cohort and year | committed record | `soh_floor_sidecar_baseline.jsonl` (line k\*), `boyd_terminal.json` |
| Available capacity per year (E^Av, S^Av) | committed record | `evaluation_record.json` `storage_per_node.<n>.published_available_capacity_terminal` |
| EFC per day | committed record | `per_cycle_record.jsonl` `efc_per_day_max` (also in the stride file) |
| Market price per hour | committed record | `interface_settlement_detail_s31c.json` `price_per_mwh` |
| Interface P and Q per hour, both sides | committed record | `interface_settlement_detail_s31c.json` |
| Block weight per (year, day) | committed record | `component_levels_terminal.json`, `interface_settlement_detail_s31c.json` |
| Per-block cost components | committed record | `component_levels_terminal.json`, `recourse_blocks_all.jsonl` (line k\*) |
| Interface consensus duals per hour (scaled, unconverted) | committed record | `interface_duals_per_cycle.jsonl` (line k\*) |
| Interface voltage per hour | committed record | `interface_voltage_terminal.json` |
| Gross operational cost Q(k\*) | committed record | `per_cycle_record.jsonl` |

**How each class was established.**
- Each field was read at k\* in the named file. For JSON this is the whole file. For an uncommitted JSONL file it is
  the last line, checked to be cycle k\*.
- The terminal files carry `cycles_run` equal to k\*: 181 for row 1 and 172 for row 2.
- **SoC key scan:**
  - Coverage: every text record of each eval dir (28 files for row 1, 202 for row 2). For JSON, every key at any
    depth. For JSONL, the header, the first data line and the last line. `.log` files are not scanned and pickles
    are not opened.
  - Regex: `(^|[^a-z])(soc|state_of_charge|stored_energy|energy_level)([^a-z]|$)`, case-insensitive. It matches the
    production names `es_soc` and `shared_es_soc`; a self-test of 5 positive and 5 negative cases passes.
  - Result: 0 hits for either row. SoC is therefore in no text record within this scope.

**Commit or hash-record.**
- **Rule:** an uncommitted text file is committed if it carries an E6 quantity at k\* that no committed record
  carries and is at most 1 MiB. Everything else that is uncommitted is hash-recorded.
- **Committed now:** 1 file, row 2's `node7_cycle172.jsonl` (see Changed).
- **Hash-recorded in `z5_rows12_file_manifest.json`:** 2,238 files in total. Per row:

  | | Row 1 | Row 2 |
  |---|---|---|
  | Pickles | 4, 177.7 MB | 4, 178.4 MB |
  | Other files in the eval dir | 548 | 520 |
  | P56A IPOPT working-dir logs | 594, 1.13 GB | 568, 1.37 GB |

- The pickles are `certified_models.pkl`, `esso_models_s39_D.pkl` and two `results/FrozenSMOPF/*cycle7.pkl` each.
  They were hashed as bytes and never opened.
- `certified_models.pkl` sha256 matches its existing pins:
  - row 1, `99ab1070…`, equal to W116's `instance.certified_models.sha256`;
  - row 2, `d0a3d512…`, equal to `post_certification.json`'s `persisted_models.sha256`.
- Uncommitted large text, hash-recorded:

  | File | Row 1 | Row 2 |
  |---|---|---|
  | `ess_entry_stride_baseline.jsonl` | 13.6 MB, `0012cef5…` | 17.9 MB, `47de0cb1…` |
  | `pf_entry_stride_s39_D.jsonl` | 111.5 MB, `f7114758…` | 105.9 MB, `1f6bbb47…` |

- Both eval dirs already carry 32 committed files each.

### Z7: row 1 (x = 0, k\* 181), the reverse-flow interface-hours, from committed records

**Source:** row 1's committed `interface_settlement_detail_s31c.json`, with `cycles_run` 181 = k\*.

**Conventions.**
- **Sign:** p_int < 0 is DN → TN, as in W116's `sign_proof`.
- **Price:** `price_per_mwh` = E[π_t], which equals `cost_energy_p[0][t]` at one market scenario
  (`shared_resources_planning._INTERFACE_REPORTING_PRICE_CONVENTION`).
- **Block weight:** years × days × 1/(1+r)^(y−2025), from `_get_admm_block_weight`. There is no discount in 2025.

All 864 interface-hours were scanned. Four are reverse on the DSO side, all material (below −1e-6 p.u. of the DN
base), and the same four are reverse on the TSO side:

| Node | Year | Day | Hour (period) | p_int DSO (MW) | p_int TSO (MW) | Block weight | Market price (€/MWh) | Block-weighted reverse energy (MWh) | Raw λ_pf,p (TSO / DSO), cycle 181 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 7 | 2025 | Autumn | 19 (18) | −3.156857 | −3.156879 | 455 | 161.80 | 1,436.37 | 0.06500 / −0.06500 |
| 7 | 2025 | Spring | 22 (21) | −2.401562 | −2.401575 | 460 | 134.53 | 1,104.72 | 0.82988 / −0.82988 |
| 7 | 2025 | Spring | 2 (1) | −1.601685 | −1.601685 | 460 | 117.73 | 736.78 | 0.03240 / −0.03240 |
| 7 | 2025 | Summer | 22 (21) | −0.066891 | −0.066891 | 455 | 131.72 | 30.44 | 0.53205 / −0.53205 |

- **Totals:** block weight 1,830; reverse energy 3,308.30 MWh, block-weighted.
- **Cross-check against W116** (`reverse_flow_count_q181.json`, written from this cell's `certified_models.pkl`,
  sha256 `99ab1070…`):
  - the four DSO values, nodes, days and periods are **bitwise equal**;
  - the count (4) and `count_day_weighted` (1,830) are equal;
  - the block-weighted energy differs by 0.0;
  - `report_v3.json`'s `coordinated_reverse_flow_count` agrees.
- **λ** is the raw scaled consensus dual as production holds it. The conversion to €/MWh lives in
  `uncoordinated_benchmark.interface_price_terms` and was **not** applied.
- The JSON also carries |p| × weight × price (sum 471,772.88 €). That is an energy × price product derived here. It is
  not a model quantity, and **not** the value of permitting reverse flow.

---

## Not confirmed

- **The launch-time HEAD of each run.** Only the freeze head and the run-end head are recorded. Equality with F was
  judged at the run-end head, and the diff from freeze to run end is reported separately.
- **Whether `uncoordinated_benchmark.py` is unused by rows 1–2 and the W118 rows rests on a static import regex.**
  Dynamic imports were not searched, and the module was not traced at run time.
- **The cfg_hash field list is my definition.** A configuration carried by a hook module and not declared in the spec
  would not be seen. The hooks' code is pinned by the campaign script sha256 in each spec (column not shown; it is in
  the JSON).
- **Z4.**
  - Whether cycle 1's extra ≈ 300 s is initialisation or solver warm-up was not separated; the records do not
    separate it.
  - Smoke-run RSS above 17 GiB was not investigated.
- **Z5.**
  - The SoC scan covers only the stated keys and lines. A homogeneous JSONL file was assumed to carry the same keys
    on every line.
  - The P56A IPOPT logs were hashed but not read.
- **Z7.** The "material" threshold assumes a DN base of 100 MVA, the value recorded in the duals header for every
  block. The four entries are below −0.0669 MW, far beyond −1e-4 MW, so this does not affect the count.
- **Z2.** The rule was not re-run on any record other than row 40, and no cycle beyond 281 exists.
- **The Overleaf clone was not touched or read.** None of Z2–Z7 needed it; row identity comes from W180b's committed
  JSON, which was made at `e5322f8`. I did not check whether E.21 at `edd7a63` still has the same 61 rows in the same
  order.
