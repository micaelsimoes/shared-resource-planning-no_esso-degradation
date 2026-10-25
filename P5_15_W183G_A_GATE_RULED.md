# P5.15 W183g (a): the two-cycle gate ruled — (b′) vs the W83 arm, and wrapper v2 with `GATE_PASS_RULED`

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 85, ruling 1 and W183g order item (a); Planner task W183g (a).
**Zero solves.** Nothing was re-run. The committed gate chain, the v1 wrapper, the W183f outputs and the W48 and W83
arms were only read.

## Blocked on Planner

1. **The (b′) prediction failed as stated, on its provenance part only.**
   - The genuine part **held**: 0 genuine diffs (and 0 tie-order) on all 12 files.
   - The provenance part **failed**: there are **4** provenance diffs, not ≤ 3. All 4 are under
     `rule_eleven_checklist`, but one of them, `w83_convergence_depth_tail`, is outside the predicted set.
   - That fourth key is the W83 arm's own tail read-back. W83 ran with the tight tail declared on
     (`declared.enabled: true`, `after.enabled: true`, `ok: true`). The W183f arm does not carry the key (`<MISSING>`).
     The gate's comparator classifies the diff as provenance, with the rule_eleven_checklist subtree reason.
   - So this is configuration provenance of the reference arm, not a trajectory difference.

   **Decision needed:** whether this outcome satisfies Addendum 85's condition for lifting the W184 stop. The
   addendum's condition text is "passed under rule (c) once (b′) returns 0 genuine diffs", and that condition is met
   as written. CLAUDE.md, however, routes a failed recorded prediction with no named fallback to the reader. I made no
   change in response, and W184 has not been touched.

## Changed

- **`4c0d8e1d`**: the scripts, committed before either run.
  - `p515_s55_w183f_bitwise_gate_v2.py` (sha256 `1d0d911c…2f2e`) is wrapper v2. It is a successor file;
    `p515_s55_w183f_bitwise_gate.py` and `p515_s53_srp1_bitwise_gate.py` are not edited.
  - `p515_s55_w183g_a_vs_w83.py` (sha256 `38dedf86…af8e`) is the (b′) comparator run.
- **This commit**: this report and the outputs of both runs (listed under Evidence).
- **Comparator.** Both scripts use the gate's own whole-arm comparator, imported and called unchanged:
  `S52G.compare_against_w39_arm`, which applies `W10.classify` to each of `W10.GATING_JSON + W10.GATING_JSONL`
  (12 files). It compares with `CP._diff` bitwise, keeps the declared exclusions, treats the rule_eleven_checklist
  subtree as provenance, normalizes arm-path tokens and handles recourse-jump tie order through TC. The only
  substitutions are the two the gate makes itself: `S52G.W39_ARM` (the reference-arm record) and `S52G.ARM`
  (`s55w183fgate`).
- **Guards, in both runs.**
  - `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0. The
    chain's own `W10.GUARD`, installed at import, was also verified at 0.
  - `pickle.load` and `pickle.loads` were blocked: 0 calls.
  - `ConcreteModel.__init__` was blocked: 0 calls.
  - JSON was written through `gate_result_io`.
  - Each run was attached and alone, with stdout and stderr captured to `launch.log`. The git HEAD at both runs was
    `4c0d8e1d`.

**Instance (both comparisons).** C\* is 0.96875 MVA / 3.875 MWh at nodes 5, 7 and 9, investment year 2025, with
candidate_key `578636da…6350c`. Objective convention: `gross_operational_cost` (settlement-excluded gross), as in the
gate.

## Found

### (b′) Committed W183f arm vs committed W83 arm (whole-arm reference)

**Precondition: every file hash-checked before comparing; 0 mismatches.**
- W83 gate directory: 100 manifest entries over `manifest_sha256.json` `f4d3f7aa…`, `w51_manifest_sha256.json`
  `8754894d…` and `w83_manifest_sha256.json` `a794ce17…`.
- W183f gate directory: 98 entries over `manifest_sha256.json` `140c33e1…`, `w51_manifest_sha256.json` `a6c78339…`
  and `w183f_manifest_sha256.json` `e7cd233e…`.
- In both directories no file on disk is missing from a manifest, and the directory is clean in git.
- The gate's own pin check (`S52G._check_w39_arm_pins`) passed on the W83 arm record.

**Prediction:** "0 genuine diffs on all 12 files; provenance diffs ≤ 3, all under `rule_eleven_checklist`
(`w10_arm`, `w183a_nrf_option`, `w84_convergence_depth_tail`)". **Outcome:** genuine 0 (**held**); provenance 4,
all under `rule_eleven_checklist`, with one key outside the predicted set (**failed**).

| file | provenance | tie_order | genuine |
|---|---|---|---|
| g_s39_D.json | 4 | 0 | 0 |
| boyd_terminal.json | 0 | 0 | 0 |
| component_levels_terminal.json | 0 | 0 | 0 |
| interface_settlement_detail_s31c.json | 0 | 0 | 0 |
| interface_voltage_terminal.json | 0 | 0 | 0 |
| recourse_jump_sidecar_baseline.jsonl | 0 | 0 | 0 |
| ess_entry_stride_baseline.jsonl | 0 | 0 | 0 |
| soh_floor_sidecar_baseline.jsonl | 0 | 0 | 0 |
| pf_entry_stride_s39_D.jsonl | 0 | 0 | 0 |
| ess_exempt_until_state_s39_D.jsonl | 0 | 0 | 0 |
| aa_per_cycle.jsonl | 0 | 0 | 0 |
| per_cycle_record.jsonl | 0 | 0 | 0 |
| **total** | **4** | **0** | **0** |

Every diff, all in `g_s39_D.json`, class provenance (reference = W83, new = W183f):

| key (`g_s39_D.json.rule_eleven_checklist.…`) | W83 | W183f |
|---|---|---|
| `w10_arm` | `s53w83tailgate` | `s55w183fgate` |
| `w183a_nrf_option` | `<MISSING>` | NRF read-back: every network flag off |
| `w83_convergence_depth_tail` | tail read-back: declared enabled true, after enabled true, ok true | `<MISSING>` |
| `w84_convergence_depth_tail` | `<MISSING>` | tail read-back: not declared, production default enabled false, ok true |

`per_cycle_record.jsonl` gives 0 diffs of any class, as expected because the two files are byte-identical (sha256
`f8506905…` in both, re-checked here). The ESSO pickle sha256 differs between the arms (`6665c436…` for W83,
`18510da9…` for W183f). This is informational, not gated and never unpickled; the W183f report records it as unequal
in the r2 and W83 runs too.

### (c) Wrapper v2, rule (c), and `GATE_PASS_RULED` on the committed W183f arm vs the W48 arm

**Rule as implemented.** It is applied on top of the gate's own classification and only to diffs the gate calls
`genuine`. Such a diff becomes `field_list_drift` only if all three conditions hold:
1. The reference side is `<MISSING>` and the new side is JSON null.
2. It resolves structurally to one key k of one record R. R is the innermost pair of dicts at the same dotted path,
   named as `CP._diff` names it. k must be absent from R_ref and None in R_new; the sentinel alone is not trusted.
3. Every key shared by R_ref and R_new is bitwise equal under the gate's comparator. Concretely, no raw `CP._diff`
   diff, of any class and at any depth, lies in R's subtree other than direct-child diffs on keys that are absent on
   one side. Any such shared-key diff disqualifies every drift candidate of R.

In addition:
- A strict, type-exact equality over all shared keys (including comparator-excluded ones) is recorded per record as
  information. It is true on both rows here.
- A key present in both records with different values stays genuine.
- A key absent in the reference and non-null in the new arm stays genuine, with the reason "absent in reference,
  non-null in new arm". Following the rule's text it is not a shared key, so it does not disqualify the record's other
  drift keys, but it keeps `GATE_PASS_RULED` False on its own. The same treatment applies to a key present in the
  reference and absent in the new arm.
- If the gate truncates a genuine list (more than 200 entries), the ruled verdict is False.

**`GATE_PASS_RULED`** is the conjunction of:
- every frozen gate item other than the zero-genuine one, as recorded;
- all 12 files present;
- the recomputed gate comparison reproducing the frozen one exactly, per file and in total;
- 0 genuine diffs after rule (c).

**Result** (`gate_ruled.json`):
- `GATE_PASS` = **False** (as frozen, read from `w183f_wrapper_record.json`, consistent with `w51_gate_addendum.json`).
- **`GATE_PASS_RULED` = True.**
- Every ruled item is true:
  - one frozen zero-genuine item (`vs_committed_w48_arm_zero_genuine_diffs: False`, replaced);
  - the other four frozen items are true as recorded (`w39_w35_gate_pass`,
    `w51_code_present_in_the_modules_that_ran`, `w51_counter_exact_and_never_acting`,
    `vs_committed_w48_arm_all_files_present`);
  - all files are present;
  - the recomputed comparison reproduces the frozen one (totals provenance 3 / tie 0 / genuine 44);
  - no file is undetermined;
  - genuine after the rule is 0.
- Totals: 44 frozen genuine → 44 `field_list_drift` + **0** genuine. All 44 are in `per_cycle_record.jsonl`: 22 per
  row, with 29 shared keys equal on each row. The other 11 files have 0 genuine before and after.
- **Drift keys (22)**: E_abs_d_p_mwh_unweighted, E_abs_d_p_mwh_weighted, covariance_dso_weighted,
  curtailed_res_dso_mwh_weighted, curtailed_res_tso_mwh_weighted, cycle_wall_s, market_part_mw2h_unweighted,
  market_part_mw2h_weighted, max_abs_d_p_mw, n_non_optimal_block_terminations, non_optimal_blocks,
  operation_part_mw2h_unweighted, operation_part_mw2h_weighted, response_capture_error, response_capture_s,
  response_captured, response_cycle, row18_charge_weighted, rss_bytes, ru_maxrss_bytes,
  sum_omega_d2_p_mw2h_unweighted, sum_omega_d2_p_mw2h_weighted.
  - This set equals `sorted(p515_s44_campaign_harness.PER_CYCLE_RESPONSE_FIELDS)` (checked: true).

**Negative controls.** Each control plants a value in an in-memory copy of the committed W183f file and is then
classified by the gate's `W10.classify` and by rule (c). The verdict is recomputed with the planted file. All behaved
as declared:

| control | planted | rule (c): drift / genuine | `GATE_PASS_RULED` |
|---|---|---|---|
| NC0 positive | nothing | 44 (22 + 22) / 0 | True |
| NC1 (i) | row 0 `recourse` + 1.0 (shared key) | 22 (row 1) / 23 (the planted diff + row 0's 22, disqualified) | **False** |
| NC2 (ii) | row 0 `rss_bytes` = 123456789; row 1 `n_non_optimal_block_terminations` = 0 (falsy, non-null); row 1 new key `planted_new_key` = 'x' | 42 (21 + 21) / 3, each with the reason "absent in reference, non-null in new arm" | **False** |
| NC3 (iii) | row 1 `boyd_v_primal_ratio` → `nextafter(v, inf)` (one ulp) | 22 (row 0) / 23 (the planted diff + row 1's 22, disqualified) | **False** |
| NC4 | row 0 `cycle_convergence` deleted (present only in the reference) | 44 / 1 (the deletion; not a shared key, so row 0 keeps its drift) | **False** |
| NC5a positive, nested | `g_s39_D.json` `instance.planted_drift_key` = None | 1 (record `g_s39_D.json.instance`) / 0 | True |
| NC5b (iii) at depth, any class | `g_s39_D.json` top-level `planted_drift_key` = None | 0 / 1: disqualified by the 3 *provenance* diffs inside the shared key `rule_eleven_checklist` | **False** |

(ii) is classified `genuine`, with its reason recorded; it never passes silently.

**`--run-gate` mode of wrapper v2: written, not run** (Addendum 85: nothing is re-run). It runs the committed gate with
v1's three substitutions plus v2 in `EXTRA_CLEAN_FILES`. It then writes `w183g_v2_wrapper_record.json`, carrying
`GATE_PASS`, `GATE_PASS_RULED` and the drift keys, through the same `ruled_comparison` and `ruled_verdict` that the
zero-solve run exercised. It requires `--arm` and `--out-rel` and refuses an existing output root.

## Commands

All were run from the repo root with the canonical interpreter, attached, with both streams captured.

```
mkdir -p data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_vs_w83 && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183g_a_vs_w83.py \
  > data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_vs_w83/launch.log 2>&1          # EXIT=0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183g_a_vs_w83.py --post-run   # EXIT=0
mkdir -p data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_ruled && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183f_bitwise_gate_v2.py \
  --ruled-from-committed > data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_ruled/launch.log 2>&1   # EXIT=0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183f_bitwise_gate_v2.py --post-run  # EXIT=0
```

Before the commit, one trial of each mode was written to the session scratchpad, outside the repository, with the same
guards. The outcomes were identical to the committed runs. After the runs, `git status` is clean on the W48, W83 and
W183f gate directories and on both committed gate scripts.

## Evidence (committed with this report)

- `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_vs_w83/`:
  - `vs_w83.json` `3f551d09…`
  - `manifest_sha256.json` `dc458a38…`
  - `launch.log` `ae140f5a…`
  - `manifest_post_run_sha256.json` `a6ce667d…`
- `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_ruled/`:
  - `gate_ruled.json` `ffa8712e…`
  - `manifest_sha256.json` `bbde2733…`
  - `launch.log` `9cdc47c2…`
  - `manifest_post_run_sha256.json` `e77fe6ae…`

## Not confirmed

- **Wrapper v2 `--run-gate` has not been executed.** Only its argument refusals were exercised (no mode given; and
  `--run-gate` without `--arm`/`--out-rel`, which exits before the gate). Its post-gate step is the code the zero-solve
  run exercised.
- **The four frozen gate items other than the zero-genuine one are carried as recorded and not re-evaluated.**
  Re-evaluating them would need the gate's solves.
- **Rule (c)'s handling of nested records and of provenance-class shared-key diffs** is exercised only by the planted
  controls NC5a and NC5b. No committed arm has a real instance of either.
- **The parallel DSO builder** is still not covered by a solve-bearing gate; this is unchanged, and Addendum 85
  ruling 1 records it as a limit.
