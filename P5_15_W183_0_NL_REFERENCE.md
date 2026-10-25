# P5.15 W183-0 — pre-change NL and component-digest reference (Addendum 83; G0 first run)

Zero solves. Builds only, as Addendum 83 permits. No production module, harness other than the new one, case, params,
cost or spec file was touched.

Objective convention: no objective value is reported here. The models are those IPOPT would receive at the ADMM
initialisation solve; nothing was solved.

## Blocked on Planner

None. Two points for the Planner's information, neither blocking:

1. **The main JSON is 109 MB.** The (iii) comparisons embed full objective strings as per-index examples. The raw
   file stays on disk, untracked (sha256 `d2245ef6…`). The commit carries a deterministic gzip of it
   (`w183_nl_digest.json.gz`, 10.2 MB, sha256 `cd58078e…`); its decompressed sha256 equals the raw file's.
   W183c and W183d reuse this harness and will hit the same size whenever 1× and 2× configurations share a run.
   Truncating the example strings needs a harness change; I did not make it after the run.
2. **Only the ESSO NL files are committed.** The network NL files (389 MB gz) exceed the ~50 MB limit and are
   hash-recorded instead: the raw sha256 is in the JSON and the gz sha256 in `manifest_sha256.json`. The ESSO NL
   files (1.3 MB, 120 files) are committed. W183c's NRF-off identity gate compares sha256 values, so it needs only
   the hashes.

## Changed

- `62c62d08`: the harness `p515_s53_w183_nl_digest.py` (sha256 `c867a4c1…`), committed before the real run.
- This commit adds:
  - the report;
  - `data/SRP1/Results/P515S55/w183_nl_reference_pre/`:
    - `w183_nl_digest.json.gz` (gzip of the result JSON, written through `gate_result_io`);
    - `manifest_sha256.json` (4,353 files: the JSON and every NL / item gz file);
    - `manifest_post_run_sha256.json` (raw JSON, its gz, both manifests, `launch.log`, the child JSON);
    - `launch.log` (stdout + stderr);
    - `determinism_child/w183_nl_digest.json` and `determinism_child/manifest_sha256.json`;
    - the 120 ESSO NL gz files (`nl/<config>/sequential/ESSO_node{5,7,9}.{nl,sym.nl,sym.row,sym.col}.gz`, main run
      and child).
- Untracked on disk, hash-recorded: the raw 109 MB JSON, the network NL gz files, and the per-block item gz files
  (539 MB in the main run, 135 MB in the child).

**Run conditions.**
- Command (attached, alone, both streams to `launch.log`):
  `NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s53_w183_nl_digest.py
  --out data/SRP1/Results/P515S55/w183_nl_reference_pre --label pre --scratch <scratchpad>/w183_0_build
  --expect-production-commit 575bfec4`.
- Exit 0; wall clock 930 s.
- Git HEAD at run: `62c62d08` (the harness commit).
- Production modules: the 38 tracked root-level `*.py` not matching `p\d`. Their working-tree content equals
  `575bfec4` (the harness refuses otherwise), and their last commit is `575bfec4`.
  - `shared_resources_planning.py` `0610d745…`
  - `shared_energy_storage_data.py` `9acd095f…`
  - Both equal the hashes pinned in the ext-v6 `e_no_ageing` campaign spec.
- 46 repo modules were imported. Their sha256 is recorded at the start and at the end, and was unchanged. All are
  tracked and clean (`p515_s44_campaign_harness.py` `9c6bda19…`).
- Inputs: 81 tracked SRP1 input files hashed, all clean. `SRP1_ESS_Params.json` is `39106f93…`, the ext-v6 pin.
- pyomo 6.9.5; Python 3.11 (canonical interpreter).
- `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0 in the
  parent and in the determinism child (counts all 0).
- Stub call counts were exact in every build:
  - sequential: TSO 1, DSO5/7/9 1 each, ESSO 1;
  - parallel: DSO5/7/9 1 each.
- Every sequential build returned `initialization_failed` True. This is production's branch that returns before
  `update_*_to_admm`.

**What was built.** Each of the 8 configurations was built twice, each time in a fresh planning object.
- Grid: {x = 0, unit = node 7 0.25 MVA / 1.0 MWh 2025} × {A = baseline, N = `model_variant` no-ageing
  (`ageing_enabled` False, 'end', calendar retention 1.0, `eol_retention_r` 0.50)} × {flex 1.0 explicit, 2.0}.
- Configuration path: `_construct_arm_planning` followed by the child's `_config_hook_factory`. The spec-like
  configuration is the ext-v6 block: case-file AA, ESS ageing baseline declared and pinned, tail declared enabled.
- Sequential builder: production's own `run_operational_planning`, with every `optimize` stubbed. It captures 12 TSO,
  36 DSO and 3 ESSO blocks.
- Parallel builder: `create_distribution_network_model`, run in-process, 36 DSO blocks.
- Per block, the harness records:
  - the component digest, using `block_state` verbatim from `p515_s47_case_file_baseline_check.py` (source equality
    asserted before the run);
  - the per-index items, and the constraint `active` flags;
  - the NL written with `symbolic_solver_labels` False, which is the production writer path (presolve and scaling
    forced off);
  - the NL written with labels True, with its `.row` and `.col`.
- Both NL copies are equal after comment stripping, except the header's "max name lengths" line, in every block (check
  `labelled_nl_equals_plain_after_comment_strip`).

**Read-backs, every configuration and both builders:**
- `w33_flex_price_multiplier.readback_all_match` True: 55,296 flex coefficients and all 8 checks, at 1.0 and at 2.0.
- N: `w20_model_variant.readback_all_match` True; apply checks pass; the floor rows are identical to the baseline
  probe.
- A: `w21_ess_ageing_baseline.readback_all_match` True.

**Eval keys.** The base key is computed as the child computes it: candidate key, case-file AA, `model_variant`,
`ess_ageing_baseline`, flex when ≠ 1, tail.
- The 8 keys are pairwise distinct.
- unit_N_1x = `054976fb…`, equal to the committed ext-v6 `e_no_ageing` record's `base_key_current_configuration`.
- The keys: x0_A_1x `5cfe69a6…`, x0_A_2x `5f6b7841…`, x0_N_1x `105c0fd3…`, x0_N_2x `c15eea62…`, unit_A_1x
  `ca8927e7…`, unit_A_2x `bde9355e…`, unit_N_1x `054976fb…`, unit_N_2x `057f9a6f…`.
- A 2× key differs from the same model's 1× key, and an N key from the A key, so the key carries both options.

## Found

### (i) x = 0, A vs N — **expert prediction G0 "identical at x = 0": HELD**

At both 1× and 2×:

- **ESSO NL:** byte-identical on every node (5, 7 and 9). This holds for the plain file and for the labelled
  `.nl`, `.row` and `.col`. The plain sha256 is `834c9bf5…` on all three nodes, A and N. It is the same hash as the
  x = 0 ESSO NL in the W21 record `case_file_baseline_before.json` (C3 file).
- **Network NL:** byte-identical on all 48 sequential blocks (12 TSO + 36 DSO) and all 36 parallel-builder DSO blocks.
- **Component digests:**
  - Network blocks are identical: 0 differing components, both builders.
  - Each ESSO node differs in exactly one component, `Constraint:energy_storage_capacity_degradation`. 12 of its
    18 indices differ: the D row and the SoH row (φ coefficient) for each cohort-year. Every differing index is
    **inactive in both A and N** (0 active).
  - That is the expected deactivated difference, not a failure. No Param, Var, Expression or Objective differs.

### (ii) unit, A vs N (2×; 1× gives the same answers)

- **Network NL:** identical on all 48 sequential and all 36 parallel blocks. Digests are identical: 0 differing
  components.
- **ESSO NL:** differs on **node 7 only**. Nodes 5 and 9, which are at x = 0, are byte-identical.
  - Node 7 plain sha256: A `e3a0940f…`, N `d7b467e2…`.
- **Node 7 `.row`/`.col` analysis:**
  - **`.col` identical:** 1,197 columns, same names, same order.
  - `.row` identical: 1,200 rows.
  - **Rows with differing segments:** 6, all `energy_storage_capacity_degradation[*]`, a subset of that family, as
    predicted:
    - `[1]`, `[4]`, `[7]`: the D rows of the 2025 cohort; J and r differ.
    - `[2]`, `[5]`, `[8]`: the SoH rows, where φ^n = 0.927… becomes 1; C and r differ.
  - **Non-row segments that differ:** the header line 8 and the `k` segment only.
    - The Jacobian nonzero count drops from 3,251 to 3,248.
    - Each of `es_avg_ch_dch_per_unit[0,0]`, `[0,1]` and `[0,2]` loses one nonzero (2 → 1), because N's D row is
      `D == 0` and no longer references the throughput.
    - These are consequences of the row change, not other rows.
- **Component digests:** each node differs exactly in `{'Constraint:energy_storage_capacity_degradation'}`.
  - Node 7: 12 of 18 indices differ, and 6 of them are active (the 2025-cohort D and SoH rows).
  - Nodes 5 and 9: 12 differ, 0 active.

### (iii) 1× vs 2× within a model (all four (instance, model) pairs, both builders)

- TSO: all 12 blocks identical, NL and digest.
- ESSO: all 3 nodes identical, NL and digest.
- DSO: all 36 blocks differ (sequential and parallel builders alike), in exactly two components:
  `Expression:flex_cost_scenario` and `Objective:objective`.
  - The flexibility cost is baked into the expression as constants, and the objective string inlines it.
  - The coefficient-level statement "only the flex coefficients change, by exactly ×2" is backed by the harness's own
    W33 read-back. It ran in every configuration: 55,296 coefficients, `max_rel_dev` 0.0, everything else identical
    on probe builds.

### (iv) Determinism (two separate processes)

The child process rebuilt `x0_A_1x` and `unit_N_2x` from scratch, with its own guard and output in
`determinism_child/`.
- All 51 sequential and 36 parallel blocks of each configuration are **byte-identical** to the parent's. That covers
  the plain NL, the labelled NL, `.row`, `.col`, the block digest and every component digest: 0 differences.
- The child's exit code was 0 and its own checks all passed.
- The parent built those configurations after six others in the same process, so this also covers build-order
  independence within a process.

## Not confirmed

- **Run-time identity.** The answers cover the models handed to IPOPT at the **initialisation** solve.
  - The ADMM-cycle models (`update_*_to_admm`, the ESSO capacity updates through `get_updated_capacities`) were not
    built.
  - At the unit, the networks see the ESSO's SoH-degraded capacity only from cycle 1 onwards, so A and N will
    diverge there by design.
  - At x = 0 that capacity is 0 under both models; this is not measured here.
- **Process isolation in the parallel builder.** The parallel builder was called in-process, not through
  `ProcessPoolExecutor`; the pickle round-trip to a worker was not exercised.
- **Coefficient-level (iii) on the built blocks.** I looked at the DSO7 2025 Spring objective string by hand: the
  differing tokens I inspected were `flex_p_down`/`flex_q_down` coefficients at twice the 1× value. That was a manual
  look and is not preserved as a statistic. The preserved coefficient check is the W33 read-back on probe builds
  (above).
- **Determinism of the other six configurations across processes.** Only `x0_A_1x` and `unit_N_2x` were rebuilt in
  the separate process.
- **The `price_taker` initialisation path** (refused by every child) and the 'mid' SoH point were not built.
