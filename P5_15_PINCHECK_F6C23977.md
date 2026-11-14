# P5.15 pin check — does HEAD (production `f6c23977`) with every new switch at its default build spec v1's models? (Addendum 89, zero solves)

Zero solves. Model builds only, with `optimize` stubbed, as in W183-0 and W183c. No code, harness, spec, record, input,
`TASKS.md`, `REVISION_CONTEXT.md` or brief was edited. Objective convention: no objective value is reported. The models
compared are those IPOPT would receive at the ADMM initialisation solve.

## Blocked on Planner

Nothing blocks this check. Two points bear on the ruling the expert is asked for.

1. **Model identity does not clear the launcher's pins on its own.** From reading the code (the launcher was not
   run): `p515_s55_step7_launcher.pins_compare` (l. 793–818) compares four things against the frozen pins, and at
   HEAD all four differ:
   - the production head, `6f43479e` → `f6c23977`;
   - the three module hashes;
   - the input set. `f6c23977` added two **tracked files under `data/SRP1`**, which is the input definition (every
     tracked file under `data/SRP1` outside `Results/` and `Diagrams/`, `tracked_inputs()` l. 745):
     - `data/SRP1/SRP1_5x3y_1x1.json` (`f916e0c6…`);
     - `data/SRP1/SharedESS/SRP1_5x3y_ESS_Params.json` (`08ade261…`).

     The count is now 83 against spec v1's 81. The spec-v1 freeze script also asserts `N_SRP1_INPUTS`
     (`p515_s55_w183i_d_spec_v1_freeze.py` l. 324–329). The digest harness records the same: 81 inputs in the
     reference, 83 at HEAD, and the 81 shared inputs have identical sha256.

   So any route that lets W187 launch under spec v1 has to deal with the two new input files as well as the three
   modules. The options and the choice belong to the expert.
2. **The pickle block is mine.** The task asked for pickle to be blocked, and the committed digest harness does not
   block it (its pickle guard exists only in the W183c gate script). I did not edit the harness. Instead I installed a
   process-wide block through `PYTHONPATH` (`sitecustomize.py`, copied as evidence to
   `pincheck/sitecustomize_pickle_block.py`, sha256 `35fd281e…`).
   - It makes `pickle.load`, `pickle.loads`, `Unpickler` and `_Unpickler` raise.
   - It applied to the parent and, through the inherited environment, to the determinism child.
   - Both processes reported **0** refused calls at exit (`launch.log`, last line of each process).

## Changed

- **No code changed.** HEAD at the run was `f7c3f601`. Production equals `f6c23977`: the harness's
  `--expect-production-commit f6c23977` check refuses otherwise, and `git diff --quiet f6c23977 HEAD` over the 38
  production modules was also clean. The only production commit since `6f43479e` is `f6c23977`
  (`git log 6f43479e..HEAD` over those modules).
- Production module sha256 values at the run (spec v1 pins in brackets):
  - `shared_energy_storage_data.py` `e8bf4f92…` [`9acd095f…`]
  - `shared_energy_storage_parameters.py` `f31a2caa…` [`5fe0c4fb…`]
  - `shared_resources_planning.py` `c9336a61…` [`ade9c843…`]
  - `shared_energy_storage.py` `d45c72d0…` [`d45c72d0…`, equal]
- **Command** (attached, alone, stdout and stderr to `launch.log`; harness unchanged, `4308bc2f…`, committed in
  `6f43479e`):

  ```
  PYTHONPATH=<scratchpad>/pincheck_guard NLP_SOLVER_PATH=/usr/local/bin/ipopt \
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183_nl_digest.py \
    --out data/SRP1/Results/P515S55/pincheck_nl_head_f6c23977 --label pincheck_head_f6c23977 \
    --scratch <scratchpad>/pincheck_build --expect-production-commit f6c23977 \
    > data/SRP1/Results/P515S55/pincheck_nl_head_f6c23977/launch.log 2>&1
  ```

  - Exit 0. ALL_OK True. Wall clock 998 s.
  - The configurations were the 8 defaults: x0/unit × A/N × 1×/2×.
  - NRF was off (no `--nrf`).
  - The determinism child was the default (x0_A_1x, unit_N_2x).
- **Comparison** (a read-only script, no project import, no build): `pincheck/pincheck_compare.py` (`14ac0828…`), output
  `pincheck/pincheck_compare.json` (`961b8350…`).
  - The reference is read from the committed `w183_nl_post_w183c_off/w183_nl_digest.json.gz` (`ca195bd3…`, the file
    spec v1 pins) and from that run's committed determinism child.
  - The per-block equality is W183c gate 2's `_block_files_equal`, reproduced field for field: plain NL, labelled
    `.nl` / `.row` / `.col`, the block digest and every component digest.
- **AST off-path check** (read-only): `pincheck/pincheck_ast.py` (`175c7d42…`) → `pincheck/pincheck_ast.json`
  (`6d490e2e…`).
- **Committed in this commit** (by filename, `git add -f` under `data/`):
  - the report;
  - in `data/SRP1/Results/P515S55/pincheck_nl_head_f6c23977/`:
    - `w183_nl_digest.json.gz` (`315ca94a…`; gzip mtime 0, level 9, as W183c; the round-trip equals the raw file
      `9d373252…`, 109 MB, untracked);
    - `manifest_sha256.json` (`64ad41a0…`, 4,353 files);
    - `manifest_post_run_sha256.json` (`306521ec…`);
    - `launch.log` (`02a77b1e…`);
    - `determinism_child/{w183_nl_digest.json, manifest_sha256.json}`;
    - the 120 ESSO NL gz files (96 from the main run, 24 from the child);
    - `pincheck/` (the 5 files above).
  - The network NL gz files and the per-block item gz files are hash-recorded in the manifests and not committed,
    as in W183-0 and W183c.

## Found

### 1. Switches introduced by `f6c23977`

| Switch | Where read | Default | Default path |
|---|---|---|---|
| **C1** `salvage_value.in_objective` | `SalvageValueParameters.read_parameters` (key `in_objective` in the ESS-params file's `salvage_value` block; must be a JSON bool; with `enabled` false it raises). Read by `_build_subproblem` as `getattr(params.salvage_value, 'in_objective', False)` | **False** (`SalvageValueParameters.__init__`) | The `else` branch is the pre-change `Objective(expr=feasibility_penalty)` verbatim. No `salvage_objective_kappa`, `salvage_credit_obj` or `_salvage_throughput_coefficients` is constructed. **No expression, row, bound or objective term changes** |
| **C3** `available_energy_soh_point` (top-level key of the ESS-params file; 'end' or 'mid', otherwise it raises) | `shared_energy_storage_parameters._read_parameters_from_file` → `SharedEnergyStorageParameters.available_energy_soh_point`; copied to `SharedEnergyStorageData.available_energy_soh_point` in `read_parameters_from_file` **only if not None** | **None** (not declared) | Nothing is assigned. The data object keeps its own default 'end', and the W20 `model_variant` path is untouched. **No change** |
| κ plumbing: `update_shared_energy_storage_model_to_admm(..., objective_scale=None)` and the one production caller passing `objective_scale=objective_scale` | ADMM set-up, `_run_operational_planning` | The new block runs only `if hasattr(models[node_id], 'salvage_objective_kappa')` | Never entered with C1 off (the attribute is never built). No Param `admm_common_objective_scale` is added. **No change** |
| `_get_esso_complementarity_diagnostics` effective-marginal branch | diagnostic, after a solve | `if hasattr(model, 'salvage_objective_kappa')` | Never entered with C1 off. The old return dict is unchanged. Reporting only |

C4 is the five-block instance. It is two new data files, not a switch, and no SRP1 path reads them.

**Neither key is declared** in spec v1's pinned `SRP1_ESS_Params.json` (`39106f93…`, unchanged; `grep` over
`data/SRP1/*.json` and `data/SRP1/SharedESS/*.json` finds the keys only in the new `SRP1_5x3y_ESS_Params.json`).

**AST check** (`pincheck_ast.json`). Each changed function at HEAD was compared with the same function at `6f43479e`,
after replacing every switch-guarded `if` by its off branch and dropping docstrings:

| Function | Result |
|---|---|
| `_build_subproblem` | AST-equal |
| `_get_esso_complementarity_diagnostics` | AST-equal |
| `SharedEnergyStorageData.read_parameters_from_file` | AST-equal |
| `shared_energy_storage_parameters._read_parameters_from_file` | AST-equal |
| `update_shared_energy_storage_model_to_admm` | Body AST-equal; the signature gains `objective_scale=None` |
| `_run_operational_planning` | AST-equal once the new `objective_scale=` keyword of its one call is dropped |
| `SalvageValueParameters.__init__`, `SharedEnergyStorageParameters.__init__` | Old statements preserved in order; each adds exactly one assignment (`self.in_objective = False`; `self.available_energy_soh_point = None`) |
| `SalvageValueParameters.read_parameters` | Not AST-equal, by construction: it adds the `in_objective` read and type check and the `in_objective`-requires-`enabled` check. No pre-existing field's read is altered (diff read by hand) |

The only functions in `shared_resources_planning.py` whose source differs between `6f43479e` and HEAD are
`_run_operational_planning` and `update_shared_energy_storage_model_to_admm`.

**`f6c23977`'s own identity gate** (G-CC-1, `P5_16_CC_C1C3C4_REPORT.md`) claimed bitwise identity and held it, on a
narrower scope:
- configurations **x0_A_1x and unit_A_1x only**, both builders;
- against a reference built from the pre-edit HEAD `f9336940` (production = `6f43479e`);
- with a secondary match to `w183_nl_post_w183c_off`, on those two configurations.

It did not cover the N model or 2×. This check covers all 8.

### 2. NL and component-digest identity at HEAD vs spec v1's reference (`w183_nl_post_w183c_off`, production `6f43479e`)

**Verdict: IDENTICAL — 696 / 696 blocks.**

- **Coverage.** Each configuration has 51 sequential blocks (12 TSO, 36 DSO, 3 ESSO) and 36 parallel DSO blocks.
  Over all 8 configurations that is 87 blocks each, 696 in total.
- **Identity on every block** for every field compared: plain NL sha256, labelled `.nl` / `.row` / `.col` sha256, the
  block digest and every component digest. No block is missing on either side, and no component differs.
- **ESSO, every node, every configuration.** The plain NL is equal to the reference:
  - `834c9bf5…` on all x = 0 nodes, and on nodes 5 and 9 of the unit;
  - node 7 of the unit: `e3a0940f…` under A and `d7b467e2…` under N, the same hashes W183-0 recorded.

  The 120 committed ESSO NL gz files (plain, `.sym.nl`, `.sym.row`, `.sym.col`; main run and child) were also
  decompressed and re-hashed from the files themselves, on both sides: **120 / 120 equal**.
- **Gz payload check from the manifests.** Paths were re-rooted, and gzip is mtime 0, hence deterministic.
  - Main run: **4,350 / 4,350** NL and per-block-item gz files have identical sha256.
  - Child: **870 / 870** identical.
- **Determinism child** (x0_A_1x, unit_N_2x): **174 / 174** blocks identical to the reference run's child. Within
  the run, the child is byte-identical to the parent (the harness's own check, OK).
- **Eval keys.** All 8 are equal to the reference's: x0_A_1x `5cfe69a6…`, x0_A_2x `5f6b7841…`, x0_N_1x `105c0fd3…`,
  x0_N_2x `c15eea62…`, unit_A_1x `ca8927e7…`, unit_A_2x `bde9355e…`, unit_N_1x `054976fb…` (= ext-v6 base key, check
  OK), unit_N_2x `057f9a6f…`.
- **Harness checks all OK.** Capture paths, every build check (stub counts exact, W33 / W20 / W21 read-backs,
  `nrf_flags_as_declared`), no NL write errors, labelled = plain after the comment strip, imported repo modules
  unchanged during the run, eval keys distinct, determinism byte-identical, child exit 0.
- **Guards.** `SolveProfileGuard(permitted=())` was verified at 0 in the parent and in the child: permitted / blocked
  solve and exec all 0, no failures. The pickle block refused 0 calls in both processes.

### 3. Q aggregation (`shared_resources_planning._get_operational_recourse_components`)

- **The function source is identical** at HEAD and at `6f43479e`: AST-equal, source sha256 `6a7aa561…` on both.
- Q is built from four things:
  - the TSO and DSO `get_primal_value`;
  - minus the contracted interface settlement;
  - minus the voltage-pin blocks;
  - minus `shared_ess_data.get_salvage_value(models['esso'])`, which sums `pe.value(models[n].salvage_value)` over
    the ESSO nodes. `get_salvage_value` is unchanged.
- **It never reads the ESSO objective.** So the C1 term (κ · `salvage_credit_obj`, inside the ESSO `objective` only)
  could not enter Q even with the switch on.
- At the default the term is not constructed at all (section 1), so **salvage C1 does not enter Q**.
- `net_operational_recourse = gross_operational_cost − terminal_salvage_value` is unchanged. The salvage is
  reported, not optimised, as at `6f43479e`.

## Unexpected findings

- **The run left 22 untracked log directories** under `data/SRP1/Results/P56A/evals/`, named
  `p515s55_w183nl_pincheck_head_f6c23977_20261110T135034_*`. They come from `_construct_arm_planning` and follow the
  same convention as W183-0 and W183c. I left them in place and untracked.
- **The P5.16 report's fixture note may matter for W187.** `P5_16_CC_C1C3C4_REPORT.md` records 112 / 139 fixtures
  unpickling on that machine. It records no new failure caused by `f6c23977`. I did not re-check fixtures here,
  because pickle was blocked by order.

## Remaining issues

- The expert's ruling on the launch route (section "Blocked on Planner" 1) is still open; this check does not choose
  one.

## Not confirmed

- **ADMM-cycle models.** As in W183-0, only the initialisation models were built.
  - The models after `update_*_to_admm` and the per-cycle ESSO capacity updates were not built.
  - The statement that `update_shared_energy_storage_model_to_admm` is unchanged at the default rests on the AST
    check: body AST-equal once the `hasattr(..., 'salvage_objective_kappa')` block is removed, and that attribute is
    absent at the default.
- **Launcher refusal set.** I did not run `p515_s55_step7_launcher` or `p515_s55_e1_launcher`, even in dry-run, so
  the exact list of `PIN DIFFERS` lines they would print is inferred from `pins_compare` and `tracked_inputs()`
  (read: `p515_s55_step7_launcher.py` l. 745–818, 1575–1600; `p515_s55_w183i_d_spec_v1_freeze.py` l. 300–335).
  The e1 launcher's `pins_compare` (l. 908) was located but not read in full.
- **Parallel builder through `ProcessPoolExecutor`.** It was called in-process, as in W183-0.
- **Other params files.** Whether any non-SRP1 params file declares the new keys was not searched; only `data/SRP1`
  was searched.
- **The 'mid' path and C1 on** were not built here. That is G-CC-2's scope, outside this check.
