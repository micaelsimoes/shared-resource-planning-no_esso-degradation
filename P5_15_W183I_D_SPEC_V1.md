# P5.15 W183i (d): Step-7 campaign spec v1, frozen on the patched harness; the real E3 / E2 freezes; the E3 preconditions-only run

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 86, W183i order item (d) and rulings 2–3; Addendum 85, W183g (d) and
rulings 2, 3, 4 and 9; Addendum 83, the campaign spec v1 paragraph. Planner task W183i (d).

**Solves: 0.** Every process was guarded:
- My script ran under `SolveProfileGuard(permitted=())`, installed before any project import. All 9 reachable guards
  were verified at exactly 0, at the write and at the end, in both runs (`--freeze-spec`, `--post-manifest`).
- The launcher processes (two `--freeze` runs, one `--preconditions-only` run) each ended `final guards: all_zero=True`.
- `pickle.load` / `loads` were blocked and called 0 times, and `ConcreteModel.__init__` was tripwired and called 0
  times, in every process. The 52 E1-launcher tripwires stayed at 0.
- No `--run` without `--preconditions-only` was executed. The screen, E1 and batch roots were **not** frozen.

Objective convention, which the spec declares (no objective value is computed here): Q = `gross_operational_cost`,
contracted interface settlement excluded.

## Blocked on Planner

Nothing blocks W184. The E3 cell's preconditions all hold. Four points need the Planner's reading.

1. **Three commits instead of two.** `--run --preconditions-only` refuses unless the stage spec and cell spec are
   committed and clean. So I committed the spec v1 document and the frozen roots first (`d8dcc139`), then ran the
   preconditions, then committed the log and this report (commit 3).
2. **Addendum 83's "closure/σ/circle/SOC/interface diagnostics" are not all recorded scalars.** Spec v1 maps each item
   to the writer that captures it (`capture_checklists.spec_required_quantities_addendum83`, 19/19 present). The
   Planner should know what kind of capture each one is:
   - **Recorded at terminal:**
     - the closure slack, per block in `component_levels_terminal.json`, from `_s31_block_components`
       `shared_ess_day_balance_slack`;
     - the interface residual ratios, in `evaluation_record.boyd_terminal_ratio_max_per_channel` and
       `boyd_terminal.json`;
     - the interface settlement, in `interface_settlement_detail_s31c.json`.
   - **Recoverable only from persisted models, by a guarded pickle read** (no recorded scalar exists): the
     converter-circle residual, max |P^E| / S^Av and SOC within limits.
     - E3 and E2 have `certified_models.pkl` (persist on) and `esso_models_s39_D.pkl`.
     - Screen and E1 cells have `esso_models_s39_D.pkl` only. These are ESSO quantities, so that file is enough.
   - **"ESSO σ slack"** is not a quantity name anywhere in the code. In production `.py`, "sigma" appears only as the
     ADMM objective scale. I mapped it to "any ESSO slack in the persisted ESSO models" and did not identify it further.

   Plan §4 says these checks are "already captured by the harness hooks; the pre-run checklist assertion lists them".
   That is true for the first group only.
   - **Options:** (a) accept the persisted models as the capture path. W184 and W185 then need guarded pickle reads
     after their runs; Addendum 85 ruling 6 already lets a guarded-read authorization cover as many reads as the task
     needs. (b) Add recorded scalars. That would be a harness change after the spec freeze, so a new spec version.
   - **Recommendation:** (a). Nothing is lost, because the models are serialized. The C\* failure mode, where the
     models were not kept, does not apply.
3. **Carried over, not ruled in this task.** The screen's `x0_A_1x` key `5cfe69a6…` still has 1 committed holder:
   W86 `…tail_recert_ddd6cd44.json`. It is reported and not reused (W183i (b) Blocked 1). W183i (b) Blocked 2–4 and
   W183i (c) decisions 1–6 are recorded in spec v1 as implemented, and I made no choice on them.
4. **A benign self-reference.** The post-preconditions manifest hashes its own launch log at the empty-file value
   `e3b0c442…`, because the log was still being written. The log's final sha256 is in the Evidence table below.

## Changed

| commit | content |
|---|---|
| `2a62ee1d` | `p515_s55_w183i_d_spec_v1_freeze.py` (new; sha256 `acc94ee2fdaec3d99bf3ac163dc86f0c7ca0de9805c5c1380f168a817f95df55`), committed before it ran on the repo. One smoke run before the commit went into the session scratchpad. |
| `d8dcc139` | spec v1 document, the frozen E3 / E2 roots (specs and freeze logs), the spec-v1 launch log and manifest |
| commit 3 | `run_e3_cnrf_x0_1x_preconditions_only.log`, the post-preconditions manifest and its log, this report |

No launcher, harness, hooks, production file or committed artifact was edited. `TASKS.md`, `REVISION_CONTEXT.md` and
the brief are untouched and unstaged. Nothing was overwritten: every write was write-once (`'x'` / `noclobber`).

### Spec v1

- **`data/SRP1/Results/P515S55/frozen_step7_campaign_spec_v1_96345241.json`**, sha256
  **`963452416fd4882bc14b0f5a28d5cef7bccf93d0ff57860292e9d6a91220f3c0`**.
- Schema `p515_s55_step7_campaign_spec_v1`, written once through `gate_result_io`. The name carries its own sha8.
- **No predecessor.** Searched: `git ls-files` for `frozen_*campaign_spec*` and `frozen_step7*` (0 hits),
  `P515S55/`, and the reports `P5_15_W183_REPORT.md` ("Spec v1 is not frozen"), `P5_15_W183F_BITWISE_GATE.md` and
  `P5_15_W183G_REPORT.md` (W183g (d) held).
- It was written at HEAD `2a62ee1d`, after both launcher freezes.

### Pinned hashes (each re-hashed from the working tree; all git-tracked and clean)

| item | sha256 |
|---|---|
| production head (38 modules, each sha256 in `pins.production.modules_sha256`) | `6f43479ed8ece7a4a5878942ca38952ce1cae1bc` |
| harness `p515_s44_campaign_harness.py` (patched) | `d9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191` |
| gates `p515_g_g1_g4_admm_gates.py` | `6f4631e10a2f8a7786390c0f96e45332706779ae9cdf6a87ddd716d9ad9ad346` |
| hooks `p515_s55_step7_resettle_hooks.py` | `9318ade3e6d0d5fec25d0c276c4d82fba5bb163c7da61ec10ee6f49aaf2fa1d2` |
| Step-7 launcher `p515_s55_step7_launcher.py` | `3b990d19bbde8e9fb6bf75d2e5447c55116d4a3d4d21c174e53776dffc561757` |
| E1 launcher `p515_s55_e1_launcher.py` v3 | `532cf6e7bf03b5cd794843614ae5a4c19ae06485d750305a90a5cd49aea76fa5` |
| `p56a_oracle.py` | `699d571a2d69b14fd89024c535fa8b3ffa0625aa647f1714060dcc9cf6fa2f11` |
| `settling_criterion.py` / v2 / v3 | `fbe5550b…f65f` / `3db13b0e…f59e` / `0d78b922…b965` |
| `settling_criterion_v4.py` / v5 / v6 | `077561ce…7775` / `5ea29fb3…976f` / `53f47711…6e2d` |
| `p515_s53_f2_certificate.py` | `59f75ce46b42f41053478f8ea50e495555259ef6b92b91766aeda1e013d21df6` |
| `p515_s47_phase_b_record.py` / `gate_result_io.py` | `97f573e5…6c62` / `26cadaa5…d07d` |
| 81 SRP1 inputs (each in `pins.srp1_inputs`): case `SRP1_params.json` / ESS params / cost `SRP1_ESS.xlsx` | `dbfdb2a0…006b` / `39106f93…2706` (= PB pin) / `e17bd588…39cd6` (= PB pin) |
| W2 table `P515S45/investment_cost/investment_cost_results.json` | `28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b` |
| Phase A tables `P515S45/phase_a_tables/phase_a_tables.json` | `a2b78541b5c50ad2a185ca3eb3d0f1b38bb388c4866c45628a984c407b4a3995` |
| G0 artifact `P515S55/w183_nl_post_w183c_off/w183_nl_digest.json.gz` | `ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a` |
| G0 record `P515S55/w183d_g0/g0.json` (G0_PASS True) | `8cef45d684bc7e5dc8b719fd504b8e7705933597a9ba07f44349650046b2911c` |
| ruled two-cycle gate record `P515S55/w183i_srp1_bitwise_gate/w183i_wrapper_record.json` (GATE_PASS False as frozen, **GATE_PASS_RULED True**) | `06b9cdc3bf7fed9256d2ecf4cd86b978356363c5c0599c58350befc1b2b95350` |
| `data/SRP1/SRP1.json` (**`ParallelExecution: false`**) | `61a794a7ce7a3fb983f2e92128eec446dd7dbb17ad75a3b3b1c5735f8bd4e4ef` |
| solver `/usr/local/bin/ipopt` (`NLP_SOLVER_PATH` from `.env` resolves to it) | `b316abbe38cd851d9a9a87a6a1a1714a4334304c9539df1976d2f7101360ff05` |
| brief / plan (the verbatim text sources) | `8b83aeb5…d1eb` / `18593428…acfb` |

Both launchers' complete pin sets are recorded as each launcher computes them: `pins.step7_launcher_pins_now`, and
`pins.e1_launcher_pins_now_refreshed_at_spec_v1`, which is the E1 launcher's pins refreshed at freeze.
`pins_recorded_check` holds on both. One limit: the `imported_repo_modules` part of each set belongs to this process,
which imports both launchers. Each phase freeze records its own.

### The real freezes (write-once; Step-7 launcher, first real use; both exit 0)

Command (e2 likewise into `e2_rho`):

```
mkdir data/SRP1/Results/P515S55/e3_cnrf && set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_step7_launcher.py --freeze --phase e3 \
  > data/SRP1/Results/P515S55/e3_cnrf/freeze_e3_launch.log 2>&1
```

| phase | stage spec | sha256 |
|---|---|---|
| e3 | `e3_cnrf/frozen_s55_step7_e3_spec_v1_bd2cfa4d.json` | `bd2cfa4d5503cdc440f01c9e2370b00ce22dd13833ef9e4d8c3d04233a465b46` |
| e2 | `e2_rho/frozen_s55_step7_e2_spec_v1_7631a6d7.json` | `7631a6d7c28cfdde5fd3d7fa4ca6db0da29b3a784aa2b08491465a897ae66e66` |

| cell (launch order) | cell spec sha256 | eval key |
|---|---|---|
| `e3_cnrf_x0_1x` | `1363bceffe198e331fb0c7cc770016acb6dbc33107c24b41344cc935cc129995` | `35b275180d897e4c…` |
| `e2_x0_1x_rho0p5` | `b156e6b47fc24be770b691c3a5b162546beacf5b86677d9e251a9d7de5650bfc` | `64f3605028f1cf87…` |
| `e2_unit_1x_rho0p5` | `e93013a06262a7fa65ce0914c1d030b149e12ae2b2e4ea8c0d4c2a6ad987fcc3` | `a895b40ec0b7ccdb…` |
| `e2_x0_1x_rho2p0` | `cbff6eb34827c009d7ddd91459a808b5c0bd1c0499f6b93416b2ad28a7eccc18` | `f4df38b251a68808…` |
| `e2_unit_1x_rho2p0` | `d7045326df5fa35ee9a0da113fdbc22301309b0cf36a9fdca3cdfb20ba342fe1` | `13145d7ed63e9f2d…` |
| `e2_x0_2x_rho0p5` | `989603350f14c58353105b6062f0b301fab505a6151dd24825df1b74e2a2bef1` | `14a161e63a4ee34f…` |
| `e2_unit_2x_rho0p5` | `476201fb4a58f30925078b5b3ef0fe4ebbaf475d1640455085546475feb06bbe` | `3e35c66844450f94…` |
| `e2_x0_2x_rho2p0` | `fe86fe098dec7be9c22b873741e97573e7b0928f12167dc25bda22f99b1b217e` | `16614c6680f4d3b4…` |
| `e2_unit_2x_rho2p0` | `37ad9cb3e44dd7b2b93289b4ba02fa6aceb507159d3744eefabaac77448b7d19` | `43074041a7942052…` |

All nine cells share these properties:
- cap 300;
- `post_certification` = {persist_certified_models True, hull_polish False, reference None};
- 44/44 `validate_campaign_spec` checks;
- the pre-launch assertion holds;
- each campaign root holds only its spec.

**The E3 launch command** (`launch_commands.e3`; the W184 command, not run here):

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_step7_launcher.py --run --phase e3 --cell e3_cnrf_x0_1x --spec-sha256 1363bceffe198e331fb0c7cc770016acb6dbc33107c24b41344cc935cc129995 > data/SRP1/Results/P515S55/e3_cnrf/run_e3_cnrf_x0_1x_launch.log 2>&1
```

Spec v1 also records:
- the eight E2 launch commands;
- each cell's preconditions-only command;
- the `--report` commands of both phases.

### What spec v1 declares

Each declaration below is extracted **verbatim** by line range from the brief or the plan, with each range's first and
last line checked against a marker and the source file's sha256 recorded. Each is paired with the code constant it
binds.

- **Roots and their phases.**
  - Frozen now: `P515S55/e3_cnrf`, `P515S55/e2_rho`.
  - Declared and frozen after Addendum 87:
    - `P515S55/e1_2x`, with phases seeds, search_A, search_N and cross. Its campaign directories are
      `campaign_s55_e1_2x_{seeds,search_N,search_A}`, and cross is `P515S55/e1_2x/cross`.
    - `P515S55/e1_2x_batch4`, with seeds only at concurrency 4.
    - `P515S55/e1_screen_1x`.
- **B+.**
  - Text: Addendum 85 ruling 2, l. 3964–3998.
  - The launcher's `B_PLUS_RULE`, and the asserted constants: window 10, k₀ 0, 30 new evaluations and 10 polls per
    arm, cap 30, n + 1 = 8.
  - The five confirmed choices (Addendum 86 ruling 3 (iv)), each with its implementation: declared roots
    {e1_2x, e1_2x_batch4}, dry-run file `e1_launcher_v3_dryrun.json`, `polled_index` in `ADVANCE_RULE_FIELDS`.
- **A7.** Text: l. 4000–4001. `PB.BARRIER_STOP_PER_POLL` 2 and `PB.BARRIER_STOP_OVERALL` 3.
- **The ρ read-back.**
  - The ruling's text: Addendum 86 ruling 2, l. 4136–4143.
  - The replication formula as implemented: `RHO_RULE_TEXT`, ×1.5 and ÷1.5, clamp [1e-4, 1e4], and the factor-1
    action regex.
  - The W183f validation target.
- **Addendum 86 ruling 3.** Text: l. 4145–4166.
  - E2 runs all 8 cells, in the order x0 then unit within each pair. H2 pairs are evaluated in the order (1×, 0.5),
    (1×, 2.0), (2×, 0.5), (2×, 2.0).
  - An uncertified E2 cell is not a stop. A failed pair is recorded and E2 continues.
  - Persist is **on** for E3 and E2. It is **off** for the screen and for E1 seeds and search: no `post_certification`
    on any entry (asserted), cap 500.
  - E1 launches are resumable, unit = batch (default asserted), with a commit between launches (the git condition in
    `RESUME_RULE`, asserted).
- **Counts and limits.** Addendum 85 ruling 9, l. 4053–4055.
- **Hypotheses**, verbatim:
  - H1a–c: plan l. 165–167.
  - H1d: plan l. 171–177.
  - H2: plan l. 119–121, with the Addendum 85 W185 line and the per-pair rule.
  - H3a–b: plan l. 87–91, with the Addendum 85 W184 line.
  - H3c: the Addendum 85 W184 line (l. 4075–4086) and plan §12 (l. 310–314).
- **Stops per phase.**
  - E3 (exit 3): not certified within the cap; NRF terminal not written or `nrf_capture_error`; n_active ≠ 864 TSO or
    ≠ 288 per DSO.
  - E2 (exit 3): ρ read-back mismatch, a harness-fault stop. H2 failure is recorded and E2 continues; the chain stops
    before E1.
  - Screen: none declared; a barrier is a fault.
  - E1: A7, completion-cap exhaustion, budget or max polls, and a harness fault stopping the arm.
- **The convention.** Addendum 83 ruling 2, l. 3785–3789.

## Found

There is no recorded prediction for W183i (d). Each condition the task names was asserted, and a failed assertion
would have refused the write. 0 refusals.

### Key equality with the committed dry runs: held, every phase

| phase | recomputed by | count | compared with | result |
|---|---|---|---|---|
| E3 | `L7.cell_table` + `key_checks` | 1 | `w183i_b_step7_dryrun.json` `f4bc728c` `key_tables.e3` | equal on 10 fields (eval / base / candidate key, key_without, checks, options) |
| E2 | same | 8 (pair order = `S7.E2_CELLS`) | `key_tables.e2` | 8/8 equal |
| screen | same | 15 (x0 under A + 14 under N) | `key_tables.screen` | 15/15 equal |
| screen alias `x0_N_1x` | `L7.screen_alias` (G0 with current production) | 1 | `screen_alias` | `105c0fd3…` equal; G0 accepted |
| E1 seeds | `E1.seed_plan` + `seed_key_assertions` + `key_table` | 7 + 1 alias | `e1_launcher_v3_dryrun.json` `1256f63b` `key_table` (cited by the gate JSON) | 8/8 rows equal on 7 fields; 11/11 seed assertions |
| E1 seeds | same | 7 + 1 alias | W183i (c) gate JSON `201771e6` `item5.seed_key_summary` | keys_to_evaluate (in order), n = 7, 1 alias, alias key `c15eea62…`, alias target `5f6b7841…`: 5/5 |
| E1 G0 alias (2×) | `E1.g0_check(require_current_production=True)` | — | — | accepted |
| E1 batch 4 | seeds rows `unit_A_2x`, `unit_N_2x`, `n7_05_2_A_2x`, `n7_05_2_N_2x` | 4 | — | `bde9355e…`, `057f9a6f…`, `933a5297…`, `e9f793f6…`, each a seed key |
| E1 search domain | `E1.make_arm_key_of` per arm | 1,705 per arm, disjoint | dry-run `domain_keys` | equal; sorted-key sha256 A `900c74bf…`, N `23066af0…` |
| cross | `L7.cell_table('cross')` | 0 | — | declared only |

The E1 seeds to evaluate are `5f6b7841`, `bde9355e`, `057f9a6f`, `933a5297`, `e9f793f6`, `c4dea62e` and `c011496d`.

Committed holders were scanned over the 263 git-tracked `data/*campaign_spec_*.json`, excluding the four Step-7 stage
roots:
- E3 / E2 keys: 0 holders;
- screen: the 1 holder above;
- E1 seed keys: 0 holders.

### Capture checklist (rule eleven), asserted present: held, every phase

| phase | checklist | items |
|---|---|---|
| E3 (on the frozen cell spec) | `H.assert_record_capture_paths` 46; `H.assert_nrf_capture_path` 13; `S7.assert_resettle_preconditions` 116; `H.assert_post_certification_capture_paths` 10 | all true |
| E2 (each of the 8 frozen specs) | 46; ρ wiring 18; 116; post-certification 10 | all true |
| screen (x0_A_1x and one N cell; phase-level) | 46 | all true |
| E1 seeds / search_N / search_A / batch4 | `E1.rule_eleven` 33 checks (+ the 46 harness items), synthetic loops, no solve | all true |
| spec-required quantities (Addendum 83) | 19 items mapped to writers by source inspection (Q, status, end / certification cycle, k\* / band / gap from `resettle_decision.json`, bar, stopping ratio, candidate and spec hash, code / harness hashes, interface ratios, settlement, closure slack, ESSO models persisted, ESSO per-cycle capture, certified models E3/E2, NRF binding hours E3, E^Av and terminal SoH E1, ρ trajectory E2, per-cycle record) | 19/19 present; see Blocked 2 for which are scalars |

### ρ read-back rule validated at spec v1

The launcher's own `rho_validation_w183f` was run on the committed W183f arm, which is at s = 1.
- Cycle 1 matched bitwise on every channel:
  - pf 0.1980000000000002, n = 48;
  - ess 0.010000000000000004, n = 51;
  - v 0.011549999999999998, n = 48.
- Every cycle reproduced bitwise.
- The model counts equal the rule-eleven block counts.

### `--run --phase e3 --cell e3_cnrf_x0_1x --preconditions-only`: every precondition holds (exit 0, zero solves)

Command: the frozen preconditions-only command, run after `d8dcc139`. Log:
`P515S55/e3_cnrf/run_e3_cnrf_x0_1x_preconditions_only.log`.

The launcher logs only failures and then one summary line. An empty failure list therefore means every check below
held:

| # | precondition (`run_preconditions`, in order) | result |
|---|---|---|
| 1 | phase is not `cross` | held |
| 2 | the base is the declared root `P515S55/e3_cnrf` | held |
| 3 | the lattice builds on the pinned W2 / Phase A tables (σ_Q 18,449.66) | held |
| 4 | the cell is a cell of e3 | held |
| 5 | the stage spec is unique, self-hashing, committed and clean | held: `frozen_s55_step7_e3_spec_v1_bd2cfa4d.json` |
| 6 | the stage spec pins the given sha256 for the cell | held: `1363bcef…` |
| 7 | the code sha256 is unchanged since the stage spec froze | held |
| 8 | launch order (no earlier cell without results) | held: index 1 of 1 |
| 9 | the cell has no results yet | held |
| 10 | the campaign root exists, and the frozen spec loads at its sha256 | held |
| 11 | the campaign root holds only its frozen spec | held (re-checked after the run: only the spec) |
| 12 | the campaign spec is committed and clean | held |
| 13 | `H.check_campaign_preconditions`: no legacy lock, no campaign lock, no forbidden live process, production and harness files clean in git | held |
| 14 | `validate_campaign_spec`, 44 checks | held, 0 failing |
| 15 | the pins of the harness (spec, and = the patched d9a944f2), hooks 9318ade3, the launcher, the case file, the ESS params file and the cost file | held |
| 16 | `pins_compare`: every frozen pin against now (production head and modules, run files, imported modules, the 81 inputs, tables, committed records, G0) | held: 0 diffs |
| 17 | the pre-launch assertion: key recomputes, no holder, working dirs absent, frozen entry = recomputed | held |
| 18 | the parent capture checklist (46 + 13 + 116) | held |
| 19 | the solver path / binary equals the frozen one | held: `/usr/local/bin/ipopt` `b316abbe…` |
| 20 | G0 (screen x0 only) | not applicable |
| 21 | memory preflight, gating at concurrency 1 | held: 16.89 GiB ≥ 2.75 GiB |
| 22 | no other copy of the launcher alive | held |

The run stopped before the campaign lock and the child (it logged "STOPPED before the campaign lock and the child
(zero solves)"), and ended `final guards: all_zero=True`.

### Evidence

| file | sha256 |
|---|---|
| `P515S55/frozen_step7_campaign_spec_v1_96345241.json` | `963452416fd4882bc14b0f5a28d5cef7bccf93d0ff57860292e9d6a91220f3c0` |
| `P515S55/w183i_d_spec_v1/launch.log` (the spec-v1 run, both streams) | `61e552375d4ec66a84cdb48bdafcd321cfda9c869ad4a57156d4afe92a1ac6bb` |
| `P515S55/w183i_d_spec_v1/w183i_d_spec_v1_manifest_sha256.json` (spec v1, the 13 frozen-root files, the script) | `0ba1ff98452bc804d293bb0ed2b2ea044052ac5c64e7a1d5c887470bf0e9f664` |
| `P515S55/e3_cnrf/run_e3_cnrf_x0_1x_preconditions_only.log` | `8291f44cf49cf58943225262d79cccc68edae58ae486fd5592f386d8b3cdb911` |
| `P515S55/w183i_d_spec_v1/w183i_d_post_preconditions_manifest_sha256.json` (19 files, including the log above) | `a0285d9e1b505b345e5ec5f8f910c70f3908be91692e9c58aefea7c139499752` |
| `P515S55/w183i_d_spec_v1/post_manifest_launch.log` | `6110517da0a39751ff4369d5f2759383ea297b798a0079bda489c774e58908be` |
| `P515S55/e3_cnrf/freeze_e3_launch.log`, `P515S55/e2_rho/freeze_e2_launch.log` | in both manifests |

The spec-v1 command:

```
mkdir data/SRP1/Results/P515S55/w183i_d_spec_v1 && set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183i_d_spec_v1_freeze.py --freeze-spec \
  > data/SRP1/Results/P515S55/w183i_d_spec_v1/launch.log 2>&1
```

It exited 0 in 11.6 s. Then `--post-manifest` ran into `post_manifest_launch.log` and exited 0.

**Smoke runs before `2a62ee1d`.** Two `--smoke` runs went into the session scratchpad, outside the repository.
- The first refused with nothing written: my B+ line range began one line early, on a blank line. I fixed the range
  to 3964–3998.
- The second passed with 0 refusals.

The repository was not written by either.

## Not confirmed

- **The run path past the preconditions** (`H.evaluate`, the campaign lock, the results write, `cell_outcome`'s stops)
  has not executed. W184 is its first exercise.
- **Recovering the circle / SOC / P^E diagnostics from the persisted pickles** (Blocked 2) was not tested: no pickle
  was read. That they are recoverable rests on `run_admm_arm` dumping the ESSO models and on the harness's
  post-certification persist, both checked by source inspection only.
- **"ESSO σ slack" was not identified in the code.** Searched: production `.py` for "sigma"; the harness, the gates
  module and the v3/v5/v6/ext-v6/W105/W118 hooks for "closure", "_slack", "circle"; all `*.md` for "σ slack".
- **The holder scan covers git-tracked `data/*campaign_spec_*.json` only.** Untracked specs and other file types were
  not searched.
- **The E1 freeze and resume paths on real roots** were not run; they wait for Addendum 87. E1's pins in spec v1 are
  the refreshed current pins, and the E1 freezes will record their own.
- **The verbatim extracts are line ranges of the brief at `8b83aeb5…` and the plan at `18593428…`.** A later edit to
  either file does not change what spec v1 holds, but the line numbers refer to those versions.
