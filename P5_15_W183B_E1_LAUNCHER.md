# P5.15 W183b — E1 search launcher and seed-key dry run (Addendum 83 ruling 3 (i), (v), (vi); W183 order item W183b)

Zero solves; no model build; pickle blocked. Production modules, `p515_s44_campaign_harness.py`, case / params / cost
files, earlier specs and frozen files, TASKS.md, REVISION_CONTEXT.md and the brief were not touched. `--freeze` and
`--run` were not executed on the repository.

Objective convention: no objective value is computed or reported here. Where the launcher defines F it is
F = I + Q, with Q = `gross_operational_cost`, contracted interface settlement excluded.

## Blocked on Planner

Nothing blocks the commit. Decisions 1–3 must be made before W186 launches E1, and decision 1 changes what E1 can
deliver.

1. **As frozen, variant B stops both searches for review at poll 0 unless the incumbent is x = 0.** This is a
   zero-solve finding: `C.build_poll` was run unchanged at the launcher's Halton start, k = 0 (t = 17).
   - With any of the three storage seeds as incumbent, only 6–7 distinct feasible poll points remain after snapping
     (fewer than n + 1 = 8). That triggers the unit-box completion, which then exceeds the cap of 30:

     | Seed | Completion size | First poll |
     |---|---:|---|
     | unit | 31 | refused, STOP_FOR_REVIEW |
     | n7 0.5/2 | 34 | refused, STOP_FOR_REVIEW |
     | hist_inc | 61 | refused, STOP_FOR_REVIEW |
     | x = 0 | 14 | runs, 14 new evaluations |

   - A and N give identical poll geometry; only the keys differ.
   - At 2× the unit's committed margin is positive (+46.3 k€ in the manuscript), so the initial incumbent is likely
     a storage seed in both arms. Which seed it is depends on the seed evaluations.
   - The result depends on k. For the same incumbent at k = 0…9 (the table is in the JSON):
     - hist_inc refuses at k = 0 and k = 8 only. At k = 4 it has 10 distinct points and no completion, which is the
       historical F2-certificate poll at t = 21.
     - unit has no completion only at k = 2.
     - n7 0.5/2 has no completion only at k = 2 and 3.
   - Options, for the expert:
     - (a) accept a stop for review at poll 0. E1 then returns its seeds only, at a cost of 7 seed evaluations.
     - (b) change the completion rule or the cap of 30 for E1. This changes the method.
     - (c) choose the Halton start. Choosing it after seeing this table is outcome-driven and needs an explicit ruling.
   - I make no recommendation on the method. Factually, (c) chosen per incumbent would differ between the arms only if
     their incumbents differ.
2. **The barrier stop rule.** The launcher uses the inherited Phase B rule A7 by import: stop after 2 new barrier
   evaluations in one poll or 3 overall (`PB.BARRIER_STOP_*`, unchanged inside `C.run_continuation`). Plan §9 says
   "stop for review on a barrier point". Please confirm A7, or rule on the stricter reading. The stricter reading
   needs a launcher-owned loop instead of the imported `run_continuation`.
3. **Four choices I made that the order does not state.** Please confirm each.
   - (i) The Halton start is k0 = 0 for each arm. F2 continued the s51 counter at k = 4.
   - (ii) Arm A declares **no** `model_variant`, because absent is the baseline in the harness. Declaring the baseline
     values as a variant would put `model_variant` into every A key and break key equality with rows 35 and 40, which
     the dry run shows to be exact. So "model_variant on every entry" holds as: N entries carry the variant and its
     label; A entries carry the baseline by absence. Every entry carries `flex_price_multiplier` = 2.0 and its label.
   - (iii) Seed and search entries have **no post-certification**, as in the Phase B searches. The ext-v6 entries
     persisted the certified models.
   - (iv) Ruling 3 (v) says the batch gate re-runs "the same four", and plan §11 says "the four node-7 seeds".
     However, 6 of the 7 evaluated seeds carry node-7 storage. The launcher accepts any subset (`--seeds`), but which
     four is for the Planner. The per-cycle bitwise comparison script of the batch gate is not part of this launcher.
4. **G0 for `--freeze` / `--run` needs W183d's artifact.**
   - The W183-0 reference passes every alias check (below), but its production differs from the current production
     in `network_parameters.py` and `shared_resources_planning.py` (W183c). So `--freeze` and `--run` would refuse
     the alias today, as designed.
   - The reader expects W183-0's schema: `summary.i_x0_A_2x_vs_x0_N_2x`, `summary.eval_keys`,
     `configuration.spec_like`, `configuration.models`, `production.modules_sha256`. If W183d writes another schema,
     `g0_check` must follow.
5. **Not built by this launcher:**
   - the rule-v6 certified cross-evaluations (x_A and x_N under A and N);
   - the 1× unit-neighbourhood screen (`e1_screen_1x`, 14 fixed points under N at 1×).

   Both need declarations that the order does not give (a re-settle declaration; the x = 0 reference at 1×). The root
   argument accepts their spec v1 root names, but those roots have no phase here.
6. **How I applied the root rule.**
   - "Refuse an existing non-empty root" is enforced for `--dry-run` and for the first phase (`seeds`): only launch
     logs may be present.
   - A search phase is accepted only when the root holds exactly:
     - the **completed** seeds phase;
     - at most the completed other arm;
     - launch logs.

     This is because both arms must sit in the root that holds their seeds.

## Changed

- **`cb36b22e`** — `p515_s55_e1_launcher.py`, sha256 `320d503cca1bb68736ca0ddac78a6927f4701a44cec6ee8b035cb143ee1d6be7`,
  committed before the dry run. It is a new file; no other file was changed.
- **This commit** — this report, plus `data/SRP1/Results/P515S55/w183b_e1_dryrun/`:

  | File | sha256 |
  |---|---|
  | `w183b_e1_dryrun.json` (written through `gate_result_io`, write-once) | `e56a534c…afb3` |
  | `w183b_e1_dryrun_manifest_sha256.json` | `95279990…5779` |
  | `launch.log` (stdout + stderr) | `87fa5f5e…5141` |
  | `manifest_post_run_sha256.json` (the three files above and the launcher) | — |

- **Command.** Run attached and alone; exit 0; 10.1 s; HEAD `cb36b22e`; production head `6f43479e`:

  ```
  mkdir data/SRP1/Results/P515S55/w183b_e1_dryrun && NLP_SOLVER_PATH=/usr/local/bin/ipopt \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_e1_launcher.py --dry-run \
    --root data/SRP1/Results/P515S55/w183b_e1_dryrun --concurrency 1 \
    --g0-artifact data/SRP1/Results/P515S55/w183_nl_reference_pre/w183_nl_digest.json.gz \
    --g0-sha256 cd58078ef0e0c297b50373c65e93b23fd449b46fb8cc13104be35841ae3b6e70 \
    --scratch <scratchpad>/w183b_dryrun_scratch > data/SRP1/Results/P515S55/w183b_e1_dryrun/launch.log 2>&1
  ```

- **Guards.**
  - Solve guards: `SolveProfileGuard(permitted=())` is installed before any project import. Seven guards (this
    launcher's and the six imported launcher modules') were all verified at exactly 0.
  - `pickle.load` / `loads` were blocked: 0 calls.
  - The `pe.ConcreteModel.__init__` tripwire (no model build): 0 calls.
  - The 52 tripwires on the committed launchers' cache / key / pin / precondition / freeze / run functions (PB, F2
    Phase B, F2 certificate): all at 0.
- **Smoke runs.** Before the commit I ran two smoke dry runs with `--allow-uncommitted` into
  `P515S55/w183b_smoke_tmp`. The first failed in the rule-eleven synthetic poll (see Found, first poll). That
  directory was moved out of the repository into the scratchpad. Nothing of it remains in the repository.

**What the launcher does** (module docstring for the full statement).
- **Phases.** Phases `seeds`, `search_N` and `search_A` live inside `--root` (strictly under `P515S55/`). Each phase
  is a write-once campaign dir with its own spec, frozen by the harness's `freeze_campaign_spec`.
- **Configuration.** The configuration block is the ext-v6 block minus the re-settle declaration.
- **Search.** Variant B is used by import (`p515_s53_f2_certificate.build_poll` / `run_continuation`, unchanged), with
  N^max 30, at most 10 polls, completion cap 30, and acceptance at max(bar_x + bar_inc, σ_Q = 18,449.66 €).
- **Concurrency.** `--concurrency N` is frozen into the spec. The memory preflight requires N × 2.75 GiB (S47 rule).
  It is recorded at dry-run / freeze and gates at `--run`.
- **Pins.** The spec records the production head, every production module, the run files and every imported repo
  module, all 81 SRP1 inputs, and the W2 / Phase A tables. `--run` refuses on any difference.
- **Concurrency guards.** The launcher refuses when another copy of itself is alive, and takes the harness campaign
  lock at `--run`.

## Found

### Prediction: "the E1 key dry run yields 7 distinct seed keys" (expert, Addendum 83) — **HELD**

| Assertion | Outcome |
|---|---|
| `len(set(keys_to_evaluate)) == 7` | true |
| no duplicates among the 7 | true |
| exactly 1 alias | true |
| alias target == the x = 0 / A / 2× key | true |
| alias key evaluated | false |
| all 8 seed × model keys pairwise distinct | true |
| every key carries the 2× price | true |
| N keys differ from A keys | true |
| every seed budget-feasible | true |

**Key table.** Flex 2×; candidate keys are shortened to 16 hex characters and eval keys in full.
- Committed specs holding each eval key: 0 for every row. The scan covered the 263 git-tracked
  `data/*campaign_spec_*.json`.
- The x = 0, unit and hist_inc eval keys equal W183-0's `x0_A_2x` / `x0_N_2x` / `unit_A_2x` / `unit_N_2x`.

| Seed | Model | Label | Candidate key | Eval key | Disposition | I (€) |
|---|---|---|---|---|---|---:|
| x = 0 | A | x0_A_2x | 8435c71859ddde68 | 5f6b784127e0eb2bd6fac06206f4329752a6b56b81247bafdbc660f5222c3a4a | evaluate | 0 |
| x = 0 | N | x0_N_2x | 8435c71859ddde68 | c15eea6214da5f31653345d3a2591e90aac3e9e8105c4699579abea800b31289 | **alias** → 5f6b7841… (G0) | 0 |
| 7: 0.25/1 2025 | A | unit_A_2x | db77e1549af855bc | bde9355eba3e09296aba8519573f6c256df68e06c87a33fc315cfd382c3eab03 | evaluate | 317,957.01 |
| 7: 0.25/1 2025 | N | unit_N_2x | db77e1549af855bc | 057f9a6f427ac78e23e6e912af51d651633c469eb3336b71ee3f6f952d1cbe64 | evaluate | 317,957.01 |
| 7: 0.5/2 2025 | A | n7_05_2_A_2x | cc4eb5df385852e6 | 933a529742ce7b625a323337231756510f107200b8db94c208fffa40e0cfc15c | evaluate | 635,914.02 |
| 7: 0.5/2 2025 | N | n7_05_2_N_2x | cc4eb5df385852e6 | e9f793f691d22c626ab8f3209feb5ed3ba2c1b07d45c15cd9c549c4960740439 | evaluate | 635,914.02 |
| hist. incumbent (2030, 5: 0.25/0.5 + 7: 1/3.5) | A | hist_inc_A_2x | 59757776a2da3ffe | c4dea62e18c62f41df3af31a520a383697810613380356014c894f57f39fea36 | evaluate | 992,268.11 |
| hist. incumbent | N | hist_inc_N_2x | 59757776a2da3ffe | c011496da24c46701b04c476728650f9a027be0b09b339b926ac07d2fb4c13e0 | evaluate | 992,268.11 |

**Alias (G0).**
- Source: the W183-0 reference `w183_nl_digest.json.gz`, sha256 `cd58078e…` (equals the pin; tracked and clean).
- `all_ok` true, and the artifact's own guard ran 0 solves.
- At x = 0, 2×, A versus N:
  - the ESSO NL is identical on nodes 5, 7 and 9, for the plain file and for all four files;
  - the network NL is identical in the sequential builder and in the parallel builder.
- The artifact's `x0_A_2x` / `x0_N_2x` keys equal the alias target and the alias key.
- Its configuration (AA, tail, ageing baseline, ESS params sha) and its N declaration equal the launcher's.
- Informative: its production is **not** the current one (Blocked 4).
- The alias is recorded under `aliases`, never as an evaluation.

### Base-key comparison with the committed final-code cells (informative)

Each committed eval key was first recomputed from its spec entry, including its wrapper, and reproduced exactly. The
base key is the same computation without the wrapper.

| E.21 row | Committed cell (spec) | Committed base key | Launcher key (2×) | Equal | Key inputs that differ |
|---|---|---|---|---|---|
| 35 | `h_74eda68d` (v6 re-settle `63ca1061`, eval `dca08628…`) | `bde9355e…` | unit_A_2x `bde9355e…` | **yes** | none |
| 40 | issuing run: W118 r2 re-settle `b1dd9b82` (eval `24c5ccb6…`) | `c4dea62e…` | hist_inc_A_2x `c4dea62e…` | **yes** | none |
| 40 | the row's namesake: s51 F2 Phase B `5ce295e1` (eval `5ca4f86c…`, no wrapper) | `5ca4f86c…` | hist_inc_A_2x `c4dea62e…` | no | `convergence_depth_tail`: F2 Phase B declared none; the launcher declares `{enabled: true, compl_inf_tol: 1e-6}` |
| 54 | `e_no_ageing` (ext-v6 re-settle `10f4aa7a`, eval `1b10e9e1…`; at 1×) | `054976fb…` (= W183-0's expected value) | unit_N_2x `057f9a6f…` | no | `flex_price_multiplier`: absent (1×) against 2.0 |

- Row 54 at its own price: the launcher's unit_N key at 1× is `054976fb…`, equal to the committed base key, with
  nothing differing. So ruling 3 (i)'s `eol_retention_r` = 0.50 keeps the key space consistent.
- Differences that are not in the key, recorded per row:
  - cap: 232 / 281 / 300 for the re-settle runs, against the production exit at cap 500;
  - post-certification: persisted in the committed cells, none in the launcher;
  - stop rule: settling v6 / v2 / v6 declarations, against the production exit.

### Configuration, labels, concurrency, pins (asserted; all true)

- **Configuration.**
  - The configuration of the would-be seeds spec equals the committed ext-v6 `e_no_ageing` configuration on every key
    except the free text (`name`, `note`). The compared keys are AA, ageing baseline and label, tail, ESS params
    path / sha / last commit, case file / sha / last commit, `apply_rho`, `full_diagnostics_in_rows`, `overrides`
    and `arm_label`. The ext-v6 spec's sha256 `10f4aa7a…` holds.
  - All seven ext-v6 specs agree on these keys.
  - The ext-v6 entry carried `settling_resettle`; the launcher's entries carry no wrapper.
- **The would-be seeds spec.**
  - It was frozen in scratch by the harness's own `freeze_campaign_spec`; sha256 `08e51f1e…`.
  - Its content is embedded in the JSON. It was not written into the repository as `campaign_spec_*`, so the dry run
    adds nothing to the committed key space.
  - Labels:
    - spec-level `model_variant_label` and `flex_price_label` are present;
    - the 3 N entries carry the variant and both labels;
    - the 4 A entries carry the flex label and no variant;
    - no `rho0_scale` / `nrf_option` or their labels anywhere.
  - Other entry checks: eval keys recompute; cap 500; 10 cycles; overrides empty.
- **Concurrency and memory.**
  - Concurrency 1 is recorded in the spec and in `extra.concurrency`. The preflight rule (N × 2.75 GiB) and the
    measurement are recorded there too.
  - Measured (non-gating):

    | Slots | Available | Required | Pass |
    |---:|---:|---:|---|
    | 1 | 14.63 GiB | 2.75 GiB | yes |
    | 4 | 14.63 GiB | 11.0 GiB | yes |

    hw.memsize is 32 GiB.
- **Pins.**
  - production head `6f43479e`, with all 38 module hashes;
  - run files: harness `c34ebdc5…`, gates `6f4631e1…`, launcher `320d503c…`, oracle `699d571a…`,
    `settling_criterion` v1–v6;
  - 53 imported repo modules;
  - 81 inputs, including case `dbfdb2a0…`, ESS params `39106f93…` (equal to the PB pin), cost `e17bd588…` (equal to
    the PB pin), and the 4 network params files;
  - the W2 table `28152120…` and Phase A `a2b78541…` (σ_Q = 18,449.66).

  Every pinned file is committed and clean.
- **Domain and preconditions.**
  - The domain is 1,705 points per arm; the A and N key sets are disjoint, and every seed key is in its arm's domain.
  - `check_campaign_preconditions` on a new root returned no failure.
  - No campaign lock was present and no other copy of the launcher was running.

### Cache scoping — how "no evaluation from another root ever seeds an arm" is enforced

- `load_seed_cache` reads one path, built from `--root`: `<root>/campaign_s55_<root>_seeds/campaign_results.json`.
  There is no glob and no committed record from elsewhere, and the other arm's search is never read.
- It refuses unless all of the following hold:
  - the results file, the spec and every point's `eval_dir` resolve (realpath) inside the root;
  - the spec is unique, hashes to the results' `campaign_spec_sha256`, and records `e1_root` = this root and
    `campaign_script` = this launcher;
  - every point's key is recomputed under the **arm's** model at 2× and equals both the recorded key and a key of
    that seeds spec;
  - the manifest hashes of the evaluation records hold;
  - the files are git-tracked and clean.
- Points of the other model are skipped, never cached. The x = 0 alias enters only arm N, only through a fresh G0
  acceptance, and only from the same root.
- Evidence:
  - The real dry-run root has no seeds, so both arms' loaders refuse.
  - Planted roots in scratch, with r1 and a sibling r2 holding the same keys at different costs:
    - r1 under N gives exactly its 3 N points plus the alias (4 entries, 4 A points skipped);
    - r1 under A gives exactly its 4 A points with no alias (3 N points skipped);
    - no r2 cost appears;
    - r3, which has one `eval_dir` pointing into r2, refuses the whole load (`eval_dir_inside_root`).

### First poll (informative; the basis of Blocked 1)

- The rule-eleven synthetic poll first used the reference unit as incumbent. Its poll ended in
  `STOP_FOR_REVIEW_completion_cap` (6 distinct < 8, completion 31 > 30), which is how the issue surfaced.
- The checklist now uses x = 0 (completion 14). With x = 0 it exercises the barrier rule, the poll-failure
  certificate and the budget refusal, and every record field is present.
- The per-seed first-poll table and the k = 0…9 table are in the JSON under
  `first_poll_if_each_seed_were_the_incumbent`.

## Not confirmed

- **`--freeze` and `--run` were never executed.** I read through them; a crude undefined-name check passed. Their
  shared parts were exercised by the dry run: `freeze_campaign_spec` + `validate_spec` (in scratch), the cache loader,
  the incumbent rule (no seeds exist), and rule eleven. Not exercised:
  - `--run`'s `H.evaluate` calls and the search loop with real records;
  - the results writer;
  - the pin comparison against a frozen spec;
  - the root-refusal paths of the search phases.
- **The child's handling of `model_variant` together with `flex_price_multiplier` at run time** is untested here.
  W183-0 built that combination; W183d / G0 is the gate for it.
- **The G0 alias on the final code.** The identity used is W183-0's, at production `575bfec4`, before W183c.
- **Barrier stop rule and Halton start** (Blocked 2, 3).
- **Memory.** The preflight is a point measurement at 18:21; it says nothing about RSS under 4 slots, which the batch
  gate measures.
- **Scope of the "0 committed holders" statement:** the git-tracked `data/*campaign_spec_*.json` at `cb36b22e`
  (263 files). Untracked specs were not scanned.
- **Tripwire scope.** The tripwires cover the 52 named functions only, not every symbol of those modules.
