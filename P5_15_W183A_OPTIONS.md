# P5.15 W183a — `rho0_scale` and `nrf_option` keyed evaluation options (Addendum 83 ruling 3 (ii), harness half of (iv))

Zero solves. Production modules, case/params/cost files, specs, frozen files and other harnesses were not touched.
`SolveProfileGuard(permitted=())` was verified at exactly 0, in the test process and in the T5 subprocess. Pickle loads
were blocked and counted: 0.

Objective convention: no objective value is reported here.

**Resumed task.** A previous Worker hit a usage limit partway through. It left an uncommitted edit of
`p515_s44_campaign_harness.py` (+119 lines). That edit covered: the module docstring paragraph, the
`EVALUATION_OPTION_KEYS` entries, the two labels and `NRF_SUPPORT_ATTRIBUTE`, `validate_rho0_scale`,
`rho0_scale_in_key`, `validate_nrf_option`, `nrf_option_in_key`, `nrf_option_production_support`, the threading of
both options through the four wrapper branches and every base branch of `evaluation_key`, and
`_add_w183a_options`. I reviewed it against the task and found no error. It is kept as written, with one exception: I
added a sentence to its module-docstring paragraph to describe the NRF off-check below. Everything else listed under
"Changed" is new.

## Blocked on Planner

Nothing blocks. Three points for the Planner:

1. **W183c must match the support marker this harness tests for.** The harness counts production as NRF-capable only
   when a freshly built `network_parameters.NetworkParameters()` has an attribute named `no_reverse_flow`.
   - That attribute must default to a falsy value (off).
   - Once the attribute exists, every evaluation that does not declare `nrf_option` True requires
     `no_reverse_flow` to be off on the TSO and on every DSO. If a params file switches it on without the key, the
     run raises.
   - Once the attribute exists, every evaluation also writes a `w183a_nrf_option` entry into its
     `rule_eleven_checklist`. This is a change of record format for undeclared runs after W183c.
   - If W183c chooses another name, `NRF_SUPPORT_ATTRIBUTE` (harness l. 387) has to follow it.
2. **Not done: recording the binding NRF hours.** Ruling 3 (iv) says "the harness records the binding hours from
   `p_int_tso_expected_mw` / `p_int_dso_expected_mw` at terminal". That capture is not in this task's list and I did
   not implement it. It must exist, and be asserted in the capture checklist, before E3 (W184). It could go in W183c or
   in the spec v1 checklist.
3. **One guard I added that the task did not name.** When `rho0_scale` is declared, the hook also requires
   `previous_iter.ess` to be off for both TSO and DSO. If it were on, `rho_ess_prev` would start from the unscaled
   `rho_previous_iter` (case file: 0.1), so "every channel scaled" would be false. The case file has it off. Please
   confirm or remove.

## Changed

- **`8d324c10`**: the harness change and the test script, committed before any test run on the repo.
  - `p515_s44_campaign_harness.py` sha256 `b2710a1b33048d82228fd00840d2e52d17052079135cfa0be36f32bcacc9cf58`. The
    pre-change blob at `33dbf690` was `9c6bda19…`.
  - `p515_s53_w183a_option_tests.py` sha256 `78393efc69b2b44d9ebd76312e6cee7bda28d6ac5fe74558801986df6fc4e4f5`.
- **This commit** adds the report and the outputs:
  - `data/SRP1/Results/P515S55/w183a_option_tests/`: the result JSON, the manifest, and `launch.log`.
  - `data/SRP1/Results/P515S55/w183_nl_post_w183a/`: the gzipped digest JSON, both manifests, `launch.log`, and the
    24 ESSO NL gz files.

**Diff summary, `33dbf690..8d324c10`, `p515_s44_campaign_harness.py`.** Line numbers are post-change. The diff is
+288 / −16.

| Lines | What | Author |
|---|---|---|
| 262–277 | Module docstring paragraph (W183a) | predecessor (+1 sentence by me) |
| 355–356 | `EVALUATION_OPTION_KEYS` gains `rho0_scale`, `nrf_option` | predecessor |
| 383–387 | `RHO0_SCALE_LABEL`, `NRF_OPTION_LABEL`, `NRF_SUPPORT_ATTRIBUTE = 'no_reverse_flow'` | predecessor |
| 704–747 | `validate_rho0_scale` (finite > 0; bool refused), `rho0_scale_in_key` (absent / 1.0 → None), `validate_nrf_option` (bool only), `nrf_option_in_key` (absent / False → None), `nrf_option_production_support` | predecessor |
| 1043–1238 | `evaluation_key`: new kwargs; docstring; both options passed to the base key in the four wrapper branches (resettle, extension, settling continuation, certification continuation); `_add_w183a_options` in every base branch; a new branch keys the options alone when nothing else is declared | predecessor |
| 1240–1250 | `_add_w183a_options` | predecessor |
| 1403–1407, 1446–1447, 1470–1477, 1510–1519, 1573–1576 | `freeze_campaign_spec`: docstring; validates both options; passes them to `evaluation_key`; entry records the value when given and the label only when non-default; spec-level labels | me |
| 1849–1872 | Barrier record carries the options (`_w183a_record_fields`, new) | me |
| 2986–3057 | `apply_rho0_scale`, `apply_nrf_option` (new) | me |
| 3061, 3095–3107, 3113–3114, 3184–3191 | `_config_hook_factory`: kwargs; docstring; validation; the two steps, placed after the S39 checks, the overrides and the tail | me |
| 5541–5552 | `_child_real`: label rule at spec and entry level for a non-default `rho0_scale` / `nrf_option` | me |
| 5820–5821 | Options passed to the hook | me |
| 5940 | Success record carries `rho0_scale` / `nrf_option`, label, `*_applied_in_child` | me |
| 6153–6154 | Exception record: same fields | me |

**Where the entry validation lives.** It is in the harness (`_child_real`), not in `p515_g_g1_g4_admm_gates.py`, so
that file is untouched.

**Other places that recompute eval keys.** Inside the harness, `evaluation_key(` is called only by
`freeze_campaign_spec` (l. 1472) and by the wrappers inside `evaluation_key` itself. The child never recomputes a key:
it looks up `--eval-key` in the frozen spec (l. 6027). 59 other `p5*` scripts call `evaluation_key`
(campaign launchers and check scripts). None passes the new kwargs, so their results are unchanged (T1). They were
not edited.

**ρ₀ is consumed after the hook (by reading).**
- `run_admm_arm` calls `pre_solve_hook` after `_construct_arm_planning` and before `run_operational_planning`
  (`p515_g_g1_g4_admm_gates.py:1275`, `:1283`).
- `_run_operational_planning` reads `admm_parameters = planning_problem.params.admm` (`shared_resources_planning.py:2782`).
  It passes that to `update_distribution_models_to_admm` / `update_transmission_model_to_admm` /
  `update_shared_energy_storage_model_to_admm` (`:2931-2933`).
- Those functions build the mutable Params from `params.rho`:
  - TSO `rho_v` / `rho_pf` / `rho_ess` at `:5112`, `:5116`, `:5122`;
  - DSO at `:5378`, `:5382`, `:5388`;
  - ESSO `rho` from `params.rho['ess']['esso']` at `:5483`.
- No other production read of `params.rho[...]` exists, apart from:
  - `prox_gamma_* = tau · rho` at `:5081-5083`, where τ = 0 is enforced by the S39 check `tau_is_0`;
  - `rho_ess_prev` from `rho_previous_iter` at `:5128`, `:5394`, built only when `previous_iter.ess` is on (point 3
    above).
- The adaptive rule scales the model Params in place (`_scale_admm_penalty(model.rho_*, factors[...])`, `:8337-8360`)
  and never re-reads `params.rho`. So the run starts from S39 × s and adapts from there.
- The warm-start path (`initial_state` given) clones the models and does not rebuild them. Campaign children are
  cold, but a scaled ρ₀ would not reach a warm start.

## Found

The recorded predictions are the expert's W183 predictions; none of them belongs to W183a. The task's own
expectations are stated as tests:

### T1 — old keys unchanged: **PASS**

- `git ls-files data/SRP1/Results` holds 263 tracked `campaign_spec_*.json`, the same count as W182e.
  - 252 have schema `p515_s44_campaign_spec_v2`, all with distinct content. Together they hold **7,213 entries**.
  - Every key was recomputed from the stored inputs, using the current harness without the new kwargs.
- Results:
  - **7,213 / 7,213 equal the stored `eval_key`.**
  - **7,213 / 7,213 equal the pre-change harness** (blob of `33dbf690`, loaded from a scratch copy, no
    `rho0_scale` kwarg).
  - 0 errors.
  - 0 committed entries already declare either option.
- **Skips: 11**, all with schema `p515_s44_campaign_spec_v1`. A v1 spec carries no `eval_key`; its eval key is the
  candidate key.
  - `P515S44/campaign_s44_gate/campaign_spec_s44_gate_4047b4e3.json`.
  - Ten specs under `P515S44/harness_checks{,_r2}/` (c2, c4f, c4l, c4n, c4).

### T2 — key algebra, refusals, freeze, child label rule: **PASS**

- **Key identities**, on 8 base configurations:
  - The configurations: bare case-file; AA override; the ext-v6 block; ext-v6 + no-ageing + flex 2×; flex 2× only;
    model variant only; premium only; x = 0 with the ext-v6 block.
  - On every one of them:
    - `rho0_scale` = 1.0, the integer 1, `nrf_option` = False, and 1.0 + False each give the key of the option absent.
    - 0.5 differs from the base, 2.0 from 0.5, and True from the base.
    - The base, 0.5, True and 0.5 + True are four distinct keys.
  - Example, bare unit candidate: base `db77e154…`, 0.5 `98fe2015…`, 2.0 `89a9b675…`, True `fdc84fc3…`,
    0.5 + True `2f59747b…`.
- **Wrappers.** Each declaration comes from a committed spec:
  - resettle: `w118 … f2_challenger`;
  - certification continuation: `w98 x0_cont`;
  - settling continuation: `w101 c_star`;
  - extension: `w105 c_star_ext`.

  In each, the key without the options equals the stored key, and so does the key with 1.0 + False. 0.5 differs, and
  True differs. The 0.5 key equals sha256{`base_evaluation_key`: the base key **with** 0.5, wrapper: declaration},
  which shows the option is threaded into the base before wrapping.
- **Refusals** (ValueError) at `evaluation_key`:
  - `rho0_scale` = True, False, 0, −0.5, NaN, inf, or `'0.5'`;
  - `nrf_option` = 1, 0, or `'True '`;
  - the same two refusals at `validate_*` and at `_config_hook_factory`.
- **Freeze refusals** (in scratch, outside the repo):
  - `rho0_scale` True and `nrf_option` 1 are refused.
  - An unknown option is refused.
  - 1.0 next to the absent entry is refused as a duplicate key, and so is False.
  - In every case the campaign root was not created.
- **A successful scratch freeze** of seven entries:
  - Every `eval_key` equals the direct `evaluation_key` call.
  - The non-default entries carry the value and the label.
  - The 1.0 / False entry carries the values without labels, and its key equals the x = 0 base key.
  - The spec carries both labels.
  - The base entry has no new fields.
- **Child label rule** (`_child_real`, stopped by a sentinel at the entry's next read):
  - For each option, the labelled entry passes.
  - A missing entry label, a wrong entry label and a missing spec label each raise a RuntimeError naming the option
    and its label.
  - The default (1.0 / False) entry with no label passes.

### T3 — configuration hook, zero solves: **PASS**

Setup: `_construct_arm_planning` + `_config_hook_factory`, with the ext-v6 spec-like block and the unit node 7
0.25/1.0, 2025. Outcomes:

- **Undeclared + case-file ρ:** passes. All 13 entries stay at S39, and no rho record is written.
- **Undeclared + one entry mutated:** raises `configuration not as frozen …`, naming only the failing check.
  - DSO `pf['case33_2']` × 1.01 → `['rho_pf_matches_D']`.
  - ESSO `ess['esso']` × 2 → `['rho_ess_matches_D']`.
  - ρ is left as it was.
- **Declared 0.5, 2.0, and 1.0 explicit:** all **13 / 13** entries equal S39 × s exactly.
  - The 13 entries are v × 4, pf × 4, and ess × 5 including `esso`.
  - Targets: 0.5 → v 0.00385, pf 0.099, ess 0.005; 2.0 → 0.0154, 0.396, 0.02.
  - The record's `before` equals the case-file ρ, and all of its checks are True.
  - The S39 checks recorded for the run are True, because they ran on the case-file values before scaling.
- **Declared 0.5 + TSO `v['case9']` × 1.5:** still raises `['rho_v_matches_D']`, and nothing is scaled.

### T4 — `nrf_option` True with no production support: **PASS**

- Production has no `no_reverse_flow` attribute.
- The hook raises: `RuntimeError: nrf_option True refused: production has no no-reverse-flow support
  (hasattr(network_parameters.NetworkParameters(), 'no_reverse_flow') is False); the option would be silently
  ignored`.
- No network carries the flag afterwards.
- `nrf_option` False passes.

**T4b (supplementary, SIMULATED support):** a test-only monkeypatch of the marker plus a class attribute, restored
afterwards. It shows nothing about production.
- True sets the flag on TSO, DSO5, DSO7 and DSO9 and reads it back.
- An undeclared run with DSO5's flag on raises `['every_network_flag_is_off']`.
- An undeclared run with every flag off passes.

### T5 — NL and component digests unchanged: **PASS**

- `p515_s53_w183_nl_digest.py` was re-run for `x0_A_1x` and `unit_A_2x` on the changed harness.
  - Flags: `--label post_w183a`, `--no-determinism`, `--expect-production-commit 575bfec4`.
  - HEAD `8d324c10`; scratch in the scratchpad `w183a_tests/w183a_nl_build`.
  - The subprocess exited 0, `all_ok` True, guard 0.
- **174 / 174 blocks byte-identical** to the committed W183-0 reference (`w183_nl_digest.json.gz`, sha256
  `cd58078e…`, tracked and clean). Per configuration: 51 sequential (12 TSO, 36 DSO, 3 ESSO) and 36 parallel DSO.
  - Compared per block: the plain NL, the labelled `.nl` / `.row` / `.col`, the block digest and every component
    digest.
  - The per-block item gz hashes are also identical (reported, not gated).
  - `eval_key_base` is equal on both configurations.
  - The imported harness hash differs (`b2710a1b…` against `9c6bda19…`), so the run did use the changed harness.
- Post JSON: raw sha256 `26c55000…` (2.66 MB, untracked); gz `81cf2b4a…` (committed; mtime 0; round-trip equal).

**Run conditions.**
- Command, attached, both streams to `w183a_option_tests/launch.log`:
  `NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s53_w183a_option_tests.py --out
  data/SRP1/Results/P515S55/w183a_option_tests --t5-out data/SRP1/Results/P515S55/w183_nl_post_w183a --scratch
  <scratchpad>/w183a_tests --expect-production-commit 575bfec4`.
- Exit 0; 207 s; `all_pass` True.
- Before the commit I ran two smoke runs of T1–T4 (`--skip-t5 --allow-uncommitted`), output to the scratchpad only.
  Both passed. The second came after a docstring-only change.

## Not confirmed

- **ρ₀ reaching the built models.** I confirmed by code reading (lines cited above) that S39 × s reaches the built
  ADMM Params. It was not measured: `update_*_to_admm` runs only after the initialisation solves, which this
  zero-solve task does not reach. The first E2 run will show it as `rho_*_before` at cycle 1 in the trajectory.
- **The NRF set path against real production support.** It is exercised only under a simulated marker (T4b). Its
  behaviour against W183c's code is untested until W183c lands.
- **The child label rule end-to-end.** It was tested by calling `_child_real` directly and stopping at a sentinel,
  not through a launched child.
- **T5's scope.** It covers A-model configurations only (`x0_A_1x`, `unit_A_2x`), with no determinism child; that was
  established in W183-0. The N configurations were not re-run.
- **Leftover working directories.** The T3 / T4 planning objects created 36 log directories,
  `data/SRP1/Results/P56A/evals/p515s55_w183a_*`: 12 per run, over two smoke runs and the real run. The T5 builds
  created 5 more (`p515s55_w183nl_post_w183a_*`). All are untracked; the same convention as W183-0's.
