# P5.15 W183i (a): Step-7 hooks module, router patch, gates (i)–(iv)

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 86, ruling 1 and W183i order item (a); Planner task W183i (a).

**Solves:** 153, all inside gate (iv), the two-cycle bitwise gate. They were declared in advance and verified exactly by
the gate's armed guard. Gates (i)–(iii) ran under `SolveProfileGuard(permitted=())`, verified at 0.

## Blocked on Planner

None. Every gate passed as stated, and the gate (iv) prediction held. Two points are recorded for the Planner.
Neither blocks W183i (b).

1. **Gate (iv): frozen verdict and ruled verdict differ, as at W183f.**
   - `GATE_PASS` as frozen is **False**. The only failing item is `vs_committed_w48_arm_zero_genuine_diffs`. It fails
     on the same 44 diffs: the 22 W64 `PER_CYCLE_RESPONSE_FIELDS` keys × 2 rows.
   - `GATE_PASS_RULED` is **True** under the Addendum 85 rule (c). That is the reading Addendum 86 orders.
2. **Cell-identity choices in the hooks module.** These are my reading of the task; each is stated in the module
   docstring.
   - `arm` is `'A'` on all nine cells. This is the Step-7 arm A, the baseline declared by the absence of
     `model_variant` (Addendum 85 ruling 4 (ii)). It is not a key of the ext-v6 `ARMS` table.
   - `gate` is `'none_first_evaluation'`, W155's convention. The ext-v6 label `'none_first_evaluation_model_variant_arm'`
     would claim a model-variant arm that these cells do not have.
   - `item` is `'E3'` / `'E2'` and `claim_group` is `'step7_E3'` / `'step7_E2'`.
   - At 1× the entry carries **no** `flex_price_multiplier`, so the price is the baseline by absence.
     `flex_price_multiplier_expected` is therefore None at 1× and 2.0 at 2×.

   Every one of these fields enters the eval key. If the Planner wants another wording, it must change before
   W183i (b) freezes any spec.

## Changed

| commit | content |
|---|---|
| `299f6a4a` | `p515_s55_step7_resettle_hooks.py` (new); the one-branch router patch in `p515_s44_campaign_harness.py`; `p515_s55_w183i_a_router_gates.py` (gates i–iii); `p515_s55_w183i_bitwise_gate.py` (gate iv). Committed before any gate ran. |
| this commit | this report and the outputs listed under Evidence |

### Hashes

| file | sha256 |
|---|---|
| harness `p515_s44_campaign_harness.py` before the patch (blob at `dd9e46d1`) | `c34ebdc5463a1fe0fd6398c6acad4dcfb61dbfbd142ef3de517ca517168d77be` |
| **harness after the patch** | **`d9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191`** |
| **hooks `p515_s55_step7_resettle_hooks.py`** | **`9318ade3e6d0d5fec25d0c276c4d82fba5bb163c7da61ec10ee6f49aaf2fa1d2`** |
| `p515_s55_w183i_a_router_gates.py` | `4f7c026a3909ca828714b9a06f8a86902445d44d2f8a3056cba8e83af8175f4d` |
| `p515_s55_w183i_bitwise_gate.py` | `9ed515308fa5c56d639b0a29d0e9da6b75285d3dbbf1929f548bf96c90d911f5` |

### Router patch

The patch adds three lines in `resettle_hooks_module`, after the W155 branch and before the W118 fallback:

```python
    import p515_s55_step7_resettle_hooks as S7C
    if S7C.is_step7_declaration(value):
        return S7C
```

Nothing else in the harness changed. Gate (i) confirmed this by difflib against the `dd9e46d1` blob: one hunk
`@@ -1048,0 +1049,3 @@`, these three added lines, nothing removed.

### Hooks module

- **Schema:** `p515_s55_step7_settling_resettle_v1`.
- **Reused by import, the W155 way:**
  - the v6 rule: `V6.settling_rule_declaration()`, i.e. `settling_criterion_v6`;
  - the v6 state: `ResettleStateStep7` subclasses `V6.ResettleStateV6`, and only the summary is stamped;
  - the nine v6 wrappers;
  - the capture checklists: E105, R `t_sum`, V exit and V6 clean;
  - the production signatures of ext v6.
- **Cell table (9 cells, launch order):**
  - `e3_cnrf_x0_1x`: x = 0, 1×, `nrf_option` True.
  - E2, in the pair order of Addendum 86 ruling 3 (i):
    - `e2_x0_1x_rho0p5`, `e2_unit_1x_rho0p5`
    - `e2_x0_1x_rho2p0`, `e2_unit_1x_rho2p0`
    - `e2_x0_2x_rho0p5`, `e2_unit_2x_rho0p5`
    - `e2_x0_2x_rho2p0`, `e2_unit_2x_rho2p0`
  - unit = node 7, 0.25 MVA / 1 MWh, 2025, key `db77e154…`; x = 0 key `8435c718…`.
  - No cross cells. Those are added after W186.
- **`declaration_for(cell)`, the ext-v6 shape:**
  - cap rule dynamic, `min(k0_run + 109, 300)` (after_first_k0 109, ceiling 300, spec cap 300). This equals ext v6's
    `cap_rule('e_no_ageing')`.
  - holds after the first residual pass;
  - the v6 rule;
  - `record_all_blocks` True;
  - captures equal to `e_no_ageing`'s;
  - `expected_blocks` TSO 12 / DSO 36 / ESSO 3;
  - `floor_row_lower_expected` 0.7;
  - `replay_reference` None, `abort_on_replay_divergence` False;
  - `first_residual_pass_expected`, `N_old` and `original_lapses_after_k0` None;
  - `model_variant` None;
  - plus `nrf_option_expected`, `rho0_scale_expected` and `flex_price_multiplier_expected`.
- **Validator:** a declaration must equal `declaration_for(cell)` exactly, compared as canonical JSON, so types count. A
  cell that is not in the table is refused, and so is an `early_stop` key.
- **`spec_entry_checks`:**
  - The spec holds one entry. Its label is the cell, and its `settling_resettle` is the declaration.
  - The entry's keys are exactly the base keys plus the declared options and their labels.
  - Each declared option must have the exact value and type, with its label at **both** entry and spec level. The
    three options are `nrf_option` + `nrf_option_label`, `rho0_scale` + `rho0_scale_label`, and
    `flex_price_multiplier` + `flex_price_label`.
  - Each undeclared option must be absent, together with its label.
  - The entry carries no `model_variant` or `model_variant_label`.
  - `overrides` is `{}`.
  - The ESS baseline is declared with minimum_soh 0.70.
  - The ESS file is pinned to `39106f93…`.
  - There is no premium and no derived instance.
  - The candidate key and investment year are the cell's.
- The label literals in the module equal the harness's `NRF_OPTION_LABEL`, `RHO0_SCALE_LABEL` and `FLEX_PRICE_LABEL`;
  gate (iii) checks this.

## Found

Objective convention, where a value appears (gate iv only): Q = `gross_operational_cost`, which excludes the
settlement. Gates (i)–(iii) compute no objective.

### Gate (i): every committed eval key recomputes — PASS

- **Method.** W183a T1 by import: `p515_s53_w183a_option_tests.t1_old_keys`. That script was not run as a whole, and
  nothing was written into its output directory. The patched working-tree harness and the `dd9e46d1` harness blob
  (sha256 `c34ebdc5…`, loaded from a scratch copy outside the repo) were compared.
- **Specs.** 263 git-tracked `campaign_spec_*.json` under `data/SRP1/Results`; **252** have schema v2, and all 252 are
  distinct in content.
- **Entries: 7,213** (the expected count).
  - patched key == stored `eval_key`: 7,213;
  - patched key == pre-patch key: 7,213;
  - errors: 0;
  - entries already declaring `rho0_scale` or `nrf_option`: 0.
- **11 skips, all for the same reason.** Each has schema `p515_s44_campaign_spec_v1`, and v1 specs carry no eval_key:
  - `P515S44/campaign_s44_gate/…4047b4e3`
  - six `P515S44/harness_checks/*`
  - four `P515S44/harness_checks_r2/*`

### Gate (ii): routing identity — PASS

- **Scope A:** 195 declarations in the entries of the 263 tracked campaign specs.
- **Scope B:** 257 declarations found in 265 of the 5,776 tracked `*.json` under `data/SRP1/Results` (every dict value
  under the key `settling_resettle`, at any depth).
- **Result.** In both scopes, every declaration is routed by the patched harness to the **same module object** as by the
  pre-patch harness: 0 mismatches. The patched `validate_settling_resettle` also gives the same outcome on every
  declaration (equal output, or the same exception text): 0 mismatches.

| schema | A | B | module (patched = pre-patch) |
|---|---|---|---|
| (no schema key: W118) | 20 | 31 | `p515_s53_w118_resettle_hooks` |
| `p515_s53_w132_settling_resettle_v3` | 38 | 39 | `p515_s53_w132_resettle_v3_hooks` |
| `p515_s53_w135_settling_resettle_ext_v1` | 7 | 7 | `p515_s53_w135_resettle_ext_hooks` |
| `p515_s53_w137_settling_resettle_v4` | 38 | 41 | `p515_s53_w137_resettle_v4_hooks` |
| `p515_s53_w139_settling_resettle_ext_v5` | 7 | 7 | `p515_s53_w139_resettle_ext_v5_hooks` |
| `p515_s53_w139_settling_resettle_v5` | 36 | 38 | `p515_s53_w139_resettle_v5_hooks` |
| `p515_s53_w142_settling_resettle_ext_v6` | 7 | 14 | `p515_s53_w142_resettle_ext_v6_hooks` |
| `p515_s53_w142_settling_resettle_v6` | 34 | 68 | `p515_s53_w142_resettle_v6_hooks` |
| `p515_s53_w155_a64_settling_resettle_v1` | 8 | 12 | `p515_s53_w155_a64_hooks` |

One scope-B item is refused by validation under **both** harnesses, with the same text:
`P515S55/w183a_option_tests/w183a_option_tests.json` `.tests.T2.wrappers.settling_resettle`. It is a W183a
test-result dict, not a declaration.

### Gate (iii): neutrality — PASS

**Committed artefacts.**
- 0 declarations in scope A or B are Step-7 declarations or are routed to the new module.
- The schema string occurs in exactly one git-tracked file, `p515_s55_step7_resettle_hooks.py`.

**Planted entries.**
- **How they were made.** One per cell, 9 in all. Each was frozen through the production `freeze_campaign_spec` into a
  scratch root outside the repo, using the configuration block of spec `10f4aa7a` (hash-checked), and then read in
  memory.
- **Results:**
  - all 9 are routed to `p515_s55_step7_resettle_hooks`;
  - the patched harness validates each to exactly its declaration;
  - the **pre-patch harness refuses each one** (it falls through to the W118 validator: "cell must be one of
    ['f2_challenger', …]"), so the branch is needed;
  - `spec_entry_checks` are all True;
  - the full capture checklist `assert_resettle_preconditions` passes, 116 checks per cell;
  - the eval key recomputes from the entry;
  - `ResettleStateStep7` constructs, and its summary schema is the new schema;
  - the 9 keys are distinct;
  - 0 of them are held by a committed v2 spec.
- **Constants:**
  - the three labels equal the harness constants;
  - the x = 0 key recomputes and equals W155's; the unit key recomputes and equals ext v6's;
  - the cap rule, settling rule, captures, expected blocks and holds_after all equal those of ext v6 `e_no_ageing`;
  - the spec cap is 300, equal to the template's.

**Refusals: 25 of 25 refused, each naming the expected failing check.**
- **Declaration level (R1–R7, by `H.validate_settling_resettle`):**
  - a renamed cell;
  - the ext-v6 cell name `e_no_ageing` under the new schema;
  - the Step-7 declaration under the ext-v6 schema (refused by ext v6);
  - `rho0_scale_expected` mutated from 0.5 to 2.0;
  - `flex_price_multiplier_expected` as the int `2`;
  - `nrf_option_expected` dropped;
  - an `early_stop` key.
- **Entry level (E1–E18, by `assert_resettle_preconditions`, before any solve):**
  - a missing option: E3 without `nrf_option`, or without it and its label; E2 without `rho0_scale`; a 2× cell without
    `flex_price_multiplier`;
  - a missing label at entry or spec level: `nrf_option_label`, `rho0_scale_label`, `flex_price_label`, each at either
    level;
  - a wrong value: `nrf_option` False, `rho0_scale` 2.0 on a 0.5 cell;
  - an undeclared option: `nrf_option` on an E2 entry, `rho0_scale` 1.0 on E3, `flex_price_multiplier` 1.0 at 1×;
  - a model variant;
  - a wrong label;
  - an AA override.

**Guards for (i)–(iii).**
- `SolveProfileGuard(permitted=())` was installed before any project import and read 0/0/0/0 at `verify(0) == []`.
- The imported W183a module's own guard also read 0.
- `pickle.load` / `pickle.loads`: 0, still blocked at the end.
- `ConcreteModel.__init__`: 0.
- Wall clock: 8.6 s.

### Gate (iv): two-cycle bitwise gate on the patched harness

**Prediction "0 genuine, `GATE_PASS_RULED` True, provenance ≤ 4": HELD.**

- **Wrapper.** I used a thin successor, `p515_s55_w183i_bitwise_gate.py`. Wrapper v2 can redirect to a fresh root, but its
  `EXTRA_CLEAN_FILES` is fixed at v1's list + v2, so it cannot add the new hooks module. It also reads against W48
  only.
- **The successor makes only the declared substitutions:**
  - STAGE / SCHEMA (`p515_s55_w183i_bitwise_gate_v1`) / ARM (`s55w183igate`) / OUT_REL
    (`P515S55/w183i_srp1_bitwise_gate`, a new root);
  - EXTRA_CLEAN_FILES = v1's list + wrapper v2 + this wrapper + `p515_s55_step7_resettle_hooks.py`;
  - EXTRA_FORBIDDEN = v1's list, which includes `p515_s55_`.
- **Imported and unchanged:** the committed gate `p515_s53_srp1_bitwise_gate.main()`, and v2's `ruled_comparison`,
  `ruled_verdict` and `check_gate_dir`.
- **Before the run:**
  - no other project Python process (`ps`);
  - neither lock present;
  - gated files clean and committed;
  - the W83 arm hash-checked: 100 manifest entries, 0 problems.
- **The run:** attached and alone, `set -o noclobber`, both streams to
  `P515S55/w183i_srp1_bitwise_gate_launch.log`. HEAD `299f6a4a`; the code was unchanged during the run.
- **Solves:** declared 153 = 51 × (cap 2 + 1).
  - observed 153, guard `permitted_solve 153 / permitted_exec 153 / blocked 0 / 0`;
  - event-level reconciliation identity holds;
  - 0 network-failure blocks, **0 retries**, 0 ESSO recovery events, 0 frozen snapshots;
  - no solves outside the arm.
  - Wall clock: arm 86 s, gate 98 s.
- **Verdicts.** `GATE_PASS` as frozen is **False**; `GATE_PASS_RULED` is **True**.
  - W35 layer: 12/12 true, including 0 diffs against the C\* reference rows and 0 trajectory-field mismatches.
  - Presence: 57 items; W51 presence 21/21.
  - W39 tripwire: 0 calls.
  - W51 counter: 36/36 each, 0 acting.
  - The only failing frozen item is `vs_committed_w48_arm_zero_genuine_diffs`.
- **Instance.** C\* is 0.96875 MVA / 3.875 MWh at nodes 5, 7 and 9, 2025, with key `578636da…`. NRF is off and the tail
  is off (production default).
- **Q = 740,922,817.2425401**, bitwise equal to W48's and to W83's.
- **Rule ten:** terminal step / threshold = 5,330.87, the same in all three arms. The run stops at cap 2 by design.

**Per-file results** (provenance / tie-order / genuine as the gate classifies them, then genuine after rule (c)):

| file | vs W48 | vs W83 |
|---|---|---|
| g_s39_D.json | 3 / 0 / 0 → 0 | 4 / 0 / 0 → 0 |
| per_cycle_record.jsonl | 0 / 0 / 44 → **0** (44 field_list_drift) | 0 / 0 / 0 → 0 |
| the other 10 files | 0 / 0 / 0 | 0 / 0 / 0 |
| **total** | provenance 3, tie 0, genuine 0 after rule (c) | provenance 4, tie 0, genuine 0 |

- **Drift keys vs W48 (22, each × 2 rows):** `E_abs_d_p_mwh_unweighted`, `E_abs_d_p_mwh_weighted`,
  `covariance_dso_weighted`, `curtailed_res_dso_mwh_weighted`, `curtailed_res_tso_mwh_weighted`, `cycle_wall_s`,
  `market_part_mw2h_unweighted`, `market_part_mw2h_weighted`, `max_abs_d_p_mw`, `n_non_optimal_block_terminations`,
  `non_optimal_blocks`, `operation_part_mw2h_unweighted`, `operation_part_mw2h_weighted`, `response_capture_error`,
  `response_capture_s`, `response_captured`, `response_cycle`, `row18_charge_weighted`, `rss_bytes`, `ru_maxrss_bytes`,
  `sum_omega_d2_p_mw2h_unweighted`, `sum_omega_d2_p_mw2h_weighted`.
- **Ruled items:** all six True (`recomputed_comparison_reproduces_the_frozen_one` included).
- **Provenance vs W48 (3),** all under `rule_eleven_checklist`: `w10_arm`, `w183a_nrf_option`,
  `w84_convergence_depth_tail`.
- **Provenance vs W83 (4),** the same keys plus `w83_convergence_depth_tail`, which is the W83 arm's own tail read-back.
- **`per_cycle_record.jsonl`** is byte-identical (`f8506905…`) across the new arm, the W183f arm and the W83 arm.

## Evidence

All committed in this commit; the manifests hash the rest.

- `data/SRP1/Results/P515S55/w183i_a_router_gates/`:
  - `w183i_a_router_gates.json` `e3d0be01…`
  - `manifest_sha256.json` `86d71f49…`
  - `launch.log` `1f71b33c…`
  - `manifest_post_run_sha256.json` `0d40eeb9…`
- `data/SRP1/Results/P515S55/w183i_srp1_bitwise_gate/`, the whole root:
  - `gate.json`, `gate.md`, `row18_gate_addendum.json`, `w51_gate_addendum.json`, the arm's files;
  - `w183i_wrapper_record.json` `06b9cdc3…`;
  - the manifests `manifest_sha256.json` `7ac5d369…`, `w51_manifest_sha256.json` `b74ab730…` and
    `w183i_manifest_sha256.json` `27185d86…`.
- `data/SRP1/Results/P515S55/w183i_srp1_bitwise_gate_launch.log` `91c1a65e…`.
- `data/SRP1/Results/P515S55/w183i_srp1_bitwise_gate_arm_logs_sha256.txt` `c511f27e…`: the 57 files of the arm-named eval
  dirs `P56A/evals/p515s44_s45_w10_snapoff_s55w183igate_578636daa6d6360d_{run,precheck}`, which stay untracked, as at
  W183f.

## Not confirmed

- **The new module's run path has not executed a solve.** That covers `settling_resettle_hooks` installing the wrappers,
  and `ResettleStateStep7.summary()` on a real run. The only exercise was zero-solve: construction, summary on a fresh
  state, and the capture checklist. The first exercise will be W184.
- **The parallel DSO builder.** Gate (iv) covers the sequential builder only (`ParallelExecution: false`), as Addendum
  85 ruling 1 records.
- **Scope of the routing and neutrality scans.** They covered git-tracked `campaign_spec_*.json` (A) and git-tracked
  `*.json` under `data/SRP1/Results` (B). Not searched:
  - untracked files;
  - `.jsonl` and `.json.gz`;
  - files outside `data/SRP1/Results` (the repository-wide schema-string search was over git-tracked files only).
- **The planted eval keys are not E3/E2 campaign keys.** They are informative only. The keys W183i (b) freezes will match
  them only if its configuration's key inputs are the same: case-file AA, ESS baseline, tail, options and declaration.
  `post_certification` (persist on, Addendum 86 ruling 3 (ii)) does not enter the key, and `spec_entry_checks` does not
  check it; that is left to the launcher.
- **E3 captures.** The declaration's captures are `e_no_ageing`'s, as ordered, and name no NRF item. The harness takes
  the NRF terminal capture by itself whenever `nrf_option` is on (W183c). That path was not exercised here.
