# P5.15 W183i (b): the Step-7 fixed-plan launcher and its zero-solve dry run

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 86 rulings 1–3 and W183i order item (b); Addendum 85 ruling 4
(iii), (vi)–(xi), ruling 7 and W183g (c); Planner task W183i (b).

**Solves: 0.** The dry run ran under `SolveProfileGuard(permitted=())`, installed before any project import and
verified at 0 (`permitted_solve 0 / permitted_exec 0 / blocked 0 / 0`, both the launcher's guard and the PB module's).
`pickle.load` / `pickle.loads` were blocked and called 0 times; `ConcreteModel.__init__` was tripwired and called 0
times. `--run` was not executed. Nothing was frozen into `P515S55/e3_cnrf`, `e2_rho`, `e1_screen_1x` or `e1_2x`; the
scratch freezes went to a directory outside the repository, which was removed at the end.

## Blocked on Planner

Nothing blocks W183i (d). Four choices I made are listed for the Planner to confirm or change before the real freeze.
Each one enters a key or a refusal.

1. **The screen's x = 0 key has a committed holder. The launcher reports it and does not refuse.**
   - `x0_A_1x` at 1× has the key `5cfe69a6…`. That is the same candidate × configuration as the W86 tail-recert cell,
     held by `P515S53/tight_tail_w86/campaign_s53_w86_tail_recert/campaign_spec_s53_w86_tail_recert_ddd6cd44.json`.
   - Addendum 85 W183g (c) says "a committed holder of a key is reported and never reused". I applied that literally
     to the screen. The cell is evaluated anew in its own root. Its working-dir ids differ, because they carry the
     campaign id.
   - For **E3 / E2**, a holder is a **refusal** at freeze and run. This is the ext-v6 pre-launch precedent. 0 holders
     exist.
   - If the Planner wants the screen to refuse too, the x = 0 screen cell cannot launch as specified.
2. **The screen at 1× carries no `flex_price_multiplier` in its entries.** The baseline price is declared by absence,
   which is the hooks module's 1× convention. Declaring 1.0 would give the same key, since
   `flex_price_multiplier_in_key(1.0)` is None, but the entry would read differently.
3. **An uncertified cell in a rule-v6 difference falls back to W132's uncertified form, unchanged.**
   - That form is 3 × max(|gap|, |slack|). For an ungated cell the slack is undefined, so the verdict is
     "indeterminate (slack undefined)".
   - In that case `report_e3` / `report_e2` still record the cell's harness bar and its gap |t| beside the verdict.
   - If E3 does not certify, Q(C_NRF) − Q(C_OPEN) will therefore read "indeterminate". Plan §3 says "the uncertified
     form with its bar". Whether the harness bar should be used as the bar for an ungated cell is a Planner/expert
     ruling. It is not implemented.
4. **The curtailment columns for the ADMM cells.**
   - For C_NRF and C_OPEN I use the record's `res_curtailment_definitional_at_weight_1` (component levels, per
     representative day), multiplied by the instance day weight, which is read from the price-taker arm's per-block
     `day_weight`.
   - That is not confirmed to be the Table 12 / E.23 phase-A definition. Table 12 prints dashes for the coordinated
     row. The P_NRF columns are read from the record Table 12 uses: 807.6 / 0.6 GWh.

## Changed

| commit | content |
|---|---|
| `02e8d467` | `p515_s55_step7_launcher.py` (new), committed **before** the dry run |
| this commit | this report and `data/SRP1/Results/P515S55/w183i_b_step7_dryrun/` |

- **Launcher sha256: `3b990d19bbde8e9fb6bf75d2e5447c55116d4a3d4d21c174e53776dffc561757`.**
- **Pins it runs on:**
  - harness `d9a944f220e791b8…` (patched, W183i (a)), checked at freeze and run;
  - hooks `p515_s55_step7_resettle_hooks.py` `9318ade3e6d0d5fe…`;
  - production head `6f43479e`.
- No other file was edited. The harness, the hooks module, production, the E1 launcher, `TASKS.md`,
  `REVISION_CONTEXT.md` and the brief are untouched.
- `p515_s55_e1_launcher.py` appears modified in the working tree. That is W183i (c)'s parallel work. It is not
  imported, not pinned and not staged.

### What the launcher does

**Phases and roots.** Each cell has one write-once root, `<phase root>/campaign_s55_step7_<cell>`.

| phase | root | cells |
|---|---|---|
| `e3` | `P515S55/e3_cnrf` | `e3_cnrf_x0_1x` |
| `e2` | `P515S55/e2_rho` | 8, in the hooks order |
| `screen` | `P515S55/e1_screen_1x` | 15: `x0_A_1x`, then the 14 lattice unit-box neighbours of x = 0 under N |
| `cross` | `P515S55/e1_2x/cross` | none; `--freeze` and `--run` refuse |

- **Persistence and exit.** E3 and E2 entries carry the hooks module's declaration and `post_certification`
  `{persist_certified_models: True, hull_polish: False, reference: None}`. The screen runs at the production exit
  (cap 500, 10 cycles) with no post-certification, as Addendum 85 ruling 4 (iii) and (vii) require.
- **Configuration.** The ext-v6 block minus the resettle declaration. It is checked key by key against the committed
  ext-v6 spec `10f4aa7a` and the seven ext-v6 specs.
- **`--freeze --phase P`.** One `H.freeze_campaign_spec` per cell, then `frozen_s55_step7_<P>_spec_v1_<sha8>.json`.
  - That file pins each cell spec's sha256, every run file, the imported modules, the 81 inputs, the W2 and Phase A
    tables, the nine committed records the report modes read, the G0 artifact and record, the solver sha256, and the
    exact launch and preconditions-only commands.
  - **Before writing anything** it checks: the declared root, write-once, the pins committed and clean, the patched
    harness and hook hashes, G0 (screen), and the pre-launch assertion for every cell.
  - `--retry-of C` freezes `campaign_s55_step7_<C>_r2` with `retry_of`, plus a write-once retry pin. A third attempt
    is refused.
- **`--run --phase P --cell C --spec-sha256 S [--preconditions-only]`.** Checks, in order:
  - the launch order;
  - the stage spec is committed, clean and self-hashing, and pins S;
  - the cell spec is committed and clean, and the root holds only that spec;
  - `validate_campaign_spec`;
  - the harness, hooks, launcher, case, ESS and cost hashes, and every frozen pin;
  - the pre-launch assertion;
  - the parent capture checklist;
  - the solver sha256;
  - G0, for the screen's x = 0;
  - the memory preflight at concurrency 1 (gating);
  - that no other copy is alive.

  Then the lock, `H.evaluate`, and write-once `campaign_results.json` + manifest. Exit codes are 0 / 1 / 3.
- **Parent capture checklist.**
  - Every phase: `H.assert_record_capture_paths()` (46 items).
  - E3 adds `H.assert_nrf_capture_path()`.
  - E3 and E2 add `S7.assert_resettle_preconditions` (116 items).
  - E2 adds the ρ wiring check (18 items): `apply_rho0_scale` is called and recorded by the config hook, and the six
    `rho_*_after` / `_action` fields are harness trajectory fields that production writes.
- **Stops (exit 3), wired in `cell_outcome`:**
  - E3 — the settling status is not certified. The uncertified form is reported: k_cap, Q_at_cap, t_sum_at_cap,
    band_window, gap refusals, drift rate, the harness bar and the stopping ratio.
  - E3 — `nrf_terminal.status != 'written'`, or the child exits 2 (nrf_capture_error).
  - E3 — `n_active` is not 864 TSO, or not 288 for each DSO.
  - E2 — the ρ read-back fails, or the model counts are not equal to the declared blocks.

  An uncertified E2 cell is **not** a stop, because E2 completes all eight (ruling 3 (i)). A barrier or another
  child exit code is a fault (exit 1).
- **`--report --phase e3 | e2`.** Zero solves. Reads committed results only, and refuses otherwise. Output is
  write-once.

## Found

Objective convention on every value: Q = certified `gross_operational_cost`, contracted interface settlement
excluded. Every value below that is marked PLANTED is test data, not a result.

### Prediction: key counts 1/8/15; 0 committed holders of the E3 and E2 keys — HELD

- **Counts.** 1 / 8 / 15 distinct keys. All 24 keys are pairwise distinct.
- **Committed holders.** 0 for all nine E3/E2 keys, and 0 for their nine base keys. The scan covered 263 git-tracked
  `data/*campaign_spec_*.json`. It excluded the four stage roots, which hold 0 specs today.
- **Screen.** 1 holder, of `x0_A_1x`, the W86 spec above (Blocked 1). The 14 neighbour keys have 0 holders.
- **Assertions.** 263 assertions in 48 groups, all true. The dry run exited 0.

### Key table

`base` = the key without the resettle declaration.

| phase | cell | eval key | base key | key without the option |
|---|---|---|---|---|
| e3 | `e3_cnrf_x0_1x` | `35b275180d897e4c…` | `759802ecc43b3ee5…` | base without nrf: `5cfe69a6…` |
| e2 | `e2_x0_1x_rho0p5` | `64f3605028f1cf87…` | `91b1a0cc3463c00f…` | rho: `87a42150…` |
| e2 | `e2_unit_1x_rho0p5` | `a895b40ec0b7ccdb…` | `d32cf6ea5a7feef0…` | rho: `2bda6371…` |
| e2 | `e2_x0_1x_rho2p0` | `f4df38b251a68808…` | `c24a24a518246870…` | rho: `79368dc1…` |
| e2 | `e2_unit_1x_rho2p0` | `13145d7ed63e9f2d…` | `61c91a566e687413…` | rho: `4de2c29d…` |
| e2 | `e2_x0_2x_rho0p5` | `14a161e63a4ee34f…` | `0df1520bfff6789a…` | rho: `14be8649…`; price: `48f12207…` |
| e2 | `e2_unit_2x_rho0p5` | `3e35c66844450f94…` | `27a2115df13d3ab0…` | rho: `c0a9eba9…`; price: `8f50888c…` |
| e2 | `e2_x0_2x_rho2p0` | `16614c6680f4d3b4…` | `1c5e99664819fe96…` | rho: `335a2337…`; price: `bbf3ef98…` |
| e2 | `e2_unit_2x_rho2p0` | `43074041a7942052…` | `b1378a2047542f28…` | rho: `0e7786ff…`; price: `2abbc3a3…` |
| screen | `x0_A_1x` | `5cfe69a615ae3708…` | = eval | — (alias `x0_N_1x` `105c0fd3…`) |
| screen | 14 × `y{2025,2030}__n{5,7,9 subsets}_p0.25_e0.5_N_1x` | `c01c5f8b`, `7c5afbc0`, `a4bce87e`, `8cf1c0a6`, `296fe199`, `383729be`, `62daa014`, `274f66d7`, `6216b355`, `50c908f6`, `ecf01f76`, `5bd7af4d`, `e943b8d6`, `de3aa8e2` | = eval | each ≠ its key without `model_variant` |

Full 64-hex keys and payloads are in `w183i_b_step7_dryrun.json` `key_tables`.

**Recompute.** Each key is `H.evaluation_key`. It equals the sha256 of a payload I composed separately from the W183a
formula: {candidate_key, overrides, effective AA, [model_variant], ESS baseline, [price if ≠ 1], tail, [rho0_scale],
[nrf_option]}. For a certified cell, the eval key is then the sha256 of {base_evaluation_key, settling_resettle}.

**Each key differs from the same cell without its option.**
- E3's payload carries `nrf_option: true`.
- **E3's base equals W183c's committed x0_A_1x + NRF key `759802ec…`.** The source is `w183c_nrf_gates.json`
  `gates.G3_on_rows.on_x0.eval_key_on`. Without NRF, the base equals its `eval_key_off`, `5cfe69a6…`.
- E2's payloads carry `rho0_scale` 0.5 or 2.0, and `flex_price_multiplier` 2.0 at 2×, with no price at 1×.

**Informative.** These were held against the G0 digest's committed keys:
- the E2 bases without `rho0_scale` equal G0's `x0_A_1x`, `unit_A_1x`, `x0_A_2x` and `unit_A_2x`;
- the screen's `x0_A_1x` equals G0 `x0_A_1x`;
- the alias equals G0 `x0_N_1x`.

**G0 alias.** G0 is accepted with current production at `6f43479e`; the artifact's production equals current. The
G0 record `w183d_g0/g0.json` `8cef45d6…` reads `G0_PASS` True. The 14 neighbours are exactly
`lattice.completion(x0)` (14 feasible).

### Pins, write-once, scan, preflight

- **Configuration.** The frozen scratch spec configurations (e3, e2, screen) equal ext-v6 `10f4aa7a` on every
  non-text key, and the seven ext-v6 specs agree.
- **Scratch freezes.**
  - e3 1, e2 8 and screen 15 cells, every spec check empty, pre-launch held, capture checklist ok.
  - The stage specs are self-hashing and pin each cell spec (scratch shas `75f79793` / `e746fed4` / `2df4ee26`; not
    committed).
  - **Write-once:** a re-freeze was refused, and a planted non-empty cell root was refused. Real mode refuses a
    non-declared root.
- **`--run` preconditions on the scratch E3 cell.** They ran and refused for the expected reasons: not the declared
  root, stage spec not committed, campaign spec not committed. Cell 2 of E2 before cell 1 was refused on launch order.
  `cross` refused both freeze and run.
- **Retry.** An `r2` retry froze with `retry_of` and campaign id `s55_step7_e3_cnrf_x0_1x_r2`. The original root's
  files were byte-unchanged. A second retry was refused, and the r2 run preconditions found the retry pin.
- **Scan exclusion.** These were tested pure:
  - the stage-root and cross-root specs are excluded;
  - `e1_2x` seeds and a sibling prefix (`e3_cnrf_other`) are not.
- **Memory preflight at 1.** 16.73 GiB available ≥ 2.75 GiB, pass.

### ρ replication (Addendum 86 ruling 2) — validated bitwise

- **Inputs.** The committed W183f arm: `w183f_srp1_bitwise_gate/arm/per_cycle_record.jsonl` `f8506905…`, s = 1. The
  counts come from `g_s39_D.json` `a147a313…` `solve_profile`: n_years 3 × n_days 4 × (1 + n_dso 3) = 48, plus ESSO
  3 = 51. They equal its rule-eleven block counts.
- **Cycle 1:**

| channel | replicated | recorded | n | action |
|---|---|---|---|---|
| pf | 0.1980000000000002 | 0.1980000000000002 | 48 | held |
| ess | 0.010000000000000004 | 0.010000000000000004 | 51 | exempt (fixed) |
| v | 0.011549999999999998 | 0.011549999999999998 | 48 | increased |

- **Every cycle.** Both cycles reproduce bitwise.
- **Informative, s = 1, the recursion across cycles.** It reproduces every cycle bitwise on two committed records:
  - the ext-v6 `e_no_ageing` record: 177 cycles, 531 checks, 0 mismatches;
  - row 1 `d110bd1a`: 181 cycles, 543 checks, 0 mismatches.

  The actions seen were increased, decreased, held, exempt (fixed) and held (frozen after 10 unchanged cycles).
- **Controls (PLANTED, E2 at s):**
  - Positive control: the 8/8 read-backs hold.
  - Negative control 1: one ulp moved on pf at cycle 37 gave exactly one mismatch and exit 3.
  - Negative control 2: an unknown action string was refused.

### Report modes exercised on PLANTED records

The planted cells copy the committed ext-v6 `e_no_ageing` eval dir, re-keyed to the Step-7 cells.

- **E3.** Planted NRF capture: 864 / 3 × 288 rows, with binding hours at the four Z7 hours.
  - The planted ok case exits 0.
  - These cases exit 3: NRF status error with child exit 2; TSO n_active 863; DSO 287; an uncertified decision. The
    uncertified case's report carries k_cap, Q_at_cap, t_sum_at_cap, band_window, gap_refusals, drift rate, harness bar
    and stopping ratio, and its difference uses the uncertified form.
  - The cross-checks all hold:
    - C_OPEN is row 1, certified;
    - its Q equals `report_v3` `coordinated.q` bitwise, and equals the evaluation record's `certified_cost`;
    - the benchmark instance is x = 0.
  - The decomposition identity on C_OPEN closes to 2.4e-7 €.
  - **W124 is reproduced.** The TN term of P_NRF minus C_OPEN gives TSO 515,832,298.77 − 445,665,710.71
    = 70,166,588.06 €, which equals W124's dTSO.
  - The P_NRF curtailment reads 807.6 / 0.6 GWh, as Table 12.
- **E2.** 8 cells and 4 pairs.
  - The baselines are read by eval key, and both agree with the committed claims:

| price | rows | margin (€) | threshold (€) | committed claims it agrees with |
|---|---|---|---|---|
| 1× | 1, 2 | −64,417.39 | 13,054.22 (13.1 k€) | `B:n7_4h_e1`, `CHECK:headline_V_minus_I_settled` |
| 2× | 34, 35 | +46,285.46 | 12,272.19 (12.3 k€) | `H:m2:value_minus_I` |

  - I(unit) = 317,957.01 € from the W2 table, equal to the claims' `I_other`.
  - H2 is evaluated per pair. The screen outcome exits 0 for a record and 1 for a barrier.

### The committed records each report mode reads (all hash-checked, tracked and clean at the dry run)

| record | sha256 | read by | what |
|---|---|---|---|
| `P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json` | `590088fe…` | e3, e2 | `tables.cells` by eval key: C_OPEN = row 1 `d110bd1a…`; baselines rows 1/2 (`d110bd1a`, `3f084f2f`) and 34/35 (`354b1bda`, `dca08628`); `tables.claims` by key pair (cross-check) |
| `P515S53/w116_benchmark_nrf/report_v3/report_v3.json` | `bf6dbf72…` | e3 | Q(P_NRF) = `per_arm_nrf.price_taker.q_best`, spread, best start; `coordinated_cell.eval_key` = row 1; `instance.candidate_key` = x0; `claim.bands_eur` (71,926.11 €, between-configuration spread at C\*); `curtailment_table` phase A |
| `P515S53/w116_benchmark_nrf/nrf_arm_price_taker_warm_from_certified_r2/…r2.json` | `b0362c51…` | e3 | TSO `block_components` (TN term of Q(P_NRF)); instance day weights |
| `P515S53/w182f_records/w182f_records.json` | `405c79c9…` | e3 | `Z7_row1_reverse_flow` (the four C_OPEN reverse-flow hours, H3c; its source is row 1's eval dir) |
| `…/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0/component_levels_terminal.json` | `033c1e7e…` | e3 | decomposition of Q(C_OPEN) |
| `…/evals/d110bd1a5977df1e_x0/evaluation_record.json` | `c08fa433…` | e3 | eval key and `certified_cost` (cross-check) |
| `P515S55/w183c_nrf_gates/w183c_nrf_gates.json` | `c4747232…` | keys | E3 base = `eval_key_on` (x0_A_1x) |
| `P515S55/w183f_srp1_bitwise_gate/arm/per_cycle_record.jsonl` / `g_s39_D.json` | `f8506905…` / `a147a313…` | ρ validation | the W183f row and its model counts |
| `P515S45/investment_cost/investment_cost_results.json` (PB pin) | `28152120…` | e2 | I(unit) |

The E3 and E2 cells' own records are read from their roots after W184/W185, committed and clean, by the cell's eval
key.

### Evidence

All in `data/SRP1/Results/P515S55/w183i_b_step7_dryrun/`:

| file | sha256 |
|---|---|
| `w183i_b_step7_dryrun.json` | `f4bc728c439edb7b2268834b29d38dd506aa8d3889c15295d981f1621595078c` |
| `w183i_b_step7_dryrun_manifest_sha256.json` | `82143c7330326e58125f375c4424a42456c39fb7ec4b70df9eefc0e43e762c52` |
| `launch.log` (both streams) | `2cb79d621ad0536aec7d6c901263f3a013f1333e4f25ebe0f5465b3af175c3ab` |

The run was at HEAD `02e8d467`, with the launcher committed and clean, and took 58.7 s.

**Command:**

```
mkdir data/SRP1/Results/P515S55/w183i_b_step7_dryrun && set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_step7_launcher.py --dry-run \
  --scratch <new dir outside the repo> > data/SRP1/Results/P515S55/w183i_b_step7_dryrun/launch.log 2>&1
```

Two earlier smoke runs used `--allow-uncommitted`, wrote into the session scratchpad only, and ran before commit 1.
Each failed only on `launcher_committed_and_clean`, by design; the first also failed one configuration assertion, which
I then fixed.

## Not confirmed

- **The run path has not executed.** `--freeze` was not run into the real roots, and `--run` / `H.evaluate` were not
  run at all. `--report` ran only on planted scratch records. The first real exercise is W183i (d)'s freeze, then
  W184's attended run.
- **The NRF capture on a real run.** The field names `nrf_terminal.row_counts_per_network`, `row_counts_per_block`,
  `binding_hours[].hour` and `per_node` were taken from `H.capture_nrf_terminal`'s source, not from a real run.
- **The hour indexing.** I matched the capture's `hour` (the periods key) against Z7's `period` (0-based), by reading
  `_get_interface_reporting_detail`.
- **The curtailment definition for the ADMM cells** (Blocked 4). Searched: `p515_g_g1_g4_admm_gates.py` ~1795, the
  report_v3 curtailment table, and `gen_appE.py`'s E.23 note.
- **The ρ read-back at s ≠ 1 on a real record.** It is validated only at s = 1 on committed records. At s = 0.5 / 2.0
  it was exercised only on planted rows. The model counts assume `solve_profile` carries `n_days`; in both committed
  records it does.
- **The holder scan's scope.** It covered git-tracked `data/*campaign_spec_*.json` only. Untracked specs and other
  file types were not searched.
- **The other worker's edits.** `p515_s55_e1_launcher.py` was being modified in the working tree during this task by
  W183i (c). This launcher neither imports nor pins it, so its state does not affect these results.
