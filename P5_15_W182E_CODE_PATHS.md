# P5.15 W182e — code paths Z1 / Z6 / C4 (Addendum 82 v2)

Zero solves; this is a code reading. The order named code head `a17f6ba7`; other W182 Workers committed in parallel,
so the run used HEAD `2f4df40c` (the commit of this harness). The script refuses any HEAD that does not descend from
`a17f6ba7`, and refuses if any cited file or production module differs between the two; none did (the files that
moved are the W182d deliverables and this harness only). The last production-code commit is `575bfec4`.

Every statement below cites `file:line` at that head. Each citation is held in
`data/SRP1/Results/P515S53/w182e_code_paths/w182e_code_paths.json` (`claims`): the verbatim excerpt, its sha256,
the file sha256, and the substrings the script asserted inside the range. There are 93 citations and 93 hold. Each
cited file was tracked and unmodified, and its working-tree blob equalled the HEAD blob.

Objective convention: Q = certified `gross_operational_cost`, settlement excluded; salvage reported, excluded.

## Blocked on Planner

Nothing in W182e blocks the W182 report. Four results change what E1–E3 need before any campaign spec can be frozen.
Each needs a code change that this task was not permitted to make; the decision is the Planner's (an expert ruling
where the formulation is touched).

1. **E1: no committed search launcher accepts the no-ageing model.**
   - The no-ageing switch is not a case-file key. It is the attribute `SharedEnergyStorageData.ageing_enabled`
     (`shared_energy_storage_data.py:58-69`). Only the campaign harness's per-entry `model_variant` sets it
     (`p515_s44_campaign_harness.py:2302-2346`, applied in the child's configuration hook at `:2998-3010`).
   - Both committed Phase-B search launchers assert that no model variant is present: PB at
     `p515_s47_phase_b_record.py:874-875` and `:1383`; F2 at `p515_s51_f2_phase_b.py:362-363` and `:851-852`.
     Every later search imports PB.
   - A no-ageing search therefore needs a new or changed search launcher. This is harness code, not production
     code.
   - Sub-decision: `model_variant` must carry `eol_retention_r` (`:366-370`). The value is inert when ageing is off,
     but it enters the evaluation key (`:1126-1133`). T8 used 0.50 (C3 lineage, `p515_s46_ageing_campaign.py:148-154`);
     the baseline is 0.80. Option (a): 0.50, the same declaration as T8. Option (b): 0.80, the baseline's value. Both
     build the same model; only the key differs.
2. **E2 stage 1: ρ₀ × 0.5 / × 2 runs through the campaign harness only after a harness change.**
   - In production, ρ₀ is a configuration parameter, so no production change is needed (Found Z6(b)).
   - But the harness accepts only one override key, `anderson_acceleration` (`p515_s44_campaign_harness.py:327`).
   - Every child refuses unless ρ equals the S39 constants (`:2897-2899`, raise at `:2924-2925`; constants at
     `p515_g_g1_g4_admm_gates.py:5878-5880`).
   - ρ is absent from the evaluation key: the scoped search of `:972-1140` found 0 hits. A case-file edit would
     therefore be refused, and if it were not refused it could collide with existing cache keys.
   - What is needed: a declared, keyed evaluation option. It could use the existing helper
     `p59_rho.apply_rho_to_params` (`p59_rho.py:89-104`) in the configuration hook.
3. **E2 stage 2: a perturbed coordinated start needs new code.**
   - The benchmark's perturbation works on static arm blocks only, after a warm start from the certified point
     (`uncoordinated_benchmark.py:912-970`). Its only caller is the uncoordinated arm runner (`:1103-1119`).
   - The coordinated ADMM builds its models and solves the initialisation inside `_run_operational_planning`
     (`shared_resources_planning.py:2858-2882`). No hook exists between build and the first solve.
   - The child's only hook runs before any model exists (`p515_g_g1_g4_admm_gates.py:1194-1198`). Every evaluation
     is a fresh start (`:1283-1285`).
   - The options I see, for the Planner to rule on:
     - (a) a production hook between build and initialisation solve (production change);
     - (b) a harness-level perturbation of an `initial_state` continuation (`shared_resources_planning.py:2964-2971`).
       This is a different start semantics: a continuation from a perturbed state, not a perturbed cold start.
     - (c) the existing alternative start mode `shared_ess_initialization = 'price_taker'`
       (`shared_resources_planning.py:2944-2947`). Production already has it, but every child refuses it
       (`p515_s44_campaign_harness.py:2910`), and it perturbs only the shared-ESS channel.
4. **E3: the NRF option is a production code change on both copies.** Found Z6(c) gives the sites. The order already
   specifies the option ("declared, off by default", Addendum 82 v2 ruling 3). One formulation point is for the
   expert:
   - The DSO copy `pg_adn` is the reference-generator P **minus** the shared-ESS net power at the reference bus
     (`model_construction_helpers.py:1287-1297`).
   - At x = 0 (the E3 case) the two are equal.
   - For x > 0, "no reverse flow" on `pg_adn` and on `pg_ref` are different constraints. W116's row uses `pg_adn`
     (`uncoordinated_benchmark.py:336-338`).

C4 needs no decision now. It tells the planning (below) that the concurrency gate must be a new one: the existing
script's reference is stale.

## Changed

- `2f4df40c`: harness `p515_s53_w182e_code_paths.py`, committed before the real run.
- This commit adds:
  - the report `P5_15_W182E_CODE_PATHS.md`;
  - `data/SRP1/Results/P515S53/w182e_code_paths/`:
    - `w182e_code_paths.json`, sha256 `2f46d524…`;
    - `w182e_code_paths.md`, sha256 `8bfb520b…`;
    - `manifest_sha256.json`;
    - `manifest_post_run_sha256.json`, which hashes all outputs and both launch logs;
    - `launch.log`;
    - `launch_attempt1.log`. That first attempt refused at the precondition: HEAD had moved to `7614bcbf` (W182d)
      and the first version required HEAD == `a17f6ba7`. It wrote nothing, made zero solves, and was kept rather than
      deleted.
- Run conditions:
  - `SolveProfileGuard(permitted=())`, verified at exactly 0;
  - `pickle.load`/`loads` blocked, counters 0;
  - no production module imported;
  - JSON written through `gate_result_io`;
  - write-once directory;
  - run attached, stdout and stderr to `launch.log`;
  - exit 0.
- No production code, spec, frozen file, clone, `manuscript_review/`, TASKS.md, REVISION_CONTEXT.md or brief was
  touched.

## Found

### Z1 — the no-ageing path, Q(0), the zero-capacity rows

**(a) Is the implementation identical to T8's? Yes, at the production level.**
- T8's no-ageing arm is cell `e_no_ageing` of ext spec v3 `84775dc4`. Its declaration is `model_variant` =
  {`ageing_enabled` false, `available_energy_soh_point` 'end', `calendar_retention_per_year` 1.0,
  `eol_retention_r` 0.5}. It was run at `55bb1e7d`.
- `575bfec4` is an ancestor of `55bb1e7d`. Between `55bb1e7d` and HEAD, the ESSO data/parameter modules, the ESS
  params file, the case file and every model file are unchanged. The only change in the set is `+3` lines in the
  campaign harness (`b4c3dc7a`, the W155 router).
- What "no ageing" means in the code:
  - **"k → ∞" is not how it is built.** The D row becomes `D == 0` and k is never consumed
    (`shared_energy_storage_data.py:699-711`). Mathematically this is the same as the limit; numerically it is
    exact.
  - **φ = 1** is forced whatever the declared retention (`:683-685`; for 'mid' at `:612`).
  - **SoH ≡ 1** follows from the SoH row, with SoH before the first block = 1 (`:713-724`).
  - **The floor row `SoH ≥ soh_min` stays built and active** (`:730-732`). It does not bind at SoH = 1. So "floor
    inactive" holds in the sense of non-binding, not deactivated.
- T8's record confirms the values. The read-back gives `d_row_form` "D == 0", `phi_cal_in_model` 1.0 and floor
  lower bound 0.7 on all three nodes. The node-7 trajectory has D = 0.0 and `soh_end` = 1.0 in all three blocks, with
  `e_available` 1.0.
- No case-file route reaches k → ∞. Retention must lie in (0, 1) and calendar retention in (0, 1]
  (`p515_s44_campaign_harness.py:646-650`; same bounds in `shared_energy_storage_parameters.py:114-115, :167-168`).
- **Does the same path serve a search?** In production, yes. Through the committed search launchers, no: see Blocked 1.

**(b) Is Q(0) exactly independent of the ageing model? Yes, by code reading.**

The ageing-bearing ESSO rows are the throughput row, the D row, the SoH row (k, φ, n) and the floor row (soh_min).
All are registered to their investment cohort:
- registration at `shared_energy_storage_data.py:1566-1570`;
- the rows at `:666` and `:700-732`.

At x = 0 every cohort is inactive (`:1588-1595`, applied for every candidate at `:1441-1447`):
- every cohort-registered row is deactivated (`:1628-1634`);
- D is fixed at 0, SoH at 1.0, and throughput, pch and pdch at 0 (`:1611-1621`);
- the cohort-allocation (H3) rows are deactivated, with shares 0.0 (`:1709-1714`);
- SoH is initialised fixed at 1.0 (`:548-550`).

The rows that stay active at x = 0 do not depend on the ageing model in 'end' mode:
- The available-energy row stays active because it is not cohort-registered. In 'end' mode it reads
  E_av = E_rated · SoH with SoH fixed at 1.0 (`:615-628`). No ageing parameter appears.
- The window rows depend on `t_cal` (`:562-570`). `model_variant` never changes `t_cal`.

Outside the ESSO rows:
- **Salvage.** It is an Expression outside every objective (`:846-854`).
- **Q.** Q sums the TSO and DSO primal values only (`shared_resources_planning.py:1237-1239`). The salvage enters
  only `net_operational_recourse` (`:1302-1306`).
- **Networks.** They see only the ESSO's available capacity (`admm_persistent_workers.py:333-339`), which is 0.
- **Price-taker initialisation.** This is the other code that reads ageing. It runs only in mode 'price_taker'
  (`shared_resources_planning.py:2944-2947`), which every child refuses (`p515_s44_campaign_harness.py:2910`).

Evidence and its limits:
- **Empirical.** Byte-identical x = 0 ESSO NL files are on record for a change of ageing *parameters* with ageing ON
  in both runs (C3 → baseline). The source is `x0_diff_analysis.json`: X5 `esso_nl_identical_before_after` true, X6
  network blocks identical, and the only differing components are the deactivated degradation rows and the salvage
  Expressions.
- **Not on record for `ageing_enabled` false at x = 0.** Scoped search S5 covered the 263 committed campaign specs;
  the four no-ageing entries are all the 7: 0.25/1 2025 unit.
- **Caveat ('mid' only).** With `available_energy_soh_point` 'mid', the active available-energy row at x = 0 carries
  the coefficient φ^(n/2) (`shared_energy_storage_data.py:620-626`). The coefficient multiplies a variable that is
  forced to 0, so the solution is unchanged, but the NL file would differ when φ ≠ 1. Both E1 arms use 'end'.

**(c) Zero capacity: no division by S^Av or E^Av is reached, and every relevant row has a guard.**
- **Network-side normalised complementarity.** It is written in product form:
  - the links are `pch − S_av·pch_hat = 0`, "so that no expression ever divides by the rated capacity"
    (`model_construction_helpers.py:906-931`);
  - the complementarity row in force (BILINEAR_RELAXATION, set in the case33 and case9 params files) is
    `pch_hat·pdch_hat ≤ 1e-4` (`:949-951`);
  - `pch_hat` and `pdch_hat` lie in [0, 1] (`network.py:401-402`).
- **Network side at zero capacity.** The block is inactive when S_av **or** E_av ≤ 1e-10
  (`model_construction_helpers.py:1067-1069`; `definitions.py:93`). Every operational variable, including the hats,
  is then fixed at 0, and every operational row is deactivated, including `sess_comp` and the links
  (`model_construction_helpers.py:1050-1064, :1137-1178`).
- **ESSO side.** A cohort is inactive iff S_inv = E_inv = 0; an inconsistent zero raises an error
  (`shared_energy_storage_data.py:1588-1593`).
- **Cohort allocation.** The only division is share = E_inv / ΣE_inv. It is computed as a Python float, guarded by
  `> 0`, and reached only when at least 2 cohorts are active (`:1716-1726`).
- **Degradation.** It is written as the product D·(2kE_inv) = 365·n·throughput (`:700-703`), and deactivated at
  zero capacity.
- **Elsewhere.** The reported SoH = E_av/E_rated is guarded at E_av = 0 (`:2213-2215`). The ESS-channel ADMM
  normalisation uses the declared 2.5 MVA reference, not S_av (`shared_resources_planning.py:5024-5028`).

**Prediction (expert) "Q(0) exactly independent": HELD by code reading.** It is exact for the model IPOPT receives in
'end' mode. It is empirically confirmed only for a parameter change with ageing on, not yet for the no-ageing switch.

### Z6 — perturbed start, ρ₀, NRF

**(a) The perturbed start does not apply to the coordinated procedure without new code.** See Blocked 3 for the
citations and options. Scoped search S2 found `perturb` in two production modules only:
- `uncoordinated_benchmark.py` (the arm starts);
- `shared_resources_planning.py:1835-2014`, the finite-difference *investment* perturbation of the retired
  sensitivity path, which is not a start.

**(b) Where ρ₀ enters.**
- **Case file.** It is `admm.rho.{v, pf, ess}.<network>`. Current values: v 0.0077, pf 0.198, ess 0.01, and ESSO
  0.01.
- **Read** at `admm_parameters.py:511-513`.
- **Used to initialise** the mutable ρ Params:
  - TSO at `shared_resources_planning.py:5112-5122`;
  - DSO at `:5378-5388`;
  - ESSO at `:5483`.
- **Adaptive rule.** With `adaptive_penalty` true, ρ moves from ρ₀ by the declared ×1.5 and ÷1.5 factors, within
  [1e-4, 1e4] (`:8275-8284`).
- **Proximal term.** It is tied to ρ but τ = 0 in the case file, so it adds nothing.

**Prediction (expert) "ρ₀ is a configuration parameter": HELD for production**, which needs no code change. It is
**not runnable through the campaign harness without a harness change** (Blocked 2).

**(c) The NRF row.**
- **No NRF option exists in the coordinated path.**
  - Scoped search S1 covered all 38 production modules. `reverse_flow|no_reverse|NO_REVERSE|nrf` occurs only in
    `uncoordinated_benchmark.py`.
  - Its row is `pg_adn ≥ 0`, on the DSO arm only (`uncoordinated_benchmark.py:192-212, :336-338, :405-407`). The
    module says so itself: "the coordinated path, production and the TSO arm never carry it" (`:206-207`).
- **The two copies in the coordinated model are the TSO `pc_adn` and the DSO `pg_adn`** (`network.py:438-445`).
  - The TSO copy is `pc + interface_delta_p` (`model_construction_helpers.py:1241-1262`).
  - `pc` is fixed to the DSO request (`shared_resources_planning.py:4233-4242`).
  - `delta` is free within ± the interface rating (`:4267-4273`).
- **Where the rows would be declared:**
  - the TSO builder `create_transmission_network_model`, beside `expected_interface_pf_p` (`:4277-4282`);
  - the DSO builders, **both** of them, beside `expected_interface_pf_p`: the sequential one at `:4372-4377` and the
    parallel `create_distribution_network_model` at `:4488`.
- **How the option would be declared.** The precedent for a per-network option is `shared_ess_model`, read from the
  case33/case9 params files (`network_parameters.py:168-174`). The harness would also need an evaluation option and
  a key entry.
- **What the harness records for interface P on each side.** It records, per (node, year, day, period):
  - `p_int_tso_expected_mw` and `p_int_dso_expected_mw` (the expected interface P of each copy);
  - the TSO `delta_p_mw` and `anchor_p_mw` per scenario;
  - `price_per_mwh`.

  The source is `shared_resources_planning.py:1154-1157, :1209-1220`. The values are written **at terminal only** to
  `interface_settlement_detail_s31c.json` (`p515_g_g1_g4_admm_gates.py:1935, :2051-2058`). Per cycle, only the
  interface duals are captured (`interface_duals_per_cycle.jsonl`, fields `lambda_pf_*`).

**Prediction (expert) "the NRF row needs a new declared option (code change, both copies)": HELD.**

### C4 — the concurrency path

**Is the bitwise gate present on the final code? Only in a form that cannot run as written.**
- The only concurrent-vs-serial bitwise gate script is `p515_s44_gate.py`. Scoped search S7 found 20 scripts with
  "concurren" within 12 lines of "bitwise"; one of them is this harness. Each of the other 19 was inspected, and
  apart from `p515_s44_gate.py` they are serial gates, serial replays or configuration text.
- It runs three slots and compares C* against the D run `P515S39_D_run` (`p515_s44_gate.py:1-21`). Its gating
  scalars are literals: 139 cycles and gross 650,966,975.2943751 (`:38-39`). These belong to the pre-ageing-baseline
  configuration.
- Its one run (committed in `f825b8b7`, 2026-09-19) gave `gate_pass` false, on tie-order diffs only. The ruling was
  "accepted (tie order from the sort fix)" (brief l. 1146).
- It has not been re-run since. On the final code it would fail its literals.

**Evidence of cross-concurrency bitwise identity on the final model code, from records:**
- The W86 tail recert ran 3 slots at `deac7d9a`. Serial W101 replays (`4ca1573d`, concurrency 1) reproduced it
  bitwise through cycles 132 (x = 0), 112 (unit) and 87 (C*), with no divergence (G13).
- No model file differs between `deac7d9a`, `4ca1573d` and HEAD.
- `pb_y2025_n5` was first run in the s47 Phase B at 5 slots (`7c563e9b`). The serial W142 replay at `55bb1e7d`
  reproduced it bitwise through cycle 110 (G13 `replay_equal_every_gated_cycle` true).
- The comparison covers the gated per-cycle fields, not every artefact.
- No campaign frozen on the final production code ran above concurrency 1: scoped search S8 found 49 of 49 at
  concurrency 1.

**Is a 4-slot batch runnable without a code change? In the harness, yes. Through any committed launcher, no.**
- **Harness.** Concurrency is a spec field (`p515_s44_campaign_harness.py:1630`), used by the slot loop (`:1872`).
  There is one campaign lock at the repo root (`:312`), so one campaign per machine with N slots inside it.
- **Isolation.** The harness's isolation argument is at `:59-65`. Each evaluation has its own log directory for
  every TSO, DSO and ESSO holder, and hence its own IPOPT `output_file` (`p56a_oracle.py:174-186`). The ESSO log
  falls back to the cwd only when `logs_dir` is unset (`shared_energy_storage_data.py:29-38`). Each child hashes the
  shared `FrozenSMOPF` directory before and after its run (`p515_g_g1_g4_admm_gates.py:128`); that is a read.
- **I found no shared or relative write path beyond these.** This was not re-verified at the process level.
- **Launchers.** Every committed launcher pins its concurrency: the latest (W142 ext v6) at 1
  (`p515_s53_w142_resettle_ext_v6_campaign.py:134, :572`), and F2 Phase B at 5 (`p515_s51_f2_phase_b.py:855`).
- So a 4-slot batch needs a launcher that declares 4, which E1–E3 need anyway. It also needs its memory preflight
  declared at 4, and a new concurrency gate with a reference serial run on the final code. That gate is the
  order's "concurrency check".
- The serial two-cycle bitwise gates do not cover the parallel DSO builder (`p515_s53_srp1_bitwise_gate.py:75-78`).
  Their last runs at `f9634f62` and `99a5d39d` passed, and no model file has changed since.

## Unexpected findings (reported, not acted on)

1. **The ESSO aggregate capability row is a radius-0 circle at x = 0.** The row
   `es_pnet² + es_qnet² ≤ es_s_rated²` (`shared_energy_storage_data.py:785-787`) is not cohort-registered. At x = 0
   it stays active with `es_s_rated` forced to 0 by active equalities. The constraint is degenerate there (zero
   gradient at the solution). It is ageing-independent and involves no division, but it is part of the zero-investment
   convention that audit 3.7 asks the manuscript to state.
2. **The two sides define the zero test differently.** The network side calls a block inactive when S **or** E is
   zero (`model_construction_helpers.py:1069`). The ESSO side requires both to be zero and raises an error on a mixed
   zero (`shared_energy_storage_data.py:1591-1593`).
3. **The DSO interface copy `pg_adn` excludes the shared-ESS net power** (Blocked 4). The NRF definition for x > 0 is
   therefore a formulation choice.

## Not confirmed

- **Q(0) bitwise under `ageing_enabled` false.** No x = 0 no-ageing evaluation or NL build is on record (scoped search
  S5: 263 committed campaign specs). A zero-solve NL comparison of the two x = 0 ESSO builds would settle it, but
  building a model was not permitted here.
- **Process-level absence of shared writes at 4 slots.** I read the harness's isolation argument and the log-path
  code. I did not trace every write of the W101+ settling hooks; those hooks ran only at concurrency 1 on the final
  code (scoped search S8).
- **Harness combination of `model_variant` with `flex_price_multiplier` (E1 at 2×).**
  - The configuration hook applies the two in sequence (`p515_s44_campaign_harness.py:2998-3010`, then `:3018`).
  - Both enter the key.
  - No committed spec combines them (scoped search S5: 0 entries).
  - The combination is therefore untested.
- **Unscoped claims.** None are made. The scopes of S1–S8 (file lists and regexes) are recorded in the JSON
  `scoped_searches`.
