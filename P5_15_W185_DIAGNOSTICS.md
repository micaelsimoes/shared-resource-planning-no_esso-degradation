# P5.15 W185 (E2): per-cell diagnostics of the eight E2 cells (plan §4)

Authority: `STEP7_SIMULATION_PLAN_2026-10-25.md` §4 ("feasibility checks per cell"); Addendum 85 ruling 6 / Addendum 86
(one guarded-read authorization covers as many guarded reads as the task needs, each logged with its run); W183i (d)
(the run's own persisted `certified_models.pkl` is the capture path); Planner task "W185 diagnostics".

**Solves and builds: none.** `SolveProfileGuard(permitted=())` was installed before any project import and verified at
exactly 0 (permitted/blocked solve 0/0, exec 0/0). `pyomo Model.__init__` was blocked; it was called 0 times.

**Pickle loads: exactly 8 in the real run**, one per allow-listed (path, sha256) pair, each logged in `launch.log` with
its run (HEAD `9587232d`). Smoke runs into the session scratchpad made 3 further logged loads (see Changed).

**Objective convention.** Q = certified `gross_operational_cost`, contracted interface settlement excluded (Addendum 83
ruling 2). Q is quoted only to identify the instance; nothing here recomputes it. The settlement in (6) is the excluded
term.

## Blocked on Planner

None. Every quantity (1)–(7) was captured for all eight cells. Two items need a reading, not a decision:

1. **Strict limit exceedances on the four unit cells, all at solver-tolerance scale.**
   - (2) **Converter circle.** pnet² + qnet² − S² is positive on 40–54 of 576 active rows per cell. The maximum is
     8.37e-05 MVA² (8.37e-09 p.u.²), at `e2_unit_2x_rho2p0`. That is 1.34e-03 of S², i.e. an apparent power up to
     1.00067 × S^Av.
   - (4) **SOC.** 222–235 of 1,152 active SOC-limit rows per cell lie below their limit. None is beyond 1e-6 MWh; the
     deepest is −9.985e-07 MWh.
   - I flagged both as the task asks ("any value outside its limit"). Whether they count as violations or as
     tolerance-level contact with the bound is the Planner's reading. The mechanism is Not confirmed (see below).
2. **The closure-slack limit.** The task's "≤ 1e-6 MWh per slack" is the W180d bound relaxation (1e-8 p.u. × 100 MVA ×
   1 h). Every slack lies in [−9.99922e-07, −9.99892e-07] MWh, i.e. at the relaxed lower bound. So |slack| ≤ 1e-6 holds.
   The **closure sum** S = s⁺ + s⁻ per block is about −2.0e-06 MWh, because each block has two slacks. I applied the
   limit per slack, as the task states, and not to the sum.

## Changed

| commit | content |
|---|---|
| `9587232d` | `p515_s55_w185_e2_diagnostics.py` (new; zero solves, no build; 8 guarded reads). Committed **before** its repository run. Code is copied from `p515_s55_w184_e3_diagnostics.py`, not imported, because importing it would arm W184's one-file pickle gate. |
| this commit | This report; `data/SRP1/Results/P515S55/w185_e2_diagnostics/`: `w185_e2_diagnostics.json` (`7d8b0ca6…`), `manifest_sha256.json` (`4c279ebc…`), `launch.log` (`762638e4…`), `manifest_post_run_sha256.json`. |

- **The run.** The output directory was fresh (created by `mkdir`, so a second run is refused). The run was attached,
  under `noclobber`, with stdout and stderr both going to `launch.log`. Exit 0; wall 67 s (21:36:28–21:37:35 UTC).
  Command:
  `mkdir data/SRP1/Results/P515S55/w185_e2_diagnostics && set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w185_e2_diagnostics.py > data/SRP1/Results/P515S55/w185_e2_diagnostics/launch.log 2>&1`,
  followed by `--post-run`.
- **Before the run.** No other Python or IPOPT process was running, and `.git/index.lock` did not exist. The only
  tracked change was `TASKS.md` (the Planner's; not staged).
- **Smoke runs (session scratchpad, not committed).** Each load in them went through the same gate and was logged in
  its own `launch.log`.
  - `w185_trial_nopickle/`: 0 loads.
  - `w185_trial_2cells/`: 2 loads (`64ec67cc…`, `a668d74a…`).
  - `w185_trial_unit2x/`: 1 load (`abfbcad0…`).
  - The script was edited after the 2-cell smoke run: per-network split, p.u. and S² normalisation of the circle
    residual, and the capacity check moved from flags to notes. The unit2x smoke run then verified the edit. The
    committed script is the edited one.
- **Not touched.** No code, spec, launcher, production file or committed artifact was edited. `TASKS.md`,
  `REVISION_CONTEXT.md` and the brief were not edited or staged. Nothing was pushed.

## Found

### Guards and pins (prediction: exactly 8 loads, 0 solves, 0 builds — held)

**Allow-list.** Exactly the eight (path, sha256) pairs. Before any load, each sha256 was checked equal to five values:
- the value pinned in the script;
- the **committed** campaign manifest (`git show HEAD:`);
- the working-tree manifest;
- `post_certification.json`;
- the bytes on disk.

For every cell, the five recorded inputs (`component_levels_terminal`, `evaluation_record`, `boyd_terminal`,
`interface_settlement_detail_s31c`, `post_certification`) equal the committed manifest and are committed, clean. The
capture checklist covered 81 paths and 101 key checks; all were present before any reading.

**Gate counts.**
- delegated **8** (expected 8);
- refused **12** (expected 12 = 4 controls + 8 second loads);
- `load` attempts 18;
- `loads` attempts 1;
- `Unpickler` attempts 1.

**Negative controls (13/13 refused, never delegated).**

| control | result |
|---|---|
| `pickle.loads(bytes)` | refused |
| `pickle.load(BytesIO)` | refused |
| `pickle.Unpickler(...)` | refused |
| non-listed `esso_models_s39_D.pkl` of cell 1 | refused at the path check, before any read |
| wrong sha256 | refused: a gate of the same class holding cell 1's path with `0…0` refused the real file ("sha256 of the bytes read 64ec67cc… != allow-listed 0000…"), delegated 0 |
| a second load of each of the 8 files | refused, 8/8 |

**Project modules imported by unpickling:** branch, energy_storage, generator, helper_functions,
hierarchical_coordination, load, model_construction_helpers, network, network_parameters, node, shared_energy_storage,
solver_parameters.

### Instances

| cell | eval key | candidate | ρ₀ × | status, k\* | Q (gross, settlement excl.) | `certified_models.pkl` |
|---|---|---|---|---|---|---|
| e2_x0_1x_rho0p5 | `64f3605028f1cf87…` | `8435c718…` x = 0 | 0.5 | certified, 177 | 653,874,468.09 | `64ec67cc…` |
| e2_unit_1x_rho0p5 | `a895b40ec0b7ccdb…` | `db77e154…` 7: 0.25/1 | 0.5 | certified, 177 | 653,620,775.49 | `a668d74a…` |
| e2_x0_1x_rho2p0 | `f4df38b251a68808…` | `8435c718…` x = 0 | 2.0 | certified, 185 | 653,872,972.52 | `b332511c…` |
| e2_unit_1x_rho2p0 | `13145d7ed63e9f2d…` | `db77e154…` 7: 0.25/1 | 2.0 | certified, 194 | 653,620,852.78 | `bada14e6…` |
| e2_x0_2x_rho0p5 | `14a161e63a4ee34f…` | `8435c718…` x = 0 | 0.5 | certified, 162 | 811,016,337.92 | `1c19b627…` |
| e2_unit_2x_rho0p5 | `3e35c66844450f94…` | `db77e154…` 7: 0.25/1 | 0.5 | certified, 178 | 810,651,421.49 | `88b6b474…` |
| e2_x0_2x_rho2p0 | `16614c6680f4d3b4…` | `8435c718…` x = 0 | 2.0 | certified, 136 | 811,014,413.27 | `191e3c7e…` |
| e2_unit_2x_rho2p0 | `43074041a7942052…` | `db77e154…` 7: 0.25/1 | 2.0 | certified, 134 | 810,650,949.46 | `abfbcad0…` |

**What each model set holds.** Every pickle holds 48 block models (12 TSO, 36 DSO) and 72 shared-ESS entries; no
network has its own ESS.
- **Unit cells:** 24 entries carry storage, the TSO copy of node 7 (entry 1) in 12 blocks and DSO7 in 12 blocks.
  S^Av = 0.25 MVA in every block. E^Av is aged by year, e.g. 0.880 / 0.779 / 0.700 MWh in 2025 / 2030 / 2035 at
  `unit_1x_rho0p5`.
  - Free operational variables: 576 per name. Fixed: 1,152, the zero-capacity entries.
- **x = 0 cells:** S = E = 0 in 72/72 entries. Every operational variable is fixed (1,728 fixed, 0 free per name).
  Of the rows, 1,728 circle and 3,456 SOC rows are constructed and none is active. pnet, qnet and SOC are 0.0
  everywhere. The 144 closure slacks are exactly 0.0, fixed, with bounds [0, 0].

### Per-cell table (1)–(7)

**Definitions.**
- (1) is in MWh.
  - Recorded: the closure sum per block S = s⁺ + s⁻, computed as `shared_ess_day_balance_slack` € / 1e3.
  - Pickle: each slack Var × baseMVA × 1 h.
  - Limit: each slack ≥ −1e-6 MWh (the relaxed floor), |slack| ≤ 1e-6 MWh, and ≤ its ub (0.05 E^Av + 1e-5 p.u. =
    0.036 MWh at the 2035 E^Av of 0.70 MWh).
- (2) is over active rows only. Limit: ≤ 0.
- (3) Limit: ≤ 1.
- (4) Margins are SOC − 0.1 E^Av and 0.9 E^Av − SOC, over active rows. A violation is a margin < 0.
- (5) Max over the primal and dual Boyd ratios per channel. Limit: ≤ 1 (`channel_pass`).
- (6) Identity: T_TSO + ΣT_DSO = −Σ priced residual (weighted).

| cell | (1) closure: recorded S per block [min, max] MWh; slack Vars [min, max] MWh; flags | (2) circle max (MVA²), sign, # active > 0 / active | (3) max \|P^E\|/S^Av (apparent/S) | (4) SOC min margins lower / upper (MWh); violations strict / beyond 1e-6 | (5) max ratio v / pf / ess | (6) settlement total (€); identity residual (€) | (7) |
|---|---|---|---|---|---|---|---|
| x0_1x_0.5 | 0 / 0 in 48 blocks; 144 Vars exactly 0, fixed, [0, 0] | n/a: 0 active of 1,728; all-row max 0.0 | undefined (S = 0); max \|pnet\| 0.0 | n/a: 0 active of 3,456; SOC 0.0 | 0.0110 / 0.3592 / 4.7e-21 | −520.37; −1.15e-07 | n/a |
| unit_1x_0.5 | 24 blocks at −2.00e-06, 24 blocks at 0; Vars [−9.99920e-07, −9.99894e-07]; 0 below floor, 0 above ub, 0 with \|s\| > 1e-6 | **+7.35e-05** (7.35e-09 p.u.², 1.18e-03 S²) at TSO 2035 Spring; **40 / 576** | 0.99999997 (**1.000588**) | −9.975e-07 / −9.976e-07; **225** / 0 (of 1,152) | 0.0317 / 0.2510 / 0.1814 | −323.86; +1.91e-07 | n/a |
| x0_1x_2.0 | as x0_1x_0.5 | n/a (0 active) | undefined; 0.0 | n/a (0 active) | 0.0474 / 0.2956 / 5.2e-21 | +1,792.01; −2.57e-07 | n/a |
| unit_1x_2.0 | 24 at −2.00e-06, 24 at 0; Vars [−9.99920e-07, −9.99894e-07]; 0 / 0 / 0 | **+3.12e-05** (3.12e-09, 5.00e-04 S²) at TSO 2025 Autumn; **48 / 576** | 0.99999996 (**1.000250**) | −9.975e-07 / −9.976e-07; **229** / 0 | 0.0368 / 0.2653 / 0.1758 | +203.97; −1.88e-07 | n/a |
| x0_2x_0.5 | as x0_1x_0.5 | n/a (0 active) | undefined; 0.0 | n/a (0 active) | 0.0562 / 0.6102 / 4.6e-21 | −7.70; −1.25e-07 | n/a |
| unit_2x_0.5 | 24 at −2.00e-06, 24 at 0; Vars [−9.99921e-07, −9.99893e-07]; 0 / 0 / 0 | **+3.07e-05** (3.07e-09, 4.91e-04 S²) at TSO 2030 Summer; **54 / 576** | 0.99999995 (**1.000245**) | −9.979e-07 / −9.983e-07; **235** / 0 | 0.0464 / 0.5195 / 0.2634 | −1,475.88; −1.37e-07 | n/a |
| x0_2x_2.0 | as x0_1x_0.5 | n/a (0 active) | undefined; 0.0 | n/a (0 active) | 0.0790 / 0.1365 / 4.5e-21 | +849.16; −1.55e-07 | n/a |
| unit_2x_2.0 | 24 at −2.00e-06, 24 at 0; Vars [−9.99922e-07, −9.99892e-07]; 0 / 0 / 0 | **+8.37e-05** (8.37e-09, 1.34e-03 S²) at TSO 2035 Autumn; **51 / 576** | 0.99999999 (**1.000670**) | −9.984e-07 / −9.985e-07; **222** / 0 | 0.0736 / 0.1244 / 0.2534 | −1,781.11; −8.60e-08 | n/a |

**Flagged (outside the limit, strict), all on the four unit cells:**
- (2) the circle residual is positive on some active rows;
- (4) SOC is below its limit on some active rows, all within 1e-6 MWh.

Nothing is flagged on the x = 0 cells, and nothing on (1), (3), (5) or (6) in any cell.

**Unit cells: TSO copy vs DSO7 copy.** Both copies behave alike.

| cell | circle max, MVA² (TSO / DSO7) | rows > 0 (TSO / DSO7) | SOC strict violations (TSO / DSO7) |
|---|---|---|---|
| unit_1x_0.5 | 7.353e-05 / 7.352e-05 | 18 / 22 | 116 / 109 |
| unit_1x_2.0 | 3.12e-05 / 2.14e-05 | 24 / 24 | 117 / 112 |
| unit_2x_0.5 | 3.07e-05 / 2.68e-05 | 24 / 30 | 119 / 116 |
| unit_2x_2.0 | 8.37e-05 / 7.17e-05 | 25 / 26 | 118 / 104 |

**Further detail, unit cells (all four):**
- **Closure row and end of day.** The `sess_soc_final` row residual is ≤ 8.7e-17 MWh. SOC(end) − 0.5 E^Av is ≤ 2.97e-11
  MWh.
- **Closure slack bounds.** The minimum closure-slack ub is 0.036 MWh, so the slacks sit far from their upper bound.
- **Power and SOC.** max |pnet| is 0.2500 MW, i.e. at S^Av, and max |qnet| is 0.021–0.047 MVAr. So the circle
  overshoot sits where the active power is at its rating and Q is non-zero.
- **The recorded closure sum agrees with the pickle.** At `unit_1x_0.5`, TSO 2035 Spring: the recorded sum is
  −1.99982e-06 and the pickle sum is −1.99982e-06.

**(5) also.**
- Boyd `channel_pass` is True on all 3 channels in all 8 cells.
- Rule-ten terminal step / threshold:

  | cell | ratio |
  |---|---|
  | x0_1x_0.5 | 4.23e-03 |
  | unit_1x_0.5 | 3.82e-03 |
  | x0_1x_2.0 | 1.30e-03 |
  | unit_1x_2.0 | 1.62e-03 |
  | x0_2x_0.5 | 2.57e-03 |
  | unit_2x_0.5 | 5.78e-04 |
  | x0_2x_2.0 | 1.06e-03 |
  | unit_2x_2.0 | 3.35e-03 |

- Per-hour interface consensus residual, max |r| over nodes 5 / 7 / 9:
  - largest at `x0_2x_2.0`: 4.48e-03 / 1.11e-03 / 2.52e-03 MW (2035 Summer h20);
  - smallest at `x0_2x_0.5`: 2.3e-04 / 9.7e-05 / 1.3e-04 MW.

**(6) also.**
- T_TSO + ΣT_DSO − the recorded total = 0.0 in all 8 cells.
- The recorded totals equal the pair reports' t_sum (e.g. −520.37 and −323.86 for pair 1).

**(7) "ESSO σ slack": not applicable.** No quantity of that name exists in the code (W183i (d) Blocked 2). The ESSO
models (`esso_models_s39_D.pkl`) are not in `certified_models.pkl`. They were not on the allow-list and were not read;
the non-listed control was refused on exactly that file. The record's `esso_feasibility_violation_D3` is −1.728e-05 in
all 8 cells, the same as W184.

### Notes (not limits)

- **E^Av in the models vs the record.** On the unit cells, E^Av as held by the network models differs from the
  record's `published_available_capacity_terminal` by up to 2.9e-06 / 1.4e-06 / 1.5e-05 / 1.6e-05 MWh (cells in table
  order). S^Av agrees exactly. Example: `unit_1x_0.5`, 2025: the model holds 0.8800430, the record 0.8800459.
  - Hypothesis, not checked: the record publishes the ESSO's terminal SoH, while the network models were last solved
    with the previous publication.
  - SOC margins are computed against the model's own E^Av, i.e. the rows as solved.

## Unexpected findings

- **The SOC and closure overshoots have the same size.** Both sit at almost exactly −1e-6 MWh: the deepest SOC
  margins per network and cell are −9.91e-07 to −9.985e-07, and the slacks −9.9989e-07 to −9.9992e-07. That is the W180d IPOPT bound relaxation
  (1e-8 p.u. × 100 MVA × 1 h).
  - This would follow if the single-variable linear SOC-limit rows reach IPOPT as variable bounds (relaxed by
    `bound_relax_factor`, with `honor_original_bounds` off) rather than as constraints.
  - Not verified. I did not inspect the NL writer's treatment, and no `.nl` file was read.
- **The circle overshoot is not of that form.** It is a nonlinear row: 3–8e-09 p.u.², up to 1.34e-03 of S² because
  S^Av is small (0.0025 p.u.).
- **A "σ" does exist in the records, but it is not a slack.** `boyd_terminal.json` carries `sigma_computed`
  (93,635,501.58 at `unit_1x_0.5`) and `sigma_fixed` (93,635,360.0). These are the ADMM objective scale
  (`p515_g_g1_g4_admm_gates.py` l. 2820–2822, `objective_scale_is_93635360`). As W184 noted, ESSO aggregate slacks
  exist under the names `slack_es_pnet_up` / `_down` (`shared_energy_storage_data.py`), but only in the ESSO models,
  which were not read.

## Not confirmed

- **Why SOC and the circle exceed their limits.**
  - SOC: the bound-relaxation hypothesis above was not tested. I searched neither the Pyomo NL writer nor IPOPT logs.
  - Circle: IPOPT's `tol` / `constr_viol_tol` as applied to that row were not read. W180d found that
    `constr_viol_tol` and `bound_relax_factor` are not set anywhere, and I did not redo that search.
- **The ESSO-side quantities.** ESSO SOC, the ESSO slacks and the ESSO pnet/qnet are in `esso_models_s39_D.pkl`,
  which was outside the allow-list and was not read.
- **The capacity difference.** The source of the model-vs-record E^Av difference (Notes) is not traced. I looked at
  `evaluation_record.json` `storage_per_node` only.
- **What "closure slack at the bound" means in plan §4.** I read it as the W180d bound-relaxation check, as the task
  states. I did not search the plan's history for another meaning; only §4's text (l. 106–108) was read.
