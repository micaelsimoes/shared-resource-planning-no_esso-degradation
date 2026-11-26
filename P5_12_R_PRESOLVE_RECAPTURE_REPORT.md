# P5.12-R — deterministic pre-solve recapture report

**Outcome: the cycle-21 target input was captured at both boundaries and stopped
before `solver.solve`. The target cycle-21 solve was NOT executed. Neither P5.12-C
arm (A or B) was executed.**

Date: 2026-09-11. Host: `Micaels-Mac-Studio.local`. Repository:
`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`.

## 1. Provenance and repository state

| Item | Value |
|---|---|
| Branch / HEAD | `feature/derivative-free-planning` / `ba202e2b0e937306f3c163de2173951c1d0c24f0` (unchanged start → end) |
| Upstream | `origin/feature/derivative-free-planning`, ahead 3 / behind 0 (no fetch) |
| Tracked state | 1946 paths; approved uncommitted edits to `REVISION_CONTEXT.md` and `LOCAL_NLP_STABILITY_PLAN.md` only (recorded in the baseline `approved_diff`); nothing staged |
| Baseline | `data/SRP1/Results/P512R/R0_v2_ba202e2b/` (generator `r0_v2_generator.py`, SHA-256 `d572fe62…f817`; `initial_repository.json` `aba7e9ea…e697`; `provenance.json` `75af2275…6c25`; `accepted_hashes.json` `95d04eec…de9e`; `runtime_identity.json` `251b3474…ccaa`) |
| Runtime | `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`, Python 3.11.11 arm64, NumPy 2.4.2, pandas 3.0.0, SciPy 1.17.0, Pyomo 6.9.5, copulas 0.14.0; IPOPT 3.14.18 / ASL 20241111 at `/usr/local/bin/ipopt`; MA97; gate passed |
| Scenario checksum | `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` (canonical) |
| Harness | `p512_r_presolve_recapture.py`, SHA-256 `f0f120c26ec2c50b774ff42051c233b87fba0341e3d959faafe70283301d86f0` (unchanged start → end; accepted rehearsal `data/SRP1/Results/P512R_REHEARSAL/20260911T160420Z/`) |
| Reconciliation with old R0 (`dd000167`) | commits `d20220dd`, `8b83b139`, `ba202e2b`; only `CLAUDE.md`, the two governing documents and four `.claude/` configuration files changed/added; no `.py` or parameter change |

The historical R0 files (`data/SRP1/Results/P512R/{provenance,initial_repository,runtime_identity,accepted_hashes}.json`)
are byte-identical before and after (`9d654a00…`, `9c96d7de…`, `1e715f1d…`, `dce5c7e6…`).
All 16 protected artifacts (including the P5.10/P5.11/P5.12-A/B/C evidence) matched at start and end.

## 2. Commands and changes

One invocation (no retry), launched in the background by the Worker:

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p512_r_presolve_recapture.py run --baseline-dir data/SRP1/Results/P512R/R0_v2_ba202e2b
```

Exit code 0; wall time 1176.3 s. No file was edited by the run stage. Production
code, parameters, IPOPT options, rho (1.5 / 300 / 1.0), adaptive rho (off),
RESCALED construction and solve order were unchanged; the cap of 21 existed only
on the in-memory planning copy. Parallel execution was off.

## 3. Cycles 1–20 reproduction

All 20 cycle rows matched every field of all four P5.12-A trajectory files
exactly (`matched: true`, no differences). All 49 P5.12-B pre-setup block
summaries (48 at cycle 20, the target at cycle 21) matched exactly. Each cycle
had 51 successful local solves (36 DSO, 12 TSO, 3 ESSO); no failure, recovery or
rho change occurred.

| Cycle | Recourse | `primal_pf` | local solves ok | converged |
|---|---|---|---|---|
| 1 | 2352009862.13208 | 0.45667 | true | false |
| 2 | 1637242223.1869924 | 0.0024786 | true | false |
| 3 | 1537781919.4896734 | 0.00063195 | true | false |
| 4 | 1520056762.6251268 | 0.00052312 | true | false |
| 5 | 1511570375.357838 | 0.00032004 | true | false |
| 6 | 1504186483.702427 | 0.00024851 | true | false |
| 7 | 1497226342.8467867 | 0.00032592 | true | false |
| 8 | 1490641013.4885292 | 0.00030113 | true | false |
| 9 | 1484326139.2354076 | 0.00018478 | true | false |
| 10 | 1478281551.1049416 | 0.00027801 | true | false |
| 11 | 1472437579.1214523 | 0.00025602 | true | false |
| 12 | 1466756869.4635346 | 0.00017636 | true | false |
| 13 | 1461174062.836328 | 0.00010891 | true | false |
| 14 | 1455670980.1222303 | 0.00031977 | true | false |
| 15 | 1450287053.5586486 | 0.00033476 | true | false |
| 16 | 1445037460.1234787 | 0.00011955 | true | false |
| 17 | 1439859937.3638337 | 0.00022671 | true | false |
| 18 | 1434760363.8607883 | 0.00043377 | true | false |
| 19 | 1429724548.6047106 | 0.00026481 | true | false |
| 20 | 1424778916.2652295 | 0.00023451 | true | false |

Markers required by the plan (cycles 1, 13, 20) are exactly equal.

## 4. Ordered solve ledger

2190 ledger entries (one `attempted` and one `completed` per solve), 1095 solves
in total: 51 at cycle 0 (initialization), 51 in each of cycles 1–20, and 24 at
cycle 21. The 24 cycle-21 solves are exactly the first 24 blocks of the recorded
order (`DSO:case33_1|*` and `DSO:case33_2|*`, all years/days), all successful.
There is no ledger entry of either phase for cycle 21 /
`DSO:case33_3|2025|Spring`.

## 5. Target cycle-20 solve

`DSO:case33_3|2025|Spring` at cycle 20: `Optimal Solution Found`, 115 iterations,
unscaled dual infeasibility `1.1323057973496387e-03`, constraint violation
`7.214950263900732e-07`, complementarity `4.541381094227932e-05` — identical to
P5.12-B. Log extract `cycle20_target.log` (`60a87008…11e8`), raw result
`cycle20_target_result.pkl` (`89f4a3f0…3a0e`).

## 6. Cycle-20 checkpoint

`cycle20_checkpoint/` — boundary "cycle20 complete; before cycle21 DSO
coordination updates". 51 blocks (48 network, 3 ESSO) with per-block ordered
state and `.nl` export of original and reloaded models; `checkpoint.pkl`
SHA-256 `488becea…6b76`; coordination/history state digest `bb55daed…e11a`;
reload equality and per-block export equality verified by the harness.

## 7. Target captures

| Capture | snapshot.pkl SHA-256 | semantic SHA-256 | `.nl` SHA-256 | symbol map | warm start | reload / NL equal |
|---|---|---|---|---|---|---|
| cycle20_pre_setup | `ba0fd23b…a790` | `393ce242…be5c` | `7c5241f6…8803` | `c13732e8…d2ff` | true | true / true |
| cycle20_prepared | `f9f72c86…1aca` | `58e2a063…c8e4` | `fbe9e6e8…dbda` | `c13732e8…d2ff` | true | true / true |
| cycle21_pre_setup | `38fa9e2c…5cd8` | `22d5d850…a0e5` | `2039c742…1308` | `c13732e8…d2ff` | true | true / true |
| cycle21_prepared | `00e8634d…6a5c` | `3fd294d7…a85f` | `5934341b…39a7` | `c13732e8…d2ff` | true | true / true |

The Planner independently reloaded both cycle-21 pickles: file hashes match the
manifests, the recomputed ordered state equals the recorded semantic hash and
`ordered_state.json`, and the original and reloaded `.nl` hashes match.
`.nl` fidelity relative to the writer invocation inside `solver.solve` is not
separately demonstrated (documented caveat).

Effective options at the prepared boundary (both cycles): `tol 1e-5`,
`acceptable_tol 1e-4`, `acceptable_iter 5`, `linear_solver ma97`,
`bound_push / bound_frac / slack_bound_push / slack_bound_frac 1e-5`,
`warm_start_init_point yes`, all five `warm_start_*` pushes/fractions `1e-5`,
`file_print_level 6`, `file_append yes`.

## 8. What solver setup changed (pre-setup → prepared)

Factual description only. Between the cycle-21 pre-setup and prepared captures,
the only differing components are the `ipopt_zL_in` and `ipopt_zU_in` suffixes;
variables, parameters, constraints, objective, `dual`, `ipopt_zL_out` and
`ipopt_zU_out` are identical. Cycle 20 shows the same pattern.

| Suffix (cycle 21) | entries | changed by refresh | `_out` entries | `_in`-only (not refreshed) | max \|value\| pre = prepared |
|---|---|---|---|---|---|
| `ipopt_zL_in` | 7492 | 7243 | 7460 | 32 | 7995263.502414585 |
| `ipopt_zU_in` | 7320 | 6040 | 7298 | 22 | 10208868.739424342 |

The production refresh is `_in.update(_out)`: prepared `_in` equals `_out` on
every `_out` key, and the remaining `_in` entries keep their previous values.
All 54 non-refreshed entries belong to `pg`/`qg` variables whose lower and upper
bounds are equal (not Pyomo-fixed); all 54 are exported to the `.nl` together
with their multipliers. The model has 102 such equal-bound variables.
The largest magnitudes in the exported warm start (`7.995e6` on
`qg[4,0,0,20]`, `1.021e7` on `pg[4,0,0,22]`) are among these non-refreshed
entries; the largest refreshed values are about `1.012e5` (`zL`) and `3.20e3`
(`zU`). The non-refreshed values are identical at cycles 20 and 21; the set
differs by one entry per suffix (`qg[2,0,0,21]`, bounds 0/0, becomes
non-refreshed in `zL` and is refreshed in `zU` at cycle 21).
No causal interpretation is made in this stage.

## 9. Stop-path and verdict evidence

- Termination: `Captured` raised inside the setup wrapper after the prepared
  capture — "cycle21 target captured after setup; solver.solve not entered".
- `guard_calls_target = 0`; `original_solve_calls_target = 0`; 1095 solver
  process launches in total, none with context cycle 21 /
  `DSO:case33_3|2025|Spring`.
- Target IPOPT log `evals/cold/logs/optim_log_case33_3_2025_Spring.log`: size at
  cycle-21 pre-setup `7229066` bytes = size at exit `7229066` (confirmed on disk
  after the run).
- Integrity before and after: no differences (HEAD, branch, staged state,
  tracked path set, tracked hashes, harness hash vs baseline and vs start).
- `final_verdict()`: all 14 checks passed; `verdict_failures = []`.
- Journal `run.json` SHA-256 `dbb17825…ab0b`; console log
  `data/SRP1/Results/P512R_console.log` `e927fc7c…b0fc`.

No stopping rule fired. The only files written outside the P512R run directory
were the console log and production's usual rewrite of the untracked
`data/SRP1/Diagrams/*.pdf` during data loading.

## 10. Limits

This stage proves lossless capture and reload of a new state that matches all
available historical evidence exactly. It does not prove bit-identical equality
with the original P5.12-B full state, for which no full-state hash exists. The
accepted 3000-iteration failure has not been reproduced from this capture; that
requires a separately authorized Arm A.

P5.12-R CAPTURED — lossless reload verified; historical failure reproduction remains pending planner review
