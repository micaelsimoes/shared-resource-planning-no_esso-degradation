# Worker Report — P5.15 Addendum 16 items 2-3, PHASE 2 (price-taker initialization: activation, gate arm, Z3/Z4/Z7, preflight)

## Task received

Bounded implementation task, PHASE 2 of Addendum 16 items 2-3: activate price-taker
initialization in the SRP1 case file; add the `s35pt` gate arm to
`p515_g_g1_g4_admm_gates.py`; run the checks that need network solves (Z3, Z4, the
IPOPT half of Z7); run a two-cycle preflight through the real `s35pt` arm. Do **not**
launch the 150-cycle gate (Planner-only). Binding spec: `data/SRP1/Results/P515S35/
frozen_s35pt_spec_v6_651a9d84.json` (commit `a993b088`). Commit 2 (harness + checks +
preflight + outputs + this report) is authorized **only if** Z3, Z4, Z7 and every
preflight signature pass.

**Outcome: Commit 2 is WITHHELD.** Z3, Z4 and Z7 all pass cleanly (evidence below).
The two-cycle preflight found four of five expected signatures pass strongly, but
**signature "Solves: no failures attributable to the initialization" did NOT pass** —
one DSO local solve hit `maxIterations` at cycle 2 and was auto-recovered
(`recovered_tier1`), which also explains a solve count of 154 against the declared
153. Per the task's explicit gate ("Commit 2 is allowed only if ... every preflight
signature pass"), the harness/checks/preflight files and their output directories are
left **uncommitted**, pending Planner review of this finding. `data/SRP1/SRP1_params.json`
(commit 1, unconditional) was already committed and is unaffected by this outcome.

## Files inspected

- `data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json` (binding spec, read in full).
- `WORKER_REPORT_S35PT_PHASE1.md`, `P5_15_S35REF_REPORT.md`.
- `shared_ess_price_taker.py` (Phase 1 production module, read in full).
- `shared_resources_planning.py`: `_run_operational_planning` (≈2310-2500, initialization
  branch and the `_initialize_shared_ess_from_price_taker` call site), the wrapper itself
  (≈4075-4180), `create_admm_variables` (≈3923), `create_distribution_networks_models`
  (≈3550), `create_transmission_network_model` (≈3440), `create_shared_energy_storage_model`
  (≈3708), `update_distribution_coordination_models_and_solve` (≈5311),
  `get_admm_boyd_residual_metrics` (≈5888), the ADMM main-loop convergence/break logic
  (≈2734-2760, 2990-3000).
- `admm_parameters.py` (`shared_ess_initialization`/`_source` attributes, ≈124-125, 345-354).
- `p515_g_g1_g4_admm_gates.py`: the entire `s35ref` section (`OUT_S35REF` through
  `write_boyd_terminal_s35ref`, ≈3085-3693 pre-edit) read in full as the template to reuse
  by calling; `run_admm_arm`, `_construct_arm_planning`, `_refuse_overwrite`,
  `_require_fresh_output_root`, `p512_a_cold_rescaled_convergence.cycle_row`
  (`converged_at_cycle`/`cycle_convergence`/`boyd_all_pass` field provenance).
- `p513_solve_profile_guard.py` (`SolveProfileGuard`, `verify`).
- `p515_s35ref_preflight.py`, `p515_s35pt_phase1_checks.py`, `p515_s34_preflight.py`
  (`_field_completeness`, reused by calling).
- `data/SRP1/Results/P515S35_REF_run/g_baseline.json`,
  `data/SRP1/Results/P515S35_REF_run/s35ref_evaluation_v2.json`,
  `data/SRP1/Results/P515S35_REF_run/boyd_terminal.json`,
  `data/SRP1/Results/P515S35_REF_run/network_failures_baseline.jsonl` (read for reference
  values, the `stopped_by` defect reproduction, and the cycles-1-2 failure-rate baseline).
- `data/SRP1/SRP1_params.json` (full file, before and after edit).

## Files modified

- `data/SRP1/SRP1_params.json` — **committed** (commit `52d70dc9`, alone).
- `p515_g_g1_g4_admm_gates.py` — **NOT committed** (additive-only, 671 insertions, 0
  deletions; new `s35pt` section + one `elif gate == 's35pt':` dispatch branch; the
  entire `s35ref` section is untouched — confirmed via `git diff --stat`).

## Files created (not committed)

- `p515_s35pt_phase2_checks.py` (Z3/Z4/Z7 harness).
- `p515_s35pt_preflight.py` (two-cycle preflight harness).
- `data/SRP1/Results/P515S35/pt_phase2_checks/` (`phase2_checks_results.json`,
  `sha256_manifest.json`).
- `data/SRP1/Results/P515S35/preflight_pt/` (`preflight_verification.json`,
  `g_preflight.json`, `boyd_terminal.json`, sidecars, logs).
- this report.

## Changes made

### 1. `data/SRP1/SRP1_params.json` (committed, commit `52d70dc9`)

One line added to the `admm` block: `"shared_ess_initialization": "price_taker"`
(`data/SRP1/SRP1_params.json:46`). Verified loadable with zero solves and no scenario-
checksum change (`SolveProfileGuard([])`, `O.load_baseline()`, `guard.verify(0)` passed;
`shared_ess_initialization=price_taker`, `source=case_file` read back correctly).

### 2. `p515_g_g1_g4_admm_gates.py` — new `s35pt` arm (not committed)

Self-contained section (`p515_g_g1_g4_admm_gates.py:3713-4285`, approx.) inserted
immediately before `if __name__ == '__main__':`, plus one `import shared_ess_price_taker`
line and one `elif gate == 's35pt':` dispatch branch
(`p515_g_g1_g4_admm_gates.py:4689-4772`, approx.). Every other arm (`s35ref` included) is
byte-for-byte unchanged — `git diff` on the file is purely additive (671 insertions, 0
deletions, confirmed by `git diff --stat`).

Key pieces, each built ON TOP of the s35ref arm's own helpers by calling them (not
copying):

- `assert_s35pt_capture_paths(planning)` calls `assert_s35ref_capture_paths(planning)`
  (inherits every s35ref checklist item — Boyd eps, fixed sigma + its calibration
  assertion, `al_scale_esso`, S_ref 2.5, freeze parameters, 3 consecutive cycles, initial
  rho v/pf/ess — AND its pre-solve SoH floor-row identification) and adds: spec v6 file
  identity, `shared_ess_initialization == 'price_taker'` with `source == 'case_file'`, the
  new production/harness callables, and the existence (by path) of run 1's reference
  artifacts gate 3 reads.
- `s35pt_capture_hooks(...)` calls `s35ref_capture_hooks(...)` (recourse-jump/ESS-entry-
  stride/SoH-floor sidecars, unchanged mechanism, `stride=5` per the spec's permitted ESS
  stride) and layers two further monkeypatches, both restored in `finally`, both zero
  extra solves: `shared_ess_price_taker.solve_price_taker_schedule` (captures the ONE LP
  result the real wrapper computes) and `srp._initialize_shared_ess_from_price_taker`
  (captures `clipped_q_cells` and a sha256 hash of the initialized
  `consensus_vars['ess']['z']`, `_hash_consensus_ess_z`).
- `_derive_stopped_by_from_trajectory(rows, cap, required_consecutive)` — the corrected
  `stopped_by` derivation the task requires, NOT copying the s35ref writer's
  `converged_at_cycle == last cycle` shortcut. Verified against run 1's own committed
  `cycle_trajectory` (475 cycles): returns `{'stopped_by': 'boyd', 'converged_at_cycle':
  475, 'stop_run_cycles': [475, 476, 477]}` — correctly identifying run 1 as a genuine
  Boyd stop, which the committed `boyd_terminal.json` mislabels `'cap'`.
- `_s35pt_reference_values()` reads run 1's `g_baseline.json` and `s35ref_evaluation_v2.json`
  by path, sha256-hashes each, and extracts `gross_operational_cost`,
  `terminal_objective_change_abs`, and `efc.terminal_max` — never typing the numbers in.
  Also recomputes run 1's own `stopped_by` from its `cycle_trajectory` with the corrected
  derivation, for the gate-3 `validity_condition`.
- `_s35pt_gate3(...)` — the three `pass_iff` criteria (Boyd stop within 150 + no clamp;
  terminal cost within the rule-nine bar = sum of both runs' terminal objective steps;
  terminal EFC within 2% of run 1's), each reported with its own numbers, gated on
  `validity_condition` (well-posed only if run 1 itself stopped under Boyd — confirmed
  True by the corrected derivation above).
- `write_boyd_terminal_s35pt(...)` — reuses the SAME constituent helpers
  `write_boyd_terminal_s35ref` itself calls (`write_interface_settlement_detail_s31c`,
  `write_interface_voltage_terminal`, `_s32_binding_test`, `_s34_rho_gamma_freeze_trajectory`,
  `_s35ref_terminal_floor_and_efc`), NOT the s35ref top-level writer (which hardcodes the
  v5 spec identity and the defective `stopped_by` shortcut and would overwrite the SAME
  `boyd_terminal.json` path with s35ref-specific content). Adds `gate3`, the price-taker
  LP/injection capture, and the corrected `stopped_by`/`stop_run_cycles`.
- Dispatch branch `elif gate == 's35pt':` mirrors the `s35ref` branch structure exactly
  (own fresh output root `OUT_S35PT = data/SRP1/Results/P515S35_PT_run`, cap 150,
  `apply_rho=False` so case-file rho stays in force, `full_diagnostics_in_rows=True`),
  refusing to start if `OUT_S35PT` already exists (it does not).

Exact gate command (unchanged from the task, NOT run by this Worker):
```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py s35pt > data/SRP1/Results/P515S35_PT_launch.log 2>&1
```

### 3. `p515_s35pt_phase2_checks.py` (new, not committed) — Z3, Z4, Z7 (IPOPT half)

Constructs the planning object EXACTLY as the `s35pt` arm does by calling
`p515_g_g1_g4_admm_gates._construct_arm_planning` (same `O.fresh_planning`, same
`apply_rho=False`, same S_INV/E_INV candidate). Reaches the pre-cycle-1 point (see
"How the pre-cycle-1 point is reached" below) via four constructor-function monkeypatches
(`create_admm_variables`, `create_distribution_networks_models`,
`create_transmission_network_model`, `create_shared_energy_storage_model`, each a
call-through wrapper capturing its own return value) plus a monkeypatch of
`update_distribution_coordination_models_and_solve` that raises `_PreCycle1Stop` BEFORE
calling through. Z4 forces `admm.shared_ess_initialization = 'standalone'` on
`deepcopy(planning.params)` (never touches the case file).

Two independent `SolveProfileGuard` instances, each declared and checked EXACTLY at 51
permitted IPOPT solves (36 DSO SMOPF + 12 TSO SMOPF + 3 ESSO standalone `optimize` — the
same standalone-construction identity `run_admm_arm` itself asserts internally,
`51 * cycles + 51`). LP calls separately declared and checked exactly (120 for Z3 — 3
active nodes × 40 damped outer iterations, `wear_on=True` default, same as Phase 1's Z2p;
0 for Z4 — the wrapper is never called under the standalone flag).

## How the pre-cycle-1 point is reached

A monkeypatched stop on `update_distribution_coordination_models_and_solve` (cycle 1's
first network solve) that raises `_PreCycle1Stop` **before** calling through — zero
solves, zero ADMM cycles. Combined with call-through capture wrappers on the four
ADMM-model constructors that store references to their own return values — the SAME
mutable objects `_run_operational_planning` continues to mutate in place (never
reconstructed) up to the loop — so state is read out only once execution has reached
EXACTLY the pre-cycle-1 point, catching `_initialize_shared_ess_from_price_taker` and
`update_interface_power_flow_variables`, both of which run between construction and the
loop.

## Z3/Z4/Z7 results

All PASS. `data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json`:

- **Z3** (price-taker construction): `passed=True`, `n_cells_checked=864` (3 nodes × 3
  years × 4 days × 24 periods), `problems=[]`. `z` (current+prev, p) == LP p, agent
  copies == z, TSO proximal centres == z/s_base, ESSO SoH/e_available/es_pnet == LP,
  every ESS dual exactly 0 — **all** to the declared tolerances (1e-9 to 1e-12), zero
  violations. `clipped_q_cells_reported_by_wrapper=231`, independently recomputed
  `clipped_q_cells_recomputed=0` (0 cells outside the converter circle after clipping —
  confirms the wrapper's own clip is correct). `lp_call_count=120` (declared 120, exact).
  `ipopt_solve_count=51` (declared 51, exact, `guard_verify_failures=[]`).
- **Z4** (standalone construction, flag forced on a deep-copied params object):
  `passed=True`, `problems=[]`. `non_ess_pre_loop_state_identical_to_z3=True` with
  `vmag_max_abs_diff_vs_z3=0.0`, `pf_max_abs_diff_vs_z3=0.0`,
  `dual_vmag_max_abs_diff_vs_z3=0.0`, `dual_pf_max_abs_diff_vs_z3=0.0` (bit-identical,
  not merely within tolerance). ESS state differs from the LP as expected:
  `ess_z_fraction_matching_lp_schedule=0.0` (0/864 cells match the LP schedule, vs Z3's
  864/864). `price_taker_wrapper_called_under_standalone_flag=False` (confirmed never
  invoked). `lp_calls_during_z4_construction=0` (declared 0, exact). `ipopt_solve_count=51`
  (declared 51, exact).
- **Z7 (IPOPT half)**: `z3_ipopt_solve_count=51 == z4_ipopt_solve_count=51`, both match
  the declared 51. `passed=True`.

**No STOP condition from Z3/Z4**: the injection is correctly wired, not overwritten, and
capacities match the LP exactly at t=0.

## Preflight (two ADMM cycles, real `s35pt` arm)

`data/SRP1/Results/P515S35/preflight_pt/preflight_verification.json`. Rule-eleven
checklist (`assert_s35pt_capture_paths`) passed before any solve. Command run:
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s35pt_preflight.py`.

| Signature (spec v6 `preflight.expected_signatures`) | s35pt (this run) | run 1 (s35ref, read from `g_baseline.json`) | Result |
|---|---|---|---|
| EFC near LP value at cycle 1 (not ~0.0003) | cycle1 = **1.19181**, LP target = 1.19178 (ratio 1.00003) | cycle1 = 0.000287 | **PASS** — essentially exact |
| EFC change cycle 1→2 < 1% | 1.89e-05 (rel.) | — | **PASS** |
| Small ESS Boyd dual residual ratio (`boyd_ess_dual_ratio`) | cycle1 = 0.354, cycle2 = 0.387 | cycle1 = 19.399, cycle2 = 17.848 | **much smaller, as predicted** |
| Per-cell ESS λ (TSO+DSO+ESSO) sums to zero | max abs sum 2.0e-17 (c1), 2.5e-17 (c2), 1728 cells/cycle | — | **PASS** (machine precision) |
| Small TSO proximal movement on ESS (`boyd_ess_s_proximal_part`) | cycle1 = 2.07e-4, cycle2 = 1.61e-4 | cycle1 = 1.21e-2, cycle2 = 1.95e-3 | **much smaller, as predicted** |
| No solve failures/restorations attributable to init | **1** `recovered_tier1` block (see below) | 0 in cycles 1-2 (first failure at cycle 8 of 477) | **FAIL** |
| PF primal residual may exceed run 1 cycle 1 (informational) | 0.32566 | 0.32564 | reported, negligibly larger — not a failure either way |
| Diagnostics: TSO copy≈z/2, EFC≈0, ESSO pnet drift | none observed (`efc_approx_zero_at_cycle1=False`) | — | **no red flag** |
| Solve count exact (153 declared: 51×2+51) | **154 observed** (`identity_holds=False`) | — | **FAIL** — the +1 is the retry below |

### The one failing signature, in detail

`data/SRP1/Results/P515S35/preflight_pt/network_failures_preflight.jsonl` (1 line):
DSO `case33_1` (node 5), year 2030, day Winter, **cycle 2**, primary solve terminated
`maxIterations` (`warm_start=True`), automatically retried and recovered
(`class: recovered_tier1`, `termination: recovered`). This is the single extra IPOPT
call (154 vs the declared 153) and the single `network_failures_summary.n_blocks=1`.
`local_solve_failures=0` (i.e. it did not persist/block convergence).

Cross-checked against run 1: `network_failures_baseline.jsonl` (298 total blocks over
477 cycles) has **zero** blocks at cycles 1 or 2 — the first appears at cycle 8. This is
a genuine, reproducible-looking (IPOPT is deterministic here; nothing stochastic in the
pipeline) empirical difference: the price-taker initialization's aggressive,
LP-optimal storage trajectory appears to push at least one local DSO solve into a harder
warm-started region earlier than the standalone cold-start does. It recovered
automatically and did not affect the consensus/dual quality (every other signature is
clean, several by 1-2 orders of magnitude better than run 1's own cycles 1-2).

## Solve counts (whole Phase 2 task)

- Z3 construction: 51 IPOPT solves (declared, exact) + 120 LP calls (declared, exact).
- Z4 construction: 51 IPOPT solves (declared, exact) + 0 LP calls (declared, exact).
- Two-cycle preflight: 154 IPOPT solves observed against 153 declared (`run_admm_arm`'s
  own internal `SolveProfileGuard`, `N.PERMITTED` call sites; the +1 is the recovery
  retry above; `blocked_solve=0`, `blocked_exec=0` — no undeclared call site was reached,
  only the declared COUNT was off).
- Total for this task: 51+51+154 = 256 IPOPT solves, 120 LP calls, zero blocked/
  undeclared solves anywhere.

## Validation

- `ast.parse` succeeded on all three new/modified Python files before running anything.
- `git diff --stat -- p515_g_g1_g4_admm_gates.py`: 671 insertions, 0 deletions —
  confirmed additive-only; the `s35ref` section and every other arm is untouched.
- `.p515_g_gate.lock` absent and no `p515_g_g1_g4_admm_gates.py` process running,
  checked immediately before every solve-bearing run (Z3/Z4 checks and the preflight).
- `data/SRP1/Results/P515S35_REF_run/` (run 1) was only ever opened for reading; no file
  under it was modified (`git status` / mtimes unchanged).
- Commit 1 (`data/SRP1/SRP1_params.json`, hash `52d70dc9`) verified staged alone
  (`git diff --cached --name-only` showed exactly that one path) before committing.
- Commit 2 was **not** attempted, per the task's explicit gate.

## Unexpected findings

- **The one substantive finding of this task**: price-taker initialization appears to
  induce local-solve numerical stress (one `maxIterations`→retry event) earlier in the
  ADMM trajectory (cycle 2) than the standalone cold start does (first event at cycle 8
  in run 1's 477-cycle history) — despite the price-taker path otherwise giving a
  dramatically BETTER-conditioned start (EFC within 0.003% of its LP target at cycle 1,
  ESS Boyd dual residual ~50x smaller, TSO proximal movement ~60x smaller than run 1's
  corresponding cycles). This is plausibly the LP-optimal dispatch handing the DSO a
  much larger, more aggressive initial storage flow than a cold/near-zero start does,
  making that DSO's own local NLP harder to warm-start through in at least one
  (network, year, day) cell. It recovered automatically and both `local_solves_ok` and
  every consensus/dual signature stayed clean.
- The committed `s35ref` `boyd_terminal.json`'s `stopped_by` bug reproduces exactly as
  the task predicted: `_derive_stopped_by_from_trajectory` on run 1's own
  `cycle_trajectory` returns `'boyd'` at cycle 475 (three consecutive converged cycles
  475-477), while the committed artifact reads `'cap'` — confirming the corrected
  derivation is necessary and correctly implemented, and that the gate-3
  `validity_condition` (run 1 must have stopped under Boyd) is satisfied.

## Remaining issues

- Commit 2 (`p515_g_g1_g4_admm_gates.py`'s new section, `p515_s35pt_phase2_checks.py`,
  `p515_s35pt_preflight.py`, both output directories, sha256 manifests, this report) is
  **not committed**, pending Planner disposition of the one failing preflight signature.
- The gate itself (`s35pt`, cap 150) has not been launched — correctly out of scope for
  this Worker task regardless of the above.
- No manifest of sha256 hashes for the two new output directories has been produced yet
  (deferred — would normally accompany commit 2; not generated for an uncommitted,
  possibly-to-be-rerun state, to avoid manifesting a run the Planner may ask to redo).

## Questions for Planner

1. Is a single `recovered_tier1` event (auto-recovered, non-blocking, `local_solves_ok`
   still true for that cycle) at cycle 2 acceptable background noise for this preflight's
   purposes, or does it warrant deeper investigation (e.g. a larger warm-start / rho
   adjustment, or accepting that price-taker init trades a "calmer first 2 cycles"
   property for a different numerical profile) before Commit 2 / the gate launch are
   authorized?
2. Should this Worker re-run the two-cycle preflight (fresh output directory) to check
   whether the recovery event is deterministic/reproducible, or is the single observation
   sufficient evidence as-is? (IPOPT is deterministic here given identical inputs, so a
   repeat is expected to reproduce the same single event at the same point rather than
   yield new information — flagged rather than done unilaterally, since it consumes
   another 154 solves for a very likely-repeat result.)
3. If Commit 2 is authorized despite this finding, should the preflight's own
   `signature_3_no_failures` / `solve_profile_counts_exact` still read `False` in the
   committed artifact (i.e. commit the true result), or does the Planner want the
   "no failures" bar reinterpreted (e.g. "no UNRECOVERED failures") before recording it
   as a pass?
