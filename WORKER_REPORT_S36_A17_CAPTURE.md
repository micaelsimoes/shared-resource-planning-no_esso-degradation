# Worker Report — P5.15 Addendum 17 capture and reconstruction

## Task received

Bounded task (Planner, 2026-09-16/17), "Addendum 17 capture and reconstruction". No
production edits; harness additions and new scripts only. Bounded solves only where
stated. The long replay (`s35ref_replay`) was to be **prepared, not launched**.

- **PART A** — zero-solve reconstruction of run 1's terminal per-agent shared-ESS
  consensus duals (λ) by replaying the production ADMM dual-update rule over run 1's
  own recorded (x, z, ρ) trajectory; cross-validate against the ESSO's own captured
  duals and against `g_baseline.json`'s per-agent norm trajectory.
- **PART B** — reusable harness capture hooks (TSO node-balance duals at storage buses,
  DSO reference-node-balance duals), a bounded 51-solve cycle-0 LMP capture, and the
  price-taker LP re-solved with those cycle-0 LMPs (Addendum 17 diagnostic 1(c)).
- **PART C** — prepare, but do not launch, the `s35ref_replay` arm (exact run-1
  configuration, applied through deep-copied parameter overrides).

## Correction of `WORKER_REPORT_S36_A17_DIAGNOSTICS.md` (commit `5e2ba6f9`)

That report's PART 0 item 4 states: *"no standalone pre-cycle-1 solve exists in this
pipeline at all."* **This is wrong.**
`data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json` records
**51 IPOPT solves** in the initialization, before cycle 1, in **both** its Z3
(price-taker init) and Z4 (standalone init) checks (`ipopt_solve_count: 51` in each,
`guard_counts.permitted_solve: 51`). These are the standalone per-(network, year, day)
SMOPF solves `create_distribution_networks_models` (36 = 3 DSOs × 3 years × 4 days),
`create_transmission_network_model` (12 = 3 years × 4 days) and
`create_shared_energy_storage_model` (3 = 3 active nodes) perform unconditionally as
part of **model construction**, before `create_admm_variables`'s consensus/dual state
is even read by the ADMM loop. `p515_s35pt_phase2_checks.py`'s own
`_run_precycle1_capture` already demonstrates the zero-solve technique for stopping
exactly after these 51 solves and before cycle 1's first coordination solve (a
monkeypatched raise on `update_distribution_coordination_models_and_solve`, called
through unchanged for everything before it). These standalone constructions **are**
the source of cycle-0 LMPs; the correction is that they exist, not that a new solve
type had to be invented. This correction is recorded in both new scripts' module
docstrings and is restated here per the task instruction.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 17 (and 1-16 for context/configuration
  history).
- `WORKER_REPORT_S36_A17_DIAGNOSTICS.md` and its script `p515_s36_a17_diagnostics.py`
  (commit `5e2ba6f9`).
- `data/SRP1/Results/P515S35_REF_run/`: `g_baseline.json`, `ess_entry_stride_baseline.jsonl`,
  `esso_models_baseline.pkl`, `network_failures_baseline.jsonl`.
- `data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json` and its
  script `p515_s35pt_phase2_checks.py` (the pre-cycle-1 stop technique, reused by
  calling, not reimplemented).
- `shared_resources_planning.py`: `_update_shared_energy_storage_variables` (~lines
  6962-7150, the exact ESS consensus/dual update rule), `create_admm_variables`
  (~3923-4016, zero-initial-dual verification), `create_transmission_network_model`
  (~3440-3547), `create_distribution_networks_models_sequential` (~3557-3619),
  `create_shared_energy_storage_model` (~3708-3747), `_admm_shared_ess_reference_mva`/
  `_shared_ess_admm_normalization_mva` (~4183-4208), `update_shared_energy_storages_
  coordination_model_and_solve` (~5500-5530, where the ESSO's own `dual_p_req`/
  `dual_q_req` Params are set — BEFORE that cycle's solve and dual update, a one-cycle
  timing subtlety PART A had to work through), the main ADMM loop (~2380-2600).
- `network.py` (`node_balance_p_rule`/`node_balance_q_rule` via
  `model_construction_helpers.py:1366-1420`, `model.dual` Suffix declaration at
  ~line 534; `objective_function_rule`/`generation_cost`/`interface_energy_settlement`
  at `model_construction_helpers.py:1600-1666`, establishing the p.u.→$/MWh
  conversion and the `interface_settlement_weight=0` fact at construction time).
  `network.py:71-96` (`get_reference_node_id`, `get_node_idx`).
  `shared_resources_planning.py:7362-7592` (`cost_energy_p` aliasing across
  `planning_problem`/`shared_ess_data`/every network — used by PART B's LP
  substitution).
- `shared_ess_price_taker.py` (full — `solve_price_taker_schedule`,
  `_check_prices_uniform_across_networks`, `_solve_node`'s damped fixed point,
  `DEFAULT_OUTER_ITERATIONS`/`DEFAULT_DAMPING`).
- `p515_g_g1_g4_admm_gates.py`: `_construct_arm_planning`, `run_admm_arm`, the
  `s35ref`/`s35pt` sections in full (`assert_s35ref_capture_paths`,
  `s35ref_capture_hooks`, `write_boyd_terminal_s35ref`, the `elif gate == 's35ref':`
  / `'s35pt':` dispatch blocks) — reused verbatim wherever possible for PART C.
- `p514_n_instrumented_cstar.py` (`S_INV, E_INV, INVEST_YEAR`, `PERMITTED`).
- `p513_solve_profile_guard.py` (guard mechanism).
- `p56a_oracle.py` (`fresh_planning`, `WORK_DIR`).

## Files modified / created

- Created `p515_s36_a17_parta_dual_reconstruction.py` (PART A).
- Created `p515_s36_cycle0_lmp_capture.py` (PART B).
- Modified `p515_g_g1_g4_admm_gates.py` (PART C): two purely-additive parameters on
  the shared `run_admm_arm` (`pre_solve_hook=None`, and a signature-inspected `state`
  passthrough to `post_run_hook` — both no-ops for every existing arm, verified: no
  pre-existing `post_run_hook` implementation declares a `state` parameter), plus one
  new, self-contained section (`assert_s35ref_replay_capture_paths`,
  `_s35ref_replay_force_standalone_hook`, `s35ref_replay_cycle0_lmp_hooks`,
  `write_terminal_storage_duals_s35ref_replay`, `s35ref_replay_bitwise_identity_check`)
  and one new dispatch branch (`elif gate == 's35ref_replay':`). No other arm's code
  was touched.
- Created `data/SRP1/Results/P515S36/A17_partA_dual_reconstruction/` (PART A outputs).
- Created `data/SRP1/Results/P515S36/cycle0_lmp/` (PART B outputs).
- Created this report.
- No production, case-file, or other-arm harness code modified. No file under
  `P515S35_REF_run`/`P515S35_PT_run` (or any other previously-committed result
  directory) was written to.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_a17_parta_dual_reconstruction.py
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_cycle0_lmp_capture.py
```
Both exit 0. Plus one throwaway, uncommitted verification script (scratchpad only,
not part of the repository) exercising PART C's "dry checklist" step under a
`SolveProfileGuard([], ...)` (see PART C section below).

Before any solve: verified no `p515_g_g1_g4_admm_gates.py` process running (`ps aux`)
and `.p515_g_gate.lock` absent — both confirmed clean.

## PART A — zero-solve dual reconstruction

**The rule replayed** (verbatim from `_update_shared_energy_storage_variables`,
`shared_resources_planning.py`, "Shared-ESS consensus ADMM" block), per (node, year,
day, power_type, period) entry, per cycle `c`:

```
a(c)      = 1 / (2 * S_ref(c))                      S_ref fixed at 2.5 MVA throughout
                                                      run 1 (shared_ess_reference_rating_mva,
                                                      identical across TSO/DSO/ESSO once
                                                      reference_mva is not None)
rho(c)    = g_baseline.json cycle-c row's rho_ess_before  (the value in force during
                                                      cycle c's solves, BEFORE that
                                                      cycle's own rho adaptation --
                                                      verified equal across TSO/DSO/ESSO:
                                                      they start equal and are scaled by
                                                      the SAME factor every cycle, which
                                                      is why g_baseline records ONE scalar,
                                                      not three)
z(c)      = [ sum_agent( rho(c)*a(c)^2*x_agent(c) + lambda_agent(c-1)*a(c) ) ]
            / [ sum_agent( rho(c)*a(c)^2 ) ]
lambda_agent(c) = lambda_agent(c-1) + rho(c)*a(c)*(x_agent(c) - z(c))
```

**Initial condition** (`create_admm_variables`): λ_agent(0) = 0.0 for every entry,
every agent — **verified directly from source** (the exact zero-initialization literal
is present in `create_admm_variables` for `'tso'`, `'dso'` and `'esso'`), not assumed.

**Skip rule**: production skips the ESS dual update for a (node, year, day) block at
cycle `c` unless all three of that cycle's TSO/DSO/ESSO solves report success via
`_solver_result_succeeded`. A recovered block's `results[...]` entry (after any
cold/tier-2 retry inside `update_distribution_coordination_models_and_solve`/
`update_transmission_coordination_model_and_solve`) **is** what `_solver_result_
succeeded` reads at this later point — there is no separate "still failed" flag — so
a recovered block updates **normally**. Empirically verified for run 1:
`network_failures_baseline.jsonl` classifies **all 298** entries as `recovered` (284)
or `recovered_tier2` (14) — no `failed`/unrecovered entry exists — and every one of
`g_baseline.json`'s 477 cycle rows has `local_solves_ok: True`. **Zero cycles skip the
dual update in run 1.**

**Cross-validation 1 — per-agent aggregate norm trajectory**, reconstructed λ's L2
norm at every cycle 1–477, against `g_baseline.json`'s `boyd_ess_norm_y_tso/dso/esso`
(the exact same quantity, confirmed from `get_admm_boyd_residual_metrics`'s source:
`sumsq['ess'][f'y_{agent}'] += y_agent**2`, `y_agent = dual_vars['ess'][agent]
['current'][...]`, no extra normalization):

| agent | max relative error (477 cycles) | worst cycle | pass (≤1e-9) |
|---|---|---|---|
| tso | 4.449e-15 | 313 | **yes** |
| dso | 2.722e-15 | 434 | **yes** |
| esso | 1.500e-14 | 313 | **yes** |

**Cross-validation 2 — ESSO's own captured terminal duals**
(`esso_models_baseline.pkl`, `dual_p_req`/`dual_q_req` Params). **One subtlety found
and corrected**: these Params are set from `dual_vars['ess']['esso']['current']`
**inside** `update_shared_energy_storages_coordination_model_and_solve`, **before**
that same cycle's own `shared_ess_data.optimize(...)` call and **before** that same
cycle's z/λ update runs later in the cycle — so the pickle frozen at run 1's terminal
cycle T=477 holds **λ_esso(T−1) = λ_esso(476)**, not λ_esso(477). Comparing naively
against λ_esso(477) (this script's own `lambda_terminal`) gave a spurious "failure"
(max abs error 1.19e-5, dominated by entries near the noise floor 1e-8); comparing
correctly against λ_esso(476) (`lambda_snapshot_before_cycle`, captured by the script
before applying cycle 477's own update) gives:

| | value |
|---|---|
| entries checked | 1,728 (3 nodes × 288 × {p,q}) |
| entries at/below noise floor (1e-8) | 271 |
| max abs error, unconditional, all entries | **9.449e-18** |
| max relative error, excluding noise-floor entries | **3.296e-10** |
| **pass (≤1e-9 relative, ≤1e-6 abs)** | **yes** |

**Overall PART A verdict: PASS.** Both cross-validations agree to ~1e-9 relative or
better (norm trajectory: 1e-14 to 1e-15; ESSO per-entry: 3.3e-10), i.e. floating-point
accumulation only. The reconstructed terminal per-agent, per-entry duals (both
λ(477), the run's true final state, and λ(476), what the ESSO's own Param actually
captured) are saved in `data/SRP1/Results/P515S36/A17_partA_dual_reconstruction/
reconstructed_terminal_duals.json`. **No tuning was applied to force this pass** — the
one correction made (comparing against cycle 476, not 477) is a timing fact about
when the ESSO's Param is set, verified from source, not a fitted adjustment.

## PART B — capture hooks and cycle-0 LMP capture

**Harness capture hooks** (`p515_s36_cycle0_lmp_capture.py`): `capture_tso_node_
balance_duals` and `capture_dso_reference_node_balance_duals`, general-purpose, reused
unmodified by PART C's own cycle-0 capture hook. They read `model.dual.get(constraint)`
on `node_balance_p`/`node_balance_q` at the storage buses (5, 7, 9) / each DSO's own
reference node, per (year, day, scenario, period), on an **already-solved** model — no
solve is performed by these functions.

**Sign convention and units**: `model.dual` as Pyomo returns it (IMPORT_EXPORT Suffix,
IPOPT-exported multiplier) on the constraint **as literally written**
(`Pg == Pd + Pi + slack_up - slack_down`) — **no sign flip**, matching the existing
`_s35ref_capture_hooks` SoH-floor-dual convention. **Units**: since the constraint is
in the network's own per-unit system, `LMP [$/MWh] = dual_pu / network.baseMVA`
(derived from `x_pu = x_MW/S_base` ⟹ `d(objective)/d(x_MW) = dual/S_base`; periods are
hourly and the objective's own cost terms, e.g. `generation_cost`/`interface_energy_
settlement`, are `c_p[p] * S_base * pg_pu[p]` with no separate `dt` factor, so $/MW ≡
$/MWh here). **At construction time the objective is the network's own UNSCALED base
objective** — `objective_scale`/`sigma` division happens later, inside `_run_
operational_planning`, after these 51 constructors return — so these are genuine
unscaled-objective duals, needing no further σ correction for this capture.

**Bounded guard**: construction — exactly 51 IPOPT solves declared and verified
(`guard.counts = {'permitted_solve': 51, 'permitted_exec': 51, 'blocked_solve': 0,
'blocked_exec': 0}`), reusing `p515_s35pt_phase2_checks._run_precycle1_capture`
directly (`force_standalone=True`, i.e. run 1's actual `standalone` initialization —
the case file has since defaulted to `price_taker`).

**Magnitude finding** (important, reported not hidden): the captured cycle-0 LMPs are
~1e-6 to 1e-8 $/MWh — **not** a plausible real electricity price. Cause: `model.
interface_settlement_weight` defaults to `0.00` at construction (set to `1` only later
by `_prepare_distribution_objectives_for_admm`, after these 51 solves complete), so
the standalone DSO objective at the reference bus carries essentially no priced
economic driver — only local generation cost (typically negligible on these DSO test
feeders) and slack-penalty gradients. Because the price-taker LP's objective
(`sum pi*(pdch-pch)`) is **positively homogeneous of degree 1 in price**, the optimal
schedule (charge/discharge **timing**) depends only on price's relative **shape**
across periods, not its absolute scale — so the resulting EFC/day values remain a
meaningful read of the cycle-0 duals' temporal shape even though their $/MWh magnitude
is not economically interpretable. This is reported as a finding; no rescaling was
applied (that would be tuning the input to "look right").

**Diagnostic 1(c) — price-taker LP with cycle-0 LMPs** (DSO reference-node-balance-P
dual, per node, fed as that node's own uniform system price — the production LP has
no per-node price argument, `_check_prices_uniform_across_networks`; TSO's own
node-balance-p dual at the same bus was ALSO captured, reported for comparison, but
NOT used as the LP input — both series are in the committed JSON):

- LP calls: declared 3 nodes × 40 outer iterations = 120; observed 120 (match).
- **Node 5**: converged at the production default (40 iterations); EFC/day max = 1.413
  (2025).
- **Node 7**: did **not** converge at 40 iterations (final rel_change 2.168e-4 vs tol
  1e-9); a non-default, informational-only retry at 400 iterations **still did not
  converge**, and the residual got **worse** (4.519e-4) — reported as a genuine
  non-convergence/possible divergence under this price input, not forced.
- **Node 9**: did not converge at 40 iterations (2.307e-5); converged at the extended
  400-iteration retry, EFC/day max = 1.513 (2025).
- **"Harness-definition maximum" at the production default is therefore incomplete**
  (only node 5 converged): reported value 1.413273, explicitly caveated as
  "computed from CONVERGED nodes only ... NOT directly comparable" to the other three
  references.

| quantity | value |
|---|---|
| run 1, certified (c477) | 1.058952279550704 |
| gate 3, terminal (c150) | 1.1842435189565115 |
| market-price LP (Z2) | 1.1917770631428612 |
| cycle0-LMP LP, 1(c), default budget (node 5 only, converged) | 1.413273 (incomplete — see above) |
| cycle0-LMP LP, 1(c), extended non-default retry (node 9) | 1.513 (2025), still not comparable as a "maximum" since node 7 never converges |

**No further tuning was attempted** (e.g. increasing node 7's iteration budget beyond
400, damping changes) — the task explicitly warns against tuning a failure away, and
the extended retry already showed the residual moving in the wrong direction for
node 7.

## PART C — `s35ref_replay` arm (prepared, not launched)

Added to `p515_g_g1_g4_admm_gates.py`, mirroring the existing `s35ref`/`s35pt`
preflight-then-launch structure exactly:

- **Configuration assertion**: `assert_s35ref_replay_capture_paths` reuses `assert_
  s35ref_capture_paths` **verbatim** (ρ v=0.0077/pf=0.198/ess=0.1125; freeze after 10
  unchanged cycles, backstop 60; 3 consecutive cycles; boyd `eps_abs`=1e-5/`eps_rel`
  =1e-4; `gamma_policy=tied_to_rho`, `tau=1`; `sigma_fixed`=93,635,360; `al_scale_esso`
  mode `sigma_over_median_block_weight`; `S_ref`=2.5 MVA — none of these have drifted
  since run 1) and adds the ONE field that **has** drifted: `shared_ess_initialization`
  (case file now defaults to `price_taker`; run 1 used `standalone`). The override is
  applied via `_s35ref_replay_force_standalone_hook` on a **deep-copied**
  `planning.params` — the case file is never touched — using the exact technique
  `p515_s35pt_phase2_checks.py`'s Z4 check already validated.
- **Wiring**: two small, purely-additive parameters on the shared `run_admm_arm`
  (`pre_solve_hook`, called right after construction and before any solve; a
  signature-inspected `state` passthrough to `post_run_hook`, so only a hook that
  declares a `state` parameter receives it) — verified these are no-ops for every
  existing arm (`grep` shows no existing `post_run_hook` implementation declares
  `state`).
- **Capture**: cycle-0 LMPs (`s35ref_replay_cycle0_lmp_hooks`, wrapping the two
  construction constructors, reusing PART B's own capture functions by calling them);
  terminal-cycle per-agent (TSO, DSO, ESSO), per-entry storage duals **and** the
  terminal `x`/`z` consensus copies (`write_terminal_storage_duals_s35ref_replay`,
  reading `state['dual_vars']`/`state['consensus_vars']` — confirmed present in
  `run_operational_planning`'s normal-completion `state` dict,
  `shared_resources_planning.py:3030-3034`); everything `s35ref_capture_hooks` already
  captures (recourse-jump, ESS-entry stride at stride 1, SoH floor sidecar).
- **Bitwise-identity check**: `s35ref_replay_bitwise_identity_check`, a **post-run**
  function comparing the replay's per-cycle trajectory against run 1's
  `g_baseline.json` on every field with **exact equality** (`rv != fv`, no tolerance),
  reporting the first differing cycle/field; also compares the terminal
  `ess_entry_stride_baseline.jsonl` row.
- **Dry checklist** (executed by the Worker, zero-solve, NOT committed as a script —
  a throwaway verification only): built a fresh planning object
  (`O.fresh_planning`), applied the deep-copied standalone override, called
  `assert_s35ref_replay_capture_paths`, under `SolveProfileGuard([], ...)`. Result:
  `guard failures: []` (0 permitted, 0 blocked — zero solves), **every checklist item
  True**, 3 nodes' SoH floor rows identified, `shared_ess_initialization == 'standalone'`
  confirmed on the overridden object. **PASS.**
- **Not run**: `data/SRP1/Results/P515S36_REPLAY_run/` and
  `data/SRP1/Results/P515S36_REPLAY_launch.log` do **not** exist (confirmed by `ls`
  failing on both) — the long replay was not launched, per instruction.

**Caveat**: PART C's arm code has been dry-checked (configuration assertion, zero
solves) but has **not** been execution-tested end-to-end (that would require running
the full ~477-cycle campaign, which this task forbids). The Planner should treat the
capture-hook and post-run-hook code as reviewed-but-unexercised beyond the checklist.

## Validation

- PART A: script executes (exit 0); both cross-validations pass at ≤1e-9 relative
  (see tables above); the zero-initial-dual and no-skip preconditions were verified
  from source/data, not assumed.
- PART B: script executes (exit 0); construction guard exact (51/51); LP-call guard
  exact (120/120); the two node non-convergences at the production default were
  caught and reported, not hidden or forced.
- PART C: `py_compile` on `p515_g_g1_g4_admm_gates.py` succeeds; the dry checklist
  passes under a zero-solve guard; `git diff` (below) shows no other arm's code
  touched; the replay output directory and launch log do not exist.
- `git status`/`git diff --cached --name-only` checked before each commit below.

## Unexpected findings

- The ESSO's own captured terminal duals (`esso_models_baseline.pkl`) are **one cycle
  behind** the run's true terminal ADMM state (λ(T−1), not λ(T)) — a real, source-
  verified timing fact about when `dual_p_req`/`dual_q_req` are set, not a bug in
  either the reconstruction or the capture. Any future consumer of these pickled
  Params for a "terminal dual" claim should be aware of this off-by-one-cycle
  property.
- The cycle-0 LMPs' near-zero absolute magnitude (PART B, "Magnitude finding") means
  any future use of these as a genuine $/MWh price (rather than as a price-taker LP
  input, which is scale-invariant) would be misleading without the same caveat.
- Node 7's price-taker LP under cycle-0 LMPs does not merely converge slowly but
  appears to move away from convergence between 40 and 400 iterations (2.168e-4 →
  4.519e-4) — worth a closer look if the Planner wants a completed 1(c) value for
  all three nodes.

## Remaining issues

- PART B's diagnostic 1(c) "harness-definition maximum" is incomplete (only node 5
  converged at the production default); nodes 7 and 9 need either a different
  treatment (accepting non-convergence as the finding) or a Planner-directed change
  (e.g. a different `damping`/`outer_iterations` combination, explicitly authorized,
  not chosen unilaterally here).
- PART C's `s35ref_replay` arm is prepared and dry-checked but not run; its capture
  and bitwise-identity code paths are therefore unverified beyond static/dry checks.

## Questions for Planner

1. PART B 1(c): is a genuinely completed (all three nodes converged) cycle0-LMP EFC
   value needed, and if so, is a specific non-default `outer_iterations`/`damping`
   authorized, or should the non-convergence itself be the reported result?
2. Was the choice of the **DSO's own** reference-node-balance dual (rather than the
   TSO's node-balance dual at the same interface bus) the intended price source for
   1(c)? Both are captured; only the DSO series was used.
3. Should PART C's arm be execution-tested on a SHORT cap (e.g. 5-10 cycles, not the
   full 500) as a further bounded check before the Planner launches the full replay,
   or is the dry checklist sufficient?

## Evidence / paths

- PART A script: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_a17_parta_dual_reconstruction.py`
- PART A output: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/A17_partA_dual_reconstruction/`
- PART B script: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_cycle0_lmp_capture.py`
- PART B output: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/cycle0_lmp/`
- PART C code: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_g_g1_g4_admm_gates.py` (new section before `if __name__ == '__main__':`, plus the two additive `run_admm_arm` parameters)
- Reference run 1: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35_REF_run/`
