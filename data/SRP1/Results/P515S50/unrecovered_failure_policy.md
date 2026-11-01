# Unrecovered-failure policy for long runs (P5.15 Addendum 35, W35 item 1(b))

**Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 35; frozen spec v20
`data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json`, item `1_harness`.

**Verdict: production ALREADY behaves this way. Nothing in production was changed.** The policy is
recorded here and in the docstring of `p515_s44_scale_measurement.py`; the only code change made by
W35 item 1 is to the *harness's* solve accounting (item 1(a) below).

**Evidence (zero solves):** `p515_s50_harness_identity_checks.py`
(sha256 `2a2caea82bc42575a13fc7c911d62b4ad751d303a1aebc34fd22182d73b139c3`) →
`data/SRP1/Results/P515S50/harness_identity_checks/harness_identity_checks.json`
(sha256 `ad6136e0301dd1fff762f2d1e25251fd53e5648701c1da288ce2999198ad25c7`), launch log
`data/SRP1/Results/P515S50/harness_identity_checks_launch.log`. `SolveProfileGuard(permitted=())`
armed for the whole run and `verify(0)` exact; the ladder module's own import-time `PARENT_GUARD`
verified at 0 as well. `git HEAD` at run: `a4d23ab9`. Every cited line was read with
`inspect.getsourcelines`; none of the cited code was executed.

> **Two runs, because line numbers move.** W35 item 3 edits
> `shared_resources_planning._get_interface_reporting_detail` (≈ line 893), which shifts every
> `shared_resources_planning.py` line cited below. The checks were therefore run twice, to two
> output directories, and both are committed:
> `harness_identity_checks/` (tree at `a4d23ab9`, before item 3;
> `shared_resources_planning.py`: break 3270-3272, `_admm_local_solves_succeeded` 6940-6951,
> `cycle_convergence` 2954, reset 2958, streak test 2959, AA skip 2848-2849, recourse 2862,
> rho 2968, residual 2831) and `harness_identity_checks_post_item3/` (tree after item 3, the
> numbers cited below). Neither run asserts a line number: both re-derive every one with
> `inspect.getsourcelines` on the tree they run against.

**Post-item-3 verification:**
`data/SRP1/Results/P515S50/harness_identity_checks_post_item3/harness_identity_checks.json`
(sha256 `f6f3da5fb9dedba7587ff063fa311171ab3a40ce3c7980e932bdb3a9f5df4215`), launch log
`data/SRP1/Results/P515S50/harness_identity_checks_post_item3_launch.log`. Same script, same
verdict (`ALL_OK=True`, guard `verify(0)` exact).

---

## The policy

### (1) Continue with the last iterate

An unrecovered local network solve neither stops the run nor overwrites the block's values.

| Mechanism | Citation |
|---|---|
| The SMOPF is solved with `load_solutions=False` | `network.py:608` — `result = solver.solve(model, tee=..., load_solutions=False)` |
| The solution is loaded into the model **only** on success | `network.py:854` (`model.solutions.load_from(result)`), guarded by `network.py:852` (`if solver_result_succeeded(result):`) |
| A failed tier-2 attempt restores the multiplier suffixes | `network.py:849-850` (`if not solver_result_succeeded(tier2_result): _restore_multiplier_suffixes(...)`) |
| The ADMM cycle loop's only `break` is on convergence | `shared_resources_planning.py:3373-3375` (`if convergence: ... break`); exactly one `break` in the loop body |

So after an unrecovered failure the block's Pyomo model still holds the previous cycle's values —
the run continues from the last iterate rather than from a partial or failed solution.

### (2) Count the event

The failure is written as **one event**, carrying its attempt flags, to
`network_failures_s<label>.jsonl` by the production-log parser
`p515_g_g1_g4_admm_gates._scan_and_write_network_failures`; the flags are created in
`_new_network_event` (`p515_g_g1_g4_admm_gates.py:558`, `'recovery_attempted'` /
`'tier2_attempted'`). The harness credits those attempts — see item 1(a) below.

### (3) No unrecovered failure inside the certifying cycles

| Mechanism | Citation |
|---|---|
| `local_solves_ok` is False for the cycle | `shared_resources_planning.py:7043-7054` (`_admm_local_solves_succeeded`) |
| `residual_convergence` is forced False and a warning is printed | `shared_resources_planning.py:2932-2934` |
| `cycle_convergence = boyd_all_pass and local_solves_ok` | `shared_resources_planning.py:3057` |
| the consecutive-converged counter is **reset to 0** | `shared_resources_planning.py:3061` |
| certification needs `consecutive_converged_cycles >= minimum_consecutive_converged_cycles` | `shared_resources_planning.py:3062` |
| the Anderson step is skipped | `shared_resources_planning.py:2951-2952` (`aa_state.skip_on_failure(iter)`) |
| no recourse is reported for the cycle | `shared_resources_planning.py:2965` (`if local_solves_ok:`) |
| rho is not updated | `shared_resources_planning.py:3071` (`allow_update=local_solves_ok`) |

Consequence, which is the operative rule for long runs: **a certified point cannot contain an
unrecovered failure inside its 10 certifying cycles.** One such failure anywhere in the run is
tolerated (the run continues); one inside the last 10 cycles cannot be, because the streak is reset
and certification is impossible until 10 further clean, Boyd-passing cycles have run.

---

## Item 1(a): the harness's solve identity (the only change)

The scale harness now GATES on the ladder's per-EVENT identity

```
observed == base + sum over network-failure events of [recovery_attempted] + [tier2_attempted]
```

(`p515_s44_scale_measurement.event_level_solve_reconciliation`), crediting every retry actually
attempted, **recovered or not**. The pre-W35 recovered-only rule
`observed == base + tier1 + 2 x tier2` (counts taken from the summary's *classes*) is computed and
reported beside it, never gated on. An ESSO recovery event, an `indeterminate` network event, or an
event file that does not hold `summary.n_blocks` events makes the reconciliation **unsupported**,
which fails loudly; it is never silently credited.

The rule is **replicated, not imported**, because `p515_s49_flex_ladder_campaign` installs
`SolveProfileGuard(permitted=())` at module import and `p515_s49_flex_price_gate` installs W10's
armed guard the same way — either import inside the scale harness's *solving* cycle child would
install a foreign guard. The replica is checked against the ladder's own
`solve_reconciliation` on identical inputs (`replica_agrees_with_ladder_everywhere: true`).

### Zero-solve verification over the committed failure summaries

| Run | declared base | observed | attempted retries credited | per-event expected | per-event identity | recovered-only rule |
|---|---|---|---|---|---|---|
| `paper_cycle_snapoff_memfix_r1` (78a9b230, ONE UNRECOVERED BLOCK) | 166 | 168 | 2 | 168 | HOLDS | 166 **FAILS** |
| `srp1_cycle_snapoff_r1` | 102 | 102 | 0 | 102 | HOLDS | 102 holds |
| `s47_recert 070f833e1e318f85_c_star` | 4488 | 4528 | 40 | 4528 | HOLDS | 4528 holds |
| `s47_recert bd504ecf5a288d44_n7_4h_e1` | 5763 | 5764 | 1 | 5764 | HOLDS | 5764 holds |
| `s49_flex_ladder 0513054158970f16_n7_4h_e1_m3` | 7344 | 7344 | 0 | 7344 | HOLDS | 7344 holds |
| `s49_flex_ladder 50dea31c5780df57_x0_m2` | 6885 | 6885 | 0 | 6885 | HOLDS | 6885 holds |
| `s49_flex_ladder 74eda68d74d990b2_n7_4h_e1_m2` | 6783 | 6784 | 1 | 6784 | HOLDS | 6784 holds |
| `s49_flex_ladder 75c65e733528d4c4_x0_m3` | 4233 | 4233 | 0 | 4233 | HOLDS | 4233 holds |
| `s49_flex_ladder aa8a76d71ae49f13_x0_m1p5` | 5253 | 5253 | 0 | 5253 | HOLDS | 5253 holds |
| `s49_flex_ladder f9eae48ff6133f6c_n7_4h_e1_m1p5` | 4845 | 4847 | 2 | 4847 | HOLDS | 4847 holds |

10 records checked. The per-event identity holds on **every** one, including the unrecovered case
where the recovered-only rule fails; the two rules agree exactly on every run with no unrecovered
or not-attempted block.

Four scale-measurement directories hold no cycle record and are listed with the reason rather than
skipped: `paper_cycle_snapoff_r1` and `paper_cycle_snapoff_r2` were memory-watchdog aborts of the
cycle child (exit 97, which writes no cycle record by design); `srp1_calibration` and
`srp1_calibration_r2` were build-only runs.
