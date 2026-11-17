# CAVEAT -- `row18_gate_addendum.json`, field `row18_code_presence_asserted_before_run`

Applies to `row18_gate_addendum.json` in this directory (sha256
`cfeb060f91efdbdf0f3ed527c13d259f2afcadd61ab33dd0a7624265b1924b76`), written by the SRP1 two-cycle
bitwise gate **r2** (`p515_s53_srp1_bitwise_gate.py`, sha256
`b63b1fb94ec305a782cb139d6e9c42be8b7a49b2ef78cb0d174f09e3d53f6766`, run at git HEAD 2167c0d9,
committed at d5a00bd9). That file is left unmodified; this note is added next to it (Planner task W62).

## What reads false, and why

Six W51 body checks under `row18_code_presence_asserted_before_run` read **false**:

- `w51:inactive_returns_where_row18_not_wired`
- `w51:inactive_deactivates_row_before_fixing_pair_at_zero`
- `w51:activate_returns_where_row18_not_wired`
- `w51:activate_refuses_unless_initialisation_state`
- `w51:activate_minimal_split_then_unfix_then_activate`
- `w51:activate_checks_defining_row_bodies_after_activation`

Cause: the committed S51G gate (`p515_s51_srp1_bitwise_gate.py`, `main()`, line 221) evaluates
`row18_code_presence()` for this field **after** `W35G.main()` returns (line 201). At that point the
W51 call counter was still installed; r2 removes it only when `S51G.main()` returns
(`p515_s53_srp1_bitwise_gate.py` at 2167c0d9, `finally: COUNTER.uninstall()`). So
`inspect.getsource` read the counter's pass-through wrappers, not the live functions. The six body
checks inspect function bodies and fail on a wrapper. The remaining 15 W51 checks in the field read
true. Despite its name, this field was **not** evaluated before the run.

## The field is not gating

`row18_code_presence_asserted_before_run` is informational. It is not one of the gate items that
decided r2's verdict.

## The gating presence evaluations passed against the live functions

Both evaluations returned **21/21 true** against the live functions:

1. **Before the run.** `w51_code_presence()` ran at the start of `main()`, before the counter was
   armed. Launch log `srp1_bitwise_gate_launch_r2.log`, 12:53:32:
   `W51 presence checks: 21/21 true`. The same 21 values are recorded as
   `w51_gate_addendum.json` -> `w51_code_presence_asserted_before_run`. The W35 presence stage also
   ran before the counter was armed (57/57 true, same log line time).
2. **After the counter was removed.** Gate item `w51_code_present_in_the_modules_that_ran`
   (`w51_gate_addendum.json` -> `gate_items`) = `all(w51_code_presence().values())`, evaluated
   after `COUNTER.uninstall()`. Result: True (log 12:55:17).

## r2 verdict: PASSED

- 153/153 solves: guard counts `permitted_solve` 153, `permitted_exec` 153, `blocked_solve` 0,
  `blocked_exec` 0.
- `GUARD.verify(153) == []`: `gate.json` -> `solve_profile` -> `process_guard`, with
  `verify_exactly` 153 and `verify_failures` [].
- W51 counter: 36/36 calls of each of `_set_row18_inactive_for_initialisation` and
  `_activate_row18_with_settlement`, 0 acting.
- 0 diffs vs the committed C* reference: `gate.json` -> `reference_comparison_GATING`, with
  `n_diffs` 0, `n_provenance_diffs` 0, over 2 cycles.
- 0 genuine diffs vs the committed W48 arm (`P515S52/srp1_bitwise_gate/arm`). Totals: provenance 1,
  tie_order 0, genuine 0.
- Retired-quadratic tripwire: 0 calls.
- Gross operational cost 740922817.2425401 (convention: `gross_operational_cost`; equal to the
  recourse at this cell). Instance: C* = 0.96875 MVA / 3.875 MWh at nodes 5, 7 and 9, investment
  year 2025, candidate_key `578636daa6d6360d...`. Cap 2.
- Launch log line: `GATE_PASS=True`.

## A fix was attempted and deliberately abandoned

W61 step 1 (20202936) changed the gate so it would remove the counter as soon as the arm finished.
The goal was a clean re-run (r3) that would write this field with correct values. r3 (d1370c4b)
refused to start before its first solve, in `p515_g_g1_g4_admm_gates._construct_arm_planning`.
The production eval directory
`data/SRP1/Results/P56A/evals/p515s44_s45_w10_snapoff_s53w51gate_578636daa6d6360d_run` already
existed from r2. That directory is named after the ARM (`s53w51gate`), and the gate's output root
(OUT_REL) cannot redirect it. Any re-run would therefore require renaming the arm. A rename
changes the gate's own provenance semantics, and the only benefit would be correcting six
non-gating values. The Planner withdrew the fix (W62). W61 step 1 was reverted in 07b25fe8, which
restores the gate script to the sha256 r2 ran. No r4 will be run.
