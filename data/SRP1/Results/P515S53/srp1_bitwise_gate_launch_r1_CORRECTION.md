# CORRECTION -- commit 8c8f55cc: "ZERO SOLVES" was inferred, not measured

This correction concerns commit 8c8f55cc, "P5.15 Addendum 40 ruling 2 W59 step 3: SRP1 two-cycle
bitwise gate FAILS at PRECONDITION (gate-harness defect: presence checks read the armed counter's
wrappers) - ZERO SOLVES, no arm run". This was gate run r1. Its launch log has no run suffix. The
evidence is `srp1_bitwise_gate_launch.log` (sha256
`b38a6fba95a67ea7269e2caf03abc8292cddd020280c8ac92279511c348f0806`) and
`srp1_bitwise_gate_launch_manifest_sha256.json`. The run was at HEAD 3343c4da. The correction is
recorded going forward (Planner task W63). The commit is not amended and its evidence files are not
modified. It is a sibling of `srp1_bitwise_gate_launch_r3_CORRECTION.md`, which corrects the same
fault in d1370c4b (r3).

## What is wrong

The commit title states "ZERO SOLVES" as a fact. It was **not measured**. The process guard
(`W10.GUARD`, a `SolveProfileGuard` installed when W10 = `p515_s45_snapshot_off_two_cycle_gate` was
imported) was armed for the run. However, its counts were never logged and `GUARD.verify` was never
called. The launch log ends at the six `[PRECONDITION FAILED]` lines. The W35 gate's
`raise SystemExit(1)` (`p515_s50_generalization_gate.py` line 179, sha256 `5abfe5c2...cc43` as run)
propagated through the `try/finally` blocks of `p515_s51_srp1_bitwise_gate.main` and
`p515_s53_srp1_bitwise_gate.main`, neither of which catches it. It ended the process with exit code 1
before the W35 solve-profile stage (`GUARD.verify` at s50 line 231), and before s53's own
`guard_counts_at_end`. The repository rule requires solve claims to be armed and verified, never
asserted. The title's claim is therefore unsupported.

The same applies to two figures in the commit body:
- "observed 0" (solves). It is not in the r1 log.
- "W51 counter never exercised (0 calls of each)". s53 logs the counter only after `S51G.main()`
  returns, and that call never returned.

The body's "GUARD counts all 0" comes from a separate in-process reproduction. It is not from the r1
process. It was printed and not verified with `GUARD.verify`. The only files committed with 8c8f55cc
are the launch log and its manifest, so the reproduction's output is not preserved there.

## What is actually known

- The refusal was raised in the W35 gate's precondition stage: the `if failures:` block at s50 lines
  176-179, following `w35_code_presence()`. The following were all never reached:
  - the capture-path checklist (s50 line 181 onward);
  - the legacy run lock `G._acquire_exclusive_run_lock()` (s50 line 208);
  - `W10.run_arm(ARM, 'on', ...)` (s50 line 216), which is the arm run itself.
- Everything that ran before the refusal is read-only with respect to the solver: s53's W48-arm pin
  check, `w51_code_presence()` and `W10.derive_solves_from_case_file()` (JSON read), and s50's
  `W10.check_preconditions` (lock-file existence, `ps aux`, `git status`, reference sha256, case-file
  parameter read, candidate key). Taken with the log, the code path therefore implies that no solve
  happened. This is an **inference**. No guard verification backs it: no guard counts were logged,
  and no `gate.json`, `row18_gate_addendum.json` or `w51_gate_addendum.json` was written by r1.
- When r1 ran, no output root `srp1_bitwise_gate/` existed. The directory that now exists at that
  path was written later, by r2.

Read 8c8f55cc's "ZERO SOLVES" (and "observed 0") as "zero solves inferred from the code path; not
guard-verified".
