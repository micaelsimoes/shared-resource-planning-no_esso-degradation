# CORRECTION -- commit d1370c4b: "ZERO SOLVES" was inferred, not measured

This correction concerns commit d1370c4b, "P5.15 Addendum 40 ruling 2 W61 step 2: SRP1 two-cycle
bitwise gate r3 FAILS before the first solve ... - ZERO SOLVES". Its evidence is
`srp1_bitwise_gate_launch_r3.log` (sha256
`403d605cba2945b60fc2569f45c5f10d51c13fd55924376ff1e6a08b5e8636f9`),
`srp1_bitwise_gate_launch_r3_manifest_sha256.json`, and
`srp1_bitwise_gate_r3/arm/stdout_s39_D.log` (0 bytes). The correction is recorded going forward
(Planner task W62). The commit is not amended and its evidence files are not modified.

## What is wrong

The commit title states "ZERO SOLVES" as a fact. It was **not measured**. The process guard
(`W10.GUARD`, a `SolveProfileGuard` installed when W10 was imported) was armed for the run. However,
its counts were never printed and `GUARD.verify` was never called. The `RuntimeError` propagated out
of the gate and ended the process before the W35 gate reached its solve-profile stage
(`p515_s50_generalization_gate.py`, `GUARD.verify` at line 231). The repository rule requires solve
claims to be armed and verified, never asserted. The title's claim is therefore unsupported.

## What is actually known

- `p515_g_g1_g4_admm_gates._construct_arm_planning` (line 1095) raised the refusal
  `refusing to start: eval dir already exists (network logs append): ...` **before**
  `O.fresh_planning(eval_name)` (line 1098). No planning object had been constructed at that point.
- Read from the code path and the traceback, no solve is **believed** to have happened. This is an
  **inference**. No guard verification backs it: no guard counts were logged, and no `gate.json`,
  `row18_gate_addendum.json` or `w51_gate_addendum.json` was written.
- The only output was the 0-byte arm stdout log, which `run_arm` created before the refusal.

Read d1370c4b's "ZERO SOLVES" as "zero solves inferred from the code path; not guard-verified".
