# P5.15 — Handoff to the external expert, round 3

**Subject: G3-init passes. G1, G2, G3-full and G4 are BLOCKED by a campaign-safety defect in the
ESSO's solver-log handling — three items, one subsystem, one of them a bug in code the Planner
committed in remedy (h). No gate result at C\* exists yet. Author authorization is needed for a
production fix before any ADMM gate can produce a valid result.**

Planner report, 2026-09-14. Continues `P5_15_EXPERT_HANDOFF_2.md`. Authority:
`PLANNER_BRIEF_2026-09-13.md`, Addendum 4. Detail: `P5_15_G1_G4_BLOCKED.md`,
`WORKER_REPORT_G1_G4.md`.

---

## 1. Status against the Addendum-4 dispatch

| step | state |
|---|---|
| G5 recorded as passed under re-specified criterion | done — verified independently, committed `df46f118` |
| commit round 2 | done — `df46f118` |
| **G3 initialization (1.00 / 1.25 / 1.62 MVA)** | **PASS** |
| **G1** control arm, C\* | **BLOCKED** — two attempts, no result |
| **G2** k = 10,000 | **BLOCKED** — not run |
| **G3 full cold evaluation, 1.62 MVA** | **BLOCKED** — not run |
| **G4** determinism | **BLOCKED** — not run |
| `REVISION_CONTEXT.md` rewrite | **held** |

**The report cannot lead with G2 and the per-cycle detector trajectory at C\*, as requested,
because neither exists.** Section 3 explains why, and why the trajectory would have been
fabricated had the campaigns run.

---

## 2. G3 initialization stage — PASS

| rung (node 7, 4× energy) | all ESSO nodes succeeded | failed | solves | detector (absolute form) |
|---|---|---|---|---|
| 1.00 MVA / 4.00 MWh | True | none | 51 | 2.4031e-05 |
| 1.25 MVA / 5.00 MWh | True | none | 51 | 2.3784e-05 |
| 1.62 MVA / 6.48 MWh | True | none | 51 | 2.3418e-05 |

Zero `maxIterations`, zero recovery events, guard `blocked_solve = 0` throughout.

**Attribution caveat, carried forward.** The pre-reformulation failure at 1.00 MVA was on the
**capacity** rows (`rated_s_capacity_unit: 6.845e-05`) with `energy_storage_normalization: 0.0`.
This pass is attributable to Candidate 2 (investments as parameters), not to the complementarity
deletion.

**One observation, not interpreted.** The initialization-stage detector (~2.4e-05 absolute, near
constant across capacity) is roughly **5× the 4.5e-06 measured on the ε fixture at
`tol = 1e-8`**. The initialization requests differ from the fixture's ±10 % duty cycle, and
`mu_final`/`s_obj` were not parsed for these solves, so the difference is **unexplained**. It is
also **not verified from the logs that `tol = 1e-8` was in force** on this path, although remedy
(h) applies it at the `optimize()` entry point the ladder uses. Near-constancy across capacity is
consistent with the leak δ being absolute rather than scaling with `s_max`, as previously
established. It remains below the Addendum-3 fallback trigger in relative-throughput terms only
if the duty at C\* is material, which G1 was meant to measure.

---

## 3. Why G1–G4 are blocked

### 3.1 The defect

Three items, all the same defect family — **the ESSO's solver-log diagnostics are not
campaign-safe**:

1. **Log path.** `shared_energy_storage_data.py:1007-1014` writes each node's IPOPT log to a
   **bare relative filename** (`optim_log_node_{id}.txt`) with `file_append = 'yes'` and no
   `logs_dir` awareness. `network.py:520-521` already resolves its `output_file` against
   `network.logs_dir`. **The ESSO is the only solver family without log isolation.**
2. **Failure-snapshot path.** Because of (1), every harness must `os.chdir` into a private
   directory to parse its own logs. Across a full ADMM campaign that window is the whole run.
   When a TSO solve fails inside it, production's handler resolves the **relative**
   `transmission_network.results_dir` against the wrong directory:
   ```
   save_failed_tso_block -> _save_frozen_network_block -> os.makedirs(save_dir)
   FileNotFoundError: [Errno 2] No such file or directory: 'data'
   ```
   **A recorded failure becomes an aborted campaign.** That is how G1 died.
3. **Parser — a bug in code the Planner committed (`df46f118`, remedy (h)).**
   `_parse_ipopt_barrier_terms` uses `re.search`, which returns the **first** match. With
   `file_append = 'yes'`, a multi-cycle campaign reports **cycle 1's `mu_final` and `s_obj` for
   every later cycle** — silently, with plausible numbers. Verified by the Planner by reading the
   parser, and empirically by the Worker on a two-cycle probe.

### 3.2 What item 3 means for the evidence already reported

**No published result is invalidated.** The ε check, the `tol` remedy check, G5 and G5B each ran
their arms in isolated directories against fresh logs, so every parse saw exactly one solve. The
bug bites only multi-cycle campaigns, and none completed.

**Had G1–G4 run, the per-cycle detector trajectory Addendum 4 asks the report to lead with would
have been fabricated.** Items (1) and (3) together are why the detector cannot be logged per cycle
as specified until the fix is in.

### 3.3 Independent corroboration

The Worker assigned to G1–G4 diagnosed the same root cause — same line numbers, same traceback —
without seeing the Planner's analysis, and then **honoured the hold** rather than running the gates
under a harness-only workaround it had built and verified. Its question whether the hold came from a
concurrent session: it did not; it was written by the Planner in the same session.

---

## 4. A real signal, lost

**A TSO local solve failed during G1 at C\*.** That is gate-relevant. It is uncounted and
unclassified: the campaign aborted inside the failure handler before it could be recorded as
recovered or not, and **no `FrozenSMOPF` snapshot was captured** (searched: all `FrozenSMOPF`
directories under `data/` for files written after 2026-09-13 22:00 — none). The raw logs of both
attempts are preserved under `data/SRP1/Results/P515G/debug_attempt1/`. Whether this failure is
isolated or the start of a P5.14-N-like pattern is **unknown**.

This is the **fourth capture gap** in the programme, and the first caused by the failure-capture
path itself.

---

## 5. Process incidents this round

- **Worker non-completion and concurrent launches.** The G1–G4 Worker returned six consecutive
  non-answers while backgrounding work. During that period it (a) ran G1 concurrently with the
  G3 ladder, which interleaved both into the same ESSO logs and silently killed G1's first
  attempt; (b) **twice launched G1, G2, G4b and G3-full simultaneously** — a configuration that
  would not have failed but would have **fabricated** every per-cycle detector value through log
  interleaving and the first-match parser; and (c) detached a G4b run with `screen -dmS`. The
  Planner stopped every one of these runs. None produced an artifact. **No production file was
  modified at any point** — verified after each intervention. The Worker's final report, once it
  completed, was accurate and correct.
- **Planner controls added (harness-only, uncommitted).** `p515_g_g1_g4_admm_gates.py` now takes
  an exclusive lock and refuses concurrent execution with an explanation; verified to refuse. A
  deliberate hold file `.p515_g_gate.lock` blocks the harness entirely until the fix is authorized;
  it states it is not a stale lock and how to release it.
- **The original gate harness captured stdout only**, so G1's first failure left a 0-byte log and
  no traceback. The relaunch captured both streams; that is how the root cause was found.

**Proposed rule for `CLAUDE.md`:** *a harness that runs a campaign must capture stderr and must
refuse concurrent execution when any production path it relies on writes to a shared or relative
location.* Same family as rule eleven: the failure mode is silent, plausible data rather than a
crash.

---

## 6. Decisions required

1. **Authorize the three-item fix** (§3.1), as one change to one subsystem:
   (i) resolve the ESSO `output_file` against a logs directory, as `network.py` does;
   (ii) absolutize `results_dir` at construction so the failure-snapshot path is cwd-independent;
   (iii) parse the **last** match in `_parse_ipopt_barrier_terms`.
   Item (iii) is unambiguous enough to fix on sight, but it is production code in a subsystem
   whose other two items need your decision, so it is held with them.
   **Recommended gate for the fix itself:** a two-cycle ADMM probe showing distinct per-cycle
   `mu_final` values, a deliberately provoked network failure producing a `FrozenSMOPF` snapshot
   without aborting, and bit-identity of single-solve detector values against the committed
   `tol_remedy_check_summary.json`.
2. **Confirm the harness-only workaround is not acceptable** for reporting G1–G4. The Planner's
   position: a workaround would unblock the runs but leave the defect live for every future caller,
   and gates run under it should not be reported as though the infrastructure were sound.
3. **After the fix, run G1–G4 strictly sequentially**, stderr captured, with the Addendum-4
   per-cycle detector logging. The report will then lead with G2 and the per-cycle trajectory as
   requested.
4. **The initialization-stage detector discrepancy** (§2, ~5× the fixture value) — decide whether it
   warrants a zero-cost log check of `tol` and `mu_final` on the existing ladder solves before G1, or
   is left for G1's own per-cycle capture.

---

## 7. What is NOT established

- **No G1, G2, G3-full or G4 result exists.** Nothing is known about SoH reconciliation against the
  old control, recourse at C\*, convergence of the k = 10,000 arm, determinism, or the per-cycle
  detector trajectory.
- The dual-magnitude drift of the leak across ADMM cycles remains a **derivation, unmeasured**.
- The number and pattern of TSO failures at C\* are unknown; one occurred.
- The initialization-stage detector values are measured; their cause and whether `tol = 1e-8` was in
  force on that path are **not verified**.
- The Addendum-3 fallback trigger for (d)/(e2) has still **not been evaluated at C\***.

---

## 8. Evidence inventory (sha256, first 16)

| artifact | hash |
|---|---|
| `data/SRP1/Results/P515G/ladder/ladder_s1.json` | `7855636c15ae6445` |
| `data/SRP1/Results/P515G/ladder/ladder_s1.25.json` | `c53a5d77fd199c9d` |
| `data/SRP1/Results/P515G/ladder/ladder_s1.62.json` | `8db00c7f4e24dc16` |
| `p515_g_g1_g4_admm_gates.py` (with Planner lock guard) | `3358422657404733` |

Failure evidence (logs, not results): `data/SRP1/Results/P515G/debug_attempt1/`,
`data/SRP1/Results/P515G/g*_console.log`.

Reports: `P5_15_G1_G4_BLOCKED.md`, `WORKER_REPORT_G1_G4.md`. Prior rounds:
`P5_15_EXPERT_HANDOFF_2.md`, `P5_15_G5_REPORT.md` (committed `df46f118`),
`P5_15_EXPERT_HANDOFF.md`, `P5_15_GATE_HOLD_REPORT.md` (committed `8af3e242`).

**Uncommitted at time of writing:** this report, `P5_15_G1_G4_BLOCKED.md`,
`WORKER_REPORT_G1_G4.md`, `p515_g_g1_g4_admm_gates.py`, the three ladder artifacts, and the hold
file `.p515_g_gate.lock` (which should not be committed).
