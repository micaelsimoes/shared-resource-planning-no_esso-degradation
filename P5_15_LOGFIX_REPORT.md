# P5.15-F — ESSO log-handling fix: implemented, four tests pass. G1 held on one ruling.

**Planner checkpoint, 2026-09-14. Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 5.**
Worker report: `WORKER_REPORT_LOGFIX.md`. Nothing in this round is committed.

## 1. The fix (production, two files, uncommitted)

`shared_energy_storage_data.py` (+118/−) and `shared_resources_planning.py` (+156/−). Planner
checked the diff for scope: no constraint, objective, tolerance, ε, penalty, rho, budget,
warm-start or recovery-policy line is touched — only signatures and plumbing.

| Addendum-5 part | implemented as | verified by |
|---|---|---|
| (1) ESSO logs in the network logs dir | `SharedEnergyStorageData.logs_dir`, propagated like the networks'; `output_file` joined onto it | T1: planning, ESSO and TSO `logs_dir` identical and absolute |
| (2) one log per solve, stamped | `optim_log_esso_node{id}_{init|cycleNNN}[_recovery].txt`, `file_append='no'`, collision → `_dupN` + warning; `cycle=iter` threaded from the ADMM loop | T1: 9 files, each exactly one IPOPT run; no `_dup` files anywhere |
| (3) last-match parser | `finditer`, last `Objective`/`Complementarity` | T1: `mu_final` differs by cycle |
| (4) absolute `results_dir`, snapshot without abort | `os.path.abspath` at construction; `_save_frozen_network_block` catches, warns, returns None | T2 |

## 2. Tests

- **T1 two-cycle C\* probe — PASS.** 131.6 s wall, guard 154 permitted / 0 blocked.
- **T2 forced network failure from a foreign cwd — PASS.** TSO `case9 2025 Spring` forced
  (primary + recovery). `FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl` written under the
  absolute results dir; cycle completed without exception. Unit test: unwritable `save_dir` →
  `[WARNING]`, returns None. 78.6 s, guard 104 / 0.
- **T3 single-solve detector values — PASS, bit-identical** (max relative difference 0.0) against
  committed `P5151/tol_remedy_check_summary.json`, guard exactly 6/6.
- **T4 initialization call site — see §4.**

## 3. New finding — the analytic estimate does not transfer to the C\* ADMM path

From T1 (C\* control configuration, per-solve logs, `ε = 1e-3`):

| node | solve | `mu_final` | detector `max min/s_max` | spurious measured | estimate `2N·μ/(2·s_obj·ε)` | measured/estimate |
|---|---|---|---|---|---|---|
| 5 | init | 3.1316e-09 | 2.5656e-05 | 7.6626e-03 | 9.0190e-03 | **0.850** |
| 7 | init | 3.1527e-09 | 2.5838e-05 | 7.7569e-03 | 9.0797e-03 | **0.854** |
| 9 | init | 3.1275e-09 | 2.5657e-05 | 7.6992e-03 | 9.0073e-03 | **0.855** |
| 5 | cycle001 | 2.9927e-09 | 2.5854e-05 | 9.2989e-03 | 8.6191e-03 | **1.079** |
| 7 | cycle001 | 3.0710e-09 | 2.5853e-05 | 9.2381e-03 | 8.8445e-03 | **1.045** |
| 9 | cycle001 | 2.9795e-09 | 2.5853e-05 | 9.2420e-03 | 8.5808e-03 | **1.077** |
| 5 | cycle002 | 4.2524e-09 | 2.5850e-05 | 8.6111e-03 | 1.2247e-02 | **0.703** |
| 7 | cycle002 | 4.3232e-09 | 2.5839e-05 | 8.5909e-03 | 1.2451e-02 | **0.690** |
| 9 | cycle002 | 4.2496e-09 | 2.5856e-05 | 8.5836e-03 | 1.2239e-02 | **0.701** |

**Observation.** measured/estimate spans **0.690 – 1.079** — i.e. roughly −31 % to +8 % —
against the **±0.35 %** established on the fixed-`pnet` ε fixture and accepted in Addendum 4.
Across the same solves `mu_final` varies **2.98e-09 – 4.32e-09** (+45 %),
while the detector stays at **2.5656e-05 – 2.5856e-05** (spread 0.78 %).

**Reading, stated as far as the evidence goes.** The *measured* detector is a direct measurement
and remains valid for the per-cycle trajectory. The *estimate* is not: on this path the worst-case
leak is essentially **insensitive to `mu_final`**, so a purely μ-scaled account of it is **not
supported** here. The mechanism setting ~2.585e-05 is **unidentified** — per-period `pch`/`pdch` at
the argmax are not captured, so it cannot be settled without a solve.

**Consequence for the manuscript wording accepted in Addendum 4:** "estimate, ±0.35 %" holds on the
ε fixture only. At C\* it must not be quoted; gates and text should use the measured detector.

## 4. T4 — the "5× gap" check

Deviation, declared: Addendum 5 asked for a **zero-solve** check on the ladder logs. **No G3-init
ESSO logs exist** — under the pre-fix code they went to the process cwd (repo root) and are gone;
searched every `optim_log_node*` under the repo written in the G3-init window, and all
`P56A/evals/ladder_s*` dirs: none. T4 therefore re-ran one rung (1.00 MVA, 51 solves, 39 s) with
the fixed logging. Its detector reproduced the committed G3-init value **bit-for-bit**
(2.4031141706579106e-05), so the init solve is deterministic and the comparison is valid.

| node | `mu_final` | `s_obj` | `μ/(2·s_obj·ε)` | measured ratio / prediction |
|---|---|---|---|---|
| 5 | 3.1489e-09 | 0.1 | 1.5745e-05 | 1.590 |
| 7 | 3.1555e-09 | 0.1 | 1.5778e-05 | 1.587 |
| 9 | 3.1304e-09 | 0.1 | 1.5652e-05 | 1.599 |

- **Addendum 5's hypothesis — "an `s_obj` difference at initialization" — is falsified:** `s_obj` is
  0.1 here, as on the fixture.
- **`tol = 1e-8` is in force on this path** — inferred, not read literally: IPOPT prints no option
  values at default verbosity; terminal `lg(mu) = −8.6` (μ ≈ 2.5e-9) matches a `1e-8` target, and
  T3 confirms the override is applied at the same `optimize()` entry point. No plumbing fix needed.
- **The gap is not explained.** `mu_final` at init is ~3.5× the fixture's, and measured/prediction
  is ~1.59 on top; the product matches the ~5.5× gap, but §3 shows the detector on this path does
  not follow `mu_final`, so that decomposition is **not** offered as a mechanism.

## 5. Process notes

- The Worker completed in the foreground and wrote its report. Its test harnesses were written to
  the session scratchpad; the Planner copied them into the repo (`p515f_t1..t4_*.py`) with their
  stdout (`data/SRP1/Results/P515F/test_stdout/`) so the evidence is preserved.
- T1's first attempt was **auto-backgrounded by the tool** after 120 s (not `screen`/`nohup`/`&`);
  it was not re-launched concurrently and left no artifacts. Relevant to the no-detach rule: tool
  timeouts can background a command without the agent choosing to.
- T4 reused eval id `ladder_s1`, appending to 51 untracked network logs in
  `P56A/evals/ladder_s1/logs/`; none is cited by any committed file (searched: all committed
  `.md`/`.py`/`.json`).
- One stray `&` on T2's first attempt was killed before output (Worker-reported).

## 6. Decisions needed before G1

1. **Execution mode.** A C\* campaign runs ~39 min (P5.14-N: 2,342 s, 67 cycles; T1 ≈ 55 s/cycle
   after init). Tool calls time out at 10 min, so "never detached/background" cannot be met
   literally. Proposal: **tool-tracked background execution** — one run, owned by the session,
   exit-notified, stderr captured, lock-guarded; no `screen`, `nohup` or `&`.
2. **Commit the fix before G1**, so G1 records a clean `head_commit`. Files: the two production
   modules, `WORKER_REPORT_LOGFIX.md`, `p515f_t1..t4_*.py`, `data/SRP1/Results/P515F/`, this report.
3. **Estimate wording** (§3): confirm that only the measured detector is quoted at C\*.

Held: `.p515_g_gate.lock` remains in place; G1 → G2 → G3-full → G4 not started.

## 7. Evidence (sha256, first 16)

| artifact | hash |
|---|---|
| `data/SRP1/Results/P515F/t1_two_cycle_probe.json` | `6a3e364bb37f0eea` |
| `data/SRP1/Results/P515F/t2_failure_probe.json` | `9d485d15bcad6616` |
| `data/SRP1/Results/P515F/t3_tol_remedy_recheck.json` | `b2fa71f32d8b63e1` |
| `data/SRP1/Results/P515F/t4_ladder_zero_solve_check.json` | `bbc92b1891c73923` |
| `data/SRP1/Results/P515F/t4_ladder/ladder_s1.json` | `f503b04474b70e9d` |
| `data/SRP1/Results/P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl` | `914580cfee424fcc` |
| `WORKER_REPORT_LOGFIX.md` | `22b2e5cbc9a33283` |
