# P5.15 — Handoff to the external expert, round 4

**Subject: the ESSO log-handling fix is implemented and passes all three authorized tests plus the
initialization check. Two findings qualify earlier conclusions: the analytic leak estimate does not
hold on the C\* ADMM path, and the Addendum-5 explanation of the 5× initialization gap is falsified.
G1 has not started: it needs a ruling on how a ~40-minute campaign may be executed.**

Planner report, 2026-09-14. Continues `P5_15_EXPERT_HANDOFF_3.md`. Authority:
`PLANNER_BRIEF_2026-09-13.md`, Addendum 5. Detail: `P5_15_LOGFIX_REPORT.md`,
`WORKER_REPORT_LOGFIX.md`. **Nothing in this round is committed.**

---

## 1. Status against the Addendum-5 order

| step | state |
|---|---|
| stage previous round by name, lock untracked | done — `138db5b5` |
| `CLAUDE.md` campaign rule (stderr, no concurrency, no detach, one gate per task) | adopted, committed `138db5b5` |
| ESSO log-handling fix, four parts | **implemented, uncommitted** |
| tests: two-cycle probe / forced failure / single-solve identity | **all pass** |
| "zero-solve" 5× check | **done with 51 solves — declared deviation (§4)** |
| G1 → G2 → G3-full → G4 | **not started — execution-mode ruling needed (§6)** |
| `REVISION_CONTEXT.md` rewrite | held |

The report cannot lead with the G1 TSO failure classification or the per-cycle detector trajectory:
G1 has not run.

---

## 2. The fix

Two production files, paths and signatures only. The Planner checked the diff: **no constraint,
objective, tolerance, ε, penalty, rho, budget, warm-start or recovery-policy line changed.**

| part | implementation | evidence |
|---|---|---|
| ESSO logs in the network logs directory | `SharedEnergyStorageData.logs_dir`, propagated like the networks'; `output_file` joined onto it | T1: planning, ESSO and TSO `logs_dir` identical and absolute |
| one log per ESSO solve, stamped | `optim_log_esso_node{id}_{init\|cycleNNN}[_recovery].txt`; `file_append='no'`; name collision → `_dupN` with a warning, never silent append; `cycle` threaded from the ADMM loop | T1: nine files, each exactly one IPOPT run; no `_dup` files anywhere |
| last-match parser | `finditer`, last `Objective` / `Complementarity` | T1: `mu_final` differs by cycle |
| absolute `results_dir`; snapshot without abort | `os.path.abspath` at construction; `_save_frozen_network_block` catches, warns, returns None | T2 |

A side-observation confirming the defect: under the pre-fix code the oracle's per-evaluation log
directories contained **no ESSO logs at all** — the ESSO wrote to whatever the process cwd was.

## 3. Tests

- **T1 — two-cycle probe at the C\* control configuration: PASS.** Nine single-run log files
  (init, cycle 1, cycle 2 × nodes 5/7/9); `mu_final` parsed per file differs across cycles.
  131.6 s; guard 154 permitted / 0 blocked.
- **T2 — deliberately triggered network failure, launched from a foreign cwd: PASS.** TSO block
  `case9 2025 Spring` forced to fail on primary and recovery. Snapshot
  `FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl` written under the absolute results
  directory; the cycle completed without exception. Unit test: an unwritable `save_dir` produces a
  `[WARNING]` and returns None. 78.6 s; guard 104 / 0.
- **T3 — single-solve detector values unchanged: PASS, bit-identical.** Maximum relative difference
  **0.0** against committed `P5151/tol_remedy_check_summary.json` (`mu_final`, `s_obj`, detector,
  spurious throughput, estimate); guard exactly 6/6.

---

## 4. Finding 1 — the analytic estimate does not hold on the C\* ADMM path

From T1's per-solve logs:

| node | solve | `mu_final` | detector | measured / estimate |
|---|---|---|---|---|
| 5 | init | 3.1316e-09 | 2.5656e-05 | 0.850 |
| 7 | init | 3.1527e-09 | 2.5838e-05 | 0.854 |
| 9 | init | 3.1275e-09 | 2.5657e-05 | 0.855 |
| 5 | cycle 1 | 2.9927e-09 | 2.5854e-05 | 1.079 |
| 7 | cycle 1 | 3.0710e-09 | 2.5853e-05 | 1.044 |
| 9 | cycle 1 | 2.9795e-09 | 2.5853e-05 | 1.077 |
| 5 | cycle 2 | 4.2524e-09 | 2.5850e-05 | 0.703 |
| 7 | cycle 2 | 4.3232e-09 | 2.5839e-05 | 0.690 |
| 9 | cycle 2 | 4.2496e-09 | 2.5856e-05 | 0.701 |

**Observation.** `mu_final` varies by **45 %** across these solves; the detector varies by
**0.78 %**. Measured/estimate spans **0.69 – 1.08**, against the **±0.35 %** established on the
fixed-`pnet` ε fixture and accepted in Addendum 4.

**Conclusion, scoped.** The **measured detector** is a direct measurement and remains valid for the
per-cycle trajectory. The **estimate** `2N·μ_final/(2·s_obj·ε)` is not reliable on this path: the
worst-case leak is essentially insensitive to `mu_final`, so a μ-scaled account of it is **not
supported** here. The identity itself is not withdrawn — it was verified on the fixture, where the
opposite leg is large — but its range of validity is narrower than recorded.

**Mechanism: unidentified.** Per-period `pch`/`pdch` at the argmax are not captured, so what fixes
the leak near 2.585e-05 cannot be determined without a solve.

**Consequence for wording.** "Estimate, ±0.35 %" (Addendum 4) holds for the fixture only. At C\*,
gates and manuscript should quote the measured detector and not the estimate.

## 5. Finding 2 — the 5× initialization gap: check done, hypothesis falsified, gap unexplained

**Declared deviation.** Addendum 5 asked for a zero-solve check on the G3-init ladder logs. **Those
logs do not exist**: pre-fix, the ESSO wrote them to the repo root, where they no longer are.
Searched: every `optim_log_node*` in the repo written in the G3-init window, and all
`P56A/evals/ladder_s*` directories — none. T4 therefore re-ran one rung (1.00 MVA / 4.00 MWh,
51 solves, 39 s) under the fixed logging. Its detector reproduced the committed G3-init value
**bit-for-bit** (2.4031141706579106e-05): the initialization solve is deterministic and the comparison
is valid.

| node | `mu_final` | `s_obj` | `μ/(2·s_obj·ε)` | measured ratio / prediction |
|---|---|---|---|---|
| 5 | 3.1489e-09 | 0.1 | 1.5745e-05 | 1.590 |
| 7 | 3.1555e-09 | 0.1 | 1.5778e-05 | 1.587 |
| 9 | 3.1304e-09 | 0.1 | 1.5652e-05 | 1.599 |

- **Addendum 5's explanation — "an `s_obj` difference at initialization" — is falsified.** `s_obj` is
  0.1 here, as on the fixture.
- **`tol = 1e-8` is in force on this path — inferred, not read.** IPOPT prints no option values at
  default verbosity. Terminal `lg(mu) = −8.6` matches a 1e-8 target, and T3 confirms the override is
  applied at the same `optimize()` entry point. No plumbing fix was needed.
- **The gap remains unexplained.** Init `mu_final` is ~3.5× the fixture's, with a further ~1.59×
  on top, and the product matches ~5.5×. Finding 1 shows the detector on this path does not follow
  `mu_final`, so that product is **not** offered as a mechanism.

---

## 6. Execution-mode conflict — needs a ruling before G1

A C\* campaign runs about 40 minutes: P5.14-N took 2,342 s for 67 cycles, and T1 implies roughly
55 s per cycle. Every tool call, including the Worker's, times out at 10 minutes, and the tool
**auto-backgrounds** long commands on its own: T1's first attempt was backgrounded after 120 s
without anyone choosing it. The adopted rule "never detached (`screen`/`nohup`/background)"
therefore cannot be met literally for G1.

**Proposal:** permit **tool-tracked background execution** for campaigns. One run, owned by the
session, exit-notified, stderr captured, lock-guarded, no `screen`, `nohup` or `&`. This
preserves the rule's purpose, which is to prevent runs that escape their owner or run concurrently.

---

## 7. Process notes

- The Worker completed in the foreground and wrote its report to disk. Its four test harnesses and
  their stdout had been left in the session scratchpad; the Planner copied them into the repo
  (`p515f_t1..t4_*.py`, `data/SRP1/Results/P515F/test_stdout/`) so the evidence is preserved.
- One stray `&` on T2's first attempt was killed before producing output (Worker-reported). No
  concurrent runs; no process remains.
- T4 reused eval id `ladder_s1` and appended to 51 untracked network logs in
  `P56A/evals/ladder_s1/logs/`. None is cited by any committed `.md`, `.py` or `.json`.
- `.p515_g_gate.lock` remains in place; the gate harness has not been run.

---

## 8. Decisions required

1. **Execution mode for campaigns** (§6).
2. **Commit the fix before G1**, so G1 records a clean `head_commit`: the two production modules,
   `WORKER_REPORT_LOGFIX.md`, `P5_15_LOGFIX_REPORT.md`, this report, `p515f_t1..t4_*.py`, and
   `data/SRP1/Results/P515F/`.
3. **At C\*, quote only the measured detector** — not the estimate (§4).
4. **Whether to identify the ~2.585e-05 leak mechanism**, which requires capturing per-period
   `pch`/`pdch` at the argmax. The cheapest route is to add that capture to G1's per-cycle
   logging, rather than running a separate experiment.

## 9. What is NOT established

- **No G1, G2, G3-full or G4 result.** The C\* TSO failure has not been re-captured or classified.
- The mechanism setting the C\*-path leak (~2.585e-05) and the 5× initialization gap.
- `tol = 1e-8` at initialization is **inferred** from the barrier value, not read from an option dump.
- T1 covers two cycles only; whether the detector remains flat over a full campaign is unknown.
- The dual-magnitude drift of the leak across a full ADMM run remains unmeasured.

## 10. Evidence (sha256, first 16)

| artifact | hash |
|---|---|
| `shared_energy_storage_data.py` (working tree, uncommitted) | `38349cc6e2bc5832` |
| `shared_resources_planning.py` (working tree, uncommitted) | `7f3133c6c8f3ca1c` |
| `data/SRP1/Results/P515F/t1_two_cycle_probe.json` | `6a3e364bb37f0eea` |
| `data/SRP1/Results/P515F/t2_failure_probe.json` | `9d485d15bcad6616` |
| `data/SRP1/Results/P515F/t3_tol_remedy_recheck.json` | `b2fa71f32d8b63e1` |
| `data/SRP1/Results/P515F/t4_ladder_zero_solve_check.json` | `bbc92b1891c73923` |
| `data/SRP1/Results/P515F/t4_ladder/ladder_s1.json` | `f503b04474b70e9d` |
| `data/SRP1/Results/P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl` | `914580cfee424fcc` |
| `WORKER_REPORT_LOGFIX.md` | `22b2e5cbc9a33283` |
| `P5_15_LOGFIX_REPORT.md` | `d3bfb454e659d263` |
| `p515f_t1_two_cycle_probe.py` | `85ec94f83bd3280f` |
| `p515f_t2_failure_probe.py` | `51e112207d38daea` |
| `p515f_t3_tol_remedy_recheck.py` | `ba1cccf2a59b5f4e` |
| `p515f_t4_ladder_init_barrier_check.py` | `fa49caad2c694ca5` |
