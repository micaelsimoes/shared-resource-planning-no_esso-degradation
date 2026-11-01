# P5.15 — Gate G3 full evaluation (1.62 MVA / 3.24 MWh at node 7): converges; "zero failures" NOT met

**Planner report, 2026-09-14.** Authority: `PLANNER_BRIEF_2026-09-13.md` Step 1 gate G3 ("then one full cold
evaluation at 1.62 MVA / 3.24 MWh (node 7 only, other nodes zero) — converged, zero failures"), Addendum-6
capture. Run r2 at `c2361c26` (the first attempt stopped on a false harness pre-check and is preserved in
`P515G3F/`). One background run through the tool, stderr captured, lock held, **Python exit code 0**
(recorded in `P515G3F_r2_exit_code.txt`). Objective convention: recourse = `gross_operational_cost`.

**Result:** converged — yes. Zero failures — **no** (two local failures, one unrecovered).

---

## 1. Network failures

| class | count |
|---|---|
| recovered | **16** (TSO 8, DSO node 7 `case33_2` 8) |
| **unrecovered** | **1** — DSO node 7 `case33_2` 2035 Autumn, cycle 38 |
| not attempted | **1** — DSO node 5 `case33_1` 2030 Spring, cycle 11 (`maxIterations`, ineligible) |
| unparsed rows | 3 (empty; parser residue, unexplained) |

Raw-print reconciliation: 17 cold retries, 16 "recovery solve succeeded", 17 primary non-convergences, 1 plain
non-convergence.

**TSO — all recovered:**

| block (`case9`) | cycle | class |
|---|---|---|
| 2030 Spring | 10 | recovered |
| 2035 Summer | 33 | recovered |
| 2035 Spring | 41 | recovered |
| 2035 Winter | 49 | recovered |
| 2025 Spring | 50 | recovered |
| 2035 Winter | 54 | recovered |
| 2035 Summer | 59 | recovered |
| 2030 Spring | 66 | recovered |

DSO node 7 recovered: 2035 Spring (11), 2035 Autumn (18), 2035 Autumn (26), 2030 Summer (31), 2030 Winter (31), 2035 Summer (39), 2025 Winter (69), 2035 Summer (69).

**The first unrecovered failure in any gate.** Node 7 `case33_2`, 2035 Autumn, cycle 38: primary solve
`maxIterations` (warm start) → cold retry → `maxIterations` again. The same block had recovered twice earlier
in this run. The non-aborting failure handler saved the failing pre-solve block **into the arm's own root**:
`P515G3F_r2/results/FrozenSMOPF/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl` (sha256
`4499cf99a4341eb9`) — a reproducible fixture of
an unrecovered network failure, the first this programme has captured under the current policy.

Cycles with a failed local solve: 11, 38. The run converged regardless.

## 2. Per-cycle detector trajectory (node 7; nodes 5 and 9 have no active cohort-periods)

- Terminal `lg(mu)`: {-8.6: 243} over all 243 ESSO solves.
- Cycles 1–80, measured detector `min/s_max`: **1.5294e-05 – 1.5462e-05**, no drift.
- **Normalisation.** The harness compares the detector (a ratio, `min/s_max`, with `s_max = 1.62`) against
  an absolute prediction, which reads 0.61. In absolute terms the small leg is `detector × s_max` ≈ 2.50e-5,
  matching the argmax min-leg exactly, and **absolute x_small / `μ_barrier/(s_obj·ε)` = 0.9864 – 0.9972**.
  The leak is the same barrier value as at C\*; at larger `s_max` the ratio-form detector is smaller.

| round | detector (`min/s_max`) | absolute small leg | `lg(mu)` | argmax `r_bar` |
|---|---|---|---|---|
| init | 0.0000e+00 | 0.0000e+00 | -8.6 | — |
| 001 | 1.5462e-05 | 2.5048e-05 | -8.6 | 0.9972 |
| 002 | 1.5453e-05 | 2.5034e-05 | -8.6 | 0.9972 |
| 003 | 1.5442e-05 | 2.5017e-05 | -8.6 | 0.9972 |
| 004 | 1.5393e-05 | 2.4937e-05 | -8.6 | 0.9972 |
| 005 | 1.5433e-05 | 2.5001e-05 | -8.6 | 0.9972 |
| 006 | 1.5350e-05 | 2.4867e-05 | -8.6 | 0.9941 |
| 007 | 1.5369e-05 | 2.4898e-05 | -8.6 | 0.9972 |
| 008 | 1.5461e-05 | 2.5046e-05 | -8.6 | 0.9972 |
| 009 | 1.5439e-05 | 2.5011e-05 | -8.6 | 0.9972 |
| 010 | 1.5442e-05 | 2.5016e-05 | -8.6 | 0.9972 |
| 011 | 1.5294e-05 | 2.4776e-05 | -8.6 | 0.9930 |
| 012 | 1.5426e-05 | 2.4990e-05 | -8.6 | 0.9972 |
| 013 | 1.5435e-05 | 2.5004e-05 | -8.6 | 0.9967 |
| 014 | 1.5445e-05 | 2.5021e-05 | -8.6 | 0.9970 |
| 015 | 1.5412e-05 | 2.4968e-05 | -8.6 | 0.9965 |
| 016 | 1.5424e-05 | 2.4988e-05 | -8.6 | 0.9966 |
| 017 | 1.5433e-05 | 2.5001e-05 | -8.6 | 0.9972 |
| 018 | 1.5418e-05 | 2.4977e-05 | -8.6 | 0.9972 |
| 019 | 1.5421e-05 | 2.4982e-05 | -8.6 | 0.9972 |
| 020 | 1.5369e-05 | 2.4897e-05 | -8.6 | 0.9956 |
| 021 | 1.5385e-05 | 2.4924e-05 | -8.6 | 0.9959 |
| 022 | 1.5404e-05 | 2.4954e-05 | -8.6 | 0.9963 |
| 023 | 1.5403e-05 | 2.4954e-05 | -8.6 | 0.9963 |
| 024 | 1.5403e-05 | 2.4953e-05 | -8.6 | 0.9963 |
| 025 | 1.5411e-05 | 2.4965e-05 | -8.6 | 0.9965 |
| 026 | 1.5406e-05 | 2.4957e-05 | -8.6 | 0.9964 |
| 027 | 1.5430e-05 | 2.4997e-05 | -8.6 | 0.9972 |
| 028 | 1.5419e-05 | 2.4978e-05 | -8.6 | 0.9966 |
| 029 | 1.5424e-05 | 2.4987e-05 | -8.6 | 0.9967 |
| 030 | 1.5429e-05 | 2.4995e-05 | -8.6 | 0.9967 |
| 031 | 1.5433e-05 | 2.5002e-05 | -8.6 | 0.9968 |
| 032 | 1.5435e-05 | 2.5004e-05 | -8.6 | 0.9968 |
| 033 | 1.5440e-05 | 2.5014e-05 | -8.6 | 0.9972 |
| 034 | 1.5450e-05 | 2.5029e-05 | -8.6 | 0.9972 |
| 035 | 1.5345e-05 | 2.4859e-05 | -8.6 | 0.9972 |
| 036 | 1.5384e-05 | 2.4923e-05 | -8.6 | 0.9972 |
| 037 | 1.5329e-05 | 2.4834e-05 | -8.6 | 0.9972 |
| 038 | 1.5333e-05 | 2.4840e-05 | -8.6 | 0.9972 |
| 039 | 1.5337e-05 | 2.4846e-05 | -8.6 | 0.9972 |
| 040 | 1.5358e-05 | 2.4880e-05 | -8.6 | 0.9964 |
| 041 | 1.5343e-05 | 2.4856e-05 | -8.6 | 0.9972 |
| 042 | 1.5366e-05 | 2.4893e-05 | -8.6 | 0.9950 |
| 043 | 1.5418e-05 | 2.4978e-05 | -8.6 | 0.9961 |
| 044 | 1.5434e-05 | 2.5003e-05 | -8.6 | 0.9964 |
| 045 | 1.5375e-05 | 2.4908e-05 | -8.6 | 0.9972 |
| 046 | 1.5392e-05 | 2.4935e-05 | -8.6 | 0.9972 |
| 047 | 1.5405e-05 | 2.4956e-05 | -8.6 | 0.9972 |
| 048 | 1.5414e-05 | 2.4971e-05 | -8.6 | 0.9972 |
| 049 | 1.5422e-05 | 2.4984e-05 | -8.6 | 0.9972 |
| 050 | 1.5429e-05 | 2.4995e-05 | -8.6 | 0.9972 |
| 051 | 1.5434e-05 | 2.5004e-05 | -8.6 | 0.9972 |
| 052 | 1.5439e-05 | 2.5012e-05 | -8.6 | 0.9972 |
| 053 | 1.5444e-05 | 2.5019e-05 | -8.6 | 0.9972 |
| 054 | 1.5448e-05 | 2.5026e-05 | -8.6 | 0.9972 |
| 055 | 1.5452e-05 | 2.5032e-05 | -8.6 | 0.9972 |
| 056 | 1.5455e-05 | 2.5037e-05 | -8.6 | 0.9972 |
| 057 | 1.5458e-05 | 2.5042e-05 | -8.6 | 0.9972 |
| 058 | 1.5461e-05 | 2.5046e-05 | -8.6 | 0.9972 |
| 059 | 1.5461e-05 | 2.5047e-05 | -8.6 | 0.9972 |
| 060 | 1.5459e-05 | 2.5044e-05 | -8.6 | 0.9972 |
| 061 | 1.5457e-05 | 2.5041e-05 | -8.6 | 0.9972 |
| 062 | 1.5455e-05 | 2.5038e-05 | -8.6 | 0.9972 |
| 063 | 1.5454e-05 | 2.5035e-05 | -8.6 | 0.9972 |
| 064 | 1.5452e-05 | 2.5033e-05 | -8.6 | 0.9972 |
| 065 | 1.5451e-05 | 2.5031e-05 | -8.6 | 0.9972 |
| 066 | 1.5450e-05 | 2.5029e-05 | -8.6 | 0.9972 |
| 067 | 1.5449e-05 | 2.5027e-05 | -8.6 | 0.9972 |
| 068 | 1.5448e-05 | 2.5026e-05 | -8.6 | 0.9972 |
| 069 | 1.5447e-05 | 2.5024e-05 | -8.6 | 0.9972 |
| 070 | 1.5446e-05 | 2.5023e-05 | -8.6 | 0.9972 |
| 071 | 1.5446e-05 | 2.5022e-05 | -8.6 | 0.9972 |
| 072 | 1.5445e-05 | 2.5021e-05 | -8.6 | 0.9972 |
| 073 | 1.5445e-05 | 2.5020e-05 | -8.6 | 0.9972 |
| 074 | 1.5444e-05 | 2.5019e-05 | -8.6 | 0.9972 |
| 075 | 1.5444e-05 | 2.5019e-05 | -8.6 | 0.9972 |
| 076 | 1.5443e-05 | 2.5018e-05 | -8.6 | 0.9972 |
| 077 | 1.5443e-05 | 2.5018e-05 | -8.6 | 0.9972 |
| 078 | 1.5443e-05 | 2.5017e-05 | -8.6 | 0.9972 |
| 079 | 1.5442e-05 | 2.5017e-05 | -8.6 | 0.9972 |
| 080 | 1.5442e-05 | 2.5016e-05 | -8.6 | 0.9972 |

## 3. Leak-mechanism classification (node 7)

- `r_bar`: **23,040 barrier-set** (every cycle period), 288 not barrier-set — exactly
  the initialization round. Predeclared `r`: 13,721 barrier-set, 9,319 indeterminate,
  288 not barrier-set.
- **Initialization round: SoH floor binding, request partly absorbed by penalized slack** (as in G2's
  initialization). In all 288 periods the largest dual is on `energy_storage_capacity_degradation`
  (≈ −2.46e5, −1.96e5); in 38 periods both legs are < 1e-6; at (y0, d0, p0) the request is `pnet` = −1.244
  while both legs sit at their relaxed bound and the aggregate-row dual is 999.99 = `PENALTY_ESSO_SLACK`.
  Elsewhere the ESSO dispatches (max leg 1.62 = `s_max`). Inferred from duals; slack values not captured.

## 4. Convergence, recourse, SoH

- Converged at cycle **80**; recourse **819,016,107.91**.
- Rule ten: terminal step 70,986.80 / tolerance 81,908.71 =
  **0.867**.
- Recourse is not comparable with G1 or G2: the instance differs (investment at node 7 only).
- Solves: 4,148; wall clock 2,579 s.
- Node 7 SoH: 0.8464 → 0.7048 → 0.5963; EFC/day: 1.0545 / 1.1577 / 1.0572 (max below the 1.4612 threshold).

## 5. Evidence guard

`shared_frozen_smopf_modified = []`, `shared_frozen_smopf_new_files = []`. All three snapshots (the two
cycle-7 comparators and the cycle-38 failure block) were written under the arm root.

## 6. What is NOT established

- Whether the 2035 Autumn `case33_2` block's failure is intrinsic to this plan or path-dependent — the saved
  pre-solve block allows this to be tested without re-running the campaign.
- The slack values at initialization (inferred from duals).
- The cause of the empty parser rows (3 here, 2 in G2, 0 in G1's validation).

## 7. Evidence (sha256, first 16; full manifest `P515G3F_r2/evidence_manifest_sha256.json`)

| artifact | hash |
|---|---|
| `P515G3F_r2/g_g3_full_node7.json` | `a1238d41d42b3a20` |
| `P515G3F_r2/stdout_g3_full_node7.log` | `cfba4f9974b41ecd` |
| `P515G3F_r2/network_failures_g3_full_node7.jsonl` | `efb2ee33b50b6587` |
| `P515G3F_r2/leak_classification_g3_full_node7.jsonl` | `8ab58d3e21d949b3` |
| `P515G3F_r2/esso_models_g3_full_node7.pkl` | `446c424dc5c85a3c` |
| `P515G3F_r2/results/FrozenSMOPF/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl` | `4499cf99a4341eb9` |
| `P515G3F_r2_launch.log` | `3447f9599cb728f2` |
| `P515G3F_r2/esso_capture/` (243 files, 29.4 MB, hash-recorded, not committed) | see manifest |
