# P5.15 — Gate G2 (perturbation arm, k = 10,000): FAILS — no convergence in 90 cycles

**Planner report, 2026-09-14.** Authority: `PLANNER_BRIEF_2026-09-13.md` Step 1 gate G2 (Addendum 1),
Addendum-6 capture. Run: `7a2d85db`, one background run through the tool, stderr captured, lock held by
the run. Command: `python -u p515_g_g1_g4_admm_gates.py g2 > data/SRP1/Results/P515G2_launch.log 2>&1`.
Instance: C\* (0.96875 MVA / 3.875 MWh per active node, 2025), `k = 10,000` in all years. Objective
convention: recourse = `gross_operational_cost`.

**Required (G2):** converges within the cap; zero `maxIterations` exits at any node; recourse and pair
difference against G1 with the rule-nine bar. **Result: neither requirement met.**

---

## 1. Network failures — classification (print-based parser, validated on G1)

| class | count |
|---|---|
| recovered | **18** (TSO 3, DSO 15) |
| unrecovered | **0** |
| not attempted | **33** — all DSO node 5 `case33_1`, termination `maxIterations` |
| unparsed rows | 2 (empty rows; parser residue, unexplained — did not occur on G1's validation) |

Counts reconcile with the raw stdout: 84 "did not converge" prints = 33 not-attempted × 2 prints + 18
recovered × 1.

**TSO failures — all recovered on the single cold retry:**

| agent | block | cycle | class |
|---|---|---|---|
| TSO `case9` | 2030 Summer | 4 | recovered |
| TSO `case9` | 2030 Autumn | 22 | recovered |
| TSO `case9` | 2025 Summer | 40 | recovered |

DSO recovered: `case33_2` 2025 Autumn (2), `case33_2` 2035 Summer (5), `case33_3` 2025 Summer (5), `case33_3` 2035 Spring (5), `case33_2` 2035 Winter (9), `case33_3` 2035 Spring (26), `case33_3` 2025 Autumn (27), `case33_3` 2035 Winter (29), `case33_2` 2025 Autumn (33), `case33_2` 2025 Winter (44), `case33_3` 2025 Spring (45), `case33_2` 2025 Spring (56), `case33_3` 2035 Spring (68), `case33_2` 2030 Spring (75), `case33_2` 2030 Spring (87).

**Not attempted (ineligible — `case33_1` has no `recovery_options`):**

| block | failures | cycles |
|---|---|---|
| 2025 Autumn | 1 | 2 |
| 2035 Winter | 1 | 2 |
| 2035 Summer | 3 | 11, 12, 13 |
| 2025 Spring | 1 | 17 |
| 2035 Spring | 2 | 22, 86 |
| 2035 Autumn | 25 | 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90 |

## 2. Why G2 did not converge

Cycles with a failed local solve: 2, 11, 12, 13, 17, 22, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90 (31 cycles). From cycle **66
to 90** the same block — node 5 `case33_1` 2035 Autumn — failed with `maxIterations` **every cycle** and
was **never retried**, because eligibility for recovery requires a non-empty `recovery_options` and
`case33_1` has none (G1 report §1). Each such cycle produces no recourse and holds all penalties
("held after solver failure"), so the run cannot satisfy its convergence test and stops at the cap.
Whether this block would recover on a cold retry is **not known**: it was never attempted.

## 3. Per-cycle detector trajectory

- ESSO solves: 273 (91 rounds × 3 nodes). Terminal `lg(mu)`: {-8.6: 273}.
- Cycles 1–90: detector **2.5340e-05 – 2.5855e-05**, no drift; detector /
  `μ_barrier/(s_obj·ε)` = **1.0088 – 1.0293**.
- Initialization round: detector **0.0** on all three nodes — see §4.

| round | detector min | detector max | `lg(mu)` | argmax `r_bar` |
|---|---|---|---|---|
| init | 0.0000e+00 | 0.0000e+00 | -8.6 | — |
| 1 | 2.5853e-05 | 2.5854e-05 | -8.6 | 0.997–0.997 |
| 2 | 2.5819e-05 | 2.5841e-05 | -8.6 | 0.997–0.997 |
| 3 | 2.5828e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 4 | 2.5519e-05 | 2.5826e-05 | -8.6 | 0.992–0.997 |
| 5 | 2.5340e-05 | 2.5800e-05 | -8.6 | 0.989–0.996 |
| 6 | 2.5468e-05 | 2.5827e-05 | -8.6 | 0.992–0.997 |
| 7 | 2.5792e-05 | 2.5849e-05 | -8.6 | 0.997–0.997 |
| 8 | 2.5524e-05 | 2.5814e-05 | -8.6 | 0.997–0.997 |
| 9 | 2.5543e-05 | 2.5736e-05 | -8.6 | 0.992–0.996 |
| 10 | 2.5687e-05 | 2.5797e-05 | -8.6 | 0.996–0.997 |
| 11 | 2.5657e-05 | 2.5816e-05 | -8.6 | 0.993–0.996 |
| 12 | 2.5707e-05 | 2.5844e-05 | -8.6 | 0.995–0.997 |
| 13 | 2.5745e-05 | 2.5852e-05 | -8.6 | 0.996–0.997 |
| 14 | 2.5771e-05 | 2.5844e-05 | -8.6 | 0.996–0.997 |
| 15 | 2.5791e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 16 | 2.5788e-05 | 2.5851e-05 | -8.6 | 0.997–0.997 |
| 17 | 2.5778e-05 | 2.5851e-05 | -8.6 | 0.996–0.998 |
| 18 | 2.5767e-05 | 2.5841e-05 | -8.6 | 0.996–0.998 |
| 19 | 2.5778e-05 | 2.5844e-05 | -8.6 | 0.996–0.998 |
| 20 | 2.5758e-05 | 2.5827e-05 | -8.6 | 0.996–0.997 |
| 21 | 2.5687e-05 | 2.5840e-05 | -8.6 | 0.997–0.997 |
| 22 | 2.5539e-05 | 2.5723e-05 | -8.6 | 0.997–0.997 |
| 23 | 2.5675e-05 | 2.5752e-05 | -8.6 | 0.997–0.997 |
| 24 | 2.5707e-05 | 2.5756e-05 | -8.6 | 0.997–0.997 |
| 25 | 2.5750e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 26 | 2.5813e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 27 | 2.5799e-05 | 2.5825e-05 | -8.6 | 0.997–0.997 |
| 28 | 2.5775e-05 | 2.5821e-05 | -8.6 | 0.997–0.997 |
| 29 | 2.5719e-05 | 2.5744e-05 | -8.6 | 0.997–0.997 |
| 30 | 2.5714e-05 | 2.5805e-05 | -8.6 | 0.997–0.997 |
| 31 | 2.5685e-05 | 2.5717e-05 | -8.6 | 0.997–0.997 |
| 32 | 2.5678e-05 | 2.5705e-05 | -8.6 | 0.997–0.997 |
| 33 | 2.5674e-05 | 2.5703e-05 | -8.6 | 0.997–0.997 |
| 34 | 2.5676e-05 | 2.5842e-05 | -8.6 | 0.997–0.997 |
| 35 | 2.5780e-05 | 2.5834e-05 | -8.6 | 0.997–0.997 |
| 36 | 2.5710e-05 | 2.5789e-05 | -8.6 | 0.997–0.997 |
| 37 | 2.5685e-05 | 2.5714e-05 | -8.6 | 0.997–0.997 |
| 38 | 2.5688e-05 | 2.5817e-05 | -8.6 | 0.997–0.997 |
| 39 | 2.5728e-05 | 2.5842e-05 | -8.6 | 0.997–0.997 |
| 40 | 2.5696e-05 | 2.5721e-05 | -8.6 | 0.997–0.997 |
| 41 | 2.5786e-05 | 2.5818e-05 | -8.6 | 0.997–0.997 |
| 42 | 2.5717e-05 | 2.5745e-05 | -8.6 | 0.997–0.997 |
| 43 | 2.5831e-05 | 2.5853e-05 | -8.6 | 0.997–0.997 |
| 44 | 2.5760e-05 | 2.5795e-05 | -8.6 | 0.997–0.997 |
| 45 | 2.5708e-05 | 2.5732e-05 | -8.6 | 0.997–0.997 |
| 46 | 2.5706e-05 | 2.5752e-05 | -8.6 | 0.995–0.997 |
| 47 | 2.5701e-05 | 2.5726e-05 | -8.6 | 0.997–0.997 |
| 48 | 2.5695e-05 | 2.5719e-05 | -8.6 | 0.997–0.997 |
| 49 | 2.5690e-05 | 2.5730e-05 | -8.6 | 0.997–0.997 |
| 50 | 2.5687e-05 | 2.5782e-05 | -8.6 | 0.997–0.997 |
| 51 | 2.5704e-05 | 2.5827e-05 | -8.6 | 0.997–0.997 |
| 52 | 2.5746e-05 | 2.5850e-05 | -8.6 | 0.997–0.997 |
| 53 | 2.5781e-05 | 2.5822e-05 | -8.6 | 0.997–0.997 |
| 54 | 2.5806e-05 | 2.5834e-05 | -8.6 | 0.997–0.997 |
| 55 | 2.5801e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 56 | 2.5802e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 57 | 2.5808e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 |
| 58 | 2.5820e-05 | 2.5825e-05 | -8.6 | 0.997–0.997 |
| 59 | 2.5810e-05 | 2.5835e-05 | -8.6 | 0.997–0.997 |
| 60 | 2.5794e-05 | 2.5849e-05 | -8.6 | 0.997–0.997 |
| 61 | 2.5777e-05 | 2.5847e-05 | -8.6 | 0.997–0.997 |
| 62 | 2.5763e-05 | 2.5855e-05 | -8.6 | 0.997–0.997 |
| 63 | 2.5757e-05 | 2.5826e-05 | -8.6 | 0.997–0.997 |
| 64 | 2.5744e-05 | 2.5850e-05 | -8.6 | 0.997–0.997 |
| 65 | 2.5740e-05 | 2.5801e-05 | -8.6 | 0.997–0.997 |
| 66 | 2.5770e-05 | 2.5793e-05 | -8.6 | 0.997–0.997 |
| 67 | 2.5766e-05 | 2.5852e-05 | -8.6 | 0.997–0.997 |
| 68 | 2.5771e-05 | 2.5807e-05 | -8.6 | 0.997–0.997 |
| 69 | 2.5777e-05 | 2.5805e-05 | -8.6 | 0.996–0.997 |
| 70 | 2.5769e-05 | 2.5790e-05 | -8.6 | 0.996–0.997 |
| 71 | 2.5735e-05 | 2.5796e-05 | -8.6 | 0.997–0.997 |
| 72 | 2.5740e-05 | 2.5803e-05 | -8.6 | 0.997–0.997 |
| 73 | 2.5747e-05 | 2.5810e-05 | -8.6 | 0.997–0.997 |
| 74 | 2.5753e-05 | 2.5817e-05 | -8.6 | 0.997–0.997 |
| 75 | 2.5757e-05 | 2.5823e-05 | -8.6 | 0.997–0.997 |
| 76 | 2.5758e-05 | 2.5827e-05 | -8.6 | 0.997–0.997 |
| 77 | 2.5759e-05 | 2.5831e-05 | -8.6 | 0.997–0.997 |
| 78 | 2.5759e-05 | 2.5834e-05 | -8.6 | 0.997–0.997 |
| 79 | 2.5760e-05 | 2.5835e-05 | -8.6 | 0.997–0.997 |
| 80 | 2.5762e-05 | 2.5836e-05 | -8.6 | 0.997–0.997 |
| 81 | 2.5762e-05 | 2.5835e-05 | -8.6 | 0.997–0.997 |
| 82 | 2.5761e-05 | 2.5833e-05 | -8.6 | 0.997–0.997 |
| 83 | 2.5778e-05 | 2.5833e-05 | -8.6 | 0.996–0.997 |
| 84 | 2.5761e-05 | 2.5830e-05 | -8.6 | 0.997–0.997 |
| 85 | 2.5804e-05 | 2.5825e-05 | -8.6 | 0.996–0.997 |
| 86 | 2.5761e-05 | 2.5822e-05 | -8.6 | 0.997–0.997 |
| 87 | 2.5761e-05 | 2.5819e-05 | -8.6 | 0.997–0.997 |
| 88 | 2.5761e-05 | 2.5818e-05 | -8.6 | 0.997–0.997 |
| 89 | 2.5761e-05 | 2.5817e-05 | -8.6 | 0.997–0.997 |
| 90 | 2.5761e-05 | 2.5817e-05 | -8.6 | 0.997–0.997 |

## 4. Leak-mechanism classification

- `r_bar` (terminal barrier parameter): **77,760 barrier-set**, 864 not
  barrier-set — the latter are **exactly the 864 initialization-round periods**
  (3 nodes × 288). Predeclared `r`: 49,801 barrier-set, 27,959
  indeterminate, 864 not barrier-set.
- **Cycles 1–90: barrier-set, as in G1.**
- **Initialization round: not an x = 0 artifact — a slack-dominated ESSO state.** At node 7, period
  (y0, d0, p0): the request is full discharge (`pnet` = −0.96867) yet both legs sit at their relaxed bound
  (`pch` = −9.99e-9, `pdch` = −7.72e-9), with `zL_pch` ≈ 1.94e3. The dual of the aggregate row
  `es_pnet = Σ(pch − pdch) + slack_up − slack_down` is **999.99 = `PENALTY_ESSO_SLACK`**, i.e. the whole
  request is absorbed by the penalized slack. The largest duals are on
  `energy_storage_capacity_degradation` (−2.55e5, −2.06e5) in every period: **the SoH floor binds**.
  At `k = 10,000` the initialization request would drive SoH below `soh_min`, so the ESSO refuses to
  dispatch and pays the slack penalty. G1's initialization (`k = 11,541.56`) at the same period dispatched
  normally (`pdch` = 0.9687, multipliers ~1e-3). **G2's ADMM therefore started from an ESSO state in which
  the storage did nothing and the request was pure penalty.** Inferred from the duals and the `pnet`/leg
  mismatch; slack values themselves were not captured per round.

## 5. Recourse, pair difference, rule ten

- Final recourse: **none** — cycle 90 had a failed local solve.
- Last valid recourse: cycle 65, **818,690,717.51** (G2 − G1 = **+1,071,919.44**).
  Its step was 131,095.44 against a tolerance of 81,882.18 — **1.60× its
  threshold**: the run had **not settled**, so the rule-nine bar does not apply and the pair difference is
  **indeterminate**. This leaves the P5.14-N objective comparison still unanswered.
- Solves: 4,659.

## 6. SoH and EFC/day per node (G1 → G2)

| node | G1 SoH (yr 1→3) | G2 SoH (yr 1→3) | G1 EFC/day (yr 1/2/3) | G2 EFC/day |
|---|---|---|---|---|
| 5 | 0.8575 → 0.7466 → 0.6413 | 0.8428 → 0.7184 → 0.6081 | 0.9725 / 0.8759 / 0.9613 | 0.9372 / 0.8748 / 0.9139 |
| 7 | 0.8575 → 0.7472 → 0.6412 | 0.8412 → 0.7177 → 0.6069 | 0.9722 / 0.8711 / 0.9674 | 0.9478 / 0.8700 / 0.9183 |
| 9 | 0.8575 → 0.7453 → 0.6397 | 0.8411 → 0.7155 → 0.6053 | 0.9724 / 0.8868 / 0.9661 | 0.9485 / 0.8858 / 0.9166 |

G2 ages faster, as a lower `k` implies; these are **unconverged** end-of-cap values.

## 7. Evidence guard

`shared_frozen_smopf_modified = []`, `shared_frozen_smopf_new_files = []`: the shared `FrozenSMOPF`
directory was not touched. The two cycle-7 comparators were written to the arm's own
`P515G2/results/FrozenSMOPF/`. The `results_dir` redirection works.

## 8. G3-full: first attempt stopped by a harness defect

G3-full (node 7 at 1.62 MVA / 3.24 MWh, others zero) stopped at the initialization solve with the
harness pre-check "`model.ipopt_zL_out` has no entries for `es_pch_per_unit` … node 5". Nodes 5 and 9
have no investment in G3-full, so no active cohort-periods (production's diagnostic prints
`n_periods=0`); their legs are fixed and IPOPT returns no multipliers. The stop is false — the
pre-check (specified by the Planner) should apply only to nodes with active cohort-periods. It is being
corrected and G3-full re-run under fresh names; the partial `P515G3F/` is preserved as a failed attempt.

## 9. What is NOT established

- Whether node 5 `case33_1` 2035 Autumn would recover if eligible, and so whether G2 would converge.
- The G2 − G1 recourse difference at convergence.
- The slack values at G2 initialization (inferred from duals, not captured).
- The cause of the two empty parser rows.

## 10. Evidence (sha256, first 16; full manifest `P515G2/evidence_manifest_sha256.json`)

| artifact | hash |
|---|---|
| `P515G2/g_k10000.json` | `149cee00d7d26f70` |
| `P515G2/stdout_k10000.log` | `78aa3192d6800513` |
| `P515G2/network_failures_k10000.jsonl` | `71963cb38285da01` |
| `P515G2/leak_classification_k10000.jsonl` | `916222f08bb8e7ad` |
| `P515G2/esso_models_k10000.pkl` | `004b7d80b28b3b45` |
| `P515G2/heartbeat_k10000.json` | `518db1859d01dffb` |
| `P515G2_launch.log` | `3c2d9b7b839368f8` |
| `P515G2/esso_capture/` (273 files, 99.2 MB, hash-recorded, not committed) | see manifest |
