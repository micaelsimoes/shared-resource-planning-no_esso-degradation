# P5.15 — Gate G1 (control arm, C\*): converged; reconciliation FAILS; leak mechanism settled

**Planner report, 2026-09-14.** Authority: `PLANNER_BRIEF_2026-09-13.md` Step 1 gate G1 as
re-specified by Addendum 3 item 3, with the Addendum-6 capture. Run: `4b56c835`, one background
run through the tool, stderr captured, lock held by the run, exit code 0.
Command: `python -u p515_g_g1_g4_admm_gates.py g1 > data/SRP1/Results/P515G1_launch.log 2>&1`.
Objective convention: recourse = `gross_operational_cost` (equal here; no salvage term reported).

---

## 1. Network failures at C\* — the classification the previous attempt lost

Source: production's own prints in `stdout_control.log` (the harness's `network_failures_control.jsonl`
is **wrong** — see §6 — and is superseded). Cycle attribution: a block printed after the
`ADMM cycle N |` summary line belongs to cycle N+1 (the summary is printed at the end of a cycle);
this rule places the six not-attempted failures exactly on the six cycles the ADMM trajectory marks
`local_solves_ok = False` (7, 39, 43, 46, 48, 49), an independent cross-check. A zero-solve
re-implementation of this parser is being validated by a Worker against these counts.

| class | count | blocks |
|---|---|---|
| **recovered** (primary not converged → cold retry → succeeded) | **12** | TSO `case9`: 2035 Winter (initialization), 2025 Autumn (cycle 5), 2030 Winter (cycle 40). DSO: `case33_2` 2025 Spring (7), 2025 Summer (33), 2035 Winter (45), 2035 Spring (59), 2035 Autumn (62); `case33_3` 2025 Summer (19), 2025 Autumn (27), 2035 Autumn (30), 2025 Winter (51) |
| **unrecovered** (retry failed) | **0** | — |
| **not attempted** (not eligible for recovery) | **6** | all DSO node 5, `case33_1`, termination `maxIterations`: 2035 Winter (7), 2035 Autumn (39), 2025 Spring (43), 2035 Autumn (46), 2035 Spring (48), 2035 Spring (49) |

**Conclusion on the lost TSO failure.** Every TSO local failure in G1 was a `maxIterations`-class
primary failure that **recovered** on the single cold retry (3 of 3). No TSO block ended unsolved.

**Why the six node-5 failures were never retried.** `network.py:_is_recoverable_network_failure`
returns False when the case's `recovery_options` is empty. `case33_1_params.json` has **no**
`recovery_options`; `case33_2`/`case33_3` hold only `{"hessian_approximation": "limited-memory"}` and
`case9` that plus `acceptable_tol`/`acceptable_iter`. The retry itself **drops**
`hessian_approximation` (Step 1a). So recovery eligibility is decided by whether a dead entry exists:
- node 5's failures were ineligible by accident of configuration, not by policy;
- **hazard:** the author's planned removal of the "dead" `limited-memory` entries (Addendum 2) would
  make `recovery_options` empty for `case33_2` and `case33_3` and silently **disable recovery** there.

Solve accounting: 3,735 solves = 51 × 72 + 51 + **12** — the 12 extra solves are exactly the 12
recoveries; the harness's `identity_holds: false` is that, not a missing solve. ESSO recovery events: 0.

## 2. Per-cycle detector trajectory at C\*

Measured detector `max min(pch,pdch)/s_max` per ESSO solve, three nodes per round, 73 rounds
(initialization + 72 cycles), `ε = 1e-3`, remedy (h) in force. Only the measured detector is quoted
(Addendum 6).

- Range over all 219 solves: **2.5037e-05 – 2.5856e-05**; no drift across cycles.
- Terminal barrier parameter `lg(mu)` = **−8.6 in all 219 solves** — the dual-magnitude drift derived
  in round 1 (`s_obj` falling as duals grow) **did not occur**: `s_obj = 0.1` throughout.

| round | detector min | detector max | `lg(mu)` | argmax `r_bar` | detector / `μ/(s_obj·ε)` |
|---|---|---|---|---|---|
| init | 2.5656e-05 | 2.5838e-05 | -8.6 | 0.995–0.997 | 1.0214–1.0286 |
| 1 | 2.5853e-05 | 2.5854e-05 | -8.6 | 0.997–0.997 | 1.0292–1.0293 |
| 2 | 2.5839e-05 | 2.5856e-05 | -8.6 | 0.997–0.997 | 1.0287–1.0293 |
| 3 | 2.5824e-05 | 2.5848e-05 | -8.6 | 0.997–0.997 | 1.0281–1.0290 |
| 4 | 2.5487e-05 | 2.5834e-05 | -8.6 | 0.991–0.997 | 1.0147–1.0285 |
| 5 | 2.5037e-05 | 2.5336e-05 | -8.6 | 0.985–0.990 | 0.9967–1.0086 |
| 6 | 2.5360e-05 | 2.5757e-05 | -8.6 | 0.990–0.996 | 1.0096–1.0254 |
| 7 | 2.5772e-05 | 2.5809e-05 | -8.6 | 0.997–0.997 | 1.0260–1.0275 |
| 8 | 2.5532e-05 | 2.5574e-05 | -8.6 | 0.997–0.997 | 1.0165–1.0181 |
| 9 | 2.5423e-05 | 2.5537e-05 | -8.6 | 0.990–0.997 | 1.0121–1.0166 |
| 10 | 2.5377e-05 | 2.5531e-05 | -8.6 | 0.991–0.997 | 1.0103–1.0164 |
| 11 | 2.5465e-05 | 2.5541e-05 | -8.6 | 0.992–0.997 | 1.0138–1.0168 |
| 12 | 2.5452e-05 | 2.5573e-05 | -8.6 | 0.990–0.992 | 1.0133–1.0181 |
| 13 | 2.5601e-05 | 2.5669e-05 | -8.6 | 0.997–0.997 | 1.0192–1.0219 |
| 14 | 2.5374e-05 | 2.5578e-05 | -8.6 | 0.989–0.992 | 1.0101–1.0183 |
| 15 | 2.5455e-05 | 2.5502e-05 | -8.6 | 0.997–0.997 | 1.0134–1.0152 |
| 16 | 2.5639e-05 | 2.5691e-05 | -8.6 | 0.997–0.997 | 1.0207–1.0228 |
| 17 | 2.5629e-05 | 2.5658e-05 | -8.6 | 0.997–0.997 | 1.0203–1.0215 |
| 18 | 2.5770e-05 | 2.5827e-05 | -8.6 | 0.997–0.997 | 1.0259–1.0282 |
| 19 | 2.5727e-05 | 2.5811e-05 | -8.6 | 0.996–0.997 | 1.0242–1.0276 |
| 20 | 2.5413e-05 | 2.5499e-05 | -8.6 | 0.997–0.997 | 1.0117–1.0151 |
| 21 | 2.5796e-05 | 2.5816e-05 | -8.6 | 0.997–0.997 | 1.0270–1.0278 |
| 22 | 2.5827e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 | 1.0282–1.0288 |
| 23 | 2.5767e-05 | 2.5789e-05 | -8.6 | 0.997–0.997 | 1.0258–1.0267 |
| 24 | 2.5661e-05 | 2.5726e-05 | -8.6 | 0.997–0.997 | 1.0216–1.0242 |
| 25 | 2.5762e-05 | 2.5814e-05 | -8.6 | 0.997–0.997 | 1.0256–1.0277 |
| 26 | 2.5820e-05 | 2.5855e-05 | -8.6 | 0.997–0.997 | 1.0279–1.0293 |
| 27 | 2.5763e-05 | 2.5810e-05 | -8.6 | 0.997–0.997 | 1.0257–1.0275 |
| 28 | 2.5813e-05 | 2.5853e-05 | -8.6 | 0.997–0.997 | 1.0276–1.0292 |
| 29 | 2.5701e-05 | 2.5744e-05 | -8.6 | 0.997–0.997 | 1.0232–1.0249 |
| 30 | 2.5687e-05 | 2.5728e-05 | -8.6 | 0.997–0.997 | 1.0226–1.0243 |
| 31 | 2.5679e-05 | 2.5718e-05 | -8.6 | 0.997–0.997 | 1.0223–1.0239 |
| 32 | 2.5676e-05 | 2.5715e-05 | -8.6 | 0.997–0.997 | 1.0222–1.0237 |
| 33 | 2.5726e-05 | 2.5791e-05 | -8.6 | 0.997–0.997 | 1.0242–1.0268 |
| 34 | 2.5827e-05 | 2.5848e-05 | -8.6 | 0.997–0.997 | 1.0282–1.0290 |
| 35 | 2.5746e-05 | 2.5809e-05 | -8.6 | 0.997–0.997 | 1.0250–1.0275 |
| 36 | 2.5685e-05 | 2.5721e-05 | -8.6 | 0.997–0.997 | 1.0226–1.0240 |
| 37 | 2.5693e-05 | 2.5743e-05 | -8.6 | 0.997–0.997 | 1.0228–1.0248 |
| 38 | 2.5802e-05 | 2.5841e-05 | -8.6 | 0.997–0.997 | 1.0272–1.0287 |
| 39 | 2.5693e-05 | 2.5732e-05 | -8.6 | 0.997–0.997 | 1.0228–1.0244 |
| 40 | 2.5693e-05 | 2.5725e-05 | -8.6 | 0.997–0.997 | 1.0229–1.0241 |
| 41 | 2.5697e-05 | 2.5725e-05 | -8.6 | 0.997–0.997 | 1.0230–1.0241 |
| 42 | 2.5813e-05 | 2.5823e-05 | -8.6 | 0.997–0.997 | 1.0276–1.0280 |
| 43 | 2.5793e-05 | 2.5803e-05 | -8.6 | 0.997–0.997 | 1.0268–1.0272 |
| 44 | 2.5724e-05 | 2.5779e-05 | -8.6 | 0.996–0.997 | 1.0241–1.0263 |
| 45 | 2.5693e-05 | 2.5722e-05 | -8.6 | 0.997–0.997 | 1.0229–1.0240 |
| 46 | 2.5693e-05 | 2.5721e-05 | -8.6 | 0.997–0.997 | 1.0229–1.0240 |
| 47 | 2.5693e-05 | 2.5721e-05 | -8.6 | 0.997–0.997 | 1.0229–1.0240 |
| 48 | 2.5693e-05 | 2.5720e-05 | -8.6 | 0.997–0.997 | 1.0228–1.0239 |
| 49 | 2.5691e-05 | 2.5805e-05 | -8.6 | 0.996–0.997 | 1.0228–1.0273 |
| 50 | 2.5691e-05 | 2.5798e-05 | -8.6 | 0.997–0.997 | 1.0228–1.0271 |
| 51 | 2.5734e-05 | 2.5839e-05 | -8.6 | 0.997–0.997 | 1.0245–1.0287 |
| 52 | 2.5771e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 | 1.0260–1.0288 |
| 53 | 2.5800e-05 | 2.5822e-05 | -8.6 | 0.997–0.997 | 1.0271–1.0280 |
| 54 | 2.5805e-05 | 2.5835e-05 | -8.6 | 0.997–0.997 | 1.0273–1.0285 |
| 55 | 2.5804e-05 | 2.5839e-05 | -8.6 | 0.997–0.997 | 1.0273–1.0287 |
| 56 | 2.5805e-05 | 2.5836e-05 | -8.6 | 0.997–0.997 | 1.0273–1.0286 |
| 57 | 2.5815e-05 | 2.5828e-05 | -8.6 | 0.997–0.997 | 1.0277–1.0282 |
| 58 | 2.5816e-05 | 2.5827e-05 | -8.6 | 0.997–0.997 | 1.0277–1.0282 |
| 59 | 2.5805e-05 | 2.5852e-05 | -8.6 | 0.997–0.997 | 1.0273–1.0292 |
| 60 | 2.5785e-05 | 2.5855e-05 | -8.6 | 0.997–0.997 | 1.0265–1.0293 |
| 61 | 2.5769e-05 | 2.5843e-05 | -8.6 | 0.997–0.997 | 1.0259–1.0288 |
| 62 | 2.5754e-05 | 2.5824e-05 | -8.6 | 0.997–0.997 | 1.0253–1.0281 |
| 63 | 2.5741e-05 | 2.5812e-05 | -8.6 | 0.997–0.997 | 1.0248–1.0276 |
| 64 | 2.5785e-05 | 2.5814e-05 | -8.6 | 0.997–0.997 | 1.0265–1.0277 |
| 65 | 2.5839e-05 | 2.5856e-05 | -8.6 | 0.997–0.997 | 1.0287–1.0293 |
| 66 | 2.5787e-05 | 2.5814e-05 | -8.6 | 0.997–0.997 | 1.0266–1.0277 |
| 67 | 2.5751e-05 | 2.5790e-05 | -8.6 | 0.997–0.997 | 1.0252–1.0267 |
| 68 | 2.5734e-05 | 2.5791e-05 | -8.6 | 0.997–0.997 | 1.0245–1.0268 |
| 69 | 2.5733e-05 | 2.5795e-05 | -8.6 | 0.997–0.997 | 1.0244–1.0269 |
| 70 | 2.5738e-05 | 2.5801e-05 | -8.6 | 0.997–0.997 | 1.0247–1.0271 |
| 71 | 2.5745e-05 | 2.5806e-05 | -8.6 | 0.997–0.997 | 1.0249–1.0274 |
| 72 | 2.5750e-05 | 2.5836e-05 | -8.6 | 0.997–0.997 | 1.0251–1.0285 |

## 3. Leak-mechanism classification — barrier-set

Per period and cohort, the small leg `x_small = min(pch,pdch)` and its bound multiplier `zL`:

- `r_bar = zL·x_small / (μ_barrier/s_obj)`, with `μ_barrier = 10^lg(mu)` (terminal barrier parameter):
  **all 63,072 cohort-periods barrier-set** (`0.5 ≤ r_bar ≤ 2`).
- Predeclared `r` (against the summary `Complementarity` line): **48,679 barrier-set,
  14,393 indeterminate, 0 not barrier-set.**
- Measured detector / `μ_barrier/(s_obj·ε)` = **0.997 – 1.029** in every solve.
- The argmax periods are idle (`pnet` ~1e-9 – 1e-6), both legs ≈2.5e-5; at idle `λ ≈ 0` so
  `zL ≈ s_obj·ε` on each leg and `x_small = μ_barrier/(s_obj·ε)` — no factor ½.

**Conclusion.** The C\* leak is set by IPOPT's terminal barrier parameter, which `tol` bounds.
Addendum 6's premise "not barrier-set (μ-insensitive)" and the Planner's round-4 Finding 1 both
arose from reading the summary `Complementarity` line — an optimality-error measure — as μ; they
are **withdrawn**. The 5× gap to the ε fixture is terminal μ (−8.6 vs −9.0: ×2.51) × leg symmetry
(idle, equal legs vs one large leg: ×2) × prefactor (0.997 vs 0.907: ×1.10) = 5.52 against 5.519.
Magnitude: spurious throughput ≈ 0.0094 % at C\*.

## 4. Gate G1 — the reconciliation gate FAILS

Re-specified criterion (Addendum 3 item 3): per node, measured Δ(SoH) between the old control
(`P514N/esso_models_control.pkl`) and this run must equal the Δ predicted from the two leak fractions
through `D ∝ throughput`, within 10 %; same for EFC/day; recourse within the rule-nine bar.

| node | yr | f_old | f_new | old SoH | new SoH | measured Δ | predicted Δ | measured/predicted | EFC/day old → new (Δ; leak-only prediction) |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 1.386 % | 0.0094 % | 0.838692 | 0.857460 | +0.018769 | +0.002034 | **9.23** | 1.1124 → 0.9725 (-0.1399; leak-only -0.0153) |
| 5 | 2 | 1.386 % | 0.0094 % | 0.728405 | 0.746560 | +0.018156 | +0.003186 | **5.70** | 0.8916 → 0.8759 (-0.0157; leak-only -0.0123) |
| 5 | 3 | 1.386 % | 0.0094 % | 0.624795 | 0.641281 | +0.016486 | +0.004060 | **4.06** | 0.9703 → 0.9613 (-0.0090; leak-only -0.0134) |
| 7 | 1 | 1.391 % | 0.0094 % | 0.838724 | 0.857500 | +0.018776 | +0.002041 | **9.20** | 1.1122 → 0.9722 (-0.1400; leak-only -0.0154) |
| 7 | 2 | 1.391 % | 0.0094 % | 0.726780 | 0.747160 | +0.020380 | +0.003212 | **6.35** | 0.9059 → 0.8711 (-0.0349; leak-only -0.0125) |
| 7 | 3 | 1.391 % | 0.0094 % | 0.623318 | 0.641175 | +0.017858 | +0.004084 | **4.37** | 0.9711 → 0.9674 (-0.0037; leak-only -0.0134) |
| 9 | 1 | 1.340 % | 0.0093 % | 0.838693 | 0.857482 | +0.018789 | +0.001966 | **9.56** | 1.1124 → 0.9724 (-0.1401; leak-only -0.0148) |
| 9 | 2 | 1.340 % | 0.0093 % | 0.726012 | 0.745291 | +0.019279 | +0.003101 | **6.22** | 0.9124 → 0.8868 (-0.0256; leak-only -0.0121) |
| 9 | 3 | 1.340 % | 0.0093 % | 0.622817 | 0.639712 | +0.016896 | +0.003937 | **4.29** | 0.9695 → 0.9661 (-0.0035; leak-only -0.0129) |

**Recourse:** new 817,618,798.07 vs old 816,121,464.16 → **+1,497,333.92**; rule-nine bar
(76,144.08 + 72,885.91) = 149,029.99; **difference / bar = 10.05** → not explained by
stopping slack.

**Rule ten:** terminal step / threshold = **0.891** (72,885.91 / 81,769.17); old control 0.933.
Settled, not stopped, but not deep inside its bound.

**Verdict: G1 FAILS as specified.** Measured ΔSoH is **4.1 – 9.6×** the leak-predicted Δ on every node
and year; year-1 EFC/day fell **−0.140** against a leak-only **−0.015**; recourse moved **10× the bar**.

**Reading.** The gate assumed the reformulated model reaches the same physical operating point as the
old one, differing only by the leak. It does not: the new run cycles the storage materially less in
the first period (EFC/day 1.112 → 0.972) and ages it less. Since the old control, the model changed in
more than the leak — reformulated ESSO (log-domain SoH, investments as parameters, deleted slack and
complementarity families), Step 1b network changes (Candidates 1, 2, 4, 5), the Step 1a solver and
recovery policy, remedy (h), H3 (inert). **This gate cannot attribute the difference among them.**
The old trajectory is a known-biased comparator (Addendum 3); the result says it is also a
different operating point, not a biased version of this one.

## 5. Evidence overwritten during G1 — and a fix before G2

Production wrote `matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` and
`matched_success_TSO_case9_2025_Summer_cycle7.pkl` into the **shared** `data/SRP1/Results/FrozenSMOPF/`
at 13:58–13:59. Their audited hashes (`8eabd9ee…` DSO, `15ce6ebe…` TSO, `P3_AUDIT_REPORT.md`,
confirmed by `P44/p44_report.json`) no longer match (`7e5aa39d…`, `fbfaa6b1…`). The P5.12-R copies
under `P512R/production_snapshots/` are different captures (`d2342ba2…`, `2814948e…`). **The audited
originals are not recoverable from disk.** Attribution to G1 is most likely but not proven: no hash was
checked between P44 and G1. Cause: `p56a_oracle.fresh_planning` isolates `logs_dir` but not
`results_dir`; the comparator callback writes to the shared results directory. The P5.12-W/X frozen
plans reference the P512R copies and are unaffected.

A Worker is redirecting every holder's `results_dir` to the arm's own root and adding a
before/after hash check of the shared `FrozenSMOPF` directory, before G2 runs.

## 6. Harness defect found by this run

`network_failures_control.jsonl` reports 12 recovered / 4 unrecovered / 0 not-attempted with failure
rows showing `primary_exit: Optimal Solution Found`, `cycle` null, "unrecovered" rows with
`recovery_attempted: false`, and two rows with no agent. Network IPOPT logs still append across cycles,
so the last `EXIT:` read is a later solve. It is superseded by §1 and is being re-implemented from
production's prints, with zero-solve validation against §1.

## 7. What is NOT established

- **Which model change moved the operating point** (§4). G1 cannot separate them.
- Whether the six node-5 failures would have recovered had they been eligible.
- Whether the audited FrozenSMOPF originals were overwritten by G1 or earlier.
- The §1 classification is the Planner's reading of the prints, cross-checked against the trajectory;
  the Worker's re-implementation is pending.

## 8. Evidence (sha256, first 16; full manifest `P515G1/evidence_manifest_sha256.json`)

| artifact | hash |
|---|---|
| `P515G1/g_control.json` | `5dcae17acccd31b1` |
| `P515G1/stdout_control.log` | `fcd3cdea2b36f058` |
| `P515G1/leak_classification_control.jsonl` | `aac46f414f03d627` |
| `P515G1/network_failures_control.jsonl` (defective, superseded) | `d953162627d67fec` |
| `P515G1/esso_models_control.pkl` | `48dac4ff8090a0de` |
| `P515G1/heartbeat_control.json` | `6021ebbe8ae47c7f` |
| `P515G1_launch.log` | `7ecdb478a7b8c0ed` |
| `P515G1/esso_capture/` (219 files, 79.7 MB, hash-recorded, not committed) | see manifest |
