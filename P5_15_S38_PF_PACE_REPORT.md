# P5.15 Addendum 20 — the PF-pace arms: both converge and both help; neither certifies

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 20. Frozen spec: `data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json`
(`f1605d9b`), written before any code or run.

- **Instance:** C\* (s = 0.96875 MVA, e = 3.875 MWh, 2025, uniform across nodes 5/7/9).
- **Common configuration:** cold standalone initialization; ρ_ess = 0.01, fixed and exempt from balancing; cap 300; freeze
  after 10 unchanged cycles plus an absolute freeze at cycle 200; tolerances unchanged.
- **Cost convention:** `gross_operational_cost`.
- **Evaluations:** `p515_s38_evaluate.py`, zero solves with the guard armed; written to `<run_dir>/s38_evaluation.json`
  with both arms present.

## 1. Verdict per arm

| arm | configuration | cycles | stop | PF first pass | V / ESS first pass | certified |
|---|---|---|---|---|---|---|
| **A** `s38_A_tau0` | τ = 0 (γ = 0 on all channels); ρ_pf held at 0.198 | **133** | Boyd, cycles 131–133 | **131** | 48 / 115 | **no** |
| **B** `s38_B_pfbal` | τ = 1; PF balancing live | **151** | Boyd, cycles 149–151 | **146** | 45 / 95 | **no** |
| run 1 (reference) | τ = 1; balanced ρ_ess; cycle-60 backstop | 477 | Boyd | 226 | 45 / 475 | yes |

**Certification criteria** (spec v9):

| criterion | A | B |
|---|---|---|
| (a) Boyd stop within 300 | pass (133) | pass (151) |
| (b) no V/PF clamp | pass | pass |
| (c) storage terminal ratio < 0.9 | **fail — 0.948** | **fail — 0.915** |
| (d) cost within rule-nine bar of run 1 | **fail** — \|Δ\| 39,149 > bar 3,140 | **fail** — \|Δ\| 16,231 > bar 1,028 |

- **Both arms help** (PF first pass ≤ 180).
- **Adoption rule (c):** both help, so one combined confirmation run (`s38_C_combined`: τ = 0 with PF balancing live) is
  authorized. It is **launched only after this review**; the code refuses to launch it until it is re-authorized.
- **Fallback not triggered for A:**
  - 3 TSO (case9) failures, all recovered, a rate of 0.023 per cycle against a trigger at 0.243;
  - no unrecovered TSO failure;
  - no PF or ESS oscillation flag.

## 2. Predictions (recorded in spec v9 before running)

| prediction | outcome |
|---|---|
| PF first pass well before 226 (≤ 180) in at least one arm | **Met** — A 131, B 146. |
| A is the stronger candidate (A's PF first pass earlier than B's) | **Met** — 131 against 146. |

## 3. What each lever did

### 3.1 ρ / γ trajectories

| | A | B |
|---|---|---|
| ρ_v | 0.0077 → 0.01155 (cycle 1) → 0.01733 (cycle 2); frozen from cycle 12 | 0.0077 → 0.01155 (cycle 1); frozen from cycle 11 |
| γ_v / γ_pf / γ_ess | 0 / 0 / 0 | = ρ on every channel |
| ρ_pf | 0.198 fixed (exempt) | 0.198 → **0.132 (cycle 79) → 0.088 (cycle 80)**; frozen from cycle 90; never at the clamp |
| ρ_ess | 0.01 fixed | 0.01 fixed |

- **B's first decrease matches the prediction.** It came at cycle 79, the cycle the zero-solve replay predicted for the s37
  ρ_ess 0.01 trajectory (`P515S38/balancing_replay/`, validated exactly on 2,331 channel-cycles). B's trajectory is
  identical to that run until then.
- **The replay's open-loop cascade to the ρ floor did not occur.** In the closed loop, balancing made two decreases and
  then held.
- **A's global τ = 0 also changed V.** Without the V proximal term, balancing took ρ_v one step higher (0.0173 against
  0.0116). As pre-registered, this is a side effect of the global τ parameter, not a PF effect.

### 3.2 PF residual ratios (primal / dual)

| cycle | run 1 dual | A | B |
|---|---|---|---|
| 10 | 164.2 | 76.5 / 71.7 | 132.0 / 163.4 |
| 30 | 24.97 | 6.27 / 11.76 | 8.40 / 24.97 |
| 50 | 11.19 | 2.98 / 6.32 | 4.17 / 11.18 |
| 75 | 6.55 | 2.55 / 3.84 | 2.82 / 6.56 |
| 100 | 4.91 | 0.59 / 1.71 | 2.13 / 3.48 |
| 125 | 3.52 | 0.08 / 1.09 | 1.24 / 1.70 |
| 133 / 150 | 2.73 (at 150) | 0.17 / 0.95 (133) | 0.70 / 0.91 (150) |

- **A: removing the proximal term speeds up the PF dual residual from the first cycles** (about half of run 1's value by
  cycle 30), consistent with HP1.
- **B: identical to the reference until cycle 79.** Lowering ρ_pf then cut the dual residual but raised the primal
  (0.91 → 2.30 by cycle 90). The net gain came later and was smaller than A's, consistent with HP2 acting on the late
  phase only.

### 3.3 Per-entry PF decomposition (new capture)

- **Capture check:** in both arms the per-entry PF records reproduce the production residual norms every cycle (maximum
  relative error 4.6e-16).
- **Where the PF dual residual (s²) sits, identical in character in A and B:**
  - it is spread across nodes, years and P/Q at cycle 1;
  - by cycle 5–10 it is **≥ 97 % active power (P)**;
  - from cycle 20 onward **75–95 % sits at the node 7 interface**, and from cycle 50 it is essentially 100 % P;
  - late in the run the dominant year shifts to 2030 (61–67 % at the stop, both arms).
- **Reading:** the slow tail is a narrow set of active-power interface entries at one node, not a system-wide pace. This is
  an observation. The mechanism (e.g. a binding interface or voltage limit at node 7 in 2030) is not established.
- **Limitation:** the evaluator's `late_phase_decay_rate` is a geometric mean over unequally spaced sampled cycles (for A:
  100, 125, 133). It is not a per-cycle rate and is not used here.

## 4. Why neither certifies: the storage channel and settling

| terminal ratio, max(primal, dual) | V | PF | ESS | rule ten (objective step / tolerance) |
|---|---|---|---|---|
| A | 0.20 | **0.949** | **0.948** | 0.044 |
| B | 0.16 | 0.892 | **0.915** | 0.012 |

- **Channels pass only at the stop.** In A the PF and ESS channels pass on just 3 and 15 cycles of 133; in B on 6 and 18
  of 151.
- **Both stops are at the threshold.** They meet Boyd's rule, but PF (A) and ESS (both) terminate at about 0.9–0.95 of
  threshold: **stopped, not settled** in the rule-ten sense. The ESS channel at ρ_ess = 0.01 hovers around its threshold,
  as seen in s37 and recorded by Addendum 20 as unsettled (two-phase ESS schedule deferred). It is now the channel that
  fails criterion (c).
- **Cost.**
  - A ends 39,149 below run 1 and B ends 16,231 below; bars 3,140 and 1,028; terminal steps 2,884 (A) and 772 (B), against
    256 for run 1.
  - A rule-nine bar bounds stopping slack only between settled runs. With the storage channel stopped at threshold in
    both arms, **these differences are not interpretable as a different or lower-cost point.** They are reported as
    unresolved.

## 5. Monitoring

| | A | B |
|---|---|---|
| local-solve failures | 0 | 0 |
| network failures (all recovered) | 35 (33 first attempt, 2 second); TSO 3 | 48 (46 first attempt, 2 second); TSO 6 |
| oscillation flags, PF / ESS | none / none | none / none |
| EFC/day max at stop | 1.146 (first ≥ 1.06 at cycle 25) | 1.141 (first ≥ 1.06 at cycle 33) |
| solves; wall clock | 6,871; 4,906 s (36.9 s/cycle) | 7,802; 5,746 s (38.1 s/cycle) |
| run 1, for comparison | — | 24,690 solves; 17,429 s |

## 6. Addendum 20 (a): balancing replay

`WORKER_REPORT_S38_REPLAY.md` (`07d973f2`, `05c6e3d6`, `9d11123e`).
- **Method:** the real `_update_admm_penalties`, validated by reproducing every logged action, ρ/γ, freeze flag and streak
  exactly for run 1 and both s37 arms.
- **Result:** without the cycle-60 backstop, ρ_pf would first have been lowered (0.198 → 0.132) at cycle **69** (run 1),
  **79** (s37 at ρ_ess 0.01) and **103** (s37 at ρ_ess 0.001).
- **Caveat:** the replay feeds the rule recorded residuals, so only the first action is predictive. Arm B confirmed cycle
  79 in the closed loop.
- **Side finding:** at the ρ floor, repeated "decreased" labels prevent the unchanged-streak freeze. It did not arise in B.

## 7. Step 3.6: per-phase timing (run in the gap between A and B)

`WORKER_REPORT_S36_TIMING.md`, `WORKER_REPORT_S36_TIMING_MEASUREMENT.md` (`fcfebbbe`, `e22c6975`, `1e7b0a69`,
`926dbc28`, `23da8685`).
- **Instrumentation:** implemented harness-side by wrapping existing functions; no production edits.
- **Measurement:** 2 cold cycles, recorder off and on. The outputs are identical apart from log paths.

| per cycle | cycle 1 | cycle 2 |
|---|---|---|
| wall | 35.1 s | 38.7 s |
| IPOPT (DSO / TSO / ESSO) | 16.6 / 1.4 / 0.1 s | 20.4 / 1.2 / 0.1 s |
| overhead absorbable by a per-block worker | 16.95 s | 16.98 s |
| — NL write / clone / `.sol` parse / load / parameter update | 6.0 / 4.1 / 3.4 / 1.6 / 1.6 s | 6.3 / 4.4 / 3.6 / 0.8 / 1.6 s |
| overhead that stays serial | 0.45 s | 0.44 s |
| **absorbable share (screening bar 70 %)** | **97.4 %** | **97.5 %** |
| projected speed-up on 8 workers (Amdahl, ignoring IPC) | 5.0× | 4.1× |

- **First analysis was defective.** It double-counted clones, mixed initialization solves into the cycle totals, and
  counted IPOPT as serial. It is retained as v1 and superseded by v2; the corrected figures above are v2.
- **The screening rule passes.** Under Addendum 19 this licenses building the persistent-worker path.
- **I have not started that build.** It is a production solve-path change supported by two cycles of evidence.
- **Model clones cost 4.1–4.4 s per cycle.** They are failure-snapshot machinery; the TSO clones on every cycle. That
  cost is removable without any parallelism.
- **Limitations:** two early cycles only (IPOPT is heavier than the audit's 13 s median); one recovery re-solve, reported
  separately; IPC cost not modelled.

## 8. Incidents

- **Working-directory collisions (the s37 precedent recurred one layer down).** The two-cycle preflights left four empty
  pre-check and probe directories under the ids the real launches check. None is cited by evidence; all were moved aside
  with the suffix `__moved_aside_20260917T134223Z` before arm A launched.
- **Arm A's first launch used the arm key instead of the lowercase CLI name** (`s38_a_tau0`). The harness printed its usage
  text and exited 1 before anything ran. That log and exit code are preserved as `*_attempt1_wrong_cli_name_*`.
- **The timing measurement's first launch refused** because its process scan matched the Planner's own launcher shell,
  whose command text contained the harness filename. Nothing ran; preserved as `*_attempt1_self_match_*`. Relaunched
  through a script.
- **Evaluator mislabel.** With only one arm present, the evaluator marks both predictions "SCORED". The recorded
  evaluations were run with both arms present, and their scoring was checked against the code.

## 9. Questions for review

1. **Combined run (c).** Both arms help, so launch `s38_C_combined` (τ = 0 with PF balancing live) at cap 300? If
   confirmed, its configuration becomes the oracle and Step 3.5 follows. Its certification would still be expected to
   fail criterion (c) on the storage channel unless that is addressed.
2. **The storage channel now fails certification in every arm.** Should the deferred two-phase ESS schedule (low ρ_ess
   while walking; balancing re-enabled after 5 cycles with dual ratio < 1) be brought forward into the combined run? Or
   should criterion (c) be read differently for the ESS channel?
3. **PF tail at node 7.** Does the concentration (P, node 7, 2030 late) warrant a zero-solve look at that interface
   (limits, voltage, flexibility bounds) before the combined run?
4. **Step 3.6.** The screening passes at 97 %. Build the persistent-worker path now? Or first remove the per-cycle
   clone cost, and re-measure over more cycles?

## Evidence

- Arm runs: `data/SRP1/Results/P515S38_A_TAU0_run/`, `.../P515S38_B_PFBAL_run/`, with evaluations, launch logs, exit
  codes, the attempt-1 record and per-arm `evidence_manifest_sha256.json`.
- `pf_entry_stride_*.jsonl` (80 / 96 MB), `esso_capture/` and `results/` are hash-recorded, not committed.
- Preflights, zero-solve checks and the replay were committed earlier (`c315d39d`, `11332cca`, `1ef32245`, `cb673854`,
  `9d11123e`). The Step 3.6 measurement was committed in `23da8685`.
