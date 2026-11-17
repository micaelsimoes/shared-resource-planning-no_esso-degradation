# P5.15 Addendum 20 — handoff to the External Expert and the Author

**Planner report, 2026-09-17.** Self-contained; continues `P5_15_ADDENDUM19_EXPERT_REPORT.md`. Full stage tables are in
`P5_15_S38_PF_PACE_REPORT.md` (`dfcfcdd6`).

- **Instance:** candidate C\* (s = 0.96875 MVA, e = 3.875 MWh, invested 2025, uniform across nodes 5/7/9).
- **Cost convention:** `gross_operational_cost`.

## 1. Verdict

**Both arms now reach a Boyd stop, 3–3.5× sooner than run 1, and both "help". Neither certifies.** The PF channel no
longer sets the pace. Certification now fails on criterion (c), storage terminal ratio < 0.9, and on the cost bar; §5
questions whether (c) is well posed.

| arm | configuration | Boyd stop | PF first pass | certified | failing criteria |
|---|---|---|---|---|---|
| **A** `s38_A_tau0` | τ = 0 (γ = 0 on V, PF, ESS); ρ_pf held at 0.198 | **cycle 133** | **131** | no | (c) storage 0.948; (d) cost |
| **B** `s38_B_pfbal` | τ = 1; PF balancing live, no backstop | **cycle 151** | **146** | no | (c) storage 0.915; (d) cost |
| run 1 (reference) | τ = 1; ESS balanced; cycle-60 backstop | cycle 477 | 226 | yes (under Addendum 16) | — |

- **Predictions (recorded before running):** both met — PF first pass ≤ 180 in at least one arm (A 131, B 146), and A
  earlier than B.
- **"Helps" (PF first pass ≤ 180):** both arms. Under Addendum 20 (c), one combined confirmation run is authorized. **It
  has not been launched**, pending this review.
- **Arm A's fallback (τ = 0.25) was not triggered.** A had 3 TSO failures, all recovered (0.023 per cycle against a
  trigger at 0.243), no unrecovered TSO failure, and no PF or ESS oscillation flag.

| item | status | commit |
|---|---|---|
| Frozen spec v9; `REVISION_CONTEXT.md` records Addenda 17–20 | done | `f1605d9b` |
| (a) zero-solve balancing replay | done, validated exactly (§2) | `07d973f2`, `05c6e3d6`, `9d11123e` |
| τ ≥ 0 loader change (the only production edit) and zero-solve checks | done | `c315d39d` |
| s38 arms, per-entry PF capture, evaluator, preflights | done | `11332cca`, `1ef32245`, `cb673854` |
| (b) arms A and B, evaluations, stage report | done | `dfcfcdd6` |
| Step 3.6 instrumentation and measurement (§6) | done; screening passes | `fcfebbbe`, `e22c6975`, `1e7b0a69`, `926dbc28`, `23da8685` |

## 2. (a) Balancing replay: would ρ_pf have been lowered without the backstop?

- **Method:** the real production `_update_admm_penalties`, driven with stand-in models and the recorded per-cycle
  residuals.
- **Validation:** it reproduced every logged action, ρ, γ, freeze flag, streak and clamp flag for run 1 and both s37 arms:
  2,331 channel-cycles, 0 mismatches.
- **Answer: yes.** ρ_pf would first have dropped 0.198 → 0.132 at cycle **69** (run 1), **79** (s37 at ρ_ess 0.01) and
  **103** (s37 at ρ_ess 0.001).
- **Caveat:** open loop — the recorded residuals don't respond to the change — so only the first action is predictive.
- **Closed-loop check:** arm B shares the s37 ρ_ess 0.01 trajectory until then, and **dropped ρ_pf at exactly cycle 79.**
- **Side finding:** at the ρ floor (1e-4), repeated "decreased" labels prevent the unchanged-streak freeze, so a channel
  could sit at the clamp until cycle 200. This did not occur in B.

## 3. (b) The arms

### 3.1 Configuration as run (spec v9)

- **Common to both arms:**
  - run 1's configuration via overrides (the case file is not edited);
  - cold standalone initialization;
  - ρ_ess = 0.01, fixed and exempt from balancing;
  - cap 300; freeze after 10 unchanged cycles plus an absolute freeze at cycle 200;
  - per-entry PF capture as standard; tolerances unchanged.
- **Planner interpretations, recorded in spec v9 before running:**
  - **Arm A holds ρ_pf fixed,** so that A isolates HP1 and B isolates HP2. Otherwise A would already be the combined run.
  - **τ is a single global parameter,** so τ = 0 removes the proximal term on V and ESS as well as PF.
- **Production change:** the loader previously rejected τ ≤ 0. This was relaxed to τ ≥ 0 after a search found no
  division by τ or γ.

### 3.2 ρ / γ trajectories

| | A | B |
|---|---|---|
| ρ_v | 0.0077 → 0.01155 (cycle 1) → **0.01733** (cycle 2); frozen from 12 | 0.0077 → 0.01155 (cycle 1); frozen from 11 |
| γ (V / PF / ESS) | 0 / 0 / 0 | = ρ on each channel |
| ρ_pf | 0.198, fixed | 0.198 → 0.132 (cycle 79) → **0.088** (cycle 80); frozen from 90; never at the clamp |
| ρ_ess | 0.01 | 0.01 |

The extra ρ_v step in A comes from the global τ, which also removes the V proximal term. It is a side effect of that
parameter, not a PF effect.

### 3.3 PF residual ratios (primal / dual)

| cycle | run 1 (dual) | A | B |
|---|---|---|---|
| 10 | 164.2 | 76.5 / 71.7 | 132.0 / 163.4 |
| 30 | 24.97 | 6.27 / 11.76 | 8.40 / 24.97 |
| 50 | 11.19 | 2.98 / 6.32 | 4.17 / 11.18 |
| 75 | 6.55 | 2.55 / 3.84 | 2.82 / 6.56 |
| 90 | — | — | 2.30 / 4.79 |
| 100 | 4.91 | 0.59 / 1.71 | 2.13 / 3.48 |
| 125 | 3.52 | 0.08 / 1.09 | 1.24 / 1.70 |
| stop | 0.09 (477) | 0.17 / 0.95 (133) | 0.70 / 0.89 (151) |

**Reading.** Both confirm the accepted HP1 + HP2 reading; HP1 is the larger lever.
- **A (HP1):** removing the proximal term speeds the PF dual residual from the first cycles; it is about half of run 1's
  value by cycle 30.
- **B (HP2):** identical to the reference until the first ρ_pf decrease at cycle 79.
  - The decrease cut the dual residual, but the primal residual rose (0.91 at cycle 80 → 2.3–2.7 over cycles 81–91).
  - The net gain came late and was smaller than A's.

### 3.4 Per-entry PF decomposition (new capture)

**Capture check:** in both arms, the per-entry records reproduce the production PF residual norms every cycle (maximum
relative error 4.6e-16).

The late PF dual residual is **not system-wide**. In both arms:
- s² is spread across nodes, years and P/Q at cycle 1;
- it is **≥ 97 % active power (P) by cycle 5–10**, and ≥ 99 % P from cycle 30;
- **75–95 % sits at the node 7 interface** from cycle 20 on;
- 2030 dominates at the stop (61–67 %).

This is an observation. The mechanism — for example a binding interface, voltage or flexibility limit at node 7 in 2030 —
has not been examined.

### 3.5 Monitoring

| | A | B |
|---|---|---|
| local-solve failures | 0 | 0 |
| network failures (all recovered) | 35 (33 on the first recovery attempt, 2 on the second); TSO 3 | 48 (46 first attempt, 2 second); TSO 6 |
| oscillation flags, PF / ESS | none / none | none / none |
| EFC/day: first ≥ 1.06 / at stop | cycle 25 / 1.146 | cycle 33 / 1.141 |
| solves; wall clock | 6,871; 4,906 s | 7,802; 5,746 s |
| run 1, for comparison | 24,690 solves; 17,429 s | |

## 4. Cost

| run | cost at cycle 100 | cost at stop | terminal step | rule ten |
|---|---|---|---|---|
| run 1 | 651,632k | 651,039,166 (477) | 256 | 0.0039 |
| A | 650,968k | 651,000,017 (133) | 2,884 | 0.044 |
| B | 651,009k | 651,022,935 (151) | 772 | 0.012 |

- **Criterion (d) fails in both arms.** A ends 39,149 below run 1 (bar 3,140); B ends 16,231 below (bar 1,028).
- **The differences are not interpretable.**
  - A rule-nine bar bounds stopping slack only between settled runs, and both arms stop with at least one channel at
    about 0.9–0.95 of its threshold (§5).
  - The single-cycle terminal step is also noisy: A's objective change was 89 at cycle 100, 43 at 125 and 2,884 at 133.
- **No claim of a lower-cost or different point is made.**
- The differences are 0.002–0.006 % of cost.

## 5. Why certification fails, and whether criterion (c) is well posed

### 5.1 Terminal ratios, max(primal, dual)

| | V | PF | ESS |
|---|---|---|---|
| A | 0.20 | **0.949** | **0.948** |
| B | 0.16 | 0.892 | **0.915** |
| run 1 | 0.013 | 0.09 | **0.988** |

### 5.2 Terminal tails, last 10 cycles

| | ESS primal ratio | PF dual ratio |
|---|---|---|
| A (cycles 124–133) | 0.90, 0.88, 0.88, 0.87, 0.87, 0.90, 0.96, 0.93, 0.89, 0.95 | 1.11 → 0.95, monotone |
| B (cycles 142–151) | 0.97, 0.93, 1.07, 1.25, 1.27, 1.21, 1.09, 1.00, 0.95, 0.92 | 1.06 → 0.89, monotone |

### 5.3 Observations

1. **Run 1 does not meet criterion (c) either.** Its storage channel stopped at 0.988 of threshold. (c) was introduced
   for the Addendum 17 gate and carried into Addenda 19–20, after run 1 had been certified under Addendum 16's criteria.
   The arms are being held to a bar the reference does not meet.
2. **The channel that closes a Boyd stop ends near 0.95 almost by construction.** The stop needs three consecutive
   passing cycles. A ratio decaying at about 0.98 per cycle is about 0.98² ≈ 0.96 two cycles after crossing 1. Hence PF
   ends at 0.95 in A and 0.89 in B, and run 1's storage at 0.988.
   - "< 0.9" is reachable only for a channel that crossed several cycles before the stop.
   - (c) therefore tests which channel closes the stop as much as how settled storage is.
3. **Storage is genuinely not settled at ρ_ess = 0.01.**
   - In A it had passed on each of the last 10 cycles, yet sat at 0.87–0.96 rather than decaying. It hovers; it does not
     converge.
   - In B it re-crossed 1 at cycle 149, in step with PF.
   - This is Addendum 20's "under-damped at low ρ once the gradient is spent". A two-phase ESS schedule is the recorded
     remedy, and it was deferred.
4. **Rule ten would call PF in A and storage in both arms "stopped, not settled".** It says the same of run 1's storage
   channel.

## 6. Step 3.6 — per-phase timing

**Implementation:**
- **Harness-side only,** by wrapping existing functions; no production edits.
- **Zero-solve checks:** 61 of 61 pass.
- **Measurement:** 2 cold cycles of the run-1 configuration class, recorder off and on. The outputs are identical except
  for log paths.

| per cycle | cycle 1 | cycle 2 |
|---|---|---|
| wall | 35.1 s | 38.7 s |
| IPOPT (DSO / TSO / ESSO) | 16.6 / 1.4 / 0.1 s | 20.4 / 1.2 / 0.1 s |
| overhead absorbable by a per-block worker | 16.95 s | 16.98 s |
| — NL write | 6.0 s | 6.3 s |
| — model clones (failure snapshots) | 4.1 s | 4.4 s |
| — `.sol` parse | 3.4 s | 3.6 s |
| — load into model | 1.6 s | 0.8 s |
| — parameter update | 1.6 s | 1.6 s |
| overhead that stays serial | 0.45 s | 0.44 s |
| **absorbable share (interim bar X = 70 %)** | **97.4 %** | **97.5 %** |
| projected speed-up on 8 workers (Amdahl) | 5.0× | 4.1× |

- **The first analysis was defective.** It subtracted the clones as if nested inside the solve, counted 156 initialization
  records inside the two cycles, and classed IPOPT as serial. It reported 0.29 and 1.04×. That version is retained,
  superseded by the corrected v2 shown above.
- **The screening rule passes by a wide margin.** Under Addendum 19 this licenses building persistent workers. **The
  build has not been started.**
- **The clone cost is removable without parallelism.** The TSO clones every block on every cycle for its failure-snapshot
  callback.
- **Limitations:**
  - only two early cycles, and IPOPT here is heavier than the earlier audit's 13 s median;
  - one recovery re-solve, reported separately;
  - the projection ignores inter-process communication.

## 7. Process notes

- **Working-directory collisions recurred one layer down.** The two-cycle preflights left four empty pre-check and probe
  directories under the ids the real launches check. None was cited by evidence; all were moved aside with a timestamp
  suffix before arm A launched.
  - The harness's preflight/arm id derivation fixes the run directory but not these two; to be closed structurally
    before arm C.
- **Two refused launches, nothing run in either:**
  - arm A's first launch used the arm key instead of the lowercase CLI name;
  - the timing measurement's process scan matched the Planner's own launcher shell, whose command text contained the
    harness filename (a limitation of substring process scans).

  Both attempts' logs and exit codes are preserved.
- **Evaluator mislabel.** With only one arm present, the evaluator labels the predictions "SCORED". The recorded
  evaluations were run with both arms present, and the scoring was checked against the code.
- **Hash-recorded, not committed:** the per-entry PF records (80 / 96 MB), `esso_capture/` and `results/`. All other
  evidence is committed with per-arm sha256 manifests.

## 8. Options (author decisions) and the Planner's recommendation

1. **Settle the certification bar first.** Criterion (c) is not met by run 1 and is partly an ordering artefact (§5).
   Options:
   - (i) keep (c) and treat the storage channel as the next lever;
   - (ii) replace it with a settling test that does not depend on which channel closes the stop, e.g. storage below 0.9
     on each of the last k cycles, or a continuation of N cycles past the Boyd stop with all channels inside tolerance;
   - (iii) accept run 1's Addendum 16 criteria (Boyd stop + rule ten on the objective), with channel ratios reported.

   **Recommended before any further run,** because it decides whether arm C can certify at all.
2. **Run the authorized combined arm C** (τ = 0 with PF balancing live; ρ_ess 0.01, exempt; cap 300), with predictions
   recorded first: PF first pass ≤ A's 131; Boyd stop ≤ 140.
   - Under the current (c) it is expected to fail on storage, for the same reason as A and B.
   - **Recommended once option 1 is decided.**
3. **Two-phase ESS schedule** (Addendum 20, deferred): low ρ_ess while storage walks, then balancing re-enabled after its
   dual ratio has been < 1 for 5 cycles.
   - This is the direct remedy for the hovering storage channel.
   - It changes a second lever, so the Planner recommends it as a **separate arm after C**, not folded into C, unless the
     Author prefers a single confirmation.
4. **Node 7 interface, zero-solve** (the PF tail concentration): inspect interface flow against its limits, voltage, and
   flexibility bounds at node 7 in 2030 at the stop. Cheap, and can run in parallel with anything.
5. **Step 3.6:**
   - (i) build persistent workers now, per the passed screening rule;
   - (ii) first gate the per-cycle TSO clones (they cost about 4 s of 35 s) and re-measure over about 10 cycles, then
     build.

   The Planner recommends **(ii)**, because the measurement rests on two early cycles.

## 9. Questions for the Expert

1. Do you accept that PF is no longer the pace-setter, and that HP1 (proximal term) is the larger lever and HP2 (late
   ρ_pf) the smaller?
2. Criterion (c): run 1 fails it (0.988), and the closing channel of a Boyd stop ends near 0.95 by construction. Keep it,
   replace it with an ordering-independent settling test, or return to Addendum 16's criteria?
3. Combined arm C: launch as authorized, with the storage channel unchanged? Or fold in the two-phase ESS schedule, or
   run that separately?
4. Does the PF tail at the node 7 interface (P, 2030) warrant the zero-solve look before arm C?
5. Step 3.6: build persistent workers now, or gate the TSO clones and re-measure first?
6. The global τ = 0 also removed the V and ESS proximal terms. Should the oracle use τ = 0 globally, or should τ become
   per-channel (a production change)?

## 10. Evidence index

| item | artifact | commit |
|---|---|---|
| prior handoff | `P5_15_ADDENDUM19_EXPERT_REPORT.md` | `e75f40f5` |
| spec v9; record update | `data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json`; `REVISION_CONTEXT.md` | `f1605d9b` |
| balancing replay | `p515_s38_balancing_replay.py`, `P515S38/balancing_replay/`, `WORKER_REPORT_S38_REPLAY.md` | `07d973f2`, `05c6e3d6`, `9d11123e` |
| τ loader change; zero-solve checks | `admm_parameters.py`, `p515_s38_zero_solve_checks.py`, `P515S38/zero_solve_checks/` | `c315d39d` |
| arms, PF capture, evaluator, preflights | `p515_g_g1_g4_admm_gates.py`, `p515_s38_evaluate.py`, `p515_s38_preflight.py`, `P515S38/preflight_{A,B}/`, `WORKER_REPORT_S38_PREP.md` | `11332cca`, `1ef32245`, `cb673854` |
| arm runs, evaluations, stage report | `P515S38_A_TAU0_run/`, `P515S38_B_PFBAL_run/` (each with `s38_evaluation.json` and `evidence_manifest_sha256.json`), launch logs, exit codes, attempt-1 record, `P5_15_S38_PF_PACE_REPORT.md` | `dfcfcdd6` |
| Step 3.6 instrumentation | `p515_s36_step36_timing*.py`, `WORKER_REPORT_S36_TIMING.md`, zero-solve checks v1–v3 | `fcfebbbe`, `e22c6975`, `1e7b0a69`, `926dbc28` |
| Step 3.6 measurement | `P515S36/step36_timing/` (off/on, v1 and v2 analyses, identity checks, manifests), `WORKER_REPORT_S36_TIMING_MEASUREMENT.md` | `23da8685` |
| run 1 reference | `P515S35_REF_run/g_baseline.json` (sha256 `8c72a156…`) | earlier |

**Zero-solve checks.** Every evaluator, replay and analysis above ran with `SolveProfileGuard` armed: zero solves.

**Sources for figures not in a stage artifact:**
- §4 cost at cycle 100, §5.2 terminal tails and §5.3's run 1 ratio 0.988 are direct field reads from the committed
  `g_*.json` trajectories.
- §3.3's cycle-90 value for B is from its trajectory.
