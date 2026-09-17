# P5.15 Addenda 17–18 — handoff to the External Expert

**Planner report, 2026-09-17.** Self-contained; continues `P5_15_ADDENDUM16_EXPERT_REPORT.md`. The decisions requested
are the same as in `P5_15_ADDENDUM17_DECISION_NOTE.md` (`44389ad6`); this report carries the technical detail behind
them. Instance throughout: candidate C\* (s = 0.96875 MVA, e = 3.875 MWh, invested 2025, uniform across nodes 5/7/9).
Configuration is run 1's (s34 re-scaling, ρ_ess start 0.1125).

## 1. Verdict

**The Addendum 17 gate was not run.** Zero-solve diagnostics showed that its dual-initialization route cannot be
implemented or validated as authorized:
- the production input (cycle-0 LMPs) is structurally zero at the storage buses;
- the mapping test cannot discriminate a correct mapping from a wrong one;
- the slow component of the storage duals is not cleanly the part that prices determine.

Run 1 remains the programme's certified evaluation. Nothing long-running was started.

| item | status | commit |
|---|---|---|
| Addenda 16–17 recorded; §6 claims-to-avoid adopted verbatim | done | `db945601` |
| Step 1 diagnostics: availability audit, dual-direction comparison | partial (§3.1) | `5e2ba6f9` |
| Exact reconstruction of run 1's storage duals, per agent, per entry, 477 cycles | **pass** (§3.2) | `b63e4d94` |
| Cycle-0 node-balance dual capture; run-1 replay arm (prepared, not launched) | done | `b8f129e0` |
| Calibration of the cycle-0 duals | the zero is real (§4, F1) | `b6384d99` |
| Identifiable-versus-split decomposition (H1/H2) | **inconclusive** (§4, F3) | `97507b98` |
| Step 3.6 audit (parallel path, cycle-time attribution) | done (§6) | `710af0de` |
| Step 4 warm-continuation validation | design only (§7) | `7f1c75bf` |
| Decision note | — | `44389ad6` |

## 2. What Addendum 17 required

1. Diagnostics: (a) compare the dual directions of gate 3 (cycle 150) and run 1 (cycle 477); (b) the price-taker LP with
   run 1's terminal nodal prices; (c) the same LP with cycle-0 LMPs from the initialization solves.
2. Initialize per-agent storage duals from cycle-0 LMPs — the marginal value of each storage copy at the LP schedule,
   mapped into the AL's normalized units and projected onto the zero-sum invariant — after validating the mapping
   against run 1's terminal duals.
3. Set the ρ defaults (0.0116 / 0.088 / 0.1125).
4. Run a gate at cap 150: Boyd on all three channels, cost within the rule-nine bar, storage terminal ratio < 0.9.

## 3. What the saved evidence supports

### 3.1 Availability
Scoped search of both runs' committed and hash-recorded artifacts — trajectories, terminal files, stride and sidecar
files, ESSO captures and pickles, `results/`, preflight and check directories:
- **Storage consensus duals:** norms only for TSO/DSO; per entry only for the ESSO (pickled).
- **Consensus z and per-agent copies x:** per entry, every cycle, in run 1 (stride 1); stride 5 in gate 3.
- **Nodal prices (node-balance duals):** never serialized, at any cycle, in either run.
- **Diagnostic (a):** limited to what exists. Per-agent norm ratios (gate 3 / run 1) are 0.31 / 0.30 / 0.29; the ESSO
  per-entry cosine is 0.59.
- **Diagnostic (b):** needs run 1's terminal prices, which only a deterministic replay (~5 h) can provide. The replay
  arm is prepared, with a bitwise-identity check against run 1 and a zero-solve dry checklist that passes, but it was
  **not launched**.

### 3.2 Exact dual reconstruction
From production's update rule, replayed over run 1's stride file:
- a = 1/(2·S_ref), with S_ref = 2.5 MVA;
- z = [Σ_i ρ·a²·x_i + a·Σ_i λ_i^{c−1}] / [Σ_i ρ·a²];
- λ_i ← λ_i + ρ·a·(x_i − z), starting from λ = 0.

Cross-checks:
- per-agent norms match the saved trajectory to **≤1.5e-14 relative** at all 477 cycles;
- ESSO per-entry duals match its pickle to **3.3e-10 relative**, once a one-cycle timing offset is accounted for (the
  ESSO loads λ from the previous cycle before its own update).

This reconstruction is exact and reusable.

## 4. The three findings

### F1 — Cycle-0 LMPs at the storage buses are structurally zero
- **The conversion is correct.** At capture time the active objective is the raw, unscaled one. The node-balance dual
  convention was validated on an independent fixture at an interior generator bus: dual / (baseMVA · c_p) =
  **0.9999999952** over 24 periods, with no sign flip and no hidden factor.
- **The zero is real.** Buses 5, 7 and 9 host no controllable generator. At initialization the interface settlement
  weight is 0, so the signed interface flexibility variable `interface_delta_p` carries no objective weight. Its KKT
  stationarity forces the node-balance dual at those buses to ≈0, independent of generation cost. The captured values
  are ≈1e-9 of the market price.
- **Consequence:** the authorized production input carries no price information, and diagnostic (c) is not computable
  this way.

### F2 — The mapping test is ill-posed
From the code's augmented Lagrangian (TSO objective divided by `effective_scale` = σ/w_b; ESSO coordination terms scaled
by c_E = `al_scale_esso` ≈ 2.272e5; common a = 1/(2·S_ref)), stationarity of each storage copy at x_i = z gives:
- **TSO:** a·λ_T = −[(w_b/σ)·LMP + κ_T], where κ_T collects the storage-row multipliers in that block;
- **DSO:** a·λ_D = −κ_D. **No price term.** The storage adds to the reference-node load, the reference generator carries
  no cost, and the settlement acts on pg_ref − pnet, so the price cancels. The DSO reference-node dual is therefore not
  a storage price.
- **ESSO:** c_E·a·λ_E = −(EPS·∂throughput/∂x + κ_E). **No price term.**
- **Q:** only the TSO has a Q price.

Summing with the code's invariant Σ_i λ_i = 0 leaves **one** condition:
(w_b/σ)·LMP + κ_T + κ_D + κ_E/c_E = 0. **Prices determine only this sum.**

The split across agents is free along the gradients of the storage constraints active at the schedule: day balance
(η_ch·dt when charging, dt/η_dch when discharging), saturated power limits, SoC bounds, and idle periods.

Run 1 shows the split moving independently of the total:
- **cycle 1:** norms 8.08 / 4.04 / 4.04e-3 — exactly the mean-subtraction pattern;
- **cycle 477:** 6.27 / 5.34 / 2.15e-3;
- **cycles 346–477:** the total norm is flat while the ESSO share rises 36 %.

"Reproduce run 1's terminal per-agent duals from its prices" can therefore be failed by a correct mapping and passed by
a wrong one.

### F3 — The slow component is not cleanly the identifiable one
**Method.** Per (node, year, day), P channel, at run 1's terminal schedule, the span S of the active storage-row
gradients was built from:
- the day-balance vector;
- power-limit periods;
- SoC-bound periods, with SoC reconstructed from pch/pdch with η_ch 0.97 / η_dch 0.96 and post-SoH energy;
- optionally, idle periods.

S was orthonormalized. Each agent's dual was split each cycle into I = P⊥λ (identifiable) and J = P_Sλ (the in-span
split). Settling was pre-registered as "relative change < 1 % per 50 cycles through 477".

**Results:**
- dim S = 5–14 of 24 on every block, in both variants, so the decomposition is not vacuous;
- **neither component settles by cycle 477:** ‖I‖ changes 2.0 % per 50 cycles and ‖J‖ 2.6 %; the ESSO's own
  components drift most, 9–11 %;
- **the identifiable share ‖I‖²/‖λ‖² falls from 0.685 (cycle 1) to 0.520 (cycle 477)**;
- gate 3 is not computable (stride 5).

**Reading.** This is formally inconclusive but leans toward H2: the slow part includes the unidentified split, which no
price-based initialization can set.

## 5. Options (author decisions) and the Planner's recommendation

1. **Price the interface during initialization, then retry.** Run the 51 initialization solves with the interface
   settlement active, so cycle-0 storage-bus LMPs carry price information. This changes the initialization problem, not
   the ADMM fixed point. Initialize **only the identifiable component**, with the split set by an explicit, stated
   choice (e.g. mean subtraction). First test is cheap: 51 solves plus the F1 check. F3 predicts the storage channel may
   still drift.
2. **Replay run 1 (~5 h).** Validates the units and sign of a revised mapping and the identifiable component, but
   cannot supply a production input.
3. **Accept run 1 as the certified oracle and proceed.** Carry storage as the one-sided statement; then Step 3.6
   (parallelism shortens long certified runs), then Step 4 with the warm-continuation validation. **Recommended.**
   Addendum 18 sequenced Step 3.6's preflight and gate after the Addendum 17 gate, so this needs confirmation.
4. **Over-relaxation** stays parked. It scales dual steps by at most ~1.5×, and F3 shows the drift is not a pure
   step-size effect.

## 6. Step 3.6 audit (parallelism)

- **Cycle time.** From run 1's per-solve IPOPT logs (12 sampled cycles, reconciled with every recorded failure):
  - IPOPT takes a median **13.0 s** — DSO 12.2 s, TSO 0.7 s, ESSO 0.04 s — of a **33.7 s** cycle;
  - overhead is **21.5 s (59.5 %)**, stable across cycles: Pyomo model updates, NL write, `.sol` read, ADMM
    bookkeeping.
- **Speed-up.** Parallelizing the solves alone gives **≤1.5× on 8 workers** (Amdahl limit 1.55×), versus the ~4× that
  39 s → 8–10 s requires. The target is reachable only if the overhead is mostly per-block work that persistent workers
  can absorb, which existing logs cannot establish. Per-phase timing instrumentation is the first implementation item.
- **The existing parallel path is not reusable as is:**
  - DSO only, split by node rather than by block;
  - processes spawned per call, whole models pickled;
  - no thread caps;
  - FrozenSMOPF callbacks dropped;
  - IPOPT log paths can interleave;
  - `SolveProfileGuard` cannot see child-process solves.
- The P5.6-B crash was at a different layer and was later withdrawn as non-reproducible.

## 7. Step 4 warm-continuation validation (designed, not run)

- **Candidates:** L (0.9375 MVA), C\*, U (1.0 MVA), E/P = 4.
- **Arms:** cold L and U; warm L and U from C\*'s certified state; and warm C\* from certified L, as a history test.
- **Warm state:** fully hashed — consensus, all duals, proximal centres, ρ, γ, freeze state. Candidate capacities are
  recomputed; warm starts from uncertified sources are forbidden.
- **Agreement:** both runs certified **and settled** (all channel ratios < 0.9); cost within the rule-nine bar; ranking
  signs preserved, with in-bar differences reported as indeterminate; history independence at C\*.
- **Cold-audit cadence:** always cold for new and final incumbents and the runner-up; every 5th warm evaluation;
  near-ties.

## 8. Process notes

- **Concurrent commits.** Two Workers committed concurrently. One used a plain `git commit` that swept the other's staged
  files, then undid it with a soft reset. Both sets are now committed correctly and nothing was lost; the orphan commit
  is unreachable. Rule clarified: the pathspec goes on the commit command itself, and Workers never reset.
- **Corrected claim.** A diagnostics report's claim that no solve precedes cycle 1 was wrong (51 initialization solves
  run) and has been corrected.

## 9. Questions for the Expert

1. Do you agree with F2's reading, that per-agent storage duals are identified only up to the active storage-row span?
   If so, what, if anything, should dual initialization target?
2. Does F3 (identifiable share falling 0.685 → 0.520, both components drifting at a certified stop) change the view that
   initialization is the right lever for the storage channel, as opposed to accepting run 1's one-sided storage
   statement?
3. Is option 1's change — pricing the interface during the initialization solves — acceptable as a method-only change?
4. Should Step 3.6 proceed with the Addendum 17 gate not run?

## 10. Evidence index

| item | artifact | commit |
|---|---|---|
| prior handoff | `P5_15_ADDENDUM16_EXPERT_REPORT.md` | `99f61c3f` |
| record update | `REVISION_CONTEXT.md` | `db945601` |
| Step 4 design | `P5_15_STEP4_WARM_CONTINUATION_DESIGN.md` | `7f1c75bf` |
| availability audit and (a) | `p515_s36_a17_diagnostics.py`, `P515S36/A17_diagnostics/`, `WORKER_REPORT_S36_A17_DIAGNOSTICS.md` | `5e2ba6f9` |
| Step 3.6 audit | `p515_s36_parallel_audit.py`, `P515S36/parallel_audit/`, `WORKER_REPORT_S36_PARALLEL_AUDIT.md` | `710af0de` |
| dual reconstruction | `p515_s36_a17_parta_dual_reconstruction.py`, `P515S36/A17_partA_dual_reconstruction/` | `b63e4d94` |
| cycle-0 capture; replay arm | `p515_s36_cycle0_lmp_capture.py`, `P515S36/cycle0_lmp/`, harness `s35ref_replay`, `WORKER_REPORT_S36_A17_CAPTURE.md` | `b8f129e0` |
| calibration | `p515_s36_cycle0_lmp_calibration.py`, `P515S36/cycle0_lmp_calibration/`, `WORKER_REPORT_S36_LMP_CALIBRATION.md` | `b6384d99` |
| H1/H2 decomposition | `p515_s36_h1h2_dual_identifiability.py`, `P515S36/H1H2_identifiability/`, `WORKER_REPORT_S36_H1H2.md` | `97507b98` |
| decision note | `P5_15_ADDENDUM17_DECISION_NOTE.md` | `44389ad6` |

Every script above ran with `SolveProfileGuard` armed and verified: zero Pyomo/IPOPT solves, except the cycle-0 capture,
which declared and matched exactly 51. Outputs carry sha256 manifests.
