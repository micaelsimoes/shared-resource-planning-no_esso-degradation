# P5.15 Step 3.4 (+3.3(b)) — the re-scaling gate (s34)

**Planner report, 2026-09-16.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 15. Frozen spec v4
`data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json` (`9f182b79`), superseding v3. Implementation `ef2040d7`
(production) and `db8eb90a` (harness). **Stops for review.**

System cost = `gross_operational_cost`, settlements excluded; salvage 0, so gross = net. Instance C\*.

## 1. Verdict

**FAIL on the pass test: 150 cycles, cap, no cycle with all three channels passing.** No channel was frozen at a ρ
clamp, so the failure is the stopping criterion itself, not a guard condition.

**But this is by a wide margin the closest the programme has come, and the Addendum 15 mechanism is supported.**

| | s32 (fixed γ) | s33e2 (γ = τρ) | **s34 (re-scaled)** |
|---|---|---|---|
| channels passing at termination | 0 | 1 (V) | **2 (V and PF)** |
| cycles with ≥2 channels passing | 0 | 0 | **18** |
| ESS dual ratio (binding) | 5.56 | 3.245 | **2.843** |
| EFC/day | 0.1038 | 0.0671 | **0.8382** |
| network failures | 17 | 18 | **138** (133 T1, 5 T2, 0 unrecovered) |
| system cost (M) | 654.41 | 653.83 | **651.18** |

Run facts: exit 0, wall 5,914 s, 7,844 solves = 51×151 + 143 retries, 0 local-solve failures, objective step 4,682
against tolerance 65,119 (rule ten 0.072), settlement cancellation closure −2.8e-7, 52 of 864 terminal interface
voltages within 1e-6 pu of a bound (330 within 0.005 pu). Scaling in force: σ fixed 9.363536e7 with the computed value
matching to 3e-9, `al_scale_esso` = 2.2721e5, `S_ref` = 2.5 MVA. The preflight **detector gate passed**: the barrier-leak
indicators did not worsen under the new storage-operator scaling (complementarity ratio and spurious throughput were
*lower* at cycle 2).

## 2. Per channel

| channel | primal r/ε_pri | dual s/ε_dual | first pass | cycles passing | ρ (frozen at) |
|---|---|---|---|---|---|
| V | **0.045** ✓ | **0.154** ✓ | 45 | 106 | 0.01155 (cycle 11) |
| PF | **0.476** ✓ | **0.726** ✓ | 133 | 18 | 0.088 (backstop, cycle 60) |
| ESS | **0.347** ✓ | **2.843** ✗ | — | 0 | 0.1125 (cycle 12) |

Balancing **raised** ρ_v (0.0077 → 0.01155) and ρ_ess (0.05 → 0.1125) from their low starts and lowered ρ_pf twice
(0.198 → 0.088); the "10 unchanged cycles" rule froze V and ESS early (cycles 11 and 12), and PF hit the cycle-60
backstop. No clamp freezes.

## 3. The storage channel: released, but not yet arrived

- **EFC/day rose 0.00029 → 0.8382**, increasing in **100 %** of cycles: 0.165 (c5), 0.433 (c20), 0.594 (c50), 0.701
  (c90), 0.838 (c150). That is **12.5 ×** the s33e2 terminal, **45 %** of the price-taker benchmark `EFC*` = 1.8520 and
  **57 %** of the SoH-binding threshold 1.4612.
- **Still rising linearly at the cap:** slope 0.00215/cycle over cycles 120–150 (0.00228 over 90–150), which
  extrapolates to ~**289 cycles** from 150 to the SoH threshold and ~471 to `EFC*`. Extrapolation is descriptive.
- **The dual residual is decaying, slowly:** 3.462 (c90) → 3.073 (c120) → 2.843 (c150), ×0.925 per 30 cycles, implying
  ~**403 more cycles** to reach 1.0 (~318 on the 90–150 window).
- **Spec v4's falsifier did not fire.** The per-entry step is **5.14 ×** s33e2's (0.219e-3 vs 4.26e-5) against a ×3
  threshold, so the conjunction "rise < 3× **and** cosine persisting" fails on its first clause: **the stiffness reading
  is not falsified.** The step norm decays slowly (0.225 at c10 → 0.0455 at c150, −7.5 % over the last 30 cycles) while
  the direction stays persistent (cos 0.9998 at c150; mean 0.9985 over the last 30). Storage is still **walking toward**
  its equilibrium, not oscillating about it.

**Reading.** Addendum 15 predicted that releasing the σ/AL stiffness would let storage move to its arbitrage
equilibrium within ~50 cycles and EFC/day reach order 1. The direction is confirmed and the magnitude is most of the
way there (0.84 against an `EFC*` of 1.85, from 0.067), but the **timescale was optimistic by roughly an order of
magnitude**: at the observed rates the equilibrium is a few hundred cycles away, and the storage dual residual cannot
pass the Boyd test until the underlying gradient is largely spent.

## 4. What the S_ref prediction can and cannot say

Spec v4 recorded that `S_ref` = 2.5 should multiply the ESS dual ratio by ≈2.58 **at matched conditions**. This gate
cannot test it: ρ, the ESSO scaling, the normalization and the trajectory all changed together, and the terminal ratio
(2.843 against s33e2's 3.245) compares two different dynamics. **Recorded as untested**, per the Advisor's warning that
a combined gate cannot isolate this multiplier.

## 5. The cost of a low ρ

**138 network failures against 18 in s33e2** — a 7.7 × increase, concentrated in the distribution blocks, every one
`maxIterations`, all recovered (133 first-tier, 5 second-tier), none unrecovered, and zero failures reached the ADMM
loop. This is the conditioning risk that was flagged before the run: the ρ/2 term is the main convexifier of the local
nonlinear problems, and starting ρ low costs one extra solve per failure (143 retries, 1.8 % of solves). It did not
corrupt the run, but it bounds how much lower ρ can sensibly start.

## 6. Decisions for review (no action taken)

1. **The stopping criterion is not reachable for storage within 150 cycles at these rates.** Options: (a) one run at a
   higher cap (~400–500) to let the storage gradient spend itself — cost ≈4 hours at the observed 39 s/cycle;
   (b) accept a per-channel verdict, record storage as non-certifiable at ε_abs 1e-5 while its arbitrage gradient is
   live, and proceed to 3.5; (c) revisit whether the ESS channel's ε should be derived differently now that its
   normalization is candidate-independent. The Planner recommends **(a) once**, because every rate in §3 is now
   measured rather than assumed, and a single longer run converts an extrapolation into a result.
2. **D5 form.** Implemented as an argmin-identical scaling of the storage operator's coordination terms rather than a
   division of its base objective, to protect the feasibility-slack magnitude and the barrier-leak identity that EFC
   depends on. The equivalence is verified numerically (`al_scale × literal = production`, to 1e-9). **Confirmation
   requested.**
3. **ρ floor.** Given §5, should ρ_ess start at 0.05 again in any re-run, or at the 0.1125 the balancing itself chose?
4. **EFC\* versus the SoH threshold.** `EFC*` = 1.852 exceeds the 1.4612 threshold, so the equilibrium should be
   expected **at the threshold with the SoH constraint active**, not at `EFC*`. If a longer run settles near 1.46, that
   is the success case and should be reported as such.

## Evidence

`data/SRP1/Results/P515S34_run/` — `g_baseline.json`, `boyd_terminal.json`, `component_levels_terminal.json`,
`interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json`, `ess_entry_stride_baseline.jsonl`,
`recourse_jump_sidecar_baseline.jsonl`, `network_failures_baseline.jsonl`, `leak_classification_baseline.jsonl`,
stdout, heartbeat, ESSO pickle; `P515S34_launch.log`, `P515S34_exit_code.txt`; analyses `s34_evaluation.json` and
`s34_closeout_analysis.json` (both zero-solve, guards armed, 0/0). Benchmarks: `P515S34/EFC_benchmark/` (`d1b8cf9c`).
`evidence_manifest_sha256.json` covers the run; `esso_capture/` and `results/` are hash-recorded, not committed.
