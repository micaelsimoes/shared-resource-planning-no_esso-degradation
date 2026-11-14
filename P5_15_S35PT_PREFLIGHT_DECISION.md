# P5.15 s35pt preflight — Planner decision on the failed "no failures" signature

**2026-09-16, before gate 3 is launched.** Evidence: `data/SRP1/Results/P515S35/preflight_pt/preflight_verification.json`,
`network_failures_preflight.jsonl`, `data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json`,
`WORKER_REPORT_S35PT_PHASE2.md`.

## What failed
Spec v6's preflight signature reads "no failures or restorations **attributable** to the initialization". The check as
coded tested "no failures at all" and failed: one DSO block (`case33_1`, node 5, 2030 Winter) hit `maxIterations` at
cycle 2 and was recovered on the first retry (`recovered_tier1`). The declared exact solve count (153) assumed zero
retries, so the recovery made it 154 and the identity check reported false.

## Why it is not attributed to the initialization
- **Known-fragile block.** The same block failed 4 times in run 1 (`s35ref`) and 3 times in `s34`.
- **Below background rate.** Run 1 averaged 0.635 failures per cycle, s34 0.92; one failure in two cycles is below
  either. Neither run failed in cycles 1–2, but both failed within their first 10 cycles (2 and 7 failures
  respectively).
- **Every attribution-bearing signature passes**, by wide margins:
  - Z3: 864/864 cells equal the LP schedule immediately before cycle 1's first network solve;
  - Z4: non-storage pre-loop state bit-identical to the standalone construction;
  - Z7: IPOPT counts equal (51 and 51);
  - EFC at cycle 1: 1.19181 against the LP's 1.19178; change from cycle 1 to 2: 1.9e-5;
  - storage dual residual ratio: 0.35/0.39 against run 1's 19.4/17.8;
  - per-cell storage λ sum: ≤2.5e-17;
  - TSO proximal movement on storage: ~60× below run 1's.
- **No zero-failure baseline.** A signature requiring zero failures in two cycles has no zero-failure baseline in any
  run of this configuration family.

## Decision
The preflight is **accepted**. The signature is recorded as **literally not met**, with attribution not established.
It is **not** re-run in search of a pass, because the 150-cycle gate measures the failure rate properly. This is a
judgement on the preflight's reading only: **gate 3's criteria are unchanged**. The failure rate under price-taker
initialization will be reported against run 1's 0.635 per cycle.
