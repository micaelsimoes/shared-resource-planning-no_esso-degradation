# P5.15 Addendum 16 run 1 — the reference equilibrium (s35ref)

**Planner report, 2026-09-16.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 16 item 1. Frozen spec v5
`data/SRP1/Results/P515S35/frozen_s35_reference_spec_v5_995548ab.json` (`f76c7574`); revised pre-registered expectation
`P5_15_Z2_FLOOR_SLACK_NOTE.md` (`a993b088`, committed before launch). Launched at `a993b088`.

System cost = `gross_operational_cost`, settlements excluded; salvage 0, so gross = net. Instance C\*.

## 1. Verdict: REFERENCE ESTABLISHED — the programme's first certified convergence

**The Boyd rule stopped the run at cycle 477**, with all three channels passing on cycles **475, 476 and 477**, inside
the 500 cap, and no channel frozen at a ρ clamp. Production logged "ADMM converged in 477 iteration(s)". Exit 0, no
tracebacks, wall 17,429 s (4 h 50 min).

| | value |
|---|---|
| **system cost** | **651,039,166** |
| terminal objective step / tolerance (rule ten) | 256 / 65,104 → **0.0039** |
| **EFC/day (max across nodes)** | **1.0590** |
| network failures | 298 (284 T1, 14 T2), **0 unrecovered**, 0 local-solve failures |
| solves | 24,690 = 51 × 478 + 312 retries |
| settlement cancellation closure | 1.9e-8 |
| interface voltages at a bound | 52 of 864 within 1e-6 pu (348 within 0.005 pu) |

Scaling in force as in s34: σ fixed 9.363536e7 (computed matches to 3e-9), `al_scale_esso` 2.2721e5, `S_ref` 2.5 MVA.

## 2. A harness defect caught at evaluation, and corrected

`boyd_terminal.json` records `stopped_by: "cap"` for this run, which is **wrong**. The `s35ref` writer sets `"boyd"`
only when `converged_at_cycle` equals the last cycle, but production records the **first** cycle of the consecutive run
(475, not 477). Any stop requiring more than one consecutive converged cycle is therefore mislabelled. Earlier gates are
unaffected: s32, s33e2 and s34 genuinely hit their caps.

The committed evaluation script trusted that field, and its first output (`s35ref_evaluation.json`) reported "NOT
ESTABLISHED". It is **retained unchanged**. `s35ref_evaluation_v2.json` derives the stop from the trajectory itself
(the last three cycles all pass, before the cap), records v1's sha256 as predecessor, and states the defect. The run
artifact `boyd_terminal.json` is **not** modified. The phase-2 gate arm must not reuse the defective logic.

## 3. Settling quality: cost settled, storage stopped at the edge

| channel | terminal ratio to threshold | first all-pass cycle | cycles passing | ρ (frozen at cycle) |
|---|---|---|---|---|
| V | 0.013 | 45 | 433 | 0.01155 (11) |
| PF | 0.090 | 226 | 252 | 0.198 (backstop, 60) |
| **ESS** | **0.988** | **475** | **3** | 0.16875 (11) |

- **The system cost is settled:** rule ten 0.0039. The cost changed by 0.0012 % over the last 77 cycles
  (651.046 M at cycle 400, 651.039 M at 477).
- **The storage channel stopped at the edge of its threshold.** Its dual ratio fell steadily: 13.9 (c10), 4.32 (c100),
  2.07 (c300), 1.37 (c400), 1.10 (c450), **0.988 (c477)**. It first passed at cycle 475 and the run stopped two cycles
  later. By the repository's own rule, a channel terminating at ~99 % of its threshold has been **stopped rather than
  settled**, so the storage schedule at the stop is a tolerance-level approximation of its limit.
- **EFC/day was still creeping up at the stop:** 1.0342 (c400), 1.0542 (c450), **1.0590 (c477)**, a late slope of
  1.76e-4 per cycle. The true limit lies somewhat above 1.059.

**Path dependence observed.** Unlike s34, where balancing lowered ρ_pf twice to 0.088, ρ_pf here never moved and froze
at the backstop at 0.198. Starting ρ_ess at 0.1125 instead of 0.05 changed which ρ values the other channels locked
in. That made the PF channel slower (first pass at cycle 226 against 133 in s34) without changing the destination.

## 4. The storage equilibrium and the degradation floor

EFC/day per cohort-year (identical across nodes 5/7/9 to three decimals), against the price-taker upper bound:

| year | s35ref | price-taker (Z2) | fraction |
|---|---|---|---|
| 2025 | 1.058 | 1.1918 | 88.8 % |
| 2030 | 0.894 | 1.1888 | 75.2 % |
| 2035 | 0.682 | 0.9647 | 70.7 % |

- **The SoH floor is slack, as the revised pre-registered expectation stated.** None of 18 floor rows is active; the
  minimum terminal SoH is 0.6594 against `soh_min` = 0.50 (price-taker: 0.5892). Floor-row duals are 2.6e-10 to
  5.7e-10, which is interior-point barrier level, i.e. zero. **The floor multiplier (shadow price of degradation) at
  C\* is 0.**
- **Spec v5's original prediction — "EFC settles at the threshold with the floor active" — failed**, exactly as the Z2
  note recorded before launch. EFC/day reached 72.5 % of the 1.4612 threshold.
- **Coordinated storage use is below the price-taker bound in every year**, increasingly so in later years. That is
  expected qualitatively (the price-taker has no network constraints and sees full arbitrage value), but the size of
  the 2030 and 2035 gaps is recorded here as a result, not explained.

## 5. System cost against earlier runs

At matched cycles s35ref sits **0.16–0.46 M above s34** (e.g. +0.459 M at cycle 30, +0.157 M at cycle 150), consistent
with its slower PF channel. Its settled terminal value, **651,039,166**, is 0.14 M below s34's unsettled cycle-150
value (651,180,637). No difference bar against s34, s33e2, s32 or s31c is valid, because none of those runs settled.

## 6. Consequences for gate 3 (no action taken)

Gate 3's reproduction criteria are **well posed**, because run 1 stopped under Boyd:
- **cost:** within the rule-nine bar, the sum of the two runs' terminal objective steps; run 1 contributes 256;
- **EFC/day:** within 2 % of 1.0590, i.e. **[1.0378, 1.0802]**.

**A risk to the EFC criterion, flagged now.** Run 1 approached its storage equilibrium **from below** and stopped with
EFC still rising, at the edge of the storage threshold. A price-taker-initialized run starts **above** the equilibrium
(the LP schedule is ≈1.19) and will approach it **from above**. If each run stops as soon as the storage residual first
crosses its threshold, they can bracket the true limit from opposite sides, and their EFC values may differ by more than
2 % **while both are valid tolerance-level approximations of the same equilibrium**. The criterion is authorized as
written. If gate 3 fails on EFC alone with the cost reproduced and Boyd passed, the Planner will report the bracket and
stop for review rather than reinterpret it.

## 7. Still open for the author

1. **Success at a slack-floor candidate.** Addendum 16's target (equilibrium at the threshold, floor active) cannot occur
   at C\*: the floor is slack under both the price-taker bound and the coordinated equilibrium. What is the success
   definition here?
2. **Insight (iii).** The degradation shadow price is zero at C\*. Should it be demonstrated at a candidate or price
   profile where the floor binds?

## Evidence

`data/SRP1/Results/P515S35_REF_run/` — `g_baseline.json`, `boyd_terminal.json` (with the defective `stopped_by` field
left as written), `component_levels_terminal.json`, `interface_settlement_detail_s31c.json`,
`interface_voltage_terminal.json`, `soh_floor_sidecar_baseline.jsonl`, `recourse_jump_sidecar_baseline.jsonl`,
`ess_entry_stride_baseline.jsonl` (hash-recorded if large), `network_failures_baseline.jsonl`,
`leak_classification_baseline.jsonl`, stdout, heartbeat, ESSO pickle; `P515S35_REF_launch.log`,
`P515S35_REF_exit_code.txt`; evaluations `s35ref_evaluation.json` (v1, defective verdict, retained) and
`s35ref_evaluation_v2.json` (zero solves, guard 0/0). `evidence_manifest_sha256.json` covers the run and hashes the case
file exactly as run 1 read it. `esso_capture/` and `results/` are hash-recorded, not committed.
