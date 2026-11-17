# P5.15 — The SoH floor is slack at C*: Addendum 16's success target is not reachable there

**Planner note, 2026-09-16, written and committed BEFORE run 1 (`s35ref`) is launched.** Evidence: Z2,
`p515_s35_z2_floor_slackness.py`, `data/SRP1/Results/P515S35/Z2/` (+ manifest), `WORKER_REPORT_S35_Z2.md`. Zero
Pyomo/IPOPT solves (guard 0/0); 756 scipy LPs declared and matched exactly; the committed `EFC*` benchmark reproduced
cell for cell (max abs diff 0.0). **No action taken on the target; this is for the author.**

## Finding

On the **harness's own EFC definition** (day-weighted annual average, efficiency-weighted, maximum over cohort-years),
with the capacity each block actually sees iterated to a converged fixed point, the price-taker schedule at C\* gives:

| year | EFC/day | E available (MWh) |
|---|---|---|
| 2025 | 1.1918 | 3.875 (nameplate) |
| 2030 | 1.1888 | — |
| 2035 | 0.9647 | 2.283 |

Terminal SoH **0.589209** at every node against `soh_min` = 0.50: **the floor is slack, margin +0.0892**. The floor
multiplier is **0** everywhere. The frozen threshold is confirmed independently: the constant EFC/day that makes the
floor bind by 2035 under the production chain is 1.461187 (frozen value 1.4612, difference 1.3e-5).

The price-taker schedule is an **upper bound** on storage use (uncoordinated, no degradation cost, perfect price
arbitrage). If even it leaves the floor slack, no coordinated equilibrium at C\* can reach the threshold.

An earlier independent estimate (mean ≈1.39) omitted capacity wear: available energy falls to 2.283 MWh by 2035, which
caps later-year throughput. Z2 applies that feedback.

## Consequences

1. **Addendum 16's success target — "equilibrium at the SoH threshold with the degradation constraint active" — cannot
   occur at candidate C\*.** Returned to the author; not redefined here.
2. **Run 1 remains valid.** Spec v5 (`f76c7574`) records floor activity and the floor multiplier as *reported*, not as a
   gate. Gate 3's criteria (cost within the rule-nine bar, EFC within 2 %, Boyd pass) do not reference the floor.
3. **Spec v5's pre-registered prediction "EFC settles at the threshold with the floor active" is now expected to fail.**
   Revised expectation, recorded here before launch so the pre-registration stays honest: run 1's terminal EFC/day
   approaches the price-taker value (≈1.19, maximum over cohort-years) from below, the floor stays slack, and the floor
   multiplier is zero. The s34 trajectory (0.838 at cycle 150, still rising) is consistent with that.
4. **The "floor multiplier as a result" (insight iii) is zero at C\*.** A positive shadow price of degradation would
   need a candidate or price profile under which the price-taker exceeds the threshold.

## For the author

- What should the success definition be at a candidate where the floor is slack — equilibrium at the price-taker-
  consistent storage use, with the floor-slack margin and a zero multiplier reported as the result?
- Should insight (iii), the shadow price of degradation, be demonstrated at a different candidate or price profile
  where the floor binds?
