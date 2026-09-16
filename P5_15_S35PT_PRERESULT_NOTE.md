# P5.15 gate s35pt — pre-result note: criterion (a) hinges on one ρ_pf balancing event

**Planner note, 2026-09-16, committed while gate 3 is running (cycle ~3 of 150), BEFORE its result is known.**

## Observation from the first three cycles
- **The storage walk is gone.** Storage-channel ratios over cycles 1–3 are primal 1.35 / 0.92 / 1.16 and dual
  0.35 / 0.39 / 0.57, against run 1's primal 120 / 66 and dual 19.4 / 17.8.
- **V and PF track run 1 almost exactly.** PF primal ratio is 2,516 / 1,980 / 1,651 against run 1's
  2,514 / 1,980 / 1,650, and ρ_pf is again held at 0.198.

## Why this matters for criterion (a) (Boyd stop within 150)
From the committed trajectories:

| | ρ_pf path | PF dual ratio at cycle 150 | PF first passes |
|---|---|---|---|
| run 1 (`s35ref`) | **never changed**; frozen at the cycle-60 backstop at 0.198 | **2.728** | cycle **226** |
| `s34` | lowered at cycles **57 and 58** (0.198 → 0.088), just before the backstop | — | cycle **133** |

Once storage converges early, **PF becomes the binding channel**, and its speed is set almost entirely by whether
balancing lowers ρ_pf before the cycle-60 freeze.

## Prediction, recorded before the result
- **If no ρ_pf decrease occurs before cycle 60:** expect PF's dual ratio near run 1's ≈2.7 at the 150 cap, and gate 3
  **fails criterion (a) because of the PF channel**, independent of how well the price-taker initialization performs.
- **If a decrease occurs** (as in `s34`): PF may pass near cycle ~130, and (a) can pass.
- Whether the decrease happens is **genuinely uncertain**. PF interface flows carry storage from cycle 1 here, so the PF
  residual balance near cycles 57–58 can differ from run 1's even though cycles 1–3 match closely.

## Consequence for interpretation
A failure of (a) on the PF channel alone would be a finding about the **ρ policy** (freeze-at-backstop locking ρ_pf at a
slow value), not about the initialization. It will be reported that way, with the storage-channel evidence stated
separately, and the Planner will stop for review. It will **not** be reinterpreted as a pass, and no criterion will be
altered.
