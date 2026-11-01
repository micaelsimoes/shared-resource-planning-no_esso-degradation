# P5.15 Addendum 21 — arm D pre-launch note: what the policy replay predicts, recorded before D runs

Planner note, written after arm C exited and before arm D launches, as spec v10
(`data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json`, `pre_launch_evidence_for_D`) requires.
This note records expectations; it does not change spec v10's predictions for D.

## 1. Arm C certified — the run D's policy is replayed on

Arm C (τ = 0, PF balancing live, ρ_ess 0.01 fixed and exempt) **certified at cycle 191** under the Addendum 21 bar
(10 consecutive cycles with all channels inside tolerance and every local solve successful), inside the 300 cap.
PF first passed at cycle 130 (spec v10 predicted ≤ 131). The full evaluation is committed with C's evidence; the
certification verdict per arm leads the stage report after D.

**Why C needed 191 cycles rather than the predicted ~150:** the storage channel. PF and V were inside tolerance from
cycle 130 and 48, but the storage primal ratio hovered around 1 (0.93–1.20), breaking the 10-cycle streak repeatedly —
the longest run before certification was 5 cycles (131–135). This is the same hovering channel Addendum 20 recorded as
under-damped at low ρ_ess, and it is exactly what D's two-phase schedule targets.

## 2. Open-loop replay of D's policy on C's trajectory (zero solves, guard armed)

`data/SRP1/Results/P515S39/d_policy_replay_on_s39_C/`, produced by `p515_s39_d_policy_replay.py` driving the **real**
`_update_admm_penalties` with C's recorded per-cycle residuals.

| donor trajectory | predicted ESS lift cycle | first ESS action after lift |
|---|---|---|
| arm C (this run, τ = 0, PF live) | **31** | cycle 32: increase, ρ_ess 0.01 → 0.015 |
| v9 arm A (τ = 0, ρ_pf held) | 35 | at the lift cycle: increase, 0.01 → 0.015 |
| v9 arm B (τ = 1, PF live) | 35 | at the lift cycle: increase, 0.01 → 0.015 |

**Reading.** D's exemption lifts early, around cycle 31, because the ESS dual ratio falls below 1 well before the
storage walk is finished (EFC/day reaches 1.06 around cycle 25–33 in these arms). The first action raises ρ_ess by the
1.5 factor, as intended: balancing re-damps the hovering channel.

## 3. The risk this replay exposes, recorded before the run

The replay then fires **28 more times, open loop, driving ρ_ess to ≈1.3e3** — far above run 1's 0.1125. This is **not
predictive**: an open-loop replay feeds the rule residuals that cannot respond to the ρ it is choosing, and a real
increase in ρ_ess shrinks the primal residual it is reacting to. Arm B's closed-loop behaviour makes the point: its
replay also cascaded open loop, while the real run took two decreases and then held.

**But the failure mode it points at is real and specific:** if ρ_ess ratchets up cycle after cycle in closed loop, the
storage walk slows again — the precise problem Addendum 19's ρ_ess reduction solved — and D would trade a hovering
storage channel for a slow one. Two mechanisms limit this in the real run: the unchanged-streak freeze (10 cycles with
no change, after at least one action) and the absolute freeze at cycle 200.

**What the evaluation must therefore report for D, whatever the verdict:** the ESS lift cycle; every ESS action with
its cycle, direction and ρ value; the ρ_ess trajectory and its maximum; the cycle ρ_ess froze; whether the ESS clamp
was reached; and the storage primal/dual ratios around and after the lift. If D certifies, the terminal storage ratio
decides between D and C as the oracle (spec v10: D if it certifies with terminal ratio < 0.5).

## 4. Predictions unchanged

Spec v10 for D, as recorded before any of this: **D certifies, with the storage channel terminal ratio < 0.5.** Nothing
in this note revises that; the replay is evidence about the policy's first action and a recorded risk, not a new
prediction.
