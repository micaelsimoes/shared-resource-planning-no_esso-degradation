# P5.15 Addendum 21 — s39 preflight: provenance re-run and the corrected bitwise reading

Planner note, written before arm C launches. Spec: `data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json`.

## 1. Why the preflights were re-run

The first C and D preflights ran at 18:36 and 18:38 against **uncommitted** code. The preparation Worker's commits
(`6d628d7e` … `d04ff043`) landed at 19:44–19:46, after further edits. That evidence therefore cannot be tied to a
commit, so it is **retained, not deleted, and not cited as the launch gate**.

The preflights were re-run at committed HEAD (`7f660765`) into their own output roots and working-dir ids
(`preflight_C_v2`, `preflight_D_v2`; ids suffixed `preflight_v2`), so no harness was re-run onto a cited artifact. The
suffix mechanism is the only change (`p515_s39_preflight.py` optional second argument; `run_s39_arm(mode_label_override=)`,
refused unless a smoke-test output root is given). No production, numerical or configuration change.

## 2. The bitwise item's premise was wrong — the Planner's error, not the Worker's

The instruction said arms C and D should reproduce v9 arm A's cycles 1–2 bitwise because "A differs only in PF exemption
and the 3-consecutive stop, neither of which can act in cycles 1–2".

**PF balancing does act at cycle 2 in C and D.** PF is live in both arms and exempt in A. At τ = 0 the PF dual ratio is
large relative to the primal from the first cycles (cycle 2: primal 361, dual 2,076, a ratio of 5.7 against a decrease
dead band of 5), so balancing lowers ρ_pf **0.198 → 0.132 at cycle 2**. In arm B (τ = 1) the same rule first fired at
cycle 79, which is why the premise looked safe.

The Worker reported this rather than adjusting the comparator to force a pass. That was the correct call.

## 3. What the re-run actually shows

Of 192 compared per-cycle fields (excluding 4 fields the comparator already excluded by configuration):

| arm | cycle 1 mismatches | cycle 2 mismatches |
|---|---|---|
| C | 2: `rho_frozen_pf`, `rho_unchanged_streak_pf` | 2: `rho_frozen_pf`, `rho_pf_after` (0.132 vs 0.198) |
| D | 3: the above plus `rho_frozen_ess` | 3: the above plus `rho_pf_after` |

**Every other field matches arm A exactly** — all residuals and ratios per channel, consensus norms, costs, EFC, γ, ρ_v,
ρ_ess. The mismatches are (i) the genuine ρ_pf action, and (ii) balancing bookkeeping flags that differ because a live
channel keeps a freeze streak where an exempt channel does not, and because arm D's conditional exemption uses its own
state fields (`exempt_until_streak` / `_lifted` / `_lift_cycle`) rather than the static `exempt` flag.

**Reading:** the preflight passes on the comparison that carries information — the numerics. The bitwise item is
recorded as **not applicable as written**; it is not a defect in the arms.

**`balancing_exempt_ess` reads `False` for arm D while the ESS channel is genuinely exempt.** This is accepted as
designed: the action label is `exempt (fixed)` and the dedicated per-cycle sidecar
`ess_exempt_until_state_s39_D.jsonl` carries streak, lifted flag and lift cycle, which is what the evaluator and the
report use. The boolean means "statically exempt", and the report must not read it as "not exempt".

## 4. Everything else the re-run verified, at committed HEAD

Per arm: solve-profile identity exact (153 = 51 × 3, no recovery retries); γ = 0 on all channels every cycle; V and PF
labels live; ESS `exempt (fixed)`; D's exemption not lifted by cycle 2 (streak 0, as its dual ratio has not yet gone
below 1); `minimum_consecutive_converged_cycles` = 10 in force; per-entry PF capture identity holds every cycle
(max relative error 2.7e-16); all capture paths populated, including the new exempt-state sidecar; evaluator dry run
returns 0; zero local-solve failures.

## 5. Consequence to watch in arm C and D

ρ_pf starts falling at cycle 2 rather than cycle 79. The zero-solve replay of the balancing rule showed that, open loop,
repeated decreases can walk ρ down to the 1e-4 clamp, where the "decreased" label prevents the unchanged-streak freeze.
That did not happen in arm B, which had two decreases and then held. **The evaluation must report the ρ_pf trajectory,
whether the clamp was reached, and the freeze cycle.** The clamp flag is reported, not gated, under the Addendum 21 bar.
