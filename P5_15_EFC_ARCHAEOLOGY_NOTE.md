# P5.15 — When did shared-storage utilisation collapse, and what changed with it?

**Planner note, 2026-09-16.** Zero solves; every figure read from committed artifacts. Written because Addendum 15's
gate prediction ("EFC/day rises from 0.067 to order 1") and its governing reading ("every prior storage-effect figure
was measured with the storage immobilised by coordination stiffness", `REVISION_CONTEXT.md`, `b63cc3de`) both depend on
when, and why, storage utilisation fell. **No action taken; this is evidence for the author.**

EFC/day = throughput / (2·E_rated), maximum across nodes 5/7/9, as computed by the campaign harness.

## 1. The collapse dates to s31c, not to the scaling

| run | cycles | converged | EFC/day | what changed |
|---|---|---|---|---|
| G1 / G4 / G2R | 71–72 | yes | 0.9725 | pre-Step-3.1 baseline |
| G1B, s30 | 71 | yes | 1.1029 | new production baseline (ε = 1e-5) |
| G3F_B | 80 | yes | 1.3398 | per-node investment arm |
| ablation C | — | yes | 1.102–1.112 | ε ablation; the 1.112 cited in `definitions.py:70–71` |
| **s31** | 90 | **no** | **1.2110** | signed penalty table; row 3 removed |
| **s31c** | 90 | **no** | **0.0285** | **row 3′: interface settlement at π_t + signed δ** |
| s32 | 150 | no | 0.1038 | Boyd stop, fixed γ |
| s33e2 | 150 | no | 0.0671 | γ = τρ, freeze at 30, 3 consecutive cycles |

**σ, `effective_scale`, the undivided AL terms and the unscaled ESSO objective are identical in every row of that
table.** The stiffness that Addendum 15 identifies is real and is visible in the s33e2 drift, but it is *constant*
across runs whose storage utilisation differs by a factor of 47. It therefore cannot, by itself, explain the change at
s31c. Nor does non-convergence explain it: s31 also hit its cap, with recourse climbing, and still cycled storage at
EFC/day 1.21.

## 2. What accompanied the collapse

Terminal cost composition, block-weighted, millions:

| run | generation | DSO-internal flexibility | row-3 charge (definitional) | ESS usage cost | gross |
|---|---|---|---|---|---|
| s31 | 308.4 | 209.4 | **2,648.5** | 0.0 | 517.8 |
| s31c | 472.6 | 188.8 | 0.0 | 0.0 | 661.4 |
| s32 | 450.7 | 203.7 | 0.0 | 0.0 | 654.4 |
| s33e2 | 446.7 | 207.1 | 0.0 | 0.0 | 653.8 |

Interface flexibility volumes (Σ|δ_P| per node, MW-periods) exist only from s31c, because row 3′ introduced the signed
δ reparametrization: s31c {5: 6151, 7: 4251, 9: 6671}; s32 {6031, 5482, 6965}; s33e2 {5958, 5871, 6988}; max |δ_P|
0.72–0.85 pu against interface ratings 2.0 / 1.0 / 1.5 pu.

So at exactly the point where storage cycling collapsed, the TSO gained a **signed interface flexibility lever that is
not priced as flexibility** — only its energy is settled at π_t — and generation cost rose by 140–165 M while
DSO-internal flexibility stayed roughly flat. Shared-ESS usage is priced at zero throughout (row 8).

## 3. Two readings, and what separates them

- **R1 — substitution.** Unpriced interface flexibility substitutes for storage cycling. Storage and δ are alternative
  ways to move energy across periods at the interface; δ is free at the margin, so it is used first. Under this reading
  low EFC/day may be the *correct* answer to the current pricing, not a symptom, and the 3.4 re-scaling will not
  restore order-1 utilisation.
- **R2 — stiffness (Addendum 15).** Coordination stiffness pins the storage schedule near its starting point; the
  s33e2 drift is the unrelieved gradient, and re-scaling will release it.

The evidence above does not decide between them, but it constrains both: R2 alone cannot explain the s31c step, and R1
alone does not explain the constant-speed drift that s33e2 measured. They are compatible: **stiffness sets the rate at
which storage can move; row 3′ moved where the optimum sits.** That is the Planner's current reading.

**What would separate them inside the 3.4 gate:** if re-scaling releases the drift and storage settles at a *low*
utilisation, R1 is supported and the EFC target is wrong. If it settles at order-1 utilisation, R2 is supported and
the target is right. Either way the *rate* question (stiffness) and the *destination* question (pricing) are answered
separately, which the current pass test conflates.

## 4. Caveat on the order-1 reference values

The order-1 EFC/day figures come from runs that either predate the penalty table or carry the row-3 charge, which at
s31's terminal point was 2,648.5 M against a 517.8 M recourse — the charge that Addendum 11 established as a transfer
payment and 99.8 % of the prior recourse. **Storage utilisation measured under that objective is not a clean economic
reference.** "EFC/day was 1.1 before" is therefore not, by itself, a target for the corrected formulation.

## 5. For the author (no action taken)

1. Should the 3.4 gate's pass test remain "Boyd stop **and** EFC/day → O(1)", or should EFC/day be reported as a
   diagnostic with this confound stated, leaving the Boyd stop as the pass test?
2. Is unpriced signed interface flexibility (δ) intended to be the system's cheapest inter-temporal lever? Row 3 priced
   it and was removed as a transfer; row 3′ prices only its energy. If storage is meant to compete with it, the
   comparison is between a priced-at-π_t energy exchange and an unpriced flexibility volume.
3. The manuscript statement that prior storage-effect figures were measured with immobilised storage should be widened:
   they were also measured under a formulation in which the interface flexibility charge, not storage economics,
   dominated the objective.

## Evidence

`data/SRP1/Results/{P515G1,P515G1B,P515S30,P515G3F_B,P515A,P515S31_run,P515S31C_run,P515S32_run,P515S33_E2_run}/`
— `g_*.json` (`esso_capture[node].efc_per_day_max`), `component_levels_terminal.json` (`totals_weighted`),
`interface_settlement_detail_s31c.json` (`flexibility_volumes_per_dso`); `definitions.py:60–74`;
`P5_15_S31_BASELINE_REPORT.md`, `P5_15_S31C_ECONOMIC_BASELINE_REPORT.md`, `P5_15_S33_E2_GATE_REPORT.md`.
