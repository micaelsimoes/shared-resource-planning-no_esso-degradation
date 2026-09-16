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

## 6. Correction (same day, after independent review and Planner verification)

Two claims in §3 are **withdrawn or qualified**, and the mechanism is restated.

- **Withdrawn: "row 3′ moved where the optimum sits."** That clause compared *transients*. All four post-table runs
  hit their caps, and within `s33e2` the per-node storage throughput rose 0.0154 → 2.219 (≈144×) while still moving in
  a perfectly persistent direction (cos = 1.000). **The 0.067 is a point on a trajectory, not an equilibrium**, and
  s31's 1.2110 is equally a transient of a diverging run. Nothing in the archaeology fixes where either optimum lies.
- **Qualified: the 47× ratio compares unlike objects.** The order-1 family (G1, G1B/s30, G3F_B, ablation C) are
  *converged* points of an objective that contained the row-3 charge; the post-table runs are *capped transients* of
  the corrected objective. The dating of the change to s31c stands; the magnitude does not carry economic meaning.
- **Mechanism restated — price deletion, verified in code.** Before s31c the only price on interface energy movement
  sat in row 3: `flexibility_cost` charged `cost_flex · baseMVA · (flex_p_down + flex_q_down)` on the TSO's
  ADN-interface loads (`model_construction_helpers.py:1692–1712`, gate `load_is_tso_adn_interface` at `:1685–1689`).
  The DSO's own import is its reference generator, which is **excluded from `generation_cost`** for distribution
  networks (`:1663–1670`), so the DSO priced interface energy at zero. The shared ESS enters the **TSO node balance**
  (`shared_es_pnet`, `:1329`), so TSO-side storage displaced exactly the volume that charge was levied on. Removing
  row 3 therefore **deleted the implicit remuneration for storage**, before any question of substitution by δ. Two
  qualifications the code imposes: the charge was **one-sided** (down legs only), and it was judged a transfer payment
  because it had no DSO receipt term — which is why it was removed and must not simply be restored.
- **The strong substitution reading is not supported.** If free δ had made storage worthless, the storage gradient
  would have vanished; instead `s33e2` shows a constant, price-aligned gradient (cos −0.516 against the within-day
  price deviation) pointing toward *more* cycling. δ may reduce storage's value; it has not eliminated it.
- **Net position.** Coordination stiffness is supported as a **rate** mechanism (ρ-invariant residual; ~14,900 cycles
  to rating). Neither stiffness nor row 3′ is established as setting the **destination**. The destination is currently
  unmeasured, which is why the price-taker benchmark `EFC*` is being computed before the 3.4 gate reads EFC/day
  against any target.

## 7. Two measurements that close the open questions (2026-09-16)

**Z4 — the oscillation is a decaying transient, and my earlier description of it was wrong.**
Commit `a3eca693`, zero solves. The oscillating quantity is `gross_operational_cost`, with period **~20 cycles**
(convergent across DFT, autocorrelation, peak spacing and zero crossings), and a **decaying** envelope: exponential fit
R² = 0.84–0.87, half-life ≈ 20 cycles, amplitude falling from 3.7 × tolerance at cycle 34 to 0.04–0.09 × at cycles
138–143. **Correction to this note and to `P5_15_S33_E2_GATE_REPORT.md` §4:** the "185,571 range = 2.8 × tolerance over
cycles 101–150" is the **raw span, which is dominated by the residual trend**; the *detrended* oscillation in that
window is only 0.09–0.38 × tolerance. The honest statement is therefore: late in `s33e2` the cost is slowly descending
with a decaying ~20-cycle oscillation superimposed — neither "still descending" without qualification, nor "oscillating
by 2.8 × tolerance". The unidentifiability of the ESS cost contribution (§4 of the gate report) is unaffected.
The mode is **not** caused by the cycle-30 ρ/γ freeze: it appears in `s32`, which has no freeze, at 2–3 × the relative
amplitude. No robust coupling to the 18 network failures (permutation p = 0.035 / 0.622 across two windows, not
significant after accounting for both). V and ESS dual residuals co-move with the cost step; PF does not.
**Capture gap found:** the `[RECOURSE JUMP]` block decomposition stops at cycle 62, so block-level localisation is
unrecoverable for the late window. Future gates must capture it unconditionally.

**EFC\* — the price-taker benchmark, which replaces "EFC/day was 1.1 before".**
Commit `d1b8cf9c`, zero production solves (36 LPs in scipy, guard armed at zero Pyomo/IPOPT entries; prices
cross-checked bitwise against the run's own serialized settlement detail, max abs diff 0.0 over 864 cells).
For C\* (s = 0.96875 MVA, e = 3.875 MWh): **`EFC*` = 1.8520 for 2025** (max across nodes 5/7/9; 0.7964–1.8520 across
the twelve year/day cells; 2.1500–2.4750 with round-trip efficiency forced to 1.0). Every cell binds **both** the
converter rating and the SoC range, a consequence of C\*'s ≈3.3 h energy-to-power ratio against a 24-period day.
Consequences:
- **"Order 1" is a defensible derived target**, and is if anything conservative: `EFC*` exceeds the retired 1.1 figure.
  The gate's storage diagnostic is read against `EFC*`, never against 1.1. Today's 0.067 is **3.6 %** of it.
- **But `EFC*` > the degradation-binding threshold 1.4612**, so a price-taker would want to cycle past the point where
  the SoH constraint binds. The achievable equilibrium under the full formulation is therefore expected to sit **near
  the threshold, not at 1.85** — and if storage settles near 1.46 with the SoH constraint active, that is success, not
  a shortfall.
- The `EFC*_row3` variant was **skipped, correctly**: no committed algebraic mapping exists from the row-3 interface
  charge onto the storage's own dispatch, and inventing one would be an approximation. The price-deletion mechanism in
  §6 therefore stands on the code reading, not on a counterfactual number.

## Evidence

`data/SRP1/Results/{P515G1,P515G1B,P515S30,P515G3F_B,P515A,P515S31_run,P515S31C_run,P515S32_run,P515S33_E2_run}/`
— `g_*.json` (`esso_capture[node].efc_per_day_max`), `component_levels_terminal.json` (`totals_weighted`),
`interface_settlement_detail_s31c.json` (`flexibility_volumes_per_dso`); `definitions.py:60–74`;
`P5_15_S31_BASELINE_REPORT.md`, `P5_15_S31C_ECONOMIC_BASELINE_REPORT.md`, `P5_15_S33_E2_GATE_REPORT.md`.
