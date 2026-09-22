# P5.15 Addenda 32–34: memory route decided, price mechanism identified, flexibility break-even measured. Stop for review

**Planner report, 2026-09-22.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 29, 32, 33, 34.
- **Frozen specs:** v18 `data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json` (`e904955b`), v19 `data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json` (`f62fc73d`). All predictions were recorded in them before the runs.
- **Baseline:** C2 (k = 35,851) + φ_cal 0.985 + soh_min 0.70. The C3 results remain the sensitivity set and appear in no baseline table.
- **Objective convention on every table:** Q = certified `gross_operational_cost`, settlement excluded; F = I + Q; value = Q(0) − Q(x); salvage and the settlement remainder reported separately.
- **Status:** nothing is running.

## 1. Verdict

1. **The paper-scale decision rule fails, on cycle time, not on memory.** The memory fix works: the footprint after initialization is **24.78 GiB** against the 26 GiB threshold, where the previous attempt had reached ~36 GiB and begun swapping. But one ADMM cycle takes **42.3 minutes** against the 25-minute limit. **The route is therefore the ≥ 64 GiB machine or the SRP1-with-caveat fallback** — and note that a larger machine does not fix a CPU-bound cycle.
2. **The storage's price signal is set by DSO load-shifting, and that is now proved.** In the hours that matter, every transmission generator is at zero output and the bus-7 price is the DSOs' own daily flexibility shadow price. This reproduces the −12.3 €/MWh flatness term exactly (residual 6e-7).
3. **The flexibility price is a real break-even axis: the smallest unit pays determinately at ×2**, by ten times the resolution. Break-even lies between ×1.5 and ×2, not between ×2 and ×3 as predicted, and the predicted value ceiling is exceeded.
4. **No defect was found behind the negative penalty levels.** They are IPOPT's bound-relaxation residue; absolute Q is biased low by ≤ 2 × 10⁻⁵ and value differences by ≤ 181 €.
5. **NOMAD is installed and the in-house direction/mesh machinery is verified against it**, in a separate environment; the canonical environment is provably untouched.

## 2. Memory task (Addendum 29; `4cf5666c`, `f5c2d87c`, `01d6d01b`, `78a9b230`)

**What the growth was.** Profiling one paper-scale DSO block and one SRP1 block, in 44 fresh processes with 5 samples per checkpoint (noise ≤ 0.6 MiB within a checkpoint):

| classification | per paper DSO block | evidence |
|---|---|---|
| **one-time bookkeeping (fixable)** | ≈ 140 MiB | Pyomo's `ModelSolution` + `SolverResults.solution`, which nothing reads afterwards; clearing returns 136–143 MiB and makes the third solve flat |
| inherent: suffix data (duals, zL/zU) | ≈ 137 MiB | needed by the warm start |
| inherent: warm-start multiplier copies | ≈ 45–58 MiB | zL_in / zU_in duplicate the out suffixes |
| **leak** | **none** | slope 0.02 MiB/solve over solves 3–10; live objects +170/solve, matching the harness's own record |

**The fix** is a switch, `release_solution_bookkeeping`, **off by default** (with it on, snapshot pickles would differ byte-wise even though no number does). Values, suffixes and warm starts are untouched. **Gate: two SRP1 cycles at C\* reproduce the committed baseline trajectory bitwise**, 0 differences over 29 fields, with 144 clears executed.

**The paper-scale re-measure** (`--release-solution-bookkeeping`, snapshots off, machine alone), against the rule as I read it **before** the run (`67dd7d71`: "resident" = macOS `phys_footprint`, because `rss_tree` understates demand under compression — in the previous run it read 12 GiB while demand was 36 GiB):

| criterion | measured | verdict |
|---|---|---|
| footprint after initialization ≤ 26 GiB | **24.78 GiB** (predicted 27 ± 1.5) | **met** |
| one cycle ≤ 25 min | **2,535 s = 42.3 min** | **failed** |
| flat across cycles | +4.4 GiB during cycle 1 (peak 29.2 GiB); swap flat at 0.93 GB | **not established** — one cycle cannot show flatness |

- **Build 296 s; initialization 1,565 s.** Initialization completed for the first time.
- **Extrapolation:** ≈ 78 h per certified evaluation (≈ 110 cycles), so ≈ 10 days for the three planned evaluations, machine dedicated.
- **σ calibration passed at paper scale for the first time:** computed σ is 0.597 of the fixed value, inside the factor-3 band.
- **One DSO block (node 7, 2025 Autumn) failed unrecovered in cycle 1**, after tier-1 and tier-2 retries.
- **The child exits 1 for an accounting reason, not a crash:** the scale script's solve identity credits retries only to *recovered* blocks, so 168 observed against 166 declared, and the armed guard correctly refused. The ladder harness already counts per failure event instead; the scale script should be aligned (small task, not yet done).

## 3. Addendum 32 items

**Q6 — negative penalty levels** (`f67ec7d4`): **neither of the two categories posed.** Every slack has a correct lower bound of 0 and the report applies no sign convention.
- They are **IPOPT's bound-relaxation residue** (default 1e-8): slack values reach −9.9e-9 and are summed unclipped.
- **Absolute Q** is biased low by ≤ 1.98 × 10⁻⁵ (≈ 12.8k € on 653M at C\*).
- **Value differences** change by ≤ 181 € (≤ 5.9 % of a bar); at the baseline unit, −19 € against a value of 259,428.
- A third category is adopted for the classification: *solver bound-relaxation residue, unclipped*.
- **This confirms an earlier statement** (`P5_15_S31_BASELINE_REPORT.md`) against the models; it is not a new discovery.
- Eight positive levels in 4 records, all in blocks whose terminal solve was a recovery retry, are the largest single contributor (+181 €). Deferred.

**Q4 — terminal TSO capture** (`c23a6cd8`, ruling `8f9d67f2`, analysis `dd0a86b3`, follow-up `d68c814d`):
- **The x = 0 evaluation reproduced A0 bitwise** on every solved quantity. 2,425 flagged fields are the declared ageing parameters echoed into diagnostics (soh_min, k, the calendar term = 1 − 0.985⁵, the ESSO pickle size). Strict gate FAIL stands in its output; **ruled PASS** with the analysis recorded.
- **What sets the bus-7 price in the peak hours:**
  - The DSOs cut imports sharply when the market price is high.
  - The TN's remaining load is covered by its own zero-cost wind and PV, and all conventional units go to zero output — the supply curve's vertical step.
  - No TN generator is marginal, so the price is what the DN side will pay, carried in by the ADMM interface consensus dual.
  - **That DN-side value is the dual of the daily flexibility energy-balance row**, μ: a daily constant per load. Where P-down is marginal the price is μ + the hourly flexibility price; where P-up is marginal it is μ. Verified hour by hour against the TSO price to ≤ 0.078 €/MWh over all 270 affected hours, with all KKT residuals ≤ 3.4e-9.
  - **Ruled out:** no DN-local storage exists; DN curtailment never sets the price; "off-peak price + flexibility price" is off by 34 €/MWh on average.
- **The flatness term reproduces exactly** (residual 6.1e-7 against W25's −12.326 €/MWh per cycle):

| term | €/MWh per cycle |
|---|---|
| peak hours capped by DSO shifting | −6.68 |
| hour reselection from the same cause | −5.25 |
| losses | −0.39 |
| voltage, interface voltage | −0.02 |

- **Prediction:** partly confirmed. LMP7 < market price in the peak hours as predicted, but **not** because of curtailed TN renewables or a binding TN limit (congestion is 0 everywhere).

**Q3 — NOMAD** (`2c1272f4`): **done.** PyNomadBBO 4.6.0 installed from a prebuilt arm64 wheel into a separate venv; canonical `pip freeze` hash identical before and after, and `import PyNomad` still fails there.
- **Verified equal on 144 logged polls:** the Householder construction, the rounding to the lattice, bound handling, the (n+1)-th direction and the rank-reduction second pass.
- **Differs by design:** NOMAD draws random directions (ours uses Halton); 1-2-5 frame ladder (ours doubles/halves); snapping vs rejection at bounds; no unit-poll termination; no completion step.
- **Error found in STEP4 §5.2:** its "double/halve … as implemented in NOMAD 4" is wrong; NOMAD 4.6.0 rounds 4 up to 5 and uses 1-2-5. **For the author to correct.**
- **Manuscript:** describe the in-house OrthoMADS with Halton directions, citing NOMAD 4 for the granular rounding machinery now verified against it.

**Q1 — citations.** Spec v17 is frozen and is **not** edited in place; the successor spec v18 carries the citations block, marked *proposed by the expert, author to confirm*: EVE MB31 314 Ah (8,000 cycles to 70 % → k = 22,430, C4's k exactly, and the 0.70 floor), Hithium 314 Ah (13,000+ to 65 % → k ≈ 30,200), with φ_cal = 0.985 as a model assumption. **The Planner has not accessed these sources**; they are recorded as stated in Addendum 32.

## 4. Addenda 33–34: what the storage competes with

**Flexibility price and the congestion/shifting split** (`a6adf0d8`):
- **The applied `cost_flex` profile** is hourly, 21.3–92.7 €/MWh, day-weighted mean 48.4, growing 2 %/year; it is about 51–55 at night and 25–30 at midday.
- **P and Q flexibility carry the same price**, but **reactive flexibility is structurally absent**: its bounds are fixed at 2e-5 p.u. (`network.py:1276-1277`) and the Q day-balance row is not wired. Addendum 34's premise holds in the objective only.
- **The DSOs buy no downward flexibility in the node-7 binding hours.** Their cost there is zero within the solver's bound relaxation. What moves is load shifted *into* those hours with **unpriced** P-up. So under Addendum 33's definition, **congestion relief is 0 and all flexibility cost is economic shifting** — the dichotomy captures nothing here, and I propose the author rule on whether to redefine it (pairing the priced down-leg with the unpriced shifting).

**P/Q split at the smallest unit** (baseline):

| component | EUR | share |
|---|---|---|
| active-power dispatch | 252,423 | 97.3 % |
| reactive dispatch | 309 | 0.1 % |
| remainder | 6,696 | 2.6 % |

- **Q dispatch** is not idle: the storage absorbs in all 288 slots, up to 8.6 % of rating; in 43 slots the converter is at ≥ 99 % loading, so P and Q share the circle.
- **The 1.1 pu bound never binds at bus 7** (closest 3.9e-6 pu); it binds at buses 5 and 9 (6 and 35–37 entries), unchanged by the unit.
- **Conclusion (Addendum 34):** voltage support is unmonetized because nothing binds at the storage's bus, and there is no reactive flexibility to displace.

## 5. Flexibility-price ladder (Addendum 34; `dc468ab4`)

Six evaluations, all certified; m multiplies the `cost_flex` profile through a harness override (no workbook edit), verified inert at m = 1 by a bitwise two-cycle gate.

| m | mean price €/MWh | Q(0) | value | value − I | resolution | verdict | value per full cycle |
|---|---|---|---|---|---|---|---|
| 1 | 48.4 | 653,859,461 | 259,428 | −58,529 | 34,735 | does not pay | 60.4 |
| 1.5 | 72.6 | 744,266,071 | 286,882 | −31,075 | 20,314 | does not pay | 67.3 |
| **2** | **96.7** | 811,016,062 | 366,353 | **+48,396** | 4,847 | **pays** | 86.5 |
| 3 | 145.1 | 876,022,706 | 434,574 | +116,617 | 35,488 | pays | 102.9 |

**Predictions versus outcome:**
- **"Indeterminate between ×2 and ×3": refuted.** The unit **pays determinately at ×2**, by 10× the resolution. Break-even lies between ×1.5 and ×2.
- **"Ceiling ≈ 65–80 €/MWh per cycle": refuted.** 86.5 at ×2 and 102.9 at ×3, still rising. The ceiling argument assumed the DSOs stop buying flexibility once it exceeds the spread; their use is largely **structural**, so they keep buying and each displaced MWh is dearer.
- **"The marginal MWh pays at no level": untested.** It needs a larger design at raised prices; the six-point ladder has only the smallest unit.

**Where break-even falls is economically legible:** at m = 2 the mean flexibility price (96.7) is essentially the market 4 h spread (98.0). Storage begins to pay when flexibility costs about what the arbitrage spread is worth.

**Caveat to carry into any table:** raising the flexibility price raises the whole system cost (Q(0) from 653.9M to 876.0M, +34 %). These are **different systems**, not a sensitivity band around one. The ladder is a break-even axis, as Addendum 34 frames it, not an uncertainty range on the baseline result.

## 6. Questions for the Expert and the Author

1. **Paper-scale route.** The rule fails on cycle time (42.3 min against 25). Options: (a) the ≥ 64 GiB machine, accepting ≈ 78 h per evaluation, ≈ 10 days for three; (b) the SRP1-with-caveat fallback plus the reduced-scenario variant; (c) reduce the paper instance (fewer days or scenarios) so a cycle fits. Which?
2. **Flexibility-price framing.** The ladder shows storage pays at about twice the current flexibility price, i.e. when flexibility costs about the arbitrage spread. Addendum 33 requires a data anchor (real DSO congestion-management prices) before any price change is adopted. Is the ladder reported as a break-even axis only, or do you have an anchor?
3. **The congestion/shifting dichotomy** captures nothing as defined (relief in binding hours is free P-up). Redefine, or report as measured?
4. **The marginal-MWh prediction** at raised prices is untested. Authorize two more evaluations (e.g. node 7, 4 h, 2 MWh at ×2 and ×3)?
5. **STEP4 §5.2** misstates NOMAD 4's frame rule; and the "four orders of magnitude" wording is replaced by §5's sentence. Both for the author.
6. **Small harness task:** align the scale script's solve identity with the ladder's per-event rule, so an unrecovered block does not fail the guard.

## 7. Evidence

| item | commit |
|---|---|
| spec v18; spec v19 | `e904955b`; `f62fc73d` |
| Q6 negative penalties | `f67ec7d4` |
| x = 0 capture: launcher, spec, run; gate ruling; analysis; DN plateau + weighted split | `49b3a9c3`, `486b2dd8`, `c23a6cd8`; `8f9d67f2`; `dd0a86b3`; `d68c814d` |
| NOMAD stub | `2c1272f4` |
| flexibility and P/Q splits | `a6adf0d8` |
| memory: profile, fix, gate; decision-rule reading; paper re-measure | `4cf5666c`, `f5c2d87c`, `01d6d01b`; `67dd7d71`; `78a9b230` |
| flexibility ladder: override, gate, launcher, spec, run | `2e82a23f`, `51e74279`, `812ee116`, `55ff2cca`, `dc468ab4` |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the manifests.
