# P5.15 Addenda 35–37: marginal MWh pays at ×2, generalization gated, row 18 blocked on four rulings

**Planner report, 2026-09-22.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 35, 36, 37.
- **Frozen spec:** v20, `data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json` (`d8bf936c`). All predictions were recorded in it before the runs.
- **Baseline:** C2 (k = 35,851) + φ_cal 0.985 + soh_min 0.70. C3 remains the sensitivity set; flexibility-price scenarios are labelled as such and never mixed with the baseline.
- **Objective convention on every table:** Q = certified `gross_operational_cost`, settlement excluded; F = I + Q; value = Q(0) − Q(x).
- **Status:** nothing is running. **The pilot is blocked**: row 18 as signed is not implemented, and four rulings are needed before it can be (§4).

## 1. Verdict

1. **The marginal MWh pays at ×2, so F2 is confirmed as the demonstration case.** At the ×2 flexibility price the second MWh returns 360,723 against its 317,957 cost, determinate at 6.5× its resolution.
2. **The four single-scenario code paths are generalized and gated.** SRP1 reproduces bitwise, and the >1×1 branches now execute correctly at 2×2 (8/8 checks, two arms).
3. **Row 18 as signed does not exist in the code, and the pilot cannot run without it.** What exists is a quadratic pin outside Q(x); under it the pilot's interface-dispersion output would be ≈ 0 by construction and uninterpretable.
4. **Four rulings are needed** (§4): the TSO's unpriced scenario direction, the storage non-anticipativity mechanism, whether deviation energy enters Q(x), and the voltage term. Two arise from facts about the code that Addendum 37's premises do not match.

## 2. Marginal-MWh test (Addendum 35 Q4; `2389d79b`, `a4d23ab9`, `0e36ec0a`)

Node 7, 4 h, 2 MWh (0.5 MVA / 2.0 MWh, 2025), both certified. x = 0 and the 1 MWh rows are taken from the committed ladder, not re-run.

| m | value | value − I | resolution | the second MWh alone | verdict |
|---|---|---|---|---|---|
| 2 | 727,076 | **+91,162** | 1,742 (52×) | Δvalue 360,723 − ΔI 317,957 = **+42,766** (resolution 6,589) | **pays** |
| 3 | 864,160 | +228,246 | 26,557 (8.6×) | +111,629 (resolution 62,045) | pays |

- **Prediction confirmed.** The author predicted value per MWh ≈ 300k against the 253,878 €/MWh energy cost; measured **363,538 €/MWh** at 2 MWh under ×2, which also clears the **true** marginal cost of the second MWh at 4 h, 317,957 €/MWh (energy plus the power that comes with it). Both figures are reported; the report leads with the true marginal cost.
- **Value is slightly concave**: the second MWh returns 360,723 against the first at 366,353. Extrapolating, the third MWh (≈ 355k) still clears 317,957, so the optimum under the €1M budget is expected at the **budget corner, 3 MWh** (I = 953,871, slack 46,129). The F2 ladder tests this directly; no extra evaluation was spent on it.
- **Budget corner computed zero-solve** from the pinned tables: the node-7 4 h ladder is feasible to 3 MWh; 4 MWh misses by 271,828.

## 3. Harness and generalization (Addenda 35 Q6, 36; `6db2491a`, `cb545653`, `f8918c38`, `f0501523`)

**Solve identity, per event** (`6db2491a`). The scale script credited retries only to *recovered* blocks, which is why the paper-scale run exited 1 (168 observed, 166 declared). The per-event rule now holds on all 10 committed records, and agrees with the old rule wherever no block stayed unrecovered.

**Unrecovered-failure policy** (`data/SRP1/Results/P515S50/unrecovered_failure_policy.md`): **production already complies**, verified from source, nothing changed.
- Continue with the last iterate: the solution is loaded only on success (`network.py:852-854`), multipliers are restored on a failed tier 2 (`:849-850`), and the ADMM loop breaks only on convergence (`shared_resources_planning.py:3373-3375`).
- The event is counted (`p515_g_g1_g4_admm_gates.py:558`).
- No unrecovered failure inside the certifying cycles: `local_solves_ok` gates `cycle_convergence` and **resets the streak to 0** (`:3057-3062`).

**The four paths** (`cb545653`), SRP1 byte-identical:
- `load_baseline` → new `install_baseline`, leaving the SRP1 path and its checksum assertion untouched.
- `run_admm_arm`'s solve count derived from the instance: 51 at SRP1 (identical), 83 at paper scale. `identity_holds` is repaired and now uses the per-event rule.
- Hull-polish helpers and the settlement reporting generalized, with the convention recorded per quantity: **above 1×1 they read and bound the expectation-level quantities the ADMM actually couples**, because bounding a scenario-(0,0) copy would read one realization as the consensus and suppress the very dispersion the pilot measures. `apply_common_values` fails loudly above 1×1 (it fixes (0,0) copies — a mechanism the formulation does not have) and is not on the campaign path.
- **Gate: PASS.** Two SRP1 cycles at C\* reproduce the committed baseline trajectory with 0 differences over 29 fields; 153 solves declared and observed, guard exact.

**2×2 smoke of the >1×1 branches** (`f0501523`), under the *current* quadratic, labelled a code-path check only:
- 2,880 hull descriptors per arm: finite, non-empty, **bitwise equal** to an independently recomputed closed hull, with three endpoints on the storage channels. The **ESSO endpoint is strictly interior on 62 entries**, so the Planner ruling to keep it is load-bearing.
- All 16 polished blocks solved; the objective decomposition reconciles to 5e-16; the S31C identity to 2e-14; the multi-market settlement branch genuinely taken (covariance term −1.78 on one block); `apply_common_values` tripwired and never called.
- **Third scaling point:** DSO block solve 3.3 s at 2×2 (SRP1 0.5 s, paper 30.5 s); cycle ≈ 37 s at one year, implying **≈ 3 min/cycle** at the pilot's five years; peak memory 2.9 GB. Single-sample, preliminary.
- **Finding:** above 1×1 the polish minimises `model.objective` (which carries the quadratic) while Δ is measured on `objective_function_rule` (which does not), so "Δ ≤ 0 by construction" does not hold there. It held empirically. **Implementing row 18 as signed removes the asymmetry**, since the quadratic is unwired and row 18 lives inside `objective_function_rule`; the implementation will assert that equality at 2×2.

## 4. Row 18: the audit, and the four rulings needed

**The finding.** Row 18 as signed (Addendum 10/11, reaffirmed in 37) is **not implemented**. What exists is `_add_tso_scenario_deviation_penalty` / `_add_dso_scenario_deviation_penalty` (`shared_resources_planning.py:3723-3788`): a **quadratic**, probability-weighted penalty on deviation from the expectation, over **voltage + interface P/Q at 9e4** and **shared-ESS P/Q at 1e4**, on **both** TSO and DSO, appended to `model.objective.expr` **after** `objective_function_rule` — hence **outside Q(x)** by construction, with two further sites subtracting it. Both the penalty table (`P5_15_S31_PENALTY_TABLE_DRAFT.md:120`) and `REVISION_CONTEXT.md:309-310` schedule row 18 for Step 5; commit `fc200780` implemented the other rows and left it.

**Why it blocks the pilot.** The existing quadratic is hard non-anticipativity in economic clothing: its marginal charge is ~1800·ΔMW €/MW, balancing a 50–100 €/MWh signal at Δ ≈ 0.03–0.06 MW, and a quadratic has *zero* marginal price at zero deviation — the opposite shape to a premium. Reporting dispersion under it would report the coefficient, not a priced result.

**Rulings (i) and (ii) are confirmed** by the independent review, with a stronger reason for (ii) than anchor-dependence: pricing against the incoming consensus value would put an L1 dead zone at the previous iterate in every scenario, so residuals would vanish trivially and **certification would be false**.

### The four items needing a decision

**A. Ruling (iii) rests on a premise the code does not satisfy.** Addendum 37 says the TSO "serves each scenario's actual interface flow through the per-scenario consensus channels". **There are no per-scenario consensus channels.** Verified directly: both augmented Lagrangians couple only `expected_interface_vmag`, `expected_interface_pf_p/q`, `expected_shared_ess_p/q` (`shared_resources_planning.py:4715-4735`, `:4875-4895`); the TSO's per-scenario `interface_delta_p/q` are freed within ±rating (`:3884-3893`) and coupled to nothing per scenario.
- **Consequence:** dropping the TSO quadratic leaves a near-flat direction — raise the deviation in one scenario, lower it in another, mean unchanged — priced only by loss and congestion differences. The solver would pick vertices that move between cycles, feeding noise into the expectation and the residuals.
- **Recommendation:** hard non-anticipativity on the TSO's δ_P/δ_Q (S−1 equalities per interface and period). That *is* the signed intent — "the TSO operates on the committed schedule" — and removes the null space. Read "per-scenario consensus kept" as keeping the per-scenario δ structure while pinning it. The alternative, adding genuine per-scenario consensus channels, is a much larger change to the coordination itself.

**B. Ruling (iv): the mechanism, reconciled.** Addendum 37 requires one scenario-free storage variable and explicitly forbids per-scenario copies tied by equalities (to avoid the Step-3 duplicated-row dual non-identifiability). The review prefers equalities, because removing the scenario index touches many rule signatures and risks preserved fixtures.
- **Both can be satisfied:** add a scenario-free variable, reference it in every scenario's balance rows, and **retain the per-scenario variables unwired**, per the repository's "deactivate and unwire, never delete" rule. Your single variable, no equality rows, fixtures still loadable.
- **Note:** non-anticipativity on net power alone is insufficient — charge and discharge must be covered, or scenarios differ in losses and stored energy at equal net.

**C. New issue, covered by neither ruling: Q(x) would omit the deviation energy.** With the TSO pinned, the settlement no longer cancels: T_TSO + ΣT_DSO becomes the covariance between price and deviation. Production subtracts the **whole** settlement from Q(x) (`:1038-1046`), so Q(x) would charge the premium on a deviation while treating the energy deviated as free.
- **Options:** split the DSO settlement into a schedule part (transfer, excluded) and a deviation-energy part (economic, included); or add the covariance back.
- **This also amends Addendum 13's cancellation gate**, which under row 18 should equal the covariance by construction rather than vanish — and reporting that identity becomes a correctness check.

**D. The voltage term.** Dropping the voltage quadratic frees each DSO to choose a different head voltage per scenario (bounded only by its limits, `:4818-4826`), since only the expectation is coupled — at exactly the feeder where the storage result was interpreted.
- **Recommendation (needs your signature, as Addendum 10 says "no deviation term"):** keep the voltage quadratic as a solver-only term, excluded from Q(x), and report the per-scenario side-to-side voltage mismatch. That preserves the signed intent for Q(x) while isolating the pilot's change to P and Q.

**Also proposed, to be adopted unless you object:**
- **A single-block 2×2 A/B costing seconds** before the 8–10 h pilot: the existing quadratic against row 18 at α ∈ {0, 0.5, large}. It checks that α → large reproduces the pin (your limit gate, cheaply), that α = 0.5 gives hour-selective deviation, and that α = 0 exposes any unpinned TSO direction.
- **Report α·π̄_t against each DSO's internal flexibility price per hour** with the dispersion. The result is a threshold outcome: deviation is zero in an hour unless the premium is below the cheapest internal alternative, so that ratio is what the number means.
- **A price floor** on α·π̄_t if any hour has a non-positive mean price (the lowest 4-hour mean on record is 10.34 €/MWh, so this looks unnecessary — it will be verified cheaply, not assumed).

## 5. What is ready once the rulings land

- The pilot instance is specified: 5 representative years, 4 days, 2 × 2 scenarios, α = 0.50, baseline ageing, AA-on, campaign cost file and budget. Per-block size is **4× SRP1** and the total model **6.7× SRP1, 0.16× paper**; 83 solves per cycle.
- Expected cost, from three measured scaling points: **≈ 3–4 min per cycle**, **7–10 h per evaluation**, **≈ 3.6–3.9 GiB** per process, two evaluations at concurrency 2.
- Gates specified in Addendum 37 are accepted as stated: the SRP1 two-cycle bitwise identity; the 2×2 limit check with α → large; the dispersion metric (RMS and max of d per DSO, in MW and as a share of mean interface flow, plus the total charge); R3.6 over α ∈ {0.25, 0.5, 1.0} with α = 0 as the free-deviation reference.

## 6. Questions for the Expert and the Author

1. **A — TSO:** adopt hard non-anticipativity on the TSO's interface deviations (recommended), or add genuine per-scenario consensus channels?
2. **B — storage:** confirm the reconciliation (scenario-free variable, per-scenario variables retained unwired), and confirm it covers charge and discharge, not only net.
3. **C — Q(x):** how should the deviation energy enter? Recommended: split the settlement, deviation energy as an economic term inside Q(x); and amend the Addendum 13 identity gate accordingly.
4. **D — voltage:** keep the voltage quadratic as a solver-only term excluded from Q(x) (recommended), or drop it and report the per-scenario mismatch?
5. **Sequencing:** may the F2 demonstration (node-7 4 h ladder at ×2, then Phase B under the budget) run while these rulings are settled? It is SRP1-only, needs none of the row-18 work, and the machine is otherwise idle.

## 7. Evidence

| item | commit |
|---|---|
| spec v20 | `d8bf936c` |
| marginal-MWh: launcher, spec, run | `2389d79b`, `a4d23ab9`, `0e36ec0a` |
| scale-script per-event identity + unrecovered-failure policy | `6db2491a` |
| generalization of the four paths; SRP1 two-cycle gate | `cb545653`; `f8918c38` |
| 2×2 smoke of the >1×1 branches | `f0501523` |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the manifests.
