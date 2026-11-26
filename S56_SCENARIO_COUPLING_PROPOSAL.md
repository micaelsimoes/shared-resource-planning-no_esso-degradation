# S56 scenario-wise TN–DN coupling: proposal for the Expert (2026-11-21; Planner, from the Advisor's review)

**Status: proposal.** Nothing is built.
- **Evidence:** audit D (`f80ec8a9`).
- **Running now:** the diagnostics D0 and V0, under a frozen spec.
- **Adoption** is the Expert's ruling. It reverses Addendum 38 (A).

## Decisions needed (Expert)

1. **Adopt per-scenario coupling (F1), or keep Addendum 38 (A) and disclose.**
   - **Adopt (recommended for the stochastic study):**
     - an additional per-scenario *deviation* consensus channel on interface P and Q;
     - the TSO's per-scenario `interface_delta` freed, so the TN re-dispatches to each scenario's realized flow;
     - the existing expectation channel unchanged;
     - the commitment defined as the DSO's expectation; row 18 kept; V on expectation with the pin.
     - **Cost:** a production change behind a default-off option, new gates, and launchers re-pinned. Stochastic
       evaluations would take a guessed 1.3–3× more ADMM cycles.
   - **Keep (A):** disclose that deviation energy is settled financially and not carried in the TN model.
     - **Cost:** none in compute.
     - **Consequence:** the stochastic arm cannot pass DN quantity uncertainty to the TN, so a null result would be
       partly structural.
   - **Fallback, if the adopted form fails validation on convergence or cost:** "pass-through". DSO deviations are handed
     to the TN as data each cycle, which makes the exchange physically consistent every cycle. It is an equilibrium, not
     an optimization, and has no centralized optimum to validate against.
2. **The Q convention under the new mode.** Gross = TN generation + DN flexibility + row-18 premium + slack terms;
   interface settlement excluded in full; voltage pin excluded; net = gross − salvage.
   - **Why the settlement goes:** with deviations physically supplied, the whole settlement is a transfer again. The
     existing formula `gross − contracted − pin` already leaves only the price-weighted consensus residual, but reporting
     the DSO covariance on its own would double-count.
   - **At one scenario** the present code path is kept, for bitwise identity.
3. **The role of row 18.** It narrows to a pure premium for a balancing cost the TN model does not carry. That is about
   25 M€ at x = 0, and stays of that order.
   - **One concern for confirmation:** row 18 holds Q at the commitment (ΔQ ≤ 2.4 × 10⁻³ MVAr). That is the restriction
     the author asked to avoid, applied to reactive power.
4. **A new gate**, scoped to the arm: per-scenario ΔP, ΔQ and ΔV at the exit and per cycle. The gate reports:
   - the maximum and the ω-RMS;
   - the currency form Σ_b w_b Σ_s ω_s Σ_t π_s |ΔP_s| against the cell's bar.

   The thresholds are the Expert's.
5. **The residual test.** Use the ω-weighted per-scenario PF norm, r² = r̄² + Σ_s ω_s (r^d_s)², and report both parts.
   It is strictly stronger than the current test and identical at one scenario.
6. **The joint-scenario assumption.** Per-scenario coupling asserts that (s_m, s_o) is the same state of the world in the
   TN and in every DN. Expectation coupling never needed that, and audit A found the scenario draws are not joint.

## The formulation (F1)

For DN k, period t, scenario s, with weight ω_s. P is shown; Q is identical.

| side | quantity | definition |
|---|---|---|
| DSO (unchanged) | realized exchange | P^D_s = `pg_adn` |
| | commitment | P̄^D = Σ ω_s P^D_s (`expected_interface_pf_p`) |
| | deviation | d^D_s = P^D_s − P̄^D, with row 18 at α π̄_t (d⁺ + d⁻) |
| TSO (changed) | realized exchange | P^T_s = pc + δ_s, δ_s free per scenario within ±rating |
| | expectation | P̄^T = Σ ω_s P^T_s |
| | deviation | d^T_s = P^T_s − P̄^T |

**Coupling.**
- **Existing channel, untouched:** P̄^T = P̄^D.
- **New channel:** d^T_s = d^D_s for every s.
- **Augmented term on each side:** Σ_s ω_s [ μ_s (d_s − d^req_s)/R + (ρ_pf/2) ((d_s − d^req_s)/R)² ].
- **Dual update:** μ_s += ρ_pf (d^T_s − d^D_s)/R, starting from μ = 0. Under this update the gauge Σ ω_s μ_s stays 0.

**Properties.**
- **It is exactly ω-weighted per-scenario consensus:** Σ ω_s (p_s − z_s)² = (p̄ − z̄)² + Σ ω_s (d_s − z^d_s)².
- **Its fixed point** is a KKT point of the merged stochastic problem with row 18.
- **At one scenario it is structurally absent,** as row 18 is (`model_construction_helpers.py:1917`).
- **It adds no duplicated rows** (Addendum 37 (iv)).
- **It leaves `consensus_vars['pf']`, its duals and `t_sum` with their present meaning.** These are used 344 times in
  59 files.

**Alternatives rejected.**
- **An explicit free commitment p̄:** under an L1 charge it becomes the ω-median.
- **Replacing the expectation channel:** it would re-index every consumer.
- **P only:** Q would then stay consistent only through row 18.
- **A TN re-dispatch premium:** it is a new economic assumption.

**Change list** (new mode on, more than one scenario; nothing deleted; the alias path stays the default):

| site | change |
|---|---|
| `model_construction_helpers.py:1258–1259, 1274–1275, 1391–1393` | reference the per-scenario δ |
| `shared_resources_planning.py:4300–4306` | free and bound every δ copy |
| `shared_resources_planning.py:5195–5214, 5467–5472` | deviation terms; TSO proximal extension |
| `shared_resources_planning.py:4784` | new consensus and dual dictionaries |
| `shared_resources_planning.py:6143–6148, 6472–6477` | per-cycle Param updates |
| `shared_resources_planning.py:8496–8564` | read the deviations, update μ |
| `shared_resources_planning.py:7101–7361` | weighted PF norm, reported in parts |
| `shared_resources_planning.py:7376–7480` | Anderson state, or AA off |
| `shared_resources_planning.py:1302–1304` | convention, if ruled |
| hooks `352–371` and campaign harness | `t_sum` deviation part, per-cycle fields, new gate with a capture-path assertion |

**Option.** Default-off, readable as `getattr(params, …, False)`, following the no-reverse-flow option.

## Validation, before any stochastic search

| step | what | status |
|---|---|---|
| D0 | TN-only re-solve at the saved 3 × 3 pair, with the realized per-scenario DN exchanges (80 guarded solves). D1 = (C_B − C_A) − DSO covariance. **It sizes what Addendum 38 (A) omits**, and settles the manuscript question for the published pair independently of adoption | **running** |
| V0 | the merged multi-scenario centralized model builds (zero solves) | **running** |
| V3 | identity: SRP1 two-cycle bitwise with the option on; a multi-scenario smoke bitwise with the option off; every preserved fixture loads | after adoption |
| V1 | centralized comparator at x = 0, α = 0.5, on two blocks (the fewest and the most TN-curtailment hours). Merged model with row 18 added on interface P and Q; ADN V free and scenario-common, bracketing the pin; default and warm starts | after adoption |
| V2 | ADMM on the same blocks, pinned versus new arm, through the exit to settling | after adoption |

`combine_networks` empties the shared ESS (`centralized_coordination.py:74–75`) and has no row 18, so V1 needs that
construction. A comparator with storage (V4) is optional.

**Pass condition for V2:**
- **Cost:** |C_S − C_central| ≤ max(3 × bar, 2τ) per block. This is a claim on cost, not dispatch.
- **New arm:** max |ΔP_s| ≤ 0.5 MW and ω-RMS ≤ 0.05 MW.

**Predictions recorded with each spec:**
- **D0:**
  - Optimal on 80 of 80 solves;
  - zero TN slack;
  - |D1| of 0.05–1 M€;
  - pair difference below 18 k€.
- **V0:** about 343,656 variables.
- **V1:**
  - the two starts agree to 1e-6;
  - the V bracket within 1e-5;
  - C_central − C_pinned ≤ 0 on the high-curtailment block.
- **V2:**
  - the new arm meets max |ΔP_s| ≤ 0.5 MW and ω-RMS ≤ 0.05 MW;
  - cycles in the ratio 1.3–3.0 (new arm over pinned);
  - cycle wall in the ratio 0.9–1.2.

## Implications

- **The uncertainty experiment.**
  - The SP arm is interpretable as a statement about the system only after adoption, or after D0 shows the gap is
    immaterial.
  - The EV arm is unaffected (one scenario), as are tasks E and F.
  - Design-note predictions 3, 5, 7 and 8 are re-recorded.
- **Effort.**
  - SP evaluations would go from about 5 h to a guessed 7–15 h.
  - The matched SP search would go from about 7 days to 10–21 days serial.
  - The five-block 3 × 3 pair would need re-running under the new mode.
- **Manuscript (round 9c).**
  - **No sentence claims that deviations are physically balanced.** Lines 1098 and 1158 describe the common TSO schedule
    and the settlement at scenario prices.
  - **Missing:** one sentence stating that deviation energy is settled financially and not carried in the TN model.
  - **The pair value of 236.7 k€** is expected to move by less than the 18 k€ resolution: the storage schedule shifts
    P_s and P̄ equally. The level Q(x) may move by 0.1–1 M€. D0 tests this.
  - **Status:** the pair evaluations are "uncertified" in the manuscript (l. 1635, 1664) and must not be called certified
    elsewhere.
- **Push-back recorded.** "Consistency absent" should read as "the approximation of Addendum 38 (A), now quantified".
  The reason for revisiting it is the changed stake: a search under uncertainty, not a fixed-plan evaluation.
