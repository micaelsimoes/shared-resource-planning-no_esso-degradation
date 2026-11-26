# S56 multi-cohort storage formulation: proposal for the Expert (2026-11-21)

**Status: proposal.** Nothing is built.
- **Inputs:** the Expert ruling of 2026-11-21 (a correction, not a flag), the Advisor's design and the Planner's code
  checks.
- **Evidence:** step 0 (`step0_gates_v2/step0_gates.json`, `aeaab933`). H3 derates 17,934 of the 37,041 staged plans
  (the mixed-duration ones; worst ratio 4/7). The older-cohort SoC excursion reaches 14.5 %.

## Decisions needed

| # | decision | recommendation |
|---|---|---|
| E1 | **Converter arrangement** | **A:** each cohort is a complete unit (own battery, own converter), AC-coupled at the node. This is what the costing implies: each cohort buys its own power, obeys its own 2–4 h duration, and expires with its energy. A′ (per-cohort DC/DC behind one shared inverter: per-cohort P and SoC, Q on the aggregate circle) is the fallback. B (paralleled batteries on one DC bus) is not consistent with the cost model |
| E2 | **Inter-cohort transfer** (charge one cohort while another discharges). Under A it is physical and costs round-trip loss, 2ε and degradation | **priced and monitored:** a circulation detector and an exit gate, no prohibition (an exact ban is a nonconvex complementarity). The alternatives are to exclude it explicitly or to admit it, and admitting it changes the networks' aggregate balance |
| E3 | **Split rule when terminal value is off (κ = 0).** The split between cohorts is unpriced without TV, so per-cohort SoH, published capacity and S are set by the solver's interior point | no κ = 0 re-optimisation for multi-cohort plans; the no-TV sensitivity for them uses level (a), repricing at fixed dispatch. Alternative: a soft equal-cycling tie-break, H3 as a preference |
| E4 | **"Oldest cohort first" behaviour.** With TV on, the 2025 cohort's throughput costs nothing in TV (remaining-life fraction 0), against ≈ 2 €/MWh for 2030 and ≈ 4 €/MWh for 2035. The ESSO will therefore cycle the 2025 cohort first, up to its converter, its window and the 0.70 SoH floor | accept, as a consequence of the TV convention; report per-cohort throughput |
| E5 | **Reactive request at initialisation.** The initialisation fixes ESSO P and Q to the TSO request. P has a slack; Q does not, so Σ q_c = Q can be infeasible at solver tolerance (the D0 v1 failure class) | a Q slack pair at the same penalty. If it breaks single-cohort NL identity, a radial clip of the fixed request on multi-cohort nodes |
| E6 | **Exit certificates added to acceptance:** ESSO slack (MW), ESS-channel residuals \|P_tso − P_esso\| and \|P_dso − P_esso\| in MW/MVAr, priced ESS residual, circulation detector | adopt; thresholds by the accuracy-first procedure (search-policy review) |
| E7 | **R8 (1e4) re-verification** on the final formulation | T0–T2 with a positive control (search-policy review) |

**Author:**
- **A-c1. Units.** Confirm that each addition is a complete unit. Energy-only augmentation of an existing converter is then
  outside the domain.
- **A-c2. Reproducibility.** Keep H3 reachable behind an explicit legacy key, for reproducing the archived results.

## Corrections to the request's premises (Advisor; verified by the Planner in code)

1. **The ESSO has no per-cohort SoC or Q to reuse.** It has no stored-energy variable at all (the SoC lives only in the
   networks' aggregate `shared_es_soc`). Q exists only as the aggregate `es_qnet`. Per-cohort SoC and Q are new
   variables and rows.
2. **H3 rows are always built and only activated** when ≥ 2 cohorts are active. The new rows follow the same pattern:
   built, active only on multi-cohort node-years.
3. **The ESSO's consensus copy carries a priced slack:** `es_pnet = Σ_c (pch − pdch) + σ⁺ − σ⁻`
   (`shared_energy_storage_data.py:785`). Once the ESSO owns cohort physics, a slack ≈ 0 at exit is the
   physical-feasibility certificate.
4. **The 48 % do not become "fully usable".** The allocation derating disappears, but each cohort stays limited by its
   own energy.
   - **The Expert's example (0.25/1 + 0.25/0.5, block 2030, from a 0.5 start):** the first hour can discharge 0.428 MW,
     against 0.375 MW under H3 and 0.5 MW in the networks' aggregate.
   - **When 0.5 MW is reachable:** for one hour only, after a pre-charge of about 0.19 MWh.
   - **Where the aggregate relaxes the per-cohort set:** on essentially every multi-cohort node-year (30,348 plans have
     one), not only the mixed-duration ones.
   - **Effect on rankings:** same-duration, different-age plans become weakly tighter than today, so rankings can move
     both ways.
5. **Price-taker initialisation refuses** more than one cohort or scenario (`shared_ess_price_taker.py:126–133`). Staged
   and stochastic evaluations use the default initialisation.

## The formulation (arrangement A; everything in the ESSO; networks and ADMM coupling unchanged)

The new rows are active only for a (node, year) with ≥ 2 active cohorts. c = cohort (`y_inv`).

| # | element | form | code |
|---|---|---|---|
| 1 | cohort schedules | p^ch_c, p^dch_c ≥ 0 | existing `es_pch/pdch_per_unit`, free once H3 is off |
| 2 | cohort power box | p^ch_c, p^dch_c ≤ s_c | existing `energy_storage_limits` (l. 750–753) |
| 3 | cohort SoC | e_t = e_{t−1} + η^ch p^ch Δt − p^dch Δt/η^dch; e_0 = 0.5·E^av_{c,y} | **new** Var and rows (coefficients as the network `sess_soc_rule`) |
| 4 | cohort window | 0.1·E^av_{c,y} ≤ e_t ≤ 0.9·E^av_{c,y} | **new**; the RHS is the existing available-energy Var, so the cohort windows sum to what the networks see and 'end' is inherited |
| 5 | cohort closure | e_T = 0.5·E^av_{c,y} | **new** |
| 6 | cohort converter | (p^ch − p^dch)² + q_c² ≤ s_c² | **new** Var q_c and row |
| 7 | aggregation P | P = Σ_c (p^ch − p^dch) + σ⁺ − σ⁻ | existing, kept |
| 8 | aggregation Q | Q = Σ_c q_c (+ slack pair, E5) | **new** |
| 9 | aggregate circle | P² + Q² ≤ (Σ s_c)² | existing; **deactivated on multi-cohort years** (implied by 6–8; keeping it fails LICQ at the binding hours) |
| 10 | degradation | throughput, D, SoH chain, floor per cohort | existing, unchanged; each cohort now degrades from its own throughput |
| 11 | objective | slack + ε·throughput + κ·salvage + AL on (P, Q) | existing, unchanged |
| 12 | H3 pro-rata rows | p_c = share_c·ΣP | **unwired** (function, Param and rows retained for preserved pickles; reachable by the legacy key) |

**Scenario-common:** the ESSO has no scenario index, and the networks alias one schedule.

**Expiry:** the existing lifetime gate applies. No cohort expires inside either grid, so a dedicated build test covers
expiry.

**Networks: unchanged.** Their aggregate storage is an outer approximation of the per-cohort set: equal efficiencies
across cohorts, discs summing to the Σ s circle, windows summing to the aggregate window. At consensus with ESSO
slack ≈ 0, the operating schedule is per-cohort feasible.

**The residual risk is one-signed.**
- **Why:** operating cost is evaluated on the network copies, so at a finite ESS residual multi-cohort plans are
  costed optimistically.
- **How it is monitored:** the ESS-channel residual and its priced form (E6).
- **Fallback (N2):** per-cohort rows in the network models too, if step v-b shows the bias is not small against σ.

**Size:** up to ≈ 480 SoC and ≈ 480 q variables and ≈ 2,300 rows per node on the three-block grid, roughly doubling
the ESSO. Its IPOPT time stays in milliseconds. The ADMM cycle count on multi-cohort plans is the unknown.

**Single-cohort identity.**
- **How it is built:** the new Vars are fixed at 0 and the new rows built deactivated, in their own registry (not
  `_esso_cohort_constraints`), activated per (node, year) only when n_active ≥ 2.
- **What it rests on:** the NL writer omitting unreferenced variables and inactive rows. That is not yet verified, and
  the gate decides it.

## Validation (frozen spec, armed guards, declared solve counts; predictions to be recorded)

| # | test | prediction |
|---|---|---|
| i | single-cohort NL identity (x0, the reference unit, a 2030-only and a 2035-only plan; init and ADMM states) + the SRP1 two-cycle bitwise gate | all equal; new rows active 0, H3 active 0 |
| i-b | row inventory for a 2025 + 2030 plan | SoC 384, closures 16, window 768, cohort circles 384, Q rows 192; aggregate circle active only in 2025 |
| i-c | zero-solve: sampled per-cohort-feasible schedules satisfy the network aggregate rows | all to 1e-12; the 0.5 MW first-hour point satisfies the network rows and violates the cohort rows |
| ii | the Expert's example: (a) −0.5 MW in hour 1; (b) a 2-h-cohort pre-charge ≥ 0.20 MWh, then −0.5 MW for one hour | (a) 0.428 ± 0.003 MW delivered (H3: 0.375); (b) 0.5 MW with slack ≤ 1e-8 |
| iii | an aged + a new cohort, same duration, full aggregate swing | every cohort inside its own 0.1–0.9 window (H3: the old cohort reaches [0.067, 0.933]) |
| iv | circulation: (a) P = 0 all day; (b) a peak needing pre-transfer | (a) detector ≤ 1e-6·s; (b) circulation > 0 and slack below the no-transfer bound |
| v-a | ESSO-only staged A/B, κ on and off | the earlier cohort's rows unchanged; per-cohort D from its own throughput to 1e-9; κ on: the 2025 cohort cycles more than its energy share |
| v-b | one full single-scenario staged evaluation (mixed duration, κ on) | ESSO slack ≤ 1e-6 MW at exit; cohort SoC inside its window; S = the hand formula; cycle count within ±30 % of an equal-capacity single cohort (stop above 1.5×) |
| vi | centralized: merged network with one storage unit per cohort at the bus | \|C_ADMM − C_central\| ≤ max(3 bar, 2τ) (whether the merge accepts two units at one bus is unchecked) |
| vii | expiry build test with a shortened calendar life | rows off and Vars fixed outside the window |

**Order:** i, ii and iv-b first (zero-solve plus a few millisecond ESSO solves). v-b and the pilots run only after the
Expert rules on E1–E3.
