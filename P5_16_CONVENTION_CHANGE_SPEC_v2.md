# P5.16 convention change — spec v2 (v1 `P5_16_CONVENTION_CHANGE_SPEC_v1.md` + Advisor review, 2026-10-30)

Planner. Supersedes v1 (kept for the record). Advisor verdict on v1: build with changes — five blocking (B1–B5),
recommended (R-a…R-g), all integrated here; two of the Advisor's findings are **formulation facts for the expert**
(§0). Still a draft until the expert confirms R1 with §0 in view; **no evaluation runs** under this spec.

## 0. Formulation facts surfaced by the review (verified by the Planner against the code)

- **F2 — the 2025 cohort's salvage is identically zero.** `_get_remaining_calendar_life`: age at the horizon end =
  Σ block widths from y_inv = 15 y = calendar life ⇒ remaining-life fraction 0 for a 2025 cohort on the current and
  the five-block instance (0 / .2 / .4 / .6 / .8 for 2025…2037). Consequences: (i) the current reference unit
  (2025) carries **no** salvage under today's basis, so the "net" column equals gross for it; (ii) under R1 option
  (a) dispatch becomes salvage-aware **only for 2028+ cohorts**; for a 2025-only plan the §3.3 objection of the
  critical review stands. Whether a calendar-expired asset with SoH above the floor should carry a health-based
  residual value is an **author/expert decision** (today's `calendar_life_basis = REMAINING_FRACTION_AT_TERMINAL`
  multiplies the health term; a health-only basis would credit the 2025 cohort at SoH ≈ 0.79).
- **F1 — scale.** `esso_al_scale = sigma_over_median_block_weight`: a = σ/w_med multiplies only the ESSO's AL terms;
  network objectives are divided by σ/w_b. Parity of a € term in the ESSO objective with the networks' € requires
  **κ = a/σ = 1/w_med** (≈ 1/412 on SRP1), read from the built models as
  `admm_esso_al_scale / admm_common_objective_scale` — not 1/σ, not σ.
- **F4 — a free valid tightening for the LB side (expert wording, Decision 2 of the analysis):** SoH_T ≤ φ_cal^age is
  dispatch-independent, so the LB may credit salvage at normalised health (φ^age − soh_min)/(1 − soh_min) (0.447 for a
  2028 cohort at φ = 0.985, soh_min = 0.7) instead of 1.0, and bound available energy by φ^age × rated — both affine in
  E. Material against the H1/H2 thresholds; replaces the plain "SoH ≡ 1" wording.
- **F9 — "horizon-end cost" is operationally the 2037 column**: the cost reader (`shared_energy_storage_data.py`
  l.1786–1796) requires an exact year column and exits otherwise; a 2040 value needs data or an extrapolation rule
  (author).

## 1. Changes (switch-controlled; every default reproduces today's build bitwise by construction — the off path does
not construct the new terms at all, as `apply_al_scale` does)

### C1 — terminal salvage term in the ESSO objective (R1 option (a))
- `objective = feasibility_penalty + κ · salvage_credit_obj`, `salvage_credit_obj = −Σ_cohorts [disc_T · f ·
  c_E(basis, y_inv) · rlf(y_inv) · residual_energy]` with residual energy built as **Param `es_e_investment_fixed[y_inv]`
  × Var `es_soh_per_unit_cumul[y_inv, T]`** (linear in one Var; the D-row's own substitution), guarded by rlf > 0 and
  (y_inv, T) inside the cohort window — never on the bilinear `e_available` row. Under 'mid' the salvage uses terminal
  SoH (l.885–888), unchanged.
- **κ is a mutable Param**, initialised 0 in `_build_subproblem` (σ and a are resolved after the initialisation
  solves), set to `admm_esso_al_scale / admm_common_objective_scale` in `update_shared_energy_storage_model_to_admm`;
  the standalone initialisation solve therefore runs with κ = 0 (stated in the manuscript's algorithm). Continuation
  reloads the baked Param — consistent.
- Switch `salvage_value.in_objective` (default **false**). With it off the term is not built.
- The ESSO complementarity leak diagnostic (`_get_esso_complementarity_diagnostics`, `eps_in_force =
  EPS_ESSO_THROUGHPUT`) becomes an overestimate once the salvage gradient dwarfs ε — adjusted to the effective
  marginal (ε + κ·∂salvage/∂throughput) or caveated in its record (diagnostic only).

### C2 — salvage basis (R2)
- New enum `cost_basis = EXPECTED_TERMINAL_ENERGY_COST` accepted in `shared_energy_storage_parameters.py` (l.41–48),
  served by **one cost-basis dispatcher** used by the expression (l.895), the results (l.1017/1040) and the
  sensitivities (l.959+) — fix at the writer. Terminal cost = the last representative year's column (2037) unless the
  author supplies a 2040 value. `energy_recovery_fraction` and `recycling_floor_fraction`: author's cited values
  (today 1.0 / 0.0). Energy only; converter not credited.

### C3 — mid-block SoH baseline
- `available_energy_soh_point = 'mid'` in the new instance's ESS params. Advisor: 'mid' (geometric mean) differs from
  the true time-average (logarithmic mean) by ≈ (ln r)²/24 ≈ 1e-4 — trapezoidal buys nothing. Floor stays end-of-block.

### C4 — five-block single-scenario instance
- Derived with `p515_s44_scale_measurement.derive_case('paper', {num_market_scenarios: 1, num_operation_scenarios: 1})`
  (prior art: p515_s52 ran this Years map), **not** by hand: `Years = {2025: 3, …, 2037: 3}`, `NumMarketScenarios = 1`.
  New ESS params file named in the spec (C2/C3 set; `objective_scale` σ_fixed = 93,635,360 was calibrated on 5-year
  blocks — the [1/3, 3] σ-band assertion fires at the first solve; **prediction recorded:** ratio inside the band,
  point estimate ≈ 0.6 by s52's 2-scenario reading). Assert `tcal_norm = round(15/3) = 5` integral; record
  `prob_market_scenarios` from the Excel 'Scenarios' sheet beside `NumMarketScenarios` (they are independent).
  `SRP1.json`, `SRP1_ESS_Params.json`, `SRP1_ESS.xlsx` unmodified.

### C5 — general lattice (harness layer)
- As v1 §C5: `p516_lattice.py`, z ∈ ℤ^{2|N||Y|}, per-cohort duration, **cumulative rated energy per node-year ≤
  max_capacity** (production's only such row is the master's `es_e_rated[e, y] ≤ max_capacity` with `es_e_rated` =
  Σ live cohorts — semantics confirmed), budget over all years by the production expression, canonical per-(node,
  year) map, key schema v2 with the x = 0 alias. Neighbourhood per R3 (pending).

## 2. Gates (zero solves)

- **G-CC-1 Switch-off identity (SRP1, unchanged files):** every switch off ⇒ ESSO + all network blocks of x = 0 and
  the unit bitwise identical to the production head (NL digests, W183 method); all preserved fixtures unpickle.
- **G-CC-2 x = 0 under the new convention:** network blocks bitwise identical to G-CC-1; ESSO **value-inert**
  (objective and solution identical) with the NL diff restricted to objective entries and enumerated; ESSO bitwise
  identity recorded as a **prediction** (NL-writer folding of constant terms unverified), not assumed. Unit cell:
  diff enumerated and restricted to C1 (objective), C2 (salvage expression), C3 (available-energy rows).
- **G-CC-3 Salvage calculus (zero solves; AD/FD of `pe.value` on the built ESSO):** on a cell with a **2028+ cohort**
  (the 2025 unit is vacuous — F2), ∂objective/∂throughput per cohort equals κ·∂salvage/∂throughput to ≤ 1e-6 relative,
  negative sign, reported in €/MWh per cohort (0 for 2025; Advisor's order of magnitude ≈ 2 €/MWh for 2028, ≈ 8 for
  2037, against ε = 1e-5); κ reproduced by a hand replica from the two recorded Params.
- **G-CC-4 Lattice:** old 7-variable `Lattice.reasons`/`domain` reproduced on the common-year slice; cumulative
  capacity rejects a two-cohort over-5-MWh plan; budget equals the master's `investment_cost` to 1e-9 — **oracle =
  the production master rows** (build the master, load the candidate via
  `_load_candidate_solution_into_master_model`, evaluate bodies; zero solves); canonicalization idempotent; keys
  round-trip.
- **G-CC-5 Instance build:** case file loads; 80 network + 3 ESSO blocks build at x = 0 and at a 2028 unit; tcal_norm
  5; weights 3 × days × 1.02^{−(y−2025)}; terminal discount 1.02⁻¹⁵; rlf {0, .2, .4, .6, .8}; x = 0 NL digests
  recorded as the new instance's G0.

## 3. Campaign items this spec does NOT authorize (recorded for the later addendum)
- The behavioural A/B of C1 (salvage on/off at a 2028+ unit): prediction — lower EFC/day and higher terminal SoH on
  the on-arm; **no** dispatch difference for a 2025-only plan; no material change in cycles-to-certification.
- The new instance's x = 0 and unit references.

## 4. Not permitted
As v1 §3; additionally no change to the ESS-channel ADMM updates (Advisor: not needed — the networks see
κ·∂salvage/∂p through λ at a fixed point).

## 5. Open for the author (before freeze)
Recovery fraction and recycling floor with citations; 2037 column vs extrapolated 2040; health-only residual value
for calendar-expired cohorts (F2).
