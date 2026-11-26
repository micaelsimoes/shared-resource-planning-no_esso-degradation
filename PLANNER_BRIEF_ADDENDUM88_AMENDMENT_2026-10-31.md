
## Amendment to Addendum 88 — author's decisions on R2(ii) and R2(v) (2026-10-31)

1. **Converter: not credited (author).**
   - The expert concurs. The credit is confined to the energy capacity, the quantity whose ageing the model tracks.
   - R2(v) and C^conv are withdrawn.
   - The manuscript states the leftover late-installation penalty beside the timing results: a late cohort carries the
     undepreciated part of its converter cost (for a 2037 cohort, roughly a third of its energy credit at 4 h, more at
     2 h).
2. **Salvage is a notional residual value, not a resale value. There is no recovery fraction (f = 1).**
   - The expert's R2(i) horizon-end basis is withdrawn in favour of the author's original-cost basis: the cohort's
     expected (probability-weighted) energy investment cost in its installation year. This is the basis already
     implemented (`EXPECTED_INSTALLATION_ENERGY_COST`).
   - Why the expert withdraws R2(i): with the original-cost basis, the credit is the undepreciated share of each
     cohort's own investment. F therefore charges each installation for the part of it used within the horizon,
     consistently with I(x), and no choice between a 2037 and a 2040 price is needed (F9 closed).
3. **The shape of the credit is unchanged from the code.**
   - S = disc_T · Σ_c c^E(y_c) · rlf_c · E_c · (SoH_{c,T} − SoH^Min)/(1 − SoH^Min), with recycling floor 0.
   - Calendar-expired cohorts (2025) therefore carry zero, and R2(iv) stands.
   - A health-only credit (no remaining-life factor) is a different convention. It would credit a 2025 cohort after the
     model has retired it at T^Cal. It is not adopted.
4. **Consequences.**
   - **C2:** no change to the salvage basis. Today's defaults are confirmed and enter the freeze: `cost_basis =
     EXPECTED_INSTALLATION_ENERGY_COST`, `energy_recovery_fraction = 1.0`, `recycling_floor_fraction = 0.0`, energy
     only. The cost-basis dispatcher is not needed.
   - **Sensitivity:** the sensitivity required by R2(ii) reduces to the salvage-off arm (the C1 A/B) on the
     timing-relevant plans.
   - **G-CC-3 prediction (unchanged):** ≈ 1 €/MWh for a 2028 cohort and ≈ 5 €/MWh for a 2037 cohort, each within a
     factor of 2; 0 for 2025.
5. **Manuscript wording.**
   - "The terminal credit is a notional residual value, not a resale price: the expected energy investment cost of
     each installation, scaled by the share of its calendar life remaining at the end of the horizon and by its usable
     capacity above the end-of-life floor, and discounted from the end of the horizon (2040)."
   - "The converter is not credited; an installation made late in the horizon therefore carries the undepreciated part
     of its converter cost."
6. **Planner order (zero solves).**
   - Revise `P5_16_CONVENTION_CHANGE_SPEC_v3.md` (written at `95148ca7`) to v4, with C2 closed per this amendment:
     - (i) basis: installation-year expected energy cost (today's code). The 2037 column, the cost-basis dispatcher
       and the F9 note are removed.
     - (ii) f = 1.0, meaning a notional residual value.
     - (v) the converter is not credited, and C^conv is removed.
     - The freeze list carries today's salvage defaults.
   - Report the v3 → v4 diff.
   - The Addendum 88 hold on the convention change is lifted once v4 is committed. The zero-solve tasks
     (G-CC-1 to G-CC-5) then proceed as ordered.
