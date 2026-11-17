
# Addendum 88 — expert rulings on `EXPERT_REQUEST_2026-10-31.md` (R1–R4, R6, F4); D0 and convention-change amendments; predictions (2026-10-31)

**Inputs.** I read the following:
- Addendum 87 (adopted 2026-10-31) and the request (R1–R4, R6, F4; R5 closed by the author);
- `REVISION_REALIGNMENT_ANALYSIS_2026-10-30.md` v2;
- `P5_16_D0_SPEC_v2.md`, including the Planner's §10 rulings after T0.1a;
- `P5_16_CONVENTION_CHANGE_SPEC_v2.md` and `P5_16_TPERF_SPEC_v1.md`;
- `P5_16_D0_T01A_REPORT.md` (`42840e42`).

I checked them against the production salvage code: `shared_energy_storage_data.py` l. 846–909 and
`_get_remaining_calendar_life` l. 915–922.

**The problem every ruling below refers to.**

P_S: min over x ∈ X of F(x) = I(x) + Q_S(x), where

Q_S(x) = min over AC-feasible coordinated operation u of [Q(x, u) − S(x, u)].

Here Q is the gross operational cost (contracted settlement excluded) and S is the terminal salvage credit of R2.

- **Upper bound:** UB(x) = I(x) + Q(x, u_cert) − S(x, u_cert), at a certified operating point.
- **Lower bound:** LB comes from the relaxation (F4), with the master over X.

Both bound P_S:
- the value of any operating point that is feasible to tolerance bounds Q_S from above, whatever rule produced the
  dispatch;
- the relaxation bounds Q_S from below provided it enlarges the feasible set and over-credits S.

Two consequences are used throughout:
1. The UB needs feasibility, not optimality.
2. Certification matters only where plans are compared.

## R1 — salvage mechanism: option (a) confirmed

1. **Why (a).**
   - Under P_S, option (a) evaluates the recourse that P_S defines.
   - Option (b) evaluates a feasible but deliberately non-optimising dispatch of the same recourse. It is still a
     valid UB, but a weaker one with a known bias, and the critical review's §3.3 objection applies to it.
   - Option (a) costs one Param and one objective term.
2. **The governing decision is restated, not reversed.**
   - New wording: "No cycling or degradation cost is priced anywhere. The only degradation-dependent term in any
     objective is the terminal value lost to ageing, which is part of the planning objective F by definition. Within
     operation, the SoH floor remains the only enforced intertemporal constraint."
   - This keeps the author's principle that a paid-for asset carries no operating cost. The salvage marginal is the
     opportunity cost of the asset's own residual value, not an imputed price.
3. **F2 does not weaken (a).**
   - Under R2(iv), a cohort whose calendar life ends at the horizon has zero terminal value. For such a cohort,
     salvage-aware and salvage-unaware dispatch coincide, so there is nothing to internalise.
   - The §3.3 objection is therefore answered for every cohort: by the term for 2028+ cohorts, and by the zero-residual
     convention for 2025 cohorts.
   - Manuscript sentence: "A cohort installed in 2025 reaches the end of its 15-year calendar life at the end of the
     horizon and carries no terminal credit; for it, salvage-aware and salvage-unaware operation coincide."
4. **Conditions added to G-CC-3 (zero solves).**
   - (i) **€ parity as an identity, checked numerically on the built models.** After the a = σ/w_med factor, a 1 €
     change of S changes the ESSO objective exactly as much as a 1 € change of a block's present-value cost changes
     that block's ADMM objective. This is in addition to the specified hand replica of κ.
   - (ii) **Slack safety.** The salvage marginal per unit of `es_pnet`, in objective units, must be
     ≤ 1e-3 × `PENALTY_ESSO_SLACK`. Otherwise the P-net slacks become cheaper than cycling the cells and the ESSO
     "dispatches" through them. If it fails: stop, and raise the slack penalty before freeze.
   - (iii) **Complementarity diagnostic.** It uses the effective marginal ε + κ·∂S/∂throughput. The salvage term
     penalises both directions of throughput, so it reinforces the structural complementarity rather than weakening
     it.
   - (iv) **Initialisation.** κ = 0 at the initialisation solve, stated in the manuscript's algorithm.
5. **Reporting.**
   - Every UB reports Q (gross), S (realised), I and F separately.
   - The value of a plan becomes Q(0) − Q(x) + S(x), one definition throughout the paper. S(0) = 0.

## R2 — salvage basis and parameters

The credit is

S(x, u) = disc_T · f · Σ_cohorts [ c^E_T · rlf_c · E_c · h_c(u) ] + C^conv(x),

where
- h_c = (SoH_{c,T} − SoH^Min)/(1 − SoH^Min);
- the recycling floor is 0;
- C^conv is given in (v).

**(i) Cost basis: horizon-end expected cost.**
- In practice this is the **2037 column**: the expected energy cost of the last representative year (which stands for
  2037–2039), probability-weighted over the cost trajectories and discounted from the horizon end (1.02⁻¹⁵).
- No extrapolation to 2040. There are no data for it, and an invented rule is worse than a stated one.
- Why: a unit of residual capacity at the horizon end has one value whatever its purchase year (the replacement-cost
  reading). A purchase-year basis would credit older cohorts at superseded prices. With the year now a decision
  variable, the timing result would then be decided by the price trajectory rather than by operation.
- Direction, to state in the paper: with declining costs, the 2037 price overstates a 2040 price. The f sensitivity in
  (ii) covers this, because S is linear in f · c_T.

**(ii) Recovery fraction: base f = 1.0, meaning value in continued use, not resale.**
- The horizon is a modelling truncation, not a sale.
- The remaining-life and health factors already depreciate the asset.
- A resale fraction well below 1 would reintroduce the late-investment penalty R3.1 objects to.
- *This differs from the Planner's recommendation and from the author's provisional agreement of 2026-10-30; the
  author decides.*
- In either case, the paper must also report:
  - the case with S off (f = 0; this is the C1 A/B arm on the timing-relevant plans);
  - a cited resale or second-life value as a further row, if the author supplies one;
  - the timing result at each.

**(iii) Recycling floor: 0.**
A scrap floor credits capacity below end of life. That is inconsistent with the continued-use meaning and with
h_c = 0 at SoH^Min.

**(iv) Calendar-expired cohorts: zero residual value.**
- T^Cal = 15 y is the asset's end of life by assumption, and the model itself withdraws the cohort at T^Cal (the
  cohort window). Crediting value after a withdrawal the model imposes would be inconsistent.
- φ_cal = 0.985 was calibrated as 80 % retention at 15 y, which is the calendar end-of-life statement.
- Manuscript sentence: as in R1.3.

**(v) Converter: credited pro rata to remaining calendar life.**
- The credit is C^conv = disc_T · f · Σ_c c^S_T · rlf_c · S_c, at the 2037 expected power cost, with no health factor
  (the converter does not age in the model).
- Why: without it, a late cohort pays the full converter cost for only its in-horizon years. That is the
  late-investment penalty R3.1 names.
- Size: at a 2037 0.25 MVA / 1 MWh cohort, the converter credit is ≈ 30 % of the energy credit (0.25 · c^S_T · 0.8 ·
  0.743 against 1 · c^E_T · 0.8 · 0.85 · 0.743, taking c^S_T ≈ c^E_T as in 2025). It is zero for 2025 cohorts.
- C^conv is dispatch-independent. It is computed in F and in the results writer, **never in the ESSO objective**, so
  the switch-off and x = 0 gates are untouched, and it is exact on the LB side.
- *This differs from the Planner's recommendation (energy-only); the author decides.* If energy-only is kept, the
  manuscript states that the converter is not credited, and the timing result carries that caveat.

**(vi) Manuscript reading of the product rlf · h.**
- Wording: "remaining capacity-years: the share of calendar life remaining times the share of usable capacity above
  the end-of-life floor; calendar fade enters both factors, so the credit is conservative."
- State the terminal date explicitly: T is the end of the last block (2040), and SoH_{c,T} is the end-of-block SoH of
  the 2037 block (critical review §3.3, last paragraph).

**(vii) Freeze.**
Cost basis, f, floor, converter treatment and terminal date go into the spec v2 freeze. Any change after the freeze is
a stop (Addendum 87).

## R3 — certificate on the general lattice: adopted, with two additions

1. **Neighbourhood N(x)** at the finest mesh (z ∈ ℤ³⁰). It consists of:
   - **(a) Coordinate steps** ±e_i on every P_{n,y} and E_{n,y}. This is the maximal positive basis, which carries
     the GPS/MADS theory at Δ = 1.
   - **(b) Activation and deactivation.** Each inactive cohort can be activated to 0.25/0.5 or 0.25/1.0. Each active
     smallest cohort can be **deactivated** to 0/0. This is the reverse move; without it, a smallest cohort is never
     tested against its own absence.
   - **(c) One-block shifts.** Each active cohort can move to y ± 1 at the same (P, E). If a cohort already exists in
     the target year, the shifted one is added to it; the per-cohort duration constraint is closed under addition.
     Timing is the question R3.1 raises, and the general lattice exists to answer it. Without (c), no year move is a
     unit move.

   (b) and (c) enter as search-step points, which the theory admits.
2. **Size in closed form.**
   - |N| ≤ 30 + 5A for A active cohorts: 4A coordinate steps, 2(15 − A) activations, ≤ A deactivations, ≤ 2A shifts.
   - Coordinate steps on inactive cohorts are inadmissible by construction and are not listed.
   - The size is recorded before any poll.
3. **Resolution by bound.**
   - A neighbour x′ is resolved without an ADMM evaluation when F_LB(x′), minus the LB's dual-bound margin, is at
     least UB(x̂). Such an x′ cannot improve on x̂ in P_S.
   - Report it separately: "resolved by the lower bound: k of |N|".
   - Under H1 this resolves most of N; under H2, possibly none of it.
4. **Run rule.**
   - Start at the master's plan with Δ = 1 and poll all of N.
   - Move on a determinate improvement.
   - Terminate when no point of N(incumbent) is determinately better.
   - Completion cap: 60 points (stop for review above it).
   - N^max: 120 new evaluations per certificate run, which is two complete neighbourhoods at A ≤ 6.
   - A run stopped by N^max reports "k of |N| evaluated", as was done for the F2 plan.
5. **Statement.**
   - Wording: "x̂ is mesh-local optimal on the investment lattice in the coordinate unit neighbourhood (unit steps in
     each converter rating and energy capacity, activation and removal of a smallest unit, and a one-block shift of
     each installation): no evaluated neighbour is determinately better; k neighbours were resolved by the lower
     bound."
   - Never write "locally optimal" without naming the neighbourhood.
   - Comparisons use the determinacy rule, as before.

## R4 — wording when the bound is valid but non-discriminating (ruled now, before D0)

**Rules for every case.**
- **Units.** Report the gap as G = UB − LB in k€ **and** as g = G/F(x̂), always both, and always beside what is at
  stake: the reference unit's value and the largest difference D between evaluated plans. A relative gap on its own
  is not admissible: g = 1e-3 of a 654 M€ cost is 654 k€, larger than every storage effect.
- **"Global optimum".** Use it only as "the optimum of the lattice planning problem with AC coordinated operation".
  Qualify the UB once: "an operating point feasible to the solver and consensus tolerances".
- **Decompose G where measured:**
  - network relaxation (rank gap, OLTC envelope);
  - degradation relaxation (calendar-only SoH against the realised SoH; plan-dependent, it grows with storage cycling);
  - salvage over-credit (zero for 2025 cohorts);
  - the UB's distance from the recourse optimum (not measured; part of G).
- **Exclusions.** Report how many evaluated plans the bound alone excludes, i.e. those with F_LB(x′) > UB(x̂).
- **Forbidden words:** "optimal", "near-optimal", "certified optimal", "the gap certifies the plan".

**Always true, in every case:** "No plan on the lattice can lower the total cost by more than G = … k€ below that of
x̂."

**H2, with P6 holding on variant C (amendment D-3):**
"The plan x̂ is within G = … k€ (g = … % of its total cost) of the optimum of the lattice planning problem with AC
coordinated operation. G is larger than the differences between the evaluated plans (at most D = … k€), so the bound
does not decide between them, and the choice of x̂ rests on the mesh-local certificate (Section …). The relaxation gap
of the network model differed by Δ = … k€ between the plan without storage and the reference unit, within resolution,
which is consistent with a gap that does not depend on the plan; the relaxed problem's optimal plan is x̂ [or: is x′,
whose evaluation is reported in Table …]. These observations support the ranking; they do not certify it."

**H2, with P6 failing:** keep the first two sentences, then add: "The gap differed by Δ between the two plans where it
was measured, beyond resolution, so the relaxed problem is not used to rank plans."

**H1:** "… G is smaller than the differences between plans; the bound excludes k of the m evaluated plans, and no plan
can improve on x̂ by more than G." For x̂ = 0, this gives the certified negative: "no lattice plan reduces the total
cost by more than G below that of the plan without storage."

**H3:** no gap claim in the abstract, highlights or conclusions. The bound is reported once, in the limitations, as
non-informative, with G stated.

**Why rule now.** In H2, the lower-bound effort pays only through the always-true sentence and the relaxed
multi-scenario and staged optima. The wording must not be chosen after seeing D0.

## R5 — noted

The author's set-up stands: Gurobi `QCPDual = 1` as primary, Clarabel per block whenever a log contains "failed to
compute QCP dual", and an IPOPT cross-check. Benders cuts and the R3.3 resolution by bound are anchored on the
Lagrangian dual bound with its margin (G-D0-5), never on a primal value.

## R6 — W184 (E3) and W185 (E2): held and re-scoped

- The paper carries one instance and one convention.
- E3 (C_NRF at x = 0) and E2 (ρ₀ × 0.5 and × 2 at the reference pair) are re-ordered on the five-block instance after
  the spec v2 freeze. Their Addendum 85/86 hypotheses are restated for that instance, and the launchers and key logic
  are reused.
- W186 and W187 are held with them; the campaign addendum re-scopes them against the R3 certificate run.
- W183i (zero-solve) completes as ordered.

## F4 — calendar-only SoH on the lower-bound side: adopted; Decision 2 reworded

**New wording of Decision 2:** "On the lower-bound side the state of health is replaced by its calendar-only value
φ_cal^age, a dispatch-independent upper bound (SoH = φ_cal^age · exp(−ΣD) with D ≥ 0). Available energy is bounded by
φ_cal^{age_mid} × rated (affine in x), and the salvage is credited at (φ_cal^{age_T} − SoH^Min)/(1 − SoH^Min): 0.447
for a 2028 cohort, 0.852 for a 2037 cohort, and 0 for 2025 by R2(iv)."

**Validity argument for the manuscript's proposition.** It is needed because a larger available energy also raises
the lower SoC limit and the initial state, so "larger capacity is looser" is not automatic.
- With SoC limits 0.10–0.90 of available energy, and the daily start and closure at 0.50, any power schedule feasible
  at E_av is feasible at every E′ ≥ E_av, with the stored energy shifted by 0.5 (E′ − E_av).
- The deviation from the start state stays within ±0.4 E_av ≤ ±0.4 E′.
- The closure-slack bound grows from 0.05 E_av to 0.05 E′.
- So enlarging available energy enlarges the set of feasible power schedules, which is all that the networks and the
  objective see.
- Condition to state: SoC^Min ≤ SoC^0 ≤ SoC^Max, with all three limits proportional to available energy.

**Recorded, not adopted.**
- A secant tightening SoH ≤ φ^age (1 − c·D) captures cycling fade validly. It is jointly linear in (E, throughput),
  because E·D is linear.
- But it couples the years and breaks the per-block separability of analysis §6.2.
- Consider it only if D0 shows that the degradation share of the gap is material.

## Amendments to the D0 spec (before T0.2; predictions below copied into spec §6 before any conic solve)

- **D-1. Variant A is not a lower bound of production.**
  - Fixing available energy at the certified point's values excludes dispatches that age the unit less (and so have
    more available energy), and the global optimum may use one of them.
  - A is a bound only conditional on the certified SoH trajectory. It is an attribution diagnostic and must never be
    quoted as a bound.
  - Relabel it in spec §3.3 and in the report.
- **D-2. Add variant C.** Calendar-only available energy per F4, which is the Phase-2 object. On SRP1's end-of-block
  convention: rated × φ_cal^{age at block end}. Same gates, same exports.
- **D-3. Read P6 per variant, and the absolute class on C** (C = R at x = 0).
  - The mechanism question (is the network gap plan-independent?) is read on A.
  - The decision relevance for R4 is read on C.
  - R is reported. Its differential contains the degradation relaxation by construction, which is plan-dependent, so
    a P6 failure on R alone says nothing about the rank gap.
- **D-4. Planner §10 rulings confirmed.**
  - **Identity gate at floating-point resolution.** A bar below a row's evaluation resolution tests rounding, not
    construction; the same-point Δ = 0 is the primary test.
  - **Envelope-rewrite class:** accepted.
  - **`sess_comp` dropped.** The convex hull of {p_ch · p_dch = 0} in the box is the triangle p_ch + p_dch ≤ S, which
    production already imposes (eq. 26), so nothing beyond the hull is lost. Any other nonconvex row is a stop.
  - **FrozenSMOPF loadability repair:** later, as housekeeping, under unwire-never-delete.
- **D-5. Transfer of the D0 class.**
  - D0 measures the superseded convention (SRP1, end-of-block). Its class transfers to the five-block instance on the
    reading that the network gap is per block and plan-independent.
  - The first Phase-2 LB evaluation on the new instance re-measures it at x = 0 and at the reference unit. This is a
    gate, not an assumption.

## Amendments to the convention-change spec (v3)

- **C2:** per R2. Horizon-end 2037 column; f = 1 base unless the author rules otherwise; floor 0; calendar-expired 0.
  The converter is credited pro rata outside the ESSO unless the author keeps energy-only. The cost-basis dispatcher
  serves the power cost too.
- **G-CC-3:** extended per R1.4: the € parity identity, slack safety ≤ 1e-3, and the diagnostic. It is run on a cell
  with a 2028+ cohort.
- **C5:** neighbourhood per R3: activation and deactivation, one-block shifts, closed-form size, and a hook for
  resolution by bound.
- **Freeze list:**
  - τ = 4,539.07 € stays frozen as an absolute threshold. The new instance's reference value is reported beside it;
    re-setting τ needs a ruling.
  - Add the salvage items of R2(vii).

## Predictions (expert; recorded before any run)

- **G-CC-3 (marginal salvage per MWh of cell throughput, 2037 basis, f = 1):**
  - 0 exactly for a 2025 cohort;
  - ≈ 1 €/MWh for a 2028 cohort and ≈ 5 €/MWh for a 2037 cohort, each within a factor of 2;
  - the slack-safety gate holds.
- **G-CC-2:** x = 0 network blocks bitwise identical. The ESSO is value-inert: S ≡ 0, and C^conv is outside the ESSO
  and 0 at x = 0.
- **D0 variants (unit cell; at x = 0, R = A = C identically):**
  - per group, gap_A ≤ gap_C ≤ gap_R;
  - summed over the 12 groups, gap_R − gap_C ∈ [10, 50] k€ and gap_C − gap_A ∈ [5, 40] k€;
  - P6 holds on A; on R it fails by about the degradation share (≥ 15 k€).
- **C1 A/B (campaign, later):**
  - at a 2028+ unit: EFC/day lower by ≤ 3 %, terminal SoH higher by ≤ 0.005, F changing by ≤ τ;
  - at a 2025-only plan: the ESSO dispatch is identical to solver tolerance. Bitwise identity is not predicted.

## Stops

As Addendum 87, plus:
- the G-CC-3 slack-safety check fails;
- any D0 artefact quotes variant A as a bound;
- salvage parameters change after the freeze.

## Author decisions requested before the freeze

1. **R2(ii) base recovery fraction.** f = 1.0 with the continued-use meaning (expert's ruling), or a cited resale
   value (Planner's recommendation). If a cited value is supplied, it is a sensitivity row in either case.
2. **R2(v) converter.** Credited pro rata (expert's ruling) or energy-only (Planner's recommendation).

## Planner order (zero solves)

1. Amend `P5_16_D0_SPEC_v2.md` per D-1 to D-5 and copy the D0 predictions above into §6 before T0.2. T0.1b proceeds
   as ordered and adds variant C to its builds and exports.
2. Write `P5_16_CONVENTION_CHANGE_SPEC_v3.md` per the C2, G-CC-3, C5 and freeze-list amendments, with R2(ii) and (v)
   marked pending the author.
3. Report the two spec diffs. Nothing else runs on the convention change until the author answers R2(ii) and (v).
