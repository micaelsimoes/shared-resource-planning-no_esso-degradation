# P5.16 convention change — salvage in F, mid-block SoH, five 3-year blocks, general lattice (spec v1, for Advisor review)

Planner, 2026-10-30. Authority: author's decisions 2–5 and agreements R1/R2 of 2026-10-30
(`PLANNER_BRIEF_ADDENDUM87_DRAFT_2026-10-30.md`); expert confirmation of R1 and ruling R3 still requested. **Draft**:
the Advisor reviews before any Worker task; code is written only after the review; **no evaluation runs** under this
spec — the new instance's references are campaign cells authorized later.

## 1. Changes (each switch-controlled; defaults reproduce today's behaviour bitwise)

### C1 — terminal salvage term in the ESSO objective (R1 option (a))
- Today (`shared_energy_storage_data.py` l.844–853): `model.objective = feasibility_penalty`; `salvage_value` and
  `salvage_credit = −salvage_value` are reporting Expressions only.
- Change: `objective = feasibility_penalty + κ · salvage_credit` where κ puts the € credit on the **same objective
  scale as the network costs** in the ADMM (the common objective scale resolved by `_resolve_esso_al_scale` /
  `objective_scale_used`; the networks' per-block weights and discounting are already inside their objectives, the
  salvage is already discounted to y₁ by `_get_terminal_discount_factor`). Switch `salvage_value.in_objective`
  (default **false** = today).
- Consequence to state: dispatch becomes salvage-aware — the ESSO trades throughput against terminal health at the
  marginal salvage value; this supersedes the governing decision "no explicit degradation cost" (expert R1).
- The planning objective F = I(x) + Q(x) − salvage(x) on the UB side; Q keeps its gross convention and the salvage is
  reported beside it (the net column already exists), so every table states both.

### C2 — salvage basis (R2)
- Today: `cost_basis = EXPECTED_INSTALLATION_ENERGY_COST`, `energy_recovery_fraction = 1.0`,
  `recycling_floor_fraction = 0.0`, remaining-life fraction at terminal, health normalised above `minimum_soh`,
  energy only (converter not credited).
- Change: new `cost_basis = EXPECTED_TERMINAL_ENERGY_COST` — the expected energy unit cost at the horizon end.
  **Open point for the author/expert:** the cost file carries 2025–2037; "horizon end" = the last representative
  year's expected cost (2037) or an extrapolated 2040 value — state which, and the recovery fraction with its
  citation (1.0 is not defensible as a market value; typical second-life/resale fractions are well below 1).
- LB side uses the same basis at SoH ≡ 1 (maximum credit, affine in x).

### C3 — mid-block SoH as the baseline point
- `available_energy_soh_point = 'mid'` in the ESS params of the new instance (W20 mechanism, l.598–617; salvage keeps
  terminal SoH, l.885–888). The floor stays an end-of-block condition — stated in the manuscript.

### C4 — five-block single-scenario instance
- New case file beside `SRP1.json`: `Years = {2025: 3, 2028: 3, 2031: 3, 2034: 3, 2037: 3}`, `NumMarketScenarios = 1`,
  the same days/instants, network data of the 3 × 3 instance's years (that file is the template — the Worker names
  its path and hash). Calendar life 15 y = five blocks (`tcal_norm = 5`). New ESS params file with C2/C3 set.
  `SRP1.json` and `SRP1_ESS_Params.json` are **not** modified (every committed result pins them).

### C5 — general lattice in the harness layer
- Production already takes `candidate_solution['investment'][node][year]['s'|'e']` per node **and year**; the
  restriction to one year is the harness's (`p515_s44_campaign_harness.canonical_candidate` with a single
  `investment_year`; `p515_s47_phase_b_record.Lattice` with z ∈ ℤ⁷ and the year as ordinal).
- New lattice module (new file, `p516_lattice.py`): z ∈ ℤ^{2|N||Y|} (30 on the five-year set), per (n, y) a pair
  (zP, zE) in granules 0.25 MVA / 0.5 MWh; admissibility: zP = 0 ⇔ zE = 0; zP ≤ zE ≤ 2 zP (2–4 h) per cohort;
  **cumulative** rated energy per node and block Σ_{y_inv ≤ y} E_{n,y_inv} ≤ 5 MWh (all cohorts live within the
  horizon); budget Σ_y (1+d)^{−(y−y₁)} Σ_m ω_m (c^S_{m,y} P_{n,y} + c^E_{m,y} E_{n,y}) ≤ €1 M (the
  `shared_energy_storage_data.py` expression). Canonical form: the per-(node, year) map; x = 0 is all zeros.
  Neighbourhood for the certificate: coordinate unit moves ±1 in one coordinate plus the activation moves of
  inactive cohorts (P 0 → 0.25 with E ∈ {0.5, 1.0}) — **pending R3**; the ∞-norm box completion is not carried
  over (3³⁰ points).
- New canonical candidate shape and **evaluation-key schema v2** (`{'nodes': {'5': {'2025': [s, e], …}}}`), with
  the x = 0 alias; old keys are not translated — every storage cell is superseded by decision 3 anyway.

## 2. Gates (zero solves except G-CC-3's declared count)

- G-CC-1 **Switch-off identity on SRP1 (unchanged case file):** with `in_objective = false`, `cost_basis` as today,
  `'end'`, the ESSO and every network block of the x = 0 **and** the unit cell build bitwise identical to the current
  production head (NL digests, the W183 method) — proves C1–C3 are inert when off. Every preserved fixture
  unpickles (unwire-never-delete).
- G-CC-2 **x = 0 inertness under the new convention:** with C1–C3 on, x = 0 builds bitwise identical to G-CC-1's
  x = 0 (salvage and storage rows inert at zero capacity); the unit cell differs **only** in the ESSO objective rows
  (C1), the salvage expression (C2) and the available-energy rows (C3) — the diff enumerated.
- G-CC-3 **Salvage-term calculus (declared ≤ 4 ESSO LP/NLP solves, guarded):** on the unit cell's ESSO alone, the
  marginal objective change per MWh of extra throughput equals κ × ∂salvage/∂throughput by finite difference to
  ≤ 1e-6 relative; sign negative (more cycling, less salvage). Scale κ reproduces the network blocks' € → scaled
  conversion exactly (equal to the recorded objective scale).
- G-CC-4 **Lattice:** the new module reproduces the old `Lattice.reasons`/`domain` on the 7-variable common-year
  slice (every old admissible point maps to an admissible general point with one nonzero year and vice versa);
  cumulative capacity rejects a two-cohort plan exceeding 5 MWh at a node; budget on a staged plan equals the
  production `investment_cost` expression to 1e-9; canonicalization idempotent; key schema v2 round-trips.
- G-CC-5 **Instance build:** the five-block case file loads; 80 network blocks + 3 ESSO build at x = 0 and at the
  unit (zero solves); `tcal_norm = 5`; block weights 3 years; the x = 0 build's NL digests recorded as the new
  instance's G0.

## 3. Not permitted
Modifying `SRP1.json`, `SRP1_ESS_Params.json`, `SRP1_ESS.xlsx`; deleting any symbol; changing the network models or
`model_construction_helpers.py`; any ADMM run; any write under an existing results root.

## 4. Questions for the Advisor
1. Is κ (the € → ADMM objective scale of the salvage term) correctly identified, and does the ESSO's proximal/AL
   structure need a corresponding adjustment (the ESSO now has an economic objective, not only a penalty)?
2. Any convergence or conditioning risk from C1 on the ESSO subproblem (terminal SoH enters through the exp chain;
   the objective becomes nonlinear in the SoH variables)?
3. Is the LB-side maximum-credit form exactly consistent with C2 as written (same basis, SoH ≡ 1, remaining-life
   fraction per cohort)?
4. Is G-CC-3 the right calculus gate, or should it be a two-cell ADMM A/B (salvage on/off at the unit) — which
   would be a campaign item, not a zero-solve gate?
5. Anything in the general-lattice admissibility that production enforces differently (e.g. `max_capacity` semantics
   on cumulative rated vs available energy)?
