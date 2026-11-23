# S56 storage-energy decomposition, closure convention and the R/4 rule (2026-11-23)

**What this note answers:** Expert ruling E2, the closure alignment, and the R/4 composition.
**Basis:** the Advisor's design; the Planner checked the network rows at HEAD (`model_construction_helpers.py` l. 945–1006,
2373–2384).
**Status: proposal.** Items D1–D4 need the Expert's confirmation. Nothing is built yet on the network side.

## Decisions needed (Expert)

| # | decision | recommendation |
|---|---|---|
| D1 | **Decomposition** | **(a), gross-flow aggregate rows.** On multi-cohort node-years, deactivate only the network aggregate complementarity row `sess_comp`. All other network storage rows stay, and P/Q consensus is unchanged. Add a per-cohort row p^ch_c + p^dch_c ≤ s_c in the ESSO. One formulation key governs both sides. One cohort: today's rows unchanged |
| D2 | **Closure convention** | **Every energy state returns to 0.5·E_av; the priced bounded slack is a numerical-feasibility device that must sit at its bound at acceptance.** One cohort: today's network row unchanged. Several cohorts: exact per-cohort closure in the ESSO, with feasibility carried by the existing ESSO P slack and the new Q slack; the network aggregate closure is then implied with zero slack. **New gate:** network closure slack ≤ τ_cl on every storage evaluation |
| D3 | **R/4 composition (one budget per pair)** | V_pair = S(x) + S(x′) + \|PR(x) − PR(x′)\| + G(x) + G(x′) ≤ 2,500 €. **Determinate** iff \|ΔF\| ≥ 10,000 €, V_pair ≤ 2,500 € and every physical gate passes for both. A **sum**, not quadrature (the errors are systematic) and not a per-channel max (the Expert excluded per-channel allowances) |
| D4 | **Cancellation guard and the role of PR** | A(x) + A(x′) ≤ R/2 = 5,000 € (the absolute priced residual), so that a signed cancellation cannot make a pair determinate on its own. PR enters only to **withhold** determinacy: it never ranks or replaces an objective. Confirm this reading of "diagnostic only" |

## Why (a)

**Where the Expert's objection sits in the code.** The network SoC row is already gross: ΔE = η_ch·pch − pdch/η_dch
(l. 970–992). Only `sess_comp` (pch_hat · pdch_hat ≤ 1e-4, l. 945–967) forces the net form, and that net form omits
the circulation loss.

**The cost of keeping the rows unchanged.** Any circulation would have to be absorbed by the closure slack:
- the slack absorbs λ·X, with λ = 1/η_dch − η_ch = 0.0717 and X the energy transferred;
- it is priced at 1,000 €/MWh per network copy, so about 72 € per MWh transferred;
- the real loss costs about 4 €/MWh;
- the resulting set is neither an inner nor an outer approximation.

**With `sess_comp` off on multi-cohort node-years,** every cohort-feasible point maps to a network-feasible point. Set
pch = Σ p^ch_c, pdch = Σ p^dch_c and soc = Σ e_c:

| network row | why it holds |
|---|---|
| SoC recursion and initial value | it is the sum of the cohort recursions (exact, because η is uniform) |
| window | it is the sum of the cohort windows |
| closure | it holds with slack 0 |
| charge plus discharge within the rating | it follows from the new per-cohort sum rows |
| converter circle | it holds by the triangle inequality |

- **Energy accounting** is exact.
- **The set stays relaxed** only in the per-cohort windows, the converters and the circulation capacity, and the ESSO
  closes those at consensus with zero slack.
- **Convexity:** removing a bilinear row makes those network subproblems more convex.
- **Costing bias:** none at consensus. At a finite ESS residual it is one-signed and enters G and PR.

**Why consensus stays on (P, Q):**
- **C/D channels are unnecessary.** Network cost sees storage only through pnet and qnet, and the day's total loss is
  a function of P: η_ch·ΣP⁺Δt − ΣP⁻Δt/η_dch = λ·(transfer + within-cohort leak).
- **C/D at fixed P is non-unique.** Consensus on it would add flat directions.

**Why not (b), energy physics only in the ESSO:**
- **No valid state-free energy constraint exists** other than the converter circle.
- **It breaks the comparison.** Applied only on multi-cohort node-years, staged plans would lose all intertemporal
  information in the networks while unstaged plans keep it, which is exactly the comparison the study makes.
- **Convergence.** All energy pricing would move to the slow ESS channel.
- **Fallback (N2):** per-cohort rows in the network models are the exact option, but the ADMM plumbing carries one unit
  per bus.

**Build rule.** The switch lives inside `configure_shared_ess_operational_state`, which re-activates the storage rows
at eight call sites.
- **Read:** a plain per-unit model attribute, default False, following the step-1 attribute precedent.
- **Set True only** when the node-year has ≥ 2 active cohorts and the key is not legacy H3.
- **Absent attribute:** today's model, so fixtures and NL identity are untouched.

## Within-cohort complementarity

**Today the ESSO has no complementarity row,** only ε = 1e-5 on throughput. Once the ESSO owns cohort SoC and closure,
dumping energy by within-cohort simultaneity costs the same as inter-cohort circulation. With TV on, the 2025 cohort's
throughput carries no TV cost. So ε alone may not restrict it (hypothesis H_B).

**Recommendation:**
- keep ε;
- add the per-cohort sum row;
- build a per-cohort relaxed complementarity row (the `sess_comp` form), deactivated;
- decide before the pilot spec from two checks:
  - a **dump positive control:** a millisecond ESSO solve, 2025 + 2030 cohorts, κ on, a requested net energy surplus;
  - a **sign screen:** the `sess_soc_def` duals in preserved solved models, zero-solve.
- **Decision rule:** if dumping has value and H_B holds, activate the row for the whole study. Otherwise keep it as a
  detector and a gate.

**Detectors** (per node, year, day):

| detector | definition |
|---|---|
| within-cohort simultaneity | L = Σ_c Σ_t min(p^ch_c, p^dch_c)·Δt, and max_t l/s_c |
| inter-cohort transfer | X = Σ_t min(Σ_c p_c⁺, Σ_c p_c⁻)·Δt |
| losses | λ·X and λ·L |
| identity assertion | λ·(X + L) = η_ch·ΣP⁺Δt − ΣP⁻Δt/η_dch, to 1e-9 |

**Transfers are reported, not gated** (E2).

## The R/4 terms (€ of total discounted planning cost)

- **S(x), objective stability:** the range of F over the last W cycles, or the validated extrapolation gap.
- **PR(x), signed priced consensus residual** over all channels jointly: per-scenario interface ΔP/ΔQ/ΔV and the
  three ESS channels, each times its consensus multiplier in physical units.
- **G(x), absolute infeasibility that does not vanish at the ADMM limit:**
  - ESSO P and Q slacks, priced at the ESS multiplier, not at the penalty;
  - the within-cohort leak;
  - the network closure slack, priced at its row dual.
- **A(x):** the absolute form of PR.

**The 12.5 kW ceiling** is an independent per-scenario physical gate.

**Calibration** (steps 3 and 5, centralized pairs):
- the working level is the loosest at which the real error |ΔF_ADMM − ΔF_ref| ≤ R/4 and V_pair ≥ the real error at
  every later cycle;
- otherwise γ = max(real/V) is fixed before the pilot and coverage is reported on held-out pairs.

**If unattainable,** the following are reported before R is touched:
- the smallest V_pair reached and its composition;
- R_eff = 4·V_pair;
- the dominant term;
- the number of pairs whose class would change.

## Validation additions

| test | prediction |
|---|---|
| zero-solve aggregation with transfers, against network rows with `sess_comp` off | violation ≤ 1e-12 |
| the same with `sess_comp` on | closure residual = λ·X to 1e-12 |
| **circulation example** (0.25/1 + 0.25/0.5). Hour 1: A discharges 0.15 MW, B charges 0.15 MW. Hour 2: B discharges 0.13968 MW, A charges 0.16108 MW. Both cohorts return to their initial energy exactly | P = (0, +0.02140) MW; X = 0.28968 MWh; loss 0.02076 MWh; network (a) slack 0; today's rows need 0.02076 MWh of closure slack |
| single-cohort NL identity and the SRP1 two-cycle gate | bitwise |
| row inventory, 2025 + 2030 plan | `sess_comp` active only in the 2025 blocks at that node; all active under the legacy key |
| **centralized per-cohort comparator:** merged network plus one shared-ESS unit per cohort at the bus (the network model accepts several units per bus; `combine_networks` empties them, so the harness appends them after the merge), κ = 0 | \|C_ADMM − C_central\| within the two-block budget |
| **E3 non-uniqueness:** within a block, three centralized solves with ± tie-breaks | C equal to solver tolerance; splits differ |
| **E3 across blocks:** extreme splits pushed through degradation, later block re-solved | \|ΔC\| reported against R/4 |
