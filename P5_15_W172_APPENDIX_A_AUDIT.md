# P5.15 W172 — equation-vs-code audit of the pasted Appendix A (main.tex l. 1404–1679, Overleaf `407f8df`) and the DN interface ratings for §3.5

**Worker, 2026-10-08. ZERO SOLVES, zero model builds, zero pickle loads. Read-only on production code, specs, case
files, the frozen JSON and the manuscript clone.** Helper `p515_s53_w172_appendix_a_checks.py` reads only JSON, text and
`git` output; it blocks `pickle` before any import and asserts at exit that no production module was imported (pyomo,
`shared_resources_planning`, `network*`, `shared_energy_storage*`, `model_construction_helpers`, `admm_*`,
`helper_functions`, `definitions`). Exit 0. W100 typing test PASS (129,310 files, 0 failures).

## Blocked on Planner

Nothing blocks the audit itself. Three items need a ruling or a forward to the expert or author:

1. **One H row, A6-1.** l. 1674–1675 says "At a single scenario all of these terms vanish identically", and the
   interface settlement is one of "these terms". The code carries that settlement at weight 1 in every TSO and DSO local
   objective of every single-scenario evaluation (contracted part = whole settlement). Only its deviation part vanishes.
   This is a wording ruling for the expert; no formulation is proposed here.
2. **CONFIRM l. 1592 (e) differs.** This repeats Round-1 Decision 9, rechecked here.
   - Recovery `acceptable_tol` 1e-4 / `acceptable_iter` 1 is set for the **TSO only** (`case9_params.json`).
     - `case33_1_params.json` has no `recovery_options` key.
     - `case33_2/3` have `{}`.

     So a DSO retry is a cold start with the primary options (`acceptable_tol` 1e-4, `acceptable_iter` 5).
   - The ESSO recovery runs with `acceptable_tol` **1e-9** and `acceptable_iter` 1. The cause is that
     `ESSO_TOL_OVERRIDES` is re-applied over the ESS file's 1e-4 (`shared_energy_storage_data.py:1285`).
   - The appendix text prints no recovery values: l. 1588 says "a documented sequence".
   - §3.5, where the values are to go, does not exist at `407f8df`: §3 has 3.1–3.4 only.
3. **Outside Appendix A, for §3.5.** The manuscript prints one IEEE 33-bus branch table for all three ADNs
   (`tab:cs1_ieee33_branches`, l. 1776–1830). It gives branch 1 (1→2, the interface transformer) **200 MVA**. That equals
   `case33_1` only: `case33_2` has 100 MVA and `case33_3` has 150 MVA. Every other branch rating in the table equals all
   three case files (81/81). The l. 1733 paragraph says the ADNs differ only in RES capacity and flexibility.

## R^I for §3.5

`get_interface_branch_rating()` (`network.py:83-94`) sums `branch.rate` (the case-JSON field `rating`, MVA) over the
in-service branches incident to the DN reference node. The code divides it by the network `baseMVA` (100 MVA, TN and DN)
for p.u. (`shared_resources_planning.py:5148` TSO, `5417` DSO, `4222` TSO Δ bound). In each DN the only such branch is
transformer 1 (bus 1–2). The value is **fixed per DN, not year-dependent**: identical in all 17 year files, 2025–2045,
which cover every year either instance loads.

- **TN node 5 — `case33_1`:** R^I = **2.0 p.u. = 200 MVA** (base 100 MVA). Source: `data/SRP1/case33_1/case33_1_<year>.json`
  `transformers[branch_id 1].rating`. Blobs at HEAD: 2025/2028 `5922cb45`, 2030/2031/2034 `ae77ca16`, 2035/2037
  `58211b8d`.
- **TN node 7 — `case33_2`:** R^I = **1.0 p.u. = 100 MVA**. Blobs at HEAD: 2025/2028 `c24bfddc`, 2030/2031/2034 `d1924a42`,
  2035/2037 `bb4420b1`.
- **TN node 9 — `case33_3`:** R^I = **1.5 p.u. = 150 MVA**. Blobs at HEAD: 2025/2028 `ee92dfbc`, 2030/2031/2034 `eaac8137`,
  2035/2037 `fb3a2814`.

The function is a pure read, so the values were read statically. No network object was built.

Cross-check: every per-cycle `worst_pf_primal_rating` in the committed `g_s39_D.json` (v6 and 3 × 3) records the same
values. By node, as node|MVA × count: 5|200 × 7,269; 7|100 × 5; 9|150 × 12.

## Differing rows

Impact as in W171b: **H** — a referee re-deriving from the text gets a different model or number; **M** — the text is
imprecise but does not describe a different model; **L** — notation or wording. Line numbers are main.tex at `407f8df`
and code at repo HEAD (production identical to all 46 campaign heads, see "Scope and configuration"); every code
location was found by exact-text anchor (105 anchors in `w172_checks.json` `F_code_anchors`).

### Preamble (l. 1407–1412)

| ID | lines | text | code (file:line; name; spec) | code's form in the manuscript's notation | how it differs | impact |
|---|---|---|---|---|---|---|
| P-1 | 1410 | "run as a Gauss–Seidel sweep (DSOs, then TSO, then the storage agent)" | `shared_resources_planning.py:3115-3210`; `update_*_and_solve` 6413-6427, 6084-6099, 6658-6681; comment 8574-8586 | interface channels: TSO solves against the DSOs' fresh copies (Gauss–Seidel). Storage channel: DSO, TSO and agent all solve against the same z^k; z and the three storage duals are updated only after the agent | "Gauss–Seidel" holds for the interface channels only; the storage channel is the parallel global-consensus form. Alg l. 1637/1641/1644 state the targets correctly | L |

### A.1 — agents, blocks, consensus variables (l. 1417–1438)

| ID | lines | text | code | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| A1-1 | 1421–1423 | weighted, scaled block objectives "so that … the sum over blocks is the discounted operating cost of (recourse_value)" | `_get_operational_recourse_components` 1234-1306 (gross Q at 1302-1304); `network_data.get_primal_value` 96-103 | Q = Σ_b w_b f_b − (contracted settlement) − (voltage pin), where f_b is the block's base objective | the sum over blocks equals Q only after the exclusions §2.2.7 and A.6 state. The contracted part cancels only at exact interface consensus; the voltage pin, which exists only in the 3 × 3 instance, does not cancel | L |
| A1-2 | 1427–1428 | multi-scenario: "the *expected* interface quantities, Σ_o ω_o P^I_{i,o,t}, of Subsection commitment" | `dn_/tn_interface_expected_pf_p_def` (`model_construction_helpers.py` 2416-2509); 3 × 3 spec `231558f0` | P̄^I_{i,t} = Σ_{m,o} ω_m ω_o P^I_{i,(m,o),t} = Σ_{s∈Ω_M×Ω_O} ω_s P^I_{i,s,t} | the sum runs over market × operation scenarios. §2.3.6 (l. 876) now writes Σ_{s∈Ω_M×Ω_O}, and the appendix shorthand Σ_o ω_o (with o = operation scenario, l. 870) contradicts it | M |
| A1-3 | 1437–1438 | κ^E "puts them [the agent's AL terms] in the same units as the network blocks' scaled objectives" | `update_shared_energy_storage_model_to_admm` 5455-5529 (docstring 5457-5466); TSO/DSO 5136-5142, 5401-5407 | network: (w/σ) f_a + AL. Agent: f_E + κ^E·AL, which has the same argmin as f_E/κ^E + AL | the network AL terms are unscaled; κ^E puts the agent's f_E on the same footing as a median block's (w/σ)f. The AL terms themselves are not put in other units | L |

### A.2 — local problems (l. 1447–1500)

| ID | lines | text | code | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| A2-1 | 1485–1489 vs §2.3 l. 851 | eq. (admm_esso_local) writes κ^E Σ[λ(·)/(2S^ref) + ρ^E/2 (·)²] and never names ℒ^{E,P}, ℒ^{E,Q}; eq. (esso_objective) l. 851 adds Σ(ℒ^{E,P} + ℒ^{E,Q}) "(\ref{app:admm_updated_implementation})" without κ^E | `shared_resources_planning.py:5512-5518` | κ^E multiplies both the dual and the ρ term in the agent | ℒ is undefined in Appendix A, and the §2.3 reader sees no κ^E. The two agree only if ℒ is read as including κ^E (W171b "Not confirmed" item, now locatable) | L |

### A.3 — consensus, dual and penalty updates (l. 1510–1555)

| ID | lines | text | code | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| A3-1 | 1528 | the z-update "is the minimiser of the sum of the three agents' storage terms over z" | z-update 8667-8691; agent AL with κ^E 5515-5518; dual update 8696-8698 | z = Σ_a(ρα²x_a + αλ_a)/Σ_a ρα², the minimiser of Σ_a[λ_a α(x_a − z) + (ρ/2)α²(x_a − z)²] **without** κ^E | with the agent's terms as written in (admm_esso_local), which carry κ^E, the minimiser would weight the agent's copy by κ^E (≈ 2.3 × 10⁵), which is a different z. The equation (admm_z_update) itself matches the code | M |
| A3-2 | 1542–1543 | "No consensus or dual update is made for a block or node in which any of the agents' solves failed in the cycle; that block keeps its previous values." | interface 8437, 8460, 8485-8489; storage 8527, 8545, 8563, 8602-8609 | each agent's own copy x_a is updated whenever **its** solve succeeded. Interface duals need TSO **and** DSO success for (y,d) and are updated before the agent solves, so an agent failure does not block them. z and the three storage duals need TSO, DSO and agent success | per-agent copies are still refreshed, and the interface update does not depend on the agent | L |
| A3-3 | 1554–1555 | "all ρ_g are frozen from the cycle after the first residual pass" | hooks `p515_s53_w118_resettle_hooks.py:526` (`c > first_pass`); W101 continuation hooks | fresh runs: from k0 + 1. The two v1 references continued from an earlier run (`ref:7aa017f0`, `ref:bd504ecf`): from N_old + 1 (W168: formally from 133, not 124) | the continuation case is omitted here; §2.2.7 l. 618 states it | L |

### A.4 — stopping test, acceleration and tail (l. 1564–1590)

| ID | lines | text | code | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| A4-1 | 1565–1566 | s_E = ‖ρ^E α Δz‖ | `get_admm_boyd_residual_metrics` 7217-7247 | s_E = ‖(ρ_a α Δz)_{a∈{TSO,DSO,E}}‖ = √3 ρ^E α ‖Δz‖, plus the TSO-proximal entries γα Δx_TSO (= 0, γ = 0); n counts the three agents | the agent sum, factor √3, is unstated for s_E. "Over all agents" is attached only to r_E | L |
| A4-2 | 1570–1571 | ε_dual = √n ε_abs + ε_rel ‖λ‖ | 7174, 7203, 7224, 7261 | ‖λ‖ = the **DSO-side** dual only for the V and PF channels (λ_TSO = −λ_DSO); all three agents' duals for the storage channel | which duals enter ‖λ‖ is unstated | L |
| A4-3 | 1574–1576 | "The single-scenario evaluations … were stopped by the certification rule …, which uses the first passing cycle k0 as its starting point." | `settling_criterion_v6`; v6 hooks (caps: gated N_old + 100; ungated min(k0 + 109, 300)); frozen tables `590088fe` | 12 single-scenario cells in the tables ended uncertified at a cap (`rule_cap`/`cap`). The rule's k0 resets on a lapse or failed cycle, while the holds stay from the first pass | "stopped by the certification rule" covers its cap and the uncertified form only implicitly. "First passing cycle" is the hold origin, not always the rule's k0 (W171b T2-4; §2.2.7 l. 618 states the reset) | L |
| A4-4 | 1578–1579 | AA "applied to the pair (z, λ/ρ) of every channel" | `admm_anderson_acceleration.py:279-319` (`collect_w`), 321-364 (`write_back_w`) | interface channels: "z" = the TSO's copy and u = λ_DSO/ρ (λ_TSO written back as −λ_DSO). Storage channel: z and three per-agent u_a = λ_a/ρ | the interface channels have no global z; the AA iterate uses the TSO copy | L |
| A4-5 | 1586–1587 | "the production setting restores the default tolerance otherwise" | `_apply_convergence_depth_tail` 7472-7505; `case9_params.json` (`compl_inf_tol` 5e-4); `case33_*_params.json` (no key → IPOPT default 1e-4); `admm_parameters.py` `convergence_depth_tail` default off; v6 spec `inputs_in_force_now.configuration_now.convergence_depth_tail`, 3 × 3 spec `configuration.convergence_depth_tail` | restores each holder's case-file state: TSO 5e-4, DSO the IPOPT default 1e-4. The tail exists only when enabled; every campaign enabled it in its spec | "default" is the case-file value for the TSO. That the tail is a declared option rather than the case-file default is unstated | L |
| A4-6 | 1582–1583, 1587–1588 | AA off "in the certifying regime, from the cycle after k0 on"; "the certifying regime holds the tail on from k0" | hooks l. 526 (`c > first_pass`), `w_next` (tail next-state forced True when held), `w_aa` l. 722 | AA, tail and ρ holds all start at k0 + 1 in fresh runs and at N_old + 1 in the two continued references. The tail first acts at k0 + 1, because production's own next-state already turns it on after the passing cycle k0 | "tail on from k0" is one cycle early. The continuation case is omitted, as in A3-3 | L |
| A4-7 | 1588–1589 | "A local solve that does not reach an optimal status is retried under a documented sequence of solver settings" | `network.py:771-790` (`_is_recoverable_network_failure`), 964-1006 in `_run_smopf` 946 (tiers); `shared_energy_storage_data.py:1218-1360`; case files | retry only on maxIterations, infeasible or internalSolverError. Tier 1: cold start with `recovery_options` minus `hessian_approximation` (TSO: `acceptable_tol` 1e-4, `acceptable_iter` 1; DSOs: none, so primary options; agent: `acceptable_tol` 1e-9 after `ESSO_TOL_OVERRIDES`, `acceptable_iter` 1). Tier 2: as tier 1 plus `mu_strategy` adaptive. Other non-optimal results, e.g. SolverStatus.warning, are not retried | the trigger set and the settings are not in the paper ("documented" points nowhere; §3.5 absent). See CONFIRM l. 1592 (e) | L |

### A.5 — algorithm (l. 1602–1658)

| ID | lines | text | code | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| A5-1 | 1605–1606 | "the plan x itself never enters a network model except through these capacities" | `create_transmission_network_model` 4204-4206; `create_distribution_networks_models_sequential` 4360-4362 (`update_data_with_candidate_solution(candidate_solution['total_capacity'])`) | the initialisation solves use the plan's capacities directly. From cycle 1 the published S^Av, E^Av overwrite them (2980, 6075-6082, 6409-6411) | true for the ADMM cycles, not for the initialisation, which Alg l. 1620 itself describes | L |
| A5-2 | 1628–1630 | "z ← the average …; initialise the interface duals from the initial solutions; ρ_g ← ρ_{g,0}; convert every model to its ADMM form" | 2925-2960 | order in the code: settlement weight 1 and row-18 activation (2925-2926) → σ, κ^E → conversion (2931-2933, ρ_0 set here) → z = mean of the three copies (2934, 4891) → one interface dual-ascent step from λ = 0 (2951-2960). The storage duals start at 0 | the conversion precedes the z and dual initialisation. "From the initial solutions" means one step λ_a = ρ_0 (x_a − x_ā)/b. Storage duals at 0 are unstated | L |
| A5-3 | 1648–1649 | "record the objective, residuals, gap and solve statuses of the cycle" | production `admm_diagnostics` 3431-3661 (no gap field); SRP1 hooks: per-cycle priced gap t_sum (W118 capture); 3 × 3: terminal `interface_settlement_detail_s31c.json` only | the per-cycle gap is a harness capture of the SRP1 campaigns; production records none | which layer records the gap is unstated | L |
| A5-4 | 1650–1655 | order: residual test → "If … stopping rule … exit" → "Balance ρ_g; apply the Anderson step; set the tight tail for the next cycle" | 3258-3280 (AA), 3384-3390 (exit predicate), 3394-3398 (tail next state), 3406-3409 (ρ balancing), 3419-3428 (AA memory cleared on a ρ change), 3697 (publish), 3710-3712 (break) | the AA step precedes ρ balancing; then the memory is cleared if any ρ changed. AA, tail next-state and ρ balancing all run before the exit `break` | with the text's order, a cycle whose ρ changes would clear the memory before the AA step, so no extrapolation would occur. The code extrapolates first. The exit-order difference does not affect returned values (AA is off on passing cycles; under the holds every hold is on) | M |

### A.6 — scenario mechanics (l. 1667–1675)

| ID | lines | text | code | code's form | how it differs | impact |
|---|---|---|---|---|---|---|
| **A6-1** | 1671–1675 | "The interface settlement is carried at full weight: its contracted part is removed …, its deviation part is kept. … **At a single scenario all of these terms vanish identically.**" | `model_construction_helpers.py:1701-1702` (`interface_settlement_weight`, `interface_settlement`), 1729-1756 (`interface_energy_settlement`), 1823 (in `objective_function_rule`), 1917 (only row 18 and the pin are skipped at one scenario); `shared_resources_planning.py:5050, 5340` (weight 1 in the ADMM path), 1251-1304 (contracted part excluded from Q) | at one scenario, every DSO local objective carries +Σ_t π_t·baseMVA·P^I_{i,t} and the TSO's carries −Σ_i Σ_t π_t·baseMVA·P^I_{i,t} (active power, weight 1). The contracted part is the whole settlement; the deviation part is identically 0 | the row-18 charge and the voltage pin vanish at one scenario; **the settlement does not**. It is a term of f_a in every SRP1 local problem (it prices the DSO's import in its own objective) and shapes every iterate. It is excluded from Q only. No other appendix line puts it into f_a | **H** |
| A6-2 | 1668–1669 | "the TSO represents each DN by that expected exchange with a scenario-free adjustment as its only interface freedom" (under "When several … scenarios are present") | `create_transmission_network_model` 4234-4273 | TSO interface = P^{0}_{DSO,i,t} (the DSO's **initialisation** expected exchange, fixed in `pc`) + Δ_{i,t}, with Δ scenario-free and bounded by ±R^I_i. The same construction holds at one scenario | "that expected exchange" is the initialisation one, held fixed. The bound ±R^I_i and the single-scenario applicability are unstated. The bound never binds: R^I is 100–200 MVA | L |

## W166 coverage

**Appendix-A rows A4–A35** (W166 audited the submitted text). Status in the pasted appendix:

| W166 | status at `407f8df` | where |
|---|---|---|
| A4 TSO initial interface = the DSO's initial solution | stated | Alg l. 1623–1625; A.6 l. 1668 (row A6-2 L) |
| A5 k ≤ k^max | stated | Alg l. 1614, 1633 |
| A6 DSO targets = TSO copies; storage target z | stated | Alg l. 1636–1637 |
| A7 DSO solves | stated | l. 1637 |
| A8 no dual update after the DSO solve | stated | l. 1528–1529, Alg l. 1643 |
| A9 convergence once per cycle after the agent; 10 consecutive; settling rule for SRP1 | stated | l. 1564, 1572–1576, Alg l. 1648–1653 (row A4-3 L) |
| A10 TSO targets = DSOs' fresh copies; storage z | stated | l. 1640–1641 |
| A11 TSO solves | stated | l. 1641 |
| A12 both sides' interface duals after the TSO; P, Q over R^I; V in p.u. | stated | eq. (admm_interface_duals) l. 1531–1539 |
| A13 agent target z | stated | l. 1644 |
| A14 agent solves | stated (solver settings absent: item 13) | l. 1644 |
| A15 z-update + three duals; skipped on failure | stated with difference → A3-1 (M), A3-2 (L) | eq. (admm_z_update), l. 1542–1543, 1646–1647 |
| A16 k ← k + 1 | stated | l. 1655 |
| A17 the agent holds no energy limits | stated | l. 1603–1605; §2.3 l. 656 |
| A18 agent objective (normalisation, κ^E, target z) | stated with difference → A2-1 (L, ℒ symbol) | eq. (admm_esso_local) |
| A19 h(·) notation | resolved (no longer used; constraints referenced to §2.3) | l. 1496–1497 |
| A20 sets E^S, Y, D, T | stated (values in §3) | l. 1420, eqs. |
| A21 f(X) | stated as f_E | l. 1497 |
| A22 ℒ^{E,P}, ℒ^{E,Q} | terms written out; symbol not defined → A2-1 | l. 1485–1489 |
| A23 target is z, not an SO request | stated | l. 1430, 1644 |
| A24 one dual per agent | stated | l. 1431–1432 |
| A25 internal constraints | resolved | — |
| A26 normalisation 2S^ref, κ^E | stated | l. 1434–1438, eqs. |
| A27 Q likewise | stated | ξ ∈ {P,Q} |
| A28 one ρ per channel, shared by P and Q | stated | l. 1498–1499 |
| A29 AL steers to consensus | stated (implicit) | — |
| A30 dual update with α, three duals, after z | stated | eq. (admm_z_update) |
| A31 Q likewise | stated | — |
| A32 mismatch against z | stated | eq. (admm_z_update) |
| A33 residual balancing (not monotone growth) | stated | l. 1545–1555 |
| A34 Q likewise (shared ρ) | stated | l. 1546 |
| A35 no per-P/Q rate | resolved | — |

**"Implemented but not in the text" (W166 items 1–53).** Items whose home is §2 are reported by presence at `407f8df`
only; their correctness belongs to W171b/W174.

| W166 item(s) | status | where |
|---|---|---|
| 1–3, 5–9 network storage physics (P = p^ch − p^dch, sum limit, circle, complementarity, SoC recursion, window, soft closure, closure penalty in Q) | stated in §2.3 | l. 705–750 (not in App. A; eq. (soc_closure) l. 724–728 omits the 1e-5 term, W171b T3-4) |
| 4 p = S p̂, p̂ ∈ [0,1] | stated in normalised form | §2.3 eq. (network_complementarity) l. 739–743 |
| publication of S^Av, E^Av (W166 text under item 9) | stated | §2.3 l. 694; App. A l. 1602–1606, Alg l. 1634 (A5-1 L) |
| zero-capacity gating (W166 text under item 9) | absent | — |
| 10 per-cohort p ≤ S^Rated,Unit | stated in §2.3 | l. 859 |
| 11 pro-rata cohort rows | stated in §2.3 | eq. (cohort_allocation) |
| 12 cohort activation/deactivation | **absent** (vacuous for single-cohort plans) | — |
| 13 agent IPOPT tol 1e-10 / acceptable 1e-9 (MA57) | **absent** | (Addendum 68 assigns it to §3.5, not in main.tex) |
| 14 salvage reporting-only | stated in §2.1/§2.2 | l. 377, 483 |
| 15 settlement weight 1; contracted excluded, deviation inside Q | **stated with difference → A6-1 (H)** | A.6 l. 1671–1675; §2.2.7 l. 612; §2.3.6 l. 898 |
| 16 RES-curtailment penalty zeroed / shared-ESS usage penalty zeroed | RES: stated (§2.2.7 l. 613); usage penalty: absent (a zeroed term) | — |
| 17 hard NA of the storage by aliasing | stated | A.6 l. 1669; §2.3 l. 750, 898 |
| 18 scenario-free TSO Δ with the ADN load fixed | stated with difference → A6-2 (L) | A.6 l. 1668; Alg l. 1624 |
| 19 row 18 inactive at initialisation, activated with the settlement | stated | A.6 l. 1670–1671 |
| 20 voltage pin, solver-only | stated (weight 9e4 not printed) | A.6 l. 1673–1674; §2.3.6 l. 898 |
| 21 σ and w_b | stated (σ value not printed) | A.1 l. 1420–1423 (A1-1 L) |
| 22–23 interface AL terms (V in p.u.; P, Q over R^I) | stated | eq. (admm_network_local) |
| 24 consensus on the expected interface | stated with difference → A1-2 (M) | l. 1427–1428 |
| 25 interface dual updates | stated | eq. (admm_interface_duals) |
| 26 TSO proximal at γ = 0 | stated | l. 1499–1500 |
| 27 three-agent weighted consensus, z-update, per-agent duals | stated | A.3 |
| 28 z initialised as the mean | stated | Alg l. 1628 (A5-2 L) |
| 29 agent initialised with P, Q fixed at the TSO's initial schedule | stated | Alg l. 1626–1627 |
| 30 one round of interface-dual initialisation | stated with difference → A5-2 (L) | Alg l. 1628 |
| 31–32 residual forms and tolerances | stated with differences → A4-1, A4-2 (L) | l. 1564–1572 |
| 33 pass AND every solve successful | stated | l. 1572–1573, 1589 |
| 34 ten consecutive passing cycles | stated | l. 1573–1574 |
| 35 objective-change tests diagnostic only | stated | l. 1576 |
| 36–39 ESS two-phase, per-channel freeze, backstop 200, failure hold | stated | l. 1550–1553 |
| 40–45 AA (type-II on (z, u), m 5, 1e-10, ratchet, keep_memory, clears, off when all pass) | stated (A4-4 L) | l. 1578–1583 |
| 46 tight tail | stated with difference → A4-5 (L) | l. 1585–1588 |
| 47 holds from k0 (SRP1 hooks) | stated with difference → A3-3, A4-6 (L) | l. 1554–1555, 1582–1583, 1587–1588 |
| 48 settling rule v6 decides | stated (by reference to §2.2.7) | l. 1574–1576, Alg l. 1650–1651 |
| 49 no update from a failed block or node | stated with difference → A3-2 (L) | l. 1542–1543 |
| 50 ρ held on failure | stated | l. 1553 |
| 51 AA memory cleared on failure | stated | l. 1581–1582 |
| 52 retry tiers; acceptable exits in the clean clause | stated with difference → A4-7 (L); clean clause stated in §2.2.7 l. 628 | l. 1588–1590 |
| 53 network `max_iter` 500 | **absent** | (only in the CONFIRM comment l. 1595) |

**Still absent from main.tex:** items 12, 13 and 53, the zero-capacity gating, and the zeroed shared-ESS usage penalty.
The numeric values that only the comments carry are also absent: σ, κ^E, S^ref = 2.5 MVA, ρ_0, the pin weight 9e4, the
production `compl_inf_tol` 5e-4 / 1e-4, the recovery settings and R^I. All of these await §3.5.

## `% [CONFIRM — W172]` comments

| comment | item | outcome | evidence at HEAD |
|---|---|---|---|
| l. 1440 | (a) w_b uses r = 0.02, 1.02^{−(y−y0)} | **confirmed** | `shared_resources_planning.py:3870-3873`; `SRP1.json` and the 3 × 3 instance `DiscountFactor` 0.02; y0 = first representative year (2025) |
| | (b) σ fixed 93,635,360 by the case file, asserted within ×3 of the computed value | **confirmed** | `SRP1_params.json` `admm.objective_scale`; factor 3.0 = `admm_parameters.py:132` default (not set in the case file); check at 3996-4003. Recorded at cycle 1: `sigma_fixed` 93,635,360; `sigma_computed` 93,635,363.6–93,635,569.5 (SRP1, 34 records), 48,004,110.5–49,556,600.3 (3 × 3, 2 records; ratio 0.51–0.53, inside ×3) |
| | (c) κ_ESSO = σ / median block weight | **confirmed** | 4014-4075 (`esso_al_scale: sigma_over_median_block_weight`); recorded 227,210.997 (SRP1) and 386,258.694 (3 × 3); recomputed from the case files (median 412.1075 and 242.4162): identical |
| | (d) R^I = `get_interface_branch_rating()` in p.u.; value per DN | **confirmed** | 5148, 5417 (`/ s_base`); `network.py:83-94`; values in "R^I for §3.5" |
| l. 1502 | (a) interface AL at 5144-5161 / 5419-5433; V unnormalised (p.u.), P, Q over R_int | **confirmed** | lines as stated |
| | (b) storage AL at 5177-5188 / 5436-5441 / 5505-5523; 1/(2 S_ref); κ only in the agent | **confirmed** | lines as stated; κ at 5514-5518 only |
| | (c) base objective × w_b/σ at 3976-4011, 5136-5142, 5401-5407 | **confirmed** | `obj = copy(...objective.expr) / effective_scale`, effective_scale = σ / w_b (5137, 5402) |
| | (d) proximal γ = 0 (tied_to_rho, τ = 0) | **confirmed** | `SRP1_params.json` `proximal_regularization.tso` {tied_to_rho, tau 0.0}; 5080-5083, 8369-8377; recorded `gamma_{v,pf,ess}_after` 0.0 at cycle 1 in 36/36 committed records |
| l. 1557 | balancing constants (ratio 5, pf decrease 3, ×/÷1.5, clamp, freeze 10, backstop 200, ESS exemption 5 cycles) and ρ_0 v 0.0077 / pf 0.198 / ess 0.01 | **confirmed** | `SRP1_params.json` `admm.penalty_update` (ratio 5.0, pf_decrease 3.0, 1.5/1.5, [1e-4, 1e4], 10, 200, ess {dual_ratio_below 1.0, consecutive 5}); `admm.rho` (v 0.0077, pf 0.198, ess 0.01, esso 0.01); code 8179-8284, 8125, 8212-8234, 8315-8322, 8421; recorded cycle-1 ρ_before 0.0077 / 0.198 / 0.01 (36/36) |
| l. 1592 | (a) residual forms at 7044-7304 | **confirmed** (the text's statement of them: A4-1, A4-2 L) | def 7044, return 7304 |
| | (b) `solver_result_succeeded` statuses | **confirmed** | `helper_functions.py:74-85`: SolverStatus.ok and TerminationCondition ∈ {optimal, locallyOptimal, globallyOptimal}. Pyomo 6.9.5 `opt/plugins/sol.py:117-121` maps every IPOPT exit 0–99 (incl. "Solved To Acceptable Level") to optimal/ok, so acceptable exits count as success (l. 1589) |
| | (c) AA: type-II on (z, u = y/ρ), memory 5, Tikhonov 1e-10, ratchet, keep_memory, cleared on ρ change and failure, off when all pass | **confirmed** (interface "z" = the TSO copy: A4-4 L) | `admm_anderson_acceleration.py`: deque maxlen m + 1 (l. 387), Tikhonov (503), ratchet (508), keep_memory (519), off branch (469-482), `clear_for_rho_change` (398), `skip_on_failure` (416); orchestration `shared_resources_planning.py:3030-3042, 3092-3096, 3272-3280, 3419-3428`; `SRP1_params.json` `anderson_acceleration` |
| | (d) tail `compl_inf_tol` 1e-6, non-latching in production, held by the campaign hooks | **confirmed** (enabled by the specs, not the case file; held from k0 + 1: A4-5, A4-6 L) | 3077-3085, 3394-3398, 7427-7519; hooks `w_next`, l. 526; v6 spec / 3 × 3 spec `convergence_depth_tail` {True, 1e-6} |
| | (e) recovery tiers `acceptable_tol` 1e-4 / `acceptable_iter` 1; network `max_iter` 500 | **differs** for the recovery values; **confirmed** for `max_iter` | `case9_params.json` `recovery_options` {1e-4, 1} (TSO only); `case33_1` no key, `case33_2/3` `{}`, so the DSO tier 1 = cold start with primary options (1e-4, `acceptable_iter` 5); agent tier 1 `acceptable_tol` 1e-9 (`shared_energy_storage_data.py:1082, 1277-1285`), `acceptable_iter` 1; tier 2 adds `mu_strategy` adaptive (`network.py:996-999`); `max_iter` 500 = `network.py:559` (no case-file override; agent 1099) |
| | bibliography key `walker_ni_2011` | **present** in the clone's `bibliography.bib` (`abb26cb5…`, l. 448-453), one entry; Walker & Ni, SIAM J. Numer. Anal. 49(4) 1715–1735, 2011, doi 10.1137/10078356X, matching the comment | — |
| l. 1660 | algorithm order against `_run_operational_planning` 3066-3210; initialisation 4197-4559, 4880-4894, 2951-2960; once-per-cycle check after the agent 3216-3390; capacity publication 3697 / 6075-6082 / 6409-6411 | **confirmed**, except the end-of-cycle order (A5-4 M) and the initialisation order (A5-2 L) | loop 3066; DSO 3115-3136, TSO 3155-3178, agent 3192-3210; check 3216-3390; publication 2980, 3697, 6075-6082, 6409-6411 |
| l. 1677 | (a) row-18 activation with the settlement weight at 5221-5343 | **confirmed** | 5221-5259 (inactive at initialisation), 5262-5313 (activation), 5340-5343 |
| | (b) settlement weight 1 and the contracted/deviation split at 1693-1724, 1251-1304 | **confirmed** (and the settlement is present at one scenario: A6-1 H) | `model_construction_helpers.py:1701-1702, 1723-1724`; `shared_resources_planning.py:5050, 5340, 1251, 1302-1304` |
| | (c) voltage pin 9e4 at 1925-1943, excluded from Q | **confirmed** | `model_construction_helpers.py:1925-1943` (weight Param 1940); `definitions.py:75`; subtracted at `shared_resources_planning.py:1294-1304` |

## Counts

Each printed statement is counted once: an equation counts as one, and a sentence with several independent claims counts
once per claim. Rows are the differing statements above.

| subsection | statements checked | consistent | H | M | L |
|---|---|---|---|---|---|
| preamble (l. 1407–1412) | 8 | 7 | 0 | 0 | 1 |
| A.1 (l. 1417–1438) | 20 | 17 | 0 | 1 | 2 |
| A.2 (l. 1447–1500) | 19 | 18 | 0 | 0 | 1 |
| A.3 (l. 1510–1555) | 26 | 23 | 0 | 1 | 2 |
| A.4 (l. 1564–1590) | 31 | 24 | 0 | 0 | 7 |
| A.5 incl. Algorithm (l. 1602–1658) | 27 | 23 | 0 | 1 | 3 |
| A.6 (l. 1667–1675) | 10 | 8 | 1 | 0 | 1 |
| **total** | **141** | **120** | **1** | **3** | **17** |

Consistent statements, by subsection:
- **Preamble:**
  - simoes_2023 ADMM plus the agent;
  - three-agent global consensus;
  - pairwise interface consensus;
  - residual balancing;
  - AA;
  - tight tail;
  - stopping test here and certification in §2.2.7.
- **A.1:**
  - block = (y, d);
  - one TSO model per block;
  - one DSO model per block;
  - scenarios joint in one model with probability weights;
  - one agent model per node spanning y, d, t;
  - no scenario index;
  - division by σ;
  - w_{y,d} formula;
  - V, P, Q interface variables;
  - storage P, Q via the global z;
  - own copy and dual per agent;
  - three of each on the storage channel;
  - one per side on the interface;
  - P, Q over R^I (interface transformer);
  - 2S^ref common to all nodes, years and candidates;
  - V in p.u.;
  - κ^E = σ / median w.
- **A.2:**
  - consensus terms added at cycle k;
  - the (w/σ) f_a term;
  - the V term;
  - the P/Q terms over R^I with ρ^PF;
  - the storage terms over 2S^ref with ρ^E and target z;
  - x̂ = the other side's current copy;
  - network constraints;
  - the storage model with the published capacities;
  - commitment terms (multi-scenario);
  - the interface sum (one DN for a DSO);
  - the storage sum;
  - the agent over all blocks;
  - eq. (admm_esso_local);
  - the agent's constraints;
  - f_E without AL;
  - one ρ per channel;
  - common to the agents;
  - common to P/Q;
  - proximal weight 0.
- **A.3:**
  - z after the agent;
  - weights ρα² with duals;
  - every agent's dual against z;
  - α = 1/(2S^ref);
  - eq. (admm_z_update) (both rows);
  - interface duals after the TSO;
  - each against the other side;
  - eq. (admm_interface_duals) V;
  - eq. (admm_interface_duals) P/Q;
  - ā;
  - Boyd balancing per channel;
  - one value for all agents;
  - the increase rule;
  - the decrease rule;
  - μ = 5;
  - μ' = 5, and 3 for the ρ^PF decrease;
  - clamp;
  - ESS exemption 5 cycles at dual ratio < 1;
  - freeze 10 after acting;
  - backstop 200;
  - failure hold;
  - ρ change clears memory.
- **A.4:**
  - once per cycle after the agent;
  - r_E;
  - interface r;
  - interface s;
  - b (R^I, 1);
  - Boyd tolerances;
  - ε_pri;
  - n;
  - ε_abs/ε_rel;
  - pass = channels and all solves;
  - production exit 10;
  - the multi-scenario cells stopped by it;
  - objective tests diagnostic only;
  - type-II AA citation;
  - memory 5;
  - Tikhonov 1e-10;
  - ratchet acceptance;
  - memory kept on rejection;
  - cleared on a ρ change;
  - cleared on failure;
  - off when all channels pass;
  - tail trigger and 1e-6 on TSO/DSO;
  - acceptable counted as success;
  - clean decided by the rule.
- **A.5:**
  - summarises one evaluation;
  - publication before each cycle;
  - degraded capacities bound SoC and converter;
  - KwIn;
  - KwOut;
  - DN built with the plan's storage;
  - DN solved at nominal interface voltage (ref Vg = 1.0 in all 51 DN year files, fixed ± 1e-4, `model_construction_helpers.py:72-80`);
  - DN exchange recorded;
  - TN with the recorded exchange;
  - agent with powers fixed at the TSO schedule;
  - z = mean;
  - ρ_0;
  - k = 1;
  - while k ≤ k^max;
  - publication;
  - DSO targets and solve;
  - TSO targets and solve;
  - interface duals;
  - agent target, solve and SoH;
  - z and storage duals gated;
  - the exit with both rules named;
  - k + 1;
  - return.
- **A.6:**
  - consensus on the expected exchange;
  - the scenario-free storage;
  - charge inactive at initialisation and activated at conversion;
  - settlement full weight;
  - contracted removed;
  - deviation kept;
  - pin in TSO and DSO;
  - pin solver-side and removed.

## Hooks vs production: which the text describes

| behaviour | production (`_run_operational_planning`) | SRP1 campaigns (v6 hooks) | what the text says | instance attribution in the text |
|---|---|---|---|---|
| exit | 10 consecutive passing cycles (3384-3390) | production exit disabled (threshold 10⁹ recorded); rule v6 certifies, or the cap ends the run | both: l. 1573–1576, Alg l. 1650–1651 | yes: "it stopped the multi-scenario evaluations"; "the single-scenario evaluations … were stopped by the certification rule" |
| AA off | only on cycles where every channel passes (non-latching) | forced off for every cycle > k0, even across a lapse | both: l. 1582–1583 | yes ("in the certifying regime") |
| tail | non-latching, next cycle after a passing cycle (when enabled) | forced on for every cycle > k0 | both: l. 1585–1588 (A4-5, A4-6 L) | yes ("production setting" / "certifying regime") |
| ρ | balancing until freeze/backstop | frozen for every cycle > k0 | both: l. 1545–1555 | yes ("In the certifying regime …") |
| the two v1 references (W101 continuations) | — | holds from N_old + 1 | not in App. A (A3-3, A4-6 L); §2.2.7 l. 618 states it | §2.2.7 only |

The 3 × 3 cells ran production behaviour: `stopped_by` "boyd", `required_consecutive_cycles` 10 recorded. The x = 0
cell was then continued past its exit (Addendum 51); §2.2.7 l. 633, not Appendix A, says so.

## Scope and configuration

**What was audited.** `manuscript/6a67305f25e8348fb71380c3/main.tex`:
- clone HEAD `407f8df`;
- sha256 `9891057c…8ade3062` verified;
- 2,169 lines;
- `main.tex` and `bibliography.bib` unmodified in the clone.

The scope is Appendix A, l. 1404–1679:
- the preamble, l. 1407–1412;
- A.1, l. 1414;
- A.2, l. 1444;
- A.3, l. 1507;
- A.4, l. 1561;
- A.5 with Algorithm `alg:operational_planning_degradation`, l. 1599;
- A.6, l. 1664;
- the six `% [CONFIRM — W172]` comments.

**Configuration in force, identified as W166/W171b did.**

- **Specs.**
  - Stage spec v6 `96c23404`.
  - Extension v3 `84775dc4`.
  - A64 `44a2dce8`.
  - 3 × 3 campaign spec `231558f0`.
  - Case files `SRP1_params.json` (`dbfdb2a0…`), `SRP1_ESS_Params.json` (`39106f93…`) and `SRP1.json`.
  - The 3 × 3 instance `SRP1__s53_3x3.json` (`2a64e3c5…`).
- **Code identity.** `git diff --stat <git_head_at_run> HEAD` was run through a subprocess argument list, with no shell
  globbing. The pathspec was:
  - the 20 production modules;
  - `SRP1.json`, `SRP1_params.json`, `SharedESS/`, `case9/`, `case33_{1,2,3}/`, `MarketData/`.

  Result: **NO CHANGE for all 46 campaign heads** (34 + 7 + 4 + 3 × 3). Control: against `353e094b` the same pathspec
  reports 5 files, +1,507/−84, so the pathspec is live. The repo HEAD at the check was `1d13858a`. W173 committed during
  this task; its commit touches no production file.
- **SRP1 cells: the certifying regime and the stop are set by harness hooks, not by production.**
  - Hooks: `p515_s53_w142_resettle_v6_hooks.py`, built on `p515_s53_w118_resettle_hooks.py`.
  - From the run's first residual pass k0 (`boyd_k` = `all_boyd_pass` in a cycle whose solves all succeeded), every
    later cycle holds:
    - AA off (forced, even across a lapse);
    - tail on;
    - ρ frozen.

    The hold predicate is `c > first_pass` (w118 hooks l. 526).
  - Production's exit is disabled: `minimum_consecutive_converged_cycles` = 10⁹, recorded as
    `required_consecutive_cycles` 1,000,000,000 at cycle 1 in all 34 committed v6 `g_s39_D.json`.
  - Rule v6 or its cap ends the run.
- **3 × 3 cells: production behaviour.**
  - `required_consecutive_cycles` 10, as recorded.
  - Cap 500.
  - Tail `{True, 1e-6}` from the spec.
  - Row 18 α 0.5.
- **Both instances:** the tail is enabled by the spec (`convergence_depth_tail`), not by the case file. Production's
  default is off (`admm_parameters.py`).

## Unexpected findings

1. **DN appendix table (outside App. A):** one branch table prints the case33_1 interface transformer rating, 200 MVA,
   for all three ADNs. See Blocked on Planner 3.
2. **§3.5 does not exist at `407f8df`.** Section 3 has 3.1–3.4 (Investment Costs, Market Data, Transmission Network,
   Active Distribution Networks). §2 cites "Section~3.5" three times in printed text (l. 750, 859, 900; also two CONFIRM comments, l. 1442, 1559), and
   "Section~3.4" for the calibrations (l. 834), which is now "Active Distribution Networks". This extends the Round-1 Decision 9
   housekeeping.
3. **The DN year files are shared content.** For each DN, the year files 2025/2028, 2030/2031/2034 and 2035/2037 are
   byte-identical blobs: three distinct network files per DN.
4. **W173 committed concurrently** (`1d13858a`) while this task ran. It touched no production file; the code-identity
   check was run against that HEAD.

## Changed

- New: `p515_s53_w172_appendix_a_checks.py` (sha256 `81afd26f…a41f292`).
- New outputs in `data/SRP1/Results/P515S53/w172_appendix_a_audit/`:

  | file | sha256 / contents |
  |---|---|
  | `w172_checks.json` | `62ea6b92…8a04e6fc` |
  | `manifest_sha256.json` | `e0881550…a399e2`: the output, the script and every input read, 152 inputs |
  | `launch.log` | `989a17cb…b06be81e` |
  | `w172_bool_typing_test.json` | `de24585a…52d1` |
  | `w172_bool_typing_test.log` | `22361b78…d1d3` |
  | `manifest_post_run_sha256.json` | hashes of the above |
- This report.
- No production file, spec, case file, frozen artifact, committed artifact or manuscript-clone file changed.

## Not confirmed

- **Compiled numbering.** Equation and algorithm numbers were not checked against a compiled PDF; the clone has none.
  References are to `\label`s and source lines.
- **IPOPT acceptable → Pyomo `optimal`.** Read from Pyomo 6.9.5 `opt/plugins/sol.py:117-121` in the canonical
  environment; the IPOPT return code for an acceptable exit (1) was not re-verified from an IPOPT log here.
- **Count of single-scenario cells stopped by a cap (A4-3).** Taken as the 12 uncertified cells W171b scored (10 v6 + the
  W118 F2 pair); not recounted from the frozen tables here.
- **Correctness of §2 at `407f8df`.** §2 changed between `42794d4` and `407f8df` (+386/−316 lines across main.tex,
  bibliography and letter). It was used here only to locate the W166 items and the cross-references (l. 612–633,
  694–750, 851–900). It was not re-audited; that is W174.
- **R^I runtime value.** Static reading only. The function is a pure read of the case JSON, with a parallel-branch merge
  that sums ratings. It is corroborated by the recorded `worst_pf_primal_rating` values, but no network object was built.
- **Scope of negative claims.**
  - "No §3.5": the `\section`/`\subsection` list of main.tex at `407f8df`.
  - "Settlement in no other appendix line as part of f_a": l. 1404–1679 searched for "settlement".
  - "No transformer ratings per DN printed": l. 992–1062 and 1733–2040 searched for MVA, transformer and rating.
  - "Gap not recorded by production": the `admm_diagnostics` dict at 3431-3661 and the 3 × 3 eval directory listing.
