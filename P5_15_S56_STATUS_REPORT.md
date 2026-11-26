# P5.15 S56: operational uncertainty and investment, and the unified core rerun. Status and decisions (2026-11-21)

**Stop point.** The audits, the inputs, the parallel gate and the interface-consistency diagnostics are done. Every
next step needs a ruling from the author or the Expert, and some need named machine time.
- **What ran:** zero-solve work, plus guarded diagnostic solves of at most a few minutes each.
- **What did not run:** no search, and no full evaluation.
- **Preserved:** every existing result is unchanged.

## Decisions needed

1. **TN–DN coupling under uncertainty (Expert).** The executed multi-scenario model is consistent only in expectation
   (Addendum 38 (A), a signed choice). The TN carries one schedule; each DN's per-scenario draw differs from it.
   - **Measured:** a mismatch of up to 48.3 MW, with ≈ 268 GWh weighted each way. D0 below gives its cost.
   - **Option (a), recommended for the stochastic study:** adopt per-scenario coupling (F1 of
     `S56_SCENARIO_COUPLING_PROPOSAL.md`).
     - **What it is:** an additional per-scenario deviation consensus channel on P and Q. The TSO re-dispatches to each
       realization. The day-ahead commitment stays the DSO's expectation; row 18 is kept; V stays on expectation with a
       new per-scenario gate.
     - **Cost:** a production change behind a default-off option, new gates and re-pinned launchers, then validation
       V3 → V1 → V2. Stochastic evaluations would need a guessed 1.3–3× more cycles.
   - **Option (b):** keep Addendum 38 (A). Search and evaluate under the pinned model. Add a TN realized-exchange re-solve
     (D0, minutes) to every reported plan, as a measured correction, and disclose it.
     - **Cost:** cheap.
     - **Limit:** the DNs never respond to the TN's true marginal cost.
   - **Option (c), the fallback:** pass-through.
   - **Why (a).** 96 % of the omitted cost sits in TN curtailment hours. That kink is exactly the channel through which
     the uncertainty study could find a difference between expected-profile and stochastic planning (H2), and the pinned
     model misprices it.
2. **Under (a), the conventions (Expert):**
   - the settlement excluded from Q in full;
   - row 18 as a pure premium. It also holds Q at the commitment, the restriction the author asked to avoid, applied
     to reactive power;
   - the per-scenario gate and its thresholds;
   - the ω-weighted residual;
   - the joint-scenario assumption. Scenario draws are not joint across networks.
3. **The search acceptance rule (Expert).** One common σ_S56 = max(18,449.66, D_EV, D_SP) for both arms, frozen before
   the pilot, plus a settled confirmation before any incumbent move (`S56_DESIGN_NOTE_v1.md` D4).
4. **The common grid (author). Recommended: 2025/2030/2035 with the 3 × 3 ensemble (built).**
   - It keeps the published years, cost table and 1,704-plan domain.
   - The published single scenario is draw 0 of this ensemble, verified bitwise in all three years.
   - The alternative, five blocks (2025–2037), reuses one archived pair, has 3,281 plans and costs 8–10 h per stochastic
     evaluation.
5. **The primary configuration (author). Recommended:** single-scenario primary, as published, with the stochastic study
   as the uncertainty finding. The coordination benchmark has no multi-scenario definition: the code refuses it, and
   defining it would need an Expert formulation ruling.
6. **Machine time (author):**
   - **The pilot:** a settled pair, {x = 0, reference unit} under the stochastic and the expected-profile models, about
     16 h (13 h stochastic). Under (a) it follows the validation.
   - **Each stochastic search launch:** every stochastic evaluation exceeds 4 h.
7. **A manuscript disclosure now (author and Expert), whatever is ruled in 1.** The archived five-block 3 × 3 section
   should state that deviation energy is settled financially and not carried in the TN model, and give its measured
   effect: a level of +0.93 M€, and a storage value change of −2.6 k€, inside its 18 k€ resolution.

## Blocked on the author

- Push from PyCharm.
- The rulings above.

## Changed (commits)

| step | commits | output |
|---|---|---|
| audits A, B, C and design note v0 | `6d677758` | `P515S56/input_audit_{A,B,C}/` |
| design note v1 (Advisor review applied) | `faccc101` | |
| audit D, interface consistency | `f80ec8a9` | |
| coupling proposal | `16cfaeec` | `S56_SCENARIO_COUPLING_PROPOSAL.md` |
| task E, instances | `e681f664` | `P515S56/instances_v1/`: SRP1-grid 3 × 3 (SP checksum `73d0a230`) and EV twin (`0040b146`); in-child injection assertion `p515_s56_ev_injection.py` |
| task F, concurrency gate | spec `2684dcbf`, result `99b1aae7` | `P5_15_S56_CONCURRENCY_GATE_REPORT.md` |
| D0 v1 and V0 | spec `d11bea28`, evidence `a781024a` | |
| D0 v2 | spec `e47ea59e`, evidence in this commit | |

No production or harness file was changed.

## Found

### Uncertainty-study inputs

- **SRP1's single scenario is draw row 0, not the expected profile.** Against the mean, energy price differs by 7.9 %,
  wind by 9 % and loads by 2 %.
- **The expected-profile instance** is built and verified bitwise. It loads through an injection hook; the case file
  alone reads as the published draw. An armed in-child assertion refuses any other read.
- **One-scenario structure:** at one scenario, row 18, the deviation variables and the voltage pin are never built. The
  expected-profile arm therefore has no deviation cost by construction (Advisor: correct for EV planning).
- **On the new grid:**
  - prefix property bitwise in 2025, 2030 and 2035;
  - 1,704 plans;
  - minimum hourly mean price 5.47 €/MWh;
  - the 4 h spread of the mean profile against the published draw: 0.947.
- **A likely structural null (Advisor):** the storage schedule is scenario-common and prices enter linearly, so both arms
  earn the mean-profile spread. Only quantity effects at kinks remain. The pilot pair tests this before the search.

### Runtime and memory (measured unless stated)

| run type | time per evaluation | memory | concurrency on 24.4 GiB usable |
|---|---|---|---|
| SRP1 single scenario (and EV on the SRP1 grid, the same size) | 0.9–1.5 h, 121–200 cycles | 2.38–2.46 GiB RSS, full run | 5 (contention 1.18×, ≈ 4.2× throughput, gate F) |
| five-block 3 × 3 (archived) | 7.9 / 8.4 h at the exit; ≈ 10 h settled (arithmetic) | 19–20.5 GiB footprint | 1 |
| SRP1-grid 3 × 3 (stochastic) | **projected** ≈ 5 h, ×1.3–3 under coupling (a) | **projected** 11–12 GiB | 1 (two would exceed the gate) |

**Core package, projected; measured per-evaluation figures, evaluation counts from the published searches:**

| part | estimate |
|---|---|
| EV searches at 1× and 2× (14–34 evaluations each, at 4.2× throughput) | ≈ 10–20 h |
| stochastic searches at 1× and 2×, serial | 140–340 h pinned; 180–1,000 h under (a) |
| fixed-plan stochastic cells: three ageing variants of the unit, 1× and 2× pairs, the cross-evaluations (≈ 8–10 cells) | 40–150 h |
| **total** | ≈ 200–1,200 h (≈ 1–7 weeks of this machine) |

The pilot replaces this projection with measurements.

### Parallel execution

- **Within a cycle: not available.**
  - The persistent-worker path is refused in code with the tight tail, because workers hold stale solver options and
    lose the IPOPT records the depth gate needs. The DSO pool loses those records too.
  - Fixing either is a production change; the bound is ≤ 1.9×, against 1.04–1.16× measured.
  - Not pursued in this study.
- **Candidate-level poll parallelism.** Numerically bitwise between serial and 5-way concurrent (gate F). The gate is
  recorded as FAILED as frozen, on two byte counts derived from a wall-time field. That field is declared excluded
  before the next gate.

### Interface consistency (audit D, D0)

- **The mismatch is in active power only.** Q and V agree to ≤ 2.4 × 10⁻³ MVAr and ≤ 8.5 × 10⁻⁴ p.u.
- **No acceptance test sees it.** All tests act on expected values.
- **D0 v2** re-solved the archived five-block 3 × 3 TN blocks TN-only, with the DN's realized per-scenario exchange
  (Arm B) against the committed exchange (Arm A). The storage entered as a fixed injection.
  - **Convention:** C = TN generation + TN flexibility, gross, settlement excluded, block-weighted.
  - **Statistic:** D1 = (C_B − C_A) − DSO covariance.

| cell | C_A | C_B | covariance | **D1** | bar |
|---|---|---|---|---|---|
| x = 0 | 624,559,055.14 | 624,737,921.44 | −748,745.77 | **+927,612.08** | 13,954.38 |
| reference unit | 624,473,545.50 | 624,656,794.00 | −746,992.08 | **+930,240.58** | 4,010.57 |

- **The pinned model is optimistic by ≈ 0.93 M€.** That is 0.15 % of the TN cost.
  - **Where:** 95.8 % (x = 0) of it lies in scenario-hours where a DN draws less than its commitment while the TN's
    controllable units are at their minimum. The TN must then curtail RES, worth 0 at the margin, while the settlement
    credits π_s (69–197 €/MWh).
  - **When:** concentrated in 2031 and 2037. It is ≤ 6 €/day in 2025 and 2028.
  - **Elsewhere,** TN marginal cost tracks π within about 1 %.
- **The storage value is unaffected within resolution.** The pair difference is −2,628.51 € against bars summing to
  17,965 €, so it is indeterminate.

| prediction (frozen) | outcome |
|---|---|
| P1, v1: 80/80 Optimal | **failed**: the unit cell's storage schedule was fixed exactly at values 1e-8 outside its own limits (arm design) |
| P1, v2 | **failed**: 78/80 strict; 2 "Acceptable", accepted by production's test, violations ≤ 2.5e-8 |
| P2, v2: x = 0 control reproduces v1 | **held**, bitwise |
| P3: \|D1\| 0.05–1 M€ | **held**: 0.928 / 0.930 M€ |
| P4: \|pair\| < 18 k€ | **held**: −2.6 k€ (indeterminate against the bars) |
| P5: Arm B TN slacks ≤ 0 | **held**, v2 40/40 |
| V0: the merged 3 × 3 model builds, 343,656 variables; 38,184 at one scenario | **held**. The merge drops the shared ESS, row 18 and the pin, so a centralized validation comparator needs that construction |

## Not confirmed

- **Under coupling (a):** convergence, cycle count and runtime. No per-scenario-coupled run exists. V1/V2 measure them.
- **The stochastic SRP1-grid evaluation:** time and memory are projections. The pilot measures them.
- **The two "Acceptable" D0 solves** were not checked against a strict optimum. They carry D1 of +1,218 and
  +708 €/day.
- **D1 includes only the TN's re-dispatch.** The DNs did not respond to the TN's marginal cost, which (a) would let
  them do.
- **A manuscript search beyond round 9c** (supplement, letter, later Overleaf heads) for sentences implying physical
  balance was not done.
