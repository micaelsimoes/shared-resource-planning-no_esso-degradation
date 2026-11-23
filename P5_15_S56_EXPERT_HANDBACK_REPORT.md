# P5.15 S56: handback to the Expert. Coupling validation (steps 1–4), cohort and search items (2026-11-23)

**Where the work stands.**
- **Done since the Expert's rulings of 2026-11-21:**
  - the per-scenario coupling option is complete and gated;
  - the centralized comparator (step 3) has run;
  - step 4 (pinned vs F1, about 10–20 h, authorized) is built and smoke-tested.
- **Held for this report:** step 4's main run. Two of its pre-run diagnostics met the conditions under which the step 4
  design defers to the Expert: X4-1 and X4-2 below.
- **Also waiting on the Expert:**
  - the cohort follow-up, which needs decisions 3 and 4;
  - the search driver, which needs decisions 5 and 6.
- **Preserved:** every existing result.

## Decisions needed (Expert)

The first two gate the next long run.

### 1. X4-2: local-solve depth during calibration

**Measurement (D-1).** Step 3's per-scenario interface P/Q/V were fixed in the ADMM-prepared local models, with the F1
option on, V-1 and α = 0, and the four local problems of each block were solved.

| block | allowance (R/4 · w_b/Σw) | offset at production depth | offset at tail depth |
|---|---|---|---|
| 2025\|Winter | 228.64 € | +597.67 € (2.61×) | +45.67 € (0.20×) |
| 2035\|Summer | 187.56 € | +499.03 € (2.66×) | +40.58 € (0.22×) |

- All values are € of total discounted cost.
- All 16 solves were Optimal, and none was infeasible.
- The bias is one-signed: the barrier leaves the local solutions above the centralized cost.

**Options**

| option | what it means | cost |
|---|---|---|
| **(a) Recommended** | No configuration change. Production already moves to tail depth after an all-channel Boyd pass, so a settled run's terminal cycles are at tail depth. Calibration and acceptance use tail-depth cycles only. The production-depth floor (≈ 2.6× the allowance per block) is reported as a limitation of earlier cycles | none |
| (b) | Tail depth from cycle 1 in every arm | a production configuration change; slower early cycles (not measured) |

### 2. X4-1: V-1 or V-2 voltage coupling

**Measurement (D-2).** The centralized cost of making the interface voltage scenario-common (a hard pin), Δ_pin:

| block | Δ_pin, €/day | Δ_pin, weighted | against the allowance |
|---|---|---|---|
| 2025\|Winter | 0.4056 | 184.5 € | 0.81× |
| 2035\|Summer | 4.8856 | 1,823.6 € | 9.72× |

**Proposed rule** (`S56_STEP4_DESIGN.md`):
- **eliminate V-2** if its ΔV plateau exceeds the ΔV gate, or if Δ_pin exceeds the allowance;
- **eliminate V-1** if its V channel does not contract, or it fails local solves repeatedly while V-2 settles;
- **if both survive,** choose V-1 unless its cycles × wall time exceed V-2's by more than 1.5×.

**Options**

| option | what it means |
|---|---|
| **(a) Recommended** | Adopt the rule. V-2 is then eliminated on the 2035\|Summer evidence. Step 4 runs V-1 (calibration), pinned with AA off (control), and pinned with AA on (published configuration) |
| (b) | Adopt the rule, but keep V-2 in step 4 for M1 and M4 (ΔV plateau and effort). Cost: one more concurrent arm per instance, about 3.4 GiB, no extra wall time |
| (c) | Another rule |

**Why V-1:**
- V-1 is the only arm whose limit is step 3's centralized optimum exactly: at α = 0 there is no row 18, no pin is
  built, and settlements cancel per scenario. Step 3's audit confirmed this by reading the code.
- V-2's soft pin may cost less than the hard pin, so Δ_pin bounds V-2's restriction from above. Even so, V-2 cannot be
  validated to R/4 against any centralized optimum.

### 3. Within-cohort simultaneity: activate the per-cohort complementarity row (E2 follow-up)

**Measurement** (isolated ESSO tests, `a1ee41b6`). With TV on, the 2025 cohort deliberately charges and discharges in
the same hours: 0.85 MWh/day in 2035, ratio 0.908. The throughput term ε does not restrict this. E2 requires
simultaneity to be "restricted or demonstrably negligible", and it is neither.

**Options**

| option | what it means |
|---|---|
| **(a) Recommended** | Activate the per-cohort relaxed complementarity row (the network `sess_comp` form) on multi-cohort node-years, together with the per-cohort row p^ch_c + p^dch_c ≤ s_c. Single-cohort node-years are unchanged |
| (b) | Keep the row as a detector and exit gate only. This leaves the measured simultaneity in place |

### 4. Decomposition and the R/4 composition: D1–D4 of `S56_DECOMPOSITION_NOTE.md`

| # | decision | recommendation |
|---|---|---|
| D1 | Decomposition | (a): on multi-cohort node-years, deactivate only the network aggregate `sess_comp`. Exact per-cohort physics in the ESSO; P/Q consensus unchanged; one formulation key for both sides |
| D2 | Closure | Every energy state returns to 0.5·E_av. The slack is a feasibility device and must sit at its bound at acceptance. New gate: network closure slack ≤ τ_cl |
| D3 | R/4 composition | V_pair = S + S′ + \|PR − PR′\| + G + G′ ≤ 2,500 €, as a sum |
| D4 | Cancellation guard | A + A′ ≤ 5,000 €. PR only withholds determinacy, consistent with "diagnostic only" |

**New input for D3.** Production's per-cycle `priced_residual_pr` holds only the P-deviation price, and its price part
equals −`interface_settlement_total` identically. It is not D3's PR. D3's PR (every channel × its multiplier) is
therefore recomputed after the run from the new entry-level capture (`430dfba8`), which records every primal copy and
dual.

### 5. Completion cap for unsuccessful polls

|N| over all 37,042 staged plans: min 3, median 39, max 75. With the planned cap of 60, 1,655 plans (4.5 %) could not be
polled completely.

**Recommended: cap 80,** so every plan in the domain can be polled completely. It costs extra only when a poll fails at
one of those plans.

### 6. Diagonal resizes in the neighbourhood

**Recommended: read "(±1, ±1), (±1, ±2)" as same-sign steps only.** These scale a cohort along its 2 h and 4 h duration
edges. Including the opposite-sign steps adds about 3 % more points, with the same maximum and the same count above 60.

### 7. One prediction recorded against the coupling proposal (for the record)

The proposal's V1 prediction "C_central − C_pinned ≤ 0 on the high-curtailment block" was recorded at α = 0.5, before
D0. D0 found the pinned accounting optimistic by about 0.93 M€. The step 4 design therefore records the opposite sign,
D_b = C_central − Q_pin ≥ 0 on the most-surplus block. Failure of that sign stops the run. Please object if you read the
mechanism differently.

## Blocked on the author

Nothing. Step 4's machine time is inside the existing 10–20 h authorization. The projection is about 4–5 h for the main
run at cap 300, with arms concurrent, and up to about 5 h more for continuation.

## Changed

All commits are on `feature/derivative-free-planning`. Production changes are default-off or default-unchanged.

| step | commits | spec |
|---|---|---|
| Step 1 completion | `f6bd136e`, `2c328c58`, `a738b7c2`, `1f2cca37`, `43058392`, `a813e999` | addendum `178bf1bb` (pred. `10c98258`) |
| Step 3 | `9d37e0df`, `5d870c81` | `4017dc70` |
| Step 4 design | `6e12761e`, `9c88d89d` | — |
| Step 4 build | `8b7d2c26`, `430dfba8`, `61468bc4` | — |
| Step 3 v2 and diagnostics D-1–D-3 | `a07d2e71`, `59cda418`, `1948bb3e` | `ff466289` |
| Step 3 v3 and step 4 smoke | `a1113aac`, `c7720654`, `4856bd46`, `bb8639cd`, `6633f35f` | `b8d07450`; smoke v2 `e36c29bd` (v1 `2e60b2ca` refused at 0 solves) |
| Earlier on 2026-11-23 | `8c9fd6ec` (cohort key), `a1ee41b6`, `39f44fde` (decomposition note), `b87819c4` (neighbourhood and K) | — |

**Step 1 completion, `f6bd136e`: Q convention, per author decision 6.** With the coupling wired, the primary
`gross_operational_cost`, and so the stopping-test input, excludes the settlement in full. Today's form and both net
figures are reported beside it.

**Step 1 completion, `2c328c58`: a defect fixed.** The harness's objective rescaling took its snapshot before the
coupling terms were added. Every harness run with the option on, before this fix, solved without the deviation terms;
these were smoke runs only. The snapshot is now re-taken after the wiring, with an armed check.

**Step 4 build, `430dfba8`:**
- the initialisation-identity grouping now follows pin construction;
- a per-cycle S56 capture (primal copies, all duals, ρ, tail state, per-block Q in both conventions) runs behind a
  capture-path assertion.

**In progress (zero-solve):** per-block Q and pin capture for the pinned arms, which the smoke showed is recorded only
at the terminal cycle. It sits behind an explicit key, off by default.

## Found

| prediction | outcome |
|---|---|
| G-f r3 (coupled smoke, 2 cycles): fields, Q identities, PF identity | **held**; 153 solves = base |
| r2's 200 MW cycle-1 interface jump comes from the option | **not supported.** The option-off control shows it too (199.998 MW). It is the ADMM box: ±rating about the DSO's initial expectation |
| Step 3 P2: 22 solves Optimal | **failed.** 3 of 27 cold single-scenario solves stall at the same μ (1.8449e-6). Warm starts cured two; 2025\|Summer single 12 stalls from both starts |
| Step 3 P3/P4: merged = Σω C_s (1e-6 / 1e-5) | **held** on both complete blocks. Merged − Σω singles = +0.0345 to +0.0354 €/day on 26 pairs (stopping slack, untested) |
| S3-R: reproduction bitwise | **held** |
| D-1, D1a: production-depth offset ≥ allowance | **held** (2.6×) |
| D-1, D1b: tail-depth offset ≤ allowance/3 | **held** (0.20–0.22×) |
| D-2: Δ_pin ≥ −0.04 €/day | **held**: +0.41 and +4.89 €/day |
| D-3: central interface values inside the ADMM-only boxes | **held** at a 1e-8·max(1,\|b\|) tolerance. That tolerance was chosen after a draft trial, and the spec discloses it. Strictly, V exceeds 1.1 pu by ≤ 4.6e-10. Minimum margins: P 37 MW, Q 85 MVAr |
| Smoke S-P1 to S-P5 (4 arms × 2 instances, 3 cycles) | **all held**: 288/288 solves; F1 identities to 9.3e-11; initialisation pinned ≡ V-2 bitwise, V-1 separate; capture = production Q to 2.5e-16; RSS ≤ 3.4 GiB; 46–52 s/cycle on two blocks |

**Centralized costs**, gross, settlement excluded, no salvage, € per representative day:

| block | C_central | status |
|---|---|---|
| 2025\|Winter | 175,188.678750 | complete |
| 2035\|Summer | 153,017.214164 | complete |
| 2025\|Summer | 185,674.155753 | merged-only; reported, not used for calibration |

**Other findings**
- **Derived instances must keep the first year.** Production grows prices and loads from the instance's first year.
  A 2035-only instance therefore lowers 2035 energy prices by 1/1.28 and DN loads by up to 1/1.29. Step 4's instance B
  is 2025 + 2035 Summer; its data are bitwise equal to the 12-block instance.
- **The DN 7 reference-generator limit is within 0.49–0.56 MW** of the centralized exchange on the summer blocks. The
  merge drops that limit, but it does not bind.
- **AA was inert over 3 cycles** (no history yet). Its effect can only appear in the main run.

## Not confirmed

| item | why, or where it was looked for |
|---|---|
| The stopping-slack reading of the +0.035 €/day merged-vs-singles offset | no tighter merged solve was run |
| Global optimality of any solve | the hard-pin solve has one start per block |
| Convergence of any arm, and the size of the ADMM error against the allowance | three smoke cycles only; cycle-3 per-block Q is 8–22 % above the reference |
| Whether the two-block calibration transfers to twelve blocks | the design applies it as a per-block gate; extending step 3 to 12 blocks (132 solves) is to be weighed at the pilot |
| Production's legacy per-cycle metrics read by the harness (`primal_pf`, `dual_pf_mean`) with the deviation channel included | not checked |
| The solved R8 check (1e4) | waits for the final cohort and TV formulation (E7) |
