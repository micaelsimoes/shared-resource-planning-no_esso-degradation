# P5.15 S56 report (ii): EV searches complete at 2× and 1× (2026-11-26)

**Report point (ii)** of the working mode (author and Expert, 2026-11-23): the expected-profile (EV) searches are complete
at both flexibility-price levels. For the author and the Expert. Prepared by the Worker; the recommendations are the
Planner's. **Nothing for the Expert to rule:** no formulation change is proposed and no must-hold inequality failed
(the 1× leak prediction Q9 held).

**Objective convention (every € figure and every table):** F = I(x) + gross_operational_cost − TV credit, in € of total
discounted planning cost (oracle mapping v2; gross is settlement-excluded).
- dF_exit = F_exit(x) − F_exit(x0), both at exit depth and at the same price level.
- dF_set = F_set(x) − F_set(reference) at K_EV = 300; the reference is named in each table.

**Measured and projected are kept apart.** Every projected figure is labelled PROJECTED. Everything else is read from
the phase records.

**Solve claim.** This report ran zero solves. `p515_s56_phase1x_report_eval.py` ran under an armed
`SolveProfileGuard(permitted=())`, verified at 0. The imported 2× evaluation module's guard and, in manifest mode, the
test module's guard were also verified at 0. `pickle.load`/`loads` were blocked and called 0 times. Nothing was written
inside `phase1x_ev_v1/`, `phase2x_ev_v*/` or `pilot_ev_v1/`. The code the phase ran is byte-identical to spec
`feb487b7`'s `code_sha256`; this was checked at the start and again at the end of the evaluation.

Sources:

| item | file | sha256 / commit |
|---|---|---|
| 2× end-of-phase report | `P5_15_S56_PHASE2X_REPORT.md` | `50d7dc67` |
| 2× evaluation | `phase2x_ev_end_report_eval/report_eval.json` | `4427e1421e0c…` |
| 1× phase results | `phase1x_ev_v1/phase_results.json` | `ee346a211993…` |
| 1× predictions (Q1–Q9) | `phase1x_ev_v1_predictions.json` | `de8313a44aae…` |
| 1× evaluation (this report) | `phase1x_ev_end_report_eval/report_eval.json` | `8dcee42963dd…` (60/60 checks) |
| memory probe, SRP1-grid SP | `memory_probe_v1/srp1grid/memory_probe_srp1grid.json` | `1aaeeadd2baf…` |

## Decisions needed (author)

1. **Settled confirmation of the closest 1× rows: one settled launch of 5 plans.**
   - **The 5 plans:**
     - the 3 unresolved rows: n9_y2030_p0.25_e0.5, n5_y2030_p0.25_e0.5 and n7_y2030_p0.25_e0.5;
     - the two worse rows closest to their boundary, by margin to R + e: **n5_y2030_p0.25_e1 (+523.5 €)** and
       **n9_y2030_p0.25_e1 (+7,075.9 €)** (Found §2).
     - x0's 1× settled record already exists and is reused.
   - **Cost: PROJECTED 2.48–2.72 h**, one launch. The low end is the measured 1× pool settled launch (2.48 h,
     3 evaluations). The high end is the 2× settled launches of 5 (2.48–2.72 h); all 2× settled launches ran
     2.01–2.72 h.
   - **Purpose:**
     - measure the 1× exit-to-settled pair offset on 5 more plans. The e_EV carry-over (D1) is contradicted on the
       one 1× pair measured: +20,418 € against e_EV = 17,494 € (Found §3);
     - settle the classifications nearest the boundary before the 1× result enters the manuscript.
   - **Build needed (not built):** a confirmation-only spec over the existing 1× exit records, with zero-solve
     tests. That spec must declare `phase1x_ev_v1` as a cache root:
     - simulated freeze scan with the root declared: 0 collisions over 361 files;
     - without it: the 7 launch records alone already collide (Found §8).
   - **Options:**
     - **(a) Run it (recommended).** "No plan improves on x0 at 1×" is a manuscript result. It rests on an exit error
       measured only at 2×.
     - **(b) Do not run it.** Report the 1× result with the exit-depth classes, and state the measured 1× offset as a
       limitation.
   - **Recommendation (Planner): (a).**

2. **Next ordered work after the searches** (plan v4.1; TASKS "Working mode" scope cuts). PROJECTED; the basis of each
   figure is stated.

| item | machine time | basis |
|---|---|---|
| TV repricing under four conventions (plan v4.1 §3a.1, 3a.3) | **0 h** (zero-solve) | every evaluated plan, from records |
| Calibration (a) re-run (deferred to idle time) | A ≈ 37 min, B ≈ 2.6 h at the declared caps 140/320 | TASKS "Deferred to idle time"; **needs spec v4** (new roots; C9 fraction-at-limit; SIGTERM handler): zero-solve build |
| Step 5 | step 5 pair: 5–12 h (plan v4.1 §8; no new measurement) | "SP calibration" in the working mode = calibration (a) above (Planner reading: TASKS "calibration (a) … gates only the SP evaluations"); no separate line |
| SP pilot (plan v4.1 §7 step 8, SP part: x0, unit, staged to settling) — already authorized, but > 4 h, so its hours are listed before launch | **EXTRAPOLATED: 77–113 h** (3 × t_SP at K = 300) | measures t_SP; its first evaluation can serve as an SP evaluation of x0 if the configuration matches |
| SP evaluations: {x0, n5_y2030_p1_e4 (2×-selected), reference unit} × {1×, 2×} = 6 | **EXTRAPOLATED: 154–226 h to K = 300; 40–139 h at exit depth** (serial: one SP at a time, 15 GiB) | t_SP below; plan v4.1 §8: 40–120 h |
| Wait-and-see bracket: 2 plans × 9 scenarios × {1×, 2×} = 36 single-scenario evaluations | **EXTRAPOLATED: 6.1–8.8 h at exit depth; 19.8–21.8 h at K = 300** (8 blocks of 5) | measured EV block walls; plan v4.1 §8: 7–11 h |
| Dependence stress test: 2 fixed-plan SP evaluations | **EXTRAPOLATED: 51–75 h to K = 300** | t_SP below; plan v4.1 §8: 13–30 h |

   **t_SP extrapolation (an extrapolation, not a measurement).**
   - The memory probe measured the SRP1-grid 3 × 3 SP instance (staged plan) at **452.0 s for cycle 1 and 307.6 s for
     cycle 2**. Only two cycles were run, and the build took a further 131.7 s.
   - t_SP(n) = n cycles × s/cycle. The low end uses the cycle-2 wall, the high end the cycle-1 wall.
   - The cycle counts are those measured in the EV phases. They are applied to SP, where they have not been measured:

| cycle count (measured, EV) | t_SP PROJECTED (h) |
|---|---|
| 1× exit read cycles, 79–88 | 6.7–11.1 |
| 2× exit read cycles, 119–185 | 10.2–23.2 |
| K_EV = 300 | **25.6–37.7** |
| plan v4.1 §8 (unmeasured at the time) | 6.5–15 |

   At K = 300 the extrapolated t_SP is 1.7–5.8× the §8 range. At exit depth it overlaps the §8 range.

   **Recommendation (Planner):**
   - authorize now, at no machine time, the zero-solve TV repricing and the calibration (a) spec v4 build;
   - decide the SP machine hours after calibration (a) reports, when the SP depth (exit or K = 300) and t_SP can be
     set against measured numbers.

## Blocked on the author

- **Push:** the commits through this report. Nothing is held at the machine; no run is in flight.
- Decisions 1 and 2 above are needed before anything runs next. Neither has a spec yet.

## Changed

| commit | content |
|---|---|
| `637090ba`, `f2354a12`, `bd3a0534`, `d7fdfb4e` | 2× phase build, spec v1 `de4e947e` (declared un-launched) |
| `5c023309`, `1e6ab91c`, `7350e5da`, `96c27840` | 2× phase spec **v2 `18a224d1`** (10 launches) |
| `4030508f`, `c0134486`, `bf4bc904`, `c4d9efcd`, `06bb5930`, `690d4737` | 2× continuation spec **v3 `600bbceb`** (3 launches; ESSO-recovery solve accounting) |
| `50d7dc67` | 2× end-of-phase report and evidence |
| `105ae315`, `57c18d22` | 1× phase build, spec **`feb487b7`** (Q1–Q9), preflight, resume check |
| (this commit) | `p515_s56_phase1x_report_eval.py` (new, read-only, zero-solve); `phase1x_ev_end_report_eval/report_eval.json` (`8dcee42963dd…`, 60/60) and its logs (stdout `75e4b84366ee…`, stderr empty); manifest `phase1x_ev_end_evidence_manifest_sha256.json` (`2d038a91b146…`) and its logs (stdout `d63de6e8aaaa…`, stderr empty); the 1× root evidence by rule (§8); this report |

- No spec, code path, production module or `TASKS.md` entry was changed.
- The script's sha256 is `22c1e4476cbd…`, recorded in `report_eval.json`.
- Configuration (both levels): SRP1 grid 2025/2030/2035, 3 × 3 EV instance, corrected cohort mode, TV in the objective.
  R = 10,000 €, e_EV = 17,494.05 €, K_EV = 300, B = 1,000,000 €, cap 80, blocks of 5. The flexibility-price multiplier
  is 2.0 (specs `18a224d1`, `600bbceb`) or 1.0 (spec `feb487b7`) at both depths.

## Found

### 1. Results at both price levels (MEASURED)

| | 2× | 1× |
|---|---|---|
| specs | v2 `18a224d1` + v3 `600bbceb` | `feb487b7` |
| launches; hours (ceiling) | 13; **25.41 h** (120 h) | 7; **6.74 h** (60 h) |
| evaluations | 52 (+ pilot cache) | 26 (no cache) |
| screen + ladder classes vs x0 at exit | 16 confirm-candidate, 6 unresolved, 0 worse | **0 confirm-candidate, 3 unresolved, 19 worse** |
| efficiency branch | settled-direct (22/22 not worse) | exit-then-confirm (3/22) |
| candidates | n5_y2030_p1_e4; n7_y2025_p0.25_e1 | none |
| best start | n5_y2030_p1_e4 | **x0** |
| search | 0 moves; terminal poll 10/10 from cache (9 worse, 1 unresolved) | 0 moves; terminal poll 18/18 from cache (15 worse, 3 unresolved) |
| **final plan** | **n5_y2030_p1_e4** (1.0 MVA / 4 MWh, node 5, 2030), F_set 811,825,206.34; **dF_set −182,404.11 vs x0**; **budget-limited** (author 2026-11-26: B kept) | **x0**, F_set 665,242,478.81 |
| x0 F_exit / bar | 812,009,208.71 / 4,271.36 (exit view of the pilot x0_i1) | 665,232,237.97 / 13,215.20 (exit run, read cycle 85) |
| x0 F_set (K = 300) | 812,007,610.45 | 665,242,478.81 |
| plans indistinguishable from the final plan (within R) | n5_y2030_p1.25_e4 (+2,668.08 settled); n9_y2030_p1_e4 (+9,940.16, exit only) | **none**: nearest settled is the unit at +78,737.79; nearest exit is n9_y2030_p0.25_e0.5 at +27,355.25 |
| controls | recorded as decided at zero solves (author 2026-11-26) | not run: the search did not move (plan v4.1 §4; author decision 5) |

**Reading (scoped).** No improvement beyond the applicable comparison criterion was identified among the evaluated
neighbouring plans, at either level. This is not a global-optimality claim. Unresolved rows stay unresolved.

**The budget binds at 2×** (`P5_15_S56_PHASE2X_REPORT.md` §4):
- I(x) = 944,672.09 € against B = 1,000,000 €;
- every energy-adding move at node 5 and every second unit is refused by the budget: 51 refused moves (budget only 30,
  budget + duration 18, duration only 3);
- neither the cumulative 5 MWh cap nor the completion cap 80 binds.

### 2. The 1× marginal-value table, with margins to the class boundaries

- x0 at exit: F_exit = 665,232,237.97 €, bar 13,215.20 €, read cycle 85.
- e = max(e_EV, bar_x + bar_x0). Because x0's bar is 13,215 €, **e exceeds e_EV on 21 of 22 rows** (18,689–40,730 €).
  Only n9_y2035_p0.25_e1 uses e_EV. At 2× every row used e_EV.
- Class rule: worse if dF_exit ≥ R + e; confirm-candidate if dF_exit ≤ −R + e; otherwise unresolved. The script
  recomputes e and the class on every row; both equal the recorded values.
- Margins: to worse = dF_exit − (R + e); to confirm = dF_exit − (−R + e).

| plan | source | I(x) € | read cycle | F_exit € | bar € | dF_exit € | e € | class | margin to worse € | margin to confirm € |
|---|---|---|---|---|---|---|---|---|---|---|
| n5_y2025_p0.25_e1 | screen | 317,957 | 83 | 665,284,016.75 | 14,097.6 | +51,778.78 | 27,312.79 | worse | +14,466.0 | +34,466.0 |
| n5_y2025_p0.25_e0.5 | screen | 191,018 | 86 | 665,289,190.36 | 5,648.7 | +56,952.39 | 18,863.87 | worse | +28,088.5 | +48,088.5 |
| **n5_y2030_p0.25_e1** | screen | 236,168 | 84 | 665,268,173.17 | 12,196.5 | +35,935.20 | 25,411.68 | worse | **+523.5** | +20,523.5 |
| **n5_y2030_p0.25_e0.5** | screen | 141,882 | 84 | 665,261,923.15 | 10,267.6 | +29,685.18 | 23,482.82 | **unresolved** | −3,797.6 | +16,202.4 |
| n5_y2035_p0.25_e1 | screen | 197,990 | 82 | 665,288,852.48 | 11,707.7 | +56,614.51 | 24,922.86 | worse | +21,691.6 | +41,691.6 |
| n5_y2035_p0.25_e0.5 | screen | 118,946 | 81 | 665,282,848.91 | 6,356.3 | +50,610.94 | 19,571.47 | worse | +21,039.5 | +41,039.5 |
| n7_y2025_p0.25_e1 (the unit) | screen | 317,957 | 83 | 665,290,557.86 | 13,188.5 | +58,319.89 | 26,403.68 | worse | +21,916.2 | +41,916.2 |
| n7_y2025_p0.25_e0.5 | screen | 191,018 | 84 | 665,287,328.42 | 9,513.1 | +55,090.45 | 22,728.26 | worse | +22,362.2 | +42,362.2 |
| n7_y2030_p0.25_e1 | screen | 236,168 | 83 | 665,281,040.71 | 9,908.5 | +48,802.74 | 23,123.67 | worse | +15,679.1 | +35,679.1 |
| **n7_y2030_p0.25_e0.5** | screen | 141,882 | 84 | 665,264,365.83 | 10,154.9 | +32,127.86 | 23,370.12 | **unresolved** | −1,242.3 | +18,757.7 |
| n7_y2035_p0.25_e1 | screen | 197,990 | 88 | 665,301,083.37 | 5,474.1 | +68,845.40 | 18,689.29 | worse | +40,156.1 | +60,156.1 |
| n7_y2035_p0.25_e0.5 | screen | 118,946 | 85 | 665,279,159.09 | 8,213.2 | +46,921.12 | 21,428.41 | worse | +15,492.7 | +35,492.7 |
| n9_y2025_p0.25_e1 | screen | 317,957 | 83 | 665,284,401.03 | 14,402.0 | +52,163.06 | 27,617.23 | worse | +14,545.8 | +34,545.8 |
| n9_y2025_p0.25_e0.5 | screen | 191,018 | 85 | 665,288,424.32 | 9,127.3 | +56,186.35 | 22,342.54 | worse | +23,843.8 | +43,843.8 |
| **n9_y2030_p0.25_e1** | screen | 236,168 | 84 | 665,275,381.55 | 12,852.5 | +43,143.58 | 26,067.73 | worse | **+7,075.9** | +27,075.9 |
| **n9_y2030_p0.25_e0.5** | screen | 141,882 | 83 | 665,259,593.22 | 12,027.6 | +27,355.25 | 25,242.80 | **unresolved** | −7,887.5 | +12,112.5 |
| n9_y2035_p0.25_e1 | screen | 197,990 | 87 | 665,295,462.31 | 4,271.5 | +63,224.34 | 17,494.05 | worse | +35,730.3 | +55,730.3 |
| n9_y2035_p0.25_e0.5 | screen | 118,946 | 84 | 665,287,933.80 | 13,771.7 | +55,695.83 | 26,986.93 | worse | +18,708.9 | +38,708.9 |
| n9_y2030_p0.5_e1 | ladder | 283,764 | 85 | 665,298,258.62 | 20,466.9 | +66,020.65 | 33,682.06 | worse | +22,338.6 | +42,338.6 |
| n9_y2030_p1_e2 | ladder | 567,528 | 81 | 665,425,423.82 | 8,199.4 | +193,185.85 | 21,414.64 | worse | +161,771.2 | +181,771.2 |
| n5_y2030_p0.5_e1 | ladder | 283,764 | 83 | 665,295,672.22 | 27,515.2 | +63,434.25 | 40,730.40 | worse | +12,703.8 | +32,703.8 |
| n5_y2030_p1_e2 | ladder | 567,528 | 79 | 665,437,364.14 | 7,927.8 | +205,126.17 | 21,143.02 | worse | +173,983.1 | +193,983.1 |

- **Every screen dF_exit is positive** (+27,355 to +68,845 €). The 3 unresolved rows are the 0.5 MWh units in 2030 at
  nodes 5, 7 and 9.
- **The unresolved rows** lie 1,242 € (n7), 3,798 € (n5) and 7,888 € (n9) below the worse boundary. They lie 12,112 to
  18,758 € above the confirm boundary.
- **The two closest worse rows** are n5_y2030_p0.25_e1 (+523.5 €) and n9_y2030_p0.25_e1 (+7,075.9 €). The next is the
  ladder row n5_y2030_p0.5_e1 (+12,703.8 €).
- **Ladder:** best duration 2 h (the best screen row is a 0.5 MWh unit); node-years (9, 2030) and (5, 2030); 0.5 MVA
  rows +63 to +66 k€, 1.0 MVA rows +193 to +205 k€.
- **Settled row:** the unit settles at dF_set = +78,737.79 € against x0 (V_pair 54.86 €; class worse), consistent with
  its exit class.

### 3. Exit-to-settled offsets at 1× (the D1 observation)

These are the plans with both an exit-depth run and a settled run in the 1× root: x0 (launch 1 exit; launch 7 settled)
and the reference unit (launch 3 exit; launch 7 settled).
- offset = F_set − F_exit;
- pair offset = [F_set(unit) − F_set(x0)] − [F_exit(unit) − F_exit(x0)].

| plan | F_exit € (read cycle, bar) | settled run's exit view (read cycle, F, bar) | equal bitwise | F_set € (K = 300) | offset € |
|---|---|---|---|---|---|
| x0 | 665,232,237.97 (85, 13,215.20) | (85, 665,232,237.97, 13,215.20) | **yes** (all 3 fields, float.hex) | 665,242,478.81 | **+10,240.84** |
| n7_y2025_p0.25_e1 | 665,290,557.86 (83, 13,188.48) | (83, 665,290,557.86, 13,188.48) | **yes** | 665,321,216.59 | **+30,658.74** |
| *pilot_staged (information only: no exit run; offset against its settled run's exit view)* | — | (82, 665,493,482.54, 7,502.05) | — | 665,499,465.61 | *+5,983.07* |

- The settled summaries' exit views also equal an independent re-read, `map_record(files, 'exit')` on the settled eval
  dirs, bitwise. This was checked on all three plans.
- **The settled runs reproduce the exit runs' F(k) bitwise on every exit-run cycle**: 85 of 85 for x0, 83 of 83 for the
  unit. The exit and settled runs are the same trajectory up to the exit point.

**Pair (unit − x0):**

| | € |
|---|---|
| dF_exit | +58,319.89 |
| dF_set | +78,737.79 |
| **pair offset** | **+20,417.90** |
| e_EV (D1 carry-over) | 17,494.05 → **exceeded by 2,923.85** |
| e actually used for the unit row: max(e_EV, 13,188.48 + 13,215.20) | **26,403.68** → covers the offset |
| staged − x0 pair offset (information only, exit view) | −4,257.77 |

**The Planner's zero-solve numbers are verified exactly:** x0 +10,241 €, unit +30,659 €, pair +20,418 €.

**Along the settled trajectories** (`report_eval.json` → `offsets.trajectory_at_cycles`):

| cycle | 80 | 83 | 85 | 90 | 100 | 125 | 150 | 175 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F(unit) − F(x0), € | 92,752 | 66,104 | 55,283 | 59,598 | 93,812 | 82,396 | 80,053 | 79,585 | 79,190 | 78,965 | 78,738 |

- From cycle 85 to 300 the pair difference ranges from **52,183 € (cycle 87) to 94,194 € (cycle 99)**.
- The exit reading compares the unit at cycle 83 with x0 at cycle 85, inside that swing.
- Terminal steps |F(300) − F(299)|: x0 1.86 €, unit 2.98 €, staged 36.87 €.

**At 2× the offsets had the opposite sign** (committed 2× evaluation):

| plan (2×) | offset F_set − F_exit € | pair offset vs x0 € |
|---|---|---|
| x0 (exit view of the pilot x0_i1) | −1,598.26 | — |
| n7_y2025_p0.25_e1 | −2,517.08 | −918.81 |
| n7_y2030_p0.25_e1 | −2,544.75 | −946.49 |
| n7_y2025_p0.25_e0.5 | −1,311.87 | +286.39 |
| n5_y2030_p1_e4 | −5,152.38 | −3,554.12 |

- At 2× every |pair offset| was ≤ 3,554 €. At 1× the one pair measured is +20,418 €.
- At 1× the exit read cycles are 79–88, against 119–185 at 2×. The exit bars are larger: x0's is 13,215 € at 1×,
  against 4,271 € at 2×.

### 4. Recorded predictions against outcomes

**2× (Q1–Q7, from `P5_15_S56_PHASE2X_REPORT.md` §5):**

| # | prediction | outcome | held |
|---|---|---|---|
| Q1 | every screen plan F_exit < F_exit(x0) | 12/18 below 0. The 6 failing rows are the 0.5 MWh units of 2025 and 2035, +10.0 to +14.7 k€, all inside e_EV | **FAILED** (recorded, non-stopping, R1) |
| Q2 | 0 harness_error, 0 local-solve failures | 0 and 0 over 52 evaluations | held |
| Q3 | best start ≠ x0 | n5_y2030_p1_e4 | held |
| Q4 | exit 0 inside 120 h | `phase_complete` at 25.41 h, in the v3 continuation (v2 stopped at exit 5 for review) | held (in the continuation) |
| Q5 | peak child RSS ≤ 3.29 GiB | 2.81 GiB | held |
| Q6 | efficiency fraction recorded | 22/22 → settled-direct | recorded |
| Q7 | exit run = exit view, bitwise, per candidate | equal on both; independent on candidate 1 only | held |

**1× (Q1–Q9, each by its frozen `evaluated_as`; `report_eval.json` → `predictions`):**

| # | prediction (verbatim) | outcome | held |
|---|---|---|---|
| Q1 | at most 9 of the 18 screen plans have dF_exit < 0 against 1× x0 | **0 of 18** (18 certified, 0 without a numeric dF_exit); min +27,355.25 €, max +68,845.40 € | held |
| Q2 | 0 harness_error and 0 local-solve failures over the phase | 0 harness_error over the 26 summaries read; 0 local-solve failures over 26 evaluations | held |
| Q3 | the best start is x0 | `pool.chosen_label` = x0; runner-up the unit at +78,737.79 € | held |
| Q4 | the search terminates (exit 0) inside the 60 h ceiling | `phase_complete`; cumulative 6.744 h | held |
| Q5 | peak child RSS ≤ 3.29 GiB | 2.834 GiB (pilot_staged, settled, launch 7) | held |
| Q6 | the efficiency fraction is recorded (no direction predicted) | recorded: 3/22 not worse (0 confirm-candidate, 3 unresolved, 19 worse) → exit-then-confirm; v1 band measure (information) 0/22 → exit-then-confirm | recorded |
| Q7 | as at 2×: for each candidate, the exit run's (read_cycle, F, bar) equals bitwise the exit_view of the candidate's settled summary | **0 candidates** (no row classed confirm-candidate): n_checked 0 | **vacuous**. The same check on x0 and the unit, which are not candidates, is equal bitwise (§3) |
| Q8 | price monotonicity: mean over the 18 screen plans of [dF_exit(1×) − dF_exit(2×)] > 0 | **mean +58,309.72 €, n = 18; 18 of 18 positive** (+32,219.73 to +88,074.23 €). 2× reference `phase2x_ev_v3/phase_results.json` sha256 `2f8778b5ed57…` = the sha256 the predictions file names; the copied 2× values equal that file's rows bitwise | held |
| Q9 | the within-cohort leak, loss-priced, is ≤ 25 € on every settled evaluation | **0.00 €** on x0, the unit and the staged plan (full-L priced 0.00 €; L = 0 MWh); 0 above the ceiling; 0 settled summaries without a numeric leak | held |

### 5. 2× against 1×, per screen plan

dF_exit at each level is taken against that level's x0 at exit depth.

| plan | dF_exit 2× € | class 2× | dF_exit 1× € | class 1× | 1× − 2× € |
|---|---|---|---|---|---|
| n5_y2025_p0.25_e1 | −25,997.41 | confirm-candidate | +51,778.78 | worse | +77,776.19 |
| n5_y2025_p0.25_e0.5 | +11,257.80 | unresolved | +56,952.39 | worse | +45,694.59 |
| n5_y2030_p0.25_e1 | −45,370.49 | confirm-candidate | +35,935.20 | worse | +81,305.69 |
| n5_y2030_p0.25_e0.5 | −12,566.58 | confirm-candidate | +29,685.18 | unresolved | +42,251.76 |
| n5_y2035_p0.25_e1 | −4,160.25 | confirm-candidate | +56,614.51 | worse | +60,774.76 |
| n5_y2035_p0.25_e0.5 | +10,046.74 | unresolved | +50,610.94 | worse | +40,564.20 |
| n7_y2025_p0.25_e1 | −20,932.80 | confirm-candidate | +58,319.89 | worse | +79,252.68 |
| n7_y2025_p0.25_e0.5 | +14,090.71 | unresolved | +55,090.45 | worse | +40,999.74 |
| n7_y2030_p0.25_e1 | −39,271.50 | confirm-candidate | +48,802.74 | worse | +88,074.23 |
| n7_y2030_p0.25_e0.5 | −8,300.65 | confirm-candidate | +32,127.86 | unresolved | +40,428.51 |
| n7_y2035_p0.25_e1 | −1,184.54 | confirm-candidate | +68,845.40 | worse | +70,029.95 |
| n7_y2035_p0.25_e0.5 | +14,701.39 | unresolved | +46,921.12 | worse | +32,219.73 |
| n9_y2025_p0.25_e1 | −20,220.47 | confirm-candidate | +52,163.06 | worse | +72,383.53 |
| n9_y2025_p0.25_e0.5 | +11,484.61 | unresolved | +56,186.35 | worse | +44,701.74 |
| n9_y2030_p0.25_e1 | −44,087.97 | confirm-candidate | +43,143.58 | worse | +87,231.55 |
| n9_y2030_p0.25_e0.5 | −8,043.79 | confirm-candidate | +27,355.25 | unresolved | +35,399.05 |
| n9_y2035_p0.25_e1 | −2,540.62 | confirm-candidate | +63,224.34 | worse | +65,764.96 |
| n9_y2035_p0.25_e0.5 | +10,973.77 | unresolved | +55,695.83 | worse | +44,722.06 |

- **x0 itself:** F_exit 812,009,208.71 (2×) against 665,232,237.97 (1×); F_set 812,007,610.45 against
  665,242,478.81.
- **The 1.0 multiplier was read back on launch 1** (TASKS 5e): DSO active flex cost 198.84 M€ at 1× against
  214.27 M€ at 2× for x0.
- **Selected plans:** 2×: n5_y2030_p1_e4, dF_set −182,404.11 € against x0, budget-limited. 1×: x0.

### 6. Hours (MEASURED)

**1×: 6.74 h of the 60 h ceiling (11.2 %); 7 launches, 0 interrupted.** The walls sum exactly to `cumulative_h`. All
stderr logs are empty.

| launch | stage | depth | evals | wall (h) | UTC 11-26 | child walls (h) | cycles run |
|---|---|---|---|---|---|---|---|
| 1 | x0 exit | exit | 1 | 0.550 | 11:06 → 11:39 | 0.55 | 85 |
| 2 | screen | exit | 5 | 0.778 | 13:27 → 14:13 | 0.67–0.78 | 82–86 |
| 3 | screen | exit | 5 | 0.761 | 14:13 → 14:59 | 0.67–0.76 | 81–84 |
| 4 | screen | exit | 5 | 0.777 | 14:59 → 15:46 | 0.69–0.78 | 83–88 |
| 5 | screen | exit | 3 | 0.685 | 15:46 → 16:27 | 0.64–0.68 | 83–87 |
| 6 | ladder | exit | 4 | 0.715 | 16:27 → 17:10 | 0.65–0.71 | 79–85 |
| 7 | pool settled | settled | 3 | 2.477 | 17:10 → 19:39 | 2.18–2.48 | 300 |

| stage | 1× (h) | 2× (h) |
|---|---|---|
| x0 exit | 0.55 | — (pilot exit view) |
| screen | 3.00 | 4.24 |
| ladder | 0.72 | 1.39 |
| candidate confirmations | 0 | 2.01 |
| candidate polls | 0 | 17.76 |
| pool settled | 2.48 | 0 (pilot cache) |
| search | 0 (18/18 from cache) | 0 (10/10 from cache) |
| **total** | **6.74** | **25.41** |

- **1× against the restated projection** (spec `phase.ceiling.projection_restated_1x`): with no confirm-candidate in the
  screen, PROJECTED 7.8–22.0 h (exit-then-confirm 7.8–11.1 h). **Measured 6.74 h, below the range.**
- The reason: exit launches of 5 ran 0.76–0.78 h at 1×, against 0.97–1.39 h at 2×. The exit runs certified at cycles
  79–88, against 119–185 at 2×.
- The gap between launch 1 and launch 2 (11:39 → 13:27) was the first-block hold. It is not machine hours.
- Peak child RSS: exit 2.67–2.75 GiB (n = 23); settled 2.78–2.83 GiB (n = 3).

### 7. Leak, slack, gates and ESSO recovery at 1×

- **Settled runs (x0, unit, staged): all `settled`; all gates pass** (d1 efficiency equality, ESSO slack, interface P
  ceiling, network closure, status and guards exact). Every settled check is true.

| settled run | S € | PR € | G € | A € | leak (loss-priced) € | ESSO slack € |
|---|---|---|---|---|---|---|
| x0 | 14.96 | 140.88 | 0.00 | 249.44 | 0.00 | 0.0 |
| n7_y2025_p0.25_e1 | 29.20 | 148.30 | 3.28 | 245.06 | 0.00 | 1.5e-10 |
| pilot_staged | 333.45 | 198.09 | 6.80 | 363.61 | 0.00 | 2.1e-10 |

- **ESSO recovery events at 1×: 0** over the 26 evaluations. 10 of them hold a 0.5 MWh cohort. The event file rows equal
  the record count on every evaluation, and every guard is exact on all 26.
  - At 2× there were 6 events on 4 plans, all recovered at tier 1, all at a node with a 0.5 MWh cohort.
- **Terminal-step-to-threshold ratio:** the 2× evaluation did not report one, so there is no 2× definition to reuse.
  It is not computed here. The settled terminal steps are given in §3.

### 8. Evidence

By the S56 selection rule (`p515_s56_ev_harness_tests.committed_by_rule`):
- excluded directories: `/planning_read/`, `/results/`, `/esso_capture/`, `/scratch`;
- excluded suffixes: `.pkl`, `.xlsx`, `.pdf`, `.png`, `.nl`;
- size limit: ≤ 1,300,000 bytes.

| manifest | files | committed | sha256 |
|---|---|---|---|
| `phase1x_ev_end_evidence_manifest_sha256.json` (the 1× root, every file; plus `phase1x_ev_end_report_eval/` and its logs) | 11,028 (4.35 GB) | 877 (183.3 MB) | `2d038a91b146…` |

- **Committed:**
  - freeze and launch logs, `phase_state.json`, `phase_results.json` and its manifest;
  - every launch record, launch manifest, campaign spec, heartbeat and init identity;
  - per eval dir, every mapping input (26/26 each): `evaluation_record.json`, `per_cycle_record.jsonl`,
    `convergence_depth_tail_state.json`, `s56_cost_breakdown.json`, `exit_code.txt`,
    `esso_recovery_events_s39_D.jsonl`, `network_failures_s39_D.jsonl`, plus `s56_pricing_terminal.json` (3/3
    settled);
  - the other sidecars ≤ 1.3 MB.
- **Hash-only:** `pf_entry_stride_s39_D.jsonl`, `ess_entry_stride_baseline.jsonl`, `interface_duals_per_cycle.jsonl`,
  `network_ipopt_solve_records*.jsonl`, the ESSO pickles, 3 of 26 `child_stdout.log` / `stdout_s39_D.log` /
  `g_s39_D.json` (the settled runs; over 1.3 MB), `results/`, `esso_capture/` and `planning_read/`.
- **The 1× campaign specs are tracked**, after the zero-solve check (`report_eval.json` → `G`):
  - their keys (26) are a subset of the launch records' keys, and equal the 26 evaluation keys;
  - a later 1× freeze must declare `phase1x_ev_v1` as a cache root regardless, because the launch records alone collide
    (7 files). Declared, the scan finds 0 collisions over 361 files.
- 3 of the committed files were already tracked: the spec and the two freeze logs.

## Not confirmed

- **The 1× offset beyond one pair.** The exit-to-settled offset at 1× is measured on x0 and the unit only. The staged
  plan's offset (+5,983 €; pair −4,258 €) is read against its settled run's exit view, not against an exit run. The
  prefix identity on x0 and the unit suggests the two would agree, but it is not an independent check. How the 3
  unresolved and the 2 closest worse rows would settle is unmeasured (Decision 1).
- **t_SP.** It is extrapolated from two SP cycles of one plan (memory probe) and from EV cycle counts. SP cycle counts to
  certification, and the depth at which SP evaluations will be read (exit or K = 300), are not measured or decided.
  The wait-and-see figure also assumes that a single-scenario evaluation costs the same as an EV evaluation.
- **"SP calibration".** Plan v4.1 §8 has no line for it. Planner reading (recorded in Decision 2): it is calibration
  (a), which TASKS says "gates only the SP evaluations". The SP part of the step 8 pilot is listed as its own row
  (77–113 h, extrapolated).
- **Terminal-step-to-threshold ratio.** It is not reported at either level. Searched for a 2× definition:
  `p515_s56_phase2x_report_eval.py`, `P5_15_S56_PHASE2X_REPORT.md`, `P5_15_S56_PILOT_REPORT.md` and
  `p515_s56_pilot_report_eval.py` (terms "terminal-step", "threshold", "ratio"). None was found.
- **Collision scope (§8).** Searched: the tracked artefacts in the driver's scan scope (361) plus the 1× root's
  spec / launch artefacts. Not searched: untracked artefacts elsewhere (other result dirs, worktrees,
  `data/SRP1/Results/P56A`).
- **The 2× rows in §1 and §4** are copied from `P5_15_S56_PHASE2X_REPORT.md` and its committed evaluation. This report
  re-verified only the 2× evaluation's sha256 and the Q8 reference values, not the 2× replay.
