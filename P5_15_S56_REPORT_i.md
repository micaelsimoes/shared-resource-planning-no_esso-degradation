# P5.15 S56 report (i): search driver ready, EV pilot result, measured hours (2026-11-24)

**Report point (i)** of the working mode (author and Expert, 2026-11-23). The searches are not launched: their machine
allocation is the author's decision. The Expert has nothing to rule. Everything below is measured unless marked
projected.

Sources:

| item | file | commit |
|---|---|---|
| pilot report | `P5_15_S56_PILOT_REPORT.md` | `1198d570` |
| pilot spec | `pilot_ev_v1/pilot_spec_92547279.json` (PP1–PP8 frozen before the run) | |
| driver design | `S56_SEARCH_DRIVER_DESIGN.md` | |
| harness | `P5_15_S56_EV_HARNESS_REPORT.md` | |
| memory probe | `P5_15_S56_MEMORY_PROBE_REPORT.md` | |

## Decisions needed (author)

1. **Machine allocation for the 2× EV phase.** The phase is the screen, then the screen candidates, then one search from
   the best start.
   - Projected: **30–120 h** of exclusive machine time. That is a screen of about 6 h, about 13 h per candidate, and a
     search of 12–97 h.
   - **Recommendation:** run the 2× phase, then report its measured hours before the 1× phase and the two 2× control
     searches are launched. Those three are projected at a further 60–300 h.
2. **How the runs are executed.**
   - **(a) You run them in Terminal with the app closed, 5 at a time** (recommended). The command re-launches each block
     until the search ends. A session exit or a reboot costs at most the block in flight, and the driver resumes from its
     own records.
   - **(b) I run them here with the app open, 4 at a time.** The measured memory per slot is 3.29 GiB, so 4 slots need
     13.2 GiB; about 14 GiB was free with the app open. That is a thin margin, and the phase takes about 25 % longer.
3. **How many screen candidates are confirmed and polled per price level.** Each costs about 13 h: a settled
   confirmation plus a complete poll of about 39 neighbours.
   - **Recommendation:** the 2 best, distinct in node and year.
   - All screen entries are reported in the marginal-value table either way.

## Blocked on the author

- **Push:** `e55eed90` … the commit of this report. Push now; no run is in flight.

## Found (EV model, 2025/2030/2035 grid, 2× flexibility price)

**Objective convention:** F = I(x) + gross operational cost − terminal-value credit, in € of total discounted planning
cost.

**Pilot:** 7 runs to 300 cycles: x0 and the reference unit (0.25 MVA / 1 MWh at DN 7, 2025), each from three ADMM
initialisations (ρ0 × 1, 0.5, 2), plus one staged plan:
- 0.5/1 at node 5 in 2030;
- 0.25/1 at node 7 in 2025;
- 0.25/0.5 at node 7 in 2030;
- 0.25/0.5 at node 9 in 2035.

The runs were launched by the author in two attached blocks. All exited 0, with 0 local-solve failures.

**Predictions PP1–PP8: all held.**

| # | prediction | outcome |
|---|---|---|
| PP1 | all runs reach 300 cycles; would-be certification cycle ≤ 200 | 109–143 |
| PP2 | settled at 300 | 7/7 |
| PP3 | S ≤ 500 € at 300 | 13–172 € |
| PP4 | largest stopping-vs-settled pair offset ≤ 15,000 € | 8,747 € |
| PP5 | unit − x0 spread across 3 × 3 initialisations ≤ 2,500 € | 1,695 € at K = 175, 382 € at K = 300 |
| PP6 | battery leak ≤ 25 € at the settled point | **0 €** on every run |
| PP7 | peak RSS ≤ 4 GiB | 2.77–2.86 GiB |
| PP8 | 22.9–27.1 s/cycle; ψ ≤ 1.3 | ψ = 1.09 |

**Settled pair differences at K = 300:**

| pair | ΔF |
|---|---|
| unit − x0 | −21.9 to −22.2 k€ |
| staged − x0 | −34.0 to −34.3 k€ |
| staged − unit | about −12.1 k€ |

- V_pair is 49–362 €, against R/4 = 2,500 €, and the absolute guard A + A′ is ≤ 189 €.
- All 15 pairs are determinate under plan v4.1 §3 (|ΔF| ≥ R = 10,000 €). So at 2× in the EV model, storage pays, and
  the staged plan beats the single reference unit.
- These are pilot plans, not search results.

**Hours.**
- An evaluation takes about 1.0 h to its stopping point and about 2.0–2.4 h to cycle 300, at 5 runs in parallel.
- Contention is 1.18–1.22×. The two blocks took 2.63 h and 2.29 h, against 2.50 h projected.

**Settled-point check.** Its first evaluation failed on all 7 runs, on one identity check. The per-block settlement
sum differed from the record total by 5e-8 to 5e-7 €, which is the summation order of two sides of about ±1.3e9 €.
The check now uses the Expert-adopted identity tolerance (`28ce9f08`), and the re-evaluation is complete.

## Planner decisions logged (working mode: numerical and harness details)

- **The settled confirmation runs to K_EV = 300, not the frozen rule's 175.** Between cycles 175 and 300, F keeps
  falling by 92–2,550 € per run, and pair differences move by up to 2,458 €. That is 6.8× the largest V_pair, so the
  10-cycle window S under-reports path drift (CLAUDE.md's stopping-slack-versus-path-divergence rule). One run's S also
  oscillates through the 500 € line just after 175, with an amplitude of about ±340 € and a period of about 18–20
  cycles. At 300 every check holds from cycle 183 on. The staged − unit pair clears R at K = 175 by only 632 €, less
  than its own drift.
  - Cost: about 1 h more per confirmation, already included in the projections above.
- **Stopping-point error bar e_EV = 17,494 €**, by the frozen rule (2 × the largest pair offset). It replaces the
  interim 15,000 €.
  - A candidate is classed "worse" if dF_exit ≥ 27.5 k€.
  - It is sent to a settled confirmation if dF_exit ≤ +7.5 k€.
- **Memory per slot: 3.29 GiB** (measured peak × 1.15).
- **Screen:** 3 nodes × 3 years × 2 durations = 18 evaluations, plus a size ladder of 0.5 and 1.0 MVA at the two best
  node-years (4 evaluations). The marginal-value table and the candidates are built separately at 1× and 2×.
- **The earlier leak note to the Expert is withdrawn.** The leak is 0 € at the settled point, so the row-tolerance
  question does not arise.

## Not confirmed

- **Drift beyond cycle 300:** not measured.
- **Projected hours:** the search hours depend on the number of moves and where the improvers sit in each poll, which
  only the search measures.
- **Leak on plans with simultaneous multi-cohort operation under TV:** the pilot plans showed none. Searches report it
  per evaluation.
- **Tail-depth contention:** gate F measured contention on cycles 1–3 only.
