# P5.15 S56 — EV 2× phase: end-of-phase report (Worker report, 2026-11-26)

Authority: `TASKS.md` step 5b ("COMPLETE 2026-11-26 03:27 UTC … End-of-phase report Worker active"); the author's
decision (1) on report (i) ("report measured hours … at the end of the phase, before the 1× phase and the 2× controls");
the Planner's end-of-phase task of 2026-11-26.

**Roots.**
- v2: `data/SRP1/Results/P515S56/phase2x_ev_v2/`, spec `18a224d1`, 10 launches, stopped at exit 5
  (`stop_for_review_not_settled`).
- v3: `data/SRP1/Results/P515S56/phase2x_ev_v3/`, continuation spec `600bbceb`, 3 launches, `phase_complete`.

**Solve claim.** This report ran zero solves. The evaluation script `p515_s56_phase2x_report_eval.py` ran under an
armed `SolveProfileGuard(permitted=())`, verified at 0, with `pickle.load`/`loads` blocked (0 calls); its manifest mode
also verified the guard that the imported selection-rule module installs (0). No file was written inside either phase
root. No code or spec was changed; `TASKS.md` is untouched.

**Objective convention (every € figure and every table):** F = I(x) + gross_operational_cost − TV credit, in € of
total discounted planning cost (oracle mapping v2; gross is settlement-excluded).
- dF_exit = F_exit(x) − F_exit(x0).
- dF_set = F_set(x) − F_set(reference), where the reference is named in each table.

**Measured and projected are kept apart.** Every projected figure is labelled PROJECTED. All other figures are read
from the phase records.

## Decisions needed (author)

1. **The 2× controls: run, build, or record as decided at zero solves?** (§7a)
   - **What is already fixed.** Both controls start from n5_y2030_p1_e4 (the same start rule). Its neighbourhood
     has |N| = 10, and every neighbour already has a settled K = 300 record from candidate 1's complete poll. None
     of them is a settled improver: 9 are worse and 1 is unresolved.
   - Neither the complete-poll rule nor the reversed class order can therefore accept a move. A settled-direct
     control ends at its first poll, with 0 moves and the same final plan as the search.
   - Options:
     - (a) **Settled-direct controls that reuse the v2 and v3 records.** PROJECTED 0 h of machine time: 0 new
       evaluations. This needs a build: no settled-direct control mode exists, and no complete-poll control is
       implemented (scoped search, §7a).
     - (b) **The reversed-order control on the existing plain search.** PROJECTED 2.9–4.2 h for that control, from
       3 exit launches. The plain search polls at exit depth, and its start pool needs exit records that x0 and the
       unit do not have. The complete-poll control would still need a build.
     - (c) **Record the controls as decided at zero solves**, with the evidence above, and run nothing.
   - **Recommendation:** the controls first, as the Planner proposed, since they cost about 0 machine hours. In
     practice (a) and (c) both cost 0 machine hours. Choosing between them, or a 1× rerun of the controls (plan v4.1
     §4: only if the 1× search moves), is the author's call.
2. **The 1× phase.** PROJECTED 16.7–32.8 h with two candidates and no search move (§7b):
   - settled-direct branch: 24.0–32.8 h;
   - exit-then-confirm branch: 16.7–23.5 h, plus 2.0–2.7 h per confirmation;
   - if no screen row is a confirm-candidate and the start is x0: 7.8–11.1 h (exit-then-confirm) or 15.9–22.0 h
     (settled-direct);
   - each search move adds about 16–22 h (settled-direct) or 7.8–11.1 h plus confirmations (exit-then-confirm), at
     the plan's median |N| of 39.

   **Nothing carries over from 2×**: the price level is part of the eval key (§8). Over 4 h, the 1× phase needs its
   own frozen spec with recorded predictions.
3. **Order.**
   - (i) Controls first, then 1×. This costs 0 h (a or c), or about 3–4 h (b), before 1×.
   - (ii) 1× first. The plan runs the controls at 1× too if the 1× search moves, and lists their hours first.

   **Recommendation: (i)**, on the Planner's rule (controls first when they cost about 0 h).

## Blocked on the author

Nothing is held at the machine. The three decisions above are needed before anything runs next. None of them has a
spec yet: the 1× phase, the controls and any control build are not authorized.

## Changed

| commit | content |
|---|---|
| (this commit) | `p515_s56_phase2x_report_eval.py` (a new read-only, zero-solve evaluation script; no existing code was edited); its output `phase2x_ev_end_report_eval/report_eval.json` (`4427e1421e0c…`, 45/45 checks) and logs (stdout `ef2d2baf3e9d…`, stderr empty); the hash inventory `phase2x_ev_end_evidence_manifest_sha256.json` (`4bc3505a00a7…`) and its logs (stdout `8a2bd3ad46fe…`, stderr empty); the evidence of both phase roots selected by rule (§8); this report |

- No spec, code path, `TASKS.md` entry or phase-root file was changed.
- The script's sha256 is `2ad09b4a4738…`, recorded in `report_eval.json`.
- `phase_results.json`: v2 `01470afc5319…`, v3 `2f8778b5ed57…`.

## Found

### 1. Hours (MEASURED)

**Total: 25.41 h of the 120 h ceiling (21.2 %).** v2 used 17.99 h over 10 launches; v3 used 7.42 h over 3 launches.
There were 0 interrupted launches. Each wall is the launch record's `wall_s`; the sum of the walls equals each
phase's `cumulative_h` exactly, and v3's `carried_h` equals v2's `cumulative_h`.

| phase | launch | stage | depth | evals | wall (h) | UTC start → end | child walls (h) |
|---|---|---|---|---|---|---|---|
| v2 | 1 | screen | exit | 5 | 1.094 | 11-24 20:35 → 21:41 | 1.00–1.09 |
| v2 | 2 | screen | exit | 5 | 1.083 | 11-24 21:46 → 22:51 | 1.00–1.08 |
| v2 | 3 | screen | exit | 5 | 1.093 | 11-24 22:51 → 23:57 | 1.01–1.09 |
| v2 | 4 | screen | exit | 2 | 0.972 | 11-24 23:57 → 11-25 00:55 | 0.95–0.97 |
| v2 | 5 | ladder | exit | 4 | 1.392 | 11-25 00:55 → 02:18 | 0.98–1.39 |
| v2 | 6 | candidate confirmation | settled | 1 | 2.014 | 02:18 → 04:19 | 2.01 |
| v2 | 7 | candidate 1 poll | settled | 5 | 2.484 | 04:19 → 06:48 | 2.40–2.48 |
| v2 | 8 | candidate 1 poll | settled | 4 | 2.567 | 06:49 → 09:23 | 2.46–2.57 |
| v2 | 9 | candidate 2 poll | settled | 5 | 2.576 | 09:23 → 11:57 | 2.49–2.57 |
| v2 | 10 | candidate 2 poll | settled | 5 | 2.715 | 11:57 → 14:40 (exit 5) | 2.60–2.71 |
| v3 | 1 | candidate 2 poll | settled | 5 | 2.569 | 11-25 20:02 → 22:36 | 2.45–2.57 |
| v3 | 2 | candidate 2 poll | settled | 5 | 2.720 | 22:36 → 11-26 01:19 | 2.61–2.72 |
| v3 | 3 | candidate 2 poll | settled | 1 | 2.134 | 01:19 → 03:27 (exit 0) | 2.13 |

**Per stage**, against the v2 spec's frozen projection. The PROJECTED column is the case that ran:
`settled_direct|cand_*|search_low`, i.e. settled-direct with no search move. Per block it uses exit 1.292 h and
settled 3.094 h.

| stage | v2 (h) | v3 (h) | total (h) | launches / evals | PROJECTED (blocks × per block) |
|---|---|---|---|---|---|
| screen | 4.24 | 0 | 4.24 | 4 / 17 | 5.17 (4 × 1.292) |
| ladder | 1.39 | 0 | 1.39 | 1 / 4 | 1.29 (1) |
| candidate confirmations | 2.01 | 0 | 2.01 | 1 / 1 | 3.09 (1 × 3.094) |
| candidate polls | 10.34 | 7.42 | 17.76 | 7 / 30 | 24.75 (8) |
| pool | 0 | 0 | 0 | 0 | — |
| search | 0 | 0 | 0 | 0 / 0 | 30.94 (10) |
| **total** | **17.99** | **7.42** | **25.41** | 13 / 52 | **65.24** |

Why the measured total is 39 % of the projection:
- **Per-launch walls were lower than projected:**
  - exit launches 0.97–1.39 h, against 1.29 h projected;
  - settled launches 2.01–2.72 h, against 3.09 h.
- **The candidate polls needed 7 blocks, not 8.**
- **The search needed no block.** The projection assumed a search from the pilot staged plan, with 46 uncached
  neighbours. The best start was candidate 1, whose complete poll was already in the cache.

Elapsed calendar time was 30.9 h, from 11-24 20:35 to 11-26 03:27. Not counted as machine hours: the first-block hold
(5 min) and the gap between v2's stop and v3's first launch (5.4 h).

Child walls: exit 0.95–1.39 h (mean 1.06 h, n = 21); settled 2.01–2.72 h (mean 2.52 h, n = 31).

### 2. Marginal-value table, under the corrected guard (the v3 re-map)

The 22 rows are the v3 phase state's table, which the replay re-derives identically.
- x0 at exit: F_exit = 812,009,208.71 €, bar 4,271.36 €. This is the exit view of the pilot's settled x0_i1, whose
  F_set = 812,007,610.45 €.
- e = max(e_EV, bar_x + bar_x0) = **e_EV = 17,494.05 € on every row**. The largest bar sum is 13,356.7 €
  (n7_y2035_p0.25_e0.5).
- Class rule: worse if dF_exit ≥ R + e; confirm-candidate if dF_exit ≤ −R + e = +7,494.05 €; otherwise unresolved.
- **"Settled" columns:** the plans later evaluated at K = 300, classified against x0's settled record by the
  plan v4.1 §3 rule (`P.classify_settled`). dF_set = F_set(x) − F_set(x0).

| plan | source | I(x) € | F_exit € | bar € | dF_exit € | class | F_set € | dF_set € | V_pair € | settled class |
|---|---|---|---|---|---|---|---|---|---|---|
| n5_y2025_p0.25_e1 | screen | 317,957 | 811,983,211.30 | 4,154.7 | −25,997.41 | confirm-candidate | | | | |
| n5_y2025_p0.25_e0.5 | screen | 191,018 | 812,020,466.51 | 1,514.2 | +11,257.80 | unresolved | | | | |
| n5_y2030_p0.25_e1 | screen | 236,168 | 811,963,838.23 | 1,327.3 | −45,370.49 | confirm-candidate | | | | |
| n5_y2030_p0.25_e0.5 | screen | 141,882 | 811,996,642.13 | 1,401.0 | −12,566.58 | confirm-candidate | | | | |
| n5_y2035_p0.25_e1 | screen | 197,990 | 812,005,048.46 | 1,095.8 | −4,160.25 | confirm-candidate | | | | |
| n5_y2035_p0.25_e0.5 | screen | 118,946 | 812,019,255.46 | 4,923.3 | +10,046.74 | unresolved | | | | |
| n7_y2025_p0.25_e1 (unit; exit view of pilot unit_i1) | screen | 317,957 | 811,988,275.92 | 1,406.1 | −20,932.80 | confirm-candidate | 811,985,758.84 | −21,851.61 | 48.6 | improver |
| n7_y2025_p0.25_e0.5 | screen | 191,018 | 812,023,299.42 | 1,704.3 | +14,090.71 | unresolved | 812,021,987.55 | +14,377.10 | 80.8 | worse |
| n7_y2030_p0.25_e1 | screen | 236,168 | 811,969,937.22 | 7,757.7 | −39,271.50 | confirm-candidate | 811,967,392.47 | −40,217.98 | 43.5 | improver |
| n7_y2030_p0.25_e0.5 | screen | 141,882 | 812,000,908.06 | 2,074.5 | −8,300.65 | confirm-candidate | | | | |
| n7_y2035_p0.25_e1 | screen | 197,990 | 812,008,024.17 | 1,653.5 | −1,184.54 | confirm-candidate | | | | |
| n7_y2035_p0.25_e0.5 | screen | 118,946 | 812,023,910.11 | 9,085.3 | +14,701.39 | unresolved | | | | |
| n9_y2025_p0.25_e1 | screen | 317,957 | 811,988,988.24 | 2,840.5 | −20,220.47 | confirm-candidate | | | | |
| n9_y2025_p0.25_e0.5 | screen | 191,018 | 812,020,693.32 | 816.5 | +11,484.61 | unresolved (v2 as run: uncertified) | | | | |
| n9_y2030_p0.25_e1 | screen | 236,168 | 811,965,120.74 | 1,544.0 | −44,087.97 | confirm-candidate | | | | |
| n9_y2030_p0.25_e0.5 | screen | 141,882 | 812,001,164.92 | 2,014.8 | −8,043.79 | confirm-candidate | | | | |
| n9_y2035_p0.25_e1 | screen | 197,990 | 812,006,668.10 | 907.0 | −2,540.62 | confirm-candidate | | | | |
| n9_y2035_p0.25_e0.5 | screen | 118,946 | 812,020,182.48 | 1,028.5 | +10,973.77 | unresolved | | | | |
| n5_y2030_p0.5_e2 | ladder | 472,336 | 811,917,782.41 | 3,197.0 | −91,426.30 | confirm-candidate | | | | |
| **n5_y2030_p1_e4** | ladder | 944,672 | 811,830,358.72 | 809.9 | −178,849.99 | confirm-candidate | 811,825,206.34 | −182,404.11 | 469.4 | improver |
| n9_y2030_p0.5_e2 | ladder | 472,336 | 811,924,597.92 | 1,731.7 | −84,610.79 | confirm-candidate | | | | |
| n9_y2030_p1_e4 | ladder | 944,672 | 811,840,298.88 | 1,633.3 | −168,909.83 | confirm-candidate | | | | |

- **Classes:** 16 confirm-candidate, 6 unresolved, 0 worse, 0 uncertified.
- **Every unresolved row is a 0.5 MWh unit in 2025 or 2035**, at each of the three nodes.
- **The ladder:** best duration 4 h; best node-years (5, 2030) and (9, 2030).
- **The settled values confirm every exit sign they can check.**
  - Rows classed confirm-candidate at exit: n5_y2030_p1_e4, the unit, n7_y2030_p0.25_e1. Each settles as a
    determinate improver against x0.
  - The unresolved row n7_y2025_p0.25_e0.5 settles as worse, +14,377 €.
  - Every settled pair passes V_pair ≤ 2,500 €, A ≤ 5,000 € and every gate.

### 3. Candidates

The candidate rule is the best row, then the next row whose node and year both differ from every pick. It picked
**n5_y2030_p1_e4** (candidate 1) and **n7_y2025_p0.25_e1** (candidate 2, which is the reference unit).

**Confirmation against x0.** Settled at K = 300; dF_set = F_set(cand) − F_set(x0).

| candidate | dF_exit € | dF_set € | offset \|dF_exit − dF_set\| € | V_pair € | A sum € | gates | result |
|---|---|---|---|---|---|---|---|
| n5_y2030_p1_e4 | −178,849.99 | −182,404.11 | 3,554.12 (≤ e 17,494.05) | 469.36 | 197.66 | pass | accepted (confirmed) |
| n7_y2025_p0.25_e1 | −20,932.80 | −21,851.61 | 918.81 | 48.57 | 70.58 | pass | accepted (confirmed) |

**Exit-view check (Q7, R4): equal on both candidates. It is an independent check on candidate 1 only.**
- **Candidate 1.** The exit run (v2 launch 5) gives read_cycle 185, F 811,830,358.7184356 (`0x1.831c62b5bf5b3p+29`)
  and bar 809.940866112709. The settled run's exit view (v2 launch 6) is equal in all three fields, bitwise.
- **Candidate 2.** Its "exit run" is itself the exit view of the pilot's settled unit_i1 (`served_as:
  exit_view_of_settled_cache_record`), so the check compares that view with itself: equal by construction, not a
  held-out test.

**Complete poll of candidate 1, n5_y2030_p1_e4.** |N| = 10, all 10 classified (9 evaluated in v2 launches 7–8, x0
from the pilot). dF_set = F_set(x) − F_set(candidate 1). No improver.

| pos | neighbour | move | class | dF_set € | V_pair € | A sum € | failed conditions |
|---|---|---|---|---|---|---|---|
| 0 | n5_y2030_p0.75_e3 | diag −1−2 (resize) | worse | +45,845.49 | 649.4 | 274.3 | — |
| 1 | n5_y2030_p1.25_e4 | zP+1 (resize) | **unresolved** | **+2,668.08** | 1,386.2 | 1,295.7 | \|dF\| < R |
| 2 | n5_y2030_p1_e3.5 | zE−1 (resize) | worse | +24,046.91 | 760.8 | 303.7 | — |
| 3 | n5_y2035_p1_e4 | shift 2030→2035 | worse | +169,397.43 | 630.0 | 220.6 | — |
| 4 | n5_y2030_p1_e3.5 + n5_y2035_p0.25_e0.5 | redistribute | worse (v2 as run: not_settled) | +51,578.02 | 830.2 | 499.2 | — |
| 5 | n5_y2030_p1_e3.5 + n7_y2030_p0.25_e0.5 | redistribute | worse | +17,989.70 | 789.5 | 307.5 | — |
| 6 | n5_y2030_p1_e3.5 + n7_y2035_p0.25_e0.5 | redistribute | worse | +37,015.38 | 747.6 | 301.0 | — |
| 7 | n5_y2030_p1_e3.5 + n9_y2030_p0.25_e0.5 | redistribute | worse | +12,972.26 | 861.7 | 310.6 | — |
| 8 | n5_y2030_p1_e3.5 + n9_y2035_p0.25_e0.5 | redistribute | worse | +34,960.01 | 804.3 | 305.2 | — |
| 9 | x0 | remove | worse | +182,404.11 | 469.4 | 197.7 | — |

**Complete poll of candidate 2, n7_y2025_p0.25_e1.** |N| = 22, all 22 classified:
- positions 0–4 evaluated in v2 launch 9, and 5–9 in v2 launch 10;
- positions 10–20 in v3 launches 1–3;
- x0 (position 21) from the pilot.

dF_set = F_set(x) − F_set(unit). Candidate polls only poll, so **8 improvers are listed and none is accepted**.
There are 8 worse and 6 unresolved; 0 are not_settled under the corrected guard.

| pos | neighbour | move | class | dF_set € | V_pair € | A sum € | failed conditions |
|---|---|---|---|---|---|---|---|
| 0 | n7_y2025_p0.25_e0.5 | zE−1 | worse (v2 as run: not_settled) | +36,228.71 | 74.1 | 110.9 | — |
| 1 | n7_y2025_p0.5_e1 | zP+1 | worse | +51,632.40 | 180.9 | 138.4 | — |
| 2 | n7_y2025_p0.5_e1.5 | diag +1+1 | worse | +11,378.41 | 200.2 | 184.2 | — |
| 3 | n7_y2025_p0.5_e2 | diag +1+2 | improver | −22,057.69 | 84.9 | 98.7 | — |
| 4 | n7_y2030_p0.25_e1 | shift 2025→2030 | improver | −18,366.37 | 44.7 | 68.2 | — |
| 5 | n5_y2025_p0.25_e0.5 + unit | activate | unresolved (v2 as run: not_settled, the stop) | +6,530.28 | 181.5 | 113.4 | \|dF\| < R |
| 6 | n5_y2025_p0.25_e1 + unit | activate | improver | −29,465.44 | 149.2 | 103.1 | — |
| 7 | n5_y2030_p0.25_e0.5 + unit | activate | improver | −10,313.15 | 137.2 | 105.0 | — |
| 8 | n5_y2030_p0.25_e1 + unit | activate | improver | −45,564.50 | 110.2 | 94.6 | — |
| 9 | n5_y2035_p0.25_e0.5 + unit | activate | worse | +11,481.25 | 77.2 | 73.7 | — |
| 10 | n5_y2035_p0.25_e1 + unit | activate | unresolved | −3,156.02 | 81.1 | 83.8 | \|dF\| < R |
| 11 | unit + n7_y2030_p0.25_e0.5 | activate | unresolved | −5,188.72 | 85.2 | 103.3 | \|dF\| < R |
| 12 | unit + n7_y2030_p0.25_e1 | activate | improver | −39,849.12 | 90.5 | 100.5 | — |
| 13 | unit + n7_y2035_p0.25_e0.5 | activate | worse | +13,935.89 | 83.8 | 101.0 | — |
| 14 | unit + n7_y2035_p0.25_e1 | activate | unresolved | −252.86 | 81.6 | 98.0 | \|dF\| < R |
| 15 | unit + n9_y2025_p0.25_e0.5 | activate | worse | +11,178.09 | 140.5 | 143.2 | — |
| 16 | unit + n9_y2025_p0.25_e1 | activate | improver | −24,971.29 | 91.0 | 97.5 | — |
| 17 | unit + n9_y2030_p0.25_e0.5 | activate | unresolved | −7,092.21 | 80.8 | 90.9 | \|dF\| < R |
| 18 | unit + n9_y2030_p0.25_e1 | activate | improver | −42,249.51 | 67.0 | 93.4 | — |
| 19 | unit + n9_y2035_p0.25_e0.5 | activate | worse | +13,074.15 | 72.7 | 76.4 | — |
| 20 | unit + n9_y2035_p0.25_e1 | activate | unresolved | −1,627.35 | 67.8 | 81.5 | \|dF\| < R |
| 21 | x0 | deactivate | worse | +21,851.61 | 48.6 | 70.6 | — |

- **The best neighbour of candidate 2:** n5_y2030_p0.25_e1 + unit, F_set 811,940,194.34 €. That is +114,988 €
  above candidate 1's F_set.
- **Exit and settled agree in sign against x0 on all four plans that have both records.**

| plan | dF_exit € | dF_set € | \|dF_exit − dF_set\| € |
|---|---|---|---|
| n5_y2030_p1_e4 | −178,849.99 | −182,404.11 | 3,554.12 |
| n7_y2025_p0.25_e1 (the unit) | −20,932.80 | −21,851.61 | 918.81 |
| n7_y2030_p0.25_e1 | −39,271.50 | −40,217.98 | 946.48 |
| n7_y2025_p0.25_e0.5 | +14,090.71 | +14,377.10 | 286.39 |

### 4. Start pool and search

**Pool.** The rule is argmin settled F over the eligible members.

| member | plan | F_set € | eligible |
|---|---|---|---|
| x0 | x0 | 812,007,610.45 | yes |
| reference_unit (= candidate 2, alias) | n7_y2025_p0.25_e1 | 811,985,758.84 | yes |
| pilot_staged | n5_y2030_p0.5_e1 + n7_y2025_p0.25_e1 + n7_y2030_p0.25_e0.5 + n9_y2035_p0.25_e0.5 | 811,973,592.57 | yes |
| **candidate_1 (chosen)** | **n5_y2030_p1_e4** | **811,825,206.34** | yes |

The runner-up is pilot_staged, +148,386.23 € above the chosen start.

**Search.**
- **Start:** n5_y2030_p1_e4. The incumbent history is the start only; **0 moves**.
- **Terminal poll:** **k = 10 of |N| = 10** classified, all from cache. 0 evaluations were requested and the
  search budget used is 0.
- **Partition:** 0 improver, 9 worse, 1 unresolved, 0 not_settled, 0 harness_error.
- **Termination:** `complete_poll_no_accepted_candidate`. The phase ends `phase_complete`.
- The terminal poll is identical, row for row, to candidate 1's poll above: same incumbent, same 10 records.

**Indistinguishable plans (plan v4.1 §3–4).** No improvement beyond the applicable comparison criterion was
identified among the evaluated neighbouring plans. Every evaluated plan within R = 10,000 € of the final incumbent is
reported as indistinguishable from it. The search covered all 52 distinct plans with a record.

| plan | depth compared | dF vs n5_y2030_p1_e4 € | detail |
|---|---|---|---|
| n5_y2030_p1.25_e4 (in N, zP+1) | settled (K = 300) | +2,668.08 | V_pair 1,386.16, A 1,295.72; fails only \|dF\| ≥ R |
| n9_y2030_p1_e4 (not in N: other node) | exit only (never settled) | +9,940.16 (exit vs exit: 811,840,298.88 vs 811,830,358.72) | e = 17,494.05 → unresolved at exit |

**Zero-solve neighbourhood check of the final incumbent** (`p515_s56_neighbourhood.s56_neighbourhood`,
`p516_lattice.GeneralLattice.neighbourhood`, `raw_added_moves`; `report_eval.json` → `neighbourhood_final_incumbent`).
- **The incumbent:** 1.0 MVA / 4.0 MWh at node 5, invested 2030.
  - **I(x) = 944,672.09 € against B = 1,000,000 €. Slack: 55,327.91 € (5.5 %).**
  - Cumulative energy at node 5 is 4.0 MWh in 2030 and 2035, against a cap of 5.0 MWh.
  - The duration is 4 h, the upper edge of the 2–4 h window.
- **Cost of the smallest increase, 2030 (discounted, expected):**
  - zP+1: 47,596.02 €;
  - zE+1: 94,286.00 €;
  - activating 0.25 MVA / 0.5 MWh: 141,882.02 € in 2030, 118,945.62 € in 2035 (the cheapest activation of any
    year), 191,018.17 € in 2025.
- **Generated:** 63 raw moves. That is R3 22 (a 4, b 16, c 2; bound 2|N||Y| + 5A = 23), removal 1, same-sign
  diagonals 4, redistribution pairs 32 (17 triggers), and 4 opposite-sign diagonals (informational, not in N).
- **N:** 10 admissible moves in N, 2 admissible but excluded by rule, 51 refused.

| move class | in N | refused: budget only | refused: budget + duration | refused: duration only |
|---|---|---|---|---|
| R3 (a) coordinate steps | zP+1 → 1.25/4 (I 992,268; slack left 7,732); zE−1 → 1/3.5 | — | zE+1 → 1/4.5 (4.5 h; I 1,038,958) | zP−1 → 0.75/4 (5.33 h) |
| R3 (b) activations (8 cohorts × 2 sizes) | — | **all 16** (I 1,063,618–1,262,629) | — | — |
| R3 (c) shifts | 2030 → 2035 | 2030 → 2025 (I 1,271,828) | — | — |
| (d) removal | → x0 | — | — | — |
| (e) same-sign diagonals | −1−2 → 0.75/3 | +1+1 → 1.25/4.5 (I 1,086,554); +1+2 → 1.25/5 (I 1,180,840) | — | −1−1 → 0.75/3.5 (4.67 h) |
| (f) redistribution (decrease here + increase elsewhere) | 5: zE−1 here + a 0.25/0.5 unit at n5 2035, n7 2030, n7 2035, n9 2030, n9 2035 (I 969,332–992,268) | 11 (zE−1 + a 1 MWh unit, or + a 0.5 MWh unit in 2025; I 1,041,404–1,168,343) | 16 (zP−1 here, 5.33 h, + any unit) | — |
| e_opp (informational; excluded by Expert ruling 6, same-sign only) | not in N: 1.25/3.5 and 1.25/3 are lattice-admissible | — | −1+2 → 0.75/5 | −1+1 → 0.75/4.5 |

The refused moves by binding limit:

| limit | refused moves |
|---|---|
| investment budget | budget only 30; budget + duration 18 |
| C5 duration window (2–4 h) | duration only 3; plus the 18 above |
| cumulative capacity, 5 MWh per node | 0 (1.25/5 at node 5 reaches exactly 5.0 MWh and is refused by the budget) |
| sign / P = 0 ⇔ E = 0 | 0 |
| completion cap 80 | not binding (\|N\| = 10) |

**What this shows.** Observation only; the reading is the Planner's.
- **The final incumbent is bounded by the budget.**
  - Every move that adds energy at node 5 is refused, as is every move that adds a second unit anywhere: the
    16 activations, both upward diagonals, the 2025 shift, and 11 of the redistribution pairs. Each fails on the
    budget alone, except zE+1, which also fails on duration.
  - **The only size increase left in N** is zP+1 (1.25 MVA / 4 MWh), which shortens the duration to 3.2 h at the
    same energy. It is indistinguishable at settled depth (+2,668 €).
- **Neither the cumulative 5 MWh cap nor the completion cap is binding.**
- The ladder itself already lists 1 MVA / 4 MWh in 2025, at all three nodes, as inadmissible
  (`inadmissible_possible_points`). The screen's largest size is admissible only from 2030.

### 5. Recorded predictions Q1–Q7 against outcomes

Phase-wide: v2 and v3 records, under the corrected guard.

| # | prediction | outcome | held |
|---|---|---|---|
| Q1 | every screen plan has F_exit < F_exit(x0) (recorded, non-stopping, R1) | 18/18 certified; **12/18 below 0**; 7/18 below −R. The 6 failing rows are exactly the 0.5 MWh units in 2025 and 2035 at nodes 5, 7 and 9, at dF_exit +10,047 to +14,701 €, all inside e_EV (unresolved) | **FAILED** (recorded; non-stopping by R1: inside the exit resolution, no decision reads it) |
| Q2 | 0 harness_error, 0 local-solve failures | 0 harness_error summaries (52 launched evaluations plus every cache record read); 0 local-solve failures over the 52 | held |
| Q3 | best start ≠ x0 | n5_y2030_p1_e4 | held |
| Q4 | terminates (exit 0) inside 120 h | v3 ended `phase_complete` at 25.41 h in total. v2 itself stopped at exit 5 (review, routed to the Planner as Q4's on_failure prescribes; resolved by the v3 continuation). PROJECTED for this case: 65.2 h | held (in the continuation) |
| Q5 | peak child RSS ≤ 3.29 GiB | 2.81 GiB (v3 launch 1, n5_y2035_p0.25_e1 + unit, settled) | held |
| Q6 | efficiency fraction recorded | 22/22 not worse (16 confirm-candidate, 6 unresolved) → settled-direct. As run in v2: 21/22 (1 uncertified), the same branch. The v1 band measure, information only: 12/22 = 6/11 → also settled-direct under the re-map (v2 as run: 11/22 → exit-then-confirm) | recorded |
| Q7 | exit run = exit view, bitwise, per candidate | equal on both. **Independent on candidate 1 only** (§3); candidate 2's exit record is the exit view itself | held (1 independent check) |

### 6. Leak, slack and memory

The per-evaluation rows are in `report_eval.json` → `evaluations`.
- **Leak detector, loss-priced: 0.00 € on all 31 settled evaluations** (full-L priced: 0.00 €; L = 0 MWh); the
  ceiling is 25 €. The exit summaries carry no leak field.
- **ESSO slack**, `V_parts.G.esso_slack`, per settled evaluation: 6.6e-11 to 1.14e-9 €; the total over the 31 is
  1.43e-8 €. `esso_slack_max_abs_net` ≤ 1.29e-15. The ESSO slack gate passes on all 31.
- **Gates:** all pass on all 31 settled evaluations: d1, ESSO slack, the interface P ceiling, network closure, and
  status and guards exact.
- **Peak child RSS:**
  - exit: 2.69–2.78 GiB (n = 21);
  - settled: 2.49–2.81 GiB (n = 31);
  - peak: 2.81 GiB, against the 3.29 GiB slot.
- **ESSO recovery events: 6 events on 4 plans, all in v2; v3's 11 evaluations had none.**
  - Every event was recovered at tier 1, and tier 2 was never attempted.
  - **Every event is on a node holding a 0.5 MWh cohort.** 23 of the 52 evaluations hold a 0.5 MWh cohort; 4 of
    them had events. No evaluation without one had an event.

| plan | depth (launch) | events | node; cycle; primary failure | the node's cohorts |
|---|---|---|---|---|
| n9_y2025_p0.25_e0.5 | exit (v2 3) | 1 | 9; 124; infeasible | 2025: 0.25/0.5 |
| n5_y2030_p1_e3.5 + n5_y2035_p0.25_e0.5 | settled (v2 7) | 3 | 5; 83 and 224 internalSolverError; 245 infeasible | 2030: 1/3.5, 2035: 0.25/0.5 |
| n7_y2025_p0.25_e0.5 | settled (v2 9) | 1 | 7; 167; infeasible | 2025: 0.25/0.5 |
| n5_y2025_p0.25_e0.5 + n7_y2025_p0.25_e1 | settled (v2 10) | 1 | 5; 73; infeasible | 2025: 0.25/0.5 |

### 7. Projections for the next steps (PROJECTED from measured walls and the cache state)

The measured inputs are 5 exit launches (walls 0.97–1.39 h) and 8 settled launches (walls 2.01–2.72 h). Each range
below uses the minimum and maximum measured wall per launch.

**(a) The 2× controls (complete-poll; reversed class order).**
- **Would they end immediately? Yes, in settled-direct mode.**
  - The start rule gives the same start, n5_y2030_p1_e4.
  - All 10 of its neighbours are cached at the settled key, from candidate 1's poll. A settled-direct control
    therefore requests 0 evaluations.
  - It classifies the 10 exactly as the search did (9 worse, 1 unresolved) and ends at its first complete poll with
    0 moves.
  - This holds for both controls: with no improver, neither "complete poll, then the best" nor a reversed order
    can move.
  - **PROJECTED: 0 h of machine time.**
- **As the code stands (scoped):**
  - **No complete-poll control mode exists.** I searched `p515_s56_search_policy.py` and
    `p515_s56_search_driver.py` for complete-poll, `complete_poll` and best-improvement. Only `poll_only` (the
    candidate polls) and the reversed class rank exist.
  - **The reversed control** is available only as the plain search (`--freeze --reversed-class-rank --cache-root …
    --start best_of_pool`). That path:
    - polls at exit depth with e_floor = 15,000 € (`E_FLOOR_EUR`, not e_EV);
    - chooses its start by argmin **exit** F over the pool;
    - requires a cached exit record for every pool member. x0, the unit and the staged plan have none: their exit
      values were pilot exit views, which only the phase serves.
  - Run that way, the first control needs 3 exit launches (the 10 neighbours, plus the exit-less pool members):
    **PROJECTED 2.9–4.2 h**. A second control on the same cache would need 0 h.
  - Its confirmations are all cached. No neighbour has dF_set ≤ −R, so none can be accepted. The only way its end
    could differ is a `stop_for_review_offset`, if an exit value lands more than e from its settled value.
- **Cache root:** v3 must be declared as a cache root, together with pilot and v2 (§8).

**(b) The 1× phase: screen 18 + ladder 4, 2 candidates, one search.**
- **No record carries over:** the price level is in the eval key (§8). x0's exit and settled records, and the
  unit's and staged plan's settled records, are re-run by the phase itself, as at 2× without a pilot cache.
- Components:
  - x0 exit: 1 exit launch, 0.97–1.39 h;
  - screen: 18 plans in 4 exit launches, 3.9–5.6 h;
  - ladder: 1 exit launch, 0.97–1.39 h;
  - candidate confirmation (x0 + 2 candidates): 1 settled launch, 2.0–2.7 h;
  - pool (unit + staged): 1 settled launch, 2.0–2.7 h.

  Base: **9.9–13.8 h**.
- **Candidate polls**, at the 2× shape (|N| = 10 and 22, so 7 blocks):

| case (no search move) | settled-direct branch | exit-then-confirm branch |
|---|---|---|
| two candidates, polls as at 2× | **24.0–32.8 h** | **16.7–23.5 h** + 2.0–2.7 h per confirmation in the final poll |
| no confirm-candidate in the screen (no candidates); start x0 | **15.9–22.0 h** (possible if every not-worse row is unresolved): 7.8–11.1 h + x0's poll at settled depth, 18 plans in 4 settled launches (4 × 2.01–2.72 h) | **7.8–11.1 h** |

- **If the start is x0:** N(x0) is exactly the 18 screen plans, checked at zero solves. They are cached at exit
  depth, so the final poll at x0 costs 0 h beyond any confirmations.
- **If the start is the unit or the staged plan:** their N is mostly uncached (22 and 46 at 2×).
- **Each search move** adds one poll. At the plan's median |N| of 39, that is about 8 blocks: about 16–22 h
  settled-direct, or 7.8–11.1 h exit-then-confirm plus confirmations.
- **The efficiency branch at 1× is not predicted here.** Both branches are given as ranges, as instructed.

### 8. Evidence

**Collision check (zero solves, part G of the script; 13 checks, all pass).** The question: can tracking v3's three
campaign specs block a later 1× or control freeze?
- **The key inputs are confirmed by recomputation:**
  - the cap enters the eval key: x0 settled at cap 300 gives `142a02890a1d630f`, at cap 400 `85c5235276a12a46`;
  - the price level enters it: the 74,084 domain keys at 2× and the 74,084 at 1× are disjoint.
- **The keys in v3's campaign specs are a subset of those in v3's launch records** (11 and 11). Their 2× domain
  keys are exactly the 11 v3 evaluations. They carry no 1× key.
- **Simulated freeze scans**, over the tracked artefacts plus v3's launch-dir artefacts:

| scan | collisions |
|---|---|
| a 1× phase freeze | 0 (360 files) |
| a 2× control declaring pilot, v2 and v3 as cache roots | 0 |
| a 2× control declaring only pilot and v2, with the v3 specs tracked | 7 |
| the same, without the v3 specs | 4 |

- **The 4 collisions without the specs** are v3's 3 launch records and its **already tracked** phase spec
  `phase_spec_600bbceb.json`, which carries v2's and the pilot's keys in its cache index. A 2× freeze must declare
  v3 as a cache root in any case. Tracking the campaign specs adds no blocking condition.
- **Decision: v3's campaign specs are tracked.**

**What is committed.** By the S56 selection rule (`p515_s56_ev_harness_tests.committed_by_rule`):
- excluded directories: `/planning_read/`, `/results/`, `/esso_capture/`, `/scratch`;
- excluded suffixes: `.pkl`, `.xlsx`, `.pdf`, `.png`, `.nl`;
- size limit: ≤ 1,300,000 bytes.

| manifest | files | committed | sha256 |
|---|---|---|---|
| `phase2x_ev_end_evidence_manifest_sha256.json` (both roots, every file; plus `phase2x_ev_end_report_eval/` and its logs) | 41,203 (17.39 GB) | 1,672 (407.2 MB) | `4bc3505a00a7…` |

- **Committed:**
  - both roots' freeze and launch logs, `phase_state.json`, `phase_results.json` and their manifests;
  - every launch record, launch manifest, campaign spec, heartbeat and init identity;
  - per eval dir, every mapping input: `evaluation_record.json`, `per_cycle_record.jsonl`,
    `convergence_depth_tail_state.json`, `s56_cost_breakdown.json`, `s56_pricing_terminal.json` (the settled runs),
    `exit_code.txt`, `esso_recovery_events_s39_D.jsonl`, `network_failures_s39_D.jsonl` (52/52 each);
  - the other sidecars ≤ 1.3 MB.
- **Hash-only:**
  - `pf_entry_stride_s39_D.jsonl` (up to 185 MB each);
  - `ess_entry_stride_baseline.jsonl`, `interface_duals_per_cycle.jsonl`, `network_ipopt_solve_records*.jsonl`,
    `g_s39_D.json`;
  - 32 of 52 `child_stdout.log` / `stdout_s39_D.log` (over 1.3 MB);
  - the ESSO pickles, `results/`, `esso_capture/` and `planning_read/`.
- 36 of the committed files were already tracked: v2's 30 launch artefacts, both phase specs, and freeze logs.

## Not confirmed

- **The 1× efficiency branch and candidate count.** Both are unknown before the 1× screen, so §7b gives ranges, not a
  prediction. The 1× projection assumes the 2× poll shape (|N| 10 and 22) and the 2× walls. Walls at 1× have not
  been measured.
- **The plain-search control path on real records.** Its refusal without pool exit records is read from
  `best_start` (`p515_s56_search_driver.py`, "start pool members without a cached exit record"), not exercised. Its
  exit classification of the 10 neighbours, and hence whether an offset review stop could occur, is unknown without
  exit runs.
- **"No complete-poll control mode exists"** is scoped to the policy and driver modules named in §7a, not to other
  branches or worktrees.
- **The ESSO recovery IPOPT logs** (`primary_log` / `recovery_log`, 12 files of 6–10 kB) live under
  `data/SRP1/Results/P56A/evals/p515s44_s56phase2x_ev_v2_l0NN_…_run/logs/`, outside both phase roots. They exist,
  but they are not in this manifest and are not hash-recorded.
- **The 1× key set** was computed under one assumption: the 1× configuration is v3's with `flex_price_multiplier`
  1.0. Any further difference only changes the keys further. No 1× spec exists to check against.
- **One timestamp differs from TASKS.** `TASKS.md` 5b gives v2's launch 1 as "from 21:4x UTC 11-24". The launch
  record says 20:35:27–21:41:05; 21:46 is launch 2's start.
