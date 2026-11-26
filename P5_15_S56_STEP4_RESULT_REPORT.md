# P5.15 S56 validation step 4: main run result (2026-11-23)

**Spec:** `step4_main_v2/frozen_step4_main_spec_v2_7ca210fb.json` (v1 `3dfb2889` verbatim except the roots; predecessor
v1). **Evidence:** `data/SRP1/Results/P515S56/step4_main_v2/`. Manifest: `step4_main_v2_manifest_sha256.json`
(sha256 `59bf89da…`; 3,592 files, 208 committed). Evaluation: `step4_main_evaluation_20261123T163553.json` (`95a4c7d6…`).

**Objective convention (every table):** EUR of total discounted cost. Coupled arms (V-1, V-2): Q = settlement excluded
in full (pin excluded, salvage separate). Pinned arms: Q = contracted settlement excluded. C_central is step 3's gross
EUR/day; e_b = Q_b − w_b·C_central; D_b = w_b·C_central − Q_pin,b.

**Status:** both main campaigns ran to completion. All 8 runs exited 0 and certified before the cap. The continuation
was not launched: its condition needs an unsettled cap-300 run, and none was unsettled. **Must-hold M1 failed as frozen,
so the order stops here for review.**

## Decisions needed (Planner, and the Expert where a gate is re-read)

1. **M1 failed its frozen tolerance on every coupled run.** The diagnosis below shows floating-point cancellation, not
   wiring. The spec says an M1 failure is a harness or wiring fault and stops the order.
   - Options:
     - (a) Record M1 as "failed as frozen, identity holds to rounding" (the G3 precedent: not re-scoped) and carry on
       with step 4's other readings.
     - (b) Re-read M1 for the next stage, with a tolerance relative to the sum of absolute summands, declared in
       advance.
     - (c) Treat it as a wiring fault and investigate.
   - Recommendation: (a) now, and (b) for future specs. Changing a gate's scope is the Expert's call.
2. **P3 failed on 2025|Winter** (instance A): e_b(K) = −1,129.86 €, 4.9× the 228.64 € allowance; R_eff = 49,417 €.
   P3 held on 2035|Summer, but its terminal |e_b| sits inside its own settling bar (below). The spec says to report
   this before anything touches R; this report does that.
3. **The cycle-40 look tool had a defect** (described under Changed). The look was run through a separate parameter
   script, with no edit to the launcher. Should the launcher be fixed before the next use?

## Blocked on the author

Nothing. Push stays with the author.

## Changed

| commit | content |
|---|---|
| `4e6bb1d3` | v1 attempt preserved: `step4_main_v1/killed_attempt_v1_note.json`, sha256 manifest (159 files; 62 committed, PDFs and P56A working dirs hash-only). The stale campaign lock (PID 88707, dead) was **moved**, byte-identical, to `killed_attempt_v1_stale_lock.p515_s44_campaign.lock` |
| `cb062834` | spec v2 `7ca210fb` (derived by `p515_s56_step4_main_spec_v2_derive.py`; the key-path diff vs v1 is only version, predecessor, why_v2, roots, runs[*].campaign_root and the continuation root name) + launcher: `apply_spec_roots` (MAIN_DIR, ROOT_TAG and absolute roots from the spec), generalised spec-tag parse, `--vmstat --spec` |
| `765c2e5c` | `p515_s56_step4_main_look40_params.py`: the declared look with its numbers (see defect below) |
| `132524d8` | `p515_s56_step4_main_evaluate.py`: zero-solve evaluator of M1–M5 and P3–P9 per the spec's operational definitions; its readings are listed in its `INTERPRETATIONS` |
| (this commit) | evidence, manifest, this report |

**The look40 defect.** `--look40` reads `look['cycle']`, `look['fit_cycles']` and the other numbers from the spec's
`iv_cycle40_look`. That block is prose only; the numbers live in the launcher constant `LOOK40` and never reached the
spec. At B's look (15:29:44) it raised `KeyError: 'cycle'` with zero solves (`look40_B_20261123T152944.json`, error
recorded). The parameter script asserts that each `LOOK40` number equals the spec prose, then calls the unchanged
`look40`.

## Runs (measured)

| inst | arm | status | cycles | solves (= declared base) | wall: sum of cycle_wall_s | mean s/cycle | peak RSS child / IPOPT GiB | tail-depth cycles |
|---|---|---|---|---|---|---|---|---|
| B | V-1 | certified | 107 | 1,188 | 4,736 s | 44.3 | 3.53 / 0.35 | 99–107 |
| B | V-2 | certified | 84 | 935 | 3,772 s | 44.9 | 3.40 / 0.35 | 76–84 |
| B | pinned AA-off | certified | 97 | 1,078 | 4,123 s | 42.5 | 3.49 / 0.35 | 80–97 (12) |
| B | pinned AA-on | certified | 52 | 583 | 2,398 s | 46.1 | 3.14 / 0.35 | 41–52 (10) |
| A | V-1 | certified | 46 | 329 | 955 s | 20.8 | 2.51 / 0.34 | 38–46 |
| A | V-2 | certified | 40 | 287 | 857 s | 21.4 | 2.53 / 0.37 | 32–40 |
| A | pinned AA-off | certified | 44 | 315 | 900 s | 20.5 | 2.66 / 0.34 | 36–44 |
| A | pinned AA-on | certified | 36 | 259 | 748 s | 20.8 | 2.50 / 0.35 | 28–36 |

- **Campaign walls:** B 4,910 s (14:56:03–16:17:53 UTC); A 1,047 s (16:18:17–16:35:45).
- **Solves:** 4,974 in total, retries 0, local-solve failures 0, non-optimal blocks 0. Every launcher and evaluator
  guard counted 0.
- **Terminal-step-to-threshold** (rule ten, objective change / tolerance):
  - B: 0.0045, 0.0073, 0.0029, 0.039;
  - A: 0.0088, 0.020, 0.0041, 0.0027 (arms in the table's order).
- **Terminal Boyd ratio max:**
  - v channel: 0.15–0.90;
  - pf channel: 0.12–0.75.
- **vm_stat (free + inactive, GiB):**
  - B: 15.30 before, 10.39 at cycle 4, 10.33 at the look, 17.19 at 15:52;
  - A: 18.60 before, 16.53 at about cycle 15, 15.76 at the look.
  - A ran after B, as spec v1 states: one campaign lock per checkout. B's steady-state 10.39 GiB was also below the
    12 GiB rule.
- **Concurrent load:** the cohort follow-up Worker's tests (PID 98013, started 15:52:53; one child at 8.6 GB RSS)
  overlapped the last 25 minutes of B and all of A. Wall times from then on include that contention.

## Found: predictions against outcomes

| # | prediction | outcome |
|---|---|---|
| M1 | F1 identities to rel 1e-9 every cycle | **FAILED as frozen** on all 4 coupled runs (cycles failing: B V-1 22/107, B V-2 21/84, A V-1 9/46, A V-2 8/40). Worst rel: (i) 5.6e-9, (ii) 5.5e-8. Held: gauge Σωμ (worst 5.4e-12 × max\|μ\|), DSO decomposition (rel 0), the V-1 pin = 0.0 at every cycle and at the terminal record. Diagnosis below |
| M2 | init identity: pinned ≡ V-2 bitwise, V-1 its own group | **held** on A and B (hex equal to the smoke records; all comparisons equal) |
| M3 | first 3 cycles bitwise = serial smoke | **held** 8/8 (per_cycle_record 0 differing keys, none in one file only; duals, capture and solve records equal) |
| M4 | capture checklist; Σ per-block Q = production Q to 1e-12 | **held** 8/8 (worst rel 6.3e-16) |
| M5 | solve-count identity; exit 0 | **held** 8/8 |
| P3 | V-1 e_b(K) ≤ allowance on tail-depth cycles | 2035\|Summer **held** (110.13 ≤ 187.56, but see the bar); 2025\|Winter **failed** (−1,129.86; 4.9×) |
| P4 | D_b ≥ 0 on 2035\|Summer; \|D(2025\|Winter)\| ≤ \|D(2035\|Summer)\|/3 | sign **held** (+61,054 €); "much smaller" **failed**: 2025\|Winter is +193,439 €, 3.2× **larger** |
| P5 | lim Q_V2 − w(C + Δ_pin) ≤ S(V2) | **held** on both blocks (B +94.1 ≤ 621.9; A −969.2) |
| P6 | cycles V-1/off ∈ [1, 2]; wall ∈ [0.9, 1.2]; off/on ≥ 2 | **failed** on both. B: 0.70, 1.10, 2.02. A: 1.07, 1.01, 1.43 |
| P7 | full-channel PR cuts \|Q − wC\| fivefold after cycle 20 | **failed** on both: PR makes it worse (Σ\|Q+PR−wC\| / Σ\|Q−wC\| = 1.51 on B, 2.54 on A). The P-deviation part matches production to 2.9e-12 |
| P8 | tail-onset Q step recorded | recorded (table below) |
| P9 | peak RSS ≤ 6 GiB | **held** (max 3.53 GiB) |

**M1 diagnosis (measured, beside the frozen test, not deciding).**
- At convergence, T and D are small residues (O(1–10³) €) of sums whose absolute summands total 3.6e8 € (T) and
  2.1e6 € (D).
- The left side of (ii) is a difference of two gross costs of about 1.4e8 €.
- Absolute differences:
  - (i) ≤ 5.4e-7 €;
  - (ii) ≤ 7.5e-8 €.
- The same differences relative to the sum of absolute summands:
  - (i) ≤ 9.5e-16;
  - (ii) ≤ 7.2e-13.
- The (ii) difference relative to the gross scale is ≤ 5.3e-16.
- So the identities hold to double-precision rounding. The frozen measure rel = |a−b|/max(|a|,|b|,1) cannot be met
  once T and D cancel. The smoke passed (worst 9.3e-11) because at cycles 1–3, T and D were large.

## V-1 per-block table (calibration arm; tail-depth cycles)

| block | K | tail cycles | e_b(K) € | S_b (tail) € | allowance € | \|e_b\|/allow. | \|PR_b(K)\| € | D-1 floor (tail) € | R_eff € | dominant | working level |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2035\|Summer (B) | 107 | 99–107 (9) | **+110.13** | 439.28 (range; 8 steps, both signs) | 187.56 | 0.59 | 716.56 | 40.58 | 5,872 | \|PR\| | s* = 0.772 at k = 106 (e_b 145.19) |
| 2025\|Winter (A) | 46 | 38–46 (9) | **−1,129.86** | 339.11 (validated: r 0.817, H 28.0, Q̂∞ − wC = −818.8) | 228.64 | 4.94 | 1,368.61 | 45.67 | 49,417 | \|PR\| | not attained |
| 2025\|Summer (B, merged-only, not calibrated) | 107 | 99–107 | +340.96 | 332.26 (range) | — | — | — | — | — | — | — |

- **2035|Summer.** The bar (439 €) exceeds the allowance (188 €). P3's "held" is a reading at K, not a determinate
  "within allowance".
- **Working-level observables at k = 106 (2035|Summer):**
  - ΔP_s max 0.0027 MW, ω-RMS 0.00055;
  - ΔQ_s max 0.0029 MVAr, ω-RMS 0.00026;
  - ΔV_s max 0.071 kV, ω-RMS 0.0079;
  - A_b 734 €.
- **2025|Winter.** Both coupled arms end below w_b·C_central:
  - V-1 by 1,130 €;
  - V-2 by 785 €, although V-2 is a restriction of the central problem.
- PR has the same sign as e_b on A, so adding it widens the gap.

## D_b (pinned AA-off, last cycle; two-block figure)

| block | w_b·C_central | Q_pin(K) | D_b | bar S (range, last 10 cycles) |
|---|---|---|---|---|
| 2035\|Summer | 57,114,972.23 | 57,053,918.54 (K 97) | **+61,053.69** | 2,019.18 |
| 2025\|Winter | 79,710,848.83 | 79,517,409.71 (K 44) | **+193,439.12** | 2,769.09 |
| 2025\|Summer (merged-only) | 84,481,740.87 | 84,409,091.98 (K 97) | +72,648.88 | 2,095.80 |

Both calibrated values are positive and far outside their bars. Expert ruling 7: a negative value would not be a
defect, and these values are recorded, not stops.

## V-2 against V-1 (V-2 measured, non-deciding)

| inst / block | e_b(K) V-1 | e_b(K) V-2 | S tail V-1 / V-2 | terminal ΔV max (prod. pu) V-1 / V-2 | cycles V-1 / V-2 | Σ cycle wall V-1 / V-2 | effort to own limit ± allowance V-1 / V-2 |
|---|---|---|---|---|---|---|---|
| B 2035\|Summer | +110.13 | +1,917.70 | 439 / 622 | 4.36e-4 / 6.36e-4 | 107 / 84 | 4,736 / 3,772 s | 105 c, 4,663 s / 82 c, 3,693 s (ratios 1.28, 1.26) |
| A 2025\|Winter | −1,129.86 | −784.66 | 339 / 641 | 5.65e-4 / 4.50e-4 | 46 / 40 | 955 / 857 s | not attained (Q(K) 311 € from its limit) / 39 c, 837 s |

V-2's ΔV is the expectation mismatch only, since V-2 carries no V deviation kind. Its V-pin total runs 111–790 € on B
and 31–142 € on A.

## Pinned AA-off against AA-on

| inst | cycles off / on | at matched residual off / on | Q off − on (per block) | bars off / on |
|---|---|---|---|---|
| B | 97 / 52 | 97 / 48 (2.02) | 2035\|Summer −610.0; 2025\|Summer −229.0 | 2,019 / 6,488; 2,096 / 1,283 |
| A | 44 / 36 | 43 / 30 (1.43) | −290.1 | 2,769 / 2,776 |

The level differences are inside the bars, so they are indeterminate.

## P8: Q step at tail onset (K1)

- **B:**
  - V-1 K1 = 99: −861.68 (2025|S −219.20, 2035|S −642.48);
  - V-2 76: −1,106.33;
  - off 80: −4,022.06;
  - on 41: +6,135.30.
- **A:**
  - V-1 38: −516.10;
  - V-2 32: −1,270.99;
  - off 36: −2,138.97;
  - on 28: −2,776.15.

## Cycle-40 look (fitted and projected; zero solves)

- **B (2035|Summer):**
  - projection r = 0.840, k* = 59.0;
  - the held-out validation was poor: max |ln| error 2.89, because e_b rose from 3.5 k€ at cycle 34 to 15.8 k€ at
    cycle 39.
- **A (2025|Winter):** r = 0.921, k* = 63.3; held-out error 0.47.
- On both, the stop rule (k* > 600) did not fire.
- Measured afterwards: V-1 certified at 107 on B and 46 on A.

## Not confirmed

- **The kill signal of the v1 attempt** was not recorded. The attribution to the session exit rests on timing and on
  the empty stderr.
- **Whether M1's tolerance failure** could hide a real wiring offset smaller than about 1e-7 € was not tested. The
  differences are at rounding level relative to every scale checked.
- **P6 wall ratios** may be biased by the concurrent cohort tests. They were not re-measured on an idle machine.
- **No limit Q̂∞ is validated** on 2035|Summer for V-1 or V-2, nor on 2025|Winter for V-2. The bars there are ranges.
- **The 2025|Summer C_central** is merged-only (reported, not calibrated).
