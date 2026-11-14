# P5.15 Step 3 — Row 3′ economic baseline (s31c)

**Planner report, 2026-09-15.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 12. Production commit `2783c223`
(interface energy settlement at π_t, signed TSO interface δ_P/δ_Q, no Q settlement); harness arm `ffd01b39`. Campaign
`s31c` at C\* (s = 0.96875 MVA, e = 3.875 MWh, year 2025, uniform across nodes 5/7/9), production defaults, cold, one
background run, stderr captured, exclusive lock, heartbeat. **Python exit code 0.** **Stops for review before 3.2.**

Objective convention in every table: `gross_operational_cost` = system cost **with settlements excluded by
construction**; terminal salvage credit is 0 here, so gross = net.

## 1. Lead figures

### Cancellation residual

| quantity (block-weighted, terminal cycle 90) | value |
|---|---|
| T_TSO | −1,157,892,562.08 |
| T_DSO node 5 / 7 / 9 | +422,538,403.39 / +341,025,637.45 / +394,261,127.86 |
| **T_TSO + ΣT_DSO** | **−67,393.39** |
| priced consensus residual Σ π·baseMVA·(p_TSO − p_DSO) | +67,393.39 |
| closure (sum of the two) | 2.1e-7 (floating-point) |
| relative to settlement magnitude Σ\|T\| = 2.316e9 | 2.9e-5 |
| relative to system cost | 1.0e-4 |
| relative to terminal objective step (129,617) | 0.52 |

The settlement vanishes to the consensus residual exactly, as expected. The largest per-period interface mismatch
|p_TSO − p_DSO| is 0.37 / 0.15 / 0.25 MW; the mean is 0.007 / 0.003 / 0.004 MW (288 entries per DSO).

### Cycles

**90 — the cycle cap. The run did not converge.**

| | Step 3.0 (G1B/s30) | s31 (row 3 removed) | **s31c (row 3′)** |
|---|---|---|---|
| converged | 71 cycles | no (cap 90) | **no (cap 90)** |
| rule ten (terminal step / tolerance) | 0.847 | 30.26 | **1.96** (129,617 / 66,153) |
| terminal-step trend | settled | rising 1.3–2.3 M/cycle | **falling** |
| recourse | 817,520,272.93 | 517,777,950.83 | **661,400,927.74** |
| network failures | 22, all tier 1 | 69 (68 tier 1, 1 tier 2) | **9, all tier 1** (all `maxIterations`), 0 unrecovered |
| local-solve failures | 0 | 0 | 0 |
| solves | 3,694 | 4,711 | 4,650 = 51×91 + 9 recovery retries |

The three recourse values are not like-for-like. Step 3.0 includes the row 3 charge, and s31/s31c are not fixed points.
No difference between them is reported as a result (see §3).

## 2. Convergence detail (from `g_baseline.json` `cycle_trajectory`)

- **Consensus residuals are met; only objective stationarity fails.** The residual criterion first holds at cycle 18
  and holds in 72 of 90 cycles, including every cycle from 80 to 90. Terminal PF primal ratio is 0.186; dual PF mean
  ratio is 0.048. ρ is held on all three channels at the end.
- **Recourse is still falling at the cap,** monotonically in each of the last 40 cycles:

  | cycle | 18 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |
  |---|---|---|---|---|---|---|---|---|
  | recourse (M) | 769.94 | 698.49 | 683.68 | 674.93 | 669.64 | 666.03 | 663.41 | 661.40 |

  The decrease per 10 cycles shrinks: 8.75 → 5.29 → 3.60 → 2.62 → 2.01 M. The last six steps are 236k, 205k, 214k,
  211k, 180k, 130k (mean ≈ 1.9e5, about 2.9× tolerance).
- **Reading.** The run is being *stopped*, not settled. Rule ten is 1.96 on the final step and about 2.9 on the recent
  mean. The limit lies below 661.4 M by an amount these artifacts do not determine; extrapolating the decay is not a
  result.
- ESSO: all 273 solves end at lg(mu) = −11.0. Maximum measured spurious throughput is 3.35e-3. EFC/day max is 0.0285
  against a threshold of 1.4612. The shared FrozenSMOPF was not modified.

## 3. System-cost recourse at cycle 90

| component (block-weighted) | value |
|---|---|
| generation cost (E) | 472,648,988.42 |
| DSO-internal flexibility cost (E) | 188,764,740.79 |
| load curtailment / RES curtailment / ESS usage | 0 / 0 / 0 |
| **economic recourse, all D excluded** | **661,413,729.22** |
| D rows total (voltage −11,796.94; flex-P day balance −946.48; shared-ESS day balance −58.06; node/branch/local ESS 0) | −12,801.48 (bound-relaxation floor) |
| **`gross_operational_cost` (system cost, settlements excluded)** | **661,400,927.74** |
| including settlement (T_TSO + ΣT_DSO) | 661,333,534.34 |
| ESSO feasibility violation (D3) | −1.7e-5 |
| row 3 definitional (TSO interface flex charge) | 0 — no legs remain |

**Reading rule** (Step 3.0 − s31c = definitional + path). The raw difference is 817.52 − 661.40 = 156.12 M. The
rule-nine bar is not valid because s31c did not settle, and the difference mixes the removed row-3 charge with path.
**Whether Q(x) moved is therefore indeterminate as a limit statement.**

## 4. Per DSO: settlement and flexibility volumes

δ = p_int − anchor, where the anchor is the DSO standalone-initialization exchange. δ is used for reporting only; the
optimization has no anchor. Σ|δ| is the unweighted sum over the 288 year/day/period entries.

| DSO node | settlement Σ π·p_int (weighted) | Σ\|δ_P\| MW | max \|δ_P\| pu / rating | Σ\|δ_Q\| MVAr | max \|δ_Q\| MVAr | priced residual |
|---|---|---|---|---|---|---|
| 5 | +422,538,403 | 6,151 | 0.736 / 2.0 (37 %) | 122.6 | 1.83 | 42,186 |
| 7 | +341,025,637 | 4,251 | 0.723 / 1.0 (72 %) | 197.6 | 3.12 | 3,728 |
| 9 | +394,261,128 | 6,671 | 0.835 / 1.5 (56 %) | 265.3 | 10.93 | 21,480 |

All three DSOs are net importers: positive settlement means the DSO pays and the TSO receives. δ_P is a
**redistribution across periods, not a net displacement**. It is negative in 195 / 197 / 195 of the 288 entries and
positive in 93 / 91 / 93. The net sums are only −32.3 / −31.8 / −77.7 MW, against Σ|δ_P| of 6,151 / 4,251 / 6,671 MW.
DSO-internal flexibility cost is 28.5 % of system cost.

## 5. Hypotheses

- **Null-space hypothesis (s31 non-convergence caused by the unpriced up/down interface legs) — supported, not proven.**
  - In s31 the implied down-flexibility sat at the sum of ratings, 4.5 pu. In s31c no interface reaches its rating
    (maximum 72 %).
  - Rule ten fell from 30.26 to 1.96 and changed sign of trend. Network failures fell from 69 to 9.
  - Caveat: the settlement and the signed reparametrization entered together, so this run cannot attribute the effect
    between them. The comparison is before/after, not a controlled ablation.
- **Settlement correctness — confirmed at the terminal point.** Cancellation holds to 2e-7 against the priced residual.
  The residual itself is 1e-4 of system cost and below one terminal step.

## 6. For review before 3.2 (no action taken)

1. **Baseline status.** s31c is consensus-feasible but not objective-stationary at the 90-cycle cap. The author should
   decide whether it serves as Step 3's economic baseline as-is, with the drift above recorded, or whether a single
   re-run with a higher cap is authorized. A re-run would write to a new directory, never onto `P515S31C_run`.
2. **Input to 3.2 (stopping rule).** This run separates the two criteria cleanly: residuals satisfied for 72 cycles
   while the objective step stayed at 2–3× its relative tolerance (≈1e-4 of recourse), decaying slowly. That is direct
   evidence for how the rule should weigh them. 3.2 has not been started.
3. **Attribution (optional).** Separating settlement from reparametrization would need one ablation run. It is not
   required for the baseline and is not proposed unless the author wants it.

## Evidence

`data/SRP1/Results/P515S31C_run/` holds `g_baseline.json`, `component_levels_terminal.json`,
`interface_settlement_detail_s31c.json`, `s31c_evaluation.json` (from `p515_s31c_evaluate.py`, zero solves),
`network_failures_baseline.jsonl`, `leak_classification_baseline.jsonl`, the stdout log, the heartbeat and the ESSO
models pickle. Also: `data/SRP1/Results/P515S31C_launch.log` and `P515S31C_exit_code.txt`.
`evidence_manifest_sha256.json` covers 288 files (118.8 MB). `esso_capture/` (273 files) and `results/` are
hash-recorded, not committed.
