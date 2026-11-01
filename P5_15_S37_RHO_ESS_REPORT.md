# P5.15 Addendum 19 — the ρ_ess arms: neither certifies at cap 150; the binding channel is PF, not storage

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 19. Frozen spec:
`data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json` (predictions recorded before any run, 7323b2fb).
Launch HEAD for both arms: `eb2337f9`. Evaluations: `p515_s37_evaluate.py` (zero solves, guard armed), written to
`<run_dir>/s37_evaluation.json`. Matched-cycle comparison with run 1: `p515_s37_matched_cycles.py` →
`data/SRP1/Results/P515S37/matched_cycles_vs_run1.json` (zero solves, guard armed; formula in its docstring).
Instance: C\* = 0.96875 MVA / 3.875 MWh per active node, year 2025, uniform assignment (recorded in each evaluation).
Cost convention throughout: `gross_operational_cost`.

## 1. Verdict per arm

| arm | ρ_ess | cycles | stopped by | (a) Boyd stop | (b) no V/PF clamp | (c) storage ratio < 0.9 | (d) cost within rule-nine bar | **CERTIFIED** |
|---|---|---|---|---|---|---|---|---|
| `s37_rho0p01` | 0.01 | **150** | cap | fail | pass | fail (0.931) | fail† | **no** |
| `s37_rho0p001` | 0.001 | **150** | cap | fail | pass | pass (0.744) | fail† | **no** |

**Adoption (spec v8 rule): neither certifies → stop for review.** No ρ_ess is adopted.

† The rule-nine bar is valid only when both runs are settled; neither arm stopped under Boyd, so criterion (d) is
not interpretable as a cost statement (§5).

## 2. EFC/day (max across nodes) trajectory

| cycle | 1 | 2 | 5 | 10 | 20 | 30 | 50 | 75 | 100 | 125 | 150 | first cycle ≥ 1.06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run 1 (balanced ρ_ess) | — | — | — | 0.146 | — | 0.403 | 0.532 | — | 0.635 | — | 0.722 | never (1.059 at 477) |
| ρ_ess = 0.01 | 0.0003 | 0.329 | 0.709 | 0.775 | 0.986 | 1.056 | 1.095 | 1.125 | 1.131 | 1.135 | **1.140** | **33** |
| ρ_ess = 0.001 | 0.0003 | 0.698 | 1.187 | 1.164 | 1.171 | 1.160 | 1.175 | 1.169 | 1.164 | 1.162 | **1.162** | **3** |

Both arms' values are consistent with run 1's certified one-sided statement EFC/day ≥ 1.059. They are transient values
of uncertified runs, not equilibrium values.

## 3. Why neither certifies: the PF dual residual, which ρ_ess does not touch

**Observation.** The PF dual-residual ratio (s/ε_dual) is the same in all three runs — run 1 and both arms — at every
matched cycle, to within 1.3 % (`matched_cycles_vs_run1.json`, `pf_dual_ratio_spread_across_runs`):

| cycle | 10 | 30 | 50 | 100 | 150 | 200 | 226 | 300 | 477 |
|---|---|---|---|---|---|---|---|---|---|
| run 1 | 164.2 | 24.97 | 11.19 | 4.911 | 2.728 | 1.255 | **0.996** | 0.517 | 0.090 |
| ρ_ess = 0.01 | 163.4 | 24.97 | 11.18 | 4.936 | 2.731 | — | — | — | — |
| ρ_ess = 0.001 | 162.6 | 25.08 | 11.20 | 4.973 | 2.731 | — | — | — | — |

- **Where each channel first passes (primal and dual together):**

  | run | V | PF | ESS | ESS passing cycles |
  |---|---|---|---|---|
  | run 1 | 45 | **226** | **475** | 3 of 477 |
  | ρ_ess = 0.01 | 45 | never within 150 | **95** | 13 of 150 |
  | ρ_ess = 0.001 | 46 | never within 150 | **87** | 26 of 150 |

- In all three runs ρ_pf is 0.198 from cycle 1. Balancing never acted on it, and it was frozen by the cycle-60
  backstop. γ_pf = τρ_pf with τ = 1.
- **Terminal ratios at cycle 150:**

  | arm | V primal / dual | PF primal / dual | ESS primal / dual |
  |---|---|---|---|
  | ρ_ess = 0.01 | 0.034 / 0.152 | 0.379 / **2.731** | 0.931 / 0.246 |
  | ρ_ess = 0.001 | 0.045 / 0.153 | 0.559 / **2.731** | 0.744 / 0.030 |

**Conclusion (supported).** Exempting ESS from balancing and lowering ρ_ess did what Addendum 19 intended on the
storage channel. ESS now passes about 5× earlier than in run 1 (cycle 87–95 against 475), and the storage dual residual
is no longer the late bottleneck. Run 1's storage dual ratio was 0.988 at its stop; the arms' storage dual ratios at 150
are 0.25 and 0.03. **The stop is now bound by the PF dual residual.** On this evidence that residual is insensitive to
ρ_ess over a 100× range (0.1125 balanced → 0.001).

**Projection (not observed).** If the PF dual trajectory stays ρ_ess-invariant beyond cycle 150, as it did through
150, its first pass is near cycle 226 in any ρ_ess arm. That makes a cap-150 certification unreachable under the
current PF configuration, whatever ρ_ess is. The runs do not observe cycles 151 onward, so this is not a result.

**Inference, for review.** Any lever aimed only at the storage channel is unlikely to change the cap-150 verdict,
because storage is not the binding channel. That includes Option 1, settlement active at initialization, which the
spec names as the parked fallback. The PF dual pace is the quantity to reason about next.

## 4. Predictions recorded in advance (spec v8)

| prediction | outcome | reading |
|---|---|---|
| storage step per cycle scales ~1/ρ_ess | 0.01 vs run 1: **6.5×** (predicted ~11×). 0.001 vs 0.01: **2.06×** (predicted ~10×) | **Not met as stated.** The cycles 2–20 RMS window saturates. At 0.001 the walk reaches EFC 1.19 by cycle 5 and then stops moving, so the window measures distance-to-go, not walk speed. At cycle 2 alone the RMS steps are 0.095 (0.01) and 0.220 (0.001). The direction is as predicted; the 1/ρ proportionality is not observed. |
| EFC/day ≥ 1.06 within ~50 cycles at 0.01 | 1.095 at cycle 50; first ≥ 1.06 at cycle 33 | **Met.** |
| at least one arm certifies (storage ratio < 0.9, cost within the rule-nine bar) | neither | **Falsified.** The binding channel was PF, not storage (§3). |

## 5. Cost

| run | cost at cycle 150 | terminal cost | terminal step | rule ten (step / tolerance) | Boyd-settled |
|---|---|---|---|---|---|
| run 1 | 651,337,600 | 651,039,166 (cycle 477) | 256.26 | 0.0039 | yes |
| ρ_ess = 0.01 | 651,016,541 | same (cycle 150) | 918.04 | 0.0141 | **no** |
| ρ_ess = 0.001 | 650,936,208 | same (cycle 150) | 1,307.28 | 0.0201 | **no** |

The arms end 22,625 and 102,958 below run 1's certified cost. The bars would be 1,174 and 1,564, but a rule-nine bar
bounds stopping slack only between settled runs. Neither arm is Boyd-settled, so these differences are **indeterminate,
not a result**. The objective has stopped moving (rule ten ≈ 0.01–0.02) while the PF channel has not converged. That is
the pattern rule ten warns about when it is read without the channel residuals. No claim of a lower-cost or different
point is made.

## 6. Monitoring (reported, not gated)

| | ρ_ess = 0.01 | ρ_ess = 0.001 |
|---|---|---|
| local-solve failures | 0 | 0 |
| network failures (all recovered) | 41: 38 first-attempt, 3 second-attempt | 53: 50 first-attempt, 3 second-attempt |
| solves (51 × cycles + 51 = 7,701, plus recovery re-solves) | 7,745 = 7,701 + 38 + 2·3 | 7,757 = 7,701 + 50 + 2·3 |
| oscillation flag (sign-change > 0.2 for 10 cycles, or cosine < 0 for 5) | **not raised** | **not raised** |
| sign-change fraction / cosine, sampled at cycles 5 · 10 · 20 · 50 · 150 | 0.25/0.70 · 0.16/0.94 · 0.11/0.94 · 0.14/0.99 · 0.03/1.00 | 0.38/0.70 · 0.23/0.75 · 0.20/0.94 · 0.12/0.98 · 0.10/0.99 |
| wall clock | 5,795 s | 6,300 s |

The harness's `identity_holds: false` reflects the recovery re-solves, as in `P5_15_G1_REPORT.md`; no solve is missing.
The 0.001 arm shows an early overshoot: EFC 1.187 at cycle 5, then 1.164 at cycle 10, with sign-change fractions of
0.38 and 0.23. It damped without tripping the flag. Its ESS primal ratio is not monotone late in the run, with spikes rather than a
settled decay. Sampled at cycles 95 · 100 · 110 · 120 · 125 · 130 · 140 · 150, it reads 1.12 · **4.66** · 1.13 · 0.84 ·
0.87 · **4.60** · 1.26 · 0.744. Criterion (c)'s pass at 0.744 is therefore a single terminal sample, not a settled
storage channel. At 0.01 the same samples stay within 0.92–1.38.

## 7. Launch record

The first launch of `s37_rho0p01` stopped before any solve with a correct refusal: the arm's working directory
`P56A/evals/p515s37_rho0p01_baseline` already existed. The two-cycle preflight had created it, and 27 committed
preflight artifacts cite it. Fix in `eb2337f9`: a one-line harness change giving the arm the working-directory id
`p515s37_rho0p01_arm`. The failed attempt's log and exit code are preserved as
`P515S37_RHO0P01_launch_attempt1_20260917T084117Z.log` and `P515S37_RHO0P01_exit_code_attempt1_20260917T084117Z.txt`.
The empty precheck directory it created was moved aside with a timestamp suffix; no evidence cites it. The arms ran one
at a time, attached, with stderr captured, and both exited with code 0.

## 8. Questions for review

1. **Cap versus PF pace.** Run 1's PF channel first passes at cycle 226, and on this evidence the PF dual trajectory
   does not depend on ρ_ess. Is a cap-150 certification the right bar for this configuration? Or is the next lever the
   PF channel? In all three runs ρ_pf was never adjusted by balancing and was frozen by the cycle-60 backstop.
2. **Adoption.** The spec's rule gives "neither certifies → stop". The storage-channel evidence favours the arms over
   run 1: ESS passes at 87–95 against 475, and EFC ≥ 1.06 at 33 or 3 against never. The two arms differ mainly in
   early overshoot (0.001) against slower approach (0.01).
3. **Step 3.6** (per-phase timing, then the persistent-worker decision) is next in sequence and has not been started.
   Two items are carried forward from its design note: the interim S = 3 / X = 70 % target needs confirmation, and both
   "cheap wins" (NL writer v2, label-free round trip) are already the Pyomo 6.9.5 defaults.

## Evidence

- Per-arm manifests: `data/SRP1/Results/P515S37_RHO0P01_run/evidence_manifest_sha256.json` and
  `data/SRP1/Results/P515S37_RHO0P001_run/evidence_manifest_sha256.json`.
- Both cover the run directory, launch log, exit code, evaluation and evaluator log, the spec, this report, the
  matched-cycle artifact and run 1's `g_baseline.json`.
- `esso_capture/` and `results/` are hash-recorded, not committed, as in earlier stages.
