# P5.15 Addendum 19 — handoff to the External Expert and the Author

**Planner report, 2026-09-17.** Self-contained; continues `P5_15_ADDENDUM17_EXPERT_REPORT.md`. The stage report with
full tables is `P5_15_S37_RHO_ESS_REPORT.md` (`321a158d`).

- **Instance:** candidate C\* (s = 0.96875 MVA, e = 3.875 MWh, invested 2025, uniform across nodes 5/7/9).
- **Configuration:** run 1's, with ESS exempt from residual balancing and ρ_ess fixed.
- **Cost convention:** `gross_operational_cost`.

## 1. Verdict

**Neither ρ_ess arm certifies. Both ran to the 150-cycle cap.** Under the adoption rule, no ρ_ess is adopted, and the
programme stops for review. Run 1 remains the certified evaluation.

| arm | cycles | Boyd stop | V/PF clamp | storage terminal ratio < 0.9 | cost within rule-nine bar | **certified** |
|---|---|---|---|---|---|---|
| ρ_ess = 0.01 | **150** (cap) | no | none | 0.931, fail | not interpretable (§5) | **no** |
| ρ_ess = 0.001 | **150** (cap) | no | none | 0.744, pass on one sample | not interpretable (§5) | **no** |

**EFC/day trajectory** (maximum across nodes; transients of uncertified runs, not equilibrium values):

| cycle | 2 | 5 | 10 | 30 | 50 | 100 | 150 | first cycle ≥ 1.06 |
|---|---|---|---|---|---|---|---|---|
| run 1 (balanced ρ_ess) | — | — | 0.146 | 0.403 | 0.532 | 0.635 | 0.722 | never (1.059 at stop, 477) |
| ρ_ess = 0.01 | 0.329 | 0.709 | 0.775 | 1.056 | 1.095 | 1.131 | **1.140** | **33** |
| ρ_ess = 0.001 | 0.698 | 1.187 | 1.164 | 1.160 | 1.175 | 1.164 | **1.162** | **3** |

Both arms are consistent with run 1's one-sided statement, EFC/day ≥ 1.059.

**Headline.** The change did what Addendum 19 intended on the storage channel, but the stop is now bound by a different
channel. The **PF dual residual** is unchanged by ρ_ess, and in run 1 it did not pass until cycle 226.

| item | status | commit |
|---|---|---|
| Frozen spec v8, predictions recorded before any run | done | `7323b2fb` |
| Per-channel balancing exemption (production), zero-solve checks | done | `dee8735f` |
| Step 3.6 per-phase timing design | design only | `2d9ebfc1` |
| Arms, evaluator, two-cycle preflight | done | `70c22e83` |
| Working-directory fix after the first launch was refused (§7) | done | `eb2337f9` |
| Both arms run, evaluated, matched against run 1, evidence committed | done | `321a158d` |

## 2. What Addendum 19 authorized

- **Runs:** two runs at cap 150, cold, no LP initialization, run 1's configuration with ESS balancing off, at
  ρ_ess = 0.01 and 0.001.
- **Predictions, recorded first:**
  - the storage step per cycle scales ~1/ρ_ess;
  - EFC/day reaches 1.06 within ~50 cycles at 0.01;
  - at least one arm certifies under Boyd, with storage ratio < 0.9 and cost within the rule-nine bar of run 1.
- **Monitoring:** local-solve failures and ESS step-direction sign changes.
- **Adoption:** if both certify, adopt the larger ρ_ess.
- **Parked:** Option 1.
- **Sequencing:** Step 3.6's preflight and gate come after the arms.

## 3. The storage channel: the intended effect was obtained

| | run 1 | ρ_ess = 0.01 | ρ_ess = 0.001 |
|---|---|---|---|
| first cycle the ESS channel passes (primal and dual) | 475 | **95** | **87** |
| ESS passing cycles | 3 of 477 | 13 of 150 | 26 of 150 |
| ESS dual ratio at stop / cap | 0.988 (cycle 477) | 0.246 | 0.030 |
| ESS primal ratio at cap | — | 0.931 | 0.744 |
| RMS storage consensus step, cycle 2 | — | 0.095 | 0.220 |

Two findings on storage:
- **The storage dual residual is no longer the late bottleneck.** In run 1 it was the last channel to pass.
- **Neither arm has a settled storage channel.** Every cycle of the trajectories was checked, not only the sampled
  ones.
  - **At 0.01** the ESS primal ratio stays between 0.92 and 1.62 from cycle 95 on. It passes on 13 of cycles 95–150 and
    reads 0.93 over the last two cycles.
  - **At 0.001** it has two excursions: cycles 98–108, peaking at **14.2 at cycle 102**, and cycles 130–136, peaking at
    4.6. It passes on 19 of cycles 95–150 and has decayed from 1.52 to 0.74 over the last ten cycles. The 0.744 at cycle
    150 is the tail of that decay, not a settled value.
  - **Co-occurrence, not established cause.** Recovered network (DSO) solve failures fall inside both windows, at cycles
    98, 100, 102, 103, 107, 130 and 135. Failures also occur at many cycles without an excursion.
- **Correction to the stage report.** `P5_15_S37_RHO_ESS_REPORT.md` §6 lists sampled cycles only, and so understates
  the 0.001 excursion (peak 14.2, not 4.66). It also states 0.92–1.38 for the 0.01 samples, where the full range is
  0.92–1.62. That report is hash-recorded and is not edited in place; this report supersedes it on these two figures.

## 4. The new binding channel: PF dual residual, invariant to ρ_ess

### 4.1 Observation

The PF dual ratio s/ε_dual is the same in run 1 and both arms at every matched cycle, to within 1.3 %:

| cycle | 1 | 10 | 30 | 50 | 100 | 150 | 200 | 226 | 300 | 477 |
|---|---|---|---|---|---|---|---|---|---|---|
| run 1 | 2780 | 164.2 | 24.97 | 11.19 | 4.911 | 2.728 | 1.255 | **0.996** | 0.517 | 0.090 |
| ρ_ess = 0.01 | 2780 | 163.4 | 24.97 | 11.18 | 4.936 | 2.731 | — | — | — | — |
| ρ_ess = 0.001 | 2781 | 162.6 | 25.08 | 11.20 | 4.973 | 2.731 | — | — | — | — |

ρ_ess ranged over about 100×, from run 1's balanced 0.1125 down to 0.001, while the storage schedules differed
strongly: EFC 0.146 against 1.164 at cycle 10.

- **The PF residual pair is the reverse of the storage pair.** At cycle 150 the PF primal ratio is 0.379 / 0.559, well
  inside tolerance, while the dual ratio is 2.73. Run 1 first passes PF at cycle 226.
- **ρ_pf was never adjusted.** ρ_pf = 0.198 from cycle 1 in all three runs; balancing never acted, and the channel was
  frozen by the cycle-60 backstop. γ_pf = τρ_pf with τ = 1.

### 4.2 Structure of the quantity (from the code)

For V and PF (`shared_resources_planning.py`, Boyd per-channel metrics):

  s = √(ρ² + γ²) · ‖z_TSO^k − z_TSO^(k−1)‖   (normalized by the interface rating)

- **Only the TSO's interface-PF copy moves s.** With ρ_pf fixed and equal across runs, s changes only through that
  movement. Its invariance means the TSO's PF interface copy moves by the same amount per cycle whatever the storage
  channel does.
- **The proximal term inflates s by exactly √2.** With γ = ρ, the proximal part equals the ρ part: the logged proximal
  share is 0.7071 at every sampled cycle. Without the proximal part, the PF dual ratio at cycle 150 would be 1.93, which
  still fails.
- **The balance-ratio gap passes the case-file dead band late in the run.**
  - Balancing uses s_ρ/ε_dual against r/ε_pri, with a PF-decrease dead band of 3.0 in the case file.
  - In the 0.01 arm this ratio is about 2.1 at cycle 30, 1.9 at cycle 60, 5.0 at cycle 100 and 5.1 at cycle 150.
  - The widening happens after the cycle-60 freeze. **Whether the balancing rule would have lowered ρ_pf is not
    established:** the rule was not replayed against these trajectories.

### 4.3 Hypotheses (for review, not tested)

- **HP1 — the pace is set by the PF block alone.** The fixed ρ_pf = γ_pf, the TSO proximal damping and the network
  curvature set the rate at which the TSO interface copy settles, and storage has negligible influence. The invariance is
  consistent with this.
- **HP2 — ρ_pf is too large for the late phase.** Primal well inside tolerance and dual 2.7× outside is the textbook
  signature. Freezing at the backstop removed the mechanism that would respond to it.
- **HP3 — the stopping-rule treatment of the proximal term costs cycles.** The √2 factor delays the PF pass. It is not
  sufficient alone: the ratio is 1.93 without it at cycle 150.

**Evidence gap.** The saved PF data are not enough to tell these apart. Per cycle, only aggregate PF norms are saved,
plus the worst-entry location and change. No per-entry PF trajectory exists among the arms' sidecars; the per-entry
sidecar covers storage only. A per-entry PF step capture, analogous to `ess_entry_stride`, would be needed to
decompose the PF dual by node, year, day and P/Q.

### 4.4 Projection, labelled as such

If the PF dual trajectory stays ρ_ess-invariant after cycle 150, as it did through 150, any ρ_ess arm would first pass PF
near cycle 226. That is well above the certification cap of 150. **Cycles 151 onward were not observed, so this is not
a result.** It implies that levers acting only on the storage channel, including the parked Option 1, are unlikely to
change a cap-150 verdict.

## 5. Cost

| run | terminal cost | terminal step | rule ten | Boyd-settled |
|---|---|---|---|---|
| run 1 | 651,039,166 (cycle 477) | 256.26 | 0.0039 | yes |
| ρ_ess = 0.01 | 651,016,541 (cycle 150) | 918.04 | 0.0141 | **no** |
| ρ_ess = 0.001 | 650,936,208 (cycle 150) | 1,307.28 | 0.0201 | **no** |

- **The differences are indeterminate, not a result.** The arms end 22,625 and 102,958 below run 1, but a rule-nine bar
  is valid only between settled runs.
- **Rule ten alone would have read the arms as settled.** Their objective step is 1–2 % of tolerance while the PF
  channel is 2.7× outside its threshold.
- **No claim is made of a lower-cost or different point.**

## 6. Predictions and monitoring

| prediction | outcome |
|---|---|
| storage step ~1/ρ_ess | **Not met as stated.** 0.01 against run 1 is 6.5× (predicted ~11×); 0.001 against 0.01 is 2.06× (predicted ~10×), over cycles 2–20. The direction is right, but the window saturates: at 0.001 the walk reaches EFC 1.19 by cycle 5, so the window measures distance-to-go, not walk speed. |
| EFC/day ≥ 1.06 within ~50 cycles at 0.01 | **Met** — cycle 33; 1.095 at cycle 50. |
| at least one arm certifies | **Falsified** — PF-bound (§4). |

| monitoring | ρ_ess = 0.01 | ρ_ess = 0.001 |
|---|---|---|
| local-solve failures | 0 | 0 |
| network failures (all recovered) | 41: 38 first attempt, 3 second | 53: 50 first attempt, 3 second |
| oscillation flag (sign-change > 0.2 for 10 cycles, or cosine < 0 for 5) | not raised | not raised |
| sign-change fraction / cosine at cycles 5 · 10 · 150 | 0.25/0.70 · 0.16/0.94 · 0.03/1.00 | 0.38/0.70 · 0.23/0.75 · 0.10/0.99 |
| solves; wall clock | 7,745; 5,795 s | 7,757; 6,300 s |

The 0.001 arm overshoots early: EFC 1.187 at cycle 5, then 1.164 at cycle 10. It damps without tripping the flag, but
its late storage primal ratio has excursions up to 14.2 (§3). The oscillation flag measures step-direction sign
changes; it does not measure residual excursions. Solve counts reconcile exactly: 51 × 150 + 51 = 7,701 solves, plus the
recovery re-solves.

## 7. Process notes

- **Launch refusal and fix.** The first launch of the 0.01 arm stopped before any solve with a correct refusal: its
  working directory already existed, created by the two-cycle preflight and cited by 27 committed preflight artifacts.
  The fix, in `eb2337f9`, is a one-line harness change giving the arm its own directory id. The failed attempt's log and
  exit code are preserved.
- **Run discipline.** The arms ran one at a time, attached, with stderr captured; both exited with code 0.
- **Zero-solve checks.** The evaluator and the matched-cycle comparison ran with `SolveProfileGuard` armed: zero solves.
- **Nothing started after the arms.** No production change and no Step 3.6 work followed.

## 8. Options (author decisions) and the Planner's recommendation

1. **Diagnose the PF pace before any further run.** This separates HP1–HP3 and costs no production change to the
   method.
   - (a) Zero-solve: replay the balancing rule on the saved trajectories, to see whether and when ρ_pf would have been
     lowered without the backstop.
   - (b) Add a per-entry PF step capture, the PF analogue of the storage sidecar, measurement only. Take a short
     capture run that reproduces the invariant cycles, e.g. cap 30 at ρ_ess = 0.01.
   - **Recommended as the next step.**
2. **A PF-lever arm at cap 150**, with ESS exempt at ρ_ess = 0.01. For example: allow PF balancing past the backstop,
   or a lower fixed ρ_pf. This changes the penalty policy and should follow option 1, so that the lever is chosen on
   evidence rather than tuned.
3. **Raise the certification cap** to cover the PF pace, e.g. 300, for one ESS-exempt arm. This re-runs from cold
   (about 3 h) rather than continuing the arms: whether the arms' saved state suffices to continue them has not been
   established. It does not address the pace and costs Step 5 runtime.
4. **Accept run 1 as the oracle and proceed** to Step 3.6 and Step 4. The storage finding is recorded as a method result
   for later use.
5. **Revisit the stopping rule's proximal term.** This is a method change to a rule the Expert set. It would not certify
   the arms alone (1.93 at cycle 150), and is listed only for completeness.

**Adoption.** If a later arm certifies, the storage evidence favours an ESS-exempt, fixed ρ_ess. The two values
tested differ in character: 0.001 approaches faster but overshoots early and has late residual excursions up to 14× threshold;
0.01 approaches more slowly and hovers within 0.92–1.62 of its threshold. Neither is adoptable today.

## 9. Questions for the Expert

1. Do you accept that the cap-150 verdict is set by the PF channel, and that ρ_ess has done its job on storage?
2. Which reading of the PF dual pace is most plausible: the PF block alone (HP1), ρ_pf too large late with balancing
   frozen at 60 (HP2), or the √2 proximal inflation in the stopping rule (HP3)? Do you endorse option 1's diagnostic
   before any PF lever?
3. Should the freeze backstop at cycle 60 stand for the PF channel? It was introduced to stop ρ oscillation, and it fixes
   ρ_pf before the late-phase imbalance appears.
4. Is cap 150 still the right certification bar, given that run 1's PF channel needed 226 cycles?
5. **Step 3.6** (per-phase timing, then the persistent-worker decision) is next in Addendum 19's sequence and has not
   started. Should it proceed now, in parallel with option 1? Please also confirm the interim S = 3 / X = 70 % target.
   Both "cheap wins" (NL writer v2, label-free `.sol` round trip) are already Pyomo 6.9.5 defaults, so neither will
   yield a gain.

## 10. Evidence index

| item | artifact | commit |
|---|---|---|
| prior handoff | `P5_15_ADDENDUM17_EXPERT_REPORT.md` | `085b0195` |
| frozen spec v8 | `data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json` | `7323b2fb` |
| balancing exemption; zero-solve checks | `admm_parameters.py`, `shared_resources_planning.py`, `p515_s37_zero_solve_checks.py`, `P515S37/zero_solve_checks/` | `dee8735f` |
| Step 3.6 timing design | `P5_15_STEP36_TIMING_DESIGN.md` | `2d9ebfc1` |
| arms, evaluator, preflight | `p515_g_g1_g4_admm_gates.py`, `p515_s37_evaluate.py`, `p515_s37_preflight.py`, `P515S37/preflight_rho0p01/`, `WORKER_REPORT_S37_PREP.md` | `70c22e83` |
| launch fix | `p515_g_g1_g4_admm_gates.py` | `eb2337f9` |
| stage report | `P5_15_S37_RHO_ESS_REPORT.md` | `321a158d` |
| arm runs and evaluations | `P515S37_RHO0P01_run/`, `P515S37_RHO0P001_run/` (each with `s37_evaluation.json` and `evidence_manifest_sha256.json`), launch logs, exit codes, attempt-1 record | `321a158d` |
| matched-cycle comparison with run 1 | `p515_s37_matched_cycles.py`, `P515S37/matched_cycles_vs_run1.json` | `321a158d` |
| run 1 reference | `P515S35_REF_run/g_baseline.json` (sha256 `8c72a156…`) | earlier |

**Notes for §3.**
- The full-range ESS primal figures (0.92–1.62; peak 14.2 at cycle 102; pass counts over cycles 95–150) are read from
  `boyd_ess_primal_ratio` and `boyd_ess_channel_pass` over every cycle of `g_s37_*.json`.
- The failure cycles are read from `network_failures_s37_rho0p001.jsonl`.
- These are direct field reads with no derived statistic, and all sources are hashed in the per-arm manifests.

**Notes for §4.2.**
- The proximal share, the balance ratios and the ρ_pf actions are read from the committed trajectories:
  `boyd_pf_proximal_share`, `boyd_pf_dual_ratio_balance`, `boyd_pf_primal_ratio`, `rho_pf_action` in `g_s37_*.json`.
- The PF-decrease dead band of 3.0 is read from `data/SRP1/SRP1_params.json`, hashed in both manifests.
- The formula for s is read from `shared_resources_planning.py` at `eb2337f9`.
