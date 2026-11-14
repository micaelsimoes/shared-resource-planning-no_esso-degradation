# P5.15 — Step 1 closing report (Addendum 8), opening Step 3

**Planner report, 2026-09-15.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 8. Continues
`P5_15_EXPERT_HANDOFF_6.md` and `P5_15_ADDENDUM7_REPORT.md`. Everything is committed; commits are listed in §9.

**Headline.** Hypothesis H-ε is **supported** against its frozen criterion: the ESSO throughput regularization at
ε = 1e-3 acted as a price on storage cycling at the ADMM fixed point. With ε = 1e-5, year-1 EFC/day returns to
within 2 % of the pre-reformulation value on every node and SoH reproduces the old trajectory. Production is now
**ε = 1e-5, ESSO tol = 1e-10, acceptable_tol = 1e-9**, with the explicit recovery policy and tier 2. The G1 re-run
under this baseline converges cleanly (71 cycles, zero local failures, EFC/day 1.103, leak 0.0029 % of
throughput), and so does the G3-full re-run (80 cycles, zero local failures). **Step 1 closes.**

---

## 1. Attribution of G1's operating-point change — complete

G1 (Step-1 reformulation, ε = 1e-3) moved two quantities away from the pre-reformulation control: recourse
(+1,497,334, 10.05× the bar) and storage cycling (year-1 EFC/day 1.112 → 0.972). Three single-change ablations,
each against criteria frozen before its run, attribute them.

| arm | change from G1 | year-1 EFC/day (n5 / n7 / n9) | EFC gap closed | recourse Δ vs G1 | frozen verdict |
|---|---|---|---|---|---|
| A | Candidate 4 reverted | 0.979 / 0.976 / 0.976 | 2.7–4.5 % | **−1,340,296** (≈90 % of the recourse gap) | fails |
| B | Candidate 1 re-wired | 0.974 / 0.975 / 0.974 | 0.8–2.2 % | +25,982 | fails |
| **C** | **ε 1e-3 → 1e-5** | **1.102 / 1.113 / 1.113** | **92.4 / 99.6 / 99.7 %** | −538,785 | **H-ε supported** |

**Conclusions.**

- **Storage cycling is attributed to ε.** Ablation C met both parts of its frozen criterion: EFC/day within 2 % of
  1.112 on all nodes (−0.96 %, +0.05 %, +0.04 %) and the maximum cycle detector at 2.580e-3, **99.8×** G1's, inside
  the predicted window [1.293e-3, 5.171e-3]. C's SoH (node 7: 0.8387 → 0.7289 → 0.6245) essentially reproduces the
  old control's (0.8387 → 0.7284 → 0.6248). The regularization, intended as a tie-breaker, was carried to the
  networks by the consensus dual and priced cycling. Addendum 3's displacement argument held only at fixed duals.
- **Recourse is mostly attributed to Candidate 4** (ablation A), which replaced the two-sided flexibility
  day-balance band with a penalized slacked equality.
- **Candidate 1 accounts for neither.**
- The old control's numbers therefore reflect (i) no ε price on cycling and (ii) the old flexibility band, as well as
  the ~1.39 % spurious throughput already recorded.

---

## 2. The new production baseline

| setting | before | now |
|---|---|---|
| `EPS_ESSO_THROUGHPUT` (`definitions.py`) | 1e-3 | **1e-5** |
| ESSO `tol` / `acceptable_tol` (`ESSO_TOL_OVERRIDES`) | 1e-8 / 1e-7 | **1e-10 / 1e-9** |
| recovery eligibility | implied by non-empty `recovery_options` | **explicit `recovery.enabled`, default true (all networks and the ESSO)** |
| tier-2 retry (cold + adaptive μ) | — | **on by default** |
| `hessian_approximation: limited-memory` in case files | present (dead) | **removed** |

The tolerance tightening compensates for ε: the barrier leak is `μ_barrier/(s_obj·ε)`, so a 100× smaller ε needs a
~100× smaller terminal μ. Verified before any campaign: ε = 1e-5 on all 864 charge-variable objective
coefficients, and a one-cycle pre-flight in which every ESSO solve converged at `lg(mu)` = −11.0 with the detector at
9.2e-6 – 9.4e-6 against an identity prediction of 1.0e-5.

---

## 3. G1 re-run — the new baseline

| | G1 (ε 1e-3, tol 1e-8) | **G1 re-run (new baseline)** | old control |
|---|---|---|---|
| converged | 72 | **71** | 67 |
| rule ten | 0.891 | **0.847** | 0.933 |
| local-solve failures | 6 cycles | **0** | — |
| network failures | 12 recovered, 6 not attempted | **22 recovered (tier 1)**, 0 tier 2, 0 unrecovered | — |
| year-1 EFC/day (n5 / n7 / n9) | 0.973 / 0.972 / 0.972 | **1.103 / 1.103 / 1.103** | 1.112 / 1.112 / 1.112 |
| SoH, node 5 (yr 1 → 3) | 0.8575 → 0.7466 → 0.6413 | **0.8400 → 0.7307 → 0.6273** | 0.8387 → 0.7284 → 0.6248 |
| recourse | 817,618,798.07 | **817,520,272.93** | 816,121,464.16 |
| terminal `lg(mu)` | −8.6 | **−11.0** (all 216 ESSO solves) | — |
| spurious throughput | 0.009 % | **0.0029 %** | ~1.39 % |

- Recourse vs G1: −98,525, **0.69× the bar — indeterminate.** Restoring cycling did not move recourse, consistent with
  §1 (recourse sits with Candidate 4, cycling with ε).
- Recourse vs ablation C: +440,259 (3.17× the bar); C differs in ESSO tolerance and recovery policy.
- Detector 5.68e-6 – 9.36e-6; 62,173 cohort-periods barrier-set, 35 indeterminate. Detector over the idle prediction
  reads 0.57–0.94 at `lg(mu)` = −11.0, below the ~1.0 seen at −8.6 — **recorded, not explained.**

---

## 4. G3-full re-run — 1.62 MVA / 3.24 MWh at node 7

| | G3-full r2 (ε 1e-3) | **G3-full re-run (new baseline)** |
|---|---|---|
| converged | 80 | **80** |
| rule ten | 0.867 | **0.863** |
| local-solve failures | 2 (1 unrecovered) | **0** |
| network failures | 16 recovered, 1 unrecovered, 1 not attempted | **15 recovered (tier 1)**; tier 2 not attempted |
| cycle-38 block (node 7 `case33_2` 2035 Autumn) | failed unrecovered at cycle 38 | **no failure event** |
| node 7 year-1 EFC/day | 1.055 | **1.340** (below the 1.4612 threshold) |
| node 7 SoH (yr 1 → 3) | 0.8464 → 0.7048 → 0.5963 | **0.8091 → 0.6730 → 0.5703** |
| recourse | 819,016,107.91 | **819,046,335.56** (+30,228, **0.21× the bar — indeterminate**) |
| terminal `lg(mu)` | −8.6 | **−11.0** (all 243 ESSO solves) |

**Tier 2 was not exercised inside a campaign.** Addendum 8 expected this run to exercise it at cycle 38; under the new
baseline that block never failed, so no retry of any tier was needed there. Tier 2 remains verified only on the saved
cycle-38 pre-solve block in isolation (path-dependent failure; cold + adaptive μ solves it in 66 iterations).

At initialization node 7's slack absorbs 19.3 % of |pnet| (106 of 288 periods), the end-of-life rule acting through
the slack as recorded; from cycle 1 onward it is at the bound-relaxation floor.

---

## 5. Also done under Addendum 8

- **Limited-memory entries removed** from `case9`, `case33_2`, `case33_3` and the ESSO parameters — behaviour-neutral
  (both retry paths already discarded them); each file verified to parse to the original minus that key.
- **Parser fix:** the "empty rows" in failure files were the frozen-snapshot inventory written into the same JSONL.
  Failure files now contain only classified `network_block` events; snapshots and ESSO recovery events go to sibling
  files. Validated with zero solves on ablation B's committed output. No previously reported count was affected.
- **Cycle-38 block closed** as path-dependent, recovered by tier 2.

---

## 6. Step 1 — closed

| item | final state |
|---|---|
| ESSO reformulation | in production; closed as a component (Addendum 7) |
| leak mechanism | barrier-set by the terminal barrier parameter; detector is the reported quantity |
| warm-start and solver policy | adopted (Step 1a) |
| recovery policy | explicit, default on, tier 2 |
| ε and ESSO tolerance | 1e-5 and 1e-10 / 1e-9 |
| Set-1 network changes | in production; attribution recorded (§1) |
| gates | G1 re-run, G3-full re-run, G4 (bitwise, earlier baseline), G5 passed; G2 converges with node 5 eligible |
| multimodality | DSO local optima ~0.3 % apart; `Q(x)` defined up to the deterministic path's optimum |
| evidence discipline | frozen criteria before every ablation; per-run sha256 manifests; campaigns attached, locked, isolated |

`REVISION_CONTEXT.md` carries a 2026-09-15 amendment at the head of its current section recording this closure.

---

## 7. What Step 3 inherits — open items

1. **Tier 2 inside a campaign** has never fired; it is verified only in isolation.
2. **Determinism of the new baseline** has not been re-established (G4 was bitwise on the ε = 1e-3 baseline).
3. **G2 under the new baseline** (`k = 10,000`, ε = 1e-5) has not been run; the converged G2 re-run used ε = 1e-3.
4. **Detector/prediction ratio of 0.57–0.94 at `lg(mu)` = −11.0** is unexplained.
5. **Every downstream result computed at ε = 1e-3 or earlier** (including G1, G2, G3-full r2, ablations A and B, and all
   pre-reformulation numbers) is superseded as a baseline; they remain valid only as the evidence for the attribution
   in §1.
6. The manuscript needs: the ε price effect (§1), the ~1.39 % spurious throughput in previously published trajectories,
   the homogeneous-fleet cohort approximation (H3), and the measured detector in place of the closed-form estimate.
7. Two `FrozenSMOPF` comparators lost in G1 remain lost (accepted).

## 8. What is NOT established

- Whether the new baseline is bitwise deterministic.
- Tier-2 behaviour inside a campaign.
- Why the detector reads below the idle prediction at `lg(mu)` = −11.0.
- The `k` effect on recourse at convergence under the new baseline.

## 9. Commits and evidence

| step | commit |
|---|---|
| expert handoff round 6 | `9f23b5cb` |
| frozen ablation C criteria | `dff42252` |
| limited-memory removal; ablation C arm | `94be62bd` |
| parser fix | `80a1e54c` |
| ablation C evidence and evaluation | `b8fb7cf2` |
| new production baseline (ε, tol) and checks | `92e3fafe` |
| G1 re-run evidence | `d8cf2dcc` |
| G3-full re-run evidence | committed immediately before this report |
| this report and the `REVISION_CONTEXT.md` amendment | this commit |

Evidence roots under `data/SRP1/Results/`, each with `evidence_manifest_sha256.json` (per-period captures
hash-recorded, not committed): `P515A/run_c/` (evaluation `P515A/ablation_c_evaluation.json`, spec
`P515A/frozen_ablation_c_spec_v1_410f8262.json`), `P515G1B/` (`g1b_evaluation.json`), `P515G3F_B/`
(`g3fb_evaluation.json`), `P515G1B_precheck/`, `P515G1B_preflight/`, `P515A/parser_fix_validation/`.
