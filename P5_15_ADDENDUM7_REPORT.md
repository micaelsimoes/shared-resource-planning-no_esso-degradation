# P5.15 — Addendum 7: ablation attribution, recovery policy, and G2 re-run

**Planner report, 2026-09-14.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 7. Continues
`P5_15_EXPERT_HANDOFF_5.md`. Everything is committed; commits are listed in §7.

**Headline.** (1) **Ablation attribution is unresolved within the two-run budget, but split.** Candidate 4
accounts for ~90 % of G1's recourse change and almost none of its storage-cycling change; Candidate 1 accounts
for neither. The drop in year-1 EFC/day (1.112 → 0.972) lies in another Step-1 change. (2) **G2 converges with
node 5 eligible**: 71 cycles, zero local failures, every network failure recovered on the first cold retry.

---

## 1. Ablation attribution (Addendum 7 item 2)

Criteria were frozen before either run: `data/SRP1/Results/P515A/frozen_ablation_spec_v1_2271c77f.json`
(committed `76ab3f99`). An arm recovers the old operating point only if it closes ≥ 90 % of the year-1 EFC/day
gap between G1 and the old control on all three nodes **and** lands within the rule-nine bar of the old recourse,
and only if it converged and settled (rule ten < 1). Both arms kept everything else as G1, including G1's
recovery behaviour (tier 2 off, node 5 ineligible), so each changed exactly one thing.

| arm | change from G1 | converged | rule ten | year-1 EFC/day (n5 / n7 / n9) | EFC gap closed | recourse | Δ vs old / bar | Δ vs G1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| old control (P5.14-N) | — | 67 | 0.933 | 1.1124 / 1.1122 / 1.1124 | — | 816,121,464.16 | — | — | reference |
| G1 | — | 72 | 0.891 | 0.9725 / 0.9722 / 0.9724 | 0 % | 817,618,798.07 | +1,497,334 / **10.05×** | — | reference |
| **A** | Candidate 4 reverted (flexibility day-balance band restored) | 69 | 0.799 | 0.9788 / 0.9760 / 0.9761 | **4.5 / 2.7 / 2.7 %** | 816,278,501.63 | +157,037 / **1.11×** | **−1,340,296** | **fails** |
| **B** | Candidate 1 re-wired (shared-ESS power-factor rows restored) | 71 | 0.846 | 0.9737 / 0.9753 / 0.9737 | **0.8 / 2.2 / 1.0 %** | 817,644,779.89 | +1,523,316 / **10.48×** | **+25,982** | **fails** |

**Per the frozen criterion, neither arm is the cause, and the two-run budget is exhausted.**

**What the two runs do establish.**

- **Candidate 4 accounts for ~89.5 % of the recourse change** (ablation A moved recourse by −1,340,296 of the
  −1,497,334 separating G1 from the old control), while leaving storage cycling and SoH where G1 had them.
  A falls short of the criterion on recourse by a small margin (1.11× the bar) and on EFC by a wide one.
- **Candidate 1 accounts for none of either**: B's recourse is +25,982 from G1 and its EFC closure ≤ 2.2 %.
- **The storage-cycling change is explained by neither.** Year-1 EFC/day in A and B stays within 0.007 of G1.
  It must lie in a Step-1 change not ablated: the reformulated ESSO (log-domain SoH, investments as parameters,
  deleted slack and complementarity families, and the leak reduction from ~1.39 % to ~0.009 %), the Step 1a solver
  and recovery policy, remedy (h), or Candidates 2 and 5. **Not established which.**

**Reading, not established.** Candidate 4 replaced a two-sided band `|Σ(p_up − p_down)| ≤ SMALL_TOLERANCE` with a
penalized slacked equality. That A moves recourse and not storage cycling is consistent with the change acting
through the flexibility cost the recourse contains rather than through the storage's dispatch. The old recourse
therefore carried a flexible-energy accounting that the new formulation prices differently; the Addendum-7 text
anticipated recording this only if A recovered both quantities, which it did not.

**Side effects observed in the ablations.**

- **A** (band restored): node 5 `case33_1` 2025 Autumn failed `maxIterations` in every cycle from 17 to 61 —
  47 local failures, never retried under G1-equivalent recovery. The run still converged.
- **B** (power-factor rows restored): 2 recovered and 1 unrecovered network failures, against 18 for G1 and 66 for A.
  B's unrecovered block is node 7 `case33_2` 2035 Autumn at **cycle 38** — the same block and cycle label as
  G3-full's unrecovered failure. Its pre-solve block is committed.

---

## 2. G2 re-run with node 5 eligible (Addendum 7 item 1)

G2's configuration (`k = 10,000` on C\*) under the new recovery policy (§3): every network and the ESSO
eligible, `case33_1` included, tier 2 on; ESSO slack values captured per period.

| | G2 (original) | **G2 re-run** |
|---|---|---|
| converged | **no** — 90-cycle cap | **yes — cycle 71** |
| local-solve failures | 31 cycles | **0** |
| network failures | 18 recovered, 33 not attempted | **23 recovered on tier 1**, 0 tier 2, 0 unrecovered, 0 not attempted |
| node 5 `case33_1` | failed every cycle 66–90, never retried | 9 failures, **all recovered on the cold retry** |
| recourse | none (last valid, cycle 65: 818,690,717.51, not settled) | **817,751,864.31** |
| rule ten | — | 0.868 |
| solves | 4,659 | 3,695 |
| SoH (node 7, yr 1 → 3) | 0.8412 → 0.7177 → 0.6069 (unconverged) | **0.8375 → 0.7144 → 0.6042** |
| year-1 EFC/day (n5 / n7 / n9) | — | 0.9720 / 0.9718 / 0.9725 |

The failure breakdown in the re-run is DSO node 5: 9, node 7: 6, node 9: 5, TSO: 3, all recovered on tier 1. The
2035 Autumn `case33_1` block that failed from cycle 66 to 90 in G2 did not fail at all: early recoveries at cycles
2–3 changed the path.

**Answer to the question left open by G2: yes — G2 converges once node 5's failures are eligible for recovery.**
Its non-convergence was the recovery-eligibility defect, not a property of the `k = 10,000` instance.

**The pair difference against G1** (P5.14-N's open objective comparison): G2 re-run − G1 = **+133,066.24** against a
rule-nine bar of 143,846.18 — **0.925× the bar**, so **indeterminate**: not distinguishable from stopping slack. The
two runs also differ in recovery policy (G1 had node 5 ineligible and no tier 2), so this is not a clean
single-variable comparison of `k`.

**ESSO slack values, now measured (Addendum 7 item 4).** At node 7:

| round | periods with slack > 1e-6 | largest `slack_es_pnet_down` | share of Σ\|pnet\| absorbed by slack |
|---|---|---|---|
| initialization | **123 of 288** | **0.9687** (the full request) | **10.5 %** |
| cycle 1 | 0 | −1.0e-8 (bound-relaxation floor) | 1e-16 |
| cycle 71 (last) | 0 | −1.0e-8 | 1e-16 |

This confirms, with measured values rather than duals, the reading recorded as intended behaviour: at
`k = 10,000` the initialization request would breach the SoH floor, the end-of-life rule acts through the slack,
and ADMM leaves that state by cycle 1.

**Leak:** `lg(mu)` = −8.6 in all 216 ESSO solves; detector 2.534e-5 – 2.585e-5 over cycles 1–71; 61,344 cohort-periods
barrier-set; the 864 not barrier-set are exactly the initialization round.

---

## 3. Recovery policy (Addendum 7 item 1, committed `cce9a7c0`)

- **`recovery.enabled`** is now an explicit flag for every network and the ESSO, **default true**, independent of
  `recovery_options`. `case33_1` is covered by the default. The `limited-memory` entries are **not yet removed**.
- **Tier 2**: after a failed tier-1 cold retry, one more solve, cold with `mu_strategy = adaptive`
  (`recovery.tier2_enabled`, default true), logged as `recovery_tier2` and counted separately. Tier-1 print strings
  are unchanged; the failure parser adds `recovered_tier1`, `recovered_tier2` and `unrecovered`, and still gives
  12 / 0 / 6 on G1's stdout.
- **Regressions, bitwise:** single ESSO solves match the committed tol-remedy check; a two-cycle C\* run under
  G1-equivalent policy matches G1's first two cycles, including its initialization TSO recovery and the detector.
- Harness-side policy setter: `p59_rho.set_recovery_policy`, used by both ablations and the G2 re-run.

---

## 4. The cycle-38 block (Addendum 7 item 3)

Single solves from G3-full's saved pre-solve block (`frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl`; fixture hash
unchanged). `case33_2` `tol` in force: 1e-5.

| arm | termination | iterations | final `lg(mu)` |
|---|---|---|---|
| P0 current policy as in G3-full: warm primary → cold retry | maxIterations → maxIterations | 500 / 500 | −8.0 / −8.0 |
| P1 cold | maxIterations | 500 | −8.0 |
| **P2 cold + `mu_strategy = adaptive`** | **optimal** | **66** | −7.2 |
| P3 cold, `tol` 1e-4 | maxIterations | 500 | −8.0 |
| **P4 new policy: warm → tier 1 → tier 2** | **optimal (tier 2)** | 500 / 500 / 66 | −8.0 / −8.0 / −7.2 |

P0 reproduces G3-full exactly. **Classification: path-dependent** — the warm primary fails and a cold arm succeeds.
Specifically, adaptive μ recovers it; plain cold and a looser tolerance do not. **Tier 2 recovers it.** The G2
re-run needed no tier-2 retry, so tier 2 has not yet been exercised inside a campaign.

---

## 5. Recorded without change (Addendum 7 items 4 and 5)

- G2's slack-dominated initialization is intended behaviour (§2 now measures it).
- The two overwritten `FrozenSMOPF` comparators are accepted as lost and not regenerated.

---

## 6. Decisions requested

1. **Attribution beyond the two-run budget.** The storage-cycling change lies outside Candidates 1 and 4. If the
   manuscript's account of the change needs a named cause, the next single-change candidates are the reformulated
   ESSO versus the pre-Step-1 ESSO (with Step-1b networks), or the Step 1a solver policy. Otherwise, accept the new
   formulation as the baseline with the attribution recorded as: recourse change mostly Candidate 4, cycling change
   unattributed.
2. **Remove the `limited-memory` entries** now that the explicit policy has landed (Addendum 7 item 1's condition).
3. **Tier 2 inside a campaign** has not yet fired. Whether to exercise it (e.g. re-run G3-full, whose cycle-38 block
   tier 2 recovers in isolation).
4. **The empty rows in the failure parser** recur in every campaign since G2 (2–3 per run). They do not affect any
   count reported here; a fix is small and harness-only.
5. **The recurring cycle-38 block** (node 7 `case33_2` 2035 Autumn) fails unrecovered at cycle 38 in both G3-full and
   ablation B, under different instances. Whether that coincidence merits a look.

---

## 7. What is NOT established

- Which Step-1 change moved year-1 EFC/day at C\*.
- Whether A's 1.11× recourse shortfall would close under the new recovery policy (A kept node 5 ineligible and saw 47
  local failures).
- The `k` effect on recourse at convergence: G2 re-run − G1 is within stopping slack, and the runs differ in recovery
  policy.
- Tier-2 behaviour inside a campaign.

## 8. Commits and evidence

| step | commit |
|---|---|
| expert handoff round 5 | `c3bbbebb` |
| frozen ablation criteria | `76ab3f99` |
| recovery policy, tier 2, cycle-38 tests | `cce9a7c0` |
| ablation A arm + precheck | `a3b39427` |
| G2 re-run arm + slack capture | `704f6e1e` |
| ablation A evidence and evaluation | `0f0bc866` |
| ablation B arm + precheck | `89170d73` |
| ablation B evidence and evaluation | `0c6fb284` |
| G2 re-run evidence and this report | this commit |

Evidence roots (each with `evidence_manifest_sha256.json`; per-period capture directories hash-recorded, not
committed): `data/SRP1/Results/P515A/run_a/`, `P515A/run_b/`, `P515G2R/`, `P515R/`. Evaluations:
`P515A/ablation_a_evaluation.json`, `P515A/ablation_b_evaluation.json`, `P515G2R/g2r_evaluation.json`.
