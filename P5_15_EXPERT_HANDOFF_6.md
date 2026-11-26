# P5.15 — Handoff to the external expert, round 6

**Subject: Addendum 7 is complete. The ablation budget ends with the attribution unresolved but split — Candidate 4
explains ~90 % of G1's recourse change, Candidate 1 explains nothing, and neither explains the drop in storage
cycling. G2 converges once node 5 is eligible for recovery. The explicit recovery policy and tier-2 retry are in
production; the cycle-38 block is path-dependent and tier 2 recovers it.**

Planner report, 2026-09-14. Continues `P5_15_EXPERT_HANDOFF_5.md`. Authority: `PLANNER_BRIEF_2026-09-13.md`,
Addendum 7. Detail: `P5_15_ADDENDUM7_REPORT.md`. Everything is committed through `1c80dd19`.

---

## 1. Status against the Addendum-7 order

| step | outcome | commit |
|---|---|---|
| commit handoff round 5 | done | `c3bbbebb` |
| recovery policy (explicit `recovery.enabled`, tier 2) | in production; regressions bitwise | `cce9a7c0` |
| cycle-38 block, single solves | **path-dependent; tier 2 recovers it** | `cce9a7c0` |
| ablation criteria frozen (before any run) | `frozen_ablation_spec_v1_2271c77f.json` | `76ab3f99` |
| ablation A — Candidate 4 reverted | converged; **fails criterion** | `a3b39427`, `0f0bc866` |
| ablation B — Candidate 1 re-wired | converged; **fails criterion** | `89170d73`, `0c6fb284` |
| G2 re-run — node 5 eligible, slack capture | **converges, zero local failures** | `704f6e1e`, `1c80dd19` |

All campaigns: one background run through the tool, stderr captured, exclusive lock, per-cycle heartbeat, fresh
output root and eval id, Python exit code recorded, shared `FrozenSMOPF` directory hash-checked (untouched in every
run).

---

## 2. Ablation attribution — the lead item

**Design.** Each arm is G1's configuration with **one** Step-1b change undone, including G1's own recovery behaviour
(tier 2 off, node 5 ineligible), so each isolates a single change. The success criterion was frozen and committed
before either ran: an arm recovers the old operating point only if it closes **≥ 90 % of the year-1 EFC/day gap**
between G1 and the old control on **all three nodes**, **and** lands **within the rule-nine bar** of the old recourse,
and only if it converged and settled (rule ten < 1).

| arm | change | cycles | rule ten | year-1 EFC/day (n5 / n7 / n9) | EFC gap closed | recourse | Δ vs old (× bar) | Δ vs G1 |
|---|---|---|---|---|---|---|---|---|
| old control (P5.14-N) | — | 67 | 0.933 | 1.1124 / 1.1122 / 1.1124 | — | 816,121,464.16 | — | — |
| G1 | — | 72 | 0.891 | 0.9725 / 0.9722 / 0.9724 | 0 % | 817,618,798.07 | +1,497,334 (10.05×) | — |
| **A** | **Candidate 4 reverted** — flexibility day-balance band restored | 69 | 0.799 | 0.9788 / 0.9760 / 0.9761 | **4.5 / 2.7 / 2.7 %** | 816,278,501.63 | +157,037 (**1.11×**) | **−1,340,296** |
| **B** | **Candidate 1 re-wired** — shared-ESS power-factor rows restored | 71 | 0.846 | 0.9737 / 0.9753 / 0.9737 | **0.8 / 2.2 / 1.0 %** | 817,644,779.89 | +1,523,316 (10.48×) | **+25,982** |

**Verdict under the frozen criterion: neither arm is the cause; the two-run budget is exhausted; attribution
unresolved.**

**What the two runs establish.**

- **Candidate 4 accounts for ~89.5 % of G1's recourse change** (−1,340,296 of −1,497,334) while leaving storage
  cycling and SoH where G1 had them. It misses the recourse criterion narrowly (1.11× the bar) and the EFC criterion
  widely.
- **Candidate 1 accounts for neither** quantity.
- **The storage-cycling drop (year-1 EFC/day 1.112 → 0.972) is explained by neither.** In A and B it stays within
  0.007 of G1. It lies in a change not ablated: the reformulated ESSO (log-domain SoH, investments as parameters,
  deleted slack and complementarity families, leak reduced from ~1.39 % to ~0.009 %), the Step 1a solver and recovery
  policy, remedy (h), or Candidates 2 and 5. **Which one is not established.**

**Reading, not established.** Candidate 4 replaced a two-sided band `|Σ(p_up − p_down)| ≤ SMALL_TOLERANCE` with a
penalized slacked equality. That reverting it moves recourse and not storage dispatch is consistent with the change
acting through the flexibility cost inside the recourse, not through the storage's operation.

**Side effects seen in the ablations.**

- **A:** with the band restored, node 5 `case33_1` 2025 Autumn failed `maxIterations` in every cycle from 17 to 61
  (47 failures, never retried under G1's policy). The run converged regardless.
- **B:** with the power-factor rows restored, network failures fell to 3 (2 recovered, 1 unrecovered) against 18 in
  G1 and 66 in A. B's unrecovered failure is node 7 `case33_2` 2035 Autumn at **cycle 38** — the same block and cycle
  label as G3-full's unrecovered failure, on a different instance.

---

## 3. G2 re-run with node 5 eligible — the second lead item

G2's configuration (`k = 10,000` on C\*) under the new recovery policy: every network and the ESSO eligible,
`case33_1` included, tier 2 on; ESSO slack values captured per period.

| | G2 (original, round 5) | **G2 re-run** |
|---|---|---|
| converged | **no** — 90-cycle cap | **yes — cycle 71** |
| local-solve failures | 31 cycles | **0** |
| network failures | 18 recovered, **33 not attempted** | **23 recovered on tier 1**; 0 tier 2; 0 unrecovered; 0 not attempted |
| node 5 `case33_1` | failed every cycle 66–90, never retried | 9 failures, **all recovered** on the cold retry |
| recourse | none (last valid, cycle 65: 818,690,717.51, not settled) | **817,751,864.31** |
| rule ten | — | 0.868 |
| solves | 4,659 | 3,695 |
| SoH, node 7 (yr 1 → 3) | 0.8412 → 0.7177 → 0.6069 (unconverged) | **0.8375 → 0.7144 → 0.6042** |
| year-1 EFC/day (n5 / n7 / n9) | — | 0.9720 / 0.9718 / 0.9725 |

**Answer: G2 converges once node 5's failures are eligible for recovery.** G2's non-convergence was the
recovery-eligibility defect, not a property of the `k = 10,000` instance. The 2035 Autumn block that failed every
cycle from 66 to 90 in G2 did not fail at all in the re-run: early recoveries at cycles 2–3 changed the path.

**Pair difference against G1** (P5.14-N's open objective comparison): +133,066.24 against a rule-nine bar of
143,846.18 — **0.925× the bar, indeterminate**, i.e. not distinguishable from stopping slack. The two runs also differ
in recovery policy (G1 had node 5 ineligible, no tier 2), so this is not a clean single-variable comparison of `k`.

**ESSO slack, now measured** (Addendum 7 item 4, recorded as intended behaviour). Node 7:

| round | periods with slack > 1e-6 | largest `slack_es_pnet_down` | share of Σ\|pnet\| absorbed |
|---|---|---|---|
| initialization | **123 of 288** | **0.9687** (the full request) | **10.5 %** |
| cycle 1 | 0 | −1.0e-8 (bound-relaxation floor) | ~1e-16 |
| cycle 71 (last) | 0 | −1.0e-8 | ~1e-16 |

The end-of-life rule acts through the slack at initialization and ADMM leaves that state by cycle 1 — now measured
rather than inferred from duals.

**Leak** unchanged: `lg(mu)` = −8.6 in all 216 ESSO solves; detector 2.534e-5 – 2.585e-5 over cycles 1–71; every
cycle period barrier-set (61,344); the 864 not barrier-set are exactly the initialization round.

---

## 4. Recovery policy (Addendum 7 item 1)

- **`recovery.enabled`** is an explicit flag for every network and the ESSO, **default true**, independent of whether
  `recovery_options` is populated. `case33_1` is eligible by default.
- **Tier 2:** after a failed tier-1 cold retry, one further solve — cold, `mu_strategy = adaptive`
  (`recovery.tier2_enabled`, default true) — logged as `recovery_tier2` and counted separately.
- **Unchanged:** tier-1 print strings; recovery options applied exactly as before. The failure parser adds
  `recovered_tier1`, `recovered_tier2`, `unrecovered` and still reproduces G1 exactly (12 / 0 / 6).
- **Regressions, bitwise:** single ESSO solves match the committed tol-remedy check; a two-cycle C\* run under
  G1-equivalent policy matches G1's first two cycles, including its initialization TSO recovery and the detector.
- **The `limited-memory` entries are not yet removed** — Addendum 7's precondition (policy landed) is now met; §6.

---

## 5. The cycle-38 block (Addendum 7 item 3)

Single solves from G3-full's saved pre-solve block (fixture hash unchanged). `case33_2` `tol` in force: 1e-5.

| arm | termination | iterations | final `lg(mu)` |
|---|---|---|---|
| P0 current policy as in G3-full: warm → cold retry | maxIterations → maxIterations | 500 / 500 | −8.0 / −8.0 |
| P1 cold | maxIterations | 500 | −8.0 |
| **P2 cold + `mu_strategy = adaptive`** | **optimal** | **66** | −7.2 |
| P3 cold, `tol` 1e-4 | maxIterations | 500 | −8.0 |
| **P4 new policy: warm → tier 1 → tier 2** | **optimal (tier 2)** | 500 / 500 / 66 | −8.0 / −8.0 / −7.2 |

P0 reproduces G3-full's failure exactly. **Classification: path-dependent** — the warm primary fails, a cold arm
succeeds. Specifically **adaptive μ** recovers it; plain cold and a decade-looser tolerance do not. **Tier 2 recovers
it.** Tier 2 has not yet fired inside a campaign (the G2 re-run did not need it).

---

## 6. Decisions requested

1. **Attribution beyond the budget.** If the manuscript's account needs a named cause for the storage-cycling change,
   the next single-change candidates are the reformulated ESSO against the pre-Step-1 ESSO (with Step-1b networks),
   or the Step 1a solver policy. Otherwise, record: *recourse change mostly Candidate 4; cycling change unattributed;
   the new formulation is the baseline.*
2. **Remove the `limited-memory` entries** from the case files now that eligibility no longer depends on them.
3. **Exercise tier 2 inside a campaign** — e.g. re-run G3-full, whose cycle-38 block tier 2 recovers in isolation.
4. **The recurring cycle-38 block** (node 7 `case33_2` 2035 Autumn) fails unrecovered at cycle 38 in both G3-full and
   ablation B, on different instances. Whether that coincidence merits investigation.
5. **Failure-parser empty rows** (2–3 per campaign since G2; no reported count affected) — a small harness-only fix.

---

## 7. What is NOT established

- Which Step-1 change moved year-1 EFC/day at C\*.
- Whether A's 1.11× recourse shortfall would close under the new recovery policy (A kept node 5 ineligible and had 47
  local failures).
- The effect of `k` on recourse at convergence (G2 re-run − G1 is within stopping slack; recovery policies differ).
- Tier-2 behaviour inside a campaign.
- Why the same block fails at cycle 38 in two different instances.

---

## 8. Evidence

Each run's evidence is committed with a full sha256 manifest; per-period capture directories are hash-recorded,
not committed. All paths under `data/SRP1/Results/`.

| item | location |
|---|---|
| frozen ablation criteria | `P515A/frozen_ablation_spec_v1_2271c77f.json` |
| ablation A | `P515A/run_a/`, evaluation `P515A/ablation_a_evaluation.json`, precheck `P515A/precheck_a.json` |
| ablation B | `P515A/run_b/`, evaluation `P515A/ablation_b_evaluation.json`, precheck `P515A/precheck_b.json` |
| G2 re-run | `P515G2R/`, evaluation `P515G2R/g2r_evaluation.json` |
| slack-capture validation | `P515A/g2r_slack_capture_validation.json` |
| recovery policy regressions and cycle-38 tests | `P515R/` (`r1_t3_recheck/`, `r2_g1policy/`, `part3_cycle38/`) |
| unrecovered cycle-38 pre-solve blocks | `P515G3F_r2/results/FrozenSMOPF/…cycle38.pkl`, `P515A/run_b/results/FrozenSMOPF/…cycle38.pkl` |

Reports: `P5_15_ADDENDUM7_REPORT.md`, `WORKER_REPORT_RECOVERY.md`.
