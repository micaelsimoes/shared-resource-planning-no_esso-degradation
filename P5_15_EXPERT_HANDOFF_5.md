# P5.15 — Handoff to the external expert, round 5

**Subject: gates G1–G4 have run under the Addendum-6 capture. The TSO failures at C\* all recover; the C\*
leak is barrier-set by IPOPT's terminal barrier parameter; G4 is bitwise deterministic. G1's reconciliation
fails, G2 does not converge, and G3-full converges with one unrecovered failure. Two of these outcomes trace to
a recovery-eligibility rule that depends on dead configuration.**

Planner report, 2026-09-14. Continues `P5_15_EXPERT_HANDOFF_4.md`. Authority: `PLANNER_BRIEF_2026-09-13.md`,
Addendum 6. Detail: `P5_15_G1_REPORT.md`, `P5_15_G2_REPORT.md`, `P5_15_G3F_REPORT.md`, `P5_15_G4_REPORT.md`.
Everything is committed through `7eb76209`.

---

## 1. Status against the Addendum-6 order

| step | state | commit |
|---|---|---|
| commit the log fix | done | `7ca40b93` |
| G1 capture instrumentation (harness) + two-cycle smoke | done | `4b56c835` |
| **G1** — C\* control | converged; **reconciliation fails** | `27b45536` |
| G2 prep — per-arm `results_dir`, fresh roots, print-based failure parser | done, validated with zero solves | `7a2d85db` |
| **G2** — `k = 10,000` | **no convergence** | `c2361c26` |
| **G3-full** — 1.62 MVA / 3.24 MWh, node 7 | converged; **zero-failures not met** | `042de767` |
| **G4** — G1 repeated | **pass, bitwise** | `7eb76209` |
| `REVISION_CONTEXT.md` head rewrite | done | `7eb76209` |

Every campaign ran as one background process through the tool, stderr captured, exclusive lock, per-cycle
heartbeat, fresh output root and eval id, never concurrently.

---

## 2. G1 TSO failure classification — the failure lost in the earlier attempt

Classified from production's own prints (`network.py` recovery messages), with cycle attribution
cross-checked against the ADMM trajectory.

| class | G1 | G2 | G3-full | G4 |
|---|---|---|---|---|
| recovered | 12 (TSO 3) | 18 (TSO 3) | 16 (TSO 8) | as G1 |
| unrecovered | 0 | 0 | **1** (DSO) | as G1 |
| not attempted (ineligible) | 6 | 33 | 1 | as G1 |

**Every TSO local failure in every gate was a primary `maxIterations`-class failure that recovered on the
single cold retry.** In G1 these were `case9` 2035 Winter (initialization / cycle 1), 2025 Autumn (cycle 5),
2030 Winter (cycle 40). No TSO block was left unsolved in any gate.

**All not-attempted failures are DSO node 5 `case33_1`, termination `maxIterations`.** In G1 they fall on
cycles 7, 39, 43, 46, 48, 49 — exactly the cycles the ADMM trajectory marks `local_solves_ok = False`.
They were never retried because `_is_recoverable_network_failure` returns False when `recovery_options` is
empty, and `case33_1_params.json` has no `recovery_options` (§6.1).

**The first unrecovered failure** occurred in G3-full: DSO node 7 `case33_2` 2035 Autumn, cycle 38 — primary
`maxIterations` (warm start), cold retry `maxIterations` again. The same block had recovered twice earlier in
that run. Its pre-solve block was saved as
`P515G3F_r2/results/FrozenSMOPF/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl` — a reproducible fixture.

Solve accounting reconciles exactly: G1's 3,735 solves = 51 × 72 + 51 + 12 recoveries.

---

## 3. Per-cycle detector trajectory at C\*

Measured detector `max min(pch,pdch)/s_max`, every ESSO solve, every cycle; only the measured detector is
quoted (Addendum 6).

| gate | rounds × nodes | detector range (cycles ≥ 1) | terminal `lg(mu)` | drift |
|---|---|---|---|---|
| G1 (C\*) | 73 × 3 | 2.504e-5 – 2.586e-5 | −8.6 in all 219 solves | none |
| G2 (C\*, k = 10,000) | 91 × 3 | 2.569e-5 – 2.585e-5 | −8.6 in all 273 | none |
| G3-full (node 7, s_max = 1.62) | 81 × 1 | 1.529e-5 – 1.546e-5 (≡ 2.50e-5 absolute) | −8.6 in all 243 | none |
| G4 | 73 × 3 | identical to G1, 0 of 219 differ | −8.6 | none |

- **The dual-magnitude drift derived in round 1 did not occur.** `s_obj = 0.1` throughout, and the terminal
  barrier parameter is the same in every solve.
- **Initialization rounds differ** (§4).
- G3-full's detector is smaller only because it is a ratio over `s_max = 1.62`; the absolute small leg is the
  same ≈2.50e-5 as at C\*. (The harness's detector/prediction comparison mixes ratio and absolute forms and reads
  0.61 there; in absolute terms it is 0.986–0.997.)

---

## 4. Leak-mechanism classification — barrier-set

Test (Addendum 6): per period and cohort, the small leg `x_small = min(pch,pdch)` and its IPOPT bound
multiplier `zL`, compared with the barrier parameter.

| gate | cohort-periods barrier-set (`r_bar = zL·x / (μ/s_obj)`) | not barrier-set |
|---|---|---|
| G1 | **63,072 of 63,072** | 0 |
| G2 | 77,760 | 864 = exactly the initialization round |
| G3-full (node 7) | 23,040 | 288 = exactly the initialization round |

`μ` here is **IPOPT's terminal barrier parameter** (`lg(mu)` column, last iteration). Measured detector /
`μ/(s_obj·ε)` = 0.997–1.029 in every G1 solve.

**Mechanism.** At the argmax periods the storage is idle (`pnet` ≈ 1e-9 – 1e-6) with both legs ≈ 2.5e-5. At
idle the aggregate-row multiplier is ≈ 0, so each leg's bound multiplier is ≈ `s_obj·ε` and
`x_small = μ/(s_obj·ε)`. With one large leg (the ε fixture), `x_small ≈ μ/(2·s_obj·ε)`.

**Two earlier readings withdrawn.** Addendum 6's premise that the C\* leak is "not barrier-set (μ-insensitive)"
and the Planner's round-4 Finding 1 both arose from treating the summary `Complementarity` line — an
optimality-error measure — as μ. Across the round-4 probe that line varied 45 % while `lg(mu)` stayed at −8.6.

**The 5× gap is closed.** Fixture vs C\* initialization = terminal μ (−9.0 vs −8.6: ×2.51) × leg symmetry
(one large leg vs idle: ×2) × prefactor (0.907 vs 0.997: ×1.10) = **5.52**, against 5.519 observed.

**Initialization rounds of G2 and G3-full: SoH floor binding, request absorbed by penalized slack.** At G2's
initialization (node 7, period y0/d0/p0) the request is full discharge (`pnet` = −0.9687) while both legs sit at
their relaxed bound (≈ −1e-8, `zL` ≈ 1.94e3); the aggregate-row dual is 999.99 = `PENALTY_ESSO_SLACK`; the
largest duals, in every period, are on `energy_storage_capacity_degradation` (≈ −2.5e5). At `k = 10,000` the
initialization request would push SoH below `soh_min`, so the ESSO does not dispatch and pays the slack
penalty; G2's ADMM therefore starts from an ESSO state that does nothing. G3-full's initialization (base k,
energy/power ratio 2) shows the same signature in part of its periods. Inferred from duals; slack values were
not captured.

---

## 5. Gate outcomes

### G1 — C\* control: converged; reconciliation FAILS

72 cycles, recourse **817,618,798.07**, rule ten 0.891.

Criterion (Addendum 3 item 3): measured Δ(SoH) against the old control equals the Δ predicted from the two
leak fractions (old 1.386–1.391 %, new ≈ 0.009 %) within 10 %.

| | node 5 | node 7 | node 9 |
|---|---|---|---|
| measured / predicted ΔSoH, yr 1 / 2 / 3 | 9.23 / 5.70 / 4.06 | 9.20 / 6.35 / 4.37 | 9.56 / 6.22 / 4.29 |
| year-1 EFC/day, old → new (leak-only prediction) | 1.112 → 0.973 (−0.015) | 1.112 → 0.972 (−0.015) | 1.112 → 0.972 (−0.015) |

Recourse vs old control: **+1,497,333.92** against a rule-nine bar of 149,029.99 — **10.05×**.

**Reading.** The reformulated model reaches a *different operating point*, not a leak-corrected version of the
old one: it cycles the storage much less in year 1 and ages it less. Since the old control the model changed in
more than the leak — reformulated ESSO, Step 1b network candidates, Step 1a solver/recovery policy, remedy (h).
**G1 cannot attribute the change among them.**

### G2 — `k = 10,000`: FAILS

No convergence in 90 cycles; recourse undefined (cycle 90 had a failed local solve). 31 failed-local-solve
cycles; **from cycle 66 to 90 the same block — node 5 `case33_1` 2035 Autumn — failed every cycle and was never
retried** (§6.1). Last valid recourse, cycle 65: 818,690,717.51 (+1,071,919.44 vs G1), with a step 1.60× its
tolerance — not settled, so the pair difference is **indeterminate** and the P5.14-N comparison remains
unanswered. SoH 0.842 → 0.718 → 0.608 (unconverged).

### G3-full — 1.62 MVA / 3.24 MWh at node 7: converged; zero-failures NOT met

80 cycles, recourse 819,016,107.91 (different instance, not comparable with G1), rule ten 0.867. Two local
failures: one unrecovered (§2), one ineligible. Node 7 SoH 0.846 → 0.705 → 0.596; EFC/day max 1.158 (below
1.4612). A first attempt stopped on a false harness pre-check (zero-investment nodes have fixed legs and no
multipliers); corrected and re-run under fresh names; the failed attempt is preserved.

### G3-init — PASS (reported round 3). G5 — PASS (re-specified, Addendum 4).

### G4 — determinism: PASS, bitwise

G1 repeated: 72 cycles, recourse 817,618,798.07215, 3,735 solves, same six failed cycles; **zero differences**
across the per-cycle recourse, step and primal/dual residual series, SoH, `D`, `avg_ch_dch`, and all 219
per-solve detector values. Production is identical between the G1 and G4 commits while the harness changed,
so the capture does not perturb the solve path.

---

## 6. Findings that need a decision

### 6.1 Recovery eligibility is decided by dead configuration

`network.py:_is_recoverable_network_failure` requires a **non-empty** `recovery_options`. The retry itself
discards `hessian_approximation` (Step 1a). Current case files:

| case | `recovery_options` | eligible |
|---|---|---|
| `case33_1` | **absent** | **never** |
| `case33_2`, `case33_3` | `{hessian_approximation: limited-memory}` only (discarded by the retry) | yes — only because of that entry |
| `case9` | `limited-memory` + `acceptable_tol`, `acceptable_iter` | yes |

Consequences: node 5's failures in every gate were ineligible by accident of configuration, and **G2's
non-convergence is driven by a block that was never retried.** **Hazard:** the planned removal of the "dead"
`limited-memory` entries (Addendum 2) would empty `recovery_options` for `case33_2` and `case33_3` and
**silently disable network recovery** there. This is a production policy defect, not changed by the Planner.

### 6.2 Two preserved comparators were overwritten and are not recoverable

During G1, production wrote `matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` and
`matched_success_TSO_case9_2025_Summer_cycle7.pkl` into the shared `data/SRP1/Results/FrozenSMOPF/`. Their
audited hashes (`8eabd9ee…`, `15ce6ebe…`; `P3_AUDIT_REPORT.md`, confirmed by `P44/p44_report.json`) no longer
match, and the P5.12-R copies are different captures. Cause: `fresh_planning` isolated `logs_dir` but not
`results_dir`. From G2 on, every arm redirects `results_dir` to its own root and hash-checks the shared
directory; G2, G3-full and G4 left it untouched. Attribution to G1 is most likely but not proven (no hash check
between P44 and G1). The P5.12-W/X frozen plans reference the P512R copies and are unaffected.

### 6.3 Smaller defects recorded

- The failure parser emits a few empty rows (G2: 2, G3-full: 3; none on G1's validation) — unexplained.
- The harness's detector/prediction comparison mixes a ratio with an absolute value (visible when `s_max ≠ 1`).
- Pickle hashes are not byte-stable across numerically identical runs and are not a determinism test.

---

## 7. Decisions requested

1. **Recovery eligibility (§6.1).** Decouple eligibility from the contents of `recovery_options` (e.g. an explicit
   flag or policy default), give `case33_1` the same policy as the other networks, and do not remove the
   `limited-memory` entries until then. After the change, **re-run G2** to learn whether it converges when node
   5's block is eligible.
2. **Attributing G1's operating-point change.** G1 shows the reformulated model reaches a different C\*
   operating point but cannot say why. Options: a single-change ablation at C\* (e.g. reformulated ESSO with the
   pre-Step-1b network, or vice versa), or accept the new operating point and re-baseline every downstream result
   on it. The Planner recommends deciding this before any manuscript numbers are regenerated.
3. **G3-full's unrecovered block.** Whether to test the saved cycle-38 pre-solve block for intrinsic vs
   path-dependent failure (zero ADMM cycles; a handful of single solves).
4. **G2's slack-dominated initialization (§4).** Whether an initialization request that drives SoH below the
   floor at `k = 10,000` is an intended property of the formulation or a defect in how ADMM is initialized.
5. **The comparator loss (§6.2).** Accept and record, or regenerate comparators under a new, recorded identity.

---

## 8. What is NOT established

- Which Step-1 change moved the C\* operating point.
- Whether G2 converges if node 5's failing block is recovery-eligible.
- Whether the G3-full cycle-38 failure is intrinsic or path-dependent.
- Slack values at the G2/G3-full initializations (inferred from duals).
- H3 on any multi-cohort instance (inert everywhere tested).
- Whether the audited FrozenSMOPF originals were overwritten by G1 or earlier.

---

## 9. Evidence

Each gate's evidence is committed with a full sha256 manifest; the large per-period capture directories are
hash-recorded, not committed.

| gate | result file (sha256, first 16) | manifest |
|---|---|---|
| G1 | `P515G1/g_control.json` `5dcae17acccd31b1` | `P515G1/evidence_manifest_sha256.json` |
| G2 | `P515G2/g_k10000.json` `149cee00d7d26f70` | `P515G2/evidence_manifest_sha256.json` |
| G3-full | `P515G3F_r2/g_g3_full_node7.json` `a1238d41d42b3a20` | `P515G3F_r2/evidence_manifest_sha256.json` |
| G4 | `P515G4/g_control_rep2.json` `19b08ffa007c87b6` | `P515G4/evidence_manifest_sha256.json` |
| G3-full failure fixture | `…/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl` `4499cf99a4341eb9` | as G3-full |
| G1 parser validation | `P515G1_parser_validation/` | committed `7a2d85db` |

All paths under `data/SRP1/Results/`. `REVISION_CONTEXT.md` now carries a 2026-09-14 current head recording
these outcomes and hazards; the previous head is retained unedited as superseded.
