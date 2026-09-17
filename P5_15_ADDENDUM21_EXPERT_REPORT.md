# P5.15 Addendum 21 — handoff to the External Expert and the Author

**Planner report, 2026-09-17.** Self-contained; continues `P5_15_ADDENDUM20_EXPERT_REPORT.md`. Stage tables:
`P5_15_S39_ORACLE_REPORT.md` (`2d786718`).

- **Instance:** C\* (0.96875 MVA / 3.875 MWh per node 5/7/9, invested 2025).
- **Cost convention:** `gross_operational_cost`.
- **Bar (Addendum 21):** all three channels inside their Boyd tolerances, every local solve successful, for **10
  consecutive cycles**, within cap 300. Terminal ratios, rule ten, clamp flags and cost are reported, not gated.

## 1. Verdict

**Both authorized arms certify under the new bar. The PF work is finished; the storage channel is now the binding one.
By spec v10's rule the oracle is C, but that rule turns on a condition no arm has ever approached.**

| arm | configuration | certified | at cycle | PF / V / ESS first pass | storage terminal | rule ten |
|---|---|---|---|---|---|---|
| **C** `s39_C` | τ = 0, PF balancing live, ρ_ess 0.01 fixed and exempt | **yes** | **191** | 130 / 48 / 131 | 0.809 | 0.0073 |
| **D** `s39_D` | C + two-phase ESS schedule | **yes** | **139** | 125 / 48 / **75** | 0.786 | 0.0481 |
| run 1 | τ = 1, ESS balanced, backstop 60 | yes, Addendum 16 rule | 477 | 226 / 45 / 475 | 0.988 | 0.0039 |

- **Predictions:** C's PF ≤ 131 **met** (130); C's certification ≤ 150 **missed** (191); D certifies **met** (139);
  D's storage < 0.5 **missed** (0.786).
- **Adoption by the rule: C.** D certifies 52 cycles sooner but misses the storage condition, so spec v10 falls back
  to C. §4 gives the case for reading this differently.
- **No clamp on any channel in either arm; no local-solve failures; the TSO-instability trigger did not fire** (so
  τ = 0 globally stands, and per-channel τ remains an unused recorded fallback).

| item | status | commit |
|---|---|---|
| Spec v10, written before any code or run; Addendum 21 recorded in `REVISION_CONTEXT.md` | done | `e2fd8e60`, `fd25503b` |
| Production two-phase ESS policy (default-off) + zero-solve checks | done | `6d628d7e`, `8cdd0978` |
| D-policy replay on v9 arms; harness arms, evaluator, preflight | done | `0f9918e8`, `3ee7bb37`, `cf480653`, `054eed0c` |
| Preflight re-run at committed HEAD; corrected bitwise reading | done (§6) | `7f660765`, `c231ddff` |
| Arm C certified; D-policy replay on C; pre-launch note | done | `4c0def22` |
| Arm D certified; both evaluations; node 7 extended; stage report | done | `2d786718` |
| Step 3.6 clone replacement | complete, **not integrated** (§5) | worktree `0c29f49d`…`bff03b7e` |

## 2. What each lever did

### 2.1 PF: settled

ρ_pf behaved identically in both arms: one decrease at **cycle 2** (0.198 → 0.132), then frozen from cycle 12 by the
10-unchanged rule. With τ = 0 the PF dual ratio is 5.7× the primal by cycle 2, so the decrease rule fires immediately,
where arm B (τ = 1) first fired at cycle 79. PF first passes at 130 (C) and 125 (D), against 226 in run 1, and ends
comfortably inside tolerance (0.325, 0.486). **HP1 + HP2 are confirmed and exhausted as levers.**

### 2.2 Storage: improved, still not settled

| | C | D |
|---|---|---|
| ρ_ess | 0.01 throughout (exempt) | 0.01 → **0.015 (cycle 32)** → **0.0225 (cycle 33)**, frozen from 43 |
| exemption lift | never | **cycle 31** |
| ESS first pass | 131 | **75** |
| storage terminal ratio | 0.809 | 0.786 |
| certification cycle | 191 | **139** |

- **The policy matched its pre-registered replay exactly** (lift 31, first action ρ_ess 0.01 → 0.015 at 32), recorded in
  `P5_15_S39_D_PRELAUNCH_NOTE.md` before D ran.
- **The ratchet risk did not materialise.** The open-loop replay predicted 28 further firings to ρ_ess ≈ 1.3e3; the real
  run took two increases and froze. ρ_ess never exceeded 0.0225.
- **It bought 52 cycles but did not settle the channel.** Both arms end at 0.79–0.81, and D still crosses 1 occasionally
  near the end. A 2.25× penalty increase damped the hovering without removing it.
- **C's missed cycle prediction is entirely this channel.** Its longest all-pass run before certification was 5 cycles
  (131–135); under the old 3-cycle rule C would have "stopped" at 133. **The new bar is doing exactly what it was
  introduced to do.**

## 3. Cost — and why it matters more than it looks

| run | certified cost | difference to run 1 | new-bar cost bar | inside? |
|---|---|---|---|---|
| run 1 | 651,039,166 | — | — | — |
| v9 arm A | 651,000,017 | −39,149 | (old bar 3,140) | no |
| C | 650,981,423 | **−57,743** | 734 | **no** |
| D | 650,966,975 | **−72,191** | 8,155 | **no** |

- **Three independent configurations now certify 39k–72k below run 1, all in the same direction**, and they differ from
  each other by up to 33k.
- **What this is not.** It is not run-to-run noise: the programme's runs are deterministic at fixed configuration
  (Step 3.0 determinism baseline; I have not re-verified determinism for τ = 0, and say so). It is **configuration
  dependence** — each arm stops at a different point of the same descent, under a different penalty policy.
- **What it is not, second.** It is not evidence of a lower-cost fixed point. The rule-nine bar is local and valid only
  between settled runs; run 1's storage channel terminated at 0.988, stopped rather than settled, and the arms' at
  0.79–0.81.
- **Why it matters for Step 5.** P5.10's accepted statistics put the **best-to-second candidate gap at 32.87** against a
  numerical floor of 10, on a cost of ≈8.3e8. Configuration changes move the evaluated cost by **10⁴–10⁵**, i.e. three
  to four orders of magnitude more than the ranking signal. Candidate ranking is therefore only meaningful if **every**
  candidate is evaluated under one frozen oracle configuration, and absolute costs from different configurations must
  never appear in the same table (run 1's 651,039,166 and C's 650,981,423 are not comparable quantities).
- **Cheapest way to close this, if wanted:** a zero-solve component decomposition of C and D against run 1 from the
  committed terminal artifacts (`component_levels_terminal.json`, `interface_settlement_detail_s31c.json`) to show
  *where* the 58k–72k sits — generation, flexibility, curtailment, storage or transfer. Not run; not authorized.

## 4. The adoption question

**By spec v10's rule, C is the oracle**, because D missed "storage terminal ratio < 0.5". Three observations bear on
whether that is the right reading; I have acted on none of them.

1. **D dominates C on convergence** — certification 52 cycles sooner (139 vs 191), storage inside tolerance 56 cycles
   sooner (75 vs 131), a slightly lower storage terminal ratio (0.786 vs 0.809), fewer network failures (38 vs 61),
   identical PF behaviour. Its whole advantage is the storage lever.
2. **The < 0.5 condition may repeat criterion (c)'s error.** Addendum 21 withdrew "storage < 0.9" because the channel
   closing a stop ends near 0.95 by construction. Under the 10-cycle bar the closing channel has longer to decay, but
   storage is still the binding channel in both arms and still lands at 0.78–0.81. **No arm in the programme has ever
   ended below 0.78 on this channel.** The condition may again be measuring which channel closes rather than how
   settled storage is.
3. **Against D: it is the less settled run by objective measures.** Rule ten 0.0481 vs C's 0.0073; largest objective
   step in the last 10 cycles 7,899 vs 478. D meets the residual bar while its cost is still moving. If the oracle's
   job is a stable cost for Step 5, that is an argument for C independent of cycle count.

**Planner's reading:** the residual bar and the objective behaviour disagree about which arm is "better settled", and
the spec's tie-breaker turns on a threshold no arm approaches. The cleanest resolution is to decide what the oracle must
deliver — fastest certification, or quietest terminal cost — and restate the condition in those terms.

## 5. Step 3.6 — clone replacement complete, not integrated

In an isolated worktree (detached at `fd25503b`): `0c29f49d`, `b2a9ecae`, `816090b0`, `bff03b7e`.

- **Equivalence:** rebuild-from-capture vs legacy clone — 0 differences across every Var value, bound and fixed flag,
  every mutable Param, every warm-start Suffix, active constraint/objective sets, and 3,378 (TSO) / 7,779 (DSO)
  expression strings, on real blocks.
- **Clones per cycle: 12 → 0**, rising to exactly 1 when a snapshot must actually be written.
- **Capture vs clone:** 0.022 s vs 0.088 s (TSO), 0.068 s vs 0.199 s (DSO).
- **All 7 preserved FrozenSMOPF fixtures still unpickle.**
- **Design deviation, flagged:** rather than re-invoking the constructor on demand (the brief's literal wording), it
  clones one pristine TSO block per (year, day) once per run and replays captured state onto it. Re-invoking the
  constructor would re-fix the interface loads from the **mutated** consensus values instead of their cycle-0 values —
  a genuine correctness bug the Worker found by reading `create_transmission_network_model`.
- **Not integrated**, pending this review: the plan is cherry-pick → 2-cycle bitwise preflight (legacy vs capture) →
  10-cycle re-measurement → then decide on persistent workers. The earlier 2-cycle measurement put absorbable overhead
  at 97% (bar 70%), with clones 4.1–4.4 s of a ~35 s cycle.

## 6. Process notes

- **Preflight provenance.** The preparation Worker ran the first C/D preflights at 18:36–18:38 against **uncommitted**
  code; its commits landed at 19:44–19:46 after further edits. Those preflights are retained but not cited. I added a
  suffix mechanism and re-ran both at committed HEAD (`7f660765`) under their own working-dir ids, then launched.
  `P5_15_S39_PREFLIGHT_NOTE.md`.
- **A Planner error, corrected by the Worker.** My preflight instruction asserted C and D must reproduce v9 arm A's
  cycles 1–2 bitwise. They cannot: PF balancing is live in C/D and exempt in A, and at τ = 0 it fires at cycle 2. Of 192
  fields, 2–3 differ per cycle — the ρ_pf action and balancing bookkeeping — and everything numerical matches. The
  Worker reported this rather than adjusting the comparator to pass.
- **Working-dir collisions closed structurally.** The s38 preflights had left pre-check and probe directories under the
  ids the real launch checks; s39 derives all three ids from the run mode, and the zero-solve checks assert
  disjointness. No collision occurred in this stage.
- **`balancing_exempt_ess` reads False for arm D while storage is exempt**, by design: the action label
  (`exempt (fixed)`) and the `ess_exempt_until_state_s39_D.jsonl` sidecar carry the true state, and the evaluator uses
  those. Anyone reading the boolean alone would misread arm D.
- **An agent worktree was provisioned 343 commits stale**, at a commit predating the whole P5 campaign; that Worker
  stopped and changed nothing. I created a correct detached worktree at the current commit and re-ran the task.

## 7. Node 7 (Addendum 21 item 3, now on four arms)

| node | rating | max utilization | periods at rating (of 288) | of which 2030 |
|---|---|---|---|---|
| 5 | 200 MVA | 0.510 | 0 | 0 |
| **7** | **100 MVA** | **1.000** | **23** | 8 |
| 9 | 150 MVA | 0.672 | 0 | 0 |

Identical in C and D to four significant figures, and consistent with v9 arms A and B (22 periods). Node 7's interface
branch is the only one that binds, and the shared storage at node 7 is also at its rating (1.00003). Interface voltage
sits within ~3e-6 pu of its bound at **all three** nodes, so it does not distinguish node 7. Flexibility usage is larger
at node 9. **The association stands; the mechanism does not.** 2030 is not over-represented among node 7's at-rating
periods (8 of 23, against its one-third share), so a binding rating alone does not explain the 2030 concentration of the
PF residual tail. The period-by-period cross-check between the largest PF-residual entries and the at-rating periods
remains the open, cheap next step.

## 8. Questions for the Expert

1. **Oracle:** accept C by the rule, or adopt D on its convergence advantage? If D, how should the storage condition be
   restated, given no arm has ended below 0.78?
2. **What must the oracle optimise** — earliest certification, or quietest terminal objective? C and D disagree, and the
   answer decides question 1.
3. **Storage binds in every arm at ~0.8.** Is a further lever wanted (larger ρ_ess steps, a second lift, an
   ESS-specific tolerance), or is ~0.8 accepted for the oracle?
4. **Cost:** with candidate gaps of order 32.87 and configuration effects of order 10⁴–10⁵, do you want the zero-solve
   component decomposition (§3) before the oracle's cost becomes the Step 5 baseline? And is the rule "never compare
   costs across configurations" adopted explicitly?
5. **Step 3.6:** integrate, run the bitwise preflight and the 10-cycle re-measurement now?
6. **Case file and closure:** once the oracle is fixed, write its configuration into `data/SRP1/SRP1_params.json`, run
   Step 3.5 (polish gap) and close Step 3?

## 9. Evidence index

| item | artifact | commit |
|---|---|---|
| prior handoff | `P5_15_ADDENDUM20_EXPERT_REPORT.md` | `a6ae3236` |
| spec v10; record update | `P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json`; `REVISION_CONTEXT.md` | `e2fd8e60`, `fd25503b` |
| two-phase ESS policy; zero-solve checks | `admm_parameters.py`, `shared_resources_planning.py`, `p515_s39_zero_solve_checks.py`, `P515S39/zero_solve_checks/` | `6d628d7e`, `8cdd0978` |
| D-policy replay (v9 arms, then arm C) | `p515_s39_d_policy_replay.py`, `P515S39/d_policy_replay_on_s38/`, `.../d_policy_replay_on_s39_C/` | `0f9918e8`, `3ee7bb37`, `4c0def22` |
| arms, evaluator, preflights (incl. re-run) | `p515_g_g1_g4_admm_gates.py`, `p515_s39_evaluate.py`, `p515_s39_preflight.py`, `P515S39/preflight_{C,D}{,_v2}/`, `WORKER_REPORT_S39_PREP.md`, `P5_15_S39_PREFLIGHT_NOTE.md` | `cf480653`, `054eed0c`, `7f660765`, `c231ddff` |
| arm C; pre-launch note for D | `P515S39_C_run/`, `P5_15_S39_D_PRELAUNCH_NOTE.md` | `4c0def22` |
| arm D; both evaluations; stage report | `P515S39_D_run/`, `s39_evaluation.json` (both), `P5_15_S39_ORACLE_REPORT.md` | `2d786718` |
| node 7 (v9 arms, then C and D) | `p515_s39_node7_interface.py`, `P515S39/node7_interface/`, `WORKER_REPORT_S39_NODE7.md` | `4c92adfa`, `3ff09607`, `2d786718` |
| Step 3.6 clone replacement | worktree, detached at `fd25503b`; `WORKER_REPORT_S36_CLONE_CAPTURE.md` | `0c29f49d`, `b2a9ecae`, `816090b0`, `bff03b7e` |
| run 1 reference | `P515S35_REF_run/g_baseline.json` (sha256 `8c72a156…`) | earlier |

Every evaluator, replay, check and analysis above ran with `SolveProfileGuard` armed and verified: zero solves.
`pf_entry_stride_*.jsonl`, `esso_capture/` and `results/` are hash-recorded, not committed; everything else carries
per-arm sha256 manifests.
