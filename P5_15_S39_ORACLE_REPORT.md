# P5.15 Addendum 21 — arms C and D: both certify under the new bar; C is the oracle by the spec's rule

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 21. Frozen spec:
`data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json` (`e2fd8e60`), written before any code or run.

- **Instance:** C\* (0.96875 MVA / 3.875 MWh per node 5/7/9, invested 2025).
- **Certification bar (Addendum 21):** all three channels inside their Boyd tolerances, every local solve successful,
  for **10 consecutive cycles**, within cap 300. Terminal ratios, rule ten, clamp flags and the cost comparison are
  **reported, not gated**.
- **Cost bar (rule nine, redefined):** max objective step over each run's last 10 cycles; run 1 contributes 256.26.
- **Evaluations:** `p515_s39_evaluate.py`, zero solves, guard armed, run with both arms present.

## 1. Certification verdict per arm

| arm | configuration | certified | certification cycle | PF / V / ESS first pass | storage terminal ratio |
|---|---|---|---|---|---|
| **C** `s39_C` | τ = 0, PF balancing live, ρ_ess 0.01 fixed and exempt | **yes** | **191** | 130 / 48 / 131 | 0.809 |
| **D** `s39_D` | C plus the two-phase ESS schedule | **yes** | **139** | 125 / 48 / **75** | 0.786 |
| run 1 (reference) | τ = 1, ESS balanced, backstop 60 | yes, under Addendum 16 | 477 (3-cycle rule) | 226 / 45 / 475 | 0.988 |

**Adoption, by the spec's rule: C is the production oracle.** D certifies, and certifies 52 cycles sooner, but spec v10
makes D the oracle only if it certifies **with storage terminal ratio < 0.5**. D ends at 0.786, so the rule falls back
to C. §5 sets out the case for reading this differently; the decision is the Author's and the Expert's.

## 2. Predictions (recorded in spec v10 before either run)

| arm | prediction | outcome |
|---|---|---|
| C | PF first pass ≤ 131 | **met** — 130 |
| C | certification by ~145 (operationalized ≤ 150) | **missed** — 191 |
| D | certifies under the bar | **met** — cycle 139 |
| D | storage terminal ratio < 0.5 at certification | **missed** — 0.786 |

**Why C missed its cycle prediction: the storage channel, not PF.** V passed from cycle 48 and PF from 130, but the
storage primal ratio hovered at 0.93–1.20 and repeatedly broke the 10-cycle streak; the longest run before certification
was 5 cycles (131–135). The 10-cycle bar is what exposed this — under the old 3-cycle rule C would have "stopped" at
cycle 133.

## 3. What the two-phase storage schedule did (arm D)

| | C | D |
|---|---|---|
| ρ_ess | 0.01 for the whole run (exempt) | 0.01 → **0.015 (cycle 32)** → **0.0225 (cycle 33)**, frozen from cycle 43 |
| ESS exemption | never lifted | **lifted at cycle 31**, exactly as the pre-launch replay predicted |
| ESS channel first pass | 131 | **75** |
| storage primal ratio, last 12 cycles | 1.85, 1.06, 0.92, 0.87, 0.84, 0.86, 0.85, 0.82, 0.81, 0.80, 0.81, 0.81 | 0.86, 1.06, 0.88, 0.97, 0.91, 0.81, 0.77, 0.77, 0.86, 0.85, 0.81, 0.79 |
| certification cycle | 191 | **139** |
| wall clock | 6,958 s (36.4 s/cycle) | 4,893 s (35.2 s/cycle) |

- **The policy worked as designed and as replayed.** The lift cycle (31) and the first action (ρ_ess 0.01 → 0.015 at
  cycle 32) match the zero-solve replay on C's own trajectory, recorded in `P5_15_S39_D_PRELAUNCH_NOTE.md` before D ran.
- **The ratchet risk did not materialise.** The replay's open-loop cascade (28 further firings, ρ_ess → 1.3e3) did not
  occur in closed loop: two increases, then the unchanged-streak freeze at cycle 43. ρ_ess never exceeded 0.0225, and no
  channel reached a clamp.
- **It bought 52 cycles**, mainly by bringing the storage channel inside tolerance at cycle 75 instead of 131.
- **It did not settle the storage channel.** Both arms end at 0.79–0.81, and D's storage ratio still crosses 1
  occasionally near the end. Raising ρ_ess 2.25× damped the hovering; it did not remove it.

## 4. Reported, not gated

| | C | D |
|---|---|---|
| terminal ratios V / PF / ESS | 0.079 / 0.325 / 0.809 | 0.186 / 0.486 / 0.786 |
| rule ten (objective step / tolerance) | **0.0073** | **0.0481** |
| clamp ever reached (V / PF / ESS) | no / no / no | no / no / no |
| ρ_pf | 0.198 → 0.132 at cycle 2, frozen from 12 | same: 0.198 → 0.132 at cycle 2, frozen from 12 |
| cost | 650,981,423 | 650,966,975 |
| difference to run 1 | −57,743, bar 734 → **outside** | −72,191, bar 8,155 → **outside** |
| max objective step, last 10 cycles | 478 | **7,899** |
| local-solve failures | 0 | 0 |
| network failures (all recovered) | 61 (57 / 4) | 38 (37 / 1) |
| TSO instability trigger | not triggered | not triggered |
| EFC/day at certification | 1.154 | 1.134 |

Two things deserve the Expert's attention:
- **Both arms land outside the redefined cost bar**, C by 79× and D by 8.9×. Neither is a claim of a cheaper solution;
  §5 discusses what it can and cannot mean.
- **D's terminal objective is far noisier than C's.** Its largest step in the last 10 cycles is 7,899 against C's 478,
  and its rule-ten ratio is 6.6× C's. D satisfies the residual bar while its objective is still moving; C is the
  quieter run.

## 5. The adoption question

**By the letter of spec v10, C is the oracle**, because D missed the storage condition. Three observations bear on
whether that is the right reading. They are for the Expert; I have not acted on them.

1. **D dominates C on every convergence measure** — certification 52 cycles sooner, storage inside tolerance 56 cycles
   sooner, a marginally lower storage terminal ratio (0.786 vs 0.809), fewer network failures, and the same τ = 0 and
   PF behaviour. Its advantage is entirely the storage lever.
2. **The < 0.5 condition may be the same artefact criterion (c) was.** Addendum 21 withdrew "storage terminal ratio
   < 0.9" because the channel that closes a stop ends near 0.95 by construction. Under the 10-cycle bar the closing
   channel has a longer tail to decay, but the storage channel is still the binding one in both arms and still ends in
   the 0.78–0.81 band. No arm has ever come close to 0.5. The condition may be measuring the same ordering property.
3. **Against D: it is the less settled run by the objective measures** (rule ten 0.048 vs 0.007; last-10 max step 7,899
   vs 478). If the oracle's purpose is a stable cost for Step 5 comparisons, C's quieter terminal behaviour is an
   argument in its favour, independent of cycle count.

## 6. Cost against run 1

Both arms are now certified under a **stricter** bar than run 1 (10 consecutive cycles against 3), and both land below
run 1 by 58k and 72k — 0.009% and 0.011% of system cost — against bars of 734 and 8,155.

- **What this does not license.** The bar is local, and valid only between settled runs. Run 1's own storage channel
  terminated at 0.988, stopped rather than settled, so the reference side of the comparison is the weaker one. The
  arms' storage channels end at 0.79–0.81, better but still not settled. No claim of a different or lower-cost fixed
  point is made.
- **What it does suggest.** Three independent configurations (A, C, D) now certify 58k–72k below run 1, consistently in
  the same direction. That pattern is worth an explanation before the oracle's cost is used as a Step 5 baseline.

## 7. Node 7 interface (Addendum 21 item 3, extended to C and D)

The zero-solve look reproduces on the new arms exactly what it found on the v9 arms:

| node | rating | max utilization | periods at rating (of 288) | of which 2030 |
|---|---|---|---|---|
| 5 | 200 MVA | 0.510 | 0 | 0 |
| **7** | **100 MVA** | **1.000** | **23** | 8 |
| 9 | 150 MVA | 0.672 | 0 | 0 |

Identical in C and D to four significant figures. Node 7's interface branch is the only one that binds; the shared
storage at node 7 is also at its rating (utilization 1.00003). Voltage sits within about 3e-6 pu of its bound at all
three nodes, so it does not distinguish node 7. The per-entry PF capture identity holds in both arms (max relative
error 4.3e-16). The association between the binding interface and the PF residual tail is unchanged; the causal
mechanism remains unestablished, and the period-by-period cross-check is still open.

## 8. Process notes

- **Preflights re-run at committed HEAD.** The preparation Worker ran the first C/D preflights against uncommitted code
  (18:36–18:38; commits 19:44–19:46). Those are retained but not cited; both were re-run at HEAD `7f660765` under their
  own ids. Details and the corrected bitwise reading: `P5_15_S39_PREFLIGHT_NOTE.md`.
- **My bitwise premise was wrong.** PF balancing genuinely fires at cycle 2 in C and D (τ = 0 makes the PF dual ratio
  5.7× the primal), so they cannot reproduce arm A's cycles 1–2 exactly. Of 192 fields, 2–3 differ per cycle, all of
  them ρ_pf or balancing bookkeeping; everything numerical matches. The Worker reported this instead of forcing a pass.
- **`balancing_exempt_ess` reads False for arm D while the channel is exempt.** Accepted as designed: the action label
  and the `ess_exempt_until_state_s39_D.jsonl` sidecar carry the true state, and the evaluator uses those.
- **Step 3.6 clone replacement is complete but not integrated** (§9).

## 9. Step 3.6 status

The TSO whole-model clone replacement is finished and committed in an isolated worktree (detached HEAD at `fd25503b`,
commits `0c29f49d`, `b2a9ecae`, `816090b0`, `bff03b7e`). Rebuild-from-capture is equivalent to a legacy clone with 0
differences across every variable, bound, parameter, suffix and thousands of constraint expressions on real TSO and DSO
blocks; clones per cycle drop from 12 to 0 (1 only when a snapshot must be written); capture is 3–4× faster per block;
all 7 preserved fixtures still unpickle. **Not integrated:** the plan is to cherry-pick the four commits, run the
2-cycle bitwise preflight (legacy clone vs capture), then the 10-cycle re-measurement, then decide on persistent
workers. Addendum 21 sequences this after C and D; I have stopped for review first.

## 10. Questions for review

1. **Oracle:** accept C by the spec's rule, or adopt D on its convergence advantage (§5)? If D, the < 0.5 condition
   needs restating — no arm has approached it.
2. **The storage channel is now the binding one in every arm**, ending at 0.78–0.81 whatever the lever. Is a further
   storage lever wanted (a larger ρ_ess step, a second lift, or an ESS-specific tolerance), or is hovering at ~0.8
   acceptable for the oracle?
3. **Cost:** three configurations certify 58k–72k below run 1. Explain before Step 5 uses the oracle's cost, or accept
   as stopping-point variation?
4. **Step 3.6:** proceed with integration, bitwise preflight and 10-cycle re-measurement now?
5. **Case file:** once the oracle is fixed, write its configuration (τ = 0, ρ_pf 0.198, ρ_ess policy, freeze 10/200,
   10-consecutive bar, cap) into `data/SRP1/SRP1_params.json`, then run Step 3.5 and close Step 3?

## Evidence

Arm runs `data/SRP1/Results/P515S39_{C,D}_run/` with `s39_evaluation.json`, launch logs, exit codes and per-arm sha256
manifests; `P515S39/{zero_solve_checks,d_policy_replay_on_s38,d_policy_replay_on_s39_C,preflight_C_v2,preflight_D_v2,
node7_interface}/`; `P5_15_S39_PREFLIGHT_NOTE.md`, `P5_15_S39_D_PRELAUNCH_NOTE.md`, `WORKER_REPORT_S39_PREP.md`,
`WORKER_REPORT_S39_NODE7.md`. `pf_entry_stride_*.jsonl`, `esso_capture/` and `results/` are hash-recorded, not
committed.
