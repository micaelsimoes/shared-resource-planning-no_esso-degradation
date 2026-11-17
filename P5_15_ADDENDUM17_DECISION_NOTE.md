# P5.15 Addendum 17 — the authorized route is blocked: decision note for the Author and the External Expert

**Planner note, 2026-09-17.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 17 and 18. **Verdict first: the
Addendum 17 gate was NOT run.** Three zero-solve findings show that steps 2–4 cannot be carried out as authorized.
Following the row-3′ precedent, the Planner stops and returns the design rather than improvising a substitute.
Nothing long-running has been started: not the ~5 h run-1 replay, not spec v7, not the dual-initialization code, not
the gate, not Step 3.6's implementation.

## 1. What was done

| item | result | commit |
|---|---|---|
| Addenda 16–17 recorded; §6 claims-to-avoid adopted verbatim | done | `db945601` |
| Step 4 warm-continuation validation | design only, not run | `7f1c75bf` |
| Step 1: data-availability audit and dual-direction comparison (a) | partial — see §2 | `5e2ba6f9` |
| Step 3.6: audit of the parallel path and cycle-time attribution | done — see §5 | `710af0de` |
| Run 1's terminal storage duals reconstructed exactly, per agent and per entry, all 477 cycles | **pass** — norms to 1e-14; ESSO per-entry duals to 3.3e-10 | `b63e4d94` |
| Cycle-0 node-balance dual capture (51 initialization solves); run-1 replay arm prepared (dry checklist passes, **not launched**) | done | `b8f129e0` |
| Calibration of the cycle-0 duals | see F1 | `b6384d99` |
| Identifiable-versus-split decomposition (H1/H2) | **inconclusive** — see F3 | `97507b98` |

## 2. The three findings that block steps 2–4

### F1 — Cycle-0 LMPs at the storage buses are structurally zero
The conversion is correct: validated on an independent fixture at a generator bus, dual/coefficient = 0.9999999952,
no sign flip, no hidden factor. The zero is real. At initialization the settlement weight is 0, so the signed interface
flexibility variable `interface_delta_p` is **unpriced**. Its own KKT condition forces the node-balance dual at buses 5,
7 and 9 to ≈0, independent of generation cost; the captured values are ~1e-9 of the market price.

**Consequence:** step 2's production input — "cycle-0 LMPs, deterministic per candidate" — carries no price
information, and diagnostic 1(c) is not computable this way. The earlier diagnostics report's claim that no solve
precedes cycle 1 was wrong (51 initialization solves run) and has been corrected in the record.

### F2 — The mapping test as authorized is ill-posed (independent review; code-verified)
- Prices determine only the **summed** storage KKT condition across the three agents:
  (w_b/σ)·LMP + κ_T + κ_D + κ_E/c_E = 0.
- **The split of the dual across TSO, DSO and ESSO is not identified.** It is free along the gradients of the storage
  constraints active at the schedule: day balance, saturated power limits, SoC bounds, and idle periods.
- Run 1 shows the split moving while the total does not. At cycle 1 the per-agent norms were exactly the 2:1:1
  mean-subtraction pattern (8.08 / 4.04 / 4.04e-3). By cycle 477 they were 6.27 / 5.34 / 2.15e-3. Over cycles 346–477
  the total norm was flat while the ESSO share rose 36 %.
- **"Reproduce run 1's terminal per-agent duals from its prices" can therefore be failed by a correct mapping and
  passed by a wrong one.**
- Two parts of the intended mapping are wrong in this code:
  - the **DSO reference-node dual is not a storage price** — the storage term cancels there, because the reference
    generator carries no cost and the settlement acts on pg_ref − pnet;
  - the **ESSO carries no price at all**. Only the TSO has a Q price.

### F3 — The slow component is not cleanly the identifiable one (H1/H2 inconclusive)
Decomposing run 1's reconstructed duals, per (node, year, day), into the price-identified component I (orthogonal to
the terminal active-constraint span) and the in-span split component J:

- **Neither settles by cycle 477** under the pre-registered rule: ‖I‖ is still changing 2.0 % per 50 cycles and ‖J‖
  2.6 %. The ESSO's own components drift most, 9–11 % per 50 cycles.
- The span is not degenerate: dimension 5–14 of 24 on every block, in both variants.
- **The identifiable share of the dual falls from 0.685 (cycle 1) to 0.520 (cycle 477).** That leans toward H2 — the slow
  part includes the unidentified split — without meeting the pre-registered H2 trigger.
- Gate 3's duals cannot be decomposed: its stride-5 capture does not cover every cycle.

**Consequence:** even a correct price-based dual initialization would address at most the identifiable part, which is a
shrinking share of a dual still drifting in both components at run 1's certified stop.

## 3. What remains sound

- **Run 1 is certified** (Boyd stop at 477; cost 651,039,166 settled). EFC/day ≥ 1.059 from below only, as recorded.
- **The exact dual reconstruction** is reusable evidence, and the run-1 replay arm is ready if a replay is authorized.
- **Step 3's ρ defaults** (0.0116 / 0.088 / 0.1125) have **not** been applied, because the gate they serve is not
  running.

## 4. Options — author decisions

1. **Price the interface during initialization, then retry.** Run the 51 initialization solves with the interface
   settlement active (weight 1, as in the ADMM path), so cycle-0 storage-bus LMPs carry price information. This changes
   the initialization problem, not the ADMM fixed point. Then re-test F1, and initialize **only the identifiable
   component**, with the per-agent split chosen explicitly (e.g. mean subtraction) and stated as a choice. F3 predicts
   the split drift will remain, so a gate may still fail on the storage channel.
2. **Replay run 1 (~5 h)** to obtain its terminal LMPs. This validates the **units and sign** of a revised mapping and the
   identifiable component, but cannot supply a production input (F1), and it is not needed unless option 1 is pursued.
3. **Accept run 1 as the certified oracle and move on.** Carry storage as the one-sided statement (§6), proceed to Step
   3.6 (parallelism shortens long certified runs), then Step 4 with the warm-continuation validation already designed.
   Addendum 18 sequences Step 3.6's preflight and gate **after** the Addendum 17 gate; with that gate not run, this
   ordering needs your confirmation.
4. **Over-relaxation** remains parked and is not recommended: it scales dual steps by at most ~1.5×, and F3 shows the
   drift is not a pure step-size problem.

**Planner's recommendation:** option 3 now, because it unblocks the programme with the certified evaluation in hand.
Option 1 can follow as a bounded experiment if you want initialization pursued; its first test is cheap (51
initialization solves plus the zero-solve F1 check).

## 5. Step 3.6 audit, relevant to option 3

- **Cycle time (run 1, 12 sampled cycles):** IPOPT takes a median **13.0 s** (DSO 12.2 s, TSO 0.7 s, ESSO 0.04 s) of a
  **33.7 s** cycle. The other **21.5 s (59.5 %) is overhead** — Pyomo model updates, NL write, `.sol` read, ADMM
  bookkeeping — stable across cycles.
- **Parallelizing the solves alone** gives **≤1.5× on 8 workers** (Amdahl limit 1.55×), far from the 39 s → 8–10 s
  target. That target needs most of the overhead to be per-block work the persistent workers can absorb, which the
  existing logs cannot establish. **Per-phase timing instrumentation** is therefore the first implementation item.
- **The existing parallel path is not reusable as is:**
  - it covers DSO only, split by node rather than by block;
  - it spawns processes per call and pickles whole models;
  - it sets no thread caps and drops the FrozenSMOPF callbacks;
  - IPOPT log paths can interleave;
  - `SolveProfileGuard` cannot see child-process solves.
- The P5.6-B crash was at a different layer (candidate evaluation) and was later withdrawn as non-reproducible.

## 6. Process incidents (no loss)

- **Concurrent commits.** Two Workers committed concurrently. One used a plain `git commit`, which swept the other's
  staged files into its commit (`44c3ff58`), then undid it with `git reset --soft HEAD~1`. Both sets of files are now
  committed separately and correctly (`b6384d99`, `97507b98`). The orphan commit is unreachable and harmless.
  **Rule clarified:** the explicit pathspec must be on the commit command itself (`git commit --only -- <paths>`), and
  Workers must never reset.
- **Corrected claim.** The earlier "no pre-cycle-1 solve exists" claim was corrected (see F1).

## 7. Questions

1. Which option in §4?
2. If option 3: may Step 3.6's preflight and gate proceed with the Addendum 17 gate not run?
3. If option 1: is changing the initialization problem (interface settlement active in the 51 initialization solves)
   acceptable as a method-only change?
