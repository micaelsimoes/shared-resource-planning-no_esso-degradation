# P5.15 Addendum 24 item 1 — exact-fix re-run with the fixed helper: prediction mostly falsified (5/12)

Planner note, 2026-09-18. Spec v13 `data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json` (`cbdedbb4`),
prediction recorded before the run. Harness `p515_s42_exact_fix_rerun.py`; helper fix `2051309c` (regression check exact,
1,728/1,728); run at `76e486f9`.

**Reproduction.** Arm D reproduced bitwise (cycle 139, cost 650,966,975.2943751) — the fourth full reproduction.
Certified TSO/DSO models persisted: `certified_models.pkl`, 165,541,129 bytes, sha256 `4e3d8bfc…a797` (hash-recorded,
not committed), so later polish variants need no reproduction.

**Exact-fix (midpoint) polish with the FIXED helper, 48 blocks:**

| blocks | predicted | observed |
|---|---|---|
| TSO spring/summer ×6 (node 7 midpoint > 100 MVA) | local infeasibility, violation ≤ 5e-4 MVA | **all 6 solve** (violation 1.5e-10 to 6.8e-7 MVA) — **falsified** |
| TSO autumn/winter ×6 | solve | **5 solve; TSO 2025 Autumn fails** — local infeasibility after tier-2, violation 1.3e-6 pu (1.3e-4 MVA), in `node_balance_q` rows — near-feasible |
| DSO-2025 ×5 (DSO5 Spring; DSO7 Spring, Autumn; DSO9 Spring, Autumn) | no prediction | the same five fail again (numerical, as in v11) |

Score 5/12. The Δ gate is not evaluated (not all blocks solved); it was never this test's purpose.

**What this establishes.**
1. **The helper fix is confirmed as the cause of the void v11 run:** 11 of 12 TSO blocks now solve, against 0 of 12.
2. **The rating-midpoint check measured the wrong thing (Addendum 24's re-examination).** The 100 MVA figure is the
   **DSO** node-7 interface branch rating (`Network.get_interface_branch_rating`, `network.py:78-88`, which errors on a
   transmission network); it is not a row of any TSO block. It could never explain TSO failures. In the DSO7 blocks,
   where it does apply, the midpoint excess (≤ 5e-4 MVA on 100 MVA ≈ 5e-6 pu) is inside IPOPT's tolerances
   (`tol` 1e-5, `acceptable_tol` 1e-4 in the case files and the logs), so those blocks solve.
3. **Exact fixing at D's point is not infeasible where node 7 binds.** The conditional manuscript clause in Addendum 24
   ("at the baseline point exact fixing is infeasible in the blocks where the node 7 interface is at its rating") is
   **not supported and must not be used.** The principle sentence stands on its own: the hull test is well-posed at
   active constraints by construction.
4. **What does fail at the exact-fix point:** one TSO block, marginally (1.3e-6 pu, reactive node balance), and five
   DSO-2025 blocks numerically. The hull polish solved all 48 at the same point.
