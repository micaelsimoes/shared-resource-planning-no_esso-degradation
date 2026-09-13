# WORKER_REPORT_DIAGA — Diag-A: does the G1 reference trajectory itself embed a leak?

Stage: P5.15 Diag-A. ZERO-SOLVE diagnostic. No fix proposed. No production code touched.

## Integrity check verdict: PRE-REFORMULATION — the pickle is a valid target for this diagnostic

`data/SRP1/Results/P514N/esso_models_control.pkl` unpickled successfully with no error, using
only production-module imports for symbol resolution (`shared_resources_planning`,
`shared_energy_storage_data`, `model_construction_helpers`) — no repair was needed at unpickle
time.

Every one of the three node models (5, 7, 9) exposes, simultaneously:

- `es_degradation_per_unit`, `es_degradation_per_unit_cumul`, `es_soh_per_unit`,
  `es_soh_per_unit_cumul` (the retired per-unit degradation/SoH chain)
- `energy_storage_complementarity`, `energy_storage_normalization` (the retired relaxed
  complementarity rows)
- `es_pch_hat_per_unit`, `es_pdch_hat_per_unit`, `es_pch_hat_agg`, `es_pdch_hat_agg`,
  `slack_es_ch_comp_per_unit` (the retired normalized/aggregate/slack variables)

None of the three expose `es_D_per_unit` (the reformulated model's log-domain replacement).

**Timing corroborates this independently of the attribute check.** The pickle's mtime is
`2026-09-13T11:35:16.960568+00:00` (`2026-09-13 12:35:16` local, matching its sibling
`n1_control.json`/`control_console.log`, both stamped identically). The reformulation commit
`b03c9b14` ("Step 1: ESSO reformulation …", which deletes exactly these attributes) is dated
`2026-09-13 16:30:51 +0100` (`15:30:51` UTC) — **just under four hours after** the pickle was
written. The pickle cannot be a product of a commit that did not yet exist.

Node 5's stored `es_soh_per_unit_cumul` chain is `1.0 → 0.838691639504732 → 0.7284046262724772
→ 0.6247952809513038`, an exact match (to the quoted 4-decimal precision) for the G1 reference
`1.0 → 0.8387 → 0.7284 → 0.6248`. This is the same run cited in the brief.

**Verdict: genuinely PRE-reformulation. The diagnostic proceeds.**

## Guard

`p513_solve_profile_guard.SolveProfileGuard` armed in blocking form, `permitted=()`, for the
entire run (unpickling + all measurement code). Observed profile: `{'permitted_solve': 0,
'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`. `guard.verify(expected_solves=0)`
returned zero failures. **No solver was reached anywhere in this run.**

## Comparison table

| quantity | OLD control (this pickle) | NEW reformulated (eps=1e-3 fixture) |
|---|---|---|
| max min(pch,pdch)/s_max | **0.007729** (node 5), 0.007729 (node 7), 0.007727 (node 9) | 4.6319e-4 |
| spurious throughput fraction | **1.386%** (node 5), 1.391% (node 7), 1.340% (node 9) | 0.9175% |

The OLD control's `min(pch,pdch)/s_max` is **~16.7x** the NEW reformulated value (not merely
"comparable" — larger), and its spurious-throughput fraction is **~1.5x** the NEW value. This
supersedes the task's own order-of-magnitude estimate ("~1e-3" from the relaxed-tolerance
argument) — the measured value is ~7.7x that estimate.

**Convention caveat (per the reporting-conventions rule):** the "spurious throughput fraction"
above uses an **unweighted** sum of `pch`, `pdch`, `min(pch,pdch)` over the 96 periods x 3 active
cohort-years actually stored in the pickle (288 samples total), with no `num_days/365`
day-count weighting and no efficiency weighting — unlike `es_avg_ch_dch_per_unit` itself, which
uses `(num_days/365)*(eff_ch*pch*dt + pdch*dt/eff_dch)`. `num_days` is a property of
`SharedEnergyStorageData.days` (the higher-level Python object), not of the pickled Pyomo model,
and could not be recovered from the pickle alone without importing current-code/data constants,
which was avoided per the task's instruction not to import current values. The NEW fixture's
0.9175% convention is not independently re-derived here (it was supplied by the Planner); the
two numbers are reported side by side, not asserted to share an identical weighting convention.
Within one model (OLD vs OLD across nodes, or the ratio measurement, which is `s_max`-relative
and convention-free), the comparison is apples-to-apples.

## Per-measurement detail

**1/2 — max min(pch,pdch), both pairs, and the ratio.** Physical pair
(`es_pch_per_unit`/`es_pdch_per_unit`) feeds `es_avg_ch_dch_per_unit` directly
(`energy_storage_charging_discharging` row) — **this is the pair that feeds degradation
accounting.** The normalized pair (`es_pch_hat_per_unit`/`es_pdch_hat_per_unit`) exists only to
feed the (now-retired) `energy_storage_complementarity`/aggregate rows. Both pairs agree exactly
once divided by `s_max` (`es_s_rated_per_unit[y_inv,y]` = 0.96875 p.u. for every active cohort at
every node here), confirming the normalization row (`pch - s_max*pch_hat == 0`) held to full
solver precision in the stored solution — a useful internal-consistency check, not itself part
of the requested measurement.

Physical max min(pch,pdch) ≈ 0.007487 p.u. (node 5, cohort (0,0)); ratio to `s_max` ≈ 0.007729,
essentially flat across the three active cohort-years and three nodes (0.00772–0.00773).

**3 — throughput / spurious fraction.** Reported above and in the JSON per-node breakdown
(`data/SRP1/Results/P5151/p5151_diagA_old_control_leak.json`, `measurements.<node>.summary`).
Total unweighted throughput and its spurious component are also broken out per cohort-year.

**4 — `es_avg_ch_dch_per_unit`: stored vs leak-removed.** Stored values reported (node 5:
8.6214, 6.9098, 7.5197 for cohorts (0,0),(0,1),(0,2); similar for 7, 9 — full table in the
JSON). **The leak-free recomputation was NOT performed.** `eff_ch`, `eff_dch`, `dt`, `cl_eff` —
the constants the `energy_storage_charging_discharging` row actually uses
(`avg_ch_dch += (num_days/365)*(eff_ch*pch*dt + pdch*dt/eff_dch)`) — are Python floats baked into
the constraint coefficients at model-construction time inside `SharedEnergyStorageData`; they are
**not** stored as retrievable `Var`/`Param` attributes on the pickled Pyomo model (`hasattr`
checks for all four returned `False` on every node). Recovering them would require importing
current-code/data constants, which the task explicitly forbids ("the current values may differ").
**Reported as NOT RECOVERABLE from the pickle alone**, not approximated.

**5 — SoH trajectory, stored vs leak-free-implied.** Stored `es_soh_per_unit_cumul` and
`es_degradation_per_unit` chains reported per node (node 5 matches the G1 reference exactly, see
above). **The leak-free-implied trajectory was NOT computed**, for the same reason as (4): the
degradation row `es_degradation_per_unit * (2*cl_eff*es_e_rated_per_unit) ==
es_avg_ch_dch_per_unit` needs `cl_eff`, which is not recoverable from the pickle, and the task
forbids importing it from current code.

**6 — complementarity slack.** `slack_es_ch_comp_per_unit` is present at all three nodes; its
maximum over all 288 evaluated (cohort-year, day, period) cells is **0.0 exactly** at every node
— **the slack is never active in this run.** The relaxed row `pch_hat*pdch_hat <= slack + 1e-4`
was therefore satisfied by the raw product alone, with zero slack cost. This is consistent with,
not contradictory to, the large `min(pch,pdch)` ratio measured in (1)/(2): the relaxed row bounds
the **product** `pch_hat*pdch_hat`, not `min(pch_hat,pdch_hat)`; a `min` of ~0.0077 is compatible
with a product ≤1e-4 provided the paired (larger) value at that same period stays ≤ ~0.013 — i.e.
the row's design (product-based, not min-based) is exactly what let both directional variables
sit simultaneously in the ~0.0077–0.013 p.u. range at the binding periods without ever touching
the slack.

## Conclusion

**(A) — the old reference is comparably (in fact more) leaky, so G1's 1e-6 SoH-trajectory gate
compares two solver artifacts and is invalid as specified.**

The old control's physical directional dispatch — the exact pair that feeds
`es_avg_ch_dch_per_unit` and hence the SoH chain G1 checks to 1e-6 absolute — carries
`min(pch,pdch)/s_max` up to 0.00773 and a measured spurious-throughput fraction up to 1.39%,
both larger than the corresponding figures already used to call the reformulated model's
4.63e-4 / 0.9175% a "leak." The reference trajectory G1 demands reproduction of was itself
produced under a feasible set that permitted materially more simultaneous charge/discharge than
the reformulated model now exhibits. A 1e-6-absolute SoH match against that reference therefore
does not test whether the new ESSO is "as clean as the old one" — the old one was not clean by
this same yardstick, and was leakier on both measures reported here. (This report does not
attempt to convert the throughput-fraction difference into a precise predicted SoH gap, since the
constants needed to do so — item 4/5 above — are not recoverable from the pickle without
violating the task's constraint against importing current-code values; the qualitative
direction, not a quantified margin, is what this diagnostic supports.)

## Flags / not measured

- Items 4 and 5's leak-free recomputations: not performed, `eff_ch`/`eff_dch`/`dt`/`cl_eff` not
  recoverable from the pickle (see above); not imported from current code per instruction.
- The spurious-throughput-fraction comparison (3) uses an unweighted convention on this side
  because `num_days` is not stored on the pickled model; the NEW-side 0.9175% figure's own
  weighting convention was not re-derived here (supplied by the Planner) — see caveat above.
- No solve, no IPOPT invocation, no model construction that triggers `optimize` occurred at any
  point (guard-verified, zero permitted and zero blocked calls).
- No file under `data/` was modified. Only new files were written:
  `data/SRP1/Results/P5151/p5151_diagA_old_control_leak.json` (did not exist before this run).

## Evidence artifact

`data/SRP1/Results/P5151/p5151_diagA_old_control_leak.json` — full per-node, per-cohort-year
measurements, provenance (sha256/mtime of the pickle and its sibling artifacts), the guard
counters, and the integrity-check attribute inventory.
