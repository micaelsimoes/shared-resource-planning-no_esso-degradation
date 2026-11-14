# Provenance note — `ladder_s1.json` is POST-repair as of 2026-09-13

`ladder_s1.json` in this directory was **overwritten on 2026-09-13T17:14:41Z** by a P5.15
verification run of the repaired `p514_l_capacity_ladder.py`. It is NOT the artifact cited by
`P5_14_L_CAPACITY_LADDER_REPORT.md`.

| | pre-repair (cited by the L report) | current file |
|---|---|---|
| node 7 | `infeasible` | `optimal` |
| `all_ok` | False, `failed=['7']` | True, `failed=[]` |
| solves | 52 | 51 |

**The pre-repair content is preserved in `rung_1.00.log` in this directory** (mtime
2026-09-12 23:59, untouched), which records node 7's full violated-component breakdown:
`rated_s_capacity_unit: 6.845e-05`, `rated_s_capacity: 2.282e-05`,
`energy_storage_operation_agg: 1e-08`, `energy_storage_normalization: 0.0`. The same figures
are quoted in the L report. Nothing was lost, but the JSON itself is not recoverable.

The difference between the two rows above is **not** attributable to the deletion of the
complementarity rows: the pre-repair failure was on the capacity rows, with the
complementarity family at exactly 0.0. See `P5_15_GATE_HOLD_REPORT.md` §4.

Cause of the overwrite: a Planner task instruction that said "run the harness" without naming
an output path. See the `CLAUDE.md` evidence rule "Never re-run a harness onto an artifact a
committed report cites."
