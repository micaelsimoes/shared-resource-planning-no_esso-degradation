# P5.15 — Gate G4 (determinism): PASS — bitwise identical to G1

**Planner report, 2026-09-14.** Authority: `PLANNER_BRIEF_2026-09-13.md` Step 1 gate G4 ("G1 run twice,
identical"). Scope: the control arm only (G2 is designed to differ and is not in scope). Run: `g4b` at
`c2361c26`, one background run through the tool, stderr captured, lock held, **Python exit code 0**. G1 ran
at `4b56c835`; production code is identical between the two commits, only the harness changed
(results-directory redirection, fresh roots, print-based failure parser, pre-check scope).

## Result

| quantity | G1 | G4 | difference |
|---|---|---|---|
| cycles / converged at | 72 / 72 | 72 / 72 | none |
| recourse (`gross_operational_cost`) | 817,618,798.07215 | 817,618,798.07215 | **0 (bitwise)** |
| solves | 3,735 | 3,735 | none |
| local failures / failed cycles | 6 / {7, 39, 43, 46, 48, 49} | 6 / same | none |
| per-cycle recourse, step, primal and dual residuals (V, PF, ESS), 72 cycles | — | — | **0 entries differ** |
| ESSO `es_soh_per_unit_cumul`, `es_D_per_unit`, `es_avg_ch_dch_per_unit`, all nodes | — | — | **0 entries differ** |
| measured detector per ESSO solve | 219 solves | 219 solves | **0 of 219 differ** |
| shared `FrozenSMOPF` modified / new files | — | [] / [] | guard held |

**Verdict: G4 passes.** The C\* control arm is bitwise reproducible under the adopted solver policy, the
reformulated ESSO and remedy (h). It also shows that the Addendum-6 capture and the G2-prep harness changes do
not perturb the solve path.

## Note on artifact hashes

The cycle-7 comparator pickles written by G2, G3-full and G4 into their own roots have different sha256 values
from one another even where the numerical run is bitwise identical (G4 `379215c3…` vs G1's `7e5aa39d…` for the
DSO comparator). Pickled Pyomo models are not byte-stable; **a pickle hash is not a determinism test** and must
not be used as one. Determinism here is established on the numerical outputs above.

## Evidence (sha256, first 16; full manifest `P515G4/evidence_manifest_sha256.json`)

| artifact | hash |
|---|---|
| `P515G4/g_control_rep2.json` | `19b08ffa007c87b6` |
| `P515G4/stdout_control_rep2.log` | `efc36ae4b8928124` |
| `P515G4/leak_classification_control_rep2.jsonl` | `c64e01f01cbf2ab0` |
| `P515G4/network_failures_control_rep2.jsonl` | `966ea0c816f24eae` |
| `P515G4/esso_models_control_rep2.pkl` | `2d4e38875f1eb54a` |
| `P515G4_launch.log` | `41b04527c337b1b8` |
| `P515G4/esso_capture/` (hash-recorded, not committed) | see manifest |
