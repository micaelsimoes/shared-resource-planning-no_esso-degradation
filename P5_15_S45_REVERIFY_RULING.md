# P5.15 Addendum 27 item 1: case-file AA-on re-verification at C*. Strict gate FAIL on one reporting field; ruled PASS

**Planner ruling, 2026-09-19.**
- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 27 item 1.
- **Frozen spec:** v15 `data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`, sections
  `configuration.reverification_gate` and `predictions_recorded_in_advance.reverification`.

## What was run
- **Evaluation:** C\* (0.96875 MVA / 3.875 MWh at nodes 5, 7 and 9, investment year 2025; candidate key `578636daa6d6360d`).
- **Harness and configuration:** the campaign harness, with the configuration taken from the case file alone. AA-on keep_memory has been in
  `data/SRP1/SRP1_params.json` since `b5629311`, and there were no overrides.
- **Cost file:** the corrected `SRP1_ESS.xlsx` (`e17bd588…`, `2cada62b`) was on disk.
- **Campaign spec:** `campaign_spec_s45_reverify_aa_c_star_4a214b99.json` (`25470496`).
- **Eval dir:** `data/SRP1/Results/P515S45/reverify_aa_c_star/evals/3e741dac72c9e1bc_c_star_aa_case_file`.
- **Reference:** the committed AA keep_memory evaluation at C\*,
  `data/SRP1/Results/P515S44/campaign_s44_aa_variant/evals/837fc982565dbba3_c_star_aa_keep_memory`. It ran with the AA
  override and the previous cost file (`14581474…`).
- **Comparator:** `p515_s45_reverify_compare.py` (`11f5ed1e`), with its self-tests committed. Output is in
  `data/SRP1/Results/P515S45/reverify_aa_c_star_compare/compare.json`.

## Result
| item | result |
|---|---|
| certified; cycles | certified at **107** (reference 107) |
| certified cost (`gross_operational_cost`, settlement excluded) | **650,982,939.9389359**, bit-identical to the reference |
| effective AA dict in the child | {enabled: true, memory 5, regularization 1e-10, keep_memory}; equals the reference's applied override |
| artifact files, sidecars, `aa_per_cycle.jsonl` | 0 genuine diffs |
| `per_cycle_record.jsonl`, `g_s39_D.json` trajectory | 75 + 75 diffs, **all in one field: `terminal_salvage_value`** |
| provenance (identity, configuration bookkeeping, wall time, RSS) | 42, non-gating |
| comparator guard | 0 solves |

The strict comparator therefore reports **GATE FAIL**, with `zero_genuine_diffs = False`.

## Analysis of the 150 diffs
The analysis was a zero-solve read of the two `per_cycle_record.jsonl` files.
- **Cycles affected:** of the 107 cycles, 32 have `terminal_salvage_value` exactly 0 in both runs. The other 75 are nonzero in both.
- **Size of the change:** in all 75, candidate/reference = **1.25** (range 1.25 to 1.2500000000000007). No other field differs in any row.
- **Magnitude:** at most **8.4e-35 EUR** (reference 6.7e-35). The net recourse `recourse` is bit-identical in every cycle,
  since gross − salvage rounds to the same double.
- **Mechanism:** salvage is priced at the workbook's expected energy unit cost (`shared_energy_storage_data.py:863-867`,
  per W2, `9e623dd3`). The corrected file multiplies every energy cost by exactly 1.25 and leaves power unchanged (cell-level diff, W2).
  - For a 2025 investment the remaining-life fraction at the horizon is 0, so salvage is 0 up to round-off in the state of health (SoH).
  - The round-off is what the ×1.25 scales.
  - Salvage is not in the storage operator's (ESSO) objective (`shared_energy_storage_data.py:795`), so it cannot feed back into the iterate. The 0 diffs in every
    other field confirm this.

## Ruling
**PASS.** The oracle reproduces bitwise: every trajectory field, every artifact, the AA record, the certification cycle and
Q(x). The one differing field is the salvage reporting expression. By the author's ruling (Addendum 27 item 3) salvage is
reported and excluded from F(x). Here it scales exactly with the corrected cost file, which is the change the author made.
The spec v15 prediction ("bitwise reproduction; the cost file does not enter Q") holds for Q. It did not anticipate the
salvage field, and I record that as a miss in the prediction's scope. The strict comparator's FAIL stands in its
output and is not overridden there.

**Consequences recorded for Phase A:**
1. **Salvage at later investment years.** For 2030 and 2035 investments salvage is materially nonzero: up to 51,565 and 95,457 EUR per MWh of residual energy
   with the corrected file (W2). F(x) uses `gross_operational_cost`, which excludes salvage, so the objective is unaffected.
   Salvage is reported per evaluation from `recourse_components.terminal_salvage_value`.
2. **The harness's bar is mislabelled; to be fixed before A1.**
   - `_max_step_last_n` (`p515_s44_campaign_harness.py:927-933`) labels the bar "|gross cost step|".
   - It actually reads `objective_change_abs`, which production computes on the **net** recourse
     (`shared_resources_planning.py:2849`: `abs(recourse - previous_recourse)`, with `recourse = net_operational_recourse`
     at `:2814`).
   - At a 2025 investment the two agree, since salvage is ~1e-35. On the 2030/2035 ladders the bar would include salvage
     movement.
   - `gross_operational_cost` is recorded in every per-cycle row. The bar will be recomputed from its cycle-to-cycle
     steps in the pre-A1 harness task, and every committed bar will be checked to be unchanged by it.
3. **AA saving and storage size.** Adopting AA-on is recorded together with the finding that its saving falls with storage size: 23 / 17 / 21 / 4 % at
   C\* / paper plan / node-7-empty / 2×C\*.

## Evidence
- Case file, loader and harness: `b5629311`.
- Re-verification spec: `25470496`.
- Comparator: `11f5ed1e`.
- This ruling commit also holds the run, its campaign manifest and the comparison output.
  - `esso_capture/`, `results/`, `esso_models_s39_D.pkl` and the per-entry strides are recorded by hash in
    `campaign_manifest_sha256.json` and `child_manifest_sha256.json`, and are not committed. This follows the S44 convention.
