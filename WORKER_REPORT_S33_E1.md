# Worker Report — P5.15 Step 3.3(a) Addendum 14, E1 (zero-solve)

## Task received

Planner's E1 diagnostic (`PLANNER_BRIEF_2026-09-13.md` Addendum 14; `P5_15_S32_BOYD_GATE_REPORT.md`
§3 and §5 item 1): answer Q1 (interface V vs bounds), Q2 (ESSO objective scale vs TSO/DSO σ, D5
evidence) and Q3 (low-priority checks: `consensus_vars['ess']['tso']['prev']` vs the TSO proximal
centre; PF `‖y‖` units), using only artifacts already serialized under
`data/SRP1/Results/P515S32_run/` (plus static case data and `SRP1_params.json`). No production,
case-file or harness edit; no solve of any kind; new script only, armed
`SolveProfileGuard(permitted=())`; output to a new directory that must not already exist.

Instance: C\* (s = 0.96875 MVA, e = 3.875 MWh, year 2025, uniform across nodes 5/7/9), same as the
s32 gate. Objective convention: `gross_operational_cost` (unchanged by this task; no new objective
values are computed here beyond citing existing ones).

## Files inspected

- `P5_15_S32_BOYD_GATE_REPORT.md`, `PLANNER_BRIEF_2026-09-13.md` (Addendum 14, §5 item 1, Step 0's
  governing decisions).
- `data/SRP1/Results/P515S32_run/`: `g_baseline.json`, `component_levels_terminal.json`,
  `interface_settlement_detail_s31c.json`, `stdout_baseline.log`, `esso_models_baseline.pkl`,
  `frozen_snapshots_baseline.jsonl`, `results/` (incl. `results/FrozenSMOPF/`),
  `s32_supplementary.json`, `s32_e0_coherence.json` (read as cross-checks, not as primary source).
- `data/SRP1/Results/P515S31C_run/stdout_baseline.log` (cross-check only).
- `data/SRP1/case9/case9_2025.json` (bus table: `Vmin`, `Vmax`, `baseKV` for nodes 5/7/9).
- `data/SRP1/SRP1_params.json` (`admm.proximal_regularization.tso.gamma`).
- `shared_resources_planning.py` (read-only; function/line citations, no modification):
  `get_admm_residual_metrics`, `get_admm_boyd_residual_metrics`,
  `_compute_common_admm_objective_scale`, `update_distribution_models_to_admm`,
  `update_shared_energy_storage_model_to_admm`, `_update_tso_proximal_centres_after_solve`,
  `_update_shared_energy_storage_variables`, `create_admm_variables`,
  `_initialize_shared_ess_consensus`, `_print_worst_primal_residual_diagnostics`,
  `_get_tso_voltage_slack_state`, `_print_tso_voltage_slack_transitions`.
- `shared_energy_storage_data.py` (ESSO model construction, `model.objective`), `definitions.py`
  (`PENALTY_ESSO_SLACK`, `EPS_ESSO_THROUGHPUT`).
- `p513_solve_profile_guard.py`, `p515_s32_supplementary.py`, `p515_s32_e0_coherence.py` (existing
  conventions followed; not modified).

## Files modified

None (production, case files, and existing result directories untouched).

## Files created

- `p515_s33_e1_zero_solve.py` — new diagnostic script.
- `data/SRP1/Results/P515S33/E1/e1_zero_solve.json` — output (write-once; script refuses if the
  target file or a non-empty target directory already exists).
- `data/SRP1/Results/P515S33/E1/evidence_manifest_sha256.json` — sha256 manifest of the above.
- `WORKER_REPORT_S33_E1.md` — this report.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s33_e1_zero_solve.py
```
Exit 0. `solve_profile_guard.counts` = `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve':
0, 'blocked_exec': 0}`; `guard.verify(expected_solves=0)` returned no failures — the zero-solve claim
is armed and satisfied for the whole script, including the `esso_models_baseline.pkl` unpickling
(read via `pe.value()` only).

Supporting ad-hoc checks (not written to any output; used to corroborate the script's logic before
finalizing it): `grep` counts of `[DIAG]`/`[ADMM OF SCALE]`/`[SLACK COMPONENTS]` lines in
`stdout_baseline.log`; manual `pe.value()` inspection of `esso_models_baseline.pkl` (base objective,
`admm_objective`, slack/throughput variable ranges, `p_req`/`dual_p_req` gaps); manual trace of
`consensus_vars['ess']['tso']['prev']` writes/reads via `grep -n` across `shared_resources_planning.py`.

## Results

### Q1 — Interface voltages versus bounds

**Bounds** (`data/SRP1/case9/case9_2025.json`, nodes 5/7/9): `Vmin = 0.9 pu`, `Vmax = 1.1 pu`,
`baseKV = 345` — identical at all three nodes.

**Per-entry terminal data: not serialized.** Searched and found absent (full list, with what each
artifact DOES contain, is in the output JSON's `per_entry_data_availability.searched`):
`g_baseline.json` cycle_trajectory (only a single worst-disagreement entry per cycle plus aggregate
norms), `component_levels_terminal.json` (`voltage_slack` penalty is aggregate and near-zero, not
per-entry), `interface_settlement_detail_s31c.json` (P/Q only, no V field),
`stdout_baseline.log` (0 of 48 `[DIAG]` lines are `[DIAG][V MAX]`; 0 `[VOLTAGE SLACK ...]` lines —
both print branches are gated on thresholds that were never crossed, per `shared_resources_planning.py`
lines 5791 and ~5920 respectively), `esso_models_baseline.pkl` (ESSO only, no node voltages),
`frozen_snapshots_baseline.jsonl` + `results/FrozenSMOPF/` (2 records, both cycle-7, not terminal).
**Confidence: high** that no per-entry terminal V array exists in this run's artifacts (exhaustive
grep/listing of the run directory); this does not rule out per-entry data existing in some other,
unindexed location outside `data/SRP1/Results/P515S32_run/`.

**Best available evidence, in place of per-entry data:**

1. *Worst-disagreement envelope* (one entry per cycle, 150 samples, from `g_baseline.json`
   `worst_v_primal_*`): pu range observed **0.9837 – 1.0392**. At the terminal cycle (150, node 5,
   2025/Winter/period 4): TSO 1.03909 pu, DSO 1.03924 pu. Minimum distance to `Vmax` across the
   sample is **0.0608 pu** (cycle 150); minimum distance to `Vmin` is **0.0837 pu** (cycle 1).
   **`Vmax` is the closer (binding) bound of the two, by this sample.**
   *Caveat:* this sample is the single largest **TSO−DSO disagreement** entry per cycle, not the
   largest-**magnitude** entry among the 864 per-cycle V-channel entries; it lower-bounds, but does
   not determine, how close the true population gets to a bound.
2. *Aggregate norm* (`boyd_v_norm_z`, 864 entries, RMS = `norm_z/√864`): rises monotonically every
   cycle from 26 to 150 (`monotone_increasing_fraction = 1.0`), from RMS 0.99971 pu (cycle 25) to
   1.00787 pu (cycle 150), i.e. **direction is up**, mean rate **6.522e-5 pu/cycle/entry** (cycles
   25–150). This matches the report's stated 0.0019/cycle aggregate rise and ≈6.5e-5/cycle
   per-entry rate (reproduced independently).
3. **Entries at/near a bound: not computable** from serialized data (no per-entry array). The
   available proxy (item 1) never sits within 0.005 pu of a bound; `voltage_slack` penalties in
   `component_levels_terminal.json` are O(1e-8)–O(1e-13) (unweighted), confirming no bound is
   currently violated but giving no distance.
4. **Direction / subset:** up. Coherence (reproduced from `g_baseline.json`, same identity as
   `p515_s32_e0_coherence.py`) is **0.7066** (mean, cycles 26–150), giving an implied active
   fraction `coherence² = 0.499` → **≈431 of 864 entries** are the ones moving (assuming the moving
   subset shares the aggregate's ~1 pu magnitude and the rest are static) — consistent with, and a
   direct reproduction of, the report's "coherence 0.71 suggests about half the entries."

**Cycles-to-bound projection** (illustrative extrapolation of the 125-cycle-observed linear trend,
NOT a guaranteed asymptote): at the naive aggregate rate, reaching +0.04/0.06/0.09 pu takes
613/920/1380 cycles; at a refined rate that credits only the ~half of entries inferred to be moving
(2× the aggregate per-entry speed), 306/459/689 cycles. Reaching the observed worst-case entry's
own remaining distance to `Vmax` (0.0608 pu) takes ≈932 (naive) or ≈465 (refined) cycles. These
match the report's "> 600 cycles ... to e.g. +0.04 pu" framing.

**Confidence:** high for the aggregate-norm-based direction/rate/coherence numbers (directly
reproduced from `g_baseline.json` with an independently re-derived `n=864` rather than a hard-coded
value). Low-to-moderate for the cycles-to-bound projection, which is explicitly a linear
extrapolation with an unverified constant-rate assumption, and low for any claim about the true
per-entry extremum (unavailable).

### Q2 — ESSO objective scale versus TSO/DSO σ (D5)

- **σ (`objective_scale`).** `9.363536e7`, printed **once**, before the ADMM loop starts
  (`stdout_baseline.log:62`, `"[ADMM OF SCALE] n=48 ... selected=9.363536e+07"`), by
  `_compute_common_admm_objective_scale` (`shared_resources_planning.py:3049-3122`), called once at
  `shared_resources_planning.py:2412`. It is `max` over the 48 TSO/DSO (year,day) blocks of
  `|admm_block_weight × raw_pre-ADMM_objective|`, from the initial (non-augmented) solve — fixed for
  the whole 150-cycle run. Cross-check: the s31c run (`data/SRP1/Results/P515S31C_run/stdout_baseline.log:62`)
  prints the **identical** value — expected, since σ depends only on the case data and the initial
  solve, not on the gate/balancing variant.
- **`effective_scale` per block.** Not itself stored as a field; reconstructed as
  `objective_scale / admm_block_weight`, using `admm_block_weight` already recorded per block in
  `component_levels_terminal.json`, per the production formula at
  `shared_resources_planning.py:3978-3983` (TSO) / `:4130-4136` (DSO). Range across the 48 blocks:
  **2.036e5 to 2.509e5**.
- **ESSO: no division by σ**, confirmed at `shared_resources_planning.py:4209`
  (`obj = copy(models[node_id].objective.expr)`, no `/effective_scale`, no `admm_objective_scale`
  Param on the ESSO model) — contrast with TSO (`:3982-3983`) and DSO (`:4133-4136`).
- **ESSO base objective at terminal** (unpickled `esso_models_baseline.pkl`, `pe.value()` only, no
  solve): **≈ −5.7227e-3** at all three nodes (5: −0.0057227, 7: −0.0057228, 9: −0.0057226).
  `admm_objective.expr` (base + AL terms) ≈ −5.7238e-3 at all three, i.e. the AL contribution is
  ≈ −1.12e-6 — itself far smaller than the already-tiny base term.
  - **This base term is not an economic cost.** `model.objective = model.feasibility_penalty`
    (`shared_energy_storage_data.py:793-795`) = `PENALTY_ESSO_SLACK (=1e3) × Σ(slack_es_pnet_up +
    slack_es_pnet_down)` + `EPS_ESSO_THROUGHPUT (=1e-5) × Σ(throughput)` (constants in
    `definitions.py:63,74`). At terminal, every `slack_es_pnet_{up,down}` sits at ≈ −1e-8
    (IPOPT's bound-multiplier tolerance below its 0 lower bound, ×~576 slack entries × 1e3 ≈
    −0.0058), and no degradation/investment cost term exists in this objective at all, per the
    governing decision that investments are fixed parameters (`PLANNER_BRIEF_2026-09-13.md`).
- **Implied relative weighting.** The factor is σ itself (2.036e5–2.509e5): dividing the ESSO base
  objective by the mean effective_scale would push it from ≈ −5.72e-3 to ≈ −2.52e-8 — six further
  orders of magnitude smaller. **Observation, not yet interpretation:** because the ESSO base
  objective is already negligible (a numerical-tolerance artifact, not a priced cost), D5's
  structural asymmetry (real: ESSO isn't divided, TSO/DSO are) does not, AT THIS TERMINAL ITERATE,
  translate into the ESSO base objective distorting or dominating its own AL penalty — there is no
  economic signal for σ to be protecting either way. **This is a single terminal snapshot; whether
  the same held earlier in the run (e.g. before the investment-fixing revision's effect settled) is
  not established here.**

**Confidence:** high for all numerical values quoted (directly read/computed from serialized
artifacts, zero solves). Moderate for the interpretive conclusion about D5's practical
(non-)impact, which the Planner should weigh against whether F5 needs separate testing at earlier
cycles or under a different objective specification.

### Q3(a) — `consensus_vars['ess']['tso']['prev']` vs the TSO proximal centre

Two independent trackers exist:

- **The Boyd-diagnostic tracker**, `consensus_vars['ess']['tso']['prev']`, written at
  `shared_resources_planning.py:6394-6395` inside `_update_shared_energy_storage_variables`
  (parameter name there is `shared_ess_vars`, bound to `consensus_vars['ess']` by the caller at
  `shared_resources_planning.py:206`), gated on `_solver_result_succeeded(results['tso'][...])`,
  called via `update_and_check_convergence(update_tn=True)` at `shared_resources_planning.py:2496-2503`.
- **The TSO's own proximal-regularization centre**, `local_model.prox_ess_p_prev`/`prox_ess_q_prev`
  (pyomo Params on the TSO model itself), written at `shared_resources_planning.py:4419-4420` inside
  `_update_tso_proximal_centres_after_solve`, gated on the same per-block success check
  (comment at line 4255: "A failed TSO block keeps its previous proximal centre"), called at
  `shared_resources_planning.py:2495` — immediately after the TSO solve and immediately BEFORE the
  consensus-var update above.

**Tracing one ADMM cycle k:** TSO solve (uses the centre set at the end of cycle k−1) → centre
updated to the cycle-k solution (line 2495) → consensus_vars `prev` ← old `current` (= cycle k−1
solution = the just-used centre), `current` ← cycle-k solution (lines 2496-2503) → ESSO solve/update
→ `get_admm_boyd_residual_metrics` called (`shared_resources_planning.py:2534`), i.e. only AFTER the
full cycle-k sequence. At that read point, `consensus_vars['ess']['tso']['prev']` **equals** the
centre that regularized this cycle's TSO solve, and `['current']` is this cycle's TSO output.

**Answer: no divergence found.** `γ·a·(x_TSO^k − x_TSO^{k−1})`, computed at
`shared_resources_planning.py:5706-5710`, is numerically identical (to floating-point precision) to
the actual proximal displacement `(x_TSO^k − centre)`, because both trackers are updated in the same
solve-triggered event, in the same order, under the same success gate, and the Boyd read happens
only after both have advanced. **Corroboration from artifacts:** the reconstructed
`rms_step_per_entry` for the ESS channel (from `boyd_ess_s_proximal_part`, same identity as
`p515_s32_supplementary.py`) is O(1e-5) at every sampled cycle (1, 10, 25, ..., 150) — small
per-cycle displacements, not O(1) absolute consensus levels, which is what a stuck/zero `prev`
tracker would produce. This is consistent with, not independent proof of, the code-level finding.

**Confidence:** high (direct code trace with exact file:line citations for both write sites and the
call order; corroborated, not merely asserted, by the magnitude check). This rules out "the Boyd
diagnostic reads a stale/mismatched centre" as a confound for the ESS drift; it does not by itself
adjudicate hypothesis (d) vs F5 for the ESS drift's cause.

### Q3(b) — PF `‖y‖` units

Traced the full round trip: (1) `constraint_p_req` in the DSO's own augmented Lagrangian
(`shared_resources_planning.py:4156`) is normalized by `interface_transf_rating = rating/s_base`
(the interface's MVA rating, expressed in the DSO's p.u.), **not** by `s_base` alone. (2)
`dual_pf_p_req[p]`, the Param actually multiplying `constraint_p_req` inside that objective, is set
via `.../s_base` (`shared_resources_planning.py:4980-4981`) from the raw accumulated dual, whose
update increment (`shared_resources_planning.py:6348-6351`) was itself built as
`rho_pf × (error_MW/rating_MW) × s_base`. (3) The `×s_base` (accumulation) and `÷s_base` (Param
set-value) cancel exactly, so `dual_pf_p_req[p]` equals the correct dual-ascent multiplier conjugate
to `constraint_p_req` (rating-normalized), with no leftover `s_base` or rating factor. (4) Boyd's
`y_pf = lambda_dso_pf / s_base_dso` (`shared_resources_planning.py:5667`) applies the identical
divisor to the same raw stored value, so `y_pf` reproduces `dual_pf_p_req[p]` exactly — the value
the DSO subproblem actually optimizes against.

**Answer: no unit mismatch.** `y_pf`, `r_pf` and `s_pf` are all in the same rating-normalized units;
the two `s_base` operations are a self-cancelling round trip through an MW-scale intermediate
storage convention, not an independent second normalization. **No correction to ε_dual is implied.**
This is an algebra-level trace, corroborating rather than duplicating the s32 report's existing
numeric "mapping diagnostic" (`p515_s32_mapping_diag.py` / `WORKER_REPORT_S32_MAPPING_DIAG.md`,
already cited in the report as verified to machine precision).

**Confidence:** high for the algebraic trace (every step is a direct file:line read of the actual
formula, and the units cancel exactly rather than approximately). This is a static trace, not a
fresh numeric replay at cycle 150 — the existing mapping diagnostic already performed the numeric
replay.

## Validation

- Script runs to completion, exit 0; `SolveProfileGuard.verify(expected_solves=0)` passes with all
  counts at 0 (`permitted_solve`, `permitted_exec`, `blocked_solve`, `blocked_exec` all 0) —
  the zero-solve claim is armed, not asserted, for the entire script including the pickle read.
  This is code-executes-correctly + diagnostic-produces-output evidence; it does not by itself
  establish that the underlying ESS/V drift mechanism is understood — see "Remaining issues."
- Spot-checked every computed number in this report against the printed script output and against
  independent manual `python -c` computations performed during investigation (aggregate norm
  deltas, coherence, effective_scale table, ESSO pickle values, `grep` counts) — all matched.
- Cross-checked the σ value against the independent s31c run's stdout (identical, as expected).
- Verified file:line citations by re-reading the cited ranges after writing the report (all
  confirmed against the current `shared_resources_planning.py`).

## Unexpected findings

- None rising to the level of a bug: the `consensus_vars['ess']['tso']['prev']` trace initially
  looked, from a narrow grep, like it might never be written (only one literal-pattern match, a
  read); the broader trace (via the `shared_ess_vars` parameter alias) showed it IS written
  correctly every cycle. Recorded here so the false lead isn't silently repeated by a future
  narrow grep.
- The ESSO base objective at this revision is dominated by an IPOPT-tolerance artifact
  (`slack_es_pnet_{up,down}` sitting ~1e-8 below their 0 lower bound) rather than any priced
  quantity — worth the Planner's attention if a future revision reintroduces a real degradation
  cost term into this objective, since the current D5 asymmetry would then matter much more.

## Remaining issues

- Per-entry terminal interface-voltage data is not recoverable from this run's artifacts; any
  future run intended to answer "how close does entry X get to its bound" should serialize a
  per-entry V snapshot (or at least the true per-cycle extremum, not just the worst-disagreement
  entry).
- The cycles-to-bound projection (Q1) is a linear extrapolation of a 125-cycle window; it is not a
  claim about the eventual limit, consistent with the report's rule-ten caution that neither s32
  nor s31c had settled.
- Q2's conclusion is a single terminal (cycle 150) snapshot; it does not establish whether the same
  held at earlier cycles.

## Questions for Planner

- For Q1, is a genuinely per-entry terminal V capture (not just the worst-disagreement sample)
  worth adding to a future gate's serialization, given E2/E3 are already authorized as separate
  experiments?
- For Q2, given the ESSO base objective is dominated by an IPOPT-tolerance artifact rather than a
  priced term, does F5 still warrant a dedicated experiment ahead of hypothesis (d), or does this
  evidence move it down the priority order relative to E2 (γ tied to ρ)?

---

## Q1–Q3 summary

- **Q1.** No per-entry terminal V array is serialized (exhaustively searched; listed above). The
  worst-disagreement sample per cycle stays within 0.061–0.084 pu of the 0.9/1.1 pu bounds
  throughout 150 cycles; the aggregate 864-entry norm is rising monotonically (RMS 0.9983 →
  1.0079 pu, cycles 1→150) at 6.522e-5 pu/cycle/entry, toward `Vmax`, with about half the entries
  (coherence-implied 431/864) apparently doing the moving. At the observed rate, illustrative
  extrapolation puts a 0.04–0.09 pu further move at roughly 300–1400 cycles out, depending on
  whether the aggregate or the "active-subset-only" rate is used — well beyond the 150-cycle run,
  consistent with the report's own framing.
- **Q2.** σ (`effective_scale`) is 2.04e5–2.51e5 across the 48 TSO/DSO blocks, computed once
  pre-loop and shared with s31c. ESSO's base objective is not divided by σ, but at terminal it is
  ≈ −5.72e-3 at every node — a numerical-tolerance artifact (IPOPT bound-multiplier residual on
  slack variables), not a priced cost, since no degradation/investment term exists in this
  objective under the current revision. Dividing it by σ would push it to ≈ −2.5e-8. D5's
  structural asymmetry is real, but at this terminal snapshot it has no practical cost-inflation
  effect because there is no economic signal in the ESSO base term either way.
- **Q3.** (a) `consensus_vars['ess']['tso']['prev']` and the TSO's own proximal centre are updated
  in the same solve-triggered event, in the same cycle, under the same success gate; the Boyd ESS
  proximal entry is numerically identical to the actual proximal displacement — no divergence
  found. (b) PF `‖y‖` is in the same rating-normalized units as `r`/`s`; the apparent double
  `s_base` operation is a self-cancelling round trip, not a units error — no correction to ε_dual
  is implied.
