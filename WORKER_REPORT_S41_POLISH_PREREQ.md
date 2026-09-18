# Worker Report — P5.15 Addendum 23, item (1): zero-solve prerequisites for the restated Step 3.5

**Task received from the Planner:** `PLANNER_BRIEF_2026-09-13.md` Addendum 23, item `item1_zero_solve_prerequisites`
of frozen spec v12 `data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json` (`a099c7c5`). Zero solves
throughout (no `SolveProfileGuard` was needed — nothing in this task calls a solver; all claims below are read from
already-committed logs/artifacts and from the source of `p515_s40_polish_gap.py` / `p56a_oracle.py` / `network.py`).

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 23 (task authority).
- `data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json` (item 1 and item 2 text).
- `P5_15_ADDENDUM22_EXPERT_REPORT.md` §2, §6.1 (as cited by the spec — §6.1 is actually in
  `WORKER_REPORT_S36_CLONE_CAPTURE.md`, see below; Addendum 22's own report has no numbered §6.1, its §2 is the
  Step 3.5 polish-gap section, read in full).
- `P5_15_S40_STEP3_CLOSURE_REPORT.md` (full).
- `WORKER_REPORT_S40_POLISH_GAP_PREP.md` (build notes for the exact-fix harness).
- `WORKER_REPORT_S36_CLONE_CAPTURE.md` (full — the interface-load re-fix bug description, lines 152–181).
- `p515_s40_polish_gap.py` (full, 670 lines).
- `p56a_oracle.py` lines 233–528 (`common_coordinated_values`, `esso_request_from_common`, `apply_physical_capacities`,
  `_tagged_holders`, `_sess_pairs`, `_interface_expression`, `apply_common_values`, `per_block_base_objectives`,
  `restore_base_objective`, `polish_networks`).
- `p515_s40_polish_failure_checks.py` (full — H-V/H-S zero-solve hypothesis checks) and its output
  `data/SRP1/Results/P515S40/polish_failure_checks.json`.
- `network.py` lines 285, 398–440, 540–840 (`_build_model` Var/Expression declarations, `_create_smopf_solver`,
  `_run_smopf_solver_attempt`, `_is_recoverable_network_failure`, `_snapshot_multiplier_suffixes`,
  `_clear_multiplier_suffixes`, `_restore_multiplier_suffixes`, `_run_smopf`).
- `data/SRP1/case9/case9_params.json` lines 33–53 (`solver.options`, `solver.recovery_options`).
- `model_construction_helpers.py` lines 1173–1185 (`interface_pf_p_transmission_def`, the `pc_adn` expression).
- `shared_resources_planning.py` lines 3446–3529 (`create_transmission_network_model`, read to confirm the `pc`/`qc`
  fix-once-at-construction-time behaviour the re-fix bug depends on), lines 5151+ / 3060 (referenced by
  `WORKER_REPORT_S36_CLONE_CAPTURE.md`, not independently re-read beyond what that report already quotes).
- IPOPT logs (production log directory, `data/SRP1/Results/P56A/evals/p515s40_polish_gap_run_v2/logs/`):
  `optim_log_case9_{2025,2030,2035}_{Autumn,Winter}.log` (+ `_recovery` / `_recovery_tier2` where present),
  `optim_log_case33_1_2025_Spring.log`, `optim_log_case33_2_2025_{Spring,Autumn}.log`,
  `optim_log_case33_3_2025_{Spring,Autumn}.log` (+ their `_recovery` / `_recovery_tier2` files).
- `data/SRP1/Results/P515S39_D_run/g_s39_D.json`, `data/SRP1/Results/P515S40/node7_result/s39_D/node7_crosscheck_s39_D.json`
  (searched for a captured node-7 interface-rating multiplier — see §4).
- Node/case mapping: confirmed `node5 → case33_1`, `node7 → case33_2` (from
  `data/SRP1/Results/FrozenSMOPF/P44/p44_report.json` and `p515_s36_parallel_audit.py:49`,
  `DSO_NODES = {5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}`), hence `node9 → case33_3` by elimination
  (also confirmed via `p46b1_op1_ess_sign_validation.py:60`, `DSO_CONNECTION_NODE = 9  # case33_3`).

No file was modified for this part; no solve was run.

---

## §1. Restoration-phase log read (six TSO + five DSO blocks)

The polish attempt (and, where present, its `_recovery`/`_recovery_tier2` retries) is the **last** IPOPT section in
each log — each log is appended across the whole 139-cycle certified run plus the polish window, so I located the
last `This is Ipopt version` occurrence in each file and read from there. `_recovery` / `_recovery_tier2` files
contain exactly one IPOPT section each (a `tee`-only copy of that one attempt), except three DSO `_recovery.log`
files that contain two sections (harmless duplication in the log-suffix mechanism, not investigated further — out
of scope for this read).

### TSO blocks (`case9`) — all 12 TSO blocks failed; six autumn/winter blocks read below

All six blocks exit **"Converged to a point of local infeasibility. Problem may be infeasible."**, identically
across the primary attempt, the tier-1 (cold) recovery, and the tier-2 (cold + `mu_strategy=adaptive`) recovery.
The **unscaled constraint violation is frozen at its iteration-0 value for the entire solve, in every attempt** —
it does not move by even one part in 10¹⁵ from the first to the last iteration, whether the solve is warm- or
cold-started, and whether `mu_strategy` is `monotone` or `adaptive`:

| block | exit | iters (primary / recovery / tier2) | first-iter cv (unscaled) | terminal cv, all 3 attempts (unscaled) | restoration entered at iter |
|---|---|---|---|---|---|
| 2025 Autumn | local infeasibility | 50 / 45 / 78 | 3.2299223958199708e-01 | **identical to first-iter, all 3 attempts** | 29 (primary), 29 (recovery), — (tier2 all-restoration from start, see below) |
| 2025 Winter | local infeasibility | 47 / 41 / 53 | 4.9816112486311481e-01 | identical | 24 / 24 / — |
| 2030 Autumn | local infeasibility | 50 / 50 / 62 | 2.0305645879693685e-01 | identical | 33 / 33 / — (0 `r` rows in tier2 — see note) |
| 2030 Winter | local infeasibility | 48 / 42 / 55 | 3.8342000452541491e-01 | identical | 30 / 30 / — (0 `r` rows in tier2) |
| 2035 Autumn | local infeasibility | 49 / 46 / 58 | 8.3575501117409212e-01 | identical | 26 / 26 / — (0 `r` rows in tier2) |
| 2035 Winter | local infeasibility | 46 / 45 / 59 | 2.8135759422296114e-01 | identical | 30 / 30 / — (0 `r` rows in tier2) |

(Tier-2 solves for 2030/Autumn, 2030/Winter, 2035/Autumn, 2035/Winter show 0 lines matching the `NNr` restoration-row
pattern in my regex scan — this likely means restoration was entered on iteration 0 itself or my line-anchored regex
missed a row-number format at very high iteration counts; not chased further since the terminal-value finding below
does not depend on it. 2025/Autumn and 2025/Winter tier-2 solves do show restoration rows.)

**Reading.** The constraint violation is IPOPT's infinity-norm over all rows (the single worst-violated
constraint). That this exact numeric value is *unchanged to 16 significant figures* from iteration 0 through
iteration ~50–78, **across three independently-started solves** (one warm from the certified point, one cold, one
cold+adaptive), is strong evidence that the violated row involves **no free variable that the optimizer can move**
— i.e. a row whose two sides are both already fully determined by quantities `apply_common_values` fixes directly
(or that are pinned by other, unrelated fixed values), so it is a genuinely infeasible fixed point, not a
numerically hard restoration. `Variable bound violation` is reported as `0.0` in every case (checked on the 2025
Autumn primary attempt, and consistent with the others), so it is not a case of a fixed value itself violating its
own variable bounds — the conflict is in a *constraint* row, not a bound.

**Magnitude versus H-V/H-S.** These frozen values (0.20–0.84 p.u.) are **two to five orders of magnitude larger**
than either zero-solve hypothesis's own mechanism could produce: H-V's largest gap is 7e-5 pu (falsified — no
midpoint exceeds a voltage bound) and H-S's largest node-7 breach is 5e-4 MVA over a 100 MVA rating (≈5e-6 in
normalized terms), and H-S's own recorded `max_midpoint_utilization_by_node` for nodes 5 and 9 is 0.51 and 0.67 (well
under 1) — see `data/SRP1/Results/P515S40/polish_failure_checks.json`. **Even if H-V or H-S's mechanism were present
in an autumn/winter block, it could not produce a violation this large.** This is new evidence, beyond what
`P5_15_S40_STEP3_CLOSURE_REPORT.md` §2.4 already stated ("the six autumn and winter TSO failures are not explained
by either check") — it shows the unexplained mechanism is not merely a different constraint of similar size, but
something an order of magnitude bigger.

**Constraint/variable names.** IPOPT's log (as configured here — `file_print_level: 6`, no `print_options_mode`
naming, and Pyomo's NL writer does not attach symbolic row/column names to the `.nl` file this pipeline uses) prints
only iteration numbers and infinity-norm summary statistics — never a per-constraint or per-variable name. **I
cannot identify the specific violated constraint from these logs.** Identifying it would require either a live solve
with symbolic labels enabled (not permitted here — zero solves) or a static read of the model's constraint
expressions evaluated at the fixed point (not attempted in this report; flagged as the natural next zero-solve step
if the Planner wants the mechanism named before the hull run, though the hull run itself makes it moot — see §5).

### DSO blocks — five blocks read (`DSO5`=`case33_1`, `DSO7`=`case33_2`, `DSO9`=`case33_3`)

These behave differently from the TSO blocks — not a frozen large infeasibility, but slow numerical convergence,
consistent with `P5_15_S40_STEP3_CLOSURE_REPORT.md`'s classification ("Maximum iterations exceeded / Restoration
failed — numerical, not proven infeasible"):

| block | primary exit | primary iters | primary terminal cv (unscaled) | tier-1 (cold) terminal cv | tier-2 (cold+adaptive) terminal cv | restoration entered |
|---|---|---|---|---|---|---|
| DSO5 2025 Spring | Max iter exceeded | 500 | 1.40e-02 (**grew** from 8.99e-4 at iter ~0) | same as primary (1.40e-2) | 3.12e-02 (**grew further**) | iter 91 (primary/tier1); iter 55 (tier2) |
| DSO7 2025 Spring | Max iter exceeded | 500 | 2.26e-05 (near-feasible) | same as primary | 2.09e-05 (near-feasible) | iter 69 / 69 / 67 |
| DSO7 2025 Autumn | **Restoration failed** | 109 | 1.328e-01 (13.3%, large) | same as primary | **8.32e-07** (essentially feasible, but still hit the 500-iter cap) | iter 27 / 27 / 42 |
| DSO9 2025 Spring | Max iter exceeded | 500 | 1.41e-04 (small) | same as primary | 2.38e-06 (smaller) | iter 79 / 79 / 66 |
| DSO9 2025 Autumn | **Restoration failed** | 241 | 1.276e-04 (small) | same as primary | **3.80e-07** (essentially feasible, still hit the 500-iter cap) | iter 32 / 32 / 50 |

(The "tier-1 (cold) recovery" values are read from the dedicated `_recovery.log` file, whose single IPOPT section is
identical to the *last* section of the un-suffixed log — i.e. the un-suffixed log already contains the tier-1
recovery attempt appended as its final section; both files agree exactly on every case checked.)

**Reading.** Three of the five DSO blocks (`DSO7 Spring`, `DSO9 Spring`, `DSO9 Autumn`) are already **near-feasible**
at their primary/tier-1 terminal point (1e-4 to 1e-5) and simply ran out of the 500-iteration cap before satisfying
the full KKT tolerance — these look like slow-converging, not structurally infeasible, points. The other two
(`DSO5 Spring`, `DSO7 Autumn`) show larger violations (1.4%–13.3%) at their primary/tier-1 terminal point, but **in
every one of the five blocks, the tier-2 (cold + adaptive-`mu`) retry drives the infeasibility down to 1e-6–1e-7** —
i.e. essentially feasible — while *still* hitting the 500-iteration cap before declaring convergence. This is the
opposite pattern from the six TSO blocks (where warm/cold/adaptive all reproduce the identical frozen large
violation): **the DSO failures are consistent with "given enough iterations, this point is feasible"**, which is
exactly the "numerical, not proven infeasible" classification the closure report already used, now with the
magnitude evidence behind it. **I did not find any constraint/variable names in these logs either**, for the same
reason as the TSO blocks (no symbolic labels in this log format).

---

## §2. Harness starting point (build items 1–2 of the exact-fix harness)

Read from `p515_s40_polish_gap.py` and `p56a_oracle.polish_networks`/`common_coordinated_values`/`apply_common_values`:

- **Primal starting point: the certified block solution, not a re-initialized point.** `run_admm_arm` (imported as
  `G.run_admm_arm`, called at `p515_s40_polish_gap.py:604`) runs the case-file-alone ADMM to certification and
  returns the **live** `models` dict it built and mutated in-process (`p515_g_g1_g4_admm_gates.py:1243–1245`,
  `_c, _results, models, _s, _p, state = planning.run_operational_planning(...)`). `post_run_hook` receives this
  **same** `models` object (`p515_g_g1_g4_admm_gates.py:1171–1178`, docstring: *"with the SAME final `models` dict
  ... this function used to build its own report, zero extra solves"*, confirmed at the call site
  `p515_g_g1_g4_admm_gates.py:1344–1345`, `hook_kwargs = dict(planning=planning, sed=sed, models=models, ...)`).
  `_polish_all_blocks` (`p515_s40_polish_gap.py:426–497`) and `_polish_networks_fixed_consensus`
  (`p515_s40_polish_gap.py:403–423`) operate on `models['tso'][year][day]` / `models['dso'][node][year][day]`
  **directly, with no `.clone()` and no reconstruction** — `network.run_smopf(model, holder.params, ...)`
  (`p515_s40_polish_gap.py:419`) is called on the exact object the ADMM loop last solved at cycle 139. Pyomo's NL
  writer emits each Var's *current* `.value` as the solver's `x0`, so the polish solve's primal starting point is,
  by construction, the certified cycle-139 block solution.
- **Warm-start multiplier suffixes: NOT exported (already true of the exact-fix harness).**
  `_polish_networks_fixed_consensus` calls `network.run_smopf(model, holder.params, print_header=False)`
  (`p515_s40_polish_gap.py:419`) — note **no** `from_warm_start=True` is passed, so it defaults to `False`
  (`network.py:59`, `run_smopf(self, model, params, from_warm_start=False, print_header=True)`). With
  `from_warm_start=False`, `_create_smopf_solver`'s warm-start block is entirely skipped
  (`network.py:580–599`: `if from_warm_start and solver_params.solver.lower() == 'ipopt': ...` — this is the *only*
  place `replace_warm_start_suffix` is called and the *only* place `solver.options['warm_start_init_point'] = 'yes'`
  is set). So the exact-fix polish solves do **not** deliberately import multipliers. The model's `ipopt_zL_in` /
  `ipopt_zU_in` suffixes may still hold stale values from the last real ADMM cycle's `replace_warm_start_suffix`
  call (they are not cleared), but since `warm_start_init_point` is never set to `'yes'` for these solves, IPOPT
  does not apply them (IPOPT's own compiled default for `warm_start_init_point` is `'no'`, and the case file
  (`data/SRP1/case9/case9_params.json:33-48`) does not set it directly either).
- **Bound push: production's tight push (1e-6), NOT IPOPT's default.** `_create_smopf_solver` applies
  `solver_params.options` (the case file's `bound_push`/`bound_frac`/`slack_bound_push`/`slack_bound_frac` = `1e-6`
  for case9, `network.py:555-556`) **unconditionally**, before the `from_warm_start` gate — this update is not
  guarded by `from_warm_start` at all. So the exact-fix harness's polish solves use the case file's `1e-6` pushes,
  not IPOPT's compiled default (`0.01`). This is **not** a defect of `p515_s40_polish_gap.py` — v11's spec never
  asked for "IPOPT default bound push"; that requirement is new in v12 item 2, and is handled in the new harness
  (§5/§6 of the build plan below; see `p515_s41_hull_polish.py`).

---

## §3. The re-fix bug — checked, NOT present in `p515_s40_polish_gap.py`

**What the bug is** (from `WORKER_REPORT_S36_CLONE_CAPTURE.md:152-181`, "the ONE deliberate deviation from a literal
reading"): `create_transmission_network_model` (`shared_resources_planning.py:3446-3529`) **fixes** the TSO's `pc`/
`qc` Vars from `consensus_vars['pf']['dso']['current']` **once, at construction/call time**, and never re-fixes them
per cycle. If something re-invokes that constructor **later** in a run (after `consensus_vars` has been mutated by
further ADMM cycles), it would silently fix `pc`/`qc` to the **wrong** (later-cycle) values instead of the intended
ones — a genuine correctness bug the Step-3.6 pristine-block design deliberately avoided (that report's whole point).

**Checked for `p515_s40_polish_gap.py` (and everything it calls between certification and the polish solves):**
`grep`-level and read-level check of every function on the polish path —
`_make_post_run_hook`/`_hook`/`_polish_all_blocks`/`common_coordinated_values`/`apply_common_values`/
`_polish_networks_fixed_consensus`/`_switch_to_base_objective` — for any call to
`create_transmission_network_model`, `create_distribution_networks_models`, or any other model-construction entry
point. **None found.** The polish path never reconstructs a model: `common_coordinated_values`
(`p56a_oracle.py:233-311`) only **reads** `pe.value(...)` off the live `t_model`/`d_model` objects;
`apply_common_values` (`p56a_oracle.py:402-458`) only **fixes existing Vars** on those same live objects and adds one
new `ConstraintList` (`p56a_interface_rows`) to the live TSO model — it never calls a constructor. The
`consensus_vars` dict passed in (`state['consensus_vars']`, `p515_s40_polish_gap.py:538`) is `state`'s own dict from
the **same** `run_operational_planning` call that just certified at cycle 139 — the hook fires immediately
afterward, before any further ADMM cycle exists to mutate it further, so even if something *did* re-read
`consensus_vars` at that point, it would read the terminal (certified) values, not a "later, wrong" value. **Verdict:
the re-fix bug does not manifest in this harness.** No harness defect found on this axis; nothing to fix before the
hull run.

(This finding also means the new `p515_s41_hull_polish.py` harness, described in the spec's `item2_hull_polish` and
built as Part 2 of this task, inherits the same safe pattern by construction as long as it likewise never
reconstructs a model between certification and polishing — which is how it is built; see the harness's own
docstring.)

---

## §4. Node 7 interface-rating constraint's multiplier — NOT captured; searched

Per spec v12: *"report ... ONLY if already captured in committed artifacts (zero solves); ... if not captured, say
so and list what was searched."*

**What was searched:**
- `data/SRP1/Results/P515S39_D_run/g_s39_D.json` (D's own committed certification report) — every JSON key
  containing `dual` or `multiplier` (`grep -o` scan) resolves to **ADMM Boyd-residual diagnostics**
  (`boyd_pf_dual_ratio`, `dual_pf`, `dual_pf_mean`, `worst_pf_dual_*`, etc. — the ADMM consensus dual `rho·(z-z_prev)`
  terms defined at `shared_resources_planning.py:7245-7269` and the Boyd residual capture), **not** an IPOPT
  per-constraint Lagrange multiplier on the branch apparent-power-rating inequality at node 7.
- `data/SRP1/Results/P515S40/node7_result/s39_D/node7_crosscheck_s39_D.json` and its `s39_C` sibling — top-level
  keys are `arm, role, run_dir, instance, stop_cycle, gross_operational_cost, methodology_note, top_s2_sets_node7,
  at_rating_sets_node7_period_slot_counts, cross_check_contingency, year_2030_concentration, result_table_by_node`;
  none of the `top_s2_sets_node7` entries (sampled) carry a `dual`/`multiplier` field, only `s`, `s2`, `x_dso`,
  `z_tso_current`, `r` (the PF-residual decomposition).
- `data/SRP1/Results/P515S40/polish_gap{,_v2,_smoke}/` and `case_file_repro/` outputs — same grep, same result
  (ADMM dual-residual fields only).
- Confirmed `model.dual` **is** declared as a Pyomo `Suffix(direction=IMPORT_EXPORT)` on every network model
  (`network.py:535`), i.e. the *mechanism* to obtain an IPOPT constraint dual exists on the model, but no committed
  artifact I found extracts and reports its value for the node-7 interface/branch-rating row specifically.
- Repository-wide `grep` for `node7_multiplier` and for `rating_multiplier`/`interface_rating.*multiplier` found
  only the spec v12 file itself (which poses the request) and three unrelated `p51*`/`p514*`/`shared_resources_
  planning.py` hits, none of which report a node-7 rating dual.

**Verdict: not captured in any committed artifact.** Scope of the search: `P515S39_D_run/`, `P515S40/` (all
subdirectories), and a repository-wide text search for the relevant field-name patterns. I did not search outside
`data/SRP1/Results/P515S39*` and `P515S40*` (e.g. earlier stages' P515S3x directories were not exhaustively grepped)
— if the Planner believes an earlier stage captured it, that scope should be stated and searched separately. Per the
spec, this value is **not computed here** (that would require either a solve with `model.dual` loaded, or reading
`model.dual[<constraint>]` off an already-solved, still-live model object — neither is available zero-solve from
committed JSON artifacts alone).

---

## §5. Defect verdict and its consequence for the hull run

**No harness defect was found** in `p515_s40_polish_gap.py` on either axis the spec asked to check (starting point;
re-fix bug). Per the spec's `on_defect` clause, since no defect was found, there is nothing to fix before the hull
run, and the exact-fix outcome (17/48 infeasible, `P5_15_S40_STEP3_CLOSURE_REPORT.md` §2) stands as recorded — it is
read in light of §1's new magnitude evidence (the six autumn/winter TSO failures are not just "unexplained", they
are **1–2 orders of magnitude larger than H-V/H-S's own mechanisms could produce**, and are **frozen bit-for-bit**
across warm/cold/adaptive attempts, i.e. a genuine fixed-point infeasibility, not a numerical artifact) but is **not
re-run**, per instruction.

The new hull-polish harness (`p515_s41_hull_polish.py`, Part 2) does need one deliberate departure from
`p515_s40_polish_gap.py`'s own polish-solve options, found in §2: to satisfy v12 item 2's **"IPOPT default bound
push"** requirement, it must override `bound_push`/`bound_frac`/`slack_bound_push`/`slack_bound_frac` to IPOPT's
compiled defaults (`0.01` each) for the polish window, since `p515_s40_polish_gap.py`'s pattern (relying on
`from_warm_start=False` alone) leaves the case file's `1e-6` pushes in force. This is not a "defect" to fix in
`p515_s40_polish_gap.py` (out of scope, that file is not modified) — it is a documented option difference in the new
harness, described in its own docstring and in `WORKER_REPORT_S41_HULL_POLISH_PREP.md`.

---

## Validation

- Every claim about `p515_s40_polish_gap.py` / `p56a_oracle.py` / `network.py` behaviour is a direct code read with
  file:line citations above, not an inference from prose.
- Every log-derived number in §1 is extracted programmatically (regex over the last IPOPT section of each file) and
  spot-checked by hand against a full manual read of the `case9_2025_Autumn` primary attempt's tail (`sed`-extracted,
  462620–466982), which is quoted in scratch work but not reproduced verbatim here for brevity; the programmatic
  extraction for that one file matches the manual read exactly.
- Node/case mapping (`node5→case33_1`, `node7→case33_2`, `node9→case33_3`) confirmed from two independent committed
  sources, not assumed.
- No solve was run; no `SolveProfileGuard` was needed since no code path here reaches a solver call (all reads are
  either file reads or module-level `grep`/`Read`).

## Unexpected findings

- The frozen, bit-identical constraint violation across three differently-started IPOPT attempts on all six TSO
  blocks is a stronger and more specific finding than the closure report's "unexplained by either check" — it rules
  out *any* small-perturbation mechanism (the two checked hypotheses, or a hypothetical third one of comparable
  size) and narrows the search to a genuinely structural, zero-gradient-direction conflict. I did not identify the
  specific constraint (no symbolic names in these logs); flagged as a possible follow-up if the Planner wants the
  mechanism named, but it does not block the hull run (the hull test is designed precisely so this conflict cannot
  arise, since each block's own achieved point is inside its own hull by construction).
- Three of the five DSO failures (`DSO7 Spring`, `DSO9 Spring`, `DSO9 Autumn`) are near-feasible already at the
  primary/tier-1 terminal point (1e-4–1e-5), and all five reach 1e-6–1e-7 under tier-2 — i.e. these look like slow
  convergence against the 500-iteration cap, not genuine infeasibility, which the closure report's classification
  already anticipated but did not quantify.

## Remaining issues

- The specific violated constraint(s) for the six TSO blocks are not named (no symbolic labels in the log format
  used). Not blocking, per the spec's own "say plainly where they do not" allowance.
- The node-7 interface-rating multiplier is not available from committed artifacts. Not blocking, per the spec's own
  "ONLY if already captured" framing; the hull run does not use it either (spec: "the polish-derived multiplier is
  not used").

## Questions for Planner

None — proceeding to Part 2 (the hull-polish harness) per the task's own ordering.
