# Stage P4.3–P4.4 — Production kappa-scaling: construction/equivalence validation and frozen regression (executed)

Executed by the user via `p43_production_kappa_validation.py` and
`p44_production_frozen_regression.py`, against the real production code
(`network.py` / `model_construction_helpers.py` / `definitions.py`) already
modified under P4.2 — not diagnostic reimplementations.

Raw output:
- `data/SRP1/Results/FrozenSMOPF/P43/p43_report.json`
- `data/SRP1/Results/FrozenSMOPF/P44/p44_report.json`

Both stages exercise the actual production functions directly
(`sess_snet_def_rule`, `shared_ess_snet_def_scale`,
`configure_shared_ess_operational_state` /
`_sync_sess_snet_def_scale`). P4.3 validates fresh-construction and
live-capacity-change behavior on freshly built DSO (node 7, `case33_2`) and
TSO (`case9`) models. P4.4 replays the actual frozen pre-P4 failure/success
pickles through the current production solver path. No changes were made to
`sess_comp`, SOC, ADMM/Benders logic, solver settings, or ordinary/standard
ESS equations in either stage.

---

## P4.3 — Construction/equivalence validation

`all_invariants_hold: true` for both DSO and TSO. Every required check
below passed with no exceptions.

### Positive capacity

| Check | DSO | TSO |
|---|---|---|
| `s_rated=0.01 -> kappa=100` (fresh reactivation) | ✅ | ✅ |
| `s_rated=0.02 -> kappa=50` (live capacity change, same model) | ✅ | ✅ |
| Scale is a fixed numeric `Param` value, not a symbolic expression | ✅ | ✅ |
| No division by a decision variable anywhere in the rewritten rule | ✅ | ✅ |

### Zero / near-zero capacity

| Check | DSO | TSO |
|---|---|---|
| Zero capacity → `kappa = 1.0` (safe placeholder), no exception | ✅ | ✅ |
| All 24 `sess_snet_def` rows deactivated | ✅ | ✅ |
| All 9 operational variable groups fixed at zero (`pch/pdch/sch/sdch/pnet/qnet/soc`, both SOC-final slacks) | ✅ | ✅ |
| Near-zero capacity (`s = 5e-11`, below any realistic rating) → still `kappa = 1.0`, no exception, still deactivated | ✅ | ✅ |

### Structure (checked after every step: baseline, near-zero, reactivate, live change, deactivate, reactivate)

| Check | DSO | TSO |
|---|---|---|
| `sess_snet_def` Python component object identity unchanged throughout | ✅ | ✅ |
| Component local name remains `sess_snet_def` | ✅ | ✅ |
| No new constraint component ever created | ✅ | ✅ |
| Total constraint-data count unchanged across every step (`7733` DSO / `3168` TSO) | ✅ | ✅ |
| All `sess_snet_def` index tuples unchanged (24 rows for the target index, both cases) | ✅ | ✅ |
| `sess_comp` untouched | ✅ | ✅ |

### Dual lifecycle (adversarial, with injected stale duals)

| Check | DSO | TSO |
|---|---|---|
| Stale dual (`-777.0`) injected while inactive, then cleared on reactivation (`s=0.01`, `kappa=100`) | ✅ | ✅ |
| Live capacity change on the **same model** (`s: 0.01→0.02`, `kappa: 100→50`) transfers every dual exactly per `dual_new = dual_old·(kappa_old/kappa_new)` — e.g. row `(*,0,0,0)`: `-12.5 → -25.0` (`×2` matches `100/50`); row `(*,0,0,23)`: `-81.5 → -163.0` | ✅ (24/24 rows exact) | ✅ (24/24 rows exact) |
| Deactivating again leaves the last-active dual untouched (not cleared, not stale-injected) | ✅ (24/24 rows) | ✅ (24/24 rows) |
| Reactivating a second time clears that now-stale leftover dual | ✅ | ✅ |

### Feasible-set equivalence (`g = 0 ⟺ κ·g = 0`, `κ = 100`)

| Sample point | `g` | `κ·g` | Zero-crossing matches | Sign matches |
|---|---|---|---|---|
| `sch=0.005, pnet=0.005` (on the constraint surface) | `0.0` | `0.0` | ✅ | ✅ |
| `sch=0.005, pnet=0.0051` | `-1.01e-06` | `-1.01e-04` | ✅ | ✅ |
| `sch=0.006, pnet=0.005` | `1.1e-05` | `1.1e-03` | ✅ | ✅ |

Identical for both DSO and TSO (same rule, same `kappa`). All three points
confirm `g` and `κ·g` are zero at exactly the same point and share sign
everywhere else — the scaled and unscaled constraints define the same
feasible set.

**P4.3 result: every structural invariant, every dual-transfer/lifecycle
rule, and feasible-set equivalence hold exactly, for both DSO and TSO, with
zero exceptions.**

---

## P4.4 — Frozen regression

`required_first_gate.both_decisive_cases_primary_success_no_recovery_within_tolerance: true`.
Because the gate passed, the script proceeded (per plan §6) to replay every
other preserved P3 failure snapshot, plus two matched-success controls not
required by the plan but included to confirm no regression on already-passing
cases.

### Required first gate (decisive cases)

| Case | A (pre-P4, primary) | B (production-normalized, primary) | Gate |
|---|---|---|---|
| `DSO_decisive_Autumn_cycle8` | `error` / `internalSolverError` | `ok` / `optimal`, no recovery | ✅ passed |
| `TSO_decisive_Summer_cycle6` | `warning` / `maxIterations` | `ok` / `optimal`, no recovery | ✅ passed |

### Production-normalization invariants (all 9 cases)

| Check | Result |
|---|---|
| `sess_snet_def` component object identity unchanged | ✅ 9/9 |
| No new constraint component created | ✅ 9/9 |
| Total constraint-data count unchanged | ✅ 9/9 |
| Index tuples / row identities unchanged (24/24 rows each case) | ✅ 9/9 |
| `kappa_new == 100.0` (matches `s_rated=0.01` in every frozen snapshot) | ✅ 9/9 |
| All transformed duals match the KKT rule exactly | ✅ 9/9 (24/24 rows each) |

### A (pre-P4, frozen) vs B (production-normalized) — full replay

| Case | A status/term | A iters (primary→recovery) | B status/term | B iters | Recovery needed | B final residual (normalized) |
|---|---|---|---|---|---|---|
| **DSO Autumn/cycle8** (decisive) | error / internalSolverError | 474 → 449 | **ok / optimal** | **420** | No | `3.21e-14` |
| **TSO Summer/cycle6** (decisive) | warning / maxIterations | 66 → 3000 | **ok / optimal** | **103** | No | `3.84e-12` |
| DSO Summer/cycle1 | warning / maxIterations | 313 → 3000 | **ok / optimal** | **342** | No | `1.12e-09` |
| DSO 2030 Winter/cycle1 | error / internalSolverError | 240 → 2058 | **ok / optimal** | **264** | No | `1.68e-16` |
| DSO Autumn/cycle12 | warning / maxIterations | 218 → 3000 | **ok / optimal** | **264** | No | `1.24e-14` |
| DSO Autumn/cycle13 | warning / maxIterations | 227 → 3000 | **ok / optimal** | **213** | No | `3.58e-10` |
| TSO Winter/cycle5 | warning / maxIterations | 87 → 3000 | **ok / optimal** | **104** | No | `4.51e-10` |
| DSO matched-success/cycle7 (bonus, not required) | ok / optimal (already passing) | 138 | **ok / optimal** (unchanged) | 184 | No | `1.83e-16` |
| TSO matched-success/cycle7 (bonus, not required) | ok / optimal (already passing) | 67 | **ok / optimal** (unchanged) | 107 | No | `1.14e-12` |

Representative detail on the two decisive cases (primary attempt only):

| | DSO Autumn/cycle8 — A primary | DSO Autumn/cycle8 — B | TSO Summer/cycle6 — A primary | TSO Summer/cycle6 — B |
|---|---|---|---|---|
| Dual infeasibility | `5.58e+08` | `9.76e-07` | `3.17e+07` | `6.06e-07` |
| Constraint violation | `4.30e-08` | `7.00e-06` | `4.00e-08` | `3.84e-14` |
| Complementarity | `8.45e-06` | `9.10e-07` | `3.39e-06` | `1.85e-06` |
| CPU (primary) | `4.07s` | `3.91s` | `0.13s` | `0.20s` |

All 7 previously-failing cases (2 decisive + 5 other preserved) now
terminate `optimal` on the primary exact-Hessian attempt, with no recovery
solve invoked. Both bonus matched-success controls remain `optimal` — no
regression on cases that already worked pre-P4.

**P4.4 result: required first gate passed; all 9 replayed cases (7
previously-failing + 2 already-passing controls) succeed cleanly under the
real production code, with zero regressions.**

---

## Interpretation

1. **This is the strongest production-code confirmation obtained in this
   investigation to date.** P4.3 and P4.4 do not exercise a diagnostic
   reimplementation of the kappa-scaling idea (as every P3.5 stage did) —
   they call the actual `sess_snet_def_rule`, `shared_ess_snet_def_scale`,
   and `_sync_sess_snet_def_scale` functions that now live in `network.py` /
   `model_construction_helpers.py` / `definitions.py`, on both freshly built
   models (P4.3) and the historical frozen failure snapshots (P4.4).

2. **Construction correctness is fully established (P4.3).** Positive
   capacity produces the expected fixed numeric `kappa`, zero and near-zero
   capacity route through the safe placeholder with no division-by-zero
   path, structure is provably unchanged (component identity, names, index
   tuples, row counts, `sess_comp`), and `g=0 ⟺ κ·g=0` holds at all sampled
   points. The adversarial stale-dual injection additionally proves the two
   design decisions from P4.2 behave exactly as specified: cleared on
   reactivation, left alone on deactivation — including across a live,
   same-model capacity change with an exact KKT-consistent dual transfer.

3. **Regression correctness is fully established (P4.4).** Both plan-
   mandated decisive cases pass the required first gate (primary success,
   no recovery, physical relation within tolerance), and — because that
   gate passed — every other preserved P3 failure snapshot was replayed and
   also converged cleanly on the primary attempt. This reproduces, under
   the real production code, the same qualitative result the P3.5-D
   diagnostic first demonstrated: the practical local-NLP failures were
   attributable to intrinsic scaling of `sess_snet_def`, not to warm-start
   bookkeeping or to any diagnostic-script artifact.

4. **No regression on already-passing cases.** The two matched-success
   bonus controls (not required by the plan) remain `optimal` under the
   production-normalized path, confirming the kappa-scaling change does not
   destabilize cases that already converged before P4.

5. **No production change is authorized beyond what P4.2 already
   implemented.** P4.3 and P4.4 are validation of the change already made
   under P4.1–P4.2; this report authorizes nothing further on its own. Per
   plan §7, P4.5 (seed-2026 distributed operational smoke) may now proceed,
   pending the planner's confirmation of the exact invocation (candidate
   identified: `python main.py -d SRP1 -f SRP1.json`, running
   `run_operational_planning(type='distributed', candidate_solution=...)`
   against `SRP1.json`'s `RandomSeed: 2026`). No full planning run is
   authorized at this stage.

---

`P4.3–P4.4 COMPLETE — required gates passed, zero regressions — waiting for planner confirmation of P4.5 invocation before proceeding`
