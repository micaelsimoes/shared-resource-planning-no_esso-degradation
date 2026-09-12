# P5.13-D — Step 0: activate calibration C3, plus two determinism fixes

**Final verdict: gate v1 FAILED as frozen; the change was REVERTED, gate v2 was frozen,
and the change was re-applied and re-run. Gate v2 PASSES on every expectation.**

The sequence below is deliberately preserved in full — the v1 failure, the revert, the
re-registration and the v2 pass — because the reason v1 failed is a real error of mine
and the remedy is the point of the stage.

Frozen gate: `data/SRP1/Results/P513D/frozen_c3_impact_gate_v1_8ee17c92.json`
(SHA-256 `8ee17c925a6917d3efb345b4bbe35067cacc6f84f04c19cbf81dd05645841949`), written
and hashed before any edit.

## 1. The error, stated first

The frozen gate says "No solver is invoked" and lists "any solve of any kind" under
`not_permitted`. **That premise was false, and I carried it over from P5.13-C without
tracing the call.** `create_shared_energy_storage_model` calls
`shared_ess_data.optimize(...)` (`shared_resources_planning.py:3156`), which reaches
`_optimize` and `_run_solver_attempt` (`shared_energy_storage_data.py:947-...`): **three
IPOPT solves per capture, one per ESSO node.**

Two consequences, both recorded rather than absorbed:

1. **P5.13-C's report and its `REVISION_CONTEXT.md` entry claimed "no solve".** That
   claim is corrected in place. The evidence there is unaffected — it is in fact broader
   than described, since the compared state includes post-solve variable values — but a
   committed report asserted a property I had not verified.
2. **My own frozen gate forbade solves, and my harness performed them.** The standing
   rule is "no solves outside an authorized stage". This stage was authorized, so the
   breach is of my specification rather than of the authorization; it is still a breach,
   and it is the direct cause of the E1 mis-specification below.

## 2. What the gate measured

| Expectation | Result |
|---|---|
| **E1** impact confined to `energy_storage_capacity_degradation` | **FAIL as frozen** — that component changed as intended, and 19 `vars.*` components also changed |
| **E2** changed rows carry `2k = 23083.120654223414` instead of `2·cl_nom = 20000` | **PASS** — 6 of 30 rows differ, all 6 carry `2k`, none still carries `20000` |
| **E3** `k = N·D/(−ln R) = 11541.560327111707` | **PASS** |
| **E4** determinism of post-change captures | inherited from P5.13-C's control; see E5 |
| **E5** reverting `status` to `DECLARED_NOT_CONSUMED` reproduces the pre-change model | **PASS, byte-identical** |
| **E6** consistency guard `cycles_n == cl_nom`, `reference_dod_d == dod_nom` | **PASS** — implemented, raises otherwise |
| **E7** exported `_in` key sets equal `_out` key sets | **PASS** — stale entry dropped, value overwritten |
| **E8** `fixed_variable_treatment` pinned on both paths, configuration still wins | **PASS** |

Sample row, before and after:

```
pre : es_degradation_per_unit[0,0]*(20000*es_e_rated_per_unit[0,0])              == es_avg_ch_dch_per_unit[0,0]
post: es_degradation_per_unit[0,0]*(23083.120654223414*es_e_rated_per_unit[0,0]) == es_avg_ch_dch_per_unit[0,0]
```

### Why E1 failed, and what the 19 components actually are

They are variable **values** — not bounds, domains, fixed flags or structure. Examples:
`es_soh_per_unit[0,0]` `0.9999906477302464 -> 0.9999918967589592`;
`es_pch_per_unit[0,0,0,0]` `8.240421280126113e-05 -> 8.24054076211846e-05`;
`slack_es_pnet_up[0,0,0]` `-9.090909080902814e-09 -> -9.090909080824915e-09`.

Because the capture solves, the recorded values are **post-solve** values. A change to
the degradation constant changes the solution, so those values must move. E1 demanded
byte-identical variables, which is impossible for any capture that solves. The
invariant was written on the false premise of section 1.

**The substantive finding stands and is the one the stage wanted:** the structural
impact of C3 is confined to the six degradation rows, and the only thing that changed
there is the constant, from `2·cl_nom` to `2k` — a factor of `1.1541560327111706`.

### E5 proves more than reversibility

The reversion capture ran with the **new** code — `cl_eff`, the pinned
`fixed_variable_treatment` and the warm-start assignment all in place — and only the
JSON `status` reverted. It reproduced the pre-change capture **byte-identically**,
including `.nl` bytes and post-solve values, against a baseline captured with the old
code. So the two determinism fixes are **empirically neutral on this path**, not merely
neutral by argument, and the C3 activation is exactly reversible through one JSON field.

## 3. The three changes

**C3 activation.** `(N, D, R) = (10000, 0.80, 0.50)`, `k = 11541.560327111707`. The law
consumes `shared_energy_storage.cl_eff`, which is `k` when the calibration is `ACTIVE`
and `cl_nom` otherwise. A consistency guard raises unless `cycles_n == cl_nom` and
`reference_dod_d == dod_nom` — this is what stops the count and the depth drifting apart
again, which was the P5.13-B primary finding. `dod_nom` is now load-bearing.

**`fixed_variable_treatment` pinned.** To `make_parameter`, read from this machine's
binary (`ipopt --print-options` on IPOPT 3.14.18, ASL 20241111) rather than from
memory or documentation, so the pin equals the current default and is neutral today.
It is applied *before* configuration on both paths, so a case file can still override it.

**Warm-start merge replaced by assignment.** Both sites — `network.py:516-517` and
`shared_energy_storage_data.py:889-890` — now call
`helper_functions.replace_warm_start_suffix`, which clears before updating. The former
`_in.update(_out)` left multipliers in place for variables absent from the current
`_out`: the equal-bound variables IPOPT removes under `make_parameter`, whose stale
values were exported to the `.nl` and then ignored. With the option pinned, assignment
is inert today and removes the latent path entirely.

## 4. Status and what I recommend

Nothing from this stage is committed to production. The working tree carries the change;
the record carries the failed gate. Two defensible options:

- **Accept under a corrected gate v2**, which separates *structure* (must be confined to
  the degradation rows) from *values* (expected to move, and quantified). v2 would be
  written after seeing the data, which is weaker than a pre-registered gate, and should
  be labelled as such.
- **Revert and re-run** under a pre-registered v2 with a solve-aware invariant. This
  costs three captures (a few minutes each) and preserves the discipline exactly.

I recommend the second. The evidence would be identical, and the difference is precisely
the property this project's gates exist to protect.


---

# Gate v2 — pre-registered, solve-aware, PASSED

`data/SRP1/Results/P513D/frozen_c3_impact_gate_v2_afc863a1.json`
(SHA-256 `afc863a1bbf0d5b6f7ad186a8f99c74d0aae966cb165f88474b27fe9381028ab`),
predecessor `8ee17c92…1949`.

**The working tree was reverted to the committed state before v2 was written**, and the
change was re-applied from a preserved patch (`p513d_step0.patch`, SHA-256
`9656b344db174b9426de3a8c5695c5dfb5c5933a83b4fe257d80dd1b5ba35920`) only afterwards, so
the re-run was not conducted against a gate written to fit it. One disclosure is on the
face of the specification: the v1 captures had already been observed, so the component
list in E1c is not blind. It is fixed before re-application, and its *quantitative*
expectations are derived from the algebra of the law rather than from the observed
numbers.

## The solve profile — specified, not denied

The v1 error was writing "no solve" for a path that solves. v2 declares the profile and
arms it: `p513_solve_profile_guard.SolveProfileGuard` permits solves reached through
`shared_energy_storage_data._run_solver_attempt`, counts them, and raises on any other
call site.

| Capture | `OptSolver.solve` | `_execute_command` | blocked | declared |
|---|---|---|---|---|
| `v2_pre` (reverted tree) | 3 | 3 | 0 | 3 / 3 |
| `v2_post` (C3 active) | 3 | 3 | 0 | 3 / 3 |
| `v2_post2` (determinism) | 3 | 3 | 0 | 3 / 3 |
| `v2_revert` (status reverted, new code) | 3 | 3 | 0 | 3 / 3 |

Exact-count matching in both directions. The launch count equalling the solve count also
shows **no recovery retry fired**: every ESSO solve converged on its first attempt.

## Results

| Expectation | Result |
|---|---|
| **E1a** `.nl` differs pre vs post; identical for determinism and reversion | **PASS** — `961f81fd…` (pre) vs `119b3868…` (post); `v2_revert` returns to `961f81fd…` |
| **E1b** structure confined to the degradation rows | **PASS** — sole differing component `constraints.energy_storage_capacity_degradation`, 6 of 30 rows |
| **E1c** values move only in the 19 named components | **PASS** — no unexpected movers |
| **E1c** ratio `es_degradation_per_unit` ≈ `cl_nom/k` | **PASS** — predicted `0.8664339757`, observed **`0.8664416996`** |
| **E1c** worst relative move ≤ 20% | **PASS** — `es_degradation_per_unit[(2,2)]` at `13.356%`, which is `1 − 0.8664` |
| **E2** rows carry `2k`, none carries `20000` | **PASS** |
| **E3** `k = 11541.560327111707` | **PASS** |
| **E4** two captures of the same tree identical | **PASS** — `v2_post` vs `v2_post2`, state and `.nl` |
| **E5** reversion reproduces the baseline byte-identically | **PASS** |
| **E6/E7/E8** guard, warm start, option pin | **PASS** |

The ratio agreement is the strongest single number here. The law gives
`δ = throughput/(2kE)`, so for unchanged throughput `δ` must scale by `cl_nom/k`. The
pre-registered prediction `0.8664339757` and the observed `0.8664416996` agree to five
significant figures; the residual is the ESSO re-optimizing throughput slightly.

## E5, restated at its proper weight

`v2_revert` ran with the **new** code — `cl_eff`, the pinned `fixed_variable_treatment`,
the warm-start assignment — with only the JSON `status` reverted, and reproduced the
`v2_pre` capture **byte-identically**, including `.nl` bytes and post-solve values, on a
path that performs three solves. Its `.nl` hash `961f81fd…` also equals the P5.13-C
baseline captured days earlier under the old code.

So the two determinism fixes are **empirically neutral, not neutral by argument**, and
the C3 activation is exactly reversible through one JSON field. That result was obtained
under v1 and is unaffected by v1's mis-specification.

## What the `cl_eff` guard closes

`effective_cycle_constant()` raises unless `cycles_n == cl_nom` and
`reference_dod_d == dod_nom`. The count and the depth can no longer drift apart
silently, which was the actual defect behind the whole calibration episode (P5.13-B, F0:
`cl_nom` untouched since 2024-04-08 while `dod_nom` moved twice). `dod_nom` is now
load-bearing rather than vestigial.
