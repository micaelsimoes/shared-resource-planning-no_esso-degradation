# Stage P4.6 — Standard/ordinary ESS normalization audit (executed)

Audit-only stage per `LOCAL_NLP_STABILITY_PLAN.md` §8. **No production code was
modified.** No solve was run. No planning problem was run.

## Provenance

- Executed by: Claude Code session of 2026-09-06, working tree at `a98a228e`
  (tracked tree clean; see "Checkpoint" below).
- Method: static source audit of the production model-construction path plus
  exhaustive inspection of the repository's network-definition data. All
  findings below cite `file:line` in production code or a concrete data file;
  nothing is inferred from the shared-ESS result by analogy.
- Data scan: every `data/**/*.json` network file cross-referenced against every
  spec file's `Years` / `DistributionNetworks` / `TransmissionNetwork` to
  determine which `(spec, network, year)` combinations actually instantiate
  positive-capacity ordinary ESS.
- Raw evidence: reproduced by the commands recorded inline in this report; no
  separate JSON artifact was produced because this stage runs no solver.

## Formulation identification — three distinct ESS families

The plan requires distinguishing the ordinary-ESS network formulation from the
shared-ESS and ESSO formulations. There is a real naming-collision trap here:

| Family | Component / expression | Where built | Namespace |
|---|---|---|---|
| **Ordinary/standard ESS** (P4.6 target) | `model.ess_snet_def` | `network.py:426`, rule `ess_snet_def_rule` (`model_construction_helpers.py:622`) | network SMOPF, vars `es_*` |
| **Shared ESS** (P4.2, already fixed) | `model.sess_snet_def` | `network.py:439`, rule `sess_snet_def_rule` (`model_construction_helpers.py:728`) | network SMOPF, vars `shared_es_*` |
| **ESSO aggregate** (out of scope) | anonymous row in `model.energy_storage_operation_agg` | `shared_energy_storage_data.py:540/542`, built by `_build_subproblem` (`:340`) | **separate** ESSO model, vars `es_*` |

The ESSO model at `shared_energy_storage_data.py:376` declares `model.es_snet`
as a `pe.Var` and uses the `es_*` prefix, **identically named to the ordinary
network ESS variables but belonging to a different model**. The network SMOPF
has no `es_snet` variable at all (`grep -c "model.es_snet" network.py` → 0);
there, `snet` is an inline expression. Any future work must not conflate these.

---

## Audit answers (the nine required questions)

### 1. Exact mathematical form and corresponding rule/component

`model_construction_helpers.py:622-624`:

```python
def ess_snet_def_rule(m, e, s_m, s_o, p):
    snet = m.es_sch[e, s_m, s_o, p] - m.es_sdch[e, s_m, s_o, p]
    return snet ** 2 == m.es_pnet[e, s_m, s_o, p] ** 2 + m.es_qnet[e, s_m, s_o, p] ** 2
```

i.e. with `g_es = (sch − sdch)² − pnet² − qnet²`, the production row is

```
g_es = 0          (unscaled; kappa identically 1)
```

- Component: `model.ess_snet_def`, `network.py:426`, indexed over
  `energy_storages × scenarios_market × scenarios_operation × periods`.
- Row gradient w.r.t. `(sch, sdch, pnet, qnet)`:
  `[2·snet, −2·snet, −2·pnet, −2·qnet]`.
- Whole ordinary-ESS block (variables `network.py:341-351`, constraints
  `network.py:424-435`) is gated by `params.es_reg`.

### 2. Is it mathematically equivalent to shared-ESS `sess_snet_def`?

**Yes — the equality is structurally identical** to the pre-P4 shared-ESS row.
Both define `snet = sch − sdch` and equate `snet²` to `pnet² + qnet²`, giving
the same weakly-scaled gradient row at low dispatch that P3.5-D proved causal.

Two differences were found; **neither affects the equality, its feasible set, or
its conditioning**, but both are recorded so equivalence is verified rather than
assumed:

1. **Opposite `pnet` sign convention.** Ordinary: `es_pnet = es_pdch − es_pch`
   (generation-positive, `model_construction_helpers.py:618-619`). Shared:
   `shared_es_pnet = shared_es_pch − shared_es_pdch` (load-positive, `:789-790`).
   Because `pnet` enters `ess_snet_def` only squared, this is immaterial here.
   It is consistent with results post-processing, which negates `q` for ordinary
   ESS (`network.py:1335`) but not for shared ESS (`network.py:1357`).
2. **Mirrored power-factor limits** (`ess_phi_limits_*` at `:645-658` vs.
   `sess_phi_limits_*` at `:700-713`) follow that same convention.

Note the ordinary family is internally inconsistent in orientation — `pnet` is
generation-positive while `snet = sch − sdch` is charge-positive — but again,
only squared terms reach this row. **Flagged, not touched** (out of P4.6 scope).

The substantive differences between the families are **not in the equation**;
they are in the rating source and lifecycle (Q3–Q6), and they make the ordinary
case *simpler*, with one exception (Q6).

### 3. Rated-power source, and is it fixed during a local solve?

- Source: `network.energy_storages[e].s` — a **plain Python float attribute** on
  the `EnergyStorage` object (`energy_storage.py`), read from the network JSON
  `energy_storages[].s` and converted to p.u. at `network.py:773`
  (`= float(...) / network.baseMVA`).
- It is **not** a Pyomo `Param` and **not** a `Var`. Contrast the shared-ESS
  case, where `model.shared_es_s_rated` is a `pe.Var` (`network.py:357`) and
  `model.shared_es_s_rated_fixed` is a mutable `Param` (`:354`) — the latter
  being what P4.2's kappa tracks.
- Where it enters the model: variable bounds `p_bounds` / `snet_bounds` /
  `q_bounds` / `s_bounds` / `slack_es_balance_bounds`
  (`model_construction_helpers.py:390-413`), `ess_s_limit_rule` (`:635-637`),
  and the bilinear-relaxation branch of `ess_comp_rule` (`:683`).
- It does **not** appear in `ess_snet_def_rule` at all.

**Fixed during a local solve: yes — and more strongly than for shared ESS.** It
is folded into expressions and bounds as a numeric literal at construction time.
There is no mechanism to change it short of rebuilding the model.

### 4. Model rebuild/reuse behavior when ordinary ESS capacity changes

- **Ordinary ESS capacity is not a planning decision variable.** The only
  runtime capacity mutation anywhere in the codebase is on
  `shared_energy_storages` (`network_data.py:2891-2906`), driven by
  `candidate_solution`. A targeted search for assignments to
  `energy_storages[...].s/.e/.e_init/.e_min/.e_max` returns **no runtime
  mutation** of ordinary ESS.
- The only assignment to `.energy_storages` is `centralized_coordination.py:72`,
  which rebuilds the list while *assembling a combined network object* (copies
  units, re-ids, remaps buses) **before** `build_model` is called for that
  network — not a live-model mutation.
- Capacity *does* vary across planning years in the data (e.g.
  `CS7/case33_1`: 0.5 MVA for 2030-2034 → 1.0 MVA for 2035-2039 → **three**
  units from 2040). But each `(year, day)` owns its own `Network` object and its
  own Pyomo model, built from its own JSON. A cross-year capacity change is
  therefore **a different model**, never an in-place update. The index set
  itself changes (1 → 3 units), which could not be applied in place regardless.

**Conclusion: ordinary ESS has no analogue of the shared-ESS "live capacity
change on a reused model" path.** `kappa_es` would be a build-time constant.

### 5. Constraint-object and `model.dual` lifecycle

- `model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)` (`network.py:473`)
  is **model-wide**, so `ess_snet_def` rows participate in IPOPT dual
  import/export and warm-starting exactly like every other row.
- Models are reused across ADMM cycles (established in P4.1), so `ess_snet_def`
  `ConstraintData` objects — and their `model.dual` entries — persist across
  cycles.
- **`ess_snet_def` rows are never activated or deactivated.** Every
  constraint-level `.activate()`/`.deactivate()` in the network model is inside
  shared-ESS-only helpers (`model_construction_helpers.py:877/879` in
  `_configure_shared_ess_expected_schedule`, and `:926/928` in
  `configure_shared_ess_operational_state`), both iterating
  `_SHARED_ESS_OPERATIONAL_CONSTRAINTS` (`:805-818`), which contains only
  `sess_*` names. The remaining pair (`shared_energy_storage_data.py:1086/1088`)
  belongs to the ESSO model.

**Conclusion: the ordinary-ESS dual lifecycle is strictly simpler** — rows are
permanently active for the life of the model, duals persist, and no
active↔inactive state machine exists.

### 6. Zero-capacity behavior / gating — **the one material asymmetry**

**There is no zero-capacity gating for ordinary ESS.** No
`configure_ordinary_ess_operational_state` exists; the only such helpers are
shared-ESS. "Absence" is expressed structurally, in two ways:

1. `params.es_reg == False` → no ordinary ESS variables or constraints are
   created at all (`network.py:341`, `:424`).
2. The network JSON has no `energy_storages` key (or an empty list) →
   `network.energy_storages == []` → `model.energy_storages = range(0)`
   (`network.py:270`) → `ess_snet_def` has **zero rows**.

Case (2) is the SRP1 situation: `es_reg` is `True` in SRP1's per-network params,
but **every** SRP1 network file lacks the `energy_storages` key. This is
precisely why P4.1–P4.5 never touched this row (see Q9).

**The asymmetry that matters:** a unit present in data with `s == 0` would
**not** be gated. Its bounds would collapse to `[0, 0]` (`s_bounds` →
`(0.0, 0 + EQUALITY_TOLERANCE)`) but the row would **remain active and be
solved**. My scan found no such `s == 0` entry today, but this means:

> For shared ESS, the `SHARED_ESS_SNET_DEF_SAFE_KAPPA` placeholder is provably
> inconsequential because the row is deactivated in that regime. **For ordinary
> ESS that argument does not hold.** A divide-by-zero guard would be
> load-bearing, not cosmetic, and any implementation must handle it explicitly
> rather than inheriting the shared-ESS justification.

### 7. Is `kappa_es = 1 / S_scale_es` mathematically appropriate?

**Yes, mathematically.** For any finite positive `kappa`, `g = 0 ⟺ kappa·g = 0`;
the feasible set is unchanged. It is a constant-multiple row scaling, exactly as
validated for shared ESS in P4.3. The rating is a genuine construction-time
constant (Q3), so no symbolic division by a decision variable is involved —
satisfying plan §2's prohibition even more directly than the shared-ESS case.

**Numerically the motivation is at least as strong.** The one active case
(Q9) has ordinary ESS at `s = 0.5 MVA` on `baseMVA = 100` → **0.005 p.u.** →
`kappa_es = 200` — i.e. the row is **twice as weakly scaled** as the shared-ESS
row (`0.01 p.u.`, `kappa = 100`) whose poor conditioning P3.5-D proved causal.

Two design caveats:

- **Per-unit indexing is required.** A network may carry several ordinary ESS
  units with different ratings (`BCK/HR1_bck/A_KPC_35_2`: 8 units spanning
  0.04–0.56 p.u.). `kappa_es` must be indexed by `e`, as the shared
  implementation is; a single global scale would be wrong.
- **Semantics differ slightly.** The ordinary rating is a *nameplate* value from
  data; the shared-ESS scale tracks the *installed candidate* capacity. Both are
  the correct characteristic power for their own row, but they are not the same
  kind of quantity.

### 8. Does a scale change on a reused model require `lambda_new = lambda_old * (kappa_old / kappa_new)`?

**No — not for ordinary ESS.** On the Q3–Q5 evidence, `kappa_es` **cannot change
on a live model**: the rating is a construction-time constant, is never mutated
at runtime, and per-year capacity changes produce different model objects. There
is no active↔inactive transition either.

Therefore the entire `_sync_sess_snet_def_scale` machinery
(`model_construction_helpers.py:942-970`) — the most delicate part of P4.2 —
has **no required counterpart** for ordinary ESS. `kappa_es` can be an immutable
build-time constant folded into the rule at construction.

Per plan §3/§4 ("Do not add dual remapping unless the audit proves it is
needed"), **do not port the remapping mechanism.**

> **Recorded dependency:** this conclusion holds *only while* ordinary ESS
> capacity remains a fixed data input. If ordinary ESS ever becomes a planning
> or live-updated quantity, Q4/Q5/Q8 must be re-audited before trusting a
> constant `kappa_es`.

### 9. Does an active positive-capacity ordinary ESS case exist?

**Yes — exactly one, in the live (non-`BCK/`) data tree:**

| | |
|---|---|
| Spec | `data/OP1/OP1.json` (Years `['2025']`, TN `case9`, DNs `case33_1`@5, `case33_2`@7, **`case33_3`@9**) |
| Network / year | `case33_3` / **2025** (`data/OP1/case33_3/case33_3_2025.json`) |
| Units | `es_id=1` @ bus 23, `es_id=2` @ bus 18 |
| Rating | `s = 0.5 MVA` each → **0.005 p.u.** (`baseMVA = 100`) → `kappa_es = 200` |
| Energy | `e = 1.0 MWh` each |
| Params | `es_reg = True`, `ess_model = EXACT`, `slacks.ess.day_balance = False` |

Within OP1, only `case33_3` carries ordinary ESS; `case33_1` and `case33_2` do
not.

**`CS7/case33_1` — reviewer-identified, but not currently instantiated.** It
does define ordinary ESS from 2030 onward (0.5 MVA 2030-2034; 1.0 MVA 2035-2039;
three units 1.0/0.5/0.5 MVA from 2040), as the planner noted. However
`data/CS7/CS7.json` currently specifies `"Years": {"2025": 1}`, and
`case33_1_2025.json` has **no** `energy_storages` key. **As currently specified,
CS7 instantiates zero ordinary-ESS rows.** It becomes a usable — and richer —
validation case only if its spec years are extended to ≥ 2030, which is a
data/spec change requiring planner authorization.

All other positive-capacity ordinary ESS lives under `data/BCK/` (archive):
`BCK/HR1_bck` (2030-2049, up to 8 units/network), `BCK/IEEE9` (2030/2035/2040),
`BCK/CS2`, `BCK/OP3`.

**Critical scope finding:** `SRP1` — the case on which all of P1–P4.5 ran — has
**no ordinary ESS in any of its network files**, despite `es_reg = True`.
`ess_snet_def` therefore had **zero rows in every solve performed in this
investigation to date**. The standard-ESS row is entirely unexercised by the
accumulated evidence, and **P4.5's result says nothing whatsoever about it.**

---

## Required-proof table

| Plan §8 audit item | Status | Evidence |
|---|---|---|
| Exact standard-ESS relation | Established | `model_construction_helpers.py:622-624` |
| Equivalent to `sess_snet_def`? | **Verified equivalent** (2 immaterial convention differences recorded) | Q2 |
| Rated-power variable/parameter semantics | Established — plain Python float, construction-time constant | `network.py:773`, `energy_storage.py`, bounds `:390-413` |
| Zero-capacity behavior | **Established — no gating exists; asymmetric to shared ESS** | Q6 |
| Model rebuild/reuse lifecycle | Established — no live capacity change possible | `network_data.py:2891-2906`, `centralized_coordination.py:72` |
| Imported constraint-dual lifecycle | Established — model-wide suffix, rows permanently active | `network.py:473`; activate/deactivate audit Q5 |
| Warm-start behavior | Established — same IMPORT_EXPORT path as every other row | `network.py:469-473` |
| Indexing | Established — `energy_storages × s_m × s_o × periods` | `network.py:426`, `:270` |
| Existing active standard-ESS cases/tests | **One found: OP1/case33_3/2025** | Q9 |
| Production code modified | **None** | tracked tree clean |

---

## Proposed smallest validation experiment (proposed only — NOT implemented)

Target: **OP1 / `case33_3` / 2025** (DSO-local, reduced operational scope).

**Stage 1 — construction/equivalence, no solve** (P4.3 pattern). Build the OP1
`case33_3` 2025 DSO model through the real `build_model` path and, calling real
production functions only:

- snapshot `ess_snet_def` component `id()`, local name, full index-tuple set and
  row count, plus whole-model constraint-data count;
- confirm 2 units × scenarios × periods rows, and that the rating enters only
  bounds / `ess_s_limit` / `ess_comp`, never the `snet_def` row;
- confirm `kappa_es = 1/0.005 = 200` per unit, and per-unit indexing;
- confirm the zero-capacity guard returns a finite placeholder with **no**
  divide-by-zero, and — given Q6 — explicitly characterise what the model does
  with an `s = 0` unit, since the row is *not* deactivated;
- confirm feasible-set equivalence `g = 0 ⟺ kappa·g = 0` at sample points;
- confirm `ess_comp`, SOC, bounds and every shared-ESS component are untouched.

**Stage 2 — representative local solve** (P4.4 pattern). Baseline vs. κ-scaled
solve of the same local SMOPF from an identical frozen starting point (fresh
`.clone()` per branch, SHA-256 verified), reporting per branch: status /
termination, primary iterations, objective, scaled & unscaled primal
infeasibility, dual infeasibility, complementarity, runtime, recovery yes/no,
and the **original unscaled** `ess_snet_def` residual normalized by `s²`.

Gate: κ-scaled must be a primary success with the unscaled physical relation
within normal feasibility tolerance, and must show no regression vs. baseline.

**Honest limitation.** OP1 has **no captured failure** in this row. This is
therefore a *no-regression + conditioning-improvement* test, **not** a
failure-repair test — strictly weaker evidence than P4.4 had for shared ESS. If
the planner wants failure-repair-grade evidence, extending the CS7 spec to
≥ 2030 (multi-year, multi-unit, capacity-varying) is the stronger route.

**Environment note (updates `docs/METHODOLOGY.md`).** Unlike the prior session,
this machine *can* run these scripts directly: Pyomo **6.9.5** is available at
`/opt/anaconda3/envs/opf_env_py311/bin/python` (the PyCharm interpreter
`opf_env_py311`), and an `ipopt` binary is on `PATH` at `/usr/local/bin/ipopt`.
MA97 availability still needs confirmation via `NLP_SOLVER_PATH` in `.env`
before any solve is trusted. The default `python3` on `PATH` has **no** Pyomo —
scripts must be run with the `opf_env_py311` interpreter explicitly.

---

## Recommendation — implement now vs. defer

Plan §8's "if no active standard-ESS test exists, return a design
recommendation" branch **does not strictly apply**, because an active
positive-capacity case does exist (OP1/case33_3/2025). Weighing it:

**For proceeding:** the equation is verified equivalent; the lifecycle is
strictly simpler (no live κ change, no dual remapping, no activation state
machine); the numerical motivation is stronger than for shared ESS (κ = 200 vs.
100); an active case exists.

**For caution:** the OP1 case yields no-regression evidence only, not
failure-repair; OP1 has never been run in this investigation, so its baseline is
unknown; and the zero-capacity guard is load-bearing here in a way it was not
for shared ESS (Q6), so the P4.2 justification cannot simply be inherited.

**Recommendation:** proceed to a **scoped P4.6-B implementation + validation
substage on OP1/case33_3/2025**, with the zero-capacity guard treated as a
first-class requirement rather than a placeholder, and with `kappa_es` as an
immutable per-unit build-time constant (no dual-remapping machinery). Establish
the OP1 baseline first. If the planner prefers failure-repair-grade evidence,
authorize extending the CS7 spec to ≥ 2030 as a stronger follow-up case.

**No production change is authorized by this report.** Per plan §8 and the
standing prohibitions, `ess_snet_def`, `ess_comp`, SOC, degradation, shared-ESS
equations, solver settings, ADMM settings and planning logic remain untouched,
and the next substage is gated on planner review.

---

## Checkpoint status (step 1 of this session's instruction)

No new commit was required or created. The validated shared-ESS production
normalization and all its diagnostic/validation scripts were **already
committed**, and the tracked working tree is **clean**:

- `0171f451` — P4.2 production fix (`definitions.py`, `model_construction_helpers.py`,
  `network.py`) + P3.5/P4.3/P4.4/P4.5 scripts + stage reports + raw JSON evidence.
- `a98a228e` (HEAD) — continuity docs (`CLAUDE.md`, `docs/METHODOLOGY.md`,
  `docs/P4_PROGRESS.md`).

`git status --porcelain --untracked-files=no` → empty. The only untracked paths
are result/diagram output directories, frozen pickles, `.env`, `.DS_Store` and
`__pycache__` — all correctly excluded, none staged. An empty commit was
deliberately **not** fabricated.

```
P4.6 AUDIT COMPLETE — waiting for planner review
```
