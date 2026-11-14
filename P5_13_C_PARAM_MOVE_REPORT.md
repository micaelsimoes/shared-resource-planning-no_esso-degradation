# P5.13-C — relocating the ESS ageing constants into the parameters file

**Behaviour-preserving refactor. Gate: PASS.**

> **CORRECTION (2026-09-12, from P5.13-D).** This report originally stated that no
> solve was run. **That was false.** The capture path calls
> `create_shared_energy_storage_model`, which calls `shared_ess_data.optimize(...)`
> (`shared_resources_planning.py:3156`) and therefore performs **three IPOPT solves,
> one per ESSO node, on every capture**. The error was mine: I asserted the property
> instead of tracing the call. See the corrected limits section at the end.

Date: 2026-09-12. Branch `feature/derivative-free-planning`, baseline HEAD
`8a25713b`. Authorized by the author as the first production code change in this
sequence, under four conditions: defaults reproduce current values exactly; neutrality
is *proved* rather than asserted; no numerical behaviour, bound or constraint changes;
and the calibration triple appears in the file schema. The Worker was unavailable, so
the Planner is both author and verifier — which is why the gate was frozen before the
edit and its verdict is mechanical.

## 1. The gate, frozen before the edit

`data/SRP1/Results/P513C/frozen_param_move_gate_v1_bd8ab535.json`
(SHA-256 `bd8ab535b76a5deb6097f41492dc14970058c8671df744e132c3fcd007da60bb`), written and
hashed **before** any source file was touched.

Invariants: **I1** ordered semantic state of each ESSO model (variables with bounds,
fixed flags, values and domains; Params; every constraint expression as a string);
**I2** `.nl` export bytes; **I3** the four ageing constants on every
`SharedEnergyStorage` object in every representative year; **I4** objective,
feasibility-penalty and salvage expression strings; **I5** the diff is confined to the
permitted files. Decision rule: any inequality fails, with no argument that a difference
is immaterial.

Harness `p513_c_param_move_gate.py`, SHA-256
`99cae59ef67bf633cdb2779117602830519a0c5a51bf33c434b7f74ffa6ab122`. It builds the three ESSO models through the production path already used by
`p54c_esso_active_energy_validation.py` (`read_planning_problem` ->
`_build_positive_bootstrap_candidate` -> `create_admm_variables` ->
`create_shared_energy_storage_model`). **That last call solves**: three IPOPT solves
per capture, one per node (correction of 2026-09-12).

## 2. Three runs, not one

| Run | Purpose | Result |
|---|---|---|
| `pre` vs `pre2` | **determinism control** — two captures of the *unmodified* tree | PASS: identical, including `.nl` bytes **and post-solve variable values**, so the ESSO solves are bit-reproducible on this machine. Without this the gate would prove nothing. |
| `pre` vs `post_final` | **neutrality** — the actual claim | **PASS on I1–I4** |
| `pre` vs `probe` | **negative control** — `cycle_life_nominal` 10000 -> 12345 and `minimum_soh` 0.50 -> 0.55 in the JSON, then reverted | **FAIL, as required**: I1, I2, I3 and the salvage expression all differ |

The negative control is the point of the exercise. A neutrality gate passes trivially if
the new code never executes; the probe proves the parameters file is genuinely
load-bearing — the JSON value reaches every `SharedEnergyStorage` object in all three
years and propagates into the model and its `.nl` export. The probe was reverted and
neutrality re-verified afterwards (`post_final`).

Note which expression moved under the probe: `salvage_value` changed, `objective` and
`feasibility_penalty` did not. That is consistent with the ESSO operational objective
being the feasibility penalty alone, with salvage entering the outer recourse.

## 3. What changed

| File | Change |
|---|---|
| `shared_energy_storage_parameters.py` | new `EnergyStorageAgeingParameters` (defaults `15`, `10000`, `0.80`, `0.50` — identical to the former hard-coded values) and `DegradationCalibrationParameters`; both wired into `SharedEnergyStorageParameters` |
| `shared_energy_storage_data.py` | `create_shared_energy_storages` now calls `self.params.ageing.apply_to(...)`; `read_parameters_from_file` runs on the immediately preceding line (`shared_resources_planning.py:6104-6105`), the only call site of each |
| `data/SRP1/SharedESS/SRP1_ESS_Params.json` | 13-line `ageing` block inserted, original tab formatting preserved (a first attempt reformatted the whole file and was discarded) |
| `shared_energy_storage.py` | **comment only** — see the I5 deviation below |

Type preservation is deliberate: `_read_optional_number` keeps a JSON integer an
integer, because `cl_nom` enters a constraint expression and coercing `10000` to
`10000.0` would change the rendered model and the `.nl` bytes. This is the mechanism by
which a "neutral" refactor most easily stops being neutral.

### I5 — a disclosed deviation from the frozen file list

The frozen gate permitted three files. A fourth, `shared_energy_storage.py`, was also
touched, with a **comment only**: five lines marking the four constants as fallback
defaults overwritten from the parameters file. The diff contains no changed assignment
(verified: no `+`/`-` line in that file contains `=`). The rationale is that leaving the
literals unannotated invites someone to edit the now-inert copy. It is nonetheless
outside what was frozen, and is recorded as a deviation rather than being absorbed by
widening the specification after the fact. Reverting it costs nothing if the author
prefers the gate held strictly.

## 4. The calibration triple — declared, not consumed

The schema carries `ageing.calibration` with `status`, `cycles_n`, `reference_dod_d` and
`eol_retention_r`, the last `null` because the `(N, D, R)` decision is open. Nothing in
the model construction path reads it. `characteristic_constant()` implements
`k = N·D/(−ln R)` and is called nowhere; any `status` other than
`DECLARED_NOT_CONSUMED` raises `NotImplementedError` naming the pending decision.

This is deliberate. Consuming the triple changes the degradation term and therefore the
objective — the very thing the author reserved — so it cannot ride along inside a
refactor advertised as neutral. The schema is prepared; the activation is a separate,
openly non-neutral change.

## 5. Evidence inventory

| Artifact | SHA-256 |
|---|---|
| `p513_c_param_move_gate.py` | `99cae59e…b122` |
| `frozen_param_move_gate_v1_bd8ab535.json` | `bd8ab535…60bb` |
| `manifest_pre.json` | `481184f9…7be4` |
| `manifest_pre2.json` | `a5111a9b…1df7` |
| `manifest_probe.json` | `d6bd9b3e…c28c` |
| `manifest_post_final.json` | `5080fe32…57e5` |
| `verdict.json` | `4bbb829e…145e` |

Per-node model-state hash (identical pre and post, all three nodes):
`bcd2d9e9…28a9d`; `.nl` hash `961f81fd…e024`. The 15 ordered-state files
(1.4 MB each, 20 MB total) are left on disk and **hash-recorded** in the manifests
rather than committed, per the artifact rule's "committed or hash-recorded" clause.

## 6. Limits — what this gate does and does not prove

- It proves the model handed to the solver is **byte-identical**, which is exactly the
  property a behaviour-preserving relocation claims. It therefore implies that every
  existing result remains valid unchanged **without re-running any of them**.
- **Corrected.** It does run solves — three per capture. What the comparison covers is
  therefore *larger* than first claimed: the captured state includes post-solve variable
  values, so `pre` vs `post_final` equality shows the ESSO solves returned identical
  results as well as identical models. What was wrong is the description, not the
  evidence. The cost of the error is that "no solve" appeared in a committed report and
  in a frozen gate written on that premise (P5.13-D, E1), where it caused a genuine
  mis-specification.
- **Coverage is narrower than "three models" suggests.** All three ESSO models produced
  identical I1 and I2 hashes, so the gate exercises one structure replicated three
  times, not three independent structures. It covers the ESSO models only — the surface
  on which `t_cal`, `cl_nom` and `soh_min` act (`shared_energy_storage_data.py`); those
  constants appear in no other module.
- It says nothing about the calibration decision, which remains open and blocking.

**P5.13-C PASS — the relocation is structurally neutral, and the parameters file is
proved load-bearing.**
