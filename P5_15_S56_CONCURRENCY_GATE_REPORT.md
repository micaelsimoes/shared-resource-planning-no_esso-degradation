# P5.15 S56: the serial-vs-concurrent equivalence gate, v1 (2026-11-21)

**Spec:** `frozen_concurrency_gate_spec_v1_0dce2472.json`. The spec and script were committed in `2684dcbf` before the
launch.

**What ran:** the published SRP1 single-scenario configuration (tail on, baseline ageing, 1×), with the harness
unchanged (sha `d9a944f2`) and a 3-cycle cap.
- **Root A:** the reference unit, run alone (concurrency 1).
- **Root B:** the same unit plus four other 2025 plans, run concurrently (5).

**Solves:** 204 per child on each of the 6 children, 0 retries. The parent guard counted 0; pickle loads 0.

## Decisions needed

None. Under the frozen rule the gate is recorded as **FAILED (G3)**, with its mechanism, and it is **not re-scoped**:
- **The failing fields:** the two byte counts of the interface-dual capture file. The only differing value in that file
  is `capture_s`, a wall time (`interface_dual_capture.py:180`). The Planner verified this independently: across the
  shared candidate's 4 lines, `capture_s` is the only differing key on cycles 1–3.
- **The fix:** the next equivalence gate, on the stochastic instance, declares wall-time-derived fields (`capture_s` and
  the byte counts that depend on it) as excluded **before** its run. This follows the precedent of Addendum 50's G6,
  option (a).
- **The numerical equivalence did hold:**
  - **G1:** per-cycle trajectory, per-cycle record and AA, bitwise.
  - **G2:** 192 of 192 IPOPT solve records and every final statistic, bitwise.
  - **G4:** solve counts.
  - **The supplementary check:** all 60 of 60 IPOPT logs are identical apart from their timing lines.

## Found

| prediction (recorded in the spec) | outcome |
|---|---|
| P1: the shared candidate's per-cycle record is bitwise identical in A and B | **not held under the frozen rule** (G3, a wall-time-derived byte count); the per-cycle record itself is bitwise identical (G1, G2) |
| P2: contention 1.12–1.22× per cycle | **held**: 1.183 (by cycle 1.159 / 1.182 / 1.207); the other B children were at 1.087–1.239 |
| P3: peak RSS per SRP1 child 2.3–3.5 GiB | **failed low**: 1.94–2.13 GiB at 3 cycles. For memory margins, use the full-run peaks of 2.38–2.46 GiB (W187) |

- **Root wall time:** 135 s (A) against 160 s (B), for 1 and 5 evaluations respectively. Candidate-level parallelism
  on this machine therefore delivers about 5 / 1.18 ≈ 4.2× throughput at 5-way.
- **Memory available during B:** 10.2 GiB at its lowest. The preflight passed: 15.2 GiB available against 13.75 GiB
  required.
- **Historical cross-check (not gated):** cycles 1–3 match the committed W86 (unit) and W187 (x0) runs bitwise.
- **A configuration note:** the published s47/s51 search specs did not declare the tail, so production's default (off)
  applied. This gate used tail on, as W86 and W187 did.

## Not confirmed

- **Whether 3-cycle peaks represent full runs.** They do not, as P3 shows.
- **The differing ESSO pickle bytes:** read as `id()`-based keys in Pyomo `ComponentSet`s. Only the first 3 differing
  runs were inspected.
- **Concurrency at multi-scenario scale.** Not tested; SP runs at concurrency 1.

**Committed:** `gate_result.json`, the manifest, the launch logs and every JSON, JSONL, log and TXT file of both roots.
The six ESSO `.pkl` files are hashed in the manifest and not committed.
