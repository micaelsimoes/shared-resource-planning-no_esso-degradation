# P5.15 — G1 blocked by an infrastructure defect. G3-init PASSES. Authorization needed.

**Planner report. G1, G2, G4 not obtained. `REVISION_CONTEXT.md` rewrite held.**

## G3 initialization stage — PASS

```
[ladder s=1    e=4   ] all_ok=True failed=[] solves=51 wall=41s
[ladder s=1.25 e=5   ] all_ok=True failed=[] solves=51 wall=30s
[ladder s=1.62 e=6.48] all_ok=True failed=[] solves=51 wall=30s
```

All three required rungs pass, zero failed nodes. **Attribution caveat:** the pre-reformulation
failure at 1.00 MVA was on the capacity rows (`rated_s_capacity_unit: 6.845e-05`) with
`energy_storage_normalization: 0.0`. This pass is attributable to Candidate 2 (investments as
parameters), **not** to the complementarity deletion.

## G1 — two failed attempts, root cause identified

**Attempt 1** died silently: 0-byte log, no artifact. Cause: it was run concurrently with the
G3-init ladder from the repo root, and both wrote into the same `optim_log_node_*.txt`.

**Attempt 2** (relaunched by the Planner under `nohup`, both streams captured) crashed with:

```
File "shared_resources_planning.py", line 4445, in save_failed_tso_block
File "shared_resources_planning.py", line 4522, in _save_frozen_network_block
    os.makedirs(save_dir, exist_ok=True)
FileNotFoundError: [Errno 2] No such file or directory: 'data'
```

### The causal chain

1. `shared_energy_storage_data.py:1008` writes the ESSO's IPOPT log to a **bare relative
   filename** (`optim_log_node_{id}.txt`) with `file_append='yes'` and **no `logs_dir`
   awareness**. `network.py:520-521` already resolves `output_file` against `network.logs_dir`.
   **The ESSO is the only solver family without log isolation.**
2. Because of (1), every P5.15 harness must `os.chdir` into a private directory to parse its own
   logs — otherwise `file_append` concatenates every cycle into one file and
   `_parse_ipopt_barrier_terms` (first match, not last) returns cycle 1's `mu_final` forever.
   The per-cycle detector Addendum 4 requires **cannot be captured without this isolation**.
3. Across a full ADMM campaign that `chdir` window is the whole run. **A TSO solve failed inside
   it.** Production's failure handler then resolved `transmission_network.results_dir` — a
   **relative** path — against the temp directory, and crashed.

### Two findings, one of them buried

- **A TSO local solve failed during G1 at C\*.** That is gate-relevant signal. We do not know
  how many failures there would have been, because the campaign **aborted inside the failure
  handler** rather than recording and continuing.
- **The failure-capture path is itself broken under any cwd change.** It converts a *recorded
  failure* into an *aborted campaign* — the worst possible behaviour for diagnostics, and the
  fourth capture gap in this programme.

## Authorization requested

G1, G2 and G4 cannot run until this is fixed, and I forbade production changes inside gate tasks
precisely so a Worker could not work around something like this. The minimal fix is a
**production change** and therefore yours:

**Give the ESSO the same `logs_dir` resolution `network.py` already has** —
`shared_energy_storage_data.py:1007-1014` resolves `output_file` against a configured logs
directory instead of the process cwd. This removes the need for `os.chdir` entirely, which
removes the crash, which unblocks the gates and makes the per-cycle detector capturable as
specified.

Secondary, independent of the above and recommended: make `results_dir` absolute at
construction, so the failure-snapshot path is cwd-independent regardless of what any caller does.

**Third item, added after independent Worker corroboration — a bug in code the Planner already
committed (`df46f118`).** `_parse_ipopt_barrier_terms` uses `re.search`, which returns the
**FIRST** match. Combined with `file_append='yes'`, a multi-cycle campaign therefore reports
**cycle 1's `mu_final`/`s_obj` for every later cycle** — silently, with plausible-looking
numbers. Verified by the Planner by reading the parser, and empirically by the Worker on a
two-cycle probe. Fix: take the LAST match (`finditer`), not the first.

**Scope of the damage: none of the reported results is invalidated.** Every measurement
published so far — the ε check, the `tol` remedy check, G5 and G5B — ran each arm in its own
isolated directory against a fresh log, so each parse saw exactly one solve. The bug bites only
multi-cycle campaigns, and none completed. Had G1–G4 run, **the entire per-cycle detector
trajectory that Addendum 4 asks the report to lead with would have been fabricated.**

I have **not** implemented either. Options if you prefer not to touch production now: a
harness-only workaround absolutizing `results_dir` before the run would unblock G1, but it
leaves the underlying defect live for every future caller, and it would not be honest to report
gates run under a workaround as if the infrastructure were sound.

## Not established

- **G1, G2, G4 have no results.** Nothing about SoH reconciliation, recourse, the k=10,000 arm,
  determinism, or the per-cycle detector trajectory at C\* is known.
- The dual-magnitude drift of the leak across ADMM cycles remains a derivation, **unmeasured**.
- The number and pattern of TSO failures at C\* is unknown — one occurred; the run aborted.
