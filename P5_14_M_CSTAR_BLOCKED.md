# P5.14-M — the C* evaluation is BLOCKED by a third production defect

**GO was authorized and cannot execute. The run crashed in a diagnostic print statement,
not in the solver, at ADMM cycle 1.** Frozen spec
`data/SRP1/Results/P514D/frozen_cstar_v1_5a6210aa.json`, frozen before the run.

## What happened

Initialization succeeded at `C* = 0.96875 MVA / 3.875 MWh` per node — as the ladder
predicted — and ADMM started. On the first cycle, after the residuals were computed, the
run raised:

```
File "shared_resources_planning.py", line 5186, in _print_worst_primal_residual_diagnostics
    f'Sch={values["sch"]:.6f} MVA, '
KeyError: 'sch'
```

**The producer and the consumer disagree.** The `charge_discharge` dictionary is built with
keys `pch` / `pdch` (`:5014`, `:5023`, `:5035`); the printer reads `sch` / `sdch`
(`:5186-5187`) — the **retired apparent-power names**.

| | commit | date |
|---|---|---|
| printer written with `sch`/`sdch` | `1777457d` | 2026-09-02 |
| producer converted to `pch`/`pdch` | **`58f4911b` "P5.4-C: ESSO active-energy conversion"** | **2026-09-06** |

P5.4-C converted the producer and not the consumer. The defect has stood for a week.

## Why it never fired before

The diagnostic is guarded (`:5152`):

```python
if worst_ess is not None and residual_metrics['primal']['ess'] > params.tol['consensus']['ess']:
```

with `tol['consensus']['ess'] = 0.1`. **At negligible capacity the ESS primal residual never
reaches 0.1, so the printer was never entered.** At material capacity it is exceeded on the
first cycle.

So this is not a defect that material capacity *caused*; it is a defect that material
capacity finally **reached**.

## The pattern, which is now the substantive finding

Three blockers, all in code paths that only material capacity or an unused mode reaches:

| # | path | defect | broken since | found by |
|---|---|---|---|---|
| 1 | uncoordinated mode (cell 1) | `expected_shared_ess_p` referenced but never built | 2026-08-18 (`99a59fec`) | D1 cell 1 |
| 2 | node 7 at 1.00 MVA | *not a defect* — a genuine feasibility boundary | — | the ladder |
| 3 | ESS complementarity diagnostic | `sch`/`sdch` against `pch`/`pdch` | 2026-09-06 (`58f4911b`) | this run |

**The tool has been exercised only in a regime where its storage-specific code paths are
inert.** That is the governing fact restated with a mechanism: not merely that the case
study is too small to show the effect, but that the code which handles a materially loaded
shared ESS **has never been executed**, so its defects have accumulated unobserved.

## Severity, which differs sharply across the three

- **This defect is trivial**: two dictionary keys in a print statement. It changes no
  model, no constraint and no number. Behaviour preservation under a gate would be
  immediate.
- **Cell 1's is small but not trivial**: guard one of three penalty components, with a
  formulation reading behind it (an uncoordinated DSO has no shared ESS, so the term should
  be absent).
- **Node 7's is not a defect at all**: a real capacity limit, and the only one of the three
  that is information about the system rather than about the code.

## Status

`C*` evaluation: **not obtained.** The GO decision stands and is unexecuted. No artifact was
written; `d1_cell3.json` still holds the earlier 1.00 MVA run and was not overwritten.

The three-point capacity series — 0 / 0.0106 / 0.96875 MVA — remains at two points.

**No production change has been made.** Fixing the printer is two keys, but it is a
production edit and is not authorized; and it would be the third defect fixed in a path
that no test exercises, which is itself worth a decision rather than a patch.

---

# Addendum — the fix, and both branches of the C* run

## The two-key fix, with its proportionate gate

Applied to `_print_worst_primal_residual_diagnostics`: `values["sch"]` -> `values["pch"]`
and `values["sdch"]` -> `values["pdch"]`. Nothing else.

Gate, sized to what the change could possibly affect:

1. **The diff is exactly those two keys** — `git diff --numstat` reports `2 2`.
2. **The function is verified print-only** — every binding in it is a local read
   (`worst_v = residual_metrics.get(...)`, `values = charge_discharge[agent]`,
   `base_text = ...`); no mutation of inputs, no append/update, no return. Its sole effect
   is `print`.
3. **The C* run proceeds past the point it previously crashed.**

No neutrality reproduction was run. A change confined to two string literals inside an
f-string, in a function whose only effect is `print`, cannot move a number — and running a
35-minute reproduction for it would be the discipline turning into ceremony.

**Checked before patching:** the other keys the printer reads — `product`, `simultaneous`,
`net`, `base_mva` — all exist in the producer dict (`:5014-5041`). A partial fix would have
crashed on the next line.

### Two cosmetic defects deferred to the bundle, not one

Both remain in the output and would mislead a future reader **in different directions**:

- **The names.** The labels read `Sch=` and `Sdch=` — the *retired apparent-power* names —
  for what are now active-power values.
- **The units.** The labels read `MVA`, but per the producer's own P5.4-C comment the ESSO
  entries "stay in p.u. (`base_mva` is None) while the TSO/DSO entries are in MW". So the
  unit is wrong for every agent, and wrong in two different ways between them.

The values printed are now correct; their names and units are not.

## The success branch, named so it has somewhere to go

A mid-trajectory failure is predeclared as the expected outcome and is itself the finding.
**If C\* instead completes**, it produces the first storage-benefit number this tool has
ever generated at material capacity — and it must not be reported as validating the
mechanism without one further check.

**Proposed follow-up (NOT authorized): a C3-style perturbation at C\* capacity.** Change
the degradation constant and see whether the recourse moves.

- If it **moves**, the ageing mechanism is live at this capacity and the number means what
  it says.
- If it **still does not move**, the degradation model is inert even where the storage is
  doing real work — **a finding about the model rather than about the case study**, and a
  far deeper one than anything in Track D so far.

At bootstrap capacity the same perturbation moved nothing to sixteen digits, which is what
makes the check worth running rather than assuming.
