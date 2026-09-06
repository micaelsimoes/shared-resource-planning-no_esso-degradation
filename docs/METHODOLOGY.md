# Diagnostic & validation methodology

The conventions this investigation has followed since P3.5, and that P4.6
and the final report should continue to follow. Consistency across stages
is what makes the accumulated evidence trustworthy — don't improvise a
different style for P4.6.

## Frozen-pickle regression discipline

- Every frozen model snapshot used as a test case is identified by its
  file path **and** its SHA-256 hash, and the hash is re-verified fresh
  (`sha256sum`) before trusting it in a new stage — never trust a hash
  quoted from an earlier report without re-checking, since files can be
  (and have been) silently overwritten by a later run.
- Load with a fresh `.clone()` per branch/case so mutations in one
  branch (e.g. a scaled-replacement experiment) can never leak into
  another branch's baseline.
- When comparing two variants of the same test (e.g. "A" = baseline,
  "B" = scaled), always run both from the **same frozen starting point**,
  never from each other's post-solve state.

## Call real production code, don't reimplement it

The single most important discipline in this investigation: P4.3 and
P4.4 do not reimplement the kappa-scaling idea in a standalone diagnostic
script (as every earlier P3.5 stage necessarily did, before the change
existed in production). They import and call the actual production
functions (`sess_snet_def_rule`, `shared_ess_snet_def_scale`,
`_sync_sess_snet_def_scale`, `configure_shared_ess_operational_state`)
directly. A diagnostic script should only ever *observe* production
behavior (capture stdout, serialize a returned `state` dict, snapshot
structural invariants) — never *reimplement* the logic under test. If
P4.6 recommends a standard-ESS analog, validate it the same way: call the
real function, don't re-derive its formula in the test script.

## Required-first-gate-then-proceed pattern

For any regression replay across multiple cases where the plan specifies
"decisive" cases that gate further work (see `LOCAL_NLP_STABILITY_PLAN.md`
§6 for the P4.4 example): implement the gate as **real control flow** in
the script, not just a note in the report. Run the decisive/mandatory
cases first, compute a boolean gate, and only proceed to the remaining
cases if the gate passes — so the "STOP if either fails" instruction in
the plan is a structural guarantee, not something that depends on a human
reading the results in the right order.

## Adversarial testing for lifecycle/state-machine logic

When validating dual-variable or other cross-cycle state lifecycle logic
(P4.3's stale-dual injection is the template), don't just check the happy
path. Deliberately inject a value that should be cleared and confirm it's
cleared; deliberately leave a value that should survive untouched and
confirm it's untouched; exercise the full cycle (deactivate → reactivate
→ deactivate → reactivate) rather than a single transition.

## Structural invariant checks, every step

At every step of a construction/validation script, snapshot and compare:
component Python object identity (`id()`), local name, full index-tuple
set, total constraint-data count for the whole model (not just the
component under test), and confirm no new component was created. This is
what makes "the production component must remain `sess_snet_def`" (plan
§2) a verified fact rather than an assumption.

## Delivery workflow when the validation environment lacks Pyomo/IPOPT

If the Claude Code session running P4.6 does not have Pyomo + IPOPT + MA97
available directly: write the validation script, verify it at least
compiles (`python -m py_compile`), and ask the user to run it in their
real environment and share back the JSON report it writes. Do not guess
at what the results would be. This was the necessary workflow for all of
P4.3–P4.5 in the prior session; it still works here if needed, but check
first whether this Claude Code environment can just run it directly
(much faster if so).

## Stage report template

Every stage report (`P3_5A_REPORT.md` through `P4_5_REPORT.md`) follows
the same shape — keep using it for P4.6 and the final report:

1. **Title**: `# Stage <id> — <one-line description> (executed)`
2. **Provenance**: who executed it, via which script, and the raw output
   file path(s) — so results are always traceable back to a script and a
   JSON/log file, never just asserted in prose.
3. **Required proof / invariants table**: the exact checks the plan
   demands for this stage, each with a pass/fail per case.
4. **Results table(s)**: quantitative before/after or case-by-case
   comparison (e.g. A vs. B, or this-stage vs. the previous baseline).
5. **Interpretation**: a numbered list of what the results mean, ending
   with an explicit statement of what is (and is not) authorized by this
   result — every report so far ends by clarifying that no production
   change is authorized beyond what was already implemented and that the
   next stage is gated on planner review.
6. **Closing phrase**: an exact, single-line status string in a code
   block — never paraphrased, never omitted. For P4.6 and the final
   report, the exact required strings are in
   `LOCAL_NLP_STABILITY_PLAN.md` §10.

## Committing work

Investigation scripts, stage reports, and production-code changes should
be committed to git as they're completed (specific files by name — never
`git add -A`/`git add .`, since this repo's working tree also contains
large data/result directories and a `.env` file that must never be
committed). Don't let multiple stages' worth of work accumulate
uncommitted — it's exactly what happened before this doc was written
(everything through P4.5 sat uncommitted until commit `0171f451`), which
is unnecessary risk for no benefit.
