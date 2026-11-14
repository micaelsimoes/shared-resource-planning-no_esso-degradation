# CLAUDE.md

Repository-wide instructions for Claude Code.

## Project

Shared Resources Planning Tool: nonlinear SMOPF and distributed
TSO-DSO coordination using ADMM.

## Current project state

Do not infer the current investigation stage from this file.

For current state, read:

1. `TASKS.md` — the current order as a ticked checklist; where the work sits now
2. `REVISION_CONTEXT.md` — the current-state summary
3. `PLANNER_BRIEF_2026-09-13.md` — the author's and expert's decision record; its addenda
   are authoritative over earlier sections and over other documents where they conflict
4. `STEP4_DFO_METHOD.md` — the planning-method definition
5. the latest `P5_15_*_REPORT.md` handoff report

`REVISION_CONTEXT.md` is the repository-wide current-state summary; its current sections
supersede older historical sections. `LOCAL_NLP_STABILITY_PLAN.md`, `EXPERT_REVIEW.md`,
`COWORK_HANDOFF.md` and `WORKER_REPORT*.md` are historical: consult them only when a current
question points there.

## Agent workflow

The project uses three roles:

- Planner — coordinates the investigation and owns technical decisions.
- Advisor — independently reviews mathematical, numerical, and algorithmic issues.
- Worker — performs bounded implementation, testing, and experiments.

Role-specific instructions are under:

`.claude/agents/`

Production-code changes should normally be performed only by Worker after
Planner authorization.

## Stopping conditions

Keep going, without asking, while the next step is inside the current order of the latest
addendum of `PLANNER_BRIEF_2026-09-13.md` and inside the frozen spec: measurements, gates,
smoke runs, zero-solve looks, report and state-file updates, Advisor consultations, Worker
tasks that touch no production path outside the spec, and launches the order already names.

Stop and report — with what is needed from the reader first — when:

- an author-level parameter is at stake: budget, cost file, ageing calibration, α, siting,
  routes, scenario draws, discount rate, horizon, machine allocation (the author decides);
- a change to the mathematical formulation, the certification rule, the objective convention,
  a gate's scope, or a production configuration outside the frozen spec is proposed (the
  expert rules);
- a recorded prediction fails and the addendum names no fallback, or the named fallback would
  itself change the configuration. A **certification-status** prediction ("the cell certifies")
  has a fallback by construction — the uncertified reporting form — so its failure is recorded
  and the campaign carries on (Addendum 62). Campaigns stop only on a harness fault, a gate
  failure other than G6/G27, or a failed **margin or sign** prediction, including the case
  where the uncertified form leaves a manuscript claim indeterminate (margin below
  max(3 × larger bar, 2τ));
- a run not named in the current order would be launched, or any run expected to exceed 4 h;
- before anything destructive or irreversible: overwriting or deleting a cited artifact,
  rewriting a reference, deleting a symbol, any git operation beyond staging by filename and
  committing;
- the review point the current order names is reached ("stop for review after …").

A stop raised for a **reporting** ruling does not idle the machine when the queued cells are
independent of the ruling: the stop applies to decisions that change what runs next, and the report
is written while the queue proceeds (Addendum 63).

Actions only the author can perform (a reboot, credentials, hardware) are requested as
actions, not framed as decisions; the run is held at the point that needs them.

Do not stop for choices the spec or an addendum already makes, nor for questions the brief
marks closed, void or withdrawn: cite the addendum and move on, unless a new measurement
contradicts it.

Permission prompts stay on for destructive commands; no role runs with prompts bypassed.

## Task checklist

`TASKS.md` at the repository root holds the current order as a checklist: one line per step,
ticked with the commit hash or run id when done, the active step marked. The Planner updates
it at every transition and reads it first when resuming; it survives context summarization
where the conversation does not. It is committed by filename with the state it records.

## Runtime environment

Machine-specific interpreter and solver paths are defined in
`CLAUDE.local.md`.

All production and diagnostic runs must use the canonical environment for
the current machine and must pass the repository provenance checks.

## Repository rules

- `.env` must never be committed.
- `data/` result and diagram directories are intentionally untracked.
- Never use `git add .` or `git add -A`.
- Stage files explicitly by filename.
- Diagnostic harnesses follow the existing `p5*.py` convention.
- Use real production functions rather than reimplementing them in diagnostics.
- Follow `docs/METHODOLOGY.md` for experimental and reporting conventions.
- Never fabricate or approximate experimental results.
- Do not guess invocation commands when they are uncertain; inspect the
  repository and existing methodology first.

## Evidence and artifact rules

Obligations on whoever creates or commits an artifact. Worked examples and the
incidents behind each rule are in `REVISION_CONTEXT.md`, section
"P5.12-G / W / X / Y / Z results".

- Name frozen artifacts with version and content hash, e.g.
  `frozen_formula_spec_v2_45edc424.json`. Never replace a frozen artifact in place;
  each version must record its predecessor's hash.
- Commit every report together with its primary evidence base and that evidence's
  hash inventory. Never commit a journal without its manifest, or a manifest without
  its journal.
- Ensure every identifier uniquely denotes its content: path-qualify fixture identity
  where basenames repeat, and state constants in frozen plans operationally rather
  than as fixture-specific literals.
- Commit or hash-record the settling artifact for any claim you commit, even when its
  directory is excluded in bulk. A claim whose evidence cannot be re-verified is not
  preserved.
- Preserve the formula, not only the inputs. A reported statistic whose definition
  exists only in prose is unpreserved, however complete its input data.
- Enforce solve claims with armed guards; never assert them. A "no solve" or bounded-solve
  claim must be backed by guards installed for the whole run, raising on entry from any
  undeclared call site, with the permitted count declared in advance and checked exactly —
  too few fails as loudly as too many, since it means the path under test did not run.
  Use `p513_solve_profile_guard.SolveProfileGuard` (bounded) or the blocking form.
  Three of the six P5.12/P5.13 stages asserted the claim instead of arming it, and one of
  those assertions was false in a committed report; the mechanism already existed and was
  simply not used.
- Scope every negative claim about the evidence base. "No prior art exists", "no
  evaluation artifact exists", "nothing in the repository decides this" — each must record
  what was searched: branches, stage scripts, stage artifacts, docstrings and reports. State
  the claim as scoped, never as absolute. S2 found that "neither lever was ever evaluated"
  was false — `p59_b_adaptive.json` is a two-arm A/B of one of them, and that stage's
  docstring already stated the mechanism a later stage then rediscovered. A branch-scoped
  search had been widened to stages without being redone. Same family as rules five and six:
  claims that read as established but were never checked.
- Record the problem instance, not only the settings. An objective, recourse or residual
  value without its candidate is uninterpretable and incomparable, however exhaustively the
  configuration around it is documented. AB1 recorded rho, cap, tolerances, the adaptive
  flag, the solve profile and what was not permitted — and never named the investment
  vector, so its recourse could not be compared with any other stage until the candidate was
  recovered retroactively. Record the instance identifier, or a hash of it, in every artifact
  that reports a value.
- Report a difference with its resolution. Any difference of two iteratively-computed
  quantities must be reported together with the error implied by where each computation
  stopped — for an ADMM recourse, the per-cycle objective change at termination. **A
  difference smaller than that error is indeterminate, not a result.**
  **Refinement (2026-09-13):** the bar bounds *stopping slack*, not *path divergence*. Two
  runs still descending toward different limits can differ by far more than the sum of their
  terminal steps — Track C1 showed exactly that when tightening the tolerance grew the offset
  9.4x. So "determinate at N x the error bar" licenses only "not explained by stopping
  slack"; it does not license "a real difference in the limit". The bar is a **local** measure,
  valid only when both runs have genuinely settled — and rule ten is what tells you whether
  they have. The cold-versus-warm
  offset of 1,055,598 carried an error bar of 766,062, so it was barely distinguishable from
  its own uncertainty, and the ranking signal it had to support was 32.87.
- Assert the capture path before executing. A harness must verify, **before it runs**, that
  a capture path exists for every quantity its frozen specification requires — a checklist
  assertion that fails fast. The C\* campaign spec *required* realized EFC/day with its
  margin to the binding threshold; the harness implemented no capture for it; nothing checked
  before the run; and the quantity was unrecoverable afterwards because the models were not
  serialized. This makes "the spec required it" a statement about the code rather than about
  intent — the same move rule six made for solve claims.
- Report the terminal-step-to-threshold ratio for every cell of every evaluation. A run that
  has genuinely settled stops well inside its threshold; one terminating at ~99% of it is
  being *stopped*, not converging. It costs nothing and is computable from artifacts already
  preserved. In Track C1 the warm cell stopped at **99.3%** of its bound and the cold cell at
  **58.0%** — the contrast diagnosed the mechanism, and reporting it earlier would have
  flagged the 2x2's warm cells before their agreement was read as a result.
- Deactivate and unwire; never delete. A callable that a preserved fixture may resolve at
  unpickling must be retained, unwired and unused, with a comment saying so — retiring a
  constraint means removing its **row** from newly built models, not removing its **symbol**
  from the module. Pickled networks hold `functools.partial` objects that resolve rule
  functions by name at load time, so deleting four retired `model_construction_helpers` rules
  in P5.15-1b broke unpickling of *every* preserved fixture, including the cycle-21 anchor
  `P512R/cycle21_pre_setup/snapshot.pkl` and the FrozenSMOPF comparator. Verify by loading
  each preserved fixture, and separately assert via `hasattr` that the row is absent from a
  freshly built model. `_add_benders_cut` is the precedent to follow.
- Never re-run a harness onto an artifact a committed report cites. Re-running writes to the
  same path by default, so the pre-change evidence is destroyed by the very act of verifying
  the change. Redirect the output to a new name, or copy the original aside first, and say
  which you did. In P5.15 a verification step re-ran the capacity ladder and overwrote
  `data/SRP1/Results/P514L/ladder_s1.json`, the pre-repair result cited as the central finding
  of `P5_14_L_CAPACITY_LADDER_REPORT.md`; the values survived only because a sibling log
  happened to hold them. The obligation is on whoever writes the task instruction, not on the
  agent following it — the instruction there said "run the harness" and named no output path.

- Run campaigns attached, alone, and with both streams captured. A harness that runs a
  campaign must capture **stderr** as well as stdout, must **refuse to run concurrently**
  with another copy of itself whenever any production path it relies on writes to a shared
  or relative location, and must **never be detached** (`screen`, `nohup`, shell
  backgrounding). Give a Worker **one gate per task, with the exact command**. The failure
  mode these prevent is not a crash but plausible false data: in P5.15 the G1–G4 gates were
  launched four at a time into ESSO IPOPT logs written to one relative, appended filename, and a
  first-match parser would have reported cycle 1's `mu_final` for every cycle of every gate;
  a stdout-only launch left G1's first failure as a 0-byte log with no traceback; and a run
  detached with `screen -dmS` survived the task that owned it.

- Fix a defect twice and you fix it at the writer, not at the caller. When a defect class
  recurs, the second fix goes where every caller passes through — a shared writer that refuses
  the bad value, and a repository-wide test that fails on it — so the next recurrence fails a
  test, not a run. Numpy booleans serialised through `json.dumps(default=str)` became the text
  `"True"`; W74–W76 patched two writers, and W98's new hooks wrote the same string, failing G14
  on every line of the 3×3 continuation because the gate checks `is True`. Gate results are
  now written only through the shared writer (Addendum 52).
- A pre-run check that scans committed artefacts excludes the run's own. A collision or
  uniqueness check over committed specs, keys or outputs must exclude the campaign root it is
  about to launch, or it will match the artefact the freeze itself committed. W105's `tests_K`
  required the C\* extension's eval key to appear in no committed campaign spec; W105 then
  committed that very spec, and the launcher refused on every attempt, with zero solves — while
  the sibling `pre_launch_assertion` already excluded its own root. Refusals must also log which
  check failed, not only that one did (Addendum 55).
- Validate a fit only on points it did not use. Agreement with the points a fit was fitted to
  is not a prediction; a model with k parameters passes through k points by construction.
  Report the held-out error beside the estimate. In the 3×3 continuation the Planner reported a
  three-extremum oscillation fit as predicting "the third extremum within 15 €" — circular —
  and its limit (−7,281 €) was wrong; the damped-cosine fit on cycles 72–84, checked on
  85–88 (errors 48–107 €), gave −6,235 € (Addendum 52).

## Stage templates

- **Scope a gate per arm.** A gate that compares every arm against a control reference is
  ill-defined for an arm designed to differ. P5.14-N's perturbation arm was handed the
  control's determinism gate and produced a **spurious** "FAIL — NON-DETERMINISM" verdict for
  a run that was supposed to differ and had not converged. Gates must state which arms they
  apply to, and skip the rest.
- **Record per-cycle state by default.** Any stage whose run can fail records the per-cycle
  trajectory as standard, rather than leaving each spec to remember. P5.14-N could not
  recover the cycle at which its failures began — the third capture gap in this programme.
  Rule eleven asserts what the spec requires; making the trajectory a default is what stops
  a fourth gap arising from a spec that simply did not think to ask.

## Reporting conventions

- Lead with what the reader must act on. A handoff report opens with `Decisions needed`
  (numbered, with options, cost and recommendation) and `Blocked on the author`; a Worker
  report opens with `Blocked on Planner`. Then `Changed` (commits by hash, spec version,
  configuration), `Found` (each recorded prediction stated against its outcome) and
  `Not confirmed` (what could not be verified, and where you looked).
- State the objective convention on every table. `gross_operational_cost` and
  `net_operational_recourse` differ by exactly the terminal salvage credit — 3,448.87 on the
  X22 warm-adaptive cell, which is why its table reads 827,738,663 (gross) where C1's reads
  827,735,215 (net). The two reconcile only once the convention is stated.

## Investigation discipline

Do not start a new stage merely because the previous stage produced a report.

Planner must first assess the evidence and authorize the next action.

Distinguish observations, hypotheses, evidence, and conclusions.

Prefer minimal diagnostic experiments over speculative algorithm changes.