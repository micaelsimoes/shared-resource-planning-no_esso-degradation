# CLAUDE.md

Repository-wide instructions for Claude Code.

## Project

Shared Resources Planning Tool: nonlinear SMOPF and distributed
TSO-DSO coordination using ADMM.

## Current project state

Do not infer the current investigation stage from this file.

For current state, read:

1. `REVISION_CONTEXT.md`
2. `LOCAL_NLP_STABILITY_PLAN.md`
3. the latest relevant stage reports
4. `EXPERT_REVIEW.md` when relevant

`REVISION_CONTEXT.md` is the repository-wide source of truth.
The current sections of that document supersede older historical sections.

## Agent workflow

The project uses three roles:

- Planner — coordinates the investigation and owns technical decisions.
- Advisor — independently reviews mathematical, numerical, and algorithmic issues.
- Worker — performs bounded implementation, testing, and experiments.

Role-specific instructions are under:

`.claude/agents/`

Production-code changes should normally be performed only by Worker after
Planner authorization.

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
  difference smaller than that error is indeterminate, not a result.** The cold-versus-warm
  offset of 1,055,598 carried an error bar of 766,062, so it was barely distinguishable from
  its own uncertainty, and the ranking signal it had to support was 32.87.

## Investigation discipline

Do not start a new stage merely because the previous stage produced a report.

Planner must first assess the evidence and authorize the next action.

Distinguish observations, hypotheses, evidence, and conclusions.

Prefer minimal diagnostic experiments over speculative algorithm changes.