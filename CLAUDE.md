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

## Investigation discipline

Do not start a new stage merely because the previous stage produced a report.

Planner must first assess the evidence and authorize the next action.

Distinguish observations, hypotheses, evidence, and conclusions.

Prefer minimal diagnostic experiments over speculative algorithm changes.