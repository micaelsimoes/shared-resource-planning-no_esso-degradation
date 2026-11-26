## Amendment 3 to Addendum 91 — W187 cell 1 interrupted (signal kill at cycle 122, 14:18 UTC); relaunch ruling (expert, 2026-11-12)

**Ruling (a) adopted.** The partial outputs and the stale lock of cell 1 are moved, not deleted, into
`w187_cell1_interruption_20261112/` under the W187 root (hashes as recorded in `996cb81a`); cell 1 is relaunched with
the frozen command, its log named `_r2`; no code, spec or pin changes; the interruption and the relaunch are recorded
in `P5_15_W187_REPORT.md` with the kill time and cycle. P3 stays blind (no cell has produced a result).

**Ruling (b) adopted, with the conditions that keep it inside the `CLAUDE.md` rule.** The author runs the 15-cell chain
from a Terminal.app window under `caffeinate -i`: the run is attached to a terminal the author owns, it is not
`screen`/`nohup`-detached, and it does not survive the author closing it. Conditions: one chain that stops at the
first failing cell (`&&` or `set -e`); both streams of every cell captured by the launcher's own log as frozen; the
campaign lock held by the chain; the machine not put to sleep or logged out until the chain ends; the Planner reads
the roots afterwards and writes the report. If the chain is interrupted again, the same procedure applies once more
and the second interruption is a stop for the expert.

**Timing.** Relaunch tonight (11-12) lands on 11-13 afternoon; the latest relaunch that still meets the 11-15 12:00
freeze is 11-14 09:00 UTC. If the freeze is missed, cover letter v5 drops the bracket and the paper goes without W187.
