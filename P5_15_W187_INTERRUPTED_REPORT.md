# P5.15 W187: cell 1 was killed by the Claude session restart at 14:18 UTC. The screen is stopped and needs a ruling (2026-11-12)

**Authority:** Addendum 91 (W187 go), and Amendment 2 ruling 2, which accepted the background-mode start.

**Spec:** spec v1.1 `25071d2f`; screen stage spec `5cbf4a55`; cell `x0_A_1x`, spec `64aac615…`, cap 500.

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded. No Q was produced.

**Status.** 0 of 15 cells have results. P3 stays blind.

## Decisions needed

1. **How to resume cell 1 (expert).**
   - **The problem.** The frozen launcher has no path for an interrupted cell:
     - `--run` refuses, because the cell root must hold only its frozen spec, and it now also holds `evals/` and
       `campaign_heartbeat.json`.
     - `--retry-of` refuses, because a retry "follows a completed (failed or stopped) run only", that is, a run with
       results.
     - The campaign lock `.p515_s44_campaign.lock` names PID 31351, which is dead. `check_campaign_preconditions`
       refuses on any lock.
   - **(a) Restore and relaunch — recommended.** No code, spec or configuration change.
     1. Move the partial, untracked outputs (`evals/`, 119 MB, and `campaign_heartbeat.json`) out of the cell root into
        `P515S55/w187_cell1_interruption_20261112/partial_root/`. Move, never delete; their sha256 inventory is already
        recorded.
     2. Move the stale lock into the same directory, after re-confirming that PID 31351 is dead. Its content is already
        copied there.
     3. Relaunch cell 1 with the frozen command, changing only the log name to `run_x0_A_1x_launch_r2.log`. The
        `noclobber` setting protects the original log, which is kept.

     The launcher's own 21 preconditions then re-check everything. The cell restarts from cycle 0; the 45 minutes run
     so far are lost.
   - **(b) Add a launcher path for interrupted cells.** This changes the launcher's sha256. Spec v1.1 and the stage
     spec pin it, so it needs a spec v1.2 and a re-freeze of the screen. Not feasible in the window. Not recommended.
   - **(c) Drop W187.** The paper goes without it, and R3.6's planning part stays "met in part", as Addendum 91
     provides.
   - **Why this is a stop.** Option (a) moves outputs and a lock file. That is in the CLAUDE.md "anything destructive"
     class, even though nothing committed or cited is touched.
2. **How to run the screen so that a session restart cannot kill it (expert, with an author action).**
   - **Why the foreground rule does not help.** The kill came from the session's process tree ending, not from
     background mode. A foreground tool call would have died in the same way. Amendment 2 ruling 2's
     foreground/background distinction does not protect a 21 h run.
   - **(i) The author starts the chain in a Terminal.app window — recommended.**
     - The chain is the 15 frozen commands joined by `&&`, so it stops at the first non-zero exit. Cell 1 uses the
       `_r2` log name.
     - It runs under `caffeinate -i`, so the Mac does not sleep.
     - The process is attached to that terminal, both streams go to each cell's log through the frozen redirection,
       and it uses no `screen` and no `nohup`.
     - A Claude restart cannot reach it. The Planner commits each cell's outputs as they land and writes the report.
     - The window must stay open for about 21 h.
     - The Planner will generate the exact chain from the stage spec's `launch_command` fields once (a) is ruled.
   - **(ii) Keep the run inside Claude Code.** Any app restart or session resume during the 21 h kills the running
     cell again.
3. **Time.**
   - Relaunched by about 15:00 UTC on 11-12, the screen lands around 12:00 UTC on 11-13.
   - To be in by 11-15 12:00 with margin, the latest relaunch is about 11-14 09:00 UTC.

## Blocked on the author

- **The Terminal launch** under decision 2 (i), once the ruling is given.
- **Push** from PyCharm.

## Changed

- **New, not committed until this report:** `data/SRP1/Results/P515S55/w187_cell1_interruption_20261112/`, holding:
  - `sha256_inventory_partial_cell1.txt`: 394 files, the whole cell root plus the launch log, hashed at
    14:20:20 UTC;
  - `stale_lock_copy.json`: a copy of the lock;
  - `state_at_capture.txt`: PID liveness, the last log lines, and the absence of a results file.
- **Untouched:** the cell root, the launch log and the lock are where they were. No code, spec or configuration was
  changed.

## Found

| item | observation |
|---|---|
| launch | 13:29:19 UTC (lock `started_utc`). The parent was PID 31351 and the child 31554. The parent's shell belonged to the previous Claude session's process tree |
| progress | the last heartbeat was 14:18:22 UTC: cycle 122, 2,943 s elapsed, 369 ESSO solves, 0 network failures |
| end | at 14:19:00 neither PID was alive. The launch log has no exit line. `child_stderr.log` ends mid-cycle (`Running SMOPF, Network case33_1, 2035, Autumn...`) with no traceback. There is no `campaign_results.json` |
| cause | the restart of this Claude session. At about 14:19 the session came back with a new scratchpad id (`c6e62c21…`) and a new shell snapshot (14:19), and the old process tree, including the launcher started from it, was gone. Signal death, not a harness fault |
| launcher semantics | `_cell_done` requires `campaign_results.json`. `freeze_retry` refuses a cell with no results. `run_preconditions` requires the root to hold only the spec. So there is no frozen path to resume |
| P3 | blind and untested. No cell has results |

## Not confirmed

- **The kill signal and its sender** were not recorded. The attribution to the session restart rests on the timing
  (last heartbeat 14:18:22; new session artifacts at 14:19) and on the absence of a traceback.
- **Whether the restart was an app restart, a resume on another device, or a context reset** is not known here.
- **Whether a Terminal.app process survives** an app update or a logout is not tested. A reboot would kill it in any
  case.
