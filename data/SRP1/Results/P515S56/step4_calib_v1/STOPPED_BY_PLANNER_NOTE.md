# Calibration re-run v3, instance A: stopped by the Planner (2026-11-23, ~19:09)

On 2026-11-23 the author and the Expert ruled that calibration (a) is deferred to idle machine time. It gates only
the SP evaluations, and the EV harness work on the critical path edits the files this campaign runs from.

**What was stopped:** the instance-A launcher (PID 14544) and its child (PID 14580), at about cycle 35 of the cap of
140. Instance B was never launched.

**What is preserved:**
- the campaign root, as left;
- the campaign lock, moved byte-identical to `stopped_A_stale_lock.p515_s44_campaign.lock`;
- spec v3 `0b3a0989` and the harness keys (`091b7745`), which stay valid for the deferred re-run.

The re-run will start fresh from cycle 1 into a new root, with the bitwise-prefix check as frozen.
