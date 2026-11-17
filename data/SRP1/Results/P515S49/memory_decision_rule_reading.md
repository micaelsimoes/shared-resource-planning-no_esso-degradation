# Memory task: reading of the Addendum 29 decision rule, recorded BEFORE the paper-scale re-measure

**Planner, 2026-09-22.**

**Rule** (spec v16 and v18): resident after initialization ≤ 26 GiB, AND flat across cycles, AND one cycle ≤ 25 min.
If all three hold, three paper-scale evaluations run on this machine. Otherwise the ≥ 64 GiB machine, or the SRP1-with-caveat fallback.

**"Resident" is read as the process's macOS `phys_footprint`** (resident plus compressed pages: the true memory demand), not `rss_tree`.
- Under compression `rss_tree` understates demand. In r2 it stayed ≤ 12 GiB while the footprint reached 36 GiB and swap began to grow.
- The rule exists to decide whether a ~40 h evaluation can run on 32 GiB without thrashing.
- `rss_tree` and swap are reported alongside. The watchdog still gates on `rss_tree` (28 GiB) plus the swap-growth guard, as committed (288cbe71).

**"Flat across cycles":** the footprint after cycle 1 minus the footprint at the end of initialization must be ≤ 0.5 GiB beyond the predicted warm-start one-time increment (W32: +60 × 45–58 MiB ≈ 2.6–3.4 GiB, plus ~0.4 GiB TSO).
- With one timed cycle, "flat" cannot be verified beyond cycle 1. The run times one cycle only, so flatness is reported as **not established by one cycle**, unless a second cycle is authorized.

**Predictions (W32, 01d6d01b), recorded before the run:**
- footprint after initialization ≈ 27 ± 1.5 GiB;
- after cycle 1 ≈ 30–31 GiB;
- no swap growth during initialization;
- one cycle ≈ 18–20 min plus ADMM overhead.

**Consequence of the reading:** with footprint as the measure, the prediction sits at the 26 GiB threshold. A narrow failure is a real possibility, and would be reported as such, not re-interpreted after the fact.

**Run:** `p515_s44_scale_measurement.py --instance paper --label paper_cycle_snapoff_memfix_r1 --snapshots off --time-one-cycle --rss-limit-gib 28 --release-solution-bookkeeping`, machine alone.
