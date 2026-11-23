# P5.15 S56: second handback to the Expert. Rulings recorded, plan v4 for review (2026-11-23)

**What I am asking you to do.** Review `S56_PLAN_v4.md` (`c65201d1`) against your rulings of 2026-11-23, and confirm or
correct the four readings below. No result has changed since `P5_15_S56_EXPERT_HANDBACK_REPORT.md` (`c5cf38a1`). Since
then I have recorded your rulings, re-planned the study around the author's SP-arm decision, and started the authorized
work.

## Decisions needed

### Expert

| # | reading of the 2026-11-23 rulings | Planner's reading | alternative |
|---|---|---|---|
| 1 | **EV screen candidates (plan v4 §4, item g)** | Candidates proposed from the marginal-value table are confirmed by evaluation and a complete poll, and then enter the opportunistic searches as **additional starts**, under the same start rule at 1× and 2× | They are confirmed and reported only, not used as starts |
| 2 | **Step 5 centralized comparator with storage** | The merged network with the reference unit appended as a shared-ESS unit at its bus after the merge. TV is on (κ as in operation). The DN 7 reference-generator limit is added (f). It is compared with the stochastic two-block ADMM pair under V-1 at α = 0. The R/4 rule is tested on the pair difference | TV off in the comparator, with TV checked separately in the single-scenario TV check |
| 3 | **Single-cohort A/B (b), the "cohort rows" for one cohort** | SoC, window, closure, complementarity, the per-cohort sum row and the converter circle. The network side is unchanged (`sess_comp` stays active with one cohort) | The ESSO rows without the converter circle |
| 4 | **D_b (step 4)** | Reported against its own settling bars, with your decomposition: pinned accounting optimism ≥ 0, plus the DN re-optimisation gain ≤ 0. A negative value is reported as the second term dominating, not as a defect | — |

### Author (for the author, not the Expert)

| # | item | recommendation |
|---|---|---|
| 5 | **EV control searches** (complete-poll; reversed class order): at which price level? | 2× only. That is the condition under which storage enters, and it saves the projected 45–110 h of controls at 1× |
| 6 | **Wait-and-see bracket:** 1× and 2×, or 2× only | Both. It is about 7–11 h elapsed in total |

## Recorded since your rulings

**Rulings.** All of them are recorded, with your wording, in TASKS.md (`0d6fbfb0`):
- X4-2 (a);
- X4-1: V-1 is the formulation of record; V-2 is measured and non-deciding;
- the complementarity row and the per-cohort sum row;
- D1–D4, plus the efficiency-equality gate;
- cap 80;
- same-sign diagonals;
- D_b recorded as a prediction, not a stop.

**Plan v4** (`S56_PLAN_v4.md`; v3 kept unchanged) incorporates:
- the rulings and the SP-arm decision;
- items a–h;
- the status of steps 1–4.

**Cost projection** (§8; a projection, not a measurement):
- about 150–400 machine hours, excluding the SP poll;
- the SP poll alone is about 85–1,125 h (|N| × t_SP);
- t_SP is unmeasured; the range comes mainly from t_SP and from the number of EV moves.

**Study-form wording:**
- no value-of-stochastic-solution claim;
- no "optimal plan under uncertainty";
- wording as in plan v3 §1.

**Running or next** (authorized; each run under a frozen spec with recorded predictions):
- **Harness:** per-block cost and pin capture for the pinned arms, zero-solve, about to commit.
- **Step 4 main run.** Arms V-1 (calibration), V-2 (measured, non-deciding), pinned with AA off, pinned with AA on;
  instances A and B; cap 300; a look at cycle 40; continuation to 600 if still contracting. Expected 5–12 h.
- **Cohort follow-up,** in a separate worktree while step 4 runs:
  - the D1 switch;
  - the complementarity and sum rows;
  - D2 closure;
  - the efficiency gate;
  - the single-cohort activation option;
  - zero-solve and millisecond tests, including the dump positive control (simultaneity predicted ≤ 1e-6 MWh/day,
    against 0.85 before).
- **After step 4:**
  - the memory probes (e);
  - EV plumbing;
  - the single-cohort A/B (b);
  - the EV replicate pairs (d);
  - step 5.

**Report-back stop point**, each item stated as measured or projected:
- step 4 against its predictions;
- the memory probes;
- the step 5 pair;
- the A/B;
- the replicate spread;
- the refreshed plan.

## Not confirmed

- **Your assessment file.** `claude/INDEPENDENT_ASSESSMENT_S56_2026-11-23.md` is not in this repository, nor anywhere
  under `~/Projects` to depth 4. Your rulings were recorded from the author's message quoting it.
- **Projected values in plan v4 §8.** Every hour figure except the step 4 smoke's seconds per cycle and the
  two-cycle time of G-f r3 is a projection.
