# P5.15 W173 — §2.1 `% [CONFIRM — W173]` points: search frame and neighbour counts

Zero solves, read-only. Manuscript clone `manuscript/6a67305f25e8348fb71380c3/` at Overleaf `407f8df`; main.tex sha256
`9891057c…` checked; clone not modified. Helper `p515_s53_w173_search_counts.py` (sha256 `21ca9cb7…`). Own guard
`SolveProfileGuard(permitted=())` verify(0) == [], plus the 6 parent guards the imported launchers install, all at 0.
pickle.load/loads blocked, 0 calls. Integrity failures 0; inputs unchanged. W100 typing test PASS. Objective convention:
F = I + Q, Q = certified `gross_operational_cost` (settlement excluded). Instances: x = 0 (`d2c96b14…`, z = 0); F2 plan
`y2030__n5_p0.25_e0.5__n7_p1_e3.5` (eval key `5ca4f86c…`, z = [1,1,4,7,0,0,1], m = 2).

## Blocked on Planner

Nothing blocks the task. Two decisions are the Planner's:

1. **The count the F2 sentence carries.** The candidates are 10 (final poll set), 17 (box neighbours evaluated at any
   stage) and 13 (box neighbours re-evaluated in T1). Recommendation: **17**, with the 13 re-evaluated ones in a
   clause (Answers, item 2). The present sentence ("twelve … better than each of them") is wrong.
2. **One re-evaluated neighbour comes out nominally better than the plan.** `y2030__n5_p0.25_e1__n7_p1_e3`
   (`ref:e28de4ac`, the W118 "f2_challenger") has d_gross = **−6,108 €** against an uncertified bar of 27,900 €, so it
   is within the bar. In the search it was the certificate's one unresolved indeterminate point (−6,571 € against a
   resolution of 18,450 €). The claim "better than each of them" fails on this point. It can be worded as "within
   resolution of the thirteenth". Whether that wording is acceptable is for the expert or the Planner.

There are also three Algorithm 1 mismatches outside the two CONFIRM comments (Evidence E1.6). They are listed for the
W174 re-audit and need no action from me.

## Answers

**Item 1.** In s47 and s51 the poll size starts at **Δ_0 = 4** lattice units. It **doubles on success**
(`delta *= 2`, `p515_s47_phase_b_record.py:720`). It **halves on failure, never below 1**
(`max(DELTA_MIN, delta // 2)`, :730). It **terminates when a poll at Δ = 1 fails** (:721). These are
`extra.poll_design` `delta_0 4 / success "Delta *= 2" / failure "Delta = max(1, Delta // 2)"` in specs `8cfa264e`
and `5ce295e1`. Realized Δ sequence: s47 4 → 2 → 1 (terminate); s51 4 → 2 → 1 (success) → 2 → 1 (stop at the cap,
61 > 30); s53 1 (fail → terminate). In every variant-A poll at Δ = 4 or 2, all 8 rounded directions were inadmissible,
so every variant-A evaluation was a completion point. Algorithm l. 429/432 matches the code; Δ_0 is never given a
value. Clause for l. 410: "Δ ← Δ_0 (Δ_0 = 4 lattice units in variant A; Δ = 1 throughout in variant B)". The other
items of l. 438:
- **Completion at Δ = 1 with cap 30: confirmed.** The cap refuses the whole poll and never truncates it.
- **N^max: confirmed** as 20 (s47, s51) and 60 (s53). In the code N^max is `MAX_NEW_EVALUATIONS`, a cap on **new**
  evaluations: cache hits, inadmissible points and duplicates do not count. Realized use was 14/20, 20/20 and 7/60.
- **Householder n + 1: confirmed.** Recomputed directions equal the recorded ones on all 9 polls.
- **s53 snap tie-break: confirmed.** It is a six-level key at `p515_s53_f2_certificate.py:345–352`. In s53's only
  poll it decided 5 of the 13 snaps.

**Item 2.** Box = admissible lattice points at ℓ∞ distance 1 (`PB.Lattice.neighbourhood`; it equals the box each run
recorded).
- **x = 0: carry 14.** Clause: "all fourteen admissible unit neighbours (ℓ∞ distance 1, the full box) were evaluated,
  and each is worse beyond the search resolution". Poll set 14 = box 14, all 14 evaluated, 0 unevaluated, 14/14
  determinately worse at 1.07–7.17× the resolution. 8 of the 14 were re-evaluated in T1, all determinately worse. The
  current l. 443 clause "(fourteen, the full box; …)" is **correct**.
- **F2: carry 17.** Clause: "seventeen of its 61 admissible unit neighbours were evaluated (the ten points of the final
  poll and seven earlier evaluations) and none is determinately better; of the thirteen re-evaluated under the
  certification rule the plan is better than twelve (seven determinately, five within resolution) and within
  resolution of the thirteenth". The current "the final poll evaluated twelve neighbours … better than each of them"
  is **incorrect**: the final poll had 10 points, and "twelve" is the Addendum 63 set (E2.4).

## Evidence

### E1 Search frame (item 1)

E1.1 Code (exact-text locations; `w173_search_counts.json` → `code_locations`):
- `p515_s47_phase_b_record.py`:
  - constants: `DELTA_0 = 4` :194, `DELTA_MIN = 1` :195, `POLL_DESIGN = 'orthomads_n_plus_1_neg'` :199,
    `MAX_NEW_EVALUATIONS = 20` :200, `MAX_POLLS = 60` :201, `COMPLETION_CAP = 30` :205;
  - directions: `householder_columns` :487, `project_direction` (`round_half_away(delta·h/‖h‖∞)`) :499,
    `poll_directions` :502, `neg = -sum(cols)` :509;
  - `run_mads` steps: `delta = int(delta0)` :594, `for k in range(max_polls)` :603, completion only at Δ = DELTA_MIN
    :607, cap check :650 (before the budget check :663), success / unit failure / halve :720 / :721 / :730.
- `p515_s51_f2_phase_b.py`: imports PB :122 and calls `PB.run_mads` unchanged :1153. It asserts `PB.DELTA_0 == 4` and
  the other PB constants :327–330.
- `p515_s53_f2_certificate.py`: `N_DIRECTIONS = 2n` :151, `MIN_FEASIBLE_POLL_POINTS = n + 1` :152,
  `'orthomads_2n'` :153, `DELTA_UNIT = 1` :154, `MAX_NEW_EVALUATIONS = 60` :156, `FIRST_HALTON_K = 4` :158,
  `directions_2n` :321, `snap_key` / `SNAP_KEY_LEVELS` :345 / :351, completion trigger `n_distinct < n + 1` :470.
- Method source: `STEP4_DFO_METHOD.md` §5.2 l. 189–191 ("Δ^p_0 = 4 … doubles on a successful poll and halves on a
  failed one … never below 1"; sha256 `d05bae1e…`).

E1.2 Specs (field `extra.poll_design`):
- s47 `campaign_spec_s47_phase_b_8cfa264e.json` (sha256 `8cfa264e…`) and s51
  `campaign_spec_s51_f2_phase_b_5ce295e1.json` (`5ce295e1…`):
  - design: `delta_0 4, delta_min 1, design orthomads_n_plus_1_neg, n_directions 8, halton_t0 17, success "Delta *= 2",
    failure "Delta = max(1, Delta // 2)"`;
  - limits: `max_new_evaluations 20, max_polls 60, completion_cap 30`.
- s53 `campaign_spec_s53_f2_certificate_r1_803571c0.json` (`803571c0…`):
  - design: `delta 1, design orthomads_2n, n_directions 14, min_feasible_poll_points 8, on_success "Delta stays 1",
    first_halton_k 4`;
  - limits: `max_new_evaluations 60, completion_cap 30`;
  - `iteration_rule`: "the STEP4 5.2 / ruling A3 doubling is NOT applied".

E1.3 Realized poll sequence (`poll_history`; results s47 `906f9da3…`, s51 `2cdd5d60…`, s53 `61b9c501…`):

| run | poll k | Halton t | Δ | dispositions | decision → next Δ |
|---|---|---|---|---|---|
| s47 | 0 | 17 | 4 | 8 directions rejected (inadmissible) | failure → 2 |
| s47 | 1 | 18 | 2 | 8 rejected | failure → 1 |
| s47 | 2 | 19 | 1 | 8 directions rejected + completion 14, 14 new | failure at unit size → **terminate** (certificate holds) |
| s51 | 0 | 17 | 4 | 8 rejected | failure → 2 |
| s51 | 1 | 18 | 2 | 8 rejected | failure → 1 |
| s51 | 2 | 19 | 1 | 8 rejected + completion 20, 20 new (N = 20 = N^max) | success → 5ca4f86c, Δ → 2 |
| s51 | 3 | 20 | 2 | 8 rejected (0 new, polled at N = N^max) | failure → 1 |
| s51 | 4 | 21 | 1 | 7 rejected, 1 feasible direction; completion 61 > 30 | **STOP_FOR_REVIEW_completion_cap**, nothing evaluated |
| s53 | 4 | 21 | 1 | 14 directions → 13 snapped + 1 feasible as rounded → 10 distinct (4 duplicates); 7 new, 3 cache hits; completion not triggered (10 ≥ 8) | failure at unit mesh → **terminate** |

E1.4 Directions. The construction is H = I − 2ww^T, with w = (2u_t − 1)/‖2u_t − 1‖₂ and u_t the Halton point
(bases 2, 3, 5, 7, 11, 13, 17) of index t = 17 + k. Variant A uses the columns h_1…h_7 and h_8 = −Σ h_j. Variant B
uses ±h_j, each negative rounded from the negated column. Each direction is rounded as
d = round_half_away(Δ h/‖h‖∞), so ‖d‖∞ = Δ. Recomputing with `PB.poll_directions` / `S53.directions_2n` reproduces
the recorded `directions` and `halton_t` on all 9 polls. Wording suggestion for l. 415: "z^inc + d_j,
d_j = round(Δ h_j/‖h_j‖∞)" in place of "z^inc + Δ v".

E1.5 s53 snap. The six-level lexicographic key is: (1) ℓ1 to the rounded point; (2) ℓ∞ to the rounded point; (3) ℓ1
to the incumbent; (4) same investment year as the incumbent; (5) lower I(z); (6) label. Level 4 was inserted by the W50
ruling 1. Decided-by counts in the single poll: ℓ1 alone 8, same-year 4, ℓ1-to-incumbent 1; 1 point was feasible as
rounded.

E1.6 Other text-vs-code points in l. 410–432. These are not in the two comments.

1. **l. 423, variant B completion.** The text reads "add admissible unit neighbours until n+1". The code adds the
   **whole** completion, every admissible box neighbour, when fewer than n + 1 = 8 distinct points remain, and the
   cap 30 refuses the poll (`p515_s53_f2_certificate.py:470–481`; spec `poll_rule`). The completion was not
   triggered in s53. For F2 the box has 61 points, so a triggered completion would have been refused.
2. **l. 412, "While N < N^max".** The code loops over at most 60 polls (:603) and refuses a poll only when its new
   points exceed N^max − N (:663). Polls with no new evaluations therefore run at N = N^max: s51 polls 3 and 4 ran with
   N = 20. Under the text's loop s51 would have ended after poll 2 with "budget exhausted"; the code instead recorded
   the cap stop.
3. **l. 401 KwIn.** Δ_0 is not listed among the inputs.

### E2 Neighbour counts (item 2)

E2.1 x = 0. Source: `campaign_s47_phase_b/campaign_results.json`, sha256 `906f9da3…`, field
`termination_certificate`: `n_feasible_neighbours 14, all_feasible_neighbours_polled true, n_no_improvement 14,
n_indeterminate_unresolved 0, n_barrier 0, n_improvement 0, holds true`.
- **(a) Final poll** (k = 2, Δ = 1): 22 entries listed (8 directions, all inadmissible, + 14 completion), so the poll
  set has 14 points: 14 new evaluations, 0 cache hits.
- **(b) Box:** 14 = 7 non-empty node subsets × {2025, 2030}, at 0.25 MVA / 0.5 MWh. The box equals the poll set; 14
  evaluated, 0 not.
- **(c) Search time:** all 14 are worse beyond the resolution max(bar_x + bar_inc, σ_Q). F − F(0) runs from
  +33,459 € to +151,807 € (1.07× to 7.17×); the smallest multiple is y2025 n5, +43,020 € against 40,189 €.
- **(c) Frozen tables v1 590088fe:** 8 of the 14 were re-evaluated, all determinately worse:
  - `phase_b` rows (v6 verdict): pb_y2025_n9 +52,460, pb_y2030_n9 +46,706, pb_y2025_n7 +54,941, pb_y2030_n7 +49,022,
    pb_y2030_n5 +43,326;
  - claims: `C:y2025__n5_p0.25_e0.5` (pb_y2025_n5_v6) +49,693, `C:…n5…n9` (c_156ce2d1) +102,097, `C:y2030…n5…n7`
    (c_6597a79d) +93,878;
  - thresholds 12.6–13.5 k€.

  The other 6 multi-node neighbours were not re-evaluated in T1. At search time they were worse by +101,829 € to
  +151,807 € (3.47× to 7.17×).

E2.2 F2. Source: `campaign_s53_f2_certificate_r1/campaign_results.json`, sha256 `61b9c501…`, field
`termination_certificate`: `scope.claim` "poll failure over the recorded poll set at unit mesh, NOT a
positive-spanning-set certificate, NOT the full-box certificate"; `holds` true; `n_poll_points 10`; `spanning.rank 5`,
`positively_spans_R7` false; `box_neighbourhood {n_feasible 61, n_in_poll_set 10, n_outside_poll_set 51}`;
`cached_box_neighbours.counts_by_class {better 0, worse 5, indeterminate 2}`.
- **(a) Final poll:** 10 points, 7 new evaluations and 3 cache hits, all evaluated. Search classes: 8 worse,
  2 indeterminate.
- **(b) Box:** 61 admissible (equal to s51's refused completion of 61). 17 were evaluated: the 10 of the poll plus 7
  cached ones, from s51 Phase B and the s51 F2 ladder. 44 were not evaluated.
- **(c) Search time:** of the 17, 13 are worse beyond the resolution and 4 within it. Of the 4, one is nominally
  better: `y2030__n5_p0.25_e1__n7_p1_e3`, −6,571 € against 18,450 €.
- **(c) Re-evaluated:** 13 of the 17 have an L claim in T1 (against ref:5ca4f86c, uncertified form). d_gross in €,
  bar in k€:

| neighbour | T1 cell | d_gross | bar | verdict |
|---|---|---|---|---|
| y2025__n7_p0.75_e3 | j_f3aa335e | +94,347 | 27.7 | determinate |
| y2030__n7_p0.75_e3 | l_8e4c220e | +61,486 | 27.7 | determinate |
| y2030__n5_p0.25_e0.5__n7_p1.25_e3 | l_2ab0ce2d | +53,259 | 27.7 | determinate |
| y2030__n7_p1_e3 | l_76c78064 | +50,617 | 27.7 | determinate |
| y2030__n7_p0.75_e3__n9_p0.25_e0.5 | l_7db09f6c | +35,336 | 27.7 | determinate |
| y2030__n7_p1_e3__n9_p0.25_e0.5 | l_45aa25a6 | +31,329 | 27.7 | determinate |
| y2030__n5_p0.25_e0.5__n7_p0.75_e3 | l_195156fa | +28,621 | 27.7 | determinate |
| y2030__n5_p0.25_e0.5__n7_p1_e3 | l_0ee93aca | +26,427 | 27.9 | within the bar |
| y2030__n7_p1_e3.5 | l_e1da0984 | +23,869 | 27.7 | within the bar |
| y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5 | l_b2251bc5 | +9,804 | 28.2 | within the bar |
| y2030__n7_p1_e4 | l_7b199ef9 | +6,384 | 27.7 | within the bar |
| y2030__n7_p1_e3.5__n9_p0.25_e0.5 | l_7c455554 | +5,577 | 27.7 | within the bar |
| **y2030__n5_p0.25_e1__n7_p1_e3** | **ref:e28de4ac** | **−6,108** | 27.9 | within the bar (neighbour nominally better) |

  The table gives 12 positive (7 determinate, 5 within the bar) and 1 negative within the bar.

E2.3 F2: the 4 evaluated box points with no T1 row. They are the four **2035** points of the final poll, new in s53:

| point | F − F_inc at the search's exit | multiple of 18,450 € |
|---|---|---|
| y2035__n5_p0.25_e0.5__n7_p1_e3 | +446,657 € | 24.2× |
| y2035__n5_p0.25_e1__n7_p1_e3 | +465,236 € | 25.2× |
| y2035__n5_p0.25_e0.5__n7_p1_e4 | +493,561 € | 26.8× |
| y2035__n5_p0.25_e1__n7_p1_e4 | +511,160 € | 27.7× |

They were never re-evaluated (scan, E2.5).

E2.4 Reconciling 12 vs 17. The "12 positive, 7 determinate, 5 within the bar" counts come from two different
twelve-point sets that happen to tally the same:
- **Box-neighbour L claims (13).** These are 12 positive (7 / 5) and 1 negative within the bar (e28de4ac).
- **The twelve `l_` cells of T1.** This is the "12 neighbours" of `P5_15_ADDENDUM63_L_AND_AGEING_REPORT.md` l. 76:
  "seven determinately worse (1.03–2.22×)". The set includes **`l_df1a5525` =
  `y2030__n5_p0.25_e0.5__n7_p0.75_e2.5`, which is not a box neighbour**: zE7 differs by 2 (T1 claim
  `L:df1a5525_not_a_neighbour`, +47,651 €, 1.72×, determinate). The set also excludes j_f3aa335e and e28de4ac.

So 17 = 13 L-claim neighbours (6 from the final poll + 7 cached) + 4 unre-evaluated 2035 points. The "12" is neither
the poll set (10) nor the evaluated box (17).

E2.5 Scoped "not evaluated" search for the 44 F2 neighbours. Scanned:
- every `evaluation_record.json` on disk under `data/`: 274 files, all git-tracked, 0 untracked;
- the `points` of every git-tracked `campaign_results.json` under `data/SRP1/Results`: 89 files.

In total 429 point records with a canonical candidate were scanned. A match requires the same canonical candidate and
`flex_price_multiplier == 2.0`. **None of the 44 matches.** The scan does find every one of the 17 evaluated points,
including the re-settle campaigns (w142_resettle_v6, w118_resettle), which confirms that it covers them.

E2.6 Current sentence at main.tex l. 443, quoted from the clone: "What the search certifies is the scope of its final
unit poll, which the paper states with each plan: at the baseline incumbent, $\boldsymbol{x} = 0$, every admissible
unit neighbour was evaluated (fourteen, the full box; at a boundary point no positive spanning set is admissible); at
the plan found under the doubled flexibility price the final poll evaluated twelve neighbours, which do not positively
span the space, and the plan is reported as better than each of them, seven determinately and five within resolution,
not as a mesh-local optimum."
- **x = 0 clause: correct.**
- **F2 clause: incorrect** on three points:
  1. the final poll had 10 points;
  2. the "twelve" set includes a non-neighbour (df1a5525) and leaves out two re-evaluated neighbours;
  3. the plan is not better than every re-evaluated neighbour: e28de4ac is −6,108 € within the bar.
- "Do not positively span": correct for the poll set (rank 5) and for the full box (rank 6).

### E3 Artefacts

- Script `p515_s53_w173_search_counts.py` (sha256 `21ca9cb73a4df7c4…`). Run at HEAD `80fa1824`, exit 0, wall a few seconds.
- Outputs in `data/SRP1/Results/P515S53/w173_search_counts/`:
  - `w173_search_counts.json` `0aa337fc…`
  - `manifest_sha256.json` `620e14f1…` (sha256 of the 13 inputs and of the output)
  - `launch.log` `9107b71c…`
  - `w173_bool_typing_test.json` `174dffb1…` and `.log` `b4d2d89c…` (PASS, 0 failures)
  - `manifest_post_run_sha256.json` (launch.log, manifest and the typing-test files)
- A first launch crashed before writing anything: the guard sweep called getattr on a lazy pyomo module and raised
  `DeferredImportError: yaml`. I restricted the sweep to `p51*` modules. A second, complete run had a narrower
  evidence scan (campaign_results points only), and its two outputs (uncommitted, cited nowhere) were deleted before
  the final run. The final run's outputs are the ones listed above.

## Not confirmed

- **Overleaf state.** I did not check whether Overleaf moved past `407f8df`; no pull was permitted. Addendum 69 and
  `STEP6_ROUND2_CORRECTIONS.md` are not in the repository (TASKS.md), so I did not read them.
- **Evaluations outside the scan.** "44 not evaluated" is scoped to E2.5. Evaluations held only in logs, xlsx files,
  pickles, untracked directories without an `evaluation_record.json`, or other branches were not searched.
- **Re-evaluated values.** These are read from the frozen tables `590088fe` as committed. I did not recompute them.
- **Search-time margins.** These use the Q values the searches read: x0 at its pinned A0 record cycle 132, and the F2
  incumbent at its s51 record. They are not the re-settled values.
- **Wording.** Whether "the scope of its final unit poll" should be restated as the evaluated box (17) or kept as the
  poll (10) is a wording choice for the Planner or the expert. I recommend 17 because 10 drops seven evaluated
  neighbours that are re-evaluated in T1.
