# P5.15 W188 — zero-solve support tasks for the refocused revision (Addendum 89), with the W184/W185 claims and a pin check

**Authority:** Addendum 89 (2026-11-10).

**Overleaf HEAD `a87e195`** (`a87e1950fbdb…`). `main.tex` md5 `39f6fd86…` = `main_edd7a63_round8.tex` (round 8); fetched
2026-11-10.

**Repo.** Start `75d8e12a`. Production at HEAD is `f6c23977`; spec v1 pins `6f43479e`.

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded.

**Runs.**
- **Zero solves** in every part. `SolveProfileGuard(permitted=())` was verified at 0 everywhere.
- **Model builds:** only in the pin check, with stubbed `optimize`.
- **Pickle loads:**
  - W188 (d): 2 guarded loads, allow-listed by sha256; plus 2 in a logged trial run.
  - Every other part: none.
- **Execution:** six parts ran in parallel on separate files, each attached, with both streams captured. Scripts were
  committed before their runs.

| part | report | commits |
|---|---|---|
| (a) slack inventory (G10, P5) | `P5_15_W188A_SLACK_INVENTORY.md` | `687c1a53`, `8ce38f88` |
| (b) supplement generator + verifier | `P5_15_W188B_SUPPLEMENT.md` | `5d7955da`, `958a658e` |
| (c) ruling-5 code reading (both heads) | `P5_15_W188C_CODE_READING.md` | `5d00d3bd`, `9bd9d66a` |
| (d) operational-figure data | `P5_15_W188D_OPERATIONAL_FIGURE.md` | `d5cb6bbb`, `547167ed` |
| W184/W185 claims (ruling 2, P1, P2) | `P5_15_A89_W184_W185_CLAIMS.md` | `5e0c2710`, `3baf0e77` |
| pin check of `f6c23977` (for W187) | `P5_15_PINCHECK_F6C23977.md` | `09b3e3cf` |

## Decisions needed

1. **Addendum 89's order presumes W183i/W184/W185 have not run; they ran on 2026-10-26 (expert, record).**
   - W183i (a)–(d), W184 (E3) and all eight W185 (E2) cells completed under spec v1
     (`frozen_step7_campaign_spec_v1_96345241.json`, production `6f43479e`) between 2026-10-26 11:10 and 21:26 UTC.
   - That was before Addendum 87's R6 hold was drafted (10-30) and before `f6c23977` changed production (10-31).
   - The reports: `P5_15_W183I_REPORT.md` (`ec87c098`), `P5_15_W184_REPORT.md` (`f1cada84`), `P5_15_W185_REPORT.md`.
     The W185 report was written 10-27; its commit was cut off and it is committed now, unchanged (`f7c3f601`).
   - All nine runs pin production `6f43479e`, harness `d9a944f2…`, hooks `9318ade3…` and launcher `3b990d19…` (claims
     part). They are exactly the runs Addendum 89 decision 1 and ruling 1 ask for.
   - **No re-run was made or is needed.** None is possible under spec v1 either: the launchers refuse, because three of
     its 160 pins changed in `f6c23977`.
   - **P1 and P2 were recorded on 2026-11-10, after the runs,** so they are checked against the records, not blind
     predictions.
2. **W187 is eligible (W185 reported before 2026-11-22) but blocked by spec v1's pins (expert).**
   - **The pin check settles the model question.** With every `f6c23977` switch at its default, the current HEAD builds
     the same initialisation models as spec v1: **696/696 blocks identical** (NL, labelled NL, `.row`, `.col`, block and
     component digests; all 8 configurations, both DSO builders). The Q aggregation is source-identical (AST-equal),
     and the C1 term can never enter Q.
   - **The launcher would still refuse on four pin differences:** the production head, the three module hashes, and two
     new tracked inputs added by `f6c23977` (`data/SRP1/SRP1_5x3y_1x1.json`,
     `data/SRP1/SharedESS/SRP1_5x3y_ESS_Params.json`, P5.16 files that no SRP1 run reads).
   - **Options:**
     - (a) **Spec v1.1** (recommended): re-pin the three modules and the head; declare the two P5.16 inputs excluded;
       predecessor `96345241`. Gated by this pin check, plus the 153-solve two-cycle gate on HEAD into a new root (about
       2 min), because the ADMM-update path is covered here only by AST equality.
     - (b) Run W187 from a worktree at `6f43479e`. That avoids re-pinning, but needs the launcher's root and lock paths
       checked across checkouts.
     - (c) Do not run W187; R3.6's planning part stays "met in part".
   - **Time:** W187 is 15 production-exit evaluations, about 1.4 h each serially, ≈ 21 h. To land before the 11-28
     freeze it should start by about 11-20.
3. **The G10 sentence (expert; W188 (a)).** P5 held on all 20 certified rows behind the named tables: every penalised
   slack is at its zero bound within the solver's bound relaxation, so the penalty paid in Q is 0 €. Three
   qualifications:
   - **A common relaxation offset.** Q contains a signed slack content of **−13,012.57 to −13,052.36 €** (−2.0e-5 of Q),
     common to every evaluation within 39.80 €, so it cancels in every reported difference to within 40 €. It equals
     D0 T0.1c's Q_cert⁺ − gross to 1.8e-12 €.

     Proposed wording: "at every certified evaluation behind the tables, the penalised slacks are at their zero bounds
     to the solver's bound relaxation; the relaxation leaves a common offset of −13.0 k€ in Q, which cancels in every
     reported difference to within 40 €".
   - **`t11_3x3_unit`** (Table 12, uncertified, production exit) cannot be shown at zero from its records: the
     flexibility day-balance positive content is ≤ 2.16 €, and that directory holds no pickle.
   - **Superseded `pb_y2025_n5`** (E.21 row 16, outside the named set) has a **positive voltage-slack sum** in TSO 2035
     Spring (+6.456e-6 p.u.², up to 170.69 € positive content). The Appendix E sentence at l. 2934 names only its
     closure.

   Also, the ESSO σ± pair is penalised only in the ESSO objective and is not in Q (srp:1237–1302). It sits at its floor
   on all 25 named rows.
4. **§4.4 sentence (expert; ruling 2).** From the records:
   - de-confounded saving **X = 12.202 %** (90.876 M€);
   - enabled saving **12.205 %** (90.897 M€; the manuscript's 12.2 %);
   - both on the price-taker denominator;
   - the four reverse-flow hours' share of the gap **0.0225 %** (20.46 k€, determinate at 1.62× its threshold).

   At one decimal both savings read 12.2 %. Drafted: "With reverse flow prohibited on the coordinated exchange as on the
   static arrangements, coordination saves 12.2 % (90.9 M€); with reverse flow enabled, 12.2 % (90.9 M€), both relative
   to the price-taker benchmark. The four reverse-flow interface-hours account for 20.5 k€, 0.023 % of the 90,896.6 k€
   gap." The expert chooses one decimal, three, or M€.
5. **Round-9 manuscript points found by W188 (expert).**
   - **Appendix D TN-connection table** (`tab:cs1_ieee33_generators`, l. 2305) prints one row at ±250 MW/MVAr for all
     ADNs. The node-7 ADN's case file (`case33_2_2025.json`, REF generator) has **±100**; case33_1 and case33_3 have ±250
     (Planner read all three). Print one row per ADN.
   - **Ruling 5 wording.**
     - The network complementarity row is `pch_hat · pdch_hat ≤ 1e-4` with P = S^Av · hat, which is the product form
       divided by (S^Av)², a positive scaling: **no difference to raise** (the row is the same at both heads; the Planner
       read `model_construction_helpers.py:949–951`).
     - The ESSO has no complementarity row; simultaneous charge and discharge carries an ε^E = 1e-5 objective term
       (already stated at l. 1914).
     - Five node-level ESSO rows stay active at x = 0, satisfied only at zero, so "inactive storage rows are deactivated"
       should read "**the rows of inactive cohorts are deactivated**".
   - **The supplement (W188 (b)).**
     - The B–D figures plot scenario means ± std from production's sampler, but the supplement carries the stored base
       profiles. Recommend keeping base profiles; exporting the curves needs a separate order.
     - E.24's CSV carries an `over_budget` column on rows 39 and 6, which the printed E.24 lacks (W183h). Keep or drop.
     - RES tables are exported for the printed years only.
6. **Figure choice (expert; W188 (d)).** Both data sets are ready:
   - **The E6 export,** re-derived independently: 1,094/1,115 numbers bitwise; 21 differ by 1–2 ulp from summation
     order.
   - **The reference unit's hourly SoC** for the four days of 2025 and 2035, both copies agreeing to 1.45e-5 MWh. There
     are 41 lower and 38 upper limit contacts, all within 1e-6 MWh.
   - **SoH per block under the seven Table 6 calibrations.** The k values reproduce, AE/EFC match the frozen and printed
     values, and the baseline trajectory equals the W101 reference unit's.

## Blocked on the author

- **Push from PyCharm.** The terminal cannot authenticate.
- **The calendar date of the deadline** (Addendum 89 decision 2), which sets W187's start-by date (decision 2).

## Changed

- See the part table.
- **New tooling:**
  - `manuscript_review/gen_supplement.py` and `verify_supplement.py`;
  - `manuscript_review/supplement/SRP1_supplement_v1_a94d8193.zip`, deterministic: three runs are byte-identical.
- **No change to:** production, harness, launchers, specs, frozen files, the clone, `gen_appE.py`/`verify_appE.py`,
  `export_W160/` or committed artifacts.
- **No P5.16 spec or `definitions.py` touched** (Addendum 89 stop).

## Found — predictions against outcomes (expert, Addendum 89; P1/P2 recorded after the runs)

| prediction | outcome |
|---|---|
| P1 (W184): NRF raises Q(0) by < 10 % of the 90,896.6 k€ gap; de-confounded saving > 11 %; four hours < 1 % of the gap; stop if the saving < 9 % | **held** (checked against the 10-26 records) — +20,463.88 € = 0.0225 % of the gap; X = 12.202 %; no stop |
| P2 (W185): every certified Q within 3b of the frozen cell's window; every decisive difference keeps sign and determinacy class; ρ read-back 8/8; stop on a sign change | **held** under both readings of b (frozen band / new band) — 7/8 inside the window; `e2_x0_2x_rho0p5` is 682.59 € above row 34's window, 0.10 (0.05) of 3b. Spread of the decisive difference: 1× 1,572.86 €, 2× 1,452.62 €, 0.11–0.13 of each arm's threshold; signs and classes kept; ρ 8/8 |
| P5 (W188 (a)): every penalised slack at its zero bound on every certified row | **held** on all 20 certified rows (penalty content 0 €), with the common −13.0 k€ relaxation offset and the two exceptions outside the certified named set (decision 3) |
| W188 (b) verifier: 0 MISMATCH over the 1,623 Appendix E entries | **held** — 1,623 entries 0 MISMATCH; 1,633 cells vs `main.tex` 0 differences (bar the declared E.24 column); B–D 70,743 cells vs the case files 0 differences; 15/15 negative controls caught |
| ruling 5: the code's complementarity row differs from the product form by at most a positive scaling | **held** — positive scaling only (1/(S^Av)²); 63/63 citations hold at both heads; 52/53 shared ranges byte-identical (one indentation-only) |

**W185 spread table (ruling 2).** Margin = Q(0) − Q(unit) − I(unit), €.

| price | baseline | ρ₀ × 0.5 | ρ₀ × 2.0 | spread | determinate, sign |
|---|---|---|---|---|---|
| 1× | −64,417.39 | −64,264.41 | −65,837.28 | 1,572.86 | all three, negative |
| 2× | +46,285.46 | +46,959.42 | +45,506.80 | 1,452.62 | all three, positive |

## Not confirmed

- **Compilation.** Nothing was compiled; table numbers are static. W188 (a) mapped tables by content: the order's
  "Table 11's coordinated row" exists only in the expert's numbering.
- **The ADMM-update path at HEAD.** It is covered by AST equality only. The pin check built initialisation models.
- **The run-time IPOPT bound-violation lines** behind the "floor" premise were not read, except W179c's one cell.
- **The B–D figure curves** (sampler output) were not reproduced.
- **The 1.2e-6 MWh E^Av gap** between the model copies and the record (2025) was not traced.
