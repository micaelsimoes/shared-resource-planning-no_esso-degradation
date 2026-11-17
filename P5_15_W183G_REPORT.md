# P5.15 W183g (+ W183h) — gate ruling applied, E1 launcher v2, round 8 checked; W183g (c)/(d), W184, W185 held (Addenda 84–85; zero solves)

**Authority:** Addendum 85 (W183g order; W184/W185 conditional) and Addendum 84 (W183h).

**State.**
- **Production `6f43479e`** (unchanged by W183g).
- **Harness `c34ebdc5…`** (unchanged).
- **E1 launcher v2:** `b0841b09…` (`7cf88b15`).
- **Overleaf `a87e195`** (`a87e1950fbdb…`), fast-forwarded from `edd7a63`. The four files are byte-identical (cmp) to the round-8 deliverables: `main.tex` = `main_edd7a63_round8.tex`; letter = v10; `highlights.tex` = v3; `cover_letter.tex` = v2.

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded.

**Runs.**
- **Zero solves** in every part. `SolveProfileGuard(permitted=())` was verified at 0 everywhere, with pickle 0 and model builds 0.
- **Execution.** Each part ran attached, with both streams captured. Scripts were committed before their runs.
- **Before any build:** the Advisor reviewed the designs of (b) and (c), and the Planner ran the Advisor's two zero-solve
  diagnostics.

| part | report | commits | outcome |
|---|---|---|---|
| W183g (a) gate ruling | `P5_15_W183G_A_GATE_RULED.md` | `4c0d8e1d`, `36fee733` | 0 genuine vs W83; `GATE_PASS_RULED=True` |
| W183g (b) E1 launcher v2 | `P5_15_W183G_B_E1_V2.md` | `7cf88b15`, `f0cff829` | GATE_PASS 175/175 |
| W183h round 8 | `P5_15_W183H_ROUND8.md` | `3707475a`, `05873be5` | 4 of 6 items held; 2 wording points |
| W183g (c) fixed-plan launcher | — | — | **held**: cannot be built as ordered (decision 1) |
| W183g (d) spec v1 freeze | — | — | **held**: depends on (c) and on the harness pin |

**Stop.** W184's condition requires W183g (a)–(d) to hold. (c) cannot be built without a harness change that Addendum 85
does not authorize, and that change alters the harness pin (d) is to freeze. Ruling 7's "exactly" check would also stop a
correct E2 run (decision 2). **Nothing solved; W184 and W185 are not launched.**

## Decisions needed

1. **The fixed-plan launcher needs a Step-7 hooks module and a harness router patch (expert).**
   - **The finding.** Ruling 4 (vi)/(xi) orders a thin launcher over `freeze_campaign_spec` + `evaluate` with "the ext-v6
     `settling_resettle` v6 declaration as in spec `10f4aa7a`". But every `settling_resettle` validator in the harness
     router chain (`p515_s44_campaign_harness.py` ~1026–1050) accepts only a declaration for a cell in its own fixed
     table:
     - ext v6: the seven W135 cells, `p515_s53_w142_resettle_ext_v6_hooks.py` ~200–209;
     - v6, W155 and the W118 fallback likewise.

     The declaration also carries cell identity (`cell`, `item`, `claim_group`, `gate`, `replay_reference`) inside the
     key.
   - **Planner check (zero-solve).** `H.validate_settling_resettle` accepts `10f4aa7a`'s declaration, and refuses it with
     `cell` renamed: "cell must be one of ['e_c2', 'e_c2_calfade', 'e_c3_midblock', 'e_c3_unit', 'e_c4', 'e_no_ageing',
     'pb_y2025_n5_v6']".
   - **Options:**
     - (a) **W155 pattern.** A new hooks module `p515_s55_step7_resettle_hooks.py` (own schema; cells E3 1, E2 8,
       cross ≤ 4 added after W186). It would reuse the v6 rule, state and capture checklists by import, and use the
       ext-v6 declaration shape with `nrf_option_expected` / `rho0_scale_expected` / `flex_price_multiplier_expected`
       and entry checks. It needs a one-branch router addition in `resettle_hooks_module`, which changes the harness
       sha256 from `c34ebdc5`. Gate, zero-solve:
       - every committed eval key recomputes (the W183a T1 method, 7,213 entries);
       - the router routes every committed schema to the same module as before;
       - a neutrality check that the new branch is never taken by a committed spec;
       - optionally, the 153-solve two-cycle gate on the patched harness into a new root, under the ruled comparator.
     - (b) Reuse an existing cell name (e.g. `e_no_ageing`): **rejected**. It would write a false identity into every key
       and record.
     - (c) Run E3/E2 at the production exit instead of the certified protocol. This changes ruling 4 (vi), so it is the
       expert's call. Not recommended: H2 is a rule-v6 verdict.
   - **Recommendation: (a).** The 1× screen is production-exit, needs no resettle declaration, and can stay thin.
   - **Spec pinning.** Spec v1 (d) would pin the patched harness hash. The W183f two-cycle arm was produced on
     `c34ebdc5`. A branch no committed spec takes is zero-effect by construction, but this project gates router patches
     (W155's neutrality cell).

2. **Ruling 7's "exactly" fails on a correct run (expert).**
   - **The finding.** The per-cycle `rho_*_after` is a **mean** over models (`_get_admm_penalty_summary`,
     `shared_resources_planning.py` ~7920–7938). In the committed W183f arm (s = 1) the first cycle reads:
     - `rho_pf_after = 0.1980000000000002` (held; S39 0.198);
     - `rho_ess_after = 0.010000000000000004` (exempt; S39 0.01);
     - `rho_v_after = 0.011549999999999998` (increased).
   - **Planner check.** All three are reproduced **bitwise** by `sum([x]*n)/n` in the canonical interpreter:
     - pf with x = 0.198, n = 48;
     - ess with x = 0.01, n = 51;
     - v with x = min(max(0.0077 × 1.5, 1e-4), 1e4), n = 48.

     Every Param equals S39 × s exactly (W183a T3); the ulp offset is the summation.
   - **Recommendation.** Restate the check as bitwise equality with the replicated arithmetic `sum([x]*n_g)/n_g`:
     - x = S39_g × s for factor-1 actions (held, fixed, exempt, held after failure);
     - x = clamp(S39_g × s × f) for increased/decreased (f = 1.5 or 1/1.5, clamp [1e-4, 1e4]);
     - n_g the recorded model count per channel.

     If a cell does not reproduce bitwise, that is a harness-fault stop.

3. **Smaller W185 and campaign declarations (expert).**
   - **H2 after a failure.** Evaluate H2 after each completed pair, in the order (1×, 0.5), (1×, 2.0), (2×, 0.5),
     (2×, 2.0). Should a failure stop at the first failing pair (recommended, as a failed margin prediction under
     CLAUDE.md) or complete all eight?
   - **Persisting certified models for E3/E2.** `persist_certified_models` was on in `10f4aa7a`. Recommend on for E3
     (one cell; it serves any later read-out) and on for E2.
   - **E1 resumability before W186.** E1 v2 runs a whole arm in one process (≤ 30 evaluations × 0.75 h ≈ 22 h), which
     is not attachable within the 4 h tool limit. Its run precondition also forbids resuming into the phase root.
     Recommend per-poll launches with a resume state rebuilt from the phase root's committed evaluation records. That is
     a launcher change, to be ordered before W186.
   - **B+ interpretation choices (W183g (b); each implemented and gated):**
     - (a) the 10-index window includes the first refused index;
     - (b) in case (iii) the terminating record is the first examined index's, byte-equal to the frozen loop's, with the
       9 further indices in `halton_indices_skipped`;
     - (c) `advance_rule.polled_index` is null on a cap refusal;
     - (d) `--freeze`/`--run` accept only `P515S55/e1_2x` and `P515S55/e1_2x_batch4`. The screen goes to the fixed-plan
       launcher, so cross entries written by (c) into `e1_2x` would be foreign to the E1 inventory; (c) should use
       `e1_2x/cross` as its own phase root;
     - (e) the dry-run output is renamed.

     Recommend confirming all five.
   - **For information: the B+ window can exhaust on E1's own seeds.** A pure scan (k = 0…99) found 35 of 1,593
     over-cap incumbents with a 10-index window below 8 distinct points. These include the unit at k = 3…12 and
     7: 0.5/2 at k = 4…13; both are reached only if the search returns with `k_next` in those windows. One stub path
     skipped 8 indices. Rule (iii) applies, so no fallback is missing; a `STOP_FOR_REVIEW_completion_cap` is a realistic
     E1 outcome.

4. **Round-9 wording (expert; from W183h).** No run depends on these.
   - **Seven label-word prefixes** at `main.tex` l. 530, 1631, 1639, 1742, 1763, 1780, 2002 (`Appendix~` or
     `Section~\ref{app:…}`). The Planner confirmed all seven. They were introduced at `edd7a63` and pass through round
     8; W182c's check let them through because it accepted "Section"/"Appendix" before `app:`. On elsarticle they print
     "Appendix Appendix A.x" (static reading).
   - **Table E.24 rows 39 and 6** (`7: 1.25/5`, 2025) carry no `$^{b}$` and the caption no legend, while §3.5 says the
     probes are "marked $^{b}$ in the tables". The Planner confirmed l. 2904 and l. 2911. The fix goes through
     `gen_appE.py`.
   - **`$^{b}$` has two meanings** in Appendix E: over budget in E.20/E.21, and "the start reported in Table 8" in E.23.
   - **"Within numerical resolution of break-even"** remains at l. 1520 and l. 1806 and in cover letter v2 l. 19, while
     letter v10 now says "within the numerical resolution of zero, so its sign is not resolved".
   - **The cover letter v2** still carries the bracketed round-9 placeholder.
   - **The letter's prediction-record bullet** (l. 286, "49 predictions, with outcomes … supplementary material") is the
     author's call (Addendum 84 ruling 1).

## Blocked on the author

- **Push from PyCharm.** The terminal cannot authenticate.
- **The prediction-record bullet** (decision 4, last item).

## Changed

| commit | content |
|---|---|
| `4c0d8e1d`, `36fee733` | `p515_s55_w183f_bitwise_gate_v2.py` (ruled comparator: `field_list_drift` iff every shared key is bitwise equal; `GATE_PASS` and `GATE_PASS_RULED` both recorded); `p515_s55_w183g_a_vs_w83.py`; `P515S55/w183f_srp1_bitwise_gate_vs_w83/`, `…_ruled/` |
| `7cf88b15`, `f0cff829` | `p515_s55_e1_launcher.py` v2 (`run_continuation_e1`, generated from `C.run_continuation`'s source, differing only in signature, explicit `k_next` and the B+ block; `--run` uses it; batch root and seeds declared; B+ text in every spec); `p515_s55_w183g_b_e1_v2_dryrun.py`; `P515S55/w183g_b_e1_v2_dryrun/` |
| `3707475a`, `05873be5` | `p515_s53_w183h_round8.py`; `P515S55/w183h_round8/` |
| this commit | `P5_15_W183G_REPORT.md`, TASKS.md, REVISION_CONTEXT.md |

- No production, harness, certificate-module, PB, spec, frozen, clone or `manuscript_review/` file changed.
- The committed gate chain, the W183f wrapper and its outputs are untouched.

## Found — predictions against outcomes

| prediction (source) | outcome |
|---|---|
| (a) 0 genuine diffs on all 12 files vs the W83 arm (A85) | **held** — 0 genuine, 0 tie-order; every file of both arms was hash-checked against its manifests first |
| (a) provenance diffs ≤ 3, all under `rule_eleven_checklist` (A85) | **failed on the count: 4**, all under `rule_eleven_checklist`. The fourth is `w83_convergence_depth_tail`, the W83 reference arm's own tail read-back (W83 ran with the tail on, as ruling 1 notes). It is configuration of the reference, not trajectory. Addendum 85 states W184's condition as "0 genuine in (a)", which is met; recorded, not stopped on |
| (a) wrapper v2: `GATE_PASS_RULED` (A85 rule c) | **True** — the 44 frozen "genuine" diffs are the 22 `PER_CYCLE_RESPONSE_FIELDS` keys × 2 rows, now `field_list_drift`; the 29 shared keys are bitwise equal. Negative controls: a planted shared-key change, a key absent in the reference but non-null in the new arm, and a deleted key each keep `GATE_PASS_RULED` False |
| (b) advance off ≡ imported loop at poll 0, 8 cases (A85) | **held** — return dicts byte-equal (same writer), stub call logs equal; 6 storage refusals, x = 0 14-point completion |
| (b) advance on → k = 2/2/1/0, sizes 8/9/8/14 (A85) | **held** — skipped [0,1], [0,1], [0], none; W183b's k = 0…9 table reproduced byte-equal |
| (b) A7 and every other path unchanged (A85) | **held** — paired runs (A7 per poll and overall, move to a storage incumbent, budgets 0/5, max_polls 1, window exhaustion at 10): advance off byte-equal to the imported loop; advance on equal up to the first advance, then only the declared differences. The source diff touches only the signature, `k_next` and the B+ block. 8 guards 0; 52 tripwires 0 |
| (b) v1 dry-run assertions on v2 | **held** — 7 keys + 1 alias; G0 `ca195bd3…` accepted with `require_current_production=True`; rows 35/40 base keys equal |
| W183h (a) identity (A84) | **held** — the four files; clone diffs equal the expert's hunk by hunk (31/12/1/2) |
| W183h (b) Appendix E (A84) | **held** — generator output once at l. 2556–2946; 33 PASS; 1,623 entries 0 MISMATCH; 0 entries changed vs `edd7a63` (16 cell edits are mark insertions); `$^{b}$` on exactly E.21 {5,6,9,10,11,38,39} and E.20 {4,5,8,9,10,26,27,30,31} |
| W183h (c) letter v10: 0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED (A84) | **held** — 172 pointers: 167 / 0 / 0 / 5; [42] Naumann, [43] EVE, [44] Boyd; l. 97 "Section 3.3" now MATCH |
| W183h (d) 0 of the 16 A7 sentences remaining (A84) | **held** |
| W183h (d) over-budget points labelled at every table/figure occurrence (A84) | **failed: 23/25** — E.24 rows 39 and 6 (decision 4) |
| W183h (e) 0 MISMATCH in the 21 edits (A84) | **held** — 114 tokens: 93 MATCH, 7 DERIVED, 0 MISMATCH. "About 3 %" = (−748,745.77 + 25,082,401.21) / 842,832,534.76 = 2.887 % |
| W183h (f) 0 CONFIRM, 0 prefixes, blue on the 23 A5 lines (A84) | **failed on prefixes: 7** (decision 4); 0 CONFIRM and blue 23/23 held |

## Not confirmed

- **Compilation.** Nothing was compiled; there is no TeX here. Numbering is static, and how elsarticle prints the seven
  prefixes is not confirmed.
- **Launchers not executed.** The E1 launcher's `--freeze`/`--run`, and the batch-gate comparison script (not yet
  written), have not run. (b)'s path tests use stub records; a `harness_error` record was not exercised.
- **Ruling 7's restatement** was validated on the one committed row (s = 1) only.
- **Overleaf** was not re-fetched after `a87e195`.
