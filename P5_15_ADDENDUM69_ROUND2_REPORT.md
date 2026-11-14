# P5.15 Addendum 69 order — W172, W173 (round 2 checks, zero-solve); W174 held

**Authority:** the author's order of 2026-10-08, which cites Addendum 69 and `STEP6_ROUND2_CORRECTIONS.md`. **Neither
file is in the repository**, in the working tree or on `origin`; the brief ends at Addendum 68. W172 and W173 were
therefore run from the order text and the `% [CONFIRM — W172/W173]` comments in the manuscript. Any prediction that
Addendum 69 records could not be scored.
**Manuscript:** the Overleaf clone at HEAD `407f8df`. main.tex sha256 is `9891057c…`, 2,169 lines; Appendix A is
l. 1404–1679.
**Runs:** zero solves (guards armed and verified at 0, pickle blocked). Nothing else ran.

## Decisions needed

1. **Appendix A.6, l. 1674–1675: "At a single scenario all of these terms vanish identically" (expert; impact H).**
   The interface settlement does not vanish.
   - It is carried at weight 1 in every ADMM local objective, single scenario included: +Σ π_t·baseMVA·P^I_t for the
     DSO and −Σ for the TSO (`model_construction_helpers.py:1701-1702, 1823`; `shared_resources_planning.py:5050`).
   - It is excluded from the reported Q (`shared_resources_planning.py:1302-1304`).
   - At a single scenario, only row 18 and the voltage pin are left unbuilt (`model_construction_helpers.py:1917`).
   - **Recommendation:** "At a single scenario, row 18 and the voltage regularisation are not constructed; the
     interface settlement remains in every local objective and is excluded from the reported cost."

2. **Appendix A, three M rows (expert).**
   - **A1-2, l. 1427–1428:** the expectation is written Σ_o ω_o. It runs over market × operation pairs, Σ_{m,o} ω_m ω_o,
     which is how §2.3.6 l. 876 already writes it.
   - **A3-1, l. 1528:** the z-update is called "the minimiser of the sum of the three agents' storage terms". That holds
     only with the ESSO's terms weighted by κ^E, as in eq. (admm_esso_local).
   - **A5-4, l. 1650–1655:** the end of the cycle is ordered as exit → ρ balancing → AA. The code runs AA step (3275)
     → ρ balancing (3406) → AA memory cleared on a ρ change (3422) → exit (3710). Under the text's order, a cycle in
     which ρ changes would never extrapolate.
   - **Recommendation:** fix all three. There are 17 L rows besides, listed in `P5_15_W172_APPENDIX_A_AUDIT.md`.

3. **Recovery settings (CONFIRM l. 1592 (e), as round-1 decision 9; author/expert, for §3.5).**
   - Recovery `acceptable_tol` 10⁻⁴ / `acceptable_iter` 1 is set for the **TSO only** (`case9_params.json`).
   - A DSO retry is a cold start with its primary options (10⁻⁴ / 5).
   - The ESSO recovery runs at 10⁻⁹ / 1 (`shared_energy_storage_data.py:1082, 1277-1285`).
   - Network `max_iter` 500 is confirmed.
   - **Recommendation:** §3.5 states them per agent; the appendix comment is corrected.

4. **DN interface ratings for §3.5, and a wrong branch table (author).**
   - `get_interface_branch_rating()` (`network.py:83-94`) gives **R^I = 2.0 / 1.0 / 1.5 p.u.**, i.e. 200 / 100 / 150 MVA
     on baseMVA 100. These are the interface transformers of `case33_1` (TN node 5), `case33_2` (node 7) and `case33_3`
     (node 9). The values are the same in every year file.
   - The Planner read the three 2025 files. The committed `g_s39_D.json` records give the same values.
   - The IEEE-33 branch table (`tab:cs1_ieee33_branches`, l. 1776–1830) prints **200 MVA for branch 1 of all three
     ADNs**. That is right for node 5 only; nodes 7 and 9 are 100 and 150. The other 81 ratings match all three files.
   - **Recommendation:** give branch 1 per ADN in the table or its caption, and give the R^I values in §3.5.

5. **§2.1 neighbour counts (CONFIRM l. 446; expert).** Source: W173.
   - **x = 0: "fourteen" is correct.** All 14 admissible unit neighbours, the full ℓ∞-1 box, were evaluated, and each
     is worse beyond the search resolution. 8 of the 14 were re-evaluated in T1, and all 8 are worse determinately.
   - **The F2 sentence at l. 443 is wrong on three counts.**
     - The final poll had **10** points, not 12.
     - The "twelve" set (Addendum 63's twelve `l_` cells) includes `l_df1a5525`, which is not a box neighbour.
     - The plan is **not** better than every re-evaluated neighbour. T1 `L:y2030__n5_p0.25_e1__n7_p1_e3` (the F2
       challenger, `ref:e28de4ac`) is −6.1 k€ against a bar of 27.9 k€ (0.22×): nominally better, within resolution.
   - **Recommendation:** "seventeen of its 61 admissible unit neighbours were evaluated (the ten points of the final
     poll and seven earlier evaluations) and none is determinately better; of the thirteen re-evaluated under the
     certification rule, the plan is better than twelve (seven determinately, five within resolution) and within
     resolution of the thirteenth."

6. **§2.1 Algorithm 1 — the frame rule and four further points (CONFIRM l. 438; expert).** Source: W173.
   - **Confirmed:** Δ₀ = **4** lattice units (`p515_s47_phase_b_record.py:194`; specs `8cfa264e`, `5ce295e1`), double on
     success, halve on failure with floor 1, terminate on a failed poll at Δ = 1. Also confirmed: completion at Δ = 1
     with cap 30 (a poll over the cap is refused whole), N^max 20 / 20 / 60 (= `MAX_NEW_EVALUATIONS`), the Householder
     n + 1 construction, and the six-level snap tie-break in s53.
   - **Realized Δ:** s47 4 → 2 → 1; s51 4 → 2 → 1 → 2 → 1, then the cap stop (61 > 30); s53 1.
   - **Differences:**
     - Δ₀ is neither given a value nor listed as an input (l. 401, 410).
     - Variant B's completion adds the whole set, capped at 30, not "until n + 1" (l. 423).
     - The loop runs up to 60 polls and refuses only a poll whose new points exceed N^max − N, not "while N < N^max"
       (l. 412). s51 ran polls 3 and 4 at N = N^max.
   - **Substantive: in variant A no direction point was ever evaluated.** At every poll, all 8 rounded directions were
     inadmissible, so every evaluated point came from the unit-neighbour completion. The text calls the search MADS, and
     the paper should say this.
   - **Recommendation:**
     - Clause at l. 410: "Δ ← Δ₀ (Δ₀ = 4 lattice units in variant A; Δ = 1 throughout in variant B)".
     - Fix l. 412 and l. 423.
     - Add one sentence: "in variant A every rounded poll direction was inadmissible at every poll, so the evaluated
       points are those of the unit-neighbour completion."

7. **§3.5 is still missing, and §2 cites it (author).** The values it must carry, all confirmed against code or records,
   are listed in W172's "W166 items still absent" (σ = 93,635,360; κ^E 227,210.997 / 386,258.694; S_ref 2.5 MVA;
   ρ₀ 0.0077 / 0.198 / 0.01; pin 9e4; production `compl_inf_tol` 5e-4 / 1e-4; ESSO IPOPT 1e-10 / 1e-9 / MA57;
   `max_iter` 500; recovery per item 3; R^I per item 4). §2 also cites "Section 3.4" for the calibrations, but §3.4 is
   now "Active Distribution Networks".

## Blocked on the author

- **Addendum 69 and `STEP6_ROUND2_CORRECTIONS.md`:** commit and push them from the Air checkout.
- **W174's trigger:** confirm that the round-2 corrections are in Overleaf. W174 re-runs the number check at that HEAD
  and re-audits §2.1–2.3 against the W171b rows only.

## Changed

| task | commit | content |
|---|---|---|
| order | `80fa1824` | TASKS.md |
| W172 | `86f75a4f` | `P5_15_W172_APPENDIX_A_AUDIT.md`, `p515_s53_w172_appendix_a_checks.py`, `data/SRP1/Results/P515S53/w172_appendix_a_audit/` (152 inputs in the manifest) |
| W173 | `1d13858a` | `P5_15_W173_SEARCH_FRAME_AND_COUNTS.md`, `p515_s53_w173_search_counts.py`, `data/SRP1/Results/P515S53/w173_search_counts/` |

- No production file, spec, case file, frozen artifact or manuscript file changed.
- The production code equals every campaign's `git_head_at_run` (W172 checked with a pathspec control).
- W173 deleted the two outputs of its own second trial before re-running. Those files were uncommitted and cited
  nowhere.

## Found

- **W172 coverage:**
  - 141 statements were checked: 120 consistent, 21 rows (H 1, M 3, L 17).
  - Every W166 row A4–A35 is now stated or resolved. The remaining differences are rows A2-1, A3-1 and A3-2.
  - CONFIRM comments at l. 1440, 1502, 1557, 1660 and 1677: confirmed, apart from the A5 ordering rows.
  - CONFIRM l. 1592: (a)–(d) confirmed, (e) differs (item 3).
  - `walker_ni_2011` is present in `bibliography.bib`.
- **Still absent from the text:** cohort activation (vacuous for single-cohort plans); ESSO IPOPT settings; `max_iter`;
  zero-capacity gating; the zeroed shared-ESS usage penalty.
- **W173:** in every s47/s51 poll all 8 rounded directions were inadmissible. Recomputed directions match the recorded
  ones on all 9 polls. The recomputed boxes are 14 (x = 0) and 61 (F2), each equal to the recorded box. "Evaluated in
  the search" agrees with the recorded `in_cache` flag on all 61 F2 neighbours.
- **Predictions:** none could be scored, because Addendum 69 is not in the repository.

## Not confirmed

- **Equation numbering:** no compiled PDF exists, so it was not checked.
- **IPOPT acceptable exit = return code 1:** taken from the Pyomo source, not from an IPOPT log.
- **"44 of 61 never evaluated" is scoped.** It covers all 274 `evaluation_record.json` files under data/ and all 89
  tracked `campaign_results.json` files. Logs, xlsx files, pickles and other branches were not searched.
- **Overleaf after `407f8df`:** not fetched.
