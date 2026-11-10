# P5.15 W188 (c): code reading for ruling 5 (Addendum 89, zero-capacity conventions, G1)

Zero solves, no model build, no pickle, no production import. Code read as text at two heads:

- **V1** is `6f43479e`, spec v1 of Addendum 85. The frozen-convention runs were made on it.
- **HEAD** is `5d00d3bd`, which descends from `75d8e12a` and includes `f6c23977` (P5.16 C1 + C3 + C4).

Objective convention: not applicable, because this report gives no objective value. Problem instance: SRP1, `data/SRP1/SRP1.json`. Its seven case and parameter files have the same blobs at both heads.

## Blocked on Planner

None. Two points should reach the expert before round 9 is pasted. Neither is a formulation change, and neither breaks the scaling condition of ruling 5.

1. **The storage agent (ESSO) has no complementarity row, at either head.** The product form in ruling 5 applies to `eq:network_complementarity` only. Printing a complementarity row for the agent would describe a row the code does not have. The manuscript already says so at l. 1914, and that agrees with the code.
2. **"Inactive storage rows are deactivated" is only partly true on the agent side.** It holds completely on the network side, and for every row attached to a single investment cohort ("cohort-registered") in the agent. Five node-level agent rows stay active at zero capacity and are satisfied only at zero:
   - rated S and E sums;
   - available S and E per (cohort, year) pair;
   - the net-power definition with its σ± slacks;
   - the converter circle, whose radius `es_s_rated[y]` is then 0.

   Suggested wording: "the rows of inactive cohorts are deactivated".

## Changed

- `5d00d3bd`: harness `p515_s55_w188c_code_reading.py` (sha256 `fa739860…`), committed before the run. How it works:
  - It holds 63 citations. Each one records the file, its line range at V1 and at HEAD, and substrings that must appear in each range.
  - It records each excerpt and its sha256 at both heads, and checks whether either range touches a hunk of `git diff -U0 V1 HEAD`.
  - It requires the working tree to equal the HEAD blob.
  - Components that exist only at HEAD (the P5.16 switches) must be absent from the whole V1 file.
  - Guards: `SolveProfileGuard(permitted=())`, verified at 0, and pickle loading blocked (0/0). JSON is written through `gate_result_io`.
- Run (attached, stdout and stderr captured): exit 0. **63/63 citations hold, 0 integrity failures**, solve guard `{permitted_solve 0, blocked_solve 0}`, pickle `{load 0, loads 0}`.
- Outputs, in a new directory `data/SRP1/Results/P515S55/w188c_code_reading/`:

  | File | sha256 |
  |---|---|
  | `w188c_code_reading.json` | `44fc65be…` |
  | `w188c_code_reading.md` | `65088727…` |
  | `manifest_sha256.json` | `ecc09a2d…` |
  | `launch.log` | `f522c334…` |
  | `manifest_post_run_sha256.json` | — |

- No production code, spec, TASKS.md, REVISION_CONTEXT.md or brief was edited.

## Found

### Scaling verdict for (1): the coded row differs from the product form by a positive scaling only

The row in force is the BILINEAR_RELAXATION branch of `sess_comp_rule`. All four case parameter files select that branch.

```
pch_hat · pdch_hat ≤ ESS_COMPLEMENTARITY_TOLERANCE (= SMALL_TOLERANCE = 1e-4 = ε^C)
```

It works together with these other rows and variables:

- two linear links, `P^Ch − S·pch_hat = 0` and `P^Dch − S·pdch_hat = 0`;
- the hat variables bounded to [0, 1];
- the power bound `P^Ch + P^Dch ≤ S`;
- S is the mutable Param `shared_es_s_rated_fixed`, set to the agent's S^Av / s_base.

Substituting the links gives `P^Ch·P^Dch = S²·pch_hat·pdch_hat`. So for S^Av > 0 the coded row is **P^Ch·P^Dch ≤ ε^C (S^Av)² divided by (S^Av)²**, which is a positive constant in each network solve.

- The projection onto (P^Ch, P^Dch) is the same as the product form's.
- The hat bounds add nothing beyond the power bound.
- The code's own comment says so: "`pch * pdch <= eps * S_rated^2` -- substituting the link equalities reproduces it identically".

Only the representation differs: the scaling is carried by two auxiliary variables and two linear rows. At S^Av = 0 the row is more than vacuous: it is **deactivated**, and P^Ch, P^Dch, the hats and every other storage variable are fixed at 0 (see (3)).

### Paragraph for the manuscript side (draft for the expert; facts cited in the table)

In every network model, simultaneous charging and discharging is limited by the complementarity condition P^Ch·P^Dch ≤ ε^C (S^Av)², with ε^C = 10⁻⁴. The code imposes it on the normalised powers P^Ch/S^Av and P^Dch/S^Av, which is the same condition for S^Av > 0, and never divides by S^Av. When the available converter or energy rating of a node is zero (≤ 10⁻¹⁰ p.u.), the node's storage block is inactive: its powers, state of charge and closure slacks are fixed at zero and its storage rows, including the complementarity row, are removed. The storage agent carries no complementarity row: the linear throughput weight ε^E = 10⁻⁵ makes simultaneous charging and discharging non-optimal. Cohort allocation and the cycling exponent are defined for active cohorts, E^Rated,Unit > 0:

- the exponent is imposed in multiplied form, 2 k_e E^Rated,Unit D = 365 Y Ē^Cell;
- the allocation share E^Rated,Unit / Σ E^Rated,Unit is computed from the installed energies before the solve, and only when two or more cohorts are active.

For an inactive cohort, its throughput, cycling-exponent, state-of-health, floor, power-limit and allocation rows are removed, its powers and exponent are fixed at zero and its state of health at one.

### Row table (lines at V1 `6f43479e` / HEAD; MS = manuscript `main_edd7a63_round8.tex`, md5 `39f6fd86…` = Overleaf a87e195)

| # | Quantity | MS (l.) | Code row (file: V1 / HEAD) | Exact coded form | Relation to the MS / ruling-5 form | Zero-capacity / inactive handling (file: V1 / HEAD) |
|---|---|---|---|---|---|---|
| 1a | Complementarity, network (TSO and DSO) | 946–952 (normalised, divides by S^Av) | `model_construction_helpers.py` 933–955 / 933–955 (row at 949–951); wired `network.py` 512 / 512 | `pch_hat·pdch_hat ≤ 1e-4` | Product form ÷ (S^Av)², a positive scaling; equivalent for S^Av > 0 | Row `sess_comp` deactivated, hats fixed at 0, when S^Av ≤ 1e-10 **or** E^Av ≤ 1e-10 (`model_construction_helpers.py` 1050–1064, 1067–1078, 1117–1202, same at both heads) |
| 1b | Hat links | — | `model_construction_helpers.py` 906–930 / 906–930; hat vars `network.py` 397–402 / 397–402 | `P^Ch − S·pch_hat = 0`, `P^Dch − S·pdch_hat = 0`, hats in [0, 1] | Carry the scaling; no division by S | Deactivated (same lines as 1a) |
| 1c | Power bound | 918–925 | `model_construction_helpers.py` 874–887 / 874–887 | `P^Ch + P^Dch ≤ S` | Same as the MS | Deactivated; pch, pdch, pnet, qnet boxes set to [0, 0] (1186–1196) |
| 1d | S in the network rows | S^Av_{e,y} | `admm_persistent_workers.py` 333–339 / 333–339; `shared_resources_planning.py` 6114–6121 / 6132–6139; `shared_energy_storage_data.py` 274–279 / 278–283, 615–619 / 619–623 | S = Σ_cohorts `es_s_available_per_unit` / s_base, with `es_s_available_per_unit = es_s_rated_per_unit` | = S^Av | — |
| 1e | ε^C, zero tolerance | 2117 (ε^C = 1e-4) | `definitions.py` 91–93 / 91–93 | `ESS_COMPLEMENTARITY_TOLERANCE = SMALL_TOLERANCE = 1e-4`; `SHARED_ESS_ZERO_CAPACITY_TOLERANCE = 1e-10` | Matches | — |
| 1f | Complementarity, ESSO | 1914 (structural, no row) | `shared_energy_storage_data.py` 470–478 / 474–482, 735–738 / 739–742, 807–812 / 811–816 (retired); objective 831–844 / 835–848 | **No row.** Objective term `EPS_ESSO_THROUGHPUT·Σ(pch + pdch)`, ε^E = 1e-5 (`definitions.py` 74) | Product form not applicable | pch, pdch fixed at 0 for inactive cohorts (row 3a) |
| 2a | Cohort allocation | 984–996 (divides by Σ E^Rated,Unit; N rows over Y^Act) | Row `shared_energy_storage_data.py` 801–805 / 805–809; config 1702–1734 / 1798–1830 | `P^Ch[i] − P^Dch[i] = share[i,y] · Σ_cohorts(P^Ch − P^Dch)` (σ± slacks excluded, 778–784 / 782–788); `share = E^Inv[i] / Σ_active E^Inv` is a Python float pushed into a mutable Param (760–761 / 764–765) | Equivalent. E^Inv = E^Rated,Unit inside the window (560–570 / 564–574). **N_active − 1 rows active** (the largest y_inv is omitted; the sum-to-one identity supplies it) where the MS prints N | Guard: division only with ≥ 2 active cohorts **and** Σ > 0. With ≤ 1 active cohort, every row of the year is deactivated and shares are 0 (1709 / 1805). Rows are cohort-registered, so they are also deactivated per cohort (row 3a). Inert on SRP1 (one cohort) |
| 2b | Cell throughput | 1000–1012 | `shared_energy_storage_data.py` 652–666 / 656–670 | `avg = Σ_d (D_d/365) Σ_t (η_ch pch dt + pdch dt/η_dch)` | Same as the MS | Cohort-registered: deactivated, avg fixed at 0 (row 3a) |
| 2c | Cycling exponent | 1019–1026 (divides by 2 k E^Rated,Unit) | `shared_energy_storage_data.py` 699–711 / 703–715 | `D · (2·cl_eff·E^Inv) = 365·Y·avg` with ageing on; `D == 0` with ageing off | MS equation × 2 k E^Rated,Unit, a positive scaling for an active cohort; no division coded. Y = `years[repr_years[y_inv]]`, the cohort's **investment** block width, which equals Y_y when all blocks are equal (SRP1: 5 years) | Built only inside the calendar window (675–689 / 679–693). Cohort-registered: deactivated, D fixed at 0, SoH fixed at 1 (row 3a) |
| 3a | Inactive cohort, ESSO | — | `shared_energy_storage_data.py` 1586–1636 / 1682–1732; registry 431–437 / 435–441, 1566–1570 / 1662–1666; window 1581–1583 / 1677–1679; applied per candidate 1427–1447 / 1523–1543 | A cohort is inactive iff \|S^Inv\| ≤ 1e-10 **and** \|E^Inv\| ≤ 1e-10; one without the other raises `ValueError` | — | For an inactive cohort, or a pair outside the window: avg, D, pch, pdch fixed at 0 and SoH at 1. **Every** cohort-registered row is deactivated: throughput, D, SoH, floor (721–732 / 725–736), the `energy_storage_limits` power bounds (739–749 / 743–753) and allocation |
| 3b | ESSO node-level rows | — | `shared_energy_storage_data.py` 573–582 / 577–586, 615–628 / 619–632, 764–787 / 768–791 | `es_s/e_rated[y] = Σ rated_per_unit`; `e_available = e_rated_per_unit · SoH`; `es_pnet = agg_pnet + σ⁺ − σ⁻`; `es_pnet² + es_qnet² ≤ es_s_rated[y]²` | — | **Not cohort-registered and not deactivated.** At x = 0 they stay active and are satisfied at zero; the circle has radius 0 |
| 3c | Zero-capacity network node | 2934 ("closure row is deactivated … slacks fixed at zero") | `model_construction_helpers.py` 1002–1013, 1043–1048, 1050–1064, 1088–1114, 1117–1202 (same at both heads) | — | — | At S^Av or E^Av ≤ 1e-10 (normalised to 0): 9 variables fixed at 0, closure slacks bounded [0, 0], 10 rows deactivated, expected-schedule variables fixed at 0 and their definition rows deactivated |

### Differences between the two heads

- **The cited rows are identical.**
  - 53 citations exist at both heads; 52 of their excerpts are byte-identical.
  - The 53rd, EC-07 (the ESSO objective `feasibility_penalty`), differs **by indentation only**: `f6c23977` moved it into the `else` branch of the C1 switch.
  - No other cited range touches a diff hunk.
  - `model_construction_helpers.py`, `network.py`, `network_parameters.py`, `definitions.py`, `admm_persistent_workers.py`, `helper_functions.py` and the four case parameter files are byte-identical at V1 and HEAD.
- **Line shifts** (all from `f6c23977`):
  - `shared_energy_storage_data.py`: +4 after l. 175, +24 after l. 854, +96 after l. 908.
  - `shared_resources_planning.py`: +18 after l. 5526.
- **What else changed in `shared_energy_storage_data.py`:** a 72-line salvage-objective builder (HEAD 934–1005) and 42 diagnostic lines (HEAD 2213–2254). Neither builds a row.
- **The switches at their defaults construct nothing.**
  - C1 `salvage_value.in_objective` defaults to False (`shared_energy_storage_parameters.py` HEAD 19–24). It changes only the ESSO objective, never a row (`shared_energy_storage_data.py` HEAD 855–878).
  - C3 `available_energy_soh_point` defaults to None (HEAD 246–249). It is applied only when the parameters file declares it (`shared_energy_storage_data.py` HEAD 176–179). If 'mid' is declared, only the available-energy row changes, not rows 1–3.
  - The frozen instance's `SRP1_ESS_Params.json` declares neither key and is identical at both heads. Only the P5.16 file `SRP1_5x3y_ESS_Params.json` declares them (outside the frozen convention).
- **None of the HEAD-only switch strings occurs anywhere in the V1 files** (asserted).

### Further observations (code text; for the Planner)

- **The two zero-capacity tests use different logic.** The network node is inactive if S^Av **or** E^Av ≤ 1e-10. The agent's cohort is inactive only if S^Inv **and** E^Inv ≤ 1e-10, and a mismatch raises. Both use a 1e-10 threshold, not exact zero; the network side also normalises values within the tolerance to 0.
- **Y differs between code and manuscript.** The manuscript's Y_y is the width of block y. The code uses the width of the cohort's investment block (the code comment at 672–674 / 676–678 records this). The two are equal on SRP1.

## Not confirmed

- **Active and inactive status was read from the code, not seen on a built model.** No model was built or loaded, so no `.active` flag or `.fixed` value of a constructed model was inspected. W182E Z1 (b) is the only empirical evidence on record: identical x = 0 ESSO NL files under a change of ageing parameters.
- **Which head each frozen run used was not re-checked.** I did not check that every frozen-convention run's record pins `6f43479e`. I took V1 from the task and from `f7c3f601`'s message ("spec v1 (production 6f43479e)"). I verified only that the SRP1 instance files are identical at both heads.
- **Branches not in force were not checked.** The EXACT and POLYNOMIAL_COMPLEMENTARITY branches of `sess_comp_rule` (lines 947–948 and 952–953) are not selected by any SRP1 file, and I did not check them further. Nor did I search for other callers of `configure_shared_ess_operational_state` beyond those at `admm_persistent_workers.py` 333–339 and `shared_resources_planning.py` 4322/4418/4536/6134/6466/6676 (HEAD).
- **The manuscript was compared through the repository copy only.** I used `manuscript_review/main_edd7a63_round8.tex` (md5 matches Addendum 89's a87e195) and did not access Overleaf. The launch log's "V1-HAS" entries on the M-* lines only show that this file already held the same text at V1. They are informational and not a check.
