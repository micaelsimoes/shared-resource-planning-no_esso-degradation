# W188c — code reading for ruling 5 at V1 6f43479e and HEAD 5d00d3bd

Script `p515_s55_w188c_code_reading.py`; zero solves; guards 0/0 verified; citations 63/63 hold; integrity OK; rows identical at both heads: True (53 citations at both heads; differing ['EC-07']; touching a diff hunk ['EC-07']).

## Verdicts

- **V1** (1) network complementarity row vs P^Ch P^Dch <= eps^C (S^Av)^2: POSITIVE SCALING ONLY. The coded row is pch_hat * pdch_hat <= eps^C (eps^C = 1e-4) with the linear links P^Ch = S^Av pch_hat, P^Dch = S^Av pdch_hat and pch_hat, pdch_hat in [0, 1]. Substituting the links gives P^Ch P^Dch = (S^Av)^2 pch_hat pdch_hat, so for S^Av > 0 the coded row is the product form divided by (S^Av)^2 (a positive constant: S^Av is a Param in every network solve). The hat bounds add nothing beyond P^Ch + P^Dch <= S^Av. The scaling is carried by two auxiliary variables and two linear rows, not by multiplying the product row by a constant; the projection onto (P^Ch, P^Dch) is identical. At S^Av <= 1e-10 (or E^Av <= 1e-10) the row is not merely vacuous: it is deactivated and P^Ch, P^Dch, the hats and every other storage variable are fixed at 0. (basis: NC-01, NC-02, NC-05, NC-06, NC-07, NC-09, NC-11, NC-12, NC-13, NC-14, NC-15, NC-17, NC-18, ZN-03, ZN-04, ZN-05)
- **V2** (1) ESSO-side complementarity row: NONE EXISTS. The ESSO has no complementarity row at either head (retired in P5.15-1); simultaneous charging and discharging is discouraged by eps^E sum(P^Ch + P^Dch) in the objective. The product form of ruling 5 applies to eq. network_complementarity only; it must not be printed for the agent. (basis: EC-01, EC-02, EC-03, EC-04, EC-05, M-05)
- **V3** (2) cohort allocation and its guard: The row is P^Ch[y_inv] - P^Dch[y_inv] = share * sum_cohorts (P^Ch - P^Dch) (sigma slacks excluded), with share = E^Inv[y_inv] / sum_active E^Inv a Python float (E^Inv = E^Rated,Unit inside the window). Division only when >= 2 cohorts are active and the sum is > 0; with <= 1 active cohort every allocation row of the year is deactivated and the shares are 0. Exactly N_active - 1 rows are active (largest y_inv omitted, supplied by the sum-to-one identity) where the manuscript prints N; equivalent. Inert on SRP1 (one cohort). (basis: A-01, A-02, A-03, A-04, A-05, A-06, A-07, A-08, M-03)
- **V4** (2) cycling exponent and its guard: Coded in product form D * (2 k E^Inv) = 365 Y E^Cell (ageing on) or D = 0 (ageing off): the manuscript equation multiplied by the positive constant 2 k E^Rated,Unit for an active cohort; no division is coded. Built only inside the calendar window; deactivated (D fixed 0, SoH fixed 1) for an inactive cohort. Y is the investment block width of the cohort, which equals Y_y when all blocks have equal width (SRP1: 5-year blocks). (basis: X-01, X-02, X-03, X-04, A-01, ZE-04, M-04)
- **V5** (3) deactivation of inactive cohorts / zero-capacity nodes: Network side: complete -- the ten storage rows deactivated and the storage variables fixed at 0 when S^Av or E^Av <= 1e-10. ESSO side: every cohort-registered row (throughput, D, SoH, floor, power bounds, allocation) is deactivated and its variables fixed for an inactive cohort; the node-level rows (rated S/E sums, available S/E per pair, net-power definition with sigma slacks, converter circle with radius es_s_rated = 0) are NOT deactivated -- they stay active and are satisfied at zero. (basis: ZN-01, ZN-02, ZN-03, ZN-04, ZN-05, ZN-06, ZE-01, ZE-02, ZE-03, ZE-04, ZE-05, ZE-06, ZE-07, ZE-08)
- **V6** differences between V1 (6f43479e) and HEAD for the cited rows: Computed below (head_comparison): every row excerpt present at both heads is byte-identical (EC-07, the ESSO objective, identical modulo indentation, moved into the else branch of C1); f6c23977 shifts line numbers (+4 / +24 / +96 in shared_energy_storage_data.py, +18 in shared_resources_planning.py past l. 5533) and adds the C1/C3 switches, which with their defaults (in_objective False, available_energy_soh_point None) construct nothing. (basis: S-01, S-02, S-03, S-04, EC-07)

## Citations

| id | file | V1 lines | HEAD lines | identical | holds | statement |
|---|---|---|---|---|---|---|
| NC-01 | `network.py` | 393-396 | 393-396 | True | True | Network copy: capacity S is a mutable Param (shared_es_s_rated_fixed); P^Ch, P^Dch are non-negative Vars. |
| NC-02 | `network.py` | 397-402 | 397-402 | True | True | Dimensionless pair pch_hat, pdch_hat in [0, 1], used only by the complementarity row. |
| NC-03 | `network.py` | 500-503 | 500-503 | True | True | Rows wired in every network model (TSO and DSO): the two hat links, the converter circle, the active sum limit. |
| NC-04 | `network.py` | 512-512 | 512-512 | True | True | The complementarity row sess_comp, one per (e, s_m, s_o, t), built from the network parameters. |
| NC-05 | `model_construction_helpers.py` | 874-887 | 874-887 | True | True | Power bound P^Ch + P^Dch <= S (eq. storage_active_power_limit). |
| NC-06 | `model_construction_helpers.py` | 906-930 | 906-930 | True | True | Linear links P^Ch - S * pch_hat = 0 and P^Dch - S * pdch_hat = 0 (no division by S). |
| NC-07 | `model_construction_helpers.py` | 933-955 | 933-955 | True | True | The complementarity row in force (BILINEAR_RELAXATION branch): pch_hat * pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE. |
| NC-08 | `definitions.py` | 34-36 | 34-36 | True | True | Storage model codes. |
| NC-09 | `definitions.py` | 91-93 | 91-93 | True | True | eps^C = 1e-4; the zero-capacity tolerance 1e-10 (network and ESSO tests). |
| NC-10 | `network_parameters.py` | 172-178 | 172-178 | True | True | The shared-storage model is read per network from its params file. |
| NC-11 | `data/SRP1/case9/case9_params.json` | 11-11 | 11-11 | True | True | TSO (case9) uses the BILINEAR_RELAXATION branch. |
| NC-12 | `data/SRP1/case33_1/case33_1_params.json` | 11-11 | 11-11 | True | True | DSO case33_1 uses the BILINEAR_RELAXATION branch. |
| NC-13 | `data/SRP1/case33_2/case33_2_params.json` | 11-11 | 11-11 | True | True | DSO case33_2 uses the BILINEAR_RELAXATION branch. |
| NC-14 | `data/SRP1/case33_3/case33_3_params.json` | 11-11 | 11-11 | True | True | DSO case33_3 uses the BILINEAR_RELAXATION branch. |
| NC-15 | `admm_persistent_workers.py` | 333-339 | 333-339 | True | True | The S in the network rows is the ESSO available capacity S^Av (per node and year, p.u.), set per ADMM cycle. |
| NC-16 | `shared_resources_planning.py` | 6114-6121 | 6132-6139 | True | True | Same assignment of S^Av on the in-process path (shifted +18 lines at HEAD by f6c23977). |
| NC-17 | `shared_energy_storage_data.py` | 274-279 | 278-283 | True | True | S^Av sent to the networks = sum over cohorts of es_s_available_per_unit. |
| NC-18 | `shared_energy_storage_data.py` | 615-619 | 619-623 | True | True | Available S per cohort-year = rated S per cohort-year (no SoH on power). |
| EC-01 | `shared_energy_storage_data.py` | 470-478 | 474-482 | True | True | ESSO side: NO complementarity row. The per-cohort and aggregate complementarity rows and the hat pair are retired. |
| EC-02 | `shared_energy_storage_data.py` | 735-738 | 739-742 | True | True | ESSO side: energy_storage_complementarity, its normalization and the hat links are retired (second statement). |
| EC-03 | `shared_energy_storage_data.py` | 807-812 | 811-816 | True | True | ESSO side: the aggregate complementarity link is retired too. |
| EC-04 | `shared_energy_storage_data.py` | 831-844 | 835-848 | True | True | ESSO side: complementarity is replaced by the objective term eps^E * sum(P^Ch + P^Dch). |
| EC-05 | `definitions.py` | 74-74 | 74-74 | True | True | eps^E = 1e-5. |
| EC-06 | `shared_energy_storage_data.py` | 739-749 | 743-753 | True | True | ESSO power bounds per cohort: P^Ch <= S^Rated,Unit, P^Dch <= S^Rated,Unit (cohort-registered). |
| EC-07 | `shared_energy_storage_data.py` | 851-854 | 875-878 | False | True | ESSO objective with the C1 switch off (default; the only form at V1): feasibility_penalty alone. At HEAD it sits in the else branch of the C1 switch, so it differs from V1 by indentation only (EXPECTED_INDENT_ONLY). |
| A-01 | `shared_energy_storage_data.py` | 560-570 | 564-574 | True | True | Within the calendar-life window E^Rated,Unit[y_inv, y] = E^Inv[y_inv] (a Param). |
| A-02 | `shared_energy_storage_data.py` | 466-468 | 470-472 | True | True | Outside the window E^Rated,Unit and S^Rated,Unit stay fixed at 0 (the lifetime test reads this fixing). |
| A-03 | `shared_energy_storage_data.py` | 760-761 | 764-765 | True | True | Allocation share is a mutable Param (initialised 0), not an expression in Vars. |
| A-04 | `shared_energy_storage_data.py` | 778-784 | 782-788 | True | True | agg_pnet = sum over cohorts of (P^Ch - P^Dch), without the sigma slacks; es_pnet carries the slacks. |
| A-05 | `shared_energy_storage_data.py` | 801-805 | 805-809 | True | True | Allocation row: P^Ch[y_inv] - P^Dch[y_inv] = share[y_inv, y] * agg_pnet, built for every y_inv, cohort-registered. |
| A-06 | `shared_energy_storage_data.py` | 1427-1447 | 1523-1543 | True | True | Per candidate: each cohort state is set first, then the allocation rows and shares once per node. |
| A-07 | `shared_energy_storage_data.py` | 1639-1673 | 1735-1769 | True | True | Allocation docstring: N_active - 1 rows active (largest y_inv omitted; sum-to-one supplies it); shares are floats. |
| A-08 | `shared_energy_storage_data.py` | 1702-1734 | 1798-1830 | True | True | Guard: the only division E^Inv / sum_active E^Inv is a float, reached only with >= 2 active cohorts and a positive sum; with <= 1 active cohort every allocation row of the year is deactivated and every share set 0. |
| X-01 | `shared_energy_storage_data.py` | 652-666 | 656-670 | True | True | Cell throughput row (eq. cell_throughput), cohort-registered, built for every (y_inv, y). |
| X-02 | `shared_energy_storage_data.py` | 675-689 | 679-693 | True | True | Degradation rows exist only inside the calendar window; Y is read from the INVESTMENT block of the cohort. |
| X-03 | `shared_energy_storage_data.py` | 699-711 | 703-715 | True | True | Cycling exponent in product form: D * (2 k E^Inv) = 365 Y E^Cell (ageing on); D == 0 (ageing off). |
| X-04 | `shared_energy_storage_data.py` | 721-732 | 725-736 | True | True | SoH row and floor row: cohort-registered, same family as the D row. |
| ZE-01 | `shared_energy_storage_data.py` | 431-437 | 435-441 | True | True | Per-cohort registry of rows and of the inactive flag. |
| ZE-02 | `shared_energy_storage_data.py` | 1566-1570 | 1662-1666 | True | True | Every row added through _add_esso_cohort_constraint is registered to its cohort. |
| ZE-03 | `shared_energy_storage_data.py` | 1581-1583 | 1677-1679 | True | True | Window test: a (y_inv, y) pair is in the window iff its rated per-unit Vars were unfixed at build. |
| ZE-04 | `shared_energy_storage_data.py` | 1586-1636 | 1682-1732 | True | True | Cohort inactive iff \|S^Inv\| <= 1e-10 and \|E^Inv\| <= 1e-10 (one without the other raises). For an inactive cohort or a pair outside the window: throughput, D, P^Ch, P^Dch fixed 0, SoH fixed 1; every registered row deactivated. |
| ZE-05 | `shared_energy_storage_data.py` | 550-550 | 554-554 | True | True | SoH is built fixed at 1.0 (unfixed only inside the window). |
| ZE-06 | `shared_energy_storage_data.py` | 573-582 | 577-586 | True | True | NOT cohort-registered: node-level rated S and E rows (stay active at zero capacity; they then read 0 = 0 sums). |
| ZE-07 | `shared_energy_storage_data.py` | 615-628 | 619-632 | True | True | NOT cohort-registered: available S/E per cohort-year rows (built for every pair; stay active). |
| ZE-08 | `shared_energy_storage_data.py` | 764-787 | 768-791 | True | True | NOT cohort-registered: node-level net-power definition (with sigma slacks) and converter circle; they stay active at zero capacity, where the circle has radius es_s_rated[y] = 0. |
| ZE-09 | `shared_energy_storage_data.py` | 157-158 | 157-158 | True | True | Public entry that applies the cohort states for a candidate. |
| ZE-10 | `helper_functions.py` | 10-10 | 10-10 | True | True | SHARED_ESS_ZERO_CAPACITY_TOLERANCE reaches shared_energy_storage_data through helper_functions (import *). |
| ZN-01 | `model_construction_helpers.py` | 1002-1013 | 1002-1013 | True | True | The nine network storage variables gated at zero capacity (hats included). |
| ZN-02 | `model_construction_helpers.py` | 1043-1048 | 1043-1048 | True | True | Variables whose box collapses to [0, 0] at zero capacity. |
| ZN-03 | `model_construction_helpers.py` | 1050-1064 | 1050-1064 | True | True | The ten network storage rows gated at zero capacity (sess_comp included). |
| ZN-04 | `model_construction_helpers.py` | 1067-1078 | 1067-1078 | True | True | Network node inactive iff S^Av <= 1e-10 OR E^Av <= 1e-10; values within the tolerance are normalised to 0. |
| ZN-05 | `model_construction_helpers.py` | 1117-1202 | 1117-1202 | True | True | At an inactive node: the nine variables fixed 0, the closure slacks bounded [0, 0], the ten rows deactivated, the gated boxes [0, 0], the expected-schedule variables fixed 0 and their rows deactivated. |
| ZN-06 | `model_construction_helpers.py` | 1088-1114 | 1088-1114 | True | True | Expected storage schedule (consensus copy) fixed at 0 and its definition rows deactivated at an inactive node. |
| S-01 | `shared_energy_storage_parameters.py` | absent | 19-24 | None | True | C1 switch salvage_value.in_objective: default False (HEAD only). |
| S-02 | `shared_energy_storage_parameters.py` | absent | 246-249 | None | True | C3 switch available_energy_soh_point: default None = not declared (HEAD only). |
| S-03 | `shared_energy_storage_data.py` | absent | 176-179 | None | True | C3 is applied only when the ESS parameters file declares it (HEAD only). |
| S-04 | `shared_energy_storage_data.py` | absent | 855-873 | None | True | C1 changes the ESSO OBJECTIVE only (no row); off by default (HEAD only). |
| M-01 | `manuscript_review/main_edd7a63_round8.tex` | absent | 946-952 | None | True | Manuscript at a87e195: the normalised complementarity (divides by S^Av). |
| M-02 | `manuscript_review/main_edd7a63_round8.tex` | absent | 918-925 | None | True | Manuscript: the power bound that makes the product form vacuous at S^Av = 0. |
| M-03 | `manuscript_review/main_edd7a63_round8.tex` | absent | 984-996 | None | True | Manuscript: the cohort allocation (divides by the sum of E^Rated,Unit), stated for all active cohorts. |
| M-04 | `manuscript_review/main_edd7a63_round8.tex` | absent | 1019-1026 | None | True | Manuscript: the cycling exponent (divides by E^Rated,Unit), with Y_y. |
| M-05 | `manuscript_review/main_edd7a63_round8.tex` | absent | 1914-1914 | None | True | Manuscript (Appendix A): ESSO-side complementarity is structural, not a row -- consistent with EC-01..04. |
| M-06 | `manuscript_review/main_edd7a63_round8.tex` | absent | 2117-2117 | None | True | Manuscript: eps^C = 1e-4, eps^E = 1e-5 (= NC-09, EC-05). |

## Files (sha256 at V1 / HEAD)

- `network.py` V1 `18acaa84af7a043898a7b831223b4eab1d91f9bcf2c199f49beecfeb3bf91fb4` HEAD `18acaa84af7a043898a7b831223b4eab1d91f9bcf2c199f49beecfeb3bf91fb4` identical True; at HEAD True
- `model_construction_helpers.py` V1 `a20a224f11008de8c0206484583b00253677267d616e1051bb1583791ed47f64` HEAD `a20a224f11008de8c0206484583b00253677267d616e1051bb1583791ed47f64` identical True; at HEAD True
- `definitions.py` V1 `7e5719af129df111db1b403feeffe0b566d0b7f2e72915829e16bf143a9321af` HEAD `7e5719af129df111db1b403feeffe0b566d0b7f2e72915829e16bf143a9321af` identical True; at HEAD True
- `network_parameters.py` V1 `30ea1b23982c8839f593e98030f99d764704912dfb6b344dd8a2454d06438a92` HEAD `30ea1b23982c8839f593e98030f99d764704912dfb6b344dd8a2454d06438a92` identical True; at HEAD True
- `data/SRP1/case9/case9_params.json` V1 `f3eff0502c606a9076af19969ec3705c2e6204fa4d9172a218345e986dbbf4ab` HEAD `f3eff0502c606a9076af19969ec3705c2e6204fa4d9172a218345e986dbbf4ab` identical True; at HEAD True
- `data/SRP1/case33_1/case33_1_params.json` V1 `8e9e5a536fd1ef2bca5694274a68795d1099884ef23f2134e7db7d3a58c80b8e` HEAD `8e9e5a536fd1ef2bca5694274a68795d1099884ef23f2134e7db7d3a58c80b8e` identical True; at HEAD True
- `data/SRP1/case33_2/case33_2_params.json` V1 `31b5fedf87b96724dc5676566fd87f1c21dc02ecfb8d2bd9fdf280a9b7be35ff` HEAD `31b5fedf87b96724dc5676566fd87f1c21dc02ecfb8d2bd9fdf280a9b7be35ff` identical True; at HEAD True
- `data/SRP1/case33_3/case33_3_params.json` V1 `a19bd5b9d18a26deff44dc62a351f2e6c193012c3f7075d12cade4f230760c49` HEAD `a19bd5b9d18a26deff44dc62a351f2e6c193012c3f7075d12cade4f230760c49` identical True; at HEAD True
- `admm_persistent_workers.py` V1 `39dcbd267bb224e65197ca427952b08d45d3bbe98785b73fd55148c0979e1125` HEAD `39dcbd267bb224e65197ca427952b08d45d3bbe98785b73fd55148c0979e1125` identical True; at HEAD True
- `shared_resources_planning.py` V1 `ade9c843bc7c5b6514ab91ecc54f327330a5249709766699ff2f50c8288473d6` HEAD `c9336a61a713ef6890fbffcda845395d757e26c41ae2422e82dd88bb881c33e8` identical False; at HEAD True
- `shared_energy_storage_data.py` V1 `9acd095fb5e50e9b64c253600ae05ebb8e6463fd7a999d3a90f96ee82f037796` HEAD `e8bf4f923e28db83480e786340c32a291eca56288fa17f15ecac7fb72fc3cb85` identical False; at HEAD True
- `helper_functions.py` V1 `808c60a08b8e157381f28e0c1da5741b0efd39e35f36dd531f88b84deebce1e2` HEAD `808c60a08b8e157381f28e0c1da5741b0efd39e35f36dd531f88b84deebce1e2` identical True; at HEAD True
- `shared_energy_storage_parameters.py` V1 `5fe0c4fb876a9c2f1e506e982f2a120c320a3fd4198b870f877d9a310cd390bb` HEAD `f31a2caa83a7c07f8f2375bf2016dd2180ff470ee33c1df0a04ece038d46aff6` identical False; at HEAD True
- `manuscript_review/main_edd7a63_round8.tex` V1 `cec2e2982dfa1509e6e066a208521fb1eea40d5c71f1cdf71a9bf00a12b3520b` HEAD `cec2e2982dfa1509e6e066a208521fb1eea40d5c71f1cdf71a9bf00a12b3520b` identical True; at HEAD True
