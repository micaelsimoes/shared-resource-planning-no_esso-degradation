# W182e — code paths Z1 / Z6 / C4 at HEAD 2f4df40c

Script `p515_s53_w182e_code_paths.py`; zero solves; guards 0/0 verified; citations 93/93 hold; integrity OK.

## Predictions

- **Z1** Q(0) exactly independent of the ageing model (expert): **HELD (by code reading; empirical NL identity on record only for an ageing-ON parameter change)** (basis: Z1b-01..Z1b-08, Z1b-09/10, Z1b-11, x0_diff_analysis, S5 any_no_ageing_entry_at_x0)
- **Z6** rho_0 is a configuration parameter (expert): **HELD in production (case-file key, no production change); NOT runnable through the campaign harness without a harness change (child refuses rho != S39; rho absent from the eval key)** (basis: Z6b-01..05, Z6b-07..11, R_rho_in_eval_key_region)
- **Z6** the NRF row needs a new declared option (code change, both copies) (expert): **HELD: no NRF option exists in the coordinated path; the only NRF row is the uncoordinated DSO arm's** (basis: Z6c-01..11, S1_reverse_flow_in_production)

## Citations

| id | file:lines | holds | excerpt sha256 | statement |
|---|---|---|---|---|
| Z1a-01 | `shared_energy_storage_data.py`:58-69 | True | `afcae5e1a492` | No-ageing is an attribute of SharedEnergyStorageData (default True), changed only by a model-variant evaluation of the campaign harness, never by a case file. |
| Z1a-02 | `shared_energy_storage_data.py`:1529-1563 | True | `67d7a290593d` | The production reader of the switch; documents D == 0, phi == 1, SoH == 1, floor row kept. |
| Z1a-03 | `shared_energy_storage_data.py`:676-711 | True | `dfa5a2afb644` | Ageing off: the D row is D == 0 (k never consumed -- not a k -> infinity limit) and phi_cal is forced to 1.0 whatever calendar_retention_per_year says; ageing on: D * (2 k E_inv) == 365 n avg_ch_dch (product form). |
| Z1a-04 | `shared_energy_storage_data.py`:713-732 | True | `5d538bc62204` | SoH row SoH_y = SoH_{y-1} exp(-D_y) phi^n with SoH_{y_inv-1} = 1; the floor row SoH >= soh_min stays BUILT and ACTIVE under no-ageing (non-binding at SoH = 1), i.e. "floor inactive" holds as non-binding, not as deactivated. |
| Z1a-05 | `shared_energy_storage_data.py`:604-628 | True | `054257a8622b` | Available energy E_av = E_rated * SoH ('end', the E1 setting); 'mid' uses the mid-block SoH with phi forced to 1.0 when ageing is off. |
| Z1a-06 | `p515_s44_campaign_harness.py`:336-339 | True | `af1f65b561e2` | model_variant is a per-entry evaluation option of the campaign harness. |
| Z1a-07 | `p515_s44_campaign_harness.py`:366-370 | True | `bc2bbf5cdcea` | Exactly four keys; eol_retention_r is REQUIRED even when ageing is off (it is then inert in the model). |
| Z1a-08 | `p515_s44_campaign_harness.py`:646-650 | True | `a0a8dd46d93c` | eol_retention_r in (0,1) and calendar retention in (0,1]: k = N D/(-ln R) is always finite, so no case-file or variant value reaches k -> infinity; the only exact no-ageing route is ageing_enabled = False. |
| Z1a-09 | `p515_s44_campaign_harness.py`:2302-2346 | True | `0d140fdfe1c4` | The variant is applied in the child to the deep-copied shared-ESS data BEFORE any ESSO model is built, through production's own apply_to and read back by production's own settings reader. |
| Z1a-10 | `p515_s44_campaign_harness.py`:2998-3010 | True | `419903a77faa` | Where the child applies it: the configuration hook (run_admm_arm pre_solve_hook). |
| Z1a-11 | `p515_s44_campaign_harness.py`:1126-1133 | True | `b9120473f586` | The model variant enters the evaluation key, so aged and no-ageing evaluations of one candidate never share a cache key (and two no-ageing declarations differing only in the inert eol_retention_r get different keys). |
| Z1a-12 | `p515_s47_phase_b_record.py`:874-875 | True | `fbeb84683bfd` | The committed Phase-B search launcher (PB, imported by every later search) asserts NO model variant. |
| Z1a-13 | `p515_s51_f2_phase_b.py`:362-363 | True | `0ee10dd1b171` | The 2x (F2) Phase-B search launcher asserts NO ageing model variant on its cache sources ... |
| Z1a-14 | `p515_s51_f2_phase_b.py`:851-852 | True | `0ee10dd1b171` | ... and on its own frozen spec: a no-ageing SEARCH is not runnable through the committed search launchers without a launcher change (harness-level; no production change). |
| Z1a-15 | `p515_s46_ageing_campaign.py`:148-154 | True | `2622bb9e4cc1` | The original W20 no-ageing arm declaration (s46, C3-era): eol 0.50 is inherited from the C3 calibration. |
| Z1b-01 | `shared_energy_storage_data.py`:1566-1570 | True | `4381d87ce8a4` | Every row built through _add_esso_cohort_constraint is registered to its investment cohort. |
| Z1b-02 | `shared_energy_storage_data.py`:666-666 | True | `23fc35cff4ce` | The throughput-definition row is cohort-registered. |
| Z1b-03 | `shared_energy_storage_data.py`:700-732 | True | `0496362c7abf` | The D row, the SoH row and the floor row (every ageing-parameter-bearing ESSO row: k, phi, n, soh_min) are cohort-registered. |
| Z1b-04 | `shared_energy_storage_data.py`:1586-1636 | True | `97ec19d40f7e` | At x = 0 every cohort is inactive: avg_ch_dch, D fixed 0, SoH fixed 1.0, pch/pdch fixed 0, and EVERY cohort-registered row (incl. D, SoH, floor) deactivated. |
| Z1b-05 | `shared_energy_storage_data.py`:1441-1447 | True | `aaa2bc528c72` | Both are applied for every candidate (the production entry for the ESSO candidate). |
| Z1b-06 | `shared_energy_storage_data.py`:1709-1714 | True | `2a7ceaaaf1fe` | Cohort-allocation (H3) rows: all deactivated, shares 0.0, when <= 1 cohort is active (x = 0: none). |
| Z1b-07 | `shared_energy_storage_data.py`:548-550 | True | `3aa5a61b84f7` | SoH initialisation: built fixed at 1.0 (unfixed only inside an active window). |
| Z1b-08 | `shared_energy_storage_data.py`:846-854 | True | `188bf016ee5e` | The ESSO objective is the feasibility penalty only; the salvage is an Expression outside every objective. |
| Z1b-09 | `shared_resources_planning.py`:1302-1306 | True | `aaaffeb9397f` | Q (gross) is the network primal values less transfers; the salvage enters only net_operational_recourse. |
| Z1b-10 | `shared_resources_planning.py`:1237-1239 | True | `7b6123a8e7f2` | Q is summed over TSO and DSO blocks only (no ESSO term). |
| Z1b-11 | `shared_energy_storage_data.py`:615-628 | True | `20f2b8f2f807` | The available-capacity rows are NOT cohort-registered (active at x = 0). In 'end' mode they read E_av = E_rated * SoH with SoH FIXED at 1.0 (no ageing parameter); in 'mid' mode the active row carries the coefficient phi**(n/2) (with D fixed 0, prev SoH fixed 1) -- the only ageing parameter that can reach an active ESSO row at x = 0, on a variable forced to 0. Both E1 arms use 'end'. |
| Z1b-12 | `shared_energy_storage_data.py`:562-570 | True | `e45145da7590` | The cohort window depends on t_cal (calendar_life_years); model_variant never changes t_cal (Z1a-07), so both E1 arms share the window. |
| Z1b-13 | `model_construction_helpers.py`:1117-1121 | True | `2934e5ecea3e` | Network side: shared-ESS rows read only the capacity Params; inactive when S or E is zero. |
| Z1b-14 | `admm_persistent_workers.py`:333-339 | True | `c01ceee53573` | The networks receive the ESSO AVAILABLE capacity (SoH-derated); at x = 0 it is 0 under any ageing model. |
| Z1b-15 | `shared_resources_planning.py`:2944-2947 | True | `bcc0e5966523` | The price-taker initialisation (the other ageing-reading path, shared_ess_price_taker.py) runs only in mode 'price_taker' ...' |
| Z1b-16 | `p515_s44_campaign_harness.py`:2910-2910 | True | `494958a48b39` | ... and every campaign child refuses unless the mode is 'standalone'. |
| Z1b-17 | `p515_s47_baseline_campaign.py`:35-38 | True | `77d69c0f80e6` | Prior empirical evidence (C3 -> baseline parameter change, ageing ON in both): x = 0 ESSO NL files byte-identical. |
| Z1c-01 | `model_construction_helpers.py`:906-931 | True | `9967df39fabf` | Normalised complementarity, network side: pch = S_av * pch_hat (product form, no division by S_av). |
| Z1c-02 | `model_construction_helpers.py`:949-951 | True | `d6d2af41c871` | The complementarity row in force (BILINEAR_RELAXATION) is pch_hat * pdch_hat <= tol on the dimensionless pair. |
| Z1c-03 | `network.py`:401-402 | True | `a59169c343b9` | pch_hat, pdch_hat in [0, 1]. |
| Z1c-04 | `model_construction_helpers.py`:1067-1069 | True | `f208b8a65d54` | Zero test on the network side: S_av OR E_av <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE. |
| Z1c-05 | `definitions.py`:93-93 | True | `8949550aee3c` | The tolerance. |
| Z1c-06 | `model_construction_helpers.py`:1137-1178 | True | `4fcb597ebdc6` | At zero capacity: every shared-ESS operational Var (incl. pch_hat/pdch_hat) fixed 0 and every row of _SHARED_ESS_OPERATIONAL_CONSTRAINTS (incl. sess_comp and the links) deactivated. |
| Z1c-07 | `model_construction_helpers.py`:1050-1064 | True | `5801b780f587` | The deactivated row list includes the complementarity and link rows. |
| Z1c-08 | `shared_energy_storage_data.py`:1588-1593 | True | `b2b6a2727d2e` | ESSO side: a cohort is inactive iff S_inv = E_inv = 0 (inconsistent zero raises). |
| Z1c-09 | `shared_energy_storage_data.py`:1716-1726 | True | `17c79f35ddaa` | Cohort allocation: the only division (share = E_inv / sum E_inv) is a Python float, guarded (> 0) and reached only with >= 2 active cohorts. |
| Z1c-10 | `shared_energy_storage_data.py`:785-787 | True | `f4335a000bbe` | OBSERVATION: the ESSO aggregate capability row is not cohort-registered: at x = 0 it stays active as a radius-0 circle (es_s_rated forced 0 by active equalities) -- no division, ageing-independent, degenerate. |
| Z1c-11 | `shared_energy_storage_data.py`:2213-2215 | True | `26c0532c9eee` | Reported SoH = E_av / E_rated is guarded at E_av = 0. |
| Z1c-12 | `shared_resources_planning.py`:5024-5028 | True | `72fcaa4db48b` | ESS-channel ADMM normalisation uses the declared reference rating (case file 2.5 MVA), not S_av. |
| Z6a-01 | `uncoordinated_benchmark.py`:912-945 | True | `7334c20247ca` | The benchmark perturbation: per free Var of ONE static block, multiplicative, seeded by (seed, crc32(label)). |
| Z6a-02 | `uncoordinated_benchmark.py`:948-970 | True | `fd1dc7bb6876` | It is applied after a warm start from the CERTIFIED coordinated point, to the arm models before their one solve. |
| Z6a-03 | `uncoordinated_benchmark.py`:1103-1119 | True | `c49b4d127ca9` | Its only call sites: the uncoordinated arm runner. |
| Z6a-04 | `shared_resources_planning.py`:2858-2882 | True | `87c793823a99` | Coordinated ADMM, fresh start: models are BUILT AND INITIALISATION-SOLVED inside _run_operational_planning; no callback exists between build and first solve. |
| Z6a-05 | `shared_resources_planning.py`:2964-2971 | True | `d4da1e4700e1` | The only start alternative in production: a continuation from a full prior state (models, consensus, duals). |
| Z6a-06 | `p515_g_g1_g4_admm_gates.py`:1194-1198 | True | `eb032a83dad1` | The campaign child's only hook runs before any model exists ... |
| Z6a-07 | `p515_g_g1_g4_admm_gates.py`:1283-1285 | True | `6f011a366bfd` | ... and the run is always a fresh (cold) start: no initial_state is passed. |
| Z6b-01 | `admm_parameters.py`:511-513 | True | `9294e78bf484` | rho_0 is read from the case file (admm.rho.{v,pf,ess}.<network>). |
| Z6b-02 | `shared_resources_planning.py`:5112-5122 | True | `0fa055e9005d` | TSO: rho_0 initialises the mutable rho Params. |
| Z6b-03 | `shared_resources_planning.py`:5378-5388 | True | `31b0a74cb529` | DSO: likewise. |
| Z6b-04 | `shared_resources_planning.py`:5483-5483 | True | `6e54bb1f712a` | ESSO: likewise. |
| Z6b-05 | `shared_resources_planning.py`:8275-8284 | True | `5fdeda928321` | The adaptive rule (case file adaptive_penalty true) moves rho from rho_0 by the declared factors. |
| Z6b-06 | `p59_rho.py`:89-104 | True | `ea5f6091c0b6` | An existing diagnostic helper that sets rho on a deep-copied planning (used by earlier G-gate arms). |
| Z6b-07 | `p515_s44_campaign_harness.py`:327-330 | True | `38cb96201c8c` | The campaign harness admits NO rho override ... |
| Z6b-08 | `p515_s44_campaign_harness.py`:2897-2899 | True | `6feaf20e4f32` | ... and every child REFUSES unless rho equals the S39 constants ... |
| Z6b-09 | `p515_s44_campaign_harness.py`:2924-2925 | True | `8086c3b43331` | ... (the refusal). |
| Z6b-10 | `p515_g_g1_g4_admm_gates.py`:5878-5880 | True | `2542b378c89d` | The constants the child compares against (= the case file today). |
| Z6b-11 | `p515_s44_campaign_harness.py`:1072-1088 | True | `7c63864aa0b0` | The evaluation-key payload (read in full by the scoped search R-rho): rho is not a key component, so a case-file rho edit would not separate cache keys. |
| Z6c-01 | `uncoordinated_benchmark.py`:192-212 | True | `b2385e2f1ffb` | The only NRF row in the repository: the uncoordinated DSO arm, DSO copy only; stated absent from the coordinated path. |
| Z6c-02 | `uncoordinated_benchmark.py`:336-338 | True | `870c8a0a339a` | The row rule. |
| Z6c-03 | `uncoordinated_benchmark.py`:405-407 | True | `53f07211d00b` | Added only in build_dso_arm_models(no_reverse_flow=True). |
| Z6c-04 | `network.py`:438-445 | True | `279ca18fb3f8` | The two interface copies in the coordinated model: TSO pc_adn (per ADN), DSO pg_adn. |
| Z6c-05 | `model_construction_helpers.py`:1241-1262 | True | `417b7a4a7721` | TSO copy = fixed anchor pc (the consensus request) + free signed interface_delta_p (scenario-free). |
| Z6c-06 | `model_construction_helpers.py`:1287-1297 | True | `241c24319de4` | DSO copy = reference-generator P MINUS the shared-ESS net power at the reference bus (equal to pg_ref at x = 0). |
| Z6c-07 | `shared_resources_planning.py`:4233-4242 | True | `f4fb02b1695a` | TSO builder: pc fixed to the DSO request (the anchor). |
| Z6c-08 | `shared_resources_planning.py`:4267-4273 | True | `3226a80b590e` | TSO builder: delta freed within +/- the interface rating -- a TSO-side NRF would bind here (pc + delta >= 0). |
| Z6c-09 | `shared_resources_planning.py`:4277-4282 | True | `8e40ff80ffcd` | TSO builder site for a declared TSO-copy NRF row (create_transmission_network_model). |
| Z6c-10 | `shared_resources_planning.py`:4372-4377 | True | `f11044a832fc` | DSO builder site, sequential (create_distribution_networks_models_sequential) ... |
| Z6c-11 | `shared_resources_planning.py`:4488-4488 | True | `7c3b8155a3d7` | ... and parallel (create_distribution_network_model): a DSO-copy option must be wired in both builders. |
| Z6c-12 | `network_parameters.py`:168-174 | True | `f367780cf233` | Precedent for a declared per-network option read from the case33/case9 params files. |
| Z6c-13 | `shared_resources_planning.py`:1154-1157 | True | `53402ab4fd4f` | What is recorded for interface P on each side: the expected interface P of the TSO copy and of the DSO copy ... |
| Z6c-14 | `shared_resources_planning.py`:1209-1220 | True | `d4456afe3c48` | ... per (node, year, day, period), with the TSO delta and anchor per scenario and the price ... |
| Z6c-15 | `p515_g_g1_g4_admm_gates.py`:2051-2058 | True | `09cc9fb8f771` | ... written at TERMINAL only, to interface_settlement_detail_s31c.json in every evaluation directory. |
| Z6c-16 | `p515_g_g1_g4_admm_gates.py`:1935-1935 | True | `bb8727cb5a4c` | The writer reads production's reporting helper. |
| C4-01 | `p515_s44_campaign_harness.py`:11-20 | True | `da0738000c1b` | The harness runs N children, one process per evaluation, at most ctx.concurrency at once. |
| C4-02 | `p515_s44_campaign_harness.py`:59-65 | True | `31b34be36a3d` | Its isolation argument (checked at Addendum 25). |
| C4-03 | `p515_s44_campaign_harness.py`:312-313 | True | `fd6a68bb0178` | One campaign lock at the repo root: one campaign (one spec) per machine; slots live inside it. |
| C4-04 | `p515_s44_campaign_harness.py`:1630-1630 | True | `c8c1878d0483` | Concurrency is a spec field. |
| C4-05 | `p515_s44_campaign_harness.py`:1872-1872 | True | `0b2c1f956f2d` | The slot loop. |
| C4-06 | `p56a_oracle.py`:174-186 | True | `f45d60447e52` | Per-evaluation log directory for every TSO/DSO/ESSO holder (the IPOPT output_file path). |
| C4-07 | `shared_energy_storage_data.py`:29-38 | True | `9387168c5499` | The ESSO log fallback is the cwd -- harmless only because fresh_planning sets logs_dir (C4-06). |
| C4-08 | `p515_s44_gate.py`:1-21 | True | `6626058eb115` | The ONLY concurrent-vs-serial bitwise gate: 3 slots, C* against the D run of P515S39 (pinned reference). |
| C4-09 | `p515_s44_gate.py`:38-39 | True | `1801636fa29b` | Its gating scalars are literals of the pre-baseline D configuration. |
| C4-10 | `PLANNER_BRIEF_2026-09-13.md`:1146-1146 | True | `444490a2a45d` | Ruling on its one run (strict FAIL on tie-order diffs only). |
| C4-11 | `p515_s53_w142_resettle_ext_v6_campaign.py`:134-134 | True | `5aa99a88fe1b` | The latest campaign launcher (W142 extension v6) fixes concurrency 1 ... |
| C4-12 | `p515_s53_w142_resettle_ext_v6_campaign.py`:572-572 | True | `cccaaa7a93a0` | ... and checks it. |
| C4-13 | `p515_s51_f2_phase_b.py`:855-855 | True | `ae6dfc595c97` | The 2x search launcher fixes concurrency 5 (Addendum 29 memory rule, pre-tight-tail). |
| C4-14 | `p515_s53_srp1_bitwise_gate.py`:75-78 | True | `0508ea930e69` | The serial two-cycle bitwise gate does not cover the parallel DSO builder. |
| C4-15 | `p515_g_g1_g4_admm_gates.py`:128-128 | True | `5dcb99b41024` | A shared repo directory every child HASHES before and after its run (read; a write by any child would be reported as an error by every concurrent sibling). |

## Files cited (sha256 at HEAD)

- `shared_energy_storage_data.py` `9acd095fb5e50e9b64c253600ae05ebb8e6463fd7a999d3a90f96ee82f037796` (2889 lines; at HEAD True)
- `p515_s44_campaign_harness.py` `9c6bda1969eca917c7875c1e4a84ab004a3b55edf52ad87179e12c816bbb7f21` (5911 lines; at HEAD True)
- `p515_s47_phase_b_record.py` `97f573e52fb574836a72dce8c61b3d5c6b8919aa509e0ffbc04a1a6473ed6c62` (1575 lines; at HEAD True)
- `p515_s51_f2_phase_b.py` `87ab3b2d23729d1fa10fe8c4bae0cec7d8daf0644f1cb8e6c314b441dcd13b29` (1270 lines; at HEAD True)
- `p515_s46_ageing_campaign.py` `3104ea46f7a181558fdc19b437a7232df0d8a97d3972f12266a5b8fa4752611f` (823 lines; at HEAD True)
- `shared_resources_planning.py` `0610d745a41c86c71dae9e776f0dc8e38c7099f73128a3d2c2e8f861ab9fc9c0` (13484 lines; at HEAD True)
- `model_construction_helpers.py` `a20a224f11008de8c0206484583b00253677267d616e1051bb1583791ed47f64` (2511 lines; at HEAD True)
- `admm_persistent_workers.py` `39dcbd267bb224e65197ca427952b08d45d3bbe98785b73fd55148c0979e1125` (969 lines; at HEAD True)
- `p515_s47_baseline_campaign.py` `abaf503d410b94201d6bfbc2f4d2e63927f86f3294883855fc020ff1274db4e2` (1021 lines; at HEAD True)
- `network.py` `18acaa84af7a043898a7b831223b4eab1d91f9bcf2c199f49beecfeb3bf91fb4` (2473 lines; at HEAD True)
- `definitions.py` `7e5719af129df111db1b403feeffe0b566d0b7f2e72915829e16bf143a9321af` (98 lines; at HEAD True)
- `uncoordinated_benchmark.py` `45d1c09db9b687040d57c8b01c1d092f3b53aa120f0e3229d998c435d261e31a` (1730 lines; at HEAD True)
- `p515_g_g1_g4_admm_gates.py` `6f4631e10a2f8a7786390c0f96e45332706779ae9cdf6a87ddd716d9ad9ad346` (7075 lines; at HEAD True)
- `admm_parameters.py` `40d0e972dcadfb4de099a0203682ba2e93badc62d9ae989f87be40b1ce3b75fa` (708 lines; at HEAD True)
- `p59_rho.py` `12d66b48fc473dd809638ea56ad470c5e9dfcaa5dc08be37f8af0bbfb17c159b` (248 lines; at HEAD True)
- `network_parameters.py` `086c7e411f8a3aeb983c4b049c14f231750d119dc126f6c92ef8df77d0bacc94` (190 lines; at HEAD True)
- `p56a_oracle.py` `699d571a2d69b14fd89024c535fa8b3ffa0625aa647f1714060dcc9cf6fa2f11` (1246 lines; at HEAD True)
- `p515_s44_gate.py` `38408b3080b78d16c583eaf711899f63acb4fcd31f0a009fa4d6d795d481b803` (309 lines; at HEAD True)
- `PLANNER_BRIEF_2026-09-13.md` `417f94f526f007dc1e0cd6988bd5fb43c75c2a5d624c24b723b2b0c1db68fc4e` (3763 lines; at HEAD True)
- `p515_s53_w142_resettle_ext_v6_campaign.py` `0a3232069d232b85815ea88b03478b52c54ff3be9b8b4b014f039addd94caa28` (2188 lines; at HEAD True)
- `p515_s53_srp1_bitwise_gate.py` `b63b1fb94ec305a782cb139d6e9c42be8b7a49b2ef78cb0d174f09e3d53f6766` (417 lines; at HEAD True)
