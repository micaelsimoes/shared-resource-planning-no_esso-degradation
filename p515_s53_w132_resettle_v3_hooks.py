"""
P5.15 Addendum 58, Planner task W132 -- the v3 RE-SETTLING CAMPAIGN hooks (claim groups 1, 2 and 4; 38 cells; frozen stage
spec `frozen_s53_resettle_spec_v3_<sha8>.json`, predecessor v2 fc791891).

WHAT THIS MODULE IS. The W118 re-settling hooks (`p515_s53_w118_resettle_hooks`, pinned by fc791891, NOT edited) with the
stop rule VERSION 3 (`settling_criterion_v3`: reading alpha decides, reading gamma report-only, per-cell cap ceiling) and
ONE MORE pass-through wrapper, on `shared_resources_planning._admm_local_solves_succeeded(planning_problem, results)`:
it calls production, returns its value UNCHANGED, and classifies the final accepted attempt of every block of the cycle
-- 12 TSO, 36 DSO and the 3 ESSO solves, in production's own iteration order over `results` -- with
`p515_s44_campaign_harness.ipopt_exit_class(result.solver.message)`. The 51 exits (`ipopt_exit_by_block`) and
`all_optimal_k` are written to each cycle line; the rule reads all_optimal_k for the cycle. The eight W118 wrappers are
W118's own (`R.make_wrappers`, reused, unchanged); the state is W118's with the v3 rule. It is installed ONLY for a
campaign-spec entry whose `settling_resettle` declaration carries this module's schema (the harness dispatches on it; a
W118 declaration keeps the W118 hooks); the declaration enters the entry's eval key.

WHY THE LOCAL-SOLVE CHECK. It is production's one per-cycle call that receives the ESSO results beside the network
results (W131: the ESSO exit is in no committed record). Production calls it once per cycle, after the ESSO solve and
the Boyd residuals, before the AA step and the recourse (asserted from source before any solve), and once at the
initialisation (round 0; recorded in the summary, not a cycle). The results it sees are the final attempt of each block:
network `_run_smopf` and ESSO `_optimize` return the recovery / tier-2 result when they ran (asserted from source).

THE RUN (per cell; everything else exactly as W118): a gated cell replays bitwise against its original record through
its first residual pass k0 (abort on the first difference); the overlap k0+1..N_old is report-only; the holds (AA off,
tight tail on, rho frozen) act after the run's first residual pass under the VERSION-2 definition; the settling rule v3
is the only exit before the cap. CAP: gated N_old + 100 (the spec cap; the per-cell ceiling is the spec's); ungated
min(k0_run + 109, ceiling 300) (dynamic, keyed on the v2 first pass).

Zero solves: nothing here solves or builds a model. Stdlib only at import (the harness parent imports it for keys).
"""
import copy
import inspect
import json
import os
import time
from contextlib import contextmanager

import gate_result_io as GRIO
import settling_criterion_v3 as SC3
import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R

SCHEMA = 'p515_s53_w132_settling_resettle_v3'
DECLARATION_SCHEMA = SCHEMA         # the declaration's 'schema' key: the harness dispatches on it (W118 has none)
OPTION_NAME = 'settling_resettle'
LABEL = ('SRP1 RE-SETTLING RUN v3 (W132, frozen_s53_resettle_spec_v3) -- current production configuration (C2, tight '
         'tail); a gated cell replayed bitwise against its original record through its first residual pass k0 (abort '
         'on divergence), an ungated cell recorded in full; the certifying regime held after the run\'s first residual '
         'pass (AA off, tight tail on, rho frozen); settling rule v3 (reading alpha: a non-Optimal accepted solve is a '
         'lapse; reading gamma report-only) until it certifies or the cap; W105 captures, t_sum, and the IPOPT exit of '
         'every block of every cycle')
P_MAX = R.P_MAX             # 30 (instance-measured; W102 x0 P_hat 29, W103 unit P_hat 30)
L_MONO = R.L_MONO           # 60
CAP_AFTER_N_OLD = R.CAP_AFTER_N_OLD     # 100
CAP_AFTER_K0 = SC3.CAP_AFTER_K0         # 109
UNGATED_CAP_CEILING = 300   # production's case-file num_max_iters (data/SRP1/SRP1_params.json); the G cells' ceiling
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = 12, 36, 3
OPTIMAL = SC3.OPTIMAL_CLASS
WRAPPED = R.WRAPPED + ('_admm_local_solves_succeeded',)
CYCLE_FILE = R.CYCLE_FILE
BLOCKS_FILE = R.BLOCKS_FILE
CREEP_FILE = R.CREEP_FILE
ESS_SCHEDULE_FILE = R.ESS_SCHEDULE_FILE
DECISION_FILE = R.DECISION_FILE
SUMMARY_KEY = R.SUMMARY_KEY
FORBIDDEN_DECLARATION_KEYS = R.FORBIDDEN_DECLARATION_KEYS
CERTIFICATION_DISABLED_THRESHOLD = R.CERTIFICATION_DISABLED_THRESHOLD
SETTLING_END_THRESHOLD = R.SETTLING_END_THRESHOLD
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the 38 cells (original records committed, in their campaign manifests; verified by the checks, section O) ---------
# Priority order (Addendum 58; TASKS.md Planner ruling (iv)): group 1 (B, D) -> group 2 (H, I, J) -> group 4 (C, G, L).
# Group 3 (ageing E) is NOT in this spec (held for the author's soh_min ruling); 7aa017f0 / bd504ecf are covered by the
# settled references (not re-run). cap_ceiling: 300 except the two ruled exceptions (Planner ruling (iii)): 2ab0ce2d 437
# and b2251bc5 320 (= their N_old + 100). k0 = the original record's first residual pass (N_old - 9 on every cell but
# j_5f3cccb4: first pass 159, a Boyd lapse at 160, passes 161..170 -- recorded in original_lapses_after_k0).
_RES = os.path.join('data', 'SRP1', 'Results')
_S47A = os.path.join(_RES, 'P515S47', 'campaign_s47_a1a_baseline')
_S47PB = os.path.join(_RES, 'P515S47', 'campaign_s47_phase_b')
_S49 = os.path.join(_RES, 'P515S49', 'campaign_s49_flex_ladder')
_S50 = os.path.join(_RES, 'P515S50', 'campaign_s50_marginal')
_S51L = os.path.join(_RES, 'P515S51', 'campaign_s51_f2_ladder')
_S51PB = os.path.join(_RES, 'P515S51', 'campaign_s51_f2_phase_b')
_S45B = os.path.join(_RES, 'P515S45', 'campaign_s45_a1b')
_S53F2 = os.path.join(_RES, 'P515S53', 'campaign_s53_f2_certificate_r1')
CELLS = {
    'b_2a0ba8b2': {'item': 'B', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '2a0ba8b2f3d3b99eded7abf9313158614ba8d55ace8a81cd77378665bb831e5e',
        'orig_eval_dir': '2a0ba8b2f3d3b99e_n5_4h_e1', 'orig_label': 'n5_4h_e1',
        'N_old': 113, 'k0': 104, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '781f1ff9afdc832492af5287408d09d253a66673252625304adcb389c6567243'},
    'b_0dd237f0': {'item': 'B', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '0dd237f0d286db795bd0416beb3e50cb8f1344dd566a9b636b9e4943a0c6a546',
        'orig_eval_dir': '0dd237f0d286db79_n9_4h_e1', 'orig_label': 'n9_4h_e1',
        'N_old': 112, 'k0': 103, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'fe078825c63abac299c514e6c7b27cfd003e60fac57c854f6e7ea3faeb843cf8'},
    'b_4649234b': {'item': 'B', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '4649234b0f6624b93cc4acf687b16234fd4f16875a60587e7c1af6eb4dcb7da7',
        'orig_eval_dir': '4649234b0f6624b9_n9_4h_e3', 'orig_label': 'n9_4h_e3',
        'N_old': 93, 'k0': 84, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '6a5e7d0e00d2cffdd6e11dfaeaf9b2a93b46cdb9306ce43cbd608ffeee99541c'},
    'd_c52e1670': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': 'c52e167077ed3bbd0d5a0e0b12f7b5c6fcedbd59d549b2fc20d0e3cdfd8f444e',
        'orig_eval_dir': 'c52e167077ed3bbd_n7_4h_e2', 'orig_label': 'n7_4h_e2',
        'N_old': 98, 'k0': 89, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '1b4ae0f892db74c4d1606f79836a8267ebf0f7709ab0c1f1a6fad6984d02c8b1'},
    'd_4a82a64a': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '4a82a64a21aa67cc1223b7449d53973c17a90dbf0211d516e7f26a0a031d9be5',
        'orig_eval_dir': '4a82a64a21aa67cc_n7_4h_e3', 'orig_label': 'n7_4h_e3',
        'N_old': 97, 'k0': 88, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'd3283f7eeff347730a36de6dba1353c4427699861942a70667233889d6d1e7e2'},
    'd_3632b0ae': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '3632b0ae8de45fc64a5e3ea29e51becdde1c5e04d77899a4370e2a545b21529f',
        'orig_eval_dir': '3632b0ae8de45fc6_n7_4h_e4', 'orig_label': 'n7_4h_e4',
        'N_old': 95, 'k0': 86, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '07d0ad2f5b968c2e56f811be8d2dc1b1e7db5ea66ad5633fdfaef388467dbe7f'},
    'd_36686489': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '366864896179e01df0f7290fb9f8b1a4981a8b486b48d4bebcb275326d981d48',
        'orig_eval_dir': '366864896179e01d_n7_4h_e5', 'orig_label': 'n7_4h_e5',
        'N_old': 91, 'k0': 82, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '1f152fb448c32307ea4be87b5398d651d9052e2176a298ab45de07fdde1777d3'},
    'd_d3709599': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': 'd3709599c7c4280727bda7406faa149fde1d045b337d6c06d60458fda92136ab',
        'orig_eval_dir': 'd3709599c7c42807_n7_2h_e1', 'orig_label': 'n7_2h_e1',
        'N_old': 113, 'k0': 104, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '8be79c5f0696ff10d261ca70e91929004501410670f74394e28027b4edf2c479'},
    'd_a12d95a2': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': 'a12d95a2a952693c0ff1274b3a208a80453e7a170b30019a012ab510590ee707',
        'orig_eval_dir': 'a12d95a2a952693c_n7_2h_e2', 'orig_label': 'n7_2h_e2',
        'N_old': 103, 'k0': 94, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'e59d376d231e8c2277b85d7666ea5b1df7ab69806b5807bd1db3bce63618fc8f'},
    'd_f759dd48': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': 'f759dd4825d3f3af4360d012f6e6f1340215150334db38fd58c3bbaf96512e79',
        'orig_eval_dir': 'f759dd4825d3f3af_n7_2h_e3', 'orig_label': 'n7_2h_e3',
        'N_old': 98, 'k0': 89, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '91811287962b2fc44d693f0024d4417e4f2e7ecfcecfd67f12887bcc89d17c75'},
    'd_c7fee8be': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': 'c7fee8beeb210466dc1150cbfea6addbccb75ceda97508538f0784149bf0aa7a',
        'orig_eval_dir': 'c7fee8beeb210466_n7_2h_e4', 'orig_label': 'n7_2h_e4',
        'N_old': 92, 'k0': 83, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '670ea0535b78c0411c60350eaf77afc4a5b1aa7909f8538c5a526149e7412474'},
    'd_9246ed01': {'item': 'D', 'gated': True, 'orig_root': _S47A, 'orig_spec': 'campaign_spec_s47_a1a_baseline_add4ebd8.json',
        'orig_campaign_id': 's47_a1a_baseline', 'orig_eval_key': '9246ed01f072b31fb93ab0d9378707142c74810990920515dbf405c2d03939a5',
        'orig_eval_dir': '9246ed01f072b31f_n7_2h_e5', 'orig_label': 'n7_2h_e5',
        'N_old': 92, 'k0': 83, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'f1f9f6ae2954fb4964ca3f138a836c770dff0f639312b441fd4522fa3401113f'},
    'h_aa8a76d7': {'item': 'H', 'gated': True, 'orig_root': _S49, 'orig_spec': 'campaign_spec_s49_flex_ladder_2203f6c1.json',
        'orig_campaign_id': 's49_flex_ladder', 'orig_eval_key': 'aa8a76d71ae49f134831ae5bf40579285249b228f21ec66556eb43b04f4dd1a1',
        'orig_eval_dir': 'aa8a76d71ae49f13_x0_m1p5', 'orig_label': 'x0_m1p5',
        'N_old': 102, 'k0': 93, 'original_lapses_after_k0': [], 'flex_price_multiplier': 1.5,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '754d02868676afab046cf7dfd557088a360eb2655b01b0aedea1af9e71af976d'},
    'h_f9eae48f': {'item': 'H', 'gated': True, 'orig_root': _S49, 'orig_spec': 'campaign_spec_s49_flex_ladder_2203f6c1.json',
        'orig_campaign_id': 's49_flex_ladder', 'orig_eval_key': 'f9eae48ff6133f6ca988cb14f511e1578fecd8f8c264675f705d3fdda7defbda',
        'orig_eval_dir': 'f9eae48ff6133f6c_n7_4h_e1_m1p5', 'orig_label': 'n7_4h_e1_m1p5',
        'N_old': 94, 'k0': 85, 'original_lapses_after_k0': [], 'flex_price_multiplier': 1.5,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'd3be5d8918490c76fea4e5855393c0d7289c46e3c8a830c5bd8222c463799e72'},
    'h_50dea31c': {'item': 'H', 'gated': True, 'orig_root': _S49, 'orig_spec': 'campaign_spec_s49_flex_ladder_2203f6c1.json',
        'orig_campaign_id': 's49_flex_ladder', 'orig_eval_key': '50dea31c5780df578cb8d6edde46ba68640d8237835b8e4f5481cce1cd2a22cf',
        'orig_eval_dir': '50dea31c5780df57_x0_m2', 'orig_label': 'x0_m2',
        'N_old': 134, 'k0': 125, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'e214ff3180b54d5f109f340ba3fd22dae2435efb77e4cf6754926c18a8c06eae'},
    'h_74eda68d': {'item': 'H', 'gated': True, 'orig_root': _S49, 'orig_spec': 'campaign_spec_s49_flex_ladder_2203f6c1.json',
        'orig_campaign_id': 's49_flex_ladder', 'orig_eval_key': '74eda68d74d990b2126dd10f59280b4ea63a450b7da5e4f7355c9ad97f8f7724',
        'orig_eval_dir': '74eda68d74d990b2_n7_4h_e1_m2', 'orig_label': 'n7_4h_e1_m2',
        'N_old': 132, 'k0': 123, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '58fa198d2137aef09743031e475e4abc88595bc36735a94bb7cded81cdf4bd1b'},
    'i_5a6a88b4': {'item': 'I', 'gated': True, 'orig_root': _S50, 'orig_spec': 'campaign_spec_s50_marginal_fa4b30ee.json',
        'orig_campaign_id': 's50_marginal', 'orig_eval_key': '5a6a88b46bd3a8fe4b15e4f158f5349ceb22a84259d5ef2e934c9a0b36127d97',
        'orig_eval_dir': '5a6a88b46bd3a8fe_n7_4h_e2_m2', 'orig_label': 'n7_4h_e2_m2',
        'N_old': 144, 'k0': 135, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'f5eda5e1c0aaa10cf160a1cbe2a6994f3a417eb6c41dce8e71d849d491b0baba'},
    'j_f3aa335e': {'item': 'J', 'gated': True, 'orig_root': _S51L, 'orig_spec': 'campaign_spec_s51_f2_ladder_c4455767.json',
        'orig_campaign_id': 's51_f2_ladder', 'orig_eval_key': 'f3aa335e6c1eda6950712aec00aff122d4578760f0dd02ff64996b95998eaace',
        'orig_eval_dir': 'f3aa335e6c1eda69_n7_4h_e3_m2', 'orig_label': 'n7_4h_e3_m2',
        'N_old': 160, 'k0': 151, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '0a6fa8525c385d7cb6314ebf87ed31d2a63ebb9bde0712b84618c371d4ceddaf'},
    'j_a11d7966': {'item': 'J', 'gated': True, 'orig_root': _S51L, 'orig_spec': 'campaign_spec_s51_f2_ladder_c4455767.json',
        'orig_campaign_id': 's51_f2_ladder', 'orig_eval_key': 'a11d79663bf5a95496b784a56251c738bdc5af261be2b3346f777713ad2fcdc9',
        'orig_eval_dir': 'a11d79663bf5a954_n7_4h_e4_m2', 'orig_label': 'n7_4h_e4_m2',
        'N_old': 136, 'k0': 127, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '2aea9ca17fa91442affc74fea47caebc42945ffeafabf968639d1f5cb2a042d9'},
    'j_5f3cccb4': {'item': 'J', 'gated': True, 'orig_root': _S51L, 'orig_spec': 'campaign_spec_s51_f2_ladder_c4455767.json',
        'orig_campaign_id': 's51_f2_ladder', 'orig_eval_key': '5f3cccb4cad36ee541b292d66f3899e9aebd7edcb512ba42b5b4744b3e80d336',
        'orig_eval_dir': '5f3cccb4cad36ee5_n7_4h_e5_m2', 'orig_label': 'n7_4h_e5_m2',
        'N_old': 170, 'k0': 159, 'original_lapses_after_k0': [160], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'fece552483a82909b9a85c4bc75982ed0af3864628a3b350e923f59e30b6ab3d'},
    'c_156ce2d1': {'item': 'C', 'gated': True, 'orig_root': _S47PB, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b', 'orig_eval_key': '156ce2d1d53d36f4ea683c42879dce9be4a3c802030083ab2ff94178ca4fc3ad',
        'orig_eval_dir': '156ce2d1d53d36f4_y2025__n5_p0_25_e0_5__n9_p0_25_e0_5', 'orig_label': 'y2025__n5_p0.25_e0.5__n9_p0.25_e0.5',
        'N_old': 114, 'k0': 105, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '8b4ac54c9f659337e9bce7f5ff7363d04edd773ebbc19cb57b7d841036b477c8'},
    'c_6597a79d': {'item': 'C', 'gated': True, 'orig_root': _S47PB, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b', 'orig_eval_key': '6597a79dbe8ed61fb45aed6e3eebfe27c4a5035976d5f0a9f20a4601ac88e12b',
        'orig_eval_dir': '6597a79dbe8ed61f_y2030__n5_p0_25_e0_5__n7_p0_25_e0_5', 'orig_label': 'y2030__n5_p0.25_e0.5__n7_p0.25_e0.5',
        'N_old': 140, 'k0': 131, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '021ab950e13ffedd689ff88eb293ec6c6dddc653e85f5ab661e87cf1d0b4b896'},
    'g_37b5c499': {'item': 'G', 'gated': False, 'orig_root': _S45B, 'orig_spec': 'campaign_spec_s45_a1b_bd3040a9.json',
        'orig_campaign_id': 's45_a1b', 'orig_eval_key': '37b5c4999f65749fc0116c342be96e06268e634b5c7852099a94e73a36822d47',
        'orig_eval_dir': '37b5c4999f65749f_n7_2h_e1_y2030', 'orig_label': 'n7_2h_e1_y2030',
        'N_old': 131, 'k0': 122, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '21dcb7723f703b38afc3edbce2b9c7ca8bb43213fc14bc679df91097c59f426a'},
    'g_47dce43c': {'item': 'G', 'gated': False, 'orig_root': _S45B, 'orig_spec': 'campaign_spec_s45_a1b_bd3040a9.json',
        'orig_campaign_id': 's45_a1b', 'orig_eval_key': '47dce43c9f4ab0aa87b4e3b2facbe9fe8a61f73b03c8fe157ece16a48d1c540c',
        'orig_eval_dir': '47dce43c9f4ab0aa_n7_2h_e1_y2035', 'orig_label': 'n7_2h_e1_y2035',
        'N_old': 139, 'k0': 130, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '2420d9daa8640402f0d8e1a7374609af910fc4d7c6cbc87aa975a127ee8a72d8'},
    'g_48749148': {'item': 'G', 'gated': False, 'orig_root': _S45B, 'orig_spec': 'campaign_spec_s45_a1b_bd3040a9.json',
        'orig_campaign_id': 's45_a1b', 'orig_eval_key': '48749148d0bebd0da9b14085df04ef1bc6a3093174de15a559b7796f0b7cc64e',
        'orig_eval_dir': '48749148d0bebd0d_n7_4h_e2_y2030', 'orig_label': 'n7_4h_e2_y2030',
        'N_old': 111, 'k0': 102, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'e7b17bccff22bac87ad405c57129484ec9612cffd7fe136a65b1a105e7b0e36c'},
    'g_9abf31d4': {'item': 'G', 'gated': False, 'orig_root': _S45B, 'orig_spec': 'campaign_spec_s45_a1b_bd3040a9.json',
        'orig_campaign_id': 's45_a1b', 'orig_eval_key': '9abf31d494b664b89aaff4e2aa75765327808fab008b7629c36dd5f76eae8e9b',
        'orig_eval_dir': '9abf31d494b664b8_n7_4h_e2_y2035', 'orig_label': 'n7_4h_e2_y2035',
        'N_old': 107, 'k0': 98, 'original_lapses_after_k0': [], 'flex_price_multiplier': None,
        'cap_ceiling': 300, 'per_cycle_record_sha256': 'a2b66fa3c27d1429411d029878a6a26869abed54b3bde0b71185454a643bb8fe'},
    'l_7b199ef9': {'item': 'L', 'gated': True, 'orig_root': _S53F2, 'orig_spec': 'campaign_spec_s53_f2_certificate_r1_803571c0.json',
        'orig_campaign_id': 's53_f2_certificate_r1', 'orig_eval_key': '7b199ef9da3880d30dfa3bfb9cfef1ef0c16efc7c16b94710a227d7e9977262a',
        'orig_eval_dir': '7b199ef9da3880d3_y2030__n7_p1_e4_m2', 'orig_label': 'y2030__n7_p1_e4_m2',
        'N_old': 175, 'k0': 166, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '41655883cbbedf2f97c94c4d7cfd7956af8ef035e4149852161c123b84a3c536'},
    'l_e1da0984': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': 'e1da0984365060a9eb8bb857e0fa176ff7c9a84f62b556d235533ee6a66454d0',
        'orig_eval_dir': 'e1da0984365060a9_y2030__n7_p1_e3_5_m2', 'orig_label': 'y2030__n7_p1_e3.5_m2',
        'N_old': 165, 'k0': 156, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '5b1bcdfc8133581622d9dc11b69e78c65220ccf9bbe96be7685a61be0a2700a4'},
    'l_0ee93aca': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '0ee93aca71ed71d26f7735aab6d274ea210f1a607212931f1e24e6959bec2706',
        'orig_eval_dir': '0ee93aca71ed71d2_y2030__n5_p0_25_e0_5__n7_p1_e3_m2', 'orig_label': 'y2030__n5_p0.25_e0.5__n7_p1_e3_m2',
        'N_old': 184, 'k0': 175, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '3f377758790791beb68bb5e9db00b6a167d4ec5234c28005adf72e43978b8063'},
    'l_df1a5525': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': 'df1a552507b92c5299e9170048d3a4e106e6b190893d535eae75bdc3255b99a5',
        'orig_eval_dir': 'df1a552507b92c52_y2030__n5_p0_25_e0_5__n7_p0_75_e2_5_m2', 'orig_label': 'y2030__n5_p0.25_e0.5__n7_p0.75_e2.5_m2',
        'N_old': 177, 'k0': 168, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '616d5368917e1b839dcdb3456965ff2bf06b919ebc8ace82d6f737ce98b4867a'},
    'l_2ab0ce2d': {'item': 'L', 'gated': True, 'orig_root': _S53F2, 'orig_spec': 'campaign_spec_s53_f2_certificate_r1_803571c0.json',
        'orig_campaign_id': 's53_f2_certificate_r1', 'orig_eval_key': '2ab0ce2dbd07d42c4fcecce5529240a9525aba5948dc5c00c4c6603bfdf22b08',
        'orig_eval_dir': '2ab0ce2dbd07d42c_y2030__n5_p0_25_e0_5__n7_p1_25_e3_m2', 'orig_label': 'y2030__n5_p0.25_e0.5__n7_p1.25_e3_m2',
        'N_old': 337, 'k0': 328, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 437, 'per_cycle_record_sha256': 'e0913d32493321e033aa4eca7a747c8383a07eacc7f423fce0fd1660484de18a'},
    'l_8e4c220e': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '8e4c220e4d03411bfb3caf8e5c50fc752addcf737b582d73faefcd86fada87c3',
        'orig_eval_dir': '8e4c220e4d03411b_y2030__n7_p0_75_e3_m2', 'orig_label': 'y2030__n7_p0.75_e3_m2',
        'N_old': 180, 'k0': 171, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '7088013943a671122b855ae1b42b186b11cfe783b83a9658f4c4e4a17bcc1f01'},
    'l_7db09f6c': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '7db09f6c11a99cd173cd4ed2a0e69b643ef689f974387d669a88936472f2b3cd',
        'orig_eval_dir': '7db09f6c11a99cd1_y2030__n7_p0_75_e3__n9_p0_25_e0_5_m2', 'orig_label': 'y2030__n7_p0.75_e3__n9_p0.25_e0.5_m2',
        'N_old': 192, 'k0': 183, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '544e9c1a8b5830e8754399bf654d132d5db1cb1d41ed61182467a89ed2861a78'},
    'l_76c78064': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '76c7806434358b47dea377831341c3c9d3a25fe9a24c1ebcd8ae48010c86428b',
        'orig_eval_dir': '76c7806434358b47_y2030__n7_p1_e3_m2', 'orig_label': 'y2030__n7_p1_e3_m2',
        'N_old': 162, 'k0': 153, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '446e294261d3f677c2c18c813e22387d9d128648d61aa991ba190a85ed78c144'},
    'l_45aa25a6': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '45aa25a6d2a5945507c58ff8a3d3649f3d771a9d597e9f38dbd425c7e677d2c6',
        'orig_eval_dir': '45aa25a6d2a59455_y2030__n7_p1_e3__n9_p0_25_e0_5_m2', 'orig_label': 'y2030__n7_p1_e3__n9_p0.25_e0.5_m2',
        'N_old': 191, 'k0': 182, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '288fde71bcecdf6d6ef85bf8059343fe136220cc0a3133cdf7d5e6c056af6ab6'},
    'l_7c455554': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '7c45555407d8f3e8b08ef3ad85bebf3509918fee21d044e3fcc3a6690eb4bbec',
        'orig_eval_dir': '7c45555407d8f3e8_y2030__n7_p1_e3_5__n9_p0_25_e0_5_m2', 'orig_label': 'y2030__n7_p1_e3.5__n9_p0.25_e0.5_m2',
        'N_old': 182, 'k0': 173, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '5a70086959933900b3c02edbf9490231245de1abe966ddedd20fb9f5381d6204'},
    'l_b2251bc5': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': 'b2251bc53ca992aa50579a57ea15f54d939fd47ccbebaf078d8e7c36b300ea4e',
        'orig_eval_dir': 'b2251bc53ca992aa_y2030__n5_p0_25_e0_5__n7_p0_75_e3__n9_p0_25_e0_5_m2', 'orig_label': 'y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5_m2',
        'N_old': 220, 'k0': 211, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 320, 'per_cycle_record_sha256': 'fd88c21bb5ea7660d3ff829516140d8df97e99924ab4e383f439d5b593f836af'},
    'l_195156fa': {'item': 'L', 'gated': True, 'orig_root': _S51PB, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b', 'orig_eval_key': '195156fa819625db81fddc4db3969613914bbcfa11ac4cc11d73d9ee6c8f4bb9',
        'orig_eval_dir': '195156fa819625db_y2030__n5_p0_25_e0_5__n7_p0_75_e3_m2', 'orig_label': 'y2030__n5_p0.25_e0.5__n7_p0.75_e3_m2',
        'N_old': 194, 'k0': 185, 'original_lapses_after_k0': [], 'flex_price_multiplier': 2.0,
        'cap_ceiling': 300, 'per_cycle_record_sha256': '06e6fd0088e2fda5b13c012850d4c1db992658979cd5cf6ea764b1636d71d7d3'},
}
CELL_ORDER = tuple(CELLS)
GROUP_OF_ITEM = {'B': 1, 'D': 1, 'H': 2, 'I': 2, 'J': 2, 'C': 4, 'G': 4, 'L': 4}
GATED_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['gated'])
UNGATED_CELLS = tuple(c for c in CELL_ORDER if not CELLS[c]['gated'])
# The Advisor's design review (TASKS.md, Addendum 58 section, commit 82fa1b61): dead-zone candidates (m = 2, ESS P >= 1.25
# MVA) and the borderline cell.
DEAD_ZONE_CANDIDATES = ('j_5f3cccb4', 'l_0ee93aca', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5')
DEAD_ZONE_BORDERLINE = ('l_2ab0ce2d',)


def original_eval_dir(cell):
    return os.path.join(CELLS[cell]['orig_root'], 'evals', CELLS[cell]['orig_eval_dir'])


def reference_path(cell):
    return os.path.join(original_eval_dir(cell), 'per_cycle_record.jsonl')


def settling_rule_declaration():
    return {'module': 'settling_criterion_v3', 'class': 'settling_criterion_v3.SettlingRuleV3', 'version': SC3.VERSION,
            'reading': 'alpha', 'report_only_readings': ['gamma'], 'retry_tier': None,
            'boyd_k_v3': 'boyd_k AND all_optimal_k', 'optimal_class': OPTIMAL,
            'n_and_dynamic_cap_keyed_on': 'the first residual pass under the version-2 definition',
            'tau': SC3.TAU, 'eps0': SC3.EPS0, 'k_excl': SC3.K_EXCL, 'w_min': SC3.W_MIN, 'w_factor': SC3.W_FACTOR,
            'p_max': P_MAX, 'l_mono': L_MONO, 'gap_bound': SC3.GAP_BOUND, 'drift_window': SC3.DRIFT_WINDOW}


def cap_rule(cell):
    c = CELLS[cell]
    if c['gated']:
        cap = c['N_old'] + CAP_AFTER_N_OLD
        return {'kind': 'fixed', 'cap': cap, 'ceiling': c['cap_ceiling'], 'formula': 'N_old + 100'}
    return {'kind': 'dynamic', 'after_first_k0': CAP_AFTER_K0, 'ceiling': c['cap_ceiling'],
            'formula': 'min(k0_run + 109, ceiling) (k0_run: the first residual pass under the version-2 definition)'}


def spec_cap(cell):
    """The harness cap (production's num_max_iters) for the cell: gated N_old + 100; ungated its ceiling."""
    r = cap_rule(cell)
    return r['cap'] if r['kind'] == 'fixed' else r['ceiling']


def declaration_for(cell):
    """The one valid declaration of a cell."""
    c = CELLS[cell]
    gated = c['gated']
    return {
        'schema': DECLARATION_SCHEMA, 'label': LABEL, 'cell': cell, 'item': c['item'],
        'claim_group': GROUP_OF_ITEM[c['item']],
        'gate': 'bitwise_through_first_residual_pass' if gated else 'none_first_c2_evaluation',
        'first_residual_pass_expected': c['k0'] if gated else None,
        'N_old': c['N_old'] if gated else None,
        'original_lapses_after_k0': list(c['original_lapses_after_k0']) if gated else None,
        'holds_after': ('the run first residual pass k0_run (version-2 definition): AA off, tight tail on, rho frozen '
                        'for every later cycle'),
        'cap_rule': cap_rule(cell),
        'settling_rule': settling_rule_declaration(),
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True, 't_sum': True,
                     'ipopt_exit_by_block': True},
        'expected_blocks': {'tso': N_TSO_BLOCKS, 'dso': N_DSO_BLOCKS, 'esso': N_ESSO},
        'replay_reference': ({'per_cycle_record': reference_path(cell), 'sha256': c['per_cycle_record_sha256'],
                              'n_cycles': c['N_old'], 'gated_through_cycle': c['k0'],
                              'original_eval_key': c['orig_eval_key']} if gated else None),
        'abort_on_replay_divergence': bool(gated),
    }


def is_v3_declaration(value):
    return isinstance(value, dict) and value.get('schema') == DECLARATION_SCHEMA


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop`
    key is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (v3) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (v3) must NOT carry {forbidden} (the settling rule is the only stop before the '
                         f'cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (v3): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (v3).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (v3) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (v3) differs from the W132 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


def load_replay_reference(decl):
    """{cycle: row} of the gated cell's original per_cycle_record.jsonl (cycles 1..N_old); refuses unless it hashes to
    the declaration, holds exactly 1..N_old, carries every gated field, its first residual pass is the declared k0, the
    cycles k0..N_old without a residual pass are exactly the declared original lapses, and N_old passes. None for an
    ungated cell."""
    ref = decl.get('replay_reference')
    if ref is None:
        return None
    path = os.path.join(REPO, ref['per_cycle_record'])
    got = C101._sha256(path)
    if got != ref['sha256']:
        raise RuntimeError(f'replay reference {ref["per_cycle_record"]} sha256 {got} != declared {ref["sha256"]}')
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[int(r['cycle'])] = r
    n = ref['n_cycles']
    if sorted(out) != list(range(1, n + 1)):
        raise RuntimeError(f'replay reference cycles != 1..{n}')
    missing = [f for f in R.REPLAY_GATED_FIELDS + R.REPLAY_POST_RUN_ONLY_FIELDS if f not in out[1]]
    if missing:
        raise RuntimeError(f'replay reference rows lack gated fields {missing}')
    passes = [c for c in sorted(out) if out[c]['boyd_all_pass'] and out[c]['local_solves_ok']]
    k0 = ref['gated_through_cycle']
    lapses = [c for c in range(k0, n + 1) if c not in passes]
    if not passes or passes[0] != k0 or lapses != list(decl['original_lapses_after_k0']) or passes[-1] != n:
        raise RuntimeError(f'replay reference: first residual pass {passes[:1]} (declared {k0}), lapses after k0 '
                           f'{lapses} (declared {decl["original_lapses_after_k0"]}), last pass {passes[-1:]} (N_old {n})')
    return out


# ======================================================================================================================
#  the IPOPT exit of every block (the ninth wrapper's capture; pure reads)
# ======================================================================================================================
def block_keys(planning_problem):
    """The 51 block keys in production's `_admm_local_solves_succeeded` iteration order."""
    keys = []
    for year in planning_problem.years:
        for day in planning_problem.days:
            keys.append(('TSO', None, year, day))
            for node_id in planning_problem.active_distribution_network_nodes:
                keys.append(('DSO', node_id, year, day))
    for node_id in planning_problem.active_distribution_network_nodes:
        keys.append(('ESSO', node_id, None, None))
    return keys


def _key_text(key):
    kind, node_id, year, day = key
    if kind == 'TSO':
        return f'TSO|{year}|{day}'
    if kind == 'DSO':
        return f'DSO|{node_id}|{year}|{day}'
    return f'ESSO|{node_id}'


def _result_of(results, key):
    kind, node_id, year, day = key
    if kind == 'TSO':
        return results['tso'][year][day]
    if kind == 'DSO':
        return results['dso'][node_id][year][day]
    return results['esso'][node_id]


def exit_by_block(planning_problem, results, exit_class):
    """{block: {'class', 'message', 'has_result', 'has_solver'}} for the 51 blocks; the message is
    `str(result.solver.message)` exactly as the harness's `_termination_record` reads it; `exit_class` is the harness's
    `ipopt_exit_class`. A missing results key RAISES (a structural capture gap); a None result or one without a solver
    record is a block that is not Optimal (class None)."""
    out = {}
    for key in block_keys(planning_problem):
        result = _result_of(results, key)
        solver = getattr(result, 'solver', None) if result is not None else None
        message = getattr(solver, 'message', None) if solver is not None else None
        message = None if message is None else str(message)
        out[_key_text(key)] = {'class': exit_class(message), 'message': message, 'has_result': result is not None,
                               'has_solver': solver is not None}
    return out


def exit_counts(by_block):
    kinds = [k.split('|')[0] for k in by_block]
    return {'tso': kinds.count('TSO'), 'dso': kinds.count('DSO'), 'esso': kinds.count('ESSO'), 'total': len(kinds)}


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve)
# ======================================================================================================================
EXIT_SOURCE_FACTS = {
    'local_check_iterates_every_block': ('_admm_local_solves_succeeded', (
        "if not _solver_result_succeeded(results['tso'][year][day]):",
        "if not _solver_result_succeeded(results['dso'][node_id][year][day]):",
        "if not _solver_result_succeeded(results['esso'][node_id]):")),
    'esso_update_returns_the_optimize_results': ('update_shared_energy_storages_coordination_model_and_solve', (
        'res = shared_ess_data.optimize(models, from_warm_start=from_warm_start, cycle=cycle)', 'return res')),
}
LOOP_EXIT_FACTS = (
    "results['esso'] = update_shared_energy_storages_coordination_model_and_solve(",
    'boyd_metrics = get_admm_boyd_residual_metrics(',
    'local_solves_ok = _admm_local_solves_succeeded(planning_problem, results)',
    'aa_record = _anderson_acceleration_cycle_step(',
    'recourse_components = _get_operational_recourse_components(planning_problem, operational_models)',
)
FINAL_ATTEMPT_FACTS = {
    'esso_optimize_collects_optimize': ('shared_energy_storage_data', 'SharedEnergyStorageData.optimize', (
        'results[node_id] = _optimize(', 'return results')),
    'esso_final_attempt_is_returned': ('shared_energy_storage_data', '_optimize', (
        'result = recovery_result if recovery_result is not None else primary_result',
        'result = tier2_result if tier2_result is not None else recovery_result', 'return result')),
    'network_final_attempt_is_returned': ('network', '_run_smopf', (
        'result = recovery_result', 'result = tier2_result', 'return result')),
}
EXIT_CLASS_CASES = (('Ipopt 3.14.18\\x3a Optimal Solution Found', 'optimal'),
                    ('Ipopt 3.14.18\\x3a Solved To Acceptable Level.', 'acceptable'),
                    ('Ipopt 3.14.18\\x3a Maximum Number of Iterations Exceeded.', 'other'), (None, None))


def exit_capture_checklist():
    """The source facts the ninth wrapper relies on (it must see every block's final attempt, ESSO included). Returns
    {name: bool}; raises nothing."""
    import shared_resources_planning as srp
    import shared_energy_storage_data as sed
    import network as net
    import p515_s44_campaign_harness as HAR
    out = {}
    for name, (fn, snippets) in EXIT_SOURCE_FACTS.items():
        f = getattr(srp, fn, None)
        src = inspect.getsource(f) if callable(f) else ''
        out[f'exit_source:{name}'] = bool(src) and all(s in src for s in snippets)
    mods = {'shared_energy_storage_data': sed, 'network': net}
    for name, (mod_name, qual, snippets) in FINAL_ATTEMPT_FACTS.items():
        obj = mods[mod_name]
        for part in qual.split('.'):
            obj = getattr(obj, part, None)
        src = inspect.getsource(obj) if obj is not None else ''
        out[f'final_attempt:{name}'] = bool(src) and all(s in src for s in snippets)
    loop_src = inspect.getsource(srp._run_operational_planning)
    loop_at = loop_src.find('for iter in range(1, admm_parameters.num_max_iters + 1):')
    pos = [loop_src.find(s, max(loop_at, 0)) for s in LOOP_EXIT_FACTS]
    out['loop:esso_then_boyd_then_local_check_then_aa_then_recourse'] = (
        0 <= loop_at < pos[0] < pos[1] < pos[2] < pos[3] < pos[4])
    out['loop:local_check_called_twice_init_and_once_per_cycle'] = (
        loop_src.count('_admm_local_solves_succeeded(planning_problem, results)') == 2
        and loop_src.find('_admm_local_solves_succeeded(planning_problem, results)') < loop_at)
    out['signature:_admm_local_solves_succeeded'] = (
        list(inspect.signature(srp._admm_local_solves_succeeded).parameters) == ['planning_problem', 'results'])
    out['harness:ipopt_exit_class_cases'] = all(HAR.ipopt_exit_class(m) == want for m, want in EXIT_CLASS_CASES)
    out['harness:termination_record_reads_result_solver_message'] = (
        "message = getattr(result.solver, 'message', None)" in inspect.getsource(HAR._termination_record))
    # the results are this process's own SolverResults (no worker pool, no parallel execution): the harness child
    # refuses a run with either on, before any solve
    hsrc = inspect.getsource(HAR)
    out['harness:child_refuses_persistent_worker_pool'] = (
        "checks['persistent_workers_off'] = not a.persistent_workers.get('enabled')" in hsrc)
    out['harness:child_refuses_parallel_execution'] = "checks['parallel_execution_off'] = not planning.parallel_execution" in hsrc
    return out


def _certificate_length_writes_in_source():
    return R._certificate_length_writes_in_source()


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child); fail fast: W118's items (a)-(h) restated for v3 (the spec cap is
    the cell's and within its per-cell ceiling; the rule constants are settling_criterion_v3's), the signatures of the
    nine wrapped functions, W105's capture paths, the t_sum source facts, and the exit-capture facts (the local-solve
    check sees every block's final attempt, ESSO included; the harness classifier). Raises on any failure; returns the
    checklist."""
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    import interface_dual_capture as IDC
    decl = validate_settling_resettle(decl)
    cell = decl['cell']
    cap = int(spec['cap'])
    params = {
        '_capture_convergence_depth_tail_baseline': ['planning_problem', 'admm_parameters'],
        '_apply_convergence_depth_tail': ['planning_problem', 'admm_parameters', 'active', 'baseline', 'cycle'],
        'get_admm_boyd_residual_metrics': ['planning_problem', 'tso_model', 'dso_models', 'esso_model',
                                           'consensus_vars', 'dual_vars', 'admm_parameters'],
        '_anderson_acceleration_cycle_step': ['aa_state', 'aa_layout', 'consensus_vars', 'dual_vars', 'w_before',
                                              'rho_channel', 'boyd_metrics', 'iter'],
        '_convergence_depth_tail_next_state': ['cycle_convergence', 'aa_enabled', 'aa_record'],
        '_update_admm_penalties': ['tso_model', 'dso_models', 'esso_model', 'residual_metrics', 'boyd_metrics',
                                   'params', 'iter', 'allow_update', 'freeze_state'],
        '_get_operational_recourse_components': ['planning_problem', 'models'],
        '_get_admm_efc_per_day_max': ['esso_model'],
        '_admm_local_solves_succeeded': ['planning_problem', 'results'],
    }
    loop_src = inspect.getsource(srp._run_operational_planning)
    step_src = inspect.getsource(aam.AndersonAccelerationState.step)
    boyd_src = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    pos = [loop_src.find(s) for s in C101.SOURCE_ORDER]
    rule = decl['settling_rule']
    cr = decl['cap_rule']
    checks = {
        'a_boyd_all_pass_key_in_production': "'all_boyd_pass'" in boyd_src,
        'a_aa_step_called_with_boyd_metrics': ('aa_record = _anderson_acceleration_cycle_step(\n'
                                               '                    aa_state, aa_layout, consensus_vars, dual_vars,\n'
                                               '                    aa_w_before, aa_rho_before, boyd_metrics, iter,')
        in loop_src,
        'a_anderson_acceleration_on': bool(aa_on),
        'b_source_order_aa_step_lt_recourse_lt_convergence_test': all(p >= 0 for p in pos) and pos[0] < pos[1] < pos[2],
        'c_no_early_stop_in_declaration': not any(k in decl for k in FORBIDDEN_DECLARATION_KEYS),
        'c_certificate_length_written_only_by_disable_restore_rule_end': (
            _certificate_length_writes_in_source() == sorted(R.CERTIFICATE_LENGTH_WRITES)),
        'c_certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(C101.CERTIFICATE_LENGTH_READS)),
        'd_spec_cap_equals_the_cell_cap_within_its_ceiling': (cap == spec_cap(cell) and cap <= cr['ceiling']
                                                              and cr['ceiling'] == CELLS[cell]['cap_ceiling']),
        'f_rule_constants_equal_settling_criterion_v3': (rule == settling_rule_declaration()
                                                         and rule['tau'] == SC3.TAU and rule['eps0'] == SC3.TAU / 100.0
                                                         and rule['gap_bound'] == SC3.TAU / 2.0
                                                         and rule['p_max'] == 30 and rule['l_mono'] == 60
                                                         and rule['version'] == 3 and rule['retry_tier'] is None),
        'f_rule_class_callable': callable(getattr(SC3, 'SettlingRuleV3', None)),
        'tail_enabled_for_this_run': bool((tail_checklist or {}).get('tail_enabled_for_this_run')),
        'aa_off_literal_is_production': (srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION == R.AA_OFF_ACTION
                                         and repr(R.AA_OFF_ACTION)[1:-1] in step_src),
        'loop_exit_is_the_convergence_break': 'if convergence:\n            print(f"[INFO] \\t - ADMM converged' in loop_src,
        'block_functions_callable': all(callable(getattr(srp, nm, None)) for nm in (
            '_get_operational_recourse_block_components', '_get_operational_objective_component_blocks')),
        'efc_read_once_per_cycle_after_penalties_before_the_row': (
            loop_src.count('_get_admm_efc_per_day_max(esso_model)') == 1
            and 0 <= loop_src.find('= _update_admm_penalties(') < loop_src.find('_get_admm_efc_per_day_max(esso_model)')
            < loop_src.find('admm_diagnostics.append({')),
        'state_carries_every_w118_state_attribute': _state_attribute_superset(),
    }
    for name, expected in params.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = callable(fn) and list(inspect.signature(fn).parameters) == expected
    if decl['replay_reference'] is not None:
        try:
            load_replay_reference(decl)
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0_lapses_as_declared'] = True
        except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0_lapses_as_declared'] = False
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    for k, v in E105.capture_path_checklist().items():
        checks[f'capture:{k}'] = bool(v)
    for k, v in R.t_sum_capture_checklist().items():
        checks[f't_sum:{k}'] = bool(v)
    for k, v in exit_capture_checklist().items():
        checks[f'exit:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W132 settling-resettle v3 preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  state, the v3 rule adapter, the ninth wrapper
# ======================================================================================================================
_MISSING = object()


class HookedRuleV3(SC3.SettlingRuleV3):
    """The v3 rule as W118's wrappers call it: `observe(k, q, boyd, t_sum)` reads all_optimal_k from the state (the
    ninth wrapper's capture of cycle k); a cycle whose exits were not captured RAISES (fail loudly)."""

    def bind(self, state):
        self._state = state
        return self

    def observe(self, k, q, boyd, t_sum, all_optimal=_MISSING):
        if all_optimal is _MISSING:
            got = self._state.all_optimal.get(k, _MISSING)
            if got is _MISSING:
                self._state.errors.append(f'cycle {k}: all_optimal_k not captured before the rule')
                raise RuntimeError(f'W132 settling-resettle v3 hook: cycle {k}: all_optimal_k not captured (the local-'
                                   f'solve wrapper did not run this cycle) -- the rule cannot decide')
            all_optimal = got
        return super().observe(k, q, boyd, t_sum, all_optimal)


class ResettleStateV3(R.ResettleState):
    """W118's state with the v3 rule. `__init__` mirrors `R.ResettleState.__init__` (pinned by fc791891) attribute by
    attribute -- W118's constructor builds a version-2 rule, which refuses a fixed cap above 300 -- plus the exit
    capture; `_state_attribute_superset` asserts before any solve that no W118 attribute is missing."""

    def __init__(self, decl, eval_dir, cap, reference=None, sink=None):
        self.decl = decl
        self.cell = decl['cell']
        self.gated = decl['replay_reference'] is not None
        self.k0_expected = decl['first_residual_pass_expected']
        self.n_old = decl['N_old']
        self.cap = cap
        self.eval_dir = eval_dir
        self.reference = reference or {}
        self.sink = sink
        cr = decl['cap_rule']
        p_max = decl['settling_rule']['p_max']
        if cr['kind'] == 'fixed':
            self.rule = HookedRuleV3(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling']).bind(self)
        else:
            self.rule = HookedRuleV3(p_max, cap_after_first_k0=cr['after_first_k0'],
                                     cap_ceiling=cr['ceiling']).bind(self)
        self.first_pass = None
        self.phase = 'before'
        self.cycle = None
        self.params = None
        self.threshold_original = None
        self.threshold_restored = None
        self.gross = {}
        self.t_sum = {}
        self.price_weight = None
        self.price_weight_sha256 = None
        self.prev_blocks = None
        self.prev_blocks_cycle = None
        self.prev_q = None
        self.prev_q_cycle = None
        self.prev_ess = None
        self.prev_ess_cycle = None
        self.ess_blocks = None
        self.consecutive = 0
        self.cur = None
        self.lines = 0
        self.creep_lines = 0
        self.ess_lines = 0
        self.replay_equal_through = 0
        self.first_divergence = None
        self.overlap = []
        self.decision = None
        self.decision_written = False
        self.ended_by = None
        self.errors = []
        self.capture_errors = []
        self.events = []
        self.t0 = time.time()
        # v3: the exit capture
        self.all_optimal = {}
        self.local_check_calls = 0
        self.init_exit = None
        self.exit_classifier = None

    def summary(self):
        s = super().summary()
        rule = self.rule
        n_cycles = self.cycle or 0
        exit_ok = (sorted(self.all_optimal) == list(range(1, n_cycles + 1)) and self.init_exit is not None
                   and self.local_check_calls == n_cycles + 1)
        s.update({
            'schema': SCHEMA, 'criterion_version': SC3.VERSION, 'reading': 'alpha',
            'first_k0_v2': rule.n, 'first_k0_alpha': rule.first_k0_alpha,
            'non_optimal_cycles': list(rule.non_optimal_cycles),
            'n_cycles_exit_captured': len(self.all_optimal), 'local_check_calls': self.local_check_calls,
            'exit_capture_init_round_0': self.init_exit, 'exit_classifier': self.exit_classifier,
            'exit_capture_complete': exit_ok,
            'gamma_report_only': rule._gamma_summary(),
            'cap_ceiling': self.decl['cap_rule']['ceiling'],
        })
        s['ok'] = bool(s['ok'] and exit_ok)
        return s


def _state_attribute_superset():
    """Every attribute a fresh W118 state carries is carried by a fresh v3 state (drift guard for the mirrored
    constructor)."""
    w118 = R.ResettleState(R.declaration_for(R.CELL_ORDER[2]), None, R.spec_cap(R.CELL_ORDER[2]), reference={}, sink=[])
    cell = CELL_ORDER[0]
    v3 = ResettleStateV3(declaration_for(cell), None, spec_cap(cell), reference={}, sink=[])
    return set(vars(w118)) <= set(vars(v3))


def make_exit_wrapper(st, original, exit_class, classifier_label=None):
    """The ninth wrapper: production's `_admm_local_solves_succeeded`, its value returned UNCHANGED; the 51 exits
    captured (the initialisation call recorded in the summary; every in-cycle call writes ipopt_exit_by_block and
    all_optimal_k into the cycle line and st.all_optimal)."""
    st.exit_classifier = classifier_label

    def raise_(msg):
        st.errors.append(msg)
        raise RuntimeError(f'W132 settling-resettle v3 hook: {msg}')

    def w_local(planning_problem, results):
        ok = original(planning_problem, results)
        st.local_check_calls += 1
        if st.phase == 'before' and st.cycle is None:
            if st.init_exit is not None:
                raise_('a second local-solve check before the first cycle')
            by_block = exit_by_block(planning_problem, results, exit_class)
            n = exit_counts(by_block)
            st.init_exit = {'round': 0, 'local_solves_ok': bool(ok), 'counts': n,
                            'all_optimal': all(v['class'] == OPTIMAL for v in by_block.values()),
                            'non_optimal_blocks': {k: v for k, v in by_block.items() if v['class'] != OPTIMAL}}
            return ok
        if st.phase != 'in_cycle' or st.cur is None or st.cur.get('finalized'):
            raise_(f'local-solve check outside a cycle (phase {st.phase}, cycle {st.cycle})')
        if 'ipopt_exit_by_block' in st.cur:
            raise_(f'cycle {st.cycle}: a second local-solve check in one cycle')
        by_block = exit_by_block(planning_problem, results, exit_class)
        n = exit_counts(by_block)
        want = st.decl['expected_blocks']
        if (n['tso'], n['dso'], n['esso']) != (want['tso'], want['dso'], want['esso']):
            raise_(f'cycle {st.cycle}: exit capture counts {n} != declared {want}')
        solverless_but_ok = [k for k, v in by_block.items() if v['has_result'] and not v['has_solver']]
        if solverless_but_ok and ok:
            raise_(f'cycle {st.cycle}: production accepted the cycle but results lack a solver record: '
                   f'{solverless_but_ok[:5]}')
        all_opt = all(v['class'] == OPTIMAL for v in by_block.values())
        st.cur['ipopt_exit_by_block'] = by_block
        st.cur['ipopt_exit_counts'] = n
        st.cur['all_optimal_k'] = all_opt
        st.cur['non_optimal_blocks'] = sorted(k for k, v in by_block.items() if v['class'] != OPTIMAL)
        st.cur['local_solves_ok_at_exit_capture'] = bool(ok)
        st.all_optimal[st.cycle] = all_opt
        return ok

    return w_local


def make_wrappers(st, originals, exit_class, srp_module=None, classifier_label=None):
    """The nine wrappers over `originals` (name -> callable): W118's eight (`R.make_wrappers`, unchanged) and the exit
    wrapper. Separated from the context manager so the zero-solve checks can drive them with stand-ins."""
    wrappers = R.make_wrappers(st, {n: originals[n] for n in R.WRAPPED}, srp_module=srp_module)
    wrappers['_admm_local_solves_succeeded'] = make_exit_wrapper(st, originals['_admm_local_solves_succeeded'],
                                                                 exit_class, classifier_label)
    return wrappers


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the nine wrappers for the run (the harness enters this FIRST, so these wrap production directly and
    every harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle v3: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateV3(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = make_wrappers(st, originals, HAR.ipopt_exit_class, srp_module=srp,
                             classifier_label=f'p515_s44_campaign_harness.ipopt_exit_class ({HAR.__file__})')
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()
