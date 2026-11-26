"""
Stage P5.10-G -- interface anchor, under the stabilized oracle.

Secondary by instruction, and no anchor policy is changed. P5.9-E already
established, under a configuration differing from this one only in the adaptive
flag, that the anchor is untouched by rescaling and by the rho override
(`max |pc(T0) - pc(rescaled)| = 0.0` exactly), that it is common across
candidates and across cold/warm starts (P5.8-E3, within 1.003037e-07 p.u.), and
that the P5.7 flexibility-against-a-frozen-reference mechanism persists at about
one third the magnitude under RESCALED (P5.9-E2: +298 653 against +865 891 over
four generations).

Only what the stabilized configuration could newly disturb is measured here, and
it needs no solves: whether disabling adaptive rho and neutralising the history
channels perturbs the fixed anchor. Nothing in either acts on `pc`/`qc`, so a
nonzero result would indicate a defect in this stage's own machinery.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p510_g_anchor.py
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p57_eval as E7  # noqa: E402
import p58_e_anchor as E8A  # noqa: E402
import p510_oracle as OR  # noqa: E402

OUT_PATH = os.path.join(OR.OUT_DIR, 'p510_g_anchor.json')


def main():
    os.makedirs(OR.OUT_DIR, exist_ok=True)
    with open(E7.T0_CACHE, 'rb') as handle:
        t0 = pickle.load(handle)
    anchor_t0 = E8A.anchor(t0['models']['tso'])

    report = {'stage': 'P5.10-G',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'anchor_policy_changed': False,
              'blocks': len(anchor_t0),
              'carried_from_P5.9_E': {
                  'rescaling_moves_anchor': '0.0 exactly for pc and qc',
                  'common_across_candidates': 'yes (P5.8-E2, 0.000000e+00)',
                  'cold_versus_warm': '1.003037e-07 p.u. (P5.8-E3)',
                  'flexibility_growth_RESCALED_vs_CURRENT':
                      '+298 653 against +865 891 over four generations (P5.9-E2)'},
              'comparisons': {}}

    for label, config in (
        ('stabilized (adaptive off, history neutralised)',
         OR.OracleConfig(scaling_mode=OR.SCALING_RESCALED, rho_v=1.5,
                         rho_pf=1000.0, rho_ess=1.0, adaptive_penalty=False,
                         neutralize_history=True)),
        ('adaptive on, history inherited (P5.9 configuration)',
         OR.OracleConfig(scaling_mode=OR.SCALING_RESCALED, rho_v=1.5,
                         rho_pf=1000.0, rho_ess=1.0, adaptive_penalty=True,
                         neutralize_history=False)),
    ):
        state, _ = OR.prepare_template(config)
        report['comparisons'][label] = E8A.compare(
            anchor_t0, E8A.anchor(state['models']['tso']))

    report['anchor_can_influence_ranking'] = (
        'no: the anchor is bit-identical across candidates under the locked T0 '
        'policy, so it enters every candidate objective as the same constant '
        'reference and cannot differentiate them')

    with open(OUT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('[P5.10-G] anchor, stabilized oracle')
    for label, block in report['comparisons'].items():
        for name, value in block.items():
            print(f"   {label:48} {name}: {value['max_over_blocks']:.6e}")
    print(f'\n[P5.10-G] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
