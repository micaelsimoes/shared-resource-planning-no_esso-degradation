"""
Stage P5.13-D -- component-level impact measurement for activating calibration C3.

Frozen gate: data/SRP1/Results/P513D/frozen_c3_impact_gate_v1_8ee17c92.json

Compares two ordered-state captures produced by p513_c_param_move_gate.py and
reports, per model component, whether it changed. The gate expects EXACTLY ONE
changed component (energy_storage_capacity_degradation) carrying 2*k in place of
2*cl_nom. Any other difference fails.

No solver is invoked.

    python p513_d_c3_impact.py --pre <state_pre_nodeN.json> --post <state_c3_post_nodeN.json>
"""

import argparse
import hashlib
import json
import math
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'SRP1', 'Results', 'P513D')

N, D, R = 10000, 0.80, 0.50
K_EXPECTED = N * D / (-math.log(R))
TWO_K = repr(2 * K_EXPECTED)
TWO_CL_NOM = '20000'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode('utf-8')).hexdigest()


def compare_states(pre, post):
    changed, unchanged = [], []
    for section in sorted(set(pre) | set(post)):
        pre_section, post_section = pre.get(section, {}), post.get(section, {})
        for name in sorted(set(pre_section) | set(post_section)):
            label = f'{section}.{name}'
            if pre_section.get(name) == post_section.get(name):
                unchanged.append(label)
            else:
                changed.append(label)
    return changed, unchanged


def row_evidence(pre, post):
    """The degradation rows, before and after, with the constant they carry."""
    key = 'energy_storage_capacity_degradation'
    pre_rows = pre.get('constraints', {}).get(key, {}).get('rows', {})
    post_rows = post.get('constraints', {}).get(key, {}).get('rows', {})
    differing = [r for r in sorted(set(pre_rows) | set(post_rows), key=lambda x: int(x))
                 if pre_rows.get(r) != post_rows.get(r)]
    sample = differing[0] if differing else None
    return {
        'row_count_pre': len(pre_rows), 'row_count_post': len(post_rows),
        'differing_row_count': len(differing),
        'sample_row_key': sample,
        'sample_pre': pre_rows.get(sample) if sample else None,
        'sample_post': post_rows.get(sample) if sample else None,
        'rows_carrying_2_cl_nom_pre': sum(1 for r in pre_rows.values() if TWO_CL_NOM in r),
        'rows_carrying_2k_post': sum(1 for r in post_rows.values() if TWO_K in r),
        'rows_carrying_2_cl_nom_post': sum(1 for r in post_rows.values() if TWO_CL_NOM in r),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre', required=True)
    parser.add_argument('--post', required=True)
    args = parser.parse_args()

    with open(args.pre) as handle:
        pre = json.load(handle)
    with open(args.post) as handle:
        post = json.load(handle)

    changed, unchanged = compare_states(pre, post)
    evidence = row_evidence(pre, post)

    expected_changed = ['constraints.energy_storage_capacity_degradation']
    failures = []
    if changed != expected_changed:
        failures.append(f'E1 confinement: changed components {changed} != {expected_changed}')
    if evidence['rows_carrying_2k_post'] != evidence['differing_row_count'] or not evidence['differing_row_count']:
        failures.append('E2 coefficient: not every differing row carries 2*k')
    if evidence['rows_carrying_2_cl_nom_post']:
        failures.append('E2 coefficient: a post row still carries 2*cl_nom')
    if repr(K_EXPECTED) != '11541.560327111707':
        failures.append('E3 derivation: k does not reproduce the authorized value')

    verdict = {
        'stage': 'P5.13-D',
        'gate_spec': 'data/SRP1/Results/P513D/frozen_c3_impact_gate_v1_8ee17c92.json',
        'calibration': {'N': N, 'D': D, 'R': R, 'k': K_EXPECTED, 'two_k': 2 * K_EXPECTED,
                        'k_over_cl_nom': K_EXPECTED / N},
        'pre_file': os.path.basename(args.pre), 'post_file': os.path.basename(args.post),
        'pre_state_sha256': digest(pre), 'post_state_sha256': digest(post),
        'changed_components': changed,
        'unchanged_component_count': len(unchanged),
        'degradation_rows': evidence,
        'failures': failures,
        'verdict': 'PASS — impact confined to the degradation rows' if not failures
                   else 'FAIL — impact is not what the change claims',
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'c3_impact_verdict.json')
    with open(out, 'w') as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2)[:2600])
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
