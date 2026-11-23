"""P5.15 S56 step 4 MAIN RUN -- the declared cycle-40 look (spec item (iv)) with its numeric parameters supplied.

Defect found at the first look (2026-11-23 15:29:44 UTC, instance B): `p515_s56_step4_main.py --look40` reads
look['cycle'], look['fit_cycles'], ... from the frozen spec's `iv_cycle40_look` block, but that block holds the look in
prose only (keys fit, output, projection, recorded_beside_not_deciding, stop_rule, what, when); the numeric form lives in
the launcher constant LOOK40 and was never written to the spec. The look raised KeyError: 'cycle' before evaluating
(zero solves; output look40_B_20261123T152944.json, error recorded). The running campaign is unaffected.

This script runs the SAME launcher function (`look40`, unchanged) with the spec block extended IN MEMORY by LOOK40,
after asserting that every LOOK40 number is the one the spec prose states. Nothing is edited; zero solves (the
launcher's permitted=() guard is installed on import and verified in the output). Output: look40_<inst>_<utc>_LOOK40params.json
in the spec's main directory.
Usage: <canonical python> p515_s56_step4_main_look40_params.py --spec S --instance A|B
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s56_step4_main as L  # noqa: E402  (installs its permitted=() guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--instance', required=True, choices=L.INSTANCES)
    a = ap.parse_args()
    spec, sha, rel = L._spec(a.spec)
    block = spec['iv_cycle40_look']
    lk = L.LOOK40
    prose = ' '.join(str(v) for v in block.values())
    checks = {
        'cycle_40': lk['cycle'] == 40 and 'cycle-40' in block['when'],
        'fit_21_34': lk['fit_cycles'] == [21, 34] and 'cycles 21..34' in block['fit'],
        'heldout_35_40': lk['heldout_cycles'] == [35, 40] and 'cycles 35..40' in block['fit'],
        'projection_21_40': lk['projection_fit_cycles'] == [21, 40] and 'refits on cycles 21..40' in block['fit'],
        'stop_600': lk['stop_if_projected_cycles_exceed'] == 600 and 'k_star > 600' in block['stop_rule'],
        'projection_formula_k40': 'k_star = 40 + ln(allowance_b / |e_hat(40)|) / ln r' in block['projection'],
    }
    if not all(checks.values()):
        raise SystemExit(f'LOOK40 does not match the spec prose: {checks}\n{prose}')
    merged = dict(block)
    merged.update(lk)

    def _spec_patched(path):
        s, h, r = L._spec.__wrapped__(path)
        s['iv_cycle40_look'] = merged
        return s, h, r

    _spec_patched.__wrapped__ = L._spec
    L._spec = _spec_patched
    stamp = L._stamp
    L._stamp = lambda: stamp() + '_LOOK40params'
    print(f'LOOK40 vs spec prose: {checks}', flush=True)
    return L.look40(argparse.Namespace(spec=a.spec, instance=a.instance))


if __name__ == '__main__':
    sys.exit(main())
