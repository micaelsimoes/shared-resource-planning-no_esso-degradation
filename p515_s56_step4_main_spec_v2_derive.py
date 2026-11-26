"""P5.15 S56 step 4 MAIN RUN: derive the frozen addendum spec v2 from spec v1 3dfb2889 (zero solves, reads only).

Why: the v1 attempt (instance B into step4_main_v1/campaign_s56s4main_v1_main_B, 2026-11-23 14:47:08 UTC) was killed by
the Claude session exit before cycle 1 (step4_main_v1/killed_attempt_v1_note.json). Its root is preserved, so the
relaunch needs new roots. v2 = v1 loaded VERBATIM with only these keys set:
  version 2; predecessor -> v1 (path, sha256); why_v2; roots (main_dir, root_tag 'v2', absolute campaign roots
  campaign_s56s4main_v2_<phase>_<inst> under data/SRP1/Results/P515S56/step4_main_v2/);
  i_arms_instances_cap.runs[*].campaign_root -> the v2 absolute roots;
  v_continuation.how -> the v2 continuation root name (the text is otherwise unchanged).
Everything else (arms, instances, predictions, cap, cycle-40 look, continuation rule, cites) is v1's, checked here by a
key-path diff that must list exactly those paths. Serialisation as v1 (json.dumps indent 1, sort_keys, default=str);
the file name carries the first 8 hex digits of the text's sha256.
Usage: <canonical python> p515_s56_step4_main_spec_v2_derive.py
"""
import hashlib
import json
import os
import sys
from copy import deepcopy

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
V1 = os.path.join(RES, 'step4_main_v1', 'frozen_step4_main_spec_v1_3dfb2889.json')
V2_DIR = os.path.join(RES, 'step4_main_v2')
V2_PREFIX = 'frozen_step4_main_spec_v2_'
TAG = 'v2'
LAUNCHER = 'p515_s56_step4_main.py'
NOTE = os.path.join(RES, 'step4_main_v1', 'killed_attempt_v1_note.json')


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def paths(obj, prefix=()):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(paths(v, prefix + (str(k),)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(paths(v, prefix + (str(i),)))
        return out
    return {'.'.join(prefix): obj}


def root(phase, inst):
    return os.path.join(REPO, V2_DIR, f'campaign_s56s4main_{TAG}_{phase}_{inst}')


def main():
    v1_sha = sha(V1)
    if not v1_sha.startswith('3dfb2889'):
        raise SystemExit(f'v1 sha256 {v1_sha} is not 3dfb2889')
    with open(V1, encoding='utf-8') as handle:
        v1 = json.load(handle)
    v2 = deepcopy(v1)
    v2['version'] = 2
    v2['predecessor'] = {'path': V1, 'sha256': v1_sha,
                         'relation': ('spec v1 3dfb2889, whose instance-B attempt was killed by the session exit before '
                                      'cycle 1 and is preserved in its own root. v2 is v1 verbatim except the campaign '
                                      'roots (new, absolute, version-tagged) and this field set; v1\'s own predecessor is '
                                      'the smoke spec v2 e36c29bd, unchanged')}
    v2['why_v2'] = {
        'reason': ('relaunch of the step 4 main run (author authorization 2026-11-23, inside the existing 10-20 h). The v1 '
                   'attempt (instance B, step4_main_v1/campaign_s56s4main_v1_main_B, launched 14:47:08 UTC) was killed '
                   'by the Claude session exit before cycle 1; its root is preserved and committed (4e6bb1d3), so v1\'s '
                   'roots cannot be reused (write-once)'),
        'killed_attempt_note': {'path': NOTE, 'sha256': sha(NOTE)},
        'changed': ['version', 'predecessor', 'why_v2', 'roots', 'i_arms_instances_cap.runs[*].campaign_root',
                    'v_continuation.how (root name only)'],
        'unchanged': 'everything else, verbatim from v1 (arms, instances, predictions, cap, cycle-40 look, continuation)',
        'launcher_change': ('p515_s56_step4_main.py takes MAIN_DIR, ROOT_TAG and the absolute campaign roots from this '
                            'spec\'s roots block (apply_spec_roots); nothing else changed. Launcher records, logs, '
                            'vm_stat snapshots and look-40 files go to roots.main_dir'),
        'derivation_script': 'p515_s56_step4_main_spec_v2_derive.py',
    }
    v2['roots'] = {'main_dir': V2_DIR, 'root_tag': TAG,
                   'campaign_roots': {p: {i: root(p, i) for i in ('A', 'B')} for p in ('main', 'cont600')}}
    for run in v2['i_arms_instances_cap']['runs']:
        run['campaign_root'] = root(run['phase'], run['instance'])
    old_how = v1['v_continuation']['how']
    assert old_how.count('campaign_s56s4main_v1_cont600_<inst>') == 1
    v2['v_continuation']['how'] = old_how.replace('campaign_s56s4main_v1_cont600_<inst>',
                                                  f'{V2_DIR}/campaign_s56s4main_{TAG}_cont600_<inst> (absolute)')
    p1, p2 = paths(v1), paths(v2)
    changed = sorted(k for k in set(p1) | set(p2) if p1.get(k, '<absent>') != p2.get(k, '<absent>'))
    allowed = ('version', 'predecessor.', 'why_v2.', 'roots.', 'v_continuation.how')
    bad = [k for k in changed if not (k.startswith(allowed) or (k.startswith('i_arms_instances_cap.runs.')
                                                                and k.endswith('.campaign_root')))]
    if bad:
        raise SystemExit(f'unexpected changes from v1: {bad}')
    os.makedirs(V2_DIR, exist_ok=True)
    if any(f.startswith(V2_PREFIX) for f in os.listdir(V2_DIR)):
        raise SystemExit('a v2 spec already exists')
    text = json.dumps(v2, indent=1, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(V2_DIR, f'{V2_PREFIX}{digest[:8]}.json')
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(text)
    print(json.dumps({'path': path, 'sha256': digest, 'changed_paths_vs_v1': changed}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
