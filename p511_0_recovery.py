"""
Stage P5.11.0 -- repository and provenance recovery, and the frozen gates.

Records the repository state, inventories and hashes the accepted P5.10
artifacts, resolves the runtime discrepancy from evidence rather than from the
documents, and reproduces the two frozen reproduction gates:

    CURRENT  polished total        828021090.3608505
    RESCALED pre-polish recourse   825814074.4930633

Read-only with respect to git: nothing is merged, pushed, committed or checked
out by this harness.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p511_0_recovery.py
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p510_oracle as OR  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P511')
OUT_PATH = os.path.join(OUT_DIR, 'p511_0_recovery.json')

GATE_CURRENT = 828021090.3608505
GATE_RESCALED = 825814074.4930633
GATE_TOL = 1.0

ARTIFACTS = ['p510_oracle.py', 'p510_a_state.py', 'p510_b_fixedrho.py',
             'p510_c_endpoint.py', 'p510_e_criteria.py', 'p510_f_replay.py',
             'p510_g_anchor.py',
             'P5_10_STABILIZED_RESCALLED_ADMM_ORACLE_REPORT.md']

INHERITED = dict(rho_v=1.5, rho_pf=2.25, rho_ess=1.0)


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as error:
        return f'<{type(error).__name__}>'


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.11.0',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'repo_root': REPO_ROOT}

    # ---- 1. repository state ------------------------------------------------
    report['repository'] = {
        'branch': _git('branch', '--show-current'),
        'head': _git('rev-parse', 'HEAD'),
        'head_subject': _git('log', '-1', '--pretty=%s'),
        'origin_url': _git('remote', 'get-url', 'origin'),
        'origin_branch_tip': _git('rev-parse', 'origin/feature/derivative-free-planning'),
        'origin_branch_subject': _git(
            'log', '-1', '--pretty=%s', 'origin/feature/derivative-free-planning'),
        'tracked_modifications': len([
            line for line in _git('status', '--porcelain').splitlines()
            if not line.startswith('??')]),
        'recent_history': _git('log', '--oneline', '-8').splitlines(),
    }

    # ---- 2/3. documentation history ----------------------------------------
    report['documentation'] = {
        'stale_commit': {
            'sha': '518aa1d8',
            'subject': _git('log', '-1', '--pretty=%s', '518aa1d8'),
            'author': _git('log', '-1', '--pretty=%an <%ae>', '518aa1d8'),
            'date': _git('log', '-1', '--pretty=%cd', '--date=iso', '518aa1d8'),
            'parent': _git('log', '-1', '--pretty=%p', '518aa1d8')},
        'restores': {
            'df77a9b9': _git('log', '-1', '--pretty=%s', 'df77a9b9'),
            '0bfeb1ae': _git('log', '-1', '--pretty=%s', '0bfeb1ae')},
        'current_doc_state': {},
    }
    for name in ('REVISION_CONTEXT.md', 'LOCAL_NLP_STABILITY_PLAN.md'):
        path = os.path.join(REPO_ROOT, name)
        text = open(path).read()
        report['documentation']['current_doc_state'][name] = {
            'lines': text.count('\n') + 1,
            'sha256': _sha256(path),
            'mentions': {k: text.count(k) for k in
                         ('P5.7', 'P5.8', 'P5.9', 'P5.10')},
            'has_stale_p56c_heading': (
                'CURRENT AUTHORIZED STAGE' in text or 'CURRENT ACTIVE STAGE' in text)}

    # ---- 4/5. artifact inventory -------------------------------------------
    inventory = {}
    for name in ARTIFACTS:
        path = os.path.join(REPO_ROOT, name)
        inventory[name] = ({'present': False} if not os.path.exists(path) else {
            'present': True, 'bytes': os.path.getsize(path),
            'sha256': _sha256(path),
            'tracked': _git('ls-files', '--error-unmatch', name) == name})
    evidence_dir = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P510')
    evidence = {}
    for root, _, files in os.walk(evidence_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            evidence[os.path.relpath(path, REPO_ROOT)] = {
                'bytes': os.path.getsize(path), 'sha256': _sha256(path)}
    report['artifacts'] = {'harnesses_and_report': inventory,
                           'evidence_files': len(evidence),
                           'evidence': evidence}
    report['artifacts_all_recovered'] = all(
        v.get('present') for v in inventory.values()) and bool(evidence)

    # ---- 6. runtime resolution ---------------------------------------------
    candidates = {
        '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python':
            'named by the P5.10 report and by the corrected documents',
        '/opt/anaconda3/envs/opf_env_py311/bin/python':
            'named by the stale plan snapshot (MacBook Air path)'}
    report['runtime_candidates'] = {
        path: {'exists': os.path.exists(path), 'note': note}
        for path, note in candidates.items()}

    try:
        provenance, planning_gate = gate('P5.11.0 recovery', OUT_DIR)
    except ProvenanceError as error:
        report['provenance_gate'] = {'passed': False, 'error': str(error)}
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)
        print(f'\n[P5.11.0] ABORTED\n{error}')
        sys.exit(1)
    report['provenance_gate'] = {'passed': True, 'provenance': provenance}
    report['canonical_runtime_selected'] = sys.executable

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('\n[P5.11.0] repository')
    for key in ('branch', 'head', 'origin_branch_tip', 'origin_branch_subject',
                'tracked_modifications'):
        print(f"    {key:24} {report['repository'][key]}")
    print(f"\n[P5.11.0] artifacts recovered: {report['artifacts_all_recovered']}  "
          f"({len(inventory)} harnesses/report, {len(evidence)} evidence files)")
    print(f"[P5.11.0] canonical runtime  : {sys.executable}")

    # ---- 7. the frozen gates ------------------------------------------------
    x0 = dict(BC.population(planning_gate))['base']
    report['gates'] = {}
    specs = [
        ('CURRENT_polished_total', OR.OracleConfig(
            scaling_mode=OR.SCALING_CURRENT, adaptive_penalty=True,
            neutralize_history=False, **INHERITED),
         'total_objective', GATE_CURRENT),
        ('RESCALED_prepolish_recourse', OR.OracleConfig(
            scaling_mode=OR.SCALING_RESCALED, adaptive_penalty=True,
            neutralize_history=False, **INHERITED),
         'admm_net_recourse_before_polish', GATE_RESCALED),
    ]
    print()
    for name, config, field, expected in specs:
        print(f'[P5.11.0] gate {name} ...', flush=True)
        record, _ = OR.evaluate(x0, config, case_id=f'p511_0_{name}')
        value = record.get(field)
        delta = (value - expected) if value is not None else None
        report['gates'][name] = {
            'config': config.as_dict(), 'field': field, 'expected': expected,
            'observed': value, 'delta': delta,
            'passed': delta is not None and abs(delta) <= GATE_TOL,
            'status': record.get('status'),
            'admm_cycles': (record.get('admm') or {}).get('cycles')}
        persist()
        print(f"          observed {value}  expected {expected}  delta {delta}  "
              f"-> {'PASS' if report['gates'][name]['passed'] else 'FAIL'}",
              flush=True)

    report['all_gates_passed'] = all(g['passed'] for g in report['gates'].values())
    persist()
    print(f"\n[P5.11.0] all gates passed: {report['all_gates_passed']}")
    print(f'[P5.11.0] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
