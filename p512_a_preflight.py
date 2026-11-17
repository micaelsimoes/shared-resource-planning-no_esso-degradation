"""P5.12-A0 preflight: provenance gate and the two frozen reproduction gates.

Writes only to data/SRP1/Results/P512A/. Accepted P5.10 and P5.11 evidence is
read but never rewritten.
"""
import json, os, subprocess, sys
from datetime import datetime, timezone
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import p56b_candidates as BC  # noqa: E402
import p510_oracle as OR  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P512A')
OUT = os.path.join(OUT_DIR, 'p512a_preflight.json')
INHERITED = dict(rho_v=1.5, rho_pf=2.25, rho_ess=1.0)
GATES = [('CURRENT_polished_total', OR.SCALING_CURRENT, 'total_objective',
          828021090.3608505),
         ('RESCALED_prepolish_recourse', OR.SCALING_RESCALED,
          'admm_net_recourse_before_polish', 825814074.4930633)]

def git(*a):
    try:
        return subprocess.check_output(['git', *a], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = {'stage': 'P5.12-A0',
           'timestamp_utc': datetime.now(timezone.utc).isoformat(),
           'repo': REPO_ROOT,
           'branch': git('branch', '--show-current'),
           'head': git('rev-parse', 'HEAD'),
           'tracked_modifications': len([l for l in (git('status', '--porcelain') or '').splitlines()
                                         if not l.startswith('??')])}
    try:
        provenance, planning_gate = gate('P5.12-A0 preflight', OUT_DIR)
    except ProvenanceError as e:
        rep['provenance_gate'] = {'passed': False, 'error': str(e)}
        json.dump(rep, open(OUT, 'w'), indent=1, default=str)
        print(f'\n[P5.12-A0] ABORTED\n{e}'); sys.exit(1)
    rep['provenance_gate'] = {'passed': True, 'provenance': provenance}
    rep['runtime'] = sys.executable
    x0 = dict(BC.population(planning_gate))['base']
    rep['gates'] = {}
    for name, mode, field, expected in GATES:
        cfg = OR.OracleConfig(scaling_mode=mode, adaptive_penalty=True,
                              neutralize_history=False, **INHERITED)
        print(f'[P5.12-A0] gate {name} ...', flush=True)
        rec, _ = OR.evaluate(x0, cfg, case_id=f'p512a0_{name}')
        obs = rec.get(field)
        delta = (obs - expected) if obs is not None else None
        rep['gates'][name] = {'expected': expected, 'observed': obs,
                              'delta': delta, 'bit_identical': obs == expected,
                              'status': rec.get('status'),
                              'cycles': (rec.get('admm') or {}).get('cycles')}
        json.dump(rep, open(OUT, 'w'), indent=1, default=str)
        print(f'          observed {obs}  delta {delta}  '
              f'bit-identical {obs == expected}', flush=True)
    rep['all_gates_bit_identical'] = all(g['bit_identical'] for g in rep['gates'].values())
    json.dump(rep, open(OUT, 'w'), indent=1, default=str)
    print(f"\n[P5.12-A0] all gates bit-identical: {rep['all_gates_bit_identical']}")
    print(f'[P5.12-A0] -> {OUT}')

if __name__ == '__main__':
    main()
