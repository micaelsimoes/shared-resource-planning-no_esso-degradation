"""
Stage P5.6-B1 -- stabilize the canonical START-2 template.

START-2 uses a fixed archived template taken from the canonical base candidate's
COLD solution.  But START-2 then finds a materially better base solution than the
cold start did, which means the template is not a fixed point of its own
procedure: re-archiving the better solution could shift it again.  This chains
the refinement to find out whether it settles.

    T0 = the archived cold-base template
    Tk -> evaluate the SAME canonical base candidate -> archive its state -> Tk+1

Every refinement starts from a fresh candidate-specific deep copy of the
baseline, uses the same candidate, the same interface-anchor policy and the same
physical ESSO/polish pipeline, and never depends on any non-base candidate.

DECLARED BEFORE EVALUATING
--------------------------
    template objective stabilization tolerance = 100 monetary units

Stop when |Q(Tk+1) - Q(Tk)| <= 100 for two CONSECUTIVE refinements and every
physical/coupling gate passes.  If it oscillates or is still moving materially
after 5 refinements, report PARTIAL rather than taking whichever happens to be
last.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56b_b1_template.py
"""

import hashlib
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_candidates as C  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p56b_policy as P  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = P.OUT_DIR
STABILIZATION_TOLERANCE = 100.0
MAX_REFINEMENTS = 5
ANCHOR_POLICY = P.POLICY_MIDPOINT_THEN_DSO


def fingerprint(result):
    """The operational-state fingerprint B1 compares across refinements."""
    blocks = result.get('per_block_polished', {})
    ranked = sorted(blocks.items(),
                    key=lambda kv: -abs(kv[1]['weighted_base_objective']))[:5]
    return {
        'admm_cycles': result['admm']['cycles'],
        'total_objective': result['total_objective'],
        'net_operational_recourse': result['net_operational_recourse'],
        'gross_operational_cost': result['gross_operational_cost'],
        'physical_salvage': result['physical_salvage'],
        'interface_anchor': result['interface_anchor'],
        'fallback_used': result['fallback_used'],
        'dominant_blocks': [{'block': k,
                             'weighted_base_objective':
                                 v['weighted_base_objective']}
                            for k, v in ranked],
        'available_capacity': result['available_capacity'],
        'max_coordinated_residual':
            result['coordination_residuals']['max_coordinated'],
        'esso_max_violation': result['esso_audit']['max_violation'],
        'network_max_violation': result['network_audit']['max_violation'],
        'network_h1_violation':
            result['network_audit']['max_h1_complementarity_violation'],
    }


def interface_summary(planning, models, state):
    """Aggregate interface and shared-ESS quantities of the polished state."""
    common = O.common_coordinated_values(planning, models,
                                         state['consensus_vars'])
    p_abs = sum(abs(v['common_p']) for v in common.values())
    q_abs = sum(abs(v['common_q']) for v in common.values())
    sp_abs = sum(abs(v['common_sess_p']) for v in common.values())
    sq_abs = sum(abs(v['common_sess_q']) for v in common.values())
    v_abs = sum(abs(v['common_v']) for v in common.values())
    return {'sum_abs_interface_p_pu': p_abs, 'sum_abs_interface_q_pu': q_abs,
            'sum_abs_shared_ess_p_pu': sp_abs, 'sum_abs_shared_ess_q_pu': sq_abs,
            'sum_abs_interface_v_pu': v_abs, 'n_entries': len(common)}


def template_id(fp, config_hash, checksum):
    blob = json.dumps({'fingerprint': fp, 'config': config_hash,
                       'checksum': checksum}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-B1 template', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[B1] ABORTED\n{error}')
        sys.exit(1)

    x = C.base_vector(planning_gate)
    print(f'[B1] declared stabilization tolerance = {STABILIZATION_TOLERANCE} '
          f'monetary units; at most {MAX_REFINEMENTS} refinements')
    print(f'[B1] anchor policy = {ANCHOR_POLICY}\n', flush=True)

    print('[B1] building T0: the archived cold-base template ...', flush=True)
    started = time.time()
    t0 = O.build_fixed_template(verbose=False)
    print(f'[B1] T0 built in {time.time() - started:.1f}s', flush=True)

    report = {'stage': 'P5.6-B1', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'stabilization_tolerance': STABILIZATION_TOLERANCE,
              'max_refinements': MAX_REFINEMENTS,
              'anchor_policy': ANCHOR_POLICY,
              'refinements': []}

    template = t0
    previous_q = None
    consecutive_within = 0
    stabilized_at = None
    for k in range(MAX_REFINEMENTS):
        print(f'[B1] refinement {k}: evaluating base with T{k} ...', flush=True)
        result = P.evaluate(x, template_state=template,
                            anchor_policy=ANCHOR_POLICY,
                            eval_id=f'b1_T{k}', keep_state=True,
                            keep_models=True)
        entry = {'refinement': k, 'template_used': f'T{k}',
                 'certificate': P.certificate_summary(result)}
        if result['status'] != O.STATUS_VALID:
            entry['status'] = result['status']
            report['refinements'].append(entry)
            print(f"      status={result['status']} -- chain stops here")
            break
        fp = fingerprint(result)
        entry['fingerprint'] = fp
        entry['interface_summary'] = interface_summary(
            result['_planning'], result['_models'], result['_state'])
        q = result['total_objective']
        entry['total_objective'] = q
        if previous_q is not None:
            entry['delta_vs_previous'] = q - previous_q
            entry['within_tolerance'] = abs(q - previous_q) <= STABILIZATION_TOLERANCE
            consecutive_within = (consecutive_within + 1
                                  if entry['within_tolerance'] else 0)
            entry['consecutive_within'] = consecutive_within
            print(f"      Q(T{k}) = {q:.6f}   delta = {q - previous_q:+.6f}   "
                  f"within = {entry['within_tolerance']}   "
                  f"consecutive = {consecutive_within}", flush=True)
        else:
            print(f"      Q(T{k}) = {q:.6f}   (baseline of the chain)", flush=True)
        report['refinements'].append(entry)
        previous_q = q
        # the state produced becomes the next template
        template = result['_state']
        if consecutive_within >= 2:
            stabilized_at = k
            print(f'[B1] stabilized: two consecutive refinements within '
                  f'{STABILIZATION_TOLERANCE}')
            break

    report['stabilized'] = stabilized_at is not None
    report['stabilized_at_refinement'] = stabilized_at

    if stabilized_at is not None:
        final = report['refinements'][-1]
        t_star = {
            'id': template_id(final['fingerprint'],
                              O._config_hash(planning_gate),
                              O.load_baseline()['checksum']),
            'source_candidate': 'canonical positive-bootstrap base candidate',
            'produced_by_refinement': stabilized_at,
            'config_hash': O._config_hash(planning_gate),
            'canonical_checksum': O.load_baseline()['checksum'],
            'oracle_version': O.ORACLE_VERSION,
            'anchor_policy': ANCHOR_POLICY,
            'total_objective': final['total_objective'],
            'fingerprint': final['fingerprint'],
        }
        report['T_STAR'] = t_star
        print(f"\n[B1] T_STAR frozen: id {t_star['id']}")
        print(f"     total objective {t_star['total_objective']:.6f}")
    else:
        print('\n[B1] NOT stabilized within the refinement budget')

    out = os.path.join(OUT_DIR, 'p56b_b1_template.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[B1] report -> {out}')


if __name__ == '__main__':
    main()
