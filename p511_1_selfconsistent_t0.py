"""
Stage P5.11.1 -- a T0 template generated entirely under the frozen RESCALED
configuration, and the inherited-versus-self-consistent comparison.

WHY THIS EXISTS.  Every oracle from P5.6-B to P5.10 warm-starts from a template
that was built COLD UNDER THE CURRENT FORMULATION and then retrofitted: P5.8's
`rescale_state_models` multiplies the already-built objectives, and P5.10 writes
rho into the already-built models. The result is correct by construction -- a
positive constant multiple leaves the argmin untouched -- but it means the
template's PRIMAL STATE is still the one a CURRENT-formulation cold solve
produced. P5.11.1 removes that last inheritance by building the template cold
with the rescaled objectives in place from the start.

`p58_rescale.patched_admm_objectives` is the right instrument: it wraps the two
production functions that BUILD the augmented objectives, so a cold construction
produces RESCALED objectives natively rather than by retrofit.
`rescale_state_models` is the retrofit path and is what the inherited template
uses.

HARNESS-LOCAL ONLY.  Production initialization is untouched; the diagnostic
template is written to the P5.11 evidence directory under its own identifier and
is never substituted for `p57_eval.T0_CACHE`. `data/SRP1/SRP1_params.json` is
never written.

    python p511_1_selfconsistent_t0.py build
    python p511_1_selfconsistent_t0.py compare "<candidate>"
"""

import hashlib
import io
import json
import os
import pickle
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import p510_oracle as OR  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P511')
TEMPLATE_PATH = os.path.join(OUT_DIR, 'p511_selfconsistent_t0.pkl')
TEMPLATE_META = os.path.join(OUT_DIR, 'p511_selfconsistent_t0_meta.json')

RHO_PRIMARY = 300.0
RHO_CROSSCHECK = 1000.0
CANDIDATES = ['base', 'se|node5|2025|-10%', 'se|node9|2025|-10%']
DELTA_TOLERANCE = 22.09          # P5.10-F provisional planning uncertainty


def config_for(rho_pf):
    return OR.OracleConfig(
        scaling_mode=OR.SCALING_RESCALED, rho_v=1.5, rho_pf=rho_pf,
        rho_ess=1.0, adaptive_penalty=False, neutralize_history=True,
        template_id='P511-SELFCONSISTENT-T0',
        initialization_policy='cold build under RESCALED, frozen',
        notes='P5.11.1 self-consistent template')


# ===========================================================================
#  a stable hash of a template's numerical state
# ===========================================================================
def state_hash(state):
    """SHA-256 over the consensus/dual content, in a fixed traversal order.

    Pyomo models do not hash stably, so the hash covers the numerical state the
    warm start actually consumes: the consensus and dual variables. Two
    templates with the same hash carry the same coordination state.
    """
    digest = hashlib.sha256()

    def walk(node, path=''):
        if isinstance(node, dict):
            for key in sorted(node, key=repr):
                walk(node[key], f'{path}/{key!r}')
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                walk(item, f'{path}[{i}]')
        elif isinstance(node, (int, float, np.floating, np.integer)):
            digest.update(f'{path}={float(node):.17g}'.encode())
        elif isinstance(node, (str, bool)) or node is None:
            digest.update(f'{path}={node!r}'.encode())

    for key in ('consensus_vars', 'dual_vars'):
        walk(state.get(key), key)
    return digest.hexdigest()


def neutralize(state):
    for key in OR.HISTORY_KEYS:
        state[key] = 0 if key == 'consecutive_converged_cycles' else None
    state['candidate_solution'] = None
    state['initialization_failed'] = False
    return state


# ===========================================================================
#  build
# ===========================================================================
def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, _ = gate('P5.11.1 self-consistent T0 build', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.11.1] ABORTED\n{error}')
        sys.exit(1)

    config = config_for(RHO_PRIMARY)
    print(f'[P5.11.1] cold build under {config.label} ...', flush=True)

    planning = O.fresh_planning('p511_selfconsistent_t0')
    rho_replaced = RH.apply_rho_to_params(planning, config.rho)
    adaptive_before = RH.set_adaptive_penalty(planning, config.adaptive_penalty)

    console = io.StringIO()
    with redirect_stdout(console):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        # the objectives are RESCALED as they are BUILT, not afterwards
        with R.patched_admm_objectives() as applied:
            _, _, _, _, _, state = planning.run_operational_planning(
                type='distributed', candidate_solution=deepcopy(candidate),
                print_results=False, debug_flag=False, return_state=True)

    scales = [v for v in applied.values() if v]
    observed = RH.observed_rho(state)
    distinct = [dict(t) for t in {tuple(sorted(v.items())) for v in observed.values()}]

    neutralize(state)
    digest = state_hash(state)
    with open(TEMPLATE_PATH, 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        'stage': 'P5.11.1 build', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'template_id': config.template_id,
        'state_sha256': digest,
        'file_sha256': hashlib.sha256(open(TEMPLATE_PATH, 'rb').read()).hexdigest(),
        'config': config.as_dict(), 'config_hash': config.config_hash,
        'blocks_rescaled_at_build': len(scales),
        'effective_scale_range': [min(scales), max(scales)] if scales else None,
        'rho_in_built_template': distinct,
        'rho_params_replaced': rho_replaced,
        'adaptive_penalty_before_override': adaptive_before,
        'history_channels_neutralized': list(OR.HISTORY_KEYS) + ['candidate_solution'],
        'construction': ('cold run_operational_planning(type=distributed) with '
                         'patched_admm_objectives active, so the augmented '
                         'objectives are RESCALED at construction rather than '
                         'retrofitted'),
    }
    with open(TEMPLATE_META, 'w') as handle:
        json.dump(meta, handle, indent=1, default=str)

    print(f'[P5.11.1] built. blocks rescaled at build: {len(scales)}')
    print(f'[P5.11.1] rho in built template: {distinct}')
    print(f'[P5.11.1] state sha256 : {digest}')
    print(f'[P5.11.1] template     -> {TEMPLATE_PATH}')
    print(f'[P5.11.1] metadata     -> {TEMPLATE_META}')


# ===========================================================================
#  compare
# ===========================================================================
def load_selfconsistent():
    with open(TEMPLATE_PATH, 'rb') as handle:
        return neutralize(pickle.load(handle))


def measure(record):
    cycles = record.get('cycle_detail') or []
    last = cycles[-1] if cycles else {}
    admm = record.get('admm') or {}
    return {
        'status': record.get('status'),
        'investment_cost': record.get('investment_cost'),
        'net_operational_recourse': record.get('net_operational_recourse'),
        'total_objective': record.get('total_objective'),
        'admm_cycles': admm.get('cycles'),
        'termination': record.get('terminating_criterion'),
        'rho_start': (record.get('config') or {}).get('rho_pf'),
        'rho_end': record.get('rho_observed_final'),
        'max_interface_disagreement_prepolish': last.get('primal_pf'),
        'polish_correction': record.get('admm_to_polish_improvement'),
        'local_block_failures': 0 if admm.get('status') is None else 'ADMM stage failed',
        'polish_failures': len(record.get('failed_blocks') or []),
        'failed_blocks': record.get('failed_blocks'),
        'wall_clock_s': record.get('wall_clock_s'),
    }


def compare(only=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.11.1 comparison', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.11.1] ABORTED\n{error}')
        sys.exit(1)
    if not os.path.exists(TEMPLATE_PATH):
        print('[P5.11.1] self-consistent template not built; run `build` first.')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    tag = (only or 'all').replace('|', '_').replace('%', 'pct').replace(' ', '_')
    out_path = os.path.join(OUT_DIR, f'p511_1_compare_{tag}.json')
    meta = json.load(open(TEMPLATE_META))

    report = {'stage': 'P5.11.1 compare', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'selfconsistent_template': {k: meta[k] for k in
                                          ('template_id', 'state_sha256',
                                           'file_sha256', 'config_hash')},
              'delta_tolerance': DELTA_TOLERANCE, 'rows': []}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for rho_pf in (RHO_PRIMARY, RHO_CROSSCHECK):
        config = config_for(rho_pf)
        for template_kind in ('inherited', 'self_consistent'):
            for name in CANDIDATES:
                if only is not None and name != only:
                    continue
                safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')
                case_id = f'p511_1_{template_kind}_pf{rho_pf:g}_{safe}'
                if template_kind == 'inherited':
                    inherited_config = OR.OracleConfig(
                        scaling_mode=OR.SCALING_RESCALED, rho_v=1.5,
                        rho_pf=rho_pf, rho_ess=1.0, adaptive_penalty=False,
                        neutralize_history=True,
                        notes='P5.11.1 inherited template')
                    state, _ = OR.prepare_template(inherited_config)
                    use_config = inherited_config
                else:
                    state = load_selfconsistent()
                    RH.apply_rho_to_state(state, config.rho)
                    use_config = config
                print(f'[P5.11.1] {template_kind:15s} rho_pf={rho_pf:g} {name} ...',
                      flush=True)
                record, _ = OR.evaluate(population[name], use_config,
                                        case_id=case_id, template_state=state)
                row = {'template': template_kind, 'rho_pf': rho_pf,
                       'candidate': name, 'config_hash': use_config.config_hash,
                       **measure(record)}
                report['rows'].append(row)
                persist()
                print(f"          status={row['status']} cycles={row['admm_cycles']} "
                      f"Q={row['total_objective']} "
                      f"polish_fail={row['polish_failures']} "
                      f"correction={row['polish_correction']}", flush=True)

    print(f'\n[P5.11.1] report -> {out_path}')


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if action == 'build':
        build()
    elif action == 'compare':
        compare(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(f'unknown action {action!r}')
        sys.exit(2)
