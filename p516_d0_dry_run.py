"""P5.16 D0 / T0.1a -- G-D0-0 builder dry run on FrozenSMOPF network snapshots and the s39_D ESSO pickle.

ZERO solves of any kind: `p513_solve_profile_guard.SolveProfileGuard` is armed in BLOCKING form (permitted=())
for the whole child process, before any production module is imported, and verified exactly at 0 at the end.

Governing spec: `P5_16_D0_SPEC_v2.md` sections 3, 4 (G-D0-0 / G-D0-2 / G-D0-3), 7. Builder:
`p516_d0_build_relaxation.py`.

Usage (one process, attached, alone):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_dry_run.py

The parent creates OUT_DIR atomically (refuses if it exists -- which also refuses a concurrent second copy),
runs the child with stdout AND stderr captured line by line into OUT_DIR/run.log (echoed to the console), then
writes OUT_DIR/manifest_sha256.json over every file in OUT_DIR, the log included.

Outputs (OUT_DIR only): predictions.json (written first, before any computation), inventory.json,
identities.json, name_map.json, *.lp, side_writes/ (production's diagram PDFs, redirected here), run.log,
manifest_sha256.json.

Mapped points evaluated: these are cycle-7 / terminal snapshot points, not the certified points; the identities
are exact at any point, which is what this dry run tests.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P516D0', 'dry_run')
# Rehearsal only: P516_D0_REHEARSAL_DIR=/tmp/... redirects every output to scratch outside the repository. Any other
# location is refused. The evidence run sets nothing and writes OUT_DIR above.
_REHEARSAL = os.environ.get('P516_D0_REHEARSAL_DIR')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516_D0_REHEARSAL_DIR must be under /tmp, got {_REHEARSAL!r}')
    OUT_DIR = os.path.abspath(_REHEARSAL)
LOG_PATH = os.path.join(OUT_DIR, 'run.log')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
SIDE_WRITES_DIR = os.path.join(OUT_DIR, 'side_writes', 'Diagrams')

FROZEN_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF')
# DSO candidates in order of preference (task: prefer 2025 Summer cycle 7); the first that unpickles is used and
# every attempt is recorded.
DSO_CANDIDATES = [
    os.path.join(FROZEN_DIR, 'frozen_DSO_node7_case33_2_2025_Summer_cycle7.pkl'),
    os.path.join(FROZEN_DIR, 'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
]
TSO_CANDIDATES = [
    os.path.join(FROZEN_DIR, 'matched_success_TSO_case9_2025_Summer_cycle7.pkl'),
]
ESSO_PICKLE = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run', 'esso_models_s39_D.pkl')
SRP1_DIR = os.path.join(REPO, 'data', 'SRP1')

THRESHOLDS = {
    'retained_abs': 1e-12,          # task gate
    'cone_rel': 1e-12,              # task gate
    'objective_rel': 1e-8,          # task gate
    'envelope_abs': 1e-9,           # spec v2 G-D0-3 (transformer envelope); also reported against 1e-12
    'stop_abs': 1e-10,              # task stop rule: identity failure beyond this is a construction error
}

PREDICTIONS = {
    'P-a': 'exactly five definitional families replaced on a network block',
    'P-b': 'retained-row identity achieved at <= 1e-13',
    'P-c': 'cone residual <= 1e-15',
    'P-d': 'objective identity <= 1e-12',
    'P-e': 'SRP1 networks contain transformer branches (case9 generator step-ups and/or the DN interface); '
           'count stated',
}

# Capture-path checklist: every quantity the task names -> (output file, top-level key). Asserted before any
# computation (each file has a writer) and again before writing (each key filled).
REQUIRED_CAPTURES = {
    'input sha256 and unpickle status (all FrozenSMOPF pickles, ESSO pickle)': ('inventory.json', 'inputs'),
    'row inventory per block (class / sub / family / form, counts)': ('inventory.json', 'blocks'),
    'definitional families by name and count': ('inventory.json', 'blocks'),
    'transformer branch count': ('inventory.json', 'transformers'),
    'capacity params carried and set (both modes)': ('identities.json', 'network'),
    'per-row residual tables (retained, envelope, cones)': ('identities.json', 'network'),
    'ESSO dropped rows, arguments, slack values': ('identities.json', 'esso'),
    'objective identity per block and summed': ('identities.json', 'objective'),
    'gate verdicts and achieved values': ('identities.json', 'gates'),
    'predictions vs outcomes': ('identities.json', 'predictions'),
    'guard counts (zero solves)': ('identities.json', 'guard'),
    'input non-mutation fingerprints': ('identities.json', 'fingerprints'),
    'write containment': ('identities.json', 'containment'),
    'LP files with sha256 and name maps': ('name_map.json', 'lp_files'),
    'provenance (interpreter, versions, git head, file hashes)': ('inventory.json', 'provenance'),
}
OUTPUT_FILES = ('predictions.json', 'inventory.json', 'identities.json', 'name_map.json')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, REPO)


# ======================================================================================================================
#  Parent
# ======================================================================================================================
def parent():
    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    if _REHEARSAL:
        print(f'[REHEARSAL] outputs to scratch {OUT_DIR} (not evidence)')
    try:
        os.mkdir(OUT_DIR)                       # atomic: refuses an existing run and a concurrent copy
    except FileExistsError:
        print(f'[REFUSED] {rel(OUT_DIR)} exists; refusing to run (no overwrite, no concurrent copy).')
        return 4
    started = time.time()
    with open(LOG_PATH, 'w', buffering=1) as log:
        log.write(f'# p516_d0_dry_run parent start {time.strftime("%Y-%m-%dT%H:%M:%S%z")} '
                  f'interpreter {sys.executable}\n')
        child = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child'], cwd=REPO,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = child.wait()
        log.write(f'# child exit code {code}; wall {time.time() - started:.1f} s\n')
    manifest = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in sorted(files):
            path = os.path.join(root, name)
            if path == MANIFEST_PATH:
                continue
            manifest[os.path.relpath(path, OUT_DIR)] = {'sha256': sha256_file(path), 'bytes': os.path.getsize(path)}
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump({'root': rel(OUT_DIR), 'child_exit_code': code, 'files': dict(sorted(manifest.items()))},
                  handle, indent=1)
    print(f'[parent] child exit code {code}; manifest {len(manifest)} files -> {rel(MANIFEST_PATH)}')
    return code


# ======================================================================================================================
#  Child
# ======================================================================================================================
def child():
    sys.path.insert(0, REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.16 D0 T0.1a builder dry run (zero solves)').install()
    run_marker = time.time()

    import gate_result_io as gio

    def write_json(name, payload, compact=False):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite {path}')
        with open(path, 'w') as handle:
            if compact:
                gio.dump(payload, handle, default=gio.json_default_item, separators=(',', ':'))
            else:
                gio.dump(payload, handle, default=gio.json_default_item, indent=1)
        print(f'[write] {rel(path)}')

    # ---- predictions first, before any computation ------------------------------------------------------------
    write_json('predictions.json', {'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                    'predictions': PREDICTIONS, 'thresholds': THRESHOLDS})

    # ---- capture-path assertion (fail fast) ---------------------------------------------------------------------
    writers = {'inventory.json': write_json, 'identities.json': write_json, 'name_map.json': write_json,
               'predictions.json': write_json}
    missing = sorted({f for f, _k in REQUIRED_CAPTURES.values() if f not in writers or f not in OUTPUT_FILES})
    if missing:
        raise RuntimeError(f'capture-path assertion failed: no writer for {missing}')
    print(f'[capture] {len(REQUIRED_CAPTURES)} required quantities -> writers present for '
          f'{sorted({f for f, _ in REQUIRED_CAPTURES.values()})}')

    import pickle
    import platform
    from contextlib import redirect_stdout

    import numpy as np
    import pyomo
    import pyomo.environ as pe

    import p516_d0_build_relaxation as B
    import shared_resources_planning as srp
    from shared_resources_planning import SharedResourcesPlanning

    # ---- provenance ---------------------------------------------------------------------------------------------
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception as error:          # recorded, never fatal
            return f'ERROR {error!r}'
    tracked_status_before = git('status', '--porcelain', '--untracked-files=no')
    files_hashed = ['p516_d0_dry_run.py', 'p516_d0_build_relaxation.py', 'model_construction_helpers.py',
                    'network.py', 'shared_resources_planning.py', 'shared_energy_storage_data.py',
                    'p513_solve_profile_guard.py', 'gate_result_io.py', 'P5_16_D0_SPEC_v2.md']
    provenance = {
        'interpreter': sys.executable, 'python': sys.version, 'platform': platform.platform(),
        'pyomo': pyomo.version.version, 'numpy': np.__version__,
        'git_head': git('rev-parse', 'HEAD'), 'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'tracked_status_before': tracked_status_before,
        'file_sha256': {f: sha256_file(os.path.join(REPO, f)) for f in files_hashed},
        'out_dir': rel(OUT_DIR),
    }

    # ---- input scan: every FrozenSMOPF pickle, unpickle attempt, sha256 -------------------------------------------
    scan = []
    for path in sorted(glob.glob(os.path.join(FROZEN_DIR, '**', '*.pkl'), recursive=True)):
        entry = {'path': rel(path), 'bytes': os.path.getsize(path), 'sha256': sha256_file(path)}
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            entry['unpickled'] = True
            entry['metadata'] = payload.get('metadata') if isinstance(payload, dict) else None
            del payload
        except Exception as error:
            entry['unpickled'] = False
            entry['error'] = repr(error)
        scan.append(entry)
    print(f'[inputs] FrozenSMOPF pickles: {len(scan)}; unpickled {sum(e["unpickled"] for e in scan)}; '
          f'failed {sum(not e["unpickled"] for e in scan)}')

    def load_first(candidates, label):
        attempts = []
        for path in candidates:
            if not os.path.exists(path):
                attempts.append({'path': rel(path), 'exists': False})
                continue
            digest = sha256_file(path)
            try:
                with open(path, 'rb') as handle:
                    payload = pickle.load(handle)
            except Exception as error:
                attempts.append({'path': rel(path), 'exists': True, 'sha256': digest, 'unpickled': False,
                                 'error': repr(error)})
                continue
            if sha256_file(path) != digest:
                raise RuntimeError(f'{path} changed while loading')
            attempts.append({'path': rel(path), 'exists': True, 'sha256': digest, 'unpickled': True,
                             'metadata': payload['metadata']})
            print(f'[inputs] {label}: using {rel(path)} sha256 {digest}')
            return payload, attempts
        raise RuntimeError(f'no {label} candidate unpickles: {attempts}')

    dso_payload, dso_attempts = load_first(DSO_CANDIDATES, 'DSO')
    tso_payload, tso_attempts = load_first(TSO_CANDIDATES, 'TSO')
    esso_sha = sha256_file(ESSO_PICKLE)
    with open(ESSO_PICKLE, 'rb') as handle:
        esso_models = pickle.load(handle)
    if sha256_file(ESSO_PICKLE) != esso_sha:
        raise RuntimeError('ESSO pickle changed while loading')
    print(f'[inputs] ESSO: {rel(ESSO_PICKLE)} sha256 {esso_sha}; nodes {sorted(esso_models)}')
    inputs = {'frozen_smopf_scan': scan, 'dso_attempts': dso_attempts, 'tso_attempts': tso_attempts,
              'esso': {'path': rel(ESSO_PICKLE), 'sha256': esso_sha, 'nodes': sorted(esso_models)}}

    # ---- block weights: production `_get_admm_block_weight` on the SRP1 case (diagram writes redirected) ---------
    os.makedirs(SIDE_WRITES_DIR)
    planning = SharedResourcesPlanning(SRP1_DIR, 'SRP1.json')
    planning.diagrams_dir = SIDE_WRITES_DIR
    planning.read_planning_problem()
    scenario_checksum = getattr(planning, 'scenario_metadata', {}).get('combined_scenario_checksum') \
        if isinstance(getattr(planning, 'scenario_metadata', None), dict) else None
    provenance['srp1_scenario_checksum'] = scenario_checksum

    blocks_spec = [
        {'label': 'DSO', 'payload': dso_payload,
         'network_data': planning.distribution_networks[dso_payload['metadata']['node_id']]},
        {'label': 'TSO', 'payload': tso_payload, 'network_data': planning.transmission_network},
    ]

    inventory_out = {'blocks': {}, 'transformers': {}}
    network_out = {}
    objective_out = {'per_block': {}, 'summed': {}}
    fingerprints = {}
    name_maps = {'lp_files': {}}
    gate_rows = []          # (gate, block, mode, achieved, threshold, pass)
    relaxed_for_sum = {'rated': [], 'available': []}
    originals_for_sum = []

    def gate(name, block, mode, achieved, threshold, rule='<='):
        ok = bool(achieved <= threshold)
        gate_rows.append({'gate': name, 'block': block, 'mode': mode, 'achieved': float(achieved),
                          'threshold': threshold, 'pass': ok})
        print(f'[gate] {name:34s} {block:10s} {mode:9s} achieved {achieved:.3e} <= {threshold:.0e}: '
              f'{"PASS" if ok else "FAIL"}')
        return ok

    # ---- network blocks -------------------------------------------------------------------------------------------
    for spec in blocks_spec:
        label, model = spec['label'], spec['payload']['model']
        metadata = spec['payload']['metadata']
        network, params = B.network_and_params(model)
        year, day = network.year, network.day
        weight = srp._get_admm_block_weight(spec['network_data'], year, day)
        fixture_weight = float(pe.value(model.admm_block_weight)) if hasattr(model, 'admm_block_weight') else None
        fp_before = B.model_fingerprint(model)

        inv = B.inventory_rows(model, 'network')
        inventory_out['blocks'][label] = {'metadata': metadata, **inv}
        inventory_out['transformers'][label] = {
            'network': network.name, 'active_transformer_branches': inv['active_transformer_branches'],
            'n_transformer_branches_any_status': inv['n_transformer_branches_any_status'],
            'branches': [{'branch': b, 'fbus': network.branches[b].fbus, 'tbus': network.branches[b].tbus,
                          'ratio': network.branches[b].ratio, 'vmag_reg': bool(network.branches[b].vmag_reg),
                          'rate': network.branches[b].rate}
                         for b in inv['active_transformer_branches']]}
        print(f'[inventory] {label} {network.name} {year} {day}: {inv["n_active_rows"]} active rows')
        for c in inv['counts']:
            if c['class'] != 'retained':
                print(f'            {c["class"]:9s} {str(c["sub"]):17s} {c["family"]:28s} {c["rows"]}')

        rated = {e: {'s': float(s.s), 'e': float(s.e)} for e, s in enumerate(network.shared_energy_storages)}
        network_out[label] = {'network': network.name, 'year': year, 'day': day, 'metadata': metadata,
                              'block_weight_production': weight, 'block_weight_fixture_param': fixture_weight,
                              'rated_capacity_source': 'network.shared_energy_storages[e].{s,e} carried by the '
                                                       'fixture (pu)', 'rated_capacity': rated, 'modes': {}}
        replaced_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'replaced'}
        retained_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'retained'}

        for mode in ('rated', 'available'):
            relaxed, record = B.relax_network_block(model, mode, rated if mode == 'rated' else None)
            legacy = B.attach_lb_objective(relaxed, weight, params)

            # inventory <-> builder consistency: replaced == active in input but not in relaxed (production rows)
            inactive_now, active_both = set(), set()
            for component in model.component_objects(pe.Constraint, active=True):
                rcomp = relaxed.find_component(component.name)
                for index, cd in component.items():
                    if cd.active:
                        (active_both if rcomp[index].active else inactive_now).add((component.name, repr(index)))
            consistency = {'replaced_equals_deactivated': replaced_inv == inactive_now,
                           'retained_equals_still_active': retained_inv == active_both,
                           'n_replaced': len(replaced_inv), 'n_deactivated': len(inactive_now),
                           'n_retained': len(retained_inv), 'n_still_active': len(active_both)}

            tables = B.mapped_point(model, relaxed, w_source='ef')
            same_point = B.production_rows_at_point(model, relaxed)

            ret = tables['retained']
            gated = [r for r in ret if not (mode == 'rated' and r['references_capacity_params'])]
            excluded = [r for r in ret if mode == 'rated' and r['references_capacity_params']]
            worst = max(gated, key=lambda r: r['delta'])
            ratio_to_resolution = max((r['delta'] / r['eval_resolution']) if r['eval_resolution'] > 0
                                      else (0.0 if r['delta'] == 0 else float('inf')) for r in gated)
            exceed_bound = [r for r in gated
                            if r['delta'] > r['first_order_w_bound'] + 4.0 * r['eval_resolution']]
            cones_w = [abs(c['rel_residual']) for c in tables['cones'] if c['kind'] == 'W']
            cones_t = [abs(c['rel_residual']) for c in tables['cones'] if c['kind'] == 'transformer_UCD']
            env = tables['envelope']
            env_max = max((r['delta'] for r in env), default=0.0)
            bound_min = min((min(b['lower_slack_rel'], b['upper_slack_rel']) for b in tables['envelope_bounds']),
                            default=None)

            # diagnostic mapped point M0 (W kept at the snapshot's own values) on a fresh relaxation
            relaxed_m0, _ = B.relax_network_block(model, mode, rated if mode == 'rated' else None)
            tables_m0 = B.mapped_point(model, relaxed_m0, w_source='snapshot')
            m0_gated = [r for r in tables_m0['retained'] if not (mode == 'rated' and r['references_capacity_params'])]
            m0 = {'retained_max_delta': max(r['delta'] for r in m0_gated),
                  'envelope_max_delta': max((r['delta'] for r in tables_m0['envelope']), default=0.0),
                  'cone_W_max_rel': max(abs(c['rel_residual']) for c in tables_m0['cones'] if c['kind'] == 'W'),
                  'cone_transformer_max_rel': max((abs(c['rel_residual']) for c in tables_m0['cones']
                                                   if c['kind'] == 'transformer_UCD'), default=None)}
            del relaxed_m0, tables_m0

            # objective identity: same component functions on the relaxed block (mapped) and the original
            expr_rel, legacy_rel = B.block_gross_expr(relaxed, params)
            expr_org, legacy_org = B.block_gross_expr(model, params)
            v_rel, v_org = float(pe.value(expr_rel)), float(pe.value(expr_org))
            obj_rel_err = abs(v_rel - v_org) / abs(v_org) if v_org != 0 else abs(v_rel - v_org)
            cross = {}
            try:
                cross['network.get_primal_value'] = float(network.get_primal_value(model, params))
            except Exception as error:
                cross['network.get_primal_value'] = f'raised {error!r}'
            try:
                cross['srp._get_local_interface_settlement_contracted'] = float(
                    srp._get_local_interface_settlement(model, part='contracted'))
            except Exception as error:
                cross['srp._get_local_interface_settlement_contracted'] = f'raised {error!r}'
            cross['srp._get_local_voltage_pin'] = float(srp._get_local_voltage_pin(model))
            cross['fixture_objective_value'] = float(pe.value(model.objective))
            cross['fixture_objective_minus_rule'] = float(pe.value(model.objective)) - v_org
            if hasattr(model, 'scenario_deviation_penalty'):
                cross['fixture_scenario_deviation_weight_x_penalty'] = float(
                    pe.value(model.scenario_deviation_weight) * pe.value(model.scenario_deviation_penalty))
            lb_value_rel = float(pe.value(relaxed.d0_lb_objective))
            objective_out['per_block'][f'{label}_{mode}'] = {
                'weight': weight, 'gross_unweighted_relaxed_at_mapped': v_rel,
                'gross_unweighted_original_at_snapshot': v_org, 'relative_error': obj_rel_err,
                'weighted_relaxed': lb_value_rel, 'weighted_original': weight * v_org,
                'pre_settlement_fixture_view_used': {'relaxed': legacy_rel, 'original': legacy_org,
                                                     'attach': legacy},
                'production_cross_checks_on_original': cross,
                'lb_objective_form': B.objective_form(relaxed.d0_lb_objective.expr)}
            relaxed_for_sum[mode].append({'model': relaxed, 'params': params, 'network_data': spec['network_data'],
                                          'year': year, 'day': day, 'label': label})
            if mode == 'rated':
                originals_for_sum.append({'model': model, 'params': params, 'network_data': spec['network_data'],
                                          'year': year, 'day': day, 'label': label})

            # export
            lp_name = f'{label.lower()}_{network.name}_{year}_{day}_c{metadata["cycle"]}_{mode}.lp'
            lp_path = os.path.join(OUT_DIR, lp_name)
            if os.path.exists(lp_path):
                raise RuntimeError(f'refusing to overwrite {lp_path}')
            lp_sha, nm, nm_counts = B.export_lp(relaxed, lp_path)
            name_maps['lp_files'][lp_name] = {'sha256': lp_sha, 'counts': nm_counts, 'name_map': nm}
            print(f'[export] {lp_name} sha256 {lp_sha[:16]}... {nm_counts}')

            network_out[label]['modes'][mode] = {
                'build_record': record, 'inventory_builder_consistency': consistency,
                'retained': {'n_rows': len(ret), 'n_gated': len(gated),
                             'n_excluded_capacity_param_rows': len(excluded),
                             'excluded_capacity_param_rows_max_delta': max((r['delta'] for r in excluded),
                                                                           default=0.0),
                             'excluded_capacity_param_rows': excluded,
                             'max_delta': worst['delta'], 'argmax': {'family': worst['family'],
                                                                     'index': worst['index'],
                                                                     'eval_resolution': worst['eval_resolution'],
                                                                     'first_order_w_bound':
                                                                         worst['first_order_w_bound']},
                             'max_delta_by_family': {},
                             'max_delta_over_eval_resolution': ratio_to_resolution,
                             'n_rows_delta_above_first_order_bound_plus_4_resolution': len(exceed_bound),
                             'same_point_production_vs_relaxed': same_point},
                'envelope': {'n_rows': len(env), 'max_delta': env_max,
                             'envelope_bounds_min_rel_slack': bound_min},
                'cones': {'n_W': len(cones_w), 'max_rel_W': max(cones_w) if cones_w else None,
                          'n_transformer': len(cones_t), 'max_rel_transformer': max(cones_t) if cones_t else None},
                'diagnostic_m0_snapshot_W': m0,
                'tables': {'retained': ret, 'envelope': env, 'cones': tables['cones'],
                           'envelope_bounds': tables['envelope_bounds'],
                           'replaced_production_at_snapshot': tables['replaced_production']},
            }
            fam = network_out[label]['modes'][mode]['retained']['max_delta_by_family']
            for r in gated:
                fam[r['family']] = max(fam.get(r['family'], 0.0), r['delta'])

            tag = f'{label}'
            gate('G-D0-3 retained |delta| (task 1e-12)', tag, mode, worst['delta'], THRESHOLDS['retained_abs'])
            gate('G-D0-3 cone rel (W)', tag, mode, max(cones_w), THRESHOLDS['cone_rel'])
            if cones_t:
                gate('G-D0-3 cone rel (transformer UCD)', tag, mode, max(cones_t), THRESHOLDS['cone_rel'])
            if env:
                gate('G-D0-3 envelope rewrite (spec 1e-9)', tag, mode, env_max, THRESHOLDS['envelope_abs'])
            gate('G-D0-3 objective rel (1e-8)', tag, mode, obj_rel_err, THRESHOLDS['objective_rel'])
            gate('same-point production==relaxed', tag, mode, same_point['max_abs_delta'], 0.0)
            gate('G-D0-2 inventory==builder (0 = consistent)', tag, mode,
                 0.0 if (consistency['replaced_equals_deactivated'] and consistency['retained_equals_still_active'])
                 else 1.0, 0.0)

        fp_after = B.model_fingerprint(model)
        fingerprints[label] = {'before': fp_before, 'after': fp_after, 'unchanged': fp_before == fp_after}
        gate('input not mutated (0 = unchanged)', label, '-', 0.0 if fp_before == fp_after else 1.0, 0.0)

    # summed LB objective through lb_objective (both network blocks; they come from different runs -- the sum
    # exercises the function, it is not a cell value)
    for mode in ('rated', 'available'):
        total_rel, per_rel = B.lb_objective(relaxed_for_sum[mode])
        total_org, _ = B.lb_objective(originals_for_sum)
        v_r, v_o = float(pe.value(total_rel)), float(pe.value(total_org))
        objective_out['summed'][mode] = {'blocks': [p['label'] for p in per_rel],
                                         'weights': [p['weight'] for p in per_rel],
                                         'relaxed_at_mapped': v_r, 'original_at_snapshot': v_o,
                                         'relative_error': abs(v_r - v_o) / abs(v_o)}
        gate('G-D0-3 objective rel summed', 'TSO+DSO', mode, abs(v_r - v_o) / abs(v_o), THRESHOLDS['objective_rel'])

    # ---- ESSO -----------------------------------------------------------------------------------------------------
    esso_out = {}
    allowed_drops = set(B.ESSO_DROP_ARGUMENTS)
    for node_id in sorted(esso_models):
        em = esso_models[node_id]
        label = f'ESSO_{node_id}'
        fp_before = B.model_fingerprint(em)
        inv = B.inventory_rows(em, 'esso')
        inventory_out['blocks'][label] = inv
        relaxed, record = B.relax_esso_block(em)
        dropped_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'dropped'}
        retained_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'retained'}
        inactive_now, active_both = set(), set()
        for component in em.component_objects(pe.Constraint, active=True):
            rcomp = relaxed.find_component(component.name)
            for index, cd in component.items():
                if cd.active:
                    (active_both if rcomp[index].active else inactive_now).add((component.name, repr(index)))
        consistency = {'dropped_equals_deactivated': dropped_inv == inactive_now,
                       'retained_equals_still_active': retained_inv == active_both,
                       'n_dropped': len(dropped_inv), 'n_retained': len(retained_inv)}
        tables = B.mapped_point(em, relaxed)
        ret = tables['retained']
        worst = max(ret, key=lambda r: r['delta'])
        subs = {r['sub'] for r in inv['rows'] if r['class'] == 'dropped'}
        lp_name = f'esso_node{node_id}_s39_D_terminal.lp'
        lp_path = os.path.join(OUT_DIR, lp_name)
        if os.path.exists(lp_path):
            raise RuntimeError(f'refusing to overwrite {lp_path}')
        lp_sha, nm, nm_counts = B.export_lp(relaxed, lp_path)
        name_maps['lp_files'][lp_name] = {'sha256': lp_sha, 'counts': nm_counts, 'name_map': nm}
        print(f'[export] {lp_name} sha256 {lp_sha[:16]}... {nm_counts}')
        cohorts = {
            'es_s_investment_fixed': {str(k): float(pe.value(v)) for k, v in em.es_s_investment_fixed.items()},
            'es_e_investment_fixed': {str(k): float(pe.value(v)) for k, v in em.es_e_investment_fixed.items()},
            'es_s_rated': {str(k): v.value for k, v in em.es_s_rated.items()},
            'es_e_rated': {str(k): v.value for k, v in em.es_e_rated.items()},
            'es_e_available_per_unit_snapshot': {str(k): v.value for k, v in em.es_e_available_per_unit.items()},
            'es_soh_per_unit_cumul_snapshot': {str(k): v.value for k, v in em.es_soh_per_unit_cumul.items()},
        }
        esso_out[label] = {'build_record': record, 'inventory_builder_consistency': consistency,
                           'dropped_subs': sorted(subs), 'cohort_state': cohorts,
                           'retained': {'n_rows': len(ret), 'max_delta': worst['delta'],
                                        'argmax': {'family': worst['family'], 'index': worst['index']}},
                           'tables': {'retained': ret, 'added_available_eq_rated': tables['added_esso'],
                                      'replaced_production_at_snapshot': tables['replaced_production']}}
        fp_after = B.model_fingerprint(em)
        fingerprints[label] = {'before': fp_before, 'after': fp_after, 'unchanged': fp_before == fp_after}
        gate('G-D0-3 retained |delta| (task 1e-12)', label, '-', worst['delta'], THRESHOLDS['retained_abs'])
        gate('ESSO dropped subs within allowed (0 = yes)', label, '-', 0.0 if subs <= allowed_drops else 1.0, 0.0)
        gate('G-D0-2 inventory==builder (0 = consistent)', label, '-',
             0.0 if (consistency['dropped_equals_deactivated'] and consistency['retained_equals_still_active'])
             else 1.0, 0.0)
        gate('input not mutated (0 = unchanged)', label, '-', 0.0 if fp_before == fp_after else 1.0, 0.0)

    # ---- guard, solver modules, containment ------------------------------------------------------------------------
    guard_failures = guard.verify(0)
    guard.uninstall()
    # solver LIBRARIES actually imported (top-level packages), separately from Pyomo's solver-plugin modules, which
    # `import pyomo.environ` registers on every run without loading any solver library
    solver_libraries = {name: any(m == name or m.startswith(name + '.') for m in sys.modules)
                        for name in ('gurobipy', 'highspy', 'clarabel', 'cplex', 'docplex', 'xpress', 'mosek')}
    solver_modules = solver_libraries
    pyomo_plugin_modules = sorted(m for m in sys.modules if m.startswith(('pyomo.solvers.plugins.solvers.',
                                                                          'pyomo.contrib.appsi.solvers',
                                                                          'pyomo.contrib.solver.solvers')))
    guard_out = {'label': guard.label, 'permitted': list(guard.permitted), 'counts': dict(guard.counts),
                 'declared_solves': 0, 'verify_failures': guard_failures,
                 'solver_libraries_imported': solver_libraries,
                 'pyomo_solver_plugin_modules_registered': len(pyomo_plugin_modules)}
    gate('guard: zero solves (failures count)', 'process', '-', float(len(guard_failures)), 0.0)

    written_outside = []
    for root, dirs, files in os.walk(REPO):
        if os.path.abspath(root).startswith(OUT_DIR) or os.path.basename(root) in ('.git', '__pycache__'):
            dirs[:] = []
            continue
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.getmtime(path) >= run_marker:
                    written_outside.append(rel(path))
            except OSError:
                pass
    tracked_status_after = git('status', '--porcelain', '--untracked-files=no')
    containment = {'files_under_repo_modified_outside_out_dir': written_outside,
                   'tracked_status_before': tracked_status_before, 'tracked_status_after': tracked_status_after,
                   'tracked_status_unchanged': tracked_status_before == tracked_status_after,
                   'side_writes_redirected_to': rel(SIDE_WRITES_DIR),
                   'n_side_write_files': sum(len(f) for _r, _d, f in os.walk(SIDE_WRITES_DIR))}
    gate('containment: writes outside OUT_DIR', 'process', '-', float(len(written_outside)), 0.0)

    # ---- predictions vs outcomes -------------------------------------------------------------------------------------
    def definitional_with_rows(label):
        counts = inventory_out['blocks'][label]['counts']
        return sorted({c['family'] for c in counts if c['family'] in B.DEFINITIONAL_FAMILIES and c['rows'] > 0})

    def max_over(key_fn):
        values = [key_fn(network_out[l]['modes'][m]) for l in network_out for m in ('rated', 'available')]
        return max(v for v in values if v is not None)

    retained_all = max([max_over(lambda d: d['retained']['max_delta'])]
                       + [esso_out[l]['retained']['max_delta'] for l in esso_out])
    cone_all = max_over(lambda d: max(x for x in (d['cones']['max_rel_W'], d['cones']['max_rel_transformer'])
                                      if x is not None))
    obj_all = max(v['relative_error'] for v in objective_out['per_block'].values())
    envelope_replaced = {l: network_out[l]['modes']['available']['build_record']['envelope_rewrite_rows_by_family']
                         for l in network_out}
    transformer_counts = {l: len(inventory_out['transformers'][l]['active_transformer_branches'])
                          for l in inventory_out['transformers']}
    predictions_out = {
        'P-a': {'prediction': PREDICTIONS['P-a'],
                'outcome': {l: {'definitional_families_with_rows': definitional_with_rows(l),
                                'n': len(definitional_with_rows(l)),
                                'also_replaced_by_envelope_rewrite': envelope_replaced[l]} for l in network_out},
                'held': all(len(definitional_with_rows(l)) == 5 and not envelope_replaced[l] for l in network_out)},
        'P-b': {'prediction': PREDICTIONS['P-b'], 'outcome_max_abs_delta': retained_all,
                'held': bool(retained_all <= 1e-13)},
        'P-c': {'prediction': PREDICTIONS['P-c'], 'outcome_max_rel': cone_all, 'held': bool(cone_all <= 1e-15)},
        'P-d': {'prediction': PREDICTIONS['P-d'], 'outcome_max_rel': obj_all, 'held': bool(obj_all <= 1e-12)},
        'P-e': {'prediction': PREDICTIONS['P-e'], 'outcome_active_transformers_in_loaded_blocks': transformer_counts,
                'held': any(n > 0 for n in transformer_counts.values())},
    }
    stop = any(r['achieved'] > THRESHOLDS['stop_abs'] for r in gate_rows
               if r['gate'].startswith('G-D0-3') and 'objective' not in r['gate'])
    gates_out = {'rows': gate_rows, 'all_pass': all(r['pass'] for r in gate_rows),
                 'failed': [r for r in gate_rows if not r['pass']],
                 'stop_rule_triggered_identity_beyond_1e-10': stop, 'thresholds': THRESHOLDS}

    inventory_payload = {'provenance': provenance, 'inputs': inputs, **inventory_out}
    identities_payload = {'network': network_out, 'esso': esso_out, 'objective': objective_out,
                          'gates': gates_out, 'predictions': predictions_out, 'guard': guard_out,
                          'fingerprints': fingerprints, 'containment': containment}
    payloads = {'inventory.json': inventory_payload, 'identities.json': identities_payload,
                'name_map.json': name_maps}
    unfilled = [(q, f, k) for q, (f, k) in REQUIRED_CAPTURES.items() if payloads.get(f, {}).get(k) is None]
    if unfilled:
        raise RuntimeError(f'capture check: unfilled required quantities {unfilled}')
    write_json('inventory.json', inventory_payload, compact=True)
    write_json('identities.json', identities_payload, compact=True)
    write_json('name_map.json', name_maps, compact=True)

    print(f'[summary] gates all pass: {gates_out["all_pass"]}; failed: '
          f'{[(r["gate"], r["block"], r["mode"], r["achieved"]) for r in gates_out["failed"]]}')
    print(f'[summary] stop rule (identity beyond 1e-10): {stop}')
    for key, value in predictions_out.items():
        print(f'[prediction] {key}: held={value["held"]}')
    print(f'[summary] guard counts {guard.counts}; solver libraries imported: {solver_modules}')
    if stop:
        return 3
    return 0 if gates_out['all_pass'] else 2


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        sys.exit(child())
    sys.exit(parent())
