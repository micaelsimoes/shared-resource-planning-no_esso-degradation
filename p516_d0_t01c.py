"""P5.16 D0 / T0.1c -- variants R, A, C on the route-A certified blocks with the spec v2 section 12 rulings applied
(ZERO solves).

Governing spec: `P5_16_D0_SPEC_v2.md` sections 2 (cells, hashes), 3 (construction), 11 (Addendum 88 D-1 ... D-5) and
section 12 (Planner rulings after T0.1b), which define this task:
  12.1  retained-row gate restated: |Delta_row| <= first-order W bound (|d row / d W| * |dW|, dW = the certified
        definitional residual, reported per block as the MAPPING OFFSET) + 4 eps sum|terms|; the same-point Delta = 0
        stays the primary construction test; cones <= 1e-12 relative at the exact rank-1 point;
  12.2  the same first-order statement for the envelope-rewrite rows (preserved here, inside the evidence root);
  12.3  variant C: cohort y_inv, block y >= y_inv, age_end(y) = sum_{k = y_inv..y} num_years_k (case-file Years map),
        E_C = E_rated * phi_cal^age_end, phi_cal = `calendar_retention_per_year` (ESS params file), S unchanged;
        ESSO available rows `es_e_available_per_unit == phi^age_end * es_e_rated_per_unit`; network copies'
        `shared_es_e_rated_fixed` = E_C / s_base;
  12.4  capacity-derived Var bounds follow the variant (R and C reset; A unchanged -- an attribution diagnostic, never
        a bound, D-1): the bounds are inventoried by probing the PRODUCTION function
        `configure_shared_ess_operational_state` (MCH l.1117) on a scratch clone, and for R / C read off the same
        production function called with the variant capacity on a scratch clone, then copied onto the relaxed clone
        (`relax_network_block(..., derived_bounds=...)`); the arithmetic e * ESS_DAY_BALANCE_SLACK_FRACTION +
        EQUALITY_TOLERANCE is checked against it;
  12.5  Q_cert+ = gross with every slack-penalty (detector) term re-evaluated at max(slack, 0), per block, group, cell.

Builder: `p516_d0_build_relaxation.py` (T0.1c opt-in extension; defaults unchanged). Constants and the cell table are
imported from `p516_d0_t01b.py`. T0.1b evidence (`P516D0/t01b/`, manifest sha256 4e5bbd45...) is reused BY REFERENCE:
its manifest is hash-checked, the captures it holds are referenced by hash (not duplicated) and the live blocks'
fingerprints are gated equal to the ones T0.1b recorded before mutation.

ZERO solves: blocking `p513_solve_profile_guard.SolveProfileGuard` (permitted=()) armed before any production import,
verified 0 / 0 at the end; a `sys.addaudithook` hook installed first RAISES on any process launch and records every
write-open; git provenance taken by the parent before and after.

Usage (one process, attached, alone):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_t01c.py

The parent creates OUT_DIR atomically (refuses if it exists, which also refuses a concurrent copy), runs the child
with stdout AND stderr merged into OUT_DIR/run.log, then writes OUT_DIR/manifest_sha256.json over every file in OUT_DIR.
Exit codes: 0 all gates pass; 2 a gate failed (no stop); 3 a stop rule fired; 4 refused (root exists); 5 refused
(an input hash mismatched).
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import p516_d0_t01b as T1B

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P516D0', 't01c')
# Rehearsal only: P516_D0_T01C_REHEARSAL_DIR=/tmp/... redirects every output to scratch outside the repository, and
# P516_D0_T01C_REHEARSAL_GROUPS=k limits each cell to its first k groups. The evidence run sets neither.
_REHEARSAL = os.environ.get('P516_D0_T01C_REHEARSAL_DIR')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516_D0_T01C_REHEARSAL_DIR must be under /tmp, got {_REHEARSAL!r}')
    OUT_DIR = os.path.abspath(_REHEARSAL)
_GROUP_LIMIT = os.environ.get('P516_D0_T01C_REHEARSAL_GROUPS')
if _GROUP_LIMIT and not _REHEARSAL:
    raise SystemExit('P516_D0_T01C_REHEARSAL_GROUPS is a rehearsal-only setting')
GROUP_LIMIT = int(_GROUP_LIMIT) if _GROUP_LIMIT else None

LOG_PATH = os.path.join(OUT_DIR, 'run.log')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
PROVENANCE_PARENT = os.path.join(OUT_DIR, 'provenance_parent.json')
SIDE_WRITES_DIR = os.path.join(OUT_DIR, 'side_writes', 'Diagrams')
SRP1_DIR = os.path.join(REPO, 'data', 'SRP1')
CASE_FILE = os.path.join('data', 'SRP1', 'SRP1.json')
ESS_PARAMS_FILE = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')

T01B_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01b')
T01B_MANIFEST_SHA256 = '4e5bbd45234598094593a8daca807a576590eee52f492f5281b63cf621876562'
T01B_FILES_READ = ('identities.json', 'coupling.json', 'lp/index.json')

CELLS = T1B.CELLS
RECORD_FILES = T1B.RECORD_FILES
MODES = ('rated', 'available', 'calendar')
VARIANT = {'rated': 'R', 'available': 'A', 'calendar': 'C'}
VARIANT_SEMANTICS = {
    'R': ('rated: network copies carry rated capacity, and every capacity-derived Var bound is reset to it '
          '(spec section 3 item 3, section 12 ruling 4); the section 3 lower-bound construction'),
    'A': ('available: network copies keep the certified available capacity and its bounds; an ATTRIBUTION '
          'DIAGNOSTIC, NOT a lower bound of production (spec section 11 D-1). The Pyomo objective name '
          'd0_lb_objective is the builder\'s export convention and does not make A a bound.'),
    'C': ('calendar-only: network copies carry E_C = E_rated * phi_cal^age_end (S rated) and every capacity-derived '
          'Var bound is reset to it; ESSO available rows es_e_available == phi^age_end * es_e_rated '
          '(spec section 11 D-2, section 12 ruling 3); the Phase-2 object'),
}

THRESHOLDS = {
    'same_point_abs': 0.0,                  # section 12.1: same-point Delta == 0 exactly (primary)
    'restated_resolution_multiple': 4.0,    # section 12.1 / 12.2: |Delta| <= first-order bound + 4 eps sum|terms|
    'cone_rel': 1e-12,                      # section 12.1
    'objective_rel': 1e-8,
    'coupling_reference': 0.0,              # coupling residual == T0.1b's recorded residual (exact)
    'legacy_ruling_10_1_ii_floor': 1e-12,   # information only (superseded by 12.1)
    'legacy_ruling_10_1_ii_multiple': 8.0,
    'legacy_envelope_abs': 1e-9,            # information only (superseded by 12.2)
    'legacy_stop_abs': 1e-10,               # information only (superseded by 12.1)
    'q_plus_paths_abs': 1e-6,               # EUR: production-detector path vs gross-repn path of the correction
}

PREDICTIONS = {
    'C1': 'every retained and envelope row within the restated gate in all block-modes (0 exceedances)',
    'C2': 'the three variants coincide at x = 0 -- the R, A and C LP files of each x = 0 group are byte-identical',
    'C3': 'Q_cert+ - gross = 13,012.57 EUR (x = 0) and 13,032.47 EUR (unit) to the cent',
    'C4': ('at the unit, E_C per block lies between E_A (available, certified) and E_R (rated): E_A <= E_C <= E_R in '
           'every (cohort, block), with E_C / E_R = 0.985^5, 0.985^10, 0.985^15 for the 2025 cohort'),
    'C5': ('the capacity-derived bound inventory contains exactly the two SoC-final slack bounds on each network copy '
           'plus whatever the ESSO carries (state the count)'),
}
PREDICTED_CORRECTION = {'x0': 13012.57, 'unit_n7': 13032.47}

REQUIRED_CAPTURES = {
    'input sha256 vs spec section 2, ESS params / case file vs cell records (refuse on mismatch)': ('inventory.json',
                                                                                                  'inputs'),
    'T0.1b manifest hash and the T0.1b files read, by reference': ('inventory.json', 't01b_reference'),
    'captures reused by reference: T0.1b capture hashes + live fingerprints == T0.1b fp_before': ('inventory.json',
                                                                                                 'captures'),
    'cell records: gross, detector totals, certified cycle': ('inventory.json', 'cell_records'),
    'variant C inputs: phi_cal (file + model), Years map, age_end, soh point': ('capacity.json', 'calendar_inputs'),
    'capacity vectors R, A, C per cohort-block (ESSO) and per group / block / e (network copies)': ('capacity.json',
                                                                                                    'vectors'),
    'capacity-derived bound inventory (static code sites + production-function probes), counts': ('capacity.json',
                                                                                                  'bound_inventory'),
    'capacity-derived bounds applied per variant (production call vs arithmetic)': ('capacity.json', 'bounds_applied'),
    'row inventory per block (class / sub / family / form) and nonconvex rows after relaxation': ('inventory.json',
                                                                                                  'blocks'),
    'retained rows: same-point, restated gate, mapping offset per block': ('identities.json', 'groups'),
    'envelope-rewrite rows: first-order table per block-mode': ('identities.json', 'envelope_first_order'),
    'ESSO identities per form (SoH == 1 / calendar)': ('identities.json', 'esso'),
    'coupling residuals vs T0.1b recorded residuals': ('identities.json', 'coupling'),
    'objective identity per block / group / 12-group sum': ('objective.json', 'cells'),
    'Q_cert+ per block, group and cell, with both computation paths': ('objective.json', 'q_cert_plus'),
    'LP exports with sha256, counts, name maps; variant comparisons (C2) and T0.1b comparison': ('lp/index.json',
                                                                                                 'lp_files'),
    'gate verdicts': ('identities.json', 'gates'),
    'predictions vs outcomes': ('identities.json', 'predictions'),
    'guard counts, process launches, solver libraries': ('identities.json', 'guard'),
    'write containment (audit hook)': ('identities.json', 'containment'),
    'provenance (interpreter, versions, git head, file hashes)': ('inventory.json', 'provenance'),
}
OUTPUT_FILES = ('predictions.json', 'inventory.json', 'identities.json', 'capacity.json', 'objective.json',
                'lp/index.json')

sha256_file = T1B.sha256_file


def rel(path):
    return os.path.relpath(path, REPO)


# ======================================================================================================================
#  Parent
# ======================================================================================================================
def parent():
    import platform
    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    if _REHEARSAL:
        print(f'[REHEARSAL] outputs to scratch {OUT_DIR} (not evidence); group limit {GROUP_LIMIT}')
    try:
        os.mkdir(OUT_DIR)                       # atomic: refuses an existing run and a concurrent copy
    except FileExistsError:
        print(f'[REFUSED] {rel(OUT_DIR)} exists; refusing to run (no overwrite, no concurrent copy).')
        return 4
    started = time.time()
    provenance = {'git_head': T1B._git('rev-parse', 'HEAD'),
                  'git_branch': T1B._git('rev-parse', '--abbrev-ref', 'HEAD'),
                  'tracked_status_before': T1B._git('status', '--porcelain', '--untracked-files=no'),
                  'platform': platform.platform(), 'parent_interpreter': sys.executable,
                  'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    with open(PROVENANCE_PARENT, 'w') as handle:
        json.dump(provenance, handle, indent=1)
    with open(LOG_PATH, 'w', buffering=1) as log:
        log.write(f'# p516_d0_t01c parent start {provenance["started"]} interpreter {sys.executable}\n')
        child = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child'], cwd=REPO,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = child.wait()
        log.write(f'# child exit code {code}; wall {time.time() - started:.1f} s\n')
    after = {'tracked_status_after': T1B._git('status', '--porcelain', '--untracked-files=no')}
    after['tracked_status_unchanged'] = after['tracked_status_after'] == provenance['tracked_status_before']
    with open(os.path.join(OUT_DIR, 'provenance_parent_after.json'), 'w') as handle:
        json.dump(after, handle, indent=1)
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
    print(f'[parent] child exit code {code}; tracked status unchanged {after["tracked_status_unchanged"]}; '
          f'manifest {len(manifest)} files -> {rel(MANIFEST_PATH)}')
    return code


# ======================================================================================================================
#  Child: audit hook (process launches raise; writes recorded) -- installed before anything else
# ======================================================================================================================
_AUDIT = {'process_launch_events': [], 'write_opens': {}, 'fs_mutations': []}


def _install_audit_hook():
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def _path(arg):
        if isinstance(arg, bytes):
            return arg.decode(errors='replace')
        return arg if isinstance(arg, str) else None

    def hook(event, args):
        if event == 'open':
            path = _path(args[0]) if args else None
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or \
                      (isinstance(flags, int) and bool(flags & write_flags))
            if writing:
                path = os.path.abspath(path)
                _AUDIT['write_opens'][path] = _AUDIT['write_opens'].get(path, 0) + 1
        elif event in T1B._LAUNCH_EVENTS:
            _AUDIT['process_launch_events'].append({'event': event, 'args': repr(args)[:300]})
            raise RuntimeError(f'P5.16 D0 T0.1c: process launch blocked in the zero-solve child ({event})')
        elif event in T1B._FS_EVENTS:
            paths = [os.path.abspath(p) for p in (_path(a) for a in args) if p]
            _AUDIT['fs_mutations'].append({'event': event, 'paths': paths})
    sys.addaudithook(hook)


def _lp_lines(path):
    with open(path, 'rb') as handle:
        return handle.read().split(b'\n')


def _compare_lp(path_a, path_b, max_lines=40):
    """Raw-byte equality, body equality (every line after the first, the Pyomo source-name comment) and the
    differing body lines (line-by-line; only meaningful when the line counts agree)."""
    a, b = _lp_lines(path_a), _lp_lines(path_b)
    out = {'raw_identical': a == b, 'header_a': a[0].decode(errors='replace'), 'header_b': b[0].decode(errors='replace'),
           'n_lines': [len(a), len(b)], 'body_identical': a[1:] == b[1:],
           'body_sha256': [hashlib.sha256(b'\n'.join(a[1:])).hexdigest(), hashlib.sha256(b'\n'.join(b[1:])).hexdigest()]}
    if len(a) == len(b):
        diff = [i for i in range(1, len(a)) if a[i] != b[i]]
        out['n_differing_body_lines'] = len(diff)
        out['differing_body_lines'] = [{'line': i + 1, 'a': a[i].decode(errors='replace'),
                                        'b': b[i].decode(errors='replace'),
                                        'nearest_preceding_label_line': next(
                                            (a[j].decode(errors='replace') for j in range(i, 0, -1)
                                             if a[j].endswith(b':')), None)}
                                       for i in diff[:max_lines]]
    else:
        out['n_differing_body_lines'] = None
    return out


def child():
    _install_audit_hook()
    sys.path.insert(0, REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.16 D0 T0.1c certified-block relaxation R/A/C (zero solves)'
                              ).install()
    t_start = time.time()

    import gate_result_io as gio

    def write_json(name, payload, compact=False):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            if compact:
                gio.dump(payload, handle, default=gio.json_default_item, separators=(',', ':'))
            else:
                gio.dump(payload, handle, default=gio.json_default_item, indent=1)
        return path

    # ---- predictions first, before any input is loaded ---------------------------------------------------------------
    write_json('predictions.json', {'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'predictions': PREDICTIONS,
                                    'predicted_correction_eur': PREDICTED_CORRECTION, 'thresholds': THRESHOLDS,
                                    'note': 'recorded by the harness before any input is loaded (task P5.16 D0 / '
                                            'T0.1c, Planner predictions C1-C5)'})
    print('[write] predictions.json')

    missing = sorted({f for f, _k in REQUIRED_CAPTURES.values() if f not in OUTPUT_FILES})
    if missing:
        raise RuntimeError(f'capture-path assertion failed: no writer for {missing}')
    print(f'[capture] {len(REQUIRED_CAPTURES)} required quantities -> writers present for {sorted(OUTPUT_FILES)}')

    # ---- input hashes: refuse on any mismatch, before anything is loaded --------------------------------------------
    inputs, mismatch = {}, []
    for cell in CELLS:
        for name, expected in cell['sha256'].items():
            path = os.path.join(REPO, cell['eval_dir'], name)
            actual = sha256_file(path) if os.path.exists(path) else None
            inputs[f'{cell["label"]}/{name}'] = {'path': os.path.join(cell['eval_dir'], name), 'expected': expected,
                                                 'actual': actual, 'match': actual == expected}
            print(f'[inputs] {cell["label"]:8s} {name:24s} {actual} match={actual == expected}')
            if actual != expected:
                mismatch.append(f'{cell["label"]}/{name}')
        ev = json.load(open(os.path.join(REPO, cell['eval_dir'], 'evaluation_record.json')))
        for key, path, expected in (
                ('ess_params', ESS_PARAMS_FILE, ev['configuration']['ess_params_file']['sha256']),
                ('case_params', os.path.join('data', 'SRP1', 'SRP1_params.json'),
                 ev['configuration']['case_file_sha256'])):
            actual = sha256_file(os.path.join(REPO, path))
            inputs[f'{cell["label"]}/{key}'] = {'path': path, 'expected_from_cell_record': expected, 'actual': actual,
                                                'match': actual == expected}
            print(f'[inputs] {cell["label"]:8s} {key:24s} {actual} match={actual == expected}')
            if actual != expected:
                mismatch.append(f'{cell["label"]}/{key}')
    t01b_reference = {'root': T01B_ROOT, 'manifest_sha256_expected': T01B_MANIFEST_SHA256}
    t01b_manifest_path = os.path.join(REPO, T01B_ROOT, 'manifest_sha256.json')
    t01b_reference['manifest_sha256_actual'] = sha256_file(t01b_manifest_path)
    if t01b_reference['manifest_sha256_actual'] != T01B_MANIFEST_SHA256:
        mismatch.append('t01b/manifest_sha256.json')
    t01b_manifest = json.load(open(t01b_manifest_path))['files']
    t01b_reference['files_read'] = {}
    for name in T01B_FILES_READ:
        actual = sha256_file(os.path.join(REPO, T01B_ROOT, name))
        ok = actual == t01b_manifest[name]['sha256']
        t01b_reference['files_read'][name] = {'sha256': actual, 'manifest': t01b_manifest[name]['sha256'], 'match': ok}
        if not ok:
            mismatch.append(f't01b/{name}')
    print(f'[inputs] t01b manifest {t01b_reference["manifest_sha256_actual"]} '
          f'match={t01b_reference["manifest_sha256_actual"] == T01B_MANIFEST_SHA256}; files read '
          f'{ {k: v["match"] for k, v in t01b_reference["files_read"].items()} }')
    if mismatch:
        write_json('inventory.json', {'inputs': inputs, 't01b_reference': t01b_reference, 'refused': True,
                                      'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5

    import pickle

    import numpy as np
    import pyomo
    import pyomo.environ as pe

    from pyomo.core.expr.visitor import identify_variables
    from pyomo.repn import generate_standard_repn

    import model_construction_helpers as mch
    import p516_d0_build_relaxation as B
    import shared_energy_storage_data as sed
    import shared_resources_planning as srp
    from shared_resources_planning import SharedResourcesPlanning

    EPS = float(np.finfo(float).eps)
    MULT = THRESHOLDS['restated_resolution_multiple']

    with open(PROVENANCE_PARENT) as handle:
        provenance = json.load(handle)
    files_hashed = ['p516_d0_t01c.py', 'p516_d0_t01b.py', 'p516_d0_build_relaxation.py', 'p516_d0_dry_run.py',
                    'model_construction_helpers.py', 'network.py', 'network_data.py', 'shared_resources_planning.py',
                    'shared_energy_storage_data.py', 'shared_energy_storage_parameters.py',
                    'p513_solve_profile_guard.py', 'gate_result_io.py', 'P5_16_D0_SPEC_v2.md',
                    'P5_16_D0_T01B_REPORT.md', CASE_FILE, 'data/SRP1/SRP1_params.json', ESS_PARAMS_FILE]
    provenance.update({'interpreter': sys.executable, 'python': sys.version, 'pyomo': pyomo.version.version,
                       'numpy': np.__version__,
                       'file_sha256': {f: sha256_file(os.path.join(REPO, f)) for f in files_hashed},
                       'out_dir': rel(OUT_DIR), 'rehearsal': bool(_REHEARSAL), 'group_limit': GROUP_LIMIT})

    t01b_identities = json.load(open(os.path.join(REPO, T01B_ROOT, 'identities.json')))
    t01b_coupling = json.load(open(os.path.join(REPO, T01B_ROOT, 'coupling.json')))
    t01b_lp_index = json.load(open(os.path.join(REPO, T01B_ROOT, 'lp', 'index.json')))['lp_files']

    # ---- planning object; production's diagram writes redirected into OUT_DIR ----------------------------------------
    os.makedirs(SIDE_WRITES_DIR)
    planning = SharedResourcesPlanning(SRP1_DIR, 'SRP1.json')
    planning.diagrams_dir = SIDE_WRITES_DIR
    planning.read_planning_problem()
    tn_data = planning.transmission_network
    esso_years = list(planning.shared_ess_data.years)
    esso_days = list(planning.shared_ess_data.days)
    print(f'[planning] read in {time.time() - t_start:.1f} s; years {list(tn_data.years)} days {list(tn_data.days)}')

    gate_rows, stops = [], []

    def gate(name, scope, achieved, threshold, rule='<=', note=None):
        achieved = float(achieved)
        ok = bool(achieved <= threshold) if rule == '<=' else bool(achieved == threshold)
        gate_rows.append({'gate': name, 'scope': scope, 'achieved': achieved, 'threshold': threshold, 'rule': rule,
                          'pass': ok, 'note': note})
        if not ok:
            print(f'[gate FAIL] {name} {scope}: achieved {achieved:.6e} {rule} {threshold:.1e}')
        return ok

    # ---- variant C inputs (section 12.3) ------------------------------------------------------------------------------
    case_json = json.load(open(os.path.join(REPO, CASE_FILE)))
    years_map = {str(k): int(v) for k, v in case_json['Years'].items()}
    ess_params_json = json.load(open(os.path.join(REPO, ESS_PARAMS_FILE)))
    phi_file = float(ess_params_json['ageing']['calendar_retention_per_year'])
    soh_point, ageing_enabled = sed._esso_ageing_model_settings(planning.shared_ess_data)
    phi_model = sorted({float(s.phi_cal) for y in planning.shared_ess_data.shared_energy_storages
                        for s in planning.shared_ess_data.shared_energy_storages[y]})
    calendar_inputs = {
        'case_file': CASE_FILE, 'case_file_sha256': sha256_file(os.path.join(REPO, CASE_FILE)),
        'years_map_case_file': years_map,
        'years_map_planning': {str(k): v for k, v in planning.shared_ess_data.years.items()},
        'ess_params_file': ESS_PARAMS_FILE, 'ess_params_sha256': sha256_file(os.path.join(REPO, ESS_PARAMS_FILE)),
        'calendar_retention_per_year_file': phi_file, 'phi_cal_on_planning_objects': phi_model,
        'available_energy_soh_point': soh_point, 'ageing_enabled': ageing_enabled,
        'age_end_definition': 'age_end(y_inv, y) = sum_{k = y_inv..y} num_years_k (k over the case-file Years map in '
                              'order); 0 for y < y_inv (empty sum: the pair is outside the cohort, rated is fixed 0)',
    }
    gate('Years map: case file == planning (0 = equal)', 'inputs',
         0.0 if list(years_map.items()) == [(str(k), v) for k, v in planning.shared_ess_data.years.items()] else 1.0,
         0.0)
    gate('phi_cal: ESS params file == every planning ESS object (0 = equal)', 'inputs',
         0.0 if phi_model == [phi_file] else 1.0, 0.0)
    gate('phi_cal == 0.985 as expected by the task (0 = equal)', 'inputs', 0.0 if phi_file == 0.985 else 1.0, 0.0)
    if soh_point != 'end' or ageing_enabled is not True:
        stops.append({'stop': 'section 12.3 defines C on the end-of-block convention with ageing on',
                      'soh_point': soh_point, 'ageing_enabled': ageing_enabled})
    year_list = list(years_map)
    nyears = [years_map[y] for y in year_list]

    def age_end(y_inv, y):
        return sum(nyears[y_inv:y + 1]) if y >= y_inv else 0

    calendar_factor = {(yi, y): phi_file ** age_end(yi, y) for yi in range(len(year_list))
                       for y in range(len(year_list))}
    calendar_inputs['calendar_factor'] = {f'({yi}, {y})': {'age_end_years': age_end(yi, y), 'factor': f}
                                          for (yi, y), f in sorted(calendar_factor.items())}
    print(f'[variant C] phi_cal {phi_file} (file) / {phi_model} (objects); Years {years_map}; soh point {soh_point}; '
          f'factors {[round(f, 6) for f in calendar_factor.values()]}')

    inventory_out = {'provenance': provenance, 'inputs': inputs, 't01b_reference': t01b_reference,
                     'cell_records': {}, 'captures': {}, 'blocks': {}, 'nonconvex_after': {}}
    identities_out = {'groups': {}, 'envelope_first_order': {}, 'esso': {}, 'coupling': {}, 'fingerprints': {},
                      'w_quadratic_audit': {}}
    capacity_out = {'calendar_inputs': calendar_inputs, 'vectors': {}, 'bound_inventory': {
        'static_code_sites': B.CAPACITY_BOUND_CODE_SITES, 'per_block': {}, 'esso_probe': {}, 'summary': {}},
        'bounds_applied': {}}
    objective_out = {'variant_semantics': VARIANT_SEMANTICS,
                     'convention': ('gross_operational_cost (settlement-EXCLUDED, salvage NOT subtracted): per block '
                                    'w_b * [objective_function_rule(b) - contracted settlement_b - voltage pin_b]; the '
                                    'ESSO contributes 0. Q_cert+ = gross with every detector (slack-penalty) term at '
                                    'max(slack, 0) (spec section 12 ruling 5).'), 'cells': {}, 'q_cert_plus': {}}
    lp_index = {'lp_files': {}, 'variant_semantics': VARIANT_SEMANTICS, 'comparisons': {}}

    channel_of = {'v_W': 'v', 'pf_p': 'pf', 'pf_q': 'pf', 'ess_p_tso_esso': 'ess', 'ess_p_dso_esso': 'ess',
                  'ess_q_tso_esso': 'ess', 'ess_q_dso_esso': 'ess'}
    detector_functions = (('voltage_slack', mch.voltage_slack_penalty),
                          ('node_balance_slack', mch.node_balance_slack_penalty),
                          ('branch_flow_slack', mch.branch_flow_slack_penalty),
                          ('flexibility_p_day_balance_slack', mch.flexibility_p_day_balance_slack_penalty),
                          ('local_ess_day_balance_slack', mch.local_ess_day_balance_slack_penalty),
                          ('shared_ess_day_balance_slack', mch.shared_ess_day_balance_slack_penalty))
    w_families = set(B._W_FAMILIES)

    def slack_projection(model, network, params):
        """Section 12.5, per block (unweighted). Path 1: the six production detector functions (the lineage of
        `srp._get_local_detector_components`, which mirrors `_get_local_slack_penalty_components`), each linear in its
        slacks; projected value = constant + sum c * max(x, 0). Path 2: the block's gross expression
        (`B.block_gross_expr`) -- its linear coefficient on every slack of path 1 -- re-evaluated the same way."""
        families, slacks, coef1 = {}, {}, {}
        for fname, fn in detector_functions:
            expr = 0
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    prob = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                    expr = expr + prob * fn(model, network, s_m, s_o, params)
            value = float(pe.value(expr))
            if isinstance(expr, (int, float)):
                const, lin = float(expr), []
            else:
                repn = generate_standard_repn(expr, compute_values=True, quadratic=True)
                if repn.nonlinear_expr is not None or repn.quadratic_vars:
                    raise RuntimeError(f'{fname}: detector term is not linear in its slacks')
                const = float(pe.value(repn.constant))
                lin = [(float(pe.value(c)), v) for c, v in zip(repn.linear_coefs, repn.linear_vars)]
            proj = const + sum(c * max(v.value, 0.0) for c, v in lin)
            from_repn = const + sum(c * v.value for c, v in lin)
            neg = [(c, v) for c, v in lin if v.value < 0.0]
            families[fname] = {'certified': value, 'certified_from_repn': from_repn, 'projected': proj,
                               'correction': proj - value, 'n_slack_terms': len(lin), 'n_negative': len(neg),
                               'min_slack': min((v.value for _c, v in lin), default=None),
                               'positive_part': sum(c * max(v.value, 0.0) for c, v in lin) + const,
                               'slack_families': sorted({v.parent_component().local_name for _c, v in lin})}
            for c, v in lin:
                slacks[v.name] = v
                coef1[v.name] = coef1.get(v.name, 0.0) + c
        lb_issues = [n for n, v in slacks.items() if v.lb != 0.0]
        ub_excess = [(n, v.value - v.ub) for n, v in slacks.items() if v.ub is not None and v.value > v.ub]
        gross_expr, _legacy = B.block_gross_expr(model, params)
        repn = generate_standard_repn(gross_expr, compute_values=True, quadratic=True)
        coef2 = {}
        for c, v in zip(repn.linear_coefs, repn.linear_vars):
            if v.name in slacks:
                coef2[v.name] = coef2.get(v.name, 0.0) + float(pe.value(c))
        in_nonlinear = []
        for v1, v2 in repn.quadratic_vars:
            in_nonlinear += [v.name for v in (v1, v2) if v.name in slacks]
        if repn.nonlinear_expr is not None:
            in_nonlinear += [v.name for v in identify_variables(repn.nonlinear_expr) if v.name in slacks]
        coef_mismatch = [n for n in slacks if abs(coef1[n] - coef2.get(n, 0.0)) > 1e-12 * max(1.0, abs(coef1[n]))]
        corr1 = sum(f['correction'] for f in families.values())
        corr2 = sum(coef2.get(n, 0.0) * (max(v.value, 0.0) - v.value) for n, v in slacks.items())
        return {'families': families, 'correction_path1': corr1, 'correction_path2': corr2,
                'n_slack_vars': len(slacks), 'slack_lb_not_zero': lb_issues, 'n_above_ub': len(ub_excess),
                'max_above_ub': max((x for _n, x in ub_excess), default=0.0),
                'slacks_in_quadratic_or_nonlinear_gross_terms': sorted(set(in_nonlinear)),
                'coefficient_mismatch_gross_vs_detector': coef_mismatch[:20],
                'n_coefficient_mismatch': len(coef_mismatch),
                'detector_total_certified': sum(f['certified'] for f in families.values())}

    # ==================================================================================================================
    #  Per cell
    # ==================================================================================================================
    q_plus_cells = {}
    for cell in CELLS:
        label = cell['label']
        cell_dir = os.path.join(REPO, cell['eval_dir'])
        t_cell = time.time()
        print(f'\n========== cell {label} ({cell["eval_dir"]}) ==========')
        records = {}
        for name in RECORD_FILES:
            path = os.path.join(cell_dir, name)
            records[name] = json.load(open(path))
            records[name + '__sha256'] = sha256_file(path)
        ev = records['evaluation_record.json']
        recorded_gross = float(ev['terminal_gross_operational_cost'])
        recorded_detector = ev['recourse_components']['detector_components']
        inventory_out['cell_records'][label] = {
            'eval_dir': cell['eval_dir'], 'candidate_key': ev.get('candidate_key'),
            'candidate_canonical': ev.get('candidate_canonical'), 'candidate_label': ev.get('candidate_label'),
            'terminal_gross_operational_cost': ev['terminal_gross_operational_cost'],
            'terminal_net_operational_recourse': ev['terminal_net_operational_recourse'],
            'objective_convention': ev['objective_convention'], 'certification_cycle': ev['certification_cycle'],
            'detector_components_recorded': recorded_detector,
            'record_sha256': {n: records[n + '__sha256'] for n in RECORD_FILES}}
        gate('certified cycle == spec section 2 (0 = equal)', label,
             0.0 if ev['certification_cycle'] == cell['certified_cycle_spec'] else 1.0, 0.0)

        cm_path = os.path.join(cell_dir, 'certified_models.pkl')
        es_path = os.path.join(cell_dir, 'esso_models_s39_D.pkl')
        with open(cm_path, 'rb') as handle:
            certified = pickle.load(handle)
        with open(es_path, 'rb') as handle:
            esso_live = pickle.load(handle)
        for path, name in ((cm_path, 'certified_models.pkl'), (es_path, 'esso_models_s39_D.pkl')):
            if sha256_file(path) != cell['sha256'][name]:
                raise RuntimeError(f'{path} changed while loading')
        tso_live, dso_live = certified['tso'], certified['dso']
        years = list(tso_live.keys())
        days = list(tso_live[years[0]].keys())
        nodes = sorted(dso_live.keys())
        if years != list(tn_data.years) or days != list(tn_data.days) or years != esso_years or days != esso_days \
                or [str(y_) for y_ in years] != year_list:
            raise RuntimeError('year/day ordering differs between blocks, planning, ESSO and the case file')
        if nodes != sorted(planning.active_distribution_network_nodes) or sorted(esso_live) != nodes:
            raise RuntimeError('node sets differ')
        groups = [(y, d) for y in years for d in days]
        if GROUP_LIMIT:
            groups = groups[:GROUP_LIMIT]

        def live_blocks(y, d):
            return [('TSO', None, tso_live[y][d])] + [(f'DSO{n}', n, dso_live[n][y][d]) for n in nodes]

        def block_data(blabel, n):
            return tn_data if blabel == 'TSO' else planning.distribution_networks[n]

        # ---- captures by reference: fingerprints == T0.1b's fp_before; T0.1b capture files hash-checked ------------
        fp_before = {}
        for y, d in [(y, d) for y in years for d in days]:
            for blabel, _n, m in live_blocks(y, d):
                fp_before[f'{y}_{d}_{blabel}'] = B.model_fingerprint(m)
        for n in nodes:
            fp_before[f'ESSO{n}'] = B.model_fingerprint(esso_live[n])
        t01b_fp = t01b_identities['fingerprints'][label]['before']
        fp_equal = {k: fp_before[k] == t01b_fp.get(k) for k in fp_before}
        cap_files = {k: v for k, v in t01b_manifest.items() if k.startswith(f'captures/{label}/')}
        cap_check = {k: sha256_file(os.path.join(REPO, T01B_ROOT, k)) == v['sha256'] for k, v in cap_files.items()}
        inventory_out['captures'][label] = {
            'method': ('reused by reference: the T0.1b captures (pickles of the live blocks / ESSO before mutation, Var '
                       'and mutable-Param vectors) are NOT duplicated; each file is hash-checked against the T0.1b '
                       'manifest, and the fingerprint of every live block loaded here equals the fingerprint T0.1b '
                       'recorded before mutation (identities.json fingerprints.before)'),
            't01b_capture_files': {k: {'sha256': v['sha256'], 'bytes': v['bytes'], 'hash_ok': cap_check[k]}
                                   for k, v in sorted(cap_files.items())},
            'fingerprints_equal_t01b': fp_equal, 'n_fingerprints': len(fp_equal)}
        gate('live-block fingerprints == T0.1b fp_before (count unequal)', label,
             float(sum(1 for v in fp_equal.values() if not v)), 0.0)
        gate('T0.1b capture files hash == T0.1b manifest (count unequal)', label,
             float(sum(1 for v in cap_check.values() if not v) + (0 if len(cap_check) == 26 else 1)), 0.0)

        # ---- production per-block gross and detector components ------------------------------------------------------
        models_live = {'tso': tso_live, 'dso': dso_live, 'esso': esso_live}
        comps = srp._get_operational_recourse_components(planning, models_live)
        contracted_blocks = srp._get_operational_interface_settlement_blocks(planning, models_live, part='contracted')
        pin_blocks = srp._get_operational_voltage_pin_blocks(planning, models_live)
        prod_block = {}
        for y, d in [(y, d) for y in years for d in days]:
            for blabel, n, m in live_blocks(y, d):
                nd = block_data(blabel, n)
                w = srp._get_admm_block_weight(nd, y, d)
                prim = nd.network[y][d].get_primal_value(m, nd.params)
                key = ('TSO', None, y, d) if blabel == 'TSO' else ('DSO', n, y, d)
                prod_block[(y, d, blabel)] = {'weight': w, 'primal_weighted': w * prim,
                                              'contracted_weighted': contracted_blocks[key],
                                              'pin_weighted': pin_blocks[key]}
                prod_block[(y, d, blabel)]['gross_weighted'] = (w * prim - contracted_blocks[key] - pin_blocks[key])
        gate('production gross recomputed on the pickles vs recorded gross (rel)', label,
             abs(comps['gross_operational_cost'] - recorded_gross) / abs(recorded_gross), THRESHOLDS['objective_rel'])
        gate('production detector total recomputed vs recorded (abs EUR)', label,
             abs(comps['detector_penalty_total'] - recorded_detector['detector_penalty_total']), 1e-6)

        # ---- capacities: ESSO cohorts (R, A, C) and network copies -------------------------------------------------
        cap_cell = {'esso': {}, 'network': {}}
        e_c_node_year, e_r_node_year = {}, {}
        c4_rows = []
        for n in nodes:
            em = esso_live[n]
            rows = {}
            for yi in range(len(years)):
                for y in range(len(years)):
                    # E_R of the cohort in block y: the investment Param inside the calendar-life window (where
                    # `rated_e_capacity_unit` pins es_e_rated_per_unit to it), 0 outside (Var fixed at 0 there)
                    in_window = not em.es_e_rated_per_unit[yi, y].fixed
                    inv_e = float(pe.value(em.es_e_investment_fixed[yi]))
                    e_r = inv_e if in_window else 0.0
                    e_r_var = em.es_e_rated_per_unit[yi, y].value
                    e_a = em.es_e_available_per_unit[yi, y].value
                    e_c = e_r * calendar_factor[(yi, y)]
                    rows[f'({yi}, {y})'] = {'cohort': year_list[yi], 'block': year_list[y], 'in_window': in_window,
                                            'E_R': e_r, 'E_R_var_certified': e_r_var, 'E_A_certified': e_a,
                                            'E_C': e_c, 'factor': calendar_factor[(yi, y)], 'age_end': age_end(yi, y),
                                            'es_e_investment_fixed': inv_e,
                                            'soh_certified': em.es_soh_per_unit_cumul[yi, y].value,
                                            'E_C_over_E_R': (e_c / e_r) if e_r else None}
                    gate('ESSO es_e_rated_per_unit (certified Var) == E_R (abs)', f'{label} ESSO{n} ({yi},{y})',
                         abs(e_r_var - e_r), 1e-12)
                    if e_r > 0.0:
                        c4_rows.append({'node': n, 'pair': (yi, y), 'E_A': e_a, 'E_C': e_c, 'E_R': e_r,
                                        'ordered': bool(e_a <= e_c <= e_r), 'ratio': e_c / e_r})
            for y in range(len(years)):
                e_c_node_year[(n, years[y])] = sum(rows[f'({yi}, {y})']['E_C'] for yi in range(len(years)))
                e_r_node_year[(n, years[y])] = sum(rows[f'({yi}, {y})']['E_R'] for yi in range(len(years)))
            cap_cell['esso'][f'ESSO{n}'] = rows
        r_check = []
        for (y, d) in groups:
            for blabel, n, m in live_blocks(y, d):
                net, _p = B.network_and_params(m)
                sb = float(net.baseMVA)
                per_e = {}
                for e in m.shared_es_s_rated_fixed:
                    node = net.shared_energy_storages[e].bus if blabel == 'TSO' else n
                    e_r, s_r = float(net.shared_energy_storages[e].e), float(net.shared_energy_storages[e].s)
                    e_r_esso = e_r_node_year[(node, y)] / sb
                    r_check.append(abs(e_r - e_r_esso))
                    per_e[e] = {'node': node, 'sb_MVA': sb,
                                'R': {'s': s_r, 'e': e_r},
                                'A': {'s': float(pe.value(m.shared_es_s_rated_fixed[e])),
                                      'e': float(pe.value(m.shared_es_e_rated_fixed[e]))},
                                'C': {'s': s_r, 'e': e_c_node_year[(node, y)] / sb},
                                'E_R_from_esso_pu': e_r_esso}
                cap_cell['network'][f'{y}_{d}_{blabel}'] = per_e
        gate('network rated E (data object) == sum ESSO E_rated / s_base (max abs pu)', label, max(r_check), 1e-15)
        cap_cell['c4_rows'] = c4_rows
        capacity_out['vectors'][label] = cap_cell

        # ---- capacity-derived bound inventory: production-function probes on every block, and the ESSO -----------
        value_dep_union = set()
        inv_cell = {}
        for (y, d) in groups:
            for blabel, n, m in live_blocks(y, d):
                probes = {}
                for e, c in cap_cell['network'][f'{y}_{d}_{blabel}'].items():
                    pos = (c['R']['s'], c['R']['e']) if min(c['R']['s'], c['R']['e']) > 0 else (0.0025, 0.01)
                    probes[e] = [(0.0, 0.0), pos, (2.0 * pos[0], 2.0 * pos[1])]
                res = B.capacity_bound_inventory(m, probes)
                inv_cell[f'{y}_{d}_{blabel}'] = res
                value_dep_union |= set(res['value_dependent'])
        capacity_out['bound_inventory']['per_block'][label] = inv_cell
        esso_probe = {}
        for n in nodes:
            scratch = esso_live[n].clone()
            per_cohort = {}
            for yi in range(len(years)):
                snaps = []
                for s_cap, e_cap in ((0.0, 0.0), (0.25, 1.0), (0.5, 2.0)):
                    sed._configure_esso_cohort_state(scratch, yi, s_cap, e_cap,
                                                     planning.shared_ess_data.params.slacks)
                    snaps.append(B._state_snapshot(scratch))
                value_dep, zero_gated = {}, {}
                for name, (_v, _f, lb0, ub0) in snaps[0][0].items():
                    b1, b2 = snaps[1][0][name][2:], snaps[2][0][name][2:]
                    fam = B._family_of(name)
                    if b1 != b2:
                        value_dep[fam] = value_dep.get(fam, 0) + 1
                    elif (lb0, ub0) != b1:
                        zero_gated[fam] = zero_gated.get(fam, 0) + 1
                per_cohort[str(yi)] = {'value_dependent_bounds': value_dep, 'zero_gated_bounds': zero_gated,
                                       'zero_vs_positive_effects': B._snapshot_diff(snaps[0], snaps[1]),
                                       'positive_vs_positive_effects': B._snapshot_diff(snaps[1], snaps[2])}
            esso_probe[f'ESSO{n}'] = {
                'method': ('production `shared_energy_storage_data._configure_esso_cohort_state` (l.1585) called on a '
                           'scratch clone per cohort at (S, E) = (0, 0), (0.25, 1.0), (0.5, 2.0) MVA / MWh; it is the '
                           'function `_update_model_with_candidate_solution` (l.1427) calls per cohort'),
                'per_cohort': per_cohort,
                'n_value_dependent_bounds': sum(sum(c['value_dependent_bounds'].values()) for c in per_cohort.values()),
                'n_zero_gated_bounds': sum(sum(c['zero_gated_bounds'].values()) for c in per_cohort.values())}
            del scratch
        capacity_out['bound_inventory']['esso_probe'][label] = esso_probe
        derived_families = sorted(value_dep_union)
        vd_counts = {}
        for k, res in inv_cell.items():
            for fam, cnt in res['value_dependent'].items():
                vd_counts[fam] = vd_counts.get(fam, 0) + cnt
        zg_counts = {}
        for k, res in inv_cell.items():
            for fam, cnt in res['zero_gated'].items():
                zg_counts[fam] = zg_counts.get(fam, 0) + cnt
        n_shared_ess_copies = sum(len(cap_cell['network'][f'{y}_{d}_{b}']) for (y, d) in groups
                                  for b, _n, _m in live_blocks(y, d))
        capacity_out['bound_inventory']['summary'][label] = {
            'value_dependent_families': derived_families, 'value_dependent_bound_count': vd_counts,
            'zero_gated_bound_count': zg_counts, 'n_shared_ess_copies': n_shared_ess_copies,
            'value_dependent_per_copy': {f: c / n_shared_ess_copies for f, c in vd_counts.items()},
            'esso_value_dependent_bounds': sum(v['n_value_dependent_bounds'] for v in esso_probe.values()),
            'esso_zero_gated_bounds': sum(v['n_zero_gated_bounds'] for v in esso_probe.values())}
        print(f'[bounds] {label}: value-dependent {vd_counts} over {n_shared_ess_copies} shared-ESS copies; '
              f'zero-gated {zg_counts}; ESSO value-dependent '
              f'{capacity_out["bound_inventory"]["summary"][label]["esso_value_dependent_bounds"]}')

        # ---- ESSO relaxations (SoH == 1 form for R / A, calendar form for C) and identities ------------------------
        esso_out = {}
        slack_rows_by_node = {}
        for n in nodes:
            em = esso_live[n]
            inv = B.inventory_rows(em, 'esso')
            inventory_out['blocks'][f'{label}/ESSO{n}'] = inv
            slack_bound = {}
            for component in em.component_objects(pe.Constraint, active=True):
                for index, cd in component.items():
                    if not cd.active:
                        continue
                    if {v.parent_component().local_name for v in identify_variables(cd.body, include_fixed=True)} \
                            & set(B.ESSO_SLACK_VARS):
                        repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
                        slack_bound[(component.name, repr(index))] = sum(
                            abs(float(pe.value(c))) * abs(v.value) for c, v in zip(repn.linear_coefs, repn.linear_vars)
                            if v.parent_component().local_name in B.ESSO_SLACK_VARS)
            slack_rows_by_node[n] = slack_bound
            for form, factors in (('soh1', None), ('calendar', calendar_factor)):
                relaxed, record = B.relax_esso_block(em, calendar_factors=factors)
                nonconvex = B.nonconvex_active_rows(relaxed)
                tables = B.mapped_point(em, relaxed)
                same_point = B.production_rows_at_point(em, relaxed)
                over, worst = [], 0.0
                for r in tables['retained']:
                    sb_ = slack_bound.get((r['family'], r['index']), 0.0)
                    thr = r['first_order_w_bound'] + sb_ + MULT * r['eval_resolution']
                    worst = max(worst, r['delta'])
                    if r['delta'] > thr:
                        over.append({**{k: r[k] for k in ('family', 'index', 'delta', 'eval_resolution')},
                                     'slack_first_order_bound': sb_, 'threshold': thr})
                added_bad = []
                for a in tables['added_esso']:
                    res = max(abs(x) for x in (a['residual']['r_lower'], a['residual']['r_upper']) if x is not None)
                    if res > MULT * EPS * max(1.0, abs(a['residual']['body'])):
                        added_bad.append(a)
                scope = f'{label} ESSO{n} {form}'
                esso_out[f'ESSO{n}/{form}'] = {
                    'build_record': record, 'nonconvex_after': nonconvex, 'same_point': same_point,
                    'retained': {'n_rows': len(tables['retained']), 'max_delta': worst, 'n_over_restated': len(over),
                                 'rows_over_restated': over,
                                 'n_rows_with_fixed_slack_offset': sum(1 for r in tables['retained']
                                                                       if (r['family'], r['index']) in slack_bound)},
                    'added_rows_at_mapped_point': tables['added_esso'], 'n_added_bad': len(added_bad)}
                inventory_out['nonconvex_after'][f'{label}/ESSO{n}/{form}'] = nonconvex
                gate('nonconvex rows after relaxation (count)', scope, float(len(nonconvex)), 0.0)
                gate('same-point Delta (exact 0)', scope, same_point['max_abs_delta'], 0.0, rule='==')
                gate('ESSO retained rows over restated gate (first-order slack-fixing offset + 4 eps sum|terms|) '
                     '(count)', scope, float(len(over)), 0.0)
                gate('ESSO added available-e rows at the mapped point over 4 eps (count)', scope,
                     float(len(added_bad)), 0.0)
                if nonconvex:
                    stops.append({'stop': 'nonconvex row remains', 'scope': scope, 'rows': nonconvex[:20]})
                if over:
                    stops.append({'stop': 'ESSO row exceeds the restated gate', 'scope': scope, 'rows': over[:20]})
                del relaxed
        identities_out['esso'][label] = esso_out

        # ---- groups ------------------------------------------------------------------------------------------------
        identities_out['groups'][label] = {}
        identities_out['envelope_first_order'][label] = {}
        identities_out['coupling'][label] = {}
        identities_out['w_quadratic_audit'][label] = {}
        capacity_out['bounds_applied'][label] = {}
        objective_cell = {'per_group': {}, 'recorded_gross': recorded_gross,
                          'production_recomputed_gross': comps['gross_operational_cost']}
        lb_sum = {m_: 0.0 for m_ in MODES}
        q_plus_cell = {'per_block': {}, 'per_group': {}}
        t01b_cpl_const = t01b_coupling['constants'][label]
        lp_paths = {}

        for (y, d) in groups:
            t_group = time.time()
            gkey = f'{y}_{d}'
            blocks = live_blocks(y, d)
            group_out = {'modes': {}, 'mapping_offset': {}}
            identities_out['groups'][label][gkey] = group_out
            inv_by = {}
            for blabel, _n, m in blocks:
                inv = B.inventory_rows(m, 'network', drop_sess_comp=True)
                inv_by[blabel] = inv
                inventory_out['blocks'][f'{label}/{gkey}/{blabel}'] = inv
                keys = [(r['family'], r['index']) for r in inv['rows']]
                n_active = sum(1 for _ in m.component_data_objects(pe.Constraint, active=True, descend_into=True))
                gate('G-D0-2 each active row classified exactly once (0 = yes)', f'{label} {gkey} {blabel}',
                     0.0 if (len(keys) == len(set(keys)) == n_active) else 1.0, 0.0)
                # W-quadratic audit: the first-order W bound is first-order exact only for rows linear in W
                nonlin_fams = {r['family'] for r in inv['rows'] if r['class'] == 'retained' and r['form'] != 'linear'}
                w_quad, n_checked = [], 0
                for fam in sorted(nonlin_fams):
                    for index, cd in getattr(m, fam).items():
                        if not cd.active:
                            continue
                        n_checked += 1
                        repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
                        if any(v.parent_component().local_name in w_families
                               for pair in repn.quadratic_vars for v in pair):
                            w_quad.append(cd.name)
                identities_out['w_quadratic_audit'][label][f'{gkey}/{blabel}'] = {
                    'families': sorted(nonlin_fams), 'n_rows_checked': n_checked, 'rows_with_W_in_quadratic': w_quad}
                gate('retained rows with a W variable inside a quadratic term (count)', f'{label} {gkey} {blabel}',
                     float(len(w_quad)), 0.0)

            # Q_cert+ (section 12.5; mode-independent)
            qg = {'gross': 0.0, 'correction_path1': 0.0, 'correction_path2': 0.0}
            for blabel, n, m in blocks:
                nd = block_data(blabel, n)
                sp = slack_projection(m, nd.network[y][d], nd.params)
                w = prod_block[(y, d, blabel)]['weight']
                gross_b = prod_block[(y, d, blabel)]['gross_weighted']
                rec = {'weight': w, 'gross_weighted': gross_b,
                       'correction_weighted_path1': w * sp['correction_path1'],
                       'correction_weighted_path2': w * sp['correction_path2'],
                       'q_cert_plus_weighted': gross_b + w * sp['correction_path1'],
                       'detector_total_weighted': w * sp['detector_total_certified'], 'unweighted': sp}
                prod_det = srp._get_local_detector_components(m, nd.network[y][d], nd.params)
                rec['production_detector_total_unweighted'] = prod_det['detector_penalty_total']
                gate('detector path 1 == production _get_local_detector_components (abs, unweighted)',
                     f'{label} {gkey} {blabel}', abs(sp['detector_total_certified']
                                                     - prod_det['detector_penalty_total']), 1e-12)
                gate('Q_cert+ correction: path 1 vs path 2 (abs EUR, weighted)', f'{label} {gkey} {blabel}',
                     abs(rec['correction_weighted_path1'] - rec['correction_weighted_path2']),
                     THRESHOLDS['q_plus_paths_abs'])
                gate('slack vars in quadratic / nonlinear gross terms or coefficient mismatch (count)',
                     f'{label} {gkey} {blabel}', float(len(sp['slacks_in_quadratic_or_nonlinear_gross_terms'])
                                                      + sp['n_coefficient_mismatch'] + len(sp['slack_lb_not_zero'])),
                     0.0)
                q_plus_cell['per_block'][f'{gkey}_{blabel}'] = rec
                qg['gross'] += gross_b
                qg['correction_path1'] += rec['correction_weighted_path1']
                qg['correction_path2'] += rec['correction_weighted_path2']
            qg['q_cert_plus'] = qg['gross'] + qg['correction_path1']
            q_plus_cell['per_group'][gkey] = qg

            # capacities per variant and capacity-derived bounds (production function on a scratch clone)
            caps, dbounds, bapp = {}, {}, {}
            for blabel, _n, m in blocks:
                per_e = cap_cell['network'][f'{gkey}_{blabel}']
                caps[blabel] = {v: {e: per_e[e][v] for e in per_e} for v in ('R', 'A', 'C')}
                carried_bounds, carried_eff = B.production_capacity_bounds(m, caps[blabel]['A'], derived_families)
                reproduces = all(carried_bounds[f][i] == (getattr(m, f)[i].lb, getattr(m, f)[i].ub)
                                 for f in derived_families for i in carried_bounds[f])
                gate('production call at the carried capacity reproduces the certified bounds (0 = yes)',
                     f'{label} {gkey} {blabel}', 0.0 if reproduces else 1.0, 0.0)
                bapp[blabel] = {'derived_families': derived_families, 'carried_call_reproduces_certified': reproduces,
                                'carried_call_other_effects': carried_eff}
                dbounds[blabel] = {}
                for v in ('R', 'C'):
                    bnds, eff = B.production_capacity_bounds(m, caps[blabel][v], derived_families)
                    arith_bad = []
                    net, _p = B.network_and_params(m)
                    for f in derived_families:
                        for i, (lb, ub) in bnds[f].items():
                            ecap = mch.normalize_shared_ess_capacity(caps[blabel][v][i[0]]['e'])
                            scap = mch.normalize_shared_ess_capacity(caps[blabel][v][i[0]]['s'])
                            inactive = mch.shared_ess_capacity_is_inactive(scap, ecap)
                            expect = 0.0 if inactive else ecap * mch.ESS_DAY_BALANCE_SLACK_FRACTION + \
                                mch.EQUALITY_TOLERANCE
                            if f not in ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down') or \
                                    (lb, ub) != (0.0, expect):
                                arith_bad.append({'var': f'{f}[{i}]', 'production': [lb, ub], 'arithmetic': [0.0,
                                                                                                          expect]})
                    dbounds[blabel][v] = bnds
                    bapp[blabel][v] = {'bounds': {f: {repr(i): list(b) for i, b in bnds[f].items()}
                                                  for f in derived_families},
                                       'certified_bounds': {f: {repr(i): [getattr(m, f)[i].lb, getattr(m, f)[i].ub]
                                                                for i in bnds[f]} for f in derived_families},
                                       'n_changed_vs_certified': sum(1 for f in derived_families for i, b in
                                                                     bnds[f].items() if b != (getattr(m, f)[i].lb,
                                                                                              getattr(m, f)[i].ub)),
                                       'production_call_other_effects': eff, 'arithmetic_mismatch': arith_bad,
                                       'method': ('production configure_shared_ess_operational_state called with the '
                                                  'variant capacity on a scratch clone; bounds of the value-dependent '
                                                  'families copied to the relaxed clone')}
                    gate('capacity-derived bounds: production call == arithmetic e * 0.05 + 1e-5 (count unequal)',
                         f'{label} {gkey} {blabel} {v}', float(len(arith_bad)), 0.0)
            capacity_out['bounds_applied'][label][gkey] = bapp
            capacity_out['vectors'][label].setdefault('per_group', {})[gkey] = caps

            prod_g = {bl: prod_block[(y, d, bl)] for bl, _n, _m in blocks}
            prod_g_sum = sum(v['gross_weighted'] for v in prod_g.values())

            for mode in MODES:
                var = VARIANT[mode]
                scope_g = f'{label} {gkey} {var}'
                relaxed, records_b = {}, {}
                for blabel, _n, m in blocks:
                    kwargs = {'drop_sess_comp': True}
                    if mode == 'rated':
                        kwargs.update(rated_capacity=caps[blabel]['R'], derived_bounds=dbounds[blabel]['R'])
                    elif mode == 'calendar':
                        kwargs.update(calendar_capacity=caps[blabel]['C'], derived_bounds=dbounds[blabel]['C'])
                    rx, rec = B.relax_network_block(m, mode, **kwargs)
                    relaxed[blabel], records_b[blabel] = rx, rec
                mode_out = {'variant': var, 'blocks': {}}
                group_out['modes'][var] = mode_out
                for blabel, _n, m in blocks:
                    inv, rx, rec = inv_by[blabel], relaxed[blabel], records_b[blabel]
                    not_retained = {(r['family'], r['index']) for r in inv['rows'] if r['class'] != 'retained'}
                    retained_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'retained'}
                    inactive_now, active_both = set(), set()
                    for component in m.component_objects(pe.Constraint, active=True):
                        rcomp = rx.find_component(component.name)
                        for index, cd in component.items():
                            if cd.active:
                                (active_both if rcomp[index].active else inactive_now).add(
                                    (component.name, repr(index)))
                    consistent = not_retained == inactive_now and retained_inv == active_both
                    bounds_now = {f: {repr(i): [vd.lb, vd.ub] for i, vd in getattr(rx, f).items()}
                                  for f in derived_families}
                    expect_b = (bapp[blabel][var]['bounds'] if var in ('R', 'C') else
                                {f: {repr(i): [vd.lb, vd.ub] for i, vd in getattr(m, f).items()}
                                 for f in derived_families})
                    mode_out['blocks'][blabel] = {
                        'build_record': rec, 'inventory_builder_consistent': consistent,
                        'capacity_params_set': rec['capacity_params_set'], 'derived_bounds_on_clone': bounds_now}
                    sc = f'{scope_g} {blabel}'
                    gate('G-D0-2 inventory == builder (0 = consistent)', sc, 0.0 if consistent else 1.0, 0.0)
                    gate('capacity-derived bounds on the relaxed clone == the variant\'s (0 = equal)', sc,
                         0.0 if bounds_now == expect_b else 1.0, 0.0)

                # ---- group model with coupling rows -------------------------------------------------------------
                g = pe.ConcreteModel(name=f'P516D0_T01c_{label}_{gkey}_{mode}')
                g.tso = relaxed['TSO']
                for n in nodes:
                    g.add_component(f'dso_{n}', relaxed[f'DSO{n}'])
                    rex, _rec = B.relax_esso_block(esso_live[n], calendar_factors=(calendar_factor if mode ==
                                                                                    'calendar' else None))
                    rex.d0_lb_objective.deactivate()
                    g.add_component(f'esso_{n}', rex)
                tso = g.tso
                st = mch.sess_na_scenario(tso)
                g.d0_cpl_nodes = pe.Set(initialize=nodes, ordered=True)
                g.d0_cpl_periods = pe.Set(initialize=list(tso.periods), ordered=True)

                def k(n):
                    return t01b_cpl_const[f'{y}_{d}_{n}']

                def _v(m_, n, p):
                    c = k(n)
                    dso = m_.find_component(f'dso_{n}')
                    sd = mch.sess_na_scenario(dso)
                    return (tso.vmag_sqr[c['tso_adn_node_idx'], st[0], st[1], p]
                            - (c['vb_dso_kV'] / c['vb_tso_kV']) ** 2
                            * dso.vmag_sqr[c['dso_ref_node_idx'], sd[0], sd[1], p]) == 0

                def _pf(kind):
                    def rule(m_, n, p):
                        c = k(n)
                        dso = m_.find_component(f'dso_{n}')
                        sd = mch.sess_na_scenario(dso)
                        t_expr = (tso.pc_adn if kind == 'p' else tso.qc_adn)[c['dn'], st[0], st[1], p]
                        d_expr = (dso.pg_adn if kind == 'p' else dso.qg_adn)[sd[0], sd[1], p]
                        return t_expr - (c['sb_dso_MVA'] / c['sb_tso_MVA']) * d_expr == 0
                    return rule

                def _ess(kind, side):
                    def rule(m_, n, p):
                        c = k(n)
                        esso = m_.find_component(f'esso_{n}')
                        e_var = (esso.es_pnet if kind == 'p' else esso.es_qnet)[c['esso_year_idx'],
                                                                              c['esso_day_idx'], p]
                        if side == 'tso':
                            x = (tso.shared_es_pnet if kind == 'p' else tso.shared_es_qnet)[c['e_tso'], st[0], st[1], p]
                            return x - e_var / c['sb_tso_MVA'] == 0
                        dso = m_.find_component(f'dso_{n}')
                        sd = mch.sess_na_scenario(dso)
                        x = (dso.shared_es_pnet if kind == 'p' else dso.shared_es_qnet)[c['e_dso'], sd[0], sd[1], p]
                        return x - e_var / c['sb_dso_MVA'] == 0
                    return rule

                g.d0_cpl_v_W = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_v)
                g.d0_cpl_pf_p = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_pf('p'))
                g.d0_cpl_pf_q = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_pf('q'))
                g.d0_cpl_ess_p_tso_esso = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_ess('p', 'tso'))
                g.d0_cpl_ess_p_dso_esso = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_ess('p', 'dso'))
                g.d0_cpl_ess_q_tso_esso = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_ess('q', 'tso'))
                g.d0_cpl_ess_q_dso_esso = pe.Constraint(g.d0_cpl_nodes, g.d0_cpl_periods, rule=_ess('q', 'dso'))

                def to_phys(ch, n):
                    c = k(n)
                    if ch == 'v_W':
                        return c['vb_tso_kV'] ** 2
                    if ch in ('pf_p', 'pf_q', 'ess_p_tso_esso', 'ess_q_tso_esso'):
                        return c['sb_tso_MVA']
                    return c['sb_dso_MVA']

                ref_rows = {(r['channel'], r['node'], r['period']): r['residual_phys_certified']
                            for r in t01b_coupling['rows'][label][gkey]}
                n_cpl, cpl_bad = 0, []
                for ch in channel_of:
                    for (n, p), cd in getattr(g, f'd0_cpl_{ch}').items():
                        resid = (pe.value(cd.body) - pe.value(cd.upper)) * to_phys(ch, n)
                        n_cpl += 1
                        if resid != ref_rows[(ch, n, p)]:
                            cpl_bad.append({'row': cd.name, 'residual': resid, 't01b': ref_rows[(ch, n, p)]})
                identities_out['coupling'][label][f'{gkey}_{var}'] = {'n_rows': n_cpl, 'n_unequal_to_t01b': len(cpl_bad),
                                                                      'unequal': cpl_bad[:20]}
                gate('coupling residual at the certified point == T0.1b recorded residual (count unequal)', scope_g,
                     float(len(cpl_bad)) + (0.0 if n_cpl == len(ref_rows) else 1.0), 0.0)

                lb_terms = []
                for blabel, n, _m in blocks:
                    sub = g.tso if blabel == 'TSO' else g.find_component(f'dso_{n}')
                    _net, prm = B.network_and_params(sub)
                    w = srp._get_admm_block_weight(block_data(blabel, n), y, d)
                    expr, legacy = B.block_gross_expr(sub, prm)
                    if legacy:
                        raise RuntimeError(f'{scope_g} {blabel}: certified block lacks the settlement attributes')
                    lb_terms.append((blabel, w, expr))
                g.d0_lb_objective = pe.Objective(expr=sum(w * e for _b, w, e in lb_terms), sense=pe.minimize)
                obj_form = B.objective_form(g.d0_lb_objective.expr)
                gate('LB objective form linear or convex (0 = yes)', scope_g,
                     0.0 if obj_form['form'] in ('linear', 'convex_quadratic_psd') else 1.0, 0.0)
                group_nonconvex = B.nonconvex_active_rows(g)
                inventory_out['nonconvex_after'][f'{label}/{gkey}/{var}/GROUP'] = group_nonconvex
                gate('nonconvex rows in the assembled group (count)', scope_g, float(len(group_nonconvex)), 0.0)
                if group_nonconvex:
                    stops.append({'stop': 'nonconvex row remains', 'scope': f'{scope_g} GROUP',
                                  'rows': group_nonconvex[:20]})
                obj_cert = {}
                for blabel, w, expr in lb_terms:
                    v_ = w * float(pe.value(expr))
                    pr = prod_g[blabel]['gross_weighted']
                    obj_cert[blabel] = {'relaxed_term': v_, 'production_gross_weighted': pr,
                                        'rel': abs(v_ - pr) / abs(pr) if pr else abs(v_ - pr)}
                lb_group_cert = float(pe.value(g.d0_lb_objective))

                lp_rec = None
                if not group_nonconvex:
                    lp_rel = os.path.join('lp', label, f'{gkey}_{mode}.lp')
                    lp_path = os.path.join(OUT_DIR, lp_rel)
                    os.makedirs(os.path.dirname(lp_path), exist_ok=True)
                    if os.path.exists(lp_path):
                        raise RuntimeError(f'refusing to overwrite {lp_path}')
                    lp_sha, nm, nm_counts = B.export_lp(g, lp_path)
                    nm_rel = os.path.join('lp', label, f'{gkey}_{mode}.name_map.json')
                    write_json(nm_rel, {'lp': lp_rel, 'lp_sha256': lp_sha, 'name_map': nm}, compact=True)
                    lp_rec = {'path': lp_rel, 'sha256': lp_sha, 'bytes': os.path.getsize(lp_path), 'counts': nm_counts,
                              'name_map': nm_rel, 'name_map_sha256': sha256_file(os.path.join(OUT_DIR, nm_rel)),
                              'cell': label, 'eval_dir': cell['eval_dir'], 'year': y, 'day': d, 'mode': mode,
                              'variant': var, 'variant_semantics': VARIANT_SEMANTICS[var],
                              'composition': ['tso'] + [f'dso_{n}' for n in nodes] + [f'esso_{n}' for n in nodes],
                              'coupling_families': [f'd0_cpl_{ch}' for ch in channel_of],
                              'objective': ('d0_lb_objective = sum_b w_b * gross_b; gross convention, ESSO term 0'),
                              'objective_form': obj_form}
                    lp_index['lp_files'][lp_rel] = lp_rec
                    lp_paths[(gkey, var)] = lp_path

                # ---- mapped point 'ef' (exact rank-1); restated gates ----------------------------------------------
                map_out = {}
                env_out = {}
                for blabel, n, m in blocks:
                    sub = g.tso if blabel == 'TSO' else g.find_component(f'dso_{n}')
                    tables = B.mapped_point(m, sub, w_source='ef')
                    same_point = B.production_rows_at_point(m, sub)
                    cap_differs = any(caps[blabel][var][e] != caps[blabel]['A'][e] for e in caps[blabel]['A'])
                    ret = tables['retained']
                    excluded = [r for r in ret if cap_differs and r['references_capacity_params']]
                    gated = [r for r in ret if not (cap_differs and r['references_capacity_params'])]
                    thr = lambda r: r['first_order_w_bound'] + MULT * r['eval_resolution']
                    over = [r for r in gated if r['delta'] > thr(r)]
                    legacy_thr = lambda r: max(THRESHOLDS['legacy_ruling_10_1_ii_floor'],
                                               THRESHOLDS['legacy_ruling_10_1_ii_multiple'] * r['eval_resolution'])
                    worst = max(gated, key=lambda r: r['delta'])
                    worst_ratio = max(gated, key=lambda r: (r['delta'] / thr(r)) if thr(r) > 0 else (
                        0.0 if r['delta'] == 0 else float('inf')))
                    cones_w = [abs(c['rel_residual']) for c in tables['cones'] if c['kind'] == 'W']
                    cones_t = [abs(c['rel_residual']) for c in tables['cones'] if c['kind'] == 'transformer_UCD']
                    defin = {}
                    for r in tables['replaced_production']:
                        if r['family'] in B.DEFINITIONAL_FAMILIES:
                            v_ = max(abs(x) for x in (r['prod']['r_lower'], r['prod']['r_upper']) if x is not None)
                            defin[r['family']] = max(defin.get(r['family'], 0.0), v_)
                    group_out['mapping_offset'][blabel] = defin
                    env_rows = B.envelope_first_order_table(m, sub) if tables['envelope'] else []
                    env_over = [r for r in env_rows if not r['within']]
                    env_worst = max(env_rows, key=lambda r: r['delta']) if env_rows else None
                    env_out[blabel] = {'n_rows': len(env_rows), 'n_over_restated': len(env_over),
                                       'max_delta': env_worst['delta'] if env_worst else None,
                                       'argmax': env_worst, 'max_delta_over_threshold': max(
                                           (r['delta'] / r['threshold'] for r in env_rows if r['threshold'] > 0),
                                           default=None),
                                       'n_over_legacy_1e-9': sum(1 for r in env_rows
                                                                 if r['delta'] > THRESHOLDS['legacy_envelope_abs']),
                                       'rows_over_restated': env_over,
                                       'rows_over_legacy_1e-9': [r for r in env_rows
                                                                 if r['delta'] > THRESHOLDS['legacy_envelope_abs']]}
                    map_out[blabel] = {
                        'same_point': same_point,
                        'mapping_offset_definitional_residuals_max_abs': defin,
                        'retained_ef': {
                            'n_rows': len(ret), 'n_gated': len(gated),
                            'n_zero': sum(1 for r in gated if r['delta'] == 0.0),
                            'max_delta': worst['delta'],
                            'argmax_delta': {k_: worst[k_] for k_ in ('family', 'index', 'delta', 'eval_resolution',
                                                                      'first_order_w_bound')},
                            'max_delta_over_restated_threshold': (worst_ratio['delta'] / thr(worst_ratio)
                                                                  if thr(worst_ratio) > 0 else
                                                                  (0.0 if worst_ratio['delta'] == 0 else None)),
                            'argmax_ratio': {k_: worst_ratio[k_] for k_ in ('family', 'index', 'delta',
                                                                            'eval_resolution', 'first_order_w_bound')},
                            'n_over_restated': len(over),
                            'rows_over_restated': [{k_: r[k_] for k_ in ('family', 'index', 'delta', 'eval_resolution',
                                                                         'first_order_w_bound')} for r in over],
                            'n_over_legacy_ruling_10_1_ii': sum(1 for r in gated if r['delta'] > legacy_thr(r)),
                            'n_over_legacy_stop_1e-10': sum(1 for r in gated
                                                            if r['delta'] > THRESHOLDS['legacy_stop_abs']),
                            'excluded_capacity_param_rows': {
                                'reason': ('rows referencing shared_es_s/e_rated_fixed are not identities when the '
                                           'variant capacity differs from the carried one'),
                                'n': len(excluded),
                                'max_delta': max((r['delta'] for r in excluded), default=0.0)}},
                        'cones_ef': {'n_W': len(cones_w), 'max_rel_W': max(cones_w) if cones_w else None,
                                     'n_transformer': len(cones_t),
                                     'max_rel_transformer': max(cones_t) if cones_t else None},
                        'envelope_bounds_min_rel_slack': min(
                            (min(b['lower_slack_rel'], b['upper_slack_rel']) for b in tables['envelope_bounds']),
                            default=None)}
                    sc = f'{scope_g} {blabel}'
                    gate('same-point Delta (exact 0)', sc, same_point['max_abs_delta'], 0.0, rule='==')
                    gate('retained rows over |Delta| <= first-order W bound + 4 eps sum|terms| (count)', sc,
                         float(len(over)), 0.0)
                    gate('cone rel (W) at ef', sc, max(cones_w), THRESHOLDS['cone_rel'])
                    if cones_t:
                        gate('cone rel (transformer UCD) at ef', sc, max(cones_t), THRESHOLDS['cone_rel'])
                    if env_rows:
                        gate('envelope rows over |Delta| <= first-order bound + 4 eps sum|terms| (count)', sc,
                             float(len(env_over)), 0.0)
                    if over:
                        stops.append({'stop': 'retained row exceeds the restated gate', 'scope': sc,
                                      'rows': map_out[blabel]['retained_ef']['rows_over_restated'][:20]})
                    if env_over:
                        stops.append({'stop': 'envelope row exceeds the restated gate', 'scope': sc,
                                      'rows': env_over[:20]})
                identities_out['envelope_first_order'][label][f'{gkey}_{var}'] = env_out
                for n in nodes:
                    B._set_esso_mapped_point(g.find_component(f'esso_{n}'))
                obj_ef = {}
                for blabel, w, expr in lb_terms:
                    v_ = w * float(pe.value(expr))
                    pr = prod_g[blabel]['gross_weighted']
                    obj_ef[blabel] = {'relaxed_term': v_, 'rel': abs(v_ - pr) / abs(pr) if pr else abs(v_ - pr)}
                mode_out['mapped'] = map_out
                mode_out['objective'] = {'certified': obj_cert, 'mapped_ef': obj_ef,
                                         'group_relaxed_objective_certified': lb_group_cert,
                                         'group_production_gross': prod_g_sum,
                                         'group_rel_certified': abs(lb_group_cert - prod_g_sum) / abs(prod_g_sum)}
                mode_out['lp'] = lp_rec
                for blabel in obj_cert:
                    gate('objective identity per block, certified (rel)', f'{scope_g} {blabel}',
                         obj_cert[blabel]['rel'], THRESHOLDS['objective_rel'])
                    gate('objective identity per block, mapped ef (rel)', f'{scope_g} {blabel}',
                         obj_ef[blabel]['rel'], THRESHOLDS['objective_rel'])
                gate('objective identity per group, certified (rel)', scope_g,
                     mode_out['objective']['group_rel_certified'], THRESHOLDS['objective_rel'])
                lb_sum[mode] += lb_group_cert
                print(f'[group] {scope_g}: same-point max '
                      f'{max(map_out[b]["same_point"]["max_abs_delta"] for b in map_out):.1e}; retained over '
                      f'{sum(map_out[b]["retained_ef"]["n_over_restated"] for b in map_out)} (max ratio '
                      f'{max((map_out[b]["retained_ef"]["max_delta_over_restated_threshold"] or 0.0) for b in map_out):.3f}); '
                      f'envelope over {sum(env_out[b]["n_over_restated"] for b in env_out)} (max ratio '
                      f'{max((env_out[b]["max_delta_over_threshold"] or 0.0) for b in env_out):.4f}); cones '
                      f'{max(map_out[b]["cones_ef"]["max_rel_W"] for b in map_out):.1e}; coupling unequal '
                      f'{len(cpl_bad)}; obj rel {mode_out["objective"]["group_rel_certified"]:.1e}; '
                      f'lp {lp_rec["counts"] if lp_rec else None}')
                del g, relaxed
                gc.collect()

            # ---- variant comparisons on the exported LPs (C2), and against T0.1b --------------------------------------
            cmp_g = {}
            if all((gkey, v) in lp_paths for v in ('R', 'A', 'C')):
                cmp_g['R_vs_A'] = _compare_lp(lp_paths[(gkey, 'R')], lp_paths[(gkey, 'A')])
                cmp_g['R_vs_C'] = _compare_lp(lp_paths[(gkey, 'R')], lp_paths[(gkey, 'C')])
                cmp_g['A_vs_C'] = _compare_lp(lp_paths[(gkey, 'A')], lp_paths[(gkey, 'C')])
                for v, mode in (('R', 'rated'), ('A', 'available')):
                    t01b_path = os.path.join(REPO, T01B_ROOT, 'lp', label, f'{gkey}_{mode}.lp')
                    t01b_sha = sha256_file(t01b_path)
                    cmp_g[f'{v}_vs_t01b_{v}'] = _compare_lp(t01b_path, lp_paths[(gkey, v)])
                    cmp_g[f'{v}_vs_t01b_{v}']['t01b_sha256_ok'] = (
                        t01b_sha == t01b_lp_index[f'lp/{label}/{gkey}_{mode}.lp']['sha256'])
            lp_index['comparisons'][f'{label}/{gkey}'] = cmp_g
            print(f'[group] {label} {gkey} done in {time.time() - t_group:.1f} s; LP compare: '
                  + '; '.join(f'{k_} raw={v["raw_identical"]} body={v["body_identical"]} '
                              f'ndiff={v.get("n_differing_body_lines")}' for k_, v in cmp_g.items()))

        # ---- cell sums ---------------------------------------------------------------------------------------------
        objective_cell['sum_over_groups'] = {
            'n_groups': len(groups), 'relaxed_objective_certified': {VARIANT[m_]: lb_sum[m_] for m_ in MODES},
            'rel_vs_recorded_gross': {VARIANT[m_]: abs(lb_sum[m_] - recorded_gross) / abs(recorded_gross)
                                      for m_ in MODES}}
        if len(groups) == len(years) * len(days):
            for m_ in MODES:
                gate('12-group LB-objective sum at the certified point vs recorded gross (rel)',
                     f'{label} {VARIANT[m_]}', objective_cell['sum_over_groups']['rel_vs_recorded_gross'][
                         VARIANT[m_]], THRESHOLDS['objective_rel'])
        for (y, d) in groups:
            objective_cell['per_group'][f'{y}_{d}'] = {
                VARIANT[m_]: identities_out['groups'][label][f'{y}_{d}']['modes'][VARIANT[m_]]['objective']
                for m_ in MODES}
        objective_out['cells'][label] = objective_cell
        tot = {'gross': sum(g_['gross'] for g_ in q_plus_cell['per_group'].values()),
               'correction_path1': sum(g_['correction_path1'] for g_ in q_plus_cell['per_group'].values()),
               'correction_path2': sum(g_['correction_path2'] for g_ in q_plus_cell['per_group'].values())}
        tot['q_cert_plus'] = tot['gross'] + tot['correction_path1']
        tot['recorded_gross'] = recorded_gross
        tot['recorded_detector_penalty_total'] = recorded_detector['detector_penalty_total']
        tot['minus_recorded_detector_total'] = -recorded_detector['detector_penalty_total']
        tot['correction_minus_neg_detector_total'] = tot['correction_path1'] + recorded_detector[
            'detector_penalty_total']
        fam_tot = {}
        for rec in q_plus_cell['per_block'].values():
            for fname, f in rec['unweighted']['families'].items():
                ft = fam_tot.setdefault(fname, {'certified_weighted': 0.0, 'correction_weighted': 0.0,
                                                'positive_part_weighted': 0.0, 'n_negative': 0})
                ft['certified_weighted'] += rec['weight'] * f['certified']
                ft['correction_weighted'] += rec['weight'] * f['correction']
                ft['positive_part_weighted'] += rec['weight'] * f['positive_part']
                ft['n_negative'] += f['n_negative']
        tot['per_detector_family'] = fam_tot
        tot['n_slack_above_ub'] = sum(r['unweighted']['n_above_ub'] for r in q_plus_cell['per_block'].values())
        tot['max_slack_above_ub'] = max(r['unweighted']['max_above_ub'] for r in q_plus_cell['per_block'].values())
        tot['predicted_correction'] = PREDICTED_CORRECTION[label]
        tot['correction_rounded_to_cent'] = round(tot['correction_path1'], 2)
        q_plus_cell['cell'] = tot
        q_plus_cells[label] = tot
        objective_out['q_cert_plus'][label] = q_plus_cell
        print(f'[Q+] {label}: gross {tot["gross"]!r}; correction {tot["correction_path1"]!r} (path 2 '
              f'{tot["correction_path2"]!r}); Q_cert+ {tot["q_cert_plus"]!r}; -recorded detector total '
              f'{tot["minus_recorded_detector_total"]!r}')

        fp_after = {}
        for y, d in [(y, d) for y in years for d in days]:
            for blabel, _n, m in live_blocks(y, d):
                fp_after[f'{y}_{d}_{blabel}'] = B.model_fingerprint(m)
        for n in nodes:
            fp_after[f'ESSO{n}'] = B.model_fingerprint(esso_live[n])
        changed = sorted(k_ for k_ in fp_before if fp_before[k_] != fp_after[k_])
        identities_out['fingerprints'][label] = {'n_models': len(fp_before), 'changed': changed}
        gate('live blocks not mutated (changed count)', label, float(len(changed)), 0.0)
        for path, name in ((cm_path, 'certified_models.pkl'), (es_path, 'esso_models_s39_D.pkl')):
            gate('input pickle unchanged at end (0 = yes)', f'{label} {name}',
                 0.0 if sha256_file(path) == cell['sha256'][name] else 1.0, 0.0)
        del certified, tso_live, dso_live, esso_live, models_live
        gc.collect()
        print(f'[cell] {label} done in {time.time() - t_cell:.1f} s')

    # ==================================================================================================================
    #  Guard, launches, containment
    # ==================================================================================================================
    guard_failures = guard.verify(0)
    guard.uninstall()
    solver_libraries = {name: any(m_ == name or m_.startswith(name + '.') for m_ in sys.modules)
                        for name in ('gurobipy', 'highspy', 'clarabel', 'cplex', 'docplex', 'xpress', 'mosek',
                                     'cyipopt', 'scs', 'ecos', 'osqp')}
    guard_out = {'label': guard.label, 'permitted': list(guard.permitted), 'counts': dict(guard.counts),
                 'declared_solves': 0, 'verify_failures': guard_failures,
                 'process_launch_events': _AUDIT['process_launch_events'],
                 'solver_libraries_imported': solver_libraries}
    gate('guard: zero solves (failures count)', 'process', float(len(guard_failures)), 0.0)
    gate('process launches in the child (audit hook)', 'process', float(len(_AUDIT['process_launch_events'])), 0.0)
    gate('solver libraries imported (count)', 'process', float(sum(solver_libraries.values())), 0.0)
    out_abs, repo_abs = os.path.abspath(OUT_DIR), os.path.abspath(REPO)
    writes_inside, writes_repo, writes_pycache, writes_outside_repo = [], [], [], []
    for path, _count in sorted(_AUDIT['write_opens'].items()):
        if path.startswith(out_abs + os.sep) or path == out_abs:
            writes_inside.append(path)
        elif '__pycache__' in path.split(os.sep):
            writes_pycache.append(path)
        elif path.startswith(repo_abs + os.sep):
            writes_repo.append(path)
        else:
            writes_outside_repo.append(path)
    fs_outside = [e for e in _AUDIT['fs_mutations']
                  if any(not (p.startswith(out_abs + os.sep) or p == out_abs or '__pycache__' in p.split(os.sep))
                         for p in e['paths'])]
    containment = {'method': 'sys.addaudithook: every open() for writing and every filesystem mutation event of the '
                             'child process (other processes are not seen)',
                   'n_write_opens_inside_out_dir': len(writes_inside),
                   'write_opens_in_repo_outside_out_dir': [rel(p) for p in writes_repo],
                   'write_opens_pycache': [rel(p) if p.startswith(repo_abs) else p for p in writes_pycache],
                   'write_opens_outside_repo': writes_outside_repo,
                   'fs_mutation_events_outside_out_dir': fs_outside,
                   'side_writes_redirected_to': rel(SIDE_WRITES_DIR)}
    gate('containment: write-opens in the repository outside OUT_DIR (count)', 'process', float(len(writes_repo)), 0.0)

    # ==================================================================================================================
    #  Predictions vs outcomes
    # ==================================================================================================================
    ret_over = sum(identities_out['groups'][lab][gk]['modes'][v]['mapped'][b]['retained_ef']['n_over_restated']
                   for lab in identities_out['groups'] for gk in identities_out['groups'][lab]
                   for v in identities_out['groups'][lab][gk]['modes']
                   for b in identities_out['groups'][lab][gk]['modes'][v]['mapped'])
    env_over_n = sum(e_['n_over_restated'] for lab in identities_out['envelope_first_order']
                     for gm in identities_out['envelope_first_order'][lab].values() for e_ in gm.values())
    env_rows_n = sum(e_['n_rows'] for lab in identities_out['envelope_first_order']
                     for gm in identities_out['envelope_first_order'][lab].values() for e_ in gm.values())
    n_block_modes = sum(len(identities_out['groups'][lab][gk]['modes'][v]['mapped'])
                        for lab in identities_out['groups'] for gk in identities_out['groups'][lab]
                        for v in identities_out['groups'][lab][gk]['modes'])
    c2 = {}
    for key, cmp_g in lp_index['comparisons'].items():
        if key.startswith('x0/') and cmp_g:
            c2[key] = {p: {'raw_identical': cmp_g[p]['raw_identical'], 'body_identical': cmp_g[p]['body_identical'],
                           'n_differing_body_lines': cmp_g[p].get('n_differing_body_lines')}
                       for p in ('R_vs_A', 'R_vs_C', 'A_vs_C')}
    c3 = {lab: {'correction': q_plus_cells[lab]['correction_path1'],
                'correction_to_cent': q_plus_cells[lab]['correction_rounded_to_cent'],
                'predicted': PREDICTED_CORRECTION[lab],
                'held': round(q_plus_cells[lab]['correction_path1'], 2) == PREDICTED_CORRECTION[lab]}
          for lab in q_plus_cells}
    c4_rows = capacity_out['vectors'].get('unit_n7', {}).get('c4_rows', [])
    c4_2025 = [r for r in c4_rows if r['pair'][0] == 0]
    c4_ratio_ok = all(abs(r['ratio'] - 0.985 ** (5 * (r['pair'][1] + 1))) <= 1e-15 for r in c4_2025)
    c5 = {lab: capacity_out['bound_inventory']['summary'][lab] for lab in capacity_out['bound_inventory']['summary']}
    c5_held = all(set(s['value_dependent_families']) == {'slack_shared_es_soc_final_up',
                                                         'slack_shared_es_soc_final_down'}
                  and all(v == 1.0 for v in s['value_dependent_per_copy'].values())
                  and s['esso_value_dependent_bounds'] == 0 for s in c5.values())
    predictions_out = {
        'C1': {'prediction': PREDICTIONS['C1'], 'n_block_modes': n_block_modes,
               'retained_rows_over_restated': ret_over, 'envelope_rows_checked': env_rows_n,
               'envelope_rows_over_restated': env_over_n, 'held': ret_over == 0 and env_over_n == 0},
        'C2': {'prediction': PREDICTIONS['C2'], 'x0_groups': c2,
               'held_raw_bytes': bool(c2) and all(v['raw_identical'] for g_ in c2.values() for v in g_.values()),
               'held_body': bool(c2) and all(v['body_identical'] for g_ in c2.values() for v in g_.values()),
               'held': bool(c2) and all(v['raw_identical'] for g_ in c2.values() for v in g_.values())},
        'C3': {'prediction': PREDICTIONS['C3'], 'per_cell': c3, 'held': bool(c3) and all(v['held'] for v in c3.values())},
        'C4': {'prediction': PREDICTIONS['C4'], 'rows': c4_rows,
               'all_ordered': bool(c4_rows) and all(r['ordered'] for r in c4_rows),
               'ratios_2025_cohort': [r['ratio'] for r in c4_2025], 'ratios_match_0.985^5k': c4_ratio_ok,
               'held': bool(c4_rows) and all(r['ordered'] for r in c4_rows) and c4_ratio_ok and len(c4_2025) == 3},
        'C5': {'prediction': PREDICTIONS['C5'], 'summary': c5, 'held': bool(c5) and c5_held},
    }
    gates_out = {'rows': gate_rows, 'all_pass': all(r['pass'] for r in gate_rows),
                 'failed': [r for r in gate_rows if not r['pass']], 'stops': stops, 'stop_triggered': bool(stops),
                 'thresholds': THRESHOLDS}
    identities_out.update({'gates': gates_out, 'predictions': predictions_out, 'guard': guard_out,
                           'containment': containment, 'wall_time_s': time.time() - t_start})

    payloads = {'inventory.json': inventory_out, 'identities.json': identities_out, 'capacity.json': capacity_out,
                'objective.json': objective_out, 'lp/index.json': lp_index}
    unfilled = [(q, f, k_) for q, (f, k_) in REQUIRED_CAPTURES.items() if not payloads.get(f, {}).get(k_)]
    if unfilled:
        raise RuntimeError(f'capture check: unfilled required quantities {unfilled}')
    for name, payload in payloads.items():
        write_json(name, payload, compact=True)
        print(f'[write] {name}')

    print(f'[summary] gates: {len(gate_rows)} rows, all pass: {gates_out["all_pass"]}; failed gate kinds: '
          f'{sorted({r["gate"] for r in gates_out["failed"]})}')
    print(f'[summary] stops: {len(stops)} ({sorted({s["stop"] for s in stops})})')
    for key, value in predictions_out.items():
        print(f'[prediction] {key}: held={value["held"]}')
    print(f'[summary] guard counts {guard.counts}; process launches {len(_AUDIT["process_launch_events"])}; '
          f'solver libraries imported: {[k_ for k_, v in solver_libraries.items() if v]}; '
          f'repo writes outside OUT_DIR: {len(writes_repo)}; wall {time.time() - t_start:.1f} s')
    if stops:
        return 3
    return 0 if gates_out['all_pass'] else 2


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        sys.exit(child())
    sys.exit(parent())
