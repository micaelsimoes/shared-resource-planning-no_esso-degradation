"""P5.16 D0 / T0.1b -- W-space relaxation on the route-A certified blocks of the two W86 reference cells (ZERO solves).

Governing spec: `P5_16_D0_SPEC_v2.md` section 2 (cells, hashes), section 3 (construction, incl. item 6 coupling rows),
section 4 (gates G-D0-2 / G-D0-3), section 7 (captures) and section 10 (Planner rulings after T0.1a: same-point
identity Delta = 0 primary; mapped-point |Delta| <= max(1e-12, 8 eps sum|terms|); envelope-rewrite rows in the
"replaced" class; `sess_comp` DROPPED as nonconvex; any other nonconvex row is a stop). Builder:
`p516_d0_build_relaxation.py` (extended for ruling 3 with the opt-in `drop_sess_comp=True`).

ZERO solves: `p513_solve_profile_guard.SolveProfileGuard` is armed in BLOCKING form (permitted=()) for the whole child
process before any production module is imported, and verified exactly at 0 / 0 at the end. A `sys.addaudithook`
hook, also installed first, RAISES on any process launch from the child (subprocess / os.system / exec / spawn /
fork) and records every file opened for writing; the git provenance is taken by the parent, before and after.

Usage (one process, attached, alone):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_t01b.py

The parent creates OUT_DIR atomically (refuses if it exists -- which also refuses a concurrent second copy), runs the
child with stdout AND stderr merged line by line into OUT_DIR/run.log (echoed), then writes
OUT_DIR/manifest_sha256.json over every file in OUT_DIR, the log included.

Per cell (x = 0 and unit 7), in order:
  1. sha256 of the four input pickles against spec section 2 -- REFUSE on any mismatch (before anything is loaded);
     the cell's evaluation_record / post_certification / boyd_terminal read and recorded (gross, certified cycle,
     Boyd r / eps_pri per channel).
  2. capture before mutation: the live TSO/DSO blocks of every (year, day) group and the ESSO dict re-pickled under
     captures/ (round-trip fingerprint checked), every Var vector and every mutable Param (ADMM duals and consensus
     targets are mutable Params on the blocks) dumped per group; the inter-copy mismatches per coupling channel read
     from the live blocks in physical units; the recorded Boyd r for the v / pf channels recomputed from the blocks'
     expected_* Vars (provenance check that the pickles are the certified iterate).
  3. inventory per block (T0.1a classes + ruling 3: `sess_comp` -> dropped); after relaxation every active row of
     every relaxed block and group is form-audited; any non-convex row left is listed and is a STOP (no export).
  4. relaxation in both capacity modes (rated = variant R / available = variant A) with the builder; the capacity
     Params carried vs rated. Variant A is an attribution diagnostic, never a lower bound of production (spec
     section 11 D-1); variant C (section 11 D-2, added to the spec during this task) is NOT built here.
  5. the group model (TSO + 3 DSO + 3 ESSO as sub-blocks) with the spec section 3.6 coupling rows (exact
     equalities, s_base / v_base conversions): voltage on vmag_sqr (Wii) TSO ADN node <-> DSO reference node;
     interface P/Q on the TSO's `pc_adn` / `qc_adn` <-> the DSO's `pg_adn` / `qg_adn`; storage P/Q TSO <-> ESSO and
     DSO <-> ESSO on `shared_es_pnet` / `shared_es_qnet` <-> `es_pnet` / `es_qnet`. Coupling residuals at the
     certified point vs the step-2 mismatch, per row, beside the recorded eps_pri.
  6. LB objective per group from the production component functions; identity per block / group against the
     production gross components, and the 12-group sum against the cell's recorded gross.
  7. gates per ruling 1-3: same-point Delta == 0 exactly; mapped-point (rank-1 'ef') |Delta| <= max(1e-12, 8 eps
     sum|terms|); cones <= 1e-12 relative; envelope-rewrite rows <= 1e-9; inventory complete. Diagnostic (not gated):
     the mapped point with W kept at the certified values ('snapshot').
  8. one .lp per group and mode with a name map (T0.2 inputs).
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P516D0', 't01b')
# Rehearsal only: P516_D0_REHEARSAL_DIR=/tmp/... redirects every output to scratch outside the repository, and
# P516_D0_T01B_REHEARSAL_GROUPS=k limits each cell to its first k groups. Any other location is refused; the
# evidence run sets neither.
_REHEARSAL = os.environ.get('P516_D0_REHEARSAL_DIR')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516_D0_REHEARSAL_DIR must be under /tmp, got {_REHEARSAL!r}')
    OUT_DIR = os.path.abspath(_REHEARSAL)
_GROUP_LIMIT = os.environ.get('P516_D0_T01B_REHEARSAL_GROUPS')
if _GROUP_LIMIT and not _REHEARSAL:
    raise SystemExit('P516_D0_T01B_REHEARSAL_GROUPS is a rehearsal-only setting')
GROUP_LIMIT = int(_GROUP_LIMIT) if _GROUP_LIMIT else None

LOG_PATH = os.path.join(OUT_DIR, 'run.log')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
PROVENANCE_PARENT = os.path.join(OUT_DIR, 'provenance_parent.json')
SIDE_WRITES_DIR = os.path.join(OUT_DIR, 'side_writes', 'Diagrams')
SRP1_DIR = os.path.join(REPO, 'data', 'SRP1')
EVALS = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals')

# Spec v2 section 2: cells, certified cycles and input hashes (route A).
CELLS = [
    {'label': 'x0', 'eval_dir': os.path.join(EVALS, '5cfe69a615ae3708_x0'), 'certified_cycle_spec': 132,
     'sha256': {'certified_models.pkl': 'd816bb4b1ade4b44ad4a77872443fe2668986663c524fc4239815d4657974e21',
                'esso_models_s39_D.pkl': '15f384090089e123a295d8a3f5d0d6b2dc4e953effed55b28eae456a38529d8d'}},
    {'label': 'unit_n7', 'eval_dir': os.path.join(EVALS, 'ca8927e75d628bd1_n7_4h_e1'), 'certified_cycle_spec': 112,
     'sha256': {'certified_models.pkl': 'fe06ca485003b4fc9c0fd9788279bd287dd813d659179b311885e8ab03efda04',
                'esso_models_s39_D.pkl': '95f7b38a6ec21833f7b5521ec8332d8022ef6801b4095d1410225438b2cedfc9'}},
]
RECORD_FILES = ('evaluation_record.json', 'post_certification.json', 'boyd_terminal.json')
MODES = ('rated', 'available')
# Spec v2 section 11 (Addendum 88) D-1: variant A is an attribution diagnostic, NEVER a lower bound of production.
VARIANT = {'rated': 'R', 'available': 'A'}
VARIANT_SEMANTICS = {
    'R': ('rated: network copies carry rated capacity (spec section 3 item 3); the section 3 lower-bound '
          'construction'),
    'A': ('available: network copies keep the certified available capacity; an ATTRIBUTION DIAGNOSTIC, NOT a lower '
          'bound of production (spec section 11 D-1). The Pyomo objective name d0_lb_objective is the builder\'s '
          'export convention and does not make A a bound.'),
    'C': 'calendar-only available energy (spec section 11 D-2): NOT built in this run (not in the T0.1b order)',
}

THRESHOLDS = {
    'same_point_abs': 0.0,              # ruling 1 (i): exactly 0
    'mapped_floor_abs': 1e-12,          # ruling 1 (ii): max(1e-12, 8 eps sum|terms|)
    'mapped_eps_multiple': 8.0,
    'stop_abs': 1e-10,                  # task stop rule: a retained-row Delta beyond this
    'cone_rel': 1e-12,
    'envelope_abs': 1e-9,
    'objective_rel': 1e-8,
    'coupling_eps_multiple': 4.0,       # coupling residual vs mismatch: |diff| <= 4 eps (row terms + |a| + |b|)
    'boyd_r_recompute_rel': 1e-9,       # provenance check (pickle == certified iterate)
    'retained_table_floor': 1e-13,      # rows with Delta above this are tabulated individually
}

PREDICTIONS = {
    'Q1': 'no nonconvex row remains after dropping sess_comp on the certified blocks',
    'Q2': 'same-point Delta = 0 on every retained row in all 24 groups',
    'Q3': 'coupling residuals equal the inter-copy mismatches and are <= the recorded Boyd eps_pri per channel',
    'Q4': 'objective identity <= 1e-12 per block and the 12-group sum equals the recorded gross to <= 1e-8',
    'Q5': "the live blocks' e_available at the unit is below rated in 2030 and 2035 (SoH < 1) and equal to rated "
          'at x = 0 (zero capacity)',
}

# Capture-path checklist (CLAUDE.md rule eleven): every quantity the task names -> (output file, top-level key).
# Asserted before any computation (a writer exists for each file) and again before writing (each key filled).
REQUIRED_CAPTURES = {
    'input sha256 vs spec section 2 (refuse on mismatch)': ('inventory.json', 'inputs'),
    'cell records: gross, certified cycle, Boyd r / eps_pri per channel': ('inventory.json', 'cell_records'),
    'live blocks + ESSO pickled before mutation, hashes, round-trip fingerprints': ('inventory.json', 'captures'),
    'certified Var vectors and mutable Params (ADMM duals / consensus targets) per block': ('inventory.json',
                                                                                             'captures'),
    'inter-copy mismatches per channel, physical units': ('coupling.json', 'mismatch_step1'),
    'Boyd r recomputed from the blocks (v, pf) vs recorded': ('coupling.json', 'boyd_recompute'),
    'row inventory per block (class / sub / family / form, counts) incl. sess_comp dropped': ('inventory.json',
                                                                                              'blocks'),
    'nonconvex rows remaining after relaxation (per block and group)': ('inventory.json', 'nonconvex_after'),
    'capacity Params carried (available) vs rated, per block': ('identities.json', 'capacity'),
    'coupling row definitions and residuals at the certified point vs mismatch, beside eps_pri': ('coupling.json',
                                                                                                  'rows'),
    'per-row identity tables (same-point, mapped ef, snapshot-W diagnostic), cones, envelope': ('identities.json',
                                                                                                'groups'),
    'ESSO relaxation identities': ('identities.json', 'esso'),
    'objective identity per block / group / 12-group sum vs recorded gross': ('objective_identity.json', 'cells'),
    'gate verdicts and achieved values': ('identities.json', 'gates'),
    'predictions vs outcomes': ('identities.json', 'predictions'),
    'guard counts, process launches, solver libraries': ('identities.json', 'guard'),
    'input non-mutation fingerprints': ('identities.json', 'fingerprints'),
    'write containment (audit hook)': ('identities.json', 'containment'),
    'LP files with sha256, counts and name maps': ('lp/index.json', 'lp_files'),
    'provenance (interpreter, versions, git head, file hashes)': ('inventory.json', 'provenance'),
}
OUTPUT_FILES = ('predictions.json', 'inventory.json', 'identities.json', 'coupling.json', 'objective_identity.json',
                'lp/index.json')


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
def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as error:          # recorded, never fatal
        return f'ERROR {error!r}'


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
    provenance = {'git_head': _git('rev-parse', 'HEAD'), 'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
                  'tracked_status_before': _git('status', '--porcelain', '--untracked-files=no'),
                  'platform': platform.platform(), 'parent_interpreter': sys.executable,
                  'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    with open(PROVENANCE_PARENT, 'w') as handle:
        json.dump(provenance, handle, indent=1)
    with open(LOG_PATH, 'w', buffering=1) as log:
        log.write(f'# p516_d0_t01b parent start {provenance["started"]} interpreter {sys.executable}\n')
        child = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child'], cwd=REPO,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = child.wait()
        log.write(f'# child exit code {code}; wall {time.time() - started:.1f} s\n')
    after = {'tracked_status_after': _git('status', '--porcelain', '--untracked-files=no')}
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
_LAUNCH_EVENTS = ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork', 'os.forkpty',
                  'os.startfile', 'pty.spawn')
_FS_EVENTS = ('os.rename', 'os.remove', 'os.rmdir', 'os.mkdir', 'shutil.rmtree', 'os.truncate', 'os.link',
              'os.symlink', 'shutil.copyfile', 'shutil.move')


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
        elif event in _LAUNCH_EVENTS:
            _AUDIT['process_launch_events'].append({'event': event, 'args': repr(args)[:300]})
            raise RuntimeError(f'P5.16 D0 T0.1b: process launch blocked in the zero-solve child ({event})')
        elif event in _FS_EVENTS:
            paths = [os.path.abspath(p) for p in (_path(a) for a in args) if p]
            _AUDIT['fs_mutations'].append({'event': event, 'paths': paths})
    sys.addaudithook(hook)


def child():
    _install_audit_hook()
    sys.path.insert(0, REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.16 D0 T0.1b certified-block relaxation (zero solves)').install()
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

    # ---- predictions first, before any computation ------------------------------------------------------------------
    write_json('predictions.json', {'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                    'predictions': PREDICTIONS, 'thresholds': THRESHOLDS,
                                    'note': 'recorded by the harness before any input is loaded'})
    print('[write] predictions.json')

    # ---- capture-path assertion (fail fast) -------------------------------------------------------------------------
    missing = sorted({f for f, _k in REQUIRED_CAPTURES.values() if f not in OUTPUT_FILES})
    if missing:
        raise RuntimeError(f'capture-path assertion failed: no writer for {missing}')
    print(f'[capture] {len(REQUIRED_CAPTURES)} required quantities -> writers present for {sorted(OUTPUT_FILES)}')

    # ---- input hashes: refuse on any mismatch, before anything is loaded --------------------------------------------
    inputs = {}
    mismatch = []
    for cell in CELLS:
        for name, expected in cell['sha256'].items():
            path = os.path.join(REPO, cell['eval_dir'], name)
            actual = sha256_file(path) if os.path.exists(path) else None
            inputs[f'{cell["label"]}/{name}'] = {'path': os.path.join(cell['eval_dir'], name), 'expected': expected,
                                                 'actual': actual, 'match': actual == expected,
                                                 'bytes': os.path.getsize(path) if actual else None}
            print(f'[inputs] {cell["label"]:8s} {name:24s} {actual} match={actual == expected}')
            if actual != expected:
                mismatch.append(f'{cell["label"]}/{name}')
    if mismatch:
        write_json('inventory.json', {'inputs': inputs, 'refused': True})
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
    import shared_resources_planning as srp
    from shared_resources_planning import SharedResourcesPlanning

    EPS = float(np.finfo(float).eps)

    with open(PROVENANCE_PARENT) as handle:
        provenance = json.load(handle)
    files_hashed = ['p516_d0_t01b.py', 'p516_d0_build_relaxation.py', 'p516_d0_dry_run.py',
                    'model_construction_helpers.py', 'network.py', 'network_data.py', 'shared_resources_planning.py',
                    'shared_energy_storage_data.py', 'p513_solve_profile_guard.py', 'gate_result_io.py',
                    'P5_16_D0_SPEC_v2.md', 'P5_16_D0_T01A_REPORT.md', 'data/SRP1/SRP1.json',
                    'data/SRP1/SRP1_params.json', 'data/SRP1/SharedESS/SRP1_ESS_Params.json']
    provenance.update({'interpreter': sys.executable, 'python': sys.version, 'pyomo': pyomo.version.version,
                       'numpy': np.__version__,
                       'file_sha256': {f: sha256_file(os.path.join(REPO, f)) for f in files_hashed},
                       'out_dir': rel(OUT_DIR), 'rehearsal': bool(_REHEARSAL), 'group_limit': GROUP_LIMIT})

    # ---- planning object (weights, normalizations); production's diagram writes redirected into OUT_DIR ----------
    os.makedirs(SIDE_WRITES_DIR)
    planning = SharedResourcesPlanning(SRP1_DIR, 'SRP1.json')
    planning.diagrams_dir = SIDE_WRITES_DIR
    planning.read_planning_problem()
    tn_data = planning.transmission_network
    admm_params = planning.params.admm
    reference_mva = srp._admm_shared_ess_reference_mva(admm_params)
    floor_mva = admm_params.shared_ess_normalization_floor_mva
    esso_years = list(planning.shared_ess_data.years)
    esso_days = list(planning.shared_ess_data.days)
    provenance['srp1_scenario_checksum'] = (planning.scenario_metadata.get('combined_scenario_checksum')
                                            if isinstance(getattr(planning, 'scenario_metadata', None), dict)
                                            else None)
    provenance['admm_normalization'] = {'shared_ess_reference_rating_mva': reference_mva,
                                        'shared_ess_normalization_floor_mva': floor_mva}
    print(f'[planning] read in {time.time() - t_start:.1f} s; years {list(tn_data.years)} days {list(tn_data.days)}; '
          f'ESSO years {esso_years} days {esso_days}; S_ref {reference_mva} MVA')

    gate_rows = []
    stops = []

    def gate(name, scope, achieved, threshold, rule='<=', note=None):
        achieved = float(achieved)
        ok = bool(achieved <= threshold) if rule == '<=' else bool(achieved == threshold)
        gate_rows.append({'gate': name, 'scope': scope, 'achieved': achieved, 'threshold': threshold, 'rule': rule,
                          'pass': ok, 'note': note})
        if not ok:
            print(f'[gate FAIL] {name} {scope}: achieved {achieved:.6e} {rule} {threshold:.1e}')
        return ok

    def var_vector(model):
        out = {}
        for v in model.component_data_objects(pe.Var, descend_into=True, sort=True):
            out[v.name] = [v.value, bool(v.fixed), v.lb, v.ub]
        return out

    def mutable_params(model):
        out = {}
        for param in model.component_objects(pe.Param, descend_into=True, sort=True):
            if not param.mutable:
                continue
            for index in sorted(param.keys(), key=repr):
                out[f'{param.name}[{index!r}]'] = pe.value(param[index], exception=False)
        return out

    def form_audit_ok(rows):
        return not rows

    inventory_out = {'provenance': provenance, 'inputs': inputs, 'cell_records': {}, 'captures': {}, 'blocks': {},
                     'nonconvex_after': {}}
    identities_out = {'capacity': {}, 'groups': {}, 'esso': {}, 'fingerprints': {}}
    coupling_out = {'definitions': None, 'mismatch_step1': {}, 'boyd_recompute': {}, 'rows': {}, 'summary': {}}
    objective_out = {'variant_semantics': VARIANT_SEMANTICS,
                     'convention': ('gross_operational_cost (settlement-EXCLUDED): per block w_b * '
                                    '[objective_function_rule(b) - contracted settlement_b - voltage pin_b], '
                                    'w_b = _get_admm_block_weight; the ESSO contributes 0 (gross excludes it); '
                                    'salvage NOT subtracted (gross, not net)'), 'cells': {}}
    lp_index = {'lp_files': {}, 'variant_semantics': VARIANT_SEMANTICS}

    coupling_definitions = {
        'v_W': {'row': 'tso.vmag_sqr[i_adn] - (vb_dso / vb_tso)^2 * dso_n.vmag_sqr[i_ref] == 0',
                'units_row': 'pu^2 on the TSO voltage base', 'to_physical': 'x vb_tso^2 -> kV^2',
                'tso_side': 'vmag_sqr at network.get_node_idx(node_id) (TSO ADN node), network.py l.290',
                'dso_side': 'vmag_sqr at the DSO reference node, network.py l.290',
                'boyd_units': 'magnitude-equivalent dV = dW / (sqrt(W_tso) + sqrt(W_dso)) (exact), pu of vb_tso; '
                              'Boyd v channel: r_v = (x_dso - z_tso) / v_base on vmag (SRP l.7208)'},
        'pf_p': {'row': 'tso.pc_adn[dn] - (sb_dso / sb_tso) * dso_n.pg_adn == 0',
                 'units_row': 'pu on the TSO s_base', 'to_physical': 'x sb_tso -> MW',
                 'tso_side': 'pc_adn: network.py l.440 (is_transmission branch), rule '
                             'interface_pf_p_transmission_def MCH l.1241 = pc[adn_load] + interface_delta_p (+ flex '
                             'legs fixed 0 in the ADMM path, SRP l.4266-4300). The TSO block has no pg_adn '
                             '(network.py l.444 is the else-branch, DSO only).',
                 'dso_side': 'pg_adn: network.py l.444, rule interface_pf_p_distribution_def MCH l.1287 = '
                             'pg[ref_gen] - shared_es_pnet at the reference node',
                 'boyd_units': 'dP_MW / interface_rating_MVA (SRP l.7231)'},
        'pf_q': {'row': 'tso.qc_adn[dn] - (sb_dso / sb_tso) * dso_n.qg_adn == 0',
                 'units_row': 'pu on the TSO s_base', 'to_physical': 'x sb_tso -> Mvar',
                 'tso_side': 'qc_adn: network.py l.441, MCH l.1265', 'dso_side': 'qg_adn: network.py l.445, MCH l.1300',
                 'boyd_units': 'dQ_Mvar / interface_rating_MVA'},
        'ess_p_tso_esso': {'row': 'tso.shared_es_pnet[e_tso, na, p] - es_pnet[y, d, p] / sb_tso == 0',
                           'units_row': 'pu on the TSO s_base', 'to_physical': 'x sb_tso -> MW',
                           'boyd_units': 'a * dP_MW, a = 1 / (2 S_norm), S_norm = '
                                         '_shared_ess_admm_normalization_mva (SRP l.7174-7182)'},
        'ess_p_dso_esso': {'row': 'dso_n.shared_es_pnet[e_dso, na, p] - es_pnet[y, d, p] / sb_dso == 0',
                           'units_row': 'pu on the DSO s_base', 'to_physical': 'x sb_dso -> MW',
                           'boyd_units': 'a * dP_MW'},
        'ess_q_tso_esso': {'row': 'tso.shared_es_qnet[e_tso, na, p] - es_qnet[y, d, p] / sb_tso == 0',
                           'units_row': 'pu on the TSO s_base', 'to_physical': 'x sb_tso -> Mvar',
                           'boyd_units': 'a * dQ_Mvar'},
        'ess_q_dso_esso': {'row': 'dso_n.shared_es_qnet[e_dso, na, p] - es_qnet[y, d, p] / sb_dso == 0',
                           'units_row': 'pu on the DSO s_base', 'to_physical': 'x sb_dso -> Mvar',
                           'boyd_units': 'a * dQ_Mvar'},
        'note': ('exact equalities, no AL or proximal terms. TSO <-> DSO storage mismatch (implied by the two rows) is '
                 'reported in mismatch_step1 as ess_*_tso_dso. Single scenario asserted on every block (with one '
                 'scenario the expected_* consensus Vars production couples equal these quantities through active '
                 'equality rows, probability 1). The DSO reference f box (+-1e-5) has no W counterpart (dropped, '
                 'rotation invariance, spec section 3 item 6). The voltage_mag_def rows are deactivated, so vmag and '
                 'the expected_interface_vmag_def rows no longer bind W; the voltage coupling is on Wii.'),
    }
    coupling_out['definitions'] = coupling_definitions
    channel_of = {'v_W': 'v', 'pf_p': 'pf', 'pf_q': 'pf', 'ess_p_tso_esso': 'ess', 'ess_p_dso_esso': 'ess',
                  'ess_q_tso_esso': 'ess', 'ess_q_dso_esso': 'ess'}

    def single_scenario(model):
        return len(list(model.scenarios_market)) == 1 and len(list(model.scenarios_operation)) == 1

    # ==================================================================================================================
    #  Per cell
    # ==================================================================================================================
    for cell in CELLS:
        label = cell['label']
        cell_dir = os.path.join(REPO, cell['eval_dir'])
        t_cell = time.time()
        print(f'\n========== cell {label} ({cell["eval_dir"]}) ==========')

        # ---- records ------------------------------------------------------------------------------------------------
        records = {}
        for name in RECORD_FILES:
            path = os.path.join(cell_dir, name)
            with open(path) as handle:
                records[name] = json.load(handle)
            records[name + '__sha256'] = sha256_file(path)
        ev, post, boyd = (records['evaluation_record.json'], records['post_certification.json'],
                          records['boyd_terminal.json'])
        recorded_gross = float(ev['terminal_gross_operational_cost'])
        boyd_channels = {ch: {k: boyd['binding_test_per_channel'][ch][k]
                              for k in ('r', 's', 'eps_pri', 'eps_dual', 'primal_ratio', 'dual_ratio')}
                         for ch in ('v', 'pf', 'ess')}
        cell_record = {
            'eval_dir': cell['eval_dir'], 'candidate_key': ev.get('candidate_key'),
            'candidate_canonical': ev.get('candidate_canonical'), 'candidate_label': ev.get('candidate_label'),
            'certified_cost_post_certification': post['certified_cost'],
            'terminal_gross_operational_cost': ev['terminal_gross_operational_cost'],
            'terminal_net_operational_recourse': ev['terminal_net_operational_recourse'],
            'terminal_salvage_value': ev['recourse_components']['terminal_salvage_value'],
            'objective_convention': ev['objective_convention'],
            'certification_cycle': ev['certification_cycle'], 'cycles_run': ev['cycles_run'],
            'certification_cycle_spec': cell['certified_cycle_spec'],
            'bar': ev['bar']['value'], 'rule_ten': ev['rule_ten'],
            'boyd_per_channel_terminal': boyd_channels, 'boyd_terminal_cycles': boyd.get('cycles'),
            'shared_ess_reference_rating_mva_recorded': boyd.get('shared_ess_reference_rating_mva'),
            'record_sha256': {n: records[n + '__sha256'] for n in RECORD_FILES},
        }
        inventory_out['cell_records'][label] = cell_record
        gate('certified cycle == spec section 2 (0 = equal)', label,
             0.0 if ev['certification_cycle'] == cell['certified_cycle_spec'] else 1.0, 0.0)
        gate('S_ref in force == boyd_terminal record (0 = equal)', label,
             0.0 if float(reference_mva) == float(boyd.get('shared_ess_reference_rating_mva')) else 1.0, 0.0)
        print(f'[record] {label}: gross {recorded_gross!r}; certified cycle {ev["certification_cycle"]}; '
              f'eps_pri v/pf/ess {[boyd_channels[c]["eps_pri"] for c in ("v", "pf", "ess")]}')

        # ---- load (hash re-verified after load) ----------------------------------------------------------------------
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
        if years != list(tn_data.years) or days != list(tn_data.days) or years != esso_years or days != esso_days:
            raise RuntimeError(f'year/day ordering differs: blocks {years} {days}, planning {list(tn_data.years)} '
                               f'{list(tn_data.days)}, ESSO {esso_years} {esso_days}')
        if nodes != sorted(planning.active_distribution_network_nodes) or sorted(esso_live) != nodes:
            raise RuntimeError(f'node sets differ: dso {nodes}, ESSO {sorted(esso_live)}, planning '
                               f'{planning.active_distribution_network_nodes}')
        groups = [(y, d) for y in years for d in days]
        if GROUP_LIMIT:
            groups = groups[:GROUP_LIMIT]
        print(f'[load] {label}: {len(years)} years x {len(days)} days, DSO nodes {nodes}, ESSO nodes '
              f'{sorted(esso_live)}; groups processed {len(groups)}')

        def live_blocks(y, d):
            out = [('TSO', None, tso_live[y][d])]
            out += [(f'DSO{n}', n, dso_live[n][y][d]) for n in nodes]
            return out

        for y, d in [(y, d) for y in years for d in days]:
            for blabel, _n, m in live_blocks(y, d):
                if not single_scenario(m):
                    raise RuntimeError(f'{label} {y} {d} {blabel}: more than one scenario; spec section 3.6 coupling '
                                       f'on pc_adn/pg_adn is production coupling only at one scenario')

        # ---- fingerprints before -----------------------------------------------------------------------------------
        fp_before = {}
        for y, d in [(y, d) for y in years for d in days]:
            for blabel, _n, m in live_blocks(y, d):
                fp_before[f'{y}_{d}_{blabel}'] = B.model_fingerprint(m)
        for n in nodes:
            fp_before[f'ESSO{n}'] = B.model_fingerprint(esso_live[n])

        # ---- captures before mutation -------------------------------------------------------------------------------
        cap_dir = os.path.join(OUT_DIR, 'captures', label)
        os.makedirs(cap_dir)
        capture_rec = {'groups': {}, 'esso': None}
        for y, d in [(y, d) for y in years for d in days]:
            payload = {'tso': tso_live[y][d], 'dso': {n: dso_live[n][y][d] for n in nodes},
                       'metadata': {'cell': label, 'eval_dir': cell['eval_dir'], 'year': y, 'day': d,
                                    'source': 'certified_models.pkl', 'source_sha256': cell['sha256'][
                                        'certified_models.pkl']}}
            path = os.path.join(cap_dir, f'blocks_{y}_{d}.pkl')
            with open(path, 'wb') as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path, 'rb') as handle:
                back = pickle.load(handle)
            rt = {'TSO': B.model_fingerprint(back['tso']) == fp_before[f'{y}_{d}_TSO']}
            for n in nodes:
                rt[f'DSO{n}'] = B.model_fingerprint(back['dso'][n]) == fp_before[f'{y}_{d}_DSO{n}']
            del back
            vectors = {blabel: {'vars': var_vector(m), 'mutable_params': mutable_params(m)}
                       for blabel, _n, m in live_blocks(y, d)}
            vpath = write_json(os.path.join('captures', label, f'vectors_{y}_{d}.json'), vectors, compact=True)
            capture_rec['groups'][f'{y}_{d}'] = {'pickle': rel(path), 'sha256': sha256_file(path),
                                                 'bytes': os.path.getsize(path), 'round_trip_fingerprint_equal': rt,
                                                 'vectors': rel(vpath), 'vectors_sha256': sha256_file(vpath),
                                                 'n_vars': {b: len(v['vars']) for b, v in vectors.items()}}
            gate('capture round-trip fingerprint (0 = equal)', f'{label} {y} {d}', 0.0 if all(rt.values()) else 1.0,
                 0.0)
            del vectors
        path = os.path.join(cap_dir, 'esso_models.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(esso_live, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path, 'rb') as handle:
            back = pickle.load(handle)
        rt = {f'ESSO{n}': B.model_fingerprint(back[n]) == fp_before[f'ESSO{n}'] for n in nodes}
        del back
        evectors = {f'ESSO{n}': {'vars': var_vector(esso_live[n]), 'mutable_params': mutable_params(esso_live[n])}
                    for n in nodes}
        vpath = write_json(os.path.join('captures', label, 'vectors_esso.json'), evectors, compact=True)
        capture_rec['esso'] = {'pickle': rel(path), 'sha256': sha256_file(path), 'bytes': os.path.getsize(path),
                               'round_trip_fingerprint_equal': rt, 'vectors': rel(vpath),
                               'vectors_sha256': sha256_file(vpath)}
        gate('capture round-trip fingerprint (0 = equal)', f'{label} ESSO', 0.0 if all(rt.values()) else 1.0, 0.0)
        del evectors
        capture_rec['note'] = ('re-pickled live objects (pickle.HIGHEST_PROTOCOL) loaded from the hash-verified '
                               'route-A pickles; ADMM duals / consensus targets / rho are mutable Params on the blocks '
                               '(dual_*_req, *_req, rho_*, prox_*) and on the ESSO (p_req, q_req, dual_p_req, '
                               'dual_q_req, rho), captured in vectors_*.json. state["dual_vars"] / the consensus '
                               'dicts themselves are not in the persisted pickles.')
        inventory_out['captures'][label] = capture_rec
        print(f'[capture] {label}: {len(capture_rec["groups"])} group pickles + ESSO; {time.time() - t_cell:.1f} s')

        # ---- production gross components on the live models ----------------------------------------------------------
        models_live = {'tso': tso_live, 'dso': dso_live, 'esso': esso_live}
        comps = srp._get_operational_recourse_components(planning, models_live)
        contracted_blocks = srp._get_operational_interface_settlement_blocks(planning, models_live, part='contracted')
        pin_blocks = srp._get_operational_voltage_pin_blocks(planning, models_live)
        prod_block = {}
        for y, d in [(y, d) for y in years for d in days]:
            w = srp._get_admm_block_weight(tn_data, y, d)
            prim = tn_data.network[y][d].get_primal_value(tso_live[y][d], tn_data.params)
            prod_block[(y, d, 'TSO')] = {'weight': w, 'primal_weighted': w * prim,
                                         'contracted_weighted': contracted_blocks[('TSO', None, y, d)],
                                         'pin_weighted': pin_blocks[('TSO', None, y, d)]}
            for n in nodes:
                dn_data = planning.distribution_networks[n]
                w = srp._get_admm_block_weight(dn_data, y, d)
                prim = dn_data.network[y][d].get_primal_value(dso_live[n][y][d], dn_data.params)
                prod_block[(y, d, f'DSO{n}')] = {'weight': w, 'primal_weighted': w * prim,
                                                 'contracted_weighted': contracted_blocks[('DSO', n, y, d)],
                                                 'pin_weighted': pin_blocks[('DSO', n, y, d)]}
        for v in prod_block.values():
            v['gross_weighted'] = v['primal_weighted'] - v['contracted_weighted'] - v['pin_weighted']
        prod_sum = sum(v['gross_weighted'] for v in prod_block.values())
        repro_rel = abs(comps['gross_operational_cost'] - recorded_gross) / abs(recorded_gross)
        gate('production gross recomputed on the pickles vs recorded gross (rel)', label, repro_rel,
             THRESHOLDS['objective_rel'])
        print(f'[gross] {label}: production recomputed {comps["gross_operational_cost"]!r} recorded '
              f'{recorded_gross!r} rel {repro_rel:.3e}; per-block sum {prod_sum!r}')

        # ---- step 2: inter-copy mismatches (physical) and Boyd r recomputation (v, pf) -------------------------------
        mismatch_rows = []
        boyd_sumsq = {'v': 0.0, 'pf': 0.0}
        boyd_count = {'v': 0, 'pf': 0}
        for y, d in [(y, d) for y in years for d in days]:
            tso = tso_live[y][d]
            tnet, _tparams = B.network_and_params(tso)
            sb_t = float(tnet.baseMVA)
            st = mch.sess_na_scenario(tso)
            for n in nodes:
                dso = dso_live[n][y][d]
                dnet, _dparams = B.network_and_params(dso)
                sb_d = float(dnet.baseMVA)
                sd = mch.sess_na_scenario(dso)
                dn = list(tnet.active_distribution_network_nodes).index(n)
                i_t = tnet.get_node_idx(n)
                ref = dnet.get_reference_node_id()
                i_r = dnet.get_node_idx(ref)
                vb_t, vb_d = float(tnet.get_node_base_kv(n)), float(dnet.get_node_base_kv(ref))
                e_t = tnet.get_shared_energy_storage_idx(n)
                e_d = dnet.get_shared_energy_storage_idx(ref)
                rating = float(dnet.get_interface_branch_rating())
                em = esso_live[n]
                yi, di = esso_years.index(y), esso_days.index(d)
                s_norm = {'tso': srp._shared_ess_admm_normalization_mva(
                              tnet.shared_energy_storages[e_t].s * sb_t, floor_mva, reference_mva=reference_mva),
                          'dso': srp._shared_ess_admm_normalization_mva(
                              dnet.shared_energy_storages[e_d].s * sb_d, floor_mva, reference_mva=reference_mva),
                          'esso': srp._shared_ess_admm_normalization_mva(
                              planning.shared_ess_data.shared_energy_storages[y][
                                  planning.shared_ess_data.get_shared_energy_storage_idx(n)].s,
                              floor_mva, reference_mva=reference_mva)}
                for p in tso.periods:
                    w_t = tso.vmag_sqr[i_t, st[0], st[1], p].value
                    w_d = dso.vmag_sqr[i_r, sd[0], sd[1], p].value
                    a_v, b_v = w_t * vb_t ** 2, w_d * vb_d ** 2
                    dv_mag = ((a_v - b_v) / vb_t ** 2) / (w_t ** 0.5 + (w_d * (vb_d / vb_t) ** 2) ** 0.5)
                    pc = pe.value(tso.pc_adn[dn, st[0], st[1], p]) * sb_t
                    pg = pe.value(dso.pg_adn[sd[0], sd[1], p]) * sb_d
                    qc = pe.value(tso.qc_adn[dn, st[0], st[1], p]) * sb_t
                    qg = pe.value(dso.qg_adn[sd[0], sd[1], p]) * sb_d
                    ep_t = tso.shared_es_pnet[e_t, st[0], st[1], p].value * sb_t
                    ep_d = dso.shared_es_pnet[e_d, sd[0], sd[1], p].value * sb_d
                    eq_t = tso.shared_es_qnet[e_t, st[0], st[1], p].value * sb_t
                    eq_d = dso.shared_es_qnet[e_d, sd[0], sd[1], p].value * sb_d
                    ep_e, eq_e = em.es_pnet[yi, di, p].value, em.es_qnet[yi, di, p].value
                    base = {'year': y, 'day': d, 'node': n, 'period': p}
                    a_ess = 1.0 / (2.0 * max(s_norm.values()))
                    for ch, a, b, unit, norm in (
                            ('v_W', a_v, b_v, 'kV^2', abs(dv_mag)),
                            ('pf_p', pc, pg, 'MW', abs(pc - pg) / rating),
                            ('pf_q', qc, qg, 'Mvar', abs(qc - qg) / rating),
                            ('ess_p_tso_esso', ep_t, ep_e, 'MW', abs(ep_t - ep_e) * a_ess),
                            ('ess_p_dso_esso', ep_d, ep_e, 'MW', abs(ep_d - ep_e) * a_ess),
                            ('ess_q_tso_esso', eq_t, eq_e, 'Mvar', abs(eq_t - eq_e) * a_ess),
                            ('ess_q_dso_esso', eq_d, eq_e, 'Mvar', abs(eq_d - eq_e) * a_ess),
                            ('ess_p_tso_dso', ep_t, ep_d, 'MW', abs(ep_t - ep_d) * a_ess),
                            ('ess_q_tso_dso', eq_t, eq_d, 'Mvar', abs(eq_t - eq_d) * a_ess)):
                        mismatch_rows.append({**base, 'channel': ch, 'a_phys': a, 'b_phys': b, 'mismatch': a - b,
                                              'unit': unit, 'boyd_normalized_abs': norm})
                    # Boyd r recomputation from the expected_* consensus Vars (SRP l.7196-7240 formulas)
                    x_v = tso.expected_interface_vmag[dn, p].value * vb_t
                    x_dso_v = dso.expected_interface_vmag[p].value * vb_d
                    boyd_sumsq['v'] += ((x_dso_v - x_v) / vb_t) ** 2
                    boyd_count['v'] += 1
                    for ptype in ('p', 'q'):
                        z = getattr(tso, f'expected_interface_pf_{ptype}')[dn, p].value * sb_t
                        x = getattr(dso, f'expected_interface_pf_{ptype}')[p].value * sb_d
                        boyd_sumsq['pf'] += ((x - z) / rating) ** 2
                        boyd_count['pf'] += 1
                coupling_out.setdefault('constants', {}).setdefault(label, {})[f'{y}_{d}_{n}'] = {
                    'dn': dn, 'tso_adn_node_idx': i_t, 'dso_ref_node': ref, 'dso_ref_node_idx': i_r,
                    'vb_tso_kV': vb_t, 'vb_dso_kV': vb_d, 'sb_tso_MVA': sb_t, 'sb_dso_MVA': sb_d,
                    'e_tso': e_t, 'e_dso': e_d, 'esso_year_idx': yi, 'esso_day_idx': di,
                    'interface_rating_MVA': rating, 's_norm_MVA': s_norm, 'na_scenario_tso': list(st),
                    'na_scenario_dso': list(sd)}
        recompute = {}
        for ch in ('v', 'pf'):
            r = boyd_sumsq[ch] ** 0.5
            rec = boyd_channels[ch]['r']
            recompute[ch] = {'r_recomputed': r, 'r_recorded': rec, 'n_entries': boyd_count[ch],
                             'rel_diff': abs(r - rec) / abs(rec) if rec else abs(r)}
            gate(f'Boyd r_{ch} recomputed from blocks vs recorded (rel; provenance)', label,
                 recompute[ch]['rel_diff'], THRESHOLDS['boyd_r_recompute_rel'])
        recompute['ess'] = {'note': 'not recomputed: Boyd ESS r = a (x_i - z) needs the end-of-cycle consensus z, '
                                    'which is not in the persisted pickles (the blocks carry the z used AT their '
                                    'solve). Pairwise mismatches are reported instead.'}
        coupling_out['boyd_recompute'][label] = recompute
        coupling_out['mismatch_step1'][label] = mismatch_rows
        mm_summary = {}
        for ch in sorted({r['channel'] for r in mismatch_rows}):
            rows_ch = [r for r in mismatch_rows if r['channel'] == ch]
            worst = max(rows_ch, key=lambda r: abs(r['mismatch']))
            mm_summary[ch] = {'n': len(rows_ch), 'max_abs_mismatch': abs(worst['mismatch']), 'unit': worst['unit'],
                              'argmax': {k: worst[k] for k in ('year', 'day', 'node', 'period')},
                              'max_boyd_normalized': max(r['boyd_normalized_abs'] for r in rows_ch),
                              'l2_boyd_normalized': sum(r['boyd_normalized_abs'] ** 2 for r in rows_ch) ** 0.5}
        coupling_out['summary'].setdefault(label, {})['mismatch_step1'] = mm_summary
        print(f'[mismatch] {label}: ' + '; '.join(f'{ch} max {s["max_abs_mismatch"]:.3e} {s["unit"]}'
                                                   for ch, s in mm_summary.items()))
        print(f'[boyd] {label}: r_v {recompute["v"]["r_recomputed"]!r} vs {recompute["v"]["r_recorded"]!r}; '
              f'r_pf {recompute["pf"]["r_recomputed"]!r} vs {recompute["pf"]["r_recorded"]!r}')
        mismatch_index = {(r['year'], r['day'], r['node'], r['period'], r['channel']): r for r in mismatch_rows}

        # ---- ESSO: inventory, relaxation, identities (mode-independent) -----------------------------------------------
        esso_out = {}
        for n in nodes:
            em = esso_live[n]
            inv = B.inventory_rows(em, 'esso')
            inventory_out['blocks'][f'{label}/ESSO{n}'] = inv
            relaxed, record = B.relax_esso_block(em)
            dropped_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'dropped'}
            retained_inv = {(r['family'], r['index']) for r in inv['rows'] if r['class'] == 'retained'}
            inactive_now, active_both = set(), set()
            for component in em.component_objects(pe.Constraint, active=True):
                rcomp = relaxed.find_component(component.name)
                for index, cd in component.items():
                    if cd.active:
                        (active_both if rcomp[index].active else inactive_now).add((component.name, repr(index)))
            consistent = dropped_inv == inactive_now and retained_inv == active_both
            nonconvex = B.nonconvex_active_rows(relaxed)
            inventory_out['nonconvex_after'][f'{label}/ESSO{n}'] = nonconvex
            tables = B.mapped_point(em, relaxed)
            same_point = B.production_rows_at_point(em, relaxed)
            ret = tables['retained']
            slack_rows = set()
            for component in em.component_objects(pe.Constraint, active=True):
                for index, cd in component.items():
                    if cd.active and {v.parent_component().local_name for v in
                                      identify_variables(cd.body, include_fixed=True)} & set(B.ESSO_SLACK_VARS):
                        slack_rows.add((component.name, repr(index)))
            fixed_slack_rows = []
            gated = []
            for r in ret:
                r['references_fixed_slack'] = (r['family'], r['index']) in slack_rows
                (fixed_slack_rows if r['references_fixed_slack'] else gated).append(r)
            thr = lambda r: max(THRESHOLDS['mapped_floor_abs'], THRESHOLDS['mapped_eps_multiple'] * r['eval_resolution'])
            over = [r for r in gated if r['delta'] > thr(r)]
            subs = {r['sub'] for r in inv['rows'] if r['class'] == 'dropped'}
            esso_out[f'ESSO{n}'] = {
                'build_record': record, 'inventory_builder_consistent': consistent,
                'dropped_subs': sorted(subs), 'nonconvex_after': nonconvex, 'same_point': same_point,
                'retained': {'n_rows': len(ret), 'n_gated': len(gated),
                             'max_delta_gated': max((r['delta'] for r in gated), default=0.0),
                             'n_over_mapped_gate': len(over),
                             'rows_referencing_fixed_slacks': {
                                 'n': len(fixed_slack_rows),
                                 'max_delta': max((r['delta'] for r in fixed_slack_rows), default=0.0),
                                 'note': 'slack_es_pnet_up/down are FIXED at 0 by spec section 3 item 4; the mapped '
                                         'point moves these rows by the certified slack value (reported, gated '
                                         'separately against the stop line only)',
                                 'rows_delta_above_floor': [r for r in fixed_slack_rows
                                                            if r['delta'] > THRESHOLDS['retained_table_floor']]},
                             'rows_over_mapped_gate': over},
                'added_available_eq_rated': tables['added_esso'],
                'capacity': {'es_s_investment_fixed': {str(k): float(pe.value(v)) for k, v in
                                                       em.es_s_investment_fixed.items()},
                             'es_e_investment_fixed': {str(k): float(pe.value(v)) for k, v in
                                                       em.es_e_investment_fixed.items()},
                             'es_e_rated_per_unit': {str(k): v.value for k, v in em.es_e_rated_per_unit.items()},
                             'es_e_available_per_unit_certified': {str(k): v.value for k, v in
                                                                   em.es_e_available_per_unit.items()},
                             'es_soh_per_unit_cumul_certified': {str(k): v.value for k, v in
                                                                 em.es_soh_per_unit_cumul.items()}}}
            scope = f'{label} ESSO{n}'
            gate('G-D0-2 inventory == builder (0 = consistent)', scope, 0.0 if consistent else 1.0, 0.0)
            gate('ESSO dropped subs within ESSO_DROP_ARGUMENTS (0 = yes)', scope,
                 0.0 if subs <= set(B.ESSO_DROP_ARGUMENTS) else 1.0, 0.0)
            gate('nonconvex rows after relaxation (count)', scope, float(len(nonconvex)), 0.0)
            gate('same-point Delta (exact 0)', scope, same_point['max_abs_delta'], 0.0, rule='==')
            gate('mapped-point rows over max(1e-12, 8 eps sum|terms|) (count)', scope, float(len(over)), 0.0)
            if nonconvex:
                stops.append({'stop': 'nonconvex row remains', 'scope': scope, 'rows': nonconvex[:20]})
            worst_any = max((r['delta'] for r in ret), default=0.0)
            if worst_any > THRESHOLDS['stop_abs']:
                stops.append({'stop': 'retained-row Delta > 1e-10', 'scope': scope, 'max_delta': worst_any})
            del relaxed
        identities_out['esso'][label] = esso_out

        # ---- groups ------------------------------------------------------------------------------------------------
        objective_cell = {'per_block': {}, 'per_group': {}, 'recorded_gross': recorded_gross,
                          'production_recomputed_gross': comps['gross_operational_cost'],
                          'production_recomputed_components': {k: v for k, v in comps.items()
                                                               if not isinstance(v, dict)},
                          'production_per_block_sum': prod_sum}
        identities_out['groups'][label] = {}
        identities_out['capacity'][label] = {}
        coupling_out['rows'][label] = {}
        lb_sum = {'certified': {m: 0.0 for m in MODES}, 'mapped_ef': {m: 0.0 for m in MODES}}
        prod_sum_groups = 0.0

        for (y, d) in groups:
            t_group = time.time()
            gkey = f'{y}_{d}'
            blocks = live_blocks(y, d)
            group_out = {'blocks': {}, 'modes': {}}
            identities_out['groups'][label][gkey] = group_out
            # inventories (mode-independent)
            inv_by = {}
            for blabel, _n, m in blocks:
                inv = B.inventory_rows(m, 'network', drop_sess_comp=True)
                inv_by[blabel] = inv
                inventory_out['blocks'][f'{label}/{gkey}/{blabel}'] = inv
                keys = [(r['family'], r['index']) for r in inv['rows']]
                n_active = sum(1 for _ in m.component_data_objects(pe.Constraint, active=True, descend_into=True))
                gate('G-D0-2 each active row classified exactly once (0 = yes)', f'{label} {gkey} {blabel}',
                     0.0 if (len(keys) == len(set(keys)) == n_active) else 1.0, 0.0)
            # capacity carried vs rated
            cap = {}
            for blabel, _n, m in blocks:
                net, _p = B.network_and_params(m)
                cap[blabel] = {str(e): {'carried_s_pu': float(pe.value(m.shared_es_s_rated_fixed[e])),
                                        'carried_e_pu': float(pe.value(m.shared_es_e_rated_fixed[e])),
                                        'rated_s_pu': float(net.shared_energy_storages[e].s),
                                        'rated_e_pu': float(net.shared_energy_storages[e].e),
                                        'bus': net.shared_energy_storages[e].bus,
                                        'sb_MVA': float(net.baseMVA)}
                               for e in m.shared_es_s_rated_fixed}
            identities_out['capacity'][label][gkey] = cap
            # production per-block gross for this group
            prod_g = {bl: prod_block[(y, d, bl)] for bl, _n, _m in blocks}
            prod_g_sum = sum(v['gross_weighted'] for v in prod_g.values())
            prod_sum_groups += prod_g_sum

            for mode in MODES:
                scope_g = f'{label} {gkey} {mode}'
                relaxed, records_b = {}, {}
                for blabel, _n, m in blocks:
                    rated = None
                    if mode == 'rated':
                        net, _p = B.network_and_params(m)
                        rated = {e: {'s': float(net.shared_energy_storages[e].s),
                                     'e': float(net.shared_energy_storages[e].e)} for e in m.shared_es_s_rated_fixed}
                    rx, rec = B.relax_network_block(m, mode, rated, drop_sess_comp=True)
                    relaxed[blabel], records_b[blabel] = rx, rec
                mode_out = {'blocks': {}}
                group_out['modes'][mode] = mode_out

                # inventory <-> builder consistency and per-family counts
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
                    counts = {}
                    for r in inv['rows']:
                        counts[(r['class'], r['sub'], r['family'])] = counts.get(
                            (r['class'], r['sub'], r['family']), 0) + 1
                    count_checks = {}
                    for fam in B.DEFINITIONAL_FAMILIES:
                        sub = B.DEFINITIONAL_FAMILIES[fam][1]
                        count_checks[f'replaced/{fam}'] = (counts.get(('replaced', sub, fam), 0),
                                                           rec['deactivated_definitional_rows'][fam])
                    for fam, nrow in rec['envelope_rewrite_rows_by_family'].items():
                        count_checks[f'envelope_rewrite/{fam}'] = (counts.get(('replaced', 'envelope_rewrite', fam),
                                                                              0), nrow)
                    inv_env = sum(v for (c, s, f), v in counts.items() if s == 'envelope_rewrite')
                    count_checks['envelope_rewrite/total'] = (inv_env, rec['n_envelope_rewrite_rows'])
                    count_checks['dropped/sess_comp'] = (counts.get(('dropped', 'sess_comp_nonconvex', 'sess_comp'),
                                                                    0), rec['dropped_rows']['sess_comp'])
                    counts_ok = all(a == b for a, b in count_checks.values())
                    consistent = not_retained == inactive_now and retained_inv == active_both
                    nonconvex = B.nonconvex_active_rows(rx)
                    inventory_out['nonconvex_after'][f'{label}/{gkey}/{mode}/{blabel}'] = nonconvex
                    mode_out['blocks'][blabel] = {
                        'build_record': rec, 'inventory_builder_consistent': consistent,
                        'family_count_checks': {k: list(v) for k, v in count_checks.items()},
                        'family_counts_consistent': counts_ok, 'nonconvex_after': nonconvex}
                    sc = f'{scope_g} {blabel}'
                    gate('G-D0-2 inventory == builder (0 = consistent)', sc, 0.0 if consistent else 1.0, 0.0)
                    gate('G-D0-2 per-family counts inventory == builder (0 = equal)', sc, 0.0 if counts_ok else 1.0,
                         0.0)
                    gate('nonconvex rows after relaxation (count)', sc, float(len(nonconvex)), 0.0)
                    if nonconvex:
                        stops.append({'stop': 'nonconvex row remains', 'scope': sc, 'rows': nonconvex[:20]})

                # ---- group model with coupling rows ---------------------------------------------------------------
                g = pe.ConcreteModel(name=f'P516D0_T01b_{label}_{gkey}_{mode}')
                g.tso = relaxed['TSO']
                for n in nodes:
                    g.add_component(f'dso_{n}', relaxed[f'DSO{n}'])
                    rex, _rec = B.relax_esso_block(esso_live[n])
                    rex.d0_lb_objective.deactivate()
                    g.add_component(f'esso_{n}', rex)
                tso = g.tso
                tnet, _tp = B.network_and_params(tso)
                st = mch.sess_na_scenario(tso)
                g.d0_cpl_nodes = pe.Set(initialize=nodes, ordered=True)
                g.d0_cpl_periods = pe.Set(initialize=list(tso.periods), ordered=True)
                const = coupling_out['constants'][label]

                def k(n):
                    return const[f'{y}_{d}_{n}']

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

                # coupling residuals at the certified point (relaxed clones carry the certified values)
                cpl_rows = []
                for ch in channel_of:
                    comp = getattr(g, f'd0_cpl_{ch}')
                    for (n, p), cd in comp.items():
                        resid = (pe.value(cd.body) - pe.value(cd.upper)) * to_phys(ch, n)
                        mm = mismatch_index[(y, d, n, p, ch)]
                        repn = generate_standard_repn(cd.body, compute_values=True, quadratic=False)
                        term_scale = (abs(float(pe.value(repn.constant)))
                                      + sum(abs(float(pe.value(c_)) * v_.value)
                                            for c_, v_ in zip(repn.linear_coefs, repn.linear_vars)))
                        tol = THRESHOLDS['coupling_eps_multiple'] * EPS * (
                            term_scale * to_phys(ch, n) + abs(mm['a_phys']) + abs(mm['b_phys']))
                        cpl_rows.append({'channel': ch, 'node': n, 'period': p, 'row': cd.name,
                                         'residual_phys_certified': resid, 'mismatch_step1': mm['mismatch'],
                                         'diff': resid - mm['mismatch'], 'fp_tol': tol, 'unit': mm['unit'],
                                         'boyd_normalized_abs': mm['boyd_normalized_abs']})

                # LB objective (production component functions) and the group form audit
                lb_terms = []
                for blabel, n, _m in blocks:
                    sub = g.tso if blabel == 'TSO' else g.find_component(f'dso_{n}')
                    _net, prm = B.network_and_params(sub)
                    nd = tn_data if blabel == 'TSO' else planning.distribution_networks[n]
                    w = srp._get_admm_block_weight(nd, y, d)
                    expr, legacy = B.block_gross_expr(sub, prm)
                    if legacy:
                        raise RuntimeError(f'{scope_g} {blabel}: certified block lacks the settlement attributes')
                    lb_terms.append((blabel, w, expr, float(pe.value(sub.admm_block_weight))))
                g.d0_lb_objective = pe.Objective(expr=sum(w * e for _b, w, e, _bw in lb_terms), sense=pe.minimize)
                obj_form = B.objective_form(g.d0_lb_objective.expr)
                gate('LB objective form linear or convex (0 = yes)', scope_g,
                     0.0 if obj_form['form'] in ('linear', 'convex_quadratic_psd') else 1.0, 0.0)
                group_nonconvex = B.nonconvex_active_rows(g)
                inventory_out['nonconvex_after'][f'{label}/{gkey}/{mode}/GROUP'] = group_nonconvex
                gate('nonconvex rows in the assembled group (count)', scope_g, float(len(group_nonconvex)), 0.0)
                if group_nonconvex:
                    stops.append({'stop': 'nonconvex row remains', 'scope': f'{scope_g} GROUP',
                                  'rows': group_nonconvex[:20]})

                obj_cert = {}
                for blabel, w, expr, bw in lb_terms:
                    v = w * float(pe.value(expr))
                    pr = prod_g[blabel]['gross_weighted']
                    obj_cert[blabel] = {'weight': w, 'block_admm_block_weight_param': bw, 'relaxed_term': v,
                                        'production_gross_weighted': pr,
                                        'rel': abs(v - pr) / abs(pr) if pr else abs(v - pr)}
                lb_group_cert = float(pe.value(g.d0_lb_objective))

                # export (point-independent) -- skipped if any nonconvex row remains
                lp_rec = None
                if not group_nonconvex and not any(mode_out['blocks'][b]['nonconvex_after'] for b in mode_out['blocks']):
                    lp_rel = os.path.join('lp', label, f'{gkey}_{mode}.lp')
                    lp_path = os.path.join(OUT_DIR, lp_rel)
                    os.makedirs(os.path.dirname(lp_path), exist_ok=True)
                    if os.path.exists(lp_path):
                        raise RuntimeError(f'refusing to overwrite {lp_path}')
                    lp_sha, nm, nm_counts = B.export_lp(g, lp_path)
                    nm_rel = os.path.join('lp', label, f'{gkey}_{mode}.name_map.json')
                    write_json(nm_rel, {'lp': lp_rel, 'lp_sha256': lp_sha, 'name_map': nm}, compact=True)
                    lp_rec = {'path': lp_rel, 'sha256': lp_sha, 'bytes': os.path.getsize(lp_path),
                              'counts': nm_counts, 'name_map': nm_rel,
                              'name_map_sha256': sha256_file(os.path.join(OUT_DIR, nm_rel)),
                              'cell': label, 'eval_dir': cell['eval_dir'], 'year': y, 'day': d, 'mode': mode,
                              'variant': VARIANT[mode], 'variant_semantics': VARIANT_SEMANTICS[VARIANT[mode]],
                              'composition': ['tso'] + [f'dso_{n}' for n in nodes] + [f'esso_{n}' for n in nodes],
                              'coupling_families': [f'd0_cpl_{ch}' for ch in channel_of],
                              'objective': ('d0_lb_objective = sum_b w_b * gross_b (see objective_identity.json); '
                                            'gross convention, ESSO term 0'),
                              'objective_form': obj_form}
                    lp_index['lp_files'][lp_rel] = lp_rec

                # ---- mapped point 'ef' (rank-1) on the sub-blocks in place; same-point check -----------------------
                map_out = {}
                for blabel, n, m in blocks:
                    sub = g.tso if blabel == 'TSO' else g.find_component(f'dso_{n}')
                    tables = B.mapped_point(m, sub, w_source='ef')
                    same_point = B.production_rows_at_point(m, sub)
                    ret = tables['retained']
                    excluded = [r for r in ret if mode == 'rated' and r['references_capacity_params']]
                    gated = [r for r in ret if not (mode == 'rated' and r['references_capacity_params'])]
                    thr = lambda r: max(THRESHOLDS['mapped_floor_abs'],
                                        THRESHOLDS['mapped_eps_multiple'] * r['eval_resolution'])
                    over = [r for r in gated if r['delta'] > thr(r)]
                    beyond_stop = [r for r in gated if r['delta'] > THRESHOLDS['stop_abs']]
                    beyond_bound = [r for r in gated if r['delta'] > r['first_order_w_bound'] + 4 * r['eval_resolution']]
                    worst = max(gated, key=lambda r: r['delta'])
                    fam_max = {}
                    for r in gated:
                        fam_max[r['family']] = max(fam_max.get(r['family'], 0.0), r['delta'])
                    cones_w = [abs(c['rel_residual']) for c in tables['cones'] if c['kind'] == 'W']
                    cones_t = [abs(c['rel_residual']) for c in tables['cones'] if c['kind'] == 'transformer_UCD']
                    env = tables['envelope']
                    env_max = max((r['delta'] for r in env), default=0.0)
                    bound_min = min((min(b['lower_slack_rel'], b['upper_slack_rel']) for b in tables['envelope_bounds']),
                                    default=None)
                    tab_rows = [{'family': r['family'], 'index': r['index'], 'delta': r['delta'],
                                 'eval_resolution': r['eval_resolution'], 'first_order_w_bound': r['first_order_w_bound'],
                                 'threshold': thr(r)} for r in gated if r['delta'] > THRESHOLDS['retained_table_floor']]
                    defin = {}
                    for r in tables['replaced_production']:
                        if r['family'] in B.DEFINITIONAL_FAMILIES:
                            v_ = max(abs(x) for x in (r['prod']['r_lower'], r['prod']['r_upper']) if x is not None)
                            defin[r['family']] = max(defin.get(r['family'], 0.0), v_)
                    map_out[blabel] = {
                        'same_point': same_point,
                        'definitional_row_residuals_at_certified_max_abs': defin,
                        'retained_ef': {'n_rows': len(ret), 'n_gated': len(gated), 'n_zero': sum(1 for r in gated
                                                                                              if r['delta'] == 0.0),
                                        'n_above_table_floor': len(tab_rows),
                                        'max_delta': worst['delta'],
                                        'argmax': {k_: worst[k_] for k_ in ('family', 'index', 'eval_resolution',
                                                                              'first_order_w_bound')},
                                        'max_delta_by_family': fam_max,
                                        'n_over_gate': len(over), 'n_beyond_stop_1e-10': len(beyond_stop),
                                        'n_beyond_first_order_bound_plus_4res': len(beyond_bound),
                                        'max_delta_over_threshold': max(r['delta'] / thr(r) for r in gated),
                                        'rows_above_table_floor': tab_rows,
                                        'excluded_capacity_param_rows': [
                                            {'family': r['family'], 'index': r['index'], 'delta': r['delta']}
                                            for r in excluded]},
                        'cones_ef': {'n_W': len(cones_w), 'max_rel_W': max(cones_w) if cones_w else None,
                                     'n_transformer': len(cones_t),
                                     'max_rel_transformer': max(cones_t) if cones_t else None},
                        'envelope_ef': {'n_rows': len(env), 'max_delta': env_max,
                                        'envelope_bounds_min_rel_slack': bound_min},
                    }
                    sc = f'{scope_g} {blabel}'
                    gate('same-point Delta (exact 0)', sc, same_point['max_abs_delta'], 0.0, rule='==')
                    gate('mapped-point (ef) rows over max(1e-12, 8 eps sum|terms|) (count)', sc, float(len(over)), 0.0)
                    gate('cone rel (W) at ef', sc, max(cones_w), THRESHOLDS['cone_rel'])
                    if cones_t:
                        gate('cone rel (transformer UCD) at ef', sc, max(cones_t), THRESHOLDS['cone_rel'])
                    if env:
                        gate('envelope-rewrite |Delta| at ef', sc, env_max, THRESHOLDS['envelope_abs'])
                    if beyond_stop:
                        stops.append({'stop': 'retained-row Delta > 1e-10 (mapped point ef)', 'scope': sc,
                                      'n_rows': len(beyond_stop), 'max_delta': worst['delta'],
                                      'rows': [{k_: r[k_] for k_ in ('family', 'index', 'delta', 'eval_resolution',
                                                                      'first_order_w_bound')} for r in beyond_stop]})
                for n in nodes:
                    B._set_esso_mapped_point(g.find_component(f'esso_{n}'))
                # objective and coupling at the ef point
                obj_ef = {}
                for blabel, w, expr, _bw in lb_terms:
                    v = w * float(pe.value(expr))
                    pr = prod_g[blabel]['gross_weighted']
                    obj_ef[blabel] = {'relaxed_term': v, 'rel': abs(v - pr) / abs(pr) if pr else abs(v - pr)}
                lb_group_ef = float(pe.value(g.d0_lb_objective))
                for row in cpl_rows:
                    cd = getattr(g, f'd0_cpl_{row["channel"]}')[row['node'], row['period']]
                    row['residual_phys_mapped_ef'] = (pe.value(cd.body) - pe.value(cd.upper)) * to_phys(
                        row['channel'], row['node'])

                # ---- diagnostic: mapped point with W kept at the certified values ('snapshot'), fresh clones -------
                snap = {}
                for blabel, _n, m in blocks:
                    rated = None
                    if mode == 'rated':
                        net, _p = B.network_and_params(m)
                        rated = {e: {'s': float(net.shared_energy_storages[e].s),
                                     'e': float(net.shared_energy_storages[e].e)} for e in m.shared_es_s_rated_fixed}
                    rx, _rec = B.relax_network_block(m, mode, rated, drop_sess_comp=True)
                    t2 = B.mapped_point(m, rx, w_source='snapshot')
                    gated2 = [r for r in t2['retained'] if not (mode == 'rated' and r['references_capacity_params'])]
                    c_w = [abs(c['rel_residual']) for c in t2['cones'] if c['kind'] == 'W']
                    c_t = [abs(c['rel_residual']) for c in t2['cones'] if c['kind'] == 'transformer_UCD']
                    snap[blabel] = {'retained_max_delta': max(r['delta'] for r in gated2),
                                    'cone_W_max_rel': max(c_w), 'cone_transformer_max_rel': max(c_t) if c_t else None,
                                    'envelope_max_delta': max((r['delta'] for r in t2['envelope']), default=0.0)}
                    del rx, t2
                for blabel in snap:
                    map_out[blabel]['diagnostic_snapshot_W'] = snap[blabel]

                mode_out['mapped'] = map_out
                mode_out['variant'] = VARIANT[mode]
                mode_out['objective'] = {'variant': VARIANT[mode], 'certified': obj_cert, 'mapped_ef': obj_ef,
                                         'group_relaxed_objective_certified': lb_group_cert,
                                         'group_relaxed_objective_mapped_ef': lb_group_ef,
                                         'group_production_gross': prod_g_sum,
                                         'group_rel_certified': abs(lb_group_cert - prod_g_sum) / abs(prod_g_sum),
                                         'group_rel_mapped_ef': abs(lb_group_ef - prod_g_sum) / abs(prod_g_sum)}
                mode_out['lp'] = lp_rec
                for blabel in obj_cert:
                    gate('objective identity per block, certified (rel)', f'{scope_g} {blabel}',
                         obj_cert[blabel]['rel'], THRESHOLDS['objective_rel'])
                    gate('objective identity per block, mapped ef (rel)', f'{scope_g} {blabel}',
                         obj_ef[blabel]['rel'], THRESHOLDS['objective_rel'])
                gate('objective identity per group, certified (rel)', scope_g,
                     mode_out['objective']['group_rel_certified'], THRESHOLDS['objective_rel'])
                lb_sum['certified'][mode] += lb_group_cert
                lb_sum['mapped_ef'][mode] += lb_group_ef

                # coupling gates (residual == mismatch to fp resolution); rows identical in both modes
                n_bad = sum(1 for r in cpl_rows if abs(r['diff']) > r['fp_tol'])
                gate('coupling residual == step-1 mismatch (rows beyond 4 eps (sum|row terms| + |a| + |b|))', scope_g, float(n_bad), 0.0)
                if mode == 'available':
                    coupling_out['rows'][label][gkey] = cpl_rows
                else:
                    group_out['coupling_rows_rated'] = cpl_rows

                print(f'[group] {scope_g}: same-point max '
                      f'{max(map_out[b]["same_point"]["max_abs_delta"] for b in map_out):.1e}; ef max '
                      f'{max(map_out[b]["retained_ef"]["max_delta"] for b in map_out):.3e} '
                      f'(over-gate {sum(map_out[b]["retained_ef"]["n_over_gate"] for b in map_out)}, '
                      f'>1e-10 {sum(map_out[b]["retained_ef"]["n_beyond_stop_1e-10"] for b in map_out)}); '
                      f'cones {max(map_out[b]["cones_ef"]["max_rel_W"] for b in map_out):.1e}; '
                      f'coupling diff max {max(abs(r["diff"]) for r in cpl_rows):.1e}; '
                      f'obj rel {mode_out["objective"]["group_rel_certified"]:.1e}; '
                      f'lp {lp_rec["counts"] if lp_rec else None}')
                del g, relaxed
                gc.collect()
            # rated and available coupling rows must agree (Var values identical in both modes)
            ra = group_out.pop('coupling_rows_rated', None)
            av = coupling_out['rows'][label][gkey]
            same = ra is not None and all(a['residual_phys_certified'] == b['residual_phys_certified']
                                          for a, b in zip(ra, av)) and len(ra) == len(av)
            gate('coupling residuals identical in rated / available (0 = yes)', f'{label} {gkey}',
                 0.0 if same else 1.0, 0.0)
            print(f'[group] {label} {gkey} done in {time.time() - t_group:.1f} s')

        objective_cell['sum_over_groups'] = {
            'n_groups': len(groups),
            'variant_by_mode': VARIANT,
            'relaxed_objective_certified': lb_sum['certified'], 'relaxed_objective_mapped_ef': lb_sum['mapped_ef'],
            'note': ('the relaxed objective evaluated at the certified (and the rank-1 mapped) point: an IDENTITY '
                     'check of the objective construction, not a bound value (no relaxation is solved here)'),
            'production_per_block_gross_sum_over_processed_groups': prod_sum_groups,
            'rel_vs_recorded_gross': {m: abs(lb_sum['certified'][m] - recorded_gross) / abs(recorded_gross)
                                      for m in MODES},
            'rel_vs_production_recomputed': {m: abs(lb_sum['certified'][m] - comps['gross_operational_cost'])
                                             / abs(comps['gross_operational_cost']) for m in MODES}}
        if len(groups) == len(years) * len(days):
            for m in MODES:
                gate('12-group LB-objective sum at the certified point vs recorded gross (rel)', f'{label} {m}',
                     objective_cell['sum_over_groups']['rel_vs_recorded_gross'][m], THRESHOLDS['objective_rel'])
        for (y, d) in groups:
            for mode in MODES:
                objective_cell['per_group'][f'{y}_{d}_{mode}'] = identities_out['groups'][label][f'{y}_{d}'][
                    'modes'][mode]['objective']
        objective_cell['production_per_block'] = {f'{y}_{d}_{b}': v for (y, d, b), v in prod_block.items()}
        objective_out['cells'][label] = objective_cell

        # coupling summary vs eps_pri (Q3)
        summ = {}
        for ch, chan in channel_of.items():
            rows = [r for gk in coupling_out['rows'][label] for r in coupling_out['rows'][label][gk]
                    if r['channel'] == ch]
            if not rows:
                continue
            summ[ch] = {'boyd_channel': chan, 'eps_pri': boyd_channels[chan]['eps_pri'],
                        'r_recorded': boyd_channels[chan]['r'], 'n_rows': len(rows),
                        'max_abs_residual_phys': max(abs(r['residual_phys_certified']) for r in rows),
                        'unit': rows[0]['unit'],
                        'max_boyd_normalized': max(r['boyd_normalized_abs'] for r in rows),
                        'l2_boyd_normalized': sum(r['boyd_normalized_abs'] ** 2 for r in rows) ** 0.5,
                        'max_abs_diff_vs_mismatch': max(abs(r['diff']) for r in rows)}
            summ[ch]['max_normalized_over_eps_pri'] = summ[ch]['max_boyd_normalized'] / summ[ch]['eps_pri']
        coupling_out['summary'][label]['rows_vs_eps_pri'] = summ

        # fingerprints after; inputs unchanged
        fp_after = {}
        for y, d in [(y, d) for y in years for d in days]:
            for blabel, _n, m in live_blocks(y, d):
                fp_after[f'{y}_{d}_{blabel}'] = B.model_fingerprint(m)
        for n in nodes:
            fp_after[f'ESSO{n}'] = B.model_fingerprint(esso_live[n])
        changed = sorted(k_ for k_ in fp_before if fp_before[k_] != fp_after[k_])
        identities_out['fingerprints'][label] = {'n_models': len(fp_before), 'changed': changed,
                                                 'before': fp_before, 'after': fp_after}
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
    solver_libraries = {name: any(m == name or m.startswith(name + '.') for m in sys.modules)
                        for name in ('gurobipy', 'highspy', 'clarabel', 'cplex', 'docplex', 'xpress', 'mosek',
                                     'cyipopt', 'scs', 'ecos', 'osqp')}
    guard_out = {'label': guard.label, 'permitted': list(guard.permitted), 'counts': dict(guard.counts),
                 'declared_solves': 0, 'verify_failures': guard_failures,
                 'process_launch_events': _AUDIT['process_launch_events'],
                 'solver_libraries_imported': solver_libraries}
    gate('guard: zero solves (failures count)', 'process', float(len(guard_failures)), 0.0)
    gate('process launches in the child (audit hook)', 'process', float(len(_AUDIT['process_launch_events'])), 0.0)
    gate('solver libraries imported (count)', 'process', float(sum(solver_libraries.values())), 0.0)

    out_abs = os.path.abspath(OUT_DIR)
    repo_abs = os.path.abspath(REPO)
    writes_inside, writes_repo, writes_pycache, writes_outside_repo = [], [], [], []
    for path, count in sorted(_AUDIT['write_opens'].items()):
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
                             'child process (concurrent T-perf writes are other processes and are not seen)',
                   'n_write_opens_inside_out_dir': len(writes_inside),
                   'write_opens_in_repo_outside_out_dir': [rel(p) for p in writes_repo],
                   'write_opens_pycache': [rel(p) if p.startswith(repo_abs) else p for p in writes_pycache],
                   'write_opens_outside_repo': writes_outside_repo,
                   'fs_mutation_events_outside_out_dir': fs_outside,
                   'side_writes_redirected_to': rel(SIDE_WRITES_DIR),
                   'n_side_write_files': sum(len(f) for _r, _d, f in os.walk(SIDE_WRITES_DIR))}
    gate('containment: write-opens in the repository outside OUT_DIR (count)', 'process', float(len(writes_repo)), 0.0)

    # ==================================================================================================================
    #  Predictions vs outcomes
    # ==================================================================================================================
    nonconvex_total = sum(len(v) for v in inventory_out['nonconvex_after'].values())
    sp_rows = [r for r in gate_rows if r['gate'] == 'same-point Delta (exact 0)']
    groups_done = {label: list(identities_out['groups'][label]) for label in identities_out['groups']}
    q3 = {}
    for label in coupling_out['summary']:
        for ch, s in coupling_out['summary'][label].get('rows_vs_eps_pri', {}).items():
            q3[f'{label}/{ch}'] = {'max_abs_diff_vs_mismatch': s['max_abs_diff_vs_mismatch'],
                                   'max_boyd_normalized': s['max_boyd_normalized'], 'eps_pri': s['eps_pri'],
                                   'le_eps_pri': s['max_boyd_normalized'] <= s['eps_pri']}
    cpl_eq_ok = all(r['pass'] for r in gate_rows if r['gate'].startswith('coupling residual == step-1'))
    obj_block_max = max((r['achieved'] for r in gate_rows if r['gate'].startswith('objective identity per block')),
                        default=None)
    sum_rel = {f'{label}/{m}': objective_out['cells'][label]['sum_over_groups']['rel_vs_recorded_gross'][m]
               for label in objective_out['cells'] for m in MODES}
    q5 = {'unit_n7': {}, 'x0': {}}
    for label in identities_out['capacity']:
        for gk, cap in identities_out['capacity'][label].items():
            for blabel, per_e in cap.items():
                for e, c in per_e.items():
                    if label == 'unit_n7' and c['rated_e_pu'] > 0:
                        q5['unit_n7'][f'{gk}/{blabel}/e{e}'] = {'carried_e_pu': c['carried_e_pu'],
                                                                'rated_e_pu': c['rated_e_pu'],
                                                                'below_rated': c['carried_e_pu'] < c['rated_e_pu']}
                    if label == 'x0':
                        q5['x0'][f'{gk}/{blabel}/e{e}'] = {'carried_e_pu': c['carried_e_pu'],
                                                           'rated_e_pu': c['rated_e_pu'],
                                                           'equal': c['carried_e_pu'] == c['rated_e_pu']}
    q5_unit_2030_2035 = [v['below_rated'] for k_, v in q5['unit_n7'].items() if k_.startswith(('2030', '2035'))]
    q5_unit_2025 = [v['below_rated'] for k_, v in q5['unit_n7'].items() if k_.startswith('2025')]
    predictions_out = {
        'Q1': {'prediction': PREDICTIONS['Q1'], 'nonconvex_rows_after_relaxation_total': nonconvex_total,
               'held': nonconvex_total == 0},
        'Q2': {'prediction': PREDICTIONS['Q2'], 'groups': groups_done, 'n_block_mode_checks': len(sp_rows),
               'max_same_point_delta': max((r['achieved'] for r in sp_rows), default=None),
               'held': bool(sp_rows) and all(r['pass'] for r in sp_rows)
                       and all(len(v) == 12 for v in groups_done.values())},
        'Q3': {'prediction': PREDICTIONS['Q3'], 'residual_equals_mismatch_all_rows': cpl_eq_ok, 'per_channel': q3,
               'held': cpl_eq_ok and all(v['le_eps_pri'] for v in q3.values())},
        'Q4': {'prediction': PREDICTIONS['Q4'], 'max_block_rel': obj_block_max, 'sum_rel_vs_recorded': sum_rel,
               'held': obj_block_max is not None and obj_block_max <= 1e-12
                       and all(v <= 1e-8 for v in sum_rel.values())},
        'Q5': {'prediction': PREDICTIONS['Q5'], 'unit_storage_copies': q5['unit_n7'],
               'x0_storage_copies_equal_rated': all(v['equal'] for v in q5['x0'].values()),
               'unit_2030_2035_all_below_rated': bool(q5_unit_2030_2035) and all(q5_unit_2030_2035),
               'unit_2025_below_rated': q5_unit_2025,
               'held': (bool(q5_unit_2030_2035) and all(q5_unit_2030_2035)
                        and all(v['equal'] for v in q5['x0'].values()))},
    }
    gates_out = {'rows': gate_rows, 'all_pass': all(r['pass'] for r in gate_rows),
                 'failed': [r for r in gate_rows if not r['pass']], 'stops': stops, 'stop_triggered': bool(stops),
                 'thresholds': THRESHOLDS}
    identities_out.update({'gates': gates_out, 'predictions': predictions_out, 'guard': guard_out,
                           'containment': containment, 'wall_time_s': time.time() - t_start})

    payloads = {'inventory.json': inventory_out, 'identities.json': identities_out, 'coupling.json': coupling_out,
                'objective_identity.json': objective_out, 'lp/index.json': lp_index}
    unfilled = [(q, f, k_) for q, (f, k_) in REQUIRED_CAPTURES.items() if not payloads.get(f, {}).get(k_)]
    if unfilled:
        raise RuntimeError(f'capture check: unfilled required quantities {unfilled}')
    for name, payload in payloads.items():
        write_json(name, payload, compact=True)
        print(f'[write] {name}')

    failed_names = sorted({r['gate'] for r in gates_out['failed']})
    print(f'[summary] gates: {len(gate_rows)} rows, all pass: {gates_out["all_pass"]}; failed gate kinds: '
          f'{failed_names}')
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
