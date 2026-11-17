"""P5.15 Addendum 42 item 2 (task W66) -- NL-VARIABLE COUNT vs MODEL VARIABLES at x = 0. ZERO SOLVES.

For the x = 0 candidate on one instance (SRP1 1x1, or the 2x2 pilot instance the alpha-row spec pins), build every
local block production hands IPOPT through production's builders, and at the moment production would call the solver
(the network / ESSO `.optimize` is replaced ON THE INSTANCE by a stub that writes the .nl of every solve unit and
returns no result) write the .nl through `Block.write(format='nl')` -- the same `WriterFactory('nl')` /
`NLWriter.__call__` path `SolverFactory('ipopt').solve` uses (linear_presolve and scaling forced off there) -- once with
symbolic labels off (production form) and once on (names, .row/.col).

Two capture phases, both production call sequences:
  init   the initialisation solves inside create_distribution_networks_models / create_transmission_network_model /
         create_shared_energy_storage_model (`_run_operational_planning`, fresh branch);
  cycle1 the first ADMM cycle's solves, after production's ADMM preparation (`_prepare_*_objectives_for_admm`,
         `update_*_to_admm`, `_initialize_shared_ess_consensus`, `get_updated_capacities`) through
         `update_distribution_coordination_models_and_solve` / `update_transmission_coordination_model_and_solve` /
         `update_shared_energy_storages_coordination_model_and_solve`.
  Deviations from `_run_operational_planning` between the two phases, stated (value-only; none alters which
  components exist, which are active, or which Vars are fixed -- except as noted):
    - `_admm_local_solves_succeeded` is not evaluated (no solve happened);
    - the objective scale is the case file's fixed sigma (`admm.objective_scale`), which production uses whenever it
      is set; `_compute_common_admm_objective_scale` is not called (it evaluates objectives at a solved point);
    - `update_interface_power_flow_variables` is not called (it only updates consensus/dual VALUES from results);
    - pristine snapshot bases are not built (None is passed; they are clones used only for failure snapshots).

Per block the probe accounts EXACTLY for model Vars vs .nl columns:
    N_model = N_written + N_fixed + N_unwritten_free
and classifies the unwritten free Vars (referenced by an active constraint/objective but folded away -- zero
coefficient; referenced only by inactive components; referenced by no constraint/objective at all). From the parsed
.nl it lists every written column that appears in no constraint with a nonzero linear coefficient, in no nonlinear
constraint expression, and has neither a nonzero objective gradient coefficient nor a nonlinear objective appearance
(the HAZARD class), plus the zero-cost-ray signature of the row-18 defect: costless columns that appear in exactly one
row, linearly, whose admissible unbounded directions can move without violating that row.

`SolveProfileGuard(permitted=())` is armed for the whole run and verified at exactly 0.
"""

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W66 NL var-count probe (never solves)').install()

import pyomo  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

INSTANCES = {
    'srp1': {'case_path': 'data/SRP1/SRP1.json',
             'case_sha256': '61a794a7ce7a3fb983f2e92128eec446dd7dbb17ad75a3b3b1c5735f8bd4e4ef',
             'scenario_checksum': '5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358'},
    '2x2': {'case_path': 'data/SRP1/Results/P515S52/pilot_instance/SRP1__s52_pilot_2x2.json',
            'case_sha256': '7ecff44a874892187d1dd2e3d5ed0664a2f90a4bfa4cdb820a4abcbbd828f949',
            'scenario_checksum': '53b4bea4142001617282a08541079952564903b6d84db066f90d71a650b7d563'},
}
ROW_SPEC = {'path': 'data/SRP1/Results/P515S53/alpha_row/campaign_s53_alpha_row_v25/'
                    'campaign_spec_s53_alpha_row_v25_70965374.json',
            'sha256': '7096537486c86a7101e7948fe1931085e29b43b9700b62748799e368f8305570'}
PARAMS_FILE = {'path': 'data/SRP1/SRP1_params.json',
               'sha256': 'dbfdb2a07d12bfedab5e66bf94b3df98e5ab0305616652b4083266476972006b'}
X0_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
MIN_AVAILABLE_GIB = 6.0
RETIRED_EXPRESSION_COMPONENTS = ('scenario_deviation_penalty', 'scenario_deviation_weight',
                                 'scenario_deviation_voltage', 'scenario_deviation_interface_power',
                                 'scenario_deviation_shared_ess', 'scenario_tracking_penalty',
                                 'scenario_tracking_weight')


def _log(msg):
    print(f'[W66-nl-varcount] {msg}', flush=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def memory_available_gib():
    """The alpha-row campaign's own definition: hw.memsize - (wired + anonymous + compressor-occupied) x page."""
    text = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    page = int(text.split('page size of')[1].split()[0])
    vals = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            try:
                vals[k.strip()] = int(v.strip().rstrip('.'))
            except ValueError:
                pass
    memsize = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout)
    used = (vals['Pages wired down'] + vals['Anonymous pages'] + vals['Pages occupied by compressor']) * page
    free_inactive = (vals['Pages free'] + vals['Pages inactive']) * page
    return {'available_gib': (memsize - used) / 2 ** 30, 'free_plus_inactive_gib': free_inactive / 2 ** 30,
            'pages_free_gib': vals['Pages free'] * page / 2 ** 30,
            'own_maxrss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30}


def memory_gate(where, record):
    m = memory_available_gib()
    record.append({'where': where, **m})
    _log(f'memory at {where}: available {m["available_gib"]:.2f} GiB (free+inactive {m["free_plus_inactive_gib"]:.2f}),'
         f' own max RSS {m["own_maxrss_gib"]:.2f} GiB')
    if m['available_gib'] < MIN_AVAILABLE_GIB:
        raise SystemExit(f'STOP: available memory {m["available_gib"]:.2f} GiB < {MIN_AVAILABLE_GIB} at {where}')


# ======================================================================================================================
#  .nl parsing
# ======================================================================================================================
_HEADERS = set('COVLFSJGdxrbk')


def parse_nl(path):
    with open(path) as handle:
        lines = [ln.split('#', 1)[0].strip() for ln in handle]
    head = [int(x) for x in lines[1].split()]
    n_vars, n_cons, n_objs = head[0], head[1], head[2]
    con_refs, obj_refs = {}, {}
    v_lin, v_refs = {}, {}
    J, G = {}, {}
    row_bounds, col_bounds = [None] * n_cons, [None] * n_vars
    i, n = 10, len(lines)

    def read_expr(start):
        refs = set()
        j = start
        while j < n and (not lines[j] or lines[j][0] not in _HEADERS):
            t = lines[j]
            if t and t[0] == 'v':
                refs.add(int(t[1:].split()[0]))
            j += 1
        return refs, j

    while i < n:
        t = lines[i]
        if not t:
            i += 1
            continue
        h, rest = t[0], t[1:].split()
        if h in 'CO':
            idx = int(rest[0])
            refs, i = read_expr(i + 1)
            (con_refs if h == 'C' else obj_refs)[idx] = refs
        elif h == 'V':
            k, nlin = int(rest[0]), int(rest[1])
            v_lin[k] = {}
            for ln in lines[i + 1:i + 1 + nlin]:
                c, a = ln.split()
                v_lin[k][int(c)] = float(a)
            refs, i = read_expr(i + 1 + nlin)
            v_refs[k] = refs
        elif h == 'F':
            i += 1
        elif h == 'S':
            i += 1 + int(rest[1])
        elif h in 'dx':
            i += 1 + int(rest[0])
        elif h == 'r':
            for r in range(n_cons):
                row_bounds[r] = [float(x) if q else int(x) for q, x in enumerate(lines[i + 1 + r].split())]
            i += 1 + n_cons
        elif h == 'b':
            for c in range(n_vars):
                col_bounds[c] = [float(x) if q else int(x) for q, x in enumerate(lines[i + 1 + c].split())]
            i += 1 + n_vars
        elif h == 'k':
            i += 1 + int(rest[0])
        elif h in 'JG':
            idx, cnt = int(rest[0]), int(rest[1])
            d = {}
            for ln in lines[i + 1:i + 1 + cnt]:
                c, a = ln.split()
                d[int(c)] = float(a)
            (J if h == 'J' else G)[idx] = d
            i += 1 + cnt
        else:
            raise RuntimeError(f'unhandled .nl segment {t!r} in {path}')

    resolved = {}

    def resolve(refs, stack=()):
        out = set()
        for r in refs:
            if r < n_vars:
                out.add(r)
            else:
                if r not in resolved:
                    if r in stack:
                        raise RuntimeError('cyclic defined variable')
                    resolved[r] = set(v_lin.get(r, {})) | resolve(v_refs.get(r, set()), stack + (r,))
                out |= resolved[r]
        return out

    con_nl = {r: resolve(refs) for r, refs in con_refs.items()}
    obj_nl = {o: resolve(refs) for o, refs in obj_refs.items()}
    return {'n_vars': n_vars, 'n_cons': n_cons, 'n_objs': n_objs, 'J': J, 'G': G, 'con_nl': con_nl,
            'obj_nl': obj_nl, 'row_bounds': row_bounds, 'col_bounds': col_bounds}


def _bounds_text(b):
    kind = b[0]
    return {0: lambda: {'lb': b[1], 'ub': b[2]}, 1: lambda: {'lb': None, 'ub': b[1]},
            2: lambda: {'lb': b[1], 'ub': None}, 3: lambda: {'lb': None, 'ub': None},
            4: lambda: {'lb': b[1], 'ub': b[1]}}[kind]()


def column_structure(nl, col_names, row_names):
    n = nl['n_vars']
    jnz = [dict() for _ in range(n)]
    jzero = [set() for _ in range(n)]
    for r, d in nl['J'].items():
        for c, a in d.items():
            (jnz[c].__setitem__(r, a) if a != 0.0 else jzero[c].add(r))
    nl_rows = [set() for _ in range(n)]
    for r, cols in nl['con_nl'].items():
        for c in cols:
            nl_rows[c].add(r)
    gcoef = [0.0] * n
    for _o, d in nl['G'].items():
        for c, a in d.items():
            gcoef[c] += abs(a)
    nl_obj = set()
    for _o, cols in nl['obj_nl'].items():
        nl_obj |= cols

    hazard, objective_only, bound_fixed, costless_singletons = [], [], [], {}
    for c in range(n):
        b = _bounds_text(nl['col_bounds'][c])
        in_con = bool(jnz[c]) or bool(nl_rows[c])
        costed = gcoef[c] != 0.0 or c in nl_obj
        if b['lb'] is not None and b['lb'] == b['ub']:
            bound_fixed.append(col_names[c])
        if not in_con and not costed:
            hazard.append({'name': col_names[c], **b, 'rows_with_zero_coefficient': [row_names[r] for r in jzero[c]]})
        elif not in_con:
            objective_only.append({'name': col_names[c], **b, 'objective_linear_abs': gcoef[c],
                                   'objective_nonlinear': c in nl_obj})
        if not costed and not nl_rows[c] and len(jnz[c]) == 1:
            (r, a), = jnz[c].items()
            costless_singletons.setdefault(r, []).append((c, a, b))

    rays = []
    for r, entries in costless_singletons.items():
        rb = nl['row_bounds'][r]
        kind = rb[0]
        signs = set()
        for _c, a, b in entries:
            if b['ub'] is None:
                signs.add(1 if a > 0 else -1)
            if b['lb'] is None:
                signs.add(-1 if a > 0 else 1)
        ray = ((kind in (0, 4) and signs >= {1, -1}) or (kind == 1 and -1 in signs)
               or (kind == 2 and 1 in signs) or (kind == 3 and signs))
        if ray:
            rays.append({'row': row_names[r], 'row_kind': kind,
                         'columns': [{'name': col_names[c], 'coef': a, **b} for c, a, b in entries]})
    n_costless_singleton_rows = len(costless_singletons)
    return {'hazard_isolated_costless': hazard, 'objective_only_columns': objective_only,
            'bound_fixed_columns_by_family': _family_counts(bound_fixed),
            'zero_cost_ray_rows': rays, 'rows_with_costless_singleton_columns': n_costless_singleton_rows,
            'n_columns_costless': sum(1 for c in range(n) if gcoef[c] == 0.0 and c not in nl_obj),
            'n_columns_nonlinear_in_constraints': sum(1 for c in range(n) if nl_rows[c]),
            'n_columns_with_zero_linear_coefficient_only_rows': sum(
                1 for c in range(n) if not jnz[c] and jzero[c] and not nl_rows[c])}


def _family(name):
    return name.split('[')[0]


def _family_counts(names):
    out = {}
    for nm in names:
        out[_family(nm)] = out.get(_family(nm), 0) + 1
    return dict(sorted(out.items()))


# ======================================================================================================================
#  one solve unit
# ======================================================================================================================
def _chain_active(comp):
    blk = comp.parent_block()
    while blk is not None:
        if not blk.active:
            return False
        blk = blk.parent_block()
    return True


def analyse_block(block, stem, nl_dir, keep_files):
    rec = {'nl': {}}
    for flag, tag in ((False, 'labels_off'), (True, 'labels_on')):
        path = os.path.join(nl_dir, f'{stem}_{tag}.nl')
        if os.path.exists(path):
            raise SystemExit(f'REFUSED: exists: {path}')
        block.write(path, format='nl', io_options={'symbolic_solver_labels': flag})
        with open(path) as handle:
            handle.readline()
            line2 = handle.readline().rstrip('\n')
        e = {'sha256': _sha256_file(path), 'bytes': os.path.getsize(path), 'header_line2': line2}
        if flag:
            for suffix in ('row', 'col'):
                e[f'{suffix}_sha256'] = _sha256_file(path[:-3] + f'.{suffix}')
        rec['nl'][tag] = e
    on = os.path.join(nl_dir, f'{stem}_labels_on.nl')
    nl = parse_nl(on)
    with open(on[:-3] + '.col') as handle:
        col_names = [ln.rstrip('\n') for ln in handle]
    with open(on[:-3] + '.row') as handle:
        row_names = [ln.rstrip('\n') for ln in handle]
    if not keep_files:
        for tag in ('labels_off', 'labels_on'):
            base = os.path.join(nl_dir, f'{stem}_{tag}')
            for suffix in ('.nl', '.row', '.col'):
                if os.path.exists(base + suffix):
                    os.remove(base + suffix)
    header_same = rec['nl']['labels_off']['header_line2'] == rec['nl']['labels_on']['header_line2']
    assert len(col_names) == nl['n_vars'], (len(col_names), nl['n_vars'])

    # ---- model-side census ----
    sub_blocks = sum(1 for _ in block.component_data_objects(pe.Block, descend_into=True))
    all_vars = list(block.component_data_objects(pe.Var, descend_into=True))
    by_name = {v.getname(fully_qualified=True): v for v in all_vars}
    if len(by_name) != len(all_vars):
        raise RuntimeError('non-unique Var names in block')
    written_ids = set()
    for nm in col_names:
        v = by_name.get(nm)
        if v is None:
            raise RuntimeError(f'.col name {nm!r} not resolvable to a model Var')
        written_ids.add(id(v))
    ref_active, ref_inactive = set(), set()
    n_con_active = n_con_inactive = n_obj_active = n_obj_inactive = 0
    for ctype in (pe.Constraint, pe.Objective):
        for comp in block.component_data_objects(ctype, active=None, descend_into=True):
            act = comp.active and _chain_active(comp)
            expr = comp.body if ctype is pe.Constraint else comp.expr
            ids = {id(v) for v in identify_variables(expr, include_fixed=True)}
            if act:
                ref_active |= ids
            else:
                ref_inactive |= ids
            if ctype is pe.Constraint:
                n_con_active += act
                n_con_inactive += not act
            else:
                n_obj_active += act
                n_obj_inactive += not act

    fam = {}
    fixed_written = []
    cats = {'written': [], 'fixed_referenced_by_active': [], 'fixed_unreferenced_by_active': [],
            'free_unwritten_referenced_by_active': [], 'free_unwritten_referenced_only_by_inactive': [],
            'free_unwritten_unreferenced': []}
    for nm, v in by_name.items():
        vid = id(v)
        if vid in written_ids:
            if v.fixed:
                fixed_written.append(nm)
            cat = 'written'
        elif v.fixed:
            cat = 'fixed_referenced_by_active' if vid in ref_active else 'fixed_unreferenced_by_active'
        elif vid in ref_active:
            cat = 'free_unwritten_referenced_by_active'
        elif vid in ref_inactive:
            cat = 'free_unwritten_referenced_only_by_inactive'
        else:
            cat = 'free_unwritten_unreferenced'
        cats[cat].append(nm)
        f = fam.setdefault(_family(nm), {k: 0 for k in cats})
        f[cat] += 1
    n_model = len(all_vars)
    n_fixed = sum(1 for v in all_vars if v.fixed)
    n_written = len(written_ids)
    n_unwritten_free = (len(cats['free_unwritten_referenced_by_active'])
                        + len(cats['free_unwritten_referenced_only_by_inactive'])
                        + len(cats['free_unwritten_unreferenced']))
    struct = column_structure(nl, col_names, row_names)
    rec.update({
        'n_model_vars': n_model, 'n_fixed': n_fixed, 'n_free': n_model - n_fixed,
        'n_nl_columns': nl['n_vars'], 'n_nl_rows': nl['n_cons'], 'n_nl_objs': nl['n_objs'],
        'n_unwritten_free': n_unwritten_free,
        'identity_holds': n_model == n_written + n_fixed + n_unwritten_free and n_written == nl['n_vars'],
        'fixed_vars_written': fixed_written,
        'category_counts': {k: len(v) for k, v in cats.items()},
        'unwritten_free_names_first_50': {k: sorted(cats[k])[:50] for k in cats if k.startswith('free_unwritten')},
        'families': dict(sorted(fam.items())),
        'model_constraints_active': n_con_active, 'model_constraints_inactive': n_con_inactive,
        'model_objectives_active': n_obj_active, 'model_objectives_inactive': n_obj_inactive,
        'sub_blocks': sub_blocks,
        'header_counts_identical_between_label_settings': header_same,
        'retired_expression_components_present': [c for c in RETIRED_EXPRESSION_COMPONENTS
                                                   if hasattr(block, c)],
        'row18_wired': hasattr(block, 'row18_alpha'),
        'structure': struct,
    })
    if hasattr(block, 'row18_alpha'):
        rec['row18'] = {'alpha': float(pe.value(block.row18_alpha)),
                        'rows_active': sum(1 for nmr in ('row18_dev_p_def', 'row18_dev_q_def')
                                           for i in getattr(block, nmr) if getattr(block, nmr)[i].active),
                        'dev_vars_fixed': sum(1 for nmv in srp._ROW18_DEVIATION_FAMILIES for nm2 in nmv[1:3]
                                              for i in getattr(block, nm2) if getattr(block, nm2)[i].fixed)}
    return rec


# ======================================================================================================================
#  capture stubs
# ======================================================================================================================
class Capture:
    def __init__(self, nl_dir, keep_files):
        self.nl_dir, self.keep_files = nl_dir, keep_files
        self.phase = None
        self.records = {}
        self.calls = {}

    def _take(self, key, block):
        full = f'{self.phase}|{key}'
        if full in self.records:
            raise RuntimeError(f'duplicate capture {full}')
        stem = full.replace('|', '__').replace(':', '_')
        self.records[full] = analyse_block(block, stem, self.nl_dir, self.keep_files)
        r = self.records[full]
        _log(f'  {full}: model vars {r["n_model_vars"]} = columns {r["n_nl_columns"]} + fixed {r["n_fixed"]} + '
             f'unwritten free {r["n_unwritten_free"]}  (identity {r["identity_holds"]}); rows {r["n_nl_rows"]}; '
             f'hazard {len(r["structure"]["hazard_isolated_costless"])}, rays '
             f'{len(r["structure"]["zero_cost_ray_rows"])}')
        del block

    def network_stub(self, agent, node_id, network):
        def stub(model, *args, **kwargs):
            k = f'{self.phase}|{agent}:{node_id}'
            self.calls[k] = self.calls.get(k, 0) + 1
            for y in network.years:
                for d in network.days:
                    self._take(f'{agent}:{node_id}:{y}:{d}', model[y][d])
            return {y: {d: None for d in network.days} for y in network.years}
        return stub

    def esso_stub(self, sed):
        def stub(models, *args, **kwargs):
            k = f'{self.phase}|ESSO'
            self.calls[k] = self.calls.get(k, 0) + 1
            for node_id in sed.active_distribution_network_nodes:
                self._take(f'ESSO:{node_id}', models[node_id])
            return {node_id: None for node_id in sed.active_distribution_network_nodes}
        return stub


class _NullStages:
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def stage(self, _name):
        return self._Ctx()


def redirect_dirs(planning, root):
    res, logs = os.path.join(root, 'Results'), os.path.join(root, 'Results', 'Logs')
    os.makedirs(logs, exist_ok=True)
    planning.results_dir, planning.logs_dir = res, logs
    holders = [planning.transmission_network] + list(planning.distribution_networks.values())
    for h in holders:
        h.results_dir, h.logs_dir = res, logs
        for y in h.years:
            for d in h.days:
                h.network[y][d].results_dir = res
                h.network[y][d].logs_dir = logs
    planning.shared_ess_data.results_dir = res
    if hasattr(planning.shared_ess_data, 'logs_dir'):
        planning.shared_ess_data.logs_dir = logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True, choices=sorted(INSTANCES))
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--keep-nl', action='store_true')
    args = ap.parse_args()
    scratch = os.path.abspath(args.scratch)
    if scratch.startswith(REPO + os.sep):
        raise SystemExit('--scratch must be outside the repository')
    out_root = os.path.abspath(args.out_root)
    if not out_root.startswith(os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53') + os.sep):
        raise SystemExit('--out-root must be under data/SRP1/Results/P515S53/')
    out_dir = os.path.join(out_root, args.label)
    work = os.path.join(scratch, args.label)
    for p in (out_dir, work):
        if os.path.exists(p):
            raise SystemExit(f'REFUSED (write-once): exists: {p}')
    inst = INSTANCES[args.instance]
    mem = []

    # ---- capture-path checklist, before anything is built ----
    checklist = {
        'case_sha256_matches': _sha256_file(os.path.join(REPO, inst['case_path'])) == inst['case_sha256'],
        'params_file_sha256_matches': _sha256_file(os.path.join(REPO, PARAMS_FILE['path'])) == PARAMS_FILE['sha256'],
        'row_spec_sha256_matches': _sha256_file(os.path.join(REPO, ROW_SPEC['path'])) == ROW_SPEC['sha256'],
        'writer_forces_presolve_off': 'config.linear_presolve = False' in open(
            os.path.join(os.path.dirname(pyomo.__file__), 'repn', 'plugins', 'nl_writer.py')).read(),
        'production_builders_present': all(hasattr(srp, f) for f in (
            'create_distribution_networks_models', 'create_transmission_network_model',
            'create_shared_energy_storage_model', '_prepare_distribution_objectives_for_admm',
            'update_distribution_models_to_admm', 'update_distribution_coordination_models_and_solve')),
    }
    if args.instance == '2x2':
        spec = json.load(open(os.path.join(REPO, ROW_SPEC['path'])))
        di = spec['configuration']['derived_instance']
        checklist['row_spec_pins_this_case'] = (di['case_path'] == inst['case_path']
                                                and di['case_sha256'] == inst['case_sha256']
                                                and di['scenario_checksum'] == inst['scenario_checksum'])
        checklist['alpha_in_row_spec'] = any(float(c['interface_deviation_premium']['alpha']) == args.alpha
                                             and c['key'] == X0_KEY for c in spec['candidates'])
    if not all(checklist.values()):
        raise SystemExit(f'REFUSED: checklist failed: {checklist}')
    _log(f'instance {args.instance} alpha {args.alpha} label {args.label}; checklist {checklist}')
    memory_gate('before anything is written or built', mem)
    os.makedirs(out_dir)
    os.makedirs(os.path.join(work, 'nl'))

    # ---- read (production reader; dirs redirected into scratch BEFORE reading) ----
    if args.instance == 'srp1':
        planning = SharedResourcesPlanning(S.DATA_DIR, 'SRP1.json')
        planning.name = 'SRP1'
        planning.results_dir = os.path.join(work, 'planning_read', 'Results')
        planning.diagrams_dir = os.path.join(work, 'planning_read', 'Diagrams')
        planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
        planning.read_planning_problem()
    else:
        planning = S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}}, work, _NullStages())
    checksum = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
    if checksum != inst['scenario_checksum']:
        raise SystemExit(f'REFUSED: scenario checksum {checksum} != {inst["scenario_checksum"]}')
    redirect_dirs(planning, os.path.join(work, 'run'))
    planning.parallel_execution = False
    a = planning.params.admm
    a.interface_deviation_premium = {'alpha': args.alpha, 'floor': None, 'source': 'W66 probe (--alpha)'}
    sed = planning.shared_ess_data

    # ---- the x = 0 candidate, exactly as _construct_arm_planning builds it ----
    canonical = H.canonical_candidate({n: (0.0, 0.0) for n in sed.active_distribution_network_nodes})
    key = H.candidate_key(canonical)
    if key != X0_KEY:
        raise SystemExit(f'REFUSED: x = 0 candidate key {key} != {X0_KEY}')
    candidate = planning.get_initial_candidate_solution()
    for node_id, (s_val, e_val) in H.investment_map_from_canonical(canonical).items():
        candidate['investment'][node_id][canonical['investment_year']]['s'] = s_val
        candidate['investment'][node_id][canonical['investment_year']]['e'] = e_val
    srp._rebuild_candidate_total_capacities(planning, candidate)
    memory_gate('after read', mem)

    cap = Capture(os.path.join(work, 'nl'), args.keep_nl)
    tn, dns = planning.transmission_network, planning.distribution_networks
    for node_id, dn in dns.items():
        dn.optimize = cap.network_stub('DSO', node_id, dn)
    tn.optimize = cap.network_stub('TSO', tn.name, tn)
    sed.optimize = cap.esso_stub(sed)
    deviations = []
    try:
        # ---------------- phase init: _run_operational_planning, fresh branch ----------------
        cap.phase = 'init'
        consensus_vars, dual_vars = srp.create_admm_variables(planning)
        prem = a.interface_deviation_premium
        dso_models, _r = srp.create_distribution_networks_models(dns, consensus_vars, candidate['total_capacity'],
                                                                 parallel_execution=False,
                                                                 premium_alpha=prem['alpha'],
                                                                 premium_floor=prem['floor'])
        memory_gate('after DSO init builds', mem)
        tso_model, _r = srp.create_transmission_network_model(planning, consensus_vars, candidate['total_capacity'])
        esso_model, _r = srp.create_shared_energy_storage_model(sed, consensus_vars, candidate['investment'])
        memory_gate('after init phase', mem)

        # ---------------- ADMM preparation (production functions, production order) ----------------
        none_valued = {}
        for node_id, m in dso_models.items():
            for y in dns[node_id].years:
                for d in dns[node_id].days:
                    blk = m[y][d]
                    if hasattr(blk, 'row18_alpha'):
                        for fam_row in srp._ROW18_DEVIATION_FAMILIES:
                            flow = getattr(blk, fam_row[3])
                            for idx in flow:
                                if flow[idx].value is None:
                                    flow[idx].set_value(0.0)
                                    none_valued[fam_row[3]] = none_valued.get(fam_row[3], 0) + 1
        if none_valued:
            deviations.append({'value_only': 'row-18 flow Vars had no value (no init solve); set to 0.0 before '
                                             '_activate_row18_with_settlement reads them', 'counts': none_valued})
        srp._prepare_distribution_objectives_for_admm(dns, dso_models)
        srp._prepare_transmission_objectives_for_admm(tn, tso_model)
        if a.objective_scale is None:
            raise SystemExit('REFUSED: admm.objective_scale not fixed in the case file')
        objective_scale = a.objective_scale
        al_scale_esso = srp._resolve_esso_al_scale(planning, a, objective_scale)[0]
        srp.update_distribution_models_to_admm(planning, dso_models, a, objective_scale)
        srp.update_transmission_model_to_admm(planning, tso_model, a, objective_scale)
        srp.update_shared_energy_storage_model_to_admm(planning, esso_model, a, al_scale_esso=al_scale_esso)
        srp._initialize_shared_ess_consensus(planning, consensus_vars)
        if getattr(a, 'shared_ess_initialization', 'standalone') != 'standalone':
            raise SystemExit('REFUSED: shared_ess_initialization is not standalone')
        sess_caps = sed.get_updated_capacities(esso_model)

        # ---------------- phase cycle1: the first ADMM cycle's solve calls ----------------
        cap.phase = 'cycle1'
        srp.update_distribution_coordination_models_and_solve(
            dns, dso_models, consensus_vars['vmag'], dual_vars['vmag']['dso'], consensus_vars['pf'],
            dual_vars['pf']['dso'], consensus_vars['ess'], dual_vars['ess']['dso'], a, sess_caps,
            from_warm_start=True, parallel_execution=False, cycle=1, dso_pristine_base=None)
        srp.update_transmission_coordination_model_and_solve(
            tn, tso_model, consensus_vars['vmag'], dual_vars['vmag']['tso'], consensus_vars['pf'],
            dual_vars['pf']['tso'], consensus_vars['ess'], dual_vars['ess']['tso'], a, sess_caps,
            from_warm_start=True, cycle=1, tso_pristine_base=None)
        srp.update_shared_energy_storages_coordination_model_and_solve(
            planning, esso_model, consensus_vars['ess']['z'], dual_vars['ess']['esso'], a, from_warm_start=True,
            cycle=1)
        memory_gate('after cycle1 phase', mem)
    finally:
        for dn in dns.values():
            dn.__dict__.pop('optimize', None)
        tn.__dict__.pop('optimize', None)
        sed.__dict__.pop('optimize', None)

    # ---- aggregate ----
    def agg(phase, agent=None):
        recs = {k: r for k, r in cap.records.items() if k.startswith(phase + '|')
                and (agent is None or k.split('|')[1].startswith(agent + ':'))}
        tot = {x: sum(r[x] for r in recs.values()) for x in ('n_model_vars', 'n_nl_columns', 'n_fixed',
                                                             'n_unwritten_free', 'n_nl_rows')}
        cats, fams = {}, {}
        for r in recs.values():
            for c, v in r['category_counts'].items():
                cats[c] = cats.get(c, 0) + v
            for f, d in r['families'].items():
                t = fams.setdefault(f, {})
                for c, v in d.items():
                    t[c] = t.get(c, 0) + v
        return {'n_blocks': len(recs), **tot, 'category_counts': cats,
                'identity_holds_every_block': all(r['identity_holds'] for r in recs.values()),
                'hazard_total': sum(len(r['structure']['hazard_isolated_costless']) for r in recs.values()),
                'zero_cost_ray_rows_total': sum(len(r['structure']['zero_cost_ray_rows']) for r in recs.values()),
                'objective_only_columns_total': sum(len(r['structure']['objective_only_columns'])
                                                    for r in recs.values()),
                'fixed_vars_written_total': sum(len(r['fixed_vars_written']) for r in recs.values()),
                'families_not_fully_written': {f: d for f, d in sorted(fams.items()) if d.get('written', 0) != sum(
                    d.values())},
                'families_fully_written': sorted(f for f, d in fams.items() if d.get('written', 0) == sum(
                    d.values()))}

    summary = {ph: {'all': agg(ph), **{ag: agg(ph, ag) for ag in ('DSO', 'TSO', 'ESSO')}}
               for ph in ('init', 'cycle1')}
    GUARD.uninstall()
    verify_failures = GUARD.verify(expected_solves=0)
    payload = {
        'schema': 'p515_s53_nl_varcount_probe_v1',
        'stage': 'P5.15 Addendum 42 item 2 (W66): NL-variable count vs model variables at x = 0',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable, 'argv': sys.argv,
        'pyomo_version': pyomo.version.version,
        'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'production_sha256': {f: _sha256_file(os.path.join(REPO, f)) for f in (
            'shared_resources_planning.py', 'model_construction_helpers.py', 'network.py', 'network_data.py',
            'shared_energy_storage_data.py', 'admm_parameters.py')},
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'capture_path_checklist': checklist,
        'instance': {'name': args.instance, **inst, 'scenario_checksum_observed': checksum,
                     'row_spec': ROW_SPEC if args.instance == '2x2' else None, 'params_file': PARAMS_FILE},
        'candidate': {'canonical': canonical, 'key': key, 'total_capacity_all_zero': all(
            v2 == 0.0 for n in candidate['total_capacity'].values() for yv in n.values() for v2 in yv.values())},
        'alpha': args.alpha, 'objective_scale_used': objective_scale, 'al_scale_esso': al_scale_esso,
        'deviations_from_run_operational_planning_value_only': deviations,
        'stub_calls': cap.calls, 'summary': summary, 'blocks': cap.records,
        'nl_files_kept': bool(args.keep_nl), 'scratch': work,
        'memory': mem,
        'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts), 'verify_failures': verify_failures},
    }
    out_path = os.path.join(out_dir, 'nl_varcount.json')
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    for ph in ('init', 'cycle1'):
        for ag in ('all', 'DSO', 'TSO', 'ESSO'):
            s = summary[ph][ag]
            _log(f'{ph} {ag}: blocks {s["n_blocks"]} model {s["n_model_vars"]} = columns {s["n_nl_columns"]} + fixed '
                 f'{s["n_fixed"]} + unwritten free {s["n_unwritten_free"]}; identity {s["identity_holds_every_block"]};'
                 f' hazard {s["hazard_total"]}; rays {s["zero_cost_ray_rows_total"]}; objective-only '
                 f'{s["objective_only_columns_total"]}; fixed written {s["fixed_vars_written_total"]}')
    _log(f'guard {dict(GUARD.counts)} verify_failures {verify_failures}; wrote {out_path}')
    return 0 if not verify_failures else 1


if __name__ == '__main__':
    sys.exit(main())
