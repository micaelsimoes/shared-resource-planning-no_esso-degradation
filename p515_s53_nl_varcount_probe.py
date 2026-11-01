"""P5.15 Addendum 42 item 2 (tasks W66 -> W68) -- NL-VARIABLE COUNT vs MODEL VARIABLES at x = 0. ZERO SOLVES.

Promoted from the W66 draft `data/SRP1/Results/P515S53/nl_varcount_w66/p515_s53_nl_varcount_probe_DRAFT_unrun.py`
(exercised only as far as its checklist and memory gate; that file is left in place, unchanged).

For the x = 0 candidate on one instance (SRP1 1x1, or the 2x2 pilot instance the alpha-row spec pins), build every
local block production hands IPOPT through production's builders, and at the moment production would call the solver
(the network / ESSO `.optimize` is replaced ON THE INSTANCE by a stub that writes the .nl of every solve unit and
returns no result) write the .nl through `Block.write(format='nl')` -- the same `WriterFactory('nl')` /
`NLWriter.__call__` path `SolverFactory('ipopt').solve` uses (linear_presolve and scaling forced off there) -- once with
symbolic labels off (production form) and once on (names, .row/.col). `NetworkData.optimize` -> `run_smopf` and
`SharedEnergyStorageData.optimize` -> `_optimize` only set solver options and warm-start suffixes before the solve, so
stubbing at `.optimize` changes nothing that decides which columns are written.

Two capture phases, both production call sequences:
  init   the initialisation solves inside create_distribution_networks_models / create_transmission_network_model /
         create_shared_energy_storage_model (`_run_operational_planning`, fresh branch);
  cycle1 the first ADMM cycle's solves, after production's ADMM preparation (`_prepare_*_objectives_for_admm`,
         `update_*_to_admm`, `_initialize_shared_ess_consensus`, `get_updated_capacities`) through
         `update_distribution_coordination_models_and_solve` / `update_transmission_coordination_model_and_solve` /
         `update_shared_energy_storages_coordination_model_and_solve`.
  Deviations from `_run_operational_planning` between/within the two phases, stated (value-only; none alters which
  components exist, which are active, or which Vars are fixed):
    - `_admm_local_solves_succeeded` is not evaluated (no solve happened);
    - the objective scale is the case file's fixed sigma (`admm.objective_scale`), which production uses whenever it
      is set; `_compute_common_admm_objective_scale` is not called (it evaluates objectives at a solved point);
    - `update_interface_power_flow_variables` and the in-cycle `update_and_check_convergence` calls are not made
      (they only update consensus/dual VALUES from solve results);
    - pristine snapshot bases are not built (None is passed; they are clones used only for failure snapshots);
    - a Var read by `_activate_row18_with_settlement` (directly, or inside the `pg_adn`/`qg_adn` Expressions) that
      has no value (no init solve) is set to 0.0 first (recorded with counts when it happens; W68 attempt r1 failed
      on the draft's assumption that `pg_adn` is a Var).

PER BLOCK the probe accounts EXACTLY for model Vars vs .nl columns:
    N_model = N_written + sum over the six unwritten categories
      fixed_referenced_by_active          fixed, folded to a constant by the writer
      fixed_referenced_only_by_inactive   fixed, and every reference is in a deactivated row/objective
      fixed_unreferenced                  fixed, referenced by no constraint/objective at all
      free_unwritten_referenced_by_active free, yet not written (the writer dropped a zero coefficient)
      free_unwritten_referenced_only_by_inactive
      free_unwritten_unreferenced
    and N_written must equal the .nl column count.

HAZARD CLASS (Planner ruling 3, W68): a column that IS written, carries NO cost, and lies in a direction NOTHING
constrains -- the row-18 defect (d+/d- written because they sat in active defining rows, costless once alpha was 0:
the direction d+ = d- = t leaves every row unchanged, so the barrier problem has no central path). Searched, per
block, on the parsed .nl (never on the Pyomo model), in three screens:
  H0  isolated: written, costless, and in no constraint (no nonzero linear coefficient, no nonlinear appearance);
  H1  null-space screen: over the costless, movable (lb != ub) columns that appear in constraints ONLY linearly, the
      connected components of the column-row incidence graph; for each, the null space of J[rows(C), C] (directions
      that change NO row at all). A null vector is reported with its columns and bounds, classified 'unbounded'
      (the row-18 class proper) or 'bounded_by_variable_bounds' (a flat face, related but not the same pathology);
  H2  one-sided: a single costless linear-only column whose every row is a one-sided inequality relaxed by one of its
      unbounded directions.
  SCOPE, stated: columns appearing in any NONLINEAR constraint are excluded from H1/H2 (counted per family instead);
  H2 covers single columns only (a combination relaxing several inequality rows jointly would need an LP, which this
  probe does not run). "Costless" = zero linear objective coefficient and no appearance in a nonlinear objective.

RETIRED / UNWIRED FAMILY TABLE: per Var family, the category counts split by scenario pair (first pair / other pairs /
no scenario index), plus named subsets (DSO reference-node voltage slacks, TSO ADN flex legs, interface_delta, row-18
pairs, shared-ESS copies, flex day-balance slacks, TSO ADN load curtailment/pc/qc) with, for written members, how they
are written (costed, in constraints, .nl bound type).

`SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 on EVERY exit path
(normal completion, refusal, memory stop, exception) -- `run_outcome.json` records it.

EXACT COMMANDS (W68; repo root, canonical interpreter, attached, BOTH streams captured, one at a time):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_nl_varcount_probe.py --instance srp1 \
      --alpha 0.0 --label srp1_a0p00_r1 --scratch <SCRATCH> --out-root data/SRP1/Results/P515S53/nl_varcount_w66 \
      > data/SRP1/Results/P515S53/nl_varcount_w66/srp1_a0p00_r1_launch.log 2>&1
  (same with --instance 2x2 --alpha 0.0 --label 2x2_a0p00_r1, and --instance 2x2 --alpha 0.5 --label 2x2_a0p50_r1)
  <SCRATCH> is any directory OUTSIDE the repository (refused otherwise); the .nl files written there are deleted after
  analysis unless --keep-nl, and their sha256 are recorded. Output (write-once): <out-root>/<label>/nl_varcount.json
  and <out-root>/<label>/run_outcome.json (the latter on EVERY exit path, with the guard verification).
  A run with --out-root inside the repository refuses unless this script is committed and unmodified.
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W68 NL var-count probe (never solves)').install()

import numpy as np  # noqa: E402
import pyomo  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

STAGE = 'P5.15 Addendum 42 item 2 (W68, re-run of W66): NL-variable count vs model variables at x = 0'
LOG_TAG = '[W68-nl-varcount]'
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
MIN_AVAILABLE_GIB = 6.0          # UNCHANGED from W66 (Planner ruling 1, W68)
NULLSPACE_COMPONENT_CAP = 4000   # H1 components with more columns are recorded as not evaluated (never silently)
NULL_WEIGHT_TOL = 1e-9
UNWRITTEN_CATEGORIES = ('fixed_referenced_by_active', 'fixed_referenced_only_by_inactive', 'fixed_unreferenced',
                        'free_unwritten_referenced_by_active', 'free_unwritten_referenced_only_by_inactive',
                        'free_unwritten_unreferenced')
CATEGORIES = ('written',) + UNWRITTEN_CATEGORIES
# Component names retired in production (source comments: network.py, model_construction_helpers.py,
# shared_energy_storage_data.py, shared_resources_planning.py); presence on a built block is recorded.
RETIRED_COMPONENT_NAMES = (
    'scenario_deviation_penalty', 'scenario_deviation_weight', 'scenario_deviation_voltage',
    'scenario_deviation_interface_power', 'scenario_deviation_shared_ess', 'scenario_tracking_penalty',
    'scenario_tracking_weight', 'scenario_tracking_voltage', 'scenario_tracking_interface_power',
    'shared_es_s_rated', 'shared_es_e_rated', 'shared_es_sch', 'shared_es_sdch', 'es_sch', 'es_sdch', 'es_snet',
    'shared_es_snet', 'es_s_investment', 'es_e_investment', 'es_soh_per_unit', 'es_degradation_per_unit',
    'es_degradation_per_unit_cumul', 'es_pch_hat_per_unit', 'es_pdch_hat_per_unit', 'slack_es_ch_comp_per_unit',
    'slack_es_snet_up', 'slack_es_snet_down')
RETIRED_NAME_FRAGMENTS = ('phi_limits', 'sensitivities', 'scenario_deviation', 'scenario_tracking')
ROW18_FAMILIES = ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up', 'row18_dev_q_down')
NO_SCENARIO_FAMILIES = ('expected_interface_vmag', 'expected_interface_pf_p', 'expected_interface_pf_q',
                        'expected_shared_ess_p', 'expected_shared_ess_q')

# ---- predictions, recorded in committed code BEFORE the run (W68) ----
PREDICTIONS = {
    'P1': 'identity N_model = N_written + sum(unwritten categories) and N_written = .nl columns holds on every block '
          'of every instance in both phases',
    'P2': 'no fixed Var is written (fixed_vars_written = 0 everywhere)',
    'P3': 'free_unwritten_referenced_by_active = 0 on every block (no free Var dropped for a zero coefficient)',
    'P4': 'on DSO and TSO blocks every unwritten Var is FIXED (all three free_unwritten_* = 0); the ESSO may differ',
    'P5': 'H0 (isolated costless written columns) = 0 on every block',
    'P6': 'H1 unbounded null directions = 0 and H2 one-sided rays = 0 on every block of all three instances',
    'P7': 'row 18 at alpha = 0: row18_dev_* ABSENT on every DSO block (2x2 and SRP1), both phases',
    'P8': 'row 18 at alpha = 0.5 (2x2): init phase -> all row18_dev_* FIXED and folded (referenced by the active '
          'objective through the charge); cycle1 -> all WRITTEN, costed, in constraints',
    'P9': 'TSO interface_delta_p/q: first scenario pair WRITTEN (free, bounded +-rating); other pairs FIXED and '
          'referenced by nothing',
    'P10': 'TSO ADN flex legs flex_p/q_up/down at ADN loads: FIXED, folded (referenced by active rows)',
    'P11': 'DSO reference-node slack_v_sqr_up/down: WRITTEN with .nl bound type 4 (lb = ub = 0), not fixed',
    'P12': 'shared-ESS operational copies at x = 0: ALL FIXED (zero-capacity gate); non-first pairs referenced by '
           'nothing, first pair folded',
    'P13': 'slack_flex_q_balance_*: all FIXED; slack_flex_p_balance_* at TSO ADN loads: FIXED',
    'P14': 'no RETIRED_COMPONENT_NAMES component is present on any built block',
    'P15': 'verdict: NO hazard of the row-18 class elsewhere in the model at x = 0',
}


def _log(msg):
    print(f'{LOG_TAG} {msg}', flush=True)


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


class MemoryStop(RuntimeError):
    pass


def memory_gate(where, record):
    m = memory_available_gib()
    record.append({'where': where, **m})
    _log(f'memory at {where}: available {m["available_gib"]:.2f} GiB (free+inactive {m["free_plus_inactive_gib"]:.2f}),'
         f' own max RSS {m["own_maxrss_gib"]:.2f} GiB')
    if m['available_gib'] < MIN_AVAILABLE_GIB:
        raise MemoryStop(f'STOP: available memory {m["available_gib"]:.2f} GiB < {MIN_AVAILABLE_GIB} at {where}')


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
    if any(b is None for b in row_bounds) or any(b is None for b in col_bounds):
        raise RuntimeError(f'incomplete r/b segments in {path}')
    return {'n_vars': n_vars, 'n_cons': n_cons, 'n_objs': n_objs, 'J': J, 'G': G, 'con_nl': con_nl,
            'obj_nl': obj_nl, 'row_bounds': row_bounds, 'col_bounds': col_bounds}


def _bounds(b):
    kind = b[0]
    return {0: lambda: {'lb': b[1], 'ub': b[2]}, 1: lambda: {'lb': None, 'ub': b[1]},
            2: lambda: {'lb': b[1], 'ub': None}, 3: lambda: {'lb': None, 'ub': None},
            4: lambda: {'lb': b[1], 'ub': b[1]}}[kind]()


def _family(name):
    return name.split('[')[0]


def _family_counts(names):
    out = {}
    for nm in names:
        out[_family(nm)] = out.get(_family(nm), 0) + 1
    return dict(sorted(out.items()))


def column_structure(nl, col_names, row_names):
    """Per-column flags and the three hazard screens, from the parsed .nl only."""
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
    bnd = [_bounds(nl['col_bounds'][c]) for c in range(n)]
    kind = [nl['col_bounds'][c][0] for c in range(n)]
    costed = [gcoef[c] != 0.0 or c in nl_obj for c in range(n)]
    in_con = [bool(jnz[c]) or bool(nl_rows[c]) for c in range(n)]
    movable = [not (bnd[c]['lb'] is not None and bnd[c]['lb'] == bnd[c]['ub']) for c in range(n)]

    def col_entry(c, **extra):
        return {'name': col_names[c], 'lb': bnd[c]['lb'], 'ub': bnd[c]['ub'], 'nl_bound_type': kind[c], **extra}

    # ---- H0 ----
    h0 = [col_entry(c, rows_with_zero_coefficient=[row_names[r] for r in sorted(jzero[c])])
          for c in range(n) if not in_con[c] and not costed[c]]
    objective_only = [col_entry(c, objective_linear_abs=gcoef[c], objective_nonlinear=c in nl_obj)
                      for c in range(n) if not in_con[c] and costed[c]]
    # ---- the screened set ----
    screened = [c for c in range(n) if not costed[c] and movable[c] and in_con[c] and not nl_rows[c]]
    costless_nonlinear = [col_names[c] for c in range(n) if not costed[c] and nl_rows[c]]
    costless_immobile = [col_names[c] for c in range(n) if not costed[c] and not movable[c]]
    # ---- H1: exact structural peeling, then connected components, then the null space per component ----
    # A direction d supported on the screened columns (every other column held at 0) must leave every row unchanged:
    # sum_c a_rc d_c = 0. A row in which exactly ONE remaining screened column has a nonzero coefficient forces that
    # column's weight to 0, so the column is removed; repeat to a fixed point. Exact -- it removes only columns that
    # cannot carry any null direction -- and it shrinks the dense SVD below to the columns that can.
    rows_of = {c: set(jnz[c]) for c in screened}
    cols_in_row = {}
    for c in screened:
        for r in jnz[c]:
            cols_in_row.setdefault(r, set()).add(c)
    alive = set(screened)
    stack = [r for r, cs in cols_in_row.items() if len(cs) == 1]
    while stack:
        r = stack.pop()
        cs = cols_in_row.get(r)
        if not cs or len(cs) != 1:
            continue
        (c,) = tuple(cs)
        alive.discard(c)
        for r2 in rows_of[c]:
            s2 = cols_in_row[r2]
            s2.discard(c)
            if len(s2) == 1:
                stack.append(r2)
    peeled = len(screened) - len(alive)
    parent = {c: c for c in alive}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_col_of_row = {}
    for c in alive:
        for r in jnz[c]:
            if r in first_col_of_row:
                ra, rb = find(c), find(first_col_of_row[r])
                if ra != rb:
                    parent[ra] = rb
            else:
                first_col_of_row[r] = c
    comps = {}
    for c in alive:
        comps.setdefault(find(c), []).append(c)
    h1, not_evaluated, max_comp = [], [], 0
    for cols in comps.values():
        max_comp = max(max_comp, len(cols))
        rows = sorted({r for c in cols for r in jnz[c]})
        if len(cols) > NULLSPACE_COMPONENT_CAP:
            not_evaluated.append({'n_columns': len(cols), 'n_rows': len(rows),
                                  'families': _family_counts(col_names[c] for c in cols)})
            continue
        ridx = {r: k for k, r in enumerate(rows)}
        M = np.zeros((len(rows), len(cols)))
        for j, c in enumerate(cols):
            for r, a in jnz[c].items():
                M[ridx[r], j] = a
        # V must be the full C x C basis; U is needed only as far as rows < columns forces it.
        _u, s, vt = np.linalg.svd(M, full_matrices=len(rows) < len(cols), compute_uv=True)
        tol = max(M.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        rank = int(np.sum(s > tol))
        nullity = len(cols) - rank
        if nullity == 0:
            continue
        for v in vt[rank:]:
            support = [j for j in range(len(cols)) if abs(v[j]) > NULL_WEIGHT_TOL]

            def admissible(sign):
                for j in support:
                    step = sign * v[j]
                    c = cols[j]
                    if step > 0 and bnd[c]['ub'] is not None:
                        return False
                    if step < 0 and bnd[c]['lb'] is not None:
                        return False
                return True
            unbounded = admissible(+1) or admissible(-1)
            h1.append({'classification': 'unbounded' if unbounded else 'bounded_by_variable_bounds',
                       'component_n_columns': len(cols), 'component_n_rows': len(rows),
                       'component_nullity': nullity,
                       'rows': [row_names[r] for r in rows][:20],
                       'columns': [col_entry(cols[j], null_weight=float(v[j])) for j in support]})
    # ---- H2: single-column one-sided relaxations ----
    h2 = []
    for c in screened:
        for sign in (+1, -1):
            if (sign > 0 and bnd[c]['ub'] is not None) or (sign < 0 and bnd[c]['lb'] is not None):
                continue
            ok = True
            for r, a in jnz[c].items():
                rk = nl['row_bounds'][r][0]
                delta = sign * a
                if rk == 1 and delta <= 0:        # body <= ub, relaxed by decreasing body
                    continue
                if rk == 2 and delta >= 0:        # body >= lb, relaxed by increasing body
                    continue
                if rk == 3:
                    continue
                ok = False
                break
            if ok:
                h2.append(col_entry(c, direction=sign, rows=[row_names[r] for r in jnz[c]]))
    return {
        'flags': {'costed': costed, 'in_con': in_con, 'nl_bound_type': kind},
        'hazard': {'H0_isolated_costless': h0,
                   'H1_null_directions': h1,
                   'H1_unbounded_count': sum(1 for e in h1 if e['classification'] == 'unbounded'),
                   'H1_bounded_count': sum(1 for e in h1 if e['classification'] != 'unbounded'),
                   'H1_components_not_evaluated_over_cap': not_evaluated,
                   'H2_one_sided_rays': h2},
        'screen_scope': {'n_columns': n, 'n_costless': sum(1 for c in range(n) if not costed[c]),
                         'n_screened_costless_movable_linear_only': len(screened),
                         'n_peeled_forced_zero': peeled, 'n_after_peeling': len(alive),
                         'n_components': len(comps), 'largest_component_columns': max_comp,
                         'costless_in_nonlinear_rows_by_family_NOT_SCREENED': _family_counts(costless_nonlinear),
                         'costless_immobile_bound_type4_by_family': _family_counts(costless_immobile)},
        'objective_only_columns': objective_only,
        'n_columns_with_zero_linear_coefficient_only_rows': sum(
            1 for c in range(n) if not jnz[c] and jzero[c] and not nl_rows[c]),
    }


def positive_control(work_dir):
    """The hazard screens must FIRE on known cases before their silence on production blocks means anything.
    A synthetic model, written to .nl through the same writer and screened by the same `column_structure`. Never
    solved (the guard is armed). Expected: H1 'unbounded' on the row-18 replica {d_up, d_dn}; H1
    'bounded_by_variable_bounds' on the bounded replica {e_up, e_dn}; H2 on y; nothing on x or w."""
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(0, 10))
    m.w = pe.Var(bounds=(0, 10))
    m.d_up = pe.Var(domain=pe.NonNegativeReals)
    m.d_dn = pe.Var(domain=pe.NonNegativeReals)
    m.e_up = pe.Var(bounds=(0, 5))
    m.e_dn = pe.Var(bounds=(0, 5))
    m.y = pe.Var(domain=pe.NonNegativeReals)
    m.row18_replica = pe.Constraint(expr=m.x - 1 == m.d_up - m.d_dn)
    m.bounded_replica = pe.Constraint(expr=m.w - 2 == m.e_up - m.e_dn)
    m.one_sided = pe.Constraint(expr=m.x + m.y >= 1)
    m.nonlinear = pe.Constraint(expr=m.x ** 2 + m.w ** 2 <= 50)
    m.obj = pe.Objective(expr=(m.x - 3) ** 2 + m.w)
    path = os.path.join(work_dir, 'positive_control.nl')
    m.write(path, format='nl', io_options={'symbolic_solver_labels': True})
    nl = parse_nl(path)
    with open(path[:-3] + '.col') as handle:
        cols = [ln.rstrip('\n') for ln in handle]
    with open(path[:-3] + '.row') as handle:
        rows = [ln.rstrip('\n') for ln in handle]
    st = column_structure(nl, cols, rows)['hazard']
    got_unb = [sorted(c['name'] for c in e['columns']) for e in st['H1_null_directions']
               if e['classification'] == 'unbounded']
    got_bnd = [sorted(c['name'] for c in e['columns']) for e in st['H1_null_directions']
               if e['classification'] != 'unbounded']
    got_h2 = sorted(c['name'] for c in st['H2_one_sided_rays'])
    result = {'H1_unbounded': got_unb, 'H1_bounded': got_bnd, 'H2': got_h2,
              'H0': [c['name'] for c in st['H0_isolated_costless']]}
    result['pass'] = (got_unb == [['d_dn', 'd_up']] and got_bnd == [['e_dn', 'e_up']] and got_h2 == ['y']
                      and result['H0'] == [])
    return result


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


def _scenario_class(family, index, block, is_network):
    if not is_network:
        return 'no_scenario_index'
    if family in NO_SCENARIO_FAMILIES or not isinstance(index, tuple):
        return 'no_scenario_index' if family in NO_SCENARIO_FAMILIES else 'unclassified'
    if family in ROW18_FAMILIES:
        pos = (0, 1)
    elif len(index) >= 3:
        pos = (1, 2)
    else:
        return 'unclassified'
    s_m0, s_o0 = mch.sess_na_scenario(block)
    return 'first_pair' if (index[pos[0]] == s_m0 and index[pos[1]] == s_o0) else 'other_pairs'


def named_subsets(agent, block, net):
    """(name, families, predicate(index, scen_class)) for the retired / unwired families named in W68."""
    out = []
    sess = tuple(mch._SHARED_ESS_OPERATIONAL_VARIABLES)
    if agent in ('DSO', 'TSO'):
        out.append(('shared_ess_operational_copies_first_pair', sess, lambda i, sc: sc == 'first_pair'))
        out.append(('shared_ess_operational_copies_other_pairs', sess, lambda i, sc: sc == 'other_pairs'))
        out.append(('flex_q_day_balance_slacks', ('slack_flex_q_balance_up', 'slack_flex_q_balance_down'),
                    lambda i, sc: True))
    if agent == 'DSO':
        ref_idx = net.get_node_idx(net.get_reference_node_id())
        out.append(('dso_reference_node_slack_v_sqr', ('slack_v_sqr_up', 'slack_v_sqr_down'),
                    lambda i, sc: i[0] == ref_idx))
        out.append(('dso_other_nodes_slack_v_sqr', ('slack_v_sqr_up', 'slack_v_sqr_down'),
                    lambda i, sc: i[0] != ref_idx))
        out.append(('row18_deviation_pairs', ROW18_FAMILIES, lambda i, sc: True))
        out.append(('flex_p_day_balance_slacks', ('slack_flex_p_balance_up', 'slack_flex_p_balance_down'),
                    lambda i, sc: True))
    if agent == 'TSO':
        adn_nodes = set(net.get_node_idx(n) for n in net.active_distribution_network_nodes)
        adn_loads = set(net.get_adn_load_idx(n) for n in net.active_distribution_network_nodes)
        out.append(('tso_interface_delta_first_pair', ('interface_delta_p', 'interface_delta_q'),
                    lambda i, sc: sc == 'first_pair'))
        out.append(('tso_interface_delta_other_pairs', ('interface_delta_p', 'interface_delta_q'),
                    lambda i, sc: sc == 'other_pairs'))
        out.append(('tso_adn_flex_legs', ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down'),
                    lambda i, sc: i[0] in adn_loads))
        out.append(('tso_non_adn_flex_legs', ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down'),
                    lambda i, sc: i[0] not in adn_loads))
        out.append(('tso_adn_load_curtailment', ('pc_curt_down', 'pc_curt_up', 'qc_curt_down', 'qc_curt_up'),
                    lambda i, sc: i[0] in adn_loads))
        out.append(('tso_adn_load_pc_qc', ('pc', 'qc'), lambda i, sc: i[0] in adn_loads))
        out.append(('tso_adn_node_slack_v_sqr', ('slack_v_sqr_up', 'slack_v_sqr_down'),
                    lambda i, sc: i[0] in adn_nodes))
        out.append(('flex_p_day_balance_slacks_at_adn_loads', ('slack_flex_p_balance_up', 'slack_flex_p_balance_down'),
                    lambda i, sc: i[0] in adn_loads))
        out.append(('flex_p_day_balance_slacks_elsewhere', ('slack_flex_p_balance_up', 'slack_flex_p_balance_down'),
                    lambda i, sc: i[0] not in adn_loads))
    return out


def _empty_counter():
    d = {k: 0 for k in CATEGORIES}
    d.update({'written_costed': 0, 'written_costless': 0, 'written_in_constraints': 0,
              'written_nl_bound_types': {}})
    return d


def _bump(counter, cat, col, flags):
    counter[cat] += 1
    if cat == 'written':
        counter['written_costed' if flags['costed'][col] else 'written_costless'] += 1
        counter['written_in_constraints'] += int(flags['in_con'][col])
        bt = str(flags['nl_bound_type'][col])
        counter['written_nl_bound_types'][bt] = counter['written_nl_bound_types'].get(bt, 0) + 1


def analyse_block(block, stem, nl_dir, keep_files, agent, net):
    rec = {'nl': {}}
    for flag, tag in ((False, 'labels_off'), (True, 'labels_on')):
        path = os.path.join(nl_dir, f'{stem}_{tag}.nl')
        if os.path.exists(path):
            raise RuntimeError(f'REFUSED: exists: {path}')
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
    if len(col_names) != nl['n_vars']:
        raise RuntimeError(f'.col lines {len(col_names)} != .nl columns {nl["n_vars"]}')

    # ---- model-side census ----
    all_vars = list(block.component_data_objects(pe.Var, descend_into=True))
    by_name = {v.getname(fully_qualified=True): v for v in all_vars}
    if len(by_name) != len(all_vars):
        raise RuntimeError('non-unique Var names in block')
    col_of = {}
    for c, nm in enumerate(col_names):
        v = by_name.get(nm)
        if v is None:
            raise RuntimeError(f'.col name {nm!r} not resolvable to a model Var')
        col_of[id(v)] = c
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

    struct = column_structure(nl, col_names, row_names)
    flags = struct.pop('flags')
    is_network = agent in ('DSO', 'TSO')
    subsets = named_subsets(agent, block, net)
    sub_counts = {name: _empty_counter() for name, _f, _p in subsets}
    fam = {}
    fixed_written = []
    cats = {k: [] for k in CATEGORIES}
    for v in all_vars:
        vid = id(v)
        nm = v.getname(fully_qualified=True)
        family = v.parent_component().local_name
        if vid in col_of:
            if v.fixed:
                fixed_written.append(nm)
            cat = 'written'
        elif v.fixed:
            cat = ('fixed_referenced_by_active' if vid in ref_active else
                   'fixed_referenced_only_by_inactive' if vid in ref_inactive else 'fixed_unreferenced')
        elif vid in ref_active:
            cat = 'free_unwritten_referenced_by_active'
        elif vid in ref_inactive:
            cat = 'free_unwritten_referenced_only_by_inactive'
        else:
            cat = 'free_unwritten_unreferenced'
        cats[cat].append(nm)
        sc = _scenario_class(family, v.index(), block, is_network)
        f = fam.setdefault(family, {})
        _bump(f.setdefault(sc, _empty_counter()), cat, col_of.get(vid), flags)
        for name, families, pred in subsets:
            if family in families and pred(v.index(), sc):
                _bump(sub_counts[name], cat, col_of.get(vid), flags)
    n_model = len(all_vars)
    n_written = len(col_of)
    cat_counts = {k: len(v) for k, v in cats.items()}
    rec.update({
        'n_model_vars': n_model, 'n_nl_columns': nl['n_vars'], 'n_nl_rows': nl['n_cons'], 'n_nl_objs': nl['n_objs'],
        'n_fixed': sum(1 for v in all_vars if v.fixed),
        'category_counts': cat_counts,
        'identity_holds': (n_model == sum(cat_counts.values()) and cat_counts['written'] == n_written == nl['n_vars']
                           and not fixed_written),
        'fixed_vars_written': fixed_written,
        'unwritten_free_names_first_50': {k: sorted(cats[k])[:50] for k in cats if k.startswith('free_unwritten')},
        'families': dict(sorted(fam.items())),
        'named_subsets': {k: v for k, v in sub_counts.items() if sum(v[c] for c in CATEGORIES) > 0},
        'named_subsets_empty': sorted(k for k, v in sub_counts.items() if sum(v[c] for c in CATEGORIES) == 0),
        'model_constraints_active': n_con_active, 'model_constraints_inactive': n_con_inactive,
        'model_objectives_active': n_obj_active, 'model_objectives_inactive': n_obj_inactive,
        'header_counts_identical_between_label_settings': header_same,
        'retired_components_present': sorted(
            {c for c in RETIRED_COMPONENT_NAMES if hasattr(block, c)}
            | {comp.local_name for comp in block.component_objects(descend_into=True)
               if any(frag in comp.local_name for frag in RETIRED_NAME_FRAGMENTS)}),
        'row18_wired': hasattr(block, 'row18_alpha'),
        'structure': struct,
    })
    if hasattr(block, 'row18_alpha'):
        rec['row18'] = {'alpha': float(pe.value(block.row18_alpha)),
                        'rows_active': sum(1 for fr in srp._ROW18_DEVIATION_FAMILIES
                                           for i in getattr(block, fr[0]) if getattr(block, fr[0])[i].active),
                        'dev_vars_fixed': sum(1 for fr in srp._ROW18_DEVIATION_FAMILIES for nm2 in fr[1:3]
                                              for i in getattr(block, nm2) if getattr(block, nm2)[i].fixed),
                        'premium_min': min(float(pe.value(block.row18_premium[p])) for p in block.periods)}
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

    def _take(self, key, block, agent, net):
        full = f'{self.phase}|{key}'
        if full in self.records:
            raise RuntimeError(f'duplicate capture {full}')
        stem = full.replace('|', '__').replace(':', '_')
        r = self.records[full] = analyse_block(block, stem, self.nl_dir, self.keep_files, agent, net)
        hz = r['structure']['hazard']
        unwritten = ' + '.join(f'{k} {r["category_counts"][k]}' for k in UNWRITTEN_CATEGORIES)
        _log(f'  {full}: model vars {r["n_model_vars"]} = columns {r["n_nl_columns"]} + {unwritten}  '
             f'(identity {r["identity_holds"]}); rows {r["n_nl_rows"]}; H0 {len(hz["H0_isolated_costless"])}, '
             f'H1 unbounded {hz["H1_unbounded_count"]} bounded {hz["H1_bounded_count"]}, '
             f'H2 {len(hz["H2_one_sided_rays"])}, over-cap {len(hz["H1_components_not_evaluated_over_cap"])}')

    def network_stub(self, agent, node_id, network):
        def stub(model, *args, **kwargs):
            k = f'{self.phase}|{agent}:{node_id}'
            self.calls[k] = self.calls.get(k, 0) + 1
            for y in network.years:
                for d in network.days:
                    self._take(f'{agent}:{node_id}:{y}:{d}', model[y][d], agent, network.network[y][d])
            return {y: {d: None for d in network.days} for y in network.years}
        return stub

    def esso_stub(self, sed):
        def stub(models, *args, **kwargs):
            k = f'{self.phase}|ESSO'
            self.calls[k] = self.calls.get(k, 0) + 1
            for node_id in sed.active_distribution_network_nodes:
                self._take(f'ESSO:{node_id}', models[node_id], 'ESSO', None)
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


def aggregate(records, phase, agent=None):
    recs = {k: r for k, r in records.items() if k.startswith(phase + '|')
            and (agent is None or k.split('|')[1].startswith(agent + ':'))}
    tot = {x: sum(r[x] for r in recs.values()) for x in ('n_model_vars', 'n_nl_columns', 'n_fixed', 'n_nl_rows')}
    cats = {c: sum(r['category_counts'][c] for r in recs.values()) for c in CATEGORIES}
    fams, subs = {}, {}

    def merge(dst, src):
        for c, v in src.items():
            if c == 'written_nl_bound_types':
                t = dst.setdefault(c, {})
                for bt, n in v.items():
                    t[bt] = t.get(bt, 0) + n
            else:
                dst[c] = dst.get(c, 0) + v
    for r in recs.values():
        for f, by_sc in r['families'].items():
            for sc, cnt in by_sc.items():
                merge(fams.setdefault(f, {}).setdefault(sc, {}), cnt)
        for s, cnt in r['named_subsets'].items():
            merge(subs.setdefault(s, {}), cnt)
    hz = [r['structure']['hazard'] for r in recs.values()]
    return {'n_blocks': len(recs), **tot, 'category_counts': cats,
            'identity_holds_every_block': all(r['identity_holds'] for r in recs.values()),
            'accounting': f'{tot["n_model_vars"]} = {cats["written"]} written + '
                          + ' + '.join(f'{cats[k]} {k}' for k in UNWRITTEN_CATEGORIES),
            'H0_total': sum(len(h['H0_isolated_costless']) for h in hz),
            'H1_unbounded_total': sum(h['H1_unbounded_count'] for h in hz),
            'H1_bounded_total': sum(h['H1_bounded_count'] for h in hz),
            'H2_total': sum(len(h['H2_one_sided_rays']) for h in hz),
            'H1_components_over_cap_total': sum(len(h['H1_components_not_evaluated_over_cap']) for h in hz),
            'fixed_vars_written_total': sum(len(r['fixed_vars_written']) for r in recs.values()),
            'retired_components_present': sorted({c for r in recs.values() for c in r['retired_components_present']}),
            'families': dict(sorted(fams.items())),
            'families_not_fully_written': sorted(
                f for f, by_sc in fams.items()
                if any(cnt.get(c, 0) for cnt in by_sc.values() for c in UNWRITTEN_CATEGORIES)),
            'named_subsets': dict(sorted(subs.items()))}


def run(args, state):
    inst = INSTANCES[args.instance]
    mem = state['memory']
    checklist = {
        'case_sha256_matches': _sha256_file(os.path.join(REPO, inst['case_path'])) == inst['case_sha256'],
        'params_file_sha256_matches': _sha256_file(os.path.join(REPO, PARAMS_FILE['path'])) == PARAMS_FILE['sha256'],
        'row_spec_sha256_matches': _sha256_file(os.path.join(REPO, ROW_SPEC['path'])) == ROW_SPEC['sha256'],
        'writer_forces_presolve_off': 'config.linear_presolve = False' in open(
            os.path.join(os.path.dirname(pyomo.__file__), 'repn', 'plugins', 'nl_writer.py')).read(),
        'production_builders_present': all(hasattr(srp, f) for f in (
            'create_distribution_networks_models', 'create_transmission_network_model',
            'create_shared_energy_storage_model', '_prepare_distribution_objectives_for_admm',
            'update_distribution_models_to_admm', 'update_distribution_coordination_models_and_solve',
            '_ROW18_DEVIATION_FAMILIES')),
        'capture_paths_present': all(callable(f) for f in (analyse_block, column_structure, named_subsets,
                                                           aggregate)),
    }
    if state['out_under_repo']:
        checklist['script_committed_unmodified'] = state['script_committed_unmodified']
    if args.instance == '2x2':
        spec = json.load(open(os.path.join(REPO, ROW_SPEC['path'])))
        di = spec['configuration']['derived_instance']
        checklist['row_spec_pins_this_case'] = (di['case_path'] == inst['case_path']
                                                and di['case_sha256'] == inst['case_sha256']
                                                and di['scenario_checksum'] == inst['scenario_checksum'])
        checklist['alpha_in_row_spec'] = any(float(c['interface_deviation_premium']['alpha']) == args.alpha
                                             and c['key'] == X0_KEY for c in spec['candidates'])
    state['checklist'] = checklist
    if not all(checklist.values()):
        raise RuntimeError(f'REFUSED: checklist failed: {checklist}')
    _log(f'instance {args.instance} alpha {args.alpha} label {args.label}; checklist {checklist}')
    memory_gate('before read', mem)
    work = state['work']
    os.makedirs(os.path.join(work, 'nl'))
    control = state['positive_control'] = positive_control(work)
    _log(f'positive control (hazard screens must fire): {control}')
    if not control['pass']:
        raise RuntimeError(f'REFUSED: hazard-screen positive control failed: {control}')

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
    state['scenario_checksum_observed'] = checksum
    if checksum != inst['scenario_checksum']:
        raise RuntimeError(f'REFUSED: scenario checksum {checksum} != {inst["scenario_checksum"]}')
    redirect_dirs(planning, os.path.join(work, 'run'))
    planning.parallel_execution = False
    a = planning.params.admm
    a.interface_deviation_premium = {'alpha': args.alpha, 'floor': None, 'source': 'W68 probe (--alpha)'}
    sed = planning.shared_ess_data

    # ---- the x = 0 candidate, exactly as _construct_arm_planning builds it ----
    canonical = H.canonical_candidate({n: (0.0, 0.0) for n in sed.active_distribution_network_nodes})
    key = H.candidate_key(canonical)
    if key != X0_KEY:
        raise RuntimeError(f'REFUSED: x = 0 candidate key {key} != {X0_KEY}')
    candidate = planning.get_initial_candidate_solution()
    for node_id, (s_val, e_val) in H.investment_map_from_canonical(canonical).items():
        candidate['investment'][node_id][canonical['investment_year']]['s'] = s_val
        candidate['investment'][node_id][canonical['investment_year']]['e'] = e_val
    srp._rebuild_candidate_total_capacities(planning, candidate)
    state['candidate'] = {'canonical': canonical, 'key': key, 'total_capacity_all_zero': all(
        v2 == 0.0 for n in candidate['total_capacity'].values() for yv in n.values() for v2 in yv.values())}
    memory_gate('after read', mem)

    cap = state['capture'] = Capture(os.path.join(work, 'nl'), args.keep_nl)
    tn, dns = planning.transmission_network, planning.distribution_networks
    for node_id, dn in dns.items():
        dn.optimize = cap.network_stub('DSO', node_id, dn)
    tn.optimize = cap.network_stub('TSO', tn.name, tn)
    sed.optimize = cap.esso_stub(sed)
    deviations = state['deviations']
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
                        # `_activate_row18_with_settlement` reads pe.value(flow) - pe.value(expectation).
                        # `pg_adn`/`qg_adn` are Expressions (W68 attempt r1 failed assuming Vars): only a
                        # None-valued Var INSIDE them is set to 0.0, and only if the Expression cannot be
                        # evaluated. Value-only; recorded with counts per Var family.
                        for fam_row in srp._ROW18_DEVIATION_FAMILIES:
                            for comp_name in (fam_row[3], fam_row[4]):
                                comp = getattr(blk, comp_name)
                                for idx in comp:
                                    item = comp[idx]
                                    if isinstance(item, pe.Var) or getattr(item, 'is_variable_type', lambda: False)():
                                        if item.value is None:
                                            item.set_value(0.0)
                                            none_valued[comp_name] = none_valued.get(comp_name, 0) + 1
                                    elif pe.value(item, exception=False) is None:
                                        for v in identify_variables(item.expr, include_fixed=True):
                                            if v.value is None:
                                                v.set_value(0.0)
                                                fam_v = v.parent_component().local_name
                                                none_valued[fam_v] = none_valued.get(fam_v, 0) + 1
        if none_valued:
            deviations.append({'value_only': 'Vars read by _activate_row18_with_settlement (directly or inside the '
                                             'pg_adn/qg_adn Expressions) had no value (no init solve); set to 0.0',
                               'counts': none_valued})
        srp._prepare_distribution_objectives_for_admm(dns, dso_models)
        srp._prepare_transmission_objectives_for_admm(tn, tso_model)
        if a.objective_scale is None:
            raise RuntimeError('REFUSED: admm.objective_scale not fixed in the case file')
        objective_scale = state['objective_scale_used'] = a.objective_scale
        al_scale_esso = state['al_scale_esso'] = srp._resolve_esso_al_scale(planning, a, objective_scale)[0]
        srp.update_distribution_models_to_admm(planning, dso_models, a, objective_scale)
        srp.update_transmission_model_to_admm(planning, tso_model, a, objective_scale)
        srp.update_shared_energy_storage_model_to_admm(planning, esso_model, a, al_scale_esso=al_scale_esso)
        srp._initialize_shared_ess_consensus(planning, consensus_vars)
        if getattr(a, 'shared_ess_initialization', 'standalone') != 'standalone':
            raise RuntimeError('REFUSED: shared_ess_initialization is not standalone')
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

    state['summary'] = {ph: {'all': aggregate(cap.records, ph),
                             **{ag: aggregate(cap.records, ph, ag) for ag in ('DSO', 'TSO', 'ESSO')}}
                        for ph in ('init', 'cycle1')}


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
        GUARD.uninstall()
        _log(f'guard verify(0) at refusal: {GUARD.verify(expected_solves=0)}')
        raise SystemExit('--scratch must be outside the repository')
    out_root = os.path.abspath(args.out_root)
    stage_root = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53') + os.sep
    out_under_repo = out_root.startswith(REPO + os.sep)
    if out_under_repo and not out_root.startswith(stage_root):
        GUARD.uninstall()
        _log(f'guard verify(0) at refusal: {GUARD.verify(expected_solves=0)}')
        raise SystemExit('--out-root must be under data/SRP1/Results/P515S53/ or outside the repository')
    out_dir = os.path.join(out_root, args.label)
    work = os.path.join(scratch, args.label)
    for p in (out_dir, work):
        if os.path.exists(p):
            GUARD.uninstall()
            _log(f'guard verify(0) at refusal: {GUARD.verify(expected_solves=0)}')
            raise SystemExit(f'REFUSED (write-once): exists: {p}')
    os.makedirs(out_dir)
    os.makedirs(work)
    script = os.path.abspath(__file__)
    rel = os.path.relpath(script, REPO)
    committed = (subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=REPO, capture_output=True).returncode
                 == 0 and subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0)
    state = {'memory': [], 'deviations': [], 'work': work, 'out_under_repo': out_under_repo,
             'script_committed_unmodified': committed}
    outcome, error = 'completed', None
    try:
        run(args, state)
    except MemoryStop as exc:
        outcome, error = 'memory_stop', str(exc)
    except BaseException as exc:  # noqa: BLE001 -- recorded, then the guard is still verified below
        outcome, error = 'exception', ''.join(traceback.format_exception(exc))
    finally:
        GUARD.uninstall()
        verify_failures = GUARD.verify(expected_solves=0)
    guard_record = {'permitted': [], 'counts': dict(GUARD.counts), 'expected_solves': 0,
                    'verify_failures': verify_failures, 'verified_on_exit_path': outcome}
    _log(f'guard {dict(GUARD.counts)} verify(0) failures {verify_failures} (exit path: {outcome})')
    common = {
        'schema': 'p515_s53_nl_varcount_probe_v2', 'stage': STAGE,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable, 'argv': sys.argv,
        'pyomo_version': pyomo.version.version, 'script': rel, 'script_sha256': _sha256_file(script),
        'script_committed_unmodified': committed,
        'production_sha256': {f: _sha256_file(os.path.join(REPO, f)) for f in (
            'shared_resources_planning.py', 'model_construction_helpers.py', 'network.py', 'network_data.py',
            'shared_energy_storage_data.py', 'admm_parameters.py', 'helper_functions.py')},
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'instance': {'name': args.instance, **INSTANCES[args.instance],
                     'scenario_checksum_observed': state.get('scenario_checksum_observed'),
                     'row_spec': ROW_SPEC if args.instance == '2x2' else None, 'params_file': PARAMS_FILE},
        'alpha': args.alpha, 'candidate': state.get('candidate'),
        'capture_path_checklist': state.get('checklist'), 'memory_threshold_gib': MIN_AVAILABLE_GIB,
        'memory': state['memory'], 'predictions_recorded_before_run': PREDICTIONS,
        'hazard_screen_positive_control': state.get('positive_control'),
        'solve_profile_guard': guard_record, 'outcome': outcome, 'error': error,
    }
    with open(os.path.join(out_dir, 'run_outcome.json'), 'w') as handle:
        json.dump(common, handle, indent=1, default=str)
    if outcome != 'completed':
        _log(f'NOT COMPLETED ({outcome}): {error}')
        return 2
    cap = state['capture']
    payload = {**common,
               'objective_scale_used': state.get('objective_scale_used'), 'al_scale_esso': state.get('al_scale_esso'),
               'deviations_from_run_operational_planning_value_only': state['deviations'],
               'nullspace_component_cap': NULLSPACE_COMPONENT_CAP, 'null_weight_tol': NULL_WEIGHT_TOL,
               'stub_calls': cap.calls, 'summary': state['summary'], 'blocks': cap.records,
               'nl_files_kept': bool(args.keep_nl), 'scratch': work}
    out_path = os.path.join(out_dir, 'nl_varcount.json')
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    for ph in ('init', 'cycle1'):
        for ag in ('all', 'DSO', 'TSO', 'ESSO'):
            s = state['summary'][ph][ag]
            _log(f'{ph} {ag}: blocks {s["n_blocks"]}; {s["accounting"]}; columns {s["n_nl_columns"]}; identity '
                 f'{s["identity_holds_every_block"]}; H0 {s["H0_total"]} H1u {s["H1_unbounded_total"]} H1b '
                 f'{s["H1_bounded_total"]} H2 {s["H2_total"]} over-cap {s["H1_components_over_cap_total"]}; fixed '
                 f'written {s["fixed_vars_written_total"]}; retired present {s["retired_components_present"]}')
    _log(f'wrote {out_path}')
    return 0 if not verify_failures else 1


if __name__ == '__main__':
    sys.exit(main())
