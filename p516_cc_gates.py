"""P5.16 convention change (spec v4) -- zero-solve gates G-CC-1, G-CC-2, G-CC-3, G-CC-5 for C1 + C3 + C4.

Authority: P5_16_CONVENTION_CHANGE_SPEC_v4.md section 2; PLANNER_BRIEF_2026-09-13.md Addendum 88 (R1.1-R1.5) and its
Amendment (C2 closed). Zero solves of any kind: importing `p515_s53_w183_nl_digest` (W183) installs its
`SolveProfileGuard(permitted=())` before any production import; it is verified at exactly 0 at the end of every gate.
G-CC-3 is AD / FD of `pe.value` on built models.

Gates (one per invocation, `--gate`):
  cc1  switch-off identity on SRP1 (unchanged files): W183's own `build_config` (sequential production path with every
       optimize stubbed + the parallel DSO worker) for x0_A_1x and unit_A_1x, compared with the HEAD reference built by
       the committed W183 harness BEFORE the production edit (`--reference`, default g_cc1/ref_head) through W183's
       `compare_blocks` (NL plain + labelled + .row + .col sha256, and per-component digests). Plus: every preserved
       fixture of W183c's gate-1 list unpickles (child process running W183c's own `gate1_fixtures`), and the D0
       T0.1a dry run (`p516_d0_dry_run.py`, rehearsal dir under /tmp; the script is executed, not edited) reproduces
       the committed LP bytes of data/SRP1/Results/P516D0/dry_run (manifest 04033ff7...).
  cc2  the new convention at x = 0 and at the unit on SRP1 with the switches ON in memory
       (salvage_value.in_objective = True, available_energy_soh_point = 'mid'; SRP1 files unchanged), same builders,
       compared with the cc1 run: network blocks bitwise; ESSO differences enumerated (components; NL rows by name via
       W183's `nl_row_analysis`); value-inertness of the salvage term from its standard repn at the would-be solve.
  cc5  the five-block single-scenario case file (data/SRP1/SRP1_5x3y_1x1.json -> SharedESS/SRP1_5x3y_ESS_Params.json)
       loads; tcal_norm, block weights, terminal discount, remaining-life fractions, prob_market_scenarios; 80 network
       + 3 ESSO blocks built at x = 0 and at the 2028 unit (node 7, 0.25 MVA / 1.0 MWh) through
       `run_operational_planning` with every optimize stubbed; x = 0 NL digests = the new instance's G0.
  cc3  salvage calculus on the five-block instance: full zero-solve ADMM build of the 2028 unit (networks + ESSO,
       production `create_*` / `_prepare_*` / `update_*_to_admm`), kappa hand replica, build order (kappa = 0 at the
       initialisation stub), AD vs FD of d(objective)/d(throughput) per cohort (2025, 2028, ..., 2037 via an
       ESSO-only probe with every cohort active), euro parity against DSO and TSO blocks, slack safety, and the
       complementarity diagnostic's effective marginal.

Output: data/SRP1/Results/P516CC/g_cc<N>/<--run-name> (write-once; refuses if it exists), with predictions.json written
first, results JSON through gate_result_io, manifest_sha256.json last. One copy at a time (lock file). Launch attached
with both streams captured, e.g.
  NLP_SOLVER_PATH=/usr/local/bin/ipopt /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_cc_gates.py --gate cc1 \\
      > data/SRP1/Results/P516CC/g_cc1_post_off_launch.log 2>&1
"""

import argparse
import gc
import gzip
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s53_w183_nl_digest as W  # noqa: E402 -- installs SolveProfileGuard(permitted=()) before production imports

GUARD = W.GUARD
import gate_result_io as GRIO  # noqa: E402

SCRIPT_REL = os.path.basename(__file__)
ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P516CC')
ROOT = os.path.join(REPO, ROOT_REL)
# Rehearsal only (precedent: p516_d0_dry_run.py): P516CC_REHEARSAL_ROOT=/tmp/... redirects every output; anything else
# is refused. Evidence runs set nothing.
_REHEARSAL = os.environ.get('P516CC_REHEARSAL_ROOT')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516CC_REHEARSAL_ROOT must be under /tmp, got {_REHEARSAL!r}')
    ROOT = os.path.abspath(_REHEARSAL)
LOCK_PATH = os.path.join(REPO, '.p516_cc_gates.lock')
NEW_CASE_REL = os.path.join('data', 'SRP1', 'SRP1_5x3y_1x1.json')
NEW_ESS_PARAMS_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_5x3y_ESS_Params.json')
UNCHANGED_FILES_REL = [os.path.join('data', 'SRP1', 'SRP1.json'),
                       os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
                       os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
                       'model_construction_helpers.py']
SRP1_CONFIGS = ['x0_A_1x', 'unit_A_1x']
D0_DRY_RUN_REL = os.path.join('data', 'SRP1', 'Results', 'P516D0', 'dry_run')
D0_DRY_RUN_MANIFEST_SHA = '04033ff7429c8e515f899a872a5d807381fad340276cf19299a2dadab69f6e67'
W183_COMMITTED_OFF_REF = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183_nl_post_w183c_off',
                                      'w183_nl_digest.json.gz')
MEMORY_STOP_BYTES = 16 * (1 << 30)
UNIT_2028 = {'node': 7, 'year': 2028, 's': 0.25, 'e': 1.0}
PROBE_YEARS = (2025, 2028, 2031, 2034, 2037)
FD_REL_STEP = 1e-6
REL_TOL = 1e-6
SLACK_SAFETY_FACTOR = 1e-3

PREDICTIONS = {
    'cc1': {
        'G-CC-1': 'every switch off => ESSO + all network blocks of x0_A_1x and unit_A_1x bitwise identical to the HEAD '
                  'reference (NL plain/labelled/.row/.col and component digests, both builders); all fixtures of '
                  "W183c's gate-1 list unpickle; the T0.1a dry run reproduces its committed LP bytes",
    },
    'cc2': {
        'network_blocks': 'bitwise identical to cc1 (expert, Addendum 88 predictions: x = 0 network blocks bitwise)',
        'esso_x0_value_inert': 'the salvage term is identically 0 at x = 0 (S == 0): objective identical as a function',
        'esso_x0_nl_bitwise': 'PREDICTION (not assumed; NL-writer folding of constant terms unverified): the x = 0 '
                              'ESSO plain NL files are bitwise identical to cc1',
        'unit_cell': 'ESSO differences restricted to C1 (objective) and C3 (available_e_capacity_unit rows); on SRP1 '
                     'the 2025 cohort has rlf = 0 and the 2030/2035 cohorts are inactive, so the C1 term is a '
                     'constant 0 and the NL diff is expected in the available rows (and their header/k consequences)',
    },
    'cc5': {
        'loads_and_builds': '80 network + 3 ESSO blocks at x = 0 and at the 2028 unit, zero solves, < 16 GiB',
        'tcal_norm': 5, 'terminal_discount': '1.02**-15', 'rlf': [0.0, 0.2, 0.4, 0.6, 0.8],
        'weights': '3 * days * 1.02**-(y - 2025)',
        'sigma_band': ('recorded, not testable here (needs the initialisation solve): sigma_computed / sigma_fixed '
                       'inside [1/3, 3], point estimate ~ 0.6 (s52 2-scenario reading); sigma_fixed = 93,635,360'),
    },
    'cc3': {
        'd_obj_d_throughput': '= kappa * d(credit)/d(throughput) to <= 1e-6 relative (AD vs FD and vs the closed form)',
        'expert_dS_dTP_eur_per_MWh': {'2025': 0.0, '2028': '~1 within a factor 2', '2037': '~5 within a factor 2'},
        'advisor_order_of_magnitude': {'2028': 2.0, '2037': 8.0},
        'kappa': 'a / sigma = 1 / w_med, reproduced by hand from admm_esso_al_scale and admm_common_objective_scale',
        'euro_parity': '(kappa / a) == 1 / sigma == d(block ADMM objective) / d(block PV cost)',
        'slack_safety': 'holds (expert): salvage marginal per unit es_pnet <= 1e-3 * PENALTY_ESSO_SLACK',
        'kappa_at_initialisation': 0.0,
    },
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [p516-cc] {msg}', flush=True)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _write(out, name, obj):
    path = os.path.join(out, name)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'x') as handle:
        GRIO.dump(W._str_keys(obj), handle, indent=1, sort_keys=True)
    return path


def _rel(a, b):
    if a == b:
        return 0.0
    den = max(abs(a), abs(b))
    return abs(a - b) / den if den else float('inf')


class MemoryWatch:
    """Peak RSS of this process (and children) sampled every 0.5 s; records a breach of the 16 GiB stop limit."""

    def __init__(self):
        import psutil
        self.proc = psutil.Process()
        self.peak = 0
        self.breach = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _rss(self):
        total = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:  # noqa: BLE001
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            rss = self._rss()
            self.peak = max(self.peak, rss)
            if rss > MEMORY_STOP_BYTES and self.breach is None:
                self.breach = {'rss': rss, 'utc': _utc()}
            self._stop.wait(0.5)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())
        return {'peak_rss_bytes': self.peak, 'peak_rss_gib': self.peak / (1 << 30), 'limit_bytes': MEMORY_STOP_BYTES,
                'breach': self.breach}


# ======================================================================================================================
#  provenance + rule eleven
# ======================================================================================================================
def production_status():
    prod = W.production_modules()
    return {'head': _git(['rev-parse', 'HEAD']).strip(),
            'modules_sha256': {f: _sha_file(os.path.join(REPO, f)) for f in prod},
            'git_status': W.git_status_of(prod),
            'diff_stat_vs_head': _git(['diff', '--stat', 'HEAD', '--'] + prod)}


def unchanged_files_check():
    status = W.git_status_of(UNCHANGED_FILES_REL)
    return {'files': UNCHANGED_FILES_REL, 'git_status': status, 'unmodified': not status,
            'sha256': {f: _sha_file(os.path.join(REPO, f)) for f in UNCHANGED_FILES_REL}}


def assert_capture_paths(gate):
    import shared_energy_storage_data as SED
    import shared_energy_storage_parameters as SEP
    import shared_resources_planning as srp
    checks = {
        'w183_build_config': callable(getattr(W, 'build_config', None)),
        'w183_compare_blocks': callable(getattr(W, 'compare_blocks', None)),
        'w183_nl_row_analysis': callable(getattr(W, 'nl_row_analysis', None)),
        'w183_capture_class': hasattr(W, 'Capture'),
        'switch_in_objective_default_false': SEP.SalvageValueParameters().in_objective is False,
        'switch_soh_point_default_none': SEP.SharedEnergyStorageParameters().available_energy_soh_point is None,
        'salvage_credit_builder': callable(getattr(SED, '_build_salvage_credit_objective_expression', None)),
        'kappa_set_in_update_to_admm': 'salvage_objective_kappa' in inspect.getsource(
            srp.update_shared_energy_storage_model_to_admm),
        'update_to_admm_takes_objective_scale': 'objective_scale' in inspect.signature(
            srp.update_shared_energy_storage_model_to_admm).parameters,
        'diagnostic_effective_marginal': 'salvage_objective_kappa' in inspect.getsource(
            SED._get_esso_complementarity_diagnostics),
        'remaining_life_fn': callable(getattr(SED, '_get_remaining_calendar_life', None)),
        'terminal_discount_fn': callable(getattr(SED, '_get_terminal_discount_factor', None)),
        'block_weight_fn': callable(getattr(srp, '_get_admm_block_weight', None)),
        'esso_al_scale_fn': callable(getattr(srp, '_resolve_esso_al_scale', None)),
    }
    if gate in ('cc3', 'cc5'):
        checks['new_case_file_exists'] = os.path.isfile(os.path.join(REPO, NEW_CASE_REL))
        checks['new_ess_params_exists'] = os.path.isfile(os.path.join(REPO, NEW_ESS_PARAMS_REL))
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise AssertionError(f'RULE ELEVEN: capture paths missing: {failed}')
    return checks


# ======================================================================================================================
#  SRP1 builds through W183's own build_config (cc1 / cc2)
# ======================================================================================================================
ESSO_PROBES = {}


def _esso_probe(model):
    """Read-only inspection of one ESSO model at the would-be solve (salvage term presence, kappa, repn)."""
    import pyomo.environ as pe
    from pyomo.repn import generate_standard_repn
    rec = {'has_kappa': hasattr(model, 'salvage_objective_kappa'),
           'has_salvage_credit_obj': hasattr(model, 'salvage_credit_obj'),
           'has_throughput_coefficients': hasattr(model, '_salvage_throughput_coefficients')}
    if rec['has_kappa']:
        rec['kappa_value'] = pe.value(model.salvage_objective_kappa)
        repn = generate_standard_repn(model.salvage_credit_obj.expr, compute_values=True)
        rec['credit_repn'] = {'constant': repn.constant,
                              'linear': {v.name: c for v, c in zip(repn.linear_vars, repn.linear_coefs)},
                              'linear_vars_fixed': {v.name: bool(v.fixed) for v in repn.linear_vars},
                              'nonlinear': repn.nonlinear_expr is not None,
                              'quadratic': bool(repn.quadratic_vars)}
        rec['credit_value'] = pe.value(model.salvage_credit_obj, exception=False)
        free_nonzero = [n for n, c in rec['credit_repn']['linear'].items()
                        if c != 0.0 and not rec['credit_repn']['linear_vars_fixed'][n]]
        rec['credit_free_nonzero_coefficients'] = free_nonzero
        rec['credit_identically_zero'] = (not free_nonzero and repn.constant == 0
                                          and repn.nonlinear_expr is None and not repn.quadratic_vars)
        rec['objective_str_has_kappa'] = 'salvage_objective_kappa' in str(model.objective.expr)
    rec['objective_active'] = model.objective.active
    rec['available_energy_rows_str_sample'] = str(model.available_e_capacity_unit[1].body)[:200]
    return rec


SWITCH_STATE = {'mode': None}
SWITCH_MODES = {'off': None,
                'convention': {'in_objective': True, 'available_energy_soh_point': 'mid'},
                'c1_only': {'in_objective': True, 'available_energy_soh_point': None}}


def install_esso_probe_and_switches():
    """Wraps W183's Capture.esso_stub (read-only probe before the original digests the model) and W183's
    construct_configured: the switches of SWITCH_STATE['mode'] are applied in memory after the configuration hook
    (None = nothing assigned)."""
    orig_stub = W.Capture.esso_stub

    def esso_stub(self, models, *args, **kwargs):
        ESSO_PROBES.setdefault(f'{SWITCH_STATE["mode"]}|{self.config}|{self.builder}', {}).update(
            {str(n): _esso_probe(m) for n, m in models.items()})
        return orig_stub(self, models, *args, **kwargs)

    W.Capture.esso_stub = esso_stub
    orig_construct = W.construct_configured

    def construct_configured(G, config, builder, ctx):
        planning, sed, candidate, readbacks, state = orig_construct(G, config, builder, ctx)
        switches = SWITCH_MODES[SWITCH_STATE['mode']]
        if switches:
            sed.params.salvage_value.in_objective = switches['in_objective']
            if switches['available_energy_soh_point'] is not None:
                sed.available_energy_soh_point = switches['available_energy_soh_point']
        state['p516_switch_mode'] = SWITCH_STATE['mode']
        state['p516_switches_in_force'] = {
            'salvage_value.in_objective': getattr(sed.params.salvage_value, 'in_objective', None),
            'available_energy_soh_point': sed.available_energy_soh_point}
        return planning, sed, candidate, readbacks, state

    W.construct_configured = construct_configured


def build_srp1(out, label, scratch):
    import p515_g_g1_g4_admm_gates as G
    import p515_s40_polish_gap as PG
    import shared_resources_planning as srp
    capture_checks = W.assert_capture_paths(G, srp, PG)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    import p515_s44_campaign_harness as H
    ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
    declared = H.load_ess_ageing_parameters(ess_path)
    pin = {'path': H.ESS_PARAMS_FILE_REL, 'sha256': _sha_file(ess_path)}
    _cc, floor_rows, _fc = PG._build_floor_rows(f'p516cc_{label}_{run_tag}_precheck')
    os.makedirs(scratch)
    ctx = {'out': out, 'label': label, 'run_tag': run_tag, 'scratch': scratch,
           'spec': W.spec_like(declared, pin), 'declared_ageing': declared, 'floor_rows': floor_rows,
           'nrf_option': None}
    records = {}
    for config in SRP1_CONFIGS:
        t0 = time.time()
        records[config] = W.build_config(G, srp, config, ctx)
        seq, par = records[config]['sequential'], records[config]['parallel']
        _log(f'{config}: sequential {seq["n_blocks"]} blocks checks {all(seq["checks"].values())}; parallel '
             f'{par["n_blocks"]} blocks checks {all(par["checks"].values())} ({time.time() - t0:.0f}s)')
    shutil.rmtree(scratch, ignore_errors=True)
    return records, capture_checks, ctx


def _load_ref(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def compare_records(rec_a, rec_b, configs):
    out = {}
    for c in configs:
        out[c] = {b: W.compare_blocks(rec_a[c], rec_b[c], b) for b in ('sequential', 'parallel')}
    return out


def identity_verdict(comp):
    per = {}
    for c, res in comp.items():
        for b, r in res.items():
            all_four = all(all(e['nl_identical'].values()) for e in r['per_block'].values())
            per[f'{c}|{b}'] = {'blocks_compared': r['blocks_compared'], 'blocks_missing': r['blocks_missing'],
                               'all_nl_plain_identical': r['all_nl_plain_identical'],
                               'all_nl_four_files_identical': all_four and not r['blocks_missing'],
                               'all_digests_identical': r['all_digests_identical'],
                               'blocks_nl_plain_differing': r['blocks_nl_plain_differing'],
                               'blocks_digest_differing': r['blocks_digest_differing']}
    return per


def fixtures_child(out):
    """W183c's own gate-1 fixture list and loader, run in a child process (it patches pickle at import)."""
    path = os.path.join(out, 'fixtures_unpickle.json')
    code = ('import sys, json; sys.path.insert(0, %r); import p515_s53_w183c_nrf_gates as C; '
            'import gate_result_io as GRIO; r = C.gate1_fixtures(); r["guard_counts"] = dict(C.GUARD.counts); '
            'r["guard_verify_0"] = C.GUARD.verify(0); '
            'GRIO.dump(r, open(%r, "x"), indent=1, sort_keys=True)' % (REPO, path))
    rc = subprocess.run([sys.executable, '-u', '-c', code], cwd=REPO).returncode
    rec = _load_ref(path) if os.path.isfile(path) else None
    return {'returncode': rc, 'path': os.path.relpath(path, REPO),
            'pass': bool(rec and rec.get('pass') is True and rc == 0),
            'n_declared': rec and rec.get('n_declared'), 'n_loaded': rec and rec.get('n_loaded'),
            'n_failed': rec and rec.get('n_failed'), 'failed': rec and rec.get('failed'),
            'absent': rec and rec.get('absent'),
            'loaded_list': [{'path': f['path'], 'sha256': f['sha256']} for f in (rec or {}).get('loaded') or []],
            'checks': rec and rec.get('checks'), 'guard_verify_0': rec and rec.get('guard_verify_0')}


HEAD_LOADER = r"""
import sys, json, os, pickle
tree, repo, paths_file, out_file = sys.argv[1:5]
sys.path.insert(0, tree)
from p513_solve_profile_guard import SolveProfileGuard
guard = SolveProfileGuard(permitted=(), label='p516cc fixtures under HEAD modules').install()
import model_construction_helpers, shared_resources_planning, shared_energy_storage_data, shared_energy_storage_parameters
mods = {m.__name__: m.__file__ for m in (model_construction_helpers, shared_resources_planning,
        shared_energy_storage_data, shared_energy_storage_parameters)}
res = {}
for rel in json.load(open(paths_file)):
    try:
        with open(os.path.join(repo, rel), 'rb') as h:
            obj = pickle.load(h)
        res[rel] = 'ok'
        del obj
    except Exception as e:
        res[rel] = f'{type(e).__name__}: {e}'
json.dump({'modules': mods, 'results': res, 'guard_counts': dict(guard.counts), 'guard_verify_0': guard.verify(0)},
          open(out_file, 'x'), indent=1, sort_keys=True)
"""


def fixtures_under_head(out, fixtures, run_tag):
    """The same fixture paths unpickled with the HEAD production modules (exported read-only by `git archive HEAD`
    to /tmp; first on sys.path, asserted by module __file__), to separate pre-existing failures from new ones."""
    tree = f'/tmp/p516cc_head_tree_{run_tag}'
    os.makedirs(tree)
    tar = tree + '.tar'
    subprocess.run(['git', 'archive', '-o', tar, 'HEAD', '--', '*.py'], cwd=REPO, check=True)
    subprocess.run(['tar', '-x', '-f', tar, '-C', tree], check=True)
    paths = sorted([f['path'] for f in fixtures['loaded_list']] + [f['path'] for f in fixtures['failed']])
    paths_file = os.path.join(out, 'fixtures_head_paths.json')
    with open(paths_file, 'x') as handle:
        json.dump(paths, handle, indent=1)
    out_file = os.path.join(out, 'fixtures_unpickle_head_modules.json')
    rc = subprocess.run([sys.executable, '-u', '-c', HEAD_LOADER, tree, REPO, paths_file, out_file], cwd='/tmp').returncode
    head = _load_ref(out_file) if os.path.isfile(out_file) else {'results': {}, 'modules': {}}
    post = {f['path']: 'ok' for f in fixtures['loaded_list']}
    post.update({f['path']: f['error'] for f in fixtures['failed']})
    newly_failing = sorted(p for p in paths if post.get(p) != 'ok' and head['results'].get(p) == 'ok')
    newly_loading = sorted(p for p in paths if post.get(p) == 'ok' and head['results'].get(p) != 'ok')
    def _norm(text):  # the error text names the module's file, which differs between the two trees by design
        return re.sub(r" from '[^']*'", " from '<module file>'", str(text))
    errors_differ = sorted(p for p in paths if _norm(post.get(p)) != _norm(head['results'].get(p)))
    modules_from_tree = all(str(v).startswith(tree) for v in head['modules'].values()) and bool(head['modules'])
    shutil.rmtree(tree, ignore_errors=True)
    os.remove(tar)
    return {'returncode': rc, 'head_tree': tree, 'modules_loaded_from_head_tree': modules_from_tree,
            'modules': head['modules'], 'n_paths': len(paths), 'n_ok_head': sum(v == 'ok' for v in head['results'].values()),
            'n_ok_post': sum(v == 'ok' for v in post.values()), 'newly_failing': newly_failing,
            'newly_loading': newly_loading, 'outcomes_differing': errors_differ,
            'outcomes_identical_to_head': not errors_differ and modules_from_tree and rc == 0,
            'guard_verify_0': head.get('guard_verify_0')}


def t01a_dry_run(out, run_tag):
    rehearsal = f'/tmp/p516cc_t01a_{run_tag}'
    if os.path.exists(rehearsal):
        raise RuntimeError(f'rehearsal dir exists: {rehearsal}')
    env = dict(os.environ)
    env['P516_D0_REHEARSAL_DIR'] = rehearsal
    log_path = os.path.join(out, 't01a_dry_run_launch.log')
    with open(log_path, 'x') as handle:
        rc = subprocess.run([sys.executable, 'p516_d0_dry_run.py'], cwd=REPO, env=env, stdout=handle,
                            stderr=subprocess.STDOUT).returncode
    committed_dir = os.path.join(REPO, D0_DRY_RUN_REL)
    committed_manifest = os.path.join(committed_dir, 'manifest_sha256.json')
    manifest_sha = _sha_file(committed_manifest)
    manifest = _load_ref(committed_manifest)
    lps = sorted(f for f in os.listdir(committed_dir) if f.endswith('.lp'))
    per = {}
    for f in lps:
        new = os.path.join(rehearsal, f)
        new_sha = _sha_file(new) if os.path.isfile(new) else None
        per[f] = {'committed_manifest_sha256': manifest['files'][f]['sha256'],
                  'committed_file_sha256': _sha_file(os.path.join(committed_dir, f)), 'rerun_sha256': new_sha}
        per[f]['identical'] = (new_sha == per[f]['committed_manifest_sha256'] == per[f]['committed_file_sha256'])
    for name in ('run.log', 'manifest_sha256.json'):
        src = os.path.join(rehearsal, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, f't01a_rehearsal_{name}'))
    committed_exit = manifest.get('child_exit_code')
    rerun_exit = None
    rerun_manifest = os.path.join(rehearsal, 'manifest_sha256.json')
    if os.path.isfile(rerun_manifest):
        rerun_exit = _load_ref(rerun_manifest).get('child_exit_code')
    res = {'rehearsal_dir': rehearsal, 'parent_returncode': rc, 'committed_manifest_sha256': manifest_sha,
           'committed_manifest_is_04033ff7': manifest_sha == D0_DRY_RUN_MANIFEST_SHA,
           'child_exit_code_committed': committed_exit, 'child_exit_code_rerun': rerun_exit,
           'n_lp': len(lps), 'per_lp': per, 'all_lp_identical': bool(lps) and all(v['identical'] for v in per.values())}
    res['pass'] = res['committed_manifest_is_04033ff7'] and res['all_lp_identical'] and committed_exit == rerun_exit
    shutil.rmtree(rehearsal, ignore_errors=True)
    return res


def gate_cc1(out, args, record):
    ref_path = os.path.join(REPO, args.reference, 'w183_nl_digest.json')
    ref = _load_ref(ref_path)
    record['reference'] = {'path': os.path.relpath(ref_path, REPO), 'sha256': _sha_file(ref_path),
                           'git_head_at_run': ref.get('git_head_at_run'),
                           'production_expected_commit': ref['production']['expected_commit'],
                           'production_equal_to_expected_commit': ref['production']['equal_to_expected_commit'],
                           'all_ok': ref.get('all_ok'), 'configs': ref['configuration']['configs'],
                           'how': ('built by the committed, unmodified p515_s53_w183_nl_digest.py at HEAD with '
                                   '--expect-production-commit HEAD, before the production edit')}
    SWITCH_STATE['mode'] = 'off'
    install_esso_probe_and_switches()
    records, capture_checks, ctx = build_srp1(out, args.label, args.scratch)
    record['w183_capture_checks'] = capture_checks
    record['records'] = records
    record['esso_probes'] = deepcopy(ESSO_PROBES)
    comp = compare_records(ref['records'], records, SRP1_CONFIGS)
    record['comparison_vs_head_reference'] = comp
    verdict = identity_verdict(comp)
    record['identity_vs_head_reference'] = verdict
    # secondary: the committed W183c-off reference (an older production head)
    sec = None
    sec_path = os.path.join(REPO, W183_COMMITTED_OFF_REF)
    if os.path.isfile(sec_path):
        sec_ref = _load_ref(sec_path)
        sec_comp = compare_records(sec_ref['records'], records, SRP1_CONFIGS)
        sec = {'path': W183_COMMITTED_OFF_REF, 'sha256': _sha_file(sec_path),
               'git_head_at_run': sec_ref.get('git_head_at_run'), 'identity': identity_verdict(sec_comp)}
    record['secondary_vs_committed_w183c_off_reference'] = sec
    no_term = all(not p['has_kappa'] and not p['has_salvage_credit_obj'] for probes in ESSO_PROBES.values()
                  for p in probes.values())
    _log('fixtures: child process running W183c gate1_fixtures')
    record['fixtures'] = fixtures_child(out)
    _log(f"fixtures: pass {record['fixtures']['pass']} loaded {record['fixtures']['n_loaded']} / "
         f"{record['fixtures']['n_declared']}")
    record['fixtures_under_head_modules'] = fixtures_under_head(out, record['fixtures'], ctx['run_tag'])
    _log(f"fixtures under HEAD modules: identical outcomes {record['fixtures_under_head_modules']['outcomes_identical_to_head']}"
         f" (ok head {record['fixtures_under_head_modules']['n_ok_head']} / post "
         f"{record['fixtures_under_head_modules']['n_ok_post']}; newly failing "
         f"{record['fixtures_under_head_modules']['newly_failing']})")
    _log('T0.1a dry run (rehearsal dir) ...')
    record['t01a_dry_run'] = t01a_dry_run(out, ctx['run_tag'])
    _log(f"T0.1a: pass {record['t01a_dry_run']['pass']} ({record['t01a_dry_run']['n_lp']} LPs)")
    checks = {
        'bitwise_all_blocks_both_builders_both_configs': all(
            v['all_nl_four_files_identical'] and v['all_digests_identical'] for v in verdict.values()),
        'no_salvage_term_constructed_with_switch_off': no_term,
        'every_build_check_passes': all(all(records[c][b]['checks'].values()) for c in records
                                        for b in ('sequential', 'parallel')),
        'fixtures_all_unpickle': record['fixtures']['pass'],
        'fixtures_no_new_failure_vs_head_modules': (record['fixtures_under_head_modules']['outcomes_identical_to_head']
                                                    and not record['fixtures_under_head_modules']['newly_failing']),
        't01a_dry_run_lp_bytes_reproduced': record['t01a_dry_run']['pass'],
        'unchanged_files_unmodified': record['unchanged_files']['unmodified'],
    }
    return checks


def _named_nl(nl_rec):
    """Column-order-invariant reading of one labelled NL file (.sym.nl + .row + .col): every C / J / r entry keyed by
    its row name, every b / x entry by its column name, O / G by the objective name, with each `v<j>` token and J/G
    column index replaced by the column name. Returns (entries, header lines)."""
    def load(rel):
        with gzip.open(os.path.join(REPO, rel), 'rt') as h:
            return h.read()
    text = load(nl_rec['symbolic']['nl']['gz'])
    rows = load(nl_rec['symbolic']['row']['gz']).splitlines()
    cols = load(nl_rec['symbolic']['col']['gz']).splitlines()
    lines = W._strip_nl_comments(text).split('\n')
    header = lines[:10]
    n_vars, n_cons, n_objs = (int(v) for v in header[1].split()[:3])
    segs, cur = [], None
    for line in lines[10:]:
        if line and line[0] in W.NL_SEGMENT_HEADS:
            cur = [line, []]
            segs.append(cur)
        elif cur is not None:
            cur[1].append(line)

    def name(j):
        j = int(j)
        return cols[j] if j < len(cols) else f'<defined var {j}>'

    def mapped(body):
        return '\n'.join(re.sub(r'^v(\d+)$', lambda m: 'v:' + name(m.group(1)), ln) for ln in body if ln != '')

    def pairs(body):
        return sorted((name(ln.split()[0]), ln.split()[1]) for ln in body if ln.strip())

    obj_names = rows[n_cons:n_cons + n_objs]
    out = {}
    for head, body in segs:
        tag = head[0]
        if tag == 'C':
            out[f'C|{rows[int(head[1:].split()[0])]}'] = mapped(body)
        elif tag == 'J':
            out[f'J|{rows[int(head[1:].split()[0])]}'] = pairs(body)
        elif tag == 'O':
            i = int(head[1:].split()[0])
            out[f'O|{obj_names[i] if i < len(obj_names) else i}'] = head.split()[1] + '\n' + mapped(body)
        elif tag == 'G':
            i = int(head[1:].split()[0])
            out[f'G|{obj_names[i] if i < len(obj_names) else i}'] = pairs(body)
        elif tag == 'r':
            for i, ln in enumerate(body[:n_cons]):
                out[f'r|{rows[i]}'] = ln
        elif tag == 'b':
            for j, ln in enumerate(body[:n_vars]):
                out[f'b|{cols[j]}'] = ln
        elif tag == 'x':
            for ln in body:
                if ln.strip():
                    j, v = ln.split()
                    out[f'x|{name(j)}'] = v
        elif tag == 'k':
            continue  # cumulative Jacobian counts in column order: implied by the J entries
        else:
            out[f'{tag}|{head}'] = mapped(body)
    return out, header


def named_nl_diff(rec_a, rec_b, builder, key):
    na, ha = _named_nl(rec_a[builder]['blocks'][key]['nl'])
    nb, hb = _named_nl(rec_b[builder]['blocks'][key]['nl'])
    keys = sorted(set(na) | set(nb))
    diff = [k for k in keys if na.get(k) != nb.get(k)]
    by_class = {}
    for k in diff:
        seg, ident = k.split('|', 1)
        comp = re.sub(r'\[.*$', '', ident)
        by_class.setdefault(f'{seg}|{comp}', []).append(ident)
    examples = {c: [{'entry': f'{c.split("|")[0]}|{i}', 'a': na.get(f'{c.split("|")[0]}|{i}'),
                     'b': nb.get(f'{c.split("|")[0]}|{i}')} for i in v[:2]] for c, v in by_class.items()}
    return {'block': key, 'n_entries_a': len(na), 'n_entries_b': len(nb), 'n_differing': len(diff),
            'differing_by_segment_and_component': {c: len(v) for c, v in sorted(by_class.items())},
            'differing_entries': {c: v for c, v in sorted(by_class.items())}, 'examples': examples,
            'header_lines_differing': [{'line': i + 1, 'a': x, 'b': y} for i, (x, y) in enumerate(zip(ha, hb))
                                       if x != y],
            'column_order_identical': rec_a[builder]['blocks'][key]['nl']['symbolic']['col']['sha256']
            == rec_b[builder]['blocks'][key]['nl']['symbolic']['col']['sha256']}


C1_COMPONENTS = {'Param:salvage_objective_kappa', 'Expression:salvage_credit_obj', 'Objective:objective'}
C2_SALVAGE_EXPRESSION = {'Expression:salvage_value', 'Expression:salvage_credit'}
C3_COMPONENTS = {'Constraint:available_e_capacity_unit'}
C1_NL_CLASSES = {'O|objective', 'G|objective'}
C3_NL_CLASSES = {'C|available_e_capacity_unit', 'J|available_e_capacity_unit'}


def gate_cc2(out, args, record):
    base_path = os.path.join(REPO, args.reference, 'g_cc_results.json')
    base = _load_ref(base_path)
    record['baseline'] = {'path': os.path.relpath(base_path, REPO), 'sha256': _sha_file(base_path),
                          'note': 'the cc1 run (switches off, post-edit), itself bitwise equal to the HEAD reference'}
    install_esso_probe_and_switches()
    arms = {}
    for mode in ('convention', 'c1_only'):
        SWITCH_STATE['mode'] = mode
        mode_out = os.path.join(out, mode)
        os.makedirs(mode_out)
        records, capture_checks, _ctx = build_srp1(mode_out, f'{args.label}_{mode}', f'{args.scratch}_{mode}')
        comp = compare_records(base['records'], records, SRP1_CONFIGS)
        networks_bitwise = True
        esso = {}
        named = {}
        for c in SRP1_CONFIGS:
            for b in ('sequential', 'parallel'):
                r = comp[c][b]
                networks_bitwise = networks_bitwise and not r['blocks_missing']
                for key, e in r['per_block'].items():
                    if key.startswith('ESSO'):
                        esso[f'{c}|{key}'] = {'nl_identical': e['nl_identical'],
                                              'components_differing': e['components_differing'],
                                              'component_detail': e.get('component_detail')}
                        named[f'{c}|{key}'] = named_nl_diff(base['records'][c], records[c], b, key)
                    elif not (all(e['nl_identical'].values()) and not e['components_differing']):
                        networks_bitwise = False
        probes = {k: v for k, v in ESSO_PROBES.items() if k.startswith(mode + '|')}
        arms[mode] = {'switches': SWITCH_MODES[mode], 'records': records, 'w183_capture_checks': capture_checks,
                      'comparison_vs_cc1': comp, 'identity_vs_cc1': identity_verdict(comp),
                      'networks_bitwise': networks_bitwise, 'esso_differences': esso, 'esso_named_nl_diff': named,
                      'esso_probes': deepcopy(probes)}
        _log(f'cc2 {mode}: networks bitwise {networks_bitwise}; ESSO NL plain identical '
             f'{ {k: v["nl_identical"]["plain"] for k, v in esso.items()} }')
    record['arms'] = arms

    def union(mode, prefix, field):
        return sorted({n for k, v in arms[mode]['esso_differences'].items() if k.startswith(prefix) for n in v[field]})

    def nl_classes(mode, prefix):
        return sorted({cl for k, v in arms[mode]['esso_named_nl_diff'].items() if k.startswith(prefix)
                       for cl in v['differing_by_segment_and_component']})

    enum = {mode: {f'{inst}_components': union(mode, inst, 'components_differing') for inst in ('x0', 'unit')}
            for mode in arms}
    for mode in arms:
        for inst in ('x0', 'unit'):
            enum[mode][f'{inst}_nl_classes'] = nl_classes(mode, inst)
            enum[mode][f'{inst}_header_lines'] = {k: v['header_lines_differing'] for k, v in
                                                  arms[mode]['esso_named_nl_diff'].items() if k.startswith(inst)}
    record['enumeration'] = enum
    all_probes = [p for mode in arms for probes in arms[mode]['esso_probes'].values() for p in probes.values()]
    x0_probes = [p for mode in arms for k, probes in arms[mode]['esso_probes'].items() if '|x0_' in k
                 for p in probes.values()]
    c1_x0_plain = {k: v['nl_identical']['plain'] for k, v in arms['c1_only']['esso_differences'].items()
                   if k.startswith('x0')}
    c1_unit_plain = {k: v['nl_identical']['plain'] for k, v in arms['c1_only']['esso_differences'].items()
                     if k.startswith('unit')}
    checks = {
        'network_blocks_bitwise_identical_to_cc1_both_arms': all(a['networks_bitwise'] for a in arms.values()),
        'switches_took_effect': bool(all_probes) and all(p['has_kappa'] for p in all_probes),
        'kappa_zero_at_would_be_solve': all(p.get('kappa_value') == 0.0 for p in all_probes),
        'x0_esso_salvage_term_identically_zero': bool(x0_probes) and all(p['credit_identically_zero']
                                                                         for p in x0_probes),
        'c1_only_components_differing_only_c1': all(set(enum['c1_only'][f'{i}_components']) <= C1_COMPONENTS
                                                    for i in ('x0', 'unit')),
        'c1_only_nl_differing_only_objective_entries': all(set(enum['c1_only'][f'{i}_nl_classes']) <= C1_NL_CLASSES
                                                           for i in ('x0', 'unit')),
        'convention_components_differing_only_c1_c2expr_c3': all(
            set(enum['convention'][f'{i}_components']) <= (C1_COMPONENTS | C2_SALVAGE_EXPRESSION | C3_COMPONENTS)
            for i in ('x0', 'unit')),
        'convention_nl_differing_only_objective_and_available_rows': all(
            set(enum['convention'][f'{i}_nl_classes']) <= (C1_NL_CLASSES | C3_NL_CLASSES) for i in ('x0', 'unit')),
        'every_build_check_passes': all(all(a['records'][c][b]['checks'].values()) for a in arms.values()
                                        for c in a['records'] for b in ('sequential', 'parallel')),
        'unchanged_files_unmodified': record['unchanged_files']['unmodified'],
    }
    record['prediction_outcomes'] = {
        'esso_x0_nl_bitwise_c1_only': {'predicted': True, 'observed': all(c1_x0_plain.values()),
                                       'by_block': c1_x0_plain},
        'esso_unit_nl_bitwise_c1_only': {'predicted': 'not stated', 'observed': all(c1_unit_plain.values()),
                                         'by_block': c1_unit_plain},
        'esso_x0_nl_bitwise_full_convention': {
            'predicted': 'not stated for C3; C3 rewrites available_e_capacity_unit at x = 0 too',
            'observed': all(v['nl_identical']['plain'] for k, v in arms['convention']['esso_differences'].items()
                            if k.startswith('x0'))},
        'network_blocks_bitwise': {'predicted': True,
                                   'observed': checks['network_blocks_bitwise_identical_to_cc1_both_arms']},
    }
    return checks


# ======================================================================================================================
#  the five-block instance (cc5 / cc3)
# ======================================================================================================================
def read_new_instance(out):
    from contextlib import redirect_stdout
    import io
    from shared_resources_planning import SharedResourcesPlanning
    side = os.path.join(out, 'side_writes')
    os.makedirs(os.path.join(side, 'Diagrams'))
    os.makedirs(os.path.join(side, 'Results', 'Logs'))
    diag_dir = os.path.join(REPO, 'data', 'SRP1', 'Diagrams')
    before = {f: os.path.getmtime(os.path.join(diag_dir, f)) for f in os.listdir(diag_dir)}
    data_dir = os.path.join(REPO, 'data', 'SRP1')
    planning = SharedResourcesPlanning(data_dir, os.path.basename(NEW_CASE_REL))
    planning.diagrams_dir = os.path.join(side, 'Diagrams')
    planning.results_dir = os.path.join(side, 'Results')
    planning.logs_dir = os.path.join(side, 'Results', 'Logs')
    stream = io.StringIO()
    t0 = time.time()
    with redirect_stdout(stream):
        planning.read_planning_problem()
    with open(os.path.join(out, 'planning_read_stdout.log'), 'x') as handle:
        handle.write(stream.getvalue())
    after = {f: os.path.getmtime(os.path.join(diag_dir, f)) for f in os.listdir(diag_dir)}
    import p515_g_g1_g4_admm_gates as G
    G._set_results_dir_for_arm(planning, os.path.join(side, 'Results'))
    for holder in [planning.transmission_network] + list(planning.distribution_networks.values()):
        holder.logs_dir = planning.logs_dir
        for year in holder.years:
            for day in holder.days:
                holder.network[year][day].logs_dir = planning.logs_dir
    planning.shared_ess_data.logs_dir = planning.logs_dir
    return planning, {'read_wall_s': time.time() - t0, 'no_side_write_into_data_SRP1_Diagrams': before == after,
                      'side_writes': sorted(os.listdir(os.path.join(side, 'Diagrams')))}


def candidate_for(planning, investments):
    import shared_resources_planning as srp
    candidate = planning.get_initial_candidate_solution()
    for node_id, year, s, e in investments:
        candidate['investment'][node_id][year]['s'] = s
        candidate['investment'][node_id][year]['e'] = e
    srp._rebuild_candidate_total_capacities(planning, candidate)
    return candidate


def instance_facts(planning):
    import shared_energy_storage_data as SED
    import shared_resources_planning as srp
    sed = planning.shared_ess_data
    years = list(sed.years)
    facts = {'years': {str(y): sed.years[y] for y in years}, 'days': {str(d): sed.days[d] for d in sed.days},
             'num_market_scenarios_case': planning.num_market_scenarios if hasattr(planning, 'num_market_scenarios')
             else None,
             'prob_market_scenarios_esso_excel': list(sed.prob_market_scenarios),
             'discount_factor': sed.discount_factor,
             'available_energy_soh_point': sed.available_energy_soh_point,
             'ageing_enabled': sed.ageing_enabled,
             'salvage': {k: getattr(sed.params.salvage_value, k) for k in (
                 'enabled', 'in_objective', 'energy_recovery_fraction', 'recycling_floor_fraction', 'cost_basis',
                 'health_basis', 'calendar_life_basis')},
             'objective_scale_fixed': planning.params.admm.objective_scale,
             'objective_scale_assert_factor': planning.params.admm.objective_scale_assert_factor,
             'esso_al_scale_cfg': dict(planning.params.admm.esso_al_scale)}
    with open(os.path.join(REPO, NEW_CASE_REL)) as handle:
        case = json.load(handle)
    facts['NumMarketScenarios_case_file'] = case['NumMarketScenarios']
    ses0 = sed.shared_energy_storages[years[0]][0]
    tcal = {str(y): {'t_cal': sed.shared_energy_storages[y][0].t_cal, 'width': sed.years[y],
                     'ratio': sed.shared_energy_storages[y][0].t_cal / sed.years[y],
                     'tcal_norm': round(sed.shared_energy_storages[y][0].t_cal / sed.years[y])} for y in years}
    facts['tcal_norm'] = tcal
    facts['terminal_discount'] = SED._get_terminal_discount_factor(sed)
    facts['terminal_discount_expected'] = 1.0 / (1.02 ** 15)
    facts['rlf'] = [SED._get_remaining_calendar_life(sed, i, sed.shared_energy_storages[y][0])[2]
                    for i, y in enumerate(years)]
    facts['expected_energy_cost_by_year'] = {str(y): SED._get_expected_energy_investment_cost(sed, y) for y in years}
    weights = []
    for holder in [planning.transmission_network] + [planning.distribution_networks[n]
                                                     for n in sorted(planning.distribution_networks)]:
        for y in holder.years:
            for d in holder.days:
                w = srp._get_admm_block_weight(holder, y, d)
                expected = 3.0 * float(holder.days[d]) * (1.02 ** -(int(y) - 2025))
                weights.append({'network': holder.name, 'year': y, 'day': d, 'weight': w, 'expected': expected,
                                'rel_err': _rel(w, expected)})
    facts['block_weights'] = weights
    facts['soh_min'] = ses0.soh_min
    facts['phi_cal'] = ses0.phi_cal
    facts['cl_eff'] = ses0.cl_eff
    facts['eff_ch'], facts['eff_dch'] = ses0.eff_ch, ses0.eff_dch
    return facts


def build_instance_blocks(planning, out, config, investments):
    """Production's `run_operational_planning` initialisation with every optimize stubbed (W183's Capture)."""
    import pyomo.environ as pe
    candidate = candidate_for(planning, investments)
    cap = W.Capture(out, config, 'sequential')
    planning.transmission_network.optimize = cap.network_stub('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = cap.network_stub(f'DSO{node_id}')
    sed = planning.shared_ess_data
    probes = {}

    def esso_stub(models, *a, **k):
        probes.update({str(n): _esso_probe(m) for n, m in models.items()})
        for n, m in models.items():
            probes[str(n)]['tcal_window_rows'] = len(m._esso_cohort_constraints)
            probes[str(n)]['kappa_at_init_stub'] = (pe.value(m.salvage_objective_kappa)
                                                    if hasattr(m, 'salvage_objective_kappa') else None)
        return cap.esso_stub(models, *a, **k)

    sed.optimize = esso_stub
    t0 = time.time()
    _c, _r, _m, _s, _p, run_state = planning.run_operational_planning(
        type='distributed', candidate_solution=deepcopy(candidate), print_results=False, debug_flag=False,
        return_state=True)
    expected_calls = {'TSO': 1, 'ESSO': 1, **{f'DSO{n}': 1 for n in planning.distribution_networks}}
    n_net = len([k for k in cap.blocks if not k.startswith('ESSO')])
    n_esso = len([k for k in cap.blocks if k.startswith('ESSO')])
    rec = {'config': config, 'investments': [list(i) for i in investments], 'candidate_investment': candidate[
        'investment'], 'stub_calls': dict(cap.calls), 'stub_calls_expected': expected_calls,
        'initialization_failed': run_state.get('initialization_failed'), 'n_network_blocks': n_net,
        'n_esso_blocks': n_esso, 'blocks': dict(sorted(cap.blocks.items())), 'esso_probes': probes,
        'wall_s': time.time() - t0}
    rec['checks'] = {'stub_calls_exact': cap.calls == expected_calls,
                     'initialization_failed_is_true': run_state.get('initialization_failed') is True,
                     'network_blocks_80': n_net == 80, 'esso_blocks_3': n_esso == 3,
                     'nl_written_every_block': all(v['nl']['error'] is None for v in cap.blocks.values())}
    del _r, _m, run_state, cap
    gc.collect()
    return rec


def gate_cc5(out, args, record):
    planning, read = read_new_instance(out)
    record['read'] = read
    facts = instance_facts(planning)
    record['instance'] = facts
    weights_ok = all(w['rel_err'] <= 1e-15 for w in facts['block_weights'])
    builds = {}
    for config, inv in (('x0', []), ('unit2028', [(UNIT_2028['node'], UNIT_2028['year'], UNIT_2028['s'],
                                                   UNIT_2028['e'])])):
        _log(f'cc5: building {config} ...')
        builds[config] = build_instance_blocks(planning, out, config, inv)
        _log(f"cc5: {config}: {builds[config]['n_network_blocks']} network + {builds[config]['n_esso_blocks']} ESSO; "
             f"checks {builds[config]['checks']} ({builds[config]['wall_s']:.0f}s)")
    record['builds'] = builds
    record['g0_x0_digests'] = {k: {'block_sha256': v['sha256'], 'nl_plain_sha256': v['nl']['plain']['sha256']}
                               for k, v in builds['x0']['blocks'].items()}
    unit_esso7 = builds['unit2028']['esso_probes'].get('7', {})
    checks = {
        'case_loads_no_side_write': read['no_side_write_into_data_SRP1_Diagrams'],
        'years_five_blocks_of_3': facts['years'] == {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3},
        'tcal_norm_5_integral': all(v['tcal_norm'] == 5 and v['ratio'] == 5.0 for v in facts['tcal_norm'].values()),
        'weights_3_days_discounted': weights_ok,
        'weights_80_blocks': len(facts['block_weights']) == 80,
        'terminal_discount_1_02_pow_minus_15': _rel(facts['terminal_discount'],
                                                     facts['terminal_discount_expected']) <= 1e-15,
        'rlf_0_02_04_06_08': all(abs(a - b) <= 1e-15 for a, b in zip(facts['rlf'], [0.0, 0.2, 0.4, 0.6, 0.8])),
        'soh_point_mid_from_file': facts['available_energy_soh_point'] == 'mid',
        'salvage_in_objective_from_file': facts['salvage']['in_objective'] is True,
        'salvage_defaults_frozen': (facts['salvage']['enabled'] is True
                                    and facts['salvage']['energy_recovery_fraction'] == 1.0
                                    and facts['salvage']['recycling_floor_fraction'] == 0.0
                                    and facts['salvage']['cost_basis'] == 'EXPECTED_INSTALLATION_ENERGY_COST'),
        'objective_scale_93635360': facts['objective_scale_fixed'] == 93635360.0,
        'x0_build_checks': all(builds['x0']['checks'].values()),
        'unit2028_build_checks': all(builds['unit2028']['checks'].values()),
        'unit2028_node7_salvage_term_has_free_soh_var': bool(unit_esso7.get('credit_free_nonzero_coefficients')),
        'x0_salvage_term_identically_zero': all(p.get('credit_identically_zero')
                                                for p in builds['x0']['esso_probes'].values()),
        'kappa_zero_at_init_stub': all(p.get('kappa_at_init_stub') == 0.0 for b in builds.values()
                                       for p in b['esso_probes'].values()),
        'unchanged_files_unmodified': record['unchanged_files']['unmodified'],
    }
    return checks


# ---------------------------------------------------------------------------------------------------- cc3 helpers
def _row_parts(model, y_inv):
    """Classify the cohort's rows from `_esso_cohort_constraints` by their variables (never by position)."""
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_variables
    rows = {'avg': {}, 'D': {}, 'soh': {}}
    for name, idx, y in model._esso_cohort_constraints[y_inv]:
        con = getattr(model, name)[idx]
        if not con.equality:
            continue
        names = {v.name for v in identify_variables(con.body, include_fixed=True)}
        avg = model.es_avg_ch_dch_per_unit[y_inv, y].name
        dvar = model.es_D_per_unit[y_inv, y].name
        soh = model.es_soh_per_unit_cumul[y_inv, y].name
        if name == 'energy_storage_charging_discharging':
            rows['avg'][y] = con
        elif dvar in names and avg in names:
            rows['D'][y] = con
        elif soh in names and dvar in names:
            rows['soh'][y] = con
    return rows


def _solve_row_for(con, var):
    """Value of `var` that zeroes the row residual, by two evaluations (exact for rows linear in `var`)."""
    import pyomo.environ as pe
    var.set_value(0.0, skip_validation=True)
    b0 = pe.value(con.body) - pe.value(con.upper)
    var.set_value(1.0, skip_validation=True)
    b1 = pe.value(con.body) - pe.value(con.upper)
    value = -b0 / (b1 - b0)
    var.set_value(value, skip_validation=True)
    return value


def replay(model, y_inv, window):
    """Set avg, D, SoH of cohort y_inv over its window consistently with the model's own equality rows."""
    rows = _row_parts(model, y_inv)
    for y in window:
        _solve_row_for(rows['avg'][y], model.es_avg_ch_dch_per_unit[y_inv, y])
        _solve_row_for(rows['D'][y], model.es_D_per_unit[y_inv, y])
        _solve_row_for(rows['soh'][y], model.es_soh_per_unit_cumul[y_inv, y])


def replay_from_avg(model, y_inv, window, y_from):
    rows = _row_parts(model, y_inv)
    for y in window:
        if y < y_from:
            continue
        if y > y_from:
            _solve_row_for(rows['avg'][y], model.es_avg_ch_dch_per_unit[y_inv, y])
        _solve_row_for(rows['D'][y], model.es_D_per_unit[y_inv, y])
        _solve_row_for(rows['soh'][y], model.es_soh_per_unit_cumul[y_inv, y])


def set_nominal_state(model, y_inv, window, s_mva, e_mwh):
    """A nominal operating point: rated vars = the cohort's Params, 0.5 * S charging in periods 0-3 and discharging in
    periods 12-15 of every day (a feasible-looking schedule; only consistency of the SoH chain matters here)."""
    for y in window:
        model.es_s_rated_per_unit[y_inv, y].set_value(s_mva)
        model.es_e_rated_per_unit[y_inv, y].set_value(e_mwh)
        for d in model.days:
            for p in model.periods:
                model.es_pch_per_unit[y_inv, y, d, p].set_value(0.5 * s_mva if p in (0, 1, 2, 3) else 0.0)
                model.es_pdch_per_unit[y_inv, y, d, p].set_value(0.5 * s_mva if p in (12, 13, 14, 15) else 0.0)
    replay(model, y_inv, window)


def fill_unset_params(model, record_key, filled):
    """Mutable Params left without a value by the ADMM update (requests / duals, zero before the first cycle in
    production) are set to 0.0 so the ADMM objective can be evaluated; every one is counted in the record."""
    import pyomo.environ as pe
    from pyomo.core.base.param import Param
    for comp in model.component_objects(pe.Param, active=None, descend_into=True):
        if not comp.mutable:
            continue
        for i in comp.index_set():
            if comp[i]._value is not Param.NoValue:
                continue
            comp[i].set_value(0.0)
            filled.setdefault(record_key, {}).setdefault(comp.name, 0)
            filled[record_key][comp.name] += 1


def cohort_calculus(model, sed, y_inv, kappa, window, terminal_idx):
    """d(S)/d(TP) and d(objective)/d(TP) per MWh of cell throughput for cohort y_inv: FD through the throughput
    aggregate (avg) with the chain replayed, AD by pyomo `differentiate` through the rows, closed form."""
    import pyomo.environ as pe
    from pyomo.core.expr.calculus.derivatives import differentiate, Modes
    import shared_energy_storage_data as SED
    years = list(sed.years)
    n = sed.years[years[y_inv]]
    ses = sed.shared_energy_storages[years[y_inv]][sed.get_shared_energy_storage_idx(7)]
    rows = _row_parts(model, y_inv)
    soh_T = model.es_soh_per_unit_cumul[y_inv, terminal_idx]
    out = {'y_inv': y_inv, 'year': years[y_inv], 'block_width': n, 'per_block': {}}
    obj_expr = model.objective.expr
    admm_expr = model.admm_objective.expr if hasattr(model, 'admm_objective') else None
    for y in window:
        avg = model.es_avg_ch_dch_per_unit[y_inv, y]
        a0 = pe.value(avg)
        h = FD_REL_STEP * max(abs(a0), 1e-3)
        vals = {}
        for sign in (+1, -1):
            avg.set_value(a0 + sign * h)
            replay_from_avg(model, y_inv, window, y)
            vals[sign] = {'S': pe.value(model.salvage_value), 'credit': pe.value(model.salvage_credit_obj),
                          'obj': pe.value(obj_expr), 'admm': pe.value(admm_expr) if admm_expr is not None else None}
        avg.set_value(a0)
        replay_from_avg(model, y_inv, window, y)
        dTP = 365.0 * n * 2 * h  # TP_y = 365 * n * avg_y (MWh of cell throughput over the block)
        fd = {k: (vals[1][k] - vals[-1][k]) / dTP for k in ('S', 'credit', 'obj')}
        if admm_expr is not None:
            fd['admm'] = (vals[1]['admm'] - vals[-1]['admm']) / dTP
        # AD: implicit derivatives of the rows, chained
        dS_dsohT = differentiate(model.salvage_value.expr, wrt=soh_T, mode=Modes.reverse_numeric)
        dcredit_dsohT = differentiate(model.salvage_credit_obj.expr, wrt=soh_T, mode=Modes.reverse_numeric)
        chain = 1.0
        for yy in [w for w in window if w > y]:
            r = rows['soh'][yy].body
            chain *= -(differentiate(r, wrt=model.es_soh_per_unit_cumul[y_inv, yy - 1], mode=Modes.reverse_numeric)
                       / differentiate(r, wrt=model.es_soh_per_unit_cumul[y_inv, yy], mode=Modes.reverse_numeric))
        r = rows['soh'][y].body
        dsoh_dD = -(differentiate(r, wrt=model.es_D_per_unit[y_inv, y], mode=Modes.reverse_numeric)
                    / differentiate(r, wrt=model.es_soh_per_unit_cumul[y_inv, y], mode=Modes.reverse_numeric))
        r = rows['D'][y].body
        dD_davg = -(differentiate(r, wrt=avg, mode=Modes.reverse_numeric)
                    / differentiate(r, wrt=model.es_D_per_unit[y_inv, y], mode=Modes.reverse_numeric))
        dsohT_dTP = chain * dsoh_dD * dD_davg / (365.0 * n)
        ad = {'S': dS_dsohT * dsohT_dTP, 'credit': dcredit_dsohT * dsohT_dTP}
        ad['obj'] = kappa * ad['credit']
        params = sed.params.salvage_value
        closed = -(SED._get_terminal_discount_factor(sed) * params.energy_recovery_fraction
                   * SED._get_expected_energy_investment_cost(sed, years[y_inv])
                   * SED._get_remaining_calendar_life(sed, y_inv, ses)[2] * (1.0 - params.recycling_floor_fraction)
                   * pe.value(soh_T) / (2.0 * ses.cl_eff * (1.0 - ses.soh_min)))
        entry = {'avg0': a0, 'h': h, 'fd': fd, 'ad': ad, 'closed_form_dS_dTP': closed, 'soh_T': pe.value(soh_T),
                 'rel_fd_vs_ad_S': _rel(fd['S'], ad['S']), 'rel_ad_vs_closed_S': _rel(ad['S'], closed),
                 'rel_obj_vs_kappa_credit_fd': _rel(fd['obj'], kappa * fd['credit']),
                 'rel_obj_fd_vs_ad': _rel(fd['obj'], ad['obj']),
                 'rel_credit_vs_minus_S_fd': _rel(fd['credit'], -fd['S'])}
        if admm_expr is not None:
            entry['rel_admm_vs_obj_fd'] = _rel(fd['admm'], fd['obj'])
        out['per_block'][str(years[y])] = entry
    return out


def gate_cc3(out, args, record):
    import pyomo.environ as pe
    import shared_energy_storage_data as SED
    import shared_resources_planning as srp
    from definitions import PENALTY_ESSO_SLACK, EPS_ESSO_THROUGHPUT
    planning, read = read_new_instance(out)
    record['read'] = read
    sed = planning.shared_ess_data
    years = list(sed.years)
    admm = planning.params.admm
    terminal_idx = len(years) - 1
    # ---------------------------------------------------------------- build order (source) + full zero-solve build
    src = inspect.getsource(srp._run_operational_planning)
    order = {'create_esso_at': src.find('create_shared_energy_storage_model('),
             'update_esso_to_admm_at': src.find('update_shared_energy_storage_model_to_admm('),
             'build_subproblem_kappa_init_0': 'salvage_objective_kappa = pe.Param(mutable=True, initialize=0.00)'
             in inspect.getsource(SED._build_subproblem)}
    order['create_before_update'] = 0 <= order['create_esso_at'] < order['update_esso_to_admm_at']
    record['build_order'] = order
    candidate = candidate_for(planning, [(UNIT_2028['node'], UNIT_2028['year'], UNIT_2028['s'], UNIT_2028['e'])])
    consensus_vars, _dual_vars = srp.create_admm_variables(planning)
    calls = {}
    kappa_at_init = {}

    def net_stub(tag):
        def stub(model, *a, **k):
            calls[tag] = calls.get(tag, 0) + 1
            return {y: {d: None for d in model[y]} for y in model}
        return stub

    def esso_stub(models, *a, **k):
        calls['ESSO'] = calls.get('ESSO', 0) + 1
        kappa_at_init.update({str(n): pe.value(m.salvage_objective_kappa) for n, m in models.items()})
        return {n: None for n in models}

    planning.transmission_network.optimize = net_stub('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = net_stub(f'DSO{node_id}')
    sed.optimize = esso_stub
    premium = admm.interface_deviation_premium
    dso_models, _ = srp.create_distribution_networks_models(
        planning.distribution_networks, consensus_vars, candidate['total_capacity'], parallel_execution=False,
        premium_alpha=premium['alpha'], premium_floor=premium['floor'])
    tso_model, _ = srp.create_transmission_network_model(planning, consensus_vars, candidate['total_capacity'])
    esso, _ = srp.create_shared_energy_storage_model(sed, consensus_vars, candidate['investment'])
    record['init_stub_calls'] = calls
    record['kappa_at_initialisation_stub'] = kappa_at_init
    srp._prepare_distribution_objectives_for_admm(planning.distribution_networks, dso_models)
    srp._prepare_transmission_objectives_for_admm(planning.transmission_network, tso_model)
    sigma = admm.objective_scale
    a_scale, applied = srp._resolve_esso_al_scale(planning, admm, sigma)
    srp.update_distribution_models_to_admm(planning, dso_models, admm, sigma)
    srp.update_transmission_model_to_admm(planning, tso_model, admm, sigma)
    srp.update_shared_energy_storage_model_to_admm(planning, esso, admm, al_scale_esso=a_scale, objective_scale=sigma)
    record['sigma'] = {'sigma_used': sigma, 'source': 'admm.objective_scale (sigma_fixed, SRP1_params.json)',
                       'sigma_computed': None,
                       'sigma_computed_note': ('needs the initialisation solve (network objective values); the '
                                               '[1/3, 3] band assertion is therefore not evaluated here')}
    # ---------------------------------------------------------------- kappa hand replica
    w_hand = []
    for holder in [planning.transmission_network] + list(planning.distribution_networks.values()):
        for y in holder.years:
            for d in holder.days:
                w_hand.append(3.0 * float(holder.days[d]) * (1.02 ** -(int(y) - 2025)))
    w_hand.sort()
    m = len(w_hand)
    w_med = 0.5 * (w_hand[m // 2 - 1] + w_hand[m // 2]) if m % 2 == 0 else w_hand[m // 2]
    kappa_rec = {}
    for node, model in esso.items():
        kappa = pe.value(model.salvage_objective_kappa)
        a_param = pe.value(model.admm_esso_al_scale) if applied else None
        s_param = pe.value(model.admm_common_objective_scale)
        kappa_rec[str(node)] = {'kappa': kappa, 'admm_esso_al_scale': a_param, 'admm_common_objective_scale': s_param,
                                'kappa_from_params': (a_param / s_param) if applied else 1.0 / s_param,
                                'kappa_hand_a_over_sigma': (sigma / w_med) / sigma, 'kappa_hand_1_over_wmed': 1 / w_med}
        r = kappa_rec[str(node)]
        r['rel_kappa_vs_params'] = _rel(kappa, r['kappa_from_params'])
        r['rel_kappa_vs_hand'] = _rel(kappa, r['kappa_hand_a_over_sigma'])
        r['rel_kappa_vs_1_over_wmed'] = _rel(kappa, r['kappa_hand_1_over_wmed'])
    record['kappa'] = {'al_scale_applied': applied, 'a': a_scale, 'w_med_hand': w_med, 'n_blocks': m,
                       'per_node': kappa_rec}
    kappa = kappa_rec['7']['kappa']
    filled = {}
    # ---------------------------------------------------------------- calculus on the unit cell (2028 cohort, node 7)
    m7 = esso[7]
    fill_unset_params(m7, 'ESSO7', filled)
    y28 = years.index(2028)
    window28 = [y for y in m7.years if not m7.es_e_rated_per_unit[y28, y].fixed]
    set_nominal_state(m7, y28, window28, UNIT_2028['s'], UNIT_2028['e'])
    S0 = pe.value(m7.salvage_value)
    credit0 = pe.value(m7.salvage_credit_obj)
    record['unit_cell_value_identity'] = {'S': S0, 'credit': credit0, 'rel_credit_vs_minus_S': _rel(credit0, -S0)}
    unit_calc = cohort_calculus(m7, sed, y28, kappa, window28, terminal_idx)
    record['unit_cell_calculus'] = unit_calc
    # ---------------------------------------------------------------- euro parity
    parity = {}
    soh_T = m7.es_soh_per_unit_cumul[y28, terminal_idx]
    s0 = pe.value(soh_T)
    h = 1e-6
    vals = {}
    for sign in (+1, -1):
        soh_T.set_value(s0 + sign * h)
        vals[sign] = (pe.value(m7.salvage_value), pe.value(m7.admm_objective.expr))
    soh_T.set_value(s0)
    r_esso = (vals[1][1] - vals[-1][1]) / (vals[1][0] - vals[-1][0])
    parity['esso'] = {'d_admm_objective_per_euro_S': r_esso, 'minus_r_over_a': -r_esso / a_scale,
                      'one_over_sigma': 1.0 / sigma, 'rel_minus_r_over_a_vs_1_over_sigma': _rel(-r_esso / a_scale,
                                                                                               1.0 / sigma),
                      'rel_minus_r_vs_kappa': _rel(-r_esso, kappa)}
    from pyomo.core.expr.visitor import identify_variables
    net_parity = {}
    for label, holder, model in (('DSO7', planning.distribution_networks[7], dso_models[7]),
                                 ('TSO', planning.transmission_network, tso_model)):
        blk = model[2028][list(holder.days)[1]]
        day = list(holder.days)[1]
        fill_unset_params(blk, label, filled)
        for v in blk.component_data_objects(pe.Var, descend_into=True):
            if v.value is None:
                v.set_value(0.0, skip_validation=True)
        admm_e = blk.admm_objective.expr
        al_vars = set()
        for arg in admm_e.args[1:]:
            al_vars |= {id(v) for v in identify_variables(arg)}
        chosen = None
        for v in identify_variables(blk.objective.expr):
            if v.fixed or id(v) in al_vars:
                continue
            v0 = v.value
            hh = 1e-4 * max(abs(v0), 1.0)
            v.set_value(v0 + hh, skip_validation=True)
            r1, a1 = pe.value(blk.objective.expr), pe.value(admm_e)
            v.set_value(v0 - hh, skip_validation=True)
            r2, a2 = pe.value(blk.objective.expr), pe.value(admm_e)
            v.set_value(v0, skip_validation=True)
            if r1 != r2:
                chosen = (v.name, r1 - r2, a1 - a2)
                break
        w_b = srp._get_admm_block_weight(holder, 2028, day)
        if chosen is None:
            net_parity[label] = {'error': 'no free cost variable outside the AL terms found'}
            continue
        name, d_raw, d_admm = chosen
        r_net = d_admm / (w_b * d_raw)
        net_parity[label] = {'block': f'{holder.name}|2028|{day}', 'variable': name, 'block_weight': w_b,
                             'admm_objective_scale_param': pe.value(blk.admm_objective_scale),
                             'd_admm_per_euro_pv': r_net, 'rel_vs_1_over_sigma': _rel(r_net, 1.0 / sigma)}
    parity['networks'] = net_parity
    parity['identity_holds'] = (parity['esso']['rel_minus_r_over_a_vs_1_over_sigma'] <= REL_TOL
                                and all(v.get('rel_vs_1_over_sigma', 1.0) <= REL_TOL for v in net_parity.values()))
    parity['literal_reading'] = {
        'd_esso_objective_per_euro_S_magnitude': -r_esso, 'd_block_admm_objective_per_euro_pv':
            {k: v.get('d_admm_per_euro_pv') for k, v in net_parity.items()},
        'note': ('without the a-normalisation the two differ by the factor a = sigma / w_med (kappa = a / sigma, '
                 'not 1 / sigma, per F1)')}
    record['euro_parity'] = parity
    record['params_filled_with_zero_for_evaluation'] = filled
    # ---------------------------------------------------------------- per-cohort probe (ESSO only, every cohort on)
    probe_inv = [(7, y, 0.25, 1.0) for y in PROBE_YEARS]
    probe_candidate = candidate_for(planning, probe_inv)
    sed.update_data_with_candidate_solution(probe_candidate['investment'])
    probe_models = sed.build_subproblem()
    sed.update_model_with_candidate_solution(probe_models, probe_candidate['investment'])
    srp.update_shared_energy_storage_model_to_admm(planning, probe_models, admm, al_scale_esso=a_scale,
                                                   objective_scale=sigma)
    pm = probe_models[7]
    fill_unset_params(pm, 'probe_ESSO7', filled)
    kappa_p = pe.value(pm.salvage_objective_kappa)
    per_cohort = {}
    for y_inv, year in enumerate(years):
        window = [y for y in pm.years if not pm.es_e_rated_per_unit[y_inv, y].fixed]
        set_nominal_state(pm, y_inv, window, 0.25, 1.0)
    for y_inv, year in enumerate(years):
        window = [y for y in pm.years if not pm.es_e_rated_per_unit[y_inv, y].fixed]
        in_term = terminal_idx in window and SED._get_remaining_calendar_life(
            sed, y_inv, sed.shared_energy_storages[year][0])[2] > 0
        if not in_term:
            # the cohort is outside the term: d(credit)/d(anything of the cohort) is exactly 0 -- shown by FD
            avg = pm.es_avg_ch_dch_per_unit[y_inv, window[0]]
            a0 = pe.value(avg)
            c0 = pe.value(pm.salvage_credit_obj)
            avg.set_value(a0 + 1.0)
            replay_from_avg(pm, y_inv, window, window[0])
            c1 = pe.value(pm.salvage_credit_obj)
            avg.set_value(a0)
            replay_from_avg(pm, y_inv, window, window[0])
            per_cohort[str(year)] = {'in_term': False, 'window': [years[y] for y in window],
                                     'credit_change_for_unit_avg_step': c1 - c0, 'dS_dTP_eur_per_MWh': 0.0}
            continue
        calc = cohort_calculus(pm, sed, y_inv, kappa_p, window, terminal_idx)
        vals_fd = [b['fd']['S'] for b in calc['per_block'].values()]
        per_cohort[str(year)] = {'in_term': True, 'window': [years[y] for y in window], 'calculus': calc,
                                 'dS_dTP_eur_per_MWh': vals_fd[0],
                                 'spread_over_blocks_rel': max(_rel(v, vals_fd[0]) for v in vals_fd)}
    record['per_cohort_probe'] = {'investments': probe_inv, 'kappa': kappa_p, 'per_cohort': per_cohort}
    # ---------------------------------------------------------------- slack safety
    coeffs = pm._salvage_throughput_coefficients
    worst = {}
    for (y_inv, y, d), (c_ch, c_dch) in coeffs.items():
        soh_t = pe.value(pm.es_soh_per_unit_cumul[y_inv, terminal_idx])
        ses = sed.shared_energy_storages[years[y_inv]][0]
        age_T = SED._get_remaining_calendar_life(sed, y_inv, ses)[0]
        soh_ub = ses.phi_cal ** age_T
        for direction, c in (('pch', c_ch), ('pdch', c_dch)):
            val = kappa_p * c * soh_t
            val_ub = kappa_p * c * soh_ub
            key = str(years[y_inv])
            if key not in worst or val_ub > worst[key]['marginal_at_calendar_soh_ub']:
                worst[key] = {'y': years[y], 'day': list(sed.days)[d], 'direction': direction,
                              'marginal_obj_units_per_MW_at_replay_soh': val,
                              'marginal_obj_units_per_MW_at_calendar_soh_ub': val_ub,
                              'marginal_at_calendar_soh_ub': val_ub, 'soh_T_replay': soh_t,
                              'soh_T_calendar_ub': soh_ub, 'euro_per_MW_period': c * soh_t}
    # FD confirmation of the worst entry per cohort: perturb the variable itself and replay the chain
    threshold = SLACK_SAFETY_FACTOR * PENALTY_ESSO_SLACK
    for key, w in worst.items():
        y_inv = years.index(int(key))
        y = years.index(w['y'])
        d = list(sed.days).index(w['day'])
        var = (pm.es_pch_per_unit if w['direction'] == 'pch' else pm.es_pdch_per_unit)[y_inv, y, d, 5]
        window = [yy for yy in pm.years if not pm.es_e_rated_per_unit[y_inv, yy].fixed]
        v0 = pe.value(var)
        hh = 1e-6
        res = {}
        for sign in (+1, -1):
            var.set_value(v0 + sign * hh, skip_validation=True)
            replay(pm, y_inv, window)
            res[sign] = (pe.value(pm.salvage_objective_kappa * pm.salvage_credit_obj), pe.value(pm.objective.expr))
        var.set_value(v0)
        replay(pm, y_inv, window)
        w['fd_salvage_marginal_obj_units_per_MW'] = (res[1][0] - res[-1][0]) / (2 * hh)
        w['fd_objective_marginal_incl_eps'] = (res[1][1] - res[-1][1]) / (2 * hh)
        w['rel_fd_vs_closed'] = _rel(w['fd_salvage_marginal_obj_units_per_MW'],
                                     w['marginal_obj_units_per_MW_at_replay_soh'])
        w['eps'] = EPS_ESSO_THROUGHPUT
    max_val = max(w['marginal_obj_units_per_MW_at_calendar_soh_ub'] for w in worst.values())
    max_replay = max(w['marginal_obj_units_per_MW_at_replay_soh'] for w in worst.values())
    record['slack_safety'] = {'PENALTY_ESSO_SLACK': PENALTY_ESSO_SLACK, 'threshold': threshold,
                              'per_cohort_worst': worst, 'max_marginal_at_calendar_soh_ub': max_val,
                              'max_marginal_at_replay_soh': max_replay,
                              'ratio_to_threshold_ub': max_val / threshold,
                              'ratio_to_penalty_ub': max_val / PENALTY_ESSO_SLACK,
                              'holds': max_val <= threshold,
                              'per_MWh_reading': {k: kappa_p * abs(v['dS_dTP_eur_per_MWh'])
                                                  for k, v in per_cohort.items()},
                              'definition': ('salvage marginal per MW of a cohort pch/pdch at ONE representative '
                                             'period (the unit es_pnet and the slack pair are measured in), in '
                                             'objective units: kappa * d(-S)/dp = kappa * c * SoH_T; worst over '
                                             'cohort, block, day and direction; SoH_T at the calendar-only upper '
                                             'bound phi_cal**age_T (F4) and at the replayed nominal point')}
    # ---------------------------------------------------------------- complementarity diagnostic
    diag = SED._get_esso_complementarity_diagnostics(pm, 7, os.path.join(out, 'no_such_ipopt.log'))
    pairs = []
    for y_inv in pm.years:
        for y in pm.years:
            if not SED._esso_cohort_pair_is_within_lifetime(pm, y_inv, y) or pm._esso_cohort_inactive.get(y_inv):
                continue
            for d in pm.days:
                c_ch, c_dch = coeffs.get((y_inv, y, d), (0.0, 0.0))
                soh_t = pe.value(pm.es_soh_per_unit_cumul[y_inv, terminal_idx])
                pairs.append(2 * EPS_ESSO_THROUGHPUT + kappa_p * (c_ch + c_dch) * soh_t)
    record['complementarity_diagnostic'] = {
        'record': diag, 'hand_pair_marginal_min': min(pairs), 'hand_pair_marginal_max': max(pairs),
        'matches_hand': (diag.get('effective_pair_marginal_min') == min(pairs)
                         and diag.get('effective_pair_marginal_max') == max(pairs)),
        'note': ('no IPOPT log exists (zero solves), so mu_final/s_obj are None and both bounds are None; the '
                 'effective marginal fields are what is checked')}
    # ---------------------------------------------------------------- checks + predictions
    unit_blocks = list(unit_calc['per_block'].values())
    exp = {}
    for year, lo, hi in (('2028', 0.5, 2.0), ('2037', 2.5, 10.0)):
        v = abs(per_cohort[year]['dS_dTP_eur_per_MWh'])
        exp[year] = {'abs_dS_dTP': v, 'band': [lo, hi], 'held': lo <= v <= hi}
    exp['2025'] = {'abs_dS_dTP': abs(per_cohort['2025']['dS_dTP_eur_per_MWh']),
                   'held': per_cohort['2025']['dS_dTP_eur_per_MWh'] == 0.0
                   and per_cohort['2025'].get('credit_change_for_unit_avg_step') == 0.0}
    record['prediction_outcomes'] = {'expert_dS_dTP': exp,
                                     'slack_safety': {'predicted': 'holds', 'held': record['slack_safety']['holds']}}
    in_term_calcs = [b for c in per_cohort.values() if c['in_term'] for b in c['calculus']['per_block'].values()]
    checks = {
        'init_stub_calls_exact': calls == {'TSO': 1, 'ESSO': 1, **{f'DSO{n}': 1 for n in
                                                                   planning.distribution_networks}},
        'kappa_zero_at_initialisation': all(v == 0.0 for v in kappa_at_init.values()) and bool(kappa_at_init),
        'build_order_create_before_update': order['create_before_update'] and order['build_subproblem_kappa_init_0'],
        'kappa_equals_params_ratio': all(v['rel_kappa_vs_params'] == 0.0 for v in kappa_rec.values()),
        'kappa_hand_replica_1e-12': all(v['rel_kappa_vs_hand'] <= 1e-12 for v in kappa_rec.values()),
        'unit_credit_equals_minus_S': record['unit_cell_value_identity']['rel_credit_vs_minus_S'] <= 1e-12,
        'd_obj_d_TP_equals_kappa_d_credit_1e-6': all(b['rel_obj_vs_kappa_credit_fd'] <= REL_TOL
                                                     and b['rel_obj_fd_vs_ad'] <= REL_TOL
                                                     for b in unit_blocks + in_term_calcs),
        'fd_vs_ad_1e-6': all(b['rel_fd_vs_ad_S'] <= REL_TOL for b in unit_blocks + in_term_calcs),
        'ad_vs_closed_form_1e-6': all(b['rel_ad_vs_closed_S'] <= REL_TOL for b in unit_blocks + in_term_calcs),
        'dS_dTP_negative_2028plus': all(per_cohort[str(y)]['dS_dTP_eur_per_MWh'] < 0 for y in PROBE_YEARS[1:]),
        'dS_dTP_zero_2025': exp['2025']['held'],
        'admm_objective_carries_term': all(b.get('rel_admm_vs_obj_fd', 1.0) <= REL_TOL for b in unit_blocks),
        'euro_parity_identity': parity['identity_holds'],
        'slack_safety': record['slack_safety']['holds'],
        'diagnostic_effective_marginal_matches': record['complementarity_diagnostic']['matches_hand'],
        'unchanged_files_unmodified': record['unchanged_files']['unmodified'],
    }
    return checks


# ======================================================================================================================
#  main
# ======================================================================================================================
GATES = {'cc1': gate_cc1, 'cc2': gate_cc2, 'cc3': gate_cc3, 'cc5': gate_cc5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', required=True, choices=sorted(GATES))
    parser.add_argument('--run-name', required=True, help='subdirectory of data/SRP1/Results/P516CC/g_<gate>')
    parser.add_argument('--reference', default=None,
                        help='cc1: dir of the HEAD W183 reference; cc2: dir of the cc1 run (repo-relative)')
    parser.add_argument('--scratch', default=None, help='cc1/cc2: scratch dir outside the repo (must not exist)')
    args = parser.parse_args()
    started = time.time()
    gate_dir = os.path.join(ROOT, f'g_{args.gate}')
    out = os.path.join(gate_dir, args.run_name)
    if os.path.exists(out):
        raise SystemExit(f'[REFUSED] output exists (write-once): {out}')
    if args.gate in ('cc1', 'cc2'):
        if not args.reference or not args.scratch:
            raise SystemExit('[REFUSED] cc1/cc2 need --reference and --scratch')
        if os.path.exists(args.scratch) or os.path.abspath(args.scratch).startswith(REPO + os.sep):
            raise SystemExit(f'[REFUSED] scratch exists or is inside the repo: {args.scratch}')
        args.label = f'p516cc_{args.gate}_{args.run_name}'
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'[REFUSED] another p516_cc_gates run holds {LOCK_PATH}')
    os.write(fd, f'{os.getpid()} {_utc()}\n'.encode())
    os.close(fd)
    try:
        os.makedirs(out)
        _write(out, 'predictions.json', {'gate': args.gate, 'recorded_utc': _utc(),
                                         'predictions': PREDICTIONS[args.gate]})
        capture = assert_capture_paths(args.gate)
        _log(f'gate {args.gate} -> {os.path.relpath(out, REPO)}; rule-eleven checks {capture}')
        record = {'gate': args.gate, 'argv': sys.argv, 'started_utc': _utc(), 'script': SCRIPT_REL,
                  'script_sha256': _sha_file(__file__), 'script_git_status': W.git_status_of([SCRIPT_REL]) or 'clean',
                  'production': production_status(), 'unchanged_files': unchanged_files_check(),
                  'new_files_sha256': {f: _sha_file(os.path.join(REPO, f)) for f in (NEW_CASE_REL, NEW_ESS_PARAMS_REL)
                                       if os.path.isfile(os.path.join(REPO, f))},
                  'capture_path_checks': capture, 'interpreter': sys.executable, 'python': sys.version,
                  'env': {k: os.environ.get(k) for k in ('NLP_SOLVER_PATH', 'LP_SOLVER_PATH')}}
        import pyomo
        record['pyomo_version'] = pyomo.version.version
        watch = MemoryWatch().start()
        modules_start = W.repo_modules_imported()
        error = None
        try:
            checks = GATES[args.gate](out, args, record)
        except Exception as exc:  # noqa: BLE001 -- recorded, the run fails
            import traceback
            error = traceback.format_exc()
            checks = {'gate_completed_without_exception': False}
            _log(f'[ERROR] {error}')
        record['memory'] = watch.stop()
        checks['memory_below_16_GiB'] = record['memory']['breach'] is None
        modules_end = W.repo_modules_imported()
        checks['imported_repo_modules_unchanged_during_run'] = modules_start == {k: modules_end.get(k)
                                                                                  for k in modules_start}
        guard_failures = GUARD.verify(0)
        checks['guard_zero_solves_verified'] = not guard_failures
        record['solve_profile_guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures}
        record['error'] = error
        record['checks'] = checks
        record['all_ok'] = all(v is True for v in checks.values())
        record['wall_clock_s'] = time.time() - started
        record['repo_modules_imported_sha256'] = modules_end
        path = _write(out, 'g_cc_results.json', record)
        manifest = {}
        for root, _dirs, files in os.walk(out):
            for f in sorted(files):
                p = os.path.join(root, f)
                if os.path.relpath(p, out) == 'manifest_sha256.json':
                    continue
                manifest[os.path.relpath(p, REPO)] = _sha_file(p)
        _write(out, 'manifest_sha256.json', dict(sorted(manifest.items())))
        for k, v in checks.items():
            _log(f'  {"OK  " if v is True else "FAIL"} {k}')
        _log(f'wrote {os.path.relpath(path, REPO)} sha256={_sha_file(path)}; manifest {len(manifest)} files; '
             f'ALL_OK={record["all_ok"]}; guard {dict(GUARD.counts)}; peak RSS '
             f'{record["memory"]["peak_rss_gib"]:.2f} GiB; wall {record["wall_clock_s"]:.0f}s')
        sys.exit(0 if record['all_ok'] else 1)
    finally:
        os.remove(LOCK_PATH)


if __name__ == '__main__':
    main()
