"""
P5.15 Addenda 28-29 (task W20, item 3) -- ZERO-SOLVE gate for the case-file
consistency edit `data/SRP1/SharedESS/SRP1_ESS_Params.json`
`max_energy_to_power_factor` 10 -> 4 (frozen spec v16 `case_file_edit`).
Armed `SolveProfileGuard(permitted=())` for the whole script (installed before
any model import), `verify(0)` at the end.

TWO PHASES, two separate processes (the baseline planning object is read once per
process by `p56a_oracle.load_baseline`):
  --phase before   run on the UNEDITED file (factor 10, sha256 ESS_PARAMS_SHA256_BEFORE)
  --phase after    run on the EDITED file (factor 4); loads the committed-to-be
                   `before` record (its sha256 is passed with --before-sha256 and
                   checked) and compares.

WHAT EACH PHASE DOES
  1. READERS. Every `.py` file under the repository (tracked and untracked, `data/`
     included) is searched for `max_energy_to_power` (the JSON key
     `max_energy_to_power_factor` and the attribute `max_energy_to_power_ratio` it
     is loaded into); every hit is listed as file:line with its enclosing function
     and a classification from READER_CLASSIFICATION. An UNCLASSIFIED hit fails
     the gate (a new reader must be looked at, not assumed away).
  2. ORACLE-PATH BUILD at C* and at x = 0, investment year 2025: the planning object
     is built by `p515_g_g1_g4_admm_gates._construct_arm_planning` (the code
     `run_admm_arm` calls; cap 500, apply_rho False), checked by the campaign
     child's configuration hook (`_config_hook_factory`, case-file AA declared),
     then `run_operational_planning`'s initialization is replayed by calling the
     production builders it calls -- `create_admm_variables`,
     `create_distribution_networks_models`, `create_transmission_network_model`,
     `create_shared_energy_storage_model` -- with each holder's `optimize` replaced
     (instance attribute) by a stub that DIGESTS every block handed to it and
     returns no result (so nothing is solved). Every TSO / DSO block (year, day)
     and every ESSO node model is digested: every Param value (mutable and
     immutable), every Var (value, raw lb, raw ub, fixed), every Constraint
     (active, lower, upper) and Objective (active), every block's active flag.
     During the build a read counter on the shared-ESS parameters object counts
     reads of `max_energy_to_power_ratio` (expected 0).
  3. `after` only: every block digest equal to `before`'s; the loaded ratio is 10.0
     before and 4.0 after; the file differs from the pre-edit file in that key only.

PINS: committed campaign specs are searched (git grep) for the pre-edit sha256 of
the ESS parameters file; the hits are reported (the case file SRP1_params.json is
a different file and is not edited).

Output (write-once): data/SRP1/Results/P515S46/case_file_edit/{before,after}/
    case_file_edit_<phase>.json, manifest_sha256.json
Launch (attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_case_file_edit_check.py \\
        --phase before > data/SRP1/Results/P515S46/case_file_edit_before_launch.log 2>&1
    (edit the file)
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_case_file_edit_check.py \\
        --phase after --before-sha256 <sha> > data/SRP1/Results/P515S46/case_file_edit_after_launch.log 2>&1
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W20 case-file edit gate (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = ('P5.15 Addenda 28-29 W20 item 3 -- SRP1_ESS_Params.json max_energy_to_power_factor 10 -> 4: '
         'zero-solve gate')
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 28',
             'data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json (case_file_edit)',
             'Planner task W20 item 3']
ESS_PARAMS_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
ESS_PARAMS_SHA256_BEFORE = 'badcdeae4d2a0ac9d369fb7cfb028ecb84b369c881bc0b4e56c590d38a3a13bb'
KEY = 'max_energy_to_power_factor'
ATTR = 'max_energy_to_power_ratio'
VALUE_BEFORE, VALUE_AFTER = 10.0, 4.0
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S46', 'case_file_edit')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
C_STAR_KEY_PIN = '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c'

ORACLE_PATH = ('p515_s44_campaign_harness child -> p515_g_g1_g4_admm_gates.run_admm_arm -> '
               '_construct_arm_planning (p56a_oracle.fresh_planning, candidate) -> '
               'SharedResourcesPlanning.run_operational_planning(type="distributed") -> '
               'shared_resources_planning._run_operational_planning (create_* builders, the ADMM loop, '
               '_get_operational_sensitivities at convergence)')
# file -> {enclosing function: classification}. Every hit of the search must be classified here.
READER_CLASSIFICATION = {
    'shared_energy_storage_parameters.py': {
        'SharedEnergyStorageParameters.__init__': 'default value of the attribute (10.00); not a reader of the file',
        '_read_parameters_from_file': ('THE LOADER: reads the JSON key into params.max_energy_to_power_ratio; '
                                       'loads the value, consumes nothing'),
    },
    'shared_energy_storage_data.py': {
        '_build_master_problem': ('the Benders MASTER problem (E <= ratio * S row); called only by '
                                  'shared_resources_planning._run_planning_problem, never on the oracle path'),
    },
    'shared_resources_planning.py': {
        '_check_candidate_first_stage_feasibility': (
            'first-stage feasibility of a candidate; called by _validate_local_sensitivities_with_finite_'
            'differences and _build_positive_bootstrap_candidate (both inside _run_planning_problem, the Benders '
            'loop), never by _run_operational_planning'),
        '_build_positive_bootstrap_candidate': ('Benders bootstrap candidate; called by _run_planning_problem '
                                                'and _complete_missing_sensitivities_with_probe (Benders loop)'),
    },
    'p56a_oracle.py': {
        '_config_hash': ('the P5.6 oracle CACHE key (evaluate_planning_candidate); the campaign harness does '
                         'not use the p56a cache (it calls run_admm_arm directly); a changed value would only '
                         'invalidate p56a cache entries'),
    },
    'p56b_b5_coordinates.py': {'*': 'P5.6-B diagnostic (candidate coordinates); not on the oracle path'},
    'p56b_candidates.py': {'*': 'P5.6-B diagnostic (candidate generation); not on the oracle path'},
    'p514_d_preflight.py': {'*': 'P5.14-D preflight record; not on the oracle path'},
    'p515_s44_addendum26_confirmations.py': {'*': 'W-stage confirmation record (reads and reports the value)'},
    'p515_s45_investment_cost_recompute.py': {'*': 'W2 I(x) recompute (records the value); not the oracle'},
    'p515_s46_case_file_edit_check.py': {'*': 'this gate'},
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W20-casefile] {msg}', flush=True)


def _git(args, check=True):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=check).stdout


def _sha(path):
    return H.sha256_file(path)


# ======================================================================================================================
#  readers
# ======================================================================================================================
def _enclosing_function(path, lineno):
    try:
        tree = ast.parse(open(path).read())
    except (SyntaxError, UnicodeDecodeError):
        return None
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, 'end_lineno', node.lineno)
            if node.lineno <= lineno <= end and (best is None or node.lineno >= best[1]):
                best = (node, node.lineno)
    if best is None:
        return '<module>'
    node = best[0]
    # qualify methods with their class
    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef) and node in parent.body:
            return f'{parent.name}.{node.name}'
    return node.name


def readers():
    out = subprocess.run(['grep', '-rn', '--include=*.py', 'max_energy_to_power', '.'], cwd=REPO,
                         capture_output=True, text=True).stdout
    tracked = set(_git(['ls-files', '--', '*.py']).splitlines())
    hits, unclassified = [], []
    for line in out.splitlines():
        rel, lineno, text = line.split(':', 2)
        rel = rel[2:] if rel.startswith('./') else rel
        lineno = int(lineno)
        func = _enclosing_function(os.path.join(REPO, rel), lineno)
        table = READER_CLASSIFICATION.get(rel, {})
        classification = table.get(func) or table.get('*')
        entry = {'file': rel, 'line': lineno, 'enclosing': func, 'text': text.strip(),
                 'tracked': rel in tracked, 'classification': classification,
                 'on_oracle_path': False if classification else None}
        if classification is None:
            unclassified.append(f'{rel}:{lineno} ({func})')
        hits.append(entry)
    return {'search': "grep -rn --include=*.py 'max_energy_to_power' . (repository root, data/ included, "
                      'tracked and untracked)',
            'oracle_path': ORACLE_PATH, 'hits': hits, 'unclassified': unclassified,
            'n_hits': len(hits)}


def spec_pins():
    """Committed files that record the pre-edit sha256 of the ESS parameters file."""
    hits = _git(['grep', '-l', ESS_PARAMS_SHA256_BEFORE, '--', 'data', '*.py', '*.md'], check=False).splitlines()
    specs = [h for h in hits if os.path.basename(h).startswith(('campaign_spec_', 'frozen_'))]
    return {'searched': f'git grep -l {ESS_PARAMS_SHA256_BEFORE} -- data *.py *.md (tracked files at HEAD)',
            'files_recording_the_pre_edit_sha256': hits, 'campaign_or_frozen_specs_among_them': specs,
            'note': ('a file that RECORDS the input hash is provenance of a past computation, not a pin that '
                     'gates a future run; none of the campaign launchers or the harness check this file\'s hash '
                     '(the harness pins data/SRP1/SRP1_params.json, a different file)')}


# ======================================================================================================================
#  oracle-path build, digests
# ======================================================================================================================
def _canonical(obj):
    if isinstance(obj, dict):
        return '{' + ','.join(f'{repr(k)}:{_canonical(v)}' for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_canonical(v) for v in obj) + ']'
    return repr(obj)


def block_digest(model):
    import pyomo.environ as pe
    state = {'params': {}, 'vars': {}, 'constraints': {}, 'objectives': {}, 'blocks': {}}
    for comp in model.component_objects(pe.Param, active=None, descend_into=True):
        state['params'][comp.name] = {idx: pe.value(comp[idx], exception=False) for idx in comp}
    for comp in model.component_objects(pe.Var, active=None, descend_into=True):
        state['vars'][comp.name] = {idx: (v.value, v._lb, v._ub, v.fixed) for idx, v in comp.items()}
    for comp in model.component_objects(pe.Constraint, active=None, descend_into=True):
        state['constraints'][comp.name] = {
            idx: (c.active, None if c.lower is None else pe.value(c.lower),
                  None if c.upper is None else pe.value(c.upper)) for idx, c in comp.items()}
    for comp in model.component_objects(pe.Objective, active=None, descend_into=True):
        state['objectives'][comp.name] = {idx: o.active for idx, o in comp.items()}
    for blk in model.block_data_objects(active=None, descend_into=True):
        state['blocks'][blk.name or '<root>'] = blk.active
    counts = {k: sum(len(v) for v in state[k].values()) if k != 'blocks' else len(state[k]) for k in state}
    return hashlib.sha256(_canonical(state).encode()).hexdigest(), counts


class _RatioReadCounter:
    """Counts reads of `max_energy_to_power_ratio` on ONE shared-ESS parameters object by swapping its
    class for a subclass whose __getattribute__ counts that name (values unchanged)."""

    def __init__(self, params):
        self.params = params
        self.original_class = params.__class__
        self.reads = []
        counter = self

        class Counting(self.original_class):
            def __getattribute__(inner_self, name):
                if name == ATTR:
                    frame = sys._getframe(1)
                    counter.reads.append(f'{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} '
                                         f'in {frame.f_code.co_name}')
                return object.__getattribute__(inner_self, name)

        self.counting_class = Counting

    def __enter__(self):
        self.params.__class__ = self.counting_class
        return self

    def __exit__(self, *exc):
        self.params.__class__ = self.original_class
        return False


def oracle_path_build(G, srp, label, investment_map, run_tag, phase, scratch):
    report = {}
    eval_id = f'p515s46_w20_casefile_{phase}_{run_tag}_{label}'
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, label), report, investment_map=investment_map, eval_id=eval_id,
        num_max_iters_override=CAP, apply_rho=False, investment_year=2025)
    holder = {}
    spec_like = {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                                   'case_file_anderson_acceleration': dict(CASE_FILE_AA)},
                 'cap': CAP, 'required_consecutive_cycles': 10}
    digests = {}

    def network_stub(tag):
        def stub(model, *args, **kwargs):
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    digests[f'{tag}|{year}|{day}'] = block_digest(model[year][day])
                    results[year][day] = None
            return results
        return stub

    def esso_stub(models, *args, **kwargs):
        for node_id, model in models.items():
            digests[f'ESSO|node{node_id}'] = block_digest(model)
        return {node_id: None for node_id in models}

    ratio_loaded = sed.params.max_energy_to_power_ratio
    with _RatioReadCounter(sed.params) as counter:
        H._config_hook_factory(spec_like, holder, overrides={})(planning=planning, sed=sed, candidate=candidate,
                                                                 report=report)
        planning.transmission_network.optimize = network_stub('TSO')
        for node_id, dn in planning.distribution_networks.items():
            dn.optimize = network_stub(f'DSO{node_id}')
        sed.optimize = esso_stub
        consensus_vars, _dual_vars = srp.create_admm_variables(planning)
        _dso_models, _r_dso = srp.create_distribution_networks_models(
            planning.distribution_networks, consensus_vars, candidate['total_capacity'],
            parallel_execution=planning.parallel_execution)
        _tso_model, _r_tso = srp.create_transmission_network_model(planning, consensus_vars,
                                                                    candidate['total_capacity'])
        _esso_model, _r_esso = srp.create_shared_energy_storage_model(sed, consensus_vars, candidate['investment'])
        reads_during_build = list(counter.reads)
        _control = sed.params.max_energy_to_power_ratio  # positive control: the counter must see this read
        counter_works = len(counter.reads) == len(reads_during_build) + 1
    canonical = H.canonical_candidate(investment_map, investment_year=2025)
    return {'label': label, 'eval_id': eval_id, 'candidate_canonical': canonical,
            'candidate_key': H.candidate_key(canonical),
            'configuration_checks': holder.get('configuration_checks'),
            'max_energy_to_power_ratio_loaded': ratio_loaded,
            'ratio_reads_during_build': reads_during_build, 'n_ratio_reads_during_build': len(reads_during_build),
            'read_counter_positive_control_seen': counter_works,
            'n_blocks': len(digests),
            'blocks': {k: {'sha256': v[0], 'counts': v[1]} for k, v in sorted(digests.items())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('before', 'after'))
    parser.add_argument('--before-sha256', default=None)
    parser.add_argument('--out', default=OUT_REL)
    args = parser.parse_args()
    started = time.time()
    out_root = os.path.join(REPO, args.out, args.phase)
    if os.path.exists(out_root):
        raise SystemExit(f'output directory exists (write-once): {out_root}')
    ess_path = os.path.join(REPO, ESS_PARAMS_REL)
    ess_sha = _sha(ess_path)
    with open(ess_path) as handle:
        ess_json_text = handle.read()
    failures = []
    if args.phase == 'before' and ess_sha != ESS_PARAMS_SHA256_BEFORE:
        failures.append(f'phase before: {ESS_PARAMS_REL} sha256 {ess_sha} != pre-edit {ESS_PARAMS_SHA256_BEFORE}')
    before_record = None
    if args.phase == 'after':
        before_path = os.path.join(REPO, args.out, 'before', 'case_file_edit_before.json')
        if not args.before_sha256:
            failures.append('phase after needs --before-sha256')
        elif not os.path.isfile(before_path) or _sha(before_path) != args.before_sha256:
            failures.append(f'before record missing or sha256 != {args.before_sha256}: {before_path}')
        else:
            with open(before_path) as handle:
                before_record = json.load(handle)
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    os.makedirs(out_root)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    head = _git(['rev-parse', 'HEAD']).strip()
    _log(f'{STAGE}; phase {args.phase}; git HEAD {head}; {ESS_PARAMS_REL} sha256 {ess_sha}')

    import p515_g_g1_g4_admm_gates as G
    import p514_n_instrumented_cstar as N
    import shared_resources_planning as srp

    rd = readers()
    pins = spec_pins()
    _log(f"readers: {rd['n_hits']} hits, unclassified {rd['unclassified']}")
    for h in rd['hits']:
        _log(f"  {h['file']}:{h['line']} in {h['enclosing']} -> {h['classification']}")
    scratch = tempfile.mkdtemp(prefix='p515s46_w20_casefile_')
    instances = {'c_star': {n: (N.S_INV, N.E_INV) for n in H.ACTIVE_NODES},
                 'x0': {n: (0.0, 0.0) for n in H.ACTIVE_NODES}}
    builds = {}
    for label, inv_map in instances.items():
        builds[label] = oracle_path_build(G, srp, label, inv_map, run_tag, args.phase, scratch)
        _log(f"{label}: key {builds[label]['candidate_key'][:16]} blocks {builds[label]['n_blocks']} "
             f"ratio loaded {builds[label]['max_energy_to_power_ratio_loaded']} "
             f"ratio reads during build {builds[label]['n_ratio_reads_during_build']}")
    shutil.rmtree(scratch, ignore_errors=True)

    expected_value = VALUE_BEFORE if args.phase == 'before' else VALUE_AFTER
    checks = {
        'readers_all_classified': not rd['unclassified'],
        'readers_none_on_oracle_path': all(h['on_oracle_path'] is False for h in rd['hits']),
        'c_star_key_is_pinned_c_star': builds['c_star']['candidate_key'] == C_STAR_KEY_PIN,
        'loaded_ratio_is_file_value': all(b['max_energy_to_power_ratio_loaded'] == expected_value
                                          for b in builds.values()),
        'json_key_value_is_expected': json.loads(ess_json_text)[KEY] == expected_value,
        'no_ratio_read_during_oracle_build': all(b['n_ratio_reads_during_build'] == 0 for b in builds.values()),
        'read_counter_positive_control': all(b['read_counter_positive_control_seen'] for b in builds.values()),
        'blocks_48_network_plus_3_esso_per_instance': all(b['n_blocks'] == 51 for b in builds.values()),
        'digest_discriminates_c_star_from_x0_in_every_block': all(
            builds['c_star']['blocks'][k]['sha256'] != builds['x0']['blocks'][k]['sha256']
            for k in builds['c_star']['blocks']),
        'configuration_hook_passed': all(all((b['configuration_checks'] or {}).values()) and b['configuration_checks']
                                         for b in builds.values()),
    }
    comparison = None
    if args.phase == 'after':
        per_instance = {}
        for label in instances:
            a, b = before_record['builds'][label]['blocks'], builds[label]['blocks']
            diffs = sorted(k for k in set(a) | set(b) if (a.get(k) or {}).get('sha256') != (b.get(k) or {}).get('sha256'))
            per_instance[label] = {'n_blocks_before': len(a), 'n_blocks_after': len(b), 'n_differing': len(diffs),
                                   'differing': diffs}
        before_json = json.loads(before_record['ess_params_file']['text'])
        after_json = json.loads(ess_json_text)
        changed_keys = sorted(k for k in set(before_json) | set(after_json) if before_json.get(k) != after_json.get(k))
        comparison = {'per_instance': per_instance, 'json_keys_changed': changed_keys,
                      'before_value': before_json.get(KEY), 'after_value': after_json.get(KEY),
                      'before_record_sha256': args.before_sha256}
        checks['every_block_digest_identical_before_after'] = all(v['n_differing'] == 0 and v['n_blocks_before'] ==
                                                                  v['n_blocks_after'] for v in per_instance.values())
        checks['only_the_one_key_changed'] = changed_keys == [KEY]
        checks['value_10_to_4'] = comparison['before_value'] == VALUE_BEFORE and comparison['after_value'] == VALUE_AFTER
        checks['before_record_phase_is_before'] = before_record.get('phase') == 'before'
    guard_failures = GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    all_ok = all(checks.values())
    payload = {'stage': STAGE, 'phase': args.phase, 'authority': AUTHORITY, 'timestamp_utc': _utc(),
               'git_head_at_run': head, 'script': os.path.basename(__file__),
               'script_sha256': _sha(os.path.abspath(__file__)), 'harness_sha256': _sha(H.HARNESS_PATH),
               'ess_params_file': {'path': ESS_PARAMS_REL, 'sha256': ess_sha, 'text': ess_json_text,
                                   'git_status': _git(['status', '--porcelain', '--', ESS_PARAMS_REL]).strip()},
               'readers': rd, 'pins': pins, 'builds': builds, 'comparison': comparison,
               'digest_definition': ('sha256 of the canonical form of {every Param value (mutable and immutable), '
                                     'every Var (value, raw _lb, raw _ub, fixed), every Constraint (active, lower, '
                                     'upper), every Objective (active), every block active flag} of the block as '
                                     'handed to optimize() at ADMM initialization. Nothing is solved, so the TSO '
                                     'blocks carry the interface values of UNSOLVED DSO blocks (the initial '
                                     'consensus values) rather than solved ones -- the same in both phases'),
               'checks': checks, 'all_ok': all_ok,
               'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
               'wall_clock_s': time.time() - started}
    path = os.path.join(out_root, f'case_file_edit_{args.phase}.json')
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {os.path.relpath(path, REPO): _sha(path)}
    with open(os.path.join(out_root, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    GUARD.uninstall()
    for k, v in checks.items():
        _log(f'  {"OK  " if v else "FAIL"} {k}')
    if comparison:
        _log(f"comparison: {json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'differing'} for k, v in comparison['per_instance'].items()})} keys changed {comparison['json_keys_changed']}")
    _log(f'wrote {os.path.relpath(path, REPO)} sha256={_sha(path)}')
    _log(f'ALL_OK={all_ok} guard={dict(GUARD.counts)} wall={time.time() - started:.1f}s')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
