"""
P5.15 Addendum 25 item 2 -- Task 1: ZERO-SOLVE checks of the three follow-ups
the Planner folded into this task (`P5_15_S44_GATE_RULING.md`, 8ce0b872,
"Follow-ups").

  F1 (a) `block_deltas` sort key. The REAL, modified
     `p515_g_g1_g4_admm_gates.s34_capture_hooks` wrapper is exercised (never
     reimplemented) on synthetic two-cycle block decompositions whose deltas
     contain EXACT ties that straddle the top-10 cut (the two production
     decomposition functions and the wrapped function are monkeypatched with
     stand-ins, the technique of `p515_s44_alias_fix_check.py`), in one child
     interpreter per PYTHONHASHSEED value (hash randomization is fixed at
     interpreter start). Pass: every seed yields the same `block_deltas` and
     the same `objective_component_block_deltas`, equal to the name order.
     Sensitivity (reported, and required, so the test is not vacuous): the
     PRE-fix key (abs_delta alone, stable sort over the same set iteration)
     yields more than one order across the same seeds.
  F2 (b) legacy one-run lock vs campaign lock, on TEMPORARY paths (the real
     `.p515_g_gate.lock` / `.p515_s44_campaign.lock` are never touched):
     refuses while a campaign lock exists and leaves no legacy lock behind;
     acquires when none exists; refuses and backs out when a campaign lock
     appears between its check and its create (simulated by wrapping os.open);
     the mirror race on `p515_s44_campaign_harness.acquire_campaign_lock`;
     the two modules name the same campaign lock path; and (static, ast) no
     reference to `_acquire_exclusive_run_lock` exists in the campaign harness
     or in `run_admm_arm`, so campaign children can never take the legacy lock.
  F3 (c) `p515_s44_tie_classifier`: reproduces the committed tie analysis
     (`gate_tie_order_analysis/tie_order_analysis.json`) row for row on its
     committed inputs; applied to the s44_gate's 43 genuine sidecar diffs
     (recomputed with the gate's own comparator and classifier, BY IMPORT) it
     explains all 43; synthetic cases: identical / resort / straddle for both
     lists, and a changed value, a changed non-list field and a changed
     prefix are UNEXPLAINED.

Armed `SolveProfileGuard(permitted=())` in the parent and in every child,
verified 0. Write-once output root (argv suffix redirects it):
    data/SRP1/Results/P515S44/followups_check/

Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_followups_check.py \\
        > data/SRP1/Results/P515S44/followups_check_launch.log 2>&1
"""

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 follow-ups zero-solve checks').install()

PYTHON = sys.executable
_CHILD_MARKER = '--child-f1'
_SUFFIX = next(('_' + a.strip('_') for a in sys.argv[1:] if not a.startswith('--')), '')
OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', f'followups_check{_SUFFIX}')
HASH_SEEDS = ['0', '1', '2', '42', '100', '999', '12345', '777', '55']  # same list as p515_s44_alias_fix_check.py

TIE_ANALYSIS = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'gate_tie_order_analysis',
                            'tie_order_analysis.json')
D_SIDECAR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run', 'recourse_jump_sidecar_baseline.jsonl')
C_SIDECAR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_gate', 'evals',
                         '578636daa6d6360d_c_star', 'recourse_jump_sidecar_baseline.jsonl')
SIDECAR_NAME = 'recourse_jump_sidecar_baseline.jsonl'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ======================================================================================================================
#  F1 -- block_deltas determinism across PYTHONHASHSEED (child side)
# ======================================================================================================================
def _synthetic_blocks():
    """Two cycles. 14 blocks. Deltas: 3 distinct large ones, then an exact tie
    group of 9 blocks with delta 5.0 (positions 4-12 -> the top-10 cut at 10
    splits it), then 2 smaller ones. Keys mirror production's
    (agent, node_id, year, day) tuples, including a None node id for the TSO."""
    keys = [('tso', None, 2025, 'Spring'), ('esso', None, 2025, 'Spring')]
    keys += [('dso', n, 2025, d) for n in (5, 7, 9) for d in ('Spring', 'Summer', 'Autumn', 'Winter')]
    prev = {k: 100.0 for k in keys}
    cur = dict(prev)
    big = {keys[0]: 60.0, keys[1]: 40.0, keys[2]: 30.0}
    tie = keys[3:12]
    small = {keys[12]: 2.0, keys[13]: 1.0}
    for k, v in big.items():
        cur[k] = prev[k] + v
    for k in tie:
        cur[k] = prev[k] + 5.0
    for k, v in small.items():
        cur[k] = prev[k] + v
    return prev, cur


def _run_f1_child():
    guard = SolveProfileGuard(permitted=(), label='P5.15-S44 F1 child').install()
    import p515_g_g1_g4_admm_gates as G
    import shared_resources_planning as srp

    tmpdir = tempfile.mkdtemp(prefix='p515s44_f1_')
    rj = os.path.join(tmpdir, 'recourse_jump.jsonl')
    es = os.path.join(tmpdir, 'ess_stride.jsonl')
    prev, cur = _synthetic_blocks()
    rec_blocks = [prev, cur]
    obj_blocks = [
        {'blockA': {'economic_market_cost': 10.0, 'generation_cost': 10.0, 'flexibility_cost': 7.0, 'other_x': 3.0}},
        {'blockA': {'economic_market_cost': 50.0, 'generation_cost': 50.0, 'flexibility_cost': 47.0, 'other_x': 3.5}},
    ]
    st = {'i': 0}

    class _FakePlanning:
        active_distribution_network_nodes = []

    orig = (srp._get_operational_recourse_block_components, srp._get_operational_objective_component_blocks,
            srp.get_admm_boyd_residual_metrics)
    srp._get_operational_recourse_block_components = lambda pp, om: rec_blocks[st['i']]
    srp._get_operational_objective_component_blocks = lambda pp, om: obj_blocks[st['i']]
    srp.get_admm_boyd_residual_metrics = lambda *a, **k: 'STUB'
    try:
        with G.s34_capture_hooks(rj, es, stride=10 ** 9):
            for i in range(2):
                st['i'] = i
                srp.get_admm_boyd_residual_metrics(
                    planning_problem=_FakePlanning(), tso_model=None, dso_models=None, esso_model={},
                    consensus_vars={'ess': {'z': {'current': {}}}}, dual_vars=None, admm_parameters=None)
    finally:
        (srp._get_operational_recourse_block_components, srp._get_operational_objective_component_blocks,
         srp.get_admm_boyd_residual_metrics) = orig
    rows = _load_jsonl(rj)
    bd = rows[1]['block_deltas']
    od = rows[1]['objective_component_block_deltas']
    # Sensitivity: the PRE-fix key on the SAME set iteration this process produces.
    pre_fix = []
    for key in set(cur) | set(prev):
        agent, node_id, year, day = key
        pre_fix.append({'agent': agent, 'node_id': node_id, 'year': str(year), 'day': str(day),
                        'abs_delta': abs(cur.get(key, 0.0) - prev.get(key, 0.0))})
    pre_fix.sort(key=lambda e: e['abs_delta'], reverse=True)
    failures = guard.verify(0)
    guard.uninstall()
    print(json.dumps({
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        'block_deltas_order': [[e['agent'], e['node_id'], e['year'], e['day'], e['abs_delta']] for e in bd],
        'objective_component_block_deltas_order': [[e['block_key'], e['component'], e['abs_delta']] for e in od],
        'pre_fix_key_top10_order': [[e['agent'], e['node_id'], e['year'], e['day']] for e in pre_fix[:10]],
        'guard_counts': dict(guard.counts), 'guard_verify_0_failures': failures,
    }, default=str))


def f1_block_deltas_determinism():
    per_seed = []
    for seed in HASH_SEEDS:
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = seed
        proc = subprocess.run([PYTHON, os.path.abspath(__file__), _CHILD_MARKER], env=env, capture_output=True,
                              text=True, cwd=REPO)
        if proc.returncode != 0:
            raise RuntimeError(f'F1 child failed (seed {seed}): {proc.stderr[-3000:]}')
        per_seed.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    bd_orders = {json.dumps(r['block_deltas_order']) for r in per_seed}
    od_orders = {json.dumps(r['objective_component_block_deltas_order']) for r in per_seed}
    pre_orders = {json.dumps(r['pre_fix_key_top10_order']) for r in per_seed}
    first = per_seed[0]['block_deltas_order']
    # Expected order, stated independently: by -abs_delta, then (str(agent), str(node_id), year, day).
    prev, cur = _synthetic_blocks()
    exp = sorted(((-(abs(cur[k] - prev[k])), str(k[0]), str(k[1]), str(k[2]), str(k[3])) for k in cur))[:10]
    expected = [[t[1], t[2], t[3], t[4], -t[0]] for t in exp]
    got_named = [[str(a), str(n), y, d, ad] for a, n, y, d, ad in first]
    tie_group_straddles_cut = sum(1 for t in exp if -t[0] == 5.0) < 9
    checks = {
        'children_ran_all_seeds': len(per_seed) == len(HASH_SEEDS),
        'children_zero_solves': all(not r['guard_verify_0_failures'] for r in per_seed),
        'block_deltas_one_order_across_seeds': len(bd_orders) == 1,
        'block_deltas_order_is_name_order': got_named == expected,
        'objective_deltas_one_order_across_seeds': len(od_orders) == 1,
        'synthetic_tie_group_straddles_top10_cut': tie_group_straddles_cut,
        'sensitivity_pre_fix_key_varies_across_seeds': len(pre_orders) > 1,
    }
    return {'checks': checks, 'hash_seeds': HASH_SEEDS, 'per_seed': per_seed,
            'expected_block_deltas_order': expected,
            'n_distinct_orders': {'block_deltas': len(bd_orders), 'objective_component_block_deltas': len(od_orders),
                                  'pre_fix_key_top10': len(pre_orders)}}


# ======================================================================================================================
#  F2 -- legacy one-run lock vs campaign lock
# ======================================================================================================================
def _names_in(source):
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _try(fn):
    try:
        fn()
    except SystemExit as error:
        return {'raised': 'SystemExit', 'message': str(error)[:300]}
    except BaseException as error:  # noqa: BLE001
        return {'raised': type(error).__name__, 'message': str(error)[:300]}
    return {'raised': None}


def f2_locks():
    import p515_g_g1_g4_admm_gates as G
    import p515_s44_campaign_harness as H
    tmp = os.path.join(OUT, 'f2_tmp_locks')
    os.makedirs(tmp)
    legacy = os.path.join(tmp, 'legacy.lock')
    campaign = os.path.join(tmp, 'campaign.lock')
    real_legacy = os.path.join(REPO, '.p515_g_gate.lock')
    real_campaign = os.path.join(REPO, '.p515_s44_campaign.lock')
    real_before = {'legacy': os.path.exists(real_legacy), 'campaign': os.path.exists(real_campaign)}
    results = {}

    # 1. campaign lock present -> refuse, no legacy lock left behind
    with open(campaign, 'w') as handle:
        handle.write('{"pid": 1, "campaign_id": "f2"}')
    results['refuse_while_campaign_lock'] = _try(
        lambda: G._acquire_exclusive_run_lock(lock_path=legacy, campaign_lock_path=campaign))
    results['refuse_while_campaign_lock']['legacy_lock_left'] = os.path.exists(legacy)
    os.remove(campaign)

    # 2. no campaign lock -> acquires
    results['acquire_without_campaign_lock'] = _try(
        lambda: G._acquire_exclusive_run_lock(lock_path=legacy, campaign_lock_path=campaign))
    results['acquire_without_campaign_lock']['legacy_lock_created'] = os.path.exists(legacy)
    results['second_acquire_refused'] = _try(
        lambda: G._acquire_exclusive_run_lock(lock_path=legacy, campaign_lock_path=campaign))
    os.remove(legacy)

    # 3. race: the campaign lock appears between the check and the O_EXCL create
    real_open = os.open

    def racing_open(path, flags, *args, **kwargs):
        if path == legacy and not os.path.exists(campaign):
            with open(campaign, 'w') as handle:
                handle.write('{"pid": 1, "campaign_id": "race"}')
        return real_open(path, flags, *args, **kwargs)

    os.open = racing_open
    try:
        results['race_campaign_appears'] = _try(
            lambda: G._acquire_exclusive_run_lock(lock_path=legacy, campaign_lock_path=campaign))
    finally:
        os.open = real_open
    results['race_campaign_appears']['legacy_lock_left'] = os.path.exists(legacy)
    os.remove(campaign)

    # 4. mirror race on the campaign side: the legacy lock appears after the campaign's first check
    def racing_open_2(path, flags, *args, **kwargs):
        if path == campaign and not os.path.exists(legacy):
            with open(legacy, 'w') as handle:
                handle.write('legacy race')
        return real_open(path, flags, *args, **kwargs)

    os.open = racing_open_2
    try:
        results['mirror_race_legacy_appears'] = _try(
            lambda: H.acquire_campaign_lock('f2', 'abc', lock_path=campaign, legacy_lock_path=legacy))
    finally:
        os.open = real_open
    results['mirror_race_legacy_appears']['campaign_lock_left'] = os.path.exists(campaign)
    os.remove(legacy)

    harness_names = _names_in(inspect.getsource(H))
    run_admm_arm_names = _names_in(inspect.getsource(G.run_admm_arm))
    real_after = {'legacy': os.path.exists(real_legacy), 'campaign': os.path.exists(real_campaign)}
    leftovers = sorted(os.listdir(tmp))
    checks = {
        'refuses_while_campaign_lock_exists': results['refuse_while_campaign_lock']['raised'] == 'SystemExit'
        and 'campaign lock exists' in results['refuse_while_campaign_lock']['message'],
        'refusal_leaves_no_legacy_lock': results['refuse_while_campaign_lock']['legacy_lock_left'] is False,
        'acquires_without_campaign_lock': results['acquire_without_campaign_lock']['raised'] is None
        and results['acquire_without_campaign_lock']['legacy_lock_created'] is True,
        'legacy_lock_still_exclusive': results['second_acquire_refused']['raised'] == 'SystemExit',
        'race_backs_out_and_refuses': results['race_campaign_appears']['raised'] == 'SystemExit'
        and results['race_campaign_appears']['legacy_lock_left'] is False,
        'mirror_race_backs_out_and_refuses': results['mirror_race_legacy_appears']['raised'] == 'SystemExit'
        and results['mirror_race_legacy_appears']['campaign_lock_left'] is False,
        'same_campaign_lock_path_in_both_modules': G.CAMPAIGN_LOCK_PATH == H.CAMPAIGN_LOCK_PATH,
        'campaign_harness_never_references_legacy_lock_fn': '_acquire_exclusive_run_lock' not in harness_names,
        'run_admm_arm_never_references_legacy_lock_fn': '_acquire_exclusive_run_lock' not in run_admm_arm_names,
        'real_repo_locks_untouched': real_before == real_after,
        'no_temporary_lock_left': leftovers == [],
    }
    os.rmdir(tmp)
    return {'checks': checks, 'scenarios': results, 'real_repo_lock_state': {'before': real_before,
                                                                           'after': real_after}}


# ======================================================================================================================
#  F3 -- the re-sort-and-straddle classifier
# ======================================================================================================================
def f3_classifier():
    import p515_s44_tie_classifier as TC
    import p515_s40_clone_capture_preflight as CP
    import p515_s43_aa_flagoff_gate as FG

    with open(TIE_ANALYSIS) as handle:
        committed = json.load(handle)
    a, b = _load_jsonl(D_SIDECAR), _load_jsonl(C_SIDECAR)
    inputs_match = (_sha(D_SIDECAR) == committed['inputs']['D_sidecar_sha256']
                    and _sha(C_SIDECAR) == committed['inputs']['this_run_sidecar_sha256'])
    committed_rows = {r['row']: r for r in committed['non_identical_rows']}
    field = 'objective_component_block_deltas'
    mismatches, counts = [], {}
    for i, (x, y) in enumerate(zip(a, b)):
        res = TC.classify_row(x, y, fields=(field,))
        row_cls = res['class']
        list_cls = res['per_field'].get(field, {}).get('class', row_cls)
        cls = row_cls if row_cls.startswith('UNEXPLAINED_other') else list_cls
        counts[cls] = counts.get(cls, 0) + 1
        want = committed_rows.get(i)
        if want is None:
            if cls != 'identical':
                mismatches.append({'row': i, 'mine': cls, 'committed': 'identical'})
            continue
        if cls != want['class']:
            mismatches.append({'row': i, 'mine': cls, 'committed': want['class']})
            continue
        if cls == 'straddle':
            d_mine = res['per_field'][field]['detail']
            d_comm = want['detail']
            if (d_mine['k0'] != d_comm['k0']
                    or d_mine['straddling_tie_abs_delta'] != d_comm['straddling_tie_abs_delta']
                    or d_mine['reference_members_at_cut'] != d_comm['D_members_at_cut']
                    or d_mine['new_members_at_cut'] != d_comm['this_run_members_at_cut']):
                mismatches.append({'row': i, 'mine_detail': d_mine, 'committed_detail': d_comm})

    # the gate use case: the s44_gate's genuine sidecar diffs, recomputed with the gate's own conventions
    raw = CP._diff(a, b, SIDECAR_NAME)
    _prov, _aa, known_tie, genuine = FG._classify_diffs(raw)
    tie_order, still_genuine, row_classes = TC.reclassify_sidecar_diffs(genuine, a, b, SIDECAR_NAME)

    # synthetic cases
    def od(block, comp, ad, prev=1.0):
        return {'block_key': block, 'component': comp, 'previous': prev, 'current': prev + ad,
                'delta': ad, 'abs_delta': ad}

    def bdl(agent, node, day, ad):
        return {'agent': agent, 'node_id': node, 'year': '2025', 'day': day, 'previous': 0.0, 'current': ad,
                'delta': ad, 'abs_delta': ad}

    new_obj = sorted([od('B1', 'x', 9.0), od('B2', 'economic_market_cost', 5.0), od('B2', 'generation_cost', 5.0),
                      od('B3', 'y', 1.0)], key=TC._obj_key)
    ref_obj_resort = [new_obj[0], new_obj[2], new_obj[1], new_obj[3]]
    # straddle: 3-entry list whose last tie group (abs 5.0, block B2) is cut; ref kept a different member
    new_cut = sorted([od('B1', 'x', 9.0), od('B2', 'economic_market_cost', 5.0), od('B2', 'flexibility_cost', 5.0)],
                     key=TC._obj_key)
    ref_cut = [od('B1', 'x', 9.0), od('B2', 'generation_cost', 5.0), od('B2', 'flexibility_cost', 5.0)]
    changed_value = [dict(e) for e in new_obj]
    changed_value[3] = od('B3', 'y', 1.5)
    changed_prefix = [od('B1', 'z', 9.0)] + new_obj[1:]
    new_blk = sorted([bdl('tso', None, 'Spring', 7.0), bdl('dso', 5, 'Spring', 3.0), bdl('dso', 7, 'Spring', 3.0),
                      bdl('dso', 9, 'Spring', 3.0)], key=TC._block_key)
    ref_blk_resort = [new_blk[0], new_blk[3], new_blk[1], new_blk[2]]
    new_blk_cut = new_blk[:3]
    ref_blk_cut = [new_blk[0], new_blk[3], new_blk[1]]
    base_row = {'cycle': 5, 'error': None, 'block_total_current': 1.0, 'block_total_previous': 0.5}
    synthetic = {
        'obj_identical': TC.classify_top_k_list(new_obj, list(new_obj), field)[0],
        'obj_resort': TC.classify_top_k_list(ref_obj_resort, new_obj, field)[0],
        'obj_straddle': TC.classify_top_k_list(ref_cut, new_cut, field)[0],
        'obj_changed_value': TC.classify_top_k_list(changed_value, new_obj, field)[0],
        'obj_changed_prefix': TC.classify_top_k_list(changed_prefix, new_obj, field)[0],
        'blk_resort': TC.classify_top_k_list(ref_blk_resort, new_blk, 'block_deltas')[0],
        'blk_straddle': TC.classify_top_k_list(ref_blk_cut, new_blk_cut, 'block_deltas')[0],
        'row_both_lists_reordered': TC.classify_row(
            dict(base_row, block_deltas=ref_blk_resort, objective_component_block_deltas=ref_obj_resort),
            dict(base_row, block_deltas=new_blk, objective_component_block_deltas=new_obj))['class'],
        'row_other_field_changed': TC.classify_row(
            dict(base_row, block_deltas=new_blk, objective_component_block_deltas=new_obj),
            dict(base_row, block_total_current=1.25, block_deltas=new_blk,
                 objective_component_block_deltas=new_obj))['class'],
    }
    expected_synth = {
        'obj_identical': 'identical', 'obj_resort': 'resort', 'obj_straddle': 'straddle',
        'obj_changed_value': 'UNEXPLAINED', 'obj_changed_prefix': 'UNEXPLAINED', 'blk_resort': 'resort',
        'blk_straddle': 'straddle', 'row_both_lists_reordered': 'explained',
        'row_other_field_changed': 'UNEXPLAINED_other_fields_differ',
    }
    synth_ok = {k: (synthetic[k].startswith(v) if v == 'UNEXPLAINED' else synthetic[k] == v)
                for k, v in expected_synth.items()}
    checks = {
        'committed_inputs_hash_match': inputs_match,
        'reproduces_committed_analysis_row_for_row': not mismatches and len(a) == len(b) == committed['n_rows']['D'],
        'reproduces_committed_class_counts': counts == committed['row_class_counts'],
        'gate_genuine_diffs_count_is_43': len(genuine) == 43,
        'all_43_explained_as_tie_order': len(tie_order) == 43 and still_genuine == [],
        'synthetic_cases_as_expected': all(synth_ok.values()),
    }
    return {'checks': checks, 'row_class_counts': counts, 'committed_row_class_counts': committed['row_class_counts'],
            'mismatches': mismatches, 'gate_use_case': {
                'n_raw_diffs': len(raw), 'n_known_tie_break_alias_pair': len(known_tie),
                'n_genuine_before': len(genuine), 'n_tie_order': len(tie_order),
                'n_genuine_after': len(still_genuine), 'row_classes': row_classes},
            'synthetic': synthetic, 'synthetic_expected': expected_synth,
            'inputs': {'tie_analysis': os.path.relpath(TIE_ANALYSIS, REPO), 'tie_analysis_sha256': _sha(TIE_ANALYSIS),
                       'D_sidecar': os.path.relpath(D_SIDECAR, REPO), 'this_run_sidecar': os.path.relpath(C_SIDECAR, REPO)}}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'output root already exists (write-once): {OUT}')
    os.makedirs(OUT)
    started = time.time()
    results = {'stage': 'P5.15 Addendum 25 item 2 -- Task 1 follow-ups: zero-solve checks',
               'authority': ['P5_15_S44_GATE_RULING.md (8ce0b872) Follow-ups', 'PLANNER_BRIEF_2026-09-13.md Addendum 25'],
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                          text=True).stdout.strip(),
               'script_sha256': _sha(os.path.abspath(__file__))}
    all_pass = True
    for name, fn in (('F1_block_deltas_determinism', f1_block_deltas_determinism), ('F2_locks', f2_locks),
                     ('F3_tie_classifier', f3_classifier)):
        print(f'[S44-FUP] ===== {name} =====', flush=True)
        try:
            out = fn()
            ok = all(out['checks'].values())
        except BaseException as error:  # noqa: BLE001
            out = {'exception': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            ok = False
        out['pass'] = ok
        all_pass = all_pass and ok
        results[name] = out
        print(f'[S44-FUP] {name}: pass={ok} checks={out.get("checks", out.get("exception"))}', flush=True)
    guard_failures = GUARD.verify(0)
    results['solve_profile_guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures}
    results['all_pass'] = all_pass and not guard_failures
    results['wall_clock_s'] = time.time() - started
    out_path = os.path.join(OUT, 'followups_check.json')
    with open(out_path, 'w') as handle:
        json.dump(results, handle, indent=1, default=str)
    with open(os.path.join(OUT, 'manifest_sha256.json'), 'w') as handle:
        json.dump({os.path.relpath(out_path, REPO): _sha(out_path)}, handle, indent=2)
    GUARD.uninstall()
    print(f'[S44-FUP] ALL PASS={results["all_pass"]} guard={GUARD.counts} failures={guard_failures}')
    if not results['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == _CHILD_MARKER:
        _run_f1_child()
    else:
        main()
