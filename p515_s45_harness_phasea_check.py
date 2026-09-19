"""
P5.15 Addendum 27, task W5 ("Phase A harness fixes") -- ZERO-SOLVE checks for
the three pre-A1 changes to `p515_s44_campaign_harness.py`, plus a write
trace of the child's pre-solve path (the concurrency-write question).
Armed `SolveProfileGuard(permitted=())` for the whole script (installed before
any model import), `verify(0)` at the end.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27; P5_15_S45_REVERIFY_RULING.md
consequence 2; frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`.

Checks (production / harness functions only, never re-implemented):
  K1 BAR. For EVERY tracked `evaluation_record.json` under P515S44/ and
     P515S45/: if its eval dir holds `per_cycle_record.jsonl`, the new bar
     (`H._max_step_last_n`, gross steps) is recomputed from it and compared
     with the recorded `bar.value` (tolerance 1e-6 EUR); the reported net
     value (`H._max_net_recourse_step_last_n`) must reproduce the recorded
     bar exactly. Records without a per-cycle record (stubs, fakes) are
     listed with the reason. Additionally, for every such record whose
     `g_s39_D.json` / component levels / boyd terminal are on disk, the whole
     record is rebuilt by `H.build_evaluation_record` and every field except
     `bar` / `bar_net_recourse_step_reported` must equal the committed record.
  K2 KEYS AND REFERENCES. Every tracked campaign spec under P515S44/ and
     P515S45/: candidate key, eval key, eval dir and working-dir ids recomputed
     == recorded (W1-style); every entry's post-certification request re-
     resolved by the NEW `resolve_post_certification` == the recorded
     resolution.
  K3 D-NESS OF A REFERENCE. Committed D records accepted, AA-override records
     refused (new == HEAD behaviour); the committed case-file-AA record
     (S45 reverify) ACCEPTED by the HEAD harness and REFUSED by the new one;
     declared fixtures with effective AA off accepted. Case-file-AA fixture
     specs: hull polish + persist WITHOUT a reference freeze; a D reference
     freezes; the case-file-AA reference is refused. `run_post_certification`
     (REAL; fakes only for persist and polish, as C10) on the certified S45
     trajectory without a reference: evaluated, gates (b)/(c) None, gate (d)
     present, persist before polish.
  K4 ERROR-RECORD SCHEMA. `main_child` IN-PROCESS on non-stub fixture specs
     (temporary lock files in the temporary fixture root; the real locks are never touched),
     with failures injected at three points: (A) before anything (both new
     fields None); (B) the REAL "working dir id already used" refusal after
     the case-file hash (sha set, AA None); (C) after the REAL configuration
     hook on a REAL unsolved C* planning object (`_construct_arm_planning`),
     i.e. a fake `run_admm_arm` that raises after the hook (both set). (D) the
     parent-synthesized record carries both keys as None. During (C) an audit
     hook records every file opened for writing (the child's pre-solve path,
     including the baseline load).

Output (new, write-once): data/SRP1/Results/P515S45/harness_phaseA_check/

Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_harness_phasea_check.py \\
        > data/SRP1/Results/P515S45/harness_phaseA_check_launch.log 2>&1
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S45 W5 harness Phase A check (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S45', 'harness_phaseA_check')
BAR_TOL_EUR = 1e-6
C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
RECORD_PATHSPECS = ('data/SRP1/Results/P515S44/**/evaluation_record.json',
                    'data/SRP1/Results/P515S45/**/evaluation_record.json')
SPEC_PATHSPECS = ('data/SRP1/Results/P515S44/*campaign_spec_*.json',
                  'data/SRP1/Results/P515S45/*campaign_spec_*.json')
_P44 = os.path.join('data', 'SRP1', 'Results', 'P515S44')
D_REFS = {  # committed D evaluations (no overrides, no declaration)
    'gate_c_star': os.path.join(_P44, 'campaign_s44_gate', 'evals', '578636daa6d6360d_c_star'),
    'gate_paper_plan': os.path.join(_P44, 'campaign_s44_gate', 'evals', 'd1a02e67107d0bac_paper_plan'),
    'gate_node7_empty': os.path.join(_P44, 'campaign_s44_gate', 'evals', 'e30704e6e4dd3765_node7_empty'),
    'aa_variant_two_c_star_d': os.path.join(_P44, 'campaign_s44_aa_variant', 'evals', '4e53fa5560bbfa10_two_c_star_d'),
}
AA_OVERRIDE_REFS = {
    'aa_variant_c_star_aa_keep_memory': os.path.join(_P44, 'campaign_s44_aa_variant', 'evals',
                                                     '837fc982565dbba3_c_star_aa_keep_memory'),
    'selection_paper_plan_aa': os.path.join(_P44, 'campaign_s44_selection_aa', 'evals',
                                            '8e48f3ec8993a283_paper_plan_aa_keep_memory'),
    'selection_node7_empty_aa': os.path.join(_P44, 'campaign_s44_selection_aa', 'evals',
                                             '0796b6dceeb0f95f_node7_empty_aa_keep_memory'),
    'selection_two_c_star_aa': os.path.join(_P44, 'campaign_s44_selection_aa', 'evals',
                                            'a6ca6c94e033e460_two_c_star_aa_keep_memory'),
}
CASE_FILE_AA_REF = os.path.join('data', 'SRP1', 'Results', 'P515S45', 'reverify_aa_c_star', 'evals',
                                '3e741dac72c9e1bc_c_star_aa_case_file')
AA_RUN_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S43', 'aa_run')  # C10's committed polish fixture
# Every fixture (specs, fixture references, in-process child campaigns, their work dirs) lives in a TEMPORARY
# directory outside the repository, so no fixture spec / record ever falls under the committed pathspecs that
# K1/K2 (and later W1-style recomputes) enumerate; the evidence is copied into check.json.
FIX = {'root': None}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _load(rel_or_abs):
    with open(os.path.join(REPO, rel_or_abs)) as handle:
        return json.load(handle)


def _norm(x):
    return json.loads(json.dumps(x, default=str, sort_keys=True))


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _head_harness_module():
    """`p515_s44_campaign_harness.py` exactly as at HEAD, as a separate module object."""
    source = _git(['show', 'HEAD:p515_s44_campaign_harness.py'])
    module = types.ModuleType('p515_s44_campaign_harness_head')
    module.__file__ = H.HARNESS_PATH
    exec(compile(source, 'HEAD:p515_s44_campaign_harness.py', 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ======================================================================================================================
#  K1 -- the bar
# ======================================================================================================================
def k1_bar():
    tracked = sorted(p for p in _git(['ls-files', '--'] + list(RECORD_PATHSPECS)).split('\n') if p)
    rows_out, skipped, rebuild = [], [], {}
    for rel in tracked:
        rec = _load(rel)
        d = os.path.dirname(rel)
        pcr = os.path.join(REPO, d, 'per_cycle_record.jsonl')
        if not os.path.isfile(pcr):
            skipped.append({'record': rel, 'status': rec.get('status'), 'stub': bool(rec.get('stub')),
                            'reason': 'no per_cycle_record.jsonl in the eval dir (stub / fake reference fixture)'})
            continue
        rows = _read_jsonl(pcr)
        new_bar = H._max_step_last_n(rows)
        net = H._max_net_recourse_step_last_n(rows)
        recorded = (rec.get('bar') or {}).get('value')
        diff = (new_bar['value'] - recorded) if (new_bar['value'] is not None and recorded is not None) else None
        per_row_gap = [abs(w['gross_step_abs'] - n_['objective_change_abs'])
                       for w, n_ in zip(new_bar['window'], net['window'])
                       if w['gross_step_abs'] is not None and n_['objective_change_abs'] is not None]
        salvage = [abs(r['terminal_salvage_value']) for r in rows if r.get('terminal_salvage_value') is not None]
        rows_out.append({
            'record': rel, 'candidate_key16': str(rec.get('candidate_key'))[:16], 'eval_key16': str(rec.get('eval_key'))[:16],
            'label': rec.get('candidate_label'), 'status': rec.get('status'),
            'investment_year': (rec.get('candidate_canonical') or {}).get('investment_year'),
            'cycles_run': len(rows), 'recorded_bar': recorded, 'new_bar_gross': new_bar['value'],
            'new_minus_recorded': diff, 'abs_diff': abs(diff) if diff is not None else None,
            'within_1e-6': diff is not None and abs(diff) <= BAR_TOL_EUR,
            'reported_net_bar_reproduces_recorded_exactly': net['value'] == recorded,
            'new_bar_n_steps_available': new_bar['n_steps_available'],
            'first_window_row': new_bar['window'][0] if new_bar['window'] else None,
            'trajectory_cycles_contiguous_from_1': new_bar['trajectory_cycles_contiguous_from_1'],
            'max_per_row_gross_vs_net_step_gap_in_window': max(per_row_gap) if per_row_gap else None,
            'max_abs_terminal_salvage_over_trajectory': max(salvage) if salvage else None,
            'per_cycle_record_sha256': H.sha256_file(pcr),
        })
        # whole-record rebuild with the NEW builder (every field except the bar pair must be unchanged)
        g_path = os.path.join(REPO, d, 'g_s39_D.json')
        cl_path = os.path.join(REPO, d, 'component_levels_terminal.json')
        bt_path = os.path.join(REPO, d, 'boyd_terminal.json')
        spec_rel = rec.get('campaign_spec_path')
        if not all(os.path.isfile(p) for p in (g_path, cl_path, bt_path)) or not spec_rel:
            rebuild[rel] = {'rebuilt': False, 'reason': 'artifacts not on disk'}
            continue
        spec = _load(spec_rel)
        entry = next(e for e in spec['candidates'] if H._entry_eval_key(e) == rec.get('eval_key', rec['candidate_key']))
        report = _load(g_path)
        caps = {n: v.get('published_available_capacity_terminal') for n, v in (rec.get('storage_per_node') or {}).items()}
        rebuilt = H.build_evaluation_record(
            spec=spec, spec_path=os.path.join(REPO, spec_rel), spec_sha256=rec['campaign_spec_sha256'], entry=entry,
            report=report, component_levels=_load(cl_path),
            floor_terminal=_load(bt_path).get('soh_floor_multiplier_and_efc_per_cohort_year_terminal'),
            published_caps=caps, peak_rss=rec.get('peak_rss'), wall=rec.get('wall_time_s'),
            eval_dir=os.path.join(REPO, d))
        rb, cm = _norm(rebuilt), _norm(rec)
        differing = sorted(k for k in rb if k not in ('bar', 'bar_net_recourse_step_reported') and rb[k] != cm.get(k))
        # pre-existing, not W5: s44_gate records were written under RECORD_SCHEMA v1 (bumped to v2 in 8f2b2736)
        schema_v1_only = (differing == ['schema'] and cm.get('schema') == 'p515_s44_evaluation_record_v1'
                          and rb.get('schema') == 'p515_s44_evaluation_record_v2')
        net_old = rb['bar_net_recourse_step_reported']
        rebuild[rel] = {
            'rebuilt': True, 'fields_compared': len([k for k in rb if k not in ('bar', 'bar_net_recourse_step_reported')]),
            'differing_fields_other_than_bar': differing,
            'only_difference_is_pre_existing_schema_v1_label': schema_v1_only,
            'net_reported_equals_recorded_bar_value_and_window': (
                net_old['value'] == cm['bar']['value'] and net_old['window'] == cm['bar']['window']
                and net_old['n_cycles_in_window'] == cm['bar']['n_cycles_in_window']),
            'rebuilt_bar_from_full_trajectory_equals_per_cycle_recompute': rb['bar']['value'] == rows_out[-1]['new_bar_gross'],
        }
    diffs = [r['abs_diff'] for r in rows_out if r['abs_diff'] is not None]
    checks = {
        'records_tracked_found': len(tracked) == 27,
        'real_records_with_per_cycle_record_11': len(rows_out) == 11,
        'every_new_bar_within_1e-6_of_recorded': all(r['within_1e-6'] for r in rows_out) and bool(rows_out),
        'every_reported_net_bar_reproduces_recorded_exactly': all(r['reported_net_bar_reproduces_recorded_exactly']
                                                                  for r in rows_out),
        'every_record_2025': all(r['investment_year'] == 2025 for r in rows_out),
        'every_record_rebuilt_unchanged_except_bar': all(v.get('rebuilt') and (
            not v['differing_fields_other_than_bar'] or v['only_difference_is_pre_existing_schema_v1_label'])
                                                         and v['net_reported_equals_recorded_bar_value_and_window']
                                                         and v['rebuilt_bar_from_full_trajectory_equals_per_cycle_recompute']
                                                         for v in rebuild.values()) and len(rebuild) == len(rows_out),
    }
    return checks, {'per_record': rows_out, 'max_abs_diff_eur': max(diffs) if diffs else None,
                    'not_within_1e-6': [r['record'] for r in rows_out if not r['within_1e-6']],
                    'skipped_no_per_cycle_record': skipped, 'whole_record_rebuild': rebuild,
                    'tolerance_eur': BAR_TOL_EUR}


# ======================================================================================================================
#  K2 -- keys and reference resolutions over every committed spec
# ======================================================================================================================
def k2_keys_and_references():
    committed = sorted(p for p in _git(['ls-files', '--'] + list(SPEC_PATHSPECS)).split('\n') if p)
    per_spec, all_ok, ref_rows = {}, True, []
    for rel in committed:
        s = _load(rel)
        cfg = s.get('configuration') or {}
        decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        rows = []
        for e in s['candidates']:
            ov = H.validate_overrides(e['overrides'] if 'overrides' in e else (cfg.get('overrides') or {}))
            recorded_ekey = e.get('eval_key', e['key'])
            new_key = H.candidate_key(e['canonical'])
            new_ekey = H.evaluation_key(new_key, ov, case_file_aa=decl)
            ok = {'candidate_key': new_key == e['key'], 'eval_key': new_ekey == recorded_ekey}
            if 'eval_dir' in e:
                ok['eval_dir'] = H.eval_dir_name(new_ekey, e['label']) == e['eval_dir']
            if 'working_dir_ids' in e:
                ok['working_dir_ids'] = H.eval_ids(s['campaign_id'], new_ekey) == e['working_dir_ids']
            if decl is not None:
                ok['effective_aa'] = e.get('effective_anderson_acceleration') == H.effective_anderson_acceleration(decl, ov)
            rows.append({'label': e['label'], 'recorded_eval_key16': recorded_ekey[:16],
                         'recomputed_eval_key16': new_ekey[:16], 'ok': ok})
            all_ok = all_ok and all(ok.values())
            pc = e.get('post_certification')
            if pc:
                req = {'persist_certified_models': pc['persist_certified_models'], 'hull_polish': pc['hull_polish']}
                if pc.get('reference'):
                    req['reference'] = {'eval_dir': pc['reference']['eval_dir']}
                try:
                    again = H.resolve_post_certification(req, e['key'])
                    same = _norm(again) == _norm(pc)
                    err = None
                except Exception as error:  # noqa: BLE001
                    same, err = False, f'{type(error).__name__}: {error}'
                ref_rows.append({'spec': rel, 'label': e['label'],
                                 'reference_eval_dir': (pc.get('reference') or {}).get('eval_dir'),
                                 'new_resolution_equals_recorded': same, 'error': err})
        per_spec[rel] = {'campaign_id': s.get('campaign_id'), 'declares_case_file_aa': decl is not None,
                         'n_entries': len(rows), 'entries': rows}
    checks = {
        'committed_specs_found_23': len(committed) == 23,
        'every_key_eval_key_dir_id_unchanged': all_ok and bool(committed),
        'every_recorded_post_certification_resolution_reproduced': all(r['new_resolution_equals_recorded']
                                                                       for r in ref_rows),
    }
    return checks, {'n_specs': len(committed), 'n_entries': sum(v['n_entries'] for v in per_spec.values()),
                    'per_spec': per_spec, 'post_certification_re_resolution': ref_rows}


# ======================================================================================================================
#  K3 -- D-ness of a post-certification reference; hull polish without a reference
# ======================================================================================================================
def _try_resolve(module, ref_dir, cand_key):
    try:
        res = module.resolve_post_certification({'reference': {'eval_dir': ref_dir}}, cand_key)
        return {'accepted': True, 'configuration_overrides': res['reference']['configuration_overrides'],
                'effective_anderson_acceleration': res['reference'].get('effective_anderson_acceleration')}
    except ValueError as error:
        return {'accepted': False, 'message': str(error)}


def _make_fixture_reference(name, base_dir, record_edit):
    """A copy of a committed reference (record + component levels) with an edited record -- a fixture."""
    d = os.path.join(FIX['root'], 'k3_fixture_refs', name)
    os.makedirs(d)
    shutil.copy(os.path.join(REPO, base_dir, 'component_levels_terminal.json'), d)
    rec = _load(os.path.join(base_dir, 'evaluation_record.json'))
    record_edit(rec)
    with open(os.path.join(d, 'evaluation_record.json'), 'w') as handle:
        json.dump(rec, handle, indent=1)
    return d  # absolute (outside the repository): resolve_post_certification joins REPO with it unchanged


def k3_reference_d_ness(head):
    matrix = {}
    for group, refs, expect in (('D', D_REFS, True), ('AA_override', AA_OVERRIDE_REFS, False)):
        for name, ref_dir in refs.items():
            key = _load(os.path.join(ref_dir, 'evaluation_record.json'))['candidate_key']
            new, old = _try_resolve(H, ref_dir, key), _try_resolve(head, ref_dir, key)
            matrix[name] = {'group': group, 'eval_dir': ref_dir, 'new': new, 'head': old,
                            'expected_accepted': expect, 'ok': new['accepted'] == expect == old['accepted']}
    key_cstar = H.candidate_key(H.canonical_candidate(C_STAR))
    new, old = _try_resolve(H, CASE_FILE_AA_REF, key_cstar), _try_resolve(head, CASE_FILE_AA_REF, key_cstar)
    matrix['s45_case_file_aa_c_star'] = {'group': 'case_file_AA', 'eval_dir': CASE_FILE_AA_REF, 'new': new, 'head': old,
                                         'expected_accepted_new': False, 'head_accepted_before_the_fix': old['accepted'],
                                         'ok': (new['accepted'] is False) and (old['accepted'] is True)}

    def _off_override(rec):
        rec['evaluation_overrides_effective'] = {'anderson_acceleration': {'enabled': False}}

    def _declared_off(rec):
        rec['configuration'] = dict(rec['configuration'])
        rec['configuration']['case_file_anderson_acceleration'] = dict(CASE_FILE_AA, enabled=False)

    fx_off = _make_fixture_reference('declared_aa_on_plus_enabled_false_override', CASE_FILE_AA_REF, _off_override)
    fx_decl_off = _make_fixture_reference('declared_aa_off_no_override', CASE_FILE_AA_REF, _declared_off)
    for name, d in (('fixture_declared_on_override_off', fx_off), ('fixture_declared_off', fx_decl_off)):
        r = _try_resolve(H, d, key_cstar)
        matrix[name] = {'group': 'fixture_effective_aa_off', 'eval_dir': d, 'new': r, 'expected_accepted_new': True,
                        'ok': r['accepted'] is True and r['effective_anderson_acceleration']['enabled'] is False}

    # case-file-AA fixture specs (temporary roots; frozen by the harness itself, never launched)
    cfg = {'name': 'W5 fixture: case-file AA declared', 'arm_label': 's39_D', 'overrides': {},
           'case_file_anderson_acceleration': dict(CASE_FILE_AA), 'note': 'zero-solve check fixture; never launched'}
    common = dict(configuration=cfg, cap=500, concurrency=1, authority=['p515_s45_harness_phasea_check.py fixture'],
                  required_consecutive_cycles=10, extra={'mode': 'zero_solve_check_fixture', 'not_for_launch': True})
    polish_only = {'persist_certified_models': True, 'hull_polish': True}
    _p, _s, spec_np = H.freeze_campaign_spec(os.path.join(FIX['root'], 'k3_spec_polish_no_reference'), 's45_w5chk_k3a',
                                             [('c_star', C_STAR, {'post_certification': polish_only})], **common)
    e_np = spec_np['candidates'][0]
    _p, _s, spec_d = H.freeze_campaign_spec(
        os.path.join(FIX['root'], 'k3_spec_polish_d_reference'), 's45_w5chk_k3b',
        [('c_star', C_STAR, {'post_certification': dict(polish_only, reference={'eval_dir': D_REFS['gate_c_star']})})],
        **common)
    refused_root = os.path.join(FIX['root'], 'k3_spec_polish_case_file_aa_reference')
    try:
        H.freeze_campaign_spec(
            refused_root, 's45_w5chk_k3c',
            [('c_star', C_STAR, {'post_certification': dict(polish_only, reference={'eval_dir': CASE_FILE_AA_REF})})],
            **common)
        aa_ref_refused = {'raised': False}
    except ValueError as error:
        aa_ref_refused = {'raised': True, 'message': str(error)}

    # run_post_certification (REAL) without a reference, on the certified S45 trajectory; fakes only for the two
    # calls that would solve / pickle live models (the C10 pattern and fixture)
    calls = []

    def fake_persist(models, out_dir):
        calls.append('persist')
        return {'path': 'FAKE', 'sha256': 'FAKE', 'size_bytes': 0}

    committed_aa = _load(os.path.join(AA_RUN_DIR, 'aa_run_results.json'))
    committed_detail = _load(os.path.join(AA_RUN_DIR, 'hull_bound_detail.json'))

    def fake_polish(planning, models, consensus_vars):
        calls.append('polish')
        return json.loads(json.dumps(committed_aa['gate_d_hull_polish'])), committed_detail

    report = _load(os.path.join(CASE_FILE_AA_REF, 'g_s39_D.json'))
    d_pc = os.path.join(FIX['root'], 'k3_post_certification_no_reference_eval')
    os.makedirs(d_pc)
    pc, _detail = H.run_post_certification(
        planning=None, models=None, rows=report['cycle_trajectory'], report=report, state={'consensus_vars': {}},
        spec={'cap': 500, 'required_consecutive_cycles': 10}, entry=e_np, eval_dir=d_pc,
        polish_fn=fake_polish, persist_fn=fake_persist)
    summary = H.post_certification_summary(pc)
    checks = {
        'reference_matrix_all_as_expected': all(v['ok'] for v in matrix.values()),
        'head_harness_accepted_case_file_aa_reference_new_refuses': matrix['s45_case_file_aa_c_star']['ok'],
        'polish_without_reference_freezes_in_case_file_aa_spec': (
            e_np['post_certification'] == {'persist_certified_models': True, 'hull_polish': True, 'reference': None}),
        'd_reference_freezes_in_case_file_aa_spec': (
            (spec_d['candidates'][0]['post_certification']['reference'] or {}).get('configuration_overrides') == {}
            and 'effective_anderson_acceleration' not in spec_d['candidates'][0]['post_certification']['reference']),
        'case_file_aa_reference_refused_at_freeze': aa_ref_refused['raised'],
        'post_certification_without_reference_evaluated': (
            pc['status'] == 'evaluated' and pc['certification']['certified'] is True
            and pc['certification']['certification_cycle'] == 107),
        'gates_b_c_none_without_reference': (pc['gate_b_cost_vs_reference'] is None
                                             and pc['gate_c_cost_decomposition_vs_reference'] is None
                                             and pc['gate_c_pass'] is None),
        'gate_d_present_persist_before_polish': pc['gate_d_hull_polish'] is not None and calls == ['persist', 'polish'],
    }
    return checks, {'matrix': matrix, 'fixture_spec_polish_no_reference_entry': e_np,
                    'fixture_spec_d_reference_entry': spec_d['candidates'][0],
                    'case_file_aa_reference_refusal': aa_ref_refused,
                    'post_certification_no_reference_summary': summary,
                    'post_certification_no_reference_note': ('gate (d) numbers come from the FAKE polish (C10 fixture: '
                                                             'the committed S43 aa_run polish); only the control flow '
                                                             'is under test')}


# ======================================================================================================================
#  K4 -- error-record schema (main_child in-process) + the write trace of the child's pre-solve path
# ======================================================================================================================
_TRACE = {'on': False, 'events': []}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _audit(event, args):
    if not _TRACE['on']:
        return
    try:
        if event == 'open':
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                mode is None and isinstance(flags, int) and flags & _WRITE_FLAGS)
            if writing and isinstance(path, (str, bytes)):
                _TRACE['events'].append(('open_write', os.path.abspath(os.fsdecode(path))))
        elif event in ('os.rename', 'os.replace'):
            _TRACE['events'].append((event, os.path.abspath(os.fsdecode(args[1]))))
        elif event in ('os.mkdir', 'os.remove', 'os.unlink'):
            _TRACE['events'].append((event, os.path.abspath(os.fsdecode(args[0]))))
        elif event == 'subprocess.Popen':
            _TRACE['events'].append(('subprocess.Popen', str(args[1])[:200]))
    except Exception:  # noqa: BLE001 -- the tracer must never break the traced code
        pass


def _fixture_child_campaign(name, campaign_id):
    root = os.path.join(FIX['root'], f'k4_{name}')
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, campaign_id, [('c_star', C_STAR)],
        configuration={'name': 'W5 fixture: case-file AA declared (in-process main_child)', 'arm_label': 's39_D',
                       'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'note': 'zero-solve check fixture; failure injected; never solves'},
        cap=500, concurrency=1, authority=['p515_s45_harness_phasea_check.py fixture'], required_consecutive_cycles=10,
        extra={'mode': 'zero_solve_check_fixture', 'not_for_launch': True})
    entry = spec['candidates'][0]
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    os.makedirs(eval_dir)
    lock = os.path.join(root, 'temporary_campaign_lock.json')  # NOT the real lock; main_child is given its path
    with open(lock, 'w') as handle:
        json.dump({'pid': os.getppid(), 'campaign_id': campaign_id, 'campaign_spec_sha256': spec_sha,
                   'note': 'W5 in-process fixture lock (pid = this process\'s parent, as main_child verifies)'}, handle)
    argv = ['--child', '--campaign-root', root, '--spec-sha256', spec_sha, '--eval-key', entry['eval_key'],
            '--lock-path', lock]
    return root, spec, entry, eval_dir, argv


def _run_main_child(argv):
    try:
        H.main_child(argv)
        return None
    except SystemExit as exit_:
        return exit_.code


def k4_error_records():
    import p515_g_g1_g4_admm_gates as G
    os.environ.update(H.THREAD_CAP_ENV)  # main_child verifies the caps are in its environment
    out = {}
    new_keys = ('anderson_acceleration_effective_in_child', 'case_file_sha256_in_child')
    case_sha = H.sha256_file(H.CASE_FILE)

    # (A) failure before anything is known
    root, _s, _e, eval_dir, argv = _fixture_child_campaign('a_fail_first', 's45_w5chk_a')
    original_assert = H.assert_record_capture_paths

    def _raise_first():
        raise RuntimeError('W5 check: injected failure at the first statement of the evaluation')
    H.assert_record_capture_paths = _raise_first
    try:
        code = _run_main_child(argv)
    finally:
        H.assert_record_capture_paths = original_assert
    rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
    out['A_fail_first'] = {'exit_code': code,  'full_record': rec,'record': {k: rec.get(k) for k in ('status', 'barrier_cause') + new_keys},
                           'keys_present': all(k in rec for k in new_keys),
                           'ok': code == 1 and all(k in rec for k in new_keys)
                           and rec['anderson_acceleration_effective_in_child'] is None
                           and rec['case_file_sha256_in_child'] is None}

    # (B) the REAL working-dir-id refusal (after the case-file hash, before the hook)
    root, _s, entry, eval_dir, argv = _fixture_child_campaign('b_workdir_used', 's45_w5chk_b')
    original_work = G.O.WORK_DIR
    G.O.WORK_DIR = os.path.join(root, 'fixture_work_dir')
    os.makedirs(os.path.join(G.O.WORK_DIR, entry['working_dir_ids']['run']))
    try:
        code = _run_main_child(argv)
    finally:
        G.O.WORK_DIR = original_work
    rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
    out['B_workdir_refusal'] = {'exit_code': code, 'full_record': rec,
                                'record': {k: rec.get(k) for k in ('status', 'barrier_cause') + new_keys},
                                'ok': code == 1 and 'working dir id already used' in rec['barrier_cause']
                                and rec['case_file_sha256_in_child'] == case_sha
                                and rec['anderson_acceleration_effective_in_child'] is None}

    # (C) after the REAL hook on a REAL unsolved planning object; audit-traced
    root, _s, entry, eval_dir, argv = _fixture_child_campaign('c_after_hook', 's45_w5chk_c')
    work_c = os.path.join(root, 'fixture_work_dir')
    original_run = G.run_admm_arm
    hook_seen = {}

    def fake_run_admm_arm(label, out_dir, k_override=None, investment_map=None, num_max_iters_override=None,
                          eval_id=None, post_run_hook=None, apply_rho=True, full_diagnostics_in_rows=False,
                          pre_solve_hook=None):
        report = {'rule_eleven_checklist': {}}
        planning, sed, candidate = G._construct_arm_planning(
            label, out_dir, report, k_override=k_override, investment_map=investment_map, eval_id=eval_id,
            num_max_iters_override=num_max_iters_override, apply_rho=apply_rho)
        pre_solve_hook(planning=planning, sed=sed, candidate=candidate, report=report)
        hook_seen['checks'] = report['rule_eleven_checklist'].get('s44_campaign_configuration_checks')
        raise RuntimeError('W5 check: injected failure after the configuration hook, before any solve')

    baseline_loaded_before = G.O._BASELINE is not None
    diag_dir = os.path.join(REPO, 'data', 'SRP1', 'Diagrams')
    diag_before = {f: H.sha256_file(os.path.join(diag_dir, f)) for f in sorted(os.listdir(diag_dir))}
    def assert_then_install_fake():
        # the REAL rule-eleven assertion inspects the REAL run_admm_arm; the fake is installed only after it passed
        checklist = original_assert()
        G.run_admm_arm = fake_run_admm_arm
        return checklist

    G.O.WORK_DIR = work_c
    H.assert_record_capture_paths = assert_then_install_fake
    _TRACE['on'] = True
    try:
        code = _run_main_child(argv)
    finally:
        _TRACE['on'] = False
        H.assert_record_capture_paths = original_assert
        G.run_admm_arm = original_run
        G.O.WORK_DIR = original_work
    diag_after = {f: H.sha256_file(os.path.join(diag_dir, f)) for f in sorted(os.listdir(diag_dir))}
    rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
    out['C_after_hook'] = {'exit_code': code, 'full_record': rec,
                           'record': {k: rec.get(k) for k in ('status', 'barrier_cause') + new_keys},
                           'hook_checks': hook_seen.get('checks'),
                           'ok': code == 1 and 'after the configuration hook' in rec['barrier_cause']
                           and rec['case_file_sha256_in_child'] == case_sha
                           and rec['anderson_acceleration_effective_in_child'] == CASE_FILE_AA}

    # (D) parent-synthesized
    ctx = types.SimpleNamespace(spec={'campaign_id': 'x'}, spec_path=os.path.join(FIX['root'], 'none.json'),
                                spec_sha256='0' * 64)
    synth = H._barrier_record_for_missing(ctx, entry, eval_dir, 3)
    out['D_parent_synthesized'] = {'record_new_keys': {k: synth.get(k, 'ABSENT') for k in new_keys},
                                   'ok': all(k in synth and synth[k] is None for k in new_keys)}

    # the write trace of (C): classify every write outside this check's own root
    own = os.path.realpath(FIX['root']) + os.sep
    outside = {}
    for kind, path in _TRACE['events']:
        if kind == 'subprocess.Popen':
            outside.setdefault('<subprocess>', []).append(path)
            continue
        if path.startswith(own):
            continue
        outside.setdefault(os.path.dirname(path), []).append(f'{kind}:{os.path.basename(path)}')
    trace = {
        'traced_scope': ('main_child in-process, fixture C: assert_record_capture_paths, _build_floor_rows (precheck '
                         'fresh_planning -> p56a_oracle.load_baseline), _construct_arm_planning, the configuration '
                         'hook; NOT the ADMM loop (it solves)'),
        'baseline_loaded_in_this_process_before_C': baseline_loaded_before,
        'n_write_events_total': len(_TRACE['events']),
        'writes_outside_this_check_root_by_directory': {d: {'n': len(v), 'sample': sorted(set(v))[:60]}
                                                         for d, v in sorted(outside.items())},
        'data_SRP1_Diagrams': {'n_files': len(diag_after), 'n_files_rewritten_content_changed': sum(
            1 for f in diag_after if diag_before.get(f) != diag_after[f]), 'tracked_in_git': bool(
            _git(['ls-files', '--', 'data/SRP1/Diagrams']).strip())},
    }
    checks = {f'{k}_ok': v['ok'] for k, v in out.items()}
    return checks, {'scenarios': out, 'write_trace_child_pre_solve_path': trace}


# ======================================================================================================================
def main():
    started = _utc()
    if os.path.exists(OUT):
        raise SystemExit(f'refusing: output root exists (write-once): {OUT}')
    os.makedirs(OUT)
    sys.addaudithook(_audit)
    with tempfile.TemporaryDirectory(prefix='p515s45_w5_fixtures_') as tmp:
        FIX['root'] = tmp
        _main_body(started)


def _main_body(started):
    head, head_sha = _head_harness_module()
    results = {'stage': 'P5.15 Addendum 27 W5 -- Phase A harness fixes: zero-solve checks',
               'started_utc': started, 'git_head': _git(['rev-parse', 'HEAD']).strip(),
               'harness_sha256_working_tree': H.sha256_file(H.HARNESS_PATH),
               'harness_sha256_head': head_sha,
               'script_sha256': H.sha256_file(os.path.abspath(__file__)), 'checks': {}, 'detail': {}}
    ok = True
    for name, fn in (('K1_bar', k1_bar), ('K2_keys_and_references', k2_keys_and_references),
                     ('K3_reference_d_ness', lambda: k3_reference_d_ness(head)),
                     ('K4_error_records', k4_error_records)):
        _log(f'[W5-CHECK] {name} ...')
        try:
            checks, detail = fn()
        except Exception as error:  # noqa: BLE001 -- recorded, the script fails
            checks, detail = {'raised': False}, {'error': f'{type(error).__name__}: {error}',
                                                 'traceback': traceback.format_exc()}
            print(detail['traceback'], file=sys.stderr, flush=True)
        results['checks'][name] = checks
        results['detail'][name] = detail
        ok = ok and all(checks.values())
        _log(f'[W5-CHECK] {name}: {checks}')
    guard_failures = GUARD.verify(0)
    results['solve_profile_guard'] = {'permitted': [], 'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures}
    results['all_checks_pass'] = ok and not guard_failures
    results['ended_utc'] = _utc()
    H._write_once_json(os.path.join(OUT, 'check.json'), results)
    manifest = {}
    for r_, _dirs, files in os.walk(OUT):
        for fname in sorted(files):
            fpath = os.path.join(r_, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(OUT, 'manifest_sha256.json'), manifest)
    k1 = results['detail'].get('K1_bar') or {}
    _log(f"[W5-CHECK] bar max |new - recorded| = {k1.get('max_abs_diff_eur')} EUR over "
         f"{len(k1.get('per_record') or [])} records; not within 1e-6: {k1.get('not_within_1e-6')}")
    _log(f'[W5-CHECK] guard {GUARD.counts} verify0_failures={guard_failures}')
    _log(f"[W5-CHECK] {'ALL PASS' if results['all_checks_pass'] else 'NOT ALL PASS'}")
    GUARD.uninstall()
    if not results['all_checks_pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
