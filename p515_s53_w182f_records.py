"""
P5.15 Addendum 82 v2, W182 order part B, Planner task W182f -- items Z2, Z3, Z4, Z5, Z7. ZERO SOLVES, NO MODEL, NO
PICKLE LOAD, NO PRODUCTION-MODEL IMPORT. Static reading of committed (and, where stated, uncommitted) records.

ITEMS
  Z2  Row 40 of Table E.21 (`ref:5ca4f86c`, the historical 2x incumbent, uncertified, "gap clause"): the issuing run's
      cycle cap, cycles used and remaining cap; whether a rule-v6 certificate is reachable within the cap. Primary
      evidence: the committed v6 decision on the record (W142 item 1, `w142_v6_from_records.json`) and the W142
      floor-replay AB arm. Cross-check: the committed zero-solve replay function `settling_criterion_v6.replay` (stdlib
      only) is CALLED on the run's committed per-cycle records with the inputs W142 used (Q = per_cycle_record
      gross_operational_cost; boyd = boyd_all_pass AND local_solves_ok; t = resettle_cycle_record t_sum; non-clean
      cycles = the W142 floor replay's; cap/ceiling = the replayed cap rule; p_max = the run's settling_rule.p_max).
      Every committed run of the same candidate key is listed with its cap and cycles.
  Z3  One table over the 61 E.21 rows (row -> issuing run from W180b's committed JSON). Per row:
        spec_freeze_head  = campaign spec `git_head` (repo HEAD when the campaign spec was frozen, before launch);
        run_end_head      = campaign_results `git_head_at_run` (repo HEAD when the results were written, after the run);
        decision_head     = for "record" rows, the git_head of the committed record that issued the v6 decision;
        harness sha256    = campaign spec `harness.sha256` (p515_s44_campaign_harness.py as pinned at freeze);
        final code head   = F = 575bfec4 (Planner: "last production-code commit"), checked here as the last commit
                            touching a PRODUCTION MODULE := a git-tracked root-level *.py whose name does not match
                            ^p[0-9] (diagnostic harnesses) -- at the commit compared;
        final_head_literal  = (run_end_head == F);
        production_equal    = every production module blob at run_end_head equals F's (no file added, removed or
                              modified);
        production_diff     = git ls-tree blob comparison, run_end_head vs F, per file: 'A' (in F only: added after),
                              'D' (in run head only), 'M' (blob differs);
        only_additions      = production_diff non-empty and every entry 'A' (modules that did not exist when the run
                              ran, hence could not be imported by it);
        harness_equal_final = harness sha256 == sha256 of p515_s44_campaign_harness.py at the repo HEAD (the harness
                              is p-prefixed, hence outside F's definition; its own last commit is reported).
      CONFIGURATION HASH (definition, preserved in the output): cfg_hash = sha256 of
        json.dumps(PAYLOAD, sort_keys=True, separators=(',', ':')) with PAYLOAD = {
          'case_file_sha256':                 spec.configuration.case_file_sha256,
          'case_file_anderson_acceleration':  spec.configuration.case_file_anderson_acceleration,
          'effective_anderson_acceleration':  entry.effective_anderson_acceleration,
          'convergence_depth_tail':           spec.configuration.convergence_depth_tail,
          'ess_ageing_baseline':              spec.configuration.ess_ageing_baseline,
          'ess_params_file_sha256':           spec.configuration.ess_params_file.sha256,
          'overrides':                        {'configuration': spec.configuration.overrides, 'entry': entry.overrides},
          'apply_rho':                        spec.configuration.apply_rho,
          'arm_label':                        spec.configuration.arm_label,
          'flex_price_multiplier':            entry.flex_price_multiplier (None when absent),
          'model_variant' / 'derived_instance' / 'interface_deviation_premium': entry value (None when absent),
          'run_declaration_model_fields': the model-affecting keys of entry.settling_resettle when present
                                          ('kind', 'minimum_soh', 'flex_price_multiplier_expected', 'model_variant'),
                                          a key whose value is null being omitted}
        -- the operational-model configuration, independent of the candidate (key/canonical), of the run control (cap,
        settling declaration, replay reference, stop rule), of post-certification options and of free text. Reported
        beside it: the eval key (the harness's own identity hash of candidate x configuration x run declaration), the
        campaign spec sha256, the stop-rule class of the stage spec, and the runtime fields (interpreter,
        NLP_SOLVER_PATH, thread caps) which are compared, not hashed.
  Z4  Wall time per cycle and peak RSS of the 3 x 3 cells (w90_3x3/campaign_s53_w91_3x3_pair and the continuation
      w98_continuation/campaign_s53_w98_x0_continuation_r2; the two w91 smoke cells reported beside, labelled) from the
      records only: per_cycle_record cycle_wall_s / rss_bytes / ru_maxrss_bytes, evaluation_record wall_time_s /
      peak_rss, wait4_rusage.json. A five-block single-scenario build: searched for (scope recorded); "not recorded"
      where absent; the nearest recorded builds reported beside, labelled as such.
  Z5  E6 inputs for rows 1-2 (W101 continuations x0 and n7_4h_e1): for each declared E6 quantity, which file carries
      it at the certified cycle k*, and whether that file is committed, uncommitted text, or pickle; then a complete
      file inventory of both eval dirs and of their P56A working dirs (sha256, size, path, mtime, git status,
      disposition). Pickles are hashed as bytes, never opened. Small uncommitted text inputs (<= 1 MiB) that carry an
      E6 quantity at k* found in no committed record are listed for commit; everything else is hash-recorded.
  Z7  Row 1 (x = 0, k* 181): the reverse-flow interface-hours (p_int < 0 on the DSO side; material: < -1e-6 p.u. of
      the DN base) with magnitude, block weight and market price, from row 1's committed
      `interface_settlement_detail_s31c.json`; cross-checked against W116's committed count
      (`reverse_flow_count_q181.json`, written from certified_models.pkl by W116); raw interface duals at those hours
      from the committed `interface_duals_per_cycle.jsonl` (unconverted).

Guards: `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0 at the end;
`pickle.load` / `pickle.loads` blocked for the whole run and counted (0 expected). No production model module imported
(asserted; the stdlib-only `settling_criterion*` modules are the only project modules besides the guard and the
writer). Refuses to run unless the output directory exists and holds nothing but launch.log, and refuses to run
concurrently with another copy of itself. A pre-run capture-path checklist asserts every required input exists before
any reading. JSON written through `gate_result_io`.

Output (data/SRP1/Results/P515S53/w182f_records/): w182f_records.json, z5_rows12_file_manifest.json,
manifest_sha256.json; `--post-run` then writes manifest_post_run_sha256.json over outputs + launch.log.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w182f_records && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182f_records.py \\
      > data/SRP1/Results/P515S53/w182f_records/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182f_records.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W182f records Z2/Z3/Z4/Z5/Z7 (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W182f: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W182f: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer)
import settling_criterion_v6 as SC6  # noqa: E402  (stdlib only; the committed zero-solve v6 replay)

SCRIPT_REL = os.path.basename(__file__)
P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(P53, 'w182f_records')
OUT_JSON = os.path.join(OUT_DIR, 'w182f_records.json')
OUT_FILES = os.path.join(OUT_DIR, 'z5_rows12_file_manifest.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
OUT_POST = os.path.join(OUT_DIR, 'manifest_post_run_sha256.json')

FINAL_HEAD = '575bfec43ff47b03ed9f6ef955025cd35cfb28a2'      # Planner: last production-code commit (575bfec4)
HARNESS_REL = 'p515_s44_campaign_harness.py'
W180B = os.path.join(P53, 'w180b_row16_class', 'w180b_row16_class.json')
W142_V6R = os.path.join(P53, 'w142_resettle_v6', 'v6_from_records', 'w142_v6_from_records.json')
W142_CERT_D = os.path.join(P53, 'w142_resettle_v6', 'v6_from_records', 'd_c52e1670_v6_from_records_certificate.json')
W142_FLOOR = os.path.join(P53, 'w142_resettle_v6', 'floor_replay', 'w142_floor_replay.json')
W116_RF = os.path.join(P53, 'w116_benchmark_nrf', 'w116_zero_solve_checks', 'reverse_flow_count_q181.json')
W116_REPORT = os.path.join(P53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
ROW40_ROOT = os.path.join(P53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_f2_incumbent')
ROW40_EVAL = os.path.join(ROW40_ROOT, 'evals', '24c5ccb6f285219f_f2_incumbent')
ROW40_W142_ID = 'f2_incumbent'
ROWS12 = {
    1: {'cell': 'ref:7aa017f0', 'label': 'x0', 'k_star': 181,
        'eval': os.path.join(P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0', 'evals',
                             'd110bd1a5977df1e_x0'),
        'workdirs': [os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                  'p515s44_s53_w101_srp1_cont_x0_d110bd1a5977df1e_run'),
                     os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                  'p515s44_s53_w101_srp1_cont_x0_d110bd1a5977df1e_precheck')]},
    2: {'cell': 'ref:bd504ecf', 'label': 'n7_4h_e1', 'k_star': 172,
        'eval': os.path.join(P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals',
                             '3f084f2ffaeef2b7_n7_4h_e1'),
        'workdirs': [os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                  'p515s44_s53_w101_srp1_cont_n7_4h_e1_3f084f2ffaeef2b7_run'),
                     os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                  'p515s44_s53_w101_srp1_cont_n7_4h_e1_3f084f2ffaeef2b7_precheck')]},
}
CELLS_3X3 = [
    ('pair', 'x0', os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair'), 'f6e9cd53fdbb8ee8_x0'),
    ('pair', 'n7_4h_e1', os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair'), 'c82522f470b35b58_n7_4h_e1'),
    ('continuation', 'x0_cont', os.path.join(P53, 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2'),
     '25b92ae0f1f2c02e_x0_cont'),
    ('smoke (beside, labelled)', 'x0 smoke_bon', os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_smoke_bon'),
     'f6e9cd53fdbb8ee8_x0'),
    ('smoke (beside, labelled)', 'x0 smoke_boff', os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_smoke_boff'),
     'f6e9cd53fdbb8ee8_x0'),
]
COMMIT_MAX_BYTES = 1 << 20
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage', 'energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions',
                             'uncoordinated_benchmark', 'p515_s44_campaign_harness')

INPUTS = {}
_TRACKED = None


def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, check=True, capture_output=True).stdout


def _tracked_set():
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = set(_git(['ls-files', '-z']).decode('utf-8').split('\0'))
    return _TRACKED


def _status(path):
    return 'committed' if os.path.relpath(os.path.abspath(path), REPO) in _tracked_set() else 'UNCOMMITTED'


def _reg(path):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': _sha(path), 'status': _status(path)}
    return INPUTS[rel]


def _json(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _jsonl(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def _jsonl_first_last(path):
    """(first line, second line, last line) parsed, without registering a hash (large uncommitted files are
    hash-recorded in the Z5 file manifest instead)."""
    first = second = last = None
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if first is None:
                first = line
            elif second is None:
                second = line
            last = line
    p = (lambda s: json.loads(s) if s else None)
    return p(first), p(second), p(last)


def other_copies():
    r = subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    out = []
    for ln in r:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(a.endswith(SCRIPT_REL) for a in argv[1:3]):
            out.append(ln.strip())
    return out


# ============================================================================================ capture-path checklist
def capture_checklist(w180b_rows):
    need = {'W180b row map': W180B, 'W142 v6 from records': W142_V6R, 'W142 d certificate': W142_CERT_D,
            'W142 floor replay': W142_FLOOR, 'W116 reverse-flow count': W116_RF, 'W116 report v3': W116_REPORT,
            'row 40 campaign_results': os.path.join(ROW40_ROOT, 'campaign_results.json'),
            'row 40 per_cycle_record': os.path.join(ROW40_EVAL, 'per_cycle_record.jsonl'),
            'row 40 resettle_cycle_record': os.path.join(ROW40_EVAL, 'resettle_cycle_record.jsonl'),
            'row 40 resettle_decision': os.path.join(ROW40_EVAL, 'resettle_decision.json'),
            'harness (current)': HARNESS_REL}
    for no, r in ROWS12.items():
        for f in ('interface_settlement_detail_s31c.json', 'interface_duals_per_cycle.jsonl', 'evaluation_record.json',
                  'per_cycle_record.jsonl', 'component_levels_terminal.json', 'soh_floor_sidecar_baseline.jsonl',
                  'recourse_blocks_all.jsonl', 'interface_voltage_terminal.json', 'boyd_terminal.json',
                  'ess_entry_stride_baseline.jsonl', 'certified_models.pkl'):
            need[f'row {no} {f}'] = os.path.join(r['eval'], f)
    for grp, lab, root, ev in CELLS_3X3:
        need[f'3x3 {grp} {lab} evaluation_record'] = os.path.join(root, 'evals', ev, 'evaluation_record.json')
        need[f'3x3 {grp} {lab} per_cycle_record'] = os.path.join(root, 'evals', ev, 'per_cycle_record.jsonl')
    for x in w180b_rows:
        need[f"E.21 row {x['no']} campaign_results"] = x['issuing']['campaign_results']
    missing = {k: v for k, v in need.items() if not os.path.exists(v)}
    return {'n_required': len(need), 'missing': missing, 'ok': not missing}


# ======================================================================================================= Z2
def z2_row40(w180b_rows):
    row = next(x for x in w180b_rows if x['no'] == 40)
    assert row['cell'] == 'ref:5ca4f86c', row['cell']
    res = _json(os.path.join(ROW40_ROOT, 'campaign_results.json'))
    spec_path = res['campaign_spec_path']
    spec = _json(spec_path)
    entry = next(e for e in spec['candidates'] if e['eval_key'] == res['eval_key'])
    sr = entry['settling_resettle']
    g15 = res['gate_detail']['G15']
    dec_file = _json(os.path.join(ROW40_EVAL, 'resettle_decision.json'))
    w142 = _json(W142_V6R)['item1_v6_on_every_record'][ROW40_W142_ID]
    floor = _json(W142_FLOOR)['reports'][ROW40_W142_ID]
    ab = floor['arms']['AB']
    # ---- cross-check: the committed replay function on the committed records, W142's inputs
    rows = _jsonl(os.path.join(ROW40_EVAL, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _jsonl(os.path.join(ROW40_EVAL, 'resettle_cycle_record.jsonl'))}
    q = {r['cycle']: r['gross_operational_cost'] for r in rows}
    b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
    t = {k: (lines.get(k) or {}).get('t_sum') for k in q}
    nc = set(floor['non_clean_cycles'])
    clean = {k: k not in nc for k in q}
    kw = {'cap': floor['cap_rule_replayed']['cap'], 'cap_ceiling': floor['cap_rule_replayed']['cap_ceiling']}
    p_max = sr['settling_rule']['p_max']
    recs, d6, _rule = SC6.replay(q, b, t, clean, p_max, last=len(rows), **kw)
    gaps = d6.get('gap_refusals') or []
    gap_abs = [g['abs_t_sum'] for g in gaps]
    gap_cycles = [g['cycle'] for g in gaps]
    # least-squares slope of |t_sum| over the refusal cycles (observation only)
    slope = None
    if len(gaps) >= 2:
        mx, my = statistics.fmean(gap_cycles), statistics.fmean(gap_abs)
        num = sum((x - mx) * (y - my) for x, y in zip(gap_cycles, gap_abs))
        den = sum((x - mx) ** 2 for x in gap_cycles)
        slope = num / den
    t_last = t.get(len(rows))
    cross = {
        'function': 'settling_criterion_v6.replay (committed, stdlib only; zero solves)',
        'inputs': {'Q': 'per_cycle_record.jsonl gross_operational_cost', 'boyd': 'boyd_all_pass AND local_solves_ok',
                   't_sum': 'resettle_cycle_record.jsonl t_sum', 'non_clean_cycles': sorted(nc),
                   'cap_kw': kw, 'p_max': p_max, 'last': len(rows)},
        'decision': {k: d6.get(k) for k in ('status', 'k_star', 'k_cap', 'cap', 'cap_ceiling', 'reasons', 'k0', 'N',
                                            'band', 'band_width', 'band_window', 'T', 'A', 'n_vetoes',
                                            'gap_clause_refused_at_cap', 'Q_at_cap', 't_sum_at_cap', 'gap_bound')},
        'n_gap_refusals': len(gaps),
        'equals_W142_item1': {'status': d6.get('status') == w142['v6']['status'],
                              'k_star': d6.get('k_star') == w142['v6']['k_star'],
                              'k_cap': d6.get('k_cap') == w142['v6']['k_cap'],
                              'band_width': d6.get('band_width') == w142['v6']['band_width'],
                              'T': json.dumps(d6.get('T')) == json.dumps(w142['v6']['T'])},
        'equals_floor_replay_AB': {'status': d6.get('status') == ab['status'],
                                   'reasons': list(d6.get('reasons') or []) == list(ab['reasons'] or []),
                                   'n_gap_refusals': len(gaps) == ab['n_gap_refusals'],
                                   'gap_refusals': json.dumps(gaps, sort_keys=True) == json.dumps(
                                       ab['gap_refusals'], sort_keys=True)},
        'n_cycles_replayed': len(recs),
    }
    cross['all_equal'] = all(cross['equals_W142_item1'].values()) and all(cross['equals_floor_replay_AB'].values())
    branch_counts = {}
    for g in gaps:
        branch_counts[g['branch']] = branch_counts.get(g['branch'], 0) + 1
    # every committed run of the same candidate
    cand = res['candidate_key']
    others = []
    hits = [p.decode() for p in _git(['grep', '-l', cand, '--', 'data/*campaign_spec_*.json']).split(b'\n') if p]
    for sp in sorted(hits):
        s = _json(sp)
        for e in s['candidates']:
            if e['key'] != cand:
                continue
            ed = os.path.join(os.path.dirname(sp), 'evals', e['eval_dir'])
            rec = None
            if os.path.isfile(os.path.join(ed, 'evaluation_record.json')):
                er = _json(os.path.join(ed, 'evaluation_record.json'))
                rec = {k: er.get(k) for k in ('status', 'cycles_run', 'certification_cycle', 'cap')}
            others.append({'campaign_spec': sp, 'spec_cap': s.get('cap'), 'label': e['label'],
                           'eval_key16': e['eval_key'][:16], 'flex_price_multiplier': e.get('flex_price_multiplier'),
                           'settling_declaration': ('settling_resettle' if 'settling_resettle' in e else None),
                           'eval_dir_exists': os.path.isdir(ed), 'evaluation_record': rec,
                           'superseded_md_present': os.path.exists(os.path.join(os.path.dirname(sp), 'SUPERSEDED.md'))})
    cap = sr['cap_rule']['cap']
    used = g15['cycles_run']
    out = {
        'row': 40, 'cell': row['cell'], 'printed': row['printed'],
        'issuing_run': {'campaign_results': os.path.join(ROW40_ROOT, 'campaign_results.json'),
                        'campaign_spec': spec_path, 'eval_key': res['eval_key'], 'candidate_key': cand,
                        'original_eval_key': res['original_eval_key'], 'stage_spec': res['stage_spec'],
                        'stop_rule': sr['settling_rule']['class'], 'flex_price_multiplier':
                            entry.get('flex_price_multiplier')},
        'cap': {'cap_rule': sr['cap_rule'], 'N_old': sr['N_old'], 'spec_cap': spec['cap'], 'G15_spec_cap': g15.get('spec_cap'),
                'G15_k_cap': g15['k_cap'], 'cycles_run': used, 'stopped_by': g15['stopped_by'],
                'remaining_cap_cycles': cap - used, 'cap_ceiling_replayed': kw['cap_ceiling'],
                'decision_file': {k: dec_file.get(k) for k in ('cap', 'cap_mode', 'decided_at_cycle',
                                                                'gap_clause_refused_at_cap', 'first_residual_pass_run')}},
        'v6_on_record_W142': {'evidence': f'{W142_V6R} item1_v6_on_every_record.{ROW40_W142_ID}.v6; '
                                          f'{W142_FLOOR} reports.{ROW40_W142_ID}.arms.AB',
                              'v6': w142['v6'], 'v6_equals_replay_arm_AB': w142['v6_equals_replay_arm_AB'],
                              'floor_AB': {k: ab.get(k) for k in ('status', 'reasons', 'n_gap_refusals', 'k0', 'N',
                                                                  'k_cap', 'n_vetoes', 'band_width')},
                              'non_clean_cycles': sorted(nc)},
        'v6_replay_crosscheck': cross,
        'gap_clause': {'gap_bound': d6.get('gap_bound'), 'tau': SC6.TAU,
                       'refusal_cycles_first_last': [min(gap_cycles), max(gap_cycles)] if gaps else None,
                       'refusals_by_branch': branch_counts,
                       'abs_t_sum_min_max_over_refusals': [min(gap_abs), max(gap_abs)] if gaps else None,
                       'abs_t_sum_over_gap_bound_min': (min(gap_abs) / d6['gap_bound']) if gaps else None,
                       'abs_t_sum_slope_per_cycle_over_refusals': slope,
                       't_sum_at_last_cycle': t_last,
                       'reading': ('every refusal is a cycle at which a branch test passed and the gap clause '
                                   '|t_sum| <= gap_bound refused; the rule does not decide before the cap')},
        'other_runs_of_candidate': others,
        'other_runs_note': ('evaluation_record.status / certification_cycle is the harness residual certificate '
                            '(consecutive Boyd-pass cycles), not a settling-rule decision; the settling-rule decision '
                            'of the issuing run is G15 / resettle_decision.json'),
    }
    reach_within = (d6.get('status') == 'certified')
    out['answer'] = {
        'remaining_cap_cycles': cap - used,
        'v6_certificate_reachable_within_cap': reach_within,
        'rests_on': [
            f'the issuing run used its whole cap: cap {cap} (rule {sr["cap_rule"]["formula"]}, kind '
            f'{sr["cap_rule"]["kind"]}), cycles_run {used}, stopped_by {g15["stopped_by"]} (campaign_results G15)',
            'v6 applied to the run\'s committed record over cycles 1..%d is UNCERTIFIED with k_cap %s (W142 item 1, '
            'equal to the floor-replay AB arm); v6.replay stops at the first decision, so no cycle k <= %d '
            'certifies under v6' % (used, w142['v6']['k_cap'], used),
            'the same decision is reproduced here by calling settling_criterion_v6.replay on the committed records '
            '(all fields compared: %s)' % cross['all_equal'],
            'the only refusal reason is the gap clause: %d refusals (cycles %s..%s, branch %s), |t_sum| %.1f..%.1f '
            'EUR against gap_bound %.1f EUR = tau/2' % (len(gaps), min(gap_cycles), max(gap_cycles), branch_counts,
                                                        min(gap_abs), max(gap_abs), d6['gap_bound']),
            'beyond the cap nothing is recorded: whether the gap would close after cycle %d is not knowable from '
            'the records (no new cycles were run)' % used],
    }
    return out


# ======================================================================================================= Z3
PROD_RE = re.compile(r'^[^/]+\.py$')
DIAG_RE = re.compile(r'^p[0-9]')


def production_blobs(commit):
    out = {}
    for ln in _git(['ls-tree', '-r', commit]).decode().splitlines():
        meta, path = ln.split('\t', 1)
        _mode, typ, blob = meta.split()
        if typ == 'blob' and PROD_RE.match(path) and not DIAG_RE.match(path):
            out[path] = blob
    return out


_BLOBS = {}


def _blobs(commit):
    if commit not in _BLOBS:
        _BLOBS[commit] = production_blobs(commit)
    return _BLOBS[commit]


def prod_diff(head, final):
    a, b = _blobs(head), _blobs(final)
    d = {}
    for p in sorted(set(a) | set(b)):
        if p not in a:
            d[p] = 'A'
        elif p not in b:
            d[p] = 'D'
        elif a[p] != b[p]:
            d[p] = 'M'
    return d


def is_ancestor(x, y):
    return subprocess.run(['git', 'merge-base', '--is-ancestor', x, y], cwd=REPO).returncode == 0


def cfg_payload(spec, entry):
    c = spec['configuration']
    return {'case_file_sha256': c.get('case_file_sha256'),
            'case_file_anderson_acceleration': c.get('case_file_anderson_acceleration'),
            'effective_anderson_acceleration': entry.get('effective_anderson_acceleration'),
            'convergence_depth_tail': c.get('convergence_depth_tail'),
            'ess_ageing_baseline': c.get('ess_ageing_baseline'),
            'ess_params_file_sha256': (c.get('ess_params_file') or {}).get('sha256'),
            'overrides': {'configuration': c.get('overrides'), 'entry': entry.get('overrides')},
            'apply_rho': c.get('apply_rho'), 'arm_label': c.get('arm_label'),
            'flex_price_multiplier': entry.get('flex_price_multiplier'),
            'model_variant': entry.get('model_variant'), 'derived_instance': entry.get('derived_instance'),
            'interface_deviation_premium': entry.get('interface_deviation_premium'),
            'run_declaration_model_fields': {k: (entry.get('settling_resettle') or {}).get(k)
                                             for k in RUN_DECL_MODEL_FIELDS
                                             if (entry.get('settling_resettle') or {}).get(k) is not None}}


RUN_DECL_MODEL_FIELDS = ('kind', 'minimum_soh', 'flex_price_multiplier_expected', 'model_variant')


def importers(module):
    """Tracked *.py files at HEAD holding an import statement of `module` (regex over the file text)."""
    rx = re.compile(r'^\s*(import\s+%s\b|from\s+%s\s+import\b)' % (module, module), re.M)
    out = []
    for path in _git(['ls-files', '*.py']).decode().splitlines():
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8', errors='replace') as h:
            if rx.search(h.read()):
                out.append(path)
    return out


def cfg_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def z3_table(w180b_rows):
    final_short = FINAL_HEAD[:8]
    last_prod = _git(['log', '-1', '--format=%H', '--', *sorted(_blobs('HEAD'))]).decode().strip()
    head = _git(['rev-parse', 'HEAD']).decode().strip()
    harness_head_sha = hashlib.sha256(_git(['show', f'HEAD:{HARNESS_REL}'])).hexdigest()
    harness_final_sha = hashlib.sha256(_git(['show', f'{FINAL_HEAD}:{HARNESS_REL}'])).hexdigest()
    harness_last_commit = _git(['log', '-1', '--format=%H', '--', HARNESS_REL]).decode().strip()
    w142_head = _json(W142_V6R).get('git_head')
    wd_cert_head = _json(W142_CERT_D).get('git_head')
    rows = []
    payloads = {}
    for x in w180b_rows:
        res = _json(x['issuing']['campaign_results'])
        spec = _json(res['campaign_spec_path'])
        entry = next(e for e in spec['candidates'] if e['eval_key'] == res['eval_key'])
        pl = cfg_payload(spec, entry)
        h = cfg_hash(pl)
        payloads[h] = pl
        run_head = res.get('git_head_at_run')
        freeze_head = spec.get('git_head')
        dif = prod_diff(run_head, FINAL_HEAD)
        dif_freeze = prod_diff(freeze_head, run_head)
        stage = _json(res['stage_spec']['path']) if isinstance(res.get('stage_spec'), dict) else None
        stop_rule = ((stage or {}).get('stop_rule') or {}).get('class')
        v6 = x.get('v6_decision') or {}
        if x['issuing'].get('kind') == 'v6 from records certificate':
            dec_kind, dec_head = 'v6 from records certificate (W142 item 2)', wd_cert_head
        elif x['printed']['rule'] == 'run':
            dec_kind, dec_head = 'v6 run (G15 of the issuing run)', run_head
        else:
            dec_kind = (v6.get('kind') or 'none') + f" -> {v6.get('status')}"
            dec_head = w142_head
        nsp = spec.get('nlp_solver_path_env') or {}
        rows.append({
            'no': x['no'], 'cell': x['cell'], 'status_printed': x['printed']['status_printed'],
            'rule_printed': x['printed']['rule'],
            'campaign_results': x['issuing']['campaign_results'], 'campaign_spec': res['campaign_spec_path'],
            'campaign_spec_sha256': res.get('campaign_spec_sha256'), 'eval_key': res['eval_key'],
            'stage_spec': (res.get('stage_spec') or {}).get('path'), 'stop_rule_class': stop_rule,
            'spec_freeze_head': freeze_head, 'run_end_head': run_head,
            'head_moved_during_run': freeze_head != run_head,
            'production_diff_freeze_to_run_end': dif_freeze,
            'decision_kind': dec_kind, 'decision_head': dec_head,
            'final_head_literal': run_head == FINAL_HEAD,
            'final_is_ancestor_of_run_head': is_ancestor(FINAL_HEAD, run_head),
            'run_head_is_ancestor_of_final': is_ancestor(run_head, FINAL_HEAD),
            'production_equal_final': not dif, 'production_diff_vs_final': dif,
            'only_additions_vs_final': bool(dif) and all(v == 'A' for v in dif.values()),
            'harness_sha256': spec['harness']['sha256'],
            'harness_equal_HEAD': spec['harness']['sha256'] == harness_head_sha,
            'harness_equal_at_final_head': spec['harness']['sha256'] == harness_final_sha,
            'campaign_script': (spec.get('extra') or {}).get('campaign_script'),
            'campaign_script_sha256': (spec.get('extra') or {}).get('campaign_script_sha256'),
            'cfg_hash': h, 'cfg_hash16': h[:16],
            'runtime': {'interpreter': spec.get('interpreter'), 'NLP_SOLVER_PATH': nsp.get('NLP_SOLVER_PATH'),
                        'thread_caps': spec.get('thread_caps')},
        })
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r['cfg_hash16'], []).append(r['no'])
    changed = sorted({f for r in rows for f, st in list(r['production_diff_vs_final'].items()) +
                      list(r['production_diff_freeze_to_run_end'].items()) if st in ('M', 'D')})
    imp = {}
    for f in changed:
        mod = f[:-3]
        who = importers(mod)
        imp[f] = {'importers_at_HEAD': who,
                  'imported_by_campaign_harness': HARNESS_REL in who,
                  'imported_by_a_production_module': [w for w in who if not DIAG_RE.match(w)],
                  'imported_by_a_row_campaign_script': sorted({r['campaign_script'] for r in rows
                                                               if r['campaign_script'] in who})}
    runtimes = {json.dumps(r['runtime'], sort_keys=True) for r in rows}
    harness_shas = {}
    for r in rows:
        harness_shas.setdefault(r['harness_sha256'][:8], []).append(r['no'])
    return {
        'final_head': {'declared': FINAL_HEAD, 'definition': ('last commit touching a production module := git-tracked '
                                                              'root-level *.py not matching ^p[0-9]'),
                       'recomputed_last_production_commit_at_HEAD': last_prod,
                       'declared_equals_recomputed': last_prod == FINAL_HEAD, 'repo_HEAD': head,
                       'n_production_modules_at_final': len(_blobs(FINAL_HEAD)),
                       'production_equal_HEAD_vs_final': not prod_diff(head, FINAL_HEAD),
                       'harness_last_commit': harness_last_commit,
                       'harness_sha256_at_HEAD': harness_head_sha, 'harness_sha256_at_final_head': harness_final_sha},
        'cfg_hash_definition': ('sha256 of json.dumps(PAYLOAD, sort_keys=True, separators=(",", ":")); PAYLOAD keys: '
                                + ', '.join(sorted(cfg_payload({'configuration': {}}, {}).keys()))
                                + ' (see the script docstring for the source of each)'),
        'cfg_payloads': payloads,
        'rows_by_cfg_hash16': by_cfg,
        'modified_production_files_and_importers': imp,
        'importer_scope': ('regex ^\\s*(import M\\b|from M import) over every git-tracked *.py at HEAD; dynamic '
                           'imports (importlib, __import__) are not detected'),
        'rows_by_harness_sha8': harness_shas,
        'n_distinct_runtime': len(runtimes), 'runtimes': sorted(runtimes),
        'rows': rows,
        'summary': {'n_rows': len(rows),
                    'n_final_head_literal': sum(r['final_head_literal'] for r in rows),
                    'n_production_equal_final': sum(r['production_equal_final'] for r in rows),
                    'n_only_additions': sum(r['only_additions_vs_final'] for r in rows),
                    'rows_with_modified_or_deleted_production_files': [
                        r['no'] for r in rows if any(v in ('M', 'D') for v in r['production_diff_vs_final'].values())],
                    'rows_head_moved_during_run_with_production_change': [
                        r['no'] for r in rows if r['production_diff_freeze_to_run_end']],
                    'n_harness_equal_HEAD': sum(r['harness_equal_HEAD'] for r in rows)},
    }


# ======================================================================================================= Z4
def _stats(v):
    v = [x for x in v if isinstance(x, (int, float))]
    if not v:
        return None
    return {'n': len(v), 'mean': statistics.fmean(v), 'median': statistics.median(v), 'min': min(v), 'max': max(v),
            'sum': sum(v)}


def z4_3x3():
    cells = []
    for grp, lab, root, ev in CELLS_3X3:
        ed = os.path.join(root, 'evals', ev)
        er = _json(os.path.join(ed, 'evaluation_record.json'))
        pcr = _jsonl(os.path.join(ed, 'per_cycle_record.jsonl'))
        w4p = os.path.join(ed, 'wait4_rusage.json')
        w4 = _json(w4p) if os.path.exists(w4p) else None
        cw = [r.get('cycle_wall_s') for r in pcr]
        rss = [r.get('rss_bytes') for r in pcr]
        mrss = [r.get('ru_maxrss_bytes') for r in pcr]
        cyc = [r.get('cycle') for r in pcr]
        wt = er.get('wall_time_s') or {}
        pk = er.get('peak_rss') or {}
        cell = {'group': grp, 'label': lab, 'campaign_root': root, 'eval_dir': ed,
                'status': er.get('status'), 'cycles_run': er.get('cycles_run'),
                'certification_cycle': er.get('certification_cycle'), 'cap': er.get('cap'),
                'per_cycle_record_cycles': [min(cyc), max(cyc)] if cyc else None,
                'n_cycles_with_cycle_wall_s': sum(1 for x in cw if isinstance(x, (int, float))),
                'cycle_wall_s': _stats(cw),
                'cycle_wall_s_first_cycle': cw[0] if cw else None,
                'cycle_wall_s_excl_first': _stats(cw[1:]),
                'rss_bytes_end_of_cycle': _stats(rss),
                'ru_maxrss_bytes_per_cycle_max': max([x for x in mrss if isinstance(x, (int, float))], default=None),
                'evaluation_record_wall_time_s': wt, 'evaluation_record_peak_rss': pk,
                'wait4_rusage': w4,
                'derived_non_cycle_wall_s': ((wt.get('child_process_s') - sum(x for x in cw if isinstance(x, (int, float))))
                                             if wt.get('child_process_s') is not None and cw else None)}
        if grp == 'continuation':
            sel = [r.get('cycle_wall_s') for r in pcr if isinstance(r.get('cycle'), int) and r['cycle'] > 72]
            cell['cycle_wall_s_cycles_73_on'] = _stats(sel)
        cells.append(cell)
    # five-block single-scenario build: search (recursive over every dict of each hit file)
    hits = [p.decode() for p in _git(['grep', '-l', '-e', '"n_network_blocks_built"', '-e', '"expected_block_counts"',
                                      '--', '*.json']).split(b'\n') if p]
    found, nearest = [], []

    def _scen(dct):
        sc = dct.get('scenario_combinations')
        if sc is None and isinstance(dct.get('planning_dimensions'), dict):
            pdm = dct['planning_dimensions']
            ops = pdm.get('num_operation_scenarios')
            ops1 = set(ops.values()) if isinstance(ops, dict) else {ops}
            if pdm.get('num_market_scenarios') is not None and None not in ops1 and ops1:
                sc = pdm['num_market_scenarios'] * max(ops1)
        return sc

    def _walk(x, path, out):
        if isinstance(x, dict):
            nb = x.get('n_network_blocks_built')
            if nb is None and isinstance(x.get('expected_block_counts'), dict):
                nb = x['expected_block_counts'].get('network_blocks')
            if nb is not None:
                out.append({'json_path': path or '/', 'network_blocks': nb, 'scenario_combinations': _scen(x),
                            'build_wall_s': x.get('build_wall_s'),
                            'models_built_rss_bytes': x.get('models_built_rss_bytes'),
                            'rss_at_build_complete': (x.get('memory_at_build_complete') or {}).get('rss_self'),
                            'ru_maxrss_self_at_build_complete': x.get('ru_maxrss_self_at_build_complete')})
            for k, v in x.items():
                _walk(v, f'{path}/{k}', out)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                _walk(v, f'{path}[{i}]', out)

    for p in sorted(hits):
        try:
            d = _json(p)
        except Exception as error:  # noqa: BLE001 -- recorded
            nearest.append({'path': p, 'error': f'{type(error).__name__}: {error}'})
            continue
        recs = []
        _walk(d, '', recs)
        for rec in recs:
            rec['path'] = p
            (found if (rec['network_blocks'] == 80 and rec['scenario_combinations'] == 1) else nearest).append(rec)
    return {'cells': cells,
            'five_block_single_scenario_build': {
                'definition': ('five-block = the 3 x 3 horizon 2025/2028/2031/2034/2037 (Addendum 72 decision 2), i.e. '
                               '80 network blocks (5 years x 4 days x 4 networks); single-scenario = 1 market x 1 '
                               'operation scenario'),
                'search_scope': ('git-tracked *.json files containing "n_network_blocks_built" or '
                                 '"expected_block_counts" (git grep at HEAD), read in full, every dict at any depth '
                                 'carrying either key; scenario count from a sibling scenario_combinations or '
                                 'planning_dimensions (market x max operation); a hit = 80 network blocks and 1 '
                                 'scenario combination'),
                'n_files_searched': len(hits), 'found': found, 'recorded': bool(found),
                'nearest_records_beside_labelled': nearest}}


# ======================================================================================================= Z5
E6_QUANTITIES = [
    # (quantity, [(file, how-to-locate, kind)])
    ('ESS hourly charge / discharge / net power, ESSO side, at k*',
     [('esso_capture/s39_D/node{n}_cycle{k}.jsonl', 'fields pch, pdch, pnet per (y_inv, y, d, p)', 'text'),
      ('ess_entry_stride_baseline.jsonl', 'line cycle k*: entries[power_type p].x.esso (24 values per block)', 'text'),
      ('certified_models.pkl', 'model variables (not opened)', 'pickle')]),
    ('ESS hourly active power, network copies (TSO / DSO) and consensus z, at k*',
     [('ess_entry_stride_baseline.jsonl', 'line cycle k*: entries[power_type p].x.tso / x.dso / z', 'text'),
      ('certified_models.pkl', 'model variables (not opened)', 'pickle')]),
    ('ESS hourly reactive power (all copies), at k*',
     [('ess_entry_stride_baseline.jsonl', 'line cycle k*: entries[power_type q]', 'text'),
      ('certified_models.pkl', 'model variables (not opened)', 'pickle')]),
    ('ESS hourly state of charge / stored energy, at k*',
     [('certified_models.pkl', 'model variables (not opened)', 'pickle')]),
    ('SoH per cohort and year, at k*',
     [('soh_floor_sidecar_baseline.jsonl', 'line cycle k*: entries[].es_soh_per_unit_cumul', 'text'),
      ('evaluation_record.json', 'storage_per_node.<n>.ageing_trajectory_terminal', 'text'),
      ('boyd_terminal.json', 'soh_floor_multiplier_and_efc_per_cohort_year_terminal', 'text')]),
    ('Available capacity per year (E^Av, S^Av), terminal',
     [('evaluation_record.json', 'storage_per_node.<n>.published_available_capacity_terminal', 'text')]),
    ('EFC per day, at k*',
     [('per_cycle_record.jsonl', 'line cycle k*: efc_per_day_max', 'text'),
      ('ess_entry_stride_baseline.jsonl', 'line cycle k*: efc_per_day_per_node', 'text')]),
    ('Market price per hour',
     [('interface_settlement_detail_s31c.json', 'interface_reporting_detail.<n>.<y>.<d>.periods.<p>.price_per_mwh',
       'text')]),
    ('Interface P / Q per hour, TSO and DSO side, terminal',
     [('interface_settlement_detail_s31c.json',
       'interface_reporting_detail.<n>.<y>.<d>.periods.<p>.{p,q}_int_{tso,dso}_expected_*', 'text')]),
    ('Block weight per (year, day)',
     [('component_levels_terminal.json', 'blocks.<block>.admm_block_weight', 'text'),
      ('interface_settlement_detail_s31c.json', 'interface_consensus_residual_per_dso.<n>.periods.<y|d|p>.admm_block_weight',
       'text')]),
    ('Per-block cost components, terminal / at k*',
     [('component_levels_terminal.json', 'blocks.<block>.{unweighted,weighted}', 'text'),
      ('recourse_blocks_all.jsonl', 'line cycle k*: objective_component_blocks', 'text')]),
    ('Interface consensus duals per hour, at k* (scaled, unconverted)',
     [('interface_duals_per_cycle.jsonl', 'line cycle k*: lambda_pf_{p,q}_{tso,dso}', 'text')]),
    ('Interface voltage per hour, terminal',
     [('interface_voltage_terminal.json', 'entries[]', 'text')]),
    ('Gross operational cost Q(k*)',
     [('per_cycle_record.jsonl', 'line cycle k*: gross_operational_cost', 'text')]),
]
SOC_KEY_RE = re.compile(r'(^|[^a-z])(soc|state_of_charge|stored_energy|energy_level)([^a-z]|$)', re.I)


SOC_SELFTEST = {'match': ('es_soc', 'shared_es_soc', 'SoC_init', 'state_of_charge', 'soc'),
                'no_match': ('associated', 'es_soh_per_unit_cumul', 'social', 'e_available', 'pnet')}


def soc_selftest():
    ok_m = {k: bool(SOC_KEY_RE.search(k)) for k in SOC_SELFTEST['match']}
    ok_n = {k: not SOC_KEY_RE.search(k) for k in SOC_SELFTEST['no_match']}
    return {'match': ok_m, 'no_match': ok_n, 'all_pass': all(ok_m.values()) and all(ok_n.values())}


def _keys(obj, out, depth=0):
    if depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _keys(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            _keys(v, out, depth + 1)


def _locate(eval_dir, k, rel_tmpl, how):
    """Verify a quantity's field is present in the file at k* (reads text only). Returns (present, detail)."""
    rels = []
    if '{n}' in rel_tmpl:
        rels = [rel_tmpl.format(n=n, k=k) for n in (5, 7, 9)]
    else:
        rels = [rel_tmpl]
    det = []
    any_present = False
    for rel in rels:
        p = os.path.join(eval_dir, rel)
        if not os.path.exists(p):
            det.append({'file': rel, 'exists': False})
            continue
        if rel.endswith('.pkl'):
            det.append({'file': rel, 'exists': True, 'opened': False, 'size': os.path.getsize(p), 'git': _status(p)})
            any_present = True
            continue
        size = os.path.getsize(p)
        info = {'file': rel, 'exists': True, 'size': size, 'git': _status(p)}
        try:
            if size == 0:
                info['present'] = False
                info['note'] = 'empty file'
            elif rel.endswith('.jsonl') and rel.startswith('esso_capture'):
                first, _s, _l = _jsonl_first_last(p)
                info['present'] = all(f in first for f in ('pch', 'pdch', 'pnet'))
                info['first_line_keys'] = sorted(first)
            elif rel.endswith('.jsonl'):
                big = size > (32 << 20) or info['git'] == 'UNCOMMITTED'
                if big:
                    _f, _s, last = _jsonl_first_last(p)
                    lines = [last]
                else:
                    lines = _jsonl(p)
                at = [x for x in lines if x.get('cycle') == k]
                info['line_at_k_star'] = bool(at)
                row = at[0] if at else {}
                if 'efc_per_day_max' in how:
                    info['present'] = 'efc_per_day_max' in row
                elif 'efc_per_day_per_node' in how:
                    info['present'] = 'efc_per_day_per_node' in row
                elif 'gross_operational_cost' in how:
                    info['present'] = 'gross_operational_cost' in row
                elif 'es_soh_per_unit_cumul' in how:
                    info['present'] = bool(row.get('entries')) and all('es_soh_per_unit_cumul' in e for e in row['entries'])
                elif 'objective_component_blocks' in how:
                    info['present'] = bool(row.get('objective_component_blocks'))
                elif 'lambda' in how:
                    info['present'] = all(f in row for f in ('lambda_pf_p_tso', 'lambda_pf_p_dso', 'lambda_pf_q_tso',
                                                             'lambda_pf_q_dso'))
                elif 'power_type q' in how:
                    info['present'] = any(e.get('power_type') == 'q' for e in row.get('entries') or [])
                elif 'power_type p' in how:
                    ents = [e for e in row.get('entries') or [] if e.get('power_type') == 'p']
                    if 'x.tso' in how:      # network copies and consensus z
                        info['present'] = bool(ents) and all('tso' in (e.get('x') or {}) and 'dso' in (e.get('x') or {})
                                                             and 'z' in e for e in ents)
                    else:                   # ESSO copy
                        info['present'] = bool(ents) and all('esso' in (e.get('x') or {}) for e in ents)
                    info['n_entries_p'] = len(ents)
                else:
                    info['present'] = bool(row)
            else:
                d = _json(p)
                if rel == 'interface_settlement_detail_s31c.json':
                    rd = d['interface_reporting_detail']
                    per = rd['7']['2025']['Spring']['periods']['0']
                    if 'price' in how:
                        info['present'] = 'price_per_mwh' in per
                    elif 'admm_block_weight' in how:
                        info['present'] = 'admm_block_weight' in d['interface_consensus_residual_per_dso']['7'][
                            'periods']['2025|Spring|0']
                    else:
                        info['present'] = all(f in per for f in ('p_int_tso_expected_mw', 'p_int_dso_expected_mw',
                                                                 'q_int_tso_expected_mvar', 'q_int_dso_expected_mvar'))
                    info['cycles_run'] = d.get('cycles_run')
                elif rel == 'component_levels_terminal.json':
                    b0 = next(iter(d['blocks'].values()))
                    info['present'] = ('admm_block_weight' in b0) if 'weight' in how else ('unweighted' in b0)
                    info['cycles_run'] = d.get('cycles_run')
                elif rel == 'interface_voltage_terminal.json':
                    info['present'] = bool(d.get('entries'))
                    info['cycle'] = (d.get('summary') or {}).get('cycle')
                elif rel == 'evaluation_record.json':
                    spn = d.get('storage_per_node') or {}
                    key = 'ageing_trajectory_terminal' if 'ageing' in how else 'published_available_capacity_terminal'
                    info['present'] = any(isinstance(v, dict) and key in v for v in spn.values())
                    info['nodes_with_field'] = sorted(n for n, v in spn.items() if isinstance(v, dict) and key in v)
                    info['cycles_run'] = d.get('cycles_run')
                elif rel == 'boyd_terminal.json':
                    info['present'] = 'soh_floor_multiplier_and_efc_per_cohort_year_terminal' in d
                    info['cycles'] = d.get('cycles')
                else:
                    info['present'] = True
        except Exception as error:  # noqa: BLE001 -- recorded
            info['present'] = False
            info['error'] = f'{type(error).__name__}: {error}'
        any_present = any_present or bool(info.get('present'))
        det.append(info)
    return any_present, det


def soc_key_scan(eval_dir):
    """Every text record of the eval dir: the key names of JSON files (whole file) and of JSONL files (header, first
    data line, last line), matched against SOC_KEY_RE. Pickles are not opened."""
    hits, scanned = {}, []
    for dp, _dn, fn in os.walk(eval_dir):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, eval_dir)
            if f.endswith('.pkl') or os.path.getsize(p) == 0:
                continue
            keys = set()
            try:
                if f.endswith('.json'):
                    with open(p, encoding='utf-8') as h:
                        _keys(json.load(h), keys)
                elif f.endswith('.jsonl'):
                    a, b, c = _jsonl_first_last(p)
                    for x in (a, b, c):
                        _keys(x, keys)
                else:
                    continue
            except Exception as error:  # noqa: BLE001
                scanned.append({'file': rel, 'error': f'{type(error).__name__}: {error}'})
                continue
            scanned.append({'file': rel, 'n_keys': len(keys)})
            m = sorted(k for k in keys if SOC_KEY_RE.search(k))
            if m:
                hits[rel] = m
    return {'pattern': SOC_KEY_RE.pattern, 'pattern_selftest': soc_selftest(), 'n_files_scanned': len(scanned),
            'hits': hits,
            'scope': ('JSON: every key at any depth (lists: first 50 elements); JSONL: header/first, second and last '
                      'lines; .log and .txt not scanned; pickles not opened')}


def file_inventory(paths_roots):
    out = []
    for role, root in paths_roots:
        if not os.path.isdir(root):
            out.append({'role': role, 'root': root, 'exists': False})
            continue
        for dp, _dn, fn in os.walk(root):
            for f in sorted(fn):
                p = os.path.join(dp, f)
                st = os.stat(p)
                out.append({'role': role, 'path': os.path.relpath(p, REPO), 'size': st.st_size,
                            'mtime_utc': datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                            'sha256': _sha(p), 'git': _status(p), 'pickle': f.endswith('.pkl')})
    return out


def z5_rows12():
    rows = {}
    commit_now = []
    all_files = []
    for no, r in ROWS12.items():
        ed, k = r['eval'], r['k_star']
        qrows = []
        for qname, sources in E6_QUANTITIES:
            src = []
            for rel, how, kind in sources:
                present, det = _locate(ed, k, rel, how)
                gits = sorted({d.get('git') for d in det if d.get('git')}) or (['UNCOMMITTED'] if kind == 'pickle' else [])
                src.append({'file': rel, 'locate': how, 'kind': kind, 'present_at_k_star': present, 'detail': det,
                            'git': gits})
            committed = [s['file'] for s in src if s['present_at_k_star'] and s['kind'] == 'text' and s['git'] == ['committed']]
            unc_text = [s['file'] for s in src if s['present_at_k_star'] and s['kind'] == 'text' and 'UNCOMMITTED' in s['git']]
            pk = [s['file'] for s in src if s['present_at_k_star'] and s['kind'] == 'pickle']
            cls = ('committed record' if committed else 'uncommitted text only' if unc_text else
                   'certified_models.pkl only' if pk else 'not recorded')
            qrows.append({'quantity': qname, 'class': cls, 'committed_sources': committed,
                          'uncommitted_text_sources': unc_text, 'pickle_sources': pk, 'sources': src})
        ess_sched = os.path.join(ed, 'ess_schedule_per_cycle.jsonl')
        scan = soc_key_scan(ed)
        inv = file_inventory([('eval_dir', ed)] + [('p56a_working_dir', w) for w in r['workdirs']])
        for f in inv:
            f['row'] = no
        all_files.extend(inv)
        # the commit rule
        for q in qrows:
            if q['class'] != 'uncommitted text only':
                continue
            for s in q['sources']:
                if s['file'] not in q['uncommitted_text_sources']:
                    continue
                for d in s['detail']:
                    if not d.get('exists') or d.get('git') != 'UNCOMMITTED' or not d.get('present'):
                        continue
                    p = os.path.join(ed, d['file'])
                    if os.path.getsize(p) <= COMMIT_MAX_BYTES:
                        commit_now.append({'row': no, 'path': os.path.relpath(p, REPO), 'size': os.path.getsize(p),
                                           'sha256': _sha(p), 'why': q['quantity']})
        rows[no] = {'cell': r['cell'], 'label': r['label'], 'k_star': k, 'eval_dir': ed,
                    'ess_schedule_per_cycle_jsonl_exists': os.path.exists(ess_sched),
                    'quantities': qrows, 'soc_key_scan': scan}
    # dedupe commit list
    seen, cn = set(), []
    for c in commit_now:
        if c['path'] not in seen:
            seen.add(c['path'])
            cn.append(c)
    # any ess_schedule_per_cycle.jsonl anywhere for these two eval keys (scope of the negative claim)
    search = []
    for dp, _dn, fn in os.walk(os.path.join('data')):
        if 'ess_schedule_per_cycle.jsonl' in fn:
            search.append(dp)
    keyhits = [d for d in search if 'd110bd1a5977df1e' in d or '3f084f2ffaeef2b7' in d]
    for f in all_files:
        if 'path' not in f:
            continue
        if f['git'] == 'committed':
            f['disposition'] = 'already committed'
        elif f['path'] in seen:
            f['disposition'] = 'commit now (small text E6 input at k*)'
        else:
            f['disposition'] = 'hash-recorded (pickle)' if f['pickle'] else 'hash-recorded (large or not an E6 input at k*)'
    return {'rows': rows, 'commit_now': cn, 'commit_rule': (
                'an UNCOMMITTED text file of the eval dir that carries an E6 quantity at k* which no committed record '
                f'carries, of size <= {COMMIT_MAX_BYTES} bytes; everything else uncommitted is hash-recorded'),
            'ess_schedule_per_cycle_search': {'scope': 'every directory under data/ (os.walk) holding a file named '
                                                       'ess_schedule_per_cycle.jsonl',
                                              'n_dirs_with_file': len(search),
                                              'dirs_matching_row1_or_row2_eval_key': keyhits},
            'file_inventory': all_files}


# ======================================================================================================= Z7
def z7_row1():
    ed = ROWS12[1]['eval']
    d = _json(os.path.join(ed, 'interface_settlement_detail_s31c.json'))
    rd = d['interface_reporting_detail']
    cr = d['interface_consensus_residual_per_dso']
    hdr, _x, _y = _jsonl_first_last(os.path.join(ed, 'interface_duals_per_cycle.jsonl'))
    duals = _jsonl(os.path.join(ed, 'interface_duals_per_cycle.jsonl'))
    lam_k = next((x for x in duals if x.get('cycle') == 181), None)
    blocks = [tuple(b) for b in hdr['blocks']]
    meta = {(m['node_id'], m['year'], m['day']): m for m in hdr['metadata']['per_block']}
    base = 100.0
    entries, n = [], 0
    for node, v in cr.items():
        for key, e in v['periods'].items():
            n += 1
            y, day, p = key.split('|')
            pd, pt = e['p_int_dso_expected_mw'], e['p_int_tso_expected_mw']
            if pd < 0 or pt < 0:
                r = rd[node][y][day]['periods'][p]
                bi = blocks.index((int(node), int(y), day))
                lam = {f: lam_k[f][bi][int(p)] for f in ('lambda_pf_p_tso', 'lambda_pf_p_dso')} if lam_k else None
                w = e['admm_block_weight']
                entries.append({'node_id': int(node), 'year': y, 'day': day, 'period': int(p), 'hour': int(p) + 1,
                                'p_int_dso_mw': pd, 'p_int_tso_mw': pt,
                                'material_dso': pd < -1e-6 * base, 'admm_block_weight': w,
                                'market_price_eur_per_mwh': r['price_per_mwh'],
                                'market_price_by_scenario': r['price_per_mwh_by_scenario'],
                                'reverse_energy_block_weighted_mwh': abs(pd) * w,
                                'first_order_value_eur_at_market_price': abs(pd) * w * r['price_per_mwh'],
                                'lambda_raw_at_cycle_181': lam,
                                'lambda_block_metadata': meta.get((int(node), int(y), day))})
    entries.sort(key=lambda x: x['p_int_dso_mw'])
    w116 = _json(W116_RF)
    lst = w116['count']['reverse_entries_most_negative_first']
    match = (len(lst) == len(entries) and all(
        a['p_int_mw'] == b['p_int_dso_mw'] and a['node_id'] == b['node_id'] and a['year'] == b['year'] and
        a['day'] == b['day'] and a['period'] == b['period'] for a, b in zip(lst, entries)))
    tot_w = sum(x['admm_block_weight'] for x in entries)
    tot_e = sum(x['reverse_energy_block_weighted_mwh'] for x in entries)
    w116_tot = w116['count']['totals']['strict']
    rep = _json(W116_REPORT)['coordinated_reverse_flow_count']
    return {'source': os.path.join(ed, 'interface_settlement_detail_s31c.json'),
            'source_git': _status(os.path.join(ed, 'interface_settlement_detail_s31c.json')),
            'source_cycles_run': d.get('cycles_run'), 'row1_k_star': ROWS12[1]['k_star'],
            'sign_convention': ('p_int > 0 import TN -> DN, p_int < 0 reverse (DN -> TN): W116 sign proof, '
                                f'{W116_RF} sign_proof'),
            'price_convention': ('price_per_mwh = E[pi_t] = cost_energy_p[0][t] at one market scenario '
                                 '(shared_resources_planning._INTERFACE_REPORTING_PRICE_CONVENTION)'),
            'block_weight_definition': ('years[y] x days[d] x 1/(1+r)^(y - y0) '
                                        '(shared_resources_planning._get_admm_block_weight); 2025 => no discount'),
            'value_column_definition': ('|p_int_dso| x block weight x market price: the energy x price product, '
                                        'derived here; NOT a model quantity and NOT the value of permitting reverse '
                                        'flow'),
            'n_interface_hours_scanned': n, 'n_reverse_any_side': len(entries),
            'n_material_dso': sum(1 for x in entries if x['material_dso']),
            'entries_most_negative_first': entries,
            'totals': {'block_weight_sum': tot_w, 'reverse_energy_block_weighted_mwh': tot_e,
                       'first_order_value_eur': sum(x['first_order_value_eur_at_market_price'] for x in entries)},
            'w116_crosscheck': {'path': W116_RF, 'instance': w116.get('instance'),
                                'entries_bitwise_equal': match,
                                'count_equal': w116_tot['count'] == len(entries),
                                'count_day_weighted_equal': w116_tot['count_day_weighted'] == tot_w,
                                'energy_block_weighted_abs_diff': abs(w116_tot['energy_mwh_block_weighted'] - tot_e),
                                'w116_totals_strict': w116_tot,
                                'report_v3_totals_strict': rep['totals']['strict']},
            'lambda_note': ('raw consensus duals as production holds them (scaled; conversion to EUR/MWh lives in '
                            'uncoordinated_benchmark.interface_price_terms and was NOT applied here)'),
            'lambda_line_found_at_181': lam_k is not None}


# ======================================================================================================= main
def main():
    global OUT_DIR, OUT_JSON, OUT_FILES, OUT_MANIFEST, OUT_POST
    if '--out-dir' in sys.argv:                    # trial runs only (a scratch directory holding only launch.log)
        OUT_DIR = sys.argv[sys.argv.index('--out-dir') + 1]
        OUT_JSON = os.path.join(OUT_DIR, 'w182f_records.json')
        OUT_FILES = os.path.join(OUT_DIR, 'z5_rows12_file_manifest.json')
        OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
        OUT_POST = os.path.join(OUT_DIR, 'manifest_post_run_sha256.json')
    if '--post-run' in sys.argv:
        rels = [OUT_JSON, OUT_FILES, OUT_MANIFEST, os.path.join(OUT_DIR, 'launch.log')]
        for r in rels:
            if not os.path.exists(r):
                raise SystemExit(f'post-run: {r} missing')
        if os.path.exists(OUT_POST):
            raise SystemExit(f'post-run: {OUT_POST} exists (write-once)')
        man = {r: _sha(r) for r in rels}
        with open(OUT_POST, 'x', encoding='utf-8') as h:
            GRIO.dump(man, h, indent=1, sort_keys=True)
            h.write('\n')
        print('post-run manifest:', json.dumps(man, indent=1))
        return
    started = datetime.now(timezone.utc).isoformat()
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'REFUSED (check: output dir exists): {OUT_DIR} does not exist')
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        raise SystemExit(f'REFUSED (check: output dir fresh): {OUT_DIR} holds {extra}')
    oc = other_copies()
    if oc:
        raise SystemExit(f'REFUSED (check: no concurrent copy): {oc}')
    head = _git(['rev-parse', 'HEAD']).decode().strip()
    dirty = _git(['status', '--porcelain', '--untracked-files=no']).decode().splitlines()
    _log(f'repo HEAD {head}; tracked changes: {dirty}')
    w180b = _json(W180B)
    chk = capture_checklist(w180b['rows'])
    if not chk['ok']:
        raise SystemExit(f'REFUSED (check: capture-path checklist): missing {chk["missing"]}')
    _log(f'capture-path checklist: {chk["n_required"]} inputs present')

    z2 = z2_row40(w180b['rows'])
    _log(f"Z2: cap {z2['cap']['G15_k_cap']} used {z2['cap']['cycles_run']} remaining {z2['cap']['remaining_cap_cycles']}; "
         f"v6 W142 {z2['v6_on_record_W142']['v6']['status']}; replay cross-check equal {z2['v6_replay_crosscheck']['all_equal']}; "
         f"reachable within cap {z2['answer']['v6_certificate_reachable_within_cap']}")
    z3 = z3_table(w180b['rows'])
    _log(f"Z3: {z3['summary']}")
    z4 = z4_3x3()
    _log(f"Z4: five-block single-scenario build recorded: {z4['five_block_single_scenario_build']['recorded']}")
    z5 = z5_rows12()
    _log(f"Z5: commit_now {[c['path'] for c in z5['commit_now']]}; files inventoried {len(z5['file_inventory'])}")
    z7 = z7_row1()
    _log(f"Z7: reverse entries {z7['n_reverse_any_side']} (material {z7['n_material_dso']}); W116 bitwise "
         f"{z7['w116_crosscheck']['entries_bitwise_equal']}")

    inv = z5.pop('file_inventory')
    bad = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    if bad or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        raise SystemExit(f'guard fault: modules {bad}; guard {gfail}; pickle {PICKLE_COUNTS}')
    result = {
        'schema': 'p515_s53_w182f_records_v1',
        'stage': 'P5.15 Addendum 82 v2, W182 part B, task W182f: Z2, Z3, Z4, Z5, Z7 (zero solves, records only)',
        'started_utc': started, 'git_HEAD': head, 'tracked_changes_at_start': dirty,
        'interpreter': sys.executable,
        'objective_convention': ('Q = gross_operational_cost, settlement excluded, where any Q is quoted; t_sum = the '
                                 'priced interface-consensus gap (EUR)'),
        'capture_checklist': chk,
        'Z2_row40': z2, 'Z3_e21_code_heads': z3, 'Z4_wall_rss': z4, 'Z5_e6_inputs_rows12': z5, 'Z7_row1_reverse_flow': z7,
        'project_modules_imported': sorted(m for m in sys.modules if m.startswith(('settling_criterion', 'gate_result_io',
                                                                                    'p513_solve_profile_guard'))),
        'guards': {'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts),
                                           'verify_0_failures': gfail},
                   'pickle_counts': dict(PICKLE_COUNTS),
                   'pickle_blocked': pickle.load is _blocked_load and pickle.loads is _blocked_loads,
                   'forbidden_modules_imported': bad},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    with open(OUT_JSON, 'x', encoding='utf-8') as fh:
        GRIO.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    files_doc = {'schema': 'p515_s53_w182f_rows12_file_manifest_v1',
                 'what': ('every file under the eval dirs of E.21 rows 1-2 (W101 continuations x0, n7_4h_e1) and of '
                          'their P56A working dirs: path, size, sha256, mtime (UTC), git status at the run, pickle '
                          'flag, disposition. Pickles hashed as bytes, never opened.'),
                 'commit_rule': z5['commit_rule'], 'commit_now': z5['commit_now'],
                 'n_files': len(inv), 'n_uncommitted': sum(1 for f in inv if f.get('git') == 'UNCOMMITTED'),
                 'bytes_uncommitted': sum(f.get('size', 0) for f in inv if f.get('git') == 'UNCOMMITTED'),
                 'files': inv}
    with open(OUT_FILES, 'x', encoding='utf-8') as fh:
        GRIO.dump(files_doc, fh, indent=1, sort_keys=False)
        fh.write('\n')
    manifest = {'outputs': {OUT_JSON: _sha(OUT_JSON), OUT_FILES: _sha(OUT_FILES)}, 'script': {SCRIPT_REL: _sha(SCRIPT_REL)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(OUT_MANIFEST, 'x', encoding='utf-8') as fh:
        GRIO.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')

    # ------------------------------------------------------------------------------------------- console summary
    print('=== Z2 row 40')
    print(json.dumps(z2['cap'], indent=1))
    print(json.dumps(z2['answer'], indent=1))
    print(json.dumps(z2['gap_clause'], indent=1))
    for o in z2['other_runs_of_candidate']:
        print('  run of candidate:', json.dumps(o))
    print('=== Z3')
    print(json.dumps(z3['final_head'], indent=1))
    print('rows by cfg hash:', json.dumps(z3['rows_by_cfg_hash16']))
    print('modified production files and importers:', json.dumps(z3['modified_production_files_and_importers'], indent=1))
    print('rows by harness sha:', json.dumps(z3['rows_by_harness_sha8']))
    print('runtimes:', z3['runtimes'])
    for r in z3['rows']:
        print(f"  {r['no']:>2} {r['cell']:<16} freeze {r['spec_freeze_head'][:8]} end {r['run_end_head'][:8]} "
              f"dec {str(r['decision_head'])[:8]} lit {r['final_head_literal']} prodeq {r['production_equal_final']} "
              f"onlyA {r['only_additions_vs_final']} diff {sorted(r['production_diff_vs_final'].items())} "
              f"harness {r['harness_sha256'][:8]} cfg {r['cfg_hash16']} rule {r['stop_rule_class']}")
    print(json.dumps(z3['summary'], indent=1))
    print('=== Z4')
    for c in z4['cells']:
        print(json.dumps({k: c[k] for k in c if k not in ('wait4_rusage',)}, indent=1))
    print(json.dumps({k: v for k, v in z4['five_block_single_scenario_build'].items()}, indent=1))
    print('=== Z5')
    for no, r in z5['rows'].items():
        print(f"row {no} {r['cell']} k* {r['k_star']} ess_schedule_per_cycle exists {r['ess_schedule_per_cycle_jsonl_exists']}")
        for q in r['quantities']:
            print(f"   {q['class']:<28} {q['quantity']}  committed={q['committed_sources']} "
                  f"uncommitted_text={q['uncommitted_text_sources']} pickle={q['pickle_sources']}")
        print('   SoC key scan:', json.dumps(r['soc_key_scan']['hits']), 'files', r['soc_key_scan']['n_files_scanned'])
    print('commit now:', json.dumps(z5['commit_now'], indent=1))
    print('ess_schedule search:', json.dumps(z5['ess_schedule_per_cycle_search']))
    print(f"inventory: {files_doc['n_files']} files, {files_doc['n_uncommitted']} uncommitted, "
          f"{files_doc['bytes_uncommitted']} bytes uncommitted")
    print('=== Z7')
    for e in z7['entries_most_negative_first']:
        print('  ', json.dumps({k: e[k] for k in e if k != 'lambda_block_metadata'}))
    print(json.dumps(z7['totals']), json.dumps(z7['w116_crosscheck']))
    print('guards:', json.dumps(result['guards']))
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')
    print(f'output {OUT_FILES} sha256 {_sha(OUT_FILES)}')


if __name__ == '__main__':
    main()
