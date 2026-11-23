"""P5.15 S56 validation step 4, item 2b -- ZERO-SOLVE checks of the two harness changes (Planner task 2026-11-23).

  --grouping  The initialisation-identity grouping by pin construction (`p515_s44_campaign_harness.
              initialisation_identity_group` used by `register_initialisation_identity`). Synthetic records only (no
              model): the harness BEFORE the change (`git show <rev>:p515_s44_campaign_harness.py`, loaded from a
              scratch copy as its own module) and AFTER (the working tree) are called with identical inputs:
                (a) option-off roots (two off records of one candidate, equal and unequal): the returned comparison and
                    the written record are identical before and after (bytes, after normalising the root path and the
                    record's own utc / pid);
                (b) a mixed root (off, on 'pin', on 'per_scenario' of one candidate): after the change off and
                    on-'pin' are compared (equal -> all_equal, a differing hex -> a mismatch), on-'per_scenario' is
                    listed and not compared; before the change no cross-declaration comparison was made.
  --capture   The per-cycle S56 capture (`s56_cycle_capture_hooks` + `assert_s56_cycle_capture_path`): the existing
              zero-solve probe `p515_s56_step1_completion_checks.probe` (stubbed solves, 2 ADMM cycles, the campaign
              child's capture chain incl. p58) on its option-on configuration `srp1grid_x0_a05_on_v1_noq`, with the new
              capture entered beside the interface-dual capture (the probe's chain is otherwise unchanged); then the
              sidecar is checked: header + one line per cycle, captured, every declared field present with its
              declared shape, and the per-block Q sums reconciled with production's own per-cycle
              `gross_operational_cost` (primary, settlement excluded in full) and
              `gross_operational_cost_contracted_excluded`.
Both modes: zero solves (the probe module's SolveProfileGuard(permitted=()) is installed at import and verified at 0).

Usage (repo root, canonical interpreter, attached, both streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --grouping --pre-rev <rev> \
      --out-dir <new dir>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --capture --out-dir <new dir>
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s56_step1_completion_checks as PC  # noqa: E402 -- installs the zero-solve guard (via the gate script)

import gate_result_io as GRIO  # noqa: E402

GS = PC.GS
H = PC.H
SCHEMA = 'p515_s56_step4_capture_check_v1'
CAPTURE_CONFIG = 'srp1grid_x0_a05_on_v1_noq'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _dump(path, obj):
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)


# ======================================================================================================================
#  --grouping
# ======================================================================================================================
def load_pre_harness(rev, scratch):
    text = subprocess.run(['git', 'show', f'{rev}:p515_s44_campaign_harness.py'], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout
    path = os.path.join(scratch, 'harness_pre.py')
    with open(path, 'x') as handle:
        handle.write(text)
    spec = importlib.util.spec_from_file_location('harness_pre', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = H.REPO
    return mod


def _record(hex_value):
    v = float.fromhex(hex_value)
    comps = {k: v if k == 'gross_operational_cost' else 0.0 for k in H.INIT_IDENTITY_FIELDS}
    return {'definition': 'synthetic (zero-solve grouping check)', 'gross_operational_cost': v,
            'gross_operational_cost_hex': hex_value, 'components': comps,
            'components_hex': {k: float.hex(x) for k, x in comps.items()}}


def _call(mod, root, eval_name, hex_value, coupling, cand='cand_A', derived=None):
    eval_dir = os.path.join(root, 'evals', eval_name)
    os.makedirs(eval_dir)
    kwargs = {} if coupling is None else {'interface_coupling': coupling}
    out = mod.register_initialisation_identity(_record(hex_value), root, eval_dir, eval_name, cand, derived, **kwargs)
    with open(os.path.join(root, mod.INIT_IDENTITY_DIR_NAME, f'{eval_name}.json')) as handle:
        written = json.load(handle)
    return out, written


def _norm(obj, root):
    text = json.dumps(obj, sort_keys=True)
    return text.replace(os.path.relpath(root, REPO), '<ROOT>')


def grouping(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    scratch = os.path.join(out_dir, 'scratch')
    os.makedirs(scratch)
    pre = load_pre_harness(args.pre_rev, scratch)
    post = H
    on_pin = H.validate_interface_coupling({'per_scenario_interface_coupling': True,
                                            'per_scenario_voltage_mode': 'pin', 'row18_charge_reactive': True})
    on_v1 = H.validate_interface_coupling({'per_scenario_interface_coupling': True,
                                           'per_scenario_voltage_mode': 'per_scenario',
                                           'row18_charge_reactive': False})
    h1, h2 = float.hex(123456.25), float.hex(123456.5)
    res = {'schema': SCHEMA + '_grouping', 'utc': _utc(), 'pre_rev': args.pre_rev,
           'pre_sha256': GS._sha_file(os.path.join(scratch, 'harness_pre.py')),
           'post_sha256': GS._sha_file(os.path.join(REPO, 'p515_s44_campaign_harness.py')),
           'provenance': GS.provenance(), 'cases': {}}
    # (a) option-off roots: byte identity of the comparison and the written record
    for case, seq in (('off_equal', [('e1', h1), ('e2', h1)]), ('off_unequal', [('e1', h1), ('e2', h2)]),
                      ('off_other_candidate', [('e1', h1), ('e2', h1)])):
        per = {}
        for label, mod in (('pre', pre), ('post', post)):
            root = os.path.join(scratch, f'a_{case}_{label}')
            outs = []
            for i, (name, hx) in enumerate(seq):
                cand = 'cand_B' if (case == 'off_other_candidate' and i == 1) else 'cand_A'
                o, w = _call(mod, root, name, hx, None, cand=cand)
                w.pop('utc', None)
                w.pop('pid', None)
                outs.append({'returned': _norm(o, root), 'written': _norm(w, root)})
            per[label] = outs
        res['cases'][case] = {'identical_pre_post': per['pre'] == per['post'],
                              'post_last_returned': json.loads(per['post'][-1]['returned'])}
    # (b) mixed root
    for label, mod in (('pre', pre), ('post', post)):
        root = os.path.join(scratch, f'b_mixed_{label}')
        seq = [('off', h1, None), ('on_pin', h1, on_pin), ('on_v1', h2, on_v1), ('on_pin_diff', h2, on_pin)]
        outs = {}
        for name, hx, cpl in seq:
            o, w = _call(mod, root, name, hx, cpl)
            outs[name] = {'n_compared': o['n_compared'], 'compared': [c['file'] for c in o['compared']],
                          'not_compared': [c['file'] for c in o['other_candidates_not_compared']],
                          'all_equal': o['all_equal'], 'mismatches': [c['file'] for c in o['mismatches']],
                          'group_returned': o.get('initialisation_identity_group'),
                          'group_written': w.get('initialisation_identity_group')}
        res['cases'][f'mixed_{label}'] = outs
    post_m = res['cases']['mixed_post']
    res['checks'] = {
        'a_off_equal_identical_pre_post': res['cases']['off_equal']['identical_pre_post'],
        'a_off_unequal_identical_pre_post': res['cases']['off_unequal']['identical_pre_post'],
        'a_off_other_candidate_identical_pre_post': res['cases']['off_other_candidate']['identical_pre_post'],
        'b_post_on_pin_compared_with_off_and_equal': (post_m['on_pin']['compared'] == ['off.json']
                                                     and post_m['on_pin']['all_equal'] is True),
        'b_post_on_v1_separate': (post_m['on_v1']['n_compared'] == 0
                                  and set(post_m['on_v1']['not_compared']) == {'off.json', 'on_pin.json'}),
        'b_post_on_pin_diff_mismatches_off_and_on_pin': (set(post_m['on_pin_diff']['mismatches'])
                                                         == {'off.json', 'on_pin.json'}
                                                         and post_m['on_pin_diff']['all_equal'] is False),
        'b_post_groups_recorded_for_on_only': (post_m['off']['group_written'] is None
                                               and post_m['on_pin']['group_written'] == H.INIT_IDENTITY_GROUP_PIN
                                               and post_m['on_v1']['group_written'] == H.INIT_IDENTITY_GROUP_NO_PIN),
        'b_pre_no_cross_declaration_comparison': (res['cases']['mixed_pre']['on_pin']['n_compared'] == 0),
    }
    res['holds'] = all(res['checks'].values())
    res['guard'] = GS.guard_report()
    _dump(os.path.join(out_dir, 'grouping_check.json'), res)
    print(f"[grouping] holds={res['holds']} checks={res['checks']} guard_zero={res['guard']['verified_zero']}",
          flush=True)
    return 0 if res['holds'] and res['guard']['verified_zero'] else 2


# ======================================================================================================================
#  --capture
# ======================================================================================================================
def capture(args):
    import interface_dual_capture as IDC
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    res = {'schema': SCHEMA + '_capture', 'utc': _utc(), 'config': CAPTURE_CONFIG,
           'declaration': GS.BUILD_CONFIGS[CAPTURE_CONFIG], 'provenance': GS.provenance(), 'error': None}
    t0 = time.time()
    try:
        res['capture_path_checklist'] = H.assert_s56_cycle_capture_path()
        original = IDC.interface_dual_capture_hooks
        seen = {}

        @contextmanager
        def composite(eval_dir, holder):
            with original(eval_dir, holder) as dual, H.s56_cycle_capture_hooks(eval_dir, holder):
                seen['eval_dir'] = eval_dir
                seen['holder'] = holder
                yield dual
        IDC.interface_dual_capture_hooks = composite
        try:
            rc = PC.probe(argparse.Namespace(config=CAPTURE_CONFIG, phase='post', out_dir=out_dir))
        finally:
            IDC.interface_dual_capture_hooks = original
        res['probe_exit'] = rc
        with open(os.path.join(out_dir, f'probe_post_{CAPTURE_CONFIG}.json')) as handle:
            probe = json.load(handle)
        res['probe'] = {k: probe.get(k) for k in ('error', 'run_error', 'n_cycles', 'guard', 'stub_calls')}
        res['summary'] = (seen.get('holder') or {}).get(H.S56_CYCLE_CAPTURE_SUMMARY_KEY)
        path = os.path.join(seen['eval_dir'], H.S56_CYCLE_CAPTURE_FILE)
        lines = [json.loads(t) for t in open(path) if t.strip()]
        header = lines[0]
        cycles = lines[1:]
        res['sidecar'] = {'path': os.path.relpath(path, REPO), 'sha256': GS._sha_file(path),
                          'bytes': os.path.getsize(path), 'n_lines': len(lines),
                          'bytes_per_cycle_line': [len(json.dumps(c)) for c in cycles]}
        nb, nt = header['n_blocks'], header['n_periods']
        n_s = len(header['deviation_scenarios'])
        kinds = header['deviation_kinds']
        checks = {'header_first': header.get('header') is True,
                  'one_line_per_cycle': len(cycles) == probe.get('n_cycles') == PC.N_CYCLES,
                  'every_line_captured_no_error': all(c['captured'] and c['error'] is None for c in cycles),
                  'deviation_kinds_p_q_v': kinds == ['p', 'q', 'v']}
        for c in cycles:
            k = c['cycle']
            for side in ('tso', 'dso'):
                for kind in ('p', 'q', 'v'):
                    arr = c['primal_expectation'][side][kind]
                    checks[f'c{k}:primal_expectation:{side}:{kind}'] = (len(arr) == nb
                                                                       and all(len(a) == nt for a in arr))
                for kind in kinds:
                    arr = c['primal_deviation'][side][kind]
                    checks[f'c{k}:primal_deviation:{side}:{kind}'] = (len(arr) == nb and all(
                        len(a) == n_s and all(len(x) == nt for x in a) for a in arr))
                arr = c['dual_expectation_v'][side]
                checks[f'c{k}:dual_expectation_v:{side}'] = len(arr) == nb and all(len(a) == nt for a in arr)
            checks[f'c{k}:rho_every_network_block'] = len(c['rho']) == header['n_network_blocks'] and all(
                set(v) >= {'rho_pf', 'rho_v'} for v in c['rho'].values())
            checks[f'c{k}:tail_every_holder'] = set(c['tail_compl_inf_tol']) == set(header['tail_holders'])
            diag = probe['admm_diagnostics'][k - 1]['values']
            q = c['q_blocks']
            prim = sum(v['settlement_excluded_in_full'] for v in q['per_year_day'].values())
            cont = sum(v['contracted_excluded'] for v in q['per_year_day'].values())
            res.setdefault('reconciliation', {})[f'cycle{k}'] = {
                'sum_blocks_settlement_excluded_in_full': prim,
                'production_gross_operational_cost': diag.get('gross_operational_cost'),
                'sum_blocks_contracted_excluded': cont,
                'production_gross_operational_cost_contracted_excluded':
                    diag.get('gross_operational_cost_contracted_excluded')}
            r = res['reconciliation'][f'cycle{k}']
            for a, b in (('sum_blocks_settlement_excluded_in_full', 'production_gross_operational_cost'),
                         ('sum_blocks_contracted_excluded', 'production_gross_operational_cost_contracted_excluded')):
                rel = abs(r[a] - r[b]) / max(1.0, abs(r[b])) if r[b] is not None else None
                r[f'rel_{a}'] = rel
                checks[f'c{k}:{a}_reconciles'] = rel is not None and rel <= 1e-12
        idc_path = os.path.join(seen['eval_dir'], IDC.INTERFACE_DUAL_FILE)
        idc_lines = [json.loads(t) for t in open(idc_path) if t.strip()]
        idc_fields = set().union(*[set(x) for x in idc_lines if not x.get('header')])
        need = {'lambda_pf_p_tso', 'lambda_pf_p_dso', 'lambda_pf_q_tso', 'lambda_pf_q_dso'} | {
            f'lambda_dev_{k}_{s}' for k in kinds for s in ('tso', 'dso')}
        checks['idc_sidecar_carries_pf_and_every_deviation_dual'] = need <= idc_fields
        res['idc_fields'] = sorted(idc_fields)
        res['checks'] = checks
        res['holds'] = all(checks.values()) and rc == 0 and (res['summary'] or {}).get('ok') is True
    except Exception as error:  # noqa: BLE001
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        res['holds'] = False
    res['guard'] = GS.guard_report()
    res['wall_s'] = time.time() - t0
    _dump(os.path.join(out_dir, 'capture_check.json'), res)
    failed = sorted(k for k, v in (res.get('checks') or {}).items() if not v)
    print(f"[capture] holds={res['holds']} error={res['error']} failed={failed[:10]} "
          f"guard_zero={res['guard']['verified_zero']}", flush=True)
    return 0 if res['holds'] and res['guard']['verified_zero'] else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grouping', action='store_true')
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--pre-rev')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    if os.environ.get('NLP_SOLVER_PATH') != GS.SOLVER_PATH:
        print(f'REFUSED: NLP_SOLVER_PATH must be {GS.SOLVER_PATH}', flush=True)
        return 2
    if args.grouping:
        if not args.pre_rev:
            ap.error('--grouping needs --pre-rev')
        return grouping(args)
    if args.capture:
        return capture(args)
    ap.error('one of --grouping / --capture')


if __name__ == '__main__':
    sys.exit(main())
