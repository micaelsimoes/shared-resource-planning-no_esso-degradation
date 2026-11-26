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

S56 step 4 PREPARATION (Planner task 2026-11-23, pinned-arm capture; S56_STEP4_DESIGN.md item 7) adds two modes:
  --arm {off_key, v1, v2}  The per-cycle S56 capture on stubbed cycles, entered exactly as the campaign child now
              selects it (`s56_cycle_capture_arm` -> `assert_s56_cycle_capture_path(arm)` -> `s56_cycle_capture_hooks(
              ..., arm)`), beside the interface-dual capture, in the zero-solve probe flow:
                off_key  option off WITH the new key (`S56_CYCLE_CAPTURE_PINNED_KEY` True), config `srp1grid_x0_a05`
                         (pinned arm): check (a) -- the capture-path assertion holds for the pinned field set, the
                         sidecar carries exactly that field set (no deviation channel), and the per-block Q
                         (contracted-excluded) sums to production's per-cycle `gross_operational_cost` (rel 1e-12);
                v1 / v2  option on, 'per_scenario' (`srp1grid_x0_a05_on_v1_noq`) / 'pin' (`srp1grid_x0_a05_on_pin`):
                         check (b) -- the new per-cycle voltage-pin field is present and its per-block sum equals
                         production's terminal `voltage_pin_total` (rel 1e-12; V-1 additionally exactly 0). v1 also
                         compares its sidecar with the committed v1 check's sidecar (`--reference-sidecar`, hash
                         verified against `--reference-manifest`): every pre-existing field identical (wall time
                         `capture_s` aside), new fields additions only.
              Every mode also reconciles the per-block Q in both conventions, as `--capture` did.
  --nokey     check (c), zero solves, no probe run: (c1) the campaign child `_child_real` before (`--pre-rev`) vs
              after: no line deleted, every inserted line listed; (c2) the inserted statements' values for entries
              WITHOUT the key (option off, option off with an explicit default coupling, V-1, V-2): no pinned capture
              selected, no record field added; (c3) `freeze_campaign_spec` before vs after on the same entries:
              identical specs (the timestamp and the harness' own path / sha256 aside); with the key the entry
              records it and its eval key is unchanged; (c4) the r5 zero-solve probe's scratch evaluation dirs
              (`--r5-dir`), pre vs post, per configuration: the same file set and every file byte-identical after
              dropping only the run-to-run wall-time / memory fields (`R5_VOLATILE_KEYS`) and the initialisation
              record's `comparison` (it lists whatever records the shared root holds).

Usage (repo root, canonical interpreter, attached, both streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --grouping --pre-rev <rev> \
      --out-dir <new dir>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --capture --out-dir <new dir>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --arm off_key --out-dir <new dir>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --arm v1 --out-dir <new dir> \
      --reference-sidecar <v1 sidecar> --reference-manifest <its manifest>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --arm v2 --out-dir <new dir>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_capture_check.py --nokey --pre-rev <rev> \
      --r5-dir <probe r5 dir> --out-dir <new dir>
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


# ======================================================================================================================
#  S56 step 4 preparation: --arm {off_key, v1, v2}
# ======================================================================================================================
ARM_CONFIGS = {'off_key': 'srp1grid_x0_a05', 'v1': 'srp1grid_x0_a05_on_v1_noq', 'v2': 'srp1grid_x0_a05_on_pin'}
REL_TOL = 1e-12


def _rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def _entry_for(arm_name):
    """The campaign entry the arm's configuration corresponds to (only the keys the child's selection reads)."""
    cfg = GS.BUILD_CONFIGS[ARM_CONFIGS[arm_name]]
    entry = {}
    if cfg['coupling'] is not None:
        entry['interface_coupling'] = cfg['coupling']
    if arm_name == 'off_key':
        entry[H.S56_CYCLE_CAPTURE_PINNED_KEY] = True
    return entry


def _select(entry):
    """The campaign child's own selection, in its order (`_child_real`): coupling validated, then the key."""
    coupling_on = H.interface_coupling_in_key(H.validate_interface_coupling(entry.get('interface_coupling'))) is not None
    pinned = H.validate_s56_cycle_capture_pinned(entry.get(H.S56_CYCLE_CAPTURE_PINNED_KEY))
    return coupling_on, pinned, H.s56_cycle_capture_arm(coupling_on, pinned)


def _strip_capture_s(obj):
    return {k: v for k, v in obj.items() if k != 'capture_s'}


def compare_with_reference_sidecar(new_lines, ref_path, manifest_path):
    """Coupled-arm additions-only check: every key of every reference line exists in the new line with an identical
    value (the per-line wall time `capture_s` aside); keys only in the new lines are listed."""
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    rel = os.path.relpath(os.path.abspath(ref_path), REPO)
    recorded = (manifest.get('not_committed_hash_recorded') or {}).get(rel) or (manifest.get('committed') or {}).get(rel)
    actual = GS._sha_file(os.path.abspath(ref_path))
    ref_lines = [json.loads(t) for t in open(ref_path) if t.strip()]
    out = {'reference': rel, 'reference_sha256': actual, 'manifest': os.path.relpath(os.path.abspath(manifest_path), REPO),
           'manifest_sha256_recorded': recorded, 'reference_hash_verified': recorded == actual,
           'n_lines_reference': len(ref_lines), 'n_lines_new': len(new_lines), 'per_line': []}
    ok = recorded == actual and len(ref_lines) == len(new_lines)
    for i, (ref, new) in enumerate(zip(ref_lines, new_lines)):
        ref_s, new_s = _strip_capture_s(ref), _strip_capture_s(new)
        missing = sorted(k for k in ref_s if k not in new_s)
        changed = sorted(k for k in ref_s if k in new_s and json.dumps(ref_s[k], sort_keys=True)
                         != json.dumps(new_s[k], sort_keys=True))
        added = sorted(k for k in new_s if k not in ref_s)
        out['per_line'].append({'line': i, 'missing_in_new': missing, 'changed': changed, 'added': added})
        ok = ok and not missing and not changed
    out['every_existing_field_identical_additions_only'] = ok
    return out


def arm_check(args):
    import interface_dual_capture as IDC
    arm_name = args.arm
    config = ARM_CONFIGS[arm_name]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    entry = _entry_for(arm_name)
    res = {'schema': SCHEMA + '_arm', 'utc': _utc(), 'arm_check': arm_name, 'config': config,
           'declaration': GS.BUILD_CONFIGS[config], 'entry_keys_read_by_selection': entry,
           'provenance': GS.provenance(), 'error': None}
    t0 = time.time()
    try:
        coupling_on, pinned, capture_arm = _select(entry)
        res['selection'] = {'coupling_on': coupling_on, 'pinned_key': pinned, 'capture_arm': capture_arm}
        expected_arm = H.S56_CAPTURE_ARM_PINNED if arm_name == 'off_key' else H.S56_CAPTURE_ARM_COUPLED
        if capture_arm != expected_arm:
            raise RuntimeError(f'selection gave {capture_arm!r}, expected {expected_arm!r}')
        res['capture_path_checklist'] = H.assert_s56_cycle_capture_path(arm=capture_arm)
        field_set = list(H.S56_CYCLE_CAPTURE_LINE_FIELDS_BY_ARM[capture_arm])
        res['declared_line_field_set'] = field_set
        original = IDC.interface_dual_capture_hooks
        seen = {}

        @contextmanager
        def composite(eval_dir, holder):
            with original(eval_dir, holder) as dual, H.s56_cycle_capture_hooks(eval_dir, holder, arm=capture_arm):
                seen['eval_dir'] = eval_dir
                seen['holder'] = holder
                yield dual
        IDC.interface_dual_capture_hooks = composite
        try:
            rc = PC.probe(argparse.Namespace(config=config, phase='post', out_dir=out_dir))
        finally:
            IDC.interface_dual_capture_hooks = original
        res['probe_exit'] = rc
        with open(os.path.join(out_dir, f'probe_post_{config}.json')) as handle:
            probe = json.load(handle)
        res['probe'] = {k: probe.get(k) for k in ('error', 'run_error', 'n_cycles', 'guard', 'stub_calls')}
        res['summary'] = (seen.get('holder') or {}).get(H.S56_CYCLE_CAPTURE_SUMMARY_KEY)
        path = os.path.join(seen['eval_dir'], H.S56_CYCLE_CAPTURE_FILE)
        lines = [json.loads(t) for t in open(path) if t.strip()]
        header, cycles = lines[0], lines[1:]
        res['sidecar'] = {'path': os.path.relpath(path, REPO), 'sha256': GS._sha_file(path),
                          'bytes': os.path.getsize(path), 'n_lines': len(lines),
                          'bytes_per_cycle_line': [len(json.dumps(c)) for c in cycles]}
        res['header_without_blocks'] = {k: v for k, v in header.items() if k != 'blocks'}
        nb, nt = header['n_blocks'], header['n_periods']
        checks = {'header_first': header.get('header') is True,
                  'header_capture_arm': header.get('capture_arm') == capture_arm,
                  'header_field_set_equals_declared': sorted(list(header.get('line_fields') or [])
                                                             + list(header.get('line_fields_added') or []))
                                                      == sorted(field_set),
                  'one_line_per_cycle': len(cycles) == probe.get('n_cycles') == PC.N_CYCLES,
                  'every_line_captured_no_error': all(c['captured'] and c['error'] is None for c in cycles),
                  'summary_ok': (res['summary'] or {}).get('ok') is True}
        if capture_arm == H.S56_CAPTURE_ARM_COUPLED:
            # production `_interface_deviation_kinds`: ('p', 'q', 'v') under 'per_scenario', ('p', 'q') under 'pin'.
            # (The first v2 run, arm_v2/, expected ['p', 'q', 'v'] for both modes -- a defect of this check, kept as
            # evidence; corrected here.)
            mode = GS.BUILD_CONFIGS[config]['coupling']['per_scenario_voltage_mode']
            expected_kinds = ['p', 'q', 'v'] if mode == 'per_scenario' else ['p', 'q']
            res['expected_deviation_kinds'] = expected_kinds
            checks['deviation_kinds_as_production_for_the_voltage_mode'] = header['deviation_kinds'] == expected_kinds
        terminal_rc = (probe.get('recourse_components') or {}).get('values') or {}
        res['reconciliation'] = {}
        for c in cycles:
            k = c['cycle']
            checks[f'c{k}:line_field_set_equals_declared'] = sorted(c.keys()) == sorted(field_set)
            for side in ('tso', 'dso'):
                for kind in ('p', 'q', 'v'):
                    arr = c['primal_expectation'][side][kind]
                    checks[f'c{k}:primal_expectation:{side}:{kind}'] = (len(arr) == nb
                                                                       and all(len(a) == nt for a in arr))
                if capture_arm == H.S56_CAPTURE_ARM_COUPLED:
                    n_s = len(header['deviation_scenarios'])
                    for kind in header['deviation_kinds']:
                        arr = c['primal_deviation'][side][kind]
                        checks[f'c{k}:primal_deviation:{side}:{kind}'] = (len(arr) == nb and all(
                            len(a) == n_s and all(len(x) == nt for x in a) for a in arr))
                arr = c['dual_expectation_v'][side]
                checks[f'c{k}:dual_expectation_v:{side}'] = len(arr) == nb and all(len(a) == nt for a in arr)
            checks[f'c{k}:rho_every_network_block'] = len(c['rho']) == header['n_network_blocks'] and all(
                set(v) >= {'rho_pf', 'rho_v'} for v in c['rho'].values())
            checks[f'c{k}:tail_every_holder'] = set(c['tail_compl_inf_tol']) == set(header['tail_holders'])
            q = c['q_blocks']
            checks[f'c{k}:q_per_agent_every_network_block'] = len(q['per_agent']) == header['n_network_blocks']
            checks[f'c{k}:q_salvage_block_recorded'] = isinstance(q.get('salvage_block'), float)
            diag = probe['admm_diagnostics'][k - 1]['values']
            prim = sum(v['settlement_excluded_in_full'] for v in q['per_year_day'].values())
            cont = sum(v['contracted_excluded'] for v in q['per_year_day'].values())
            r = {'sum_blocks_settlement_excluded_in_full': prim, 'sum_blocks_contracted_excluded': cont,
                 'production_gross_operational_cost': diag.get('gross_operational_cost'),
                 'production_gross_operational_cost_contracted_excluded':
                     diag.get('gross_operational_cost_contracted_excluded')}
            if capture_arm == H.S56_CAPTURE_ARM_PINNED:
                # option off: production's primary gross IS today's convention (contracted part and pin excluded)
                pairs = (('sum_blocks_contracted_excluded', 'production_gross_operational_cost'),)
                checks[f'c{k}:production_has_no_settlement_in_full_field'] = (
                    diag.get('gross_operational_cost_settlement_excluded_in_full') is None)
            else:
                pairs = (('sum_blocks_settlement_excluded_in_full', 'production_gross_operational_cost'),
                         ('sum_blocks_contracted_excluded', 'production_gross_operational_cost_contracted_excluded'))
            for a, b in pairs:
                rel = _rel(r[a], r[b]) if r[b] is not None else None
                r[f'rel_{a}_vs_{b}'] = rel
                checks[f'c{k}:{a}_reconciles_with_{b}'] = rel is not None and rel <= REL_TOL
            vp = c['voltage_pin']
            per_block_sum = sum(v['weighted'] for v in vp['per_agent'].values())
            r['voltage_pin_per_block_sum'] = per_block_sum
            r['voltage_pin_total_recorded'] = vp['total']
            r['voltage_pin_n_blocks_constructed'] = sum(1 for v in vp['per_agent'].values() if v['pin_constructed'])
            checks[f'c{k}:voltage_pin_present_every_network_block'] = (
                len(vp['per_agent']) == header['n_network_blocks'])
            checks[f'c{k}:voltage_pin_per_year_day_sums_to_total'] = _rel(sum(vp['per_year_day'].values()),
                                                                           vp['total']) <= REL_TOL
            res['reconciliation'][f'cycle{k}'] = r
        last = cycles[-1]['voltage_pin']
        term = terminal_rc.get('voltage_pin_total')
        last_sum = sum(v['weighted'] for v in last['per_agent'].values())
        res['terminal_voltage_pin'] = {'production_voltage_pin_total': term,
                                       'production_voltage_pin_total_hex': float.hex(term) if term is not None else None,
                                       'capture_last_cycle_per_block_sum': last_sum,
                                       'capture_last_cycle_total': last['total'],
                                       'capture_last_cycle_total_hex': float.hex(last['total']),
                                       'rel_per_block_sum': _rel(last_sum, term) if term is not None else None,
                                       'total_bitwise_equal': term is not None and last['total'] == term}
        checks['terminal:voltage_pin_per_block_sum_equals_production_total'] = (
            term is not None and _rel(last_sum, term) <= REL_TOL)
        if arm_name == 'v1':
            checks['v1:voltage_pin_zero_every_cycle'] = all(c['voltage_pin']['total'] == 0.0 for c in cycles)
            checks['v1:voltage_pin_not_constructed_on_any_block'] = all(
                not v['pin_constructed'] for c in cycles for v in c['voltage_pin']['per_agent'].values())
        else:
            checks[f'{arm_name}:voltage_pin_constructed_on_every_block'] = all(
                v['pin_constructed'] for c in cycles for v in c['voltage_pin']['per_agent'].values())
        if capture_arm == H.S56_CAPTURE_ARM_PINNED:
            checks['pinned:no_deviation_channel_in_header'] = header['deviation_kinds'] == []
        idc_path = os.path.join(seen['eval_dir'], IDC.INTERFACE_DUAL_FILE)
        idc_lines = [json.loads(t) for t in open(idc_path) if t.strip()]
        idc_fields = set().union(*[set(x) for x in idc_lines if not x.get('header')])
        checks['idc_sidecar_carries_expectation_pf_duals_both_sides'] = {
            'lambda_pf_p_tso', 'lambda_pf_p_dso', 'lambda_pf_q_tso', 'lambda_pf_q_dso'} <= idc_fields
        res['idc_fields'] = sorted(idc_fields)
        if args.reference_sidecar:
            res['additions_only_vs_reference'] = compare_with_reference_sidecar(lines, args.reference_sidecar,
                                                                                args.reference_manifest)
            checks['coupled:every_existing_field_identical_additions_only'] = (
                res['additions_only_vs_reference']['every_existing_field_identical_additions_only'])
        res['checks'] = checks
        res['holds'] = all(checks.values()) and rc == 0
    except Exception as error:  # noqa: BLE001
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        res['holds'] = False
    res['guard'] = GS.guard_report()
    res['wall_s'] = time.time() - t0
    _dump(os.path.join(out_dir, f'arm_check_{arm_name}.json'), res)
    failed = sorted(k for k, v in (res.get('checks') or {}).items() if not v)
    print(f"[arm {arm_name}] holds={res['holds']} error={res['error']} failed={failed[:10]} "
          f"terminal_pin={res.get('terminal_voltage_pin')} guard_zero={res['guard']['verified_zero']}", flush=True)
    return 0 if res['holds'] and res['guard']['verified_zero'] else 2


# ======================================================================================================================
#  S56 step 4 preparation: --nokey (check (c))
# ======================================================================================================================
R5_CONFIGS = PC.PROBE_CONFIGS
# Run-to-run wall-time / memory fields of the probe's sidecars (identified on the committed r4 pre/post pair, where
# they were the ONLY differing values apart from the initialisation record's `comparison`).
R5_VOLATILE_KEYS = frozenset({'capture_s', 'rss_bytes', 'response_capture_s', 'cycle_wall_s', 'ru_maxrss_bytes'})


def _drop_volatile(obj, top=True, name=None):
    if isinstance(obj, dict):
        return {k: _drop_volatile(v, False) for k, v in obj.items()
                if k not in R5_VOLATILE_KEYS and not (top and name == 'initialisation_identity.json'
                                                      and k == 'comparison')}
    if isinstance(obj, list):
        return [_drop_volatile(v, False) for v in obj]
    return obj


def _load_json_or_jsonl(path):
    with open(path) as handle:
        text = handle.read()
    if path.endswith('.jsonl'):
        return [json.loads(t) for t in text.splitlines() if t.strip()]
    return json.loads(text)


def _child_source_diff(pre, post):
    import difflib
    import inspect
    a = inspect.getsource(pre._child_real).splitlines()
    b = inspect.getsource(post._child_real).splitlines()
    deleted, inserted = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ('replace', 'delete'):
            deleted.extend(a[i1:i2])
        if tag in ('replace', 'insert'):
            inserted.extend(b[j1:j2])
    return {'n_lines_pre': len(a), 'n_lines_post': len(b), 'deleted': deleted, 'inserted': inserted,
            'unified_diff': list(difflib.unified_diff(a, b, 'pre:_child_real', 'post:_child_real', lineterm='', n=1))}


def _freeze(mod, root, cid, candidates):
    path, digest, spec = mod.freeze_campaign_spec(
        root, cid, candidates, {'name': 'S56 step 4 prep zero-solve spec check', 'overrides': {}}, 3, 1,
        ['zero-solve check (c3)'], required_consecutive_cycles=10)
    return spec


def _spec_norm(spec):
    s = json.loads(json.dumps(spec, sort_keys=True, default=str))
    s.pop('frozen_utc', None)
    s.pop('harness', None)
    return s


def nokey(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    scratch = os.path.join(out_dir, 'scratch')
    os.makedirs(scratch)
    res = {'schema': SCHEMA + '_nokey', 'utc': _utc(), 'pre_rev': args.pre_rev, 'provenance': GS.provenance(),
           'error': None, 'checks': {}}
    checks = res['checks']
    try:
        pre = load_pre_harness(args.pre_rev, scratch)
        pre.CASE_FILE = H.CASE_FILE
        res['pre_sha256'] = GS._sha_file(os.path.join(scratch, 'harness_pre.py'))
        res['post_sha256'] = GS._sha_file(os.path.join(REPO, 'p515_s44_campaign_harness.py'))
        # (c1) the child, before vs after
        c1 = _child_source_diff(pre, H)
        res['c1_child_source'] = c1
        checks['c1:no_line_of_the_child_deleted'] = not c1['deleted']
        # (c2) the inserted statements for entries WITHOUT the key
        default_cpl = dict(GS.COUPLING_DEFAULT)
        v2 = dict(GS.COUPLING_V1_NOQ, per_scenario_voltage_mode='pin')
        entries = {'off_no_options': {}, 'off_explicit_default_coupling': {'interface_coupling': default_cpl},
                   'v1': {'interface_coupling': dict(GS.COUPLING_V1_NOQ)}, 'v2': {'interface_coupling': v2}}
        c2 = {}
        for name, entry in entries.items():
            coupling_on, pinned, arm = _select(entry)
            fields = H.s56_cycle_capture_pinned_record_fields(entry, {}, arm)
            c2[name] = {'coupling_on': coupling_on, 'pinned_key': pinned, 'capture_arm': arm,
                        'pinned_record_fields': fields}
            checks[f'c2:{name}:no_pinned_capture_selected'] = arm != H.S56_CAPTURE_ARM_PINNED
            checks[f'c2:{name}:no_record_field_added'] = fields == {}
            checks[f'c2:{name}:arm_as_before'] = arm == (H.S56_CAPTURE_ARM_COUPLED if coupling_on else None)
        try:
            H.s56_cycle_capture_arm(True, True)
            checks['c2:key_refused_on_a_coupled_entry'] = False
        except ValueError as error:
            c2['key_on_coupled_refusal'] = str(error)
            checks['c2:key_refused_on_a_coupled_entry'] = True
        res['c2'] = c2
        # (c3) freeze_campaign_spec before vs after (no key), and with the key
        x0 = GS.CANDIDATES['x0']
        cands = [('x0_off', x0, {}), ('x0_off_aa', x0, {'overrides': {'anderson_acceleration': {'enabled': False}}}),
                 ('x0_v1', x0, {'interface_coupling': dict(GS.COUPLING_V1_NOQ)}), ('x0_v2', x0, {'interface_coupling': v2})]
        # an explicit default coupling has the SAME eval key as no coupling (not keyed), so it is frozen in a campaign
        # of its own (the first nokey run, nokey/, put it beside x0_off and was refused as a duplicate key -- a defect
        # of this check, kept as evidence)
        cands_dflt = [('x0_off_dflt', x0, {'interface_coupling': default_cpl})]
        specs, specs_dflt = {}, {}
        for label, mod in (('pre', pre), ('post', H)):
            specs[label] = _spec_norm(_freeze(mod, os.path.join(scratch, f'spec_{label}'), 's56s4prep_c3', cands))
            specs_dflt[label] = _spec_norm(_freeze(mod, os.path.join(scratch, f'spec_dflt_{label}'), 's56s4prep_c3d',
                                                   cands_dflt))
        c3 = {'identical_pre_post': specs['pre'] == specs['post'] and specs_dflt['pre'] == specs_dflt['post'],
              'differences': PC._diff(specs['pre'], specs['post'])[:50],
              'differences_explicit_default_coupling': PC._diff(specs_dflt['pre'], specs_dflt['post'])[:50]}
        keyed = _spec_norm(_freeze(H, os.path.join(scratch, 'spec_post_key'), 's56s4prep_c3', [
            ('x0_off', x0, {H.S56_CYCLE_CAPTURE_PINNED_KEY: True}),
            ('x0_off_aa', x0, {'overrides': {'anderson_acceleration': {'enabled': False}},
                               H.S56_CYCLE_CAPTURE_PINNED_KEY: True})]))
        by_label = {e['label']: e for e in specs['post']['candidates']}
        c3['with_key'] = {e['label']: {'eval_key_equals_no_key': e['eval_key'] == by_label[e['label']]['eval_key'],
                                       'entry_records_key': e.get(H.S56_CYCLE_CAPTURE_PINNED_KEY) is True,
                                       'entry_otherwise_identical': {k: v for k, v in e.items()
                                                                     if k != H.S56_CYCLE_CAPTURE_PINNED_KEY}
                                       == by_label[e['label']]}
                          for e in keyed['candidates']}
        try:
            _freeze(H, os.path.join(scratch, 'spec_post_key_on_coupled'), 's56s4prep_c3',
                    [('x0_v1', x0, {'interface_coupling': dict(GS.COUPLING_V1_NOQ),
                                    H.S56_CYCLE_CAPTURE_PINNED_KEY: True})])
            c3['key_on_coupled_refused_at_freeze'] = False
        except ValueError as error:
            c3['key_on_coupled_refused_at_freeze'] = str(error)
        res['c3'] = c3
        checks['c3:spec_identical_pre_post_without_key'] = c3['identical_pre_post']
        for label, v in c3['with_key'].items():
            checks[f'c3:{label}:key_recorded_eval_key_unchanged_entry_otherwise_identical'] = all(v.values())
        checks['c3:key_on_coupled_refused_at_freeze'] = c3['key_on_coupled_refused_at_freeze'] is not False
        # (c4) the r5 probe scratch evaluation dirs, pre vs post
        r5 = os.path.abspath(args.r5_dir)
        c4 = {}
        for config in R5_CONFIGS:
            da = os.path.join(r5, f'scratch_pre_{config}', f'eval_pre_{config}')
            db = os.path.join(r5, f'scratch_post_{config}', f'eval_post_{config}')
            fa, fb = sorted(os.listdir(da)), sorted(os.listdir(db))
            per = {'files_pre': fa, 'same_file_set': fa == fb, 'raw_byte_identical': [], 'identical_after_volatile': [],
                   'differ': []}
            for name in fa:
                if name not in fb:
                    continue
                pa, pb = os.path.join(da, name), os.path.join(db, name)
                if GS._sha_file(pa) == GS._sha_file(pb):
                    per['raw_byte_identical'].append(name)
                elif (json.dumps(_drop_volatile(_load_json_or_jsonl(pa), name=name), sort_keys=True)
                      == json.dumps(_drop_volatile(_load_json_or_jsonl(pb), name=name), sort_keys=True)):
                    per['identical_after_volatile'].append(name)
                else:
                    per['differ'].append(name)
            c4[config] = per
            checks[f'c4:{config}:same_file_set_and_identical'] = per['same_file_set'] and not per['differ']
        res['c4'] = c4
        res['c4_volatile_keys'] = sorted(R5_VOLATILE_KEYS)
    except Exception as error:  # noqa: BLE001
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
    res['holds'] = res['error'] is None and bool(checks) and all(checks.values())
    res['guard'] = GS.guard_report()
    _dump(os.path.join(out_dir, 'nokey_check.json'), res)
    failed = sorted(k for k, v in checks.items() if not v)
    print(f"[nokey] holds={res['holds']} error={res['error']} failed={failed[:10]} "
          f"guard_zero={res['guard']['verified_zero']}", flush=True)
    return 0 if res['holds'] and res['guard']['verified_zero'] else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grouping', action='store_true')
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--arm', choices=sorted(ARM_CONFIGS))
    ap.add_argument('--nokey', action='store_true')
    ap.add_argument('--pre-rev')
    ap.add_argument('--r5-dir')
    ap.add_argument('--reference-sidecar')
    ap.add_argument('--reference-manifest')
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
    if args.arm:
        if bool(args.reference_sidecar) != bool(args.reference_manifest):
            ap.error('--reference-sidecar and --reference-manifest go together')
        return arm_check(args)
    if args.nokey:
        if not (args.pre_rev and args.r5_dir):
            ap.error('--nokey needs --pre-rev and --r5-dir')
        return nokey(args)
    ap.error('one of --grouping / --capture / --arm / --nokey')


if __name__ == '__main__':
    sys.exit(main())
