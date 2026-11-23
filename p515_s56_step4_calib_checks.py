"""P5.15 S56 step 4 CALIBRATION RE-RUN -- ZERO-SOLVE checks of the two harness keys (Expert handback 3 decision 1 (a),
2026-11-23; Planner task of 2026-11-23): `s56_certification_stop_disabled` and `s56_interface_v_multiplier_capture`
(`p515_s44_campaign_harness`, both default absent, never keyed). Every mode is zero-solve: the gate module's blocking
SolveProfileGuard(permitted=()) is installed at import (through `p515_s56_step1_completion_checks`) and verified at 0.

  --nokey --pre-rev R   OPTION-OFF IDENTITY of the harness (no probe run):
                          (c1) `_child_real` and the whole harness module before (`git show R:...`) vs after: the
                               deleted lines are exactly the one declared (`'nrf_capture_error': nrf_error}` -- the
                               outcome dict's closing line, re-written with the added key); every inserted line listed;
                          (c2) entries WITHOUT the keys (option off, V-1, V-2): nothing selected, no record field;
                               True on an option-off entry refused; non-bool refused;
                          (c3) `freeze_campaign_spec` before vs after on entries without the keys: identical specs
                               (timestamp and the harness' own path / sha256 aside); with the keys on V-1: the eval key
                               is unchanged and the entry otherwise identical; keys on an option-off entry refused.
  --stub-run --keys off|on   One STUBBED run on instance A (the step 4 derived instance 2025|Winter, x0, V-1 coupling
                          'per_scenario', no row 18, alpha 0; spec v2's configuration through the gate module's
                          `spec_like`): every network / ESSO `optimize` replaced by a stub that (i) at ADMM cycles 1-3
                          writes every TSO / DSO block's NL file (plain, labels off -- the production form) and hashes
                          it, BEFORE moving the point, (ii) sets a deterministic stand-in point, (iii) puts deterministic
                          synthetic values into the import suffixes production reads (`dual` of the interface rows
                          `voltage_magnitude_upper_cons`, `ipopt_zU_out` of vmag_sqr / vmag / e at the interface buses)
                          and records them; production's `get_admm_boyd_residual_metrics` is wrapped (innermost) to
                          report `all_boyd_pass` True, so production's own certificate (10 consecutive) would stop the
                          run at cycle 10; cap 12. The campaign child's capture chain is entered (interface-dual
                          capture, S56 per-cycle capture) and, with --keys on, the two new hooks.
  --compare             off vs on: (a) off stops at cycle 10 (certificate), on runs to the cap 12, its in-memory
                          `required_consecutive_cycles` reads 10**9 every cycle and the length is restored to 10 after
                          the run; (b) every per-cycle diagnostics value (float.hex) of cycles 1-10 identical except
                          the declared `required_consecutive_cycles`; (c) NL IDENTITY: every hashed NL (cycles 1-3,
                          every block) identical; (d) the S56 per-cycle capture and the interface-dual sidecar lines
                          1-10 identical (wall time `capture_s` aside); (e) on: the multiplier sidecar has a header and
                          one captured line per cycle, and every captured interface value equals what the stub put in
                          the suffix / the Var (bitwise).

Usage (repo root, canonical interpreter, attached, both streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_calib_checks.py --nokey --pre-rev <rev> --out-dir D
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_calib_checks.py --stub-run --keys off --out-dir D
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_calib_checks.py --stub-run --keys on --out-dir D
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_calib_checks.py --compare --out-dir D
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s56_step1_completion_checks as PC  # noqa: E402 -- installs the zero-solve guard (via the gate script)
import p515_s56_step4_capture_check as SC  # noqa: E402 -- pre-harness loader, child diff, freeze helpers

import gate_result_io as GRIO  # noqa: E402

GS = PC.GS
H = PC.H
SCHEMA = 'p515_s56_step4_calib_checks_v1'
STEP4_SPEC_V2 = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'step4_main_v2',
                             'frozen_step4_main_spec_v2_7ca210fb.json')
STEP4_SPEC_V2_SHA256 = '7ca210fb'
STUB_CAP = 12
REQUIRED = GS.REQUIRED_CONSECUTIVE_CYCLES   # 10, = spec v2 required_consecutive_cycles
NL_CYCLES = (1, 2, 3)   # ADMM cycles whose NL files are hashed (network stub call n = cycle + 1)
DECLARED_DELETED_LINES = ("            'nrf_capture_error': nrf_error}",)
DECLARED_DIAG_DIFFERENCE = ('required_consecutive_cycles',)
KEYS = (H.S56_CERTIFICATION_STOP_DISABLED_KEY, H.S56_INTERFACE_V_MULTIPLIER_CAPTURE_KEY)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _dump(path, obj):
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)


def _sha(path):
    return GS._sha_file(path)


def derived_A():
    with open(STEP4_SPEC_V2, encoding='utf-8') as handle:
        text = handle.read()
    if not hashlib.sha256(text.encode()).hexdigest().startswith(STEP4_SPEC_V2_SHA256):
        raise RuntimeError('spec v2 hash differs')
    spec = json.loads(text)
    return H.validate_derived_instance(spec['i_arms_instances_cap']['instances']['A']['derived_instance'])


# ======================================================================================================================
#  --nokey
# ======================================================================================================================
def _module_diff(pre_path, post_path):
    import difflib
    with open(pre_path) as handle:
        a = handle.read().splitlines()
    with open(post_path) as handle:
        b = handle.read().splitlines()
    deleted, inserted = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ('replace', 'delete'):
            deleted.extend(a[i1:i2])
        if tag in ('replace', 'insert'):
            inserted.extend(b[j1:j2])
    return {'n_lines_pre': len(a), 'n_lines_post': len(b), 'deleted': deleted, 'n_inserted': len(inserted),
            'inserted': inserted}


def nokey(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    scratch = os.path.join(out_dir, 'scratch')
    os.makedirs(scratch)
    res = {'schema': SCHEMA + '_nokey', 'utc': _utc(), 'pre_rev': args.pre_rev, 'provenance': GS.provenance(),
           'declared_deleted_lines': list(DECLARED_DELETED_LINES), 'error': None, 'checks': {}}
    checks = res['checks']
    try:
        pre = SC.load_pre_harness(args.pre_rev, scratch)
        pre.CASE_FILE = H.CASE_FILE
        pre_path = os.path.join(scratch, 'harness_pre.py')
        post_path = os.path.join(REPO, 'p515_s44_campaign_harness.py')
        res['pre_sha256'], res['post_sha256'] = _sha(pre_path), _sha(post_path)
        c1 = SC._child_source_diff(pre, H)
        c1m = _module_diff(pre_path, post_path)
        res['c1_child_source'] = c1
        res['c1_module'] = c1m
        checks['c1:child_deleted_lines_are_the_declared_one'] = c1['deleted'] == list(DECLARED_DELETED_LINES)
        checks['c1:module_deleted_lines_are_the_declared_one'] = c1m['deleted'] == list(DECLARED_DELETED_LINES)
        checks['c1:declared_line_rewritten_with_added_key'] = (
            "            'nrf_capture_error': nrf_error," in c1m['inserted']
            and "            's56_calibration_error': s56_calibration_error}" in c1m['inserted'])
        # (c2)
        v2 = dict(GS.COUPLING_V1_NOQ, per_scenario_voltage_mode='pin')
        entries = {'off_no_options': ({}, False), 'off_pinned_key': ({H.S56_CYCLE_CAPTURE_PINNED_KEY: True}, False),
                   'v1': ({'interface_coupling': dict(GS.COUPLING_V1_NOQ)}, True),
                   'v2': ({'interface_coupling': v2}, True)}
        c2 = {}
        for name, (entry, coupling_on) in entries.items():
            sel = H.s56_calibration_keys_in_force(entry, coupling_on)
            fields = H.s56_calibration_record_fields(entry, {})
            c2[name] = {'keys_in_force': list(sel), 'record_fields': fields}
            checks[f'c2:{name}:nothing_selected'] = sel == (False, False)
            checks[f'c2:{name}:no_record_field_added'] = fields == {}
        for key in KEYS:
            try:
                H.s56_calibration_keys_in_force({key: True}, False)
                checks[f'c2:{key}:refused_on_option_off'] = False
            except ValueError as error:
                c2[f'{key}_option_off_refusal'] = str(error)
                checks[f'c2:{key}:refused_on_option_off'] = True
            try:
                H.s56_calibration_keys_in_force({key: 1}, True)
                checks[f'c2:{key}:non_bool_refused'] = False
            except ValueError:
                checks[f'c2:{key}:non_bool_refused'] = True
            checks[f'c2:{key}:False_selects_nothing'] = H.s56_calibration_keys_in_force({key: False}, True) == (
                False, False)
        res['c2'] = c2
        # (c3)
        x0 = GS.CANDIDATES['x0']
        cands = [('x0_off', x0, {}), ('x0_off_aa', x0, {'overrides': {'anderson_acceleration': {'enabled': False}}}),
                 ('x0_v1', x0, {'interface_coupling': dict(GS.COUPLING_V1_NOQ)}), ('x0_v2', x0, {'interface_coupling': v2}),
                 ('x0_off_pinned', x0, {'overrides': {'anderson_acceleration': {'enabled': True}},
                                        H.S56_CYCLE_CAPTURE_PINNED_KEY: True})]
        specs = {}
        for label, mod in (('pre', pre), ('post', H)):
            specs[label] = SC._spec_norm(SC._freeze(mod, os.path.join(scratch, f'spec_{label}'), 's56s4calib_c3',
                                                    cands))
        c3 = {'identical_pre_post': specs['pre'] == specs['post'],
              'differences': PC._diff(specs['pre'], specs['post'])[:50]}
        keyed = SC._spec_norm(SC._freeze(H, os.path.join(scratch, 'spec_post_keys'), 's56s4calib_c3', [
            ('x0_v1', x0, {'interface_coupling': dict(GS.COUPLING_V1_NOQ), KEYS[0]: True, KEYS[1]: True})]))
        by_label = {e['label']: e for e in specs['post']['candidates']}
        e = keyed['candidates'][0]
        c3['with_keys'] = {'eval_key_equals_no_key': e['eval_key'] == by_label['x0_v1']['eval_key'],
                           'entry_records_keys': all(e.get(k) is True for k in KEYS),
                           'entry_otherwise_identical': {k: v for k, v in e.items() if k not in KEYS}
                           == by_label['x0_v1']}
        for key in KEYS:
            try:
                SC._freeze(H, os.path.join(scratch, f'spec_post_{key}_on_off'), 's56s4calib_c3',
                           [('x0_off', x0, {key: True})])
                c3[f'{key}_on_option_off_refused_at_freeze'] = False
            except ValueError as error:
                c3[f'{key}_on_option_off_refused_at_freeze'] = str(error)
            checks[f'c3:{key}:refused_on_option_off_at_freeze'] = c3[f'{key}_on_option_off_refused_at_freeze'] \
                is not False
        res['c3'] = c3
        checks['c3:spec_identical_pre_post_without_keys'] = c3['identical_pre_post']
        checks['c3:with_keys:eval_key_unchanged_entry_records_keys_otherwise_identical'] = all(
            c3['with_keys'].values())
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


# ======================================================================================================================
#  --stub-run
# ======================================================================================================================
def _synthetic(tag, n, year, day, idx, s_m, s_o, p, which):
    """A deterministic value per (stub call, block, entry, quantity) -- the same in the off and the on run."""
    h = hashlib.sha256(f'{tag}|{n}|{year}|{day}|{idx}|{s_m}|{s_o}|{p}|{which}'.encode()).digest()
    return (int.from_bytes(h[:6], 'big') / float(1 << 48)) * 10.0 - 5.0


class NLStubs(PC.Stubs):
    """PC.Stubs + NL hashing at ADMM cycles 1-3 + synthetic interface suffix values (recorded)."""

    def __init__(self, planning, nl_dir):
        super().__init__(planning)
        self.nl_dir = nl_dir
        self.nl_hashes = {}
        self.suffix_set = {}   # (tag, n, year, day) -> {(idx, s_m, s_o, p): {...}}

    def _interface_idx(self, tag, year, day):
        if tag == 'TSO':
            net = self.planning.transmission_network.network[year][day]
            return [net.get_node_idx(n) for n in self.planning.transmission_network.active_distribution_network_nodes]
        node_id = int(tag[3:])
        net = self.planning.distribution_networks[node_id].network[year][day]
        return [net.get_node_idx(net.get_reference_node_id())]

    def network(self, tag):
        def stub(model, *args, **kwargs):
            n = self.calls[tag] = self.calls.get(tag, 0) + 1
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    block = model[year][day]
                    if (n - 1) in NL_CYCLES:
                        path = os.path.join(self.nl_dir, f'{tag}_{year}_{day}_n{n}.nl')
                        block.write(path, format='nl', io_options={'symbolic_solver_labels': False})
                        self.nl_hashes[f'{tag}|{year}|{day}|cycle{n - 1}'] = _sha(path)
                        os.remove(path)
                    PC._stand_in_point(block, self._salt(tag, n, year, day))
                    rec = {}
                    for idx in self._interface_idx(tag, year, day):
                        for s_m in block.scenarios_market:
                            for s_o in block.scenarios_operation:
                                for p in block.periods:
                                    key = (idx, s_m, s_o, p)
                                    r = {}
                                    if key in block.voltage_magnitude_upper_cons:
                                        r['row_dual'] = _synthetic(tag, n, year, day, idx, s_m, s_o, p, 'row')
                                        block.dual[block.voltage_magnitude_upper_cons[key]] = r['row_dual']
                                    r['zU_vmag_sqr'] = _synthetic(tag, n, year, day, idx, s_m, s_o, p, 'vsq')
                                    block.ipopt_zU_out[block.vmag_sqr[key]] = r['zU_vmag_sqr']
                                    if idx in block.vmag_nodes:
                                        r['zU_vmag'] = _synthetic(tag, n, year, day, idx, s_m, s_o, p, 'v')
                                        block.ipopt_zU_out[block.vmag[key]] = r['zU_vmag']
                                    r['zU_e'] = _synthetic(tag, n, year, day, idx, s_m, s_o, p, 'e')
                                    block.ipopt_zU_out[block.e[key]] = r['zU_e']
                                    r['vmag_sqr'] = float(block.vmag_sqr[key].value)
                                    r['e'] = float(block.e[key].value)
                                    rec[key] = r
                    self.suffix_set[(tag, n, year, day)] = rec
                    results[year][day] = GS._fake_ok()
            return results
        return stub


def stub_run(args):
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    import p58_rescale as R
    import interface_dual_capture as IDC
    from contextlib import nullcontext
    keys_on = args.keys == 'on'
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'stub_run_keys_{args.keys}.json')
    if os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    scratch = os.path.join(out_dir, f'scratch_keys_{args.keys}')
    os.makedirs(scratch, exist_ok=False)
    nl_dir = os.path.join(scratch, 'nl')
    os.makedirs(nl_dir)
    t0 = time.time()
    coupling = dict(GS.COUPLING_V1_NOQ)
    out = {'schema': SCHEMA + '_stub_run', 'keys': args.keys, 'started_utc': _utc(), 'provenance': GS.provenance(),
           'script_sha256': _sha(os.path.abspath(__file__)),
           'harness_sha256': _sha(os.path.join(REPO, 'p515_s44_campaign_harness.py')), 'cap': STUB_CAP,
           'required_consecutive_cycles': REQUIRED, 'coupling': coupling, 'nl_cycles': list(NL_CYCLES), 'error': None}
    holder = {}
    try:
        derived = derived_A()
        out['derived_installed'] = H.install_derived_instance(derived, os.path.join(scratch, 'derived'))
        report = {}
        eval_id = f"p515s56calib_{args.keys}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        planning, sed, candidate = G._construct_arm_planning(
            's39_D', os.path.join(scratch, 'arm'), report, investment_map=GS.CANDIDATES['x0'], eval_id=eval_id,
            num_max_iters_override=STUB_CAP, apply_rho=False, investment_year=GS.INVESTMENT_YEAR)
        spec = GS.spec_like(derived)
        spec['cap'] = STUB_CAP
        if spec['required_consecutive_cycles'] != REQUIRED:
            raise RuntimeError('required_consecutive_cycles differs')
        H._config_hook_factory(spec, holder, overrides={}, model_variant=None, investment_year=GS.INVESTMENT_YEAR,
                               expected_floor_rows=None, flex_price_multiplier=None, interface_deviation_premium=None,
                               release_solution_bookkeeping=None, interface_coupling=coupling)(
            planning=planning, sed=sed, candidate=candidate, report=report)
        a = planning.params.admm
        out['length_in_force_before_run'] = a.minimum_consecutive_converged_cycles
        tail_checklist = H.assert_convergence_depth_tail_capture(spec)
        out['tail_checklist'] = tail_checklist
        entry = {'interface_coupling': coupling}
        if keys_on:
            entry.update({KEYS[0]: True, KEYS[1]: True})
        stop_on, vcap_on = H.s56_calibration_keys_in_force(entry, True)
        if stop_on:
            holder['s56_certification_stop_checklist'] = H.assert_s56_certification_stop_path(tail_checklist,
                                                                                               REQUIRED)
        if vcap_on:
            holder['s56_interface_v_multiplier_checklist'] = H.assert_s56_interface_v_multiplier_path()
        holder['s56_cycle_capture_checklist'] = H.assert_s56_cycle_capture_path()
        stubs = NLStubs(planning, nl_dir)
        planning.transmission_network.optimize = stubs.network('TSO')
        for node_id, dn in planning.distribution_networks.items():
            dn.optimize = stubs.network(f'DSO{node_id}')
        sed.optimize = stubs.esso
        eval_dir = os.path.join(scratch, 'eval')
        os.makedirs(eval_dir)
        paths = {k: os.path.join(eval_dir, f'{k}.jsonl') for k in ('recourse_jump', 'ess_stride', 'floor',
                                                                   'pf_stride', 'exempt')}
        if a.objective_scale != GS.FIXED_SIGMA:
            raise RuntimeError(f'fixed sigma {a.objective_scale} != {GS.FIXED_SIGMA}')
        original_scale = srp._compute_common_admm_objective_scale
        srp._compute_common_admm_objective_scale = (
            lambda planning_problem, tso_model, dso_models: float(planning_problem.params.admm.objective_scale))
        # innermost: production's Boyd metrics, reported as all-pass (so production's certificate would stop at 10)
        original_boyd = srp.get_admm_boyd_residual_metrics

        def forced_pass(*a_, **k_):
            r = original_boyd(*a_, **k_)
            r['all_boyd_pass'] = True
            return r
        srp.get_admm_boyd_residual_metrics = forced_pass
        state = None
        stop_cm = H.s56_certification_stop_disabled_hooks(holder, REQUIRED) if stop_on else nullcontext()
        vmult_cm = H.s56_interface_v_multiplier_hooks(eval_dir, holder) if vcap_on else nullcontext()
        try:
            with R.patched_admm_objectives(), \
                    IDC.interface_dual_capture_hooks(eval_dir, holder) as dual_capture, \
                    H.s56_cycle_capture_hooks(eval_dir, holder), \
                    stop_cm, \
                    vmult_cm, \
                    G.s38_pf_capture_hooks(paths['recourse_jump'], paths['ess_stride'], paths['floor'],
                                           paths['pf_stride'], {}, stride=1), \
                    G.s39_exempt_until_capture_hooks(paths['exempt']):
                try:
                    _c, _res, models, _s, _p, state = planning.run_operational_planning(
                        type='distributed', candidate_solution=deepcopy(candidate), print_results=False,
                        debug_flag=False, return_state=True)
                    out['run_error'] = None
                except Exception as error:  # noqa: BLE001 -- recorded
                    out['run_error'] = f'{type(error).__name__}: {error}'
                    out['run_traceback'] = traceback.format_exc()
                dual_capture.close()
        finally:
            srp._compute_common_admm_objective_scale = original_scale
            srp.get_admm_boyd_residual_metrics = original_boyd
        out['length_in_force_after_run'] = a.minimum_consecutive_converged_cycles
        out['stub_calls'] = dict(stubs.calls)
        out['nl_hashes'] = stubs.nl_hashes
        diags = (state or {}).get('admm_diagnostics') or []
        out['n_cycles'] = len(diags)
        out['admm_diagnostics'] = [PC._diag_record(e) for e in diags]
        out['required_consecutive_cycles_per_cycle'] = [e.get('required_consecutive_cycles') for e in diags]
        out['consecutive_converged_cycles_per_cycle'] = [e.get('consecutive_converged_cycles') for e in diags]
        out['summaries'] = {k: holder.get(k) for k in (H.S56_CYCLE_CAPTURE_SUMMARY_KEY, IDC.SUMMARY_KEY,
                                                       H.S56_CERTIFICATION_STOP_SUMMARY_KEY,
                                                       H.S56_INTERFACE_V_MULTIPLIER_SUMMARY_KEY)}
        out['checklists'] = {k: holder.get(k) for k in ('s56_certification_stop_checklist',
                                                        's56_interface_v_multiplier_checklist')}
        for name in (H.S56_CYCLE_CAPTURE_FILE, IDC.INTERFACE_DUAL_FILE):
            p = os.path.join(eval_dir, name)
            with open(p) as handle:
                lines = [json.loads(t) for t in handle if t.strip()]
            out[f'sha256_lines_without_capture_s:{name}'] = [
                hashlib.sha256(json.dumps({k: v for k, v in x.items() if k != 'capture_s'}, sort_keys=True)
                               .encode()).hexdigest() for x in lines]
        # (e) the multiplier sidecar against what the stub put in (the last network call of each cycle)
        vp = os.path.join(eval_dir, H.S56_INTERFACE_V_MULTIPLIER_FILE)
        if vcap_on:
            with open(vp) as handle:
                vlines = [json.loads(t) for t in handle if t.strip()]
            hdr, body = vlines[0], vlines[1:]
            out['vmult_header'] = hdr
            out['vmult_n_lines'] = len(body)
            mism, n_cmp = [], 0
            blocks = [tuple(b) for b in hdr['blocks']]
            scen = [tuple(s) for s in hdr['scenarios']]
            for ln in body:
                k = ln['cycle']
                for side in ('tso', 'dso'):
                    for bi, (node, year, day) in enumerate(blocks):
                        tag = 'TSO' if side == 'tso' else f'DSO{node}'
                        rec = stubs.suffix_set[(tag, k + 1, year, day)]
                        _net, idx = H.s56_interface_v_side_index(planning, side, node, year, day)
                        for si, (s_m, s_o) in enumerate(scen):
                            for p in range(hdr['n_periods']):
                                want = rec[(idx, s_m, s_o, p)]
                                for arr in H.S56_INTERFACE_V_MULTIPLIER_ARRAYS:
                                    got = ln[side][arr][bi][si][p]
                                    n_cmp += 1
                                    if got != want.get(arr):
                                        mism.append({'cycle': k, 'side': side, 'block': [node, year, day],
                                                     'scenario': [s_m, s_o], 'period': p, 'array': arr,
                                                     'captured': got, 'stub': want.get(arr)})
            out['vmult_compare'] = {'n_compared': n_cmp, 'n_mismatch': len(mism), 'first_mismatches': mism[:10]}
        else:
            out['vmult_file_absent'] = not os.path.exists(vp)
        out['eval_dir_files'] = sorted(os.listdir(eval_dir))
    except Exception as error:  # noqa: BLE001 -- recorded
        out['error'] = f'{type(error).__name__}: {error}'
        out['traceback'] = traceback.format_exc()
    out['guard'] = GS.guard_report()
    out['wall_s'] = time.time() - t0
    _dump(out_path, out)
    print(f"[stub-run keys {args.keys}] error={out['error']} run_error={out.get('run_error')} "
          f"cycles={out.get('n_cycles')} nl={len(out.get('nl_hashes') or {})} "
          f"vmult={out.get('vmult_compare')} guard_zero={out['guard']['verified_zero']} -> {out_path}", flush=True)
    return 0 if out['error'] is None and out.get('run_error') is None and out['guard']['verified_zero'] else 1


# ======================================================================================================================
#  --compare
# ======================================================================================================================
def compare(args):
    out_dir = os.path.abspath(args.out_dir)
    with open(os.path.join(out_dir, 'stub_run_keys_off.json')) as handle:
        off = json.load(handle)
    with open(os.path.join(out_dir, 'stub_run_keys_on.json')) as handle:
        on = json.load(handle)
    checks = {}
    res = {'schema': SCHEMA + '_compare', 'utc': _utc(), 'declared_diag_difference': list(DECLARED_DIAG_DIFFERENCE),
           'inputs': {n: _sha(os.path.join(out_dir, f'stub_run_keys_{n}.json')) for n in ('off', 'on')}}
    checks['both_runs_error_free'] = all(x['error'] is None and x.get('run_error') is None for x in (off, on))
    checks['both_guards_zero'] = all(x['guard']['verified_zero'] for x in (off, on))
    checks['same_harness_and_script'] = (off['harness_sha256'] == on['harness_sha256']
                                         and off['script_sha256'] == on['script_sha256'])
    # (a)
    checks['a:off_stops_at_certificate_cycle_10'] = off['n_cycles'] == REQUIRED
    checks['a:on_runs_to_the_cap'] = on['n_cycles'] == STUB_CAP
    checks['a:off_length_10_every_cycle'] = off['required_consecutive_cycles_per_cycle'] == [REQUIRED] * REQUIRED
    checks['a:on_length_threshold_every_cycle'] = (on['required_consecutive_cycles_per_cycle']
                                                   == [H.S56_CERTIFICATION_DISABLED_THRESHOLD] * STUB_CAP)
    checks['a:on_length_restored_after_run'] = on['length_in_force_after_run'] == REQUIRED
    checks['a:consecutive_count_identical'] = (on['consecutive_converged_cycles_per_cycle'][:REQUIRED]
                                               == off['consecutive_converged_cycles_per_cycle'])
    stop = (on.get('summaries') or {}).get(H.S56_CERTIFICATION_STOP_SUMMARY_KEY) or {}
    checks['a:on_stop_summary_ok'] = stop.get('ok') is True
    # (b)
    diffs = []
    for k in range(REQUIRED):
        a_, b_ = off['admm_diagnostics'][k], on['admm_diagnostics'][k]
        for part in ('values', 'hex'):
            for key in sorted(set(a_[part]) | set(b_[part])):
                if key in DECLARED_DIAG_DIFFERENCE:
                    continue
                if a_[part].get(key) != b_[part].get(key):
                    diffs.append({'cycle': k + 1, 'part': part, 'key': key})
    res['b_diag_differences'] = diffs[:50]
    res['b_n_diag_differences'] = len(diffs)
    checks['b:diagnostics_cycles_1_10_identical_except_declared'] = not diffs
    # (c)
    nl_off, nl_on = off['nl_hashes'], on['nl_hashes']
    res['c_n_nl'] = {'off': len(nl_off), 'on': len(nl_on)}
    res['c_nl_differ'] = sorted(k for k in set(nl_off) | set(nl_on) if nl_off.get(k) != nl_on.get(k))
    checks['c:nl_identity_every_block_cycles_1_3'] = bool(nl_off) and not res['c_nl_differ']
    # (d)
    for name in (H.S56_CYCLE_CAPTURE_FILE, 'interface_duals_per_cycle.jsonl'):
        a_ = off[f'sha256_lines_without_capture_s:{name}']
        b_ = on[f'sha256_lines_without_capture_s:{name}'][:len(a_)]
        checks[f'd:{name}:lines_identical_through_cycle_10'] = a_ == b_ and len(a_) == REQUIRED + 1
    # (e)
    vs = (on.get('summaries') or {}).get(H.S56_INTERFACE_V_MULTIPLIER_SUMMARY_KEY) or {}
    checks['e:on_multiplier_summary_ok'] = vs.get('ok') is True and vs.get('cycle_lines_written') == STUB_CAP
    checks['e:on_multiplier_values_equal_stub_bitwise'] = ((on.get('vmult_compare') or {}).get('n_mismatch') == 0
                                                          and (on.get('vmult_compare') or {}).get('n_compared', 0) > 0)
    checks['e:off_no_multiplier_file'] = off.get('vmult_file_absent') is True
    res['e_vmult_compare'] = on.get('vmult_compare')
    res['checks'] = checks
    res['holds'] = all(checks.values())
    res['guard'] = GS.guard_report()
    _dump(os.path.join(out_dir, f'compare_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json'), res)
    failed = sorted(k for k, v in checks.items() if not v)
    print(f"[compare] holds={res['holds']} failed={failed} nl={res['c_n_nl']} guard_zero="
          f"{res['guard']['verified_zero']}", flush=True)
    return 0 if res['holds'] and res['guard']['verified_zero'] else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nokey', action='store_true')
    ap.add_argument('--stub-run', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--keys', choices=('off', 'on'))
    ap.add_argument('--pre-rev')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    if a.nokey and a.pre_rev:
        return nokey(a)
    if a.stub_run and a.keys:
        return stub_run(a)
    if a.compare:
        return compare(a)
    ap.error('--nokey --pre-rev R | --stub-run --keys off|on | --compare')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
