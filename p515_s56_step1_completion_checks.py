"""
P5.15 S56 step 1 COMPLETION -- zero-solve record / wiring probe (Planner task, 2026-11-23, items 2 and 3).

WHAT IT DOES. For one build configuration of the frozen gate script `p515_s56_coupling_gates.py` (`BUILD_CONFIGS`;
spec `frozen_coupling_gates_spec_v1_10c98258.json`), it constructs the evaluation EXACTLY as the build child does
(`p515_g_g1_g4_admm_gates._construct_arm_planning` + the campaign harness's `_config_hook_factory`), replaces every
network / ESSO `optimize` by a stub (call 1 = initialisation: Vars with no value -> 0 clipped to bounds; every call
returns an ok / optimal SolverResults and leaves the values as they are), and runs production's
`run_operational_planning` for TWO stubbed ADMM cycles inside the SAME capture chain the campaign child installs
around `run_admm_arm` (`p58_rescale.patched_admm_objectives`, `interface_dual_capture.interface_dual_capture_hooks`,
`p515_g_g1_g4_admm_gates.s38_pf_capture_hooks` (the S38 PF entry identity, which raised in G-f r2),
`s39_exempt_until_capture_hooks`, and -- on a derived instance -- the harness's `alpha_row_run_hooks`). Zero solves:
the gate script's blocking `SolveProfileGuard(permitted=())` is installed at import and verified at 0.

It records, per configuration:
  * the wiring that IPOPT would see: at each stubbed network call, which objectives are ACTIVE on every TSO block and
    on every DSO block, and whether the active objective references the per-scenario deviation dual Params
    (`dual_interface_dev_p_req`) -- the p58 rescaled objective is a snapshot taken inside `update_*_to_admm`;
  * the per-cycle `admm_diagnostics` fields of production (gross / net Q, every Boyd field, the S56 per-cycle fields),
    as values and float.hex;
  * production's recourse components and block decomposition on the final models, and the harness's
    multiscenario terminal identity checks (derived instances only);
  * the S38 PF stride sidecar lines (reconstructed vs production, identity flags) and the interface-dual sidecar.

It is run once on the tree BEFORE the step-1-completion edits and once AFTER (same script bytes, recorded), into a
new directory; `--compare` then diffs the two per configuration: option-off configurations must be identical
(values and hex), the option-on configuration differs only where the edits say it should.

Usage (repo root, canonical interpreter, NLP_SOLVER_PATH set as production requires):
  <python> -u p515_s56_step1_completion_checks.py --probe --config <name> --phase <pre|post> --out-dir <dir>
  <python> -u p515_s56_step1_completion_checks.py --compare --out-dir <dir>
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s56_coupling_gates as GS  # noqa: E402 -- installs the zero-solve guard and the pickle guard on this process

H = GS.H
SCHEMA = 'p515_s56_step1_completion_probe_v1'
N_CYCLES = 2
PROBE_CONFIGS = ('srp1_x0', 'srp1grid_x0_a05', 'srp1_x0_on', 'srp1grid_x0_a05_on_v1_noq')
DEV_DUAL_PARAM = 'dual_interface_dev_p_req'


def _hex(v):
    return float.hex(float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def _objective_wiring(block):
    """Active objectives of a block and whether the active one references the deviation dual Params."""
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_mutable_parameters
    active = [o.name for o in block.component_objects(pe.Objective, active=True, descend_into=False)]
    out = {'active_objectives': active, 'has_deviation_params': hasattr(block, DEV_DUAL_PARAM)}
    if len(active) == 1:
        obj = block.component(active[0])
        names = {p.parent_component().name for p in identify_mutable_parameters(obj.expr)}
        out['active_objective_references_deviation_dual'] = DEV_DUAL_PARAM in names
    if hasattr(block, 'admm_objective'):
        names = {p.parent_component().name for p in identify_mutable_parameters(block.admm_objective.expr)}
        out['admm_objective_references_deviation_dual'] = DEV_DUAL_PARAM in names
        out['admm_objective_active'] = bool(block.admm_objective.active)
    return out


def _stand_in_point(block, salt):
    """A deterministic, NON-degenerate stand-in point (so residuals, settlements and Q are not identically 0): every
    free Var, in Pyomo's own component order, is set to lb + f (ub - lb) on a box clipped to +-10 around 0, with
    f = ((i * 2654435761 + salt) mod 1000003) / 1000003. Depends only on (position, salt), so the pre and post runs
    see the same point. Returns the count set."""
    import pyomo.environ as pe
    n = 0
    for i, v in enumerate(block.component_data_objects(pe.Var, descend_into=True)):
        if v.fixed:
            continue
        lb = v.lb if (v.lb is not None and abs(v.lb) <= 10.0) else -10.0
        ub = v.ub if (v.ub is not None and abs(v.ub) <= 10.0) else 10.0
        if v.lb is not None and lb < v.lb:
            lb = v.lb
        if v.ub is not None and ub > v.ub:
            ub = v.ub
        if ub < lb:
            ub = lb
        f = ((i * 2654435761 + salt) % 1000003) / 1000003.0
        v.set_value(lb + f * (ub - lb), skip_validation=True)
        n += 1
    return n


class Stubs:
    def __init__(self, planning):
        self.calls = {}
        self.wiring = {}
        self.planning = planning

    @staticmethod
    def _salt(tag, n, year=None, day=None):
        return sum(ord(c) for c in f'{tag}|{year}|{day}') * 7919 + n * 104729

    def network(self, tag):
        def stub(model, *args, **kwargs):
            n = self.calls[tag] = self.calls.get(tag, 0) + 1
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    if n == 2 and (tag == 'TSO' or (year, day) == (next(iter(model)), next(iter(model[year])))):
                        # at the first ADMM-cycle call, BEFORE the stand-in point: every TSO block, and the first
                        # block of each DSO -- what IPOPT would be handed
                        self.wiring[f'{tag}|{year}|{day}'] = _objective_wiring(model[year][day])
                    _stand_in_point(model[year][day], self._salt(tag, n, year, day))
                    results[year][day] = GS._fake_ok()
            return results
        return stub

    def esso(self, models, *args, **kwargs):
        n = self.calls['ESSO'] = self.calls.get('ESSO', 0) + 1
        for node_id, model in models.items():
            _stand_in_point(model, self._salt('ESSO', n, node_id))
        return {node_id: GS._fake_ok() for node_id in models}


DIAG_FIELD_PREFIXES = ('boyd_', 'interface_coupling', 'scenario_mismatch', 'priced_residual', 'gross_operational',
                       'net_operational', 'q_convention')
DIAG_FIELDS = ('cycle', 'recourse', 'gross_operational_cost', 'terminal_salvage_value', 'objective_change_abs',
               'objective_tolerance', 'local_solves_ok', 'primal_pf', 'dual_pf_mean', 'primal_v', 'aa_action')


def _diag_record(entry):
    keep = {k: v for k, v in entry.items() if k in DIAG_FIELDS or k.startswith(DIAG_FIELD_PREFIXES)}
    return {'values': _jsonable(keep), 'hex': {k: _hex(v) for k, v in keep.items() if _hex(v) is not None}}


def probe(args):
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    import p58_rescale as R
    import interface_dual_capture as IDC
    cfg = GS.BUILD_CONFIGS[args.config]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'probe_{args.phase}_{args.config}.json')
    if os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    scratch = os.path.join(out_dir, f'scratch_{args.phase}_{args.config}')
    os.makedirs(scratch, exist_ok=False)
    t0 = time.time()
    out = {'schema': SCHEMA, 'config': args.config, 'phase': args.phase, 'declaration': cfg,
           'started_utc': datetime.now(timezone.utc).isoformat(), 'provenance': GS.provenance(),
           'probe_script_sha256': GS._sha_file(os.path.abspath(__file__)), 'error': None}
    try:
        derived = GS.derived_declaration(cfg['instance'])
        if derived is not None:
            out['derived_installed'] = H.install_derived_instance(derived, os.path.join(scratch, 'derived'))
        report, holder = {}, {}
        eval_id = f"p515s56cmp_{args.phase}_{args.config}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        planning, sed, candidate = G._construct_arm_planning(
            's39_D', os.path.join(scratch, 'arm'), report, investment_map=GS.CANDIDATES[cfg['candidate']],
            eval_id=eval_id, num_max_iters_override=N_CYCLES, apply_rho=False, investment_year=GS.INVESTMENT_YEAR)
        hook_kwargs = dict(overrides={}, model_variant=None, investment_year=GS.INVESTMENT_YEAR,
                           expected_floor_rows=None, flex_price_multiplier=None,
                           interface_deviation_premium=cfg['premium'], release_solution_bookkeeping=None)
        if cfg['coupling'] is not None:
            hook_kwargs['interface_coupling'] = cfg['coupling']
        spec = GS.spec_like(derived)
        spec['cap'] = N_CYCLES   # the hook asserts num_max_iters == the spec cap
        H._config_hook_factory(spec, holder, **hook_kwargs)(
            planning=planning, sed=sed, candidate=candidate, report=report)
        a = planning.params.admm
        out['admm_in_force'] = {k: getattr(a, k, 'ABSENT') for k in (
            'per_scenario_interface_coupling', 'per_scenario_voltage_mode', 'row18_charge_reactive')}
        stubs = Stubs(planning)
        planning.transmission_network.optimize = stubs.network('TSO')
        for node_id, dn in planning.distribution_networks.items():
            dn.optimize = stubs.network(f'DSO{node_id}')
        sed.optimize = stubs.esso
        eval_dir = os.path.join(scratch, f'eval_{args.phase}_{args.config}')
        os.makedirs(eval_dir)
        paths = {k: os.path.join(eval_dir, f'{k}.jsonl') for k in ('recourse_jump', 'ess_stride', 'floor',
                                                                   'pf_stride', 'exempt')}
        premium = H.validate_interface_deviation_premium(cfg['premium']) if cfg['premium'] else None
        capture_multiscenario = derived is not None or premium is not None
        from contextlib import nullcontext
        alpha_hooks = (H.alpha_row_run_hooks(eval_dir, 'probe', f'probe_{args.config}',
                                             H.derived_instance_identity(derived) if derived else None,
                                             float((premium or {}).get('alpha') or 0.0), holder)
                       if capture_multiscenario else nullcontext())
        state = None
        # As the build child: the stand-in point has a zero base objective, so production's computed scale would
        # raise; the ADMM objectives use the fixed sigma either way (asserted equal to the case file).
        if a.objective_scale != GS.FIXED_SIGMA:
            raise RuntimeError(f'fixed sigma {a.objective_scale} != {GS.FIXED_SIGMA}')
        original_scale = srp._compute_common_admm_objective_scale
        srp._compute_common_admm_objective_scale = (
            lambda planning_problem, tso_model, dso_models: float(planning_problem.params.admm.objective_scale))
        with R.patched_admm_objectives(), \
                IDC.interface_dual_capture_hooks(eval_dir, holder) as dual_capture, \
                G.s38_pf_capture_hooks(paths['recourse_jump'], paths['ess_stride'], paths['floor'],
                                       paths['pf_stride'], {}, stride=1), \
                G.s39_exempt_until_capture_hooks(paths['exempt']), \
                alpha_hooks:
            try:
                _c, _res, models, _s, _p, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate), print_results=False,
                    debug_flag=False, return_state=True)
                out['run_error'] = None
            except Exception as error:  # noqa: BLE001 -- recorded: this IS the evidence (e.g. a sidecar raising)
                out['run_error'] = f'{type(error).__name__}: {error}'
                out['run_traceback'] = traceback.format_exc()
                models = None
            finally:
                srp._compute_common_admm_objective_scale = original_scale
            dual_capture.close()
        out['stub_calls'] = dict(stubs.calls)
        out['objective_wiring_at_cycle1'] = stubs.wiring
        out['interface_dual_capture'] = holder.get(IDC.SUMMARY_KEY)
        diags = (state or {}).get('admm_diagnostics') or []
        out['n_cycles'] = len(diags)
        out['admm_diagnostics'] = [_diag_record(e) for e in diags]
        if models is not None:
            rc = srp._get_operational_recourse_components(planning, models)
            blocks = srp._get_operational_recourse_block_components(planning, models)
            out['recourse_components'] = {'values': _jsonable(rc),
                                          'hex': {k: _hex(v) for k, v in rc.items() if _hex(v) is not None}}
            out['block_components_sum'] = sum(blocks.values())
            out['block_components_sum_minus_net'] = sum(blocks.values()) - rc['net_operational_recourse']
            if capture_multiscenario:
                try:
                    _payload, summary = H.multiscenario_terminal_capture(planning, models, state=state,
                                                                         premium=premium)
                    out['multiscenario_terminal'] = {
                        k: _jsonable(summary.get(k)) for k in ('checks', 'all_checks_pass', 'identity_worst_rel_diffs',
                                                               'recourse_components', 'objective_convention')}
                except Exception as error:  # noqa: BLE001
                    out['multiscenario_terminal'] = {'error': f'{type(error).__name__}: {error}',
                                                     'traceback': traceback.format_exc()}
        for name in ('pf_stride',):
            lines = []
            if os.path.isfile(paths[name]):
                with open(paths[name]) as handle:
                    for text in handle:
                        rec = json.loads(text)
                        rec.pop('entries', None)
                        lines.append(rec)
            out[f'{name}_lines_without_entries'] = lines
        idc_path = os.path.join(eval_dir, IDC.INTERFACE_DUAL_FILE)
        if os.path.isfile(idc_path):
            with open(idc_path) as handle:
                idc_lines = [json.loads(t) for t in handle if t.strip()]
            out['interface_dual_sidecar_keys_per_line'] = [sorted(x.keys()) for x in idc_lines]
            # content hash without the per-line wall time ('capture_s'), which differs run to run
            import hashlib
            out['interface_dual_sidecar_sha256'] = hashlib.sha256(json.dumps(
                [{k: v for k, v in x.items() if k != 'capture_s'} for x in idc_lines],
                sort_keys=True).encode()).hexdigest()
        ident = holder.get('initialisation_identity') or {}
        out['initialisation_identity'] = {k: ident.get(k) for k in ('gross_operational_cost_hex', 'n_compared',
                                                                    'all_equal', 'mismatches')}
    except Exception as error:  # noqa: BLE001 -- recorded
        out['error'] = f'{type(error).__name__}: {error}'
        out['traceback'] = traceback.format_exc()
    out['guard'] = GS.guard_report()
    out['wall_s'] = time.time() - t0
    with open(out_path, 'x') as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    print(f"[probe {args.phase}/{args.config}] error={out['error']} run_error={out.get('run_error')} "
          f"cycles={out.get('n_cycles')} guard_zero={out['guard']['verified_zero']} -> {out_path}", flush=True)
    return 0 if out['error'] is None and out['guard']['verified_zero'] else 1


def _diff(a, b, path='', out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({'path': f'{path}/{k}', 'only_in': 'post' if k not in a else 'pre'})
            else:
                _diff(a[k], b[k], f'{path}/{k}', out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f'{path}[{i}]', out)
    elif a != b:
        out.append({'path': path, 'pre': a, 'post': b})
    return out


COMPARED = ('admm_diagnostics', 'recourse_components', 'block_components_sum', 'block_components_sum_minus_net',
            'multiscenario_terminal', 'pf_stride_lines_without_entries', 'interface_dual_sidecar_keys_per_line',
            'interface_dual_sidecar_sha256', 'n_cycles', 'run_error', 'error', 'objective_wiring_at_cycle1',
            'stub_calls', 'admm_in_force')


def compare(args):
    out_dir = os.path.abspath(args.out_dir)
    res = {'schema': SCHEMA + '_compare', 'utc': datetime.now(timezone.utc).isoformat(), 'per_config': {}}
    for config in PROBE_CONFIGS:
        pa, pb = (os.path.join(out_dir, f'probe_{ph}_{config}.json') for ph in ('pre', 'post'))
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            res['per_config'][config] = {'present': False}
            continue
        with open(pa) as h1, open(pb) as h2:
            a, b = json.load(h1), json.load(h2)
        diffs = _diff({k: a.get(k) for k in COMPARED}, {k: b.get(k) for k in COMPARED})
        res['per_config'][config] = {
            'present': True, 'coupling': GS.BUILD_CONFIGS[config]['coupling'],
            'probe_script_sha256_equal': a.get('probe_script_sha256') == b.get('probe_script_sha256'),
            'guard_zero_both': (a.get('guard') or {}).get('verified_zero') is True
                               and (b.get('guard') or {}).get('verified_zero') is True,
            'identical': not diffs, 'n_differences': len(diffs), 'differences': diffs[:200],
            'pre_head': (a.get('provenance') or {}).get('git_head'),
            'post_status': (b.get('provenance') or {}).get('git_status_porcelain')}
    path = os.path.join(out_dir, f"compare_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    with open(path, 'x') as handle:
        json.dump(res, handle, indent=1, sort_keys=True)
    for config, r in res['per_config'].items():
        print(f"[compare] {config}: identical={r.get('identical')} n_diff={r.get('n_differences')}", flush=True)
    print(f'-> {path}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--config', choices=PROBE_CONFIGS)
    ap.add_argument('--phase', choices=('pre', 'post'))
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    if args.probe:
        return probe(args)
    if args.compare:
        return compare(args)
    ap.error('one of --probe / --compare')


if __name__ == '__main__':
    sys.exit(main())
