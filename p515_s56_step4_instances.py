"""P5.15 S56 validation step 4, item 2a -- two SINGLE-BLOCK derived instances and their zero-solve gates.

Authority: S56_STEP4_DESIGN.md (Planner decision 4: "two blocks as two single-block derived instances, not a mask";
decision 5: per-block allowance); S56_PLAN_v3.md section 7 row 4; the Planner's task of 2026-11-23 (item 2a).

SOURCE: the SRP1-grid 3 x 3 instance (12 blocks; instances_v1/SRP1__s56_srp1grid_3x3.json, combined scenario checksum
73d0a230...). DERIVED (route of p515_s56_build_instances: json.load, change keys, json.dumps(case, indent='\\t'); the
route is re-checked here -- with no key changed it reproduces the source bytes): one case file per target block with
ONLY `Years` restricted to the block's year and `Days` to the block's day (the represented counts kept).
  s56_step4_2025_Winter   Years {"2025": 5}, Days {"Winter": 91}
  s56_step4_2035_Summer   Years {"2035": 5}, Days {"Summer": 91}
Read with the production reader of the derived-instance tooling (p515_s44_scale_measurement.
read_planning_from_derived_case); the 12-block source is read the same way.

GATES (zero solves; each recorded expected vs observed; a failure is recorded, never tuned):
  G1 per-block data bitwise equal to the 12-block instance's same block: every scenario array of audit A's enumeration
     (p515_s56_expected_profiles.snapshot: market energy / flexibility prices; per network block every load pd, qd,
     flexibility P/Q up/down, every generator pg, qg) with its probability vector, bitwise (float64 bytes); the
     block's network objects (TN and every DN) by audit A's recursive walk (its declared exclusions: the scenario
     arrays compared above, and the output directories); the network objects' own cost_energy_p / cost_flex arrays;
     the parameter objects (planning, TN, every DN) by the same walk; the shared-ESS per-year data of the year.
  G2 the weight ratio w_b(derived) / w_b(12-block), production `_get_admm_block_weight` (recorded, not a pass/fail).
  G3 sigma inside its assert window, read from code: the window is [sigma_fixed / f, sigma_fixed * f] with sigma_fixed
     = admm_parameters.objective_scale and f = objective_scale_assert_factor as READ (production
     `_resolve_common_admm_objective_scale`); sigma_computed is the largest weighted block-agent objective AT THE
     INITIALISATION SOLUTION (`_compute_common_admm_objective_scale`), which needs the initialisation solves. Zero-solve
     PROXIES from step 3's merged default-start solution (run_v1): lower = w_b(derived) x sum_s omega_s (TN generator
     cost) (a TSO-objective lower proxy), upper = w_b(derived) x the merged objective (the whole block). Determinate
     only when both proxies fall on the same side of the window's lower bound.
  G4 the harness and the ESSO accept the instance: a zero-solve build child (below) constructs the evaluation through
     `p515_g_g1_g4_admm_gates._construct_arm_planning` (with the harness default investment year, then -- if refused --
     the instance's own first investment year, both recorded) and runs production's `run_operational_planning` to
     the ESSO's ADMM-cycle-1 call under stubbed solves.
  G5 NL identity of the ADMM-prepared block models (TSO and every DSO of the block, at the initialisation solve and at
     ADMM cycle 1, p58 rescale applied as the harness applies it) against the 12-block instance's same block, up to
     the recorded scale: the derived build's fixed sigma is set to sigma' with fl(sigma' / w_b(derived)) ==
     fl(sigma_fixed / w_b(12-block)) bitwise, so its effective scale equals the 12-block block's and every NL byte
     must be equal. sigma' / sigma_fixed is the recorded scale (= the weight ratio up to rounding).
Also recorded: sum over the 12 blocks of w_b (12-block instance), and each block's allowance (R/4) w_b / sum_12 w with
R = 10,000 EUR (S56_STEP4_DESIGN.md decision 5; R not rescaled).

BUILD CHILD (G4/G5): the mechanism of p515_s56_coupling_gates (its `Capture`, `spec_like`, `_fake_ok`, `_fill_none`,
`write_nl`, by import), configuration = option ON, voltage mode 'per_scenario', row 18 Q charge off (the G-f
declaration), alpha = 0 (no premium), candidate x0, p58 `patched_admm_objectives` around the run. One child per
build: the 12-block reference (NL of both target blocks) and one per derived instance.

GUARDS: importing p515_s56_coupling_gates installs SolveProfileGuard(permitted=()) and its pickle guard on every
process; each is verified at exactly 0 solves. Output: NEW directory (refuses if it exists).

INSTANCE B (--instance-b; Planner ruling 2 of 2026-11-23): the 2035-only instance failed G1 (production grows prices
and loads from the instance's first year), so it is retired (artifacts kept, not used) and replaced by ONE derived case
with Years {"2025": 5, "2035": 5} x Days {"Summer": 91} (years[0] stays 2025: data and weights equal the 12-block
instance's by construction; 2025|Summer rides along). Gates G1-G5 run for BOTH blocks against the 12-block instance;
one build child for the instance (NL of both blocks) against a 12-block reference build of the same block keys.
Output `step4_instance_b.json` in a NEW directory.

Usage (repo root, canonical interpreter, attached, alone, both streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_instances.py --out-dir <new dir> \
      > <log> 2> <err>
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_step4_instances.py --instance-b --out-dir <new dir> \
      --scratch <scratch dir> > <log> 2> <err>
"""
import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

import p515_s56_coupling_gates as CG  # noqa: E402  (installs SolveProfileGuard(permitted=()) + the pickle guard)

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

H = CG.H
SCRIPT = os.path.basename(os.path.abspath(__file__))
SCHEMA = 'p515_s56_step4_instances_v1'
SOURCE_RECORD = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'instance_record.json')
SOURCE_CHECKSUM = '73d0a230a0d53d24d214e13774206844acce824dfbb84bb2d07e16f8ed6333f7'
STEP3_RESULTS = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'step3_central_v1', 'run_v1',
                             'central_ms_x0_results.json')
TARGETS = (('2025', 'Winter'), ('2035', 'Summer'))
R_EUR = 10000.0
COUPLING = dict(CG.COUPLING_V1_NOQ)          # option on, 'per_scenario', row 18 Q charge off (the G-f declaration)
PREMIUM = None                               # alpha = 0
SIGMA_SEARCH_ULPS = 64


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 step4 instances] {msg}', flush=True)


def label_of(year, day):
    return f's56_step4_{year}_{day}'


def block_keys(year, day):
    return [f'TSO|{year}|{day}'] + [f'DSO{n}|{year}|{day}' for n in (5, 7, 9)]


# ======================================================================================================================
#  derivation and reads (parent)
# ======================================================================================================================
def derive(source_case, year, day):
    case = json.loads(json.dumps(source_case))
    changes = [{'key': 'Years', 'from': case['Years'], 'to': {year: case['Years'][year]}},
               {'key': 'Days', 'from': case['Days'], 'to': {day: case['Days'][day]}}]
    case['Years'] = {year: case['Years'][year]}
    case['Days'] = {day: case['Days'][day]}
    return case, changes


class _Stages:
    @contextmanager
    def stage(self, _name):
        yield


def production_read(case_rel, read_dir):
    import p515_s44_scale_measurement as S
    os.makedirs(read_dir)
    stream = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(stream):
            planning = S.read_planning_from_derived_case({'derived_case': {'path': case_rel}}, read_dir, _Stages())
    finally:
        with open(os.path.join(read_dir, 'production_stdout.log'), 'w') as handle:
            handle.write(stream.getvalue())
    return planning, {'wall_s': time.time() - t0, 'read_dir': os.path.relpath(read_dir, REPO),
                      'combined_scenario_checksum': (planning.scenario_metadata or {}).get('combined_scenario_checksum')}


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def gate_data(p12, pd, year, day):
    """G1. Zero solves."""
    import p515_s56_expected_profiles as AA
    y = int(year)
    s12, sd = AA.snapshot(p12), AA.snapshot(pd)

    def in_block(k):
        return (k[2] == y and k[3] == day) if k[0] == 'market' else (k[1] == y and k[2] == day)
    keys12 = [k for k in s12 if in_block(k)]
    # the derived snapshot restricted to the same block (a no-op for a single-block instance, whose every key is in
    # the block; needed for the two-block instance B, Planner ruling 2 of 2026-11-23)
    keys_d = [k for k in sd if in_block(k)]
    arrays = {'n_arrays_12_block_restricted': len(keys12), 'n_arrays_derived': len(keys_d),
              'same_key_list_in_order': keys12 == keys_d, 'n_bitwise_equal': 0, 'n_prob_equal': 0, 'mismatches': []}
    for k in keys_d:
        if k not in s12:
            arrays['mismatches'].append({'key': AA.key_str(k), 'what': 'absent from the 12-block snapshot'})
            continue
        eq = _bitwise(s12[k]['A'], sd[k]['A'])
        weq = _bitwise(s12[k]['w'], sd[k]['w'])
        arrays['n_bitwise_equal'] += int(eq)
        arrays['n_prob_equal'] += int(weq)
        if not (eq and weq):
            a, b = s12[k]['A'], sd[k]['A']
            ratio = None
            if a.shape == b.shape and np.all(b != 0):
                r = a / b
                ratio = [float(r.min()), float(r.max())]
            if len(arrays['mismatches']) < 40:
                arrays['mismatches'].append({'key': AA.key_str(k), 'family': AA.family_of(k), 'array_equal': eq,
                                             'prob_equal': weq, 'ratio_12_over_derived_min_max': ratio})
    fam = {}
    for k in keys_d:
        f = AA.family_of(k)
        e = fam.setdefault(f, [0, 0])
        e[0] += 1
        e[1] += int(k in s12 and _bitwise(s12[k]['A'], sd[k]['A']))
    arrays['by_family_n_and_n_equal'] = fam
    arrays['holds'] = (arrays['same_key_list_in_order'] and arrays['n_bitwise_equal'] == len(keys_d) == len(keys12)
                       and arrays['n_prob_equal'] == len(keys_d))
    walks = {}
    pairs = [('TN.network', p12.transmission_network.network[y][day], pd.transmission_network.network[y][day])]
    for n in sorted(pd.distribution_networks):
        pairs.append((f'DN{n}.network', p12.distribution_networks[n].network[y][day],
                      pd.distribution_networks[n].network[y][day]))
    pairs.append(('planning.params', p12.params, pd.params))
    pairs.append(('TN.params', p12.transmission_network.params, pd.transmission_network.params))
    for n in sorted(pd.distribution_networks):
        pairs.append((f'DN{n}.params', p12.distribution_networks[n].params, pd.distribution_networks[n].params))
    pairs.append(('shared_ess.params', p12.shared_ess_data.params, pd.shared_ess_data.params))
    pairs.append((f'shared_ess.shared_energy_storages[{y}]', p12.shared_ess_data.shared_energy_storages[y],
                  pd.shared_ess_data.shared_energy_storages[y]))
    for name, a, b in pairs:
        w = AA.walk_compare(a, b)
        walks[name] = {'n_leaves_compared': w['n_leaves_compared'], 'n_diffs': w['n_diffs'],
                       'diffs_first': w['diffs_first'][:20], 'uncompared_types': w['uncompared_types']}
    cost = {}
    for name, a, b in pairs[:1 + len(pd.distribution_networks)]:
        cost[name] = {k: _bitwise(getattr(a, k), getattr(b, k)) for k in ('cost_energy_p', 'cost_flex')}
        cost[name]['prob_market_equal'] = list(a.prob_market_scenarios) == list(b.prob_market_scenarios)
        cost[name]['prob_operation_equal'] = list(a.prob_operation_scenarios) == list(b.prob_operation_scenarios)
    holds = (arrays['holds'] and all(w['n_diffs'] == 0 for w in walks.values())
             and all(all(v.values()) for v in cost.values()))
    return {'arrays': arrays, 'object_walks': walks, 'network_cost_arrays_and_probabilities': cost, 'holds': holds}


def weights(p12, pd, year, day):
    import shared_resources_planning as srp
    y = int(year)
    tn12, tnd = p12.transmission_network, pd.transmission_network
    w12 = srp._get_admm_block_weight(tn12, y, day)
    wd = srp._get_admm_block_weight(tnd, y, day)
    per_holder = {'TN': [w12, wd]}
    for n in sorted(pd.distribution_networks):
        per_holder[f'DN{n}'] = [srp._get_admm_block_weight(p12.distribution_networks[n], y, day),
                                srp._get_admm_block_weight(pd.distribution_networks[n], y, day)]
    return {'w_b_12_block': w12, 'w_b_derived': wd, 'ratio_derived_over_12': wd / w12,
            'first_year_12_block': int(list(tn12.years)[0]), 'first_year_derived': int(list(tnd.years)[0]),
            'per_holder_12_and_derived': per_holder,
            'all_holders_agree': all(v == [w12, wd] for v in per_holder.values()),
            'formula': 'shared_resources_planning._get_admm_block_weight: years[y] * days[d] / (1 + r)^(y - years[0])'}


def sum12_weights(p12):
    import shared_resources_planning as srp
    tn = p12.transmission_network
    per = {f'{y}|{d}': srp._get_admm_block_weight(tn, y, d) for y in tn.years for d in tn.days}
    return {'per_block': per, 'sum_12': sum(per.values()), 'n_blocks': len(per)}


def sigma_window(pd):
    a = pd.params.admm
    s, f = a.objective_scale, a.objective_scale_assert_factor
    return {'sigma_fixed': s, 'assert_factor': f, 'window': [s / f, s * f] if s is not None else None,
            'source': ('admm_parameters.objective_scale / objective_scale_assert_factor as read from the derived '
                       'instance; production _resolve_common_admm_objective_scale asserts 1/f <= sigma_computed / '
                       'sigma_fixed <= f')}


def sigma_proxy(step3, bk, w_derived, window):
    if bk not in step3.get('solves', {}):  # instance B: 2025|Summer has no step 3 run_v1 solve (G3 settles at D-3)
        return {'verdict': 'not available zero-solve (the block has no step 3 run_v1 solve); settled only by the '
                           'initialisation round', 'window_lower': window[0], 'window_upper': window[1]}
    md = step3['solves'][bk]['merged_default']['costs']
    tn_gen = sum(v['probability'] * v['gen_cost_TN_generators'] for v in md['per_scenario'].values())
    lo, hi = w_derived * tn_gen, w_derived * md['objective']
    lb, ub = window
    if lo >= lb and hi <= ub:
        verdict = 'inside (both proxies inside the window)'
    elif hi < lb:
        verdict = 'below (even the upper proxy is below the window)'
    elif lo > ub:
        verdict = 'above'
    else:
        verdict = 'indeterminate zero-solve (the proxies straddle a window bound); settled only by the initialisation round'
    return {'sum_omega_TN_generator_cost_eur_per_day': tn_gen, 'merged_objective_eur_per_day': md['objective'],
            'lower_proxy_weighted': lo, 'upper_proxy_weighted': hi, 'window_lower': lb, 'window_upper': ub,
            'verdict': verdict,
            'definition': ('lower = w_b(derived) x sum_s omega_s TN-generator cost of step 3 merged default start (a '
                           'proxy of the TSO block objective, which also carries TN flexibility / curtailment / '
                           'slack); upper = w_b(derived) x merged objective (TSO + 3 DSOs). sigma_computed is the '
                           'max over the 4 block-agents at the INITIALISATION solution (not run here).')}


def compensated_sigma(sigma, w12, wd):
    """sigma' with fl(sigma' / wd) == fl(sigma / w12) bitwise (searched over +-SIGMA_SEARCH_ULPS ulps)."""
    eff = sigma / w12
    cand = eff * wd
    tried = [cand]
    if cand / wd == eff:
        return cand, {'effective_scale_target': eff, 'sigma_prime': cand, 'exact': True, 'ulps_moved': 0}
    for k in range(1, SIGMA_SEARCH_ULPS + 1):
        for direction in (math.inf, -math.inf):
            c = cand
            for _ in range(k):
                c = math.nextafter(c, direction)
            tried.append(c)
            if c / wd == eff:
                return c, {'effective_scale_target': eff, 'sigma_prime': c, 'exact': True,
                           'ulps_moved': k if direction > 0 else -k}
    return None, {'effective_scale_target': eff, 'exact': False, 'n_tried': len(tried)}


# ======================================================================================================================
#  zero-solve build child (G4 / G5)
# ======================================================================================================================
def child_build(args):
    t0 = time.time()
    with open(args.child_in) as handle:
        cin = json.load(handle)
    out = {'schema': SCHEMA + '_build_child', 'input': cin, 'started_utc': _utc(), 'provenance': CG.provenance(),
           'error': None, 'acceptance': {}}
    try:
        import p515_g_g1_g4_admm_gates as G
        import p58_rescale as R
        import shared_resources_planning as srp
        derived = H.validate_derived_instance(cin['derived'])
        scratch = os.path.abspath(cin['scratch'])
        os.makedirs(scratch, exist_ok=False)
        out['derived_installed'] = H.install_derived_instance(derived, os.path.join(scratch, 'derived'))
        report, holder = {}, {}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        planning = sed = candidate = None
        for attempt, iy in (('harness_default', CG.INVESTMENT_YEAR), ('instance_first_year', None)):
            if attempt == 'instance_first_year':
                if planning is not None:
                    break
                import p56a_oracle as O
                iy = int(list(O.load_baseline()['planning'].shared_ess_data.years)[0])
            eval_id = f"p515s56s4_{cin['name']}_{attempt}_{stamp}"
            try:
                planning, sed, candidate = G._construct_arm_planning(
                    's39_D', os.path.join(scratch, f'arm_{attempt}'), report, investment_map=CG.X0,
                    eval_id=eval_id, num_max_iters_override=CG.CAP_BUILD, apply_rho=False, investment_year=iy)
                out['acceptance'][attempt] = {'investment_year': iy, 'accepted': True, 'eval_id': eval_id}
                out['investment_year_used'] = iy
            except Exception as error:  # noqa: BLE001 -- recorded (this IS the acceptance evidence)
                out['acceptance'][attempt] = {'investment_year': iy, 'accepted': False,
                                              'error': f'{type(error).__name__}: {error}'}
        if planning is None:
            raise RuntimeError(f"no construction accepted: {out['acceptance']}")
        hook = H._config_hook_factory(CG.spec_like(derived), holder, overrides={}, model_variant=None,
                                      investment_year=out['investment_year_used'], expected_floor_rows=None,
                                      flex_price_multiplier=None, interface_deviation_premium=PREMIUM,
                                      release_solution_bookkeeping=None, interface_coupling=COUPLING)
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        a = planning.params.admm
        if a.objective_scale != CG.FIXED_SIGMA:
            raise RuntimeError(f'fixed sigma {a.objective_scale} != {CG.FIXED_SIGMA}')
        sigma_used = cin['sigma_prime'] if cin.get('sigma_prime') is not None else a.objective_scale
        a.objective_scale = sigma_used
        out['admm_in_force'] = {k: getattr(a, k, 'ABSENT') for k in (
            'per_scenario_interface_coupling', 'per_scenario_voltage_mode', 'row18_charge_reactive',
            'objective_scale', 'objective_scale_assert_factor')}
        out['admm_in_force']['interface_deviation_premium'] = dict(a.interface_deviation_premium)
        out['sigma_used'] = sigma_used
        original_scale = srp._compute_common_admm_objective_scale
        srp._compute_common_admm_objective_scale = (
            lambda planning_problem, tso_model, dso_models: float(planning_problem.params.admm.objective_scale))
        nl_dir = os.path.join(scratch, 'nl')
        os.makedirs(nl_dir)
        cfg = {'coupling': COUPLING, 'premium': PREMIUM, 'nl_blocks': cin['nl_blocks']}
        cap = CG.Capture(cfg, planning, nl_dir)
        planning.transmission_network.optimize = cap.network_stub('TSO')
        for node_id, dn in planning.distribution_networks.items():
            dn.optimize = cap.network_stub(f'DSO{node_id}')
        sed.optimize = cap.esso_stub
        stopped = False
        p58_applied = {}
        try:
            with R.patched_admm_objectives(p58_applied):
                planning.run_operational_planning(type='distributed', candidate_solution=deepcopy(candidate),
                                                  print_results=False, debug_flag=False, return_state=True)
        except CG._StopCapture as stop:
            stopped = True
            out['stopped_by'] = str(stop)
        finally:
            srp._compute_common_admm_objective_scale = original_scale
        tn = planning.transmission_network
        out['effective_scale'] = {k: v for k, v in p58_applied.items() if k in cin['nl_blocks']}
        out['stub_calls'] = dict(cap.calls)
        out['nl'] = cap.nl
        out['checks'] = {s: {k: v for k, v in cap.checks[s].items() if k in cin['nl_blocks']} for s in CG.STAGES}
        out['coupling_active'] = cap.coupling_active
        n_net = len(tn.years) * len(tn.days)
        expected_calls = {'TSO': 2, 'ESSO': 2, **{f'DSO{n}': 2 for n in planning.distribution_networks}}
        out['build_checks'] = {
            'stopped_by_esso_cycle1_sentinel': stopped,
            'stub_calls_exact': cap.calls == expected_calls,
            'every_network_block_checked_both_stages': all(
                sum(1 for k in cap.checks[s] if not k.startswith('ESSO')) == n_net * (1 + len(planning.distribution_networks))
                for s in CG.STAGES),
            'nl_blocks_as_declared': all(set(cap.nl[s]) == set(cin['nl_blocks']) for s in CG.STAGES),
            'no_nl_write_error': all(r.get('error') is None for s in CG.STAGES for r in cap.nl[s].values()),
            'labelled_equals_plain_after_stripping': all(r.get('stripped_equal') is True for s in CG.STAGES
                                                         for r in cap.nl[s].values()),
            'structural_checks_all_true_on_declared_blocks': all(
                all(v.values()) for s in CG.STAGES for v in out['checks'][s].values()),
            'p58_rescale_applied_on_declared_blocks': all(p58_applied.get(k) is not None for k in cin['nl_blocks']),
            'scale_replacement_restored': srp._compute_common_admm_objective_scale is original_scale,
        }
    except Exception as error:  # noqa: BLE001 -- recorded
        out['error'] = f'{type(error).__name__}: {error}'
        out['traceback'] = traceback.format_exc()
    out['guard'] = CG.guard_report()
    out['wall_s'] = time.time() - t0
    out['finished_utc'] = _utc()
    with open(args.child_out, 'x') as handle:
        GRIO.dump(out, handle, indent=1, default=GRIO.json_default_item)
    return 0 if out['error'] is None else 1


def run_child(name, cin, out_dir):
    cin_path = os.path.join(out_dir, f'build_{name}_input.json')
    with open(cin_path, 'x') as handle:
        GRIO.dump(cin, handle, indent=1, default=GRIO.json_default_item)
    child_out = os.path.join(out_dir, f'build_{name}.json')
    log = os.path.join(out_dir, f'build_{name}.log')
    cmd = [sys.executable, '-u', SCRIPT, '--child-build', '--child-in', cin_path, '--child-out', child_out]
    t0 = time.time()
    with open(log, 'x') as handle:
        rc = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode
    with open(child_out) as handle:
        res = json.load(handle)
    _log(f"build {name}: exit {rc} in {time.time() - t0:.0f} s, error {res.get('error')}")
    return rc, res, os.path.relpath(child_out, REPO)


def nl_compare(ref, new, keys):
    out = {'per_block': {}, 'n_compared': 0, 'n_identical': 0}
    for stage in CG.STAGES:
        for k in keys:
            a, b = (ref.get('nl') or {}).get(stage, {}).get(k), (new.get('nl') or {}).get(stage, {}).get(k)
            rec = {'present_both': a is not None and b is not None}
            if rec['present_both']:
                rec.update({part: a[part]['sha256'] == b[part]['sha256'] for part in ('plain', 'sym_nl', 'sym_row',
                                                                                     'sym_col')})
                rec['bytes'] = [a['plain']['bytes'], b['plain']['bytes']]
                rec['identical'] = all(rec[p] for p in ('plain', 'sym_nl', 'sym_row', 'sym_col'))
            else:
                rec['identical'] = False
            out['per_block'][f'{stage}:{k}'] = rec
            out['n_compared'] += 1
            out['n_identical'] += int(rec['identical'])
    out['all_identical'] = out['n_compared'] > 0 and out['n_identical'] == out['n_compared']
    return out


# ======================================================================================================================
#  parent
# ======================================================================================================================
def other_processes():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = {os.getpid(), os.getppid()}
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) >= 3 and int(tok[0]) not in me and os.path.basename(tok[1]).startswith('python') and any(
                os.path.basename(t).startswith('p5') and t.endswith('.py') for t in tok[2:4]):
            found.append(ln.strip()[:300])
    return found


def main_parent(args):
    t_start = time.time()
    out_dir = os.path.abspath(args.out_dir)
    pre = []
    if os.environ.get('NLP_SOLVER_PATH') != CG.SOLVER_PATH:
        pre.append(f'NLP_SOLVER_PATH must be {CG.SOLVER_PATH}')
    if os.path.exists(out_dir):
        pre.append(f'{out_dir} exists')
    if os.path.exists(os.path.join(REPO, '.p515_s44_campaign.lock')):
        pre.append('campaign lock exists')
    others = other_processes()
    if others:
        pre.append(f'other project processes: {others}')
    if pre:
        _log(f'REFUSED: {pre}')
        return 2
    os.makedirs(out_dir)
    res = {'schema': SCHEMA, 'utc_start': _utc(), 'script': SCRIPT, 'script_sha256': CG._sha_file(SCRIPT),
           'provenance': CG.provenance(), 'R_eur': R_EUR, 'coupling': COUPLING, 'premium': PREMIUM,
           'objective_convention': ('allowances in EUR of total discounted planning cost: (R/4) w_b / sum_12 w, '
                                    'R = 10,000 EUR, w_b the 12-block instance weight'),
           'error': None}
    try:
        with open(SOURCE_RECORD) as handle:
            src_rec = json.load(handle)
        src_decl = H.validate_derived_instance(src_rec['derived_instance_declaration'])
        src_path = src_decl['case_path']
        src_bytes = open(src_path, 'rb').read()
        res['source'] = {'record': SOURCE_RECORD, 'declaration': src_decl,
                         'case_sha256_now': CG._sha_file(src_path),
                         'case_sha256_matches': CG._sha_file(src_path) == src_decl['case_sha256'],
                         'checksum_declared_matches_task': src_decl['scenario_checksum'] == SOURCE_CHECKSUM}
        src_case = json.loads(src_bytes)
        res['route_check_no_change_reproduces_source_bytes'] = json.dumps(src_case, indent='\t').encode() == src_bytes
        # derive
        derived = {}
        for year, day in TARGETS:
            case, changes = derive(src_case, year, day)
            name = f'SRP1__{label_of(year, day)}.json'
            path = os.path.join(out_dir, name)
            with open(path, 'x') as handle:
                handle.write(json.dumps(case, indent='\t'))
            derived[(year, day)] = {'path': os.path.relpath(path, REPO), 'changes': changes}
        # reads
        _log('reading the 12-block source')
        p12, r12 = production_read(src_path, os.path.join(out_dir, 'reads', 's56_srp1grid_3x3'))
        res['read_12_block'] = r12
        res['read_12_block']['checksum_matches'] = r12['combined_scenario_checksum'] == SOURCE_CHECKSUM
        res['sum12'] = sum12_weights(p12)
        with open(STEP3_RESULTS) as handle:
            step3 = json.load(handle)
        res['step3_results'] = {'path': STEP3_RESULTS, 'sha256': CG._sha_file(STEP3_RESULTS)}
        res['blocks'] = {}
        for year, day in TARGETS:
            bk = f'{year}|{day}'
            d = derived[(year, day)]
            _log(f'reading derived {bk}')
            pdv, rd = production_read(d['path'], os.path.join(out_dir, 'reads', label_of(year, day)))
            decl = {'instance_label': label_of(year, day), 'case_path': d['path'],
                    'case_sha256': CG._sha_file(d['path']), 'scenario_checksum': rd['combined_scenario_checksum'],
                    'source_case_path': src_path, 'source_case_sha256': src_decl['case_sha256'],
                    'changes_vs_source': d['changes']}
            decl = H.validate_derived_instance(decl)
            b = {'derived_instance_declaration': decl, 'read': rd,
                 'planning_years': [int(y) for y in pdv.transmission_network.years],
                 'planning_days': list(pdv.transmission_network.days),
                 'shared_ess_years': [int(y) for y in pdv.shared_ess_data.years]}
            _log(f'{bk}: G1 data')
            b['G1_data'] = gate_data(p12, pdv, year, day)
            b['G2_weights'] = weights(p12, pdv, year, day)
            w12 = b['G2_weights']['w_b_12_block']
            b['allowance_eur'] = (R_EUR / 4.0) * w12 / res['sum12']['sum_12']
            b['allowance_definition'] = '(R/4) * w_b(12-block) / sum_12 w, R = 10,000 EUR'
            win = sigma_window(pdv)
            b['G3_sigma'] = {'window_from_code': win,
                             'proxy': sigma_proxy(step3, bk, b['G2_weights']['w_b_derived'], win['window'])}
            sp, sprec = compensated_sigma(win['sigma_fixed'], w12, b['G2_weights']['w_b_derived'])
            b['G5_scale'] = dict(sprec, recorded_scale_sigma_prime_over_sigma=(sp / win['sigma_fixed']) if sp else None)
            res['blocks'][bk] = b
            del pdv
        del p12
        # builds (G4 / G5)
        scratch_root = os.path.abspath(args.scratch)
        ref_keys = [k for y, d in TARGETS for k in block_keys(y, d)]
        rc, ref, ref_path = run_child('ref12', {'name': 'ref12', 'derived': src_decl, 'sigma_prime': None,
                                                'nl_blocks': ref_keys,
                                                'scratch': os.path.join(scratch_root, 'ref12')}, out_dir)
        res['build_ref12'] = {'exit': rc, 'path': ref_path, 'error': ref.get('error'),
                              'build_checks': ref.get('build_checks'), 'guard': ref.get('guard'),
                              'acceptance': ref.get('acceptance')}
        for year, day in TARGETS:
            bk = f'{year}|{day}'
            b = res['blocks'][bk]
            name = label_of(year, day)
            rc, new, new_path = run_child(name, {'name': name, 'derived': b['derived_instance_declaration'],
                                                 'sigma_prime': b['G5_scale'].get('sigma_prime'),
                                                 'nl_blocks': block_keys(year, day),
                                                 'scratch': os.path.join(scratch_root, name)}, out_dir)
            b['G4_acceptance'] = {'exit': rc, 'path': new_path, 'error': new.get('error'),
                                  'acceptance': new.get('acceptance'),
                                  'investment_year_used': new.get('investment_year_used'),
                                  'build_checks': new.get('build_checks'), 'guard': new.get('guard'),
                                  'sigma_used': new.get('sigma_used'), 'effective_scale_p58': new.get('effective_scale'),
                                  'ref_effective_scale_p58': {k: v for k, v in (ref.get('effective_scale') or {}).items()
                                                              if k in block_keys(year, day)}}
            b['G4_acceptance']['holds'] = (rc == 0 and new.get('error') is None
                                           and all((new.get('build_checks') or {}).values())
                                           and (new.get('guard') or {}).get('verified_zero') is True)
            b['G5_nl'] = nl_compare(ref, new, block_keys(year, day))
            b['G5_nl']['effective_scales_equal'] = all(
                b['G4_acceptance']['effective_scale_p58'].get(k) == b['G4_acceptance']['ref_effective_scale_p58'].get(k)
                for k in block_keys(year, day))
        res['gates_summary'] = {bk: {'G1_data_bitwise': b['G1_data']['holds'],
                                     'G2_weight_ratio': b['G2_weights']['ratio_derived_over_12'],
                                     'G3_sigma_proxy_verdict': b['G3_sigma']['proxy']['verdict'],
                                     'G4_accepted': b['G4_acceptance']['holds'],
                                     'G4_default_investment_year_accepted':
                                         ((b['G4_acceptance'].get('acceptance') or {}).get('harness_default') or {})
                                         .get('accepted'),
                                     'G5_nl_identical_up_to_scale': b['G5_nl']['all_identical'],
                                     'allowance_eur': b['allowance_eur']}
                                for bk, b in res['blocks'].items()}
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        _log(res['traceback'])
    res['guard'] = CG.guard_report()
    if 'p515_s56_expected_profiles' in sys.modules:
        aa = sys.modules['p515_s56_expected_profiles']
        res['guard_audit_A_module'] = {'label': aa.GUARD.label, 'counts': dict(aa.GUARD.counts),
                                       'verify_0': aa.GUARD.verify(0)}
    res['wall_s'] = time.time() - t_start
    res['utc_end'] = _utc()
    path = os.path.join(out_dir, 'step4_instances.json')
    with open(path, 'x') as handle:
        GRIO.dump(res, handle, indent=1, default=GRIO.json_default_item)
    outputs = {}
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            outputs[os.path.relpath(p, REPO)] = CG._sha_file(p)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump({'outputs': outputs, 'script': {'path': SCRIPT, 'sha256': res['script_sha256']},
                   'source_case': {'path': SOURCE_RECORD}}, handle, indent=1)
    _log(f"wrote {path}; error {res['error']}; summary {res.get('gates_summary')}; guard {res['guard']['verified_zero']}")
    return 0 if (res['error'] is None and res['guard']['verified_zero']
                 and not (res.get('guard_audit_A_module') or {}).get('verify_0')) else 1


# ======================================================================================================================
#  instance B (Planner ruling 2 of 2026-11-23): ONE derived case, Years {2025, 2035} x Days {Summer}
# ======================================================================================================================
# The 2035-only instance failed G1 because production grows prices and loads from the instance's FIRST year; keeping
# 2025 as years[0] makes every block's data and weight equal to the 12-block instance's by construction. 2025|Summer
# rides along. Same gates G1-G5, per block, against the 12-block instance; the 2035-only artifacts are retired (kept).
INSTANCE_B = {'label': 's56_step4_2025_2035_Summer', 'years': ('2025', '2035'), 'days': ('Summer',),
              'blocks': (('2025', 'Summer'), ('2035', 'Summer'))}


def derive_multi(source_case, years, days):
    case = json.loads(json.dumps(source_case))
    changes = [{'key': 'Years', 'from': case['Years'], 'to': {y: case['Years'][y] for y in years}},
               {'key': 'Days', 'from': case['Days'], 'to': {d: case['Days'][d] for d in days}}]
    case['Years'] = {y: case['Years'][y] for y in years}
    case['Days'] = {d: case['Days'][d] for d in days}
    return case, changes


def main_parent_instance_b(args):
    t_start = time.time()
    out_dir = os.path.abspath(args.out_dir)
    pre = []
    if os.environ.get('NLP_SOLVER_PATH') != CG.SOLVER_PATH:
        pre.append(f'NLP_SOLVER_PATH must be {CG.SOLVER_PATH}')
    if os.path.exists(out_dir):
        pre.append(f'{out_dir} exists')
    if os.path.exists(os.path.join(REPO, '.p515_s44_campaign.lock')):
        pre.append('campaign lock exists')
    others = other_processes()
    if others:
        pre.append(f'other project processes: {others}')
    if pre:
        _log(f'REFUSED: {pre}')
        return 2
    os.makedirs(out_dir)
    inst = INSTANCE_B
    res = {'schema': SCHEMA + '_instance_b', 'utc_start': _utc(), 'script': SCRIPT, 'script_sha256': CG._sha_file(SCRIPT),
           'provenance': CG.provenance(), 'R_eur': R_EUR, 'coupling': COUPLING, 'premium': PREMIUM,
           'instance_b': {k: list(v) if isinstance(v, tuple) else v for k, v in inst.items()},
           'authority': 'Planner ruling 2 of 2026-11-23 (TASKS.md fc5dc766): instance B = SRP1-grid 3 x 3 restricted '
                        'to Years {2025, 2035} x Days {Summer}; gates G1-G5 for both blocks against the 12-block '
                        'instance; the 2035-only instance is retired (kept, not used)',
           'objective_convention': ('allowances in EUR of total discounted planning cost: (R/4) w_b / sum_12 w, '
                                    'R = 10,000 EUR, w_b the 12-block instance weight'),
           'error': None}
    try:
        with open(SOURCE_RECORD) as handle:
            src_rec = json.load(handle)
        src_decl = H.validate_derived_instance(src_rec['derived_instance_declaration'])
        src_path = src_decl['case_path']
        src_bytes = open(src_path, 'rb').read()
        res['source'] = {'record': SOURCE_RECORD, 'declaration': src_decl,
                         'case_sha256_now': CG._sha_file(src_path),
                         'case_sha256_matches': CG._sha_file(src_path) == src_decl['case_sha256'],
                         'checksum_declared_matches_task': src_decl['scenario_checksum'] == SOURCE_CHECKSUM}
        src_case = json.loads(src_bytes)
        res['route_check_no_change_reproduces_source_bytes'] = json.dumps(src_case, indent='\t').encode() == src_bytes
        case, changes = derive_multi(src_case, inst['years'], inst['days'])
        path = os.path.join(out_dir, f"SRP1__{inst['label']}.json")
        with open(path, 'x') as handle:
            handle.write(json.dumps(case, indent='\t'))
        case_rel = os.path.relpath(path, REPO)
        _log('reading the 12-block source')
        p12, r12 = production_read(src_path, os.path.join(out_dir, 'reads', 's56_srp1grid_3x3'))
        res['read_12_block'] = r12
        res['read_12_block']['checksum_matches'] = r12['combined_scenario_checksum'] == SOURCE_CHECKSUM
        res['sum12'] = sum12_weights(p12)
        with open(STEP3_RESULTS) as handle:
            step3 = json.load(handle)
        res['step3_results'] = {'path': STEP3_RESULTS, 'sha256': CG._sha_file(STEP3_RESULTS)}
        _log(f"reading derived {inst['label']}")
        pdv, rd = production_read(case_rel, os.path.join(out_dir, 'reads', inst['label']))
        decl = H.validate_derived_instance({
            'instance_label': inst['label'], 'case_path': case_rel, 'case_sha256': CG._sha_file(case_rel),
            'scenario_checksum': rd['combined_scenario_checksum'], 'source_case_path': src_path,
            'source_case_sha256': src_decl['case_sha256'], 'changes_vs_source': changes})
        res['derived_instance_declaration'] = decl
        res['read'] = rd
        res['planning_years'] = [int(y) for y in pdv.transmission_network.years]
        res['planning_days'] = list(pdv.transmission_network.days)
        res['shared_ess_years'] = [int(y) for y in pdv.shared_ess_data.years]
        res['blocks'] = {}
        sigma_primes = {}
        for year, day in inst['blocks']:
            bk = f'{year}|{day}'
            b = {}
            _log(f'{bk}: G1 data')
            b['G1_data'] = gate_data(p12, pdv, year, day)
            b['G2_weights'] = weights(p12, pdv, year, day)
            w12 = b['G2_weights']['w_b_12_block']
            b['allowance_eur'] = (R_EUR / 4.0) * w12 / res['sum12']['sum_12']
            b['allowance_definition'] = '(R/4) * w_b(12-block) / sum_12 w, R = 10,000 EUR'
            win = sigma_window(pdv)
            b['G3_sigma'] = {'window_from_code': win,
                             'proxy': sigma_proxy(step3, bk, b['G2_weights']['w_b_derived'], win['window']),
                             'note': ('instance B: sigma is the max over BOTH blocks\' agents at the initialisation '
                                      'solution; settled by the D-3 initialisation round')}
            sp, sprec = compensated_sigma(win['sigma_fixed'], w12, b['G2_weights']['w_b_derived'])
            b['G5_scale'] = dict(sprec, recorded_scale_sigma_prime_over_sigma=(sp / win['sigma_fixed']) if sp else None)
            sigma_primes[bk] = sp
            res['blocks'][bk] = b
        del pdv
        del p12
        # one sigma per derived build: every block must ask for the same sigma' (weight ratio 1 by construction)
        distinct = sorted({v for v in sigma_primes.values() if v is not None})
        res['sigma_prime_single_value_for_the_instance'] = {'per_block': sigma_primes, 'distinct': distinct,
                                                            'holds': len(distinct) == 1
                                                            and None not in sigma_primes.values()}
        sigma_prime = distinct[0] if len(distinct) == 1 else None
        scratch_root = os.path.abspath(args.scratch)
        b_keys = [k for y, d in inst['blocks'] for k in block_keys(y, d)]
        rc, ref, ref_path = run_child('ref12', {'name': 'ref12', 'derived': src_decl, 'sigma_prime': None,
                                                'nl_blocks': b_keys,
                                                'scratch': os.path.join(scratch_root, 'ref12')}, out_dir)
        res['build_ref12'] = {'exit': rc, 'path': ref_path, 'error': ref.get('error'),
                              'build_checks': ref.get('build_checks'), 'guard': ref.get('guard'),
                              'acceptance': ref.get('acceptance')}
        rc, new, new_path = run_child(inst['label'], {'name': inst['label'], 'derived': decl,
                                                      'sigma_prime': sigma_prime, 'nl_blocks': b_keys,
                                                      'scratch': os.path.join(scratch_root, inst['label'])}, out_dir)
        g4 = {'exit': rc, 'path': new_path, 'error': new.get('error'), 'acceptance': new.get('acceptance'),
              'investment_year_used': new.get('investment_year_used'), 'build_checks': new.get('build_checks'),
              'guard': new.get('guard'), 'sigma_used': new.get('sigma_used'),
              'effective_scale_p58': new.get('effective_scale'),
              'ref_effective_scale_p58': {k: v for k, v in (ref.get('effective_scale') or {}).items() if k in b_keys}}
        g4['holds'] = (rc == 0 and new.get('error') is None and all((new.get('build_checks') or {}).values())
                       and (new.get('guard') or {}).get('verified_zero') is True)
        res['G4_acceptance'] = g4
        for year, day in inst['blocks']:
            bk = f'{year}|{day}'
            b = res['blocks'][bk]
            b['G4_acceptance'] = {'holds': g4['holds'], 'note': 'one build for the instance (both blocks)'}
            b['G5_nl'] = nl_compare(ref, new, block_keys(year, day))
            b['G5_nl']['effective_scales_equal'] = all(
                (g4['effective_scale_p58'] or {}).get(k) == g4['ref_effective_scale_p58'].get(k)
                for k in block_keys(year, day))
        res['gates_summary'] = {bk: {'G1_data_bitwise': b['G1_data']['holds'],
                                     'G2_weight_ratio': b['G2_weights']['ratio_derived_over_12'],
                                     'G3_sigma_proxy_verdict': b['G3_sigma']['proxy']['verdict'],
                                     'G4_accepted': b['G4_acceptance']['holds'],
                                     'G4_default_investment_year_accepted':
                                         ((g4.get('acceptance') or {}).get('harness_default') or {}).get('accepted'),
                                     'G5_nl_identical_up_to_scale': b['G5_nl']['all_identical'],
                                     'G5_effective_scales_equal': b['G5_nl']['effective_scales_equal'],
                                     'allowance_eur': b['allowance_eur']}
                                for bk, b in res['blocks'].items()}
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        _log(res['traceback'])
    res['guard'] = CG.guard_report()
    if 'p515_s56_expected_profiles' in sys.modules:
        aa = sys.modules['p515_s56_expected_profiles']
        res['guard_audit_A_module'] = {'label': aa.GUARD.label, 'counts': dict(aa.GUARD.counts),
                                       'verify_0': aa.GUARD.verify(0)}
    res['wall_s'] = time.time() - t_start
    res['utc_end'] = _utc()
    path = os.path.join(out_dir, 'step4_instance_b.json')
    with open(path, 'x') as handle:
        GRIO.dump(res, handle, indent=1, default=GRIO.json_default_item)
    outputs = {}
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            outputs[os.path.relpath(p, REPO)] = CG._sha_file(p)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump({'outputs': outputs, 'script': {'path': SCRIPT, 'sha256': res['script_sha256']},
                   'source_case': {'path': SOURCE_RECORD}}, handle, indent=1)
    _log(f"wrote {path}; error {res['error']}; summary {res.get('gates_summary')}; guard {res['guard']['verified_zero']}")
    return 0 if (res['error'] is None and res['guard']['verified_zero']
                 and not (res.get('guard_audit_A_module') or {}).get('verify_0')) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir')
    ap.add_argument('--scratch')
    ap.add_argument('--instance-b', action='store_true',
                    help='build instance B (Years {2025, 2035} x Days {Summer}) and its gates G1-G5 (both blocks)')
    ap.add_argument('--child-build', action='store_true')
    ap.add_argument('--child-in')
    ap.add_argument('--child-out')
    args = ap.parse_args()
    if args.child_build:
        return child_build(args)
    if not (args.out_dir and args.scratch):
        ap.error('--out-dir and --scratch are required')
    if args.instance_b:
        return main_parent_instance_b(args)
    return main_parent(args)


if __name__ == '__main__':
    sys.exit(main())
