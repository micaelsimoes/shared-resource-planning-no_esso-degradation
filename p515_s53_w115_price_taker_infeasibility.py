"""P5.15 W115 -- which constraint makes the PRICE-TAKER-arm TSO block 2025 Spring infeasible. DIAGNOSTIC ONLY.

A THIN WRAPPER around W114's p515_s53_w114_passive_infeasibility.py (code 0e8f209f), which is IMPORTED, not edited
(its sha256 stays the one W114's evidence b0ed4d14 records). Every helper is W114's own, called unchanged: its
SolveProfileGuard (installed at import, same permitted sites: uncoordinated_benchmark.py:_solve_block and W114's
elastic_solve), _check_guard, tn_state, schedule_table, build_elastic_copy (with its E1 classification that raises on
an unclassified family), elastic_solve, read_slacks, interface_moves, compare_to_w113 and the ELASTIC_VARIANTS E1-E4.
Only W114's run() hard-codes the arm and the block, so this file carries the same sequence with those parameterised:

    arm          price_taker (UB.ARM_PRICE_TAKER), decision tie-breaker TIE_BREAKER['decision']['price_taker_dso'] = 0
    block        TSO 2025 Spring (the arm's first TSO block, where W114 Part A's three price-taker starts failed)
    start        cold
    reference    W114 Part A evidence 209f4829: w106_uncoordinated_settled/arm_price_taker_cold/{per_solve_record.jsonl,
                 failure.json} (the TSO primary attempt's constraint violation 0.04068590331761268)

Two ADDED zero-solve reads (no W114 logic changed): the TN generators' P/Q bounds on the arm block itself (W114's
tn_state has no P lower bound), and the per-bus net load / net generation expressions (pc_node, qc_node, pg_node,
qg_node) at each elastic solution, from which the TN's losses and its feasible total-interface range at that solution
are derived (range definition stated in the output). Output keys W114 names 'passive_*' in schedule_table are
renamed 'price_taker_*'; compare_to_w113's 'w114' / 'w113' keys are renamed 'this_run' / 'reference'.

SOLVE COUNT (declared in advance, W114's SolveProfileGuard, checked EXACTLY at every boundary):
  3 DSO solves (price-taker, cold, 2025 Spring, nodes 5/7/9) + 3 TSO attempts (expected ArmSolveFailure after primary,
  tier-1, tier-2) + 4 elastic (E1-E4, one attempt each) = 10 solves = 10 process launches.

INSTANCE: SRP1, x = 0 (candidate_key 8435c71859ddde68...), block TSO 2025 Spring; the coordinated schedule is the
settled cycle-181 x = 0 cell d110bd1a5977df1e_x0 (certified_models.pkl sha256 99ab1070..., verified before
unpickling). The TSO build's targets for every OTHER (year, day) are the coordinated DSO schedule as a placeholder
(those TSO blocks are never solved), as in W114.

OUTPUT (write-once, new directory): data/SRP1/Results/P515S53/w115_price_taker_infeasibility/
    w115_price_taker_infeasibility.json, per_solve_record.jsonl, manifest_sha256.json (--manifest, after the run)
IPOPT logs: data/SRP1/Results/P56A/evals/p515s53w115_price_taker_infeasibility/logs (gitignored; hash-recorded by
--manifest). The elastic logs carry W114's suffix 'w114_<variant>' (W114's elastic_solve, unchanged).

LAUNCH (attached, alone, both streams captured), then the manifest:
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w115_price_taker_infeasibility.py > data/SRP1/Results/P515S53/w115_price_taker_infeasibility_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w115_price_taker_infeasibility.py --manifest
Zero-solve build check (W114's blocking guard; writes only to the given scratch directory):
    ... p515_s53_w115_price_taker_infeasibility.py --dry-run <scratch dir>

Objective convention: no Q is computed here (Q = gross_operational_cost, settlement excluded, elsewhere); the
`active_objective_value` fields are the blocks' own decision objectives, the elastic objectives are slack sums.
"""
import copy
import json
import os
import sys
import time

# W114's module installs its SolveProfileGuard at import (blocking under --dry-run, none under --manifest).
import p515_s53_w114_passive_infeasibility as W  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

THIS_FILE = os.path.basename(os.path.abspath(__file__))
REPO = W.REPO
ARM = 'price_taker'
START = 'cold'
YEAR, DAY = 2025, 'Spring'
OUT_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w115_price_taker_infeasibility')
LAUNCH_LOG_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w115_price_taker_infeasibility_launch.log')
EVAL_ID = 'p515s53w115_price_taker_infeasibility'
REF_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w106_uncoordinated_settled', 'arm_price_taker_cold')
DECLARED_SOLVES = {'dso_price_taker_cold_2025_spring': 3, 'tso_arm_block_reproduction': 3, 'elastic': 4}
DECLARED_TOTAL = sum(DECLARED_SOLVES.values())
if W._GUARD is not None:
    W._GUARD.label = 'P5.15 W115 price-taker infeasibility (W114 guard, imported)'
_T0 = time.time()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W115 +{time.time() - _T0:7.1f}s] {msg}', flush=True)


def _rename(obj, old, new):
    if isinstance(obj, dict):
        return {(k.replace(old, new) if isinstance(k, str) else k): _rename(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename(v, old, new) for v in obj]
    return obj


def _reference_records():
    recs = {}
    with open(W._abs(os.path.join(REF_DIR, 'per_solve_record.jsonl'))) as handle:
        for line in handle:
            r = json.loads(line)
            recs[r['block']] = r
    with open(W._abs(os.path.join(REF_DIR, 'failure.json'))) as handle:
        failure = json.load(handle)
    return recs, failure


def _compare(record, reference):
    out = W.compare_to_w113(record, reference)
    out['this_run'] = out.pop('w114')
    out['reference'] = out.pop('w113')
    out['active_objective_value_this_run'] = out.pop('active_objective_value_w114')
    out['active_objective_value_reference'] = out.pop('active_objective_value_w113')
    return out


def generator_bounds(block, network):
    """ADDED zero-solve read: every TN generator's P and Q bounds per hour on the arm block (MW / MVAr)."""
    base = network.baseMVA
    sm, so = next(iter(block.scenarios_market)), next(iter(block.scenarios_operation))
    out = []
    for g in block.generators:
        gen = network.generators[g]
        rows = []
        for p in block.periods:
            pg, qg = block.pg[g, sm, so, p], block.qg[g, sm, so, p]
            rows.append({'hour': p + 1, 'pg_fixed': bool(pg.fixed),
                         'pg_lb_mw': None if pg.lb is None else pg.lb * base,
                         'pg_ub_mw': None if pg.ub is None else pg.ub * base,
                         'qg_lb_mvar': None if qg.lb is None else qg.lb * base,
                         'qg_ub_mvar': None if qg.ub is None else qg.ub * base})
        out.append({'gen_idx': g, 'bus': gen.bus, 'gen_type': getattr(gen, 'gen_type', None),
                    'curtaillable': bool(gen.is_curtaillable()), 'per_hour': rows})
    return out


def tn_balance(model, network, gen_bounds):
    """ADDED zero-solve read at a solution: per hour and bus the net load / net generation expressions, the TN's losses
    (sum pg_node - sum pc_node over every bus: branch and shunt losses plus any node-balance slack), and the feasible
    TOTAL interface P range AT THIS SOLUTION'S LOSSES, defined as
        [sum Pg_min - losses - non-ADN net load, sum Pg_max - losses - non-ADN net load]   (import positive),
    i.e. the range of the summed DN import the TN generators can serve holding losses fixed (a first-order range,
    not a re-optimisation). The same for Q with the Q bounds (reactive losses include line charging)."""
    base = network.baseMVA
    adn = set(network.active_distribution_network_nodes)
    sm, so = next(iter(model.scenarios_market)), next(iter(model.scenarios_operation))
    hours = []
    for p in model.periods:
        buses = []
        for i in model.nodes:
            node = network.nodes[i]
            buses.append({'bus': node.bus_i, 'is_adn_interface': node.bus_i in adn,
                          'pc_node_mw': (W._val(model.pc_node[i, sm, so, p]) or 0.0) * base,
                          'qc_node_mvar': (W._val(model.qc_node[i, sm, so, p]) or 0.0) * base,
                          'pg_node_mw': (W._val(model.pg_node[i, sm, so, p]) or 0.0) * base,
                          'qg_node_mvar': (W._val(model.qg_node[i, sm, so, p]) or 0.0) * base})
        pc_all = sum(b['pc_node_mw'] for b in buses)
        qc_all = sum(b['qc_node_mvar'] for b in buses)
        pg_all = sum(b['pg_node_mw'] for b in buses)
        qg_all = sum(b['qg_node_mvar'] for b in buses)
        pc_non_adn = sum(b['pc_node_mw'] for b in buses if not b['is_adn_interface'])
        qc_non_adn = sum(b['qc_node_mvar'] for b in buses if not b['is_adn_interface'])
        losses_p, losses_q = pg_all - pc_all, qg_all - qc_all
        pmin = sum(g['per_hour'][p]['pg_lb_mw'] or 0.0 for g in gen_bounds)
        pmax = sum(g['per_hour'][p]['pg_ub_mw'] or 0.0 for g in gen_bounds)
        qmin = sum(g['per_hour'][p]['qg_lb_mvar'] or 0.0 for g in gen_bounds)
        qmax = sum(g['per_hour'][p]['qg_ub_mvar'] or 0.0 for g in gen_bounds)
        hours.append({'hour': p + 1, 'buses': buses, 'pg_total_mw': pg_all, 'qg_total_mvar': qg_all,
                      'pc_total_mw': pc_all, 'qc_total_mvar': qc_all,
                      'pc_adn_interface_buses_mw': pc_all - pc_non_adn, 'pc_non_adn_mw': pc_non_adn,
                      'qc_adn_interface_buses_mvar': qc_all - qc_non_adn, 'qc_non_adn_mvar': qc_non_adn,
                      'losses_p_mw': losses_p, 'losses_q_mvar': losses_q,
                      'pg_bounds_total_mw': [pmin, pmax], 'qg_bounds_total_mvar': [qmin, qmax],
                      'feasible_interface_p_range_mw_at_these_losses': [pmin - losses_p - pc_non_adn,
                                                                        pmax - losses_p - pc_non_adn],
                      'feasible_interface_q_range_mvar_at_these_losses': [qmin - losses_q - qc_non_adn,
                                                                          qmax - losses_q - qc_non_adn]})
    return hours


def run(dry_run_dir=None):
    import pyomo.environ as pe  # noqa: F401
    import p515_s53_w93_uncoordinated_benchmark as B
    import shared_resources_planning as srp
    import uncoordinated_benchmark as UB
    import p56a_oracle as O

    out_dir = W._abs(OUT_DIR_REL) if dry_run_dir is None else dry_run_dir
    eval_dir = os.path.join(O.WORK_DIR, EVAL_ID)
    lock = False
    if dry_run_dir is None:
        failures = B.check_preconditions(out_dir)
        if os.path.exists(eval_dir):
            failures.append(f'IPOPT eval dir already exists (write-once): {eval_dir}')
        if failures:
            print('REFUSING TO RUN w115_price_taker_infeasibility:', *failures, sep='\n  ', flush=True)
            return 2
        B.acquire_lock('w115_price_taker_infeasibility')
        lock = True
    try:
        os.makedirs(out_dir, exist_ok=False)
        _log(f'W115 price-taker infeasibility diagnostic ({"DRY RUN, zero solves" if dry_run_dir else "declared solves "}'
             f'{"" if dry_run_dir else DECLARED_SOLVES}); output {out_dir}')
        W._check_guard(0, 'start')
        if dry_run_dir is None:
            planning = O.fresh_planning(EVAL_ID)
        else:
            planning = copy.deepcopy(O.load_baseline()['planning'])
        solver_options = B.apply_arm_solver_options(srp, planning)
        candidate = B._x0_candidate(srp, planning)
        tn = planning.transmission_network
        Y, D = W._key(tn.years, YEAR), W._key(tn.days, DAY)
        network = tn.network[Y][D]
        tie_dso = B.TIE_BREAKER['decision'][f'{ARM}_dso']

        certified = B._load_pickle_verified(B.COORDINATED['certified_models'])
        reference = UB.coordinated_reference_structure(planning, certified)
        coord_dso = UB.get_dso_interface_schedule(planning, certified['dso'])
        coord_tso = UB.get_tso_interface_schedule(planning, certified['tso'])
        coord_state = W.tn_state(certified['tso'][Y][D], network)
        del certified

        dso_models, dso_build = UB.build_dso_arm_models(planning, candidate['total_capacity'], arm=UB.ARM_PRICE_TAKER,
                                                        curtailment_penalty=tie_dso)
        structure = UB.check_arm_structures(planning, {'dso': dso_models}, reference, dso_build_record=dso_build)
        start_records = UB.apply_start(planning, {'tso': None, 'dso': dso_models}, start=UB.START_COLD,
                                       agents=('DSO',))
        result = {'stage': 'P5.15 W115 price-taker-arm TSO 2025 Spring infeasibility diagnostic', 'utc_start': W._utc(),
                  'dry_run': dry_run_dir is not None,
                  'git_head': B.H._git(['rev-parse', 'HEAD']), 'script': THIS_FILE,
                  'script_sha256': B.H.sha256_file(os.path.abspath(__file__)),
                  'wrapped_script': W.THIS_FILE, 'wrapped_script_sha256': B.H.sha256_file(W.__file__),
                  'module_sha256': {n: B.H.sha256_file(W._abs(n)) for n in B.FROZEN_SPEC_BOUND_FILES},
                  'frozen_benchmark_spec': B._frozen_spec_identity(),
                  'instance': {'problem': 'SRP1', 'label': 'x0', 'candidate_key': B.X0['candidate_key'],
                               'block': f'TSO|-|{YEAR}|{DAY}', 'arm': ARM, 'start': START,
                               'dso_decision_tie_breaker': tie_dso,
                               'tso_decision_tie_breaker': B.TIE_BREAKER['decision']['tso'],
                               'coordinated_cell': B.COORDINATED['eval_key'],
                               'coordinated_models_sha256': B.COORDINATED['certified_models']['sha256']},
                  'reference_dir': REF_DIR,
                  'diagnostic_only': ('elastic copies are diagnostics; they never feed any arm value and no arm '
                                      'definition changes'),
                  'declared_solves': DECLARED_SOLVES, 'declared_total': DECLARED_TOTAL,
                  'solver_options_arm': solver_options, 'elastic_option_overrides': W.ELASTIC_OPTION_OVERRIDES,
                  'slack_report_threshold_model_units': W.SLACK_REPORT_THRESHOLD,
                  'key_renames': {'schedule_table': "W114 'passive_*' -> 'price_taker_*'",
                                  'compare_to_w113': "'w114'/'w113' -> 'this_run'/'reference'"},
                  f'dso_structure_check_{YEAR}_{DAY.lower()}': {k: v for k, v in structure.items()
                                                                if k.endswith(f'|{YEAR}|{DAY}')},
                  f'dso_start_records_{YEAR}_{DAY.lower()}': {k: v for k, v in start_records.items()
                                                              if k.endswith(f'|{YEAR}|{DAY}')},
                  f'coordinated_tn_state_{YEAR}_{DAY.lower()}': coord_state, 'guard': []}
        sink = None if dry_run_dir else B.SolveSink(out_dir)
        ref_recs, ref_failure = _reference_records()

        # ---- step 1: the three price-taker DSO blocks of the block's (year, day), cold
        dso_records = {}
        arm_sched = copy.deepcopy(coord_dso)
        if dry_run_dir is None:
            for node_id in sorted(planning.distribution_networks):
                dn = planning.distribution_networks[node_id]
                y, d = W._key(dn.years, YEAR), W._key(dn.days, DAY)
                _res, rec = UB._solve_block(planning, dn, dn.network[y][d], dso_models[node_id][y][d], kind='DSO',
                                            node_id=node_id, year=y, day=d, phase=f'w115:{ARM}:{START}:dso',
                                            record_callback=sink)
                dso_records[rec['block']] = _compare(rec, ref_recs[rec['block']])
            result['guard'].append(W._check_guard(DECLARED_SOLVES['dso_price_taker_cold_2025_spring'], 'after DSO'))
            sched = UB.get_dso_interface_schedule(planning, dso_models)
            for node_id in arm_sched:
                arm_sched[node_id][Y][D] = sched[node_id][W._key(sched[node_id].keys(), YEAR)][DAY]
        result['dso_reproduction_vs_reference'] = dso_records
        arm_block_sched = {n: arm_sched[n][Y][D] for n in arm_sched}
        result['targets_note'] = (f'TSO build targets: {YEAR} {DAY} = the {ARM} schedule (these three DSO solves); every '
                                  'other (year, day) = the settled coordinated DSO schedule as a placeholder (those '
                                  'TSO blocks are never solved)' if dry_run_dir is None else
                                  'DRY RUN: every (year, day) = the coordinated schedule (no DSO solve)')
        result[f'schedule_{YEAR}_{DAY.lower()}_{ARM}_vs_coordinated'] = _rename(
            W.schedule_table(arm_sched, coord_dso, coord_tso, Y, D), 'passive', ARM)

        # ---- step 2: the TSO arm block at that schedule, and its reproduction
        tso_model, tso_build = UB.build_tso_arm_model(planning, candidate['total_capacity'], arm_sched,
                                                      curtailment_penalty=B.TIE_BREAKER['decision']['tso'],
                                                      coupling=UB.TSO_COUPLING_FIXED,
                                                      pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
        tso_structure = UB.check_arm_structures(planning, {'tso': tso_model}, reference, tso_build_record=tso_build,
                                                tso_coupling=UB.TSO_COUPLING_FIXED)
        UB.apply_start(planning, {'tso': tso_model, 'dso': None}, start=UB.START_COLD, agents=('TSO',))
        block = tso_model[Y][D]
        gen_bounds = generator_bounds(block, network)
        result[f'tso_structure_check_{YEAR}_{DAY.lower()}'] = {k: v for k, v in tso_structure.items()
                                                               if k.endswith(f'|{YEAR}|{DAY}')}
        result['tn_generator_bounds_arm_block'] = gen_bounds
        result[f'tso_build_point_capacity_{YEAR}_{DAY.lower()}'] = [
            {'hour': h['hour'], 'pg_capacity_total_mw': h['pg_capacity_total_mw'],
             'qg_capacity_total_mvar': h['qg_capacity_total_mvar'],
             'generators_pg_ub_mw': {str(g['gen_idx']): g['pg_ub_mw'] for g in h['generators']}}
            for h in W.tn_state(block, network)]
        if dry_run_dir is None:
            try:
                _r, rec = UB._solve_block(planning, tn, network, block, kind='TSO', node_id=None, year=Y, day=D,
                                          phase=f'w115:{ARM}:{START}:tso', record_callback=sink)
                result['tso_reproduction'] = {'raised_arm_solve_failure': False, 'record_summary': rec['summary']}
            except UB.ArmSolveFailure as failure:
                result['tso_reproduction'] = {'raised_arm_solve_failure': True, 'error': str(failure),
                                              'vs_reference': _compare(failure.record, ref_failure['record'])}
            result['guard'].append(W._check_guard(DECLARED_SOLVES['dso_price_taker_cold_2025_spring']
                                                  + DECLARED_SOLVES['tso_arm_block_reproduction'], 'after TSO repro'))
            _log(f"TSO reproduction: {result['tso_reproduction'].get('vs_reference', {}).get('identical_bitwise')}")

        # ---- step 3: the elastic copies (W114's builders, variants and readers)
        result['elastic'] = {}
        for label, kind, rows in W.ELASTIC_VARIANTS:
            c, targets, bound_map, classification = W.build_elastic_copy(block, kind, rows)
            entry = {'kind': kind, 'target_components': [t.name for t in targets],
                     'n_slack_vars': len(list(c.component('_core_add_slack_variables').component_objects(pe.Var))),
                     'classification': classification}
            if dry_run_dir is None:
                entry['solve'] = W.elastic_solve(planning, tn, network, c, label)
                entry['slacks'] = W.read_slacks(c, targets, network, bound_map)
                entry['elastic_objective_value'] = W._val(c.component('_core_add_slack_variables')._slack_objective)
                entry[f'interface_accepted_vs_{ARM}_target'] = W.interface_moves(c, network, arm_block_sched)
                entry['tn_state'] = W.tn_state(c, network)
                entry['tn_balance'] = tn_balance(c, network, gen_bounds)
                _log(f"{label}: {entry['solve']['termination_condition']} loaded {entry['solve']['solution_loaded']} "
                     f"slack total {entry['slacks']['slack_total_model_units']:.6g} nonzero "
                     f"{ {k: v['n_nonzero'] for k, v in entry['slacks']['per_family'].items()} }")
            else:
                entry['slacks_dry'] = W.read_slacks(c, targets, network, bound_map, allow_none=True)
                entry['tn_state_dry_hours'] = len(W.tn_state(c, network))
                entry['tn_balance_dry_hours'] = len(tn_balance(c, network, gen_bounds))
            result['elastic'][label] = entry
        if dry_run_dir is None:
            result['guard'].append(W._check_guard(DECLARED_TOTAL, 'end'))
        else:
            result['guard'].append(W._check_guard(0, 'dry-run end'))
        result['utc_end'] = W._utc()
        path = os.path.join(out_dir, 'w115_price_taker_infeasibility.json')
        with open(path, 'x') as handle:
            GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
        _log(f'wrote {path}; guard {dict(W._GUARD.counts)}')
        return 0
    finally:
        if W._GUARD is not None:
            W._GUARD.uninstall()
        if lock:
            B.release_lock()


def manifest():
    import p515_s44_campaign_harness as H
    out_dir = W._abs(OUT_DIR_REL)
    entries = {}
    for name in sorted(os.listdir(out_dir)):
        if name != 'manifest_sha256.json':
            entries[os.path.join(OUT_DIR_REL, name)] = H.sha256_file(os.path.join(out_dir, name))
    entries[LAUNCH_LOG_REL] = H.sha256_file(W._abs(LAUNCH_LOG_REL))
    logs = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals', EVAL_ID, 'logs')
    ipopt = {os.path.join(logs, n): H.sha256_file(W._abs(os.path.join(logs, n)))
             for n in sorted(os.listdir(W._abs(logs)))}
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump({'files': entries, 'ipopt_logs_gitignored_hash_record': ipopt}, handle, indent=1, sort_keys=True)
    print(f'manifest: {len(entries)} files, {len(ipopt)} IPOPT logs')
    return 0


if __name__ == '__main__':
    if W.MANIFEST:
        sys.exit(manifest())
    if W.DRY_RUN:
        sys.exit(run(dry_run_dir=sys.argv[sys.argv.index('--dry-run') + 1]))
    sys.exit(run())
