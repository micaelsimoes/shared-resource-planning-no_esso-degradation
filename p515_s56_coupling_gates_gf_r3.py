"""
P5.15 S56 step 1 COMPLETION -- G-f THIRD attempt (option on) and its option-OFF CONTROL (launcher only; no gate logic
for the on arm).

Declared by the addendum spec `frozen_coupling_gates_addendum_gf_r3_v1_<hash>.json` (predecessor: the frozen gate
spec `frozen_coupling_gates_spec_v1_10c98258.json`, not edited). Why a third attempt: G-f r1 was refused by the
capture-path assertion (fixed in 21c49cec); G-f r2 raised at ADMM cycle 1 in the S38 PF entry-capture identity
(expectation-only reconstruction against production's omega-weighted total). The step-1 completion fixed that check and
the other expectation-only harness sites, re-took `p58_rescale`'s rescaled objective after the per-scenario wiring (the
solved objective had carried NO deviation term -- verified zero-solve), and applied author decision 6's Q convention.

  --arm on       imports the frozen gate script UNCHANGED (`p515_s56_coupling_gates.py`, sha256 in the spec),
                 replaces only its module constant GF_ROOT with 's56cpl_gf_smoke_r3' and calls its own `smoke()`
                 (configuration, options, cap, checks and verdict writer are the frozen script's) -- as the r2 launcher.
  --arm control  the same instance (SRP1-grid 3 x 3), candidate (x0), cap (2), configuration and row-18 premium
                 (alpha 0.5) with the per-scenario coupling OFF (no `interface_coupling` option), through the frozen
                 script's own `_run_campaign`, into 's56cpl_gf_control_off'; its own verdict file.
  --compare      zero solves: the cycle-1 / cycle-2 interface movements of the two runs (production's [TSO PROX]
                 lines; the S38 PF stride's z_TSO current vs previous per entry), both Q conventions, the new per-cycle
                 fields; written to `gf_r3_vs_control_<utc>.json`.

Usage (repo root, canonical interpreter, attached, alone, both streams captured; on, then control, then compare):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_coupling_gates_gf_r3.py --arm on \
      --spec-sha256 <S> --addendum-sha256 <A> > <log> 2>&1
"""
import argparse
import json
import math
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s56_coupling_gates as G  # noqa: E402  (installs its zero-solve guard on this process, as when run directly)

GF_ROOT_R3 = 's56cpl_gf_smoke_r3'
CONTROL_ROOT = 's56cpl_gf_control_off'
ADDENDUM_PREFIX = 'frozen_coupling_gates_addendum_gf_r3_v1_'
CONTROL_OPTIONS = {'interface_deviation_premium': dict(G.ALPHA_05)}   # G-f's options without `interface_coupling`
NEW_Q_FIELDS = ('net_operational_recourse_settlement_excluded_in_full', 'gross_operational_cost_contracted_excluded',
                'net_operational_recourse_contracted_excluded')
R2_EVAL = os.path.join('campaign_s56cpl_gf_smoke_r2', 'evals', '972942a49f415577_x0')
TSO_PROX = re.compile(r'^\[TSO PROX\]\[(V|PF) MAX\] node=(\d+), year=(\d+), day=(\w+), period=(\d+)'
                      r'(?:, type=(\w+))? \| (?:dV|delta)=([-\d.eE+]+) (kV|MW|MVAr) \| normalized=([-\d.eE+]+)')
TSO_PROX_HEAD = re.compile(r'^\[TSO PROX\] cycle=(\d+) \|')


def _find_addendum(sha):
    d = G._gate_dir()
    hits = [f for f in sorted(os.listdir(d)) if f.startswith(ADDENDUM_PREFIX) and f.endswith('.json')]
    match = [f for f in hits if G._sha_file(os.path.join(d, f)) == sha]
    if len(match) != 1:
        raise RuntimeError(f'addendum spec with sha256 {sha} not found exactly once: {hits}')
    path = os.path.join(d, match[0])
    obj = G._load(path)
    return path, obj


def _check_addendum(args):
    spec = G._find_spec(args.spec_sha256)
    path, add = _find_addendum(args.addendum_sha256)
    if (add.get('predecessor') or {}).get('sha256') != spec['_sha256']:
        raise RuntimeError(f'addendum predecessor {add.get("predecessor")} is not the gate spec {spec["_sha256"]}')
    return spec, {'path': G._rel(path), 'sha256': args.addendum_sha256}


def run_on(args):
    _spec, add = _check_addendum(args)
    if os.path.exists(os.path.join(G._gate_dir(), f'campaign_{GF_ROOT_R3}')):
        G._log(f'REFUSING: campaign_{GF_ROOT_R3} exists')
        return 1
    G._log(f'G-f third attempt (addendum {add["path"]}): GF_ROOT {G.GF_ROOT!r} -> {GF_ROOT_R3!r} '
           f'(frozen script otherwise unchanged)')
    G.GF_ROOT = GF_ROOT_R3
    return G.smoke(args)


def run_control(args):
    spec, add = _check_addendum(args)
    if os.path.exists(os.path.join(G._gate_dir(), f'campaign_{CONTROL_ROOT}')):
        G._log(f'REFUSING: campaign_{CONTROL_ROOT} exists')
        return 1
    cfg = deepcopy(G.CONFIGURATION_SRP1)
    cfg['name'] = ('S56 step 1 G-f CONTROL: SRP1-grid 3 x 3, per-scenario interface coupling OFF (no interface_coupling '
                   'option), row 18 alpha 0.5; otherwise the published configuration -- the G-f smoke\'s instance, '
                   'candidate, cap and premium')
    cfg['derived_instance'] = G.derived_declaration('srp1grid')
    res = G._run_campaign(spec, CONTROL_ROOT, [(G.GF_LABEL, G.X0, deepcopy(CONTROL_OPTIONS))], cfg, G.GF_CAP,
                          'G-f control (option off)')
    p = (res.get('per_evaluation') or {}).get(G.GF_LABEL) or {}
    rows = []
    if p.get('eval_dir') and os.path.isfile(G._abs(os.path.join(p['eval_dir'], 'per_cycle_record.jsonl'))):
        rows = G._read_jsonl(G._abs(os.path.join(p['eval_dir'], 'per_cycle_record.jsonl')))
    checks = {'launched': bool(res.get('launched')), 'no_harness_error': not res.get('harness_error'),
              'exit_0': p.get('exit_code') == 0, 'two_cycles': p.get('cycles_run') == G.GF_CAP,
              'per_cycle_rows_2': len(rows) == G.GF_CAP,
              'no_coupling_field_in_rows': bool(rows) and not any(
                  f in r for r in rows for f in G.PER_CYCLE_COUPLING_FIELDS + NEW_Q_FIELDS)}
    res.update({'role': 'G-f control (option off)', 'addendum_spec': add, 'checks': checks,
                'holds': all(checks.values()), 'rows': rows, 'parent_guard': G.guard_report()})
    path = os.path.join(G._gate_dir(), f'verdict_gf_control_off_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    G._write_once_json(path, res)
    G._log(f"G-f control holds {res['holds']} {checks} -> {G._rel(path)}")
    return 0 if res['holds'] else 2


def _tso_prox(stdout_path):
    """Production's per-cycle [TSO PROX] V / PF maxima (the TSO expected interface vs its proximal centre)."""
    out = {}
    if not os.path.isfile(stdout_path):
        return out
    cycle = None
    with open(stdout_path, errors='replace') as handle:
        for line in handle:
            head = TSO_PROX_HEAD.match(line)
            if head:
                cycle = int(head.group(1))
                continue
            m = TSO_PROX.match(line)
            if m and cycle is not None:
                ch, node, year, day, period, typ, val, unit, norm = m.groups()
                out.setdefault(str(cycle), {})[ch] = {'node': int(node), 'year': year, 'day': day,
                                                      'period': int(period), 'type': typ, 'movement': float(val),
                                                      'unit': unit, 'normalized': float(norm)}
    return out


def _stride_movements(path, top=10):
    """From the S38 PF stride: per cycle, z_TSO current - previous per entry (MW / MVAr); the top entries by
    |movement| / rating, the max per (node, P|Q), and the count of bound-to-bound moves (|dz| >= 1.99 rating)."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as handle:
        for text in handle:
            rec = json.loads(text)
            ents = []
            for e in rec.get('entries') or []:
                dz = e['z_tso_current'] - e['z_tso_prev']
                ents.append({'node_id': e['node_id'], 'year': e['year'], 'day': e['day'], 'type': e['power_type'],
                             'period': e['period'], 'dz': dz, 'normalized': abs(dz) / e['interface_rating'],
                             'z_prev': e['z_tso_prev'], 'z_current': e['z_tso_current'], 'x_dso': e['x_dso'],
                             'rating': e['interface_rating']})
            ents.sort(key=lambda x: (-x['normalized'], str(x['node_id']), x['year'], x['day'], x['type'],
                                     x['period']))
            per_node = {}
            for x in ents:
                k = f"{x['node_id']}|{x['type']}"
                per_node[k] = max(per_node.get(k, 0.0), abs(x['dz']))
            out[str(rec['cycle'])] = {
                'top': ents[:top], 'max_abs_dz_per_node_type': per_node,
                'n_entries': len(ents), 'n_bound_to_bound': sum(1 for x in ents if x['normalized'] >= 1.99),
                'n_normalized_ge_0p5': sum(1 for x in ents if x['normalized'] >= 0.5),
                'identity': {k: rec.get(k) for k in rec if k != 'entries'}}
    return out


def _run_summary(root_rel):
    eval_root = os.path.join(G._gate_dir(), root_rel, 'evals')
    dirs = sorted(os.listdir(eval_root)) if os.path.isdir(eval_root) else []
    if len(dirs) != 1:
        return {'error': f'expected one eval dir under {root_rel}: {dirs}'}
    ed = os.path.join(eval_root, dirs[0])
    rows = G._read_jsonl(os.path.join(ed, 'per_cycle_record.jsonl')) if os.path.isfile(
        os.path.join(ed, 'per_cycle_record.jsonl')) else []
    q = []
    for r in rows:
        entry = {k: r.get(k) for k in ('cycle', 'gross_operational_cost', 'recourse', 'terminal_salvage_value',
                                       'objective_change_abs', 'objective_tolerance', 'local_solves_ok',
                                       'gross_operational_cost_settlement_excluded_in_full') + NEW_Q_FIELDS
                 if k in r}
        entry['boyd'] = {k: r.get(k) for k in r if k.startswith('boyd_') and (
            k.endswith('_part') or k.endswith('_ratio') or k.endswith('_r') or k.endswith('_s'))}
        entry['scenario_mismatch'] = {k: r.get(k) for k in r if k.startswith('scenario_mismatch_')
                                      and k != 'scenario_mismatch_by_dn'}
        entry['new_fields_present_and_finite'] = all(
            isinstance(r.get(f), float) and math.isfinite(r[f]) for f in NEW_Q_FIELDS) if any(
            f in r for f in NEW_Q_FIELDS) else None
        if 'gross_operational_cost_settlement_excluded_in_full' in r:
            entry['primary_equals_settlement_excluded_in_full'] = (
                r.get('gross_operational_cost') == r.get('gross_operational_cost_settlement_excluded_in_full'))
            entry['net_equals_primary_minus_salvage'] = (
                r.get('recourse') == r.get('net_operational_recourse_settlement_excluded_in_full'))
        q.append(entry)
    rec_path = os.path.join(ed, 'evaluation_record.json')
    rec = G._load(rec_path) if os.path.isfile(rec_path) else {}
    rc = rec.get('recourse_components') or {}
    terminal = {k: rc.get(k) for k in ('gross_operational_cost', 'net_operational_recourse', 'terminal_salvage_value',
                                       'gross_operational_cost_contracted_excluded',
                                       'net_operational_recourse_contracted_excluded',
                                       'gross_operational_cost_settlement_excluded_in_full',
                                       'net_operational_recourse_settlement_excluded_in_full', 'q_convention',
                                       'interface_settlement_total', 'interface_settlement_contracted_total',
                                       'interface_settlement_deviation_total', 'voltage_pin_total')}
    if rc.get('q_convention') == 'settlement_excluded_in_full':
        gap = rc['gross_operational_cost_contracted_excluded'] - rc['gross_operational_cost']
        part = rc['interface_settlement_total'] - rc['interface_settlement_contracted_total']
        scale = max(abs(rc['interface_settlement_total']), abs(rc['interface_settlement_contracted_total']), 1.0)
        terminal['identity_contracted_minus_primary_equals_total_minus_contracted'] = {
            'lhs': gap, 'rhs': part, 'rel_diff': abs(gap - part) / scale, 'holds_1e-9': abs(gap - part) / scale <= 1e-9}
    ms_path = os.path.join(ed, G.H.MULTISCENARIO_TERMINAL_FILE)
    ms = (G._load(ms_path).get('summary') or {}) if os.path.isfile(ms_path) else {}
    return {'eval_dir': G._rel(ed), 'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
            'objective_convention': rec.get('objective_convention'), 'per_cycle_q_and_boyd': q,
            'terminal_recourse': terminal,
            'terminal_record_contracted_fields': {k: rec.get(k) for k in (
                'terminal_gross_operational_cost', 'terminal_gross_operational_cost_contracted_excluded',
                'terminal_net_operational_recourse', 'terminal_net_operational_recourse_contracted_excluded')},
            'multiscenario_terminal_checks': ms.get('checks'), 'multiscenario_all_checks_pass': ms.get('all_checks_pass'),
            'solve_profile': rec.get('solve_profile'),
            'tso_prox_per_cycle': _tso_prox(os.path.join(ed, 'child_stdout.log')),
            'pf_stride_movements': _stride_movements(os.path.join(ed, 'pf_entry_stride_s39_D.jsonl'))}


def compare(args):
    res = {'schema': 'p515_s56_gf_r3_vs_control_v1', 'utc': G._utc(),
           'definition': ('movement = TSO expected interface (z_TSO) at the end of cycle k minus its value before the '
                          'cycle (the S38 PF stride z_tso_current - z_tso_prev, MW / MVAr; normalized by the interface '
                          'rating); production [TSO PROX] = max |expected interface - proximal centre| per channel'),
           'on_r3': _run_summary(f'campaign_{GF_ROOT_R3}'), 'control_off': _run_summary(f'campaign_{CONTROL_ROOT}'),
           'r2_reference_tso_prox': _tso_prox(os.path.join(G._gate_dir(), R2_EVAL, 'child_stdout.log')),
           'parent_guard': G.guard_report()}
    path = os.path.join(G._gate_dir(), f'gf_r3_vs_control_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    G._write_once_json(path, res)
    for arm in ('on_r3', 'control_off'):
        tp = res[arm].get('tso_prox_per_cycle') or {}
        G._log(f"{arm}: " + '; '.join(f"cycle {c}: PF {v.get('PF', {}).get('movement')} MW "
                                      f"(norm {v.get('PF', {}).get('normalized')}, node {v.get('PF', {}).get('node')} "
                                      f"{v.get('PF', {}).get('year')} {v.get('PF', {}).get('day')} "
                                      f"t{v.get('PF', {}).get('period')}), V {v.get('V', {}).get('movement')} kV"
                                      for c, v in sorted(tp.items())))
    G._log(f'-> {G._rel(path)}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('on', 'control', 'compare'))
    ap.add_argument('--spec-sha256')
    ap.add_argument('--addendum-sha256')
    args = ap.parse_args()
    if os.environ.get('NLP_SOLVER_PATH') != G.SOLVER_PATH:
        G._log(f'REFUSING: NLP_SOLVER_PATH must be {G.SOLVER_PATH}')
        return 1
    if args.arm == 'compare':
        return compare(args)
    if not (args.spec_sha256 and args.addendum_sha256):
        ap.error('--spec-sha256 and --addendum-sha256 are required to launch')
    return run_on(args) if args.arm == 'on' else run_control(args)


if __name__ == '__main__':
    sys.exit(main())
