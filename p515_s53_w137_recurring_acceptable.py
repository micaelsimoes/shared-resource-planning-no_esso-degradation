"""P5.15 Addendum 59 ("Report the recurring block") and Planner task W137 item 3 -- the RECURRING-ACCEPTABLE REPORT.
ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (the imported W131 module installs its own zero-permit guard; it is verified at 0 as well). Only JSON / JSONL records
and IPOPT text logs are read.

WHAT IS REPORTED, per cell and across cells: every NON-OPTIMAL ACCEPTED SOLVE at or after the cell's k0 -- the final
accepted attempt of a block (12 TSO, 36 DSO, 3 ESSO per cycle) whose IPOPT exit is not "Optimal Solution Found." --
with its block, network, year, day, cycle, attempt chain and the IPOPT statistics its log carries (iterations, the final
scaled / unscaled dual infeasibility, constraint violation, variable bound violation, complementarity and overall NLP
error, "Total seconds in IPOPT", the options in force). k0 := the cell's FIRST RESIDUAL PASS (the first cycle with
Q not None and all_boyd_pass AND local_solves_ok; version 2's N, the cycle from which the settling rule operates); the
count before k0 is reported beside, not listed.
"Hours": the IPOPT logs (file_print_level 6) carry aggregate norms only -- no per-period (hour) attribution of the
infeasibilities; each entry records `hours_in_log: None` and the reason. The block's hours are not recoverable from the
logs.

THE CELLS (W136's 13): the 10 W118 r2 re-settle cells and the two settled references (W131's sources: the committed
network_ipopt_solve_records.jsonl -- final attempt = the last record of a (round, agent, network, year, day) chain,
every record's EXIT cross-checked against its log byte range -- and the per-solve ESSO logs), and W133's v3 cell 1
b_2a0ba8b2 (ed71177e; the same two sources in its own run directory). Readers: W131's `cycle_optimality`, unchanged.
Every committed input is sha256-checked against the campaign manifest that records it; every log read is sha256-
recorded in the log inventory.

SEEDED (TASKS.md, Addendum 59 entries; checked, not assumed): DSO7 (case33_2) 2025 Winter on cell 1 at cycles 120, 128,
141 and on pb_y2025_n9 at cycle 134; TSO (case9) 2035 Spring on pb_y2025_n5 at cycle 167.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w137_resettle_v4/recurring_acceptable/):
  w137_recurring_acceptable.json, w137_recurring_acceptable_log_inventory.json, manifest_sha256.json
Launch (attached, alone, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w137_resettle_v4/recurring_acceptable && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w137_recurring_acceptable.py \\
        > data/SRP1/Results/P515S53/w137_resettle_v4/recurring_acceptable/launch.log 2>&1
"""
import collections
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W137 recurring-Acceptable report (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w131_prefreeze_diagnostics as W131  # noqa: E402 -- W131's readers (arms its own guard)

GUARDS = (('w137_recurring_acceptable', GUARD), ('w131_imported', W131._GUARD))
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w137_resettle_v4', 'recurring_acceptable')
OUT_JSON = 'w137_recurring_acceptable.json'
OUT_INV = 'w137_recurring_acceptable_log_inventory.json'
OUT_MAN = 'manifest_sha256.json'
OPTIMAL = W131.OPTIMAL
CELLS = {**{c: os.path.join(W131.W118, sub) for c, sub in W131.CELLS_V2.items()},
         **{c: os.path.join(W131.W101, sub) for c, sub in W131.CELLS_V1.items()},
         'b_2a0ba8b2': os.path.join(S53, 'w132_resettle_v3', 'campaign_s53_w132_resettle_v3_b_2a0ba8b2', 'evals',
                                    'cf592cc94ce0d1ef_b_2a0ba8b2')}
CELL_NOTES = {'b_2a0ba8b2': 'W133 v3 cell 1 (ed71177e), run to the cap 213 under v3 alpha'}
SEEDED = (
    {'cell': 'b_2a0ba8b2', 'agent': 'DSO7', 'network': 'case33_2', 'year': '2025', 'day': 'Winter',
     'cycles': [120, 128, 141]},
    {'cell': 'pb_y2025_n9', 'agent': 'DSO7', 'network': 'case33_2', 'year': '2025', 'day': 'Winter', 'cycles': [134]},
    {'cell': 'pb_y2025_n5', 'agent': 'TSO', 'network': 'case9', 'year': '2035', 'day': 'Spring', 'cycles': [167]},
)
_NUM = r'([-+0-9.eE]+)'
_FINAL_FIELDS = ('Objective', 'Dual infeasibility', 'Constraint violation', 'Variable bound violation',
                 'Complementarity', 'Overall NLP error')
_OPTIONS = ('tol', 'acceptable_tol', 'acceptable_iter', 'compl_inf_tol', 'max_iter', 'mu_strategy', 'linear_solver',
            'warm_start_init_point', 'file_print_level')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ipopt_stats(text):
    """The statistics an IPOPT log section carries (the LAST solve in `text`). Pure."""
    out = {}
    m = re.findall(r'Number of Iterations\.+:\s*(\d+)', text)
    out['iterations'] = int(m[-1]) if m else None
    tail = text[text.rfind('Number of Iterations'):] if 'Number of Iterations' in text else ''
    for f in _FINAL_FIELDS:
        pat = re.escape(f) + r'\.*:\s*' + _NUM + r'\s+' + _NUM
        mm = re.search(pat, tail)
        key = f.lower().replace(' ', '_')
        out[f'{key}_scaled'] = float(mm.group(1)) if mm else None
        out[f'{key}_unscaled'] = float(mm.group(2)) if mm else None
    m = re.findall(r'Total seconds in IPOPT\s*=\s*' + _NUM, text) or re.findall(
        r'Total CPU secs in IPOPT[^=]*=\s*' + _NUM, text)
    out['total_seconds_in_ipopt'] = float(m[-1]) if m else None
    m = re.findall(r'EXIT: ([^\n]*)', text)
    out['exit'] = m[-1].strip() if m else None
    opts = {}
    for name in _OPTIONS:
        mm = re.findall(r'^\s*' + re.escape(name) + r'\s*=\s*(\S+)', text, flags=re.M)
        if mm:
            opts[name] = mm[-1]
    out['options_in_force'] = opts
    out['dual_infeasibility_over_tol_scaled'] = (out['dual_infeasibility_scaled'] / float(opts['tol'])
                                                 if out.get('dual_infeasibility_scaled') is not None and 'tol' in opts
                                                 else None)
    out['complementarity_unscaled_over_compl_inf_tol'] = (
        out['complementarity_unscaled'] / float(opts['compl_inf_tol'])
        if out.get('complementarity_unscaled') is not None and 'compl_inf_tol' in opts else None)
    out['hours_in_log'] = None
    out['hours_note'] = ('the IPOPT log carries aggregate norms only (no per-period attribution of the '
                         'infeasibilities); the block\'s hours are not recoverable from it')
    return out


def _network_log_text(rec):
    with open(rec['log_path'], 'rb') as handle:
        handle.seek(rec['log_bytes'][0])
        return handle.read(rec['log_bytes'][1] - rec['log_bytes'][0]).decode(errors='replace')


def cell_report(cell, eval_rel):
    names = ['per_cycle_record.jsonl', 'network_ipopt_solve_records.jsonl', 'network_failures_s39_D.jsonl',
             'esso_recovery_events_s39_D.jsonl', 'leak_classification_s39_D.jsonl']
    inputs = W131._verify(eval_rel, names)
    rows = W131._jsonl(os.path.join(eval_rel, 'per_cycle_record.jsonl'))
    n = len(rows)
    if [r['cycle'] for r in rows] != list(range(1, n + 1)):
        raise RuntimeError(f'{cell}: per_cycle_record cycles not 1..n')
    k0 = next((r['cycle'] for r in rows if r['gross_operational_cost'] is not None and r['boyd_all_pass']
               and r['local_solves_ok']), None)
    _all_opt, nonopt, meta = W131.cycle_optimality(eval_rel, n)
    if not meta['coverage_every_cycle_48_network_3_esso'] or meta['network']['log_byte_crosscheck']['n_disagree']:
        raise RuntimeError(f'{cell}: coverage / log cross-check fails: {meta["network"]["log_byte_crosscheck"]}')
    recs = W131._jsonl(os.path.join(eval_rel, 'network_ipopt_solve_records.jsonl'))
    last_rec = {}
    for r in recs:
        last_rec[(r['round'], r['agent'], r['network'], str(r['year']), r['day'])] = r
    logs_dir = os.path.join(REPO, meta['logs_dir'])
    listed = []
    for x in nonopt:
        if k0 is None or x['cycle'] < k0:
            continue
        if x['family'] == 'network':
            rec = last_rec[(x['cycle'], x['agent'], x['network'], str(x['year']), x['day'])]
            stats = ipopt_stats(_network_log_text(rec))
            block = (f"TSO|{x['year']}|{x['day']}" if x['agent'] == 'TSO'
                     else f"DSO|{int(x['agent'][3:])}|{x['year']}|{x['day']}")
            entry = {'cycle': x['cycle'], 'block': block, 'family': 'network', 'agent': x['agent'],
                     'network': x['network'], 'year': str(x['year']), 'day': x['day'],
                     'final_attempt': x['final_attempt'], 'final_exit': x['final_exit'],
                     'primary_exit': x['primary_exit'], 'attempts': x['attempts'],
                     'log': {'path': os.path.relpath(rec['log_path'], REPO), 'bytes': rec['log_bytes'],
                             'sha256': _sha(rec['log_path'])},
                     'record_mu_final': rec.get('mu_final'), 'record_mu_over_floor': rec.get('mu_over_floor'),
                     'record_compl_inf_tol_in_force': rec.get('compl_inf_tol_in_force'),
                     'record_iterations': rec.get('iterations'), 'ipopt': stats}
        else:
            suffix = {'primary': '', 'recovery': '_recovery', 'recovery_tier2': '_recovery_tier2'}[x['final_attempt']]
            p = os.path.join(logs_dir, f"optim_log_esso_node{x['node']}_cycle{x['cycle']:03d}{suffix}.txt")
            with open(p, errors='replace') as handle:
                stats = ipopt_stats(handle.read())
            entry = {'cycle': x['cycle'], 'block': f"ESSO|{x['node']}", 'family': 'esso', 'agent': 'ESSO',
                     'network': None, 'node': x['node'], 'year': None, 'day': None,
                     'final_attempt': x['final_attempt'], 'final_exit': x['final_exit'],
                     'primary_exit': x['primary_exit'],
                     'log': {'path': os.path.relpath(p, REPO), 'sha256': _sha(p)}, 'ipopt': stats}
        if stats.get('exit') is None or stats['exit'] != x['final_exit'].strip():
            raise RuntimeError(f'{cell} cycle {x["cycle"]} {entry["block"]}: log EXIT {stats.get("exit")!r} != '
                               f'record {x["final_exit"]!r}')
        listed.append(entry)
    before = [x for x in nonopt if k0 is not None and 1 <= x['cycle'] < k0]
    return {'eval_dir': eval_rel, 'note': CELL_NOTES.get(cell), 'cycles_recorded': n, 'k0_first_residual_pass': k0,
            'n_non_optimal_solves_at_or_after_k0': len(listed),
            'non_optimal_cycles_at_or_after_k0': sorted({e['cycle'] for e in listed}),
            'non_optimal_solves_at_or_after_k0': listed,
            'n_non_optimal_solves_before_k0_report_only': len(before),
            'non_optimal_cycles_before_k0_report_only': sorted({x['cycle'] for x in before}),
            'init_round_0_non_optimal_report_only': [x for x in nonopt if x['cycle'] == 0],
            'sources': {'network_log_byte_crosscheck': meta['network']['log_byte_crosscheck']['n_disagree'],
                        'coverage_every_cycle_48_network_3_esso': meta['coverage_every_cycle_48_network_3_esso'],
                        'esso_missing': meta['esso']['missing'], 'logs_dir': meta['logs_dir']},
            'inputs_sha256': inputs}


def across_cells(cells):
    by_block = collections.OrderedDict()
    for cell, rep in cells.items():
        for e in rep['non_optimal_solves_at_or_after_k0']:
            key = (e['family'], e['agent'], e['network'], e['year'], e['day'], e.get('node'))
            b = by_block.setdefault(key, {'block': e['block'], 'agent': e['agent'], 'network': e['network'],
                                          'year': e['year'], 'day': e['day'], 'node': e.get('node'),
                                          'cells': {}, 'n_solves': 0})
            b['cells'].setdefault(cell, []).append(e['cycle'])
            b['n_solves'] += 1
    out = []
    for b in by_block.values():
        b['n_cells'] = len(b['cells'])
        b['recurs_across_cells'] = b['n_cells'] > 1
        b['recurs_within_a_cell'] = any(len(v) > 1 for v in b['cells'].values())
        out.append(b)
    return sorted(out, key=lambda b: (-b['n_cells'], -b['n_solves'], b['block']))


def seeded_check(cells):
    out = []
    for s in SEEDED:
        rep = cells[s['cell']]
        got = sorted(e['cycle'] for e in rep['non_optimal_solves_at_or_after_k0']
                     if e['agent'] == s['agent'] and e['network'] == s['network'] and e['year'] == s['year']
                     and e['day'] == s['day'])
        out.append({**s, 'found_cycles': got, 'as_seeded': got == s['cycles']})
    return out


def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_JSON, OUT_INV, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    cells = {}
    for cell, eval_rel in CELLS.items():
        cells[cell] = cell_report(cell, eval_rel)
        r = cells[cell]
        _log(f'{cell}: k0 {r["k0_first_residual_pass"]}; non-Optimal at/after k0: '
             + ('; '.join(f'{e["block"]} c{e["cycle"]} {e["final_exit"]} ({e["final_attempt"]})'
                          for e in r['non_optimal_solves_at_or_after_k0']) or 'none')
             + f'; before k0 (report only): {r["non_optimal_cycles_before_k0_report_only"]}')
    table = across_cells(cells)
    seeded = seeded_check(cells)
    for b in table:
        _log(f'block {b["block"]} ({b["network"] or "ESSO"}): {b["n_solves"]} solves in {b["n_cells"]} cell(s) '
             f'{b["cells"]}')
    for s in seeded:
        _log(f'seeded {s["cell"]} {s["agent"]} {s["year"]} {s["day"]}: seeded {s["cycles"]} found {s["found_cycles"]} '
             f'-> {"AS SEEDED" if s["as_seeded"] else "DIFFERS"}')
    guards = {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}
    doc = {'schema': 'p515_s53_w137_recurring_acceptable_v1',
           'task': 'W137 item 3 (PLANNER_BRIEF_2026-09-13.md Addendum 59: report the recurring block)',
           'utc': _utc(), 'git_head': W131.subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                                         text=True).stdout.strip(),
           'script_sha256': _sha(os.path.abspath(__file__)), 'definition': __doc__,
           'cells': cells, 'across_cells_by_block': table,
           'recurring_blocks': [b for b in table if b['recurs_across_cells'] or b['recurs_within_a_cell']],
           'seeded_entries': seeded, 'all_seeded_as_seeded': all(s['as_seeded'] for s in seeded),
           'n_logs_inventoried': len(W131.INVENTORY), 'guards': guards, 'wall_s': time.time() - t0}
    jp = os.path.join(out_dir, OUT_JSON)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True)
    ip = os.path.join(out_dir, OUT_INV)
    with open(ip, 'x') as handle:
        GRIO.dump(W131.INVENTORY, handle, indent=1, sort_keys=True)
    man = {os.path.relpath(jp, REPO): _sha(jp), os.path.relpath(ip, REPO): _sha(ip)}
    for rep in cells.values():
        for rel, v in rep['inputs_sha256'].items():
            man[rel] = v['sha256']
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    _log(f'wrote {os.path.relpath(jp, REPO)} ({len(W131.INVENTORY)} logs inventoried); guards {guards}; '
         f'wall {time.time() - t0:.1f} s')
    for _n, g in reversed(GUARDS):
        g.uninstall()
    ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
