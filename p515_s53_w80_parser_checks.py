"""
P5.15 W80 -- ZERO-SOLVE checks of the v27 changes to p515_s53_w78_scaling_pin_test.py (merge-aware options parser,
v27 C0 / baseline record evaluator, spec_v27_content scope assertions). Reads only committed calibration logs
(the v26 calibration runs) and synthetic text; never the decisive arm's logs (those are re-evaluated only after
spec v27 is frozen). A blocking SolveProfileGuard(permitted=()) is armed; both guards verify(0).
Synthetic option blocks are built with IPOPT 3.14.18 OptionsList::PrintList's own format
"%40s = %-20s %6d\\n" truncated to 254 characters (Snprintf(buffer, 255, ...)).
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w80_parser_checks.py \\
        > data/SRP1/Results/P515S53/scaling_pin_w78/launch_logs/w80_parser_checks.log 2>&1
"""
import sys, os, json, re, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p515_s53_w78_scaling_pin_test as T
from p513_solve_profile_guard import SolveProfileGuard
blk = SolveProfileGuard((), label='W80 parser tests -- zero solves').install()
fails = []
def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name, detail if not cond else '')
    if not cond: fails.append(name)

def ipopt_entry(name, value, counter):
    full = '%40s = %-20s %6d\n' % (name, value, counter)
    return full[:254]          # Snprintf(buffer, 255, ...) keeps at most 254 chars

def block(output_file_value, extra=()):
    opts = sorted([('acceptable_iter', '5', 1), ('bound_frac', '1e-05', 3), ('nlp_scaling_method', 'user-scaling', 1),
                   ('obj_scaling_factor', '0.001', 1), ('output_file', output_file_value, 1),
                   ('slack_bound_frac', '1e-05', 1), ('slack_bound_push', '1e-05', 1), ('tol', '1e-05', 2)] + list(extra))
    text = '\nList of options:\n\n' + '%40s   %-20s %s\n' % ('Name', 'Value', '# times used')
    text += ''.join(ipopt_entry(*o) for o in opts)
    text += '\n******************************************************************************\nThis program contains Ipopt\n'
    return text, opts

for n in (198, 203, 204, 205, 210, 211, 212, 220):
    path = '/' + 'a' * (n - 5) + '.log'
    assert len(path) == n
    text, opts = block(path)
    lines = text.split('List of options:')[1].splitlines()
    options, ev = T._parse_options_block(lines)
    names_ok = set(options) == {o[0] for o in opts}
    others_ok = all(options[o[0]] == {'value': o[1], 'times_used': o[2]} for o in opts if o[0] != 'output_file')
    of = options.get('output_file', {})
    exp_merged = 1 if n >= 204 else 0
    exp_counter = 1 if n <= 204 else None
    exp_value = path if n <= 211 else path[:254 - 43]
    check(f'path {n}: all 8 entries', names_ok, sorted(options))
    check(f'path {n}: other entries exact', others_ok, options)
    check(f'path {n}: merged lines {exp_merged}', ev['n_merged_option_physical_lines'] == exp_merged, ev)
    check(f'path {n}: output_file counter {exp_counter}', of.get('times_used') == exp_counter, of)
    check(f'path {n}: output_file value', of.get('value') == exp_value, (of.get('value', '')[-12:], exp_value[-12:]))
    # v26 parser on the same block, for the record
    v26 = {}
    for line in lines[:80]:
        m = T._OPTION_LINE.match(line)
        if m and m.group(1) != 'Name':
            v26[m.group(1)] = m.group(2).strip()
    print(f'      (v26 parser at path {n}: slack_bound_frac present {"slack_bound_frac" in v26})')

# duplicate detection
text, _ = block('/x.log', extra=[('tol', '1e-05', 2)])
options, ev = T._parse_options_block(text.split('List of options:')[1].splitlines())
check('duplicate entry recorded', ev['duplicate_option_entries'] == ['tol'], ev)

# (b) v26 vs v27 parser on the committed calibration logs (non-option fields identical, options identical w/o merges)
import glob
V26_FIELDS = T.PER_SOLVE_V26_FIELDS
n_seg = n_diff = n_optdiff = n_merged = 0
for cell, rel in T.CALIBRATION_RUNS.items():
    for f in sorted(glob.glob(os.path.join(T._abs(rel), 'optim_log_case*.log')))[:48]:
        segs = T.parse_ipopt_log(f)
        text = open(f, errors='replace').read()
        for seg_text, s in zip(text.split('List of options:')[1:], segs):
            n_seg += 1
            v26opts = {}
            for line in seg_text.splitlines()[:80]:
                if 'This program contains Ipopt' in line or line.startswith('*****'):
                    break
                m = T._OPTION_LINE.match(line)
                if m and m.group(1) != 'Name':
                    v26opts[m.group(1)] = {'value': m.group(2).strip(), 'times_used': int(m.group(3))}
            n_merged += 1 if s['option_parse']['n_merged_option_physical_lines'] else 0
            if s['option_parse']['n_merged_option_physical_lines'] == 0 and v26opts != s['options']:
                n_optdiff += 1
            ev = [t for k, t in s['scaling_events'] if k == 'effective']
            gr = [t for k, t in s['scaling_events'] if k == 'gradient']
            if ev != s['objective_scaling_factor_lines'] or gr != s['gradient_based_objective_scaling_lines']:
                n_diff += 1
            if s['x_scaling'] != (s['x_scaling_all'][0] if s['x_scaling_all'] else None):
                n_diff += 1
check(f'calibration logs: {n_seg} segments, scaling events consistent with v26 line lists', n_diff == 0, n_diff)
check(f'calibration logs: options identical to the v26 parse on unmerged blocks ({n_merged} merged segments)',
      n_optdiff == 0, n_optdiff)

# (c) spec v27 content in memory (not written): nothing-else-changes assertions
v26 = json.load(open(T._abs(T.SPEC_V26['path'])))
c = T.spec_v27_content(v26)
check('spec_v27_content builds; unchanged keys asserted', bool(c['unchanged_from_v26_asserted_at_freeze']['top_level_keys']),
      c.get('unchanged_from_v26_asserted_at_freeze'))
print('      unchanged top-level:', c['unchanged_from_v26_asserted_at_freeze']['top_level_keys'])
print('      changed/added top-level:', sorted(set(c) - set(c['unchanged_from_v26_asserted_at_freeze']['top_level_keys'])))
check('criteria other than C0 equal v26', all(c['criteria'][k] == v26['criteria'][k] for k in v26['criteria'] if k != 'C0_decisive_effective_factor'))

# (d) evaluator on synthetic records: restoration second line passes; a wrong line fails; baseline pairing
def rec(events, resto, opts, net='case33_1', attempt='primary', xs=('No x scaling provided',), cs=('No c scaling provided',), ds=('No d scaling provided',)):
    return {'options': opts, 'scaling_events': events, 'n_restoration_entries': resto, 'log': 'L', 'segment_index': 0,
            'network': net, 'attempt': attempt, 'x_scaling_all': list(xs), 'c_scaling_all': list(cs), 'd_scaling_all': list(ds),
            'x_scaling': xs[0] if xs else None, 'c_scaling': cs[0] if cs else None, 'd_scaling': ds[0] if ds else None,
            'option_parse': {'n_merged_option_physical_lines': 1, 'option_entries_without_counter': []}}
pin = {'per_network': {'DSO5': {'network_name': 'case33_1', 'options_before': {'tol': 1e-05}, 'recovery_options': {}}}}
base_opts = {'tol': {'value': '1e-05', 'times_used': 2}, 'max_iter': {'value': '500', 'times_used': 1},
             'fixed_variable_treatment': {'value': 'make_parameter', 'times_used': 1}}
pin_opts = dict(base_opts, nlp_scaling_method={'value': 'user-scaling', 'times_used': 1}, obj_scaling_factor={'value': '0.001', 'times_used': 1})
lf = {f'x{i}.log': 'h' for i in range(48)}
c0, c0b = T.evaluate_c0_c0b_v27([rec([['effective', '0.001'], ['effective', '0.001']], 1, pin_opts)], 0.001, pin, 1, lf, [])
check('pinned: two 0.001 lines (restoration) -> C0 holds', c0['holds'] and c0b['holds'], (c0['failures_first'], c0b['failures_first']))
c0, _ = T.evaluate_c0_c0b_v27([rec([['effective', '0.001'], ['effective', '1']], 1, pin_opts)], 0.001, pin, 1, lf, [])
check('pinned: a line != v -> C0 fails', not c0['holds'])
c0, _ = T.evaluate_c0_c0b_v27([rec([], 0, pin_opts)], 0.001, pin, 1, lf, [])
check('pinned: no line -> C0 fails', not c0['holds'])
c0, _ = T.evaluate_c0_c0b_v27([rec([['effective', '0.001']], 0, pin_opts, cs=('No c scaling provided', 'c scaling provided'))], 0.001, pin, 1, lf, [])
check('pinned: a c-scaling-provided line -> C0 fails', not c0['holds'])
c0, _ = T.evaluate_c0_c0b_v27([rec([['gradient', '1.000000e-03'], ['effective', '0.001'], ['effective', '1']], 1, base_opts, cs=('c scaling provided', 'No c scaling provided'))], None, pin, 1, lf, [])
check('baseline: gradient + paired effective + restoration 1 -> record holds', c0['holds'], c0['failures_first'])
c0, _ = T.evaluate_c0_c0b_v27([rec([['gradient', '2.000000e-03'], ['effective', '0.001']], 0, base_opts)], None, pin, 1, lf, [])
check('baseline: unpaired values -> record fails', not c0['holds'])
c0, _ = T.evaluate_c0_c0b_v27([rec([['effective', '1']], 0, base_opts)], None, pin, 1, lf, [])
check('baseline: no gradient line -> record fails', not c0['holds'])
c0, _ = T.evaluate_c0_c0b_v27([rec([['gradient', '1.000000e-03'], ['effective', '0.001']], 0, pin_opts)], None, pin, 1, lf, [])
check('baseline: user-scaling option present -> record fails', not c0['holds'])
c0, _ = T.evaluate_c0_c0b_v27([rec([['effective', '0.001']], 0, pin_opts)], 0.001, pin, 2, lf, [])
check('segment count != guard -> C0 fails', not c0['holds'])
print('guards:', blk.verify(0), T.GUARD.verify(0))
print('FAILURES:', fails)
sys.exit(1 if fails or blk.verify(0) or T.GUARD.verify(0) else 0)
