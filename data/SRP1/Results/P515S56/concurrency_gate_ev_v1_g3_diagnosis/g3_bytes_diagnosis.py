"""Zero-solve diagnosis of gate F ev_v1's G3 failure (gate_result.json f4d6905c): the only non-excluded G3 differences
are interface_dual_capture.bytes_written / bytes_per_cycle_line_max (1-2 bytes) on 3 of 5 candidates. Hypothesis: the
lines of interface_duals_per_cycle.jsonl carry a timing field `capture_s` whose decimal text length varies, so the
byte counts differ while every other value of the file is bitwise equal. Reads the A and B eval dirs only (no solve,
no unpickle). For each candidate: per line, (a) the parsed JSON with `capture_s` removed, compared bitwise (floats by
IEEE bytes via repr of float.hex); (b) byte length A - B; (c) len(repr(capture_s)) A - B. Holds iff (a) equal on every
line and (b) == (c) on every line."""
import json, os, struct
G = 'data/SRP1/Results/P515S56/concurrency_gate_ev_v1'
res = json.load(open(os.path.join(G, 'gate_result.json')))
def canon(o):
    if isinstance(o, float): return ('f', struct.pack('<d', o).hex())
    if isinstance(o, dict): return {k: canon(v) for k, v in o.items()}
    if isinstance(o, list): return [canon(v) for v in o]
    return o
out = {'hypothesis': __doc__, 'per_candidate': {}}
for lab in res['gate']['G1_G3_per_compared_label']:
    pa = os.path.join(res['roots']['A']['per_evaluation'][lab]['eval_dir'], 'interface_duals_per_cycle.jsonl')
    pb = os.path.join(res['roots']['B']['per_evaluation'][lab]['eval_dir'], 'interface_duals_per_cycle.jsonl')
    la = open(pa, 'rb').read().split(b'\n'); lb = open(pb, 'rb').read().split(b'\n')
    rows = []
    for a, b in zip(la, lb):
        if not a.strip() and not b.strip():
            continue
        ja, jb = json.loads(a), json.loads(b)
        ca, cb = ja.pop('capture_s', None), jb.pop('capture_s', None)
        rows.append({'rest_bitwise_equal': canon(ja) == canon(jb), 'bytes_A_minus_B': len(a) - len(b),
                     'capture_s_A': ca, 'capture_s_B': cb,
                     'capture_s_text_len_A_minus_B': (len(json.dumps(ca)) - len(json.dumps(cb))) if ca is not None else 0})
    rec_a = json.load(open(os.path.join(os.path.dirname(pa), 'evaluation_record.json')))['interface_dual_capture']
    rec_b = json.load(open(os.path.join(os.path.dirname(pb), 'evaluation_record.json')))['interface_dual_capture']
    out['per_candidate'][lab] = {
        'n_lines_A': len([x for x in la if x.strip()]), 'n_lines_B': len([x for x in lb if x.strip()]),
        'file_bytes_A': os.path.getsize(pa), 'file_bytes_B': os.path.getsize(pb),
        'record_bytes_written_A_B': [rec_a['bytes_written'], rec_b['bytes_written']],
        'record_line_max_A_B': [rec_a['bytes_per_cycle_line_max'], rec_b['bytes_per_cycle_line_max']],
        'lines': rows,
        'holds': all(r['rest_bitwise_equal'] and r['bytes_A_minus_B'] == r['capture_s_text_len_A_minus_B'] for r in rows)}
out['all_hold'] = all(v['holds'] for v in out['per_candidate'].values())
print(json.dumps(out, indent=1, sort_keys=True))
