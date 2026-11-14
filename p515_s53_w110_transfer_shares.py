"""P5.15 W110 (Addendum 54 Ruling 1 / Addendum 55) -- the Planner's pre-registered REPORT-ONLY transfer shares on the
C* settling extension. ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Nothing is built and nothing is solved; only committed JSONL records are read.

PRE-REGISTRATION (TASKS.md, commit 4b2a3e8c; report-only, NOT in spec v41, the v41 verdict logic is unchanged): over
188-287 report (i) absolute shares |D_component| / sum |D_component| per agent x component; (ii) the transfer ratio
-D(DSO_7 flexibility) / D(TSO generation); (iii) whether ESS sum |dp| at node 7 co-moves with the transfer. Planner task
W110: compute all three for 188-287 and for 143-187 (the 143-187 figures replace W105's uncommitted pass: TSO generation
+2.008 x DQ, DSO_7 flexibility -1.052 x DQ, net DQ -11,674.5).

INSTANCE: SRP1, the corner plan C* (recert cell c_star, candidate key = the W104 / recert key), campaign
s53_w105_c_star_ext_r2 (campaign spec 7c1ee4da under stage spec v41 fcea4b38), eval key 8864266d...; cycles 1..187 are a
bitwise replay of W104 (82dcebb7), 188..287 the extension.

INPUTS (sha256 verified against the harness-written campaign_manifest_sha256.json before use):
    <eval dir>/creep_diagnostic_per_cycle.jsonl   q_decomposition.agents / agents_delta, ess_movement.per_node, gross
    <eval dir>/per_cycle_record.jsonl             gross_operational_cost, boyd_{pf,ess}_primal_ratio (extras only)

OBJECTIVE CONVENTION: Q = gross_operational_cost (settlement excluded); the ESSO does not enter gross Q (its agent row is
identically 0). Components C = generation_cost, flexibility_cost, load_curtailment_cost, res_curtailment_penalty,
ess_usage_penalty, ess_complementarity_penalties, slack_penalties, other (= p515_s53_w105_settling_extension_hooks
.Q_COMPONENTS_ALL; they sum to the block's contribution to Q). Agents A = TSO, DSO_5, DSO_7, DSO_9, ESSO.

FORMULAS. A[a, c](k) = q_decomposition.agents[a][c] at cycle k. For a window [s, e] two conventions are computed:
    'steps'     : D[a, c] = A[a, c](e) - A[a, c](s - 1)   (= the sum of the per-cycle changes over cycles s..e; the
                                                           frozen v41 scorer's DQ = Q_287 - Q_187 for 188-287)
    'endpoints' : D[a, c] = A[a, c](e) - A[a, c](s)       (the convention that reproduces W105's net -11,674.5 for
                                                           143-187: Q_187 - Q_143)
    DQ = Q(e) - Q(s - 1)  resp.  Q(e) - Q(s), Q = the creep line's gross (asserted equal to per_cycle_record's).
    (i)   abs_share[a, c]   = |D[a, c]| / sum_{a', c'} |D[a', c']|  over A x C (the 'value' / 'ess_terms' aggregates
                              excluded; ESSO rows 0); signed multiple[a, c] = D[a, c] / DQ reported beside.
    (ii)  transfer_ratio    = -D[DSO_7, flexibility_cost] / D[TSO, generation_cost].
    (iii) corr = Pearson correlation over cycles k in [s, e] of
              x_k = ess_movement.per_node['7']['p_esso']  (sum over years, days, periods of |p(k) - p(k-1)| of the
                    ESSO's own es_pnet schedule at node 7, MW; the v41 P_b primary side)
              y_k = |agents_delta['TSO']['generation_cost']|  (= |A[TSO, generation_cost](k) - A(k-1)|)
          Spearman rank correlation (average ranks for ties) reported beside, supplementary; the same pair for
          DSO_7 flexibility_cost reported beside.
    Report-only extras: Q_287 - Q_187, Q_287 - Q_87; boyd pf / ess primal ratios at 187 and 287.

Output (write-once, new directory): data/SRP1/Results/P515S53/w110_transfer_shares/
    w110_transfer_shares.json, launch.log (captured by the launcher), manifest_sha256.json (--manifest)

Launch (attached, alone, both streams captured), then the manifest:
    mkdir data/SRP1/Results/P515S53/w110_transfer_shares
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w110_transfer_shares.py \\
        > data/SRP1/Results/P515S53/w110_transfer_shares/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w110_transfer_shares.py --manifest
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W110 transfer shares zero-solve').install()

import gate_result_io as GRIO  # noqa: E402

THIS = os.path.abspath(__file__)
CAMPAIGN_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w105_c_star_extension',
                                 'campaign_s53_w105_c_star_ext_r2')
EVAL_DIR_REL = os.path.join(CAMPAIGN_ROOT_REL, 'evals', '8864266d3c064ed0_c_star_ext')
CAMPAIGN_MANIFEST_REL = os.path.join(CAMPAIGN_ROOT_REL, 'campaign_manifest_sha256.json')
CREEP_REL = os.path.join(EVAL_DIR_REL, 'creep_diagnostic_per_cycle.jsonl')
PCR_REL = os.path.join(EVAL_DIR_REL, 'per_cycle_record.jsonl')
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'w110_transfer_shares')
OUT_JSON = os.path.join(OUT_DIR, 'w110_transfer_shares.json')
INSTANCE = {'case': 'SRP1', 'plan': 'C* (recert cell c_star)', 'campaign_id': 's53_w105_c_star_ext_r2',
            'campaign_spec_sha256': '7c1ee4dacc1d334a775071861cfa9c97e21f561ddf500803c465943faaea09ad',
            'stage_spec': 'frozen_s53_spec_v41_fcea4b38.json',
            'eval_key': '8864266d3c064ed08783f6eb7c2ef078ee5d0a97793c88f55b17f11a1e1e1238'}
AGENTS = ('TSO', 'DSO_5', 'DSO_7', 'DSO_9', 'ESSO')
COMPONENTS = ('generation_cost', 'flexibility_cost', 'load_curtailment_cost', 'res_curtailment_penalty',
              'ess_usage_penalty', 'ess_complementarity_penalties', 'slack_penalties', 'other')
WINDOWS = {'188_287': (188, 287), '143_187': (143, 187)}
W105_UNCOMMITTED_143_187 = {'TSO_generation_multiple_of_DQ': 2.008, 'DSO_7_flexibility_multiple_of_DQ': -1.052,
                            'DQ': -11674.5, 'source': 'W105 uncommitted read-only pass (TASKS.md provenance note)'}
ESS_NODE = '7'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_by_cycle(rel):
    out = {}
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if 'cycle' in row:
                    if row['cycle'] in out:
                        raise RuntimeError(f'{rel}: cycle {row["cycle"]} appears twice')
                    out[row['cycle']] = row
    return out


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


def _shares(creep, q, s, e, convention):
    start = s - 1 if convention == 'steps' else s
    a0, a1 = creep[start]['q_decomposition']['agents'], creep[e]['q_decomposition']['agents']
    dq = q[e] - q[start]
    D = {a: {c: a1[a][c] - a0[a][c] for c in COMPONENTS} for a in AGENTS}
    Dv = {a: a1[a]['value'] - a0[a]['value'] for a in AGENTS}
    denom = sum(abs(D[a][c]) for a in AGENTS for c in COMPONENTS)
    abs_share = {a: {c: (abs(D[a][c]) / denom if denom else None) for c in COMPONENTS} for a in AGENTS}
    multiple = {a: {c: (D[a][c] / dq if dq else None) for c in COMPONENTS} for a in AGENTS}
    top = sorted(((a, c, abs_share[a][c]) for a in AGENTS for c in COMPONENTS), key=lambda t: -(t[2] or 0.0))[:6]
    tso_gen, dso7_flex = D['TSO']['generation_cost'], D['DSO_7']['flexibility_cost']
    return {
        'convention': convention, 'from_cycle': start, 'to_cycle': e, 'Q_from': q[start], 'Q_to': q[e], 'DQ': dq,
        'D': D, 'D_agent_value': Dv,
        'reconciliation': {'DQ_minus_sum_D_value': dq - sum(Dv.values()),
                           'DQ_minus_sum_D_components': dq - sum(D[a][c] for a in AGENTS for c in COMPONENTS)},
        'sum_abs_D': denom,
        'i_abs_share': abs_share, 'i_abs_share_top6': [{'agent': a, 'component': c, 'abs_share': v} for a, c, v in top],
        'signed_multiple_of_DQ': multiple,
        'ii_transfer_ratio': (-dso7_flex / tso_gen) if tso_gen else None,
        'ii_inputs': {'D_TSO_generation_cost': tso_gen, 'D_DSO_7_flexibility_cost': dso7_flex,
                      'TSO_generation_multiple_of_DQ': multiple['TSO']['generation_cost'],
                      'DSO_7_flexibility_multiple_of_DQ': multiple['DSO_7']['flexibility_cost']},
    }


def _comove(creep, s, e):
    ks = list(range(s, e + 1))
    x, y_tso, y_dso7 = [], [], []
    for k in ks:
        mv = creep[k]['ess_movement']
        if not mv.get('available'):
            raise RuntimeError(f'cycle {k}: ess_movement unavailable')
        x.append(float(mv['per_node'][ESS_NODE]['p_esso']))
        ad = creep[k]['q_decomposition']['agents_delta']
        y_tso.append(abs(ad['TSO']['generation_cost']))
        y_dso7.append(abs(ad['DSO_7']['flexibility_cost']))
    n = len(ks)
    half = n // 2
    return {'cycles': [s, e], 'n': n,
            'x_definition': "ess_movement.per_node['7']['p_esso'] (MW, ESSO side)",
            'y_definition': "|agents_delta['TSO']['generation_cost']| (EUR)",
            'iii_pearson': _pearson(x, y_tso), 'spearman_supplementary': _spearman(x, y_tso),
            'beside_dso7_flexibility': {'y_definition': "|agents_delta['DSO_7']['flexibility_cost']|",
                                        'pearson': _pearson(x, y_dso7), 'spearman': _spearman(x, y_dso7)},
            'beside_tso_gen_vs_dso7_flex_abs_steps_pearson': _pearson(y_tso, y_dso7),
            'x_mean': sum(x) / n, 'y_mean': sum(y_tso) / n,
            'x_mean_first_half_last_half': [sum(x[:half]) / half, sum(x[half:]) / (n - half)],
            'series': {'cycle': ks, 'x_node7_p_esso_abs_dp': x, 'y_abs_dTSO_generation_cost': y_tso,
                       'abs_dDSO_7_flexibility_cost': y_dso7}}


def run():
    t0 = time.time()
    if os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')
    manifest = json.load(open(os.path.join(REPO, CAMPAIGN_MANIFEST_REL)))
    inputs = {}
    for rel in (CREEP_REL, PCR_REL):
        now, pinned = _sha(os.path.join(REPO, rel)), manifest.get(rel)
        if now != pinned:
            raise RuntimeError(f'{rel}: sha256 {now} differs from the campaign manifest ({pinned})')
        inputs[rel] = now
    _log(f'inputs verified against {CAMPAIGN_MANIFEST_REL}: {inputs}')
    creep, pcr = _read_by_cycle(CREEP_REL), _read_by_cycle(PCR_REL)
    if sorted(creep) != list(range(1, 288)) or sorted(pcr) != list(range(1, 288)):
        raise RuntimeError(f'cycles: creep {min(creep)}..{max(creep)} n={len(creep)}; pcr n={len(pcr)} (need 1..287)')
    q = {k: creep[k]['gross'] for k in creep}
    mism = [k for k in q if q[k] != pcr[k]['gross_operational_cost']]
    if mism:
        raise RuntimeError(f'creep gross != per_cycle_record gross at cycles {mism[:10]}')
    for k in creep:
        if set(creep[k]['q_decomposition']['agents']) != set(AGENTS):
            raise RuntimeError(f'cycle {k}: agents {sorted(creep[k]["q_decomposition"]["agents"])}')
    windows = {}
    for name, (s, e) in WINDOWS.items():
        windows[name] = {'steps': _shares(creep, q, s, e, 'steps'),
                         'endpoints': _shares(creep, q, s, e, 'endpoints'),
                         'iii_comove': _comove(creep, s, e)}
        st = windows[name]['steps']
        _log(f"{name} steps: DQ {st['DQ']!r} TSO gen x{st['ii_inputs']['TSO_generation_multiple_of_DQ']:.4f} "
             f"DSO_7 flex x{st['ii_inputs']['DSO_7_flexibility_multiple_of_DQ']:.4f} ratio {st['ii_transfer_ratio']!r}; "
             f"corr {windows[name]['iii_comove']['iii_pearson']!r}")
    ep = windows['143_187']['endpoints']
    repro = {'w105_uncommitted': W105_UNCOMMITTED_143_187,
             'recomputed_endpoints': {'DQ': ep['DQ'],
                                      'TSO_generation_multiple_of_DQ': ep['ii_inputs']['TSO_generation_multiple_of_DQ'],
                                      'DSO_7_flexibility_multiple_of_DQ': ep['ii_inputs']['DSO_7_flexibility_multiple_of_DQ']},
             'reproduces_at_stated_precision': (round(ep['DQ'], 1) == W105_UNCOMMITTED_143_187['DQ']
                                                and round(ep['ii_inputs']['TSO_generation_multiple_of_DQ'], 3) == 2.008
                                                and round(ep['ii_inputs']['DSO_7_flexibility_multiple_of_DQ'], 3) == -1.052)}
    _log(f'143-187 reproduction of W105 pass: {repro}')
    extras = {'Q_287_minus_Q_187': q[287] - q[187], 'Q_287_minus_Q_87': q[287] - q[87],
              'Q_87': q[87], 'Q_187': q[187], 'Q_287': q[287],
              'primal_ratios': {str(k): {g: pcr[k][f'boyd_{g}_primal_ratio'] for g in ('v', 'pf', 'ess')}
                                for k in (187, 287)}}
    guard_failures = _GUARD.verify(0)
    result = {
        'stage': 'P5.15 W110 pre-registered report-only transfer shares (TASKS.md 4b2a3e8c; not in v41)', 'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
        'script_sha256': _sha(THIS), 'instance': INSTANCE, 'inputs_sha256': inputs,
        'objective_convention': 'Q = gross_operational_cost (settlement excluded); ESSO not in gross Q',
        'agents': list(AGENTS), 'components': list(COMPONENTS),
        'solve_profile_guard': {'permitted': [], 'verify_0': guard_failures, 'counts': dict(_GUARD.counts)},
        'windows': windows, 'w105_143_187_reproduction': repro, 'report_only_extras': extras,
        'wall_s': time.time() - t0,
    }
    with open(OUT_JSON, 'x') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    _log(f'wrote {OUT_JSON}; guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {result["wall_s"]:.1f} s')
    return 0 if not guard_failures else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    for rel in (CREEP_REL, PCR_REL):
        entries[rel + ' (input)'] = _sha(os.path.join(REPO, rel))
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else run())
