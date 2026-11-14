"""
P5.15 Addendum 51, Planner task W99 -- stage-1 post-hoc analyses FROM RECORDS ONLY, ZERO SOLVES.

Stage 1 = campaign s53_w98_x0_continuation_r2 (spec c2b02e21 under v38 8bc0ffa6; evidence commit 4689475e): the
certified x = 0 cell replayed bitwise through cycle 72, continued with the certification rule disabled (AA off, tail
on, rho frozen) and stopped early at K = 88.

Items (numbering as the task):
  2  G14 mechanism: what `every_line_reconciles_to_net` tests, why each line's flag reads "True" while the gate reads
     False, and whether any line's |block_sum - net| exceeds floating-point summation rounding.  From the gate / writer
     source (the code the run used, hash-checked against spec v38 code_sha256) and recourse_blocks_all.jsonl.
  3  POST-HOC damped-oscillation analysis of Q_k - Q_72 (k >= 72): extrema (parabolic vertices), three-extremum
     estimates (including the Planner's rough variant, reproduced), a damped-cosine model
         y_k = L + exp(-lam (k - 72)) (a cos(om (k - 72)) + b sin(om (k - 72)))
     fitted by variable projection (grid over (lam, om), linear least squares for (L, a, b)) then Gauss-Newton on all
     five, with the formal covariance, a leave-one-out jackknife, window variants, and the AR(2) (Prony) linear fit
     Q_k = c + a1 Q_(k-1) + a2 Q_(k-2) as an independent estimator.  R under the post-hoc limit beside the frozen
     D_x0.  POST-HOC throughout: none of this was frozen before the run; the frozen rule (D_x0 = D_measured =
     4,492.39) decided stage 2 and is NOT replaced.
  4  Per-block dQ decomposition, all 80 network blocks + SALVAGE, cycles 63-88 (Addendum 50 item (i), now complete),
     phases, concentration, oscillation attribution, and comparison with W95's truncated top-10 lead set.
  5  Scoring of every recorded prediction (v38 frozen operationalisation + the task's wording), H1/H2/H3.

Constraints: records only; standard library plus p513_solve_profile_guard (armed with permitted=() for the whole run,
verify(0) checked before write and on exit -- the guard is the only non-stdlib import and imports pyomo.opt; no
production module, no harness, no model, no pickle); JSONL streamed line by line; every input hashed at read time and
the stage-1 inputs checked against the committed campaign manifest; outputs written only to a NEW directory (refuses
to overwrite).

Run: /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s53_w99_stage1_posthoc_analysis.py \
       > data/SRP1/Results/P515S53/w99_stage1_posthoc/w99_run_stdout_stderr.log 2>&1
"""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W99 stage-1 post-hoc analyses (never solves)').install()

LOCKS = (os.path.join(REPO, '.p515_s44_campaign.lock'), os.path.join(REPO, '.p515_g_gate.lock'))
OUT_REL_DEFAULT = 'data/SRP1/Results/P515S53/w99_stage1_posthoc'
FORBIDDEN_MODULES = ('shared_resources_planning', 'network', 'network_data', 'model_construction_helpers',
                     'p515_s44_campaign_harness', 'uncoordinated_benchmark', 'p515_g_g1_g4_admm_gates',
                     'p515_s53_w90_3x3_campaign', 'p515_s53_w98_continuation_campaign',
                     'p515_s53_w98_continuation_hooks', 'energy_storage', 'helper_functions',
                     'admm_anderson_acceleration', 'numpy')

W98 = 'data/SRP1/Results/P515S53/w98_continuation'
ROOT = f'{W98}/campaign_s53_w98_x0_continuation_r2'
EVAL = f'{ROOT}/evals/25b92ae0f1f2c02e_x0_cont'
SPEC_V38 = 'data/SRP1/Results/P515S53/frozen_s53_spec_v38_8bc0ffa6.json'
SPEC_V38_SHA = '8bc0ffa6'
W95_JSON = 'data/SRP1/Results/P515S53/w95_x0_drift/w95_x0_drift_diagnostics.json'
W95_MANIFEST = 'data/SRP1/Results/P515S53/w95_x0_drift/manifest_sha256.json'
CERT_PER_CYCLE = 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl'
GATE_SRC = 'p515_s53_w98_continuation_campaign.py'
HOOK_SRC = 'p515_s53_w98_continuation_hooks.py'

N = 72
CERT_WINDOW = (63, 72)           # the certification window (Addendum 50 item (i))
FIRST_DELTA_CYCLE = 63
U = 2.0 ** -53                   # unit roundoff, binary64

INPUTS = {}


# ======================================================================================================================
#  tracked reads (W97's pattern)
# ======================================================================================================================
def _hash_whole(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _register(path, sha, size_before, note):
    size_after = os.path.getsize(path)
    INPUTS[os.path.relpath(path, REPO)] = {
        'sha256': sha, 'size_bytes': size_before, 'size_unchanged_during_read': size_after == size_before,
        'mtime_utc': datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(),
        'hashed_at_utc': datetime.now(timezone.utc).isoformat(), 'role': note}
    if size_after != size_before:
        raise RuntimeError(f'input changed size while being read: {path}')


def iter_jsonl_raw(rel, note):
    path = os.path.join(REPO, rel)
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            if raw.strip():
                yield raw
    _register(path, h.hexdigest(), size, note)


def load_json(rel, note, limit=2 * 1024 * 1024):
    path = os.path.join(REPO, rel)
    size = os.path.getsize(path)
    if size > limit:
        raise RuntimeError(f'{path} is {size} bytes; this script loads only small JSON documents whole')
    with open(path, 'rb') as handle:
        data = handle.read()
    _register(path, hashlib.sha256(data).hexdigest(), size, note)
    return json.loads(data)


def load_text(rel, note):
    path = os.path.join(REPO, rel)
    size = os.path.getsize(path)
    with open(path, 'rb') as handle:
        data = handle.read()
    _register(path, hashlib.sha256(data).hexdigest(), size, note)
    return data.decode('utf-8')


# ======================================================================================================================
#  small linear algebra (standard library)
# ======================================================================================================================
def solve(a, b):
    n = len(b)
    m = [list(map(float, row)) + [float(b[i])] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if m[p][c] == 0.0:
            raise ZeroDivisionError('singular system')
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                if f:
                    for j in range(c, n + 1):
                        m[r][j] -= f * m[c][j]
    return [m[i][n] / m[i][i] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def lstsq(x, y):
    p = len(x[0])
    ata = [[math.fsum(x[i][r] * x[i][c] for i in range(len(y))) for c in range(p)] for r in range(p)]
    aty = [math.fsum(x[i][r] * y[i] for i in range(len(y))) for r in range(p)]
    return solve(ata, aty)


# ======================================================================================================================
#  inputs
# ======================================================================================================================
def read_inputs():
    cm = load_json(f'{ROOT}/campaign_manifest_sha256.json', 'campaign manifest (committed 4689475e)')
    spec = load_json(SPEC_V38, 'frozen stage spec v38 (constants, predictions, code hashes)')
    res = load_json(f'{ROOT}/campaign_results.json', 'stage-1 campaign results (gates, frozen settling)')
    q = {}
    for raw in iter_jsonl_raw(f'{EVAL}/per_cycle_record.jsonl', 'stage-1 per-cycle Q (frozen Q source)'):
        x = json.loads(raw)
        q[x['cycle']] = x['gross_operational_cost']
    q_cert = {}
    for raw in iter_jsonl_raw(CERT_PER_CYCLE, 'certified x = 0 per-cycle record (replay reference)'):
        x = json.loads(raw)
        q_cert[x['cycle']] = x['gross_operational_cost']
    cont = [json.loads(raw) for raw in iter_jsonl_raw(f'{EVAL}/continuation_cycle_record.jsonl',
                                                      'continuation cycle record (early-stop streak)')]
    lines, raw_flags = [], []
    for raw in iter_jsonl_raw(f'{EVAL}/recourse_blocks_all.jsonl', 'all-block capture (items 2 and 4)'):
        x = json.loads(raw)
        x.pop('objective_component_blocks', None)       # not used; keeps memory flat
        raw_flags.append({'cycle': x['cycle'],
                          'raw_token_is_json_string_True': b'"reconciles_to_net": "True"' in raw,
                          'raw_token_is_json_true': b'"reconciles_to_net": true' in raw})
        lines.append(x)
    w95_manifest = load_json(W95_MANIFEST, 'W95 manifest (committed d757394e)')
    w95 = load_json(W95_JSON, 'W95 x = 0 drift diagnostics (committed d757394e): truncated top-10 lead set')
    if INPUTS[W95_JSON]['sha256'] != w95_manifest['outputs'][W95_JSON]['sha256']:
        raise RuntimeError('W95 output does not match its committed manifest')
    gate_src = load_text(GATE_SRC, 'gate source (G14 block_capture) -- hash-checked against spec v38 code_sha256')
    hook_src = load_text(HOOK_SRC, 'writer source (_capture_blocks, _write) -- hash-checked against spec v38')
    manifest_check = {}
    for rel, meta in INPUTS.items():
        if rel in cm:
            manifest_check[rel] = cm[rel] == meta['sha256']
    if not all(manifest_check.values()):
        raise RuntimeError(f'stage-1 input differs from the committed campaign manifest: {manifest_check}')
    code_check = {name: INPUTS[name]['sha256'] == spec['code_sha256'][name] for name in (GATE_SRC, HOOK_SRC)}
    return dict(cm=cm, spec=spec, res=res, q=q, q_cert=q_cert, cont=cont, lines=lines, raw_flags=raw_flags,
                w95=w95, gate_src=gate_src, hook_src=hook_src, manifest_check=manifest_check, code_check=code_check)


def _src_lines(text, needles):
    out = {}
    for i, line in enumerate(text.splitlines(), start=1):
        for needle in needles:
            if needle in line:
                out.setdefault(needle, []).append({'line': i, 'text': line.strip()})
    return out


def bkey(r):
    if r['agent'] == 'SALVAGE':
        return 'SALVAGE'
    if r['agent'] == 'TSO':
        return f"TSO|{r['year']}|{r['day']}"
    return f"DSO{r['node_id']}|{r['year']}|{r['day']}"


def bgroup(key):
    return key.split('|')[0]


# ======================================================================================================================
#  item 2 -- G14 mechanism
# ======================================================================================================================
def item2(d):
    lines, raw_flags = d['lines'], d['raw_flags']
    per_line, prev_vals = [], None
    integrity = {'contiguous_cycles_1_to_K': [x['cycle'] for x in lines] == list(range(1, len(lines) + 1)),
                 'n_blocks_81_every_line': all(x['n_blocks'] == 81 and len(x['blocks']) == 81 for x in lines),
                 'no_none_block_value': all(r['value'] is not None for x in lines for r in x['blocks']),
                 'unique_block_keys_every_line': all(len({bkey(r) for r in x['blocks']}) == 81 for x in lines),
                 'same_key_set_every_line': len({tuple(sorted(bkey(r) for r in x['blocks'])) for x in lines}) == 1,
                 'gross_equals_per_cycle_record_bitwise': all(x['gross_operational_cost'] == d['q'][x['cycle']]
                                                               for x in lines),
                 'salvage_zero_every_line': all(x['terminal_salvage_value'] == 0.0 for x in lines)}
    deltas_ok, deltas_checked = True, 0
    for x in lines:
        vals = {bkey(r): r['value'] for r in x['blocks']}
        vs = [r['value'] for r in x['blocks']]
        net = x['net_operational_recourse']
        fs = math.fsum(vs)
        tol = max(1e-4, 1e-10 * max(abs(net), 1.0))
        diff = x['block_sum'] - net
        sabs = math.fsum(abs(v) for v in vs)
        n = len(vs)
        gamma = (n - 1) * U / (1 - (n - 1) * U)
        per_line.append({
            'cycle': x['cycle'], 'flag_python_type_after_json_load': type(x['reconciles_to_net']).__name__,
            'flag_value': x['reconciles_to_net'], 'flag_is_True': x['reconciles_to_net'] is True,
            'block_sum_minus_net_recorded': x['block_sum_minus_net'],
            'block_sum_minus_net_recomputed': diff, 'writer_tolerance_eur': tol,
            'abs_diff_over_tolerance': abs(diff) / tol, 'within_writer_tolerance': abs(diff) <= tol,
            'ulp_of_net': math.ulp(net), 'abs_diff_in_ulps_of_net': abs(diff) / math.ulp(net),
            'fsum_blocks_minus_net': fs - net, 'block_sum_minus_fsum': x['block_sum'] - fs,
            'recursive_summation_bound_gamma_n_minus_1_sum_abs': gamma * sabs,
            'abs_diff_over_summation_bound': abs(diff) / (gamma * sabs)})
        if 'deltas_vs_previous_cycle' in x:
            for e in x['deltas_vs_previous_cycle']:
                k = bkey(e)
                deltas_checked += 1
                if not (prev_vals is not None and e['previous'] == prev_vals[k] and e['current'] == vals[k]
                        and e['delta'] == e['current'] - e['previous']):
                    deltas_ok = False
        prev_vals = vals
    integrity['recorded_deltas_equal_consecutive_line_values_bitwise'] = deltas_ok
    integrity['n_deltas_checked'] = deltas_checked
    worst = max(per_line, key=lambda r: abs(r['block_sum_minus_net_recomputed']))
    worst_cont = max((r for r in per_line if r['cycle'] > N), key=lambda r: abs(r['block_sum_minus_net_recomputed']))
    gate_lines = _src_lines(d['gate_src'], ("'every_line_reconciles_to_net'", "'all_blocks_every_line'",
                                            "'one_line_per_successful_cycle'"))
    hook_lines = _src_lines(d['hook_src'], ("tol = max(", "'reconciles_to_net':", "json.dumps(obj, default=str)",
                                            "total = sum(blocks.values())", "net = rc.get('net_operational_recourse')"))
    types = sorted({r['flag_python_type_after_json_load'] for r in per_line})
    return {
        'label': 'records + the source the run used (hash-checked against spec v38 code_sha256)',
        'source_hash_matches_spec_v38': d['code_check'],
        'gate_part_source': gate_lines, 'writer_source': hook_lines,
        'what_the_gate_part_tests': ("all(x.get('reconciles_to_net') is True for x in lines): an IDENTITY test on the "
                                     "parsed flag of every line; the gate applies no tolerance of its own and never "
                                     "compares block_sum with net itself"),
        'what_the_flag_encodes': ("writer: abs(block_sum - net) <= max(1e-4, 1e-10 * max(|net|, 1)) -- a TOLERANCE "
                                  "test (not exact equality); tolerance = 0.0843 EUR at Q ~ 8.43e8"),
        'n_lines': len(per_line),
        'flag_types_after_json_load': types,
        'n_flag_string_True': sum(1 for r in per_line if r['flag_value'] == 'True'),
        'n_flag_is_True': sum(1 for r in per_line if r['flag_is_True']),
        'raw_json_token': {'n_lines_string_"True"': sum(1 for r in raw_flags if r['raw_token_is_json_string_True']),
                           'n_lines_boolean_true': sum(1 for r in raw_flags if r['raw_token_is_json_true'])},
        'gate_part_as_evaluated': all(r['flag_is_True'] for r in per_line),
        'n_lines_within_writer_tolerance_recomputed': sum(1 for r in per_line if r['within_writer_tolerance']),
        'max_abs_block_sum_minus_net': {'cycle': worst['cycle'], 'eur': worst['block_sum_minus_net_recomputed'],
                                        'ulps_of_net': worst['abs_diff_in_ulps_of_net'],
                                        'over_tolerance': worst['abs_diff_over_tolerance']},
        'max_abs_block_sum_minus_net_continuation_73_K': {'cycle': worst_cont['cycle'],
                                                          'eur': worst_cont['block_sum_minus_net_recomputed'],
                                                          'ulps_of_net': worst_cont['abs_diff_in_ulps_of_net']},
        'max_abs_diff_over_tolerance': max(r['abs_diff_over_tolerance'] for r in per_line),
        'max_abs_diff_in_ulps_of_net': max(r['abs_diff_in_ulps_of_net'] for r in per_line),
        'max_abs_diff_over_recursive_summation_bound': max(r['abs_diff_over_summation_bound'] for r in per_line),
        'max_abs_fsum_minus_net': max(abs(r['fsum_blocks_minus_net']) for r in per_line),
        'integrity': integrity,
        'mechanism': (
            "The flag is stored as the JSON STRING \"True\" on every line, not the JSON boolean true. The writer "
            "serialises with json.dumps(obj, default=str); default=str is only invoked for an object json cannot "
            "encode, so the flag was not a Python bool at write time. The blocks are Python floats (production "
            "wraps each in float()), so the non-bool must come from the other operand: net_operational_recourse "
            "(gross - salvage; the salvage from pe.value sums) is inferred to be a numpy.float64, which makes both "
            "abs(total - net) and the max(...) tolerance numpy.float64 and the comparison a numpy.bool, whose str() "
            "is 'True'. The gate then tests `is True` on the string and reads False for every line. INFERRED from "
            "the records and the source; the numpy type of net was not observed in this run (no model is loaded). "
            "The pre-run self-tests could not see it: the synthetic run fed Python floats through an in-memory sink "
            "(no JSON round-trip)."),
        'block_data_missing_or_wrong': not (all(v for k, v in integrity.items() if isinstance(v, bool))
                                            and all(r['within_writer_tolerance'] for r in per_line)),
        'per_line': per_line,
    }


# ======================================================================================================================
#  item 3 -- post-hoc damped oscillation
# ======================================================================================================================
def _basis(k, lam, om, t0):
    t = k - t0
    e = math.exp(-lam * t)
    return e * math.cos(om * t), e * math.sin(om * t)


def _model(p, k, t0):
    """p = (L, a, b, lam, om) or, with a linear trend, (L, a, b, lam, om, m): y = L + m (k - t0) + oscillation."""
    big_l, a, b, lam, om = p[:5]
    c, s = _basis(k, lam, om, t0)
    return big_l + a * c + b * s + (p[5] * (k - t0) if len(p) > 5 else 0.0)


def _varpro(ks, ys, lam, om, t0, trend=False):
    x = [[1.0, *_basis(k, lam, om, t0)] + ([float(k - t0)] if trend else []) for k in ks]
    beta = lstsq(x, ys)
    rss = math.fsum((ys[i] - math.fsum(beta[j] * x[i][j] for j in range(len(beta)))) ** 2 for i in range(len(ys)))
    return rss, beta


def _rss(p, ks, ys, t0):
    return math.fsum((ys[i] - _model(p, ks[i], t0)) ** 2 for i in range(len(ks)))


def _jac(p, ks, t0):
    rows = []
    for k in ks:
        row = []
        for j in range(len(p)):
            h = 1e-6 * max(abs(p[j]), 1e-3)
            pp, pm = list(p), list(p)
            pp[j] += h
            pm[j] -= h
            row.append((_model(pp, k, t0) - _model(pm, k, t0)) / (2 * h))
        rows.append(row)
    return rows


def fit_damped(ks, ys, t0=N, init=None, trend=False):
    """Variable projection on a grid over (lam, om), then Gauss-Newton on (L, a, b, lam, om[, m])."""
    if init is None:
        best = None
        for i in range(0, 61):
            lam = i * 0.01
            for j in range(1, 121):
                om = j * 0.01
                try:
                    rss, beta = _varpro(ks, ys, lam, om, t0, trend)
                except ZeroDivisionError:
                    continue
                if best is None or rss < best[0]:
                    best = (rss, beta, lam, om)
        p = [best[1][0], best[1][1], best[1][2], best[2], best[3]] + ([best[1][3]] if trend else [])
    else:
        p = list(init)
    npar = len(p)
    rss = _rss(p, ks, ys, t0)
    for _ in range(200):
        jm = _jac(p, ks, t0)
        r = [ys[i] - _model(p, ks[i], t0) for i in range(len(ks))]
        jtj = [[math.fsum(jm[i][a] * jm[i][b] for i in range(len(ks))) for b in range(npar)] for a in range(npar)]
        jtr = [math.fsum(jm[i][a] * r[i] for i in range(len(ks))) for a in range(npar)]
        try:
            step = solve(jtj, jtr)
        except ZeroDivisionError:
            break
        t, improved = 1.0, False
        while t > 1e-6:
            q = [p[j] + t * step[j] for j in range(npar)]
            rq = _rss(q, ks, ys, t0)
            if rq < rss:
                improved = True
                break
            t *= 0.5
        if not improved:
            break
        conv = abs(rss - rq) <= 1e-12 * max(rss, 1e-30)
        p, rss = q, rq
        if conv:
            break
    n, k_par = len(ks), npar
    s2 = rss / (n - k_par) if n > k_par else float('nan')
    jm = _jac(p, ks, t0)
    jtj = [[math.fsum(jm[i][a] * jm[i][b] for i in range(n)) for b in range(npar)] for a in range(npar)]
    cov = [[s2 * v for v in row] for row in inverse(jtj)]
    big_l, a, b, lam, om = p[:5]

    def derived(q):
        return {'damping_per_half_period': math.exp(-q[3] * math.pi / q[4]), 'half_period_cycles': math.pi / q[4],
                'per_cycle_contraction': math.exp(-q[3])}
    dv = derived(p)
    se_derived = {}
    for name in dv:
        g = []
        for j in range(npar):
            h = 1e-6 * max(abs(p[j]), 1e-3)
            pp, pm = list(p), list(p)
            pp[j] += h
            pm[j] -= h
            g.append((derived(pp)[name] - derived(pm)[name]) / (2 * h))
        se_derived[name] = math.sqrt(max(0.0, math.fsum(g[i] * cov[i][j] * g[j] for i in range(npar)
                                                        for j in range(npar))))
    fitted = {str(k): _model(p, k, t0) for k in ks}
    resid = {str(k): ys[i] - _model(p, k, t0) for i, k in enumerate(ks)}
    rl = [resid[str(k)] for k in ks]
    lag1 = (math.fsum(rl[i] * rl[i + 1] for i in range(len(rl) - 1)) / math.fsum(v * v for v in rl)) if rss > 0 else None
    extra = {'m_trend_eur_per_cycle': p[5]} if npar > 5 else {}
    return {'window': [ks[0], ks[-1]], 'n_points': n, 't0': t0, 'n_params': npar,
            'params': {'L': big_l, 'a': a, 'b': b, 'lam': lam, 'omega': om, **extra},
            'se_formal': {'L': math.sqrt(cov[0][0]), 'lam': math.sqrt(cov[3][3]), 'omega': math.sqrt(cov[4][4]),
                          **({'m': math.sqrt(cov[5][5])} if npar > 5 else {}),
                          **{k: v for k, v in se_derived.items()}},
            **dv, 'amplitude_at_t0': math.hypot(a, b), 'rss': rss, 'rms_residual_eur': math.sqrt(rss / n),
            'residual_lag1_autocorrelation': lag1, 'fitted': fitted, 'residuals': resid}


def fit_ar2(y, k0, k1):
    ks = list(range(k0, k1 + 1))
    x = [[1.0, y[k - 1], y[k - 2]] for k in ks]
    c, a1, a2 = lstsq(x, [y[k] for k in ks])
    rss = math.fsum((y[k] - (c + a1 * y[k - 1] + a2 * y[k - 2])) ** 2 for k in ks)
    out = {'window_targets': [k0, k1], 'n_equations': len(ks), 'c': c, 'a1': a1, 'a2': a2,
           'L': c / (1 - a1 - a2), 'rms_one_step_residual_eur': math.sqrt(rss / len(ks))}
    disc = a1 * a1 + 4 * a2
    if disc < 0:
        rho = math.sqrt(-a2)
        om = math.acos(max(-1.0, min(1.0, a1 / (2 * rho))))
        out.update({'roots': 'complex', 'per_cycle_contraction': rho, 'omega': om, 'half_period_cycles': math.pi / om,
                    'damping_per_half_period': rho ** (math.pi / om)})
    else:
        out.update({'roots': 'real', 'root_values': [(a1 + math.sqrt(disc)) / 2, (a1 - math.sqrt(disc)) / 2]})
    return out


def three_extremum(a1, a2, a3):
    big_l = (a1 * a3 - a2 * a2) / (a1 + a3 - 2 * a2)
    return {'L': big_l, 'damping_per_half_period': -(a3 - a2) / (a2 - a1),
            'check_ratio_2': -(a3 - big_l) / (a2 - big_l), 'note': 'exact through three points (3 equations, 3 unknowns)'}


def item3(d, spec):
    q, q_cert = d['q'], d['q_cert']
    q72 = q[N]
    y = {k: q[k] - q72 for k in q}
    K = max(q)
    consts = spec['stage_2_decision_rule']['constants']
    V, R_ref, bar_sum = consts['V_eur'], consts['R_ref'], consts['bar_sum_resolution_eur']
    frozen = d['res']['settling_analysis']
    steps = {k: q[k] - q[k - 1] for k in range(2, K + 1)}
    # extrema (cycles 64..K-1): step sign change
    extrema = []
    for k in range(64, K):
        s0, s1 = steps[k], steps[k + 1]
        if s0 > 0 > s1 or s0 < 0 < s1:
            y0, y1, y2 = y[k - 1], y[k], y[k + 1]
            den = y0 - 2 * y1 + y2
            dt = 0.5 * (y0 - y2) / den if den else 0.0
            extrema.append({'cycle': k, 'kind': 'max' if s0 > 0 else 'min', 'y_discrete': y1,
                            'vertex_cycle': k + dt, 'vertex_y': y1 - 0.25 * (y0 - y2) * dt,
                            'in_aa_off_swing_63_65': k <= 65})
    ex = {e['cycle']: e for e in extrema}
    three = {
        'extrema_found_64_to_K': extrema,
        'proper_66_77_86_vertices': {'cycles': [66, 77, 86],
                                     **three_extremum(ex[66]['vertex_y'], ex[77]['vertex_y'], ex[86]['vertex_y']),
                                     'caveat': ('the 66 maximum follows the AA-off swing 63-65 (steps +13,954, '
                                                '-11,470, +8,834) and the 66-72 segment is NOT fitted by the same '
                                                'mode as 72-88 (see the window variants: rms 20-30 EUR on 72-88 vs '
                                                'hundreds with 65/67 onward)')},
        'planner_rough_variant_72_77_86_reproduced': {
            'cycles': [72, 77, 86], 'values': [y[72], y[77], y[86]],
            **three_extremum(y[72], y[77], y[86]),
            'why_it_differs': (f'cycle 72 is NOT an extremum: its step is {steps[72]:.2f} EUR, the largest descent step '
                               f'since the swing (73: {steps[73]:.2f}); treating it as a turning point forces the '
                               'first half-swing amplitude to 0 - L, inflating the damping ratio and |L|. Three points, '
                               'three unknowns: the fit passes through all three, so it cannot predict the third')},
    }
    # the damped-cosine model on the post-72 series (primary) and variants
    primary_ks = list(range(N, K + 1))
    primary = fit_damped(primary_ks, [y[k] for k in primary_ks])
    variants = {}
    for k0 in (73, 74, 75, 67, 65):
        ks = list(range(k0, K + 1))
        f = fit_damped(ks, [y[k] for k in ks])
        variants[f'{k0}_{K}'] = {kk: f[kk] for kk in ('window', 'n_points', 'params', 'se_formal',
                                                      'damping_per_half_period', 'half_period_cycles',
                                                      'per_cycle_contraction', 'rms_residual_eur',
                                                      'residual_lag1_autocorrelation')}
    # the constant-limit assumption tested: the same model plus a linear trend m (k - 72) on the primary window
    ftr = fit_damped(primary_ks, [y[k] for k in primary_ks], trend=True)
    trend_variant = {kk: ftr[kk] for kk in ('window', 'n_points', 'n_params', 'params', 'se_formal',
                                            'damping_per_half_period', 'half_period_cycles', 'rms_residual_eur')}
    trend_variant['m_over_se'] = ftr['params']['m_trend_eur_per_cycle'] / ftr['se_formal'].get('m', float('nan')) \
        if 'm' in ftr['se_formal'] else None
    trend_variant['note'] = ('L here is the intercept of a drifting centre at cycle 72, NOT a limit; reported to show '
                             'whether a slow drift under the oscillation is detectable in 17 points; excluded from the '
                             'L range')
    # jackknife on the primary window
    loo = []
    for drop in primary_ks:
        ks = [k for k in primary_ks if k != drop]
        f = fit_damped(ks, [y[k] for k in ks], init=[primary['params'][n] for n in ('L', 'a', 'b', 'lam', 'omega')])
        loo.append({'dropped': drop, 'L': f['params']['L'], 'damping_per_half_period': f['damping_per_half_period'],
                    'half_period_cycles': f['half_period_cycles']})
    nj = len(loo)
    mean_l = math.fsum(x['L'] for x in loo) / nj
    jack_se_l = math.sqrt((nj - 1) / nj * math.fsum((x['L'] - mean_l) ** 2 for x in loo))
    # predictive check: fit through cycle 84 only, predict 85..88
    ks84 = list(range(N, 85))
    f84 = fit_damped(ks84, [y[k] for k in ks84])
    pred84 = {str(k): {'predicted': _model([f84['params'][n] for n in ('L', 'a', 'b', 'lam', 'omega')], k, N),
                       'observed': y[k]} for k in range(85, K + 1)}
    # AR(2) / Prony
    ar2 = {f'{k0}_{K}': fit_ar2(y, k0, K) for k0 in (74, 75, 76, 69, 67)}
    # projection beyond K from the primary fit
    pp = [primary['params'][n] for n in ('L', 'a', 'b', 'lam', 'omega')]
    proj = {k: _model(pp, k, N) for k in range(K + 1, 201)}
    proj_steps = {k: proj[k] - (proj[k - 1] if k - 1 in proj else primary['fitted'][str(k - 1)]) for k in proj}
    # next extremum of the projection
    nxt = None
    for k in range(K + 1, 200):
        if (proj_steps[k] < 0 < proj_steps[k + 1]) or (proj_steps[k] > 0 > proj_steps[k + 1]):
            nxt = {'cycle': k, 'kind': 'min' if proj_steps[k] < 0 else 'max', 'y': proj[k]}
            break
    last_big = max([k for k in proj_steps if abs(proj_steps[k]) >= 500.0] or [None])
    obs_and_proj_steps = {**{k: steps[k] for k in range(N + 1, K + 1)}, **proj_steps}
    first_all_below = None
    for k0 in sorted(obs_and_proj_steps):
        if all(abs(obs_and_proj_steps[k]) < 500.0 for k in obs_and_proj_steps if k >= k0):
            first_all_below = k0
            break
    path_from_86 = [y[k] for k in range(86, K + 1)] + [proj[k] for k in range(K + 1, 201)]
    stop_value_range = [min(path_from_86), max(path_from_86)]
    # fitted maximum time near the stop
    om, lam = primary['params']['omega'], primary['params']['lam']
    a, b = primary['params']['a'], primary['params']['b']

    def dy(t):
        e = math.exp(-lam * t)
        return e * ((-lam * a + om * b) * math.cos(om * t) + (-lam * b - om * a) * math.sin(om * t))
    turning = []
    for i in range(0, 1601):
        t0_, t1_ = i * 0.01, (i + 1) * 0.01
        if dy(t0_) * dy(t1_) < 0:
            lo, hi = t0_, t1_
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if dy(lo) * dy(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            tt = 0.5 * (lo + hi)
            turning.append({'cycle': N + tt, 'y': _model(pp, N + tt, N)})
    big_l = primary['params']['L']
    se_l = primary['se_formal']['L']
    all_l = ([big_l] + [v['params']['L'] for v in variants.values()] + [v['L'] for v in ar2.values()]
             + [x['L'] for x in loo])
    l_min, l_max = min(all_l), max(all_l)

    def r_range(desc):
        return [(V - desc) / R_ref, V / R_ref]
    r_table = {
        'frozen_D_x0': {'D': frozen['D_x0'], 'R_range': r_range(frozen['D_x0']),
                        'status': 'FROZEN - decided stage 2 (do_not_run); NOT replaced'},
        'posthoc_primary_L': {'D': -big_l, 'R_range': r_range(-big_l)},
        'posthoc_primary_L_plus_2_formal_se': {'D': -big_l + 2 * se_l, 'R_range': r_range(-big_l + 2 * se_l)},
        'posthoc_primary_L_plus_2_jackknife_se': {'D': -big_l + 2 * jack_se_l,
                                                  'R_range': r_range(-big_l + 2 * jack_se_l)},
        'posthoc_largest_descent_any_estimator': {'D': -l_min, 'R_range': r_range(-l_min)},
        'posthoc_smallest_descent_any_estimator': {'D': -l_max, 'R_range': r_range(-l_max)},
        'V': V, 'R_ref': R_ref, 'formula': '[(V - D)/R_ref, V/R_ref], D = Q_72 - Q_limit = -L',
    }
    stage2 = {'threshold_bar_sum': bar_sum, 'frozen_decision': frozen['stage_2']['decision'],
              'posthoc_primary_abs_L': -big_l, 'posthoc_max_abs_L_any_estimator': -l_min,
              'posthoc_abs_L_plus_2_jackknife_se': -big_l + 2 * jack_se_l,
              'decision_under_posthoc_limit': 'do_not_run' if max(-l_min, -big_l + 2 * jack_se_l,
                                                                 -big_l + 2 * se_l) < bar_sum else 'run',
              'unchanged': max(-l_min, -big_l + 2 * jack_se_l, -big_l + 2 * se_l) < bar_sum}
    return {
        'LABEL': 'POST-HOC -- not frozen before the run; the frozen rule decided stage 2 and is not replaced',
        'Q_72_run': q72, 'Q_72_certified': q_cert[N], 'Q_72_bitwise_equal_to_certified': q72 == q_cert[N],
        'objective_convention': 'Q = gross_operational_cost, settlement excluded; gross = net at x = 0 (salvage 0)',
        'instance': {'eval_key': '25b92ae0f1f2c02eea9b3811f16edf210479e461e653f474bd23d451ff7bf3e3',
                     'candidate': spec['instance_and_certified_cell']['candidate'],
                     'candidate_key': spec['instance_and_certified_cell']['candidate_key']},
        'y_k_equals_Q_k_minus_Q_72': {str(k): y[k] for k in range(36, K + 1)},
        'steps': {str(k): steps[k] for k in range(36, K + 1)},
        'context_observation_pre_72': ('Q oscillated before the certifying regime too: min at 38 (-40,323), max at 47 '
                                       '(+65,598), low near 59-62, then the AA-off swing 63-65; half-periods 9 and ~13 '
                                       'cycles, under AA on (a different regime; reported, not fitted)'),
        'extrema_and_three_point_estimates': three,
        'damped_cosine_primary': primary,
        'damped_cosine_window_variants': variants,
        'damped_cosine_plus_linear_trend_primary_window': trend_variant,
        'jackknife_primary_leave_one_out': {'fits': loo, 'L_mean': mean_l, 'L_se_jackknife': jack_se_l},
        'predictive_check_fit_72_84_predict_85_88': {'fit': {kk: f84[kk] for kk in ('params', 'rms_residual_eur',
                                                                                    'damping_per_half_period',
                                                                                    'half_period_cycles')},
                                                     'prediction': pred84},
        'ar2_prony': ar2,
        'L_range_all_estimators': [l_min, l_max],
        'turning_points_of_primary_fit': turning[:6],
        'next_extremum_of_projection': nxt,
        'projection_steps_89_100': {str(k): proj_steps[k] for k in range(K + 1, 101)},
        'last_projected_cycle_with_abs_step_ge_500': last_big,
        'max_abs_projected_step_after_K': max(abs(v) for v in proj_steps.values()),
        'first_cycle_from_which_every_observed_and_projected_step_is_below_500': first_all_below,
        'range_of_y_over_cycles_86_to_200_observed_then_projected': stop_value_range,
        'remaining_descent_beyond_Q88_to_L': y[K] - big_l,
        'approx_amplitude_admitted_by_a_500_step_bound': 500.0 / (2 * math.sin(om / 2)),
        'early_stop_implication': (
            f'The frozen rule (|dQ| < 500 EUR on 3 consecutive cycles) fired at 88 on steps +327.24, -1.10, -301.29: '
            f'the three cycles straddling the fitted Q maximum at cycle {turning[1]["cycle"]:.2f} '
            f'(y = {turning[1]["y"]:.1f}), where the step of an oscillation passes through zero. So the stop '
            'coincided with a turning point, and D_measured = Q72 - Q88 is the descent to an oscillation MAXIMUM of '
            f'Q: the fitted limit lies {y[K] - big_l:.1f} EUR further down, and the projection reaches '
            f'{nxt["y"]:.1f} at cycle {nxt["cycle"]} before settling. QUALIFICATION of "steps momentarily small": '
            f'under the fit the steps do NOT grow back above 500 (largest projected |step| after 88 = '
            f'{max(abs(v) for v in proj_steps.values()):.1f} EUR); every step from cycle {first_all_below} on is '
            'below 500. The defect is that a step bound does not bound the distance to the limit: with a half-period '
            f'of {math.pi / om:.2f} cycles a 500 EUR step bound admits an oscillation amplitude of about '
            f'{500.0 / (2 * math.sin(om / 2)):.0f} EUR, and a stop anywhere from cycle 86 on would have returned a y '
            f'between {stop_value_range[0]:.1f} and {stop_value_range[1]:.1f} depending only on phase. At the first '
            'turning point (77) the rule did not fire because the amplitude was still larger (steps -686.88, '
            '+135.03, +748.53).'),
        'R_under_posthoc_limit': r_table,
        'stage_2_decision_check': stage2,
        'planner_rough_fit_as_stated_in_task': {'damping_ratio': 0.65, 'L': -7281,
                                                'predicted_third_extremum': -4205, 'observed': -4190},
    }


# ======================================================================================================================
#  item 4 -- per-block dQ
# ======================================================================================================================
def _concentration(contrib, total):
    absv = sorted((abs(v) for v in contrib.values()), reverse=True)
    s_abs = math.fsum(absv)
    s_sq = math.fsum(v * v for v in absv)
    need = {}
    for frac in (0.5, 0.8, 0.9):
        acc, n = 0.0, 0
        for v in absv:
            acc += v
            n += 1
            if acc >= frac * s_abs:
                break
        need[f'n_blocks_for_{int(frac * 100)}pct_of_sum_abs'] = n
    ranked = sorted(contrib.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    return {'sum_signed': total, 'sum_abs': s_abs, 'cancellation_abs_sum_over_sum_abs': abs(total) / s_abs if s_abs else None,
            'participation_ratio_N_eff': (s_abs * s_abs / s_sq) if s_sq else None, **need,
            'top5_signed_share_of_sum': math.fsum(v for _, v in ranked[:5]) / total if total else None,
            'top10_signed_share_of_sum': math.fsum(v for _, v in ranked[:10]) / total if total else None,
            'top5_share_of_sum_abs': math.fsum(abs(v) for _, v in ranked[:5]) / s_abs if s_abs else None}


def _groups(contrib):
    g, season, year, gs = {}, {}, {}, {}
    for k, v in contrib.items():
        g[bgroup(k)] = g.get(bgroup(k), 0.0) + v
        if k != 'SALVAGE':
            _, yr, dy = k.split('|')
            season[dy] = season.get(dy, 0.0) + v
            year[yr] = year.get(yr, 0.0) + v
            gs[f'{bgroup(k)}|{dy}'] = gs.get(f'{bgroup(k)}|{dy}', 0.0) + v
    return {'by_agent_node': g, 'by_season': season, 'by_year': year, 'by_agent_node_and_season': gs}


def item4(d, i3):
    lines = d['lines']
    q = d['q']
    by = {x['cycle']: x for x in lines}
    K = max(by)
    delta = {}
    for c in range(FIRST_DELTA_CYCLE, K + 1):
        delta[c] = {bkey(e): e['delta'] for e in by[c]['deltas_vs_previous_cycle']}
    keys = sorted(delta[FIRST_DELTA_CYCLE])
    w95_lead = d['w95']['supplementary_context_cycles_53_72']['lead_blocks']
    # validation against W95's truncated top-10 (63-72): same replay, same production function
    w95_pc = d['w95']['C_i_per_block_dQ']['per_cycle']
    n_cmp = n_eq = 0
    for c in range(CERT_WINDOW[0], CERT_WINDOW[1] + 1):
        for e in w95_pc[str(c)]['top10']:
            n_cmp += 1
            n_eq += delta[c][e['block']] == e['delta']
    per_cycle = {}
    for c in range(FIRST_DELTA_CYCLE, K + 1):
        dq = q[c] - q[c - 1]
        con = _concentration(delta[c], math.fsum(delta[c].values()))
        ranked = sorted(delta[c].items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        per_cycle[str(c)] = {'dQ': dq, 'sum_block_deltas': math.fsum(delta[c].values()),
                             'sum_minus_dQ': math.fsum(delta[c].values()) - dq, **con,
                             'groups': _groups(delta[c])['by_agent_node'],
                             'w95_lead6_signed_sum': math.fsum(delta[c][k] for k in w95_lead),
                             'top5': [[k, v] for k, v in ranked[:5]], 'all_81': delta[c]}
    phases = {'certification_window_63_72': (63, 72), 'aa_off_swing_63_65': (63, 65),
              'descent_from_66_max_to_72': (67, 72), 'descent_72_to_77_min': (73, 77),
              'descent_from_66_max_to_77_min': (67, 77), 'rebound_77_min_to_86_max': (78, 86),
              'turn_87_88': (87, 88), 'continuation_73_88': (73, 88)}
    phase_out = {}
    for name, (c0, c1) in phases.items():
        cum = {k: math.fsum(delta[c][k] for c in range(c0, c1 + 1)) for k in keys}
        dq = q[c1] - q[c0 - 1]
        ranked = sorted(cum.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        rank_of = {k: i + 1 for i, (k, _) in enumerate(ranked)}
        top10 = [k for k, _ in ranked[:10]]
        phase_out[name] = {'cycles': [c0, c1], 'dQ': dq, **_concentration(cum, math.fsum(cum.values())),
                           **_groups(cum), 'top15': [[k, v] for k, v in ranked[:15]],
                           'w95_lead6': {k: {'cum': cum[k], 'rank': rank_of[k]} for k in w95_lead},
                           'w95_lead6_signed_share_of_dQ': math.fsum(cum[k] for k in w95_lead) / dq,
                           'w95_lead6_in_top10': sorted(set(w95_lead) & set(top10)),
                           'top10_not_in_w95_lead6': [k for k in top10 if k not in w95_lead],
                           'dso7_signed_share_of_dQ': math.fsum(v for k, v in cum.items()
                                                                if k.startswith('DSO7|')) / dq}
    # oscillation attribution over 73..K: beta_b = sum_k d_bk s_k / sum_k s_k^2 (sum_b beta_b = 1)
    cs = list(range(N + 1, K + 1))
    s = {c: q[c] - q[c - 1] for c in cs}
    ss = math.fsum(v * v for v in s.values())
    beta = {k: math.fsum(delta[c][k] * s[c] for c in cs) / ss for k in keys}
    corr = {}
    for k in keys:
        xs = [delta[c][k] for c in cs]
        mx, ms = math.fsum(xs) / len(xs), math.fsum(s.values()) / len(cs)
        num = math.fsum((xs[i] - mx) * (s[c] - ms) for i, c in enumerate(cs))
        den = math.sqrt(math.fsum((v - mx) ** 2 for v in xs) * math.fsum((s[c] - ms) ** 2 for c in cs))
        corr[k] = num / den if den else None
    rb = sorted(beta.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    cum_cont = {k: math.fsum(delta[c][k] for c in cs) for k in keys}
    mono = [k for k in keys if k != 'SALVAGE' and (all(delta[c][k] > 0 for c in cs) or all(delta[c][k] < 0 for c in cs))]
    osc = {'definition': ('beta_b = sum_k d_bk s_k / sum_k s_k^2 over k = 73..K (projection of each block\'s delta '
                          'series on the dQ series; sum_b beta_b = 1 exactly up to rounding)'),
           'sum_beta': math.fsum(beta.values()), **_concentration(beta, math.fsum(beta.values())),
           **_groups(beta), 'top15': [[k, v, corr[k]] for k, v in rb[:15]],
           'n_blocks_beta_negative': sum(1 for v in beta.values() if v < 0),
           'n_blocks_corr_gt_0p8': sum(1 for v in corr.values() if v is not None and v > 0.8),
           'n_blocks_corr_lt_minus_0p8': sum(1 for v in corr.values() if v is not None and v < -0.8),
           'monotone_blocks_73_K': {
               'definition': 'blocks whose delta keeps one sign on every cycle 73..K (no reversal with Q)',
               'n': len(mono), 'keys_sorted_by_abs_cum': sorted(mono, key=lambda k: -abs(cum_cont[k])),
               'sum_cum': math.fsum(cum_cont[k] for k in mono),
               'sum_abs_cum': math.fsum(abs(cum_cont[k]) for k in mono),
               'share_of_sum_abs_cum_all_blocks': (math.fsum(abs(cum_cont[k]) for k in mono)
                                                   / math.fsum(abs(v) for v in cum_cont.values())),
               'beta_sum': math.fsum(beta[k] for k in mono),
               'w95_lead6_monotone': [k for k in w95_lead if k in mono]},
           'w95_lead6_beta': {k: beta[k] for k in w95_lead},
           'w95_lead6_beta_sum': math.fsum(beta[k] for k in w95_lead),
           'dso7_beta_sum': math.fsum(v for k, v in beta.items() if k.startswith('DSO7|')),
           'beta_all_81': beta, 'corr_all_81': corr}
    return {'label': 'records only: recourse_blocks_all.jsonl deltas_vs_previous_cycle (all 81 blocks, 80 network + '
                     'SALVAGE; SALVAGE is 0 every cycle at x = 0)',
            'w95_lead6': w95_lead,
            'validation_vs_w95_truncated_top10_63_72': {'n_compared': n_cmp, 'n_bitwise_equal': n_eq},
            'phases': phase_out, 'oscillation_attribution_73_K': osc, 'per_cycle': per_cycle}


# ======================================================================================================================
#  item 5 -- predictions
# ======================================================================================================================
def item5(d, i3):
    res = d['res']
    fr = res['settling_analysis']
    steps = {int(k): v for k, v in fr['steps'].items()}
    recomputed = {
        'replay_status': res['replay_gate']['status'],
        'k_peak_first_max_descent_73_K': min((k for k in steps if -steps[k] == max(-v for v in steps.values()))),
        'D_measured': d['q'][N] - d['q'][max(d['q'])],
        'max_abs_post_N_step': max(abs(v) for v in steps.values()),
        'certified_bar': d['spec']['instance_and_certified_cell']['bar'],
        'descent_step_ratios_73_77': {str(k): steps[k] / steps[k - 1] for k in range(74, 78)},
        'step_at_72': d['q'][72] - d['q'][71],
    }
    prim = i3['damped_cosine_primary']
    pr = fr['predictions_scored']
    rows = [
        {'prediction': 'replay bitwise through cycle 72', 'frozen_operationalisation': 'replay gate reads '
         'bitwise_through_N', 'outcome': f"{recomputed['replay_status']}; 72/72 cycles, init hex equal",
         'score_frozen': pr['replay_bitwise_through_72'], 'posthoc_note': None},
        {'prediction': 'steps peak within a few cycles of certification (scored as k_peak <= 77)',
         'frozen_operationalisation': 'k_peak <= N + 5', 'outcome': f"k_peak = {recomputed['k_peak_first_max_descent_73_K']}",
         'score_frozen': pr['peak_within_a_few_cycles'],
         'posthoc_note': (f"passes, but not as a hump: the largest descent step was at 72 itself "
                          f"({recomputed['step_at_72']:.2f}) and every post-72 step is smaller; k_peak = 73 is the first "
                          'post-N step of a decline already under way at certification')},
        {'prediction': 'then decay geometrically with ratio 0.80-0.95', 'frozen_operationalisation':
         'valid fit and 0.8 <= r <= 0.95', 'outcome': 'no valid fit (V2 all-descending False, V3 False); r None',
         'score_frozen': pr['geometric_decay_ratio_0.80_0.95'],
         'posthoc_note': (f"the descent steps shrank FASTER than geometric (ratios 74/73..77/76: "
                          f"{', '.join(f'{v:.3f}' for v in recomputed['descent_step_ratios_73_77'].values())}) and "
                          f"changed sign at 78; POST-HOC the oscillation envelope contracts by "
                          f"{prim['per_cycle_contraction']:.3f} per cycle (inside 0.80-0.95) but the form is "
                          'oscillatory, not a monotone geometric sequence')},
        {'prediction': 'D_x0 = 20-60 kEUR', 'frozen_operationalisation': '20000 <= D_x0 <= 60000',
         'outcome': f"D_x0 = {fr['D_x0']:.2f} (measured, early stop, lower bound)", 'score_frozen': pr['D_x0_20_60_k'],
         'posthoc_note': f"post-hoc limit descent |L| = {-prim['params']['L']:.0f} also outside 20-60 k"},
        {'prediction': 'stage 2 triggered', 'frozen_operationalisation': 'stage-2 decision is "run"',
         'outcome': f"{fr['stage_2']['decision']} (D_x0 {fr['D_x0']:.2f} < {fr['stage_2']['bar_sum']})",
         'score_frozen': pr['stage_2_triggered'],
         'posthoc_note': f"unchanged under the post-hoc limit: {i3['stage_2_decision_check']['decision_under_posthoc_limit']}"},
        {'prediction': 'R_settled >= 0.80', 'frozen_operationalisation': 'stage 2 not run: (V - D_x0)/R_ref >= 0.80',
         'outcome': f"R range {fr['stage_2']['R_range_if_not_run']}", 'score_frozen': pr['R_settled_ge_0.80'],
         'posthoc_note': (f"post-hoc lower end {i3['R_under_posthoc_limit']['posthoc_primary_L']['R_range'][0]:.4f} "
                          f"(largest-descent estimator "
                          f"{i3['R_under_posthoc_limit']['posthoc_largest_descent_any_estimator']['R_range'][0]:.4f}) "
                          '>= 0.80')},
    ]
    hyp = {
        'frozen_classification': fr['classification'],
        'H1_hump_then_geometric_decay': ('NO: no hump (the steps were already shrinking at 72) and no geometric decay '
                                         '(the steps change sign at 78 and 87); frozen: no valid fit'),
        'H2_near_constant_steps_ratio_about_1': 'NO: the steps fell from 3,814 to 687 in four cycles, then reversed',
        'H3_growing_steps_or_jump': (f"NO: max |step| {recomputed['max_abs_post_N_step']:.2f} < certified bar "
                                     f"{recomputed['certified_bar']:.2f}; no growth; early stop, so no late peak"),
        'observed_pattern_posthoc': (f"an underdamped oscillation that settles: damping {prim['damping_per_half_period']:.3f} "
                                     f"per half-period, half-period {prim['half_period_cycles']:.2f} cycles, limit "
                                     f"L = {prim['params']['L']:.1f} EUR vs Q72 -- fits none of H1/H2/H3 exactly; "
                                     'closest to H1 in that the objective settles, but through an oscillation'),
    }
    V = d['spec']['stage_2_decision_rule']['constants']['V_eur']
    R_ref = d['spec']['stage_2_decision_rule']['constants']['R_ref']
    rescored = {'replay_bitwise_through_72': recomputed['replay_status'] == 'bitwise_through_N',
                'peak_within_a_few_cycles': recomputed['k_peak_first_max_descent_73_K'] <= N + 5,
                'geometric_decay_ratio_0.80_0.95': bool(fr['validity']['valid']
                                                        and (fr.get('fit') or {}).get('r') is not None
                                                        and 0.8 <= fr['fit']['r'] <= 0.95),
                'D_x0_20_60_k': 20000 <= fr['D_x0'] <= 60000,
                'stage_2_triggered': fr['stage_2']['decision'] == 'run',
                'R_settled_ge_0.80': (V - fr['D_x0']) / R_ref >= 0.80}
    match = (rescored == pr and recomputed['D_measured'] == fr['D_measured']
             and recomputed['k_peak_first_max_descent_73_K'] == fr['k_peak'])
    return {'recomputed_from_records': recomputed, 'rescored_from_frozen_operationalisation': rescored,
            'frozen_scores_match_campaign_results': match,
            'predictions': rows, 'hypotheses': hyp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_REL_DEFAULT)
    args = ap.parse_args()
    out_dir = os.path.join(REPO, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    out_json = os.path.join(out_dir, 'w99_stage1_posthoc_analysis.json')
    out_manifest = os.path.join(out_dir, 'manifest_sha256.json')
    for lock in LOCKS:
        if os.path.exists(lock):
            raise RuntimeError(f'lock present ({lock}); a campaign may be running -- refusing')
    os.makedirs(out_dir, exist_ok=True)
    for path in (out_json, out_manifest):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    out = {'schema': 'p515_s53_w99_stage1_posthoc_v1',
           'task': 'W99 (PLANNER_BRIEF_2026-09-13.md Addendum 51; stage-1 evidence commit 4689475e)',
           'campaign': 's53_w98_x0_continuation_r2', 'campaign_spec_sha256_prefix': 'c2b02e21',
           'stage_spec': SPEC_V38, 'stage_spec_sha256_prefix': SPEC_V38_SHA,
           'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED; gross = net at x = 0',
           'constraint': ('records only; no production module, no pickle, no model; SolveProfileGuard(permitted=()) '
                          'armed for the whole run'),
           'started_utc': datetime.now(timezone.utc).isoformat()}
    d = read_inputs()
    out['input_check_vs_campaign_manifest'] = d['manifest_check']
    out['source_check_vs_spec_v38_code_sha256'] = d['code_check']
    out['item2_g14_mechanism'] = item2(d)
    i3 = item3(d, d['spec'])
    out['item3_POSTHOC_damped_oscillation'] = i3
    out['item4_per_block_dQ'] = item4(d, i3)
    out['item5_predictions'] = item5(d, i3)
    loaded = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULES)
    if loaded:
        raise RuntimeError(f'forbidden modules imported: {loaded}')
    out['forbidden_modules_imported'] = loaded
    out['pickle_loads_performed'] = 0
    out['guard_at_write'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    if out['guard_at_write']['verify_0_failures']:
        raise RuntimeError(f"guard verify(0) failed before write: {out['guard_at_write']}")
    out['peak_rss_bytes_ru_maxrss_self'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['peak_rss_units'] = 'bytes (macOS ru_maxrss)'
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'x') as handle:
        json.dump(out, handle, indent=1)
        handle.write('\n')
    script = os.path.abspath(__file__)
    manifest = {
        'schema': 'p515_s53_w99_manifest_v1',
        'script': {'path': os.path.relpath(script, REPO), 'sha256': _hash_whole(script)},
        'guard_module': {'path': 'p513_solve_profile_guard.py',
                         'sha256': _hash_whole(os.path.join(REPO, 'p513_solve_profile_guard.py'))},
        'interpreter': sys.executable, 'python_version': sys.version,
        'inputs': INPUTS, 'n_inputs': len(INPUTS),
        'outputs': {os.path.relpath(out_json, REPO): {'sha256': _hash_whole(out_json),
                                                      'size_bytes': os.path.getsize(out_json)}},
    }
    with open(out_manifest, 'x') as handle:
        json.dump(manifest, handle, indent=1)
        handle.write('\n')
    i2 = out['item2_g14_mechanism']
    p = i3['damped_cosine_primary']
    print(f"[W99] G14: flags {i2['flag_types_after_json_load']} n_string_True {i2['n_flag_string_True']} "
          f"n_is_True {i2['n_flag_is_True']} within_tol {i2['n_lines_within_writer_tolerance_recomputed']}/"
          f"{i2['n_lines']} max|diff| {i2['max_abs_block_sum_minus_net']}")
    print(f"[W99] POST-HOC primary fit {p['window']}: L {p['params']['L']:.2f} +- {p['se_formal']['L']:.2f} (formal), "
          f"jackknife se {i3['jackknife_primary_leave_one_out']['L_se_jackknife']:.2f}; damping/half-period "
          f"{p['damping_per_half_period']:.4f}; half-period {p['half_period_cycles']:.3f}; rms {p['rms_residual_eur']:.2f}")
    print(f"[W99] L range all estimators {i3['L_range_all_estimators']}; stage 2 {i3['stage_2_decision_check']}")
    print(f'wrote {os.path.relpath(out_json, REPO)} and {os.path.relpath(out_manifest, REPO)}; '
          f'{len(INPUTS)} inputs hashed; peak RSS {out["peak_rss_bytes_ru_maxrss_self"] / 2**20:.1f} MiB')


if __name__ == '__main__':
    try:
        main()
    finally:
        GUARD.uninstall()
        failures = GUARD.verify(0)
        print(f'[W99] guard counts {GUARD.counts} verify(0) -> {failures}', flush=True)
        if failures:
            raise SystemExit(f'guard verify(0) failed: {failures}')
