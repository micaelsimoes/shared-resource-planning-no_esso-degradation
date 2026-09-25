"""P5.15 Addendum 40 ruling 1 (tasks W71 -> W72) -- alpha-row RECOMPUTATION and the W72 rulings. ZERO SOLVES.

Promoted from the W71 scratch script `w71_recompute.py` (never committed; its per-cell quantities are reproduced here
unchanged) so that every figure the Planner reports from it has its FORMULA preserved, not only its inputs
(CLAUDE.md: "Preserve the formula, not only the inputs"). Read-only: nothing is built or solved. A
`SolveProfileGuard(permitted=())` is installed BEFORE any Pyomo / production / harness import and verified at exactly 0
at the end; the verification is written into the output. No existing file is modified; outputs are write-once.

INSTANCE: campaign s53_alpha_row_v25 (spec 70965374, frozen spec v25 407a4b33) -- the 2 x 2 pilot instance, BASELINE
flexibility price (NOT the m = 2 model variant), x = 0 (candidate 8435c718...) at alpha in {0, 0.1, 0.25, 0.5, 1.0}
and the node-7 unit (0.25 MVA / 1.0 MWh, 2025; candidate recorded per cell) at alpha = 0.5. Every per-eval input is
verified against its pair manifest, and each pair manifest against row_analysis_manifest_sha256.json (commit 9e88d7bd),
before it is read. External references are verified against pinned sha256 (see INPUT_PINS).

OBJECTIVE CONVENTION: Q = certified gross_operational_cost (settlement-excluded), UNPOLISHED; V = Q + voltage_pin_total;
C = Q - charge (charge = row-18 charge, weighted); P_used = charge / alpha for alpha > 0, P_posthoc at alpha = 0.
The hull-polished figures are REPORTED BESIDE, never substituted.

The formulas are stated operationally in FORMULAS below; the dict is also written into the output JSON.

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber; one at a time):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_recompute.py --run \
      > data/SRP1/Results/P515S53/alpha_row/recompute_w72/recompute_w72_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_recompute.py --manifest
  (--run writes OUT_DIR/alpha_row_recompute.json, refusing if it exists; --manifest writes
   OUT_DIR/recompute_w72_manifest_sha256.json over the script, the output, the launch log and every input read,
   refusing if it exists, after re-checking the output's recorded input hashes.)
"""

import ast
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W72 alpha-row recompute (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402  -- decoder only: load_response_terminal

SCRIPT_REL = os.path.basename(os.path.abspath(__file__))
ALPHA_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'alpha_row')
ROOT = os.path.join(ALPHA_ROOT, 'campaign_s53_alpha_row_v25')
OUT_DIR = os.path.join(ALPHA_ROOT, 'recompute_w72')
OUT_JSON = os.path.join(OUT_DIR, 'alpha_row_recompute.json')
OUT_LOG = os.path.join(OUT_DIR, 'recompute_w72_launch.log')
OUT_MANIFEST = os.path.join(OUT_DIR, 'recompute_w72_manifest_sha256.json')

CELLS = {'x0_a0p00': ('b123cd978794d690_x0_a0p00', 2), 'x0_a0p10': ('7516903c91153a29_x0_a0p10', 3),
         'x0_a0p25': ('62b46280a65f7744_x0_a0p25', 3), 'x0_a0p50': ('7d53b6f21b686a44_x0_a0p50', 1),
         'x0_a1p00': ('1bb2d63a07273887_x0_a1p00', 2), 'n7_4h_e1_a0p50': ('711fce9aa74d6878_n7_4h_e1_a0p50', 1)}
EVAL_FILES = ('evaluation_record.json', 'multiscenario_terminal.json', 'response_terminal.json',
              'per_cycle_record.jsonl', 'post_certification.json')
X0_UNIT_ALPHA_LABEL, UNIT_LABEL = 'x0_a0p50', 'n7_4h_e1_a0p50'

ROW_ANALYSIS_MANIFEST = (os.path.join(ALPHA_ROOT, 'row_analysis_manifest_sha256.json'),
                         '673b1ed0a94a1e144305e449088b2d4b70d557981711927d97444b580c300c25')   # commit 9e88d7bd
INPUT_PINS = {
    'pilot_manifest': (os.path.join('data', 'SRP1', 'Results', 'P515S52', 'campaign_s52_pilot_nopersist',
                                    'campaign_manifest_sha256.json'),
                       '2f06df9063bdd6ae6856f0c1ffb99437dd2fe5a551deab9019bed12174645d80'),       # commit f239ee02
    'srp1_s47_results': (os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert',
                                      'campaign_results.json'),
                         'e48dad43470829ee1d936cdab5e6433cffd688f9439e86e011b84dad74e19a3e'),     # commit 79b99b59
    'phase_a_tables': (os.path.join('data', 'SRP1', 'Results', 'P515S45', 'phase_a_tables', 'phase_a_tables.json'),
                       'a2b78541b5c50ad2a185ca3eb3d0f1b38bb388c4866c45628a984c407b4a3995'),       # commit c1b64278
    'smoke_r1_gate': (os.path.join(ALPHA_ROOT, 'campaign_s53_alpha_row_smoke', 'smoke_gate.json'),
                      '62c32adb059694b85e9b81d650e95a8a58b27151467900c9a132f887ed2f9048'),        # commit 776c2344
    'smoke_r2_manifest': (os.path.join(ALPHA_ROOT, 'campaign_s53_alpha_row_smoke_r2', 'smoke_manifest_sha256.json'),
                          '5e7212d80239397b612fca42135962825125fec6c71bfdeae3169103fd49033f'),    # commit ca22be17
    'launcher': ('p515_s53_alpha_row_campaign.py',
                 '61d99609edb998a54d84a6843b36c47060bdda2de3e5049f031f8431235db425'),
    'harness': ('p515_s44_campaign_harness.py', '5c67acc27d2e56c8783fc93705cd5de7201cf0800a969281ae8ebbb5c8a50589'),
    'curtailment_audit': ('p515_s53_curtailment_audit.py', None),   # hashed and recorded; D_TOL_MW read by ast
}
PILOT_RESULTS = os.path.join('data', 'SRP1', 'Results', 'P515S52', 'campaign_s52_pilot_nopersist',
                             'campaign_results.json')
SMOKE_R2_GATE = os.path.join(ALPHA_ROOT, 'campaign_s53_alpha_row_smoke_r2', 'smoke_gate.json')
ANALYSIS_JSON = os.path.join(ROOT, 'alpha_row_analysis.json')

SLACK_SWEEP = (1e-6, 1e-5, 1e-4, 1e-3)      # p.u.^2; 1e-6 is the recorded CAP_SLACK_TOL
DUAL_SWEEP = (0.1, 1.0)                     # raw |capability dual|, SENSITIVITY ONLY (units of the DSO objective)
SEPARATION_DECADE = 10.0                    # "cleanly separated" needs a >= 10x gap in BOTH slack and |dual|
N_NEAR = 10                                 # lowest-slack DSO interior entries listed per cell

FORMULAS = {
    'weights': 'w_b = admm_block_weight of the block; omega = scenario probability of the entry; B = baseMVA, derived per '
               'entry as B = lmp_bus_dual_raw / (lmp_bus_eur_mwh * omega) and asserted equal across entries.',
    'Q_C_V_P': 'Q = evaluation_record.certified_cost; bar = record bar; charge = multiscenario summary all_dso '
               'row18_charge_weighted; C = Q - charge; V = Q + voltage_pin_total; P_charge = charge / alpha (alpha > 0); '
               'P_posthoc = sum_b w_b sum_s omega_s sum_t pibar_t (|d_p[s,t]| + |d_q[s,t]|) over DSO blocks; '
               'P_used = P_charge if alpha > 0 else P_posthoc.',
    'rule_ten': '|gross[-1] - gross[-2]| / objective_tolerance[-1] from per_cycle_record.jsonl.',
    'dispersion': 'E|d|_w = sum_b w_b sum_{s,t} omega_s |d_p|; S2 = sum_b w_b sum_{s,t} omega_s d_p^2; market part '
                  '= sum_t sum_m om_m dbar_m(t)^2 with dbar_m the conditional mean of d_p over operation scenarios of '
                  'market m; operation part = sum_t sum_s omega_s (d_p - dbar_m(s))^2; _u = unweighted (w_b = 1).',
    'curtailment_totals': 'E_plus_mwh_w[net] = sum_b w_b sum_s omega_s sum_t V_plus[s,t]; priced_eur_w[net] = sum_b w_b '
                          'sum_s omega_s sum_t price[s,t] V_plus[s,t] (response_terminal curtailment_by_block; ALL '
                          'generator-hours, independent of the entry classification).',
    'entry_mwh_eur': 'for an entry set S: mwh_w(S) = sum_{e in S} w_b(e) omega(e) c_mw(e); eur_w(S) = sum_{e in S} '
                     'w_b(e) omega(e) c_mw(e) price_scenario_eur_mwh(e) (entries are generator-hour-scenarios with '
                     'c > TOL_MW = 1e-3 MW).',
    'primal_indicator': 'primal_pass(e) = (sg_avail_mva - sg_mva) >= 0.5 c_mw. LIMITATION (stated W72): here '
                        'sg_avail = pg_avail, so when |qg| << pg one has sg_avail - sg ~= c identically and the test '
                        'passes whatever the duals say; it separates P-displaced-by-Q at the S limit from the rest, NOT '
                        'active from inactive. It is not corroboration of "interior".',
    'dual_shares': 'kappa(e) = 2 (pg_mw / B) |sg_capability_dual_raw| / |lmp_bus_dual_raw| (share of the bus marginal '
                   'value of energy carried by the capability row through d(pg^2 + qg^2)/dpg = 2 pg); phi(e) = kappa + '
                   '|pg_zU_raw| / |lmp_bus_dual_raw| (with pg\'s upper bound). phi = 1 means stationarity in pg closes '
                   'on these two terms. Ill-conditioned when |lmp| ~ 0 (then both numerator and denominator are '
                   'barrier-sized); read with the |lmp| range.',
    'barrier_product': '|sg_capability_dual_raw| * sg_capability_slack_pu2: constant across a group when the dual is '
                       'set by barrier complementarity (dual = mu_eff / slack) rather than by an economic signal.',
    'classification_recorded': "response_terminal 'class': 'capability_bound' iff sg_capability_slack_pu2 <= "
                               'CAP_SLACK_TOL = 1e-6 p.u.^2, else interior (harness, the W53 audit rule).',
    'classification_corrected': 'W72 Planner ruling Q3 (the duals are authoritative): every entry of network TSO '
                                'recorded interior is reported capability_bound. Supporting check recorded per entry: '
                                'kappa >= 0.5 and |phi - 1| <= 1e-3. DSO entries keep their recorded class.',
    'dso_conflict': 'two tests on DSO entries recorded interior. (a) in-kind raw dual: |capability dual| >= min '
                    '|capability dual| over the SAME cell\'s DSO entries recorded capability_bound (raw duals are not '
                    'compared across networks, whose objectives differ in scale). (b) the TSO signature, i.e. the SAME '
                    'supporting check the TSO reclassification records: kappa >= 0.5 and |phi - 1| <= 1e-3; reported '
                    'also with c_mw <= 2 TOL_MW (TOL_MW = TOL_FACTOR x B from response_terminal constants: at '
                    'availability within the barrier offset, as the TSO entries are). The DSO entries meeting (b) are '
                    'NOT reclassified (the ruling covers TSO); their count and energy are reported with the DSO '
                    'interior figures that would result if they were.',
    'dso_separation': 'dual_gap = min_capb|dual| / max_int|dual|; slack_gap = min_int slack / max_capb slack; '
                      'zU_overlap = max_int|zU| >= min_capb|zU|. "cleanly separated" iff dual_gap >= 10 AND slack_gap >= '
                      '10 AND not zU_overlap; otherwise "contiguous (merely smaller)".',
    'dso_sensitivity': 'DSO interior n and mwh_w when the slack threshold is 1e-6 (recorded), 1e-5, 1e-4, 1e-3 (entries '
                       'with slack <= threshold counted capability_bound), and when entries with |capability dual| >= t '
                       '(t = 0.1, 1.0 raw) are moved to capability_bound. Sensitivity only.',
    'row18_condition': 'DSO entry, alpha > 0, d_p_mw <= D_TOL_MW (1e-3, p515_s53_curtailment_audit) and '
                       'price_scenario_eur_mwh < alpha * row18_premium_eur_mwh (the launcher\'s definition). overlap = '
                       'condition AND class interior.',
    'q_leg_share': 'row18_charge_q_weighted / (row18_charge_p_weighted + row18_charge_q_weighted).',
    'envelope_equivalence': 'with C = Q - alpha P and dP = P(a2) - P(a1) < 0: (a2-a1) P(a2) <= dQ <= (a2-a1) P(a1) '
                            '<=> a1 <= -dC/dP <= a2 (algebraic identity; dQ = dC + a2 P(a2) - a1 P(a1)). The ratio table '
                            'is NOT independent evidence; its informative part is position = (-dC/dP - a1)/(a2 - a1).',
    'value': 'value = Q(x0, 0.5) - Q(unit, 0.5); resolution = bar(x0, 0.5) + bar(unit, 0.5).',
    'polished_value': 'value_polished = Qpol(x0, 0.5) - Qpol(unit, 0.5), Qpol = post_certification hull_polish_full '
                      'gate reported_not_gated gross_operational_cost_after (settlement-excluded); REPORTED, never '
                      'substituted; compared with the unpolished value against the same two-cell resolution.',
    'value_change_vs_pilot': 'dvalue = value(v25) - value(pilot s52); each value is a difference of two certified cells, '
                             'so its error is the FOUR-cell bar sum resolution(v25) + resolution(pilot) (W72 ruling '
                             'Q2). |dvalue| <= that sum -> INDETERMINATE. sigma_Q (A4, phase_a_tables T3 '
                             'residual_max_abs_eur) is the MADS threshold for accepting an improvement BETWEEN TWO CELLS '
                             'and is shown beside it, NOT used as this comparison\'s error.',
    'ratio': 'R = value / SRP1_value (s47 recert n7_4h_e1 value_eur); ratio_resolution = hypot(resolution / SRP1_value, '
             'value * SRP1_resolution / SRP1_value^2) (the frozen launcher formula; SRP1_resolution = s47 '
             'bar_sum_with_x0). R - R_pred with |.| <= ratio_resolution -> INDETERMINATE.',
    'tso_reclass_scope': 'response_terminal.json files are NOT modified: the corrected class exists only in this '
                         'output.',
}


def _sha(path):
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    with open(os.path.join(REPO, path)) as f:
        return json.load(f)


def _ast_constant(path, name):
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f'{name} not a module-level literal in {path}')


def verify_inputs():
    """Verify every input before it is read. Returns {path: sha256} of everything read."""
    read, problems = {}, []

    def check(path, expect):
        got = _sha(path)
        read[path] = got
        if expect is not None and got != expect:
            problems.append(f'{path}: sha256 {got} != pinned {expect}')

    check(*ROW_ANALYSIS_MANIFEST)
    ram = _load(ROW_ANALYSIS_MANIFEST[0])
    check(ANALYSIS_JSON, ram[ANALYSIS_JSON])
    for n in (1, 2, 3):
        check(os.path.join(ROOT, f'pair_{n}_manifest_sha256.json'), ram[os.path.join(ROOT, f'pair_{n}_manifest_sha256.json')])
    pair_manifests = {n: _load(os.path.join(ROOT, f'pair_{n}_manifest_sha256.json')) for n in (1, 2, 3)}
    for label, (ed, pair) in CELLS.items():
        for fn in EVAL_FILES:
            p = os.path.join(ROOT, 'evals', ed, fn)
            if p not in pair_manifests[pair]:
                problems.append(f'{p} not in pair_{pair} manifest')
                continue
            check(p, pair_manifests[pair][p])
    for key, (path, expect) in INPUT_PINS.items():
        check(path, expect)
    pm = _load(INPUT_PINS['pilot_manifest'][0])
    check(PILOT_RESULTS, pm.get(PILOT_RESULTS))
    sm = _load(INPUT_PINS['smoke_r2_manifest'][0])
    check(SMOKE_R2_GATE, sm.get(SMOKE_R2_GATE))
    return read, problems


def _mm(vals):
    vals = [v for v in vals if v is not None]
    return [min(vals), max(vals)] if vals else None


def group_stats(entries, wb, base):
    out = {'n': len(entries)}
    if not entries:
        return out
    kap, phi, lam = [], [], []
    for e in entries:
        la = e.get('lmp_bus_dual_raw')
        if la:
            k = 2.0 * (e['pg_mw'] / base) * abs(e['sg_capability_dual_raw'] or 0.0) / abs(la)
            kap.append(k)
            phi.append(k + abs(e['pg_zU_raw'] or 0.0) / abs(la))
            lam.append(abs(la))
    out.update({
        'slack_pu2': _mm([e['sg_capability_slack_pu2'] for e in entries]),
        'abs_cap_dual_raw': _mm([abs(e['sg_capability_dual_raw'] or 0.0) for e in entries]),
        'abs_pg_zU_raw': _mm([abs(e['pg_zU_raw'] or 0.0) for e in entries]),
        'barrier_product': _mm([abs((e['sg_capability_dual_raw'] or 0.0) * (e['sg_capability_slack_pu2'] or 0.0))
                                for e in entries]),
        'c_mw': _mm([e['c_mw'] for e in entries]),
        'abs_lmp_bus_dual_raw': _mm(lam),
        'kappa': [min(kap), statistics.median(kap), max(kap)] if kap else None,
        'phi': [min(phi), statistics.median(phi), max(phi)] if phi else None,
        'mwh_w': sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in entries),
        'eur_w': sum(wb[e['block']] * e['omega'] * e['c_mw'] * e['price_scenario_eur_mwh'] for e in entries),
        'n_primal_pass': sum(int((e['sg_avail_mva'] - e['sg_mva']) >= 0.5 * e['c_mw']) for e in entries),
    })
    return out


def cell_quantities(label, ed, d_tol):
    E = os.path.join(ROOT, 'evals', ed)
    rec = _load(os.path.join(E, 'evaluation_record.json'))
    ms = _load(os.path.join(E, 'multiscenario_terminal.json'))
    rt = H.load_response_terminal(os.path.join(E, 'response_terminal.json'))
    rows = [json.loads(line) for line in open(os.path.join(E, 'per_cycle_record.jsonl')) if line.strip()]
    pc = _load(os.path.join(E, 'post_certification.json'))
    alpha = float(rec['interface_deviation_premium']['alpha'])
    Q = rec['certified_cost']
    bar = rec['bar']['value']
    charge = ms['summary']['all_dso']['row18_charge_weighted']
    vp = ms['summary']['recourse_components']['voltage_pin_total']
    # ---- P |d| form, dispersion, market / operation split (W71, unchanged) ----
    Ppost = Eabs_w = Eabs_u = S2_w = S2_u = mk_w = mk_u = op_w = op_u = peak = 0.0
    worst = None
    n_dso = 0
    for key, b in ms['blocks'].items():
        if b['kind'] != 'DSO':
            continue
        n_dso += 1
        d = b['dispersion']
        w = b['admm_block_weight']
        pr, pib, psd = d['probabilities'], d['pibar_by_hour'], d['per_scenario_d']
        nh = len(pib)
        Ppost += w * sum(pr[s] * pib[t] * (abs(psd[s]['d_p_mw'][t]) + abs(psd[s]['d_q_mvar'][t]))
                         for s in psd for t in range(nh))
        ea = sum(pr[s] * abs(psd[s]['d_p_mw'][t]) for s in psd for t in range(nh))
        s2 = sum(pr[s] * psd[s]['d_p_mw'][t] ** 2 for s in psd for t in range(nh))
        Eabs_w += w * ea
        Eabs_u += ea
        S2_w += w * s2
        S2_u += s2
        peak = max(peak, max(abs(x) for s in psd for x in psd[s]['d_p_mw']))
        om = {}
        for s in psd:
            om[s.split('_')[0]] = om.get(s.split('_')[0], 0.0) + pr[s]
        mk = op = 0.0
        for t in range(nh):
            dbar = {m: sum(pr[s] * psd[s]['d_p_mw'][t] for s in psd if s.split('_')[0] == m) / om[m] for m in om}
            mk += sum(om[m] * dbar[m] ** 2 for m in om)
            op += sum(pr[s] * (psd[s]['d_p_mw'][t] - dbar[s.split('_')[0]]) ** 2 for s in psd)
        mk_w += w * mk
        mk_u += mk
        op_w += w * op
        op_u += op
        rms = math.sqrt(s2 / nh)
        share = rms / d['p']['mean_abs_committed_flow_mw']
        if worst is None or share > worst['share']:
            worst = {'block': key, 'rms_mw': rms, 'share': share,
                     'mean_abs_committed_flow_mw': d['p']['mean_abs_committed_flow_mw']}
    # ---- curtailment totals per network (all generator-hours; classification-free) ----
    curt = {}
    for key, blk in rt['curtailment_by_block'].items():
        net = 'TSO' if blk['kind'] == 'TSO' else f"DSO{blk['node_id']}"
        w = blk['admm_block_weight']
        c = curt.setdefault(net, {'E_plus_mwh_w': 0.0, 'priced_eur_w': 0.0})
        for s, sc in blk['scenarios'].items():
            c['E_plus_mwh_w'] += w * sc['omega'] * sum(sc['V_plus_mw'])
            c['priced_eur_w'] += w * sc['omega'] * sum(p * v for p, v in zip(sc['price_eur_mwh'], sc['V_plus_mw']))
    dso_E_plus = sum(v['E_plus_mwh_w'] for k, v in curt.items() if k != 'TSO')
    wb = {k: v['admm_block_weight'] for k, v in rt['curtailment_by_block'].items()}
    entries = rt['curtailment_entries']
    # ---- baseMVA derived per entry ----
    bases = [e['lmp_bus_dual_raw'] / (e['lmp_bus_eur_mwh'] * e['omega']) for e in entries
             if e.get('lmp_bus_dual_raw') and e.get('lmp_bus_eur_mwh')]
    base = statistics.median(bases)
    base_ok = all(abs(b / base - 1.0) <= 1e-9 for b in bases)
    tol_mw = rt['constants']['TOL_FACTOR'] * base
    # ---- classification: recorded and corrected ----
    for e in entries:
        e['_kind'] = 'TSO' if e['network'] == 'TSO' else 'DSO'
        e['_class_corrected'] = ('capability_bound' if (e['_kind'] == 'TSO' and e['class'] == 'interior')
                                 else e['class'])
    classes_rec, classes_cor = {}, {}
    for e in entries:
        classes_rec[f"{e['network']}|{e['class']}"] = classes_rec.get(f"{e['network']}|{e['class']}", 0) + 1
        k2 = f"{e['network']}|{e['_class_corrected']}"
        classes_cor[k2] = classes_cor.get(k2, 0) + 1
    groups = {}
    for kind in ('TSO', 'DSO'):
        for cls in ('capability_bound', 'interior'):
            sel = [e for e in entries if e['_kind'] == kind and e['class'] == cls]
            groups[f'{kind}|{cls}|recorded'] = group_stats(sel, wb, base)
    interior_by_net = {}
    for key_cls in ('class', '_class_corrected'):
        tag = 'recorded' if key_cls == 'class' else 'corrected'
        for e in entries:
            if e[key_cls] != 'interior':
                continue
            it = interior_by_net.setdefault(tag, {}).setdefault(e['network'], {'n': 0, 'mwh_w': 0.0, 'eur_w': 0.0,
                                                                             'n_primal_pass': 0})
            it['n'] += 1
            it['mwh_w'] += wb[e['block']] * e['omega'] * e['c_mw']
            it['eur_w'] += wb[e['block']] * e['omega'] * e['c_mw'] * e['price_scenario_eur_mwh']
            it['n_primal_pass'] += int((e['sg_avail_mva'] - e['sg_mva']) >= 0.5 * e['c_mw'])
    # ---- TSO reclassification evidence (W72 Q3) ----
    tso_re = []
    for e in entries:
        if e['_kind'] == 'TSO' and e['class'] == 'interior':
            la = e['lmp_bus_dual_raw']
            kap = 2.0 * (e['pg_mw'] / base) * abs(e['sg_capability_dual_raw']) / abs(la)
            ph = kap + abs(e['pg_zU_raw']) / abs(la)
            tso_re.append({'block': e['block'], 'scenario': e['scenario'], 'hour': e['hour'], 'gen': e['gen'],
                           'omega': e['omega'], 'c_mw': e['c_mw'], 'pg_mw': e['pg_mw'], 'qg_mvar': e['qg_mvar'],
                           'sg_avail_mva': e['sg_avail_mva'], 'sg_mva': e['sg_mva'],
                           'slack_pu2': e['sg_capability_slack_pu2'],
                           'sg_capability_dual_raw': e['sg_capability_dual_raw'], 'pg_zU_raw': e['pg_zU_raw'],
                           'lmp_bus_dual_raw': la, 'kappa': kap, 'phi': ph,
                           'barrier_product': abs(e['sg_capability_dual_raw'] * e['sg_capability_slack_pu2']),
                           'primal_pass': (e['sg_avail_mva'] - e['sg_mva']) >= 0.5 * e['c_mw'],
                           'class_recorded': e['class'], 'class_corrected': e['_class_corrected'],
                           'supporting_check': kap >= 0.5 and abs(ph - 1.0) <= 1e-3})
    # ---- DSO conflict / separation / sensitivity ----
    d_int = [e for e in entries if e['_kind'] == 'DSO' and e['class'] == 'interior']
    d_cap = [e for e in entries if e['_kind'] == 'DSO' and e['class'] == 'capability_bound']
    sep = None
    if d_int and d_cap:
        min_capb_dual = min(abs(e['sg_capability_dual_raw'] or 0.0) for e in d_cap)
        max_int_dual = max(abs(e['sg_capability_dual_raw'] or 0.0) for e in d_int)
        min_int_slack = min(e['sg_capability_slack_pu2'] for e in d_int)
        max_capb_slack = max(e['sg_capability_slack_pu2'] for e in d_cap)
        max_int_zu = max(abs(e['pg_zU_raw'] or 0.0) for e in d_int)
        min_capb_zu = min(abs(e['pg_zU_raw'] or 0.0) for e in d_cap)
        n_conf = sum(1 for e in d_int if abs(e['sg_capability_dual_raw'] or 0.0) >= min_capb_dual)
        dual_gap = min_capb_dual / max_int_dual
        slack_gap = min_int_slack / max_capb_slack
        zu_overlap = max_int_zu >= min_capb_zu
        clean = dual_gap >= SEPARATION_DECADE and slack_gap >= SEPARATION_DECADE and not zu_overlap
        near = sorted(d_int, key=lambda e: e['sg_capability_slack_pu2'])[:N_NEAR]
        sig, sig_c = [], []
        for e in d_int:
            la = e.get('lmp_bus_dual_raw')
            if not la:
                continue
            kap = 2.0 * (e['pg_mw'] / base) * abs(e['sg_capability_dual_raw'] or 0.0) / abs(la)
            ph = kap + abs(e['pg_zU_raw'] or 0.0) / abs(la)
            if kap >= 0.5 and abs(ph - 1.0) <= 1e-3:
                sig.append(e)
                if e['c_mw'] <= 2.0 * tol_mw:
                    sig_c.append(e)
        sig_ids = {id(e) for e in sig}
        rest = [e for e in d_int if id(e) not in sig_ids]
        sep = {'n_conflict': n_conf,
               'tso_signature': {'n': len(sig), 'mwh_w': sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in sig),
                                 'c_max_mw': max((e['c_mw'] for e in sig), default=None),
                                 'n_with_c_le_2tol': len(sig_c),
                                 'mwh_w_with_c_le_2tol': sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in sig_c),
                                 'dso_interior_if_moved': {'n': len(rest), 'mwh_w': sum(
                                     wb[e['block']] * e['omega'] * e['c_mw'] for e in rest)}},
               'min_capb_abs_dual': min_capb_dual, 'max_int_abs_dual': max_int_dual,
               'dual_gap': dual_gap, 'min_int_slack': min_int_slack, 'max_capb_slack': max_capb_slack,
               'slack_gap': slack_gap, 'max_int_abs_zU': max_int_zu, 'min_capb_abs_zU': min_capb_zu,
               'zU_overlap': zu_overlap,
               'verdict': 'cleanly separated' if clean else 'contiguous (merely smaller)',
               'lowest_slack_interior': [
                   {'network': e['network'], 'block': e['block'], 'scenario': e['scenario'], 'hour': e['hour'],
                    'gen': e['gen'], 'slack_pu2': e['sg_capability_slack_pu2'],
                    'sg_capability_dual_raw': e['sg_capability_dual_raw'], 'pg_zU_raw': e['pg_zU_raw'],
                    'lmp_bus_dual_raw': e['lmp_bus_dual_raw'], 'c_mw': e['c_mw'],
                    'kappa': (2.0 * (e['pg_mw'] / base) * abs(e['sg_capability_dual_raw'] or 0.0)
                              / abs(e['lmp_bus_dual_raw'])) if e.get('lmp_bus_dual_raw') else None}
                   for e in near]}
    d_all = [e for e in entries if e['_kind'] == 'DSO']
    sens = {'slack_threshold': [], 'dual_cut': []}
    for thr in SLACK_SWEEP:
        sel = [e for e in d_all if e['sg_capability_slack_pu2'] > thr]
        sens['slack_threshold'].append({'threshold_pu2': thr, 'n_interior': len(sel),
                                        'mwh_w_interior': sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in sel)})
    for t in DUAL_SWEEP:
        sel = [e for e in d_int if abs(e['sg_capability_dual_raw'] or 0.0) < t]
        sens['dual_cut'].append({'abs_dual_cut_raw': t, 'n_interior': len(sel),
                                 'mwh_w_interior': sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in sel)})
    # ---- row-18 condition and its overlap with interior ----
    cond, cond_int = [], []
    for e in entries:
        if e['_kind'] != 'TSO' and alpha > 0 and e.get('d_p_mw') is not None and e.get('row18_premium_eur_mwh'):
            if e['d_p_mw'] <= d_tol and e['price_scenario_eur_mwh'] < alpha * e['row18_premium_eur_mwh']:
                cond.append(e)
                if e['_class_corrected'] == 'interior':
                    cond_int.append(e)

    def _set(sel):
        return {'n': len(sel), 'sum_c_mw_unweighted': sum(e['c_mw'] for e in sel),
                'mwh_w': sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in sel)}
    dso_int_cor = [e for e in d_all if e['_class_corrected'] == 'interior']
    cond_ids = {id(e) for e in cond}
    s = rt['summary']
    qp, qq = s['row18_charge_p_weighted'], s['row18_charge_q_weighted']
    step = abs(rows[-1]['gross_operational_cost'] - rows[-2]['gross_operational_cost'])
    tol = rows[-1]['objective_tolerance']
    g = pc['hull_polish_full']['gate']
    return {
        'alpha': alpha, 'status': rec['status'], 'cycles': rec['cycles_run'], 'candidate_key': rec['candidate_key'],
        'eval_key': rec['eval_key'], 'eval_dir': E, 'Q': Q, 'bar': bar, 'charge': charge, 'C': Q - charge, 'VP': vp,
        'V': Q + vp, 'P_charge': charge / alpha if alpha > 0 else None, 'P_posthoc': Ppost,
        'P_used': charge / alpha if alpha > 0 else Ppost,
        'rule10': step / tol, 'rule10_step': step, 'rule10_tol': tol,
        'rule10_record': rec.get('rule_ten', {}).get('terminal_step_over_threshold'),
        'E_abs_d_w': Eabs_w, 'E_abs_d_u': Eabs_u, 'S2_w': S2_w, 'S2_u': S2_u, 'peak_abs_d': peak, 'worst_share': worst,
        'market_share_u': mk_u / S2_u, 'market_share_w': mk_w / S2_w, 'market_u': mk_u, 'op_u': op_u,
        'market_w': mk_w, 'op_w': op_w, 'n_dso_blocks': n_dso,
        'baseMVA_derived': base, 'baseMVA_consistent': base_ok, 'TOL_MW': tol_mw,
        'curtailment_totals': curt, 'dso_E_plus_mwh_w': dso_E_plus,
        'classes_recorded': classes_rec, 'classes_corrected': classes_cor,
        'group_stats_recorded_class': groups,
        'interior_by_network': interior_by_net,
        'dso_interior_corrected': _set(dso_int_cor),
        'tso_reclassified_entries': tso_re,
        'dso_separation': sep, 'dso_sensitivity': sens,
        'row18_condition': _set(cond), 'row18_condition_and_interior': _set(cond_int),
        'interior_not_condition': _set([e for e in dso_int_cor if id(e) not in cond_ids]),
        'q_leg': {'p': qp, 'q': qq, 'share_q_over_p_plus_q': qq / (qp + qq) if (qp + qq) else None,
                  'recorded': s['row18_q_leg_share_of_charge'], 'legs_minus_charge': (qp + qq) - charge},
        'hull': {'delta_sum_blocks': g['delta_sum_blocks'], 'sign_le_0': g['delta_sign_as_expected_le_0'],
                 'rel_pct': g['relative_pct'], 'gross_after': g['reported_not_gated']['gross_operational_cost_after'],
                 'gross_change': g['reported_not_gated']['gross_operational_cost_change_settlement_excluded'],
                 'gross_after_minus_Q_minus_change': (g['reported_not_gated']['gross_operational_cost_after'] - Q
                                                      - g['reported_not_gated'][
                                                          'gross_operational_cost_change_settlement_excluded']),
                 'retries': pc['hull_polish_full']['solve_profile']['retries_beyond_one_per_block'],
                 'full_gate_pass': g['pass'], 'full_gate_pass_type': type(g['pass']).__name__,
                 'gate_d_pass': pc['gate_d_hull_polish']['pass'],
                 'gate_d_pass_type': type(pc['gate_d_hull_polish']['pass']).__name__,
                 'blocks_solved': pc['gate_d_hull_polish']['blocks_solved']},
    }


def row_quantities(cells, pilot, s47, sigma_q, launcher_consts, smoke_r1, smoke_r2):
    xs = sorted((c for k, c in cells.items() if k.startswith('x0_')), key=lambda c: c['alpha'])
    pairs = []
    for c1, c2 in zip(xs, xs[1:]):
        a1, a2 = c1['alpha'], c2['alpha']
        dq, dv, dc = c2['Q'] - c1['Q'], c2['V'] - c1['V'], c2['C'] - c1['C']
        dp = c2['P_used'] - c1['P_used']
        lb, ub = (a2 - a1) * c2['P_used'], (a2 - a1) * c1['P_used']
        ratio = -dc / dp
        pairs.append({'alpha1': a1, 'alpha2': a2, 'resolution': c1['bar'] + c2['bar'], 'dQ': dq, 'dV': dv,
                      'lb': lb, 'ub': ub, 'within_Q': lb <= dq <= ub, 'within_V': lb <= dv <= ub, 'dC': dc, 'dP': dp,
                      'minus_dC_over_dP': ratio, 'ratio_within_alpha_interval': a1 <= ratio <= a2,
                      'equivalence_consistent': (lb <= dq <= ub) == (a1 <= ratio <= a2),
                      'position_in_interval': (ratio - a1) / (a2 - a1)})
    x05, unit = cells[X0_UNIT_ALPHA_LABEL], cells[UNIT_LABEL]
    value = x05['Q'] - unit['Q']
    res = x05['bar'] + unit['bar']
    srp1_v, srp1_res = s47['value_eur'], s47['bar_sum_with_x0']
    R = value / srp1_v
    r_res = math.hypot(res / srp1_v, value * srp1_res / srp1_v ** 2)
    pv = pilot['value']
    four = res + pv['resolution_eur']
    dvalue = value - pv['value_eur']
    r_pred = pv['r_prediction']
    vpol = x05['hull']['gross_after'] - unit['hull']['gross_after']
    return {
        'envelope_pairs': pairs,
        'envelope_equivalence_note': ('With C = Q - alpha P the Q-form envelope (a2-a1) P(a2) <= dQ <= (a2-a1) P(a1) is '
                                      'ALGEBRAICALLY EQUIVALENT to a1 <= -dC/dP <= a2 (dP < 0). The -dC/dP table is '
                                      'therefore NOT independent evidence of the envelope; its informative part is the '
                                      'position of each ratio within its interval. (The V form adds the voltage-pin '
                                      'change and is not covered by this identity.)'),
        'unit': {
            'value_eur': value, 'resolution_two_cell': res, 'value_over_resolution': value / res,
            'value_determinate': abs(value) > res,
            'sigma_Q_A4': sigma_q, 'value_over_sigma_Q': value / sigma_q,
            'polished_value_eur': vpol, 'polished_minus_unpolished': vpol - value,
            'polished_minus_unpolished_over_resolution': (vpol - value) / res,
            'polished_vs_unpolished_verdict': ('INDETERMINATE' if abs(vpol - value) <= res else 'DETERMINATE'),
            'polished_note': 'reported beside the certified (unpolished) value, never substituted',
            'pilot_value_committed': pv['value_eur'], 'pilot_resolution_two_cell': pv['resolution_eur'],
            'pilot_Q0': pv['Q0'], 'pilot_Q_unit': pv['Q_unit'],
            'value_change_vs_pilot': dvalue, 'four_cell_bar_sum': four,
            'value_change_over_four_cell_bar_sum': dvalue / four,
            'value_change_verdict': 'INDETERMINATE' if abs(dvalue) <= four else 'DETERMINATE',
            'value_change_over_sigma_Q_not_the_governing_error': dvalue / sigma_q,
            'value_change_ruling': ('W72 ruling Q2: the pilot-to-v25 value change compares two value estimates, each a '
                                    'difference of two certified cells, so the governing error is the four-cell bar '
                                    'sum. At |dvalue| / four-cell sum <= 1 the change is INDETERMINATE: the '
                                    'initialisation fix did NOT determinately change the storage\'s value. sigma_Q '
                                    '(A4) is the MADS threshold for accepting an improvement between two cells, not '
                                    'the error of a comparison of two value estimates.'),
            'srp1_value': srp1_v, 'srp1_resolution': srp1_res,
            'R': R, 'ratio_resolution': r_res,
            'R_prediction': r_pred, 'R_minus_prediction': R - r_pred,
            'R_minus_prediction_over_ratio_resolution': (R - r_pred) / r_res,
            'R_vs_prediction_verdict': 'INDETERMINATE' if abs(R - r_pred) <= r_res else 'DETERMINATE',
            'recorded_band': {'W-R10': 'R within 0.01 of 0.942', 'band_half_width': 0.01,
                              'R_minus_0942': R - 0.942, 'falsified': abs(R - 0.942) > 0.01,
                              'ratio_resolution_over_band_half_width': r_res / 0.01},
            'R_ruling': ('W72 ruling Q2 applied to R: the recorded +/-0.01 band is FALSIFIED (R - 0.942 = '
                         f'{R - 0.942:.4f}) AND the difference from the pre-registered 0.937 is INDETERMINATE '
                         f'({(R - r_pred) / r_res:.2f} x the ratio resolution {r_res:.4f}); the band was '
                         f'{r_res / 0.01:.1f} x tighter than the measurement can resolve. Neither statement stands '
                         'alone.'),
            'pilot_R_exact': pv['ratio_pilot_over_srp1'], 'pilot_ratio_resolution': pv['ratio_resolution'],
            'R_change_vs_pilot': R - pv['ratio_pilot_over_srp1'],
            'R_change_resolution_shared_denominator': four / srp1_v,
            'R_change_over_its_resolution': (R - pv['ratio_pilot_over_srp1']) / (four / srp1_v),
            'W-R10_value_band': {'prediction': 'unit value within 10,000 of 244,321', 'miss': dvalue,
                                 'falsified': abs(dvalue) > 10000.0},
        },
        'q_leg': {
            'certified': {k: c['q_leg']['share_q_over_p_plus_q'] for k, c in cells.items()},
            'max_abs_certified_share': max(abs(c['q_leg']['share_q_over_p_plus_q']) for c in cells.values()
                                           if c['q_leg']['share_q_over_p_plus_q'] is not None),
            'smoke_r1_2_cycles': smoke_r1, 'smoke_r2_2_cycles': smoke_r2,
            'W-R9': {'prediction': 'Q-leg share of the charge at alpha = 0.5 certified: 2-25 %',
                     'observed_x0_a0p50': cells[X0_UNIT_ALPHA_LABEL]['q_leg']['share_q_over_p_plus_q'],
                     'falsified': not (0.02 <= cells[X0_UNIT_ALPHA_LABEL]['q_leg']['share_q_over_p_plus_q'] <= 0.25)},
            'note': ('The certified Q-leg share is zero to solver tolerance in every alpha > 0 cell (|share| <= a few '
                     'e-6); the smoke\'s 51.3 % at 2 cycles does not survive certification. W-R9 FALSIFIED.'),
        },
        'launcher_constants': {
            'PILOT_VALUE_EUR': launcher_consts['PILOT_VALUE_EUR'],
            'PILOT_VALUE_EUR_minus_committed': launcher_consts['PILOT_VALUE_EUR'] - pv['value_eur'],
            'PILOT_R_RATIO': launcher_consts['PILOT_R_RATIO'],
            'PILOT_R_RATIO_minus_committed_exact': launcher_consts['PILOT_R_RATIO'] - pv['ratio_pilot_over_srp1'],
            'note': ('Recorded, NOT patched: the launcher hash 61d99609 is pinned to the completed campaign. '
                     'PILOT_VALUE_EUR differs from the committed pilot value by ~4.3e-5 EUR (immaterial); '
                     'PILOT_R_RATIO is the rounded 0.942 of the exact committed ratio.'),
        },
    }


def classification_summary(cells):
    """W72 ruling Q3 across the row: the TSO reclassification, the slack threshold's inadequacy, the DSO check."""
    tso = [e for c in cells.values() for e in c['tso_reclassified_entries']]
    per_cell = {k: {'n_tso_reclassified': len(c['tso_reclassified_entries']),
                    'dso_interior_recorded': {'n': c['dso_interior_corrected']['n'],
                                              'mwh_w': c['dso_interior_corrected']['mwh_w']},
                    'dso_E_plus_mwh_w_all_hours': c['dso_E_plus_mwh_w'],
                    'dso_conflict_in_kind_dual': (c['dso_separation'] or {}).get('n_conflict'),
                    'dso_tso_signature': (c['dso_separation'] or {}).get('tso_signature'),
                    'dso_separation_verdict': (c['dso_separation'] or {}).get('verdict')}
                for k, c in cells.items()}
    tol = next(iter(cells.values()))['TOL_MW']
    rng = lambda key: [min(e[key] for e in tso), max(e[key] for e in tso)] if tso else None  # noqa: E731
    return {
        'tso_reclassified_total': len(tso),
        'tso_all_supporting_checks_pass': all(e['supporting_check'] for e in tso),
        'tso_ranges': {k: rng(k) for k in ('slack_pu2', 'sg_capability_dual_raw', 'pg_zU_raw', 'lmp_bus_dual_raw',
                                           'kappa', 'phi', 'barrier_product', 'c_mw', 'qg_mvar', 'sg_avail_mva')},
        'tso_primal_pass': sum(int(e['primal_pass']) for e in tso),
        'per_cell': per_cell,
        'statement': (
            f'{len(tso)} TSO entries ({len(tso) // max(1, sum(1 for c in cells.values() if c["tso_reclassified_entries"]))}'
            ' per x = 0 cell, none in the unit cell) were recorded interior because their capability-row slack '
            f'({rng("slack_pu2")[0]:.3g} to {rng("slack_pu2")[1]:.3g} p.u.^2) exceeds CAP_SLACK_TOL = 1e-6; their '
            f'duals contradict the label: capability dual {rng("sg_capability_dual_raw")[0]:.4g} to '
            f'{rng("sg_capability_dual_raw")[1]:.4g}, pg zU {rng("pg_zU_raw")[0]:.4g} to {rng("pg_zU_raw")[1]:.4g} '
            f'(raw), and the capability row plus pg\'s upper bound carry the whole bus marginal value (phi '
            f'{rng("phi")[0]:.6f} to {rng("phi")[1]:.6f}; kappa {rng("kappa")[0]:.3f} to {rng("kappa")[1]:.3f}). '
            'They are reported capability_bound (W72 ruling Q3). Primally they are at availability within the barrier '
            f'offset: c = {rng("c_mw")[0]:.3g} to {rng("c_mw")[1]:.3g} MW ({rng("c_mw")[1] / tol:.2f} x TOL_MW), '
            f'|qg| <= {max(abs(v) for v in rng("qg_mvar")):.2g} MVAr, so sg_avail^2 - sg^2 ~= 2 sg_avail c: the '
            'slack of the squared row grows with unit size and a fixed 1e-6 p.u.^2 threshold misclassifies large units '
            'at availability. The primal indicator passes on all of them and so does not corroborate "interior". '
            'The duals obey barrier complementarity (|dual| x slack constant at '
            f'{rng("barrier_product")[0]:.3g} to {rng("barrier_product")[1]:.3g}), as do the DSO interior duals at '
            'their own, smaller, constant.'),
    }


def cross_check(cells, row, analysis):
    """Every cell / row quantity that alpha_row_analysis.json also carries: max relative difference."""
    diffs = []

    def cmp(name, a, b):
        if a is None or b is None:
            diffs.append((name, None if a == b else float('inf')))
            return
        diffs.append((name, abs(a - b) / max(abs(b), 1e-300) if a != b else 0.0))
    exact = []
    for k, c in cells.items():
        a = analysis['cells'][k]
        for q in ('Q', 'bar', 'charge', 'C', 'VP', 'V', 'P_used'):
            cmp(f'{k}.{q}', c[q], a[q])
        cmp(f'{k}.rule10', c['rule10'], a['rule_ten']['terminal_step_over_threshold'])
        cmp(f'{k}.q_leg', c['q_leg']['share_q_over_p_plus_q'], a['q_leg_share'])
        cmp(f'{k}.E_abs_d_w', c['E_abs_d_w'], a['dispersion']['E_abs_d_p_mwh_weighted'])
        cmp(f'{k}.E_abs_d_u', c['E_abs_d_u'], a['dispersion']['E_abs_d_p_mwh_unweighted'])
        cmp(f'{k}.S2_w', c['S2_w'], a['dispersion']['sum_omega_d2_p_mw2h_weighted'])
        cmp(f'{k}.S2_u', c['S2_u'], a['dispersion']['sum_omega_d2_p_mw2h_unweighted'])
        cmp(f'{k}.peak_abs_d', c['peak_abs_d'], a['dispersion']['peak_abs_d_mw'])
        cmp(f'{k}.market_share_u', c['market_share_u'], a['split_unweighted_w46']['market_share_of_sum_omega_d2'])
        cmp(f'{k}.row18_cond_sum_c', c['row18_condition']['sum_c_mw_unweighted'],
            a['row18_condition_entries']['sum_c_mw_unweighted'])
        for net, v in c['curtailment_totals'].items():
            cmp(f'{k}.curt.{net}.E_plus', v['E_plus_mwh_w'], a['curtailment_per_network'][net]['E_plus_mwh_weighted'])
            cmp(f'{k}.curt.{net}.priced', v['priced_eur_w'], a['curtailment_per_network'][net]['priced_eur_weighted'])
        exact.append((f'{k}.classes', c['classes_recorded'] == a['curtailment_entry_classes']))
        exact.append((f'{k}.row18_cond_n', c['row18_condition']['n'] == a['row18_condition_entries']['n']))
    ar = analysis['row']
    for p, ap in zip(row['envelope_pairs'], ar['adjacent_pairs']):
        tag = f"pair({p['alpha1']},{p['alpha2']})"
        cmp(f'{tag}.dQ', p['dQ'], ap['envelope']['q_only']['dQ'])
        cmp(f'{tag}.dV', p['dV'], ap['envelope']['dV'])
        cmp(f'{tag}.lb', p['lb'], ap['envelope']['lb'])
        cmp(f'{tag}.ub', p['ub'], ap['envelope']['ub'])
        cmp(f'{tag}.dC', p['dC'], ap['dC'])
        cmp(f'{tag}.dP', p['dP'], ap['dP'])
        cmp(f'{tag}.resolution', p['resolution'], ap['resolution'])
    u, au = row['unit'], ar['unit']
    cmp('unit.value', u['value_eur'], au['value_eur'])
    cmp('unit.resolution', u['resolution_two_cell'], au['resolution'])
    cmp('unit.R', u['R'], au['ratio_to_srp1'])
    cmp('unit.ratio_resolution', u['ratio_resolution'], au['ratio_resolution'])
    finite = [d for _, d in diffs if d is not None]
    return {'n_compared': len(diffs), 'max_rel_diff': max(finite) if finite else None,
            'worst': max(diffs, key=lambda t: -1 if t[1] is None else t[1]),
            'exact_checks': dict(exact), 'all_exact_checks_pass': all(v for _, v in exact)}


def run():
    if os.path.exists(OUT_JSON):
        raise SystemExit(f'REFUSED: {OUT_JSON} exists (write-once)')
    started = datetime.now(timezone.utc).isoformat()
    read, problems = verify_inputs()
    print(f'[W72] inputs verified: {len(read)} files, problems: {problems}', flush=True)
    if problems:
        fails = GUARD.verify(0)
        raise SystemExit(f'REFUSED: input verification failed {problems}; guard verify(0) {fails}')
    d_tol = _ast_constant(INPUT_PINS['curtailment_audit'][0], 'D_TOL_MW')
    launcher_consts = {n: _ast_constant(INPUT_PINS['launcher'][0], n) for n in ('PILOT_VALUE_EUR', 'PILOT_R_RATIO')}
    pilot = _load(PILOT_RESULTS)
    s47 = _load(INPUT_PINS['srp1_s47_results'][0])['points']['n7_4h_e1']
    sigma_q = float(_load(INPUT_PINS['phase_a_tables'][0])['T3']['residual_max_abs_eur'])
    smoke_r1 = _load(INPUT_PINS['smoke_r1_gate'][0])['reported_not_gated']['q_leg_share']
    smoke_r2 = _load(SMOKE_R2_GATE)['reported_not_gated']['q_leg_share']
    cells = {}
    for label, (ed, _pair) in CELLS.items():
        cells[label] = cell_quantities(label, ed, d_tol)
        c = cells[label]
        print(f"[W72] {label}: alpha={c['alpha']} Q={c['Q']!r} bar={c['bar']!r} classes_corrected="
              f"{c['classes_corrected']} dso_interior={c['dso_interior_corrected']} "
              f"dso_E_plus={c['dso_E_plus_mwh_w']!r}", flush=True)
    row = row_quantities(cells, pilot, s47, sigma_q, launcher_consts, smoke_r1, smoke_r2)
    xc = cross_check(cells, row, _load(ANALYSIS_JSON))
    print(f"[W72] cross-check vs alpha_row_analysis.json: n={xc['n_compared']} max_rel={xc['max_rel_diff']} "
          f"exact={xc['all_exact_checks_pass']}", flush=True)
    fails = GUARD.verify(0)
    counts = dict(GUARD.counts)
    GUARD.uninstall()
    out = {
        'stage': 'P5.15 Addendum 40 ruling 1, W71 -> W72: alpha-row recomputation (promoted) + the W72 rulings',
        'utc_started': started, 'utc_finished': datetime.now(timezone.utc).isoformat(),
        'git_head': os.popen('git rev-parse HEAD').read().strip(),
        'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
        'instance': {'campaign': 's53_alpha_row_v25', 'spec': '70965374', 'frozen_spec': 'v25 407a4b33',
                     'flexibility_price': 'BASELINE (NOT the m = 2 variant)',
                     'cells': {k: {'eval_dir': v['eval_dir'], 'candidate_key': v['candidate_key'],
                                   'eval_key': v['eval_key'], 'alpha': v['alpha']} for k, v in cells.items()}},
        'objective_convention': ('Q = certified gross_operational_cost (settlement-excluded), UNPOLISHED; V = Q + '
                                 'voltage_pin_total; C = Q - charge; polished figures reported beside, never '
                                 'substituted.'),
        'formulas': FORMULAS,
        'constants': {'D_TOL_MW': d_tol, 'SLACK_SWEEP': SLACK_SWEEP, 'DUAL_SWEEP': DUAL_SWEEP,
                      'SEPARATION_DECADE': SEPARATION_DECADE, 'N_NEAR': N_NEAR},
        'inputs_sha256': read,
        'harness_imported_curtailment_audit': 'p515_s53_curtailment_audit' in sys.modules,
        'classification_w72': classification_summary(cells),
        'cells': cells, 'row': row, 'cross_check_vs_alpha_row_analysis': xc,
        'guard': {'counts': counts, 'verify_0': fails, 'declared_solves': 0},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'x') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'[W72] wrote {OUT_JSON}; guard {counts} verify(0) {fails}', flush=True)
    if fails:
        raise SystemExit(1)


def manifest():
    if os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
    out = _load(OUT_JSON)
    bad = [p for p, h in out['inputs_sha256'].items() if _sha(p) != h]
    if out['script_sha256'] != _sha(SCRIPT_REL):
        bad.append(SCRIPT_REL)
    if bad:
        raise SystemExit(f'REFUSED: changed since the run: {bad}')
    m = {OUT_JSON: _sha(OUT_JSON), OUT_LOG: _sha(OUT_LOG), SCRIPT_REL: _sha(SCRIPT_REL)}
    m.update(out['inputs_sha256'])
    with open(OUT_MANIFEST, 'x') as f:
        json.dump(m, f, indent=1)
    fails = GUARD.verify(0)
    print(f'[W72] wrote {OUT_MANIFEST}: {len(m)} entries; guard {dict(GUARD.counts)} verify(0) {fails}', flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--run']:
        run()
    elif sys.argv[1:] == ['--manifest']:
        manifest()
    else:
        raise SystemExit('usage: p515_s53_alpha_row_recompute.py --run | --manifest')
