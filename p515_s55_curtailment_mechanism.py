"""P5.15 S55 -- records-only: WHAT CAUSES THE TN RES CURTAILMENT IN THE PRICE-TAKER (NRF) ARRANGEMENT?
ZERO SOLVES. Nothing is built or solved; the canonical planning problem is READ (p56a_oracle.load_baseline, the
production reader, for the TN network data and prices) and the certified coordinated x = 0 models are UNPICKLED
(sha-verified) and only evaluated with pe.value. An armed `SolveProfileGuard(permitted=())` is installed BEFORE any
project import and verified at exactly 0 at the end.

QUESTION (Expert, Section 4.1): the price-taker arm curtails ~807.7 GWh of TN RES (day-weighted, net), coordinated
~0. Candidate mechanisms: H1 fixed DN interface schedules the TN must absorb, mis-timed against TN RES availability;
H2 minimum output of conventional generators; H3 the no-reverse-flow rule; H4 TN network limits; H5 (other) no
sink for TN surplus besides the DN interfaces (no TN own load, no interconnection / export).

INPUTS (each sha-pinned; see `inputs_sha256` in the output)
  PT   w116 price-taker NRF warm_from_certified_r2 arm record (the best start, Q 744,770,310.59; manuscript's 808.x):
         phase_A.tso_interface_schedule[node][year][day].p_mw   TN-side interface P per hour (MW; DN import > 0)
         phase_A.curtailment_signed_parts.per_block['TSO|-|y|d'] TN curtailment, rep-day MWh (net / positive part)
           -- the rows report_v3 `curtailment_table` uses (phase A); phase C read beside it
         phase_A.evaluation.block_components['TSO|-|y|d']       TSO block gross cost, block-weighted EUR
  CERT certified_models.pkl of the settled x = 0 cell d110bd1a (cycle 181): coordinated TN dispatch per hour
  LL   w106 lambda_look.json: coordinated lmp at the TN interface buses and pi_t per hour (context)
  TCC  w106 tso_coupling_check.json: coordinated TSO block gross cost (validates "TSO cost = conv energy x pi")
  R3   report_v3.json: curtailment_table (totals cross-check); W124 w124_block_energy_check.json (the 10/12 rule)
  NET  planning.transmission_network.network[y][d] (production reader): generator type / Pmin / Pmax / bus,
       RES availability gen.pg[0][p], loads and their buses, branch ratings, cost_energy_p (pi_t)

FORMULAS (block b = (year, day); h = 1..24; MW = MWh per hour of the representative day)
  R[h]      = sum over curtaillable TN generators of baseMVA x pg_avail                    (TN RES available)
  D_pt[h]   = sum_n PT phase_A tso_interface_schedule[n].p_mw[h]                          (fixed TN demand, arm)
  D_co[h]   = sum_n baseMVA x expected_interface_pf_p[n,h] (CERT TN block)                (coordinated)
  S[h]      = R[h] - D_pt[h]   (surplus of TN RES over the only TN sink); X[b] = sum_h max(0, S[h])
  conv, RES, losses (coordinated, CERT): L_co[h] = sum_g baseMVA pg - sum_nodes baseMVA pc_node
  Balance of the arm's TN at hour h (single scenario; no TN load other than the ADN loads; no storage at x = 0):
      conv[h] + R[h] - c[h] = D_pt[h] + L_pt[h],  conv[h] >= Pmin_total = 0,  c[h] >= 0
    => c[h] >= S[h] - L_pt[h]   (with equality iff conv[h] = 0)
  C_A[b]    = recorded arm TN curtailment (phase A net, rep-day MWh)
  implied loss at surplus hours = (X - C_A) / n_surplus_hours   (if curtailment occurs only at surplus hours, at
             conv = 0; compare with the coordinated hourly losses L_co -- the arm's own losses are not recorded)
  proxy (a) c_hat[h] = max(0, S[h] - L_co[h]) (coordinated loss at the same hour as a proxy), C_hat_a = sum_h c_hat
  proxy (b) one constant loss L_b per block fitted so that sum_h max(0, S[h] - L_b) = C_A
  HELD-OUT CHECK (the fit uses curtailment only; the cost was not used): predicted TSO block cost
      K_hat = block_weight x sum_h pi[h] x baseMVA-free max(0, D_pt[h] + L[h] - R[h])  (MWh x EUR/MWh)
    against the recorded PT block cost, for L = L_co (a) and L = L_b (b). Validated first on the coordinated side:
    block_weight x sum_h pi x conv_co (CERT) against TCC certified_coordinated.block_gross_cost_weighted.

Output: write-once NEW directory data/SRP1/Results/P515S55/curtailment_mechanism/ (refuses if it exists):
  curtailment_mechanism.json, launch.log (the caller's capture of stdout+stderr), manifest_sha256.json.
Smoke (nothing written): --dry. Launch (attached, alone, both streams captured), then the manifest:
    mkdir data/SRP1/Results/P515S55/curtailment_mechanism && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_curtailment_mechanism.py \\
        > data/SRP1/Results/P515S55/curtailment_mechanism/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_curtailment_mechanism.py --manifest
"""
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 S55 curtailment mechanism zero-solve').install()

import gate_result_io as GRIO  # noqa: E402

THIS = os.path.abspath(__file__)
RES = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'curtailment_mechanism')
OUT_DIR = os.path.join(REPO, OUT_DIR_REL)
OUT_JSON = os.path.join(OUT_DIR, 'curtailment_mechanism.json')
LAUNCH_LOG = os.path.join(OUT_DIR, 'launch.log')
MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')

PT_RUN = 'nrf_arm_price_taker_warm_from_certified_r2'
PT = os.path.join(RES, 'w116_benchmark_nrf', PT_RUN, f'{PT_RUN}.json')
LL = os.path.join(RES, 'w106_uncoordinated_settled', 'lambda_look', 'lambda_look.json')
TCC = os.path.join(RES, 'w106_uncoordinated_settled', 'tso_coupling_check', 'tso_coupling_check.json')
R3 = os.path.join(RES, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
W124 = os.path.join(RES, 'w124_block_energy_check', 'w124_block_energy_check.json')
CHILD_MANIFEST = os.path.join(RES, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0', 'evals',
                              'd110bd1a5977df1e_x0', 'child_manifest_sha256.json')

NODES = ['5', '7', '9']
NRF_AT_BOUND_MW = 1e-4          # 1e-6 p.u. x 100 MVA: the material tolerance of the NRF count
BOUND_TOL_PU2 = 1e-4            # voltage-squared within this of a bound counts as "at bound"


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


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


INPUTS = {}


def _git_status(rel):
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], capture_output=True, cwd=REPO).returncode == 0
    clean = tracked and subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0
    commit = subprocess.run(['git', 'log', '-1', '--format=%H', '--', rel], capture_output=True, text=True,
                            cwd=REPO).stdout.strip() if tracked else None
    return tracked, clean, commit


def _pin(rel, manifest_rel=None, require_tracked=True):
    """sha256 of an input against the manifest that hash-records it; raise on mismatch."""
    manifest_rel = manifest_rel or os.path.join(os.path.dirname(rel), 'manifest_sha256.json')
    now = _sha(os.path.join(REPO, rel))
    man = _load(manifest_rel)
    pinned = man.get(rel)
    if pinned is None and isinstance(man.get('files'), dict):
        pinned = man['files'].get(rel)
        pinned = pinned.get('sha256') if isinstance(pinned, dict) else pinned
    if pinned is None:
        raise RuntimeError(f'{rel}: not hash-recorded in {manifest_rel}')
    if pinned != now:
        raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned} ({manifest_rel})')
    tracked, clean, commit = _git_status(rel)
    if require_tracked and not (tracked and clean):
        raise RuntimeError(f'{rel}: tracked {tracked}, clean {clean}')
    INPUTS[rel] = {'sha256': now, 'manifest': manifest_rel, 'git_tracked': tracked, 'git_clean': clean,
                   'git_last_commit': commit}
    return now


def _fit_constant_loss(S, target):
    """L >= 0 with sum_h max(0, S[h] - L) = target (monotone, bisection). None if target > sum S+ (no L >= 0)."""
    xs = sum(max(0.0, s) for s in S)
    if target > xs + 1e-9 or target <= 0.0:
        return None          # undefined: curtailment beyond the surplus, or no curtailment to fit
    lo, hi = 0.0, max(S)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(max(0.0, s - mid) for s in S) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    dry = '--dry' in sys.argv
    t0 = time.time()
    if not dry:
        # write-once: the caller creates the directory to capture launch.log; nothing else may be in it
        if not os.path.isdir(OUT_DIR):
            raise RuntimeError(f'{OUT_DIR_REL} must be created by the launch command (launch.log capture)')
        extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
        if extra or os.path.exists(OUT_JSON):
            raise RuntimeError(f'refusing: {OUT_DIR_REL} already holds {extra} (write-once)')

    # ---- pin inputs ---------------------------------------------------------------------------------------------
    _pin(PT)
    _pin(LL)
    _pin(TCC)
    _pin(R3)
    _pin(W124)
    pt = _load(PT)
    ll = _load(LL)
    tcc = _load(TCC)
    r3 = _load(R3)
    w124 = _load(W124)
    cert_entry = pt['coordinated_cell']['certified_models']
    _pin(CHILD_MANIFEST, manifest_rel=pt['coordinated_cell']['campaign_manifest'])
    child_manifest = _load(CHILD_MANIFEST)
    cert_sha = _sha(os.path.join(REPO, cert_entry['path']))
    child_pin = child_manifest.get(cert_entry['path'])
    if child_pin is None and isinstance(child_manifest.get('files'), dict):
        child_pin = child_manifest['files'].get(cert_entry['path'])
        child_pin = child_pin.get('sha256') if isinstance(child_pin, dict) else child_pin
    if not (cert_sha == cert_entry['sha256'] == child_pin):
        raise RuntimeError(f'certified models sha {cert_sha} vs arm record {cert_entry["sha256"]} vs child {child_pin}')
    tracked, clean, commit = _git_status(cert_entry['path'])
    INPUTS[cert_entry['path']] = {'sha256': cert_sha, 'manifest': CHILD_MANIFEST, 'git_tracked': tracked,
                                  'git_clean': clean, 'git_last_commit': commit,
                                  'also_declared_in': f'{PT} coordinated_cell.certified_models'}
    _log(f'{len(INPUTS)} inputs pinned')
    if pt['instance']['candidate_key'] != ll['instance']['candidate_key']:
        raise RuntimeError('instance mismatch between PT and LL')

    # ---- production readers (no build, no solve) ------------------------------------------------------------
    import pyomo.environ as pe
    import p56a_oracle as O
    import uncoordinated_benchmark as UB
    import shared_resources_planning as srp
    from definitions import GEN_CONV, GEN_INTERCONNECTION, GEN_REFERENCE

    planning = O.load_baseline()['planning']
    tn = planning.transmission_network
    adn_nodes = [int(n) for n in tn.active_distribution_network_nodes]
    _log(f'planning read; TN years {list(tn.years)} days {list(tn.days)}; ADN nodes {adn_nodes}')

    # ---- TN static data (H2, H5, ratings for H4) --------------------------------------------------------------
    static = {}
    for year in tn.years:
        net = tn.network[year][list(tn.days)[0]]
        base = net.baseMVA
        gens = [{'gen_id': g.gen_id, 'bus': g.bus, 'type_code': g.gen_type, 'type': net.get_gen_type(g.gen_id),
                 'curtaillable': bool(g.is_curtaillable()), 'controllable': bool(g.is_controllable()),
                 'pmin_mw': g.pmin * base, 'pmax_mw': g.pmax * base} for g in net.generators]
        loads = [{'load_id': c.load_id, 'bus': c.bus, 'is_adn_node': c.bus in adn_nodes} for c in net.loads]
        branches = [{'branch_id': br.branch_id, 'fbus': br.fbus, 'tbus': br.tbus, 'rating_mva': br.rate,
                     'is_transformer': bool(br.is_transformer)} for br in net.branches]
        static[str(year)] = {
            'baseMVA': base, 'generators': gens, 'loads': loads, 'branches': branches,
            'conv_pmin_total_mw': sum(g['pmin_mw'] for g in gens if not g['curtaillable']),
            'res_pmax_total_mw': sum(g['pmax_mw'] for g in gens if g['curtaillable']),
            'n_interconnection_or_ref_generators': sum(1 for g in net.generators
                                                       if g.gen_type in (GEN_INTERCONNECTION, GEN_REFERENCE)),
            'all_tn_loads_at_adn_nodes': all(c['is_adn_node'] for c in loads),
            'n_energy_storages': len(net.energy_storages),
            'min_branch_rating_mva': min(br.rate for br in net.branches),
        }
    _log('TN static: ' + json.dumps({y: {k: v for k, v in s.items() if k not in ('generators', 'loads', 'branches')}
                                      for y, s in static.items()}))

    # ---- coordinated certified models -------------------------------------------------------------------------
    _log(f"unpickling {cert_entry['path']} (sha256 verified {cert_sha[:8]})")
    with open(os.path.join(REPO, cert_entry['path']), 'rb') as handle:
        certified = pickle.load(handle)
    _log(f'unpickled in {time.time() - t0:.1f} s')
    co_curt = UB.curtailment_report(planning, {'tso': certified['tso'], 'dso': certified['dso']})
    r3_co = r3['curtailment_table']['coordinated']['settled_cycle_181_recomputed_from_models']['by_agent']
    co_curt_check = {
        'tso_mwh_rep_day_sum_here': co_curt['totals']['TSO']['mwh_rep_day_sum'],
        'tso_mwh_rep_day_sum_report_v3': r3_co['TSO']['mwh_rep_day_sum'],
        'dso_mwh_rep_day_sum_here': co_curt['totals']['DSO']['mwh_rep_day_sum'],
        'dso_mwh_rep_day_sum_report_v3': r3_co['DSO']['mwh_rep_day_sum'],
    }
    co_curt_check['passed'] = (abs(co_curt_check['tso_mwh_rep_day_sum_here'] - co_curt_check['tso_mwh_rep_day_sum_report_v3']) < 1e-9
                               and abs(co_curt_check['dso_mwh_rep_day_sum_here'] - co_curt_check['dso_mwh_rep_day_sum_report_v3']) < 1e-9)
    _log(f'coordinated curtailment cross-check vs report_v3: {co_curt_check}')

    # lambda look index
    lam = {}
    for row in ll['table_coordinated']:
        lam[(str(row['year']), row['day'], int(row['hour']), str(row['node_id']))] = row

    pa = pt['phase_A']
    pc = pt['phase_C_sequential_pass']
    # report_v3 sweep (price-taker, NO interface rule): hours at which the TN could not accept the DN exchange
    sweep_hours = r3['sweep']['sweep_price_taker_cold']['per_block_hours_not_accepted']
    blocks = {}
    identity_checks = {'res_avail_pickle_vs_network_data_max_abs_mw': 0.0, 'pi_network_vs_lambda_look_max_abs': 0.0,
                       'conv_pg_lb_max_mw': -math.inf, 'coordinated_balance_pc_node_vs_interface_max_abs_mw': 0.0,
                       'pt_tso_vs_dso_schedule_max_abs_mw': 0.0, 'min_pi_eur_mwh': math.inf}
    for year in tn.years:
        for day in tn.days:
            ys = str(year)
            label = UB.block_label('TSO', None, year, day)
            net = tn.network[year][day]
            blk = certified['tso'][year][day]
            base = net.baseMVA
            periods = list(blk.periods)
            res_idx = [g for g, gen in enumerate(net.generators) if gen.is_curtaillable()]
            conv_idx = [g for g, gen in enumerate(net.generators) if not gen.is_curtaillable()]
            weight = srp._get_admm_block_weight(tn, year, day)
            day_weight = float(tn.years[year]) * float(tn.days[day])
            pi = [float(net.cost_energy_p[0][p]) for p in periods]
            identity_checks['min_pi_eur_mwh'] = min(identity_checks['min_pi_eur_mwh'], min(pi))
            R, conv_co, res_co, D_co, L_co, pcn = [], [], [], [], [], []
            for p in periods:
                r = sum(base * pe.value(blk.pg_avail[g, 0, p]) for g in res_idx)
                r_data = sum(base * max(0.0, net.generators[g].pg[0][p]) for g in res_idx)
                identity_checks['res_avail_pickle_vs_network_data_max_abs_mw'] = max(
                    identity_checks['res_avail_pickle_vs_network_data_max_abs_mw'], abs(r - r_data))
                for g in conv_idx:
                    identity_checks['conv_pg_lb_max_mw'] = max(identity_checks['conv_pg_lb_max_mw'],
                                                               base * blk.pg[g, 0, 0, p].lb)
                R.append(r)
                conv_co.append(sum(base * pe.value(blk.pg[g, 0, 0, p]) for g in conv_idx))
                res_co.append(sum(base * pe.value(blk.pg[g, 0, 0, p]) for g in res_idx))
                D_co.append(sum(base * pe.value(blk.expected_interface_pf_p[dn, p]) for dn in range(len(adn_nodes))))
                pcn.append(sum(base * pe.value(blk.pc_node[n, 0, 0, p]) for n in blk.nodes))
                L_co.append(conv_co[-1] + res_co[-1] - pcn[-1])
                identity_checks['coordinated_balance_pc_node_vs_interface_max_abs_mw'] = max(
                    identity_checks['coordinated_balance_pc_node_vs_interface_max_abs_mw'], abs(pcn[-1] - D_co[-1]))
                for n in NODES:
                    row = lam.get((ys, day, p + 1, n))
                    if row is not None:
                        identity_checks['pi_network_vs_lambda_look_max_abs'] = max(
                            identity_checks['pi_network_vs_lambda_look_max_abs'], abs(row['pi'] - pi[p]))
            # coordinated branch loading (current-limited lines: sqrt(flow_ij_sqr) / rating in p.u.)
            branch_load = {}
            for b in blk.branches:
                br = net.branches[b]
                rating = br.rate / base
                vals = [math.sqrt(max(0.0, pe.value(blk.flow_ij_sqr[b, 0, 0, p]))) / rating for p in periods]
                if b in blk.apparent_power_limited_branches:
                    vals = [max(v, math.sqrt(max(0.0, pe.value(blk.flow_ji_sqr[b, 0, 0, p]))) / rating)
                            for v, p in zip(vals, periods)]
                k = max(range(len(vals)), key=lambda i: vals[i])
                branch_load[f'{br.fbus}-{br.tbus}'] = {'max_loading': vals[k], 'hour_of_max': k + 1,
                                                       'rating_mva': br.rate}
            v_at_bound = 0
            v_slack_max = 0.0
            for n in blk.nodes:
                for p in periods:
                    var = blk.vmag_sqr[n, 0, 0, p]
                    val = pe.value(var)
                    if (var.lb is not None and val - var.lb < BOUND_TOL_PU2) or (var.ub is not None and var.ub - val < BOUND_TOL_PU2):
                        v_at_bound += 1
                    if hasattr(blk, 'slack_v_sqr_up'):
                        v_slack_max = max(v_slack_max, pe.value(blk.slack_v_sqr_up[n, 0, 0, p]),
                                          pe.value(blk.slack_v_sqr_down[n, 0, 0, p]))
            # arm
            D_pt_n = {n: [float(x) for x in pa['tso_interface_schedule'][n][ys][day]['p_mw']] for n in NODES}
            D_pt_dso = {n: [float(x) for x in pa['dso_interface_schedule'][n][ys][day]['p_mw']] for n in NODES}
            identity_checks['pt_tso_vs_dso_schedule_max_abs_mw'] = max(
                identity_checks['pt_tso_vs_dso_schedule_max_abs_mw'],
                max(abs(a - b) for n in NODES for a, b in zip(D_pt_n[n], D_pt_dso[n])))
            D_pt = [sum(D_pt_n[n][p] for n in NODES) for p in periods]
            S = [R[p] - D_pt[p] for p in periods]
            surplus_hours = [p for p in periods if S[p] > 0.0]
            X = sum(S[p] for p in surplus_hours)
            cA = pa['curtailment_signed_parts']['per_block'][label]
            cC = pc['curtailment_signed_parts']['per_block'][label]
            C_A = cA['net_mwh_rep_day']
            n_sur = len(surplus_hours)
            implied_loss = (X - C_A) / n_sur if n_sur else None
            c_hat = [max(0.0, S[p] - L_co[p]) for p in periods]
            C_hat_a = sum(c_hat)
            L_b = _fit_constant_loss(S, C_A)
            K_rec = pa['evaluation']['block_components'][label]
            K_rec_C = pc['evaluation']['block_components'][label]
            K_co_tcc = tcc['per_block'][label]['certified_coordinated']['block_gross_cost_weighted']
            K_co_conv = weight * sum(pi[p] * conv_co[p] for p in periods)
            K_hat_a = weight * sum(pi[p] * max(0.0, D_pt[p] + L_co[p] - R[p]) for p in periods)
            K_hat_b = (weight * sum(pi[p] * max(0.0, D_pt[p] + L_b - R[p]) for p in periods)
                       if L_b is not None else None)
            S_co = [R[p] - D_co[p] for p in periods]
            hours = []
            for p in surplus_hours:
                lam_n = [lam[(ys, day, p + 1, n)]['lmp_tn_bus'] for n in NODES if (ys, day, p + 1, n) in lam]
                hours.append({
                    'hour': p + 1, 'res_avail_mw': R[p], 'dn_import_pt_mw': D_pt[p], 'surplus_mw': S[p],
                    'dn_import_pt_per_node_mw': {n: D_pt_n[n][p] for n in NODES},
                    'nodes_at_nrf_bound': [n for n in NODES if D_pt_n[n][p] <= NRF_AT_BOUND_MW],
                    'dn_import_co_mw': D_co[p], 'co_minus_pt_import_mw': D_co[p] - D_pt[p],
                    'coordinated_conv_mw': conv_co[p], 'coordinated_res_used_mw': res_co[p],
                    'coordinated_tn_losses_mw': L_co[p], 'curtailment_proxy_a_mw': c_hat[p],
                    'pi_eur_mwh': pi[p], 'coordinated_lmp_tn_interface_mean_eur_mwh':
                        (sum(lam_n) / len(lam_n)) if lam_n else None,
                })
            pis = sorted(pi)
            blocks[label] = {
                'year': ys, 'day': day, 'block_weight': weight, 'day_weight': day_weight,
                'tn_curtailment_price_taker': {
                    'phase_A_net_mwh_rep_day': C_A, 'phase_A_positive_part_mwh_rep_day': cA['positive_part_mwh_rep_day'],
                    'phase_A_net_mwh_day_weighted': C_A * day_weight,
                    'phase_C_net_mwh_rep_day': cC['net_mwh_rep_day'],
                    'phase_C_net_mwh_day_weighted': cC['net_mwh_rep_day'] * day_weight,
                },
                'tn_curtailment_coordinated': {
                    'net_mwh_rep_day': co_curt['per_block'][label]['mwh_rep_day'],
                    'net_mwh_day_weighted': co_curt['per_block'][label]['mwh_rep_day'] * day_weight,
                    'hourly_max_mw': max(R[p] - res_co[p] for p in periods),
                },
                'energy_rep_day_mwh': {'res_avail': sum(R), 'dn_import_pt': sum(D_pt), 'dn_import_co': sum(D_co),
                                       'conv_co': sum(conv_co), 'res_used_co': sum(res_co),
                                       'tn_losses_co': sum(L_co)},
                'surplus': {
                    'n_hours': n_sur, 'hours': [p + 1 for p in surplus_hours], 'X_mwh_rep_day': X,
                    'X_mwh_day_weighted': X * day_weight,
                    'X_minus_C_A_mwh_rep_day': X - C_A,
                    'implied_mean_loss_at_surplus_hours_mw': implied_loss,
                    'coordinated_loss_range_mw': [min(L_co), max(L_co)],
                    'coordinated_loss_at_surplus_hours_mean_mw': (sum(L_co[p] for p in surplus_hours) / n_sur)
                    if n_sur else None,
                    'proxy_a_C_hat_mwh_rep_day': C_hat_a,
                    'proxy_a_C_hat_minus_C_A_mwh_rep_day': C_hat_a - C_A,
                    'proxy_b_constant_loss_fit_mw': L_b,
                    'coordinated_surplus_positive_part_mwh_rep_day': sum(max(0.0, s) for s in S_co),
                    'coordinated_n_surplus_hours': sum(1 for s in S_co if s > 0.0),
                },
                'tso_cost_check_block_weighted_eur': {
                    'coordinated_tcc_recorded': K_co_tcc, 'coordinated_pi_x_conv_from_models': K_co_conv,
                    'coordinated_rel_diff': (K_co_conv - K_co_tcc) / K_co_tcc,
                    'pt_recorded_phase_A': K_rec, 'pt_recorded_phase_C': K_rec_C,
                    'pt_predicted_a_loss_coordinated': K_hat_a, 'pt_rel_err_a': (K_hat_a - K_rec) / K_rec,
                    'pt_predicted_b_loss_fitted': K_hat_b,
                    'pt_rel_err_b': ((K_hat_b - K_rec) / K_rec) if K_hat_b is not None else None,
                    # energy the cost residual of (a) could at most represent, at the block's cheapest hour:
                    # |K_rec - K_hat_a| / (block_weight x min_h pi) [MWh per rep day]; an upper bound on curtailment
                    # not explained by the surplus accounting (whatever the residual's cause, e.g. arm losses)
                    'residual_energy_bound_mwh_rep_day': abs(K_rec - K_hat_a) / (weight * min(pi)),
                    'residual_energy_bound_over_C_A': (abs(K_rec - K_hat_a) / (weight * min(pi)) / C_A)
                    if C_A > 0 else None,
                },
                'sweep_no_rule_hours_not_accepted': sweep_hours.get(label, []),
                'sweep_hours_that_are_nrf_binding_surplus_hours': sorted(
                    set(sweep_hours.get(label, [])) & {h['hour'] for h in hours if h['nodes_at_nrf_bound']}),
                'coordinated_conv_at_surplus_hours_max_mw': max((conv_co[p] for p in surplus_hours), default=None),
                'surplus_hours_detail': hours,
                'pi_block_median_eur_mwh': 0.5 * (pis[11] + pis[12]),
                'pi_rank_of_surplus_hours_1_cheapest': [sorted(range(24), key=lambda i: pi[i]).index(p) + 1
                                                        for p in surplus_hours],
                'h4_bound': {
                    'max_dn_import_pt_at_surplus_hours_mw': max((D_pt[p] for p in surplus_hours), default=None),
                    'max_res_avail_mw': max(R),
                    'coordinated_branch_loading': branch_load,
                    'coordinated_max_branch_loading': max(v['max_loading'] for v in branch_load.values()),
                    'coordinated_vmag_sqr_node_hours_at_bound': v_at_bound,
                    'coordinated_voltage_slack_max_pu2': v_slack_max,
                },
                'nrf': {
                    'surplus_hours_with_any_node_at_nrf_bound': sum(1 for h in hours if h['nodes_at_nrf_bound']),
                    'min_dn_import_pt_at_surplus_hours_mw': min((min(D_pt_n[n][p] for n in NODES)
                                                                 for p in surplus_hours), default=None),
                },
                'hourly': {'res_avail_mw': R, 'dn_import_pt_mw': D_pt, 'dn_import_co_mw': D_co, 'surplus_mw': S,
                           'coordinated_conv_mw': conv_co, 'coordinated_res_used_mw': res_co,
                           'coordinated_tn_losses_mw': L_co, 'curtailment_proxy_a_mw': c_hat, 'pi_eur_mwh': pi},
            }
    del certified

    # ---- aggregates --------------------------------------------------------------------------------------------
    agg = {'by_year': {}, 'by_day': {}, 'by_hour_of_day_day_weighted_mwh': {}}
    tot = {'C_A_day_weighted': 0.0, 'C_A_rep_day': 0.0, 'X_day_weighted': 0.0, 'X_rep_day': 0.0,
           'C_hat_a_day_weighted': 0.0, 'C_co_day_weighted': 0.0}
    for label, v in blocks.items():
        dw = v['day_weight']
        for key, grp in (('by_year', v['year']), ('by_day', v['day'])):
            a = agg[key].setdefault(grp, {'C_A_mwh_day_weighted': 0.0, 'X_mwh_day_weighted': 0.0,
                                          'C_hat_a_mwh_day_weighted': 0.0})
            a['C_A_mwh_day_weighted'] += v['tn_curtailment_price_taker']['phase_A_net_mwh_day_weighted']
            a['X_mwh_day_weighted'] += v['surplus']['X_mwh_day_weighted']
            a['C_hat_a_mwh_day_weighted'] += v['surplus']['proxy_a_C_hat_mwh_rep_day'] * dw
        for p in range(24):
            h = agg['by_hour_of_day_day_weighted_mwh'].setdefault(str(p + 1), {'surplus_positive': 0.0,
                                                                               'curtailment_proxy_a': 0.0})
            h['surplus_positive'] += max(0.0, v['hourly']['surplus_mw'][p]) * dw
            h['curtailment_proxy_a'] += v['hourly']['curtailment_proxy_a_mw'][p] * dw
        tot['C_A_day_weighted'] += v['tn_curtailment_price_taker']['phase_A_net_mwh_day_weighted']
        tot['C_A_rep_day'] += v['tn_curtailment_price_taker']['phase_A_net_mwh_rep_day']
        tot['X_day_weighted'] += v['surplus']['X_mwh_day_weighted']
        tot['X_rep_day'] += v['surplus']['X_mwh_rep_day']
        tot['C_hat_a_day_weighted'] += v['surplus']['proxy_a_C_hat_mwh_rep_day'] * dw
        tot['C_co_day_weighted'] += v['tn_curtailment_coordinated']['net_mwh_day_weighted']
    tot['report_v3_pt_tso_net_mwh_day_weighted'] = (
        r3['curtailment_table']['arms_and_variants_phase_A'][PT_RUN]['by_agent']['TSO']['mwh_day_weighted'])
    tot['C_A_matches_report_v3'] = abs(tot['C_A_day_weighted'] - tot['report_v3_pt_tso_net_mwh_day_weighted']) < 1e-6

    w124_rule = {'verdict_rule': w124.get('verdict_rule'), 'script': 'p515_s53_w124_block_energy_check.py',
                 'verdicts': {b: {'R_total': w124['blocks'][b].get('verdict_R_total'),
                                  'ratio_dtso_per_curtailed_mwh_eq': w124['blocks'][b].get(
                                      'ratio_dtso_per_curtailed_mwh_eq_eur_mwh'),
                                  'pi_range': w124['blocks'][b].get('pi_range_eur_mwh')}
                              for b in w124['blocks']}}

    guard_failures = _GUARD.verify(0)
    out = {
        'stage': 'P5.15 S55 -- records-only mechanism of TN RES curtailment in the price-taker NRF arm (zero solves; '
                 'planning read, certified coordinated models unpickled, nothing built or solved)',
        'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
        'script': os.path.basename(THIS), 'script_sha256': _sha(THIS),
        'interpreter': sys.executable,
        'instance': {'price_taker': pt['instance'], 'coordinated_eval_key': pt['coordinated_cell']['eval_key'],
                     'coordinated_certification_cycle': pt['coordinated_cell']['certification_cycle'],
                     'price_taker_run': PT_RUN, 'price_taker_q_gross': pt['arm_cost']['gross_operational_cost']},
        'objective_convention': 'TSO block costs: gross, settlement excluded, block-weighted EUR (years x days x '
                                'discount); curtailment: production definitional helper, net, MWh per representative '
                                'day and day-weighted (years x days, undiscounted)',
        'inputs_sha256': INPUTS,
        'tn_static': static,
        'identity_checks': identity_checks,
        'coordinated_curtailment_cross_check': co_curt_check,
        'blocks': blocks,
        'aggregates': agg, 'totals': tot,
        'w124_rule_behind_10_of_12': w124_rule,
        'solve_profile_guard': {'permitted': [], 'verify_0_failures': guard_failures, 'counts': dict(_GUARD.counts)},
        'wall_s': time.time() - t0,
    }
    text = GRIO.dumps(out, indent=1, sort_keys=True, default=GRIO.json_default_item)
    if dry:
        _log(f'--dry: nothing written ({len(text)} bytes)')
    else:
        with open(OUT_JSON, 'x') as handle:
            handle.write(text)
        _log(f'wrote {OUT_JSON} sha256 {_sha(OUT_JSON)}')

    # ---- log table ---------------------------------------------------------------------------------------------
    _log(f'TN static per year: ' + json.dumps({y: {'conv_pmin_total_mw': s['conv_pmin_total_mw'],
                                                    'res_pmax_total_mw': s['res_pmax_total_mw'],
                                                    'interconnection_or_ref': s['n_interconnection_or_ref_generators'],
                                                    'all_loads_at_adn': s['all_tn_loads_at_adn_nodes'],
                                                    'n_ess': s['n_energy_storages'],
                                                    'min_rating': s['min_branch_rating_mva']}
                                                for y, s in static.items()}))
    _log(f'identity checks: {json.dumps(identity_checks)}')
    _log('block         C_A(rep)   C_A(dw MWh)   C_co(rep)  nSur   X(rep)   X-C_A  impl.L  Lco@sur  C_hat_a  '
         'L_b    K_rec(EUR)    err_a    err_b   co_cost_rel  maxDpt@sur  co_maxload  NRF@sur')
    for label, v in blocks.items():
        s, k, c = v['surplus'], v['tso_cost_check_block_weighted_eur'], v['tn_curtailment_price_taker']
        f = (lambda x, w=7, d=2: f'{x:{w}.{d}f}' if x is not None else ' ' * (w - 4) + 'None')
        _log(f"{label:13s} {c['phase_A_net_mwh_rep_day']:9.3f} {c['phase_A_net_mwh_day_weighted']:12.1f} "
             f"{v['tn_curtailment_coordinated']['net_mwh_rep_day']:9.4f} {s['n_hours']:4d} {s['X_mwh_rep_day']:8.3f} "
             f"{s['X_minus_C_A_mwh_rep_day']:7.3f} {f(s['implied_mean_loss_at_surplus_hours_mw'])} "
             f"{f(s['coordinated_loss_at_surplus_hours_mean_mw'])} {s['proxy_a_C_hat_mwh_rep_day']:8.3f} "
             f"{f(s['proxy_b_constant_loss_fit_mw'])} {k['pt_recorded_phase_A']:13.1f} {k['pt_rel_err_a']:+.4%} "
             f"{(k['pt_rel_err_b'] if k['pt_rel_err_b'] is not None else float('nan')):+.4%} "
             f"{k['coordinated_rel_diff']:+.2e} {f(v['h4_bound']['max_dn_import_pt_at_surplus_hours_mw'], 9)} "
             f"{v['h4_bound']['coordinated_max_branch_loading']:.3f} {v['nrf']['surplus_hours_with_any_node_at_nrf_bound']}")
        _log('    surplus hours (h: R / D_pt / S / D_co / pi / lmp_co): ' + '; '.join(
            f"{h['hour']}: {h['res_avail_mw']:.1f}/{h['dn_import_pt_mw']:.1f}/{h['surplus_mw']:+.1f}/"
            f"{h['dn_import_co_mw']:.1f}/{h['pi_eur_mwh']:.1f}/"
            f"{(h['coordinated_lmp_tn_interface_mean_eur_mwh'] if h['coordinated_lmp_tn_interface_mean_eur_mwh'] is not None else float('nan')):.1f}"
            for h in v['surplus_hours_detail']))
        _log(f"    residual energy bound {k['residual_energy_bound_mwh_rep_day']:.3f} MWh/rep-day "
             f"(/C_A {k['residual_energy_bound_over_C_A']}); coordinated conv max at surplus hours "
             f"{v['coordinated_conv_at_surplus_hours_max_mw']}; pi ranks {v['pi_rank_of_surplus_hours_1_cheapest']}; "
             f"sweep hours {v['sweep_no_rule_hours_not_accepted']} of which NRF-binding surplus "
             f"{v['sweep_hours_that_are_nrf_binding_surplus_hours']}; coordinated V at bound "
             f"{v['h4_bound']['coordinated_vmag_sqr_node_hours_at_bound']}, V slack max "
             f"{v['h4_bound']['coordinated_voltage_slack_max_pu2']:.2e}; W124 "
             f"{w124_rule['verdicts'].get(v['year'] + '|' + v['day'], {}).get('R_total')}")
    _log(f'aggregates: {json.dumps(agg["by_year"])} | {json.dumps(agg["by_day"])}')
    _log('by hour of day (day-weighted MWh: surplus+ / proxy_a): ' + ', '.join(
        f"{h}:{x['surplus_positive']:.0f}/{x['curtailment_proxy_a']:.0f}"
        for h, x in agg['by_hour_of_day_day_weighted_mwh'].items()))
    _log(f'totals: {json.dumps(tot)}')
    _log(f'guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {time.time() - t0:.1f} s')
    return 0 if (not guard_failures and co_curt_check['passed'] and tot['C_A_matches_report_v3']) else 1


def write_manifest():
    if os.path.exists(MANIFEST):
        raise RuntimeError(f'{MANIFEST} exists (write-once)')
    files = {}
    for name in ('curtailment_mechanism.json', 'launch.log'):
        rel = os.path.join(OUT_DIR_REL, name)
        files[rel] = _sha(os.path.join(REPO, rel))
    files[os.path.basename(THIS)] = _sha(THIS)
    with open(MANIFEST, 'x') as handle:
        json.dump(files, handle, indent=1, sort_keys=True)
    _log(f'wrote {MANIFEST}: {json.dumps(files)}')
    return 0


if __name__ == '__main__':
    if '--manifest' in sys.argv:
        _GUARD.uninstall()
        sys.exit(write_manifest())
    sys.exit(main())
