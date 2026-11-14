"""
P5.15 Addendum 51, Planner task W97 -- the diagnostics report owed before stage 1, FROM RECORDS ONLY, ZERO SOLVES.

Addendum 51, verbatim: "Diagnostics owed. Addendum 50's records-only items (i)-(v) -- per-block dQ decomposition,
consensus and dual movement per channel, residual margins, row-18 terms per cycle, iteration counts and mu -- are
reported for both cells before stage 1 launches, or the omission is stated. Add the rho history and the AA state around
cycles 55-72 on each cell."

Cells (pair campaign s53_w91_3x3_pair, spec 231558f0):
  x0        evals/f6e9cd53fdbb8ee8_x0        certified cycle 72, tail active 64-72
  n7_4h_e1  evals/c82522f470b35b58_n7_4h_e1  certified cycle 69, tail active 61-69

Part 1 CONSOLIDATES (does not re-derive) items (i)-(v) from the committed W95 output (d757394e,
w95_x0_drift/w95_x0_drift_diagnostics.json) and W96 output (c2d4b4ce, w96_n7_drift/w96_n7_drift_diagnostics.json),
carrying their stated omissions forward verbatim. Neither script nor output is edited.

Part 2 is NEW: per cycle 55..certification on each cell, from the primary records --
  * rho per channel (V, PF, ESS) before/after, the penalty action, the freeze flags / unchanged streak / clamp /
    balancing-exempt state, and gamma (the proximal weight) before/after -- g_s39_D.json cycle_trajectory,
    cross-checked against child_stdout.log's [ADMM RHO BOYD] lines (printed to 6 significant digits);
  * the Anderson-acceleration (AA) state -- aa_per_cycle.jsonl (action, accepted, memory before/after, gamma
    columns = m_k, combined residual vs the safeguard mark, reset, rho-changed channels, boyd_all_pass) and the AA
    configuration from the evaluation record;
  * the tail state -- convergence_depth_tail_state.json, cross-checked against the child stdout
    "Convergence-depth tail ON for cycle N" lines;
  * aligned against the gross Q step per cycle (per_cycle_record.jsonl).
  * DERIVED, not recorded: the PF-channel part of every AA write-back, from pf_entry_stride_s39_D.jsonl. The sidecar is
    captured inside get_admm_boyd_residual_metrics, i.e. BEFORE the AA decision of the same cycle, and holds per PF
    entry: x_dso, z_tso_current (TSO copy after this cycle's TSO update, pre-AA), z_tso_prev (the TSO copy at the START
    of this cycle = what the previous cycle left in the store, post-AA), lambda_dso (after this cycle's dual update,
    pre-AA) and rho_pf. With production's update order (shared_resources_planning.py _update_interface_power_flow_
    variables: TSO update sets prev := current then current := new; lambda_dso += rho_pf (x_dso - z_tso) / rating *
    s_base_dso), the value AA wrote back at the end of cycle k is recovered exactly for z and to round-off for lambda:
        z_written(k)      = z_tso_prev(k+1)
        lambda_written(k) = lambda_dso(k+1) - rho_pf(k+1) (x_dso(k+1) - z_tso_current(k+1)) / rating * s_base_dso
    and the AA extrapolation on PF is (z_written(k) - z_tso_current(k), lambda_written(k) - lambda_dso(k)). It is
    expressed in AA's own normalisation (admm_anderson_acceleration.collect_w: z / interface_rating;
    lambda / s_base_dso / rho_pf). SELF-CHECK: on every cycle whose AA action is NOT 'accepted' the z part must be
    exactly 0 and the lambda part at round-off; the script records the check per cycle and raises if it fails. The V
    and ESS parts of the write-back are NOT recoverable (no per-entry V capture; the ESS stride holds z and x only,
    no duals) -- stated as omissions.

Constraints: records only; standard library plus p513_solve_profile_guard (armed with permitted=() for the whole run,
verify(0) checked at write and on exit -- the guard is the only non-stdlib import and imports pyomo.opt; no
production module, no harness, no model, no pickle); JSONL streamed line by line; every input hashed at read time;
outputs written only to a NEW directory (refuses to overwrite).

Run: /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s53_w97_consolidated_diagnostics.py
"""
import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W97 consolidated records-only diagnostics (never solves)').install()

PAIR_REL = 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair'
CELLS = {
    'x0': {'eval_rel': f'{PAIR_REL}/evals/f6e9cd53fdbb8ee8_x0', 'cert_cycle': 72, 'tail_first': 64,
           'expected_Q': 842832534.7623764, 'w_json': 'data/SRP1/Results/P515S53/w95_x0_drift/w95_x0_drift_diagnostics.json',
           'w_manifest': 'data/SRP1/Results/P515S53/w95_x0_drift/manifest_sha256.json', 'w_label': 'W95 (d757394e)',
           'w_context_key': 'supplementary_context_cycles_53_72'},
    'n7_4h_e1': {'eval_rel': f'{PAIR_REL}/evals/c82522f470b35b58_n7_4h_e1', 'cert_cycle': 69, 'tail_first': 61,
                 'expected_Q': 842595839.5131028,
                 'w_json': 'data/SRP1/Results/P515S53/w96_n7_drift/w96_n7_drift_diagnostics.json',
                 'w_manifest': 'data/SRP1/Results/P515S53/w96_n7_drift/manifest_sha256.json',
                 'w_label': 'W96 (c2d4b4ce)', 'w_context_key': 'supplementary_context_cycles_50_69'},
}
ALIGN_FIRST = 55                 # Addendum 51: "around cycles 55-72"
RUN_CONTEXT_FIRST = 40           # rho / AA event history reported from here (and the whole run summarised)
CHANNELS = ('v', 'pf', 'ess')
LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
OUT_REL_DEFAULT = 'data/SRP1/Results/P515S53/w97_diagnostics'
FORBIDDEN_MODULES = ('shared_resources_planning', 'network', 'network_data', 'model_construction_helpers',
                     'p515_s44_campaign_harness', 'uncoordinated_benchmark', 'p515_g_g1_g4_admm_gates',
                     'p515_s53_w90_3x3_campaign', 'energy_storage', 'helper_functions', 'admm_anderson_acceleration',
                     'numpy')
LAMBDA_ROUNDOFF_REL = 1e-9       # lambda part of a non-accepted cycle's "extrapolation" must be below this x ||lambda||

INPUTS = {}


# ======================================================================================================================
#  tracked reads (W96's pattern)
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


def iter_jsonl(path, note):
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            if raw.strip():
                yield json.loads(raw)
    _register(path, h.hexdigest(), size, note)


def iter_jsonl_raw(path, note):
    """Streams (raw_bytes) lines, hashing all; the caller decides which lines to parse (large sidecars)."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            yield raw
    _register(path, h.hexdigest(), size, note)


def load_json(path, note, limit=2 * 1024 * 1024):
    size = os.path.getsize(path)
    if size > limit:
        raise RuntimeError(f'{path} is {size} bytes; this script loads only small JSON documents whole')
    with open(path, 'rb') as handle:
        data = handle.read()
    _register(path, hashlib.sha256(data).hexdigest(), size, note)
    return json.loads(data)


def iter_text_lines(path, note):
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            yield raw.decode('utf-8', errors='replace')
    _register(path, h.hexdigest(), size, note)


def _l2(values):
    return math.sqrt(sum(v * v for v in values))


# ======================================================================================================================
#  Part 1 -- consolidation of W95 / W96 items (i)-(v)
# ======================================================================================================================
def consolidate(label, cfg):
    wj_path = os.path.join(REPO, cfg['w_json'])
    wm_path = os.path.join(REPO, cfg['w_manifest'])
    wm = load_json(wm_path, f'{cfg["w_label"]} manifest (committed) -- verifies the output read below')
    w = load_json(wj_path, f'{cfg["w_label"]} diagnostics output (committed) -- items (i)-(v) consolidated, not re-derived')
    recorded_out = wm['outputs'][cfg['w_json']]['sha256']
    if INPUTS[cfg['w_json']]['sha256'] != recorded_out:
        raise RuntimeError(f'{cfg["w_json"]} does not match its committed manifest hash')
    window = w['window_cycles']
    ci, cii, ciii, civ, cv = (w['C_i_per_block_dQ'], w['C_ii_consensus_and_dual_movement'], w['C_iii_residual_margins'],
                              w['C_iv_row18_terms'], w['C_v_iterations_and_mu'])
    rows = {}
    for k in window:
        s = str(k)
        a, b, c, d, e = ci['per_cycle'][s], cii['per_cycle'][s], ciii['per_cycle'][s], civ['per_cycle'][s], cv['per_cycle'][s]
        rows[s] = {
            'dQ_gross': a['dQ_from_per_cycle_record'],
            'i_top10_signed_share_of_dQ': a['top10_signed_share_of_dQ'],
            'i_remainder_70_unlisted_share_of_dQ': a['remainder_share_of_dQ'],
            'i_min_unlisted_blocks_contributing': a['min_unlisted_blocks_contributing'],
            'i_top10_composition_by_agent': a['top10_composition_by_agent'],
            'i_top10_sign_counts': a['top10_sign_counts'],
            'i_cancellation_ratio_bounds_abs_dQ_over_sum_abs': a['cancellation_ratio_bounds_abs_dQ_over_sum_abs'],
            'ii_aa_action': b['aa_action'],
            'ii_dz_tso_l2_v_eq_s_over_rho': b['v']['dz_tso_l2_normalized_eq_s_over_rho'],
            'ii_dz_tso_l2_pf_eq_s_over_rho': b['pf']['dz_tso_l2_normalized_eq_s_over_rho'],
            'ii_dz_tso_l2_ess_eq_s_over_rho': b['ess']['dz_tso_l2_normalized_eq_s_over_rho'],
            'ii_dy_dso_l2_v_eq_rho_r': b['v']['dy_dso_l2_normalized_eq_rho_times_r'],
            'ii_dy_dso_l2_pf_eq_rho_r': b['pf']['dy_dso_l2_normalized_eq_rho_times_r'],
            'ii_norm_y_change_v': b['v']['norm_y_change_vs_previous_cycle'],
            'ii_norm_y_change_pf': b['pf']['norm_y_change_vs_previous_cycle'],
            'ii_norm_y_change_ess': b['ess']['norm_y_change_vs_previous_cycle'],
            'ii_tso_prox_max_normalized': ({x: b['tso_prox_inf_norm_movement'][f'{x}_max_normalized'] for x in CHANNELS}
                                           if b.get('tso_prox_inf_norm_movement') else None),
            'iii_primal_dual_ratios': {ch: [c[ch]['primal_ratio'], c[ch]['dual_ratio']] for ch in CHANNELS},
            'iii_binding': c['binding'],
            'iii_objective_step_ratio': c['objective_step_ratio'],
            'iv_delta_row18_charge_weighted': d['delta_row18_charge_weighted'],
            'iv_row18_share_of_dQ': d['share_of_dQ_row18_charge'],
            'iv_delta_covariance_dso_weighted': d['delta_covariance_dso_weighted'],
            'iv_delta_E_abs_d_p_mwh_weighted': d['delta_E_abs_d_p_mwh_weighted'],
            'iv_delta_sum_omega_d2_p_mw2h_weighted': d['delta_sum_omega_d2_p_mw2h_weighted'],
            'v_tail_active': e['tail_active'],
            'v_compl_inf_tol_in_force': e['compl_inf_tol_in_force'],
            'v_iterations_median_all_tso_dso': [e['iterations_all']['median'], e['iterations_tso']['median'],
                                                e['iterations_dso']['median']],
            'v_iterations_max_all': e['iterations_all']['max'],
            'v_mu_over_floor_median_min_max': [e['mu_over_floor_all']['median'], e['mu_over_floor_all']['min'],
                                               e['mu_over_floor_all']['max']],
            'v_floor_status': e['floor_status'],
            'v_exits': e['exits'],
            'v_cycle_wall_s': e['cycle_wall_s'],
        }
    omissions = {
        'i_per_block_dQ': ci['not_recoverable_from_records'],
        'ii_consensus_and_dual_movement': cii['not_recoverable_from_records'],
        'ii_derivation_caveat': cii['derived'],
        'iv_row18_terms': civ['not_recoverable_from_records'],
        'D_H_row18_part2': w['D_scoring_evidence']['H_row18_part2'],
    }
    return {
        'source': cfg['w_json'], 'source_label': cfg['w_label'], 'source_sha256': recorded_out,
        'window_cycles': window, 'certification_cycle': w['certification_cycle'],
        'what_each_item_records': {'i': ci['recorded'], 'ii': cii['recorded'], 'iii': ciii['recorded'],
                                   'iv': civ['recorded'], 'v': cv['recorded']},
        'omissions_carried_forward_verbatim': omissions,
        'row18_cumulative': civ['cumulative'],
        'blocks_recurring_in_window_top10': ci['blocks_recurring_in_window_top10'],
        'context_lead_blocks': w[cfg['w_context_key']]['lead_blocks'],
        'context_cycles_available': sorted(int(x) for x in w[cfg['w_context_key']]['per_cycle']),
        'per_cycle': rows,
    }


# ======================================================================================================================
#  Part 2 -- rho history, AA state, tail, Q steps, cycles 55..cert
# ======================================================================================================================
_RHO_BOYD_RE = re.compile(r'\[ADMM RHO BOYD\] (V|PF|ESS) \|.*\| rho_before=(\S+) \| rho_after=(\S+) \| '
                          r'gamma_before=(\S+) \| gamma_after=(\S+) \| action=(.*)$')
_TAIL_ON_RE = re.compile(r'Convergence-depth tail ON for cycle (\d+): compl_inf_tol=(\S+) on every')
_AA_WORDS_RE = re.compile(r'anderson|extrapolat|\[AA\]', re.I)


def pf_extrapolation(eval_dir, cycles_needed, aa):
    """Derive the PF part of every AA write-back and the plain PF step, from pf_entry_stride (module docstring)."""
    path = os.path.join(eval_dir, 'pf_entry_stride_s39_D.jsonl')
    lines = {}
    n_lines = 0
    for raw in iter_jsonl_raw(path, 'PF per-entry sidecar (captured inside get_admm_boyd_residual_metrics, pre-AA): '
                                    'x_dso, z_tso current/prev, lambda_dso, rho_pf per entry -- AA PF write-back derived'):
        n_lines += 1
        head = raw[:64].decode('ascii', errors='replace')
        m = re.match(r'\{"cycle": (\d+),', head)
        if not m:
            raise RuntimeError(f'unexpected PF stride line head: {head!r}')
        c = int(m.group(1))
        if c in cycles_needed:
            rec = json.loads(raw)
            if not rec['identity_holds'] or rec['stride'] != 1:
                raise RuntimeError(f'PF stride cycle {c}: identity_holds={rec["identity_holds"]} stride={rec["stride"]}')
            lines[c] = {(e['node_id'], e['year'], e['day'], e['power_type'], e['period']): e for e in rec['entries']}
    out = {}
    for k in sorted(cycles_needed):
        if k not in lines or (k + 1) not in lines:
            continue
        cur, nxt = lines[k], lines[k + 1]
        if set(cur) != set(nxt):
            raise RuntimeError(f'PF stride entry sets differ between cycles {k} and {k + 1}')
        dz_aa, du_aa, gz, gu, lam_scaled = [], [], [], [], []
        max_abs_dz_aa, max_abs_du_aa, argmax = 0.0, 0.0, None
        rho_set = set()
        for key, e in cur.items():
            n = nxt[key]
            rating, sb, rho_k, rho_n = e['interface_rating'], e['s_base_dso'], e['rho_pf'], n['rho_pf']
            rho_set.update((rho_k, rho_n))
            # plain step of cycle k (AA normalisation): z part (z_cur - z_prev)/rating; u part = r entry
            gz.append((e['z_tso_current'] - e['z_tso_prev']) / rating)
            gu.append((e['x_dso'] - e['z_tso_current']) / rating)
            # what AA wrote back at the end of cycle k
            z_written = n['z_tso_prev']
            lam_written = n['lambda_dso'] - rho_n * (n['x_dso'] - n['z_tso_current']) / rating * sb
            dz = (z_written - e['z_tso_current']) / rating
            du = (lam_written - e['lambda_dso']) / sb / rho_k
            dz_aa.append(dz)
            du_aa.append(du)
            lam_scaled.append(e['lambda_dso'] / sb / rho_k)
            if abs(dz) > max_abs_dz_aa:
                max_abs_dz_aa, argmax = abs(dz), key
            max_abs_du_aa = max(max_abs_du_aa, abs(du))
        accepted = aa[k]['aa_action'] == 'accepted'
        n_z = _l2(dz_aa)
        n_u = _l2(du_aa)
        n_lam = _l2(lam_scaled)
        check = None
        if not accepted:
            check = {'z_part_exactly_zero': all(v == 0.0 for v in dz_aa),
                     'u_part_over_norm_u': (n_u / n_lam) if n_lam else n_u,
                     'u_part_at_roundoff': ((n_u / n_lam) if n_lam else n_u) <= LAMBDA_ROUNDOFF_REL}
            if not (check['z_part_exactly_zero'] and check['u_part_at_roundoff']):
                raise RuntimeError(f'PF write-back derivation self-check FAILED at cycle {k} (AA not accepted): {check}')
        n_gz, n_gu = _l2(gz), _l2(gu)
        out[k] = {
            'aa_action_end_of_cycle': aa[k]['aa_action'],
            'plain_step_pf_l2': {'z_part': n_gz, 'u_part': n_gu, 'total': math.hypot(n_gz, n_gu)},
            'aa_writeback_minus_plain_pf_l2': {'z_part': n_z, 'u_part': n_u, 'total': math.hypot(n_z, n_u)},
            'aa_writeback_over_plain_step_pf': (math.hypot(n_z, n_u) / math.hypot(n_gz, n_gu)
                                                if math.hypot(n_gz, n_gu) else None),
            'aa_writeback_max_abs_pf': {'z_part': max_abs_dz_aa, 'u_part': max_abs_du_aa,
                                        'z_argmax_entry': list(argmax) if argmax else None},
            'norm_u_pf_scaled_dual': n_lam,
            'rho_pf_values_seen_k_and_k_plus_1': sorted(rho_set),
            'n_entries': len(cur),
            'self_check_non_accepted': check,
        }
    return out, n_lines


def alignment(label, cfg):
    ev_dir = os.path.join(REPO, cfg['eval_rel'])
    cert = cfg['cert_cycle']
    ev = load_json(os.path.join(ev_dir, 'evaluation_record.json'), 'evaluation record (certification, AA configuration)')
    assert ev['certification_cycle'] == cert and ev['certified_cost'] == cfg['expected_Q'], label
    pcr = {r['cycle']: r for r in iter_jsonl(os.path.join(ev_dir, 'per_cycle_record.jsonl'),
                                             'per-cycle record: gross_operational_cost per cycle')}
    g = load_json(os.path.join(ev_dir, 'g_s39_D.json'), 'stage report: cycle_trajectory (rho/gamma/freeze/Boyd per cycle)')
    traj = {r['cycle']: r for r in g['cycle_trajectory']}
    aa = {r['cycle']: r for r in iter_jsonl(os.path.join(ev_dir, 'aa_per_cycle.jsonl'), 'Anderson acceleration per cycle')}
    ts = load_json(os.path.join(ev_dir, 'convergence_depth_tail_state.json'), 'convergence-depth tail state per cycle')
    tail = {r['cycle']: r for r in ts['per_cycle']}
    assert sorted(pcr) == sorted(traj) == sorted(aa) == sorted(tail) == list(range(1, cert + 1)), label
    for k in pcr:
        assert pcr[k]['gross_operational_cost'] == traj[k]['gross_operational_cost'], (label, k)
        for f in ('aa_action', 'aa_accepted', 'aa_memory_size_before', 'aa_memory_size_after', 'aa_combined_residual'):
            assert traj[k][f] == aa[k][f], (label, k, f)
    Q = {k: pcr[k]['gross_operational_cost'] for k in pcr}
    dQ = {k: Q[k] - Q[k - 1] for k in range(2, cert + 1)}

    # ---- child stdout: rho lines, tail lines, any AA printing
    rho_lines, tail_lines, aa_word_lines = [], [], []
    for line in iter_text_lines(os.path.join(ev_dir, 'child_stdout.log'),
                                'child stdout: [ADMM RHO BOYD] per channel per cycle; tail ON lines; searched for AA lines'):
        m = _RHO_BOYD_RE.search(line)
        if m:
            rho_lines.append(m.groups())
            continue
        m = _TAIL_ON_RE.search(line)
        if m:
            tail_lines.append((int(m.group(1)), float(m.group(2))))
            continue
        if _AA_WORDS_RE.search(line) and 'convergence-depth tail' not in line.lower():
            aa_word_lines.append(line.strip()[:200])
    aa_word_lines_production_stdout = []
    for line in iter_text_lines(os.path.join(ev_dir, 'stdout_s39_D.log'),
                                'production stdout (stage capture): searched for AA lines'):
        if _AA_WORDS_RE.search(line) and 'convergence-depth tail' not in line.lower():
            aa_word_lines_production_stdout.append(line.strip()[:200])
    if len(rho_lines) != 3 * cert:
        raise RuntimeError(f'{label}: {len(rho_lines)} [ADMM RHO BOYD] lines, expected {3 * cert}')
    rho_xcheck_fail = []
    for i, (ch, rb, ra, gb, ga, act) in enumerate(rho_lines):
        k = i // 3 + 1
        c = ch.lower()
        assert c == CHANNELS[i % 3], (label, i, ch)
        t = traj[k]
        ok = (f'{t[f"rho_{c}_before"]:.6e}' == rb and f'{t[f"rho_{c}_after"]:.6e}' == ra
              and f'{t[f"gamma_{c}_before"]:.6e}' == gb and f'{t[f"gamma_{c}_after"]:.6e}' == ga
              and t[f'rho_{c}_action'] == act.strip())
        if not ok:
            rho_xcheck_fail.append({'cycle': k, 'channel': c, 'stdout': [rb, ra, gb, ga, act.strip()]})
    tail_active_cycles = [k for k in sorted(tail) if tail[k]['active']]
    assert tail_active_cycles == list(range(cfg['tail_first'], cert + 1)), (label, tail_active_cycles)
    tail_stdout_cycles = [k for k, _ in tail_lines]

    # ---- whole-run rho events and freeze onsets
    rho_changes = []
    for k in sorted(traj):
        t = traj[k]
        for c in CHANNELS:
            if t[f'rho_{c}_after'] != t[f'rho_{c}_before']:
                rho_changes.append({'cycle': k, 'channel': c, 'before': t[f'rho_{c}_before'], 'after': t[f'rho_{c}_after'],
                                    'factor': t[f'rho_{c}_after'] / t[f'rho_{c}_before'], 'action': t[f'rho_{c}_action']})
    first_frozen = {c: next((k for k in sorted(traj) if traj[k][f'rho_frozen_{c}']), None) for c in CHANNELS}
    first_freeze_active = next((k for k in sorted(traj) if traj[k]['rho_freeze_active']), None)
    last_change = {c: max((e['cycle'] for e in rho_changes if e['channel'] == c), default=None) for c in CHANNELS}
    gamma_nonzero = [(k, c) for k in traj for c in CHANNELS
                     if traj[k][f'gamma_{c}_before'] != 0.0 or traj[k][f'gamma_{c}_after'] != 0.0]

    # ---- whole-run AA events
    accepted_cycles = [k for k in sorted(aa) if aa[k]['aa_action'] == 'accepted']
    off_cycles = [k for k in sorted(aa) if aa[k]['aa_action'] == 'off (all channels within Boyd tolerance)']
    first_off = off_cycles[0] if off_cycles else None
    off_contiguous_to_end = off_cycles == list(range(first_off, cert + 1)) if first_off else False
    last_accepted_before_off = max((k for k in accepted_cycles if first_off is None or k < first_off), default=None)
    aa_action_counts = {}
    for k in aa:
        aa_action_counts[aa[k]['aa_action']] = aa_action_counts.get(aa[k]['aa_action'], 0) + 1
    resets = [{'cycle': k, 'reason': aa[k]['aa_reset_reason'], 'rho_changed_channels': aa[k]['aa_rho_changed_channels'],
               'memory_after': aa[k]['aa_memory_size_after']} for k in sorted(aa)
              if aa[k]['aa_reset'] or aa[k]['aa_rho_changed_channels']]

    # ---- PF write-back derivation, cycles ALIGN_FIRST-1 .. cert
    need = set(range(RUN_CONTEXT_FIRST, cert + 1))
    pfx, n_pf_lines = pf_extrapolation(ev_dir, need, aa)
    if n_pf_lines != cert:
        raise RuntimeError(f'{label}: PF stride has {n_pf_lines} lines, expected {cert}')

    # ---- ESS stride: record what it holds (no duals)
    ess_keys, ess_lines = set(), 0
    for raw in iter_jsonl_raw(os.path.join(ev_dir, 'ess_entry_stride_baseline.jsonl'),
                              'ESS per-entry sidecar -- inspected for dual fields (none): ESS AA write-back not recoverable'):
        ess_lines += 1
        if ess_lines == ALIGN_FIRST:
            rec = json.loads(raw)
            for e in rec['entries']:
                ess_keys.update(e.keys())
                ess_keys.update(f'x.{a}' for a in e['x'])

    # ---- per-cycle alignment table
    rows = {}
    for k in range(ALIGN_FIRST, cert + 1):
        t, a = traj[k], aa[k]
        prev_acc = aa[k - 1]['aa_action'] == 'accepted'
        rows[str(k)] = {
            'Q_gross': Q[k], 'dQ_gross': dQ[k], 'dQ_increment_vs_previous_step': dQ[k] - dQ[k - 1],
            'state_entering_this_cycle': ('AA-extrapolated (accepted at end of cycle %d)' % (k - 1)) if prev_acc
                                         else 'plain ADMM iterate',
            'tail_active_this_cycle': tail[k]['active'],
            'tail_compl_inf_tol_TSO_DSO': sorted({h['after'] for h in tail[k]['holders'].values()}, key=str),
            'aa_action_end_of_cycle': a['aa_action'], 'aa_accepted': a['aa_accepted'],
            'aa_boyd_all_pass': a['aa_boyd_all_pass'],
            'aa_memory_size_before_after': [a['aa_memory_size_before'], a['aa_memory_size_after']],
            'aa_gamma_columns_m_k': a['aa_gamma_columns'],
            'aa_combined_residual': a['aa_combined_residual'],
            'aa_safeguard_mark_before': a['aa_baseline_residual_before'],
            'aa_reset': a['aa_reset'], 'aa_reset_reason': a['aa_reset_reason'],
            'aa_rho_changed_channels': a['aa_rho_changed_channels'],
            'consecutive_converged_cycles': t['consecutive_converged_cycles'],
            'rho_before': {c: t[f'rho_{c}_before'] for c in CHANNELS},
            'rho_after': {c: t[f'rho_{c}_after'] for c in CHANNELS},
            'rho_action': {c: t[f'rho_{c}_action'] for c in CHANNELS},
            'rho_frozen': {c: t[f'rho_frozen_{c}'] for c in CHANNELS},
            'rho_unchanged_streak': {c: t[f'rho_unchanged_streak_{c}'] for c in CHANNELS},
            'rho_at_clamp': {c: t[f'rho_at_clamp_{c}'] for c in CHANNELS},
            'balancing_exempt': {c: t[f'balancing_exempt_{c}'] for c in CHANNELS},
            'rho_freeze_active': t['rho_freeze_active'],
            'gamma_before_after': {c: [t[f'gamma_{c}_before'], t[f'gamma_{c}_after']] for c in CHANNELS},
            'boyd_ratios_primal_dual': {c: [t[f'boyd_{c}_primal_ratio'], t[f'boyd_{c}_dual_ratio']] for c in CHANNELS},
            'pf_aa_writeback_derived': pfx.get(k),
        }

    # ---- the question: where does the accelerating run begin?
    run_peak = cert                                  # terminal monotone run of strictly decreasing signed steps
    while run_peak - 1 in dQ and dQ[run_peak] < dQ[run_peak - 1]:
        run_peak -= 1                                # stops at k0: dQ_(k0+1) < dQ_k0 but not dQ_k0 < dQ_(k0-1)
    run_start = run_peak + 1
    neg_start = cert                                 # terminal run of negative steps
    while neg_start - 1 in dQ and dQ[neg_start - 1] < 0:
        neg_start -= 1
    earlier_desc = []                                # negative-step runs of >= 3 cycles between 40 and the AA switch-off
    k = RUN_CONTEXT_FIRST
    while k <= (first_off or cert):
        if dQ.get(k, 0) < 0:
            j = k
            while j + 1 <= (first_off or cert) and dQ.get(j + 1, 0) < 0:
                j += 1
            if j - k + 1 >= 3:
                steps = {str(x): dQ[x] for x in range(k, j + 1)}
                peak = min(range(k, j + 1), key=lambda x: dQ[x])
                earlier_desc.append({'first': k, 'last': j, 'sum': Q[j] - Q[k - 1], 'steps': steps,
                                     'largest_step_cycle': peak, 'largest_step': dQ[peak],
                                     'aa_actions_end_of_cycle': {str(x): aa[x]['aa_action'] for x in range(k, j + 1)},
                                     'rho_changes_inside': [e for e in rho_changes if k <= e['cycle'] <= j]})
            k = j + 1
        else:
            k += 1
    answer = {
        'definitions': {
            'aa_off_first_cycle': ('first cycle whose AA record action is "off (all channels within Boyd tolerance)"; '
                                   'decided at the END of that cycle; the state entering the NEXT cycle is plain'),
            'last_aa_extrapolation': ('last cycle before aa_off_first_cycle with action "accepted"; the extrapolated '
                                      'state is written back at the END of that cycle and is what the NEXT cycle solves '
                                      'from, so it first shows in Q at cycle last_accepted + 1'),
            'tail_first_cycle': 'first cycle with compl_inf_tol = 1e-6 in force (tail state active) = aa_off_first_cycle + 1 by construction',
            'accelerating_run_peak_cycle': ('first cycle of the terminal MONOTONE run of signed steps: the largest '
                                            'k0 with dQ_j < dQ_(j-1) for every j in k0+1..certification; dQ_k0 is the '
                                            'run\'s peak (least negative / most positive) step'),
            'accelerating_run_first_decreasing_cycle': 'k0 + 1 -- the first cycle whose step is below its predecessor\'s, the run continuing to certification',
            'negative_run_first_cycle': 'first cycle of the terminal run of negative steps',
        },
        'last_aa_extrapolation_cycle': last_accepted_before_off,
        'first_cycle_solving_from_last_extrapolated_state': (last_accepted_before_off + 1) if last_accepted_before_off else None,
        'aa_off_first_cycle': first_off, 'aa_off_contiguous_to_certification': off_contiguous_to_end,
        'first_cycle_solving_from_plain_state_after_last_extrapolation': (last_accepted_before_off + 2)
        if last_accepted_before_off else None,
        'tail_first_cycle': tail_active_cycles[0] if tail_active_cycles else None,
        'accelerating_run_peak_cycle': run_peak, 'accelerating_run_peak_step': dQ[run_peak],
        'accelerating_run_first_decreasing_cycle': run_start,
        'step_before_peak': {str(run_peak - 1): dQ.get(run_peak - 1)},
        'accelerating_run_steps': {str(x): dQ[x] for x in range(run_peak, cert + 1)},
        'accelerating_run_increments': {str(x): dQ[x] - dQ[x - 1] for x in range(run_start, cert + 1)},
        'negative_run_first_cycle': neg_start,
        'run_peak_minus_aa_off_first': run_peak - first_off if first_off else None,
        'run_peak_minus_tail_first': (run_peak - tail_active_cycles[0]) if tail_active_cycles else None,
        'run_peak_minus_first_plain_after_last_extrapolation':
            (run_peak - (last_accepted_before_off + 2)) if last_accepted_before_off else None,
        'negative_run_first_minus_aa_off_first': neg_start - first_off if first_off else None,
        'negative_run_first_minus_tail_first': (neg_start - tail_active_cycles[0]) if tail_active_cycles else None,
        'earlier_negative_step_runs_of_3plus_cycles_from_40_to_aa_off': earlier_desc,
    }

    return {
        'cell': cfg['eval_rel'], 'certification_cycle': cert, 'certified_Q_gross': Q[cert],
        'candidate_label': ev['candidate_label'], 'eval_key': ev['eval_key'], 'candidate_key': ev['candidate_key'],
        'aa_configuration_effective_in_child': ev.get('anderson_acceleration_effective_in_child'),
        'aa_policy_notes': ('admm_anderson_acceleration.py (read as source, not imported): type-II AA, memory 5, '
                            'Tikhonov 1e-10; reject_policy keep_memory => a safeguard rejection keeps the plain iterate '
                            'and the memory; memory is cleared only on a rho change or a local-solve failure; '
                            '"off" is a PER-CYCLE predicate (boyd_all_pass at the end of the cycle): no extrapolation, '
                            'the (w, g) pair still pushed; AA resumes the first cycle a channel leaves tolerance. '
                            'Safeguard: accept iff combined residual < the mark set at the last acceptance'),
        'rho_config': {'gamma_policy': traj[cert]['gamma_policy'], 'gamma_tau': traj[cert]['gamma_tau'],
                       'freeze_after_unchanged_cycles': traj[cert]['freeze_after_unchanged_cycles'],
                       'freeze_backstop_cycle': traj[cert]['freeze_backstop_cycle'],
                       'freeze_after_cycle': traj[cert]['freeze_after_cycle']},
        'rho_history_whole_run': {
            'changes': rho_changes, 'last_change_cycle': last_change, 'first_cycle_rho_frozen': first_frozen,
            'first_cycle_rho_freeze_active_all_channels': first_freeze_active,
            'changes_in_alignment_window': [e for e in rho_changes if e['cycle'] >= ALIGN_FIRST],
            'gamma_nonzero_any_cycle': gamma_nonzero,
            'stdout_cross_check': {'n_lines': len(rho_lines), 'mismatches': rho_xcheck_fail,
                                   'precision': 'stdout prints %.6e; compared as formatted strings'},
        },
        'aa_history_whole_run': {'action_counts': aa_action_counts, 'accepted_cycles': accepted_cycles,
                                 'off_cycles': off_cycles, 'resets_and_rho_clears': resets,
                                 'aa_lines_in_child_stdout': aa_word_lines,
                                 'aa_lines_in_stdout_s39_D': aa_word_lines_production_stdout},
        'tail': {'active_cycles': tail_active_cycles, 'stdout_tail_on_cycles': tail_stdout_cycles,
                 'stdout_matches_state': tail_stdout_cycles == tail_active_cycles,
                 'tail_first_equals_aa_off_first_plus_1': (tail_active_cycles[0] == first_off + 1)
                 if (tail_active_cycles and first_off) else None},
        'ess_stride_fields_seen': sorted(ess_keys), 'ess_stride_lines': ess_lines,
        'alignment_per_cycle': rows,
        'pf_aa_writeback_context_before_window': {
            'note': (f'SUPPLEMENTARY: the same PF derivation for cycles {RUN_CONTEXT_FIRST}-{ALIGN_FIRST - 1} (the '
                     'earlier descent), with dQ and the AA action; self-check applied identically'),
            'per_cycle': {str(x): {'dQ_gross': dQ.get(x), 'aa_action_end_of_cycle': aa[x]['aa_action'],
                                   'rho_changed_channels_end_of_cycle': aa[x]['aa_rho_changed_channels'],
                                   'pf': pfx.get(x)} for x in range(RUN_CONTEXT_FIRST, ALIGN_FIRST)}},
        'pf_aa_writeback_accepted_cycles_summary': {
            str(x): {'total_l2': pfx[x]['aa_writeback_minus_plain_pf_l2']['total'],
                     'over_plain_step': pfx[x]['aa_writeback_over_plain_step_pf'],
                     'z_part_l2': pfx[x]['aa_writeback_minus_plain_pf_l2']['z_part'],
                     'u_part_l2': pfx[x]['aa_writeback_minus_plain_pf_l2']['u_part']}
            for x in sorted(pfx) if pfx[x]['aa_action_end_of_cycle'] == 'accepted'},
        'pf_self_check_cycles': {'n_non_accepted_checked': sum(1 for x in pfx if pfx[x]['self_check_non_accepted']),
                                 'all_pass': all(pfx[x]['self_check_non_accepted']['z_part_exactly_zero']
                                                 and pfx[x]['self_check_non_accepted']['u_part_at_roundoff']
                                                 for x in pfx if pfx[x]['self_check_non_accepted']),
                                 'max_u_part_over_norm_u': max((pfx[x]['self_check_non_accepted']['u_part_over_norm_u']
                                                                for x in pfx if pfx[x]['self_check_non_accepted']),
                                                               default=None)},
        'answer_inputs': answer,
    }


OMISSIONS_PART2 = {
    'aa_extrapolation_magnitude': (
        'NOT RECORDED by production: aa_per_cycle.jsonl / the trajectory hold action, accepted, memory size, gamma '
        'columns (= m_k, the number of secant columns), combined residual and the safeguard mark -- not ||w_hat - '
        'w_plain||, not the gamma* coefficients, not w. Searched: aa_per_cycle.jsonl, g_s39_D.json cycle_trajectory '
        '(aa_* fields), child_stdout.log and stdout_s39_D.log (no AA print exists in the loop; lines matching '
        'anderson/extrapolat/[AA] listed per cell), evaluation_record.json. The PF part is DERIVED here from '
        'pf_entry_stride_s39_D.jsonl (docstring; self-checked on every non-accepted cycle); the V part and the ESS part '
        'are NOT recoverable (no per-entry V capture exists; ess_entry_stride_baseline.jsonl holds z and x per agent, '
        'no duals -- fields listed per cell).'),
    'aa_gamma_coefficients': 'not recorded (only the column count m_k); searched as above',
    'rho_per_block': ('rho is recorded per channel (one scalar per channel, shared by every agent -- asserted by the AA '
                      'caller each cycle); no per-block rho exists to record'),
    'n7_cycles_70_72': 'the node-7 cell certified at cycle 69; cycles 70-72 do not exist on that cell',
    'cause_attribution': ('the records align events in time; they cannot separate an AA-off transient from the tail '
                          'switch on either cell, because the tail is ON at cycle k+1 exactly when AA is OFF at cycle k '
                          '(shared_resources_planning._convergence_depth_tail_next_state raises if they disagree)'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_REL_DEFAULT,
                    help='output directory relative to the repository (default: the W97 artifact path)')
    args = ap.parse_args()
    out_dir = os.path.join(REPO, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    out_json = os.path.join(out_dir, 'w97_consolidated_diagnostics.json')
    out_manifest = os.path.join(out_dir, 'manifest_sha256.json')
    if os.path.exists(LOCK):
        raise RuntimeError(f'campaign lock present ({LOCK}); a campaign may be running -- refusing')
    os.makedirs(out_dir, exist_ok=True)
    for path in (out_json, out_manifest):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    out = {'schema': 'p515_s53_w97_consolidated_v1',
           'task': 'W97 (PLANNER_BRIEF_2026-09-13.md Addendum 51, "Diagnostics owed")',
           'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (as W95/W96); dQ_k = Q_k - Q_(k-1); '
                                    'Q_k is evaluated on cycle k\'s local solves, which start from the state left at the '
                                    'end of cycle k-1 (plain or AA-extrapolated)'),
           'constraint': ('records only; no production module, no pickle, no model; SolveProfileGuard(permitted=()) '
                          'armed for the whole run'),
           'started_utc': datetime.now(timezone.utc).isoformat()}
    out['part1_consolidated_i_to_v'] = {label: consolidate(label, cfg) for label, cfg in CELLS.items()}
    out['part2_rho_aa_tail_alignment'] = {label: alignment(label, cfg) for label, cfg in CELLS.items()}
    out['part2_omissions'] = OMISSIONS_PART2
    loaded = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULES)
    if loaded:
        raise RuntimeError(f'forbidden modules imported: {loaded}')
    out['forbidden_modules_imported'] = loaded
    out['pyomo_modules_loaded_via_guard_only'] = sorted(m for m in sys.modules if m.split('.')[0] == 'pyomo')[:5]
    out['pickle_loads_performed'] = 0
    out['guard_at_write'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    if out['guard_at_write']['verify_0_failures']:
        raise RuntimeError(f"guard verify(0) failed before write: {out['guard_at_write']}")
    out['peak_rss_bytes_ru_maxrss_self'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['peak_rss_units'] = 'bytes (macOS ru_maxrss)'
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'x') as handle:
        json.dump(out, handle, indent=1, sort_keys=False)
        handle.write('\n')
    script = os.path.abspath(__file__)
    manifest = {
        'schema': 'p515_s53_w97_manifest_v1',
        'script': {'path': os.path.relpath(script, REPO), 'sha256': _hash_whole(script)},
        'guard_module': {'path': 'p513_solve_profile_guard.py',
                         'sha256': _hash_whole(os.path.join(REPO, 'p513_solve_profile_guard.py'))},
        'interpreter': sys.executable, 'python_version': sys.version,
        'inputs': INPUTS, 'n_inputs': len(INPUTS),
        'outputs': {os.path.relpath(out_json, REPO): {'sha256': _hash_whole(out_json),
                                                      'size_bytes': os.path.getsize(out_json)}},
        'note': ('pf_entry_stride_s39_D.jsonl and ess_entry_stride_baseline.jsonl are large and not committed (hash-'
                 'recorded in the pair manifests and here)'),
    }
    with open(out_manifest, 'x') as handle:
        json.dump(manifest, handle, indent=1)
        handle.write('\n')
    print(f'wrote {os.path.relpath(out_json, REPO)} and {os.path.relpath(out_manifest, REPO)}; '
          f'{len(INPUTS)} inputs hashed; peak RSS {out["peak_rss_bytes_ru_maxrss_self"] / 2**20:.1f} MiB')


if __name__ == '__main__':
    try:
        main()
    finally:
        GUARD.uninstall()
        failures = GUARD.verify(0)
        print(f'[W97] guard counts {GUARD.counts} verify(0) -> {failures}', flush=True)
        if failures:
            raise SystemExit(f'guard verify(0) failed: {failures}')
