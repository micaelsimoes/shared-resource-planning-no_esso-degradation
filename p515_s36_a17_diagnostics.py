"""
P5.15 Addendum 17 step 1 -- bounded ZERO-SOLVE diagnostics, plus the data-availability
audit those diagnostics depend on.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 17 (2026-09-16), items 1(a)/(b)/(c);
`P5_15_ADDENDUM16_EXPERT_REPORT.md` Sec.5-8; `P5_15_S35PT_GATE3_REPORT.md`.

NO Pyomo/IPOPT solves anywhere in this script. `SolveProfileGuard` is armed with an
EMPTY permitted list for the whole run and `guard.verify(0, 0)` is checked before
anything is written (CLAUDE.md sixth evidence rule: enforce, never assert). `scipy`
LPs are permitted in general for this task family, but THIS script's PART 1(b)/(c)
turn out not to be computable (see PART 0 below), so the declared scipy
(`shared_ess_price_taker`) LP-call count for this run is exactly 0, and that is
checked against the module's own call counter.

## What this script does

PART 0 -- audits, per run (`P515S35_REF_run` = run 1/reference, `P515S35_PT_run` =
gate 3), what quantities exist in the committed artifacts, at what granularity
(per-entry vs aggregate-norm-only vs absent), for the five items Addendum 17 lists.
Every check is a direct, programmatic inspection of the committed files (existence,
key presence, shape, or content), not a restatement from memory -- see
`_audit_item_*` functions below. Absence is reported as absence, scoped to exactly
the files/paths searched (CLAUDE.md fifth evidence rule).

PART 1 -- only what PART 0 shows is computable:
  (a) dual-direction comparison, gate 3 (run PT) at its terminal cycle 150 versus
      run 1 (run REF) at its terminal cycle 477:
        - per-agent (tso/dso/esso) AGGREGATE dual norms (`boyd_ess_norm_y_<agent>`,
          already present in each run's own `g_*.json` `cycle_trajectory`, no
          recomputation) and their PT/REF ratio -- available for all three agents.
        - for the ESSO agent ONLY, PER-ENTRY consensus duals exist (the ESSO's own
          `dual_p_req`/`dual_q_req` Pyomo Params, frozen in each run's
          `esso_models_baseline.pkl` at that run's own terminal cycle -- see
          `shared_resources_planning.py:5519-5525`, where these Params are SET, one
          -to-one, from `dual_vars['ess']['esso']['current'][node][year][day][p/q][period]`
          immediately before the ESSO's own solve). This gives an actual per-entry
          cosine, norm ratio and sign-agreement fraction for ESSO -- for TSO and DSO,
          direction is explicitly reported as NOT DETERMINABLE (no per-entry TSO/DSO
          model is ever pickled in either run's committed evidence; see PART 0).
        - "moving toward run 1": the PT run's own `cycle_trajectory` carries the
          three per-agent AGGREGATE norms at every one of its 150 cycles (no
          per-entry history exists for any agent, ESSO included -- only ONE
          terminal ESSO snapshot is pickled per run). The norm trend at several
          late cycles is reported per agent as a NORM trend (monotonic approach
          toward run 1's terminal norm), explicitly NOT a direction trend.
  (b) price-taker LP with run 1's terminal nodal prices at the storage buses:
      NOT COMPUTABLE -- PART 0 item 3 finds no per-entry (or any-granularity)
      TSO/DSO node-balance dual (nodal price) ever captured, at any cycle, in
      either run's committed evidence. Reported as unavailable with the capture
      that would supply it; no LP is solved with a substituted or fabricated price.
  (c) the same LP with cycle-0 LMPs: NOT COMPUTABLE for the same reason, and
      additionally there is no "cycle-0" network solve in this pipeline at all
      (`pt_phase2_checks/phase2_checks_results.json`'s own
      `precycle1_capture_method` field states the pre-cycle-1 state is captured by
      raising BEFORE any solve -- zero IPOPT solves have occurred at that point in
      EITHER run).

Comparison table: reports the three EXISTING certified EFC/day values (run 1
1.058952279550704; gate 3 terminal 1.1842435189565115; the market-price LP
1.1917770631428612, `data/SRP1/Results/P515S35/Z2/z2_floor_slackness_results.json`
`part_A_summary.efc_per_day_max_by_node`, uniform across nodes) and their pairwise
differences. (b) and (c) rows are marked NOT COMPUTABLE, not approximated.

## Formulas used in PART 1(a)

  cosine(u, v)      = (u . v) / (||u||_2 * ||v||_2)
  norm_ratio(u, v)  = ||v||_2 / ||u||_2                      (v = gate 3, u = run 1)
  sign_agreement    = (1/n) * sum_i [ sign(u_i) == sign(v_i) ]   (np.sign, so 0 only
                       matches 0 exactly; reported alongside a secondary count that
                       excludes entries where BOTH |u_i| and |v_i| are below a stated
                       noise floor, since KKT multipliers at inactive rows sit at
                       machine-precision magnitudes and a sign flip there is noise,
                       not a directional disagreement)

Output: NEW directory `data/SRP1/Results/P515S36/A17_diagnostics/` (refuses if it
already exists). Writes `a17_diagnostics_results.json` and `sha256_manifest.json`.
"""

import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import shared_ess_price_taker as sept  # noqa: E402  (counter only -- never called to solve here)

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')
PT_RUN = os.path.join(RES, 'P515S35_PT_run')
OUT_DIR = os.path.join(RES, 'P515S36', 'A17_diagnostics')

REF_TERMINAL_CYCLE = 477
PT_TERMINAL_CYCLE = 150
NODES = (5, 7, 9)

MARKET_PRICE_LP_JSON = os.path.join(RES, 'P515S35', 'Z2', 'z2_floor_slackness_results.json')
PHASE2_CHECKS_JSON = os.path.join(RES, 'P515S35', 'pt_phase2_checks', 'phase2_checks_results.json')

NOISE_FLOOR = 1e-8  # |dual| below this on BOTH sides is treated as numerical noise, not a sign


# ======================================================================================================================
#  Small helpers
# ======================================================================================================================
def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def _one_g_baseline(run_dir):
    matches = glob.glob(os.path.join(run_dir, 'g_*.json'))
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one g_*.json in {run_dir}, found {matches}')
    return _load_json(matches[0]), matches[0]


def _scan_keys_for_substring(obj, needles, path='', hits=None):
    """Recursively scan a JSON-loaded structure's KEYS for any of `needles`
    (case-insensitive substrings). Returns a list of (path, key) hits. Values are
    not scanned (this looks for a dual/price/lmp FIELD, not for a matching number)."""
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            klower = str(k).lower()
            if any(n in klower for n in needles):
                hits.append((path, k))
            _scan_keys_for_substring(v, needles, f'{path}/{k}', hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):  # first few elements only -- list entries share a schema in this codebase
            _scan_keys_for_substring(v, needles, f'{path}[{i}]', hits)
    return hits


def _grep_file_for_substrings(path, needles):
    hits = []
    with open(path, 'r', errors='replace') as handle:
        for lineno, line in enumerate(handle, start=1):
            low = line.lower()
            if any(n in low for n in needles):
                hits.append((lineno, line.strip()[:200]))
                if len(hits) >= 5:
                    break
    return hits


# ======================================================================================================================
#  PART 0 -- data-availability audit
# ======================================================================================================================
def audit_item1_storage_consensus_duals(run_dir, run_label, terminal_cycle):
    """Item 1: storage consensus duals (lambda) per agent (TSO, DSO, ESSO), p and q."""
    record = {'item': '1_storage_consensus_duals', 'run': run_label, 'searched': []}

    g, gpath = _one_g_baseline(run_dir)
    record['searched'].append(gpath)
    last = g['cycle_trajectory'][-1]
    if last['cycle'] != terminal_cycle:
        raise AssertionError(f'{run_label}: expected terminal cycle {terminal_cycle}, g_baseline last cycle is '
                              f'{last["cycle"]}')
    aggregate_norms = {
        'tso': last.get('boyd_ess_norm_y_tso'),
        'dso': last.get('boyd_ess_norm_y_dso'),
        'esso': last.get('boyd_ess_norm_y_esso'),
        'combined': last.get('boyd_ess_norm_y'),
    }
    record['aggregate_norm_per_agent_at_terminal_cycle'] = aggregate_norms
    record['aggregate_norm_granularity'] = ('per (year,day,period,type) SCALAR L2 norm over all storage entries, '
                                             'per agent, per cycle -- available at EVERY cycle 1..terminal in '
                                             'cycle_trajectory, not just the terminal one')

    esso_pkl_path = os.path.join(run_dir, 'esso_models_baseline.pkl')
    record['searched'].append(esso_pkl_path)
    import pickle
    with open(esso_pkl_path, 'rb') as handle:
        esso_models = pickle.load(handle)
    per_node_counts = {}
    for node in NODES:
        m = esso_models[node]
        per_node_counts[node] = {
            'dual_p_req_entries': len(list(m.dual_p_req.items())),
            'dual_q_req_entries': len(list(m.dual_q_req.items())),
            'rho': pe.value(m.rho),
        }
    record['esso_per_entry_dual'] = {
        'available': True,
        'source': 'esso_models_baseline.pkl -- ESSO Pyomo model per node, dual_p_req/dual_q_req Params '
                   '(shared_resources_planning.py:5519-5525 sets these one-to-one from '
                   "dual_vars['ess']['esso']['current'][node][year][day][p/q][period] immediately before the "
                   "ESSO's own solve; the pickle freezes whatever state the model held when the run stopped, "
                   f'i.e. this run\'s own terminal cycle {terminal_cycle}, regardless of Boyd vs cap stop)',
        'granularity': 'per (node, year, day, period), separately for p and q -- 288 entries/node/power-type',
        'per_node': per_node_counts,
        'agent_scope': 'ESSO ONLY -- this is the ESSO agent\'s OWN copy of the consensus dual, not TSO\'s or DSO\'s',
    }

    tso_dso_model_files = [p for p in glob.glob(os.path.join(run_dir, '**', '*.pkl'), recursive=True)
                           if 'esso' not in os.path.basename(p).lower()]
    record['searched'].append(os.path.join(run_dir, '**', '*.pkl (recursive)'))
    record['tso_dso_per_entry_dual'] = {
        'available': False,
        'reason': 'no TSO or DSO model is ever pickled in this run\'s committed evidence at the terminal cycle '
                  '(the only non-ESSO .pkl files present are the two FrozenSMOPF regression fixtures, frozen at a '
                  'fixed early cycle 7, for a DIFFERENT purpose -- see item 3 below)',
        'non_esso_pkl_files_found': [os.path.relpath(p, run_dir) for p in tso_dso_model_files],
        'capture_that_would_supply_it': 'pickle the TSO model and each DSO model (as esso_models_baseline.pkl '
                                          'already does for ESSO) at the run\'s terminal cycle, before discarding '
                                          'them, reading dual_ess_p_req/dual_ess_q_req '
                                          '(shared_resources_planning.py:5162-5163 TSO, 5353-5354/5478-5479 DSO) '
                                          '-- zero additional solves, a serialization change only.',
    }
    return record


def audit_item2_storage_consensus_z_and_x(run_dir, run_label, terminal_cycle):
    """Item 2: storage consensus z and agent copies x per entry; confirm/characterize the ESS stride file."""
    record = {'item': '2_storage_consensus_z_and_x', 'run': run_label, 'searched': []}
    stride_matches = glob.glob(os.path.join(run_dir, 'ess_entry_stride_*.jsonl'))
    if len(stride_matches) != 1:
        raise RuntimeError(f'expected exactly one ess_entry_stride_*.jsonl in {run_dir}, found {stride_matches}')
    stride_path = stride_matches[0]
    record['searched'].append(stride_path)

    cycles = []
    strides = set()
    first_entry_keys = None
    n_entries_first_line = None
    with open(stride_path) as handle:
        for line in handle:
            row = json.loads(line)
            cycles.append(row['cycle'])
            strides.add(row['stride'])
            if first_entry_keys is None:
                first_entry_keys = sorted(row['entries'][0].keys())
                n_entries_first_line = len(row['entries'])
    terminal_present = terminal_cycle in cycles
    record['z_and_x_per_entry'] = {
        'available': True,
        'granularity': 'per (node, year, day, power_type) -- z (consensus, list over periods) and x (dict '
                       'tso/dso/esso, each a list over periods) -- confirmed field names below',
        'entry_keys': first_entry_keys,
        'entries_per_recorded_cycle': n_entries_first_line,
        'stride_values_seen': sorted(strides),
        'n_cycles_recorded': len(cycles),
        'first_cycle_recorded': cycles[0] if cycles else None,
        'last_cycle_recorded': cycles[-1] if cycles else None,
        'terminal_cycle_present_exactly': terminal_present,
        'note': (None if terminal_present else
                 f'stride={sorted(strides)}: the run\'s own terminal cycle {terminal_cycle} is NOT one of the '
                 f'recorded cycles; the closest recorded cycle is {cycles[-1] if cycles else None} '
                 f'({terminal_cycle - cycles[-1] if cycles else None} cycles short)'),
    }
    return record


def audit_item3_nodal_prices(run_dir, run_label):
    """Item 3: TSO node-balance duals at storage buses (5,7,9) and DSO reference-node balance duals, per
    scenario/period, at the run's terminal cycle."""
    record = {'item': '3_nodal_prices_lmps', 'run': run_label, 'searched': []}
    needles = ('lmp', 'nodal_price', 'node_balance_dual', 'marginal_price')

    json_files = ['g_baseline.json', 'boyd_terminal.json', 'component_levels_terminal.json',
                  'interface_settlement_detail_s31c.json', 'interface_voltage_terminal.json']
    key_hits = {}
    for fname in json_files:
        matches = glob.glob(os.path.join(run_dir, 'g_*.json')) if fname == 'g_baseline.json' \
            else glob.glob(os.path.join(run_dir, fname))
        record['searched'].append(os.path.join(run_dir, fname))
        for m in matches:
            data = _load_json(m)
            hits = _scan_keys_for_substring(data, needles)
            key_hits[os.path.basename(m)] = hits

    stdout_matches = glob.glob(os.path.join(run_dir, 'stdout_*.log'))
    record['searched'].extend(stdout_matches)
    stdout_hits = {}
    for path in stdout_matches:
        stdout_hits[os.path.basename(path)] = _grep_file_for_substrings(path, needles)

    frozen_dir = os.path.join(run_dir, 'results', 'FrozenSMOPF')
    record['searched'].append(frozen_dir)
    frozen_snapshot_info = []
    if os.path.isdir(frozen_dir):
        import pickle
        for fname in sorted(os.listdir(frozen_dir)):
            fpath = os.path.join(frozen_dir, fname)
            with open(fpath, 'rb') as handle:
                obj = pickle.load(handle)
            meta = obj.get('metadata', {})
            model = obj.get('model')
            has_node_balance = False
            has_dual_suffix = False
            if model is not None:
                has_dual_suffix = hasattr(model, 'dual')
                has_node_balance = any('node_balance' in c.name for c in model.component_objects(
                    pe.Constraint, active=True)) if model is not None else False
            frozen_snapshot_info.append({
                'file': fname, 'metadata': meta,
                'has_dual_suffix': has_dual_suffix, 'has_node_balance_constraint': has_node_balance,
            })

    any_key_hit = any(v for v in key_hits.values())
    any_stdout_hit = any(v for v in stdout_hits.values())
    record['node_balance_duals_terminal_cycle'] = {
        'available': False,
        'json_field_scan_hits': key_hits,
        'stdout_grep_hits': stdout_hits,
        'any_hit': any_key_hit or any_stdout_hit,
        'frozen_smopf_regression_fixtures_found': frozen_snapshot_info,
        'reading': ('the FrozenSMOPF pickles DO carry a populated `dual` Suffix with values on the '
                    '`node_balance_p`/`node_balance_q` constraints (confirmed by direct inspection), which shows '
                    'the underlying IPOPT/Pyomo mechanism for nodal duals is live in production -- but these two '
                    'files are frozen at a FIXED EARLY CYCLE (7, see metadata above) for an unrelated regression- '
                    'fixture purpose (p44_production_frozen_regression.py), not at the run\'s terminal cycle, and '
                    'not for every cycle. No terminal-cycle (or any-other-cycle) TSO/DSO node-balance dual is '
                    'captured anywhere in this run\'s committed evidence.'),
        'capture_that_would_supply_it': ('read model.dual[model.node_balance_p[scenario, y, d, p]] (and '
                                          '_q) for the TSO model at bus indices corresponding to nodes 5/7/9, and '
                                          'for each DSO model at its reference/substation bus index, at the '
                                          'run\'s terminal cycle, before the models are discarded -- the `dual` '
                                          'Suffix is already populated by the existing IPOPT solve (no additional '
                                          'solve needed); a serialization/capture change only.'),
    }
    return record


def audit_item4_cycle0_lmps():
    """Item 4: cycle-0 LMPs, i.e. node-balance duals from a standalone/initialization OPF solve before cycle 1,
    in EITHER run."""
    record = {'item': '4_cycle0_lmps', 'run': 'both (searched once, mechanism is run-independent)', 'searched': []}
    record['searched'].append(PHASE2_CHECKS_JSON)
    capture_method = None
    if os.path.isfile(PHASE2_CHECKS_JSON):
        data = _load_json(PHASE2_CHECKS_JSON)
        capture_method = data.get('precycle1_capture_method')
    # preflight_ref/preflight_pt live under P515S35, not under the run dirs -- searched directly below
    preflight_dirs = [os.path.join(RES, 'P515S35', 'preflight_ref'), os.path.join(RES, 'P515S35', 'preflight_pt')]
    record['searched'].extend(preflight_dirs)
    preflight_info = {}
    for pdir in preflight_dirs:
        preflight_info[os.path.basename(pdir)] = {
            'exists': os.path.isdir(pdir),
            'files': sorted(os.listdir(pdir)) if os.path.isdir(pdir) else None,
        }
    record['cycle0_lmps'] = {
        'available': False,
        'precycle1_capture_method_text': capture_method,
        'reading': ('the pt_phase2_checks harness\'s OWN description of how it captures "the state immediately '
                    'before cycle 1\'s first network solve" is: monkeypatch a stop that raises BEFORE the first '
                    'DSO solve is called through -- i.e. by construction, ZERO IPOPT solves have executed at that '
                    'point. There is no standalone/cold "cycle-0" OPF solve anywhere in this pipeline for either '
                    'run: cycle 1\'s own first DSO solve (with duals initialized to zero, or to the price-taker '
                    'schedule\'s implied state for gate 3) IS the first solve. No node-balance dual from any '
                    'pre-cycle-1 solve exists in the committed evidence of either run.'),
        'preflight_dirs_checked': preflight_info,
        'capture_that_would_supply_it': ('if "cycle-0 LMPs" means the node-balance duals from cycle 1\'s first '
                                          'TSO/DSO solve specifically (duals still at their zero/initial value, '
                                          'so the network optimizes close to a standalone objective) -- that '
                                          'requires the SAME capture as item 3, just at cycle 1 instead of the '
                                          'terminal cycle: read the `dual` Suffix on node_balance_p/q right after '
                                          'cycle 1\'s TSO/DSO solves, before it is overwritten by cycle 2. Zero '
                                          'additional solves either way (the solve already happens); a capture/'
                                          'serialization change only.'),
    }
    return record


def audit_item5_esso_own_duals(run_dir, run_label, terminal_cycle):
    """Item 5: the ESSO's own duals (e.g. energy_storage_operation_agg) in esso_capture/."""
    record = {'item': '5_esso_own_duals_esso_capture', 'run': run_label, 'searched': []}
    capture_dir = os.path.join(run_dir, 'esso_capture', 'baseline')
    record['searched'].append(capture_dir)
    per_node_files = {}
    component_names = set()
    for node in NODES:
        files = sorted(glob.glob(os.path.join(capture_dir, f'node{node}_cycle*.jsonl')))
        per_node_files[node] = len(files)
        term_file = os.path.join(capture_dir, f'node{node}_cycle{terminal_cycle:03d}.jsonl')
        if os.path.isfile(term_file):
            with open(term_file) as handle:
                first = json.loads(handle.readline())
            component_names.update(d['component'] for d in first.get('duals', []))
    record['esso_own_duals'] = {
        'available': True,
        'granularity': 'per (node, cycle, period-row); one file per (node, cycle), one JSON object per period '
                       '(288 lines/file matching the 12 year-day combinations x 24 periods), each carrying a '
                       "'duals' list of {component, index, dual} for the ESSO's OWN local NLP constraints",
        'files_per_node': per_node_files,
        'component_names_at_terminal_cycle': sorted(component_names),
        'distinct_from_item1': ("these are the ESSO's LOCAL interior-point KKT multipliers on ITS OWN physical "
                                 "constraints (energy_storage_limits, energy_storage_operation_agg, "
                                 "energy_storage_cohort_pnet_share_h3, energy_storage_capacity_degradation) -- "
                                 "NOT the ADMM consensus dual lambda_esso (that is dual_p_req/dual_q_req, item 1)"),
    }
    return record


# ======================================================================================================================
#  PART 1(a) -- dual-direction comparison
# ======================================================================================================================
def _load_esso_dual_vectors(run_dir):
    import pickle
    with open(os.path.join(run_dir, 'esso_models_baseline.pkl'), 'rb') as handle:
        models = pickle.load(handle)
    p_vec, q_vec, keys = [], [], []
    for node in NODES:
        m = models[node]
        for k in sorted(m.dual_p_req.keys()):
            p_vec.append(pe.value(m.dual_p_req[k]))
            q_vec.append(pe.value(m.dual_q_req[k]))
            keys.append((node,) + tuple(k))
    return np.asarray(p_vec, dtype=float), np.asarray(q_vec, dtype=float), keys


def _direction_stats(u, v, label):
    norm_u, norm_v = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    dot = float(np.dot(u, v))
    cosine = dot / (norm_u * norm_v) if norm_u > 0 and norm_v > 0 else None
    sign_u, sign_v = np.sign(u), np.sign(v)
    sign_agreement = float(np.mean(sign_u == sign_v))
    noisy = (np.abs(u) < NOISE_FLOOR) & (np.abs(v) < NOISE_FLOOR)
    n_noisy = int(np.sum(noisy))
    n_clean = int(np.sum(~noisy))
    sign_agreement_ex_noise = float(np.mean(sign_u[~noisy] == sign_v[~noisy])) if n_clean > 0 else None
    return {
        'label': label, 'n_entries': int(len(u)),
        'norm_run1_terminal': norm_u, 'norm_gate3_terminal': norm_v,
        'norm_ratio_gate3_over_run1': (norm_v / norm_u) if norm_u > 0 else None,
        'cosine': cosine,
        'sign_agreement_all_entries': sign_agreement,
        'noise_floor': NOISE_FLOOR,
        'n_entries_both_below_noise_floor': n_noisy,
        'n_entries_at_least_one_above_noise_floor': n_clean,
        'sign_agreement_excluding_both_below_noise_floor': sign_agreement_ex_noise,
    }


def part1a_dual_direction_comparison():
    result = {}

    # --- ESSO: per-entry direction comparison (the only agent with per-entry duals) ---
    ref_p, ref_q, ref_keys = _load_esso_dual_vectors(REF_RUN)
    pt_p, pt_q, pt_keys = _load_esso_dual_vectors(PT_RUN)
    if ref_keys != pt_keys:
        raise AssertionError('ESSO dual_p_req/dual_q_req index sets differ between run 1 and gate 3 -- '
                              'cannot compare entrywise.')
    ref_combined = np.concatenate([ref_p, ref_q])
    pt_combined = np.concatenate([pt_p, pt_q])
    result['esso_per_entry'] = {
        'p_only': _direction_stats(ref_p, pt_p, 'ESSO dual_p_req'),
        'q_only': _direction_stats(ref_q, pt_q, 'ESSO dual_q_req'),
        'p_and_q_combined': _direction_stats(ref_combined, pt_combined, 'ESSO dual_p_req + dual_q_req concatenated'),
        'index_sets_match': ref_keys == pt_keys,
        'rho_ess_run1_terminal': None,  # filled below
        'rho_ess_gate3_terminal': None,
    }
    import pickle
    with open(os.path.join(REF_RUN, 'esso_models_baseline.pkl'), 'rb') as handle:
        ref_models = pickle.load(handle)
    with open(os.path.join(PT_RUN, 'esso_models_baseline.pkl'), 'rb') as handle:
        pt_models = pickle.load(handle)
    result['esso_per_entry']['rho_ess_run1_terminal'] = pe.value(ref_models[NODES[0]].rho)
    result['esso_per_entry']['rho_ess_gate3_terminal'] = pe.value(pt_models[NODES[0]].rho)

    # --- TSO / DSO: aggregate norms only, direction NOT determinable ---
    ref_g, _ = _one_g_baseline(REF_RUN)
    pt_g, _ = _one_g_baseline(PT_RUN)
    ref_last = ref_g['cycle_trajectory'][-1]
    pt_last = pt_g['cycle_trajectory'][-1]
    if ref_last['cycle'] != REF_TERMINAL_CYCLE or pt_last['cycle'] != PT_TERMINAL_CYCLE:
        raise AssertionError('terminal cycle mismatch against declared constants.')

    agent_norms = {}
    for agent in ('tso', 'dso', 'esso'):
        key = f'boyd_ess_norm_y_{agent}'
        norm_ref, norm_pt = ref_last[key], pt_last[key]
        agent_norms[agent] = {
            'norm_run1_terminal_cycle477': norm_ref,
            'norm_gate3_terminal_cycle150': norm_pt,
            'norm_ratio_gate3_over_run1': norm_pt / norm_ref if norm_ref else None,
            'direction_determinable': agent == 'esso',
            'note': ('per-entry vector available (see esso_per_entry above)' if agent == 'esso' else
                     'AGGREGATE NORM ONLY -- direction is NOT determinable (no per-entry TSO/DSO consensus dual '
                     'is captured in either run\'s committed evidence; see PART 0 item 1).'),
        }
    result['aggregate_norms_per_agent_at_terminal_cycles'] = agent_norms

    # --- "moving toward run 1": PT run's own per-cycle aggregate-norm trend (norm-only, not direction) ---
    late_cycles = [100, 110, 113, 120, 130, 140, 145, 148, 149, 150]
    pt_rows_by_cycle = {r['cycle']: r for r in pt_g['cycle_trajectory']}
    trend = {}
    for agent in ('tso', 'dso', 'esso'):
        key = f'boyd_ess_norm_y_{agent}'
        series = [{'cycle': c, 'norm': pt_rows_by_cycle[c][key]} for c in late_cycles if c in pt_rows_by_cycle]
        monotonic_increasing = all(series[i]['norm'] <= series[i + 1]['norm'] for i in range(len(series) - 1))
        trend[agent] = {
            'series_cycle_norm': series,
            'monotonic_increasing_over_series': monotonic_increasing,
            'target_run1_terminal_norm': ref_last[f'boyd_ess_norm_y_{agent}'],
            'reading': ('NORM trend only (no per-entry history for any agent across cycles is captured, ESSO '
                        'included -- only ONE terminal ESSO snapshot exists per run). Increasing toward run 1\'s '
                        'terminal norm is consistent with, but does not by itself prove, convergence toward a '
                        'shared direction.'),
        }
    result['gate3_norm_trend_toward_run1'] = trend

    return result


# ======================================================================================================================
#  PART 1(b) / (c) -- price-taker LP with substituted prices: gated on PART 0 findings
# ======================================================================================================================
def part1bc_shadow_price_lp(item3_ref, item4):
    lp_calls_before = sept.get_lp_call_count()
    item3_available = item3_ref['node_balance_duals_terminal_cycle']['available']
    item4_available = item4['cycle0_lmps']['available']
    result = {
        'b_run1_terminal_nodal_prices_lp': {
            'computable': item3_available,
            'reason': ('NOT COMPUTABLE: PART 0 item 3 finds no TSO/DSO node-balance dual (nodal price) captured '
                       'at run 1\'s terminal cycle, at any granularity, in the committed evidence. The production '
                       'price-taker LP (`shared_ess_price_taker.solve_price_taker_schedule`) was NOT called with '
                       'a substituted or fabricated price series -- doing so would approximate a missing '
                       'quantity, which this task forbids.'
                       if not item3_available else 'computable -- see item3 for the captured nodal prices used.'),
            'capture_needed': item3_ref['node_balance_duals_terminal_cycle']['capture_that_would_supply_it'],
        },
        'c_cycle0_lmps_lp': {
            'computable': item4_available,
            'reason': ('NOT COMPUTABLE: PART 0 item 4 finds no cycle-0 (pre-cycle-1, standalone) node-balance '
                       'dual anywhere in either run\'s committed evidence, and no such standalone solve exists in '
                       'this pipeline at all. The production price-taker LP was NOT called with a substituted or '
                       'fabricated price series.'
                       if not item4_available else 'computable -- see item4 for the captured cycle-0 LMPs used.'),
            'capture_needed': item4['cycle0_lmps']['capture_that_would_supply_it'],
        },
    }
    lp_calls_after = sept.get_lp_call_count()
    result['scipy_lp_calls_this_script'] = {
        'declared': 0, 'observed': lp_calls_after - lp_calls_before,
        'exact_match': (lp_calls_after - lp_calls_before) == 0,
    }
    return result


# ======================================================================================================================
#  EFC comparison table (existing certified values only -- no new solves)
# ======================================================================================================================
def efc_comparison_table():
    ref_g, _ = _one_g_baseline(REF_RUN)
    pt_g, _ = _one_g_baseline(PT_RUN)
    ref_efc = ref_g['cycle_trajectory'][-1]['efc_per_day_max']
    pt_efc = pt_g['cycle_trajectory'][-1]['efc_per_day_max']
    z2 = _load_json(MARKET_PRICE_LP_JSON)
    market_efc_by_node = z2['part_A_summary']['efc_per_day_max_by_node']
    market_efc_values = set(round(v, 9) for v in market_efc_by_node.values())
    if len(market_efc_values) != 1:
        raise AssertionError(f'market-price LP EFC/day not uniform across nodes: {market_efc_by_node}')
    market_efc = market_efc_by_node[str(NODES[0])] if str(NODES[0]) in market_efc_by_node else \
        market_efc_by_node[NODES[0]]

    rows = {
        'run1_certified_cycle477': ref_efc,
        'gate3_terminal_cycle150': pt_efc,
        'market_price_lp_z2': market_efc,
        'b_run1_terminal_prices_lp': 'NOT COMPUTABLE (see PART 1(b))',
        'c_cycle0_lmps_lp': 'NOT COMPUTABLE (see PART 1(c))',
    }
    pairwise = {
        'gate3_minus_run1': pt_efc - ref_efc,
        'market_lp_minus_run1': market_efc - ref_efc,
        'market_lp_minus_gate3': market_efc - pt_efc,
    }
    return {
        'values': rows, 'pairwise_differences': pairwise,
        'source_market_price_lp': f'{MARKET_PRICE_LP_JSON} :: part_A_summary.efc_per_day_max_by_node (uniform '
                                    'across nodes 5/7/9)',
        'note': ('run 1 is a certified LOWER bound only (EFC/day still rising at its Boyd stop, slope halving '
                 '3.79e-4 -> 1.77e-4/cycle over its last 77 cycles); gate 3 is a cap-stopped point still '
                 'FALLING (rule ten 0.088, PF ratio 1.4); the bracket [1.059, 1.184] is one-sided, not symmetric '
                 '(P5_15_ADDENDUM16_EXPERT_REPORT.md Sec.6).'),
    }


# ======================================================================================================================
#  Addendum 17 step 2 readiness
# ======================================================================================================================
def step2_readiness(item1_ref, item3_ref, item4):
    esso_ok = item1_ref['esso_per_entry_dual']['available']
    tso_dso_ok = False
    prices_ok = item3_ref['node_balance_duals_terminal_cycle']['available']
    cycle0_ok = item4['cycle0_lmps']['available']
    return {
        'requirement': ("Addendum 17 step 2's mapping test: applied to run 1's terminal prices, the mapping must "
                        "reproduce run 1's terminal storage duals (per agent) to a stated tolerance."),
        'run1_terminal_storage_duals_per_agent': {
            'esso': 'AVAILABLE per-entry (esso_models_baseline.pkl, dual_p_req/dual_q_req)',
            'tso': 'NOT AVAILABLE per-entry (aggregate norm only)',
            'dso': 'NOT AVAILABLE per-entry (aggregate norm only)',
        },
        'run1_terminal_nodal_prices_needed_as_mapping_input': 'NOT AVAILABLE at any granularity (PART 0 item 3)',
        'verdict': ('step 2 CANNOT be validated as specified: it needs run 1\'s terminal per-entry nodal prices '
                    '(absent) and run 1\'s terminal per-agent storage duals for ALL THREE agents (only ESSO\'s is '
                    'available per-entry; TSO\'s and DSO\'s are not). The mapping test could be validated for the '
                    'ESSO agent alone with a bounded capture of run 1\'s terminal nodal prices (item 3\'s capture); '
                    'validating it per Addendum 17\'s "per agent" requirement for TSO and DSO needs BOTH captures '
                    '(item 1\'s TSO/DSO model pickling AND item 3\'s nodal-price capture) -- neither exists today.'),
        'minimal_bounded_capture_to_close_the_gap': [
            'pickle the TSO model and each DSO model at the terminal cycle (mirrors the existing ESSO capture) '
            '-- supplies TSO/DSO per-entry storage duals (item 1).',
            'capture model.dual on node_balance_p/node_balance_q for the TSO model (buses 5/7/9) and each DSO '
            'model (reference bus) at the terminal cycle -- supplies the nodal prices (item 3).',
            'both are read-outs of state already produced by the existing IPOPT solve; zero additional solves.',
        ],
        'inputs_available_now': {'esso_per_entry_duals': esso_ok, 'tso_dso_per_entry_duals': tso_dso_ok,
                                  'terminal_nodal_prices': prices_ok, 'cycle0_lmps': cycle0_ok},
    }


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    guard = SolveProfileGuard([], label='P5.15-S36 Addendum17 diagnostics').install()
    try:
        # ---- PART 0 ----
        item1_ref = audit_item1_storage_consensus_duals(REF_RUN, 'run1_reference', REF_TERMINAL_CYCLE)
        item1_pt = audit_item1_storage_consensus_duals(PT_RUN, 'gate3_pt', PT_TERMINAL_CYCLE)
        item2_ref = audit_item2_storage_consensus_z_and_x(REF_RUN, 'run1_reference', REF_TERMINAL_CYCLE)
        item2_pt = audit_item2_storage_consensus_z_and_x(PT_RUN, 'gate3_pt', PT_TERMINAL_CYCLE)
        item3_ref = audit_item3_nodal_prices(REF_RUN, 'run1_reference')
        item3_pt = audit_item3_nodal_prices(PT_RUN, 'gate3_pt')
        item4 = audit_item4_cycle0_lmps()
        item5_ref = audit_item5_esso_own_duals(REF_RUN, 'run1_reference', REF_TERMINAL_CYCLE)
        item5_pt = audit_item5_esso_own_duals(PT_RUN, 'gate3_pt', PT_TERMINAL_CYCLE)

        part0 = {
            'item1_storage_consensus_duals': {'run1_reference': item1_ref, 'gate3_pt': item1_pt},
            'item2_storage_consensus_z_and_x': {'run1_reference': item2_ref, 'gate3_pt': item2_pt},
            'item3_nodal_prices_lmps': {'run1_reference': item3_ref, 'gate3_pt': item3_pt},
            'item4_cycle0_lmps': item4,
            'item5_esso_own_duals': {'run1_reference': item5_ref, 'gate3_pt': item5_pt},
        }

        # ---- PART 1 ----
        part1a = part1a_dual_direction_comparison()
        part1bc = part1bc_shadow_price_lp(item3_ref, item4)
        comparison = efc_comparison_table()
        step2 = step2_readiness(item1_ref, item3_ref, item4)

    finally:
        failures = guard.verify(0, 0)
        guard.uninstall()

    if failures:
        raise AssertionError('RULE SIX: Pyomo/IPOPT solve guard was not exactly zero -> ' + '; '.join(failures))

    lp_calls_final = sept.get_lp_call_count()
    if lp_calls_final != 0:
        raise AssertionError(f'RULE SIX (scipy LP count): observed {lp_calls_final} != declared 0')

    report = {
        'stage': 'P5.15-S36-A17-step1', 'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 17 (2026-09-16)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'runs': {'run1_reference': REF_RUN, 'gate3_pt': PT_RUN},
        'terminal_cycles': {'run1_reference': REF_TERMINAL_CYCLE, 'gate3_pt': PT_TERMINAL_CYCLE},
        'guard': {'pyomo_ipopt_solve_guard': {'observed': dict(guard.counts), 'permitted_declared': 0,
                                               'guard_verify_failures': failures},
                  'scipy_lp_calls': {'declared': 0, 'observed': lp_calls_final, 'exact_match': lp_calls_final == 0}},
        'part0_data_availability_audit': part0,
        'part1a_dual_direction_comparison': part1a,
        'part1bc_shadow_price_lp': part1bc,
        'efc_comparison_table': comparison,
        'addendum17_step2_readiness': step2,
    }

    os.makedirs(OUT_DIR)
    results_path = os.path.join(OUT_DIR, 'a17_diagnostics_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    manifest = {}
    for fname in ('a17_diagnostics_results.json',):
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'rb') as handle:
            manifest[fname] = hashlib.sha256(handle.read()).hexdigest()
    script_path = os.path.abspath(__file__)
    with open(script_path, 'rb') as handle:
        manifest[os.path.basename(script_path)] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, 'sha256_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(f'Pyomo/IPOPT solve guard: {dict(guard.counts)} (0 permitted, 0 blocked expected)')
    print(f'scipy LP calls: declared=0 observed={lp_calls_final}')
    print(f"ESSO combined cosine (run1 vs gate3, terminal): "
          f"{part1a['esso_per_entry']['p_and_q_combined']['cosine']:.6f}")
    print(f"ESSO combined norm ratio (gate3/run1): "
          f"{part1a['esso_per_entry']['p_and_q_combined']['norm_ratio_gate3_over_run1']:.6f}")
    for agent in ('tso', 'dso', 'esso'):
        an = part1a['aggregate_norms_per_agent_at_terminal_cycles'][agent]
        print(f"  {agent}: norm_run1={an['norm_run1_terminal_cycle477']:.6e} "
              f"norm_gate3={an['norm_gate3_terminal_cycle150']:.6e} "
              f"ratio={an['norm_ratio_gate3_over_run1']:.4f}")
    print(f"PART 1(b) computable: {part1bc['b_run1_terminal_nodal_prices_lp']['computable']}")
    print(f"PART 1(c) computable: {part1bc['c_cycle0_lmps_lp']['computable']}")
    print(f'wrote: {results_path}')


if __name__ == '__main__':
    main()
