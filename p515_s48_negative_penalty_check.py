"""P5.15 Addendum 32 Q6 (task W26) -- negative penalty levels. ZERO SOLVES, read-only analysis.

Armed `SolveProfileGuard(permitted=())` is installed before any production import; `verify(0)` is checked at the
end (exact count: 0 solves, 0 Pyomo solver launches).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 32 Q6; frozen spec v18
(data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json, key `Q6_negative_penalties`); Planner task W26.

QUESTION. component_levels_terminal.json reports NEGATIVE levels for some penalty components (e.g. the voltage-slack
component -11,797 at the baseline smallest node-7 unit). Classify every negative level as (a) a reporting convention
(the report subtracts / differences / signs something as a credit) or (b) an unbounded (or wrongly bounded) slack.

WHAT IS COMPUTED (formulas are also written into the JSON and the Markdown)
  P1 Inventory. Scope: every component_levels_terminal.json TRACKED IN GIT under data/SRP1/Results/P515S44 ..
     P515S47 (`git ls-files`). For each file: the sibling evaluation_record.json status (certified / not certified /
     no record), candidate_key; every (block, component) whose level is < 0 (strictly; IEEE -0.0 is counted
     separately as a signed zero), both conventions ("unweighted" = scenario-probability-weighted per block;
     "weighted" = unweighted * admm_block_weight). Also the most positive level of every component.
  P2 Code trace. file:line citations resolved at run time (unique-needle search) for the level builder, the penalty
     helpers, the objective, the slack declarations and bounds; the source text of every penalty helper is hashed
     and stored. Scoped search for IPOPT bound-relaxation options in the repository code and the SRP1 case files.
  P3 Models. The five persisted certified TSO/DSO model sets (P515S44 `certified_models.pkl`, sha256 verified
     against each evaluation's child manifest), loaded ONE AT A TIME. Per block and slack family: domain, raw
     declared bounds (`var.lower`/`var.upper`), effective bounds (`var.lb`/`var.ub`), fixed flags, terminal values
     (min/max/sum), counts below the original lower bound and below IPOPT's relaxed lower bound
        lb_relaxed = lb - min(bound_relax_factor * max(1, |lb|), constr_viol_tol),  1e-8 and 1e-4 (IPOPT defaults,
        not overridden anywhere in the searched scope -- P2),
     the objective coefficient of every slack (standard repn of `total_slack_penalties +
     total_ess_complementarity_penalties`, and separately of the ACTIVE solver objective), the level recomputed
        (i)  with the production builder (p515_g_g1_g4_admm_gates._s31_block_components -> srp.
             _get_local_detector_components, using the build-time `network`/`params` objects recovered from the
             pickled model's own bound-rule partial),
        (ii) independently as sum(coef * value) over the repn,
     both matched against the reported value; the level at the slack values projected onto their ORIGINAL bounds,
        level_proj = sum(coef * min(max(value, lb), ub)),   bias = level_reported - level_proj,
     and the objective re-evaluation check: sum_blocks w * (Network.get_primal_value(model, params) - settlement)
     == recourse gross_operational_cost (so the reported detector levels are inside Q).
     RES curtailment (a definitional, not a slack): pg_avail - pg, pg's declared upper bound, and the objective
     weight `penalty_gen_curtailment`.
  P4 Classification and effect on Q. For every certified record: detector_penalty_total (weighted),
     d_det(x) = det(x) - det(0) against the x = 0 record, value = Q(0) - Q(x), and |d_det| / bar(x). A floor test
     over the whole inventory: every negative reported level must lie at or above the relaxed-bound floor
        floor(agent, node, component) = -sum over FREE slack entries of coef * min(1e-8 * max(1, |lb|), 1e-4)
     (free = not fixed and lb < ub; the most negative floor over the five model sets and all blocks).

Output (write-once, new directory) data/SRP1/Results/P515S48/negative_penalty_check/:
    negative_penalty_check.json, negative_penalty_check.md, launch.log, manifest_sha256.json

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S48/negative_penalty_check
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s48_negative_penalty_check.py \\
        > data/SRP1/Results/P515S48/negative_penalty_check/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s48_negative_penalty_check.py --manifest
"""
import gc
import hashlib
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W26 Q6 negative penalty levels (zero solves)').install()

import functools  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.repn import generate_standard_repn  # noqa: E402

import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
from definitions import (PENALTY_VOLTAGE_SQUARED, PENALTY_NODE_BALANCE, PENALTY_CURRENT,  # noqa: E402
                         PENALTY_FLEXIBILITY, PENALTY_ESS_BALANCE, PENALTY_SHARED_ESS_BALANCE,
                         PENALTY_GENERATION_CURTAILMENT, EQUALITY_TOLERANCE)

STAGE = 'P5.15 Addendum 32 Q6 W26 -- negative penalty levels: reporting convention or unbounded slack (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 32 Q6',
             'data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json Q6_negative_penalties',
             'Planner task W26, 2026-09-22']
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S48', 'negative_penalty_check')
# W26_SCRATCH_OUT (optional, test runs only): write to a scratch directory instead; the value is recorded.
OUT_DIR = os.environ.get('W26_SCRATCH_OUT') or os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'negative_penalty_check.json'
MD_NAME = 'negative_penalty_check.md'
MANIFEST_NAME = 'manifest_sha256.json'
LOCK_NAME = '.w26.lock'

INVENTORY_STAGES = ('P515S44', 'P515S45', 'P515S46', 'P515S47')
MODEL_EVAL_DIRS = (
    'data/SRP1/Results/P515S44/campaign_s44_aa_variant/evals/837fc982565dbba3_c_star_aa_keep_memory',
    'data/SRP1/Results/P515S44/campaign_s44_aa_variant/evals/4e53fa5560bbfa10_two_c_star_d',
    'data/SRP1/Results/P515S44/campaign_s44_selection_aa/evals/0796b6dceeb0f95f_node7_empty_aa_keep_memory',
    'data/SRP1/Results/P515S44/campaign_s44_selection_aa/evals/8e48f3ec8993a283_paper_plan_aa_keep_memory',
    'data/SRP1/Results/P515S44/campaign_s44_selection_aa/evals/a6ca6c94e033e460_two_c_star_aa_keep_memory',
)
X0_DIR = 'data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0'
X0_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
BASELINE_DIR = 'data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1'
BASELINE_KEY = 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'

# IPOPT defaults (Ipopt 3.14 documentation; `ipopt --print-options` output is captured in P2 as evidence).
BOUND_RELAX_FACTOR = 1e-8
CONSTR_VIOL_TOL = 1e-4

# Tolerances, stated before the committed run:
#  - recompute vs reported level (same production function, same model point): |diff| <= 1e-9 * max(1, |level|)
#  - repn recompute vs reported level: |diff| <= 1e-9 * max(1, |level|) + 1e-12
#  - gross reconciliation (objective re-evaluation vs recourse gross): |diff| <= 1e-6 relative
#  - "at or above the relaxed floor": level >= floor - 1e-12 * max(1, |floor|)
LEVEL_TOL_REL = 1e-9
GROSS_TOL_REL = 1e-6
FLOOR_TOL_REL = 1e-12

DETECTOR = ('voltage_slack', 'node_balance_slack', 'branch_flow_slack', 'flexibility_p_day_balance_slack',
            'local_ess_day_balance_slack', 'shared_ess_day_balance_slack')
FAMILIES = {
    'voltage_slack': ('slack_v_sqr_down', 'slack_v_sqr_up'),
    'node_balance_slack': ('slack_node_balance_p_up', 'slack_node_balance_p_down',
                           'slack_node_balance_q_up', 'slack_node_balance_q_down'),
    'branch_flow_slack': ('slack_flow_ij_sqr', 'slack_flow_ji_sqr'),
    'flexibility_p_day_balance_slack': ('slack_flex_p_balance_up', 'slack_flex_p_balance_down'),
    'local_ess_day_balance_slack': ('slack_es_soc_final_up', 'slack_es_soc_final_down'),
    'shared_ess_day_balance_slack': ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down'),
    'orphan_flex_q_day_balance (not in objective)': ('slack_flex_q_balance_up', 'slack_flex_q_balance_down'),
}
OBJ_CONV = ('Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); '
            'value = Q(0) - Q(x); EUR; "weighted" = unweighted * admm_block_weight (num_years * num_days / '
            '1.02^(y - 2025)); terminal salvage is reported with each record.')


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _load(path):
    with open(path if os.path.isabs(path) else os.path.join(REPO, path)) as handle:
        return json.load(handle)


def _lines(rel, needle):
    with open(os.path.join(REPO, rel)) as handle:
        return [i + 1 for i, line in enumerate(handle) if needle in line]


def _one_line(rel, needle):
    found = _lines(rel, needle)
    if len(found) != 1:
        raise RuntimeError(f'citation needle not unique in {rel}: {needle!r} -> {found}')
    return f'{rel}:{found[0]}'


def _signed_zero(x):
    return x == 0 and math.copysign(1.0, x) < 0


def _record_of(eval_dir):
    path = os.path.join(REPO, eval_dir, 'evaluation_record.json')
    if not os.path.exists(path):
        return None
    return _load(path)


# ======================================================================================================================
#  P1 inventory
# ======================================================================================================================
def p1_inventory():
    tracked = [f for f in _git(['ls-files']).split('\n')
               if f.endswith('/component_levels_terminal.json') and any(f'/{s}/' in f for s in INVENTORY_STAGES)]
    files, rows, per_comp = [], [], {}
    positive_rows = []
    signed_zero_count = {}
    for fidx, rel in enumerate(sorted(tracked)):
        d = _load(rel)
        rec = _record_of(os.path.dirname(rel))
        status = rec.get('status') if rec else 'no_evaluation_record'
        entry = {'index': fidx, 'path': rel, 'sha256': _sha256(os.path.join(REPO, rel)), 'status': status,
                 'campaign_id': rec.get('campaign_id') if rec else None,
                 'candidate_label': rec.get('candidate_label') if rec else None,
                 'candidate_key': rec.get('candidate_key') if rec else None,
                 'ess_ageing_baseline_label': rec.get('ess_ageing_baseline_label') if rec else None,
                 'n_blocks': len(d['blocks']), 'n_negative_block_levels': 0,
                 'totals_weighted_negative': {k: v for k, v in d.get('totals_weighted', {}).items()
                                              if isinstance(v, (int, float)) and v < 0},
                 'recourse_detector_penalty_total': (d.get('recourse_components') or {}).get('detector_penalty_total')}
        for bkey, blk in d['blocks'].items():
            agent = blk['kind']
            node = blk.get('node_id')
            for comp, u in blk['unweighted'].items():
                w = blk['weighted'][comp]
                ck = (agent, comp)
                pc = per_comp.setdefault(ck, {'agent': agent, 'component': comp, 'n_block_levels': 0,
                                              'n_negative': 0, 'n_negative_certified': 0, 'n_positive': 0,
                                              'min_unweighted': 0.0, 'max_unweighted': 0.0,
                                              'min_weighted': 0.0, 'max_weighted': 0.0})
                pc['n_block_levels'] += 1
                pc['min_unweighted'] = min(pc['min_unweighted'], u)
                pc['max_unweighted'] = max(pc['max_unweighted'], u)
                pc['min_weighted'] = min(pc['min_weighted'], w)
                pc['max_weighted'] = max(pc['max_weighted'], w)
                if u > 0:
                    pc['n_positive'] += 1
                    if comp in DETECTOR:
                        positive_rows.append({'file_index': fidx, 'path': rel, 'status': status,
                                              'candidate_key': entry['candidate_key'], 'block': bkey,
                                              'agent': agent, 'node_id': node, 'year': blk['year'], 'day': blk['day'],
                                              'component': comp, 'unweighted': u, 'weighted': w,
                                              'cycles_run': d.get('cycles_run')})
                if _signed_zero(u):
                    signed_zero_count[f'{agent}|{comp}'] = signed_zero_count.get(f'{agent}|{comp}', 0) + 1
                if u < 0:
                    pc['n_negative'] += 1
                    if status == 'certified':
                        pc['n_negative_certified'] += 1
                    entry['n_negative_block_levels'] += 1
                    rows.append([fidx, bkey, agent, node, comp, u, w])
        files.append(entry)
    # join every positive detector-family level with the run's network-failure log at the TERMINAL cycle
    for r in positive_rows:
        ed = os.path.join(REPO, os.path.dirname(r['path']))
        logs = sorted(n for n in os.listdir(ed) if n.startswith('network_failures_') and n.endswith('.jsonl'))
        match = []
        for n in logs:
            with open(os.path.join(ed, n)) as handle:
                for line in handle:
                    x = json.loads(line)
                    if (x.get('agent') == r['agent'] and str(x.get('year')) == r['year'] and x.get('day') == r['day']
                            and (r['agent'] == 'TSO' or str(x.get('node_id')) == str(r['node_id']))):
                        match.append({'log': n, 'cycle': x.get('cycle'), 'class': x.get('class'),
                                      'primary_termination': x.get('primary_termination')})
        r['network_failure_rows_same_block'] = match
        r['terminal_cycle_solve_was_a_recovery'] = any(
            str(m['cycle']) == str(r['cycles_run']) for m in match)
    return {
        'positive_detector_rows': positive_rows,
        'positive_detector_note': ('detector-family block levels > 0, joined with network_failures_*.jsonl of the same '
                                   'evaluation (same agent/node/year/day); terminal_cycle_solve_was_a_recovery = a row '
                                   'exists at cycle == cycles_run (the block\'s terminal solve was a recovery retry)'),
        'scope': ('every component_levels_terminal.json tracked in git (git ls-files at HEAD) whose path contains '
                  '/P515S44/, /P515S45/, /P515S46/ or /P515S47/; certification status from the sibling '
                  'evaluation_record.json (absent = harness checks / gates / zero-check runs, listed but flagged)'),
        'n_files': len(files),
        'status_counts': {s: sum(1 for f in files if f['status'] == s) for s in sorted({f['status'] for f in files})},
        'files': files,
        'negative_rows_columns': ['file_index', 'block', 'agent', 'node_id', 'component', 'unweighted', 'weighted'],
        'negative_rows': rows,
        'per_agent_component': sorted(per_comp.values(), key=lambda r: (r['agent'], r['component'])),
        'signed_zero_counts': signed_zero_count,
        'signed_zero_note': ('IEEE -0.0 values (e.g. res_curtailment_penalty = 0 * negative) are NOT negative; '
                             'counted here only so the "-0.0" entries in the files are accounted for'),
    }


# ======================================================================================================================
#  P2 code trace
# ======================================================================================================================
def p2_code_trace():
    srp_f, mch_f, net_f, gh = 'shared_resources_planning.py', 'model_construction_helpers.py', 'network.py', \
        'p515_g_g1_g4_admm_gates.py'
    cite = {
        'level_builder_harness_s31_block_components': _one_line(gh, 'def _s31_block_components(model, network, params):'),
        'level_builder_harness_calls_production_detector': _one_line(gh, 'detector = srp._get_local_detector_components(model, network, params)'),
        'level_builder_harness_weighted_is_weight_times_unweighted':
            [f'{gh}:{n}' for n in _lines(gh, 'weighted = {name: weight * value for name, value in unweighted.items()}')],
        'level_builder_harness_res_curtailment_definitional': _one_line(gh, "out['res_curtailment_definitional_at_weight_1'] = _s31_scenario_weighted("),
        'level_builder_harness_writer': _one_line(gh, 'def write_component_levels_terminal(planning, sed, models, rows, report, out_dir, label):'),
        'production_detector_components': _one_line(srp_f, 'def _get_local_detector_components(model, network, params):'),
        'production_detector_voltage_accumulation': _one_line(srp_f, "components['voltage_slack'] += probability * pe.value(voltage_slack_penalty(model, network, s_m, s_o, params))"),
        'production_detector_total_is_plain_sum': _one_line(srp_f, "components['detector_penalty_total'] = sum("),
        'production_recourse_gross_minus_settlement_only': _one_line(srp_f, 'gross_operational_cost = gross_operational_cost_including_settlement - interface_settlement_total'),
        'production_recourse_D_excluded_reporting_field': _one_line(srp_f, "'economic_recourse_all_D_excluded': net_operational_recourse - detector_penalty_total,"),
        'penalty_voltage': _one_line(mch_f, 'total += PENALTY_VOLTAGE_SQUARED * (model.slack_v_sqr_down[i, s_m, s_o, p] + model.slack_v_sqr_up[i, s_m, s_o, p])'),
        'penalty_flex_p_day_balance': _one_line(mch_f, 'total += base * PENALTY_FLEXIBILITY * (model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o])'),
        'penalty_local_ess_day_balance': _one_line(mch_f, 'total += base * PENALTY_ESS_BALANCE * (model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o])'),
        'penalty_shared_ess_day_balance': _one_line(mch_f, 'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m, s_o] + model.slack_shared_es_soc_final_down[e, s_m, s_o])'),
        'objective_function_rule': _one_line(mch_f, 'def objective_function_rule(model, params):'),
        'objective_adds_slack_penalties': _one_line(mch_f, '    obj += model.total_slack_penalties'),
        'objective_adds_ess_day_balance_penalties': _one_line(mch_f, '    obj += model.total_ess_complementarity_penalties'),
        'get_primal_value_evaluates_objective_function_rule': _one_line(net_f, 'primal_value = pe.value(objective_function_rule(model, params))'),
        'decl_slack_v_sqr_down': _one_line(net_f, 'model.slack_v_sqr_down = pe.Var('),
        'decl_slack_v_sqr_up': _one_line(net_f, 'model.slack_v_sqr_up = pe.Var('),
        'decl_slack_flex_p_balance_up': _one_line(net_f, 'model.slack_flex_p_balance_up = pe.Var('),
        'decl_slack_es_soc_final_up': _one_line(net_f, 'model.slack_es_soc_final_up = pe.Var('),
        'decl_slack_shared_es_soc_final_up': _one_line(net_f, 'model.slack_shared_es_soc_final_up = pe.Var('),
        'bounds_voltage_down': _one_line(mch_f, 'def voltage_slack_down_bounds(m, i, s_m, s_o, p, network, params):'),
        'bounds_voltage_up': _one_line(mch_f, 'def voltage_slack_up_bounds(m, i, s_m, s_o, p, network, params):'),
        'bounds_shared_ess_slack_loop': _one_line(mch_f, "for variable_name in ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down'):"),
        'bounds_shared_ess_slack_setub': _one_line(mch_f, 'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE'),
        'pg_bounds_curtaillable_upper_tolerance': _one_line(mch_f, 'return (0.0, gen.pg[s_o][p] + EQUALITY_TOLERANCE)'),
        'res_curtailment_definitional_term': _one_line(mch_f, 'total += weight * network.baseMVA * (model.pg_avail[g, s_o, p] - model.pg[g, s_m, s_o, p])'),
        'prior_art_voltage_slack_diagnostics_clips_at_0': _one_line(mch_f, 'effective_down = max(slack_down, 0.00)'),
        'network_solver_options_merge': _one_line(net_f, 'options.update(solver_params.options)'),
    }
    sources = {}
    for fn in (mch.voltage_slack_penalty, mch.node_balance_slack_penalty, mch.branch_flow_slack_penalty,
               mch.flexibility_p_day_balance_slack_penalty, mch.local_ess_day_balance_slack_penalty,
               mch.shared_ess_day_balance_slack_penalty, mch.objective_function_rule,
               mch.total_slack_penalties_rule, mch.total_ess_complementarity_penalties_rule,
               srp._get_local_detector_components, G._s31_block_components):
        src = inspect.getsource(fn)
        sources[fn.__qualname__] = {'module': fn.__module__, 'sha256': hashlib.sha256(src.encode()).hexdigest(),
                                    'source': src}
    # scoped negative claim: IPOPT bound-relaxation options set anywhere in the searched scope?
    needles = ('bound_relax_factor', 'honor_original_bounds', 'constr_viol_tol')
    searched_py = [f for f in _git(['ls-files', '*.py']).split('\n') if f]
    case_files = [f for f in _git(['ls-files', 'data/SRP1']).split('\n')
                  if f.endswith('.json') and '/Results/' not in f]
    hits = []
    for rel in searched_py + case_files:
        try:
            with open(os.path.join(REPO, rel), errors='replace') as handle:
                for i, line in enumerate(handle):
                    if any(n in line for n in needles):
                        hits.append(f'{rel}:{i + 1}: {line.strip()[:160]}')
        except FileNotFoundError:
            continue
    production_py = {'network.py', 'shared_resources_planning.py', 'shared_energy_storage_data.py',
                     'model_construction_helpers.py', 'helper_functions.py', 'network_data.py',
                     'network_parameters.py', 'definitions.py'}
    production_hits = [h for h in hits if h.split(':', 1)[0] in production_py or h.startswith('data/')]
    # IPOPT documentation query (NOT a solve: no .nl file, no model, not routed through Pyomo; the guard's counts
    # are unaffected by construction). Recorded so the default values the classification relies on are evidence.
    ipopt_doc = {}
    try:
        out = subprocess.run(['/usr/local/bin/ipopt', '--print-options'], capture_output=True, text=True,
                             timeout=60).stdout
        lines = out.split('\n')
        for key in needles:
            idx = [i for i, ln in enumerate(lines) if ln.startswith(key)]
            ipopt_doc[key] = lines[idx[0]:idx[0] + 3] if idx else None
        ver = subprocess.run(['/usr/local/bin/ipopt', '--version'], capture_output=True, text=True,
                             timeout=60).stdout.strip()
        ipopt_doc['version'] = ver
    except Exception as exc:  # noqa: BLE001
        ipopt_doc['error'] = repr(exc)
    return {'citations': cite, 'helper_sources': sources,
            'ipopt_option_search': {'needles': needles,
                                    'scope': ('every git-tracked *.py file (production and p5* harnesses) and every '
                                              'git-tracked data/SRP1 *.json outside Results/ (case, params, ESS '
                                              'params files)'),
                                    'n_files_searched': len(searched_py) + len(case_files),
                                    'all_hits': hits, 'hits_in_production_or_case_files': production_hits},
            'ipopt_documentation_query': {
                'what': ('/usr/local/bin/ipopt --print-options and --version: documentation output only; no NLP, '
                         'no .nl file, not a Pyomo solver call (the armed guard counts Pyomo solves/launches)'),
                'result': ipopt_doc},
            'reading': ('Every D-row penalty helper adds PENALTY * (slack_up + slack_down) with a POSITIVE constant '
                        'coefficient (no subtraction, no reference level, no credit sign); the harness level is the '
                        'production helper value times the probability (unweighted) and times admm_block_weight '
                        '(weighted); detector_penalty_total is a plain sum. The "detector" semantics are a '
                        'classification label (D rows = feasibility detectors reported separately), not a sign '
                        'convention: economic_recourse_all_D_excluded = net - detector_penalty_total is an '
                        'ADDITIONAL reporting field. A negative level therefore requires negative slack VALUES.')}


# ======================================================================================================================
#  P3 models
# ======================================================================================================================
def _build_time_objects(block):
    """The `network` / `params` objects the block was BUILT with, recovered from the pickled bound-rule partial of
    `slack_v_sqr_down` (network.py: bounds=partial(voltage_slack_down_bounds, network=network, params=params))."""
    fcn = block.slack_v_sqr_down._rule_bounds._initializer._fcn
    if not (isinstance(fcn, functools.partial) and fcn.func is mch.voltage_slack_down_bounds):
        raise RuntimeError(f'unexpected bound rule on {block.name}: {fcn!r}')
    return fcn.keywords['network'], fcn.keywords['params']


def _linear_coefs(expr):
    repn = generate_standard_repn(expr, compute_values=True, quadratic=True)
    coefs = {}
    for v, c in zip(repn.linear_vars, repn.linear_coefs):
        coefs[id(v)] = coefs.get(id(v), 0.0) + float(c)
    return coefs, float(pe.value(repn.constant)), repn


def _block_analysis(block, weight_reported, reported_unweighted):
    network, params = _build_time_objects(block)
    out = {'baseMVA': network.baseMVA, 'network_name': network.name, 'year': str(network.year), 'day': network.day,
           'admm_block_weight_model': float(pe.value(block.admm_block_weight)),
           'admm_block_weight_reported': weight_reported,
           'solver_options_build_time': dict(params.solver_params.options or {}),
           'families': {}}
    pen_expr = block.total_slack_penalties + block.total_ess_complementarity_penalties
    coefs, pen_const, _ = _linear_coefs(pen_expr)
    active = [o for o in block.component_objects(pe.Objective, active=True)]
    if len(active) != 1:
        raise RuntimeError(f'{block.name}: {len(active)} active objectives')
    obj_coefs, _, obj_repn = _linear_coefs(active[0].expr)
    out['active_objective'] = active[0].name
    out['penalty_expr_constant'] = pen_const
    out['penalty_expr_value'] = float(pe.value(pen_expr))
    lvl_repn_total, lvl_proj_total = 0.0, 0.0
    for fam, names in FAMILIES.items():
        fs = {'variables_present': [], 'n': 0, 'n_fixed': 0, 'n_lb_eq_ub': 0, 'n_free': 0, 'domains': [],
              'raw_lower_min': None, 'raw_lower_max': None, 'raw_upper_min': None, 'raw_upper_max': None,
              'lb_min': None, 'lb_max': None, 'ub_min': None, 'ub_max': None, 'n_ub_none': 0, 'n_lb_none': 0,
              'value_min': None, 'value_max': None, 'value_sum': 0.0, 'n_value_none': 0,
              'n_negative': 0, 'n_below_lb': 0, 'max_below_lb': 0.0, 'n_below_relaxed_lb': 0,
              'n_above_ub': 0, 'n_above_relaxed_ub': 0,
              'coef_min': None, 'coef_max': None, 'n_in_penalty_repn': 0,
              'objective_coef_min': None, 'objective_coef_max': None, 'n_in_objective_repn': 0,
              'free_coef_sum': 0.0, 'relaxed_floor': 0.0,
              'level_repn': 0.0, 'level_projected': 0.0}

        def _mm(key_min, key_max, x):
            if x is None:
                return
            fs[key_min] = x if fs[key_min] is None else min(fs[key_min], x)
            fs[key_max] = x if fs[key_max] is None else max(fs[key_max], x)

        for name in names:
            comp = getattr(block, name, None)
            if comp is None:
                continue
            fs['variables_present'].append(name)
            for vd in comp.values():
                fs['n'] += 1
                dom = vd.domain.name if hasattr(vd.domain, 'name') else str(vd.domain)
                if dom not in fs['domains']:
                    fs['domains'].append(dom)
                raw_l = pe.value(vd.lower) if vd.lower is not None else None
                raw_u = pe.value(vd.upper) if vd.upper is not None else None
                _mm('raw_lower_min', 'raw_lower_max', raw_l)
                _mm('raw_upper_min', 'raw_upper_max', raw_u)
                lb, ub = vd.lb, vd.ub
                if lb is None:
                    fs['n_lb_none'] += 1
                if ub is None:
                    fs['n_ub_none'] += 1
                _mm('lb_min', 'lb_max', lb)
                _mm('ub_min', 'ub_max', ub)
                if vd.fixed:
                    fs['n_fixed'] += 1
                if lb is not None and ub is not None and lb == ub:
                    fs['n_lb_eq_ub'] += 1
                free = (not vd.fixed) and not (lb is not None and ub is not None and lb == ub)
                v = vd.value
                if v is None:
                    fs['n_value_none'] += 1
                    continue
                _mm('value_min', 'value_max', v)
                fs['value_sum'] += v
                if v < 0:
                    fs['n_negative'] += 1
                if lb is not None and v < lb:
                    fs['n_below_lb'] += 1
                    fs['max_below_lb'] = max(fs['max_below_lb'], lb - v)
                if lb is not None and v < lb - min(BOUND_RELAX_FACTOR * max(1.0, abs(lb)), CONSTR_VIOL_TOL):
                    fs['n_below_relaxed_lb'] += 1
                if ub is not None and v > ub:
                    fs['n_above_ub'] += 1
                if ub is not None and v > ub + min(BOUND_RELAX_FACTOR * max(1.0, abs(ub)), CONSTR_VIOL_TOL):
                    fs['n_above_relaxed_ub'] += 1
                c = coefs.get(id(vd))
                if c is not None:
                    fs['n_in_penalty_repn'] += 1
                    _mm('coef_min', 'coef_max', c)
                    fs['level_repn'] += c * v
                    proj = v
                    if lb is not None:
                        proj = max(proj, lb)
                    if ub is not None:
                        proj = min(proj, ub)
                    fs['level_projected'] += c * proj
                    if free:
                        fs['free_coef_sum'] += c
                        relax = min(BOUND_RELAX_FACTOR * max(1.0, abs(lb if lb is not None else 0.0)), CONSTR_VIOL_TOL)
                        fs['relaxed_floor'] += c * ((lb if lb is not None else 0.0) - relax)
                oc = obj_coefs.get(id(vd))
                if oc is not None:
                    fs['n_in_objective_repn'] += 1
                    _mm('objective_coef_min', 'objective_coef_max', oc)
                if free:
                    fs['n_free'] += 1
        if fam in reported_unweighted:
            fs['level_reported_unweighted'] = reported_unweighted[fam]
            fs['level_bias_unweighted'] = reported_unweighted[fam] - fs['level_projected']
        lvl_repn_total += fs['level_repn']
        lvl_proj_total += fs['level_projected']
        out['families'][fam] = fs
    # (i) production builder recompute on the build-time network/params
    recomputed = G._s31_block_components(block, network, params)
    out['recomputed_production'] = recomputed
    diffs = {}
    for k, rv in reported_unweighted.items():
        if k in recomputed:
            diffs[k] = recomputed[k] - rv
    out['recompute_minus_reported'] = diffs
    out['recompute_max_abs_diff'] = max((abs(v) for v in diffs.values()), default=0.0)
    out['detector_total_repn'] = lvl_repn_total - out['families']['orphan_flex_q_day_balance (not in objective)']['level_repn']
    out['detector_total_projected'] = lvl_proj_total - out['families']['orphan_flex_q_day_balance (not in objective)']['level_projected']
    # objective re-evaluation: this block's contribution to gross (production Network.get_primal_value)
    out['primal_value'] = float(network.get_primal_value(block, params))
    out['interface_settlement_local'] = float(srp._get_local_interface_settlement(block))
    # RES curtailment definitional: pg_avail - pg and pg's declared upper bound
    rc = {'penalty_gen_curtailment_param': float(pe.value(block.penalty_gen_curtailment)),
          'PENALTY_GENERATION_CURTAILMENT_definitional_weight': PENALTY_GENERATION_CURTAILMENT,
          'n_entries': 0, 'min_avail_minus_pg': None, 'max_avail_minus_pg': None, 'n_pg_above_avail': 0,
          'max_ub_minus_avail': None, 'min_ub_minus_avail': None, 'n_pg_above_ub': 0,
          'n_pg_above_relaxed_ub': 0}
    if params.rg_curt:
        for g in block.generators:
            if not network.generators[g].is_curtaillable():
                continue
            for s_m in block.scenarios_market:
                for s_o in block.scenarios_operation:
                    for p in block.periods:
                        pgv = block.pg[g, s_m, s_o, p]
                        avail = float(pe.value(block.pg_avail[g, s_o, p]))
                        d = avail - float(pgv.value)
                        rc['n_entries'] += 1
                        rc['min_avail_minus_pg'] = d if rc['min_avail_minus_pg'] is None else min(rc['min_avail_minus_pg'], d)
                        rc['max_avail_minus_pg'] = d if rc['max_avail_minus_pg'] is None else max(rc['max_avail_minus_pg'], d)
                        if d < 0:
                            rc['n_pg_above_avail'] += 1
                        if pgv.ub is not None and pgv.ub > 0:
                            m = float(pgv.ub) - avail
                            rc['max_ub_minus_avail'] = m if rc['max_ub_minus_avail'] is None else max(rc['max_ub_minus_avail'], m)
                            rc['min_ub_minus_avail'] = m if rc['min_ub_minus_avail'] is None else min(rc['min_ub_minus_avail'], m)
                        if pgv.ub is not None and pgv.value > pgv.ub:
                            rc['n_pg_above_ub'] += 1
                        if pgv.ub is not None and pgv.value > pgv.ub + min(BOUND_RELAX_FACTOR * max(1.0, abs(pgv.ub)), CONSTR_VIOL_TOL):
                            rc['n_pg_above_relaxed_ub'] += 1
    out['res_curtailment'] = rc
    del obj_repn
    return out


def p3_models():
    results = {}
    for eval_dir in MODEL_EVAL_DIRS:
        pkl_rel = f'{eval_dir}/certified_models.pkl'
        pkl = os.path.join(REPO, pkl_rel)
        manifest = _load(f'{eval_dir}/child_manifest_sha256.json')
        sha = _sha256(pkl)
        rec = _record_of(eval_dir)
        levels = _load(f'{eval_dir}/component_levels_terminal.json')
        print(f'[W26] P3 loading {pkl_rel} ({os.path.getsize(pkl)} bytes)', flush=True)
        with open(pkl, 'rb') as handle:
            payload = pickle.load(handle)
        blocks = {}
        gross_recomputed = 0.0
        det_rep_w, det_repn_w, det_proj_w = 0.0, 0.0, 0.0
        for year, days in payload['tso'].items():
            for day, blk in days.items():
                key = f'TSO|{year}|{day}'
                rb = levels['blocks'][key]
                a = _block_analysis(blk, rb['admm_block_weight'], rb['unweighted'])
                blocks[key] = a
        for node, years in payload['dso'].items():
            for year, days in years.items():
                for day, blk in days.items():
                    key = f'DSO|{node}|{year}|{day}'
                    rb = levels['blocks'][key]
                    a = _block_analysis(blk, rb['admm_block_weight'], rb['unweighted'])
                    blocks[key] = a
        for key, a in blocks.items():
            w = levels['blocks'][key]['admm_block_weight']
            gross_recomputed += w * (a['primal_value'] - a['interface_settlement_local'])
            det_rep_w += levels['blocks'][key]['weighted']['detector_penalty_total']
            det_repn_w += w * a['detector_total_repn']
            det_proj_w += w * a['detector_total_projected']
        rcmp = levels['recourse_components']
        summary = {'eval_dir': eval_dir, 'certified_models_path': pkl_rel, 'sha256': sha,
                   'sha256_child_manifest': manifest.get(pkl_rel), 'sha256_matches_manifest': sha == manifest.get(pkl_rel),
                   'status': rec.get('status') if rec else None, 'candidate_key': rec.get('candidate_key') if rec else None,
                   'candidate_label': rec.get('candidate_label') if rec else None,
                   'candidate_canonical': rec.get('candidate_canonical') if rec else None,
                   'component_levels_sha256': _sha256(os.path.join(REPO, eval_dir, 'component_levels_terminal.json')),
                   'n_blocks': len(blocks),
                   'gross_recomputed_from_models': gross_recomputed,
                   'gross_reported': rcmp['gross_operational_cost'],
                   'gross_rel_diff': abs(gross_recomputed - rcmp['gross_operational_cost']) / abs(rcmp['gross_operational_cost']),
                   'detector_total_weighted_reported_blocks': det_rep_w,
                   'detector_total_weighted_reported_recourse': rcmp['detector_penalty_total'],
                   'detector_total_weighted_repn': det_repn_w,
                   'detector_total_weighted_projected_onto_original_bounds': det_proj_w,
                   'bias_weighted_reported_minus_projected': det_rep_w - det_proj_w,
                   'gross_at_projected_slacks': rcmp['gross_operational_cost'] - (det_rep_w - det_proj_w),
                   'bias_over_gross': (det_rep_w - det_proj_w) / rcmp['gross_operational_cost'],
                   'max_abs_recompute_minus_reported': max(a['recompute_max_abs_diff'] for a in blocks.values()),
                   'max_abs_repn_minus_reported_detector': max(
                       abs(a['detector_total_repn'] - levels['blocks'][k]['unweighted']['detector_penalty_total'])
                       for k, a in blocks.items()),
                   'max_abs_penalty_expr_minus_reported_detector': max(
                       abs(a['penalty_expr_value'] - levels['blocks'][k]['unweighted']['detector_penalty_total'])
                       for k, a in blocks.items()),
                   'block_weight_model_equals_reported': all(
                       abs(a['admm_block_weight_model'] - a['admm_block_weight_reported']) <= 1e-9 * a['admm_block_weight_reported']
                       for a in blocks.values()),
                   }
        # family roll-up over blocks
        roll = {}
        for key, a in blocks.items():
            agent = key.split('|')[0]
            for fam, fs in a['families'].items():
                r = roll.setdefault(f'{agent}|{fam}', {
                    'n': 0, 'n_fixed': 0, 'n_lb_eq_ub': 0, 'n_free': 0, 'domains': set(),
                    'lb_min': None, 'lb_max': None, 'ub_min': None, 'ub_max': None, 'n_ub_none': 0, 'n_lb_none': 0,
                    'raw_lower_min': None, 'raw_lower_max': None,
                    'value_min': None, 'value_max': None, 'value_sum': 0.0, 'n_negative': 0,
                    'n_below_lb': 0, 'max_below_lb': 0.0, 'n_below_relaxed_lb': 0, 'n_above_ub': 0,
                    'n_above_relaxed_ub': 0, 'coef_min': None, 'coef_max': None,
                    'objective_coef_min': None, 'objective_coef_max': None,
                    'level_repn_weighted': 0.0, 'level_projected_weighted': 0.0})
                w = levels['blocks'][key]['admm_block_weight']
                for k in ('n', 'n_fixed', 'n_lb_eq_ub', 'n_free', 'n_ub_none', 'n_lb_none', 'n_negative',
                          'n_below_lb', 'n_below_relaxed_lb', 'n_above_ub', 'n_above_relaxed_ub'):
                    r[k] += fs[k]
                r['value_sum'] += fs['value_sum']
                r['max_below_lb'] = max(r['max_below_lb'], fs['max_below_lb'])
                r['domains'].update(fs['domains'])
                for lo, hi in (('lb_min', 'lb_max'), ('ub_min', 'ub_max'), ('raw_lower_min', 'raw_lower_max'),
                               ('value_min', 'value_max'), ('coef_min', 'coef_max'),
                               ('objective_coef_min', 'objective_coef_max')):
                    if fs[lo] is not None:
                        r[lo] = fs[lo] if r[lo] is None else min(r[lo], fs[lo])
                    if fs[hi] is not None:
                        r[hi] = fs[hi] if r[hi] is None else max(r[hi], fs[hi])
                r['level_repn_weighted'] += w * fs['level_repn']
                r['level_projected_weighted'] += w * fs['level_projected']
        for r in roll.values():
            r['domains'] = sorted(r['domains'])
        summary['family_rollup'] = roll
        # floors per (agent, node, family) for P4
        floors = {}
        for key, a in blocks.items():
            parts = key.split('|')
            agent, node = parts[0], (parts[1] if parts[0] == 'DSO' else None)
            for fam, fs in a['families'].items():
                fk = f'{agent}|{node}|{fam}'
                floors[fk] = min(floors.get(fk, 0.0), fs['relaxed_floor'])
        summary['relaxed_floor_unweighted_min_over_blocks'] = floors
        summary['res_curtailment_rollup'] = {
            agent: {
                'penalty_gen_curtailment_param_values': sorted({a['res_curtailment']['penalty_gen_curtailment_param']
                                                                for k, a in blocks.items() if k.startswith(agent)}),
                'min_avail_minus_pg': min((a['res_curtailment']['min_avail_minus_pg'] for k, a in blocks.items()
                                           if k.startswith(agent) and a['res_curtailment']['min_avail_minus_pg'] is not None), default=None),
                'n_pg_above_avail': sum(a['res_curtailment']['n_pg_above_avail'] for k, a in blocks.items() if k.startswith(agent)),
                'n_entries': sum(a['res_curtailment']['n_entries'] for k, a in blocks.items() if k.startswith(agent)),
                'max_ub_minus_avail': max((a['res_curtailment']['max_ub_minus_avail'] for k, a in blocks.items()
                                           if k.startswith(agent) and a['res_curtailment']['max_ub_minus_avail'] is not None), default=None),
                'min_ub_minus_avail': min((a['res_curtailment']['min_ub_minus_avail'] for k, a in blocks.items()
                                           if k.startswith(agent) and a['res_curtailment']['min_ub_minus_avail'] is not None), default=None),
                'n_pg_above_ub': sum(a['res_curtailment']['n_pg_above_ub'] for k, a in blocks.items() if k.startswith(agent)),
                'n_pg_above_relaxed_ub': sum(a['res_curtailment']['n_pg_above_relaxed_ub'] for k, a in blocks.items() if k.startswith(agent)),
            } for agent in ('TSO', 'DSO')}
        summary['solver_options_build_time'] = sorted({json.dumps(a['solver_options_build_time'], sort_keys=True)
                                                       for a in blocks.values()})
        summary['active_objectives'] = sorted({a['active_objective'] for a in blocks.values()})
        summary['blocks'] = blocks
        results[eval_dir] = summary
        del payload, blocks
        gc.collect()
        print(f'[W26] P3 done {eval_dir}: gross_rel_diff={summary["gross_rel_diff"]:.3e} '
              f'recompute_max_abs={summary["max_abs_recompute_minus_reported"]:.3e} '
              f'bias_w={summary["bias_weighted_reported_minus_projected"]:.4f}', flush=True)
    return results


def ageing_independence_check():
    """Slack declarations / bounds / penalty helpers are in network.py, model_construction_helpers.py,
    definitions.py, helper_functions.py (+ the level builders). List every commit touching them after the earliest
    P515S44 model-set launch, and whether any added/removed line mentions a slack/penalty/bound token."""
    starts = [_load(f'{d}/launch.json').get('started_utc') for d in MODEL_EVAL_DIRS]
    earliest = min(starts)
    files = ['network.py', 'model_construction_helpers.py', 'definitions.py', 'helper_functions.py',
             'shared_resources_planning.py', 'p515_g_g1_g4_admm_gates.py', 'network_data.py', 'network_parameters.py']
    log = _git(['log', f'--since={earliest}', '--format=%H %cI %s', '--'] + files).strip().split('\n')
    tokens = ('slack_', 'PENALTY_', '_penalty(', 'bounds', 'setlb', 'setub', 'objective_function_rule',
              '_get_local_detector_components', '_s31_block_components')
    commits = []
    for line in [ln for ln in log if ln]:
        h = line.split(' ', 1)[0]
        diff = _git(['show', '--format=', h, '--'] + files)
        changed = [ln for ln in diff.split('\n') if (ln.startswith('+') or ln.startswith('-'))
                   and not ln.startswith('+++') and not ln.startswith('---')]
        hits = [ln[:200] for ln in changed if any(t in ln for t in tokens)]
        commits.append({'commit': line, 'n_changed_lines': len(changed), 'token_hits': hits})
    # the ESSO ageing parameters live in shared_energy_storage_data.py / SRP1_ESS_Params.json -- neither declares
    # a TSO/DSO slack; record whether they define any of the slack variable names.
    esso_defines = {}
    for rel in ('shared_energy_storage_data.py',):
        esso_defines[rel] = [n for fam in FAMILIES.values() for n in fam if _lines(rel, f'model.{n} = ')]
    return {'earliest_model_set_launch_utc': earliest, 'files_checked': files, 'tokens': tokens,
            'commits_since': commits,
            'esso_module_defines_tso_dso_slack_vars': esso_defines}


# ======================================================================================================================
#  P4 classification and Q effect
# ======================================================================================================================
def p4(inv, models):
    floors = {}
    for s in models.values():
        for k, v in s['relaxed_floor_unweighted_min_over_blocks'].items():
            floors[k] = min(floors.get(k, 0.0), v)
    detector_comps = set(DETECTOR)
    below, tested = [], 0
    worst_ratio = {}
    for r in inv['negative_rows']:
        fidx, bkey, agent, node, comp, u, w = r
        if comp not in detector_comps:
            continue
        fk = f'{agent}|{node if agent == "DSO" else None}|{comp}'
        fl = floors.get(fk)
        tested += 1
        if fl is None or u < fl - FLOOR_TOL_REL * max(1.0, abs(fl)):
            below.append({'row': r, 'floor': fl})
        if fl:
            worst_ratio[fk] = max(worst_ratio.get(fk, 0.0), u / fl)
    # positive detector-family levels: implied mean slack value per FREE entry = level / free_coef_sum,
    # free_coef_sum = -floor / 1e-8 (lb = 0 for every slack, P3)
    for r in inv['positive_detector_rows']:
        fk = f'{r["agent"]}|{r["node_id"] if r["agent"] == "DSO" else None}|{r["component"]}'
        fl = floors.get(fk)
        r['free_coef_sum_from_models'] = (-fl / BOUND_RELAX_FACTOR) if fl else None
        r['implied_mean_value_per_free_entry'] = (r['unweighted'] / (-fl / BOUND_RELAX_FACTOR)) if fl else None
    # detector totals of certified records vs x = 0
    x0 = _load(f'{X0_DIR}/component_levels_terminal.json')
    x0rec = _record_of(X0_DIR)
    det0 = x0['recourse_components']['detector_penalty_total']
    q0 = x0rec['certified_cost']
    rows = []
    for f in inv['files']:
        if f['status'] != 'certified':
            continue
        d = os.path.dirname(f['path'])
        rec = _record_of(d)
        lv = _load(f['path'])
        det = lv['recourse_components']['detector_penalty_total']
        qx = rec['certified_cost']
        bar = (rec.get('bar') or {}).get('value')
        dd = det - det0
        value = q0 - qx
        rows.append({'record': d, 'campaign_id': rec.get('campaign_id'), 'candidate_label': rec.get('candidate_label'),
                     'candidate_key': rec.get('candidate_key'), 'ageing_label': rec.get('ess_ageing_baseline_label'),
                     'certified_cost': qx, 'terminal_salvage_value': lv['recourse_components'].get('terminal_salvage_value'),
                     'detector_penalty_total_weighted': det, 'detector_over_Q': det / qx,
                     'd_detector_vs_x0': dd, 'value_Q0_minus_Qx': value, 'bar': bar,
                     'abs_d_detector_over_bar': (abs(dd) / bar) if bar else None,
                     'abs_d_detector_over_abs_value': (abs(dd) / abs(value)) if value else None})
    nonzero_value = [r for r in rows if r['candidate_key'] != X0_KEY]
    base = [r for r in rows if r['record'] == BASELINE_DIR]
    return {
        'floor_test': {
            'formula': ('floor(agent, node, component) = min over the five model sets and all blocks of '
                        'sum_{free slack entries} coef * (lb - min(1e-8 * max(1, |lb|), 1e-4)); free = not fixed '
                        'and lb < ub; every reported negative detector level must satisfy level >= floor '
                        f'- {FLOOR_TOL_REL} * max(1, |floor|)'),
            'floors_unweighted': floors, 'n_negative_detector_levels_tested': tested,
            'n_below_floor': len(below), 'below_floor_rows': below[:200],
            'max_level_over_floor_ratio': worst_ratio,
            'note': ('ratio = level / floor in (0, 1]: 1 means every free slack of that family sits at its relaxed '
                     'bound; structure (which entries are free) is read from the P515S44 model sets; a record '
                     'whose structure had MORE free entries could legitimately go below this floor')},
        'x0_record': X0_DIR, 'x0_candidate_key': x0rec['candidate_key'], 'x0_detector_penalty_total': det0, 'Q0': q0,
        'certified_rows': rows,
        'max_abs_d_detector_vs_x0': max(abs(r['d_detector_vs_x0']) for r in nonzero_value),
        'max_abs_d_detector_over_bar': max(r['abs_d_detector_over_bar'] for r in nonzero_value if r['abs_d_detector_over_bar'] is not None),
        'max_abs_detector_over_Q': max(abs(r['detector_over_Q']) for r in rows),
        'baseline_row': base[0] if base else None,
        'formulas': {'d_detector_vs_x0': 'det(x) - det(0), det = recourse_components.detector_penalty_total (weighted, EUR)',
                     'value': 'Q(0) - Q(x), Q = evaluation_record.certified_cost',
                     'effect_on_value': ('Q_reported(x) = Q_projected(x) + bias(x), bias = level_reported - '
                                         'level_projected (P3); value_reported = value_projected + bias(0) - bias(x). '
                                         'Where no slack value is positive (every P515S44 model set, P3) bias = det, '
                                         'so value_reported - value_projected = det(0) - det(x) = -d_detector_vs_x0. '
                                         'Where a block carries positive slack values (P1 positive_detector_rows) the '
                                         'report does not separate the positive and negative parts, so bias(x) is not '
                                         'determined from the report for that record')}}


# ======================================================================================================================
#  Markdown
# ======================================================================================================================
def _markdown(res):
    a = []
    w = a.append
    w(f'# P5.15 Addendum 32 Q6 (W26) -- negative penalty levels\n\n{STAGE}\n\n{OBJ_CONV}\n')
    w(f'Git HEAD {res["git_HEAD"]}. Zero solves: armed SolveProfileGuard(permitted=()), counts '
      f'{res["solve_profile_guard"]["counts"]}, verify(0) failures {res["solve_profile_guard"]["verify_0_failures"]}.\n')
    c = res.get('classification', {})
    w('## Classification\n')
    w(f'**{c.get("verdict")}**\n\n{c.get("statement")}\n')
    inv = res['P1_inventory']
    w('## P1 Inventory\n')
    w(f'Scope: {inv["scope"]}. Files: {inv["n_files"]}; status counts {inv["status_counts"]}. Negative block levels '
      f'(rows): {len(inv["negative_rows"])} (full list in the JSON, `P1_inventory.negative_rows`).\n')
    w('| agent | component | block levels | negative | negative (certified) | positive | min unweighted | '
      'max unweighted | min weighted | max weighted |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for r in inv['per_agent_component']:
        if r['n_negative'] or r['n_positive'] or r['component'] in DETECTOR:
            w(f'| {r["agent"]} | {r["component"]} | {r["n_block_levels"]} | {r["n_negative"]} | '
              f'{r["n_negative_certified"]} | {r["n_positive"]} | {r["min_unweighted"]:.6g} | {r["max_unweighted"]:.6g} | '
              f'{r["min_weighted"]:.6g} | {r["max_weighted"]:.6g} |')
    w(f'\nSigned zeros (-0.0, not negative): {inv["signed_zero_counts"]}.\n')
    w(f'Positive detector-family block levels ({len(inv["positive_detector_rows"])}; {inv["positive_detector_note"]}):\n')
    w('| file | status | block | component | unweighted | weighted | implied mean value per free entry (P4) | '
      'terminal-cycle solve was a recovery | network-failure rows, same block (cycle:class) |')
    w('|---|---|---|---|---|---|---|---|---|')
    for r in inv['positive_detector_rows']:
        w(f'| {r["path"].split("Results/")[1].rsplit("/", 1)[0]} | {r["status"]} | {r["block"]} | {r["component"]} | '
          f'{r["unweighted"]:.6g} | {r["weighted"]:.6g} | {r.get("implied_mean_value_per_free_entry")} | '
          f'{r["terminal_cycle_solve_was_a_recovery"]} | '
          + ', '.join(f'{m["cycle"]}:{m["class"]}' for m in r['network_failure_rows_same_block']) + ' |')
    w('')
    w('## P2 Code trace\n')
    for k, v in res['P2_code_trace']['citations'].items():
        w(f'- {k}: `{v}`')
    w(f'\n{res["P2_code_trace"]["reading"]}\n')
    s = res['P2_code_trace']['ipopt_option_search']
    w(f'IPOPT bound-relaxation option search ({s["scope"]}; {s["n_files_searched"]} files; needles {list(s["needles"])}): '
      f'{len(s["hits_in_production_or_case_files"])} hits in production modules or case files; all hits: '
      f'{len(s["all_hits"])} (listed in the JSON). IPOPT documentation query: '
      f'{res["P2_code_trace"]["ipopt_documentation_query"]["result"]}\n')
    w('## P3 Model read-back (P515S44 certified model sets)\n')
    ai = res['ageing_independence']
    w(f'Ageing independence: earliest model-set launch {ai["earliest_model_set_launch_utc"]}; commits touching '
      f'{ai["files_checked"]} since then: ' + '; '.join(
          f'{x["commit"][:60]} ({x["n_changed_lines"]} changed lines, slack/penalty/bound token hits: {len(x["token_hits"])})'
          for x in ai['commits_since']) + f'. ESSO module defines TSO/DSO slack vars: {ai["esso_module_defines_tso_dso_slack_vars"]}.\n')
    w('| model set | candidate | sha256 = manifest | gross recomputed / reported - 1 | max abs(recompute - reported) | '
      'detector weighted (reported) | projected | bias | bias / gross |')
    w('|---|---|---|---|---|---|---|---|---|')
    for d, m in res['P3_models'].items():
        w(f'| {d.split("/")[-1]} | {m["candidate_label"]} `{(m["candidate_key"] or "")[:12]}` | {m["sha256_matches_manifest"]} | '
          f'{m["gross_rel_diff"]:.2e} | {m["max_abs_recompute_minus_reported"]:.2e} | '
          f'{m["detector_total_weighted_reported_recourse"]:.4f} | {m["detector_total_weighted_projected_onto_original_bounds"]:.4f} | '
          f'{m["bias_weighted_reported_minus_projected"]:.4f} | {m["bias_over_gross"]:.3e} |')
    first = next(iter(res['P3_models'].values()))
    w(f'\nSlack families, model set {next(iter(res["P3_models"]))} (all blocks; values in p.u. of the variable):\n')
    w('| agent / family | n | fixed | lb=ub | free | domains | lb min..max | ub min..max (None) | value min | value max | '
      'negative | below lb | max below lb | below relaxed lb | above relaxed ub | penalty coef min..max | '
      'objective coef min..max |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, r in first['family_rollup'].items():
        if r['n'] == 0:
            continue
        w(f'| {k} | {r["n"]} | {r["n_fixed"]} | {r["n_lb_eq_ub"]} | {r["n_free"]} | {r["domains"]} | '
          f'{r["lb_min"]}..{r["lb_max"]} | {r["ub_min"]}..{r["ub_max"]} ({r["n_ub_none"]}) | {r["value_min"]} | '
          f'{r["value_max"]} | {r["n_negative"]} | {r["n_below_lb"]} | {r["max_below_lb"]:.3e} | '
          f'{r["n_below_relaxed_lb"]} | {r["n_above_relaxed_ub"]} | {r["coef_min"]}..{r["coef_max"]} | '
          f'{r["objective_coef_min"]}..{r["objective_coef_max"]} |')
    w('\nAll five model sets (roll-up totals):\n')
    w('| model set | slack entries | negative | below original lb | below relaxed lb | above relaxed ub | min value | '
      'lb min | domains |')
    w('|---|---|---|---|---|---|---|---|---|')
    for d, m in res['P3_models'].items():
        rr = m['family_rollup'].values()
        w(f'| {d.split("/")[-1]} | {sum(r["n"] for r in rr)} | {sum(r["n_negative"] for r in rr)} | '
          f'{sum(r["n_below_lb"] for r in rr)} | {sum(r["n_below_relaxed_lb"] for r in rr)} | '
          f'{sum(r["n_above_relaxed_ub"] for r in rr)} | '
          f'{min((r["value_min"] for r in rr if r["value_min"] is not None), default=None)} | '
          f'{min((r["lb_min"] for r in rr if r["lb_min"] is not None), default=None)} | '
          f'{sorted({x for r in rr for x in r["domains"]})} |')
    w('\nRES curtailment definitional (not a slack; objective weight read from the model):\n')
    for d, m in res['P3_models'].items():
        w(f'- {d.split("/")[-1]}: {m["res_curtailment_rollup"]}')
    p = res['P4']
    w('\n## P4 Floor test and effect on Q\n')
    ft = p['floor_test']
    w(f'{ft["formula"]}. Tested {ft["n_negative_detector_levels_tested"]} negative detector block levels; below floor: '
      f'{ft["n_below_floor"]}. Floors (unweighted): {ft["floors_unweighted"]}. Max level/floor ratio: '
      f'{ft["max_level_over_floor_ratio"]}. {ft["note"]}\n')
    w(f'x = 0 record {p["x0_record"]} (`{p["x0_candidate_key"][:12]}`): detector total {p["x0_detector_penalty_total"]:.4f}, '
      f'Q(0) = {p["Q0"]:.4f}. Max |d_det| over certified x != 0 records: {p["max_abs_d_detector_vs_x0"]:.4f}; '
      f'max |d_det| / bar(x): {p["max_abs_d_detector_over_bar"]:.4f}; max |det| / Q: {p["max_abs_detector_over_Q"]:.3e}.\n')
    b = p['baseline_row']
    if b:
        w(f'Baseline (s47_recert:n7_4h_e1, `{b["candidate_key"][:12]}`): detector {b["detector_penalty_total_weighted"]:.4f}, '
          f'd_det vs x0 {b["d_detector_vs_x0"]:.4f}, value {b["value_Q0_minus_Qx"]:.4f}, bar {b["bar"]:.4f}, '
          f'|d_det|/bar {b["abs_d_detector_over_bar"]:.4f}, |d_det|/|value| {b["abs_d_detector_over_abs_value"]:.3e}.\n')
    w(f'Formulas: {p["formulas"]}\n')
    w('| record | label | key | Q(x) | detector | d_det vs x0 | value | bar | abs(d_det)/bar |')
    w('|---|---|---|---|---|---|---|---|---|')
    for r in p['certified_rows']:
        w(f'| {"/".join(r["record"].split("/")[3:5])}/{r["record"].split("/")[-1][:16]} | {r["candidate_label"]} | '
          f'`{r["candidate_key"][:12]}` | {r["certified_cost"]:.2f} | '
          f'{r["detector_penalty_total_weighted"]:.2f} | {r["d_detector_vs_x0"]:.2f} | {r["value_Q0_minus_Qx"]:.2f} | '
          f'{r["bar"] if r["bar"] is None else round(r["bar"], 2)} | '
          f'{"" if r["abs_d_detector_over_bar"] is None else round(r["abs_d_detector_over_bar"], 4)} |')
    w(f'\nChecks: {len(res["checks"])}; failed: {res["failed_checks"]}; errors: {sorted(res["errors"])}.\n')
    return '\n'.join(a) + '\n'


# ======================================================================================================================
#  main
# ======================================================================================================================
def _manifest():
    files = {}
    for name in sorted(os.listdir(OUT_DIR)):
        if name in (MANIFEST_NAME, LOCK_NAME):
            continue
        path = os.path.join(OUT_DIR, name)
        files[os.path.relpath(path, REPO)] = {'sha256': _sha256(path), 'bytes': os.path.getsize(path)}
    res = _load(os.path.join(OUT_DIR, RESULTS_NAME))
    inputs = {f['path']: f['sha256'] for f in res['P1_inventory']['files']}
    for d, m in res['P3_models'].items():
        inputs[m['certified_models_path']] = m['sha256']
    manifest = {'stage': STAGE, 'generated_utc': _utc(), 'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
                'instance_candidate_keys': {'x0': X0_KEY, 'baseline': BASELINE_KEY,
                                            **{d: m['candidate_key'] for d, m in res['P3_models'].items()}},
                'script': {'path': os.path.basename(__file__), 'sha256': _sha256(os.path.abspath(__file__))},
                'files': files, 'evidence_inputs_sha256': inputs}
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite the manifest: {path}')
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    print(f'[W26] manifest: {len(files)} files, {len(inputs)} inputs -> {os.path.relpath(path, REPO)}', flush=True)


def main():
    started = _utc()
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'output directory must be created by the launcher: {OUT_DIR}')
    res_path, md_path = os.path.join(OUT_DIR, RESULTS_NAME), os.path.join(OUT_DIR, MD_NAME)
    for p in (res_path, md_path):
        if os.path.exists(p):
            raise SystemExit(f'refusing to overwrite (write-once): {p}')
    lock = os.path.join(OUT_DIR, LOCK_NAME)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # refuses a concurrent copy of this script
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    print(f'[W26] start {started}; out {OUT_DIR}', flush=True)
    res = {'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started,
           'git_HEAD': _git(['rev-parse', 'HEAD']).strip(), 'objective_convention': OBJ_CONV, 'out_dir': OUT_DIR,
           'constants': {'PENALTY_VOLTAGE_SQUARED': PENALTY_VOLTAGE_SQUARED, 'PENALTY_NODE_BALANCE': PENALTY_NODE_BALANCE,
                         'PENALTY_CURRENT': PENALTY_CURRENT, 'PENALTY_FLEXIBILITY': PENALTY_FLEXIBILITY,
                         'PENALTY_ESS_BALANCE': PENALTY_ESS_BALANCE,
                         'PENALTY_SHARED_ESS_BALANCE': PENALTY_SHARED_ESS_BALANCE,
                         'PENALTY_GENERATION_CURTAILMENT': PENALTY_GENERATION_CURTAILMENT,
                         'EQUALITY_TOLERANCE': EQUALITY_TOLERANCE,
                         'IPOPT_bound_relax_factor_default': BOUND_RELAX_FACTOR,
                         'IPOPT_constr_viol_tol_default': CONSTR_VIOL_TOL},
           'tolerances_stated_before_run': {'LEVEL_TOL_REL': LEVEL_TOL_REL, 'GROSS_TOL_REL': GROSS_TOL_REL,
                                            'FLOOR_TOL_REL': FLOOR_TOL_REL}}
    checks, errors = {}, {}
    try:
        inv = p1_inventory()
        res['P1_inventory'] = inv
        print(f'[W26] P1 files={inv["n_files"]} negative rows={len(inv["negative_rows"])}', flush=True)
        res['P2_code_trace'] = p2_code_trace()
        res['ageing_independence'] = ageing_independence_check()
        models = p3_models()
        res['P3_models'] = models
        res['P4'] = p4(inv, models)

        # ---- checks ----
        for d, m in models.items():
            tag = d.split('/')[-1]
            checks[f'{tag}:sha256_matches_manifest'] = m['sha256_matches_manifest']
            checks[f'{tag}:status_certified'] = m['status'] == 'certified'
            checks[f'{tag}:recompute_equals_reported'] = all(
                abs(v) <= LEVEL_TOL_REL * max(1.0, abs(a['recomputed_production'][k]))
                for a in m['blocks'].values() for k, v in a['recompute_minus_reported'].items())
            checks[f'{tag}:repn_equals_reported_detector'] = m['max_abs_repn_minus_reported_detector'] <= 1e-9
            checks[f'{tag}:penalty_expr_equals_reported_detector'] = m['max_abs_penalty_expr_minus_reported_detector'] <= 1e-9
            checks[f'{tag}:gross_reconciles'] = m['gross_rel_diff'] <= GROSS_TOL_REL
            checks[f'{tag}:block_weights_match'] = m['block_weight_model_equals_reported']
            roll = m['family_rollup']
            checks[f'{tag}:all_slack_lb_zero'] = all((r['lb_min'] in (0.0, None) and r['lb_max'] in (0.0, None))
                                                     for r in roll.values() if r['n'])
            checks[f'{tag}:no_slack_lb_none'] = all(r['n_lb_none'] == 0 for r in roll.values())
            checks[f'{tag}:all_slack_domain_nonnegative'] = all(r['domains'] == ['NonNegativeReals']
                                                                for r in roll.values() if r['n'])
            checks[f'{tag}:no_value_below_relaxed_lb'] = all(r['n_below_relaxed_lb'] == 0 for r in roll.values())
            checks[f'{tag}:no_value_above_relaxed_ub'] = all(r['n_above_relaxed_ub'] == 0 for r in roll.values())
            checks[f'{tag}:penalty_coefs_positive'] = all((r['coef_min'] is None or r['coef_min'] > 0)
                                                          for r in roll.values())
            checks[f'{tag}:objective_coefs_positive'] = all((r['objective_coef_min'] is None or r['objective_coef_min'] > 0)
                                                            for r in roll.values())
            checks[f'{tag}:no_positive_slack_values'] = all((r['value_max'] is None or r['value_max'] <= 0.0)
                                                            for k, r in roll.items() if 'orphan' not in k)
        checks['floor_test_no_level_below_floor'] = res['P4']['floor_test']['n_below_floor'] == 0
        checks['x0_key'] = res['P4']['x0_candidate_key'] == X0_KEY
        checks['baseline_present'] = res['P4']['baseline_row'] is not None and \
            res['P4']['baseline_row']['candidate_key'] == BASELINE_KEY
        checks['no_bound_relax_option_in_production_or_case_files'] = \
            len(res['P2_code_trace']['ipopt_option_search']['hits_in_production_or_case_files']) == 0
        checks['no_slack_token_change_since_model_sets'] = all(
            len(c['token_hits']) == 0 for c in res['ageing_independence']['commits_since'])

        # ---- classification (data-driven; stated so it can fail) ----
        slack_ok = all(checks[k] for k in checks if any(t in k for t in (
            'all_slack_lb_zero', 'no_slack_lb_none', 'all_slack_domain_nonnegative', 'no_value_below_relaxed_lb',
            'penalty_coefs_positive', 'recompute_equals_reported', 'repn_equals_reported_detector')))
        floor_ok = checks['floor_test_no_level_below_floor']
        if slack_ok and floor_ok:
            verdict = ('NEITHER (a) NOR (b) AS POSED: every slack feeding a negative level is correctly bounded '
                       '(lb = 0, NonNegativeReals) and the report carries no sign convention; the negative levels are '
                       'the solver\'s bound-relaxation residue (IPOPT bound_relax_factor = 1e-8, '
                       'honor_original_bounds = no), reported and priced unclipped')
        else:
            verdict = 'STOP: at least one slack-bound or floor check failed -- see failed checks (candidate (b))'
        m0 = next(iter(models.values()))
        b = res['P4']['baseline_row']
        pos = inv['positive_detector_rows']
        checks['res_curtailment_weight_zero_all_blocks'] = all(
            m['res_curtailment_rollup'][ag]['penalty_gen_curtailment_param_values'] == [0.0]
            for m in models.values() for ag in ('TSO', 'DSO'))
        checks['no_pg_above_relaxed_ub'] = all(
            m['res_curtailment_rollup'][ag]['n_pg_above_relaxed_ub'] == 0 for m in models.values() for ag in ('TSO', 'DSO'))
        res['classification'] = {
            'verdict': verdict,
            'statement': (
                'Code: every D-row level is PENALTY * (slack_up + slack_down) with a positive coefficient, summed '
                'without any reference subtraction or credit sign (P2), so (a) is excluded. Models: every slack '
                'entry has lower bound exactly 0 (domain NonNegativeReals; raw declared bounds also 0), and no '
                'terminal value lies below IPOPT\'s relaxed bound lb - 1e-8 (P3 checks, the five P515S44 model sets); '
                'for the P515S45-S47 records, whose models are not persisted, (b) is excluded by inference: the '
                'declaring code is unchanged since the model sets were built (ageing_independence) and every '
                'negative reported level lies at or above the relaxed-bound floor (P4 floor test). The '
                'negative values are within [-1e-8, 0): IPOPT relaxes every bound by bound_relax_factor = 1e-8 '
                '(not overridden anywhere in the searched scope) and, with honor_original_bounds = no, returns the '
                'final point unprojected; Pyomo loads it as-is; a slack whose penalty drives it to its lower bound '
                'sits at the relaxed bound. The report (and the objective re-evaluation behind Q) sums those values '
                'unclipped. Size: the projected-onto-bounds level differs from the reported level by exactly the '
                f'reported level in every model set (no positive slack value); e.g. {m0["candidate_label"]}: bias '
                f'{m0["bias_weighted_reported_minus_projected"]:.4f} EUR = {m0["bias_over_gross"]:.3e} of gross. '
                'Absolute Q is therefore biased LOW by the negative slack values (|detector_penalty_total| <= '
                f'{res["P4"]["max_abs_detector_over_Q"]:.2e} of Q over all certified records); value differences '
                'move by d_det = det(x) - det(0), which is not identically 0 but at most '
                f'{res["P4"]["max_abs_d_detector_vs_x0"]:.2f} EUR (max |d_det|/bar = '
                f'{res["P4"]["max_abs_d_detector_over_bar"]:.4f})'
                + (f'; baseline n7_4h_e1: d_det = {b["d_detector_vs_x0"]:.2f} EUR against value '
                   f'{b["value_Q0_minus_Qx"]:.2f} and bar {b["bar"]:.2f}.' if b else '.')
                + f' Positive detector-family block levels: {len(pos)} in {len({r["path"] for r in pos})} files; '
                  f'{sum(1 for r in pos if r["terminal_cycle_solve_was_a_recovery"])} of them sit in a block whose '
                  'terminal-cycle solve was a recovery retry (network_failures log); implied mean slack value per free '
                  'entry = level / free_coef_sum: '
                  + ', '.join(sorted({f'{r["component"]} {r["implied_mean_value_per_free_entry"]:.3e}' for r in pos
                                      if r['implied_mean_value_per_free_entry'] is not None}))
                  + ' (p.u.; per-entry values are not available -- these models are not persisted). '
                  'RES curtailment definitional (negative in TSO blocks): not a slack; pg\'s declared upper bound is '
                  'pg_avail + EQUALITY_TOLERANCE (1e-5 p.u.), so pg_avail - pg >= -1e-5 - relaxation by '
                  'construction (P3: no pg above its relaxed upper bound); its objective weight '
                  'penalty_gen_curtailment is 0 in every block, so it does not enter Q.')}
    except Exception:  # noqa: BLE001 -- recorded; the run then fails
        errors['run'] = traceback.format_exc()
        print(errors['run'], file=sys.stderr, flush=True)

    failures = GUARD.verify(0)
    res.update({'checks': checks, 'failed_checks': sorted(k for k, v in checks.items() if not v), 'errors': errors,
                'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts), 'verify_0_failures': failures},
                'ended_utc': _utc()})
    res['all_pass'] = bool(checks) and not res['failed_checks'] and not errors and not failures

    def _default(o):
        if isinstance(o, (tuple, set)):
            return list(o)
        return str(o)
    with open(res_path, 'x') as handle:
        json.dump(res, handle, indent=1, default=_default)
    if not errors:
        with open(md_path, 'x') as handle:
            handle.write(_markdown(res))
    os.remove(lock)
    GUARD.uninstall()
    print(f"[W26] checks={len(checks)} failed={res['failed_checks']} errors={sorted(errors)} "
          f"guard={dict(GUARD.counts)} verify0_failures={failures} all_pass={res['all_pass']}", flush=True)
    if not res['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        GUARD.uninstall()
        _manifest()
        sys.exit(0)
    main()
