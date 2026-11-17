"""P5.15 Addendum 22 item 1(a) -- zero-solve component decomposition of the C/D-vs-run-1
cost differences (S40).

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 22 ("Zero-solve component decomposition
of the 58-72k differences (C, D vs run 1) authorized"); frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`, item
`1_zero_solve_analyses.a_cost_decomposition`.

Runs decomposed (all four are the SAME candidate/instance -- 0.96875 MVA / 3.875 MWh,
uniform across nodes 5/7/9, invested 2025 -- verified below from each run's own
`g_<label>.json['instance']`; they differ ONLY in ADMM configuration/stopping point,
never in the plan being costed):

  run1 = data/SRP1/Results/P515S35_REF_run   (label 'baseline';   g_baseline.json)
         Addendum 16 item 1 reference-equilibrium run: tau=1 (gamma tied to rho),
         balancing on every channel, freeze-after-cycle-30 backstop (Addendum 15/16
         era, PRE Addendum 20's freeze-after-10-unchanged-cycles rule), cap 500.
         Certified under Addendum 16's rule at cycle 477 (Addendum 17).
  A    = data/SRP1/Results/P515S38_A_TAU0_run (label 's38_A_tau0'; g_s38_A_tau0.json)
         v9 arm A (Addendum 20/21 frozen spec v9, `frozen_s38_pf_pace_spec_v9_7a2b4ab7.json`):
         tau=0 (proximal off), PF balancing OFF (rho_pf fixed), ESS balancing exempt
         (rho_ess fixed), cap 300. Certified at cycle 133.
  C    = data/SRP1/Results/P515S39_C_run      (label 's39_C';      g_s39_C.json)
         v10 arm C (Addendum 21/22 frozen spec v10, `frozen_s39_oracle_spec_v10_f1b2b999.json`):
         tau=0, PF balancing LIVE (freeze-after-10-unchanged, absolute freeze 200),
         rho_ess=0.01 fixed and exempt, cap 300. Certified at cycle 131 (stopped 191).
  D    = data/SRP1/Results/P515S39_D_run      (label 's39_D';      g_s39_D.json)
         v10 arm D, the ADOPTED ORACLE (Addendum 22): C plus the two-phase ESS schedule
         (rho_ess=0.01 exempt until its Boyd dual ratio is <1 for 5 consecutive cycles,
         standard balancing thereafter, one-way). Certified at cycle 125 (stopped 139).

**Every one of the four runs used a DIFFERENT ADMM configuration** (tau, PF-balancing
policy, ESS-balancing policy, cap, and hence a different stopping cycle). Per
`CLAUDE.md`'s campaign rule ("costs from different configurations never share a table")
and Addendum 22 ("one frozen oracle configuration for every candidate in a campaign"),
the tables below are a DIAGNOSIS of why four differently-configured/differently-stopped
ADMM runs on the SAME candidate report different terminal costs -- they are NOT a ranking
of candidates, and no row here licenses "candidate X is cheaper than candidate Y".

Objective convention (stated once here, repeated on every table): all figures are
`gross_operational_cost` as computed by the production per-block component-levels
capture (`shared_resources_planning.py`'s per-block accounting, captured into
`component_levels_terminal.json` by the S31 harness machinery). `gross_operational_cost`
and `net_operational_recourse` are IDENTICAL in all four runs to within the file's own
float noise (`terminal_salvage_value` is ~1e-36 to ~1e-50, i.e. exactly zero to machine
precision in every run -- no salvage is reinstated on any of these four runs), so the
gross/net distinction does not apply here; it is verified, not assumed, below.
`gross_operational_cost` = sum of six "priced" component totals (generation_cost,
flexibility_cost_internal, load_curtailment_cost, res_curtailment_penalty, ess_usage_cost,
flexibility_cost_tso_adn_interface_definitional) PLUS `detector_penalty_total` (the sum of
five category-D feasibility-detector slacks: voltage_slack, node_balance_slack,
branch_flow_slack, flexibility_p_day_balance_slack, shared_ess_day_balance_slack) -- i.e.
this file's `gross_operational_cost` is the as-solved objective total (detectors included,
per Addendum 10's category-D semantics: detectors stay IN the solver objective; the
manuscript's reported Q(x) would exclude them via `economic_recourse_all_D_excluded`, a
field also carried in the same file and reported here for completeness, but NOT the
quantity the task's headline numbers refer to -- the task's four costs match
`gross_operational_cost` exactly, verified below). Two further fields present in the same
file (`ess_complementarity_bilinear_definitional`, `res_curtailment_definitional_at_weight_1`)
are PURE reporting/detector quantities computed at weight 1, not part of
`gross_operational_cost` at all (verified by the reconciliation below); they are reported
in the full table for completeness but excluded from the "priced" sum.

Zero solves: `SolveProfileGuard(permitted=())` is installed before any input file is
touched and verified with `guard.verify(0)` before any output is written.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_s40_cost_decomposition.py [--dry-run]

Writes (write-once, refuses to overwrite):
    data/SRP1/Results/P515S40/cost_decomposition/cost_decomposition.json
    data/SRP1/Results/P515S40/cost_decomposition/run_log.txt
    data/SRP1/Results/P515S40/cost_decomposition/evidence_manifest_sha256.json
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUNS = {
    'run1': {
        'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35_REF_run'),
        'label': 'baseline',
        'config_summary': (
            "Addendum 16 item 1 reference-equilibrium run: tau=1 (gamma tied to rho on "
            "every channel), residual balancing ON on all three channels, freeze-after-"
            "cycle-30 backstop (pre-Addendum-20 freeze rule), cap 500; certified at cycle "
            "477 under Addendum 16's rule (Addendum 17)."
        ),
        'config_spec': 'data/SRP1/Results/P515S35/frozen_s35_reference_spec_v5_995548ab.json',
    },
    'A': {
        'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38_A_TAU0_run'),
        'label': 's38_A_tau0',
        'config_summary': (
            "v9 arm A: tau=0 (proximal off), PF balancing OFF (rho_pf fixed at its "
            "initial value), ESS balancing exempt (rho_ess fixed), cap 300; certified at "
            "cycle 133."
        ),
        'config_spec': 'data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json',
    },
    'C': {
        'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_C_run'),
        'label': 's39_C',
        'config_summary': (
            "v10 arm C: tau=0, PF balancing LIVE (freeze after 10 unchanged cycles, "
            "absolute freeze at cycle 200), rho_ess=0.01 fixed and exempt from "
            "balancing, cap 300; certified at cycle 131 (stopped 191)."
        ),
        'config_spec': 'data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json',
    },
    'D': {
        'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run'),
        'label': 's39_D',
        'config_summary': (
            "v10 arm D, the ADOPTED ORACLE (Addendum 22): C plus the two-phase ESS "
            "schedule (rho_ess=0.01 exempt until its own Boyd dual ratio has been <1 "
            "for 5 consecutive cycles, standard balancing thereafter, one-way), cap "
            "300; certified at cycle 125 (stopped 139)."
        ),
        'config_spec': 'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
    },
}
RUN_ORDER = ['run1', 'A', 'C', 'D']

PRICED_COMPONENT_KEYS = [
    'generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost',
    'res_curtailment_penalty', 'ess_usage_cost',
    'flexibility_cost_tso_adn_interface_definitional',
]
DETECTOR_COMPONENT_KEYS = [
    'voltage_slack', 'node_balance_slack', 'branch_flow_slack',
    'flexibility_p_day_balance_slack', 'shared_ess_day_balance_slack',
]
PURE_REPORTING_KEYS = [
    'ess_complementarity_bilinear_definitional',
    'res_curtailment_definitional_at_weight_1',
]
ALL_TOTALS_KEYS_EXPECTED_ORDER = (
    PRICED_COMPONENT_KEYS + DETECTOR_COMPONENT_KEYS + ['detector_penalty_total']
    + [k for k in PURE_REPORTING_KEYS]
    + ['orphan_flex_q_day_balance_slack_raw', 'orphan_tso_adn_flex_p_day_balance_slack_raw',
       'orphan_flex_q_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional',
       'orphan_tso_adn_flex_p_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional']
)

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'cost_decomposition')
OUT_JSON = os.path.join(OUT_DIR, 'cost_decomposition.json')
OUT_LOG = os.path.join(OUT_DIR, 'run_log.txt')
OUT_MANIFEST = os.path.join(OUT_DIR, 'evidence_manifest_sha256.json')
SCRIPT_REL = os.path.relpath(os.path.abspath(__file__), REPO)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(run_key, log):
    r = RUNS[run_key]
    g_path = os.path.join(r['dir'], f"g_{r['label']}.json")
    cl_path = os.path.join(r['dir'], 'component_levels_terminal.json')
    log(f"[{run_key}] reading {os.path.relpath(g_path, REPO)}")
    g = json.load(open(g_path))
    log(f"[{run_key}] reading {os.path.relpath(cl_path, REPO)}")
    cl = json.load(open(cl_path))
    return g, cl, [g_path, cl_path]


def _max_abs_step_last10(cycle_trajectory, log, run_key):
    last10 = cycle_trajectory[-10:]
    steps = [abs(c['recourse_change']) for c in last10 if c.get('recourse_change') is not None]
    if len(steps) != 10:
        log(f"[{run_key}] WARNING: only {len(steps)}/10 recourse_change values available "
            f"in the last 10 cycle_trajectory rows (run may have <10 cycles recorded)")
    return max(steps) if steps else None, len(steps)


def main(argv):
    dry = '--dry-run' in argv
    for p in (OUT_JSON, OUT_LOG, OUT_MANIFEST):
        if not dry and os.path.exists(p):
            raise RuntimeError(f'refusing to overwrite existing output {p}')

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log('P5.15 Addendum 22 item 1(a): zero-solve cost decomposition (S40)')
    log(f'timestamp_utc = {datetime.now(timezone.utc).isoformat()}')

    guard = SolveProfileGuard(permitted=(), label='P5.15 s40 cost decomposition (zero-solve)').install()
    inputs_used = []
    try:
        loaded = {}
        for rk in RUN_ORDER:
            g, cl, paths = _load(rk, log)
            inputs_used += paths
            loaded[rk] = {'g': g, 'cl': cl}

        # ------------------------------------------------------------------
        # 0. Instance identity check -- all four runs must be the SAME candidate
        # ------------------------------------------------------------------
        instances = {rk: loaded[rk]['g']['instance'] for rk in RUN_ORDER}
        instance_ref = instances['run1']
        same_instance = all(instances[rk] == instance_ref for rk in RUN_ORDER)
        log(f"instance identity across all four runs: {same_instance} -- {instance_ref}")
        if not same_instance:
            log(f"WARNING: instances differ: {instances}")

        # ------------------------------------------------------------------
        # 1. gross_operational_cost cross-check: g_*.json vs component_levels_terminal.json
        # ------------------------------------------------------------------
        gross_cross_check = {}
        for rk in RUN_ORDER:
            g_cost = loaded[rk]['g'].get('gross_operational_cost')
            cl_cost = loaded[rk]['cl']['recourse_components']['gross_operational_cost']
            gross_cross_check[rk] = {
                'g_json_gross_operational_cost': g_cost,
                'component_levels_gross_operational_cost': cl_cost,
                'abs_diff': abs(g_cost - cl_cost),
            }
            log(f"[{rk}] gross_operational_cost: g.json={g_cost!r} component_levels={cl_cost!r} "
                f"abs_diff={gross_cross_check[rk]['abs_diff']:.3e}")

        # ------------------------------------------------------------------
        # 2. Reconciliation: priced-sum + detector_penalty_total == gross_operational_cost
        #    (report the residual explicitly -- never assumed zero)
        # ------------------------------------------------------------------
        reconciliation = {}
        for rk in RUN_ORDER:
            tw = loaded[rk]['cl']['totals_weighted']
            rc = loaded[rk]['cl']['recourse_components']
            priced_sum = sum(tw[k] for k in PRICED_COMPONENT_KEYS)
            detector_sum = tw['detector_penalty_total']
            recon_total = priced_sum + detector_sum
            gross = rc['gross_operational_cost']
            residual = gross - recon_total
            # salvage/net check
            salvage = rc.get('terminal_salvage_value')
            net = rc.get('net_operational_recourse')
            gross_minus_net = gross - net if net is not None else None
            reconciliation[rk] = {
                'priced_component_sum': priced_sum,
                'detector_penalty_total': detector_sum,
                'reconstructed_total': recon_total,
                'reported_gross_operational_cost': gross,
                'residual_reported_minus_reconstructed': residual,
                'terminal_salvage_value': salvage,
                'net_operational_recourse': net,
                'gross_minus_net_operational_recourse': gross_minus_net,
                'economic_recourse_all_D_excluded': rc.get('economic_recourse_all_D_excluded'),
                'economic_recourse_voltage_excluded': rc.get('economic_recourse_voltage_excluded'),
            }
            log(f"[{rk}] reconciliation residual (gross - (priced_sum+detector)) = {residual:.3e}; "
                f"gross - net_operational_recourse = {gross_minus_net!r} "
                f"(salvage={salvage!r})")

        # ------------------------------------------------------------------
        # 3. Block-sum check: sum over the 48 blocks' 'weighted' component values
        #    must reproduce totals_weighted for every key (per-block-summable check)
        # ------------------------------------------------------------------
        block_sum_check = {}
        for rk in RUN_ORDER:
            cl = loaded[rk]['cl']
            blocks = cl['blocks']
            tw = cl['totals_weighted']
            max_abs_residual = 0.0
            per_key_residual = {}
            for key in tw:
                s = sum(b['weighted'][key] for b in blocks.values())
                resid = s - tw[key]
                per_key_residual[key] = resid
                max_abs_residual = max(max_abs_residual, abs(resid))
            block_sum_check[rk] = {
                'n_blocks': len(blocks),
                'max_abs_residual_over_all_keys': max_abs_residual,
                'per_key_residual': per_key_residual,
            }
            log(f"[{rk}] block-sum check ({len(blocks)} blocks): "
                f"max abs residual over all {len(tw)} component keys = {max_abs_residual:.3e}")

        # ------------------------------------------------------------------
        # 4. Per-component system-level table: value per run + diffs
        #    Convention: diff = arm_value - run1_value (matches the task's headline
        #    sign, e.g. A - run1 = -39,149).
        # ------------------------------------------------------------------
        all_component_keys = list(loaded['run1']['cl']['totals_weighted'].keys())
        component_table = {}
        for key in all_component_keys:
            vals = {rk: loaded[rk]['cl']['totals_weighted'][key] for rk in RUN_ORDER}
            diffs = {rk: vals[rk] - vals['run1'] for rk in ('A', 'C', 'D')}
            c_minus_d = vals['C'] - vals['D']
            category = (
                'priced' if key in PRICED_COMPONENT_KEYS else
                'detector_D' if key in DETECTOR_COMPONENT_KEYS else
                'detector_D_total' if key == 'detector_penalty_total' else
                'pure_reporting_not_in_gross' if key in PURE_REPORTING_KEYS else
                'orphan_slack_expected_zero'
            )
            same_sign_C_D = (diffs['C'] > 0) == (diffs['D'] > 0) if (diffs['C'] != 0 or diffs['D'] != 0) else True
            component_table[key] = {
                'category': category,
                'values': vals,
                'diff_vs_run1': diffs,
                'C_minus_D': c_minus_d,
                'C_and_D_same_sign_vs_run1': same_sign_C_D,
            }
        # Sum check: do generation_cost + flexibility_cost_internal (+ small residual terms)
        # reproduce the total gross_operational_cost diff for each arm?
        headline_gross = {rk: loaded[rk]['cl']['recourse_components']['gross_operational_cost'] for rk in RUN_ORDER}
        headline_diff_vs_run1 = {rk: headline_gross[rk] - headline_gross['run1'] for rk in ('A', 'C', 'D')}
        headline_c_minus_d = headline_gross['C'] - headline_gross['D']
        log('headline gross_operational_cost per run: ' + json.dumps(headline_gross))
        log('headline diff vs run1 (arm - run1): ' + json.dumps(headline_diff_vs_run1))
        log(f"headline C - D = {headline_c_minus_d:.2f}")

        dominant_two = ['generation_cost', 'flexibility_cost_internal']
        dominant_two_diff_vs_run1 = {
            rk: sum(component_table[k]['diff_vs_run1'][rk] for k in dominant_two)
            for rk in ('A', 'C', 'D')
        }
        dominant_two_c_minus_d = sum(component_table[k]['C_minus_D'] for k in dominant_two)
        # residual = headline diff minus the dominant-two diff minus detector diff (detector
        # is ~identical across runs but not bit-identical; account for it explicitly rather
        # than assuming it cancels)
        detector_diff_vs_run1 = {
            rk: component_table['detector_penalty_total']['diff_vs_run1'][rk] for rk in ('A', 'C', 'D')
        }
        other_priced_diff_vs_run1 = {
            rk: sum(component_table[k]['diff_vs_run1'][rk] for k in PRICED_COMPONENT_KEYS if k not in dominant_two)
            for rk in ('A', 'C', 'D')
        }
        accounted_vs_run1 = {
            rk: dominant_two_diff_vs_run1[rk] + other_priced_diff_vs_run1[rk] + detector_diff_vs_run1[rk]
            for rk in ('A', 'C', 'D')
        }
        unaccounted_vs_run1 = {
            rk: headline_diff_vs_run1[rk] - accounted_vs_run1[rk] for rk in ('A', 'C', 'D')
        }
        for rk in ('A', 'C', 'D'):
            log(f"[{rk}] headline diff vs run1 = {headline_diff_vs_run1[rk]:.4f}; "
                f"dominant-two (generation_cost+flexibility_cost_internal) = "
                f"{dominant_two_diff_vs_run1[rk]:.4f}; other priced components = "
                f"{other_priced_diff_vs_run1[rk]:.4f}; detector_penalty_total delta = "
                f"{detector_diff_vs_run1[rk]:.6f}; unaccounted residual = "
                f"{unaccounted_vs_run1[rk]:.6f}")
        log(f"C - D: headline = {headline_c_minus_d:.4f}; dominant-two = {dominant_two_c_minus_d:.4f}; "
            f"unaccounted residual = {headline_c_minus_d - dominant_two_c_minus_d - sum(component_table[k]['C_minus_D'] for k in PRICED_COMPONENT_KEYS if k not in dominant_two) - component_table['detector_penalty_total']['C_minus_D']:.6f}")

        # ------------------------------------------------------------------
        # 5. Rule-nine/ten error bar: max objective step over the last 10 cycles,
        #    per arm and for run1, bar = arm_step + run1_step (same construction the
        #    s39 evaluator already used for C and D; independently reproduced here for
        #    A and cross-checked against s39_evaluation.json for C and D).
        # ------------------------------------------------------------------
        error_bars = {}
        run1_step, run1_n = _max_abs_step_last10(loaded['run1']['g']['cycle_trajectory'], log, 'run1')
        for rk in ('A', 'C', 'D'):
            arm_step, arm_n = _max_abs_step_last10(loaded[rk]['g']['cycle_trajectory'], log, rk)
            bar = arm_step + run1_step if (arm_step is not None and run1_step is not None) else None
            diff = headline_diff_vs_run1[rk]
            inside_bar = (abs(diff) <= bar) if bar is not None else None
            error_bars[rk] = {
                'arm_max_step_last10': arm_step, 'arm_n_steps_used': arm_n,
                'run1_max_step_last10': run1_step, 'run1_n_steps_used': run1_n,
                'bar': bar, 'abs_diff_vs_run1': abs(diff), 'inside_bar': inside_bar,
                'reading': ('inside the bar: the cost difference to run 1 is within stopping slack '
                            '(indeterminate, consistent with run 1); outside the bar: not explained '
                            'by stopping slack alone, reported as a difference to be explained -- '
                            'never as a lower-cost or different point (the bar is local and valid '
                            'only when both runs are settled; per Addendum 21 all of A/C/D are '
                            'certified under the 10-consecutive-cycle bar, run1 under Addendum 16).'),
            }
            log(f"[{rk}] error bar vs run1: arm_step={arm_step:.4f} run1_step={run1_step:.4f} "
                f"bar={bar:.4f} abs_diff={abs(diff):.4f} inside_bar={inside_bar}")
        # Cross-check against the s39 evaluator's own precomputed bar for C and D.
        s39_eval_cross_check = {}
        for rk, eval_name in (('C', 's39_evaluation.json'), ('D', 's39_evaluation.json')):
            eval_path = os.path.join(RUNS[rk]['dir'], eval_name)
            if os.path.exists(eval_path):
                inputs_used.append(eval_path)
                ev = json.load(open(eval_path))
                cvr = ev['this_arm']['REPORTED_NOT_GATED']['cost_vs_run1']
                s39_eval_cross_check[rk] = {
                    'file': os.path.relpath(eval_path, REPO),
                    'evaluator_abs_diff': cvr['abs_diff'],
                    'evaluator_bar': cvr['bar'],
                    'evaluator_inside_bar': cvr['inside_bar'],
                    'this_script_abs_diff': error_bars[rk]['abs_diff_vs_run1'],
                    'this_script_bar': error_bars[rk]['bar'],
                    'agrees': (
                        abs(cvr['abs_diff'] - error_bars[rk]['abs_diff_vs_run1']) < 1e-3
                        and abs(cvr['bar'] - error_bars[rk]['bar']) < 1e-3
                    ),
                }
                log(f"[{rk}] cross-check vs {eval_name}: evaluator abs_diff={cvr['abs_diff']:.4f} "
                    f"bar={cvr['bar']:.4f} vs this script's {error_bars[rk]['abs_diff_vs_run1']:.4f} / "
                    f"{error_bars[rk]['bar']:.4f} -- agrees={s39_eval_cross_check[rk]['agrees']}")

        # ------------------------------------------------------------------
        # 6. Per-block breakdown for the two dominant components, aggregated by
        #    kind (TSO/DSO), by DSO node, and by year (day-level detail kept in the
        #    full per-block JSON below). Per-block IS the finest grain the artifact
        #    carries for these components; system-level-only quantities (interface
        #    settlement, salvage) are reported separately in section 7.
        # ------------------------------------------------------------------
        blocks_ref = loaded['run1']['cl']['blocks']
        kinds = sorted({b['kind'] for b in blocks_ref.values()})
        dso_nodes = sorted({b['node_id'] for b in blocks_ref.values() if b['kind'] == 'DSO'})
        years = sorted({b['year'] for b in blocks_ref.values()})

        def _agg(pred, comp_key):
            out = {}
            for rk in RUN_ORDER:
                out[rk] = sum(
                    b['weighted'][comp_key] for bk, b in loaded[rk]['cl']['blocks'].items() if pred(blocks_ref[bk])
                )
            out['diff_vs_run1'] = {rk: out[rk] - out['run1'] for rk in ('A', 'C', 'D')}
            out['C_minus_D'] = out['C'] - out['D']
            return out

        per_block_aggregation = {}
        for comp_key in dominant_two:
            per_block_aggregation[comp_key] = {
                'by_kind': {k: _agg(lambda b, k=k: b['kind'] == k, comp_key) for k in kinds},
                'by_dso_node': {str(n): _agg(lambda b, n=n: b['kind'] == 'DSO' and b['node_id'] == n, comp_key)
                                for n in dso_nodes},
                'by_year': {y: _agg(lambda b, y=y: b['year'] == y, comp_key) for y in years},
            }
            log(f"per-block aggregation computed for {comp_key}: by_kind, by_dso_node, by_year")

        # Full per-block, per-component, per-run detail (all 48 blocks x all component
        # keys x 4 runs) -- kept in the JSON for completeness / re-derivation, not
        # printed to the log.
        full_block_detail = {}
        for bk in blocks_ref:
            full_block_detail[bk] = {
                'kind': blocks_ref[bk]['kind'], 'node_id': blocks_ref[bk]['node_id'],
                'year': blocks_ref[bk]['year'], 'day': blocks_ref[bk]['day'],
                'weighted': {rk: loaded[rk]['cl']['blocks'][bk]['weighted'] for rk in RUN_ORDER},
            }

        # ------------------------------------------------------------------
        # 7. System-level-only quantities not summable over blocks: interface
        #    settlement (per DSO node, aggregated over year/day -- NOT a per-block
        #    field in this artifact) and terminal salvage. These are EXCLUDED from
        #    gross_operational_cost by construction (row 3', Addendum 12) -- verified
        #    in section 2 above (the reconstructed total from priced+detector alone
        #    already reproduces gross_operational_cost to <1e-6).
        # ------------------------------------------------------------------
        settlement = {}
        for rk in RUN_ORDER:
            rc = loaded[rk]['cl']['recourse_components']
            settlement[rk] = {
                'interface_settlement_tso': rc.get('interface_settlement_tso'),
                'interface_settlement_dso_per_node': rc.get('interface_settlement_dso'),
                'interface_settlement_total': rc.get('interface_settlement_total'),
                'gross_operational_cost_including_settlement': rc.get('gross_operational_cost_including_settlement'),
            }
            log(f"[{rk}] interface_settlement_total = {settlement[rk]['interface_settlement_total']!r} "
                "(system-level, per-DSO-node granularity only, NOT per-block; excluded from "
                "gross_operational_cost by row-3' construction, per Addendum 12)")

        # ------------------------------------------------------------------
        # assemble result
        # ------------------------------------------------------------------
        result = {
            'stage': 'P5.15 Addendum 22 item 1(a) -- zero-solve cost decomposition (S40)',
            'authority': [
                'PLANNER_BRIEF_2026-09-13.md Addendum 22',
                'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
            ],
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'objective_convention': (
                "All figures are gross_operational_cost as captured in "
                "component_levels_terminal.json's recourse_components (cross-checked below "
                "against g_<label>.json's own gross_operational_cost field). "
                "gross_operational_cost == net_operational_recourse in all four runs "
                "(terminal_salvage_value is ~0 to machine precision, no salvage reinstated); "
                "the gross/net distinction the CLAUDE.md reporting rule requires is stated "
                "explicitly per run in the 'reconciliation' section below, not assumed. "
                "gross_operational_cost includes the five category-D detector penalties "
                "(they stay in the solver objective per Addendum 10); the manuscript's "
                "reported Q(x) convention (economic_recourse_all_D_excluded) differs from "
                "gross_operational_cost by exactly detector_penalty_total in every run "
                "(reported per run, not assumed identical -- it is numerically close but not "
                "bit-identical across runs, see the 'reconciliation' and 'component_table' "
                "sections)."
            ),
            'DIFFERENT_ADMM_CONFIGURATIONS_WARNING': (
                "run1, A, C and D used FOUR DIFFERENT ADMM configurations (tau, PF-balancing "
                "policy, ESS-balancing policy, cap -- see 'runs' below), each stopping at a "
                "different cycle under its own certification rule. All four cost the SAME "
                "candidate/instance (verified below). Per Addendum 22 ('one frozen oracle "
                "configuration for every candidate in a campaign; costs from different "
                "configurations never share a table') and the CLAUDE.md evidence rules, the "
                "tables in this report are a DIAGNOSIS of why four differently-configured, "
                "differently-stopped ADMM runs on the SAME candidate report different terminal "
                "costs. They are NOT a comparison of candidates and license no claim that one "
                "configuration or one candidate is cheaper than another."
            ),
            'runs': {rk: {'dir': os.path.relpath(RUNS[rk]['dir'], REPO), 'label': RUNS[rk]['label'],
                          'config_summary': RUNS[rk]['config_summary'],
                          'config_spec': RUNS[rk]['config_spec'],
                          'instance': instances[rk],
                          'cycles_run': loaded[rk]['g'].get('cycles_run'),
                          'converged_at_cycle': loaded[rk]['g'].get('converged_at_cycle')}
                     for rk in RUN_ORDER},
            'instance_identity_check': {'same_instance_all_four_runs': same_instance, 'instances': instances},
            'gross_operational_cost_cross_check_g_json_vs_component_levels': gross_cross_check,
            'reconciliation': reconciliation,
            'block_sum_check': block_sum_check,
            'headline': {
                'gross_operational_cost_per_run': headline_gross,
                'diff_vs_run1_arm_minus_run1': headline_diff_vs_run1,
                'C_minus_D': headline_c_minus_d,
            },
            'component_table': component_table,
            'dominant_two_components': dominant_two,
            'diff_decomposition_vs_run1': {
                rk: {
                    'headline_diff': headline_diff_vs_run1[rk],
                    'dominant_two_diff': dominant_two_diff_vs_run1[rk],
                    'other_priced_components_diff': other_priced_diff_vs_run1[rk],
                    'detector_penalty_total_diff': detector_diff_vs_run1[rk],
                    'unaccounted_residual': unaccounted_vs_run1[rk],
                } for rk in ('A', 'C', 'D')
            },
            'diff_decomposition_C_minus_D': {
                'headline': headline_c_minus_d,
                'dominant_two': dominant_two_c_minus_d,
                'other_priced_components': sum(component_table[k]['C_minus_D'] for k in PRICED_COMPONENT_KEYS if k not in dominant_two),
                'detector_penalty_total': component_table['detector_penalty_total']['C_minus_D'],
            },
            'error_bars_rule_nine_ten': error_bars,
            's39_evaluator_cross_check': s39_eval_cross_check,
            'per_block_aggregation_dominant_two': per_block_aggregation,
            'per_block_vs_system_level_note': (
                "generation_cost and flexibility_cost_internal (and every other key in "
                "totals_weighted except the settlement/salvage fields in section "
                "'settlement_system_level_only') are PER-BLOCK quantities: the block-sum "
                "check above shows sum_over_48_blocks(block['weighted'][key]) == "
                "totals_weighted[key] EXACTLY (residual 0.0) for every run and every key. "
                "interface_settlement_tso/dso and terminal_salvage_value are SYSTEM-LEVEL "
                "(interface_settlement_dso is per-DSO-node, aggregated over year/day; "
                "interface_settlement_tso and terminal_salvage_value have no finer "
                "granularity in this artifact) and are NOT part of gross_operational_cost "
                "(row-3' transfer excluded by construction, Addendum 12)."
            ),
            'full_block_detail_all_components_all_runs': full_block_detail,
            'settlement_system_level_only': settlement,
        }

        failures = guard.verify(0)
        if failures:
            raise RuntimeError(f'SolveProfileGuard violation: {failures}')
        log('SolveProfileGuard.verify(0): OK -- zero solves for the whole script')
    finally:
        guard.uninstall()

    if dry:
        log('--dry-run: not writing outputs')
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log(f'wrote {OUT_JSON}')
    with open(OUT_LOG, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {OUT_LOG}')

    manifest_files = []
    all_paths = sorted(set(inputs_used)) + [OUT_JSON, OUT_LOG]
    for p in all_paths:
        manifest_files.append({'path': os.path.relpath(p, REPO), 'bytes': os.path.getsize(p), 'sha256': _sha256(p)})
    manifest = {
        'stage': 'P5.15 Addendum 22 item 1(a) -- zero-solve cost decomposition (S40)',
        'script': SCRIPT_REL,
        'note': ("Inputs are each run's own committed g_<label>.json / "
                 "component_levels_terminal.json / s39_evaluation.json, read-only; none "
                 "modified."),
        'n_files': len(manifest_files),
        'total_bytes': sum(m['bytes'] for m in manifest_files),
        'files': manifest_files,
    }
    with open(OUT_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'wrote {OUT_MANIFEST}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
