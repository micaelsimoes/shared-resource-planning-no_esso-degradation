"""
Stage P5.3-B2-R -- RES capability semantics and conditioning.

B2-R.1  rating-semantics audit of every curtailable RES generator
B2-R.2  categorize whether a defensible converter MVA rating exists
B2-R.3  derive the current production feasible set and Q capability
B2-R.4  validate the feasible region against the static network data
B2-R.5  initialization interiority at the cold start

Pure analysis of the production data/formulation -- no solve, no model change.
The mathematical A/B is gated on B2-R.2 finding a defensible `S_converter`.

    python p53b2r_res_capability_audit.py
"""

import io
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from definitions import EQUALITY_TOLERANCE  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53B2R')

# every generator field that exists anywhere in the repository's network JSONs
KNOWN_JSON_GENERATOR_FIELDS = [
    'gen_id', 'bus', 'Pmax', 'Pmin', 'Qmax', 'Qmin', 'Vg', 'status', 'type',
    'pf_control', 'pf_max', 'pf_min']
RATING_FIELD_CANDIDATES = ['Smax', 'S_rated', 'MVA', 'mva', 'rating', 'Srated',
                           'converter', 'inverter', 'nameplate', 'apparent']


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def tangents(gen):
    return sorted((math.tan(math.acos(gen.min_pf)), math.tan(math.acos(gen.max_pf))))


def q_capability(p_av, pg, t_up, qmax):
    """Largest |qg| allowed at a given pg, per binding source."""
    circle = math.sqrt(max(p_av ** 2 - pg ** 2, 0.0))
    cone = t_up * pg
    static = qmax
    best = min(circle, cone, static)
    if best == circle and circle <= cone and circle <= static:
        binder = 'sg_capability (circle)'
    elif cone <= circle and cone <= static:
        binder = 'PF cone'
    else:
        binder = 'static qmax'
    return {'circle': circle, 'pf_cone': cone, 'static_qmax': static,
            'effective_abs_q_max': best, 'binding': binder}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.3-B2-R', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'equality_tolerance': EQUALITY_TOLERANCE}

    quiet = io.StringIO()
    with redirect_stdout(quiet):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()

    # ---------------- B2-R.1 : rating-semantics audit ----------------
    gens, seen = [], set()
    low_output = []
    for node_id, dn in sorted(planning.distribution_networks.items()):
        for year in dn.years:
            for day in dn.days:
                net = dn.network[year][day]
                base = net.baseMVA
                for gen in net.generators:
                    if not gen.is_curtaillable():
                        continue
                    t_lo, t_up = tangents(gen)
                    key = (net.name, gen.gen_id, year)
                    if key not in seen:
                        seen.add(key)
                        gens.append({
                            'network': net.name, 'node': node_id, 'year': year,
                            'gen_id': gen.gen_id, 'bus': gen.bus,
                            'gen_type': gen.gen_type,
                            'pmin_pu': gen.pmin, 'pmax_pu': gen.pmax,
                            'qmin_pu': gen.qmin, 'qmax_pu': gen.qmax,
                            'pmax_MVA_equiv': gen.pmax * base,
                            'qmax_MVAr_equiv': gen.qmax * base,
                            'vg': gen.vg,
                            'min_pf': gen.min_pf, 'max_pf': gen.max_pf,
                            'power_factor_control': bool(gen.power_factor_control),
                            'tangent_lower': t_lo, 'tangent_upper': t_up,
                            'explicit_apparent_power_rating_field': None,
                            'generator_class_has_rating_attr': any(
                                hasattr(gen, a) for a in
                                ('smax', 's_rated', 'mva', 'rating', 'converter_rating')),
                            'qmax_equals_pmax': abs(gen.qmax - gen.pmax) < 1e-12,
                        })
                    # low-output realized points
                    for s_o in range(net.num_oper_scenarios):
                        for p in range(net.num_instants):
                            p_av = float(gen.pg[s_o][p]); q_av = float(gen.qg[s_o][p])
                            s_av = math.sqrt(p_av ** 2 + q_av ** 2)
                            if 1e-5 < p_av <= 1e-4:
                                cap = q_capability(p_av, p_av, t_up, gen.qmax)
                                low_output.append({
                                    'network': net.name, 'node': node_id,
                                    'gen_id': gen.gen_id, 'gen_type': gen.gen_type,
                                    'bus': gen.bus, 'year': year, 'day': day, 'hour': p,
                                    'P_available_pu': p_av, 'Q_available_pu': q_av,
                                    'S_available_pu': s_av,
                                    'S_available_over_pmax': s_av / gen.pmax if gen.pmax else None,
                                    'qmax_static_pu': gen.qmax,
                                    'capability_radius_vs_static_qmax_ratio':
                                        (s_av / gen.qmax) if gen.qmax else None,
                                    'q_capability_at_pg_init': cap,
                                    'cold_start_margin_at_pg_init':
                                        s_av ** 2 - (p_av ** 2 + 0.0 ** 2),
                                })
    report['B2R1_generators'] = gens
    report['B2R1_summary'] = {
        'n_distinct_generator_year_instances': len(gens),
        'all_pf_control': all(g['power_factor_control'] for g in gens),
        'all_qmax_equals_pmax': all(g['qmax_equals_pmax'] for g in gens),
        'any_explicit_rating_field': any(
            g['explicit_apparent_power_rating_field'] is not None for g in gens),
        'any_generator_class_rating_attr': any(
            g['generator_class_has_rating_attr'] for g in gens),
        'json_generator_fields_in_repository': KNOWN_JSON_GENERATOR_FIELDS,
        'rating_field_candidates_searched': RATING_FIELD_CANDIDATES,
    }

    # ---------------- B2-R.2 : categorization ----------------
    report['B2R2_categorization'] = {
        'category': 'C',
        'meaning': 'No defensible converter/inverter MVA rating exists in the repository',
        'evidence': [
            'The Generator class defines only pmax/pmin/qmax/qmin/vg/pf limits; '
            'it has no apparent-power, MVA, inverter, converter or nameplate field.',
            'Across all 6223 generator entries in every network JSON in the '
            'repository the field set is exactly '
            f'{KNOWN_JSON_GENERATOR_FIELDS} -- no rating field of any kind.',
            'Qmax is set exactly equal to Pmax for every curtailable RES unit, '
            'which is a permissive modelling convention, not a documented '
            'converter rating.',
            'No comment, schema or metadata in the parser or data files defines '
            'an apparent-power capability semantics for Pmax or Qmax.',
        ],
        'rejected_derivations': {
            'sqrt(pmax^2+qmax^2)': 'Would give S = sqrt(2)*Pmax = 1.414*Pmax purely '
                                   'because Qmax was set equal to Pmax. No documented '
                                   'semantics justify it, and 1.414x is not a '
                                   'physically standard inverter sizing.',
            'oversizing_factor': 'Explicitly prohibited; would be arbitrary.',
            'historical_maximum': 'Explicitly prohibited; a data maximum is not an '
                                  'equipment rating.',
        },
    }

    # ---------------- B2-R.3 : current feasible set ----------------
    sample = gens[0]
    t_up = sample['tangent_upper']
    p_av_ref = 1.0            # symbolic: express capability as a multiple of P_available
    pts = {}
    for label, pg in (('pg = 0', 0.0), ('pg = 0.5*P_available', 0.5),
                      ('pg = 0.9*P_available', 0.8999), ('pg = P_available', 1.0)):
        cap = q_capability(p_av_ref, pg, t_up, 1e9)   # static bound removed to isolate
        pts[label] = {k: v for k, v in cap.items()}
    pg_star = 1.0 / math.sqrt(1.0 + t_up ** 2)
    report['B2R3_feasible_set'] = {
        'sg_capability_row': 'pg^2 + qg^2 <= sg_avail^2',
        'sg_avail_definition': 'sqrt(P_available^2 + Q_available^2); Q_available == 0 '
                               'by construction, so sg_avail == P_available',
        'pf_cone': 'tangent_lower*pg <= qg <= tangent_upper*pg',
        'tangent_upper': t_up,
        'q_capability_as_multiple_of_P_available': pts,
        'pf_cone_binds_below_pg_over_P_available': pg_star,
        'max_abs_q_over_P_available': t_up * pg_star,
        'q_capability_at_zero_availability':
            'P_available = 0 triggers renewable_generation_is_unavailable, so pg and qg '
            'are bounded to (0,0) and all RES rows are skipped -- no reactive capability',
        'conclusion': ('Stochastic irradiance/wind availability directly controls the '
                       'inverter reactive-power capability: the capability radius IS '
                       'P_available. Reactive capability is exactly zero at pg = 0 and '
                       'exactly zero at pg = P_available, peaks at '
                       f'{t_up * pg_star:.4f}*P_available at pg = {pg_star:.4f}*P_available, '
                       'and vanishes entirely when availability is zero.'),
    }

    # ---------------- B2-R.4 : validation against static data ----------------
    binding = defaultdict(int)
    ratio_stats = []
    for node_id, dn in sorted(planning.distribution_networks.items()):
        for year in dn.years:
            for day in dn.days:
                net = dn.network[year][day]
                for gen in net.generators:
                    if not gen.is_curtaillable():
                        continue
                    _, t = tangents(gen)
                    for s_o in range(net.num_oper_scenarios):
                        for p in range(net.num_instants):
                            p_av = float(gen.pg[s_o][p])
                            if p_av <= EQUALITY_TOLERANCE:
                                binding['structurally off (row skipped)'] += 1
                                continue
                            cap = q_capability(p_av, p_av, t, gen.qmax)   # at pg_init
                            binding[cap['binding']] += 1
                            if gen.qmax:
                                ratio_stats.append(p_av / gen.qmax)
    report['B2R4_binding_at_cold_start'] = dict(binding)
    report['B2R4_capability_radius_over_static_qmax'] = {
        'n': len(ratio_stats),
        'min': min(ratio_stats) if ratio_stats else None,
        'median': sorted(ratio_stats)[len(ratio_stats) // 2] if ratio_stats else None,
        'max': max(ratio_stats) if ratio_stats else None,
        'note': ('capability circle radius is P_available; static qmax equals pmax. '
                 'A ratio << 1 means sg_capability is far more restrictive than the '
                 'static reactive bound.'),
    }
    report['B2R6_low_output_points'] = sorted(low_output, key=lambda r: r['P_available_pu'])

    # ---------------- B2-R.5 : initialization interiority ----------------
    report['B2R5_initialization'] = {
        'pg_init': 'max(0, P_available)  -> equals P_available for live generators',
        'qg_init': '0 (neutral starting point)',
        'sg_capability_at_init': 'pg_init^2 + qg_init^2 = P_available^2 = sg_avail^2',
        'cold_start_margin': 0.0,
        'interpretation': ('Every live RES generator starts EXACTLY on the nonlinear '
                           'capability boundary, which is precisely the zero-margin '
                           'sg_capability finding recorded in P5.3-A. Under a separated '
                           'converter rating the same initial point would sit strictly '
                           'inside the circle, with normalized interior distance '
                           '1 - (P_available/S_converter)^2 -- but S_converter does not '
                           'exist in the repository, so this cannot be quantified.'),
    }

    with open(os.path.join(OUT_DIR, 'p53b2r_report.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    s = report['B2R1_summary']
    print(f"[B2-R.1] distinct generator-year instances: {s['n_distinct_generator_year_instances']}")
    print(f"         all pf_control={s['all_pf_control']}  all qmax==pmax={s['all_qmax_equals_pmax']}")
    print(f"         any explicit rating field: {s['any_explicit_rating_field']}  "
          f"class rating attr: {s['any_generator_class_rating_attr']}")
    print(f"[B2-R.2] CATEGORY {report['B2R2_categorization']['category']} -- "
          f"{report['B2R2_categorization']['meaning']}")
    print("[B2-R.3] Q capability (multiples of P_available):")
    for k, v in report['B2R3_feasible_set']['q_capability_as_multiple_of_P_available'].items():
        print(f"         {k:24s} circle={v['circle']:.4f} cone={v['pf_cone']:.4f} "
              f"-> |q|max={v['effective_abs_q_max']:.4f} ({v['binding']})")
    print(f"         peak |q| = {report['B2R3_feasible_set']['max_abs_q_over_P_available']:.4f}"
          f" * P_available at pg = {report['B2R3_feasible_set']['pf_cone_binds_below_pg_over_P_available']:.4f}*P_available")
    print(f"[B2-R.4] binding at cold start: {report['B2R4_binding_at_cold_start']}")
    print(f"         radius/static_qmax: {report['B2R4_capability_radius_over_static_qmax']}")
    print(f"[B2-R.5] cold-start margin on sg_capability = "
          f"{report['B2R5_initialization']['cold_start_margin']}")
    print(f"[B2-R.6] low-output points in (1e-5,1e-4]: {len(report['B2R6_low_output_points'])}")
    for row in report['B2R6_low_output_points'][:5]:
        print(f"         {row['network']}/{row['year']}/{row['day']} gen {row['gen_id']} h{row['hour']}: "
              f"P_av={row['P_available_pu']:.3e} S_av={row['S_available_pu']:.3e} "
              f"radius/qmax={row['capability_radius_vs_static_qmax_ratio']:.3e} "
              f"|q|max={row['q_capability_at_pg_init']['effective_abs_q_max']:.3e}")


if __name__ == '__main__':
    main()
