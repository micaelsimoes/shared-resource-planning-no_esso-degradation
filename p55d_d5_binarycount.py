"""
Stage P5.5-D5 -- exact binary-population derivation.

D5's SOLVING is conditional on D4 being promising, and D4's gate fired against
it.  D5's ANALYSIS is not conditional: the stage explicitly requires the exact
binary count implied by the current model structure, its factorisation, and a
determination of whether one common mode binary per physical ESS / year / day /
period is mathematically sufficient given the retained separate-copy
architecture.  That question has an answer independent of whether the MISOCP is
ever solved, and it is answered here.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55d_d5_binarycount.py
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from definitions import ESS_COMPLEMENTARITY_TOLERANCE  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55D')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-D5 binary count', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D5] ABORTED\n{error}')
        sys.exit(1)

    delta = math.sqrt(float(ESS_COMPLEMENTARITY_TOLERANCE))
    esso = planning.shared_ess_data
    nodes = list(esso.active_distribution_network_nodes)
    years = list(esso.years)
    days = list(planning.days)
    tso = planning.transmission_network
    network = tso.network[years[0]][days[0]]
    periods = network.num_instants
    scen_m = len(network.prob_market_scenarios)
    scen_o = len(network.prob_operation_scenarios)

    # how many model copies does each physical ESS have?
    tso_copies = {node: sum(1 for e in tso.network[years[0]][days[0]]
                            .shared_energy_storages if e.bus == node)
                  for node in nodes}
    dso_copies = {}
    for node, dso in sorted(planning.distribution_networks.items()):
        d_net = dso.network[years[0]][days[0]]
        ref = d_net.get_reference_node_id()
        dso_copies[node] = sum(1 for e in d_net.shared_energy_storages
                               if e.bus == ref)

    n_ess = len(nodes)
    n_years, n_days = len(years), len(days)
    shared = n_ess * n_years * n_days * periods * scen_m * scen_o
    per_copy = shared * 2          # one TSO copy + one DSO copy per physical ESS

    factorisation = {
        'physical_shared_ESS_count': n_ess,
        'physical_shared_ESS_nodes': nodes,
        'network_copies_per_physical_ESS': 2,
        'network_copies_detail': {'transmission': tso_copies,
                                  'distribution': dso_copies},
        'year_count': n_years, 'years': years,
        'representative_day_count': n_days, 'days': days,
        'period_count': periods,
        'market_scenarios': scen_m, 'operation_scenarios': scen_o,
        'binaries_one_per_physical_ESS': shared,
        'binaries_one_per_network_copy': per_copy,
        'arithmetic_shared': f'{n_ess} x {n_years} x {n_days} x {periods} '
                             f'x {scen_m} x {scen_o} = {shared}',
        'arithmetic_per_copy': f'{shared} x 2 = {per_copy}',
    }

    # ------------------------------------------------------------------------
    # Is ONE common binary per physical ESS sufficient?
    # ------------------------------------------------------------------------
    sufficiency = {
        'question': ('can one mode binary be shared between the TSO and DSO '
                     'copies of the same physical ESS?'),
        'what_the_couplings_force': (
            'the interface_ess_p / interface_ess_q rows force '
            'shared_es_pnet_TSO == shared_es_pnet_DSO (both networks are on a '
            '100 MVA base, so the conversion is the identity).  They force the '
            'NET power to agree; they do not force the (pch, pdch) split to '
            'agree.'),
        'physical_answer': (
            'physically the two copies are one device, so a common mode is the '
            'correct physics and one binary per physical ESS / year / day / '
            'period is what the device actually has.'),
        'mathematical_answer': (
            'NOT unconditionally valid as an OUTER approximation of the '
            'two-copy H1 set.  Write pi for the common net power.  A '
            'charge-dominant copy has pdch <= delta*S, hence pch = pi + pdch in '
            '[pi, pi + delta*S].  A discharge-dominant copy has pch <= delta*S, '
            'hence pdch = pch - pi >= 0 requires pi <= delta*S.  The two copies '
            'can therefore legitimately sit in OPPOSITE modes precisely when '
            '|pi| <= delta*S.  Outside that band the modes must agree and one '
            'shared binary loses nothing; inside it, a shared binary would '
            'exclude combinations that are H1-feasible for each copy '
            'separately, which breaks the lower-bound guarantee in that band.'),
        'opposite_mode_band': f'|pnet| <= delta*S = {delta:g} * S',
        'band_width_over_S': 2 * delta,
        'rigorous_options': [
            f'{per_copy} binaries (one per network copy) -- unconditionally a '
            f'valid outer approximation;',
            f'{shared} binaries (one per physical ESS) -- valid only with an '
            f'explicit argument that the |pnet| <= delta*S band contributes '
            f'nothing, which has NOT been proved here;',
        ],
        'recommendation': (
            f'if the MISOCP were ever built, start from {shared} shared '
            f'binaries for tractability and treat the |pnet| <= {delta:g}*S band '
            f'as a documented approximation, or use {per_copy} for a formally '
            f'clean bound.  Do not assume the shared form is free.'),
    }

    status = {
        'D5_solving_performed': False,
        'why': ('D4 is the gate for D5, and D4 fired against it: fixing every '
                'ESS mode moved the full-model objective by 0.049 % and closed '
                '0.42 % of the gap to the polished nonlinear feasible point.  '
                'Because Q_MISOCP <= Q_fixed_mode, the unrestricted MISOCP '
                'optimum cannot close more.  Running MI-1..MI-4 would only '
                'measure the cost of confirming a conclusion already reached.'),
        'stages_not_run': ['MI-1', 'MI-2', 'MI-3', 'MI-4'],
    }

    report = {'stage': 'P5.5-D5', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'delta': delta, 'binary_factorisation': factorisation,
              'shared_binary_sufficiency': sufficiency,
              'execution_status': status}

    out = os.path.join(OUT_DIR, 'p55d_d5_binarycount.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('[D5] exact binary population implied by the current model structure\n')
    print(f"    physical shared ESS          : {n_ess}  (nodes {nodes})")
    print(f"    network copies per ESS       : 2  (one TSO block, one DSO block)")
    print(f"    years                        : {n_years}  {years}")
    print(f"    representative days          : {n_days}  {days}")
    print(f"    periods per day              : {periods}")
    print(f"    scenarios (market x operation): {scen_m} x {scen_o}")
    print(f"\n    one binary per PHYSICAL ESS  : {factorisation['arithmetic_shared']}")
    print(f"    one binary per NETWORK COPY  : {factorisation['arithmetic_per_copy']}")
    print('\n[D5] is one shared binary per physical ESS sufficient?')
    print(f"    couplings force  : {sufficiency['what_the_couplings_force']}")
    print(f"    physically       : {sufficiency['physical_answer']}")
    print(f"    mathematically   : {sufficiency['mathematical_answer']}")
    print(f"    opposite-mode band: {sufficiency['opposite_mode_band']} "
          f"(width {2 * delta:g} S)")
    print('\n[D5] MI stages NOT run')
    print(f"    {status['why']}")
    print(f'\n[D5] report -> {out}')


if __name__ == '__main__':
    main()
