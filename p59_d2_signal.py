"""
Stage P5.9-D2 -- the investment signal as a function of the interface penalty.

D found that RESCALED at rho_pf = 1000 stabilizes the refinement chain and
simultaneously destroys what the chain is for: at generation 8 the three
candidates separate by 626 and 645 against a planning signal of 33 031, where
the CURRENT formulation separates them by 317 675 and 296 558.  A stable oracle
that cannot tell two investments apart is not a usable oracle.

This isolates the trade directly.  For each candidate, ONE generation from
frozen T0 at several rho_pf values, comparing the PRE-POLISH objective -- which
is defined whether or not the polish succeeds, so the low-penalty points that
fail the polish still contribute a measurement.

The question is narrow: is there any rho_pf that both admits a successful
exact-consensus polish and preserves an investment signal larger than
tau_planning?

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_d2_signal.py <candidate>
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

RHO_PF = [2.25, 100.0, 300.0]     # 1000.0 already measured for all three in D
CANDIDATES = ['base', 'se|node5|2025|-10%', 'se|node9|2025|-10%']


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tag = (only or 'all').replace('|', '_').replace('%', 'pct').replace(' ', '_')
    out_path = os.path.join(EV.OUT_DIR, f'p59_d2_signal_{tag}.json')

    try:
        provenance, planning_gate = gate('P5.9-D2 investment signal', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    report = {'stage': 'P5.9-D2', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'design': ('one generation from frozen T0 per (candidate, rho_pf); '
                         'pre-polish objective is the comparison quantity because '
                         'it is defined even where the polish fails'),
              'rho_pf_values': RHO_PF, 'rows': []}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for name in CANDIDATES:
        if only is not None and name != only:
            continue
        for rho_pf in RHO_PF:
            rho = {'v': 1.5, 'pf': rho_pf}
            safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')
            case_id = f'd2_{safe}_pf{rho_pf:g}'
            print(f'[P5.9-D2] {name}  rho_pf={rho_pf} ...', flush=True)
            state, _ = EV.load_template(rescale=True, rho=rho)
            record, _ = EV.evaluate(x=population[name], template_state=state,
                                    case_id=case_id, rho=rho,
                                    archive_label=case_id)
            row = EV.row(record)
            row['candidate'] = name
            row['rho_pf'] = rho_pf
            report['rows'].append(row)
            persist()
            print(f"          pre-polish={row['admm_net_recourse_before_polish']} "
                  f"polished={row['total_objective']} "
                  f"fail={row['n_failed_polish_blocks']}", flush=True)

    print(f'\n[P5.9-D2] report -> {out_path}')


if __name__ == '__main__':
    main()
