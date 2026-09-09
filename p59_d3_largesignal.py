"""
Stage P5.9-D3 -- does the rescaled oracle respond to a LARGE capacity change?

D2 produced the stage's most consequential measurement and it was not the one
expected.  At generation 1 the RESCALED oracle separates the +-10% candidates
from the base by 316..631 currency units AT EVERY PENALTY TESTED, including the
inherited rho_pf = 2.25 where nothing was raised.  The CURRENT formulation
separates the same candidates by 162 172 and 293 999 at the same generation.
The collapse is therefore a property of the RESCALING, not of rho.

Two readings survive that measurement and they lead to opposite conclusions:

  (a) the small separation is PHYSICAL.  P5.7 §2 measured the shared ESS as
      essentially inert at the base bootstrap capacity (schedules moved 1.7e-05
      p.u. across twelve refinements), so a +-10% capacity perturbation would
      genuinely be worth only a few hundred currency units, and the ~3e5
      separation the CURRENT formulation reports is an artefact of subproblems
      that stop 8.9e6 short of base-optimality.  P5.6-C's Spearman of -0.033
      and 6-of-9 sign reversals would then be exactly what ranking on numerical
      residue looks like.

  (b) the small separation is SUPPRESSED by the rescaled coordination, in which
      case the oracle is blind and unusable regardless of how stable it is.

A +-10% perturbation cannot separate these.  A LARGE one can: if the oracle
responds proportionally to a 19x capacity change, reading (a) holds and the
small signal is real physics; if it still returns a few hundred, the oracle is
not resolving capacity at all.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_d3_largesignal.py <candidate>
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

RHO = {'v': 1.5, 'pf': 1000.0}
CANDIDATES = ['base', 'se|ALL|-10%', 'se|ALL|x19 (budget boundary)',
              'e|node9|2030|+50%']


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tag = (only or 'all').replace('|', '_').replace('%', 'pct').replace(' ', '_')
    out_path = os.path.join(EV.OUT_DIR, f'p59_d3_largesignal_{tag}.json')
    try:
        provenance, planning_gate = gate('P5.9-D3 large signal', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    report = {'stage': 'P5.9-D3', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'rho': RHO, 'generations': 1, 'rows': []}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for name in CANDIDATES:
        if only is not None and name != only:
            continue
        candidate = O.vector_to_candidate(planning_gate, population[name])
        investment = O.investment_cost(planning_gate, candidate)
        for arm, rescale, rho in (('CURRENT', False, None),
                                  ('RESCALED', True, RHO)):
            safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')
            case_id = f'd3_{safe}_{arm}'
            print(f'[P5.9-D3] {name}  {arm} ...', flush=True)
            state, _ = EV.load_template(rescale=rescale, rho=rho)
            record, _ = EV.evaluate(x=population[name], template_state=state,
                                    case_id=case_id, rho=rho,
                                    archive_label=case_id)
            row = EV.row(record)
            row.update({'candidate': name, 'arm': arm,
                        'investment_cost': investment})
            report['rows'].append(row)
            persist()
            print(f"          pre-polish={row['admm_net_recourse_before_polish']} "
                  f"polished={row['total_objective']} "
                  f"fail={row['n_failed_polish_blocks']} "
                  f"status={row['status']}", flush=True)

    print(f'\n[P5.9-D3] report -> {out_path}')


if __name__ == '__main__':
    main()
