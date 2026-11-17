"""
Stage P5.9-A, refinement -- locating the polish boundary in rho_pf.

INFORMED BY THE SWEEP, DECLARED AS SUCH.  The A sweep used the specified
absolute grid {10, 100, 1000}.  It found the exact-consensus polish failing on
3 blocks at rho_pf = 100 and cleanly succeeding at rho_pf = 1000, and it found
that reaching 1000 gives back 2 031 318 of the 2 157 016 the rescaling had
bought.  The question this refinement answers is therefore narrow and worth
asking: is there an intermediate rho_pf that polishes cleanly AND keeps more of
the gain, or is the whole gain the price of a working polish?

THE SELECTION RULE IS UNCHANGED -- the same one declared in `p59_a_sweep.py`
before any result existed -- and it is re-applied over the UNION of the sweep
and this refinement.  No rule is re-fitted to the data.

A4 (rho_ess) is NOT run.  It was conditional on ESS consensus binding, and it
does not: `primal_ess_ratio` sits between 0.0012 and 0.0057 across all eleven
sweep cases, i.e. 175x to 833x inside tolerance, under every rho tested.  That
independently reconfirms P5.8-A0 under conditions P5.8 never varied.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_a_refine.py
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
from p59_a_sweep import INHERITED_RHO, select_best  # noqa: E402

A_PATH = os.path.join(EV.OUT_DIR, 'p59_a_sweep.json')
OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_a_refine.json')

CASES = [
    ('A3b_rho_pf_300',  {'v': 1.5, 'pf': 300.0}),
    ('A3b_rho_pf_500',  {'v': 1.5, 'pf': 500.0}),
]


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.9-A refinement', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    with open(A_PATH) as handle:
        a_report = json.load(handle)

    x0 = dict(BC.population(planning_gate))['base']
    report = {'stage': 'P5.9-A refinement', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'rationale': ('the polish boundary lies between rho_pf=100 (3 '
                            'failed blocks) and rho_pf=1000 (0 failed blocks); '
                            'this locates it'),
              'selection_rule': a_report['selection_rule'],
              'a4_rho_ess': ('not run; ESS consensus does not bind '
                             '(primal_ess_ratio 0.0012..0.0057 across all '
                             'eleven sweep cases)'),
              'cases': []}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for case_id, rho in CASES:
        print(f'[P5.9-A+] {case_id}  rho_v={rho["v"]} rho_pf={rho["pf"]} ...',
              flush=True)
        state, info = EV.load_template(rescale=True, rho=rho)
        record, _ = EV.evaluate(x0, template_state=state, case_id=case_id,
                                rho=rho, archive_label=case_id)
        row = EV.row(record)
        row['template_info'] = info
        row['cycle_detail'] = record.get('cycle_detail')
        row['rho_ratio_to_inherited'] = {
            g: round(rho[g] / INHERITED_RHO[g], 4) for g in rho}
        report['cases'].append(row)
        persist()
        print(f"           status={row['status']}  cycles={row['admm_cycles']}  "
              f"pre-polish={row['admm_net_recourse_before_polish']}  "
              f"polished={row['total_objective']}  "
              f"failed_blocks={row['n_failed_polish_blocks']}  "
              f"rho_final={row['rho_observed_final']}", flush=True)

    # ---- re-apply the ORIGINAL rule over the union -------------------------
    union = a_report['cases'] + report['cases']
    best, reason = select_best(union)
    report['best_over_union'] = best['case_id'] if best else None
    report['best_over_union_reason'] = reason
    persist()

    a_report['refinement_cases'] = report['cases']
    a_report['best_case'] = best['case_id'] if best else None
    a_report['best_case_reason'] = (
        f'{reason} (re-applied over sweep + refinement)')
    a_report['best_case_row'] = best
    a_report['a4_rho_ess'] = report['a4_rho_ess']
    with open(A_PATH, 'w') as handle:
        json.dump(a_report, handle, indent=1, default=str)

    print(f"\n[P5.9-A+] best over sweep + refinement: {report['best_over_union']}")
    print(f"          ({reason})")
    print(f'[P5.9-A+] report -> {OUT_PATH}; selection updated in {A_PATH}')


if __name__ == '__main__':
    main()
