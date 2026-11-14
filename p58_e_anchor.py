"""
Stage P5.8-E -- interface anchor audit.  Run only after B/C.

`_prepare_transmission_objectives_for_admm` fixes the TSO's ADN load `pc`/`qc` at
the DSO's then-current consensus interface power
(`shared_resources_planning.py:2905-2919`, via `fix_or_set`), and the
transmission system then pays `flexibility_cost` for downward deviation from that
anchor.  On a warm start `_run_operational_planning` clones
`initial_state['models']` (`2173-2179`) and never re-runs that preparation.

P5.7 measured the consequence for one chain: `max |pc(T0) - pc(step 12)| = 0.0`.
P5.8-E closes the two questions P5.7 left explicitly unmeasured:

  E1  anchor evolution      -- across generations, CURRENT and RESCALED chains;
  E2  candidate dependence  -- a DIFFERENT candidate, warm-started from T0;
  E3  cold versus warm      -- the same different candidate, started COLD.

No anchor policy is changed.  Nothing is written to production data.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p58_e_anchor.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p56b_policy as P  # noqa: E402
import p57_eval as E7  # noqa: E402
import p58_eval as E  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p58_e_anchor.json')
OTHER_CANDIDATE = 'se|node5|2025|-10%'    # VALID under direct T0 (P5.6-B)


def anchor(tso_models):
    """The fixed TSO ADN interface anchor, per block, in the model's own order."""
    out = {}
    for year, days in tso_models.items():
        for day, model in days.items():
            row = {}
            for name in ('pc', 'qc'):
                component = getattr(model, name, None)
                if component is None:
                    continue
                values, fixed = [], []
                for idx in component:
                    data = component[idx]
                    values.append(np.nan if data.value is None
                                  else float(data.value))
                    fixed.append(bool(data.fixed))
                row[name] = np.asarray(values)
                row[f'{name}_n_fixed'] = int(sum(fixed))
            out[f'{year}|{day}'] = row
    return out


def compare(a, b):
    """Max |a - b| per family over every shared block."""
    out = {}
    for key in sorted(set(a) & set(b)):
        for name in ('pc', 'qc'):
            if name not in a[key] or name not in b[key]:
                continue
            va, vb = a[key][name], b[key][name]
            if va.shape != vb.shape:
                out.setdefault(name, {})[key] = 'shape mismatch'
                continue
            out.setdefault(name, {})[key] = float(np.nanmax(np.abs(va - vb)))
    return {name: {'max_over_blocks': max(v for v in blocks.values()
                                          if isinstance(v, float)),
                   'worst_block': max((kv for kv in blocks.items()
                                       if isinstance(kv[1], float)),
                                      key=lambda kv: kv[1])[0],
                   'per_block': blocks}
            for name, blocks in out.items()}


def main():
    os.makedirs(E.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.8-E anchor audit', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.8] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    x0 = population['base']
    if OTHER_CANDIDATE not in population:
        print(f'[P5.8] ABORTED: population has no {OTHER_CANDIDATE}')
        sys.exit(1)
    x_other = population[OTHER_CANDIDATE]

    report = {'stage': 'P5.8-E', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'other_candidate': OTHER_CANDIDATE,
              'anchor_policy_changed': False, 'comparisons': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.8] obtaining T0 ...', flush=True)
    t0 = E7.t0_template()
    t0_anchor = anchor(t0['models']['tso'])
    report['t0_fixed_counts'] = {
        key: {k: v for k, v in row.items() if k.endswith('n_fixed')}
        for key, row in list(t0_anchor.items())[:2]}
    persist()

    # ------------------------------------------------------------------ E1 --
    print('\n[P5.8] E1: anchor evolution along the CURRENT and RESCALED chains',
          flush=True)
    e1 = {}
    for label, pattern in (('CURRENT (P5.7 chain)', 'chain_j{}__polished.npz'),
                           ('RESCALED (P5.8-C chain)', 'c_rescaled_j{}__polished.npz')):
        directory = (E7.ARCHIVE_DIR if pattern.startswith('chain')
                     else E.ARCHIVE_DIR)
        rows = {}
        reference = None
        for j in (1, 2, 4, 8, 12):
            path = os.path.join(directory, pattern.format(j))
            if not os.path.exists(path):
                continue
            data = np.load(path)
            keys = [k for k in data.files
                    if k.startswith('TSO|') and k.endswith('|var|pc')]
            values = np.concatenate([data[k] for k in sorted(keys)])
            if reference is None:
                reference = values
                rows[j] = 0.0
            else:
                rows[j] = float(np.nanmax(np.abs(values - reference)))
        e1[label] = {'max_abs_pc_change_vs_generation_1': rows}
        print(f'      {label}: {rows}', flush=True)
    report['comparisons']['E1_anchor_evolution'] = e1
    persist()

    # ------------------------------------------------------------------ E2 --
    print(f'\n[P5.8] E2: candidate dependence -- {OTHER_CANDIDATE} warm from T0',
          flush=True)
    planning = O.fresh_planning('p58_e_warm_other')
    candidate = O.vector_to_candidate(planning, x_other)
    started = time.time()
    models, state, admm = P.run_operational(planning, candidate, t0)
    report['E2_admm'] = admm
    if admm.get('status'):
        report['comparisons']['E2_candidate_dependence'] = {
            'error': admm['status']}
    else:
        report['comparisons']['E2_candidate_dependence'] = compare(
            t0_anchor, anchor(models['tso']))
        print(f"      max |pc(T0) - pc(other, warm)| = "
              f"{report['comparisons']['E2_candidate_dependence']['pc']['max_over_blocks']:.6e}",
              flush=True)
    report['E2_runtime_s'] = time.time() - started
    persist()
    del models, state, planning

    # ------------------------------------------------------------------ E3 --
    print(f'\n[P5.8] E3: cold versus warm -- {OTHER_CANDIDATE} started COLD '
          f'(this rebuilds every model, ~500 s)', flush=True)
    planning = O.fresh_planning('p58_e_cold_other')
    candidate = O.vector_to_candidate(planning, x_other)
    started = time.time()
    models, state, admm = P.run_operational(planning, candidate, None)
    report['E3_admm'] = admm
    if admm.get('status'):
        report['comparisons']['E3_cold_vs_warm'] = {'error': admm['status']}
    else:
        cold_anchor = anchor(models['tso'])
        report['comparisons']['E3_cold_vs_warm'] = {
            'cold_vs_T0': compare(t0_anchor, cold_anchor)}
        pc = report['comparisons']['E3_cold_vs_warm']['cold_vs_T0']['pc']
        print(f"      max |pc(T0) - pc(other, cold)| = "
              f"{pc['max_over_blocks']:.6e}  worst {pc['worst_block']}",
              flush=True)
    report['E3_runtime_s'] = time.time() - started
    persist()

    print(f'\n[P5.8] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
