"""P5.15 Diag-A -- ZERO-SOLVE integrity check + leak measurement on the OLD
(pre-reformulation) ESSO control-arm pickle.

Does NOT solve anything. Loads
data/SRP1/Results/P514N/esso_models_control.pkl (read-only) and reads Var
values already baked into the pickled Pyomo models. SolveProfileGuard is armed
in BLOCKING form (permitted=()) for the whole run so any solver entry raises.

Writes ONLY new files under data/SRP1/Results/P5151/:
  - p5151_diagA_old_control_leak.json
  - p5151_diagA_console.log (via redirected print, saved by caller)
"""
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PKL = os.path.join(REPO, 'data/SRP1/Results/P514N/esso_models_control.pkl')
OUT_DIR = os.path.join(REPO, 'data/SRP1/Results/P5151')
OUT_JSON = os.path.join(OUT_DIR, 'p5151_diagA_old_control_leak.json')

REPORT = {
    'stage': 'P5.15 Diag-A',
    'objective': 'zero-solve integrity check + leak measurement on the OLD control pickle',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'pickle_path': os.path.relpath(PKL, REPO),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    # ---- provenance, before anything else ----
    st = os.stat(PKL)
    REPORT['provenance'] = {
        'sha256': sha256(PKL),
        'size_bytes': st.st_size,
        'mtime_utc': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
    sibling_files = ['control_console.log', 'n1_control.json',
                      'frozen_n1_control_arm_v1_f1bbddf4.json']
    REPORT['sibling_artifacts'] = {}
    for name in sibling_files:
        p = os.path.join(os.path.dirname(PKL), name)
        if os.path.exists(p):
            s = os.stat(p)
            REPORT['sibling_artifacts'][name] = {
                'sha256': sha256(p), 'size_bytes': s.st_size,
                'mtime_utc': datetime.fromtimestamp(s.st_mtime, tz=timezone.utc).isoformat(),
            }
        else:
            REPORT['sibling_artifacts'][name] = None

    commit_check = os.popen(
        f'cd "{REPO}" && git log -1 --format=%H,%ci b03c9b14 2>&1').read().strip()
    REPORT['reformulation_commit_b03c9b14'] = commit_check

    # ---- arm the blocking guard for the ENTIRE run, permitted count 0 ----
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.15 Diag-A (zero-solve)').install()

    try:
        import pyomo.environ as pe  # noqa: F401
        # Import production modules so any custom classes referenced by the
        # pickle (if any) are resolvable at unpickle time.
        import shared_resources_planning as srp  # noqa: F401
        import shared_energy_storage_data as sed_mod  # noqa: F401
        import model_construction_helpers as mch  # noqa: F401

        with open(PKL, 'rb') as f:
            try:
                models = pickle.load(f)
            except Exception as error:
                REPORT['integrity_check'] = {
                    'verdict': 'STOP -- UNPICKLE FAILED',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'traceback': traceback.format_exc(),
                }
                write_report()
                print('UNPICKLE FAILED -- see report. STOPPING.')
                return

        REPORT['models_container_type'] = type(models).__name__
        REPORT['node_ids'] = list(models.keys())

        # ---------------- integrity check ----------------
        integrity = {}
        any_pre = False
        any_post = False
        for node_id, model in models.items():
            attrs_present = [a for a in (
                'es_degradation_per_unit', 'es_degradation_per_unit_cumul',
                'es_soh_per_unit', 'es_soh_per_unit_cumul',
                'energy_storage_complementarity', 'energy_storage_normalization',
                'es_pch_hat_per_unit', 'es_pdch_hat_per_unit',
                'es_pch_hat_agg', 'es_pdch_hat_agg',
                'slack_es_ch_comp_per_unit',
                'es_D_per_unit',  # POST-reformulation marker; absence expected if PRE
            ) if hasattr(model, a)]
            integrity[str(node_id)] = attrs_present
            if 'es_degradation_per_unit' in attrs_present or 'es_soh_per_unit' in attrs_present:
                any_pre = True
            if 'es_D_per_unit' in attrs_present and not (
                    'energy_storage_complementarity' in attrs_present):
                any_post = True

        pre_markers = any('es_degradation_per_unit' in v or 'es_soh_per_unit' in v
                           for v in integrity.values())
        comp_row_present = any('energy_storage_complementarity' in v
                                or 'energy_storage_normalization' in v
                                for v in integrity.values())
        d_present = any('es_D_per_unit' in v for v in integrity.values())

        if pre_markers and comp_row_present and not d_present:
            verdict = 'PRE-REFORMULATION -- diagnostic proceeds'
        elif d_present and not comp_row_present:
            verdict = 'POST-REFORMULATION -- diagnostic VOID, STOP'
        else:
            verdict = ('AMBIGUOUS -- attributes found do not cleanly match either '
                       'profile; reported as-is, proceed with caution')

        REPORT['integrity_check'] = {
            'attrs_present_per_node': integrity,
            'pre_markers_present': pre_markers,
            'complementarity_row_present': comp_row_present,
            'es_D_per_unit_present': d_present,
            'verdict': verdict,
        }

        print('INTEGRITY VERDICT:', verdict)
        if verdict.startswith('POST'):
            write_report()
            print('POST-REFORMULATION DETECTED -- STOPPING per instructions.')
            return

        # ---------------- measurements ----------------
        measurements = {}
        for node_id, model in models.items():
            node_key = str(node_id)
            node_out = {'cohort_year_pairs': {}}

            # discover which (y_inv, y) pairs are "active" (s_max > 0)
            active_pairs = []
            for y_inv in model.years:
                for y in model.years:
                    try:
                        s_max = pe.value(model.es_s_rated_per_unit[y_inv, y], exception=False)
                    except Exception:
                        s_max = None
                    if s_max is not None and s_max > 0:
                        active_pairs.append((y_inv, y))

            node_out['active_cohort_year_pairs'] = [list(p) for p in active_pairs]

            max_min_phys = 0.0
            max_min_phys_ratio = 0.0
            max_min_hat = 0.0
            sum_pch_phys = 0.0
            sum_pdch_phys = 0.0
            sum_min_phys = 0.0
            n_periods_total = 0

            has_hat = hasattr(model, 'es_pch_hat_per_unit') and hasattr(model, 'es_pdch_hat_per_unit')

            for (y_inv, y) in active_pairs:
                s_max = pe.value(model.es_s_rated_per_unit[y_inv, y], exception=False)
                cohort_key = f'({y_inv}, {y})'
                cy_min_phys = 0.0
                cy_min_hat = 0.0
                cy_sum_pch = 0.0
                cy_sum_pdch = 0.0
                cy_n = 0
                for d in model.days:
                    for p in model.periods:
                        pch = pe.value(model.es_pch_per_unit[y_inv, y, d, p], exception=False)
                        pdch = pe.value(model.es_pdch_per_unit[y_inv, y, d, p], exception=False)
                        if pch is None or pdch is None:
                            continue
                        m = min(pch, pdch)
                        cy_min_phys = max(cy_min_phys, m)
                        sum_pch_phys += pch
                        sum_pdch_phys += pdch
                        cy_sum_pch += pch
                        cy_sum_pdch += pdch
                        sum_min_phys += m
                        cy_n += 1
                        n_periods_total += 1
                        if has_hat:
                            pch_hat = pe.value(model.es_pch_hat_per_unit[y_inv, y, d, p], exception=False)
                            pdch_hat = pe.value(model.es_pdch_hat_per_unit[y_inv, y, d, p], exception=False)
                            if pch_hat is not None and pdch_hat is not None:
                                cy_min_hat = max(cy_min_hat, min(pch_hat, pdch_hat))

                max_min_phys = max(max_min_phys, cy_min_phys)
                max_min_hat = max(max_min_hat, cy_min_hat)
                if s_max and s_max > 0:
                    max_min_phys_ratio = max(max_min_phys_ratio, cy_min_phys / s_max)

                node_out['cohort_year_pairs'][cohort_key] = {
                    's_max_p_u': s_max,
                    'max_min_pch_pdch_physical': cy_min_phys,
                    'max_min_pch_pdch_physical_over_s_max': (cy_min_phys / s_max) if s_max else None,
                    'max_min_pch_pdch_hat': cy_min_hat if has_hat else None,
                    'sum_pch_physical_unweighted': cy_sum_pch,
                    'sum_pdch_physical_unweighted': cy_sum_pdch,
                    'n_periods': cy_n,
                }

            # aggregate hat pair, if present
            agg_out = None
            if hasattr(model, 'es_pch_hat_agg') and hasattr(model, 'es_pdch_hat_agg'):
                max_min_agg_hat = 0.0
                for y in model.years:
                    for d in model.days:
                        for p in model.periods:
                            a = pe.value(model.es_pch_hat_agg[y, d, p], exception=False)
                            b = pe.value(model.es_pdch_hat_agg[y, d, p], exception=False)
                            if a is not None and b is not None:
                                max_min_agg_hat = max(max_min_agg_hat, min(a, b))
                agg_out = max_min_agg_hat

            total_throughput_unweighted = sum_pch_phys + sum_pdch_phys
            spurious_component_unweighted = 2.0 * sum_min_phys
            spurious_fraction_pct = (
                100.0 * spurious_component_unweighted / total_throughput_unweighted
                if total_throughput_unweighted > 0 else None)

            node_out['summary'] = {
                'max_min_pch_pdch_physical_p_u': max_min_phys,
                'max_min_pch_pdch_physical_over_s_max': max_min_phys_ratio,
                'max_min_pch_pdch_hat': max_min_hat if has_hat else None,
                'max_min_pch_pdch_hat_agg': agg_out,
                'note_hat_vs_physical': (
                    'hat pair is pch/s_max, pdch/s_max (normalized); physical pair feeds '
                    'avg_ch_dch_per_unit directly (energy_storage_charging_discharging row). '
                    'Degradation accounting is fed by the PHYSICAL pair (es_pch_per_unit / '
                    'es_pdch_per_unit), not the hat pair -- the hat pair exists ONLY to '
                    'feed the complementarity row.'),
                'sum_pch_physical_unweighted_total': sum_pch_phys,
                'sum_pdch_physical_unweighted_total': sum_pdch_phys,
                'total_throughput_unweighted': total_throughput_unweighted,
                'spurious_component_2x_sum_min_unweighted': spurious_component_unweighted,
                'spurious_fraction_pct_unweighted': spurious_fraction_pct,
                'n_periods_total': n_periods_total,
                'convention_note': (
                    'UNWEIGHTED convention: plain sum of pch, pdch, min(pch,pdch) over all '
                    '(active cohort-year, day, period) samples actually stored in the '
                    'pickle, with NO num_days/365 day-count weighting and no efficiency '
                    'weighting (unlike es_avg_ch_dch_per_unit, which uses '
                    '(num_days/365)*(eff_ch*pch*dt + pdch*dt/eff_dch)). num_days is a '
                    'property of SharedEnergyStorageData.days, not stored on the pickled '
                    'model, so a day-weighted throughput figure could not be recomputed '
                    'without external data not authorized for use here (measurement '
                    'reported unweighted; flagged as a limitation).'),
            }

            # ---- es_avg_ch_dch_per_unit: stored vs leak-removed ----
            avg_ch_dch_out = {}
            if hasattr(model, 'es_avg_ch_dch_per_unit'):
                for y_inv in model.years:
                    for y in model.years:
                        stored = pe.value(model.es_avg_ch_dch_per_unit[y_inv, y], exception=False)
                        if stored is None:
                            continue
                        avg_ch_dch_out[f'({y_inv}, {y})'] = {'stored': stored}

            # recompute leak-free: substitute min(pch,pdch)->0 keeping net identical.
            # Need eff_ch, eff_dch, dt, cl_eff etc, which are NOT stored on the model
            # (they are Python floats baked into the constraint coefficients at
            # construction time, not retrievable as model attributes in general).
            # Attempt to recover them if the model exposes Params for them.
            eff_recoverable = {}
            for cand in ('eff_ch', 'eff_dch', 'dt', 'cl_eff'):
                eff_recoverable[cand] = hasattr(model, cand)
            node_out['avg_ch_dch_per_unit'] = {
                'stored': avg_ch_dch_out,
                'leak_free_recomputation': (
                    'NOT PERFORMED: eff_ch/eff_dch/dt/cl_eff are not stored as model '
                    'attributes (they are Python floats baked into constraint '
                    'coefficients at construction time inside SharedEnergyStorageData, '
                    'not on the pickled Pyomo model). Recovering them would require '
                    'importing current-code constants, which the task explicitly '
                    'forbids ("the current values may differ"). Reported as NOT '
                    'RECOVERABLE from the pickle alone.'),
                'params_found_on_model': eff_recoverable,
            }

            # ---- SoH trajectory ----
            soh_out = {}
            if hasattr(model, 'es_soh_per_unit_cumul'):
                for y_inv in model.years:
                    for y in model.years:
                        v = pe.value(model.es_soh_per_unit_cumul[y_inv, y], exception=False)
                        if v is not None:
                            soh_out[f'({y_inv}, {y})'] = v
            deg_out = {}
            if hasattr(model, 'es_degradation_per_unit'):
                for y_inv in model.years:
                    for y in model.years:
                        v = pe.value(model.es_degradation_per_unit[y_inv, y], exception=False)
                        if v is not None:
                            deg_out[f'({y_inv}, {y})'] = v
            node_out['soh_chain'] = {
                'es_soh_per_unit_cumul': soh_out,
                'es_degradation_per_unit': deg_out,
                'leak_free_soh_trajectory': (
                    'NOT COMPUTED: the degradation constants (cl_eff, e_rated used in '
                    'the daily-degradation row) are recoverable in principle '
                    '(es_e_rated_per_unit IS a stored Var), but cl_eff (the '
                    'calendar/cycling efficiency constant multiplying e_rated in '
                    '`es_degradation_per_unit * (2*cl_eff*e_rated) == avg_ch_dch`) is a '
                    'Python float baked into the constraint coefficient at build time '
                    'and is NOT stored as a retrievable model attribute or Param. '
                    'Without it we cannot solve back for a leak-free D from a leak-free '
                    'avg_ch_dch. Not imported from current code per task instructions.'),
            }

            # ---- complementarity slack ----
            slack_out = None
            if hasattr(model, 'slack_es_ch_comp_per_unit'):
                max_slack = 0.0
                any_active = False
                n_slack = 0
                for (y_inv, y) in active_pairs:
                    for d in model.days:
                        for p in model.periods:
                            v = pe.value(model.slack_es_ch_comp_per_unit[y_inv, y, d, p], exception=False)
                            if v is None:
                                continue
                            n_slack += 1
                            if v > max_slack:
                                max_slack = v
                            if v > 1e-9:
                                any_active = True
                slack_out = {'max_slack': max_slack, 'any_active_above_1e-9': any_active,
                             'n_evaluated': n_slack}
            node_out['complementarity_slack'] = slack_out

            measurements[node_key] = node_out

        REPORT['measurements'] = measurements

        # cross-node comparison table vs the stated reformulated fixture values
        table = []
        for node_key, node_out in measurements.items():
            s = node_out['summary']
            table.append({
                'node': node_key,
                'max_min_pch_pdch_over_s_max_OLD': s['max_min_pch_pdch_physical_over_s_max'],
                'spurious_fraction_pct_OLD_unweighted': s['spurious_fraction_pct_unweighted'],
            })
        REPORT['comparison_table_vs_reformulated_fixture'] = {
            'NEW_reformulated_max_min_over_s_max': 4.6319e-4,
            'NEW_reformulated_spurious_fraction_pct': 0.9175,
            'OLD_control_per_node': table,
        }

    finally:
        failures = guard.verify(expected_solves=0)
        REPORT['guard'] = {
            'permitted': [],
            'counts': dict(guard.counts),
            'verify_failures': failures,
        }
        guard.uninstall()

    write_report()
    print('DONE. Report written to', OUT_JSON)
    print('guard counts:', guard.counts)


def write_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_JSON):
        raise RuntimeError(f'refusing to overwrite existing artifact: {OUT_JSON}')
    with open(OUT_JSON, 'w') as f:
        json.dump(REPORT, f, indent=1, default=str)


if __name__ == '__main__':
    main()
