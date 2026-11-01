"""
P5.7 -- solution fingerprints for nonlinear operational branch comparison.

P5.6-D established that repeated deterministic refinement of the same fixed
investment keeps finding lower local branches without converging.  P5.7 has to
say WHICH variables move between those branches, so this module records enough
of a solved state to answer that, and nothing that would perturb it.

Nothing here solves, modifies or re-parameterizes a production model.  Every
function reads an already-solved Pyomo model.

Two artefacts per solution:

  * a JSON-safe SUMMARY   -- per block, per variable family: n, sum, sum|.|,
    max|.|, and per constraint family: n, n_active, max violation; plus the
    weighted base objective and its cost families;
  * a numpy ARCHIVE       -- the full flattened vector of every variable family
    in a fixed index order, and the packed boolean active-set masks, so two
    solutions can be differenced exactly.

The index order is the deterministic iteration order of the Pyomo component,
which is identical across models built from the same case, so vectors from two
different solutions of the same block are aligned element by element.
"""

import os
import sys

import numpy as np
import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

# a variable index counts as sitting ON a bound within this window
BOUND_ACTIVE_TOL = 1e-7
# an inequality body counts as ACTIVE within this window of its bound
CONSTRAINT_ACTIVE_TOL = 1e-7

# Every network variable family P5.7 needs.  Families absent from a given case
# (ordinary ESS in SRP1, TSO-only pc/qc) are skipped silently.
VAR_FAMILIES = (
    # dispatch
    'pg', 'qg',
    # load / interface
    'pc', 'qc', 'pc_curt_down', 'pc_curt_up', 'qc_curt_down', 'qc_curt_up',
    # flexibility
    'flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down',
    'slack_flex_p_balance_up', 'slack_flex_p_balance_down',
    'slack_flex_q_balance_up', 'slack_flex_q_balance_down',
    # voltage / state
    'e', 'f', 'vmag', 'vmag_sqr',
    'voltage_product_real', 'voltage_product_imag',
    # voltage and balance slacks
    'slack_v_sqr_up', 'slack_v_sqr_down',
    'slack_node_balance_p_up', 'slack_node_balance_p_down',
    'slack_node_balance_q_up', 'slack_node_balance_q_down',
    'slack_flow_ij_sqr', 'slack_flow_ji_sqr',
    # branch flows and the continuous OLTC
    'pij', 'qij', 'pji', 'qji', 'r', 'r_sqr',
    # shared ESS
    'shared_es_pch', 'shared_es_pdch', 'shared_es_pch_hat', 'shared_es_pdch_hat',
    'shared_es_pnet', 'shared_es_qnet', 'shared_es_soc',
    'shared_es_s_rated', 'shared_es_e_rated',
    'slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down',
    # ordinary ESS (absent in SRP1, retained so the fingerprint is case-general)
    'es_pch', 'es_pdch', 'es_pnet', 'es_qnet', 'es_soc',
    'slack_es_soc_final_up', 'slack_es_soc_final_down',
)

COST_FAMILIES = ('generation_cost', 'flexibility_cost', 'load_curtailment_cost',
                 'gen_curtailment_penalty', 'ess_utilization_cost_penalty',
                 'slack_penalties', 'ess_complementarity_penalties')


# ---------------------------------------------------------------------------
#  one variable family
# ---------------------------------------------------------------------------
def _var_vector(model, name):
    """Flattened values and at-bound mask, in the component's own index order."""
    component = getattr(model, name, None)
    if component is None or component.ctype is not pe.Var:
        # several families are Params rather than Vars on some networks -- most
        # importantly pc/qc, which are decision variables on the TRANSMISSION
        # model (the TSO/DSO interface) but fixed load parameters on a DSO
        return None, None
    values, at_bound = [], []
    for idx in component:
        data = component[idx]
        raw = data.value          # None when uninitialized; no logger noise
        if raw is None:
            values.append(np.nan)
            at_bound.append(0)
            continue
        value = float(raw)
        values.append(value)
        flag = 0
        lb, ub = data.lb, data.ub
        if lb is not None and abs(value - float(lb)) <= BOUND_ACTIVE_TOL:
            flag |= 1
        if ub is not None and abs(value - float(ub)) <= BOUND_ACTIVE_TOL:
            flag |= 2
        if data.fixed:
            flag |= 4
        at_bound.append(flag)
    return np.asarray(values, dtype=np.float64), np.asarray(at_bound, dtype=np.uint8)


# ---------------------------------------------------------------------------
#  constraint activity
# ---------------------------------------------------------------------------
def _constraint_activity(model):
    """Per active constraint family: n, n_active, worst violation, active mask.

    An equality is never reported as 'active' -- it always is.  Only genuine
    one- or two-sided inequalities contribute to the active set, which is what
    P5.7-C compares between branches.
    """
    families, masks = {}, {}
    for con in model.component_objects(pe.Constraint, active=True,
                                       descend_into=False):
        flags, n_active, worst, n_ineq = [], 0, 0.0, 0
        for idx in con:
            data = con[idx]
            equality = data.equality
            try:
                body = float(pe.value(data.body))
            except Exception:
                flags.append(0)
                continue
            flag = 0
            if data.has_lb():
                lower = float(pe.value(data.lower))
                worst = max(worst, lower - body)
                if not equality and abs(body - lower) <= CONSTRAINT_ACTIVE_TOL:
                    flag |= 1
            if data.has_ub():
                upper = float(pe.value(data.upper))
                worst = max(worst, body - upper)
                if not equality and abs(body - upper) <= CONSTRAINT_ACTIVE_TOL:
                    flag |= 2
            if not equality:
                n_ineq += 1
                if flag:
                    n_active += 1
            else:
                flag |= 4
            flags.append(flag)
        name = con.local_name
        families[name] = {'n': len(flags), 'n_inequality': n_ineq,
                          'n_active': n_active, 'max_violation': worst}
        masks[name] = np.asarray(flags, dtype=np.uint8)
    return families, masks


# ---------------------------------------------------------------------------
#  one block
# ---------------------------------------------------------------------------
def _block_fingerprint(model, network, params, weight, archive, prefix):
    summary = {'weight': weight, 'variables': {}, 'constraints': {}}
    for name in VAR_FAMILIES:
        values, at_bound = _var_vector(model, name)
        if values is None or values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        summary['variables'][name] = {
            'n': int(values.size),
            'sum': float(finite.sum()),
            'sum_abs': float(np.abs(finite).sum()),
            'max_abs': float(np.abs(finite).max()) if finite.size else 0.0,
            'n_at_lb': int((at_bound & 1 > 0).sum()),
            'n_at_ub': int((at_bound & 2 > 0).sum()),
            'n_fixed': int((at_bound & 4 > 0).sum()),
        }
        archive[f'{prefix}|var|{name}'] = values
        archive[f'{prefix}|bnd|{name}'] = at_bound

    families, masks = _constraint_activity(model)
    summary['constraints'] = families
    for name, mask in masks.items():
        archive[f'{prefix}|con|{name}'] = mask

    summary['weighted_base_objective'] = weight * float(
        pe.value(mch.objective_function_rule(model, params)))
    costs = {}
    for name in COST_FAMILIES:
        try:
            if name == 'ess_complementarity_penalties':
                value = float(pe.value(mch.ess_complementarity_penalties_rule(
                    model, 0, 0, network=network, params=params)))
            else:
                value = float(pe.value(getattr(mch, name)(
                    model, network, 0, 0, params)))
        except Exception:
            value = None
        costs[name] = None if value is None else weight * value
    summary['cost_families'] = costs
    return summary


# ---------------------------------------------------------------------------
#  one full solution
# ---------------------------------------------------------------------------
def fingerprint(planning, models, label, archive_dir, esso_models=None):
    """Summarize and archive a complete 48-block operational solution."""
    os.makedirs(archive_dir, exist_ok=True)
    # the label becomes a filename, so keep it path-safe
    label = ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_'
                    for c in label)
    archive, blocks = {}, {}
    for tag, holder in O._tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                key = f'{tag}|{year}|{day}'
                blocks[key] = _block_fingerprint(
                    model, network, holder.params,
                    srp._get_admm_block_weight(holder, year, day),
                    archive, key)

    summary = {'label': label, 'blocks': blocks,
               'total_weighted_base_objective':
                   sum(b['weighted_base_objective'] for b in blocks.values())}

    if esso_models is not None:
        esso = {}
        for node, model in esso_models.items():
            entry = {'variables': {}}
            for name in ('es_pch', 'es_pdch', 'es_pnet', 'es_qnet', 'es_soc',
                         'es_s_rated', 'es_e_rated', 'es_e_available',
                         'es_soh', 'es_s_investment', 'es_e_investment'):
                values, at_bound = _var_vector(model, name)
                if values is None or values.size == 0:
                    continue
                entry['variables'][name] = {
                    'n': int(values.size), 'sum': float(values.sum()),
                    'sum_abs': float(np.abs(values).sum()),
                    'max_abs': float(np.abs(values).max())}
                archive[f'ESSO{node}|var|{name}'] = values
                archive[f'ESSO{node}|bnd|{name}'] = at_bound
            esso[str(node)] = entry
        summary['esso'] = esso

    path = os.path.join(archive_dir, f'{label}.npz')
    np.savez_compressed(path, **archive)
    summary['archive'] = os.path.relpath(path, REPO_ROOT)
    return summary


# ---------------------------------------------------------------------------
#  differencing two archives
# ---------------------------------------------------------------------------
def compare(path_a, path_b):
    """Per (block, family) movement between two archived solutions."""
    a, b = np.load(path_a), np.load(path_b)
    variables, bounds, constraints = {}, {}, {}
    for key in a.files:
        if key not in b.files:
            continue
        # block keys themselves contain '|' (e.g. TSO|2025|Spring), so the
        # split has to come from the RIGHT: <block>|<kind>|<family>
        block, kind, name = key.rsplit('|', 2)
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            continue
        if kind == 'var':
            delta = vb.astype(np.float64) - va.astype(np.float64)
            finite = delta[np.isfinite(delta)]
            if finite.size == 0:
                continue
            scale = max(float(np.abs(va[np.isfinite(va)]).max(initial=0.0)), 1e-12)
            variables.setdefault(name, {})[block] = {
                'max_abs_delta': float(np.abs(finite).max()),
                'l1_delta': float(np.abs(finite).sum()),
                'l2_delta': float(np.sqrt((finite ** 2).sum())),
                'reference_max_abs': scale,
                'relative_max_delta': float(np.abs(finite).max()) / scale,
                'n_moved_1e-6': int((np.abs(finite) > 1e-6).sum()),
                'n': int(finite.size),
            }
        elif kind == 'bnd':
            changed = int((va != vb).sum())
            bounds.setdefault(name, {})[block] = {
                'n': int(va.size), 'n_changed': changed,
                'n_at_bound_a': int((va & 3 > 0).sum()),
                'n_at_bound_b': int((vb & 3 > 0).sum())}
        elif kind == 'con':
            act_a, act_b = (va & 3) > 0, (vb & 3) > 0
            constraints.setdefault(name, {})[block] = {
                'n': int(va.size),
                'n_active_a': int(act_a.sum()), 'n_active_b': int(act_b.sum()),
                'n_entered': int((~act_a & act_b).sum()),
                'n_left': int((act_a & ~act_b).sum()),
                'symmetric_difference': int((act_a ^ act_b).sum())}
    return {'variables': variables, 'bound_activity': bounds,
            'constraint_activity': constraints}


def rollup(comparison):
    """Collapse a per-block comparison to one row per family."""
    out = {'variables': {}, 'bound_activity': {}, 'constraint_activity': {}}
    for name, blocks in comparison['variables'].items():
        out['variables'][name] = {
            'max_abs_delta': max(b['max_abs_delta'] for b in blocks.values()),
            'max_relative_delta': max(b['relative_max_delta']
                                      for b in blocks.values()),
            'total_l1_delta': sum(b['l1_delta'] for b in blocks.values()),
            'total_moved_1e-6': sum(b['n_moved_1e-6'] for b in blocks.values()),
            'total_n': sum(b['n'] for b in blocks.values()),
            'worst_block': max(blocks.items(),
                               key=lambda kv: kv[1]['max_abs_delta'])[0]}
    for name, blocks in comparison['bound_activity'].items():
        out['bound_activity'][name] = {
            'total_changed': sum(b['n_changed'] for b in blocks.values()),
            'total_n': sum(b['n'] for b in blocks.values())}
    for name, blocks in comparison['constraint_activity'].items():
        out['constraint_activity'][name] = {
            'total_entered': sum(b['n_entered'] for b in blocks.values()),
            'total_left': sum(b['n_left'] for b in blocks.values()),
            'total_symmetric_difference': sum(b['symmetric_difference']
                                              for b in blocks.values()),
            'total_active_a': sum(b['n_active_a'] for b in blocks.values()),
            'total_active_b': sum(b['n_active_b'] for b in blocks.values()),
            'total_n': sum(b['n'] for b in blocks.values())}
    return out


# ---------------------------------------------------------------------------
#  moving an initial point between solutions  (hypothesis B)
# ---------------------------------------------------------------------------
def capture_values(planning, models):
    """{block: {family: ndarray}} in the same index order the archives use."""
    out = {}
    for tag, holder in O._tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                block = {}
                for name in VAR_FAMILIES:
                    values, _ = _var_vector(model, name)
                    if values is not None and values.size:
                        block[name] = values
                out[f'{tag}|{year}|{day}'] = block
    return out


def load_archive_values(path):
    """The variable half of an archive, keyed the same way as `capture_values`."""
    data = np.load(path)
    out = {}
    for key in data.files:
        block, kind, name = key.rsplit('|', 2)
        if kind != 'var':
            continue
        out.setdefault(block, {})[name] = data[key]
    return out


def cold_values(planning):
    """Default initial values, from freshly built copies of the same networks.

    This is a pure VALUE source: the models built here are discarded, and only
    their declared initializations are transferred.  No structure, bound, fixing
    or capacity of the model being initialized is touched.
    """
    out = {}
    for tag, holder in O._tagged_holders(planning):
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                fresh = network.build_model(holder.params)
                block = {}
                for name in VAR_FAMILIES:
                    values, _ = _var_vector(fresh, name)
                    if values is not None and values.size:
                        block[name] = values
                out[f'{tag}|{year}|{day}'] = block
                del fresh
    return out


def apply_values(planning, models, captured, skip_fixed=True):
    """Set the initial point of every non-fixed variable from `captured`.

    Fixed variables are skipped by default: at the polish stage the coordinated
    interface variables are FIXED at the exact consensus, and overwriting them
    would change the problem rather than its starting point.
    """
    n_set, n_skipped_fixed, n_missing = 0, 0, 0
    for tag, holder in O._tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                key = f'{tag}|{year}|{day}'
                block = captured.get(key)
                if block is None:
                    n_missing += 1
                    continue
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                for name, values in block.items():
                    component = getattr(model, name, None)
                    if component is None or component.ctype is not pe.Var:
                        continue
                    indices = list(component)
                    if len(indices) != len(values):
                        n_missing += 1
                        continue
                    for idx, value in zip(indices, values):
                        data = component[idx]
                        if data.fixed:
                            if skip_fixed:
                                n_skipped_fixed += 1
                                continue
                        if not np.isfinite(value):
                            continue
                        data.set_value(float(value), skip_validation=True)
                        n_set += 1
    return {'n_set': n_set, 'n_skipped_fixed': n_skipped_fixed,
            'n_missing_or_misaligned': n_missing}
