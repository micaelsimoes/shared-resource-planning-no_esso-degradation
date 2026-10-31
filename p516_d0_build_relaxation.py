"""P5.16 D0 / T0.1a -- W-space relaxation builder on live production block objects (ZERO solves).

Governing spec: `P5_16_D0_SPEC_v2.md` section 3 (construction), section 4 gates G-D0-0 / G-D0-2 / G-D0-3,
section 7 (captures). Design reference only (NOT imported, does not build): `convex_oracle.py` module docstring
(the B2 (U, C, D) uniform branch treatment).

This module NEVER solves and never builds a production model from scratch. It works on production block objects
that already exist (preserved fixtures, or the live certified blocks at T0.1 route A) by

    clone  ->  deactivate the definitional rank-1 families  ->  add the W-space rows  ->  (no solve)

so every retained production row is the production row object itself (cloned), not a re-typed copy.

Definitional families (the ONLY rows through which rank-1 enters a network block; spec section 3):

    component (network.py)        rule (model_construction_helpers.py)          relaxed by
    ---------------------------   -------------------------------------------   ----------------------------------
    voltage_mag_sqr_def  l.451    vmag_sqr_def               l.424-425  Wii == e^2 + f^2      rotated cone (below)
    voltage_mag_def      l.450    vmag_def                   l.428-429  Wii == vmag^2         decoupled (coupling is on Wii)
    voltage_product_real_def l.455 voltage_product_real_rule l.456-460  WijR == e_i e_j + f_i f_j   rotated cone
    voltage_product_imag_def l.456 voltage_product_imag_rule l.463-467  WijI == f_i e_j - e_i f_j   rotated cone
    r_sqr_def            l.462    r_sqr_rule                 l.1549-1552 r_sqr == r^2 (transformers) B2 envelope

Rotated cone, per active branch and period (written as the bilinear inequality; Wii >= v_min^2 > 0):

    WijR^2 + WijI^2 <= Wii * Wjj

Transformer branches (`branch.is_transformer and branch.status`): production's node balance
(`node_balance_p_rule` / `_q_rule`, MCH l.1456-1546) and terminal-power rows (`pij_rule` ... `qji_rule`,
MCH l.1582-1599, via `_branch_terminal_power_expressions` l.1555-1579) carry the bilinears r * WijR, r * WijI and
r_sqr * Wii(from). Every active row that references the transformer's `r` / `r_sqr` is REPLACED by its exact
B2 rewrite: the row's own standard representation (degree <= 2, generated from the production row object -- not
re-typed) with the substitutions

    r * WijR -> C,   r * WijI -> D,   r_sqr * Wii(from) -> U,

plus the envelope  r_lb^2 * Wii(from) <= U <= r_ub^2 * Wii(from)  (r_lb / r_ub read from the production Var's own
bounds, i.e. `transformer_ratio_bounds`, MCH l.367-375) and the transformer cone  C^2 + D^2 <= U * Wjj.
Any other quadratic term in such a row is refused (raises): the rewrite is exact or it does not happen.

ESSO (spec section 3 item 4): SoH == 1 -- the exp SoH chain rows, the soh_min floor rows and the bilinear
`available_e_capacity_unit` rows are deactivated; `es_e_available_per_unit == es_e_rated_per_unit` is added for
every (y_inv, y) whose available-e row was active; H3 pro-rata rows deactivated; `slack_es_pnet_up/down` FIXED at
0; every ESSO objective deactivated (gross excludes the ESSO; the relaxed ESSO carries a zero objective only so that
the LP writer accepts it).

Objective (spec section 3 item 5): Sigma_b w_b * [objective_function_rule(b) - contracted settlement_b
- voltage pin_b], w_b = `shared_resources_planning._get_admm_block_weight`, built from `objective_function_rule`
(MCH l.1811) and the model's own settlement / pin components, selected exactly as
`shared_resources_planning._get_local_interface_settlement(part='contracted')` and `_get_local_voltage_pin` select
them (the `_get_operational_recourse_components` lineage). Pre-Step-3.1-C fixtures carry no settlement term at all;
for them -- and only for them -- `objective_function_rule` is evaluated through `_PreSettlementFixtureView`, which
supplies 0 for the two absent settlement attributes and delegates everything else (recorded per block).

T0.1b extension (spec v2 section 10 ruling 3): `inventory_rows(..., drop_sess_comp=True)` and
`relax_network_block(..., drop_sess_comp=True)` drop the nonconvex `sess_comp` rows (form verified first, see
DROPPED_NETWORK_FAMILIES); `nonconvex_active_rows` lists any active row still outside CONVEX_FORMS after
relaxation (a stop for the caller, never a drop). The defaults keep the T0.1a dry-run behaviour unchanged.

Nothing here calls a solver. Callers arm `p513_solve_profile_guard.SolveProfileGuard` (blocking form).
"""

import hashlib

import numpy as np
import pyomo.environ as pe
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.core.expr.visitor import identify_mutable_parameters, identify_variables
from pyomo.repn import generate_standard_repn

import model_construction_helpers as mch
import shared_resources_planning as srp

# ----------------------------------------------------------------------------------------------------------------------
#  Classification tables
# ----------------------------------------------------------------------------------------------------------------------
# component name -> (expected production rule function name, relaxed_by, one-line note)
DEFINITIONAL_FAMILIES = {
    'voltage_mag_sqr_def': ('vmag_sqr_def', 'cone',
                            'Wii == e^2 + f^2 (MCH l.424); rank-1 magnitude; carried by the rotated cone'),
    'voltage_mag_def': ('vmag_def', 'decoupled',
                        'Wii == vmag^2 (MCH l.428), interface nodes; vmag is decoupled -- W-space coupling is on '
                        'Wii (spec section 3 item 6); vmag-only rows stay but no longer bind W'),
    'voltage_product_real_def': ('voltage_product_real_rule', 'cone',
                                 'WijR == e_i e_j + f_i f_j (MCH l.456); carried by the rotated cone'),
    'voltage_product_imag_def': ('voltage_product_imag_rule', 'cone',
                                 'WijI == f_i e_j - e_i f_j (MCH l.463); carried by the rotated cone'),
    'r_sqr_def': ('r_sqr_rule', 'envelope',
                  'r_sqr == r^2 (MCH l.1549), transformers only; replaced by the B2 U-envelope'),
}

ESSO_DROP_ARGUMENTS = {
    'soh_exp_chain': ('energy_storage_capacity_degradation (exp row): SoH_y == SoH_{y-1} exp(-D_y) phi^n. '
                      'LB-safe: removing an equality only enlarges the feasible set; SoH then enters no retained row.'),
    'soh_floor': ('energy_storage_capacity_degradation (floor row): SoH_y >= soh_min. '
                  'LB-safe: removing an inequality only enlarges the feasible set (SoH == 1 satisfies it anyway).'),
    'available_e_bilinear': ('available_e_capacity_unit: E_avail == E_rated * SoH (bilinear). LB-safe: production '
                             'SoH <= 1 (D >= 0, phi <= 1) so E_avail <= E_rated; replaced by E_avail == E_rated '
                             '(SoH == 1 upper envelope). E_avail enters no gross-objective term (salvage is excluded '
                             'from gross) and reaches the network copies only through shared_es_e_rated_fixed, which '
                             'variant R sets to rated -- the network feasible set is enlarged, never shrunk.'),
    'h3_pro_rata': ('energy_storage_cohort_pnet_share_h3: per-cohort pro-rata equalities. LB-safe: removing '
                    'equalities only enlarges the feasible set; inert at <= 1 active cohort per year '
                    '(0 active rows).'),
}

ESSO_SLACK_VARS = ('slack_es_pnet_up', 'slack_es_pnet_down')

# T0.1b (spec v2 section 10 ruling 3): the ONE network family dropped in the relaxation, opt-in
# (`drop_sess_comp=True`); the T0.1a dry run keeps the default False and is unchanged.
# component name -> (expected production rule function name, sub-label, LB-safety argument)
DROPPED_NETWORK_FAMILIES = {
    'sess_comp': ('sess_comp_rule', 'sess_comp_nonconvex',
                  'sess_comp (network.py l.512, MCH sess_comp_rule l.933, BILINEAR_RELAXATION branch): '
                  'shared_es_pch_hat * shared_es_pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE (1e-4). Nonconvex '
                  'bilinear inequality; removing it only enlarges the feasible set (LB-safe). P5.5-D2: the convex '
                  'hull of the complementarity set inside the box is the triangle production already imposes '
                  '(sess_active_sum_limit). Spec v2 section 10 ruling 3.'),
}

# Row forms a convex conic solver accepts (labels of `_quadratic_form_label`).
CONVEX_FORMS = ('linear', 'convex_quadratic_psd', 'rotated_cone', 'lorentz_cone')

_D0_PREFIX = 'd0_'


# ----------------------------------------------------------------------------------------------------------------------
#  Small helpers
# ----------------------------------------------------------------------------------------------------------------------
def rule_function_name(component):
    """Name of the production rule function a Constraint component was built from (None for ConstraintLists)."""
    rule = getattr(component, '_rule', None)
    if rule is None:
        return None
    fcn = getattr(rule, '_fcn', rule)
    fcn = getattr(fcn, 'func', fcn)          # functools.partial
    return getattr(fcn, '__name__', None)


def network_and_params(block):
    """The `network` / `params` objects the production rules of `block` carry (functools.partial keywords).

    Read off `node_balance_p` (built with partial(node_balance_p_rule, network=..., params=...), network.py l.521),
    so they are the exact objects the block was built with -- not a re-read of the case files."""
    keywords = block.node_balance_p._rule._fcn.keywords
    return keywords['network'], keywords['params']


def active_transformer_branches(network):
    return [b for b, branch in enumerate(network.branches) if branch.is_transformer and branch.status]


def _row_bound_values(cd):
    lower = pe.value(cd.lower) if cd.lower is not None else None
    upper = pe.value(cd.upper) if cd.upper is not None else None
    return lower, upper


def row_residual(cd):
    """Signed residuals of one row at the model's current point: body - lower and body - upper (None if absent)."""
    body = pe.value(cd.body)
    lower, upper = _row_bound_values(cd)
    return {
        'body': float(body),
        'r_lower': None if lower is None else float(body - lower),
        'r_upper': None if upper is None else float(body - upper),
    }


def residual_delta(a, b):
    """max |a - b| over the residual components both rows carry; raises if the bound structure differs."""
    delta = 0.0
    for key in ('r_lower', 'r_upper'):
        if (a[key] is None) != (b[key] is None):
            raise ValueError(f'bound structure differs ({key}): {a} vs {b}')
        if a[key] is not None:
            delta = max(delta, abs(a[key] - b[key]))
    return delta


def _quadratic_form_label(cd, repn):
    """Convexity label of one degree-<=2 row at the current Param values (an audit; it gates nothing here)."""
    if repn.nonlinear_expr is not None:
        return 'nonlinear'
    if not repn.quadratic_vars:
        return 'linear'
    has_lower = cd.lower is not None
    has_upper = cd.upper is not None
    if has_lower and has_upper:
        return 'quadratic_equality_or_range_nonconvex'
    sign = 1.0 if has_upper else -1.0          # body <= u  (Q psd)   |   body >= l  (-Q psd)
    variables = []
    position = {}
    for v1, v2 in repn.quadratic_vars:
        for v in (v1, v2):
            if id(v) not in position:
                position[id(v)] = len(variables)
                variables.append(v)
    q = np.zeros((len(variables), len(variables)))
    for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        c = sign * float(pe.value(coef))
        i, j = position[id(v1)], position[id(v2)]
        if i == j:
            q[i, i] += c
        else:
            q[i, j] += 0.5 * c
            q[j, i] += 0.5 * c
    eig = np.linalg.eigvalsh(q)
    if eig.min() >= -1e-12 * max(1.0, abs(eig).max()):
        return 'convex_quadratic_psd'
    # rotated cone  x^2 + y^2 - u v <= 0  with u, v >= 0 (and no square on u, v)
    off = [(i, j) for i in range(len(variables)) for j in range(i + 1, len(variables)) if q[i, j] != 0.0]
    if len(off) == 1:
        i, j = off[0]
        u, v = variables[i], variables[j]
        diag_ok = q[i, i] == 0.0 and q[j, j] == 0.0 and q[i, j] < 0.0
        rest = [k for k in range(len(variables)) if k not in (i, j)]
        rest_ok = all(q[k, k] > 0.0 for k in rest) and all(q[k, l] == 0.0 for k in rest for l in rest if k != l)
        nonneg = all(x.lb is not None and pe.value(x.lb) >= 0.0 for x in (u, v))
        if diag_ok and rest_ok and nonneg and sign > 0:
            return 'rotated_cone'
    # Lorentz cone  x^2 + y^2 - z^2 <= 0 with z >= 0
    negative = [k for k in range(len(variables)) if q[k, k] < 0.0]
    if (sign > 0 and len(negative) == 1 and np.count_nonzero(q - np.diag(np.diag(q))) == 0
            and variables[negative[0]].lb is not None and pe.value(variables[negative[0]].lb) >= 0.0):
        return 'lorentz_cone'
    return 'nonconvex_quadratic'


def model_fingerprint(model):
    """sha256 over every Var (value, fixed, bounds), Param value, Constraint/Objective active flag of `model`.

    Used to prove the builder never mutates its input."""
    h = hashlib.sha256()
    for v in model.component_data_objects(pe.Var, descend_into=True, sort=True):
        h.update(repr((v.name, v.value, v.fixed, v.lb, v.ub)).encode())
    for param in model.component_objects(pe.Param, descend_into=True, sort=True):
        for index in sorted(param.keys(), key=repr):
            h.update(repr((param.name, index, pe.value(param[index], exception=False))).encode())
    for c in model.component_data_objects(pe.Constraint, active=None, descend_into=True, sort=True):
        h.update(repr((c.name, c.active)).encode())
    for o in model.component_data_objects(pe.Objective, active=None, descend_into=True, sort=True):
        h.update(repr((o.name, o.active)).encode())
    return h.hexdigest()


def _assert_no_d0_components(model):
    clash = [c.name for c in model.component_objects(descend_into=False) if c.local_name.startswith(_D0_PREFIX)]
    if clash:
        raise RuntimeError(f'input block already carries {_D0_PREFIX}* components: {clash}')


# ----------------------------------------------------------------------------------------------------------------------
#  Inventory
# ----------------------------------------------------------------------------------------------------------------------
def _transformer_rows(block, network):
    """{(component name, index)} of every active row that references r / r_sqr of an active transformer branch."""
    transformers = set(active_transformer_branches(network))
    rows = set()
    if not transformers:
        return rows
    r_comp, r_sqr_comp = block.r, block.r_sqr
    for component in block.component_objects(pe.Constraint, active=True, descend_into=True):
        for index, cd in component.items():
            if not cd.active:
                continue
            for v in identify_variables(cd.body, include_fixed=True):
                parent = v.parent_component()
                if (parent is r_comp or parent is r_sqr_comp) and v.index()[0] in transformers:
                    rows.add((component.name, index))
                    break
    return rows


def _check_definitional_rules(block):
    found = {}
    for name, (expected_rule, _by, _note) in DEFINITIONAL_FAMILIES.items():
        component = getattr(block, name, None)
        if component is None:
            raise RuntimeError(f'definitional family {name} absent from the block')
        actual = rule_function_name(component)
        if actual != expected_rule:
            raise RuntimeError(f'{name}: built from rule {actual!r}, expected {expected_rule!r}')
        found[name] = actual
    return found


def _check_sess_comp_rows(block):
    """Verify the `sess_comp` family is exactly ruling 3's row before it is dropped: built from `sess_comp_rule`,
    every active row of the BILINEAR_RELAXATION form  1.0 * shared_es_pch_hat[k] * shared_es_pdch_hat[k] <= tol
    (same index k, no linear or constant part, no lower bound, tol == ESS_COMPLEMENTARITY_TOLERANCE).
    Raises on any other form (a different form needs a new ruling, not a silent drop). Returns the active count."""
    component = getattr(block, 'sess_comp', None)
    if component is None:
        return 0
    actual = rule_function_name(component)
    if actual != DROPPED_NETWORK_FAMILIES['sess_comp'][0]:
        raise RuntimeError(f'sess_comp: built from rule {actual!r}, expected sess_comp_rule')
    n = 0
    for cd in component.values():
        if not cd.active:
            continue
        repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
        ok = (repn.nonlinear_expr is None and not repn.linear_vars and float(pe.value(repn.constant)) == 0.0
              and len(repn.quadratic_vars) == 1 and float(pe.value(repn.quadratic_coefs[0])) == 1.0
              and cd.lower is None and cd.upper is not None
              and float(pe.value(cd.upper)) == float(mch.ESS_COMPLEMENTARITY_TOLERANCE))
        if ok:
            v1, v2 = repn.quadratic_vars[0]
            names = {v1.parent_component().local_name, v2.parent_component().local_name}
            ok = names == {'shared_es_pch_hat', 'shared_es_pdch_hat'} and v1.index() == v2.index()
        if not ok:
            raise RuntimeError(f'{cd.name}: not the BILINEAR_RELAXATION sess_comp row ruling 3 drops: {cd.expr}')
        n += 1
    return n


def nonconvex_active_rows(model):
    """Every ACTIVE row of `model` (descending into sub-blocks) whose form at the current Param values is not in
    CONVEX_FORMS, as [{'row', 'form'}]. Used after relaxation: a non-empty list is a stop (spec v2 section 10
    ruling 3), never a silent drop."""
    rows = []
    for cd in model.component_data_objects(pe.Constraint, active=True, descend_into=True):
        repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
        form = _quadratic_form_label(cd, repn)
        if form not in CONVEX_FORMS:
            rows.append({'row': cd.name, 'form': form})
    return rows


def _classify_esso_row(name, cd):
    """(class, sub-label) for one active ESSO row; raises if a degradation-list row matches no known form."""
    if name == 'available_e_capacity_unit':
        return 'dropped', 'available_e_bilinear'
    if name == 'energy_storage_cohort_pnet_share_h3':
        return 'dropped', 'h3_pro_rata'
    if name == 'energy_storage_capacity_degradation':
        repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
        variables = list(identify_variables(cd.body, include_fixed=True))
        parents = {v.parent_component().local_name for v in variables}
        if repn.nonlinear_expr is not None and 'exp(' in str(cd.body):
            return 'dropped', 'soh_exp_chain'
        if (not cd.equality and cd.lower is not None and cd.upper is None
                and parents == {'es_soh_per_unit_cumul'} and repn.nonlinear_expr is None):
            return 'dropped', 'soh_floor'
        if cd.equality and repn.nonlinear_expr is None and not repn.quadratic_vars and 'es_D_per_unit' in parents:
            return 'retained', 'D_row_linear'
        raise RuntimeError(f'unclassifiable ESSO degradation row {cd.name}: {cd.expr}')
    return 'retained', None


def inventory_rows(block, kind, drop_sess_comp=False):
    """Classify every ACTIVE constraint row of `block` (kind 'network' or 'esso').

    Network: class 'replaced' (definitional family, sub-label cone / decoupled / envelope; or a row referencing a
    transformer's r / r_sqr, sub-label 'envelope_rewrite'), else 'retained'. Nothing is dropped on a network block.
    ESSO: class 'dropped' (SoH exp chain, soh floor, bilinear available-e, H3; each with its LB-safety argument),
    else 'retained'.
    drop_sess_comp=True (T0.1b, spec v2 section 10 ruling 3): network rows of `sess_comp` are class 'dropped',
    sub-label 'sess_comp_nonconvex' (form verified by `_check_sess_comp_rows`). Default False = T0.1a behaviour.
    Each row also gets a form label (linear / convex_quadratic_psd / rotated_cone / lorentz_cone /
    nonconvex_quadratic / nonlinear) at the current Param values -- an audit only.
    Returns {'rows': [...], 'counts': {...}, ...}."""
    if kind not in ('network', 'esso'):
        raise ValueError(kind)
    rows = []
    extra = {}
    if kind == 'network':
        network, params = network_and_params(block)
        extra['definitional_rule_check'] = _check_definitional_rules(block)
        transformer_rows = _transformer_rows(block, network)
        extra['active_transformer_branches'] = active_transformer_branches(network)
        extra['n_transformer_branches_any_status'] = sum(1 for br in network.branches if br.is_transformer)
        if drop_sess_comp:
            extra['drop_sess_comp'] = True
            extra['sess_comp_rows_verified'] = _check_sess_comp_rows(block)
    for component in block.component_objects(pe.Constraint, active=True, descend_into=True):
        name = component.name
        for index, cd in component.items():
            if not cd.active:
                continue
            repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
            form = _quadratic_form_label(cd, repn)
            if kind == 'network':
                if name in DEFINITIONAL_FAMILIES:
                    cls, sub = 'replaced', DEFINITIONAL_FAMILIES[name][1]
                elif drop_sess_comp and name in DROPPED_NETWORK_FAMILIES:
                    cls, sub = 'dropped', DROPPED_NETWORK_FAMILIES[name][1]
                elif (name, index) in transformer_rows:
                    cls, sub = 'replaced', 'envelope_rewrite'
                else:
                    cls, sub = 'retained', None
            else:
                cls, sub = _classify_esso_row(name, cd)
            rows.append({'family': name, 'index': repr(index), 'class': cls, 'sub': sub, 'form': form})
    counts = {}
    for row in rows:
        key = (row['class'], row['sub'], row['family'])
        counts[key] = counts.get(key, 0) + 1
    count_list = [{'class': c, 'sub': s, 'family': f, 'rows': n} for (c, s, f), n in sorted(
        counts.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2]))]
    form_counts = {}
    for row in rows:
        key = (row['class'], row['family'], row['form'])
        form_counts[key] = form_counts.get(key, 0) + 1
    form_list = [{'class': c, 'family': f, 'form': fm, 'rows': n} for (c, f, fm), n in sorted(form_counts.items())]
    objectives = [{'name': o.name, 'active': bool(o.active)}
                  for o in block.component_data_objects(pe.Objective, active=None, descend_into=True)]
    return {'kind': kind, 'n_active_rows': len(rows), 'rows': rows, 'counts': count_list,
            'form_counts': form_list, 'objectives': objectives, **extra}


# ----------------------------------------------------------------------------------------------------------------------
#  Network relaxation
# ----------------------------------------------------------------------------------------------------------------------
def _branch_nodes(network, b):
    branch = network.branches[b]
    return network.get_node_idx(branch.fbus), network.get_node_idx(branch.tbus)


def _envelope_rewrite_expr(relaxed, cd, network, transformers):
    """Exact B2 rewrite of one production row: its degree-2 standard repn with r*WijR -> C, r*WijI -> D,
    r_sqr*Wii(from) -> U. Linear / constant parts are kept as produced by the repn (mutable Params stay symbolic).
    Refuses any other quadratic term, any nonlinear part and any linear occurrence of r / r_sqr."""
    repn = generate_standard_repn(cd.body, compute_values=False, quadratic=True)
    if repn.nonlinear_expr is not None:
        raise RuntimeError(f'{cd.name}: nonlinear part; B2 rewrite refused')
    expr = repn.constant
    for coef, v in zip(repn.linear_coefs, repn.linear_vars):
        parent = v.parent_component()
        if (parent is relaxed.r or parent is relaxed.r_sqr) and v.index()[0] in transformers:
            raise RuntimeError(f'{cd.name}: linear occurrence of {v.name}; B2 rewrite refused')
        expr = expr + coef * v
    for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        pair = {v1.parent_component().local_name: v1, v2.parent_component().local_name: v2}
        if 'r' in pair and 'voltage_product_real' in pair:
            idx = pair['r'].index()
            if pair['voltage_product_real'].index() != idx or idx[0] not in transformers:
                raise RuntimeError(f'{cd.name}: r*WijR index mismatch {v1.name}*{v2.name}')
            expr = expr + coef * relaxed.d0_tr_C[idx]
        elif 'r' in pair and 'voltage_product_imag' in pair:
            idx = pair['r'].index()
            if pair['voltage_product_imag'].index() != idx or idx[0] not in transformers:
                raise RuntimeError(f'{cd.name}: r*WijI index mismatch {v1.name}*{v2.name}')
            expr = expr + coef * relaxed.d0_tr_D[idx]
        elif 'r_sqr' in pair and 'vmag_sqr' in pair:
            idx = pair['r_sqr'].index()
            b, s_m, s_o, p = idx
            fnode, _tnode = _branch_nodes(network, b)
            if pair['vmag_sqr'].index() != (fnode, s_m, s_o, p) or b not in transformers:
                raise RuntimeError(f'{cd.name}: r_sqr*Wii not on the from-node: {v1.name}*{v2.name}')
            expr = expr + coef * relaxed.d0_tr_U[idx]
        else:
            raise RuntimeError(f'{cd.name}: quadratic term {v1.name}*{v2.name} has no B2 substitute')
    return expr


def relax_network_block(block, capacity_mode, rated_capacity=None, drop_sess_comp=False):
    """Clone `block` (never mutated) and build its W-space relaxation (spec v2 section 3 items 1-3, 5).

    capacity_mode 'available': `shared_es_s/e_rated_fixed` left at the values the block carried (variant A).
    capacity_mode 'rated': reset to `rated_capacity` = {e: {'s': pu, 'e': pu}} (variant R).
    drop_sess_comp=True (T0.1b, spec v2 section 10 ruling 3): the active `sess_comp` rows are verified to be the
    BILINEAR_RELAXATION row and deactivated (dropped, LB-safe). Default False = T0.1a behaviour.
    Returns (relaxed, record). `relaxed._d0_meta` carries the row maps `mapped_point` needs."""
    if capacity_mode not in ('rated', 'available'):
        raise ValueError(f'capacity_mode must be rated or available, got {capacity_mode!r}')
    _assert_no_d0_components(block)
    network, params = network_and_params(block)
    _check_definitional_rules(block)
    transformer_rows = _transformer_rows(block, network)
    relaxed = block.clone()
    record = {'capacity_mode': capacity_mode, 'network_name': network.name, 'year': network.year,
              'day': network.day, 'is_transmission': bool(network.is_transmission)}

    # 1. deactivate the five definitional families
    deactivated = {}
    for name in DEFINITIONAL_FAMILIES:
        component = getattr(relaxed, name)
        deactivated[name] = sum(1 for cd in component.values() if cd.active)
        component.deactivate()
    record['deactivated_definitional_rows'] = deactivated

    # 1b. (opt-in) drop the nonconvex sess_comp rows -- spec v2 section 10 ruling 3
    if drop_sess_comp:
        record['drop_sess_comp'] = True
        n_verified = _check_sess_comp_rows(relaxed)
        dropped = 0
        for cd in relaxed.sess_comp.values():
            if cd.active:
                cd.deactivate()
                dropped += 1
        if dropped != n_verified:
            raise RuntimeError(f'sess_comp: verified {n_verified} rows, dropped {dropped}')
        record['dropped_rows'] = {'sess_comp': dropped}
        record['drop_arguments'] = {'sess_comp': DROPPED_NETWORK_FAMILIES['sess_comp'][2]}

    # 2. rotated cone per active branch and period, on the production W variables
    active_branches = [b for b, br in enumerate(network.branches) if br.status]
    cone_index = [idx for idx in relaxed.voltage_product_real.index_set() if idx[0] in active_branches]

    def _cone_rule(m, b, s_m, s_o, p):
        fnode, tnode = _branch_nodes(network, b)
        return (m.voltage_product_real[b, s_m, s_o, p] ** 2 + m.voltage_product_imag[b, s_m, s_o, p] ** 2
                <= m.vmag_sqr[fnode, s_m, s_o, p] * m.vmag_sqr[tnode, s_m, s_o, p])
    relaxed.d0_cone_index = pe.Set(initialize=cone_index, dimen=4, ordered=True)
    relaxed.d0_cone = pe.Constraint(relaxed.d0_cone_index, rule=_cone_rule)
    record['n_cone_rows'] = len(relaxed.d0_cone)

    # 3. B2 envelope on transformer branches
    transformers = active_transformer_branches(network)
    tr_index = [idx for idx in relaxed.r.index_set() if idx[0] in transformers]
    relaxed.d0_tr_index = pe.Set(initialize=tr_index, dimen=4, ordered=True)
    relaxed.d0_tr_U = pe.Var(relaxed.d0_tr_index, domain=pe.NonNegativeReals, initialize=1.0)
    relaxed.d0_tr_C = pe.Var(relaxed.d0_tr_index, domain=pe.Reals, initialize=1.0)
    relaxed.d0_tr_D = pe.Var(relaxed.d0_tr_index, domain=pe.Reals, initialize=0.0)
    ratio_bounds = {}
    for idx in tr_index:
        r_var = relaxed.r[idx]
        if r_var.lb is None or r_var.ub is None or r_var.lb < 0.0:
            raise RuntimeError(f'transformer ratio {r_var.name} lacks finite nonnegative bounds: {r_var.bounds}')
        ratio_bounds[idx] = (float(r_var.lb), float(r_var.ub))

    def _u_lower(m, b, s_m, s_o, p):
        fnode, _ = _branch_nodes(network, b)
        return m.d0_tr_U[b, s_m, s_o, p] >= ratio_bounds[(b, s_m, s_o, p)][0] ** 2 * m.vmag_sqr[fnode, s_m, s_o, p]

    def _u_upper(m, b, s_m, s_o, p):
        fnode, _ = _branch_nodes(network, b)
        return m.d0_tr_U[b, s_m, s_o, p] <= ratio_bounds[(b, s_m, s_o, p)][1] ** 2 * m.vmag_sqr[fnode, s_m, s_o, p]

    def _tr_cone(m, b, s_m, s_o, p):
        _, tnode = _branch_nodes(network, b)
        return (m.d0_tr_C[b, s_m, s_o, p] ** 2 + m.d0_tr_D[b, s_m, s_o, p] ** 2
                <= m.d0_tr_U[b, s_m, s_o, p] * m.vmag_sqr[tnode, s_m, s_o, p])
    relaxed.d0_tr_U_lower = pe.Constraint(relaxed.d0_tr_index, rule=_u_lower)
    relaxed.d0_tr_U_upper = pe.Constraint(relaxed.d0_tr_index, rule=_u_upper)
    relaxed.d0_tr_cone = pe.Constraint(relaxed.d0_tr_index, rule=_tr_cone)

    # replace every active row that references a transformer's r / r_sqr (other than r_sqr_def itself)
    rewrite_rows = sorted((name, index) for (name, index) in transformer_rows if name not in DEFINITIONAL_FAMILIES)
    relaxed.d0_env_rewrite = pe.ConstraintList()
    envelope_map = {}
    for name, index in rewrite_rows:
        cd = getattr(relaxed, name)[index]
        new_body = _envelope_rewrite_expr(relaxed, cd, network, set(transformers))
        lower, upper = cd.lower, cd.upper
        if cd.equality:
            relaxed.d0_env_rewrite.add(new_body == upper)
        elif lower is not None and upper is not None:
            relaxed.d0_env_rewrite.add(pe.inequality(lower, new_body, upper))
        elif upper is not None:
            relaxed.d0_env_rewrite.add(new_body <= upper)
        else:
            relaxed.d0_env_rewrite.add(new_body >= lower)
        cd.deactivate()
        envelope_map[len(relaxed.d0_env_rewrite)] = (name, index)
    record['transformer_branches'] = transformers
    record['n_tr_index'] = len(tr_index)
    record['n_envelope_rewrite_rows'] = len(rewrite_rows)
    record['envelope_rewrite_rows_by_family'] = {}
    for name, _index in rewrite_rows:
        record['envelope_rewrite_rows_by_family'][name] = record['envelope_rewrite_rows_by_family'].get(name, 0) + 1
    record['transformer_ratio_bounds'] = sorted({v for v in ratio_bounds.values()})

    # 4. capacity Params
    capacity_before = {e: {'s': float(pe.value(relaxed.shared_es_s_rated_fixed[e])),
                           'e': float(pe.value(relaxed.shared_es_e_rated_fixed[e]))}
                       for e in relaxed.shared_es_s_rated_fixed}
    if capacity_mode == 'rated':
        if rated_capacity is None or set(rated_capacity) != set(capacity_before):
            raise ValueError(f'rated mode needs rated_capacity for every shared ESS {sorted(capacity_before)}')
        for e, cap in rated_capacity.items():
            relaxed.shared_es_s_rated_fixed[e].set_value(cap['s'])
            relaxed.shared_es_e_rated_fixed[e].set_value(cap['e'])
    record['capacity_params_carried'] = capacity_before
    record['capacity_params_set'] = {e: {'s': float(pe.value(relaxed.shared_es_s_rated_fixed[e])),
                                         'e': float(pe.value(relaxed.shared_es_e_rated_fixed[e]))}
                                     for e in relaxed.shared_es_s_rated_fixed}

    # 5. objectives: every production / ADMM objective deactivated; the LB objective is attached by the caller
    deactivated_objectives = []
    for o in relaxed.component_objects(pe.Objective, active=True, descend_into=True):
        deactivated_objectives.append(o.name)
        o.deactivate()
    record['deactivated_objectives'] = deactivated_objectives

    # structural check: e, f appear in no active row; r / r_sqr of transformers appear in no active row
    rank1_left = []
    for cd in relaxed.component_data_objects(pe.Constraint, active=True, descend_into=True):
        for v in identify_variables(cd.body, include_fixed=True):
            parent = v.parent_component()
            if parent is relaxed.e or parent is relaxed.f:
                rank1_left.append((cd.name, v.name))
                break
            if (parent is relaxed.r or parent is relaxed.r_sqr) and v.index()[0] in transformers:
                rank1_left.append((cd.name, v.name))
                break
    record['active_rows_referencing_e_f_or_transformer_r'] = len(rank1_left)
    if rank1_left:
        raise RuntimeError(f'rank-1 variables remain in active rows: {rank1_left[:5]}')

    relaxed._d0_meta = {'kind': 'network', 'network': network, 'params': params, 'envelope_map': envelope_map,
                        'transformers': transformers, 'capacity_mode': capacity_mode}
    return relaxed, record


# ----------------------------------------------------------------------------------------------------------------------
#  ESSO relaxation
# ----------------------------------------------------------------------------------------------------------------------
def relax_esso_block(esso_model):
    """Clone the ESSO (never mutated) and apply spec v2 section 3 item 4. Returns (relaxed, record)."""
    _assert_no_d0_components(esso_model)
    relaxed = esso_model.clone()
    record = {'dropped_rows': {}, 'drop_arguments': ESSO_DROP_ARGUMENTS}
    available_pairs = []
    for component in list(relaxed.component_objects(pe.Constraint, active=True, descend_into=True)):
        name = component.name
        drop_all_rows = name == 'energy_storage_cohort_pnet_share_h3'
        for index, cd in component.items():
            if drop_all_rows:
                if cd.active:
                    record['dropped_rows']['h3_pro_rata'] = record['dropped_rows'].get('h3_pro_rata', 0) + 1
                cd.deactivate()
                continue
            if not cd.active:
                continue
            cls, sub = _classify_esso_row(name, cd)
            if cls == 'dropped':
                if sub == 'available_e_bilinear':
                    avail = [v for v in identify_variables(cd.body, include_fixed=True)
                             if v.parent_component().local_name == 'es_e_available_per_unit']
                    if len(avail) != 1:
                        raise RuntimeError(f'{cd.name}: expected one es_e_available_per_unit, got {avail}')
                    available_pairs.append(avail[0].index())
                cd.deactivate()
                record['dropped_rows'][sub] = record['dropped_rows'].get(sub, 0) + 1
    record['h3_rows_total_deactivated'] = len(relaxed.energy_storage_cohort_pnet_share_h3)

    relaxed.d0_available_e_eq_rated = pe.ConstraintList()
    for pair in sorted(available_pairs):
        relaxed.d0_available_e_eq_rated.add(
            relaxed.es_e_available_per_unit[pair] == relaxed.es_e_rated_per_unit[pair])
    record['available_eq_rated_pairs'] = [list(p) for p in sorted(available_pairs)]

    slack_record = {}
    for name in ESSO_SLACK_VARS:
        var = getattr(relaxed, name, None)
        if var is None:
            raise RuntimeError(f'ESSO slack family {name} absent')
        values = [float(v.value) for v in var.values() if v.value is not None]
        slack_record[name] = {'n': len(var), 'snapshot_max_abs': max((abs(x) for x in values), default=0.0),
                              'snapshot_sum': float(sum(values))}
        for v in var.values():
            v.fix(0.0)
    record['slacks_fixed_at_zero'] = slack_record

    deactivated_objectives = []
    for o in relaxed.component_objects(pe.Objective, active=True, descend_into=True):
        deactivated_objectives.append(o.name)
        o.deactivate()
    record['deactivated_objectives'] = deactivated_objectives
    # gross excludes the ESSO (spec section 3 item 5): its LB-objective contribution is identically zero. The LP
    # writer requires one active objective, so a constant zero objective is attached.
    relaxed.d0_lb_objective = pe.Objective(expr=0.0, sense=pe.minimize)

    nonlinear_left = []
    for cd in relaxed.component_data_objects(pe.Constraint, active=True, descend_into=True):
        if generate_standard_repn(cd.body, compute_values=True, quadratic=True).nonlinear_expr is not None:
            nonlinear_left.append(cd.name)
    record['active_nonlinear_rows_left'] = len(nonlinear_left)
    if nonlinear_left:
        raise RuntimeError(f'non-polynomial rows remain active in the relaxed ESSO: {nonlinear_left[:5]}')
    relaxed._d0_meta = {'kind': 'esso', 'available_pairs': sorted(available_pairs)}
    return relaxed, record


# ----------------------------------------------------------------------------------------------------------------------
#  LB objective
# ----------------------------------------------------------------------------------------------------------------------
class _PreSettlementFixtureView:
    """Read-only view used ONLY for blocks that carry no `interface_settlement_weight` (built before P5.15
    Step 3.1-C): such a block has no settlement term in its objective at all, so the two absent attributes that
    `objective_function_rule` reads are supplied as 0; every other attribute is the block's own."""

    _ABSENT = ('interface_settlement_weight', 'interface_settlement')

    def __init__(self, model):
        object.__setattr__(self, '_model', model)

    def __getattr__(self, name):
        if name in self._ABSENT:
            return 0.0
        return getattr(self._model, name)


def _contracted_settlement_expr(model):
    """Expression form of `srp._get_local_interface_settlement(model, part='contracted')` (same selection)."""
    if not hasattr(model, 'interface_settlement_weight'):
        return 0.0
    if not hasattr(model, 'interface_settlement_contracted'):
        return model.interface_settlement_weight * model.interface_settlement
    return model.interface_settlement_weight * model.interface_settlement_contracted


def _voltage_pin_expr(model):
    """Expression form of `srp._get_local_voltage_pin(model)` (same selection)."""
    if not hasattr(model, 'scenario_voltage_pin'):
        return 0.0
    return model.scenario_voltage_pin_weight * model.scenario_voltage_pin


def block_gross_expr(model, params):
    """objective_function_rule(b) - contracted settlement_b - voltage pin_b, unweighted, as a Pyomo expression."""
    legacy = not hasattr(model, 'interface_settlement_weight')
    target = _PreSettlementFixtureView(model) if legacy else model
    expr = mch.objective_function_rule(target, params) - _contracted_settlement_expr(model) - _voltage_pin_expr(model)
    return expr, legacy


def lb_objective(blocks):
    """blocks: iterable of dicts {'model', 'params', 'network_data', 'year', 'day', 'label'}.

    Returns (total_expr, per_block) with total_expr = Sigma_b w_b * gross_b(expr), w_b from
    `srp._get_admm_block_weight(network_data, year, day)`. ESSO blocks contribute nothing (gross excludes them)."""
    total = 0.0
    per_block = []
    for blk in blocks:
        weight = srp._get_admm_block_weight(blk['network_data'], blk['year'], blk['day'])
        expr, legacy = block_gross_expr(blk['model'], blk['params'])
        total = total + weight * expr
        per_block.append({'label': blk['label'], 'weight': float(weight), 'expr': expr,
                          'pre_settlement_fixture_view_used': legacy})
    return total, per_block


def attach_lb_objective(relaxed, weight, params):
    """Attach `d0_lb_objective` = weight * gross(relaxed) to one relaxed network block (for export)."""
    expr, legacy = block_gross_expr(relaxed, params)
    relaxed.d0_lb_objective = pe.Objective(expr=weight * expr, sense=pe.minimize)
    return legacy


# ----------------------------------------------------------------------------------------------------------------------
#  Mapped point and identities
# ----------------------------------------------------------------------------------------------------------------------
def _set_network_mapped_point(relaxed, w_source='ef'):
    """w_source 'ef' (the task's mapped point): W from the snapshot's e, f -- exactly rank-1.
    w_source 'snapshot' (diagnostic only): W kept at the snapshot's own values (rank-1 only up to the definitional
    rows' residuals at the snapshot); U, C, D then from the snapshot's r_sqr, r and W."""
    if w_source not in ('ef', 'snapshot'):
        raise ValueError(w_source)
    network = relaxed._d0_meta['network']
    e, f = relaxed.e, relaxed.f
    if w_source == 'ef':
        for (i, s_m, s_o, p), w in relaxed.vmag_sqr.items():
            w.set_value(e[i, s_m, s_o, p].value ** 2 + f[i, s_m, s_o, p].value ** 2, skip_validation=True)
        for (b, s_m, s_o, p), wr in relaxed.voltage_product_real.items():
            fnode, tnode = _branch_nodes(network, b)
            ef, et = e[fnode, s_m, s_o, p].value, e[tnode, s_m, s_o, p].value
            ff, ft = f[fnode, s_m, s_o, p].value, f[tnode, s_m, s_o, p].value
            wr.set_value(ef * et + ff * ft, skip_validation=True)
            relaxed.voltage_product_imag[b, s_m, s_o, p].set_value(ff * et - ef * ft, skip_validation=True)
    for idx in relaxed.d0_tr_index:
        b, s_m, s_o, p = idx
        fnode, _ = _branch_nodes(network, b)
        r = relaxed.r[idx].value
        r_sqr = r ** 2 if w_source == 'ef' else relaxed.r_sqr[idx].value
        relaxed.d0_tr_U[idx].set_value(r_sqr * relaxed.vmag_sqr[fnode, s_m, s_o, p].value, skip_validation=True)
        relaxed.d0_tr_C[idx].set_value(r * relaxed.voltage_product_real[idx].value, skip_validation=True)
        relaxed.d0_tr_D[idx].set_value(r * relaxed.voltage_product_imag[idx].value, skip_validation=True)


def _set_esso_mapped_point(relaxed):
    for pair in relaxed._d0_meta['available_pairs']:
        relaxed.es_e_available_per_unit[pair].set_value(relaxed.es_e_rated_per_unit[pair].value,
                                                        skip_validation=True)
    # slack_es_pnet_up/down are fixed at 0 by relax_esso_block (their value is 0 at the mapped point)


def _mutable_param_families(cd):
    """Local names of the mutable-Param components a row's expression (body and bounds) references."""
    return {prm.parent_component().local_name for prm in identify_mutable_parameters(cd.expr)}


_W_FAMILIES = ('vmag_sqr', 'voltage_product_real', 'voltage_product_imag')
_EPS = np.finfo(float).eps


def _row_scale_and_w_bound(cd, relaxed):
    """(evaluation resolution, first-order W bound) of one production row at the snapshot point.

    evaluation resolution = eps * (|constant| + sum |c_k x_k| + sum |c_kl x_k x_l|): the floating-point resolution
    of the row's own body at that point (catastrophic cancellation makes |body| << the terms).
    first-order W bound = sum over W variables of |d body / d W| * |W(mapped) - W(snapshot)| (rows retained on a
    network block are linear in W)."""
    repn = generate_standard_repn(cd.body, compute_values=True, quadratic=True)
    scale = abs(float(pe.value(repn.constant)))
    w_bound = 0.0
    for coef, v in zip(repn.linear_coefs, repn.linear_vars):
        c = float(pe.value(coef))
        scale += abs(c * v.value) if v.value is not None else 0.0
        if v.parent_component().local_name in _W_FAMILIES:
            w_bound += abs(c) * abs(relaxed.find_component(v.name).value - v.value)
    for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        scale += abs(float(pe.value(coef)) * v1.value * v2.value)
    if repn.nonlinear_expr is not None:
        scale += abs(float(pe.value(repn.nonlinear_expr)))
    return _EPS * scale, w_bound


def production_rows_at_point(block, relaxed):
    """Same-point check: clone the PRODUCTION block, copy every Var value and mutable Param value of the relaxed
    block into it, and return max |residual(production row) - residual(relaxed row)| over the retained rows.
    Zero means the retained relaxed row is the production row (construction identity), independent of where the
    mapped point lies."""
    twin = block.clone()
    for var in twin.component_data_objects(pe.Var, descend_into=True):
        src = relaxed.find_component(var.name)
        if src is not None:
            var.set_value(src.value, skip_validation=True)
    for param in twin.component_objects(pe.Param, descend_into=True):
        if not param.mutable:
            continue
        src = relaxed.find_component(param.name)
        for index in param:
            param[index].set_value(pe.value(src[index]))
    worst, n = 0.0, 0
    for component in twin.component_objects(pe.Constraint, active=True, descend_into=True):
        relaxed_component = relaxed.find_component(component.name)
        for index, cd in component.items():
            rcd = relaxed_component[index]
            if not (cd.active and rcd.active):
                continue
            worst = max(worst, residual_delta(row_residual(cd), row_residual(rcd)))
            n += 1
    return {'n_rows': n, 'max_abs_delta': worst}


def mapped_point(block, relaxed, w_source='ef'):
    """Set the relaxed block's lifted variables at the rank-1 image of the snapshot point and return the per-row
    residual tables.

    Network: Wii = e^2 + f^2, WijR = e_i e_j + f_i f_j, WijI = f_i e_j - e_i f_j (from the snapshot's e, f);
    U = r^2 Wii(from), C = r WijR, D = r WijI on transformers; every other variable keeps the snapshot value.
    ESSO: E_avail = E_rated on the replaced pairs; slacks at 0 (fixed); every other variable keeps its value.

    Tables (rows of dicts):
      retained   -- every active retained row: residual(relaxed at mapped) vs residual(production at snapshot)
      envelope   -- every B2-rewritten row vs the production row it replaced (network)
      cones      -- relative residual (Wii Wjj - WijR^2 - WijI^2) / (Wii Wjj) per cone row, and the transformer cone
      envelope_bounds -- (U - r_lb^2 Wii)/Wii and (r_ub^2 Wii - U)/Wii (must be >= 0)
      replaced_production -- residual of each deactivated production row at the snapshot (information)
      added_esso -- E_avail == E_rated rows at the mapped point (ESSO)
    """
    meta = relaxed._d0_meta
    if meta['kind'] == 'network':
        _set_network_mapped_point(relaxed, w_source)
    else:
        _set_esso_mapped_point(relaxed)

    capacity_params = {'shared_es_s_rated_fixed', 'shared_es_e_rated_fixed'}
    retained, replaced_prod = [], []
    for component in block.component_objects(pe.Constraint, active=True, descend_into=True):
        relaxed_component = relaxed.find_component(component.name)
        for index, cd in component.items():
            if not cd.active:
                continue
            rcd = relaxed_component[index]
            prod = row_residual(cd)
            if rcd.active:
                rel = row_residual(rcd)
                touches_capacity = bool(_mutable_param_families(cd) & capacity_params)
                resolution, w_bound = _row_scale_and_w_bound(cd, relaxed)
                retained.append({'family': component.name, 'index': repr(index),
                                 'prod': prod, 'relaxed': rel, 'delta': residual_delta(prod, rel),
                                 'eval_resolution': resolution, 'first_order_w_bound': w_bound,
                                 'references_capacity_params': touches_capacity})
            else:
                replaced_prod.append({'family': component.name, 'index': repr(index), 'prod': prod})

    envelope, cones, bounds, added = [], [], [], []
    if meta['kind'] == 'network':
        for k, (name, index) in meta['envelope_map'].items():
            prod = row_residual(getattr(block, name)[index])
            rel = row_residual(relaxed.d0_env_rewrite[k])
            envelope.append({'family': name, 'index': repr(index), 'prod': prod, 'relaxed': rel,
                             'delta': residual_delta(prod, rel)})
        network = meta['network']
        for (b, s_m, s_o, p) in relaxed.d0_cone_index:
            fnode, tnode = _branch_nodes(network, b)
            wii = relaxed.vmag_sqr[fnode, s_m, s_o, p].value
            wjj = relaxed.vmag_sqr[tnode, s_m, s_o, p].value
            wr = relaxed.voltage_product_real[b, s_m, s_o, p].value
            wi = relaxed.voltage_product_imag[b, s_m, s_o, p].value
            cones.append({'kind': 'W', 'index': repr((b, s_m, s_o, p)),
                          'rel_residual': (wii * wjj - wr ** 2 - wi ** 2) / (wii * wjj)})
        for idx in relaxed.d0_tr_index:
            b, s_m, s_o, p = idx
            fnode, tnode = _branch_nodes(network, b)
            u, c, d = (relaxed.d0_tr_U[idx].value, relaxed.d0_tr_C[idx].value, relaxed.d0_tr_D[idx].value)
            wii = relaxed.vmag_sqr[fnode, s_m, s_o, p].value
            wjj = relaxed.vmag_sqr[tnode, s_m, s_o, p].value
            cones.append({'kind': 'transformer_UCD', 'index': repr(idx),
                          'rel_residual': (u * wjj - c ** 2 - d ** 2) / (u * wjj)})
            r_lb, r_ub = relaxed.r[idx].lb, relaxed.r[idx].ub
            bounds.append({'index': repr(idx), 'lower_slack_rel': (u - r_lb ** 2 * wii) / wii,
                           'upper_slack_rel': (r_ub ** 2 * wii - u) / wii,
                           'lower_row_residual': row_residual(relaxed.d0_tr_U_lower[idx]),
                           'upper_row_residual': row_residual(relaxed.d0_tr_U_upper[idx])})
    else:
        for k, cd in relaxed.d0_available_e_eq_rated.items():
            added.append({'index': k, 'residual': row_residual(cd)})
    return {'retained': retained, 'envelope': envelope, 'cones': cones, 'envelope_bounds': bounds,
            'replaced_production': replaced_prod, 'added_esso': added}


# ----------------------------------------------------------------------------------------------------------------------
#  Export
# ----------------------------------------------------------------------------------------------------------------------
def export_lp(relaxed, path):
    """Write the relaxed block to CPLEX-LP `path` with symbolic labels; return (sha256, name_map, counts).

    Requires exactly one active objective (`d0_lb_objective`). The name map is read from the writer's own symbol
    map: LP symbol -> Pyomo component-data name, split into variables / constraints / objectives."""
    active_objectives = [o.name for o in relaxed.component_data_objects(pe.Objective, active=True)]
    if active_objectives != ['d0_lb_objective']:
        raise RuntimeError(f'export needs exactly d0_lb_objective active, found {active_objectives}')
    _filename, smap_id = relaxed.write(path, io_options={'symbolic_solver_labels': True})
    symbol_map = relaxed.solutions.symbol_map[smap_id]
    name_map = {'variables': {}, 'constraints': {}, 'objectives': {}, 'other': {}}
    for symbol, obj in symbol_map.bySymbol.items():
        target = obj() if callable(obj) and not isinstance(obj, (VarData, ConstraintData, ObjectiveData)) else obj
        if isinstance(target, VarData):
            name_map['variables'][symbol] = target.name
        elif isinstance(target, ConstraintData):
            name_map['constraints'][symbol] = target.name
        elif isinstance(target, ObjectiveData):
            name_map['objectives'][symbol] = target.name
        else:
            name_map['other'][symbol] = str(target)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    counts = {k: len(v) for k, v in name_map.items()}
    return h.hexdigest(), name_map, counts


def objective_form(expr):
    """Form of a minimisation objective at the current Param values: 'linear', 'convex_quadratic_psd',
    'nonconvex_quadratic' or 'nonlinear', with the minimum eigenvalue of its quadratic part (an audit only)."""
    repn = generate_standard_repn(expr, compute_values=True, quadratic=True)
    if repn.nonlinear_expr is not None:
        return {'form': 'nonlinear', 'min_eig': None, 'n_quadratic_terms': len(repn.quadratic_vars)}
    if not repn.quadratic_vars:
        return {'form': 'linear', 'min_eig': None, 'n_quadratic_terms': 0}
    variables, position = [], {}
    for v1, v2 in repn.quadratic_vars:
        for v in (v1, v2):
            if id(v) not in position:
                position[id(v)] = len(variables)
                variables.append(v)
    q = np.zeros((len(variables), len(variables)))
    bilinear = []
    for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        c = float(pe.value(coef))
        i, j = position[id(v1)], position[id(v2)]
        if i == j:
            q[i, i] += c
        else:
            q[i, j] += 0.5 * c
            q[j, i] += 0.5 * c
            bilinear.append((v1.parent_component().local_name, v2.parent_component().local_name))
    min_eig = float(np.linalg.eigvalsh(q).min())
    form = 'convex_quadratic_psd' if min_eig >= -1e-12 * max(1.0, float(abs(q).max())) else 'nonconvex_quadratic'
    families = sorted({tuple(sorted(pair)) for pair in bilinear})
    return {'form': form, 'min_eig': min_eig, 'n_quadratic_terms': len(repn.quadratic_vars),
            'bilinear_families': [list(pair) for pair in families]}
