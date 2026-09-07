import os
import sys
import json
import hashlib
import psutil
import pyomo.environ as pe
import pyomo.opt as po
from copulas.univariate import GaussianKDE
from pyomo.core.expr.visitor import polynomial_degree
from definitions import *


def read_json_file(filename):
    try:
        input_file = open(filename, 'r')
    except OSError:
        print(f'[ERROR] Could not open file {filename}. Exiting...')
        sys.exit()

    with input_file:
        file_contents = input_file.read()
        return file_contents


def convert_json_to_dict(json_string):
    try:
        data_dict = json.loads(json_string)
        return data_dict
    except json.JSONDecodeError as e:
        print(f'[ERROR] Could not convert JSON to dict. JSONDecodeError: {e}')
        exit(ERROR_SPECIFICATION_FILE)
    except TypeError as e:
        print(f'[ERROR] Could not convert JSON to dict. TypeError: {e}')
        exit(ERROR_SPECIFICATION_FILE)


def derive_random_seed(base_seed, *labels):
    if base_seed is None:
        return None
    payload = json.dumps([int(base_seed), *labels], separators=(',', ':'), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='big', signed=False)


def is_int(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def print_memory_usage(label="", debug=False):
    if debug:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 ** 2)  # MB
        vms = process.memory_info().vms / (1024 ** 2)  # MB
        print(f"[MEMORY] {label} - RSS: {rss:.2f} MB | VMS: {vms:.2f} MB")


def log_debug(message, debug=False):
    if debug:
        print(f"[DEBUG]\t{message}")


def solver_result_succeeded(result):
    if result is None or not hasattr(result, 'solver'):
        return False
    accepted_termination_conditions = {
        po.TerminationCondition.optimal,
        po.TerminationCondition.locallyOptimal,
        po.TerminationCondition.globallyOptimal,
    }
    return (
        result.solver.status == po.SolverStatus.ok
        and result.solver.termination_condition in accepted_termination_conditions
    )


def solver_result_summary(result):
    if result is None or not hasattr(result, 'solver'):
        return 'no SolverResults returned'
    status = getattr(result.solver, 'status', 'unknown')
    termination = getattr(result.solver, 'termination_condition', 'unknown')
    message = getattr(result.solver, 'message', None)
    summary = f'status={status}, termination={termination}'
    if message is not None and str(message) not in ('', '<undefined>'):
        summary += f', message={message}'
    return summary


def report_out_of_bound_initial_values(model, tol=0.0):
    """
    Scan all active variables and report those whose .value is outside [lb, ub].
    tol can be used to ignore tiny numerical differences.
    """
    print("Checking variable initial values against bounds...")
    count = 0
    for v in model.component_data_objects(pe.Var, active=True, descend_into=True):
        val = v.value
        lb = v.lb
        ub = v.ub

        # If not initialized, skip
        if val is None:
            continue

        # Compare with tolerance
        lower_violation = (lb is not None) and (val < lb - tol)
        upper_violation = (ub is not None) and (val > ub + tol)

        if lower_violation or upper_violation:
            count += 1
            print(f"- Var {v.name}: value={val}, lb={lb}, ub={ub}")

    if count == 0:
        print("All initialized variable values are within their bounds.")
    else:
        print(f"Total variables with out-of-bound initial values: {count}")


def print_model_polynomial_degrees(model, model_name):

    degree_counts = {}
    offenders = []

    for obj in model.component_data_objects(pe.Objective, active=True, descend_into=True):
        degree = polynomial_degree(obj.expr)
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        if degree is None or degree > 2:
            offenders.append(('Objective', obj.name, degree))

    for con in model.component_data_objects(pe.Constraint, active=True, descend_into=True):
        degree = polynomial_degree(con.body)
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        if degree is None or degree > 2:
            offenders.append(('Constraint', con.name, degree))

    print(f'[DEBUG][POLYNOMIAL DEGREE] model={model_name}, counts={degree_counts}')

    if offenders:
        print('[WARNING] Non-QCQP expressions detected:')
        for component_type, name, degree in offenders[:50]:
            print(f'  {component_type}: {name}, degree={degree}')
    else:
        print(f'[DEBUG][POLYNOMIAL DEGREE] model={model_name}: all active expressions have degree <= 2.')


def fix_or_set(var, val):
    if var.fixed:
        var.set_value(val)
    else:
        var.fix(val)


def write_value(sheet, line_idx, col_idx, value, number_format=None):
    sheet.cell(row=line_idx, column=col_idx).value = value
    if number_format:
        sheet.cell(row=line_idx, column=col_idx).number_format = number_format


class CustomGaussianKDE(GaussianKDE):

    def __init__(self, bandwidth=0.2):
        super().__init__()
        self.custom_bandwidth = bandwidth

    def _fit(self, X):
        super()._fit(X)
        self.kde.set_bandwidth(bw_method=self.custom_bandwidth)
