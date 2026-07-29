import os
import sys
import json
from copy import deepcopy
import psutil
import pyomo.environ as pe
from copulas.univariate import GaussianKDE
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


def fix_or_set(var, val):
    if var.fixed:
        var.set_value(val)
    else:
        var.fix(val)


def write_value(sheet, line_idx, col_idx, value, number_format=None):
    sheet.cell(row=line_idx, column=col_idx).value = value
    if number_format:
        sheet.cell(row=line_idx, column=col_idx).number_format = number_format


def get_present_worth_factor(representative_year, base_year, num_years, discount_rate):
    num_years = int(num_years)
    year_offset = int(representative_year) - int(base_year)
    return sum(1.0 / ((1.0 + discount_rate) ** (year_offset + offset)) for offset in range(num_years))


def finite_difference_sensitivity_test(planning_problem, candidate_solution, base_recourse_value, dual_sensitivities, node_id, year, capacity_type, epsilon_values=(0.001, 0.01, 0.05)):
    """
    Compare the aggregated dual sensitivity against forward finite
    differences of the complete operational recourse function.

    capacity_type:
        "s" for MVA
        "e" for MWh
    """

    if capacity_type not in ("s", "e"):
        raise ValueError("capacity_type must be either 's' or 'e'.")

    dual_sensitivity = (dual_sensitivities[capacity_type][year][node_id])
    print(f"\n[FD TEST] node={node_id}, year={year} type={capacity_type}")
    print(f"[FD TEST] Dual sensitivity: {dual_sensitivity:.8f}")

    results = []
    for epsilon in epsilon_values:

        candidate_perturbed = deepcopy(candidate_solution)
        candidate_perturbed["total_capacity"][node_id][year][capacity_type] += epsilon

        perturbed_convergence, _, perturbed_models, _, _ = planning_problem.run_operational_planning(candidate_solution=candidate_perturbed, print_results=False, filename=(f"{planning_problem.name}_fd_{capacity_type}_{node_id}_{year}_{epsilon}"))

        if not perturbed_convergence:
            print(f"[FD TEST] epsilon={epsilon}: operational problem did not converge.")
            continue

        perturbed_recourse_value = planning_problem.get_operational_recourse_value(perturbed_models["tso"])
        finite_difference = (perturbed_recourse_value - base_recourse_value) / epsilon
        scale = max(1.0, abs(finite_difference), abs(dual_sensitivity))
        relative_difference = (abs(finite_difference - dual_sensitivity) / scale )

        results.append(
            {
                "epsilon": epsilon,
                "dual": dual_sensitivity,
                "finite_difference": finite_difference,
                "relative_difference": relative_difference,
            }
        )

        print(
            f"[FD TEST] epsilon={epsilon:.6f} | "
            f"FD={finite_difference:.8f} | "
            f"dual={dual_sensitivity:.8f} | "
            f"relative difference="
            f"{100.0 * relative_difference:.2f}%"
        )

    return results



class CustomGaussianKDE(GaussianKDE):

    def __init__(self, bandwidth=0.2):
        super().__init__()
        self.custom_bandwidth = bandwidth

    def _fit(self, X):
        super()._fit(X)
        self.kde.set_bandwidth(bw_method=self.custom_bandwidth)

