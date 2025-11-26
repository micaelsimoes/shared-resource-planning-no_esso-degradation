import os
import sys
import json
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


class CustomGaussianKDE(GaussianKDE):

    def __init__(self, bandwidth=0.2):
        super().__init__()
        self.custom_bandwidth = bandwidth

    def _fit(self, X):
        super()._fit(X)
        self.kde.set_bandwidth(bw_method=self.custom_bandwidth)

