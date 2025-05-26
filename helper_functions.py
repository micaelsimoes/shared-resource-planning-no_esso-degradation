import os
import sys
import json
import psutil
import pyomo.environ as pe
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


def print_memory_usage(label=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 ** 2)  # MB
    vms = process.memory_info().vms / (1024 ** 2)  # MB
    print(f"[MEMORY] {label} - RSS: {rss:.2f} MB | VMS: {vms:.2f} MB")


def init_and_fix_var(name, value):
    v = pe.Var(domain=pe.NonNegativeReals)
    v.fix(value)
    return v


def fix_or_set(var, val):
    if var.fixed:
        var.set_value(val)
    else:
        var.fix(val)
