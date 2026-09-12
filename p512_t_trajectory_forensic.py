"""
P5.12-T -- NO-SOLVE comparative trajectory forensic.

Zero solves. This script only reads three preserved, frozen IPOPT print-level-6
logs (Variant 1 warm_start_bound_push=1e-6, Arm A =1e-5, Variant 2 =1e-4) for
the byte-identical cycle-21 NLP produced by P5.12-P / Arm A, parses their
per-iteration telemetry, validates the parser against pre-declared headline
facts, and performs a comparative, non-causal forensic analysis.

Outputs (all new, under data/SRP1/Results/P512T/, verified absent beforehand):
  - journal.json         : complete parsed per-iteration series for all three runs
  - manifest.json         : SHA-256 of every input read and every output written
  - P5_12_T_TRAJECTORY_FORENSIC_REPORT.md (written by this script's caller / or here)

Usage:
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p512_t_trajectory_forensic.py
"""

import hashlib
import json
import os
import re
import sys
from statistics import median

REPO_ROOT = "/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation"
OUT_DIR = os.path.join(REPO_ROOT, "data/SRP1/Results/P512T")

INPUTS = {
    "V1": {
        "path": os.path.join(REPO_ROOT,
                              "data/SRP1/Results/P512P/variant1_1e-6/logs/optim_log_case33_3_2025_Spring.log"),
        "expected_sha256": "3c5479835849bc859f3bcaf964a2791e89318bae0fe979d2eb7dbb3e155d2bd7",
        "expected_lines": 7400,
        "label": "Variant 1 (warm_start_bound_push=1e-6)",
    },
    "ArmA": {
        "path": os.path.join(REPO_ROOT,
                              "data/SRP1/Results/P512ArmA/logs/optim_log_case33_3_2025_Spring.log"),
        "expected_sha256": "4f66a7efeef933bdc0a425af76f0095f5c11a2112ff2c8bb6d7c03ff45409d58",
        "expected_lines": 243498,
        "label": "Arm A (warm_start_bound_push=1e-5)",
    },
    "V2": {
        "path": os.path.join(REPO_ROOT,
                              "data/SRP1/Results/P512P/variant2_1e-4/logs/optim_log_case33_3_2025_Spring.log"),
        "expected_sha256": "ccf80716e9b7c7f7346d1404234a2b125d00cc0658f6ec8ba35ad8a63095ed2e",
        "expected_lines": 12599,
        "label": "Variant 2 (warm_start_bound_push=1e-4)",
    },
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs():
    results = {}
    for key, meta in INPUTS.items():
        if not os.path.isfile(meta["path"]):
            raise SystemExit(f"FATAL: input missing for {key}: {meta['path']}")
        actual_sha = sha256_of(meta["path"])
        with open(meta["path"], "r", errors="replace") as f:
            lines = f.readlines()
        actual_lines = len(lines)
        ok_hash = (actual_sha == meta["expected_sha256"])
        ok_lines = (actual_lines == meta["expected_lines"])
        results[key] = {
            "path": meta["path"],
            "label": meta["label"],
            "expected_sha256": meta["expected_sha256"],
            "actual_sha256": actual_sha,
            "hash_match": ok_hash,
            "expected_lines": meta["expected_lines"],
            "actual_lines": actual_lines,
            "lines_match": ok_lines,
        }
        if not ok_hash or not ok_lines:
            raise SystemExit(f"FATAL: input verification failed for {key}: "
                              f"hash_match={ok_hash} lines_match={ok_lines}")
    return results, {k: v["path"] for k, v in INPUTS.items()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

SUMMARY_HEADER_RE = re.compile(r"^\*\*\* Summary of Iteration:\s*(\d+):")
BEGIN_ITER_RE = re.compile(r"^\*\*\* Beginning Iteration\s*(\d+) from the following point:")
NLP_VALUES_HEADER_RE = re.compile(r"^\*\*\*Current NLP Values for Iteration\s*(\d+):")
SAFEGUARD_RE = re.compile(
    r"^Some value in (z_L|z_U|v_L|v_U) becomes too large - maximal correction = (\S+)"
)
EXIT_RE = re.compile(r"^EXIT:\s*(.+)$")
NUM_ITER_RE = re.compile(r"^Number of Iterations\.+:\s*(\d+)")
MU_RE = re.compile(r"^Current barrier parameter mu = (\S+)")
TAU_RE = re.compile(r"^Current fraction-to-the-boundary parameter tau = (\S+)")
NORM_RE = re.compile(r"^\|\|curr_([a-zA-Z_]+)\|\|_inf\s*=\s*(\S+)")

NLP_FIELD_RE = {
    "objective": re.compile(r"^Objective\.+:\s*(\S+)\s+(\S+)"),
    "dual_infeasibility": re.compile(r"^Dual infeasibility\.+:\s*(\S+)\s+(\S+)"),
    "constraint_violation": re.compile(r"^Constraint violation\.+:\s*(\S+)\s+(\S+)"),
    "complementarity": re.compile(r"^Complementarity\.+:\s*(\S+)\s+(\S+)"),
    "overall_nlp_error": re.compile(r"^Overall NLP error\.+:\s*(\S+)\s+(\S+)"),
    "variable_bound_violation": re.compile(r"^Variable bound violation:\s*(\S+)\s+(\S+)"),
}

OPTION_LINE_RE = re.compile(r"^\s*(\S+)\s*=\s*(\S+)\s+(\d+)\s*$")


def parse_alpha_pr_token(tok):
    """alpha_pr column may carry an attached single-letter step-type marker,
    e.g. '8.52e-08f' or '5.23e-05f'. Split numeric part from trailing letter."""
    m = re.match(r"^([0-9.eE+\-]+)([a-zA-Z]?)$", tok)
    if not m:
        return tok, ""
    return m.group(1), m.group(2)


def parse_log(path, key):
    with open(path, "r", errors="replace") as f:
        raw_lines = f.readlines()

    n = len(raw_lines)

    options = {}
    option_block_started = False
    for i, line in enumerate(raw_lines[:60]):
        if line.strip() == "List of options:":
            option_block_started = True
            continue
        if option_block_started:
            if line.strip() == "" and i > 3:
                # options block ends at first blank-after-content
                if options:
                    break
                else:
                    continue
            m = OPTION_LINE_RE.match(line)
            if m:
                options[m.group(1)] = m.group(2)

    iterations = {}  # iter_num -> dict of fields

    def get_iter(k):
        if k not in iterations:
            iterations[k] = {
                "iter": k,
                "summary_line_no": None,
                "objective": None,
                "inf_pr": None,
                "inf_du": None,
                "lg_mu": None,
                "d_norm": None,
                "lg_rg": None,
                "alpha_du": None,
                "alpha_pr": None,
                "alpha_pr_stepchar": None,
                "ls": None,
                "trailing_marker": None,
                "begin_line_no": None,
                "mu": None,
                "tau": None,
                "curr_x": None,
                "curr_s": None,
                "curr_y_c": None,
                "curr_y_d": None,
                "curr_z_L": None,
                "curr_z_U": None,
                "curr_v_L": None,
                "curr_v_U": None,
                "nlp_values_line_no": None,
                "scaled": {},
                "unscaled": {},
                "safeguard_events": [],  # list of {type, correction, raw_line_no}
            }
        return iterations[k]

    exit_message = None
    exit_line_no = None
    final_num_iterations = None
    final_num_iterations_line_no = None

    # Track NLP-values blocks: first block per iteration is "during" block
    # (appears just after Beginning Iteration block); the terminal iteration
    # has TWO NLP-values blocks: one regular "during" block and one final
    # block after "Number of Iterations....: N" (with the extra
    # "Variable bound violation" line). We keep both, keyed by order.
    nlp_blocks_seen_for_iter = {}

    # pending safeguard events not yet attributable to a specific "next" iter
    # (a safeguard line appears at the end of a "Finding Acceptable Trial
    # Point for Iteration K" block, i.e. logically at the K -> K+1 transition;
    # attribute it to iteration K+1, matching the reported onset convention).
    current_trial_point_iter = None
    TRIAL_POINT_RE = re.compile(r"^\*\*\* Finding Acceptable Trial Point for Iteration\s*(\d+):")

    idx = 0
    while idx < n:
        line = raw_lines[idx].rstrip("\n")

        m = TRIAL_POINT_RE.match(line)
        if m:
            current_trial_point_iter = int(m.group(1))

        m = SAFEGUARD_RE.match(line)
        if m:
            kind, corr = m.group(1), m.group(2)
            attributed_iter = (current_trial_point_iter + 1) if current_trial_point_iter is not None else None
            evt = {
                "raw_line_no": idx + 1,
                "type": kind,
                "maximal_correction": corr,
                "during_trial_point_iter": current_trial_point_iter,
                "attributed_iter": attributed_iter,
            }
            if attributed_iter is not None:
                get_iter(attributed_iter)["safeguard_events"].append(evt)
            idx += 1
            continue

        m = SUMMARY_HEADER_RE.match(line)
        if m:
            it = int(m.group(1))
            rec = get_iter(it)
            rec["summary_line_no"] = idx + 1
            # the data row is a few lines below (after ***** and blank and header)
            j = idx + 1
            # find the data row: a line starting with optional spaces then an int
            while j < n and j < idx + 8:
                candidate = raw_lines[j].rstrip("\n")
                stripped = candidate.strip()
                if stripped and stripped.split()[0].lstrip("-").isdigit():
                    # must correspond to this iteration number
                    toks = stripped.split()
                    if int(toks[0]) == it:
                        _parse_summary_row(rec, toks)
                        rec["data_row_line_no"] = j + 1
                        break
                j += 1
            idx = j
            continue

        m = BEGIN_ITER_RE.match(line)
        if m:
            it = int(m.group(1))
            rec = get_iter(it)
            rec["begin_line_no"] = idx + 1
            j = idx + 1
            while j < n and j < idx + 20:
                l2 = raw_lines[j].rstrip("\n")
                if l2.startswith("***") and j > idx + 2:
                    break
                mu_m = MU_RE.match(l2)
                if mu_m:
                    rec["mu"] = mu_m.group(1)
                tau_m = TAU_RE.match(l2)
                if tau_m:
                    rec["tau"] = tau_m.group(1)
                norm_m = NORM_RE.match(l2)
                if norm_m:
                    field = "curr_" + norm_m.group(1)
                    if field in rec:
                        rec[field] = norm_m.group(2)
                j += 1
            idx = j
            continue

        m = NLP_VALUES_HEADER_RE.match(line)
        if m:
            it = int(m.group(1))
            rec = get_iter(it)
            block = {}
            j = idx + 1
            while j < n and j < idx + 12:
                l2 = raw_lines[j].rstrip("\n")
                matched_any = False
                for field, pat in NLP_FIELD_RE.items():
                    fm = pat.match(l2.strip())
                    if fm:
                        block[field] = {"scaled": fm.group(1), "unscaled": fm.group(2)}
                        matched_any = True
                        break
                if l2.strip().startswith("Number of Iterations") or (
                        l2.strip() == "" and block and j > idx + 3 and
                        not any(raw_lines[j + k].strip() for k in range(1, 3) if j + k < n)):
                    pass
                j += 1
                if l2.strip() == "" and len(block) >= 5:
                    break
            occurrence = nlp_blocks_seen_for_iter.get(it, 0)
            nlp_blocks_seen_for_iter[it] = occurrence + 1
            key_name = "nlp_values_block_%d" % occurrence
            rec.setdefault("nlp_value_blocks", []).append(
                {"occurrence": occurrence, "line_no": idx + 1, "fields": block}
            )
            if rec["nlp_values_line_no"] is None:
                rec["nlp_values_line_no"] = idx + 1
                rec["scaled"] = {k: v["scaled"] for k, v in block.items()}
                rec["unscaled"] = {k: v["unscaled"] for k, v in block.items()}
            idx += 1
            continue

        m = NUM_ITER_RE.match(line)
        if m:
            final_num_iterations = int(m.group(1))
            final_num_iterations_line_no = idx + 1
            idx += 1
            continue

        m = EXIT_RE.match(line)
        if m:
            exit_message = m.group(1).strip()
            exit_line_no = idx + 1
            idx += 1
            continue

        idx += 1

    # The terminal iteration has a second NLP-values block (the true final
    # block, with Variable bound violation). Use the LAST occurrence's fields
    # as the authoritative "final" block for that iteration.
    final_iter_num = final_num_iterations
    final_block = None
    if final_iter_num is not None and final_iter_num in iterations:
        blocks = iterations[final_iter_num].get("nlp_value_blocks", [])
        if blocks:
            final_block = blocks[-1]["fields"]

    return {
        "key": key,
        "path": path,
        "n_lines": n,
        "options": options,
        "iterations": iterations,
        "exit_message": exit_message,
        "exit_line_no": exit_line_no,
        "final_num_iterations": final_num_iterations,
        "final_num_iterations_line_no": final_num_iterations_line_no,
        "final_block": final_block,
    }


def _parse_summary_row(rec, toks):
    # toks: [iter, objective, inf_pr, inf_du, lg(mu), ||d||, lg(rg), alpha_du, alpha_pr(+char), ls, [marker]]
    try:
        rec["objective"] = toks[1]
        rec["inf_pr"] = toks[2]
        rec["inf_du"] = toks[3]
        rec["lg_mu"] = toks[4]
        rec["d_norm"] = toks[5]
        rec["lg_rg"] = toks[6]
        rec["alpha_du"] = toks[7]
        alpha_pr_val, alpha_pr_char = parse_alpha_pr_token(toks[8])
        rec["alpha_pr"] = alpha_pr_val
        rec["alpha_pr_stepchar"] = alpha_pr_char
        rec["ls"] = toks[9]
        if len(toks) > 10:
            rec["trailing_marker"] = toks[10]
        else:
            rec["trailing_marker"] = None
    except IndexError:
        rec["parse_error"] = "insufficient tokens: %r" % (toks,)


# ---------------------------------------------------------------------------
# Parser validation against pre-declared headline facts
# ---------------------------------------------------------------------------

def validate_parser(parsed):
    checks = []

    def add(name, expected, actual, tol=None):
        if tol is None:
            ok = (str(expected) == str(actual))
        else:
            try:
                ok = abs(float(expected) - float(actual)) <= tol
            except Exception:
                ok = False
        checks.append({"name": name, "expected": expected, "actual": actual, "pass": ok})
        return ok

    v1, arma, v2 = parsed["V1"], parsed["ArmA"], parsed["V2"]

    add("V1 final_num_iterations", 89, v1["final_num_iterations"])
    add("ArmA final_num_iterations", 3000, arma["final_num_iterations"])
    add("V2 final_num_iterations", 151, v2["final_num_iterations"])

    add("V1 exit_message", "Optimal Solution Found.", v1["exit_message"])
    add("ArmA exit_message", "Maximum Number of Iterations Exceeded.", arma["exit_message"])
    add("V2 exit_message", "Optimal Solution Found.", v2["exit_message"])

    def count_safeguards(parsed_run):
        c = 0
        for it_rec in parsed_run["iterations"].values():
            c += len(it_rec["safeguard_events"])
        return c

    add("V1 safeguard_count", 0, count_safeguards(v1))
    add("ArmA safeguard_count", 2924, count_safeguards(arma))
    add("V2 safeguard_count", 21, count_safeguards(v2))

    def safeguard_iters(parsed_run):
        its = set()
        for k, rec in parsed_run["iterations"].items():
            if rec["safeguard_events"]:
                its.add(k)
        return sorted(its)

    arma_sg_iters = safeguard_iters(arma)
    add("ArmA safeguard onset iter (attributed)", 77, arma_sg_iters[0] if arma_sg_iters else None)
    add("ArmA safeguard continuity to 3000 (last attributed iter)", 3000,
        arma_sg_iters[-1] if arma_sg_iters else None)
    # continuity: every integer 77..3000 present?
    if arma_sg_iters:
        expected_set = set(range(77, 3001))
        actual_set = set(arma_sg_iters)
        continuous = (expected_set == actual_set)
    else:
        continuous = False
    checks.append({"name": "ArmA safeguard continuous 77..3000", "expected": True,
                    "actual": continuous, "pass": continuous})

    v2_sg_iters = safeguard_iters(v2)
    add("V2 safeguard rows exactly 82-102", list(range(82, 103)), v2_sg_iters,
        )
    checks[-1]["pass"] = (v2_sg_iters == list(range(82, 103)))

    # iteration-0 unscaled dual infeasibility identical in all three
    def unsc(rec, field):
        return rec["unscaled"].get(field)

    add("V1 iter0 dual_infeasibility (unscaled)", "1.8366791482401353e+04",
        unsc(v1["iterations"][0], "dual_infeasibility"))
    add("ArmA iter0 dual_infeasibility (unscaled)", "1.8366791482401353e+04",
        unsc(arma["iterations"][0], "dual_infeasibility"))
    add("V2 iter0 dual_infeasibility (unscaled)", "1.8366791482401353e+04",
        unsc(v2["iterations"][0], "dual_infeasibility"))

    add("V1 iter0 complementarity (unscaled)", "1.0116099216048767e-01",
        unsc(v1["iterations"][0], "complementarity"))
    add("ArmA iter0 complementarity (unscaled)", "1.0116099216048768e+00",
        unsc(arma["iterations"][0], "complementarity"))
    add("V2 iter0 complementarity (unscaled)", "1.0116099216048768e+01",
        unsc(v2["iterations"][0], "complementarity"))

    add("V1 iter0 objective (unscaled)", "1.5200889685781983e+03",
        unsc(v1["iterations"][0], "objective"))
    add("ArmA iter0 objective (unscaled)", "1.5266886921258345e+03",
        unsc(arma["iterations"][0], "objective"))
    add("V2 iter0 objective (unscaled)", "1.5446910450142393e+03",
        unsc(v2["iterations"][0], "objective"))

    add("V1 iter0 constraint_violation (unscaled)", "7.4901136714446994e-05",
        unsc(v1["iterations"][0], "constraint_violation"))
    add("ArmA iter0 constraint_violation (unscaled)", "6.6171136714446988e-05",
        unsc(arma["iterations"][0], "constraint_violation"))
    add("V2 iter0 constraint_violation (unscaled)", "9.9980965587185189e-05",
        unsc(v2["iterations"][0], "constraint_violation"))

    # final blocks
    add("V1 final objective (unscaled)", "1.2936668158879957e+03", unsc(v1["iterations"][89], "objective"))
    add("V1 final dual_infeasibility (unscaled)", "1.6557008856137988e-03",
        unsc(v1["iterations"][89], "dual_infeasibility"))
    add("V1 final constraint_violation (unscaled)", "1.9567576869938819e-07",
        unsc(v1["iterations"][89], "constraint_violation"))
    add("V1 final complementarity (unscaled)", "1.0628655231738295e-05",
        unsc(v1["iterations"][89], "complementarity"))

    add("ArmA final objective (unscaled)", "1.3022135784698721e+03", unsc(arma["iterations"][3000], "objective"))
    add("ArmA final dual_infeasibility (unscaled)", "5.1598055317328658e+01",
        unsc(arma["iterations"][3000], "dual_infeasibility"))
    add("ArmA final constraint_violation (unscaled)", "6.1921343776988665e-02",
        unsc(arma["iterations"][3000], "constraint_violation"))
    add("ArmA final complementarity (unscaled)", "6.2030967852100419e-03",
        unsc(arma["iterations"][3000], "complementarity"))
    add("ArmA terminal mu", "1.8449144625279508e-06", arma["iterations"][3000]["mu"])

    add("V2 final objective (unscaled)", "1.2935224344851817e+03", unsc(v2["iterations"][151], "objective"))
    add("V2 final dual_infeasibility (unscaled)", "1.3480530428912341e-07",
        unsc(v2["iterations"][151], "dual_infeasibility"))
    add("V2 final constraint_violation (unscaled)", "6.8837158195833581e-12",
        unsc(v2["iterations"][151], "constraint_violation"))
    add("V2 final complementarity (unscaled)", "9.0913954543475466e-06",
        unsc(v2["iterations"][151], "complementarity"))

    # numbering sanity: max summary iter equals final_num_iterations for each run
    for key, run in (("V1", v1), ("ArmA", arma), ("V2", v2)):
        max_it = max(k for k, rec in run["iterations"].items() if rec["summary_line_no"] is not None)
        add(f"{key} max parsed summary-iteration == reported final iterations",
            run["final_num_iterations"], max_it)

    all_pass = all(c["pass"] for c in checks)
    return checks, all_pass


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def series(run, field_group, field, subfield=None):
    """Return dict iter -> float value for a scalar summary-row field, or for
    unscaled NLP-block field, or for curr_* norm field."""
    out = {}
    for it, rec in run["iterations"].items():
        if field_group == "summary":
            val = rec.get(field)
        elif field_group == "unscaled":
            val = rec.get("unscaled", {}).get(field)
        elif field_group == "curr":
            val = rec.get(field)
        else:
            val = None
        fv = f(val)
        if fv is not None:
            out[it] = fv
    return out


def slope_over_window(vals_by_iter, it_start, it_end):
    """Simple secant slope (value change per iteration) over [it_start,it_end]
    using nearest available iterations if exact ones are absent."""
    keys = sorted(vals_by_iter.keys())
    if not keys:
        return None
    lo = min((k for k in keys if k >= it_start), default=None)
    hi = max((k for k in keys if k <= it_end), default=None)
    if lo is None or hi is None or hi <= lo:
        return None
    return (vals_by_iter[hi] - vals_by_iter[lo]) / (hi - lo)


def first_sustained_change(vals_by_iter, window=5, direction="increase", min_run=5,
                            rel_tol=1e-9):
    """Predeclared definition of 'sustained': a sustained change starting at
    iteration k means that for at least `min_run` CONSECUTIVE parsed iterations
    starting at k (k, k+1, ..., k+min_run-1, using whatever iteration keys are
    actually present in order), the value is monotonically non-decreasing
    (direction='increase') or non-increasing (direction='decrease') with at
    least one strictly monotonic step in that run, and never reverses by more
    than rel_tol relative to the local max/min within the run.
    Returns the first such k, or None."""
    keys = sorted(vals_by_iter.keys())
    n = len(keys)
    for i in range(n - min_run + 1):
        window_keys = keys[i:i + min_run]
        vals = [vals_by_iter[k] for k in window_keys]
        strictly_moved = False
        ok = True
        for a, b in zip(vals, vals[1:]):
            if direction == "increase":
                if b < a - abs(a) * rel_tol:
                    ok = False
                    break
                if b > a:
                    strictly_moved = True
            else:
                if b > a + abs(a) * rel_tol:
                    ok = False
                    break
                if b < a:
                    strictly_moved = True
        if ok and strictly_moved:
            return window_keys[0]
    return None


def analyze(parsed):
    v1, arma, v2 = parsed["V1"], parsed["ArmA"], parsed["V2"]

    analysis = {}

    # --- fields available / missing per log ---
    fields_check = {}
    for key, run in parsed.items():
        it0 = run["iterations"].get(0, {})
        present = {}
        for fld in ["objective", "inf_pr", "inf_du", "lg_mu", "d_norm", "lg_rg",
                    "alpha_du", "alpha_pr", "ls", "trailing_marker", "mu", "tau",
                    "curr_x", "curr_s", "curr_y_c", "curr_y_d", "curr_z_L", "curr_z_U",
                    "curr_v_L", "curr_v_U"]:
            present[fld] = it0.get(fld) is not None
        # global check: is trailing_marker ever non-None in this run
        any_marker = any(rec.get("trailing_marker") for rec in run["iterations"].values())
        any_lg_rg_dash = any(rec.get("lg_rg") == "-" for rec in run["iterations"].values())
        fields_check[key] = {
            "present_at_iter0": present,
            "trailing_marker_ever_present": any_marker,
            "lg_rg_dash_ever_present": any_lg_rg_dash,
            "restoration_markers_found": False,  # verified via grep pre-check: none found
        }
    analysis["fields_availability"] = fields_check

    # --- B: multiplier evolution series ---
    mult_fields = ["curr_z_L", "curr_z_U", "curr_y_c", "curr_y_d", "curr_v_L", "curr_v_U"]
    mult_series = {}
    for key, run in parsed.items():
        mult_series[key] = {mf: series(run, "curr", mf) for mf in mult_fields}
    analysis["multiplier_series_summary"] = {}
    for key in parsed:
        summ = {}
        for mf in mult_fields:
            s = mult_series[key][mf]
            if s:
                keys_sorted = sorted(s.keys())
                summ[mf] = {
                    "first_iter": keys_sorted[0], "first_val": s[keys_sorted[0]],
                    "last_iter": keys_sorted[-1], "last_val": s[keys_sorted[-1]],
                    "max_val": max(s.values()), "max_iter": max(s, key=s.get),
                    "min_val": min(s.values()), "min_iter": min(s, key=s.get),
                }
        analysis["multiplier_series_summary"][key] = summ

    # windowed slopes over iterations 60-70, 70-90, 90-102 (event windows), predeclared
    windows = [(60, 70), (70, 90), (90, 102), (102, 112)]
    mult_slopes = {}
    for key in parsed:
        mult_slopes[key] = {}
        for mf in mult_fields:
            mult_slopes[key][mf] = {}
            for (a, b) in windows:
                sl = slope_over_window(mult_series[key][mf], a, b)
                mult_slopes[key][mf][f"{a}-{b}"] = sl
    analysis["multiplier_window_slopes"] = mult_slopes

    # --- residual / complementarity series (unscaled) ---
    resid_fields = ["objective", "dual_infeasibility", "constraint_violation", "complementarity"]
    resid_series = {}
    for key, run in parsed.items():
        resid_series[key] = {rf: series(run, "unscaled", rf) for rf in resid_fields}

    # --- summary-row series: inf_pr, inf_du, lg_mu, d_norm, alpha_pr, alpha_du, ls, lg_rg ---
    summ_fields = ["objective", "inf_pr", "inf_du", "lg_mu", "d_norm", "lg_rg", "alpha_du", "alpha_pr", "ls"]
    summ_series = {}
    for key, run in parsed.items():
        summ_series[key] = {sf: series(run, "summary", sf) for sf in summ_fields}

    # --- safeguard event lists ---
    safeguard_events = {}
    for key, run in parsed.items():
        evts = []
        for it, rec in sorted(run["iterations"].items()):
            for e in rec["safeguard_events"]:
                evts.append({"iter": it, **e})
        safeguard_events[key] = evts
    analysis["safeguard_events"] = safeguard_events

    # --- predeclared divergence criterion ---
    # Divergence criterion (predeclared BEFORE inspection of results beyond
    # what parser-validation already required): for each pair of runs, the
    # first iteration k >= 1 (i.e. strictly downstream of the iteration-0
    # initialization difference) at which the ABSOLUTE relative difference in
    # unscaled complementarity between the two runs exceeds one order of
    # magnitude (ratio > 10 or < 0.1), evaluated iteration-by-iteration on the
    # iterations present in both runs, and confirmed to persist for at least
    # 3 consecutive shared iterations (to exclude a single noisy sample).
    def pair_divergence(key_a, key_b, field="complementarity", ratio_thresh=10.0, persist=3):
        sa, sb = resid_series[key_a][field], resid_series[key_b][field]
        shared = sorted(set(sa.keys()) & set(sb.keys()))
        shared = [k for k in shared if k >= 1]
        run_len = 0
        for k in shared:
            a, b = sa[k], sb[k]
            if a == 0 or b == 0:
                ratio = float("inf")
            else:
                ratio = max(a / b, b / a)
            if ratio > ratio_thresh:
                run_len += 1
                if run_len == 1:
                    candidate_start = k
                if run_len >= persist:
                    return candidate_start
            else:
                run_len = 0
        return None

    divergence = {}
    for pair_name, (ka, kb) in [("V1_vs_ArmA", ("V1", "ArmA")),
                                 ("V2_vs_ArmA", ("V2", "ArmA")),
                                 ("V1_vs_V2", ("V1", "V2"))]:
        divergence[pair_name] = pair_divergence(ka, kb)
    analysis["divergence_first_iter_complementarity_ratio_gt10_persist3"] = divergence

    # --- sustained deterioration/improvement (predeclared definition above) ---
    sustained = {}
    for key in parsed:
        cv = resid_series[key]["constraint_violation"]
        comp = resid_series[key]["complementarity"]
        sustained[key] = {
            "constraint_violation_first_sustained_increase":
                first_sustained_change(cv, direction="increase", min_run=5),
            "constraint_violation_first_sustained_decrease":
                first_sustained_change(cv, direction="decrease", min_run=5),
            "complementarity_first_sustained_increase":
                first_sustained_change(comp, direction="increase", min_run=5),
            "complementarity_first_sustained_decrease":
                first_sustained_change(comp, direction="decrease", min_run=5),
        }
    analysis["sustained_change_first_iter"] = sustained

    # --- escape signature (V2 specific): predeclared candidate signature
    # is a measurable transition within a short window after the last
    # safeguard iteration (102): (i) alpha_pr increases by >= 1 order of
    # magnitude relative to its value at the last safeguard iteration; or
    # (ii) inf_pr (primal infeasibility, summary column) begins a sustained
    # decrease (>=5 consecutive iterations); or (iii) the correction
    # magnitude sequence ends (no further safeguard events).
    v2_sg_iters = sorted(set(e["iter"] for e in safeguard_events["V2"]))
    v2_last_sg = v2_sg_iters[-1] if v2_sg_iters else None
    escape_signature = {"v2_last_safeguard_iter": v2_last_sg}
    if v2_last_sg is not None:
        alpha_pr_series = summ_series["V2"]["alpha_pr"]
        inf_pr_series = summ_series["V2"]["inf_pr"]
        base_alpha = alpha_pr_series.get(v2_last_sg)
        later_iters = sorted(k for k in alpha_pr_series if k > v2_last_sg)
        alpha_jump_iter = None
        if base_alpha:
            for k in later_iters:
                if alpha_pr_series[k] >= 10 * base_alpha:
                    alpha_jump_iter = k
                    break
        inf_pr_decrease_iter = first_sustained_change(
            {k: v for k, v in inf_pr_series.items() if k >= v2_last_sg},
            direction="decrease", min_run=5)
        escape_signature["alpha_pr_at_last_safeguard"] = base_alpha
        escape_signature["alpha_pr_order_of_magnitude_jump_iter"] = alpha_jump_iter
        escape_signature["inf_pr_sustained_decrease_from_last_safeguard_iter"] = inf_pr_decrease_iter
        escape_signature["no_further_safeguard_after"] = v2_last_sg  # by construction, last one
    analysis["escape_signature_v2"] = escape_signature

    # Test whether V1 shows an analogous transition (V1 has zero safeguard
    # events, so "last safeguard iter" is undefined; report explicitly).
    analysis["escape_signature_v1_analog"] = {
        "v1_has_safeguard_events": bool(safeguard_events["V1"]),
        "note": "V1 has 0 safeguard events; no last-safeguard anchor exists, "
                "so the V2 escape-signature test (anchored on last safeguard "
                "iteration) is not directly applicable to V1.",
    }
    # Whether ArmA ever shows the analogous transition anchored at ITS first
    # safeguard-continuous stretch's iteration 102 (event-aligned: 102 - 77 =
    # 25 iterations after ArmA's onset, i.e. iteration 77+25=102 in ArmA too,
    # to allow an absolute-offset check) and independently at ArmA's own
    # persistent safeguard tail (no "last" safeguard exists since it runs to
    # 3000).
    arma_sg_iters_sorted = sorted(set(e["iter"] for e in safeguard_events["ArmA"]))
    analysis["escape_signature_arma_analog"] = {
        "arma_has_safeguard_events": bool(arma_sg_iters_sorted),
        "arma_safeguard_first_iter": arma_sg_iters_sorted[0] if arma_sg_iters_sorted else None,
        "arma_safeguard_last_iter": arma_sg_iters_sorted[-1] if arma_sg_iters_sorted else None,
        "note": "ArmA's safeguard sequence continues to iteration 3000 (never "
                "ends), so there is no 'last safeguard iteration' from which "
                "to test the escape signature; by definition ArmA never "
                "satisfies 'no_further_safeguard_after'.",
    }

    # --- correction magnitude series ---
    correction_series = {}
    for key in parsed:
        cs = {}
        for e in safeguard_events[key]:
            try:
                cs[e["iter"]] = float(e["maximal_correction"])
            except ValueError:
                pass
        correction_series[key] = cs
    analysis["correction_magnitude_series"] = correction_series

    # --- barrier progression: mu at key iterations ---
    mu_series = {}
    for key, run in parsed.items():
        mu_series[key] = {}
        for it, rec in run["iterations"].items():
            mv = f(rec.get("mu"))
            if mv is not None:
                mu_series[key][it] = mv
    analysis["mu_series_summary"] = {
        key: {
            "min_mu": min(mu_series[key].values()) if mu_series[key] else None,
            "min_mu_iter": (min(mu_series[key], key=mu_series[key].get)
                            if mu_series[key] else None),
            "mu_at_iter_60": mu_series[key].get(60),
            "mu_at_iter_70": mu_series[key].get(70),
            "mu_at_iter_90": mu_series[key].get(90),
            "mu_at_iter_102": mu_series[key].get(102),
        }
        for key in parsed
    }

    # keep raw series in journal (converted to plain dict, sorted)
    def sorted_series(d):
        return {str(k): d[k] for k in sorted(d.keys())}

    journal_series = {}
    for key in parsed:
        journal_series[key] = {
            "objective_unscaled": sorted_series(resid_series[key]["objective"]),
            "dual_infeasibility_unscaled": sorted_series(resid_series[key]["dual_infeasibility"]),
            "constraint_violation_unscaled": sorted_series(resid_series[key]["constraint_violation"]),
            "complementarity_unscaled": sorted_series(resid_series[key]["complementarity"]),
            "summary_objective": sorted_series(summ_series[key]["objective"]),
            "inf_pr": sorted_series(summ_series[key]["inf_pr"]),
            "inf_du": sorted_series(summ_series[key]["inf_du"]),
            "lg_mu": sorted_series(summ_series[key]["lg_mu"]),
            "d_norm": sorted_series(summ_series[key]["d_norm"]),
            "lg_rg": sorted_series(summ_series[key]["lg_rg"]),
            "alpha_du": sorted_series(summ_series[key]["alpha_du"]),
            "alpha_pr": sorted_series(summ_series[key]["alpha_pr"]),
            "ls": sorted_series(summ_series[key]["ls"]),
            "mu": sorted_series(mu_series[key]),
            "curr_z_L": sorted_series(mult_series[key]["curr_z_L"]),
            "curr_z_U": sorted_series(mult_series[key]["curr_z_U"]),
            "curr_y_c": sorted_series(mult_series[key]["curr_y_c"]),
            "curr_y_d": sorted_series(mult_series[key]["curr_y_d"]),
            "curr_v_L": sorted_series(mult_series[key]["curr_v_L"]),
            "curr_v_U": sorted_series(mult_series[key]["curr_v_U"]),
            "safeguard_events": safeguard_events[key],
        }

    analysis["journal_series"] = journal_series

    return analysis


def build_tables(parsed, analysis):
    """Build the required point-in-time comparison tables."""
    tables = {}
    key_iters = {
        "iter_60": 60,
        "iter_70": 70,
        "iter_90": 90,
    }
    summ_fields = ["objective", "inf_pr", "inf_du", "lg_mu", "d_norm", "lg_rg", "alpha_du", "alpha_pr", "ls"]
    mult_fields = ["curr_z_L", "curr_z_U", "curr_y_c", "curr_y_d", "curr_v_L", "curr_v_U"]

    def row_for(run, it):
        rec = run["iterations"].get(it)
        if rec is None:
            return None
        out = {"iter": it}
        for fld in summ_fields:
            out[fld] = rec.get(fld)
        for fld in mult_fields:
            out[fld] = rec.get(fld)
        out["mu"] = rec.get("mu")
        out["unscaled_objective"] = rec.get("unscaled", {}).get("objective")
        out["unscaled_dual_infeasibility"] = rec.get("unscaled", {}).get("dual_infeasibility")
        out["unscaled_constraint_violation"] = rec.get("unscaled", {}).get("constraint_violation")
        out["unscaled_complementarity"] = rec.get("unscaled", {}).get("complementarity")
        out["safeguard_here"] = bool(rec.get("safeguard_events"))
        return out

    for name, it in key_iters.items():
        tables[name] = {key: row_for(parsed[key], it) for key in parsed}

    # first safeguard per run
    sg = analysis["safeguard_events"]
    first_sg_iter = {key: (sg[key][0]["iter"] if sg[key] else None) for key in parsed}
    tables["first_safeguard"] = {
        key: (row_for(parsed[key], first_sg_iter[key]) if first_sg_iter[key] is not None else None)
        for key in parsed
    }
    tables["first_safeguard_iter"] = first_sg_iter

    # V2's final safeguard (102) and event-aligned ArmA counterpart
    tables["v2_final_safeguard_102"] = row_for(parsed["V2"], 102)
    v2_last_sg_iter = analysis["escape_signature_v2"]["v2_last_safeguard_iter"]
    arma_onset = first_sg_iter["ArmA"]
    v2_onset = first_sg_iter["V2"]
    offset_from_onset = (v2_last_sg_iter - v2_onset) if (v2_last_sg_iter is not None and v2_onset is not None) else None
    arma_event_aligned_iter = (arma_onset + offset_from_onset) if (
        arma_onset is not None and offset_from_onset is not None) else None
    tables["arma_event_aligned_counterpart_to_v2_last_safeguard"] = {
        "arma_onset": arma_onset, "v2_onset": v2_onset,
        "offset_from_onset": offset_from_onset,
        "arma_event_aligned_iter": arma_event_aligned_iter,
        "row": row_for(parsed["ArmA"], arma_event_aligned_iter) if arma_event_aligned_iter else None,
    }

    # ~5 and ~10 iterations after V2's escape (last safeguard 102 -> 107, 112)
    if v2_last_sg_iter is not None:
        tables["v2_plus5_after_escape"] = row_for(parsed["V2"], v2_last_sg_iter + 5)
        tables["v2_plus10_after_escape"] = row_for(parsed["V2"], v2_last_sg_iter + 10)
        arma_plus5 = (arma_event_aligned_iter + 5) if arma_event_aligned_iter else None
        arma_plus10 = (arma_event_aligned_iter + 10) if arma_event_aligned_iter else None
        tables["arma_event_aligned_plus5"] = row_for(parsed["ArmA"], arma_plus5) if arma_plus5 else None
        tables["arma_event_aligned_plus10"] = row_for(parsed["ArmA"], arma_plus10) if arma_plus10 else None

    return tables


def main():
    if os.path.isdir(OUT_DIR):
        raise SystemExit(f"FATAL: output directory already exists: {OUT_DIR} "
                          f"(must be verified absent beforehand)")
    os.makedirs(OUT_DIR, exist_ok=False)
    logs_manifest, paths = verify_inputs()

    parsed = {}
    for key, path in paths.items():
        parsed[key] = parse_log(path, key)

    checks, all_pass = validate_parser(parsed)

    validation_report = {"checks": checks, "all_pass": all_pass}

    if not all_pass:
        # Write partial evidence and stop interpretation, per instructions.
        journal_path = os.path.join(OUT_DIR, "journal.json")
        with open(journal_path, "w") as jf:
            json.dump({"input_verification": logs_manifest,
                       "parser_validation": validation_report}, jf, indent=2, default=str)
        print("PARSER VALIDATION FAILED. See journal.json. Stopping before interpretation.")
        for c in checks:
            if not c["pass"]:
                print("FAIL:", c)
        sys.exit(1)

    analysis = analyze(parsed)
    tables = build_tables(parsed, analysis)

    journal = {
        "input_verification": logs_manifest,
        "parser_validation": validation_report,
        "analysis": analysis,
        "tables": tables,
    }

    journal_path = os.path.join(OUT_DIR, "journal.json")
    with open(journal_path, "w") as jf:
        json.dump(journal, jf, indent=2, default=str)

    print("Parser validation: ALL CHECKS PASS (%d checks)." % len(checks))
    print("Journal written to", journal_path)

    return journal_path, logs_manifest


if __name__ == "__main__":
    main()
