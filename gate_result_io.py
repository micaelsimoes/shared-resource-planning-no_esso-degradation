"""The ONE writer for gate-result JSON / JSONL (P5.15 Addendum 52, W100).

WHY. The stringified-boolean defect occurred twice: a numpy boolean -- which is NOT a subclass of `bool` (unlike
numpy.float64 of `float`) -- reached `json.dumps(..., default=str)` and was written as the STRING "True"/"False".
W74 and W76 patched two writers (`p515_s44_campaign_harness`, `p515_g_g1_g4_admm_gates`) with a `_json_default` hook;
W98's new hooks module then wrote `reconciles_to_net` with `default=str` again and gate G14 (`is True`) failed on every
line. Addendum 52 ruling: a defect fixed twice is fixed at the writer, not at the caller. Every gate-result writer
goes through `dump` / `dumps` here, and a string-typed flag is REFUSED at write time.

API
  dump(obj, fp, *, default=json_default, **json_kwargs)   -> None   (json.dump, after the refusal check)
  dumps(obj, *, default=json_default, **json_kwargs)      -> str    (json.dumps, after the refusal check)
  check(obj, default=json_default)                         -> None   (raises StringFlagError; writes nothing)
  find_string_flags(obj, default=None, limit=None)         -> [(json_path, field, value), ...]
  is_string_flag(value)                                    -> bool
  json_default(obj)       the W74/W76 hook: numpy bool -> JSON bool; anything else -> str(obj)
  json_default_item(obj)  the W93 hook: set/frozenset/tuple -> list; numpy scalar -> .item(); anything else -> str(obj)
  StringFlagError(ValueError)  .flags = the offending [(json_path, field, value), ...]

BYTE IDENTITY. `dump` / `dumps` call `json.dump` / `json.dumps` with exactly the arguments given (the caller's own
`default=` hook, indent, separators, sort_keys, ...) -- nothing is converted, reordered or re-encoded here. So for every
payload the check accepts, the bytes are identical to the pre-W100 call with the same hook. The two hooks are carried
here verbatim from their callers, so each migrated writer keeps its own non-boolean behaviour (NOTE: `json_default`
writes a numpy INTEGER as a string, e.g. "5", exactly as the W74/W76 writers always have; only `json_default_item`
writes numpy numbers as numbers). The check runs over the whole payload BEFORE the first byte is written, so a refused
payload leaves no partial file and no partial JSONL line.

REFUSAL RULE (the flag definition shared with `p515_gate_result_bool_typing_test.py`). A STRING FLAG is any JSON
value -- an object member's value or an array element, at any depth; object KEYS are not values and are never refused
-- that is a string whose text, after `str.strip()` and `str.lower()`, is "true" or "false". So "True", "False",
"true", "FALSE", " True", "False\\n" are all refused. Rationale: a boolean has no legitimate reason to be written as
text in a gate result under ANY spelling; the defect's own spelling is "True"/"False" (str of a Python or numpy bool),
but "true"/"false" is what `str(x).lower()` or a JSON text embedded as a value produces, and a spelling-exact rule
would let the next variant through. Detection is by VALUE, not by field name, because the defect has hit a new field
name every time (`objective_convergence`, `determinate_at_gt_error_bar`, `pass`, `reconciles_to_net`, ...).
What is checked is what the file will contain: string values in dicts / lists / tuples of the payload, and every value
the `default=` hook returns for a non-JSON-native object (hooks are called once more here, before encoding; the hooks
above are pure). A genuine string that happens to spell a boolean cannot be written through this module: write a
bool, or a different string.

Stdlib only; no numpy import (numpy types are detected by type identity).
"""

import json

__all__ = ['StringFlagError', 'is_string_flag', 'json_default', 'json_default_item', 'find_string_flags', 'check',
           'dump', 'dumps', 'FLAG_SPELLINGS']

FLAG_SPELLINGS = ('true', 'false')
_MAX_DEPTH = 10000
_NATIVE_SCALARS = (int, float, type(None))   # bool is a subclass of int


class StringFlagError(ValueError):
    """A payload carries a string-typed flag (see the module's REFUSAL RULE). Nothing was written."""

    def __init__(self, flags):
        self.flags = list(flags)
        shown = '; '.join(f'{p} = {v!r}' for p, _f, v in self.flags[:10])
        more = f' (+{len(self.flags) - 10} more)' if len(self.flags) > 10 else ''
        super().__init__(f'refusing to write a string-typed flag ({len(self.flags)} value(s)): {shown}{more}. '
                         f'Write a JSON boolean (bool(x)), not its text.')


def is_string_flag(value):
    """True when `value` is a str whose stripped, lower-cased text is "true" or "false" (the REFUSAL RULE)."""
    return isinstance(value, str) and value.strip().lower() in FLAG_SPELLINGS


def json_default(obj):
    """The `default=` hook of the W74 harness / W76 gates writers, verbatim: a numpy boolean is written as a JSON
    boolean; everything else exactly as `default=str` always wrote it (so a numpy integer is written as its string)."""
    if type(obj).__module__ == 'numpy' and type(obj).__name__ in ('bool', 'bool_'):
        return bool(obj)
    return str(obj)


def json_default_item(obj):
    """The `default=` hook of the W93 benchmark harness, verbatim: sets / frozensets / tuples as lists, numpy scalars
    as their Python value (`.item()`: numpy bool -> bool, numpy integer -> int, numpy float -> float), anything else
    as `str`."""
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, 'item'):          # numpy scalars
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _child_path(path, key):
    if isinstance(key, int):
        return f'{path}[{key}]'
    return f'{path}.{key}'


def find_string_flags(obj, default=None, limit=None):
    """Every string flag in `obj` as (json_path, field, value); `field` is the innermost object key above the value
    (array elements inherit their parent's key; None at the top level). With `default`, a non-JSON-native object is
    replaced by `default(obj)` and that result is inspected (what `json.dump(..., default=default)` would encode).
    Without `default` (a parsed JSON document), non-native objects are ignored."""
    found = []
    stack = [(obj, '$', None, 0)]
    while stack:
        node, path, field, depth = stack.pop()
        if depth > _MAX_DEPTH:
            raise ValueError(f'gate_result_io: nesting deeper than {_MAX_DEPTH} at {path} (circular payload?)')
        if isinstance(node, str):
            if is_string_flag(node):
                found.append((path, field, node))
                if limit is not None and len(found) >= limit:
                    break
        elif isinstance(node, dict):
            for key, value in reversed(list(node.items())):
                stack.append((value, _child_path(path, str(key)), str(key), depth + 1))
        elif isinstance(node, (list, tuple)):
            for index in range(len(node) - 1, -1, -1):
                stack.append((node[index], _child_path(path, index), field, depth + 1))
        elif isinstance(node, _NATIVE_SCALARS):
            continue
        elif default is not None:
            stack.append((default(node), path, field, depth + 1))
    return found


def check(obj, default=json_default):
    """Raise StringFlagError if `obj` (as encoded with `default`) carries a string flag. Writes nothing."""
    flags = find_string_flags(obj, default=default)
    if flags:
        raise StringFlagError(flags)


def dumps(obj, *, default=json_default, **json_kwargs):
    """`json.dumps(obj, default=default, **json_kwargs)` after `check`: byte-identical text, or StringFlagError."""
    check(obj, default=default)
    return json.dumps(obj, default=default, **json_kwargs)


def dump(obj, fp, *, default=json_default, **json_kwargs):
    """`json.dump(obj, fp, default=default, **json_kwargs)` after `check` (so nothing reaches `fp` when refused)."""
    check(obj, default=default)
    json.dump(obj, fp, default=default, **json_kwargs)
