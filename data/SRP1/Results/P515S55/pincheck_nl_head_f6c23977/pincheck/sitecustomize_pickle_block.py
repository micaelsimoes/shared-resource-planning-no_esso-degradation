# Pin check f6c23977 (zero solves): process-wide pickle block, installed via PYTHONPATH so that it also applies to
# the digest harness's determinism child (same environment). Any unpickle raises; counts reported at exit on stderr.
import atexit
import os
import pickle
import sys

_COUNTS = {'load': 0, 'loads': 0, 'Unpickler': 0, '_Unpickler': 0}


def _refuse(kind):
    _COUNTS[kind] += 1
    sys.stderr.write(f'[PINCHECK PICKLE GUARD] {kind} refused (pid {os.getpid()})\n')
    raise RuntimeError(f'pincheck: pickle {kind} refused -- unpickling is not permitted in this run')


def _load(*a, **k):
    _refuse('load')


def _loads(*a, **k):
    _refuse('loads')


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *a, **k):
        _refuse('Unpickler')


class _BlockedPyUnpickler(pickle._Unpickler):
    def __init__(self, *a, **k):
        _refuse('_Unpickler')


pickle.load, pickle.loads, pickle._load, pickle._loads = _load, _loads, _load, _loads
pickle.Unpickler, pickle._Unpickler = _BlockedUnpickler, _BlockedPyUnpickler


@atexit.register
def _report():
    sys.stderr.write(f'[PINCHECK PICKLE GUARD] pid {os.getpid()} argv0 {os.path.basename(sys.argv[0]) if sys.argv else ""} '
                     f'refused counts {_COUNTS} total {sum(_COUNTS.values())}\n')
