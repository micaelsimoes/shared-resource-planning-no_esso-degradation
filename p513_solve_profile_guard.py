"""Solve-profile guard — enforce a declared solve profile, never assert one.

P5.13-E. The P5.12-K guards block every solve and raise on entry, which enforces a
zero-solve claim. This module generalizes them to a *bounded* claim: solves reached
through a declared call site are counted and allowed through; solves reached any other
way raise immediately.

The rule this exists to serve (CLAUDE.md, sixth evidence rule): a no-solve or
bounded-solve claim must be ENFORCED BY ARMED GUARDS, never asserted. Three of the six
P5.12/P5.13 stages asserted it; this is the mechanism the other three already used.
"""

import sys

from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.solver.shellcmd import SystemCallSolver


class SolverInvocationBlocked(RuntimeError):
    pass


class SolveProfileGuard:
    """Count solves from permitted call sites; raise on every other one.

    `permitted` is a sequence of (file_suffix, function_name) pairs. A solve is
    permitted when some frame on the live call stack matches one of them.
    """

    def __init__(self, permitted, label='stage'):
        self.permitted = tuple(permitted)
        self.label = label
        self.counts = {'permitted_solve': 0, 'permitted_exec': 0,
                       'blocked_solve': 0, 'blocked_exec': 0}
        self.permitted_sites = {}
        self._original_solve = None
        self._original_exec = None

    def _matching_frame(self):
        frame = sys._getframe(2)
        while frame is not None:
            name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            for suffix, function in self.permitted:
                if name == function and filename.endswith(suffix):
                    return f'{suffix}:{function}'
            frame = frame.f_back
        return None

    def install(self):
        self._original_solve = OptSolver.solve
        self._original_exec = SystemCallSolver._execute_command
        guard = self

        def guarded_solve(solver_self, *args, **kwargs):
            site = guard._matching_frame()
            if site is None:
                guard.counts['blocked_solve'] += 1
                raise SolverInvocationBlocked(
                    f'OptSolver.solve reached from an undeclared call site; '
                    f'{guard.label} permits only {guard.permitted}')
            guard.counts['permitted_solve'] += 1
            guard.permitted_sites[site] = guard.permitted_sites.get(site, 0) + 1
            return guard._original_solve(solver_self, *args, **kwargs)

        def guarded_exec(solver_self, *args, **kwargs):
            site = guard._matching_frame()
            if site is None:
                guard.counts['blocked_exec'] += 1
                raise SolverInvocationBlocked(
                    f'SystemCallSolver._execute_command reached from an undeclared '
                    f'call site; {guard.label} permits only {guard.permitted}')
            guard.counts['permitted_exec'] += 1
            return guard._original_exec(solver_self, *args, **kwargs)

        OptSolver.solve = guarded_solve
        SystemCallSolver._execute_command = guarded_exec
        return self

    def uninstall(self):
        if self._original_solve is not None:
            OptSolver.solve = self._original_solve
        if self._original_exec is not None:
            SystemCallSolver._execute_command = self._original_exec

    def verify(self, expected_solves, expected_execs=None):
        """Exact-count check. Too few fails as loudly as too many."""
        expected_execs = expected_solves if expected_execs is None else expected_execs
        failures = []
        if self.counts['permitted_solve'] != expected_solves:
            failures.append(
                f'solve count {self.counts["permitted_solve"]} != declared {expected_solves} '
                '(too few means the path under test did not run)')
        if self.counts['permitted_exec'] != expected_execs:
            failures.append(
                f'process-launch count {self.counts["permitted_exec"]} != declared '
                f'{expected_execs} (a retry would raise this above the solve count)')
        if self.counts['blocked_solve'] or self.counts['blocked_exec']:
            failures.append('a blocked call site was reached')
        return failures
