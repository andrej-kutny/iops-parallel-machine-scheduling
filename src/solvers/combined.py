from __future__ import annotations

import copy
import time
from typing import Callable

from src.models.instance import SchedulingInstance
from src.models.solution import SchedulingSolution
from src.stopping_criteria import StoppingCriterion
from src.solvers.base import SolverBase, VerboseCallback


class CombinedSolver(SolverBase):
    """Runs multiple solvers in sequence, restarting the cycle when a new best is found.

    Flow:
    1. Start with remaining = [solver_0, solver_1, ..., solver_n]
    2. Pop the first solver, run it with criteria until a criterion fires
    3. If a new global best was found during this solver's run,
       reset remaining to all solvers (so each gets another chance)
    4. If remaining is empty (all solvers tried without improvement), stop
    5. Return the overall best solution
    """

    def __init__(
        self,
        solvers: list[SolverBase],
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(criteria)
        self.solvers = solvers

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        return self._random_solution(instance)

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return solution

    def _fresh_criteria(self) -> list[StoppingCriterion]:
        """Return deep-copied and reset criteria for each sub-solver run."""
        criteria = copy.deepcopy(self.criteria)
        for c in criteria:
            c.reset()
        return criteria

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
        on_solver_switch: Callable[[str, str], None] | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        best_solution: SchedulingSolution | None = None
        best_cost = float("inf")
        overall_history: list[float] = []
        start = time.monotonic()

        remaining = list(self.solvers)
        prev_solver_name: str | None = None

        while remaining:
            solver = remaining.pop(0)
            solver_name = type(solver).__name__

            if prev_solver_name is not None and on_solver_switch is not None:
                on_solver_switch(prev_solver_name, solver_name)

            # Give this sub-solver fresh copies of the criteria
            solver.criteria = self._fresh_criteria()

            # Track whether this sub-solver finds a new global best
            improved = [False]

            def sub_callback(sub_gen, solution, cost, _elapsed, _sn=solver_name):
                nonlocal best_cost, best_solution
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution.copy()
                    improved[0] = True
                    if on_new_best is not None:
                        on_new_best(sub_gen, solution, cost, time.monotonic() - start)

            solution, cost, sub_history = solver.solve(instance, on_new_best=sub_callback)

            # Update best in case callback didn't fire
            if cost < best_cost:
                best_cost = cost
                best_solution = solution.copy()
                improved[0] = True
            overall_history.extend(sub_history)

            # Mirror triggered flags back to self.criteria
            for orig, sub_copy in zip(self.criteria, solver.criteria):
                if sub_copy.triggered:
                    orig.triggered = True

            # If this solver improved, reset remaining to all solvers
            # starting from the next one (current solver just stagnated)
            if improved[0]:
                idx = self.solvers.index(solver)
                n = len(self.solvers)
                remaining = [self.solvers[(idx + 1 + i) % n] for i in range(n)]

            prev_solver_name = solver_name

        return best_solution, best_cost, overall_history
