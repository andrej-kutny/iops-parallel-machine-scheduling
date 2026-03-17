"""
Iterated Local Search (ILS) for parallel machine scheduling.

Inspired by IOPS Assignment 3 (Job Shop): same pattern — local search to a local optimum,
then perturb the solution and repeat. Uses the project's local_search and perturb.
"""

from __future__ import annotations

import time

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion
from local_search.operators import local_search, perturb
from solvers.base import SolverBase, VerboseCallback


class ILSSolver(SolverBase):
    """Iterated Local Search: repeated (perturb → local search), keep best."""

    DEFAULT_PERTURBATION_STRENGTH = 4

    def __init__(
        self,
        perturbation_strength: int = DEFAULT_PERTURBATION_STRENGTH,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(criteria)
        self.perturbation_strength = max(1, perturbation_strength)

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        return self._random_solution(instance)

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return local_search(solution, instance)

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        """ILS main loop: improve(initial), then repeatedly perturb and improve."""
        rng = np.random.default_rng()
        start = time.monotonic()

        current = self._construct(instance)
        current = self._improve(current, instance)
        best = current.copy()
        best_cost = best.compute_makespan()
        history: list[float] = [best_cost]

        if on_new_best is not None:
            on_new_best(0, best, best_cost, 0.0)

        for c in self.criteria:
            c.reset()

        gen = 0
        while True:
            current = perturb(current, instance, self.perturbation_strength, rng)
            current = self._improve(current, instance)
            cost = current.compute_makespan()

            if cost < best_cost:
                best = current.copy()
                best_cost = cost
                if on_new_best is not None:
                    on_new_best(gen, best, best_cost, time.monotonic() - start)

            history.append(best_cost)

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best, best_cost, history
