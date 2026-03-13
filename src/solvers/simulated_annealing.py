from __future__ import annotations

import time
from typing import Callable

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion
from local_search.operators import random_neighbor
from solvers.base import SolverBase, VerboseCallback


class SimulatedAnnealingSolver(SolverBase):
    """Simulated Annealing with geometric cooling and optional reheating."""

    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        reheat_factor: float = 1.5,
        reheat_patience: int = 200,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(criteria)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.reheat_factor = reheat_factor
        self.reheat_patience = reheat_patience

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        return self._random_solution(instance)

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return solution

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        """SA main loop with geometric cooling and reheating."""
        rng = np.random.default_rng()
        start = time.monotonic()

        current = self._random_solution(instance, rng)
        current_cost = current.compute_makespan()
        best = current.copy()
        best_cost = current_cost
        history: list[float] = []

        if on_new_best is not None:
            on_new_best(0, best, best_cost, 0.0)

        temp = self.initial_temp
        no_improve_count = 0

        for c in self.criteria:
            c.reset()

        gen = 0
        while True:
            neighbor = random_neighbor(current, instance, rng)
            neighbor_cost = neighbor.compute_makespan()

            delta = neighbor_cost - current_cost
            if delta < 0 or (temp > 1e-10 and rng.random() < np.exp(-delta / temp)):
                current = neighbor
                current_cost = neighbor_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best = current.copy()
                no_improve_count = 0
                if on_new_best is not None:
                    on_new_best(gen, best, best_cost, time.monotonic() - start)
            else:
                no_improve_count += 1

            history.append(best_cost)

            # Geometric cooling
            temp *= self.cooling_rate

            # Reheat if stagnating
            if no_improve_count >= self.reheat_patience:
                temp = min(temp * self.reheat_factor, self.initial_temp)
                no_improve_count = 0

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best, best_cost, history
