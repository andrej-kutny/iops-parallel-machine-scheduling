from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion, TimeLimit, GenMinImprovement


DEFAULT_CRITERIA = [TimeLimit(120), GenMinImprovement(window=50, min_pct=0.05)]

# Callback signature: (gen, solution, cost, elapsed) -> None
VerboseCallback = Callable[[int, "SchedulingSolution", float, float], None]


class SolverBase(ABC):
    """Base class for scheduling metaheuristic solvers."""

    def __init__(self, criteria: list[StoppingCriterion] | None = None):
        self.criteria: list[StoppingCriterion] = criteria if criteria else list(DEFAULT_CRITERIA)

    @abstractmethod
    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        """Build a new solution from scratch."""
        ...

    @abstractmethod
    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        """Improve an existing solution."""
        ...

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        """Main loop: construct, improve, track best. Stops when any criterion fires.

        Args:
            on_new_best: Optional callback called whenever a new best solution is found.
                         Receives (generation, new_best_cost, elapsed_seconds).
        """
        best_solution: SchedulingSolution | None = None
        best_cost: float = float("inf")
        history: list[float] = []
        start = time.monotonic()

        for c in self.criteria:
            c.reset()

        gen = 0
        while True:
            solution = self._construct(instance)
            solution = self._improve(solution, instance)

            cost = solution.compute_makespan()
            if cost < best_cost:
                best_cost = cost
                best_solution = solution.copy()
                if on_new_best is not None:
                    on_new_best(gen, best_solution, best_cost, time.monotonic() - start)
            history.append(best_cost)

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best_solution, best_cost, history

    @staticmethod
    def _random_solution(instance: SchedulingInstance, rng: np.random.Generator | None = None) -> SchedulingSolution:
        """Create a random feasible solution: assign each job to a random capable machine."""
        if rng is None:
            rng = np.random.default_rng()

        schedule: dict[int, list[int]] = {k: [] for k in range(instance.m)}
        jobs = list(range(instance.n))
        rng.shuffle(jobs)

        for j in jobs:
            capable = instance.capable[j]
            machine = capable[rng.integers(len(capable))]
            schedule[machine].append(j + 1)

        return SchedulingSolution(schedule, instance)
