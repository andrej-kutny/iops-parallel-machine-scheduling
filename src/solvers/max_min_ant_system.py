from __future__ import annotations

import time

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion
from solvers.aco_base import ACOSolverBase
from solvers.base import VerboseCallback


class MaxMinAntSystem(ACOSolverBase):
    """MAX-MIN Ant System. Only best-so-far deposits. Periodic reinitialization."""

    DEFAULT_N_ANTS = 25
    DEFAULT_ALPHA = 0.9331
    DEFAULT_BETA = 2.2477
    DEFAULT_RHO = 0.1293
    DEFAULT_REINIT_FREQUENCY = 124

    def __init__(
        self,
        n_ants: int = DEFAULT_N_ANTS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        reinit_frequency: int = DEFAULT_REINIT_FREQUENCY,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(n_ants, alpha, beta, rho, criteria)
        self.reinit_frequency = reinit_frequency
        self._best_cost = float("inf")
        self._best_solution = None
        self._tau_max = None

    def _update_pheromones(self):
        self.tau *= (1.0 - self.rho)

        if self._best_solution is not None:
            delta = 1.0 / self._best_cost
            for machine_id, jobs in self._best_solution.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta

    def solve(self, instance: SchedulingInstance, on_new_best: VerboseCallback | None = None) -> tuple[SchedulingSolution, float, list[float]]:
        self._init_pheromone(instance)
        self._tau_max = self.tau[0][0]
        self._best_solution = None
        self._best_cost = float("inf")

        best_solution = None
        best_cost = float("inf")
        history = []
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
                self._best_cost = cost
                self._best_solution = solution.copy()
                if on_new_best is not None:
                    on_new_best(gen, best_solution, best_cost, time.monotonic() - start)
            history.append(best_cost)

            self._update_pheromones()

            # Periodic reinitialization
            if gen > 0 and gen % self.reinit_frequency == 0:
                self.tau = np.full_like(self.tau, self._tau_max)

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best_solution, best_cost, history
