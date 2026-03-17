from __future__ import annotations

import time

import numpy as np

from stopping_criteria import StoppingCriterion
from solvers.aco_base import ACOSolverBase
from solvers.base import VerboseCallback


class AntSystem(ACOSolverBase):
    """Standard Ant System – all ants deposit pheromone proportional to solution quality."""

    DEFAULT_N_ANTS = 19
    DEFAULT_ALPHA = 1.1999
    DEFAULT_BETA = 2.0252
    DEFAULT_RHO = 0.1103
    DEFAULT_Q_CT = 1.0725

    def __init__(
        self,
        n_ants: int = DEFAULT_N_ANTS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        q_ct: float = DEFAULT_Q_CT,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(n_ants, alpha, beta, rho, criteria)
        self.q_ct = q_ct

    def _update_pheromones(self):
        self.tau *= (1.0 - self.rho)
        for sol, cost in zip(self._last_solutions, self._last_costs):
            if cost == 0 or np.isinf(cost):
                continue
            delta = self.q_ct / cost
            for machine_id, jobs in sol.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta

    def solve(self, instance, on_new_best: VerboseCallback | None = None):
        self._init_pheromone(instance)
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
                if on_new_best is not None:
                    on_new_best(gen, best_solution, best_cost, time.monotonic() - start)
            history.append(best_cost)

            self._update_pheromones()

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best_solution, best_cost, history


class RankedAntSystem(ACOSolverBase):
    """Rank-based Ant System – only the top w ants deposit, weighted by rank."""

    DEFAULT_N_ANTS = 23
    DEFAULT_ALPHA = 0.8490
    DEFAULT_BETA = 2.0829
    DEFAULT_RHO = 0.0963
    DEFAULT_Q_CT = 0.9129

    def __init__(
        self,
        n_ants: int = DEFAULT_N_ANTS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        q_ct: float = DEFAULT_Q_CT,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(n_ants, alpha, beta, rho, criteria)
        self.q_ct = q_ct

    def _update_pheromones(self):
        self.tau *= (1.0 - self.rho)
        sorted_indices = np.argsort(self._last_costs)
        w = max(1, self.n_ants // 4)
        for rank, idx in enumerate(sorted_indices[:w]):
            sol = self._last_solutions[idx]
            cost = self._last_costs[idx]
            if cost == 0 or np.isinf(cost):
                continue
            weight = w - rank
            delta = self.q_ct * weight / cost
            for machine_id, jobs in sol.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta

    def solve(self, instance, on_new_best: VerboseCallback | None = None):
        self._init_pheromone(instance)
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
                if on_new_best is not None:
                    on_new_best(gen, best_solution, best_cost, time.monotonic() - start)
            history.append(best_cost)

            self._update_pheromones()

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best_solution, best_cost, history


class EasAntSystem(ACOSolverBase):
    """Elitist Ant System – all ants deposit plus an extra bonus for the best-so-far solution."""

    DEFAULT_N_ANTS = 19
    DEFAULT_ALPHA = 1.2102
    DEFAULT_BETA = 1.5898
    DEFAULT_RHO = 0.0965
    DEFAULT_Q_CT = 0.8518
    DEFAULT_SIGMA = 0.9436

    def __init__(
        self,
        n_ants: int = DEFAULT_N_ANTS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        q_ct: float = DEFAULT_Q_CT,
        sigma: float = DEFAULT_SIGMA,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(n_ants, alpha, beta, rho, criteria)
        self.q_ct = q_ct
        self.sigma = sigma
        self._best_cost = float("inf")
        self._best_solution = None

    def _update_pheromones(self):
        self.tau *= (1.0 - self.rho)

        for sol, cost in zip(self._last_solutions, self._last_costs):
            if cost == 0 or np.isinf(cost):
                continue
            delta = self.q_ct / cost
            for machine_id, jobs in sol.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta

        if self._best_solution is not None:
            delta_best = self.q_ct * self.sigma / self._best_cost
            for machine_id, jobs in self._best_solution.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta_best

    def solve(self, instance, on_new_best: VerboseCallback | None = None):
        self._init_pheromone(instance)
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

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best_solution, best_cost, history
