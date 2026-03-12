from __future__ import annotations

import time

import numpy as np

from src.models.instance import SchedulingInstance
from src.models.solution import SchedulingSolution
from src.stopping_criteria import StoppingCriterion
from src.solvers.aco_base import ACOSolverBase
from src.solvers.base import VerboseCallback


class AntMultiTourSystem(ACOSolverBase):
    """Ant Multi-Tour System. Tracks assignment usage and penalizes frequently used pairs."""

    def __init__(
        self,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q_tours: int = 5,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(n_ants, alpha, beta, rho, criteria)
        self.q_tours = q_tours
        self.usage: np.ndarray | None = None

    def _construct_ant_solution(self, instance: SchedulingInstance) -> SchedulingSolution:
        """Construct with usage penalty: attractiveness / (1 + sqrt(usage))."""
        machine_jobs: dict[int, list[int]] = {k: [] for k in range(instance.m)}
        jobs = list(range(instance.n))
        np.random.shuffle(jobs)

        for j in jobs:
            capable = instance.capable[j]
            if len(capable) == 1:
                machine_jobs[capable[0]].append(j)
                continue

            pheromones = np.array([self.tau[j][k] for k in capable]) ** self.alpha
            heuristics = np.array([self.eta[j][k] for k in capable]) ** self.beta
            penalty = 1.0 + np.sqrt(np.maximum(np.array([self.usage[j][k] for k in capable]), 0))
            attractiveness = (pheromones * heuristics) / penalty
            total = attractiveness.sum()

            if total <= 0:
                chosen_k = capable[np.random.randint(len(capable))]
            else:
                probs = attractiveness / total
                chosen_k = capable[np.random.choice(len(capable), p=probs)]

            machine_jobs[chosen_k].append(j)

        schedule = {}
        for k in range(instance.m):
            schedule[k] = self._order_jobs_on_machine(machine_jobs[k], k, instance)

        return SchedulingSolution(schedule, instance)

    def _update_usage(self, solutions: list[SchedulingSolution]):
        """Track assignment usage frequency."""
        for sol in solutions:
            for machine_id, jobs in sol.schedule.items():
                for job in jobs:
                    self.usage[job - 1][machine_id] += 1

    def solve(self, instance: SchedulingInstance, on_new_best: VerboseCallback | None = None) -> tuple[SchedulingSolution, float, list[float]]:
        self._init_pheromone(instance)
        self.usage = np.zeros((instance.n, instance.m), dtype=float)

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

            self._update_usage(self._last_solutions)

            # Periodically decay usage
            if gen > 0 and gen % self.q_tours == 0:
                self.usage *= 0.9

            self._update_pheromones()

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best_solution, best_cost, history
