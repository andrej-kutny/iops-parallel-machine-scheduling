from __future__ import annotations

import time

import numpy as np

from src.models.instance import SchedulingInstance
from src.models.solution import SchedulingSolution
from src.stopping_criteria import StoppingCriterion
from src.local_search.operators import local_search
from src.solvers.aco_base import ACOSolverBase
from src.solvers.base import VerboseCallback


class AntColonySystem(ACOSolverBase):
    """Ant Colony System with exploitation/exploration balance and local pheromone updates."""

    def __init__(
        self,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.9,
        local_decay: float = 0.1,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(n_ants, alpha, beta, rho, criteria)
        self.q0 = q0
        self.local_decay = local_decay
        self._tau_0 = None
        self._best_cost = float("inf")
        self._best_solution = None

    def _construct_ant_solution(self, instance: SchedulingInstance) -> SchedulingSolution:
        """ACS construction: exploit with probability q0, otherwise explore."""
        machine_jobs: dict[int, list[int]] = {k: [] for k in range(instance.m)}
        jobs = list(range(instance.n))
        np.random.shuffle(jobs)

        for j in jobs:
            capable = instance.capable[j]
            if len(capable) == 1:
                chosen_k = capable[0]
            else:
                pheromones = np.array([self.tau[j][k] for k in capable]) ** self.alpha
                heuristics = np.array([self.eta[j][k] for k in capable]) ** self.beta
                attractiveness = pheromones * heuristics

                q = np.random.random()
                if q < self.q0:
                    chosen_k = capable[int(np.argmax(attractiveness))]
                else:
                    total = attractiveness.sum()
                    if total <= 0:
                        chosen_k = capable[np.random.randint(len(capable))]
                    else:
                        probs = attractiveness / total
                        chosen_k = capable[np.random.choice(len(capable), p=probs)]

            # Local pheromone update
            self.tau[j][chosen_k] = (1.0 - self.local_decay) * self.tau[j][chosen_k] + self.local_decay * self._tau_0
            machine_jobs[chosen_k].append(j)

        schedule = {}
        for k in range(instance.m):
            schedule[k] = self._order_jobs_on_machine(machine_jobs[k], k, instance)

        return SchedulingSolution(schedule, instance)

    def _update_pheromones(self):
        """Global update from best-so-far only."""
        self.tau *= (1.0 - self.rho)

        if self._best_solution is not None:
            delta = 1.0 / self._best_cost
            for machine_id, jobs in self._best_solution.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta

    def solve(self, instance: SchedulingInstance, on_new_best: VerboseCallback | None = None) -> tuple[SchedulingSolution, float, list[float]]:
        self._init_pheromone(instance)
        self._tau_0 = self.tau[0][0]
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
