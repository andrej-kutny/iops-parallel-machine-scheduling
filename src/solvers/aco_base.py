from __future__ import annotations

import time

import numpy as np

from src.models.instance import SchedulingInstance
from src.models.solution import SchedulingSolution
from src.stopping_criteria import StoppingCriterion
from src.local_search.operators import local_search
from src.solvers.base import SolverBase, VerboseCallback


class ACOSolverBase(SolverBase):
    """Base ACO solver for parallel machine scheduling.

    Pheromone matrix tau[job][machine] of shape (n, m) represents
    desirability of assigning job j to machine k.
    """

    def __init__(
        self,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(criteria)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau: np.ndarray | None = None
        self.eta: np.ndarray | None = None
        self._last_solutions: list[SchedulingSolution] = []
        self._last_costs: list[float] = []

    def _init_pheromone(self, instance: SchedulingInstance):
        """Initialize pheromone and heuristic matrices."""
        greedy_cost = self._greedy_cost(instance)
        initial_tau = self.n_ants / max(greedy_cost, 1)
        self.tau = np.full((instance.n, instance.m), initial_tau, dtype=float)

        # Heuristic: 1 / (duration + avg_setup)
        self.eta = np.zeros((instance.n, instance.m), dtype=float)
        for j in range(instance.n):
            for k in instance.capable[j]:
                avg_setup = instance.setup[:, j, k].mean()
                val = instance.duration[j][k] + avg_setup
                self.eta[j][k] = 1.0 / max(val, 1e-10)

    def _greedy_cost(self, instance: SchedulingInstance) -> float:
        """Greedy nearest-neighbor heuristic cost for initialization."""
        schedule: dict[int, list[int]] = {k: [] for k in range(instance.m)}
        machine_time = np.zeros(instance.m)
        machine_last = [None] * instance.m

        unassigned = list(range(instance.n))
        np.random.shuffle(unassigned)

        for j in unassigned:
            best_completion = float("inf")
            best_k = instance.capable[j][0]
            for k in instance.capable[j]:
                release = instance.release[j][k]
                if machine_last[k] is not None:
                    setup = instance.setup[machine_last[k]][j][k]
                    start = max(release, machine_time[k] + setup)
                else:
                    start = max(release, machine_time[k])
                completion = start + instance.duration[j][k]
                if completion < best_completion:
                    best_completion = completion
                    best_k = k

            schedule[best_k].append(j + 1)
            release = instance.release[j][best_k]
            if machine_last[best_k] is not None:
                setup = instance.setup[machine_last[best_k]][j][best_k]
                start = max(release, machine_time[best_k] + setup)
            else:
                start = max(release, machine_time[best_k])
            machine_time[best_k] = start + instance.duration[j][best_k]
            machine_last[best_k] = j

        sol = SchedulingSolution(schedule, instance)
        return sol.compute_makespan()

    def _order_jobs_on_machine(self, jobs_0idx: list[int], machine_id: int, instance: SchedulingInstance) -> list[int]:
        """Order jobs on a machine greedily by earliest feasible start time considering setup."""
        if len(jobs_0idx) <= 1:
            return [j + 1 for j in jobs_0idx]

        ordered = []
        remaining = list(jobs_0idx)
        time = 0
        prev_job = None

        while remaining:
            best_completion = float("inf")
            best_j = remaining[0]
            for j in remaining:
                release = instance.release[j][machine_id]
                if prev_job is not None:
                    setup = instance.setup[prev_job][j][machine_id]
                    start = max(release, time + setup)
                else:
                    start = max(release, time)
                completion = start + instance.duration[j][machine_id]
                if completion < best_completion:
                    best_completion = completion
                    best_j = j

            ordered.append(best_j + 1)
            release = instance.release[best_j][machine_id]
            if prev_job is not None:
                setup = instance.setup[prev_job][best_j][machine_id]
                start = max(release, time + setup)
            else:
                start = max(release, time)
            time = start + instance.duration[best_j][machine_id]
            prev_job = best_j
            remaining.remove(best_j)

        return ordered

    def _construct_ant_solution(self, instance: SchedulingInstance) -> SchedulingSolution:
        """Single ant: assign jobs to machines probabilistically, then order."""
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
            attractiveness = pheromones * heuristics
            total = attractiveness.sum()
            if total <= 0:
                chosen_k = capable[np.random.randint(len(capable))]
            else:
                probs = attractiveness / total
                chosen_k = capable[np.random.choice(len(capable), p=probs)]

            machine_jobs[chosen_k].append(j)

        # Order jobs on each machine
        schedule = {}
        for k in range(instance.m):
            schedule[k] = self._order_jobs_on_machine(machine_jobs[k], k, instance)

        return SchedulingSolution(schedule, instance)

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        """Run all ants, return best."""
        solutions = []
        costs = []
        for _ in range(self.n_ants):
            sol = self._construct_ant_solution(instance)
            cost = sol.compute_makespan()
            solutions.append(sol)
            costs.append(cost)

        self._last_solutions = solutions
        self._last_costs = costs
        best_idx = int(np.argmin(costs))
        return solutions[best_idx]

    def _update_pheromones(self):
        """Default: evaporate + deposit from all ants."""
        self.tau *= (1.0 - self.rho)

        for sol, cost in zip(self._last_solutions, self._last_costs):
            if cost == 0 or np.isinf(cost):
                continue
            delta = 1.0 / cost
            for machine_id, jobs in sol.schedule.items():
                for job in jobs:
                    self.tau[job - 1][machine_id] += delta

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return local_search(solution, instance)

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        """ACO main loop."""
        self._init_pheromone(instance)
        start = time.monotonic()

        best_solution: SchedulingSolution | None = None
        best_cost = float("inf")
        history: list[float] = []

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
