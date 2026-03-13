from __future__ import annotations

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion
from local_search.operators import local_search
from solvers.base import SolverBase


class GraspSolver(SolverBase):
    """GRASP metaheuristic for parallel machine scheduling."""

    def __init__(self, alpha: float = 0.5, criteria: list[StoppingCriterion] | None = None):
        super().__init__(criteria)
        self.alpha = alpha

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        """Greedy randomized adaptive construction.

        For each unassigned job, compute earliest completion time on each capable machine
        (considering release dates, setup times, and processing times).
        Build RCL and pick randomly.
        """
        schedule: dict[int, list[int]] = {k: [] for k in range(instance.m)}
        machine_time: dict[int, int] = {k: 0 for k in range(instance.m)}
        machine_last_job: dict[int, int | None] = {k: None for k in range(instance.m)}

        unassigned = list(range(instance.n))
        np.random.shuffle(unassigned)

        while unassigned:
            candidates = []
            for j in unassigned:
                for k in instance.capable[j]:
                    release = instance.release[j][k]
                    if machine_last_job[k] is not None:
                        setup = instance.setup[machine_last_job[k]][j][k]
                        start = max(release, machine_time[k] + setup)
                    else:
                        start = max(release, machine_time[k])
                    completion = start + instance.duration[j][k]
                    candidates.append((completion, j, k))

            if not candidates:
                break

            costs = np.array([c[0] for c in candidates], dtype=float)
            c_min, c_max = costs.min(), costs.max()
            threshold = c_min + self.alpha * (c_max - c_min)
            rcl = [c for c, cost in zip(candidates, costs) if cost <= threshold + 1e-10]

            chosen = rcl[np.random.randint(len(rcl))]
            _, job, machine = chosen

            schedule[machine].append(job + 1)
            release = instance.release[job][machine]
            if machine_last_job[machine] is not None:
                setup = instance.setup[machine_last_job[machine]][job][machine]
                start = max(release, machine_time[machine] + setup)
            else:
                start = max(release, machine_time[machine])
            machine_time[machine] = start + instance.duration[job][machine]
            machine_last_job[machine] = job
            unassigned.remove(job)

        return SchedulingSolution(schedule, instance)

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return local_search(solution, instance)
