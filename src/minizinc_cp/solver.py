"""
MiniZinc-based CP solver for parallel machine scheduling.

This module is a very thin adapter between:
- the project representation (`SchedulingInstance` / `SchedulingSolution`), and
- the generic MiniZinc Python API + the `scheduling.mzn` model.

High-level flow:
1. `_instance_to_minizinc_data` converts the JSON-based instance into
   plain Python lists/arrays that match the MiniZinc model parameters.
2. `solve_minizinc` loads `scheduling.mzn`, creates a MiniZinc `Instance`,
   assigns the converted data, calls `Instance.solve`, and translates the
   resulting arrays back into a `SchedulingSolution`.
3. `MinizincSolver` wraps this in the same interface as the metaheuristic
   solvers (same `solve(instance, on_new_best)` signature) so it can be
   used from `main.py --solver minizinc`.

Requirements:
- Python package: `minizinc` (see pyproject optional dependency).
- MiniZinc 2.6+ installed on the system with at least one backend solver
  (e.g. Gecode) configured.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import minizinc
import numpy as np

from models.solution import SchedulingSolution
from solvers.base import SolverBase
from stopping_criteria import StoppingCriterion, TimeLimit

if TYPE_CHECKING:
    from models.instance import SchedulingInstance
    from models.solution import SchedulingSolution


def _instance_to_minizinc_data(instance: "SchedulingInstance") -> dict:
    """
    Convert a `SchedulingInstance` into a dictionary of MiniZinc parameters.

    We keep Python arrays in natural 0-based `n x m` / `n x n x m` shape;
    the MiniZinc Python API automatically maps them to the 1-based array
    declarations in `scheduling.mzn` (1..n, 1..m).
    """
    n, m = instance.n, instance.m
    horizon = int(instance.horizon)

    # capable[j][k] = 1 iff job j can run on machine k  -> 1..n x 1..m in MiniZinc
    capable = [[1 if k in instance.capable[j] else 0 for k in range(m)] for j in range(n)]

    # duration[j][k], release[j][k]
    duration = [[int(instance.duration[j, k]) for k in range(m)] for j in range(n)]
    release = [[int(instance.release[j, k]) for k in range(m)] for j in range(n)]

    # setup[i][j][k]  -> 1..n x 1..n x 1..m in MiniZinc
    setup = [[[int(instance.setup[i, j, k]) for k in range(m)] for j in range(n)] for i in range(n)]

    return {
        "n": n,
        "m": m,
        "horizon": horizon,
        "capable": capable,
        "duration": duration,
        "release": release,
        "setup": setup,
    }


def _result_to_schedule(instance: "SchedulingInstance", assign: list, start: list) -> dict[int, list[int]]:
    """
    Build a `schedule` dict in the project format from MiniZinc arrays.

    MiniZinc returns:
    - `assign[j]`: 1-based machine index for job j (1..n).
    - `start[j]`: start time for job j.

    We group jobs by machine, sort by start time, and produce:
      { machine_id_0_based: [job_id_1_based, ...], ... }
    """
    n, m = instance.n, instance.m
    # assign/start from MiniZinc are 1-based index (assign[0] = machine for job 1)
    jobs_by_machine: dict[int, list[tuple[int, float]]] = {k: [] for k in range(m)}
    for j in range(1, n + 1):
        machine_1based = int(assign[j - 1])
        machine_0based = machine_1based - 1
        st = float(start[j - 1])
        jobs_by_machine[machine_0based].append((j, st))  # j is 1-based job ID
    schedule = {}
    for k in range(m):
        jobs_by_machine[k].sort(key=lambda x: x[1])
        schedule[k] = [job_id for job_id, _ in jobs_by_machine[k]]
    return schedule


def solve_minizinc(
    instance: "SchedulingInstance",
    solver_name: str = "gecode",
    time_limit_seconds: float | None = None,
) -> tuple["SchedulingSolution", float]:
    """
    Solve a scheduling instance with MiniZinc.

    Returns:
        (SchedulingSolution, makespan).

    Notes:
    - Any ImportError for the `minizinc` package will propagate up.
    - If MiniZinc fails to find a solution (e.g. timeout), we fall back
      to a random feasible solution so the rest of the pipeline can keep
      running without special-casing this solver.
    """
    Instance = minizinc.Instance
    Model = minizinc.Model
    Solver = minizinc.Solver

    model_dir = os.path.join(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "scheduling.mzn")
    model = Model(model_path)
    solver = Solver.lookup(solver_name)
    mz_instance = Instance(solver, model)

    data = _instance_to_minizinc_data(instance)

    for key, value in data.items():
        mz_instance[key] = value

    kwargs = {}
    if time_limit_seconds is not None and time_limit_seconds > 0:
        kwargs["time_limit"] = timedelta(seconds=time_limit_seconds)

    result = mz_instance.solve(**kwargs)

    if result.status.name not in ("SATISFIED", "OPTIMAL_SOLUTION"):
        # No solution found: return a feasible random solution as fallback
        # so the pipeline doesn't crash.
        rng = np.random.default_rng()
        fallback = SolverBase._random_solution(instance, rng)
        return fallback, float(fallback.compute_makespan())

    assign = list(result["assign"])
    start = list(result["start"])
    end = list(result["end"])
    makespan = int(max(end))
    schedule = _result_to_schedule(instance, assign, start)
    solution = SchedulingSolution(schedule, instance)

    return solution, float(makespan)


def solve_minizinc_with_intermediates(
    instance: "SchedulingInstance",
    solver_name: str = "gecode",
    time_limit_seconds: float | None = None,
    on_new_best=None,
) -> tuple["SchedulingSolution", float, list[float]]:
    """
    Solve with MiniZinc, streaming intermediate improving solutions.

    Uses the async ``Instance.solutions(intermediate_solutions=True)``
    generator to capture every improving bound the CP solver finds.
    Each improvement triggers `on_new_best(gen, solution, makespan, elapsed)`.

    Returns:
        (best_solution, best_makespan, history) — history contains the
        makespan of every intermediate improving solution.
    """

    async def _run():
        Instance = minizinc.Instance
        Model = minizinc.Model
        Solver = minizinc.Solver

        model_path = os.path.join(os.path.dirname(__file__), "scheduling.mzn")
        model = Model(model_path)
        solver = Solver.lookup(solver_name)
        mz_instance = Instance(solver, model)

        data = _instance_to_minizinc_data(instance)
        for key, value in data.items():
            mz_instance[key] = value

        kwargs = {"intermediate_solutions": True}
        if time_limit_seconds is not None and time_limit_seconds > 0:
            kwargs["time_limit"] = timedelta(seconds=time_limit_seconds)

        best_sol = None
        best_makespan = float("inf")
        history: list[float] = []
        gen = 0
        wall_start = time.monotonic()

        async for result in mz_instance.solutions(**kwargs):
            if result.status.name not in ("SATISFIED", "OPTIMAL_SOLUTION"):
                continue

            try:
                assign = list(result["assign"])
                start = list(result["start"])
                end = list(result["end"])
            except (KeyError, TypeError):
                # Final status-only result has no variable assignments
                continue

            makespan = float(max(end))
            if makespan < best_makespan:
                best_makespan = makespan
                schedule = _result_to_schedule(instance, assign, start)
                best_sol = SchedulingSolution(schedule, instance)
                history.append(best_makespan)
                gen += 1
                elapsed = time.monotonic() - wall_start
                if on_new_best is not None:
                    on_new_best(gen, best_sol, best_makespan, elapsed)

        return best_sol, best_makespan, history

    best_sol, best_makespan, history = asyncio.run(_run())

    # Fallback if no solution was found
    if best_sol is None:
        rng = np.random.default_rng()
        best_sol = SolverBase._random_solution(instance, rng)
        best_makespan = float(best_sol.compute_makespan())
        history = [best_makespan]

    return best_sol, best_makespan, history


class MinizincSolver:
    """
    CP solver using MiniZinc.

    This class deliberately does **not** subclass `SolverBase`, because
    MiniZinc runs a single CP solve instead of an iterative metaheuristic
    loop. However, it exposes the same high-level `solve(instance, ...)`
    signature so it can be selected via `--solver minizinc` alongside
    the other solvers.

    Behaviour:
    - Looks for a `TimeLimit` in its `criteria` and maps it to the
      MiniZinc `time_limit` argument (in seconds).
    - Runs one MiniZinc solve and calls `on_new_best` once with the
      obtained solution.
    - Returns a history list containing a single entry: the final best
      makespan.
    """

    def __init__(
        self,
        solver_name: str = "gecode",
        criteria: list[StoppingCriterion] | None = None,
    ):
        self.solver_name = solver_name
        self.criteria = criteria if criteria else [TimeLimit(60)]

    def solve(
        self,
        instance: "SchedulingInstance",
        on_new_best=None,
    ) -> tuple["SchedulingSolution", float, list[float]]:
        time_limit = None

        for c in self.criteria:
            c.reset()
            if isinstance(c, TimeLimit):
                time_limit = c.seconds
                break
        if time_limit is None:
            time_limit = 60.0

        solution, makespan, history = solve_minizinc_with_intermediates(
            instance,
            solver_name=self.solver_name,
            time_limit_seconds=time_limit,
            on_new_best=on_new_best,
        )

        for c in self.criteria:
            c.check(history)
        # Mark the criterion we used as triggered so displays correctly
        for c in self.criteria:
            if isinstance(c, TimeLimit):
                c.triggered = True
                break
        else:
            if self.criteria:
                self.criteria[0].triggered = True

        return solution, makespan, history
