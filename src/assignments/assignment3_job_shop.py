"""
Optional algorithm from IOPS Assignment 3: Job Shop Scheduling.

This module solves the Job Shop Scheduling problem (each job has multiple
operations on different machines; solution is a permutation of task indices).
Instance format: .txt with first line n_jobs n_machines, then per-job lines
of (machine, duration) pairs. See load_instance().

Use as standalone:
    from assignments.assignment3_job_shop import load_instance, simulated_annealing
    data = load_instance("path/to/ft06.txt")
    best_sol, best_makespan, history = simulated_annealing(data["jobs_data"], iters=200)
"""

from __future__ import annotations

import collections
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Instance loading (Job Shop format: not the same as parallel machine JSON)
# ---------------------------------------------------------------------------


def load_instance(filepath: str) -> dict:
    """
    Parse a single Job Shop instance file.
    Format: first line n_jobs n_machines; then n_jobs lines of (machine, duration) pairs.
    Returns dict with jobs_data, num_jobs, num_machines, machines_count, all_machines, horizon.
    """
    path = Path(filepath)
    lines = path.read_text().strip().splitlines()

    n_jobs, n_machines = map(int, lines[0].split())
    jobs_data = []

    for i in range(1, 1 + n_jobs):
        parts = lines[i].split()
        job = [(int(parts[k]), int(parts[k + 1])) for k in range(0, len(parts), 2)]
        jobs_data.append(job)

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    horizon = sum(task[1] for job in jobs_data for task in job)

    return {
        "jobs_data": jobs_data,
        "num_jobs": n_jobs,
        "num_machines": n_machines,
        "machines_count": machines_count,
        "all_machines": all_machines,
        "horizon": horizon,
    }


assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")


def build_assigned_jobs(jobs_data, schedule):
    """Build per-machine list of assigned tasks from schedule dict (job_id, task_id) -> start_time."""
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, (machine, duration) in enumerate(job):
            start = schedule[(job_id, task_id)]
            assigned_jobs[machine].append(
                assigned_task_type(start=start, job=job_id, index=task_id, duration=duration)
            )
    return assigned_jobs


def print_schedule(assigned_jobs, all_machines, makespan):
    """Print schedule per machine (task names and [start, end] intervals)."""
    print("Solution:")
    for machine in all_machines:
        assigned_jobs[machine].sort()
        sol_line_tasks = "Machine " + str(machine) + ": "
        sol_line = "           "
        for assigned_task in assigned_jobs[machine]:
            name = f"job_{assigned_task.job}_task_{assigned_task.index}"
            sol_line_tasks += f"{name:15}"
            start, duration = assigned_task.start, assigned_task.duration
            sol_tmp = f"[{start},{start + duration}]"
            sol_line += f"{sol_tmp:15}"
        sol_line += "\n"
        sol_line_tasks += "\n"
        print(sol_line_tasks)
        print(sol_line)
    print(f"Schedule Length (makespan): {makespan}")


# ---------------------------------------------------------------------------
# Solution representation and evaluation
# ---------------------------------------------------------------------------


def create_initial_solution(jobs_data):
    """Random permutation of job indices (each job repeated for its number of tasks)."""
    solution = []
    for job_id, job in enumerate(jobs_data):
        solution += [job_id] * len(job)
    random.shuffle(solution)
    return solution


def create_neighbor(solution, mutations=1):
    """Return a new solution by swapping two random positions."""
    new_solution = solution.copy()
    for _ in range(mutations):
        x, y = random.sample(range(len(new_solution)), 2)
        new_solution[x], new_solution[y] = new_solution[y], new_solution[x]
    return new_solution


def create_schedule(solution, jobs_data):
    """Compute start times and makespan for a solution (permutation). Returns (schedule, makespan)."""
    schedule = {}
    job_time = {}
    machine_time = {}
    job_task_count = {}

    for job_id in range(len(jobs_data)):
        job_time[job_id] = 0
        job_task_count[job_id] = 0

    for job_id in solution:
        task_id = job_task_count[job_id]
        machine, duration = jobs_data[job_id][task_id]

        if machine not in machine_time:
            machine_time[machine] = 0

        start = max(machine_time[machine], job_time[job_id])
        schedule[(job_id, task_id)] = start

        machine_time[machine] = start + duration
        job_time[job_id] = start + duration
        job_task_count[job_id] += 1

    makespan = max(job_time.values())
    return schedule, makespan


# ---------------------------------------------------------------------------
# Metaheuristics
# ---------------------------------------------------------------------------


def hill_climbing(jobs_data, iters=200, initial_solution=None):
    """Hill climbing: accept only improving neighbors. Returns (best_solution, best_makespan, history)."""
    if initial_solution is not None:
        current_solution = initial_solution
    else:
        current_solution = create_initial_solution(jobs_data)
    _, current_makespan = create_schedule(current_solution, jobs_data)

    best_solution = current_solution[:]
    best_makespan = current_makespan
    history = []

    for _ in range(iters):
        neighbor_solution = create_neighbor(current_solution)
        _, neighbor_makespan = create_schedule(neighbor_solution, jobs_data)

        if neighbor_makespan < best_makespan:
            current_solution = neighbor_solution
            best_solution = neighbor_solution[:]
            best_makespan = neighbor_makespan

        history.append(best_makespan)
    return best_solution, best_makespan, history


def simulated_annealing(jobs_data, t_init=100, cooling_rate=0.95, iters=200):
    """Simulated annealing: accept worse solutions with probability exp(-delta/t). Returns (best_solution, best_makespan, history)."""
    current_solution = create_initial_solution(jobs_data)
    _, current_makespan = create_schedule(current_solution, jobs_data)
    best_solution = current_solution
    best_makespan = current_makespan
    t = t_init
    history = []

    for _ in range(iters):
        new_solution = create_neighbor(current_solution)
        _, new_makespan = create_schedule(new_solution, jobs_data)
        delta = new_makespan - current_makespan

        if delta < 0 or random.random() < np.exp(-delta / t):
            current_solution = new_solution
            current_makespan = new_makespan

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = new_solution

        t *= cooling_rate
        history.append(best_makespan)

    return best_solution, best_makespan, history


def ils(jobs_data, restarts=5, hc_iters=200, perturbation_strength=5):
    """Iterated Local Search: repeated hill-climbing with perturbation. Returns (best_solution, best_makespan, history)."""
    current_solution = create_initial_solution(jobs_data)
    _, best_makespan = create_schedule(current_solution, jobs_data)
    best_solution = current_solution[:]
    history = []

    for _ in range(restarts):
        local_solution, local_makespan, local_history = hill_climbing(
            jobs_data, iters=hc_iters, initial_solution=current_solution
        )
        history.extend(local_history)

        if local_makespan < best_makespan:
            best_solution = local_solution[:]
            best_makespan = local_makespan

        current_solution = create_neighbor(local_solution, mutations=perturbation_strength)

    return best_solution, best_makespan, history


def evolution_strategy(jobs_data, mu, lam, strategy, max_gen=200, adaptive=False):
    """Evolution strategy (mu+lambda or (mu,lambda)). Returns (best_solution, best_makespan, history)."""
    parents = []
    history = []
    mutations = 1

    for _ in range(mu):
        sol = create_initial_solution(jobs_data)
        _, makespan = create_schedule(sol, jobs_data)
        parents.append((makespan, sol))
    best_makespan, best_solution = min(parents)

    for _ in range(max_gen):
        new_solutions = []
        parent_makespans = []

        for _ in range(lam):
            parent = random.choice(parents)
            new_solutions.append(create_neighbor(parent[1], mutations))
            parent_makespans.append(parent[0])

        evaluated_new_solutions = []
        for o in new_solutions:
            _, makespans = create_schedule(o, jobs_data)
            evaluated_new_solutions.append((makespans, o))

        successes = sum(1 for i in range(len(evaluated_new_solutions)) if evaluated_new_solutions[i][0] < parent_makespans[i])
        if adaptive:
            if successes / lam > 1 / 5:
                mutations += 1
            elif (successes / lam < 1 / 5) and (mutations > 1):
                mutations -= 1

        if strategy == "plus":
            best_mu = sorted(parents + evaluated_new_solutions)[:mu]
        else:
            best_mu = sorted(evaluated_new_solutions)[:mu]

        parents = best_mu
        mu_best_makespan, mu_best_solution = min(best_mu)
        if mu_best_makespan < best_makespan:
            best_makespan = mu_best_makespan
            best_solution = mu_best_solution

        history.append(best_makespan)

    return best_solution, best_makespan, history


# ---------------------------------------------------------------------------
# Entry point for optional solver dispatch (different instance format)
# ---------------------------------------------------------------------------


def solve_job_shop(instance_path: str, algorithm: str = "simulated_annealing", **kwargs):
    """
    Load a Job Shop instance from instance_path and run the chosen algorithm.
    instance_path must be a .txt file in Job Shop format (see load_instance).
    algorithm: one of "hill_climbing", "simulated_annealing", "ils", "evolution_strategy".
    Returns (best_solution, best_makespan, history).
    """
    data = load_instance(instance_path)
    jobs_data = data["jobs_data"]

    if algorithm == "hill_climbing":
        return hill_climbing(jobs_data, **kwargs)
    if algorithm == "simulated_annealing":
        return simulated_annealing(jobs_data, **kwargs)
    if algorithm == "ils":
        return ils(jobs_data, **kwargs)
    if algorithm == "evolution_strategy":
        return evolution_strategy(jobs_data, **kwargs)
    raise ValueError(f"Unknown algorithm: {algorithm}")
