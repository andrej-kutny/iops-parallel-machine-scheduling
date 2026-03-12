from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.instance import SchedulingInstance
    from src.models.solution import SchedulingSolution


def _machine_completion_time(jobs: list[int], machine_id: int, instance: SchedulingInstance) -> int:
    """Compute completion time for a single machine's job sequence."""
    time = 0
    prev_job_idx = None
    for idx, job in enumerate(jobs):
        job_index = job - 1
        release_time = instance.release[job_index][machine_id]
        if idx == 0:
            start_time = max(release_time, time)
        else:
            setup_time = instance.setup[prev_job_idx][job_index][machine_id]
            start_time = max(release_time, time + setup_time)
        proc_time = instance.duration[job_index][machine_id]
        time = start_time + proc_time
        prev_job_idx = job_index
    return time


def swap_within_machine(solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
    """Swap two jobs within the same machine. First-improvement."""
    best = solution
    best_makespan = solution.compute_makespan()

    for machine_id, jobs in solution.schedule.items():
        if len(jobs) < 2:
            continue
        original_time = _machine_completion_time(jobs, machine_id, instance)
        for i in range(len(jobs) - 1):
            for j in range(i + 1, len(jobs)):
                new_jobs = list(jobs)
                new_jobs[i], new_jobs[j] = new_jobs[j], new_jobs[i]
                new_time = _machine_completion_time(new_jobs, machine_id, instance)
                if new_time < original_time:
                    new_schedule = {k: list(v) for k, v in solution.schedule.items()}
                    new_schedule[machine_id] = new_jobs
                    from src.models.solution import SchedulingSolution as SS
                    candidate = SS(new_schedule, instance)
                    candidate_makespan = candidate.compute_makespan()
                    if candidate_makespan < best_makespan:
                        best = candidate
                        best_makespan = candidate_makespan
                        return best
    return best


def move_within_machine(solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
    """Remove a job from position i and reinsert at position j on the same machine. First-improvement."""
    best = solution
    best_makespan = solution.compute_makespan()

    for machine_id, jobs in solution.schedule.items():
        if len(jobs) < 2:
            continue
        original_time = _machine_completion_time(jobs, machine_id, instance)
        for i in range(len(jobs)):
            job = jobs[i]
            remaining = jobs[:i] + jobs[i + 1:]
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                new_jobs = remaining[:j] + [job] + remaining[j:]
                new_time = _machine_completion_time(new_jobs, machine_id, instance)
                if new_time < original_time:
                    new_schedule = {k: list(v) for k, v in solution.schedule.items()}
                    new_schedule[machine_id] = new_jobs
                    from src.models.solution import SchedulingSolution as SS
                    candidate = SS(new_schedule, instance)
                    candidate_makespan = candidate.compute_makespan()
                    if candidate_makespan < best_makespan:
                        return candidate
    return best


def move_to_other_machine(solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
    """Move a job from one machine to another capable machine at the best position. First-improvement."""
    best = solution
    best_makespan = solution.compute_makespan()

    for src_machine, jobs in solution.schedule.items():
        if len(jobs) == 0:
            continue
        for job_pos, job in enumerate(jobs):
            job_index = job - 1
            src_without = jobs[:job_pos] + jobs[job_pos + 1:]
            for dst_machine in instance.capable[job_index]:
                if dst_machine == src_machine:
                    continue
                dst_jobs = solution.schedule.get(dst_machine, [])
                for insert_pos in range(len(dst_jobs) + 1):
                    new_dst = dst_jobs[:insert_pos] + [job] + dst_jobs[insert_pos:]
                    new_schedule = {k: list(v) for k, v in solution.schedule.items()}
                    new_schedule[src_machine] = src_without
                    new_schedule[dst_machine] = new_dst
                    from src.models.solution import SchedulingSolution as SS
                    candidate = SS(new_schedule, instance)
                    candidate_makespan = candidate.compute_makespan()
                    if candidate_makespan < best_makespan:
                        return candidate
    return best


def swap_between_machines(solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
    """Swap one job from machine A with one job from machine B (checking capability). First-improvement."""
    best = solution
    best_makespan = solution.compute_makespan()

    machines = list(solution.schedule.keys())
    for ai in range(len(machines)):
        ma = machines[ai]
        jobs_a = solution.schedule[ma]
        if len(jobs_a) == 0:
            continue
        for bi in range(ai + 1, len(machines)):
            mb = machines[bi]
            jobs_b = solution.schedule[mb]
            if len(jobs_b) == 0:
                continue
            for ia, job_a in enumerate(jobs_a):
                ja_idx = job_a - 1
                if mb not in instance.capable[ja_idx]:
                    continue
                for ib, job_b in enumerate(jobs_b):
                    jb_idx = job_b - 1
                    if ma not in instance.capable[jb_idx]:
                        continue
                    new_a = list(jobs_a)
                    new_b = list(jobs_b)
                    new_a[ia] = job_b
                    new_b[ib] = job_a
                    new_schedule = {k: list(v) for k, v in solution.schedule.items()}
                    new_schedule[ma] = new_a
                    new_schedule[mb] = new_b
                    from src.models.solution import SchedulingSolution as SS
                    candidate = SS(new_schedule, instance)
                    candidate_makespan = candidate.compute_makespan()
                    if candidate_makespan < best_makespan:
                        return candidate
    return best


def local_search(solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
    """Apply all operators in sequence until no improvement found."""
    operators = [swap_within_machine, move_within_machine, move_to_other_machine, swap_between_machines]
    improved = True
    while improved:
        improved = False
        for operator in operators:
            new_sol = operator(solution, instance)
            if new_sol.compute_makespan() < solution.compute_makespan():
                solution = new_sol
                improved = True
                break
    return solution


def random_neighbor(solution: SchedulingSolution, instance: SchedulingInstance, rng: np.random.Generator | None = None) -> SchedulingSolution:
    """Generate a random neighbor by applying a random perturbation. Used by SA."""
    if rng is None:
        rng = np.random.default_rng()

    from src.models.solution import SchedulingSolution as SS

    move_type = rng.integers(0, 4)
    new_schedule = {k: list(v) for k, v in solution.schedule.items()}

    if move_type == 0:
        # Swap within machine
        machines_with_jobs = [m for m, j in new_schedule.items() if len(j) >= 2]
        if machines_with_jobs:
            m = machines_with_jobs[rng.integers(len(machines_with_jobs))]
            jobs = new_schedule[m]
            i, j = rng.choice(len(jobs), size=2, replace=False)
            jobs[i], jobs[j] = jobs[j], jobs[i]
            return SS(new_schedule, instance)

    elif move_type == 1:
        # Move within machine
        machines_with_jobs = [m for m, j in new_schedule.items() if len(j) >= 2]
        if machines_with_jobs:
            m = machines_with_jobs[rng.integers(len(machines_with_jobs))]
            jobs = new_schedule[m]
            i = rng.integers(len(jobs))
            job = jobs.pop(i)
            j = rng.integers(len(jobs) + 1)
            jobs.insert(j, job)
            return SS(new_schedule, instance)

    elif move_type == 2:
        # Move to other machine
        machines_with_jobs = [m for m, j in new_schedule.items() if len(j) > 0]
        if machines_with_jobs:
            src = machines_with_jobs[rng.integers(len(machines_with_jobs))]
            jobs_src = new_schedule[src]
            idx = rng.integers(len(jobs_src))
            job = jobs_src[idx]
            job_index = job - 1
            other_machines = [m for m in instance.capable[job_index] if m != src]
            if other_machines:
                dst = other_machines[rng.integers(len(other_machines))]
                jobs_src.pop(idx)
                insert_pos = rng.integers(len(new_schedule[dst]) + 1)
                new_schedule[dst].insert(insert_pos, job)
                return SS(new_schedule, instance)

    elif move_type == 3:
        # Swap between machines
        machines_with_jobs = [m for m, j in new_schedule.items() if len(j) > 0]
        if len(machines_with_jobs) >= 2:
            ma, mb = rng.choice(machines_with_jobs, size=2, replace=False)
            jobs_a, jobs_b = new_schedule[ma], new_schedule[mb]
            for _ in range(10):
                ia = rng.integers(len(jobs_a))
                ib = rng.integers(len(jobs_b))
                ja_idx = jobs_a[ia] - 1
                jb_idx = jobs_b[ib] - 1
                if mb in instance.capable[ja_idx] and ma in instance.capable[jb_idx]:
                    jobs_a[ia], jobs_b[ib] = jobs_b[ib], jobs_a[ia]
                    return SS(new_schedule, instance)

    return solution.copy()
