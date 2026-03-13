from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .instance import SchedulingInstance


class SchedulingSolution:
    """Schedule mapping machine IDs to ordered lists of 1-indexed job IDs."""

    def __init__(self, schedule: dict[int, list[int]], instance: SchedulingInstance):
        self.schedule = schedule
        self.instance = instance
        self._makespan: int | None = None

    def compute_makespan(self) -> int:
        """Compute makespan matching checker.py logic exactly."""
        if self._makespan is not None:
            return self._makespan

        inst = self.instance
        makespan = 0

        for machine_id, jobs in self.schedule.items():
            time = 0
            prev_job_idx = None
            for idx, job in enumerate(jobs):
                job_index = job - 1
                release_time = inst.release[job_index][machine_id]
                if idx == 0:
                    start_time = max(release_time, time)
                else:
                    setup_time = inst.setup[prev_job_idx][job_index][machine_id]
                    start_time = max(release_time, time + setup_time)
                proc_time = inst.duration[job_index][machine_id]
                completion_time = start_time + proc_time
                time = completion_time
                prev_job_idx = job_index
            makespan = max(makespan, time)

        self._makespan = int(makespan)
        return self._makespan

    def compute_machine_makespan(self, machine_id: int) -> int:
        """Compute completion time for a single machine (for delta evaluation)."""
        inst = self.instance
        jobs = self.schedule.get(machine_id, [])
        time = 0
        prev_job_idx = None
        for idx, job in enumerate(jobs):
            job_index = job - 1
            release_time = inst.release[job_index][machine_id]
            if idx == 0:
                start_time = max(release_time, time)
            else:
                setup_time = inst.setup[prev_job_idx][job_index][machine_id]
                start_time = max(release_time, time + setup_time)
            proc_time = inst.duration[job_index][machine_id]
            time = start_time + proc_time
            prev_job_idx = job_index
        return time

    def is_feasible(self) -> tuple[bool, str]:
        """Check all jobs assigned once and capability constraints."""
        inst = self.instance
        assigned = []
        for machine_id, jobs in self.schedule.items():
            if machine_id < 0 or machine_id >= inst.m:
                return False, f"Invalid machine index {machine_id}"
            for job in jobs:
                job_index = job - 1
                if job_index < 0 or job_index >= inst.n:
                    return False, f"Invalid job ID {job}"
                if machine_id not in inst.capable[job_index]:
                    return False, f"Job {job} not capable on machine {machine_id}"
            assigned.extend(jobs)

        assigned_set = set(assigned)
        expected = set(range(1, inst.n + 1))

        if len(assigned) != len(assigned_set):
            return False, "Some jobs assigned more than once"
        if assigned_set != expected:
            missing = expected - assigned_set
            extra = assigned_set - expected
            if missing:
                return False, f"Missing jobs: {sorted(missing)}"
            if extra:
                return False, f"Invalid job IDs: {sorted(extra)}"

        return True, "Feasible"

    def invalidate_makespan(self):
        """Call after modifying the schedule to force recomputation."""
        self._makespan = None

    def to_json(self) -> dict:
        """Return JSON-serializable output matching expected format."""
        return {
            "makespan": self.compute_makespan(),
            "schedule": {str(k): v for k, v in self.schedule.items()},
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_json(), indent=2)

    def copy(self) -> SchedulingSolution:
        new_schedule = {k: list(v) for k, v in self.schedule.items()}
        sol = SchedulingSolution(new_schedule, self.instance)
        sol._makespan = self._makespan
        return sol

    def __repr__(self) -> str:
        inst = self.instance
        lines = [f"SchedulingSolution(makespan={self.compute_makespan()}, n={inst.n}, m={inst.m})"]
        for machine_id in sorted(self.schedule.keys()):
            jobs = self.schedule[machine_id]
            if not jobs:
                lines.append(f"  Machine {machine_id}: (idle)")
                continue
            lines.append(f"  Machine {machine_id}:")
            time = 0
            prev_job_idx = None
            for idx, job in enumerate(jobs):
                job_index = job - 1
                release_time = inst.release[job_index][machine_id]
                if idx == 0:
                    start_time = max(release_time, time)
                    setup_time = 0
                else:
                    setup_time = inst.setup[prev_job_idx][job_index][machine_id]
                    start_time = max(release_time, time + setup_time)
                proc_time = inst.duration[job_index][machine_id]
                completion_time = start_time + proc_time
                wait = start_time - time - (setup_time if idx > 0 else 0)
                lines.append(
                    f"    Job {job:3d}: release={release_time:4d}  setup={setup_time:3d}"
                    f"  start={start_time:4d}  proc={proc_time:4d}  done={completion_time:4d}"
                    + (f"  (waited {wait})" if wait > 0 else "")
                )
                time = completion_time
                prev_job_idx = job_index
            lines.append(f"    Machine done at: {time}")
        return "\n".join(lines)
