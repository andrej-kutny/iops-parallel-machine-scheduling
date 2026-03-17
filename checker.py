#!/usr/bin/env python3
import json
import sys

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_and_evaluate(instance, solution):
    n = instance["n"]
    m = instance["m"]
    capable = instance["capable"]  # list of lists
    duration = instance["duration"]  # [job][machine]
    release = instance["release"]    # [job][machine]
    setup = instance["setup"]        # [prev_job][next_job][machine]

    schedule = solution.get("schedule", {})

    # --- 1. Check that all jobs are assigned exactly once ---
    assigned_jobs = []
    for machine_id, jobs in schedule.items():
        assigned_jobs.extend(jobs)
    assigned_jobs_set = set(assigned_jobs)

    if len(assigned_jobs) != len(assigned_jobs_set):
        return False, "Some jobs are assigned more than once"

    if assigned_jobs_set != set(range(1, n + 1)):
        missing = set(range(1, n + 1)) - assigned_jobs_set
        extra = assigned_jobs_set - set(range(1, n + 1))
        if missing:
            return False, f"Missing jobs: {sorted(missing)}"
        if extra:
            return False, f"Invalid job IDs in solution: {sorted(extra)}"

    # --- 2. Check capability constraints ---
    for machine_id_str, jobs in schedule.items():
        machine_id = int(machine_id_str)
        if machine_id < 0 or machine_id >= m:
            return False, f"Invalid machine index {machine_id}"
        for job in jobs:
            job_index = job - 1
            if machine_id not in capable[job_index]:
                return False, f"Job {job} assigned to machine {machine_id} which is not capable"

    # --- 3. Compute makespan ---
    makespan = 0
    for machine_id_str, jobs in schedule.items():
        machine_id = int(machine_id_str)
        time = 0
        prev_job = None
        for idx, job in enumerate(jobs):
            job_index = job - 1
            release_time = release[job_index][machine_id]
            if idx == 0:
                start_time = max(release_time, time)
            else:
                setup_time = setup[prev_job][job_index][machine_id]
                start_time = max(release_time, time + setup_time)
            proc_time = duration[job_index][machine_id]
            completion_time = start_time + proc_time
            time = completion_time
            prev_job = job_index
        makespan = max(makespan, time)

    return True, makespan

def main():
    if len(sys.argv) != 3:
        print("Usage: python checker.py instance.json solution.json")
        sys.exit(1)

    instance_path = sys.argv[1]
    solution_path = sys.argv[2]

    try:
        instance = load_json(instance_path)
        solution = load_json(solution_path)
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        sys.exit(1)

    feasible, result = check_and_evaluate(instance, solution)

    if feasible:
        print(f"Feasible: makespan = {result}")
    else:
        print(f"Infeasible: {result}")

if __name__ == "__main__":
    main()
