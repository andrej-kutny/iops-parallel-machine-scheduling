import json
import subprocess
import os

from src.models.solution import SchedulingSolution


def test_makespan_matches_checker(small_instance, valid_schedule, tmp_path):
    """Verify our makespan computation matches checker.py."""
    makespan = valid_schedule.compute_makespan()
    assert makespan > 0

    # Also verify against checker.py
    sol_path = tmp_path / "solution.json"
    with open(sol_path, "w") as f:
        json.dump(valid_schedule.to_json(), f)

    inst_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "75_3_5_H.json")
    checker_path = os.path.join(os.path.dirname(__file__), "..", "checker.py")

    result = subprocess.run(
        ["python", checker_path, inst_path, str(sol_path)],
        capture_output=True, text=True,
    )
    assert "Feasible" in result.stdout
    # Extract checker makespan
    checker_makespan = int(result.stdout.strip().split("=")[-1].strip())
    assert makespan == checker_makespan


def test_is_feasible_valid(valid_schedule):
    feasible, msg = valid_schedule.is_feasible()
    assert feasible is True


def test_is_feasible_missing_job(small_instance):
    schedule = {0: [], 1: [5], 2: [2, 3, 1]}  # missing job 4
    sol = SchedulingSolution(schedule, small_instance)
    feasible, msg = sol.is_feasible()
    assert feasible is False
    assert "Missing" in msg


def test_is_feasible_duplicate_job(small_instance):
    schedule = {0: [], 1: [5, 5], 2: [2, 3, 1, 4]}
    sol = SchedulingSolution(schedule, small_instance)
    feasible, msg = sol.is_feasible()
    assert feasible is False


def test_is_feasible_wrong_machine(small_instance):
    # Job 1 (index 0) can only run on machine 2, assigning to machine 0
    schedule = {0: [1], 1: [5], 2: [2, 3, 4]}
    sol = SchedulingSolution(schedule, small_instance)
    feasible, msg = sol.is_feasible()
    assert feasible is False
    assert "not capable" in msg


def test_to_json(valid_schedule):
    output = valid_schedule.to_json()
    assert "makespan" in output
    assert "schedule" in output
    assert isinstance(output["makespan"], int)


def test_copy(valid_schedule):
    copied = valid_schedule.copy()
    assert copied.compute_makespan() == valid_schedule.compute_makespan()
    # Mutating copy should not affect original
    copied.schedule[1].append(99)
    assert 99 not in valid_schedule.schedule[1]


def test_compute_machine_makespan(valid_schedule):
    """Test per-machine makespan computation."""
    m2_time = valid_schedule.compute_machine_makespan(2)
    assert m2_time > 0
    m0_time = valid_schedule.compute_machine_makespan(0)
    assert m0_time == 0  # no jobs on machine 0
