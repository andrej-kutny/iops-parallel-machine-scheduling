import numpy as np

from models.solution import SchedulingSolution
from local_search.operators import (
    swap_within_machine,
    move_within_machine,
    move_to_other_machine,
    swap_between_machines,
    local_search,
    random_neighbor,
)


def test_swap_within_machine(small_instance):
    # Create a schedule where swapping might help
    schedule = {0: [], 1: [5], 2: [4, 3, 2, 1]}
    sol = SchedulingSolution(schedule, small_instance)
    original_makespan = sol.compute_makespan()

    result = swap_within_machine(sol, small_instance)
    assert result.compute_makespan() <= original_makespan


def test_move_within_machine(small_instance):
    schedule = {0: [], 1: [5], 2: [4, 3, 2, 1]}
    sol = SchedulingSolution(schedule, small_instance)
    original_makespan = sol.compute_makespan()

    result = move_within_machine(sol, small_instance)
    assert result.compute_makespan() <= original_makespan


def test_move_to_other_machine(small_instance):
    # Job 5 is on machine 2 but can go to 0 or 1
    schedule = {0: [], 1: [], 2: [2, 3, 1, 4, 5]}
    sol = SchedulingSolution(schedule, small_instance)
    original_makespan = sol.compute_makespan()

    result = move_to_other_machine(sol, small_instance)
    assert result.compute_makespan() <= original_makespan


def test_swap_between_machines(small_instance):
    # Job 5 on machine 2, could swap with... but only job 5 can go to other machines
    schedule = {0: [], 1: [5], 2: [2, 3, 1, 4]}
    sol = SchedulingSolution(schedule, small_instance)
    original_makespan = sol.compute_makespan()

    result = swap_between_machines(sol, small_instance)
    assert result.compute_makespan() <= original_makespan


def test_local_search_improves(small_instance):
    schedule = {0: [], 1: [], 2: [4, 3, 2, 1, 5]}
    sol = SchedulingSolution(schedule, small_instance)
    original_makespan = sol.compute_makespan()

    result = local_search(sol, small_instance)
    assert result.compute_makespan() <= original_makespan
    feasible, _ = result.is_feasible()
    assert feasible


def test_random_neighbor_feasible(small_instance):
    schedule = {0: [], 1: [5], 2: [2, 3, 1, 4]}
    sol = SchedulingSolution(schedule, small_instance)
    rng = np.random.default_rng(42)

    for _ in range(20):
        neighbor = random_neighbor(sol, small_instance, rng)
        feasible, msg = neighbor.is_feasible()
        assert feasible, f"Random neighbor infeasible: {msg}"


def test_local_search_feasibility(small_instance):
    """Ensure local search always produces feasible solutions."""
    schedule = {0: [], 1: [], 2: [1, 2, 3, 4, 5]}
    sol = SchedulingSolution(schedule, small_instance)
    result = local_search(sol, small_instance)
    feasible, msg = result.is_feasible()
    assert feasible, f"Local search produced infeasible: {msg}"
