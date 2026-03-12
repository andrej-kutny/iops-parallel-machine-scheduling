import pytest

from src.stopping_criteria import MaxGenerations, TimeLimit, GenMinImprovement
from src.solvers.grasp import GraspSolver
from src.solvers.simulated_annealing import SimulatedAnnealingSolver
from src.solvers.combined import CombinedSolver
from src.models.solution import SchedulingSolution
from src.solvers.base import SolverBase


class TestCombinedSolver:
    def test_produces_feasible(self, small_instance):
        solvers = [
            GraspSolver(alpha=0.5),
            SimulatedAnnealingSolver(initial_temp=50.0, cooling_rate=0.99),
        ]
        combined = CombinedSolver(
            solvers=solvers,
            criteria=[GenMinImprovement(window=5)],
        )
        solution, cost, history = combined.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0
        assert len(history) > 0

    def test_runs_multiple_solvers(self, small_instance):
        """Verify the combined solver runs multiple sub-solvers."""
        solvers = [
            GraspSolver(alpha=0.5),
            SimulatedAnnealingSolver(initial_temp=50.0, cooling_rate=0.99),
        ]
        combined = CombinedSolver(
            solvers=solvers,
            criteria=[GenMinImprovement(window=3, min_pct=0.5)],
        )
        solution, cost, history = combined.solve(small_instance)
        # Should have history entries from multiple sub-solver runs
        assert len(history) > 1

    def test_solver_switch_callback(self, small_instance):
        """Verify on_solver_switch is called when switching solvers."""
        switches = []
        solvers = [
            GraspSolver(alpha=0.5),
            SimulatedAnnealingSolver(initial_temp=50.0, cooling_rate=0.99),
        ]
        combined = CombinedSolver(
            solvers=solvers,
            criteria=[GenMinImprovement(window=3, min_pct=0.5)],
        )
        combined.solve(
            small_instance,
            on_solver_switch=lambda prev, nxt: switches.append((prev, nxt)),
        )
        # At least one switch should have occurred
        assert len(switches) >= 1

    def test_triggered_criteria(self, small_instance):
        """Verify triggered flags are propagated back."""
        solvers = [GraspSolver(alpha=0.5)]
        criteria = [GenMinImprovement(window=3)]
        combined = CombinedSolver(solvers=solvers, criteria=criteria)
        combined.solve(small_instance)
        assert any(c.triggered for c in combined.criteria)
