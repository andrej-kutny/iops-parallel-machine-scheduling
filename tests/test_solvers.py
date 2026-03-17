import pytest

from stopping_criteria import MaxGenerations, TimeLimit
from solvers.grasp import GraspSolver
from solvers.simulated_annealing import SimulatedAnnealingSolver
from solvers.evolution_strategy import EvolutionStrategySolver
from solvers.ant_system import AntSystem, RankedAntSystem, EasAntSystem
from solvers.max_min_ant_system import MaxMinAntSystem
from solvers.ant_colony_system import AntColonySystem
from solvers.ant_multi_tour_system import AntMultiTourSystem
from solvers.iterated_local_search import ILSSolver
from solvers.genetic_algorithm import GeneticAlgorithmSolver


SHORT_CRITERIA = [MaxGenerations(3)]
SHORT_TIME_CRITERIA = [TimeLimit(2)]


class TestGrasp:
    def test_produces_feasible(self, small_instance):
        solver = GraspSolver(alpha=0.5, criteria=SHORT_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0
        assert len(history) > 0

    def test_alpha_zero_greedy(self, small_instance):
        solver = GraspSolver(alpha=0.0, criteria=SHORT_CRITERIA)
        solution, cost, _ = solver.solve(small_instance)
        feasible, _ = solution.is_feasible()
        assert feasible


class TestSimulatedAnnealing:
    def test_produces_feasible(self, small_instance):
        solver = SimulatedAnnealingSolver(criteria=SHORT_TIME_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0


class TestEvolutionStrategy:
    def test_produces_feasible(self, small_instance):
        solver = EvolutionStrategySolver(mu=5, lam=10, criteria=[MaxGenerations(3)])
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0


class TestAntSystem:
    def test_produces_feasible(self, small_instance):
        solver = AntSystem(n_ants=5, criteria=SHORT_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg


class TestRankedAntSystem:
    def test_produces_feasible(self, small_instance):
        solver = RankedAntSystem(n_ants=5, criteria=SHORT_CRITERIA)
        solution, cost, _ = solver.solve(small_instance)
        feasible, _ = solution.is_feasible()
        assert feasible


class TestEasAntSystem:
    def test_produces_feasible(self, small_instance):
        solver = EasAntSystem(n_ants=5, criteria=SHORT_CRITERIA)
        solution, cost, _ = solver.solve(small_instance)
        feasible, _ = solution.is_feasible()
        assert feasible


class TestMaxMinAntSystem:
    def test_produces_feasible(self, small_instance):
        solver = MaxMinAntSystem(n_ants=5, reinit_frequency=2, criteria=SHORT_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg


class TestAntColonySystem:
    def test_produces_feasible(self, small_instance):
        solver = AntColonySystem(n_ants=5, q0=0.9, criteria=SHORT_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg


class TestAntMultiTourSystem:
    def test_produces_feasible(self, small_instance):
        solver = AntMultiTourSystem(n_ants=5, q_tours=2, criteria=SHORT_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg


class TestILSSolver:
    def test_produces_feasible(self, small_instance):
        solver = ILSSolver(perturbation_strength=3, criteria=SHORT_CRITERIA)
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0
        assert len(history) > 0

    def test_history_non_increasing_best(self, small_instance):
        solver = ILSSolver(perturbation_strength=2, criteria=[MaxGenerations(5)])
        _, cost, history = solver.solve(small_instance)
        # Best-so-far is non-increasing
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-9  # allow float noise


class TestGeneticAlgorithmSolver:
    def test_produces_feasible(self, small_instance):
        solver = GeneticAlgorithmSolver(
            population_size=10, offspring_per_generation=10, mutation_strength=1, criteria=SHORT_CRITERIA
        )
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0
        assert len(history) > 0

    def test_crossover_produces_valid_schedule(self, small_instance):
        """Crossover + ordering should yield a solution with all jobs assigned once."""
        from solvers.base import SolverBase
        from solvers.genetic_algorithm import _crossover
        import numpy as np
        rng = np.random.default_rng(42)
        p1 = SolverBase._random_solution(small_instance, rng)
        p2 = SolverBase._random_solution(small_instance, rng)
        child = _crossover(p1, p2, small_instance, rng)
        feasible, msg = child.is_feasible()
        assert feasible, msg
        assert child.compute_makespan() > 0


class TestMinizincSolver:
    def test_produces_feasible(self, small_instance):
        minizinc = pytest.importorskip("minizinc")
        if getattr(minizinc, "default_driver", None) is None:
            pytest.skip("MiniZinc driver not found on the system")
        
        from minizinc_cp import MinizincSolver
        from stopping_criteria import TimeLimit
        solver = MinizincSolver(solver_name="gecode", criteria=[TimeLimit(5)])
        solution, cost, history = solver.solve(small_instance)
        feasible, msg = solution.is_feasible()
        assert feasible, msg
        assert cost > 0
        assert len(history) >= 1
