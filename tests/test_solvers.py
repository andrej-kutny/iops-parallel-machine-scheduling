import pytest

from src.stopping_criteria import MaxGenerations, TimeLimit
from src.solvers.grasp import GraspSolver
from src.solvers.simulated_annealing import SimulatedAnnealingSolver
from src.solvers.evolution_strategy import EvolutionStrategySolver
from src.solvers.ant_system import AntSystem, RankedAntSystem, EasAntSystem
from src.solvers.max_min_ant_system import MaxMinAntSystem
from src.solvers.ant_colony_system import AntColonySystem
from src.solvers.ant_multi_tour_system import AntMultiTourSystem


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
