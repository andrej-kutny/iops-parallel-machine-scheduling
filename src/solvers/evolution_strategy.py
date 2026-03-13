from __future__ import annotations

import time

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion
from local_search.operators import random_neighbor
from solvers.base import SolverBase, VerboseCallback


class EvolutionStrategySolver(SolverBase):
    """(μ+λ) Evolution Strategy with Rechenberg's 1/5 success rule."""

    def __init__(
        self,
        mu: int = 10,
        lam: int = 50,
        c: float = 0.85,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(criteria)
        assert 0 < c < 1, "c must be in (0, 1)"
        self.mu = mu
        self.lam = lam
        self.c = c

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        return self._random_solution(instance)

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return solution

    def _mutate(self, solution: SchedulingSolution, instance: SchedulingInstance, sigma: float, rng: np.random.Generator) -> SchedulingSolution:
        """Apply sigma random perturbations."""
        current = solution.copy()
        num_moves = max(1, int(sigma + 0.5))
        for _ in range(num_moves):
            current = random_neighbor(current, instance, rng)
        return current

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        """ES main loop with (μ+λ) selection and 1/5 success rule."""
        rng = np.random.default_rng()
        start = time.monotonic()

        # Initial population
        population = [self._random_solution(instance, rng) for _ in range(self.mu + self.lam)]
        fitness = [ind.compute_makespan() for ind in population]

        # Mutation strength
        sigma = max(1.0, instance.n / 10)
        sigma_min = 1.0
        sigma_max = instance.n / 2

        # Track best
        best_idx = int(np.argmin(fitness))
        best = population[best_idx].copy()
        best_cost = fitness[best_idx]
        history: list[float] = []

        if on_new_best is not None:
            on_new_best(0, best, best_cost, 0.0)

        for c in self.criteria:
            c.reset()

        gen = 0
        while True:
            # Truncation selection: keep best mu
            sorted_indices = np.argsort(fitness)[:self.mu]
            parents = [population[i] for i in sorted_indices]
            parent_fitness = [fitness[i] for i in sorted_indices]

            # Generate offspring
            offspring = []
            offspring_fitness = []
            success_count = 0

            offspring_per_parent = self.lam // self.mu
            for p_idx in range(self.mu):
                for _ in range(offspring_per_parent):
                    child = self._mutate(parents[p_idx], instance, sigma, rng)
                    child_fit = child.compute_makespan()
                    offspring.append(child)
                    offspring_fitness.append(child_fit)
                    if child_fit < parent_fitness[p_idx]:
                        success_count += 1

            # Remainder
            remainder = self.lam - offspring_per_parent * self.mu
            for r in range(remainder):
                p_idx = r % self.mu
                child = self._mutate(parents[p_idx], instance, sigma, rng)
                child_fit = child.compute_makespan()
                offspring.append(child)
                offspring_fitness.append(child_fit)
                if child_fit < parent_fitness[p_idx]:
                    success_count += 1

            # (mu + lambda) merge
            population = parents + offspring
            fitness = parent_fitness + offspring_fitness

            # Rechenberg's 1/5 success rule
            total_offspring = len(offspring)
            if total_offspring > 0:
                success_rate = success_count / total_offspring
                if success_rate > 0.2:
                    sigma /= self.c
                elif success_rate < 0.2:
                    sigma *= self.c
                sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Update best
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_cost:
                best = population[gen_best_idx].copy()
                best_cost = fitness[gen_best_idx]
                if on_new_best is not None:
                    on_new_best(gen, best, best_cost, time.monotonic() - start)

            history.append(best_cost)

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best, best_cost, history
