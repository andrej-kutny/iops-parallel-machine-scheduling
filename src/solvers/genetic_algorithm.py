"""
Genetic Algorithm for parallel machine scheduling.

Inspired by IOPS Assignment 4 (p-Median): population-based with crossover and mutation.
Crossover: for each job, take machine assignment from parent1 or parent2 (uniform).
Mutation: random neighbor moves. Selection: truncation (best mu from parents + offspring).
"""

from __future__ import annotations

import time

import numpy as np

from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import StoppingCriterion
from local_search.operators import random_neighbor
from solvers.base import SolverBase, VerboseCallback


def _schedule_from_assignments(assignments: list[int], instance: SchedulingInstance, rng: np.random.Generator) -> dict[int, list[int]]:
    """Build schedule dict from job->machine assignments. Jobs on each machine ordered by release date then shuffle tie-break."""
    schedule: dict[int, list[int]] = {k: [] for k in range(instance.m)}
    for j in range(instance.n):
        m = assignments[j]
        schedule[m].append(j + 1)
    # Order jobs on each machine by release date (optional: could leave random for diversity)
    for m in schedule:
        jobs = schedule[m]
        jobs.sort(key=lambda job_id: (instance.release[job_id - 1][m], rng.random()))
        schedule[m] = jobs
    return schedule


def _crossover(parent1: SchedulingSolution, parent2: SchedulingSolution, instance: SchedulingInstance, rng: np.random.Generator) -> SchedulingSolution:
    """Uniform crossover: for each job j, assign to parent1's machine or parent2's machine (at random)."""
    assignments = []
    for j in range(instance.n):
        m1 = next(m for m, jobs in parent1.schedule.items() if (j + 1) in jobs)
        m2 = next(m for m, jobs in parent2.schedule.items() if (j + 1) in jobs)
        m = m1 if rng.random() < 0.5 else m2
        assignments.append(m)
    schedule = _schedule_from_assignments(assignments, instance, rng)
    return SchedulingSolution(schedule, instance)


class GeneticAlgorithmSolver(SolverBase):
    """Genetic Algorithm: population, uniform crossover (machine assignment), mutation (random neighbor), (mu+lam) selection."""

    DEFAULT_POPULATION_SIZE = 53
    DEFAULT_OFFSPRING_PER_GENERATION = 3
    DEFAULT_MUTATION_STRENGTH = 2

    def __init__(
        self,
        population_size: int = DEFAULT_POPULATION_SIZE,
        offspring_per_generation: int = DEFAULT_OFFSPRING_PER_GENERATION,
        mutation_strength: int = DEFAULT_MUTATION_STRENGTH,
        criteria: list[StoppingCriterion] | None = None,
    ):
        super().__init__(criteria)
        self.population_size = max(2, population_size)
        self.offspring_per_generation = max(1, offspring_per_generation)
        self.mutation_strength = max(1, mutation_strength)

    def _construct(self, instance: SchedulingInstance) -> SchedulingSolution:
        return self._random_solution(instance)

    def _improve(self, solution: SchedulingSolution, instance: SchedulingInstance) -> SchedulingSolution:
        return solution

    def _mutate(self, solution: SchedulingSolution, instance: SchedulingInstance, rng: np.random.Generator) -> SchedulingSolution:
        current = solution.copy()
        for _ in range(self.mutation_strength):
            current = random_neighbor(current, instance, rng)
        return current

    def solve(
        self,
        instance: SchedulingInstance,
        on_new_best: VerboseCallback | None = None,
    ) -> tuple[SchedulingSolution, float, list[float]]:
        """GA main loop: population, crossover + mutation, truncation selection."""
        rng = np.random.default_rng()
        start = time.monotonic()

        population = [self._random_solution(instance, rng) for _ in range(self.population_size)]
        fitness = [ind.compute_makespan() for ind in population]

        best_idx = int(np.argmin(fitness))
        best = population[best_idx].copy()
        best_cost = fitness[best_idx]
        history: list[float] = [best_cost]

        if on_new_best is not None:
            on_new_best(0, best, best_cost, 0.0)

        for c in self.criteria:
            c.reset()

        gen = 0
        while True:
            # Selection: keep best population_size
            sorted_indices = np.argsort(fitness)[: self.population_size]
            parents = [population[i] for i in sorted_indices]

            # Offspring: crossover then mutate
            offspring = []
            for _ in range(self.offspring_per_generation):
                i, j = rng.choice(len(parents), size=2, replace=False)
                child = _crossover(parents[i], parents[j], instance, rng)
                child = self._mutate(child, instance, rng)
                offspring.append(child)

            offspring_fitness = [ind.compute_makespan() for ind in offspring]

            # (mu+lam): merge and keep best population_size
            combined = parents + offspring
            combined_fitness = [ind.compute_makespan() for ind in combined]
            best_combined_idx = int(np.argmin(combined_fitness))
            if combined_fitness[best_combined_idx] < best_cost:
                best = combined[best_combined_idx].copy()
                best_cost = combined_fitness[best_combined_idx]
                if on_new_best is not None:
                    on_new_best(gen, best, best_cost, time.monotonic() - start)

            next_indices = np.argsort(combined_fitness)[: self.population_size]
            population = [combined[i] for i in next_indices]
            fitness = [combined_fitness[i] for i in next_indices]

            history.append(best_cost)

            if any(c.check(history) for c in self.criteria):
                break
            gen += 1

        return best, best_cost, history
