"""
Optional algorithm from IOPS Assignment 4: Capacitated p-Median (multi-objective).

This module solves the multi-objective p-Median problem: minimize service distance
and maximize facility dispersion, with capacity constraints (penalized if violated).
Instance format: see parse_all_instances() (p_median_capacitated.txt style).
Solution: list of p facility indices (from customer locations).

Use as standalone:
    from assignments.assignment4_p_median import parse_all_instances, build_distance_matrix, nsga_2, spea_2
    instances = parse_all_instances("p_median_capacitated.txt")
    dist = build_distance_matrix(instances[0]["customers"])
    solutions, objectives = nsga_2(instances[0], dist, pop_size=100, generations=100)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Instance loading (p-Median format: not the same as parallel machine JSON)
# ---------------------------------------------------------------------------


def parse_all_instances(filepath: str) -> list[dict]:
    """
    Parse p-Median capacitated instances from file.
    Format: per instance: line1 instance_id best_known, line2 n p capacity, then n lines customer_id x y demand.
    Returns list of dicts with instance_id, best_known, n, p, capacity, customers.
    """
    path = Path(filepath)
    lines = path.read_text().strip().splitlines()
    instances = []
    i = 0

    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 2:
            i += 1
            continue

        instance_id = int(parts[0])
        best_known = int(parts[1])
        i += 1

        if i >= len(lines):
            break

        n, p, capacity = map(int, lines[i].split())
        i += 1
        customers = []

        for _ in range(n):
            if i >= len(lines):
                break
            row = lines[i].split()
            cust_id, x, y, demand = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            customers.append((cust_id, x, y, demand))
            i += 1

        instances.append({
            "instance_id": instance_id,
            "best_known": best_known,
            "n": n,
            "p": p,
            "capacity": capacity,
            "customers": customers,
        })

    return instances


def build_distance_matrix(customers: list) -> np.ndarray:
    """Euclidean distance matrix from list of (id, x, y, demand) tuples."""
    n = len(customers)
    coords = np.array([[c[1], c[2]] for c in customers], dtype=float)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist[i, j] = np.sqrt(dx * dx + dy * dy)
    return dist


# ---------------------------------------------------------------------------
# Objectives and evaluation
# ---------------------------------------------------------------------------


def assign_customers_to_facilities(instance: dict, facility_indices: list, dist_matrix: np.ndarray) -> list:
    """Assign each customer to nearest facility. Returns list of facility local indices per customer."""
    n_customers = instance["n"]
    assignments = []
    for i in range(n_customers):
        best_facility_local_index = np.argmin([dist_matrix[i, f_index] for f_index in facility_indices])
        assignments.append(best_facility_local_index)
    return assignments


def compute_service_distance(instance: dict, facility_indices: list, assignments: list, dist_matrix: np.ndarray) -> float:
    """Total weighted distance: sum of demand[i] * distance[i, assigned_facility]."""
    total_weighted_distance = 0.0
    customers = instance["customers"]
    for i in range(len(assignments)):
        facility_global_index = facility_indices[assignments[i]]
        demand = customers[i][3]
        distance = dist_matrix[i, facility_global_index]
        total_weighted_distance += demand * distance
    return total_weighted_distance


def compute_dispersion(facility_indices: list, dist_matrix: np.ndarray) -> float:
    """Median of pairwise distances between facilities."""
    if len(facility_indices) < 2:
        return 0.0
    pairwise_distances = []
    for i in range(len(facility_indices)):
        for j in range(i + 1, len(facility_indices)):
            pairwise_distances.append(dist_matrix[facility_indices[i], facility_indices[j]])
    return float(np.median(pairwise_distances))


def evaluate_solution(instance: dict, facility_indices: list, dist_matrix: np.ndarray) -> tuple[float, float]:
    """Returns (service_distance + penalty, -dispersion) so both objectives are minimised."""
    penalty_weight = 1e5
    assignments = assign_customers_to_facilities(instance, facility_indices, dist_matrix)
    service_distances = compute_service_distance(instance, facility_indices, assignments, dist_matrix)
    facility_loads = np.zeros(len(facility_indices))

    for i, f_local_index in enumerate(assignments):
        facility_loads[f_local_index] += instance["customers"][i][3]

    penalty = 0.0
    for load in facility_loads:
        if load > instance["capacity"]:
            penalty += (load - instance["capacity"]) * penalty_weight
    service_distances += penalty
    dispersion = compute_dispersion(facility_indices, dist_matrix)
    return service_distances, -dispersion


# ---------------------------------------------------------------------------
# Evolutionary operators
# ---------------------------------------------------------------------------


def mutate_solution(solution: list, n: int, p: int) -> list:
    """Replace one facility with a random non-facility location."""
    new_solution = solution[:]
    non_facilities_locations = list(set(range(n)) - set(solution))
    if non_facilities_locations:
        index_to_remove = np.random.choice(p)
        new_location = np.random.choice(non_facilities_locations)
        new_solution[index_to_remove] = new_location
    return new_solution


def crossover_solutions(parent1: list, parent2: list, p: int) -> list:
    """Offspring from union of parents, random choice of p facilities."""
    combined = list(set(parent1) | set(parent2))
    offspring = np.random.choice(combined, size=p, replace=False).tolist()
    return offspring


def create_random_solution(n: int, p: int) -> list:
    """Random set of p facility indices from 0..n-1."""
    return np.random.choice(n, size=p, replace=False).tolist()


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Returns list of fronts (each front is list of solution indices)."""
    n_pop = objectives.shape[0]
    dominance_count = np.zeros(n_pop, dtype=int)
    dominated_sets = [[] for _ in range(n_pop)]
    fronts = [[]]

    for i in range(n_pop):
        for j in range(n_pop):
            if all(objectives[i] <= objectives[j]) and any(objectives[i] < objectives[j]):
                dominated_sets[i].append(j)
            elif all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                dominance_count[i] += 1
        if dominance_count[i] == 0:
            fronts[0].append(i)

    current_front_index = 0
    while len(fronts[current_front_index]) > 0:
        next_front = []
        for i in fronts[current_front_index]:
            for j in dominated_sets[i]:
                dominance_count[j] -= 1
                if dominance_count[j] == 0:
                    next_front.append(j)
        current_front_index += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(objectives: np.ndarray, front_indices: list) -> np.ndarray:
    """Crowding distance for solutions in the front."""
    num_in_front = len(front_indices)
    distances = np.zeros(num_in_front)
    front_obj = objectives[front_indices]

    if num_in_front <= 2:
        distances.fill(float("inf"))
        return distances

    for m in range(objectives.shape[1]):
        sorted_indices = sorted(range(num_in_front), key=lambda i: front_obj[i, m])
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")
        obj_min = front_obj[sorted_indices[0], m]
        obj_max = front_obj[sorted_indices[-1], m]
        obj_range = obj_max - obj_min
        if obj_range == 0:
            continue
        for i in range(1, num_in_front - 1):
            left_val = front_obj[sorted_indices[i - 1], m]
            right_val = front_obj[sorted_indices[i + 1], m]
            gap = (right_val - left_val) / obj_range
            distances[sorted_indices[i]] += gap

    return distances


def nsga_2(instance: dict, dist_matrix: np.ndarray, pop_size: int, generations: int) -> tuple[list, np.ndarray]:
    """NSGA-II for capacitated p-Median. Returns (population_solutions, objectives_array)."""
    n = instance["n"]
    p = instance["p"]
    pop = [create_random_solution(n, p) for _ in range(pop_size)]
    objectives = np.array([evaluate_solution(instance, sol, dist_matrix) for sol in pop])

    for _ in range(generations):
        offspring = []
        while len(offspring) < pop_size:
            idx = np.random.choice(len(pop), size=2, replace=False)
            parent1, parent2 = pop[idx[0]], pop[idx[1]]
            child = crossover_solutions(parent1, parent2, p)
            if np.random.random() < 0.2:
                child = mutate_solution(child, n, p)
            offspring.append(child)

        offspring_objectives = np.array([evaluate_solution(instance, sol, dist_matrix) for sol in offspring])
        combined_pop = pop + offspring
        combined_objectives = np.concatenate((objectives, offspring_objectives), axis=0)

        fronts = non_dominated_sort(combined_objectives)
        new_pop = []

        for front_indices in fronts:
            if len(new_pop) + len(front_indices) <= pop_size:
                new_pop.extend(front_indices)
            else:
                distances = crowding_distance(combined_objectives, front_indices)
                last_front = sorted(zip(front_indices, distances), key=lambda x: x[1], reverse=True)
                needed = pop_size - len(new_pop)
                for i in range(needed):
                    new_pop.append(last_front[i][0])
                break

        pop = [combined_pop[i] for i in new_pop]
        objectives = combined_objectives[new_pop]

    return pop, objectives


# ---------------------------------------------------------------------------
# SPEA-2
# ---------------------------------------------------------------------------


def compute_strength(objectives: np.ndarray) -> np.ndarray:
    """Number of solutions each solution dominates."""
    n_pop = objectives.shape[0]
    strength = np.zeros(n_pop, dtype=int)
    for i in range(n_pop):
        for j in range(n_pop):
            if i == j:
                continue
            if all(objectives[i] <= objectives[j]) and any(objectives[i] < objectives[j]):
                strength[i] += 1
    return strength


def spea2_fitness(objectives: np.ndarray, strength: np.ndarray, k: int) -> np.ndarray:
    """Fitness = raw fitness (sum of strength of dominators) + density. Lower is better."""
    n_pop = objectives.shape[0]
    raw_fitness = np.zeros(n_pop)
    for i in range(n_pop):
        for j in range(n_pop):
            if i == j:
                continue
            if all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                raw_fitness[i] += strength[j]

    objective_dists = np.zeros((n_pop, n_pop))
    for i in range(n_pop):
        for j in range(n_pop):
            dx = objectives[i, 0] - objectives[j, 0]
            dy = objectives[i, 1] - objectives[j, 1]
            objective_dists[i, j] = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(objective_dists, np.inf)
    kth_neighbor_index = min(k, n_pop - 1)
    distance_to_kth_nearest = np.sort(objective_dists, axis=1)[:, kth_neighbor_index]
    density = 1.0 / (distance_to_kth_nearest + 2.0)
    return raw_fitness + density


def reduce_non_dominated(objectives: np.ndarray, front_indices: list, elite_size: int, k: int) -> list:
    """Reduce Pareto front to elite_size by removing most crowded solutions."""
    current_indices = list(front_indices)
    front_objectives = objectives[current_indices]

    while len(current_indices) > elite_size:
        n_current = len(current_indices)
        dx = front_objectives[:, np.newaxis, 0] - front_objectives[np.newaxis, :, 0]
        dy = front_objectives[:, np.newaxis, 1] - front_objectives[np.newaxis, :, 1]
        pairwise_dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(pairwise_dists, np.inf)
        kth_neighbor_index = min(k, n_current - 1)
        distance_to_kth_nearest = np.sort(pairwise_dists, axis=1)[:, kth_neighbor_index]
        index_to_remove = np.argmin(distance_to_kth_nearest)
        current_indices.pop(index_to_remove)
        front_objectives = objectives[current_indices]

    return current_indices


def spea_2(instance: dict, dist_matrix: np.ndarray, pop_size: int, generations: int) -> tuple[list, np.ndarray]:
    """SPEA-2 for capacitated p-Median. Returns (elite_set solutions, objectives_array)."""
    n = instance["n"]
    p = instance["p"]
    elite_size = pop_size
    k_nearest = max(1, int(np.sqrt(2 * pop_size)))

    pop = [create_random_solution(n, p) for _ in range(pop_size)]
    pop_objectives = np.array([evaluate_solution(instance, sol, dist_matrix) for sol in pop])
    elite_set = []
    elite_objectives = np.zeros((0, 2))

    for _ in range(generations):
        combined_pop = pop + elite_set
        combined_objectives = np.concatenate((pop_objectives, elite_objectives), axis=0)
        n_pop = len(combined_pop)

        strength = compute_strength(combined_objectives)
        fitness = spea2_fitness(combined_objectives, strength, k_nearest)

        fronts = non_dominated_sort(combined_objectives)
        non_dominated = list(fronts[0])

        if len(non_dominated) > elite_size:
            non_dominated = reduce_non_dominated(combined_objectives, non_dominated, elite_size, k_nearest)
        elif len(non_dominated) < elite_size:
            remaining = [i for i in range(n_pop) if i not in non_dominated]
            remaining.sort(key=lambda i: fitness[i])
            non_dominated = non_dominated + remaining[: elite_size - len(non_dominated)]

        elite_set = [combined_pop[i] for i in non_dominated]
        elite_objectives = combined_objectives[non_dominated].copy()

        selected_parents = []
        for _ in range(pop_size):
            i, j = np.random.choice(len(elite_set), size=2, replace=False)
            selected_parent_idx = i if fitness[non_dominated[i]] <= fitness[non_dominated[j]] else j
            selected_parents.append(elite_set[selected_parent_idx][:])

        offspring = []
        for i in range(0, pop_size, 2):
            c1 = crossover_solutions(selected_parents[i], selected_parents[(i + 1) % pop_size], p)
            c2 = crossover_solutions(selected_parents[(i + 1) % pop_size], selected_parents[i], p)
            if np.random.random() < 0.2:
                c1 = mutate_solution(c1, n, p)
            if np.random.random() < 0.2:
                c2 = mutate_solution(c2, n, p)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        offspring = offspring[:pop_size]

        pop = offspring
        pop_objectives = np.array([evaluate_solution(instance, sol, dist_matrix) for sol in pop])

    combined_pop = pop + elite_set
    combined_objectives = np.concatenate((pop_objectives, elite_objectives), axis=0)
    fronts = non_dominated_sort(combined_objectives)
    final_indices = list(fronts[0])
    if len(final_indices) > elite_size:
        final_indices = reduce_non_dominated(combined_objectives, final_indices, elite_size, k_nearest)

    solutions = [combined_pop[i] for i in final_indices]
    objectives = combined_objectives[final_indices]
    return solutions, objectives


# ---------------------------------------------------------------------------
# Entry point for optional solver dispatch (different instance format)
# ---------------------------------------------------------------------------


def solve_p_median(
    instance_path: str,
    algorithm: str = "nsga2",
    instance_index: int = 0,
    pop_size: int = 100,
    generations: int = 100,
):
    """
    Load p-Median instances from instance_path, pick instance at instance_index,
    run algorithm (nsga2 or spea2). Returns (solutions, objectives).
    """
    instances = parse_all_instances(instance_path)
    if instance_index >= len(instances):
        raise IndexError(f"instance_index {instance_index} out of range (max {len(instances) - 1})")
    inst = instances[instance_index]
    dist = build_distance_matrix(inst["customers"])

    if algorithm == "nsga2":
        return nsga_2(inst, dist, pop_size=pop_size, generations=generations)
    if algorithm == "spea2":
        return spea_2(inst, dist, pop_size=pop_size, generations=generations)
    raise ValueError(f"Unknown algorithm: {algorithm}")
