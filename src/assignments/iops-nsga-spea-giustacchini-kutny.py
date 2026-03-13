# %% [markdown]
# # Algorithms

# %%
import matplotlib.pyplot as plt
import numpy as np
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

# np.random.seed(123)
FINE_TUNE_SPEA = 25
FINE_TUNE_NSGA3 = 25

# %% [markdown]
# ## Preprocessing

# %%
@dataclass
class PMedianInstance:
    instance_id: int
    best_known: int
    num_customers: int
    num_medians: int
    capacity: int
    coords: np.ndarray
    demands: np.ndarray

    def __repr__(self):
        return (
            f"PMedianInstance(id={self.instance_id}, best_known={self.best_known}, "
            f"customers={self.num_customers}, medians={self.num_medians}, "
            f"capacity={self.capacity})"
        )


def parse_instances(raw: str) -> np.ndarray:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    parsed: list[PMedianInstance] = []
    i = 0
    while i < len(lines):
        # First line: instance_id  best_known_solution
        parts = lines[i].split()
        instance_id = int(parts[0])
        best_known = int(parts[1])
        i += 1

        # Second line: num_customers  num_medians  capacity
        parts = lines[i].split()
        num_customers = int(parts[0])
        num_medians = int(parts[1])
        capacity = int(parts[2])
        i += 1

        # Remaining lines for this instance: customer_id  x  y  demand
        cids, xs, ys, ds = [], [], [], []
        for _ in range(num_customers):
            parts = lines[i].split()
            cids.append(int(parts[0]))
            xs.append(int(parts[1]))
            ys.append(int(parts[2]))
            ds.append(int(parts[3]))
            i += 1

        parsed.append(PMedianInstance(
            instance_id=instance_id,
            best_known=best_known,
            num_customers=num_customers,
            num_medians=num_medians,
            capacity=capacity,
            coords=np.column_stack([xs, ys]),
            demands=np.array(ds),
        ))

    # 0-indexed: arr[idx] is the instance with instance_id == idx+1
    arr = np.empty(len(parsed), dtype=object)
    for inst in parsed:
        arr[inst.instance_id - 1] = inst
    return arr


def load_instance(instance_id: int, filepath: str = "p_median_capacitated.txt") -> PMedianInstance:
    with open(filepath) as f:
        raw = f.read()
    return parse_instances(raw)[instance_id - 1]


def load_instance_from_string(raw: str, instance_id: int = 999) -> PMedianInstance:
    return parse_instances(raw)[instance_id - 1]


def load_all_instances(filepath: str = "p_median_capacitated.txt") -> np.ndarray:
    with open(filepath) as f:
        raw = f.read()
    return parse_instances(raw)

# %%
# Sanity check - load instance 1 and verify against known values from the assignment
inst = load_instance(1)
print(inst)
assert inst.num_customers == 50
assert inst.num_medians == 5
assert inst.capacity == 120
assert inst.best_known == 713
assert list(inst.coords[0]) == [2, 62]
assert inst.demands[0] == 3
print(f"Customers: {inst.num_customers}, first coords: {inst.coords[0]}, first demand: {inst.demands[0]}")
print(f"Total demand: {inst.demands.sum()}, total capacity: {inst.capacity * inst.num_medians}")

# %%
# Overview of all 18 benchmark instances
all_instances = load_all_instances()
for inst in all_instances:
    print(
        f"Instance {inst.instance_id:2d}: {inst.num_customers:3d} customers, "
        f"{inst.num_medians:2d} medians, capacity={inst.capacity}, "
        f"best_known={inst.best_known}, total_demand={inst.demands.sum()}"
    )

# %% [markdown]
# ### Stopping criteria
# 
# Multiple stopping criteria can be combined. The algorithm halts when **any** criterion triggers:
# 
# - **MaxGenerations** — fixed generation budget.
# - **TimeLimit** — wall-clock time budget.
# - **MinImprovement** — stops when the best service distance improves by less than a threshold over a sliding window of generations (convergence detection).
# - **TargetObjective** — stops when the best service distance reaches or beats a known target value.

# %%
class StoppingCriterion(ABC):
    def reset(self):
        pass
    @abstractmethod
    def should_stop(self, gen: int, history: dict) -> bool: ...
    @abstractmethod
    def __repr__(self) -> str: ...


class MaxGenerations(StoppingCriterion):
    def __init__(self, n: int):
        assert n > 0, "n must be > 0"
        self.n = n
    def should_stop(self, gen, history):
        return gen + 1 >= self.n
    def __repr__(self):
        return f"MaxGenerations({self.n})"


class TimeLimit(StoppingCriterion):
    def __init__(self, seconds: float):
        assert seconds > 0, "seconds must be > 0"
        self.seconds = seconds
        self._start = None
    def reset(self):
        self._start = time.monotonic()
    def should_stop(self, gen, history):
        return (time.monotonic() - self._start) >= self.seconds
    def __repr__(self):
        return f"TimeLimit({self.seconds}s)"


class MinImprovement(StoppingCriterion):
    def __init__(self, window: int, min_pct: float):
        assert window > 0, "window must be > 0"
        assert 0 <= min_pct < 1, "min_pct must be in [0, 1)"
        self.window = window
        self.min_pct = np.clip(min_pct, 1e-15, 1. - 1e-15)
        self._buf: np.ndarray = np.empty(0)
    def reset(self):
        self._buf = np.empty(0)
    def should_stop(self, gen, history):
        current = history["best_service_dist"][-1]
        self._buf = np.append(self._buf, current)[-self.window:]
        if len(self._buf) < self.window:
            return False
        oldest = self._buf[0]
        if oldest == 0:
            return False
        improvement = (oldest - current) / oldest
        return improvement <= self.min_pct
    def __repr__(self):
        return f"MinImprovement(window={self.window}, min_pct={self.min_pct})"


class TargetObjective(StoppingCriterion):
    def __init__(self, target: float):
        assert target > 0, "target must be > 0"
        self.target = target
    def should_stop(self, gen, history):
        h = history["best_service_dist"]
        return bool(h) and h[-1] <= self.target
    def __repr__(self):
        return f"TargetObjective({self.target})"

# %% [markdown]
# ## Strength Pareto Evolutionary Algorithm (SPEA2)
# 
# SPEA2 is an elitist multi-objective evolutionary algorithm that maintains an external **archive** of non-dominated solutions across generations. Our implementation follows Algorithms 105–107 from *Essentials of Metaheuristics*.
# 
# **Representation** an integer array of length `num_customers` where `individual[i]` = median ID (0 to *p*−1) that customer *i* is assigned to. Facility locations are derived as the customer nearest to each cluster's centroid.
# 
# **Objectives** (both treated as minimisation):
# 1. Service distance + capacity penalty (minimise)
# 2. Negative facility dispersion (minimise == maximise dispersion)
# 
# **Constraint handling** uses a dual approach:
# - A **penalty function** adds a weighted sum of capacity excess to objective 1, providing selection pressure away from infeasible regions.
# - A **repair operator** applied after crossover and mutation guarantees all offspring are feasible by reassigning customers from overloaded facilities to the nearest under-capacity facility.

# %% [markdown]
# #### Problem-specific helpers
# 
# Functions for evaluating a solution on the capacitated p-median problem:
# 
# - `compute_facility_locations` — derives facility coordinates from a cluster assignment by finding the customer closest to each cluster's centroid.
# - `compute_service_distance` — total Euclidean distance from each customer to its assigned facility.
# - `compute_dispersion` — sum of pairwise Euclidean distances between all facility locations (higher = more spread out).
# - `compute_capacity_penalty` — sums the excess demand over capacity for all overloaded facilities, weighted by `penalty_weight`.
# - `is_feasible` — checks whether all facilities satisfy the capacity constraint.
# - `repair_capacity` — Lamarckian repair operator: for each overloaded facility, moves the farthest assigned customers to the nearest under-capacity facility until the constraint is satisfied.
# - `evaluate_population` — computes both objectives for every individual in the population.

# %%
def compute_facility_locations(individual, coords, num_medians):
    facility_coords = np.empty((num_medians, 2), dtype=float)
    for m in range(num_medians):
        mask = individual == m
        if mask.any():
            centroid = coords[mask].mean(axis=0)
            dists = np.linalg.norm(coords[mask] - centroid, axis=1)
            closest_idx = np.where(mask)[0][np.argmin(dists)]
            facility_coords[m] = coords[closest_idx]
        else:
            # Empty cluster - place at a random customer location
            facility_coords[m] = coords[np.random.randint(len(coords))]
    return facility_coords


def compute_service_distance(individual, coords, facility_coords):
    return np.linalg.norm(coords - facility_coords[individual], axis=1).sum()


def compute_dispersion(facility_coords):
    p = len(facility_coords)
    total = 0.0
    for i in range(p):
        diff = facility_coords[i + 1:] - facility_coords[i]
        total += np.sqrt((diff ** 2).sum(axis=1)).sum()
    return total


def compute_capacity_penalty(individual, demands, capacity, num_medians, penalty_weight):
    penalty = 0.0
    for m in range(num_medians):
        excess = demands[individual == m].sum() - capacity
        if excess > 0:
            penalty += excess
    return penalty_weight * penalty


def is_feasible(individual, demands, capacity, num_medians):
    for m in range(num_medians):
        if demands[individual == m].sum() > capacity:
            return False
    return True


def repair_capacity(individual, coords, demands, capacity, num_medians):
    child = individual.copy()
    loads = np.array([demands[child == m].sum() for m in range(num_medians)])

    for m in range(num_medians):
        if loads[m] <= capacity:
            continue

        max_passes = num_medians * len(individual)  # safety cap
        for _ in range(max_passes):
            if loads[m] <= capacity:
                break

            fac_coords = compute_facility_locations(child, coords, num_medians)
            assigned = np.where(child == m)[0]
            if len(assigned) == 0:
                break

            dists_to_fac = np.linalg.norm(coords[assigned] - fac_coords[m], axis=1)
            order = np.argsort(-dists_to_fac)  # farthest first

            moved_any = False
            for idx in order:
                if loads[m] <= capacity:
                    break
                cust = assigned[idx]
                cust_demand = demands[cust]

                # Find nearest under-capacity facility for this customer
                best_target = -1
                best_dist = np.inf
                for t in range(num_medians):
                    if t == m:
                        continue
                    if loads[t] + cust_demand <= capacity:
                        d = np.linalg.norm(coords[cust] - fac_coords[t])
                        if d < best_dist:
                            best_dist = d
                            best_target = t

                if best_target >= 0:
                    child[cust] = best_target
                    loads[m] -= cust_demand
                    loads[best_target] += cust_demand
                    moved_any = True
                    break  # restart with updated assignments

            if not moved_any:
                break

    return child
def evaluate_population(pop, instance, penalty_weight):
    n = len(pop)
    objectives = np.empty((n, 2))
    raw_dists = np.empty(n)
    for i in range(n):
        ind = pop[i]
        fac = compute_facility_locations(ind, instance.coords,
                                         instance.num_medians)
        dist = compute_service_distance(ind, instance.coords, fac)
        pen = compute_capacity_penalty(ind, instance.demands,
                                       instance.capacity,
                                       instance.num_medians,
                                       penalty_weight)
        disp = compute_dispersion(fac)
        raw_dists[i] = dist
        objectives[i, 0] = dist + pen     # minimise (internal, penalized)
        objectives[i, 1] = -disp          # minimise (maximise dispersion)
    return objectives, raw_dists

# %% [markdown]
# #### Pareto utilities
# 
# **Pareto dominance** (Algorithm 98): solution *a* dominates *b* if *a* is at least as good in all objectives and strictly better in at least one. The set of all non-dominated solutions forms the **Pareto front** (Algorithm 100).

# %%
def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))

def compute_pareto_front(objectives):
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(objectives[j], objectives[i]):
                is_dominated[i] = True
                break
    return ~is_dominated

# %% [markdown]
# #### SPEA2 fitness assignment
# 
# Fitness assignment follows Algorithm 105:
# 
# 1. **Strength** of individual *i* = number of individuals in the population that *i* dominates.
# 2. **Wimpiness** (raw fitness) of *i* = sum of the strengths of all individuals that dominate *i*. Non-dominated individuals have wimpiness 0.
# 3. **k-th nearest neighbour distance** (Algorithm 105) in normalised objective space, with *k* = ⌈√n⌉, provides a density estimate for diversity preservation.
# 4. **Unfitness** combines both: `Unfitness(i) = Wimpiness(i) + 1/(2 + kth_dist(i))`. Lower unfitness is better. Final fitness is `1/(1 + Unfitness)`.

# %%
def compute_strength(objectives):
    n = len(objectives)
    strength = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(objectives[i], objectives[j]):
                strength[i] += 1
    return strength


def compute_wimpiness(objectives, strength):
    n = len(objectives)
    wimpiness = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(objectives[j], objectives[i]):
                wimpiness[i] += strength[j]
    return wimpiness


def compute_kth_nearest_distance(objectives, k):
    n = len(objectives)
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    normed = (objectives - obj_min) / obj_range

    # Pairwise Euclidean distance matrix
    diff = normed[:, np.newaxis, :] - normed[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

    kth_dist = np.empty(n)
    for i in range(n):
        sorted_d = np.sort(dist_matrix[i])
        # sorted_d[0] == 0 (self), so k-th nearest is at index k
        kth_dist[i] = sorted_d[min(k, n - 1)]
    return kth_dist


def compute_fitness(objectives):
    n = len(objectives)
    k = max(1, int(np.ceil(np.sqrt(n))))
    strength = compute_strength(objectives)
    wimpiness = compute_wimpiness(objectives, strength)
    kth_dist = compute_kth_nearest_distance(objectives, k)
    unfitness = wimpiness + 1.0 / (2.0 + kth_dist)
    return 1.0 / (1.0 + unfitness)

# %% [markdown]
# #### SPEA2 archive construction
# 
# Archive construction follows Algorithm 106:
# 
# 1. Start with the Pareto non-dominated front.
# 2. If the front is **smaller** than the target archive size, fill remaining slots with the fittest individuals from the dominated set.
# 3. If the front is **larger** than the archive size, iteratively remove the individual with the smallest distance to its nearest neighbour (k=1), breaking ties with k=2, k=3, etc. This truncation preserves diversity along the front.

# %%
def construct_spea2_archive(pop, objectives, raw_dists, fitness, archive_size):
    # Step 1 - start with non-dominated front
    front_mask = compute_pareto_front(objectives)
    archive_idx = list(np.where(front_mask)[0])
    rest_idx = list(np.where(~front_mask)[0])

    # Step 2 - if too small, fill with fittest remaining
    if len(archive_idx) < archive_size and rest_idx:
        rest_fit = fitness[np.array(rest_idx)]
        order = np.argsort(-rest_fit)
        needed = archive_size - len(archive_idx)
        archive_idx.extend(rest_idx[i] for i in order[:needed])

    # Step 3 - if too large, iteratively remove closest individual
    # Always protect the solution with the best (lowest) service distance
    # so elitism prevents its loss from the archive.
    pinned = int(np.argmin(objectives[archive_idx, 0]))
    pinned_global = archive_idx[pinned]

    while len(archive_idx) > archive_size:
        arch_objs = objectives[archive_idx]
        obj_min = arch_objs.min(axis=0)
        obj_max = arch_objs.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
        normed = (arch_objs - obj_min) / obj_range

        m = len(archive_idx)
        diff = normed[:, np.newaxis, :] - normed[np.newaxis, :, :]
        dm = np.sqrt((diff ** 2).sum(axis=2))
        sorted_dm = np.sort(dm, axis=1)          # (m, m)

        # Find individual with smallest kth-closest (k=1, break ties k=2...)
        to_remove = 0
        for kk in range(1, m):
            col = sorted_dm[:, kk]
            min_val = col.min()
            candidates = np.where(np.isclose(col, min_val))[0]
            # Skip the pinned solution - it must not be removed
            unpinned = [c for c in candidates if archive_idx[c] != pinned_global]
            if unpinned:
                if len(unpinned) == 1:
                    to_remove = unpinned[0]
                    break
                candidates = np.array(unpinned)
            elif len(candidates) == 1:
                to_remove = candidates[0]
                break
        else:
            to_remove = candidates[0]

        archive_idx.pop(to_remove)

    idx = np.array(archive_idx)
    return pop[idx], objectives[idx], raw_dists[idx]


# %% [markdown]
# #### Genetic operators
# 
# - **Initialisation**: random assignment of each customer to one of *p* medians, with a repair step to ensure every median ID appears at least once (no empty clusters).
# - **Parent selection**: binary tournament selection based on SPEA2 fitness (Algorithm 107, Luke).
# - **Crossover**: uniform crossover — each gene (customer assignment) is randomly taken from either parent with probability 0.5. Applied with a configurable crossover rate.
# - **Mutation**: random resetting — each gene is independently re-randomised to a different median with a configurable per-gene mutation rate.
# - **Repair**: after crossover and mutation, the Lamarckian repair operator is applied to fix any capacity violations before the offspring enters the population.

# %%
def initialize_population(pop_size, num_customers, num_medians, rng):
    pop = rng.integers(0, num_medians, size=(pop_size, num_customers))
    for i in range(pop_size):
        used = set(pop[i])
        for m in range(num_medians):
            if m not in used:
                pop[i, rng.integers(num_customers)] = m
    return pop


def tournament_selection(fitness, rng, tournament_size=2):
    candidates = rng.integers(0, len(fitness), size=tournament_size)
    return candidates[np.argmax(fitness[candidates])]


def uniform_crossover(p1, p2, rng, rate=0.8):
    if rng.random() > rate:
        return p1.copy(), p2.copy()
    mask = rng.random(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def random_resetting_mutation(individual, num_medians, rng, rate):
    child = individual.copy()
    mask = rng.random(len(child)) < rate
    child[mask] = rng.integers(0, num_medians, size=mask.sum())
    return child


def breed(archive, archive_fitness, pop_size, num_medians, rng, crossover_rate, mutation_rate, instance=None):
    n_cust = archive.shape[1]
    children = np.empty((pop_size, n_cust), dtype=int)
    idx = 0
    while idx < pop_size:
        i1 = tournament_selection(archive_fitness, rng)
        i2 = tournament_selection(archive_fitness, rng)
        c1, c2 = uniform_crossover(archive[i1], archive[i2], rng, crossover_rate)
        c1 = random_resetting_mutation(c1, num_medians, rng, mutation_rate)
        c2 = random_resetting_mutation(c2, num_medians, rng, mutation_rate)
        # Repair capacity violations
        if instance is not None:
            c1 = repair_capacity(c1, instance.coords, instance.demands, instance.capacity, num_medians)
            c2 = repair_capacity(c2, instance.coords, instance.demands, instance.capacity, num_medians)
        children[idx] = c1
        idx += 1
        if idx < pop_size:
            children[idx] = c2
            idx += 1
    return children

# %% [markdown]
# #### SPEA2 main loop
# 
# The main evolutionary loop follows Algorithm 107:
# 
# 1. Initialise population *P* randomly.
# 2. Each generation: evaluate *P*, merge with archive *A* → combined pool.
# 3. Compute SPEA2 fitness on the combined pool (strength, wimpiness, k-th distance).
# 4. Construct new archive from the combined pool (Algorithm 106).
# 5. Breed new population from the archive via tournament selection, crossover, mutation, and repair.
# 6. Repeat until a stopping criterion triggers.
# 
# The final output filters the Pareto front of the archive to **feasible solutions only** — any individual violating capacity constraints is excluded.

# %%
inst = load_instance(1)
# inst = load_instance_from_string("""
#  0 1000 
#  100 10 120
#  1 2 62 5
#  2 80 25 8
#  3 36 88 2
#  4 57 23 8
#  5 33 17 18
#  6 76 43 20
#  7 77 85 17
#  8 94 6 9
#  9 89 11 1
#  10 59 72 19
#  11 39 82 10
#  12 87 24 20
#  13 44 76 4
#  14 2 83 5
#  15 19 43 5
#  16 5 27 14
#  17 58 72 19
#  18 14 50 19
#  19 43 18 4
#  20 87 7 16
#  21 11 56 14
#  22 31 16 13
#  23 51 94 18
#  24 55 13 14
#  25 84 57 13
#  26 12 2 13
#  27 53 33 14
#  28 53 10 7
#  29 33 32 15
#  30 69 67 3
#  31 43 5 6
#  32 10 75 10
#  33 8 26 20
#  34 3 1 13
#  35 96 22 7
#  36 6 48 3
#  37 59 22 18
#  38 66 69 13
#  39 22 50 14
#  40 75 21 7
#  41 4 81 8
#  42 41 97 16
#  43 92 34 18
#  44 12 64 9
#  45 60 84 8
#  46 35 100 12
#  47 38 2 19
#  48 9 9 13
#  49 54 59 19
#  50 1 58 12
#  51 15 66 5
#  52 3 66 2
#  53 94 10 7
#  54 68 30 4
#  55 35 27 12
#  56 46 86 13
#  57 11 29 5
#  58 100 17 19
#  59 70 55 7
#  60 94 73 1
#  61 75 16 12
#  62 62 65 17
#  63 21 80 4
#  64 11 31 3
#  65 70 81 19
#  66 15 13 10
#  67 58 69 6
#  68 100 62 10
#  69 47 42 11
#  70 28 87 19
#  71 31 96 12
#  72 41 2 2
#  73 37 22 18
#  74 1 34 6
#  75 44 9 12
#  76 58 7 17
#  77 44 37 20
#  78 18 3 16
#  79 3 87 16
#  80 1 77 7
#  81 95 72 16
#  82 10 66 15
#  83 28 94 16
#  84 45 26 19
#  85 68 62 13
#  86 41 86 15
#  87 41 30 11
#  88 8 99 18
#  89 78 55 4
#  90 10 30 12
#  91 13 62 16
#  92 95 24 19
#  93 50 77 1
#  94 78 20 3
#  95 87 67 10
#  96 45 19 13
#  97 36 90 6
#  98 4 31 7
#  99 70 56 1
#  100 18 34 5
# """, 0)

if inst.num_customers <= 50:
    # instance 1 results
    # [ 11/100] pop=285, arch=49, penalty=1.27, cx=0.111, mut=0.083, seed=102, mult=0.301 ... dist=728.4  *** new best (732.2 -> 728.4) ***
    # [  7/500] pop=257, arch=41, penalty=1.84, cx=0.092, mut=0.058, seed=62, mult=0.329 ... dist=728.4  *** new best (732.2 -> 728.4) ***
    best_pop_size = 193
    best_archive_size = 51
    best_penalty = 1.62
    best_crossover_rate = 0.089
    best_mutation_rate = 0.076
    best_seed = 62
else:
    # Instance 15 results
    # [ 19/100] pop=288, arch=35, penalty=1.84, cx=0.059, mut=0.042, seed=90, mult=0.275 ... dist=1142.1 time=66.5633  *** new best (1153.0 -> 1142.1) ***

    # Hackathon:
    # [  3/500] pop=191, arch=27, penalty=1.89, cx=0.088, mut=0.023, seed=88, mult=0.332 ... dist=1053.7 time=41.8276  *** new best (1107.1 -> 1053.7) ***
    best_pop_size = 191
    best_archive_size = 27
    best_penalty = 1.89
    best_crossover_rate = 0.088
    best_mutation_rate = 0.023
    best_seed = 88

def spea2(instance, pop_size=best_pop_size, archive_size=best_archive_size,
          stopping_criteria=None,
          penalty_weight=best_penalty, crossover_rate=best_crossover_rate, mutation_rate=best_mutation_rate,
          seed=best_seed, verbose=True):
    if stopping_criteria is None:
        stopping_criteria = [TimeLimit(60 * 5), MinImprovement(50, min_pct=0.02)]

    rng = np.random.default_rng(seed)
    n_cust = instance.num_customers
    n_med = instance.num_medians
    if mutation_rate is None:
        mutation_rate = 1.0 / n_cust

    # Initialise stopping criteria
    for criterion in stopping_criteria:
        criterion.reset()

    # Line 3 - initialise population
    P = initialize_population(pop_size, n_cust, n_med, rng)

    # Line 4 - empty archive
    A = np.empty((0, n_cust), dtype=int)
    A_obj = np.empty((0, 2))
    A_raw = np.empty(0)

    history = {
        "best_service_dist": [],   # true service distance, no penalty
        "best_dispersion": [],
        "front_sizes": [],
    }

    gen = 0
    while True:
        # Line 6 - evaluate new population
        P_obj, P_raw = evaluate_population(P, instance, penalty_weight)

        # Line 7 - P ← P ∪ A
        if len(A) > 0:
            combined = np.vstack([P, A])
            combined_obj = np.vstack([P_obj, A_obj])
            combined_raw = np.concatenate([P_raw, A_raw])
        else:
            combined = P
            combined_obj = P_obj
            combined_raw = P_raw

        # Compute SPEA2 fitness on combined pool
        combined_fit = compute_fitness(combined_obj)

        # Line 8 - best Pareto front (for tracking)
        front_mask = compute_pareto_front(combined_obj)

        # Line 9 - construct archive
        A, A_obj, A_raw = construct_spea2_archive(
            combined, combined_obj, combined_raw, combined_fit, archive_size
        )

        # Record history - use true service distance (no penalty)
        front_raw = combined_raw[front_mask]
        front_obj = combined_obj[front_mask]
        history["best_service_dist"].append(front_raw.min())
        history["best_dispersion"].append((-front_obj[:, 1]).max())
        history["front_sizes"].append(front_mask.sum())

        if verbose and ((gen + 1) % 50 == 0 or gen == 0):
            print(
                f"Gen {gen+1:4d}  |  front size {front_mask.sum():3d}  |  "
                f"best dist {front_raw.min():.1f}  |  "
                f"best disp {(-front_obj[:, 1]).max():.1f}"
            )

        # Check stopping criteria
        triggered = [c for c in stopping_criteria if c.should_stop(gen, history)]
        if triggered:
            if verbose:
                print(f"Stopped at gen {gen + 1} - triggered: {triggered}")
            break

        # Line 10 - breed new population from archive (with repair)
        A_fit = compute_fitness(A_obj) if len(A) > 1 else np.ones(len(A))
        P = breed(A, A_fit, pop_size, n_med, rng, crossover_rate,
                  mutation_rate, instance=instance)
        gen += 1

    # Final front from archive - filter to feasible solutions only
    final_mask = compute_pareto_front(A_obj)
    front_pop = A[final_mask]
    front_obj = A_obj[final_mask]
    front_raw = A_raw[final_mask]

    feasible_mask = np.array([
        is_feasible(ind, instance.demands, instance.capacity, n_med)
        for ind in front_pop
    ])

    if feasible_mask.any():
        front_pop = front_pop[feasible_mask]
        front_obj = front_obj[feasible_mask]
        front_raw = front_raw[feasible_mask]
    else:
        if verbose:
            print("WARNING: no feasible solution found on the Pareto front, returning all.")

    return {
        "archive": A,
        "archive_objectives": A_obj,
        "archive_raw_dists": A_raw,
        "front": front_pop,
        "front_objectives": front_obj,
        "front_raw_dists": front_raw,
        "history": history,
        "generations_run": gen + 1,
    }


print("SPEA2 implementation loaded.")

# %% [markdown]
# #### Fine-tuning

# %%
if FINE_TUNE_SPEA > 0:
    multiplier = 0.333
    min_multiplier = 0.01
    max_i = FINE_TUNE_SPEA
    multiplier_step = (multiplier - min_multiplier) / max_i

    best_result = float(inst.best_known * 2.)
    best_time = 99999
    best_run_result = None

    for i in range(max_i):
        pop_low = max(2, int(best_pop_size * (1.0 - multiplier)))
        pop_high = max(pop_low + 1, int(best_pop_size * (1.0 + multiplier)) + 1)
        pop_size = np.random.randint(pop_low, pop_high)

        arch_low = max(1, int(best_archive_size * (1.0 - multiplier)))
        arch_high = min(pop_size, int(best_archive_size * (1.0 + multiplier)))
        archive_size = np.random.randint(arch_low, max(arch_low + 1, arch_high + 1))

        penalty = np.random.uniform(
            max(0.0, best_penalty * (1.0 - multiplier)),
            max(0.0, best_penalty * (1.0 + multiplier))
        )
        crossover_rate = np.random.uniform(
            max(0.0, best_crossover_rate * (1.0 - multiplier)),
            min(1.0, best_crossover_rate * (1.0 + multiplier))
        )
        mutation_rate = np.random.uniform(
            max(0.0, best_mutation_rate * (1.0 - multiplier)),
            min(1.0, best_mutation_rate * (1.0 + multiplier))
        )

        seed_low = max(0, int(best_seed * (1.0 - multiplier)))
        seed_high = min(np.iinfo(np.uint32).max, int(best_seed * (1.0 + multiplier)))
        seed = np.random.randint(seed_low, max(seed_low + 1, seed_high + 1))

        print(
            f"[{i+1:3d}/{max_i}] pop={pop_size}, arch={archive_size}, "
            f"penalty={penalty:.2f}, cx={crossover_rate:.3f}, "
            f"mut={mutation_rate:.3f}, seed={seed}, mult={multiplier:.3f}",
            end=" ... "
        )

        start_time = time.monotonic()
        result = spea2(
            inst,
            pop_size=pop_size,
            archive_size=archive_size,
            penalty_weight=penalty,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            seed=seed,
            stopping_criteria=[
                TimeLimit(240),
                # MinImprovement(50, min_pct=0.03),
                MinImprovement(50, min_pct=0.01),
            ],
            verbose=False,
        )
        local_time = time.monotonic() - start_time

        local_best = result["front_raw_dists"].min()
        print(f"dist={local_best:.1f} time={local_time:.4f}", end="")
        is_best = local_best < best_result

        if is_best or (local_best == best_result and local_time < best_time):
            if is_best:
                print(f"  *** new best ({best_result:.1f} -> {local_best:.1f}) ***", end="")
            else:
                print(f"  *** time improvement ({best_time:.4f} -> {local_time:.4f}) ***", end="")
            best_idx = np.argmin(result["front_raw_dists"])
            best_ind = result["front"][best_idx]
            best_fac = compute_facility_locations(best_ind, inst.coords, inst.num_medians)
            
            best_fac_idx = [int(np.where((inst.coords == f).all(axis=1))[0][0]) for f in best_fac]
            print(f" idx_to_facilities = {best_ind.tolist()} ; facility_customer_id = {[idx + 1 for idx in best_fac_idx]}")
            best_result = local_best
            best_time = local_time
            best_pop_size = pop_size
            best_archive_size = archive_size
            best_penalty = penalty
            best_crossover_rate = crossover_rate
            best_mutation_rate = mutation_rate
            best_seed = seed
            best_run_result = result

            # Plot the new best solution

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"Fine-tune iter {i+1} - new best dist = {local_best:.1f} "
                f"(known best = {inst.best_known})",
                fontsize=12
            )

            # Left: convergence of this run
            gens = range(1, len(best_run_result["history"]["best_service_dist"]) + 1)
            axes[0].plot(gens, best_run_result["history"]["best_service_dist"], color="tab:blue", label=f"Our result = {local_best:.0f}")
            # axes[0].axhline(y=inst.best_known, color="tab:red", linestyle="--",
            #                 linewidth=1.2, label=f"Best known = {inst.best_known}")
            axes[0].set_xlabel("Generation")
            axes[0].set_ylabel("Best Service Distance")
            axes[0].set_title("Convergence")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Right: facility map
            cmap = plt.colormaps.get_cmap("tab10").resampled(inst.num_medians)
            for m in range(inst.num_medians):
                mask = best_ind == m
                axes[1].scatter(
                    inst.coords[mask, 0], inst.coords[mask, 1],
                    color=cmap(m), alpha=0.6, s=40, label=f"Cluster {m}"
                )
                for ci in np.where(mask)[0]:
                    axes[1].plot(
                        [inst.coords[ci, 0], best_fac[m, 0]],
                        [inst.coords[ci, 1], best_fac[m, 1]],
                        color=cmap(m), alpha=0.15, linewidth=0.8
                    )
            axes[1].scatter(
                best_fac[:, 0], best_fac[:, 1],
                marker="o", s=125, color="red", facecolors="none", zorder=10, label="Facility"
            )
            axes[1].set_title("Best Facility Map")
            axes[1].legend(loc="upper right", fontsize=8)
            axes[1].grid(True, alpha=0.2)
            axes[1].set_aspect("equal")

            plt.tight_layout()
            plt.show()
        else:
            print()

        multiplier = max(min_multiplier, multiplier - multiplier_step)

    print("\nBest tuning result:")
    print(f"  best_result      = {best_result:.4f}")
    print(f"  pop_size         = {best_pop_size}")
    print(f"  archive_size     = {best_archive_size}")
    print(f"  penalty          = {best_penalty:.4f}")
    print(f"  crossover_rate   = {best_crossover_rate:.4f}")
    print(f"  mutation_rate    = {best_mutation_rate:.4f}")
    print(f"  seed             = {best_seed}")

# %%
result = spea2(
    inst,
    pop_size=best_pop_size,
    archive_size=best_archive_size,
    stopping_criteria=[
        TimeLimit(60 * 5),
        MinImprovement(50, min_pct=0.),
        TargetObjective(inst.best_known),
    ],
    penalty_weight=best_penalty,
    seed=best_seed,
)

our_distance = result['front_raw_dists'].min()
print(f"\nGenerations run:         {result['generations_run']}")
print(f"Final Pareto front size: {len(result['front'])}")
print(f"Best service distance:   {our_distance:.1f}")
print(f"Best dispersion:         {(-result['front_objectives'][:, 1]).max():.1f}")
print(f"Known best (single-obj): {inst.best_known}")

# %% [markdown]
# ### Visualisations

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

gens = range(1, len(result["history"]["best_service_dist"]) + 1)

# Service distance convergence
axes[0].plot(gens, result["history"]["best_service_dist"], color="tab:blue", label=f"Our result = {our_distance:.0f}")
axes[0].axhline(y=inst.best_known, color="tab:red", linestyle="--", linewidth=1.2,
                label=f"Best known = {inst.best_known}")
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Best Service Distance")
axes[0].set_title("Service Distance")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dispersion convergence
axes[1].plot(gens, result["history"]["best_dispersion"], color="tab:green")
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Best Dispersion")
axes[1].set_title("Facility Dispersion")
axes[1].grid(True, alpha=0.3)

# Pareto front size
axes[2].plot(gens, result["history"]["front_sizes"], color="tab:orange")
axes[2].set_xlabel("Generation")
axes[2].set_ylabel("Front Size")
axes[2].set_title("Pareto Front Size")
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"SPEA2 Generation Improvement Process - Instance {inst.instance_id}", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# - Left: the best service distance found on the Pareto front at each generation, compared against the known best single-objective solution (red dashed line). A decreasing curve indicates the algorithm is converging toward better solutions. 
# - Center: the best facility dispersion over generations — higher values mean facilities are more geographically spread out. 
# - Right: the size of the Pareto non-dominated front in the population.

# %%
# Facility map - best solution (lowest service distance) from the Pareto front
best_idx = np.argmin(result["front_raw_dists"])
best_ind = result["front"][best_idx]
best_fac = compute_facility_locations(best_ind, inst.coords, inst.num_medians)

fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.colormaps.get_cmap("tab10").resampled(inst.num_medians)

for m in range(inst.num_medians):
    mask = best_ind == m
    ax.scatter(
        inst.coords[mask, 0], inst.coords[mask, 1],
        color=cmap(m), alpha=0.6, s=40, label=f"Cluster {m}"
    )
    # Draw lines from customers to their facility
    for ci in np.where(mask)[0]:
        ax.plot(
            [inst.coords[ci, 0], best_fac[m, 0]],
            [inst.coords[ci, 1], best_fac[m, 1]],
            color=cmap(m), alpha=0.15, linewidth=0.8
        )

# Facility locations as stars
ax.scatter(
    best_fac[:, 0], best_fac[:, 1],
    marker="o", s=125, color="red", facecolors='none', zorder=10, label="Facility"
)

ax.set_title(f"Best Solution Facility Map - Instance {inst.instance_id}")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# Print capacity utilisation
print("Capacity utilisation:")
for m in range(inst.num_medians):
    demand = inst.demands[best_ind == m].sum()
    print(f"  Median {m}: demand={demand:3d} / capacity={inst.capacity}  "
          f"{'VIOLATED' if demand > inst.capacity else 'OK'}")

# %% [markdown]
# ## Non-dominated Sorting Genetic Algorithm (NSGA)
# 
# NSGA2 - Non-dominated Sorting Genetic Algorithm 2
# 
# Implementation follows Algorithms 103-104 from "Essentials of Metaheuristics" (Sean Luke), adapted for the capacitated p-median problem.
# 
# Representation (n-Queens style, per lecture slide 17):
#   An integer array of length num_customers where
#   individual[i] = median ID (0 to p-1) that customer i is assigned to.
#   Facility locations are derived as the customer nearest to each cluster's centroid.
# 
# Objectives (both treated as minimisation):
# 1. Service distance + capacity penalty (minimize, internal use only)
# 2. Negative facility dispersion (minimize == maximize dispersion)

# %% [markdown]
# #### Front Rank Assignment by Non-Dominated Sorting
# 
# Algorithm 101

# %%
def front_rank_assignment(population, objectives):
    population_prime = population.copy() # individuals will be gradually removed from this copy
    ranks = np.empty(len(population), dtype=int) # initially empty ordered vector of pareto front ranks
    i = 1

    # We must track indices of "population_prime" w.r.t. the original population,
    # otherwise we cannot write ranks back correctly.
    remaining_idx = np.arange(len(population), dtype=int)

    # Basic sanity checks
    assert objectives.shape[0] == len(population), "objectives must align with population length"
    assert objectives.ndim == 2 and objectives.shape[1] == 2, "objectives must be shape (n, 2)"

    while population_prime.shape[0] > 0:
        # compute_pareto_front expects an objective matrix; we pass only remaining rows
        front_mask_local = compute_pareto_front(objectives[remaining_idx])

        # front_mask_local is a boolean mask over remaining_idx
        assert front_mask_local.shape[0] == remaining_idx.shape[0], "Pareto mask length mismatch"

        # Map local mask back to global indices, assign rank i
        front_global_idx = remaining_idx[front_mask_local]
        ranks[front_global_idx] = i

        # Remove this front from the working copy
        keep_mask_local = ~front_mask_local
        population_prime = population_prime[keep_mask_local]
        remaining_idx = remaining_idx[keep_mask_local]

        i += 1

    # Ensure all ranks were assigned
    assert np.all(ranks >= 1), "Some individuals did not receive a front rank"
    return ranks

# %% [markdown]
# Algorithm 102

# %%
def compute_sparsities(front_rank, objectives):
    # for each individual in the front, sparsity is initially assigned 0
    # NOTE: front_rank is expected to be the *indices* of individuals in the front
    front_idx = np.asarray(front_rank, dtype=int)

    sparsity = np.zeros(front_idx.shape[0], dtype=float)

    # Sanity checks
    assert objectives.ndim == 2 and objectives.shape[1] == 2, "objectives must be shape (n, 2)"
    assert np.all((front_idx >= 0) & (front_idx < objectives.shape[0])), "front indices out of range"

    # for each objective, sort the front by that objective and compute normalised distances
    for m in range(objectives.shape[1]):
        # sort front by objective m
        order = np.argsort(objectives[front_idx, m])
        sorted_front_idx = front_idx[order]

        # assign infinite sparsity to boundary solutions (positions in sorted order)
        # We store sparsity in the *front-local* array, so we need a mapping:
        # global_idx -> position in front_idx
        pos_in_front = {g: j for j, g in enumerate(front_idx)}

        sparsity[pos_in_front[sorted_front_idx[0]]] = np.inf
        sparsity[pos_in_front[sorted_front_idx[-1]]] = np.inf

        denom = objectives[sorted_front_idx[-1], m] - objectives[sorted_front_idx[0], m]
        if denom == 0:
            # all equal on this objective; contribute nothing
            continue

        # interior points (need two neighbors)
        for j in range(1, len(sorted_front_idx) - 1):
            g_mid = sorted_front_idx[j]
            g_prev = sorted_front_idx[j - 1]
            g_next = sorted_front_idx[j + 1]
            sparsity[pos_in_front[g_mid]] += (
                objectives[g_next, m] - objectives[g_prev, m]
            ) / denom

    return sparsity

# %% [markdown]
# Algorithm 103

# %%
def breed_nsga(population_with_pareto_front_ranks, tournament_size = 1):
    # for i from 2 to t
    #
    # Expected input element structure:
    #   (individual, pareto_rank, sparsity)
    #
    # Returns:
    #   the selected individual (same type as the stored "individual")
    #
    # Sanity checks / fail-fast
    assert len(population_with_pareto_front_ranks) > 0, "selection pool is empty"
    first = population_with_pareto_front_ranks[0]
    assert isinstance(first, (list, tuple)) and len(first) >= 3, \
        "population_with_pareto_front_ranks must contain (individual, rank, sparsity)"

    # Use local "best" variable to avoid mutating the input list accidentally
    best = population_with_pareto_front_ranks[np.random.randint(len(population_with_pareto_front_ranks))]

    for i in range(2, tournament_size + 1):
        # individual picked at random from population with replacement
        next_ind = population_with_pareto_front_ranks[np.random.randint(len(population_with_pareto_front_ranks))]

        # if the next individual has a better pareto front rank than the current best, it becomes the new best
        if next_ind[1] < best[1]:
            best = next_ind

        # if the next individual has the same pareto front rank as the current best, the one with higher sparsity becomes the new best
        elif next_ind[1] == best[1] and next_ind[2] > best[2]:
            best = next_ind

    return best[0]

# %% [markdown]
# #### NSGA2 main loop
# Algorithm 104

# %%
def nsga2(instance, pop_size=best_pop_size, archive_size=best_archive_size,
          stopping_criteria=None,
          penalty_weight=best_penalty, mutation_rate=best_mutation_rate,
          seed=best_seed, verbose=True):
    if stopping_criteria is None:
        stopping_criteria = [MaxGenerations(200)]

    rng = np.random.default_rng(seed)
    n_cust = instance.num_customers
    n_med = instance.num_medians
    if mutation_rate is None:
        mutation_rate = 1.0 / n_cust

    # Initialise stopping criteria
    for criterion in stopping_criteria:
        criterion.reset()

    # Line 3 — initialise population
    P = initialize_population(pop_size, n_cust, n_med, rng)

    # Line 4 — empty archive
    A = np.empty((0, n_cust), dtype=int)
    A_obj = np.empty((0, 2))
    A_raw = np.empty(0)

    history = {
        "best_service_dist": [],   # true service distance, no penalty
        "best_dispersion": [],
        "front_sizes": [],
    }

    gen = 0
    while True:
        # Line 6 — evaluate new population
        P_obj, P_raw = evaluate_population(P, instance, penalty_weight)

        # Line 7 — P ← P ∪ A
        if len(A) > 0:
            combined = np.vstack([P, A])
            combined_obj = np.vstack([P_obj, A_obj])
            combined_raw = np.concatenate([P_raw, A_raw])
        else:
            combined = P
            combined_obj = P_obj
            combined_raw = P_raw

        # Line 8 — best Pareto front (for tracking)
        front_mask = compute_pareto_front(combined_obj)

        # Line 9 - compute front ranks
        ranks = front_rank_assignment(combined, combined_obj)

        # Line 10 - empty archive
        archive_idx = []

        # Line 11 - for each front rank belonging to the front ranks computed in line 9
        for rank in sorted(np.unique(ranks)):
            # Line 12 - compute sparsities of individuals in the selected front rank
            front_rank_mask = (ranks == rank)
            front_idx = np.where(front_rank_mask)[0]              # indices into combined

            # compute_sparsities expects indices into the same objective matrix
            sparsities = compute_sparsities(front_idx, combined_obj)

            # Line 13 - if ||A|| + ||selected front rank|| >= desired archive size
            if len(archive_idx) + len(front_idx) >= archive_size:
                # Line 14 - A ← A ∪ the most sparse (desired archive size - ||A||) individuals from the selected front rank
                n_to_add = archive_size - len(archive_idx)

                # sparsities is front-local (aligned with front_idx), so sort its positions
                sorted_pos = np.argsort(sparsities)[::-1][:n_to_add]
                archive_idx.extend(front_idx[sorted_pos].tolist())
                break
            else:
                # Line 17 - A ← A ∪ selected front rank
                archive_idx.extend(front_idx.tolist())

        # Materialize archive arrays (A, A_obj, A_raw) from the combined pool
        A = combined[np.array(archive_idx, dtype=int)]
        A_obj = combined_obj[np.array(archive_idx, dtype=int)]
        A_raw = combined_raw[np.array(archive_idx, dtype=int)]

        # Record history — use true service distance (no penalty)
        front_raw = combined_raw[front_mask]
        front_obj = combined_obj[front_mask]
        history["best_service_dist"].append(front_raw.min())
        history["best_dispersion"].append((-front_obj[:, 1]).max())
        history["front_sizes"].append(front_mask.sum())

        if verbose and ((gen + 1) % 50 == 0 or gen == 0):
            print(
                f"Gen {gen+1:4d}  |  front size {front_mask.sum():3d}  |  "
                f"best dist {front_raw.min():.1f}  |  "
                f"best disp {(-front_obj[:, 1]).max():.1f}"
            )

        # Check stopping criteria
        triggered = [c for c in stopping_criteria if c.should_stop(gen, history)]
        if triggered:
            if verbose:
                print(f"Stopped at gen {gen + 1} — triggered: {triggered}")
            break

        # Line 18 — breed new population from archive (with repair)
        #
        # Build selection tuples: (individual, rank, sparsity)
        # We reuse ranks from the combined pool by mapping A rows back via archive_idx.
        A_ranks = ranks[np.array(archive_idx, dtype=int)]

        # Compute sparsities inside each rank for the archive pool
        A_sparsity = np.zeros(len(A), dtype=float)
        for r in sorted(np.unique(A_ranks)):
            mask = (A_ranks == r)
            idx_local = np.where(mask)[0]
            # compute_sparsities wants indices into an objective matrix; here use A_obj
            sp = compute_sparsities(idx_local, A_obj)
            A_sparsity[idx_local] = sp

        selection_pool = [(A[i], int(A_ranks[i]), float(A_sparsity[i])) for i in range(len(A))]

        # Fill the next population by repeated tournament selection (same logic, no extra operators added here)
        P = np.empty((pop_size, n_cust), dtype=int)
        for i in range(pop_size):
            P[i] = breed_nsga(selection_pool, tournament_size=2)

        gen += 1

    # Final front from archive — filter to feasible solutions only
    final_mask = compute_pareto_front(A_obj)
    front_pop = A[final_mask]
    front_obj = A_obj[final_mask]
    front_raw = A_raw[final_mask]

    feasible_mask = np.array([
        is_feasible(ind, instance.demands, instance.capacity, n_med)
        for ind in front_pop
    ])

    if feasible_mask.any():
        front_pop = front_pop[feasible_mask]
        front_obj = front_obj[feasible_mask]
        front_raw = front_raw[feasible_mask]
    else:
        if verbose:
            print("WARNING: no feasible solution found on the Pareto front, returning all.")

    # Final sanity checks to avoid returning inconsistent shapes
    assert A.shape[0] == A_obj.shape[0] == A_raw.shape[0], "Archive arrays are misaligned"
    assert front_pop.shape[0] == front_obj.shape[0] == front_raw.shape[0], "Front arrays are misaligned"

    return {
        "archive": A,
        "archive_objectives": A_obj,
        "archive_raw_dists": A_raw,
        "front": front_pop,
        "front_objectives": front_obj,
        "front_raw_dists": front_raw,
        "history": history,
        "generations_run": gen + 1,
    }

# %%
# Run NSGA2 on instance 1 (50 customers, 5 medians)
result = nsga2(
    inst,
    pop_size=best_pop_size,
    archive_size=best_archive_size,
    stopping_criteria=[
        MaxGenerations(500),
        TimeLimit(120),
        MinImprovement(50, min_pct=0.05),
        TargetObjective(inst.best_known),
    ],
    penalty_weight=best_penalty,
    mutation_rate=best_mutation_rate,
    seed=best_seed,
)

our_distance = result['front_raw_dists'].min()
print(f"\nGenerations run:         {result['generations_run']}")
print(f"Final Pareto front size: {len(result['front'])}")
print(f"Best service distance:   {our_distance:.1f}")
print(f"Best dispersion:         {(-result['front_objectives'][:, 1]).max():.1f}")
print(f"Known best (single-obj): {inst.best_known}")

# %% [markdown]
# ### NSGA 3
# 
# Kalyanmoy Deb, Himanshu Jain
# "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints"

# %% [markdown]
# Algorithm 2

# %%

# Normalize (f**n, S, Z**r, Z**s/Z**a)
# input S, Z**s or Z**a
# output f**n, Z**r
def normalize_objectives(objectives, Z_s):
    # NOTE:
    # - `objectives` is expected to be a numpy array of shape (|S|, M)
    # - `S` is not used directly here (kept to match the lecture pseudocode signature)
    # - `Z_s` is expected to be a numpy array of reference points of shape (K, M)

    objectives = np.asarray(objectives, dtype=float)
    Z_s = np.asarray(Z_s, dtype=float)

    if objectives.ndim != 2:
        raise ValueError("normalize_objectives: `objectives` must be 2D (n_solutions, n_objectives).")
    if Z_s.ndim != 2 or Z_s.shape[1] != objectives.shape[1]:
        raise ValueError("normalize_objectives: `Z_s` must be 2D with same number of objectives as `objectives`.")
    
    if objectives.shape[0] == 0:
        return objectives, Z_s

    # for j = 1 to M
    z_min = objectives.min(axis=0)                 # ideal point per objective
    obj_translated = objectives - z_min            # translate objectives

    # line 6 - compute intercepts a_j for j = 1 to M
    a = np.zeros(objectives.shape[1])
    for j in range(objectives.shape[1]):
        # find the point in S with the maximum value on objective j
        idx = np.argmax(obj_translated[:, j])
        a[j] = obj_translated[idx, j]

    # avoid division by zero; keep normalization stable
    a_safe = np.where(a > 0, a, 1.0)

    # line 7 - normalize objectives f**n = (obj_translated) / a_j
    obj_normalized = obj_translated / a_safe

    Z_r = Z_s
    return obj_normalized, Z_r


# %% [markdown]
# Algorithm 3

# %%
# Associate(S, Z**r)
def associate_solutions(objectives, Z_r):
    objectives = np.asarray(objectives, dtype=float)
    Z_r = np.asarray(Z_r, dtype=float)

    if objectives.ndim != 2:
        raise ValueError("associate_solutions: `objectives` must be 2D (n_solutions, n_objectives).")
    if Z_r.ndim != 2 or Z_r.shape[1] != objectives.shape[1]:
        raise ValueError("associate_solutions: `Z_r` must be 2D with same number of objectives as `objectives`.")

    n_solutions = objectives.shape[0]
    n_refs = Z_r.shape[0]
    
    if n_solutions == 0 or n_refs == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # for each reference point z in Z**r, compute reference line w = z
    # compute perpendicular distance d(s, w) from s to w
    # equal to ||(s - w_T*s*w/||w||^2)||
    perp_dists = np.empty((n_solutions, n_refs), dtype=float)

    for w in range(n_refs):
        wv = Z_r[w]
        denom = float(np.dot(wv, wv))
        if denom == 0.0:
            # degenerate reference direction: distance becomes norm of objective vector
            perp_dists[:, w] = np.linalg.norm(objectives, axis=1)
        else:
            proj_scale = (objectives @ wv) / denom                     # shape (n_solutions,)
            proj = np.outer(proj_scale, wv)                            # shape (n_solutions, M)
            perp_dists[:, w] = np.linalg.norm(objectives - proj, axis=1)

    # assign pi(s) = w : argmin(w belonging to Z**r) perp_dist(s, w)
    pi_s = np.argmin(perp_dists, axis=1)                               # shape (n_solutions,)

    # assign d(s) = perp_dist(s, pi(s))
    d_s = perp_dists[np.arange(n_solutions), pi_s]                     # shape (n_solutions,)

    return pi_s, d_s


# %% [markdown]
# Algorithm 4

# %%
# Niching (K, rho_j, pi(s belonging to objectives), d(s belonging to objectives), Z**r, F_l))
def niching(K, rho_j, pi_s, d_s, Z_r, F_l):
    # NOTE:
    # - `rho_j` is expected to be an array of niche counts, length = |Z_r|
    # - `pi_s` is expected to be an array of reference indices for solutions in F_l
    # - `d_s` is expected to be distances for solutions in F_l to their associated reference
    # - `F_l` is expected to be an iterable of solution identifiers aligned with pi_s/d_s

    rho_j = np.asarray(rho_j, dtype=int)
    pi_s = np.asarray(pi_s, dtype=int)
    d_s = np.asarray(d_s, dtype=float)
    Z_r = np.asarray(Z_r)

    F_l = list(F_l)

    if len(F_l) != len(pi_s) or len(F_l) != len(d_s):
        raise ValueError("niching: F_l, pi_s, and d_s must have the same length.")
    
    if K <= 0:
        return []

    next_gen = []
    k = 1

    # we keep a list of active reference point indices (instead of deleting rows from Z_r)
    active_refs = list(range(len(rho_j)))

    while k <= K and len(active_refs) > 0:
        # J_min = {j: argmin(j belonging to Z**r) rho_j}
        min_val = rho_j[active_refs].min()
        J_min = [j for j in active_refs if rho_j[j] == min_val]

        # _j = random(J_min)
        _j = np.random.choice(J_min)

        # I__j = {s: pi(s) == _j}
        I__j = [idx for idx in range(len(F_l)) if pi_s[idx] == _j and F_l[idx] not in next_gen]

        # if I__j is not empty
        if len(I__j) > 0:
            # if rho__j = 0 then next_gen = next_gen ∪ {s: argmin(s belonging to I__j) d(s)}
            if rho_j[_j] == 0:
                chosen_local = min(I__j, key=lambda idx: d_s[idx])
                next_gen.append(F_l[chosen_local])
            else:
                chosen_local = np.random.choice(I__j)
                next_gen.append(F_l[chosen_local])

            rho_j[_j] += 1
            k += 1
        else:
            # Z**r = Z**r \ {z_j}
            # (implement by disabling this reference index)
            active_refs = [j for j in active_refs if j != _j]

    return next_gen


# %% [markdown]
# ##### Main NSGA3 function
# 
# Algorithm 1

# %%
def generate_reference_points_das_dennis(num_objectives, divisions):
    """Generate reference points using Das & Dennis method."""
    if num_objectives == 1:
        return np.array([[1.0]])
    
    points = []
    
    def generate_recursive(remaining_objectives, remaining_divisions, current_point):
        if remaining_objectives == 1:
            points.append(current_point + [remaining_divisions])
        else:
            for i in range(remaining_divisions + 1):
                generate_recursive(remaining_objectives - 1, remaining_divisions - i, current_point + [i])
    
    generate_recursive(num_objectives, divisions, [])
    points = np.array(points, dtype=float)
    points = points / divisions
    return points


def compute_niche_counts_nsga3(pi_combined, n_refs):
    """Compute niche counts: how many solutions are associated with each reference point."""
    rho = np.zeros(n_refs, dtype=int)
    for j in pi_combined:
        rho[j] += 1
    return rho


# generation t of nsga-iii procedure

# input: h structured reference points Z**s or supplied aspiration points Z**a, parent population P_t
# output: P_{t+1}
def nsga3(instance, pop_size=best_pop_size, archive_size=best_archive_size,
          stopping_criteria=None,
          penalty_weight=best_penalty, crossover_rate=best_crossover_rate, mutation_rate=best_mutation_rate,
          seed=best_seed, verbose=True):
    if stopping_criteria is None:
        stopping_criteria = [MaxGenerations(200)]

    rng = np.random.default_rng(seed)
    n_cust = instance.num_customers 
    n_med = instance.num_medians
    if mutation_rate is None:
        mutation_rate = 1.0 / n_cust

    # Initialise stopping criteria
    for criterion in stopping_criteria:
        criterion.reset()

    # Generate reference points using Das & Dennis method (2 objectives, 5 divisions)
    reference_points = generate_reference_points_das_dennis(2, 5)
    n_refs = len(reference_points)
    
    # Adjust archive size to be close to number of reference points
    effective_archive_size = max(archive_size, n_refs)

    # Line 3 — initialise population
    P = initialize_population(pop_size, n_cust, n_med, rng)

    # Line 4 — empty archive
    A = np.empty((0, n_cust), dtype=int)
    A_obj = np.empty((0, 2))
    A_raw = np.empty(0)

    history = {
        "best_service_dist": [],   # true service distance, no penalty
        "best_dispersion": [],
        "front_sizes": [],
    }

    gen = 0
    while True:
        # Line 6 — evaluate new population
        P_obj, P_raw = evaluate_population(P, instance, penalty_weight)

        # Line 7 — P ← P ∪ A
        if len(A) > 0:
            combined = np.vstack([P, A])
            combined_obj = np.vstack([P_obj, A_obj])
            combined_raw = np.concatenate([P_raw, A_raw])
        else:
            combined = P
            combined_obj = P_obj
            combined_raw = P_raw

        # Line 8 — best Pareto front (for tracking)
        front_mask = compute_pareto_front(combined_obj)

        # Non-dominated sorting to get fronts
        ranks = front_rank_assignment(combined, combined_obj)
        
        # Normalization and association
        if len(combined) > 0:
            objectives_norm, _ = normalize_objectives(combined_obj, reference_points)
            pi_combined, d_combined = associate_solutions(objectives_norm, reference_points)
        else:
            pi_combined = np.array([], dtype=int)
            d_combined = np.array([], dtype=float)

        # Archive construction using NSGA-III selection
        archive_idx = []
        
        for rank in sorted(np.unique(ranks)):
            front_rank_mask = (ranks == rank)
            front_idx = np.where(front_rank_mask)[0]
            
            if len(archive_idx) + len(front_idx) >= effective_archive_size:
                # Compute niche counts for current archive + new front
                rho_j = compute_niche_counts_nsga3(pi_combined[archive_idx], n_refs)
                
                # Niching on the last front to fill remaining slots
                n_to_add = effective_archive_size - len(archive_idx)
                pi_Fl = pi_combined[front_idx]
                d_Fl = d_combined[front_idx]
                
                chosen_pos = niching(n_to_add, rho_j, pi_Fl, d_Fl, reference_points, 
                                    list(range(len(front_idx))))
                archive_idx.extend(front_idx[chosen_pos].tolist())
                break
            else:
                archive_idx.extend(front_idx.tolist())

        # Materialize archive arrays
        A = combined[np.array(archive_idx, dtype=int)]
        A_obj = combined_obj[np.array(archive_idx, dtype=int)]
        A_raw = combined_raw[np.array(archive_idx, dtype=int)]

        # Record history — use true service distance (no penalty)
        front_raw = combined_raw[front_mask]
        front_obj = combined_obj[front_mask]
        history["best_service_dist"].append(front_raw.min())
        history["best_dispersion"].append((-front_obj[:, 1]).max())
        history["front_sizes"].append(front_mask.sum())

        if verbose and ((gen + 1) % 50 == 0 or gen == 0):
            print(
                f"Gen {gen+1:4d}  |  front size {front_mask.sum():3d}  |  "
                f"best dist {front_raw.min():.1f}  |  "
                f"best disp {(-front_obj[:, 1]).max():.1f}"
            )

        # Check stopping criteria
        triggered = [c for c in stopping_criteria if c.should_stop(gen, history)]
        if triggered:
            if verbose:
                print(f"Stopped at gen {gen + 1} — triggered: {triggered}")
            break

        # Line 10 — breed new population from archive (with repair)
        A_fit = compute_fitness(A_obj) if len(A) > 1 else np.ones(len(A))
        P = breed(A, A_fit, pop_size, n_med, rng, crossover_rate,
                  mutation_rate, instance=instance)
        gen += 1

    # Final front from archive — filter to feasible solutions only
    final_mask = compute_pareto_front(A_obj)
    front_pop = A[final_mask]
    front_obj = A_obj[final_mask]
    front_raw = A_raw[final_mask]

    feasible_mask = np.array([
        is_feasible(ind, instance.demands, instance.capacity, n_med)
        for ind in front_pop
    ])

    if feasible_mask.any():
        front_pop = front_pop[feasible_mask]
        front_obj = front_obj[feasible_mask]
        front_raw = front_raw[feasible_mask]
    else:
        if verbose:
            print("WARNING: no feasible solution found on the Pareto front, returning all.")

    return {
        "archive": A,
        "archive_objectives": A_obj,
        "archive_raw_dists": A_raw,
        "front": front_pop,
        "front_objectives": front_obj,
        "front_raw_dists": front_raw,
        "history": history,
        "generations_run": gen + 1,
    }


# %%
# Run NSGA3 on instance 1 (50 customers, 5 medians)
inst = load_instance(1)
result = nsga3(
    inst,
    pop_size=best_pop_size,
    archive_size=best_archive_size,
    stopping_criteria=[
        MaxGenerations(500),
        TimeLimit(120),
        MinImprovement(50, min_pct=0.05),
        TargetObjective(inst.best_known),
    ],
    penalty_weight=best_penalty,
    crossover_rate=best_crossover_rate,
    mutation_rate=best_mutation_rate,
    seed=best_seed,
)

our_distance = result['front_raw_dists'].min()
print(f"\nGenerations run:         {result['generations_run']}")
print(f"Final Pareto front size: {len(result['front'])}")
print(f"Best service distance:   {our_distance:.1f}")
print(f"Best dispersion:         {(-result['front_objectives'][:, 1]).max():.1f}")
print(f"Known best (single-obj): {inst.best_known}")

# %% [markdown]
# #### Fine-tuning NSGA3
# 

# %%
if FINE_TUNE_NSGA3:
    multiplier = 0.5
    min_multiplier = 0.01
    max_i = FINE_TUNE_NSGA3
    multiplier_step = (multiplier - min_multiplier) / max_i

    best_result = float(inst.best_known * 3.)
    best_run_result = None

    for i in range(max_i):
        pop_low = max(2, int(best_pop_size * (1.0 - multiplier)))
        pop_high = max(pop_low + 1, int(best_pop_size * (1.0 + multiplier)) + 1)
        pop_size = np.random.randint(pop_low, pop_high)

        arch_low = max(1, int(best_archive_size * (1.0 - multiplier)))
        arch_high = min(pop_size, int(best_archive_size * (1.0 + multiplier)))
        archive_size = np.random.randint(arch_low, max(arch_low + 1, arch_high + 1))

        penalty = np.random.uniform(
            max(0.0, best_penalty * (1.0 - multiplier)),
            max(0.0, best_penalty * (1.0 + multiplier))
        )
        crossover_rate = np.random.uniform(
            max(0.0, best_crossover_rate * (1.0 - multiplier)),
            min(1.0, best_crossover_rate * (1.0 + multiplier))
        )
        mutation_rate = np.random.uniform(
            max(0.0, best_mutation_rate * (1.0 - multiplier)),
            min(1.0, best_mutation_rate * (1.0 + multiplier))
        )

        seed_low = max(0, int(best_seed * (1.0 - multiplier)))
        seed_high = min(np.iinfo(np.uint32).max, int(best_seed * (1.0 + multiplier)))
        seed = np.random.randint(seed_low, max(seed_low + 1, seed_high + 1))

        print(
            f"[{i+1:3d}/{max_i}] pop={pop_size}, arch={archive_size}, "
            f"penalty={penalty:.2f}, cx={crossover_rate:.3f}, "
            f"mut={mutation_rate:.3f}, seed={seed}, mult={multiplier:.3f}",
            end=" ... "
        )

        result = nsga3(
            inst,
            pop_size=pop_size,
            archive_size=archive_size,
            penalty_weight=penalty,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            seed=seed,
            stopping_criteria=[
                TimeLimit(120),
                MinImprovement(50, min_pct=0.05),
            ],
            verbose=False,
        )

        local_best = result["front_raw_dists"].min()
        print(f"dist={local_best:.1f}", end="")

        if local_best < best_result:
            print(f"  *** new best ({best_result:.1f} → {local_best:.1f}) ***")
            best_result = local_best
            best_pop_size = pop_size
            best_archive_size = archive_size
            best_penalty = penalty
            best_crossover_rate = crossover_rate
            best_mutation_rate = mutation_rate
            best_seed = seed
            best_run_result = result

            # Plot the new best solution
            best_idx = np.argmin(best_run_result["front_raw_dists"])
            best_ind = best_run_result["front"][best_idx]
            best_fac = compute_facility_locations(best_ind, inst.coords, inst.num_medians)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"Fine-tune iter {i+1} — new best dist = {local_best:.1f} "
                f"(known best = {inst.best_known})",
                fontsize=12
            )

            # Left: convergence of this run
            gens = range(1, len(best_run_result["history"]["best_service_dist"]) + 1)
            axes[0].plot(gens, best_run_result["history"]["best_service_dist"], color="tab:blue", label=f"Our result = {local_best:.0f}")
            axes[0].set_xlabel("Generation")
            axes[0].set_ylabel("Best Service Distance")
            axes[0].set_title("Convergence")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Right: facility map
            cmap = plt.colormaps.get_cmap("tab10").resampled(inst.num_medians)
            for m in range(inst.num_medians):
                mask = best_ind == m
                axes[1].scatter(
                    inst.coords[mask, 0], inst.coords[mask, 1],
                    color=cmap(m), alpha=0.6, s=40, label=f"Cluster {m}"
                )
                for ci in np.where(mask)[0]:
                    axes[1].plot(
                        [inst.coords[ci, 0], best_fac[m, 0]],
                        [inst.coords[ci, 1], best_fac[m, 1]],
                        color=cmap(m), alpha=0.15, linewidth=0.8
                    )
            axes[1].scatter(
                best_fac[:, 0], best_fac[:, 1],
                marker="o", s=125, color="red", facecolors="none", zorder=10, label="Facility"
            )
            axes[1].set_title("Best Facility Map")
            axes[1].legend(loc="upper right", fontsize=8)
            axes[1].grid(True, alpha=0.2)
            axes[1].set_aspect("equal")

            plt.tight_layout()
            plt.show()
        else:
            print()

        multiplier = max(min_multiplier, multiplier - multiplier_step)

    print("\nBest tuning result:")
    print(f"  best_result      = {best_result:.4f}")
    print(f"  pop_size         = {best_pop_size}")
    print(f"  archive_size     = {best_archive_size}")
    print(f"  penalty          = {best_penalty:.4f}")
    print(f"  crossover_rate   = {best_crossover_rate:.4f}")
    print(f"  mutation_rate    = {best_mutation_rate:.4f}")
    print(f"  seed             = {best_seed}")


# %%
result = nsga3(
    inst,
    pop_size=best_pop_size,
    archive_size=best_archive_size,
    stopping_criteria=[
        MaxGenerations(500),
        TimeLimit(120),
        MinImprovement(50, min_pct=0.05),
        TargetObjective(inst.best_known),
    ],
    penalty_weight=best_penalty,
    crossover_rate=best_crossover_rate,
    mutation_rate=best_mutation_rate,
    seed=best_seed,
)

our_distance = result['front_raw_dists'].min()
print(f"\nGenerations run:         {result['generations_run']}")
print(f"Final Pareto front size: {len(result['front'])}")
print(f"Best service distance:   {our_distance:.1f}")
print(f"Best dispersion:         {(-result['front_objectives'][:, 1]).max():.1f}")
print(f"Known best (single-obj): {inst.best_known}")


# %% [markdown]
# ### NSGA3 Visualisations
# 

# %%

# Convergence and metric plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

gens = range(1, len(result["history"]["best_service_dist"]) + 1)

# Service distance convergence
axes[0].plot(gens, result["history"]["best_service_dist"], color="tab:blue", label=f"Our result = {our_distance:.0f}")
axes[0].axhline(y=inst.best_known, color="tab:red", linestyle="--", linewidth=1.2,
                label=f"Best known = {inst.best_known}")
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Best Service Distance")
axes[0].set_title("Service Distance over Generations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dispersion convergence
axes[1].plot(gens, result["history"]["best_dispersion"], color="tab:green")
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Best Dispersion")
axes[1].set_title("Facility Dispersion over Generations")
axes[1].grid(True, alpha=0.3)

# Pareto front size
axes[2].plot(gens, result["history"]["front_sizes"], color="tab:orange")
axes[2].set_xlabel("Generation")
axes[2].set_ylabel("Front Size")
axes[2].set_title("Pareto Front Size over Generations")
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"NSGA3 Improvement Process — Instance {inst.instance_id}", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# %%

# Pareto front scatter plot — final archive
fig, ax = plt.subplots(figsize=(8, 6))

# All archive members
ax.scatter(
    result["archive_raw_dists"],
    -result["archive_objectives"][:, 1],
    alpha=0.4, label="Archive", color="tab:gray", edgecolors="none", s=30
)

# Non-dominated front
ax.scatter(
    result["front_raw_dists"],
    -result["front_objectives"][:, 1],
    alpha=0.9, label="Pareto Front", color="tab:red", edgecolors="black",
    linewidths=0.5, s=60, zorder=5
)

# Sort front by service distance and connect with line
sorted_idx = np.argsort(result["front_raw_dists"])
ax.plot(
    result["front_raw_dists"][sorted_idx],
    -result["front_objectives"][sorted_idx, 1],
    color="tab:red", alpha=0.5, linestyle="--", linewidth=1
)

ax.set_xlabel("Service Distance")
ax.set_ylabel("Facility Dispersion")
ax.set_title(f"NSGA3 Pareto Front — Instance {inst.instance_id}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%

# Facility map — best solution (lowest service distance) from the Pareto front
best_idx = np.argmin(result["front_raw_dists"])
best_ind = result["front"][best_idx]
best_fac = compute_facility_locations(best_ind, inst.coords, inst.num_medians)

fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.colormaps.get_cmap("tab10").resampled(inst.num_medians)

for m in range(inst.num_medians):
    mask = best_ind == m
    ax.scatter(
        inst.coords[mask, 0], inst.coords[mask, 1],
        color=cmap(m), alpha=0.6, s=40, label=f"Cluster {m}"
    )
    # Draw lines from customers to their facility
    for ci in np.where(mask)[0]:
        ax.plot(
            [inst.coords[ci, 0], best_fac[m, 0]],
            [inst.coords[ci, 1], best_fac[m, 1]],
            color=cmap(m), alpha=0.15, linewidth=0.8
        )

# Facility locations as stars
ax.scatter(
    best_fac[:, 0], best_fac[:, 1],
    marker="o", s=125, color="red", facecolors='none', zorder=10, label="Facility"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Best Solution Facility Map — Instance {inst.instance_id}")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# Print capacity utilisation
print("Capacity utilisation:")
for m in range(inst.num_medians):
    demand = inst.demands[best_ind == m].sum()
    print(f"  Median {m}: demand={demand:3d} / capacity={inst.capacity}  "
          f"{'VIOLATED' if demand > inst.capacity else 'OK'}")



