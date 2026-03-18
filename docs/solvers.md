# Solvers & Algorithms Documentation (Parallel Machine Scheduling)

This repository implements multiple approaches for the **parallel machine scheduling** problem with:
- **Machine-dependent release dates** r[j,k]
- **Machine-dependent processing times** d[j,k]
- **Sequence-dependent setup times** s[i,j,k] that apply **only when job j is processed immediately after job i** on machine k
- **Capability constraints**: job j can only run on machines in `capable[j]`
- Objective: **minimize makespan** C_max = max over jobs of completion time C[j]

The goal of this document is to (1) explain how every implemented solver works, (2) highlight shared logic across solvers, and (3) document **theory conformance** where an approach deviates from a standard textbook formulation.

---

## 1) Core data model (shared by all solvers)

### 1.1 `SchedulingInstance`
File: `src/models/instance.py`

Holds the instance data loaded from JSON:
- `n`, `m`, `horizon`
- `capable[j]`: list of machines allowed for job j
- `duration[j][k]`, `release[j][k]`, `setup[i][j][k]`

Also exposes the inverse mapping `machine_capable_jobs` (jobs each machine can run).

### 1.2 `SchedulingSolution`
File: `src/models/solution.py`

Represents a schedule as:
- `schedule: dict[int, list[int]]` mapping machine k → ordered list of **1-indexed** job IDs.

**Important: objective evaluation is centralized here.**

#### Makespan evaluation (the “ground truth” used by all metaheuristics)
`compute_makespan()` matches `checker.py` exactly:
- For the **first** job on a machine: start = max(release[j,k], 0)
- For **later** jobs: start = max(release[j,k], prev_completion + setup[prev][j][k])  
  where **prev** is the job immediately before j on that machine (same indices as in the JSON: previous job index, current job index j, machine k).
- Completion C = start + duration[j,k]
- Machine completion = completion of its last job; makespan = max over machines.

This is the theoretical model used to score solutions in all metaheuristics and in the checker.

#### Feasibility checks implemented
`is_feasible()` checks:
- Each job appears **exactly once**
- Each job assigned to a capable machine

Note: it does **not** verify the full timing feasibility because start times are derived during makespan evaluation; if the schedule is an ordering on each machine, timing is implied by the recurrence above.

---

## 2) Shared building blocks (common logic across many solvers)

### 2.1 Neighborhood operators and local search
File: `src/local_search/operators.py`

All local-search-based methods share the same “move set” and the same evaluation (via `SchedulingSolution.compute_makespan()`).

#### Deterministic improvement operators (first-improvement)
- **`swap_within_machine`**: swap two jobs on the same machine if it improves.
- **`move_within_machine`**: remove a job at position i and reinsert at position j on the same machine if it improves.
- **`move_to_other_machine`**: move a job to another **capable** machine and try all insertion positions, accept first improving.
- **`swap_between_machines`**: swap one job on machine A with one on machine B if both remain capable, accept first improving.

#### `local_search(solution)`
Applies the operators in a fixed sequence repeatedly until none improves (standard hill-climb to a local optimum).

#### `random_neighbor(solution)`
Generates a random neighbor using one random move type from the same operator family.
This is the shared perturbation mechanism for SA/ES/GA mutations and ILS perturbations.

#### `perturb(solution, strength)`
Applies `strength` random neighbors in a row (escape mechanism for ILS).

### 2.2 Stopping criteria (shared termination logic)
File: `src/stopping_criteria/criteria.py`

All iterative solvers (and the combined solver) stop based on one or more criteria:
- **`TimeLimit(seconds)`**: wall-clock time limit.
- **`MaxGenerations(n)`**: maximum number of iterations/checks.
- **`GenMinImprovement(window, min_pct)`**: stop if relative improvement over the last `window` best-so-far values is ≤ `min_pct`.
- **`TimeMinImprovement(window_secs, min_pct)`**: similar but based on elapsed time marks.
- **`TargetObjective(target)`**: stop once best makespan ≤ target.

The base metaheuristic loop stores a **best-so-far history** (non-increasing list) which these criteria inspect.

---

## 3) Metaheuristic solvers (implementation + theory mapping)

All metaheuristic solvers ultimately produce a `SchedulingSolution` and compare candidates by **makespan** (lower is better). The “theory conformance” question for these methods is mostly about:
- Is the solver implementing the standard *pattern*? (construction / neighborhood / acceptance / selection / pheromones)
- Are feasibility and objective evaluation consistent with the problem definition?

In this project, feasibility and objective evaluation are consistent because all solvers rely on `SchedulingSolution.compute_makespan()`.

### 3.1 GRASP (`--solver grasp`)
File: `src/solvers/grasp.py`

**Theory (GRASP):**
Repeatedly:
1. Greedy randomized construction using a **restricted candidate list (RCL)**.
2. Local search from the constructed solution.
3. Keep best solution found.

**Implementation:**
- Construction considers each unassigned job j and each capable machine k, and computes the **earliest completion** if j is appended **next** on k given the current partial sequence:

  **completion(j,k) = max(r[j,k], t_k + s[last,j,k]) + d[j,k]**

  where t_k is the completion time of the last job already placed on k, **last** is that job’s index (setup term omitted if the machine is still empty).

- Builds an RCL by threshold: **threshold = c_min + α · (c_max − c_min)**, then picks uniformly at random inside the RCL.
- Improvement uses the shared `local_search()`.

### 3.2 Simulated Annealing (SA) (`--solver sa`)
File: `src/solvers/simulated_annealing.py`, cooling schedules in `src/solvers/cooling.py`

**Theory (SA):**
Maintain a current solution; propose a neighbor; accept improving moves always and sometimes accept worse moves with probability **exp(−Δ / T)**, and reduce temperature T over time.

**Implementation:**
- Starts from a random feasible schedule (`SolverBase._random_solution`).
- Neighbor generation: `random_neighbor()`.
- Acceptance: if Δ < 0 accept; else accept with probability **exp(−Δ / T)**.
- Cooling is pluggable, default is **geometric**: T ← α · T.
- Adds **reheating**: if no improvement for `reheat_patience`, increase T by `reheat_factor` (capped at initial T).

### 3.3 Evolution Strategy (ES) (`--solver es`)
File: `src/solvers/evolution_strategy.py`

**Theory ((μ+λ)-ES with 1/5 rule):**
- Keep μ parents.
- Generate λ offspring by mutation.
- Select next generation from the best μ individuals out of parents + offspring.
- Adapt mutation strength σ using Rechenberg’s 1/5 success rule.

**Implementation:**
- Initializes μ+λ random schedules, then selects top μ each generation (truncation).
- Mutation: apply **num_moves ≈ σ** times `random_neighbor()`.
- Success is counted when an offspring beats its parent’s makespan.
- Updates σ with the 1/5 rule and parameter c: if success_rate > 0.2 then σ ← σ/c; if success_rate < 0.2 then σ ← σ·c.

### 3.4 Iterated Local Search (ILS) (`--solver ils`)
File: `src/solvers/iterated_local_search.py`

**Theory (ILS):**
1. Start from a solution.
2. Apply local search to reach a local optimum.
3. Perturb the solution to escape the basin.
4. Local search again.
5. Keep best solution seen; repeat.

**Implementation:**
- Initial: random solution → `local_search`.
- Iteration: `perturb(strength)` → `local_search`.
- Keeps best-so-far.

### 3.5 Genetic Algorithm (GA) (`--solver ga`)
File: `src/solvers/genetic_algorithm.py`

**Theory (GA template):**
Maintain a population; generate offspring via crossover + mutation; select survivors by fitness.

**Representation used here:**
This GA effectively represents a schedule mainly by **job → machine assignment** (not a full permutation/sequence encoding).
After choosing assignments, it derives a within-machine order by sorting.

**Implementation:**
- Population initialized randomly.
- **Uniform crossover on machine assignment**: for each job j, take the machine from parent 1 or parent 2 with probability 0.5.
- Reconstruct schedule from assignments: jobs on each machine sorted by `release[j,k]` with random tie-break.
- Mutation: apply `random_neighbor()` a fixed number of times (`mutation_strength`).
- Selection: truncation (keep best `population_size` from parents + offspring) — effectively (μ+λ) selection.

**Theory conformance (deviations to be aware of):**
- This is closer to an **EA with assignment-based crossover** than a “full sequencing GA”, because sequence information is not inherited directly; ordering is rebuilt by a heuristic (release-date sort).
- Mutation reintroduces sequencing changes via neighborhood moves, so over time sequences can diverge from release-date ordering.

### 3.6 Ant Colony Optimization family (ACO)
Files:
- Base framework: `src/solvers/aco_base.py`
- Variants: `src/solvers/ant_system.py`, `src/solvers/max_min_ant_system.py`, `src/solvers/ant_colony_system.py`, `src/solvers/ant_multi_tour_system.py`

#### Common representation in this project
Pheromones are on **assignment edges**:
- `tau[j][k]` for job j assigned to machine k

Heuristic information is also on job–machine pairs:
- `eta[j][k] = 1 / (duration[j,k] + avg_setup_to_j_on_k)`

Construction:
1. Each ant assigns each job to a machine probabilistically using **τ^α · η^β**.
2. For each machine, the assigned jobs are then **ordered by a greedy sequencing heuristic**: repeatedly choose the next job that minimizes earliest completion given the current last job (see `_order_jobs_on_machine`).
3. Local search is applied to the constructed schedule.

**This means**: the ACO learns “good machine assignments” more directly than it learns “good sequences”; sequencing is mainly handled by a greedy ordering step and subsequent local search.

#### 3.6.1 Ant System (AS)
Class: `AntSystem`

**Theory:** all ants deposit pheromone, proportional to solution quality.

**Implementation:** evaporate then for each ant deposit **Δτ = q_ct / cost** on each `(job, machine)` used.

#### 3.6.2 Ranked Ant System (Rank-based AS)
Class: `RankedAntSystem`

**Theory:** only top w ants deposit pheromone, weighted by rank.

**Implementation:** choose w ≈ n_ants/4, deposit **Δτ ∝ (w − rank) / cost**.

#### 3.6.3 Elitist Ant System (EAS)
Class: `EasAntSystem`

**Theory:** all ants deposit + extra deposit from best-so-far.

**Implementation:** normal AS deposit plus σ-weighted extra deposit from stored global best.

#### 3.6.4 MAX–MIN Ant System (MMAS)
Class: `MaxMinAntSystem`

**Theory:** only best-so-far deposits; pheromones constrained to [τ_min, τ_max] to avoid stagnation; sometimes reinitialization.

**Implementation note:** this implementation does:
- evaporation
- deposit from best-so-far only
- periodic reinitialization to a constant τ value

It **does not explicitly enforce** τ_min bounds; “max–min” behavior is approximated mainly via best-only update + reinit.

#### 3.6.5 Ant Colony System (ACS)
Class: `AntColonySystem`

**Theory:** exploitation/exploration with q0, plus **local pheromone update** during construction, plus global update from best.

**Implementation:**
- With probability q0: choose argmax of attractiveness (exploit), else sample (explore).
- Local update: τ ← (1 − φ)·τ + φ·τ0 on the chosen `(job, machine)`.
- Global update from best-so-far only.

#### 3.6.6 Ant Multi-Tour System (AMTS)
Class: `AntMultiTourSystem`

**Idea implemented:**
Track how often each `(job, machine)` assignment has been used recently and penalize frequently used assignments to encourage exploration.

Implementation:
- Maintain `usage[j][k]`.
- During construction, attractiveness is divided by **1 + sqrt(usage)**.
- Usage decays periodically.

---

## 4) Combined solver (“hyper-heuristic” orchestration) (`--solver combined`)
File: `src/solvers/combined.py` and wiring in `src/main.py`

**What it is:**
Not a single algorithm, but an orchestration strategy that runs several solvers in sequence and restarts the cycle when progress is made.

**Implementation behavior:**
- Each sub-solver is created fresh from a factory each time (avoids state leakage like pheromones/populations).
- Runs a sub-solver until its stopping criteria fires.
- If the sub-solver improves the **global** best, the remaining list is reset so each solver gets another chance (starting from the next).
- Stops once all solvers have run without improving the global best.

Default factories (in `main.py`) are:
`ILSSolver`, `MaxMinAntSystem`, `RankedAntSystem`, `GraspSolver`, `AntMultiTourSystem`, `EvolutionStrategySolver`.

**Theory conformance (deviation):**
- This solver is intentionally *not* a single textbook metaheuristic; it is a **portfolio / hyper-heuristic** that cycles through several different algorithms based on improvement, which is a pragmatic competition strategy rather than a canonical “combined solver” definition.

---

## 5) MiniZinc CP solver (`--solver minizinc`)
Files:
- Adapter: `src/minizinc_cp/solver.py`
- Model: `src/minizinc_cp/scheduling.mzn`

### 5.1 What the adapter does
The Python adapter:
- Converts `SchedulingInstance` to MiniZinc parameters (`capable`, `duration`, `release`, `setup`, etc.).
- Runs MiniZinc with `intermediate_solutions=True` and optional time limit (mapped from `TimeLimit`).
- Converts MiniZinc arrays (`assign`, `start`) back to a `SchedulingSolution` by sorting jobs per machine by returned start time.
- Uses **Chuffed** as the backend; `TimeLimit` in criteria becomes the CP solver’s wall-clock limit. With intermediate solutions enabled, the result is the **best incumbent** within that limit (on large instances this need not be optimal or proven).

### 5.2 Theory conformance of the MiniZinc model
The model matches the problem definition (release dates, capability, processing time, **immediate-predecessor setup**, minimize makespan).

**Variables:**
- `assign[j]`: machine for job j
- `rank[j]`: position of job j on that machine (1 = first, 2 = second, …); jobs on the same machine have distinct ranks in 1..n_k where n_k is the number of jobs on machine k
- `start[j]`, `end[j]`

**Timing:**
- `start[j] >= release[j, assign[j]]` and `end[j] = start[j] + duration[j, assign[j]]`
- **Only consecutive pairs on the same machine** pay setup: if `rank[j] = rank[j1] + 1` and `assign[j1] = assign[j]`, then `start[j] >= end[j1] + setup[j1, j, assign[j]]`. The first job on each machine has no predecessor, so only the release constraint applies for that link.

This is equivalent to the checker’s per-machine recurrence (first job: release only; later jobs: previous completion + setup to current job).

---

## 6) “Theory vs implementation” summary (what to remember)

- **Objective & feasibility**: implemented faithfully and consistently in `SchedulingSolution.compute_makespan()` and `checker.py`.
- **GRASP / SA / ES / ILS**: follow standard textbook patterns; differences are practical tuning (e.g., SA reheating).
- **GA**: crossover is on **assignments**; sequencing is rebuilt by a heuristic then refined via mutation moves.
- **ACO family**: pheromones are on **job→machine assignment**, not on sequencing edges; sequencing is handled by greedy ordering + local search.
- **MiniZinc CP**: uses `assign` + `rank` so setup applies only between **consecutive** jobs on each machine, aligned with `checker.py`.
