# Competition Problem: Parallel Machine Scheduling

We tackle a **challenging scheduling problem** on parallel machines with sequence-dependent setup times and machine-dependent release dates.
This simplified version is based on a **real-world semiconductor workshop scheduling** problem.

---

## Problem Description

We have **m** parallel machines and **n** jobs.

- Each job must be processed by exactly one machine.
- Each machine can process at most one job at a time.
- Jobs cannot be interrupted once started (**no preemption**).

Some machines can only process specific jobs. For each job *j*, `cap(j)` is the set of machines capable of processing it.

For each job *j* and machine *k* we know:

- **Release date** `r[j,k]` – earliest time job *j* can start on machine *k* (due to transport time or other constraints).
- **Processing time** `d[j,k]` – time it takes to complete job *j* on machine *k*.
- **Setup time** `s[i,j,k]` – time to prepare machine *k* for job *j* immediately after job *i*. This is *sequence-dependent*.

A **schedule** consists of:

1. Assigning each job to a machine capable of processing it.
2. Ordering the assigned jobs on each machine.

**Start times:**

- First job on a machine: start ≥ its release date.
- Subsequent jobs: start ≥ max(its release date, completion time of its predecessor).

**Completion time** = start time + processing time (+ setup time if not first).  
**Makespan** = the maximum completion time among all jobs.

**Goal:** Find a feasible schedule that **minimizes the makespan**.

---

## Input Format

Input is given in JSON with the following fields:

- `n`: number of jobs
- `m`: number of machines
- `horizon`: upper bound on the makespan
- `capable`: list of length `n`, where `capable[j]` is the list of machines that can process job *j*
- `duration`: `n × m` array – `duration[j][k]` is the processing time of job *j* on machine *k*
- `release`: `n × m` array – `release[j][k]` is the release date of job *j* on machine *k*
- `setup`: `n × n × m` array – `setup[i][j][k]` is the setup time for job *j* after job *i* on machine *k* (0 if *i* = *j*)

**Example (5 jobs, 3 machines):** [75_3_5_H.json](./src/data/75_3_5_H.json)

---

## Output Format

Your solution must be JSON with:

- `"makespan"` – the makespan you computed
- `"schedule"` – mapping from machine IDs (`0` to `m-1`) to ordered lists of job IDs (`1` to `n`)

**Example output:**

```json
{
  "makespan": 1049,
  "schedule": {
    "0": [],
    "1": [5],
    "2": [2, 3, 1, 4]
  }
}
```

**Interpretation:**

- Machine 0: no jobs.
- Machine 1: job 5.
- Machine 2: jobs 2 → 3 → 1 → 4 in that order.

---

## Verifying Solutions

We provide a **Python checker** ([checker.py](./checker.py)) to ensure your schedule is:

- **Complete:** all jobs appear exactly once.
- **Valid:** each job is on a capable machine.
- **Feasible:** release dates and setup times are respected.
- **Correct:** reported makespan matches the computed one.

All solutions must pass the checker to be graded.

---

## Competition Instance

You will receive one **large competition instance**.  
Your task: **find the smallest makespan you can** using any methods you choose.

Possible approaches:

- **Exact:** CP (e.g., MiniZinc), MIP, SAT, ASP, etc.
- **Heuristics:** tabu search, simulated annealing, evolutionary algorithms, LNS, ILS, GA, etc.

**Instance file:** [357_15_146_H.json](./src/data/357_15_146_H.json)

---

## Solvers

Run from the `src/` directory with:

```bash
python main.py path/to/instance.json --solver <solver_name>
```

| Solver | Description |
|--------|-------------|
| **grasp** | GRASP: greedy randomized construction (RCL by completion time) + local search. |
| **sa** | Simulated annealing with geometric cooling and optional reheating; random-neighbor moves. |
| **es** | (μ+λ) evolution strategy with Rechenberg's 1/5 rule; mutation via random neighbor. |
| **ils** | Iterated local search: repeated (perturb → local search), keep best. |
| **ga** | Genetic algorithm: population, uniform crossover on machine assignment, mutation, (μ+λ) selection. |
| **as** | Ant system: ants build solutions with pheromone + heuristic, local search, pheromone update. |
| **mmas** | Max–min ant system: pheromone bounds, best-so-far update, periodic reinit. |
| **acs** | Ant colony system: exploitation/exploration (q0), local pheromone update, global best update. |
| **amts** | Ant multi-tour system: edge-usage penalty to encourage exploration. |
| **combined** | Runs GRASP, SA, ACS, and ES in sequence; switches after stagnation (default 60 s). |

Default solver is **combined**. Use `--time-limit N`, `--max-generations N`, `--gen-min-improvement`, or `--time-min-improvement` to control stopping (see `python main.py --help`).