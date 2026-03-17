# %% [markdown]
# # Preprocessing
# 
# Important imports

# %%
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# np.random.seed(123) # REMOVEME

# %% [markdown]
# ## Loading dataset
# 
# ### Loading string data
# 
# **Parameters:**
# - `data_string`: A string containing the data in the format:
#   - First line: number of jobs and number of machines (e.g., "10 10")
#   - Following lines: for each job, alternating machine numbers and processing times
# 
# **Returns:**
# - `np.ndarray`: A numpy array of shape `(num_jobs, num_machines, 2)` where the last dimension is `[machine, time]` pairs.

# %%
def load_from_string(data_string):
    lines = data_string.strip().split('\n')
    
    num_jobs, num_machines = map(int, lines[0].split())
    
    data = []
    for i in range(1, num_jobs + 1):
        nums = list(map(int, lines[i].split()))
        pairs = [[nums[j], nums[j+1]] for j in range(0, len(nums), 2)]
        data.append(pairs)
    
    return np.array(data)

# %% [markdown]
# ### Loading txt file
# 
# **Parameters:**
# - `key`: The instance key (e.g., 'abz5', 'ft06', 'la01') to load from the dataset.
# - `filepath`: Path to the jobshop.txt file. Default is `./data/jobshop.txt`.
# 
# **Returns:**
# - `np.ndarray`: A numpy array of shape `(num_jobs, num_machines, 2)` where the last dimension is `[machine, time]` pairs.

# %%
def load_from_dataset(key, filepath=os.path.join('.', 'data', 'jobshop.txt')):    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    instance_marker = f'instance {key}'
    start_idx = None
    
    for i, line in enumerate(lines):
        if instance_marker in line:
            start_idx = i
            break
        
    
    if start_idx is None:
        raise ValueError(f"Instance '{key}' not found in the dataset")
    
    dimensions_line = None
    data_start_idx = None
    
    for i in range(start_idx, min(start_idx + 10, len(lines))):
        line = lines[i].strip()
        if line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
                dimensions_line = line
                data_start_idx = i + 1
                break
    
    if dimensions_line is None:
        raise ValueError(f"Could not parse dimensions for instance '{key}'")
    
    num_jobs, _ = map(int, dimensions_line.split()[:2])
    
    data = []
    for i in range(data_start_idx, data_start_idx + num_jobs):
        if i < len(lines):
            line = lines[i].strip()
            if line and not line.startswith('+'):
                nums = list(map(int, line.split()))
                pairs = [[nums[j], nums[j+1]] for j in range(0, len(nums), 2)]
                data.append(pairs)
    
    if len(data) != num_jobs:
        raise ValueError(f"Expected {num_jobs} jobs, but found {len(data)}")
    
    return np.array(data)

# %% [markdown]
# ---
# 
# # Algorithms

# %% [markdown]
# ## Simulated Annealing
# 
# ### Generating an initial random solution
# 
# As we know the number of jobs and of machines, we can proceed to create an initial random solution.

# %%
def generate_random_solution(num_jobs, num_machines):
    # This function generates a random candidate solution for the job shop scheduling problem.
    steps_job = np.zeros(num_jobs, dtype=int)
    candidate_solution = np.zeros(num_jobs * num_machines, dtype=int) # This array will hold the sequence of job assignments for the candidate solution.
    counter = 0
    while (steps_job != np.repeat(num_machines, num_jobs)).any(): # This loop continues until all jobs have been assigned to all machines.
        random_job = np.random.randint(num_jobs)
        while steps_job[random_job] == num_machines:
            random_job = np.random.randint(num_jobs)
        candidate_solution[counter] = random_job # Assign the random job to the current position in the candidate solution.
        counter += 1
        steps_job[random_job] += 1
    return candidate_solution   # Return the generated random candidate solution.

# %% [markdown]
# ### Calculate the makespan for a candidate solution
# 
# With this function we calculate how much time is spent by a candidate solution to schedule all the assigned tasks.

# %%
def compute_sa_makespan(candidate_solution, result_from_dataset):
    num_jobs, num_machines, _ = result_from_dataset.shape
    steps_job = np.zeros(num_jobs, dtype=int)
    steps_machines = np.zeros(num_machines, dtype=int)
    machines = np.array(np.repeat([np.repeat(np.zeros(shape=(1, 4)), num_jobs, axis=0)], num_machines, axis=0), dtype=int)
    time = 0
    for job in candidate_solution:
        machine, processing_time = result_from_dataset[job, steps_job[job]]
        if steps_job[job] > 0:
            machine_used_for_job, _ = result_from_dataset[job, steps_job[job] - 1]
            if machines[machine_used_for_job][steps_machines[machine_used_for_job] - 1][3] > time:
                # if a machine has been employed already for the same job and is not available at the current time
                # the time is updated to the time when the machine will be available
                time = machines[machine_used_for_job][steps_machines[machine_used_for_job] - 1][3]
        if machines[machine][steps_machines[machine] - 1][3] > time:
            # if a machine is used already for another job and is not available at the current time
            # the time is updated to the time when the machine will be available
            time = machines[machine][steps_machines[machine] - 1][3]
        machines[machine][steps_machines[machine]] = [job, steps_job[job], time, time + processing_time]
        steps_job[job] += 1
        steps_machines[machine] += 1
    return time

# %% [markdown]
# ### Tweak a candidate solution
# 
# With this function the position of two random indexes of the candidate solution are swapped, in order to obtain a new candidate solution.

# %%
def tweak_candidate_solution(candidate_solution):
    index_one, index_two = np.random.randint(len(candidate_solution)), np.random.randint(len(candidate_solution))
    while index_one == index_two:
        index_two = np.random.randint(len(candidate_solution))
    candidate_solution[index_one], candidate_solution[index_two] = candidate_solution[index_two], candidate_solution[index_one] 
    return candidate_solution

# %% [markdown]
# ### Main loop
# 
# Why this function has been chosen as the trajectory based function?
# > It is flexible, and allows us to get from high exploration at the beginning to high exploitation at the end.

# %%
def simulated_annealing(input_solution, data, t = 120, target_time = 60, max_elapsed_time = 86400.):
    start_time = time.time()
    candidate_solution = input_solution.copy()
    best_solution = candidate_solution.copy()
    best_makespan = compute_sa_makespan(best_solution, data)
    history = [best_makespan]
    time_history = [(0.0, best_makespan)]
    while t > 0 and best_makespan > target_time and time.time() - start_time < max_elapsed_time: 
        tweaked_solution = tweak_candidate_solution(candidate_solution.copy())
        tweaked_makespan = compute_sa_makespan(tweaked_solution, data)
        candidate_makespan = compute_sa_makespan(candidate_solution, data)
        quality_temp_ratio = -(tweaked_makespan - candidate_makespan) / t
        
        if tweaked_makespan < candidate_makespan or np.random.rand() < np.exp(quality_temp_ratio):
            candidate_solution = tweaked_solution

        if candidate_makespan < best_makespan:
            best_solution = candidate_solution.copy()
            best_makespan = candidate_makespan
        history.append(best_makespan)
        time_history.append((time.time() - start_time, best_makespan))
        t -= 1

    elapsed = time.time() - start_time
    return best_solution, best_makespan, history, time_history, elapsed

# %% [markdown]
# ### Testing functions for dataset abz5

# %%
result_from_dataset = load_from_dataset('abz5')
num_jobs, num_machines, _ = result_from_dataset.shape
candidate_solution = generate_random_solution(num_jobs, num_machines)
print(candidate_solution, compute_sa_makespan(candidate_solution, result_from_dataset))
best_solution, makespan, _, _, elapsed = simulated_annealing(candidate_solution, result_from_dataset)
print(best_solution, makespan)

# %% [markdown]
# ## Evolution Strategy (μ + λ) 
# 
# ### Data representation
# 
# An individual is a **permutation** of length `num_jobs * num_machines`, where each job index appears exactly `num_machines` times.

# %%
def create_individual(num_jobs, num_machines):
    individual = np.repeat(np.arange(num_jobs), num_machines)
    np.random.shuffle(individual)
    return individual

def create_population(size, num_jobs, num_machines):
    return [create_individual(num_jobs, num_machines) for _ in range(size)]

# %% [markdown]
# ### Fitness function
# 
# The fitness function is **makespan** -> total completion time of all jobs. Since we are trying to minimize completition time -> **lower makespan = better fitness**.

# %%
def compute_es_makespan(individual, data):
    num_jobs, num_machines, _ = data.shape
    
    job_op_count = np.zeros(num_jobs, dtype=int)
    job_end_time = np.zeros(num_jobs, dtype=int)
    machine_end_time = np.zeros(num_machines, dtype=int)
    
    for gene in individual:
        j = gene
        k = job_op_count[j]
        machine_id, proc_time = data[j, k, :]
        
        start = max(job_end_time[j], machine_end_time[machine_id])
        end = start + proc_time
        
        job_end_time[j] = end
        machine_end_time[machine_id] = end
        job_op_count[j] += 1
    
    return np.max(job_end_time)

# %% [markdown]
# ### Swap Mutation
# 
# For mutation we pick two random positions and swap their values. This also ensures that each job still appears the correct number of times  
# 
# The `step_size` controls mutation strength by determining **how many swaps** to perform. Larger step size means more mutations (more exploration), while smaller step size results in more exploitation

# %%
def mutate(individual, step_size):
    child = individual.copy()
    num_swaps = max(1, int(step_size + 0.5))
    length = len(child)
    
    for _ in range(num_swaps):
        i, j = np.random.randint(0, length, size=2)
        child[i], child[j] = child[j], child[i]
    
    return child

# %% [markdown]
# ### Self-Adaptation
# 
# **Rechenberg's 1/5 Success Rule** adapts the mutation strenght based on success rate:
# 
# - **Success rate > 1/5**: Too much exploitation - more than 1/5 offspring are better than the parents -> **increase** the mutation strength  
# - **Success rate < 1/5**: Too much exploration - more than 1/5 offspring are better than the parents -> **decrease** the mutation strength  
# - **Success rate = 1/5**: don't change anything  

# %%
def self_adaptation(sigma, success_count, total_offspring, c, sigma_min=1.0, sigma_max=float('inf')):
    if total_offspring == 0:
        return sigma
    
    success_rate = success_count / total_offspring
    
    if success_rate > 0.2:
        sigma = sigma / c
    elif success_rate < 0.2:
        sigma = sigma * c

    return np.clip(sigma, sigma_min, sigma_max)

# %% [markdown]
# ### Main loop
# 
# Implementation of **Algorithm 19** from the Essentials of Metaheuristics by Sean Luke

# %%
def evolution_strategy(data, mu, lam, max_generations, c=0.85, max_elapsed_time = 86400.):
    assert c < 1. and c > 0., f"c must be in range (0, 1)"  
    start_time = time.time()
    num_jobs, num_machines, _ = data.shape
    genome_length = num_jobs * num_machines
    
    # Initial mutation strength
    sigma = genome_length / 10
    sigma_min = 1.0
    sigma_max = genome_length / 2
    
    # (mu + lambda) strategy: initial population includes both parents and offspring slots
    population = create_population(mu + lam, num_jobs, num_machines)
    fitness = [compute_es_makespan(ind, data) for ind in population]
    
    # Track the globally best individual across all generations
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_makespan = fitness[best_idx]
    history = [best_makespan]
    time_history = [(0.0, best_makespan)]
    
    for gen in range(max_generations):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_elapsed_time:
            break
        # Selection: keep only the mu best individuals as parents (truncation selection)
        sorted_indices = np.argsort(fitness)[:mu]
        parents = [population[i] for i in sorted_indices]
        parent_fitness = [fitness[i] for i in sorted_indices]
        
        # Each parent produces an equal share of offspring (lambda / mu each)
        offspring_per_parent = lam // mu
        offspring = []
        offspring_fitness = []
        success_count = 0
        
        for p_idx in range(mu):
            for _ in range(offspring_per_parent):
                child = mutate(parents[p_idx], sigma)
                child_fit = compute_es_makespan(child, data)
                offspring.append(child)
                offspring_fitness.append(child_fit)
                # Count successes for Rechenberg's 1/5 rule
                if child_fit < parent_fitness[p_idx]:
                    success_count += 1
        
        # Distribute remaining offspring (when lambda is not evenly divisible by mu)
        remainder = lam - offspring_per_parent * mu
        for r in range(remainder):
            p_idx = r % mu
            child = mutate(parents[p_idx], sigma)
            child_fit = compute_es_makespan(child, data)
            offspring.append(child)
            offspring_fitness.append(child_fit)
            if child_fit < parent_fitness[p_idx]:
                success_count += 1
        
        # (mu + lambda): next generation is parents AND offspring combined
        population = parents + offspring
        fitness = parent_fitness + offspring_fitness
        
        # Adapt mutation strength using Rechenberg's 1/5 success rule
        total_offspring = len(offspring)
        sigma = self_adaptation(sigma, success_count, total_offspring, c, sigma_min, sigma_max)
        
        # Update global best if this generation produced a better individual
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_makespan:
            best_individual = population[gen_best_idx].copy()
            best_makespan = fitness[gen_best_idx]
        
        history.append(best_makespan)
        time_history.append((time.time() - start_time, best_makespan))
    
    elapsed = time.time() - start_time
    return best_individual, best_makespan, history, time_history, elapsed

# %% [markdown]
# ---
# 
# # Hackathon

# %%
hackathon_str = """
20 20
 16 34 17 38  0 21  6 15 15 42  8 17  7 41 18 10 10 26 11 24  1 31 19 25 14 31 13 33  4 35  9 30  3 16 12 16  5 30  2 13  
  5 41 11 33  6 15 16 38  0 40 14 38  3 37  1 20 13 22  4 34  7 16 17 39  9 15  2 19 10 36 12 39 18 26  8 19 15 39 19 34  
 17 34  1 12 16 10  7 47 13 28 15 27  0 19  6 34 19 33 12 40  9 37 14 24  8 15 10 34  2 44  3 37 18 22 11 31  4 39  5 26  
  5 48  7 46 16 47 10 45 14 15  8 25  0 34  3 24 12 35 18 15  2 48 13 19 11 10  1 48 17 16 15 28  4 18  6 17  9 44 19 41  
 12 47  3 23  9 48 16 45 14 39  6 42  8 32 15 11 13 16  5 14 11 19  1 46 19 10 10 17  7 41  2 47 17 32  4 17  0 21 18 17  
 18 14 16 20  1 18 12 14 13 10  6 16  5 24  4 18  0 24 11 18 15 42 19 13  3 23 14 40  9 48  8 12  2 24 10 23  7 45 17 30  
  0 27 12 15  4 26 13 19 17 14  5 49  7 16 18 28 16 16  8 20  9 36  2 21 14 30  3 36  1 17 15 22  6 43 11 32 10 23 19 17  
  0 32 16 15 17 12  7 46  3 37 18 43 11 40 13 43  9 48  4 36 15 24  8 25  1 33 14 32  5 26  6 37 12 24 10 24  2 15 19 22  
 10 34  6 33 15 25  8 46  0 20 18 33  4 19 13 45  2 47  1 32  3 12 11 29 16 29  5 46 12 17  7 48 14 39 17 40 19 41  9 37  
 13 26  3 47  5 44  6 49  1 22 17 12 10 28 19 36  9 27  4 25 14 48  7 11 16 49 12 24 11 48  2 19  0 47 18 49  8 46 15 36  
 13 23 18 48 14 15  0 42  3 36  8 15  6 32 10 18  1 45 15 23 11 45  2 13 17 21 12 32  7 44  5 25 19 34 16 22  9 11  4 43  
 17 37  7 49 15 45  2 28  9 15  8 35 12 29 13 44  1 26  4 25  5 30  3 39  0 15 14 28 18 23  6 42 11 33 16 45 10 10 19 20  
  0 10  6 37  3 15 13 13 10 11  2 49  1 28 14 28 15 13  8 29 12 21 16 32 11 21  4 48  5 11 17 26  9 33 18 22  7 21 19 49  
 18 38  0 41  4 30 13 43  6 11  2 43 14 27  3 26  9 30 15 19 16 36  1 31 17 47  5 41 10 34  8 40 12 32  7 13 11 18 19 27  
  6 24  5 30  7 10 10 35  8 28 16 43 19 12  9 44 15 15  3 15  2 35 18 43  0 38  4 16  1 29 17 40 14 49 13 38 12 16 11 30  
  3 48  6 35 13 43  2 37 17 18  5 27  9 27  7 41  1 22 15 28 16 18 10 37 18 48  4 10  8 14 11 18 14 43  0 48 12 12 19 49  
  0 13 13 38  7 34  6 42  1 36  5 45 18 24  8 35 14 26 19 30 12 47 16 24 11 47  4 40 10 43  3 16 15 10  2 12  9 39 17 22  
 16 30 13 47 19 49  8 20  4 40  3 46 17 21 14 33  6 44  7 23  9 24  0 48 10 43 15 41  2 32  5 29 11 36  1 38 12 47 18 12  
 13 10  5 36 12 18 16 48  0 27 14 43 10 46  6 27  7 46 19 35 11 31  2 18  8 24  3 23 17 29 18 14  9 19  1 40 15 38  4 13  
  9 45 16 44  0 43 17 31 14 35 13 17 12 42  3 14 18 37 10 39  6 48  7 38 15 26  4 49  2 28 11 35  1 42  5 24  8 44 19 38
"""
data = load_from_string(hackathon_str)
best_makespan = 1150
c_min = 0.75
c_max = 0.95
mu_min = 15
mu_max = 30
children_c_min = 1.
children_c_max = 2.
t = 15.
for i in range(100):
  # t = np.random.uniform(t_min, t_max)
  c = np.random.uniform(c_min, c_max)
  mu_val = np.random.randint(mu_min, mu_max)
  children_c = np.random.uniform(children_c_min, children_c_max)
  for mu, lam in [(mu_val, int(mu_val*children_c))]:
      best_individual, makespan, history, time_history, elapsed = evolution_strategy(data, mu=mu, lam=lam, c=c, max_generations=10**9, max_elapsed_time=t)
      print(f"{i} - mu: {mu}, lam: {lam}, c: {c}, makespan: {makespan}")
      if makespan < best_makespan:
        best_makespan = makespan
        # t_min *= 0.75
        # t_max *= 1.23  
        c_min = max(0.5, c_min * 0.75)
        c_max = min(0.99, c_max * 1.25)
        mu_min *= 0.75
        mu_max *= 1.25
        children_c_min *= 0.75
        children_c_max *= 1.25
        print(best_individual)
        print()
        print(f"-"* 100)
        print()



# %% [markdown]
# ---
# 
# # Visualizations

# %% [markdown]
# ## Datasets performance
# 
# ### Max time performance comparation

# %%
for dataset_name, max_time in [('ft06', 2.), ('abz5', 4.), ('la36', 8.), ('swv10', 12.)]:
    data = load_from_dataset(dataset_name)
    num_jobs, num_machines, _ = data.shape
    label = f"{dataset_name} ({num_jobs} x {num_machines})"

    # ES
    _, es_makespan, _, es_time_hist, es_elapsed = evolution_strategy(data, mu=10, lam=70, max_generations=5000, max_elapsed_time=max_time)
    print(f"ES - {label} - Best makespan: {es_makespan}, elapsed: {es_elapsed:.2f}s")

    # SA
    candidate_solution = generate_random_solution(num_jobs, num_machines)
    _, sa_makespan, _, sa_time_hist, sa_elapsed = simulated_annealing(candidate_solution, data, t=10**9, target_time=0, max_elapsed_time=max_time)
    print(f"SA - {label} - Best makespan: {sa_makespan}, elapsed: {sa_elapsed:.2f}s")

    # Convert to % of time budget
    es_pct = [t / max_time * 100 for t, _ in es_time_hist]
    es_vals = [m for _, m in es_time_hist]
    sa_pct = [t / max_time * 100 for t, _ in sa_time_hist]
    sa_vals = [m for _, m in sa_time_hist]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(es_pct, es_vals, linewidth=1, label=f'ES (best: {es_makespan})')
    plt.plot(sa_pct, sa_vals, linewidth=1, label=f'SA (best: {sa_makespan})')
    plt.xlabel('Elapsed time (%)')
    plt.ylabel('Best Makespan')
    plt.title(f'{label} - max time: {max_time}s')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

# %% [markdown]
# 
# ### Number of generations performance comparation
# 
# Running multiple datasets of different sizes to observe results and computation time

# %%
es_full_history = []
sa_full_history = []

GENERATIONS = 500

for dataset_name in ['ft06', 'abz5', 'la36', 'swv10']:
    data = load_from_dataset(dataset_name)
    num_jobs, num_machines, _ = data.shape
    label = f"{dataset_name} ({num_jobs} x {num_machines})"

    # ES
    _, es_makespan, es_history, _, es_time = evolution_strategy(data, mu=10, lam=70, max_generations=GENERATIONS)
    print(f"ES - {label} - Best makespan found: {es_makespan}, finished in {es_time:.2f} seconds")
    es_full_history.append((es_history, label))

    # SA
    candidate_solution = generate_random_solution(num_jobs, num_machines)
    _, sa_makespan, sa_history, _, sa_time = simulated_annealing(candidate_solution, data, t=GENERATIONS, target_time=0)
    print(f"SA - {label} - Best makespan found: {sa_makespan}, finished in {sa_time:.2f} seconds")
    sa_full_history.append((sa_history, label))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(es_history, linewidth=1, label=f'ES (best: {es_makespan})')
    plt.plot(sa_history, linewidth=1, label=f'SA (best: {sa_makespan})')
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title(f'{label} convergence')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

# %% [markdown]
# ## Normalised plot
# 
# ### Evolution Strategy
# 
# Multiple runs showed that smaller datasets (usually) need much less generations than for larger datasets to get better results. Results were not as stable as expected though, sometimes even smallest dataset took much more generations to get best result.

# %%
plt.figure(figsize=(10, 5))
for history, label in es_full_history:
    arr = np.array(history)
    min_val, max_val = arr.min(), arr.max()
    norm = (arr - min_val) / (max_val - min_val)
    plt.plot(norm, label=label)
plt.xlabel('Generation')
plt.ylabel('Best normalised Makespan')
plt.title('ES Convergence - normalised datasets')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Simulated Annealing

# %%
plt.figure(figsize=(10, 5))
for history, label in sa_full_history:
    arr = np.array(history)
    min_val, max_val = arr.min(), arr.max()
    norm = (arr - min_val) / (max_val - min_val)
    plt.plot(norm, label=label)
plt.xlabel('Generation')
plt.ylabel('Best normalised Makespan')
plt.title('SA Convergence - normalised datasets')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# %% [markdown]
# ---
# 
# # Results
# 
# ... SA is much faster, but much less accurate in small generation size  
# ... would be worth it to have "elapsed time" stoppage criteria to observe best_makespan in specified time instead of number of generations


