from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import (
    TimeLimit, GenMinImprovement, TimeMinImprovement,
    MaxGenerations, TargetObjective,
)

import datetime
import time
import threading

current_date = datetime.datetime.now()
BASE_PATH = "src/results/" + current_date.strftime("%Y_%m_%d_%H_%M_%S")

print(f"Results will be saved to: {BASE_PATH}")

from solvers import (
    EvolutionStrategySolver, GeneticAlgorithmSolver, AntColonySystem, 
    AntMultiTourSystem, AntSystem, GraspSolver, ILSSolver, EasAntSystem, RankedAntSystem
)

import numpy as np

import os
import json
if not os.path.isdir(BASE_PATH):
    os.makedirs(BASE_PATH)

# Solver names and folder mappings
SOLVER_NAMES = {
    'evolution_strategy': 'evolution_strategy',
    'genetic_algorithm': 'genetic_algorithm',
    'grasp': 'grasp',
    'ils': 'ils',
    'ant_system': 'ant_system',
    'eas_ant_system': 'eas_ant_system',
    'ranked_ant_system': 'ranked_ant_system',
    'ant_colony_system': 'ant_colony_system',
    'ant_multi_tour_system': 'ant_multi_tour_system',
}

# Create solver-specific folders
for solver_folder in SOLVER_NAMES.values():
    solver_path = os.path.join(BASE_PATH, solver_folder)
    if not os.path.isdir(solver_path):
        os.makedirs(solver_path)

# Configuration
INSTANCE_PATH = "src/data/357_15_146_H.json"
TIMEOUT_PER_SOLVER = 10  # 30 minutes in seconds
N_ITERATIONS_PER_CONFIG = 1  # Number of times to repeat each configuration

# Criteria parameter ranges
CRITERIA_CONFIG = {
    'gen_min_improv_nr_pts': np.linspace(0.01, 0.99, 5),
    'gen_min_improv_window': np.linspace(20, 100, 5).astype("int"),
    'tm_min_improv_nr_pts': np.linspace(0.01, 0.99, 5),
    'tm_min_improv_window': np.linspace(20, 100, 5).astype("int"),
    'target_obj': np.linspace(500, 1500, 5).astype("int"),
    'tm_limit_seconds': np.linspace(5, 100, 5).astype("int"),
    'nr_max_generations': np.linspace(1, 100, 5).astype("int"),
}

# Algorithm-specific parameter ranges
ALGORITHM_PARAMS = {
    'evolution_strategy': {
        'mu': np.linspace(10, 100, 5).astype("int"),
        'lam': np.linspace(10, 50, 5).astype("int"),
        'c': np.linspace(0.1, 0.9, 5),
    },
    'genetic_algorithm': {
        'population_size': np.linspace(30, 300, 5).astype("int"),
        'offspring_per_generation': np.linspace(1, 5, 5).astype("int"),
        'mutation_strength': np.linspace(2, 10, 5).astype("int"),
    },
    'grasp': {
        'alpha': np.linspace(0.01, 0.99, 5),
    },
    'ils': {
        'perturbation_strength': np.linspace(1, 20, 5).astype("int"),
    },
    'ant_system': {
        'n_ants': np.linspace(20, 300, 5).astype("int"),
        'alpha': np.linspace(1, 5, 5),
        'beta': np.linspace(1, 5, 5),
        'rho': np.linspace(0.1, 0.9, 5),
        'q_ct': np.linspace(1, 5, 5),
    },
    'eas_ant_system': {
        'n_ants': np.linspace(20, 300, 5).astype("int"),
        'alpha': np.linspace(1, 5, 5),
        'beta': np.linspace(1, 5, 5),
        'rho': np.linspace(0.1, 0.9, 5),
        'q_ct': np.linspace(1, 5, 5),
        'sigma': np.linspace(1, 5, 5),
    },
    'ranked_ant_system': {
        'n_ants': np.linspace(20, 300, 5).astype("int"),
        'alpha': np.linspace(1, 5, 5),
        'beta': np.linspace(1, 5, 5),
        'rho': np.linspace(0.1, 0.9, 5),
        'q_ct': np.linspace(1, 5, 5),
    },
    'ant_colony_system': {
        'n_ants': np.linspace(20, 300, 5).astype("int"),
        'alpha': np.linspace(1, 5, 5),
        'beta': np.linspace(1, 5, 5),
        'rho': np.linspace(0.1, 0.9, 5),
        'q0': np.linspace(0.1, 0.9, 5),
        'local_decay': np.linspace(0.1, 0.9, 5),
    },
    'ant_multi_tour_system': {
        'n_ants': np.linspace(20, 300, 5).astype("int"),
        'alpha': np.linspace(1, 5, 5),
        'beta': np.linspace(1, 5, 5),
        'rho': np.linspace(0.1, 0.9, 5),
        'q_tours': np.linspace(1, 20, 5).astype("int"),
    },
}

instance = SchedulingInstance(INSTANCE_PATH)

# Global state for tracking best solution per solver
best_solution_state = {
    solver: {
        'best': None,
        'best_cost': np.inf,
        'history': [],
    }
    for solver in SOLVER_NAMES.keys()
}

# Global lists to accumulate all results text
all_results_text = []

# Lock for thread-safe updates
best_solution_lock = threading.Lock()
results_text_lock = threading.Lock()


def create_criteria(gen_min_improv_nr_pts, gen_min_improv_window, tm_min_improv_nr_pts, 
                   tm_min_improv_window, target_obj, tm_limit_seconds, nr_max_generations):
    """Factory function to create criteria with given parameters."""
    return [
        GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),
        TargetObjective(target=target_obj),
        TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),
        TimeLimit(seconds=tm_limit_seconds),
        MaxGenerations(n=nr_max_generations)
    ]


def update_best_solution(solver_name, solution, cost, history):
    """Thread-safe update of best solution for a specific solver."""
    with best_solution_lock:
        if cost < best_solution_state[solver_name]['best_cost']:
            best_solution_state[solver_name]['best'] = solution
            best_solution_state[solver_name]['best_cost'] = cost
            best_solution_state[solver_name]['history'] = history


def append_results_text(text):
    """Append results text to global results list."""
    with results_text_lock:
        all_results_text.append(text)


def tune_evolution_strategy(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Evolution Strategy with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['evolution_strategy']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for mu in params['mu']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for lam in params['lam']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                if mu < lam:
                    for c in params['c']:
                        if time.time() - start_time > timeout_seconds:
                            return text, True
                        
                        evol_solver = EvolutionStrategySolver(mu=mu, lam=lam, c=c, criteria=criteria)
                        e1, e2, e3 = evol_solver.solve(instance)
                        update_best_solution(solver_name, e1, e2, e3)
                        text += f"  mu={mu}, lam={lam}, c={c}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_genetic_algorithm(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Genetic Algorithm with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['genetic_algorithm']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for pop_size in params['population_size']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for off_per_gen in params['offspring_per_generation']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                for mut_str in params['mutation_strength']:
                    if time.time() - start_time > timeout_seconds:
                        return text, True
                    ga_solver = GeneticAlgorithmSolver(
                        population_size=pop_size,
                        offspring_per_generation=off_per_gen,
                        mutation_strength=mut_str,
                        criteria=criteria
                    )
                    e1, e2, e3 = ga_solver.solve(instance)
                    update_best_solution(solver_name, e1, e2, e3)
                    text += f"  pop_size={pop_size}, off_per_gen={off_per_gen}, mut_str={mut_str}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_grasp(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for GRASP with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['grasp']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for alpha in params['alpha']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            grasp_solver = GraspSolver(alpha=alpha, criteria=criteria)
            e1, e2, e3 = grasp_solver.solve(instance)
            update_best_solution(solver_name, e1, e2, e3)
            text += f"  alpha={alpha}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_ils(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Iterated Local Search with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['ils']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for perturbation_strength in params['perturbation_strength']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            ils_solver = ILSSolver(perturbation_strength=perturbation_strength, criteria=criteria)
            e1, e2, e3 = ils_solver.solve(instance)
            update_best_solution(solver_name, e1, e2, e3)
            text += f"  perturbation_strength={perturbation_strength}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_ant_system(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Ant System with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['ant_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for n_ants in params['n_ants']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for alpha in params['alpha']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                for beta in params['beta']:
                    if time.time() - start_time > timeout_seconds:
                        return text, True
                    for rho in params['rho']:
                        if time.time() - start_time > timeout_seconds:
                            return text, True
                        for q_ct in params['q_ct']:
                            if time.time() - start_time > timeout_seconds:
                                return text, True
                            as_solver = AntSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_ct=q_ct, criteria=criteria)
                            e1, e2, e3 = as_solver.solve(instance)
                            update_best_solution(solver_name, e1, e2, e3)
                            text += f"  n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_ct={q_ct}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_eas_ant_system(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for EAS Ant System with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['eas_ant_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for n_ants in params['n_ants']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for alpha in params['alpha']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                for beta in params['beta']:
                    if time.time() - start_time > timeout_seconds:
                        return text, True
                    for rho in params['rho']:
                        if time.time() - start_time > timeout_seconds:
                            return text, True
                        for q_ct in params['q_ct']:
                            if time.time() - start_time > timeout_seconds:
                                return text, True
                            for sigma in params['sigma']:
                                if time.time() - start_time > timeout_seconds:
                                    return text, True
                                eas_solver = EasAntSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_ct=q_ct, sigma=sigma, criteria=criteria)
                                e1, e2, e3 = eas_solver.solve(instance)
                                update_best_solution(solver_name, e1, e2, e3)
                                text += f"  n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_ct={q_ct}, sigma={sigma}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_ranked_ant_system(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Ranked Ant System with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['ranked_ant_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for n_ants in params['n_ants']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for alpha in params['alpha']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                for beta in params['beta']:
                    if time.time() - start_time > timeout_seconds:
                        return text, True
                    for rho in params['rho']:
                        if time.time() - start_time > timeout_seconds:
                            return text, True
                        for q_ct in params['q_ct']:
                            if time.time() - start_time > timeout_seconds:
                                return text, True
                            ras_solver = RankedAntSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_ct=q_ct, criteria=criteria)
                            e1, e2, e3 = ras_solver.solve(instance)
                            update_best_solution(solver_name, e1, e2, e3)
                            text += f"  n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_ct={q_ct}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_ant_colony_system(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Ant Colony System with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['ant_colony_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for n_ants in params['n_ants']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for alpha in params['alpha']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                for beta in params['beta']:
                    if time.time() - start_time > timeout_seconds:
                        return text, True
                    for rho in params['rho']:
                        if time.time() - start_time > timeout_seconds:
                            return text, True
                        for q0 in params['q0']:
                            if time.time() - start_time > timeout_seconds:
                                return text, True
                            for local_decay in params['local_decay']:
                                if time.time() - start_time > timeout_seconds:
                                    return text, True
                                acs_solver = AntColonySystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q0=q0, local_decay=local_decay, criteria=criteria)
                                e1, e2, e3 = acs_solver.solve(instance)
                                update_best_solution(solver_name, e1, e2, e3)
                                text += f"  n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q0={q0}, local_decay={local_decay}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def tune_ant_multi_tour_system(solver_name, criteria, start_time, timeout_seconds):
    """Parameter tuning for Ant Multi-Tour System with timeout check."""
    text = ""
    params = ALGORITHM_PARAMS['ant_multi_tour_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        if time.time() - start_time > timeout_seconds:
            return text, True
        for n_ants in params['n_ants']:
            if time.time() - start_time > timeout_seconds:
                return text, True
            for alpha in params['alpha']:
                if time.time() - start_time > timeout_seconds:
                    return text, True
                for beta in params['beta']:
                    if time.time() - start_time > timeout_seconds:
                        return text, True
                    for rho in params['rho']:
                        if time.time() - start_time > timeout_seconds:
                            return text, True
                        for q_tours in params['q_tours']:
                            if time.time() - start_time > timeout_seconds:
                                return text, True
                            amts_solver = AntMultiTourSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_tours=q_tours, criteria=criteria)
                            e1, e2, e3 = amts_solver.solve(instance)
                            update_best_solution(solver_name, e1, e2, e3)
                            text += f"  n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_tours={q_tours}, cost={e2}; best_cost={best_solution_state[solver_name]['best_cost']}\n"
    
    return text, False


def save_solver_best_solution(solver_name):
    """Save best solution for a specific solver to its folder."""
    if best_solution_state[solver_name]['best'] is not None:
        best = best_solution_state[solver_name]['best']
        solver_folder = os.path.join(BASE_PATH, SOLVER_NAMES[solver_name])
        
        json_best_sol = '{'
        json_best_sol += f'"makespan": {best._makespan}, '
        json_best_sol += '"schedule": {'
        for key in best.schedule.keys():
            if key != list(best.schedule.keys())[-1]:
                json_best_sol += f'"{key}": {best.schedule[key]}, '
            else:
                json_best_sol += f'"{key}": {best.schedule[key]}'
        json_best_sol += '}}'

        with open(os.path.join(solver_folder, "best_solution.json"), "w") as f:
            json.dump(json_best_sol, f)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(best_solution_state[solver_name]['history'], linewidth=1, label=f'Best (makespan: {best._makespan})')
        plt.xlabel('Generation')
        plt.ylabel('Best Makespan')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(solver_folder, "best_solution_convergence.png"))
        plt.close()


# Main parameter tuning loop
print("Starting parameter tuning for all algorithms...")
print(f"Criteria configuration: {CRITERIA_CONFIG}")
print(f"Timeout per solver: {TIMEOUT_PER_SOLVER} seconds (30 minutes)\n")

# List of solvers and their tuning functions
SOLVER_TUNING_FUNCTIONS = {
    'evolution_strategy': tune_evolution_strategy,
    'genetic_algorithm': tune_genetic_algorithm,
    'grasp': tune_grasp,
    'ils': tune_ils,
    'ant_system': tune_ant_system,
    'eas_ant_system': tune_eas_ant_system,
    'ranked_ant_system': tune_ranked_ant_system,
    'ant_colony_system': tune_ant_colony_system,
    'ant_multi_tour_system': tune_ant_multi_tour_system,
}

# Iterate over solvers
for solver_name, tune_function in SOLVER_TUNING_FUNCTIONS.items():
    print(f"\n{'='*60}")
    print(f"Starting tuning for {solver_name.upper()}")
    print(f"{'='*60}")
    
    solver_text = f"\n{'='*60}\n{solver_name.upper()}\n{'='*60}\n"
    
    solver_start_time = time.time()
    solver_timed_out = False
    
    # Iterate over criteria sets
    for gen_min_improv_nr_pts in CRITERIA_CONFIG['gen_min_improv_nr_pts']:
        if solver_timed_out:
            break
        for gen_min_improv_window in CRITERIA_CONFIG['gen_min_improv_window']:
            if solver_timed_out:
                break
            for tm_min_improv_nr_pts in CRITERIA_CONFIG['tm_min_improv_nr_pts']:
                if solver_timed_out:
                    break
                for tm_min_improv_window in CRITERIA_CONFIG['tm_min_improv_window']:
                    if solver_timed_out:
                        break
                    for target_obj in CRITERIA_CONFIG['target_obj']:
                        if solver_timed_out:
                            break
                        for tm_limit_seconds in CRITERIA_CONFIG['tm_limit_seconds']:
                            if solver_timed_out:
                                break
                            for nr_max_generations in CRITERIA_CONFIG['nr_max_generations']:
                                # Check timeout before running tuning
                                if time.time() - solver_start_time > TIMEOUT_PER_SOLVER:
                                    solver_timed_out = True
                                    print(f"[TIMEOUT] {solver_name} tuning stopped after {time.time() - solver_start_time:.1f} seconds")
                                    break
                                
                                criteria = create_criteria(
                                    gen_min_improv_nr_pts, gen_min_improv_window,
                                    tm_min_improv_nr_pts, tm_min_improv_window,
                                    target_obj, tm_limit_seconds, nr_max_generations
                                )
                                
                                solver_text += f"\nCRITERIA: {criteria}\n"
                                
                                # Call the tuning function
                                tuning_text, timed_out = tune_function(
                                    solver_name, criteria, 
                                    solver_start_time, TIMEOUT_PER_SOLVER
                                )
                                solver_text += tuning_text
                                
                                if timed_out:
                                    solver_timed_out = True
                                    break
    
    elapsed_time = time.time() - solver_start_time
    if solver_timed_out:
        solver_text += f"\n[TIMEOUT] {solver_name} tuning stopped after {elapsed_time:.1f} seconds (30 min limit reached).\n"
        solver_text += f"[INFO] Best cost found for {solver_name}: {best_solution_state[solver_name]['best_cost']}\n"
    else:
        solver_text += f"\n[COMPLETED] {solver_name} tuning finished in {elapsed_time:.1f} seconds.\n"
        solver_text += f"[INFO] Best cost found for {solver_name}: {best_solution_state[solver_name]['best_cost']}\n"
    
    # Append to global results
    append_results_text(solver_text)
    
    # Save best solution for this solver
    save_solver_best_solution(solver_name)
    
    print(f"[DONE] {solver_name} - Best cost: {best_solution_state[solver_name]['best_cost']}")

print("\n[INFO] Parameter tuning completed!")

# Save all results to a single file
with open(os.path.join(BASE_PATH, "parameter_tuning.txt"), "w") as f:
    f.write("".join(all_results_text))

print(f"[INFO] All results saved to: {BASE_PATH}/parameter_tuning.txt")

# Print overall summary
print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)
best_overall_cost = np.inf
best_overall_solver = None
for solver_name in SOLVER_NAMES.keys():
    cost = best_solution_state[solver_name]['best_cost']
    print(f"{solver_name:30s}: {cost}")
    if cost < best_overall_cost:
        best_overall_cost = cost
        best_overall_solver = solver_name

print("="*60)
print(f"Best solver: {best_overall_solver} with cost: {best_overall_cost}")
print(f"Results saved to: {BASE_PATH}")