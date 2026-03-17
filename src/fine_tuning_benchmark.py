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
solver_type = "evol"

PATH = "src/results/" + current_date.strftime("%Y_%m_%d_%H_%M_%S") + "/" + solver_type

print(PATH)

from solvers import (
    EvolutionStrategySolver, GeneticAlgorithmSolver, AntColonySystem, 
    AntMultiTourSystem, AntSystem, GraspSolver, ILSSolver, EasAntSystem, RankedAntSystem
)

import numpy as np

import os
if os.path.isdir(PATH) == False:
    os.makedirs(PATH)

# Configuration
INSTANCE_PATH = "src/data/357_15_146_H.json"
TIMEOUT_PER_SOLVER = 1800  # 30 minutes in seconds
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

# Global state for tracking best solution
best_solution_state = {
    'best': None,
    'best_cost': np.inf,
    'history': [],
}

# Lock for thread-safe updates
best_solution_lock = threading.Lock()


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


def update_best_solution(solution, cost, history):
    """Thread-safe update of best solution."""
    with best_solution_lock:
        if cost < best_solution_state['best_cost']:
            best_solution_state['best'] = solution
            best_solution_state['best_cost'] = cost
            best_solution_state['history'] = history


def save_results(filename, content):
    """Append results to file."""
    with open(PATH + "/" + filename, "a") as f:
        f.write(content)


def tune_evolution_strategy(criteria):
    """Parameter tuning for Evolution Strategy."""
    text = ""
    params = ALGORITHM_PARAMS['evolution_strategy']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for mu in params['mu']:
            for lam in params['lam']:
                if mu < lam:
                    for c in params['c']:
                        evol_solver = EvolutionStrategySolver(mu=mu, lam=lam, c=c, criteria=criteria)
                        e1, e2, e3 = evol_solver.solve(instance)
                        update_best_solution(e1, e2, e3)
                        text += f"mu={mu}, lam={lam}, c={c}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_genetic_algorithm(criteria):
    """Parameter tuning for Genetic Algorithm."""
    text = ""
    params = ALGORITHM_PARAMS['genetic_algorithm']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for pop_size in params['population_size']:
            for off_per_gen in params['offspring_per_generation']:
                for mut_str in params['mutation_strength']:
                    ga_solver = GeneticAlgorithmSolver(
                        population_size=pop_size,
                        offspring_per_generation=off_per_gen,
                        mutation_strength=mut_str,
                        criteria=criteria
                    )
                    e1, e2, e3 = ga_solver.solve(instance)
                    update_best_solution(e1, e2, e3)
                    text += f"pop_size={pop_size}, off_per_gen={off_per_gen}, mut_str={mut_str}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_grasp(criteria):
    """Parameter tuning for GRASP."""
    text = ""
    params = ALGORITHM_PARAMS['grasp']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for alpha in params['alpha']:
            grasp_solver = GraspSolver(alpha=alpha, criteria=criteria)
            e1, e2, e3 = grasp_solver.solve(instance)
            update_best_solution(e1, e2, e3)
            text += f"alpha={alpha}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_ils(criteria):
    """Parameter tuning for Iterated Local Search."""
    text = ""
    params = ALGORITHM_PARAMS['ils']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for perturbation_strength in params['perturbation_strength']:
            ils_solver = ILSSolver(perturbation_strength=perturbation_strength, criteria=criteria)
            e1, e2, e3 = ils_solver.solve(instance)
            update_best_solution(e1, e2, e3)
            text += f"perturbation_strength={perturbation_strength}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_ant_system(criteria):
    """Parameter tuning for Ant System."""
    text = ""
    params = ALGORITHM_PARAMS['ant_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for n_ants in params['n_ants']:
            for alpha in params['alpha']:
                for beta in params['beta']:
                    for rho in params['rho']:
                        for q_ct in params['q_ct']:
                            as_solver = AntSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_ct=q_ct, criteria=criteria)
                            e1, e2, e3 = as_solver.solve(instance)
                            update_best_solution(e1, e2, e3)
                            text += f"n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_ct={q_ct}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_eas_ant_system(criteria):
    """Parameter tuning for EAS Ant System."""
    text = ""
    params = ALGORITHM_PARAMS['eas_ant_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for n_ants in params['n_ants']:
            for alpha in params['alpha']:
                for beta in params['beta']:
                    for rho in params['rho']:
                        for q_ct in params['q_ct']:
                            for sigma in params['sigma']:
                                eas_solver = EasAntSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_ct=q_ct, sigma=sigma, criteria=criteria)
                                e1, e2, e3 = eas_solver.solve(instance)
                                update_best_solution(e1, e2, e3)
                                text += f"n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_ct={q_ct}, sigma={sigma}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_ranked_ant_system(criteria):
    """Parameter tuning for Ranked Ant System."""
    text = ""
    params = ALGORITHM_PARAMS['ranked_ant_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for n_ants in params['n_ants']:
            for alpha in params['alpha']:
                for beta in params['beta']:
                    for rho in params['rho']:
                        for q_ct in params['q_ct']:
                            ras_solver = RankedAntSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_ct=q_ct, criteria=criteria)
                            e1, e2, e3 = ras_solver.solve(instance)
                            update_best_solution(e1, e2, e3)
                            text += f"n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_ct={q_ct}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_ant_colony_system(criteria):
    """Parameter tuning for Ant Colony System."""
    text = ""
    params = ALGORITHM_PARAMS['ant_colony_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for n_ants in params['n_ants']:
            for alpha in params['alpha']:
                for beta in params['beta']:
                    for rho in params['rho']:
                        for q0 in params['q0']:
                            for local_decay in params['local_decay']:
                                acs_solver = AntColonySystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q0=q0, local_decay=local_decay, criteria=criteria)
                                e1, e2, e3 = acs_solver.solve(instance)
                                update_best_solution(e1, e2, e3)
                                text += f"n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q0={q0}, local_decay={local_decay}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def tune_ant_multi_tour_system(criteria):
    """Parameter tuning for Ant Multi-Tour System."""
    text = ""
    params = ALGORITHM_PARAMS['ant_multi_tour_system']
    
    for iter in range(N_ITERATIONS_PER_CONFIG):
        for n_ants in params['n_ants']:
            for alpha in params['alpha']:
                for beta in params['beta']:
                    for rho in params['rho']:
                        for q_tours in params['q_tours']:
                            amts_solver = AntMultiTourSystem(n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, q_tours=q_tours, criteria=criteria)
                            e1, e2, e3 = amts_solver.solve(instance)
                            update_best_solution(e1, e2, e3)
                            text += f"n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}, q_tours={q_tours}, cost={e2}; best_cost={best_solution_state['best_cost']}\n"
    
    return text


def run_algorithm_tuning_with_timeout(algorithm_name, tune_function, criteria, timeout_seconds):
    """Run algorithm tuning with a timeout."""
    result_text = ""
    start_time = time.time()
    
    class TuningThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self.result = ""
            self.daemon = False
        
        def run(self):
            self.result = tune_function(criteria)
    
    thread = TuningThread()
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    elapsed_time = time.time() - start_time
    result_text = thread.result if hasattr(thread, 'result') else ""
    
    if thread.is_alive():
        result_text += f"\n[TIMEOUT] {algorithm_name} tuning stopped after {elapsed_time:.1f} seconds (30 min limit reached).\n"
        result_text += f"[INFO] Best cost so far: {best_solution_state['best_cost']}\n"
        print(f"[TIMEOUT] {algorithm_name} tuning stopped after {elapsed_time:.1f} seconds")
    else:
        result_text += f"\n[COMPLETED] {algorithm_name} tuning finished in {elapsed_time:.1f} seconds.\n"
        print(f"[COMPLETED] {algorithm_name} tuning finished in {elapsed_time:.1f} seconds")
    
    return result_text


# Main parameter tuning loop
print("Starting parameter tuning for all algorithms...")
print(f"Criteria configuration: {CRITERIA_CONFIG}")
print(f"Timeout per solver: {TIMEOUT_PER_SOLVER} seconds (30 minutes)\n")

for gen_min_improv_nr_pts in CRITERIA_CONFIG['gen_min_improv_nr_pts']:
    for gen_min_improv_window in CRITERIA_CONFIG['gen_min_improv_window']:
        for tm_min_improv_nr_pts in CRITERIA_CONFIG['tm_min_improv_nr_pts']:
            for tm_min_improv_window in CRITERIA_CONFIG['tm_min_improv_window']:
                for target_obj in CRITERIA_CONFIG['target_obj']:
                    for tm_limit_seconds in CRITERIA_CONFIG['tm_limit_seconds']:
                        for nr_max_generations in CRITERIA_CONFIG['nr_max_generations']:
                            criteria = create_criteria(
                                gen_min_improv_nr_pts, gen_min_improv_window,
                                tm_min_improv_nr_pts, tm_min_improv_window,
                                target_obj, tm_limit_seconds, nr_max_generations
                            )
                            
                            print(f"\nTesting criteria: {criteria}")
                            
                            # Evolution Strategy
                            print("Tuning Evolution Strategy...")
                            text = "CRITERIA: " + str(criteria) + "\n"
                            text += "EVOLUTION STRATEGY SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("Evolution Strategy", tune_evolution_strategy, criteria, TIMEOUT_PER_SOLVER)
                            save_results("evol_parameter_tuning.txt", text)
                            
                            # Genetic Algorithm
                            print("Tuning Genetic Algorithm...")
                            text = "GENETIC ALGORITHM SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("Genetic Algorithm", tune_genetic_algorithm, criteria, TIMEOUT_PER_SOLVER)
                            save_results("ga_parameter_tuning.txt", text)
                            
                            # GRASP
                            print("Tuning GRASP...")
                            text = "GRASP SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("GRASP", tune_grasp, criteria, TIMEOUT_PER_SOLVER)
                            save_results("grasp_parameter_tuning.txt", text)
                            
                            # Iterated Local Search
                            print("Tuning ILS...")
                            text = "ITERATED LOCAL SEARCH SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("ILS", tune_ils, criteria, TIMEOUT_PER_SOLVER)
                            save_results("ils_parameter_tuning.txt", text)
                            
                            # Ant System
                            print("Tuning Ant System...")
                            text = "ANT SYSTEM SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("Ant System", tune_ant_system, criteria, TIMEOUT_PER_SOLVER)
                            save_results("as_parameter_tuning.txt", text)
                            
                            # EAS Ant System
                            print("Tuning EAS Ant System...")
                            text = "EAS ANT SYSTEM SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("EAS Ant System", tune_eas_ant_system, criteria, TIMEOUT_PER_SOLVER)
                            save_results("eas_parameter_tuning.txt", text)
                            
                            # Ranked Ant System
                            print("Tuning Ranked Ant System...")
                            text = "RANKED ANT SYSTEM SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("Ranked Ant System", tune_ranked_ant_system, criteria, TIMEOUT_PER_SOLVER)
                            save_results("ras_parameter_tuning.txt", text)
                            
                            # Ant Colony System
                            print("Tuning Ant Colony System...")
                            text = "ANT COLONY SYSTEM SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("Ant Colony System", tune_ant_colony_system, criteria, TIMEOUT_PER_SOLVER)
                            save_results("acs_parameter_tuning.txt", text)
                            
                            # Ant Multi-Tour System
                            print("Tuning Ant Multi-Tour System...")
                            text = "ANT MULTI-TOUR SYSTEM SOLVER\n"
                            text += run_algorithm_tuning_with_timeout("Ant Multi-Tour System", tune_ant_multi_tour_system, criteria, TIMEOUT_PER_SOLVER)
                            save_results("amts_parameter_tuning.txt", text)

print("\n[INFO] Parameter tuning completed!")
print(f"[INFO] Best solution found with cost: {best_solution_state['best_cost']}")

import json

# Save best solution
if best_solution_state['best'] is not None:
    best = best_solution_state['best']
    json_best_sol = '{'
    json_best_sol += f'"makespan": {best._makespan}, '
    json_best_sol += '"schedule": {'
    for key in best.schedule.keys():
        if key != list(best.schedule.keys())[-1]:
            json_best_sol += f'"{key}": {best.schedule[key]}, '
        else:
            json_best_sol += f'"{key}": {best.schedule[key]}'
    json_best_sol += '}}'

    with open(PATH + "/best_solution.json", "w") as f:
        json.dump(json_best_sol, f)

    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(best_solution_state['history'], linewidth=1, label=f'Best found (makespan: {best._makespan})')
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(PATH + "/best_solution_convergence.png")
    
    print(f"\nFinal Results:")
    print(f"Best makespan found: {best._makespan}")
    print(f"Results saved to: {PATH}")
else:
    print("\nNo valid solution found during parameter tuning.")