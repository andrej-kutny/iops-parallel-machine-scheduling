from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import (
    StoppingCriterion, TimeLimit, GenMinImprovement, TimeMinImprovement,
    MaxGenerations, TargetObjective,
)

import datetime

current_date = datetime.datetime.now()
solver_type = "evol"

PATH = "src/results/" + current_date.strftime("%Y_%m_%d_%H_%M_%S") + "/" + solver_type

print(PATH)

from solvers.evolution_strategy import EvolutionStrategySolver

import numpy as np

import os
if os.path.isdir(PATH) == False:
    os.makedirs(PATH)

instance = SchedulingInstance("src/data/357_15_146_H.json")


evol_solver_1 = EvolutionStrategySolver(mu=10, lam=50, c=0.85, criteria=[GenMinImprovement(window=20)])

best, _, _ = evol_solver_1.solve(instance)
best_cost = np.inf
best_mu, best_lam, best_c = np.inf, np.inf, np.inf
history = []
text = ""

for gen_min_improv_nr_pts in np.linspace(0.01, 0.99, 5):
    for gen_min_improv_window in np.linspace(20, 100, 5).astype("int"):
        for tm_min_improv_nr_pts in np.linspace(0.01, 0.99, 5):
            for tm_min_improv_window in np.linspace(20, 100, 5).astype("int"):
                for target_obj in np.linspace(500, 1500, 5).astype("int"):
                    for tm_limit_seconds in np.linspace(5, 100, 5).astype("int"):
                        for nr_max_generations in np.linspace(1, 100, 5).astype("int"):
                            for criteria in [
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),TimeLimit(seconds=tm_limit_seconds)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds)],
                                [GenMinImprovement(window=gen_min_improv_window, min_pct=gen_min_improv_nr_pts),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),MaxGenerations(n=nr_max_generations)],
                                [TargetObjective(target=target_obj)],
                                [TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts)],
                                [TargetObjective(target=target_obj),TimeLimit(seconds=tm_limit_seconds)],
                                [TargetObjective(target=target_obj),MaxGenerations(n=nr_max_generations)],
                                [TargetObjective(target=target_obj),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds)],
                                [TargetObjective(target=target_obj),TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),MaxGenerations(n=nr_max_generations)],
                                [TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts)],
                                [TimeLimit(seconds=tm_limit_seconds)],
                                [MaxGenerations(n=nr_max_generations)],
                                [TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds),MaxGenerations(n=nr_max_generations)],
                                [TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),TimeLimit(seconds=tm_limit_seconds)],
                                [TimeMinImprovement(window=tm_min_improv_window, min_pct=tm_min_improv_nr_pts),MaxGenerations(n=nr_max_generations)]
                                ]:
                                print(criteria)
                                text += f"CRITERIA: {criteria}\n"
                                for iter in range(1):
                                    for mu in np.linspace(10, 100, 5).astype("int"):
                                        for lam in np.linspace(10, 50, 5).astype("int"):
                                            if mu < lam:
                                                for c in np.linspace(0.1, 0.9, 5):
                                                    evol_solver = EvolutionStrategySolver(mu=mu, lam=lam, c=c, criteria=criteria)
                                                    e1, e2, e3 = evol_solver.solve(instance)
                                                    if e2 < best_cost:
                                                        best_mu = mu
                                                        best_lam = lam
                                                        best_c = c
                                                        best = e1
                                                        best_cost = e2
                                                        history = e3
                                                    text += f"mu={mu}, lam={lam}, c={c}, cost = {e2}; best cost = {best_cost} from ({best_mu}, {best_lam}, {best_c}, {criteria})\n"
                
with open(PATH + "/evol_parameter_tuning.txt", "w") as f:
    f.write(text)

import json

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
plt.plot(history, linewidth=1, label=f'ES (best: {best._makespan})')
plt.xlabel('Generation')
plt.ylabel('Best Makespan')
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig(PATH + "/best_solution_convergence.png")