from models.instance import SchedulingInstance
from models.solution import SchedulingSolution
from stopping_criteria import (
    StoppingCriterion, TimeLimit, GenMinImprovement, TimeMinImprovement,
    MaxGenerations, TargetObjective,
)

import datetime

current_date = datetime.datetime.now()
solver_type = "evol"

PATH = "src/results/" + solver_type + current_date.strftime("_%Y_%m_%d_%H_%M_%S")

print(PATH)

from solvers.evolution_strategy import EvolutionStrategySolver

import numpy as np

import os
if os.path.isdir(PATH) == False:
    os.makedirs(PATH)

instance = SchedulingInstance("src/data/357_15_146_H.json")

criteria: list[StoppingCriterion] = []
criteria.append(GenMinImprovement(window=20))


evol_solver_1 = EvolutionStrategySolver(mu=10, lam=50, c=0.85, criteria=criteria)

best, _, _ = evol_solver_1.solve(instance)
best_cost = np.inf
best_mu, best_lam, best_c = np.inf, np.inf, np.inf
history = []

text = ""

for iter in range(1):
    for mu in np.linspace(10, 100, 2).astype("int"):
        for lam in np.linspace(10, 50, 2).astype("int"):
            if mu < lam:
                for c in np.linspace(0.1, 0.9, 2):
                    evol_solver = EvolutionStrategySolver(mu=mu, lam=lam, c=c, criteria=criteria)
                    e1, e2, e3 = evol_solver.solve(instance)
                    if e2 < best_cost:
                        best_mu = mu
                        best_lam = lam
                        best_c = c
                        best = e1
                        best_cost = e2
                        history = e3
                    text += f"mu={mu}, lam={lam}, c={c}, cost = {e2}; best cost = {best_cost} from ({best_mu}, {best_lam}, {best_c})\n"
                
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