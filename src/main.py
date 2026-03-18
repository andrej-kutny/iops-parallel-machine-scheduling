from datetime import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

from cli import parse_args
from models.instance import SchedulingInstance
from stopping_criteria import (
    StoppingCriterion, TimeLimit, GenMinImprovement, TimeMinImprovement,
    MaxGenerations, TargetObjective,
)
from solvers.grasp import GraspSolver
from solvers.simulated_annealing import SimulatedAnnealingSolver
from solvers.evolution_strategy import EvolutionStrategySolver
from solvers.iterated_local_search import ILSSolver
from solvers.genetic_algorithm import GeneticAlgorithmSolver
from solvers.ant_system import AntSystem, RankedAntSystem, EasAntSystem
from solvers.max_min_ant_system import MaxMinAntSystem
from solvers.ant_colony_system import AntColonySystem
from solvers.ant_multi_tour_system import AntMultiTourSystem
from solvers.combined import CombinedSolver


def build_criteria(args) -> list[StoppingCriterion]:
    """Build stopping criteria from CLI arguments.

    If none are explicitly set, defaults to GenMinImprovement(window=20).
    """
    criteria: list[StoppingCriterion] = []

    if args.time_limit is not None:
        criteria.append(TimeLimit(args.time_limit))

    if args.max_generations is not None:
        criteria.append(MaxGenerations(args.max_generations))

    if args.gen_min_improvement is not None:
        vals = args.gen_min_improvement
        window = int(vals[0]) if len(vals) >= 1 else 20
        min_pct = float(vals[1]) if len(vals) >= 2 else None
        kw = {"window": window}
        if min_pct is not None:
            kw["min_pct"] = min_pct
        criteria.append(GenMinImprovement(**kw))

    if args.time_min_improvement is not None:
        vals = args.time_min_improvement
        window = float(vals[0]) if len(vals) >= 1 else 30.0
        min_pct = float(vals[1]) if len(vals) >= 2 else None
        kw = {"window": window}
        if min_pct is not None:
            kw["min_pct"] = min_pct
        criteria.append(TimeMinImprovement(**kw))

    if args.target_objective is not None:
        criteria.append(TargetObjective(args.target_objective))

    # Default: if nothing set, use GenMinImprovement
    if not criteria:
        criteria.append(GenMinImprovement(window=20))

    return criteria


def build_solver(solver_name: str, criteria: list[StoppingCriterion]):
    if solver_name == "grasp":
        return GraspSolver(criteria=criteria)
    elif solver_name == "sa":
        return SimulatedAnnealingSolver(criteria=criteria)
    elif solver_name == "es":
        return EvolutionStrategySolver(criteria=criteria)
    elif solver_name == "as":
        return AntSystem(criteria=criteria)
    elif solver_name == "mmas":
        return MaxMinAntSystem(criteria=criteria)
    elif solver_name == "acs":
        return AntColonySystem(criteria=criteria)
    elif solver_name == "amts":
        return AntMultiTourSystem(criteria=criteria)
    elif solver_name == "ils":
        return ILSSolver(criteria=criteria)
    elif solver_name == "ga":
        return GeneticAlgorithmSolver(criteria=criteria)
    elif solver_name == "minizinc":
        try:
            from minizinc_cp import MinizincSolver
        except ImportError as e:
            raise ValueError(
                "MiniZinc solver requires: pip install minizinc (and MiniZinc 2.6+ with a backend e.g. Gecode)"
            ) from e
        return MinizincSolver(solver_name="gecode", criteria=criteria)
    elif solver_name == "combined":
        solver_factories = [
            ILSSolver,
            MaxMinAntSystem,
            RankedAntSystem,
            GraspSolver,
            AntMultiTourSystem,
            EvolutionStrategySolver,
        ]
        # If no min-improvement criterion was explicitly set, add a default
        # so each sub-solver switches after 60s of stagnation.
        has_min_improvement = any(
            isinstance(c, (GenMinImprovement, TimeMinImprovement))
            for c in criteria
        )
        combined_criteria = list(criteria)
        if not has_min_improvement:
            combined_criteria.append(TimeMinImprovement(window=300.0, min_pct=0.005))
        return CombinedSolver(
            solver_factories=solver_factories,
            criteria=combined_criteria,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.quiet:
        verbosity = 0
    elif args.verbose > 0:
        verbosity = args.verbose
    else:
        verbosity = 1  # default: show new-best updates

    def log(msg, min_level=1):
        if verbosity >= min_level:
            print(msg, file=sys.stderr)

    instance = SchedulingInstance(args.instance)
    log(f"Loaded {instance}")

    on_new_best = None
    on_solver_switch = None
    prev_best = [float("inf")]

    if verbosity >= 1:
        def on_new_best(gen, solution, cost, elapsed):
            old = prev_best[0]
            prev_best[0] = cost
            if old == float("inf"):
                log(f"  [gen={gen:6d}  t={elapsed:7.2f}s]  New best: {cost:.1f}")
            else:
                log(f"  [gen={gen:6d}  t={elapsed:7.2f}s]  New best: {old:.1f} -> {cost:.1f}")
            if verbosity >= 2:
                log(solution)

    # if verbosity >= 2:
        def on_solver_switch(prev, nxt):
            log(f"  [solver switch]  {prev} -> {nxt}")
    
    out_path = os.path.join("results", f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{args.solver}") if args.output_dir is None else args.output_dir
    os.makedirs(out_path, exist_ok=True)

    if args.forever:
        log("Running in FOREVER mode. Press Ctrl+C to stop.")
        best_overall_cost = float('inf')
        best_overall_solution = None
        best_overall_history = None
        
        iteration = 0
        try:
            while True:
                iteration += 1
                log(f"\n--- Forever Loop: Iteration {iteration} ---")
                criteria = build_criteria(args)
                solver = build_solver(args.solver, criteria)
                log(f"Stopping criteria: {criteria}")
                
                prev_best[0] = float("inf")
                
                if args.solver == "combined":
                    import random
                    random.shuffle(solver.solver_factories)
                    solution, makespan, history = solver.solve(
                        instance, on_new_best=on_new_best, on_solver_switch=on_solver_switch
                    )
                else:
                    solution, makespan, history = solver.solve(instance, on_new_best=on_new_best)
                
                log(f"Iteration {iteration} finished with makespan {makespan:.2f}")
                if makespan < best_overall_cost:
                    best_overall_cost = makespan
                    best_overall_solution = solution
                    best_overall_history = history
                    file_path = os.path.join(out_path, f"solution_{makespan:.2f}.json")
                    with open(file_path, "w") as f:
                        json.dump(best_overall_solution.to_json(), f, indent=2)
                    log(f"*** NEW OVERALL BEST: {makespan:.2f} saved to {file_path} ***")

        except KeyboardInterrupt:
            log("\nCtrl+C pressed! Stopping forever loop.")
            if best_overall_solution is None:
                log("No solution found before stopping.", min_level=0)
                sys.exit(1)
            solution = best_overall_solution
            makespan = best_overall_cost
            history = best_overall_history
            # Mock solver.criteria to not crash when printing triggered later
            class DummyCriteria:
                triggered = False
            solver.criteria = [DummyCriteria()]
    else:
        criteria = build_criteria(args)
        solver = build_solver(args.solver, criteria)
        log(f"Running {args.solver} solver...")
        log(f"Stopping criteria: {criteria}")

        if args.solver == "combined":
            solution, makespan, history = solver.solve(
                instance, on_new_best=on_new_best, on_solver_switch=on_solver_switch
            )
        else:
            solution, makespan, history = solver.solve(instance, on_new_best=on_new_best)

    feasible, msg = solution.is_feasible()
    if not args.forever:
        triggered = [c for c in solver.criteria if c.triggered]
    else:
        triggered = []

    log("--- Done ---", min_level=0)
    if feasible:
        log(f"Makespan: {int(makespan)}, Feasible", min_level=0)
    else:
        log(f"Makespan: {int(makespan)}, Infeasible: {msg}", min_level=0)
    log(f"Generations: {len(history)}")
    log(f"Triggered: {triggered}", min_level=0)
    log(solution, min_level=0)

    output = solution.to_json()
    print(json.dumps(output, indent=2))

    file_path = os.path.join(out_path, f"solution_{makespan:.2f}.json")
    with open(file_path, "w") as f:
        json.dump(output, f, indent=2)

    log(f"Solution {makespan:.2f} saved to {file_path}")


if __name__ == "__main__":
    main()
