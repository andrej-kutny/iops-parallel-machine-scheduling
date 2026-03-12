import json
import os
import sys

# Ensure project root is on sys.path so `src.*` imports work
# regardless of how main.py is invoked
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from src.cli import parse_args
from src.models.instance import SchedulingInstance
from src.stopping_criteria import (
    StoppingCriterion, TimeLimit, GenMinImprovement, TimeMinImprovement,
    MaxGenerations, TargetObjective,
)
from src.solvers.grasp import GraspSolver
from src.solvers.simulated_annealing import SimulatedAnnealingSolver
from src.solvers.evolution_strategy import EvolutionStrategySolver
from src.solvers.ant_system import AntSystem
from src.solvers.max_min_ant_system import MaxMinAntSystem
from src.solvers.ant_colony_system import AntColonySystem
from src.solvers.ant_multi_tour_system import AntMultiTourSystem
from src.solvers.combined import CombinedSolver


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
        return GraspSolver(alpha=0.5, criteria=criteria)
    elif solver_name == "sa":
        return SimulatedAnnealingSolver(
            initial_temp=100.0, cooling_rate=0.995, criteria=criteria
        )
    elif solver_name == "es":
        return EvolutionStrategySolver(mu=10, lam=50, c=0.85, criteria=criteria)
    elif solver_name == "as":
        return AntSystem(n_ants=20, alpha=1.0, beta=2.0, rho=0.1, criteria=criteria)
    elif solver_name == "mmas":
        return MaxMinAntSystem(
            n_ants=20, alpha=1.0, beta=2.0, rho=0.1,
            reinit_frequency=100, criteria=criteria
        )
    elif solver_name == "acs":
        return AntColonySystem(
            n_ants=20, alpha=1.0, beta=2.0, rho=0.1,
            q0=0.9, local_decay=0.1, criteria=criteria
        )
    elif solver_name == "amts":
        return AntMultiTourSystem(
            n_ants=20, alpha=1.0, beta=2.0, rho=0.1,
            q_tours=5, criteria=criteria
        )
    elif solver_name == "combined":
        sub_solvers = [
            GraspSolver(alpha=0.5),
            SimulatedAnnealingSolver(initial_temp=100.0, cooling_rate=0.995),
            AntColonySystem(n_ants=20, alpha=1.0, beta=2.0, rho=0.1, q0=0.9),
            EvolutionStrategySolver(mu=10, lam=50, c=0.85),
        ]
        # If no min-improvement criterion was explicitly set, add a default
        # so each sub-solver switches after 60s of stagnation.
        has_min_improvement = any(
            isinstance(c, (GenMinImprovement, TimeMinImprovement))
            for c in criteria
        )
        combined_criteria = list(criteria)
        if not has_min_improvement:
            combined_criteria.append(TimeMinImprovement(window=60.0, min_pct=0.005))
        return CombinedSolver(
            solvers=sub_solvers,
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

    criteria = build_criteria(args)
    solver = build_solver(args.solver, criteria)
    log(f"Running {args.solver} solver...")
    log(f"Stopping criteria: {criteria}")

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

    if args.solver == "combined":
        solution, makespan, history = solver.solve(
            instance, on_new_best=on_new_best, on_solver_switch=on_solver_switch
        )
    else:
        solution, makespan, history = solver.solve(instance, on_new_best=on_new_best)

    feasible, msg = solution.is_feasible()
    triggered = [c for c in solver.criteria if c.triggered]

    log("--- Done ---", min_level=0)
    if feasible:
        log(f"Makespan: {makespan}, Feasible", min_level=0)
    else:
        log(f"Makespan: {makespan}, Infeasible: {msg}", min_level=0)
    log(f"Generations: {len(history)}")
    log(f"Triggered: {triggered}", min_level=0)
    log(solution, min_level=0)

    output = solution.to_json()
    print(json.dumps(output, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        log(f"Solution saved to {args.output}")


if __name__ == "__main__":
    main()
