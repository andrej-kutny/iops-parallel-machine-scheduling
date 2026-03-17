"""
Fine-tuning benchmark for parallel machine scheduling solvers.

Adaptive parameter search inspired by the fine_tune() pattern from
iops-hackathon-3: a randomized search that narrows the parameter window
over time (multiplier decays from --max-multiplier toward --min-multiplier
as the allocated time elapses).

Usage examples:
    # Run all algorithms, 10 min each, 60s per solver call
    python fine_tuning_benchmark.py src/data/357_15_146_H.json -at 10 -st 60

    # Only tune SA variants and GRASP, 5 min each
    python fine_tuning_benchmark.py src/data/357_15_146_H.json \
        sa_geometric sa_logarithmic sa_linear grasp -at 5

    # Aggressive narrowing
    python fine_tuning_benchmark.py src/data/357_15_146_H.json \
        --min-multiplier 0.005 --max-multiplier 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.instance import SchedulingInstance
from stopping_criteria import TimeLimit, TimeMinImprovement

from solvers import (
    EvolutionStrategySolver,
    GeneticAlgorithmSolver,
    GraspSolver,
    ILSSolver,
    AntSystem,
    RankedAntSystem,
    EasAntSystem,
    MaxMinAntSystem,
    AntColonySystem,
    AntMultiTourSystem,
    SimulatedAnnealingSolver,
    GeometricCooling,
    LinearCooling,
    LogarithmicCooling,
)

try:
    from minizinc_cp import MinizincSolver

    HAS_MINIZINC = True
except Exception:
    HAS_MINIZINC = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning benchmark for scheduling solvers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("instance", help="Path to instance JSON file")
    p.add_argument(
        "algorithms",
        nargs="*",
        default=[],
        help="Algorithms to tune (empty = all)",
    )
    p.add_argument(
        "-at",
        "--algorithm-time",
        type=float,
        default=10.0,
        help="Time budget per algorithm, in minutes",
    )
    p.add_argument(
        "-mi",
        "--min-multiplier",
        type=float,
        default=0.01,
        help="Minimum multiplier (narrow search at end), dimensionless",
    )
    p.add_argument(
        "-ma",
        "--max-multiplier",
        type=float,
        default=0.333,
        help="Starting multiplier (wide search at start), dimensionless",
    )
    p.add_argument(
        "-st",
        "--stop-time",
        type=float,
        default=90.0,
        help="TimeLimit stopping criterion per solver call, in seconds",
    )
    p.add_argument(
        "-si",
        "--stop-improvement",
        nargs=2,
        type=float,
        default=[15.0, 0.01],
        metavar=("WINDOW_S", "MIN_PCT"),
        help="TimeMinImprovement stopping criterion: window in seconds and min_pct (ratio, e.g. 0.01 = 1%%)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Override output directory (default: src/results/YYYY-MM-DD_HHMM)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stopping criteria factory
# ---------------------------------------------------------------------------

def make_criteria(stop_time: float, stop_improvement: tuple[float, float]):
    """Create fresh stopping criteria for each solver call."""
    window_s, min_pct = stop_improvement
    return [
        TimeLimit(stop_time),
        TimeMinImprovement(window=window_s, min_pct=min_pct),
    ]


# ---------------------------------------------------------------------------
# Adaptive multiplier
# ---------------------------------------------------------------------------

def get_multiplier(
    elapsed: float,
    total_time: float,
    max_mult: float,
    min_mult: float,
) -> float:
    """
    Compute current multiplier based on how much time has elapsed.
    Linearly interpolates from max_mult -> min_mult as elapsed goes 0 -> total_time.
    """
    if total_time <= 0:
        return min_mult
    progress = min(1.0, elapsed / total_time)
    return max_mult - progress * (max_mult - min_mult)


# ---------------------------------------------------------------------------
# Random config helpers
# ---------------------------------------------------------------------------

def _rand_around(center, mult, lo=None, hi=None, as_int=False):
    """Sample uniformly around center * (1 ± mult/2)."""
    half = mult / 2.0
    low = center * (1.0 - half)
    high = center * (1.0 + half)
    if lo is not None:
        low = max(lo, low)
    if hi is not None:
        high = min(hi, high)
    if as_int:
        low, high = int(low), int(high)
        if high <= low:
            high = low + 1
        return np.random.randint(low, high + 1)
    return round(np.random.uniform(low, high), 4)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

# Each entry:
#   "name": display name
#   "defaults": dict of best-known parameter defaults (center of search)
#   "config_gen": (defaults, multiplier) -> config dict
#   "solver_factory": (config, criteria) -> solver instance

ALGORITHMS: dict[str, dict] = {}


def _register(key, name, defaults, config_gen, solver_factory):
    ALGORITHMS[key] = {
        "name": name,
        "defaults": defaults,
        "config_gen": config_gen,
        "solver_factory": solver_factory,
    }


# --- GRASP -----------------------------------------------------------------

_GRASP_DEFAULTS = {"alpha": 0.5}


def _grasp_config(defaults, m):
    return {"alpha": _rand_around(defaults["alpha"], m, 0.01, 0.99)}


def _grasp_factory(cfg, criteria):
    return GraspSolver(alpha=cfg["alpha"], criteria=criteria)


_register("grasp", "GRASP", _GRASP_DEFAULTS, _grasp_config, _grasp_factory)


# --- Evolution Strategy ----------------------------------------------------

_ES_DEFAULTS = {"mu": 20, "lam": 50, "c": 0.85}


def _es_config(defaults, m):
    mu = _rand_around(defaults["mu"], m, 2, 200, as_int=True)
    lam = _rand_around(defaults["lam"], m, mu + 1, 300, as_int=True)
    if lam <= mu:
        lam = mu + 1
    return {
        "mu": mu,
        "lam": lam,
        "c": _rand_around(defaults["c"], m, 0.5, 0.99),
    }


def _es_factory(cfg, criteria):
    return EvolutionStrategySolver(
        mu=cfg["mu"], lam=cfg["lam"], c=cfg["c"], criteria=criteria
    )


_register("evolution_strategy", "Evolution Strategy", _ES_DEFAULTS, _es_config, _es_factory)


# --- Genetic Algorithm -----------------------------------------------------

_GA_DEFAULTS = {"population_size": 50, "offspring_per_generation": 3, "mutation_strength": 4}


def _ga_config(defaults, m):
    return {
        "population_size": _rand_around(defaults["population_size"], m, 4, 500, as_int=True),
        "offspring_per_generation": _rand_around(defaults["offspring_per_generation"], m, 1, 20, as_int=True),
        "mutation_strength": _rand_around(defaults["mutation_strength"], m, 1, 20, as_int=True),
    }


def _ga_factory(cfg, criteria):
    return GeneticAlgorithmSolver(
        population_size=cfg["population_size"],
        offspring_per_generation=cfg["offspring_per_generation"],
        mutation_strength=cfg["mutation_strength"],
        criteria=criteria,
    )


_register("genetic_algorithm", "Genetic Algorithm", _GA_DEFAULTS, _ga_config, _ga_factory)


# --- ILS -------------------------------------------------------------------

_ILS_DEFAULTS = {"perturbation_strength": 5}


def _ils_config(defaults, m):
    return {
        "perturbation_strength": _rand_around(defaults["perturbation_strength"], m, 1, 30, as_int=True),
    }


def _ils_factory(cfg, criteria):
    return ILSSolver(perturbation_strength=cfg["perturbation_strength"], criteria=criteria)


_register("ils", "Iterated Local Search", _ILS_DEFAULTS, _ils_config, _ils_factory)


# --- Ant System ------------------------------------------------------------

_AS_DEFAULTS = {"n_ants": 20, "alpha": 1.0, "beta": 2.0, "rho": 0.1, "q_ct": 1.0}


def _as_config(defaults, m):
    return {
        "n_ants": _rand_around(defaults["n_ants"], m, 2, 300, as_int=True),
        "alpha": _rand_around(defaults["alpha"], m, 0.1, 5.0),
        "beta": _rand_around(defaults["beta"], m, 0.1, 5.0),
        "rho": _rand_around(defaults["rho"], m, 0.01, 0.99),
        "q_ct": _rand_around(defaults["q_ct"], m, 0.1, 10.0),
    }


def _as_factory(cfg, criteria):
    return AntSystem(
        n_ants=cfg["n_ants"], alpha=cfg["alpha"], beta=cfg["beta"],
        rho=cfg["rho"], q_ct=cfg["q_ct"], criteria=criteria,
    )


_register("ant_system", "Ant System", _AS_DEFAULTS, _as_config, _as_factory)


# --- EAS Ant System --------------------------------------------------------

_EAS_DEFAULTS = {"n_ants": 20, "alpha": 1.0, "beta": 2.0, "rho": 0.1, "q_ct": 1.0, "sigma": 1.0}


def _eas_config(defaults, m):
    return {
        "n_ants": _rand_around(defaults["n_ants"], m, 2, 300, as_int=True),
        "alpha": _rand_around(defaults["alpha"], m, 0.1, 5.0),
        "beta": _rand_around(defaults["beta"], m, 0.1, 5.0),
        "rho": _rand_around(defaults["rho"], m, 0.01, 0.99),
        "q_ct": _rand_around(defaults["q_ct"], m, 0.1, 10.0),
        "sigma": _rand_around(defaults["sigma"], m, 0.1, 10.0),
    }


def _eas_factory(cfg, criteria):
    return EasAntSystem(
        n_ants=cfg["n_ants"], alpha=cfg["alpha"], beta=cfg["beta"],
        rho=cfg["rho"], q_ct=cfg["q_ct"], sigma=cfg["sigma"], criteria=criteria,
    )


_register("eas_ant_system", "Elitist Ant System", _EAS_DEFAULTS, _eas_config, _eas_factory)


# --- Ranked Ant System -----------------------------------------------------

_RAS_DEFAULTS = {"n_ants": 20, "alpha": 1.0, "beta": 2.0, "rho": 0.1, "q_ct": 1.0}


def _ras_config(defaults, m):
    return {
        "n_ants": _rand_around(defaults["n_ants"], m, 2, 300, as_int=True),
        "alpha": _rand_around(defaults["alpha"], m, 0.1, 5.0),
        "beta": _rand_around(defaults["beta"], m, 0.1, 5.0),
        "rho": _rand_around(defaults["rho"], m, 0.01, 0.99),
        "q_ct": _rand_around(defaults["q_ct"], m, 0.1, 10.0),
    }


def _ras_factory(cfg, criteria):
    return RankedAntSystem(
        n_ants=cfg["n_ants"], alpha=cfg["alpha"], beta=cfg["beta"],
        rho=cfg["rho"], q_ct=cfg["q_ct"], criteria=criteria,
    )


_register("ranked_ant_system", "Ranked Ant System", _RAS_DEFAULTS, _ras_config, _ras_factory)


# --- Max-Min Ant System ----------------------------------------------------

_MMAS_DEFAULTS = {"n_ants": 20, "alpha": 1.0, "beta": 2.0, "rho": 0.1, "reinit_frequency": 100}


def _mmas_config(defaults, m):
    return {
        "n_ants": _rand_around(defaults["n_ants"], m, 2, 300, as_int=True),
        "alpha": _rand_around(defaults["alpha"], m, 0.1, 5.0),
        "beta": _rand_around(defaults["beta"], m, 0.1, 5.0),
        "rho": _rand_around(defaults["rho"], m, 0.01, 0.99),
        "reinit_frequency": _rand_around(defaults["reinit_frequency"], m, 10, 500, as_int=True),
    }


def _mmas_factory(cfg, criteria):
    return MaxMinAntSystem(
        n_ants=cfg["n_ants"], alpha=cfg["alpha"], beta=cfg["beta"],
        rho=cfg["rho"], reinit_frequency=cfg["reinit_frequency"], criteria=criteria,
    )


_register("max_min_ant_system", "Max-Min Ant System", _MMAS_DEFAULTS, _mmas_config, _mmas_factory)


# --- Ant Colony System -----------------------------------------------------

_ACS_DEFAULTS = {"n_ants": 20, "alpha": 1.0, "beta": 2.0, "rho": 0.1, "q0": 0.9, "local_decay": 0.1}


def _acs_config(defaults, m):
    return {
        "n_ants": _rand_around(defaults["n_ants"], m, 2, 300, as_int=True),
        "alpha": _rand_around(defaults["alpha"], m, 0.1, 5.0),
        "beta": _rand_around(defaults["beta"], m, 0.1, 5.0),
        "rho": _rand_around(defaults["rho"], m, 0.01, 0.99),
        "q0": _rand_around(defaults["q0"], m, 0.01, 0.99),
        "local_decay": _rand_around(defaults["local_decay"], m, 0.01, 0.99),
    }


def _acs_factory(cfg, criteria):
    return AntColonySystem(
        n_ants=cfg["n_ants"], alpha=cfg["alpha"], beta=cfg["beta"],
        rho=cfg["rho"], q0=cfg["q0"], local_decay=cfg["local_decay"], criteria=criteria,
    )


_register("ant_colony_system", "Ant Colony System", _ACS_DEFAULTS, _acs_config, _acs_factory)


# --- Ant Multi-Tour System -------------------------------------------------

_AMTS_DEFAULTS = {"n_ants": 20, "alpha": 1.0, "beta": 2.0, "rho": 0.1, "q_tours": 5}


def _amts_config(defaults, m):
    return {
        "n_ants": _rand_around(defaults["n_ants"], m, 2, 300, as_int=True),
        "alpha": _rand_around(defaults["alpha"], m, 0.1, 5.0),
        "beta": _rand_around(defaults["beta"], m, 0.1, 5.0),
        "rho": _rand_around(defaults["rho"], m, 0.01, 0.99),
        "q_tours": _rand_around(defaults["q_tours"], m, 1, 50, as_int=True),
    }


def _amts_factory(cfg, criteria):
    return AntMultiTourSystem(
        n_ants=cfg["n_ants"], alpha=cfg["alpha"], beta=cfg["beta"],
        rho=cfg["rho"], q_tours=cfg["q_tours"], criteria=criteria,
    )


_register("ant_multi_tour_system", "Ant Multi-Tour System", _AMTS_DEFAULTS, _amts_config, _amts_factory)


# --- Simulated Annealing (Geometric) --------------------------------------

_SA_GEO_DEFAULTS = {
    "initial_temp": 150.0,
    "cooling_rate_param": 0.995,
    "reheat_factor": 1.5,
    "reheat_patience": 200,
}


def _sa_geo_config(defaults, m):
    return {
        "initial_temp": _rand_around(defaults["initial_temp"], m, 10.0, 1000.0),
        "cooling_rate_param": _rand_around(defaults["cooling_rate_param"], m, 0.9, 0.9999),
        "reheat_factor": _rand_around(defaults["reheat_factor"], m, 1.0, 5.0),
        "reheat_patience": _rand_around(defaults["reheat_patience"], m, 20, 2000, as_int=True),
    }


def _sa_geo_factory(cfg, criteria):
    return SimulatedAnnealingSolver(
        initial_temp=cfg["initial_temp"],
        cooling_rate=GeometricCooling(cfg["cooling_rate_param"]),
        reheat_factor=cfg["reheat_factor"],
        reheat_patience=cfg["reheat_patience"],
        criteria=criteria,
    )


_register("sa_geometric", "SA Geometric Cooling", _SA_GEO_DEFAULTS, _sa_geo_config, _sa_geo_factory)


# --- Simulated Annealing (Logarithmic) ------------------------------------

_SA_LOG_DEFAULTS = {
    "initial_temp": 150.0,
    "reheat_factor": 1.5,
    "reheat_patience": 200,
}


def _sa_log_config(defaults, m):
    return {
        "initial_temp": _rand_around(defaults["initial_temp"], m, 10.0, 1000.0),
        "reheat_factor": _rand_around(defaults["reheat_factor"], m, 1.0, 5.0),
        "reheat_patience": _rand_around(defaults["reheat_patience"], m, 20, 2000, as_int=True),
    }


def _sa_log_factory(cfg, criteria):
    return SimulatedAnnealingSolver(
        initial_temp=cfg["initial_temp"],
        cooling_rate=LogarithmicCooling(),
        reheat_factor=cfg["reheat_factor"],
        reheat_patience=cfg["reheat_patience"],
        criteria=criteria,
    )


_register("sa_logarithmic", "SA Logarithmic Cooling", _SA_LOG_DEFAULTS, _sa_log_config, _sa_log_factory)


# --- Simulated Annealing (Linear) -----------------------------------------

_SA_LIN_DEFAULTS = {
    "initial_temp": 150.0,
    "cooling_rate_iterations": 5000,
    "reheat_factor": 1.5,
    "reheat_patience": 200,
}


def _sa_lin_config(defaults, m):
    return {
        "initial_temp": _rand_around(defaults["initial_temp"], m, 10.0, 1000.0),
        "cooling_rate_iterations": _rand_around(defaults["cooling_rate_iterations"], m, 500, 50000, as_int=True),
        "reheat_factor": _rand_around(defaults["reheat_factor"], m, 1.0, 5.0),
        "reheat_patience": _rand_around(defaults["reheat_patience"], m, 20, 2000, as_int=True),
    }


def _sa_lin_factory(cfg, criteria):
    return SimulatedAnnealingSolver(
        initial_temp=cfg["initial_temp"],
        cooling_rate=LinearCooling(cfg["cooling_rate_iterations"]),
        reheat_factor=cfg["reheat_factor"],
        reheat_patience=cfg["reheat_patience"],
        criteria=criteria,
    )


_register("sa_linear", "SA Linear Cooling", _SA_LIN_DEFAULTS, _sa_lin_config, _sa_lin_factory)


# --- MiniZinc (CP) --------------------------------------------------------

if HAS_MINIZINC:
    _MZ_DEFAULTS = {"solver_name": "gecode"}

    def _mz_config(defaults, m):
        # MiniZinc has no tunable continuous parameters;
        # we just run it once with the stop-time criterion.
        return {"solver_name": defaults["solver_name"]}

    def _mz_factory(cfg, criteria):
        return MinizincSolver(solver_name=cfg["solver_name"], criteria=criteria)

    _register("minizinc", "MiniZinc CP", _MZ_DEFAULTS, _mz_config, _mz_factory)


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def _save_convergence_plot(time_history: list[tuple[float, float]], title: str, filepath: str):
    """Save convergence plot with elapsed time (seconds) on x-axis.

    Args:
        time_history: list of (elapsed_seconds, best_cost) pairs.
    """
    if not time_history:
        return
    times = [t for t, _ in time_history]
    costs = [c for _, c in time_history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, costs, linewidth=1, color="tab:blue")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Best Makespan")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _save_combined_convergence(
    all_best_histories: dict[str, list[tuple[float, float]]],
    filepath: str,
):
    """Save a combined plot with the best convergence curve per algorithm.

    Each history is a list of (elapsed_seconds, best_cost) pairs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, history in sorted(all_best_histories.items()):
        if history:
            times = [t for t, _ in history]
            costs = [c for _, c in history]
            ax.plot(times, costs, linewidth=1.2, label=label)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Best Makespan")
    ax.set_title("Best Fine-Tune Result per Algorithm — Convergence")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _save_best_solution_json(solution, cost: float, config: dict, filepath: str):
    """Persist the best solution schedule + metadata as JSON."""
    data = {
        "makespan": cost,
        "config": {k: (v if not isinstance(v, np.integer) else int(v)) for k, v in config.items()},
        "schedule": {
            str(k): list(v) for k, v in solution.schedule.items()
        },
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Fine-tune one algorithm
# ---------------------------------------------------------------------------

def fine_tune_algorithm(
    algo_key: str,
    instance: SchedulingInstance,
    algo_time: float,
    stop_time: float,
    stop_improvement: tuple[float, float],
    max_mult: float,
    min_mult: float,
    output_dir: str,
) -> dict:
    """
    Fine-tune a single algorithm for `algo_time` seconds.

    Returns a summary dict with best cost, config, history.
    """
    algo = ALGORITHMS[algo_key]
    algo_name = algo["name"]
    defaults = dict(algo["defaults"])  # mutable copy — we update center on improvement
    config_gen = algo["config_gen"]
    solver_factory = algo["solver_factory"]

    algo_dir = os.path.join(output_dir, algo_key)
    os.makedirs(algo_dir, exist_ok=True)

    csv_path = os.path.join(algo_dir, "runs.csv")
    header_written = False

    best_cost = np.inf
    best_solution = None
    best_config = None
    best_time_history: list[tuple[float, float]] = []  # (elapsed_s, cost)
    best_run_time = np.inf

    algo_start = time.monotonic()
    iteration = 0

    # MiniZinc is not a metaheuristic — run once with the full time budget
    is_minizinc = algo_key == "minizinc"

    while True:
        elapsed = time.monotonic() - algo_start
        if elapsed >= algo_time:
            break

        multiplier = get_multiplier(elapsed, algo_time, max_mult, min_mult)
        config = config_gen(defaults, multiplier)

        if is_minizinc:
            # Give MiniZinc the full algorithm time budget
            criteria = [TimeLimit(algo_time)]
        else:
            criteria = make_criteria(stop_time, stop_improvement)
        solver = solver_factory(config, criteria)

        iteration += 1
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        print(
            f"  [{iteration:4d}] mult={multiplier:.3f} {config_str}",
            end=" ... ",
            flush=True,
        )

        # Capture (elapsed, cost) via on_new_best callback
        run_time_history: list[tuple[float, float]] = []
        run_start = time.monotonic()

        def _on_new_best(gen, sol, cost_val, elapsed_val):
            run_time_history.append((elapsed_val, cost_val))

        try:
            solution, cost, history = solver.solve(instance, on_new_best=_on_new_best)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue
        run_time = time.monotonic() - run_start
        n_gens = len(history)

        # Ensure we have at least the final point in time history
        if not run_time_history:
            run_time_history.append((run_time, cost))
        # Add a final point at the end time with the best cost for this run
        if run_time_history[-1][0] < run_time:
            run_time_history.append((run_time, run_time_history[-1][1]))

        # Build CSV row
        row = {
            "iteration": iteration,
            "multiplier": round(multiplier, 4),
            **{k: v for k, v in config.items()},
            "cost": round(cost, 2),
            "generations": n_gens,
            "time_s": round(run_time, 2),
        }

        if not header_written:
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"algorithm={algo_name}", f"instance={instance.n}x{instance.m}"])
                w.writerow(list(row.keys()))
            header_written = True

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(list(row.values()))

        is_new_best = cost < best_cost
        is_time_improvement = cost == best_cost and run_time < best_run_time

        print(f"cost={cost:.1f} gens={n_gens} time={run_time:.1f}s", end="")

        if is_new_best or is_time_improvement:
            if is_new_best:
                print(f"  *** new best ({best_cost:.1f} -> {cost:.1f}) ***", end="")
            else:
                print(f"  *** time improvement ({best_run_time:.1f}s -> {run_time:.1f}s) ***", end="")

            best_cost = cost
            best_run_time = run_time
            best_solution = solution
            best_config = dict(config)
            best_time_history = list(run_time_history)

            # Update defaults to center future search around best config
            for k, v in config.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    defaults[k] = v

            # Save best solution
            _save_best_solution_json(
                solution, cost, config,
                os.path.join(algo_dir, "best_solution.json"),
            )

            # Save convergence plot for this best
            _save_convergence_plot(
                best_time_history,
                f"{algo_name} — Best Makespan: {cost:.1f}",
                os.path.join(algo_dir, "best_convergence.png"),
            )

            # Save history data for later combined plotting (Nx2: time, cost)
            np.save(
                os.path.join(algo_dir, "best_history.npy"),
                np.array(best_time_history),
            )

        print()

        # MiniZinc: single run is enough
        if is_minizinc:
            break

    total_elapsed = time.monotonic() - algo_start
    status = "TIMEOUT" if total_elapsed >= algo_time else "COMPLETED"
    print(
        f"  [{status}] {algo_name} — {iteration} iterations in {total_elapsed:.1f}s"
        f" — best cost: {best_cost}"
    )

    return {
        "algo_key": algo_key,
        "algo_name": algo_name,
        "best_cost": best_cost,
        "best_config": best_config,
        "best_time_history": best_time_history,
        "iterations": iteration,
        "elapsed": total_elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    instance = SchedulingInstance(args.instance)
    instance_label = os.path.basename(args.instance)

    if args.output_dir:
        base_path = args.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        base_path = os.path.join("src", "results", ts)
    os.makedirs(base_path, exist_ok=True)

    algo_time_seconds = args.algorithm_time * 60.0

    print(f"Instance: {instance_label} ({instance.n} jobs, {instance.m} machines)")
    print(f"Results: {base_path}")
    print(f"Algorithm time: {args.algorithm_time} min ({algo_time_seconds:.0f}s) per algorithm")
    print(f"Stop criteria: TimeLimit({args.stop_time}s), TimeMinImprovement(window={args.stop_improvement[0]}s, min_pct={args.stop_improvement[1]})")
    print(f"Multiplier: {args.max_multiplier} -> {args.min_multiplier}")
    print()

    # Resolve algorithm whitelist
    if args.algorithms:
        algo_keys = [k for k in args.algorithms if k in ALGORITHMS]
        unknown = [k for k in args.algorithms if k not in ALGORITHMS]
        if unknown:
            # Check if user explicitly asked for minizinc but it's not installed
            if "minizinc" in unknown and not HAS_MINIZINC:
                print("WARNING: 'minizinc' requested but the minizinc Python package is not installed.")
                print("  Install it with: pip install minizinc")
                print("  Also ensure MiniZinc 2.6+ is installed on the system.")
                unknown.remove("minizinc")
            if unknown:
                print(f"WARNING: Unknown algorithms ignored: {unknown}")
                print(f"  Available: {list(ALGORITHMS.keys())}")
    else:
        algo_keys = list(ALGORITHMS.keys())

    if not algo_keys:
        print("No algorithms to run. Exiting.")
        return

    print(f"Algorithms to tune ({len(algo_keys)}): {algo_keys}\n")

    summaries: list[dict] = []

    for algo_key in algo_keys:
        algo_name = ALGORITHMS[algo_key]["name"]
        print(f"\n{'=' * 70}")
        print(f"  {algo_name.upper()}")
        print(f"{'=' * 70}")

        summary = fine_tune_algorithm(
            algo_key=algo_key,
            instance=instance,
            algo_time=algo_time_seconds,
            stop_time=args.stop_time,
            stop_improvement=tuple(args.stop_improvement),
            max_mult=args.max_multiplier,
            min_mult=args.min_multiplier,
            output_dir=base_path,
        )
        summaries.append(summary)

    # --- Combined results ---------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 70}")

    best_overall_cost = np.inf
    best_overall_algo = None
    all_best_histories: dict[str, list[tuple[float, float]]] = {}

    for s in summaries:
        cost = s["best_cost"]
        label = f"{s['algo_name']} ({cost:.1f})"
        print(f"  {s['algo_name']:30s}: cost={cost:<10.1f} iters={s['iterations']:<6d} time={s['elapsed']:.1f}s")
        if s["best_time_history"]:
            all_best_histories[label] = s["best_time_history"]
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_algo = s["algo_name"]

    print(f"\n  Best solver: {best_overall_algo} with cost: {best_overall_cost:.1f}")
    print(f"  Results saved to: {base_path}")

    # Combined convergence plot
    if all_best_histories:
        _save_combined_convergence(
            all_best_histories,
            os.path.join(base_path, "combined_convergence.png"),
        )
        print(f"  Combined convergence plot: {os.path.join(base_path, 'combined_convergence.png')}")

    # Save overall summary JSON
    summary_data = {
        "instance": instance_label,
        "best_solver": best_overall_algo,
        "best_cost": best_overall_cost,
        "algorithms": [
            {
                "key": s["algo_key"],
                "name": s["algo_name"],
                "best_cost": s["best_cost"],
                "best_config": s["best_config"],
                "iterations": s["iterations"],
                "elapsed_s": round(s["elapsed"], 2),
            }
            for s in summaries
        ],
    }
    summary_path = os.path.join(base_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
