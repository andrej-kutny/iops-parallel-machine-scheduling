import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Parallel Machine Scheduling Solver"
    )
    parser.add_argument(
        "instance",
        help="Path to instance JSON file",
    )
    parser.add_argument(
        "--solver",
        choices=["grasp", "sa", "es", "as", "mmas", "acs", "amts", "combined"],
        default="combined",
        help="Solver to use (default: combined)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit in seconds (default: None = no time limit)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Maximum number of generations",
    )
    parser.add_argument(
        "--gen-min-improvement",
        type=float,
        nargs="*",
        default=None,
        metavar="VALUE",
        help="Stop on stagnation over generations: [window] [min_pct]. "
             "E.g. --gen-min-improvement 20 0.01 or --gen-min-improvement 20",
    )
    parser.add_argument(
        "--time-min-improvement",
        type=float,
        nargs="*",
        default=None,
        metavar="VALUE",
        help="Stop on stagnation over time: [window_secs] [min_pct]. "
             "E.g. --time-min-improvement 30 0.01 or --time-min-improvement 30",
    )
    parser.add_argument(
        "--target-objective",
        type=float,
        default=None,
        help="Stop when makespan reaches this target value",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file path (prints to stdout if not set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Verbosity: -v = new-best updates, -vv = solver-switch + full solution on each new best",
    )
    parser.add_argument(
        "-q", "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Quiet mode: suppress progress output, print only final result",
    )
    return parser.parse_args(args)
