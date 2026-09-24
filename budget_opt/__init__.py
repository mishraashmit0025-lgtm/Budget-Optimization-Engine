"""Budget optimization engine: Monte Carlo scenarios + mixed-integer programming."""
from .model import Channel, GroupLimit, Problem, channels_from_frame, sample_scenarios, scenario_returns
from .optimize import Allocation, baseline_equal_split, efficient_frontier, empirical_cvar, optimize, optimize_continuous
from .simulate import monte_carlo

__all__ = [
    "Allocation", "Channel", "GroupLimit", "Problem", "baseline_equal_split", "channels_from_frame",
    "efficient_frontier", "empirical_cvar", "monte_carlo", "optimize", "optimize_continuous",
    "sample_scenarios", "scenario_returns",
]
