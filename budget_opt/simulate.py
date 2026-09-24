"""Out-of-sample Monte Carlo evaluation of an allocation."""
from __future__ import annotations

import numpy as np

from .model import Problem, sample_scenarios, scenario_returns
from .optimize import empirical_cvar


def monte_carlo(problem: Problem, spend: np.ndarray, n: int = 20000, seed: int = 12345,
                alpha: float = 0.1, target: float | None = None) -> dict:
    """Simulate returns on fresh scenarios (a different seed from the optimizer's)."""
    C, S = sample_scenarios(problem.channels, n, seed)
    r = scenario_returns(problem.channels, spend, C, S)
    cost = float(np.sum(spend) + sum(c.fixed_cost for c, x in zip(problem.channels, spend) if x > 0))
    out = {
        "returns": r,
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "p05": float(np.percentile(r, 5)),
        "p50": float(np.percentile(r, 50)),
        "p95": float(np.percentile(r, 95)),
        "cvar": empirical_cvar(r, alpha),
        "cost": cost,
        "roi": float(r.mean() / cost - 1) if cost > 0 else float("nan"),
        "mc_std_error": float(r.std(ddof=1) / np.sqrt(n)),
    }
    if target is not None:
        out["p_meet_target"] = float((r >= target).mean())
    return out
