"""Scenario-based mixed-integer budget allocation (PuLP/CBC) plus a SciPy baseline.

Formulation
-----------
Spend on channel i is ``step * n_i`` with integer n_i. Because every scenario's
response curve is concave, it is represented exactly at integer points by
incremental segment variables z_ij in [0, 1] with decreasing marginal returns
d_kij = r_ki(j*step) - r_ki((j-1)*step). Filling earlier segments first is always
at least as good in *every* scenario, so the optimum fills in order and the
scenario returns R_k = sum_ij d_kij z_ij are exact.

Objective: maximize (1 - lam) * E[R] + lam * CVaR_alpha(R), where CVaR is the mean
of the worst alpha-fraction of scenario returns (Rockafellar-Uryasev LP form).
Binary y_i models channel activation: min spend, max spend and fixed cost.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pulp
from scipy.optimize import minimize

from .model import Problem, sample_scenarios, scenario_returns


@dataclass
class Allocation:
    spend: dict[str, float]
    expected_return: float
    cvar: float
    objective: float
    status: str
    solve_seconds: float

    def vector(self, problem: Problem) -> np.ndarray:
        return np.array([self.spend[c.name] for c in problem.channels])


def _solver(time_limit: int):
    """CBC via PuLP; prefers a system/pip CBC (COIN_CMD) and falls back to PuLP's bundled binary."""
    coin = pulp.COIN_CMD(msg=False, timeLimit=time_limit)
    if coin.available():
        return coin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)


def empirical_cvar(returns: np.ndarray, alpha: float) -> float:
    """Mean of the worst ceil(alpha * n) outcomes."""
    r = np.sort(np.asarray(returns))
    k = max(1, int(np.ceil(alpha * len(r) - 1e-9)))
    return float(r[:k].mean())


def optimize(
    problem: Problem,
    n_scenarios: int = 200,
    risk_aversion: float = 0.0,
    alpha: float = 0.1,
    seed: int = 0,
    time_limit: int = 60,
    scenarios: tuple[np.ndarray, np.ndarray] | None = None,
) -> Allocation:
    problem.validate()
    if not 0 <= risk_aversion <= 1:
        raise ValueError("risk_aversion must be in [0, 1]")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    chans, step = problem.channels, problem.step
    C, S = scenarios if scenarios is not None else sample_scenarios(chans, n_scenarios, seed)
    K = C.shape[0]

    m = pulp.LpProblem("budget_allocation", pulp.LpMaximize)

    def var(name, low=None, up=None, cat="Continuous"):
        # PuLP >= 3.3 attaches variables to the model; older versions construct them directly
        if hasattr(m, "add_variable"):
            return m.add_variable(name, low, up, cat=cat)
        return pulp.LpVariable(name, low, up, cat=cat)

    z, y, delta = {}, {}, {}
    for i, c in enumerate(chans):
        U = problem.units(c)
        grid = step * np.arange(U + 1)
        vals = C[:, [i]] * (1 - np.exp(-grid[None, :] / S[:, [i]]))  # (K, U+1)
        delta[i] = np.diff(vals, axis=1)  # (K, U), decreasing in j
        z[i] = [var(f"z_{i}_{j}", 0, 1) for j in range(U)]
        y[i] = var(f"y_{i}", 0, 1, cat="Binary")
    n = {i: var(f"n_{i}", 0, problem.units(c), cat="Integer") for i, c in enumerate(chans)}
    spend = {i: step * n[i] for i in n}

    for i, c in enumerate(chans):
        m += pulp.lpSum(z[i]) == n[i]
        m += n[i] <= problem.units(c) * y[i]
        if c.min_spend > 0:
            m += spend[i] >= c.min_spend * y[i]
    total = pulp.lpSum(spend.values()) + pulp.lpSum(c.fixed_cost * y[i] for i, c in enumerate(chans))
    if problem.spend_all:
        # allow for rounding to the step grid: total within one step of the budget
        m += total >= problem.budget - step + 1e-9
    m += total <= problem.budget

    for g in problem.groups:
        members = [i for i, c in enumerate(chans) if c.group == g.group]
        if not members:
            raise ValueError(f"group {g.group!r} has no channels")
        gsum = pulp.lpSum(spend[i] for i in members)
        if g.max_share is not None:
            m += gsum <= g.max_share * problem.budget
        if g.min_share is not None:
            m += gsum >= g.min_share * problem.budget

    flat_vars = [v for i in z for v in z[i]]
    D = np.hstack([delta[i] for i in z])  # (K, total segments)
    R = [pulp.LpAffineExpression(zip(flat_vars, D[k].tolist())) for k in range(K)]
    mean_R = pulp.LpAffineExpression(zip(flat_vars, D.mean(axis=0).tolist()))
    if risk_aversion > 0:
        eta = var("eta")
        u = [var(f"u_{k}", 0) for k in range(K)]
        for k in range(K):
            m += R[k] + u[k] - eta >= 0
        cvar_expr = eta - pulp.lpSum(u) * (1.0 / (alpha * K))
        m += (1 - risk_aversion) * mean_R + risk_aversion * cvar_expr
    else:
        m += mean_R

    t0 = time.perf_counter()
    m.solve(_solver(time_limit))
    secs = time.perf_counter() - t0
    status = pulp.LpStatus[m.status]
    if status not in ("Optimal",):
        if status == "Not Solved" or m.status == 0:
            status = "TimeLimit"
        if status in ("Infeasible", "Unbounded", "Undefined"):
            raise ValueError(f"optimization {status.lower()}: check budget, min spends and group limits")

    x = np.array([step * round(n[i].value() or 0) for i in range(len(chans))])
    rets = scenario_returns(chans, x, C, S)
    # the RU LP uses a continuous tail fraction; report the exact empirical CVaR
    ev, cv = float(rets.mean()), empirical_cvar(rets, alpha)
    return Allocation(
        spend={c.name: float(v) for c, v in zip(chans, x)},
        expected_return=ev,
        cvar=cv,
        objective=(1 - risk_aversion) * ev + risk_aversion * cv,
        status=status,
        solve_seconds=secs,
    )


def optimize_continuous(problem: Problem, n_scenarios: int = 200, seed: int = 0) -> Allocation:
    """Continuous relaxation (no integrality, activation or fixed costs) solved with SciPy SLSQP.

    Useful as an upper bound on the expected-return MILP and as a sanity check.
    """
    problem.validate()
    chans = problem.channels
    C, S = sample_scenarios(chans, n_scenarios, seed)
    ub = np.array([min(c.max_spend, problem.budget) for c in chans])

    f = lambda x: -scenario_returns(chans, x, C, S).mean()
    g = lambda x: -((C / S) * np.exp(-x[None, :] / S)).mean(axis=0)
    cons = [{"type": "ineq", "fun": lambda x: problem.budget - x.sum(), "jac": lambda x: -np.ones_like(x)}]
    for grp in problem.groups:
        mask = np.array([c.group == grp.group for c in chans], dtype=float)
        if grp.max_share is not None:
            cons.append({"type": "ineq", "fun": lambda x, m=mask, s=grp.max_share: s * problem.budget - m @ x, "jac": lambda x, m=mask: -m})
        if grp.min_share is not None:
            cons.append({"type": "ineq", "fun": lambda x, m=mask, s=grp.min_share: m @ x - s * problem.budget, "jac": lambda x, m=mask: m})
    x0 = np.minimum(ub, problem.budget / len(chans))
    t0 = time.perf_counter()
    res = minimize(f, x0, jac=g, method="SLSQP", bounds=list(zip(np.zeros(len(chans)), ub)), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    x = np.clip(res.x, 0, ub)
    rets = scenario_returns(chans, x, C, S)
    return Allocation({c.name: float(v) for c, v in zip(chans, x)}, float(rets.mean()), empirical_cvar(rets, 0.1),
                      float(rets.mean()), "Optimal" if res.success else res.message, time.perf_counter() - t0)


def baseline_equal_split(problem: Problem) -> np.ndarray:
    """Split the budget equally (respecting max spend), rounded down to the step grid."""
    per = problem.budget / len(problem.channels)
    x = np.array([min(per, c.max_spend) for c in problem.channels])
    return np.floor(x / problem.step) * problem.step


def efficient_frontier(problem: Problem, lambdas=(0, 0.25, 0.5, 0.75, 1.0), n_scenarios: int = 200,
                       alpha: float = 0.1, seed: int = 0) -> list[dict]:
    scen = sample_scenarios(problem.channels, n_scenarios, seed)
    out = []
    for lam in lambdas:
        a = optimize(problem, risk_aversion=lam, alpha=alpha, scenarios=scen)
        out.append({"risk_aversion": lam, "expected_return": a.expected_return, "cvar": a.cvar, "spend": a.spend})
    return out
