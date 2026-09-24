import itertools

import numpy as np
import pandas as pd
import pytest

from budget_opt import (
    Channel, GroupLimit, Problem, baseline_equal_split, channels_from_frame, empirical_cvar, monte_carlo,
    optimize, optimize_continuous, sample_scenarios, scenario_returns,
)


def small_problem(**kw):
    chans = [
        Channel("A", 100, 20, 0.3, 0.2, min_spend=10, max_spend=60, fixed_cost=5, group="g1"),
        Channel("B", 80, 10, 0.1, 0.1, max_spend=40, group="g1"),
        Channel("C", 150, 50, 0.5, 0.3, min_spend=20, fixed_cost=10, group="g2"),
    ]
    return Problem(chans, budget=kw.pop("budget", 100), step=kw.pop("step", 5), **kw)


def brute_force(problem, C, S, lam, alpha):
    best, best_x = -np.inf, None
    grids = [np.arange(problem.units(c) + 1) * problem.step for c in problem.channels]
    for x in itertools.product(*grids):
        x = np.array(x, float)
        on = x > 0
        cost = x.sum() + sum(c.fixed_cost for c, o in zip(problem.channels, on) if o)
        if cost > problem.budget + 1e-9:
            continue
        if any(o and xi < c.min_spend for c, o, xi in zip(problem.channels, on, x)):
            continue
        if any(xi > c.max_spend for c, xi in zip(problem.channels, x)):
            continue
        ok = True
        for g in problem.groups:
            s = sum(xi for c, xi in zip(problem.channels, x) if c.group == g.group)
            if g.max_share is not None and s > g.max_share * problem.budget + 1e-9:
                ok = False
            if g.min_share is not None and s < g.min_share * problem.budget - 1e-9:
                ok = False
        if not ok:
            continue
        r = scenario_returns(problem.channels, x, C, S)
        v = (1 - lam) * r.mean() + lam * empirical_cvar(r, alpha)
        if v > best:
            best, best_x = v, x
    return best, best_x


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_milp_matches_brute_force(lam):
    # alpha * K is an integer, so the RU LP's CVaR equals the empirical CVaR exactly
    p = small_problem(groups=[GroupLimit("g1", max_share=0.6)])
    C, S = sample_scenarios(p.channels, 40, seed=3)
    a = optimize(p, risk_aversion=lam, alpha=0.25, scenarios=(C, S))
    best, _ = brute_force(p, C, S, lam, 0.25)
    assert a.status == "Optimal"
    assert a.objective == pytest.approx(best, rel=1e-6)


def test_constraints_respected():
    p = small_problem(groups=[GroupLimit("g1", max_share=0.4), GroupLimit("g2", min_share=0.3)])
    a = optimize(p, n_scenarios=50)
    x = a.vector(p)
    fixed = sum(c.fixed_cost for c, xi in zip(p.channels, x) if xi > 0)
    assert x.sum() + fixed <= p.budget + 1e-9
    assert x[0] + x[1] <= 0.4 * p.budget + 1e-9
    assert x[2] >= 0.3 * p.budget - 1e-9
    for c, xi in zip(p.channels, x):
        assert xi == 0 or c.min_spend <= xi <= c.max_spend
        assert abs(xi / p.step - round(xi / p.step)) < 1e-9


def test_continuous_relaxation_bounds_milp():
    chans = [Channel(n, 100 + 20 * i, 15 + 5 * i) for i, n in enumerate("ABCD")]
    p = Problem(chans, budget=120, step=1)
    milp = optimize(p, n_scenarios=100, seed=1)
    cont = optimize_continuous(p, n_scenarios=100, seed=1)
    assert cont.expected_return >= milp.expected_return - 1e-6
    assert milp.expected_return >= 0.995 * cont.expected_return  # step of 1 is fine-grained


def test_risk_aversion_trades_mean_for_tail():
    p = small_problem()
    C, S = sample_scenarios(p.channels, 200, seed=0)
    neutral = optimize(p, risk_aversion=0, scenarios=(C, S))
    averse = optimize(p, risk_aversion=1, scenarios=(C, S))
    assert neutral.expected_return >= averse.expected_return - 1e-6
    assert averse.cvar >= neutral.cvar - 1e-6


def test_beats_equal_split_out_of_sample():
    df = pd.read_csv("examples/channels.csv")
    p = Problem(channels_from_frame(df), budget=600, step=5)
    a = optimize(p, n_scenarios=200)
    opt = monte_carlo(p, a.vector(p), n=5000)
    eq = monte_carlo(p, baseline_equal_split(p), n=5000)
    assert opt["mean"] > eq["mean"]


def test_infeasible_min_share_raises():
    p = small_problem(groups=[GroupLimit("g1", min_share=0.99)])
    with pytest.raises(ValueError):
        optimize(p, n_scenarios=20)


def test_scenarios_are_mean_preserving():
    chans = [Channel("A", 100, 10, ceiling_cv=0.5)]
    C, _ = sample_scenarios(chans, 200000, seed=0)
    assert C.mean() == pytest.approx(100, rel=0.01)
