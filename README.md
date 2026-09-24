# Budget Optimization Engine

This tool splits a budget across channels or programs whose returns are **uncertain** and **diminishing**. It combines two techniques:

1. **Monte Carlo simulation.** Each channel's response curve `ceiling · (1 − e^(−spend/scale))` has lognormal uncertainty on both parameters. The tool draws hundreds of scenarios to optimize over, then re-evaluates the chosen plan on 20,000 fresh scenarios.
2. **Mixed-integer programming (PuLP + CBC).** Spend is allocated in whole steps, and each funded channel has a **minimum spend** and a **fixed activation cost**. You can also set **group share limits** (e.g. digital ≤ 60%) and a **risk preference**:

   maximize  (1 − λ) · E[return] + λ · CVaR_α(return)

   CVaR_α is the average return across the worst α-fraction of scenarios, so raising λ trades expected return for a better worst case.

A SciPy (SLSQP) continuous relaxation serves as a cross-check and upper bound. A **Streamlit** app lets you edit channels, set constraints, compare against an equal-split plan, and plot the risk-return frontier.

## How the MILP stays exact

Every scenario's response curve is concave, so the tool represents it at integer spend levels with *incremental segment* variables `z_ij ∈ [0,1]` whose marginal returns decrease in `j`. Filling earlier segments first is at least as good in **every** scenario, so the optimum fills them in order and each scenario's return is represented exactly. There are no piecewise-linear approximation errors and no SOS2 constraints. The test suite checks the MILP optimum against brute-force enumeration for λ ∈ {0, 0.5, 1}.

## Quick start

```bash
pip install -r requirements.txt

# command line
python -m budget_opt examples/channels.csv --budget 600 --step 5 --risk 0.3 --group-max digital=0.6

# web app
streamlit run app.py

# tests
python -m pytest -q
```

Example CLI output:

```
status: Optimal  (solved in 0.41s)
  Search Ads                 100.00
  Social Media                95.00
  ...
out-of-sample return: mean 1501.4  P5 1225.8  P95 1847.1  CVaR10 1214.6
equal-split baseline: mean 1405.9  CVaR10 1163.5  -> uplift +6.8%
```

(The example channel data in `examples/channels.csv` is illustrative, not real campaign data.)

## Library use

```python
from budget_opt import Channel, GroupLimit, Problem, optimize, monte_carlo

chans = [
    Channel("Search", ceiling=420, scale=60, min_spend=10, max_spend=200, group="digital"),
    Channel("TV", ceiling=650, scale=180, ceiling_cv=0.35, min_spend=60, fixed_cost=15, group="offline"),
]
problem = Problem(chans, budget=300, step=5, groups=[GroupLimit("digital", max_share=0.6)])
plan = optimize(problem, n_scenarios=300, risk_aversion=0.5, alpha=0.1)
print(plan.spend, monte_carlo(problem, plan.vector(problem))["mean"])
```

## Layout

```
budget_opt/model.py      Channel / Problem definitions, scenario sampling
budget_opt/optimize.py   MILP (PuLP/CBC) with CVaR, SciPy relaxation, frontier, baselines
budget_opt/simulate.py   out-of-sample Monte Carlo evaluation
app.py                   Streamlit UI
tests/                   brute-force optimality, constraints, relaxation bound, app smoke test
```
