"""CLI: python -m budget_opt examples/channels.csv --budget 600 --step 5 --risk 0.5"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import GroupLimit, Problem, baseline_equal_split, channels_from_frame, monte_carlo, optimize


def main(argv=None):
    ap = argparse.ArgumentParser(prog="budget_opt")
    ap.add_argument("channels_csv")
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--risk", type=float, default=0.0, help="weight on CVaR in [0,1]")
    ap.add_argument("--alpha", type=float, default=0.1, help="CVaR tail fraction")
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-max", action="append", default=[], metavar="GROUP=SHARE")
    ap.add_argument("--group-min", action="append", default=[], metavar="GROUP=SHARE")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    groups: dict[str, GroupLimit] = {}
    for spec, attr in [(s, "max_share") for s in a.group_max] + [(s, "min_share") for s in a.group_min]:
        g, v = spec.split("=")
        setattr(groups.setdefault(g, GroupLimit(g)), attr, float(v))
    prob = Problem(channels_from_frame(pd.read_csv(a.channels_csv)), a.budget, a.step, list(groups.values()))
    alloc = optimize(prob, a.scenarios, a.risk, a.alpha, a.seed)
    x = alloc.vector(prob)
    mc = monte_carlo(prob, x, alpha=a.alpha)
    eq = monte_carlo(prob, baseline_equal_split(prob), alpha=a.alpha)
    if a.json:
        print(json.dumps({"status": alloc.status, "spend": alloc.spend,
                          "simulated": {k: v for k, v in mc.items() if k != "returns"}}, indent=2))
        return
    print(f"status: {alloc.status}  (solved in {alloc.solve_seconds:.2f}s)")
    w = max(len(k) for k in alloc.spend)
    for k, v in alloc.spend.items():
        print(f"  {k:<{w}}  {v:10.2f}")
    print(f"  {'total':<{w}}  {x.sum():10.2f} / {a.budget:g}")
    print(f"out-of-sample return: mean {mc['mean']:.1f}  P5 {mc['p05']:.1f}  P95 {mc['p95']:.1f}  "
          f"CVaR{int(a.alpha * 100)} {mc['cvar']:.1f}")
    print(f"equal-split baseline: mean {eq['mean']:.1f}  CVaR{int(a.alpha * 100)} {eq['cvar']:.1f}  "
          f"-> uplift {100 * (mc['mean'] / eq['mean'] - 1):+.1f}%")


if __name__ == "__main__":
    main()
