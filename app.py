"""Streamlit front end: streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from budget_opt import (
    GroupLimit, Problem, baseline_equal_split, channels_from_frame, efficient_frontier, monte_carlo, optimize,
    optimize_continuous,
)

st.set_page_config(page_title="Budget Optimization Engine", layout="wide")
st.title("Budget Optimization Engine")
st.caption(
    "Monte Carlo scenarios of uncertain channel response curves plus mixed-integer programming (PuLP/CBC) "
    "to allocate a budget under minimum spends, fixed costs and group limits."
)

EXAMPLE = Path(__file__).parent / "examples" / "channels.csv"

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Total budget", min_value=1.0, value=600.0, step=10.0)
    step = st.number_input("Allocation step", min_value=0.1, value=5.0, step=1.0,
                           help="Spend is allocated in whole multiples of this amount.")
    risk = st.slider("Risk aversion (weight on CVaR)", 0.0, 1.0, 0.3, 0.05)
    alpha = st.slider("CVaR tail", 0.01, 0.5, 0.10, 0.01, help="CVaR = average return in the worst tail fraction.")
    n_scen = st.slider("Optimization scenarios", 50, 1000, 200, 50)
    seed = st.number_input("Random seed", value=0, step=1)
    spend_all = st.checkbox("Must spend the whole budget", value=False)
    target = st.number_input("Return target (optional, 0 = none)", min_value=0.0, value=0.0)
    uploaded = st.file_uploader("Load channels CSV", type="csv")

base = pd.read_csv(uploaded) if uploaded else pd.read_csv(EXAMPLE)
st.subheader("Channels")
st.caption("Expected return at spend x = ceiling x (1 - exp(-x / scale)). The *_cv columns set the lognormal uncertainty.")
channels_df = st.data_editor(base, num_rows="dynamic", width="stretch", key="channels")

groups = sorted({str(g) for g in channels_df.get("group", pd.Series(dtype=str)).dropna() if str(g)})
limits: list[GroupLimit] = []
if groups:
    with st.expander("Group limits (share of total budget)"):
        cols = st.columns(len(groups))
        for col, g in zip(cols, groups):
            with col:
                st.markdown(f"**{g}**")
                mx = st.number_input("max share", 0.0, 1.0, 1.0, 0.05, key=f"max_{g}")
                mn = st.number_input("min share", 0.0, 1.0, 0.0, 0.05, key=f"min_{g}")
                if mx < 1 or mn > 0:
                    limits.append(GroupLimit(g, max_share=mx if mx < 1 else None, min_share=mn if mn > 0 else None))

if st.button("Optimize", type="primary"):
    try:
        problem = Problem(channels_from_frame(channels_df.dropna(subset=["name"])), budget, step, limits, spend_all)
        with st.spinner("Solving mixed-integer program..."):
            alloc = optimize(problem, n_scen, risk, alpha, int(seed))
            cont = optimize_continuous(problem, n_scen, int(seed))
        st.session_state["result"] = (problem, alloc, cont)
    except ValueError as e:
        st.error(str(e))

if "result" in st.session_state:
    problem, alloc, cont = st.session_state["result"]
    x = alloc.vector(problem)
    tgt = target or None
    mc = monte_carlo(problem, x, alpha=alpha, target=tgt)
    eq = monte_carlo(problem, baseline_equal_split(problem), alpha=alpha, target=tgt)

    st.subheader("Result")
    c = st.columns(5)
    c[0].metric("Expected return", f"{mc['mean']:,.1f}", f"{100 * (mc['mean'] / eq['mean'] - 1):+.1f}% vs equal split")
    c[1].metric(f"CVaR {int(alpha * 100)}%", f"{mc['cvar']:,.1f}", f"{mc['cvar'] - eq['cvar']:+,.1f} vs equal split")
    c[2].metric("Spend incl. fixed costs", f"{mc['cost']:,.1f}", f"of {problem.budget:,.0f}", delta_color="off")
    c[3].metric("ROI", f"{100 * mc['roi']:.1f}%")
    if tgt:
        c[4].metric("P(return >= target)", f"{100 * mc['p_meet_target']:.1f}%")
    else:
        c[4].metric("Solver", alloc.status, f"{alloc.solve_seconds:.2f}s", delta_color="off")
    st.caption(f"Out-of-sample Monte Carlo on 20,000 fresh scenarios (standard error of mean {mc['mc_std_error']:.2f}). "
               f"Continuous SciPy relaxation (no integrality/fixed costs) expects {cont.expected_return:,.1f}.")

    left, right = st.columns(2)
    names = [ch.name for ch in problem.channels]
    with left:
        fig = go.Figure([
            go.Bar(name="Optimized", x=names, y=x),
            go.Bar(name="Equal split", x=names, y=baseline_equal_split(problem), opacity=0.5),
        ])
        fig.update_layout(title="Allocation", barmode="group", yaxis_title="spend", height=420, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure([
            go.Histogram(x=mc["returns"], name="Optimized", opacity=0.75, nbinsx=60),
            go.Histogram(x=eq["returns"], name="Equal split", opacity=0.5, nbinsx=60),
        ])
        fig.add_vline(x=mc["cvar"], line_dash="dash", annotation_text=f"CVaR {int(alpha * 100)}%")
        if tgt:
            fig.add_vline(x=tgt, line_dash="dot", annotation_text="target")
        fig.update_layout(title="Simulated total return", barmode="overlay", xaxis_title="return", height=420, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame({
        "channel": names,
        "spend": x,
        "share": x / max(x.sum(), 1e-9),
        "expected return": [ch.response(v) for ch, v in zip(problem.channels, x)],
        "marginal return / unit": [ch.ceiling / ch.scale * np.exp(-v / ch.scale) for ch, v in zip(problem.channels, x)],
    })
    st.dataframe(table.style.format({"spend": "{:,.1f}", "share": "{:.1%}", "expected return": "{:,.1f}",
                                     "marginal return / unit": "{:.3f}"}), width="stretch")
    st.download_button("Download allocation CSV", table.to_csv(index=False), "allocation.csv", "text/csv")

    with st.expander("Risk-return frontier"):
        if st.button("Compute frontier (5 solves)"):
            fr = efficient_frontier(problem, n_scenarios=n_scen, alpha=alpha, seed=int(seed))
            fig = go.Figure(go.Scatter(
                x=[f["cvar"] for f in fr], y=[f["expected_return"] for f in fr], mode="lines+markers+text",
                text=[f"λ={f['risk_aversion']}" for f in fr], textposition="top center"))
            fig.update_layout(xaxis_title=f"CVaR {int(alpha * 100)}% (in-sample)", yaxis_title="expected return", height=400)
            st.plotly_chart(fig, width="stretch")
