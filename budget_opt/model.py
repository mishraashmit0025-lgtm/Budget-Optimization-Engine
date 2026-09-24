"""Problem definition: channels with uncertain saturating response curves."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Channel:
    """A spend channel / program whose return saturates with spend.

    Expected return at spend x is ``ceiling * (1 - exp(-x / scale))``: the first
    unit of spend earns about ``ceiling / scale`` and returns diminish toward ``ceiling``.
    ``ceiling_cv`` and ``scale_cv`` are coefficients of variation of the lognormal
    uncertainty on those parameters.
    """

    name: str
    ceiling: float
    scale: float
    ceiling_cv: float = 0.2
    scale_cv: float = 0.15
    min_spend: float = 0.0  # if the channel is funded at all, spend at least this much
    max_spend: float = float("inf")
    fixed_cost: float = 0.0  # one-off cost incurred if the channel is funded
    group: str = ""

    def response(self, x, ceiling=None, scale=None):
        c = self.ceiling if ceiling is None else ceiling
        s = self.scale if scale is None else scale
        return c * (1 - np.exp(-np.asarray(x, dtype=float) / s))


@dataclass
class GroupLimit:
    group: str
    max_share: float | None = None  # at most this share of the total budget
    min_share: float | None = None


@dataclass
class Problem:
    channels: list[Channel]
    budget: float
    step: float = 1.0  # spend is allocated in integer multiples of this unit
    groups: list[GroupLimit] = field(default_factory=list)
    spend_all: bool = False  # force the whole budget to be used

    def validate(self) -> None:
        if self.budget <= 0 or self.step <= 0:
            raise ValueError("budget and step must be positive")
        if not self.channels:
            raise ValueError("need at least one channel")
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        for c in self.channels:
            if c.ceiling <= 0 or c.scale <= 0:
                raise ValueError(f"{c.name}: ceiling and scale must be positive")
            if c.min_spend > min(c.max_spend, self.budget):
                raise ValueError(f"{c.name}: min_spend exceeds max_spend/budget")

    def units(self, c: Channel) -> int:
        return int(np.floor(min(c.max_spend, self.budget) / self.step + 1e-9))


def sample_scenarios(channels: list[Channel], n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw (ceiling, scale) parameter scenarios, shape (n, n_channels) each.

    Lognormal draws are mean-preserving: E[ceiling] equals the channel's ``ceiling``.
    """
    rng = np.random.default_rng(seed)

    def lognormal(mean, cv, size):
        sigma2 = np.log1p(cv**2)
        return rng.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2), size)

    C = np.column_stack([lognormal(c.ceiling, c.ceiling_cv, n) if c.ceiling_cv > 0 else np.full(n, c.ceiling) for c in channels])
    S = np.column_stack([lognormal(c.scale, c.scale_cv, n) if c.scale_cv > 0 else np.full(n, c.scale) for c in channels])
    return C, S


def scenario_returns(channels: list[Channel], alloc: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Total return in every scenario for a given spend vector."""
    alloc = np.asarray(alloc, dtype=float)
    return (C * (1 - np.exp(-alloc[None, :] / S))).sum(axis=1)


def channels_from_frame(df: pd.DataFrame) -> list[Channel]:
    cols = {f for f in Channel.__dataclass_fields__}
    out = []
    for rec in df.to_dict(orient="records"):
        kw = {k: v for k, v in rec.items() if k in cols and not (isinstance(v, float) and np.isnan(v))}
        if "group" in kw:
            kw["group"] = str(kw["group"])
        out.append(Channel(**kw))
    return out
