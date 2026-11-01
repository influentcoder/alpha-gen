"""Predictive-power and performance metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int) -> pd.Series:
    a_d = a.sub(a.mean(axis=1), axis=0)
    b_d = b.sub(b.mean(axis=1), axis=0)
    cov = (a_d * b_d).sum(axis=1)
    denom = np.sqrt((a_d**2).sum(axis=1) * (b_d**2).sum(axis=1)).replace(0.0, np.nan)
    corr = cov / denom
    corr[a.notna().sum(axis=1) < min_obs] = np.nan
    return corr


def information_coefficient(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    min_obs: int = 3,
) -> pd.Series:
    """Cross-sectional correlation between signal and forward returns, one value per bar.

    ``method="spearman"`` (rank IC, the default) is robust to outliers; ``"pearson"`` uses raw
    values. Only symbols with both a signal and a return in a given bar are used.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    sig, fwd = signal.align(forward_returns, join="inner")
    mask = sig.notna() & fwd.notna()
    sig, fwd = sig.where(mask), fwd.where(mask)
    if method == "spearman":
        sig, fwd = sig.rank(axis=1), fwd.rank(axis=1)
    return _rowwise_corr(sig, fwd, min_obs).rename("ic")


@dataclass(frozen=True)
class ICSummary:
    """Summary statistics of an IC time series.

    ``t_stat`` assumes independent observations; for horizons > 1 bar the forward returns overlap
    and it is overstated by roughly ``sqrt(horizon)``.
    """

    mean: float
    std: float
    ir: float
    t_stat: float
    hit_rate: float
    n: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def ic_summary(ic: pd.Series) -> ICSummary:
    clean = ic.dropna()
    n = len(clean)
    if n < 2:
        nan = float("nan")
        return ICSummary(nan, nan, nan, nan, nan, n)
    mean, std = float(clean.mean()), float(clean.std())
    ir = mean / std if std > 0 else float("nan")
    return ICSummary(
        mean=mean,
        std=std,
        ir=ir,
        t_stat=ir * math.sqrt(n),
        hit_rate=float((clean > 0).mean()),
        n=n,
    )


def ic_decay(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizons: Iterable[int] = (1, 2, 4, 8, 24, 72),
    *,
    method: str = "spearman",
) -> pd.Series:
    """Mean IC of ``signal`` against forward returns at several horizons (alpha decay curve)."""
    out = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        out[h] = information_coefficient(signal, fwd, method=method).mean()
    return pd.Series(out, name="mean_ic").rename_axis("horizon")


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Per-bar traded notional as a fraction of capital: ``sum(|w_t - w_{t-1}|)``."""
    w = weights.fillna(0.0)
    traded = w.diff().abs().sum(axis=1)
    if len(w):
        traded.iloc[0] = w.iloc[0].abs().sum()
    return traded.rename("turnover")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")


def performance_summary(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    """Annualised return/vol, Sharpe, Sortino, max drawdown and hit rate of a return series."""
    r = returns.dropna()
    if len(r) < 2:
        return {}
    ann_return = float(r.mean() * periods_per_year)
    ann_vol = float(r.std() * math.sqrt(periods_per_year))
    downside = float(r[r < 0].std() * math.sqrt(periods_per_year)) if (r < 0).sum() > 1 else 0.0
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else float("nan"),
        "sortino": ann_return / downside if downside > 0 else float("nan"),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "total_return": float((1.0 + r).prod() - 1.0),
    }
