"""Walk-forward train/test splits and out-of-sample model selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


def walk_forward_splits(
    index: pd.Index,
    train_size: int,
    test_size: int,
    *,
    step: int | None = None,
    gap: int = 0,
    expanding: bool = False,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield ``(train_index, test_index)`` windows moving forward through time.

    Args:
        train_size: Bars in each training window (the initial one when ``expanding``).
        test_size: Bars in each test window.
        step: Bars to advance between splits (defaults to ``test_size``: non-overlapping tests).
        gap: Bars dropped between train and test. Set it to at least the prediction horizon so
            training labels (forward returns) do not overlap the test period.
        expanding: Grow the training window from the start instead of rolling it.
    """
    if min(train_size, test_size) < 1 or gap < 0:
        raise ValueError("train_size and test_size must be >= 1 and gap >= 0")
    step = step or test_size
    start = 0
    while True:
        train_end = start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > len(index):
            return
        train_start = 0 if expanding else start
        yield index[train_start:train_end], index[test_start:test_end]
        start += step


def annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    clean = returns.dropna()
    std = float(clean.std()) if len(clean) > 1 else 0.0
    return float(clean.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample returns stitched from test windows, and the choice made for each window."""

    returns: pd.Series
    choices: pd.DataFrame


def walk_forward_select(
    candidate_returns: Mapping[str, pd.Series],
    splits: Iterable[tuple[pd.Index, pd.Index]],
    *,
    periods_per_year: float,
    score: Callable[[pd.Series], float] | None = None,
) -> WalkForwardResult:
    """Pick the best candidate on each training window and keep only its *test* returns.

    Each candidate's return series must itself be causal (computed bar by bar from past data),
    which holds for any factor + backtest in this package. Choosing among candidates is the step
    that overfits; this function makes that choice using training data only, so the stitched
    returns are an honest out-of-sample estimate of "run the selection procedure live".
    """
    if not candidate_returns:
        raise ValueError("Need at least one candidate")
    frame = pd.DataFrame(dict(candidate_returns))
    scorer = score or (lambda r: annualized_sharpe(r, periods_per_year))
    pieces, rows = [], []
    for train_idx, test_idx in splits:
        train_scores = {name: scorer(frame.loc[train_idx, name]) for name in frame.columns}
        valid = {k: v for k, v in train_scores.items() if np.isfinite(v)}
        if not valid:
            continue
        chosen = max(valid, key=lambda k: valid[k])
        test = frame.loc[test_idx, chosen]
        pieces.append(test)
        rows.append(
            {
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "test_start": test_idx[0],
                "test_end": test_idx[-1],
                "chosen": chosen,
                "train_score": valid[chosen],
                "test_score": scorer(test),
            }
        )
    returns = pd.concat(pieces).rename("oos") if pieces else pd.Series(dtype=float, name="oos")
    return WalkForwardResult(returns=returns, choices=pd.DataFrame(rows))
