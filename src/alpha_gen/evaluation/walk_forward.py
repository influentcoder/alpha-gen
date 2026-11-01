"""Walk-forward train/test splits for time-ordered data."""

from __future__ import annotations

from collections.abc import Iterator

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
