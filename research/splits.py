"""Anchored walk-forward split generation with an embargo gap and an
optional trailing holdout window carved off before the folds are cut.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    name: str
    start: int   # inclusive bar index
    end: int     # exclusive bar index

    @property
    def n_bars(self) -> int:
        return self.end - self.start


def anchored_walk_forward(
    n_bars: int,
    *,
    reserve_bars: int = 0,
    n_folds: int = 4,
    embargo: int = 10,
    holdout_frac: float = 0.20,
) -> tuple:
    """Build ``n_folds`` anchored ``(train, valid)`` pairs over
    ``[reserve_bars, holdout_start)``, plus a ``holdout`` window over the
    trailing ``holdout_frac`` of all bars.

    Anchored: every fold's train window starts at ``reserve_bars`` (enough
    leading history for indicator warmup) and only its end advances --
    matching what a live strategy would actually have available at each
    point in time, unlike a sliding window that would forget early history.
    ``embargo`` bars are dropped between each train window's end and its
    paired valid window's start, so a position or an indicator's lookback
    can't bridge the boundary.

    ``holdout_frac=0.0`` is allowed and means no reserved tail: the folds
    then span every bar after warmup, and the empty ``holdout`` window comes
    back as ``[n_bars, n_bars)``. That is what a caller wants when nothing is
    being *selected* on this data -- with the parameters already fixed there
    is no search to hold a window back from, and the last fold's valid window
    is the most recent stretch either way. A positive fraction still carves
    the tail off first, for a caller that does have something to protect.
    """
    if not (0.0 <= holdout_frac < 1.0):
        raise ValueError('holdout_frac must be in [0, 1)')
    if n_folds < 1:
        raise ValueError('n_folds must be >= 1')
    if reserve_bars < 0 or reserve_bars >= n_bars:
        raise ValueError(f'reserve_bars={reserve_bars} leaves no bars in a {n_bars}-bar market')

    holdout_start = max(int(round(n_bars * (1.0 - holdout_frac))), reserve_bars)
    holdout = Window('holdout', holdout_start, n_bars)

    # Chop [reserve_bars, holdout_start) into n_folds + 1 equal chunks, not
    # n_folds: chunk 0 anchors fold 1's train window, and chunks 1..n_folds
    # each extend train by one more chunk before its embargo+valid pair.
    # That leaves the *last* fold a full spare chunk (plus the floor-
    # division remainder) to absorb its embargo in, so `valid_end ==
    # holdout_start` by construction can never invert into `valid_start >=
    # valid_end` the way a naive n_folds-way split can for the final fold.
    usable = holdout_start - reserve_bars
    fold_size = usable // (n_folds + 1)
    if fold_size <= embargo:
        raise ValueError(
            f"Only {usable} bars between reserve_bars={reserve_bars} and the holdout "
            f"boundary at {holdout_start} -- {n_folds} folds at embargo={embargo} need "
            f"fold_size > embargo (got fold_size={fold_size}). Use fewer folds, a "
            f"smaller embargo, a smaller holdout_frac, or a longer history."
        )

    folds = []
    for k in range(1, n_folds + 1):
        train_end = reserve_bars + fold_size * k
        valid_start = train_end + embargo
        valid_end = holdout_start if k == n_folds else reserve_bars + fold_size * (k + 1)
        train = Window(f'train_{k}', reserve_bars, train_end)
        valid = Window(f'valid_{k}', valid_start, valid_end)
        folds.append((train, valid))

    return folds, holdout
