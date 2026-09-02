"""Trade log -> ``(X, y, w)``, and the purge rule that keeps folds honest.

One realized trade is one training sample. The ledger already produces
everything needed (``core/ledger.py``): ``open_bar``, ``close_bar``,
``net_pnl`` net of commission, ``direction``, ``open_price``, ``size``.

Two alignment facts do all the work of preventing leakage, and both are easy
to get wrong by one bar:

* **Features come from ``open_bar - 1``.** An order placed in ``on_bar`` fills
  at the *next* bar's open, so ``open_bar`` is the fill bar and the decision
  was made on the close before it. Reading features at ``open_bar`` would hand
  the model the bar it is supposed to predict into.
* **A training trade must have *closed* before the training window ends.**
  Not merely opened. A trade still running at the boundary has a label that is
  only knowable later; including it is exactly the leak that purging exists to
  prevent (López de Prado, *AFML* ch.7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from datafeed.products import product_costs

from meta.features import FEATURE_NAMES, feature_row

logger = logging.getLogger(__name__)

__all__ = ['Samples', 'samples_from_trades', 'purged_train_mask', 'window_mask']


@dataclass
class Samples:
    """Design matrix plus the bookkeeping needed to split it in time."""

    X: np.ndarray            # float64[n, n_features]
    y: np.ndarray            # int8[n], 1 = the trade made money
    w: np.ndarray            # float64[n], mean 1
    ret: np.ndarray          # float64[n], return on entry notional
    open_bar: np.ndarray     # int64[n], fill bar
    close_bar: np.ndarray    # int64[n]
    decision_bar: np.ndarray # int64[n], == open_bar - 1, where features were read
    symbol: np.ndarray       # object[n]
    side: np.ndarray         # int8[n]
    feature_names: Sequence[str] = tuple(FEATURE_NAMES)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> 'Samples':
        return Samples(
            X=self.X[mask], y=self.y[mask], w=self.w[mask], ret=self.ret[mask],
            open_bar=self.open_bar[mask], close_bar=self.close_bar[mask],
            decision_bar=self.decision_bar[mask], symbol=self.symbol[mask],
            side=self.side[mask], feature_names=self.feature_names,
        )

    @property
    def base_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else float('nan')


def samples_from_trades(trade_logs: List[dict], arrays: dict, *, label_threshold: float = 0.0) -> Samples:
    """Build the training set from one unfiltered backtest's trade log.

    ``arrays`` is ``meta.features.build_feature_arrays``' output on the *same*
    market the trades came from, with bar indices on the same basis
    (``research.runner_api.run_window`` already remaps them back to absolute).

    Dropped, with a count logged for each reason:

    * ``open_at_end`` trades -- the run ended mid-position, so the label is
      censored, not negative.
    * ``open_bar == 0`` -- there is no decision bar before the first bar.
    * rows with an incomplete feature vector (a product still in warmup).

    The label is ``trade_ret > label_threshold`` where ``trade_ret`` is P&L
    over entry notional. Normalising by notional is what makes a JM trade and
    a CF trade comparable; the *sign* is unaffected by it (both gross P&L and
    commission scale with size), so ``label_threshold=0`` is exactly
    ``net_pnl > 0``. Sample weight is ``|trade_ret|``, normalised to mean 1,
    so a scratch trade near the decision boundary counts for less than one
    that ran -- a coin-flip trade genuinely carries less information about the
    entry than a decisive one.
    """
    rows, censored, no_decision_bar, no_features = [], 0, 0, 0

    for t in trade_logs:
        if t.get('open_at_end'):
            censored += 1
            continue
        open_bar = t.get('open_bar')
        close_bar = t.get('close_bar')
        if open_bar is None or close_bar is None or open_bar <= 0:
            no_decision_bar += 1
            continue

        side = 1 if t['direction'] == 'long' else -1
        decision_bar = int(open_bar) - 1
        x = feature_row(arrays, t['symbol'], decision_bar, side)
        if x is None:
            no_features += 1
            continue

        notional = abs(float(t['open_price'])) * abs(int(t['size'])) * product_costs(t['symbol'])['multiplier']
        if notional <= 0:
            no_features += 1
            continue
        ret = float(t['net_pnl']) / notional

        rows.append((x, ret, int(open_bar), int(close_bar), decision_bar, t['symbol'], side))

    if censored or no_decision_bar or no_features:
        logger.info(
            'Samples: dropped %d open-at-end, %d without a decision bar, %d without features.',
            censored, no_decision_bar, no_features,
        )
    if not rows:
        empty_i, empty_f = np.empty(0, dtype='int64'), np.empty(0, dtype='float64')
        return Samples(
            X=np.empty((0, len(FEATURE_NAMES))), y=np.empty(0, dtype='int8'), w=empty_f,
            ret=empty_f, open_bar=empty_i, close_bar=empty_i, decision_bar=empty_i,
            symbol=np.empty(0, dtype=object), side=np.empty(0, dtype='int8'),
        )

    X = np.vstack([r[0] for r in rows])
    ret = np.array([r[1] for r in rows], dtype='float64')
    y = (ret > label_threshold).astype('int8')
    w = np.abs(ret)
    w = w / w.mean() if w.mean() > 0 else np.ones_like(w)

    return Samples(
        X=X, y=y, w=w, ret=ret,
        open_bar=np.array([r[2] for r in rows], dtype='int64'),
        close_bar=np.array([r[3] for r in rows], dtype='int64'),
        decision_bar=np.array([r[4] for r in rows], dtype='int64'),
        symbol=np.array([r[5] for r in rows], dtype=object),
        side=np.array([r[6] for r in rows], dtype='int8'),
    )


def purged_train_mask(
    samples: Samples, train_window, valid_window=None, embargo: int = 0,
    *, anchor_to_window: bool = False,
) -> np.ndarray:
    """Training rows for one fold: every trade that **closed before the
    training window ends**.

    ``close_bar < train.end`` is the purge, and it is the whole correctness
    rule: a trade still running at the boundary has a label that is only
    knowable later. Under ``research.splits.anchored_walk_forward``'s layout
    (valid starts at ``train.end + embargo``) it already implies no overlap
    with the validation window, but the assertion below is written out anyway
    -- it costs nothing, and it is what would catch a future change to the
    split scheme instead of letting the leak run silently for a study or two.

    ``anchor_to_window`` additionally drops trades opened before
    ``train_window.start``. It is **off by default**, because that bound is a
    windowing choice rather than a leakage rule and on this universe it is an
    expensive one: ``train_window.start`` is ``reserve_bars``, which
    ``probe_warmup`` sets from the *slowest* product, so one late listing
    (SA, from 2019-12) pushes it to bar 1261 and throws away every 2015-2019
    trade from the eight products that were warm the whole time. Those trades
    are realized, labelled, and closed years before any validation window
    begins; excluding them costs most of the training set and buys nothing.
    Turn it on only when the earlier regime is believed not to generalise.
    """
    mask = samples.close_bar < train_window.end
    if anchor_to_window:
        mask = mask & (samples.open_bar >= train_window.start)

    if valid_window is not None and mask.any():
        latest = int(samples.close_bar[mask].max())
        barrier = valid_window.start - embargo
        if latest >= barrier:
            raise AssertionError(
                f'purge failed for {train_window.name}/{valid_window.name}: a training '
                f'trade closes at bar {latest}, at or past the embargo barrier {barrier} '
                f'(valid starts {valid_window.start}, embargo {embargo}). Its label is '
                'not knowable at training time.'
            )
    return mask


def window_mask(samples: Samples, window) -> np.ndarray:
    """Rows whose *decision* falls inside ``window`` -- the out-of-sample set.

    Keyed on the decision bar rather than the fill bar so a sample is
    attributed to the fold whose model would actually have scored it.
    """
    return (samples.decision_bar >= window.start) & (samples.decision_bar < window.end)
