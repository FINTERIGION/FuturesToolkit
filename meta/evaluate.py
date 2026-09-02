"""Diagnostics for a meta-label filter, and the controls that stop it lying.

A meta-filter always removes trades, and removing trades changes Sharpe on its
own -- fewer, more concentrated bets, a different exposure profile, a
different drawdown path. So "filtered Sharpe > base Sharpe" is not evidence of
anything by itself, and the equity curve is the *last* thing to look at rather
than the first. Two controls separate the filter's skill from that artifact:

* **Classifier metrics before portfolio metrics.** Out-of-sample AUC near 0.5
  means there is no signal, and no threshold applied to a coin flip will
  produce one. Precision lift over the base win rate is the direct statement
  of "the kept trades really are better".
* **The shuffled-label control.** Refit the identical pipeline on permuted
  labels and re-run the backtest. Whatever Sharpe improvement survives that is
  the improvement caused by trading less, not by choosing better. If the real
  model does not clear its shuffled twin by a clear margin, the filter is
  noise dressed up as an edge.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'classifier_report', 'compare_metrics', 'summarize_folds', 'format_table', 'verdict',
]

_COMPARE_KEYS = [
    'sharpe_ratio', 'annualized_return', 'max_drawdown', 'win_rate',
    'profit_factor', 'expectancy', 'n_trades',
]


def _finite(x, default=float('nan')) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def classifier_report(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Out-of-sample classifier quality for one fold.

    ``precision_lift`` is the headline: the win rate among kept trades minus
    the win rate over all of them. Positive means the filter selected; zero
    means it merely subtracted.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype='float64')
    kept = p >= threshold
    base_rate = float(y_true.mean()) if len(y_true) else float('nan')
    kept_rate = float(y_true[kept].mean()) if kept.any() else float('nan')

    auc = float('nan')
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, p))

    return {
        'n': int(len(y_true)),
        'auc': auc,
        'base_rate': base_rate,
        'keep_rate': float(kept.mean()) if len(kept) else float('nan'),
        'n_kept': int(kept.sum()),
        'precision': kept_rate,
        'precision_lift': kept_rate - base_rate,
        'threshold': float(threshold),
    }


def compare_metrics(base: dict, filtered: dict) -> dict:
    """Side-by-side of the metrics that matter, plus deltas."""
    out: Dict[str, dict] = {}
    for key in _COMPARE_KEYS:
        b, f = _finite(base.get(key)), _finite(filtered.get(key))
        out[key] = {'base': b, 'filtered': f, 'delta': f - b}
    return out


def summarize_folds(fold_reports: List[dict]) -> dict:
    """Aggregate per-fold results into the numbers worth acting on.

    Averages are unweighted across folds on purpose: a fold is one
    independent trial of the whole procedure, and weighting by trade count
    would let the longest fold decide the verdict.
    """
    def _mean(path_a: str, path_b: Optional[str] = None) -> float:
        vals = []
        for r in fold_reports:
            v = r[path_a] if path_b is None else r[path_a].get(path_b)
            v = _finite(v.get('delta') if isinstance(v, dict) else v)
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float('nan')

    n_folds = len(fold_reports)
    sharpe_deltas = [
        _finite(r['metrics']['sharpe_ratio']['delta']) for r in fold_reports
    ]
    sharpe_deltas = [d for d in sharpe_deltas if np.isfinite(d)]

    return {
        'n_folds': n_folds,
        'mean_auc': _mean('classifier', 'auc'),
        'mean_precision_lift': _mean('classifier', 'precision_lift'),
        'mean_keep_rate': _mean('classifier', 'keep_rate'),
        'mean_sharpe_delta': float(np.mean(sharpe_deltas)) if sharpe_deltas else float('nan'),
        'folds_sharpe_improved': int(sum(d > 0 for d in sharpe_deltas)),
        'mean_trade_delta': _mean('metrics', 'n_trades'),
    }


def format_table(title: str, comparison: dict) -> str:
    """A base-vs-filtered block for the CLI."""
    lines = [f'  {title}', f"  {'metric':<20}{'base':>12}{'filtered':>12}{'delta':>12}"]
    for key, row in comparison.items():
        lines.append(
            f"  {key:<20}{row['base']:>12.4f}{row['filtered']:>12.4f}{row['delta']:>+12.4f}"
        )
    return '\n'.join(lines)


def verdict(summary: dict, shuffled: Optional[dict] = None) -> str:
    """A plain-language read of the summary, including the shuffled control.

    Written to be quotable in a report without further interpretation, and to
    say "no" clearly -- that is the outcome this whole protocol is built to be
    able to reach.
    """
    auc = summary.get('mean_auc', float('nan'))
    lift = summary.get('mean_precision_lift', float('nan'))
    d_sharpe = summary.get('mean_sharpe_delta', float('nan'))

    if not np.isfinite(auc) or auc < 0.52:
        return (
            f'NO EDGE: mean out-of-sample AUC {auc:.3f} is at chance. The filter is not '
            'selecting trades; any change in the equity curve is the effect of trading '
            'less. Do not tune keep_rate to fix this.'
        )
    if np.isfinite(lift) and lift <= 0:
        return (
            f'NO EDGE: AUC {auc:.3f} but precision lift {lift:+.3f} -- the kept trades are '
            'no better than the full set at the threshold actually used.'
        )
    if shuffled is not None:
        d_shuf = _finite(shuffled.get('mean_sharpe_delta'))
        if np.isfinite(d_shuf) and np.isfinite(d_sharpe) and d_sharpe <= d_shuf:
            return (
                f'NOT PROVEN: real Sharpe delta {d_sharpe:+.3f} does not beat the '
                f'shuffled-label control {d_shuf:+.3f}. The improvement is explained by '
                'trading less, not by choosing better.'
            )
    return (
        f'SIGNAL: AUC {auc:.3f}, precision lift {lift:+.3f}, mean Sharpe delta '
        f'{d_sharpe:+.3f} across {summary.get("folds_sharpe_improved")}/'
        f'{summary.get("n_folds")} improved folds. Confirm on the holdout before use.'
    )
