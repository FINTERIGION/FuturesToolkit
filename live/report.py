"""Rendering and persistence for a :class:`~live.signal.SignalReport`.

Split from ``live.signal`` so the computation can be imported (and tested,
and called from a notebook) without dragging along a fixed opinion about how
it should look on a terminal.

The table grows three columns when a meta-model is in play and is otherwise
identical, rather than being two separate renderers: a plain strategy's row
has nothing to say about ``proba``, and printing a column of dashes for it
would be noise on the path most runs take.
"""

from __future__ import annotations

import json
import logging
import os

from live.signal import SignalReport, SignalRow

logger = logging.getLogger(__name__)

__all__ = ['render', 'save', 'emit']

SEP = '=' * 84


def _signed(n: int) -> str:
    return f'{n:+d}' if n else '0'


def _num(x, digits=4) -> str:
    return '-' if x is None else f'{x:.{digits}f}'


def _side(row: SignalRow) -> str:
    return {1: 'long', -1: 'short'}.get(row.primary_side, '-')


def _note(row: SignalRow) -> str:
    if row.verdict == 'vetoed':
        return f'VETOED (primary wanted {row.vetoed_target:+d})'
    if not row.tradable:
        return 'no session on the last bar'
    if row.verdict:
        return row.verdict
    if row.stop_rule and row.stop is None:
        return 'stop arms after the fill'
    return 'hold' if row.current_simulated else '-'


#  (header, width, align, getter, meta_only)
_COLUMNS = (
    ('sym',      6, '<', lambda r: r.symbol,                    False),
    ('contract', 10, '<', lambda r: r.contract or '-',          False),
    ('sim pos',   9, '>', lambda r: _signed(r.current_simulated), False),
    ('target',    8, '>', lambda r: _signed(r.target),          False),
    ('action',   10, '>', lambda r: r.action,                   False),
    ('side',      7, '>', _side,                                True),
    ('proba',     9, '>', lambda r: _num(r.proba),              True),
    ('thresh',    9, '>', lambda r: _num(r.threshold),          True),
    ('stop',     11, '>', lambda r: _num(r.stop, 2),            False),
    ('note',      0, '<', _note,                                False),
)


def _columns(meta: bool):
    return [c for c in _COLUMNS if not c[4] or meta]


def _line(values, cols) -> str:
    parts = []
    for value, (_h, width, align, _g, _m) in zip(values, cols):
        parts.append(f'{value:{align}{width}}' if width else f'  {value}')
    return '  ' + ''.join(parts)


def render(report: SignalReport) -> str:
    """The human-readable block. Pure: it writes nothing and returns a string."""
    spec = report.spec
    cols = _columns(report.is_meta)

    kind = 'meta-filtered signal' if report.is_meta else 'signal'
    lines = [
        SEP,
        f'  FuturesToolkit {kind}   as of {report.as_of}',
        f'  Strategy: {spec.strategy_name}   Universe: {" ".join(spec.symbols)}',
    ]
    if spec.params:
        lines.append(f'  Params (overrides): {spec.params}')
    if report.is_meta:
        model = spec.model
        lines.append(
            f'  Model: {os.path.basename(spec.model_path or "?")} '
            f'(keep_rate {model.keep_rate:.2f}, threshold {model.threshold_value:.4f}, '
            f'trained through {model.trained_through})'
        )
    lines += [
        '  Execute at the NEXT session open.',
        SEP,
        _line([c[0] for c in cols], cols),
    ]
    for row in report.rows:
        lines.append(_line([getter(row) for _h, _w, _a, getter, _m in cols], cols))

    lines += [
        SEP,
        '  "sim pos" is simulated from the start of the window and WILL drift from',
        '  your real account. Reconcile against your actual book: the actionable',
        '  column is "target", not "action". "contract" is the main contract as of',
        '  the last bar -- re-check it if a roll falls on the next session.',
    ]
    if not report.actionable:
        lines.append('  No position changes for the next session.')
    if not spec.effective_params.get('lots'):
        lines += [
            '  This strategy sizes off equity (lots=0), so the lot counts above were',
            f'  sized on the SIMULATED equity of {report.simulated_equity:,.0f} '
            f'(from cash {spec.cash:,.0f}).',
            '  Pass --cash equal to your real account equity to size on that instead.',
        ]
    if report.deferred:
        lines.append(f'  WARNING: orders still deferred and never filled: {report.deferred}')
    lines.append(SEP)
    return '\n'.join(lines)


def save(report: SignalReport, results_dir: str) -> str:
    """Write the machine-readable copy. Named per strategy *and* date so two
    strategies signalled on the same day do not overwrite each other."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(
        results_dir, f'signal_{report.spec.strategy_name}_{report.as_of}.json',
    )
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    return path


def emit(report: SignalReport, results_dir: str) -> str:
    """Log the table and write the JSON -- what every CLI path wants."""
    logger.info('%s', render(report))
    path = save(report, results_dir)
    logger.info('  Signal JSON: %s', path)
    return path
