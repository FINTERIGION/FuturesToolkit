"""``BarContext`` wrappers that make meta-labeling strategy-agnostic.

Both wrappers work by composition, not inheritance: they hold a reference
to a real ``BarContext`` and delegate everything they don't explicitly
override via ``__getattr__``. That means they apply to *any*
``Strategy`` subclass's ``on_bar`` unchanged -- neither wrapper needs to
know what the wrapped strategy actually does.

- ``make_recording_bar_context`` -- read side. Lets a strategy trade
  normally while recording, per bar and symbol, the start-of-bar position
  and the final target it ended the bar wanting. Used by
  ``research.metalabel.extract_events`` to find entry/reversal events
  without touching engine internals.
- ``MetaFilteredStrategy`` / ``_GatingContext`` -- write side. Wraps an
  already-instantiated inner ``Strategy`` and gates its entries (flat ->
  open, or a reversal) on a fitted meta-label probability. Exits are never
  gated, and ``on_missing`` decides what happens at bars the model has no
  probability for.
"""

from __future__ import annotations

import math

from strategies.base import BarContext, Strategy


def make_recording_bar_context(log: list) -> type:
    """Return a ``BarContext``-compatible class that appends one dict per
    ``(bar, symbol)`` actually touched to ``log``: ``{'bar', 'symbol',
    'start', 'target'}``. ``start`` is the true broker position at the top
    of the bar (before any calls this bar); ``target`` is the position the
    strategy ends the bar wanting, correctly reflecting ``set_target``
    overwrite semantics and ``buy``/``sell`` accumulation within the same
    bar (matching ``core.engine.BarContext``'s own semantics exactly).
    """

    class _RecordingBarContext:
        def __init__(self, engine, i, date):
            self._ctx = BarContext(engine, i, date)
            self._touched: dict = {}

        def __getattr__(self, name):
            return getattr(self._ctx, name)

        def _pending(self, sym):
            rec = self._touched.get((self._ctx.i, sym))
            return rec['target'] if rec is not None else self._ctx.position(sym)

        def _emit(self, sym, target):
            key = (self._ctx.i, sym)
            rec = self._touched.get(key)
            if rec is None:
                rec = {'bar': self._ctx.i, 'symbol': sym, 'start': self._ctx.position(sym), 'target': target}
                self._touched[key] = rec
                log.append(rec)  # same dict object; later in-place edits show up in `log` too
            else:
                rec['target'] = target
            self._ctx.set_target(sym, target)

        def set_target(self, sym, lots):
            self._emit(sym, int(lots))

        def close(self, sym):
            self._emit(sym, 0)

        def buy(self, sym, lots=1):
            self._emit(sym, self._pending(sym) + int(lots))

        def sell(self, sym, lots=1):
            self._emit(sym, self._pending(sym) - int(lots))

    return _RecordingBarContext


class _GatingContext:
    """Wraps a real ``BarContext``. Exits and same-direction resizes pass
    through untouched; a flat->open or a reversal is allowed only if
    ``gate.proba[sym][ctx.i] >= gate.p['meta_threshold']``. A blocked
    reversal is turned into a flat close, never a reverse-and-flip -- the
    inner strategy gets another chance to re-enter on a later bar once its
    own logic re-evaluates from flat.
    """

    def __init__(self, ctx: BarContext, gate: "MetaFilteredStrategy"):
        self._ctx = ctx
        self._gate = gate

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def _passes(self, sym: str) -> bool:
        proba = self._gate.proba.get(sym)
        if proba is None:
            return True  # no model for this symbol at all -- don't gate it
        # ``ctx.i`` is slice-local (the engine only ever sees the window slice
        # `run_window` built); ``proba`` is indexed by absolute bar, the
        # numbering its caller works in. ``bar_offset`` is the slice origin
        # that bridges them -- see MetaFilteredStrategy's docstring.
        bar = self._ctx.i + self._gate.bar_offset
        if 0 <= bar < len(proba):
            p = proba[bar]
            if math.isfinite(p):
                if p >= self._gate.p['meta_threshold']:
                    return True
                self._gate.threshold_rejected_count += 1
                return False
            self._gate.nan_count += 1
        else:
            self._gate.out_of_range_count += 1

        # The model has no usable opinion about this bar. Which way that
        # should fall is the caller's call, not this class's -- see
        # ``on_missing`` in the docstring.
        return self._gate.on_missing == 'pass'

    def set_target(self, sym: str, lots: int) -> None:
        lots = int(lots)
        current = self._ctx.position(sym)

        is_exit = lots == 0
        is_same_direction = current != 0 and lots != 0 and (lots > 0) == (current > 0)
        if is_exit or is_same_direction:
            self._ctx.set_target(sym, lots)
            return

        # Flat -> open, or a reversal: gate it.
        if self._passes(sym):
            self._ctx.set_target(sym, lots)
        else:
            self._gate.blocked_count += 1
            self._ctx.set_target(sym, 0)

    def close(self, sym: str) -> None:
        self._ctx.set_target(sym, 0)  # exits always allowed, bypass gating entirely

    def buy(self, sym: str, lots: int = 1) -> None:
        self.set_target(sym, self._ctx.position(sym) + int(lots))

    def sell(self, sym: str, lots: int = 1) -> None:
        self.set_target(sym, self._ctx.position(sym) - int(lots))


class MetaFilteredStrategy(Strategy):
    """Wraps an already-instantiated ``inner`` strategy and filters its
    entries through a fitted meta-label probability. Applies to any
    ``Strategy`` subclass with zero changes to it.

    ``proba``: ``{symbol: np.ndarray[n_bars]}`` of ``P(win)`` indexed by
    *absolute* bar in the full market, ``nan`` where a prediction isn't
    available (e.g. before the model's own feature warmup) -- a ``nan`` is
    always treated as "does not pass", as is a bar outside the array.

    ``bar_offset``: the absolute bar the engine's bar 0 corresponds to, i.e.
    ``research.runner_api.slice_start(window, pad)`` for the run this gate is
    attached to. It defaults to 0 (the engine is running the whole market),
    but a windowed run whose pad does not reach back to bar 0 has a nonzero
    origin, and without this every ``proba`` lookup would silently land
    ``bar_offset`` bars too early. Always pass the same ``window``/``pad``
    that ``run_window`` is being called with.

    ``on_missing``: what to do at a bar with no usable probability (``nan``,
    or outside the array). The two callers want opposite things and neither
    is more correct in general, so it is a parameter rather than a policy
    baked in here:

    - ``'block'`` (default) is right for **deployment**, where ``proba`` comes
      from the bundle's own pipeline and a ``nan`` means the model genuinely
      cannot score this bar yet (feature warmup). Trading un-vetted there
      would defeat the point of having fitted a filter.
    - ``'pass'`` is right for **evaluation**
      (``research.metalabel.evaluate_meta_backtest``), where ``proba`` is
      sparse for a reason that has nothing to do with the market: only events
      inside a walk-forward valid window get an out-of-fold prediction at all.
      Blocking the rest would charge the filter for the harness's coverage
      gaps and make gated-vs-baseline measure "how many bars had an OOF
      prediction" instead of "what did the filter decide".

    Counters, all per-run: ``threshold_rejected_count`` (entries stopped by a
    real ``p < threshold`` decision), ``nan_count`` / ``out_of_range_count``
    (bars with no usable probability, whichever way ``on_missing`` sent them),
    and ``blocked_count`` (total entries flattened; minus
    ``threshold_rejected_count`` gives how many were lost to missing data).

    Note on ``buy``/``sell``: unlike ``set_target``, repeated ``buy``/
    ``sell`` calls for the *same* symbol within a single bar are resolved
    against the broker's position at the top of the bar, not against any
    earlier call within that same bar -- correct for the common case of at
    most one order per symbol per bar (true of every bundled strategy);
    a strategy that issues multiple accumulating ``buy``/``sell`` calls for
    one symbol in one bar should use ``set_target`` instead.
    """

    params = {'meta_threshold': 0.5}

    def __init__(self, inner: Strategy, proba: dict, bar_offset: int = 0,
                 on_missing: str = 'block', **overrides):
        if on_missing not in ('block', 'pass'):
            raise ValueError(f"on_missing must be 'block' or 'pass', got {on_missing!r}")
        super().__init__(**overrides)
        self.inner = inner
        self.proba = proba
        self.bar_offset = int(bar_offset)
        self.on_missing = on_missing
        self.threshold_rejected_count = 0
        self.nan_count = 0
        self.out_of_range_count = 0
        self.blocked_count = 0

    def setup(self, ctx) -> None:
        self.inner.setup(ctx)

    def on_bar(self, ctx) -> None:
        self.inner.on_bar(_GatingContext(ctx, self))

    def on_finish(self, engine) -> None:
        fn = getattr(self.inner, 'on_finish', None)
        if callable(fn):
            fn(engine)
