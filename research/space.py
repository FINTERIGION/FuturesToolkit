"""Parameter-range resolution for any ``Strategy`` subclass.

A strategy's ``space`` declares the range each parameter is *plausible*
over. Nothing searches it: ``ft.py validate`` steps one notch either side of
the value in use to see whether the result depends on the exact number, and
the web panel's parameter editor uses the same bounds to flag a value typed
outside them.

Priority, highest first: an explicit override > the strategy's declared
``space`` class attribute > a heuristic inferred from its ``params``
defaults. Nothing here knows about any concrete strategy -- ``resolve_space``
works off ``Strategy.params`` / ``Strategy.space`` / ``Strategy.fixed_params``
alone, so it applies unchanged to any strategy
``strategies.discover_strategies()`` finds, present or future.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from core.params import Categorical, Float, Int, Spec

logger = logging.getLogger(__name__)


def spec_to_json(spec: Spec) -> dict:
    """Render one search-space spec as a JSON-safe dict.

    Shared by the ``validate`` report (``research.validate``) and the web
    panel's parameter editor, so both describe a strategy's parameter surface
    identically -- one had this logic first and the other would otherwise
    have to duplicate it.
    """
    if isinstance(spec, Int):
        return {'kind': 'int', 'low': spec.low, 'high': spec.high, 'step': spec.step, 'log': spec.log}
    if isinstance(spec, Float):
        return {'kind': 'float', 'low': spec.low, 'high': spec.high, 'step': spec.step, 'log': spec.log}
    if isinstance(spec, Categorical):
        return {'kind': 'categorical', 'choices': list(spec.choices)}
    raise TypeError(f'Unknown space spec: {spec!r}')


def _infer_spec(value) -> Optional[Spec]:
    """Heuristic range for a param that declared no ``space`` entry.

    Deliberately crude (``value / 4 .. value * 4``-ish) -- it exists so a
    strategy that declared nothing still gets a sensitivity scan out of the
    box, not to be a good default for every parameter. ``show-space`` prints
    exactly what gets inferred so a user can promote it into the strategy's
    own ``space`` once they've looked at it.
    """
    if isinstance(value, bool):
        return Categorical((True, False))
    if isinstance(value, int):
        if value <= 0:
            return None
        return Int(max(1, value // 4), max(value * 4, value + 2))
    if isinstance(value, float):
        if value == 0:
            return None
        magnitude = abs(value)
        return Float(magnitude / 4, magnitude * 4, log=True)
    return None


def resolve_space(strategy_cls: type, overrides: Optional[dict] = None) -> dict:
    """Return ``{param_name: Spec}`` for ``strategy_cls``."""
    fixed = set(getattr(strategy_cls, 'fixed_params', ()) or ())
    declared = dict(getattr(strategy_cls, 'space', {}) or {})
    defaults = dict(getattr(strategy_cls, 'params', {}) or {})

    space: dict = {}
    inferred = {}
    for name, default in defaults.items():
        if name in declared:
            space[name] = declared[name]
        elif name in fixed:
            continue
        else:
            spec = _infer_spec(default)
            if spec is not None:
                space[name] = spec
                inferred[name] = spec

    if inferred:
        logger.info(
            "%s: no `space` declared for %s -- inferred %s from defaults; "
            "run `ft.py show-space --strategy ...` to inspect "
            "and consider promoting these into the strategy's own `space`.",
            strategy_cls.__name__, ', '.join(sorted(inferred)), inferred,
        )

    if overrides:
        space.update(overrides)

    if not space:
        raise ValueError(
            f"{strategy_cls.__name__} has no parameter ranges: `params` is "
            f"empty, or every key is in `fixed_params` with no `space`/override."
        )
    return space


def check_constraints(strategy_cls: type, params: dict) -> bool:
    """True iff every ``strategy_cls.constraints`` predicate accepts ``params``."""
    return all(fn(params) for fn in (getattr(strategy_cls, 'constraints', ()) or ()))


# ---------------------------------------------------------------------
# Concrete parameter *values* (as opposed to the ranges above)
# ---------------------------------------------------------------------

def parse_param_value(raw: str) -> tuple:
    """Parse one ``--param name=value`` CLI token into ``(name, value)``,
    casting the value to int, then float, then bool, else leaving it a string.
    """
    if '=' not in raw:
        raise ValueError(f"Invalid --param {raw!r}; expected name=value")
    name, value = raw.split('=', 1)
    for caster in (int, float):
        try:
            return name, caster(value)
        except ValueError:
            continue
    if value.lower() in ('true', 'false'):
        return name, value.lower() == 'true'
    return name, value


def resolve_params(strategy_cls: type, args) -> dict:
    """Build the ``strategy_cls(**overrides)`` dict from the CLI, lowest
    precedence first: a ``validate`` report's ``params``, then ``--lots``,
    then ``--param``. Only what the user actually asked to change is returned
    -- ``Strategy.__init__`` merges the class's own ``params`` defaults under
    it -- so an untouched run behaves exactly as before.
    """
    params: dict = {}
    if getattr(args, 'params_from', None):
        with open(args.params_from, encoding='utf-8') as f:
            report = json.load(f)
        if 'params' not in report:
            raise ValueError(
                f"{args.params_from!r} has no 'params' key -- expected a "
                f"`ft.py validate` report."
            )
        loaded = report['params']
        params.update(loaded)
        logger.info('Params from %s: %s', args.params_from, loaded)
    if getattr(args, 'lots', None) is not None:
        params['lots'] = args.lots
    for raw in getattr(args, 'param', None) or []:
        name, value = parse_param_value(raw)
        params[name] = value

    known = set(getattr(strategy_cls, 'params', {}) or {})
    unknown = sorted(set(params) - known)
    if unknown:
        # Not fatal: `Strategy.__init__` merges anything into `self.p`, and
        # `--lots` is documented as harmless for strategies that ignore it.
        # But a typo would otherwise vanish without a trace, so say so.
        logger.warning(
            '%s declares no param(s) %s -- passing them through, but the strategy '
            'will not read them (declared params: %s).',
            strategy_cls.__name__, unknown, sorted(known) or '<none>',
        )
    return params
