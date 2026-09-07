"""Search-space resolution for any ``Strategy`` subclass.

Priority, highest first: an explicit CLI ``--param`` override > the
strategy's declared ``space`` class attribute > a heuristic inferred from
its ``params`` defaults. Nothing here knows about any concrete strategy --
``resolve_space`` works off ``Strategy.params`` / ``Strategy.space`` /
``Strategy.fixed_params`` alone, so it applies unchanged to any strategy
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

    Shared by the ``optimize`` report (``research.optimize.run_study``) and
    the web panel's parameter editor, so both describe a strategy's tunable
    surface identically -- one had this logic first and the other would
    otherwise have to duplicate it.
    """
    if isinstance(spec, Int):
        return {'kind': 'int', 'low': spec.low, 'high': spec.high, 'step': spec.step, 'log': spec.log}
    if isinstance(spec, Float):
        return {'kind': 'float', 'low': spec.low, 'high': spec.high, 'step': spec.step, 'log': spec.log}
    if isinstance(spec, Categorical):
        return {'kind': 'categorical', 'choices': list(spec.choices)}
    raise TypeError(f'Unknown space spec: {spec!r}')


def _infer_spec(value) -> Optional[Spec]:
    """Heuristic space for a param that declared no ``space`` entry.

    Deliberately crude (``value / 4 .. value * 4``-ish) -- it exists so a
    strategy with no tuning declared at all is still optimizable out of the
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
            f"{strategy_cls.__name__} has no tunable parameters: `params` is "
            f"empty, or every key is in `fixed_params` with no `space`/override."
        )
    return space


def suggest(trial, space: dict) -> dict:
    """Sample one parameter set from ``space`` via an Optuna ``trial``."""
    params = {}
    for name, spec in space.items():
        if isinstance(spec, Int):
            params[name] = trial.suggest_int(name, spec.low, spec.high, step=spec.step, log=spec.log)
        elif isinstance(spec, Float):
            kwargs = {'log': spec.log}
            if spec.step is not None:
                kwargs['step'] = spec.step
            params[name] = trial.suggest_float(name, spec.low, spec.high, **kwargs)
        elif isinstance(spec, Categorical):
            params[name] = trial.suggest_categorical(name, list(spec.choices))
        else:
            raise TypeError(f"Unknown space spec for {name!r}: {spec!r}")
    return params


def check_constraints(strategy_cls: type, params: dict) -> bool:
    """True iff every ``strategy_cls.constraints`` predicate accepts ``params``."""
    return all(fn(params) for fn in (getattr(strategy_cls, 'constraints', ()) or ()))


def parse_param_override(spec: str) -> tuple:
    """Parse one ``--param name=kind:args`` CLI token into ``(name, Spec)``.

    Examples::

        slow_period=int:20:200
        slow_period=int:20:200:5          # step 5
        risk_pct=float:0.005:0.05:log
        mode=cat:trend|reversion
    """
    if '=' not in spec:
        raise ValueError(f"Invalid --param {spec!r}; expected name=kind:args")
    name, rest = spec.split('=', 1)
    kind, *args = rest.split(':')
    kind = kind.lower()

    if kind == 'int':
        low, high = int(args[0]), int(args[1])
        step, log = 1, False
        for extra in args[2:]:
            if extra == 'log':
                log = True
            else:
                step = int(extra)
        return name, Int(low, high, step=step, log=log)

    if kind == 'float':
        low, high = float(args[0]), float(args[1])
        step, log = None, False
        for extra in args[2:]:
            if extra == 'log':
                log = True
            else:
                step = float(extra)
        return name, Float(low, high, step=step, log=log)

    if kind in ('cat', 'categorical'):
        return name, Categorical(tuple(args[0].split('|')))

    raise ValueError(f"Unknown space kind {kind!r} in --param {spec!r}")


# ---------------------------------------------------------------------
# Concrete parameter *values* (as opposed to the search ranges above)
# ---------------------------------------------------------------------

def parse_param_value(raw: str) -> tuple:
    """Parse one ``name=value`` CLI token into ``(name, value)``, casting the
    value to int, then float, then bool, else leaving it a string.

    Distinct from :func:`parse_param_override` above, which reads
    ``name=kind:args`` and declares a search *range*. ``ft.py backtest``
    takes this form; ``ft.py optimize`` takes that one.
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
    precedence first: an optimize report's ``best_params``, then ``--lots``,
    then ``--param``. Only what the user actually asked to change is returned
    -- ``Strategy.__init__`` merges the class's own ``params`` defaults under
    it -- so an untouched run behaves exactly as before.
    """
    params: dict = {}
    if getattr(args, 'params_from', None):
        with open(args.params_from, encoding='utf-8') as f:
            best = json.load(f)['best_params']
        params.update(best)
        logger.info('Params from %s: %s', args.params_from, best)
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
