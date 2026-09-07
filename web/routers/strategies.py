"""Strategy catalog: params, tunable search space, and a hot-reload hook so
a strategy edited in the editor (the private, gitignored modules under
``strategies/``) appears without restarting the server.
"""

from __future__ import annotations

import importlib
import inspect

from fastapi import APIRouter, HTTPException

from research.space import resolve_space, spec_to_json
from strategies import discover_strategies, load_registered_strategy

from web.jobs import manager as job_manager
from web.serialize import jsonable

router = APIRouter(prefix='/api/strategies', tags=['strategies'])


def _describe(key: str, cls: type) -> dict:
    try:
        space = {k: spec_to_json(v) for k, v in resolve_space(cls).items()}
        space_error = None
    except ValueError as e:
        space, space_error = {}, str(e)

    return {
        'key': key,
        'class_name': cls.__name__,
        'module': cls.__module__,
        'file': inspect.getfile(cls),
        'docstring': inspect.getdoc(cls) or '',
        'params': dict(getattr(cls, 'params', {}) or {}),
        'fixed_params': list(getattr(cls, 'fixed_params', ()) or ()),
        'space': space,
        'space_error': space_error,
    }


@router.get('')
def list_strategies():
    return [_describe(key, cls) for key, cls in sorted(discover_strategies().items())]


@router.get('/{key}')
def get_strategy(key: str):
    try:
        cls = load_registered_strategy(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _describe(key, cls)


@router.post('/reload')
def reload_strategies():
    if job_manager.any_active():
        raise HTTPException(
            status_code=409,
            detail='A job is running; reloading strategy modules mid-run could '
                   'swap the class object out from under it.',
        )
    import pkgutil

    package = importlib.import_module('strategies')
    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(package.__path__, prefix='strategies.'):
        if mod_name == 'strategies.base':
            # ``discover_strategies``'s ``issubclass(obj, Strategy)`` checks
            # against the ``Strategy`` name bound in ``strategies/__init__.py``,
            # which is not itself reloaded here. Reloading ``base`` would swap
            # in a *new* ``Strategy`` class object that every already-defined
            # subclass (reloaded or not) no longer descends from, silently
            # emptying the registry rather than picking anything up. Editing
            # the shared framework module, unlike a strategy, needs a restart.
            continue
        module = importlib.import_module(mod_name)
        importlib.reload(module)

    found = discover_strategies()
    return jsonable({'reloaded': True, 'strategies': sorted(found)})
