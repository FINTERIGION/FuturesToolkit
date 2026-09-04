"""Paths and defaults for the web panel's backend."""

from __future__ import annotations

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
WEB_RESULTS_DIR = os.path.join(RESULTS_DIR, 'web')
OPTUNA_RESULTS_DIR = os.path.join(RESULTS_DIR, 'optuna')
LIVE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'live')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
DB_PATH = os.path.join(RESULTS_DIR, 'webpanel.db')

MODELS_DIR = os.path.join(ROOT_DIR, 'models')

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000

MARKET_CACHE_SIZE = 3
JOB_MAX_WORKERS = 2
JOB_LOG_BUFFER = 2000
JOB_RETENTION = 200


def is_within(path: str, root: str) -> bool:
    """True if the already-resolved ``path`` sits inside ``root``.

    One definition of the containment rule shared by the panel's two places
    that resolve a caller-supplied path: the SPA static handler in
    ``web.app`` and ``resolve_model_path`` below. ``path`` is expected to
    have been through ``os.path.realpath`` already, so ``..`` segments and
    symlinks are gone by the time it gets here.
    """
    root = os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def resolve_model_path(raw: str) -> str:
    """Resolve a caller-supplied meta-model path, confined to ``models/``.

    ``joblib.load`` unpickles, and unpickling runs whatever the file says to
    run before any validation gets a chance -- ``meta.model.load_model`` says
    so in its own docstring. That is an acceptable contract for the CLI's
    ``--model``, where the path comes from the person at the keyboard, and
    not for a request body, where it comes from anything that can reach the
    port. Confining it here is the same defence
    ``web.routers.optimize._report_path`` already applies to Optuna reports
    and ``web.routers.signals.signal_detail`` to saved signal files.

    Accepts the three spellings already in use -- ``models/x.joblib`` (the
    Signals page placeholder and the ``meta_runner.py`` docs), a bare
    ``x.joblib``, or an absolute path -- and raises ``ValueError`` for
    anything resolving outside ``models/``, symlinks included. Existence is
    deliberately left to ``load_model``, so a typo stays a 404 rather than
    being reported as an access violation.
    """
    text = str(raw).strip()
    if not text:
        raise ValueError('Model path must not be empty')
    for base in (ROOT_DIR, MODELS_DIR):
        candidate = os.path.realpath(os.path.join(base, text))
        if is_within(candidate, MODELS_DIR):
            return candidate
    raise ValueError(
        f'Model path {raw!r} resolves outside '
        f'{os.path.relpath(MODELS_DIR, ROOT_DIR)}/; the panel loads models only from there.'
    )
