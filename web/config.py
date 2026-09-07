"""Paths and defaults for the web panel's backend."""

from __future__ import annotations

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
WEB_RESULTS_DIR = os.path.join(RESULTS_DIR, 'web')
OPTUNA_RESULTS_DIR = os.path.join(RESULTS_DIR, 'optuna')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
DB_PATH = os.path.join(RESULTS_DIR, 'webpanel.db')

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000

MARKET_CACHE_SIZE = 3
JOB_MAX_WORKERS = 2
JOB_LOG_BUFFER = 2000
JOB_RETENTION = 200
# Run-history rows kept in the SQLite index. Unlike JOB_RETENTION, which bounds
# an in-memory dict that empties on restart, this bounds something on disk: each
# run also writes results/web/{id}.json, and a backtest artifact carries every
# symbol's full OHLCV series -- megabytes apiece. Without a cap those files
# accumulate for the life of the install, and rows past `list_runs`' limit are
# not even reachable from the panel to delete by hand.
RUN_RETENTION = 200


def is_within(path: str, root: str) -> bool:
    """True if the already-resolved ``path`` sits inside ``root``.

    The containment rule for the one place the panel resolves a
    caller-supplied path into the filesystem: the SPA static handler in
    ``web.app``. ``path`` is expected to have been through
    ``os.path.realpath`` already, so ``..`` segments and symlinks are gone by
    the time it gets here.

    ``web.routers.optimize._report_path`` deliberately does not use this. It
    guards a flat directory of generated reports, where rejecting any name
    containing a path separator (and requiring a ``.json`` suffix) leaves
    nothing for ``..`` to traverse *through* -- a narrower rule than
    resolve-then-contain, and one that does not touch the filesystem to
    decide.
    """
    root = os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)
