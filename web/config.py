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

# Host header allowlist, enforced by TrustedHostMiddleware in `web.app`.
#
# Binding to 127.0.0.1 keeps other machines out, but it does not keep a *web
# page* out: a site the user is browsing can point a hostname it controls at
# 127.0.0.1 (DNS rebinding) and drive this API from their own browser, which
# is an unauthenticated surface that rewrites the product registry and deletes
# data files. The browser sends the attacker's hostname in the Host header, so
# refusing anything but the names the panel is actually served under closes it.
#
# `ft.py web` derives FT_WEB_ALLOWED_HOSTS from the bind address when asked to
# bind somewhere other than loopback, because the Host header is then whatever
# name the operator reaches the box by and no default here could guess it.
ALLOWED_HOSTS_ENV = 'FT_WEB_ALLOWED_HOSTS'
DEFAULT_ALLOWED_HOSTS = ('localhost', '127.0.0.1', '[::1]')


def allowed_hosts() -> list:
    """Host names this panel will answer to, newest environment wins.

    Read at call time rather than import time so a test (or `ft.py web`
    setting the variable before uvicorn imports the app) can change it.
    Ports are not included: ``host_allowed`` strips the port before matching.
    """
    raw = os.environ.get(ALLOWED_HOSTS_ENV, '').strip()
    if not raw:
        return list(DEFAULT_ALLOWED_HOSTS)
    return [h.strip() for h in raw.split(',') if h.strip()]


def normalize_host(raw: str) -> str:
    """The host name out of a ``Host`` header, port removed, lowercased.

    Split on the first colon and an IPv6 literal loses everything after its
    first group: ``[::1]:8000`` becomes ``[``. That is what Starlette's
    ``TrustedHostMiddleware`` does, and it is why the ``[::1]`` entry in
    ``DEFAULT_ALLOWED_HOSTS`` above never matched anything -- browsing the
    panel at ``http://[::1]:8000`` answered 400 to a name the config
    explicitly allows. The bracketed form is kept intact here, so the entry
    means what it reads as.

    Returns ``''`` for a header with no closing bracket, which no client
    sends and which therefore matches nothing.
    """
    host = (raw or '').strip()
    if host.startswith('['):
        end = host.find(']')
        return host[: end + 1].lower() if end != -1 else ''
    return host.split(':', 1)[0].lower()


def host_allowed(raw_host: str, allowed=None) -> bool:
    """Whether a request's ``Host`` header names this panel.

    Exact match, with two carry-overs from the middleware this replaces: a
    lone ``*`` allows anything, and a leading ``*.`` matches subdomains but
    not the bare domain. Both sides are lowercased; host names are
    case-insensitive and an allowlist typed by hand should not have to be.
    """
    patterns = [p.strip().lower() for p in (allowed if allowed is not None else allowed_hosts())]
    if '*' in patterns:
        return True
    host = normalize_host(raw_host)
    if not host:
        return False
    for pattern in patterns:
        if pattern == host:
            return True
        if pattern.startswith('*.') and host.endswith(pattern[1:]):
            return True
    return False


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
