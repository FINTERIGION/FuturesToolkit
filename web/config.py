"""Paths and defaults for the web panel's backend."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
WEB_RESULTS_DIR = os.path.join(RESULTS_DIR, 'web')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
DB_PATH = os.path.join(RESULTS_DIR, 'webpanel.db')

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000

# Host header allowlist, enforced by HostHeaderMiddleware in `web.app`.
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


# Origins the Vite dev server serves the frontend from. It proxies /api to this
# process, so in development a browser write arrives carrying one of these
# rather than the panel's own origin. Shared with the CORS setup in `web.app`.
DEV_ORIGINS = ('http://localhost:5173', 'http://127.0.0.1:5173')
ALLOWED_ORIGINS_ENV = 'FT_WEB_ALLOWED_ORIGINS'
UNSAFE_METHODS = frozenset({'POST', 'PUT', 'PATCH', 'DELETE'})


def _normalize_origin(origin: str) -> str:
    return origin.strip().lower().rstrip('/')


def allowed_origins() -> list:
    """Origins allowed to write besides the panel's own: the dev server's,
    plus anything listed in ``FT_WEB_ALLOWED_ORIGINS``.

    The variable is for a reverse proxy that rewrites the Host header, where
    the origin the browser reports and the Host this process sees stop
    matching. Read at call time, like ``allowed_hosts``.
    """
    raw = os.environ.get(ALLOWED_ORIGINS_ENV, '')
    return list(DEV_ORIGINS) + [_normalize_origin(o) for o in raw.split(',') if o.strip()]


def write_origin_allowed(origin, sec_fetch_site, raw_host: str, allowed=None) -> bool:
    """Whether a state-changing request came from the panel's own pages.

    The Host check stops a rebinding page, whose requests carry the attacker's
    hostname. It does not stop an ordinary cross-site request: a page on any
    site can POST straight to ``http://127.0.0.1:8000``, and that request's
    Host header is the panel's own. CORS does not stop it either. A POST whose
    body is a type-less Blob goes out with no Content-Type, which makes it a
    "simple" request sent without a preflight -- and FastAPI parses a body
    with no Content-Type as JSON. A page the user merely had open could start
    a forced re-download of every product or write one into the registry.

    What such a request cannot hide is where it came from. Browsers send
    ``Origin`` on every cross-origin POST/PUT/PATCH/DELETE, so a request that
    carries one is accepted only when it names this panel: the same
    ``host[:port]`` the request addressed, or an entry in ``allowed``
    (default ``allowed_origins()``). ``Origin: null`` -- a sandboxed frame, a
    ``no-referrer`` form -- matches neither.

    No ``Origin`` at all means no browser cross-origin fetch: curl, the test
    client, a script, all already running as someone who can reach the port.
    ``Sec-Fetch-Site`` is the second opinion for the rare browser request that
    leaves ``Origin`` out.
    """
    if origin is None:
        return (sec_fetch_site or '').strip().lower() in ('', 'same-origin', 'none')
    origin = _normalize_origin(origin)
    patterns = {_normalize_origin(o) for o in (allowed if allowed is not None else allowed_origins())}
    if origin in patterns:
        return True
    parts = urlsplit(origin)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        return False
    return parts.netloc == (raw_host or '').strip().lower()


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

    It is the only such rule the panel needs today: every other route names
    its file from an id the panel itself generated, never from caller text.
    """
    root = os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)
