"""FastAPI app for the FuturesToolkit web panel.

Run with ``python ft.py web`` (which owns the CLI flags and the
bind-address warning) or ``uvicorn web.app:app`` (prod-ish, still
single-process/single-machine).
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from web.config import STATIC_DIR, allowed_hosts, is_within
from web.routers import backtest, data, jobs, optimize, products, runs, strategies

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

app = FastAPI(title='FuturesToolkit Web Panel')

# Host header allowlist. CORS below stops another origin *reading* a response,
# but it does not stop the request being made, and this API has no
# authentication: a POST or DELETE lands whether or not the attacking page can
# see the answer. Binding to loopback does not help either -- a page the user
# has open can resolve a hostname it owns to 127.0.0.1 and reach this process
# from inside the browser. The one thing that request cannot forge is the Host
# header, so requiring it to name the panel itself is what actually closes it.
# See `web.config.allowed_hosts` for how to widen this deliberately.
_ALLOWED_HOSTS = allowed_hosts()
if '*' not in _ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_ALLOWED_HOSTS)
    logging.getLogger('futurestoolkit.web').info(
        'Host header restricted to: %s', ', '.join(_ALLOWED_HOSTS),
    )

# Dev-only: the Vite dev server runs on its own port (5173) and proxies /api
# to this process, but the browser still sees two origins until that proxy
# kicks in on first load. A production build is served from this same
# process (see the static mount below), where CORS is moot.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)

for router in (products, data, strategies, backtest, optimize, runs, jobs):
    app.include_router(router.router)


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.api_route(
    '/api/{rest:path}',
    methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    include_in_schema=False,
)
def api_not_found(rest: str):
    """Catch anything under /api that no router claimed.

    Declared after every router (so real routes still win) and before the SPA
    fallback below (so it wins over that). Without it a misspelled endpoint
    fell through to the SPA and came back as 200 text/html, which the
    frontend's fetch wrapper then failed to parse -- a JSON decode error
    somewhere in the UI instead of the 404 that actually happened.

    Kept out of the OpenAPI schema. FastAPI derives a route's operation id
    once per *route*, not per method, so these five methods shared one id and
    every schema build warned about the duplicate. It does not belong in the
    docs regardless: it is the absence of an endpoint, and listing it offered
    five phantom operations accepting any path under /api.
    """
    raise HTTPException(status_code=404, detail=f'No such endpoint: /api/{rest}')


if os.path.isdir(STATIC_DIR):
    app.mount('/assets', StaticFiles(directory=os.path.join(STATIC_DIR, 'assets')), name='assets')

    _STATIC_ROOT = os.path.realpath(STATIC_DIR)

    # index.html names the hashed bundles, so it must never be served stale:
    # without an explicit Cache-Control browsers fall back to heuristic caching
    # (a fraction of the file's age) and keep showing the previous build after a
    # rebuild until someone hard-refreshes. ``no-cache`` still caches the file,
    # it just forces a revalidation, which the ETag answers with a 304. The
    # hashed files under /assets stay freely cacheable -- their names change.
    _NO_CACHE = {'Cache-Control': 'no-cache'}

    # Out of the schema for the same reason as the /api catch-all above: this
    # serves the frontend, and as an OpenAPI operation it reads as a GET that
    # accepts every path on the server.
    @app.get('/{full_path:path}', include_in_schema=False)
    def spa(full_path: str):
        """Serve a built frontend file, falling back to index.html so the
        SPA's client-side routes survive a reload.

        ``full_path`` is whatever the client put in the request line, with
        no normalisation: the ASGI server percent-decodes it but does not
        collapse ``..``. A browser collapses those segments before the
        request leaves, which is why this reads as safe from the UI, but
        curl ``--path-as-is``, any non-browser client, and anything arriving
        through a proxy do not -- so resolve first and require the result to
        sit under the build directory, or this hands out every file the
        process can read. ``/assets`` above is Starlette's ``StaticFiles``,
        which already does its own containment check.
        """
        candidate = os.path.realpath(os.path.join(_STATIC_ROOT, full_path))
        if full_path and is_within(candidate, _STATIC_ROOT) and os.path.isfile(candidate):
            if os.path.basename(candidate) == 'index.html':
                return FileResponse(candidate, headers=_NO_CACHE)
            return FileResponse(candidate)
        return FileResponse(os.path.join(_STATIC_ROOT, 'index.html'), headers=_NO_CACHE)
