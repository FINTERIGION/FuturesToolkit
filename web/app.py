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
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers

from web.config import (
    DEV_ORIGINS, STATIC_DIR, UNSAFE_METHODS, allowed_hosts, allowed_origins, host_allowed,
    is_within, write_origin_allowed,
)
from web.routers import backtest, data, jobs, products, runs, strategies

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

app = FastAPI(title='FuturesToolkit Web Panel')


class HostHeaderMiddleware:
    """Refuse any request whose ``Host`` header does not name this panel.

    Stands in for Starlette's ``TrustedHostMiddleware``, which matches on
    ``header.split(':')[0]`` and so reduces every IPv6 literal to ``[`` --
    see ``web.config.normalize_host``. The allowlist is read once, at import,
    the same as before; ``web.config.host_allowed`` owns the matching.
    """

    def __init__(self, app, allowed: list):
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'):
            await self.app(scope, receive, send)
            return
        if host_allowed(Headers(scope=scope).get('host', ''), self.allowed):
            await self.app(scope, receive, send)
            return
        if scope['type'] == 'websocket':
            # No websocket routes today, but denying rather than falling
            # through keeps the check total: a route added later is covered
            # by this without anyone having to remember it.
            await send({'type': 'websocket.close', 'code': 1008})
            return
        await PlainTextResponse('Invalid host header', status_code=400)(scope, receive, send)


class WriteOriginMiddleware:
    """Refuse a state-changing request a browser sent from another site.

    The complement to ``HostHeaderMiddleware``. That one stops a page that
    rebinds a hostname it owns onto this machine; this one stops a page that
    simply addresses ``127.0.0.1`` directly, whose Host header is therefore the
    panel's own. Reads are left alone -- CORS already keeps another origin from
    seeing a response. ``web.config.write_origin_allowed`` owns the rule.
    """

    def __init__(self, app, allowed: list):
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in UNSAFE_METHODS:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if write_origin_allowed(
            headers.get('origin'), headers.get('sec-fetch-site'), headers.get('host', ''), self.allowed,
        ):
            await self.app(scope, receive, send)
            return
        await PlainTextResponse('Cross-origin write refused', status_code=403)(scope, receive, send)


# Host header allowlist. This API has no authentication, and binding to
# loopback keeps other machines out but not other *pages*: a site the user has
# open can resolve a hostname it owns to 127.0.0.1 and reach this process from
# inside the browser. That request carries the attacker's hostname in its Host
# header, so requiring the header to name the panel closes that route. It is
# not the only route -- see the Origin check below.
# See `web.config.allowed_hosts` for how to widen this deliberately.
_ALLOWED_HOSTS = allowed_hosts()
app.add_middleware(HostHeaderMiddleware, allowed=_ALLOWED_HOSTS)
if '*' in _ALLOWED_HOSTS:
    logging.getLogger('futurestoolkit.web').warning(
        'Host header check is OFF (%s contains "*"): this panel will answer to any '
        'hostname, including one an attacker points at it from a page the user has '
        'open. Name the hosts you serve it under instead.', 'FT_WEB_ALLOWED_HOSTS',
    )
else:
    logging.getLogger('futurestoolkit.web').info(
        'Host header restricted to: %s', ', '.join(_ALLOWED_HOSTS),
    )

# Origin check on writes. A page that addresses 127.0.0.1 directly sends the
# panel's own Host, so the check above lets it through, and CORS below does not
# stop a request being *made* -- only its response being read. A POST with no
# Content-Type skips the preflight altogether, and FastAPI still parses its body
# as JSON, so without this a page the user merely had open could start jobs and
# write the product registry. See `web.config.write_origin_allowed`.
app.add_middleware(WriteOriginMiddleware, allowed=allowed_origins())

# Dev-only: the Vite dev server runs on its own port (5173) and proxies /api
# to this process, but the browser still sees two origins until that proxy
# kicks in on first load. A production build is served from this same
# process (see the static mount below), where CORS is moot.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(DEV_ORIGINS),
    allow_methods=['*'],
    allow_headers=['*'],
)

for router in (products, data, strategies, backtest, runs, jobs):
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
