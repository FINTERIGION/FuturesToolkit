"""FastAPI app for the FuturesToolkit web panel.

Run with ``python -m web.app`` (dev) or ``uvicorn web.app:app`` (prod-ish,
still single-process/single-machine -- see the binding note below).
"""

from __future__ import annotations

import argparse
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.config import DEFAULT_HOST, DEFAULT_PORT, STATIC_DIR, is_within
from web.routers import backtest, data, jobs, optimize, products, runs, signals, strategies

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

app = FastAPI(title='FuturesToolkit Web Panel')

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

for router in (products, data, strategies, backtest, optimize, signals, runs, jobs):
    app.include_router(router.router)


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.api_route('/api/{rest:path}', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def api_not_found(rest: str):
    """Catch anything under /api that no router claimed.

    Declared after every router (so real routes still win) and before the SPA
    fallback below (so it wins over that). Without it a misspelled endpoint
    fell through to the SPA and came back as 200 text/html, which the
    frontend's fetch wrapper then failed to parse -- a JSON decode error
    somewhere in the UI instead of the 404 that actually happened.
    """
    raise HTTPException(status_code=404, detail=f'No such endpoint: /api/{rest}')


if os.path.isdir(STATIC_DIR):
    app.mount('/assets', StaticFiles(directory=os.path.join(STATIC_DIR, 'assets')), name='assets')

    _STATIC_ROOT = os.path.realpath(STATIC_DIR)

    @app.get('/{full_path:path}')
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
            return FileResponse(candidate)
        return FileResponse(os.path.join(_STATIC_ROOT, 'index.html'))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description='FuturesToolkit web panel.')
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    if args.host not in ('127.0.0.1', 'localhost'):
        logging.getLogger('futurestoolkit.web').warning(
            'Binding to %s: this panel has no authentication and can rewrite the '
            'product registry and delete data files. Only do this on a network '
            'you trust.', args.host,
        )

    uvicorn.run('web.app:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
