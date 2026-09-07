"""Run-history index: a small SQLite table, one row per backtest or optimize
run. Equity curves and trade logs are deliberately NOT stored here --
they go to ``results/web/{run_id}.json`` (see ``web.config.WEB_RESULTS_DIR``)
so the index stays a few KB per row and the history/compare list is a plain
table scan, not a JSON blob scan.

``metrics_json`` is stored via a plain ``json.dumps``/``json.loads`` round
trip, deliberately *not* run through ``web.serialize.jsonable``: a handful of
``compute_metrics`` fields (``calmar_ratio``, ``profit_factor``,
``profit_loss_ratio``) can legitimately be ``float('inf')``, and Python's
``json`` module round-trips a bare ``Infinity`` token fine between two
``json.dumps``/``json.loads`` calls even though it is not valid JSON to send
to a browser. Nulling it out here, before the API layer's
``annotate_inf_metrics`` ever sees it, would permanently erase which fields
were infinite. Everything that does reach the browser goes through
``jsonable``/``annotate_inf_metrics`` in the router layer instead.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Iterable, List, Optional

from web.config import DB_PATH, RUN_RETENTION, WEB_RESULTS_DIR
from web.serialize import jsonable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    strategy TEXT,
    symbols TEXT,
    start TEXT,
    end TEXT,
    cash REAL,
    slippage REAL,
    params_json TEXT,
    status TEXT NOT NULL,
    metrics_json TEXT,
    artifact_path TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind);
"""

# Serialises writes. Deliberately NOT reentrant, and deliberately not the
# lock used to build the connection: `create_run`/`finish_run` call
# `_get_conn` while already holding this one.
_lock = threading.Lock()
_conn_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(_SCHEMA)
    return conn


_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Build the shared connection once, under its own lock.

    Unguarded, two threads arriving together each opened one and the loser's
    was dropped on the floor still holding a WAL read lock. The lock has to be
    ``_conn_lock`` rather than ``_lock``: the write helpers call this while
    already inside ``with _lock``, so reusing that non-reentrant lock here
    deadlocks the first write of the process.
    """
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                _conn = _connect()
    return _conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d['symbols'] = json.loads(d['symbols']) if d.get('symbols') else []
    d['params'] = json.loads(d.pop('params_json')) if d.get('params_json') else {}
    d['metrics'] = json.loads(d.pop('metrics_json')) if d.get('metrics_json') else None
    return d


def _prune_locked(conn: sqlite3.Connection) -> None:
    """Drop the oldest runs past ``RUN_RETENTION``, artifact files and all.

    Caller must already hold ``_lock``. The newest ``RUN_RETENTION`` rows are
    kept whatever their status, and anything older is kept anyway while it is
    still ``running`` -- a study can outlive a few hundred quick backtests
    queued behind it, and deleting the row it is about to ``finish_run`` would
    make that update a silent no-op.

    Artifact files go first: a row deleted with its JSON left behind is an
    orphan nothing can ever find again, whereas a file deleted with its row
    left behind is what ``get_artifact`` already handles (it returns ``None``
    for a missing path).
    """
    stale = conn.execute(
        "SELECT id, artifact_path FROM runs "
        "WHERE status != 'running' AND id NOT IN ("
        '    SELECT id FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?'
        ')',
        (RUN_RETENTION,),
    ).fetchall()
    if not stale:
        return
    for row in stale:
        path = row['artifact_path']
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                # Not worth failing the run that triggered the prune over.
                pass
    conn.executemany('DELETE FROM runs WHERE id=?', [(row['id'],) for row in stale])


def create_run(
    *, kind: str, strategy: str = None, symbols: Iterable[str] = (),
    start: str = None, end: str = None, cash: float = None, slippage: float = None,
    params: dict = None,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        conn = _get_conn()
        conn.execute(
            'INSERT INTO runs (id, kind, created_at, strategy, symbols, start, end, '
            'cash, slippage, params_json, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (
                run_id, kind, time.time(), strategy, json.dumps(list(symbols)),
                start, end, cash, slippage, json.dumps(params or {}), 'running',
            ),
        )
        # On insert, so the history is bounded at the one point it can grow.
        _prune_locked(conn)
        conn.commit()
    return run_id


def finish_run(
    run_id: str, *, status: str, metrics: dict = None,
    artifact: dict = None, error: str = None,
) -> None:
    """``artifact`` (equity records, trade logs, per-symbol price+signal
    data -- whatever the run type produces) is written to
    ``results/web/{run_id}.json`` rather than into the row itself.

    Run through ``jsonable`` first: the engine's records carry
    ``datetime.date`` (every ``equity_records``/``trade_logs`` row) and numpy
    scalars, neither of which the stdlib ``json`` module can serialize on its
    own. Metrics are the one exception -- see the module docstring on why
    ``metrics_json`` below is *not* run through this.
    """
    artifact_path = None
    if artifact is not None:
        os.makedirs(WEB_RESULTS_DIR, exist_ok=True)
        artifact_path = os.path.join(WEB_RESULTS_DIR, f'{run_id}.json')
        with open(artifact_path, 'w', encoding='utf-8') as f:
            json.dump(jsonable(artifact), f)

    with _lock:
        conn = _get_conn()
        conn.execute(
            'UPDATE runs SET status=?, metrics_json=?, artifact_path=?, error=? WHERE id=?',
            (status, json.dumps(metrics) if metrics is not None else None, artifact_path, error, run_id),
        )
        conn.commit()


def get_run(run_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def get_artifact(run_id: str) -> Optional[dict]:
    row = get_run(run_id)
    if row is None or not row.get('artifact_path'):
        return None
    path = row['artifact_path']
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def list_runs(*, kind: str = None, limit: int = 100) -> List[dict]:
    conn = _get_conn()
    if kind:
        rows = conn.execute(
            'SELECT * FROM runs WHERE kind=? ORDER BY created_at DESC LIMIT ?', (kind, limit),
        ).fetchall()
    else:
        rows = conn.execute('SELECT * FROM runs ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_run(run_id: str) -> bool:
    row = get_run(run_id)
    if row is None:
        return False
    if row.get('artifact_path') and os.path.exists(row['artifact_path']):
        os.remove(row['artifact_path'])
    with _lock:
        conn = _get_conn()
        conn.execute('DELETE FROM runs WHERE id=?', (run_id,))
        conn.commit()
    return True
