"""Background job manager for anything too slow to answer within one HTTP
request: a backtest over a long window is a minute or two, and a multi-symbol
data download is close to an hour on a cold cache.

Jobs run on a small thread pool -- pandas/numpy release the GIL for the
heavy lifting, so this genuinely parallelizes rather than just queuing.
Anything the CLI already logs via the ``logging`` module (roll-calendar
notices, warmup warnings, blow-up notices, incomplete-download warnings)
reaches the browser unchanged, because the manager attaches a handler to
the loggers those modules already use and threads it into each job's own
log by thread id.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional

from web.config import JOB_LOG_BUFFER, JOB_MAX_WORKERS, JOB_RETENTION

logger = logging.getLogger('futurestoolkit.web')

_MONITORED_LOGGERS = (
    'futurestoolkit', 'datafeed', 'research', 'core',
)


@dataclasses.dataclass
class LogLine:
    ts: float
    level: str
    message: str
    seq: int = 0
    """Position in the job's whole log, not in the buffer currently holding it.

    ``Job.logs`` is a ring buffer, so a line's index within it shifts every
    time an older one is evicted. An open stream needs a mark that survives
    that, or it cannot tell "already sent" from "scrolled out of the buffer".
    """


@dataclasses.dataclass
class Job:
    id: str
    kind: str
    status: str = 'queued'          # queued -> running -> done|error|cancelled
    progress: float = 0.0
    message: str = ''
    logs: Deque[LogLine] = dataclasses.field(default_factory=lambda: deque(maxlen=JOB_LOG_BUFFER))
    result: Any = None
    error: Optional[str] = None
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel_requested: bool = False
    _log_seq: "itertools.count" = dataclasses.field(default_factory=itertools.count)
    _cancel_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    # (event loop, queue) per open SSE stream. The loop is kept alongside the
    # queue because the thread that wakes a waiter is never the thread the
    # queue belongs to -- see `_notify`.
    _waiters: List[tuple] = dataclasses.field(default_factory=list)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def to_dict(self) -> dict:
        return {
            'id': self.id, 'kind': self.kind, 'status': self.status,
            'progress': self.progress, 'message': self.message,
            'error': self.error, 'created_at': self.created_at,
            'started_at': self.started_at, 'finished_at': self.finished_at,
            'cancel_requested': self.cancel_requested,
        }


class _ThreadLogHandler(logging.Handler):
    """Routes a log record to whichever job owns the emitting thread."""

    def __init__(self, manager: "JobManager"):
        super().__init__()
        self._manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        job = self._manager._job_for_thread(record.thread)
        if job is None:
            return
        try:
            line = LogLine(
                ts=record.created, level=record.levelname,
                message=self.format(record), seq=next(job._log_seq),
            )
            # Under the job lock: `stream` iterates this same deque, and a
            # bare append from a worker thread mid-iteration raises
            # "deque mutated during iteration" there -- which surfaces as the
            # SSE connection dying, not as a dropped line.
            with job._lock:
                job.logs.append(line)
            self._manager._notify(job)
        except Exception:
            pass


class JobManager:
    """Owns the thread pool, the job registry, and the log-fan-out handler.

    ``cancel`` is cooperative: it sets a per-job ``threading.Event`` and a
    ``cancel_requested`` flag the callable can check between units of work
    (a per-symbol loop in a data update). A callable that never checks either
    just runs to completion -- cancel is a request, not a kill switch, so the
    API surfaces it as best-effort.
    """

    def __init__(self, max_workers: int = JOB_MAX_WORKERS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ftk-job')
        self._jobs: Dict[str, Job] = {}
        self._thread_job: Dict[int, str] = {}
        self._lock = threading.Lock()
        # Held while a job starts, and by `hold_starts` while a caller mutates
        # state a running job would read. Separate from `_lock`, which only
        # guards the registry dict itself and is taken for far shorter spans.
        self._start_lock = threading.RLock()
        self._handler = _ThreadLogHandler(self)
        self._handler.setFormatter(logging.Formatter('%(message)s'))
        for name in _MONITORED_LOGGERS:
            logging.getLogger(name).addHandler(self._handler)

    def _job_for_thread(self, thread_id: int) -> Optional[Job]:
        job_id = self._thread_job.get(thread_id)
        return self._jobs.get(job_id) if job_id else None

    def _notify(self, job: Job) -> None:
        """Wake every open stream for ``job``, from whatever thread we are on.

        Always a cross-thread hand-off in practice: jobs run on the pool and
        cancels arrive on FastAPI's request threadpool, while the queue being
        poked belongs to the event loop serving the SSE response.
        ``asyncio.Queue`` is not thread-safe, and a bare ``put_nowait`` from
        off-loop does not wake the selector -- the stream then sat until its
        own 15s timeout, so "live" progress arrived up to 15 seconds late and
        a cancel took just as long to show. ``call_soon_threadsafe`` is the
        supported way in, and works from the loop thread too.
        """
        with job._lock:
            waiters = list(job._waiters)
        for loop, q in waiters:
            try:
                loop.call_soon_threadsafe(q.put_nowait, None)
            except RuntimeError:
                # Loop already closed -- its stream is gone and will be
                # dropped from _waiters by its own `finally`.
                pass

    def notify(self, job_id: str) -> None:
        """Wake any open SSE stream for ``job_id`` immediately, rather than
        waiting for the stream's own timeout. For a caller that mutates
        ``job.progress``/``job.message`` directly from inside its own
        callable -- the log handler triggers this automatically for anything
        that goes through ``logging``, but a bare field update has no log line
        to piggyback on.
        """
        job = self._jobs.get(job_id)
        if job is not None:
            self._notify(job)

    def _prune(self) -> None:
        if len(self._jobs) <= JOB_RETENTION:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.finished_at is not None),
            key=lambda j: j.finished_at,
        )
        for j in finished[: len(self._jobs) - JOB_RETENTION]:
            self._jobs.pop(j.id, None)

    def submit(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        """Run ``fn(job)`` on the pool. ``fn`` should periodically update
        ``job.progress``/``job.message`` and, for long loops, check
        ``job.cancel_requested`` and raise or return early when set."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._start_lock, self._lock:
            self._jobs[job.id] = job
            self._prune()

        def _run():
            self._thread_job[threading.get_ident()] = job.id
            job.status = 'running'
            job.started_at = time.time()
            self._notify(job)
            try:
                job.result = fn(job)
                job.status = 'cancelled' if job.cancel_requested else 'done'
            except Exception as exc:
                logger.exception('Job %s (%s) failed', job.id, kind)
                job.status = 'error'
                job.error = str(exc)
            finally:
                job.finished_at = time.time()
                self._thread_job.pop(threading.get_ident(), None)
                self._notify(job)

        self._executor.submit(_run)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def any_active(self) -> bool:
        """True while any job is queued or running.

        Used to refuse a product-registry write mid-run: the engine reads
        ``product_costs``/``roll_rule`` live per fill
        (``core/broker.py``, ``core/ledger.py``), so a multiplier or margin
        edit landing mid-backtest would silently corrupt that run's numbers.
        """
        return any(j.status in ('queued', 'running') for j in self._jobs.values())

    @contextlib.contextmanager
    def hold_starts(self):
        """Block new job starts for the duration of the block.

        ``any_active()`` alone is a check, not a guarantee: a job could be
        submitted in the gap between reading it and acting on the answer, so
        a registry write that had just confirmed "nothing is running" could
        still land on a run that started a microsecond later. Doing the check
        and the write in here makes the pair atomic against new starts.

        Deliberately does not wait for running jobs to finish -- callers pair
        this with ``any_active()`` and refuse, rather than blocking a request
        behind an hour-long data update.
        """
        with self._start_lock:
            yield

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status not in ('queued', 'running'):
            return False
        job.cancel_requested = True
        job._cancel_event.set()
        self._notify(job)
        return True

    async def stream(self, job_id: str):
        """Async generator of SSE-formatted events for ``job_id``: one
        ``data:`` line per state change (a log line, a progress update, or
        the terminal status), ending after the job reaches a terminal state.
        """
        job = self._jobs.get(job_id)
        if job is None:
            yield 'event: error\ndata: {"error": "unknown job"}\n\n'
            return

        queue: "asyncio.Queue" = asyncio.Queue()
        waiter = (asyncio.get_running_loop(), queue)
        with job._lock:
            job._waiters.append(waiter)
        # Track the last line *sent*, by sequence number rather than by count.
        # Counting worked only until the job emitted more than JOB_LOG_BUFFER
        # lines: past that the deque starts evicting, its length stops
        # growing, and a running total of what had been sent runs off the end
        # of it -- so the slice went permanently empty and the stream fell
        # silent for the rest of the job, exactly on the long data-update
        # runs this streaming exists for.
        last_seq = -1
        try:
            import json as _json
            while True:
                with job._lock:
                    pending_logs = [line for line in job.logs if line.seq > last_seq]
                if pending_logs:
                    # A gap means the buffer overwrote lines before this
                    # stream drained them. Say so: an unannounced hole reads
                    # as a job that simply went quiet, which is the failure
                    # this whole change is about.
                    dropped = pending_logs[0].seq - last_seq - 1
                    if dropped > 0:
                        payload = _json.dumps({
                            'type': 'log', 'level': 'WARNING',
                            'message': f'[{dropped} earlier log line(s) dropped; '
                                       f'this job\'s buffer keeps the last {JOB_LOG_BUFFER}]',
                        })
                        yield f'data: {payload}\n\n'
                    last_seq = pending_logs[-1].seq
                for line in pending_logs:
                    payload = _json.dumps({'type': 'log', 'level': line.level, 'message': line.message})
                    yield f'data: {payload}\n\n'

                payload = _json.dumps({'type': 'state', **job.to_dict()})
                yield f'data: {payload}\n\n'

                if job.status in ('done', 'error', 'cancelled'):
                    return
                try:
                    await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    continue
        finally:
            with job._lock:
                if waiter in job._waiters:
                    job._waiters.remove(waiter)


manager = JobManager()
