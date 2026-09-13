"""Job status, cancellation, and the SSE log/progress stream."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from web.jobs import manager as job_manager
from web.serialize import jsonable

router = APIRouter(prefix='/api/jobs', tags=['jobs'])


@router.get('')
def list_jobs():
    return [j.to_dict() for j in job_manager.list()]


@router.get('/{job_id}')
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Unknown job {job_id!r}')
    d = job.to_dict()
    d['result'] = jsonable(job.result)
    return d


@router.get('/{job_id}/stream')
async def stream_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Unknown job {job_id!r}')
    return StreamingResponse(
        job_manager.stream(job_id),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.post('/{job_id}/cancel')
def cancel_job(job_id: str):
    ok = job_manager.cancel(job_id)
    if not ok:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f'Unknown job {job_id!r}')
        raise HTTPException(status_code=409, detail=f'Job {job_id!r} is already {job.status}')
    return {'cancelled': job_id}
