"""Pydantic request models for the web API.

Responses are deliberately plain dicts (built by each router, run through
``web.serialize.jsonable``) rather than response models: the payloads mirror
whatever the engine already returns (``compute_metrics``, Optuna reports),
and re-declaring their shape here would be a second copy of
core/metrics.py's field list that drifts the moment a field is added there.
Requests get real models because they are hand-authored by this layer and
validation here is what turns a bad form submission into a clean 422 instead
of a traceback three calls deep in the engine.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ProductIn(BaseModel):
    exchange: str
    name: str
    name_zh: str
    start_year: int
    multiplier: float
    tick_size: float
    margin_rate: Optional[float] = None
    commission_rate: Optional[float] = None
    commission_per_lot: Optional[float] = None
    main_months: Optional[List[int]] = None
    roll_lead_months: Optional[int] = None

    def to_registry_entry(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class DataUpdateRequest(BaseModel):
    """The three update modes are mutually exclusive, the same way the CLI's
    ``--force`` / ``--rebuild-only`` group is: with both set, ``DataUpdate``
    skips the sync entirely and ``force`` is silently dropped, so a caller
    asking for a re-download gets a cache rebuild and no warning."""

    # ``None`` (the field omitted) means every registered product; an explicit
    # ``[]`` is a caller who picked nothing and is a 422 in the router. They
    # used to be the same thing, so submitting the panel's product picker with
    # nothing selected kicked off a download of the entire catalogue -- close
    # to an hour on a cold cache, from a click that asked for no products at
    # all.
    symbols: Optional[List[str]] = None
    force: bool = False
    rebuild_only: bool = False

    @model_validator(mode='after')
    def _one_mode_only(self):
        if self.force and self.rebuild_only:
            raise ValueError(
                'force and rebuild_only are mutually exclusive: pick '
                'force (re-download everything), rebuild_only (rebuild from '
                'the local cache), or neither (incremental update).'
            )
        return self


# Slippage is a cost, so it only ever moves a fill against you. A negative
# value moves it *for* you -- every entry and exit filling better than the
# market -- which does not fail, it just quietly inflates the result: on this
# repo's own SA/double_ma window, slippage=-50 takes Sharpe from 0.54 to 3.21
# and the run is recorded in history looking like any other. Rejected here
# rather than clamped, because a caller who asked for -50 asked for something
# that cannot be honoured, and silently substituting 0 would hand them numbers
# for a run they did not request.
_SLIPPAGE = Field(default=0.0, ge=0, description='Per-fill slippage; a cost, so it cannot be negative.')


class BacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start: str
    end: str
    cash: float = 100_000.0
    slippage: float = _SLIPPAGE
    params: Dict[str, object] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start: str
    end: str
    cash: float = 100_000.0
    slippage: float = _SLIPPAGE
    n_trials: int = 200
    n_folds: int = 4
    embargo: int = 10
    holdout_frac: float = 0.20
    lambda_std: float = 0.5
    min_trades_per_year: float = 4.0
    dd_cap: float = 0.35
    sparse_penalty: Optional[float] = None
    param_overrides: Dict[str, str] = Field(default_factory=dict)
    seed: int = 42
    probe_samples: int = 20
    study_name: Optional[str] = None
