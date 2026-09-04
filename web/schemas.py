"""Pydantic request models for the web API.

Responses are deliberately plain dicts (built by each router, run through
``web.serialize.jsonable``) rather than response models: the payloads mirror
whatever the engine already returns (``compute_metrics``, ``SignalReport``,
Optuna reports), and re-declaring their shape here would be a second copy of
core/metrics.py's field list that drifts the moment a field is added there.
Requests get real models because they are hand-authored by this layer and
validation here is what turns a bad form submission into a clean 422 instead
of a traceback three calls deep in the engine.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
    symbols: List[str]
    force: bool = False
    rebuild_only: bool = False


class BacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start: str
    end: str
    cash: float = 100_000.0
    slippage: float = 0.0
    params: Dict[str, object] = Field(default_factory=dict)
    meta_model: Optional[str] = None


class OptimizeRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start: str
    end: str
    cash: float = 100_000.0
    slippage: float = 0.0
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


class SignalRequest(BaseModel):
    strategy: Optional[str] = None
    model: Optional[str] = None
    symbols: Optional[List[str]] = None
    start: str = '2015-01-01'
    end: str = '2026-12-31'
    cash: Optional[float] = None
    slippage: Optional[float] = None
    params: Dict[str, object] = Field(default_factory=dict)
