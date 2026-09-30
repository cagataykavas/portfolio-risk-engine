from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.engine import PortfolioRiskEngine, PortfolioSpec
from src.synthetic import FACTOR_LOADINGS, synthetic_returns

app = FastAPI(title="Portfolio Risk Engine", version="1.0.0")
ENGINE = PortfolioRiskEngine()


class RiskRequest(BaseModel):
    assets: list[str] = Field(min_length=1)
    weights: list[float] = Field(min_length=1)
    returns: list[dict[str, float]] | None = None
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    monte_carlo_scenarios: int = Field(default=20_000, ge=1000, le=200_000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo() -> dict[str, object]:
    returns = synthetic_returns()
    assets = list(returns.columns)
    weights = [0.30, 0.20, 0.25, 0.15, 0.10]
    return ENGINE.analyze(
        returns,
        PortfolioSpec(tuple(assets), tuple(weights)),
        factor_loadings=FACTOR_LOADINGS,
    )


@app.post("/risk")
def risk(payload: RiskRequest) -> dict[str, object]:
    if len(payload.assets) != len(payload.weights):
        raise HTTPException(status_code=422, detail="assets and weights must have equal length")
    if payload.returns is None:
        frame = synthetic_returns()
        factor_loadings = FACTOR_LOADINGS
    else:
        frame = pd.DataFrame(payload.returns)
        factor_loadings = None
    try:
        return ENGINE.analyze(
            frame,
            PortfolioSpec(tuple(payload.assets), tuple(payload.weights)),
            confidence=payload.confidence,
            monte_carlo_scenarios=payload.monte_carlo_scenarios,
            factor_loadings=factor_loadings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
