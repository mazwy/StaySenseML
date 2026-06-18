"""FastAPI prediction service. Loads the production model from the MLflow
registry at startup and exposes /predict, /predict/batch, /healthz, /info."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from hotels.api import inference
from hotels.api.schemas import (
    BatchRequest,
    BatchResponse,
    Booking,
    HealthResponse,
    ModelInfo,
    Prediction,
)
from hotels.experiments import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME

logger = logging.getLogger("hotels.api")
logging.basicConfig(level=logging.INFO)

state: dict[str, object | None] = {"model": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("loading production model from MLflow registry...")
        state["model"] = inference.load_production_model()
        loaded = state["model"]
        logger.info(
            "loaded %s v%s (kind=%s, auc=%s)",
            REGISTERED_MODEL_NAME,
            loaded.version,
            loaded.model_kind,
            loaded.auc,
        )
    except Exception as exc:
        state["error"] = repr(exc)
        logger.exception("failed to load model: %s", exc)
    yield


app = FastAPI(
    title="hotels-cancellation",
    version="0.1.0",
    description="Predicts the probability that a hotel booking will be cancelled.",
    lifespan=lifespan,
)


def _require_model() -> inference.LoadedModel:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {state.get('error')}")
    return model  # type: ignore[return-value]


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    model = state["model"]
    if model is None:
        return HealthResponse(status="error" if state.get("error") else "loading", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=model.version,
        model_kind=model.model_kind,
    )


@app.get("/info", response_model=ModelInfo)
def info() -> ModelInfo:
    model = _require_model()
    return ModelInfo(
        registered_name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=model.version,
        model_kind=model.model_kind,
        auc=model.auc,
        tracking_uri=model.tracking_uri,
    )


@app.post("/predict", response_model=Prediction)
def predict_one(booking: Booking) -> Prediction:
    model = _require_model()
    rows = inference.predict(model, [booking.model_dump()], inference.DEFAULT_THRESHOLD)
    return Prediction(**rows[0])


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    model = _require_model()
    if not req.bookings:
        return BatchResponse(predictions=[])
    rows = inference.predict(
        model,
        [b.model_dump() for b in req.bookings],
        inference.DEFAULT_THRESHOLD,
    )
    return BatchResponse(predictions=[Prediction(**r) for r in rows])
