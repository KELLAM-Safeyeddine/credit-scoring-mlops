"""Thin FastAPI routes around accepted Registry snapshots."""

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import create_service
from api.schemas import Application
from src.serving.service import ScoringService


def create_app(service: ScoringService | None = None, reload_token: str | None = None) -> FastAPI:
    """Start without substituting a rejected model when the champion is absent."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or create_service()
        app.state.service.reload()
        yield

    app = FastAPI(title="Credit scoring", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(request, error: RequestValidationError):
        # Do not echo private inputs or nonfinite values into JSON error responses.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: item[key] for key in ("type", "loc", "msg")} for item in error.errors()
                ]
            },
        )

    def current():
        snapshot = app.state.service.snapshot
        if snapshot is None:
            raise HTTPException(503, "Aucun champion accepté disponible.")
        return snapshot

    @app.get("/live")
    def live():
        """Process liveness; readiness remains conditional on an accepted champion."""
        return {"status": "alive"}

    @app.get("/health")
    def health():
        snapshot = current()
        return {
            "status": "ready",
            "model_version": snapshot.version,
            "reload_status": app.state.service.error or "ok",
        }

    @app.get("/model-info")
    def model_info():
        snapshot = current()
        return {
            "version": snapshot.version,
            "run_id": snapshot.run_id,
            "created_at": snapshot.created_at,
            "threshold": snapshot.threshold,
            "metrics": snapshot.evidence["metrics"],
            "data_hash": snapshot.evidence.get("training_data_sha256"),
            "evaluation_data_hash": snapshot.evidence.get("evaluation_data_sha256"),
            "raw_dvc_hash": snapshot.evidence.get("raw_dvc_hash"),
            "dvc_lock_sha256": snapshot.evidence.get("dvc_lock_sha256"),
            "feature_count": len(snapshot.columns),
            "score_interpretation": "Score du modèle, probabilité non calibrée.",
        }

    def infer(payload: Application, explain: bool):
        current()
        inputs = payload.model_dump(exclude={"customer_id", "history"})
        if payload.history:
            inputs.update(payload.history.model_dump())
        try:
            return app.state.service.score(payload.customer_id, inputs, explain)
        except LookupError as error:
            raise HTTPException(503, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(503, "Inférence ou journalisation indisponible.") from error

    @app.post("/predict")
    def predict(payload: Application):
        return infer(payload, False)

    @app.post("/explain")
    def explain(payload: Application):
        return infer(payload, True)

    @app.post("/model/reload")
    def reload_model(x_reload_token: str = Header(default="")):
        expected = reload_token or os.getenv("API_RELOAD_TOKEN", "")
        if not expected or not hmac.compare_digest(expected.encode(), x_reload_token.encode()):
            raise HTTPException(403, "Jeton de rechargement requis.")
        if not app.state.service.reload():
            raise HTTPException(503, "Rechargement rejeté ; modèle précédent conservé.")
        return {"model_version": current().version}

    return app


app = create_app()
