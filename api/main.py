"""
main.py — FastAPI application entry point.

Exposes:
  GET  /          → health check
  POST /predict   → surgery duration prediction
"""

from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import predict, load_artifacts

# ── App instance ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Surgery Duration Predictor",
    description="Predicts OR surgery duration from procedure details.",
    version="RF",
)


# ── Startup: load model into memory before accepting requests ──────────────────
@app.on_event("startup")
def startup_event():
    load_artifacts()


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Surgery Duration Predictor API is running"}


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_duration(request: PredictionRequest):
    try:
        predicted_duration = predict(
            surgical_priority=request.surgical_priority,
            patient_type=request.patient_type,
            procedure_description=request.procedure_description,
            room=request.room,
            specialty=request.specialty,
        )
        return PredictionResponse(predicted_duration_minutes=predicted_duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
