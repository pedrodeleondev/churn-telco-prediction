#!/usr/bin/env python3
"""API REST para predecir churn con el modelo entrenado.

Ejecutar localmente:
    uvicorn scripts.api:app --reload --port 8000

Luego probar con:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @ejemplo_cliente.json

Documentación interactiva (Swagger UI) disponible en http://127.0.0.1:8000/docs
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/modelo_churn_final_v1.joblib")
UMBRAL_OPERATIVO = float(os.environ.get("UMBRAL_OPERATIVO", "0.30"))
COLUMNAS_ESPERADAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS

_modelo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        cargar_modelo()
    except FileNotFoundError:
        pass  # /health reportará el problema con detalle
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predice la probabilidad de abandono (churn) de clientes de Telco.",
    version="1.0",
    lifespan=lifespan,
)


class Cliente(BaseModel):
    customerID: Optional[str] = Field(default=None, description="Identificador opcional, solo de referencia.")
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PrediccionChurn(BaseModel):
    customerID: Optional[str] = None
    probabilidad_churn: float
    prediccion: int
    clasificacion: str
    umbral_utilizado: float


def cargar_modelo():
    global _modelo
    if _modelo is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró el modelo en '{MODEL_PATH}'. "
                "Genera el modelo con 'python scripts/train.py' antes de levantar la API."
            )
        _modelo = joblib.load(MODEL_PATH)
    return _modelo


@app.get("/health")
def health():
    """Verifica que la API esté viva y el modelo cargado."""
    try:
        cargar_modelo()
        return {"status": "ok", "modelo": MODEL_PATH, "umbral_operativo": UMBRAL_OPERATIVO}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _predecir_uno(modelo, cliente: Cliente) -> PrediccionChurn:
    datos = cliente.model_dump(exclude={"customerID"})
    df = pd.DataFrame([datos])[COLUMNAS_ESPERADAS]

    probabilidad = float(modelo.predict_proba(df)[0, 1])
    prediccion = int(probabilidad >= UMBRAL_OPERATIVO)
    clasificacion = "Posible abandono" if prediccion == 1 else "No clasificado como abandono"

    return PrediccionChurn(
        customerID=cliente.customerID,
        probabilidad_churn=round(probabilidad, 4),
        prediccion=prediccion,
        clasificacion=clasificacion,
        umbral_utilizado=UMBRAL_OPERATIVO,
    )


@app.post("/predict", response_model=PrediccionChurn)
def predict(cliente: Cliente):
    """Predice la probabilidad de churn para un cliente."""
    modelo = cargar_modelo()
    return _predecir_uno(modelo, cliente)


@app.post("/predict/batch", response_model=List[PrediccionChurn])
def predict_batch(clientes: List[Cliente]):
    """Predice la probabilidad de churn para una lista de clientes."""
    modelo = cargar_modelo()
    return [_predecir_uno(modelo, cliente) for cliente in clientes]
