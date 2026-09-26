"""Módulos reutilizables del proyecto de predicción de Churn en Telco."""

from .preprocessing import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    RANDOM_STATE,
    cargar_datos,
    construir_preprocesador,
    dividir_datos,
    limpiar_datos,
    preparar_dataset,
    separar_X_y,
)
from .evaluation import (
    SCORING,
    comparar_modelos,
    evaluar_modelo,
    evaluar_umbrales,
    graficar_curva_precision_recall,
    graficar_curva_roc,
    graficar_matriz_confusion,
)

__all__ = [
    "COLUMNAS_CATEGORICAS",
    "COLUMNAS_NUMERICAS",
    "RANDOM_STATE",
    "cargar_datos",
    "construir_preprocesador",
    "dividir_datos",
    "limpiar_datos",
    "preparar_dataset",
    "separar_X_y",
    "SCORING",
    "comparar_modelos",
    "evaluar_modelo",
    "evaluar_umbrales",
    "graficar_curva_precision_recall",
    "graficar_curva_roc",
    "graficar_matriz_confusion",
]
