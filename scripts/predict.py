#!/usr/bin/env python3
"""CLI para predecir churn de clientes con el modelo entrenado.

Ejemplos de uso:
    python scripts/predict.py --input data/nuevos_clientes.csv --output results/predicciones.csv
    python scripts/predict.py --input data/nuevos_clientes.csv --threshold 0.5
    python scripts/predict.py --model models/modelo_churn_final_v1.joblib --input data/nuevos_clientes.csv

El CSV de entrada debe tener las mismas columnas que el dataset original
(sin la columna Churn). Si incluye customerID, se conserva solo como
referencia en la salida, no se usa para predecir.
"""

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, COLUMNA_ID

UMBRAL_OPERATIVO_DEFAULT = 0.30
COLUMNAS_ESPERADAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predice la probabilidad de churn para uno o varios clientes."
    )
    parser.add_argument(
        "--model", default="models/modelo_churn_final_v1.joblib",
        help="Ruta al pipeline entrenado (.joblib). Default: models/modelo_churn_final_v1.joblib",
    )
    parser.add_argument(
        "--input", required=True,
        help="Ruta al CSV con los clientes a predecir.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Ruta del CSV de salida. Si no se indica, se imprime en pantalla.",
    )
    parser.add_argument(
        "--threshold", type=float, default=UMBRAL_OPERATIVO_DEFAULT,
        help=f"Umbral de decisión para clasificar como abandono. Default: {UMBRAL_OPERATIVO_DEFAULT}",
    )
    return parser.parse_args()


def cargar_modelo(ruta_modelo):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{ruta_modelo}'. "
            "Entrena y guarda el modelo (joblib.dump) desde el notebook antes de usar este script."
        )
    return joblib.load(ruta_modelo)


def preparar_entrada(df):
    """Limpia tipos y valida que estén todas las columnas que el modelo espera."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV de entrada: {sorted(faltantes)}")

    return df


def predecir(modelo, df, threshold):
    X = df[COLUMNAS_ESPERADAS]
    probabilidades = modelo.predict_proba(X)[:, 1]
    predicciones = (probabilidades >= threshold).astype(int)

    resultado = pd.DataFrame({
        "Probabilidad_churn": probabilidades.round(4),
        "Prediccion": predicciones,
        "Clasificacion": ["Posible abandono" if p == 1 else "No clasificado como abandono" for p in predicciones],
    })

    if COLUMNA_ID in df.columns:
        resultado.insert(0, COLUMNA_ID, df[COLUMNA_ID].values)

    return resultado


def main():
    args = parse_args()

    print(f"Cargando modelo desde: {args.model}")
    modelo = cargar_modelo(args.model)

    print(f"Leyendo clientes desde: {args.input}")
    df = pd.read_csv(args.input)
    df = preparar_entrada(df)

    print(f"Generando predicciones con umbral = {args.threshold} ...")
    resultado = predecir(modelo, df, args.threshold)

    if args.output:
        resultado.to_csv(args.output, index=False)
        print(f"Predicciones guardadas en: {args.output}")
    else:
        print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()
