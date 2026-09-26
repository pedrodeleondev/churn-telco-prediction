#!/usr/bin/env python3
"""Entrena el modelo final y genera los archivos de models/ y results/.

Uso:
    python scripts/train.py
    python scripts/train.py --data data/TelcoCustomer.csv --threshold 0.30
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    RANDOM_STATE,
    cargar_datos,
    construir_preprocesador,
    dividir_datos,
    limpiar_datos,
    separar_X_y,
)
from src.evaluation import (
    comparar_modelos,
    evaluar_umbrales,
    graficar_curva_precision_recall,
    graficar_curva_roc,
    graficar_matriz_confusion,
)

UMBRAL_OPERATIVO_DEFAULT = 0.30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo final de churn y genera model.joblib + resultados."
    )
    parser.add_argument("--data", default="data/TelcoCustomer.csv", help="Ruta al CSV del dataset.")
    parser.add_argument("--models-dir", default="models", help="Carpeta donde guardar el modelo entrenado.")
    parser.add_argument("--results-dir", default="results", help="Carpeta donde guardar gráficos y métricas.")
    parser.add_argument("--threshold", type=float, default=UMBRAL_OPERATIVO_DEFAULT, help="Umbral operativo.")
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Omite la comparación de modelos por validación cruzada (más rápido).",
    )
    return parser.parse_args()


def construir_modelos_comparacion(preprocesador):
    return {
        "DummyClassifier": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ]),
        "Regresion Logistica": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Arbol de Decision": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "SVM Lineal": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE)),
        ]),
        "SVM RBF": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"[1/6] Cargando y limpiando datos desde {args.data} ...")
    df = cargar_datos(args.data)
    df = limpiar_datos(df)
    X, y = separar_X_y(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)
    print(f"      Train={X_train.shape[0]}  Val={X_val.shape[0]}  Test={X_test.shape[0]}")

    preprocesador = construir_preprocesador()

    if not args.skip_comparison:
        print("[2/6] Comparando modelos con validacion cruzada (StratifiedKFold=5) ...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        modelos = construir_modelos_comparacion(preprocesador)
        tabla_comparacion = comparar_modelos(modelos, X_train, y_train, cv)
        ruta_tabla = os.path.join(args.results_dir, "comparacion_modelos.csv")
        tabla_comparacion.round(4).to_csv(ruta_tabla, index=False)
        print(f"      Guardado: {ruta_tabla}")
        print(tabla_comparacion[["Modelo", "Recall", "Precision", "F1", "Roc_auc"]].round(4).to_string(index=False))
    else:
        print("[2/6] Comparacion de modelos omitida (--skip-comparison).")

    print("[3/6] Entrenando modelo final (Regresion Logistica) ...")
    modelo_final = Pipeline([
        ("preprocesador", preprocesador),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    modelo_final.fit(X_train, y_train)

    print("[4/6] Evaluando en conjunto de prueba ...")
    y_prob_test = modelo_final.predict_proba(X_test)[:, 1]
    tabla_umbrales = evaluar_umbrales(y_test, y_prob_test, umbrales=(0.30, 0.40, 0.50, 0.60, 0.70))
    ruta_umbrales = os.path.join(args.results_dir, "analisis_umbrales.csv")
    tabla_umbrales.round(4).to_csv(ruta_umbrales, index=False)
    print(f"      Guardado: {ruta_umbrales}")
    print(tabla_umbrales.round(4).to_string(index=False))

    y_pred_test = (y_prob_test >= args.threshold).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metricas_finales = {
        "umbral_operativo": args.threshold,
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1_score": f1_score(y_test, y_pred_test),
        "roc_auc": roc_auc_score(y_test, y_prob_test),
    }
    ruta_metricas = os.path.join(args.results_dir, "metricas_finales.json")
    with open(ruta_metricas, "w") as f:
        json.dump(metricas_finales, f, indent=2)
    print(f"      Guardado: {ruta_metricas}")
    print(json.dumps(metricas_finales, indent=2))

    print("[5/6] Generando graficos de resultados ...")
    graficar_matriz_confusion(y_test, y_pred_test, titulo=f"Matriz de confusion (umbral={args.threshold})")
    plt.savefig(os.path.join(args.results_dir, "matriz_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close()

    graficar_curva_roc({"Regresion Logistica": y_prob_test}, y_test, titulo="Curva ROC - Modelo final")
    plt.savefig(os.path.join(args.results_dir, "curva_roc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    graficar_curva_precision_recall({"Regresion Logistica": y_prob_test}, y_test, titulo="Curva Precision-Recall - Modelo final")
    plt.savefig(os.path.join(args.results_dir, "curva_precision_recall.png"), dpi=150, bbox_inches="tight")
    plt.close()

    preprocesador_ajustado = modelo_final.named_steps["preprocesador"]
    coeficientes = modelo_final.named_steps["model"].coef_[0]
    nombres_variables = preprocesador_ajustado.get_feature_names_out()
    importancias = pd.DataFrame({
        "Variable": nombres_variables,
        "Coeficiente": coeficientes,
        "Importancia": np.abs(coeficientes),
    }).sort_values("Importancia", ascending=False)
    ruta_importancias = os.path.join(args.results_dir, "importancia_variables.csv")
    importancias.to_csv(ruta_importancias, index=False)

    top15 = importancias.head(15).sort_values("Importancia")
    plt.figure(figsize=(10, 6))
    plt.barh(top15["Variable"], top15["Importancia"])
    plt.xlabel("Importancia (|coeficiente|)")
    plt.ylabel("Variable")
    plt.title("Variables mas importantes - Regresion Logistica")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "importancia_variables.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Guardados graficos y {ruta_importancias} en {args.results_dir}/")

    print("[6/6] Guardando el pipeline entrenado ...")
    ruta_modelo = os.path.join(args.models_dir, "modelo_churn_final_v1.joblib")
    joblib.dump(modelo_final, ruta_modelo)
    print(f"      Modelo guardado en: {ruta_modelo}")

    print("\nListo. Artefactos generados en 'models/' y 'results/'.")


if __name__ == "__main__":
    main()
