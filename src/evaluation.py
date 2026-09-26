"""Funciones de evaluación y visualización para los modelos de Churn.

Replica la lógica de evaluación usada en notebooks/ProyectoFinal_AAA.ipynb:
métricas por modelo, comparación con validación cruzada, matriz de confusión,
curva ROC, curva Precision-Recall y análisis de umbrales de decisión.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate

SCORING = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


def evaluar_modelo(nombre, modelo, X_eval, y_eval):
    """Calcula accuracy, precision, recall, f1 y ROC-AUC para un modelo entrenado.

    Devuelve (metricas: dict, y_pred, y_prob).
    """
    y_pred = modelo.predict(X_eval)
    y_prob = modelo.predict_proba(X_eval)[:, 1] if hasattr(modelo, "predict_proba") else None

    metricas = {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, pos_label=1),
        "F1-score": f1_score(y_eval, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_eval, y_prob) if y_prob is not None else None,
    }
    return metricas, y_pred, y_prob


def comparar_modelos(modelos, X, y, cv, scoring=None):
    """Compara varios modelos con validación cruzada, ordenados por Recall.

    modelos: dict {nombre: pipeline_sklearn}.
    Devuelve un DataFrame con medias y desviaciones estándar de cada métrica.
    """
    scoring = scoring or SCORING
    resultados = []

    for nombre, modelo in modelos.items():
        cv_res = cross_validate(modelo, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        fila = {"Modelo": nombre}
        for metrica in scoring:
            fila[metrica.capitalize()] = cv_res[f"test_{metrica}"].mean()
            fila[f"{metrica.capitalize()} Std"] = cv_res[f"test_{metrica}"].std()
        fila["Tiempo entrenamiento (s)"] = cv_res["fit_time"].mean()
        fila["Tiempo inferencia (s)"] = cv_res["score_time"].mean()
        resultados.append(fila)

    tabla = pd.DataFrame(resultados)
    if "Recall" in tabla.columns:
        tabla = tabla.sort_values(by="Recall", ascending=False).reset_index(drop=True)
    return tabla


def graficar_matriz_confusion(y_true, y_pred, titulo="Matriz de confusión",
                               etiquetas=("No churn", "Churn"), ax=None):
    """Dibuja la matriz de confusión para un conjunto de predicciones."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(etiquetas))
    disp.plot(ax=ax)
    plt.title(titulo)
    return cm


def graficar_curva_roc(modelos_prob, y_true, titulo="Curva ROC"):
    """Dibuja curvas ROC comparando varios modelos.

    modelos_prob: dict {nombre: y_prob} con probabilidades de la clase positiva.
    """
    plt.figure()
    for nombre, y_prob in modelos_prob.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{nombre} (AUC={auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Azar (AUC=0.5)")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos (Recall)")
    plt.title(titulo)
    plt.legend()


def graficar_curva_precision_recall(modelos_prob, y_true, titulo="Curva Precision-Recall"):
    """Dibuja curvas Precision-Recall comparando varios modelos."""
    plt.figure()
    for nombre, y_prob in modelos_prob.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=nombre)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(titulo)
    plt.legend()


def evaluar_umbrales(y_true, y_prob, umbrales=(0.30, 0.40, 0.50, 0.60, 0.70)):
    """Calcula precision, recall, f1 y errores para distintos umbrales de decisión.

    Útil para elegir un umbral operativo que priorice Recall (minimizar falsos negativos).
    """
    resultados = []
    for umbral in umbrales:
        y_pred_umbral = (y_prob >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_umbral).ravel()
        resultados.append({
            "Umbral": umbral,
            "Precision": precision_score(y_true, y_pred_umbral, zero_division=0),
            "Recall": recall_score(y_true, y_pred_umbral),
            "F1-score": f1_score(y_true, y_pred_umbral),
            "Falsos positivos": fp,
            "Falsos negativos": fn,
        })
    return pd.DataFrame(resultados)
