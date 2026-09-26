"""Funciones de preprocesamiento sacadas del notebook (limpieza, split y ColumnTransformer)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

COLUMNA_ID = "customerID"
COLUMNA_OBJETIVO = "Churn"

COLUMNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

COLUMNAS_CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def cargar_datos(ruta_csv):
    """Lee el dataset Telco Customer Churn desde un archivo CSV."""
    return pd.read_csv(ruta_csv)


def limpiar_datos(df):
    """TotalCharges a numérico (tenía 11 registros con espacios) y Churn Yes/No a 1/0."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if COLUMNA_OBJETIVO in df.columns:
        valores_unicos = set(df[COLUMNA_OBJETIVO].dropna().unique())
        if valores_unicos <= {"Yes", "No"}:
            df[COLUMNA_OBJETIVO] = df[COLUMNA_OBJETIVO].map({"No": 0, "Yes": 1})

    return df


def separar_X_y(df):
    """Separa en X (predictoras) y y (Churn); descarta customerID."""
    X = df.drop(columns=[COLUMNA_ID, COLUMNA_OBJETIVO])
    y = df[COLUMNA_OBJETIVO]
    return X, y


def construir_preprocesador():
    """ColumnTransformer del proyecto: numéricas con mediana+escalado, categóricas con moda+OneHot."""
    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocesador = ColumnTransformer(transformers=[
        ("num", pipeline_numerico, COLUMNAS_NUMERICAS),
        ("cat", pipeline_categorico, COLUMNAS_CATEGORICAS),
    ])

    return preprocesador


def dividir_datos(X, y, test_size=0.30, val_size=0.50, random_state=RANDOM_STATE):
    """Split estratificado 70/15/15 (train/val/test), igual que en el notebook."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preparar_dataset(ruta_csv):
    """Atajo: carga, limpia y separa X/y en un solo paso."""
    df = cargar_datos(ruta_csv)
    df = limpiar_datos(df)
    X, y = separar_X_y(df)
    return X, y
