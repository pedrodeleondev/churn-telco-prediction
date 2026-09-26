# Predicción de Churn en Telco

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)
![Status](https://img.shields.io/badge/status-completado-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

Proyecto de clasificación binaria para predecir el abandono de clientes (**Churn**) en Telco, una empresa de telecomunicaciones, usando el dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 clientes, 21 variables).

Notebook original en Colab: **[ProyectoFinal_AAA.ipynb](https://colab.research.google.com/drive/1GRVuCeHBBpEiS5yPVvj8m46F6V9HLGSu?usp=sharing)**

## Contexto y objetivo de negocio

| Elemento | Descripción |
|---|---|
| Contexto organizacional | Empresa de servicios por suscripción que busca reducir la pérdida de clientes identificando anticipadamente los casos con mayor riesgo de abandono |
| Unidad de análisis | Cada fila representa a un cliente único, con sus atributos demográficos, servicios contratados, condiciones y método de pago |
| Variable objetivo | `Churn` — indica si el cliente abandonó (`Yes`) o permaneció (`No`) con el servicio |
| Clase positiva | `Churn = Yes` → codificada como `1` |
| Usuarios potenciales | Equipo de retención de clientes, análisis de datos, marketing y experiencia de cliente |
| Decisión que apoya el modelo | A qué clientes contactar primero, o con qué prioridad ofrecer una acción de retención |
| Riesgo de uso incorrecto | Tratar la probabilidad de abandono como algo determinístico para automatizar decisiones, generando injusticias o acciones desfavorables hacia clientes |
| Criterio de éxito | Superar el baseline (`DummyClassifier`) en Recall y F1, manteniendo una precisión razonable |

## Resumen del proyecto

| Elemento | Detalle |
|---|---|
| Problema | Clasificación binaria (Churn: Sí/No) |
| Dataset | 7,043 registros × 21 columnas |
| Métrica principal | **Recall** (minimizar falsos negativos: clientes que abandonan y no se detectan) |
| Modelos evaluados | Regresión Logística, Árbol de Decisión, Random Forest, SVM (lineal y RBF), con `DummyClassifier` como baseline |
| Modelo final | Regresión Logística (mejor balance Recall/interpretabilidad/costo operativo) |
| Umbral operativo | 0.30 (en vez de 0.50, para priorizar Recall) |
| Reproducibilidad | `random_state=42` en todas las particiones y modelos |

## Cumplimiento de los requisitos de reproducibilidad

Este repositorio sigue la estructura esperada para un proyecto de ML reproducible:

| Requisito | Ubicación | Estado |
|---|---|---|
| Notebooks | [`notebooks/ProyectoFinal_AAA.ipynb`](notebooks/ProyectoFinal_AAA.ipynb) | Análisis, EDA, modelado y evaluación completos |
| Scripts | [`src/preprocessing.py`](src/preprocessing.py), [`src/evaluation.py`](src/evaluation.py), [`scripts/train.py`](scripts/train.py), [`scripts/predict.py`](scripts/predict.py), [`scripts/api.py`](scripts/api.py) | Módulos reutilizables + entrenamiento + CLI + API |
| Datos / instrucciones de descarga | [`data/TelcoCustomer.csv`](data/TelcoCustomer.csv), [`data/README.md`](data/README.md) | Dataset incluido y documentado |
| Artefactos del modelo | [`models/modelo_churn_final_v1.joblib`](models/modelo_churn_final_v1.joblib) | Pipeline entrenado (preprocesamiento + Regresión Logística) generado con `scripts/train.py` |
| Aplicación / API | [`scripts/predict.py`](scripts/predict.py) (CLI) + [`scripts/api.py`](scripts/api.py) (API REST con FastAPI) | Predicción por línea de comandos y por HTTP (`/predict`, `/predict/batch`) |
| Resultados | [`results/`](results/) | Gráficos, tablas de comparación de modelos y métricas finales generados con `scripts/train.py` |
| Documentación | Este README, [`GITHUB_SETUP.md`](GITHUB_SETUP.md), [`notebooks/COLAB_INICIO.md`](notebooks/COLAB_INICIO.md), [`data/README.md`](data/README.md) | Completa |
| Dependencias | [`requirements.txt`](requirements.txt) | Versiones fijadas, incluye extras de la API |

> Nota: `models/*.joblib` y los archivos generados en `results/` sí están versionados en este repositorio (son pequeños: el modelo pesa ~9 KB) para que el proyecto quede completo y ejecutable sin pasos adicionales. El `.gitignore` los excluye por defecto como buena práctica general para modelos grandes; si tu modelo creciera mucho, quítalos del control de versiones y usa `scripts/train.py` para regenerarlos.

## Estructura del proyecto

```
proyecto_churn_telecom/
├── data/
│   ├── TelcoCustomer.csv          # Dataset original
│   └── README.md                  # Descripción y diccionario de variables
├── notebooks/
│   ├── ProyectoFinal_AAA.ipynb    # Notebook completo del análisis y modelado
│   └── COLAB_INICIO.md            # Guía para ejecutar en Google Colab
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Limpieza, ColumnTransformer, split train/val/test
│   └── evaluation.py              # Métricas, comparación de modelos, gráficos
├── scripts/
│   ├── train.py                   # Entrena el modelo final y genera models/ + results/
│   ├── predict.py                 # CLI para predecir churn con el modelo entrenado
│   └── api.py                     # API REST (FastAPI) para predecir churn por HTTP
├── models/
│   └── modelo_churn_final_v1.joblib  # Pipeline entrenado (preprocesamiento + Regresión Logística)
├── results/
│   ├── comparacion_modelos.csv    # Métricas de validación cruzada por modelo
│   ├── analisis_umbrales.csv      # Precision/Recall/F1 por umbral de decisión
│   ├── metricas_finales.json      # Métricas del modelo final en el conjunto de prueba
│   ├── importancia_variables.csv  # Ranking de variables por coeficiente
│   ├── matriz_confusion.png
│   ├── curva_roc.png
│   ├── curva_precision_recall.png
│   └── importancia_variables.png
├── requirements.txt
├── .gitignore
├── GITHUB_SETUP.md
└── README.md
```

## Instalación y uso local

### 1. Clonar y crear entorno virtual

```bash
git clone <url-del-repositorio>
cd proyecto_churn_telecom

python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Ejecutar el notebook

```bash
jupyter notebook notebooks/ProyectoFinal_AAA.ipynb
```

El notebook lee el dataset desde `../data/TelcoCustomer.csv` (ruta relativa) y usa los módulos en `src/`.

### 3. Usar los módulos reutilizables directamente

```python
from src.preprocessing import cargar_datos, limpiar_datos, separar_X_y, construir_preprocesador, dividir_datos
from src.evaluation import evaluar_modelo, comparar_modelos, evaluar_umbrales

df = cargar_datos("data/TelcoCustomer.csv")
df = limpiar_datos(df)
X, y = separar_X_y(df)

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)
```

### 4. Entrenar el modelo y generar los artefactos (`models/`, `results/`)

El repositorio ya incluye `models/modelo_churn_final_v1.joblib` y los archivos de `results/` generados. Si quieres regenerarlos desde cero (por ejemplo, tras modificar `src/preprocessing.py` o `src/evaluation.py`), usa el script de entrenamiento en vez de correr el notebook manualmente:

```bash
python scripts/train.py
```

Esto reproduce la metodología completa del notebook:

- Carga y limpia `data/TelcoCustomer.csv`.
- Divide en train/val/test (70/15/15, estratificado, `random_state=42`).
- Compara 6 modelos (`DummyClassifier`, Regresión Logística, Árbol de Decisión, Random Forest, SVM lineal, SVM RBF) con validación cruzada `StratifiedKFold(5)` → guarda `results/comparacion_modelos.csv`.
- Entrena el modelo final (Regresión Logística) y lo evalúa en el conjunto de prueba con varios umbrales → guarda `results/analisis_umbrales.csv` y `results/metricas_finales.json`.
- Genera y guarda los gráficos: matriz de confusión, curva ROC, curva Precision-Recall e importancia de variables.
- Guarda el pipeline entrenado en `models/modelo_churn_final_v1.joblib`.

Flags disponibles: `--data`, `--models-dir`, `--results-dir`, `--threshold`, `--skip-comparison` (omite la validación cruzada para una corrida más rápida).

### 5. Hacer predicciones con el modelo entrenado

**Opción A — CLI (por lotes, desde un CSV):**

```bash
python scripts/predict.py --input data/nuevos_clientes.csv --output results/predicciones.csv
```

| Flag | Descripción | Default |
|---|---|---|
| `--model` | Ruta al pipeline `.joblib` | `models/modelo_churn_final_v1.joblib` |
| `--input` | CSV con los clientes a predecir (obligatorio) | — |
| `--output` | CSV de salida (si se omite, imprime en pantalla) | — |
| `--threshold` | Umbral de decisión para clasificar como abandono | `0.30` |

**Opción B — API REST (por HTTP, un cliente o varios a la vez):**

```bash
uvicorn scripts.api:app --reload --port 8000
```

Con el servidor corriendo:

```bash
# Verificar que el modelo cargó correctamente
curl http://127.0.0.1:8000/health

# Predecir un cliente
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
      }'
```

Respuesta esperada:

```json
{
  "customerID": "7590-VHVEG",
  "probabilidad_churn": 0.6031,
  "prediccion": 1,
  "clasificacion": "Posible abandono",
  "umbral_utilizado": 0.3
}
```

También hay un endpoint `POST /predict/batch` que acepta una lista de clientes, y documentación interactiva (Swagger UI) en `http://127.0.0.1:8000/docs`.

La ruta del modelo y el umbral operativo son configurables por variables de entorno: `MODEL_PATH` y `UMBRAL_OPERATIVO`.

## Uso en Google Colab

Ver la guía detallada en [notebooks/COLAB_INICIO.md](notebooks/COLAB_INICIO.md), que explica cómo montar Google Drive, instalar `requirements.txt` y usar los módulos de `src/` desde Colab.

## Metodología

1. **Análisis exploratorio**: distribución del target, variables numéricas y categóricas, correlaciones, desbalance de clases.
2. **Calidad de datos**: conversión de `TotalCharges` a numérico, exclusión de `customerID`, revisión de fuga de información.
3. **Preprocesamiento**: `ColumnTransformer` con imputación + `StandardScaler` (numéricas) e imputación + `OneHotEncoder` (categóricas), encapsulado en un `Pipeline` de scikit-learn junto con el modelo.
4. **Partición**: 70% entrenamiento / 15% validación / 15% prueba, estratificada por `Churn`, `random_state=42`.
5. **Modelado**: comparación de baseline (`DummyClassifier`), Regresión Logística, Árbol de Decisión, Random Forest y SVM (lineal/RBF), con validación cruzada `StratifiedKFold(5)` y `GridSearchCV` para ajuste de hiperparámetros.
6. **Selección del modelo**: matriz de decisión ponderada (Recall, Precision, F1, ROC-AUC, estabilidad, interpretabilidad, costo operativo) entre los dos finalistas (Regresión Logística vs. Random Forest optimizado).
7. **Ajuste de umbral**: análisis de umbrales de 0.30 a 0.70 sobre el conjunto de prueba; se elige 0.30 como umbral operativo para maximizar Recall.
8. **Interpretabilidad**: ranking de variables por coeficiente, análisis de casos individuales (verdaderos/falsos positivos/negativos) y revisión de sesgos por grupo.
9. **Serialización**: pipeline completo (preprocesamiento + modelo) guardado con `joblib` y verificado tras la recarga.

## Resultados en conjunto de prueba (modelo final)

| Métrica | Umbral 0.50 | Umbral 0.30 (operativo) |
|---|---|---|
| Accuracy | 0.7947 | 0.7654 |
| Precision | 0.6884 | 0.5429 |
| Recall | 0.5267 | **0.7438** |
| F1-score | 0.5968 | 0.6276 |
| ROC-AUC | 0.8447 | 0.8447 (no cambia con el umbral) |

Se prioriza Recall sobre Precision: con el umbral 0.30 el modelo detecta ~74% de los clientes que realmente abandonan (vs. ~53% con el umbral por defecto de 0.50), a costa de más falsas alarmas — un intercambio deseable cuando el costo de no detectar un abandono es mayor que el de contactar innecesariamente a un cliente.

Comparación completa de los 6 modelos evaluados (validación cruzada, ordenados por Recall):

| Modelo | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| **Regresión Logística** | 0.5458 | 0.6536 | 0.5943 | 0.8442 |
| SVM Lineal | 0.5329 | 0.6491 | 0.5849 | 0.8322 |
| Árbol de Decisión | 0.5214 | 0.6173 | 0.5623 | 0.8223 |
| SVM RBF | 0.4954 | 0.6749 | 0.5708 | 0.7981 |
| Random Forest | 0.4862 | 0.6266 | 0.5473 | 0.8245 |
| DummyClassifier (baseline) | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

> Estos valores se generaron con `python scripts/train.py` y quedaron guardados en [`results/comparacion_modelos.csv`](results/comparacion_modelos.csv), [`results/analisis_umbrales.csv`](results/analisis_umbrales.csv) y [`results/metricas_finales.json`](results/metricas_finales.json). Los gráficos correspondientes (matriz de confusión, curvas ROC/PR, importancia de variables) están en `results/*.png`. El notebook (`notebooks/ProyectoFinal_AAA.ipynb`) documenta además el análisis completo de interpretabilidad, casos individuales y revisión de sesgos por grupo.

## Equipo

Frida Garza, Paola Urdiales, Pedro De León, Ángel Robles y Cristian Flores.

## Licencia

MIT — ver [GITHUB_SETUP.md](GITHUB_SETUP.md) para instrucciones de publicación.
