# Guía de inicio rápido en Google Colab

Notebook del proyecto: **[ProyectoFinal_AAA.ipynb en Colab](https://colab.research.google.com/drive/1GRVuCeHBBpEiS5yPVvj8m46F6V9HLGSu?usp=sharing)**

Esta guía explica cómo dejar el notebook funcionando en Colab con el dataset y los módulos de `src/` disponibles, sin depender de rutas locales.

## Opción A — Subir el proyecto completo a Google Drive (recomendada)

1. Sube la carpeta completa `proyecto_churn_telecom/` (o al menos `data/`, `src/` y `notebooks/`) a tu Google Drive, por ejemplo a `MyDrive/proyecto_churn_telecom/`.
2. En la primera celda del notebook, monta Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Define la ruta base del proyecto y añade `src/` al path de Python:

   ```python
   import sys

   RUTA_PROYECTO = "/content/drive/MyDrive/proyecto_churn_telecom"
   sys.path.append(RUTA_PROYECTO)

   RUTA_DATOS = f"{RUTA_PROYECTO}/data/TelcoCustomer.csv"
   ```

4. Instala las dependencias exactas del proyecto (Colab ya trae pandas/sklearn, pero puede tener otra versión):

   ```python
   !pip install -q -r "{RUTA_PROYECTO}/requirements.txt"
   ```

5. Ya puedes usar los módulos reutilizables igual que en local:

   ```python
   from src.preprocessing import cargar_datos, limpiar_datos, separar_X_y, construir_preprocesador, dividir_datos
   from src.evaluation import evaluar_modelo, comparar_modelos, graficar_matriz_confusion

   df = cargar_datos(RUTA_DATOS)
   df = limpiar_datos(df)
   X, y = separar_X_y(df)
   X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)
   ```

## Opción B — Subir solo el CSV (sesión temporal, sin Drive)

Si solo quieres correr el notebook sin usar `src/` ni persistir archivos entre sesiones:

```python
from google.colab import files
subido = files.upload()  # selecciona TelcoCustomer.csv desde tu computadora

import pandas as pd
df = pd.read_csv("TelcoCustomer.csv")
```

Ten en cuenta que los archivos subidos así se **borran al reiniciar el entorno de ejecución**; para trabajo continuo, usa la Opción A.

## Guardar el modelo entrenado

Al final del notebook se serializa el pipeline completo con `joblib`:

```python
import joblib
joblib.dump(modelo_final, f"{RUTA_PROYECTO}/models/modelo_churn_final_v1.joblib")
```

Guardarlo dentro de `models/` en Drive permite reutilizarlo después con `scripts/predict.py` en local, sin tener que reentrenar.

## Notas de reproducibilidad

- Todos los splits y modelos usan `random_state=42`.
- Verifica versiones antes de comparar resultados con el reporte original:

  ```python
  import sys, pandas as pd, numpy as np, sklearn
  print(sys.version, pd.__version__, np.__version__, sklearn.__version__)
  ```

- Versiones usadas al generar el proyecto: Python 3.13, pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1 (ver [requirements.txt](../requirements.txt)).
