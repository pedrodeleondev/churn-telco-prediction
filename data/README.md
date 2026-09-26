# Dataset: Telco Customer Churn

## Descripción general

| Propiedad | Valor |
|---|---|
| Archivo | `TelcoCustomer.csv` |
| Registros | 7,043 clientes |
| Columnas | 21 |
| Variable objetivo | `Churn` (Yes/No) |
| Clase positiva | `Churn = Yes` → codificada como `1` |
| Balance de clases | ~73.5% No abandona / ~26.5% Abandona |
| Fuente | [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

## Diccionario de variables

| Columna | Descripción | Tipo | Tratamiento en el pipeline |
|---|---|---|---|
| `customerID` | Identificador único del cliente | Texto | Excluida del modelado (no predictiva) |
| `gender` | Género del cliente | Categórica binaria | One-Hot Encoding |
| `SeniorCitizen` | Adulto mayor (0/1) | Categórica binaria | One-Hot Encoding |
| `Partner` | Tiene pareja | Categórica binaria | One-Hot Encoding |
| `Dependents` | Tiene dependientes económicos | Categórica binaria | One-Hot Encoding |
| `tenure` | Meses de antigüedad (0-72) | Numérica continua | Imputación (mediana) + StandardScaler |
| `PhoneService` | Tiene servicio telefónico | Categórica | One-Hot Encoding |
| `MultipleLines` | Varias líneas telefónicas | Categórica | One-Hot Encoding |
| `InternetService` | Tipo de internet (DSL/Fiber optic/No) | Categórica | One-Hot Encoding |
| `OnlineSecurity` | Seguridad en línea | Categórica | One-Hot Encoding |
| `OnlineBackup` | Respaldo en línea | Categórica | One-Hot Encoding |
| `DeviceProtection` | Protección de dispositivo | Categórica | One-Hot Encoding |
| `TechSupport` | Soporte técnico | Categórica | One-Hot Encoding |
| `StreamingTV` | Streaming de TV | Categórica | One-Hot Encoding |
| `StreamingMovies` | Streaming de películas | Categórica | One-Hot Encoding |
| `Contract` | Tipo de contrato (Month-to-month/One year/Two year) | Categórica | One-Hot Encoding |
| `PaperlessBilling` | Facturación electrónica | Categórica binaria | One-Hot Encoding |
| `PaymentMethod` | Método de pago (4 categorías) | Categórica | One-Hot Encoding |
| `MonthlyCharges` | Cargo mensual actual (USD) | Numérica continua | Imputación (mediana) + StandardScaler |
| `TotalCharges` | Cargo total acumulado (USD) | Numérica mal tipada como texto | `pd.to_numeric(errors='coerce')` + imputación (mediana) |
| `Churn` | Abandonó el servicio | Binaria (objetivo) | Yes=1 / No=0 |

## Problemas de calidad detectados

- **`TotalCharges`** viene como texto y tiene 11 registros con espacio en blanco en vez de un número. Se convierte con `pd.to_numeric(errors='coerce')` y se imputa con la mediana **dentro del pipeline** (ajustado solo con datos de entrenamiento, para evitar fuga de información).
- **`customerID`** tiene 7,043 valores únicos (uno por fila): se excluye como variable predictora por ser un identificador sin poder predictivo.
- Las columnas de servicios (`OnlineSecurity`, `TechSupport`, etc.) incluyen la categoría `"No internet service"` / `"No phone service"`, redundante con `InternetService`/`PhoneService`. Se conservan tal cual para no perder información.
- No se detectaron registros duplicados ni columnas con una sola categoría.
- La variable objetivo está desbalanceada (73.5%/26.5%): se usan métricas robustas al desbalance (Recall, F1, ROC-AUC) y estratificación (`stratify=y`) en las particiones.

## Cómo cargarlo

```python
from src.preprocessing import cargar_datos, limpiar_datos, separar_X_y

df = cargar_datos("data/TelcoCustomer.csv")
df = limpiar_datos(df)
X, y = separar_X_y(df)
```
