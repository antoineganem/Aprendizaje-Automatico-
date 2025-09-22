## Proyecto: Regresión lineal OLS (sin frameworks)

Este proyecto implementa un modelo de regresión lineal por mínimos cuadrados ordinarios (OLS) sin usar frameworks de Machine Learning. Se usa únicamente `pandas` para lectura del CSV y preparación del dataset; el entrenamiento, estandarización, álgebra lineal y métricas están implementados a mano.

### Requisitos
- **Python**: 3.9 o superior
- **Dependencias (script manual)**: `pandas`, `matplotlib`
- **Dependencias (script con librerías)**: `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`, `numpy`

Instalar dependencias (manual):
```bash
python3 -m pip install --upgrade pip
python3 -m pip install pandas matplotlib
```

Instalar dependencias (con librerías):
```bash
python3 -m pip install --upgrade pip
python3 -m pip install pandas matplotlib scikit-learn statsmodels numpy
```

Opcional (usar entorno virtual):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib
```

### Datos
El script espera el archivo CSV llamado exactamente:
- `Sample - Superstore.csv`

y debe estar en el mismo directorio que `lineal_regresion.py`.

Estructura mínima:
```
Proyecto Algoritmo de machine learning/
├─ lineal_regresion.py
└─ Sample - Superstore.csv
```

### Cómo ejecutar
Desde el directorio del proyecto, ejecuta:
```bash
python3 "lineal_regresion.py"
```

### Cómo ejecutar (versión con librerías)
El script `lineal_regresion_librerias.py` replica el flujo usando `scikit-learn` para el preprocesamiento (imputación, estandarización, one-hot) y `LinearRegression` como OLS. Además, opcionalmente imprime un resumen OLS de `statsmodels` sobre las features transformadas.

```bash
python3 "lineal_regresion_librerias.py"
```

Por defecto el script realiza:
- **Lectura** del CSV con `pandas`.
- **EDA rápida**: nulos, estadísticas, top categorías y correlaciones.
- **Preparación de fechas** y variables temporales.
- **Split temporal** en train/validación/test.
- **One-hot** de categóricas y **estandarización manual** de numéricas.
- **Entrenamiento OLS** sin regularización (implementado a mano).
- **Métricas** y **coeficientes** principales.
- **Gráfica** de `Profit` predicho vs real. En la versión manual se guarda como `pred_vs_actual_test_ols.png` y `pred_vs_actual_test_ols_aug.png`. En la versión con librerías se guarda como `pred_vs_actual_test_lr.png` y `pred_vs_actual_test_lr_aug.png`.

### Parámetros
No requiere parámetros de línea de comandos. Por defecto usa 20% validación y 20% test.

### Salida esperada (resumen)
- **EDA**
  - Filas/columnas y lista de columnas
  - Nulos por columna (top 10)
  - Estadísticos de `Sales`, `Profit`, `Quantity`, `Discount`
  - Top de `Category`, `Sub-Category`, `Region`, `Segment`
  - Correlaciones con `Profit` y `Sales`
- **Métricas del modelo (train/val/test)**
  - RMSE, MAE, R²
- **Coeficientes principales (escala original)**
  - Top positivos y negativos para interpretación rápida

### Notas importantes
- **Objetivo**: el modelo intenta predecir `Profit` si existe; si no, usa `Sales`.
- **Split temporal**: se basa en `Order Date`. Si falta, se hace split aleatorio estable.
- **Versión manual vs con librerías**: `lineal_regresion.py` implementa OLS sin frameworks; `lineal_regresion_librerias.py` usa `scikit-learn`/`statsmodels` para un flujo más práctico y extensible.
- **Rutas con espacios**: el CSV se busca automáticamente junto al script; asegúrate del nombre exacto.

### Solución de problemas
- "No existe el archivo CSV esperado": confirma que `Sample - Superstore.csv` está en el mismo directorio que `lineal_regresion.py`.
- Avisos de `pandas`: mantén `pandas` actualizado si ves deprecaciones.
- Estabilidad numérica: OLS puede ser sensible a outliers y multicolinealidad; considera limpiar datos o agrupar categorías raras si notas métricas inestables.
