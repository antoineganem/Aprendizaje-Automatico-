import os
from typing import Tuple, Dict, List, Optional
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de features (sin CLI)
FEATURE_SET_DEFAULT = "baseline"  # opciones: "baseline", "augmented"


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    """Lee un CSV con pandas usando codificación robusta y tipos flexibles."""
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin-1")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas de fecha si existen, maneja errores y crea la diferencia de días de envío.
    """
    date_cols = {
        "Order Date": "Order Date",
        "Ship Date": "Ship Date",
    }
    for col in list(date_cols.values()):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Order Date" in df.columns and "Ship Date" in df.columns:
        df["Days To Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days
    else:
        df["Days To Ship"] = float("nan")

    # Derivados temporales
    if "Order Date" in df.columns:
        df["OrderYear"] = df["Order Date"].dt.year
        df["OrderMonth"] = df["Order Date"].dt.month
        df["OrderDOW"] = df["Order Date"].dt.dayofweek
    else:
        df["OrderYear"] = float("nan")
        df["OrderMonth"] = float("nan")
        df["OrderDOW"] = float("nan")

    return df


def temporal_train_test_split(
    df: pd.DataFrame, date_col: str, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide por tiempo usando el percentil de fecha.
    """
    if date_col not in df.columns:
        # Si no hay fecha, usar una división aleatoria estable
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int((1 - test_fraction) * len(df))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    cutoff_date = df_sorted[date_col].quantile(1 - test_fraction)
    train_df = df_sorted[df_sorted[date_col] <= cutoff_date].copy()
    test_df = df_sorted[df_sorted[date_col] > cutoff_date].copy()
    return train_df, test_df


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide por tiempo en tres conjuntos: train, validación y test.
    Usa percentiles de la columna de fecha. Si no hay fecha, hace split aleatorio estable.
    """
    if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1:
        raise ValueError("Fracciones inválidas: val_fraction + test_fraction debe ser < 1 y no negativas.")

    if date_col not in df.columns:
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df_shuffled)
        n_test = int(round(test_fraction * n))
        n_val = int(round(val_fraction * n))
        n_train = n - n_val - n_test
        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
        test_df = df_shuffled.iloc[n_train + n_val:].copy()
        return train_df, val_df, test_df

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    q_train_end = 1.0 - (val_fraction + test_fraction)
    q_val_end = 1.0 - test_fraction
    cutoff_train_end = df_sorted[date_col].quantile(q_train_end)
    cutoff_val_end = df_sorted[date_col].quantile(q_val_end)

    train_df = df_sorted[df_sorted[date_col] <= cutoff_train_end].copy()
    val_df = df_sorted[(df_sorted[date_col] > cutoff_train_end) & (df_sorted[date_col] <= cutoff_val_end)].copy()
    test_df = df_sorted[df_sorted[date_col] > cutoff_val_end].copy()
    return train_df, val_df, test_df


class StandardScalerManual:
    """Estandarizador sencillo (media/desviación) implementado a mano sin librerías."""

    def __init__(self) -> None:
        self.feature_means: Optional[List[float]] = None
        self.feature_stds: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> None:
        if not X:
            self.feature_means = []
            self.feature_stds = []
            return
        num_rows = len(X)
        num_cols = len(X[0])
        means = [0.0] * num_cols
        for row in X:
            for j in range(num_cols):
                means[j] += row[j]
        for j in range(num_cols):
            means[j] /= max(num_rows, 1)

        vars_ = [0.0] * num_cols
        for row in X:
            for j in range(num_cols):
                diff = row[j] - means[j]
                vars_[j] += diff * diff
        for j in range(num_cols):
            vars_[j] /= max(num_rows, 1)
        stds = [sqrt(v) if v > 0 else 1.0 for v in vars_]

        self.feature_means = means
        self.feature_stds = stds

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        if self.feature_means is None or self.feature_stds is None:
            raise RuntimeError("Scaler no ajustado. Llama a fit primero.")
        num_cols = len(self.feature_means)
        X_out: List[List[float]] = []
        for row in X:
            X_out.append([(row[j] - self.feature_means[j]) / (self.feature_stds[j] or 1.0) for j in range(num_cols)])
        return X_out

    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        self.fit(X)
        return self.transform(X)


def _transpose(M: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*M)]


def _matvec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a_ij * v[j] for j, a_ij in enumerate(row)) for row in A]


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    Bt = _transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Resuelve Ax=b por eliminación gaussiana con pivoteo parcial."""
    n = len(A)
    # Construir matriz aumentada
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        # Pivoteo parcial
        pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot_row][i]) < 1e-12:
            raise ValueError("Matriz singular o casi singular al resolver el sistema.")
        if pivot_row != i:
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]
        # Normalizar fila i
        pivot = aug[i][i]
        inv_pivot = 1.0 / pivot
        for j in range(i, n + 1):
            aug[i][j] *= inv_pivot
        # Eliminar en otras filas
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor == 0.0:
                continue
            for c in range(i, n + 1):
                aug[r][c] -= factor * aug[i][c]
    # Extraer solución
    return [aug[i][n] for i in range(n)]


def ols_closed_form(X: List[List[float]], y: List[float]) -> List[float]:
    """OLS sin librerías: resuelve (X^T X) w = X^T y y retorna w."""
    Xt = _transpose(X)
    XtX = _matmul(Xt, X)
    Xty = _matvec(Xt, y)
    w = _solve_linear_system(XtX, Xty)
    return w


 

def build_features(
    df: pd.DataFrame,
    target_preference: List[str] | None = None,
    feature_set: str = "baseline",
) -> Tuple[pd.DataFrame, str]:
    """
    Construye features a partir del dataframe y retorna (X_df, target_col).
    """
    if target_preference is None:
        target_preference = ["Profit", "Sales"]

    # Elegir objetivo disponible
    target_col = None
    for col in target_preference:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError("No se encontró columna objetivo 'Profit' ni 'Sales'.")

    # Candidatos categóricos y numéricos
    categorical_cols = [
        c
        for c in ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
        if c in df.columns
    ]

    numeric_cols = [c for c in ["Quantity", "Discount", "Sales", "Days To Ship", "OrderYear", "OrderMonth", "OrderDOW"] if c in df.columns]

    # Evitar usar la misma columna como feature si es el objetivo
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Filas válidas
    use_cols = numeric_cols + categorical_cols + [target_col]
    work = df[use_cols].copy()

    # Rellenos simples
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[numeric_cols] = work[numeric_cols].fillna(work[numeric_cols].median())
    work[categorical_cols] = work[categorical_cols].fillna("Desconocido")

    # Interacciones opcionales
    if feature_set == "augmented":
        if ("Sales" in work.columns) and ("Discount" in work.columns):
            work["Sales_x_Discount"] = work["Sales"] * work["Discount"]
        if ("Quantity" in work.columns) and ("Discount" in work.columns):
            work["Quantity_x_Discount"] = work["Quantity"] * work["Discount"]

    # One-hot
    feature_cols = [c for c in work.columns if c != target_col]
    X_df = pd.get_dummies(work[feature_cols], drop_first=True)

    return X_df, target_col


def fit_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    val_df: Optional[pd.DataFrame] = None,
    feature_set: str = FEATURE_SET_DEFAULT,
) -> Dict[str, object]:
    """
    Ajusta un modelo de regresión lineal (OLS) y evalúa. Devuelve métricas y coeficientes en escala original.
    """
    # Construir features coherentes entre train/test
    X_train_df, _ = build_features(train_df, target_preference=[target_col], feature_set=feature_set)
    X_test_df, _ = build_features(test_df, target_preference=[target_col], feature_set=feature_set)
    X_val_df = None
    if val_df is not None:
        X_val_df, _ = build_features(val_df, target_preference=[target_col], feature_set=feature_set)

    # Alinear columnas
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)
    if X_val_df is not None:
        X_val_df = X_val_df.reindex(columns=X_train_df.columns, fill_value=0)

    y_train = [float(v) for v in train_df[target_col].tolist()]
    y_test = [float(v) for v in test_df[target_col].tolist()]
    y_val: Optional[List[float]] = None
    if val_df is not None:
        y_val = [float(v) for v in val_df[target_col].tolist()]

    # Estandarizar features (no la y)
    scaler = StandardScalerManual()
    X_train_list = [list(map(float, row)) for row in X_train_df.itertuples(index=False, name=None)]
    X_test_list = [list(map(float, row)) for row in X_test_df.itertuples(index=False, name=None)]
    X_val_list: Optional[List[List[float]]] = None
    if X_val_df is not None:
        X_val_list = [list(map(float, row)) for row in X_val_df.itertuples(index=False, name=None)]

    X_train_std = scaler.fit_transform(X_train_list)
    X_test_std = scaler.transform(X_test_list)
    X_val_std: Optional[List[List[float]]] = None
    if X_val_list is not None:
        X_val_std = scaler.transform(X_val_list)

    # Añadir intercepto
    def add_intercept(X: List[List[float]]) -> List[List[float]]:
        return [[1.0] + row for row in X]

    X_train_design = add_intercept(X_train_std)
    X_test_design = add_intercept(X_test_std)
    X_val_design: Optional[List[List[float]]] = None
    if X_val_std is not None:
        X_val_design = add_intercept(X_val_std)

    # Entrenar (OLS)
    w = ols_closed_form(X_train_design, y_train)

    # Predicciones
    def predict(X: List[List[float]], wv: List[float]) -> List[float]:
        return [sum(x[j] * wv[j] for j in range(len(wv))) for x in X]

    y_pred_train = predict(X_train_design, w)
    y_pred_test = predict(X_test_design, w)
    y_pred_val: Optional[List[float]] = None
    if X_val_design is not None:
        y_pred_val = predict(X_val_design, w)

    # Métricas
    def rmse(y_true: List[float], y_pred: List[float]) -> float:
        n = max(len(y_true), 1)
        return sqrt(sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n)

    def mae(y_true: List[float], y_pred: List[float]) -> float:
        n = max(len(y_true), 1)
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    def r2(y_true: List[float], y_pred: List[float]) -> float:
        mean_y = sum(y_true) / max(len(y_true), 1)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    metrics: Dict[str, float] = {
        "train_rmse": rmse(y_train, y_pred_train),
        "test_rmse": rmse(y_test, y_pred_test),
        "train_mae": mae(y_train, y_pred_train),
        "test_mae": mae(y_test, y_pred_test),
        "train_r2": r2(y_train, y_pred_train),
        "test_r2": r2(y_test, y_pred_test),
    }
    if y_val is not None and y_pred_val is not None:
        metrics.update({
            "val_rmse": rmse(y_val, y_pred_val),
            "val_mae": mae(y_val, y_pred_val),
            "val_r2": r2(y_val, y_pred_val),
        })

    # Convertir coeficientes a escala original
    feature_names = list(X_train_df.columns)
    means = scaler.feature_means if scaler.feature_means is not None else [0.0] * len(feature_names)
    stds = scaler.feature_stds if scaler.feature_stds is not None else [1.0] * len(feature_names)
    w0 = float(w[0])
    w_std = w[1:]
    betas = [w_std[j] / (stds[j] or 1.0) for j in range(len(feature_names))]
    intercept = w0 - sum(betas[j] * means[j] for j in range(len(feature_names)))

    coef_series = pd.Series(betas, index=feature_names).sort_values(ascending=False)

    return {
        "weights_standardized": w,
        "intercept_original": intercept,
        "coefs_original": coef_series,
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "y_test": y_test,
        "y_pred_val": y_pred_val,
    }
def plot_pred_vs_actual(y_true: List[float], y_pred: List[float], title: str, out_path: Optional[str] = None) -> None:
    """Grafica scatter de y_pred vs y_true con línea y=x y ejes igualados.

    Se recorta la vista a los percentiles [1%, 99%] combinados de y_true/y_pred para
    evitar que outliers compriman la nube de puntos.
    """
    if not y_true or not y_pred:
        return

    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    low = float(min(s_true.quantile(0.01), s_pred.quantile(0.01)))
    high = float(max(s_true.quantile(0.99), s_pred.quantile(0.99)))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.5, edgecolor="none")
    plt.plot([low, high], [low, high], color="red", linestyle="--", linewidth=1.5, label="y = x")
    ax = plt.gca()
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("Profit real")
    plt.ylabel("Profit predicho")
    plt.title(title + " (vista 1-99% percentiles)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()



def simple_eda(df: pd.DataFrame) -> None:
    """Exploración de datos sencilla y legible para humanos."""
    print("\n=== Vista rápida del dataset ===")
    print(f"Filas: {len(df):,}  Columnas: {len(df.columns)}")
    print("Columnas:")
    print(", ".join(df.columns))

    print("\n=== Nulos por columna (top 10) ===")
    nulls = df.isna().sum().sort_values(ascending=False)
    print(nulls.head(10))

    # Variables numéricas básicas
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        print("\n=== Estadísticas numéricas (selección) ===")
        sel = [c for c in ["Sales", "Profit", "Quantity", "Discount"] if c in numeric_cols]
        if sel:
            print(df[sel].describe().T)

    # Top categorías
    for c in ["Category", "Sub-Category", "Region", "Segment"]:
        if c in df.columns:
            print(f"\n=== Top valores de {c} ===")
            print(df[c].value_counts().head(10))

    # Correlaciones con Profit / Sales
    for target in ["Profit", "Sales"]:
        if target in df.columns and numeric_cols:
            corr_series = df[numeric_cols].corrwith(df[target]).dropna().sort_values(ascending=False)
            print(f"\n=== Correlaciones con {target} (top 10) ===")
            print(corr_series.head(10))
            print(f"\n=== Correlaciones con {target} (bottom 10) ===")
            print(corr_series.tail(10))


def main() -> None:
    # Fracciones fijas para validación y test
    VAL_FRACTION = 0.2
    TEST_FRACTION = 0.2

    # 1) Cargar dataset desde CSV local fijo junto al script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Sample - Superstore.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe el archivo CSV esperado: {csv_path}")
    print(f"Leyendo CSV local con pandas: {csv_path}")
    df = read_csv_robust(csv_path)

    # 2) EDA simple
    simple_eda(df)

    # 3) Preparación básica de fechas y división temporal
    df = parse_dates(df)
    train_df, val_df, test_df = temporal_train_val_test_split(
        df,
        date_col="Order Date",
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )
 
    # 4) Construcción de features y entrenamiento del modelo
    # Preferimos predecir Profit; si no existe, Sales
    target_col = "Profit" if "Profit" in df.columns else ("Sales" if "Sales" in df.columns else None)
    if target_col is None:
        raise ValueError("El dataset no tiene 'Profit' ni 'Sales'.")

    # 5) Entrenar y reportar OLS baseline y OLS con features aumentadas
    results_ols = fit_and_evaluate(train_df, test_df, target_col=target_col, val_df=val_df, feature_set="baseline")
    results_ols_aug = fit_and_evaluate(train_df, test_df, target_col=target_col, val_df=val_df, feature_set="augmented")

    # Métricas
    print("\n=== Métricas OLS (train/val/test) ===")
    for k, v in results_ols["metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\n=== Métricas OLS con features aumentadas (train/val/test) ===")
    for k, v in results_ols_aug["metrics"].items():
        print(f"{k}: {v:.4f}")

    # Coeficientes principales
    print("\n=== Principales coeficientes OLS (escala original) ===")
    coefs_ols: pd.Series = results_ols["coefs_original"]  # type: ignore
    print("Top positivos:")
    print(coefs_ols.head(15))
    print("\nTop negativos:")
    print(coefs_ols.tail(15))

    print("\n=== Principales coeficientes OLS (features aumentadas) ===")
    coefs_aug: pd.Series = results_ols_aug["coefs_original"]  # type: ignore
    print("Top positivos:")
    print(coefs_aug.head(15))
    print("\nTop negativos:")
    print(coefs_aug.tail(15))

    # Gráficas: y_pred vs y_true en test para ambos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    y_test_list_ols: List[float] = results_ols.get("y_test", [])  # type: ignore
    y_pred_test_list_ols: List[float] = results_ols.get("y_pred_test", [])  # type: ignore
    out_png_ols = os.path.join(script_dir, "pred_vs_actual_test_ols.png")
    plot_pred_vs_actual(y_test_list_ols, y_pred_test_list_ols, title="Profit predicho vs real (test) - OLS", out_path=out_png_ols)

    # Gráfica para OLS aumentado
    y_test_list_aug: List[float] = results_ols_aug.get("y_test", [])  # type: ignore
    y_pred_test_list_aug: List[float] = results_ols_aug.get("y_pred_test", [])  # type: ignore
    out_png_aug = os.path.join(script_dir, "pred_vs_actual_test_ols_aug.png")
    plot_pred_vs_actual(y_test_list_aug, y_pred_test_list_aug, title="Profit predicho vs real (test) - OLS (features aumentadas)", out_path=out_png_aug)


if __name__ == "__main__":
    main()

