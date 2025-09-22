import os
from typing import Tuple, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

# Librerías de ML/estadística
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import statsmodels.api as sm


FEATURE_SET_DEFAULT = "baseline"  # "baseline" o "augmented"


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin-1")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = ["Order Date", "Ship Date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Order Date" in df.columns and "Ship Date" in df.columns:
        df["Days To Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days
    else:
        df["Days To Ship"] = float("nan")

    if "Order Date" in df.columns:
        df["OrderYear"] = df["Order Date"].dt.year
        df["OrderMonth"] = df["Order Date"].dt.month
        df["OrderDOW"] = df["Order Date"].dt.dayofweek
    else:
        df["OrderYear"] = float("nan")
        df["OrderMonth"] = float("nan")
        df["OrderDOW"] = float("nan")

    return df


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1:
        raise ValueError("Fracciones inválidas: val_fraction + test_fraction debe ser < 1 y no negativas.")

    if date_col not in df.columns:
        # División estable aleatoria
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


def build_features(df: pd.DataFrame, target_preference: list[str] | None = None, feature_set: str = "baseline") -> tuple[pd.DataFrame, str]:
    if target_preference is None:
        target_preference = ["Profit", "Sales"]

    target_col: Optional[str] = None
    for col in target_preference:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError("No se encontró columna objetivo 'Profit' ni 'Sales'.")

    categorical_cols = [c for c in ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"] if c in df.columns]
    numeric_cols = [c for c in ["Quantity", "Discount", "Sales", "Days To Ship", "OrderYear", "OrderMonth", "OrderDOW"] if c in df.columns]

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    use_cols = numeric_cols + categorical_cols + [target_col]
    work = df[use_cols].copy()

    # Interacciones opcionales
    if feature_set == "augmented":
        if ("Sales" in work.columns) and ("Discount" in work.columns):
            work["Sales_x_Discount"] = work["Sales"] * work["Discount"]
        if ("Quantity" in work.columns) and ("Discount" in work.columns):
            work["Quantity_x_Discount"] = work["Quantity"] * work["Discount"]

    feature_cols = [c for c in work.columns if c != target_col]
    X_df = work[feature_cols].copy()
    return X_df, target_col


def make_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def fit_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, val_df: Optional[pd.DataFrame] = None, feature_set: str = FEATURE_SET_DEFAULT) -> Dict[str, object]:
    X_train_df, _ = build_features(train_df, target_preference=[target_col], feature_set=feature_set)
    X_test_df, _ = build_features(test_df, target_preference=[target_col], feature_set=feature_set)
    X_val_df = None
    if val_df is not None:
        X_val_df, _ = build_features(val_df, target_preference=[target_col], feature_set=feature_set)

    y_train = train_df[target_col].astype(float).to_numpy()
    y_test = test_df[target_col].astype(float).to_numpy()
    y_val = None
    if val_df is not None:
        y_val = val_df[target_col].astype(float).to_numpy()

    preprocessor = make_preprocessor(X_train_df)
    model = LinearRegression()  # OLS sin regularización
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    pipe.fit(X_train_df, y_train)

    y_pred_train = pipe.predict(X_train_df)
    y_pred_test = pipe.predict(X_test_df)
    y_pred_val = pipe.predict(X_val_df) if X_val_df is not None else None

    def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else float("nan")),
            "mae": float(mean_absolute_error(y_true, y_pred) if len(y_true) else float("nan")),
            "r2": float(r2_score(y_true, y_pred) if len(y_true) else float("nan")),
        }

    metrics: Dict[str, float] = {}
    m_train = metrics_block(y_train, y_pred_train)
    metrics.update({f"train_{k}": v for k, v in m_train.items()})
    m_test = metrics_block(y_test, y_pred_test)
    metrics.update({f"test_{k}": v for k, v in m_test.items()})
    if y_val is not None and y_pred_val is not None:
        m_val = metrics_block(y_val, y_pred_val)
        metrics.update({f"val_{k}": v for k, v in m_val.items()})

    # Extraer coeficientes en el espacio transformado y mapear a nombres de columnas
    # Usamos el método get_feature_names_out del preprocesador
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    intercept = float(pipe.named_steps["model"].intercept_)
    coefs_series = pd.Series(coefs, index=feature_names).sort_values(ascending=False)

    return {
        "pipeline": pipe,
        "intercept": intercept,
        "coefs": coefs_series,
        "metrics": metrics,
        "y_pred_test": y_pred_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred_val": None if y_pred_val is None else y_pred_val.tolist(),
    }


def plot_pred_vs_actual(y_true: list[float], y_pred: list[float], title: str, out_path: Optional[str] = None) -> None:
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


def main() -> None:
    VAL_FRACTION = 0.2
    TEST_FRACTION = 0.2

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Sample - Superstore.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe el archivo CSV esperado: {csv_path}")
    print(f"Leyendo CSV con pandas: {csv_path}")
    df = read_csv_robust(csv_path)

    # Preparación de fechas y split temporal
    df = parse_dates(df)
    train_df, val_df, test_df = temporal_train_val_test_split(
        df,
        date_col="Order Date",
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )

    target_col = "Profit" if "Profit" in df.columns else ("Sales" if "Sales" in df.columns else None)
    if target_col is None:
        raise ValueError("El dataset no tiene 'Profit' ni 'Sales'.")

    # Baseline y augmented
    results_lr = fit_and_evaluate(train_df, test_df, target_col=target_col, val_df=val_df, feature_set="baseline")
    results_lr_aug = fit_and_evaluate(train_df, test_df, target_col=target_col, val_df=val_df, feature_set="augmented")

    print("\n=== Métricas LinearRegression (train/val/test) ===")
    for k, v in results_lr["metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\n=== Métricas LinearRegression con features aumentadas (train/val/test) ===")
    for k, v in results_lr_aug["metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\n=== Principales coeficientes (espacio transformado) ===")
    coefs_lr: pd.Series = results_lr["coefs"]  # type: ignore
    print("Top positivos:")
    print(coefs_lr.head(15))
    print("\nTop negativos:")
    print(coefs_lr.tail(15))

    print("\n=== Principales coeficientes (features aumentadas) ===")
    coefs_lr_aug: pd.Series = results_lr_aug["coefs"]  # type: ignore
    print("Top positivos:")
    print(coefs_lr_aug.head(15))
    print("\nTop negativos:")
    print(coefs_lr_aug.tail(15))

    # Gráficas
    y_test_list_lr: list[float] = results_lr.get("y_test", [])  # type: ignore
    y_pred_test_list_lr: list[float] = results_lr.get("y_pred_test", [])  # type: ignore
    out_png_lr = os.path.join(script_dir, "pred_vs_actual_test_lr.png")
    plot_pred_vs_actual(y_test_list_lr, y_pred_test_list_lr, title="Profit predicho vs real (test) - LinearRegression", out_path=out_png_lr)

    y_test_list_lr_aug: list[float] = results_lr_aug.get("y_test", [])  # type: ignore
    y_pred_test_list_lr_aug: list[float] = results_lr_aug.get("y_pred_test", [])  # type: ignore
    out_png_lr_aug = os.path.join(script_dir, "pred_vs_actual_test_lr_aug.png")
    plot_pred_vs_actual(y_test_list_lr_aug, y_pred_test_list_lr_aug, title="Profit predicho vs real (test) - LinearRegression (features aumentadas)", out_path=out_png_lr_aug)

    # Opcional: Resumen statsmodels OLS con las features transformadas (solo baseline)
    try:
        pipe: Pipeline = results_lr["pipeline"]  # type: ignore
        X_train_df, _ = build_features(train_df, target_preference=[target_col], feature_set="baseline")
        X_transformed = pipe.named_steps["preprocess"].fit_transform(X_train_df)
        X_sm = sm.add_constant(X_transformed, has_constant="add")
        model_sm = sm.OLS(train_df[target_col].astype(float).to_numpy(), X_sm)
        results_sm = model_sm.fit()
        print("\n=== Resumen OLS (statsmodels) sobre features transformadas (baseline) ===")
        print(results_sm.summary())
    except Exception as e:
        print(f"Aviso: no se pudo generar el resumen de statsmodels OLS: {e}")


if __name__ == "__main__":
    main()


