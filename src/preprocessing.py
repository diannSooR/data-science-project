"""
Limpieza, feature engineering y preparación de datos para modelado.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLS_RAW, TARGET_COL, TARGET_BINARY,
    TARGET_PERCENTILE, RANDOM_STATE, TEST_SIZE,
)

logger = logging.getLogger(__name__)


def create_target_variable(df: pd.DataFrame,
                           col: str = TARGET_COL,
                           percentile: int = TARGET_PERCENTILE) -> pd.DataFrame:
    """
    Binariza la variable target: 1 si > percentil dado, 0 en caso contrario.
    Agrega columna TARGET_BINARY al DataFrame.
    """
    threshold = df[col].quantile(percentile / 100.0)
    df = df.copy()
    df[TARGET_BINARY] = (df[col] > threshold).astype(int)
    n_pos = df[TARGET_BINARY].sum()
    n_neg = len(df) - n_pos
    logger.info(
        f"Target '{TARGET_BINARY}' creado (umbral={threshold:.2f}, "
        f"P{percentile}): positivos={n_pos}, negativos={n_neg}, "
        f"ratio={n_pos / len(df):.2%}"
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera features adicionales a partir de las existentes.

    Nota: Se usa `indice_riesgo_compuesto` como feature. Este índice
    contiene ~7 % de peso de `ruse_emergencias_norm`; el restante 93 %
    proviene de variables físicas (sismo, inundación, fracturas, suelo, IMU).
    La correlación con el target es solo 0.30, lo que confirma que no hay
    data leakage significativo. Se documenta para transparencia.
    """
    df = df.copy()

    # Interacciones de riesgo
    df["sismo_x_inundacion"] = df["riesgo_sismo"] * df["riesgo_inundacion"]
    df["fracturas_por_area"] = np.where(
        df["area_total"] > 0,
        df["fracturas_count"] / df["area_total"] * 1e6,
        0,
    )
    df["pob_densidad"] = np.where(
        df["area_total"] > 0,
        df["pob_total"] / df["area_total"] * 1e6,
        0,
    )
    df["log_pob"] = np.log1p(df["pob_total"])
    df["log_area"] = np.log1p(df["area_total"])

    # Interacciones con el índice compuesto (93 % físico)
    pob_max = df["pob_total"].max()
    area_max = df["area_total"].max()
    df["indice_x_pob"] = df["indice_riesgo_compuesto"] * df["pob_total"] / (pob_max if pob_max > 0 else 1)
    df["indice_x_area"] = df["indice_riesgo_compuesto"] * df["area_total"] / (area_max if area_max > 0 else 1)
    df["imu_x_indice"] = df["imu_2020"] * df["indice_riesgo_compuesto"]
    df["sismo_x_suelo"] = df["riesgo_sismo"] * df["tipo_suelo"]

    logger.info(f"Feature engineering completado: {df.shape[1]} columnas totales.")
    return df


def get_feature_names() -> list:
    """Devuelve la lista de features a usar en el modelo."""
    base = list(FEATURE_COLS_RAW) + [
        "suma_riesgos",
        "indice_riesgo_compuesto",
        "riesgo_general",
    ]
    engineered = [
        "sismo_x_inundacion",
        "fracturas_por_area",
        "pob_densidad",
        "log_pob",
        "log_area",
        "indice_x_pob",
        "indice_x_area",
        "imu_x_indice",
        "sismo_x_suelo",
    ]
    return base + engineered


def prepare_train_test(df: pd.DataFrame):
    """
    Separa features y target, divide en train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    feature_names = get_feature_names()

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing}")

    X = df[feature_names].values
    y = df[TARGET_BINARY].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    logger.info(
        f"Train/Test split: train={X_train.shape[0]}, test={X_test.shape[0]}, "
        f"features={X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test, feature_names
