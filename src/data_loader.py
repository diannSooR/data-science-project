"""
Carga y validación de datos desde zonas_ageb_clean_v3.json.
Fuente única: siempre 2 453 registros (2 431 urbanas + 22 rurales).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd

from src.config import AGEB_V3_PATH, FEATURE_COLS_RAW, TARGET_COL
from src.utils import validate_dataframe

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 2453


def load_ageb_v3(path: Optional[Path] = None, as_geodataframe: bool = False):
    """
    Carga zonas_ageb_clean_v3.json completo (2 453 AGEBs).

    Parameters
    ----------
    path : Path, optional
        Ruta al archivo. Por defecto usa la configuración central.
    as_geodataframe : bool
        Si True devuelve GeoDataFrame con geometría; si False, DataFrame plano.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
    """
    path = path or AGEB_V3_PATH
    logger.info(f"Cargando datos desde {path} ...")

    if as_geodataframe:
        gdf = gpd.read_file(path)
        logger.info(f"GeoDataFrame cargado: {gdf.shape}")
        validate_dataframe(gdf, expected_rows=EXPECTED_ROWS,
                           required_cols=FEATURE_COLS_RAW + [TARGET_COL],
                           name="AGEB_v3_geo")
        return gdf

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = [feat["properties"] for feat in raw["features"]]
    df = pd.DataFrame(records)

    validate_dataframe(df, expected_rows=EXPECTED_ROWS,
                       required_cols=FEATURE_COLS_RAW + [TARGET_COL],
                       name="AGEB_v3")
    return df


def load_raw_csv(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Carga un CSV crudo desde la carpeta de datos."""
    from src.config import DATA_DIR
    data_dir = data_dir or DATA_DIR
    path = data_dir / filename
    logger.info(f"Cargando CSV: {path}")
    df = pd.read_csv(path)
    logger.info(f"  → {df.shape[0]} filas × {df.shape[1]} columnas")
    return df
