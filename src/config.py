"""
Configuración central del proyecto: rutas, constantes y parámetros.
Usa pathlib para compatibilidad Windows / Linux.
"""

from pathlib import Path
import logging

# ──────────────────────────────────────────────
# Rutas base (relativas a la raíz del proyecto)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Datos de entrada
DATA_DIR = PROJECT_ROOT  # los CSV / JSON están en la raíz del proyecto
AGEB_V3_PATH = DATA_DIR / "zonas_ageb_clean_v3.json"

# Outputs organizados
OUTPUTS_DIR        = PROJECT_ROOT / "outputs"
CHARTS_DIR         = OUTPUTS_DIR / "charts"
MAPS_DIR           = OUTPUTS_DIR / "maps"
MODELO_DIR         = OUTPUTS_DIR / "modelo"
MODELO_CHARTS_DIR  = MODELO_DIR / "charts"
MODELO_MAPS_DIR    = MODELO_DIR / "maps"
MODELO_COMP_DIR    = MODELO_DIR / "comparativas"
MODELO_INSPECTOR_DIR = MODELO_DIR / "modelo_inspector"
AGBS_REGEN_DIR     = OUTPUTS_DIR / "agbs_regeneradas"

ALL_OUTPUT_DIRS = [
    CHARTS_DIR, MAPS_DIR,
    MODELO_CHARTS_DIR, MODELO_MAPS_DIR, MODELO_COMP_DIR,
    MODELO_INSPECTOR_DIR,
    AGBS_REGEN_DIR,
]

# ──────────────────────────────────────────────
# Features para el modelo supervisado
# ──────────────────────────────────────────────
FEATURE_COLS_RAW = [
    "riesgo_sismo",
    "pct_afectacion_sismo",
    "riesgo_inundacion",
    "severidad_inundacion",
    "pct_afectacion_inundacion",
    "riesgo_laderas",
    "pct_afectacion_laderas",
    "fracturas_count",
    "fracturas_longitud_m",
    "tipo_suelo",
    "pob_total",
    "imu_2020",
    "area_total",
]

TARGET_COL = "ruse_emergencias"
TARGET_BINARY = "alto_riesgo"       # 1 si > percentil 75
TARGET_PERCENTILE = 75

# ──────────────────────────────────────────────
# Parámetros del modelo
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
METRIC_TARGETS = {
    "accuracy": 0.80,
    "auc_roc": 0.85,
}

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging(level: int = LOG_LEVEL) -> None:
    """Configura logging global del proyecto."""
    logging.basicConfig(format=LOG_FORMAT, level=level)


def ensure_output_dirs() -> None:
    """Crea todos los directorios de salida si no existen."""
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
