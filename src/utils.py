"""
Funciones auxiliares generales del proyecto.
"""

from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


def save_path(directory: Path, filename: str) -> Path:
    """
    Construye un path de salida, creando el directorio si no existe.

    Parameters
    ----------
    directory : Path
        Carpeta destino.
    filename : str
        Nombre del archivo (con extensión).

    Returns
    -------
    Path
        Ruta completa al archivo.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def validate_dataframe(df, expected_rows: int = None, required_cols: list = None,
                       name: str = "DataFrame") -> bool:
    """
    Valida que un DataFrame cumpla requisitos mínimos.
    
    Returns True si pasa; lanza ValueError si falla.
    """
    if df is None or df.empty:
        raise ValueError(f"{name} está vacío o es None.")

    if expected_rows is not None and len(df) != expected_rows:
        logger.warning(
            f"{name}: se esperaban {expected_rows} filas, se encontraron {len(df)}."
        )

    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name}: faltan columnas requeridas: {missing}")

    logger.info(f"✓ {name} validado: {df.shape[0]} filas × {df.shape[1]} columnas.")
    return True


def print_metrics_summary(metrics: dict, targets: dict = None) -> None:
    """Imprime un resumen legible de métricas con indicador de cumplimiento."""
    print("\n" + "=" * 55)
    print("  RESUMEN DE MÉTRICAS DEL MODELO")
    print("=" * 55)
    for key, val in metrics.items():
        line = f"  {key:<20s}: {val:.4f}"
        if targets and key in targets:
            ok = "✅" if val >= targets[key] else "❌"
            line += f"  (objetivo ≥ {targets[key]:.2f}) {ok}"
        print(line)
    print("=" * 55 + "\n")
