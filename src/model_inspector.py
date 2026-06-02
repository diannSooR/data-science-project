"""
Model Inspector — Carga modelos .joblib y genera reportes detallados.

Soporta AveragingEnsemble (custom) y modelos sklearn / XGBoost.
Genera reportes en HTML, JSON, Markdown y opcionalmente PDF.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELO_DIR,
    MODELO_INSPECTOR_DIR,
    RANDOM_STATE,
)
from src.preprocessing import get_feature_names

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _safe_params(model) -> dict:
    """Extrae hiperparámetros de un modelo de forma segura."""
    if hasattr(model, "get_params"):
        try:
            return {k: _serializable(v) for k, v in model.get_params(deep=False).items()}
        except Exception:
            pass
    return {}


def _serializable(val: Any) -> Any:
    """Convierte valor a tipo serializable JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def _model_type_name(model) -> str:
    """Devuelve el nombre legible del tipo de modelo."""
    cls = type(model).__name__
    module = type(model).__module__ or ""
    if "xgboost" in module.lower():
        return f"XGBoost ({cls})"
    if "sklearn" in module.lower() or "gradient" in cls.lower():
        return f"Scikit-learn ({cls})"
    return cls


# ──────────────────────────────────────────────────────
# Extracción de información del modelo
# ──────────────────────────────────────────────────────

def extract_model_info(
    model,
    feature_names: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Extrae toda la información relevante de un modelo cargado.

    Parameters
    ----------
    model : objeto modelo (AveragingEnsemble, sklearn, xgboost)
    feature_names : lista de nombres de features usadas
    metrics : dict con métricas de evaluación (accuracy, auc_roc, etc.)
    model_path : ruta del archivo .joblib

    Returns
    -------
    dict con toda la información del modelo
    """
    if feature_names is None:
        feature_names = get_feature_names()

    info: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path) if model_path else None,
    }

    # ── Detectar tipo de modelo ──
    cls_name = type(model).__name__

    if cls_name == "AveragingEnsemble":
        info["model_type"] = "AveragingEnsemble"
        info["description"] = "Ensemble que promedia probabilidades de múltiples modelos"
        info["n_sub_models"] = len(model.models)
        info["sub_models"] = []
        for i, sub in enumerate(model.models):
            sub_info = {
                "index": i,
                "type": _model_type_name(sub),
                "class": type(sub).__name__,
                "hyperparameters": _safe_params(sub),
            }
            if hasattr(sub, "feature_importances_"):
                imp = sub.feature_importances_
                sub_info["feature_importances"] = dict(zip(feature_names, imp.tolist()))
            info["sub_models"].append(sub_info)

        # Importancia promedio del ensemble
        if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
            imp = model.feature_importances_
            imp_dict = dict(zip(feature_names, imp.tolist()))
            info["feature_importances"] = dict(
                sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            )
    else:
        info["model_type"] = _model_type_name(model)
        info["class"] = cls_name
        info["hyperparameters"] = _safe_params(model)
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp_dict = dict(zip(feature_names, imp.tolist()))
            info["feature_importances"] = dict(
                sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            )

    # ── Features ──
    info["n_features"] = len(feature_names)
    info["feature_names"] = feature_names

    # ── Métricas ──
    if metrics:
        info["metrics"] = {
            k: _serializable(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }

    return info


def load_and_inspect(
    model_filename: str = "modelo_final.joblib",
    metrics_filename: str = "metricas_modelos.csv",
) -> Dict[str, Any]:
    """
    Carga modelo y métricas desde disco y extrae toda la información.

    Returns
    -------
    dict con información completa del modelo
    """
    model_path = MODELO_DIR / model_filename
    logger.info(f"Cargando modelo desde {model_path}")
    model = joblib.load(model_path)

    # Intentar cargar métricas
    metrics = None
    from src.config import AGBS_REGEN_DIR
    metrics_path = AGBS_REGEN_DIR / metrics_filename
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        # Buscar la fila del ensemble o la mejor
        for search in ["Ensemble", "ensemble", "Averaging"]:
            match = df_metrics[df_metrics["modelo"].str.contains(search, case=False, na=False)]
            if not match.empty:
                row = match.iloc[0]
                metrics = {k: v for k, v in row.to_dict().items() if k != "modelo"}
                break
        if metrics is None and not df_metrics.empty:
            metrics = {
                k: v for k, v in df_metrics.iloc[0].to_dict().items() if k != "modelo"
            }

    feature_names = get_feature_names()
    info = extract_model_info(model, feature_names, metrics, model_path)
    return info


# ──────────────────────────────────────────────────────
# Generación de reportes
# ──────────────────────────────────────────────────────

def _ensure_dir():
    MODELO_INSPECTOR_DIR.mkdir(parents=True, exist_ok=True)


def generate_json_report(info: Dict[str, Any]) -> Path:
    """Genera reporte JSON."""
    _ensure_dir()
    path = MODELO_INSPECTOR_DIR / "reporte_modelo.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Reporte JSON guardado: {path}")
    return path


def generate_markdown_report(info: Dict[str, Any]) -> Path:
    """Genera reporte en Markdown."""
    _ensure_dir()
    lines = [
        "# 🔍 Reporte de Inspección del Modelo",
        "",
        f"**Fecha:** {info.get('timestamp', 'N/A')}",
        f"**Archivo:** `{info.get('model_path', 'N/A')}`",
        "",
        "---",
        "",
        "## Tipo de Modelo",
        "",
        f"**{info.get('model_type', 'N/A')}**",
        "",
    ]

    if info.get("description"):
        lines.append(f"> {info['description']}")
        lines.append("")

    # Sub-modelos (Ensemble)
    if info.get("sub_models"):
        lines.append(f"### Sub-modelos ({info.get('n_sub_models', '?')})")
        lines.append("")
        for sub in info["sub_models"]:
            lines.append(f"#### {sub['index'] + 1}. {sub['type']}")
            lines.append("")
            if sub.get("hyperparameters"):
                lines.append("| Parámetro | Valor |")
                lines.append("|-----------|-------|")
                for k, v in sub["hyperparameters"].items():
                    lines.append(f"| `{k}` | `{v}` |")
                lines.append("")

    # Hyperparams (modelo individual)
    elif info.get("hyperparameters"):
        lines.append("## Hiperparámetros")
        lines.append("")
        lines.append("| Parámetro | Valor |")
        lines.append("|-----------|-------|")
        for k, v in info["hyperparameters"].items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    # Features
    lines.append(f"## Features ({info.get('n_features', '?')})")
    lines.append("")
    for i, f_name in enumerate(info.get("feature_names", []), 1):
        lines.append(f"{i}. `{f_name}`")
    lines.append("")

    # Feature importance
    if info.get("feature_importances"):
        lines.append("## Importancia de Features")
        lines.append("")
        lines.append("| # | Feature | Importancia |")
        lines.append("|---|---------|-------------|")
        for i, (feat, imp) in enumerate(info["feature_importances"].items(), 1):
            bar = "█" * int(imp * 100)
            lines.append(f"| {i} | `{feat}` | {imp:.4f} {bar} |")
        lines.append("")

    # Métricas
    if info.get("metrics"):
        lines.append("## Métricas de Evaluación")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        for k, v in info["metrics"].items():
            emoji = ""
            if k == "accuracy" and v >= 0.80:
                emoji = " ✅"
            elif k == "auc_roc" and v >= 0.85:
                emoji = " ✅"
            lines.append(f"| **{k}** | {v:.4f}{emoji} |")
        lines.append("")

    path = MODELO_INSPECTOR_DIR / "reporte_modelo.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Reporte Markdown guardado: {path}")
    return path


def generate_html_report(info: Dict[str, Any]) -> Path:
    """Genera reporte HTML con estilos."""
    _ensure_dir()

    # CSS
    css = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f8f9fa; color: #333; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #ecf0f1; }
        th { background: #3498db; color: white; }
        tr:nth-child(even) { background: #f2f6fc; }
        .metric-good { color: #27ae60; font-weight: bold; }
        .metric-warn { color: #e67e22; font-weight: bold; }
        .bar { display: inline-block; height: 14px; background: linear-gradient(90deg, #3498db, #2ecc71); border-radius: 3px; margin-left: 8px; vertical-align: middle; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }
        .badge-primary { background: #3498db; color: white; }
        .badge-success { background: #27ae60; color: white; }
        code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
        .footer { text-align: center; color: #95a5a6; margin-top: 40px; font-size: 0.85em; }
    </style>
    """

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='es'><head><meta charset='UTF-8'>",
        "<title>Reporte de Inspección del Modelo</title>",
        css,
        "</head><body>",
        "<h1>🔍 Reporte de Inspección del Modelo</h1>",
        f"<p><strong>Fecha:</strong> {info.get('timestamp', 'N/A')}</p>",
        f"<p><strong>Archivo:</strong> <code>{info.get('model_path', 'N/A')}</code></p>",
    ]

    # Tipo de modelo
    html_parts.append("<div class='card'>")
    html_parts.append(f"<h2>Tipo de Modelo</h2>")
    html_parts.append(f"<p><span class='badge badge-primary'>{info.get('model_type', 'N/A')}</span></p>")
    if info.get("description"):
        html_parts.append(f"<p><em>{info['description']}</em></p>")
    html_parts.append("</div>")

    # Sub-modelos
    if info.get("sub_models"):
        html_parts.append("<div class='card'>")
        html_parts.append(f"<h2>Sub-modelos ({info.get('n_sub_models', '?')})</h2>")
        for sub in info["sub_models"]:
            html_parts.append(f"<h3>{sub['index'] + 1}. {sub['type']}</h3>")
            if sub.get("hyperparameters"):
                html_parts.append("<table><tr><th>Parámetro</th><th>Valor</th></tr>")
                for k, v in sub["hyperparameters"].items():
                    html_parts.append(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>")
                html_parts.append("</table>")
        html_parts.append("</div>")
    elif info.get("hyperparameters"):
        html_parts.append("<div class='card'>")
        html_parts.append("<h2>Hiperparámetros</h2>")
        html_parts.append("<table><tr><th>Parámetro</th><th>Valor</th></tr>")
        for k, v in info["hyperparameters"].items():
            html_parts.append(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>")
        html_parts.append("</table></div>")

    # Features
    html_parts.append("<div class='card'>")
    html_parts.append(f"<h2>Features ({info.get('n_features', '?')})</h2>")
    html_parts.append("<ol>")
    for f_name in info.get("feature_names", []):
        html_parts.append(f"<li><code>{f_name}</code></li>")
    html_parts.append("</ol></div>")

    # Feature importance
    if info.get("feature_importances"):
        html_parts.append("<div class='card'>")
        html_parts.append("<h2>Importancia de Features</h2>")
        html_parts.append("<table><tr><th>#</th><th>Feature</th><th>Importancia</th><th>Barra</th></tr>")
        max_imp = max(info["feature_importances"].values()) if info["feature_importances"] else 1
        for i, (feat, imp) in enumerate(info["feature_importances"].items(), 1):
            width = int((imp / max_imp) * 200)
            html_parts.append(
                f"<tr><td>{i}</td><td><code>{feat}</code></td>"
                f"<td>{imp:.4f}</td>"
                f"<td><span class='bar' style='width:{width}px'></span></td></tr>"
            )
        html_parts.append("</table></div>")

    # Métricas
    if info.get("metrics"):
        html_parts.append("<div class='card'>")
        html_parts.append("<h2>Métricas de Evaluación</h2>")
        html_parts.append("<table><tr><th>Métrica</th><th>Valor</th><th>Estado</th></tr>")
        for k, v in info["metrics"].items():
            css_class = ""
            status = ""
            if k == "accuracy":
                css_class = "metric-good" if v >= 0.80 else "metric-warn"
                status = "✅ ≥ 0.80" if v >= 0.80 else "⚠ < 0.80"
            elif k == "auc_roc":
                css_class = "metric-good" if v >= 0.85 else "metric-warn"
                status = "✅ ≥ 0.85" if v >= 0.85 else "⚠ < 0.85"
            else:
                css_class = "metric-good" if v >= 0.75 else ""
            html_parts.append(
                f"<tr><td><strong>{k}</strong></td>"
                f"<td class='{css_class}'>{v:.4f}</td>"
                f"<td>{status}</td></tr>"
            )
        html_parts.append("</table></div>")

    html_parts.append(f"<div class='footer'>Generado el {info.get('timestamp', '')} — Análisis de Riesgos CDMX</div>")
    html_parts.append("</body></html>")

    path = MODELO_INSPECTOR_DIR / "reporte_modelo.html"
    path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info(f"Reporte HTML guardado: {path}")
    return path


def generate_all_reports(
    model_filename: str = "modelo_final.joblib",
    metrics_filename: str = "metricas_modelos.csv",
) -> Dict[str, Path]:
    """
    Pipeline completo: carga modelo → extrae info → genera todos los reportes.

    Returns
    -------
    dict con rutas de los archivos generados
    """
    info = load_and_inspect(model_filename, metrics_filename)

    paths = {
        "json": generate_json_report(info),
        "markdown": generate_markdown_report(info),
        "html": generate_html_report(info),
    }

    logger.info(f"✅ Todos los reportes generados en {MODELO_INSPECTOR_DIR}")
    return paths


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()
    paths = generate_all_reports()
    print("\nArchivos generados:")
    for fmt, p in paths.items():
        print(f"  [{fmt.upper()}] {p}")
