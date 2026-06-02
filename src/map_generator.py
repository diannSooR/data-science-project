"""
Generador de mapas interactivos (Folium) y estáticos (Matplotlib + GeoPandas).

Tipos de mapa:
  Interactivos → outputs/modelo/maps/
    - Predicción de riesgo, probabilidades, importancia espacial, comparativo
  Estáticos → outputs/maps/
    - Segmentación, grid múltiple, hexbin heatmap, errores FP/FN, incertidumbre
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import branca.colormap as cm

from src.config import (
    AGEB_V3_PATH, MODELO_DIR, MODELO_MAPS_DIR, MAPS_DIR,
    AGBS_REGEN_DIR,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Carga y preparación de datos
# ──────────────────────────────────────────────────────

def load_geodata(
    geojson_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Carga GeoJSON de AGEBs y fusiona con predicciones del CSV procesado.

    Returns
    -------
    GeoDataFrame con geometría + todas las columnas del CSV
    """
    geojson_path = geojson_path or AGEB_V3_PATH
    csv_path = csv_path or (AGBS_REGEN_DIR / "ageb_v3_procesadas.csv")

    logger.info(f"Cargando GeoJSON: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)

    if csv_path.exists():
        logger.info(f"Fusionando con CSV: {csv_path}")
        df_csv = pd.read_csv(csv_path)
        # Usar CVEGEO como llave de merge
        if "CVEGEO" in gdf.columns and "CVEGEO" in df_csv.columns:
            # Evitar columnas duplicadas
            cols_to_keep = [c for c in df_csv.columns if c not in gdf.columns or c == "CVEGEO"]
            gdf = gdf.merge(df_csv[cols_to_keep], on="CVEGEO", how="left")
        logger.info(f"GeoDataFrame resultante: {gdf.shape}")
    else:
        logger.warning(f"CSV no encontrado: {csv_path}")

    return gdf


def _add_predictions(gdf: gpd.GeoDataFrame, model=None, feature_names=None) -> gpd.GeoDataFrame:
    """Agrega columnas de predicción si el modelo está disponible."""
    if model is None:
        return gdf

    if feature_names is None:
        from src.preprocessing import get_feature_names
        feature_names = get_feature_names()

    missing = [c for c in feature_names if c not in gdf.columns]
    if missing:
        logger.warning(f"Faltan features para predicción: {missing[:5]}...")
        return gdf

    X = gdf[feature_names].values
    try:
        probas = model.predict_proba(X)[:, 1]
        gdf = gdf.copy()
        gdf["pred_proba"] = probas
        gdf["pred_class"] = (probas >= 0.26).astype(int)  # umbral óptimo
        logger.info("Predicciones agregadas al GeoDataFrame")
    except Exception as e:
        logger.error(f"Error al predecir: {e}")

    return gdf


def _get_cdmx_center() -> Tuple[float, float]:
    """Centro aproximado de CDMX."""
    return (19.36, -99.14)


def _ensure_dirs():
    MODELO_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────
# MAPAS INTERACTIVOS (Folium)
# ──────────────────────────────────────────────────────

def mapa_riesgo_prediccion(gdf: gpd.GeoDataFrame) -> Path:
    """
    Mapa interactivo de predicción de riesgo (alto_riesgo o pred_class).
    Colores: verde (bajo riesgo) → rojo (alto riesgo).
    """
    _ensure_dirs()
    center = _get_cdmx_center()
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    col = "pred_class" if "pred_class" in gdf.columns else "alto_riesgo"
    if col not in gdf.columns:
        logger.warning(f"Columna '{col}' no encontrada, omitiendo mapa de predicción")
        return None

    def style_fn(feature):
        val = feature["properties"].get(col, 0)
        color = "#e74c3c" if val == 1 else "#27ae60"
        return {
            "fillColor": color,
            "color": "#333",
            "weight": 0.5,
            "fillOpacity": 0.65,
        }

    folium.GeoJson(
        gdf.to_json(),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["CVEGEO", col, "riesgo_general"],
            aliases=["AGEB:", "Pred. Riesgo:", "Riesgo Gral:"],
        ),
        name="Predicción de Riesgo",
    ).add_to(m)

    # Leyenda
    _add_legend(m, "Predicción de Riesgo",
                {"Alto riesgo": "#e74c3c", "Bajo riesgo": "#27ae60"})

    folium.LayerControl().add_to(m)
    path = MODELO_MAPS_DIR / "mapa_riesgo_prediccion.html"
    m.save(str(path))
    logger.info(f"Mapa interactivo guardado: {path}")
    return path


def mapa_probabilidad(gdf: gpd.GeoDataFrame) -> Path:
    """
    Mapa interactivo de probabilidades de riesgo (gradiente continuo).
    """
    _ensure_dirs()

    col = "pred_proba"
    if col not in gdf.columns:
        # Intentar con indice_riesgo_compuesto como proxy
        col = "indice_riesgo_compuesto"
        if col not in gdf.columns:
            logger.warning("No hay columna de probabilidad disponible")
            return None

    center = _get_cdmx_center()
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    colormap = cm.LinearColormap(
        colors=["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"],
        vmin=gdf[col].min(),
        vmax=gdf[col].max(),
        caption=f"Probabilidad de riesgo ({col})",
    )

    def style_fn(feature):
        val = feature["properties"].get(col, 0)
        return {
            "fillColor": colormap(val) if val is not None else "#ccc",
            "color": "#555",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    tooltip_fields = ["CVEGEO", col]
    tooltip_aliases = ["AGEB:", "Probabilidad:"]
    if "riesgo_general" in gdf.columns:
        tooltip_fields.append("riesgo_general")
        tooltip_aliases.append("Riesgo Gral:")

    folium.GeoJson(
        gdf.to_json(),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases),
        name="Probabilidad de Riesgo",
    ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    path = MODELO_MAPS_DIR / "mapa_probabilidad.html"
    m.save(str(path))
    logger.info(f"Mapa de probabilidad guardado: {path}")
    return path


def mapa_importancia_espacial(
    gdf: gpd.GeoDataFrame,
    feature_name: str = "indice_riesgo_compuesto",
) -> Path:
    """
    Mapa interactivo de una feature específica sobre el territorio.
    """
    _ensure_dirs()

    if feature_name not in gdf.columns:
        logger.warning(f"Feature '{feature_name}' no encontrada")
        return None

    center = _get_cdmx_center()
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    vals = gdf[feature_name].dropna()
    colormap = cm.LinearColormap(
        colors=["#ecf0f1", "#3498db", "#8e44ad", "#e74c3c"],
        vmin=vals.min(),
        vmax=vals.max(),
        caption=feature_name,
    )

    def style_fn(feature):
        val = feature["properties"].get(feature_name, None)
        return {
            "fillColor": colormap(val) if val is not None else "#ccc",
            "color": "#555",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        gdf.to_json(),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["CVEGEO", feature_name],
            aliases=["AGEB:", f"{feature_name}:"],
        ),
        name=feature_name,
    ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    safe_name = feature_name.replace(" ", "_")
    path = MODELO_MAPS_DIR / f"mapa_feature_{safe_name}.html"
    m.save(str(path))
    logger.info(f"Mapa de feature guardado: {path}")
    return path


def mapa_comparativo(gdf: gpd.GeoDataFrame) -> Path:
    """
    Mapa interactivo comparativo con múltiples capas (riesgo, probabilidad, general).
    """
    _ensure_dirs()
    center = _get_cdmx_center()
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Capa 1: Riesgo general (1-5)
    if "riesgo_general" in gdf.columns:
        colors_gen = {1: "#27ae60", 2: "#2ecc71", 3: "#f1c40f", 4: "#e67e22", 5: "#e74c3c"}
        fg1 = folium.FeatureGroup(name="Riesgo General (1-5)", show=True)
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda f: {
                "fillColor": colors_gen.get(f["properties"].get("riesgo_general", 1), "#ccc"),
                "color": "#555", "weight": 0.3, "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["CVEGEO", "riesgo_general"],
                aliases=["AGEB:", "Riesgo General:"],
            ),
        ).add_to(fg1)
        fg1.add_to(m)

    # Capa 2: Riesgo sísmico
    if "riesgo_sismo" in gdf.columns:
        colors_sismo = {1: "#dff9fb", 2: "#c7ecee", 3: "#f1c40f", 4: "#e67e22", 5: "#e74c3c"}
        fg2 = folium.FeatureGroup(name="Riesgo Sísmico", show=False)
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda f: {
                "fillColor": colors_sismo.get(f["properties"].get("riesgo_sismo", 1), "#ccc"),
                "color": "#555", "weight": 0.3, "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["CVEGEO", "riesgo_sismo"],
                aliases=["AGEB:", "Riesgo Sísmico:"],
            ),
        ).add_to(fg2)
        fg2.add_to(m)

    # Capa 3: Riesgo inundación
    if "riesgo_inundacion" in gdf.columns:
        colors_inun = {1: "#dff9fb", 2: "#74b9ff", 3: "#0984e3", 4: "#6c5ce7", 5: "#2d3436"}
        fg3 = folium.FeatureGroup(name="Riesgo Inundación", show=False)
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda f: {
                "fillColor": colors_inun.get(f["properties"].get("riesgo_inundacion", 1), "#ccc"),
                "color": "#555", "weight": 0.3, "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["CVEGEO", "riesgo_inundacion"],
                aliases=["AGEB:", "Riesgo Inundación:"],
            ),
        ).add_to(fg3)
        fg3.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    path = MODELO_MAPS_DIR / "mapa_comparativo.html"
    m.save(str(path))
    logger.info(f"Mapa comparativo guardado: {path}")
    return path


def _add_legend(m, title: str, items: dict):
    """Agrega una leyenda HTML al mapa Folium."""
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
         background: white; padding: 12px 16px; border-radius: 8px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;">
    <strong>{title}</strong><br>
    """
    for label, color in items.items():
        legend_html += f'<i style="background:{color}; width:14px; height:14px; display:inline-block; margin-right:6px; border-radius:2px;"></i> {label}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))


# ──────────────────────────────────────────────────────
# MAPAS ESTÁTICOS (Matplotlib + GeoPandas)
# ──────────────────────────────────────────────────────

def _base_fig(gdf, figsize=(14, 12)):
    """Crea figura base con fondo de mapa."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")
    return fig, ax


def _add_basemap(ax, crs="EPSG:4326"):
    """Agrega basemap con contextily si está disponible."""
    try:
        import contextily as ctx
        # Reproyectar a web mercator para basemap
        ax_crs = ax.get_xlim()  # dummy check
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
    except Exception as e:
        logger.debug(f"No se pudo agregar basemap: {e}")


def mapa_segmentacion(gdf: gpd.GeoDataFrame) -> Path:
    """
    Mapa estático de segmentación de riesgo por riesgo_general (1-5).
    """
    _ensure_dirs()

    col = "riesgo_general"
    if col not in gdf.columns:
        logger.warning(f"Columna '{col}' no encontrada")
        return None

    fig, ax = _base_fig(gdf)

    cmap = mcolors.ListedColormap(["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    gdf.plot(column=col, ax=ax, cmap=cmap, norm=norm,
             edgecolor="#666", linewidth=0.2, legend=False)

    # Leyenda manual
    labels = ["1 - Muy bajo", "2 - Bajo", "3 - Medio", "4 - Alto", "5 - Muy alto"]
    colors_list = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=12, label=l)
               for c, l in zip(colors_list, labels)]
    ax.legend(handles=handles, title="Nivel de Riesgo", loc="lower left",
              fontsize=10, title_fontsize=11, framealpha=0.9)

    ax.set_title("Segmentación de Riesgo General — AGEBs CDMX", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    _add_basemap(ax)

    path = MAPS_DIR / "mapa_segmentacion_riesgo.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Mapa de segmentación guardado: {path}")
    return path


def mapa_grid_multiple(gdf: gpd.GeoDataFrame) -> Path:
    """
    Grid 2×2 de mapas: sismo, inundación, laderas, riesgo general.
    """
    _ensure_dirs()

    cols = ["riesgo_sismo", "riesgo_inundacion", "riesgo_laderas", "riesgo_general"]
    titles = ["Riesgo Sísmico", "Riesgo Inundación", "Riesgo Laderas", "Riesgo General"]
    available = [(c, t) for c, t in zip(cols, titles) if c in gdf.columns]

    if not available:
        logger.warning("No se encontraron columnas de riesgo para el grid")
        return None

    n = len(available)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 8))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    cmap = plt.cm.RdYlGn_r

    for idx, (col, title) in enumerate(available):
        row, c = divmod(idx, ncols)
        ax = axes[row, c]
        gdf.plot(column=col, ax=ax, cmap=cmap, edgecolor="#888", linewidth=0.15,
                 legend=True, legend_kwds={"shrink": 0.6, "label": col})
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_axis_off()

    # Ocultar ejes sobrantes
    for idx in range(len(available), nrows * ncols):
        row, c = divmod(idx, ncols)
        axes[row, c].set_visible(False)

    fig.suptitle("Mapas de Riesgo por Categoría — AGEBs CDMX", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = MAPS_DIR / "mapa_grid_riesgos.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Grid de mapas guardado: {path}")
    return path


def mapa_hexbin_heatmap(gdf: gpd.GeoDataFrame) -> Path:
    """
    Heatmap hexbin basado en el índice de riesgo compuesto.
    """
    _ensure_dirs()

    col = "indice_riesgo_compuesto"
    if col not in gdf.columns:
        col = "riesgo_general"
    if col not in gdf.columns:
        logger.warning("No hay columna para hexbin")
        return None

    # Extraer centroides (reproyectar para cálculo correcto)
    gdf_proj = gdf.to_crs(epsg=6372)  # CRS proyectado México
    centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    x = centroids.x.values
    y = centroids.y.values
    vals = gdf[col].values

    fig, ax = plt.subplots(figsize=(14, 12))
    hb = ax.hexbin(x, y, C=vals, gridsize=40, cmap="RdYlGn_r",
                   reduce_C_function=np.mean, mincnt=1)

    cb = fig.colorbar(hb, ax=ax, shrink=0.7, label=col)
    ax.set_title(f"Heatmap Hexbin — {col}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")

    path = MAPS_DIR / "mapa_hexbin_riesgo.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Mapa hexbin guardado: {path}")
    return path


def mapa_errores(gdf: gpd.GeoDataFrame) -> Path:
    """
    Mapa de errores de predicción: Falsos Positivos y Falsos Negativos.
    Requiere columnas 'pred_class' y 'alto_riesgo'.
    """
    _ensure_dirs()

    pred_col = "pred_class"
    true_col = "alto_riesgo"

    if pred_col not in gdf.columns or true_col not in gdf.columns:
        logger.warning("Faltan columnas para mapa de errores (pred_class, alto_riesgo)")
        return None

    gdf = gdf.copy()
    gdf["error_type"] = "Correcto"
    fp_mask = (gdf[pred_col] == 1) & (gdf[true_col] == 0)
    fn_mask = (gdf[pred_col] == 0) & (gdf[true_col] == 1)
    gdf.loc[fp_mask, "error_type"] = "Falso Positivo"
    gdf.loc[fn_mask, "error_type"] = "Falso Negativo"

    fig, ax = _base_fig(gdf)

    color_map = {"Correcto": "#bdc3c7", "Falso Positivo": "#e74c3c", "Falso Negativo": "#3498db"}

    for etype, color in color_map.items():
        subset = gdf[gdf["error_type"] == etype]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="#666", linewidth=0.2,
                        label=f"{etype} (n={len(subset)})")

    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=12,
                       label=f"{l} ({len(gdf[gdf['error_type'] == l])})")
               for l, c in color_map.items()]
    ax.legend(handles=handles, title="Tipo", loc="lower left", fontsize=10, framealpha=0.9)

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    ax.set_title(f"Mapa de Errores — FP={n_fp}, FN={n_fn}", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    _add_basemap(ax)

    path = MAPS_DIR / "mapa_errores_fp_fn.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Mapa de errores guardado: {path}")
    return path


def mapa_incertidumbre(gdf: gpd.GeoDataFrame) -> Path:
    """
    Mapa de incertidumbre: zonas donde la probabilidad está cerca de 0.5
    (mayor incertidumbre en la predicción).
    """
    _ensure_dirs()

    col = "pred_proba"
    if col not in gdf.columns:
        col = "indice_riesgo_compuesto"
    if col not in gdf.columns:
        logger.warning("No hay columna de probabilidad para mapa de incertidumbre")
        return None

    gdf = gdf.copy()
    # Incertidumbre = qué tan cerca está de 0.5
    gdf["incertidumbre"] = 1 - 2 * np.abs(gdf[col] - 0.5)
    gdf["incertidumbre"] = gdf["incertidumbre"].clip(0, 1)

    fig, ax = _base_fig(gdf)

    gdf.plot(column="incertidumbre", ax=ax, cmap="YlOrRd",
             edgecolor="#888", linewidth=0.15,
             legend=True, legend_kwds={"shrink": 0.6, "label": "Incertidumbre"})

    ax.set_title("Mapa de Incertidumbre del Modelo", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    _add_basemap(ax)

    path = MAPS_DIR / "mapa_incertidumbre.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Mapa de incertidumbre guardado: {path}")
    return path


# ──────────────────────────────────────────────────────
# Pipeline completo
# ──────────────────────────────────────────────────────

def generate_all_maps(
    model=None,
    feature_names=None,
    geojson_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    """
    Genera todos los mapas (interactivos y estáticos).

    Parameters
    ----------
    model : modelo entrenado (opcional, para predicciones)
    feature_names : nombres de features (opcional)
    geojson_path : ruta al GeoJSON de AGEBs
    csv_path : ruta al CSV con datos procesados

    Returns
    -------
    dict con rutas de archivos generados
    """
    _ensure_dirs()

    logger.info("=" * 50)
    logger.info("  GENERACIÓN DE MAPAS")
    logger.info("=" * 50)

    # Cargar datos
    gdf = load_geodata(geojson_path, csv_path)

    # Agregar predicciones si hay modelo
    if model is not None:
        gdf = _add_predictions(gdf, model, feature_names)

    paths = {}

    # ── Mapas interactivos ──
    logger.info("\n--- Mapas interactivos (Folium) ---")
    paths["riesgo_prediccion"] = mapa_riesgo_prediccion(gdf)
    paths["probabilidad"] = mapa_probabilidad(gdf)
    paths["importancia_espacial"] = mapa_importancia_espacial(gdf, "indice_riesgo_compuesto")
    paths["comparativo"] = mapa_comparativo(gdf)

    # ── Mapas estáticos ──
    logger.info("\n--- Mapas estáticos (Matplotlib) ---")
    paths["segmentacion"] = mapa_segmentacion(gdf)
    paths["grid_multiple"] = mapa_grid_multiple(gdf)
    paths["hexbin_heatmap"] = mapa_hexbin_heatmap(gdf)
    paths["errores"] = mapa_errores(gdf)
    paths["incertidumbre"] = mapa_incertidumbre(gdf)

    generated = {k: v for k, v in paths.items() if v is not None}
    logger.info(f"\n✅ Mapas generados: {len(generated)}/{len(paths)}")
    for name, p in generated.items():
        logger.info(f"  [{name}] → {p}")

    return paths


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()

    # Intentar cargar el modelo guardado
    model = None
    model_path = MODELO_DIR / "modelo_final.joblib"
    if model_path.exists():
        import joblib
        logger.info(f"Cargando modelo: {model_path}")
        model = joblib.load(model_path)

    paths = generate_all_maps(model=model)

    print("\n" + "=" * 50)
    print("  MAPAS GENERADOS")
    print("=" * 50)
    for name, p in paths.items():
        status = "✅" if p else "⏭ (omitido)"
        print(f"  {status} {name}: {p or 'N/A'}")
