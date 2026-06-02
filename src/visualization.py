"""
Generación de gráficos y mapas para el proyecto de riesgos CDMX.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)

from src.config import (
    CHARTS_DIR, MODELO_CHARTS_DIR, MODELO_COMP_DIR, MAPS_DIR,
)
from src.utils import save_path

logger = logging.getLogger(__name__)

# Estilo global
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ──────────────────────────────────────────────
# Gráficos generales (EDA)
# ──────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, cols: list, filename: str = "correlacion_features.png"):
    """Heatmap de correlación entre features."""
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Matriz de Correlación — Features Físicas CDMX", fontsize=14, pad=15)
    path = save_path(CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Correlación guardada: {path}")
    return path


def plot_target_distribution(df: pd.DataFrame, target_col: str, binary_col: str,
                             filename: str = "distribucion_target.png"):
    """Distribución de la variable target original y binarizada."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df[target_col], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(df[target_col].quantile(0.75), color="red", ls="--", lw=2,
                    label=f"P75 = {df[target_col].quantile(0.75):.1f}")
    axes[0].set_title("Distribución de Emergencias (ruse_emergencias)")
    axes[0].set_xlabel("Número de emergencias")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()

    counts = df[binary_col].value_counts().sort_index()
    bars = axes[1].bar(["Bajo Riesgo (0)", "Alto Riesgo (1)"],
                       counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(val), ha="center", fontsize=12, fontweight="bold")
    axes[1].set_title("Distribución de Clases (Target Binarizado)")
    axes[1].set_ylabel("Número de AGEBs")

    fig.suptitle("Variable Target: Emergencias Históricas", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_path(CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Distribución target guardada: {path}")
    return path


def plot_feature_distributions(df: pd.DataFrame, cols: list,
                               filename: str = "distribuciones_features.png"):
    """Histogramas de todas las features."""
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=25, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribución de Features Físicas", fontsize=14, y=1.01)
    fig.tight_layout()
    path = save_path(CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Distribuciones guardadas: {path}")
    return path


# ──────────────────────────────────────────────
# Gráficos del modelo
# ──────────────────────────────────────────────

def plot_roc_curve(y_test, y_proba, model_name: str = "Modelo",
                   filename: str = "curva_roc.png"):
    """Curva ROC con AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title(f"Curva ROC — {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    path = save_path(MODELO_CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Curva ROC guardada: {path}")
    return path


def plot_confusion_matrix(y_test, y_pred, model_name: str = "Modelo",
                          filename: str = "matriz_confusion.png"):
    """Matriz de confusión."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bajo Riesgo", "Alto Riesgo"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=14)

    path = save_path(MODELO_CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Matriz confusión guardada: {path}")
    return path


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                            model_name: str = "Modelo", top_n: int = 20,
                            filename: str = "feature_importance.png"):
    """Importancia de features (barras horizontales)."""
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top_names)))
    ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Importancia")
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    path = save_path(MODELO_CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Feature importance guardada: {path}")
    return path


def plot_precision_recall_curve(y_test, y_proba, model_name: str = "Modelo",
                                filename: str = "curva_precision_recall.png"):
    """Curva Precision-Recall."""
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rec, prec, color="#3498db", lw=2.5, label=model_name)
    ax.fill_between(rec, prec, alpha=0.15, color="#3498db")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curva Precision-Recall — {model_name}", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    path = save_path(MODELO_CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Curva PR guardada: {path}")
    return path


def plot_model_comparison(metrics_dict: dict, filename: str = "comparativa_modelos.png"):
    """
    Gráfico comparativo de múltiples modelos.
    metrics_dict: {nombre_modelo: {accuracy: .., auc_roc: .., f1: ..}}
    """
    models = list(metrics_dict.keys())
    metric_names = ["accuracy", "auc_roc", "precision", "recall", "f1"]
    available_metrics = [m for m in metric_names if m in list(metrics_dict.values())[0]]

    x = np.arange(len(available_metrics))
    width = 0.8 / len(models)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(models):
        vals = [metrics_dict[model_name].get(m, 0) for m in available_metrics]
        ax.bar(x + i * width, vals, width, label=model_name,
               color=colors[i % len(colors)], edgecolor="white")

    # Líneas de objetivo
    ax.axhline(y=0.80, color="gray", ls="--", alpha=0.5, label="Objetivo Accuracy (0.80)")
    ax.axhline(y=0.85, color="gray", ls=":", alpha=0.5, label="Objetivo AUC-ROC (0.85)")

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in available_metrics])
    ax.set_ylabel("Score")
    ax.set_title("Comparativa de Modelos", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    path = save_path(MODELO_COMP_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Comparativa guardada: {path}")
    return path


def plot_learning_curves(model, X_train, y_train, model_name: str = "Modelo",
                         filename: str = "learning_curves.png"):
    """Curvas de aprendizaje."""
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="#e74c3c")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color="#3498db")
    ax.plot(train_sizes, train_mean, "o-", color="#e74c3c", label="Training")
    ax.plot(train_sizes, val_mean, "o-", color="#3498db", label="Validación")
    ax.set_xlabel("Tamaño de entrenamiento")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Curvas de Aprendizaje — {model_name}", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = save_path(MODELO_CHARTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Learning curves guardadas: {path}")
    return path
