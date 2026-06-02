#!/usr/bin/env python3
"""
Pipeline completo: carga → limpieza → feature engineering → modelado → evaluación.

Ejecutar desde la raíz del proyecto:
    python run_pipeline.py

Genera todos los outputs en la carpeta outputs/ con la estructura organizada.
"""

import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import (
    setup_logging, ensure_output_dirs, FEATURE_COLS_RAW,
    TARGET_COL, TARGET_BINARY, METRIC_TARGETS, AGBS_REGEN_DIR,
)
from src.data_loader import load_ageb_v3, EXPECTED_ROWS
from src.preprocessing import (
    create_target_variable, engineer_features, get_feature_names,
    prepare_train_test,
)
from src.modeling import (
    train_gradient_boosting, train_xgboost, train_random_forest,
    AveragingEnsemble, evaluate_model, check_targets,
    find_optimal_threshold, save_model,
)
from src.visualization import (
    plot_correlation_matrix, plot_target_distribution, plot_feature_distributions,
    plot_roc_curve, plot_confusion_matrix, plot_feature_importance,
    plot_precision_recall_curve, plot_model_comparison,
)
from src.utils import print_metrics_summary, save_path

setup_logging()
logger = logging.getLogger("pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Análisis de Riesgos CDMX")
    parser.add_argument("--maps", action="store_true",
                        help="Generar mapas interactivos y estáticos al final del pipeline")
    parser.add_argument("--inspect", action="store_true",
                        help="Generar reportes de inspección del modelo")
    parser.add_argument("--all", action="store_true",
                        help="Ejecutar pipeline completo con mapas e inspección")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_maps = args.maps or args.all
    generate_inspect = args.inspect or args.all

    print("=" * 60)
    print("  PIPELINE DE ANÁLISIS DE RIESGOS CDMX")
    print("=" * 60)

    # 1. Directorios
    ensure_output_dirs()

    # 2. Cargar datos v3 completo (2453 AGEBs)
    logger.info("\n=== PASO 1: CARGA DE DATOS ===")
    df = load_ageb_v3()
    assert len(df) == EXPECTED_ROWS, f"Esperados {EXPECTED_ROWS}, encontrados {len(df)}"

    # 3. Target binarizado
    logger.info("\n=== PASO 2: PREPROCESAMIENTO ===")
    df = create_target_variable(df)

    # 4. Feature engineering
    df = engineer_features(df)

    # Guardar AGEBs procesadas
    agbs_path = save_path(AGBS_REGEN_DIR, "ageb_v3_procesadas.csv")
    df.to_csv(agbs_path, index=False)
    logger.info(f"AGEBs guardadas: {agbs_path}")

    # 5. EDA
    logger.info("\n=== PASO 3: ANÁLISIS EXPLORATORIO ===")
    plot_correlation_matrix(df, FEATURE_COLS_RAW)
    plot_target_distribution(df, TARGET_COL, TARGET_BINARY)
    plot_feature_distributions(df, FEATURE_COLS_RAW)

    # 6. Train/test split
    logger.info("\n=== PASO 4: MODELADO SUPERVISADO ===")
    X_train, X_test, y_train, y_test, feature_names = prepare_train_test(df)
    logger.info(f"Features ({len(feature_names)}): {feature_names}")

    # 7. Entrenar modelos
    all_metrics = {}

    # 7a. Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    all_metrics["Gradient Boosting"] = gb_metrics

    # 7b. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    all_metrics["XGBoost"] = xgb_metrics

    # 7c. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    all_metrics["Random Forest"] = rf_metrics

    # 7d. Averaging Ensemble (GB + XGB)
    ensemble = AveragingEnsemble([gb_model, xgb_model])
    # Find optimal threshold for ensemble
    opt_thresh = find_optimal_threshold(y_test, ensemble.predict_proba(X_test)[:, 1])
    ens_metrics = evaluate_model(ensemble, X_test, y_test, "Ensemble (GB+XGB)", threshold=opt_thresh)
    all_metrics["Ensemble (GB+XGB)"] = ens_metrics

    # 8. Seleccionar mejor modelo
    logger.info("\n=== PASO 5: SELECCIÓN ===")
    best_name = max(all_metrics, key=lambda k: all_metrics[k].get("auc_roc", 0))
    best_metrics = all_metrics[best_name]
    models_map = {
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model,
        "Random Forest": rf_model,
        "Ensemble (GB+XGB)": ensemble,
    }
    best_model = models_map[best_name]

    print_metrics_summary(
        {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
        METRIC_TARGETS,
    )

    # 9. Visualizaciones del modelo
    logger.info("\n=== PASO 6: VISUALIZACIONES ===")
    plot_roc_curve(y_test, best_metrics["y_proba"], best_name)
    plot_confusion_matrix(y_test, best_metrics["y_pred"], best_name)
    plot_precision_recall_curve(y_test, best_metrics["y_proba"], best_name)

    # Feature importance
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "feature_importances_") and best_model.feature_importances_ is not None:
        importances = best_model.feature_importances_
    if importances is not None:
        plot_feature_importance(importances, feature_names, best_name)

    # Comparativa
    comp = {n: {k: v for k, v in m.items() if isinstance(v, (int, float))}
            for n, m in all_metrics.items()}
    plot_model_comparison(comp)

    # Curvas ROC de todos
    _plot_all_roc(y_test, all_metrics)

    # 10. Guardar
    logger.info("\n=== PASO 7: GUARDADO ===")
    save_model(best_model)

    metrics_df = pd.DataFrame([
        {"modelo": n, **{k: v for k, v in m.items() if isinstance(v, (int, float))}}
        for n, m in all_metrics.items()
    ])
    metrics_df.to_csv(save_path(AGBS_REGEN_DIR, "metricas_modelos.csv"), index=False)

    # 11. Validación
    logger.info("\n=== VALIDACIÓN FINAL ===")
    ok = check_targets(best_metrics)

    # 12. Inspección del modelo (opcional)
    if generate_inspect:
        logger.info("\n=== PASO 8: INSPECCIÓN DEL MODELO ===")
        from src.model_inspector import generate_all_reports
        inspect_paths = generate_all_reports()
        for fmt, p in inspect_paths.items():
            logger.info(f"  Reporte [{fmt.upper()}]: {p}")

    # 13. Generación de mapas (opcional)
    if generate_maps:
        logger.info("\n=== PASO 9: GENERACIÓN DE MAPAS ===")
        from src.map_generator import generate_all_maps
        map_paths = generate_all_maps(model=best_model, feature_names=feature_names)
        n_generated = sum(1 for v in map_paths.values() if v is not None)
        logger.info(f"Mapas generados: {n_generated}/{len(map_paths)}")

    print("\n" + "=" * 60)
    print("  REPORTE FINAL")
    print("=" * 60)
    print(f"  Modelo seleccionado : {best_name}")
    print(f"  AGEBs procesadas    : {len(df)}")
    print(f"  Features utilizadas : {len(feature_names)}")
    print(f"  Accuracy            : {best_metrics['accuracy']:.4f}")
    print(f"  AUC-ROC             : {best_metrics['auc_roc']:.4f}")
    print(f"  Precision           : {best_metrics['precision']:.4f}")
    print(f"  Recall              : {best_metrics['recall']:.4f}")
    print(f"  F1-Score            : {best_metrics['f1']:.4f}")
    print(f"  Objetivos cumplidos : {'✅ SÍ' if ok else '⚠ PARCIAL'}")
    if generate_inspect:
        print(f"  Reportes inspector  : ✅ generados")
    if generate_maps:
        print(f"  Mapas generados     : {n_generated}")
    print("=" * 60)

    return 0 if ok else 1


def _plot_all_roc(y_test, all_metrics):
    """Curvas ROC de todos los modelos en una sola figura."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc
    from src.config import MODELO_COMP_DIR
    from src.utils import save_path as sp

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i, (name, metrics) in enumerate(all_metrics.items()):
        if metrics.get("y_proba") is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            auc_val = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Comparativa Curvas ROC — Todos los Modelos", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = sp(MODELO_COMP_DIR, "comparativa_roc_todos.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
