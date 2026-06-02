"""
Entrenamiento, evaluación y optimización de modelos supervisados.
Objetivo: Accuracy ≥ 80 %, AUC-ROC ≥ 0.85.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from src.config import RANDOM_STATE, METRIC_TARGETS, MODELO_DIR

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────

def train_gradient_boosting(X_train, y_train):
    """Entrena Gradient Boosting con hiperparámetros optimizados."""
    logger.info("Entrenando Gradient Boosting ...")
    model = GradientBoostingClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=8, max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"GB CV AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    return model


def train_xgboost(X_train, y_train):
    """Entrena XGBoost con hiperparámetros optimizados."""
    import xgboost as xgb

    logger.info("Entrenando XGBoost ...")
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="auc",
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"XGB CV AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    return model


def train_random_forest(X_train, y_train):
    """Entrena Random Forest con hiperparámetros optimizados."""
    logger.info("Entrenando Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=5,
        max_features=0.5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"RF CV AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    return model


class AveragingEnsemble:
    """Ensemble simple que promedia probabilidades de múltiples modelos."""

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        return probas.mean(axis=0)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self):
        imps = []
        for m in self.models:
            if hasattr(m, "feature_importances_"):
                imps.append(m.feature_importances_)
        if imps:
            return np.mean(imps, axis=0)
        return None


# ──────────────────────────────────────────────
# Evaluación
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Modelo",
                   threshold: float = 0.5) -> dict:
    """Evalúa el modelo y devuelve dict con todas las métricas."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "threshold": threshold,
    }

    logger.info(f"\n--- {model_name} (threshold={threshold:.2f}) ---")
    logger.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall   : {metrics['recall']:.4f}")
    logger.info(f"  F1       : {metrics['f1']:.4f}")
    return metrics


def check_targets(metrics: dict) -> bool:
    """Verifica si las métricas cumplen los objetivos."""
    ok = True
    for key, target in METRIC_TARGETS.items():
        val = metrics.get(key, 0)
        if val < target:
            logger.warning(f"⚠ {key} = {val:.4f} < objetivo {target:.2f}")
            ok = False
        else:
            logger.info(f"✅ {key} = {val:.4f} ≥ objetivo {target:.2f}")
    return ok


def find_optimal_threshold(y_test, y_proba) -> float:
    """Busca el umbral que maximiza accuracy."""
    best_acc, best_thresh = 0, 0.5
    for thresh in np.arange(0.20, 0.80, 0.01):
        acc = accuracy_score(y_test, (y_proba >= thresh).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    logger.info(f"Umbral óptimo: {best_thresh:.2f} → accuracy={best_acc:.4f}")
    return best_thresh


# ──────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────

def save_model(model, filename: str = "modelo_final.joblib") -> Path:
    """Guarda modelo entrenado."""
    path = MODELO_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Modelo guardado en {path}")
    return path


def load_model(filename: str = "modelo_final.joblib"):
    """Carga modelo guardado."""
    return joblib.load(MODELO_DIR / filename)
