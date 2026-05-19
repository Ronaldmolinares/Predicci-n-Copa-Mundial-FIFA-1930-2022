"""
==============================================================================
FASE 3 — MODELADO SUPERVISADO: RANDOM FOREST
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/model_supervised.py

Descripción:
    Módulo de entrenamiento, optimización y evaluación del clasificador
    Random Forest para predecir resultados de partidos de la Copa del Mundo.
    Diseñado para ser importado desde notebooks/fase_3_random_forest.py
    y compatible con FastAPI en Fase 5.

Arquitectura del pipeline completo:
    ┌────────────────────────────────────────────────────────────┐
    │  X_raw (DataFrame)                                         │
    │      ↓                                                     │
    │  rf_preprocessor (ColumnTransformer — ya ajustado Fase 2) │
    │      ↓                                                     │
    │  X_transformed (numpy array)                               │
    │      ↓                                                     │
    │  RandomForestClassifier(class_weight="balanced")           │
    │      ↓                                                     │
    │  y_pred {0=away_win, 1=draw, 2=home_win}                  │
    └────────────────────────────────────────────────────────────┘

Decisiones técnicas:
    - class_weight="balanced": compensa el desbalance home_win >> draw/away_win
    - TimeSeriesSplit: respeta la naturaleza temporal del dataset
    - Pipeline completo serializado: garantiza compatibilidad con FastAPI
    - GridSearchCV con refit=True: el mejor modelo se reentrena con todo el train

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado — UPTC
==============================================================================
"""

import warnings
import json
import os
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES
# =============================================================================

RANDOM_STATE = 42
N_JOBS       = -1

# Clases del target (orden LabelEncoder — alfabético)
CLASS_NAMES = ["away_win", "draw", "home_win"]

# Rutas de salida por defecto
DEFAULT_PATHS = {
    "model":    "models/random_forest_model.pkl",
    "pipeline": "models/random_forest_pipeline.pkl",
    "metrics":  "models/random_forest_metrics.json",
}

# Grid de hiperparámetros para tuning
RF_PARAM_GRID = {
    "classifier__n_estimators":    [100, 300, 500],
    "classifier__max_depth":       [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf":  [1, 2, 4],
    "classifier__max_features":    ["sqrt", "log2"],
}

# Grid reducido para pruebas rápidas
RF_PARAM_GRID_FAST = {
    "classifier__n_estimators":    [100, 300],
    "classifier__max_depth":       [None, 15],
    "classifier__min_samples_split": [2, 10],
    "classifier__min_samples_leaf":  [1, 4],
    "classifier__max_features":    ["sqrt"],
}


# =============================================================================
# CARGA DE ARTEFACTOS DE FASE 2
# =============================================================================

def load_phase2_artifacts(
    xy_path:          str = "models/Xy_phase2.pkl",
    preprocessor_path: str = "models/rf_preprocessor.pkl",
    le_path:          str = "models/label_encoder.pkl",
    metadata_path:    str = "models/preprocessing_metadata.json",
) -> dict:
    """
    Carga todos los artefactos generados en Fase 2.

    Returns
    -------
    dict con claves:
        X_train, y_train, X_test, y_test  → matrices de datos
        preprocessor                       → ColumnTransformer ajustado
        label_encoder                      → LabelEncoder ajustado
        metadata                           → dict con metadata JSON
    """
    print("\n  Cargando artefactos de Fase 2...")

    # Matrices X/y
    X_train, y_train, X_test, y_test = joblib.load(xy_path)
    print(f"    Xy_phase2.pkl        → X_train:{X_train.shape} | X_test:{X_test.shape}")

    # Preprocessor (ya ajustado en train)
    preprocessor = joblib.load(preprocessor_path)
    print(f"    rf_preprocessor.pkl  → {type(preprocessor).__name__} cargado")

    # LabelEncoder
    label_encoder = joblib.load(le_path)
    print(f"    label_encoder.pkl    → clases: {list(label_encoder.classes_)}")

    # Metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"    preprocessing_metadata.json → cargado")

    # Distribución de clases
    print(f"\n    Distribución y_train:")
    for cls_idx, cnt in sorted(pd.Series(y_train).value_counts().items()):
        cls_name = label_encoder.classes_[cls_idx]
        print(f"      {cls_name:<12} ({cls_idx}): {cnt:>4} ({cnt/len(y_train)*100:.1f}%)")

    print(f"\n    Distribución y_test:")
    for cls_idx, cnt in sorted(pd.Series(y_test).value_counts().items()):
        cls_name = label_encoder.classes_[cls_idx]
        print(f"      {cls_name:<12} ({cls_idx}): {cnt:>4} ({cnt/len(y_test)*100:.1f}%)")

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_test":        X_test,
        "y_test":        y_test,
        "preprocessor":  preprocessor,
        "label_encoder": label_encoder,
        "metadata":      metadata,
    }


# =============================================================================
# CONSTRUCCIÓN DEL PIPELINE COMPLETO
# =============================================================================

def build_rf_pipeline(
    preprocessor,
    n_estimators:     int = 300,
    max_depth:        int = None,
    min_samples_split: int = 2,
    min_samples_leaf:  int = 1,
    max_features:     str = "sqrt",
    class_weight:     str = "balanced",
) -> Pipeline:
    """
    Construye el Pipeline completo: preprocessor + RandomForestClassifier.

    Parámetros
    ----------
    preprocessor : ColumnTransformer
        El transformador YA AJUSTADO de Fase 2. No se vuelve a ajustar.
        Solo se usa su método transform() sobre los datos.

    class_weight : str
        "balanced" → sklearn ajusta automáticamente los pesos de clase
        inversamente proporcional a la frecuencia:
            w_i = n_samples / (n_classes * n_samples_i)
        Esto beneficia a la clase minoritaria "draw" durante el entrenamiento.

    Returns
    -------
    Pipeline sklearn completo.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        oob_score=True,   # Out-of-bag score como estimación interna
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   rf),
    ])

    return pipeline


# =============================================================================
# MODELO BASELINE
# =============================================================================

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    preprocessor,
) -> dict:
    """
    Entrena un Random Forest baseline con hiperparámetros por defecto.

    Propósito:
        Establecer un punto de referencia antes del tuning.
        Permite cuantificar la mejora que aporta GridSearchCV.

    Returns
    -------
    dict con modelo, métricas y predicciones.
    """
    print("\n  Entrenando modelo baseline...")
    t0 = time.time()

    pipeline = build_rf_pipeline(preprocessor, n_estimators=100)
    pipeline.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"    Tiempo de entrenamiento: {elapsed:.1f}s")

    # OOB score
    oob = pipeline.named_steps["classifier"].oob_score_
    print(f"    OOB Score (estimación interna): {oob:.4f}")

    # Predicciones
    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    metrics = compute_metrics(y_train, y_pred_train, y_test, y_pred_test,
                              label="BASELINE")

    return {
        "pipeline":      pipeline,
        "y_pred_train":  y_pred_train,
        "y_pred_test":   y_pred_test,
        "metrics":       metrics,
        "oob_score":     oob,
        "train_time":    elapsed,
    }


# =============================================================================
# CROSS-VALIDATION TEMPORAL
# =============================================================================

def temporal_cross_validation(
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    preprocessor,
    n_splits:    int = 5,
) -> dict:
    """
    Cross-validation respetando la naturaleza temporal del dataset.

    ¿Por qué TimeSeriesSplit y no StratifiedKFold?
        El dataset es una serie temporal de partidos de Copa del Mundo.
        Un fold aleatorio mezclaría partidos de 1970 (test) con 2010 (train),
        lo cual es temporalmente imposible en producción.

        TimeSeriesSplit garantiza que en cada fold:
            TRAIN → partidos anteriores al fold actual
            VAL   → partidos posteriores (nunca vistos en train)

    Esquema con n_splits=5 y ~836 muestras de train:
        Fold 1: train[0:139]    → val[139:279]
        Fold 2: train[0:279]    → val[279:419]
        Fold 3: train[0:419]    → val[419:558]
        Fold 4: train[0:558]    → val[558:697]
        Fold 5: train[0:697]    → val[697:836]

    Returns
    -------
    dict con métricas de cada fold y promedios.
    """
    print(f"\n  Cross-validation temporal (TimeSeriesSplit, n_splits={n_splits})...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipeline = build_rf_pipeline(preprocessor, n_estimators=200)

    scoring = {
        "accuracy":  "accuracy",
        "f1_macro":  "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro":    "recall_macro",
    }

    t0 = time.time()
    cv_results = cross_validate(
        pipeline,
        X_train, y_train,
        cv=tscv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=N_JOBS,
    )
    elapsed = time.time() - t0

    print(f"    CV completado en {elapsed:.1f}s\n")
    print(f"    {'Métrica':<22} {'Train mean':>12} {'Val mean':>12} {'Val std':>10}")
    print(f"    {'─'*58}")

    summary = {}
    for key in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
        tr_mean = cv_results[f"train_{key}"].mean()
        va_mean = cv_results[f"test_{key}"].mean()
        va_std  = cv_results[f"test_{key}"].std()
        print(f"    {key:<22} {tr_mean:>12.4f} {va_mean:>12.4f} {va_std:>10.4f}")
        summary[key] = {
            "train_mean": float(tr_mean),
            "val_mean":   float(va_mean),
            "val_std":    float(va_std),
        }

    # Detección de overfitting
    acc_gap = summary["accuracy"]["train_mean"] - summary["accuracy"]["val_mean"]
    print(f"\n    Gap Train-Val Accuracy: {acc_gap:.4f} "
          f"({'⚠️ posible overfitting' if acc_gap > 0.15 else '✓ controlado'})")

    return {"cv_results": cv_results, "summary": summary, "elapsed": elapsed}


# =============================================================================
# GRIDSEARCHCV CON SPLIT TEMPORAL
# =============================================================================

def grid_search_rf(
    X_train:  pd.DataFrame,
    y_train:  pd.Series,
    preprocessor,
    param_grid: dict = None,
    n_splits:   int = 5,
    fast_mode:  bool = False,
) -> dict:
    """
    Búsqueda de hiperparámetros con GridSearchCV + TimeSeriesSplit.

    Combinaciones exploradas (grid completo):
        n_estimators:     3 valores
        max_depth:        3 valores
        min_samples_split: 3 valores
        min_samples_leaf:  3 valores
        max_features:     2 valores
        ─────────────────────────────
        Total: 3×3×3×3×2 = 162 combinaciones × 5 folds = 810 entrenamientos

    Parámetros
    ----------
    fast_mode : bool
        Si True, usa RF_PARAM_GRID_FAST (reducido) para pruebas rápidas.

    Returns
    -------
    dict con el mejor estimador, los mejores parámetros y los resultados.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID_FAST if fast_mode else RF_PARAM_GRID

    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)

    print(f"\n  GridSearchCV — {n_combinations} combinaciones × {n_splits} folds "
          f"= {n_combinations * n_splits} entrenamientos")
    print(f"    Modo: {'RÁPIDO (grid reducido)' if fast_mode else 'COMPLETO'}")

    tscv     = TimeSeriesSplit(n_splits=n_splits)
    pipeline = build_rf_pipeline(preprocessor)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="f1_weighted",   # métrica principal para ranking de configs
        refit=True,              # el mejor modelo se reentrena con TODO el train
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=True,
    )

    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n    GridSearchCV completado en {elapsed/60:.1f} min")
    print(f"    Mejor f1_weighted (CV): {gs.best_score_:.4f}")
    print(f"\n    Mejores hiperparámetros:")
    for k, v in gs.best_params_.items():
        print(f"      {k:<45} = {v}")

    return {
        "grid_search":    gs,
        "best_estimator": gs.best_estimator_,
        "best_params":    gs.best_params_,
        "best_score":     gs.best_score_,
        "elapsed":        elapsed,
        "cv_results_df":  pd.DataFrame(gs.cv_results_),
    }


# =============================================================================
# MÉTRICAS
# =============================================================================

def compute_metrics(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    label: str = "",
) -> dict:
    """
    Calcula el conjunto exhaustivo de métricas de clasificación.

    Métricas incluidas:
        - Accuracy (global)
        - Precision / Recall / F1 weighted
        - F1 por clase (incluyendo "draw" — clase crítica)
        - Classification report completo
        - Matriz de confusión

    ¿Por qué f1_weighted como métrica principal?
        Pondera el F1 de cada clase por su soporte (frecuencia).
        Es más informativa que accuracy en presencia de desbalance.
        La clase "draw" tiene menos soporte → su F1 impacta menos,
        pero se reporta por separado para análisis específico.
    """
    def _metrics_split(y_true, y_pred, split_name):
        acc = accuracy_score(y_true, y_pred)
        prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_w  = recall_score(y_true, y_pred, average="weighted",  zero_division=0)
        f1_w   = f1_score(y_true, y_pred, average="weighted",      zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro",         zero_division=0)
        f1_per = f1_score(y_true, y_pred, average=None,            zero_division=0)
        cm     = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=CLASS_NAMES,
                                       zero_division=0)
        return {
            "split":        split_name,
            "accuracy":     float(acc),
            "precision_w":  float(prec_w),
            "recall_w":     float(rec_w),
            "f1_weighted":  float(f1_w),
            "f1_macro":     float(f1_mac),
            "f1_per_class": {CLASS_NAMES[i]: float(f1_per[i])
                             for i in range(len(CLASS_NAMES))},
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    train_m = _metrics_split(y_train, y_pred_train, "train")
    test_m  = _metrics_split(y_test,  y_pred_test,  "test")

    # Reporte en consola
    header = f"  [{label}]" if label else "  [EVALUACIÓN]"
    print(f"\n{header}")
    print(f"  {'Métrica':<20} {'Train':>10} {'Test':>10}")
    print(f"  {'─'*42}")
    for m in ["accuracy", "precision_w", "recall_w", "f1_weighted", "f1_macro"]:
        print(f"  {m:<20} {train_m[m]:>10.4f} {test_m[m]:>10.4f}")
    print(f"\n  F1 por clase (Test):")
    for cls, val in test_m["f1_per_class"].items():
        flag = " ← clase crítica" if cls == "draw" else ""
        print(f"    {cls:<12}: {val:.4f}{flag}")

    overfitting_gap = train_m["accuracy"] - test_m["accuracy"]
    print(f"\n  Overfitting gap (Accuracy): {overfitting_gap:.4f} "
          f"({'⚠️' if overfitting_gap > 0.15 else '✓'})")

    print(f"\n  Classification Report (Test):\n")
    print(test_m["classification_report"])

    return {"train": train_m, "test": test_m}


# =============================================================================
# IMPORTANCIA DE FEATURES
# =============================================================================

def get_feature_importance(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Extrae y ordena la importancia de features del RandomForest.

    Reconstruye los nombres de las features después de la transformación
    del ColumnTransformer (OHE expande las categóricas).

    Returns
    -------
    pd.DataFrame con columnas: feature, importance, rank
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    rf           = pipeline.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        n_features = rf.n_features_in_
        feature_names = [f"feature_{i}" for i in range(n_features)]

    importances = rf.feature_importances_
    std         = np.std([tree.feature_importances_
                          for tree in rf.estimators_], axis=0)

    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
        "std":        std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df_imp["rank"] = df_imp.index + 1

    # Limpiar prefijos del ColumnTransformer
    df_imp["feature_clean"] = (
        df_imp["feature"]
        .str.replace(r"^(num__|ord__|cat__)", "", regex=True)
    )

    print(f"\n  Top {top_n} features más importantes:")
    print(f"  {'#':<4} {'Feature':<40} {'Importance':>12} {'Std':>8}")
    print(f"  {'─'*66}")
    for _, row in df_imp.head(top_n).iterrows():
        print(f"  {int(row['rank']):<4} {row['feature_clean']:<40} "
              f"{row['importance']:>12.4f} {row['std']:>8.4f}")

    return df_imp


# =============================================================================
# SERIALIZACIÓN
# =============================================================================

def save_phase3_artifacts(
    best_pipeline:  Pipeline,
    baseline_result: dict,
    tuned_result:   dict,
    cv_result:      dict,
    feature_imp:    pd.DataFrame,
    paths:          dict = None,
) -> None:
    """
    Serializa todos los artefactos de Fase 3.

    Archivos generados:
        models/random_forest_model.pkl    → solo el clasificador RF
        models/random_forest_pipeline.pkl → Pipeline completo (preprocessor + RF)
        models/random_forest_metrics.json → métricas, hiperparámetros, CV results

    El pipeline completo es el artefacto principal para FastAPI:
        pipeline.predict(X_new_raw) → predicción directa sin preprocesar manualmente
    """
    if paths is None:
        paths = DEFAULT_PATHS

    os.makedirs("models", exist_ok=True)

    # 1. Solo el clasificador
    rf_model = best_pipeline.named_steps["classifier"]
    joblib.dump(rf_model, paths["model"])
    print(f"\n    Modelo RF guardado:     {paths['model']}")

    # 2. Pipeline completo
    joblib.dump(best_pipeline, paths["pipeline"])
    print(f"    Pipeline guardado:      {paths['pipeline']}")

    # 3. Métricas JSON
    metrics_payload = {
        "project": "Copa Mundial FIFA 1930–2022 — ML Predicción",
        "phase":   "Fase 3 — Modelado Supervisado Random Forest",
        "model":   "RandomForestClassifier",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "classes": CLASS_NAMES,

        "baseline": {
            "train": baseline_result["metrics"]["train"],
            "test":  baseline_result["metrics"]["test"],
            "oob_score": float(baseline_result["oob_score"]),
        },

        "tuned": {
            "best_params":    tuned_result["best_params"],
            "best_cv_score":  float(tuned_result["best_score"]),
            "train": tuned_result["metrics"]["train"],
            "test":  tuned_result["metrics"]["test"],
        },

        "cross_validation": cv_result["summary"],

        "feature_importance_top20": (
            feature_imp[["feature_clean", "importance", "std"]]
            .head(20)
            .rename(columns={"feature_clean": "feature"})
            .to_dict(orient="records")
        ),

        "anti_leakage_notes": [
            "Preprocessor ajustado SOLO sobre X_train (Fase 2)",
            "X_test solo usa .transform() nunca .fit_transform()",
            "TimeSeriesSplit en CV y GridSearchCV",
            "Features históricas calculadas con expanding().mean().shift(1)",
            "home_score/away_score/xg excluidos de features",
        ],
    }

    # Limpiar arrays numpy para JSON
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"No serializable: {type(obj)}")

    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False,
                  default=_json_safe)
    print(f"    Métricas JSON guardadas: {paths['metrics']}")
