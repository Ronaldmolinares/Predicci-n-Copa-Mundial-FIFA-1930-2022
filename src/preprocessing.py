# Módulo para limpieza de datos y Feature Engineering
"""
==============================================================================
FASE 2 — PREPROCESAMIENTO, PIPELINES Y SEPARACIÓN TRAIN/TEST
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/preprocessing.py

Descripción:
    Limpieza del dataset, construcción de Pipelines de Scikit-learn
    y separación temporal de conjuntos de entrenamiento y prueba.

Decisiones de diseño:
    ┌─────────────────────────────────────────────────────────────────┐
    │  ¿Por qué Pipeline de Scikit-learn y no transformaciones        │
    │  manuales previas al split?                                     │
    │                                                                 │
    │  Si escalamos o codificamos ANTES de hacer el split,            │
    │  el encoder/scaler "ve" los datos de test durante el fit,       │
    │  introduciendo data leakage estadístico.                        │
    │                                                                 │
    │  El Pipeline garantiza que:                                     │
    │    • .fit()       → solo sobre datos de ENTRENAMIENTO           │
    │    • .transform() → aplicado a test con los parámetros          │
    │                     aprendidos del train (sin reajustar)        │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  ¿Por qué split temporal y no aleatorio (train_test_split)?     │
    │                                                                 │
    │  El dataset es una serie temporal de partidos.                  │
    │  Un split aleatorio mezclaría partidos de 2018 en train con     │
    │  partidos de 1970 en test, lo cual es irreal:                   │
    │  en producción solo predecimos hacia el futuro.                 │
    │                                                                 │
    │  Estrategia: los últimos K torneos van a test.                  │
    │  Default: últimos 2 torneos (2018 + 2022) → ~128 partidos       │
    └─────────────────────────────────────────────────────────────────┘

Pipelines construidos:
    rf_pipeline_base   → ColumnTransformer + (placeholder para RF en Fase 3)
    kmeans_pipeline    → StandardScaler para el vector de perfil por equipo

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado — UPTC
==============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import json
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer

# Importar STAGE_ORDER desde feature_engineering (mismo directorio src/)
from .feature_engineering import STAGE_ORDER



warnings.filterwarnings("ignore")

# =============================================================================
# DEFINICIÓN DE FEATURES (CATÁLOGO CENTRAL)
# =============================================================================
# Este catálogo es la fuente de verdad de qué columnas entran en cada modelo.
# Se exporta a metadata.json para que la API FastAPI lo use en validación.

# Features para Random Forest — agrupadas por tipo de transformación
RF_FEATURES = {

    # Numéricas continuas → StandardScaler
    "numeric": [
        "home_hist_win_rate",
        "home_hist_draw_rate",
        "home_hist_avg_gf",
        "home_hist_avg_ga",
        "home_hist_avg_gd",
        "home_hist_matches",
        "away_hist_win_rate",
        "away_hist_draw_rate",
        "away_hist_avg_gf",
        "away_hist_avg_ga",
        "away_hist_avg_gd",
        "away_hist_matches",
        "home_recent_win_rate",
        "away_recent_win_rate",
        "delta_win_rate",
        "delta_avg_gf",
        "delta_avg_gd",
        "delta_wc_exp",
        "home_wc_appearances",
        "away_wc_appearances",
        "Year",
    ],

    # Numéricas ordinales → StandardScaler (ya tienen orden implícito)
    "ordinal": [
        "stage_encoded",
    ],

    # Categóricas nominales → OneHotEncoder (identidad de los equipos)
    "categorical": [
        "home_team",
        "away_team",
    ],
}

# Features para K-Means — todas numéricas, normalizadas con StandardScaler
KMEANS_FEATURES = [
    "total_matches",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "avg_goals_for",
    "avg_goals_against",
    "avg_goal_diff",
    "num_tournaments",
    "best_stage",
    "goals_per_tournament",
]

# Variable objetivo (target)
TARGET_COL = "result"

# Clases del target (para metadata)
TARGET_CLASSES = ["away_win", "draw", "home_win"]  # orden alfabético de LabelEncoder

# Columnas post-partido (PROHIBIDAS en Random Forest — data leakage)
LEAKAGE_COLS = [
    "home_score", "away_score",
    "home_xg", "away_xg",
    "goal_diff", "total_goals",
    "went_to_pen",
    "home_goal", "away_goal",
    "home_goal_long", "away_goal_long",
    "home_own_goal", "away_own_goal",
    "home_penalty_goal", "away_penalty_goal",
    "home_penalty", "away_penalty",
    "home_manager", "away_manager",
    "home_captain", "away_captain",
]

# =============================================================================
# PASO 1: LIMPIEZA BÁSICA
# =============================================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza básica al dataset de partidos.

    Transformaciones aplicadas:
    1. Eliminación de duplicados exactos.
    2. Eliminación de registros sin equipo local o visitante.
    3. Eliminación de registros sin variable objetivo (result).
    4. Normalización de nombres de equipos (strip + title case).
    5. Clamping de scores negativos a 0 (inconsistencia histórica).
    6. Reporte de filas eliminadas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de partidos.

    Returns
    -------
    pd.DataFrame
        Dataset limpio.
    """
    n_initial = len(df)
    df = df.copy()

    # 1. Duplicados exactos
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup = n_before - len(df)
    if n_dup > 0:
        print(f"    {n_dup} filas duplicadas eliminadas.")

    # 2. Filas sin equipo
    mask_teams = df["home_team"].isna() | df["away_team"].isna()
    n_teams = mask_teams.sum()
    if n_teams > 0:
        df = df[~mask_teams]
        print(f"    {n_teams} filas sin equipo eliminadas.")

    # 3. Filas sin variable objetivo
    if TARGET_COL in df.columns:
        mask_target = df[TARGET_COL].isna()
        n_target = mask_target.sum()
        if n_target > 0:
            df = df[~mask_target]
            print(f"    {n_target} filas sin target eliminadas.")

    # 4. Normalización de nombres de equipos
    df["home_team"] = df["home_team"].str.strip().str.title()
    df["away_team"] = df["away_team"].str.strip().str.title()

    # 5. Scores negativos → 0
    for col in ["home_score", "away_score"]:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                df[col] = df[col].clip(lower=0)
                print(f"    {n_neg} valores negativos en '{col}' → 0.")

    n_final = len(df)
    n_removed = n_initial - n_final
    print(f"   Limpieza: {n_initial} → {n_final} filas "
          f"({n_removed} eliminadas, {n_final/n_initial*100:.1f}% retenido)")
    return df


# =============================================================================
# PASO 2: SPLIT TEMPORAL TRAIN / TEST
# =============================================================================

def temporal_train_test_split(
    df: pd.DataFrame,
    test_tournaments: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en conjuntos de entrenamiento y prueba
    respetando el orden temporal.

    Estrategia:
        Los `test_tournaments` torneos más recientes van a TEST.
        Todo lo anterior va a TRAIN.

    ¿Por qué no sklearn.train_test_split?
        train_test_split con shuffle=True mezclaría partidos de 2022
        con partidos de 1930, lo cual es imposible en un escenario real
        de predicción (no conocemos el futuro). El split temporal garantiza
        que el modelo nunca "vio" información futura durante el entrenamiento.

    ¿Por qué los últimos 2 torneos?
        2018 (Rusia) y 2022 (Qatar) representan ~128 partidos (~14% del total).
        Esto da suficiente muestra de test con la estructura moderna
        del torneo (32 equipos, 64 partidos).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'Year'.
    test_tournaments : int
        Número de últimos torneos reservados para test. Default: 2.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test)
    """
    df = df.copy()
    
    # Normalizar nombre de columna de año
    if "Year" not in df.columns:
        raise ValueError(" Columna 'Year' requerida para split temporal.")

    all_years  = sorted(df["Year"].unique())
    test_years = all_years[-test_tournaments:]
    train_years = all_years[:-test_tournaments]

    df_train = df[df["Year"].isin(train_years)].copy()
    df_test  = df[df["Year"].isin(test_years)].copy()

    print(f"\n  Split temporal:")
    print(f"     TRAIN: {len(train_years)} torneos "
          f"({int(min(train_years))}–{int(max(train_years))}) "
          f"→ {len(df_train):,} partidos")
    print(f"     TEST : {len(test_years)} torneos "
          f"({int(min(test_years))}–{int(max(test_years))}) "
          f"→ {len(df_test):,} partidos")
    print(f"\n     Distribución del target en TRAIN:")
    for cls, cnt in df_train[TARGET_COL].value_counts().items():
        print(f"       {cls:<15} {cnt:>4} ({cnt/len(df_train)*100:.1f}%)")
    print(f"\n     Distribución del target en TEST:")
    for cls, cnt in df_test[TARGET_COL].value_counts().items():
        print(f"       {cls:<15} {cnt:>4} ({cnt/len(df_test)*100:.1f}%)")

    return df_train, df_test


# =============================================================================
# PASO 3: CONSTRUCCIÓN DEL PIPELINE PARA RANDOM FOREST
# =============================================================================

def build_rf_preprocessor(df_train: pd.DataFrame) -> ColumnTransformer:
    """
    Construye el ColumnTransformer de preprocesamiento para Random Forest.

    Arquitectura del transformador:

        ┌─────────────────────────────────────────────────────────┐
        │              ColumnTransformer                          │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ numeric_pipeline                                 │   │
        │  │   → SimpleImputer(strategy='median')             │   │
        │  │   → StandardScaler()                             │   │
        │  │   Columnas: hist_*, delta_*, Year, wc_*          │   │
        │  ├──────────────────────────────────────────────────┤   │
        │  │ ordinal_pipeline                                 │   │
        │  │   → SimpleImputer(strategy='most_frequent')      │   │
        │  │   → StandardScaler()                             │   │
        │  │   Columnas: stage_encoded                        │   │
        │  ├──────────────────────────────────────────────────┤   │
        │  │ categorical_pipeline                             │   │
        │  │   → SimpleImputer(strategy='most_frequent')      │   │
        │  │   → OneHotEncoder(handle_unknown='ignore')       │   │
        │  │   Columnas: home_team, away_team                 │   │
        │  └──────────────────────────────────────────────────┘   │
        └─────────────────────────────────────────────────────────┘

    Decisiones técnicas:
    - SimpleImputer(median) para numéricas: robusto a outliers.
    - StandardScaler: necesario para que las variables en distintas
      escalas (rates 0-1 vs matches 0-100) no distorsionen la importancia.
      Random Forest es teóricamente agnóstico a la escala, pero mejora
      con features en escala similar en la práctica.
    - OneHotEncoder(handle_unknown='ignore'): si aparece un equipo nuevo
      en test/producción que no existía en train, lo ignora silenciosamente
      en lugar de lanzar error. Fundamental para robustez en API.
    - drop='first' en OHE NO se aplica: Random Forest no sufre
      multicolinealidad como los modelos lineales.

    Parameters
    ----------
    df_train : pd.DataFrame
        Solo se usa para extraer las categorías válidas de OHE.
        El fit real ocurrirá dentro del Pipeline final.

    Returns
    -------
    ColumnTransformer
        Transformador no entrenado, listo para incluir en Pipeline.
    """
    # Detectar qué features realmente están en el dataframe
    # (evita error si alguna columna no fue generada por el dataset actual)
    available_numeric = [c for c in RF_FEATURES["numeric"]
                         if c in df_train.columns]
    available_ordinal = [c for c in RF_FEATURES["ordinal"]
                         if c in df_train.columns]
    available_cat     = [c for c in RF_FEATURES["categorical"]
                         if c in df_train.columns]

    print(f"\n  Features para RF:")
    print(f"    Numéricas  ({len(available_numeric)}): {available_numeric}")
    print(f"    Ordinales  ({len(available_ordinal)}): {available_ordinal}")
    print(f"    Categóricas({len(available_cat)}): {available_cat}")

    # Sub-pipeline numérico
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Sub-pipeline ordinal (mismo tratamiento que numérico)
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler",  StandardScaler()),
    ])

    # Sub-pipeline categórico
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(
                        handle_unknown="ignore",  # robustez para API
                        sparse_output=False,      # matriz densa (compatibilidad)
                    )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,      available_numeric),
            ("ord",  ordinal_pipeline,      available_ordinal),
            ("cat",  categorical_pipeline,  available_cat),
        ],
        remainder="drop",       # descartar columnas no listadas
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_rf_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Construye el Pipeline completo para Random Forest.

    En Fase 2, el paso del clasificador es un placeholder (passthrough).
    En Fase 3 se reemplazará por RandomForestClassifier.

    El Pipeline tiene esta estructura:

        Pipeline([
            ("preprocessor", ColumnTransformer),   ← se construye aquí
            ("classifier",   "passthrough"),        ← se reemplaza en Fase 3
        ])

    Ventaja: el mismo objeto Pipeline sirve para:
        - .fit(X_train, y_train)     en Fase 3
        - .predict(X_new)            en FastAPI Fase 6
        - joblib.dump(pipeline, ...) serialización completa

    Parameters
    ----------
    preprocessor : ColumnTransformer
        El transformador construido con build_rf_preprocessor().

    Returns
    -------
    Pipeline
        Pipeline completo (sin clasificador real todavía).
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # Placeholder: se reemplaza en Fase 3 por:
        # ("classifier", RandomForestClassifier(class_weight="balanced", ...))
        ("classifier",   "passthrough"),
    ])
    return pipeline


# =============================================================================
# PASO 4: CONSTRUCCIÓN DEL PIPELINE PARA K-MEANS
# =============================================================================

def build_kmeans_pipeline() -> Pipeline:
    """
    Construye el Pipeline de preprocesamiento para K-Means.

    K-Means usa distancia euclidiana → las variables DEBEN estar
    en la misma escala. Sin StandardScaler, variables con alta magnitud
    (como total_matches = 0-100) dominarían sobre tasas (0-1).

    Arquitectura:

        Pipeline([
            ("imputer", SimpleImputer(median)),   ← por si hay NaN en perfil
            ("scaler",  StandardScaler()),         ← centralizar + normalizar
        ])

    En Fase 4 se añadirá:
        ("kmeans", KMeans(n_clusters=k, ...))

    Returns
    -------
    Pipeline
        Pipeline de preprocesamiento para K-Means.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        # Placeholder → en Fase 4:
        # ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10))
        ("model",   "passthrough"),
    ])
    return pipeline


# =============================================================================
# PASO 5: PREPARACIÓN DE MATRICES X e y
# =============================================================================

def prepare_Xy_rf(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extrae X (features) e y (target) del dataset para Random Forest.

    Garantías:
    - Ninguna columna de LEAKAGE_COLS está en X.
    - El target se codifica con LabelEncoder → entero (0, 1, 2).
    - X solo contiene columnas listadas en RF_FEATURES.

    Codificación del target:
        LabelEncoder sobre ['away_win', 'draw', 'home_win'] (orden alfabético)
        → 0 = away_win  |  1 = draw  |  2 = home_win
        Se guarda en metadata para que la API invierta la codificación.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de partidos con features históricas y columna 'result'.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y)  donde y es la serie codificada como entero.
    """
    # Verificar que el target existe
    if TARGET_COL not in df.columns:
        raise ValueError(f" Columna target '{TARGET_COL}' no encontrada.")

    # Columnas de features disponibles en el DataFrame
    all_feature_cols = (
        RF_FEATURES["numeric"]   +
        RF_FEATURES["ordinal"]   +
        RF_FEATURES["categorical"]
    )
    available_cols = [c for c in all_feature_cols if c in df.columns]
    missing_cols   = [c for c in all_feature_cols if c not in df.columns]

    if missing_cols:
        print(f"    Features no disponibles en este df: {missing_cols}")

    X = df[available_cols].copy()
    y_raw = df[TARGET_COL].copy()

    # Codificación del target
    le = LabelEncoder()
    y  = pd.Series(le.fit_transform(y_raw), index=df.index, name="result_encoded")

    print(f"\n   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Clases: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    return X, y, le


def prepare_X_kmeans(profile_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Extrae la matriz de features para K-Means desde el perfil por equipo.

    Parameters
    ----------
    profile_df : pd.DataFrame
        DataFrame con una fila por equipo (salida de build_kmeans_profile).

    Returns
    -------
    tuple[pd.DataFrame, list]
        (X_kmeans, team_names)
    """
    available = [c for c in KMEANS_FEATURES if c in profile_df.columns]
    missing   = [c for c in KMEANS_FEATURES if c not in profile_df.columns]

    if missing:
        print(f"    Features K-Means no disponibles: {missing}")

    X_kmeans   = profile_df[available].copy()
    team_names = profile_df["team"].tolist() if "team" in profile_df.columns else []

    print(f"\n   X_kmeans shape: {X_kmeans.shape}")
    print(f"   Equipos incluidos: {len(team_names)}")

    return X_kmeans, team_names


# =============================================================================
# PASO 6: GENERACIÓN Y GUARDADO DE METADATA
# =============================================================================

def save_preprocessing_metadata(
    le: LabelEncoder,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_path: str = "models/preprocessing_metadata.json",
) -> dict:
    """
    Guarda los metadatos de preprocesamiento en JSON.

    Este archivo será cargado por:
    - Fase 3: para conocer las clases del target
    - Fase 6 (FastAPI): para validar inputs y devolver predicciones legibles

    Contenido:
        feature_names_rf        Lista de features de entrada para RF
        target_classes          Clases del target (orden del LabelEncoder)
        target_mapping          {int: nombre_clase}
        leakage_cols_excluded   Columnas explícitamente excluidas
        train_info              Información del conjunto de entrenamiento
        test_info               Información del conjunto de prueba
        stage_encoding          Mapa de fases a enteros

    Parameters
    ----------
    le : LabelEncoder
        LabelEncoder ajustado sobre y_train.
    df_train / df_test : pd.DataFrame
        Para calcular estadísticas del split.
    output_path : str
        Ruta de salida del JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_rf_features = (
        [c for c in RF_FEATURES["numeric"]      if c in df_train.columns] +
        [c for c in RF_FEATURES["ordinal"]       if c in df_train.columns] +
        [c for c in RF_FEATURES["categorical"]   if c in df_train.columns]
    )

    metadata = {
        "project": "Copa Mundial FIFA 1930-2022 — ML Predicción",
        "phase": "Fase 2 — Preprocesamiento",
        "feature_names_rf": all_rf_features,
        "feature_groups": {
            "numeric":      [c for c in RF_FEATURES["numeric"]     if c in df_train.columns],
            "ordinal":      [c for c in RF_FEATURES["ordinal"]     if c in df_train.columns],
            "categorical":  [c for c in RF_FEATURES["categorical"] if c in df_train.columns],
        },
        "kmeans_features": KMEANS_FEATURES,
        "target_col": TARGET_COL,
        "target_classes": list(le.classes_),
        "target_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
        "leakage_cols_excluded": LEAKAGE_COLS,
        "train_info": {
            "n_samples":     int(len(df_train)),
            "years":         [int(y) for y in sorted(df_train["Year"].unique())]
                             if "Year" in df_train.columns else [],
            "class_dist":    {k: int(v)
                              for k, v in df_train[TARGET_COL].value_counts().items()},
        },
        "test_info": {
            "n_samples":     int(len(df_test)),
            "years":         [int(y) for y in sorted(df_test["Year"].unique())]
                             if "Year" in df_test.columns else [],
            "class_dist":    {k: int(v)
                              for k, v in df_test[TARGET_COL].value_counts().items()},
        },
        "stage_encoding": STAGE_ORDER,
        "recent_window": 5,
        "split_strategy": "temporal — últimos 2 torneos como test",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n   Metadata guardada en: {output_path}")
    return metadata


def save_label_encoder(
    le: LabelEncoder,
    output_path: str = "models/label_encoder.pkl",
) -> None:
    """
    Serializa el LabelEncoder para reutilizarlo en Fase 3 y Fase 6.

    En la API FastAPI necesitaremos invertir la codificación:
        le.inverse_transform([2]) → 'home_win'
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(le, output_path)
    print(f"   LabelEncoder guardado en: {output_path}")