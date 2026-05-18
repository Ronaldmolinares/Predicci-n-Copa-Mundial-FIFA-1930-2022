# Script ejecutable: FASE 2 - Limpieza de datos y Feature Engineering
"""
==============================================================================
FASE 2 — NOTEBOOK EJECUTABLE
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  notebooks/fase_2_preprocessing.py

Flujo de ejecución:
    1. Carga del dataset enriquecido de Fase 1
    2. Limpieza de datos
    3. Ingeniería de características históricas (anti-leakage)
    4. Split temporal train/test
    5. Construcción de Pipelines Scikit-learn
    6. Validaciones de calidad
    7. Persistencia de artefactos para Fase 3

Prerequisito:
    Haber ejecutado notebooks/fase_1_eda.py primero.
    El archivo outputs/matches_enriched_phase1.csv debe existir.

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
==============================================================================
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_matches, report_initial_inspection
from src.feature_engineering import assemble_features, build_kmeans_profile
from src.preprocessing import (
    clean_dataset,
    temporal_train_test_split,
    build_rf_preprocessor,
    build_rf_pipeline,
    build_kmeans_pipeline,
    prepare_Xy_rf,
    prepare_X_kmeans,
    save_preprocessing_metadata,
    save_label_encoder,
    RF_FEATURES,
    LEAKAGE_COLS,
    TARGET_COL,
)

OUTPUT_DIR = "outputs/phase2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

PALETTE = {
    "primary":   "#1a3a5c",
    "secondary": "#c8a951",
    "accent":    "#c0392b",
    "win":       "#27ae60",
    "draw":      "#f39c12",
    "loss":      "#c0392b",
}

# =============================================================================
# BANNER
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         FASE 2: PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS     ║
║         Copa Mundial FIFA 1930–2022                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# BLOQUE 1: CARGA
# =============================================================================

print("═" * 68)
print("  BLOQUE 1: CARGA DEL DATASET")
print("═" * 68)

# Intentar cargar desde Fase 1 (enriquecido) o desde raw
enriched_path = "outputs/matches_enriched_phase1.csv"
if os.path.exists(enriched_path):
    df_raw = pd.read_csv(enriched_path)
    print(f"  Cargado desde Fase 1: {enriched_path}")
    print(f"     Shape: {df_raw.shape}")
else:
    print(f"    '{enriched_path}' no encontrado.")
    print("     Cargando desde dataset original (data/matches_1930_2022.csv)...")
    df_raw = load_matches()
    # Recriar columnas básicas de Fase 1 si es necesario
    if "result" not in df_raw.columns:
        def classify_result(row):
            if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
                return np.nan
            if row["home_score"] > row["away_score"]:   return "home_win"
            elif row["home_score"] == row["away_score"]: return "draw"
            else:                                         return "away_win"
        df_raw["result"] = df_raw.apply(classify_result, axis=1)

print(f"\n  Columnas disponibles ({len(df_raw.columns)}):")
for col in df_raw.columns:
    null_pct = df_raw[col].isna().mean() * 100
    print(f"    • {col:<40} nulos={null_pct:.1f}%")

# =============================================================================
# BLOQUE 2: LIMPIEZA
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 2: LIMPIEZA DE DATOS")
print("═" * 68)
print()

df_clean = clean_dataset(df_raw)

# =============================================================================
# BLOQUE 3: INGENIERÍA DE CARACTERÍSTICAS HISTÓRICAS
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 3: INGENIERÍA DE CARACTERÍSTICAS HISTÓRICAS")
print("═" * 68)
print("""
    Principio anti-leakage:
     Toda estadística histórica se calcula con partidos ANTERIORES
     al partido actual usando: .expanding().mean().shift(1)
     El partido actual NO se incluye en sus propias features.
""")

df_featured = assemble_features(df_clean)

# Verificar que NO hay columnas de leakage en las features para RF
all_rf_feature_cols = (
    RF_FEATURES["numeric"] + RF_FEATURES["ordinal"] + RF_FEATURES["categorical"]
)
leakage_present = [c for c in LEAKAGE_COLS if c in all_rf_feature_cols]
if leakage_present:
    raise RuntimeError(f" LEAKAGE DETECTADO en features RF: {leakage_present}")
else:
    print("\n   Verificación anti-leakage: NINGUNA columna prohibida en features RF.")

# =============================================================================
# BLOQUE 4: PERFIL PARA K-MEANS
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 4: VECTOR DE PERFIL PARA K-MEANS")
print("═" * 68)

kmeans_profile = build_kmeans_profile(df_featured)

print(f"\n  Vista previa del perfil (top 10 por win_rate):")
top10 = kmeans_profile.sort_values("win_rate", ascending=False).head(10)
print(top10[["team", "total_matches", "win_rate", "avg_goal_diff",
             "num_tournaments", "best_stage"]].to_string(index=False))

# =============================================================================
# BLOQUE 5: SPLIT TEMPORAL TRAIN / TEST
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 5: SEPARACIÓN TEMPORAL TRAIN / TEST")
print("═" * 68)
print("""
    Justificación del split temporal:
     El dataset es una serie temporal. Un split aleatorio introduciría
     datos del futuro en el entrenamiento (leakage temporal).
     Estrategia: últimos 2 torneos (2018 + 2022) → TEST.
""")

df_train, df_test = temporal_train_test_split(df_featured, test_tournaments=2)

# =============================================================================
# BLOQUE 6: PREPARACIÓN DE MATRICES X e y
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 6: MATRICES X e y PARA RANDOM FOREST")
print("═" * 68)

X_train, y_train, label_encoder = prepare_Xy_rf(df_train)
X_test,  y_test,  _             = prepare_Xy_rf(df_test)
# Reusar el mismo label_encoder (ajustado en train) para test
from sklearn.preprocessing import LabelEncoder as _LE
_le_test = _LE()
_le_test.classes_ = label_encoder.classes_
y_test = pd.Series(
    _le_test.transform(df_test[TARGET_COL]),
    index=df_test.index,
    name="result_encoded"
)

print(f"\n  X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")

X_kmeans, team_names = prepare_X_kmeans(kmeans_profile)

# =============================================================================
# BLOQUE 7: CONSTRUCCIÓN DE PIPELINES
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 7: CONSTRUCCIÓN DE PIPELINES SCIKIT-LEARN")
print("═" * 68)

# Pipeline RF
print("\n   Pipeline Random Forest:")
rf_preprocessor = build_rf_preprocessor(X_train)
rf_pipeline      = build_rf_pipeline(rf_preprocessor)
print(f"   rf_pipeline construido:")
print(f"     Pasos: {[s[0] for s in rf_pipeline.steps]}")

# Pipeline K-Means
print("\n   Pipeline K-Means:")
kmeans_pipeline = build_kmeans_pipeline()
print(f"   kmeans_pipeline construido:")
print(f"     Pasos: {[s[0] for s in kmeans_pipeline.steps]}")

# Verificar que el preprocessor transforma sin error
# (Fit solo sobre X_train → correcto, no toca X_test)
print("\n   Verificando transformación del preprocessor en X_train...")
try:
    rf_preprocessor.fit(X_train)
    X_train_transformed = rf_preprocessor.transform(X_train)
    X_test_transformed  = rf_preprocessor.transform(X_test)   # ← solo transform!
    print(f"   X_train transformado: {X_train_transformed.shape}")
    print(f"   X_test  transformado: {X_test_transformed.shape}")
    print(f"     (Sin NaN: {np.isnan(X_train_transformed).sum() == 0})")
except Exception as e:
    print(f"   Error en transformación: {e}")

print("\n   Verificando pipeline K-Means...")
try:
    kmeans_pipeline_fitted = joblib.load.__class__  # solo para typing
    from sklearn.pipeline import Pipeline as _P
    _p_test = _P([
        ("imputer", rf_preprocessor.named_transformers_["num"].named_steps["imputer"].__class__()),
        ("scaler",  __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()),
    ])
    _p_test.fit(X_kmeans)
    X_kmeans_t = _p_test.transform(X_kmeans)
    print(f"   X_kmeans transformado: {X_kmeans_t.shape}")
except Exception as e:
    print(f"   Verificación K-Means: {e}")

# =============================================================================
# BLOQUE 8: VISUALIZACIONES DE VALIDACIÓN
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 8: VISUALIZACIONES DE VALIDACIÓN")
print("═" * 68)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Fase 2 — Validación del Preprocesamiento",
             fontsize=14, fontweight="bold", color=PALETTE["primary"])

# 8a. Distribución de clases en train vs test
ax1 = fig.add_subplot(gs[0, 0])
classes_map = {i: c for i, c in enumerate(label_encoder.classes_)}
train_dist = y_train.value_counts().sort_index()
test_dist  = y_test.value_counts().sort_index()
x = np.arange(len(label_encoder.classes_))
w = 0.35
ax1.bar(x - w/2, train_dist.values / len(y_train) * 100,
        w, label="Train", color=PALETTE["primary"], alpha=0.85)
ax1.bar(x + w/2, test_dist.values  / len(y_test)  * 100,
        w, label="Test",  color=PALETTE["secondary"], alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels([classes_map[i] for i in range(len(label_encoder.classes_))],
                    rotation=15, fontsize=9)
ax1.set_ylabel("% de partidos", fontsize=10)
ax1.set_title("Distribución del Target\nTrain vs Test", fontsize=10)
ax1.legend(fontsize=9)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
sns.despine(ax=ax1)

# 8b. Partidos por torneo (train=azul, test=naranja)
ax2 = fig.add_subplot(gs[0, 1])
if "Year" in df_featured.columns:
    yr_counts = df_featured.groupby("Year").size().reset_index(name="count")
    yr_counts["split"] = yr_counts["Year"].apply(
        lambda y: "test" if y in df_test["Year"].values else "train"
    )
    colors_bar = [PALETTE["accent"] if s == "test" else PALETTE["primary"]
                  for s in yr_counts["split"]]
    ax2.bar(yr_counts["Year"].astype(int).astype(str),
            yr_counts["count"], color=colors_bar, edgecolor="white")
    ax2.set_title("Partidos por Torneo\n(🔵 Train / 🔴 Test)", fontsize=10)
    ax2.set_ylabel("Nº de partidos", fontsize=10)
    ax2.tick_params(axis="x", rotation=65, labelsize=7)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2)

# 8c. Distribución de win_rate histórico (train)
ax3 = fig.add_subplot(gs[0, 2])
if "home_hist_win_rate" in X_train.columns:
    ax3.hist(X_train["home_hist_win_rate"].dropna(),
             bins=25, color=PALETTE["win"], alpha=0.7,
             edgecolor="white", label="Local")
    ax3.hist(X_train["away_hist_win_rate"].dropna(),
             bins=25, color=PALETTE["loss"], alpha=0.6,
             edgecolor="white", label="Visitante")
    ax3.set_title("Distribución hist_win_rate\n(Train)", fontsize=10)
    ax3.set_xlabel("Win rate histórico", fontsize=9)
    ax3.legend(fontsize=9)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)
    sns.despine(ax=ax3)

# 8d. Distribución de delta_win_rate
ax4 = fig.add_subplot(gs[1, 0])
if "delta_win_rate" in X_train.columns:
    ax4.hist(X_train["delta_win_rate"].dropna(),
             bins=30, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    ax4.axvline(0, color=PALETTE["accent"], linestyle="--", linewidth=2,
                label="0 = misma fuerza")
    ax4.set_title("delta_win_rate\n(local - visitante)", fontsize=10)
    ax4.set_xlabel("Diferencia de win rate", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax4.set_axisbelow(True)
    sns.despine(ax=ax4)

# 8e. Perfil K-Means: scatter win_rate vs avg_goal_diff
ax5 = fig.add_subplot(gs[1, 1])
sc = ax5.scatter(
    kmeans_profile["win_rate"],
    kmeans_profile["avg_goal_diff"],
    c=kmeans_profile["num_tournaments"],
    cmap="viridis", alpha=0.7, s=50, edgecolors="white", linewidth=0.5
)
plt.colorbar(sc, ax=ax5, label="Nº torneos", shrink=0.8)
# Etiquetar top equipos
top_teams = kmeans_profile.nlargest(6, "win_rate")
for _, row in top_teams.iterrows():
    ax5.annotate(row["team"], (row["win_rate"], row["avg_goal_diff"]),
                 fontsize=6, ha="left", va="bottom",
                 color=PALETTE["primary"], fontweight="bold")
ax5.set_xlabel("Tasa de victoria histórica", fontsize=9)
ax5.set_ylabel("Diferencia de goles media", fontsize=9)
ax5.set_title("Espacio K-Means\n(win_rate vs avg_goal_diff)", fontsize=10)
ax5.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax5.axvline(kmeans_profile["win_rate"].mean(), color="gray",
            linestyle=":", alpha=0.5)
sns.despine(ax=ax5)

# 8f. Nulos en X_train por columna
ax6 = fig.add_subplot(gs[1, 2])
null_counts = X_train.isna().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if null_counts.empty:
    ax6.text(0.5, 0.5, " Sin valores nulos\nen X_train",
             ha="center", va="center", fontsize=12,
             color=PALETTE["win"], fontweight="bold",
             transform=ax6.transAxes)
    ax6.set_title("Nulos en X_train", fontsize=10)
    ax6.axis("off")
else:
    ax6.barh(null_counts.index[::-1], null_counts.values[::-1],
             color=PALETTE["accent"], edgecolor="white")
    ax6.set_title("Nulos en X_train (por feature)", fontsize=10)
    ax6.set_xlabel("Conteo de nulos", fontsize=9)
    sns.despine(ax=ax6)

plot_path = os.path.join(OUTPUT_DIR, "phase2_validation.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  💾 Visualizaciones guardadas en: {plot_path}")

# =============================================================================
# BLOQUE 9: GUARDADO DE ARTEFACTOS
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 9: PERSISTENCIA DE ARTEFACTOS")
print("═" * 68)

# 9a. Dataset con features históricas
path_featured = "outputs/matches_featured_phase2.csv"
df_featured.to_csv(path_featured, index=False)
print(f"\n   Dataset con features: {path_featured}")

# 9b. Perfil K-Means
path_kmeans = "outputs/kmeans_profile_phase2.csv"
kmeans_profile.to_csv(path_kmeans, index=False)
print(f"   Perfil K-Means: {path_kmeans}")

# 9c. Splits
path_train = "outputs/train_phase2.csv"
path_test  = "outputs/test_phase2.csv"
df_train.to_csv(path_train, index=False)
df_test.to_csv(path_test, index=False)
print(f"   Train: {path_train}  ({len(df_train):,} filas)")
print(f"   Test : {path_test}   ({len(df_test):,} filas)")

# 9d. LabelEncoder
save_label_encoder(label_encoder, "models/label_encoder.pkl")

# 9e. Metadata JSON
metadata = save_preprocessing_metadata(
    label_encoder, df_train, df_test,
    output_path="models/preprocessing_metadata.json"
)

# 9f. Preprocessor ya ajustado (para Fase 3)
joblib.dump(rf_preprocessor, "models/rf_preprocessor.pkl")
print(f"   RF Preprocessor (ajustado): models/rf_preprocessor.pkl")

joblib.dump(kmeans_pipeline, "models/kmeans_pipeline_base.pkl")
print(f"   KMeans Pipeline base: models/kmeans_pipeline_base.pkl")

# 9g. Matrices X e y para Fase 3
joblib.dump((X_train, y_train, X_test, y_test), "models/Xy_phase2.pkl")
print(f"   Matrices X/y: models/Xy_phase2.pkl")

joblib.dump((X_kmeans, team_names), "models/Xkmeans_phase2.pkl")
print(f"   Matriz K-Means: models/Xkmeans_phase2.pkl")

# =============================================================================
# BLOQUE 10: REPORTE FINAL
# =============================================================================

print("\n" + "═" * 68)
print("  BLOQUE 10: REPORTE FINAL DE FASE 2")
print("═" * 68)

print(f"""
   RESUMEN DE ARTEFACTOS GENERADOS:

  Dataset:
    • Partidos con features históricas : {len(df_featured):,} filas
    • Perfiles K-Means                 : {len(kmeans_profile)} selecciones

  Split temporal:
    • TRAIN : {len(df_train):,} partidos  (torneos {int(df_train['Year'].min())}–{int(df_train['Year'].max())})
    • TEST  : {len(df_test):,} partidos  (torneos {int(df_test['Year'].min())}–{int(df_test['Year'].max())})

  Features para Random Forest:
    • Numéricas  : {len([c for c in RF_FEATURES['numeric']     if c in X_train.columns])}
    • Ordinales  : {len([c for c in RF_FEATURES['ordinal']     if c in X_train.columns])}
    • Categóricas: {len([c for c in RF_FEATURES['categorical'] if c in X_train.columns])}
    • TOTAL input: {X_train.shape[1]}
    • Post-transform: {X_train_transformed.shape[1]} (tras OHE expansión)

  Features para K-Means:
    • Dimensiones: {X_kmeans.shape[1]} features × {X_kmeans.shape[0]} equipos

  Anti-leakage verificado:
     Estadísticas históricas con shift(1)
     Columnas post-partido excluidas del set de features RF
     Split temporal (no aleatorio)
     Preprocessor ajustado SOLO en train
""")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FASE 2 COMPLETADA                                                   ║
║                                                                      ║
║  Artefactos listos para Fase 3:                                      ║
║    models/rf_preprocessor.pkl     ← ColumnTransformer ajustado       ║
║    models/Xy_phase2.pkl           ← X_train, y_train, X_test, y_test ║
║    models/label_encoder.pkl       ← Para invertir predicciones       ║
║    models/preprocessing_metadata.json ← Metadata para FastAPI        ║
║    models/Xkmeans_phase2.pkl      ← Perfil K-Means normalizado       ║
║                                                                      ║
║  → Próximo paso: FASE 3 — Modelado Supervisado (Random Forest)       ║
╚══════════════════════════════════════════════════════════════════════╝
""")