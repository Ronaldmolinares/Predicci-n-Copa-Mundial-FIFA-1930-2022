"""
==============================================================================
FASE 3 — NOTEBOOK EJECUTABLE: RANDOM FOREST
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  notebooks/fase_3_random_forest.py

Flujo:
    1. Carga de artefactos Fase 2
    2. Modelo Baseline
    3. Cross-Validation Temporal
    4. GridSearchCV
    5. Evaluación exhaustiva (tuned vs baseline)
    6. Análisis de feature importance
    7. Visualizaciones profesionales
    8. Serialización de artefactos

Prerequisito: haber ejecutado notebooks/fase_2_preprocessing.py
Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
==============================================================================
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_supervised import (
    load_phase2_artifacts,
    build_rf_pipeline,
    train_baseline,
    temporal_cross_validation,
    grid_search_rf,
    compute_metrics,
    get_feature_importance,
    save_phase3_artifacts,
    CLASS_NAMES,
    DEFAULT_PATHS,
)

# ─── Paleta y estilo ─────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1a3a5c",
    "secondary": "#c8a951",
    "accent":    "#c0392b",
    "win":       "#27ae60",
    "draw":      "#e67e22",
    "loss":      "#c0392b",
    "neutral":   "#7f8c8d",
    "bg":        "#f8f9fa",
}
CLASS_COLORS = [PALETTE["loss"], PALETTE["draw"], PALETTE["win"]]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.facecolor":  PALETTE["bg"],
    "figure.facecolor": "white",
})

OUTPUT_DIR = "outputs/phase3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
# BANNER
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║         FASE 3: MODELADO SUPERVISADO — RANDOM FOREST                 ║
║         Copa Mundial FIFA 1930–2022                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# BLOQUE 1: CARGA DE ARTEFACTOS
# =============================================================================
print("═" * 68)
print("  BLOQUE 1: CARGA DE ARTEFACTOS DE FASE 2")
print("═" * 68)

arts = load_phase2_artifacts()
X_train = arts["X_train"]
y_train = arts["y_train"]
X_test  = arts["X_test"]
y_test  = arts["y_test"]
preprocessor  = arts["preprocessor"]
label_encoder = arts["label_encoder"]
metadata      = arts["metadata"]

# =============================================================================
# BLOQUE 2: MODELO BASELINE
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 2: MODELO BASELINE")
print("═" * 68)
print("""
  Propósito del baseline:
    Entrenar con hiperparámetros por defecto para establecer una
    referencia mínima antes de la optimización. Nos permite cuantificar
    cuánto mejora GridSearchCV respecto al punto de partida.
""")

baseline = train_baseline(X_train, y_train, X_test, y_test, preprocessor)

# =============================================================================
# BLOQUE 3: CROSS-VALIDATION TEMPORAL
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 3: CROSS-VALIDATION TEMPORAL (TimeSeriesSplit)")
print("═" * 68)
print("""
  TimeSeriesSplit garantiza que cada fold valide sobre partidos
  posteriores a los usados en entrenamiento. Nunca mezcla futuro
  y pasado — reproduce el escenario real de predicción.
""")

cv_result = temporal_cross_validation(X_train, y_train, preprocessor, n_splits=5)

# =============================================================================
# BLOQUE 4: GRIDSEARCHCV
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 4: TUNING — GridSearchCV + TimeSeriesSplit")
print("═" * 68)
print("""
  Hiperparámetros explorados:
    n_estimators:      número de árboles en el bosque
    max_depth:         profundidad máxima (None = sin límite)
    min_samples_split: mínimo de muestras para dividir un nodo
    min_samples_leaf:  mínimo de muestras en una hoja
    max_features:      features a evaluar en cada split
  
  Scoring: f1_weighted (maneja desbalance de clases)
  Estrategia CV: TimeSeriesSplit(n_splits=5)
""")

# Cambiar a fast_mode=False para grid completo (más lento)
gs_result = grid_search_rf(
    X_train, y_train, preprocessor,
    n_splits=5,
    fast_mode=True,  # Cambiar a False para grid completo
)

best_pipeline = gs_result["best_estimator"]

# =============================================================================
# BLOQUE 5: EVALUACIÓN DEL MODELO TUNED
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 5: EVALUACIÓN EXHAUSTIVA — MODELO TUNED")
print("═" * 68)

y_pred_train_tuned = best_pipeline.predict(X_train)
y_pred_test_tuned  = best_pipeline.predict(X_test)

tuned_metrics = compute_metrics(
    y_train, y_pred_train_tuned,
    y_test,  y_pred_test_tuned,
    label="TUNED",
)
gs_result["metrics"] = tuned_metrics

# Importancia de features
print("\n" + "─" * 68)
print("  Análisis de Feature Importance")
print("─" * 68)
feature_imp = get_feature_importance(best_pipeline, top_n=20)

# =============================================================================
# BLOQUE 6: COMPARATIVA BASELINE vs TUNED
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 6: COMPARATIVA BASELINE vs TUNED")
print("═" * 68)

metrics_compare = {
    "Accuracy":      ("accuracy",    "accuracy"),
    "Precision_W":   ("precision_w", "precision_w"),
    "Recall_W":      ("recall_w",    "recall_w"),
    "F1_Weighted":   ("f1_weighted", "f1_weighted"),
    "F1_Macro":      ("f1_macro",    "f1_macro"),
}

print(f"\n  {'Métrica':<18} {'Baseline Train':>14} {'Baseline Test':>14} "
      f"{'Tuned Train':>12} {'Tuned Test':>12} {'Δ Test':>8}")
print(f"  {'─'*80}")

for name, (k1, k2) in metrics_compare.items():
    b_tr = baseline["metrics"]["train"][k1]
    b_te = baseline["metrics"]["test"][k2]
    t_tr = tuned_metrics["train"][k1]
    t_te = tuned_metrics["test"][k2]
    delta = t_te - b_te
    arrow = "▲" if delta > 0 else "▼"
    print(f"  {name:<18} {b_tr:>14.4f} {b_te:>14.4f} "
          f"{t_tr:>12.4f} {t_te:>12.4f} {arrow}{abs(delta):>6.4f}")

print(f"\n  F1 por clase — Test (Tuned):")
print(f"  {'Clase':<14} {'F1':>8}   Interpretación")
print(f"  {'─'*55}")
for cls, val in tuned_metrics["test"]["f1_per_class"].items():
    interp = {
        "home_win": "Clase mayoritaria — más fácil de predecir",
        "draw":     "Clase minoritaria — más difícil (empates son raros)",
        "away_win": "Clase minoritaria — segundo más difícil",
    }.get(cls, "")
    print(f"  {cls:<14} {val:>8.4f}   {interp}")

# =============================================================================
# BLOQUE 7: VISUALIZACIONES PROFESIONALES
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 7: VISUALIZACIONES PROFESIONALES")
print("═" * 68)

# ─── Figura 1: Panel principal ───────────────────────────────────────────────
fig1 = plt.figure(figsize=(20, 16))
fig1.patch.set_facecolor("white")
gs_layout = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.5, wspace=0.38)
fig1.suptitle(
    "Fase 3 — Random Forest: Evaluación Completa\nCopa Mundial FIFA 1930–2022",
    fontsize=15, fontweight="bold", color=PALETTE["primary"], y=0.98,
)

# ── 7a. Matriz de confusión baseline ─────────────────────────────────────────
ax1 = fig1.add_subplot(gs_layout[0, 0])
cm_base = np.array(baseline["metrics"]["test"]["confusion_matrix"])
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax1, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax1.set_title("Confusión — Baseline\n(Test 2018–2022)", fontsize=10, fontweight="bold")
ax1.set_xlabel("Predicho", fontsize=9)
ax1.set_ylabel("Real", fontsize=9)
ax1.tick_params(axis="x", rotation=30, labelsize=8)
ax1.tick_params(axis="y", rotation=0,  labelsize=8)

# ── 7b. Matriz de confusión tuned ────────────────────────────────────────────
ax2 = fig1.add_subplot(gs_layout[0, 1])
cm_tuned = np.array(tuned_metrics["test"]["confusion_matrix"])
sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax2, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax2.set_title("Confusión — Tuned\n(Test 2018–2022)", fontsize=10, fontweight="bold")
ax2.set_xlabel("Predicho", fontsize=9)
ax2.set_ylabel("Real", fontsize=9)
ax2.tick_params(axis="x", rotation=30, labelsize=8)
ax2.tick_params(axis="y", rotation=0,  labelsize=8)

# ── 7c. F1 por clase: baseline vs tuned ──────────────────────────────────────
ax3 = fig1.add_subplot(gs_layout[0, 2])
f1_base  = [baseline["metrics"]["test"]["f1_per_class"][c] for c in CLASS_NAMES]
f1_tuned = [tuned_metrics["test"]["f1_per_class"][c] for c in CLASS_NAMES]
x  = np.arange(len(CLASS_NAMES))
w  = 0.35
b1 = ax3.bar(x - w/2, f1_base,  w, label="Baseline", color=PALETTE["neutral"], alpha=0.85)
b2 = ax3.bar(x + w/2, f1_tuned, w, label="Tuned",    color=PALETTE["primary"],  alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(CLASS_NAMES, rotation=15, fontsize=9)
ax3.set_ylabel("F1-Score", fontsize=9)
ax3.set_ylim(0, 1.05)
ax3.set_title("F1 por Clase\nBaseline vs Tuned (Test)", fontsize=10, fontweight="bold")
ax3.legend(fontsize=9)
for bar in b1:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# ── 7d. Feature Importance Top 15 ────────────────────────────────────────────
ax4 = fig1.add_subplot(gs_layout[1, :2])
top15 = feature_imp.head(15).copy()
colors_imp = [PALETTE["primary"]] * 15
ax4.barh(
    top15["feature_clean"][::-1],
    top15["importance"][::-1],
    xerr=top15["std"][::-1],
    color=colors_imp, alpha=0.85,
    error_kw={"ecolor": PALETTE["neutral"], "capsize": 3, "linewidth": 1},
)
ax4.set_xlabel("Importancia (Mean Decrease Impurity)", fontsize=9)
ax4.set_title("Top 15 Features más Importantes — Random Forest Tuned",
              fontsize=10, fontweight="bold")
ax4.tick_params(axis="y", labelsize=8)
for i, (_, row) in enumerate(top15[::-1].iterrows()):
    ax4.text(row["importance"] + row["std"] + 0.002, i,
             f"{row['importance']:.3f}", va="center", fontsize=7.5)

# ── 7e. CV temporal: accuracy por fold ───────────────────────────────────────
ax5 = fig1.add_subplot(gs_layout[1, 2])
cv_acc_train = cv_result["cv_results"]["train_accuracy"]
cv_acc_val   = cv_result["cv_results"]["test_accuracy"]
folds = np.arange(1, len(cv_acc_train) + 1)
ax5.plot(folds, cv_acc_train, "o-", color=PALETTE["primary"],
         label="Train", linewidth=2, markersize=7)
ax5.plot(folds, cv_acc_val,   "s--", color=PALETTE["secondary"],
         label="Validación", linewidth=2, markersize=7)
ax5.fill_between(folds, cv_acc_val - cv_result["cv_results"]["test_accuracy"].std(),
                 cv_acc_val + cv_result["cv_results"]["test_accuracy"].std(),
                 alpha=0.15, color=PALETTE["secondary"])
ax5.set_xticks(folds)
ax5.set_xticklabels([f"Fold {i}" for i in folds], fontsize=8)
ax5.set_ylabel("Accuracy", fontsize=9)
ax5.set_ylim(0, 1.05)
ax5.set_title("CV Temporal — Accuracy\nTrain vs Validación por Fold",
              fontsize=10, fontweight="bold")
ax5.legend(fontsize=9)
ax5.axhline(cv_acc_val.mean(), color=PALETTE["accent"], linestyle=":",
            alpha=0.7, label=f"Media val: {cv_acc_val.mean():.3f}")

# ── 7f. Comparativa métricas globales ────────────────────────────────────────
ax6 = fig1.add_subplot(gs_layout[2, :])
metric_names = ["Accuracy", "Precision_W", "Recall_W", "F1_Weighted", "F1_Macro"]
metric_keys  = ["accuracy", "precision_w", "recall_w", "f1_weighted", "f1_macro"]

vals = {
    "Baseline Train": [baseline["metrics"]["train"][k] for k in metric_keys],
    "Baseline Test":  [baseline["metrics"]["test"][k]  for k in metric_keys],
    "Tuned Train":    [tuned_metrics["train"][k] for k in metric_keys],
    "Tuned Test":     [tuned_metrics["test"][k]  for k in metric_keys],
}
bar_colors = [PALETTE["neutral"], PALETTE["secondary"],
              PALETTE["primary"], PALETTE["win"]]
x = np.arange(len(metric_names))
w = 0.2
for i, (label_bar, v) in enumerate(vals.items()):
    offset = (i - 1.5) * w
    bars = ax6.bar(x + offset, v, w, label=label_bar,
                   color=bar_colors[i], alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax6.set_xticks(x)
ax6.set_xticklabels(metric_names, fontsize=10)
ax6.set_ylim(0, 1.15)
ax6.set_ylabel("Score", fontsize=10)
ax6.set_title("Comparativa Global — Baseline vs Tuned (Train y Test)",
              fontsize=11, fontweight="bold")
ax6.legend(fontsize=9, loc="upper right", ncol=4)

path_fig1 = os.path.join(OUTPUT_DIR, "rf_evaluacion_completa.png")
fig1.savefig(path_fig1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Figura 1 guardada: {path_fig1}")

# ─── Figura 2: Análisis de clases y overfitting ──────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.patch.set_facecolor("white")
fig2.suptitle("Análisis de Desbalance de Clases y Overfitting — RF Tuned",
              fontsize=13, fontweight="bold", color=PALETTE["primary"])

# 2a. Distribución de predicciones vs real
ax = axes[0]
classes_idx = np.arange(len(CLASS_NAMES))
real_dist  = [np.sum(np.array(y_test) == i) for i in range(3)]
pred_dist  = [np.sum(y_pred_test_tuned == i) for i in range(3)]
ax.bar(classes_idx - 0.2, real_dist, 0.4, label="Real",
       color=PALETTE["primary"], alpha=0.85)
ax.bar(classes_idx + 0.2, pred_dist, 0.4, label="Predicho",
       color=PALETTE["secondary"], alpha=0.85)
ax.set_xticks(classes_idx)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_ylabel("Nº de partidos", fontsize=10)
ax.set_title("Distribución Real vs Predicha\n(Test 2018–2022)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

# 2b. Probabilidades de predicción para clase "draw"
ax = axes[1]
proba = best_pipeline.predict_proba(X_test)
proba_draw = proba[:, 1]  # índice 1 = draw
y_test_arr = np.array(y_test)
ax.hist(proba_draw[y_test_arr == 1], bins=15, alpha=0.7,
        color=PALETTE["draw"], label="Real: draw", density=True)
ax.hist(proba_draw[y_test_arr != 1], bins=15, alpha=0.5,
        color=PALETTE["neutral"], label="Real: no draw", density=True)
ax.axvline(0.5, color=PALETTE["accent"], linestyle="--", linewidth=2,
           label="Umbral 0.5")
ax.set_xlabel("Probabilidad predicha (draw)", fontsize=10)
ax.set_ylabel("Densidad", fontsize=10)
ax.set_title("Distribución de P(draw)\nSegún resultado real", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

# 2c. Train vs Test accuracy por epocas (n_estimators acumulados)
ax = axes[2]
rf_clf = best_pipeline.named_steps["classifier"]
X_tr_t = best_pipeline.named_steps["preprocessor"].transform(X_train)
X_te_t = best_pipeline.named_steps["preprocessor"].transform(X_test)

staged_train, staged_test = [], []
step = max(1, rf_clf.n_estimators // 20)
checkpoints = list(range(step, rf_clf.n_estimators + 1, step))

for n in checkpoints:
    preds_tr = np.array([t.predict(X_tr_t) for t in rf_clf.estimators_[:n]]).astype(int)
    preds_te = np.array([t.predict(X_te_t) for t in rf_clf.estimators_[:n]]).astype(int)
    vote_tr  = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=3).argmax(), 0, preds_tr)
    vote_te  = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=3).argmax(), 0, preds_te)
    from sklearn.metrics import accuracy_score as _acc
    staged_train.append(_acc(y_train, vote_tr))
    staged_test.append(_acc(y_test,   vote_te))

ax.plot(checkpoints, staged_train, "-", color=PALETTE["primary"],
        label="Train", linewidth=2)
ax.plot(checkpoints, staged_test,  "--", color=PALETTE["accent"],
        label="Test", linewidth=2)
ax.fill_between(checkpoints,
                [t - v for t, v in zip(staged_train, staged_test)],
                [0]*len(checkpoints),
                alpha=0.1, color=PALETTE["accent"], label="Gap (overfitting)")
ax.set_xlabel("Nº de árboles acumulados", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_title("Curva de Aprendizaje\n(n_estimators acumulados)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.3, 1.0)

plt.tight_layout()
path_fig2 = os.path.join(OUTPUT_DIR, "rf_clases_overfitting.png")
fig2.savefig(path_fig2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Figura 2 guardada: {path_fig2}")

# =============================================================================
# BLOQUE 8: SERIALIZACIÓN
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 8: SERIALIZACIÓN DE ARTEFACTOS")
print("═" * 68)

save_phase3_artifacts(
    best_pipeline=best_pipeline,
    baseline_result=baseline,
    tuned_result=gs_result,
    cv_result=cv_result,
    feature_imp=feature_imp,
    paths=DEFAULT_PATHS,
)

# =============================================================================
# BLOQUE 9: ANÁLISIS TÉCNICO Y CONCLUSIONES
# =============================================================================
print("\n" + "═" * 68)
print("  BLOQUE 9: ANÁLISIS TÉCNICO Y CONCLUSIONES")
print("═" * 68)

b_test_acc = baseline["metrics"]["test"]["accuracy"]
t_test_acc = tuned_metrics["test"]["accuracy"]
b_f1w      = baseline["metrics"]["test"]["f1_weighted"]
t_f1w      = tuned_metrics["test"]["f1_weighted"]
draw_f1    = tuned_metrics["test"]["f1_per_class"]["draw"]
ov_gap     = tuned_metrics["train"]["accuracy"] - tuned_metrics["test"]["accuracy"]

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  ANÁLISIS TÉCNICO — FASE 3                                      │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  1. DESEMPEÑO GENERAL                                           │
  │     Accuracy  Baseline: {b_test_acc:.4f}  →  Tuned: {t_test_acc:.4f}           │
  │     F1-Weight Baseline: {b_f1w:.4f}  →  Tuned: {t_f1w:.4f}           │
  │                                                                 │
  │  2. CLASE "DRAW" (empate — clase crítica)                       │
  │     F1-draw tuned: {draw_f1:.4f}                                    │
  │     El empate es la clase más difícil de predecir porque:       │
  │       • Es la clase minoritaria en el dataset                   │
  │       • Los indicadores históricos (win_rate, delta) no la      │
  │         diferencian bien — el empate es un resultado "neutro"   │
  │       • class_weight="balanced" ayuda parcialmente              │
  │                                                                 │
  │  3. OVERFITTING                                                 │
  │     Gap Train-Test Accuracy: {ov_gap:.4f}                           │
  │     {'⚠️ Moderado: el modelo memoriza parte del train' if ov_gap > 0.15 else '✓ Controlado: brecha train-test aceptable'}  │
  │     Mitigación: max_depth limita la profundidad de los árboles  │
  │                                                                 │
  │  4. IMPORTANCIA DE FEATURES                                     │
  │     Las features de rendimiento histórico (hist_win_rate,       │
  │     delta_win_rate) dominan sobre el contexto del partido.      │
  │     Year captura la evolución del fútbol a través del tiempo.   │
  │                                                                 │
  │  5. LIMITACIONES DEL DATASET                                    │
  │     • Solo 964 partidos en 92 años → muestra pequeña            │
  │     • Equipos históricos (URSS, Yugoslavia) sin continuidad     │
  │     • Primer partido de cada equipo sin historial → NaN→0       │
  │     • Ventaja de local no capturada (la Copa es en sede única)  │
  │     • Sin datos de jugadores individuales (Messi, Ronaldo, etc) │
  │                                                                 │
  │  6. COMPATIBILIDAD FASTAPI                                      │
  │     Pipeline completo serializado en:                           │
  │     models/random_forest_pipeline.pkl                           │
  │     Uso: pipeline.predict(X_new_raw_dataframe)                  │
  └─────────────────────────────────────────────────────────────────┘
""")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FASE 3 COMPLETADA                                                   ║
║                                                                      ║
║  Artefactos generados:                                               ║
║    models/random_forest_model.pkl      ← Clasificador RF             ║
║    models/random_forest_pipeline.pkl   ← Pipeline completo FastAPI   ║
║    models/random_forest_metrics.json   ← Métricas y metadatos        ║
║    outputs/phase3/rf_evaluacion_completa.png                         ║
║    outputs/phase3/rf_clases_overfitting.png                          ║
║                                                                      ║
║  → Próximo paso: FASE 4 — K-Means (Clustering)                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
