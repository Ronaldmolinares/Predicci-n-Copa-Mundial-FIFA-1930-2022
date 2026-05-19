"""
==============================================================================
FASE 4 — NOTEBOOK EJECUTABLE: K-MEANS CLUSTERING
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  notebooks/fase_4_kmeans.py

Flujo:
    1. Carga artefactos Fase 2
    2. Escalado StandardScaler
    3. Evaluación K óptimo (Elbow + Silhouette + CH + DB)
    4. Entrenamiento modelo final
    5. Asignación e interpretación de clusters
    6. PCA 2D + t-SNE
    7. Visualizaciones profesionales
    8. Serialización artefactos

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
==============================================================================
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kmeans_modeling import (
    load_kmeans_artifacts,
    fit_scaler,
    evaluate_k_range,
    select_optimal_k,
    train_kmeans,
    build_final_kmeans_pipeline,
    assign_clusters,
    describe_clusters,
    apply_pca,
    apply_tsne,
    save_phase4_artifacts,
    CLUSTER_LABELS,
    DEFAULT_PATHS,
    K_RANGE,
)

# ── Paleta y estilo ──────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1a3a5c",
    "secondary": "#c8a951",
    "accent":    "#c0392b",
    "bg":        "#f8f9fa",
}
CLUSTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#e91e63", "#00bcd4", "#8bc34a", "#ff5722",
]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.facecolor":   PALETTE["bg"],
    "figure.facecolor": "white",
})

OUTPUT_DIR = "outputs/phase4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║         FASE 4: CLUSTERING NO SUPERVISADO — K-MEANS                  ║
║         Copa Mundial FIFA 1930–2022                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# BLOQUE 1: CARGA
# =============================================================================
print("=" * 68)
print("  BLOQUE 1: CARGA DE ARTEFACTOS")
print("=" * 68)

arts       = load_kmeans_artifacts()
X_kmeans   = arts["X_kmeans"]
team_names = arts["team_names"]
profile_df = arts["profile_df"]
features   = arts["features"]

# =============================================================================
# BLOQUE 2: ESCALADO
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 2: ESCALADO (StandardScaler)")
print("=" * 68)
print("""
  K-Means minimiza inercia euclidiana. Sin escalar, variables de
  magnitud alta (total_matches: 1-114) dominan sobre variables de
  magnitud baja (win_rate: 0-0.67), produciendo clusters sesgados.
  StandardScaler garantiza contribucion equitativa de cada feature.
""")

scaler_pipeline, X_scaled = fit_scaler(X_kmeans)

# =============================================================================
# BLOQUE 3: SELECCION K OPTIMO
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 3: SELECCION DEL K OPTIMO")
print("=" * 68)
print("""
  Estrategia multi-criterio:
    Elbow Method  → identifica punto de retorno decreciente en inercia
    Silhouette    → cuantifica cohesion y separacion [-1, 1]
    Calinski-H    → ratio dispersión inter/intra cluster (mayor=mejor)
    Davies-Bouldin → similitud entre clusters adyacentes (menor=mejor)
""")

df_metrics = evaluate_k_range(X_scaled, k_range=K_RANGE)
optimal_k  = select_optimal_k(df_metrics)

# =============================================================================
# BLOQUE 4: ENTRENAMIENTO FINAL
# =============================================================================
print("\n" + "=" * 68)
print(f"  BLOQUE 4: ENTRENAMIENTO K-MEANS (K={optimal_k})")
print("=" * 68)

km = train_kmeans(X_scaled, k=optimal_k)
labels = km.labels_

# Pipeline completo para FastAPI
final_pipeline = build_final_kmeans_pipeline(scaler_pipeline, km)

# =============================================================================
# BLOQUE 5: ASIGNACION E INTERPRETACION
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 5: ASIGNACION E INTERPRETACION DE CLUSTERS")
print("=" * 68)

df_clustered  = assign_clusters(
    profile_df, team_names, labels, km, X_scaled, features
)
centroids_df  = describe_clusters(df_clustered, features, km, scaler_pipeline)

# Silhouette score global
from sklearn.metrics import silhouette_score as _sil
sil_score = _sil(X_scaled, labels)

# =============================================================================
# BLOQUE 6: REDUCCION DE DIMENSIONALIDAD
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 6: PCA 2D + t-SNE")
print("=" * 68)

X_pca, pca_model, var_exp = apply_pca(X_scaled, n_components=2)
X_tsne = apply_tsne(X_scaled, perplexity=min(10.0, len(team_names) / 4))

# Cargas de componentes PCA
loadings = pd.DataFrame(
    pca_model.components_.T,
    index=features,
    columns=["PC1", "PC2"],
)
print(f"\n  Cargas PCA (contribucion de cada feature):")
print(loadings.round(3).to_string())

# =============================================================================
# BLOQUE 7: VISUALIZACIONES PROFESIONALES
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 7: VISUALIZACIONES PROFESIONALES")
print("=" * 68)

cluster_ids = sorted(df_clustered["cluster"].unique())
colors      = CLUSTER_COLORS[:len(cluster_ids)]
color_map   = dict(zip(cluster_ids, colors))

# ─── Figura 1: Panel de seleccion de K ──────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor("white")
fig1.suptitle("Seleccion del K Optimo — K-Means\nCopa Mundial FIFA 1930–2022",
              fontsize=14, fontweight="bold", color=PALETTE["primary"])

# 1a. Elbow Method
ax = axes1[0, 0]
ax.plot(df_metrics["k"], df_metrics["inertia"], "o-",
        color=PALETTE["primary"], linewidth=2.5, markersize=8)
ax.axvline(optimal_k, color=PALETTE["accent"], linestyle="--",
           linewidth=2, label=f"K optimo = {optimal_k}")
ax.set_xlabel("Numero de clusters K", fontsize=10)
ax.set_ylabel("Inercia (WCSS)", fontsize=10)
ax.set_title("Elbow Method", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xticks(list(K_RANGE))

# 1b. Silhouette
ax = axes1[0, 1]
ax.plot(df_metrics["k"], df_metrics["silhouette"], "s-",
        color="#2ecc71", linewidth=2.5, markersize=8)
ax.axvline(optimal_k, color=PALETTE["accent"], linestyle="--",
           linewidth=2, label=f"K optimo = {optimal_k}")
best_sil = df_metrics[df_metrics["k"] == optimal_k]["silhouette"].values[0]
ax.scatter([optimal_k], [best_sil], s=150, zorder=5,
           color=PALETTE["accent"], edgecolors="white", linewidth=2)
ax.set_xlabel("Numero de clusters K", fontsize=10)
ax.set_ylabel("Silhouette Score", fontsize=10)
ax.set_title("Silhouette Score", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xticks(list(K_RANGE))

# 1c. Calinski-Harabasz
ax = axes1[1, 0]
ax.bar(df_metrics["k"], df_metrics["calinski"],
       color=PALETTE["secondary"], alpha=0.85, edgecolor="white")
ax.axvline(optimal_k, color=PALETTE["accent"], linestyle="--",
           linewidth=2, label=f"K optimo = {optimal_k}")
ax.set_xlabel("Numero de clusters K", fontsize=10)
ax.set_ylabel("Calinski-Harabasz", fontsize=10)
ax.set_title("Calinski-Harabasz Index (mayor = mejor)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)

# 1d. Davies-Bouldin
ax = axes1[1, 1]
ax.bar(df_metrics["k"], df_metrics["davies"],
       color=PALETTE["accent"], alpha=0.8, edgecolor="white")
ax.axvline(optimal_k, color=PALETTE["primary"], linestyle="--",
           linewidth=2, label=f"K optimo = {optimal_k}")
ax.set_xlabel("Numero de clusters K", fontsize=10)
ax.set_ylabel("Davies-Bouldin Score", fontsize=10)
ax.set_title("Davies-Bouldin Score (menor = mejor)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(OUTPUT_DIR, "kmeans_k_selection.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Figura 1 guardada: {p1}")

# ─── Figura 2: PCA 2D scatter ────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.patch.set_facecolor("white")
fig2.suptitle(f"Perfiles Competitivos FIFA — K-Means (K={optimal_k})\nCopa Mundial 1930–2022",
              fontsize=14, fontweight="bold", color=PALETTE["primary"])

col_arr = df_clustered["cluster"].map(color_map)

for ax_idx, (ax, X_2d, title) in enumerate(zip(
    axes2,
    [X_pca, X_tsne if X_tsne is not None else X_pca],
    [f"PCA 2D ({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% = {sum(var_exp)*100:.1f}%)",
     "t-SNE 2D" if X_tsne is not None else "PCA 2D (bis)"],
)):
    for c in cluster_ids:
        mask = df_clustered["cluster"].values == c
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color_map[c], s=90, alpha=0.85,
            edgecolors="white", linewidth=0.8,
            label=f"C{c}: {CLUSTER_LABELS.get(c, 'Cluster '+str(c))} ({mask.sum()})",
            zorder=3,
        )
        sub = df_clustered[df_clustered["cluster"] == c].nlargest(3, "win_rate")
        for _, row in sub.iterrows():
            idx_team = team_names.index(row["team"])
            ax.annotate(
                row["team"],
                (X_2d[idx_team, 0], X_2d[idx_team, 1]),
                fontsize=6.5, ha="left", va="bottom",
                color=color_map[c], fontweight="bold",
                xytext=(3, 3), textcoords="offset points",
            )

    # Centroides PCA (solo en PCA)
    if ax_idx == 0:
        centroids_2d = pca_model.transform(km.cluster_centers_)
        ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            c="black", s=200, marker="X", zorder=5,
            edgecolors="white", linewidth=1.5, label="Centroides",
        )

    ax.set_xlabel("Componente 1" if ax_idx == 0 else "t-SNE 1", fontsize=10)
    ax.set_ylabel("Componente 2" if ax_idx == 0 else "t-SNE 2", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p2 = os.path.join(OUTPUT_DIR, "kmeans_pca_tsne.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Figura 2 guardada: {p2}")

# ─── Figura 3: Silhouette por equipo + Radar de centroides ──────────────────
fig3 = plt.figure(figsize=(20, 10))
fig3.patch.set_facecolor("white")
gs3  = gridspec.GridSpec(1, 2, figure=fig3, wspace=0.38)
fig3.suptitle("Analisis de Silhouette por Equipo y Perfil de Centroides",
              fontsize=14, fontweight="bold", color=PALETTE["primary"])

# 3a. Silhouette plot
ax3a = fig3.add_subplot(gs3[0, 0])
y_lower = 10
for c in cluster_ids:
    sub   = df_clustered[df_clustered["cluster"] == c].sort_values("silhouette_val")
    vals  = sub["silhouette_val"].values
    y_upper = y_lower + len(vals)
    ax3a.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                       facecolor=color_map[c], alpha=0.75, edgecolor="none")
    ax3a.text(-0.05, (y_lower + y_upper) / 2,
              f"C{c}", color=color_map[c], fontsize=9, fontweight="bold", va="center")
    y_lower = y_upper + 5

ax3a.axvline(sil_score, color=PALETTE["accent"], linestyle="--", linewidth=2,
             label=f"Sil. medio = {sil_score:.3f}")
ax3a.set_xlabel("Silhouette Score individual", fontsize=10)
ax3a.set_ylabel("Equipos (agrupados por cluster)", fontsize=10)
ax3a.set_title(f"Silhouette por Equipo (K={optimal_k})", fontsize=11, fontweight="bold")
ax3a.legend(fontsize=10)
ax3a.set_xlim(-0.3, 1.0)
ax3a.set_yticks([])

# 3b. Heatmap de centroides normalizados
ax3b = fig3.add_subplot(gs3[0, 1])
centroid_scaled_df = pd.DataFrame(
    km.cluster_centers_,
    columns=features,
    index=[f"C{i}\n{CLUSTER_LABELS.get(i,'')[:12]}" for i in range(optimal_k)],
)
sns.heatmap(
    centroid_scaled_df,
    ax=ax3b,
    cmap="RdYlGn",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "z-score", "shrink": 0.8},
    annot_kws={"size": 8},
)
ax3b.set_title("Centroides (valores estandarizados z-score)", fontsize=11, fontweight="bold")
ax3b.tick_params(axis="x", rotation=35, labelsize=8)
ax3b.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p3 = os.path.join(OUTPUT_DIR, "kmeans_silhouette_heatmap.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Figura 3 guardada: {p3}")

# ─── Figura 4: Distribución de métricas por cluster ─────────────────────────
key_features = ["win_rate", "avg_goal_diff", "num_tournaments", "best_stage", "total_matches"]
fig4, axes4  = plt.subplots(1, len(key_features), figsize=(20, 6))
fig4.patch.set_facecolor("white")
fig4.suptitle("Distribucion de Features Clave por Cluster",
              fontsize=14, fontweight="bold", color=PALETTE["primary"])

for ax, feat in zip(axes4, key_features):
    data_by_cluster = [
        df_clustered[df_clustered["cluster"] == c][feat].values
        for c in cluster_ids
    ]
    bp = ax.boxplot(
        data_by_cluster,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
    )
    for patch, c in zip(bp["boxes"], cluster_ids):
        patch.set_facecolor(color_map[c])
        patch.set_alpha(0.8)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=9)
    ax.set_title(feat.replace("_", "\n"), fontsize=9, fontweight="bold")
    ax.set_ylabel("Valor", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p4 = os.path.join(OUTPUT_DIR, "kmeans_features_boxplot.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Figura 4 guardada: {p4}")

# =============================================================================
# BLOQUE 8: SERIALIZACIÓN
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 8: SERIALIZACION DE ARTEFACTOS")
print("=" * 68)

save_phase4_artifacts(
    km=km,
    final_pipeline=final_pipeline,
    df_clustered=df_clustered,
    centroids_df=centroids_df,
    df_metrics=df_metrics,
    optimal_k=optimal_k,
    sil_score=sil_score,
    paths=DEFAULT_PATHS,
)

# =============================================================================
# BLOQUE 9: REPORTE FINAL
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 9: REPORTE TECNICO FINAL")
print("=" * 68)

print(f"""
  RESUMEN DE CLUSTERING K-MEANS — COPA MUNDIAL FIFA 1930–2022

  Configuracion:
    Equipos analizados  : {len(team_names)}
    Features utilizadas : {len(features)}
    K optimo            : {optimal_k}
    Silhouette Score    : {sil_score:.4f}
    Escalado            : StandardScaler (z-score)
    Inicializacion      : k-means++  (n_init={20})

  Interpretacion de clusters:
""")

for c in cluster_ids:
    sub   = df_clustered[df_clustered["cluster"] == c]
    label = CLUSTER_LABELS.get(c, f"Cluster {c}")
    teams = sorted(sub["team"].tolist())
    wr    = sub["win_rate"].mean()
    gd    = sub["avg_goal_diff"].mean()
    nt    = sub["num_tournaments"].mean()
    bs    = sub["best_stage"].mean()
    sv    = sub["silhouette_val"].mean()
    print(f"  Cluster {c} — {label} ({len(sub)} selecciones)")
    print(f"    Win Rate medio       : {wr:.3f}")
    print(f"    Dif. goles media     : {gd:.3f}")
    print(f"    Torneos promedio     : {nt:.1f}")
    print(f"    Mejor fase promedio  : {bs:.2f}")
    print(f"    Silhouette medio     : {sv:.3f}")
    print(f"    Equipos              : {', '.join(teams[:8])}{'...' if len(teams)>8 else ''}")
    print()

print("""
  Decisiones metodologicas:
    1. StandardScaler obligatorio: K-Means usa distancia euclidiana.
       Sin escalar, total_matches (1-114) domina a win_rate (0-0.67).
    2. k-means++: inicializacion inteligente de centroides, reduce
       sensibilidad a condiciones iniciales aleatorias.
    3. n_init=20: se ejecutan 20 corridas independientes, se conserva
       la de menor inercia, mitigando optimos locales.
    4. Silhouette primo sobre Elbow: el codo es visual y ambiguo;
       Silhouette es una metrica cuantitativa objetiva.
    5. Sin leakage: K-Means usa el perfil historico COMPLETO por
       equipo (no hay target futuro que predecir).

  Compatibilidad FastAPI:
    pipeline.predict(X_new_raw)  -> cluster_id
    Uso: cargar models/kmeans_pipeline.pkl en el endpoint de perfil.
""")

print("""
FASE 4 COMPLETADA

Artefactos generados:
  models/kmeans_model.pkl
  models/kmeans_pipeline.pkl
  models/kmeans_metrics.json
  outputs/phase4/kmeans_clusters.csv
  outputs/phase4/kmeans_k_selection.png
  outputs/phase4/kmeans_pca_tsne.png
  outputs/phase4/kmeans_silhouette_heatmap.png
  outputs/phase4/kmeans_features_boxplot.png

Proximo paso: FASE 5 - API FastAPI
""")
