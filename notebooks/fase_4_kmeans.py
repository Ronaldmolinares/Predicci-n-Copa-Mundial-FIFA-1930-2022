"""
==============================================================================
FASE 4 — NOTEBOOK EJECUTABLE: K-MEANS CLUSTERING (v2 corregido)
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  notebooks/fase_4_kmeans.py

Correcciones v2:
    - K seleccionado por Silhouette maximo directo (no heuristico del codo)
    - Comparativa explicita K=3 vs K=4
    - Naming dinamico basado en estadisticas reales del centroide
    - Eliminada sobresegmentacion K=11

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
==============================================================================
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kmeans_modeling import (
    load_kmeans_artifacts,
    fit_scaler,
    evaluate_k_range,
    select_optimal_k,
    compare_k_candidates,
    train_kmeans,
    build_final_kmeans_pipeline,
    assign_clusters,
    describe_clusters,
    apply_pca,
    apply_tsne,
    save_phase4_artifacts,
    DEFAULT_PATHS,
    K_RANGE,
)

# ── Paleta y estilos ─────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1a3a5c",
    "secondary": "#c8a951",
    "accent":    "#c0392b",
    "bg":        "#f8f9fa",
}
# Paleta segura para cualquier K ≤ 9
CLUSTER_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#e91e63",
]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.facecolor":    PALETTE["bg"],
    "figure.facecolor":  "white",
})

OUTPUT_DIR = "outputs/phase4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
print("""
FASE 4: CLUSTERING NO SUPERVISADO - K-MEANS (v2 corregido)
Copa Mundial FIFA 1930-2022
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
print("  BLOQUE 2: ESCALADO StandardScaler")
print("=" * 68)
print("""
  Justificacion:
    K-Means minimiza WCSS (distancia euclidiana al cuadrado).
    total_matches rango [1, 114] vs win_rate rango [0, 0.67]:
    sin escalar, total_matches DOMINA la distancia euclidiana.
    StandardScaler: z = (x - media) / std -> media=0, std=1
    Cada feature contribuye EQUITATIVAMENTE a la distancia.
""")

scaler, X_scaled = fit_scaler(X_kmeans)

# =============================================================================
# BLOQUE 3: EVALUACION K OPTIMO
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 3: EVALUACION DEL K OPTIMO")
print("=" * 68)
print("""
  Criterio principal: Silhouette Score MAXIMO
  Correccion v2: ya no se usa el heuristico del codo (producia K=11
  por acumulacion de candidatos tardios donde la inercia ya estabilizo).
  El codo se reporta solo como referencia visual.
""")

df_metrics = evaluate_k_range(X_scaled, k_range=K_RANGE)
optimal_k  = select_optimal_k(df_metrics)

# Comparativa K=3 vs K=4 (candidatos mas probables para datos FIFA)
print("\n" + "-" * 68)
print("  COMPARATIVA EXPLICITA: K=3 vs K=4")
print("-" * 68)
compare_k_candidates(X_scaled, [3, 4], df_metrics)

# =============================================================================
# BLOQUE 4: ENTRENAMIENTO
# =============================================================================
print("\n" + "=" * 68)
print(f"  BLOQUE 4: ENTRENAMIENTO K-MEANS FINAL (K={optimal_k})")
print("=" * 68)
print(f"""
  Configuracion final:
    K         = {optimal_k}  (argmax Silhouette Score)
    init      = k-means++  (inicializacion inteligente)
    n_init    = 20         (20 corridas, se conserva la mejor)
    max_iter  = 500        (convergencia garantizada)
    n=86 equipos   (suficiente para K en [3,5] — ~17-28 por cluster)
""")

km     = train_kmeans(X_scaled, k=optimal_k)
labels = km.labels_

final_pipeline = build_final_kmeans_pipeline(scaler, km)

# =============================================================================
# BLOQUE 5: ASIGNACION E INTERPRETACION
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 5: ASIGNACION E INTERPRETACION (naming dinamico)")
print("=" * 68)
print("""
  Correccion v2: los nombres de cluster se derivan de los valores
  reales del centroide (win_rate, avg_goal_diff, num_tournaments,
  best_stage), NO de un diccionario fijo que asignaba "Selecciones Dominantes"
  a clusters con Bolivia, Haiti y Zaire.
""")

df_clustered = assign_clusters(
    profile_df, team_names, labels, km, X_scaled, features
)
centroids_df, cluster_labels = describe_clusters(
    df_clustered, features, km, scaler
)

# Agregar etiqueta al dataframe
df_clustered["cluster_label"] = df_clustered["cluster"].map(
    lambda c: cluster_labels.get(int(c), f"Cluster {c}")
)

sil_score = silhouette_score(X_scaled, labels)

# =============================================================================
# BLOQUE 6: REDUCCION DE DIMENSIONALIDAD
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 6: PCA 2D + t-SNE")
print("=" * 68)

X_pca, pca_model, var_exp = apply_pca(X_scaled, n_components=2)
perp  = min(15.0, len(team_names) / 4)
X_tsne = apply_tsne(X_scaled, perplexity=perp)

loadings = pd.DataFrame(pca_model.components_.T, index=features,
                        columns=["PC1", "PC2"])
print(f"\n  Cargas PCA:")
print(loadings.round(3).to_string())

# =============================================================================
# BLOQUE 7: VISUALIZACIONES
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 7: VISUALIZACIONES PROFESIONALES")
print("=" * 68)

cluster_ids = sorted(df_clustered["cluster"].unique())
color_map   = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
               for i, c in enumerate(cluster_ids)}

# ─── Figura 1: Seleccion de K ────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor("white")
fig1.suptitle(
    f"Seleccion del K Optimo — K-Means v2\nCriterio: Silhouette Score Maximo -> K={optimal_k}",
    fontsize=13, fontweight="bold", color=PALETTE["primary"],
)

# 1a. Elbow
ax = axes1[0, 0]
ax.plot(df_metrics["k"], df_metrics["inertia"], "o-",
        color=PALETTE["primary"], lw=2.5, ms=8)
ax.axvline(optimal_k, color=PALETTE["accent"], ls="--", lw=2,
           label=f"K optimo = {optimal_k}")
ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Inercia (WCSS)", fontsize=10)
ax.set_title("Elbow Method (referencia visual)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.set_xticks(list(K_RANGE))

# 1b. Silhouette — marcado el maximo
ax = axes1[0, 1]
ax.plot(df_metrics["k"], df_metrics["silhouette"], "s-",
        color="#2ecc71", lw=2.5, ms=8)
ax.axvline(optimal_k, color=PALETTE["accent"], ls="--", lw=2,
           label=f"K optimo = {optimal_k}")
best_sil = float(df_metrics[df_metrics["k"] == optimal_k]["silhouette"].values[0])
ax.scatter([optimal_k], [best_sil], s=200, zorder=5,
           color=PALETTE["accent"], edgecolors="white", lw=2)
ax.annotate(f"Max = {best_sil:.4f}", (optimal_k, best_sil),
            xytext=(optimal_k + 0.3, best_sil + 0.01), fontsize=9,
            color=PALETTE["accent"], fontweight="bold")
ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Silhouette Score", fontsize=10)
ax.set_title("Silhouette Score (CRITERIO PRINCIPAL)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.set_xticks(list(K_RANGE))

# 1c. Calinski-Harabasz
ax = axes1[1, 0]
ax.bar(df_metrics["k"], df_metrics["calinski"],
       color=PALETTE["secondary"], alpha=0.85, edgecolor="white")
ax.axvline(optimal_k, color=PALETTE["accent"], ls="--", lw=2,
           label=f"K optimo = {optimal_k}")
ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Calinski-Harabasz", fontsize=10)
ax.set_title("Calinski-Harabasz (mayor = mejor)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)

# 1d. Davies-Bouldin
ax = axes1[1, 1]
ax.bar(df_metrics["k"], df_metrics["davies"],
       color=PALETTE["accent"], alpha=0.8, edgecolor="white")
ax.axvline(optimal_k, color=PALETTE["primary"], ls="--", lw=2,
           label=f"K optimo = {optimal_k}")
ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Davies-Bouldin", fontsize=10)
ax.set_title("Davies-Bouldin (menor = mejor)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(OUTPUT_DIR, "kmeans_k_selection.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig1)
print(f"  Figura 1: {p1}")

# ─── Figura 2: PCA + t-SNE scatter ──────────────────────────────────────────
ncols  = 2 if X_tsne is not None else 1
fig2, axes2 = plt.subplots(1, ncols, figsize=(9 * ncols, 8))
if ncols == 1:
    axes2 = [axes2]
fig2.patch.set_facecolor("white")
fig2.suptitle(
    f"Perfiles Competitivos FIFA — K-Means K={optimal_k}\n"
    f"Naming dinamico basado en centroides reales",
    fontsize=13, fontweight="bold", color=PALETTE["primary"],
)

panels = [(X_pca, f"PCA 2D ({var_exp[0]*100:.1f}%+{var_exp[1]*100:.1f}%)")]
if X_tsne is not None:
    panels.append((X_tsne, "t-SNE 2D"))

for ax, (X_2d, title) in zip(axes2, panels):
    for c in cluster_ids:
        mask  = df_clustered["cluster"].values == c
        label = cluster_labels.get(int(c), f"Cluster {c}")
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color_map[c], s=100, alpha=0.85,
                   edgecolors="white", lw=0.8, zorder=3,
                   label=f"C{c}: {label} ({mask.sum()})")
        # Etiquetar top 2 por win_rate
        sub_top = df_clustered[df_clustered["cluster"] == c].nlargest(2, "win_rate")
        for _, row in sub_top.iterrows():
            idx = team_names.index(row["team"])
            ax.annotate(row["team"], (X_2d[idx, 0], X_2d[idx, 1]),
                        fontsize=6.5, color=color_map[c], fontweight="bold",
                        xytext=(3, 3), textcoords="offset points")

    if title.startswith("PCA"):
        c2d = pca_model.transform(km.cluster_centers_)
        ax.scatter(c2d[:, 0], c2d[:, 1], c="black", s=220, marker="X",
                   zorder=5, edgecolors="white", lw=1.5, label="Centroides")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.92])
p2 = os.path.join(OUTPUT_DIR, "kmeans_pca_tsne.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig2)
print(f"  Figura 2: {p2}")

# ─── Figura 3: Silhouette por equipo + Heatmap centroides ───────────────────
fig3 = plt.figure(figsize=(20, 9))
fig3.patch.set_facecolor("white")
gs3  = gridspec.GridSpec(1, 2, figure=fig3, wspace=0.38)
fig3.suptitle("Silhouette Individual y Heatmap de Centroides",
              fontsize=13, fontweight="bold", color=PALETTE["primary"])

ax3a = fig3.add_subplot(gs3[0, 0])
y_lo = 10
for c in cluster_ids:
    sub  = df_clustered[df_clustered["cluster"] == c].sort_values("silhouette_val")
    vals = sub["silhouette_val"].values
    y_hi = y_lo + len(vals)
    ax3a.fill_betweenx(np.arange(y_lo, y_hi), 0, vals,
                       facecolor=color_map[c], alpha=0.75)
    lbl = cluster_labels.get(int(c), f"C{c}")[:14]
    ax3a.text(-0.05, (y_lo + y_hi) / 2, f"C{c}\n{lbl}",
              color=color_map[c], fontsize=7, fontweight="bold", va="center")
    y_lo = y_hi + 5

ax3a.axvline(sil_score, color=PALETTE["accent"], ls="--", lw=2,
             label=f"Sil. medio = {sil_score:.3f}")
ax3a.set_xlabel("Silhouette Score individual", fontsize=10)
ax3a.set_title(f"Silhouette por Equipo (K={optimal_k})", fontsize=11, fontweight="bold")
ax3a.legend(fontsize=10); ax3a.set_xlim(-0.35, 1.0); ax3a.set_yticks([])

ax3b = fig3.add_subplot(gs3[0, 1])
idx_labels = [f"C{i}: {cluster_labels.get(i,'')[:14]}" for i in range(optimal_k)]
cent_df_plot = pd.DataFrame(km.cluster_centers_, columns=features, index=idx_labels)
sns.heatmap(cent_df_plot, ax=ax3b, cmap="RdYlGn", center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "z-score", "shrink": 0.8},
            annot_kws={"size": 8})
ax3b.set_title("Centroides — z-score (escala estandarizada)", fontsize=11, fontweight="bold")
ax3b.tick_params(axis="x", rotation=35, labelsize=8)
ax3b.tick_params(axis="y", rotation=0, labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p3 = os.path.join(OUTPUT_DIR, "kmeans_silhouette_heatmap.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig3)
print(f"  Figura 3: {p3}")

# ─── Figura 4: Boxplots por cluster ─────────────────────────────────────────
key_feats = ["win_rate", "avg_goal_diff", "num_tournaments", "best_stage", "total_matches"]
fig4, axes4 = plt.subplots(1, len(key_feats), figsize=(20, 6))
fig4.patch.set_facecolor("white")
fig4.suptitle("Distribucion de Features Clave por Cluster",
              fontsize=13, fontweight="bold", color=PALETTE["primary"])

for ax, feat in zip(axes4, key_feats):
    data = [df_clustered[df_clustered["cluster"] == c][feat].values
            for c in cluster_ids]
    bp   = ax.boxplot(data, patch_artist=True,
                      medianprops={"color": "black", "lw": 2},
                      whiskerprops={"lw": 1.5}, capprops={"lw": 1.5},
                      flierprops={"marker": "o", "ms": 4, "alpha": 0.6})
    for patch, c in zip(bp["boxes"], cluster_ids):
        patch.set_facecolor(color_map[c]); patch.set_alpha(0.8)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=9)
    ax.set_title(feat.replace("_", "\n"), fontsize=9, fontweight="bold")
    ax.set_ylabel("Valor", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p4 = os.path.join(OUTPUT_DIR, "kmeans_features_boxplot.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(fig4)
print(f"  Figura 4: {p4}")

# ─── Figura 5: Ranking de equipos por cluster ────────────────────────────────
fig5, axes5 = plt.subplots(1, optimal_k, figsize=(5 * optimal_k, 6))
if optimal_k == 1:
    axes5 = [axes5]
fig5.patch.set_facecolor("white")
fig5.suptitle(f"Equipos por Cluster — K={optimal_k}  (ordenados por win_rate)",
              fontsize=13, fontweight="bold", color=PALETTE["primary"])

for ax, c in zip(axes5, cluster_ids):
    sub   = df_clustered[df_clustered["cluster"] == c].sort_values(
        "win_rate", ascending=True
    )
    label = cluster_labels.get(int(c), f"Cluster {c}")
    ax.barh(sub["team"], sub["win_rate"], color=color_map[c], alpha=0.85,
            edgecolor="white")
    ax.set_title(f"C{c}: {label}\n({len(sub)} equipos)",
                 fontsize=9, fontweight="bold", color=color_map[c])
    ax.set_xlabel("Win Rate", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, 0.8)
    mean_wr = sub["win_rate"].mean()
    ax.axvline(mean_wr, color="black", ls="--", lw=1.2, alpha=0.6,
               label=f"Media={mean_wr:.2f}")
    ax.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p5 = os.path.join(OUTPUT_DIR, "kmeans_teams_by_cluster.png")
fig5.savefig(p5, dpi=150, bbox_inches="tight"); plt.close(fig5)
print(f"  Figura 5: {p5}")

# =============================================================================
# BLOQUE 8: SERIALIZACIÓN
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 8: SERIALIZACION")
print("=" * 68)

save_phase4_artifacts(
    km=km,
    final_pipeline=final_pipeline,
    df_clustered=df_clustered,
    centroids_df=centroids_df,
    cluster_labels=cluster_labels,
    df_metrics=df_metrics,
    optimal_k=optimal_k,
    sil_score=sil_score,
    paths=DEFAULT_PATHS,
)

# =============================================================================
# BLOQUE 9: REPORTE FINAL
# =============================================================================
print("\n" + "=" * 68)
print("  BLOQUE 9: REPORTE FINAL")
print("=" * 68)

print(f"""
  CLUSTERING K-MEANS — Copa Mundial FIFA 1930-2022

  Configuracion final:
    K optimo           : {optimal_k}
    Silhouette Score   : {sil_score:.4f}
    Criterio seleccion : argmax(Silhouette) — metodo directo
    Equipos            : {len(team_names)}
    Features           : {len(features)}
    Escalado           : StandardScaler (z-score)
    Inicializacion     : k-means++ (n_init=20)

  Por que K={optimal_k} y no K=11:
    K=11 producia microsegmentos de 3-4 equipos, clusters con
    Silhouette negativo y nombres sin coherencia semantica.
    K={optimal_k} maximiza la metrica objetiva (Silhouette) y genera
    grupos de ~{len(team_names)//optimal_k}-{len(team_names)//optimal_k+8} selecciones con perfil interpretable.

  Clusters generados:
""")

for c in cluster_ids:
    sub   = df_clustered[df_clustered["cluster"] == c]
    label = cluster_labels.get(int(c), f"Cluster {c}")
    teams = sorted(sub["team"].tolist())
    print(f"  Cluster {c}: {label} ({len(sub)} selecciones)")
    print(f"    Win rate medio   : {sub['win_rate'].mean():.3f}")
    print(f"    Dif. goles media : {sub['avg_goal_diff'].mean():.3f}")
    print(f"    Torneos promedio : {sub['num_tournaments'].mean():.1f}")
    print(f"    Mejor fase media : {sub['best_stage'].mean():.2f}")
    print(f"    Silhouette medio : {sub['silhouette_val'].mean():.4f}")
    print(f"    Equipos: {', '.join(teams[:10])}{'...' if len(teams) > 10 else ''}")
    print()

print("""
  Metodologia (decisiones tecnicas):
    1. StandardScaler obligatorio: K-Means usa distancia euclidiana.
    2. k-means++ + n_init=20: evita optimos locales pobres.
    3. Silhouette Score como criterio PRIMARIO (no el codo visual).
    4. Naming dinamico: etiquetas derivadas de valores reales del centroide.
    5. Sin leakage: K-Means usa perfil historico completo por equipo.
    6. Pipeline serializado: compatible con FastAPI.

FASE 4 COMPLETADA
""")
