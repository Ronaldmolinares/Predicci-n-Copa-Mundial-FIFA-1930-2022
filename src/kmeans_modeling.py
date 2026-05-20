"""
==============================================================================
FASE 4 — CLUSTERING NO SUPERVISADO: K-MEANS (v2 — corregido)
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/kmeans_modeling.py

Correcciones v2:
    - select_optimal_k: usa Silhouette máximo directo (sin heurístico
      del codo que producía K=11 por acumulación de candidatos).
    - naming dinámico: etiquetas derivadas de los centroides reales,
      no de un diccionario fijo que asignaba "Élite Mundial" a Bolivia.
    - compare_k_candidates: comparativa explícita K=3 vs K=4.

Estrategia de selección del K óptimo:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Silhouette Score [PRIMARIO]                             │
    │     s(i) = (b - a) / max(a, b)   ∈ [-1, 1]                 │
    │     Mayor = clusters más compactos y separados.             │
    │     Se selecciona el K que MAXIMIZA este valor.             │
    │                                                             │
    │  2. Elbow Method [VISUAL — solo confirmación]               │
    │     La caída de inercia se reporta pero NO decide el K.     │
    │     Es ambiguo y susceptible a acumulación de candidatos.   │
    │                                                             │
    │  3. Calinski-Harabasz y Davies-Bouldin [DESEMPATE]          │
    │     Si dos K tienen Silhouette similar (Δ < 0.02),          │
    │     se usa CH/DB como criterio de desempate.                │
    │                                                             │
    │  4. Interpretabilidad [RESTRICCIÓN]                         │
    │     K ∈ [3, 6] garantiza clusters FIFA interpretables.      │
    │     K > 8 genera microsegmentos sin sentido semántico.      │
    └─────────────────────────────────────────────────────────────┘

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado — UPTC
==============================================================================
"""

import warnings
import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES
# =============================================================================

RANDOM_STATE = 42
K_RANGE      = range(2, 10)     # Rango razonable para 86 equipos
N_INIT       = 20
MAX_ITER     = 500

DEFAULT_PATHS = {
    "model":    "models/kmeans_model.pkl",
    "pipeline": "models/kmeans_pipeline.pkl",
    "metrics":  "models/kmeans_metrics.json",
    "clusters": "outputs/phase4/kmeans_clusters.csv",
}


# =============================================================================
# NAMING DINÁMICO BASADO EN CENTROIDES
# =============================================================================

def _label_cluster_from_centroid(centroid: dict, cluster_id: int) -> str:
    """
    Genera una etiqueta interpretativa a partir del centroide real.

    Lógica basada en win_rate, avg_goal_diff, num_tournaments, best_stage.
    Evita el problema de nombres fijos que no corresponden al contenido.
    """
    wr  = centroid.get("win_rate", 0)
    gd  = centroid.get("avg_goal_diff", 0)
    nt  = centroid.get("num_tournaments", 0)
    bs  = centroid.get("best_stage", 0)
    tm  = centroid.get("total_matches", 0)

    # Élite: alta win_rate + goles a favor + muchos torneos + llegan a finales
    if wr >= 0.45 and gd >= 0.3 and nt >= 8:
        return "Selecciones Dominantes"

    # Potencia histórica: buenos resultados pero no top
    if wr >= 0.35 and gd >= 0.0 and nt >= 6:
        return "Potencia Historica"

    # Competitivo con experiencia: han jugado mucho pero balance negativo
    if tm >= 25 and wr >= 0.20 and gd >= -0.8:
        return "Competitivo Recurrente"

    # Sobreviviente de fase de grupos: poca experiencia, resultados mixtos
    if nt <= 3 and bs <= 2 and wr >= 0.15:
        return "Participante Ocasional"

    # Sin victorias / debutante puro
    if wr < 0.05 or (wr < 0.15 and nt <= 2 and tm <= 6):
        return "Debutante / Sin Victorias"

    # Especialista en empates (draw_rate alta)
    dr = centroid.get("draw_rate", 0)
    if dr >= 0.35:
        return "Especialista en Empate"

    # Competitivo intermedio: experiencia moderada, balance levemente negativo
    # Corresponde al perfil más numeroso (selecciones que han participado
    # varias veces pero sin dominar — win_rate ~0.10-0.35, gd ~-1.5 a 0)
    if wr >= 0.10 and gd >= -1.6 and nt >= 1:
        return "Selecciones Competitivas Intermedias"

    return "Selecciones Competitivas Intermedias"


# =============================================================================
# CARGA DE ARTEFACTOS
# =============================================================================

def load_kmeans_artifacts(
    xkmeans_path:  str = "models/Xkmeans_phase2.pkl",
    profile_path:  str = "outputs/kmeans_profile_phase2.csv",
    metadata_path: str = "models/preprocessing_metadata.json",
) -> dict:
    """Carga los artefactos de K-Means generados en Fase 2."""
    print("\n  Cargando artefactos K-Means de Fase 2...")

    X_kmeans, team_names = joblib.load(xkmeans_path)
    profile_df = pd.read_csv(profile_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = list(X_kmeans.columns)
    n_null   = X_kmeans.isna().sum().sum()

    print(f"    Equipos: {len(team_names)} | Features: {len(features)} | Nulos: {n_null}")
    print(f"    Rango total_matches: [{X_kmeans['total_matches'].min():.0f}, {X_kmeans['total_matches'].max():.0f}]")
    print(f"    Rango win_rate:      [{X_kmeans['win_rate'].min():.3f}, {X_kmeans['win_rate'].max():.3f}]")

    return {
        "X_kmeans":   X_kmeans,
        "team_names": team_names,
        "profile_df": profile_df,
        "metadata":   metadata,
        "features":   features,
    }


# =============================================================================
# ESCALADO
# =============================================================================

def build_kmeans_scaler_pipeline() -> Pipeline:
    """Pipeline de preprocesamiento: Imputer + StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def fit_scaler(X: pd.DataFrame) -> tuple:
    """Ajusta el scaler y transforma X. Retorna (pipeline, X_scaled)."""
    scaler = build_kmeans_scaler_pipeline()
    X_sc   = scaler.fit_transform(X)
    print(f"\n  StandardScaler aplicado:")
    print(f"    Media post-escala: {X_sc.mean():.4f} (esperado ≈0)")
    print(f"    Std  post-escala:  {X_sc.std():.4f}  (esperado ≈1)")
    return scaler, X_sc


# =============================================================================
# EVALUACIÓN DE K
# =============================================================================

def evaluate_k_range(X_scaled: np.ndarray, k_range: range = K_RANGE) -> pd.DataFrame:
    """
    Evalúa K candidatos con cuatro métricas de clustering.

    Returns
    -------
    pd.DataFrame con columnas: k, inertia, silhouette, calinski, davies
    """
    print(f"\n  Evaluando K ∈ [{min(k_range)}, {max(k_range)}]...")
    print(f"  {'K':>4} {'Inercia':>12} {'Silhouette':>12} {'Calinski':>12} {'Davies':>10}")
    print(f"  {'─'*52}")

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, max_iter=MAX_ITER,
                    random_state=RANDOM_STATE)
        labels  = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil     = silhouette_score(X_scaled, labels)
        cal     = calinski_harabasz_score(X_scaled, labels)
        dav     = davies_bouldin_score(X_scaled, labels)
        rows.append({"k": k, "inertia": inertia, "silhouette": sil,
                     "calinski": cal, "davies": dav})
        print(f"  {k:>4} {inertia:>12.1f} {sil:>12.4f} {cal:>12.2f} {dav:>10.4f}")

    df = pd.DataFrame(rows)

    best_sil = int(df.loc[df["silhouette"].idxmax(), "k"])
    best_cal = int(df.loc[df["calinski"].idxmax(),  "k"])
    best_dav = int(df.loc[df["davies"].idxmin(),    "k"])

    print(f"\n  Recomendaciones por metrica:")
    print(f"    Silhouette maximo  -> K = {best_sil}  "
          f"({df.loc[df['k']==best_sil,'silhouette'].values[0]:.4f})")
    print(f"    Calinski maximo    -> K = {best_cal}  "
          f"({df.loc[df['k']==best_cal,'calinski'].values[0]:.2f})")
    print(f"    Davies minimo      -> K = {best_dav}  "
          f"({df.loc[df['k']==best_dav,'davies'].values[0]:.4f})")

    return df


def select_optimal_k(df_metrics: pd.DataFrame) -> int:
    """
    Selecciona el K óptimo usando Silhouette Score máximo como criterio
    primario, con Davies-Bouldin como criterio de desempate.

    Corrección v2:
        Ya NO usa el heurístico del codo (que acumulaba candidatos tardíos
        y terminaba seleccionando K elevados como K=11).
        En su lugar: K* = argmax(Silhouette) dentro del rango evaluado.

    Si dos K tienen Silhouette con diferencia < 0.02, se prefiere el
    de menor Davies-Bouldin (clusters más separados).
    """
    best_sil_idx = df_metrics["silhouette"].idxmax()
    best_sil_k   = int(df_metrics.loc[best_sil_idx, "k"])
    best_sil_val = df_metrics.loc[best_sil_idx, "silhouette"]

    # Buscar empates (diferencia < 0.02)
    near_best = df_metrics[
        df_metrics["silhouette"] >= best_sil_val - 0.02
    ].copy()

    if len(near_best) > 1:
        # Desempatar con Davies-Bouldin (menor es mejor)
        final_k = int(near_best.loc[near_best["davies"].idxmin(), "k"])
        print(f"\n  Empate en Silhouette (delta<0.02) -> desempate por Davies-Bouldin")
    else:
        final_k = best_sil_k

    print(f"\n  K optimo seleccionado: K = {final_k}")
    print(f"    Silhouette = {df_metrics[df_metrics['k']==final_k]['silhouette'].values[0]:.4f}")
    print(f"    Davies-DB  = {df_metrics[df_metrics['k']==final_k]['davies'].values[0]:.4f}")
    return final_k


def compare_k_candidates(
    X_scaled:  np.ndarray,
    candidates: list,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Imprime tabla comparativa detallada de K candidatos específicos.
    Usada para justificar la selección entre K=3 y K=4.
    """
    print(f"\n  Comparativa detallada de K candidatos: {candidates}")
    print(f"\n  {'K':>4} {'Silhouette':>12} {'Calinski':>12} {'Davies':>10} "
          f"{'Inercia':>12} {'Recomendado':>14}")
    print(f"  {'─'*66}")

    best_sil_k = int(df_metrics.loc[df_metrics["silhouette"].idxmax(), "k"])

    for k in candidates:
        row = df_metrics[df_metrics["k"] == k].iloc[0]
        rec = "*** OPTIMO ***" if k == best_sil_k else ""
        print(f"  {k:>4} {row['silhouette']:>12.4f} {row['calinski']:>12.2f} "
              f"{row['davies']:>10.4f} {row['inertia']:>12.1f} {rec:>14}")

    print(f"\n  Interpretacion:")
    for k in candidates:
        row = df_metrics[df_metrics["k"] == k].iloc[0]
        print(f"\n  K={k}:")
        print(f"    Silhouette {row['silhouette']:.4f} -> "
              f"{'buena cohesion y separacion' if row['silhouette']>0.3 else 'separacion moderada'}")
        print(f"    Davies-DB  {row['davies']:.4f}  -> "
              f"{'clusters bien diferenciados' if row['davies']<1.0 else 'clusters algo superpuestos'}")
        print(f"    Calinski   {row['calinski']:.2f}  -> "
              f"{'alta dispersion inter/intra' if row['calinski']>100 else 'dispersion moderada'}")


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """Entrena KMeans con k-means++ y n_init=20."""
    print(f"\n  Entrenando K-Means (K={k}, n_init={N_INIT}, init=k-means++)...")
    km = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT,
                max_iter=MAX_ITER, random_state=RANDOM_STATE)
    km.fit(X_scaled)

    labels  = km.labels_
    sil     = silhouette_score(X_scaled, labels)
    cal     = calinski_harabasz_score(X_scaled, labels)
    dav     = davies_bouldin_score(X_scaled, labels)

    print(f"    Iteraciones         : {km.n_iter_}")
    print(f"    Inercia (WCSS)      : {km.inertia_:.2f}")
    print(f"    Silhouette Score    : {sil:.4f}")
    print(f"    Calinski-Harabasz   : {cal:.2f}")
    print(f"    Davies-Bouldin      : {dav:.4f}")

    for c in range(k):
        n = np.sum(labels == c)
        print(f"    Cluster {c}: {n:>3} equipos ({n/len(labels)*100:.1f}%)")

    return km


def build_final_kmeans_pipeline(scaler: Pipeline, km: KMeans) -> Pipeline:
    """Pipeline completo para FastAPI: imputer + scaler + kmeans."""
    return Pipeline([
        ("imputer", scaler.named_steps["imputer"]),
        ("scaler",  scaler.named_steps["scaler"]),
        ("kmeans",  km),
    ])


# =============================================================================
# ASIGNACIÓN E INTERPRETACIÓN
# =============================================================================

def assign_clusters(
    profile_df: pd.DataFrame,
    team_names: list,
    labels:     np.ndarray,
    km:         KMeans,
    X_scaled:   np.ndarray,
    features:   list,
) -> pd.DataFrame:
    """Asigna cluster, silhouette individual y distancia al centroide."""
    sil_samples = silhouette_samples(X_scaled, labels)
    centroids   = km.cluster_centers_
    dist_list   = [np.linalg.norm(X_scaled[i] - centroids[labels[i]])
                   for i in range(len(labels))]

    df = profile_df.copy()
    team_to_cluster = dict(zip(team_names, labels))
    team_to_sil     = dict(zip(team_names, sil_samples))
    team_to_dist    = dict(zip(team_names, dist_list))

    df["cluster"]        = df["team"].map(team_to_cluster)
    df["silhouette_val"] = df["team"].map(team_to_sil)
    df["dist_centroid"]  = df["team"].map(team_to_dist)
    return df


def label_clusters_dynamically(
    df_clustered: pd.DataFrame,
    centroids_original: np.ndarray,
    features: list,
) -> dict:
    """
    Genera etiquetas dinámicas basadas en los centroides en escala original.
    Retorna un dict {cluster_id: label_string}.
    """
    labels_map = {}
    for c in sorted(df_clustered["cluster"].unique()):
        centroid_dict = dict(zip(features, centroids_original[int(c)]))
        labels_map[int(c)] = _label_cluster_from_centroid(centroid_dict, int(c))
    return labels_map


def describe_clusters(
    df_clustered: pd.DataFrame,
    features:     list,
    km:           KMeans,
    scaler:       Pipeline,
) -> tuple:
    """
    Calcula perfiles de clusters y genera etiquetas dinámicas.

    Returns
    -------
    (centroids_df_original, cluster_labels_map)
    """
    centroids_scaled   = km.cluster_centers_
    centroids_original = scaler.named_steps["scaler"].inverse_transform(
        centroids_scaled
    )
    centroids_df = pd.DataFrame(
        centroids_original,
        columns=features,
        index=[f"Cluster {i}" for i in range(km.n_clusters)],
    )

    # Etiquetas dinámicas
    cluster_labels = label_clusters_dynamically(
        df_clustered, centroids_original, features
    )

    print(f"\n  Centroides en escala original:")
    print(centroids_df.round(3).to_string())

    print(f"\n  Etiquetas generadas dinamicamente desde centroides:")
    for c, lbl in cluster_labels.items():
        print(f"    Cluster {c}: {lbl}")

    for c in sorted(df_clustered["cluster"].unique()):
        sub   = df_clustered[df_clustered["cluster"] == c]
        label = cluster_labels.get(int(c), f"Cluster {c}")
        teams = sorted(sub["team"].tolist())
        print(f"\n  {'─'*60}")
        print(f"  CLUSTER {c} — {label} ({len(sub)} selecciones)")
        print(f"  Equipos: {', '.join(teams)}")
        for feat in ["win_rate", "avg_goal_diff", "num_tournaments", "best_stage"]:
            print(f"    {feat:<25}: {sub[feat].mean():.3f}")
        print(f"  Silhouette medio: {sub['silhouette_val'].mean():.4f}")

    return centroids_df, cluster_labels


# =============================================================================
# PCA + t-SNE
# =============================================================================

def apply_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    """Aplica PCA para visualización 2D."""
    pca    = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    print(f"\n  PCA 2D: PC1={var_exp[0]*100:.1f}% | PC2={var_exp[1]*100:.1f}% | "
          f"Total={sum(var_exp)*100:.1f}%")
    return X_pca, pca, var_exp


def apply_tsne(X_scaled: np.ndarray, perplexity: float = 10.0) -> np.ndarray:
    """Aplica t-SNE con compatibilidad sklearn >=1.5 y <1.5."""
    try:
        import sklearn
        from sklearn.manifold import TSNE
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        kwargs = dict(n_components=2, perplexity=perplexity,
                      random_state=RANDOM_STATE, learning_rate="auto", init="pca")
        kwargs["max_iter" if sk_ver >= (1, 5) else "n_iter"] = 1000
        X_tsne = TSNE(**kwargs).fit_transform(X_scaled)
        print(f"  t-SNE 2D aplicado (perplexity={perplexity})")
        return X_tsne
    except Exception as e:
        print(f"  t-SNE omitido: {e}")
        return None


# =============================================================================
# SERIALIZACIÓN
# =============================================================================

def save_phase4_artifacts(
    km:              KMeans,
    final_pipeline:  Pipeline,
    df_clustered:    pd.DataFrame,
    centroids_df:    pd.DataFrame,
    cluster_labels:  dict,
    df_metrics:      pd.DataFrame,
    optimal_k:       int,
    sil_score:       float,
    paths:           dict = None,
) -> None:
    """Serializa todos los artefactos de Fase 4."""
    if paths is None:
        paths = DEFAULT_PATHS

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/phase4", exist_ok=True)

    joblib.dump(km, paths["model"])
    print(f"\n    KMeans model     : {paths['model']}")

    joblib.dump(final_pipeline, paths["pipeline"])
    print(f"    KMeans pipeline  : {paths['pipeline']}")

    df_clustered.to_csv(paths["clusters"], index=False)
    print(f"    Clusters CSV     : {paths['clusters']}")

    def _safe(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(type(obj))

    cluster_summary = {}
    for c in sorted(df_clustered["cluster"].unique()):
        sub = df_clustered[df_clustered["cluster"] == c]
        cluster_summary[str(int(c))] = {
            "label":           cluster_labels.get(int(c), f"Cluster {c}"),
            "n_teams":         int(len(sub)),
            "teams":           sorted(sub["team"].tolist()),
            "silhouette_mean": float(sub["silhouette_val"].mean()),
            "centroid":        {col: float(centroids_df.iloc[int(c)][col])
                                for col in centroids_df.columns},
        }

    payload = {
        "project":    "Copa Mundial FIFA 1930-2022 — ML Prediccion",
        "phase":      "Fase 4 — Clustering K-Means v2",
        "model":      "KMeans",
        "optimal_k":  int(optimal_k),
        "silhouette": float(sil_score),
        "selection_method": "argmax(Silhouette) — v2 corregido",
        "k_evaluation": df_metrics.to_dict(orient="records"),
        "clusters":   cluster_summary,
        "cluster_label_map": {str(k): v for k, v in cluster_labels.items()},
        "methodology": {
            "scaling":    "StandardScaler",
            "init":       "k-means++",
            "n_init":     N_INIT,
            "k_primary":  "argmax Silhouette Score",
            "k_tiebreak": "min Davies-Bouldin (delta Silhouette < 0.02)",
        },
    }

    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_safe)
    print(f"    Metricas JSON    : {paths['metrics']}")
