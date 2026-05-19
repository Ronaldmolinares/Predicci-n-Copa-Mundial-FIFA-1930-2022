"""
==============================================================================
FASE 4 — CLUSTERING NO SUPERVISADO: K-MEANS
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/kmeans_modeling.py

Descripción:
    Módulo de clustering K-Means para construir perfiles competitivos de
    selecciones nacionales basados en su desempeño histórico en Copas del Mundo.

    A diferencia del Random Forest (predictivo), K-Means es DESCRIPTIVO:
    agrupa selecciones por similitud histórica sin usar un target futuro.
    Por tanto NO hay riesgo de leakage — usamos estadísticas completas.

Estrategia de selección del K óptimo:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Elbow Method (inercia WCSS)                             │
    │     Identifica visualmente el punto de retorno decreciente  │
    │     al añadir más clusters. Heurístico pero intuitivo.      │
    │                                                             │
    │  2. Silhouette Score [-1, 1]                                │
    │     s(i) = (b - a) / max(a, b)                              │
    │     a = dist. media intra-cluster (cohesión)                │
    │     b = dist. media al cluster vecino más cercano           │
    │     Mayor = clusters más compactos y separados. MÉTRICA     │
    │     CUANTITATIVA — prima sobre el codo ambiguo.             │
    │                                                             │
    │  3. Calinski-Harabasz Index (bonus)                         │
    │     Ratio dispersión inter / intra cluster.                 │
    │     Mayor = mejor separación.                               │
    │                                                             │
    │  4. Davies-Bouldin Score (bonus)                            │
    │     Media de similitudes entre clusters adyacentes.         │
    │     Menor = mejor (clusters más distintos entre sí).        │
    │                                                             │
    │  Regla de decisión:                                         │
    │    Se selecciona el K con mayor Silhouette entre los K      │
    │    candidatos identificados en el codo, priorizando         │
    │    la interpretabilidad táctica (K óptimo típico: 4–6).    │
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

RANDOM_STATE   = 42
K_RANGE        = range(2, 13)       # K candidatos a evaluar
N_INIT         = 20                 # Inicializaciones aleatorias por K
MAX_ITER       = 500                # Iteraciones máximas por run
DEFAULT_PATHS  = {
    "model":    "models/kmeans_model.pkl",
    "pipeline": "models/kmeans_pipeline.pkl",
    "metrics":  "models/kmeans_metrics.json",
    "clusters": "outputs/phase4/kmeans_clusters.csv",
}

# Interpretación táctica de clusters FIFA (se actualiza tras ver centroides)
CLUSTER_LABELS = {
    0: "Élite Mundial",
    1: "Potencia Regional",
    2: "Competitivo Estable",
    3: "Emergente / Irregular",
    4: "Participante Ocasional",
    5: "Debutante / Histórico",
}


# =============================================================================
# CARGA DE ARTEFACTOS
# =============================================================================

def load_kmeans_artifacts(
    xkmeans_path:  str = "models/Xkmeans_phase2.pkl",
    profile_path:  str = "outputs/kmeans_profile_phase2.csv",
    metadata_path: str = "models/preprocessing_metadata.json",
) -> dict:
    """
    Carga los artefactos de K-Means generados en Fase 2.

    Returns
    -------
    dict con claves:
        X_kmeans    → DataFrame (86 equipos × 10 features)
        team_names  → lista de nombres de selecciones
        profile_df  → DataFrame completo del perfil por equipo
        metadata    → dict con metadata JSON
        features    → lista de nombres de features
    """
    print("\n  Cargando artefactos K-Means de Fase 2...")

    X_kmeans, team_names = joblib.load(xkmeans_path)
    print(f"    Xkmeans_phase2.pkl → shape: {X_kmeans.shape}")
    print(f"    Equipos cargados  : {len(team_names)}")

    profile_df = pd.read_csv(profile_path)
    print(f"    kmeans_profile     → shape: {profile_df.shape}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = list(X_kmeans.columns)
    print(f"    Features K-Means   : {features}")

    # Verificar que no hay NaN
    n_null = X_kmeans.isna().sum().sum()
    print(f"    Valores nulos      : {n_null} {'✓' if n_null == 0 else '⚠️ requieren imputación'}")

    print(f"\n    Estadísticas descriptivas de X_kmeans:")
    print(f"    {'Feature':<25} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std':>8}")
    print(f"    {'─'*60}")
    for col in features:
        print(f"    {col:<25} {X_kmeans[col].min():>8.3f} "
              f"{X_kmeans[col].mean():>8.3f} "
              f"{X_kmeans[col].max():>8.3f} "
              f"{X_kmeans[col].std():>8.3f}")

    return {
        "X_kmeans":   X_kmeans,
        "team_names": team_names,
        "profile_df": profile_df,
        "metadata":   metadata,
        "features":   features,
    }


# =============================================================================
# PIPELINE DE PREPROCESAMIENTO K-MEANS
# =============================================================================

def build_kmeans_scaler_pipeline() -> Pipeline:
    """
    Pipeline de preprocesamiento para K-Means.

    ¿Por qué StandardScaler es OBLIGATORIO para K-Means?
        K-Means usa distancia euclidiana. Sin escalar, variables con alta
        magnitud (total_matches: 1–114) dominarían a variables con baja
        magnitud (win_rate: 0–0.67), distorsionando los clusters.

        StandardScaler → media=0, std=1 para cada feature.
        Así todas las variables contribuyen equitativamente a la distancia.

    Arquitectura:
        SimpleImputer(median) → maneja NaN si existen
        StandardScaler()      → estandarización z-score
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def fit_scaler(X: pd.DataFrame) -> tuple:
    """
    Ajusta el pipeline de escalado y transforma X.

    Returns
    -------
    (scaler_pipeline, X_scaled_array)
    """
    scaler_pipeline = build_kmeans_scaler_pipeline()
    X_scaled = scaler_pipeline.fit_transform(X)
    print(f"\n  Escalado aplicado: media≈0, std≈1 por feature")
    print(f"    X_scaled shape: {X_scaled.shape}")
    print(f"    Media global post-escala  : {X_scaled.mean():.4f} (esperado ≈0)")
    print(f"    Std global post-escala    : {X_scaled.std():.4f} (esperado ≈1)")
    return scaler_pipeline, X_scaled


# =============================================================================
# SELECCIÓN DEL K ÓPTIMO
# =============================================================================

def evaluate_k_range(
    X_scaled: np.ndarray,
    k_range:  range = K_RANGE,
) -> pd.DataFrame:
    """
    Evalúa múltiples valores de K y calcula cuatro métricas de clustering.

    Métricas calculadas:
        inertia     → WCSS (Within-Cluster Sum of Squares). Menor = mejor.
                      Usado para Elbow Method.
        silhouette  → Cohesión y separación media [-1, 1]. Mayor = mejor.
        calinski    → Ratio dispersión inter/intra. Mayor = mejor.
        davies      → Similitud entre clusters adyacentes. Menor = mejor.

    Parameters
    ----------
    X_scaled : np.ndarray
        Datos escalados.
    k_range : range
        Rango de K a evaluar.

    Returns
    -------
    pd.DataFrame con métricas por K.
    """
    print(f"\n  Evaluando K ∈ [{min(k_range)}, {max(k_range)}]...")
    print(f"  {'K':>4} {'Inercia':>12} {'Silhouette':>12} {'Calinski':>12} {'Davies':>10}")
    print(f"  {'─'*52}")

    rows = []
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )
        labels    = km.fit_predict(X_scaled)
        inertia   = km.inertia_
        sil       = silhouette_score(X_scaled, labels)
        cal       = calinski_harabasz_score(X_scaled, labels)
        dav       = davies_bouldin_score(X_scaled, labels)

        rows.append({
            "k": k, "inertia": inertia,
            "silhouette": sil, "calinski": cal, "davies": dav,
        })
        print(f"  {k:>4} {inertia:>12.1f} {sil:>12.4f} {cal:>12.2f} {dav:>10.4f}")

    df_metrics = pd.DataFrame(rows)

    # Selección automática del K óptimo
    best_sil_k = df_metrics.loc[df_metrics["silhouette"].idxmax(), "k"]
    best_cal_k = df_metrics.loc[df_metrics["calinski"].idxmax(),  "k"]
    best_dav_k = df_metrics.loc[df_metrics["davies"].idxmin(),    "k"]

    print(f"\n  Recomendaciones por métrica:")
    print(f"    Silhouette máximo  → K = {best_sil_k}  ({df_metrics.loc[df_metrics['k']==best_sil_k, 'silhouette'].values[0]:.4f})")
    print(f"    Calinski máximo    → K = {best_cal_k}  ({df_metrics.loc[df_metrics['k']==best_cal_k, 'calinski'].values[0]:.2f})")
    print(f"    Davies mínimo      → K = {best_dav_k}  ({df_metrics.loc[df_metrics['k']==best_dav_k, 'davies'].values[0]:.4f})")

    return df_metrics


def select_optimal_k(df_metrics: pd.DataFrame) -> int:
    """
    Aplica la regla de decisión para seleccionar el K óptimo.

    Regla:
        1. Candidatos del codo: K donde la reducción de inercia cae < 15%
           respecto al paso anterior (retorno decreciente).
        2. Entre esos candidatos, seleccionar el de mayor Silhouette.
        3. Si el Silhouette máximo global está entre [4, 7], priorizar
           interpretabilidad táctica.

    Returns
    -------
    int : K seleccionado.
    """
    # Calcular reducción porcentual de inercia en cada paso
    inertias = df_metrics["inertia"].values
    k_vals   = df_metrics["k"].values
    pct_drops = np.abs(np.diff(inertias) / inertias[:-1])   # reducción relativa

    # Candidatos del codo: donde la caída de inercia ya es < 15%
    elbow_candidates = []
    for i, drop in enumerate(pct_drops):
        if drop < 0.15:
            elbow_candidates.append(k_vals[i + 1])

    if not elbow_candidates:
        elbow_candidates = list(k_vals)

    # Entre candidatos del codo → mayor Silhouette
    cand_df  = df_metrics[df_metrics["k"].isin(elbow_candidates)]
    best_k   = int(cand_df.loc[cand_df["silhouette"].idxmax(), "k"])

    print(f"\n  Candidatos del codo (Δinercia < 15%): {elbow_candidates}")
    print(f"  K óptimo seleccionado (max Silhouette): K = {best_k}")
    return best_k


# =============================================================================
# ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """
    Entrena el modelo K-Means final con el K óptimo.

    n_init=20: reinicia 20 veces con centroides aleatorios distintos
    y queda con el resultado de menor inercia. Mitiga la sensibilidad
    a la inicialización aleatoria de K-Means.

    init="k-means++": inicialización inteligente que esparce los
    centroides iniciales, convergiendo más rápido y a mejores soluciones
    que la inicialización puramente aleatoria ("random").
    """
    print(f"\n  Entrenando K-Means final con K = {k}...")
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    km.fit(X_scaled)

    labels   = km.labels_
    inertia  = km.inertia_
    sil      = silhouette_score(X_scaled, labels)
    cal      = calinski_harabasz_score(X_scaled, labels)
    dav      = davies_bouldin_score(X_scaled, labels)
    n_iter   = km.n_iter_

    print(f"    Iteraciones hasta convergencia : {n_iter}")
    print(f"    Inercia (WCSS)                 : {inertia:.2f}")
    print(f"    Silhouette Score               : {sil:.4f}")
    print(f"    Calinski-Harabasz              : {cal:.2f}")
    print(f"    Davies-Bouldin                 : {dav:.4f}")

    # Tamaño de cada cluster
    print(f"\n    Distribución de equipos por cluster:")
    for c in range(k):
        n = np.sum(labels == c)
        print(f"      Cluster {c}: {n:>3} equipos  ({n/len(labels)*100:.1f}%)")

    return km


# =============================================================================
# CONSTRUCCIÓN DEL PIPELINE COMPLETO K-MEANS
# =============================================================================

def build_final_kmeans_pipeline(scaler_pipeline: Pipeline, km: KMeans) -> Pipeline:
    """
    Ensambla el pipeline completo: scaler + kmeans.

    Este objeto es el que se serializa para FastAPI.
    Uso en producción: pipeline.predict(X_new_raw)
    """
    final_pipeline = Pipeline([
        ("imputer", scaler_pipeline.named_steps["imputer"]),
        ("scaler",  scaler_pipeline.named_steps["scaler"]),
        ("kmeans",  km),
    ])
    return final_pipeline


# =============================================================================
# ASIGNACIÓN E INTERPRETACIÓN DE CLUSTERS
# =============================================================================

def assign_clusters(
    profile_df: pd.DataFrame,
    team_names: list,
    labels:     np.ndarray,
    km:         KMeans,
    X_scaled:   np.ndarray,
    features:   list,
) -> pd.DataFrame:
    """
    Asigna clusters a cada selección y enriquece con métricas.

    Agrega:
        cluster         → ID del cluster asignado
        silhouette_val  → Silhouette individual [-1, 1]
        dist_centroid   → Distancia euclidiana al centroide propio
    """
    sil_samples = silhouette_samples(X_scaled, labels)

    # Distancia de cada equipo a su centroide
    centroids     = km.cluster_centers_
    dist_list     = []
    for i, (lab, x) in enumerate(zip(labels, X_scaled)):
        dist = np.linalg.norm(x - centroids[lab])
        dist_list.append(dist)

    df = profile_df.copy()

    # Alinear por nombre de equipo
    team_to_cluster = dict(zip(team_names, labels))
    team_to_sil     = dict(zip(team_names, sil_samples))
    team_to_dist    = dict(zip(team_names, dist_list))

    df["cluster"]        = df["team"].map(team_to_cluster)
    df["silhouette_val"] = df["team"].map(team_to_sil)
    df["dist_centroid"]  = df["team"].map(team_to_dist)

    # Etiqueta interpretativa
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)

    return df


def describe_clusters(
    df_clustered: pd.DataFrame,
    features:     list,
    km:           KMeans,
    scaler:       Pipeline,
) -> pd.DataFrame:
    """
    Calcula y muestra el perfil estadístico de cada cluster.

    Para interpretar los centroides se invierten las transformaciones
    del StandardScaler → valores en escala original.
    """
    print(f"\n  Perfil estadístico por cluster:")
    agg = df_clustered.groupby("cluster")[features].mean()

    # Invertir escala de centroides para interpretación
    centroids_scaled   = km.cluster_centers_
    centroids_original = scaler.named_steps["scaler"].inverse_transform(
        centroids_scaled
    )
    centroids_df = pd.DataFrame(
        centroids_original,
        columns=features,
        index=[f"Cluster {i}" for i in range(km.n_clusters)],
    )

    print(f"\n  Centroides en escala original:")
    print(centroids_df.round(3).to_string())

    # Reporte por cluster
    for c in sorted(df_clustered["cluster"].unique()):
        subset = df_clustered[df_clustered["cluster"] == c]
        label  = CLUSTER_LABELS.get(c, f"Cluster {c}")
        teams  = sorted(subset["team"].tolist())
        print(f"\n  {'─'*62}")
        print(f"  CLUSTER {c} — {label} ({len(subset)} selecciones)")
        print(f"  {'─'*62}")
        print(f"  Equipos: {', '.join(teams)}")
        print(f"  Métricas medias:")
        for feat in features:
            val = subset[feat].mean()
            print(f"    {feat:<25}: {val:.4f}")
        print(f"  Silhouette medio: {subset['silhouette_val'].mean():.4f}")

    return centroids_df


# =============================================================================
# REDUCCIÓN DE DIMENSIONALIDAD
# =============================================================================

def apply_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    """
    Aplica PCA para visualización 2D de los clusters.

    PCA (Principal Component Analysis):
        Encuentra las direcciones de máxima varianza en el espacio
        de 10 dimensiones y proyecta en 2D para visualización.
        No es una reducción para modelado — solo para graficar.

    Returns
    -------
    (X_pca, pca_model, explained_variance)
    """
    pca     = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca   = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    print(f"\n  PCA 2D:")
    print(f"    PC1 varianza explicada: {var_exp[0]*100:.1f}%")
    print(f"    PC2 varianza explicada: {var_exp[1]*100:.1f}%")
    print(f"    Total explicada       : {sum(var_exp)*100:.1f}%")

    return X_pca, pca, var_exp


def apply_tsne(X_scaled: np.ndarray, perplexity: float = 10.0) -> np.ndarray:
    """
    Aplica t-SNE para visualización 2D no lineal (opcional).

    t-SNE preserva mejor las relaciones locales que PCA.
    Útil cuando los clusters tienen formas no lineales.
    Perplexity recomendada: entre 5 y n_samples/3.
    Con 86 equipos → perplexity ≈ 10-20.
    """
    try:
        import sklearn
        from sklearn.manifold import TSNE
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        tsne_kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            random_state=RANDOM_STATE,
            learning_rate="auto",
            init="pca",
        )
        if sk_ver >= (1, 5):
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000
        tsne   = TSNE(**tsne_kwargs)
        X_tsne = tsne.fit_transform(X_scaled)
        print(f"\n  t-SNE 2D aplicado (perplexity={perplexity})")
        return X_tsne
    except Exception as e:
        print(f"\n  t-SNE no disponible: {e}")
        return None


# =============================================================================
# SERIALIZACIÓN
# =============================================================================

def save_phase4_artifacts(
    km:             KMeans,
    final_pipeline: Pipeline,
    df_clustered:   pd.DataFrame,
    centroids_df:   pd.DataFrame,
    df_metrics:     pd.DataFrame,
    optimal_k:      int,
    sil_score:      float,
    paths:          dict = None,
) -> None:
    """
    Serializa todos los artefactos de Fase 4.

    Archivos generados:
        models/kmeans_model.pkl       → solo el modelo KMeans
        models/kmeans_pipeline.pkl    → Pipeline completo (scaler + kmeans)
        models/kmeans_metrics.json    → métricas, centroides, interpretación
        outputs/phase4/kmeans_clusters.csv → asignación de clusters por equipo
    """
    if paths is None:
        paths = DEFAULT_PATHS

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/phase4", exist_ok=True)

    # 1. Solo el modelo
    joblib.dump(km, paths["model"])
    print(f"\n    KMeans model guardado    : {paths['model']}")

    # 2. Pipeline completo
    joblib.dump(final_pipeline, paths["pipeline"])
    print(f"    KMeans pipeline guardado : {paths['pipeline']}")

    # 3. CSV de clusters
    df_clustered.to_csv(paths["clusters"], index=False)
    print(f"    Clusters CSV guardado    : {paths['clusters']}")

    # 4. Métricas JSON
    def _safe(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(type(obj))

    cluster_summary = {}
    for c in sorted(df_clustered["cluster"].unique()):
        sub = df_clustered[df_clustered["cluster"] == c]
        cluster_summary[str(int(c))] = {
            "label":    CLUSTER_LABELS.get(int(c), f"Cluster {c}"),
            "n_teams":  int(len(sub)),
            "teams":    sorted(sub["team"].tolist()),
            "silhouette_mean": float(sub["silhouette_val"].mean()),
            "centroid": {
                col: float(centroids_df.iloc[int(c)][col])
                for col in centroids_df.columns
            },
        }

    metrics_payload = {
        "project":   "Copa Mundial FIFA 1930–2022 — ML Predicción",
        "phase":     "Fase 4 — Clustering K-Means",
        "model":     "KMeans",
        "optimal_k": int(optimal_k),
        "silhouette_score": float(sil_score),
        "k_evaluation": df_metrics.to_dict(orient="records"),
        "clusters":  cluster_summary,
        "cluster_label_map": CLUSTER_LABELS,
        "methodology": {
            "scaling":      "StandardScaler (media=0, std=1)",
            "init":         "k-means++",
            "n_init":       N_INIT,
            "k_selection":  "Silhouette Score + Elbow Method",
            "visualization": "PCA 2D + t-SNE 2D",
        },
    }

    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False, default=_safe)
    print(f"    Métricas JSON guardadas  : {paths['metrics']}")
