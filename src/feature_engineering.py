"""
==============================================================================
FASE 2 — INGENIERÍA DE CARACTERÍSTICAS HISTÓRICAS
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/feature_engineering.py

Descripción:
    Construcción de variables predictivas a partir del historial de cada
    selección. La regla fundamental es:

    ╔══════════════════════════════════════════════════════════╗
    ║  Para predecir el partido P en la fecha D,               ║
    ║  SOLO se permite usar información de partidos            ║
    ║  jugados ANTES de la fecha D.                            ║
    ╚══════════════════════════════════════════════════════════╝

    Esto se implementa ordenando el dataset cronológicamente y calculando
    estadísticas acumulativas con un desplazamiento temporal (shift),
    de forma que el partido actual NUNCA se incluye en sus propias features.

Técnica anti-leakage utilizada:
    → groupby(team).expanding().mean().shift(1)
    El .shift(1) desplaza la ventana un paso hacia adelante, garantizando
    que la estadística del partido i solo considera los partidos 0..i-1.

Features generadas (por partido, desde perspectiva del equipo local):
    Bloque A — Rendimiento histórico acumulado
        home_hist_win_rate      Tasa victorias histórica equipo local
        home_hist_draw_rate     Tasa empates histórica equipo local
        home_hist_avg_gf        Promedio goles a favor (local)
        home_hist_avg_ga        Promedio goles en contra (local)
        home_hist_avg_gd        Diferencia de goles media (local)
        home_hist_matches       Partidos jugados hasta ese momento

    Bloque B — Ídem para equipo visitante
        away_hist_win_rate, away_hist_draw_rate,
        away_hist_avg_gf, away_hist_avg_ga,
        away_hist_avg_gd, away_hist_matches

    Bloque C — Features comparativas (diferencia entre equipos)
        delta_win_rate          home_hist_win_rate - away_hist_win_rate
        delta_avg_gf            home_hist_avg_gf  - away_hist_avg_gf
        delta_avg_gd            home_hist_avg_gd  - away_hist_avg_gd

    Bloque D — Forma reciente (últimas N apariciones)
        home_recent_win_rate    Tasa de victorias en últimos N partidos
        away_recent_win_rate    Ídem para visitante

    Bloque E — Contexto del partido
        Year                    Año del torneo (captura la época)
        stage_encoded           Fase del torneo (codificada ordinalmente)
        home_wc_appearances     Cantidad de Copas del Mundo anteriores

Features EXPLÍCITAMENTE EXCLUIDAS (data leakage):
    home_score, away_score  → son el resultado (se derivan para el TARGET)
    home_xg, away_xg        → Expected Goals calculadas durante el partido
    goal_diff, total_goals  → derivadas del resultado
    went_to_pen             → info post-partido
    home_goal, away_goal    → info post-partido
    home_penalty, away_penalty → info post-partido

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado — UPTC
==============================================================================
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES
# =============================================================================

# Ventana de "forma reciente": últimos N partidos de cada equipo
RECENT_WINDOW = 5

# Codificación ordinal de fases del torneo
# Mayor valor = fase más avanzada (cuartos > grupos)
STAGE_ORDER = {
    # Fase de grupos (distintos nombres históricos)
    "group stage":            1,
    "first group stage":      1,
    "second group stage":     1,
    "group":                  1,
    "first round":            1,
    "preliminary round":      1,
    "pool":                   1,
    # Rondas intermedias y desempates
    "group stage play-off":   2,
    "second round":           2,
    "round of 16":            2,
    "last 16":                2,
    # Cuartos de final
    "quarter-finals":         3,
    "quarter finals":         3,
    "quarterfinals":          3,
    # Semifinales
    "semi-finals":            4,
    "semi finals":            4,
    "semifinals":             4,
    # Tercer puesto / Final stage
    "third place":            5,
    "third-place match":      5,
    "third-place play-off":   5,
    "third place play-off":   5,
    "play-off for third place": 5,
    "final stage":            5,
    # Final
    "final":                  6,
    # Valor por defecto para casos no mapeados
    "unknown":                0,
}

# =============================================================================
# PASO 1: PREPARACIÓN TEMPORAL
# =============================================================================

def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el dataset de partidos cronológicamente.

    Detecta automáticamente la columna de fecha entre candidatos comunes
    del dataset piterfm. Si no existe una columna de fecha explícita, ordena
    por 'Year' + número de partido como proxy.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de partidos (matches_1930_2022.csv enriquecido).

    Returns
    -------
    pd.DataFrame
        Dataset ordenado cronológicamente con columna 'match_date' parseada.
    """
    df = df.copy()

    # Detectar columna de fecha
    date_candidates = ["date", "datetime", "match_date", "Date", "Datetime"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col:
        df["match_date"] = pd.to_datetime(df[date_col], errors="coerce")
        n_null = df["match_date"].isna().sum()
        if n_null > 0:
            print(f"    {n_null} fechas no parseadas en '{date_col}' → se imputarán por año")
            # Imputación por año cuando la fecha exacta no está disponible
            if "Year" in df.columns:
                mask = df["match_date"].isna()
                df.loc[mask, "match_date"] = pd.to_datetime(
                    df.loc[mask, "Year"].astype(str) + "-06-01"
                )
        print(f"   Columna de fecha: '{date_col}' → 'match_date' (datetime)")
    elif "Year" in df.columns:
        # Fallback: solo tenemos año — usamos 1-jun de cada año como proxy
        df["match_date"] = pd.to_datetime(df["Year"].astype(str) + "-06-01")
        print("    Sin columna de fecha exacta. Usando 'Year' como proxy de fecha.")
    else:
        raise ValueError(
            " No se encontró columna de fecha ('date', 'Year', etc.). "
            "Verifica el dataset descargado."
        )

    # Ordenar cronológicamente y reiniciar índice
    df = df.sort_values("match_date").reset_index(drop=True)
    df["match_index"] = df.index  # índice temporal de partido

    print(f"   Rango temporal: {df['match_date'].min().date()} "
          f"→ {df['match_date'].max().date()}")
    print(f"   Total partidos ordenados: {len(df):,}")
    return df


# =============================================================================
# PASO 2: TABLA MAESTRA DE ESTADÍSTICAS HISTÓRICAS POR EQUIPO
# =============================================================================

def _build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla larga (long format) donde cada fila es
    una aparición de un equipo en un partido, con su resultado
    desde su propia perspectiva.

    Estructura resultante:
        match_index | match_date | team | goals_for | goals_against |
        is_win | is_draw | is_loss

    Esta tabla es el insumo para calcular estadísticas acumulativas.
    """
    # Perspectiva local
    home = df[["match_index", "match_date", "home_team",
               "home_score", "away_score", "result"]].copy()
    home.columns = ["match_index", "match_date", "team",
                    "goals_for", "goals_against", "result"]
    home["is_win"]  = (home["result"] == "home_win").astype(float)
    home["is_draw"] = (home["result"] == "draw").astype(float)
    home["is_loss"] = (home["result"] == "away_win").astype(float)

    # Perspectiva visitante
    away = df[["match_index", "match_date", "away_team",
               "away_score", "home_score", "result"]].copy()
    away.columns = ["match_index", "match_date", "team",
                    "goals_for", "goals_against", "result"]
    away["is_win"]  = (away["result"] == "away_win").astype(float)
    away["is_draw"] = (away["result"] == "draw").astype(float)
    away["is_loss"] = (away["result"] == "home_win").astype(float)

    log = pd.concat([home, away], ignore_index=True)
    log["goal_diff"] = log["goals_for"] - log["goals_against"]
    log = log.sort_values(["team", "match_date", "match_index"]).reset_index(drop=True)
    return log


def build_historical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas históricas acumuladas por equipo, usando SOLO
    información de partidos anteriores al partido actual.

    Técnica anti-leakage:
        .expanding().mean() calcula la media acumulada hasta la fila i.
        .shift(1)           desplaza el resultado un paso → la fila i
                            no se incluye en su propia estadística.

    Returns
    -------
    pd.DataFrame
        Tabla con columnas:
            team | match_index | hist_win_rate | hist_draw_rate |
            hist_avg_gf | hist_avg_ga | hist_avg_gd | hist_matches
    """
    log = _build_team_match_log(df)

    # Estadísticas acumulativas por equipo (ordenadas por fecha)
    def _calc_hist(g):
        team_name = g.name   # groupby key = team name
        return pd.DataFrame({
            "match_index": g["match_index"].values,
            "match_date":  g["match_date"].values,
            "team":        team_name,
            # .expanding().mean() = media acumulada hasta fila i
            # .shift(1)           = excluir la fila i de su propio cálculo
            "hist_win_rate":  g["is_win"].expanding().mean().shift(1).values,
            "hist_draw_rate": g["is_draw"].expanding().mean().shift(1).values,
            "hist_avg_gf":    g["goals_for"].expanding().mean().shift(1).values,
            "hist_avg_ga":    g["goals_against"].expanding().mean().shift(1).values,
            "hist_avg_gd":    g["goal_diff"].expanding().mean().shift(1).values,
            # Número de partidos previos jugados por este equipo
            "hist_matches":   g["is_win"].expanding().count().shift(1).values,
        })

    agg = log.groupby("team", group_keys=False).apply(_calc_hist).reset_index(drop=True)

    return agg


# =============================================================================
# PASO 3: FORMA RECIENTE (ventana deslizante)
# =============================================================================

def build_recent_form(df: pd.DataFrame,
                      window: int = RECENT_WINDOW) -> pd.DataFrame:
    """
    Calcula la tasa de victorias en los últimos `window` partidos
    de cada equipo, desplazada para excluir el partido actual.

    Usa rolling(window=window, min_periods=1) para no generar NaN
    cuando un equipo tiene menos de `window` partidos previos.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de partidos ordenado cronológicamente.
    window : int
        Tamaño de la ventana de forma reciente (default: 5).

    Returns
    -------
    pd.DataFrame
        Tabla con: team | match_index | recent_win_rate
    """
    log = _build_team_match_log(df)

    def _calc_recent(g):
        team_name = g.name
        return pd.DataFrame({
            "match_index":     g["match_index"].values,
            "team":            team_name,
            "recent_win_rate": (
                g["is_win"]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)          # excluir partido actual
                .values
            ),
        })

    recent = log.groupby("team", group_keys=False).apply(_calc_recent).reset_index(drop=True)

    return recent


# =============================================================================
# PASO 4: PARTICIPACIONES HISTÓRICAS EN COPAS DEL MUNDO
# =============================================================================

def build_wc_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula cuántas Copas del Mundo distintas ha jugado cada equipo
    ANTES del torneo actual.

    La "aparición" se define por año del torneo. Para cada partido,
    se cuenta el número de años únicos anteriores en que el equipo participó.

    Returns
    -------
    pd.DataFrame
        Tabla con: team | Year | wc_appearances_before
    """
    if "Year" not in df.columns:
        return pd.DataFrame(columns=["team", "Year", "wc_appearances_before"])

    # Todos los equipos que aparecen en cada año
    home = df[["Year", "home_team"]].rename(columns={"home_team": "team"})
    away = df[["Year", "away_team"]].rename(columns={"away_team": "team"})
    appearances = pd.concat([home, away]).drop_duplicates()
    appearances = appearances.sort_values("Year")

    # Para cada (team, Year), contar años anteriores
    result_rows = []
    for _, row in appearances.iterrows():
        team = row["team"]
        year = row["Year"]
        prev_years = appearances[
            (appearances["team"] == team) &
            (appearances["Year"] < year)
        ]["Year"].nunique()
        result_rows.append({
            "team": team,
            "Year": year,
            "wc_appearances_before": prev_years,
        })

    return pd.DataFrame(result_rows)


# =============================================================================
# PASO 5: CODIFICACIÓN ORDINAL DE FASES
# =============================================================================

def encode_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica ordinalmente la fase del torneo.

    La codificación ordinal es preferible a OHE aquí porque las fases
    tienen un orden natural: grupos < 16avos < cuartos < semi < final.
    Esta relación ordinal es útil como señal para el modelo.

    Detecta automáticamente la columna de fase entre candidatos comunes.
    Búsqueda case-insensitive para capturar variaciones como 'Round', 'round', etc.
    """
    df = df.copy()

    # Búsqueda case-insensitive de la columna de fase
    stage_col = None
    candidates = ["stage", "round", "phase", "match_phase"]
    for c in candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            stage_col = matches[0]
            break

    if stage_col is None:
        print("    Columna de fase no encontrada. stage_encoded = 0.")
        df["stage_encoded"] = 0
        return df

    df["stage_encoded"] = (
        df[stage_col]
        .str.lower()
        .str.strip()
        .map(STAGE_ORDER)
        .fillna(0)
        .astype(int)
    )

    unmapped = df[df["stage_encoded"] == 0][stage_col].unique()
    if len(unmapped) > 0 and not (len(unmapped) == 1 and unmapped[0] is np.nan):
        print(f"    Fases no mapeadas (→ 0): {list(unmapped)}")

    return df


# =============================================================================
# PASO 6: ENSAMBLAJE — AÑADIR FEATURES AL DATASET DE PARTIDOS
# =============================================================================

def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función principal de la Fase 2.

    Orquesta todos los pasos anteriores y retorna el dataset de partidos
    con todas las features históricas adjuntas, listo para:
        - Entrenamiento del Random Forest (con target = result)
        - Construcción del vector de perfil para K-Means (agregación por equipo)

    Flujo:
        1. Ordenar cronológicamente
        2. Codificar fase del torneo
        3. Calcular estadísticas históricas acumuladas
        4. Calcular forma reciente
        5. Calcular apariciones en WC previas
        6. Unir todas las features al dataset principal
        7. Calcular features comparativas (delta)
        8. Imputar NaN del primer partido de cada equipo

    Parameters
    ----------
    df : pd.DataFrame
        Dataset enriquecido de Fase 1 (con columna 'result').

    Returns
    -------
    pd.DataFrame
        Dataset completo con features históricas, listo para modelado.
    """
    print("\n" + "─" * 62)
    print("  ENSAMBLAJE DE FEATURES HISTÓRICAS")
    print("─" * 62)

    # ── 1. Orden temporal ────────────────────────────────────────
    print("\n  [1/8] Ordenando cronológicamente...")
    df = sort_chronologically(df)

    # ── 2. Codificación de fases ─────────────────────────────────
    print("\n  [2/8] Codificando fases del torneo...")
    df = encode_stage(df)
    print(f"   stage_encoded — distribución:")
    if "stage_encoded" in df.columns:
        dist = df["stage_encoded"].value_counts().sort_index()
        for val, cnt in dist.items():
            label = {v: k for k, v in STAGE_ORDER.items()}.get(val, f"código {val}")
            print(f"       {val}: {cnt:>4} partidos  ← {label}")

    # ── 3. Estadísticas históricas acumuladas ────────────────────
    print("\n  [3/8] Calculando estadísticas históricas acumuladas...")
    hist_stats = build_historical_stats(df)
    print(f"   hist_stats generado: {hist_stats.shape}")

    # Unir perspectiva LOCAL
    home_hist = hist_stats.rename(columns={
        "hist_win_rate":  "home_hist_win_rate",
        "hist_draw_rate": "home_hist_draw_rate",
        "hist_avg_gf":    "home_hist_avg_gf",
        "hist_avg_ga":    "home_hist_avg_ga",
        "hist_avg_gd":    "home_hist_avg_gd",
        "hist_matches":   "home_hist_matches",
    })
    df = df.merge(
        home_hist[["match_index", "team",
                   "home_hist_win_rate", "home_hist_draw_rate",
                   "home_hist_avg_gf", "home_hist_avg_ga",
                   "home_hist_avg_gd", "home_hist_matches"]],
        left_on=["match_index", "home_team"],
        right_on=["match_index", "team"],
        how="left"
    ).drop(columns=["team"])

    # Unir perspectiva VISITANTE
    away_hist = hist_stats.rename(columns={
        "hist_win_rate":  "away_hist_win_rate",
        "hist_draw_rate": "away_hist_draw_rate",
        "hist_avg_gf":    "away_hist_avg_gf",
        "hist_avg_ga":    "away_hist_avg_ga",
        "hist_avg_gd":    "away_hist_avg_gd",
        "hist_matches":   "away_hist_matches",
    })
    df = df.merge(
        away_hist[["match_index", "team",
                   "away_hist_win_rate", "away_hist_draw_rate",
                   "away_hist_avg_gf", "away_hist_avg_ga",
                   "away_hist_avg_gd", "away_hist_matches"]],
        left_on=["match_index", "away_team"],
        right_on=["match_index", "team"],
        how="left"
    ).drop(columns=["team"])

    # ── 4. Forma reciente ─────────────────────────────────────────
    print(f"\n  [4/8] Calculando forma reciente (ventana={RECENT_WINDOW})...")
    recent = build_recent_form(df, window=RECENT_WINDOW)

    # Forma reciente local
    df = df.merge(
        recent[["match_index", "team", "recent_win_rate"]].rename(
            columns={"team": "_team_h",
                     "recent_win_rate": "home_recent_win_rate"}
        ),
        left_on=["match_index", "home_team"],
        right_on=["match_index", "_team_h"],
        how="left"
    ).drop(columns=["_team_h"])

    # Forma reciente visitante
    df = df.merge(
        recent[["match_index", "team", "recent_win_rate"]].rename(
            columns={"team": "_team_a",
                     "recent_win_rate": "away_recent_win_rate"}
        ),
        left_on=["match_index", "away_team"],
        right_on=["match_index", "_team_a"],
        how="left"
    ).drop(columns=["_team_a"])
    print("   Forma reciente añadida.")

    # ── 5. Apariciones en WC previas ─────────────────────────────
    print("\n  [5/8] Calculando apariciones en Copas del Mundo previas...")
    if "Year" in df.columns:
        wc_app = build_wc_appearances(df)

        df = df.merge(
            wc_app.rename(columns={
                "team": "home_team",
                "wc_appearances_before": "home_wc_appearances"
            }),
            on=["home_team", "Year"], how="left"
        )
        df = df.merge(
            wc_app.rename(columns={
                "team": "away_team",
                "wc_appearances_before": "away_wc_appearances"
            }),
            on=["away_team", "Year"], how="left"
        )
        df["home_wc_appearances"] = df["home_wc_appearances"].fillna(0).astype(int)
        df["away_wc_appearances"] = df["away_wc_appearances"].fillna(0).astype(int)
        print("   wc_appearances añadido.")
    else:
        df["home_wc_appearances"] = 0
        df["away_wc_appearances"] = 0

    # ── 6. Features comparativas (delta) ─────────────────────────
    print("\n  [6/8] Calculando features comparativas (delta)...")
    df["delta_win_rate"]  = df["home_hist_win_rate"] - df["away_hist_win_rate"]
    df["delta_avg_gf"]    = df["home_hist_avg_gf"]   - df["away_hist_avg_gf"]
    df["delta_avg_gd"]    = df["home_hist_avg_gd"]   - df["away_hist_avg_gd"]
    df["delta_wc_exp"]    = df["home_wc_appearances"] - df["away_wc_appearances"]
    print("   delta_win_rate, delta_avg_gf, delta_avg_gd, delta_wc_exp.")

    # ── 7. Imputación de NaN del primer partido ───────────────────
    # El primer partido de cada equipo en la Copa no tiene historial previo.
    # Se imputan con 0 (estadística neutra):
    #   win_rate = 0   → equipo sin victorias previas conocidas
    #   avg_gf   = 0   → sin goles a favor históricos
    # Esto es conservador y evita sesgo (no inventar valores).
    print("\n  [7/8] Imputando NaN por ausencia de historial previo...")
    hist_cols = [c for c in df.columns if any(
        c.startswith(p) for p in
        ["home_hist_", "away_hist_", "home_recent_", "away_recent_",
         "delta_", "home_wc_", "away_wc_"]
    )]
    for col in hist_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            df[col] = df[col].fillna(0)
    print(f"   {len(hist_cols)} columnas históricas imputadas con 0 donde NaN.")

    # ── 8. Reporte final ──────────────────────────────────────────
    print("\n  [8/8] Reporte de features generadas:")
    feature_cols = [c for c in df.columns if any(
        c.startswith(p) for p in
        ["home_hist_", "away_hist_", "home_recent_", "away_recent_",
         "delta_", "home_wc_", "away_wc_", "stage_encoded"]
    )]
    print(f"  Total features históricas generadas: {len(feature_cols)}")
    for col in feature_cols:
        null_pct = df[col].isna().mean() * 100
        print(f"    • {col:<35} dtype={str(df[col].dtype):<10} nulos={null_pct:.1f}%")

    return df


# =============================================================================
# CONSTRUCCIÓN DEL VECTOR DE PERFIL PARA K-MEANS
# =============================================================================

def build_kmeans_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el vector de perfil histórico GLOBAL por selección,
    para usar como input de K-Means.

    A diferencia del Random Forest, K-Means NO es predictivo:
    agrupa selecciones por similitud de su desempeño histórico TOTAL.
    Por tanto, aquí SÍ podemos usar todas las estadísticas disponibles
    (incluyendo goles) sin riesgo de leakage (no hay target futuro).

    Features del perfil K-Means (por equipo):
        total_matches           Total de partidos en WC
        win_rate                Tasa de victorias global
        draw_rate               Tasa de empates global
        loss_rate               Tasa de derrotas global
        avg_goals_for           Promedio goles a favor
        avg_goals_against       Promedio goles en contra
        avg_goal_diff           Diferencia de goles media
        num_tournaments         Copas del Mundo en que participó
        best_stage              Fase máxima alcanzada (valor ordinal)
        goals_per_tournament    Goles totales / torneos disputados

    Returns
    -------
    pd.DataFrame
        Una fila por selección con sus métricas históricas globales.
    """
    # Tabla larga de apariciones
    home = df[["home_team", "home_score", "away_score", "result",
               "stage_encoded"]].copy()
    home.columns = ["team", "gf", "ga", "result", "stage_encoded"]
    home["is_win"]  = (home["result"] == "home_win").astype(float)
    home["is_draw"] = (home["result"] == "draw").astype(float)
    home["is_loss"] = (home["result"] == "away_win").astype(float)

    away = df[["away_team", "away_score", "home_score", "result",
               "stage_encoded"]].copy()
    away.columns = ["team", "gf", "ga", "result", "stage_encoded"]
    away["is_win"]  = (away["result"] == "away_win").astype(float)
    away["is_draw"] = (away["result"] == "draw").astype(float)
    away["is_loss"] = (away["result"] == "home_win").astype(float)

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["gd"] = long_df["gf"] - long_df["ga"]

    # Agregar por equipo
    profile = long_df.groupby("team").agg(
        total_matches   = ("is_win",         "count"),
        win_rate        = ("is_win",          "mean"),
        draw_rate       = ("is_draw",         "mean"),
        loss_rate       = ("is_loss",         "mean"),
        avg_goals_for   = ("gf",              "mean"),
        avg_goals_against=("ga",              "mean"),
        avg_goal_diff   = ("gd",              "mean"),
        best_stage      = ("stage_encoded",   "max"),
        total_goals_for = ("gf",              "sum"),
    ).reset_index()

    # Número de torneos distintos en que participó cada equipo
    if "Year" in df.columns:
        home_years = df[["home_team", "Year"]].rename(columns={"home_team": "team"})
        away_years = df[["away_team", "Year"]].rename(columns={"away_team": "team"})
        team_years = pd.concat([home_years, away_years]).drop_duplicates()
        n_tournaments = team_years.groupby("team")["Year"].nunique().reset_index()
        n_tournaments.columns = ["team", "num_tournaments"]
        profile = profile.merge(n_tournaments, on="team", how="left")
        profile["goals_per_tournament"] = (
            profile["total_goals_for"] / profile["num_tournaments"]
        )
    else:
        profile["num_tournaments"]    = 1
        profile["goals_per_tournament"] = profile["total_goals_for"]

    # Eliminar columna auxiliar
    profile = profile.drop(columns=["total_goals_for"])

    print(f"\n  Perfil K-Means construido: {profile.shape[0]} selecciones × "
          f"{profile.shape[1]-1} features")

    return profile