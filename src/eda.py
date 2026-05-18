# Módulo para el Análisis Exploratorio de Datos (EDA)
"""
==============================================================================
FASE 1 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/eda.py

Descripción:
    Análisis exploratorio completo y profesional del dataset piterfm.
    Genera visualizaciones, detecta inconsistencias y produce hallazgos
    técnicos relevantes para la Fase 2 (Preprocesamiento) y el modelado.

Cobertura del EDA:
    1.  Inspección inicial de estructura
    2.  Variable objetivo: distribución de resultados
    3.  Análisis temporal: evolución por torneo
    4.  Análisis de goles (distribución, promedio, extremos)
    5.  Ventaja de localía
    6.  Fases del torneo
    7.  Selecciones: participación y rendimiento histórico
    8.  Análisis de nulos y estrategia de imputación
    9.  Correlaciones entre variables numéricas
    10. Identificación de variables para RF y K-Means
    11. Riesgos de data leakage
    12. Conclusiones y recomendaciones

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado
Universidad: UPTC
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN VISUAL GLOBAL
# =============================================================================

# Paleta académica coherente con el proyecto
PALETTE = {
    "primary":    "#1a3a5c",   # Azul oscuro FIFA
    "secondary":  "#c8a951",   # Dorado Copa
    "accent":     "#c0392b",   # Rojo alerta
    "neutral":    "#7f8c8d",   # Gris medio
    "light":      "#ecf0f1",   # Fondo claro
    "win":        "#27ae60",   # Victoria
    "draw":       "#f39c12",   # Empate
    "loss":       "#c0392b",   # Derrota
    "bg":         "#0d1b2a",   # Fondo oscuro para portada
}

FONT_TITLE  = {"fontsize": 15, "fontweight": "bold", "color": PALETTE["primary"]}
FONT_LABEL  = {"fontsize": 11, "color": "#2c3e50"}
FONT_TICK   = {"labelsize": 9}

OUTPUT_DIR = "outputs/eda_plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig: plt.Figure, filename: str) -> None:
    """Guarda figura en alta resolución."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"   Guardado: {path}")
    plt.close(fig)


# =============================================================================
# SECCIÓN 0: PREPARACIÓN — COLUMNAS DERIVADAS
# =============================================================================

def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas derivadas esenciales para el EDA.

     AVISO DE DATA LEAKAGE:
        home_xg y away_xg son métricas post-partido (NO se usarán como
        features en el modelo predictivo). Se documentan aquí solo para EDA.

    Columnas añadidas
    -----------------
    result        : 'home_win' | 'draw' | 'away_win'
    total_goals   : home_score + away_score
    goal_diff     : home_score - away_score (perspectiva local)
    went_to_pen   : bool — partido se definió en penales
    Year          : int — si no existe ya en el DF
    """
    df = df.copy()

    # --- Variable objetivo (TARGET para Random Forest) ---
    # Se deriva de home_score y away_score ÚNICAMENTE
    def classify_result(row):
        if row["home_score"] > row["away_score"]:
            return "home_win"
        elif row["home_score"] == row["away_score"]:
            return "draw"
        else:
            return "away_win"

    df["result"]      = df.apply(classify_result, axis=1)
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"]   = df["home_score"] - df["away_score"]

    # Partidos definidos por penales (columna presente solo en piterfm dataset)
    if "home_penalty" in df.columns and "away_penalty" in df.columns:
        df["went_to_pen"] = (
            df["home_penalty"].notna() | df["away_penalty"].notna()
        )
    else:
        df["went_to_pen"] = False

    # Año del torneo (columna suele venir incluida; si no, se crea como NaN)
    if "Year" not in df.columns:
        if "date" in df.columns:
            df["Year"] = pd.to_datetime(
                df["date"], errors="coerce"
            ).dt.year
        else:
            df["Year"] = np.nan
            print("  Columna 'Year' no encontrada. Se requiere verificación manual.")

    return df


# =============================================================================
# SECCIÓN 1: VARIABLE OBJETIVO
# =============================================================================

def plot_result_distribution(df: pd.DataFrame) -> None:
    """
    Gráfico 1: Distribución de la variable objetivo (result).

    Muestra la proporción de victorias locales, empates y victorias visitantes
    en los 900+ partidos históricos de la Copa Mundial.
    """
    print("\n[1/9] Distribución de la variable objetivo...")

    counts = df["result"].value_counts()
    labels_map = {
        "home_win":  "Victoria Local",
        "draw":      "Empate",
        "away_win":  "Victoria Visitante",
    }
    colors = [PALETTE["win"], PALETTE["draw"], PALETTE["loss"]]
    order  = ["home_win", "draw", "away_win"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribución de Resultados — Copa Mundial FIFA 1930–2022",
                 **FONT_TITLE, y=1.02)

    # --- Barplot ---
    ax1 = axes[0]
    vals  = [counts.get(k, 0) for k in order]
    bars  = ax1.bar(
        [labels_map[k] for k in order], vals,
        color=colors, edgecolor="white", linewidth=1.5, width=0.55
    )
    for bar, val in zip(bars, vals):
        pct = val / counts.sum() * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val}\n({pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=PALETTE["primary"]
        )
    ax1.set_ylabel("Número de Partidos", **FONT_LABEL)
    ax1.set_title("Conteo Absoluto y Porcentaje", fontsize=11)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.tick_params(**FONT_TICK)
    sns.despine(ax=ax1)

    # --- Pie chart ---
    ax2 = axes[1]
    wedge_props = {"linewidth": 2, "edgecolor": "white"}
    wedges, texts, autotexts = ax2.pie(
        vals,
        labels=[labels_map[k] for k in order],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=wedge_props,
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title("Proporción Relativa", fontsize=11)

    plt.tight_layout()
    _save(fig, "01_result_distribution.png")

    # Hallazgo textual
    total = counts.sum()
    hw_pct = counts.get("home_win", 0) / total * 100
    draw_pct = counts.get("draw", 0) / total * 100
    aw_pct = counts.get("away_win", 0) / total * 100
    print(f"""
   HALLAZGO: Distribución de resultados
     • Victoria local  : {hw_pct:.1f}%  ← clase mayoritaria
     • Empate          : {draw_pct:.1f}%  ← clase minoritaria (MÁS DIFÍCIL DE PREDECIR)
     • Victoria visitante: {aw_pct:.1f}%
       Desbalance de clases: requiere estrategia (class_weight o SMOTE) en RF.
""")


# =============================================================================
# SECCIÓN 2: ANÁLISIS TEMPORAL
# =============================================================================

def plot_temporal_analysis(df: pd.DataFrame) -> None:
    """
    Gráfico 2: Evolución temporal de goles y partidos por torneo.
    """
    print("\n[2/9] Análisis temporal...")

    if "Year" not in df.columns or df["Year"].isna().all():
        print("    Columna 'Year' no disponible. Saltando análisis temporal.")
        return

    yearly = df.groupby("Year").agg(
        partidos      = ("result", "count"),
        goles_totales = ("total_goals", "sum"),
        goles_prom    = ("total_goals", "mean"),
        victorias_loc = ("result", lambda x: (x == "home_win").sum()),
        empates       = ("result", lambda x: (x == "draw").sum()),
        victorias_vis = ("result", lambda x: (x == "away_win").sum()),
    ).reset_index()

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    fig.suptitle("Evolución Temporal — Copa Mundial FIFA 1930–2022",
                 **FONT_TITLE, y=1.01)

    # 2a. Partidos por torneo
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(yearly["Year"], yearly["partidos"],
            color=PALETTE["primary"], alpha=0.85, width=2.5)
    ax1.set_title("Partidos Disputados por Torneo", fontsize=11)
    ax1.set_ylabel("Nº de partidos", **FONT_LABEL)
    ax1.tick_params(**FONT_TICK)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1)

    # 2b. Promedio de goles por partido
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(yearly["Year"], yearly["goles_prom"],
             color=PALETTE["secondary"], linewidth=2.5, marker="o",
             markersize=6, markerfacecolor="white", markeredgewidth=2)
    ax2.axhline(yearly["goles_prom"].mean(), color=PALETTE["accent"],
                linestyle="--", linewidth=1.5, label="Promedio global")
    ax2.set_title("Promedio de Goles por Partido", fontsize=11)
    ax2.set_ylabel("Goles / partido", **FONT_LABEL)
    ax2.legend(fontsize=9)
    ax2.tick_params(**FONT_TICK)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2)

    # 2c. Resultados apilados por torneo
    ax3 = fig.add_subplot(gs[1, :])
    bar_w = 2.5
    ax3.bar(yearly["Year"], yearly["victorias_loc"],
            width=bar_w, label="Victoria Local",   color=PALETTE["win"],  alpha=0.9)
    ax3.bar(yearly["Year"], yearly["empates"],
            width=bar_w, bottom=yearly["victorias_loc"],
            label="Empate", color=PALETTE["draw"], alpha=0.9)
    ax3.bar(yearly["Year"],
            yearly["victorias_vis"],
            width=bar_w,
            bottom=yearly["victorias_loc"] + yearly["empates"],
            label="Victoria Visitante", color=PALETTE["loss"], alpha=0.9)
    ax3.set_title("Distribución de Resultados por Torneo (apilada)", fontsize=11)
    ax3.set_ylabel("Nº de partidos", **FONT_LABEL)
    ax3.set_xlabel("Año del torneo", **FONT_LABEL)
    ax3.legend(fontsize=9, loc="upper left")
    ax3.tick_params(**FONT_TICK)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)
    sns.despine(ax=ax3)

    _save(fig, "02_temporal_analysis.png")
    print("   HALLAZGO: Los torneos recientes (1998–2022) tienen el doble de\n"
          "     partidos que las ediciones anteriores a 1982 (expansión del torneo).")


# =============================================================================
# SECCIÓN 3: DISTRIBUCIÓN DE GOLES
# =============================================================================

def plot_goals_analysis(df: pd.DataFrame) -> None:
    """
    Gráfico 3: Distribución de goles por partido y por equipo.
    """
    print("\n[3/9] Análisis de goles...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Distribución de Goles — Copa Mundial FIFA 1930–2022",
                 **FONT_TITLE, y=1.02)

    # 3a. Histograma goles totales por partido
    ax1 = axes[0]
    ax1.hist(df["total_goals"].dropna(), bins=range(0, 15),
             color=PALETTE["primary"], edgecolor="white", linewidth=1.2, density=False)
    ax1.axvline(df["total_goals"].mean(), color=PALETTE["secondary"],
                linestyle="--", linewidth=2,
                label=f'Media: {df["total_goals"].mean():.2f}')
    ax1.axvline(df["total_goals"].median(), color=PALETTE["accent"],
                linestyle=":", linewidth=2,
                label=f'Mediana: {df["total_goals"].median():.1f}')
    ax1.set_title("Goles Totales por Partido", fontsize=11)
    ax1.set_xlabel("Goles en el partido", **FONT_LABEL)
    ax1.set_ylabel("Frecuencia", **FONT_LABEL)
    ax1.legend(fontsize=9)
    ax1.tick_params(**FONT_TICK)
    sns.despine(ax=ax1)

    # 3b. Goles equipo local
    ax2 = axes[1]
    counts_home = df["home_score"].dropna().value_counts().sort_index()
    ax2.bar(counts_home.index.astype(int), counts_home.values,
            color=PALETTE["win"], alpha=0.85, edgecolor="white")
    ax2.set_title("Goles del Equipo Local", fontsize=11)
    ax2.set_xlabel("Goles marcados", **FONT_LABEL)
    ax2.set_ylabel("Frecuencia", **FONT_LABEL)
    ax2.tick_params(**FONT_TICK)
    sns.despine(ax=ax2)

    # 3c. Goles equipo visitante
    ax3 = axes[2]
    counts_away = df["away_score"].dropna().value_counts().sort_index()
    ax3.bar(counts_away.index.astype(int), counts_away.values,
            color=PALETTE["loss"], alpha=0.85, edgecolor="white")
    ax3.set_title("Goles del Equipo Visitante", fontsize=11)
    ax3.set_xlabel("Goles marcados", **FONT_LABEL)
    ax3.set_ylabel("Frecuencia", **FONT_LABEL)
    ax3.tick_params(**FONT_TICK)
    sns.despine(ax=ax3)

    plt.tight_layout()
    _save(fig, "03_goals_distribution.png")

    # Estadísticas de goles
    print(f"""
   HALLAZGO: Estadísticas de goles
     • Promedio goles/partido      : {df['total_goals'].mean():.2f}
     • Mediana goles/partido       : {df['total_goals'].median():.1f}
     • Máximo goles en un partido  : {int(df['total_goals'].max())} (outlier)
     • Partidos sin goles (0–0)    : {(df['total_goals'] == 0).sum()}
     • Equipo local anota en media : {df['home_score'].mean():.2f} goles
     • Equipo visitante anota media: {df['away_score'].mean():.2f} goles
""")


# =============================================================================
# SECCIÓN 4: VENTAJA DE LOCALÍA
# =============================================================================

def plot_home_advantage(df: pd.DataFrame) -> None:
    """
    Gráfico 4: Cuantificación de la ventaja de jugar como equipo local.
    """
    print("\n[4/9] Ventaja de localía...")

    # Tasa de victoria local vs visitante
    total     = len(df)
    home_wins = (df["result"] == "home_win").sum()
    away_wins = (df["result"] == "away_win").sum()
    draws     = (df["result"] == "draw").sum()

    # Promedio de goles por condición
    avg_home = df["home_score"].mean()
    avg_away = df["away_score"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ventaja de Localía — Copa Mundial FIFA", **FONT_TITLE, y=1.02)

    # 4a. Tasa de victorias por condición
    ax1 = axes[0]
    categories = ["Local\n(home_win)", "Empate\n(draw)", "Visitante\n(away_win)"]
    rates = [home_wins/total*100, draws/total*100, away_wins/total*100]
    bars = ax1.bar(categories, rates,
                   color=[PALETTE["win"], PALETTE["draw"], PALETTE["loss"]],
                   width=0.5, edgecolor="white", linewidth=1.5)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{rate:.1f}%",
                 ha="center", fontsize=11, fontweight="bold",
                 color=PALETTE["primary"])
    ax1.set_ylabel("Tasa (%)", **FONT_LABEL)
    ax1.set_title("Tasa de Resultados Histórica", fontsize=11)
    ax1.set_ylim(0, max(rates) * 1.2)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1)

    # 4b. Goles promedio: local vs visitante
    ax2 = axes[1]
    x     = [0, 1]
    avgs  = [avg_home, avg_away]
    clrs  = [PALETTE["win"], PALETTE["loss"]]
    lbls  = ["Equipo Local", "Equipo Visitante"]
    bars2 = ax2.bar(lbls, avgs, color=clrs, width=0.4,
                    edgecolor="white", linewidth=1.5)
    for bar, avg in zip(bars2, avgs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{avg:.3f} goles",
                 ha="center", fontsize=11, fontweight="bold",
                 color=PALETTE["primary"])
    ax2.set_ylabel("Goles promedio por partido", **FONT_LABEL)
    ax2.set_title("Promedio de Goles por Condición", fontsize=11)
    ax2.set_ylim(0, max(avgs) * 1.3)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2)

    plt.tight_layout()
    _save(fig, "04_home_advantage.png")

    print(f"""
   HALLAZGO: Ventaja de localía
     • Local gana en promedio {home_wins/total*100:.1f}% de los partidos
     • Visitante gana solo el {away_wins/total*100:.1f}%
     • Diferencia de goles: Local {avg_home:.3f} vs Visitante {avg_away:.3f}
     → 'home_team' como feature proxy de localía es válido y no genera leakage.
""")


# =============================================================================
# SECCIÓN 5: FASES DEL TORNEO
# =============================================================================

def plot_stage_analysis(df: pd.DataFrame) -> None:
    """
    Gráfico 5: Análisis de resultados y goles según fase del torneo.

    Requiere columna 'stage' o 'round' en el dataset.
    """
    print("\n[5/9] Análisis por fase del torneo...")

    # Detectar columna de fase
    stage_col = None
    for candidate in ["stage", "round", "phase", "match_phase"]:
        if candidate in df.columns:
            stage_col = candidate
            break

    if stage_col is None:
        print(f"    No se encontró columna de fase del torneo "
              f"(stage/round/phase). Saltando sección 5.")
        return

    stage_stats = df.groupby(stage_col).agg(
        partidos   = ("result", "count"),
        goles_prom = ("total_goals", "mean"),
        local_win  = ("result", lambda x: (x == "home_win").mean() * 100),
        draw_pct   = ("result", lambda x: (x == "draw").mean() * 100),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Análisis por Fase del Torneo", **FONT_TITLE, y=1.02)

    # 5a. Promedio de goles por fase
    ax1 = axes[0]
    stage_sorted = stage_stats.sort_values("goles_prom", ascending=True)
    colors_bar = [PALETTE["primary"]] * len(stage_sorted)
    ax1.barh(stage_sorted[stage_col], stage_sorted["goles_prom"],
             color=colors_bar, edgecolor="white")
    ax1.set_xlabel("Promedio de goles", **FONT_LABEL)
    ax1.set_title("Promedio de Goles por Fase", fontsize=11)
    ax1.tick_params(**FONT_TICK)
    sns.despine(ax=ax1)

    # 5b. Tasa de empates por fase
    ax2 = axes[1]
    stage_sorted2 = stage_stats.sort_values("draw_pct", ascending=True)
    ax2.barh(stage_sorted2[stage_col], stage_sorted2["draw_pct"],
             color=PALETTE["draw"], alpha=0.85, edgecolor="white")
    ax2.set_xlabel("% de empates en tiempo regular", **FONT_LABEL)
    ax2.set_title("Tasa de Empates por Fase", fontsize=11)
    ax2.tick_params(**FONT_TICK)
    sns.despine(ax=ax2)

    plt.tight_layout()
    _save(fig, "05_stage_analysis.png")
    print("   HALLAZGO: Las fases eliminatorias suelen tener menos empates\n"
          "     en tiempo regular (muchos se definen por penales o prórroga).")


# =============================================================================
# SECCIÓN 6: SELECCIONES — PARTICIPACIÓN Y RENDIMIENTO
# =============================================================================

def plot_teams_analysis(df: pd.DataFrame) -> None:
    """
    Gráfico 6: Top selecciones por partidos jugados y tasa de victoria.
    Relevante especialmente para K-Means (perfiles de rendimiento).
    """
    print("\n[6/9] Análisis de selecciones...")

    # Construir tabla de rendimiento histórico por equipo
    # Cada partido aparece dos veces: como local y como visitante
    home_df = df[["home_team", "home_score", "away_score", "result"]].copy()
    home_df.columns = ["team", "goals_for", "goals_against", "match_result"]
    home_df["is_win"]  = (home_df["match_result"] == "home_win").astype(int)
    home_df["is_draw"] = (home_df["match_result"] == "draw").astype(int)
    home_df["is_loss"] = (home_df["match_result"] == "away_win").astype(int)

    away_df = df[["away_team", "away_score", "home_score", "result"]].copy()
    away_df.columns = ["team", "goals_for", "goals_against", "match_result"]
    away_df["is_win"]  = (away_df["match_result"] == "away_win").astype(int)
    away_df["is_draw"] = (away_df["match_result"] == "draw").astype(int)
    away_df["is_loss"] = (away_df["match_result"] == "home_win").astype(int)

    all_df = pd.concat([home_df, away_df], ignore_index=True)

    team_stats = all_df.groupby("team").agg(
        partidos         = ("is_win", "count"),
        victorias        = ("is_win", "sum"),
        empates          = ("is_draw", "sum"),
        derrotas         = ("is_loss", "sum"),
        goles_favor      = ("goals_for", "sum"),
        goles_contra     = ("goals_against", "sum"),
    ).reset_index()

    team_stats["win_rate"]   = team_stats["victorias"] / team_stats["partidos"]
    team_stats["goal_diff"]  = team_stats["goles_favor"] - team_stats["goles_contra"]
    team_stats["avg_goals"]  = team_stats["goles_favor"] / team_stats["partidos"]

    top20_part = team_stats.nlargest(20, "partidos")
    top20_wr   = team_stats[team_stats["partidos"] >= 10].nlargest(20, "win_rate")

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle("Selecciones Nacionales — Rendimiento Histórico",
                 **FONT_TITLE, y=1.01)

    # 6a. Top 20 por partidos jugados
    ax1 = axes[0]
    cmap_val = top20_part["win_rate"].values
    norm = plt.Normalize(cmap_val.min(), cmap_val.max())
    colors = plt.cm.RdYlGn(norm(cmap_val))

    bars = ax1.barh(top20_part["team"][::-1],
                    top20_part["partidos"][::-1],
                    color=colors[::-1], edgecolor="white", linewidth=0.8)
    ax1.set_xlabel("Partidos disputados en Copas del Mundo", **FONT_LABEL)
    ax1.set_title("Top 20 Selecciones — Mayor Participación Histórica\n"
                  "(Color = Tasa de victoria)", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.01, shrink=0.8)
    cbar.set_label("Tasa de victoria", fontsize=9)
    ax1.tick_params(**FONT_TICK)
    sns.despine(ax=ax1)

    # 6b. Top 20 por tasa de victoria (mín. 10 partidos)
    ax2 = axes[1]
    ax2.barh(top20_wr["team"][::-1], top20_wr["win_rate"][::-1] * 100,
             color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    ax2.axvline(50, color=PALETTE["accent"], linestyle="--",
                linewidth=1.5, label="50% (referencia)")
    ax2.set_xlabel("Tasa de victoria (%)", **FONT_LABEL)
    ax2.set_title("Top 20 Selecciones — Mayor Tasa de Victoria (mín. 10 partidos)",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.tick_params(**FONT_TICK)
    sns.despine(ax=ax2)

    plt.tight_layout()
    _save(fig, "06_teams_analysis.png")

    print(f"""
   HALLAZGO: Perfiles de selecciones
     • Total de selecciones únicas : {team_stats['team'].nunique()}
     • Equipo con más partidos     : {top20_part.iloc[0]['team']} ({int(top20_part.iloc[0]['partidos'])} partidos)
     • Mayor tasa de victoria (≥10): {top20_wr.iloc[0]['team']} ({top20_wr.iloc[0]['win_rate']:.1%})
     → Esta tabla se usará como base para el vector de features de K-Means.
""")

    return team_stats   # Retorna para reutilizar en otras secciones


# =============================================================================
# SECCIÓN 7: ANÁLISIS DE NULOS
# =============================================================================

def plot_nulls_analysis(df: pd.DataFrame) -> None:
    """
    Gráfico 7: Mapa de calor de valores nulos con estrategia de imputación.
    """
    print("\n[7/9] Análisis de valores nulos...")

    null_pct = df.isnull().mean() * 100
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False)

    if null_pct.empty:
        print("   No se detectaron valores nulos en el dataset de partidos.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(null_pct) * 0.4)))
    fig.suptitle("Valores Nulos por Columna — matches_1930_2022.csv",
                 **FONT_TITLE)

    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#27ae60"
              for v in null_pct.values]

    bars = ax.barh(null_pct.index[::-1], null_pct.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, null_pct.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9,
                color=PALETTE["primary"], fontweight="bold")

    ax.set_xlabel("% de valores nulos", **FONT_LABEL)
    ax.axvline(20, color=PALETTE["accent"], linestyle="--",
               alpha=0.7, label="Umbral 20%")
    ax.axvline(50, color="#8e44ad", linestyle="--",
               alpha=0.7, label="Umbral 50%")

    legend_elements = [
        Patch(color="#27ae60", label="< 20%  → Imputar (mediana/moda)"),
        Patch(color="#f39c12", label="20–50% → Evaluar + imputar/eliminar"),
        Patch(color="#e74c3c", label="> 50%  → Eliminar como feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.tick_params(**FONT_TICK)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, "07_nulls_analysis.png")

    print("   Columnas con nulos detectadas:")
    for col, pct in null_pct.items():
        strategy = ("Imputar (mediana/moda)" if pct <= 20
                    else "Evaluar caso por caso" if pct <= 50
                    else "Eliminar como feature predictivo")
        print(f"     {col:<30} {pct:5.1f}%  → {strategy}")


# =============================================================================
# SECCIÓN 8: CORRELACIONES
# =============================================================================

def plot_correlations(df: pd.DataFrame) -> None:
    """
    Gráfico 8: Matriz de correlación entre variables numéricas.

     AVISO: xg (Expected Goals) se incluye en el EDA pero NO en el modelo.
    """
    print("\n[8/9] Análisis de correlaciones...")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Excluir la variable objetivo ya codificada si existe
    exclude = []
    num_cols = [c for c in num_cols if c not in exclude]

    if len(num_cols) < 2:
        print("  ⚠️  Insuficientes columnas numéricas para matriz de correlación.")
        return

    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
    fig.suptitle("Matriz de Correlación — Variables Numéricas",
                 **FONT_TITLE, y=1.02)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title(
        "  home_xg / away_xg: métricas POST-partido — NO usar en modelo predictivo",
        fontsize=9, color=PALETTE["accent"], pad=8
    )
    ax.tick_params(axis="x", rotation=45, **FONT_TICK)
    ax.tick_params(axis="y", rotation=0, **FONT_TICK)

    plt.tight_layout()
    _save(fig, "08_correlations.png")

    # Pares con alta correlación
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:
                high_corr.append((corr_matrix.columns[i],
                                  corr_matrix.columns[j], val))

    if high_corr:
        print("  📊 Pares con correlación |r| > 0.70:")
        for c1, c2, r in high_corr:
            print(f"     {c1} ↔ {c2}: r = {r:.3f}")
    else:
        print("   No hay pares con correlación excesiva (|r| > 0.70) "
              "entre features candidatos.")


# =============================================================================
# SECCIÓN 9: RESUMEN — FEATURES PARA RF Y K-MEANS
# =============================================================================

def print_feature_recommendations() -> None:
    """
    Imprime el análisis de variables relevantes para cada modelo
    y los riesgos de data leakage identificados.
    """
    print("\n" + "=" * 68)
    print("  RESUMEN: VARIABLES RELEVANTES Y RIESGOS DE DATA LEAKAGE")
    print("=" * 68)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  VARIABLE OBJETIVO (TARGET)                                         │
│  result: 'home_win' | 'draw' | 'away_win'                           │
│  Derivada de: home_score y away_score (NUNCA como features)         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  FEATURES PARA RANDOM FOREST (supervisado)                          │
│  ─────────────────────────────────────────────────────────────────  │
│ SEGURAS (no generan leakage)                                        │
│    • home_team              (equipo local — identidad)              │
│    • away_team              (equipo visitante — identidad)          │
│    • year                   (época del torneo)                      │
│    • stage / round          (fase del torneo)                       │
│    • [feat. engineered]     victorias históricas previas del equipo │
│    • [feat. engineered]     promedio goles previos del equipo       │
│    • [feat. engineered]     diferencia goles histórica              │
│    • [feat. engineered]     participaciones previas en WC           │
│                                                                     │
│ PROHIBIDAS (data leakage — métricas POST-partido)                   │
│    • home_score, away_score  ← son el RESULTADO del partido         │
│    • home_xg, away_xg        ← Expected Goals = métrica del juego   │
│    • home_goal, away_goal    ← info post-partido                    │
│    • home_own_goal, etc.     ← info post-partido                    │
│    • total_goals, goal_diff  ← derivadas del resultado              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  FEATURES PARA K-MEANS (no supervisado — perfil de selecciones)     │
│  Se agregan históricamente POR EQUIPO (no por partido):             │
│  ─────────────────────────────────────────────────────────────────  │
│  • total_partidos_wc     Total de partidos en Copas del Mundo       │
│  • win_rate_wc           Tasa de victorias histórica                │
│  • draw_rate_wc          Tasa de empates histórica                  │
│  • avg_goals_for_wc      Promedio goles a favor por partido         │
│  • avg_goals_against_wc  Promedio goles en contra por partido       │
│  • avg_goal_diff_wc      Diferencia de goles media                  │
│  • num_torneos_wc        Cantidad de torneos en que participó       │
│  • best_stage            Fase máxima alcanzada (codificada)         │
│                                                                     │
│  NOTA: K-Means NO tiene target. Todas las features son históricas   │
│  y de carácter descriptivo. No hay riesgo de leakage aquí.          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  RIESGOS DE DATA LEAKAGE IDENTIFICADOS                              │
│  ─────────────────────────────────────────────────────────────────  │
│  RIESGO 1 (CRÍTICO): Usar home_score / away_score como features.    │
│  Son el resultado del partido → filtrar antes del Pipeline.         │
│                                                                     │
│  RIESGO 2 (ALTO): Usar xg (Expected Goals) como feature.            │
│  xG se calcula durante el partido → no disponible pre-partido.      │
│                                                                     │
│  RIESGO 3 (MEDIO): Feature engineering con estadísticas             │
│  acumuladas que incluyan el partido actual → calcular stats         │
│  SOLO con partidos ANTERIORES al partido a predecir.                │
│                                                                     │
│  RIESGO 4 (BAJO): Normalización/escalado sobre todo el dataset.     │
│  → Usar Pipeline Scikit-learn: fit solo en train, transform en test.│
└─────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# SECCIÓN 10: CONCLUSIONES DEL EDA
# =============================================================================

def print_eda_conclusions() -> None:
    """Imprime las conclusiones técnicas del EDA."""
    print("\n" + "=" * 68)
    print("  CONCLUSIONES DEL EDA — FASE 1")
    print("=" * 68)
    print("""
  1. DESBALANCE DE CLASES
     La variable objetivo muestra desbalance (local gana con mayor
     frecuencia que empate/visitante). Estrategia: class_weight='balanced'
     en Random Forest o técnicas de oversampling para el empate.

  2. FEATURE ENGINEERING CRÍTICO
     El dataset base NO contiene features predictivas directas suficientes.
     Se deben construir variables históricas agregadas (Fase 2): rendimiento
     pasado de cada equipo ANTES de cada partido.

  3. NULOS EN COLUMNAS RICAS
     Columnas textuales (goleadores, capitanes, entrenadores) tienen
     nulos históricos (antes de 1966, estadísticas incompletas).
     Para el modelado supervisado estas columnas se excluirán; para
     K-Means solo se usarán columnas numéricas básicas.

  4. EXPANSIÓN DEL TORNEO
     Desde 1998 hay 64 partidos por Copa (32 equipos). Los modelos
     deben considerar la variable 'Year' para no mezclar eras con
     estructuras de torneo muy diferentes.

  5. DATOS PARA K-MEANS
     Se puede construir un perfil sólido por selección con las columnas
     disponibles. Se recomienda normalización (StandardScaler) antes de
     K-Means, ya que las magnitudes difieren (partidos vs. tasas).

  6. ESTRATEGIA METODOLÓGICA
     → Fase 2: Feature Engineering + Pipeline Scikit-learn
     → Fase 3: Random Forest con cross-validation
     → Fase 4: K-Means con elbow + silhouette
     → Fase 5: Evaluación integral de ambos modelos
     → Fase 6: Serialización + API FastAPI
""")


# =============================================================================
# FUNCIÓN PRINCIPAL DEL EDA
# =============================================================================

def run_full_eda(df_matches: pd.DataFrame,
                 df_world_cup: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Ejecuta el EDA completo de Fase 1.

    Parameters
    ----------
    df_matches : pd.DataFrame
        Dataset de partidos cargado con data_loader.load_matches().
    df_world_cup : pd.DataFrame, opcional
        Dataset de torneos (world_cup.csv).

    Returns
    -------
    pd.DataFrame
        Dataset de partidos enriquecido con columnas derivadas (result, etc.)
    """
    print("\n" + "=" * 68)
    print("  FASE 1 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("  Proyecto: Predicción Copa Mundial FIFA 1930–2022")
    print("=" * 68)

    # 0. Preparar columnas derivadas
    print("\n[0/9] Preparando columnas derivadas...")
    df = prepare_matches(df_matches)
    print(f"  Columnas en df_matches: {list(df.columns)}")

    # 1–8. Visualizaciones
    plot_result_distribution(df)
    plot_temporal_analysis(df)
    plot_goals_analysis(df)
    plot_home_advantage(df)
    plot_stage_analysis(df)
    team_stats = plot_teams_analysis(df)
    plot_nulls_analysis(df)
    plot_correlations(df)

    # 9. Recomendaciones y conclusiones
    print_feature_recommendations()
    print_eda_conclusions()

    print("\nEDA COMPLETO. Visualizaciones guardadas en: outputs/eda_plots/")
    print("   → Listo para iniciar FASE 2: Preprocesamiento y Feature Engineering\n")

    return df