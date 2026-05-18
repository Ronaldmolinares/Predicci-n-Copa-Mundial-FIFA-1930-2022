# Script ejecutable: FASE 1 - Análisis Exploratorio de Datos

"""
==============================================================================
FASE 1 — NOTEBOOK EJECUTABLE
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  notebooks/fase_1_eda.py

Instrucciones de uso:
──────────────────────────────────────────────────────────────────────────────
1. Descarga el dataset de Kaggle:
   https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup

2. Coloca los archivos CSV en la carpeta  data/  del proyecto:
      data/matches_1930_2022.csv
      data/fifa_ranking_2022-10-06.csv
      data/world_cup.csv

3. Instala las dependencias:
      pip install pandas numpy matplotlib seaborn scikit-learn joblib fastapi pydantic

4. Ejecuta desde la raíz del proyecto:
      python notebooks/fase_1_eda.py

5. Las visualizaciones se guardarán en:
      outputs/eda_plots/
──────────────────────────────────────────────────────────────────────────────

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado
Universidad: UPTC — Ingeniería de Sistemas y Computación
==============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
import sys
import os

# Asegura que Python encuentre el módulo src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_all, report_initial_inspection
from src.eda import run_full_eda

# ---------------------------------------------------------------------------
# 1. CARGA DEL DATASET
# ---------------------------------------------------------------------------

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         FASE 1: COMPRENSIÓN DE DATOS Y EDA                           ║
║         Copa Mundial FIFA 1930–2022                                  ║
║         Electiva II — Machine Learning Aplicado — UPTC               ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# load_all() verifica existencia de archivos y emite errores claros
# si el dataset no ha sido descargado desde Kaggle
datasets = load_all()

df_matches   = datasets["matches"]
df_ranking   = datasets["ranking"]
df_world_cup = datasets["world_cup"]

# ---------------------------------------------------------------------------
# 2. INSPECCIÓN INICIAL DE CADA ARCHIVO
# ---------------------------------------------------------------------------

print("\n" + "═" * 68)
print("  BLOQUE 2: INSPECCIÓN INICIAL DE LOS TRES ARCHIVOS")
print("═" * 68)

report_initial_inspection(df_matches,   "matches_1930_2022")
report_initial_inspection(df_ranking,   "fifa_ranking_2022-10-06")
report_initial_inspection(df_world_cup, "world_cup")

# ---------------------------------------------------------------------------
# 3. INSPECCIÓN EXTRA: COLUMNAS REALES DEL DATASET
# ---------------------------------------------------------------------------

print("\n" + "═" * 68)
print("  BLOQUE 3: INVENTARIO COMPLETO DE COLUMNAS")
print("═" * 68)

print("\n Columnas en matches_1930_2022.csv:")
for i, col in enumerate(df_matches.columns, 1):
    dtype  = str(df_matches[col].dtype)
    n_null = df_matches[col].isna().sum()
    pct    = n_null / len(df_matches) * 100
    print(f"   {i:>2}. {col:<35} dtype={dtype:<12} nulos={n_null} ({pct:.1f}%)")

print("\n Columnas en world_cup.csv:")
for col in df_world_cup.columns:
    print(f"   • {col}")

print("\n Columnas en fifa_ranking_2022-10-06.csv:")
for col in df_ranking.columns:
    print(f"   • {col}")

# ---------------------------------------------------------------------------
# 4. VALORES ÚNICOS — VARIABLES CATEGÓRICAS CLAVE
# ---------------------------------------------------------------------------

print("\n" + "═" * 68)
print("  BLOQUE 4: VALORES ÚNICOS EN COLUMNAS CATEGÓRICAS")
print("═" * 68)

cat_cols_to_check = ["home_team", "away_team", "stage", "round",
                     "phase", "match_phase"]
for col in cat_cols_to_check:
    if col in df_matches.columns:
        n_unique = df_matches[col].nunique()
        examples = df_matches[col].dropna().unique()[:8]
        print(f"\n   {col}: {n_unique} valores únicos")
        print(f"     Ejemplos: {list(examples)}")

# Equipos únicos
if "home_team" in df_matches.columns:
    teams = set(df_matches["home_team"].dropna()) | set(df_matches["away_team"].dropna())
    print(f"\n   Total de selecciones únicas en el dataset: {len(teams)}")

# Torneos (años)
if "year" in df_matches.columns:
    years = sorted(df_matches["year"].dropna().unique())
    print(f"\n   Torneos disponibles ({len(years)} ediciones): {[int(y) for y in years]}")

# ---------------------------------------------------------------------------
# 5. ANÁLISIS EXPLORATORIO COMPLETO + VISUALIZACIONES
# ---------------------------------------------------------------------------

print("\n" + "═" * 68)
print("  BLOQUE 5: ANÁLISIS EXPLORATORIO COMPLETO (EDA)")
print("═" * 68)

df_enriched = run_full_eda(df_matches, df_world_cup)

# ---------------------------------------------------------------------------
# 6. GUARDAR DATASET ENRIQUECIDO PARA FASE 2
# ---------------------------------------------------------------------------

output_path = "outputs/matches_enriched_phase1.csv"
df_enriched.to_csv(output_path, index=False)
print(f"\n Dataset enriquecido guardado en: {output_path}")
print("   (Contiene columnas derivadas: result, total_goals, goal_diff, went_to_pen)")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FASE 1 COMPLETADA                                                   ║
║                                                                      ║
║  Hallazgos clave:                                                    ║
║  • Variable objetivo con desbalance de clases (empate minoritario)   ║
║  • Features post-partido identificadas (xg, score) — NO usar en RF   ║
║  • Feature engineering necesario: estadísticas históricas por equipo ║
║  • Dataset de partidos: fuente de verdad única (no modificada)       ║
║                                                                      ║
║  → Próximo paso: FASE 2 — Preprocesamiento y Feature Engineering     ║
╚══════════════════════════════════════════════════════════════════════╝
""")