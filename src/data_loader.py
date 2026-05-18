"""
==============================================================================
FASE 1 — MÓDULO DE CARGA DE DATOS
Proyecto: Predicción Copa Mundial FIFA 1930–2022
Archivo:  src/data_loader.py

Descripción:
    Carga controlada y validada de los tres archivos CSV del dataset
    piterfm/fifa-football-world-cup de Kaggle.

Buenas prácticas aplicadas:
    - Verificación de existencia de archivos antes de cargar.
    - Tipos de datos explícitos donde es posible.
    - Registro de inconsistencias detectadas durante la carga.
    - Separación clara entre datos crudos y datos de trabajo.
    - Documentación de columnas esperadas por archivo.

Autor: Ronald Samir Molinares Sanabria / María Fernanda Sogamoso González
Asignatura: Electiva II — Machine Learning Aplicado
Universidad: UPTC
==============================================================================
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# AJUSTA ESTA RUTA al directorio donde descargaste el dataset de Kaggle
DATA_DIR = "data/"

# Nombres de archivo confirmados del dataset piterfm (Kaggle)
FILES = {
    "matches":   "matches_1930_2022.csv",
    "ranking":   "fifa_ranking_2022-10-06.csv",
    "world_cup": "world_cup.csv",
}

# Columnas mínimas esperadas en matches_1930_2022.csv
# (Se validan después de cargar; cualquier extra se reporta como bonus)
EXPECTED_MATCH_COLS = [
    "home_team", "away_team",
    "home_score", "away_score",
]

# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

def _build_path(filename: str) -> str:
    """Construye la ruta absoluta del archivo y valida su existencia."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n Archivo no encontrado: {path}"
            f"\n   Asegúrate de haber descargado el dataset desde:"
            f"\n   https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup"
            f"\n   y colocado los CSV en la carpeta: {DATA_DIR}"
        )
    return path


def load_matches() -> pd.DataFrame:
    """
    Carga matches_1930_2022.csv — archivo principal del dataset.

    Cada fila = un partido disputado en alguna Copa Mundial FIFA (1930–2022).

    Returns
    -------
    pd.DataFrame
        DataFrame con todos los partidos históricos.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en DATA_DIR.
    ValueError
        Si faltan columnas mínimas esperadas.
    """
    path = _build_path(FILES["matches"])
    print(f"📂 Cargando: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    # Validar columnas mínimas
    missing = [c for c in EXPECTED_MATCH_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f" Columnas faltantes en matches: {missing}\n"
            f"   Columnas encontradas: {list(df.columns)}"
        )

    print(f" matches cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def load_ranking() -> pd.DataFrame:
    """
    Carga fifa_ranking_2022-10-06.csv — ranking FIFA a octubre 2022.

    Returns
    -------
    pd.DataFrame
        Ranking FIFA de selecciones.
    """
    path = _build_path(FILES["ranking"])
    print(f" Cargando: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f" ranking cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def load_world_cup() -> pd.DataFrame:
    """
    Carga world_cup.csv — resumen por edición del torneo.

    Returns
    -------
    pd.DataFrame
        Una fila por Copa Mundial (22 ediciones, 1930–2022).
    """
    path = _build_path(FILES["world_cup"])
    print(f" Cargando: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f" world_cup cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def load_all() -> dict:
    """
    Carga los tres archivos del dataset y los devuelve en un diccionario.

    Returns
    -------
    dict
        {
            "matches":   pd.DataFrame,  # partidos 1930-2022
            "ranking":   pd.DataFrame,  # ranking FIFA oct-2022
            "world_cup": pd.DataFrame,  # resumen por torneo
        }
    """
    print("=" * 60)
    print("  CARGA DEL DATASET FIFA WORLD CUP 1930–2022")
    print("=" * 60)

    data = {
        "matches":   load_matches(),
        "ranking":   load_ranking(),
        "world_cup": load_world_cup(),
    }

    print("\n Dataset completo cargado correctamente.")
    print("=" * 60)
    return data


# =============================================================================
# UTILIDADES DE INSPECCIÓN INICIAL
# =============================================================================

def report_initial_inspection(df: pd.DataFrame, name: str) -> None:
    """
    Imprime un reporte de inspección inicial profesional de un DataFrame.

    Muestra: dimensiones, tipos, nulos, duplicados y primeras filas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a inspeccionar.
    name : str
        Nombre descriptivo del DataFrame (para el reporte).
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  INSPECCIÓN: {name.upper()}")
    print(sep)

    # 1. Dimensiones
    print(f"\n Dimensiones : {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # 2. Tipos de datos
    print(f"\n Tipos de datos:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {str(dtype):<12} → {count} columnas")

    # 3. Valores nulos
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    null_df     = pd.DataFrame({
        "nulos":      null_counts,
        "porcentaje": null_pct
    }).query("nulos > 0").sort_values("porcentaje", ascending=False)

    if null_df.empty:
        print(f"\n Valores nulos: Ninguno")
    else:
        print(f"\n  Valores nulos ({len(null_df)} columnas afectadas):")
        print(null_df.to_string())

    # 4. Duplicados
    n_dup = df.duplicated().sum()
    status = "✅" if n_dup == 0 else "⚠️"
    print(f"\n{status} Filas duplicadas: {n_dup:,}")

    # 5. Primeras filas
    print(f"\n Primeras 3 filas:")
    print(df.head(3).to_string())

    # 6. Estadísticas descriptivas (solo numéricas)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n Estadísticas descriptivas (columnas numéricas):")
        print(df[num_cols].describe().round(3).to_string())

    print(f"\n{sep}\n")


# =============================================================================
# PUNTO DE ENTRADA PARA PRUEBA DIRECTA
# =============================================================================

if __name__ == "__main__":
    datasets = load_all()

    for name, df in datasets.items():
        report_initial_inspection(df, name)