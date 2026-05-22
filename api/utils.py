import json
import pandas as pd
from pathlib import Path
import sys

# Agregar src al path para importar stage_cleaner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage_cleaner import clean_stages_dataframe

# Directorio base y subdirectorios
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Archivos de la fase de modelado y dataset
RF_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.pkl"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "preprocessing_metadata.json"
DATASET_PATH = DATA_DIR / "matches_1930_2022.csv"

# Clusters K-Means
CLUSTER_LABELS = {
    0: "Debutante / Sin Victorias",
    1: "Elite Mundial",
    2: "Competidor Intermedio"
}

def load_metadata() -> dict:
    """Carga la configuración y mapeos guardados durante la Fase 2."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_historical_data() -> pd.DataFrame:
    """
    Carga el dataset maestro de partidos que actuará como base de conocimiento.
    Aplica limpieza y unificación de fases (12 → 6 estándar).
    """
    df = pd.read_csv(DATASET_PATH)
    
    # Aplicar limpieza de fases si la columna 'Round' existe
    if 'Round' in df.columns and 'Stage_Clean' not in df.columns:
        df = clean_stages_dataframe(df, column_name='Round')
    
    return df
