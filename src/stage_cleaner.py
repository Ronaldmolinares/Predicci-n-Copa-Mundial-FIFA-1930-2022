"""
Módulo de Limpieza y Unificación de Fases del Dataset
======================================================

Este módulo proporciona funciones para estandarizar la columna 'Round' 
(etapas del torneo) en el dataset de la Copa del Mundo FIFA.

Mapeo de Fases:
- Group Stage ← First group stage, Second group stage, Group stage play-off, Group stage
- Round of 16 ← First round, Second round, Round of 16
- Quarter-finals ← Quarter-finals (sin cambios)
- Semi-finals ← Semi-finals (sin cambios)
- Third-place match ← Third-place match (sin cambios)
- Final ← Final stage, Final

Ventaja: Preserva datos originales en columna 'Round' para auditoría.
"""

import pandas as pd
from typing import Dict, List


# Diccionario de mapeo de fases
STAGE_MAPPING: Dict[str, str] = {
    # Group Stage
    "First group stage": "Group Stage",
    "Second group stage": "Group Stage",
    "Group stage play-off": "Group Stage",
    "Group stage": "Group Stage",
    
    # Round of 16
    "First round": "Round of 16",
    "Second round": "Round of 16",
    "Round of 16": "Round of 16",
    
    # Fases que se mantienen igual
    "Quarter-finals": "Quarter-finals",
    "Semi-finals": "Semi-finals",
    "Third-place match": "Third-place match",
    
    # Final
    "Final stage": "Final",
    "Final": "Final",
}

# Orden cronológico de las fases
STAGES_ORDERED: List[str] = [
    "Group Stage",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Third-place match",
    "Final",
]


def clean_stage(stage_value: str) -> str:
    """
    Limpia y unifica una etiqueta de fase individual.
    
    Args:
        stage_value (str): Valor original de la fase
        
    Returns:
        str: Fase unificada. Retorna el valor original si no está en el mapeo.
    """
    if pd.isna(stage_value):
        return stage_value
    
    stage_str = str(stage_value).strip()
    return STAGE_MAPPING.get(stage_str, stage_str)


def clean_stages_dataframe(df: pd.DataFrame, column_name: str = "Round") -> pd.DataFrame:
    """
    Aplica la limpieza de fases a un DataFrame completo.
    Crea una nueva columna 'Stage_Clean' preservando los datos originales.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de partidos
        column_name (str): Nombre de la columna original con las fases (default: "Round")
        
    Returns:
        pd.DataFrame: DataFrame con nueva columna 'Stage_Clean'
        
    Example:
        >>> df = pd.read_csv('matches_1930_2022.csv')
        >>> df_clean = clean_stages_dataframe(df)
        >>> print(df_clean['Stage_Clean'].unique())
        ['Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Third-place match', 'Final']
    """
    df = df.copy()
    
    if column_name not in df.columns:
        raise ValueError(f"Columna '{column_name}' no encontrada en el DataFrame")
    
    # Crear columna limpia aplicando la función de mapeo
    df['Stage_Clean'] = df[column_name].apply(clean_stage)
    
    return df


def get_unique_stages() -> List[str]:
    """
    Retorna la lista de fases unificadas en orden cronológico.
    
    Returns:
        List[str]: Lista de 6 fases estándar ordenadas
        
    Example:
        >>> stages = get_unique_stages()
        >>> print(stages)
        ['Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Third-place match', 'Final']
    """
    return STAGES_ORDERED.copy()


def get_stage_mapping() -> Dict[str, str]:
    """
    Retorna el diccionario completo de mapeo de fases.
    
    Returns:
        Dict[str, str]: Diccionario {original: unificada}
    """
    return STAGE_MAPPING.copy()


if __name__ == "__main__":
    # Ejemplo de uso
    print("=" * 70)
    print("MÓDULO DE LIMPIEZA DE FASES - COPA DEL MUNDO FIFA")
    print("=" * 70)
    
    print("\n📋 Fases Unificadas (Orden Cronológico):")
    for i, stage in enumerate(get_unique_stages(), 1):
        print(f"  {i}. {stage}")
    
    print("\n📊 Diccionario de Mapeo:")
    for original, cleaned in sorted(get_stage_mapping().items()):
        if original != cleaned:
            print(f"  {original:30} → {cleaned}")
    
    print("\n✅ Módulo cargado exitosamente")
