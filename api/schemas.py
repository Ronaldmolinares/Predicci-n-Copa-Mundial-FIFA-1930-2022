from pydantic import BaseModel, Field
from typing import Dict

# ---------------------------------------------------------
# Esquemas para Random Forest (Modelo Supervisado)
# ---------------------------------------------------------

class MatchRequest(BaseModel):
    """
    Datos de entrada naturales enviados por el usuario final.
    La API calculará internamente las variables técnicas (win rates, deltas).
    """
    home_team: str = Field(..., example="Argentina", description="Nombre histórico exacto del equipo local")
    away_team: str = Field(..., example="France", description="Nombre histórico exacto del equipo visitante")
    year: int = Field(..., example=2022, description="Año en el que se disputa el partido")
    stage: str = Field(..., example="Final", description="Fase del torneo (ej. Group, Round of 16, Final)")

class MatchPredictionResponse(BaseModel):
    """
    Respuesta final ya procesada y legible.
    """
    predicted_class: str = Field(description="Clase predicha mapeada (ej. home_win, draw, away_win)")
    probabilities: Dict[str, float] = Field(description="Probabilidades exactas por cada escenario")

class MatchFeaturesRaw(BaseModel):
    """
    (Endpoint técnico/Debug) Las 24 features pre-calculadas para predicción de partidos.
    """
    # Home features (6)
    home_hist_win_rate: float = Field(..., example=0.55)
    home_hist_draw_rate: float = Field(..., example=0.25)
    home_hist_avg_gf: float = Field(..., example=1.8)
    home_hist_avg_ga: float = Field(..., example=1.2)
    home_hist_avg_gd: float = Field(..., example=0.6)
    home_hist_matches: int = Field(..., example=50)
    
    # Away features (6)
    away_hist_win_rate: float = Field(..., example=0.45)
    away_hist_draw_rate: float = Field(..., example=0.28)
    away_hist_avg_gf: float = Field(..., example=1.5)
    away_hist_avg_ga: float = Field(..., example=1.4)
    away_hist_avg_gd: float = Field(..., example=0.1)
    away_hist_matches: int = Field(..., example=40)
    
    # Recientes (2)
    home_recent_win_rate: float = Field(..., example=0.60)
    away_recent_win_rate: float = Field(..., example=0.40)
    
    # Deltas (4)
    delta_win_rate: float = Field(..., example=0.10)
    delta_avg_gf: float = Field(..., example=0.3)
    delta_avg_gd: float = Field(..., example=0.5)
    delta_wc_exp: int = Field(..., example=2)
    
    # Apariciones mundialistas (2)
    home_wc_appearances: int = Field(..., example=5)
    away_wc_appearances: int = Field(..., example=3)
    
    # Otros (2)
    year: int = Field(..., example=2022)
    stage_encoded: int = Field(..., example=6, description="Código numérico de la fase (1-6)")

# ---------------------------------------------------------
# Esquemas para K-Means (Modelo No Supervisado)
# ---------------------------------------------------------

class ClusterRequest(BaseModel):
    """
    Datos de entrada naturales enviados por el usuario final para agrupar un equipo.
    La API calculará internamente las estadísticas acumuladas hasta ese año.
    """
    team: str = Field(..., example="Colombia", description="Nombre histórico exacto del equipo")
    year: int = Field(..., example=2022, description="Año hasta el cual se evaluará el rendimiento")

class TeamStats(BaseModel):
    """
    (Endpoint técnico) 10 features crudas.
    """
    total_matches: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_goals_for: float
    avg_goals_against: float
    avg_goal_diff: float
    num_tournaments: int
    best_stage: int
    goals_per_tournament: float

class ClusterResponse(BaseModel):
    cluster_id: int
    cluster_label: str
