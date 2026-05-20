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

# ---------------------------------------------------------
# Esquemas para K-Means (Modelo No Supervisado)
# ---------------------------------------------------------

class TeamStats(BaseModel):
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
