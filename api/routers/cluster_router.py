from fastapi import APIRouter, HTTPException
from api.schemas import TeamStats, ClusterResponse
from api.ml_service import ml_service

router = APIRouter(prefix="/api/v1/cluster", tags=["Clustering de Equipos"])

@router.post("/team", response_model=ClusterResponse)
def predict_cluster(stats: TeamStats):
    """
    Recibe las estadísticas históricas agregadas de un equipo y,
    utilizando el modelo KMeans, determina a qué clúster pertenece.
    
    Devuelve el ID numérico del clúster y su etiqueta descriptiva.
    """
    try:
        cluster_id, label = ml_service.predict_cluster(stats)
        return ClusterResponse(
            cluster_id=cluster_id,
            cluster_label=label
        )
    except RuntimeError as re:
        # Error si los modelos no están cargados
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        # Cualquier otro error
        raise HTTPException(status_code=500, detail=f"Error interno durante el clustering: {str(e)}")
