from fastapi import APIRouter, HTTPException
from api.schemas import ClusterRequest, TeamStats, ClusterResponse
from api.ml_service import ml_service

router = APIRouter(prefix="/api/v1/cluster", tags=["Clustering de Equipos"])

@router.post("/team", response_model=ClusterResponse)
def predict_cluster(request: ClusterRequest):
    """
    Recibe el nombre de un equipo y un año.
    Calcula dinámicamente sus 10 estadísticas históricas acumuladas
    hasta ese momento y lo asigna a un clúster usando KMeans.
    """
    try:
        cluster_id, label = ml_service.predict_cluster(request, debug=True)
        return ClusterResponse(
            cluster_id=cluster_id,
            cluster_label=label
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante el clustering: {str(e)}")

@router.post("/team/raw", response_model=ClusterResponse)
def predict_cluster_raw(stats: TeamStats):
    """
    (Endpoint Técnico/Debug) Recibe las 10 features ya pre-calculadas manualmente.
    """
    try:
        cluster_id, label = ml_service.predict_cluster_raw(stats)
        return ClusterResponse(
            cluster_id=cluster_id,
            cluster_label=label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante el clustering raw: {str(e)}")
