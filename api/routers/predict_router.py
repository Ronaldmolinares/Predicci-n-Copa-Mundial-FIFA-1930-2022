from fastapi import APIRouter, HTTPException
from api.schemas import MatchRequest, MatchPredictionResponse, MatchFeaturesRaw
from api.ml_service import ml_service

router = APIRouter(prefix="/api/v1/predict", tags=["Predicción de Partidos"])

@router.post("/match", response_model=MatchPredictionResponse)
def predict_match(request: MatchRequest):
    """
    Predice el resultado de un partido basándose en los nombres de los equipos, el año y la fase.
    
    La API calculará dinámicamente las estadísticas históricas de ambos equipos
    antes de inyectarlas en el modelo, asegurando total coherencia.
    """
    try:
        # debug=True imprimirá las 24 features en la consola del servidor Uvicorn
        pred_class, probas = ml_service.predict_match(request, debug=True)
        
        return MatchPredictionResponse(
            predicted_class=pred_class,
            probabilities=probas
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción: {str(e)}")

@router.post("/match/raw", response_model=MatchPredictionResponse)
def predict_match_raw(features: MatchFeaturesRaw):
    """
    (Endpoint Técnico/Debug) Predice el resultado de un partido usando las 24 features ya pre-calculadas.
    
    Útil para:
    - Testing y validación del modelo
    - Simular escenarios hipotéticos sin dependencias del historial
    - Comparar predicciones entre enfoque natural vs raw
    """
    try:
        pred_class, probas = ml_service.predict_match_raw(features)
        
        return MatchPredictionResponse(
            predicted_class=pred_class,
            probabilities=probas
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción raw: {str(e)}")
