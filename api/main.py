from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.routers import predict_router, cluster_router
from api.ml_service import ml_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Se ejecuta una sola vez al iniciar el servidor.
    Aquí cargamos los modelos en memoria para que estén listos para las peticiones.
    """
    print("Iniciando FastAPI - Cargando modelos de ML...")
    ml_service.load_models()
    yield
    # Limpieza al apagar la API
    print("Apagando FastAPI - Liberando recursos...")
    ml_service.rf_model = None
    ml_service.kmeans_model = None

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="FIFA World Cup Prediction API",
    description="API robusta para servir modelos de Machine Learning que predicen resultados de partidos y agrupan equipos históricos.",
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS (Cross-Origin Resource Sharing)
# Permite que un Frontend (React, Vue, Angular) alojado en otro dominio o puerto consuma la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, se recomienda restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

# Incluir las rutas modulares
app.include_router(predict_router.router)
app.include_router(cluster_router.router)

@app.get("/", tags=["General"])
def root():
    """
    Endpoint base para comprobar conectividad básica.
    """
    return {"message": "Bienvenido a la API de Predicción del Mundial de la FIFA (Fase 5). Ve a /docs para interactuar."}

@app.get("/health", tags=["General"])
def health_check():
    """
    Endpoint de comprobación de salud de la API.
    Verifica no solo que la API responda, sino que los modelos ML estén cargados.
    """
    if ml_service.is_ready:
        return {"status": "ok", "models_loaded": True}
    else:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: La API está arriba pero los modelos no pudieron ser cargados."
        )
