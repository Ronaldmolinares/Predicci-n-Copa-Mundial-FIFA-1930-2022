import joblib
import pandas as pd
from api.utils import RF_MODEL_PATH, KMEANS_MODEL_PATH, CLUSTER_LABELS, load_metadata, load_historical_data
from api.schemas import MatchRequest, TeamStats, ClusterRequest
from api.feature_builder import FeatureBuilder

class MLService:
    def __init__(self):
        self.rf_model = None
        self.kmeans_model = None
        self.metadata = None
        self.feature_builder = None
        self.is_ready = False

    def load_models(self):
        try:
            self.rf_model = joblib.load(RF_MODEL_PATH)
            self.kmeans_model = joblib.load(KMEANS_MODEL_PATH)
            self.metadata = load_metadata()
            
            # Cargar dataset histórico y configurar el FeatureBuilder
            hist_df = load_historical_data()
            self.feature_builder = FeatureBuilder(hist_df, self.metadata)
            
            self.is_ready = True
            print("MLService: Modelos, Metadatos y FeatureBuilder cargados exitosamente.")
        except Exception as e:
            print(f"Error inicializando MLService: {e}")
            self.is_ready = False

    def predict_match(self, request: MatchRequest, debug: bool = False):
        if not self.is_ready:
            raise RuntimeError("Los modelos no están listos.")
            
        # 1. Feature Engineering
        features_dict = self.feature_builder.build_match_features(
            home_team=request.home_team,
            away_team=request.away_team,
            year=request.year,
            stage=request.stage,
            debug=debug
        )
        
        # 2. Convertir a DataFrame de Pandas (formato esperado por el Pipeline)
        df = pd.DataFrame([features_dict])
        
        # 3. Predicción raw
        pred_raw = self.rf_model.predict(df)[0]
        proba_raw = self.rf_model.predict_proba(df)[0]
        
        # 4. Mapeo de salida usando target_mapping de metadatos
        target_mapping = self.metadata.get("target_mapping", {})
        predicted_class_name = target_mapping.get(str(pred_raw), str(pred_raw))
        
        # Mapeo de probabilidades
        classes_raw = self.rf_model.classes_
        proba_dict = {}
        for c, p in zip(classes_raw, proba_raw):
            class_name = target_mapping.get(str(c), str(c))
            proba_dict[class_name] = float(p)
            
        return predicted_class_name, proba_dict

    def predict_cluster(self, request: ClusterRequest, debug: bool = False):
        if not self.is_ready:
            raise RuntimeError("Los modelos no están listos.")
        
        # Feature Engineering para KMeans
        features_dict = self.feature_builder.build_cluster_features(
            team=request.team,
            year=request.year,
            debug=debug
        )
        
        df = pd.DataFrame([features_dict])
        cluster_id = int(self.kmeans_model.predict(df)[0])
        label = CLUSTER_LABELS.get(cluster_id, "Desconocido")
        return cluster_id, label

    def predict_cluster_raw(self, stats: TeamStats):
        """Endpoint técnico directo."""
        if not self.is_ready:
            raise RuntimeError("Los modelos no están listos.")
        
        df = pd.DataFrame([stats.model_dump()])
        cluster_id = int(self.kmeans_model.predict(df)[0])
        label = CLUSTER_LABELS.get(cluster_id, "Desconocido")
        return cluster_id, label

ml_service = MLService()
