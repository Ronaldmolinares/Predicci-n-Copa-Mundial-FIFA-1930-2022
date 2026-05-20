import pandas as pd
from typing import Dict, Any

class FeatureBuilder:
    """
    Motor inteligente de la API que calcula variables históricas en tiempo real.
    Asegura cero data leakage filtrando el historial ANTES de predecir.
    Mantiene independencia de selecciones históricas.
    """
    def __init__(self, historical_df: pd.DataFrame, metadata: Dict[str, Any]):
        self.df = historical_df
        self.metadata = metadata
        self.recent_window = metadata.get("recent_window", 5)
        self.feature_names_rf = metadata["feature_names_rf"]
        self.stage_encoding = metadata["stage_encoding"]
        
        self._prepare_log()

    def _prepare_log(self):
        """
        Transforma el dataset histórico en una tabla larga (log) de 
        apariciones por equipo para facilitar el cálculo rápido.
        """
        # Detectar columna de fase
        stage_col = next((c for c in self.df.columns if c.lower() in ["stage", "round", "phase", "match_phase"]), None)
        if stage_col:
            stage_series = self.df[stage_col].str.lower().str.strip().map(self.stage_encoding).fillna(0).astype(int)
        else:
            stage_series = pd.Series(0, index=self.df.index)

        # Home log
        home = self.df[["Year", "home_team", "home_score", "away_score"]].copy()
        home.columns = ["Year", "team", "gf", "ga"]
        home["is_win"] = (home["gf"] > home["ga"]).astype(float)
        home["is_draw"] = (home["gf"] == home["ga"]).astype(float)
        home["is_loss"] = (home["gf"] < home["ga"]).astype(float)
        home["stage_encoded"] = stage_series

        # Away log
        away = self.df[["Year", "away_team", "away_score", "home_score"]].copy()
        away.columns = ["Year", "team", "gf", "ga"]
        away["is_win"] = (away["gf"] > away["ga"]).astype(float)
        away["is_draw"] = (away["gf"] == away["ga"]).astype(float)
        away["is_loss"] = (away["gf"] < away["ga"]).astype(float)
        away["stage_encoded"] = stage_series

        self.log = pd.concat([home, away], ignore_index=True)
        self.log["gd"] = self.log["gf"] - self.log["ga"]

    def _get_team_stats(self, team: str, target_year: int) -> dict:
        """
        Calcula las estadísticas del equipo usando SOLO años anteriores a target_year.
        Si es debutante, devuelve una estructura neutra (ceros).
        """
        # Filtrar el pasado estrictamente
        past = self.log[(self.log["team"] == team) & (self.log["Year"] < target_year)]
        
        stats = {
            "hist_matches": 0,
            "hist_win_rate": 0.0,
            "hist_draw_rate": 0.0,
            "hist_avg_gf": 0.0,
            "hist_avg_ga": 0.0,
            "hist_avg_gd": 0.0,
            "recent_win_rate": 0.0,
            "wc_appearances": 0
        }

        # Fallback para debutantes sin historial en mundiales
        if len(past) == 0:
            return stats

        # Estadísticas históricas acumuladas
        stats["hist_matches"] = len(past)
        stats["hist_win_rate"] = past["is_win"].mean()
        stats["hist_draw_rate"] = past["is_draw"].mean()
        stats["hist_avg_gf"] = past["gf"].mean()
        stats["hist_avg_ga"] = past["ga"].mean()
        stats["hist_avg_gd"] = past["gd"].mean()
        
        # Conteo de torneos distintos disputados
        stats["wc_appearances"] = past["Year"].nunique()

        # Forma reciente (Ventana móvil)
        recent = past.sort_values("Year").tail(self.recent_window)
        stats["recent_win_rate"] = recent["is_win"].mean()

        return stats

    def build_match_features(self, home_team: str, away_team: str, year: int, stage: str, debug: bool = False) -> Dict[str, Any]:
        """
        Genera el diccionario con las 24 features exactas requeridas por RF.
        """
        home_stats = self._get_team_stats(home_team, year)
        away_stats = self._get_team_stats(away_team, year)

        # Mapear la fase a su encoding de Fase 2
        stage_norm = str(stage).lower().strip()
        stage_encoded = self.stage_encoding.get(stage_norm, 0)

        # Derivar Deltas
        delta_win_rate = home_stats["hist_win_rate"] - away_stats["hist_win_rate"]
        delta_avg_gf = home_stats["hist_avg_gf"] - away_stats["hist_avg_gf"]
        delta_avg_gd = home_stats["hist_avg_gd"] - away_stats["hist_avg_gd"]
        delta_wc_exp = home_stats["wc_appearances"] - away_stats["wc_appearances"]

        # Vector inicial de características
        features = {
            "home_hist_win_rate": home_stats["hist_win_rate"],
            "home_hist_draw_rate": home_stats["hist_draw_rate"],
            "home_hist_avg_gf": home_stats["hist_avg_gf"],
            "home_hist_avg_ga": home_stats["hist_avg_ga"],
            "home_hist_avg_gd": home_stats["hist_avg_gd"],
            "home_hist_matches": home_stats["hist_matches"],
            
            "away_hist_win_rate": away_stats["hist_win_rate"],
            "away_hist_draw_rate": away_stats["hist_draw_rate"],
            "away_hist_avg_gf": away_stats["hist_avg_gf"],
            "away_hist_avg_ga": away_stats["hist_avg_ga"],
            "away_hist_avg_gd": away_stats["hist_avg_gd"],
            "away_hist_matches": away_stats["hist_matches"],
            
            "home_recent_win_rate": home_stats["recent_win_rate"],
            "away_recent_win_rate": away_stats["recent_win_rate"],
            
            "delta_win_rate": delta_win_rate,
            "delta_avg_gf": delta_avg_gf,
            "delta_avg_gd": delta_avg_gd,
            "delta_wc_exp": delta_wc_exp,
            
            "home_wc_appearances": home_stats["wc_appearances"],
            "away_wc_appearances": away_stats["wc_appearances"],
            
            "Year": year,
            "stage_encoded": stage_encoded,
            "home_team": home_team,
            "away_team": away_team
        }

        # Validación estructural y ordenamiento basado en metadatos
        if set(features.keys()) != set(self.feature_names_rf):
            missing = set(self.feature_names_rf) - set(features.keys())
            extra = set(features.keys()) - set(self.feature_names_rf)
            raise ValueError(f"Fallo estructural al construir features. Faltan: {missing}. Sobran: {extra}")

        # Garantizar que el diccionario retorne exactamente en el orden del RF Pipeline
        ordered_features = {k: features[k] for k in self.feature_names_rf}

        if debug:
            print("\n" + "="*50)
            print(f" DEBUG: Features generadas RF ({home_team} vs {away_team} - {year})")
            print("="*50)
            for k, v in ordered_features.items():
                print(f"{k}: {v}")
            print("="*50 + "\n")

        return ordered_features

    def build_cluster_features(self, team: str, year: int, debug: bool = False) -> Dict[str, Any]:
        """
        Genera el diccionario con las 10 features exactas requeridas por KMeans.
        """
        kmeans_features_list = self.metadata.get("kmeans_features", [
            "total_matches", "win_rate", "draw_rate", "loss_rate", 
            "avg_goals_for", "avg_goals_against", "avg_goal_diff", 
            "num_tournaments", "best_stage", "goals_per_tournament"
        ])

        past = self.log[(self.log["team"] == team) & (self.log["Year"] < year)]
        
        # Fallback neutro para debutantes
        if len(past) == 0:
            features = {k: 0.0 for k in kmeans_features_list}
            features["total_matches"] = 0
            features["num_tournaments"] = 0
            features["best_stage"] = 0
            return features

        total_matches = len(past)
        total_gf = past["gf"].sum()
        num_tournaments = past["Year"].nunique()

        features = {
            "total_matches": total_matches,
            "win_rate": past["is_win"].mean(),
            "draw_rate": past["is_draw"].mean(),
            "loss_rate": past["is_loss"].mean(),
            "avg_goals_for": past["gf"].mean(),
            "avg_goals_against": past["ga"].mean(),
            "avg_goal_diff": past["gd"].mean(),
            "num_tournaments": num_tournaments,
            "best_stage": past["stage_encoded"].max(),
            "goals_per_tournament": total_gf / num_tournaments if num_tournaments > 0 else 0.0
        }

        # Validar estructura y ordenar
        if set(features.keys()) != set(kmeans_features_list):
            missing = set(kmeans_features_list) - set(features.keys())
            extra = set(features.keys()) - set(kmeans_features_list)
            raise ValueError(f"Fallo estructural en clustering. Faltan: {missing}. Sobran: {extra}")

        ordered_features = {k: features[k] for k in kmeans_features_list}

        if debug:
            print("\n" + "="*50)
            print(f" DEBUG: Features generadas KMeans ({team} - {year})")
            print("="*50)
            for k, v in ordered_features.items():
                print(f"{k}: {v}")
            print("="*50 + "\n")

        return ordered_features
