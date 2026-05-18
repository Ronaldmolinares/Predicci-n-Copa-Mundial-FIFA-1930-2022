from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

api = KaggleApi()
api.authenticate()

# Obtiene la ruta absoluta de la carpeta "data" (donde está este script)
data_dir = Path(__file__).parent

# Descarga el dataset exactamente en esa carpeta
api.dataset_download_files('piterfm/fifa-football-world-cup', path=data_dir, unzip=True)
