# ⚽ Predicción Copa Mundial FIFA 1930–2022
### Machine Learning Aplicado — ELECTIVA II
**Universidad Pedagógica y Tecnológica de Colombia (UPTC)**
Facultad de Ingeniería — Ingeniería de Sistemas y Computación

---

##  Autores
- Ronald Samir Molinares Sanabria
- María Fernanda Sogamoso González

**Docente:** Viviana Alexandra Villanueva Cipagauta

---

##  Estructura del Proyecto

```
fifa_ml_project/
│
├── data/                          ← Dataset original Kaggle (NO modificar)
│   ├── matches_1930_2022.csv      ← Archivo principal (partidos)
│   ├── fifa_ranking_2022-10-06.csv
│   └── world_cup.csv
│
├── src/                           ← Módulos Python del sistema
│   ├── data_loader.py             ← Carga y validación del dataset
│   ├── eda.py                     ← Análisis exploratorio completo
│   ├── preprocessing.py           ← Limpieza + Feature Engineering [FASE 2]
│   ├── model_supervised.py        ← Random Forest [FASE 3]
│   ├── model_unsupervised.py      ← K-Means [FASE 4]
│   └── evaluate.py                ← Métricas y evaluación [FASE 5]
│
├── notebooks/                     ← Scripts ejecutables por fase
│   ├── fase_1_eda.py              ← ✅ FASE 1 (actual)
│   ├── fase_2_preprocessing.py    ← [pendiente]
│   ├── fase_3_supervised.py       ← [pendiente]
│   ├── fase_4_unsupervised.py     ← [pendiente]
│   └── fase_5_evaluation.py       ← [pendiente]
│
├── models/                        ← Modelos serializados
│   ├── random_forest_pipeline.pkl ← [Fase 3]
│   └── metadata.json              ← [Fase 3]
│
├── api/                           ← FastAPI REST service
│   ├── main.py                    ← [Fase 6]
│   └── schemas.py                 ← [Fase 6]
│
├── outputs/
│   └── eda_plots/                 ← Visualizaciones generadas
│
├── requirements.txt
└── README.md
```

---

##  Fases del Proyecto (CRISP-DM)

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Comprensión de Datos + EDA |  Completada |
| 2 | Preparación de Datos + Feature Engineering |  Pendiente |
| 3 | Modelado Supervisado (Random Forest) |  Pendiente |
| 4 | Modelado No Supervisado (K-Means) |  Pendiente |
| 5 | Evaluación Integral |  Pendiente |
| 6 | Despliegue (FastAPI) |  Pendiente |

---

##  Instalación

```bash
pip install -r requirements.txt
```

---

## ▶ Ejecución Fase 1

```bash
# 1. Descarga el dataset:
#    https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup
#    → coloca los CSV en data/

# 2. Ejecuta:
python notebooks/fase_1_eda.py
```

---

##  Dataset

- **Fuente:** Kaggle — piterfm/fifa-football-world-cup
- **Archivo principal:** `matches_1930_2022.csv`
- **Cobertura:** 22 ediciones de la Copa Mundial (1930–2022)
- **Metodología:** CRISP-DM
- **Stack:** Python · Pandas · Scikit-learn · FastAPI · Joblib · Pydantic