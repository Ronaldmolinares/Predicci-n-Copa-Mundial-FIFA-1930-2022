#  Predicción Copa Mundial FIFA 1930–2022
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
│   ├── fase_1_eda.py              ← FASE 1: EDA
│   ├── fase_2_preprocessing.py    ← FASE 2: Preprocesamiento
│   ├── fase_3_supervised.py       ← FASE 3: Random Forest
│   ├── fase_4_unsupervised.py     ← FASE 4: K-Means
│   └── fase_5_evaluation.py       ← FASE 5: Evaluación
│
├── models/                        ← Modelos serializados
│   ├── random_forest_pipeline.pkl ← Random Forest
│   └── metadata.json              ← Metadata
│
├── api/                           ← FastAPI REST service
│   ├── main.py                    ← API REST
│   └── schemas.py                 ← Esquemas
│
├── outputs/
│   └── eda_plots/                 ← Visualizaciones generadas
│
├── requirements.txt
└── README.md
```

---

##  Fases del Proyecto (CRISP-DM)

| Fase | Descripción 
|------|-------------|
| 1 | Comprensión de Datos + EDA 
| 2 | Preparación de Datos + Feature Engineering 
| 3 | Modelado Supervisado (Random Forest) 
| 4 | Modelado No Supervisado (K-Means) 
| 5 | Evaluación Integral 
| 6 | Despliegue (FastAPI) | 

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

## Ejecución de Fases

1. **Fase 1: EDA**
```bash
python notebooks/fase_1_eda.py
```

2. **Fase 2: Preprocesamiento**
```bash
python notebooks/fase_2_preprocessing.py
```

3. **Fase 3: Random Forest**
```bash
python notebooks/fase_3_supervised.py
```

4. **Fase 4: K-Means**
```bash
python notebooks/fase_4_unsupervised.py
```

5. **Fase 5: Evaluación**
```bash
python notebooks/fase_5_evaluation.py
```

---

##  Dataset

- **Fuente:** Kaggle — piterfm/fifa-football-world-cup
- **Archivo principal:** `matches_1930_2022.csv`
- **Cobertura:** 22 ediciones de la Copa Mundial (1930–2022)
- **Metodología:** CRISP-DM
- **Stack:** Python · Pandas · Scikit-learn · FastAPI · Joblib · Pydantic

---

## Interfaz gráfica (Tkinter)

Se agregó una interfaz básica para orquestar la carga, preprocesamiento,
entrenamiento y exportación de artefactos.

Ejecutar:

```bash
python gui/app.py
```

La GUI permite:
- Cargar el CSV de partidos (o usar `data/matches_1930_2022.csv`).
- Ejecutar preprocesamiento (Fase 2) y generar artefactos en `models/` y `outputs/`.
- Entrenar un Random Forest baseline y ejecutar K-Means (guardando resultados).
