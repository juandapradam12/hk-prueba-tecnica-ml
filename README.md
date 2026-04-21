# H&K Prueba Tecnica — ML Engineer
### Clasificacion de Churn y Priorizacion Comercial

**Autor:** Juan Prada · **Fecha:** Abril 2026

---

## Contexto

Prueba tecnica para el rol de ML Engineer en H&K. El caso implementado es prediccion de churn (clasificacion binaria) sobre un dataset de clientes de telecomunicaciones. El objetivo es generar un scoring de riesgo por cliente que permita priorizar las visitas de la fuerza comercial en campo.

---

## Estructura del proyecto

```
hk-prueba-tecnica-ml/
├── docs/
│   └── Prueba_Tecnica_ML_Engineer.pdf  # Enunciado original de la prueba
├── data/
│   └── telco_churn.csv             # Dataset Telco Customer Churn (Kaggle)
├── src/
│   ├── data/
│   │   └── loader.py               # Carga y validacion de datos
│   ├── features/
│   │   └── engineering.py          # Preprocesamiento y feature engineering
│   ├── models/
│   │   └── train.py                # Entrenamiento, evaluacion y serializacion
│   └── visualization/
│       └── plots.py                # Visualizaciones reutilizables
├── notebooks/
│   └── churn_analysis.ipynb        # Notebook narrativo con explicaciones
├── output/
│   ├── models/                     # Modelos serializados (.pkl)
│   ├── figures/                    # Graficas (.png)
│   └── reports/                    # Metricas y scoring (.csv)
├── main.py                         # Pipeline ejecutable end-to-end
├── requirements.txt
└── README.md
```

---

## Como ejecutar

### 1. Crear entorno virtual e instalar dependencias

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo

```bash
python main.py
```

Genera en `output/`:
- `models/` — modelos serializados (LogisticRegression, RandomForest, XGBoost)
- `figures/` — graficas de EDA, curvas ROC, SHAP, scoring
- `reports/model_comparison.csv` — tabla comparativa de metricas
- `reports/churn_scoring.csv` — ranking de clientes por riesgo de churn

### 3. Ver el analisis narrativo

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

Contiene el analisis completo con explicaciones de cada decision tecnica.

---

## Modelos comparados

| Modelo | Descripcion |
|--------|-------------|
| Logistic Regression | Baseline lineal interpretable |
| Random Forest | Ensemble de arboles, captura no-linealidades |
| XGBoost | Gradient boosting, mejor rendimiento en datos tabulares |

**Metrica principal:** F1-Score  
**Justificacion:** Con un desbalanceo del 26% de churn, la accuracy es enganosa. El F1 penaliza tanto falsos negativos (clientes que se van sin detectar) como falsos positivos (recursos desperdiciados). El AUC-ROC complementa midiendo la capacidad discriminativa general.

---

## Output de negocio: Scoring de riesgo

El modelo produce un score (0-1) por cliente, segmentado en tres niveles:

| Nivel | Score | Accion recomendada |
|-------|-------|-------------------|
| High | > 0.6 | Visita urgente — oferta de retencion personalizada |
| Medium | 0.3 – 0.6 | Contacto proactivo — revision de contrato |
| Low | < 0.3 | Mantenimiento — comunicacion periodica |

---

## Planteamiento de los otros dos casos

### Caso 2: Prediccion (Potencial Comercial)
- **Variables clave:** `tenure`, `MonthlyCharges`, servicios contratados, tipo de contrato
- **Modelo:** XGBoost Regressor con validacion cruzada k-fold
- **Metricas:** MAE, RMSE, R²
- **Negocio:** identificar clientes con bajo gasto actual y alto potencial para upsell

### Caso 3: Deteccion de Anomalias
- **Variables clave:** `charge_ratio`, variaciones en `MonthlyCharges`, patrones de uso
- **Modelo:** Isolation Forest
- **Metricas:** Precision@K, validacion manual de casos detectados
- **Negocio:** detectar errores de facturacion, posible fraude o senales tempranas de fuga

---

## Tratamiento del desbalanceo de clases

El dataset tiene un 26.5% de churn — desbalanceo moderado pero suficiente para que un modelo naive aprenda a predecir siempre "No Churn" y obtenga 73% de accuracy sin detectar ningun cliente en riesgo.

La estrategia adoptada es **ponderacion de clases**:

- `LogisticRegression` y `RandomForest` usan `class_weight="balanced"`, que calcula automaticamente un peso inversamente proporcional a la frecuencia de cada clase. Con la distribucion del dataset, la clase churn recibe un peso ~2.8x mayor.
- `XGBoost` usa `scale_pos_weight = n_negativos / n_positivos ≈ 2.83`, que tiene el mismo efecto dentro del framework de gradient boosting.

Esto obliga a los modelos a penalizar mas los falsos negativos (clientes que se van sin ser detectados), que es el error mas costoso desde el punto de vista de negocio.

**Por que no SMOTE u otras tecnicas:**  
Con un desbalanceo del 26.5% (no extremo), la ponderacion de clases es suficiente y mas interpretable. SMOTE genera muestras sinteticas que pueden introducir ruido con variables categoricas, que son mayoritarias en este dataset. El threshold tuning y la calibracion de probabilidades son mejoras validas para una siguiente iteracion.

---

## Limitaciones y mejoras futuras

- El dataset es de telecomunicaciones. Aplicarlo a otros sectores requiere revalidar el feature engineering.
- No se modelan efectos temporales ni estacionalidad del churn.
- El scoring asume que la distribucion de clientes es estable. Se recomienda reentrenamiento periodico.
- Como mejoras al tratamiento del desbalanceo: optimizacion del threshold de clasificacion, SMOTE, o calibracion de probabilidades (Platt scaling).
