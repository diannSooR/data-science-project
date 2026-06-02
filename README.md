# Análisis de Riesgos Multivariable — Ciudad de México

Proyecto de ciencia de datos que analiza y predice zonas de alto riesgo en la CDMX
combinando datos de sismos, inundaciones, fracturas geológicas, laderas y marginación urbana.

## Estructura del proyecto

```
├── src/                          # Módulos Python reutilizables
│   ├── __init__.py
│   ├── config.py                 # Configuración, rutas, constantes
│   ├── data_loader.py            # Carga y validación de datos
│   ├── preprocessing.py          # Limpieza y feature engineering
│   ├── modeling.py               # Entrenamiento y evaluación de modelos
│   ├── visualization.py          # Generación de gráficos y mapas
│   └── utils.py                  # Funciones auxiliares
│
├── outputs/                      # Resultados generados
│   ├── charts/                   # Gráficos EDA (correlaciones, distribuciones)
│   ├── maps/                     # Mapas generales
│   ├── modelo/
│   │   ├── charts/               # Métricas del modelo (ROC, confusion, importance)
│   │   ├── maps/                 # Mapas de predicciones
│   │   └── comparativas/         # Comparación entre modelos
│   └── agbs_regeneradas/         # AGEBs procesadas y métricas exportadas
│
├── 01_limpieza_datos.ipynb       # Limpieza y validación (2 453 AGEBs)
├── 02_analisis_exploratorio.ipynb# EDA con todas las features
├── 03_modelado_supervisado.ipynb # Modelo optimizado (GB, XGBoost, RF, Ensemble)
├── run_pipeline.py               # Script que ejecuta el pipeline completo
├── requirements.txt              # Dependencias del proyecto
└── zonas_ageb_clean_v3.json      # Fuente de datos principal (2 453 AGEBs)
```

## Requisitos

- Python 3.10 – 3.14
- Git
- VS Code con Jupyter, o Jupyter Notebook/Lab

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/angelrz/data-science-project.git
cd data-science-project/
```

### 2. Crear entorno virtual

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (CMD):**
```cmd
py -m venv venv
.\venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

> Si PowerShell da error de política, ejecuta una vez:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

### Opción 1: Pipeline completo (recomendado)

```bash
python run_pipeline.py
```

Esto ejecuta todo el flujo:
1. Carga y valida 2 453 AGEBs desde `zonas_ageb_clean_v3.json`
2. Crea el target binarizado (alto riesgo si emergencias > P75)
3. Genera features de ingeniería
4. Entrena 4 modelos (Gradient Boosting, XGBoost, Random Forest, Ensemble)
5. Selecciona el mejor modelo y genera todas las visualizaciones
6. Guarda resultados en `outputs/`

### Opción 2: Notebooks individuales

Ejecutar en orden:
1. `01_limpieza_datos.ipynb` — Limpieza y validación
2. `02_analisis_exploratorio.ipynb` — Análisis exploratorio
3. `03_modelado_supervisado.ipynb` — Entrenamiento y evaluación

## Resultados del modelo

| Modelo | Accuracy | AUC-ROC | Precision | Recall | F1 |
|--------|----------|---------|-----------|--------|-----|
| Gradient Boosting | 0.8778 | 0.9399 | 0.7788 | 0.6864 | 0.7297 |
| XGBoost | 0.8819 | 0.9421 | 0.7885 | 0.6949 | 0.7387 |
| Random Forest | 0.8574 | 0.9148 | 0.6905 | 0.7373 | 0.7131 |
| **Ensemble (GB+XGB)** | **0.8941** | **0.9422** | **0.7426** | **0.8559** | **0.7953** |

**Objetivos cumplidos:**
- ✅ Accuracy ≥ 80 % → 89.41 %
- ✅ AUC-ROC ≥ 0.85 → 0.9422

## Datos

- **Fuente principal:** `zonas_ageb_clean_v3.json` — 2 453 AGEBs (2 431 urbanas + 22 rurales)
- **Features:** 13 features físicas base + índice compuesto + features de ingeniería = 25 total
- **Target:** `ruse_emergencias` binarizada (1 si > percentil 75)

### Nota sobre el índice de riesgo compuesto

El `indice_riesgo_compuesto` se usa como feature y contiene un 7 % de peso
de `ruse_emergencias_norm`. El 93 % restante proviene de variables físicas.
La correlación con el target es 0.30, confirmando que no hay data leakage
significativo. Se documenta para transparencia.

## Notas importantes

- Todos los paths usan `pathlib.Path` para compatibilidad Windows/Linux.
- No instales paquetes con `!pip install` dentro del notebook.
- Si VS Code marca errores de importación, verifica que el kernel sea el del entorno `venv`.
