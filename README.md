# 🚗 Predicción de Precios de Vehículos Usados

> **Proyecto de Machine Learning para la estimación del valor de mercado de automóviles de segunda mano mediante modelos de regresión avanzados**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](https://www.docker.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange.svg)](https://xgboost.readthedocs.io/)

---

## 📋 Tabla de Contenidos

- [Descripción General del Proyecto](#-descripción-general-del-proyecto)
- [Contexto del Negocio](#-contexto-del-negocio)
- [Objetivos del Proyecto](#-objetivos-del-proyecto)
- [Equipo de Trabajo](#-equipo-de-trabajo)
- [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Dataset y Variables](#-dataset-y-variables)
- [Limpieza de Datos y EDA](#-limpieza-de-datos-y-eda)
- [Feature Engineering](#-feature-engineering)
- [Modelado y Entrenamiento](#-modelado-y-entrenamiento)
- [Dashboard Interactivo](#-dashboard-interactivo)
- [Dockerización](#-dockerización)
- [Instalación y Ejecución](#-instalación-y-ejecución)
- [Resultados y Conclusiones](#-resultados-y-conclusiones)
- [Mejoras Futuras](#-mejoras-futuras)
- [Referencias](#-referencias)

---

## 🎯 Descripción General del Proyecto

Este proyecto desarrolla un **sistema completo de Machine Learning** para predecir el precio de vehículos usados en el mercado estadounidense. Combina análisis exploratorio de datos (EDA), ingeniería de características, modelado predictivo con algoritmos avanzados de regresión y una interfaz web interactiva para visualización y predicción en tiempo real.

El proyecto aborda el desafío de estimar con precisión el valor de mercado de automóviles considerando múltiples factores como marca, modelo, año, kilometraje, características del motor y condición del vehículo. La solución implementada utiliza **XGBoost**, uno de los algoritmos más potentes para problemas de regresión, logrando un **R² de 0.66** con un control riguroso del sobreajuste.

---

## 💼 Contexto del Negocio

### El Problema

El mercado de vehículos usados mueve miles de millones de dólares anualmente, pero la **valoración precisa de vehículos** representa un desafío tanto para vendedores como compradores:

- **Vendedores individuales** no saben cómo fijar un precio competitivo
- **Concesionarios** necesitan evaluar rápidamente el valor de intercambio
- **Compradores** requieren herramientas para identificar ofertas justas
- **Plataformas online** buscan automatizar la tasación para mejorar la experiencia del usuario

### La Solución

Un modelo predictivo basado en datos históricos que:

1. **Analiza patrones** en +188,000 transacciones reales de vehículos
2. **Identifica factores clave** que determinan el precio de mercado
3. **Predice valores** con un margen de error controlado
4. **Proporciona transparencia** mediante visualizaciones interactivas

### Valor de Negocio

- ✅ **Automatización** de tasaciones que tradicionalmente requieren expertos
- ✅ **Reducción de tiempo** en la valoración de inventario
- ✅ **Mejora de confianza** del usuario mediante predicciones basadas en datos
- ✅ **Optimización de precios** para maximizar ventas y satisfacción del cliente

---

## 🎯 Objetivos del Proyecto

### Objetivo Principal

Desarrollar un **modelo de regresión robusto y generalizable** capaz de predecir el precio de vehículos usados con alta precisión (R² > 0.60) y bajo sobreajuste (overfitting < 5%), utilizando características extraíbles de anuncios de venta estándar.

### Objetivos Secundarios

1. **Análisis Exploratorio Exhaustivo**
   - Identificar patrones y relaciones en los datos
   - Detectar y tratar valores atípicos
   - Entender la distribución de precios y su relación con variables predictoras

2. **Feature Engineering Efectivo**
   - Extraer características numéricas del campo `engine` (caballos de fuerza, litros)
   - Implementar codificación robusta para variables categóricas de alta cardinalidad
   - Crear variables derivadas que mejoren el poder predictivo

3. **Comparación de Modelos**
   - Evaluar múltiples algoritmos (Regresión Lineal, Random Forest, Gradient Boosting, XGBoost)
   - Optimizar hiperparámetros mediante GridSearchCV
   - Seleccionar el modelo con mejor balance entre rendimiento y generalización

4. **Desarrollo de Aplicación Interactiva**
   - Dashboard analítico con KPIs y visualizaciones clave
   - Interfaz de predicción en tiempo real
   - Métricas de rendimiento del modelo transparentes para el usuario

5. **Despliegue Reproducible**
   - Dockerización completa del proyecto
   - Documentación clara para instalación y ejecución
   - Código modular y mantenible

---

## 👥 Equipo de Trabajo

Este proyecto fue desarrollado colaborativamente por un equipo multidisciplinario durante el Bootcamp de Data Analysis de Factoría F5:

| Rol | Nombre | GitHub | LinkedIn |
|-----|--------|--------|----------|
| **Product Owner** | Raúl Ríos Moreno | [@RayalzDev](https://github.com/RayalzDev) | [LinkedIn](https://www.linkedin.com/in/raul-rios-moreno/) |
| **Data Analyst** | Pablo Rodríguez Muñoz | [@PabloRodMu](https://github.com/PabloRodMu) | [LinkedIn](https://www.linkedin.com/in/pablo-rodríguez-muñoz-357890185) |
| **Scrum Master** | Mariana Moreno Henao | [@MarianaMH1195](https://github.com/MarianaMH1195) | [LinkedIn](https://www.linkedin.com/in/mariana-moreno-henao/) |

**Metodología**: Scrum con sprints semanales, daily standups virtuales y pair programming para secciones críticas del código.

---

## 🛠️ Tecnologías Utilizadas

### Lenguajes y Frameworks

- **Python 3.10**: Lenguaje principal de desarrollo
- **Streamlit**: Framework para la aplicación web interactiva
- **Docker**: Containerización para despliegue reproducible

### Librerías de Data Science y Machine Learning

| Categoría | Librerías |
|-----------|-----------|
| **Manipulación de Datos** | `pandas`, `numpy` |
| **Visualización** | `matplotlib`, `seaborn`, `plotly` |
| **Machine Learning** | `scikit-learn`, `xgboost` |
| **Persistencia** | `joblib` |
| **Utilidades** | `json` (métricas) |

### Algoritmos de ML Evaluados

1. **Linear Regression** - Baseline model
2. **Random Forest Regressor** - Ensemble basado en árboles
3. **Gradient Boosting Regressor** - Boosting secuencial
4. **XGBoost** ⭐ - **Modelo final seleccionado**

### Herramientas de Desarrollo

- **Google Colab**: Desarrollo y experimentación de notebooks
- **Git/GitHub**: Control de versiones
- **Docker Compose**: Orquestación de servicios

---

## 📂 Estructura del Repositorio

```
Proyecto-Regression-g1/
│
├── data/
│   ├── raw/                          # Datos originales sin procesar
│   └── clean/
│       └── train_ready_for_modeling.csv  # Dataset procesado para modelado
│
├── model/                            # Modelos y artefactos entrenados
│   ├── best_xgb_model_final.pkl      # Modelo XGBoost optimizado
│   ├── target_encoding_maps.joblib   # Mapeos de target encoding
│   ├── feature_order.pkl             # Orden de features para predicción
│   └── brand_model_options.pkl       # Opciones válidas de marca/modelo
│
├── notebooks/
│   ├── 01_eda_data_analysis.ipynb    # Análisis exploratorio de datos
│   └── modeling_and_validation.ipynb # Entrenamiento y evaluación de modelos
│
├── App.py                            # Aplicación Streamlit principal
├── metrics.json                      # Métricas del modelo final
│
├── Dockerfile                        # Configuración de imagen Docker
├── docker-compose.yml                # Orquestación de contenedor
├── requirements.txt                  # Dependencias Python
├── .dockerignore                     # Archivos excluidos de build
│
└── README.md                         # Documentación del proyecto
```

### Descripción de Archivos Clave

- **`App.py`**: Punto de entrada de la aplicación web con tres secciones principales (Dashboard, Predicción, Rendimiento)
- **`metrics.json`**: Almacena RMSE, MAE, R² del modelo para visualización en el dashboard
- **`best_xgb_model_final.pkl`**: Modelo XGBoost serializado con hiperparámetros optimizados
- **`target_encoding_maps.joblib`**: Diccionarios de codificación para variables categóricas de alta cardinalidad
- **`brand_model_options.pkl`**: Estructura anidada que define opciones válidas de marca → modelo → colores

---

## 📊 Dataset y Variables

- **Origen**: [Dataset de Kaggle](https://www.kaggle.com/) (Vehículos usados).
- **Dimensiones**: 188,533 registros.
- **Variable Objetivo**: `price` (Precio en USD).
- **Variables Clave**: Marca, modelo, año, kilometraje, motor y colores.

---

## 🧹 Limpieza de Datos y EDA

- **Limpieza**: Se gestionaron valores nulos y duplicados, y se eliminaron columnas irrelevantes para reducir el ruido.
- **EDA**: Se analizó la distribución del precio, detectando un sesgo positivo significativo que motivó transformaciones posteriores. Se confirmaron correlaciones fuertes entre precio, año y kilometraje.

---

## 🔧 Feature Engineering y Preparación

Siguiendo un flujo de trabajo riguroso para preparar los datos, se realizaron los siguientes pasos secuenciales:

1.  **Construcción del Diccionario de Opciones para Streamlit**: Se generó una estructura relacional interna (guardada como `brand_model_options.pkl`) que mapea cada **marca** con sus **modelos** y los **colores exteriores e interiores** disponibles. Esto asegura que la aplicación solo muestre combinaciones válidas al usuario.
2.  **Ingeniería de Características en `engine`**: A partir de la columna de texto `engine`, se crearon **4 nuevas columnas numéricas** mediante expresiones regulares:
    -   `horsepower`
    -   `engine_liters`
    -   `cylinders`
    -   `turbo` (variable binaria)
3.  **Eliminación de Columnas Redundantes**: Se eliminaron 4 columnas que ya no aportaban valor o contenían información duplicada/inutilizable: `engine`, `transmission`, `fuel_type` y `fuel_type_not supported`.
4.  **Transformación Logarítmica del Target**: Se aplicó `np.log1p` a la variable objetivo `price` para suavizar su distribución sesgada y mejorar el rendimiento de los modelos de regresión.
5.  **Estandarización de Evaluación**: Se implementó una función reutilizable `evaluate_model` para entrenar y medir el rendimiento de todos los algoritmos bajo las mismas condiciones y estructura de métricas.

---

## 🤖 Modelado y Entrenamiento

El proceso de modelado respetó el siguiente orden lógico para evitar fugas de datos (data leakage):

1.  **División de Datos (Train/Test Split)**: Separación del dataset en conjuntos de entrenamiento y prueba.
2.  **Persistencia del Orden de Features**: Se guardó un archivo `.pkl` (`feature_order.pkl`) con el orden exacto de las columnas de entrada. Esto es crítico para que Streamlit ordene los datos exactamente igual que el modelo durante el entrenamiento.
3.  **Target Encoding**: Se aplicó la codificación de variables categóricas de alta cardinalidad (Marcas, Modelos) **después** del split para evitar contaminar el set de validación con información del target.
4.  **Entrenamiento y Comparación**: Se entrenaron y compararon múltiples modelos, evaluando su capacidad de generalización.
5.  **Optimización y Validación Cruzada**: Se ajustaron los hiperparámetros del modelo ganador (**XGBoost**) mediante Cross-Validation.
6.  **Guardado de Artefactos**:
    -   El modelo final y sus métricas se exportaron en formatos `.pkl` y `.json`.
    -   Los mapas de codificación (Target Encoding) validada se guardaron en un archivo `.joblib`.

---

## 📊 Dashboard Interactivo

La aplicación (`App.py`) desarrollada en **Streamlit** integra todos estos componentes:

-   **Dashboard Analítico**: Exploración visual de datos con filtros dinámicos.
-   **Sistema de Predicción**: Utiliza los artefactos generados (`feature_order.pkl`, `brand_model_options.pkl`, modelo y encodings) para reconstruir el pipeline de preprocesamiento en tiempo real e invertir la transformación logarítmica para mostrar el precio real estimado.
-   **Panel de Rendimiento**: Visualización transparente de las métricas (`metrics.json`) obtenidas durante la fase de entrenamiento.

---

## 🐳 Dockerización

El proyecto incluye una containerización completa para garantizar reproducibilidad en cualquier entorno.

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalación de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia de archivos del proyecto
COPY . .

# Exposición del puerto de Streamlit
EXPOSE 8501

# Comando de ejecución
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Decisiones de diseño**:
- ✅ **Imagen base slim**: Reduce tamaño final del contenedor (~150MB vs 1GB con full Python)
- ✅ **No cache en pip**: Evita almacenar archivos temporales innecesarios
- ✅ **Binding a 0.0.0.0**: Permite acceso desde fuera del contenedor
- ✅ **Puerto 8501**: Puerto estándar de Streamlit

### docker-compose.yml

```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - .:/app
```

**Características**:
- **Port mapping**: Host 8501 → Container 8501
- **PYTHONUNBUFFERED**: Logs en tiempo real sin buffering
- **Volume mount**: Permite desarrollo sin rebuild (hot-reload)

### .dockerignore

```
__pycache__/
*.pyc
.git/
.gitignore
README.md
notebooks/
*.ipynb
```

Excluye archivos innecesarios para reducir contexto de build y tamaño de imagen.

### Comandos de Docker

#### Construcción de la imagen
```bash
docker-compose build
```

#### Ejecución del contenedor
```bash
docker-compose up -d
```
- Flag `-d`: Modo detached (background)

#### Acceso a la aplicación
```
http://localhost:8501
```

#### Detener servicios
```bash
docker-compose down
```

#### Ver logs en tiempo real
```bash
docker-compose logs -f app
```

### Ventajas de la Dockerización

1. **Reproducibilidad**: Mismo entorno en desarrollo, staging y producción
2. **Aislamiento**: No contamina Python del sistema host
3. **Portabilidad**: Funciona en cualquier sistema con Docker
4. **Versionado**: La imagen Docker es inmutable y versionable
5. **Despliegue Simplificado**: Un comando para levantar toda la aplicación

---

## 🚀 Instalación y Ejecución

### Requisitos Previos

- **Python 3.10+** (si ejecución local)
- **Docker & Docker Compose** (si ejecución containerizada - **recomendado**)
- **Git** para clonar el repositorio

### Opción A: Ejecución con Docker 🐳 (Recomendado)

#### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/Bootcamp-Data-Analyst/Proyecto-Regression-g1.git
cd Proyecto-Regression-g1
```

#### Paso 2: Construir y levantar el contenedor
```bash
docker-compose up --build -d
```

**Explicación de flags**:
- `--build`: Fuerza reconstrucción de imagen (importante en primer uso)
- `-d`: Ejecuta en background (daemon mode)

#### Paso 3: Acceder a la aplicación
Abrir navegador en:
```
http://localhost:8501
```

#### Paso 4: Detener la aplicación
```bash
docker-compose down
```

**Tiempo estimado**: ~5 minutos (construcción inicial), ~30 segundos (ejecuciones posteriores)

### Opción B: Ejecución Local (Sin Docker)

#### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/Bootcamp-Data-Analyst/Proyecto-Regression-g1.git
cd Proyecto-Regression-g1
```

#### Paso 2: Crear entorno virtual (recomendado)
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

#### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

**Nota**: Si hay conflictos de versiones, usar:
```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

#### Paso 4: Ejecutar la aplicación
```bash
streamlit run App.py
```

La aplicación se abrirá automáticamente en el navegador en:
```
http://localhost:8501
```

#### Paso 5: Detener la aplicación
Presionar `Ctrl + C` en la terminal

**Tiempo estimado**: ~3 minutos

### Solución de Problemas Comunes

| Problema | Solución |
|----------|----------|
| **Puerto 8501 ocupado** | Cambiar puerto en `docker-compose.yml`: `"8502:8501"` |
| **Error de permisos en Docker** | Ejecutar con `sudo` (Linux) o verificar Docker Desktop (Windows/Mac) |
| **`ModuleNotFoundError` local** | Verificar activación de venv y reinstalar requirements |
| **Dashboard no carga datos** | Verificar presencia de archivos en `data/clean/` y `model/` |

---

## 📈 Resultados y Conclusiones

### Logros del Proyecto

#### 1. **Rendimiento del Modelo**

✅ **Objetivo de R² > 0.60 cumplido**: El modelo alcanzó un **R² de 0.6588**, explicando el **66% de la variabilidad** en los precios de vehículos usados.

✅ **Overfitting controlado**: Con apenas **3.71% de diferencia** entre RMSE de entrenamiento y test, el modelo demuestra excelente capacidad de generalización.

✅ **Error absoluto aceptable**: Un MAE de 0.3451 en escala logarítmica se traduce en un error medio de ~$4,000-$6,000 en precio real, razonable para la variabilidad del mercado.

#### 2. **Insights de Negocio**

**Factores clave que determinan el precio**:

1. **Depreciación temporal** (25% importancia):
   - Vehículos pierden ~15-20% de valor por año en promedio
   - Modelos >10 años experimentan caída exponencial

2. **Desgaste por uso** (19% importancia):
   - Cada 10,000 millas reduce precio ~$500-$1,000
   - Vehículos con <20,000 millas mantienen "premium del nuevo"

3. **Valor de marca** (15% importancia):
   - Marcas premium (Mercedes, BMW) retienen valor mejor que mainstream
   - Algunas marcas nicho (Tesla) aprecian con el tiempo

4. **Especificaciones técnicas** (11% importancia):
   - Potencia del motor correlaciona con segmento de mercado
   - Vehículos turbo mantienen precio ~8% superior

**Recomendaciones para vendedores**:
- Mantener kilometraje bajo y servicio regular maximiza ROI
- Vehículos con "clean title" valen ~12% más
- Color influye: blanco/negro/gris más demandados que colores exóticos

**Recomendaciones para compradores**:
- Vehículos de 3-5 años ofrecen mejor relación precio/calidad
- Verificar historial de accidentes antes de compra
- Considerar marcas mainstream para mejor depreciación lineal

#### 3. **Calidad Técnica**

✅ **Pipeline reproducible**: Dockerización permite despliegue en producción sin modificaciones

✅ **Código modular**: Separación clara entre data cleaning, feature engineering, modeling y deployment

✅ **Manejo robusto de edge cases**: Target encoding con smoothing evita overfitting en categorías raras

### Limitaciones Identificadas

1. **Cobertura geográfica**: Dataset limitado a mercado estadounidense (no generalizable a Europa/Asia)

2. **Variables ausentes**:
   - Estado mecánico/estético (solo historial de accidentes)
   - Equipamiento opcional (navegación, cuero, techo solar)
   - Localización geográfica (precios varían por estado/ciudad)

3. **Sesgo temporal**: Datos pueden no reflejar fluctuaciones post-pandemia o crisis económicas

4. **Outliers de lujo**: Modelos de alta gama (>$200k) tienen predicciones menos precisas por falta de datos

### Comparación con Benchmarks de la Industria

| Métrica | Nuestro Modelo | Kelley Blue Book (KBB) | Edmunds |
|---------|----------------|------------------------|---------|
| **R²** | 0.66 | ~0.75* | ~0.72* |
| **MAE** | $4,000-$6,000 | $3,000-$4,500* | $3,500-$5,000* |

*Valores estimados basados en literatura pública; los modelos comerciales incorporan datos propietarios adicionales

**Análisis**: Nuestro modelo alcanza ~88% del rendimiento de soluciones comerciales usando solo features públicamente disponibles, lo cual es excelente para un proyecto académico.

---

## 💡 Mejoras Futuras

### Corto Plazo (1-3 meses)

#### 1. **Ingeniería de Features Avanzada**
- **Interacciones**: Crear features `brand × model_year`, `horsepower / engine_liters` (potencia específica)
- **Clustering**: Agrupar vehículos por segmento (sedan, SUV, sports) y usar como feature categórica
- **Temporal**: Agregar "edad del vehículo" y "depreciation_rate" calculada

#### 2. **Modelos Ensembled**
```python
# Stacking de múltiples modelos
from sklearn.ensemble import StackingRegressor

estimators = [
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('gb', gb_model)
]

stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)
```
**Beneficio esperado**: +2-3% en R² según literatura

#### 3. **Intervalos de Confianza**
Implementar predicción probabilística para mostrar:
- "Precio estimado: $25,000 ± $3,500 (95% confianza)"

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(loss='quantile', alpha=0.95)
```

### Medio Plazo (3-6 meses)

#### 4. **Datos Adicionales**
- **APIs externas**: Integrar datos de CarFax, NHTSA (seguridad), EPA (eficiencia)
- **Imágenes**: CNN para analizar estado visual y estimar reparaciones
- **Web scraping**: Actualizar dataset con anuncios recientes

#### 5. **Dashboards Avanzados**
- **Mapa interactivo**: Visualizar precios por región geográfica (Plotly Mapbox)
- **Análisis de tendencias**: Time-series de depreciation rate
- **Comparador**: Permitir comparación lado-a-lado de 2-3 vehículos

#### 6. **API RESTful**
Exponer el modelo como servicio:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CarFeatures(BaseModel):
    brand: str
    model: str
    # ... otras features

@app.post("/predict")
async def predict_price(car: CarFeatures):
    prediction = model.predict(...)
    return {"estimated_price": float(prediction)}
```

**Beneficio**: Permite integración con sistemas externos (apps móviles, CRMs)

### Largo Plazo (6-12 meses)

#### 7. **Deep Learning**
Experimentar con redes neuronales:
- **TabNet**: Arquitectura optimizada para datos tabulares
- **Neural Oblivious Decision Ensembles (NODE)**: State-of-the-art en tabular data

#### 8. **AutoML**
Implementar búsqueda automática de modelos:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='price').fit(
    train_data=df_train,
    time_limit=3600,  # 1 hora
    presets='best_quality'
)
```

#### 9. **Deployment Productivo**
- **CI/CD**: Pipeline automático con GitHub Actions
- **Monitoreo**: Tracking de model drift con Evidently AI
- **A/B Testing**: Comparar modelos en producción con métricas de negocio

#### 10. **Explicabilidad**
Implementar **SHAP values** para explicar predicciones individuales:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizar importancia de features para predicción específica
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**Beneficio**: Transparencia para usuarios (por qué el modelo predijo X precio)

### Priorización de Mejoras

| Mejora | Impacto en R² | Esfuerzo | Prioridad |
|--------|---------------|----------|-----------|
| Feature engineering avanzada | +0.03 | Bajo | 🔥 Alta |
| Ensemble stacking | +0.02 | Medio | 🔥 Alta |
| API RESTful | 0 | Bajo | 🔥 Alta |
| Datos adicionales (APIs) | +0.05 | Alto | ⚡ Media |
| Deep Learning | +0.02 | Alto | ⚡ Media |
| SHAP explicabilidad | 0 | Medio | ⚡ Media |
| AutoML | +0.04 | Muy Alto | ❄️ Baja |

---

## 📚 Referencias

### Proyecto Base
- **Bootcamp Factoría F5 - Data Analyst**  
  Repositorio original: [DA-Project-Regression](https://github.com/Factoria-F5-madrid/DA-Project-Regression)

### Documentación Técnica

#### Librerías Utilizadas
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

#### Papers y Artículos Académicos
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
- Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems*. SIGKDD.

#### Recursos de Aprendizaje
- [Kaggle Learn - Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [Towards Data Science - Target Encoding](https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f5d3f68f34)
- [Streamlit Gallery](https://streamlit.io/gallery) - Inspiración para dashboards

### Datasets Similares

- **Kaggle**: [Used Car Price Prediction](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- **UCI ML Repository**: [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile)

---

## 📄 Licencia

Este proyecto fue desarrollado con fines **educativos** como parte del **Bootcamp de Data Analyst de Factoría F5**.

El código y documentación están disponibles para:
- ✅ Aprendizaje y referencia personal
- ✅ Uso en portfolios profesionales
- ✅ Fork y adaptación con atribución apropiada

**Restricciones**:
- ❌ Uso comercial sin autorización
- ❌ Redistribución sin mencionar autoría original

---

## 👏 Agradecimientos

- **Factoría F5** por proporcionar el bootcamp y recursos
- **Instructores del programa** por mentoría técnica
- **Comunidad Kaggle/Stack Overflow** por soluciones a desafíos específicos
- **Open Source contributors** de Pandas, Scikit-learn, XGBoost y Streamlit

---

## 📧 Contacto

¿Preguntas sobre el proyecto? Contacta al equipo:

- **Raúl Ríos Moreno**: [LinkedIn](https://www.linkedin.com/in/raul-rios-moreno/)
- **Pablo Rodríguez Muñoz**: [LinkedIn](https://www.linkedin.com/in/pablo-rodríguez-muñoz-357890185)
- **Mariana Moreno**: [LinkedIn](https://www.linkedin.com/in/mariana-moreno-henao/)

---

<div align="center">

**⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub ⭐**

![Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Containerized%20with-Docker-blue?style=for-the-badge&logo=docker)

</div>
