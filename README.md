# 📊 Regression Project – Data Analysis Bootcamp

Este proyecto tiene como objetivo predecir el precio de coches usados utilizando modelos de **Machine Learning** (XGBoost). Se ha desarrollado una interfaz interactiva con **Streamlit** para visualizar los datos y realizar predicciones en tiempo real.

El proyecto ha sido **dockerizado** para garantizar una fácil reproducción y despliegue.

---

## 🚀 Características Principales

*   **Dashboard Analítico**: Visualización de KPIs y gráficas exploratorias (distribución de precios, kilometraje vs precio, etc.).
*   **Predicción de Precios**: Formulario interactivo para estimar el precio de un coche en base a sus características (marca, modelo, año, etc.).
*   **Métricas del Modelo**: Sección dedicada a evaluar el rendimiento del modelo (RMSE, MAE, R²).

---

## 🛠️ Tecnologías Utilizadas

*   **Lenguaje**: Python 3.10
*   **Librerías Principales**:
    *   `streamlit`: Interfaz de usuario web.
    *   `xgboost`: Modelo de regresión.
    *   `pandas` & `numpy`: Manipulación de datos.
    *   `matplotlib` & `seaborn`: Visualización.
    *   `scikit-learn`: Métricas y preprocesamiento.
*   **Infraestructura**: Docker & Docker Compose.

---

## 🔧 Instalación y Ejecución

Puedes ejecutar el proyecto de dos formas: usando Docker (recomendado) o instalando las dependencias localmente.

### Opción A: Usando Docker (Recomendado)

Asegúrate de tener instalado [Docker](https://www.docker.com/) y [Docker Compose](https://docs.docker.com/compose/).

1.  **Clonar el repositorio** (si no lo has hecho ya):
    ```bash
    git clone https://github.com/Bootcamp-Data-Analyst/Proyecto-Regression-g1.git
    cd Proyecto-Regression-g1
    ```

2.  **Construir y levantar el contenedor**:
    ```bash
    docker-compose up --build -d
    ```

3.  **Acceder a la aplicación**:
    Abre tu navegador en: [http://localhost:8501](http://localhost:8501)

4.  **Detener la aplicación**:
    ```bash
    docker-compose down
    ```

### Opción B: Ejecución Local

1.  **Crear un entorno virtual (opcional pero recomendado)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/Mac
    .\venv\Scripts\activate   # En Windows
    ```

2.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar la aplicación**:
    ```bash
    streamlit run App.py
    ```

---

## 📂 Estructura del Proyecto

```
Proyecto-Regression-g1/
├── data/               # Datasets (raw, clean)
├── model/              # Modelos entrenados (.pkl, .joblib)
├── notebooks/          # Notebooks de Jupyter para EDA y modelado
├── src/                # Código fuente auxiliar (si aplica)
├── App.py              # Punto de entrada de la aplicación Streamlit
├── Dockerfile          # Configuración de la imagen Docker
├── docker-compose.yml  # Orquestación del contenedor
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación
```

---

## 👥 Equipo

Este proyecto ha sido desarrollado por el Grupo 1 del Bootcamp de Data Analysis.

*   Análisis Exploratorio (EDA)
*   Ingeniería de Características y Modelado
*   Desarrollo de la Aplicación y Dockerización

---

## � Licencia

Este proyecto es para fines educativos dentro del Bootcamp de Data Analyst.
