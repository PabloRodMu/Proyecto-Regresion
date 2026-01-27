# Importaciones
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib
import json

# Funciones
@st.cache_data # Mete en caché la carga de datos


def load_data():
    return pd.read_csv("data/clean/train_ready_for_modeling.csv")

df = load_data() # Carga de datos


modelo = joblib.load("best_xgb_model_final.pkl") # Carga del modelo

final_columns = modelo.get_booster().feature_names # Columnas finales usadas en el modelo

with open("metrics.json", "r") as f:
    metrics = json.load(f)



# Layout

st.set_page_config(page_title="App de Regresión", layout="wide")


TARGET = "price"


st.sidebar.title("Menú")

section = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "📊 Dashboard Analítico",
        "🔮 Predicción",
        "📈 Rendimiento del Modelo",
        "📝 Feedback",
    ],
)





# Secciones

if section == "📊 Dashboard Analítico":
    st.title("📊 Dashboard Analítico")
    st.write("Aquí irán los gráficos del dataset.")
    st.markdown(
        """
    ### ¿Qué muestra este dashboard?

    Este dashboard permite explorar el dataset utilizado para entrenar
    un modelo de regresión. El objetivo es entender la distribución de la
    variable objetivo y su relación con otras variables.
    """
    )
    st.markdown(
        f"""
    ### 🎯 Variable objetivo
    El objetivo del modelo es predecir el **precio (`{TARGET}`)** de un coche usado
    en función de sus características técnicas y de uso.
    """
    )
    
    # Filtro de marcas
    with st.expander("Filtro de marcas"):
            
            selected_brands = st.multiselect(
            "Filtrar marcas:",
            df["brand"].unique(),
            default=df["brand"].unique()
            )
            filtered_df = df[df["brand"].isin(selected_brands)]
    # KPI del precio
    with st.expander("Métricas clave del precio"):
        
        col1, col2, col3 = st.columns(3)

        col1.metric("Precio medio", f"${filtered_df[TARGET].mean():,.0f}")
        col2.metric("Precio mediano", f"${filtered_df[TARGET].median():,.0f}")
        col3.metric("Desviación estándar", f"${filtered_df[TARGET].std():,.0f}")

        st.markdown(
            """
        **Insight:**  
        La diferencia entre el precio medio y mediano sugiere una distribución sesgada,
        probablemente debido a la presencia de coches de lujo con precios muy altos.
        """
        )


    # Gráfico de distribución de la variable objetivo

    with st.expander("Distribución del precio (hasta 500.000 $)"):

        fig, ax = plt.subplots(figsize=(5, 2))

        filtered_prices = filtered_df[filtered_df[TARGET] < 500_000][TARGET]
        ax.hist(filtered_prices, bins=40)

        ax.set_xlabel("Precio ($)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución del precio de coches usados")

        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
        )

        st.pyplot(fig)

        st.markdown(
            """
        **Insight:**  
        La mayoría de los coches usados se concentran por debajo de 500.000 $.
        Los valores extremos corresponden a vehículos premium o de alta gama.
        """
        )


    
    # Boxplot de precio
    with st.expander("Boxplot del precio"):
        
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.boxplot(filtered_df[TARGET], vert=False)
        ax.set_title("Boxplot del precio")
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
        )
        st.pyplot(fig)

        st.markdown(
            """
        **Insight:**  
        Algunos coches tienen precios extremadamente altos, lo que indica la presencia de outliers en el dataset.
        """
        )

    # Scatter plot de precio vs kilometraje
    with st.expander("Ver relación entre kilometraje y precio"):

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.scatter(filtered_df["milage"], filtered_df[TARGET], alpha=0.4)
        ax.set_xlabel("Kilometros (KM)")
        ax.set_ylabel("Precio ($)")
        ax.set_title("Kilometraje vs Precio")
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
        )

        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
        )

        st.pyplot(fig)

        st.markdown(
            """
        **Insight:**  
        Existe una tendencia negativa entre el kilometraje y el precio: a mayor kilometraje, menor precio.
        """
        )

    # Scatter  plot precio vs año
    with st.expander("Ver relación entre año del modelo y precio"):

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.scatter(filtered_df["model_year"], filtered_df[TARGET], alpha=0.4)
        ax.set_xlabel("Año del modelo")
        ax.set_ylabel("Precio ($)")
        ax.set_title("Año del modelo vs Precio")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

        st.pyplot(fig)

        st.markdown(
            """
        **Insight:**  
        Hay una clara tendencia positiva entre el año del modelo y el precio: los modelos más nuevos tienden a tener precios más altos.
        """
        )

elif section == "🔮 Predicción":
    st.title("🔮 Predicción")
    st.markdown("Introduce las características del coche:")

    # Inputs de usuario
    
    brand = st.selectbox("Marca", df["brand"].unique())
    model_year = st.number_input("Año del modelo", min_value=int(df["model_year"].min()), max_value=int(df["model_year"].max()), value=2020)
    milage = st.number_input("Kilometraje", min_value=0, max_value=int(df["milage"].max()), value=50000, step=1000)
    engine = st.selectbox("Motor", df["engine"].unique())
    transmission = st.selectbox("Transmisión", df["transmission"].unique())
    ext_col = st.text_input("Color exterior", value="Negro")
    int_col = st.text_input("Color interior", value="Negro")
    
    # Ejemplo para fuel_type
    fuel_type = st.selectbox("Tipo de combustible", ["E85 Flex Fuel", "gasoline", "hybrid", "unkown", "not supported", "-"])
    
    clean_title_yes = st.checkbox("Título limpio", value=True)
    accident_none_reported = st.checkbox("Sin accidentes reportados", value=True)

    # Botón de predicción
    if st.button("Predecir precio"):
        
        # Recopilar inputs en un diccionario
        user_inputs = {
            "brand": brand,
            "transmission": transmission,
            "fuel_type": fuel_type,
            "ext_col": ext_col,
            "int_col": int_col
        }
                
        # Aquí se prepara el input para el modelo
        input_dict = { col: 0 for col in final_columns }
        
        input_dict["milage"] = milage
        input_dict["model_year"] = model_year
        input_dict["clean_title_yes"] = int(clean_title_yes)
        input_dict["accident_none_reported"] = int(accident_none_reported)
        
        for category, value in user_inputs.items():
        # Construimos el nombre de la columna que debe activarse
            col_name = f"{category}_{value}"
            if col_name in final_columns:
                input_dict[col_name] = 1
                # El resto ya están en 0 gracias a la inicialización


        # Convertir a DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[final_columns]  # Asegurar el orden correcto de columnas

        # Predicción con modelo
        pred_log = 10.679376
        price_real = np.exp(pred_log)
        print(price_real)
        
        preds = modelo.predict(input_df)
        print(np.min(preds), np.max(preds), np.mean(preds))

        
        # Mostrar resultado
        st.success(f"💰 Precio estimado: ${price_real:,.0f}")

elif section == "📈 Rendimiento del Modelo":
    st.title("📈 Rendimiento del Modelo")
    st.write("Aquí se muestran métricas del modelo.")
    
    #pinga
    
    st.subheader("Métricas del modelo")

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", round(metrics["rmse"], 3))
    col2.metric("MAE", round(metrics["mae"], 3))
    col3.metric("R²", round(metrics["r2"], 3))


    

elif section == "📝 Feedback":
    st.title("📝 Feedback")
    st.write("Aquí se recoge feedback del usuario.")


df = pd.DataFrame(
    {
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
        "target": np.random.rand(100) * 1000,
    }
)
