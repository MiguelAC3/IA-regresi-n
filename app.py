import streamlit as st
import tensorflow as tf
import numpy as np
import joblib # Para cargar el modelo de regresión logística

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Personalidad", page_icon="🧠")

st.title("🧠 Detector de Personalidad con IA")
st.write("Ingresa tus datos y elige qué Inteligencia Artificial quieres usar para el análisis.")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Configuración del Modelo")
tipo_modelo = st.sidebar.radio(
    "Elige el modelo de predicción:",
    ("Red Neuronal (Deep Learning)", "Regresión Logística (Clásico)")
)

st.sidebar.info(
    "ℹ️ **Nota:** La Red Neuronal suele captar patrones complejos, "
    "mientras que la Regresión Logística es excelente para relaciones lineales directas."
)

# 2. Funciones para cargar modelos (con caché para velocidad)
@st.cache_resource
def cargar_red_neuronal():
    return tf.keras.models.load_model('modelo_personalidad.h5')

@st.cache_resource
def cargar_logistica():
    return joblib.load('modelo_logistica.pkl')

# 3. Formulario de entrada (Es el mismo para ambos modelos)
col1, col2 = st.columns(2)

with col1:
    time_spent_alone = st.slider("Tiempo que pasas solo (horas)", 0, 24, 5)
    social_event = st.number_input("Asistencia a eventos sociales (aprox)", min_value=0, value=10)
    friends_circle = st.number_input("Tamaño de círculo de amigos", min_value=0, value=5)
    going_outside = st.slider("Frecuencia de salir (escala)", 0, 100, 50)

with col2:
    stage_fear_opt = st.selectbox("¿Tienes miedo escénico?", ["No", "Sí"])
    stage_fear = 1 if stage_fear_opt == "Sí" else 0

    drained_opt = st.selectbox("¿Te sientes agotado tras socializar?", ["No", "Sí"])
    drained = 1 if drained_opt == "Sí" else 0
    
    post_frequency = st.slider("Frecuencia de posteo en redes", 0.0, 50.0, 1.0)

# 4. Lógica de Predicción
if st.button("Analizar Personalidad"):
    
    # Preparar datos (el formato debe ser numpy array 2D)
    datos_entrada = np.array([[
        time_spent_alone, stage_fear, social_event, going_outside,
        drained, friends_circle, post_frequency
    ]]).astype('float32')

    st.write("---")
    
    try:
        if tipo_modelo == "Red Neuronal (Deep Learning)":
            modelo = cargar_red_neuronal()
            # La red neuronal devuelve una probabilidad (ej: 0.85)
            probabilidad = modelo.predict(datos_entrada)[0][0]
            es_extrovertido = probabilidad > 0.5
            confianza = probabilidad if es_extrovertido else (1 - probabilidad)
            
        else: # Regresión Logística
            modelo = cargar_logistica()
            # Logística también puede dar probabilidad con predict_proba
            # Devuelve array [[prob_0, prob_1]]
            probs = modelo.predict_proba(datos_entrada)
            probabilidad = probs[0][1] # Probabilidad de ser clase 1 (Extrovertido)
            es_extrovertido = probabilidad > 0.5
            confianza = probabilidad if es_extrovertido else (1 - probabilidad)

        # Mostrar Resultados
        st.subheader(f"Modelo usado: {tipo_modelo}")
        
        if es_extrovertido:
            st.success(f"Resultado: **EXTROVERTIDO**")
            st.progress(float(confianza))
            st.write(f"Nivel de confianza del modelo: {confianza*100:.2f}%")
            st.balloons()
        else:
            st.info(f"Resultado: **INTROVERTIDO**")
            st.progress(float(confianza))
            st.write(f"Nivel de confianza del modelo: {confianza*100:.2f}%")

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.warning("Asegúrate de que los archivos .h5 y .pkl estén subidos en GitHub.")