import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Personalidad", page_icon="🧠")

st.title("🧠 Detector de Personalidad con IA")
st.write("Ingresa tus datos para que la Red Neuronal prediga si eres Introvertido o Extrovertido.")

# 2. Cargar el modelo entrenado (asegúrate de que el archivo .h5 esté en la misma carpeta)
@st.cache_resource # Esto hace que no recargue el modelo en cada click
def load_model():
    return tf.keras.models.load_model('modelo_personalidad.h5')

try:
    modelo = load_model()
except:
    st.error("No se encontró el archivo 'modelo_personalidad.h5'. Asegúrate de subirlo.")
    st.stop()

# 3. Crear el formulario para el usuario (Interfaz Gráfica)
# Usamos columnas para que se vea ordenado
col1, col2 = st.columns(2)

with col1:
    time_spent_alone = st.slider("Tiempo que pasas solo (horas)", 0, 24, 5)
    social_event = st.number_input("Asistencia a eventos sociales (aprox)", min_value=0, value=10)
    friends_circle = st.number_input("Tamaño de círculo de amigos", min_value=0, value=5)
    going_outside = st.slider("Frecuencia de salir (escala)", 0, 100, 50)

with col2:
    # Inputs binarios (Sí/No) convertidos a 1/0
    stage_fear_opt = st.selectbox("¿Tienes miedo escénico?", ["No", "Sí"])
    stage_fear = 1 if stage_fear_opt == "Sí" else 0

    drained_opt = st.selectbox("¿Te sientes agotado tras socializar?", ["No", "Sí"])
    drained = 1 if drained_opt == "Sí" else 0
    
    post_frequency = st.slider("Frecuencia de posteo en redes", 0.0, 50.0, 1.0)

# 4. Botón de Predicción
if st.button("Analizar Personalidad"):
    # Crear el array con los datos en el MISMO ORDEN que usaste para entrenar (X)
    datos_entrada = np.array([[
        time_spent_alone,
        stage_fear,
        social_event,
        going_outside,
        drained,
        friends_circle,
        post_frequency
    ]]).astype('float32')
    
    # Predicción
    prediction_prob = modelo.predict(datos_entrada)
    prediction_class = (prediction_prob > 0.5).astype(int)[0][0]
    
    # Mostrar resultados
    st.write("---")
    if prediction_class == 1:
        st.success(f"Resultados: **EXTROVERTIDO** (Confianza: {prediction_prob[0][0]*100:.2f}%)")
        st.balloons()
    else:
        st.info(f"Resultados: **INTROVERTIDO** (Confianza: {(1-prediction_prob[0][0])*100:.2f}%)")