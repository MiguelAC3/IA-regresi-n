import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# PASO 1: CARGA Y PREPROCESAMIENTO DE DATOS
# =================================================================
df = pd.read_csv("DataSets/personality_datasert.csv")

df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Personality_label'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

#Si hay valores que no pueden ser procesador, se cambian por el promedio de la columna.
for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Definición de X (Características) y Y (Variable Objetivo)
X = df[[
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]].astype('float32')

y = df['Personality_label'].astype('float32')

# División de datos de entrenamiento y prueba (70% Train, 30% Test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# =================================================================
# PASO 2 & 3: DEFINICIÓN Y COMPILACIÓN DEL MODELO (Keras)
# =================================================================

print("\nConstruyendo Red Neuronal...")

# Arquitectura: Multi-Layer Perceptron (MLP)
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =================================================================
# PASO 4: ENTRENAMIENTO
# =================================================================

print("Iniciando entrenamiento...")
historial = modelo.fit(x_train, y_train, epochs=100, verbose=0, validation_split=0.2)
print("Entrenamiento finalizado.")

# =================================================================
# PASO 5: EVALUACIÓN Y PREDICCIÓN
# =================================================================
plt.figure(figsize=(8, 5))
plt.plot(historial.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(historial.history['val_loss'], label='Pérdida (Validación)')
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = modelo.predict(x_test)
y_pred = (y_proba > 0.5).astype(int)

# Evaluación de la precisión
accuracy = accuracy_score(y_test, y_pred)
matrixC = confusion_matrix(y_test, y_pred)

print(f"\nPrecisión de la Red Neuronal: {accuracy:.4f}")
print("\nMatriz de Confusión:\n", matrixC)
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=['Introvert', 'Extrovert']))

# Visualización de la Matriz de Confusión
cm_display = metrics.ConfusionMatrixDisplay(matrixC, display_labels=['Introvert', 'Extrovert'])
cm_display.plot()
plt.show()


modelo.save('modelo_personalidad.h5')
print("Modelo guardado exitosamente como 'modelo_personalidad.h5'")

print('\nFin del script de Red Neuronal.')