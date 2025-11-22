import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# =================================================================
# PASO 1: CARGA Y PREPROCESAMIENTO DE DATOS
# =================================================================

DATA_PATH = Path(__file__).parent / 'DataSets' / 'personality_datasert.csv'

df = pd.read_csv(DATA_PATH)

print(df.describe())

print("\nColumnas:", df.columns.tolist())

print("\nInfo:")
print(df.info())

print("\nPrimeras filas:")
print(df.head())

df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Personality_label'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

print("\nValores faltantes por columna:")
print(df.isna().sum())

X = df[[
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency',
    'Personality_label'
]]

y = df['Time_spent_Alone']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# =================================================================
# PASO 2: ENTRENAMIENTO DEL MODELO
# =================================================================

model = LinearRegression()
model.fit(x_train, y_train)

print("\nCoeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

y_pred = model.predict(x_test)

# =================================================================
# PASO 3: EVALUACIÓN Y VISUALIZACIÓN
# =================================================================

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

results_df = x_test.copy().reset_index(drop=True)
results_df['y_true'] = y_test.reset_index(drop=True)
results_df['y_pred'] = y_pred


plt.figure(figsize=(8,6))
plt.scatter(results_df['y_true'], results_df['y_pred'], alpha=0.6)
plt.plot([results_df['y_true'].min(), results_df['y_true'].max()],
         [results_df['y_true'].min(), results_df['y_true'].max()], 'r--')
plt.xlabel('Time_spent_Alone (real)')
plt.ylabel('Time_spent_Alone (predicho)')
plt.title('Reales vs Predichos - Time_spent_Alone')
plt.tight_layout()

plt.show()

residuals = results_df['y_true'] - results_df['y_pred']
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residual (y_true - y_pred)')
plt.ylabel('Frecuencia')
plt.title('Distribución de residuos - Time_spent_Alone')
plt.tight_layout()
plt.show()

print('\nFin del script.')
