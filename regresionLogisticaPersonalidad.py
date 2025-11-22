import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# PASO 1: CARGA Y PREPROCESAMIENTO DE DATOS
# =================================================================

df = pd.read_csv("DataSets/personality_datasert.csv")
print(df.describe())

print("\nColumnas:", df.columns.tolist())

print("\nInfo:")
print(df.info())

# print("\nPrimeras filas:")
# print(df.head())

df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Personality_label'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

print("\nValores faltantes por columna:")
print(df.isna().sum())

X = df[[
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]]

y = df['Personality_label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# =================================================================
# PASO 2: ENTRENAMIENTO DEL MODELO
# =================================================================

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

print("\nCoeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

results_df = x_test.copy()
results_df = results_df.reset_index(drop=True)
results_df['y_true'] = y_test.reset_index(drop=True)
results_df['y_pred'] = y_pred
results_df['prob_extrovert'] = y_proba


# print("\nMostrando primeras 50 filas de la tabla de resultados:")
# print(results_df.head(50).to_string(index=False))

# =================================================================
# PASO 3: EVALUACIÓN Y VISUALIZACIÓN
# =================================================================

y_pred = model.predict(x_test)
matrixC = metrics.confusion_matrix(y_test, y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
error = 1 - accuracy
log_loss = metrics.log_loss(y_test, y_proba)

print("\n" + "="*50)
print("MÉTRICAS DE EVALUACIÓN - REGRESIÓN LOGÍSTICA")
print("="*50)
print(f"Error:          {error:.4f}")
print(f"Exactitud:      {accuracy:.4f}")
print(f"Precisión:      {precision:.4f}")
print(f"Exhaustividad:  {recall:.4f}")
print(f"F1-Score:       {f1:.4f}")
print(f"Log Loss:       {log_loss:.4f}")
print("="*50)

print("\nMatriz de Confusión:")
print(matrixC)

cm_display = metrics.ConfusionMatrixDisplay(matrixC, display_labels=['Introvert', 'Extrovert'])
cm_display.plot()
plt.show()

print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=['Introvert', 'Extrovert']))

cols_corr = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency',
    'Personality_label'
]

corr_df = df[cols_corr].copy().corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlación entre variables (personality_datasert)')
plt.tight_layout()
plt.show()


