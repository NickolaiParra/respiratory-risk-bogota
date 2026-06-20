import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

os.makedirs("data/figures", exist_ok=True)

df = pd.read_csv("data/processed/rmcab_con_ica.csv")
features = ["PM10","PM2.5","CO","OZONO","NO","NO2","SO2","hora_sin","hora_cos","mes_sin","mes_cos"]
X = df[features].fillna(0)
y = df["categoria_ICA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf  = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")

categorias = ["Buena", "Aceptable", "D. Sensibles", "Dañina", "Muy dañina"]

resultados = {}
for nombre, modelo in [("Random Forest", rf), ("XGBoost", xgb)]:
    y_pred = modelo.predict(X_test)
    resultados[nombre] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "F1-weighted": f1_score(y_test, y_pred, average="weighted"),
        "Precision":   precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":      recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n{nombre}:")
    print(classification_report(y_test, y_pred, target_names=categorias, zero_division=0))

# Tabla comparativa
df_res = pd.DataFrame(resultados).T.round(4)
print("\nComparación de modelos:")
print(df_res)

# Fig 5: Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (nombre, modelo) in zip(axes, [("Random Forest", rf), ("XGBoost", xgb)]):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=categorias, yticklabels=categorias)
    ax.set_title(f"Matriz de confusión - {nombre}")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("data/figures/fig5_matrices_confusion.png", dpi=150)
plt.close()
print("\nFigura guardada.")