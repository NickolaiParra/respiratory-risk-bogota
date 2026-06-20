import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

os.makedirs("data/figures", exist_ok=True)

df = pd.read_csv("data/processed/rmcab_con_ica.csv")
features = ["PM10","PM2.5","CO","OZONO","NO","NO2","SO2","hora_sin","hora_cos","mes_sin","mes_cos"]
X = df[features].fillna(0)
X_sample = X.sample(2000, random_state=42)

xgb = joblib.load("models/xgboost.pkl")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)  # shape: (2000, 11, 5)

# Fig 6: Importancia global (promedio absoluto por feature sobre todas las clases)
mean_shap = np.abs(shap_values).mean(axis=(0, 2))  # shape: (11,)
sorted_idx = np.argsort(mean_shap)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([features[i] for i in sorted_idx], mean_shap[sorted_idx], color="steelblue")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Importancia global de características (SHAP - XGBoost)")
plt.tight_layout()
plt.savefig("data/figures/fig6_shap_global.png", dpi=150)
plt.close()

# Fig 7: Beeswarm para clase Buena (índice 0)
shap_clase0 = shap_values[:, :, 0]  # shape: (2000, 11)
plt.figure()
shap.summary_plot(shap_clase0, X_sample, feature_names=features, show=False)
plt.title("SHAP - Categoría Buena (ICA 0-50)")
plt.tight_layout()
plt.savefig("data/figures/fig7_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

print("Figuras SHAP guardadas.")