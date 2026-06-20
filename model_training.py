import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/rmcab_procesado.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
df = df.sort_values(["estacion", "fecha_hora"])

# --- 1. Desescalar contaminantes para calcular ICA en unidades reales ---
# Necesitamos los valores originales, cargar desde raw
df_raw = pd.read_csv("data/raw/rmcab_2022_2023.csv")
df_raw["fecha_hora"] = pd.to_datetime(df_raw["fecha_hora"])
contaminantes = ["PM10", "PM2.5", "CO", "OZONO", "NO", "NO2", "SO2"]
df_raw = df_raw[df_raw["contaminante"].isin(contaminantes)]
df_raw = df_raw[df_raw["valor"] >= 0]

pivot_raw = df_raw.pivot_table(index=["fecha_hora", "estacion"], columns="contaminante", values="valor", aggfunc="mean").reset_index()
pivot_raw.columns.name = None
pivot_raw = pivot_raw.sort_values(["estacion", "fecha_hora"])

# --- 2. Promedios móviles por ventana temporal ---
def rolling_by_station(df, col, window):
    return df.groupby("estacion")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())

pivot_raw["PM10_24h"]  = rolling_by_station(pivot_raw, "PM10",  24)
pivot_raw["PM25_24h"]  = rolling_by_station(pivot_raw, "PM2.5", 24)
pivot_raw["CO_8h"]     = rolling_by_station(pivot_raw, "CO",    8)
pivot_raw["OZONO_8h"]  = rolling_by_station(pivot_raw, "OZONO", 8)
pivot_raw["SO2_1h"]    = pivot_raw["SO2"]
pivot_raw["NO2_1h"]    = pivot_raw["NO2"]

# --- 3. Calcular sub-ICA por contaminante (interpolación lineal IDEAM) ---
breakpoints = {
    "PM10_24h":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,604,301,500)],
    "PM25_24h":  [(0,12,0,50),(13,37,51,100),(38,55,101,150),(56,150,151,200),(151,250,201,300),(251,500,301,500)],
    "CO_8h":     [(0,5094,0,50),(5095,10819,51,100),(10820,14254,101,150),(14255,17688,151,200),(17689,34862,201,300),(34863,57703,301,500)],
    "SO2_1h":    [(0,93,0,50),(94,197,51,100),(198,486,101,150),(487,797,151,200),(798,1583,201,300),(1584,2629,301,500)],
    "NO2_1h":    [(0,100,0,50),(101,189,51,100),(190,677,101,150),(678,1221,151,200),(1222,2349,201,300),(2350,3853,301,500)],
    "OZONO_8h":  [(0,106,0,50),(107,138,51,100),(139,167,101,150),(168,207,151,200),(208,393,201,300),(394,1185,301,500)],
}

def calc_subica(valor, bp_list):
    if pd.isna(valor):
        return np.nan
    for (c_lo, c_hi, i_lo, i_hi) in bp_list:
        if c_lo <= valor <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (valor - c_lo) + i_lo
    return 500 if valor > bp_list[-1][1] else 0

for col, bp in breakpoints.items():
    pivot_raw[f"ICA_{col}"] = pivot_raw[col].apply(lambda x: calc_subica(x, bp))

ica_cols = [f"ICA_{c}" for c in breakpoints.keys()]
pivot_raw["ICA"] = pivot_raw[ica_cols].max(axis=1)

# --- 4. Clasificar en categorías ---
def clasificar_ica(ica):
    if ica <= 50:   return 0  # Buena
    elif ica <= 100: return 1  # Aceptable
    elif ica <= 150: return 2  # Dañina grupos sensibles
    elif ica <= 200: return 3  # Dañina
    else:            return 4  # Muy dañina / Peligrosa

pivot_raw["categoria_ICA"] = pivot_raw["ICA"].apply(clasificar_ica)
print("Distribución de categorías ICA:")
print(pivot_raw["categoria_ICA"].value_counts().sort_index())

# --- 5. Unir con dataset procesado ---
df_model = df.merge(pivot_raw[["fecha_hora","estacion","ICA","categoria_ICA"]], on=["fecha_hora","estacion"], how="inner")
print(f"Shape final para modelado: {df_model.shape}")

features = ["PM10","PM2.5","CO","OZONO","NO","NO2","SO2","hora_sin","hora_cos","mes_sin","mes_cos"]
X = df_model[features].fillna(0)
y = df_model["categoria_ICA"]

# --- 6. Random Forest con Grid Search + CV k=5 ---
print("\nEntrenando Random Forest...")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_params, cv=5, scoring="f1_weighted", verbose=1)
rf.fit(X, y)
print(f"Mejores params RF: {rf.best_params_}")
print(f"F1 CV RF: {rf.best_score_:.4f}")

# --- 7. XGBoost con Grid Search + CV k=5 ---
print("\nEntrenando XGBoost...")
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1]
}
xgb = GridSearchCV(XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"), xgb_params, cv=5, scoring="f1_weighted", verbose=1)
xgb.fit(X, y)
print(f"Mejores params XGBoost: {xgb.best_params_}")
print(f"F1 CV XGBoost: {xgb.best_score_:.4f}")

# --- 8. Guardar modelos ---
joblib.dump(rf.best_estimator_, "models/random_forest.pkl")
joblib.dump(xgb.best_estimator_, "models/xgboost.pkl")
df_model.to_csv("data/processed/rmcab_con_ica.csv", index=False)
print("\nModelos guardados en models/")