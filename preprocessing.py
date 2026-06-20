import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/figures", exist_ok=True)

df = pd.read_csv("data/raw/rmcab_2022_2023.csv")

# Solo contaminantes relevantes
contaminantes = ["PM10", "PM2.5", "CO", "OZONO", "NO", "NO2", "SO2"]
df = df[df["contaminante"].isin(contaminantes)]

# 1. Limpieza de outliers: eliminar negativos
registros_antes = len(df)
df = df[df["valor"] >= 0].copy()

# 2. Limpieza por IQR por contaminante
dfs_limpios = []
for cont, group in df.groupby("contaminante"):
    Q1 = group["valor"].quantile(0.25)
    Q3 = group["valor"].quantile(0.75)
    IQR = Q3 - Q1
    filtrado = group[(group["valor"] >= Q1 - 3 * IQR) & (group["valor"] <= Q3 + 3 * IQR)].copy()
    filtrado["contaminante"] = cont
    dfs_limpios.append(filtrado)

df = pd.concat(dfs_limpios, ignore_index=True)
print(f"Registros antes de limpieza: {registros_antes}")
print(f"Registros tras limpieza: {len(df)}")

# 3. Pivot a formato ancho
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
pivot = df.pivot_table(index=["fecha_hora", "estacion"], columns="contaminante", values="valor", aggfunc="mean")
pivot = pivot.reset_index()
pivot.columns.name = None
print(f"Shape pivot: {pivot.shape}")

# 4. Codificación seno-coseno
pivot["hora"] = pivot["fecha_hora"].dt.hour
pivot["mes"] = pivot["fecha_hora"].dt.month
pivot["hora_sin"] = np.sin(2 * np.pi * pivot["hora"] / 24)
pivot["hora_cos"] = np.cos(2 * np.pi * pivot["hora"] / 24)
pivot["mes_sin"] = np.sin(2 * np.pi * pivot["mes"] / 12)
pivot["mes_cos"] = np.cos(2 * np.pi * pivot["mes"] / 12)

# 5. Normalización Z-score
scaler = StandardScaler()
pivot[contaminantes] = scaler.fit_transform(pivot[contaminantes].fillna(0))

# 6. Figura: primeras 10 filas del dataset procesado
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")
cols_mostrar = ["fecha_hora", "estacion"] + contaminantes + ["hora_sin", "hora_cos", "mes_sin", "mes_cos"]
muestra = pivot[cols_mostrar].head(10).copy()
muestra["fecha_hora"] = muestra["fecha_hora"].astype(str).str[:16]
muestra["estacion"] = muestra["estacion"].str.replace("_", " ")
muestra[contaminantes + ["hora_sin", "hora_cos", "mes_sin", "mes_cos"]] = \
    muestra[contaminantes + ["hora_sin", "hora_cos", "mes_sin", "mes_cos"]].round(3)

tabla = ax.table(
    cellText=muestra.values,
    colLabels=muestra.columns,
    cellLoc="center",
    loc="center"
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(6.5)
tabla.auto_set_column_width(col=list(range(len(muestra.columns))))
plt.title("Muestra representativa del dataset procesado (primeras 10 filas)", fontsize=9, pad=10)
plt.tight_layout()
plt.savefig("data/figures/fig3_dataset_procesado.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figura guardada en data/figures/fig3_dataset_procesado.png")

pivot.to_csv("data/processed/rmcab_procesado.csv", index=False)
print("Preprocesamiento completo.")