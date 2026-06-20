import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data/figures", exist_ok=True)

df = pd.read_csv("data/raw/rmcab_2022_2023.csv")

# Solo contaminantes relevantes para riesgo respiratorio
contaminantes = ["PM10", "PM2.5", "CO", "OZONO", "NO", "NO2", "SO2"]
df_cont = df[df["contaminante"].isin(contaminantes)]

# Fig 1: Registros por estacion
fig, ax = plt.subplots(figsize=(10, 5))
df.groupby("estacion").size().sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Número de registros")
ax.set_title("Distribución de registros por estación de monitoreo")
plt.tight_layout()
plt.savefig("data/figures/fig1_registros_estacion.png", dpi=150)
plt.close()

# Fig 2: Valores faltantes por contaminante
pivot = df_cont.pivot_table(index="fecha_hora", columns="contaminante", values="valor", aggfunc="mean")
missing = pivot.isnull().mean() * 100
fig, ax = plt.subplots(figsize=(8, 4))
missing.sort_values(ascending=True).plot(kind="barh", ax=ax, color="salmon")
ax.set_xlabel("% valores faltantes")
ax.set_title("Porcentaje de valores faltantes por contaminante")
plt.tight_layout()
plt.savefig("data/figures/fig2_missing.png", dpi=150)
plt.close()

# Fig 3: Serie de tiempo PM2.5 estacion centro_de_alto_rendimiento
df_pm25 = df[(df["contaminante"] == "PM2.5") & (df["estacion"] == "centro_de_alto_rendimiento")].copy()
df_pm25["fecha_hora"] = pd.to_datetime(df_pm25["fecha_hora"])
df_pm25 = df_pm25.set_index("fecha_hora").resample("D")["valor"].mean()
fig, ax = plt.subplots(figsize=(12, 4))
df_pm25.plot(ax=ax, color="darkgreen", linewidth=0.8)
ax.set_ylabel("PM₂.₅ (µg/m³)")
ax.set_title("Serie de tiempo diaria de PM₂.₅ – Estación Centro de Alto Rendimiento")
plt.tight_layout()
plt.savefig("data/figures/fig3_serie_pm25.png", dpi=150)
plt.close()

print("Figuras generadas en data/figures/")