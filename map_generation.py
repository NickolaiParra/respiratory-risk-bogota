import pandas as pd
import numpy as np
import joblib
import json
import os

os.makedirs("docs", exist_ok=True)

coordenadas = {
    "centro_de_alto_rendimiento": (4.658467, -74.083967),
    "guaymaral":                  (4.783750, -74.044139),
    "las_ferias":                 (4.690700, -74.082483),
    "minambiente":                (4.625486, -74.066981),
    "suba":                       (4.761247, -74.093461),
    "usaquen":                    (4.710350, -74.030417),
    "ciudad_bolivar":             (4.577806, -74.166278),
    "colina":                     (4.737194, -74.069472),
    "jazmin":                     (4.609467, -74.113333),
    "movil_fontibon":             (4.668000, -74.148500),
    "usme":                       (4.472500, -74.111944),
    "san_cristobal":              (4.572553, -74.083814),
    "tunal":                      (4.576225, -74.130956),
    "bolivia":                    (4.735867, -74.125883),
    "carvajal_-_sevillana":       (4.595833, -74.148500),
    "fontibon":                   (4.678242, -74.143819),
    "kennedy":                    (4.625050, -74.161333),
    "movil_7ma":                  (4.643889, -74.058333),
    "puente_aranda":              (4.631767, -74.117483),
}

cat_nombres = {0:"Buena", 1:"Aceptable", 2:"Dañina - Grupos Sensibles", 3:"Dañina", 4:"Muy dañina"}
cat_colores = {0:"#2ecc71", 1:"#f1c40f", 2:"#e67e22", 3:"#e74c3c", 4:"#8e44ad"}

rf = joblib.load("models/random_forest.pkl")
df = pd.read_csv("data/processed/rmcab_con_ica.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

features = ["PM10","PM2.5","CO","OZONO","NO","NO2","SO2","hora_sin","hora_cos","mes_sin","mes_cos"]
cont_cols = ["PM10","PM2.5","CO","OZONO","NO","NO2","SO2"]

ultimo = df.sort_values("fecha_hora").groupby("estacion").last().reset_index()
X_pred = ultimo[features].fillna(0)
ultimo["categoria_pred"] = rf.predict(X_pred)

ica_prom = df.groupby("estacion")["ICA"].mean()
promedios = df.groupby("estacion")[cont_cols].mean()

estaciones = []
for _, row in ultimo.iterrows():
    estacion = row["estacion"]
    if estacion not in coordenadas:
        continue
    lat, lon = coordenadas[estacion]
    cat = int(row["categoria_pred"])
    prom = promedios.loc[estacion] if estacion in promedios.index else {}

    estaciones.append({
        "id": estacion,
        "nombre": estacion.replace("_", " ").title(),
        "lat": lat,
        "lon": lon,
        "categoria": cat,
        "cat_nombre": cat_nombres[cat],
        "color": cat_colores[cat],
        "ica_promedio": round(float(ica_prom.get(estacion, 0)), 1),
        "PM25": round(float(prom.get("PM2.5", 0)), 3),
        "PM10": round(float(prom.get("PM10", 0)), 3),
        "CO":   round(float(prom.get("CO", 0)), 3),
        "NO2":  round(float(prom.get("NO2", 0)), 3),
        "SO2":  round(float(prom.get("SO2", 0)), 3),
        "OZONO":round(float(prom.get("OZONO", 0)), 3),
    })

with open("docs/data.js", "w", encoding="utf-8") as f:
    f.write(f"const estaciones = {json.dumps(estaciones, ensure_ascii=False, indent=2)};")

print(f"Exportadas {len(estaciones)} estaciones a docs/data.js")