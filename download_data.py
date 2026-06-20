import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import os
from datetime import date, timedelta
import time

os.makedirs("data/raw", exist_ok=True)

BASE_URL = "http://rmcab.ambientebogota.gov.co/Report/HourlyReports?id=1&UserDateString={}"

def scrape_day(fecha):
    url = BASE_URL.format(fecha)
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    
    registros = []
    
    # Cada tabla tiene su propia sección; buscar tablas con ID talbe_*
    tablas = soup.find_all("table", id=re.compile(r"talbe_"))
    
    for tabla in tablas:
        estacion = tabla.get("id", "").replace("talbe_", "")
        checkboxes = tabla.find_all("input", onclick=True)
        
        for cb in checkboxes:
            onclick = cb.get("onclick", "")
            match = re.search(r"\[\{.*?\}\]", onclick)
            if not match:
                continue
            json_str = match.group(0).replace("&quot;", '"')
            try:
                datos = json.loads(json_str)
            except:
                continue
            
            for entry in datos:
                keys = [k for k in entry.keys() if k != "DATE_TIME" and not k.startswith("STATUS")]
                if not keys:
                    continue
                contaminante = keys[0]
                registros.append({
                    "fecha_hora": entry["DATE_TIME"],
                    "estacion": estacion,
                    "contaminante": contaminante,
                    "valor": entry[contaminante],
                })
    
    return pd.DataFrame(registros) if registros else None

# --- Loop 2 años ---
start = date(2024, 1, 1)
end = date(2025, 12, 31)
delta = timedelta(days=1)

all_dfs = []
current = start

while current <= end:
    fecha_str = current.strftime("%Y-%m-%d")
    print(f"Descargando {fecha_str}...")
    df = scrape_day(fecha_str)
    if df is not None:
        all_dfs.append(df)
    time.sleep(1)  # respetar el servidor
    current += delta

final = pd.concat(all_dfs, ignore_index=True)
final.to_csv("data/raw/rmcab_2022_2023.csv", index=False)
print(f"Listo. Total registros: {len(final)}")